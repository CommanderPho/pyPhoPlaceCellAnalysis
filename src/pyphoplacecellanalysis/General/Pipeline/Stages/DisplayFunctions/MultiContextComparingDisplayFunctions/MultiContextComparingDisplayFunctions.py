from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import pandas as pd
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for _display_2d_placefield_result_plot_raw

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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager


class MultiContextComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ MultiContextComparingDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='grid_bin_bounds_validation', tags=['grid_bin_bounds','validation','pandas','1D','position', 'LONG_SHORT_SPECIFIC'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-14 18:17', related_items=[], is_global=True)
    def _display_grid_bin_bounds_validation(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, is_x_axis: bool = True, ax=None, **kwargs):
        """ Renders a single figure that shows the 1D linearized position from several different sources to ensure sufficient overlap. Useful for validating that the grid_bin_bounds are chosen reasonably.

        """
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import NotableTrackPositions, perform_add_1D_track_bounds_lines

        assert owning_pipeline_reference is not None
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds = [owning_pipeline_reference.computation_results[a_name].computation_config['pf_params'].grid_bin_bounds for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # print(long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds)
        
        long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        
        long_pos_df, short_pos_df, global_pos_df, all_pos_df = [a_sess.position.to_dataframe() for a_sess in (long_session, short_session, global_session, owning_pipeline_reference.sess)] ## note I add `, owning_pipeline_reference.sess` at the end
    
        combined_pos_df = deepcopy(all_pos_df)
        
        if include_includelist is None:
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names
            include_includelist = ['long', 'short', 'global']
            
        if is_x_axis:
            ## plot x-positions
            pos_var_names = ['x', 'lin_pos'] # , 'long_lin_pos', 'short_pos_df', 'global_lin_pos'
            
            if 'long' in include_includelist:
                combined_pos_df.loc[long_pos_df.index, 'long_lin_pos'] = long_pos_df.lin_pos
                pos_var_names.append('long_lin_pos')
            if 'short' in include_includelist:
                combined_pos_df.loc[short_pos_df.index, 'short_pos_df'] = short_pos_df.lin_pos
                pos_var_names.append('short_pos_df')
            if 'global' in include_includelist:
                combined_pos_df.loc[global_pos_df.index, 'global_lin_pos'] = global_pos_df.lin_pos
                pos_var_names.append('global_lin_pos')

            combined_pos_df_plot_kwargs = dict(x='t', y=pos_var_names, title='grid_bin_bounds validation across epochs - positions along x-axis')
        else:
            ## plot y-positions
            pos_var_names = ['y']
            # combined_pos_df_plot_kwargs = dict(x='y', y='t', title='grid_bin_bounds validation across epochs - positions along y-axis')
            combined_pos_df_plot_kwargs = dict(x='t', y=pos_var_names, title='grid_bin_bounds validation across epochs - positions along y-axis')

        
        title = f'grid_bin_bounds validation across epochs'
        if is_x_axis:
            title = f'{title} - X-axis'
        else:
            title = f'{title} - Y-axis'
            
        const_line_text_label_y_offset: float = 0.05
        const_line_text_label_x_offset: float = 0.1
            
        did_create_new_figure: bool = False
        # Plot all 1D position variables:
        if ax is None:
            fig, ax = plt.subplots()
            did_create_new_figure = True
        else:
            fig = ax.figure
            did_create_new_figure = False
            
        combined_pos_df.plot(ax=ax, **combined_pos_df_plot_kwargs)
        ax.set_title(title)
        if did_create_new_figure:
            fig.canvas.manager.set_window_title(title)
        
        # Accessing the lines after plotting
        position_lines_2D = ax.get_lines() # plt.Lines2D

        ax.legend(loc='upper left') # Move legend inside the plot, in the top-left corner
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside the plot
        
        final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='_display_grid_bin_bounds_validation')
        
        ## Add grid_bin_bounds, track limits, and midpoint lines:
        curr_config = owning_pipeline_reference.active_configs['maze_any']

        grid_bin_bounds = curr_config.computation_config.pf_params.grid_bin_bounds # ((37.0773897438341, 250.69004399129707), (137.925447118083, 145.16448776601297))
        # curr_config.computation_config.pf_params.grid_bin # (3.793023081021702, 1.607897707662558)
        # loaded_track_limits = curr_config.active_session_config.loaded_track_limits

        # curr_config.active_session_config.y_midpoint
        
        (long_notable_x_platform_positions, short_notable_x_platform_positions), (long_notable_y_platform_positions, short_notable_y_platform_positions) = NotableTrackPositions.init_notable_track_points_from_session_config(owning_pipeline_reference.sess.config)
        
        if is_x_axis:
            ## plot x-positions
            perform_add_1D_track_bounds_lines_kwargs = dict(long_notable_x_platform_positions=tuple(long_notable_x_platform_positions), short_notable_x_platform_positions=tuple(short_notable_x_platform_positions), is_vertical=False)
        else:
            ## plot y-positions
            perform_add_1D_track_bounds_lines_kwargs = dict(long_notable_x_platform_positions=tuple(long_notable_y_platform_positions), short_notable_x_platform_positions=tuple(short_notable_y_platform_positions), is_vertical=False)
            
        long_track_line_collection, short_track_line_collection = perform_add_1D_track_bounds_lines(**perform_add_1D_track_bounds_lines_kwargs, ax=ax)


        # Plot REAL `grid_bin_bounds` ________________________________________________________________________________________ #
        ((grid_bin_bounds_x0, grid_bin_bounds_x1), (grid_bin_bounds_y0, grid_bin_bounds_y1)) = grid_bin_bounds
        if is_x_axis:
            ## horizontal lines:
            common_ax_bound_kwargs = dict(xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1])
        else:
            # common_ax_bound_kwargs = dict(ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1]) 
            common_ax_bound_kwargs = dict(xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1])  ## y-axis along x (like a 1D plot) Mode

        if is_x_axis:
            ## plot x-positions
            ## horizontal lines:
            ## midpoint line: dotted blue line centered in the bounds (along y)
            x_midpoint = curr_config.active_session_config.x_midpoint # 143.88489208633095
            midpoint_line_collection = ax.hlines(x_midpoint, label='x_midpoint', xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1], colors='#0000FFAA', linewidths=1.0, linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection midpoint_line_collection
            ax.text(ax.get_xbound()[1], (x_midpoint + const_line_text_label_y_offset), 'x_mid', ha='right', va='bottom', fontsize=8, color='#0000FFAA', zorder=-98) # Add right-aligned text label slightly above the hline
        
            ## 2 lines corresponding to the x0 and x1 of the grid_bin_bounds:
            grid_bin_bounds_line_collection = ax.hlines([grid_bin_bounds_x0, grid_bin_bounds_x1], label='grid_bin_bounds - after - dark blue', **common_ax_bound_kwargs, colors='#2e2e20', linewidths=2.0, linestyles='solid', zorder=-98) # grid_bin_bounds_line_collection
            # _grid_bin_bound_labels_x_pos = (ax.get_xbound()[1] - const_line_text_label_x_offset)
            _grid_bin_bound_labels_x_pos: float = ax.get_xbound()[0] + ((ax.get_xbound()[1] - ax.get_xbound()[0])/2.0) # center point
            print(f'_grid_bin_bound_labels_x_pos: {_grid_bin_bound_labels_x_pos}')
            ax.text(_grid_bin_bound_labels_x_pos, (grid_bin_bounds_x0 - const_line_text_label_y_offset), 'grid_bin_bounds[x0]', ha='center', va='bottom', fontsize=9, color='#2e2d20', zorder=-97) # Add right-aligned text label slightly above the hline
            ax.text(_grid_bin_bound_labels_x_pos, (grid_bin_bounds_x1 + const_line_text_label_y_offset), 'grid_bin_bounds[x1]', ha='center', va='top', fontsize=9, color='#2e2d20', zorder=-97) # this will be the top (highest y-pos) line.
            
        else:
            ## plot y-positions
            midpoint_line_collection = None
            # grid_bin_bounds_line_collection = None
            grid_bin_bounds_positions_list = [grid_bin_bounds_y0, grid_bin_bounds_y1]
            grid_bin_bounds_label_names_list = ['grid_bin_bounds[y0]', 'grid_bin_bounds[y1]']
            ## 2 lines corresponding to the x0 and x1 of the grid_bin_bounds:
            grid_bin_bounds_line_collection = ax.hlines(grid_bin_bounds_positions_list, label='grid_bin_bounds - after - dark blue', **common_ax_bound_kwargs, colors='#2e2e20', linewidths=2.0, linestyles='solid', zorder=-98) # grid_bin_bounds_line_collection
            # _grid_bin_bound_labels_x_pos = (ax.get_xbound()[1] - const_line_text_label_x_offset)
            _grid_bin_bound_labels_x_pos: float = ax.get_xbound()[0] + ((ax.get_xbound()[1] - ax.get_xbound()[0])/2.0) # center point
            # print(f'_grid_bin_bound_labels_y_pos: {_grid_bin_bound_labels_y_pos}')
            # ax.text(_grid_bin_bound_labels_y_pos, (grid_bin_bounds_y0 - const_line_text_label_y_offset), 'grid_bin_bounds[x0]', ha='center', va='bottom', fontsize=9, color='#2e2d20', zorder=-97) # Add right-aligned text label slightly above the hline
            # ax.text(_grid_bin_bound_labels_y_pos, (grid_bin_bounds_x1 + const_line_text_label_y_offset), 'grid_bin_bounds[x1]', ha='center', va='top', fontsize=9, color='#2e2d20', zorder=-97) # this will be the top (highest y-pos) line.	
            # Iterate through the hlines in the LineCollection and add labels
            assert len(grid_bin_bounds_label_names_list) == len(grid_bin_bounds_positions_list)
            for pos, a_txt_label in zip(grid_bin_bounds_positions_list, grid_bin_bounds_label_names_list):
                # ax.text(ax.get_xbound()[1], (pos + const_line_text_label_y_offset), a_txt_label, color='#2e2d20', fontsize=9, ha='center', va='center', zorder=-97)
                ax.text(_grid_bin_bound_labels_x_pos, (pos + const_line_text_label_y_offset), a_txt_label, color='#2e2d20', fontsize=9, ha='center', va='center', zorder=-97)
        
        # Show legend
        # ax.legend()

        if save_figure:
            saved_figure_paths = owning_pipeline_reference.output_figure(final_context, fig)
        else:
            saved_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='_display_grid_bin_bounds_validation', figures=(fig,), axes=(ax,),
                                                plot_data={'position_lines_2D': position_lines_2D, 'midpoint_line_collection': midpoint_line_collection, 'grid_bin_bounds_line_collection': grid_bin_bounds_line_collection, 'long_track_line_collection': long_track_line_collection, 'short_track_line_collection': short_track_line_collection},
                                                context=final_context, saved_figures=[])
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


    @function_attributes(short_name='trial_to_trial_reliability', tags=['trial-to-trial-reliability', 'display'], is_global=True, input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-30 13:27', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['time_binned_position_df']))
    def _display_trial_to_trial_reliability(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, drop_below_threshold = 0.0000001, save_figure=False, **kwargs):
        """ Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18)
        NOTE: Ignores `active_config` because context_nested_docks is for all contexts

        Input:
            owning_pipeline_reference: A reference to the pipeline upon which this display function is being called

        Usage:

        display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_filtered_session_ctx, enable_gui=False, debug_print=False) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
        master_dock_win = display_output['master_dock_win']
        app = display_output['app']
        out_items = display_output['out_items']

        curr_active_pipeline.reload_default_display_functions()
        _out = curr_active_pipeline.display(display_function='_display_trial_to_trial_reliability', active_session_configuration_context=None)


        """
        from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TrialByTrialActivityWindow import TrialByTrialActivityWindow
        assert owning_pipeline_reference is not None
        #
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        directional_trial_by_trial_activity_result: TrialByTrialActivityResult = owning_pipeline_reference.global_computation_results.computed_data.get('TrialByTrialActivity', None)
        if directional_trial_by_trial_activity_result is None:
            # if `KeyError: 'TrialByTrialActivity'` recompute
            print(f'TrialByTrialActivity is not computed, computing it...')
            owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['pf_computation', 'pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
            owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['trial_by_trial_metrics'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
            directional_trial_by_trial_activity_result = owning_pipeline_reference.global_computation_results.computed_data.get('TrialByTrialActivity', None) ## try again to get the result
            assert directional_trial_by_trial_activity_result is not None, f"directional_trial_by_trial_activity_result is None even after forcing recomputation!!"
            print(f'\t done.')

        ## unpack either way:
        # any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
        # active_pf_dt = directional_trial_by_trial_activity_result.active_pf_dt # PfND_TimeDependent, this version does not work!
        # any_decoder_neuron_IDs: NDArray = deepcopy(track_templates.any_decoder_neuron_IDs)
        
        # active_pf = deepcopy(directional_trial_by_trial_activity_result.active_pf_dt)
        any_decoder_neuron_IDs: NDArray = deepcopy(directional_trial_by_trial_activity_result.any_decoder_neuron_IDs)
            
        ## OUTPUTS: directional_trial_by_trial_activity_result, directional_active_lap_pf_results_dicts
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        # active_pf = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf2D_dt']) # PfND_TimeDependent
        # active_pf = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf2D'])
        active_pf = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf1D'])
        
        any_decoder_neuron_IDs: NDArray = deepcopy(directional_trial_by_trial_activity_result.any_decoder_neuron_IDs)
        override_active_neuron_IDs = deepcopy(any_decoder_neuron_IDs)
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pf_computation', 'pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

        ## Uses `plot_trial_to_trial_reliability_all_decoders_image_stack` to plot the reliability trial-by-trial indicators over time
        ## INPUTS: a_pf2D_dt, z_scored_tuning_map_matrix
        # directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = deepcopy(directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts)
        modified_directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = directional_trial_by_trial_activity_result.build_separated_nan_filled_decoded_epoch_z_scored_tuning_map_matrix()
        modified_directional_active_lap_pf_results_dicts = {k:v.sliced_by_neuron_id(included_neuron_ids=override_active_neuron_IDs) for k, v in modified_directional_active_lap_pf_results_dicts.items()}
        _a_trial_by_trial_window = TrialByTrialActivityWindow.plot_trial_to_trial_reliability_all_decoders_image_stack(directional_active_lap_pf_results_dicts=modified_directional_active_lap_pf_results_dicts,
                                                                                                                        active_one_step_decoder=deepcopy(active_pf), drop_below_threshold=drop_below_threshold,
                                                                                                                        override_active_neuron_IDs=override_active_neuron_IDs)


        ## split mode:
        # directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts # : Dict[str, Epoch]
        # _a_trial_by_trial_window = TrialByTrialActivityWindow.plot_trial_to_trial_reliability_all_decoders_image_stack(directional_active_lap_pf_results_dicts=directional_active_lap_pf_results_dicts, 
        #                                                                                                                 active_one_step_decoder=deepcopy(active_pf_dt),
        #                                                                                                                 drop_below_threshold=drop_below_threshold, is_overlaid_heatmaps_mode=True)

        # **overriding_dict_with(lhs_dict={'enable_gui': False, 'debug_print': False}, **kwargs)
    
        final_context = owning_pipeline_reference.build_display_context_for_session(display_fn_name='trial_to_trial_reliability')
        footer_annotation_text = final_context.get_description(separator='\t|\t')
        _a_trial_by_trial_window.ui.lblFooter.setText(footer_annotation_text, size='12pt', bold=True) # update the footer

        if save_figure:
            saved_figure_paths = owning_pipeline_reference.output_figure(final_context, _a_trial_by_trial_window.root_render_widget)
            _a_trial_by_trial_window.plot_data.saved_figure_paths = saved_figure_paths
        else:
            saved_figure_paths = []

        return _a_trial_by_trial_window
    


    @function_attributes(short_name='meas_v_decoded_occupancy', tags=['context-decoder-comparison', 'hairly-plot', 'decoded_position', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], requires_global_keys=["global_computation_results.computed_data['EpochComputations']"], uses=['MeasuredVsDecodedOccupancy', '_helper_add_interpolated_position_columns_to_decoded_result_df', '_display_grid_bin_bounds_validation', 'FigureCollector'], used_by=[], creation_date='2025-05-16 00:00', related_items=[], is_global=True)
    def _display_measured_vs_decoded_occupancy_distributions(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, size=[18, 8], dpi=100, constrained_layout=True, override_fig_man: Optional[FileOutputManager]=None, **kwargs):
            """ Plots the measured vs. decoded occupancy (position) for each of the four decoders for both Pre-delta and Post-delta.
            
            Usage:
            
                curr_active_pipeline.reload_default_display_functions()

                _out = curr_active_pipeline.display('_display_measured_vs_decoded_occupancy_distributions')

            """
            from neuropy.utils.result_context import IdentifyingContext
            
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from neuropy.utils.matplotlib_helpers import FormattedFigureText

            from neuropy.utils.result_context import IdentifyingContext
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationsComputationsContainer
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MeasuredVsDecodedOccupancy
            

            display_fn_name: str = 'meas_v_decoded_occupancy' # same as "short_name"

            export_dpi_multiplier: float = kwargs.pop('export_dpi_multiplier', 2.0)
            export_dpi: int = int(np.ceil(dpi * export_dpi_multiplier))
            
            # ==================================================================================================================================================================================================================================================================================== #
            # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
            # ==================================================================================================================================================================================================================================================================================== #

            ## Unpack from pipeline:
            valid_EpochComputations_result: EpochComputationsComputationsContainer = owning_pipeline_reference.global_computation_results.computed_data['EpochComputations'] # owning_pipeline_reference.global_computation_results.computed_data['EpochComputations']
            assert valid_EpochComputations_result is not None
            epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
            print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
            assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: nonPBE_results.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing

            ## INPUTS: a_new_fully_generic_result
            # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
            epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
            assert epochs_decoding_time_bin_size is not None

            ## Get the needed data:

            ## get from parameters:
            minimum_inclusion_fr_Hz: float = owning_pipeline_reference.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = owning_pipeline_reference.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

            directional_laps_results: DirectionalLapsResult = owning_pipeline_reference.global_computation_results.computed_data['DirectionalLaps']
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
            
            # common_constraint_dict = dict(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, masked_time_bin_fill_type='ignore')
            common_constraint_dict = dict(trained_compute_epochs='laps', time_bin_size=epochs_decoding_time_bin_size, masked_time_bin_fill_type='nan_filled') # , pfND_ndim=1

            ## OUTPUTS: a_decoded_marginal_posterior_df

            complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                display_context = active_context.adding_context('display_fn', display_fn_name=display_fn_name)
            else:
                # active_context = owning_pipeline_reference.sess.get_context()
                active_context = deepcopy(complete_session_context) # owning_pipeline_reference.sess.get_context()
                
                # Build the active context directly:
                display_context = owning_pipeline_reference.build_display_context_for_session(display_fn_name)

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')


            ## OUTPUTS: active_context, display_context
            
            # ==================================================================================================================================================================================================================================================================================== #
            # Add display function critical parameters                                                                                                                                                                                                                                             #
            # ==================================================================================================================================================================================================================================================================================== #
            # active_display_context = deepcopy(display_context) #.overwriting_context(extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier) ## include any that are just the slightest big different
            
            active_display_context = display_context.overwriting_context(t_bin=epochs_decoding_time_bin_size) ## include any that are just the slightest big different
            active_display_context = active_display_context.overwriting_context(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) ## include any that are just the slightest big different


            # perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))

            global_measured_position_df: pd.DataFrame = deepcopy(owning_pipeline_reference.sess.position.to_dataframe())
            a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)


            # ==================================================================================================================================================================================================================================================================================== #
            # Start Building Figure                                                                                                                                                                                                                                                                #
            # ==================================================================================================================================================================================================================================================================================== #

            graphics_output_dict = {}

            if override_fig_man is not None:
                print(f'override_fig_man is not None! Custom output path will be used!')
                test_display_output_path = override_fig_man.get_figure_save_file_path(active_display_context, make_folder_if_needed=False)
                print(f'\ttest_display_output_path: "{test_display_output_path}"')
    

            def _perform_write_to_file_callback(final_context, fig):
                """ captures: override_fig_man, export_dpi """
                if save_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig, override_fig_man=override_fig_man, dpi=export_dpi)
                else:
                    pass # do nothing, don't save
                
            # ==================================================================================================================================================================================================================================================================================== #
            # Titles/Formatting/Marginas and Saving                                                                                                                                                                                                                                                #
            # ==================================================================================================================================================================================================================================================================================== #
            def _subfn_apply_formatting_footer_and_etc(fig, figure_title, subtitle_string=None, graphics_output_dict=None):
                """ builds titles, footers, etc
                """
                if graphics_output_dict is None:
                    graphics_output_dict = {}
                # active_config = deepcopy(a_decoder.pf.config)

                # subtitle_string = active_config.str_for_display(is_2D=False) # , normal_to_extras_line_sep=","
                if subtitle_string is not None:
                    # subtitle_string = f"{subtitle_string} - only extreme context probs (P(Ctx) > {extreme_threshold}) are shown"
                    print(f'subtitle_string: {subtitle_string}')

                # print(f'subtitle_string: {subtitle_string}')

                ## BUild figure titles:
                # INPUTS: main_fig
                # fig.suptitle('')
                # out_axes_list[0].set_title('')
                
                # text_formatter = FormattedFigureText() # .init_from_margins(left_margin=0.01)
                # text_formatter.setup_margins(fig, left_margin=0.01) # , left_margin=0.1
                text_formatter = FormattedFigureText.init_from_margins(left_margin=0.01, right_margin=0.99) # , top_margin=0.9
                # text_formatter.setup_margins(fig, left_margin=0.01, top_margin=0.9)
                text_formatter.setup_margins(fig)
                title_string: str = f"{display_fn_name}: {figure_title}"
                # session_footer_string: str =  active_context.get_description(subset_includelist=['format_name', 'animal', 'exper_name', 'session_name'], separator=' | ') 
                session_footer_string: str =  active_context.get_description(separator=' | ') 

                # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
                # header_text_obj = flexitext(text_formatter.left_margin, 0.9, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                formatted_title_str: str = f'<size:20><weight:bold>{title_string}</></>'
                if subtitle_string is not None:
                    formatted_title_str = formatted_title_str + f'\n<size:9>{subtitle_string}</>'
                                    

                header_text_obj = flexitext(0.01, 0.85, formatted_title_str, va="bottom", xycoords="figure fraction") # , wrap=False
                footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
                
                complete_title_string: str = f"{title_string} - {session_footer_string}"
                # complete_title_string: str = f"{complete_title_string} - {subtitle_string}"
                
                window_title_string: str = complete_title_string
                
                fig.canvas.manager.set_window_title(window_title_string) # sets the window's title
                if ((_perform_write_to_file_callback is not None) and (active_display_context is not None)):
                    _perform_write_to_file_callback(active_display_context, fig)

                graphics_output_dict['label_objects'] = {'header': header_text_obj, 'footer': footer_text_obj, 'formatter': text_formatter}
                return graphics_output_dict



            with mpl.rc_context({'figure.dpi': str(dpi), 'savefig.transparent': True, 'ps.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': size, }): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name=display_fn_name, base_context=active_display_context) as collector:
                    # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_plot_hairy_overlayed_position

                    ## INPUTS: a_decoded_marginal_posterior_df

                    ## Laps context:
                    figure_title ='Laps'
                    known_named_decoding_epochs_type = 'laps'
                    a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type=known_named_decoding_epochs_type, data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
                    best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
                    an_active_display_context = active_display_context.overwriting_context(known_named_decoding_epochs_type=known_named_decoding_epochs_type, figure_title=figure_title)
                    fig, ax_dict = MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title=figure_title)
                    _graphics_output_dict = _subfn_apply_formatting_footer_and_etc(fig, figure_title=figure_title, subtitle_string=None)
                    collector.post_hoc_append(figs=[fig,], axes=ax_dict, contexts=[an_active_display_context])


                    ## Global context:
                    figure_title ='Global (all-time)'
                    known_named_decoding_epochs_type='global'
                    a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type=known_named_decoding_epochs_type, data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
                    best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
                    an_active_display_context = active_display_context.overwriting_context(known_named_decoding_epochs_type=known_named_decoding_epochs_type, figure_title=figure_title)
                    fig, ax_dict = MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title=figure_title)
                    _graphics_output_dict = _subfn_apply_formatting_footer_and_etc(fig, figure_title=figure_title, subtitle_string=None)
                    collector.post_hoc_append(figs=[fig,], axes=ax_dict, contexts=[an_active_display_context])

                    ## PBEs context:
                    figure_title ='PBEs'
                    known_named_decoding_epochs_type='pbe'
                    a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type=known_named_decoding_epochs_type, data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
                    best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
                    an_active_display_context = active_display_context.overwriting_context(known_named_decoding_epochs_type=known_named_decoding_epochs_type, figure_title=figure_title)
                    fig, ax_dict = MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title=figure_title)
                    _graphics_output_dict = _subfn_apply_formatting_footer_and_etc(fig, figure_title=figure_title, subtitle_string=None)
                    collector.post_hoc_append(figs=[fig,], axes=ax_dict, contexts=[an_active_display_context])


            ## END with mpl.rc_context({'figure.dpi': '...


            graphics_output_dict['collector'] = collector

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
    # include_includelist = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']

    if enable_gui:
        master_dock_win, app = DockAreaWrapper.build_default_dockAreaWindow(title='active_global_window', defer_show=False)
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
