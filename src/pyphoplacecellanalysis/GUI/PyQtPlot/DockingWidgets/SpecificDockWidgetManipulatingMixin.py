from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.General.type_aliases as types
from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from neuropy.utils.mixins.dynamic_conformance_updating_mixin import BaseDynamicInstanceConformingMixin
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaOwningMixin, DynamicDockDisplayAreaContentMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, TrainTestSplitResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from neuropy.utils.indexing_helpers import PandasHelpers
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult


class SpecificDockWidgetManipulatingMixin(BaseDynamicInstanceConformingMixin):
    """ Factors out the specific plots added to Spike2DRaster

    Aims to replace: `AddNewDecodedPosteriors_MatplotlibPlotCommand` and their heavy dependencies for plotting
    

    from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.SpecificDockWidgetManipulatingMixin import SpecificDockWidgetManipulatingMixin


    Variables:

    self.dock_manager_widget: DynamicDockDisplayAreaContentMixin
    self.ui.matplotlib_view_widgets: Dict
    self.ui.connections['tracks']: Dict
        

    """
    @classmethod
    def _perform_overlay_measured_position(cls, matplotlib_fig_axes, measured_position_df: pd.DataFrame, variable_name = 'x'):
        """ 
        
        if a_position_decoder.pf.filtered_pos_df is not None:
            measured_position_df = deepcopy(a_position_decoder.pf.filtered_pos_df)
        else:
            # fallback to session
            measured_position_df = curr_active_pipeline.sess.position.to_dataframe()
            
        """
        # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes
        actual_postion_plot_kwargs = {'color': '#ff000066', 'alpha': 0.35, 'marker': 'none', 'animated': False}
        _out_artists = {}
        for an_ax in matplotlib_fig_axes:                
            line_measured_position = an_ax.plot(measured_position_df['t'].to_numpy(), measured_position_df[variable_name].to_numpy(), label=f'measured {variable_name}', **actual_postion_plot_kwargs) # Opaque RED # , linestyle='dashed', linewidth=2, color='#ff0000ff'
            _out_artists[an_ax] = line_measured_position
        return _out_artists


    def add_docked_decoded_continuous_result_track(self, name: str):
        pass

    def add_docked_decoded_posterior_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None):
        """ adds a decoded 1D posterior 

        Aims to replace `AddNewDecodedPosteriors_MatplotlibPlotCommand._perform_add_new_decoded_posterior_row`
        
        
        Usage 0:
            time_bin_size = epochs_decoding_time_bin_size
            info_string: str = f" - t_bin_size: {time_bin_size}"
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_decoded_posterior_track(name='non-PBE_marginal_over_track_ID',
                                                                                                    time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID,
                                                                                                    xbin = deepcopy(a_position_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                    extended_dock_title_info=info_string)
                                                                                                    
        Usage 1:
            _out_tuple = active_2d_plot.add_docked_decoded_posterior_track(name=f'DirectionalDecodersDecoded[{a_decoder_name}]', a_dock_config=a_dock_config,
                                                                                            time_window_centers=a_1D_continuous_decoded_result.time_bin_container.centers, a_1D_posterior=a_1D_continuous_decoded_result.p_x_given_n,
                                                                                            xbin = deepcopy(a_1D_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                            extended_dock_title_info=info_string)
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        if a_variable_name is None:
            a_variable_name = name

        if a_dock_config is None:
            override_dock_group_name: str = None ## this feature doesn't work
            a_dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)
            a_dock_config.dock_group_names = [override_dock_group_name] # , 'non-PBE Continuous Decoding'


        n_xbins, n_t_bins = np.shape(a_1D_posterior)

        if xbin is None:
            xbin = np.arange(n_xbins)

        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = name
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        # print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes, dock_item = self.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(65, 200), display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        variable_name: str = a_variable_name
        
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        active_posterior = deepcopy(a_1D_posterior)
        
        posterior_heatmap_imshow_kwargs = dict(
            cmap = get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red'),
        )

        # most_likely_positions_mode: 'standard'|'corrected'
        ## Actual plotting portion:
        fig, an_ax = plot_1D_most_likely_position_comparsions(measured_position_df=None, time_window_centers=time_window_centers, xbin=deepcopy(xbin),
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs)

        ## Update the params
        widget.params.variable_name = variable_name
        widget.params.posterior_heatmap_imshow_kwargs = deepcopy(posterior_heatmap_imshow_kwargs)
        widget.params.enable_flat_line_drawing = False
        if extended_dock_title_info is not None:
            widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
            
        ## Update the plots_data - used for crosshairs tracing and other things
        if time_window_centers is not None:
            widget.plots_data.time_window_centers = deepcopy(time_window_centers)
        if xbin is not None:
            widget.plots_data.xbin = deepcopy(xbin)
        if active_most_likely_positions is not None:
            widget.plots_data.active_most_likely_positions = deepcopy(active_most_likely_positions)
        widget.plots_data.variable_name = variable_name
        if a_1D_posterior is not None:
            widget.plots_data.matrix = deepcopy(active_posterior)
        # if a_position_decoder is not None:
        #     widget.plots_data.a_decoder = deepcopy(a_position_decoder)

        ## measured positions
        widget.plots_data.measured_position_df = None
        widget.plots.measured_position_artists = None
        if measured_position_df is not None:
            widget.plots_data.measured_position_df = measured_position_df
            _out_artists = self._perform_overlay_measured_position(matplotlib_fig_axes=[an_ax], measured_position_df=measured_position_df)
            widget.plots.measured_position_artists = _out_artists


        widget.draw() # alternative to accessing through full path?
        self.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item
    

    def add_docked_marginal_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None):
        """ adds a marginal (such as Long v. Short, or Long_LR v. Long_RL v. Short_LR v. Short_RL) 
        
        time_bin_size = epochs_decoding_time_bin_size
        info_string: str = f" - t_bin_size: {time_bin_size}"
        identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_marginal_track(name='non-PBE_marginal_over_track_ID',
                                                                                                time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        if a_variable_name is None:
            a_variable_name = name

        if a_dock_config is None:
            override_dock_group_name: str = None ## this feature doesn't work
            a_dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)
            a_dock_config.dock_group_names = [override_dock_group_name] # , 'non-PBE Continuous Decoding'


        n_xbins, n_t_bins = np.shape(a_1D_posterior)

        if xbin is None:
            xbin = np.arange(n_xbins)

        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = name
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes, dock_item = self.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(25, 200), display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        variable_name: str = a_variable_name
        
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        active_posterior = deepcopy(a_1D_posterior)
        
        posterior_heatmap_imshow_kwargs = dict()	
        
        # most_likely_positions_mode: 'standard'|'corrected'
        ## Actual plotting portion:
        fig, an_ax = plot_1D_most_likely_position_comparsions(measured_position_df=None, time_window_centers=time_window_centers, xbin=deepcopy(xbin),
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs)

        ## Update the params
        widget.params.variable_name = variable_name
        widget.params.posterior_heatmap_imshow_kwargs = deepcopy(posterior_heatmap_imshow_kwargs)
        widget.params.enable_flat_line_drawing = False
        if extended_dock_title_info is not None:
            widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
            
        ## Update the plots_data
        if time_window_centers is not None:
            widget.plots_data.time_window_centers = deepcopy(time_window_centers)
        if xbin is not None:
            widget.plots_data.xbin = deepcopy(xbin)
        if active_most_likely_positions is not None:
            widget.plots_data.active_most_likely_positions = deepcopy(active_most_likely_positions)
        widget.plots_data.variable_name = variable_name
        if a_1D_posterior is not None:
            widget.plots_data.matrix = deepcopy(a_1D_posterior)


        widget.draw() # alternative to accessing through full path?
        self.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item



    @function_attributes(short_name=None, tags=['UNFINISHED', 'plotting', 'computing'], input_requires=[], output_provides=[], uses=['AddNewDecodedPosteriors_MatplotlibPlotCommand', '_perform_plot_multi_decoder_meas_pred_position_track'], used_by=[], creation_date='2025-02-13 14:58', related_items=['_perform_plot_multi_decoder_meas_pred_position_track'])
    @classmethod
    def add_continuous_decoded_posterior(cls, spike_raster_window, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
        """ computes the continuously decoded position posteriors (if needed) using the pipeline, then adds them as a new track to the SpikeRaster2D 
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_continuous_decoded_posterior

            (nested_dock_items, nested_dynamic_docked_widget_container_widgets), (a_continuously_decoded_dict, pseudo2D_decoder, all_directional_pf1D_Decoder_dict) = add_continuous_decoded_posterior(spike_raster_window=spike_raster_window, curr_active_pipeline=curr_active_pipeline, desired_time_bin_size=0.05, debug_print=True)

        """
        # ==================================================================================================================== #
        # COMPUTING                                                                                                            #
        # ==================================================================================================================== #
        
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.058}], #computation_kwargs_list=[{'time_bin_size': 0.025}], 
        #                                                   enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous'], computation_kwargs_list=[{'laps_decoding_time_bin_size': 0.058}, {'time_bin_size': 0.058, 'should_disable_cache':False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        ## get the result data:
        try:
            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            a_continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(desired_time_bin_size, None)
            info_string: str = f" - t_bin_size: {desired_time_bin_size}"

        except (KeyError, AttributeError) as e:
            # KeyError: 'DirectionalDecodersDecoded'
            print(f'add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(...) failed to add any tracks, perhaps because the pipeline is missing any computed "DirectionalDecodersDecoded" global results. Error: "{e}". Skipping.')
            a_continuously_decoded_dict = None
            pseudo2D_decoder = None        
            pass

        except Exception as e:
            raise


        # # output_dict = _cmd.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=_cmd._active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, **enable_rows_config_kwargs)
        # output_dict = AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, a_pseudo2D_decoder=pseudo2D_decoder, debug_print=debug_print, **kwargs)
        # for a_key, an_output_tuple in output_dict.items():
        #     identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple                
        #     # if a_key not in all_time_bin_sizes_output_dict:
        #     #     all_time_bin_sizes_output_dict[a_key] = [] ## init empty list
        #     # all_time_bin_sizes_output_dict[a_key].append(an_output_tuple)
            
        #     assert (identifier_name not in flat_all_time_bin_sizes_output_tuples_dict), f"identifier_name: {identifier_name} already in flat_all_time_bin_sizes_output_tuples_dict: {list(flat_all_time_bin_sizes_output_tuples_dict.keys())}"
        #     flat_all_time_bin_sizes_output_tuples_dict[identifier_name] = an_output_tuple



        # ==================================================================================================================== #
        # PLOTTING                                                                                                             #
        # ==================================================================================================================== #
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand

        display_output = {}
        AddNewDecodedPosteriors_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=None, active_context=None, display_output=display_output, action_identifier='actionPseudo2DDecodedEpochsDockedMatplotlibView')

        all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
        if debug_print:
            print(list(global_flat_action_dict.keys()))


        ## extract the components so the `background_static_scroll_window_plot` scroll bar is the right size:
        active_2d_plot = spike_raster_window.spike_raster_plt_2d
        
        active_2d_plot.params.enable_non_marginalized_raw_result = False
        active_2d_plot.params.enable_marginal_over_direction = False
        active_2d_plot.params.enable_marginal_over_track_ID = True


        menu_commands = [
            # 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView',
            # 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView',
            # 'DockedWidgets.TrackTemplatesDecodedEpochsDockedMatplotlibView',
            'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView',
            #  'DockedWidgets.ContinuousPseudo2DDecodedMarginalsDockedMatplotlibView',

        ]
        # menu_commands = ['actionPseudo2DDecodedEpochsDockedMatplotlibView', 'actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView'] # , 'AddTimeIntervals.SessionEpochs'
        for a_command in menu_commands:
            # all_global_menus_actionsDict[a_command].trigger()
            global_flat_action_dict[a_command].trigger()


        ## Dock all Grouped results from `'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView'`
        ## INPUTS: active_2d_plot
        grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
        nested_dock_items = {}
        nested_dynamic_docked_widget_container_widgets = {}
        for dock_group_name, flat_group_dockitems_list in grouped_dock_items_dict.items():
            dDisplayItem, nested_dynamic_docked_widget_container = active_2d_plot.ui.dynamic_docked_widget_container.build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
            nested_dock_items[dock_group_name] = dDisplayItem
            nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container

        ## OUTPUTS: nested_dock_items, nested_dynamic_docked_widget_container_widgets

        return (nested_dock_items, nested_dynamic_docked_widget_container_widgets), (a_continuously_decoded_dict, pseudo2D_decoder, all_directional_pf1D_Decoder_dict)

