from neuropy.utils.mixins.dynamic_conformance_updating_mixin import BaseDynamicInstanceConformingMixin
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaOwningMixin, DynamicDockDisplayAreaContentMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, TrainTestSplitResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from neuropy.utils.indexing_helpers import PandasHelpers



class SpecificDockWidgetManipulatingMixin(BaseDynamicInstanceConformingMixin)
    """ Factors out the specific plots added to Spike2DRaster

    from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.SpecificDockWidgetManipulatingMixin import SpecificDockWidgetManipulatingMixin


    """



    @function_attributes(short_name=None, tags=['UNFINISHED', 'plotting', 'computing'], input_requires=[], output_provides=[], uses=['AddNewDecodedPosteriors_MatplotlibPlotCommand', '_perform_plot_multi_decoder_meas_pred_position_track'], used_by=[], creation_date='2025-02-13 14:58', related_items=['_perform_plot_multi_decoder_meas_pred_position_track'])
    @classmethod
    def add_continuous_decoded_posterior(spike_raster_window, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
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

