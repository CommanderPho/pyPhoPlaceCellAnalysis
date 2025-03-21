from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
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
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, TrainTestSplitResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from neuropy.utils.indexing_helpers import PandasHelpers
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers


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


    @function_attributes(short_name=None, tags=['IMPORTANT', 'FINAL', 'track', 'posterior', '1D'], input_requires=[], output_provides=[], uses=[], used_by=['add_docked_decoded_posterior_track_from_result'], creation_date='2025-03-21 08:32', related_items=[])
    def add_docked_decoded_posterior_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None, should_defer_render:bool=False, **kwargs):
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

        xbin_labels = kwargs.pop('xbin_labels', None)
        if xbin_labels is not None:
            assert len(xbin_labels) == len(xbin)
        
        dockSize = kwargs.pop('dockSize', (65, 200)) 
        
        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = name
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        # print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes, dock_item = self.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=dockSize, display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        variable_name: str = a_variable_name
        
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        active_posterior = deepcopy(a_1D_posterior)
        
        posterior_heatmap_imshow_kwargs = kwargs.pop('posterior_heatmap_imshow_kwargs', dict(
            cmap = get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red'),
        ))

        # most_likely_positions_mode: 'standard'|'corrected'
        ## Actual plotting portion:
        fig, an_ax = plot_1D_most_likely_position_comparsions(measured_position_df=None, time_window_centers=time_window_centers, xbin=deepcopy(xbin),
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs)

        if xbin_labels is not None:
            ## add the labels:
            widget.plots.label_artists_dict = {}
            y_bin_labels = list(reversed(xbin_labels))
            widget.params.y_bin_labels = y_bin_labels
            widget.plots.label_artists_dict[an_ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(an_ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False)


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


        if not should_defer_render:
            widget.draw() # alternative to accessing through full path?
        # end if not should_defer_render
        self.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item
    
    def add_docked_decoded_posterior_slices_track(self, name: str, slices_time_window_centers: List[NDArray], slices_posteriors: List[NDArray], xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None, should_defer_render:bool=False, **kwargs):
        """ adds a decoded 1D posterior 

        Aims to replace `AddNewDecodedPosteriors_MatplotlibPlotCommand._perform_add_new_decoded_posterior_row`
        
        
        Usage 0:
            time_bin_size = epochs_decoding_time_bin_size
            info_string: str = f" - t_bin_size: {time_bin_size}"
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_decoded_posterior_slices_track(name='non-PBE_marginal_over_track_ID',
                                                                                                    slices_time_window_centers=[v.centers for v in long_results_obj.time_bin_containers], slices_posteriors=long_results_obj.p_x_given_n_list,
                                                                                                    xbin = deepcopy(a_position_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                    extended_dock_title_info=info_string)
                                                                                                    
        Usage 1:
            _out_tuple = active_2d_plot.add_docked_decoded_posterior_slices_track(name=f'DirectionalDecodersDecoded[{a_decoder_name}]', a_dock_config=a_dock_config,
                                                                                            slices_time_window_centers=a_1D_continuous_decoded_result.time_bin_container.centers, slices_posteriors=long_results_obj.p_x_given_n_list,
                                                                                            xbin = deepcopy(a_1D_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                            extended_dock_title_info=info_string)
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_slices_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        
        if a_variable_name is None:
            a_variable_name = name

        if a_dock_config is None:
            override_dock_group_name: str = None ## this feature doesn't work
            a_dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)
            a_dock_config.dock_group_names = [override_dock_group_name] # , 'non-PBE Continuous Decoding'

        # ((59, 2, 66), (59, 2, 102), (59, 2, 226), ...)
        # TODO: get flattened posterior instead of the first slice only
        a_1D_posterior = slices_posteriors[0]
        n_xbins, n_t_bins = np.shape(a_1D_posterior)

        if xbin is None:
            xbin = np.arange(n_xbins)

        xbin_labels = kwargs.pop('xbin_labels', None)
        if xbin_labels is not None:
            assert len(xbin_labels) == len(xbin)
            
        dockSize = kwargs.pop('dockSize', (65, 200)) 
        
        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = name
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        # print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes, dock_item = self.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=dockSize, display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        variable_name: str = a_variable_name
        
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        slices_active_most_likely_positions_1D = None
        
        posterior_heatmap_imshow_kwargs = kwargs.pop('posterior_heatmap_imshow_kwargs', dict(
            cmap = get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red'),
        ))

        # most_likely_positions_mode: 'standard'|'corrected'
        ## Actual plotting portion:
        fig, an_ax, out_img_list = plot_slices_1D_most_likely_position_comparsions(measured_position_df=None, slices_time_window_centers=slices_time_window_centers, xbin=deepcopy(xbin),
                                                        slices_posteriors=slices_posteriors,
                                                        slices_active_most_likely_positions_1D=slices_active_most_likely_positions_1D,
                                                        ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                        posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs)
        


        if xbin_labels is not None:
            ## add the labels:
            widget.plots.label_artists_dict = {}
            y_bin_labels = list(reversed(xbin_labels))
            widget.params.y_bin_labels = y_bin_labels
            widget.plots.label_artists_dict[an_ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(an_ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False)


        ## Update the params
        widget.params.variable_name = variable_name
        widget.params.posterior_heatmap_imshow_kwargs = deepcopy(posterior_heatmap_imshow_kwargs)
        widget.params.enable_flat_line_drawing = False
        if extended_dock_title_info is not None:
            widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
            
        ## Update the plots_data - used for crosshairs tracing and other things
        if slices_time_window_centers is not None:
            widget.plots_data.slices_time_window_centers = deepcopy(slices_time_window_centers)
        if xbin is not None:
            widget.plots_data.xbin = deepcopy(xbin)
        if slices_active_most_likely_positions_1D is not None:
            widget.plots_data.slices_active_most_likely_positions_1D = deepcopy(slices_active_most_likely_positions_1D)
        widget.plots_data.variable_name = variable_name
        if slices_posteriors is not None:
            widget.plots_data.slices_posteriors = deepcopy(slices_posteriors)
            
        if a_1D_posterior is not None:
            widget.plots_data.matrix = deepcopy(a_1D_posterior)
        # if a_position_decoder is not None:
        #     widget.plots_data.a_decoder = deepcopy(a_position_decoder)

        ## measured positions
        widget.plots_data.measured_position_df = None
        widget.plots.measured_position_artists = None
        if measured_position_df is not None:
            widget.plots_data.measured_position_df = measured_position_df
            _out_artists = self._perform_overlay_measured_position(matplotlib_fig_axes=[an_ax], measured_position_df=measured_position_df)
            widget.plots.measured_position_artists = _out_artists


        if not should_defer_render:
            widget.draw() # alternative to accessing through full path?
            
        # end if not should_defer_render
        self.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item
    

    @function_attributes(short_name=None, tags=['IMPORTANT', 'FINAL', 'track', 'posterior'], input_requires=[], output_provides=[], uses=['add_docked_decoded_posterior_track'], used_by=[], creation_date='2025-03-21 08:10', related_items=[])
    def add_docked_decoded_posterior_track_from_result(self, name: str, a_1D_decoded_result: Union[SingleEpochDecodedResult, DecodedFilterEpochsResult], xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, **kwargs):
            """ adds a decoded 1D posterior from a decoded result

            Aims to replace `AddNewDecodedPosteriors_MatplotlibPlotCommand._perform_add_new_decoded_posterior_row`
            
            
            Usage 0:
                time_bin_size = epochs_decoding_time_bin_size
                info_string: str = f" - t_bin_size: {time_bin_size}"
                identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_decoded_posterior_track_from_result(name='non-PBE_marginal_over_track_ID',
                                                                                                        a_1D_decoded_result=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID,
                                                                                                        xbin = deepcopy(a_position_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                        extended_dock_title_info=info_string)
                                                                                                        
            Usage 1:
                _out_tuple = active_2d_plot.add_docked_decoded_posterior_track_from_result(name=f'DirectionalDecodersDecoded[{a_decoder_name}]', a_dock_config=a_dock_config,
                                                                                                a_1D_decoded_result=a_1D_continuous_decoded_result.time_bin_container.centers, a_1D_posterior=a_1D_continuous_decoded_result.p_x_given_n,
                                                                                                xbin = deepcopy(a_1D_decoder.xbin), measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                extended_dock_title_info=info_string)
            """
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
            if isinstance(a_1D_decoded_result, SingleEpochDecodedResult):
                return self.add_docked_decoded_posterior_track(name=name, time_window_centers=a_1D_decoded_result.time_bin_container.centers, a_1D_posterior=a_1D_decoded_result.p_x_given_n, xbin=xbin, measured_position_df=measured_position_df, **kwargs)
            elif isinstance(a_1D_decoded_result, DecodedFilterEpochsResult):
                ## multiple
                if a_1D_decoded_result.num_filter_epochs == 1:
                    ## equivalent to a single result
                    a_1D_decoded_result = a_1D_decoded_result.get_result_for_epoch(0) ## will be a `SingleEpochDecodedResult`
                    return self.add_docked_decoded_posterior_track(name=name, time_window_centers=a_1D_decoded_result.time_bin_container.centers, a_1D_posterior=a_1D_decoded_result.p_x_given_n, xbin=xbin, measured_position_df=measured_position_df, **kwargs)
                else:
                    ## more than one
                    return self.add_docked_decoded_posterior_slices_track(name=name, slices_time_window_centers=[v.centers for v in a_1D_decoded_result.time_bin_containers], slices_posteriors=a_1D_decoded_result.p_x_given_n_list, xbin=xbin, measured_position_df=measured_position_df, **kwargs)
                    
            else:
                raise NotImplementedError(f'type(a_1D_decoded_result): {type(a_1D_decoded_result)} is unknown')

    @function_attributes(short_name=None, tags=['IMPORTANT', 'track', 'posterior', 'marginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-21 08:10', related_items=[])
    def add_docked_marginal_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None):
        """ adds a marginal (such as Long v. Short, or Long_LR v. Long_RL v. Short_LR v. Short_RL) 
        
        time_bin_size = epochs_decoding_time_bin_size
        info_string: str = f" - t_bin_size: {time_bin_size}"
        identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_marginal_track(name='non-PBE_marginal_over_track_ID',
                                                                                                time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
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



    # ==================================================================================================================== #
    # Multiple Tracks at once:                                                                                             #
    # ==================================================================================================================== #
    def add_docked_decoded_results_dict_tracks(self, name: str, a_decoded_result_dict: Dict[str, Union[SingleEpochDecodedResult, DecodedFilterEpochsResult]], dock_configs: Dict[str, CustomDockDisplayConfig], pf1D_Decoder_dict: Dict[str, BasePositionDecoder], measured_position_df: Optional[pd.DataFrame]=None, **kwargs):
        """
        info_string: str = f'{active_time_bin_size:.3f}'
        dock_group_sep_character: str = '_'
        showCloseButton = True
        _common_dock_config_kwargs = {'dock_group_names': [dock_group_sep_character.join([f'ContinuousDecode', info_string])], 'showCloseButton': showCloseButton}
        dock_configs: Dict[str, CustomDockDisplayConfig] = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'),
                                (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs))))
                                
        output_dict = active_2d_plot.add_docked_decoded_results_dict_tracks(name=f'DirectionalDecodersDecoded', a_decoded_result_dict=a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict, dock_configs=dock_configs, pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict,
                                                                                                    measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                    extended_dock_title_info=info_string)
                                                                                                    
                                                                                                    


        Usage 2:
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, DockDisplayColors

            
            ## INPUTS: laps_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
            unique_decoder_names = ['long', 'short']
            laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)

            active_time_bin_size: float = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size
            info_string: str = f'{active_time_bin_size:.3f}'
            dock_group_sep_character: str = '_'
            showCloseButton = True
            _common_dock_config_kwargs = {'dock_group_names': [dock_group_sep_character.join([f'LapsDecode', info_string])], 'showCloseButton': showCloseButton}
            dock_configs: Dict[str, CustomDockDisplayConfig] = dict(zip(unique_decoder_names,
                                    (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Epochs.get_long_dock_colors, **_common_dock_config_kwargs),
                                    CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Epochs.get_short_dock_colors, **_common_dock_config_kwargs))))
                                    

            pf1D_Decoder_dict = {k:deepcopy(v) for k, v in results1D.decoders.items() if k in unique_decoder_names}

            output_dict = active_2d_plot.add_docked_decoded_results_dict_tracks(name=f'LapsDecode', a_decoded_result_dict=laps_pseudo2D_split_to_1D_continuous_results_dict, dock_configs=dock_configs, pf1D_Decoder_dict=pf1D_Decoder_dict,
                                                                                                        measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                        extended_dock_title_info=info_string)
                                                                                                                                                                                                            
                                                                                                    
        """
        output_dict = {}
        for a_decoder_name, a_1D_decoded_result in a_decoded_result_dict.items():
            ## a_1D_continuous_decoded_result: SingleEpochDecodedResult
            a_dock_config = dock_configs[a_decoder_name]
            a_1D_decoder: BasePositionDecoder = pf1D_Decoder_dict[a_decoder_name]
            _out_tuple = self.add_docked_decoded_posterior_track_from_result(name=f'{name}[{a_decoder_name}]', a_dock_config=a_dock_config, a_1D_decoded_result=a_1D_decoded_result,
                                                                                                    xbin = deepcopy(a_1D_decoder.xbin), measured_position_df=deepcopy(measured_position_df), **kwargs) # , should_defer_render=False
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = _out_tuple
            ## Add `a_decoded_result` to the plots_data
            widget.plots_data.a_decoded_result = a_1D_decoded_result
            widget.plots_data.a_decoder = deepcopy(a_1D_decoder)
            output_dict[a_decoder_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem) ## add again
        ## END for a_decoder_name, a_1D_continuous_decoded_result 
        return output_dict



    def compute_if_needed_and_add_continuous_decoded_posterior(self, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
        """ computes the continuously decoded position posteriors (if needed) using the pipeline, then adds them as a new track to the SpikeRaster2D 
        from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs`
        based off of `pyphoplacecellanalysis.SpecificResults.PendingNotebookCode.add_continuous_decoded_posterior`
        
        Usage:    
            output_dict = active_2d_plot.compute_if_needed_and_add_continuous_decoded_posterior(curr_active_pipeline=curr_active_pipeline, desired_time_bin_size=0.050, debug_print=True)
            output_dict = active_2d_plot.compute_if_needed_and_add_continuous_decoded_posterior(curr_active_pipeline=curr_active_pipeline, desired_time_bin_size=0.025, debug_print=True)
        """
        # ==================================================================================================================== #
        # COMPUTING                                                                                                            #
        # ==================================================================================================================== #
        
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        
        ## get the result data:
        try:
            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            # pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            a_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(desired_time_bin_size, None)
            all_time_bin_sizes_output_dict: Dict[float, Dict[types.DecoderName, SingleEpochDecodedResult]] = directional_decoders_decode_result.split_pseudo2D_continuous_result_to_1D_continuous_result()
            a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict: Dict[types.DecoderName, SingleEpochDecodedResult] = all_time_bin_sizes_output_dict.get(desired_time_bin_size, None)
            
            assert a_continuously_decoded_dict is not None, f"a_continuously_decoded_dict is None even after recomputing!"
            assert a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict is not None, f"a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict is None even after recomputing!"
            info_string: str = f" - t_bin_size: {desired_time_bin_size:.3f}"

        except (KeyError, AttributeError) as e:
            # KeyError: 'DirectionalDecodersDecoded'
            print(f'add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(...) failed to add any tracks, perhaps because the pipeline is missing any computed "DirectionalDecodersDecoded" global results. Error: "{e}". Skipping.')
            a_continuously_decoded_dict = None
            pseudo2D_decoder = None        
            pass

        except Exception as e:
            raise


        # ==================================================================================================================== #
        # PLOTTING                                                                                                             #
        # ==================================================================================================================== #
        dock_group_sep_character: str = '_'
        showCloseButton = True
        _common_dock_config_kwargs = {'dock_group_names': [dock_group_sep_character.join([f'ContinuousDecode', info_string])], 'showCloseButton': showCloseButton}
        dock_configs: Dict[str, CustomDockDisplayConfig] = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'),
                                (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs))))
        
        output_dict = self.add_docked_decoded_results_dict_tracks(name=f'ContinuousDecode', a_decoded_result_dict=a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict, dock_configs=dock_configs, pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict,
                                                                                                    measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                    extended_dock_title_info=info_string)
        ## layout the dockGroups:
        nested_dock_items, nested_dynamic_docked_widget_container_widgets = self.ui.dynamic_docked_widget_container.layout_dockGroups()
        
        # OUTPUTS: output_dict
        return output_dict



    @function_attributes(short_name=None, tags=['UNFINISHED', 'plotting', 'computing'], input_requires=[], output_provides=[], uses=['AddNewDecodedPosteriors_MatplotlibPlotCommand', '_perform_plot_multi_decoder_meas_pred_position_track'], used_by=[], creation_date='2025-02-13 14:58', related_items=['_perform_plot_multi_decoder_meas_pred_position_track'])
    @classmethod
    def perform_add_continuous_decoded_posterior(cls, spike_raster_window, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
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
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        ## get the result data:
        try:
            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            a_continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(desired_time_bin_size, None)
            if a_continuously_decoded_dict is None:
                ## recompute
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
                directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
                pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
                all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
                a_continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(desired_time_bin_size, None)
                assert a_continuously_decoded_dict is not None, f"a_continuously_decoded_dict is None even after recomputing!"

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


        # output_dict = self.add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(self._active_pipeline, active_2d_plot, **kwargs)
        
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



    # ==================================================================================================================== #
    # MARK: PyQtGraph                                                                                                      #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['intervals', 'tracks', 'pyqtgraph', 'specific', 'dynamic_ui', 'group_matplotlib_render_plot_widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 07:29', related_items=[])
    def prepare_pyqtgraph_intervalPlot_tracks(self, enable_interval_overview_track: bool = False, should_remove_all_and_re_add: bool=True, name_modifier_suffix: str='', should_link_to_main_plot_widget:bool=True, debug_print=False):
        """ adds to separate pyqtgraph-backed tracks to the SpikeRaster2D plotter for rendering intervals, and updates `active_2d_plot.params.custom_interval_rendering_plots` so the intervals are rendered on these new tracks in addition to any normal ones
        
        enable_interval_overview_track: bool: if True, renders a track to show all the intervals during the sessions (overview) in addition to the track for the intervals within the current active window
        should_remove_all_and_re_add: bool: if True, all intervals are removed from all plots and then re-added (safer) method
        
        Updates:
            active_2d_plot.params.custom_interval_rendering_plots


        This should be a separate file, and there should be multiple classes of tracks (raster, instervals, etc) 
            
        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme

        _interval_tracks_out_dict = {}
        if enable_interval_overview_track:
            dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.grey, showCloseButton=False, corner_radius='0px', hideTitleBar=True)
            name = f'interval_overview{name_modifier_suffix}'
            intervals_overview_time_sync_pyqtgraph_widget, intervals_overview_root_graphics_layout_widget, intervals_overview_plot_item, intervals_overview_dock = self.add_new_embedded_pyqtgraph_render_plot_widget(name=name, dockSize=(500, 60), display_config=dock_config)
            _interval_tracks_out_dict[name] = (dock_config, intervals_overview_time_sync_pyqtgraph_widget, intervals_overview_root_graphics_layout_widget, intervals_overview_plot_item)
        ## Enables creating a new pyqtgraph-based track to display the intervals/epochs
        interval_window_dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.grey, showCloseButton=False, corner_radius='0px', hideTitleBar=True)
        name = f'intervals{name_modifier_suffix}'
        intervals_time_sync_pyqtgraph_widget, intervals_root_graphics_layout_widget, intervals_plot_item, intervals_dock = self.add_new_embedded_pyqtgraph_render_plot_widget(name=name, dockSize=(10, 4), display_config=interval_window_dock_config)
        
        self.params.custom_interval_rendering_plots.append(intervals_plot_item) # = [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget, intervals_plot_item]

        # self.params.custom_interval_rendering_plots = [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget, intervals_plot_item]
        # active_2d_plot.params.custom_interval_rendering_plots = [active_2d_plot.plots.background_static_scroll_window_plot, active_2d_plot.plots.main_plot_widget, intervals_plot_item, intervals_overview_plot_item]
        if enable_interval_overview_track:
            self.params.custom_interval_rendering_plots.append(intervals_overview_plot_item)
            
        # active_2d_plot.interval_rendering_plots

        _interval_tracks_out_dict[name] = (interval_window_dock_config, intervals_time_sync_pyqtgraph_widget, intervals_root_graphics_layout_widget, intervals_plot_item)

        ## #TODO 2024-12-31 07:20: - [ ] need to clear/re-add the epochs to make this work
        extant_rendered_interval_plots_lists = {k:list(v.keys()) for k, v in self.list_all_rendered_intervals(debug_print=False).items()}
        # active_target_interval_render_plots = active_2d_plot.params.custom_interval_rendering_plots

        # active_target_interval_render_plots = [v.objectName() for v in active_2d_plot.interval_rendering_plots]
        active_target_interval_render_plots_dict = {v.objectName():v for v in self.interval_rendering_plots}

        for a_name in self.interval_datasource_names:
            a_ds = self.interval_datasources[a_name]
            an_already_added_plot_list = extant_rendered_interval_plots_lists[a_name]
            if debug_print:
                print(f'a_name: {a_name}\n\tan_already_added_plot_list: {an_already_added_plot_list}')
                

            if should_remove_all_and_re_add:
                extant_already_added_plots = {k:v for k, v in active_target_interval_render_plots_dict.items() if k in an_already_added_plot_list}
                extant_already_added_plots_list = list(extant_already_added_plots.values())
                self.remove_rendered_intervals(name=a_name, child_plots_removal_list=extant_already_added_plots_list)
                self.add_rendered_intervals(interval_datasource=a_ds, name=a_name, child_plots=self.interval_rendering_plots) ## re-add ALL
                
            else:
                remaining_new_plots = {k:v for k, v in active_target_interval_render_plots_dict.items() if k not in an_already_added_plot_list}
                remaining_new_plots_list = list(remaining_new_plots.values())
                self.add_rendered_intervals(interval_datasource=a_ds, name=a_name, child_plots=remaining_new_plots_list) ## ADD only the new
            
        if should_link_to_main_plot_widget and (self.plots.main_plot_widget is not None):
            main_plot_widget = self.plots.main_plot_widget # PlotItem
            intervals_plot_item.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)
        else:
            ## setup the synchronization:
            # Perform Initial (one-time) update from source -> controlled:
            # intervals_time_sync_pyqtgraph_widget.on_window_changed(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
            intervals_plot_item.setXRange(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, padding=0)
            # disable active window syncing if it's enabled:
            sync_connection = self.ui.connections.get(name, None)
            if sync_connection is not None:
                # have an existing sync connection, need to disconnect it.
                print(f'disconnecting window_scrolled for "{name}"')
                self.window_scrolled.disconnect(sync_connection)
                    
            # sync_connection = self.window_scrolled.connect(intervals_time_sync_pyqtgraph_widget.on_window_changed)
            sync_connection = self.window_scrolled.connect(lambda earliest_t, latest_t: intervals_plot_item.setXRange(earliest_t, latest_t, padding=0)) ## explicitly captures `raster_plot_item`
            self.ui.connections[name] = sync_connection # add the connection to the connections array

        return _interval_tracks_out_dict


    @function_attributes(short_name=None, tags=['raster', 'tracks', 'pyqtgraph', 'specific', 'dynamic_ui', 'group_matplotlib_render_plot_widget'], input_requires=[], output_provides=[], uses=['self.add_new_embedded_pyqtgraph_render_plot_widget'], used_by=[], creation_date='2025-01-09 10:50', related_items=[])
    def prepare_pyqtgraph_rasterPlot_track(self, name_modifier_suffix: str='', should_link_to_main_plot_widget:bool=True, debug_print=False):
        """ adds to separate pyqtgraph-backed tracks to the SpikeRaster2D plotter for rendering a 2D raster `active_2d_plot.params.custom_interval_rendering_plots` so the intervals are rendered on these new tracks in addition to any normal ones
        
        enable_interval_overview_track: bool: if True, renders a track to show all the intervals during the sessions (overview) in addition to the track for the intervals within the current active window
        should_remove_all_and_re_add: bool: if True, all intervals are removed from all plots and then re-added (safer) method
        
        Updates:
            active_2d_plot.params.custom_interval_rendering_plots


        This should be a separate file, and there should be multiple classes of tracks (raster, instervals, etc) 
            

        #TODO 2025-01-09 12:04: - [ ] Needs to respond to signals:
        self.on_neuron_colors_changed

        Usage:
        
        _raster_tracks_out_dict = active_2d_plot.prepare_pyqtgraph_rasterPlot_track(name_modifier_suffix='raster_window', should_link_to_main_plot_widget=has_main_raster_plot)
        
        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot #, NewSimpleRaster, paired_separately_sort_neurons
        from neuropy.utils.indexing_helpers import find_desired_sort_indicies
        from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState # required for the different emphasis states in ._build_cell_configs()

        _raster_tracks_out_dict = {}
        ## Enables creating a new pyqtgraph-based track to display the intervals/epochs
        dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.grey, showCloseButton=True, showCollapseButton=False, showGroupButton=False, corner_radius="0px", hideTitleBar=True)
        name = f'rasters[{name_modifier_suffix}]'
        time_sync_pyqtgraph_widget, raster_root_graphics_layout_widget, raster_plot_item, raster_dock = self.add_new_embedded_pyqtgraph_render_plot_widget(name=name, dockSize=(10, 4), display_config=dock_config)

        if raster_plot_item not in self.params.custom_interval_rendering_plots:
            self.params.custom_interval_rendering_plots.append(raster_plot_item) ## this signals that it should recieve updates for its intervals somewhere else
        
        # active_2d_plot.params.custom_interval_rendering_plots = [active_2d_plot.plots.background_static_scroll_window_plot, active_2d_plot.plots.main_plot_widget, raster_plot_item, intervals_overview_plot_item]

        # active_2d_plot.interval_rendering_plots
        # main_plot_widget = self.plots.main_plot_widget # PlotItem
        # raster_plot_item.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)
        
        # self.unit_sort_order exists too

        # self.spikes_window # SpikesDataframeWindow
        spikes_df: pd.DataFrame = self.spikes_window.df ## all spikes ( for all time )
        # self.spikes_df # use this instead?
        
        # # an_included_unsorted_neuron_ids = deepcopy(included_any_context_neuron_ids_dict[a_decoder_name])
        an_included_unsorted_neuron_ids = deepcopy(self.neuron_ids)
        a_sorted_neuron_ids = deepcopy(self.ordered_neuron_ids)

        unit_sort_order, desired_sort_arr = find_desired_sort_indicies(an_included_unsorted_neuron_ids, a_sorted_neuron_ids)
        
        #TODO 2025-01-09 11:32: - [ ] Ignores each cell's actual emphasis state and forces default:
        curr_spike_emphasis_state: SpikeEmphasisState = SpikeEmphasisState.Default
        unsorted_unit_colors_map: Dict[types.aclu_index, pg.QColor] = {aclu:v[2][curr_spike_emphasis_state].color() for aclu, v in self.params.config_items.items()} # [2] is hardcoded and the only element of the tuple used, something legacy I guess

        # Get only the spikes for the shared_aclus:
        a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(an_included_unsorted_neuron_ids)
        a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

        time_sync_pyqtgraph_widget.plots.root_plot = raster_plot_item # name the plotItem "root_plot" so `new_plot_raster_plot` can reuse it
        rasters_display_outputs_tuple = new_plot_raster_plot(a_spikes_df, an_included_unsorted_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=deepcopy(unsorted_unit_colors_map),
                                                        scatter_app_name=name, defer_show=True, active_context=None,
                                                        win=raster_root_graphics_layout_widget, plots_data=time_sync_pyqtgraph_widget.plots_data, plots=time_sync_pyqtgraph_widget.plots,
                                                        add_debug_header_label=False,
                                                        scatter_plot_kwargs=dict(size=5, hoverable=False, tick_width=0.0, tick_height=1.0),
                                                        ) # defer_show=True so we can add it manually to the track view

        # an_app, a_win, a_plots, a_plots_data, an_on_update_active_epoch, an_on_update_active_scatterplot_kwargs = rasters_display_outputs_tuple
        # raster_root_graphics_layout_widget.addWidget(a_win, row=1, col=1)
        
        _raster_tracks_out_dict[name] = (dock_config, time_sync_pyqtgraph_widget, raster_root_graphics_layout_widget, raster_plot_item, rasters_display_outputs_tuple)
        # Setup range for plot:
        # earliest_t, latest_t = active_2d_plot.spikes_window.total_df_start_end_times # global
        earliest_t, latest_t = self.spikes_window.active_time_window # current
        raster_plot_item.setXRange(earliest_t, latest_t, padding=0)
        neuron_y_pos = np.array(list(deepcopy(time_sync_pyqtgraph_widget.plots_data.new_sorted_raster.neuron_y_pos).values()))
        raster_plot_item.setYRange(np.nanmin(neuron_y_pos), np.nanmax(neuron_y_pos), padding=0)

        if should_link_to_main_plot_widget:
            main_plot_widget = self.plots.main_plot_widget # PlotItem
            raster_plot_item.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)
        else:
            ## setup the synchronization:
            # Perform Initial (one-time) update from source -> controlled:
            # time_sync_pyqtgraph_widget.on_window_changed(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
            # disable active window syncing if it's enabled:
            sync_connection = self.ui.connections.get(name, None)
            if sync_connection is not None:
                # have an existing sync connection, need to disconnect it.
                print(f'disconnecting window_scrolled for "{name}"')
                self.window_scrolled.disconnect(sync_connection)
                        
            # sync_connection = self.window_scrolled.connect(time_sync_pyqtgraph_widget.on_window_changed)
            sync_connection = self.window_scrolled.connect(lambda earliest_t, latest_t: raster_plot_item.setXRange(earliest_t, latest_t, padding=0)) ## explicitly captures `raster_plot_item`
            self.ui.connections[name] = sync_connection # add the connection to the connections array
            

        return _raster_tracks_out_dict


