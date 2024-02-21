# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple, Any, Union
import matplotlib as mpl
import napari
from neuropy.analyses.placefields import PfND
from neuropy.core.epoch import TimeColumnAliasesProtocol
import numpy as np
import pandas as pd
from attrs import define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes



# ==================================================================================================================== #
# 2024-02-15 - Radon Transform / Weighted Correlation, etc helpers                                                     #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalMergedDecodersResult # _compute_complete_df_metrics

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import PhoDockAreaContainingWindow
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import DockAreaWrapper
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices_paginated
from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import WeightedCorrelationPaginatedPlotDataProvider


def align_decoder_pagination_controller_windows(pagination_controller_dict):
    """ resizes and aligns all windows. Not needed with PhoPaginatedMultiDecoderDecodedEpochsWindow (only used when plotting in separate windows) 
    Usage:
        align_decoder_pagination_controller_windows(pagination_controller_dict)

    """
    from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers, DesiredWidgetLocation, WidgetGeometryInfo
    ## Connects the first plotter's pagination controls to the other three controllers so that they are directly driven, by the first.
    a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
    a_controlling_widget = a_controlling_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
    # controlled widgets
    controlled_pagination_controllers_list = (pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])

    fixed_height_pagination_control_bar: float = 21.0
    target_height: float = a_controlling_widget.window().height()
    ratio_content_height = (target_height - fixed_height_pagination_control_bar) / target_height
    print(f'fixed_height_pagination_control_bar: {fixed_height_pagination_control_bar}, target_height: {target_height}, ratio_content_height: {ratio_content_height}')

    target_window = a_controlling_widget.window()
    for a_controlled_pagination_controller in controlled_pagination_controllers_list:
        # hide the pagination widget:
        a_controlled_widget = a_controlled_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
        WidgetPositioningHelpers.align_window_edges(target_window, a_controlled_widget.window(), relative_position = 'right_of', resize_to_main=(1.0, ratio_content_height)) # use ratio_content_height to compensate for the lack of a pagination scroll bar
        target_window = a_controlled_widget.window() # update to reference the newly moved window
        ratio_content_height = 1.0 # after the first window, 1.0 should be used since they're all the same height




class PhoPaginatedMultiDecoderDecodedEpochsWindow(PhoDockAreaContainingWindow):
    """ a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
        pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper.PhoDockAreaContainingWindow
        
        Inherited Properties: .area
        
    """
    @property
    def contents(self):
        return self.ui._contents
    
    @property
    def pagination_controllers(self) -> Dict:
        return self.contents.pagination_controllers

    @property
    def paginator_controller_widget(self) -> PaginationControlWidget:
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        paginator_controller_widget = a_controlling_pagination_controller.ui.mw.ui.paginator_controller_widget
        return paginator_controller_widget

    def __init__(self, title='PhoPaginatedMultiDecoderDecodedEpochsWindow', *args, **kwargs):
        super(PhoPaginatedMultiDecoderDecodedEpochsWindow, self).__init__(*args, **kwargs)
        self.ui._contents = None
        # self.setup()
        # self.buildUI()
        
    @classmethod
    def init_from_pagination_controller_dict(cls, pagination_controller_dict, name = 'CombinedDirectionalDecoderDecodedEpochsWindow', title='Pho Combined Directional Decoder Decoded Epochs', defer_show=False):
        """ 2024-02-14 - Copied from `RankOrderRastersDebugger`'s approach. Merges the four separate decoded epoch windows into single figure with a separate dock for each decoder.
        [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261)

        Usage:
            app, root_dockAreaWindow, _out_dock_widgets, dock_configs = merge_single_window(pagination_controller_dict)

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum

        ## Convert to controlled first
        new_connections_dict = cls.convert_decoder_pagination_controller_dict_to_controlled(pagination_controller_dict)

        # pagination_controller_dict = _obj.plots.rasters_display_outputs
        all_widgets = {a_decoder_name:a_pagination_controller.ui.mw for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_windows = {a_decoder_name:a_pagination_controller.ui.mw.window() for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots = {a_decoder_name:a_pagination_controller.plots for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots_data = {a_decoder_name:a_pagination_controller.plots_data for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_params = {a_decoder_name:a_pagination_controller.params for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}

        main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        
        # all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # # Extract the plot/renderable items
        # all_separate_root_plots = {a_decoder_name:a_pagination_controller.plots.root_plot for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        # all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # Embedding in docks:
        
        # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Combined Directional Decoder Decoded Epochs')
        
        # Instantiate the class ______________________________________________________________________________________________ #
        # root_dockAreaWindow = PhoDockAreaContainingWindow(title=title)
        root_dockAreaWindow = cls(title=title)
        root_dockAreaWindow.setWindowTitle(f'{title}: dockAreaWindow')
        app = root_dockAreaWindow.app
        
            
        # icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        # if icon is not None:
        #     root_dockAreaWindow.setWindowIcon(icon)

        ## Build Dock Widgets:
        _out_dock_widgets = {}
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))

        for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(430,700), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name], autoOrientation=False)

        # #TODO 2024-02-14 18:44: - [ ] Comgbine the separate items into one of the single `DecodedEpochSlicesPaginatedFigureController` objects (or a new one)?
        # root_dockAreaWindow.resize(600, 900)

        # ## Build final .plots and .plots_data:
        # PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)

        # root_dockAreaWindow.ui.connections.controlled_pagination_controllers_from_leftmost = new_connections_dict #
        root_dockAreaWindow.ui._contents = PhoUIContainer(name=name, names=main_plot_identifiers_list, pagination_controllers=pagination_controller_dict, 
                                                    dock_widgets=_out_dock_widgets, dock_configs=dock_configs,
                                                    widgets=all_widgets, windows=all_windows, plots=all_separate_plots, plots_data=all_separate_plots_data, params=all_separate_params) # do I need this extracted data or is it redundant?
        
        # _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
        #                             root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
        #                             dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        # _obj.plots_data = RenderPlotsData(name=name, main_plot_identifiers_list=main_plot_identifiers_list,
        #                                     seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs, seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
        #                                     on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs, **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']})
        # _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)
        # _obj.params = VisualizationParameters(name=name, is_laps=False, enable_show_spearman=True, enable_show_pearson=False, enable_show_Z_values=True, use_plaintext_title=False, **param_kwargs)


        # Add functions ______________________________________________________________________________________________________ #

        
        # ## Cleanup when done:
        # for a_decoder_name, a_root_plot in _obj.plots.root_plots.items():
        #     a_root_plot.setTitle(title=a_decoder_name)
        #     # a_root_plot.setTitle(title="")
        #     a_left_axis = a_root_plot.getAxis('left')# axisItem
        #     a_left_axis.setLabel(a_decoder_name)
        #     a_left_axis.setStyle(showValues=False)
        #     a_left_axis.setTicks([])
        #     # a_root_plot.hideAxis('bottom')
        #     # a_root_plot.hideAxis('bottom')
        #     a_root_plot.hideAxis('left')
        #     a_root_plot.setYRange(-0.5, float(_obj.max_n_neurons))

        # app, root_dockAreaWindow, _out_dock_widgets, dock_configs


        if not defer_show:
            root_dockAreaWindow.show()
            
        return app, root_dockAreaWindow


    ## Add a jump to page function
    def jump_to_page(self, page_idx: int):
        self.paginator_controller_widget.programmatically_update_page_idx(page_idx, block_signals=False) # don't block signals and then we don't have to call updates.        
        # for a_name, a_pagination_controller in self.contents.pagination_controllers.items():
        #     a_pagination_controller.on_paginator_control_widget_jump_to_page(page_idx=page_idx)
        

    def add_data_overlays(self, track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, included_columns=None):
        """ builds the Radon Transforms and Weighted Correlation data and adds them to the plot.
        
        REFINEMENT: note that it only plots either 'laps' or 'ripple', not both, so it doesn't need all this data.
        """
        ## Choose which columns from the filter_epochs dataframe to include on the plot.
        if included_columns is None:
            included_columns = []

        epoch_type_names_list: List[str] = [a_pagination_controller.params.active_identifying_figure_ctx.epochs for a_pagination_controller in self.pagination_controllers.values()]
        # All epoch_type_names should be the same, either 'laps' or 'ripple':
        assert (len(set(epoch_type_names_list)) == 1), f"All epoch_type_names should be the same, either 'laps' or 'ripple', but they are not: epoch_type_names_list: {epoch_type_names_list}"
        epoch_type_name: str = epoch_type_names_list[0]
        assert epoch_type_name in ['laps', 'ripple']

        # Build Radon Transforms and add them:
        if epoch_type_name == 'laps':
            radon_transform_epochs_data_dict = RadonTransformPlotDataProvider.decoder_build_radon_transform_data_dict(track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict)
        elif epoch_type_name == 'ripple':
            radon_transform_epochs_data_dict = RadonTransformPlotDataProvider.decoder_build_radon_transform_data_dict(track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict)
        else:
            raise NotImplementedError(f"epoch_type_name: {epoch_type_name}")
    
        if radon_transform_epochs_data_dict is not None:
            ## Add the radon_transform_lines to each of the four figures:
            for a_name, a_pagination_controller in self.pagination_controllers.items():
                if radon_transform_epochs_data_dict[a_name] is not None:
                    RadonTransformPlotDataProvider.add_data_to_pagination_controller(a_pagination_controller, radon_transform_epochs_data_dict[a_name], update_controller_on_apply=False)
            

        # Build Weighted Correlation Data Info and add them:
        if epoch_type_name == 'laps':
            wcorr_epochs_data_dict = WeightedCorrelationPaginatedPlotDataProvider.decoder_build_weighted_correlation_data_dict(track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict)

        elif epoch_type_name == 'ripple':
            wcorr_epochs_data_dict = WeightedCorrelationPaginatedPlotDataProvider.decoder_build_weighted_correlation_data_dict(track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict)
        else:
            raise NotImplementedError(f"epoch_type_name: {epoch_type_name}")
    
        ## Add the radon_transform_lines to each of the four figures:
        if wcorr_epochs_data_dict is not None:
            for a_name, a_pagination_controller in self.pagination_controllers.items():
                if wcorr_epochs_data_dict[a_name] is not None:
                    WeightedCorrelationPaginatedPlotDataProvider.add_data_to_pagination_controller(a_pagination_controller, wcorr_epochs_data_dict[a_name], update_controller_on_apply=False)




    @classmethod
    def _subfn_prepare_plot_multi_decoders_stacked_epoch_slices(cls, curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict, epochs_name:str ='laps', included_epoch_indicies=None, defer_render=True, save_figure=True, **kwargs):
        """ 2024-02-14 - Adapted from the function that plots the Long/Short decoded epochs side-by-side for comparsion and updated to work with the multi-decoder track templates.
        
        ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
        
        Usage:
            (pagination_controller_L, pagination_controller_S), (fig_L, fig_S), (ax_L, ax_S), (final_context_L, final_context_S), (active_out_figure_paths_L, active_out_figure_paths_S) = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(curr_active_pipeline, defer_render=False)
        """

        params_kwargs = dict(skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True)
        pagination_controller_dict = {}
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            pagination_controller_dict[a_name] = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(decoder_decoded_epochs_result_dict[a_name].filter_epochs,
                                                                                                decoder_decoded_epochs_result_dict[a_name],
                                                                                                xbin=a_decoder.xbin, global_pos_df=curr_active_pipeline.sess.position.df,
                                                                                                a_name='DecodedEpochSlices', active_context=curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name),
                                                                                                max_subplots_per_page=8, debug_print=False, included_epoch_indicies=included_epoch_indicies, **params_kwargs) # , save_figure=save_figure

            # pagination_controller_dict[a_name], active_out_figure_paths_L, final_context_L = plot_decoded_epoch_slices_paginated(curr_active_pipeline, decoder_laps_filter_epochs_decoder_result_dict[a_name], curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj'), included_epoch_indicies=included_epoch_indicies, save_figure=save_figure, **kwargs)
            # fig_L = pagination_controller_L.plots.fig
            # ax_L = fig_L.get_axes()
            # if defer_render:
            #     widget_L = pagination_controller_L.ui.mw # MatplotlibTimeSynchronizedWidget
            #     widget_L.close()
            #     pagination_controller_L = None

        # Constrains each of the plotters at least to the minimum height:
        for a_name, a_pagination_controller in pagination_controller_dict.items():
            # a_pagination_controller.params.all_plots_height
            # resize to minimum height
            a_widget = a_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
            # a_widget.size()
            a_widget.setMinimumHeight(a_pagination_controller.params.all_plots_height)

        return pagination_controller_dict

    @classmethod
    def convert_decoder_pagination_controller_dict_to_controlled(cls, pagination_controller_dict):
        ## Connects the first plotter's pagination controls to the other three controllers so that they are directly driven, by the first.
        a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        a_controlling_widget = a_controlling_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget

        # controlled widgets
        controlled_pagination_controllers_list = (pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])

        new_connections_dict = []

        for a_controlled_pagination_controller in controlled_pagination_controllers_list:
            # hide the pagination widget:
            a_controlled_widget = a_controlled_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
            # a_controlled_widget.on_paginator_control_widget_jump_to_page(page_idx=0)
            a_connection = a_controlling_pagination_controller.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(a_controlled_pagination_controller.on_paginator_control_widget_jump_to_page) # bind connection
            new_connections_dict.append(a_connection)
            # a_controlled_widget.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection
            a_controlled_widget.ui.paginator_controller_widget.hide()

        return new_connections_dict





# ==================================================================================================================== #
# Usability/Conveninece Helpers                                                                                        #
# ==================================================================================================================== #

def plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin, point_dict=None, ax_dict=None, extra_decoder_values_dict=None, tuning_curves_dict=None, include_tuning_curves=False):
    """ 2024-02-06 - Plots the four position-binned-activity maps (for each directional decoding epoch) as a 4x4 subplot grid using matplotlib. 

    """
    from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap
    if tuning_curves_dict is None:
        assert include_tuning_curves == False
    
    # figure_kwargs = dict(layout="tight")
    figure_kwargs = dict(layout="none")

    if ax_dict is None:
        if not include_tuning_curves:
            # fig = plt.figure(layout="constrained", figsize=[9, 7], dpi=220, clear=True) # figsize=[Width, height] in inches.
            fig = plt.figure(figsize=[8, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
        else:
            # tuning curves mode:
            fig = plt.figure(figsize=[9, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR_curve", "ax_long_RL_curve"],
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                    ["ax_short_LR_curve", "ax_short_RL_curve"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                height_ratios=[1, 7, 7, 1], # tuning curves are smaller than laps
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
            curve_ax_names = ["ax_long_LR_curve", "ax_long_RL_curve", "ax_short_LR_curve", "ax_short_RL_curve"]

    else:
        if not include_tuning_curves:
            # figure already exists, reuse the axes
            assert len(ax_dict) == 4
            assert list(ax_dict.keys()) == ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]
        else:
            # tuning curves mode:
            assert len(ax_dict) == 8
            assert list(ax_dict.keys()) == ["ax_long_LR_curve", "ax_long_RL_curve", "ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL", "ax_short_LR_curve", "ax_short_RL_curve"]
        

    
    # Get the colormap to use and set the bad color
    cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
    cmap.set_bad(color='black')

    # Compute extents for imshow:
    imshow_kwargs = {
        'origin': 'lower',
        # 'vmin': 0,
        # 'vmax': 1,
        'cmap': cmap,
        'interpolation':'nearest',
        'aspect':'auto',
        'animated':True,
        'show_xticks':False,
    }

    _old_data_to_ax_mapping = dict(zip(['maze1_odd', 'maze1_even', 'maze2_odd', 'maze2_even'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))
    data_to_ax_mapping = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))

    # ['long_LR', 'long_RL', 'short_LR', 'short_RL']

    
    for k, v in curr_aclu_z_scored_tuning_map_matrix_dict.items():
        # is_first_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[0])
        is_last_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[-1])
        
        curr_ax = ax_dict[data_to_ax_mapping[k]]
        curr_ax.clear()
        
        # hist_data = np.random.randn(1_500)
        # xbin_centers = np.arange(len(hist_data))+0.5
        # ax_dict["ax_LONG_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, (-1.0 * curr_cell_normalized_tuning_curve), ax_dict["ax_LONG_pf_tuning_curve"], is_horizontal=True)

        n_epochs:int = np.shape(v)[1]
        epoch_indicies = np.arange(n_epochs)

        # Posterior distribution heatmaps at each point.
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], epoch_indicies[0], epoch_indicies[-1])           
        imshow_kwargs['extent'] = (xmin, xmax, ymin, ymax)

        # plot heatmap:
        curr_ax.set_xticklabels([])
        curr_ax.set_yticklabels([])
        fig, ax, im = visualize_heatmap(v.copy(), ax=curr_ax, title=f'{k}', layout='none', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))


        if include_tuning_curves:
            tuning_curve = tuning_curves_dict[k]
            curr_curve_ax = ax_dict[f"{data_to_ax_mapping[k]}_curve"]
            curr_curve_ax.clear()

            if tuning_curve is not None:
                # plot curve heatmap:
                if not is_last_item:
                    curr_curve_ax.set_xticklabels([])
                    # Leave the position x-ticks on for the last item

                curr_curve_ax.set_yticklabels([])
                ymin, ymax = 0, 1
                imshow_kwargs['extent'] = (xmin, xmax, 0, 1)
                fig, curr_curve_ax, im = visualize_heatmap(tuning_curve.copy(), ax=curr_curve_ax, title=f'{k}', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
                curr_curve_ax.set_xlim((xmin, xmax))
                curr_curve_ax.set_ylim((0, 1))
                
            point_ax = curr_curve_ax # draw the lines on the tuning curve axis
            
        else:
            point_ax = ax

        if point_dict is not None:
            if k in point_dict:
                # have points to plot
                point_ax.vlines(point_dict[k], ymin=ymin, ymax=ymax, colors='r', label=f'{k}_peak')
                

    # fig.tight_layout()
    # NOTE: these layout changes don't seem to take effect until the window containing the figure is resized.
    # fig.set_layout_engine('compressed') # TAKEWAY: Use 'compressed' instead of 'constrained'
    fig.set_layout_engine('none') # disabling layout engine. Strangely still allows window to resize and the plots scale, so I'm not sure what the layout engine is doing.


    # ax_dict["ax_SHORT_activity_v_time"].plot([1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 1, 2, 3, 3, 1, 2, 0, 0])
    # ax_dict["ax_SHORT_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, curr_cell_normalized_tuning_curve, ax_dict["ax_SHORT_pf_tuning_curve"], is_horizontal=True)
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_xticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_yticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_box

    return fig, ax_dict


def pho_jointplot(*args, **kwargs):
    """ wraps sns.jointplot to allow adding titles/axis labels/etc."""
    import seaborn as sns
    title = kwargs.pop('title', None)
    _out = sns.jointplot(*args, **kwargs)
    if title is not None:
        plt.suptitle(title)
    return _out


def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a stacked histogram of the many time-bin sizes """
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    time_bin_sizes = pre_delta_df['time_bin_size'].unique()
    
    figure_identifier: str = f"{descriptor_str}_preDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size)) 
    
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.legend()
    plt.show()

    # plot post-delta histogram
    time_bin_sizes = post_delta_df['time_bin_size'].unique()
    figure_identifier: str = f"{descriptor_str}_postDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size)) 
    
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    plt.legend()
    plt.show()


def pho_plothelper(data, **kwargs):
    """ 2024-02-06 - Provides an interface like plotly's classes provide to extract keys fom DataFrame columns or dicts and generate kwargs to pass to a plotting function.
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import pho_plothelper
            extracted_value_kwargs = pho_plothelper(data=an_aclu_conv_overlap_output['valid_subset'], x='x', y='normalized_convolved_result')
            extracted_value_kwargs

    """
    # data is a pd.DataFrame or Dict-like
    extracted_value_kwargs = {}
    for k,v in kwargs.items():
        extracted_value_kwargs[k] = data[v]
    # end up with `extracted_value_kwargs` containing the real values to plot.
    return extracted_value_kwargs
# ==================================================================================================================== #
# 2024-02-02 - Napari Export Helpers - Batch export all images                                                         #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_set_time_windw_index



import napari
from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict


def napari_add_aclu_slider(viewer, neuron_ids):
    """ adds a neuron aclu index overlay

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_add_aclu_slider


    """
    def on_update_slider(event):
        """ captures: viewer, neuron_ids
        
        Adds a little text label to the bottom right corner
        
        """
        # only trigger if update comes from first axis (optional)
        # print('inside')
        #ind_lambda = viewer.dims.indices[0]

        time = viewer.dims.current_step[0]
        matrix_aclu_IDX = int(time)
        # find the aclu value for this index:
        aclu: int = neuron_ids[matrix_aclu_IDX]
        viewer.text_overlay.text = f"aclu: {aclu}, IDX: {matrix_aclu_IDX}"
        
        # viewer.text_overlay.text = f"{time:1.1f} time"


    # viewer = napari.Viewer()
    # viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
    viewer.text_overlay.visible = True
    _connected_on_update_slider_event = viewer.dims.events.current_step.connect(on_update_slider)
    # viewer.dims.events.current_step.disconnect(on_update_slider)
    return _connected_on_update_slider_event


def napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts, include_trial_by_trial_correlation_matrix:bool = True):
    """ Plots the directional trial-by-trial activity visualization:
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_plot_directional_trial_by_trial_activity_viz
        
        directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict = napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts)
    
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict

    custom_direction_split_layers_dict = {}
    layers_list_sort_order = ['maze1_odd_z_scored_tuning_maps', 'maze1_odd_C_trial_by_trial_correlation_matrix', 
    'maze1_even_z_scored_tuning_maps', 'maze1_even_C_trial_by_trial_correlation_matrix',
    'maze2_odd_z_scored_tuning_maps', 'maze2_odd_C_trial_by_trial_correlation_matrix', 
    'maze2_even_z_scored_tuning_maps', 'maze2_even_C_trial_by_trial_correlation_matrix']

    ## Build the image data layers for each
    # for an_epoch_name, (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) in directional_active_lap_pf_results_dicts.items():
    for an_epoch_name, active_trial_by_trial_activity_obj in directional_active_lap_pf_results_dicts.items():
        # (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids)
        z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix
        custom_direction_split_layers_dict[f'{an_epoch_name}_z_scored_tuning_maps'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)) # reshape to be compatibile with C_i's dimensions
        if include_trial_by_trial_correlation_matrix:
            C_trial_by_trial_correlation_matrix = active_trial_by_trial_activity_obj.C_trial_by_trial_correlation_matrix
            custom_direction_split_layers_dict[f'{an_epoch_name}_C_trial_by_trial_correlation_matrix'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix)

    # custom_direction_split_layers_dict

    # directional_viewer, directional_image_layer_dict = napari_trial_by_trial_activity_viz(None, None, layers_dict=custom_direction_split_layers_dict)

    ## sort the layers dict:
    custom_direction_split_layers_dict = {k:custom_direction_split_layers_dict[k] for k in reversed(layers_list_sort_order) if k in custom_direction_split_layers_dict}

    directional_viewer, directional_image_layer_dict = napari_from_layers_dict(layers_dict=custom_direction_split_layers_dict, title='Directional Trial-by-Trial Activity', axis_labels=('aclu', 'lap', 'xbin'))
    if include_trial_by_trial_correlation_matrix:
        directional_viewer.grid.shape = (-1, 4)
    else:
        directional_viewer.grid.shape = (2, -1)

    return directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict


def napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix, layers_dict=None, **viewer_kwargs):
    """ Visualizes position binned activity matrix beside the trial-by-trial correlation matrix.
    

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_trial_by_trial_activity_viz
        viewer, image_layer_dict = napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix)

        
        image_layer_dict
        # can find peak spatial shift distance by performing convolution and finding time of maximum value?
        _layer_z_scored_tuning_maps = image_layer_dict['z_scored_tuning_maps']
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 56]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 55.5]]), step=array([1, 1, 1]))

        _layer_C_trial_by_trial_correlation_matrix = image_layer_dict['C_trial_by_trial_correlation_matrix']
        _layer_C_trial_by_trial_correlation_matrix.extent

        # _layer_z_scored_tuning_maps.extent
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 84]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 83.5]]), step=array([1, 1, 1]))

        # array([0, 0, 0])

    , title='Trial-by-trial Correlation Matrix C', axis_labels=('aclu', 'lap', 'xbin')
        
    Viewer properties:
        # viewer.grid # GridCanvas(stride=1, shape=(-1, -1), enabled=True)
        viewer.grid.enabled = True
        https://napari.org/0.4.15/guides/preferences.html
        https://forum.image.sc/t/dividing-the-display-in-the-viewer-window/42034
        https://napari.org/stable/howtos/connecting_events.html
        https://napari.org/stable/howtos/headless.html
        https://forum.image.sc/t/napari-how-add-a-text-label-time-always-in-the-same-spot-in-viewer/52932/3
        https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html
        https://napari.org/stable/gallery/add_labels.html
        
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict
    

    # inputs: z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix
    image_layer_dict = {}
    if layers_dict is None:
        # build the default from the values:
        layers_dict = {
            'z_scored_tuning_maps': dict(blending='translucent', colormap='viridis', name='z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)), # reshape to be compatibile with C_i's dimensions
            'C_trial_by_trial_correlation_matrix': dict(blending='translucent', colormap='viridis', name='C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix),
        }

    viewer = None
    for i, (a_name, layer_dict) in enumerate(layers_dict.items()):
        img_data = layer_dict.pop('img_data').astype(float) # assumes integrated img_data in the layers dict
        if viewer is None: #i == 0:
            # viewer = napari.view_image(img_data) # rgb=True
            viewer = napari.Viewer(**viewer_kwargs)

        image_layer_dict[a_name] = viewer.add_image(img_data, **(dict(name=a_name)|layer_dict))

    viewer.dims.axis_labels = ('aclu', 'lap', 'xbin')
    viewer.grid.enabled = True # enables the grid layout of the data so adjacent data is displayed next to each other

    # outputs: viewer, image_layer_dict
    return viewer, image_layer_dict



def napari_export_image_sequence(viewer: napari.viewer.Viewer, imageseries_output_directory='output/videos/imageseries/', slider_axis_IDX: int = 0, build_filename_from_viewer_callback_fn=None):
    """ 
    
    Based off of `napari_export_video_frames`
    
    Usage:
            
        desired_save_parent_path = Path('/home/halechr/Desktop/test_napari_out').resolve()
        imageseries_output_directory = napari_export_image_sequence(viewer=viewer, imageseries_output_directory=desired_save_parent_path, slider_axis_IDX=0, build_filename_from_viewer_callback_fn=build_filename_from_viewer)

    
    """
    # Get the slide info:
    slider_min, slider_max, slider_step = viewer.dims.range[slider_axis_IDX]
    slider_range = np.arange(start=slider_min, step=slider_step, stop=slider_max)

    # __MAX_SIMPLE_EXPORT_COUNT: int = 5
    n_time_windows = np.shape(slider_range)[0]
    # n_time_windows = min(__MAX_SIMPLE_EXPORT_COUNT, n_time_windows) ## Limit the export to 5 items for testing

    if not isinstance(imageseries_output_directory, Path):
        imageseries_output_directory: Path = Path(imageseries_output_directory).resolve()
        
    for window_idx in np.arange(n_time_windows):
        # napari_set_time_windw_index(viewer, window_idx+1)
        napari_set_time_windw_index(viewer, window_idx)
        
        if build_filename_from_viewer_callback_fn is not None:
            image_out_path = build_filename_from_viewer_callback_fn(viewer, desired_save_parent_path=imageseries_output_directory, slider_axis_IDX=slider_axis_IDX)
        else:
            image_out_path = imageseries_output_directory.joinpath(f'screenshot_{window_idx}.png').resolve()
                
        screenshot = viewer.screenshot(path=image_out_path, canvas_only=True, flash=False)

    return imageseries_output_directory



# ==================================================================================================================== #
# 2024-02-02 - Trial-by-trial Correlation Matrix C                                                                     #
# ==================================================================================================================== #

from nptyping import NDArray
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
@define(slots=False)
class TrialByTrialActivity:
    """ 2024-02-12 - Computes lap-by-lap placefields and helps display correlation matricies and such.
    
    """
    active_epochs_df: pd.DataFrame = field()
    C_trial_by_trial_correlation_matrix: NDArray = field()
    z_scored_tuning_map_matrix: NDArray = field()
    aclu_to_matrix_IDX_map: Dict = field() # factory=Factory(dict)
    neuron_ids: NDArray = field()
    
    @property 
    def stability_score(self) -> NDArray:
        return np.nanmedian(self.C_trial_by_trial_correlation_matrix, axis=(1,2))
    
    @property 
    def aclu_to_stability_score_dict(self) -> Dict[int, NDArray]:
        return dict(zip(self.neuron_ids, self.stability_score))
    


    @classmethod
    def compute_spatial_binned_activity_via_pfdt(cls, active_pf_dt: PfND_TimeDependent, epochs_df: pd.DataFrame, included_neuron_IDs=None):
        """ 2024-02-01 - Use pfND_dt to compute spatially binned activity during the epochs.
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_spatial_binned_activity_via_pfdt
            
            if 'pf1D_dt' not in curr_active_pipeline.computation_results[global_epoch_name].computed_data:
                # if `KeyError: 'pf1D_dt'` recompute
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)


            active_pf_1D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'])
            active_pf_2D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt'])


            laps_df = deepcopy(global_any_laps_epochs_obj.to_dataframe())
            n_laps = len(laps_df)

            active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_1D_dt)
            # active_pf_dt = deepcopy(active_pf_2D_dt) # 2D
            historical_snapshots = compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=laps_df)

        """
        use_pf_dt_obj = False

        if included_neuron_IDs is None:
            included_neuron_IDs = deepcopy(active_pf_dt.included_neuron_IDs) # this may be under-included. Is there like an "all-times-neuron_IDs?"
            
        if isinstance(epochs_df, pd.DataFrame):
            # dataframes are treated weird by PfND_dt, convert to basic numpy array of shape (n_epochs, 2)
            time_intervals = epochs_df[['start', 'stop']].to_numpy() # .shape # (n_epochs, 2)
        else:
            time_intervals = epochs_df # assume already a numpy array
            
        assert np.shape(time_intervals)[-1] == 2
        n_epochs: int = np.shape(time_intervals)[0]
            
        ## Entirely independent computations for binned_times:
        if use_pf_dt_obj:
            active_pf_dt.reset()

        # if included_neuron_IDs is not None:
        #     # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        
        if not use_pf_dt_obj:
            historical_snapshots = {} # build a dict<float:PlacefieldSnapshot>

        for start_t, end_t in time_intervals:
            ## Inline version that reuses active_pf_1D_dt directly:
            if use_pf_dt_obj:
                # active_pf_1D_dt.update(end_t, should_snapshot=True) # use this because it correctly integrates over [0, end_t] instead of [start_t, end_t]
                # active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=True, should_snapshot=True)
                historical_snapshots[float(end_t)] = active_pf_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=False, should_snapshot=False) # Integrates each [start_t, end_t] independently
            else:
                # Static version that calls PfND_TimeDependent.perform_time_range_computation(...) itself using just the computed variables of `active_pf_1D_dt`:
                all_time_filtered_spikes_df: pd.DataFrame = deepcopy(active_pf_dt.all_time_filtered_spikes_df).spikes.sliced_by_neuron_id(included_neuron_IDs)
                historical_snapshots[float(end_t)] = PfND_TimeDependent.perform_time_range_computation(all_time_filtered_spikes_df, active_pf_dt.all_time_filtered_pos_df, position_srate=active_pf_dt.position_srate,
                                                                            xbin=active_pf_dt.xbin, ybin=active_pf_dt.ybin,
                                                                            start_time=start_t, end_time=end_t,
                                                                            included_neuron_IDs=included_neuron_IDs, active_computation_config=active_pf_dt.config, override_smooth=active_pf_dt.smooth)

        # {1.9991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x16c2b74fb20>, 2.4991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x168acfb3bb0>, ...}
        if use_pf_dt_obj:
            historical_snapshots = active_pf_dt.historical_snapshots

        epoch_pf_results_dict = {'historical_snapshots': historical_snapshots}
        epoch_pf_results_dict['num_position_samples_occupancy'] = np.stack([placefield_snapshot.num_position_samples_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['seconds_occupancy'] = np.stack([placefield_snapshot.seconds_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['normalized_occupancy'] = np.stack([placefield_snapshot.normalized_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['spikes_maps_matrix'] = np.stack([placefield_snapshot.spikes_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        # active_lap_pf_results_dict['snapshot_occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in active_lap_pf_results_dict['historical_snapshots'].values()])

        # len(historical_snapshots)
        return epoch_pf_results_dict


    @classmethod
    def compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, occupancy_weighted_tuning_maps_matrix: NDArray, included_neuron_IDs=None):
        """ 2024-02-02 - computes the Trial-by-trial Correlation Matrix C 
        
        Returns:
            C_trial_by_trial_correlation_matrix: .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
            z_scored_tuning_map_matrix

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_trial_by_trial_correlation_matrix

            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix = compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix)

        """
        if included_neuron_IDs is None:
            neuron_ids = deepcopy(np.array(active_pf_dt.ratemap.neuron_ids))
        else:
            neuron_ids = np.array(included_neuron_IDs)
            

        n_aclus = len(neuron_ids)
        n_xbins = len(active_pf_dt.xbin_centers)

        assert np.shape(occupancy_weighted_tuning_maps_matrix)[1] == n_aclus
        assert np.shape(occupancy_weighted_tuning_maps_matrix)[2] == n_xbins

        epsilon_value: float = 1e-12
        # Assuming 'occupancy_weighted_tuning_maps_matrix' is your dataset with shape (trials, positions)
        # Z-score along the position axis (axis=1)
        position_axis_idx: int = 2
        z_scored_tuning_map_matrix: NDArray = (occupancy_weighted_tuning_maps_matrix - np.nanmean(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True)) / ((np.nanstd(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True))+epsilon_value)

        # trial-by-trial correlation matrix C
        M = float(n_xbins)
        C_list = []
        for i, aclu in enumerate(neuron_ids):
            A_i = np.squeeze(z_scored_tuning_map_matrix[:,i,:])
            C_i = (1/(M-1)) * (A_i @ A_i.T) # Perform matrix multiplication using the @ operator
            # C_i.shape # (n_epochs, n_epochs) - (84, 84) - gives the correlation between each epoch and the others
            C_list.append(C_i)
        # occupancy_weighted_tuning_maps_matrix

        C_trial_by_trial_correlation_matrix = np.stack(C_list, axis=0) # .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
        # outputs: C_trial_by_trial_correlation_matrix

        # n_laps: int = len(laps_unique_ids)
        aclu_to_matrix_IDX_map = dict(zip(neuron_ids, np.arange(n_aclus)))

        return C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map

    ## MAIN CALL:
    @classmethod
    def directional_compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, directional_lap_epochs_dict, included_neuron_IDs=None) -> Dict[str, "TrialByTrialActivity"]:
        """ 
        
        2024-02-02 - 10pm - Have global version working but want seperate directional versions. Seperately do `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`:
        
        """
        directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = {}

        # # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        # if included_neuron_IDs is not None:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()


        # Seperately do (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj):
        for an_epoch_name, active_laps_epoch in directional_lap_epochs_dict.items():
            active_laps_df = deepcopy(active_laps_epoch.to_dataframe())
            active_lap_pf_results_dict = cls.compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=active_laps_df, included_neuron_IDs=included_neuron_IDs)
            # Unpack the variables:
            historical_snapshots = active_lap_pf_results_dict['historical_snapshots']
            occupancy_weighted_tuning_maps_matrix = active_lap_pf_results_dict['occupancy_weighted_tuning_maps'] # .shape: (n_epochs, n_aclus, n_xbins) - (84, 80, 56)
            # 2024-02-02 - Trial-by-trial Correlation Matrix C
            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map = cls.compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix, included_neuron_IDs=included_neuron_IDs)
            neuron_ids = np.array(list(aclu_to_matrix_IDX_map.keys()))
            
            # directional_active_lap_pf_results_dicts[an_epoch_name] = (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) # currently discards: occupancy_weighted_tuning_maps_matrix, historical_snapshots, active_lap_pf_results_dict, active_laps_df
            directional_active_lap_pf_results_dicts[an_epoch_name] = TrialByTrialActivity(active_epochs_df=active_laps_df, C_trial_by_trial_correlation_matrix=C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix=z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map=aclu_to_matrix_IDX_map, neuron_ids=neuron_ids)
            
        return directional_active_lap_pf_results_dicts


# ==================================================================================================================== #
# 2024-02-01 - Spatial Information                                                                                     #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

def _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy):
    """ function to calculate Spatial Information (SI) score
    
    # f_i is the trial-averaged activity per position bin i -- sounds like the average number of spikes in each position bin within the trial

    # f is the mean activity rate over the whole session, computed as the sum of f_i * p_i over all N (position) bins

    ## What they call "p_i" - "occupancy probability per position bin per trial" ([Sosa et al., 2023, p. 23](zotero://select/library/items/I5FLMP5R)) ([pdf](zotero://open-pdf/library/items/C3Y8AKEB?page=23&annotation=GAHX9PYH))
    occupancy_probability = a_spikes_bin_counts_mat.copy()
    occupancy_probability = occupancy_probability / occupancy_probability.sum(axis=1, keepdims=True) # quotient is "total number of samples in each trial"
    occupancy_probability

    # We then summed the occupancy probabilities across trials and divided by the total per session to get an occupancy probability per position bin per session

    # To get the spatial “tuning curve” over the session, we averaged the activity in each bin across trials

    Usage:    
    SI = calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy)
    """
    ## SI Calculator: fi/<f>
    p_i = probability_normalized_occupancy.copy()

    # f_rate_over_all_session = global_all_spikes_counts['rate_Hz'].to_numpy()
    # f_rate_over_all_session
    check_f = np.nansum((p_i *  epoch_averaged_activity_per_pos_bin), axis=-1) # a check for f (rate over all session)
    f_rate_over_all_session = check_f # temporarily use check_f instead of the real f_rate

    fi_over_mean_f = epoch_averaged_activity_per_pos_bin / f_rate_over_all_session.reshape(-1, 1) # the `.reshape(-1, 1)` fixes the broadcasting

    log_base_2_of_fi_over_mean_f = np.log2(fi_over_mean_f) ## Here is where some entries become -np.inf

    _summand = (p_i * fi_over_mean_f * log_base_2_of_fi_over_mean_f) # _summand.shape # (77, 56)

    SI = np.nansum(_summand, axis=1)
    return SI


def compute_spatial_information(all_spikes_df: pd.DataFrame, an_active_pf: PfND, global_session_duration:float):
    """ Calculates the spatial information (SI) for each cell and returns all intermediates.

    Usage: 
        global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df).drop(columns=['neuron_type'], inplace=False)
        an_active_pf = deepcopy(global_pf1D)
        SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts = compute_spatial_information(all_spikes_df=global_spikes_df, an_active_pf=an_active_pf, global_session_duration=global_session.duration)


    """
    from neuropy.core.flattened_spiketrains import SpikesAccessor
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns

    #  Inputs: global_spikes_df: pd.DataFrame, an_active_pf: PfND, 
    # Build the aclu indicies:
    # neuron_IDs = global_spikes_df.aclu.unique()
    # n_aclus = global_spikes_df.aclu.nunique()
    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    all_spikes_df = deepcopy(all_spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # global_spikes_df


    # Get <f> for each sell, the rate over the entire session.
    global_all_spikes_counts = all_spikes_df.groupby(['aclu']).agg(t_count=('t', 'count')).reset_index()
    global_all_spikes_counts['rate_Hz'] = global_all_spikes_counts['t_count'] / global_session_duration
    # global_all_spikes_counts

    assert len(global_all_spikes_counts) == n_aclus
    
    ## Next need epoch-averaged activity per position bin:

    # Build the full matrix:
    global_per_position_bin_spikes_counts = all_spikes_df.groupby(['aclu', 'binned_x', 'binned_y']).agg(t_count=('t', 'count')).reset_index()
    a_spikes_df_bin_grouped = global_per_position_bin_spikes_counts.groupby(['aclu', 'binned_x']).agg(t_count_sum=('t_count', 'sum')).reset_index() ## for 1D plotting mode, collapse over all y-bins
    # a_spikes_df_bin_grouped

    assert n_aclus is not None
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)

    print(f'{n_aclus = }, {n_xbins = }')

    # a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell
    epoch_averaged_activity_per_pos_bin = np.zeros((n_aclus, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        # lap = int(row['lap'])
        aclu = int(row['aclu'])
        neuron_fragile_IDX: int = neuron_id_to_new_IDX_map[aclu]
        binned_x = int(row['binned_x'])
        count = row['t_count_sum']
        # a_spikes_bin_counts_mat[lap - 1][binned_x - 1] = count
        epoch_averaged_activity_per_pos_bin[neuron_fragile_IDX - 1][binned_x - 1] = count

    # an_active_pf.occupancy.shape # (n_xbins,) - (56,)
    # epoch_averaged_activity_per_pos_bin.shape # (n_aclus, n_xbins) - (77, 56)
    assert np.shape(an_active_pf.occupancy)[0] == np.shape(epoch_averaged_activity_per_pos_bin)[1]
        
    ## Compute actual Spatial Information for each cell:
    SI = _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy=an_active_pf.ratemap.probability_normalized_occupancy)

    return SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts


def permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100):
    """ Not yet implemented. 2024-02-01
    
    Based off of the following quote:
    To determine the significance of the SI scores, we created a null distribution by circularly permuting the position data relative to the timeseries of each cell, by a random amount of at least 1 sec and a maximum amount of the length of the trial, independently on each trial. SI was calculated from the trial-averaged activity of each shuffle, and this shuffle procedure was repeated 100 times per cell. A cell’s true SI was considered significant if it exceeded 95% of the SI scores from all shuffles within animal (i.e. shuffled scores were pooled across cells within animal to produce this threshold, which is more stringent than comparing to the shuffle of each individual cell
    
    Usage:
        # True place field rate maps for all cells
        rate_maps = np.array('your rate maps')
        # True occupancy maps for all cells
        occupancy_maps = np.array('your occupancy maps')
        # Your position data
        position_data = np.array('your position data')

        # Call the permutation test function with the given number of permutations
        sig_cells = permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100)

        print(f'Indices of cells with significant SI: {sig_cells}')

    
    """
    # function to calculate Spatial Information (SI) score
    def calc_SI(rate_map, occupancy):
        # Place your existing SI calculation logic here
        pass

    # function to calculate rate map for given position data
    def calc_rate_map(position_data):
        # logic to calculate rate map
        pass

    # function to calculate occupancy map for given position data
    def calc_occupancy_map(position_data):
        # logic to calculate occupancy map
        pass

    n_cells = rate_maps.shape[0]  # number of cells
    si_scores = np.empty((n_cells, n_permutations))  # Initialize container for SI scores per cell per permutation
    true_si_scores = np.empty(n_cells)  # Initialize container for true SI scores per cell
   
    for cell_idx in range(n_cells):
        true_si_scores[cell_idx] = calc_SI(rate_maps[cell_idx], occupancy_maps[cell_idx])
        
        for perm_idx in range(n_permutations):
            shift_val = np.random.randint(1, len(position_data))  # A random shift amount
            shuffled_position_data = np.roll(position_data, shift_val)  # Shift the position data
        
            shuffled_rate_map = calc_rate_map(shuffled_position_data)
            shuffled_occupancy_map = calc_occupancy_map(shuffled_position_data)

            si_scores[cell_idx][perm_idx] = calc_SI(shuffled_rate_map, shuffled_occupancy_map)
   
    pooled_scores = si_scores.flatten() # Pool scores within animal
    threshold = np.percentile(pooled_scores, 95)  # Get the 95th percentile of the pooled scores

    return np.where(true_si_scores > threshold)  # Return indices where true SI scores exceed 95 percentile


def compute_activity_by_lap_by_position_bin_matrix(a_spikes_df: pd.DataFrame, lap_id_to_matrix_IDX_map: Dict, n_xbins: int): # , an_active_pf: Optional[PfND] = None
    """ 2024-01-31 - Note that this does not take in position tracking information, so it cannot compute real occupancy. 
    
    Plots for a single neuron.
    
    an_active_pf: is just so we have access to the placefield's properties later
    
    
    Currently plots raw spikes counts (in number of spikes).
    
    """
    # Filter rows based on column: 'binned_x'
    a_spikes_df = a_spikes_df[a_spikes_df['binned_x'].astype("string").notna()]
    # a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y', 'lap']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    # a_spikes_df_bin_grouped

    ## for 1D plotting mode, collapse over all y-bins:
    a_spikes_df_bin_grouped = a_spikes_df_bin_grouped.groupby(['binned_x', 'lap']).agg(t_seconds_count_sum=('t_seconds_count', 'sum')).reset_index()
    # a_spikes_df_bin_grouped
    assert n_xbins is not None
    assert lap_id_to_matrix_IDX_map is not None
    n_laps: int = len(lap_id_to_matrix_IDX_map)
    
    a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        lap_id = int(row['lap'])
        lap_IDX = lap_id_to_matrix_IDX_map[lap_id]
        
        binned_x = int(row['binned_x'])
        count = row['t_seconds_count_sum']
        a_spikes_bin_counts_mat[lap_IDX][binned_x - 1] = count
        
    # active_out_matr = occupancy_probability
    
    # active_out_matr = a_spikes_bin_counts_mat
    # “calculated the occupancy (number of imaging samples) in each bin on each trial, and divided this by the total number of samples in each trial to get an occupancy probability per position bin per trial” 
    return a_spikes_bin_counts_mat

def compute_spatially_binned_activity(an_active_pf: PfND): # , global_any_laps_epochs_obj
    """ 
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_spatially_binned_activity
        
        # a_spikes_df = None
        # a_spikes_df: pd.DataFrame = deepcopy(long_one_step_decoder_1D.spikes_df) #.drop(columns=['neuron_type'], inplace=False)

        # an_active_pf = deepcopy(global_pf2D)
        # an_active_pf = deepcopy(global_pf1D)
        # an_active_pf.linear_pos_obj

        # an_active_pf = active_pf_2D_dt
        an_active_pf = active_pf_1D_dt
        position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map) = compute_spatially_binned_activity(an_active_pf)
        # 14.8s
    """
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
    # from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # needed to add laps column

    ## need global laps positions now.

    # # Position:
    # position_df: pd.DataFrame = deepcopy(an_active_pf.filtered_pos_df) # .drop(columns=['neuron_type'], inplace=False)
    # position_df, (xbin,), bin_infos = build_df_discretized_binned_position_columns(position_df, bin_values=(an_active_pf.xbin,), position_column_names=('lin_pos',), binned_column_names=('binned_x',), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # if 'lap' not in position_df:
    #     position_df = add_epochs_id_identity(position_df, epochs_df=deepcopy(global_any_laps_epochs_obj.to_dataframe()), epoch_id_key_name='lap', epoch_label_column_name='lap_id', no_interval_fill_value=-1, override_time_variable_name='t')
    #     # drop the -1 indicies because they are below the speed:
    #     position_df = position_df[position_df['lap'] != -1] # Drop all non-included spikes
    # position_df

    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    # all_spikes_df: pd.DataFrame = deepcopy(all_spikes_df) # require passed-in value
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df)
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.filtered_spikes_df)
    all_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df) # Use placefields all spikes 
    all_spikes_df = all_spikes_df.spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df = all_spikes_df[all_spikes_df['lap'] > -1] # get only the spikes within a lap
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)

    split_spikes_dfs_list = all_spikes_df.spikes.get_split_by_unit()
    split_spikes_df_dict = dict(zip(neuron_IDs, split_spikes_dfs_list))
    
    laps_unique_ids = all_spikes_df.lap.unique()
    n_laps: int = len(laps_unique_ids)
    lap_id_to_matrix_IDX_map = dict(zip(laps_unique_ids, np.arange(n_laps)))

    # n_laps: int = position_df.lap.nunique()
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)
    
    # idx: int = 9
    # aclu: int = neuron_IDs[idx]
    # print(f'aclu: {aclu}')
    
    position_binned_activity_matr_dict = {}

    # for a_spikes_df in split_spikes_dfs:
    for aclu, a_spikes_df in split_spikes_df_dict.items():
        # split_spikes_df_dict[aclu], (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(a_spikes_df.drop(columns=['neuron_type'], inplace=False), bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
        a_position_binned_activity_matr = compute_activity_by_lap_by_position_bin_matrix(a_spikes_df=a_spikes_df, lap_id_to_matrix_IDX_map=lap_id_to_matrix_IDX_map, n_xbins=n_xbins)
        position_binned_activity_matr_dict[aclu] = a_position_binned_activity_matr
        
    # output: split_spikes_df_dict
    return position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map)





# ==================================================================================================================== #
# 2024-01-29 - Ideal Pho Plotting Interface - UNFINISHED                                                               #
# ==================================================================================================================== #
def map_dataframe_to_plot(df: pd.DataFrame, **kwargs):
    """ 2024-01-29 - My ideal desired function that allows the user to map any column in a dataframe to a plot command, including rows/columns.
    Not yet finished.
     maps any column in the dataframe to a property in a plot. 
     
     Usage:
         fully_resolved_kwargs = map_dataframe_to_plot(df=all_sessions_laps_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size') # , title=f"Laps - {laps_title_string_suffix}"
        fully_resolved_kwargs

    """
    all_column_names: List[str] = list(df.columns)
    all_kwargs_keys: List[str] = list(kwargs.keys())
    all_kwargs_values: List[Union[str, Any]] = list(kwargs.values()) # expected to be either a column name to map or a literal.
    num_rows: int = len(df)
    
    should_fully_extract_dataframe_values: bool = True # if True, extracts the values from the dataframe as an array
    fully_resolved_kwargs = {}
    
    # for a_key in all_kwargs_keys:
    # 	assert a_key in df.columns, f'key "{a_key}" specified in kwargs is not a column in df! \n\tdf.columns: {list(df.columns)}'
    known_keys = ['x', 'y', 'color', 'size', 'row', 'column', 'page', 'xlabel', 'ylabel', 'title']
    for a_key, a_value in kwargs.items():
        if a_key not in known_keys:
            print(f'WARN: key "{a_key}" is not in the known keys list: known_keys: {known_keys}')
        if not isinstance(a_value, str):
            # not a string
            raise ValueError(f"value {a_value} is not a string and its length is not equal to the length of the dataframe.")
            #TODO 2024-01-29 23:45: - [ ] Allow passing literal list-like values with the correct length to be passed directly
            assert (len(a_value) == num_rows), f"(len(a_value) == num_rows) but (len(a_value): {len(a_value)} == num_rows: {num_rows})"
            fully_resolved_kwargs[a_key] = a_value # Set the passed value directly
            
        else:
            # it is a string, assume that it's a column in the dataframe
            assert a_value in all_column_names, f'key:value pair <"{a_key}":"{a_value}"> specified in kwargs has a value that is not a valid column in df! \n\tspecified_value: {a_value}\n\tdf.columns: {list(df.columns)}'
            if should_fully_extract_dataframe_values:
                fully_resolved_kwargs[a_key] = df[a_value].to_numpy()
            else:
                # leave as the validated column name
                fully_resolved_kwargs[a_key] = a_value
                
    return fully_resolved_kwargs


def _embed_in_subplots(scatter_fig):
    import plotly.subplots as sp
    import plotly.graph_objs as go
    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01)

    # adding first histogram
    # Calculate the histogram data
    hist1, bins1 = np.histogram(X[:split], bins='auto')

    # Adding the first histogram as a bar graph and making x negative
    fig.add_trace(
        go.Bar(
            x=-bins1[:-1],
            y=hist1,
            marker_color='#EB89B5',
            name='first half',
            orientation='h',
        ),
        row=1, col=1
    )


    # adding scatter plot
    fig.add_trace(scatter_fig, row=1, col=2)
    # fig.add_trace(
    #     go.Scatter(
    #         x=X,
    #         y=Y,
    #         mode='markers',
    #         marker_color='rgba(152, 0, 0, .8)',
    #     ),
    #     row=1, col=2
    # )

    # adding the second histogram

    # Calculate the histogram data for second half
    hist2, bins2 = np.histogram(X[split:], bins='auto')

    # Adding the second histogram
    fig.add_trace(
        go.Bar(
            x=bins2[:-1],
            y=hist2,
            marker_color='#330C73',
            name='second half',
            orientation='h',
        ),
        row=1, col=3
    )
    return fig



# def plotly_plot_1D_most_likely_position_comparsions(time_window_centers, xbin, posterior): # , ax=None
#     """ 
#     Analagous to `plot_1D_most_likely_position_comparsions`
#     """
#     import plotly.graph_objects as go
    
#     # Posterior distribution heatmap:
#     assert posterior is not None

#     # print(f'time_window_centers: {time_window_centers}, posterior: {posterior}')
#     # Compute extents
#     xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
#     # Create a heatmap
#     fig = go.Figure(data=go.Heatmap(
#                     z=posterior,
#                     x=time_window_centers,  y=xbin, 
#                     zmin=0, zmax=1,
#                     # colorbar=dict(title='z'),
#                     showscale=False,
#                     colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
#                     hoverongaps = False))

#     # Update layout
#     fig.update_layout(
#         autosize=False,
#         xaxis=dict(type='linear', range=[xmin, xmax]),
#         yaxis=dict(type='linear', range=[ymin, ymax]))

#     return fig

@function_attributes(short_name=None, tags=['plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list, xbin, posterior_list): # , ax=None
    """ 
    Analagous to `plot_1D_most_likely_position_comparsions`
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp
    # Ensure input lists are of the same length
    assert len(time_window_centers_list) == len(posterior_list)

    # Compute layout grid dimensions
    num_rows = len(time_window_centers_list)

    # Create subplots
    fig = sp.make_subplots(rows=num_rows, cols=1)

    for row_idx, (time_window_centers, posterior) in enumerate(zip(time_window_centers_list, posterior_list)):
        # Compute extents
        xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
        # Add heatmap trace to subplot
        fig.add_trace(go.Heatmap(
                        z=posterior,
                        x=time_window_centers,  y=xbin, 
                        zmin=0, zmax=1,
                        # colorbar=dict(title='z'),
                        showscale=False,
                        colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
                        hoverongaps = False),
                      row=row_idx+1, col=1)

        # Update layout for each subplot
        fig.update_xaxes(range=[xmin, xmax], row=row_idx+1, col=1)
        fig.update_yaxes(range=[ymin, ymax], row=row_idx+1, col=1)

    return fig


@function_attributes(short_name=None, tags=['plotly', 'blue_yellow'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plot_blue_yellow_points(a_df, specific_point_list):
    """ Renders a figure containing one or more yellow-blue plots (marginals) for a given hoverred point. Used with Dash app.
    
    specific_point_list: List[Dict] - specific_point_list = [{'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03, 'epoch_idx': 0, 'delta_aligned_start_t': -713.908702568122}]
    """
    time_window_centers_list = []
    posterior_list = []

    # for a_single_epoch_row_idx, a_single_epoch_idx in enumerate(selected_epoch_idxs):
    for a_single_epoch_row_idx, a_single_custom_data_dict in enumerate(specific_point_list):
        # a_single_epoch_idx = selected_epoch_idxs[a_single_epoch_row_idx]
        a_single_epoch_idx: int = int(a_single_custom_data_dict['epoch_idx'])
        a_single_session_name: str = str(a_single_custom_data_dict['session_name'])
        a_single_time_bin_size: float = float(a_single_custom_data_dict['time_bin_size'])
        ## Get the dataframe entries:
        a_single_epoch_df = a_df.copy()
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.epoch_idx == a_single_epoch_idx] ## filter by epoch idx
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.session_name == a_single_session_name] ## filter by session
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.time_bin_size == a_single_time_bin_size] ## filter by time-bin-size	

        posterior = a_single_epoch_df[['P_Long', 'P_Short']].to_numpy().T
        time_window_centers = a_single_epoch_df['delta_aligned_start_t'].to_numpy()
        xbin = np.arange(2)
        time_window_centers_list.append(time_window_centers)
        posterior_list.append(posterior)
        
        # fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers=time_window_centers, xbin=xbin, posterior=posterior)
        # fig.show()
        
    fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list=time_window_centers_list, xbin=xbin, posterior_list=posterior_list)
    return fig

@function_attributes(short_name=None, tags=['Dash', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start: float, latest_delta_aligned_t_end: float):
    """ builds an interactive Across Sessions Dash app
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_dash_app
    
    app = _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end)
    """
    from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
    from dash.dash_table import DataTable, FormatTemplate
    from dash.dash_table.Format import Format, Padding

    import dash_bootstrap_components as dbc
    import pandas as pd
    from pathlib import Path
    # import plotly.express as px
    import plotly.io as pio
    template: str = 'plotly_dark' # set plotl template
    pio.templates.default = template


    ## DATA:    
    options_list = list(final_dfs_dict.keys())
    initial_option = options_list[0]
    initial_dataframe: pd.DataFrame = final_dfs_dict[initial_option].copy()
    unique_sessions: List[str] = initial_dataframe['session_name'].unique().tolist()
    num_unique_sessions: int = initial_dataframe['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    assert 'epoch_idx' in initial_dataframe.columns

    ## Extract the unique time bin sizes:
    time_bin_sizes: List[float] = initial_dataframe['time_bin_size'].unique().tolist()
    num_unique_time_bins: int = initial_dataframe.time_bin_size.nunique(dropna=True)
    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
    enabled_time_bin_sizes = [time_bin_sizes[0], time_bin_sizes[-1]] # [0.03, 0.058, 0.10]

    ## prune to relevent columns:
    all_column_names = [
        ['P_Long', 'P_Short', 'P_LR', 'P_RL'],
        ['delta_aligned_start_t'], # 'lap_idx', 
        ['session_name'],
        ['time_bin_size'],
        ['epoch_idx'],
    ]
    all_column_names_flat = [item for sublist in all_column_names for item in sublist]
    print(f'\tall_column_names_flat: {all_column_names_flat}')
    initial_dataframe = initial_dataframe[all_column_names_flat]

    # Initialize the app
    # app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    # Slate
    
    # # money = FormatTemplate.money(2)
    # percentage = FormatTemplate.percentage(2)
    # # percentage = FormatTemplate.deci
    # column_designators = [
    #     dict(id='a', name='delta_aligned_start_t', type='numeric', format=Format()),
    #     dict(id='a', name='session_name', type='text', format=Format()),
    #     dict(id='a', name='time_bin_size', type='numeric', format=Format(padding=Padding.yes).padding_width(9)),
    #     dict(id='a', name='P_Long', type='numeric', format=dict(specifier='05')),
    #     dict(id='a', name='P_LR', type='numeric', format=dict(specifier='05')),
    # ]

    # App layout
    app.layout = dbc.Container([
        dbc.Row([
                html.Div(children='My Custom App with Data, Graph, and Controls'),
                html.Hr()
        ]),
        dbc.Row([
            dbc.Col(dcc.RadioItems(options=options_list, value=initial_option, id='controls-and-radio-item'), width=3),
            dbc.Col(dcc.Checklist(options=time_bin_sizes, value=enabled_time_bin_sizes, id='time-bin-checkboxes', inline=True), width=3), # Add CheckboxGroup for time_bin_sizes
        ]),
        dbc.Row([
            dbc.Col(DataTable(data=initial_dataframe.to_dict('records'), page_size=16, id='tbl-datatable',
                        # columns=column_designators,
                        columns=[{"name": i, "id": i} for i in initial_dataframe.columns],
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': 'rgb(70, 70, 70)',
                                'color': 'white'
                            },
                            {
                                'if': {'column_editable': True},
                                'backgroundColor': 'rgb(100, 100, 100)',
                                'color': 'white'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(30, 30, 30)',
                            'color': 'white'
                        },
                        row_selectable="multi",
                ) # end DataTable
            , align='stretch', width=3),
            dbc.Col(dcc.Graph(figure={}, id='controls-and-graph', hoverData={'points': [{'customdata': []}]},
                            ), align='end', width=9),
        ]), # end Row
        dbc.Row(dcc.Graph(figure={}, id='selected-yellow-blue-marginals-graph')),
    ]) # end Container

    # Add controls to build the interaction
    @callback(
        Output(component_id='controls-and-graph', component_property='figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_graph(col_chosen, chose_bin_sizes):
        print(f'update_graph(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        # Filter dataframe by chosen bin sizes
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)]
        
        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes
        fig = _helper_build_figure(data_results_df=data_results_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode='separate_row_per_session', title=f"{col_chosen}")        
        # 'delta_aligned_start_t', 'session_name', 'time_bin_size'
        tuples_data = data_results_df[['session_name', 'time_bin_size', 'epoch_idx', 'delta_aligned_start_t']].to_dict(orient='records')
        print(f'tuples_data: {tuples_data}')
        fig.update_traces(customdata=tuples_data)
        fig.update_layout(hovermode='closest') # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        return fig


    @callback(
        Output(component_id='tbl-datatable', component_property='data'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
            Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_datatable(col_chosen, chose_bin_sizes):
        """ captures: final_dfs_dict, all_column_names_flat
        """
        print(f'update_datatable(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        a_df = final_dfs_dict[col_chosen].copy()
        ## prune to relevent columns:
        a_df = a_df[all_column_names_flat]
        # Filter dataframe by chosen bin sizes
        a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        data = a_df.to_dict('records')
        return data

    @callback(
        Output('selected-yellow-blue-marginals-graph', 'figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        Input(component_id='tbl-datatable', component_property='selected_rows'),
        Input(component_id='controls-and-graph', component_property='hoverData'),
        ]
    )
    def get_selected_rows(col_chosen, chose_bin_sizes, indices, hoverred_rows):
        print(f'get_selected_rows(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes}, indices: {indices}, hoverred_rows: {hoverred_rows})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)] # Filter dataframe by chosen bin sizes
        # ## prune to relevent columns:
        data_results_df = data_results_df[all_column_names_flat]
        
        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        # print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes

        print(f'hoverred_rows: {hoverred_rows}')
        # get_selected_rows(col_chosen: AcrossSession_Laps_per-Epoch, chose_bin_sizes: [0.03, 0.1], indices: None, hoverred_rows: {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]})
        # hoverred_rows: 
        hoverred_row_points = hoverred_rows.get('points', [])
        num_hoverred_points: int = len(hoverred_row_points)
        extracted_custom_data = [p['customdata'] for p in hoverred_row_points if (p.get('customdata', None) is not None)] # {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}
        num_custom_data_hoverred_points: int = len(extracted_custom_data)

        print(f'extracted_custom_data: {extracted_custom_data}')
        # {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]}
            # selection empty!

        # a_df = final_dfs_dict[col_chosen].copy()
        # ## prune to relevent columns:
        # a_df = a_df[all_column_names_flat]
        # # Filter dataframe by chosen bin sizes
        # a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        # data = a_df.to_dict('records')
        if (indices is not None) and (len(indices) > 0):
            selected_rows = data_results_df.iloc[indices, :]
            print(f'\tselected_rows: {selected_rows}')
        else:
            print(f'\tselection empty!')
            
        if (extracted_custom_data is not None) and (num_custom_data_hoverred_points > 0):
            # selected_rows = data_results_df.iloc[indices, :]
            print(f'\tnum_custom_data_hoverred_points: {num_custom_data_hoverred_points}')
            fig = plot_blue_yellow_points(a_df=data_results_df.copy(), specific_point_list=extracted_custom_data)
        else:
            print(f'\thoverred points empty!')
            fig = go.Figure()

        return fig

    return app


# ==================================================================================================================== #
# 2024-01-29 - Across Session CSV Import and Plotting                                                                  #
# ==================================================================================================================== #
""" 

from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_across_sessions_scatter_results, plot_histograms, plot_stacked_histograms

"""

import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go


def complete_plotly_figure(data_results_df: pd.DataFrame, out_scatter_fig, histogram_bins:int=25):
    """ 
    Usage:

        histogram_bins: int = 25

        new_laps_fig = complete_plotly_figure(data_results_df=deepcopy(all_sessions_laps_df), out_scatter_fig=fig_laps, histogram_bins=histogram_bins)
        new_laps_fig

    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go

    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')


    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    # X_all = data_results_df['delta_aligned_start_t'].to_numpy()
    # Y_all = data_results_df['P_Long'].to_numpy()

    # X_pre_delta = pre_delta_df['delta_aligned_start_t'].to_numpy()
    # X_post_delta = post_delta_df['delta_aligned_start_t'].to_numpy()

    # Y_pre_delta = pre_delta_df['P_Long'].to_numpy()
    # Y_post_delta = post_delta_df['P_Long'].to_numpy()

    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01, shared_yaxes=True, column_titles=["Pre-delta",f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"])

    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    pre_delta_fig = px.histogram(pre_delta_df, y="P_Long", color="time_bin_size", opacity=0.5, title="Pre-delta", range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay')
    print(f'len(pre_delta_fig.data): {len(pre_delta_fig.data)}')
    # time_bin_sizes
    for a_trace in pre_delta_fig.data:
        fig.add_trace(a_trace, row=1, col=1)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

    # Calculate the histogram data
    # hist1, bins1 = np.histogram(X_pre_delta, bins=histogram_bins)

    # # Adding the first histogram as a bar graph and making x negative
    # fig.add_trace(
    #     # go.Bar(x=bins1[:-1], y=hist1, marker_color='#EB89B5', name='first half', orientation='h', ),
    # 	go.Histogram(y=Y_pre_delta, name='pre-delta', marker_color='#EB89B5'),
    #     row=1, col=1
    # )
    # fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

    # Scatter Plot _______________________________________________________________________________________________________ #
    # adding scatter plot
    for a_trace in out_scatter_fig.data:
        fig.add_trace(a_trace, row=1, col=2)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))


    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    post_delta_fig = px.histogram(post_delta_df, y="P_Long", color="time_bin_size", opacity=0.5, title="Post-delta", range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay')

    for a_trace in post_delta_fig.data:
        fig.add_trace(a_trace, row=1, col=3)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
        
    # Calculate the histogram data for second half
    # hist2, bins2 = np.histogram(X_post_delta, bins=histogram_bins)
    # Adding the second histogram
    # fig.add_trace(
    # 	go.Histogram(y=Y_post_delta, name='post-delta', marker_color='#EB89B5',),
    #     # go.Bar(x=bins2[:-1], y=hist2, marker_color='#330C73', name='second half', orientation='h', ),
    #     row=1, col=3
    # )

    # fig.update_layout(layout_yaxis_range=[0.0, 1.0])
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]), barmode='overlay')
    return fig




def _helper_build_figure(data_results_df: pd.DataFrame, histogram_bins:int=25, earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          **build_fig_kwargs):
    """ factored out of the subfunction in plot_across_sessions_scatter_results
    adds scatterplots as well
    Captures: None 
    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers, LongShortDisplayConfigManager

    barmode='overlay'
    # barmode='stack'
    histogram_kwargs = dict(barmode=barmode)
    # px_histogram_kwargs = dict(nbins=histogram_bins, barmode='stack', opacity=0.5, range_y=[0.0, 1.0])
    scatter_title = build_fig_kwargs.pop('title', None)
    debug_print: bool = build_fig_kwargs.pop('debug_print', False)
    
    # Filter dataframe by chosen bin sizes
    if (enabled_time_bin_sizes is not None) and (len(enabled_time_bin_sizes) > 0):
        print(f'filtering data_results_df to enabled_time_bin_sizes: {enabled_time_bin_sizes}...')
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(enabled_time_bin_sizes)]
        
    data_results_df = deepcopy(data_results_df)
    
    # convert time_bin_sizes column to a string so it isn't colored continuously
    data_results_df["time_bin_size"] = data_results_df["time_bin_size"].astype(str)

    
    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
    
    ## Build KWARGS
    known_main_plot_modes = ['default', 'separate_facet_row_per_session', 'separate_row_per_session']
    assert main_plot_mode in known_main_plot_modes
    print(f'main_plot_mode: {main_plot_mode}')

    enable_histograms: bool = True
    enable_scatter_plot: bool = True
    enable_epoch_shading_shapes: bool = True
    px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0]} #, 'histnorm': 'probability density'
    
    if (main_plot_mode == 'default'):
        # main_plot_mode: str = 'default'
        enable_scatter_plot: bool = False
        num_cols: int = int(enable_scatter_plot) + 2 * int(enable_histograms) # 2 histograms and one scatter
        print(f'num_cols: {num_cols}')
        is_col_included = np.array([enable_histograms, enable_scatter_plot, enable_histograms])
        column_widths = list(np.array([0.1, 0.8, 0.1])[is_col_included])
        column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]
        
        # sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': column_titles}
        sp_make_subplots_kwargs = {'rows': 1, 'cols': num_cols, 'column_widths': column_widths, 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': list(np.array(column_titles)[is_col_included])}
        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'session_name', 'size': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
        
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        # main_plot_mode: str = 'separate_facet_row_per_session'
        raise NotImplementedError(f"DOES NOT WORK")
        sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': ["Pre-delta",f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0],
                            'facet_row': 'session_name', 'facet_row_spacing': 0.04, # 'facet_col_wrap': 2, 'facet_col_spacing': 0.04,
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_histogram_kwargs = {**px_histogram_kwargs,
                                'facet_row': 'session_name', 'facet_row_spacing': 0.04, 'facet_col_wrap': 2, 'facet_col_spacing': 0.04, 'height': (num_unique_sessions*200), 'width': 1024}
        enable_histograms = False
        enable_epoch_shading_shapes = False

    elif (main_plot_mode == 'separate_row_per_session'):
        # main_plot_mode: str = 'separate_row_per_session'
        # , subplot_titles=("Plot 1", "Plot 2")
        # column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]
        column_titles = ["Pre-delta", f"{scatter_title}", "Post-delta"]
        session_titles = [str(v) for v in unique_sessions]
        subplot_titles = []
        for a_row_title in session_titles:
            subplot_titles.extend(["Pre-delta", f"{a_row_title}", "Post-delta"])
        # subplot_titles = [["Pre-delta", f"{a_row_title}", "Post-delta"] for a_row_title in session_titles].flatten()
        
        sp_make_subplots_kwargs = {'rows': num_unique_sessions, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'vertical_spacing': 0.04, 'shared_yaxes': True,
                                    'column_titles': column_titles,
                                    'row_titles': session_titles,
                                    'subplot_titles': subplot_titles,
                                    }
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'range_y': [0.0, 1.0],
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
    else:
        raise ValueError(f'main_plot_mode is not a known mode: main_plot_mode: "{main_plot_mode}", known modes: known_main_plot_modes: {known_main_plot_modes}')
    

    def __sub_subfn_plot_histogram(fig, histogram_data_df, hist_title="Post-delta", row=1, col=3):
        """ captures: px_histogram_kwargs, histogram_kwargs
        
        """
        is_first_item: bool = ((row == 1) and (col == 1))
        a_hist_fig = px.histogram(histogram_data_df, y="P_Long", color="time_bin_size", **px_histogram_kwargs, title=hist_title)

        for a_trace in a_hist_fig.data:
            if debug_print:
                print(f'a_trace.legend: {a_trace.legend}, a_trace.legendgroup: {a_trace.legendgroup}, a_trace.legendgrouptitle: {a_trace.legendgrouptitle}, a_trace.showlegend: {a_trace.showlegend}, a_trace.offsetgroup: {a_trace.offsetgroup}')
            
            if (not is_first_item):
                a_trace.showlegend = False
                
            fig.add_trace(a_trace, row=row, col=col)
            fig.update_layout(yaxis=dict(range=[0.0, 1.0]), **histogram_kwargs)
            

    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]
    # creating subplots
    fig = sp.make_subplots(**sp_make_subplots_kwargs)
    next_subplot_col_idx: int = 1 
    
    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_pre_delta_df: pd.DataFrame = pre_delta_df[pre_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_pre_delta_df, hist_title="Pre-delta", row=row_index, col=histogram_col_idx)
                fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=1)
                                
        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=pre_delta_df, hist_title="Pre-delta", row=1, col=histogram_col_idx)
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column

    # Scatter Plot _______________________________________________________________________________________________________ #
    if enable_scatter_plot:
        scatter_column: int = next_subplot_col_idx # default 2
        
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                is_first_item: bool = ((row_index == 1) and (scatter_column == 1))
                a_session_data_results_df: pd.DataFrame = data_results_df[data_results_df['session_name'] == a_session_name]
                #  fig.add_scatter(x=a_session_data_results_df['delta_aligned_start_t'], y=a_session_data_results_df['P_Long'], row=row_index, col=2, name=a_session_name)
                scatter_fig = px.scatter(a_session_data_results_df, **px_scatter_kwargs, title=f"{a_session_name}")
                for a_trace in scatter_fig.data:
                    if (not is_first_item):
                        a_trace.showlegend = False
    
                    fig.add_trace(a_trace, row=row_index, col=scatter_column)
                    # fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

                fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=row_index, col=scatter_column)
                #  fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
                
            #  fig.update_xaxes(matches='x')
        
        else:
            scatter_fig = px.scatter(data_results_df, **px_scatter_kwargs)

            # for a_trace in scatter_traces:
            for a_trace in scatter_fig.data:
                # a_trace.legend = "legend"
                # a_trace['visible'] = 'legendonly'
                # a_trace['visible'] = 'legendonly' # 'legendonly', # this trace will be hidden initially
                fig.add_trace(a_trace, row=1, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
            
            # Update xaxis properties
            fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=1, col=scatter_column)
            
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column
    # else:
    #     # no scatter
    #     next_subplot_col_idx = next_subplot_col_idx
        

    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx #default 3
        
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_post_delta_df: pd.DataFrame = post_delta_df[post_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_post_delta_df, hist_title="Post-delta", row=row_index, col=histogram_col_idx)                
        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=post_delta_df, hist_title="Post-delta", row=1, col=histogram_col_idx)
        
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column
        
    ## Add the delta indicator:
    if (enable_scatter_plot and enable_epoch_shading_shapes):
        
        t_split: float = 0.0
        #TODO 2024-02-02 04:36: - [ ] Should get the specific session t_start/t_end instead of using the general `earliest_delta_aligned_t_start`
        # _extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end, build_only=True)
        # for a_shape_name, a_shape in _extras_output_dict.items():
        #     if (main_plot_mode == 'separate_row_per_session'):
        #         for a_session_i, a_session_name in enumerate(unique_sessions):    
        #             row_index: int = a_session_i + 1 # 1-indexed
        #             fig.add_shape(a_shape, name=a_shape_name, row=row_index, col=scatter_column)
        #     else:
        #         fig.add_shape(a_shape, name=a_shape_name, row=1, col=scatter_column)
    

        # 2024-02-08 Rewrite of shape plotting using simpler functions _______________________________________________________ #
        t_start: float = earliest_delta_aligned_t_start
        t_end: float = latest_delta_aligned_t_end

        yrange = [0.0, 1.0]
        assert (yrange is not None) and (len(yrange) == 2)
        ymin, ymax = yrange # unpack y-range
        assert (ymin < ymax)

        _extras_output_dict = {}
        ## Get the track configs for the colors:
        long_short_display_config_manager = LongShortDisplayConfigManager()
        # long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        # short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

        # long_epoch_kwargs = dict(fillcolor="blue")
        # short_epoch_kwargs = dict(fillcolor="red")
        long_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.long_epoch_config.mpl_color)
        short_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.short_epoch_config.mpl_color)
        
        # blue_shape = dict(type="rect", xref="x", yref="paper", x0=t_start, y0=ymin, x1=t_split, y1=ymax, opacity=0.5, layer="below", line_width=1, **long_epoch_kwargs) 
        # red_shape = dict(type="rect", xref="x", yref="paper", x0=t_split, y0=ymin, x1=t_end, y1=ymax, opacity=0.5, layer="below", line_width=1, **short_epoch_kwargs)
        # vertical_divider_line = dict(type="line", x0=t_split, y0=ymin, x1=t_split, y1=ymax, line=dict(color="rgba(0,0,0,.25)", width=3, ), )
            
        row_column_kwargs = dict(row='all', col=scatter_column)

        ## new methods
        _extras_output_dict["y_zero_line"] = fig.add_hline(y=0.0, line=dict(color="rgba(0,0,0,.25)", width=3, ), **row_column_kwargs)
        vertical_divider_line = fig.add_vline(x=0.0, line=dict(color="rgba(0,0,0,.25)", width=3, ), **row_column_kwargs)

        # fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)

        blue_shape = fig.add_vrect(x0=t_start, x1=t_split, label=dict(text="Long", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **long_epoch_kwargs, **row_column_kwargs) # , fillcolor="green", opacity=0.25
        red_shape = fig.add_vrect(x0=t_split, x1=t_end, label=dict(text="Short", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **short_epoch_kwargs, **row_column_kwargs)

        _extras_output_dict["long_region"] = blue_shape
        _extras_output_dict["short_region"] = red_shape
        _extras_output_dict["divider_line"] = vertical_divider_line
        



    # Update title and height
    
    
    if (main_plot_mode == 'separate_row_per_session'):
        row_height = 250
        required_figure_height = (num_unique_sessions*row_height)
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        row_height = 200
        required_figure_height = (num_unique_sessions*row_height)
    else:
        required_figure_height = 700
        
    fig.update_layout(title_text=scatter_title, width=2048, height=required_figure_height)
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]), template='plotly_dark')
    # Update y-axis range for all created figures
    fig.update_yaxes(range=[0.0, 1.0])

    # Add a footer
    fig.update_layout(
        legend_title_text='tBin Size',
        # annotations=[
        #     dict(x=0.5, y=-0.15, showarrow=False, text="Footer text here", xref="paper", yref="paper")
        # ],
        # margin=dict(b=140), # increase bottom margin to show the footer
    )
    return fig

    


@function_attributes(short_name=None, tags=['scatter', 'multi-session', 'plot', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_across_sessions_scatter_results(directory: Union[Path, str], concatenated_laps_df: pd.DataFrame, concatenated_ripple_df: pd.DataFrame,
                                          earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          laps_title_prefix: str = f"Laps", ripple_title_prefix: str = f"Ripples",
                                          save_figures=False, figure_save_extension='.png', debug_print=False):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    Unknowingly captured: session_name
    
    - [ ] Truncate each session to their start/end instead of the global x bounds.
    
    
    """
    from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    # import plotly.graph_objs as go
    
    # def _subfn_build_figure(data, **build_fig_kwargs):
    #     return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    
    # def _subfn_build_figure(data_results_df: pd.DataFrame, **build_fig_kwargs):
    #     # return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    #     scatter_title = build_fig_kwargs.pop('title', None) 
    #     return go.Figure(px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title), layout_yaxis_range=[0.0, 1.0])
    
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    if not isinstance(directory, Path):
        directory = Path(directory).resolve()
    assert directory.exists()
    print(f'plot_across_sessions_results(directory: {directory})')
    if save_figures:
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder = Path(directory, 'figures')
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: {file_uri_from_path(figures_folder)}')
    
    # Create an empty list to store the figures
    all_figures = []

    ## delta_t aligned:
    # Create a bubble chart for laps
    laps_num_unique_sessions: int = concatenated_laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    laps_num_unique_time_bins: int = concatenated_laps_df.time_bin_size.nunique(dropna=True)
    laps_title_string_suffix: str = f'{laps_num_unique_sessions} Sessions'
    laps_title: str = f"{laps_title_prefix} - {laps_title_string_suffix}"
    fig_laps = _helper_build_figure(data_results_df=concatenated_laps_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=laps_title)

    # Create a bubble chart for ripples
    ripple_num_unique_sessions: int = concatenated_ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    ripple_num_unique_time_bins: int = concatenated_ripple_df.time_bin_size.nunique(dropna=True)
    ripple_title_string_suffix: str = f'{ripple_num_unique_sessions} Sessions'
    ripple_title: str = f"{ripple_title_prefix} - {ripple_title_string_suffix}"
    fig_ripples = _helper_build_figure(data_results_df=concatenated_ripple_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=ripple_title)

    if save_figures:
        # Save the figures to the 'figures' subfolder
        assert figure_save_extension is not None
        if isinstance(figure_save_extension, str):
             figure_save_extension = [figure_save_extension] # a list containing only this item
        
        print(f'\tsaving figures...')
        for a_fig_save_extension in figure_save_extension:
            if a_fig_save_extension.lower() == '.html':
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_html(a_save_name)
            else:
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_image(a_save_name)
    
            fig_laps_name = Path(figures_folder, f"{laps_title_string_suffix.replace(' ', '-')}_{laps_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_laps_name)}"...')
            a_save_fn(fig_laps, fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_{ripple_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_ripple_name)}"...')
            a_save_fn(fig_ripples, fig_ripple_name)
            

    # Append both figures to the list
    all_figures.append((fig_laps, fig_ripples))
    
    return all_figures


@function_attributes(short_name=None, tags=['histogram', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, column='P_Long', **kwargs) -> None:
    """ plots a set of two histograms in subplots, split at the delta for each session.
    from PendingNotebookCode import plot_histograms
    
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    
    fig = plt.figure(layout=layout, **kwargs) # layout="constrained", 
    ax_dict = fig.subplot_mosaic(
        [
            ["epochs_pre_delta", ".", "epochs_post_delta"],
        ],
        # set the height ratios between the rows
        # height_ratios=[8, 1],
        # height_ratios=[1, 1],
        # set the width ratios between the columns
        # width_ratios=[1, 8, 8, 1],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )

    histogram_kwargs = dict(orientation="horizontal", bins=25)
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    pre_delta_df.hist(ax=ax_dict['epochs_pre_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_pre_delta'].set_title(f'{descriptor_str} - pre-$\Delta$ time bins')

    # plot post-delta histogram
    post_delta_df.hist(ax=ax_dict['epochs_post_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_post_delta'].set_title(f'{descriptor_str} - post-$\Delta$ time bins')
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_histograms', figures=[fig], axes=ax_dict)


@function_attributes(short_name=None, tags=['histogram', 'stacked', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_stacked_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, **kwargs) -> None:
    """ plots a colorful stacked histogram for each of the many time-bin sizes
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    figure_identifier: str = f"{descriptor_str}_PrePostDelta"

    fig = plt.figure(num=figure_identifier, clear=True, figsize=(12, 2), layout=layout, **kwargs) # layout="constrained", 
    fig.suptitle(f'{descriptor_str}')
    
    ax_dict = fig.subplot_mosaic(
        [
            # ["epochs_pre_delta", ".", "epochs_post_delta"],
             ["epochs_pre_delta", "epochs_post_delta"],
        ],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )
    
    histogram_kwargs = dict(orientation="horizontal", bins=25)
    
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    time_bin_sizes: int = pre_delta_df['time_bin_size'].unique()
    
    # plot pre-delta histogram:
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_pre_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_pre_delta'].set_title(f'pre-$\Delta$ time bins')
    ax_dict['epochs_pre_delta'].legend()

    # plot post-delta histogram:
    time_bin_sizes: int = post_delta_df['time_bin_size'].unique()
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_post_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_post_delta'].set_title(f'post-$\Delta$ time bins')
    ax_dict['epochs_post_delta'].legend()
    
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_stacked_histograms', figures=[fig], axes=ax_dict)



# ==================================================================================================================== #
# 2024-01-27 - Across Session CSV Import and Processing                                                                #
# ==================================================================================================================== #
""" 
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import find_csv_files, find_HDF5_files, find_most_recent_files, process_csv_file

"""
def find_csv_files(directory: str, recurrsive: bool=False):
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    

def find_HDF5_files(directory: str):
    directory_path = Path(directory) # Convert string path to a Path object
    return list(directory_path.glob('**/*.h5')) # Return a list of all .h5 files in the directory and its subdirectories


def parse_filename(path: Path, debug_print:bool=False) -> Tuple[datetime, str, str]:
    """ 
    # from the found_session_export_paths, get the most recently exported laps_csv, ripple_csv (by comparing `export_datetime`) for each session (`session_str`)
    a_export_filename: str = "2024-01-12_0420PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv"
    export_datetime = "2024-01-12_0420PM"
    session_str = "kdiba_pin01_one_fet11-01_12-58-54"
    export_file_type = "(laps_marginals_df)" # .csv

    # return laps_csv, ripple_csv
    laps_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv").resolve()
    ripple_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(ripple_marginals_df).csv").resolve()

    """
    filename = path.stem   # Get filename without extension
    decoding_time_bin_size_str = None
    
    pattern = r"(?P<export_datetime_str>.*_\d{2}\d{2}[APMF]{2})-(?P<session_str>.*)-(?P<export_file_type>\(?.+\)?)(?:_tbin-(?P<decoding_time_bin_size_str>[^)]+))"
    match = re.match(pattern, filename)
    
    if match is not None:
        # export_datetime_str, session_str, export_file_type = match.groups()
        export_datetime_str, session_str, export_file_type, decoding_time_bin_size_str = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type'), match.group('decoding_time_bin_size_str')
        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d_%I%M%p")
    else:
        if debug_print:
            print(f'did not match pattern with time.')
        # day_date_only_pattern = r"(.*(?:_\d{2}\d{2}[APMF]{2})?)-(.*)-(\(.+\))"
        day_date_only_pattern = r"(\d{4}-\d{2}-\d{2})-(.*)-(\(?.+\)?)" # 
        day_date_only_match = re.match(day_date_only_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if day_date_only_match is not None:
            export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
            # print(export_datetime_str, session_str, export_file_type)
            # parse the datetime from the export_datetime_str and convert it to datetime object
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        else:
            # Try H5 pattern:
            # matches '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
            day_date_with_variant_suffix_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})_?(?P<variant_suffix>[^-_]*)-(?P<session_str>.+?)_(?P<export_file_type>[A-Za-z_]+)"
            day_date_with_variant_suffix_match = re.match(day_date_with_variant_suffix_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26', 
            if day_date_with_variant_suffix_match is not None:
                export_datetime_str, session_str, export_file_type = day_date_with_variant_suffix_match.group('export_datetime_str'), day_date_with_variant_suffix_match.group('session_str'), day_date_with_variant_suffix_match.group('export_file_type')
                # parse the datetime from the export_datetime_str and convert it to datetime object
                try:
                    export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
                except ValueError as e:
                    print(f'ERR: Could not parse date "{export_datetime_str}" of filename: "{filename}"') # 2024-01-18_GL_t_split_df
                    return None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files
            else:
                print(f'ERR: Could not parse filename: "{filename}"') # 2024-01-18_GL_t_split_df
                return None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        
    if export_file_type[0] == '(' and export_file_type[-1] == ')':
        # Trim the brackets from the file type if they're present:
        export_file_type = export_file_type[1:-1]

    return export_datetime, session_str, export_file_type, decoding_time_bin_size_str


def find_most_recent_files(found_session_export_paths: List[Path], debug_print: bool = False) -> Dict[str, Dict[str, Tuple[Path, datetime]]]:
    """
    Returns a dictionary representing the most recent files for each session type among a list of provided file paths.

    Parameters:
    found_session_export_paths (List[Path]): A list of Paths representing files to be checked.
    debug_print (bool): A flag to trigger debugging print statements within the function. Default is False.

    Returns:
    Dict[str, Dict[str, Tuple[Path, datetime]]]: A nested dictionary where the main keys represent 
    different session types. The inner dictionary's keys represent file types and values are the most recent 
    Path and datetime for this combination of session and file type.
    
    # now sessions is a dictionary where the key is the session_str and the value is another dictionary.
    # This inner dictionary's key is the file type and the value is the most recent path for this combination of session and file type
    # Thus, laps_csv and ripple_csv can be obtained from the dictionary for each session

    """
    # Function 'parse_filename' should be defined in the global scope
    parsed_paths = [(*parse_filename(p), p) for p in found_session_export_paths if (parse_filename(p)[0] is not None)]
    parsed_paths.sort(reverse=True)

    if debug_print:
        print(f'parsed_paths: {parsed_paths}')

    sessions = {}
    for export_datetime, session_str, file_type, path, decoding_time_bin_size_str in parsed_paths:
        if session_str not in sessions:
            sessions[session_str] = {}

        if (file_type not in sessions[session_str]) or (sessions[session_str][file_type][-1] < export_datetime):
            sessions[session_str][file_type] = (path, decoding_time_bin_size_str, export_datetime)
    
    return sessions
    

def process_csv_file(file: str, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
    """ reads the CSV file and adds the 'session_name' column if it is missing. 
    
    """
    df = pd.read_csv(file)
    df['session_name'] = session_name 
    if curr_session_t_delta is not None:
        df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
    return df


@define(slots=False)
class AcrossSessionCSVOutputFormat:
    data_description = ["AcrossSession"]
    epoch_description = ["Laps", "Ripple"]
    granularity_description = ["per-Epoch", "per-TimeBin"]
    
    parts_names = ["export_date", "date_name", "epochs", "granularity"]
    
    def parse_filename(self, a_filename: str):
        if a_filename.endswith('.csv'):
            a_filename = a_filename.removesuffix('.csv') # drop the .csv suffix
        # split on the underscore into the parts
        parts = a_filename.split('_')
        if len(parts) == 4:
            export_date, date_name, epochs, granularity  = parts
        else:
            raise NotImplementedError(f"a_csv_filename: '{a_filename}' expected four parts but got {len(parts)} parts.\n\tparts: {parts}")
        return export_date, date_name, epochs, granularity
    


# ==================================================================================================================== #
# 2024-01-23 - Writes the posteriors out to file                                                                       #
# ==================================================================================================================== #

def save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point, parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'lap', epoch_id: int = 9):
    """ 2024-01-23 - Writes the posteriors out to file 
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import save_posterior

        collapsed_per_lap_epoch_marginal_track_identity_point = laps_marginals_df[['P_Long', 'P_Short']].to_numpy().astype(float)
        collapsed_per_lap_epoch_marginal_dir_point = laps_marginals_df[['P_LR', 'P_RL']].to_numpy().astype(float)

        for epoch_id in np.arange(laps_filter_epochs_decoder_result.num_filter_epochs):
            raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple = save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point,
                                                                                        parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='lap', epoch_id=epoch_id)

    """
    from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image

    assert parent_array_as_image_output_folder.exists()
    
    epoch_id_str = f"{epoch_id_identifier_str}[{epoch_id}]"
    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_raw_marginal.png').resolve()
    img_data = raw_posterior_laps_marginals[epoch_id]['p_x_given_n'].astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
    raw_tuple = save_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_raw, path_raw = raw_tuple

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir.png').resolve()
    img_data = laps_directional_marginals[epoch_id]['p_x_given_n'].astype(float)
    marginal_dir_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_marginal_dir, path_marginal_dir = marginal_dir_tuple

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity.png').resolve()
    img_data = laps_track_identity_marginals[epoch_id]['p_x_given_n'].astype(float)
    marginal_track_identity_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_marginal_track_identity, path_marginal_track_identity = marginal_track_identity_tuple


    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity_point.png').resolve()
    img_data = np.atleast_2d(collapsed_per_lap_epoch_marginal_track_identity_point[epoch_id,:]).T
    marginal_dir_point_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir_point.png').resolve()
    img_data = np.atleast_2d(collapsed_per_lap_epoch_marginal_dir_point[epoch_id,:]).T
    marginal_track_identity_point_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)


    return raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple
    
