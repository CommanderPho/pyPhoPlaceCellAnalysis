# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from matplotlib import cm, pyplot as plt
from neuropy.utils.result_context import IdentifyingContext
from nptyping import NDArray
import attrs
import matplotlib as mpl
import numpy as np
import pandas as pd
from attrs import asdict, astuple, define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes


# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
DecoderName = NewType('DecoderName', str)


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


@define(slots=False, eq=False)
class TrialByTrialActivityWindow:
    """ DockPlanningHelperWindow displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TrialByTrialActivityWindow import TrialByTrialActivityWindow


    _out = TrialByTrialActivityWindow.init_dock_area_builder(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.

    """
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)

    # Plot Convenience Accessors _________________________________________________________________________________________ #
    @property
    def root_render_widget(self):
        return self.ui.root_render_widget


    @function_attributes(short_name=None, tags=['matplotlib', 'trial-to-trial-variability', 'laps'], input_requires=[], output_provides=[], uses=[], used_by=['plot_trial_to_trial_reliability_all_decoders_image_stack'], creation_date='2024-08-29 03:26', related_items=[])
    @classmethod
    def _plot_trial_to_trial_reliability_image_array(cls, active_one_step_decoder, z_scored_tuning_map_matrix, max_num_columns=5, drop_below_threshold=0.0000001, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False):
        """ plots the reliability across laps for each decoder
        
        ## Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_trial_to_trial_reliability_image_array

            directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = deepcopy(a_trial_by_trial_result.directional_active_lap_pf_results_dicts)

            ## first decoder:
            a_decoder_name = 'long_LR'
            active_trial_by_trial_activity_obj = directional_active_lap_pf_results_dicts[a_decoder_name]
            active_z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix # shape (n_epochs, n_neurons, n_pos_bins),
            app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array = plot_trial_to_trial_reliability_image_array(active_one_step_decoder=deepcopy(a_pf2D_dt), z_scored_tuning_map_matrix=active_z_scored_tuning_map_matrix)

        
        """
        from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config
        from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_common import pyqtplot_common_setup
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent, set_small_title
        from neuropy.utils.matplotlib_helpers import _scale_current_placefield_to_acceptable_range, _build_neuron_identity_label # for display_all_pf_2D_pyqtgraph_binned_image_rendering
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        
        
        title_row_fixed_height: int = 10
        
        # Get flat list of images:
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves # (78, 57, 6)	- (n_neurons, n_xbins, n_ybins)
        occupancy = active_one_step_decoder.ratemap.occupancy # (57, 6) - (n_xbins, n_ybins)
        # Need to go from (n_epochs, n_neurons, n_pos_bins) -> (n_neurons, n_xbins, n_ybins)
        images = z_scored_tuning_map_matrix.transpose(1, 2, 0) # (71, 57, 22)
        xbin_edges=active_one_step_decoder.xbin
        ybin_edges=active_one_step_decoder.ybin
        # images=images
        # occupancy=occupancy
        root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'TrialByTrialActivityArray: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget) ## 🚧 TODO: BUG: this makes a new QMainWindow to hold this item, which is inappropriate if it's to be rendered as a child of another control

        pg.setConfigOptions(imageAxisOrder='col-major') # this causes the placefields to be rendered horizontally, like they were in _temp_pyqtplot_plot_image_array

        # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        # cmap = pg.colormap.get('gray','matplotlib') # prepare a linear color map
        cmap = ColormapHelpers.create_transparent_colormap(cmap_name='Reds', lower_bound_alpha=0.01) # prepare a linear color map

        image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
        # image_aspect_ratio, image_width_height_tuple = compute_data_aspect_ratio(x_range, y_range)
        # print(f'image_aspect_ratio: {image_aspect_ratio} - xScale/yScale: {float(image_width_height_tuple.width) / float(image_width_height_tuple.height)}')

        # Compute Images:
        included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
        nMapsToShow = len(included_unit_indicies)

        # Paging Management: Constrain the subplots values to just those that you need
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
        page_idx = 0 # page_idx is zero here because we only have one page:

        plot_data_array = []
        
        img_item_array = []
        other_components_array = []
        plot_array = []

        # ==================================================================================================================== #
        # Header Title                                                                                                        #
        # ==================================================================================================================== #
        
        # Create a title label item
        lblTitle = pg.LabelItem(justify='center')
        lblTitle.setText('TrialByTrialActivity - trial_to_trial_reliability_image_array', size='16pt') # , bold=True

        # Add the title label to the first row, spanning all columns
        root_render_widget.addItem(lblTitle, row=0, col=0, colspan=max_num_columns)  # Adjust colspan based on number of columns
        plots_start_row_idx: int = 1
        # root_render_widget.nextRow()
        


        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            is_first_column = (curr_page_relative_col == 0)
            is_first_row = (curr_page_relative_row == 0)
            is_last_column = (curr_page_relative_col == (page_grid_sizes[page_idx].num_columns-1))
            is_last_row = (curr_page_relative_row == (page_grid_sizes[page_idx].num_rows-1))
            if debug_print:
                print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

            neuron_IDX = curr_included_unit_index
            curr_cell_identifier_string = f'Cell[{neuron_IDX}]'
            curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'

            # # Pre-filter the data:
            image = _scale_current_placefield_to_acceptable_range(np.squeeze(images[a_linear_index,:,:]), occupancy=occupancy, drop_below_threshold=drop_below_threshold)

            # Build the image item:
            img_item = pg.ImageItem(image=image, levels=(0,1))
            #     # Viewbox version:
            #     # vb = layout.addViewBox(lockAspect=False)
            #     # # Build the ImageItem (which I'm guessing is like pg.ImageView) to add the image
            #     # imv = pg.ImageItem() # Create it with the current image
            #     # vb.addItem(imv) # add the item to the view box: why do we need the wrapping view box?
            #     # vb.autoRange()
            
            formatted_title: str = cls.build_formatted_title_string(title=curr_cell_identifier_string)   
            
            # # plot mode:
            curr_plot: pg.PlotItem = root_render_widget.addPlot(row=(curr_row + plots_start_row_idx), col=curr_col, title=formatted_title) # , name=curr_plot_identifier_string 
            curr_plot.setObjectName(curr_plot_identifier_string)
            curr_plot.showAxes(False)
            curr_plot.setDefaultPadding(0.0)  # plot without padding data range

            # Set the plot title:
            curr_plot.setTitle(formatted_title)    
            set_small_title(curr_plot, title_row_fixed_height)
            curr_plot.setMouseEnabled(x=False, y=False)
            ## Common formatting:    
        
            if is_last_row:
                curr_plot.showAxes('x', True)
                curr_plot.showAxis('bottom', show=True)
            else:
                curr_plot.showAxes('x', False)
                curr_plot.showAxis('bottom', show=False)
                
            if is_first_column:
                curr_plot.showAxes('y', True)
                curr_plot.showAxis('left', show=True)
            else:
                curr_plot.showAxes('y', False)
                curr_plot.showAxis('left', show=False)
            
            curr_plot.hideButtons() # Hides the auto-scale button
            curr_plot.addItem(img_item, defaultPadding=0.0)  # add ImageItem to PlotItem

            # Update the image:
            img_item.setImage(image, rect=image_bounds_extent, autoLevels=False) # rect: [x, y, w, h]
            img_item.setOpacity(1.0)  # Set transparency for overlay
            if isinstance(cmap, NDArray):
                img_item.setLookupTable(cmap, update=False)
            else:
                img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
        
            curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
            # Sets only the panning limits:
            curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
            # Link Axes to previous item:
            if a_linear_index > 0:
                prev_plot_item = plot_array[a_linear_index-1]
                curr_plot.setXLink(prev_plot_item)
                curr_plot.setYLink(prev_plot_item)
                
            # Interactive Color Bar:
            if not isinstance(cmap, NDArray):
                bar = pg.ColorBarItem(values=(0, 1), colorMap=cmap, width=5, interactive=False) # prepare interactive color bar
                # Have ColorBarItem control colors of img and appear in 'plot':
                bar.setImageItem(img_item, insert_in=curr_plot)
            else:
                bar = None
                
            img_item_array.append(img_item)
            plot_array.append(curr_plot)
            other_components_array.append({'color_bar':bar}) # note this is a list of Dicts, one for every image
            plot_data_array.append({'image_bounds_extent': image_bounds_extent, 'x_range': x_range, 'y_range': y_range}) # note this is a list of Dicts, one for every image


        # Post images loop:
        enable_show = False
        
        if parent_root_widget is not None:
            if enable_show:
                parent_root_widget.show()

            parent_root_widget.setWindowTitle('TrialByTrialActivity - trial_to_trial_reliability_image_array')       


        ## Hide all colorbars, they aren't needed:
        for i, a_plot_components_dict in enumerate(other_components_array):
            if a_plot_components_dict.get('color_bar', None) is not None:
                a_plot_components_dict['color_bar'].setEnabled(False)
                a_plot_components_dict['color_bar'].hide()

        if other_components_array[0].get('color_bar', None) is not None:
            other_components_array[0]['color_bar'].setEnabled(False)
            other_components_array[0]['color_bar'].hide()


        # ==================================================================================================================== #
        # Footer Label                                                                                                       #
        # ==================================================================================================================== #
    
        # Create a label item for the footer
        lblFooter = pg.LabelItem(justify='left')
        lblFooter.setText('Footer Text Here')

        footer_row_idx: int = (curr_row + plots_start_row_idx) + 1
        # Add the footer label below the plots
        # root_render_widget.addItem(footer, row=2, col=0)
        root_render_widget.addItem(lblFooter, row=footer_row_idx, col=0, colspan=max_num_columns)

        if not defer_show:
            parent_root_widget.show()
            
                    
        return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, (lblTitle, lblFooter)
    
    
    @classmethod
    def build_formatted_title_string(cls, title: str) -> str:
        return f"<span style = 'font-size : 12px;' >{title}</span>"
    

    @function_attributes(short_name=None, tags=['reliability', 'decoders', 'all', 'pyqtgraph', 'display', 'figure'], input_requires=[], output_provides=[], uses=['plot_trial_to_trial_reliability_image_array', 'create_transparent_colormap'], used_by=[], creation_date='2024-08-29 04:34', related_items=[])
    @classmethod
    def plot_trial_to_trial_reliability_all_decoders_image_stack(cls, directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity], active_one_step_decoder, drop_below_threshold=0.0000001, is_overlaid_heatmaps_mode: bool = True,
                                                                  app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False, name:str = 'TrialByTrialActivityWindow',
                                                                   **param_kwargs):
        """ Calls `plot_trial_to_trial_reliability_image_array` for each decoder's reliability from lap-top-lap, overlaying the results as different color heatmaps
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_trial_to_trial_reliability_all_decoders_image_stack
        
            directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = deepcopy(a_trial_by_trial_result.directional_active_lap_pf_results_dicts)
            drop_below_threshold = 0.0000001
            app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, additional_img_items_dict, legend_layout = plot_trial_to_trial_reliability_all_decoders_image_stack(directional_active_lap_pf_results_dicts=directional_active_lap_pf_results_dicts, active_one_step_decoder=deepcopy(a_pf2D_dt), drop_below_threshold=drop_below_threshold)


        """
        from neuropy.utils.matplotlib_helpers import _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range, _build_neuron_identity_label # for display_all_pf_2D_pyqtgraph_binned_image_rendering
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, long_short_display_config_manager, apply_LR_to_RL_adjustment
        from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter, debug_print_color, build_adjusted_color
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
        

        
        
        ## Usage:
        
        directional_active_lap_pf_results_dicts = {k:v for k, v in directional_active_lap_pf_results_dicts.items() if k in TrackTemplates.get_decoder_names()}

        if is_overlaid_heatmaps_mode:
            ## first decoder:
            a_decoder_name = 'long_LR'
            active_trial_by_trial_activity_obj = directional_active_lap_pf_results_dicts[a_decoder_name]
            active_z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix # shape (n_epochs, n_neurons, n_pos_bins),
            print(f'np.shape(active_z_scored_tuning_map_matrix): {np.shape(active_z_scored_tuning_map_matrix)}')
        

        else:
            ## MODE 2: Concatenates them before passing them along        
            # active_z_scored_tuning_map_matrix = [[directional_active_lap_pf_results_dicts['long_LR'].z_scored_tuning_map_matrix, directional_active_lap_pf_results_dicts['short_LR'].z_scored_tuning_map_matrix],
            #                                     [directional_active_lap_pf_results_dicts['long_RL'].z_scored_tuning_map_matrix, directional_active_lap_pf_results_dicts['short_RL'].z_scored_tuning_map_matrix]]
            active_z_scored_tuning_map_matrix = np.vstack([np.vstack([directional_active_lap_pf_results_dicts['long_LR'].z_scored_tuning_map_matrix, directional_active_lap_pf_results_dicts['long_RL'].z_scored_tuning_map_matrix]),
                                                 np.vstack([directional_active_lap_pf_results_dicts['short_LR'].z_scored_tuning_map_matrix, directional_active_lap_pf_results_dicts['short_RL'].z_scored_tuning_map_matrix]),
            ])
            # active_z_scored_tuning_map_matrix = np.concatenate(active_z_scored_tuning_map_matrix)

        # Plots only the first data-series ('long_LR')
        app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, (lblTitle, lblFooter) = cls._plot_trial_to_trial_reliability_image_array(active_one_step_decoder=active_one_step_decoder, z_scored_tuning_map_matrix=active_z_scored_tuning_map_matrix, drop_below_threshold=drop_below_threshold)
        additional_heatmaps_data = {}
        additional_img_items_dict = {}
        
        # Extract the heatmaps from the other decoders
        ## INPUTS: directional_active_lap_pf_results_dicts

        # MATPLOTLIB way
        # additional_cmap_names['long_LR'] = 'Reds'
        # additional_cmap_names['long_RL'] = 'Purples'
        # additional_cmap_names['short_LR'] = 'Greens'
        # additional_cmap_names['short_RL'] = 'Oranges'
        # additional_cmap_names['maze_all'] = 'Greys'
        # additional_cmaps = {k: create_transparent_colormap(cmap_name=v, lower_bound_alpha=0.1) for k, v in additional_cmap_names.items()}

        # additional_cmap_names = dict(zip(TrackTemplates.get_decoder_names(), ['red', 'purple', 'green', 'orange'])) # {'long_LR': 'red', 'long_RL': 'purple', 'short_LR': 'green', 'short_RL': 'orange'}
        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

        color_dict = {'long_LR': long_epoch_config['brush'].color(), 'long_RL': apply_LR_to_RL_adjustment(long_epoch_config['brush'].color()),
                        'short_LR': short_epoch_config['brush'].color(), 'short_RL': apply_LR_to_RL_adjustment(short_epoch_config['brush'].color())}
        additional_cmap_names = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

        ## new
        additional_cmap_names = {'long_LR': 'royalblue', 'long_RL': 'blue',
                        'short_LR': 'crimson', 'short_RL': 'red'}
        # additional_cmap_names = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

        # plot_trial_to_trial_reliability_all_decoders_image_stack

        additional_cmaps = {k: ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1) for k, v in additional_cmap_names.items()}

        # additional_cmaps = {name: pg.ColorMap(np.array([0.0, 1.0]), np.array([pg.mkColor(color).getRgb()[:3] + (0,), pg.mkColor(color).getRgb()[:3] + (255,)], dtype=np.ubyte)) for name, color in additional_cmap_names.items()}

        additional_legend_entries = list(zip(directional_active_lap_pf_results_dicts.keys(), additional_cmap_names.values() )) # ['red', 'purple', 'green', 'orange']

        if is_overlaid_heatmaps_mode:
            enable_stacked_long_and_short: bool = False # not currently working, they have to be overlayed exactly on top of each other
            additional_decoder_y_offsets = {'long_LR': 0, 'long_RL': 0,
                            'short_LR': 1, 'short_RL': 1}
            
            for decoder_name, active_trial_by_trial_activity_obj in directional_active_lap_pf_results_dicts.items():  # Replace with actual decoder names
                if decoder_name != 'long_LR':
                    ## we already did 'long_LR', so skip that one    
                    # additional_heatmaps.append(active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix.transpose(1, 2, 0))
                    additional_heatmaps_data[decoder_name] = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix.transpose(1, 2, 0)
                    # additional_cmaps[decoder_name] = pg.colormap.get('gray','matplotlib') # prepare a linear color map


            # Overlay additional heatmaps if provided
            ## INPUTS: additional_heatmaps, additional_cmaps, plot_array
            ## UPDATES: plot_array


            if additional_heatmaps_data:
                for i, (decoder_name, heatmap_matrix) in enumerate(additional_heatmaps_data.items()):
                    if decoder_name not in additional_img_items_dict:
                        additional_img_items_dict[decoder_name] = []
                    cmap = additional_cmaps[decoder_name]
                    # Assuming heatmap_matrix is of shape (n_neurons, n_xbins, n_ybins)
                    for a_linear_index in range(len(plot_array)):
                        curr_image_bounds_extent = plot_data_array[a_linear_index]['image_bounds_extent']
                        # print(f'curr_image_bounds_extent[{a_linear_index}]: {curr_image_bounds_extent}')
                        additional_image = np.squeeze(heatmap_matrix[a_linear_index, :, :])
                        # additional_image = _scale_current_placefield_to_acceptable_range(additional_image, occupancy=occupancy, drop_below_threshold=drop_below_threshold)
                        additional_image = _scale_current_placefield_to_acceptable_range(additional_image, occupancy=None, drop_below_threshold=None) # , occupancy=occupancy, drop_below_threshold=drop_below_threshold !! occupancy is not correct,it's the global one I think
                        # print(f'\tadditional_image: {np.shape(additional_image)}')
                        additional_img_item = pg.ImageItem(image=additional_image, levels=(0, 1))
                        # Update the image:
                        # additional_img_item.setImage(additional_image, autoLevels=False) # rect: [x, y, w, h] , rect=image_bounds_extent
                        shifted_curr_image_bounds_extent = deepcopy(curr_image_bounds_extent)

                        if enable_stacked_long_and_short:
                            curr_item_y_offset = additional_decoder_y_offsets.get(decoder_name, 0)
                            shifted_curr_image_bounds_extent[1] = curr_image_bounds_extent[1] + (curr_image_bounds_extent[3] * (curr_item_y_offset + 1)) # offset y = y + (h * (curr_item_y_offset + 1))
                            shifted_curr_image_bounds_extent[3] = curr_image_bounds_extent[3] + (curr_image_bounds_extent[3] * curr_item_y_offset) # increase h = h + (h * (curr_item_y_offset))
                        else:
                            pass # do nothing, use the same bounds for each image

                        additional_img_item.setImage(additional_image, rect=shifted_curr_image_bounds_extent, autoLevels=False) # rect: [x, y, w, h] 
                        additional_img_item.setOpacity(0.5)  # Set transparency for overlay
                        if isinstance(cmap, NDArray):
                            additional_img_item.setLookupTable(cmap, update=False)
                        else:
                            additional_img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
                            
                        plot_array[a_linear_index].addItem(additional_img_item)
                        additional_img_items_dict[decoder_name].append(additional_img_item)

            ## Add the legend below all the rows:
            root_render_widget.nextRow()
            # Create a layout for the legend at the new row
            # Add a layout for the legend at the bottom, spanning all columns
            # legend_layout = root_render_widget.addLayout(row=root_render_widget.rowCount(), col=0, colspan=root_render_widget.columnCount())
            legend_layout = root_render_widget.addLayout()  # Automatically places in the next available row

            # Add labels for each entry in the legend
            for i, (label, color) in enumerate(additional_legend_entries):
                legend_text = pg.LabelItem(label, color=color)
                # legend_layout.addItem(legend_text, row=0, col=i)  # Place all labels in a single row
                legend_layout.addItem(legend_text, row=i, col=0)  # Place all labels in a single columns

        else:
            legend_layout = None
            
        # END if is_overlaid_heatmaps_mode                
        parent_root_widget.setWindowTitle('TrialByTrialActivity - trial_to_trial_reliability_all_decoders_image_stack')


        _obj = cls()
        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name,
                                 root_render_widget=root_render_widget,
                                 plot_array=plot_array,
                                 legend_layout=legend_layout,
                                 other_components_array=other_components_array,
                                 img_item_array=img_item_array,
                                 additional_img_items_dict=additional_img_items_dict) # , ctrl_widgets={'slider': slider}
        _obj.plots_data = RenderPlotsData(name=name, 
                                          plot_data_array=plot_data_array,
                                            color_dict=color_dict,
                                            # **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']},
                                            )
        _obj.ui = PhoUIContainer(name=name, app=app, root_render_widget=root_render_widget, parent_root_widget=parent_root_widget,
                                 lblTitle=lblTitle, lblFooter=lblFooter, controlled_references=None) # , **utility_controls_ui_dict, **info_labels_widgets_dict
        _obj.params = VisualizationParameters(name=name, use_plaintext_title=False, **param_kwargs)


        # return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, additional_img_items_dict, legend_layout
        return _obj