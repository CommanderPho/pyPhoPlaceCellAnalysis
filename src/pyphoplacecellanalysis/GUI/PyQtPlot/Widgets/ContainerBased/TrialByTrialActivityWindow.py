# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from matplotlib import cm, pyplot as plt
from neuropy.utils.result_context import IdentifyingContext
import nptyping as ND
from nptyping import NDArray
import attrs
import matplotlib as mpl
import numpy as np
import pandas as pd
from attrs import asdict, astuple, define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.assertion_helpers import Assert

# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
DecoderName = NewType('DecoderName', str)


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.External.pyqtgraph_extensions.PlotItem.SelectablePlotItem import SelectablePlotItem
from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.LabelItem.ClickableLabelItem import SelectableLabelItem

@define(slots=False, eq=False)
class TrialByTrialActivityWindow:
    """ TrialByTrialActivityWindow 
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TrialByTrialActivityWindow import TrialByTrialActivityWindow


    _out = TrialByTrialActivityWindow.init_dock_area_builder(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.

    """
    is_publication_ready_figure: bool = field(default=True)
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)

    # Plot Convenience Accessors _________________________________________________________________________________________ #
    @property
    def root_render_widget(self) -> pg.GraphicsLayoutWidget:
        return self.ui.root_render_widget

    @property
    def plot_array(self) -> List[SelectablePlotItem]:
        return self.plots.plot_array

    # def __attrs_post_init__(self):
    #     ## add selection changed callbacks
    #     self.build_internal_callbacks()
    

    # ==================================================================================================================== #
    # Class Methods                                                                                                        #
    # ==================================================================================================================== #
    
    @classmethod
    def build_formatted_title_string(cls, title: str, is_publication_ready_figure: bool = False) -> str:
        """ returns the title of the entire plot
        """
        if is_publication_ready_figure:
            return f"<span style = 'font-family: Arial; font-size : 10pt;' >{title}</span>"
        else:
            return f"<span style = 'font-size : 12px;' >{title}</span>"
    

    @classmethod
    def perform_build_single_cell_formatted_descriptor_string(cls, active_one_step_decoder, aclu, is_publication_ready_figure: bool = False) -> str:
        """ Builds a formatted title for each cell, like "aclu: 19, (shank 2, cluster 22)"
        
        cls.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=override_active_one_step_decoder, aclu=aclu, is_publication_ready_figure=is_publication_ready_figure)
        """
        # neuron_i: int = list(self.plots_data.active_one_step_decoder.included_neuron_IDs).index(aclu)
        if is_publication_ready_figure:
            # For publication figures: just "Cell ID: {aclu}" with Arial font and 9pt size
            final_title_str: str = f"<span style = 'font-family: Arial; font-size : 9pt;' >Cell ID: {aclu}</span>"
        else:
            # Original formatting with extended info
            curr_extended_id_string: str = active_one_step_decoder.ratemap.get_extended_neuron_id_string(neuron_id=aclu) # 2025-01-16 05:42  -- AssertionError: neuron_id: 16 is not in self.neuron_ids: [2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 43, 44, 47, 48, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 95, 98, 101, 102, 103, 104]
            # final_title_str: str = f"aclu: {aclu}: {curr_extended_id_string}" # _build_neuron_identity_label(neuron_extended_id=curr_extended_id_string, brev_mode=None, formatted_max_value_string=None, use_special_overlayed_title=True)
            final_title_str: str = f"aclu: <span style = 'font-size : 14px;' >{aclu}</span>:\n<span style = 'font-size : 11px;' >{curr_extended_id_string}</span>"
        return final_title_str
    

    @function_attributes(short_name=None, tags=['matplotlib', 'trial-to-trial-variability', 'laps'], input_requires=[], output_provides=[], uses=[], used_by=['plot_trial_to_trial_reliability_all_decoders_image_stack'], creation_date='2024-08-29 03:26', related_items=[])
    @classmethod
    def _plot_trial_to_trial_reliability_image_array(cls, active_one_step_decoder, z_scored_tuning_map_matrix, active_neuron_IDs=None, max_num_columns=5, drop_below_threshold=0.0000001, cmap=None, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False, is_publication_ready_figure: bool = False):
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
        occupancy = deepcopy(active_one_step_decoder.ratemap.occupancy) # (57, 6) - (n_xbins, n_ybins)
        # occupancy = None # to match the others
        assert (np.sum(occupancy) > 0.0), f"occupancy is zero for the passed `active_one_step_decoder`. Did you pass an uncalculated pf_dt?"
        
        # Need to go from (n_epochs, n_neurons, n_pos_bins) -> (n_neurons, n_xbins, n_ybins)
        n_epochs, n_neurons, n_pos_bins = np.shape(z_scored_tuning_map_matrix)
        images = z_scored_tuning_map_matrix.transpose(1, 2, 0) # (71, 57, 22)
        xbin_edges=active_one_step_decoder.xbin
        assert (len(xbin_edges)-1) == n_pos_bins, f"n_pos_bins: {n_pos_bins}, len(xbin_edges): {len(xbin_edges)} "
        # ybin_edges=active_one_step_decoder.ybin
        ybin_edges = np.arange(n_epochs+1) - 0.5 # correct ybin_edges are n_epochs
        root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'TrialByTrialActivityArray: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget) ## 🚧 TODO: BUG: this makes a new QMainWindow to hold this item, which is inappropriate if it's to be rendered as a child of another control

        pg.setConfigOptions(imageAxisOrder='col-major') # this causes the placefields to be rendered horizontally, like they were in _temp_pyqtplot_plot_image_array

        if cmap is None:
            # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
            # cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
            # cmap = pg.colormap.get('gray','matplotlib') # prepare a linear color map
            print(f'WARNING: no colormap provided for first decoder. Falling back to "Reds".')
            cmap = ColormapHelpers.create_transparent_colormap(cmap_name='Reds', lower_bound_alpha=0.01, should_return_LinearSegmentedColormap=False) # prepare a linear color map


        image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges=xbin_edges, ybin_edges=ybin_edges, margin=2.0, debug_print=debug_print)
        # image_aspect_ratio, image_width_height_tuple = compute_data_aspect_ratio(x_range, y_range)
        # print(f'image_aspect_ratio: {image_aspect_ratio} - xScale/yScale: {float(image_width_height_tuple.width) / float(image_width_height_tuple.height)}')

        # Compute Images:
        has_active_neuron_IDs: bool = False
        if active_neuron_IDs is not None:
            assert (len(active_neuron_IDs) == np.shape(images)[0]), f"np.shape(images)[0]: {np.shape(images)[0]} should equal len(active_neuron_IDs): {len(active_neuron_IDs)}\nactive_neuron_IDs: {active_neuron_IDs}"
            included_unit_indicies = np.squeeze(np.array(active_neuron_IDs))
            has_active_neuron_IDs = True
        else:
            print(f'WARNING: no active_neuron_IDs provided!')
            included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
            
        nMapsToShow: int = len(included_unit_indicies)

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
        

        ## This page only:
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

            _curr_plot_data_dict = {'a_linear_index': a_linear_index,
             'curr_page_relative_row': curr_page_relative_row, 'curr_page_relative_col': curr_page_relative_col,
            }
            
            if (not has_active_neuron_IDs):
                neuron_IDX = curr_included_unit_index
                curr_cell_identifier_string = f'Cell[{neuron_IDX}]'
                _curr_plot_data_dict['neuron_IDX'] = neuron_IDX
                _curr_plot_data_dict['neuron_aclu'] = None
            else:
                ## `has_active_neuron_IDs`
                neuron_aclu = curr_included_unit_index
                # curr_cell_identifier_string = f'Cell[{neuron_aclu}]'
                curr_cell_identifier_string = cls.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=active_one_step_decoder, aclu=neuron_aclu, is_publication_ready_figure=is_publication_ready_figure)
                _curr_plot_data_dict['neuron_IDX'] = None
                _curr_plot_data_dict['neuron_aclu'] = neuron_aclu

            _curr_plot_data_dict['curr_cell_identifier_string'] = curr_cell_identifier_string
            curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'
            _curr_plot_data_dict['curr_plot_identifier_string'] = curr_plot_identifier_string
            # # Pre-filter the data:
            image = _scale_current_placefield_to_acceptable_range(np.squeeze(images[a_linear_index,:,:]), occupancy=occupancy, drop_below_threshold=drop_below_threshold)

            # Build the image item:
            img_item = pg.ImageItem(image=image, levels=(0,1))
            
            formatted_title: str = cls.build_formatted_title_string(title=curr_cell_identifier_string, is_publication_ready_figure=is_publication_ready_figure)   
            _curr_plot_data_dict['formatted_title'] = formatted_title
            

            # # plot mode:
            curr_plot: SelectablePlotItem = SelectablePlotItem(title=formatted_title, is_selected=False)
            root_render_widget.addItem(curr_plot, row=(curr_row + plots_start_row_idx), col=curr_col)            
            curr_plot.setObjectName(curr_plot_identifier_string)
            # curr_plot.showAxes(False)
            if is_publication_ready_figure:
                curr_plot.showAxes(False)
            else:
                curr_plot.showAxes(True)
            curr_plot.setDefaultPadding(0.0)  # plot without padding data range

            # Set the plot title:
            curr_plot.setTitle(formatted_title)    
            set_small_title(curr_plot, title_row_fixed_height) ## title set to a constant height here
            curr_plot.setMouseEnabled(x=False, y=False)
            ## Common formatting:    
        
            if not is_publication_ready_figure:
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
            else:
                # Publication figure formatting
                # Hide y-axis labels completely
                curr_plot.getAxis('left').setStyle(showValues=False)
                curr_plot.getAxis('left').setLabel('')
                curr_plot.getAxis('left').showLabel(False)
                curr_plot.getAxis('right').setStyle(showValues=False)
                curr_plot.getAxis('right').setLabel('')
                curr_plot.getAxis('right').showLabel(False)
                
                # Hide x-axis labels as well
                curr_plot.getAxis('bottom').setLabel('')
                curr_plot.getAxis('bottom').showLabel(False)
                curr_plot.getAxis('top').setLabel('')
                curr_plot.getAxis('top').showLabel(False)
                
                # Set x-axis to show only ~10 major ticks
                x_range_span = x_range[1] - x_range[0]
                x_tick_spacing = x_range_span / 10.0  # Approximately 10 major ticks
                curr_plot.getAxis('bottom').setTickSpacing(major=x_tick_spacing, minor=x_tick_spacing/5)
            
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
                
            # Add horizontal grid lines for publication figures
            if is_publication_ready_figure:
                # Add efficient horizontal grid lines using GridItem
                grid_item = pg.GridItem()
                grid_item.setTickSpacing(x=[], y=[1.0])  # Only horizontal lines, spaced by 1 unit
                
                # Create pen for grid lines
                pen = pg.mkPen(color='darkgray', width=1.0, style=pg.QtCore.Qt.SolidLine)
                pen.setCosmetic(True)  # Ensures consistent width regardless of zoom
                grid_item.setPen(pen)
                grid_item.setTextPen(None)  # disable text
                
                # Try to disable any background fill/brush
                transparent_brush = pg.mkBrush(None)  # No brush/fill
                if hasattr(grid_item, 'setBrush'):
                    grid_item.setBrush(transparent_brush)
                if hasattr(grid_item, 'setBackgroundBrush'):
                    grid_item.setBackgroundBrush(transparent_brush)
                
                # Set opacity to see if that helps
                grid_item.setOpacity(1.0)
                
                curr_plot.addItem(grid_item)
                
            img_item_array.append(img_item)
            plot_array.append(curr_plot)
            other_components_array.append({'color_bar':bar}) # note this is a list of Dicts, one for every image
            plot_data_array.append({'image_bounds_extent': deepcopy(image_bounds_extent), 'x_range': deepcopy(x_range), 'y_range': deepcopy(y_range)} | _curr_plot_data_dict) # note this is a list of Dicts, one for every image


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
    def _build_hover_preview_y_row_label_items(cls, hover_preview_plot, n_epochs: int, x_range) -> List[pg.TextItem]:
        """Tiny white row-index labels along the y-axis of the hover-preview plot."""
        label_items: List[pg.TextItem] = []
        label_x = x_range[0]
        for row_idx in range(n_epochs):
            label_text = pg.TextItem(html=f"<span style='color:white; font-size:6pt;'>{row_idx}</span>", anchor=(0, 0.5))
            label_text.setPos(label_x, float(row_idx))
            hover_preview_plot.addItem(label_text)
            label_items.append(label_text)
        ## END for row_idx in range(n_epochs)....

        return label_items


    @function_attributes(short_name=None, tags=['reliability', 'decoders', 'all', 'pyqtgraph', 'display', 'figure', 'main'], input_requires=[], output_provides=[], uses=['plot_trial_to_trial_reliability_image_array', 'create_transparent_colormap'], used_by=[], creation_date='2024-08-29 04:34', related_items=[])
    @classmethod
    def plot_trial_to_trial_reliability_all_decoders_image_stack(cls, directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity], active_one_step_decoder, drop_below_threshold=0.0000001,
                                                                  app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False, name:str = 'TrialByTrialActivityWindow',
                                                                  override_active_neuron_IDs=None, is_publication_ready_figure: bool = False,
                                                                   **param_kwargs):
        """ Calls `plot_trial_to_trial_reliability_image_array` for each decoder's reliability from lap-top-lap, overlaying the results as different color heatmaps
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_trial_to_trial_reliability_all_decoders_image_stack
        
            directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = deepcopy(a_trial_by_trial_result.directional_active_lap_pf_results_dicts)
            drop_below_threshold = 0.0000001
            app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, additional_img_items_dict, legend_layout = plot_trial_to_trial_reliability_all_decoders_image_stack(directional_active_lap_pf_results_dicts=directional_active_lap_pf_results_dicts, active_one_step_decoder=deepcopy(a_pf2D_dt), drop_below_threshold=drop_below_threshold)


        """
        from neuropy.utils.matplotlib_helpers import _scale_current_placefield_to_acceptable_range # for display_all_pf_2D_pyqtgraph_binned_image_rendering
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DecoderIdentityColors, long_short_display_config_manager, apply_LR_to_RL_adjustment
        from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter, debug_print_color, build_adjusted_color
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
        
        ## Usage:
        
        directional_active_lap_pf_results_dicts = {k:v for k, v in directional_active_lap_pf_results_dicts.items() if k in TrackTemplates.get_decoder_names()}
        directional_active_pf_neuron_IDS_dict = {k:v.neuron_ids for k, v in directional_active_lap_pf_results_dicts.items()}
        print(f'directional_active_pf_neuron_IDS_dict: {directional_active_pf_neuron_IDS_dict}')
        active_neuron_IDs = deepcopy(list(directional_active_pf_neuron_IDS_dict.values())[0]) ## gets the first aclus:
        assert np.allclose([list(v) for v in list(directional_active_pf_neuron_IDS_dict.values())], active_neuron_IDs), f"All neuron_IDs must be the same!"
        if override_active_neuron_IDs is not None:
            active_neuron_IDs = active_neuron_IDs[np.isin(active_neuron_IDs, override_active_neuron_IDs)] # only get the allowed elements
        
        ## first decoder:
        a_decoder_name = 'long_LR'
        active_trial_by_trial_activity_obj = directional_active_lap_pf_results_dicts[a_decoder_name]
        active_z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix # shape (n_epochs, n_neurons, n_pos_bins),
        print(f'np.shape(active_z_scored_tuning_map_matrix): {np.shape(active_z_scored_tuning_map_matrix)}')

        color_dict: Dict[types.DecoderName, pg.QtGui.QColor] = DecoderIdentityColors.build_decoder_color_dict(wants_hex_str=False)
        additional_cmap_names: Dict[types.DecoderName, str] = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

        ## new
        # additional_cmap_names = {'long_LR': 'royalblue', 'long_RL': 'blue',
        #                 'short_LR': 'crimson', 'short_RL': 'red'}
        
        # additional_cmap_names = {'long_LR': '#0099ff', 'long_RL': '#7a00ff', 'short_LR': '#f51616', 'short_RL': '#e3f516'}

        additional_cmap_names = {'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'}
        # additional_cmap_names = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}

        # plot_trial_to_trial_reliability_all_decoders_image_stack

        additional_cmaps = {k: ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1, should_return_LinearSegmentedColormap=False) for k, v in additional_cmap_names.items()}
        additional_legend_entries = list(zip(directional_active_lap_pf_results_dicts.keys(), additional_cmap_names.values() )) # ['red', 'purple', 'green', 'orange']

        # Plots only the first data-series ('long_LR')
        app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, (lblTitle, lblFooter) = cls._plot_trial_to_trial_reliability_image_array(active_one_step_decoder=active_one_step_decoder, z_scored_tuning_map_matrix=active_z_scored_tuning_map_matrix, active_neuron_IDs=active_neuron_IDs,
                                                                                                                                                     drop_below_threshold=drop_below_threshold, cmap=additional_cmaps['long_LR'], is_publication_ready_figure=is_publication_ready_figure)
        

        occupancy = deepcopy(active_one_step_decoder.ratemap.occupancy)
        # occupancy = None # previous
        
        assert (np.sum(occupancy) > 0.0), f"occupancy is zero for the passed `active_one_step_decoder`. Did you pass an uncalculated pf_dt?"
        
        ## list of image items img_item_array
        
        additional_heatmaps_data = {}
        additional_img_items_dict = {}
        
        # Extract the heatmaps from the other decoders
        ## INPUTS: directional_active_lap_pf_results_dicts

        # enable_stacked_long_and_short: bool = False # not currently working, they have to be overlayed exactly on top of each other
        # additional_decoder_y_offsets = {'long_LR': 0, 'long_RL': 0, 'short_LR': 1, 'short_RL': 1}
        
        for decoder_name, active_trial_by_trial_activity_obj in directional_active_lap_pf_results_dicts.items():  # Replace with actual decoder names
            if decoder_name != 'long_LR':
                ## we already did 'long_LR', so skip that one    
                # additional_heatmaps.append(active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix.transpose(1, 2, 0))
                additional_heatmaps_data[decoder_name] = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix.transpose(1, 2, 0)
                # additional_cmaps[decoder_name] = pg.colormap.get('gray','matplotlib') # prepare a linear color map


        # Overlay additional heatmaps if provided
        ## INPUTS: additional_heatmaps, additional_cmaps, plot_array
        ## UPDATES: plot_array
        for i, (decoder_name, heatmap_matrix) in enumerate(additional_heatmaps_data.items()):
            if decoder_name not in additional_img_items_dict:
                additional_img_items_dict[decoder_name] = []
            cmap = additional_cmaps[decoder_name]
            # Assuming heatmap_matrix is of shape (n_neurons, n_xbins, n_ybins)
            for a_linear_index in range(len(plot_array)):
                curr_image_bounds_extent = plot_data_array[a_linear_index]['image_bounds_extent']
                # print(f'curr_image_bounds_extent[{a_linear_index}]: {curr_image_bounds_extent}')
                additional_image = np.squeeze(heatmap_matrix[a_linear_index, :, :])
                additional_image = _scale_current_placefield_to_acceptable_range(additional_image, occupancy=deepcopy(occupancy), drop_below_threshold=drop_below_threshold)
                # additional_image = _scale_current_placefield_to_acceptable_range(additional_image, occupancy=None, drop_below_threshold=None) # , occupancy=occupancy, drop_below_threshold=drop_below_threshold !! occupancy is not correct,it's the global one I think
                # print(f'\tadditional_image: {np.shape(additional_image)}')
                additional_img_item = pg.ImageItem(image=additional_image, levels=(0, 1))
                # Update the image:
                # additional_img_item.setImage(additional_image, autoLevels=False) # rect: [x, y, w, h] , rect=image_bounds_extent
                shifted_curr_image_bounds_extent = deepcopy(curr_image_bounds_extent)
                # use the same bounds for each image
                additional_img_item.setImage(additional_image, rect=shifted_curr_image_bounds_extent, autoLevels=False) # rect: [x, y, w, h] 
                additional_img_item.setOpacity(1.0)  # Set transparency for pre-separated overlay
                if isinstance(cmap, NDArray):
                    additional_img_item.setLookupTable(cmap, update=False)
                else:
                    additional_img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
                    
                plot_array[a_linear_index].addItem(additional_img_item)
                additional_img_items_dict[decoder_name].append(additional_img_item)

        ## END for i, (decoder_name, heatmap_matrix) in enumerate(addition...


        ## Add the legend below all the rows:
        root_render_widget.nextRow()
        # Create a layout for the legend at the new row
        # Add a layout for the legend at the bottom, spanning all columns
        # legend_layout: pg.GraphicsLayout = root_render_widget.addLayout(row=root_render_widget.rowCount(), col=0, colspan=root_render_widget.columnCount())
        legend_layout: pg.GraphicsLayout = root_render_widget.addLayout(rowspan=2, colspan=1)  # Automatically places in the next available row
        
        # Set compact spacing for publication figures
        if is_publication_ready_figure:
            legend_layout.setSpacing(1)  # Reduce spacing between items
            legend_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        
        legend_entries_dict = {}
        # Add labels for each entry in the legend
        for i, (label, color) in enumerate(additional_legend_entries):
            # legend_text = pg.LabelItem(label, color=color)
            legend_text = SelectableLabelItem(label, color=color, is_selected=True)
            # Set smaller font size for publication figures
            if is_publication_ready_figure:
                legend_text.setText(label, size='8pt')
            legend_entries_dict[label] = legend_text
            # legend_layout.addItem(legend_text, row=0, col=i)  # Place all labels in a single row
            legend_layout.addItem(legend_text, row=i, col=0)  # Place all labels in a single columns
        ## END for i, (label, color) in enumerate(additional_legend_entries)...            

        legend_layout.setMaximumWidth(100)

        additional_img_items_dict['long_LR'] = img_item_array # set first decoder to original image items

        ## Large right-side axes: hover-preview of the hovered subplot (non-publication only)
        plots_start_row_idx: int = 1
        max_num_columns: int = 5
        num_plot_rows: int = max(d['curr_page_relative_row'] for d in plot_data_array) + 1
        n_epochs: int = int(np.shape(active_z_scored_tuning_map_matrix)[0])
        hover_preview_img_items_dict: Dict[types.DecoderName, pg.ImageItem] = {}
        hover_preview_y_row_label_items: List[pg.TextItem] = []
        if not is_publication_ready_figure:
            hover_preview_plot = root_render_widget.addPlot(row=plots_start_row_idx, col=max_num_columns, rowspan=num_plot_rows, colspan=1)
            hover_preview_plot.setDefaultPadding(0.0)
            hover_preview_plot.setTitle('')
            hover_preview_plot.hideButtons()
            hover_preview_plot.setMouseEnabled(x=False, y=False)

            ref_plot_data = plot_data_array[0]
            x_range = ref_plot_data['x_range']
            y_range = ref_plot_data['y_range']
            hover_preview_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
            hover_preview_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])

            for decoder_name in directional_active_lap_pf_results_dicts.keys():
                cmap = additional_cmaps[decoder_name]
                preview_img_item = pg.ImageItem(levels=(0, 1))
                if isinstance(cmap, NDArray):
                    preview_img_item.setLookupTable(cmap, update=False)
                else:
                    preview_img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
                preview_img_item.setOpacity(1.0)
                hover_preview_plot.addItem(preview_img_item)
                hover_preview_img_items_dict[decoder_name] = preview_img_item
            ## END for decoder_name in directional_active_lap_pf_results_dicts.keys()....

            hover_preview_y_row_label_items = cls._build_hover_preview_y_row_label_items(hover_preview_plot=hover_preview_plot, n_epochs=n_epochs, x_range=x_range)

            position_plot = hover_preview_plot  # notebook-compat alias
        else:
            hover_preview_plot = None
            position_plot = None

        parent_root_widget.setWindowTitle('TrialByTrialActivity - trial_to_trial_reliability_all_decoders_image_stack')

        _obj = cls(is_publication_ready_figure=is_publication_ready_figure)
        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name,
                                 root_render_widget=root_render_widget,
                                 plot_array=plot_array,
                                 legend_layout=legend_layout,
                                 legend_entries_dict=legend_entries_dict,
                                 other_components_array=other_components_array,
                                 img_item_array=img_item_array,
                                 additional_img_items_dict=additional_img_items_dict,
                                 position_plot=position_plot,
                                 hover_preview_plot=hover_preview_plot,
                                 hover_preview_img_items_dict=hover_preview_img_items_dict,
                                 hover_preview_y_row_label_items=hover_preview_y_row_label_items,
                                 ) # , ctrl_widgets={'slider': slider} # .plots.additional_img_items_dict
        _obj.plots_data = RenderPlotsData(name=name, 
                                          plot_data_array=plot_data_array,
                                          active_neuron_IDs=deepcopy(active_neuron_IDs),
                                          active_one_step_decoder=deepcopy(active_one_step_decoder),
                                          color_dict=color_dict,
                                            # **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']},
                                            )
        _obj.ui = PhoUIContainer(name=name, app=app, root_render_widget=root_render_widget, parent_root_widget=parent_root_widget,
                                 lblTitle=lblTitle, lblFooter=lblFooter, controlled_references=None) # , **utility_controls_ui_dict, **info_labels_widgets_dict
        _obj.params = VisualizationParameters(name=name, use_plaintext_title=False, hovered_linear_index=None, **param_kwargs)
        _obj.build_internal_callbacks()
        return _obj


    # ==================================================================================================================== #
    # Instance Methods                                                                                                     #
    # ==================================================================================================================== #
    def build_single_cell_formatted_descriptor_string(self, aclu, override_active_one_step_decoder=None) -> str:
        """ Builds a formatted title for each cell, like "aclu: 19, (shank 2, cluster 22)"
        
        self.build_single_cell_formatted_descriptor_string(aclu=neuron_ID, override_active_one_step_decoder=active_one_step_decoder)
        
        """
        if override_active_one_step_decoder is None:
            override_active_one_step_decoder = self.plots_data.active_one_step_decoder
        return self.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=override_active_one_step_decoder, aclu=aclu, is_publication_ready_figure=self.is_publication_ready_figure)


    def build_internal_callbacks(self):
        ## add selection changed callbacks
        for a_linear_index, a_plot_item in enumerate(self.plot_array):
            a_plot_item.sigSelectedChanged.connect(self.on_change_selection)
        ## END for a_linear_index, a_plot_item in enumerate(self.plot_array)....

        for a_decoder_name, a_label_item in self.plots.legend_entries_dict.items():
            a_label_item.sigSelectedChanged.connect(self.on_change_series_legend_selection)
        ## END for a_decoder_name, a_label_item in self.plots.legend_entries_dict.items()....

        hover_preview_plot = getattr(self.plots, 'hover_preview_plot', None)
        if hover_preview_plot is not None:
            self.ui.root_render_widget.scene().sigMouseHover.connect(self.on_scene_mouse_hover)


    @function_attributes(short_name=None, tags=['opacity', 'series'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-12 00:00', related_items=[])
    def set_series_opacity(self, target_decoder_name: types.DecoderName, target_opacity: float = 0.1):
        if 'long_LR' not in self.plots.additional_img_items_dict:
            self.plots.additional_img_items_dict['long_LR'] = self.plots.img_item_array
            
        for an_img_item in self.plots.additional_img_items_dict[target_decoder_name]:
            an_img_item.setOpacity(target_opacity)
        ## END for an_img_item in self.plots.additional_img_items_dict[target_decoder_name]....

        hover_preview_img_items_dict = getattr(self.plots, 'hover_preview_img_items_dict', None) or {}
        if target_decoder_name in hover_preview_img_items_dict:
            hover_preview_img_items_dict[target_decoder_name].setOpacity(target_opacity)


    @function_attributes(short_name=None, tags=['opacity', 'series'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-12 00:00', related_items=[])
    def restore_all_series_opacity(self, override_all_opacity: Optional[float] = None):
        if 'long_LR' not in self.plots.additional_img_items_dict:
            self.plots.additional_img_items_dict['long_LR'] = self.plots.img_item_array
            
        if override_all_opacity is None:
            override_all_opacity = 1.0
            
        for a_decoder_name, an_img_item_arr in self.plots.additional_img_items_dict.items():
            for an_img_item in an_img_item_arr:
                an_img_item.setOpacity(override_all_opacity)
            ## END for an_img_item in an_img_item_arr....
        ## END for a_decoder_name, an_img_item_arr in self.plots.additional_img_items_dict.items()....

        hover_preview_img_items_dict = getattr(self.plots, 'hover_preview_img_items_dict', None) or {}
        for a_decoder_name, preview_img_item in hover_preview_img_items_dict.items():
            preview_img_item.setOpacity(override_all_opacity)
        ## END for a_decoder_name, preview_img_item in hover_preview_img_items_dict.items()....


    def on_change_series_legend_selection(self, a_selectable_label, new_is_selected: bool):
        """ called when one of the aclu subplots selection changes 
        """
        print(f'on_change_series_legend_selection(a_selectable_label: {a_selectable_label}, new_is_selected: {new_is_selected})')
        self.update_all_series_opacities_from_legend()
        # a_decoder_name: str = str(a_selectable_label.text)
        # self.set_series_opacity(target_decoder_name=a_decoder_name, target_opacity=0.1)


    def update_all_series_opacities_from_legend(self):
        """ uses the legend label's selected status to determine the opacity for the data series. """
        for a_decoder_name, an_img_item_arr in self.plots.additional_img_items_dict.items():
            a_label_item = self.plots.legend_entries_dict[a_decoder_name]
            if a_label_item.is_selected:
                curr_desired_opacity: float = 1.0
            else:
                curr_desired_opacity: float = 0.1
            for an_img_item in an_img_item_arr:
                an_img_item.setOpacity(curr_desired_opacity)
            ## END for an_img_item in an_img_item_arr....
        ## END for a_decoder_name, an_img_item_arr in self.plots.additional_img_items_dict.items()....

        ## Keep hover-preview layers in sync with legend selection
        hover_preview_img_items_dict = getattr(self.plots, 'hover_preview_img_items_dict', None) or {}
        for a_decoder_name, preview_img_item in hover_preview_img_items_dict.items():
            a_label_item = self.plots.legend_entries_dict[a_decoder_name]
            preview_desired_opacity: float = 1.0 if a_label_item.is_selected else 0.1
            preview_img_item.setOpacity(preview_desired_opacity)
        ## END for a_decoder_name, preview_img_item in hover_preview_img_items_dict.items()....


    def on_scene_mouse_hover(self, items):
        """Resolve which subplot is under the cursor and update the large preview axes."""
        hover_preview_plot = getattr(self.plots, 'hover_preview_plot', None)
        if hover_preview_plot is None:
            return

        hovered_plot = None
        for item in items:
            p = item
            while p is not None:
                if p is hover_preview_plot:
                    break
                if isinstance(p, SelectablePlotItem) and (p in self.plot_array):
                    hovered_plot = p
                    break
                p = p.parentItem()
            ## END while p is not None....

            if hovered_plot is not None:
                break
        ## END for item in items....

        if hovered_plot is not None:
            a_linear_index: int = self.plot_array.index(hovered_plot)
            self.update_hover_preview(a_linear_index)
    ## END def on_scene_mouse_hover(self, items)...


    @function_attributes(short_name=None, tags=['hover', 'preview', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-08-24 14:50', related_items=[])
    def update_hover_preview(self, a_linear_index: int):
        """Copy the hovered subplot's multi-decoder heatmaps into the large right-side preview axes."""
        hover_preview_plot = getattr(self.plots, 'hover_preview_plot', None)
        hover_preview_img_items_dict = getattr(self.plots, 'hover_preview_img_items_dict', None) or {}
        if (hover_preview_plot is None) or (len(hover_preview_img_items_dict) == 0):
            return

        if getattr(self.params, 'hovered_linear_index', None) == a_linear_index:
            return

        self.params.hovered_linear_index = a_linear_index
        if 'long_LR' not in self.plots.additional_img_items_dict:
            self.plots.additional_img_items_dict['long_LR'] = self.plots.img_item_array

        plot_data = self.plots_data.plot_data_array[a_linear_index]
        image_bounds_extent = plot_data['image_bounds_extent']
        x_range = plot_data['x_range']
        y_range = plot_data['y_range']

        for decoder_name, preview_img_item in hover_preview_img_items_dict.items():
            source_img_item = self.plots.additional_img_items_dict[decoder_name][a_linear_index]
            preview_img_item.setImage(source_img_item.image, rect=image_bounds_extent, autoLevels=False)
            preview_img_item.setOpacity(source_img_item.opacity())
        ## END for decoder_name, preview_img_item in hover_preview_img_items_dict.items()....

        hover_preview_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
        formatted_title = plot_data.get('formatted_title') or plot_data.get('curr_cell_identifier_string', '')
        hover_preview_plot.setTitle(formatted_title)

        neuron_aclu = plot_data.get('neuron_aclu', None)
        if (self.ui.lblFooter is not None) and (neuron_aclu is not None):
            self.ui.lblFooter.setText(f'Hovered: aclu {neuron_aclu}')

        self._update_hover_preview_peak_markers(neuron_aclu)
        self._update_hover_preview_aclu_field_peak_id_labels(neuron_aclu)
    ## END def update_hover_preview(self, a_linear_index: int)...


    def _clear_hover_preview_peak_markers(self):
        """Remove any peak-center vertical markers from the hover-preview axes."""
        hover_preview_plot = self.plots.get('hover_preview_plot', None)
        existing = self.plots.get('hover_preview_peak_center_vertical_markers', None)
        if (hover_preview_plot is not None) and (existing is not None):
            if isinstance(existing, (list, tuple)):
                for a_line in existing:
                    hover_preview_plot.removeItem(a_line)
                ## END for a_line in existing....
            else:
                hover_preview_plot.removeItem(existing)

        self.plots.hover_preview_peak_center_vertical_markers = None



    def _clear_aclu_field_peak_id_debug_labels(self):
        """Remove aclu_field_peak_id debug TextItems from subplots and hover-preview."""
        existing_labels = self.plots.get('aclu_field_peak_id_debug_labels', None)
        if existing_labels is not None:
            aclu_to_plot_idx: Dict[int, int] = {}
            for a_plot_data_dict in self.plots_data.plot_data_array:
                neuron_aclu = a_plot_data_dict.get('neuron_aclu', None)
                if neuron_aclu is not None:
                    aclu_to_plot_idx[int(neuron_aclu)] = int(a_plot_data_dict['a_linear_index'])
            ## END for a_plot_data_dict in self.plots_data.plot_data_array....

            for aclu, label_items in existing_labels.items():
                plot_idx = aclu_to_plot_idx.get(int(aclu), None)
                if plot_idx is None:
                    continue
                curr_plot = self.plots.plot_array[plot_idx]
                if isinstance(label_items, (list, tuple)):
                    for a_label in label_items:
                        curr_plot.removeItem(a_label)
                    ## END for a_label in label_items....
                else:
                    curr_plot.removeItem(label_items)
            ## END for aclu, label_items in existing_labels.items()....

        self.plots.aclu_field_peak_id_debug_labels = None
        self.plots_data.aclu_field_peak_id_color_maps_dict = None

        hover_preview_plot = self.plots.get('hover_preview_plot', None)
        existing_hover = self.plots.get('hover_preview_aclu_field_peak_id_debug_labels', None)
        if (hover_preview_plot is not None) and (existing_hover is not None):
            if isinstance(existing_hover, (list, tuple)):
                for a_label in existing_hover:
                    hover_preview_plot.removeItem(a_label)
                ## END for a_label in existing_hover....
            else:
                hover_preview_plot.removeItem(existing_hover)

        self.plots.hover_preview_aclu_field_peak_id_debug_labels = None



    @classmethod
    def _build_vertical_tick_symbol(cls, trial_half_height: float = 0.45) -> pg.QtGui.QPainterPath:
        """Data-coordinate vertical tick symbol for ``ScatterPlotItem`` (``pxMode=False``, ``size=1``)."""
        path = pg.QtGui.QPainterPath()
        path.moveTo(0.0, -float(trial_half_height))
        path.lineTo(0.0, float(trial_half_height))
        return path



    @classmethod
    def _peak_marker_opacity_for_summit_idx(cls, summit_idx_val: int) -> float:
        """Return marker opacity for a given ``summit_idx`` rank."""
        summit_idx_val = int(summit_idx_val)
        if summit_idx_val <= 0:
            return 1.0
        if summit_idx_val == 1:
            return 0.75
        if summit_idx_val == 2:
            return 0.65
        return 0.5



    @classmethod
    def _peak_marker_pen_for_summit_idx(cls, summit_idx_val: int, base_pen=None):
        """Return a pen with width ``base / 2**summit_idx_val`` and summit-rank opacity."""
        if base_pen is None:
            base_pen = pg.mkPen('w', width=1.5)
        else:
            base_pen = pg.mkPen(base_pen)

        base_pen_width: float = float(base_pen.widthF()) if (base_pen.widthF() > 0.0) else 1.5
        pen_width: float = float(base_pen_width) / float(2 ** int(summit_idx_val))
        pen_opacity: float = cls._peak_marker_opacity_for_summit_idx(summit_idx_val=int(summit_idx_val))
        pen_color = pg.mkColor(base_pen.color())
        pen_color.setAlphaF(float(pen_opacity))
        a_pen = pg.QtGui.QPen(base_pen)
        a_pen.setWidthF(pen_width)
        a_pen.setColor(pen_color)
        return a_pen



    @classmethod
    def _build_peak_marker_scatter(cls, peak_center_x: NDArray, trial_idx: NDArray, pen=None, trial_half_height: float = 0.45) -> Optional[pg.ScatterPlotItem]:
        """One batched scatter of vertical ticks (avoids PlotCurveItem OpenGL LINE_STRIP ignoring connect='pairs')."""
        peak_center_x = np.asarray(peak_center_x, dtype=float).ravel()
        trial_idx = np.asarray(trial_idx, dtype=float).ravel()
        valid_mask = np.isfinite(peak_center_x) & np.isfinite(trial_idx)
        peak_center_x = peak_center_x[valid_mask]
        trial_idx = trial_idx[valid_mask]
        if peak_center_x.size == 0:
            return None

        if pen is None:
            pen = pg.mkPen('w', width=1.5)

        a_scatter = pg.ScatterPlotItem(
            x=peak_center_x,
            y=trial_idx,
            symbol=cls._build_vertical_tick_symbol(trial_half_height=trial_half_height),
            size=1.0,
            pxMode=False,
            pen=pen,
            brush=pg.mkBrush(None),
            hoverable=False,
        )
        a_scatter.setZValue(100)
        return a_scatter



    @classmethod
    def _build_peak_marker_scatter_items(cls, peak_center_x: NDArray, trial_idx: NDArray, pen=None, trial_half_height: float = 0.45, summit_idx: Optional[NDArray] = None) -> List[pg.ScatterPlotItem]:
        """Build one or more batched scatters; groups by ``summit_idx`` when present (one pen per group)."""
        peak_center_x = np.asarray(peak_center_x, dtype=float).ravel()
        trial_idx = np.asarray(trial_idx, dtype=float).ravel()
        summit_idx_arr = None if (summit_idx is None) else np.asarray(summit_idx, dtype=int).ravel()
        valid_mask = np.isfinite(peak_center_x) & np.isfinite(trial_idx)
        peak_center_x = peak_center_x[valid_mask]
        trial_idx = trial_idx[valid_mask]
        if summit_idx_arr is not None:
            summit_idx_arr = summit_idx_arr[valid_mask]
        if peak_center_x.size == 0:
            return []

        if summit_idx_arr is None:
            a_scatter = cls._build_peak_marker_scatter(peak_center_x=peak_center_x, trial_idx=trial_idx, pen=pen, trial_half_height=trial_half_height)
            return [] if (a_scatter is None) else [a_scatter]

        scatter_items: List[pg.ScatterPlotItem] = []
        for a_summit_idx_val in np.unique(summit_idx_arr):
            summit_mask = summit_idx_arr == a_summit_idx_val
            group_pen = cls._peak_marker_pen_for_summit_idx(summit_idx_val=int(a_summit_idx_val), base_pen=pen)
            a_scatter = cls._build_peak_marker_scatter(peak_center_x=peak_center_x[summit_mask], trial_idx=trial_idx[summit_mask], pen=group_pen, trial_half_height=trial_half_height)
            if a_scatter is not None:
                scatter_items.append(a_scatter)
        ## END for a_summit_idx_val in np.unique(summit_idx_arr)...

        return scatter_items



    @classmethod
    def _build_aclu_field_peak_id_color_map(cls, aclu_field_peak_ids: NDArray) -> Dict[int, str]:
        """Map each unique ``aclu_field_peak_id`` to a distinct hex color (stable sort by id)."""
        unique_aclu_field_peak_ids = sorted({int(v) for v in np.asarray(aclu_field_peak_ids, dtype=float).ravel() if np.isfinite(v)})
        if len(unique_aclu_field_peak_ids) == 0:
            return {}

        n_colors: int = len(unique_aclu_field_peak_ids)
        color_map: Dict[int, str] = {}
        for color_idx, a_aclu_field_peak_id in enumerate(unique_aclu_field_peak_ids):
            color_map[a_aclu_field_peak_id] = pg.intColor(color_idx, hues=n_colors).name()
        ## END for color_idx, a_aclu_field_peak_id in enumerate(unique_aclu_field_peak_ids)...

        return color_map



    @classmethod
    def _build_aclu_field_peak_id_label_items(cls, peak_center_x: NDArray, trial_idx: NDArray, aclu_field_peak_id: NDArray, trial_half_height: float = 0.45, label_alpha: float = 0.5, font_size_pt: int = 6, aclu_field_peak_id_color_map: Optional[Dict[int, str]] = None) -> List[pg.TextItem]:
        """Build tiny semi-transparent ``aclu_field_peak_id`` labels above each peak vertical tick."""
        peak_center_x = np.asarray(peak_center_x, dtype=float).ravel()
        trial_idx = np.asarray(trial_idx, dtype=float).ravel()
        aclu_field_peak_id = np.asarray(aclu_field_peak_id, dtype=float).ravel()
        valid_mask = np.isfinite(peak_center_x) & np.isfinite(trial_idx) & np.isfinite(aclu_field_peak_id)
        peak_center_x = peak_center_x[valid_mask]
        trial_idx = trial_idx[valid_mask]
        aclu_field_peak_id = aclu_field_peak_id[valid_mask]
        if peak_center_x.size == 0:
            return []

        if aclu_field_peak_id_color_map is None:
            aclu_field_peak_id_color_map = cls._build_aclu_field_peak_id_color_map(aclu_field_peak_ids=aclu_field_peak_id)

        label_items: List[pg.TextItem] = []
        label_y_offset: float = 0.05
        for a_x, a_trial_idx, a_aclu_field_peak_id in zip(peak_center_x, trial_idx, aclu_field_peak_id):
            track_color: str = aclu_field_peak_id_color_map.get(int(a_aclu_field_peak_id), '#ffffff')
            label_text = pg.TextItem(html=f"<span style='color:{track_color}; font-size:{int(font_size_pt)}pt;'>{int(a_aclu_field_peak_id)}</span>", anchor=(0.5, 1.0))
            label_text.setOpacity(float(label_alpha))
            label_text.setPos(float(a_x), float(a_trial_idx) + float(trial_half_height) + label_y_offset)
            label_text.setZValue(101)
            label_items.append(label_text)
        ## END for a_x, a_trial_idx, a_aclu_field_peak_id in zip(peak_center_x, trial_idx, aclu_field_peak_id)...

        return label_items



    def _update_hover_preview_peak_markers(self, neuron_aclu):
        """Draw the hovered aclu's peak-center vertical markers on the hover-preview axes (one batched scatter)."""
        hover_preview_plot = self.plots.get('hover_preview_plot', None)
        if hover_preview_plot is None:
            return

        self._clear_hover_preview_peak_markers()

        peaks_df = self.plots_data.get('peak_center_markers_df', None)
        if (peaks_df is None) or (neuron_aclu is None) or (len(peaks_df) == 0):
            return

        pen = self.params.get('peak_center_marker_pen', None)
        if pen is None:
            pen = pg.mkPen('w', width=1.5)
        trial_half_height: float = float(self.params.get('peak_center_marker_trial_half_height', 0.45))

        aclu_peaks_df = peaks_df.loc[peaks_df['aclu'].astype(int) == int(neuron_aclu)]
        if len(aclu_peaks_df) == 0:
            return

        summit_idx = aclu_peaks_df['summit_idx'].to_numpy() if ('summit_idx' in aclu_peaks_df.columns) else None
        scatter_items = self._build_peak_marker_scatter_items(peak_center_x=aclu_peaks_df['peak_center_x'].to_numpy(), trial_idx=aclu_peaks_df['trial_idx'].to_numpy(), pen=pen, trial_half_height=trial_half_height, summit_idx=summit_idx)
        if len(scatter_items) == 0:
            return

        for a_scatter in scatter_items:
            hover_preview_plot.addItem(a_scatter)
        ## END for a_scatter in scatter_items....

        self.plots.hover_preview_peak_center_vertical_markers = scatter_items if (len(scatter_items) > 1) else scatter_items[0]



    def _update_hover_preview_aclu_field_peak_id_labels(self, neuron_aclu):
        """Draw the hovered aclu's aclu_field_peak_id debug labels on the hover-preview axes."""
        hover_preview_plot = self.plots.get('hover_preview_plot', None)
        if hover_preview_plot is None:
            return

        existing_hover = self.plots.get('hover_preview_aclu_field_peak_id_debug_labels', None)
        if existing_hover is not None:
            if isinstance(existing_hover, (list, tuple)):
                for a_label in existing_hover:
                    hover_preview_plot.removeItem(a_label)
                ## END for a_label in existing_hover....
            else:
                hover_preview_plot.removeItem(existing_hover)

        self.plots.hover_preview_aclu_field_peak_id_debug_labels = None

        peaks_df = self.plots_data.get('aclu_field_peak_id_labels_df', None)
        if (peaks_df is None) or (neuron_aclu is None) or (len(peaks_df) == 0):
            return

        trial_half_height: float = float(self.params.get('aclu_field_peak_id_label_trial_half_height', self.params.get('peak_center_marker_trial_half_height', 0.45)))
        label_alpha: float = float(self.params.get('aclu_field_peak_id_label_alpha', 0.5))
        font_size_pt: int = int(self.params.get('aclu_field_peak_id_label_font_size_pt', 6))

        aclu_peaks_df = peaks_df.loc[peaks_df['aclu'].astype(int) == int(neuron_aclu)]
        if len(aclu_peaks_df) == 0:
            return

        aclu_field_peak_id_color_maps_dict = self.plots_data.get('aclu_field_peak_id_color_maps_dict', None)
        aclu_field_peak_id_color_map = None if (aclu_field_peak_id_color_maps_dict is None) else aclu_field_peak_id_color_maps_dict.get(int(neuron_aclu), None)

        label_items = self._build_aclu_field_peak_id_label_items(peak_center_x=aclu_peaks_df['peak_center_x'].to_numpy(), trial_idx=aclu_peaks_df['trial_idx'].to_numpy(), aclu_field_peak_id=aclu_peaks_df['aclu_field_peak_id'].to_numpy(), trial_half_height=trial_half_height, label_alpha=label_alpha, font_size_pt=font_size_pt, aclu_field_peak_id_color_map=aclu_field_peak_id_color_map)
        if len(label_items) == 0:
            return

        for a_label in label_items:
            hover_preview_plot.addItem(a_label)
        ## END for a_label in label_items....

        self.plots.hover_preview_aclu_field_peak_id_debug_labels = label_items if (len(label_items) > 1) else label_items[0]



    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 01:11', related_items=[])
    def on_change_selection(self, a_plot_item, new_is_selected: bool):
        """ called when one of the aclu subplots selection changes 
        """
        print(f'on_change_selection(a_plot_item: {a_plot_item}, new_is_selected: {new_is_selected})')

        # plot_data_array

    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 01:11', related_items=[])
    def get_selected_aclus(self, return_only_selected_aclus: bool=True):
        """ gets the user-selected aclus """
        # is_aclu_selected = []
        selected_aclus_list = []
        is_aclu_selected_dict = {}

        for a_linear_index, a_plot_item in enumerate(self.plot_array):
            # is_aclu_selected.append(a_plot_item.is_selected)
            curr_plot_data_dict = self.plots_data.plot_data_array[a_linear_index]
            neuron_aclu = curr_plot_data_dict.get('neuron_aclu', None)
            assert neuron_aclu is not None
            
            if return_only_selected_aclus:
                if a_plot_item.is_selected:
                    selected_aclus_list.append(neuron_aclu)
            else:
                is_aclu_selected_dict[neuron_aclu] = a_plot_item.is_selected
            # curr_image_bounds_extent = plot_data_array[a_linear_index]['image_bounds_extent']
                    
        if return_only_selected_aclus:
            return selected_aclus_list
        else:
            ## return map from aclu to is_selected
            return is_aclu_selected_dict
        


    @function_attributes(short_name=None, tags=['plot', 'pyqtgraph', 'peak', 'marker', 'vertical_line'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-08-25 11:50', related_items=[])
    def add_peak_center_vertical_markers(self, peaks_df: pd.DataFrame, pen=None, trial_half_height: float = 0.45, clear_existing: bool = True):
        """Add short vertical line markers at each peak center on the matching aclu subplot.

        Efficient: one ``ScatterPlotItem`` per aclu (all ticks batched). Avoids ``PlotCurveItem(connect='pairs')``,
        which becomes a continuous zigzag under OpenGL ``paintGL`` (``GL_LINE_STRIP`` ignores ``connect``).

        Each row of ``peaks_df`` contributes one vertical tick spanning
        ``[trial_idx - trial_half_height, trial_idx + trial_half_height]`` at ``x = peak_center_x``.

        Parameters
        ----------
        peaks_df : pd.DataFrame
            Required columns: ``['aclu', 'trial_idx', 'peak_center_x']``.
            Optional column: ``'summit_idx'`` — when present, marker pen width is scaled as
            ``base_width / 2**summit_idx`` (0 = full thickness, 1 = half, 2 = quarter, …) and opacity is
            reduced for higher ranks (0→1.0, 1→0.75, 2→0.65, 3+→0.5).
        pen : optional
            pyqtgraph pen for the markers. Defaults to a thin white pen.
        trial_half_height : float
            Half-height of each vertical tick in trial/y units (default 0.45 ≈ one trial row).
        clear_existing : bool
            If True, remove any previously added peak-center markers first.

        Returns
        -------
        Dict[int, pg.ScatterPlotItem]
            Mapping ``aclu → batched scatter item`` added for that cell.

        Usage
        -----
            peaks_df = pd.DataFrame({'aclu': [...], 'trial_idx': [...], 'peak_center_x': [...]})
            a_TbyT_activity_win.add_peak_center_vertical_markers(peaks_df)

        """
        required_cols = {'aclu', 'trial_idx', 'peak_center_x'}
        missing_cols = required_cols - set(peaks_df.columns)
        assert len(missing_cols) == 0, f"peaks_df missing required columns: {missing_cols}"

        ## Transform to correct indexing:
        # peaks_df = pd.DataFrame({'aclu': [...], 'trial_idx': [...], 'peak_center_x': [...]})
        peaks_df['trial_idx'] = peaks_df['trial_idx'] - 1 # convert from 1-based to 0-based indexing
        peaks_df['trial_idx'] = peaks_df['trial_idx'] * 2 ## handle the double-spacing of results

        if pen is None:
            pen = pg.mkPen('w', width=1.5)

        ## Build aclu → plot index map from plot_data_array
        aclu_to_plot_idx: Dict[int, int] = {}
        for a_plot_data_dict in self.plots_data.plot_data_array:
            neuron_aclu = a_plot_data_dict.get('neuron_aclu', None)
            if neuron_aclu is not None:
                aclu_to_plot_idx[int(neuron_aclu)] = int(a_plot_data_dict['a_linear_index'])
        ## END for a_plot_data_dict in self.plots_data.plot_data_array....

        ## Optionally clear prior markers (supports old List-per-aclu or new single-item-per-aclu)
        existing_markers = self.plots.get('peak_center_vertical_markers', None)
        if clear_existing and (existing_markers is not None):
            for aclu, line_items in existing_markers.items():
                plot_idx = aclu_to_plot_idx.get(int(aclu), None)
                if plot_idx is None:
                    continue
                curr_plot = self.plots.plot_array[plot_idx]
                if isinstance(line_items, (list, tuple)):
                    for a_line in line_items:
                        curr_plot.removeItem(a_line)
                    ## END for a_line in line_items....
                else:
                    curr_plot.removeItem(line_items)
            ## END for aclu, line_items in existing_markers.items()....

        self._clear_hover_preview_peak_markers()

        new_markers: Dict[int, Union[pg.ScatterPlotItem, List[pg.ScatterPlotItem]]] = {}
        marker_cols = ['aclu', 'trial_idx', 'peak_center_x']
        if 'summit_idx' in peaks_df.columns:
            marker_cols.append('summit_idx')
        active_peaks_df = peaks_df.loc[peaks_df['aclu'].isin(list(aclu_to_plot_idx.keys())), marker_cols].copy()
        active_peaks_df['aclu'] = active_peaks_df['aclu'].astype(int)

        for aclu, aclu_peaks_df in active_peaks_df.groupby('aclu', sort=False):
            summit_idx = aclu_peaks_df['summit_idx'].to_numpy() if ('summit_idx' in aclu_peaks_df.columns) else None
            scatter_items = self._build_peak_marker_scatter_items(peak_center_x=aclu_peaks_df['peak_center_x'].to_numpy(), trial_idx=aclu_peaks_df['trial_idx'].to_numpy(), pen=pen, trial_half_height=trial_half_height, summit_idx=summit_idx)
            if len(scatter_items) == 0:
                continue
            plot_idx = aclu_to_plot_idx[int(aclu)]
            curr_plot = self.plots.plot_array[plot_idx]
            for a_scatter in scatter_items:
                curr_plot.addItem(a_scatter)
            ## END for a_scatter in scatter_items....
            new_markers[int(aclu)] = scatter_items if (len(scatter_items) > 1) else scatter_items[0]
        ## END for aclu, aclu_peaks_df in active_peaks_df.groupby('aclu', sort=False)....

        self.plots.peak_center_vertical_markers = new_markers
        self.plots_data.peak_center_markers_df = deepcopy(active_peaks_df)
        self.params.peak_center_marker_pen = pen
        self.params.peak_center_marker_trial_half_height = float(trial_half_height)

        ## Refresh hover-preview markers for the currently hovered cell (if any)
        hovered_idx = self.params.get('hovered_linear_index', None)
        if hovered_idx is not None:
            hovered_plot_data = self.plots_data.plot_data_array[int(hovered_idx)]
            self._update_hover_preview_peak_markers(hovered_plot_data.get('neuron_aclu', None))

        return new_markers


    @function_attributes(short_name=None, tags=['plot', 'pyqtgraph', 'peak', 'marker', 'debug', 'aclu_field_peak_id'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-08-25 15:30', related_items=['add_peak_center_vertical_markers'])
    def add_aclu_field_peak_id_debug_labels(self, peaks_df: Optional[pd.DataFrame] = None, label_alpha: float = 0.5, font_size_pt: int = 6, trial_half_height: Optional[float] = None, clear_existing: bool = True, include_hover_preview: bool = True) -> Dict[int, List[pg.TextItem]]:
        """Add tiny semi-transparent ``aclu_field_peak_id`` labels above each peak vertical tick (debug overlay).

        Each unique ``aclu_field_peak_id`` within an aclu is assigned a distinct color so tracks
        are visually identifiable across trials on that subplot.

        Parameters
        ----------
        peaks_df : pd.DataFrame, optional
            Required columns: ``['aclu', 'trial_idx', 'peak_center_x', 'aclu_field_peak_id']``.
            If None, uses ``self.plots_data.peak_center_markers_df`` (must include ``aclu_field_peak_id``).
        label_alpha : float
            Text opacity (default 0.5).
        font_size_pt : int
            Label font size in points (default 6).
        trial_half_height : float, optional
            Half-height of peak ticks in trial/y units; defaults to peak marker param or 0.45.
        clear_existing : bool
            If True, remove any previously added aclu_field_peak_id debug labels first.
        include_hover_preview : bool
            If True, refresh hover-preview labels for the currently hovered cell.

        Returns
        -------
        Dict[int, List[pg.TextItem]]
            Mapping ``aclu → label items`` added for that cell.

        Usage
        -----
            a_TbyT_activity_win.add_peak_center_vertical_markers(tracked_df[['aclu', 'trial_idx', 'peak_center_x', 'summit_idx']])
            a_TbyT_activity_win.add_aclu_field_peak_id_debug_labels(tracked_df)

        """
        if peaks_df is None:
            peaks_df = self.plots_data.get('peak_center_markers_df', None)
        assert peaks_df is not None, "peaks_df is None and no peak_center_markers_df stored on plots_data"

        required_cols = {'aclu', 'trial_idx', 'peak_center_x', 'aclu_field_peak_id'}
        missing_cols = required_cols - set(peaks_df.columns)
        assert len(missing_cols) == 0, f"peaks_df missing required columns: {missing_cols}"

        peaks_df = deepcopy(peaks_df)
        peaks_df['trial_idx'] = peaks_df['trial_idx'] - 1 # convert from 1-based to 0-based indexing
        peaks_df['trial_idx'] = peaks_df['trial_idx'] * 2 ## handle the double-spacing of results

        if trial_half_height is None:
            trial_half_height = float(self.params.get('peak_center_marker_trial_half_height', 0.45))

        aclu_to_plot_idx: Dict[int, int] = {}
        for a_plot_data_dict in self.plots_data.plot_data_array:
            neuron_aclu = a_plot_data_dict.get('neuron_aclu', None)
            if neuron_aclu is not None:
                aclu_to_plot_idx[int(neuron_aclu)] = int(a_plot_data_dict['a_linear_index'])
        ## END for a_plot_data_dict in self.plots_data.plot_data_array....

        if clear_existing:
            self._clear_aclu_field_peak_id_debug_labels()

        label_cols = ['aclu', 'trial_idx', 'peak_center_x', 'aclu_field_peak_id']
        active_peaks_df = peaks_df.loc[peaks_df['aclu'].isin(list(aclu_to_plot_idx.keys())), label_cols].copy()
        active_peaks_df['aclu'] = active_peaks_df['aclu'].astype(int)

        new_labels: Dict[int, List[pg.TextItem]] = {}
        aclu_field_peak_id_color_maps_dict: Dict[int, Dict[int, str]] = {}
        for aclu, aclu_peaks_df in active_peaks_df.groupby('aclu', sort=False):
            aclu_field_peak_id_color_map = self._build_aclu_field_peak_id_color_map(aclu_field_peak_ids=aclu_peaks_df['aclu_field_peak_id'].to_numpy())
            aclu_field_peak_id_color_maps_dict[int(aclu)] = aclu_field_peak_id_color_map
            label_items = self._build_aclu_field_peak_id_label_items(peak_center_x=aclu_peaks_df['peak_center_x'].to_numpy(), trial_idx=aclu_peaks_df['trial_idx'].to_numpy(), aclu_field_peak_id=aclu_peaks_df['aclu_field_peak_id'].to_numpy(), trial_half_height=float(trial_half_height), label_alpha=float(label_alpha), font_size_pt=int(font_size_pt), aclu_field_peak_id_color_map=aclu_field_peak_id_color_map)
            if len(label_items) == 0:
                continue
            plot_idx = aclu_to_plot_idx[int(aclu)]
            curr_plot = self.plots.plot_array[plot_idx]
            for a_label in label_items:
                curr_plot.addItem(a_label)
            ## END for a_label in label_items....
            new_labels[int(aclu)] = label_items
        ## END for aclu, aclu_peaks_df in active_peaks_df.groupby('aclu', sort=False)....

        self.plots.aclu_field_peak_id_debug_labels = new_labels
        self.plots_data.aclu_field_peak_id_labels_df = deepcopy(active_peaks_df)
        self.plots_data.aclu_field_peak_id_color_maps_dict = aclu_field_peak_id_color_maps_dict
        self.params.aclu_field_peak_id_label_alpha = float(label_alpha)
        self.params.aclu_field_peak_id_label_font_size_pt = int(font_size_pt)
        self.params.aclu_field_peak_id_label_trial_half_height = float(trial_half_height)

        if include_hover_preview:
            hovered_idx = self.params.get('hovered_linear_index', None)
            if hovered_idx is not None:
                hovered_plot_data = self.plots_data.plot_data_array[int(hovered_idx)]
                self._update_hover_preview_aclu_field_peak_id_labels(hovered_plot_data.get('neuron_aclu', None))

        return new_labels


    @function_attributes(short_name=None, tags=['plot', 'pyqtgraph', 'pf_stable_formation_time', 'AcluFirstPlacefieldStabilityThresholdFigure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-20 10:19', related_items=['AcluFirstPlacefieldStabilityThresholdFigure', 'AcluFirstPlacefieldStabilityThresholdFigure.plot_aclus_first_significance_figure'])
    def add_pf_stable_formation_time_distribution_results(self, df_merged: pd.DataFrame):
        """ 2025-08-20 - Not yet finished - add each aclu first-stable-pf time to TrialByTrialActivity figure
        The difficulty lies in mapping the from one of the `df_merged` columns to trial/lap # (ranging [-2.5, 163.5] on the y-axis of each subplot)  

        #TODO 2025-08-20 10:20: - [ ] y-axis is inverted and not quite working
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import AcluFirstPlacefieldStabilityThresholdFigure
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TrialByTrialActivityWindow import TrialByTrialActivityWindow

            df_merged, decoder_outputs, pf1D_dt_outputs, pf1D_dt_snapshot_outputs = AcluFirstPlacefieldStabilityThresholdFigure._compute_for_all_decoders(curr_active_pipeline, track_templates, fr_threshold_Hz=2.0)
            
            _out = dict()
            _out['_display_trial_to_trial_reliability'] = curr_active_pipeline.display(display_function='_display_trial_to_trial_reliability', active_session_configuration_context=None) # _display_trial_to_trial_reliability
            a_TbyT_activity_win: TrialByTrialActivityWindow = _out['_display_trial_to_trial_reliability']
            a_TbyT_activity_win.add_pf_stable_formation_time_distribution_results(df_merged=df_merged) # add the current aclu pf stability times

        
        """
        ## INPUTS: df_merged, a_TbyT_activity_win
        ## get the first plot to determine the y-range:
        curr_plot = self.plots.plot_array[0] 
        vb = pg.ViewBox = curr_plot.vb
        # vb.viewRect()  # Get the current view rectangle of the plot
        y_min, y_max = vb.viewRange()[1] # [-2.5, 163.5]
        y_ptp: float = np.abs(y_max) - np.abs(y_min)
        half_y_ptp: float = y_ptp / 2.0 ## corresponds (roughly) to just the pre/post delta laps

        ## OUTPUTS: y_min, y_max, half_y_ptp
        active_neuron_IDs = deepcopy(self.plots_data.active_neuron_IDs) # .img_item_array[0].getImage().shape # (62, 29951)

        ## INPUTS: df_merged
        active_df_merged = deepcopy(df_merged)[np.isin(df_merged['aclu'], active_neuron_IDs)] ## add to TrialByTrialActivity figure
        self.plots_data.aclu_pf_stable_formation_time_df = deepcopy(active_df_merged) ## add to plots_data
        # active_df_merged

        Assert.same_length(self.plots.img_item_array, active_df_merged)
        # active_df_merged.columns ['aclu', 'duration_fraction_long_LR', 'snap_t_long_LR', 'snap_idx_long_LR', 'duration_fraction_long_RL', 'snap_t_long_RL', 'snap_idx_long_RL', 'duration_fraction_short_LR', 'snap_t_short_LR', 'snap_idx_short_LR', 'duration_fraction_short_RL', 'snap_t_short_RL', 'snap_idx_short_RL', 'delta_rel_snap_t_long_LR', 'delta_rel_snap_t_long_RL', 'delta_rel_snap_t_short_LR', 'delta_rel_snap_t_short_RL']
        active_cols = ['aclu', 'duration_fraction_long_LR', 'duration_fraction_long_RL', 'duration_fraction_short_LR', 'duration_fraction_short_RL']
        decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']

        ## INPUTS: df_merged, 
        # color_dict = {'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'}
        color_dict: Dict[types.DecoderName, pg.QtGui.QColor] = deepcopy(self.plots_data.color_dict)

        # _all_common_kwargs = dict(lw=0.5, alpha=0.7)
        _left_common_kwargs = dict() #dict(ymin=ymid, ymax=ymax, **_all_common_kwargs)
        _right_common_kwargs = dict() # dict(ymin=ymin, ymax=ymid, **_all_common_kwargs)
        # active_decoder_kwargs_list = [dict(**_left_common_kwargs, label='Long_LR', pen='red'),
        #                         dict(**_right_common_kwargs, label='Long_RL', pen='orange'),
        #                         dict(**_left_common_kwargs, label='Short_LR', pen='blue'),
        #                         dict(**_right_common_kwargs, label='Short_RL', pen='cyan')]

        active_decoder_kwargs_list = [dict(**_left_common_kwargs, label='Long_LR'),
                                dict(**_right_common_kwargs, label='Long_RL'),
                                dict(**_left_common_kwargs, label='Short_LR'),
                                dict(**_right_common_kwargs, label='Short_RL')]

        active_decoder_kwargs_dict = dict(zip(decoder_names, active_decoder_kwargs_list)) # {'long_LR': {'ymin': -2.5, 'ymax': 81.5, 'label': 'Long_LR', 'pen': 'red'}, ...}
        ## OUTPUTS: decoder_names, active_decoder_kwargs_dict
        active_data_cols = ['duration_fraction_long_LR', 'duration_fraction_long_RL', 'duration_fraction_short_LR', 'duration_fraction_short_RL']
        active_long_cols = active_data_cols[:2] # ['duration_fraction_long_LR', 'duration_fraction_long_RL']
        active_short_cols = active_data_cols[2:] # ['duration_fraction_short_LR', 'duration_fraction_short_RL']
        # active_df_merged[active_cols]

        # mapping the from one of the `df_merged` columns [0.0, 1.0] to trial/lap # (ranging [-2.5, 163.5] on the y-axis of each subplot)  
        ## Handle inversion:
        active_df_merged[active_long_cols] = (1.0 - active_df_merged[active_long_cols])
        active_df_merged[active_short_cols] = (1.0 - active_df_merged[active_short_cols])
        ## perform mapping:
        active_df_merged[active_long_cols] = (active_df_merged[active_long_cols] * half_y_ptp) + y_min # y_min + (active_df_merged[active_long_cols] * half_y_ptp).values
        active_df_merged[active_short_cols] = (active_df_merged[active_short_cols] * half_y_ptp) + y_min + half_y_ptp # the 2nd `+ half_y_ptp` is to offset the short lap values to be after the long lap values
        # active_df_merged[active_cols]

        # a_TbyT_activity_win.plots.img_item_array
        # curr_plot = a_TbyT_activity_win.plots.plot_array[0]

        new_lines = {}
        for i, aclu in enumerate(self.plots_data.active_neuron_IDs):
            curr_plot = self.plots.plot_array[i]
            vb = curr_plot.vb
            y_min, y_max = vb.viewRange()[1]

            def map_to_viewbox(y_val):
                # flip if necessary
                if vb.invertY():  
                    return y_max - (y_val - y_min)
                else:
                    return y_val

            a_row = active_df_merged[active_df_merged['aclu'] == aclu] 
            a_row_dict = dict(zip(decoder_names, a_row[active_data_cols].to_numpy().flatten())) # (4,)
            new_lines[aclu] = {}
            for a_decoder_name, a_line_y_value in a_row_dict.items():
                if a_line_y_value is not None and not np.isnan(a_line_y_value):
                    # print(f'adding line for aclu: {aclu}, a_decoder_name: {a_decoder_name}, a_line_y_value: {a_line_y_value}')
                    corrected_y = a_line_y_value
                    # corrected_y = y_max - (a_line_y_value - y_min) # Correct y-value for inverted y-axis
                    # corrected_y = map_to_viewbox(a_line_y_value)  # Ensure the y-value is mapped to the viewbox
                    a_line = pg.InfiniteLine(pos=corrected_y, angle=0, **(active_decoder_kwargs_dict[a_decoder_name] | dict(pen=pg.mkPen(color_dict[a_decoder_name], width=2))))  # horizontal line at y=a_line_y_value            
                    # a_line = pg.InfiniteLine(pos=2, angle=0, pen='r')  # vertical line at x=2
                    curr_plot.addItem(a_line)
                    new_lines[aclu][a_decoder_name] = a_line


        self.plots.aclu_pf_stable_formation_time_lines = new_lines
        # self.plots_data.aclu_pf_stable_formation_time_df = deepcopy(active_df_merged)
        