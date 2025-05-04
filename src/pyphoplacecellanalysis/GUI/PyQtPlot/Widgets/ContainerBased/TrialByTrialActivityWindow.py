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
    def build_formatted_title_string(cls, title: str) -> str:
        """ returns the title of the entire plot
        """
        return f"<span style = 'font-size : 12px;' >{title}</span>"
    

    @classmethod
    def perform_build_single_cell_formatted_descriptor_string(cls, active_one_step_decoder, aclu) -> str:
        """ Builds a formatted title for each cell, like "aclu: 19, (shank 2, cluster 22)"
        
        cls.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=override_active_one_step_decoder, aclu=aclu)
        """
        # neuron_i: int = list(self.plots_data.active_one_step_decoder.included_neuron_IDs).index(aclu)
        curr_extended_id_string: str = active_one_step_decoder.ratemap.get_extended_neuron_id_string(neuron_id=aclu) # 2025-01-16 05:42  -- AssertionError: neuron_id: 16 is not in self.neuron_ids: [2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 43, 44, 47, 48, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 95, 98, 101, 102, 103, 104]
        # final_title_str: str = f"aclu: {aclu}: {curr_extended_id_string}" # _build_neuron_identity_label(neuron_extended_id=curr_extended_id_string, brev_mode=None, formatted_max_value_string=None, use_special_overlayed_title=True)
        final_title_str: str = f"aclu: <span style = 'font-size : 14px;' >{aclu}</span>:\n<span style = 'font-size : 11px;' >{curr_extended_id_string}</span>"
        return final_title_str
    

    @function_attributes(short_name=None, tags=['matplotlib', 'trial-to-trial-variability', 'laps'], input_requires=[], output_provides=[], uses=[], used_by=['plot_trial_to_trial_reliability_all_decoders_image_stack'], creation_date='2024-08-29 03:26', related_items=[])
    @classmethod
    def _plot_trial_to_trial_reliability_image_array(cls, active_one_step_decoder, z_scored_tuning_map_matrix, active_neuron_IDs=None, max_num_columns=5, drop_below_threshold=0.0000001, cmap=None, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False):
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
            cmap = ColormapHelpers.create_transparent_colormap(cmap_name='Reds', lower_bound_alpha=0.01) # prepare a linear color map


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
                curr_cell_identifier_string = cls.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=active_one_step_decoder, aclu=neuron_aclu)
                _curr_plot_data_dict['neuron_IDX'] = None
                _curr_plot_data_dict['neuron_aclu'] = neuron_aclu

            _curr_plot_data_dict['curr_cell_identifier_string'] = curr_cell_identifier_string
            curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'
            _curr_plot_data_dict['curr_plot_identifier_string'] = curr_plot_identifier_string
            # # Pre-filter the data:
            image = _scale_current_placefield_to_acceptable_range(np.squeeze(images[a_linear_index,:,:]), occupancy=occupancy, drop_below_threshold=drop_below_threshold)

            # Build the image item:
            img_item = pg.ImageItem(image=image, levels=(0,1))
            
            formatted_title: str = cls.build_formatted_title_string(title=curr_cell_identifier_string)   
            _curr_plot_data_dict['formatted_title'] = formatted_title
            

            # # plot mode:
            curr_plot: SelectablePlotItem = SelectablePlotItem(title=formatted_title, is_selected=False)
            root_render_widget.addItem(curr_plot, row=(curr_row + plots_start_row_idx), col=curr_col)            
            curr_plot.setObjectName(curr_plot_identifier_string)
            # curr_plot.showAxes(False)
            curr_plot.showAxes(True)
            curr_plot.setDefaultPadding(0.0)  # plot without padding data range

            # Set the plot title:
            curr_plot.setTitle(formatted_title)    
            set_small_title(curr_plot, title_row_fixed_height) ## title set to a constant height here
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
    

    @function_attributes(short_name=None, tags=['reliability', 'decoders', 'all', 'pyqtgraph', 'display', 'figure', 'main'], input_requires=[], output_provides=[], uses=['plot_trial_to_trial_reliability_image_array', 'create_transparent_colormap'], used_by=[], creation_date='2024-08-29 04:34', related_items=[])
    @classmethod
    def plot_trial_to_trial_reliability_all_decoders_image_stack(cls, directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity], active_one_step_decoder, drop_below_threshold=0.0000001,
                                                                  app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, defer_show:bool=False, name:str = 'TrialByTrialActivityWindow',
                                                                  override_active_neuron_IDs=None,
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

        additional_cmaps = {k: ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1) for k, v in additional_cmap_names.items()}
        additional_legend_entries = list(zip(directional_active_lap_pf_results_dicts.keys(), additional_cmap_names.values() )) # ['red', 'purple', 'green', 'orange']

        # Plots only the first data-series ('long_LR')
        app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array, plot_data_array, (lblTitle, lblFooter) = cls._plot_trial_to_trial_reliability_image_array(active_one_step_decoder=active_one_step_decoder, z_scored_tuning_map_matrix=active_z_scored_tuning_map_matrix, active_neuron_IDs=active_neuron_IDs, drop_below_threshold=drop_below_threshold, cmap=additional_cmaps['long_LR'])
        

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

        ## Add the legend below all the rows:
        root_render_widget.nextRow()
        # Create a layout for the legend at the new row
        # Add a layout for the legend at the bottom, spanning all columns
        # legend_layout: pg.GraphicsLayout = root_render_widget.addLayout(row=root_render_widget.rowCount(), col=0, colspan=root_render_widget.columnCount())
        legend_layout: pg.GraphicsLayout = root_render_widget.addLayout()  # Automatically places in the next available row
        legend_entries_dict = {}
        # Add labels for each entry in the legend
        for i, (label, color) in enumerate(additional_legend_entries):
            # legend_text = pg.LabelItem(label, color=color)
            legend_text = SelectableLabelItem(label, color=color, is_selected=True)
            legend_entries_dict[label] = legend_text
            # legend_layout.addItem(legend_text, row=0, col=i)  # Place all labels in a single row
            legend_layout.addItem(legend_text, row=i, col=0)  # Place all labels in a single columns
            
        legend_layout.setMaximumWidth(100)

        #TODO 2024-11-12 12:22: - [ ] Add position plot to the right-most column of the figure, spanning all rows after the first.
        ## Add position plot
        root_render_widget.nextRow()
        # position_plot_layout: pg.GraphicsLayout = root_render_widget.addLayout()  # Automatically places in the next available row
        position_plot = root_render_widget.addPlot(row=2, col=5, rowspan=4, colspan=1) # start below the legend. Ideally span to rows until the end of the figure.
        # position_plot.addCurve() # 
        # active_trial_by_trial_activity_obj # don't have the position, tragic
        ## Usage:
        # position_plot = _a_trial_by_trial_window.plots.position_plot # PlotItem
        # pos_df: pd.DataFrame = deepcopy(active_pf_dt.position.to_dataframe())
        # position_plot.clearPlots()
        # position_plot.plot(x=pos_df['x'].to_numpy(), y=pos_df['t'].to_numpy())

            
        # END if is_overlaid_heatmaps_mode                
        parent_root_widget.setWindowTitle('TrialByTrialActivity - trial_to_trial_reliability_all_decoders_image_stack')

        additional_img_items_dict['long_LR'] = img_item_array # set first decoder to original image items

        _obj = cls()
        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name,
                                 root_render_widget=root_render_widget,
                                 plot_array=plot_array,
                                 legend_layout=legend_layout,
                                 legend_entries_dict=legend_entries_dict,
                                 other_components_array=other_components_array,
                                 img_item_array=img_item_array,
                                 additional_img_items_dict=additional_img_items_dict, 
                                #  position_plot_layout=position_plot_layout,
                                 position_plot=position_plot, 
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
        _obj.params = VisualizationParameters(name=name, use_plaintext_title=False, **param_kwargs)
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
        return self.perform_build_single_cell_formatted_descriptor_string(active_one_step_decoder=override_active_one_step_decoder, aclu=aclu)

    def build_internal_callbacks(self):
        ## add selection changed callbacks
        for a_linear_index, a_plot_item in enumerate(self.plot_array):
            a_plot_item.sigSelectedChanged.connect(self.on_change_selection)

        for a_decoder_name, a_label_item in self.plots.legend_entries_dict.items():
            a_label_item.sigSelectedChanged.connect(self.on_change_series_legend_selection)


    @function_attributes(short_name=None, tags=['opacity', 'series'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-12 00:00', related_items=[])
    def set_series_opacity(self, target_decoder_name: types.DecoderName, target_opacity: float = 0.1):
        if 'long_LR' not in self.plots.additional_img_items_dict:
            self.plots.additional_img_items_dict['long_LR'] = self.plots.img_item_array
            
        for an_img_item in self.plots.additional_img_items_dict[target_decoder_name]:
            an_img_item.setOpacity(target_opacity)
            
    @function_attributes(short_name=None, tags=['opacity', 'series'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-12 00:00', related_items=[])
    def restore_all_series_opacity(self, override_all_opacity: Optional[float] = None):
        if 'long_LR' not in self.plots.additional_img_items_dict:
            self.plots.additional_img_items_dict['long_LR'] = self.plots.img_item_array
            
        if override_all_opacity is None:
            override_all_opacity = 1.0
            
        for a_decoder_name, an_img_item_arr in self.plots.additional_img_items_dict.items():
            for an_img_item in an_img_item_arr:
                an_img_item.setOpacity(override_all_opacity)

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
        
