import numpy as np

from neuropy.utils.misc import safe_item
from neuropy.utils.dynamic_container import overriding_dict_with # used in display_all_pf_2D_pyqtgraph_binned_image_rendering to only get the valid kwargs to pass from the display config
from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range, _build_neuron_identity_label # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering

from pyphocorehelpers.function_helpers import function_attributes
# from pyphocorehelpers.geometry_helpers import compute_data_aspect_ratio, compute_data_extent
from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_common import pyqtplot_common_setup

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent


@function_attributes(short_name='pyqtplot_plot_image_array', tags=['display','pyqtgraph','plot','image','binned','2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2022-09-29 00:00')
def pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Plots an array of images provided in 'images' argument
    images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
        or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
        
    NOTES:
        2022-09-29 - Extracted from Notebook
            🚧 Needs subplot labels changed from Cell[i] to the appropriate standardized titles. Needs other minor refinements.
            🚧 pyqtplot_plot_image_array needs major improvements to achieve feature pairity with display_all_pf_2D_pyqtgraph_binned_image_rendering, so probably just use display_all_pf_2D_pyqtgraph_binned_image_rendering.
        
    Example:
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        app, win, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy)
        win.show()
        
        
    # 🚧 TODO: COMPATIBILITY: replace compute_paginated_grid_config with standardized `_determine_best_placefield_2D_layout` block (see below):
    ```
    from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons),
        **overriding_dict_with(lhs_dict={'subplots': (40, 3), 'fig_column_width': 8.0, 'fig_row_height': 1.0, 'resolution_multiplier': 1.0, 'max_screen_figure_size': (None, None), 'last_figure_subplots_same_layout': True, 'debug_print': True}, **figure_format_config)) 

    print(f'nfigures: {nfigures}\ndata_aspect_ratio: {data_aspect_ratio}')
    # Loop through each page/figure that's required:
    for page_fig_ind, page_fig_size, page_grid_size in zip(np.arange(nfigures), page_figure_sizes, page_grid_sizes):
        print(f'\tpage_fig_ind: {page_fig_ind}, page_fig_size: {page_fig_size}, page_grid_size: {page_grid_size}')
        # print(f'\tincluded_combined_indicies_pages: {included_combined_indicies_pages}\npage_grid_sizes: {page_grid_sizes}\npage_figure_sizes: {page_figure_sizes}')
    ```
        
        
        
    """
    root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'pyqtplot_plot_image_array: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget) ## 🚧 TODO: BUG: this makes a new QMainWindow to hold this item, which is inappropriate if it's to be rendered as a child of another control
    
    pg.setConfigOptions(imageAxisOrder='col-major') # this causes the placefields to be rendered horizontally, like they were in _temp_pyqtplot_plot_image_array
    
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map

    image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    # image_aspect_ratio, image_width_height_tuple = compute_data_aspect_ratio(x_range, y_range)
    # print(f'image_aspect_ratio: {image_aspect_ratio} - xScale/yScale: {float(image_width_height_tuple.width) / float(image_width_height_tuple.height)}')
    
    # Compute Images:
    included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
    nMapsToShow = len(included_unit_indicies)

    # Paging Management: Constrain the subplots values to just those that you need
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
    page_idx = 0 # page_idx is zero here because we only have one page:
    
    img_item_array = []
    other_components_array = []
    plot_array = []

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
        
        # # plot mode:
        curr_plot = root_render_widget.addPlot(row=curr_row, col=curr_col, title=curr_cell_identifier_string) # , name=curr_plot_identifier_string 
        curr_plot.setObjectName(curr_plot_identifier_string)
        curr_plot.showAxes(False)
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
        # curr_plot.setAspectLocked(lock=True, ratio=image_aspect_ratio)
        # curr_plot.showAxes(True)
        # curr_plot.showGrid(True, True, 0.7)
        # curr_plot.setLabel('bottom', "Label to test offset")
        
        # # Overlay cell identifier text:
        # curr_label = pg.TextItem(f'Cell[{neuron_IDX}]', color=(230, 230, 230))
        # curr_label.setPos(30, 60)
        # curr_label.setParentItem(img_item)
        # # curr_plot.addItem(curr_label, ignoreBounds=True)
        # curr_plot.addItem(curr_label)

        # Update the image:
        img_item.setImage(image, rect=image_bounds_extent, autoLevels=False) # rect: [x, y, w, h]
        img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)

        # curr_plot.set
        # margin = 2.0
        # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
        # curr_plot.setXRange(*x_range)
        # curr_plot.setYRange(*y_range)
        curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
        # Sets only the panning limits:
        curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
        # Link Axes to previous item:
        if a_linear_index > 0:
            prev_plot_item = plot_array[a_linear_index-1]
            curr_plot.setXLink(prev_plot_item)
            curr_plot.setYLink(prev_plot_item)
            
            
        # Interactive Color Bar:
        bar = pg.ColorBarItem(values= (0, 1), colorMap=cmap, width=5, interactive=False) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem(img_item, insert_in=curr_plot)

        img_item_array.append(img_item)
        plot_array.append(curr_plot)
        other_components_array.append({'color_bar':bar}) # note this is a list of Dicts, one for every image
        
    # Post images loop:
    enable_show = False
    
    if parent_root_widget is not None:
        if enable_show:
            parent_root_widget.show()
        
        parent_root_widget.setWindowTitle('pyqtplot image array')

    # pg.exec()
    return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array


@function_attributes(short_name='all_pf_2D_pyqtgraph_binned_image_rendering', tags=['display','pyqtgraph','plot','image','binned','2D'], input_requires=[], output_provides=[], uses=['BasicBinnedImageRenderingWindow'], used_by=['_display_placemaps_pyqtplot_2D'], creation_date='2022-08-16 00:00')
def display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config, debug_print=True): # , **kwargs
    """ 2022-08-16 - A fresh implementation of a pf_2D placefield renderer that uses the BasicBinnedImageRenderingWindow subclass. 
    
    Uses the common `_determine_best_placefield_2D_layout(...)` setup so that its returned subplots layout is the same as the matplotlib version in NeuroPy.neuropy.plotting.ratemaps.plot_ratemap_2D(...) (the main Matplotlib version that works)
    
    Analagous to:
        NeuroPy.neuropy.plotting.ratemaps.plot_ratemap_2D: the matplotlib-based version

    Uses:
        active_pf_2D.xbin, ybin=active_pf_2D.ybin
        active_pf_2D.occupancy
    
        active_pf_2D.ratemap.neuron_ids, active_pf_2D.ratemap.neuron_extended_ids
        
        active_pf_2D.ratemap.tuning_curves || active_pf_2D.ratemap.spikes_maps

        LayoutScrollability.NON_SCROLLABLE
        
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import display_all_pf_2D_pyqtgraph_binned_image_rendering

        out_all_pf_2D_pyqtgraph_binned_image_fig = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config)
        
    """

    wants_crosshairs= figure_format_config.get('wants_crosshairs', False) 
    
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    # cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
    # color_map = figure_format_config.get('color_map', 'viridis')
    color_map = figure_format_config.get('color_map', pg.colormap.get('viridis','matplotlib'))
    # color_map = figure_format_config.get('color_map', 'viridis')
    # color_bar_mode = figure_format_config.get('color_bar_mode', 'each')
    color_bar_mode = figure_format_config.get('color_bar_mode', None) # no colorbars rendered  
    use_special_overlayed_title = True
    brev_mode = figure_format_config.get('brev_mode', PlotStringBrevityModeEnum.CONCISE)
    plot_variable = figure_format_config.get('plot_variable', enumTuningMap2DPlotVariables.TUNING_MAPS)
    drop_below_threshold = figure_format_config.get('drop_below_threshold', 0.0000001) # try to get the 'drop_below_threshold' argument
    included_unit_indicies = figure_format_config.get('included_unit_indicies', None)
    included_unit_neuron_IDs = figure_format_config.get('included_unit_neuron_IDs', None)
    scrollability_mode = figure_format_config.get('scrollability_mode', LayoutScrollability.SCROLLABLE) 
    
    missing_aclu_string_formatter = figure_format_config.get('missing_aclu_string_formatter', None)
    # missing_aclu_string_formatter: a lambda function that takes the current aclu string and returns a modified string that reflects that this aclu value is missing from the current result (e.g. missing_aclu_string_formatter('3') -> '3 <shared>')
    if missing_aclu_string_formatter is None:
        # missing_aclu_string_formatter = lambda curr_extended_id_string: f'{curr_extended_id_string} <shared>'
        missing_aclu_string_formatter = lambda curr_extended_id_string: f'{curr_extended_id_string}-'

    if included_unit_neuron_IDs is not None:
        if debug_print:
            print(f'included_unit_neuron_IDs: {included_unit_neuron_IDs}')
        if not isinstance(included_unit_neuron_IDs, np.ndarray):
            included_unit_neuron_IDs = np.array(included_unit_neuron_IDs) # convert to np.array if needed

        n_neurons = np.size(included_unit_neuron_IDs)
        if debug_print:
            print(f'\t n_neurons: {n_neurons}')

        shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == active_pf_2D.ratemap.neuron_ids)), default=None) for aclu in included_unit_neuron_IDs] # [0, 1, None, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]

        if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
            active_maps = active_pf_2D.ratemap.tuning_curves
            title_substring = 'Placemaps'
        elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
            active_maps = active_pf_2D.ratemap.spikes_maps
            title_substring = 'Spikes Maps'
        else:
            raise ValueError

        ## Non-pre-build method where shared_IDXs_map is directly passed as included_unit_indicies so it's returned in the main loop:
        included_unit_indicies = shared_IDXs_map
        if debug_print:
            print(f'active_maps.shape: {np.shape(active_maps)}, type: {type(active_maps)}') # _local_active_maps.shape: (70, 63, 16), type: <class 'numpy.ndarray'>

    else:
        ## normal (non-shared mode)
        shared_IDXs_map = None
        active_maps = None

        if included_unit_indicies is None:
            included_unit_indicies = np.arange(active_pf_2D.ratemap.n_neurons) # include all unless otherwise specified
        
        ## Get Data to plot:
        if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
            active_maps = active_pf_2D.ratemap.tuning_curves[included_unit_indicies]
            title_substring = 'Placemaps'
        elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
            active_maps = active_pf_2D.ratemap.spikes_maps[included_unit_indicies]
            title_substring = 'Spikes Maps'
        else:
            raise ValueError

    # Build the formatter for rendering the max values such as the peak firing rate or max spike counts:
    if brev_mode.should_show_firing_rate_label:
        max_value_formatter = _build_variable_max_value_label(plot_variable=plot_variable)
    else:
        max_value_formatter = None
        
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=included_unit_indicies,
        **overriding_dict_with(lhs_dict={'subplots': (40, 3), 'fig_column_width': 8.0, 'fig_row_height': 1.0, 'resolution_multiplier': 1.0, 'max_screen_figure_size': (None, None), 'last_figure_subplots_same_layout': True, 'debug_print': True}, **figure_format_config))

    active_xbins = active_pf_2D.xbin
    active_ybins = active_pf_2D.ybin
    out = None
    # New page-based version:
    for page_idx in np.arange(num_pages):
        if debug_print:
            print(f'page_idx: {page_idx}')
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

            ## non-pre create version:
            if curr_included_unit_index is not None:
                # valid neuron ID, access like normal
                pfmap = np.squeeze(active_maps[curr_included_unit_index,:,:]).copy() # matplotlib-version approach with active_maps
                # normal (non-shared mode)
                neuron_IDX = curr_included_unit_index
                curr_extended_id_string = active_pf_2D.ratemap.get_extended_neuron_id_string(neuron_i=neuron_IDX) 
                
                ## Labeling:
                formatted_max_value_string = None
                if brev_mode.should_show_firing_rate_label:
                    assert max_value_formatter is not None
                    ## NOTE: must set max_value_formatter on the pfmap BEFORE the `_scale_current_placefield_to_acceptable_range` is called to have it show accurate labels!
                    formatted_max_value_string = max_value_formatter(np.nanmax(pfmap))
                    
                final_title_str = _build_neuron_identity_label(neuron_extended_id=active_pf_2D.ratemap.neuron_extended_ids[neuron_IDX], brev_mode=brev_mode, formatted_max_value_string=formatted_max_value_string, use_special_overlayed_title=use_special_overlayed_title)

            else:
                # invalid neuron ID, generate blank entry
                pfmap = np.zeros((np.shape(active_maps)[1], np.shape(active_maps)[2])) # fully allocated new array of zeros
                curr_extended_id_string = f'{included_unit_neuron_IDs[a_linear_index]}' # get the aclu value (which is all that's known about the missing cell and use that as the curr_extended_id_string
                final_title_str = missing_aclu_string_formatter(curr_extended_id_string)

            ## Once the max_value_formatter is called with the unscaled pfmap, we can call _scale_current_placefield_to_acceptable_range to scale it appropriately:
            pfmap = _scale_current_placefield_to_acceptable_range(pfmap, occupancy=active_pf_2D.occupancy, drop_below_threshold=drop_below_threshold)                       

            if out is None:
                # first iteration only
                out = BasicBinnedImageRenderingWindow(pfmap, active_xbins, active_ybins, name=f'pf[{final_title_str}]', title=final_title_str, variable_label=curr_extended_id_string, wants_crosshairs=wants_crosshairs, color_map=color_map, color_bar_mode=color_bar_mode, scrollability_mode=scrollability_mode)
            else:
                out.add_data(row=(out.params.plot_row_offset + curr_page_relative_row), col=curr_page_relative_col, matrix=pfmap, xbins=active_xbins, ybins=active_ybins, name=f'pf[{final_title_str}]', title=final_title_str, variable_label=curr_extended_id_string)
        
    # ## Debugging only:
    # out.plots_data.included_unit_neuron_IDs = included_unit_neuron_IDs
    # out.plots_data.included_unit_indicies = included_unit_indicies
    # out.plots_data.shared_IDXs_map = shared_IDXs_map
    # out.plots_data._local_active_maps = _local_active_maps
    # out.plots_data.active_maps = active_maps

    return out
    
