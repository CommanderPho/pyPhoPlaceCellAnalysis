# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import numpy as np

from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range


from pyphocorehelpers.geometry_helpers import compute_data_aspect_ratio, compute_data_extent
from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_basic import pyqtplot_common_setup

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent


def pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, enable_LUT_Histogram=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Plots an array of images provided in 'images' argument
    images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
        or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
        
    Example:
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        app, win, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy)
        win.show()
    """
    root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'pyqtplot_plot_image_array: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget)
    
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
        other_components_array.append({'color_bar':bar})
        
    # Post images loop:
    enable_show = False
    
    if parent_root_widget is not None:
        if enable_show:
            parent_root_widget.show()
        
        parent_root_widget.setWindowTitle('pyqtplot image array')

    # pg.exec()
    return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array


