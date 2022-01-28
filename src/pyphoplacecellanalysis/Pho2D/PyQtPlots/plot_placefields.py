# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from dataclasses import dataclass

from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config

# @dataclass
# class BasicPyQtPlotApp(object):
#     """Docstring for BasicPyQtPlotApp."""
#     app: Any
#     win: QtGui.QMainWindow
#     w: pg.GraphicsLayoutWidget


def pyqtplot_common_setup(a_title):
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    pg.setConfigOptions(antialias = True)
    app = pg.mkQApp(a_title)
    print(f'type(app): {type(app)}')
    # Create window to hold the image:
    win = QtGui.QMainWindow()
    win.resize(1600, 1600)
    # Creating a GraphicsLayoutWidget as the central widget
    w = pg.GraphicsLayoutWidget()
    win.setCentralWidget(w)
    
    return w, win, app


def pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, enable_LUT_Histogram=False):
    """ Plots an array of images provided in 'images' argument
    images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
        or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
    """

    w, win, app = pyqtplot_common_setup(f'pyqtplot_plot_image_array: {np.shape(images)}')
    
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map

    global_min_x = np.nanmin(xbin_edges)
    global_max_x = np.nanmax(xbin_edges)

    global_min_y = np.nanmin(ybin_edges)
    global_max_y = np.nanmax(ybin_edges)

    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y

    # print(f'global_min_x: {global_min_x}, global_max_x: {global_max_x}, global_min_y: {global_min_y}, global_max_y: {global_max_y}\nglobal_width: {global_width}, global_height: {global_height}')
    # Get rect image extent in the form [x, y, w, h]:
    image_bounds_extent = [global_min_x, global_min_y, global_width, global_height]

    # Compute Images:
    included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
    nMapsToShow = len(included_unit_indicies)

    # Paging Management: Constrain the subplots values to just those that you need
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
    page_idx = 0 # page_idx is zero here because we only have one page:

    # ## Reshape images mode: reshape them into rows, columns, *, *
    # flat_images_shape = np.shape(images)
    # new_images_shape = (subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns, flat_images_shape[-2], flat_images_shape[-1])
    # images = np.reshape(images, new_images_shape)
    # # print(f'flat_images_shape: {flat_images_shape}, new_images_shape: {new_images_shape}, np.shape(images): {np.shape(images)}')

    for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
        # Need to convert to page specific:
        curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
        curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
        curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
        # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

        cell_idx = curr_included_unit_index

        image = np.squeeze(images[a_linear_index,:,:])
        # Pre-filter the data:
        image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        if drop_below_threshold is not None:
            image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy

        # Build the image item:
        img_item = pg.ImageItem(image=image, levels=(0,1))

        # # plot mode:
        curr_plot = w.addPlot(row=curr_row, col=curr_col)
        curr_plot.addItem(img_item)  # add ImageItem to PlotItem
        curr_plot.showAxes(True)
        # curr_plot.showGrid(True, True, 0.7)

        curr_label = pg.TextItem(f'Cell[{cell_idx}]', color=(230, 230, 230))
        curr_label.setPos(30, 60)
        curr_label.setParentItem(img_item)
        # curr_plot.addItem(curr_label, ignoreBounds=True)
        curr_plot.addItem(curr_label)

        # Update the image:
        img_item.setImage(image, rect=image_bounds_extent)
        img_item.setLookupTable(cmap.getLookupTable(nPts=256))

        margin = 2.0
        # curr_plot.setXRange(np.min(geometry[:, 0])-margin, np.max(geometry[:, 0])+margin)
        # curr_plot.setYRange(np.min(geometry[:, 1])-margin, np.max(geometry[:, 1])+margin)
        curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        curr_plot.setYRange(global_min_y-margin, global_max_y+margin)

        # # Interactive Color Bar:
        # bar = pg.ColorBarItem(values= (0, 1), colorMap=cmap) # prepare interactive color bar
        # # Have ColorBarItem control colors of img and appear in 'plot':
        # bar.setImageItem(image, insert_in=curr_plot)



    # ## Dual Index Iteration Mode:
    # for i in np.arange(np.shape(images)[0]):
    #     for j in np.arange(np.shape(images)[1]):
    #         image = np.squeeze(images[i,j,:,:])
    #         # Pre-filter the data:
    #         image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
    #         if drop_below_threshold is not None:
    #             image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
    #
    #         # ImageView version:
    #         # imv = pg.ImageView()
    #         # img_item = pg.ImageItem(image=np.eye(3), levels=(0,1))
    #         img_item = pg.ImageItem(image=image, levels=(0,1))
    #         # curr_item = w.addItem(img_item, row=i, col=j)
    #
    #         # # plot mode:
    #         curr_plot = w.addPlot(row=i, col=j)
    #         curr_plot.addItem(img_item)  # add ImageItem to PlotItem
    #         curr_plot.showAxes(True)
    #         # curr_plot.showGrid(True, True, 0.7)
    #
    #         curr_label = pg.TextItem(f'Cell[{i}]', color=(230, 230, 230))
    #         curr_label.setPos(30, 60)
    #         curr_label.setParentItem(img_item)
    #         curr_plot.addItem(curr_label, ignoreBounds=True)
    #
    #         # img_item.setImage(image, xvals=xbin_edges, rect=image_bounds_extent)
    #         img_item.setImage(image, rect=image_bounds_extent)
    #         # img_item.setColorMap(cmap)
    #         img_item.setLookupTable(cmap.getLookupTable(nPts=256))
    #
    #         margin = 2.0
    #         # curr_plot.setXRange(np.min(geometry[:, 0])-margin, np.max(geometry[:, 0])+margin)
    #         # curr_plot.setYRange(np.min(geometry[:, 1])-margin, np.max(geometry[:, 1])+margin)
    #         curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
    #         curr_plot.setYRange(global_min_y-margin, global_max_y+margin)


    # p1 = w.addPlot(row=0, col=0)

    # # minimum width value of the label
    # label.setFixedWidth(130)


    # # Old linear mode:
    # num_images = np.shape(images)[0]
    # image_indicies = np.arange(num_images)
    # for i in image_indicies:
    #     image = np.squeeze(images[i,:,:])
    #     ## Add 3 plots into the first row (automatic position)
    #     # active_ax = layout.addPlot(title=f'Plot[{i}]')
    #
    #     # ImageView version:
    #     imv = pg.ImageView()
    #     # imv.setImage(image, xvals=xbin_edges)
    #     layout.addWidget(imv)
    #
    #     # Viewbox version:
    #     # vb = layout.addViewBox(lockAspect=False)
    #     # # Build the ImageItem (which I'm guessing is like pg.ImageView) to add the image
    #     # imv = pg.ImageItem() # Create it with the current image
    #     # vb.addItem(imv) # add the item to the view box: why do we need the wrapping view box?
    #     # vb.autoRange()
    #
    #     # layout.addItem(imv)
    #
    #     ## Display the data and assign each frame a time value from 1.0 to 3.0
    #     # imv.setImage(image, xvals=xbin_edges)
    #     imv.setImage(image, xvals=xbin_edges)
    #     # Set the color map:
    #     # imv.setColorMap(cmap)

    # if enable_LUT_Histogram:
    #     lut = pg.HistogramLUTItem(orientation="horizontal")
    #     imv.addItem(lut)
    #     imv.setLookupTable(lut, autoLevel=True)
    #     h = imv.getHistogram()
    #     lut.plot.setData(*h)

    # Post images loop:
    win.show()
    win.setWindowTitle('pyqtplot image array')



    return app, win
