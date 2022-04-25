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

from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config
from pyphocorehelpers.plotting.pyqtplot_basic import pyqtplot_common_setup


# class PlotLocationIdentifier(object):
#     """docstring for PlotLocationIdentifier."""
#     def __init__(self, arg):
#         super(PlotLocationIdentifier, self).__init__()
#     arg



def _pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin = 2.0, debug_print=False):
    """ Returns the proper bounds for the image, and the proper x_range and y_range given the margin.
    Used by pyqtplot_plot_image_array(...) to plot binned data.

    Usage:

        # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
    
    """
    global_min_x = np.nanmin(xbin_edges)
    global_max_x = np.nanmax(xbin_edges)

    global_min_y = np.nanmin(ybin_edges)
    global_max_y = np.nanmax(ybin_edges)

    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y

    if debug_print:
        print(f'global_min_x: {global_min_x}, global_max_x: {global_max_x}, global_min_y: {global_min_y}, global_max_y: {global_max_y}\nglobal_width: {global_width}, global_height: {global_height}')
    # Get rect image extent in the form [x, y, w, h]:
    image_bounds_extent = [global_min_x, global_min_y, global_width, global_height]

    x_range = (global_min_x-margin, global_max_x+margin)
    y_range = (global_min_y-margin, global_max_y+margin)

    return image_bounds_extent, x_range, y_range
    

def pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, enable_LUT_Histogram=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Plots an array of images provided in 'images' argument
    images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
        or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
        
    Example:
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        app, win = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy)
        win.show()
    """
    root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'pyqtplot_plot_image_array: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget)
    
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map

    image_bounds_extent, x_range, y_range = _pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    
    # Compute Images:
    included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
    nMapsToShow = len(included_unit_indicies)

    # Paging Management: Constrain the subplots values to just those that you need
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
    page_idx = 0 # page_idx is zero here because we only have one page:

    for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
        # Need to convert to page specific:
        curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
        curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
        curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
        if debug_print:
            print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

        cell_idx = curr_included_unit_index
        curr_cell_identifier_string = f'Cell[{cell_idx}]'
        curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'

        image = np.squeeze(images[a_linear_index,:,:])
        # Pre-filter the data:
        image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        if drop_below_threshold is not None:
            image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy

        # Build the image item:
        img_item = pg.ImageItem(image=image, levels=(0,1))
        #     # Viewbox version:
        #     # vb = layout.addViewBox(lockAspect=False)
        #     # # Build the ImageItem (which I'm guessing is like pg.ImageView) to add the image
        #     # imv = pg.ImageItem() # Create it with the current image
        #     # vb.addItem(imv) # add the item to the view box: why do we need the wrapping view box?
        #     # vb.autoRange()
        
        # # plot mode:
        curr_plot = root_render_widget.addPlot(row=curr_row, col=curr_col, name=curr_plot_identifier_string, title=curr_cell_identifier_string)
        curr_plot.addItem(img_item)  # add ImageItem to PlotItem
        curr_plot.showAxes(True)
        # curr_plot.showGrid(True, True, 0.7)
        # curr_plot.setLabel('bottom', "Label to test offset")
        
        # # Overlay cell identifier text:
        # curr_label = pg.TextItem(f'Cell[{cell_idx}]', color=(230, 230, 230))
        # curr_label.setPos(30, 60)
        # curr_label.setParentItem(img_item)
        # # curr_plot.addItem(curr_label, ignoreBounds=True)
        # curr_plot.addItem(curr_label)

        # Update the image:
        img_item.setImage(image, rect=image_bounds_extent)
        img_item.setLookupTable(cmap.getLookupTable(nPts=256))

        # margin = 2.0
        # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
        curr_plot.setXRange(*x_range)
        curr_plot.setYRange(*y_range)

        # Interactive Color Bar:
        bar = pg.ColorBarItem(values= (0, 1), colorMap=cmap) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem(img_item, insert_in=curr_plot)

    # Post images loop:
    
    enable_show = False
    
    if parent_root_widget is not None:
        if enable_show:
            parent_root_widget.show()
        
        parent_root_widget.setWindowTitle('pyqtplot image array')

    # pg.exec()
    return app, parent_root_widget, root_render_widget



def pyqtplot_plot_image(xbin_edges, ybin_edges, image, enable_LUT_Histogram=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Single image plot using pyqtplot: 
    Holy crap! It actually works to plot the maze, and the adjustable slider works as well!
    
    # Example: test single image plot:
        curr_im = np.squeeze(active_one_step_decoder.ratemap.normalized_tuning_curves[0,:,:]) # (43, 63, 63)
        app, win, imv = pyqtplot_plot_image(active_one_step_decoder.xbin, active_one_step_decoder.ybin, curr_im)
        win.show()
    """
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    if app is None:
        app = pg.mkQApp("pyqtplot_plot_image Figure")
        
    if root_render_widget is None:
        if parent_root_widget is None:
            # Create window to hold the image:
            parent_root_widget = QtGui.QMainWindow()
            parent_root_widget.resize(800,800)
        
        # Build a single image view to display the image:
        root_render_widget = pg.ImageView()
        parent_root_widget.setCentralWidget(root_render_widget)
        # imv.setImage(image, xvals=np.linspace(1., 3., data.shape[0]))
        parent_root_widget.show()
        parent_root_widget.setWindowTitle('pyqtplot image')

    ## Display the data and assign each frame a time value from 1.0 to 3.0
    root_render_widget.setImage(image, xvals=xbin_edges)
    # Set the color map:
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
    root_render_widget.setColorMap(cmap)
    
    # if enable_LUT_Histogram:
    #     lut = pg.HistogramLUTItem(orientation="horizontal")
    #     imv.addItem(lut)
    #     imv.setLookupTable(lut, autoLevel=True)
    #     h = imv.getHistogram()
    #     lut.plot.setData(*h)

    # bar = pg.ColorBarItem( values= (0, 20_000), cmap=cm ) # prepare interactive color bar
    # Have ColorBarItem control colors of img and appear in 'plot':
    # bar.setImageItem(image, insert_in=imv) 

    return app, parent_root_widget, root_render_widget
 
 
 