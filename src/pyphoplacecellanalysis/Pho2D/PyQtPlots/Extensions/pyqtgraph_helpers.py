import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


def pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin = 2.0, debug_print=False):
    """ Returns the proper bounds for the image, and the proper x_range and y_range given the margin.
    Used by pyqtplot_plot_image_array(...) to plot binned data.

    Usage:
    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
    

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
    
    
def build_root_graphics_layout_widget_ui(name, window_title=None, ui=None):
    """ Updates or builds the ui properties to display a GraphicsLayoutWidget with scrollable rows:
    Usage:
    ## Build non-scrollable UI version:
    ui = build_root_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    
    """
    if ui is None:
        ui = PhoUIContainer(name=name)
        ui.connections = PhoUIContainer(name=name)
        
    if window_title is None:
        window_title = name
    
    ## Plot Version:
    ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
    ui.graphics_layout.setWindowTitle(window_title)
    ui.graphics_layout.resize(1000, 800)
    # lw.ci.setBorder((50, 50, 100))
    return ui


def build_scrollable_graphics_layout_widget_ui(name, window_title=None, ui=None):
    """ Updates or builds the ui properties to display a GraphicsLayoutWidget with scrollable rows:
    Usage:
    ## Build scrollable UI version:
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ui.rootWindow.show()
    
    """
    if ui is None:
        ui = PhoUIContainer(name=name)
        ui.connections = PhoUIContainer(name=name)
        
    if window_title is None:
        window_title = name
    
    ui.rootWindow = QtWidgets.QMainWindow()
    ui.rootWindow.resize(1000, 800)

    ui.graphics_layout = pg.GraphicsLayoutWidget()
    ui.graphics_layout.setFixedWidth(1000)
    ui.graphics_layout.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    ui.scrollAreaWidget = QtWidgets.QScrollArea()
    ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    ui.scrollAreaWidget.setWidget(ui.graphics_layout)
    ui.rootWindow.setCentralWidget(ui.scrollAreaWidget)
    ui.rootWindow.setWindowTitle(window_title)
    
    # ui.rootWindow.show()
    return ui



    
def stacked_epoch_slices_view(epoch_slices, position_times_list, position_traces_list, epoch_description_list, name='stacked_epoch_slices_view', debug_print=False):
    """ 
    
    epoch_description_list: list of length 
    
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_slices_view

    stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name='stacked_epoch_slices_view_laps')
    params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers

    """
    num_slices = np.shape(epoch_slices)[0]
    assert len(epoch_description_list) == num_slices
    assert len(position_times_list) == num_slices
    assert len(position_times_list) == num_slices
    
    ## Init containers:
    params = VisualizationParameters(name=name)
    plots_data = RenderPlotsData(name=name)
    plots = RenderPlots(name=name)
    ui = PhoUIContainer(name=name)
    ui.connections = PhoUIContainer(name=name)    

    params.name = name
    params.window_title = 'Stacked Epoch Slices View - PlotItem Version'
    params.num_slices = num_slices
    
    params._debug_test_max_num_slices = 70
    params.active_num_slices = min(num_slices, params._debug_test_max_num_slices)

    params.single_plot_fixed_height = 200.0
    params.all_plots_height = float(params.active_num_slices) * float(params.single_plot_fixed_height)
    
    ## Plot Version:
    ## Build non-scrollable UI version:
    # ui = build_root_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ## Build scrollable UI version:
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ui.rootWindow.show()
    
    for a_slice_idx in np.arange(params.active_num_slices):
        if debug_print:
            print(f'a_slice_idx: {a_slice_idx}')
        
        ## Get values:
        curr_row = a_slice_idx
        curr_col = 0
        curr_plot_identifier_string = f'{params.window_title} - item[{curr_row}][{curr_col}]'
        
        if epoch_description_list is not None:
            curr_name = epoch_description_list[a_slice_idx]
        else:
            # curr_name = f'a_slice_idx: {a_slice_idx}'
            curr_name = f'[slice_idx: {a_slice_idx}][row: {curr_row}][col: {curr_col}]'
    
        curr_epoch_identifier_string = curr_name    
        curr_slice_t_start, curr_slice_t_end = epoch_slices[a_slice_idx, :]
        times = position_times_list[a_slice_idx] # (173,)
        values = position_traces_list[a_slice_idx] # (2, 173)    
        x_values = np.squeeze(values[0, :])
        y_values = np.squeeze(values[1, :])
        # lw.addLabel(curr_name)
        
        ## Build main plot:
        # imi = pg.ImageItem(bar_data)
        # imi.setLookupTable(cm.getLookupTable(alpha=True))
        
        # # plot mode:
        curr_plot = ui.graphics_layout.addPlot(row=curr_row, col=curr_col, title=curr_epoch_identifier_string) # , name=curr_plot_identifier_string 
        curr_plot.setObjectName(curr_plot_identifier_string)
        # curr_plot.showAxes(True)
        curr_plot.showAxes(True, showValues=(True, True, True, False)) # showValues=(left: True, bottom: True, right: False, top: False) # , size=10       
        curr_plot.hideButtons() # Hides the auto-scale button
        curr_plot.setDefaultPadding(0.0)  # plot without padding data range
        curr_plot.setMouseEnabled(x=False, y=False)
        curr_plot.setMenuEnabled(enableMenu=False)
            
        # curr_plot.showAxis('right')
        # curr_plot.getAxis('left').setLabel('left axis label')
        # curr_plot.getAxis('bottom').setLabel('bottom axis label')
        # curr_plot.getAxis('right').setLabel('right axis label')
        
        curr_plot.getAxis('left').setLabel(f'Epoch[{a_slice_idx}]: {curr_slice_t_start}')
        curr_plot.getAxis('bottom').setLabel('t')
        curr_plot.getAxis('right').setLabel(f'Epoch[{a_slice_idx}]: {curr_slice_t_end}')
        
        
        
            
        curr_plotItem = curr_plot.plot(times, x_values, defaultPadding=0.0)
        # curr_plot.addItem(img_item, defaultPadding=0.0)  # add ImageItem to PlotItem

        ## Local plots_data and plots:
        local_plots_data = RenderPlotsData(name=curr_name)
        local_plots_data.times = times.copy()
        local_plots_data.x_values = x_values.copy()
        local_plots_data.y_values = y_values.copy()
                
        local_plots = RenderPlots(name=curr_name)
        local_plots.plot = curr_plot
        local_plots.plot_item = curr_plotItem
        
        # Set global/output variables
        plots_data[curr_name] = local_plots_data                
        plots[curr_name] = local_plots
        plots[curr_name].mainPlotItem = curr_plotItem
        
        ui.graphics_layout.nextRow()
    
    ui.graphics_layout.setFixedHeight(params.all_plots_height)
    
    return params, plots_data, plots, ui
    
    
# def build_vertically_scrollable_graphics_area():
#     """ copied from pyphoplacecellanalysis.External.pyqtgraph.examples.colorMaps example """
#     app = pg.mkQApp()
    
#     ui = PhoUIContainer('')

#     ui.rootWindow = QtWidgets.QMainWindow()
#     ui.rootWindow.resize(1000,800)

#     ui.graphics_layout = pg.GraphicsLayoutWidget()
#     ui.graphics_layout.setFixedWidth(1000)
#     ui.graphics_layout.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

#     ui.scrollAreaWidget = QtWidgets.QScrollArea()
#     ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#     ui.scrollAreaWidget.setWidget(ui.graphics_layout)
#     ui.rootWindow.setCentralWidget(ui.scrollAreaWidget)
#     ui.rootWindow.setWindowTitle('pyqtgraph example: Color maps')
#     ui.rootWindow.show()

#     # bar_width = 32
#     # bar_data = pg.colormap.modulatedBarData(width=bar_width)

#     # num_bars = 0

#     # def add_heading(lw, name):
#     #     global num_bars
#     #     lw.addLabel('=== '+name+' ===')
#     #     num_bars += 1
#     #     lw.nextRow()

#     # def add_bar(lw, name, cm):
#     #     global num_bars
#     #     lw.addLabel(name)
#     #     imi = pg.ImageItem( bar_data )
#     #     imi.setLookupTable( cm.getLookupTable(alpha=True) )
#     #     vb = lw.addViewBox(lockAspect=True, enableMouse=False)
#     #     vb.addItem( imi )
#     #     num_bars += 1
#     #     lw.nextRow()

#     # # Run the setup:
#     # add_heading(lw, 'local color maps')
#     # list_of_maps = pg.colormap.listMaps()
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.swapcase() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name)
#     #     add_bar(lw, map_name, cm)

#     # add_heading(lw, 'Matplotlib import')
#     # list_of_maps = pg.colormap.listMaps('matplotlib')
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.lower() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name, source='matplotlib', skipCache=True)
#     #     if cm is not None:
#     #         add_bar(lw, map_name, cm)

#     # add_heading(lw, 'ColorCET import')
#     # list_of_maps = pg.colormap.listMaps('colorcet')
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.lower() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name, source='colorcet', skipCache=True)
#     #     if cm is not None:
#     #         add_bar(lw, map_name, cm)

#     # ui.graphics_layout.setFixedHeight(num_bars * (bar_width+5) )
#     # return ui, add_heading, add_bar
#     return ui


