import numpy as np

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMainAppWindowBase import PhoMainAppWindowBase # for pyqtplot_plot_image

import matplotlib.pyplot as plt # for stacked_epoch_slices_matplotlib_view(...)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # for stacked_epoch_slices_matplotlib_view(...)


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
        
        
    # image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    
        
    if root_render_widget is None:
        if parent_root_widget is None:
            # Create window to hold the image:
            
            # parent_root_widget = QtGui.QMainWindow()
            parent_root_widget = PhoMainAppWindowBase()
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
 
 
def stacked_epoch_basic_setup(epoch_slices, name='stacked_epoch_slices_view', plot_function_name='Stacked Epoch Slices View - PlotItem Version', debug_test_max_num_slices=70, debug_print=False):
    """ Builds the common setup/containers for all stacked-epoch type plots:
    
    epoch_description_list: list of length 
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_basic_setup
        plot_function_name = 'Stacked Epoch Slices View - Viewbox Version'
        params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, name=name, plot_function_name=plot_function_name, debug_print=debug_print)
    

    """
    num_slices = np.shape(epoch_slices)[0]
    
    ## Init containers:
    params = VisualizationParameters(name=name)
    plots_data = RenderPlotsData(name=name)
    plots = RenderPlots(name=name)
    ui = PhoUIContainer(name=name)
    ui.connections = PhoUIContainer(name=name)    

    params.name = name
    params.window_title = plot_function_name
    params.num_slices = num_slices
    
    params._debug_test_max_num_slices = debug_test_max_num_slices
    params.active_num_slices = min(num_slices, params._debug_test_max_num_slices)
    params.global_epoch_start_t = np.nanmin(epoch_slices[:, 0], axis=0)
    params.global_epoch_end_t = np.nanmax(epoch_slices[:, 1], axis=0)
    # params.global_epoch_start_t, params.global_epoch_end_t # (1238.0739798661089, 2067.4688883359777)
    params.single_plot_fixed_height = 200.0
    params.all_plots_height = float(params.active_num_slices) * float(params.single_plot_fixed_height)

    plots_data.epoch_slices = epoch_slices
    
    return params, plots_data, plots, ui


# ==================================================================================================================== #
# UI Building Helpers                                                                                                  #
# ==================================================================================================================== #
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

def build_scrollable_graphics_layout_widget_with_nested_viewbox_ui(name, window_title=None, ui=None):
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
    
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=window_title, ui=ui)
    ## Adds the root_viewbox to the graphics layout
    # ui.root_viewbox = ui.graphics_layout.addViewBox(enableMouse=False) # lockAspect=True
    
    # ui.root_viewbox = ui.graphics_layout.addViewBox(enableMouse=False, defaultPadding=0.0, enableMenu=False, border='r') # lockAspect=True
    # pg.mkColor('r')
    # ui.root_viewbox.setBackgroundColor('r')
    
    # ui.root_viewbox = ui.graphics_layout.addLayout(enableMouse=False, defaultPadding=0.0, enableMenu=False, border='r') # lockAspect=True
    
    ui.nested_graphics_layout = ui.graphics_layout.addLayout(border=(50,0,0))
    ui.nested_graphics_layout.setContentsMargins(10, 10, 10, 10)
    return ui





# ==================================================================================================================== #
# Stacked Epoch Slices View                                                                                            #
# ==================================================================================================================== #
def stacked_epoch_slices_view(epoch_slices, position_times_list, position_traces_list, epoch_description_list, name='stacked_epoch_slices_view', debug_print=False):
    """ 
    
    epoch_description_list: list of length 
    
    
    Usage:
        ## Laps 
        from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.LapsVisualizationMixin import LapsVisualizationMixin
        curr_position_df, lap_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(curr_active_pipeline.sess)
        lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in sess.laps.lap_id] # dataframes split for each ID:

        laps_position_times_list = [np.squeeze(lap_pos_df[['t']].to_numpy()) for lap_pos_df in lap_specific_position_dfs]
        laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]

        epochs = sess.laps.to_dataframe()
        epoch_slices = epochs[['start', 'stop']].to_numpy()
        epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in epochs[['lap_id','maze_id','lap_dir']].itertuples()]
        print(f'epoch_description_list: {epoch_description_list}')


        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_slices_view

        stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name='stacked_epoch_slices_view_laps')
        params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers

    """
    plot_function_name = 'Stacked Epoch Slices View - PlotItem Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, name=name, plot_function_name=plot_function_name, debug_print=debug_print)
    assert len(epoch_description_list) == params.num_slices
    assert len(position_times_list) == params.num_slices
    # assert len(position_traces_list) == params.num_slices
    
    ## Plot Version:
    ## Build non-scrollable UI version:
    # ui = build_root_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ## Build scrollable UI version:
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    _add_plot_target = ui.graphics_layout
    _layout_next_row_target_command = lambda x: ui.graphics_layout.nextRow()    
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



def stacked_epoch_slices_view_viewbox(epoch_slices, position_times_list, position_traces_list, epoch_description_list, name='stacked_epoch_slices_view_viewbox', debug_print=False):
    """ The viewbox version 
    
    epoch_description_list: list of length 
    
    
    Usage:
        ## Laps 
        from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.LapsVisualizationMixin import LapsVisualizationMixin
        curr_position_df, lap_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(curr_active_pipeline.sess)
        lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in sess.laps.lap_id] # dataframes split for each ID:

        laps_position_times_list = [np.squeeze(lap_pos_df[['t']].to_numpy()) for lap_pos_df in lap_specific_position_dfs]
        laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]

        epochs = sess.laps.to_dataframe()
        epoch_slices = epochs[['start', 'stop']].to_numpy()
        epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in epochs[['lap_id','maze_id','lap_dir']].itertuples()]
        print(f'epoch_description_list: {epoch_description_list}')


        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_slices_view

        stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name='stacked_epoch_slices_view_laps')
        params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers

    """
    plot_function_name = 'Stacked Epoch Slices View - Viewbox Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, name=name, plot_function_name=plot_function_name, debug_print=debug_print)
    assert len(epoch_description_list) == params.num_slices
    assert len(position_times_list) == params.num_slices
    # assert len(position_traces_list) == params.num_slices
    
    ## Build scrollable UI version with a nested viewbox: 
    ui = build_scrollable_graphics_layout_widget_with_nested_viewbox_ui(name, window_title=params.window_title, ui=ui)
    ui.rootWindow.show()
    
    # ui.root_viewbox.setBackgroundColor('r')
    # box2.setParentItem(box1)
    # box2.setPos(5, 5)
    # box2.setScale(0.2)

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
        
        curr_label = ui.nested_graphics_layout.addLabel(curr_name, row=a_slice_idx, col=1, angle=-90) # -90 makes the label vertical
                
        ## Add ViewBox
        # ui.root_viewbox.addItem(
    
        # curr_vb = ui.graphics_layout.addViewBox(row=a_slice_idx, col=1)
        # curr_vb = ui.graphics_layout.addViewBox(enableMouse=False, defaultPadding=0, enableMenu=False, border='w') # lockAspect=True, parent=ui.root_viewbox
        

        # curr_vb = ui.nested_graphics_layout.addViewBox(enableMouse=False, defaultPadding=0, enableMenu=False, border='w') # lockAspect=True, parent=ui.root_viewbox
        curr_vb = ui.nested_graphics_layout.addViewBox(row=a_slice_idx, col=2, enableMouse=False, defaultPadding=0, enableMenu=False, border='w') # lockAspect=True, parent=ui.root_viewbox
        
        # curr_vb.showAxRect(True)
        # curr_vb.setMouseEnabled(x=False, y=True)
        curr_vb.enableAutoRange(x=False, y=True)
        # curr_vb.setXRange(curr_slice_t_start, curr_slice_t_end) # Local Data Series (Slice t_start, t_end)
        curr_vb.setXRange(params.global_epoch_start_t, params.global_epoch_end_t) # Local Data Series (Slice t_start, t_end)
        curr_vb.setAutoVisible(x=False, y=True)
        # curr_vb.setLimits(xMin=curr_slice_t_start, xMax=curr_slice_t_end, 
        #          minXRange=20, maxXRange=500, 
        #          yMin=-10, yMax=10,
        #          minYRange=1, maxYRange=10)
        
        
        # curr_plotItem = curr_plot.plot(times, x_values, defaultPadding=0.0)
        # curr_plot.addItem(img_item, defaultPadding=0.0)  # add ImageItem to PlotItem

        curr_plotItem = pg.PlotDataItem(times, x_values)
        
        curr_vb.addItem(curr_plotItem) # PlotItem

        ## Local plots_data and plots:
        local_plots_data = RenderPlotsData(name=curr_name)
        local_plots_data.times = times.copy()
        local_plots_data.x_values = x_values.copy()
        local_plots_data.y_values = y_values.copy()
                
        local_plots = RenderPlots(name=curr_name)
        local_plots.viewbox = curr_vb
        local_plots.plot_item = curr_plotItem
        
        # Set global/output variables
        plots_data[curr_name] = local_plots_data                
        plots[curr_name] = local_plots
        plots[curr_name].mainPlotItem = curr_plotItem
        
        # ui.graphics_layout.nextRow()
        ui.nested_graphics_layout.nextRow()
    
    ui.nested_graphics_layout.setFixedHeight(params.all_plots_height)
    ui.graphics_layout.setFixedHeight(params.all_plots_height)
    
    return params, plots_data, plots, ui



def stacked_epoch_slices_matplotlib_build_view(epoch_slices, name='stacked_epoch_slices_matplotlib_subplots_laps', plot_function_name=None, debug_test_max_num_slices=12, debug_print=False):
    """ Builds a matplotlib figure view with empty subplots that can be plotted after the fact by iterating through plots.axs
        
    epoch_description_list: list of length 
    
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_slices_matplotlib_build_view
        plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB subplots Version'
        params, plots_data, plots, ui = stacked_epoch_slices_matplotlib_build_view(epoch_slices, name='stacked_epoch_slices_matplotlib_subplots_laps', plot_function_name=plot_function_name, debug_test_max_num_slices=12, debug_print=False)

        ## Test Plotting just a single dimension of the 2D posterior:
        pho_custom_decoder = active_one_step_decoder
        active_posterior = pho_custom_decoder.p_x_given_n
        # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
        marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
        marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        for i, curr_ax in enumerate(plots.axs):
            plots.fig, curr_ax = plot_1D_most_likely_position_comparsions(sess.position.to_dataframe(), ax=curr_ax, time_window_centers=pho_custom_decoder.active_time_window_centers, xbin=pho_custom_decoder.xbin,
                                                            posterior=marginal_posterior_x,
                                                            active_most_likely_positions_1D=pho_custom_decoder.most_likely_positions[:,0].T,
                                                            enable_flat_line_drawing=True, debug_print=False)
            curr_ax.set_xlim(*plots_data.epoch_slices[i,:])
    """
    ## Inset Subplots Version:
    if plot_function_name is not None:
        plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB subplots Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, debug_print=debug_print)
    plots.figure_id = 'stacked_epoch_slices_matplotlib'
    plots.fig, plots.axs = plt.subplots(num=plots.figure_id, ncols=1, nrows=params.active_num_slices, figsize=(15,15), clear=True, sharex=False, sharey=False, constrained_layout=True)
    plots.fig.suptitle(plots.name)
    
    for a_slice_idx, curr_ax in enumerate(plots.axs):
        if debug_print:
            print(f'a_slice_idx: {a_slice_idx}')
        
        ## Get values:
        # Create inset in data coordinates using ax.transData as transform
        curr_ax.set_xlim(*plots_data.epoch_slices[a_slice_idx,:])
        curr_ax.tick_params(labelleft=False, labelbottom=True)
        curr_ax.set_title('') # remove the title
    
    return params, plots_data, plots, ui





def stacked_epoch_slices_matplotlib_build_insets_view(epoch_slices, name='stacked_epoch_slices_matplotlib_INSET_subplots_laps', plot_function_name=None, debug_test_max_num_slices=12, debug_print=False):
    """ Builds a matplotlib figure view with empty subplots that can be plotted after the fact by iterating through plots.axs
        
    epoch_description_list: list of length 
    
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import stacked_epoch_basic_setup, stacked_epoch_slices_matplotlib_view

        plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB INSET SUBPLOTS Version'
        params, plots_data, plots, ui = stacked_epoch_slices_matplotlib_build_insets_view(epoch_slices, name='stacked_epoch_slices_matplotlib_INSET_subplots_laps', plot_function_name=plot_function_name, debug_test_max_num_slices=12, debug_print=False)

        ## Test Plotting just a single dimension of the 2D posterior:
        pho_custom_decoder = active_one_step_decoder
        active_posterior = pho_custom_decoder.p_x_given_n
        # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
        marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
        marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

        for a_slice_idx, curr_ax in enumerate(plots.axs):
            plots.fig, curr_ax = plot_1D_most_likely_position_comparsions(sess.position.to_dataframe(), ax=curr_ax, time_window_centers=pho_custom_decoder.active_time_window_centers, xbin=pho_custom_decoder.xbin,
                                                                posterior=marginal_posterior_x,
                                                                active_most_likely_positions_1D=pho_custom_decoder.most_likely_positions[:,0].T,
                                                                enable_flat_line_drawing=True,  debug_print=False)
            curr_ax.set_xlim(*plots_data.epoch_slices[a_slice_idx,:])
    """
    ## Inset Subplots Version:
    # debug_print = False
    if plot_function_name is not None:
        plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB INSET SUBPLOTS Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, debug_print=debug_print)

    global_xrange = (params.global_epoch_start_t, params.global_epoch_end_t)
    global_xduration = params.global_epoch_end_t - params.global_epoch_start_t
    epoch_durations = np.squeeze(np.diff(plots_data.epoch_slices, axis=1))
    # epoch_durations
    epoch_slices_max_duration = np.max(epoch_durations) # 28.95714869396761
    epoch_slice_relative_durations = epoch_durations / epoch_slices_max_duration # computes the relative duration/xlength foe ach epoch slice as a range from 0.0-1.0 to determine relative sizes to parent
    # epoch_slice_relative_durations
    inset_plot_heights = np.full((params.active_num_slices,), (100.0 / float(params.active_num_slices))) # array([11.1111, 11.1111, 11.1111, 11.1111, 11.1111, 11.1111, 11.1111, 11.1111, 11.1111])
    # inset_plot_heights
    inset_plot_widths = epoch_slice_relative_durations * 100.0 # convert to percent width of parent
    # inset_plot_widths.shape

    plots.figure_id = 'stacked_epoch_slices_INSET_matplotlib'
    plots.fig, plots.parent_ax = plt.subplots(num=plots.figure_id, ncols=1, nrows=1, figsize=(15,15), clear=True, sharex=False, sharey=False, constrained_layout=True)
    plots.axs = [] # an empty list of core axes
    plots.fig.suptitle(plots.name)
    plots.parent_ax.set(xlim=(0.0, epoch_slices_max_duration), ylim=(0, float(params.active_num_slices)))

    for a_slice_idx in np.arange(params.active_num_slices):
        if debug_print:
            print(f'a_slice_idx: {a_slice_idx}')
        
        ## Get values:
        if debug_print:
            print(f'plotting axis[{a_slice_idx}]: {a_slice_idx}')
        # Create inset in data coordinates using ax.transData as transform
        curr_percent_width="100%"
        curr_percent_height="95%"
        # curr_percent_width=f"{inset_plot_widths[i]}%"
        # curr_percent_height=f"{inset_plot_heights[i]}%"
        if debug_print:
            print(f'\tcurr_percent_width: {curr_percent_width}, curr_percent_height: {curr_percent_height}')
        curr_ax = inset_axes(plots.parent_ax, width=curr_percent_width, height=curr_percent_height,
                            bbox_transform=plots.parent_ax.transData, bbox_to_anchor=(0.0, float(a_slice_idx), epoch_durations[a_slice_idx], 1.0), # [left, bottom, width, height]
                            loc='lower left', borderpad=1.0)
        
        curr_ax.set_xlim(*plots_data.epoch_slices[a_slice_idx,:])
        curr_ax.tick_params(labelleft=False, labelbottom=False)
        curr_ax.set_title('') # remove the title
        # Appends:
        plots.axs.append(curr_ax)
    
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


