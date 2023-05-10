from copy import deepcopy
import numpy as np
import pandas as pd

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg

import matplotlib.pyplot as plt # for stacked_epoch_slices_matplotlib_view(...)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # for stacked_epoch_slices_matplotlib_view(...)


from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui

from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget

""" 
These functions help render a vertically stacked column of subplots that represent (potentially non-contiguous) slices of a time range. 




"""

# ==================================================================================================================== #
# Stacked Epoch Slices View                                                                                            #
# ==================================================================================================================== #
def stacked_epoch_basic_setup(epoch_slices, epoch_labels=None, name='stacked_epoch_slices_view', plot_function_name='Stacked Epoch Slices View - PlotItem Version', single_plot_fixed_height=100.0, debug_test_max_num_slices=70, single_plot_fixed_width=200.0, debug_test_max_num_variants=64, debug_print=False):
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
    params.single_plot_fixed_height = single_plot_fixed_height
    params.all_plots_height = float(params.active_num_slices) * float(params.single_plot_fixed_height)

    if (epoch_labels is None) or (epoch_labels == []):
        # Build defaults for the plots
        epoch_labels = [f'epoch[{a_slice_idx}]' for a_slice_idx in np.arange(num_slices)]
    else:
        assert len(epoch_labels) == params.num_slices, f"len(epoch_labels): {len(epoch_labels)} != params.num_slices: {params.num_slices}"

    params.epoch_labels = epoch_labels
    
    plots_data.epoch_slices = epoch_slices
    
    return params, plots_data, plots, ui

# ==================================================================================================================== #
# PyQtGraph-based Versions                                                                                             #
# ==================================================================================================================== #
@function_attributes(short_name='stacked_epoch_slices_view', tags=['display','slices','stacked', 'scrollable'], input_requires=[], output_provides=[], uses=['stacked_epoch_basic_setup', 'build_scrollable_graphics_layout_widget_ui'], used_by=[], creation_date='2023-03-29 18:01')
def stacked_epoch_slices_view(epoch_slices, position_times_list, position_traces_list, epoch_description_list, name='stacked_epoch_slices_view', debug_print=False):
    """ I think this version displays line plots of the position traces, but it's not explicitly used anywhere that I know of.

    position_times_list: list of timestamps
    position_traces_list: list of traces to be plotted. Must have the same first dimension as timestamps
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


        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import stacked_epoch_slices_view

        stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name='stacked_epoch_slices_view_laps')
        params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers

    """
    plot_function_name = 'Stacked Epoch Slices View - PlotItem Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_description_list, name=name, plot_function_name=plot_function_name, debug_print=debug_print)
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
    """ The viewbox version - not primarily used
    
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
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_description_list, name=name, plot_function_name=plot_function_name, debug_print=debug_print)
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



# ==================================================================================================================== #
# matplotlib-based versions                                                                                            #
# ==================================================================================================================== #

# Pieces of plotting for pagination __________________________________________________________________________________ #
def _pagination_helper_plot_single_epoch_slice(curr_ax, params, plots_data, plots, ui, a_slice_idx, is_first_setup=True, debug_print=False):
    """ plots the data corresponding to `a_slice_idx` on the provided axes (`curr_ax`) 
    
    is_first_setup is True by default, and performs the axes initialization such as setting non-changing labels and such
    when calling to update the plot, set is_first_setup=False to skip this initialization and it will only update the stuff the changes every time
    
    """
    if debug_print:
        print(f'a_slice_idx: {a_slice_idx}')
    a_slice_start_t = plots_data.epoch_slices[a_slice_idx, 0]
    a_slice_end_t = plots_data.epoch_slices[a_slice_idx, 1]
    a_slice_label = params.epoch_labels[a_slice_idx]
    if debug_print:
        print(f'a_slice_start_t: {a_slice_start_t}, a_slice_end_t: {a_slice_end_t}, a_slice_label: {a_slice_label}')
    curr_ax.set_xlim(*plots_data.epoch_slices[a_slice_idx,:])
    
    if is_first_setup:
        curr_ax.tick_params(labelleft=False, labelbottom=True)
        curr_ax.set_title('') # remove the title

    if is_first_setup or (not plots.has_attr('secondary_yaxes')): # if it were a plain dict, `hasattr(plots, 'secondary_yaxes')` would be correct. But for a DynamicParameters object I need to do 
        plots.secondary_yaxes = {} # initialize to an empty dictionary

    # Left side y-label for the start time
    curr_ax.set_ylabel(f'{a_slice_label}\n{a_slice_start_t:.2f}') # format to two decimal places
    
    ## Add the right-aligned axis
    # From http://notes.brooks.nu/2008/03/plotting-on-left-and-right-axis-simulateously-using-matplotlib-and-numpy
    # Create right axis and plots.  It is the frameon=False that makes this plot transparent so that you can see the left axis plot that will be underneath it. The sharex option causes
    # if is_first_setup:
    secax_y = plots.secondary_yaxes.get(curr_ax, None) # get the existing one or create one
    if secax_y is None:
        secax_y = curr_ax.secondary_yaxis('right', functions=None)
        plots.secondary_yaxes[curr_ax] = secax_y # set the secondary axis for this curr_ax
        
    assert secax_y is not None
    secax_y.set_ylabel(f'{a_slice_end_t:.2f}')
    if is_first_setup:
        secax_y.tick_params(labelleft=False, labelbottom=False, labelright=False) # Turn off all ticks for the secondary axis
    # Do I need to save this temporary axes? No, it appears that's not needed


# Helper Figure/Plots Builders _______________________________________________________________________________________ #
def stacked_epoch_slices_matplotlib_build_view(epoch_slices, name='stacked_epoch_slices_matplotlib_subplots_laps', plot_function_name=None, epoch_labels=None, single_plot_fixed_height=100.0, debug_test_max_num_slices=127, debug_print=False):
    """ Builds a matplotlib figure view with empty subplots that can be plotted after the fact by iterating through plots.axs
        
    epoch_description_list: list of length 
    
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import stacked_epoch_slices_matplotlib_build_view
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
    if plot_function_name is None:
        plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB subplots Version'
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, single_plot_fixed_height=single_plot_fixed_height, debug_print=debug_print)
    # plots.figure_id = 'stacked_epoch_slices_matplotlib'    
    plots.figure_id = plots.name # copy the name as the figure_id
    
    ## Create the main figure and plot axes:
    
    ## Basic Matplotlib Version:
    # plots.fig, plots.axs = plt.subplots(num=plots.figure_id, ncols=1, nrows=params.active_num_slices, figsize=(15,15), clear=True, sharex=False, sharey=False, constrained_layout=True)
    
    ## MatplotlibTimeSynchronizedWidget-embedded Version:
    ui.mw = MatplotlibTimeSynchronizedWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, scrollAreaContents_MinimumHeight=params.all_plots_height)
    plots.fig = ui.mw.getFigure()
    plots.fig.suptitle(plots.name)
    plots.axs = plots.fig.subplots(ncols=1, nrows=params.active_num_slices, sharex=False, sharey=False)
    if not isinstance(plots.axs, (list, tuple, np.ndarray)):
        # if it's not wrapped in a list-like container, wrap it (this should only happen when (params.active_num_slices == 1)
        print(f'type(plots.axs): {type(plots.axs)}: {plots.axs}')
        plots.axs = [plots.axs]

    for a_slice_idx, curr_ax in enumerate(plots.axs):
        _pagination_helper_plot_single_epoch_slice(curr_ax, params, plots_data, plots, ui, a_slice_idx=a_slice_idx, is_first_setup=True, debug_print=debug_print)

    ## Required only for MatplotlibTimeSynchronizedWidget-embedded version:
    ui.mw.draw()
    # ui.mw.ui.scrollAreaContentsWidget.setMinimumHeight(params.all_plots_height)
    ui.mw.show()
    
    return params, plots_data, plots, ui

def stacked_epoch_slices_matplotlib_build_insets_view(epoch_slices, name='stacked_epoch_slices_matplotlib_INSET_subplots_laps', plot_function_name=None, epoch_labels=None, debug_test_max_num_slices=12, debug_print=False):
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
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, debug_print=debug_print)

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
    
    ## Basic Matplotlib Version:
    # plots.fig, plots.parent_ax = plt.subplots(num=plots.figure_id, ncols=1, nrows=1, figsize=(15,15), clear=True, sharex=False, sharey=False, constrained_layout=True)

    ## MatplotlibTimeSynchronizedWidget-embedded Version:
    ui.mw = MatplotlibTimeSynchronizedWidget(size=(15,15), dpi=72, constrained_layout=True) # , clear=True
    plots.fig = ui.mw.getFigure()
    plots.parent_ax = plots.fig.subplots(ncols=1, nrows=1, sharex=False, sharey=False) # , figsize=(15,15), clear=True, constrained_layout=True

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
    
    ## Required only for MatplotlibTimeSynchronizedWidget-embedded version:
    ui.mw.draw()
    ui.mw.show()
    
    return params, plots_data, plots, ui


# ==================================================================================================================== #
# matplotview-based version                                                                                            #
# ==================================================================================================================== #

# from matplotview import view

# # Create a view! Turn axes 2 into a view of axes 1.

#  view(ax2, ax1)

#  # Modify the second axes data limits so we get a slightly zoomed out view

#  ax2.set_xlim(-5, 15)

#  ax2.set_ylim(-5, 15)



# ==================================================================================================================== #
# Paginated Versions                                                                                                   #
# ==================================================================================================================== #
from neuropy.core.epoch import Epoch

from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


from pyphoplacecellanalysis.External.pyqtgraph import QtCore
from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedFigureController



class DecodedEpochSlicesPaginatedFigureController(PaginatedFigureController):
    """2023-05-09 - Aims to refactor `plot_paginated_decoded_epoch_slices`, a series of nested functions, into a stateful class
        Builds a matplotlib Figure in a CustomMatplotlibWidget that displays paginated axes using a Paginator by creating a `PaginationControlWidget`
        Specifically uses `plot_rr_aclu`, not general
        
    2023-05-09 - Refactored from  `build_figure_and_control_widget_from_paginator`
    
    ## [X]: would have to add reuse of figure and ax to `plot_rr_aclu` as a minimum - 5 minutes
    ## [X] 2023-05-02 - would have to add the concept of the current page index, the next/previous/jump operations (that could be triggered by arrows in the GUI for example) - 30 minutes
    ## [X] 2023-05-02 - then would have to add something to hold the resultant fig, ax objects, the initial plot function, and then the plot update function. - 30 minutes
    ## [X] 2023-05-03 - sorting is currently done locally (by page) which isn't good, need to move out and make copies of all the variables.

    Usage:
    
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController
        _out_pagination_controller = DecodedEpochSlicesPaginationController.init_from_decoder_data(long_results_obj.active_filter_epochs, long_results_obj.all_included_filter_epochs_decoder_result, xbin=long_results_obj.original_1D_decoder.xbin, global_pos_df=global_session.position.df, a_name='TestDecodedEpochSlicesPaginationController', max_subplots_per_page=20)
        _out_pagination_controller
    """

    @property
    def paginator(self):
        """The paginator property."""
        return self.plots_data.paginator

    @property
    def current_page_idx(self):
        """The curr_page_index property."""
        return self.ui.mw.ui.paginator_controller_widget.current_page_idx

    @property
    def total_number_of_items_to_show(self):
        """The total number of items (subplots usually) to be shown across all pages)."""
        return self.paginator.nItemsToShow

    def __init__(self, params, plots_data, plots, ui, parent=None):
        super(DecodedEpochSlicesPaginatedFigureController, self).__init__(params, plots_data, plots, ui, parent=parent)

    @classmethod
    def init_from_decoder_data(cls, active_filter_epochs, filter_epochs_decoder_result, xbin, global_pos_df, a_name:str = 'DecodedEpochSlicesPaginationController', active_context=None, max_subplots_per_page=20, parent=None):
        new_obj = cls(params=VisualizationParameters(name=a_name), plots_data=RenderPlotsData(name=a_name), plots=RenderPlots(name=a_name), ui=PhoUIContainer(name=a_name), parent=parent)
        ## Real setup:
        new_obj.params.active_identifying_figure_ctx = active_context # set context before calling `plot_paginated_decoded_epoch_slices` which will set the rest of the properties
        new_obj.plot_paginated_decoded_epoch_slices(active_filter_epochs, filter_epochs_decoder_result, xbin, global_pos_df, max_subplots_per_page=max_subplots_per_page)
        ## TODO: context should still be there after setup
        return new_obj
    
    def configure(self, **kwargs):
        """ assigns and computes needed variables for rendering. """
        pass
        

    def initialize(self, **kwargs):
        """ sets up Figures """
        # self.fig, self.axs = plt.subplots(nrows=len(rr_replays))
        pass

    def update(self, **kwargs):
        """ called to specifically render data on the figure. """
        pass

    def on_close(self):
        """ called when the figure is closed. """
        pass

    @staticmethod
    def _subfn_helper_build_paginator(active_filter_epochs, filter_epochs_decoder_result, max_subplots_per_page, debug_print) -> Paginator:
        epoch_labels = filter_epochs_decoder_result.epoch_description_list.copy()
        if epoch_labels is None or len(epoch_labels) < active_filter_epochs.n_epochs:
            if 'label' not in active_filter_epochs._df.columns:
                active_filter_epochs._df['label'] = active_filter_epochs._df.index.to_numpy() # integer ripple indexing
            # active_filter_epochs.labels ?
            active_labels = active_filter_epochs._df['label'].to_numpy()
            # active_labels = np.arange(active_filter_epochs.n_epochs)
            epoch_labels = np.array([f"Epoch[{epoch_idx}]" for epoch_idx in active_labels])
            if debug_print:
                print(f'epoch_labels: {epoch_labels}')
            filter_epochs_decoder_result.epoch_description_list = epoch_labels.copy() # assign the new labels

        time_bin_containers = np.array(filter_epochs_decoder_result.time_bin_containers.copy())
        posterior_containers = filter_epochs_decoder_result.marginal_x_list

        # Provide a tuple or list containing equally sized sequences of items:
        ## Build Epochs:
        if isinstance(active_filter_epochs, pd.DataFrame):
            epochs_df = active_filter_epochs
        elif isinstance(active_filter_epochs, Epoch):
            epochs_df = active_filter_epochs.to_dataframe()
        else:
            raise NotImplementedError

        epoch_slices = epochs_df[['start', 'stop']].to_numpy()
        
        epoch_slices_paginator = Paginator.init_from_data((epoch_slices, epoch_labels, time_bin_containers, posterior_containers), max_num_columns=1, max_subplots_per_page=max_subplots_per_page, data_indicies=None, last_figure_subplots_same_layout=False)
        return epoch_slices_paginator

    @QtCore.pyqtSlot(int)
    def on_paginator_control_widget_jump_to_page(self, page_idx: int):
        """ Update: made to depend on self """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions # used in `plot_decoded_epoch_slices`
        
        # print(f'on_paginator_control_widget_jump_to_page(page_idx: {page_idx})')
        included_page_data_indicies, (curr_page_active_filter_epochs, curr_page_epoch_labels, curr_page_time_bin_containers, curr_page_posterior_containers) = self.plots_data.paginator.get_page_data(page_idx=page_idx)
        # print(f'\tincluded_page_data_indicies: {included_page_data_indicies}')
        # params, plots_data, plots, ui = stacked_epoch_slices_matplotlib_build_view(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, debug_print=debug_print)

        for i, curr_ax in enumerate(self.plots.axs):
            curr_slice_idxs = included_page_data_indicies[i]
            curr_epoch_slice = curr_page_active_filter_epochs[i]
            curr_time_bin_container = curr_page_time_bin_containers[i]
            curr_posterior_container = curr_page_posterior_containers[i]
            curr_time_bins = curr_time_bin_container.centers
            curr_posterior = curr_posterior_container.p_x_given_n
            curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D
            
            if self.params.debug_print:
                print(f'i : {i}, curr_posterior.shape: {curr_posterior.shape}')

            # Update the axes appropriately:
            _pagination_helper_plot_single_epoch_slice(curr_ax, self.params, self.plots_data, self.plots, self.ui, a_slice_idx=curr_slice_idxs, is_first_setup=False, debug_print=self.params.debug_print)

            _temp_fig, curr_ax = plot_1D_most_likely_position_comparsions(self.plots_data.global_pos_df, ax=curr_ax, time_window_centers=curr_time_bins, variable_name=self.params.variable_name, xbin=self.params.xbin,
                                                            posterior=curr_posterior,
                                                            active_most_likely_positions_1D=curr_most_likely_positions,
                                                            enable_flat_line_drawing=self.params.enable_flat_line_drawing, debug_print=self.params.debug_print)
            if _temp_fig is not None:
                self.plots.fig = _temp_fig
            
            curr_ax.set_xlim(*curr_epoch_slice)
            curr_ax.set_title(f'') # needs to be set to empty string '' because this is the title that appears above each subplot/slice
            # Update selections:
            self.perform_update_ax_selected_state(ax=curr_ax, is_selected=self.params.is_selected.get(curr_slice_idxs, False))

        # # Update selection (could also do just in above loop):
        # self.perform_update_selections()

        self.perform_update_titles_from_context(page_idx=page_idx, included_page_data_indicies=included_page_data_indicies, collision_prefix='_DecodedEpochSlices_plot_test', display_fn_name='plot_single_epoch_slice', plot_result_set='shared')

        self.ui.mw.draw()



    def plot_paginated_decoded_epoch_slices(self, active_filter_epochs, filter_epochs_decoder_result, xbin, global_pos_df, max_subplots_per_page=20, debug_print=False):
        """ 2023-05-08 - plots a paginated decoded_epoch_slices figure """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices #, _helper_update_decoded_single_epoch_slice_plot #, _subfn_update_decoded_epoch_slices

        active_filter_epochs = deepcopy(active_filter_epochs)
        filter_epochs_decoder_result = deepcopy(filter_epochs_decoder_result) # DecodedFilterEpochsResult
        
        self.params, self.plots_data, self.plots, self.ui = plot_decoded_epoch_slices(active_filter_epochs, filter_epochs_decoder_result, global_pos_df=global_pos_df, variable_name='lin_pos', xbin=xbin,
                                                                name='stacked_epoch_slices_long_results_obj', debug_print=False, debug_test_max_num_slices=max_subplots_per_page)
        
        self.params.debug_print = debug_print
        self.plots_data.paginator = self._subfn_helper_build_paginator(active_filter_epochs, filter_epochs_decoder_result, max_subplots_per_page, self.params.debug_print)  # assign the paginator


        ## Add the PaginationControlWidget
        self._subfn_helper_add_pagination_control_widget(self.plots_data.paginator, self.ui.mw, defer_render=False)

        ## Setup Selectability
        self._subfn_helper_setup_selectability()

        ## 2. Update:
        self.on_paginator_control_widget_jump_to_page(page_idx=0)
        _a_connection = self.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(self.on_paginator_control_widget_jump_to_page) # bind connection
        self.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection

