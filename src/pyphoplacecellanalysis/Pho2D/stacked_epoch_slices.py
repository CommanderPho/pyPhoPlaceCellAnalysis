import builtins
from copy import deepcopy
from typing import Dict, List, Optional
from matplotlib.backend_bases import MouseButton
from neuropy.utils.result_context import IdentifyingContext
import numpy as np
import pandas as pd
from nptyping import NDArray
from attrs import define, field, Factory
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

import matplotlib.pyplot as plt # for stacked_epoch_slices_matplotlib_view(...)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # for stacked_epoch_slices_matplotlib_view(...)

from pyphoplacecellanalysis.External.pyqtgraph import QtCore

from neuropy.core.epoch import Epoch
from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset
from neuropy.utils.matplotlib_helpers import set_ax_emphasis_color


from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.indexing_helpers import safe_find_index_in_list
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.exception_helpers import ExceptionPrintingContext

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import PhoDockAreaContainingWindow # for PhoPaginatedMultiDecoderDecodedEpochsWindow
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget, PaginationControlWidgetState
from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedFigureController
from neuropy.core.user_annotations import UserAnnotationsManager # used in `interactive_good_epoch_selections`
from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject # used in `interactive_good_epoch_selections`, `PhoPaginatedMultiDecoderDecodedEpochsWindow`



""" 
These functions help render a vertically stacked column of subplots that represent (potentially non-contiguous) slices of a time range. 




"""

# ==================================================================================================================== #
# Stacked Epoch Slices View                                                                                            #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['helper', 'common', 'setup', 'axes', 'figure', 'stacked'], input_requires=[], output_provides=[], uses=[], used_by=['stacked_epoch_slices_view'], creation_date='2023-03-28 00:00', related_items=[])
def stacked_epoch_basic_setup(epoch_slices, epoch_labels=None, name='stacked_epoch_slices_view', plot_function_name='Stacked Epoch Slices View - PlotItem Version', single_plot_fixed_height=100.0, debug_test_max_num_slices=70, single_plot_fixed_width=200.0, debug_test_max_num_variants=64, should_use_MatplotlibTimeSynchronizedWidget=True, debug_print=False, **additional_params_kwargs):
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


    default_params_dict = {'name': name, 'window_title': plot_function_name,
    'num_slices': num_slices, '_debug_test_max_num_slices': debug_test_max_num_slices, 'active_num_slices': min(num_slices, debug_test_max_num_slices),
        'global_epoch_start_t': np.nanmin(epoch_slices[:, 0], axis=0), 'global_epoch_end_t': np.nanmax(epoch_slices[:, 1], axis=0),
        'single_plot_fixed_height': single_plot_fixed_height, 'all_plots_height': float(min(num_slices, debug_test_max_num_slices)) * float(single_plot_fixed_height),
        'should_use_MatplotlibTimeSynchronizedWidget': should_use_MatplotlibTimeSynchronizedWidget, **additional_params_kwargs}

    params.update(**default_params_dict)

    # params.name = name
    # params.window_title = plot_function_name
    # params.num_slices = num_slices
    
    # params._debug_test_max_num_slices = debug_test_max_num_slices
    # params.active_num_slices = min(num_slices, params._debug_test_max_num_slices)
    # params.global_epoch_start_t = np.nanmin(epoch_slices[:, 0], axis=0)
    # params.global_epoch_end_t = np.nanmax(epoch_slices[:, 1], axis=0)
    # # params.global_epoch_start_t, params.global_epoch_end_t # (1238.0739798661089, 2067.4688883359777)
    # params.single_plot_fixed_height = single_plot_fixed_height
    # params.all_plots_height = float(params.active_num_slices) * float(params.single_plot_fixed_height)
    # params.should_use_MatplotlibTimeSynchronizedWidget = should_use_MatplotlibTimeSynchronizedWidget


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

    if not plots.has_attr('secondary_yaxes'): # if it were a plain dict, `hasattr(plots, 'secondary_yaxes')` would be correct. But for a DynamicParameters object I need to do 
        plots.secondary_yaxes = {} # initialize to an empty dictionary

    # Left side y-label for the start time
    should_disable_ylabel = params.get('disable_y_label', False)
    if not should_disable_ylabel:
        curr_ax.set_ylabel(f'{a_slice_label}\n{a_slice_start_t:.3f}') # format to two decimal places
    else:
        curr_ax.set_ylabel('')

    ## Add the right-aligned axis
    # From http://notes.brooks.nu/2008/03/plotting-on-left-and-right-axis-simulateously-using-matplotlib-and-numpy
    # Create right axis and plots.  It is the frameon=False that makes this plot transparent so that you can see the left axis plot that will be underneath it. The sharex option causes
    # if is_first_setup:
    secax_y = plots.secondary_yaxes.get(curr_ax, None) # get the existing one or create one
    if secax_y is None:
        secax_y = curr_ax.secondary_yaxis('right', functions=None)
        plots.secondary_yaxes[curr_ax] = secax_y # set the secondary axis for this curr_ax
        
    assert secax_y is not None
    if not should_disable_ylabel:
        secax_y.set_ylabel(f'{a_slice_end_t:.3f}')
    else:
        secax_y.set_ylabel('')
    secax_y.tick_params(labelleft=False, labelbottom=False, labelright=False) # Turn off all ticks for the secondary axis
    # Do I need to save this temporary axes? No, it appears that's not needed


# Helper Figure/Plots Builders _______________________________________________________________________________________ #
@function_attributes(short_name=None, tags=['epoch', 'stacked', 'matplotlib', 'TODO:PERFORMANCE'], input_requires=[], output_provides=[], uses=['stacked_epoch_basic_setup', 'MatplotlibTimeSynchronizedWidget', '_pagination_helper_plot_single_epoch_slice'], used_by=['plot_decoded_epoch_slices'], creation_date='2023-05-30 10:05', related_items=[])
def stacked_epoch_slices_matplotlib_build_view(epoch_slices, name='stacked_epoch_slices_matplotlib_subplots_laps', plot_function_name=None, epoch_labels=None,
                                                single_plot_fixed_height=100.0, debug_test_max_num_slices=127,
                                                size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, should_use_MatplotlibTimeSynchronizedWidget=True,
                                                debug_print=False, **kwargs):
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
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, should_use_MatplotlibTimeSynchronizedWidget=should_use_MatplotlibTimeSynchronizedWidget, debug_test_max_num_slices=debug_test_max_num_slices, single_plot_fixed_height=single_plot_fixed_height, debug_print=debug_print)
    # plots.figure_id = 'stacked_epoch_slices_matplotlib'
    plots.figure_id = plots.name # copy the name as the figure_id
    
    ## Create the main figure and plot axes:
    if not params.should_use_MatplotlibTimeSynchronizedWidget:
        ## Basic Matplotlib Version:
        plots.fig, plots.parent_ax = plt.subplots(num=plots.figure_id, ncols=1, nrows=1, figsize=(15,15), clear=True, sharex=False, sharey=False, constrained_layout=True)
        ui.mw = None
    else:
        ## MatplotlibTimeSynchronizedWidget-embedded Version:
        ui.mw = MatplotlibTimeSynchronizedWidget(size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure, scrollAreaContents_MinimumHeight=params.all_plots_height, name=name, plot_function_name=plot_function_name, **kwargs)
        plots.fig = ui.mw.getFigure()

    plots.fig.suptitle(plots.name)

    # plots.axs = plots.fig.subplots(ncols=1, nrows=params.active_num_slices, sharex=False, sharey=False)

    ## Begin to support a separate column for showing labels
    subplots_kwargs = dict()
    params.setdefault('should_use_separate_details_column', False)
    if params.should_use_separate_details_column:
        params.n_active_columns = 2
        subplots_kwargs.update(dict(gridspec_kw={'width_ratios': [1, 2]}))

    else:
        params.n_active_columns = 1
        
    plots.axs = plots.fig.subplots(ncols=params.n_active_columns, nrows=params.active_num_slices, sharex=False, sharey=False, **subplots_kwargs)


    if not isinstance(plots.axs, (list, tuple, np.ndarray)):
        # if it's not wrapped in a list-like container, wrap it (this should only happen when (params.active_num_slices == 1)
        print(f'type(plots.axs): {type(plots.axs)}: {plots.axs}')
        plots.axs = [plots.axs]

    for a_slice_idx, curr_ax in enumerate(plots.axs):
        _pagination_helper_plot_single_epoch_slice(curr_ax, params, plots_data, plots, ui, a_slice_idx=a_slice_idx, is_first_setup=True, debug_print=debug_print)

    if params.should_use_MatplotlibTimeSynchronizedWidget:
        ## Required only for MatplotlibTimeSynchronizedWidget-embedded version:
        ui.mw.draw() #TODO 2023-07-06 15:08: - [ ] TODO: PERFORMANCE - uneeded-draw
        # ui.mw.ui.scrollAreaContentsWidget.setMinimumHeight(params.all_plots_height)
        ui.mw.show()
        
    # It seems that the title must not be updated until after ui.mw.show() is called.
    return params, plots_data, plots, ui


@function_attributes(short_name=None, tags=['matplotlib', 'plot', 'figure', 'variant', 'helper'], input_requires=[], output_provides=[], uses=['stacked_epoch_basic_setup', 'MatplotlibTimeSynchronizedWidget'], used_by=[], creation_date='2023-05-30 10:06', related_items=[])
def stacked_epoch_slices_matplotlib_build_insets_view(epoch_slices, name='stacked_epoch_slices_matplotlib_INSET_subplots_laps', plot_function_name=None, epoch_labels=None,
                                                        single_plot_fixed_height=100.0, debug_test_max_num_slices=12,
                                                        size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, should_use_MatplotlibTimeSynchronizedWidget=True,
                                                        debug_print=False, **kwargs):
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
        
    params, plots_data, plots, ui = stacked_epoch_basic_setup(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, should_use_MatplotlibTimeSynchronizedWidget=should_use_MatplotlibTimeSynchronizedWidget, debug_test_max_num_slices=debug_test_max_num_slices, single_plot_fixed_height=single_plot_fixed_height, debug_print=debug_print)
    
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
    # plots.figure_id = 'stacked_epoch_slices_INSET_matplotlib'
    plots.figure_id = plots.name # copy the name as the figure_id

    ## Build Core Figure and its single axis:
    if not params.should_use_MatplotlibTimeSynchronizedWidget:
        ## Basic Matplotlib Version:
        plots.fig, plots.parent_ax = plt.subplots(num=plots.figure_id, ncols=1, nrows=1, figsize=(15,15), clear=True, sharex=False, sharey=False) # , constrained_layout=True, frameon=False
        # plots.fig, plots.parent_ax = plt.subplots(**({'num': plots.figure_id, 'ncols': 1, 'nrows': 1, 'dpi': 72, 'clear': True, 'sharex': False, 'sharey': False, 'constrained_layout': constrained_layout, 'frameon': False} | kwargs))
        ui.mw = None
    else:
        ## MatplotlibTimeSynchronizedWidget-embedded Version:
        ui.mw = MatplotlibTimeSynchronizedWidget(size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure, scrollAreaContents_MinimumHeight=params.all_plots_height, name=name, plot_function_name=plot_function_name, **kwargs) # , clear=True
        plots.fig = ui.mw.getFigure()
        plots.parent_ax = plots.fig.subplots(ncols=1, nrows=1, sharex=False, sharey=False) # , figsize=(15,15), clear=True, constrained_layout=True

    # Remove frames/spines:
    plots.parent_ax.axis('off')
    plots.fig.patch.set_visible(False)

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
        curr_ax.axis('off') # remove the box and spines

        # Appends:
        plots.axs.append(curr_ax)
    
    if params.should_use_MatplotlibTimeSynchronizedWidget:
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



@define(slots=False, eq=False)
class EpochSelectionsObject(SelectionsObject):
    """ represents a selection of epochs in an interactive paginated viewer.
    """
    epoch_times: Optional[NDArray] = field()

    def to_dataframe(self) -> pd.DataFrame:
        # to dataframe:
        dict_repr = self.to_dict()
        selection_epochs_df = pd.DataFrame(dict_repr.subset(['epoch_times', 'epoch_labels', 'flat_all_data_indicies']))
        selection_epochs_df['is_selected'] = dict_repr['is_selected'].values()
        return selection_epochs_df


    @classmethod
    def init_from_visualization_params(cls, params: VisualizationParameters, epoch_times=None):
        active_params_dict: benedict = benedict(params.to_dict())
        active_params_dict = active_params_dict.subset(['global_epoch_start_t', 'global_epoch_end_t', 'variable_name', 'active_identifying_figure_ctx', 'flat_all_data_indicies', 'epoch_labels', 'is_selected'])
        active_params_dict['is_selected'] = np.array(list(active_params_dict['is_selected'].values())) # dump the keys
        return cls(**active_params_dict, epoch_times=epoch_times)


    def update_selections_from_annotations(self, user_annotations_dict:dict, debug_print=True):
        """ 

        saved_selection_L.is_selected


        saved_selection_L = pagination_controller_L.save_selection()
        saved_selection_S = pagination_controller_S.save_selection()

        saved_selection_L.update_selections_from_annotations(user_annotations_dict=user_annotations)
        saved_selection_S.update_selections_from_annotations(user_annotations_dict=user_annotations)
                
        ## re-apply the selections:
        pagination_controller_L.restore_selections(saved_selection_L)
        pagination_controller_S.restore_selections(saved_selection_S)

        
        """
        final_figure_context = self.figure_ctx
        was_annotation_found = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_annotations_dict.items():
            an_item_diff = a_ctx.diff(final_figure_context)
            if debug_print:
                print(an_item_diff)
                print(f'\t{len(an_item_diff)}')
            if an_item_diff == {('user_annotation', 'selections')}:
                print(f'item found: {a_ctx}\nselections_array: {selections_array}')
                was_annotation_found = True
                self.is_selected = np.isin(self.flat_all_data_indicies, selections_array) # update the is_selected
                break # done looking
            
            # print(IdentifyingContext.subtract(a_ctx, final_context_L))
        if not was_annotation_found:
            print(f'WARNING: no matching context found in {len(user_annotations_dict)} annotations. `saved_selection` will be returned unaltered.')
        return self
    


## Potential Callbacks
class ClickActionCallbacks:
    """ Potential callbacks to be performed when an epoch is clicked. 
    
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import ClickActionCallbacks

    paginated_multi_decoder_decoded_epochs_window.params.on_middle_click_item_callbacks['copy_axis_image_to_clipboard_callback'] = ClickActionCallbacks.copy_axis_image_to_clipboard_callback

    """
    def copy_axis_image_to_clipboard_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ called when the user middle-clicks an epoch 
        
        captures: nothing
        """
        print(f'copy_axis_image_to_clipboard_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                # print(f'start_t: {start_t}')
                print(f'clicked widget at {clicked_ax}. Copying to clipboard...')
                self.ui.mw.copy_axis_to_clipboard(an_ax=clicked_ax)
                print(f'done.')


    def copy_to_epoch_times_to_clipboard_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ called to copy the clicked epoch's start/end time to the clipboard
        
        captures: nothing
        """
        from pyphocorehelpers.programming_helpers import copy_to_clipboard
        print(f'copy_to_epoch_times_to_clipboard_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                print(f'clicked widget at {clicked_ax}. Copying to clipboard...')
                code_string: str = f"clicked_epoch = np.array([{start_t}, {end_t}])"
                copy_to_clipboard(code_string, message_print=True)
                print(f'done.')


    def log_clicked_epoch_times_to_message_box_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ called to copy the clicked epoch's start/end time to the clipboard
        
        captures: nothing
        """
        print(f'log_clicked_epoch_times_to_message_box_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                # print(f'clicked widget at {clicked_ax}. Copying to clipboard...')
                code_string: str = f"[{start_t}, {end_t}]"
                self.ui.thin_button_bar_widget.label_message = f"<clicked> {code_string}"
                self.show_message(message=f"{code_string}", durationMs=1000)
                # print(f'done.')



class DecodedEpochSlicesPaginatedFigureController(PaginatedFigureController):
    """2023-05-09 - A stateful class containing decoded epoch posteriors.

        Builds a matplotlib Figure in a CustomMatplotlibWidget that displays paginated axes using a Paginator by creating a `PaginationControlWidget`
        Specifically uses `plot_rr_aclu`, not general
        
        Ultimately plots the epochs via calls to `plot_1D_most_likely_position_comparsions`
        
    History:
        2023-05-09 - Refactored from  `build_figure_and_control_widget_from_paginator` - Aims to refactor `plot_paginated_decoded_epoch_slices`, a series of nested functions, into a stateful class
    
    ## [X]: would have to add reuse of figure and ax to `plot_rr_aclu` as a minimum - 5 minutes
    ## [X] 2023-05-02 - would have to add the concept of the current page index, the next/previous/jump operations (that could be triggered by arrows in the GUI for example) - 30 minutes
    ## [X] 2023-05-02 - then would have to add something to hold the resultant fig, ax objects, the initial plot function, and then the plot update function. - 30 minutes
    ## [X] 2023-05-03 - sorting is currently done locally (by page) which isn't good, need to move out and make copies of all the variables.

    Usage:
    
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController
        _out_pagination_controller = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(long_results_obj.active_filter_epochs, long_results_obj.all_included_filter_epochs_decoder_result, xbin=long_results_obj.original_1D_decoder.xbin, global_pos_df=global_session.position.df, a_name='TestDecodedEpochSlicesPaginationController', max_subplots_per_page=20)
        _out_pagination_controller
    """

    @classmethod
    def init_from_decoder_data(cls, active_filter_epochs, filter_epochs_decoder_result, xbin, global_pos_df, included_epoch_indicies=None, a_name:str = 'DecodedEpochSlicesPaginationController', active_context=None, max_subplots_per_page=20, debug_print=False, params_kwargs: Optional[Dict]=None, **kwargs):
        """ new version (replacing `plot_paginated_decoded_epoch_slices`) calls `plot_decoded_epoch_slices` which produces the state variables (params, plots_data, plots, ui), a new instance of this object type is then initialized with those variables and then updated with any specific properties. """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices #, _helper_update_decoded_single_epoch_slice_plot #, _subfn_update_decoded_epoch_slices
        
        # single_plot_fixed_height=100.0, debug_test_max_num_slices=20, size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True
        if isinstance(active_filter_epochs, pd.DataFrame):
            # convert to native epoch object.
            active_filter_epochs = Epoch(epochs=deepcopy(active_filter_epochs)) # convert to native Epoch object

        if params_kwargs is None:
            params_kwargs = {} # empty dict


        ## Pop Window-level params:
        disable_toolbar = params_kwargs.pop('disable_toolbar', True)
        kwargs['disable_toolbar'] = disable_toolbar


        params, plots_data, plots, ui = plot_decoded_epoch_slices(filter_epochs=deepcopy(active_filter_epochs), filter_epochs_decoder_result=deepcopy(filter_epochs_decoder_result), global_pos_df=global_pos_df, variable_name='lin_pos', xbin=xbin, included_epoch_indicies=included_epoch_indicies,
                                                                name=a_name, debug_print=False, debug_test_max_num_slices=max_subplots_per_page, params_kwargs=params_kwargs, **kwargs)

        new_obj = cls(params, plots_data, plots, ui)
        
        new_obj.params.debug_print = debug_print
        new_obj.ui.print = builtins.print # the print function to use

        new_obj.plots_data.paginator = new_obj._subfn_helper_build_paginator(active_filter_epochs, filter_epochs_decoder_result, max_subplots_per_page, new_obj.params.debug_print)  # assign the paginator
        new_obj.params.active_identifying_figure_ctx = active_context # set context before calling `plot_paginated_decoded_epoch_slices` which will set the rest of the properties

        ## Resize the widget to meet the minimum height requirements:
        a_widget = new_obj.ui.mw # MatplotlibTimeSynchronizedWidget
        # resize to minimum height
        screen = a_widget.screen()
        screen_size = screen.size()

        target_height = new_obj.params.get('scrollAreaContents_MinimumHeight', None)
        if target_height is None:
            target_height = (new_obj.params.all_plots_height + 30) # 30 is for the scrollbar footer
        # target_height = new_obj.params.get('scrollAreaContents_MinimumHeight', None) | (new_obj.params.all_plots_height + 30)
        desired_final_height = int(min(target_height, screen_size.height())) # don't allow the height to exceed the screen height.
        # print(f'target_height: {target_height}, {  desired_final_height = }')
        # a_widget.size()
        a_widget.setMinimumHeight(desired_final_height) # the 30 is for the control bar
    
        # new_obj.params.scrollability_mode

        ## Real setup:
        new_obj.configure()
        new_obj.initialize()

        # ## 2. Update:
        # new_obj.on_paginator_control_widget_jump_to_page(page_idx=0)
        # _a_connection = new_obj.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(new_obj.on_paginator_control_widget_jump_to_page) # bind connection
        # new_obj.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection

        return new_obj
    
    
    @property
    def selected_epoch_times(self):
        """The Determine the Epochs that have actually been selected so they can be saved/stored somehow.
        Returns: S x 2 array of epoch start/end times
        """
        assert np.shape(self.plots_data.epoch_slices)[0] == len(self.is_selected), f"Selection length must be the same as the number of epoch_slices, otherwise we do not know what we are selecting! np.shape(self.plots_data.epoch_slices): {np.shape(self.plots_data.epoch_slices)}, len(self.params.is_selected): {len(self.params.is_selected)}"
        return self.plots_data.epoch_slices[self.is_selected] # returns an S x 2 array of epoch start/end times that are currently selected.


    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            self.plots_data.epoch_slices
        """
        from neuropy.core.epoch import find_data_indicies_from_epoch_times
        epoch_slices_df = pd.DataFrame(self.plots_data.epoch_slices, columns=['start', 'stop'])
        return find_data_indicies_from_epoch_times(epoch_slices_df, epoch_times=epoch_times, atol=0.01, not_found_action='skip_index', debug_print=self.params.get('debug_print', False))
        # return np.nonzero(np.isclose(self.plots_data.epoch_slices, epoch_times[:, None], atol=1e-3, rtol=1e-3).all(axis=2).any(axis=0))[0]

    def save_selection(self) -> EpochSelectionsObject:
        active_selections_object = EpochSelectionsObject.init_from_visualization_params(self.params, epoch_times=deepcopy(self.selected_epoch_times))
        return active_selections_object


    def restore_selections(self, selections: EpochSelectionsObject, is_additive_update:bool=False, defer_render=False):
        """ sets the current selections from the provided selections object

        additive_update: if True, provided epoch_times are appended to the current selection, otherwise the current selection is replaced by the epoch_times

        """
        # Validate the restore by making sure that we're restoring onto the valid objects
        assert self.params.active_identifying_figure_ctx == selections.figure_ctx
        assert self.params.variable_name == selections.variable_name
        if not is_additive_update:
            ## sets all except these to False
            self._perform_clear_all_selections()
        if selections.epoch_times is not None:
            # Always use the epoch_times by default. This avoids any indexing issues
            matching_indices = self.find_data_indicies_from_epoch_times(selections.epoch_times)
            for a_selected_index in matching_indices:
                assert a_selected_index in self.params.flat_all_data_indicies, f"a_selected_index: {a_selected_index} is not in flat_all_data_indicies: {self.params.flat_all_data_indicies}"
                self.params.is_selected[a_selected_index] = True
        else:
            # Revert to the old way using indicies to handle legacy saved annotations
            self.ui.print(f'falling back to the old index-only method of setting selected epochs.')
            for a_selected_index in selections.selected_indicies:
                assert a_selected_index in self.params.flat_all_data_indicies, f"a_selected_index: {a_selected_index} is not in flat_all_data_indicies: {self.params.flat_all_data_indicies}"
                self.params.is_selected[a_selected_index] = True

        # Post:
        self.perform_update_selections(defer_render=defer_render)
        return self.params.is_selected
    

    def restore_selections_from_epoch_times(self, epoch_times: NDArray, is_additive_update:bool=False, defer_render=False):
        """ Unsafe: sets the selects from the provided epoch_times
        additive_update: if True, provided epoch_times are appended to the current selection, otherwise the current selection is replaced by the epoch_times

        """
        assert epoch_times is not None
        # Always use the epoch_times by default. This avoids any indexing issues
        matching_indices = self.find_data_indicies_from_epoch_times(epoch_times)
        
        if not is_additive_update:
            ## sets all except these to False
            self._perform_clear_all_selections()
        for a_selected_index in matching_indices:
            assert (a_selected_index in self.params.flat_all_data_indicies), f"a_selected_index: {a_selected_index} is not in flat_all_data_indicies: {self.params.flat_all_data_indicies}"
            self.params.is_selected[a_selected_index] = True

        # Post:
        self.perform_update_selections(defer_render=defer_render)
        return self.params.is_selected
    

    ## Supposed to be the right callback:
    def on_selected_epochs_changed(self, clicked_ax, clicked_data_index: int) -> bool:
        """ 

        Called by `_out_pagination_controller.on_click` to try to toggle the clicked_data_index sis_selected value, and performs the graphical updates if anything changed.

        Updates:
            `_out_pagination_controller.params.is_selected`

        History:
            Used to be the opposite direction. This function was bound to the matplotlib axis click callback, and it forwarded the click event to the `_out_pagination_controller.on_click` callback. 

        """
        self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.on_selected_epochs_changed(...)')
        pre_selected_times = deepcopy(self.selected_epoch_times)

        # Actually perform the changes here: _________________________________________________________________________________ #
        # Toggle the selection status of the clicked Axes
        self.params.is_selected[clicked_data_index] = not self.params.is_selected.get(clicked_data_index, False) # if never set before, assume that it's not selected
        
        ## Determine the Epochs that have actually been selected so they can be saved/stored somehow:
        # selected_epoch_times = self.selected_epoch_times # returns an S x 2 array of epoch start/end times that are currently selected.
        if self.params.debug_print:
            self.ui.print(f'\tselection_indicies: {self.selected_indicies}')

        if self.params.debug_print:
            self.ui.print(f'\tselected_epoch_times: {self.selected_epoch_times}')
        
        post_selected_times = deepcopy(self.selected_epoch_times)
        # did_selection_change: bool = (pre_selected_times != post_selected_times)
        # did_selection_change: bool = np.logical_not(np.all(pre_selected_times == post_selected_times))
        did_selection_change: bool = not np.array_equal(pre_selected_times, post_selected_times)
        if did_selection_change:
            self.ui.print(f'\tDecodedEpochSlicesPaginatedFigureController.on_selected_epochs_changed: selection changed!')
            ## Update visual apperance of axis:
            self.perform_update_ax_selected_state(ax=clicked_ax, is_selected=self.params.is_selected[clicked_data_index])
            self.draw() # redraws

        return did_selection_change


    # Lifecycle Methods __________________________________________________________________________________________________ #
    def configure(self, **kwargs):
        """ assigns and computes needed variables for rendering. """
        self.ui.print = self.private_print # builtins.print # the print function to use
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.configure(**kwargs: {kwargs})')
        
        ## Add the PaginationControlWidget or setup the PaginationControlWidgetState depending on isPaginatorControlWidgetBackedMode
        self._helper_setup_pagination_control()
        ## Setup Selectability
        self._subfn_helper_setup_selectability()
        
        
    def initialize(self, **kwargs):
        """ sets up Figures """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.initialize(**kwargs: {kwargs})')
        
        ## Setup on_click callback:
        self.connect_on_click_callback() # connect the page
        # if not self.params.has_attr('callback_id') or self.params.get('callback_id', None) is None:
        #     self.params.callback_id = self.plots.fig.canvas.mpl_connect('button_press_event', self.on_click) ## TypeError: unhashable type: 'DecodedEpochSlicesPaginatedFigureController'

        if self.params.get('build_internal_callbacks', True):
            self.build_internal_callbacks()

        ## 2. Update:
        self.on_jump_to_page(page_idx=0)
        if self.params.setdefault('isPaginatorControlWidgetBackedMode', True):
            _a_connection = self.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(self.on_paginator_control_widget_jump_to_page) # bind connection
            self.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection

        # self._subfn_build_selectibility_rects_if_needed(axs=)
        self.perform_update_selections(defer_render=False)
        

    def update(self, **kwargs):
        """ called to specifically render data on the figure. """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.update(**kwargs: {kwargs})')


    def on_close(self):
        """ called when the figure is closed. """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.on_close()')


    def private_print(self, *args, **kwargs):
        """ wraps the default print function or whatever needs to be called. """
        print(*args, **kwargs)




    # Other Methods ______________________________________________________________________________________________________ #

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


    def connect_on_click_callback(self):
        """ connects the button_press_event callback to self.on_click """
        if not self.params.has_attr('callback_id') or self.params.get('callback_id', None) is None:
            # _out_pagination_controller.params.callback_id = _out_pagination_controller.plots.fig.canvas.mpl_connect('button_press_event', _out_pagination_controller.on_click) ## TypeError: unhashable type: 'DecodedEpochSlicesPaginatedFigureController'
            self.params.callback_id = self.plots.fig.canvas.mpl_connect('button_press_event', self.on_click) ## TypeError: unhashable type: 'DecodedEpochSlicesPaginatedFigureController'

    def disconnect_on_click_callback(self):
        """ disconnects the button_press_event callback for the figure."""
        self.plots.fig.canvas.mpl_disconnect(self.params.callback_id)
        self.params.callback_id = None


    def build_internal_callbacks(self):
        """ """
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        if not self.params.has_attr('on_middle_click_item_callbacks'):
            self.params['on_middle_click_item_callbacks'] = {}

        self.params.on_middle_click_item_callbacks['copy_axis_image_to_clipboard_callback'] = ClickActionCallbacks.copy_axis_image_to_clipboard_callback

        if not self.params.has_attr('on_secondary_click_item_callbacks'):
            self.params['on_secondary_click_item_callbacks'] = {}

        self.params.on_secondary_click_item_callbacks['copy_to_epoch_times_to_clipboard_callback'] = ClickActionCallbacks.copy_to_epoch_times_to_clipboard_callback
        self.params.on_secondary_click_item_callbacks['log_clicked_epoch_times_to_message_box_callback'] = ClickActionCallbacks.log_clicked_epoch_times_to_message_box_callback
        


    def on_jump_to_page(self, page_idx: int):
        """ Called when the page index is changed to update the figure
        
        Performs `self.params.on_render_page_callbacks`

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions # used in `plot_decoded_epoch_slices`
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.on_jump_to_page(page_idx: {page_idx}): {self.params.name}') # for page_idx == max_index this is called but doesn't continue
        included_page_data_indicies, (curr_page_active_filter_epochs, curr_page_epoch_labels, curr_page_time_bin_containers, curr_page_posterior_containers) = self.plots_data.paginator.get_page_data(page_idx=page_idx)
        if self.params.debug_print:
            self.ui.print(f'\tincluded_page_data_indicies: {included_page_data_indicies}')
    
        for i, curr_ax in enumerate(self.plots.axs):
            try:
                curr_slice_idx: int = included_page_data_indicies[i]
                curr_epoch_slice = curr_page_active_filter_epochs[i]
                curr_time_bin_container = curr_page_time_bin_containers[i]
                curr_posterior_container = curr_page_posterior_containers[i]
                curr_time_bins = curr_time_bin_container.centers
                curr_posterior = curr_posterior_container.p_x_given_n
                curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D
                
                if self.params.debug_print:
                    self.ui.print(f'i : {i}, curr_posterior.shape: {curr_posterior.shape}')

                ## Clear the axis here:
                curr_ax.clear()

                # Update the axes appropriately:
                _pagination_helper_plot_single_epoch_slice(curr_ax, self.params, self.plots_data, self.plots, self.ui, a_slice_idx=curr_slice_idx, is_first_setup=False, debug_print=self.params.debug_print) # calling with is_first_setup=False doesn't set the right-hand Epoch end time labels right

                skip_plotting_measured_positions: bool = self.params.get('skip_plotting_measured_positions', False)
                skip_plotting_most_likely_positions: bool = self.params.get('skip_plotting_most_likely_positions', False)
                
                ## NOTE: the actual heatmp is plotted using: 
                _temp_fig, curr_ax = plot_1D_most_likely_position_comparsions(self.plots_data.global_pos_df, ax=curr_ax, time_window_centers=curr_time_bins, variable_name=self.params.variable_name, xbin=self.params.xbin,
                                                                posterior=curr_posterior,
                                                                active_most_likely_positions_1D=curr_most_likely_positions,
                                                                enable_flat_line_drawing=self.params.enable_flat_line_drawing, debug_print=self.params.debug_print,
                                                                skip_plotting_measured_positions=skip_plotting_measured_positions, skip_plotting_most_likely_positions=skip_plotting_most_likely_positions,
                                                                posterior_heatmap_imshow_kwargs=self.params.get('posterior_heatmap_imshow_kwargs', None),
                                                                )
                
                
                if _temp_fig is not None:
                    self.plots.fig = _temp_fig

                ## Perform callback here:
                on_render_page_callbacks = self.params.get('on_render_page_callbacks', {})
                if self.params.debug_print:
                    self.ui.print(f'{self.params.name}: performing on_render_page_callbacks ({len(on_render_page_callbacks)} callbacks):')
                for a_callback_name, a_callback in on_render_page_callbacks.items():
                    if self.params.debug_print:
                        self.ui.print(f'\tperforming callback with name: "{a_callback_name}" for page_idx: {page_idx}, i: {i}, data_idx: {curr_slice_idx}, curr_ax: {curr_ax}')
                    with ExceptionPrintingContext(suppress=self.params.get("should_suppress_callback_exceptions", True), exception_print_fn=(lambda formatted_exception_str: self.ui.print(f'\t\t WARNING: encountered exception in callback with name "{a_callback_name}" for page_idx: {page_idx}, i: {i}, data_idx: {curr_slice_idx}, curr_ax: {curr_ax}: exception: {formatted_exception_str}'))):
                        self.params, self.plots_data, self.plots, self.ui = a_callback(curr_ax, self.params, self.plots_data, self.plots, self.ui, curr_slice_idx, curr_time_bins, curr_posterior, curr_most_likely_positions, debug_print=self.params.debug_print, epoch_slice=curr_epoch_slice, curr_time_bin_container=curr_time_bin_container)
                        if self.params.debug_print:
                            self.ui.print(f'\t\tcallback with name: "{a_callback_name}" complete.')
                    
                    
                curr_ax.set_xlim(*curr_epoch_slice)
                curr_ax.set_title(f'') # needs to be set to empty string '' because this is the title that appears above each subplot/slice
                # Update selections:
                assert curr_slice_idx in self.params.is_selected, f"curr_slice_idx: {curr_slice_idx} is not in self.params.is_selected!"
                curr_is_selected: bool = self.params.is_selected[curr_slice_idx] # need to get whether the data indicies (curr_slice_idxs) are selected
                if self.params.debug_print:
                    self.ui.print(f'updating selection for ax[i={i}]\n\tcurr_slice_idxs: {curr_slice_idx}')
                    self.ui.print(f'\tcurr_is_selected: {curr_is_selected}')
                    self.ui.print(f'\tself.current_page_idx: {self.current_page_idx}')

                self.perform_update_ax_selected_state(ax=curr_ax, is_selected=curr_is_selected)
                curr_ax.set_visible(True)
                
            except IndexError as e:
                # Occurs when there are more plots on the page than there are data to plot for that page (happens on the last page)
                if self.params.debug_print:
                    self.ui.print(f'WARNING: exceeded data indicies (probably on last page). (for page_idx: {page_idx}, i: {i}, curr_ax: {curr_ax})')
                curr_ax.set_visible(False)

            except Exception as e:
                raise e
            

        # # Update selection (could also do just in above loop):
        # self.perform_update_selections()

        if self.params.get('enable_update_window_title_on_page_change', True):
            # this call was found to be slow:
            self.perform_update_titles_from_context(page_idx=page_idx, included_page_data_indicies=included_page_data_indicies) # , collision_prefix='_DecodedEpochSlices_plot_test_', display_fn_name='plot_single_epoch_slice', plot_result_set='shared'

        
        self.ui.mw.draw()



    def on_paginator_control_widget_jump_to_page(self, page_idx: int):
        """ Called when the pagination control widget jumps to a particular page """
        return self.on_jump_to_page(page_idx=page_idx)



    def refresh_current_page(self):
        """ called to refresh the currently selected page to redraw the data widgets or axes.
        
        """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.refresh_current_page():') # for page_idx == max_index this is called but doesn't continue
        self.on_jump_to_page(page_idx=self.current_page_idx)


    # ==================================================================================================================== #
    # Interactive Selection Overrides                                                                                      #
    # ==================================================================================================================== #
    def on_click(self, event):
        """ called when an axis is clicked via any mouse button. """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.on_click(...) OVERRIDE:')

        # key = event.key # The key(s) pressed. Could be None, a single case sensitive Unicode character ("g", "G", "#", etc.), a special key ("control", "shift", "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g", "ctrl+alt+G").
        # modifiers = event.modifiers
        # is_dblclick = event.dblclick

        # Get the clicked Axes object
        ax = event.inaxes
        # Find the axes
        found_index = safe_find_index_in_list(self.plots.axs, ax) # find the index on the page of the ax that was clicked
        if found_index is not None:
            # print(f'{found_index = }')
            current_page_idx = self.current_page_idx
            curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
            found_data_index = curr_page_data_indicies[found_index]
            self.ui.print(f'{current_page_idx = }, {found_data_index =}') # array([[0, 1, 2, 3, 4, 5, 6, 7]])
        else:
            self.ui.print(f'\tWARNING: could not find the clicked ax: {ax} in the list of axes: {self.plots.axs}')
            found_data_index = None

        # Get the event extra info
        mouse_button = event.button # None or MouseButton or {'up', 'down'}
        if mouse_button is not None:
            if self.params.debug_print:
                self.ui.print(f'mouse_button: {mouse_button}')
            if (mouse_button == MouseButton.LEFT):
                if self.params.debug_print:
                    self.ui.print(f'\tPrimary click recognized.')
                self.on_primary_click(event, clicked_ax=ax, clicked_data_index=found_data_index)
            elif (mouse_button == MouseButton.MIDDLE):
                if self.params.debug_print:
                    self.ui.print(f'\tMiddle click recognized.')
                self.on_middle_click(event, clicked_ax=ax, clicked_data_index=found_data_index)
            elif (mouse_button == MouseButton.RIGHT):
                if self.params.debug_print:
                    self.ui.print(f'\tSecondary click recognized.')
                self.on_secondary_click(event, clicked_ax=ax, clicked_data_index=found_data_index)
            else:
                self.ui.print(f'\tWARNING:unhandled mouse button: {mouse_button}')

        # Redraw the figure to show the updated selection
        # event.canvas.draw()
        # event.canvas.draw_idle()

    def try_get_clicked_epoch(self, clicked_data_index):
        """ a helper function to try to find the epoch info corresponding to the clicked_data_index. Used by on_middle_click and on_secondary_click.
        """
        if clicked_data_index is not None:
            clicked_epoch_is_selected = self.params.is_selected[clicked_data_index]
            clicked_epoch_start_stop_time = self.plots_data.epoch_slices[clicked_data_index]
        else:
            clicked_epoch_is_selected = None
            clicked_epoch_start_stop_time = None
        return clicked_epoch_start_stop_time, clicked_epoch_is_selected
            

    def on_primary_click(self, event, clicked_ax=None, clicked_data_index=None):
        """ a primary (Usually left)-click event. Called manually from self.on_click(...) for appropriate mouse button events.
        """
        if self.params.debug_print:
            self.ui.print(f'on_primary_click(event, clicked_ax={clicked_ax}, clicked_data_index={clicked_data_index})')
        if ((clicked_data_index is not None) and (clicked_ax is not None)):
            # Toggle the selection status of the clicked Axes
            self.on_selected_epochs_changed(clicked_ax=clicked_ax, clicked_data_index=clicked_data_index)

    

    def on_middle_click(self, event, clicked_ax=None, clicked_data_index=None):
        """ a middle-click event. Called manually from self.on_click(...) for appropriate mouse button events.
        """
        self.ui.print(f'on_middle_click(event, clicked_ax={clicked_ax}, clicked_data_index={clicked_data_index})')
        clicked_epoch_start_stop_time, clicked_epoch_is_selected = self.try_get_clicked_epoch(clicked_data_index=clicked_data_index)
        
        # Middle Click Callbacks _____________________________________________________________________________________________ #
        on_middle_click_item_callbacks = self.params.get('on_middle_click_item_callbacks', {})
        for a_callback_name, a_callback in on_middle_click_item_callbacks.items():
            if self.params.debug_print:
                self.ui.print(f'\tperforming callback with name: "{a_callback_name}" for clicked_data_index: {clicked_data_index}, clicked_ax: {clicked_ax}')

            with ExceptionPrintingContext(suppress=self.params.get("should_suppress_callback_exceptions", True), exception_print_fn=(lambda formatted_exception_str: self.ui.print(f'\t\t WARNING: encountered exception in callback with name "{a_callback_name}" for clicked_data_index: {clicked_data_index}, clicked_ax: {clicked_ax}: exception: {formatted_exception_str}'))):        
                a_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time)



    def on_secondary_click(self, event, clicked_ax=None, clicked_data_index=None):
        """ a secondary (usually right)-click event. Called manually from self.on_click(...) for appropriate mouse button events.
        """
        self.ui.print(f'on_secondary_click(event, clicked_ax={clicked_ax}, clicked_data_index={clicked_data_index})')
        clicked_epoch_start_stop_time, clicked_epoch_is_selected = self.try_get_clicked_epoch(clicked_data_index=clicked_data_index)
        
        # Secondary-Click Callbacks _____________________________________________________________________________________________ #
        on_secondary_click_item_callbacks = self.params.get('on_secondary_click_item_callbacks', {})
        for a_callback_name, a_callback in on_secondary_click_item_callbacks.items():
            if self.params.debug_print:
                self.ui.print(f'\tperforming callback with name: "{a_callback_name}" for clicked_data_index: {clicked_data_index}, clicked_ax: {clicked_ax}')

            with ExceptionPrintingContext(suppress=self.params.get("should_suppress_callback_exceptions", True), exception_print_fn=(lambda formatted_exception_str: self.ui.print(f'\t\t WARNING: encountered exception in callback with name "{a_callback_name}" for clicked_data_index: {clicked_data_index}, clicked_ax: {clicked_ax}: exception: {formatted_exception_str}'))):        
                a_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time)




    def perform_update_ax_selected_state(self, ax, is_selected: bool):
        """ simply updates the visual appearance of the provided ax to indicate whether it's selected. """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.perform_update_ax_selected_state(ax: {ax}, is_selected: {is_selected}) OVERRIDE:')
        # Set the face color of the clicked Axes based on its selection status
        if is_selected:
            ax.patch.set_facecolor('seagreen')
            set_ax_emphasis_color(ax, emphasis_color='darkgreen', defer_draw=True)
        else:
            ax.patch.set_facecolor('white')
            set_ax_emphasis_color(ax, emphasis_color='dimgrey', defer_draw=True)

        # Update the selection rectangles for this ax if we have them:
        selection_artists_dict = self.plots.get('selection_artists_dict', {})
        a_selection_artists = selection_artists_dict.get(ax, {})
        


        # Update the selection rectangles for this ax if we have them:
        a_selection_rect = a_selection_artists.get('rectangles', None)        
        if a_selection_rect is not None:
            if self.params.debug_print:
                self.ui.print(f'\ta_selection_rect.set_visible({is_selected})')
            a_selection_rect.set_visible(is_selected)
        else:
            if self.params.debug_print:
                self.ui.print(f'\ta_selection_rect is None!')

        an_action_buttons_list = a_selection_artists.get('action_buttons', None)
        if an_action_buttons_list is not None:
            ## TODO: do something here?
            if self.params.debug_print:
                self.ui.print(f'\tan_action_buttons_list({is_selected})')
        if self.params.debug_print:
            self.ui.print(f'\tdone.')


    def perform_update_selections(self, defer_render:bool=True):
        """ called to update the selection when the page is changed or something else happens. """
        if self.params.debug_print:
            self.ui.print(f'DecodedEpochSlicesPaginatedFigureController.perform_update_selections(...) OVERRIDE:')
        current_page_idx = self.current_page_idx
        curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
        
        ## add Nones to end of curr_page_data_indicies list:
        num_unused_axes: int = len(self.plots.axs) - len(curr_page_data_indicies)
        # curr_page_data_indicies: np.array([120])
        curr_page_data_indicies = np.array(list(curr_page_data_indicies) + (num_unused_axes * [None]))
        # curr_page_data_indicies: [120, None, None, None, None, None, None, None]
        assert len(self.plots.axs) == len(curr_page_data_indicies), f"len(plots.axs): {len(self.plots.axs)}, len(curr_page_data_indicies): {len(curr_page_data_indicies)}" ## fails on last page with AssertionError: len(plots.axs): 8, len(curr_page_data_indicies): 1

        ## This seems uneeeded, but we'll see:
        self._subfn_build_selectibility_rects_if_needed(self.plots.axs, list(curr_page_data_indicies))

        for ax, found_data_idx in zip(self.plots.axs, list(curr_page_data_indicies)): # TODO: might fail for the last page?
            is_selected = self.params.is_selected.get(found_data_idx, False)
            self.perform_update_ax_selected_state(ax=ax, is_selected=is_selected)
                
        # Redraw the figure to show the updated selection
        # assert defer_render
        if (not defer_render):
            self.draw()



    def draw(self):
        """ Calls .draw() on child MatplotlibTimeSynchronizedWidget.
        """
        a_widget = self.ui.mw # MatplotlibTimeSynchronizedWidget
        assert a_widget is not None
        a_widget.draw() # Call Draw on the contained widget
        # If that doesn't work, used to do:
        # self.plots.fig.canvas.draw_idle()
        # or with ax being the clicked axis:
        # ax.get_figure().canvas.draw()


    # ==================================================================================================================== #
    # Single pagination_controller versions of PhoPaginatedMultiDecoderDecodedEpochsWindow's methods                       #
    # ==================================================================================================================== #

    def add_data_overlays(self, decoder_decoded_epochs_result, included_columns=None, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data for this decoder and adds them to the plot.
        
        I think decoder_decoded_epochs_result:DecodedFilterEpochsResult
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import WeightedCorrelationPaginatedPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DecodedPositionsPlotDataProvider

        # self: PaginatedFigureController

        # a_pagination_controller = self

        ## Choose which columns from the filter_epochs dataframe to include on the plot.
        if included_columns is None:
            included_columns = []

        # Build Radon Transforms and add them:
        radon_transform_epochs_data = RadonTransformPlotDataProvider.decoder_build_single_radon_transform_data(deepcopy(decoder_decoded_epochs_result))
        if radon_transform_epochs_data is not None:
            RadonTransformPlotDataProvider.add_data_to_pagination_controller(self, radon_transform_epochs_data, update_controller_on_apply=False)
    
        # Build Weighted Correlation Data Info and add them:    
        wcorr_epochs_data = WeightedCorrelationPaginatedPlotDataProvider.decoder_build_single_weighted_correlation_data(deepcopy(decoder_decoded_epochs_result))
        if wcorr_epochs_data is not None:
            WeightedCorrelationPaginatedPlotDataProvider.add_data_to_pagination_controller(self, wcorr_epochs_data, update_controller_on_apply=False)


        # Build Decoded Positions Data and add them:    
        decoded_position_curves_epochs_data = DecodedPositionsPlotDataProvider.decoder_build_single_decoded_position_curves_data(deepcopy(decoder_decoded_epochs_result))
        if decoded_position_curves_epochs_data is not None:
            DecodedPositionsPlotDataProvider.add_data_to_pagination_controller(self, decoded_position_curves_epochs_data, update_controller_on_apply=False)

        if not defer_refresh:
            self.refresh_current_page()


    def remove_data_overlays(self, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data for this decoder and adds them to the plot.
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import WeightedCorrelationPaginatedPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DecodedPositionsPlotDataProvider

        RadonTransformPlotDataProvider.remove_data_from_pagination_controller(self, update_controller_on_apply=False)
        WeightedCorrelationPaginatedPlotDataProvider.remove_data_from_pagination_controller(self, update_controller_on_apply=False)
        DecodedPositionsPlotDataProvider.remove_data_from_pagination_controller(self, update_controller_on_apply=False)

        if not defer_refresh:
            self.refresh_current_page()




    def restore_selections_from_user_annotations(self, user_annotations: Optional[Dict]=None, defer_render:bool=False, **additional_selections_context):
        """ Restores the user's selections to this pagination controller

        Uses: self.params.active_identifying_figure_ctx

        from neuropy.core.user_annotations import UserAnnotationsManager
        
        user_annotations = UserAnnotationsManager().get_user_annotations()
        new_selections = a_ctrl.restore_selections_from_user_annotations(user_annotations)

        
        """
        if user_annotations is None:
            from neuropy.core.user_annotations import UserAnnotationsManager
            annotations_man = UserAnnotationsManager()
            user_annotations = annotations_man.get_user_annotations()
        
        a_figure_ctx = self.params.active_identifying_figure_ctx
        a_selections_ctx = a_figure_ctx.adding_context_if_missing(user_annotation='selections', **additional_selections_context)
        loaded_selections = user_annotations.get(a_selections_ctx, None)
        new_selections = loaded_selections
        if loaded_selections is not None:
            # a_start_stop_arr = self.selected_epoch_times # NOPE, these are the current selections
            a_start_stop_arr = deepcopy(new_selections) # 
            if (a_start_stop_arr is not None) and (len(a_start_stop_arr) > 0):
                assert np.shape(a_start_stop_arr)[1] == 2, f"input should be start, stop times as a numpy array"
                new_selections = self.restore_selections_from_epoch_times(a_start_stop_arr, defer_render=defer_render) # TODO: only accepts epoch_times specifications
        else:
            print(f'no found selections.')
            
        return new_selections
    

    def show_message(self, message: str, durationMs:int=4000):
        """ show a toast message """
        self.ui.mw.toast.should_position_to_parent_window = False
        self.ui.mw.toast.show_message(message=message, durationMs=durationMs)


    
@metadata_attributes(short_name=None, tags=['paginated', 'multi-decoder', 'epochs', 'widget', 'window', 'ui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-23 13:54', related_items=[])
class PhoPaginatedMultiDecoderDecodedEpochsWindow(PhoDockAreaContainingWindow):
    """ a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
        pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper.PhoDockAreaContainingWindow
        
        Inherited Properties: .area

    Usage:
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import PhoPaginatedMultiDecoderDecodedEpochsWindow

    ## Ripples:
    pagination_controller_dict =  PhoPaginatedMultiDecoderDecodedEpochsWindow._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epochs_name='ripple', included_epoch_indicies=None, defer_render=False, save_figure=False)
    app, root_dockAreaWindow = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_pagination_controller_dict(pagination_controller_dict) # Combine to a single figure
    root_dockAreaWindow.add_data_overlays(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
            
    ## Laps:
    laps_pagination_controller_dict =  PhoPaginatedMultiDecoderDecodedEpochsWindow._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epochs_name='laps', included_epoch_indicies=None, defer_render=False, save_figure=False)
    laps_app, laps_root_dockAreaWindow = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_pagination_controller_dict(laps_pagination_controller_dict) # Combine to a single figure
    laps_root_dockAreaWindow.add_data_overlays(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)

    """
    @property
    def contents(self):
        return self.ui._contents
    
    @property
    def pagination_controllers(self) -> Dict[str, DecodedEpochSlicesPaginatedFigureController]:
        return self.contents.pagination_controllers

    @property
    def paginator_controller_widget(self) -> PaginationControlWidget:
        """ the widget that goes left and right by pages in the bottom of the left plot. """
        assert self.isPaginatorControlWidgetBackedMode
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        paginator_controller_widget = a_controlling_pagination_controller.ui.mw.ui.paginator_controller_widget
        return paginator_controller_widget
    
    @property
    def isPaginatorControlWidgetBackedMode(self) -> bool:
        """ whether it's isPaginatorControlWidgetBackedMode """
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        return a_controlling_pagination_controller.params.isPaginatorControlWidgetBackedMode
        

    @property
    def paginated_widgets(self) -> Dict[str, MatplotlibTimeSynchronizedWidget]:
        """ the list of plotting child widgets. """
        return {a_decoder_name:a_pagination_controller.ui.mw for a_decoder_name, a_pagination_controller in self.contents.pagination_controllers.items()}


    @property
    def debug_print(self):
        """The debug_print property."""
        return np.all([v.params.debug_print for a_name, v in self.pagination_controllers.items()])
    @debug_print.setter
    def debug_print(self, value):
        for a_name, v in self.pagination_controllers.items():
            v.params.debug_print = value

    @property
    def figure_ctx_dict(self) -> Dict[str, IdentifyingContext]:
        """ the list of plotting child widgets. """
        return {a_name:v.params.active_identifying_figure_ctx for a_name, v in self.pagination_controllers.items()} 


    # ==================================================================================================================== #
    # Initializers                                                                                                         #
    # ==================================================================================================================== #

    def __init__(self, title='PhoPaginatedMultiDecoderDecodedEpochsWindow', *args, **kwargs):
        super(PhoPaginatedMultiDecoderDecodedEpochsWindow, self).__init__(*args, **kwargs)
        self.ui._contents = None
        # self.setup()
        # self.buildUI()
        
    @classmethod
    def init_from_pagination_controller_dict(cls, pagination_controller_dict, name = 'CombinedDirectionalDecoderDecodedEpochsWindow', title='Pho Combined Directional Decoder Decoded Epochs', defer_show=False):
        """ 2024-02-14 - Copied from `RankOrderRastersDebugger`'s approach. Merges the four separate decoded epoch windows into single figure with a separate dock for each decoder.
        [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261)

        Usage:
            app, root_dockAreaWindow, _out_dock_widgets, dock_configs = merge_single_window(pagination_controller_dict)

        """
        from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        
        from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon

        ## Convert to controlled first
        new_connections_dict = cls.convert_decoder_pagination_controller_dict_to_controlled(pagination_controller_dict)

        # pagination_controller_dict = _obj.plots.rasters_display_outputs
        all_widgets = {a_decoder_name:a_pagination_controller.ui.mw for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_windows = {a_decoder_name:a_pagination_controller.ui.mw.window() for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots = {a_decoder_name:a_pagination_controller.plots for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots_data = {a_decoder_name:a_pagination_controller.plots_data for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_params = {a_decoder_name:a_pagination_controller.params for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}

        main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        
        # all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # # Extract the plot/renderable items
        # all_separate_root_plots = {a_decoder_name:a_pagination_controller.plots.root_plot for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        # all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # Embedding in docks:
        
        # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Combined Directional Decoder Decoded Epochs')
        
        # Instantiate the class ______________________________________________________________________________________________ #
        # root_dockAreaWindow = PhoDockAreaContainingWindow(title=title)
        root_dockAreaWindow = cls(title=title)
        root_dockAreaWindow.setWindowTitle(f'{title}: dockAreaWindow')
        app = root_dockAreaWindow.app
        
        icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/paginated_multi_decoder_decoded_epochs.ico")
        if icon is not None:
            root_dockAreaWindow.setWindowIcon(icon)

        ## Build Dock Widgets:
        _out_dock_widgets = {}
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))

        for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(430,780), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name], autoOrientation=False)

        # #TODO 2024-02-14 18:44: - [ ] Comgbine the separate items into one of the single `DecodedEpochSlicesPaginatedFigureController` objects (or a new one)?
        # root_dockAreaWindow.resize(600, 900)

        # ## Build final .plots and .plots_data:
        # PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)

        # root_dockAreaWindow.ui.connections.controlled_pagination_controllers_from_leftmost = new_connections_dict #
        root_dockAreaWindow.ui._contents = PhoUIContainer(name=name, names=main_plot_identifiers_list, pagination_controllers=pagination_controller_dict, 
                                                    dock_widgets=_out_dock_widgets, dock_configs=dock_configs,
                                                    widgets=all_widgets, windows=all_windows, plots=all_separate_plots, plots_data=all_separate_plots_data, params=all_separate_params) # do I need this extracted data or is it redundant?
        
        # _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
        #                             root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
        #                             dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        # _obj.plots_data = RenderPlotsData(name=name, main_plot_identifiers_list=main_plot_identifiers_list,
        #                                     seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs, seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
        #                                     on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs, **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']})
        # _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)
        # _obj.params = VisualizationParameters(name=name, is_laps=False, enable_show_spearman=True, enable_show_pearson=False, enable_show_Z_values=True, use_plaintext_title=False, **param_kwargs)


        # Add functions ______________________________________________________________________________________________________ #

        root_dockAreaWindow.ui.print = print

        # ## Cleanup when done:
        # 'disable_y_label': True
        
        # for a_decoder_name, a_root_plot in _obj.plots.root_plots.items():
        #     a_root_plot.setTitle(title=a_decoder_name)
        #     # a_root_plot.setTitle(title="")
        #     a_left_axis = a_root_plot.getAxis('left')# axisItem
        #     a_left_axis.setLabel(a_decoder_name)
        #     a_left_axis.setStyle(showValues=False)
        #     a_left_axis.setTicks([])
        #     # a_root_plot.hideAxis('bottom')
        #     # a_root_plot.hideAxis('bottom')
        #     a_root_plot.hideAxis('left')
        #     a_root_plot.setYRange(-0.5, float(_obj.max_n_neurons))

        # app, root_dockAreaWindow, _out_dock_widgets, dock_configs

        if not defer_show:
            root_dockAreaWindow.show()
            
        return app, root_dockAreaWindow

    @classmethod
    def init_from_track_templates(cls, curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict, epochs_name:str ='laps', included_epoch_indicies=None,
                                   name='CombinedDirectionalDecoderDecodedEpochsWindow', title='Pho Combined Directional Decoder Decoded Epochs', defer_show=False, **kwargs):
        """ 2024-02-28 - Combines the previously separate ._subfn_prepare_plot_multi_decoders_stacked_epoch_slices +  .init_from_pagination_controller_dict approaches. 
        Usage:
            ## Example 1 Ripples:
            app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epochs_name='ripple', included_epoch_indicies=None)

            ## Example 2 Laps:
            laps_app, laps_paginated_multi_decoder_decoded_epochs_window, laps_pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epochs_name='laps', included_epoch_indicies=None)

        """
        # 'enable_update_window_title_on_page_change'
        pagination_controller_dict =  cls._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_decoded_epochs_result_dict, epochs_name=epochs_name, included_epoch_indicies=included_epoch_indicies, defer_render=True, save_figure=False, **kwargs)
        app, paginated_multi_decoder_decoded_epochs_window = cls.init_from_pagination_controller_dict(pagination_controller_dict, name=name, title=title, defer_show=defer_show) # Combine to a single figure
        return app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict
    

    ## Add a jump to page function
    def jump_to_page(self, page_idx: int):
        if self.isPaginatorControlWidgetBackedMode:
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            # updates the embedded pagination widget
            # self.paginator_controller_widget.programmatically_update_page_idx(page_idx, block_signals=False) # don't block signals and then we don't have to call updates.
            self.paginator_controller_widget.programmatically_update_page_idx(page_idx, block_signals=True) # don't block signals and then we don't have to call updates.
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            #TODO 2024-03-06 08:16: - [ ] If we add a footer pagination widget to the window we would update it here.
            pass
        
        ## Call programmatically_update_page_idx on the children
        for a_name, a_paginated_controller in self.pagination_controllers.items():
            a_paginated_controller.programmatically_update_page_idx(updated_page_idx=page_idx, block_signals=False) # should ensure a_paginated_controller.current_page_idx is updated
            assert (a_paginated_controller.current_page_idx == page_idx), f"a_paginated_controller.current_page_idx: {a_paginated_controller.current_page_idx} does not equal the desired page index: {page_idx}"
            a_paginated_controller.perform_update_selections(defer_render=False) # update selections

        self.draw()
        

    def add_data_overlays(self, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, included_columns=None, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data and adds them to the plot.
        
        REFINEMENT: note that it only plots either 'laps' or 'ripple', not both, so it doesn't need all this data.
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import WeightedCorrelationPaginatedPlotDataProvider

        if decoder_laps_filter_epochs_decoder_result_dict is not None:
            assert isinstance(decoder_laps_filter_epochs_decoder_result_dict, dict), f"type(decoder_laps_filter_epochs_decoder_result_dict) is {type(decoder_laps_filter_epochs_decoder_result_dict)}, if it's TrackTemplates we forgot to update the function calls"

        ## Choose which columns from the filter_epochs dataframe to include on the plot.
        if included_columns is None:
            included_columns = []

        epoch_type_names_list: List[str] = [a_pagination_controller.params.active_identifying_figure_ctx.epochs for a_pagination_controller in self.pagination_controllers.values()]
        # All epoch_type_names should be the same, either 'laps' or 'ripple':
        assert (len(set(epoch_type_names_list)) == 1), f"All epoch_type_names should be the same, either 'laps' or 'ripple', but they are not: epoch_type_names_list: {epoch_type_names_list}"
        epoch_type_name: str = epoch_type_names_list[0]
        assert epoch_type_name in ['laps', 'ripple']
        if epoch_type_name == 'laps':
            decoder_decoded_epochs_result_dict = decoder_laps_filter_epochs_decoder_result_dict
        elif epoch_type_name == 'ripple':
            decoder_decoded_epochs_result_dict = decoder_ripple_filter_epochs_decoder_result_dict
        else:
            raise NotImplementedError(f"epoch_type_name: {epoch_type_name}")

        ## Add the radon_transform_lines to each of the four figures:
        for a_name, a_pagination_controller in self.pagination_controllers.items():            
            a_pagination_controller.add_data_overlays(decoder_decoded_epochs_result=decoder_decoded_epochs_result_dict[a_name], included_columns=included_columns, defer_refresh=True)

        if not defer_refresh:
            self.refresh_current_page()



    def remove_data_overlays(self, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data for this decoder and adds them to the plot.
        """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.remove_data_overlays(defer_refresh=True)

        if not defer_refresh:
            self.refresh_current_page()





    @classmethod
    def _subfn_prepare_plot_multi_decoders_stacked_epoch_slices(cls, curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict, epochs_name:str ='laps', included_epoch_indicies=None, defer_render=True, save_figure=True, **kwargs):
        """ 2024-02-14 - Adapted from the function that plots the Long/Short decoded epochs side-by-side for comparsion and updated to work with the multi-decoder track templates.
        
        ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
        
        Usage:
            (pagination_controller_L, pagination_controller_S), (fig_L, fig_S), (ax_L, ax_S), (final_context_L, final_context_S), (active_out_figure_paths_L, active_out_figure_paths_S) = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(curr_active_pipeline, defer_render=False)
        """
        debug_print = kwargs.get('debug_print', False)

        ## Extract params_kwargs
        params_kwargs = kwargs.pop('params_kwargs', {})
        params_kwargs = dict(skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, isPaginatorControlWidgetBackedMode=True) | params_kwargs
        # print(f'params_kwargs: {params_kwargs}')
        max_subplots_per_page: int = kwargs.pop('max_subplots_per_page', params_kwargs.pop('max_subplots_per_page', 8)) # kwargs overrides params_kwargs
        
        decoder_names: List[str] = track_templates.get_decoder_names()
        controlling_pagination_item_name: str = decoder_names[0] # first item # 'long_LR'
        # controlled_pagination_controller_names_list = decoder_names[1:]

        pagination_controller_dict = {}
        for i, (a_name, a_decoder) in enumerate(track_templates.get_decoders_dict().items()):
            is_controlling_widget: bool = (a_name == controlling_pagination_item_name)

            curr_params_kwargs = deepcopy(params_kwargs)
            curr_params_kwargs['is_controlled_widget'] = (not is_controlling_widget)
            if ('disable_y_label' not in curr_params_kwargs):
                # If user didn't provide an explicit 'disable_y_label' option, use the defaults which is to hide labels on all the but the controlling widget
                if is_controlling_widget:
                    curr_params_kwargs['disable_y_label'] = False
                else:
                    curr_params_kwargs['disable_y_label'] = True

            # a_name: str = 
            a_decoder_decoded_epochs_result = decoder_decoded_epochs_result_dict[a_name]
            pagination_controller_dict[a_name] = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(a_decoder_decoded_epochs_result.filter_epochs,
                                                                                                a_decoder_decoded_epochs_result,
                                                                                                xbin=a_decoder.xbin, global_pos_df=curr_active_pipeline.sess.position.df,
                                                                                                a_name=f'DecodedEpochSlices[{a_name}]', active_context=curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name),
                                                                                                max_subplots_per_page=max_subplots_per_page, debug_print=debug_print, included_epoch_indicies=included_epoch_indicies, params_kwargs=curr_params_kwargs) # , save_figure=save_figure


        # Constrains each of the plotters at least to the minimum height:
        for a_name, a_pagination_controller in pagination_controller_dict.items():
            # a_pagination_controller.params.all_plots_height
            # resize to minimum height
            a_widget = a_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
            screen = a_widget.screen()
            screen_size = screen.size()

            target_height = a_pagination_controller.params.get('scrollAreaContents_MinimumHeight', None)
            if target_height is None:
                target_height = (a_pagination_controller.params.all_plots_height + 30)
            desired_final_height = int(min(target_height, screen_size.height())) # don't allow the height to exceed the screen height.
            print(f'target_height: {target_height}, {  desired_final_height = }')
            # a_widget.size()
            a_widget.setMinimumHeight(desired_final_height) # the 30 is for the control bar

        return pagination_controller_dict

    @classmethod
    def convert_decoder_pagination_controller_dict_to_controlled(cls, pagination_controller_dict):
        """
        
        """
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget


        ## Connects the first plotter's pagination controls to the other three controllers so that they are directly driven, by the first.
        a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        a_controlling_widget = a_controlling_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget

        # controlled widgets
        controlled_pagination_controllers_list = (pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])

        new_connections_dict = []

        for a_controlled_pagination_controller in controlled_pagination_controllers_list:
            # hide the pagination widget:
            a_controlled_widget = a_controlled_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget

            if a_controlled_pagination_controller.params.get('isPaginatorControlWidgetBackedMode', True):
                # a_controlled_widget.on_paginator_control_widget_jump_to_page(page_idx=0)
                # a_connection = a_controlling_pagination_controller.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(a_controlled_pagination_controller.on_paginator_control_widget_jump_to_page) # bind connection
                a_connection = a_controlling_pagination_controller.paginator_controller_widget.jump_to_page.connect(a_controlled_pagination_controller.paginator_controller_widget.update_page_idx)
                new_connections_dict.append(a_connection)
                # a_controlled_widget.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection
                a_controlled_widget.ui.paginator_controller_widget.hide()

                ## Enable a equally sized (ThinButtonBarWidget) placeholder widget instead:
                a_controlled_widget.ui.thin_button_bar_widget = ThinButtonBarWidget()
                a_controlled_widget.ui.root_vbox.addWidget(a_controlled_widget.ui.thin_button_bar_widget) # add the pagination control widget
                a_controlled_widget.ui.thin_button_bar_widget.setFixedHeight(21)
                
                a_controlled_widget.ui.thin_button_bar_widget.label_message = "<controlled>"

        return new_connections_dict


    ## ==================================================================================================================== #
    #region Selections/Annotations                                                                                               
    # ==================================================================================================================== #
        
    # User Selections/Annotations ________________________________________________________________________________________ #
    def save_selections(self) -> Dict[str, EpochSelectionsObject]:
        """ Capture current user selections for each child controller 
        Usage:
            saved_selections_dict: Dict[str, SelectionsObject] = self.save_selections()
        """
        saved_selections_dict: Dict[str, EpochSelectionsObject] = {a_name:a_ctrlr.save_selection() for a_name, a_ctrlr in self.pagination_controllers.items()}
        return saved_selections_dict



    def print_user_annotations(self, should_copy_to_clipboard=True):
        """ Builds user annotations and outputs them. 

        >>> Prints Output Like:
        Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_LR',user_annotation='selections')] = np.array([])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_RL',user_annotation='selections')] = np.array([array([120.645, 120.862]), array([169.956, 170.16])])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_LR',user_annotation='selections')] = np.array([array([105.4, 105.563]), array([125.06, 125.21]), array([132.511, 132.791]), array([149.959, 150.254]), array([169.956, 170.16])])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_RL',user_annotation='selections')] = np.array([array([125.06, 125.21]), array([149.959, 150.254])])
        
        """
        def _subfn_listify(arr):
            return [list(a) for a in arr]
        
        def _sub_subfn_wrapped_in_brackets(s: str, bracket_strings = ("[", "]")) -> str:
                return bracket_strings[0] + s + bracket_strings[1]
            
        def _sub_subfn_format_nested_list(arr, precision:int=3, num_sep=", ", array_sep=', ') -> str:
            """
            Converts a nested list of floats into a single string,
            with each float formatted to the specified precision.
            
            arr = np.array([[491.798, 492.178], [940.016, 940.219]])
            _sub_subfn_format_nested_list(arr)

            >> '[[491.798, 492.178], [940.016, 940.219]]'

            arr = np.array([[785.738, 785.923]])
            _sub_subfn_format_nested_list(arr)
            >> '[[785.738, 785.923]]'
            """
            return _sub_subfn_wrapped_in_brackets(array_sep.join([_sub_subfn_wrapped_in_brackets(num_sep.join([f"{num:.{precision}f}" for num in row])) for row in arr]))
            


        def _subfn_build_new_nested_context_str(common_context, user_annotations):
            """ Builds a nested hierarchy of annotations like:
                with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
                    with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                        user_annotations[ctx + Ctx(decoder='long_results_obj')] = [5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123]
                        user_annotations[ctx + Ctx(decoder='short_results_obj')] = [ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131]
                    with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                        user_annotations[ctx + Ctx(decoder='long_LR')] = [[292.624, 292.808], [304.44, 304.656], [380.746, 380.904], [873.001, 873.269], [953.942, 954.258], [2212.47, 2212.54], [2214.24, 2214.44], [2214.65, 2214.68], [2219.73, 2219.87], [2422.6, 2422.82], [2451.06, 2451.23], [2452.07, 2452.22], [2453.38, 2453.55], [2470.82, 2470.97], [2473, 2473.15]]
                        user_annotations[ctx + Ctx(decoder='long_RL')] = [[487.205, 487.451], [518.52, 518.992], [802.912, 803.114], [803.592, 803.901], [804.192, 804.338], [831.621, 831.91], [893.989, 894.103], [982.605, 982.909], [1034.82, 1034.86], [1035.12, 1035.31], [1200.7, 1200.9], [1273.35, 1273.54], [1274.12, 1274.44], [1380.75, 1380.89], [1448.17, 1448.34], [1746.25, 1746.43], [1871, 1871.22], [2050.89, 2050.99], [2051.25, 2051.68]]
                        user_annotations[ctx + Ctx(decoder='short_LR')] = [[876.27, 876.452], [950.183, 950.448], [953.942, 954.258], [1044.95, 1045.45], [1129.65, 1129.84], [1259.29, 1259.44], [1259.72, 1259.88], [1511.2, 1511.43], [1511.97, 1512.06], [1549.24, 1549.37], [1558.47, 1558.68], [1560.66, 1560.75], [1561.31, 1561.41], [1561.82, 1561.89], [1655.99, 1656.21], [1730.89, 1731.07], [1734.81, 1734.95], [1861.41, 1861.53], [1909.78, 1910.04], [1967.74, 1968.09], [2036.97, 2037.33], [2038.03, 2038.27], [2038.53, 2038.73], [2042.39, 2042.64], [2070.82, 2071.03], [2153.03, 2153.14], [2191.26, 2191.39], [2192.12, 2192.36], [2193.78, 2193.99], [2194.56, 2194.76], [2200.65, 2200.8], [2201.85, 2202.03], [2219.73, 2219.87], [2248.61, 2248.81], [2249.7, 2249.92], [2313.89, 2314.06], [2422.6, 2422.82], [2462.67, 2462.74], [2482.13, 2482.61], [2484.41, 2484.48], [2530.72, 2530.92], [2531.22, 2531.3], [2556.11, 2556.38], [2556.6, 2556.92]]
                        user_annotations[ctx + Ctx(decoder='short_RL')] = [[66.6616, 66.779], [888.227, 888.465], [890.87, 891.037], [910.571, 911.048], [1014.1, 1014.28], [1200.7, 1200.9], [1211.21, 1211.33], [1214.61, 1214.83], [1317.71, 1318.22], [1333.49, 1333.69], [1380.75, 1380.89], [1381.96, 1382.32], [1448.17, 1448.34], [1499.59, 1499.71], [1744.34, 1744.59], [1798.64, 1798.77], [1970.81, 1970.95], [1994.07, 1994.25], [2050.89, 2050.99], [2051.25, 2051.68], [2132.66, 2132.98], [2203.73, 2203.82], [2204.54, 2204.66], [2317.03, 2317.12], [2330.01, 2330.16], [2331.84, 2331.96], [2403.11, 2403.41], [2456.24, 2456.33], [2456.47, 2456.57], [2457.49, 2458.01]]

            """
            def _indent_str(an_indent_level: int) -> str:
                return "\t" * an_indent_level
            
            def _with_block_template(an_indent_level: int, ctxt):
                # global indent_level
                return f"{_indent_str(an_indent_level)}with {ctxt.get_initialization_code_string(class_name_override='Ctx')} as ctx:"
            def _sub_ctxt_block_template(an_indent_level: int, ctxt):
                # global indent_level
                # indent_level = indent_level + 1
                return f"{_indent_str(an_indent_level)}with (ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}) as ctx:"
            def _leaf_ctxt_assignment_template(an_indent_level: int, ctxt, value):
                # indent_level = indent_level + 1
                return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {value}"
                # return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {list(value)}"
                # return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {_sub_subfn_format_nested_list(value)}"
            
            
            indent_level: int = 0
            code_strs: List[str] = []
            code_strs.append(_with_block_template(indent_level, common_context))
            indent_level = indent_level + 1
            common_context_user_annotations = IdentifyingContext.converting_to_relative_contexts(common_context, user_annotations)
            for k, v in common_context_user_annotations.items():
                code_strs.append(_leaf_ctxt_assignment_template(indent_level, k, v))

            # code_str = code_str + '\n'.join(code_strs)
            return code_strs


        if should_copy_to_clipboard:
            from pyphocorehelpers.programming_helpers import copy_to_clipboard
            
        from neuropy.core.user_annotations import UserAnnotationsManager
        annotations_man = UserAnnotationsManager()
        user_annotations = annotations_man.get_user_annotations()
        saved_selections_dict: Dict[str, SelectionsObject] = self.save_selections()
        saved_selections_context_dict = {a_name:v.figure_ctx.adding_context_if_missing(user_annotation='selections') for a_name, v in saved_selections_dict.items()}
        
        for a_name, a_saved_selection in saved_selections_dict.items():
            a_context = saved_selections_context_dict[a_name]
            # user_annotations[a_context] = a_saved_selection.flat_all_data_indicies[a_saved_selection.is_selected]
            user_annotations[a_context] = a_saved_selection.epoch_times

        # Updates the context. Needs to generate the code.

        use_new_concise_nested_context_format = True # 2024-03-04 - Concise 

        # ## Generate code to insert int user_annotations:
        self.ui.print('Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:')

        if use_new_concise_nested_context_format:
            # Post 2024-03-04 method of nested strings:
            # active_annotations_dict = {a_context:user_annotations[a_context] for a_name, a_context in saved_selections_context_dict.items()}
            # active_annotations_dict = {a_context:saved_selections_dict[a_name].epoch_times for a_name, a_context in saved_selections_context_dict.items()}
            active_annotations_dict = {a_context:_sub_subfn_format_nested_list(saved_selections_dict[a_name].epoch_times) for a_name, a_context in saved_selections_context_dict.items()} # active_annotations_strs_dict
            common_context = IdentifyingContext.find_longest_common_context(active_annotations_dict)
            code_strings: List[str] = _subfn_build_new_nested_context_str(common_context, user_annotations=active_annotations_dict)

        else:
            # Pre 2024-03-04 method of explicit string representations:
            code_strings: List[str] = []
            for a_name, a_saved_selection in saved_selections_dict.items():
                a_context = saved_selections_context_dict[a_name]
                if use_new_concise_nested_context_format:
                    pass
                else:
                    # a_string = f"user_annotations[{a_context.get_initialization_code_string()}] = {a_saved_selection.epoch_times}"
                    # a_string = f"user_annotations[{a_context.get_initialization_code_string()}] = array({list(a_saved_selection.epoch_times)})"
                    a_string = f"user_annotations[{a_context.get_initialization_code_string(class_name_override='Ctx')}] = {list(a_saved_selection.epoch_times)}"

                code_strings.append(a_string)
                # print(a_string)
        

        code_string: str = '\n'.join(code_strings)
        code_string = f"\n{code_string}\n" # make it easier to copy by adding newlines before and after it

        if should_copy_to_clipboard:
            copy_to_clipboard(code_string, message_print=True)
        else:
            self.ui.print(code_string)
        return code_strings
    

    def restore_selections_from_user_annotations(self, user_annotations: Optional[Dict]=None, defer_render:bool=False, **additional_selections_context):
        """
        # , source='pho_algo'
        , source='diba_evt_file' # source='diba_evt_file': # gets the annotations for the kdiba-evt file exported ripples, consistent with his 2009 paper

        """
        if user_annotations is None:
            from neuropy.core.user_annotations import UserAnnotationsManager
            annotations_man = UserAnnotationsManager()
            user_annotations = annotations_man.get_user_annotations()
        
        # Uses: paginated_multi_decoder_decoded_epochs_window, user_annotations
        # figure_ctx_dict = {a_name:v.params.active_identifying_figure_ctx for a_name, v in self.pagination_controllers.items()} 
        figure_ctx_dict = self.figure_ctx_dict
        loaded_selections_context_dict = {a_name:a_figure_ctx.adding_context_if_missing(user_annotation='selections', **additional_selections_context) for a_name, a_figure_ctx in figure_ctx_dict.items()}
        loaded_selections_dict = {a_name:user_annotations.get(a_selections_ctx, None) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}

        new_selections_dict = {a_decoder_name:a_pagination_controller.restore_selections_from_user_annotations(user_annotations, defer_render=defer_render, **additional_selections_context) for a_decoder_name, a_pagination_controller in self.pagination_controllers.items()}
        # self.draw()
        return new_selections_dict
    

    @property
    def any_good_selected_epoch_times(self) -> NDArray:
        """ returns the selected epoch times for any of the self.pagination_controllers 
        """
        concatenated_selected_epoch_times = np.concatenate([a_ctrlr.selected_epoch_times for a_name, a_ctrlr in self.pagination_controllers.items()], axis=0)
        any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
        return any_good_selected_epoch_times


    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        All the self.pagination_controllers should be displaying the same epochs, so searching each controller for the times should result in the same returned indicies.

        Uses:
            self.pagination_controllers
        """
        from pyphocorehelpers.indexing_helpers import NumpyHelpers
        any_good_epoch_idxs_list = [a_ctrlr.find_data_indicies_from_epoch_times(epoch_times) for a_name, a_ctrlr in self.pagination_controllers.items()]
        assert NumpyHelpers.all_array_equal(any_good_epoch_idxs_list), f"all indicies should be identical, but they are not! any_good_epoch_idxs_list: {any_good_epoch_idxs_list}"
        any_good_epoch_idxs: NDArray = any_good_epoch_idxs_list[0]
        return any_good_epoch_idxs

    #endregion Selections/Annotations ______________________________________________________________________________________________________ #


    ## ==================================================================================================================== #
    #region Export/Output                                                                                              
    # ==================================================================================================================== #
    
    # Export/Output ______________________________________________________________________________________________________ #
    def export_decoder_pagination_controller_figure_page(self, curr_active_pipeline, **kwargs):
        """ exports each pages single-decoder figures separately

        Usage:
            export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

        """
        import matplotlib as mpl

        output_figure_kwargs = dict(write_vector_format=True, write_png=True) | kwargs
        pagination_controller_dict = self.pagination_controllers

        out_fig_paths_dict = {}

        for a_name, a_pagination_controller in pagination_controller_dict.items():
            display_context = a_pagination_controller.params.get('active_identifying_figure_ctx', IdentifyingContext())

            # Get context for current page of items:
            current_page_idx: int = int(a_pagination_controller.current_page_idx)
            a_paginator = a_pagination_controller.paginator
            total_num_pages = int(a_paginator.num_pages)
            page_context = display_context.overwriting_context(page=current_page_idx, num_pages=total_num_pages)
            self.ui.print(page_context)

            ## Get the figure/axes:
            a_plots = a_pagination_controller.plots # RenderPlots
            # a_params = a_pagination_controller.params
            
            # with mpl.rc_context({'figure.figsize': (8.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, }):
            with mpl.rc_context({'figure.figsize': (16.8, 4.8), 'figure.dpi': '420', 'savefig.transparent': True, 'ps.fonttype': 42, }):
                figs = a_plots.fig
                # axs = a_plots.axs
                active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=page_context, fig=figs, **output_figure_kwargs)
                out_fig_paths_dict[final_context] = active_out_figure_paths

        # end for

        return out_fig_paths_dict

    def export_all_pages(self, curr_active_pipeline, **kwargs):
        """ exports each pages single-decoder figures separately

        Usage:
            export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

        """
        output_figure_kwargs = dict(write_vector_format=True, write_png=True) | kwargs

        # assert self.isPaginatorControlWidgetBackedMode
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        a_paginator = a_controlling_pagination_controller.paginator
        total_num_pages = int(a_paginator.num_pages)
        page_idx_sweep = np.arange(total_num_pages)
        page_num_sweep = page_idx_sweep + 1 # switch to 1-indexed
        # page_num_sweep
        print(f'export_all_pages(...): preparing to export {total_num_pages} pages from 4 decoders:')

        out_fig_paths_dict_list = {}

        for a_page_idx, a_page_num in zip(page_idx_sweep, page_num_sweep):
            print(f'switching to page: a_page_idx: {a_page_idx}, a_page_num: {a_page_num} of total_num_pages: {total_num_pages}')
            # a_pagination_controller.on_paginator_control_widget_jump_to_page(page_idx=a_page_idx)
            # a_pagination_controller.ui.mw.draw()
            # export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

            self.jump_to_page(page_idx=a_page_idx)
            self.draw()
            out_fig_paths_dict_list[a_page_idx] = self.export_decoder_pagination_controller_figure_page(curr_active_pipeline=curr_active_pipeline, **output_figure_kwargs)

            # import matplotlib as mpl

            # pagination_controller_dict = self.pagination_controllers

            # page_idx_sweep = np.arange(total_num_pages)
            # page_num_sweep = page_idx_sweep + 1 # switch to 1-indexed
            
            # # paginator_dict = {a_name:a_pagination_controller.paginator for a_name, a_pagination_controller in pagination_controller_dict.items()}

            # for a_name, a_pagination_controller in pagination_controller_dict.items():
            #     display_context = a_pagination_controller.params.get('active_identifying_figure_ctx', IdentifyingContext())

            #     # Get context for current page of items:
            #     current_page_idx: int = int(a_pagination_controller.current_page_idx)
            #     a_paginator = a_pagination_controller.paginator
            #     total_num_pages = int(a_paginator.num_pages)
            #     page_context = display_context.overwriting_context(page=current_page_idx, num_pages=total_num_pages)
            #     self.ui.print(page_context)

            #     page_idx_sweep = np.arange(total_num_pages)
            #     page_num_sweep = page_idx_sweep + 1 # switch to 1-indexed
                

            #     ## Get the figure/axes:
            #     a_plots = a_pagination_controller.plots # RenderPlots
            #     a_params = a_pagination_controller.params
                
            #     # with mpl.rc_context({'figure.figsize': (8.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, }):
            #     with mpl.rc_context({'figure.figsize': (16.8, 4.8), 'figure.dpi': '420', 'savefig.transparent': True, 'ps.fonttype': 42, }):
            #         figs = a_plots.fig
            #         axs = a_plots.axs
            #         curr_active_pipeline.output_figure(final_context=page_context, fig=figs, write_vector_format=True)

        print(f'\tdone.')
        return out_fig_paths_dict_list
    
    
    #endregion Export/Output ______________________________________________________________________________________________________ #

    # ==================================================================================================================== #
    # MatplotlibTimeSynchronizedWidget Wrappers                                                                            #
    # ==================================================================================================================== #
                
    # def getFigure(self):
    #     return self.plots.fig
        
    

    def draw(self):
        """ Calls .draw() on all children MatplotlibTimeSynchronizedWidget items. 
        Successfully redraws items.

        """
        #TODO 2023-07-06 15:05: - [ ] PERFORMANCE - REDRAW
        for a_name, a_child_paginated_widget in self.paginated_widgets.items():
            # a_child_paginated_widget.ui.canvas.draw()
            a_child_paginated_widget.draw()
        
    

    def refresh_current_page(self):
        """ called to refresh the currently selected page for all controllers (to redraw the data widgets or axes).
        """
        # if self.debug_print:
        #     self.ui.print(f'PhoPaginatedMultiDecoderDecodedEpochsWindow.refresh_current_page():') # for page_idx == max_index this is called but doesn't continue
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.refresh_current_page()

    def update_params(self, **updated_values):
        """ called to change the .params on all of the child controllers simultaneously.
         
          
        
            paginated_multi_decoder_decoded_epochs_window.update_params(posterior_heatmap_imshow_kwargs = dict(vmin=0.0))
            paginated_multi_decoder_decoded_epochs_window.refresh_current_page()


        """
        # if self.debug_print:
        #     self.ui.print(f'PhoPaginatedMultiDecoderDecodedEpochsWindow.refresh_current_page():') # for page_idx == max_index this is called but doesn't continue
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.params.update(**updated_values)
        
    # ==================================================================================================================== #
    # Passthrough methods/properties                                                                                       #
    # ==================================================================================================================== #

    # def get_children_props(self, prop_name):
    #     # return [getattr(child, prop_name) for child in self.findChildren(QWidget)]
    #     return {a_name:getattr(a_pagination_controller, prop_name) for a_name, a_pagination_controller in self.pagination_controllers.items()}

    def get_children_props(self, prop_path):
        def get_nested_prop(obj, prop_path):
            attrs = prop_path.split(".")
            for attr in attrs:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            return obj

        return {a_name:get_nested_prop(a_pagination_controller, prop_path) for a_name, a_pagination_controller in self.pagination_controllers.items()}

    # def set_child_props(self, prop_name, value):
    #     for child in self.findChildren(QWidget):
    #         setattr(child, prop_name, value)


    def show_message(self, message: str, durationMs:int=4000):
        """ show a toast message """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.show_message(message=message, durationMs=durationMs)



    def update_titles(self, window_title: str, children_titles: Optional[Dict[str, Optional[str]]] = None):
        """ sets the suptitle and window title for the figure """
        # Set the window title:
        self.setWindowTitle(window_title)

        ## Update embedded figures:
        if children_titles is not None:
            for a_name, a_pagination_controller in self.pagination_controllers.items():
                desired_child_title = children_titles.get(a_name, None)
                if desired_child_title is not None:
                    a_pagination_controller.ui.mw.fig.suptitle(desired_child_title, wrap=True) # set the plot suptitle
                    a_pagination_controller.ui.mw.draw()
        

    def enable_middle_click_selected_epoch_times_to_clipboard(self, is_enabled:bool=True):
        """ sets the copying of epoch times to the clipboard """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            # a_pagination_controller.params.debug_print = True
            if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
                a_pagination_controller.params['on_middle_click_item_callbacks'] = {}

            if is_enabled:
                a_pagination_controller.params.on_middle_click_item_callbacks['copy_to_epoch_times_to_clipboard_callback'] = ClickActionCallbacks.copy_to_epoch_times_to_clipboard_callback
            else:
                a_pagination_controller.params.on_middle_click_item_callbacks.pop('copy_to_epoch_times_to_clipboard_callback')


def interactive_good_epoch_selections(annotations_man: UserAnnotationsManager, curr_active_pipeline) -> dict:
    """Allows the user to interactively select good epochs and generate hardcoded user_annotation entries from the results:
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import interactive_good_epoch_selections
        
        user_annotation_man = UserAnnotationsManager()
        user_annotations = user_annotation_man.get_user_annotations()
        user_annotations = interactive_good_epoch_selections(annotations_man=user_annotation_man, curr_active_pipeline=curr_active_pipeline) # perform interactive selection. Should block here.
        
        
    History:
        Extracted from `UserAnnotationsManager.interactive_good_epoch_selections(...)
        
        Old usage:
            user_annotation_man = UserAnnotationsManager()
            user_annotations = user_annotation_man.get_user_annotations()
            user_annotation_man.interactive_good_epoch_selections(curr_active_pipeline=curr_active_pipeline) # perform interactive selection. Should block here.
    """
    ## Stacked Epoch Plot
    
    # from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController


    ## Stacked Epoch Plot
    example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=False)
    pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
    ax_L, ax_S = example_stacked_epoch_graphics.axes
    figure_context_L, figure_context_S = example_stacked_epoch_graphics.context


    # user_annotations = UserAnnotationsManager.get_user_annotations()
    user_annotations = annotations_man.get_user_annotations()

    ## Capture current user selection
    saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
    saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
    final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
    final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
    user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
    user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
    # Updates the context. Needs to generate the code.

    ## Generate code to insert int user_annotations:
    print('Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:')
    print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
    print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")

    return user_annotations

def align_decoder_pagination_controller_windows(pagination_controller_dict):
    """ resizes and aligns all windows. Not needed with PhoPaginatedMultiDecoderDecodedEpochsWindow (only used when plotting in separate windows) 
    Usage:
        align_decoder_pagination_controller_windows(pagination_controller_dict)

    """
    from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers, DesiredWidgetLocation, WidgetGeometryInfo
    ## Connects the first plotter's pagination controls to the other three controllers so that they are directly driven, by the first.
    a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
    a_controlling_widget = a_controlling_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
    # controlled widgets
    controlled_pagination_controllers_list = (pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])

    fixed_height_pagination_control_bar: float = 21.0
    target_height: float = a_controlling_widget.window().height()
    ratio_content_height = (target_height - fixed_height_pagination_control_bar) / target_height
    print(f'fixed_height_pagination_control_bar: {fixed_height_pagination_control_bar}, target_height: {target_height}, ratio_content_height: {ratio_content_height}')

    target_window = a_controlling_widget.window()
    for a_controlled_pagination_controller in controlled_pagination_controllers_list:
        # hide the pagination widget:
        a_controlled_widget = a_controlled_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
        WidgetPositioningHelpers.align_window_edges(target_window, a_controlled_widget.window(), relative_position = 'right_of', resize_to_main=(1.0, ratio_content_height)) # use ratio_content_height to compensate for the lack of a pagination scroll bar
        target_window = a_controlled_widget.window() # update to reference the newly moved window
        ratio_content_height = 1.0 # after the first window, 1.0 should be used since they're all the same height


# ==================================================================================================================== #
# 2024-04-25 - Factoring Out Helper GUI Code                                                                           #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['raster', 'attached'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-29 17:14', related_items=[])
def _build_attached_raster_viewer(paginated_multi_decoder_decoded_epochs_window, track_templates, active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df):
    """ creates a new RankOrderRastersDebugger for use by `paginated_multi_decoder_decoded_epochs_window`.
    
    You can middle-click on any epoch heatmap in `paginated_multi_decoder_decoded_epochs_window` to display that corresponding epoch in the RankOrderRastersDebugger
    
    

    paginated_multi_decoder_decoded_epochs_window

    Captures: paginated_multi_decoder_decoded_epochs_window
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import _build_attached_raster_viewer
        _out_ripple_rasters = _build_attached_raster_viewer(paginated_multi_decoder_decoded_epochs_window, track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df=filtered_ripple_simple_pf_pearson_merged_df)
    
    
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
    # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
    global_spikes_df = deepcopy(active_spikes_df)
    _out_ripple_rasters: RankOrderRastersDebugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, deepcopy(filtered_ripple_simple_pf_pearson_merged_df),
                                                                                                    track_templates, None,
                                                                                                        None, None,
                                                                                                        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right']))),
                                                                                                        )
    _out_ripple_rasters.set_top_info_bar_visibility(False)
    _out_ripple_rasters.set_bottom_controls_visibility(False)

    ## Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior.
    def update_attached_raster_viewer_epoch_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior. 
        called when the user middle-clicks an epoch 
        
        captures: _out_ripple_rasters
        """
        print(f'update_attached_raster_viewer_epoch_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                print(f'start_t: {start_t}')
                _out_ripple_rasters.programmatically_update_epoch_IDX_from_epoch_start_time(start_t)

    ## Attach the update to the pagination controllers:
    for a_name, a_pagination_controller in paginated_multi_decoder_decoded_epochs_window.pagination_controllers.items():
        # a_pagination_controller.params.debug_print = True
        if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
            a_pagination_controller.params['on_middle_click_item_callbacks'] = {}
        a_pagination_controller.params.on_middle_click_item_callbacks['update_attached_raster_viewer_epoch_callback'] = update_attached_raster_viewer_epoch_callback

        
    return _out_ripple_rasters


def build_attached_raster_viewer_widget(paginated_multi_decoder_decoded_epochs_window, track_templates, active_spikes_df: pd.DataFrame, filtered_ripple_simple_pf_pearson_merged_df: pd.DataFrame):
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import build_attached_raster_viewer_widget

    _out_ripple_rasters, update_attached_raster_viewer_epoch_callback = build_attached_raster_viewer_widget(paginated_multi_decoder_decoded_epochs_window=paginated_multi_decoder_decoded_epochs_window, track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df=filtered_ripple_simple_pf_pearson_merged_df)

    """

    print(f'Middle-click any epoch to adjust the Attached Raster Window to that epoch.')
    
    _out_ripple_rasters = _build_attached_raster_viewer(paginated_multi_decoder_decoded_epochs_window, track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df=filtered_ripple_simple_pf_pearson_merged_df)

    ## Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior.
    @function_attributes(short_name=None, tags=['callback', 'raster'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-29 17:13', related_items=[])
    def update_attached_raster_viewer_epoch_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior. 
        called when the user middle-clicks an epoch 
        
        captures: _out_ripple_rasters
        """
        print(f'update_attached_raster_viewer_epoch_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                print(f'start_t: {start_t}')
                _out_ripple_rasters.programmatically_update_epoch_IDX_from_epoch_start_time(start_t)

    for a_name, a_pagination_controller in paginated_multi_decoder_decoded_epochs_window.pagination_controllers.items():
        # a_pagination_controller.params.debug_print = True
        if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
            a_pagination_controller.params['on_middle_click_item_callbacks'] = {}
        a_pagination_controller.params.on_middle_click_item_callbacks['update_attached_raster_viewer_epoch_callback'] = update_attached_raster_viewer_epoch_callback


    _out_ripple_rasters.setWindowTitle(f'Template Rasters <Controlled by DecodedEpochSlices window>')

    return _out_ripple_rasters, update_attached_raster_viewer_epoch_callback