import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.indexing_helpers import interleave_elements, partition

# Windowing helpers for spikes_df:
from PhoPositionalData.plotting.visualization_window import VisualizationWindow # Used to build "Windows" into the data points such as the window defining the fixed time period preceeding the current time where spikes had recently fired, etc.
from numpy.lib.stride_tricks import sliding_window_view


def plot_raster_plot(x=np.arange(100), y=np.random.normal(size=100)):
    
    print(f'plot_raster_plot(np.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)})')
    print(f'\t x: {x}\n y: {y}')
    
    app = pg.mkQApp("Pyqtgraph Raster Plot")
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)
    
    win = pg.GraphicsLayoutWidget(show=True, title="Pyqtgraph Raster Plot")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph: Raster Spikes Plotting')
    
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    # Actually setup the plot:
    p1 = win.addPlot(title="SpikesDataframe", x=x, y=y, connect='pairs')
    p1.setLabel('bottom', 'Timestamp', units='[sec]') # set the x-axis label

    return [p1], win, app


def _display_pyqtgraph_raster_plot(curr_spikes_df, debug_print=False):
    """ Renders a primitive 2D raster plot using pyqtgraph.
    
    curr_epoch_name = 'maze1'
    curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
    curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
    curr_spikes_df = curr_sess.spikes_df
    _display_pyqtgraph_raster_plot(curr_spikes_df)
    
    """
    curr_unit_id = curr_spikes_df['unit_id'].to_numpy() # this will map to the y position
    curr_spike_t = curr_spikes_df[curr_spikes_df.spikes.time_variable_name].to_numpy() # this will map to the depth dimension in 3D or x-pos in 2D

    if debug_print:
        print(f'_test_display_pyqtgraph_raster_plot(np.shape(curr_unit_id): {np.shape(curr_unit_id)}, np.shape(curr_spike_t): {np.shape(curr_spike_t)})')
        print(f'\t curr_unit_id: {curr_unit_id}\n curr_spike_t: {curr_spike_t}')
    
    # For the unit Ids, perform a transformation:
    normalized_unit_ids = curr_unit_id / np.max(curr_unit_id)
    upper_unit_bounds = (normalized_unit_ids*0.9) + 0.05 # 0.05 to 0.95
    lower_unit_bounds = upper_unit_bounds - 0.05 # 0.00 to 0.90
    # curr_unit_id_repeats = curr_unit_id.copy()
    # curr_spike_t_repeats = curr_spike_t.copy()
    curr_spike_t_repeats = np.atleast_2d(curr_spike_t.copy())
    lower_unit_bounds = np.atleast_2d(lower_unit_bounds)
    upper_unit_bounds = np.atleast_2d(upper_unit_bounds)
    if debug_print:
        print(f'np.atleast_2d(lower_unit_bounds): {np.shape(np.atleast_2d(lower_unit_bounds))}') # (1, 819170)
    
    # the paired arrays should be twice as long as the original arrays and are to be used with the connected='pair' argument
    # curr_paired_unit_id = interleave_elements(curr_unit_id, curr_unit_id_repeats)
    curr_paired_unit_id = np.squeeze(interleave_elements(lower_unit_bounds.T, upper_unit_bounds.T)) # use the computed ranges instead
    curr_paired_spike_t = np.squeeze(interleave_elements(curr_spike_t_repeats.T, curr_spike_t_repeats.T))
    if debug_print:
        print(f'curr_paired_unit_id: {np.shape(curr_paired_unit_id)}, curr_paired_spike_t: {np.shape(curr_paired_spike_t)}')
    
    # out_q_path = pg.arrayToQPath(curr_paired_spike_t, curr_paired_unit_id, connect='pairs', finiteCheck=True) # connect='pairs' details how to connect points in the path
    
    return plot_raster_plot(x=curr_paired_spike_t, y=curr_paired_unit_id)    
    
    # return plot_raster_plot(curr_spike_t, curr_unit_id)
 
    # np.unique(curr_unit_id) # np.arange(62) (0-62)
    # curr_spike_t

    # app = pg.mkQApp()
    # win = pg.GraphicsLayoutWidget(show=True)

    # p1 = win.addPlot()
    # p1.setLabel('bottom', 'Timestamp', units='[sec]') # set the x-axis label
    
    
    # # p1.setYRange(0, nPlots)
    # # p1.setXRange(0, nSamples)
    
    # data1 = np.random.normal(size=300) # 300x300
    # connected = np.round(np.random.rand(300)) # 300x300
    
    
    # # add the curve:
    # curve1 = p1.plot(data1, connect=connected)
    # def update1():
    #     global data1, connected
    #     data1[:-1] = data1[1:]  # shift data in the array one sample left
    #                             # (see also: np.roll)
    #     connected = np.roll(connected, -1)
    #     data1[-1] = np.random.normal()
    #     curve1.setData(data1, connect=connected)

    # timer = pg.QtCore.QTimer()
    # timer.timeout.connect(update1)
    # timer.start(50)
    # # timer.stop()    
    # app.exec_()

def _compute_windowed_spikes_raster(curr_spikes_df, render_window_duration=6.0):
    # curr_spikes_df
    
    
    recent_spikes_window = VisualizationWindow(duration_seconds=6.0, sampling_rate=self.active_session.position.sampling_rate) # increasing this increases the length of the position tail
    curr_view_window_length_samples = self.params.recent_spikes_window.duration_num_frames # number of samples the window should last
    print('recent_spikes_window - curr_view_window_length_samples - {}'.format(curr_view_window_length_samples))
    ## Build the sliding windows:
    # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
    active_epoch_position_linear_indicies = np.arange(np.size(self.active_session.position.time))
    pre_computed_window_sample_indicies = recent_spikes_window.build_sliding_windows(active_epoch_position_linear_indicies)
    # print('pre_computed_window_sample_indicies: {}\n shape: {}'.format(pre_computed_window_sample_indicies, np.shape(pre_computed_window_sample_indicies)))

    ## New Pre Computed Indicies Way:
    z_fixed = np.full((recent_spikes_window.duration_num_frames,), 1.1) # this seems to be about position, not spikes
    
        
    
    unit_split_spikes_df = partition(curr_spikes_df, 'unit_id') # split on the unitID
    
    


def plot_3d_raster_plot(curr_spikes_df):
    """ plots a 3-dimensional raster plot for neural spiking data. """
# def plot_3d_raster_plot(x=np.linspace(-10,10,100), y=np.random.normal(size=100), z=None):
    # curr_unit_id = curr_spikes_df['unit_id'].to_numpy() # this will map to the y position
    # curr_spike_t = curr_spikes_df[curr_spikes_df.spikes.time_variable_name].to_numpy() # this will map to the depth dimension in 3D or x-pos in 2D
    unit_ids = np.unique(curr_spikes_df['unit_id'].to_numpy())
    n = len(unit_ids)
    render_window_duration = 60.0 # in seconds, 1minute by default
    spike_start_z = -10.0
    spike_end_z = 0.1
    
    n_half_cells = np.ceil(float(n)/2.0)
    n_full_cell_grid = 2.0 * n_half_cells # could be one more than n
    half_render_window_duration = np.ceil(float(render_window_duration)/2.0) # 10 by default
    
    print(f'plot_3d_raster_plot(...): unit_ids: {unit_ids}, n: {n}')
    # print(f'plot_3d_raster_plot(np.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)})')
    # print(f'\t x: {x}\n y: {y}')
    
    app = pg.mkQApp("Pyqtgraph 3D Raster Plot")
    w = gl.GLViewWidget()
    w.show()
    w.resize(1000,600)
    w.setWindowTitle('pyqtgraph: 3D Raster Spikes Plotting')
    w.setCameraPosition(distance=40)
    
    
    # Add axes planes:
    
    # X-plane:
    gx = gl.GLGridItem(color=(255, 155, 155, 76.5))
    gx.rotate(90, 0, 1, 0)
    gx.translate(-half_render_window_duration, 0, 0) # shift backwards
    gx.setSize(20, n_full_cell_grid) # std size in z-dir, n_cell size across
    gx.setSpacing(10.0, 1) 
    w.addItem(gx)
    
    # Y-plane:
    gy = gl.GLGridItem(color=(155, 255, 155, 76.5))
    gy.rotate(90, 1, 0, 0)
    # gy.translate(0, -10, 0)
    gy.translate(0, -n_half_cells, 0) # offset by half the number of units in the -y direction
    gy.setSize(render_window_duration, 20)
    # gy.setSpacing(1, 1)
    w.addItem(gy)
    
    # XY-plane (with normal in z-dir):
    gz = gl.GLGridItem(color=(155, 155, 255, 76.5))
    gz.translate(0, 0, -10) # Shift down by 10 units in the z-dir
    gz.setSize(render_window_duration, n_full_cell_grid)
    gz.setSpacing(20.0, 1)
    # gz.setSize(n_full_cell_grid, n_full_cell_grid)
    w.addItem(gz)

    
    # # For scatter plot:
    # pos = np.empty((53, 3))
    # size = np.empty((53))
    # color = np.empty((53, 4))
    # pos[0] = (1,0,0); size[0] = 0.5;   color[0] = (1.0, 0.0, 0.0, 0.5)


    # Custom 3D raster plot:
    # y = np.linspace(-10,10,n) # the line location I think
    # # x = np.linspace(-10,10,100) # the temporal location, size 100 by default
    # x = np.linspace(-10, 25, 100) # the temporal location, size 100 by default
    # # x = np.linspace(0.0, render_window_duration, 100) # the temporal location, size 100 by default
    
    # New attempt
    y = np.linspace(-n_half_cells, n_half_cells, n) + 0.5 # add 0.5 so they're centered
    # x = np.linspace(-half_render_window_duration, half_render_window_duration, 100)
    
    # Filter based on the current spikes window to show:
    
    curr_render_window_start_time = 30.0
    curr_render_window_end_time = curr_render_window_start_time + render_window_duration
    curr_time_windowed_spikes_df = curr_spikes_df[curr_spikes_df[curr_spikes_df.spikes.time_variable_name].between(curr_render_window_start_time, curr_render_window_end_time)]
    
    
    # np.interp(a, (a.min(), a.max()), (-1, +1))
    
    # Plot each unit one at a time:
    for cell_id in unit_ids:
        curr_color = pg.mkColor((cell_id, n*1.3))
        # curr_color.SetAlpha(120) # alpha should be between 0-255
        curr_color.setAlphaF(0.5)
        print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
        
        
                        
        # Filter the dataframe using that column and value from the list
        curr_cell_df = curr_time_windowed_spikes_df[curr_time_windowed_spikes_df['unit_id']==cell_id].copy()
        # curr_unit_id = curr_cell_df['unit_id'].to_numpy() # this will map to the y position
        curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
        yi = y[cell_id] # get the correct y-position for all spikes of this cell
        # print(f'cell_id: {cell_id}, yi: {yi}')
        # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
        curr_x = np.interp(curr_spike_t, (curr_render_window_start_time, curr_render_window_end_time), (-half_render_window_duration, +half_render_window_duration))
        curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
        
        # Z-positions:
        # z = curr_spike_t[np.arange(100)] # get the first 20 spikes for each
        spike_bottom_zs = np.full_like(curr_x, spike_start_z)
        spike_top_zs = np.full_like(curr_x, spike_end_z)
        curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
     
        # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        # sp1.translate(5,5,0)
        # w.addItem(sp1)
        
        # Build lines:
        pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
        # pts = np.column_stack([x, np.full_like(x, yi), z]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
        # plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((cell_id,n*1.3)), width=(cell_id+1)/10., antialias=True)
        plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
        w.addItem(plt)

        # # Adds a helper widget that displays the x/y/z vector at the origin:
        # ref_axes_indicator = gl.GLAxisItem()
        # ref_axes_indicator.setSize(x=10.0, y=10.0, z=5.0)
        # w.addItem(ref_axes_indicator)
    
    # # Example 3D wave plot made of lines:
    # n = 51
    # y = np.linspace(-10,10,n) # the line location I think
    # x = np.linspace(-10,10,100) # the temporal location
    # for i in range(n):
    #     yi = y[i]
    #     if z is None:
    #         d = np.hypot(x, yi)
    #         z = 10 * np.cos(d) / (d+1)
    #     pts = np.column_stack([x, np.full_like(x, yi), z])
    #     plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((i,n*1.3)), width=(i+1)/10., antialias=True)
    #     w.addItem(plt)

    return w, app





if __name__ == '__main__':
    # win, app = plot_raster_plot()
    pg.exec()