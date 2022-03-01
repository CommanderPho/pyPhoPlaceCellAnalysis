import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.indexing_helpers import interleave_elements

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



def plot_3d_raster_plot(curr_spikes_df):
# def plot_3d_raster_plot(x=np.linspace(-10,10,100), y=np.random.normal(size=100), z=None):
    # curr_unit_id = curr_spikes_df['unit_id'].to_numpy() # this will map to the y position
    # curr_spike_t = curr_spikes_df[curr_spikes_df.spikes.time_variable_name].to_numpy() # this will map to the depth dimension in 3D or x-pos in 2D
    unit_ids = np.unique(curr_spikes_df['unit_id'].to_numpy())
    n = len(unit_ids)
    
    # print(f'plot_3d_raster_plot(np.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)})')
    # print(f'\t x: {x}\n y: {y}')
    
    app = pg.mkQApp("Pyqtgraph 3D Raster Plot")
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)
    w = gl.GLViewWidget()
    w.show()
    w.resize(1000,600)
    w.setWindowTitle('pyqtgraph: 3D Raster Spikes Plotting')
    w.setCameraPosition(distance=40)
    # Add axes planes:
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    w.addItem(gx)
    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    w.addItem(gy)
    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    w.addItem(gz)

    # Custom 3D raster plot:
    y = np.linspace(-10,10,n) # the line location I think
    x = np.linspace(-10,10,100) # the temporal location
    
    for cell_id in unit_ids:
        # Filter the dataframe using that column and value from the list
        curr_cell_df = curr_spikes_df[curr_spikes_df['unit_id']==cell_id].copy()
        curr_unit_id = curr_cell_df['unit_id'].to_numpy() # this will map to the y position
        curr_spike_t = curr_cell_df[curr_spikes_df.spikes.time_variable_name].to_numpy() # this will map 

        yi = y[cell_id]
        z = curr_spike_t[np.arange(20)] # get the first 20 spikes for each
        pts = np.column_stack([x, np.full_like(x, yi), z])
        plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((cell_id,n*1.3)), width=(cell_id+1)/10., antialias=True)
        w.addItem(plt)
        
    
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