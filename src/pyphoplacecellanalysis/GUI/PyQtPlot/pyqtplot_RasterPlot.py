import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


# Windowing helpers for spikes_df:
from pyphoplacecellanalysis.PhoPositionalData.plotting.visualization_window import VisualizationWindow # Used to build "Windows" into the data points such as the window defining the fixed time period preceeding the current time where spikes had recently fired, etc.
from numpy.lib.stride_tricks import sliding_window_view

# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QHBoxLayout, QSlider, QCheckBox 


""" This example from the internet displays a dark-themed MainWindow containing an interactive pyqtgraph plot that allows you to inspect eigenvalues and drag an animated slider. """



# Note that these raster plots could implement some variant of HideShowSpikeRenderingMixin, SpikeRenderingMixin, etc but these classes frankly suck. 

def plot_raster_plot(x=np.arange(100), y=np.random.normal(size=100)):
    """ Called by _display_pyqtgraph_raster_plot """
    
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
    curr_fragile_linear_neuron_IDX = curr_spikes_df['fragile_linear_neuron_IDX'].to_numpy() # this will map to the y position
    curr_spike_t = curr_spikes_df[curr_spikes_df.spikes.time_variable_name].to_numpy() # this will map to the depth dimension in 3D or x-pos in 2D

    if debug_print:
        print(f'_test_display_pyqtgraph_raster_plot(np.shape(curr_fragile_linear_neuron_IDX): {np.shape(curr_fragile_linear_neuron_IDX)}, np.shape(curr_spike_t): {np.shape(curr_spike_t)})')
        print(f'\t curr_fragile_linear_neuron_IDX: {curr_fragile_linear_neuron_IDX}\n curr_spike_t: {curr_spike_t}')
    
    # For the unit Ids, perform a transformation:
    normalized_fragile_linear_neuron_IDXs = curr_fragile_linear_neuron_IDX / np.max(curr_fragile_linear_neuron_IDX)
    upper_unit_bounds = (normalized_fragile_linear_neuron_IDXs*0.9) + 0.05 # 0.05 to 0.95
    lower_unit_bounds = upper_unit_bounds - 0.05 # 0.00 to 0.90
    # curr_fragile_linear_neuron_IDX_repeats = curr_fragile_linear_neuron_IDX.copy()
    # curr_spike_t_repeats = curr_spike_t.copy()
    curr_spike_t_repeats = np.atleast_2d(curr_spike_t.copy())
    lower_unit_bounds = np.atleast_2d(lower_unit_bounds)
    upper_unit_bounds = np.atleast_2d(upper_unit_bounds)
    if debug_print:
        print(f'np.atleast_2d(lower_unit_bounds): {np.shape(np.atleast_2d(lower_unit_bounds))}') # (1, 819170)
    
    # the paired arrays should be twice as long as the original arrays and are to be used with the connected='pair' argument
    # curr_paired_fragile_linear_neuron_IDX = interleave_elements(curr_fragile_linear_neuron_IDX, curr_fragile_linear_neuron_IDX_repeats)
    curr_paired_fragile_linear_neuron_IDX = np.squeeze(interleave_elements(lower_unit_bounds.T, upper_unit_bounds.T)) # use the computed ranges instead
    curr_paired_spike_t = np.squeeze(interleave_elements(curr_spike_t_repeats.T, curr_spike_t_repeats.T))
    if debug_print:
        print(f'curr_paired_fragile_linear_neuron_IDX: {np.shape(curr_paired_fragile_linear_neuron_IDX)}, curr_paired_spike_t: {np.shape(curr_paired_spike_t)}')
    
    # out_q_path = pg.arrayToQPath(curr_paired_spike_t, curr_paired_fragile_linear_neuron_IDX, connect='pairs', finiteCheck=True) # connect='pairs' details how to connect points in the path
    
    return plot_raster_plot(x=curr_paired_spike_t, y=curr_paired_fragile_linear_neuron_IDX)    
    
    # return plot_raster_plot(curr_spike_t, curr_fragile_linear_neuron_IDX)
 
    # np.unique(curr_fragile_linear_neuron_IDX) # np.arange(62) (0-62)
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
    """ TODO: Not yet implemented: """
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
    
        
    
    unit_split_spikes_df = partition(curr_spikes_df, 'fragile_linear_neuron_IDX') # split on the unitID
    
    



if __name__ == '__main__':
    # win, app = plot_raster_plot()
    pg.exec()