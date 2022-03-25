from pyqtgraph.Qt import QtCore
import numpy as np

from pyphoplacecellanalysis.General.Model.TimeWindow import TimeWindow



""" Windowed Spiking Datasource Features

Transforming the events into either 2D or 3D representations for visualization should NOT be part of this class' function.
Separate 2D and 3D event visualization functions should be made to transform events from this class into appropriate point/datastructure representations for the visualization framework being used.

# Local window properties
[X] Given by .active_time_window
    Get (window_start, window_end) times

# Global data properties
[X] Given by .total_df_start_end_times
    Get (earliest_datapoint_time, latest_datapoint_time) # globally, for the entire timeseries


"""
class SpikesDataframeWindow(TimeWindow):
# class SpikesDataframeWindow(QtCore.QObject):
    """ a zoomable (variable sized) window into a dataframe with a time axis
    Used by Spike3DRaster
    
    active_window_start_time can be adjusted to set the location of the current window.

    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(curr_spikes_df, window_duration=render_window_duration)
        curr_spikes_df_window

    """
    spike_dataframe_changed_signal = QtCore.pyqtSignal() # signal emitted when the spike dataframe is changed, which might change the number of units, number of spikes, and other properties.
    
    # Require TimeWindow and Datasource:
    @property
    def active_windowed_df(self):
        """The dataframe sliced to the current time window (active_time_window)"""
        return self.df[self.df[self.df.spikes.time_variable_name].between(self.active_time_window[0], self.active_time_window[1])]

    @property
    def active_window_num_spikes(self):
        """The number of spikes (across all units) in the active window."""
        return self.active_windowed_df.shape[0]
    
    
    # Properties belonging to DataframeDatasource:
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest spiketimes in the total df """
        earliest_df_time = np.nanmin(self.df[self.df.spikes.time_variable_name])
        latest_df_time = np.nanmax(self.df[self.df.spikes.time_variable_name])
        
        df_timestamps = self.df[self.df.spikes.time_variable_name].to_numpy()
        earliest_df_time = df_timestamps[0]
        latest_df_time = df_timestamps[-1]
        return (earliest_df_time, latest_df_time)
            
    ##### Get/Set Properties ####:
    @property
    def df(self):
        """The df property."""
        return self._df
    @df.setter
    def df(self, value):
        self._df = value
        self.spike_dataframe_changed_signal.emit()
        
    
    # Initializer:
    def __init__(self, spikes_df, window_duration=15.0, window_start_time=0.0):
        TimeWindow.__init__(self, window_duration=window_duration, window_start_time=window_start_time)
        self._df = spikes_df
        # self.window_changed_signal.connect(self.on_window_changed)
        
        
    # def on_window_changed(self):
    #     print(f'SpikesDataframeWindow.on_window_changed(): window_changed_signal emitted. self.active_time_window: {self.active_time_window}')
        
      
      

class SpikesWindowOwningMixin:
    """ Implementors own a SpikesWindow and can use it to get the current windowed dataframe
    
    Requires:
        self._spikes_window
    
    """    
    @property
    def spikes_window(self):
        """The spikes_window property."""
        return self._spikes_window

    @property
    def active_windowed_df(self):
        """ """
        return self.spikes_window.active_windowed_df
    
    # from SpikesDataframeOwningMixin
    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self.spikes_window.df
  
  
    # Require TimeWindow
    @property
    def render_window_duration(self):
        """The render_window_duration property."""
        return float(self.spikes_window.window_duration)
    @render_window_duration.setter
    def render_window_duration(self, value):
        self.spikes_window.window_duration = value
    
    @property
    def half_render_window_duration(self):
        """ """
        return np.ceil(float(self.spikes_window.window_duration)/2.0) # 10 by default 
    
    