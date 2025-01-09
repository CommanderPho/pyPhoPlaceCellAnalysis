from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
import numpy as np

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.General.Model.TimeWindow import TimeWindow
from pyphoplacecellanalysis.General.Model.LiveWindowedData import LiveWindowedData
from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource, SpikesDataframeDatasource


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
@metadata_attributes(short_name=None, tags=['time'], input_requires=[], output_provides=[], uses=[], used_by=['SpikeRasterBase'], creation_date='2023-01-01 00:00', related_items=[])
class SpikesDataframeWindow(LiveWindowedData):
    """ a zoomable (variable sized) window into a dataframe with a time axis
    Used by Spike3DRaster
    
    active_window_start_time can be adjusted to set the location of the current window.

    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(curr_spikes_df, window_duration=render_window_duration)
        curr_spikes_df_window
        
    Known Uses:
        SpikeRasterBase

    Inherited Signals:
    
    windowed_data_window_duration_changed_signal = QtCore.pyqtSignal(float, float, float, object) # (start_time, end_time, window_duration, data_value)
    windowed_data_window_updated_signal = QtCore.pyqtSignal(float, float, object) # (start_time, end_time, data_value)
    
    """
    spike_dataframe_changed_signal = QtCore.pyqtSignal() # signal emitted when the spike dataframe is changed, which might change the number of units, number of spikes, and other properties.
    
    ## TimeWindow Convenince properties:
    @property
    def active_time_window(self):
        """ the active time window (2 element start, end tuple)"""
        return self.timeWindow.active_time_window
    
    @property
    def window_duration(self):
        """The window_duration property."""
        return self.timeWindow.window_duration
    
    @property
    def active_window_end_time(self):
        """The active_window_end_time property."""
        return (self.active_window_start_time + self.window_duration)
    
    @property
    def active_window_start_time(self):
        """The current start time of the sliding time window"""
        return self.timeWindow.active_window_start_time
    
    
    # Require TimeWindow and Datasource:
    @property
    def active_windowed_df(self):
        """The dataframe sliced to the current time window (active_time_window)"""
        # return self.df[self.df[self.df.spikes.time_variable_name].between(self.active_time_window[0], self.active_time_window[1])]
        return self.dataSource.get_updated_data_window(self.active_time_window[0], self.active_time_window[1])

    @property
    def active_window_num_spikes(self):
        """The number of spikes (across all units) in the active window."""
        return self.active_windowed_df.shape[0]
    
    # Properties belonging to DataframeDatasource:
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest spiketimes in the total df """
        return self.dataSource.total_df_start_end_times
            
    @property
    def total_data_start_time(self):
        """returns the earliest_df_time: The earliest spiketimes in the total df """
        return self.total_df_start_end_times[0]
    @property
    def total_data_end_time(self):
        """returns the latest_df_time: The latest spiketimes in the total df """
        return self.total_df_start_end_times[1]
            
            
    ##### Get/Set Properties ####:
    @property
    def df(self):
        """The df property."""
        # return self._df
        return self.dataSource.df
    @df.setter
    def df(self, value):
        self.dataSource.df = value
        # self.spike_dataframe_changed_signal.emit()
        
    
    # Initializer:
    def __init__(self, spikes_df, window_duration=15.0, window_start_time=0.0):
        # TimeWindow.__init__(self, window_duration=window_duration, window_start_time=window_start_time)
        # self._df = spikes_df
        
        # TODO: Time window needs to be passed in or kept a reference to:
        curr_time_window = TimeWindow(window_duration=window_duration, window_start_time=window_start_time)
        spikes_dataSource = SpikesDataframeDatasource(spikes_df)
        LiveWindowedData.__init__(self, curr_time_window, spikes_dataSource) # Call base class

        # self.spikes_dataSource.source_data_changed_signal.connect(self.spike_dataframe_changed_signal)
        self.dataSource.source_data_changed_signal.connect(self.on_general_datasource_changed)
        # self.window_changed_signal.connect(self.on_window_changed)
        
    @pyqtExceptionPrintingSlot(object)
    def on_general_datasource_changed(self, datasource):
        """ emit our own custom signal when the general datasource update method returns """
        self.spike_dataframe_changed_signal.emit()
    
    def debug_print_spikes_window(self, prefix_string='spikes_window.', indent_string = '\t'):
        print(f'{indent_string}{prefix_string}total_df_start_end_times: {self.total_df_start_end_times}')
        print(f'{indent_string}{prefix_string}active_time_window: {self.active_time_window}')
        print(f'{indent_string}{prefix_string}window_duration: {self.window_duration}')
        

# ==================================================================================================================== #
# SpikesWindowOwningMixin                                                                                              #
# ==================================================================================================================== #
class SpikesWindowOwningMixin:
    """ Implementors own a SpikesWindow and can use it to get the current windowed dataframe
    
    Requires:
        self._spikes_window
        
    """
    @property
    def spikes_window(self):
        """The spikes_window property."""
        raise NotImplementedError
        # return self._spikes_window

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
    
    def debug_print_spikes_window(self, prefix_string='spikes_window.', indent_string = '\t'):
        self.spikes_window.debug_print_spikes_window(prefix_string=prefix_string, indent_string=indent_string)