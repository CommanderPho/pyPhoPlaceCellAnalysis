from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot


""" Windowing Class Features:

Transforming the events into either 2D or 3D representations for visualization should NOT be part of this class' function.
Separate 2D and 3D event visualization functions should be made to transform events from this class into appropriate point/datastructure representations for the visualization framework being used.

# Local window properties
[X] Given by .active_time_window
    Get (window_start, window_end) times

	!NO: Not for this class.
	!# # Global data properties
	!# [X] Given by .total_df_start_end_times
	!#     Get (earliest_datapoint_time, latest_datapoint_time) # globally, for the entire timeseries


"""
class TimeWindow(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ a zoomable (variable sized) window into a dataset with a time axis
    Used by Spike3DRaster
    
    active_window_start_time can be adjusted to set the location of the current window.

    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(window_duration=render_window_duration)
        curr_spikes_df_window

    """
    window_duration_changed_signal = QtCore.pyqtSignal(float, float, float) # (start_time, end_time, window_duration) more conservitive singal that only changes when the duration of the window changes.
    window_changed_signal = QtCore.pyqtSignal(float, float) # (start_time, end_time)
    
    @property
    def active_time_window(self) -> Tuple[float, float]:
        """ a 2-element time window [start_time, end_time]"""
        return (self.active_window_start_time, self.active_window_end_time)
        
    @property
    def active_window_end_time(self) -> float:
        """The active_window_end_time property."""
        return (self.active_window_start_time + self.window_duration)
                
    ##### Get/Set Properties ####:
    @property
    def window_duration(self) -> float:
        """The window_duration property."""
        return self._window_duration
    @window_duration.setter
    def window_duration(self, value):
        self._window_duration = value
        self.window_duration_changed_signal.emit(self._active_window_start_time, self.active_window_end_time, self.window_duration) # emit window duration changed signal
        self.window_changed_signal.emit(self.active_window_start_time, self.active_window_end_time) # emit window changed signal
        

    @property
    def active_window_start_time(self) -> float:
        """The current start time of the sliding time window"""
        return self._active_window_start_time
    @active_window_start_time.setter
    def active_window_start_time(self, value):
        self._active_window_start_time = value
        self.window_changed_signal.emit(self._active_window_start_time, self.active_window_end_time) # emit window changed signal
    
    
    
    def __init__(self, window_duration=15.0, window_start_time=0.0):
        QtCore.QObject.__init__(self)
        self._window_duration = window_duration
        self._active_window_start_time = window_start_time
        self.animationThread = None
        # self.window_changed_signal.connect(self.on_window_changed)
            
    @pyqtExceptionPrintingSlot(float)
    def update_window_start(self, new_value):
        self.active_window_start_time = new_value


    @pyqtExceptionPrintingSlot(float, float)
    def update_window_start_end(self, new_start, new_end):
        prev_duration = self.window_duration
        proposed_new_duration = new_end - new_start
        will_duration_change = not np.isclose(prev_duration, proposed_new_duration)
        # Set the private variables so the signal isn't emitted on set (we'll emit them at the end)
        self._active_window_start_time = new_start
        if will_duration_change:
            self._window_duration = proposed_new_duration
            
            self.window_duration_changed_signal.emit(self.active_window_start_time, self.active_window_end_time, self.window_duration)
        self.window_changed_signal.emit(self.active_window_start_time, self.active_window_end_time)
        
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @pyqtExceptionPrintingSlot(object)
    def update_window_start_rate_limited(self, evt):
        self.update_window_start(*evt)
    
    @pyqtExceptionPrintingSlot(object)
    def update_window_start_end_rate_limited(self, evt):
        self.update_window_start_end(*evt)
        

        
    # def on_window_changed(self):
    #     print(f'SpikesDataframeWindow.on_window_changed(): window_changed_signal emitted. self.active_time_window: {self.active_time_window}')
        
      
      
class TimeWindowOwningMixin:
    """ Implementors own a TimeWindow and can use it to get the current windowed dataframe
    
    Requires:
        self._time_window
    
    """    
    @property
    def time_window(self) -> TimeWindow:
        """The spikes_window property."""
        return self._time_window
    
    @property
    def render_window_duration(self) -> float:
        return float(self.time_window.window_duration)
    @render_window_duration.setter
    def render_window_duration(self, value):
        self.time_window.window_duration = value
    
    @property
    def half_render_window_duration(self) -> float:
        """ """
        return np.ceil(float(self.time_window.window_duration)/2.0) # 10 by default 
    
    
    
    