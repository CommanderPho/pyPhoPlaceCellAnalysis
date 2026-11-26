from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
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
    def _sanitize_duration(self, value: float) -> Optional[float]:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        min_dur = float(getattr(self, 'min_duration', 0.005) or 0.005)
        max_dur = getattr(self, 'max_duration', None)
        if (max_dur is not None) and np.isfinite(max_dur) and (float(max_dur) > min_dur):
            val = float(np.clip(val, min_dur, float(max_dur)))
        else:
            val = max(min_dur, val)
        return val


    @property
    def window_duration(self) -> float:
        """The window_duration property."""
        return self._window_duration
    @window_duration.setter
    def window_duration(self, value):
        sanitized_duration = self._sanitize_duration(value)
        if sanitized_duration is None:
            return
        self._window_duration = sanitized_duration
        self.window_duration_changed_signal.emit(self._active_window_start_time, self.active_window_end_time, self.window_duration) # emit window duration changed signal
        self.window_changed_signal.emit(self.active_window_start_time, self.active_window_end_time) # emit window changed signal
        

    @property
    def active_window_start_time(self) -> float:
        """The current start time of the sliding time window"""
        return self._active_window_start_time
    @active_window_start_time.setter
    def active_window_start_time(self, value):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(val):
            return
        self._active_window_start_time = val
        self.window_changed_signal.emit(self._active_window_start_time, self.active_window_end_time) # emit window changed signal
    
    
    
    def __init__(self, window_duration=15.0, window_start_time=0.0, min_duration: float = 0.005, max_duration: Optional[float] = None):
        QtCore.QObject.__init__(self)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration) if (max_duration is not None and np.isfinite(max_duration)) else None
        self._window_duration = self._sanitize_duration(window_duration) or 15.0
        self._active_window_start_time = float(window_start_time) if np.isfinite(window_start_time) else 0.0
        self.animationThread = None
        # self.window_changed_signal.connect(self.on_window_changed)
            
    @pyqtExceptionPrintingSlot(float)
    def update_window_start(self, new_value):
        self.active_window_start_time = new_value


    @pyqtExceptionPrintingSlot(float, float)
    def update_window_start_end(self, new_start, new_end):
        try:
            new_start_f = float(new_start)
            new_end_f = float(new_end)
        except (TypeError, ValueError):
            return
        if (not np.isfinite(new_start_f)) or (not np.isfinite(new_end_f)):
            return
        if new_end_f < new_start_f:
            new_start_f, new_end_f = new_end_f, new_start_f
        prev_duration = self.window_duration
        proposed_new_duration = self._sanitize_duration(new_end_f - new_start_f)
        if proposed_new_duration is None:
            return
        will_duration_change = not np.isclose(prev_duration, proposed_new_duration)
        # Set the private variables so the signal isn't emitted on set (we'll emit them at the end)
        self._active_window_start_time = new_start_f
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
    
    
    
    