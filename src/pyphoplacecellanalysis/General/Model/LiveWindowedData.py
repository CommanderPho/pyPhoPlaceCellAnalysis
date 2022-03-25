# LiveWindowedData.py
from pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd

from pyphoplacecellanalysis.General.Model.Datasources import DataframeDatasource
from pyphoplacecellanalysis.General.Model.TimeWindow import TimeWindow


class LiveWindowedData(QtCore.QObject):
    """ an optional adapter between a DataSource and the GUI/graphic that uses it.
    Serves as an intermediate to TimeWindow and Datasource.
    It subscribes to TimeWindow updates, and for each update it fetches the appropriate data from its internally owned DataSource and emits a singal containing this data that can be used to update the GUI/graphic classes that subscribe to it.
 
    
    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(window_duration=render_window_duration)
        curr_spikes_df_window

    """
    window_duration_changed_signal = QtCore.pyqtSignal(float, float, float, object) # (start_time, end_time, window_duration) more conservitive singal that only changes when the duration of the window changes.
    window_updated_signal = QtCore.pyqtSignal(float, float, object) # (start_time, end_time)
    

    def __init__(self, time_window: TimeWindow, dataSource: DataframeDatasource):
        QtCore.QObject.__init__(self)
        # DO store the datasource on the other hand:
        self.dataSource = dataSource
        
        # Do NOT store an internal reference to time_window. Just connect the signals to receive updates and store the connections:
        self._time_window_duration_connection = time_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        self._time_window_changed_connection = time_window.window_changed_signal.connect(self.on_window_changed)
        
    
    @QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        print(f'LiveWindowedData.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.window_duration_changed_signal.emit(start_t, end_t, duration, data_value)
        


    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'LiveWindowedData.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.window_updated_signal.emit(start_t, end_t, data_value)