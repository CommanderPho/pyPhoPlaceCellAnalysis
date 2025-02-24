# LiveWindowedData.py
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd

from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource
from pyphoplacecellanalysis.General.Model.TimeWindow import TimeWindow

@metadata_attributes(short_name=None, tags=['time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-18 12:45', related_items=[])
class LiveWindowedData(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ an optional adapter between a DataSource and the GUI/graphic that uses it.
    Serves as an intermediate to TimeWindow and Datasource.
    It subscribes to TimeWindow updates, and for each update it fetches the appropriate data from its internally owned DataSource and emits a singal containing this data that can be used to update the GUI/graphic classes that subscribe to it.
 
    
    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(window_duration=render_window_duration)
        curr_spikes_df_window
        
    Known Usages:
        SpikesDataframeWindow

    """
    
    # # Basic signals only:
    # window_duration_changed_signal = QtCore.pyqtSignal(float, float, float) # (start_time, end_time, window_duration) more conservitive singal that only changes when the duration of the window changes.
    # window_changed_signal = QtCore.pyqtSignal(float, float) # (start_time, end_time)

    # Signals with data:
    windowed_data_window_duration_changed_signal = QtCore.pyqtSignal(float, float, float, object) # (start_time, end_time, window_duration, data_value) more conservitive singal that only changes when the duration of the window changes.
    windowed_data_window_updated_signal = QtCore.pyqtSignal(float, float, object) # (start_time, end_time, data_value)
    
    
    # Simple TimeWindow passthrough properties
    @property
    def window_duration(self) -> float:
        """The render_window_duration property."""
        return float(self.timeWindow.window_duration)
    @window_duration.setter
    def window_duration(self, value):
        self.timeWindow.window_duration = value
    @property
    def half_window_duration(self) -> float:
        """ """
        return np.ceil(float(self.timeWindow.window_duration)/2.0) # 10 by default 
    
    def __init__(self, time_window: TimeWindow, dataSource: DataframeDatasource):
        QtCore.QObject.__init__(self)
        # DO store the datasource on the other hand:
        self.dataSource = dataSource
        
        ## TODO: alternative mode, store the time window (hopefully a reference)
        self.timeWindow = time_window
        
        # TODO: Do NOT store an internal reference to time_window. Just connect the signals to receive updates and store the connections:
        self._time_window_duration_connection = time_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        self._time_window_changed_connection = time_window.window_changed_signal.connect(self.on_window_changed)
        
    
    @pyqtExceptionPrintingSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        # print(f'LiveWindowedData.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_duration_changed_signal.emit(start_t, end_t, duration, data_value)
        
    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        # if self.enable_debug_print:
        #     print(f'LiveWindowedData.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_updated_signal.emit(start_t, end_t, data_value)
        
    ## Called to update its internal TimeWindow
    @pyqtExceptionPrintingSlot(float)
    def update_window_start(self, new_value):
        self.timeWindow.update_window_start(new_value)

    @pyqtExceptionPrintingSlot(float, float)
    def update_window_start_end(self, new_start, new_end):
        self.timeWindow.update_window_start_end(new_start, new_end)
        
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

        
        