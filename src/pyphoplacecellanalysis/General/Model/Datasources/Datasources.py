from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore

import numpy as np
import pandas as pd


from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial




class BaseDatasource(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
        
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):

    """
    source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
        
    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        raise NotImplementedError
        # return (earliest_df_time, latest_df_time)
    
    
    ##### Get/Set Properties ####:
        

    def __init__(self, datasource_name='default_base_datasource'):
        # Initialize the datasource as a QObject
        QtCore.QObject.__init__(self)
        # Custom Setup
        self.custom_datasource_name = datasource_name        
        
        
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        raise NotImplementedError
    
    
    
    
    


class DataframeDatasource(BaseDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
   
    Contains a dataframe.
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):

    """
    
    @property
    def time_column_name(self):
        """ the name of the relevant time column. Defaults to 't' """
        return 't' 
    
    @property
    def time_column_values(self):
        """ the values of only the relevant time columns """
        return self._df[self.time_column_name] # get only the relevant time column
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self._df.columns, np.array([self.time_column_name])) # get only the non-time columns
    
    @property
    def data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.data_column_names]
    
    @property
    def datasource_UIDs(self):
        """The datasource_UID property.
        
        Note: Assumes multiple series are given by the non-time columns:
        
        """
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]
    
    
    ## Active-Only versions of data_column_names, data_column_values, and datasource_UIDs that can be overriden to enable only a subset of the values
    @property
    def active_data_column_names(self):
        """ the names of only the non-time columns """
        return self.data_column_values
        # return self.data_column_names # TODO: why does this return the self.data_column_values instead of self.data_column_names??
    
    @property
    def active_data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.active_data_column_names]
    
    @property
    def active_datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.active_data_column_values]
    
    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        return self.total_df_start_end_times
        
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        earliest_df_time = np.nanmin(self.df[self.time_column_name])
        latest_df_time = np.nanmax(self.df[self.time_column_name])
        df_timestamps = self.df[self.time_column_name].to_numpy()
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
        self.source_data_changed_signal.emit()
        

    def __init__(self, df, datasource_name='default_plot_datasource'):
        # Initialize the datasource as a BaseDatasource
        BaseDatasource.__init__(self, datasource_name=datasource_name)
        self._df = df
        assert self.time_column_name in df.columns, "dataframe must have a time column with name 't'"
        
        
    @classmethod
    def init_from_times_values(cls, times, values):
        plot_df = pd.DataFrame({'t': times, 'v': values})
        return cls(plot_df)
        
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        return self.df[self.df[self.time_column_name].between(new_start, new_end)]
    










class SpikesDataframeDatasource(DataframeDatasource):
    """ Provides neural spiking data for one or more neuron (unit) and the timestamps at which they occur 't'.
    
    Signals:
    	source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):
    """

    @property
    def time_column_name(self):
        """ the name of the relevant time column. Gets the values from the spike dataframe """
        return self.df.spikes.time_variable_name
    
    
    def __init__(self, df, datasource_name='default_spikes_datasource'):
        # Initialize the datasource as a QObject
        DataframeDatasource.__init__(self, df, datasource_name=datasource_name)

    """
        ## It seems the relevant columns are:
        'visualization_raster_y_location' # for y-offsets in 3D plot
        'fragile_linear_neuron_IDX' # for colors
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
    """ 
    