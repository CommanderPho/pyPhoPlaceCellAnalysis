from pyqtgraph.Qt import QtCore

import numpy as np
import pandas as pd


from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial




class DataframeDatasource(QtCore.QObject):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
    Externally should 
    
    Contains a dataframe.
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):
        
        
        
    """
    source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
    
    @property
    def time_column_name(self):
        """ the name of the relevant time column. Defaults to 't' """
        return 't' 
    
    @property
    def time_column_values(self):
        """ the values of only the relevant time columns """
        return self.df[self.time_column_name] # get only the relevant time column
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self.df.columns, np.array([self.time_column_name])) # get only the non-time columns
    
    @property
    def data_column_values(self):
        """ The values of only the non-time columns """
        return self.df[self.data_column_names]

    @property
    def datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]
    

    def __init__(self, df, datasource_name='default_plot_datasource'):
        # Initialize the datasource as a QObject
        QtCore.QObject.__init__(self)
        self.df = df
        self.custom_datasource_name = datasource_name
        assert self.time_column_name in df.columns, "dataframe must have a time column with name 't'"
        
        
    @classmethod
    def init_from_times_values(cls, times, values):
        plot_df = pd.DataFrame({'t': times, 'v': values})
        return cls(plot_df)
        
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        return self.df[self.df[self.time_column_name].between(new_start, new_end)]
    








class SpikesDatasource(DataframeDatasource):
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
        'unit_id' # for colors
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
    """ 
    