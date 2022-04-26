import numpy as np
import pandas as pd

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource


class CurveDatasource(DataframeDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
    
    Contains a dataframe.
    
    Note that "data_series_specs" refers to an object of type RenderDataseries
    
    
    Signals:
    	source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
    """
    
    data_series_specs_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the data_series_specs have changed.

    @property
    def data_series_specs(self):
        """The data_series_specs property."""
        return self._data_series_specs
    @data_series_specs.setter
    def data_series_specs(self, value):
        self._data_series_specs = value
        self.data_series_specs_changed_signal.emit(self._data_series_specs)
        
    @property
    def has_data_series_specs(self):
        """The data_series_specs property."""
        return (self.data_series_specs is not None)
      
    @property
    def datasource_UIDs(self):
        """The datasource_UID property."""
        if self.data_series_specs is not None:
            return [f'{self.custom_datasource_name}.{series_name}' for series_name in self.data_series_specs.data_series_names]
        else:
            return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]


    ## Active-Only versions of data_column_names, data_column_values, and datasource_UIDs that can be overriden to enable only a subset of the values
    @property
    def active_data_column_names(self):
        """ the names of only the non-time columns """
        if self.data_series_specs is not None:
            return self.data_series_specs.data_series_names
        else:
            return self.data_column_values
    
    @property
    def active_data_column_values(self):
        """ The values of only the non-time columns """
        # map(upper, mylis) # WTF is this?
        return self.data_column_values
    
    @property
    def active_datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.active_data_column_values]
    
    
    
    
    def __init__(self, df, datasource_name='default_plot_datasource', data_series_specs=None):
        # Initialize the datasource as a QObject
        DataframeDatasource.__init__(self, df, datasource_name=datasource_name)
        self._data_series_specs = data_series_specs
    
