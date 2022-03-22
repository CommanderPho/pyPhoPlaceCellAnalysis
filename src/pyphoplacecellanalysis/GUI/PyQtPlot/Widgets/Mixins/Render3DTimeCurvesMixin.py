import pyphoplacecellanalysis
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
import pandas as pd

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.geometry_helpers import find_ranges_in_window

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial



# class RenderEpochs(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
#     def __init__(self, name) -> None:
#         # super(RenderEpochs, self).__init__(**kwargs)
#         self.name = name
#         # self.__dict__ = (self.__dict__ | kwargs)
        
#     # def __init__(self, name, **kwargs) -> None:
#     #     # super(VisualizationParameters, self).__init__(**kwargs)
#     #     self.name = name
#     #     # self.__dict__ = (self.__dict__ | kwargs)
    
    
    
class CurveDatasource(QtCore.QObject):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
    Externally should 
        
    """
    source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
    # window_duration_changed_signal = QtCore.pyqtSignal(float) # more conservitive singal that only changes when the duration of the window changes
    # window_changed_signal = QtCore.pyqtSignal(float, float) # new_start, new_end
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self.df.columns, np.array(['t'])) # get only the non-time columns
    
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
        assert 't' in df.columns, "dataframe must have a time column with name 't'"
        self.df = df
        self.custom_datasource_name = datasource_name
        
        # Sets the dict with the value:
        # self.__dict__ = (self.__dict__ | kwargs)
        
        
    # @classmethod
    # def init_from_dataframe(cls, df):
    #     assert 't' in df.columns, "dataframe must have a time column with name 't'"

    #     self._df = df
        
        
    @classmethod
    def init_from_times_values(cls, times, values):
        plot_df = pd.DataFrame({'t': times, 'v': values})
        return cls(plot_df)
        
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        return self.df[self.df['t'].between(new_start, new_end)]
    

    

class TimeCurvesViewMixin:
    
    @property
    def data_z_scaling_factor(self):
        """ the factor required to scale the data_values_range to fit within the z_max_value """
        return self.calculate_data_z_scaling_factor()
    
    def calculate_data_z_scaling_factor(self):
        """ Calculate the factor required to scale the data_values_range to fit within the z_max_value """
        data_values_range = np.ptp(self.params.time_curves_datasource.data_column_values)
        z_max_value = np.abs(self.z_floor) # get the z-height of the floor so as not to go below it.
        data_z_scaling_factor = z_max_value / data_values_range
        return data_z_scaling_factor


    def init_TimeCurvesViewMixin(self):
        self.params.time_curves_datasource = None # initialize datasource variable
        self.plots.time_curves = dict()
        

    def add_3D_time_curves(self, plot_dataframe):
        self.params.time_curves_datasource = CurveDatasource(plot_dataframe)
        self.plots.time_curves = dict()
        self.update_3D_time_curves()


    # def _build_3D_time_curves(self):
    def update_3D_time_curves(self):
        """ initialize the graphics objects if needed, or update them if they already exist. """
        if self.params.time_curves_datasource is None:
            return
        else:
            curr_plot_column_name = self.params.time_curves_datasource.data_column_names[0]
            curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[0]
            
            # Get current plot items:
            curr_plot3D_active_window_data = self.params.time_curves_datasource.get_updated_data_window(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time) # get updated data for the active window from the datasource
            curve_y_value = -self.n_half_cells
            curr_x = DataSeriesToSpatial.temporal_to_spatial_map(curr_plot3D_active_window_data['t'].to_numpy(), self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
            pts = np.column_stack([curr_x, np.full_like(curr_x, curve_y_value), curr_plot3D_active_window_data[curr_plot_column_name].to_numpy()])
            
            if curr_plot_name in self.plots.time_curves:
                # Plot already exists, update it instead.
                plt = self.plots.time_curves[curr_plot_name]
                plt.setData(pos=pts)
            else:
                # plot doesn't exist, built it fresh.
                plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor('white'), width=0.5, antialias=True)
                plt.scale(1.0, 1.0, self.data_z_scaling_factor) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.
                self.ui.main_gl_widget.addItem(plt)
                self.plots.time_curves[curr_plot_name] = plt # add it to the dictionary.
                
    
    # @QtCore.pyqtSlot(float, float)
    # def TimeCurvesViewMixin_on_window_update(self, new_start, new_end):
    #     """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
    #     self.update_3D_time_curves()
    
    
    @QtCore.pyqtSlot(float, float)
    def TimeCurvesViewMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        self.update_3D_time_curves()
        
        
    

    
