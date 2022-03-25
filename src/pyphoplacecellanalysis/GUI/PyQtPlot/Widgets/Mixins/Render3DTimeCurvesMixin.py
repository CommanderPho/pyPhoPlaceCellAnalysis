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
from pyphoplacecellanalysis.General.Model.Datasources import DataframeDatasource


class CurveDatasource(DataframeDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
    Externally should 
    
    Contains a dataframe.
    
    Signals:
    	source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
    """

    def __init__(self, df, datasource_name='default_plot_datasource'):
        # Initialize the datasource as a QObject
        DataframeDatasource.__init__(self, df, datasource_name=datasource_name)
    


class TimeCurvesViewMixin:
    """ Renders 3D line plots that are dependent on time.
    
    Usage:
        # plot_data = pd.DataFrame({'t': curr_sess.mua.time, 'mua_firing_rate': curr_sess.mua.firing_rate, 'mua_spike_counts': curr_sess.mua.spike_counts})
        plot_data = pd.DataFrame({'t': curr_sess.mua.time, 'mua_firing_rate': curr_sess.mua.firing_rate})
        spike_raster_plt_3d.add_3D_time_curves(plot_data)
    """
    
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

    @QtCore.pyqtSlot()
    def TimeCurvesViewMixin_on_init(self):
        self.params.time_curves_datasource = None # initialize datasource variable
        self.params.time_curves_no_update = False # called to disabling updating time curves internally
        
        self.plots.time_curves = dict()
        

    def add_3D_time_curves(self, plot_dataframe):
        self.params.time_curves_datasource = CurveDatasource(plot_dataframe)
        self.plots.time_curves = dict()
        self.update_3D_time_curves()


    def detach_3d_time_curves_datasource(self):
        """ Called to remove the current time_curves_datasource. Safely removes the plot objects before doing so. """
        self.params.time_curves_no_update = True # freeze updating
        self.clear_all_3D_time_curves()
        self.params.time_curves_datasource = None # clear the datasource, this will prevent plots from being re-added during the self.TimeCurvesViewMixin_on_window_update() call.     
        self.params.time_curves_no_update = False # safe to re-enable updating.
        
        
    def clear_all_3D_time_curves(self):
        for (UID, plt) in self.plots.time_curves.items():
            self.ui.main_gl_widget.removeItem(plt)
            # plt.delete_later() #?
        # Clear the dict
        self.plots.time_curves.clear()

    
    def remove_3D_time_curves(self, UID=None, original_dataframe=None):
        """ TODO: unfortunately with this setup they would be recreated again in self.update_3D_time_curves() because the datasource would still be attached but the plot wouldn't exist. """
        raise NotImplementedError
    
        if UID is not None:
            plot_UIDs = self.params.time_curves_datasource.datasource_UIDs # ['default_plot_datasource.mua_firing_rate']
            plt = self.plots.time_curves.get(UID, None)
        elif original_dataframe is not None:
            self.par
            raise NotImplementedError
        else:
            plt = None

        if plt is not None:
            # perform remove:
            print(f'removing 3D time curve with UID: {UID}...')
            self.ui.main_gl_widget.removeItem(plt)
            # plt.delete_later() #?
            del self.plots.time_curves[UID] # delete the dictionary entry, meaning it's valid to do a
            print('\t done.')
            
        

    # def _build_3D_time_curves(self):
    def update_3D_time_curves(self):
        """ initialize the graphics objects if needed, or update them if they already exist. """
        if self.params.time_curves_datasource is None:
            return
        elif self.params.time_curves_no_update:
            # don't update because we're in no_update mode
            return
        else:
            # TODO: currently only gets the first data_column. (doesn't yet support multiple)
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
        
        
    

    
