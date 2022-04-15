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
from pyphoplacecellanalysis.General.Model.Datasources.CurveDatasource import CurveDatasource


# An OrderedList of dictionaries of values to be provided by the datasource:
# {'name':'linear position','x':'t','y':None,'z':'lin_pos'}

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
        if self.params.time_curves_z_normalization_mode == 'None':
            return 1.0
        elif self.params.time_curves_z_normalization_mode == 'global':
            data_values_range = np.ptp(self.params.time_curves_datasource.data_column_values)
            z_max_value = np.abs(self.z_floor) # get the z-height of the floor so as not to go below it.
            data_z_scaling_factor = z_max_value / data_values_range
        else:
            raise NotImplementedError
        
        return data_z_scaling_factor
    
    

    @QtCore.pyqtSlot()
    def TimeCurvesViewMixin_on_init(self):
        self.params.time_curves_datasource = None # initialize datasource variable
        self.params.time_curves_no_update = False # called to disabling updating time curves internally
        self.params.time_curves_z_normalization_mode = 'None'
            
        self.plots.time_curves = dict()
        

    def add_3D_time_curves(self, curve_datasource: CurveDatasource=None, plot_dataframe:pd.DataFrame=None):
        assert (curve_datasource is not None) or (plot_dataframe is not None), "At least one of curve_datasource or plot_dataframe must be non-None."
    
        if self.params.time_curves_datasource is not None:
            # TODO: detach any extant datasource:
            self.detach_3d_time_curves_datasource()
        
        if curve_datasource is not None:
            self.params.time_curves_datasource = curve_datasource
        else:
            if plot_dataframe is not None: 
                # build a new CurveDatasource from the provided dataframe 
                self.params.time_curves_datasource = CurveDatasource(plot_dataframe)
                
                
        ## Connect the specs update singnal
            
        ## Connect the data_series_specs_changed_signal:
        self.params.time_curves_datasource.data_series_specs_changed_signal.connect(self.TimeCurvesViewMixin_on_data_series_specs_changed)
            
        
        # TODO: should this really be overwritten here? Seems like previous plots should be removed first right?
        #       ## Specifically, it seems like self.clear_all_3D_time_curves() should be called if we are just going to overwrite it
        self.plots.time_curves = dict()
        self.update_3D_time_curves()


    def detach_3d_time_curves_datasource(self):
        """ Called to remove the current time_curves_datasource. Safely removes the plot objects before doing so. """
        self.params.time_curves_no_update = True # freeze updating
        
        # Disconnect signals?
        connected_signal_recievers = self.params.time_curves_datasource.receivers(self.params.time_curves_datasource.data_series_specs_changed_signal)
        print(f'connected_signal_recievers: {connected_signal_recievers}')
        if connected_signal_recievers > 0:
            print(f'disconnecting {connected_signal_recievers} receivers...')
            self.params.time_curves_datasource.data_series_specs_changed_signal.disconnect()
            print('\tdone')
        
        self.clear_all_3D_time_curves()
        self.params.time_curves_datasource = None # clear the datasource, this will prevent plots from being re-added during the self.TimeCurvesViewMixin_on_window_update() call.     
        self.params.time_curves_no_update = False # safe to re-enable updating.
        
        
    def clear_all_3D_time_curves(self):
        for (aUID, plt) in self.plots.time_curves.items():
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
            
        
    def _build_or_update_plot(self, plot_name, points, **kwargs):
        # build the plot arguments (color, line thickness, etc)        
        plot_args = ({'color_name':'white','line_width':0.5,'z_scaling_factor':1.0} | kwargs)
        
        if plot_name in self.plots.time_curves:
            # Plot already exists, update it instead.
            plt = self.plots.time_curves[plot_name]
            plt.setData(pos=points)
        else:
            # plot doesn't exist, built it fresh.
            
            line_color = plot_args.get('color', None)
            if line_color is None:
                # if no explicit color value is provided, build a new color from the 'color_name' key, or if that's missing just use white.
                line_color = pg.mkColor(plot_args.setdefault('color_name', 'white'))
                line_color.setAlphaF(0.8)
                
            # plt = gl.GLLinePlotItem(pos=points, color=pg.mkColor('white'), width=0.5, antialias=True)
            plt = gl.GLLinePlotItem(pos=points, color=line_color, width=plot_args.setdefault('line_width',0.5), antialias=True)
            plt.scale(1.0, 1.0, plot_args.setdefault('z_scaling_factor',1.0)) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.            
            # plt.scale(1.0, 1.0, self.data_z_scaling_factor) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.
            self.ui.main_gl_widget.addItem(plt)
            self.plots.time_curves[plot_name] = plt # add it to the dictionary.
        return plt
            

    # def _build_3D_time_curves(self):
    def update_3D_time_curves(self):
        """ initialize the graphics objects if needed, or update them if they already exist. """
        if self.params.time_curves_datasource is None:
            return
        elif self.params.time_curves_no_update:
            # don't update because we're in no_update mode
            return
        else:
            # Common to both:
            # Get current plot items:
            curr_plot3D_active_window_data = self.params.time_curves_datasource.get_updated_data_window(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time) # get updated data for the active window from the datasource
            is_data_series_mode = self.params.time_curves_datasource.has_data_series_specs
            if is_data_series_mode:
                data_series_spaital_values_list = self.params.time_curves_datasource.data_series_specs.get_data_series_spatial_values(curr_plot3D_active_window_data)
                num_data_series = len(data_series_spaital_values_list)
            else:
                # old compatibility mode:
                num_data_series = 1

            # curr_data_series_index = 0
            # Loop through the active data series:                
            for curr_data_series_index in np.arange(num_data_series):
                # Data series mode:
                if is_data_series_mode:
                    # Get the current series:
                    curr_data_series_dict = data_series_spaital_values_list[curr_data_series_index]
                    
                    curr_plot_column_name = curr_data_series_dict.get('name', f'series[{curr_data_series_index}]') # get either the specified name or the generic 'series[i]' name otherwise
                    curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
                    # points for the current plot:
                    pts = np.column_stack([curr_data_series_dict['x'], curr_data_series_dict['y'], curr_data_series_dict['z']])
                    
                    # Extra options:
                    # color_name = curr_data_series_dict.get('color_name','white')
                    extra_plot_options_dict = {'color_name':curr_data_series_dict.get('color_name', 'white'),
                                               'line_width':curr_data_series_dict.get('line_width', 0.5),
                                               'z_scaling_factor':curr_data_series_dict.get('z_scaling_factor', 0.5)}
                    
                else:
                    # TODO: currently only gets the first data_column. (doesn't yet support multiple)
                    curr_plot_column_name = self.params.time_curves_datasource.data_column_names[curr_data_series_index]
                    curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
                    
                    curve_y_value = -self.n_half_cells
                    
                    # Get y-values:
                    curr_x = self.temporal_to_spatial(curr_plot3D_active_window_data['t'].to_numpy())
                    pts = np.column_stack([curr_x, np.full_like(curr_x, curve_y_value), curr_plot3D_active_window_data[curr_plot_column_name].to_numpy()])
                    
                    extra_plot_options_dict = {}
                
                # outputs of either mode are curr_plot_name, pts
                curr_plt = self._build_or_update_plot(curr_plot_name, pts, **extra_plot_options_dict)
                # end for curr_data_series_index in np.arange(num_data_series)
    
    # @QtCore.pyqtSlot(float, float)
    # def TimeCurvesViewMixin_on_window_update(self, new_start, new_end):
    #     """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
    #     self.update_3D_time_curves()
    
    
    @QtCore.pyqtSlot(object)
    def TimeCurvesViewMixin_on_data_series_specs_changed(self, updated_data_series_specs):
        """ called when the data series specs are udpated. """
        print(f'TimeCurvesViewMixin_on_data_series_specs_changed(...)')
        # self.clear_all_3D_time_curves()
        self.add_3D_time_curves(self.params.time_curves_datasource) # Just re-adding the current datasource is sufficient to update. TODO: inefficient?
        # self.update_3D_time_curves()
        

    @QtCore.pyqtSlot(float, float)
    def TimeCurvesViewMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        self.update_3D_time_curves()
        
        
    

    
