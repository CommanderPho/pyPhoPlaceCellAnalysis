from pyqtgraph.Qt import QtCore

import numpy as np
import pandas as pd


from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial


class RenderDataseries(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ 
    
    Known Usages:
    
        Specific3DTimeCurves
        
        
    Usage:
    
        from pyphoplacecellanalysis.General.Model.RenderDataseries import RenderDataseries

        data_series_pre_spatial_list = [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos'},
            {'name':'x position','t':'t','v_alt':None,'v_main':'x'},
            {'name':'y position','t':'t','v_alt':None,'v_main':'y'}
        ]

        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: spike_raster_plt_3d.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells)
        z_map_fn = lambda v_main: v_main

        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
        ]

        active_dataseries = RenderDataseries(data_series_pre_spatial_list, data_series_pre_spatial_to_spatial_mappings)
        # Creates a separate 3D curve for each specified data-series in data_series_list:
        curr_windowed_df = position_dataSource.get_updated_data_window(0.0, 100.0)
        # curr_windowed_df
        data_series_spaital_values_list = active_dataseries.get_data_series_spatial_values(curr_windowed_df)
        data_series_spaital_values_list
    """
    
    any_expected_keys = np.array(['name','x','t','y','v_alt','z','v_main','x_map_fn','y_map_fn','z_map_fn'])
    pre_spatial_expected_keys = np.array(['name','t','v_alt','v_main'])
    spatial_expected_keys = np.array(['name','x','y','z'])
    
    def __init__(self, direct_spatial_data_series_list=None, pre_spatial_data_series_list=None, pre_spatial_to_spatial_mappings=None):
        # Initialize the datasource as a QObject
        QtCore.QObject.__init__(self)
        self.direct_spatial_data_series_list = direct_spatial_data_series_list
        self.data_series_pre_spatial_list = pre_spatial_data_series_list
        self.data_series_pre_spatial_to_spatial_mappings = pre_spatial_to_spatial_mappings


    @classmethod
    def init_from_pre_spatial_data_series_list(cls, data_series_list, pre_spatial_to_spatial_mappings):
        return cls(pre_spatial_data_series_list=data_series_list, pre_spatial_to_spatial_mappings=pre_spatial_to_spatial_mappings)
    
    @classmethod
    def init_from_direct_spatial_data_series_list(cls, spatial_data_series_list):
        return cls(direct_spatial_data_series_list=spatial_data_series_list)
        
    
    @property
    def data_series_config_list(self):
        """The data_series_config_list property."""
        if self.direct_spatial_data_series_list is not None:
            return self.direct_spatial_data_series_list
        else:
            return self.data_series_pre_spatial_list
    
    
    @property
    def data_series_names(self):
        """The data_series_names property."""
        return [a_series['name'] for a_series in self.data_series_config_list]
     
    @property
    def num_data_series(self):
        """The number of different data series."""
        return len(self.data_series_config_list)
      
      
    def get_data_series_spatial_values(self, curr_windowed_df):
        if self.direct_spatial_data_series_list is not None:
            # Use direct spatial dataseries list (no mapping needed)
            data_series_spaital_values_list = RenderDataseries._get_spatial_data_series_values(curr_windowed_df, self.direct_spatial_data_series_list)
        else:           
            # perfrom the mapping from pre_spatial_values to spatial_values
            data_series_pre_spatial_values_list = RenderDataseries._get_data_series_pre_spatial_values(curr_windowed_df, self.data_series_pre_spatial_list)
            data_series_spaital_values_list = RenderDataseries._evaluate_spatial_values_from_pre_spatial_values(data_series_pre_spatial_values_list, self.data_series_pre_spatial_to_spatial_mappings)
        return data_series_spaital_values_list
    
    
    
    ## Pre-Spatial Transform Series
    
    @classmethod
    def _get_data_series_pre_spatial_values(cls, curr_windowed_df, data_series_list, enable_debug_print=False):
        """ Gets the pre-spatial values from the dataframe with the column names specified in each of the column
            Pre-Spatial Data Series have keys: ['name','t','v_alt','v_main']
            'v_alt' maps to y-axis (depth) and 'v_main' maps to z-axis (vertical offset) by default
        """
        data_series_values_list = []
        for a_series_config_dict in data_series_list:
            series_name = a_series_config_dict.get('name', '')

            series_t_column = a_series_config_dict.get('t', None)
            if series_t_column is not None:
                curr_series_t_values = curr_windowed_df[series_t_column].to_numpy()
            else:
                curr_series_t_values = None
            series_v_alt_column = a_series_config_dict.get('v_alt', None)
            if series_v_alt_column is not None:
                curr_series_v_alt_values = curr_windowed_df[series_v_alt_column].to_numpy()
            else:
                curr_series_v_alt_values = None

            series_v_main_column = a_series_config_dict.get('v_main', None)
            if series_v_main_column is not None:
                curr_series_v_main_values = curr_windowed_df[series_v_main_column].to_numpy()
            else:
                curr_series_v_main_values = None

            if enable_debug_print:
                print(f"a_series_config_dict: {a_series_config_dict}")
                
            a_series_value_dict_all_keys = np.array(list(a_series_config_dict.keys()))
            extra_series_keys = np.setdiff1d(a_series_value_dict_all_keys, cls.pre_spatial_expected_keys) # get only the unexpected/unhandled keys,  # ['color_name', 'line_width', 'z_scaling_factor']
            extra_series_options_dict = {an_extra_key:a_series_config_dict[an_extra_key] for an_extra_key in extra_series_keys} # # {'color_name': 'yellow', 'line_width': 1.25, 'z_scaling_factor': 1.0}
            # print(f"'name':{series_name},'t':{series_t},'v_alt':{series_v_alt},'v_main':{series_v_main}")
            data_series_values_list.append({'name':series_name,'t':curr_series_t_values,'v_alt':curr_series_v_alt_values,'v_main':curr_series_v_main_values} | extra_series_options_dict)

        return data_series_values_list


    @classmethod
    def _evaluate_spatial_values_from_pre_spatial_values(cls, data_series_pre_spatial_values_list, data_series_pre_spatial_to_spatial_mappings, enable_debug_print=False):
        """
        data_series_pre_spatial_values_list: Values computed by _get_data_series_pre_spatial_values(curr_windowed_df, data_series_pre_spatial_list)
            Example:
                data_series_pre_spatial_list = [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos'},
                    {'name':'x position','t':'t','v_alt':None,'v_main':'x'},
                    {'name':'y position','t':'t','v_alt':None,'v_main':'y'}
                ]
                data_series_pre_spatial_values_list = _get_data_series_pre_spatial_values(curr_windowed_df, data_series_pre_spatial_list)


        
        data_series_pre_spatial_to_spatial_mappings: Mappings from the pre-spatial values to the spatial values:
            Example:
                x_map_fn = lambda t: spike_raster_plt_3d.temporal_to_spatial(t)
                y_map_fn = lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells)
                z_map_fn = lambda v_main: v_main

                data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
                    {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
                    {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
                ]
        """
        data_series_spaital_values_list = []
        for i, a_series_value_dict in enumerate(data_series_pre_spatial_values_list):
            curr_series_pre_to_spatial_mapping_dict = data_series_pre_spatial_to_spatial_mappings[i]
            curr_series_x_map_fn = curr_series_pre_to_spatial_mapping_dict.get('x_map_fn', lambda t: t)
            curr_series_y_map_fn = curr_series_pre_to_spatial_mapping_dict.get('y_map_fn', lambda v_alt: v_alt)
            curr_series_z_map_fn = curr_series_pre_to_spatial_mapping_dict.get('z_map_fn', lambda v_main: v_main)

            # get the names for the corresponding columns in data_series_pre_spatial_values_list
            series_name = a_series_value_dict[curr_series_pre_to_spatial_mapping_dict['name']]
            curr_series_x_values = curr_series_x_map_fn(a_series_value_dict[curr_series_pre_to_spatial_mapping_dict['x']])
            # 
            if a_series_value_dict[curr_series_pre_to_spatial_mapping_dict['y']] is None:
                curr_series_y_values = curr_series_y_map_fn(curr_series_x_values) # pass x values to the y_map_fn to repeat the value for all size
            else:
                curr_series_y_values = curr_series_y_map_fn(a_series_value_dict[curr_series_pre_to_spatial_mapping_dict['y']])
            curr_series_z_values = curr_series_z_map_fn(a_series_value_dict[curr_series_pre_to_spatial_mapping_dict['z']])
            ## Additional (Optional) Unhandled values:
            a_series_value_dict_all_keys = np.array(list(a_series_value_dict.keys()))
            extra_series_keys = np.setdiff1d(a_series_value_dict_all_keys, cls.any_expected_keys) # get only the unexpected/unhandled keys,  # ['color_name', 'line_width', 'z_scaling_factor']
            if enable_debug_print:
                print(f'for i={i}, a_series_value_dict.keys(): {a_series_value_dict.keys()}, extra_series_keys: {extra_series_keys}')
            extra_series_options_dict = {an_extra_key:a_series_value_dict[an_extra_key] for an_extra_key in extra_series_keys} # # {'color_name': 'yellow', 'line_width': 1.25, 'z_scaling_factor': 1.0}
            data_series_spaital_values_list.append({'name':series_name,'x':curr_series_x_values,'y':curr_series_y_values,'z':curr_series_z_values} | extra_series_options_dict)
        return data_series_spaital_values_list


    @classmethod
    def _get_spatial_data_series_values(cls, curr_windowed_df, data_series_list, enable_debug_print=False):
        """ Evaluates the directly spatial values for a list data_series_list of dicts with keys ['name','x','y','z']
        
        data_series_spatial_list = [{'name':'linear position','x':'t','y':None,'z':'lin_pos'},
            {'name':'x position','x':'t','y':None,'z':'x'},
            {'name':'y position','x':'t','y':None,'z':'y'}
        ]
        
        data_series_values_list = _get_data_series_values(curr_windowed_df, data_series_spatial_list)
        data_series_values_list

        """
        # for series_name, series_x, series_y, series_z in data_series_list.items():
        data_series_values_list = []
        for a_series_config_dict in data_series_list:
            # series_name, series_x, series_y, series_z = series_config_dict
            series_name = a_series_config_dict.get('name', '')

            series_x_column = a_series_config_dict.get('x', None)
            if series_x_column is not None:
                curr_series_x_values = curr_windowed_df[series_x_column].to_numpy()
            else:
                curr_series_x_values = None
            series_y_column = a_series_config_dict.get('y', None)
            if series_y_column is not None:
                curr_series_y_values = curr_windowed_df[series_y_column].to_numpy()
            else:
                curr_series_y_values = None

            series_z_column = a_series_config_dict.get('z', None)
            if series_z_column is not None:
                curr_series_z_values = curr_windowed_df[series_z_column].to_numpy()
            else:
                curr_series_z_values = None

            if enable_debug_print:
                print(f"a_series_config_dict: {a_series_config_dict}")
                
            a_series_value_dict_all_keys = np.array(list(a_series_config_dict.keys()))
            extra_series_keys = np.setdiff1d(a_series_value_dict_all_keys, cls.any_expected_keys) # get only the unexpected/unhandled keys,  # ['color_name', 'line_width', 'z_scaling_factor']
            extra_series_options_dict = {an_extra_key:a_series_config_dict[an_extra_key] for an_extra_key in extra_series_keys} # # {'color_name': 'yellow', 'line_width': 1.25, 'z_scaling_factor': 1.0}
            
                # print(f"'name':{series_name},'x':{series_x},'y':{series_y},'z':{series_z}")
            data_series_values_list.append({'name':series_name,'x':curr_series_x_values,'y':curr_series_y_values,'z':curr_series_z_values} | extra_series_options_dict)

        return data_series_values_list

