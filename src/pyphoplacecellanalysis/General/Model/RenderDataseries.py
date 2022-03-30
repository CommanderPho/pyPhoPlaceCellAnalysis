from pyqtgraph.Qt import QtCore

import numpy as np
import pandas as pd


from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial


class RenderDataseries(QtCore.QObject):
    """ 
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

    """
    def __init__(self, pre_spatial_data_series_list, pre_spatial_to_spatial_mappings):
        # Initialize the datasource as a QObject
        QtCore.QObject.__init__(self)
        self.data_series_pre_spatial_list = pre_spatial_data_series_list
        self.data_series_pre_spatial_to_spatial_mappings = pre_spatial_to_spatial_mappings

      
    def get_data_series_spatial_values(self, curr_windowed_df):
        data_series_pre_spatial_values_list = RenderDataseries._get_data_series_pre_spatial_values(curr_windowed_df, self.data_series_pre_spatial_list)
        # data_series_pre_spatial_values_list
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
            # print(f"'name':{series_name},'t':{series_t},'v_alt':{series_v_alt},'v_main':{series_v_main}")
            data_series_values_list.append({'name':series_name,'t':curr_series_t_values,'v_alt':curr_series_v_alt_values,'v_main':curr_series_v_main_values})

        return data_series_values_list


    @classmethod
    def _evaluate_spatial_values_from_pre_spatial_values(cls, data_series_pre_spatial_values_list, data_series_pre_spatial_to_spatial_mappings):
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
            data_series_spaital_values_list.append({'name':series_name,'x':curr_series_x_values,'y':curr_series_y_values,'z':curr_series_z_values})
        return data_series_spaital_values_list
