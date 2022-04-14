### Complex Dataseries-based CurveDatasource approach:
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyphoplacecellanalysis.General.Model.RenderDataseries import RenderDataseries
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render3DTimeCurvesMixin import CurveDatasource


class Specific3DTimeCurvesHelper:
    """ Static helper methods that build commonly known 3D time curve datasources and add them to the provided plot.
    
    Provided Curves:
        MUA
        position
    
    """
    ##########################################
    ## MUA 3D Time Curves
    @staticmethod
    def build_MUA_3D_time_curves_datasource(mua_plot_df, pre_spatial_to_spatial_mappings):
        # a value scalar for the z-axis
        z_scaler = MinMaxScaler()
        # Build the scalar:
        mua_plot_df[['mua_firing_rate']] = z_scaler.fit_transform(mua_plot_df[['mua_firing_rate']]) # scale mua_firing_rate separately
        mua_plot_df[['mua_spike_counts']] = z_scaler.fit_transform(mua_plot_df[['mua_spike_counts']]) # scale mua_spike_counts separately
        mua_data_series_pre_spatial_list = [{'name':'mua_firing_rate','t':'t','v_alt':None,'v_main':'mua_firing_rate','color_name':'white', 'line_width':2.0, 'z_scaling_factor':1.0},
                                        {'name':'mua_spike_counts','t':'t','v_alt':None,'v_main':'mua_spike_counts','color_name':'grey', 'line_width':0.5, 'z_scaling_factor':1.0}]
        return CurveDatasource(mua_plot_df, data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(mua_data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))


    @classmethod
    def build_MUA_3D_time_curves(cls, curr_sess, spike_raster_plt_3d):
        """ builds the MUA (Multi-Unit Activity) 3D Curves and adds them to the spike_raster_plot
        Usage:
            active_mua_plot_curve_datasource = Specific3DTimeCurvesHelper.build_MUA_3D_time_curves(curr_sess, spike_raster_plt_3d)
        """
        mua_plot_df = pd.DataFrame({'t': curr_sess.mua.time, 'mua_firing_rate': curr_sess.mua.firing_rate, 'mua_spike_counts': curr_sess.mua.spike_counts}).copy()
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: spike_raster_plt_3d.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells) # This is what places all values along the back wall
        z_map_fn = lambda v_main: v_main # returns the un-transformed primary value
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
                                {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}]
        
        active_mua_plot_curve_datasource = cls.build_MUA_3D_time_curves_datasource(mua_plot_df, data_series_pre_spatial_to_spatial_mappings)
        spike_raster_plt_3d.add_3D_time_curves(curve_datasource=active_mua_plot_curve_datasource) # Add the curves from the datasource
        return active_mua_plot_curve_datasource


    
 
    ##########################################
    ## MUA 3D Time Curves
    @staticmethod
    def build_position_3D_time_curves_datasource(position_df, pre_spatial_to_spatial_mappings):
        # a value scalar for the z-axis
        z_scaler = MinMaxScaler()

        # data_series_pre_spatial_list = [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos'},
        #     {'name':'x position','t':'t','v_alt':None,'v_main':'x'},
        #     {'name':'y position','t':'t','v_alt':None,'v_main':'y'}
        # ]

        # additional properties:
        data_series_pre_spatial_list = [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos','color_name':'yellow', 'line_width':1.25, 'z_scaling_factor':1.0},
            {'name':'x position','t':'t','v_alt':None,'v_main':'x', 'color_name':'red', 'line_width':0.5, 'z_scaling_factor':1.0},
            {'name':'y position','t':'t','v_alt':None,'v_main':'y', 'color_name':'green', 'line_width':0.5, 'z_scaling_factor':1.0}
        ]

        active_indirect_dataseries = RenderDataseries.init_from_pre_spatial_data_series_list(data_series_pre_spatial_list, pre_spatial_to_spatial_mappings)
        active_plot_curve_dataframe = position_df[['t','x','y','lin_pos']].copy()
        # Build the scalar:
        # dfTest[['A', 'B']] = z_scaler.fit_transform(dfTest[['A', 'B']])
        active_plot_curve_dataframe[['x','y']] = z_scaler.fit_transform(active_plot_curve_dataframe[['x','y']]) # scale x and y positions
        active_plot_curve_dataframe[['lin_pos']] = z_scaler.fit_transform(active_plot_curve_dataframe[['lin_pos']]) # scale lin_pos position separately
        active_plot_curve_datasource = CurveDatasource(active_plot_curve_dataframe, data_series_specs=active_indirect_dataseries)
        return active_plot_curve_datasource
        
    @classmethod
    def build_position_3D_time_curves(cls, curr_sess, spike_raster_plt_3d):
        """ builds the animal position 3D Curves and adds them to the spike_raster_plot
        Usage:
            active_plot_curve_datasource = Specific3DTimeCurvesHelper.build_position_3D_time_curves(curr_sess, spike_raster_plt_3d)
        """
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: spike_raster_plt_3d.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells)
        z_map_fn = lambda v_main: v_main
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
        ]
        
        active_plot_curve_datasource = cls.build_position_3D_time_curves_datasource(curr_sess.position.to_dataframe(), data_series_pre_spatial_to_spatial_mappings)
        spike_raster_plt_3d.add_3D_time_curves(curve_datasource=active_plot_curve_datasource) # Add the curves from the datasource
        return active_plot_curve_datasource
