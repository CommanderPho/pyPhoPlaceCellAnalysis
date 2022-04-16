### Complex Dataseries-based CurveDatasource approach:
from typing import OrderedDict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyphoplacecellanalysis.General.Model.RenderDataseries import RenderDataseries
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves3D.Render3DTimeCurvesMixin import CurveDatasource


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
    ## Animal Behavioral Position 3D Time Curves
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



    ##########################################
    ## Randomly Generated (Testing) 3D Time Curves
    @staticmethod
    def build_test_3D_time_curves_datasource(test_data_random_df, pre_spatial_to_spatial_mappings, debug_print=False):
        # a value scalar for the z-axis
        # z_scaler = MinMaxScaler()
        # iterate through the data columns and build the configs from the names with default properties:
        valid_data_values_column_names = test_data_random_df.columns[1:]
        
        active_data_series_pre_spatial_list = [{'name':data_col_name,'t':'t','v_alt':None,'v_main':data_col_name,'color_name':'white', 'line_width': 1.0, 'z_scaling_factor':1.0}                                              
                                               for data_col_name in list(valid_data_values_column_names)]
        if debug_print:
            print(f'pre_spatial_to_spatial_mappings: {len(pre_spatial_to_spatial_mappings)}\nvalid_data_values_column_names: {valid_data_values_column_names}\nlen(active_data_series_pre_spatial_list): {len(active_data_series_pre_spatial_list)}')
        return CurveDatasource(test_data_random_df.copy(), data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(active_data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))

    @classmethod
    def build_test_3D_time_curves(cls, spike_raster_plt_3d, test_data_random_df=None, sample_rate_sec = 0.25, n_value_columns = 15):
        """ builds some randomly-generated 3D Curves for testing/debugging purposes and adds them to the spike_raster_plot
        Usage:
            active_random_test_plot_curve_datasource = build_test_3D_time_curves(test_data_random_df, spike_raster_plt_3d)
        """        
        def _generate_sample_random_data_series(t_start, t_end, sample_rate_sec = 0.25, n_value_columns = 15):
            """ Build a random dataframe with N data series (N curves) sampled every sample_rate_sec seconds """
            # test_data_t_series = np.linspace(test_data_start_t, test_data_end_t, 1000)
            test_data_t_series = np.arange(t_start, t_end, sample_rate_sec) # create a uniformly sampled timeseries with values every 1/4 second
            num_t_points = np.shape(test_data_t_series)[0]
            test_data_random_values_matrix = np.random.random(size=(n_value_columns, num_t_points))

            test_data_random_df = pd.DataFrame(np.concatenate((np.atleast_2d(test_data_t_series), test_data_random_values_matrix)).T) # 6868 rows × 16 columns
            test_data_random_df.columns = ['t'] + [f'v{i}' for i in np.arange(n_value_columns)]
            return test_data_random_df

        if test_data_random_df is not None:
            # If the passed-in argument is valid (not None), use that instead of generating a new random one.
            active_plot_df = test_data_random_df.copy()
        else:
            # otherwise generate a new random dataframe
            test_data_start_t, test_data_end_t = spike_raster_plt_3d.spikes_window.total_df_start_end_times
            active_plot_df = _generate_sample_random_data_series(test_data_start_t, test_data_end_t, sample_rate_sec = sample_rate_sec, n_value_columns = n_value_columns)
        
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: spike_raster_plt_3d.temporal_to_spatial(t) # returns the x-values, transforming from the times t appropriately.
        # y_map_fn = lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells) # This is what places all values along the back wall
        z_map_fn = lambda v_main: v_main + spike_raster_plt_3d.floor_z # returns the un-transformed primary value
        
        ## we want each test curve to be rendered with a unit_id (series of spikes), so we'll need custom y_map_fn's for each column
        num_t_points = np.shape(active_plot_df)[0]
        n_value_columns = np.shape(active_plot_df)[1] - 1 # get the total num columns, then subtract 1 to account for the 0th ('t') column

        ## want a separate y_map_fn for each data series so it returns the correct index
        # lambda v: np.full_like(v, -spike_raster_plt_3d.n_half_cells)
        # data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn} for i in np.arange(1, n_value_columns)]
        # data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':(lambda v, bound_i=i: np.full_like(v, bound_i)),'z_map_fn':z_map_fn} for i in np.arange(n_value_columns)]
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':(lambda v, bound_i=i: np.full_like(v, spike_raster_plt_3d.unit_id_to_spatial(bound_i))),'z_map_fn':z_map_fn} for i in np.arange(n_value_columns)]
        
        active_test_random_plot_curve_datasource = cls.build_test_3D_time_curves_datasource(active_plot_df, data_series_pre_spatial_to_spatial_mappings)
        # Add the datasource to the actual plotter object: this will cause it to build and add the 3D time curves:
        spike_raster_plt_3d.add_3D_time_curves(curve_datasource=active_test_random_plot_curve_datasource) # Add the curves from the datasource
                
        return active_test_random_plot_curve_datasource


    ##########################################
    ## Time Binned Unit Spike Counts 3D Time Curves
    @staticmethod
    def add_unit_time_binned_spike_counts_curves(curr_computations_results, active_curve_plotter_3d):
        """ Adds a a 3D line plot for each unit that displays the time binned spike rate
        
        Gets the result out of curr_computations_results.computed_data['pf2D_Decoder']
        
        Usage:
            ## Adds the binned_spike_counts curves:
            active_curve_plotter_3d = spike_raster_plt_3d ## PyQtGraph Mode
            # active_curve_plotter_3d = spike_raster_plt_3d_vedo ## Vedo Mode
            binned_spike_counts_curve_datasource = add_unit_time_binned_spike_counts_curves(curr_computations_results, active_curve_plotter_3d)
        """

        # Index Mapping:
        curr_neuron_IDs = np.array(curr_computations_results.computed_data['pf2D_Decoder'].neuron_IDs)
        # curr_neuron_IDXs = curr_computations_results.computed_data['pf2D_Decoder'].neuron_IDXs

        # need to filter spike_raster_plt_3d_vedo.cell_ids to only include the ones present in curr_neuron_IDs
        is_included_in_computation_result_neuron_IDs = np.isin(active_curve_plotter_3d.cell_ids, curr_neuron_IDs)
        included_neuron_ids = active_curve_plotter_3d.cell_ids[is_included_in_computation_result_neuron_IDs]
        # excluded_neuron_ids = active_curve_plotter_3d.cell_ids[~is_included_in_computation_result_neuron_IDs]
        
        # Data Mapping:
        data_values_column_names = [str(an_id) for an_id in included_neuron_ids]
        active_plot_df = pd.DataFrame(np.concatenate((np.atleast_2d(curr_computations_results.computed_data['pf2D_Decoder'].time_window_centers), curr_computations_results.computed_data['pf2D_Decoder'].unit_specific_time_binned_spike_counts)).T,
                                                             columns=(['t'] + data_values_column_names))
        # active_plot_df = test_unit_time_binned_spike_counts_df.copy()
        
        # a value scaler for the z-axis
        z_scaler = MinMaxScaler(feature_range=(0, 10.0), copy=True)
        columns = active_plot_df.columns.drop('t')
        active_plot_df[columns] = z_scaler.fit_transform(active_plot_df[columns])
        
        print(f'np.shape(active_plot_df[columns]): {np.shape(active_plot_df[columns])}')
        
        valid_data_values_column_names = active_plot_df.columns[1:]  # all but the 't' column
        
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: active_curve_plotter_3d.temporal_to_spatial(t) # returns the x-values, transforming from the times t appropriately.
        # y_map_fn = lambda v: np.full_like(v, -active_curve_plotter_3d.n_half_cells) # This is what places all values along the back wall
        # z_map_fn = lambda v_main: v_main + active_curve_plotter_3d.floor_z + active_curve_plotter_3d.params.spike_end_z # returns the un-transformed primary value
        # z_map_fn = lambda v_main: v_main + active_curve_plotter_3d.params.spike_end_z # returns the un-transformed primary value
        
        spike_height = active_curve_plotter_3d.params.spike_end_z - active_curve_plotter_3d.params.spike_start_z
        # z_map_fn = lambda v_main: v_main + active_curve_plotter_3d.params.spike_end_z + spike_height # returns the un-transformed primary value
        z_map_fn = lambda v_main: v_main + 5.0 # returns the un-transformed primary value
        
        
        ## we want each test curve to be rendered with a unit_id (series of spikes), so we'll need custom y_map_fn's for each column
        n_value_columns = np.shape(active_plot_df)[1] - 1 # get the total num columns, then subtract 1 to account for the 0th ('t') column

        ## want a separate y_map_fn for each data series so it returns the correct index
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':(lambda v, bound_i=i: np.full_like(v, active_curve_plotter_3d.unit_id_to_spatial(bound_i))),'z_map_fn':z_map_fn} for i in np.arange(n_value_columns)]

        data_col_name_to_unit_plot_color_rgba_map = dict()
        for data_col_name in list(valid_data_values_column_names):
            curr_color = active_curve_plotter_3d.params.neuron_qcolors_map[active_curve_plotter_3d.cell_id_to_unit_id_map[int(data_col_name)]] # a QColor
            curr_color.setAlphaF(0.2) # set the alpha
            data_col_name_to_unit_plot_color_rgba_map[data_col_name] = curr_color.getRgbF()
            
        # [active_curve_plotter_3d.params.neuron_qcolors_map[active_curve_plotter_3d.cell_id_to_unit_id_map[int(data_col_name)]].getRgbF() for data_col_name in list(valid_data_values_column_names)]
        
        active_data_series_pre_spatial_list = [{'name':data_col_name,'t':'t','v_alt':None,'v_main':data_col_name,
                                                # 'color_name':'black', # this will be overriden by the 'color' value below
                                                # 'color': active_curve_plotter_3d.params.neuron_qcolors_map[active_curve_plotter_3d.cell_id_to_unit_id_map[int(data_col_name)]].getRgbF(), # gets the color for a specified data_col_name
                                                'color': data_col_name_to_unit_plot_color_rgba_map[data_col_name], # gets the color for a specified data_col_name
                                                'line_width': 2.0, 'z_scaling_factor':1.0}
                                                    for data_col_name in list(valid_data_values_column_names)]
        # Build the finalized datasource for this object:
        binned_spike_counts_curve_datasource = CurveDatasource(active_plot_df.copy(), data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(active_data_series_pre_spatial_list, data_series_pre_spatial_to_spatial_mappings))
        # Add the datasource to the actual plotter object: this will cause it to build and add the 3D time curves:
        active_curve_plotter_3d.add_3D_time_curves(curve_datasource=binned_spike_counts_curve_datasource) # Add the curves from the datasource
        return binned_spike_counts_curve_datasource
