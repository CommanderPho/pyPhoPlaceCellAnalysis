### Complex Dataseries-based CurveDatasource approach:
from typing import OrderedDict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Model.RenderDataseries import RenderDataseries
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin import CurveDatasource

##########################################
## General Render Time Curves
class GeneralRenderTimeCurves(object):
    """docstring for GeneralRenderTimeCurves.
    Analagous to the class-based General2DRenderTimeEpochs in pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs
    """
    def __init__(self):
        super(GeneralRenderTimeCurves, self).__init__()
    
    @classmethod
    def build_render_time_curves_datasource(cls, plot_df, pre_spatial_to_spatial_mappings, **kwargs):
        # additional properties:
        data_series_pre_spatial_list = [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos','color_name':'yellow', 'line_width':1.25, 'z_scaling_factor':1.0},
            {'name':'x position','t':'t','v_alt':None,'v_main':'x', 'color_name':'red', 'line_width':0.5, 'z_scaling_factor':1.0},
            {'name':'y position','t':'t','v_alt':None,'v_main':'y', 'color_name':'green', 'line_width':0.5, 'z_scaling_factor':1.0}
        ]
        
        # a value scalar for the z-axis
        z_scaler = MinMaxScaler()
        active_plot_curve_dataframe = plot_df[['t','x','y','lin_pos']].copy()
        active_plot_curve_dataframe[['x','y']] = z_scaler.fit_transform(active_plot_curve_dataframe[['x','y']]) # scale x and y positions
        active_plot_curve_dataframe[['lin_pos']] = z_scaler.fit_transform(active_plot_curve_dataframe[['lin_pos']]) # scale lin_pos position separately

        general_curve_interval_datasource = CurveDatasource(active_plot_curve_dataframe, data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))
        return general_curve_interval_datasource

    @classmethod
    def add_render_time_curves(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells)
        z_map_fn = lambda v_main: v_main
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
        ]
        active_plot_curve_datasource = cls.build_render_time_curves_datasource(curr_sess.position.to_dataframe(), data_series_pre_spatial_to_spatial_mappings)
        destination_plot.add_3D_time_curves(curve_datasource=active_plot_curve_datasource) # Add the curves from the datasource
        return active_plot_curve_datasource



# ==================================================================================================================== #
# General Position Dataframe                                                                                           #
# ==================================================================================================================== #
class BasePositionDataframeRenderTimeCurves(GeneralRenderTimeCurves):
    """ An abstract base class for RenderTimeCurves that use the position dataframe
        
    Examples include: PositionRenderTimeCurves, VelocityRenderTimeCurves

    add_render_time_curves
        build_pre_spatial_to_spatial_mappings
        build_render_time_curves_datasource
            prepare_dataframe
            data_series_pre_spatial_list
    
    """
    default_datasource_name = 'BasePositionDataframeTimeCurves'
    
    @classmethod
    def data_series_pre_spatial_list(cls, *args, **kwargs):
        """ returns the pre_spatial list for the dataseries. Usually just returns a constant, only a function in case a class wants to do separate setup based on a class property. """
        raise NotImplementedError # MUST OVERRIDE
         
    @classmethod
    def prepare_dataframe(cls, plot_df, *args, **kwargs):
        """ preforms and pre-processing of the dataframe needed (such as scaling/renaming columns/etc and returns a COPY """
        raise NotImplementedError # MUST OVERRIDE

    @classmethod
    def build_pre_spatial_to_spatial_mappings(cls, destination_plot, *args, **kwargs):
        """ builds and returns the mappings from the pre-spatial values to the spatial values, frequently using information from the destination_plot and passed-in variables. """
        raise NotImplementedError # MUST OVERRIDE

    @classmethod
    def build_render_time_curves_datasource(cls, plot_df, pre_spatial_to_spatial_mappings, **kwargs):
        """ CONSTANT: typically shouldn't need to be overriden, just set up this way for customizability """
        data_series_pre_spatial_list = cls.data_series_pre_spatial_list()
        active_plot_curve_dataframe = cls.prepare_dataframe(plot_df)
        general_curve_interval_datasource = CurveDatasource(active_plot_curve_dataframe, data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))
        return general_curve_interval_datasource

    @classmethod
    def add_render_time_curves(cls, curr_sess, destination_plot, **kwargs):
        """ CONSTANT: directly-called method 
        destination_plot should implement add_rendered_intervals
        Calls `destination_plot.add_3D_time_curves(...)`
        
        ## TODO: figure out how data should be provided to enable maximum generality. It seems that all datasources are currently dataframe based. 
        
        curr_sess: The session containing the data to be plotted. 
        
        """
        plot_df = curr_sess.position.to_dataframe()
        data_series_pre_spatial_to_spatial_mappings = cls.build_pre_spatial_to_spatial_mappings(destination_plot)
        active_plot_curve_datasource = cls.build_render_time_curves_datasource(plot_df, data_series_pre_spatial_to_spatial_mappings)
        destination_plot.add_3D_time_curves(curve_datasource=active_plot_curve_datasource) # Add the curves from the datasource
        return active_plot_curve_datasource




##########################################
## Animal Position Curves
class PositionRenderTimeCurves(BasePositionDataframeRenderTimeCurves):
    """ 
    add_render_time_curves
        build_pre_spatial_to_spatial_mappings
        build_render_time_curves_datasource
            prepare_dataframe
            data_series_pre_spatial_list
    
    """
    default_datasource_name = 'PositionTimeCurves'
    
    @classmethod
    def data_series_pre_spatial_list(cls, *args, **kwargs):
        """ returns the pre_spatial list for the dataseries. Usually just returns a constant, only a function in case a class wants to do separate setup based on a class property. """
        return [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos','color_name':'yellow', 'line_width':1.25, 'z_scaling_factor':1.0},
            {'name':'x position','t':'t','v_alt':None,'v_main':'x', 'color_name':'red', 'line_width':0.5, 'z_scaling_factor':1.0},
            {'name':'y position','t':'t','v_alt':None,'v_main':'y', 'color_name':'green', 'line_width':0.5, 'z_scaling_factor':1.0}
        ]
         
    @classmethod
    def prepare_dataframe(cls, plot_df, *args, **kwargs):
        """ preforms and pre-processing of the dataframe needed (such as scaling/renaming columns/etc and returns a COPY """
        z_scaler = MinMaxScaler()
        transformed_df = plot_df[['t','x','y','lin_pos']].copy()
        transformed_df[['x','y']] = z_scaler.fit_transform(transformed_df[['x','y']]) # scale x and y positions
        transformed_df[['lin_pos']] = z_scaler.fit_transform(transformed_df[['lin_pos']]) # scale lin_pos position separately
        return transformed_df


    @classmethod
    def build_pre_spatial_to_spatial_mappings(cls, destination_plot, *args, **kwargs):
        """ builds and returns the mappings from the pre-spatial values to the spatial values, frequently using information from the destination_plot and passed-in variables. """
        if destination_plot.time_curve_render_dimensionality == 2:
            # SpikeRaster2D needs different x_map_fn than the 3D plots:
            x_map_fn = lambda t: t
        else:            
            x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)

        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells)
        z_map_fn = lambda v_main: v_main
        return [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
        ]


# ==================================================================================================================== #
# Animal Velocity Curves                                                                                               #
# ==================================================================================================================== #
class VelocityRenderTimeCurves(BasePositionDataframeRenderTimeCurves):
    """ 
    add_render_time_curves
        build_pre_spatial_to_spatial_mappings
        build_render_time_curves_datasource
            prepare_dataframe
            data_series_pre_spatial_list
    
    """
    default_datasource_name = 'VelocityTimeCurves'
    
    @classmethod
    def data_series_pre_spatial_list(cls, *args, **kwargs):
        """ returns the pre_spatial list for the dataseries. Usually just returns a constant, only a function in case a class wants to do separate setup based on a class property. """
        return [{'name':'speed','t':'t','v_alt':None,'v_main':'speed','color_name':'orange', 'line_width':1.25, 'z_scaling_factor':1.0}]
         
    @classmethod
    def prepare_dataframe(cls, plot_df, *args, **kwargs):
        """ preforms and pre-processing of the dataframe needed (such as scaling/renaming columns/etc and returns a COPY """
        z_scaler = MinMaxScaler()
        transformed_df = plot_df[['t','speed']].copy()
        transformed_df[['speed']] = z_scaler.fit_transform(transformed_df[['speed']]) # scale speed position separately
        return transformed_df


    @classmethod
    def build_pre_spatial_to_spatial_mappings(cls, destination_plot, *args, **kwargs):
        """ builds and returns the mappings from the pre-spatial values to the spatial values, frequently using information from the destination_plot and passed-in variables. """
        if destination_plot.time_curve_render_dimensionality == 2:
            # SpikeRaster2D needs different x_map_fn than the 3D plots:
            x_map_fn = lambda t: t
        else:            
            x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells)
        z_map_fn = lambda v_main: v_main
        return [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}]




# ==================================================================================================================== #
# Test Custom Position DF                                                                                              #
# ==================================================================================================================== #
class ConfigurableRenderTimeCurves(BasePositionDataframeRenderTimeCurves):
    """ 
    add_render_time_curves
        build_pre_spatial_to_spatial_mappings
        build_render_time_curves_datasource
            prepare_dataframe
            data_series_pre_spatial_list
    
    """
    default_datasource_name = 'ConfigurableMovementTimeCurves'
    # active_variable_names = ['x','y','lin_pos','speed']
    active_variable_names = ['lin_pos','speed']

    @classmethod
    def data_series_pre_spatial_list(cls, *args, **kwargs):
        """ returns the pre_spatial list for the dataseries. Usually just returns a constant, only a function in case a class wants to do separate setup based on a class property. """
        return [v for v in [{'name':'linear position','t':'t','v_alt':None,'v_main':'lin_pos','color_name':'yellow', 'line_width':1.25, 'z_scaling_factor':1.0},
            {'name':'x position','t':'t','v_alt':None,'v_main':'x', 'color_name':'red', 'line_width':0.5, 'z_scaling_factor':1.0},
            {'name':'y position','t':'t','v_alt':None,'v_main':'y', 'color_name':'green', 'line_width':0.5, 'z_scaling_factor':1.0},
            {'name':'speed','t':'t','v_alt':None,'v_main':'speed','color_name':'orange', 'line_width':1.25, 'z_scaling_factor':1.0}
        ] if v['v_main'] in cls.active_variable_names]
         
    @classmethod
    def prepare_dataframe(cls, plot_df, *args, **kwargs):
        """ preforms and pre-processing of the dataframe needed (such as scaling/renaming columns/etc and returns a COPY """
        z_scaler = MinMaxScaler()
        included_columns = ['t', *cls.active_variable_names]
        transformed_df = plot_df[included_columns].copy()
        if ('x' in included_columns) and ('y' in included_columns):
            transformed_df[['x','y']] = z_scaler.fit_transform(transformed_df[['x','y']]) # scale x and y positions
        elif ('x' in included_columns):
            transformed_df[['x']] = z_scaler.fit_transform(transformed_df[['x']]) # scale x and y positions
        if 'lin_pos' in included_columns:
            transformed_df[['lin_pos']] = z_scaler.fit_transform(transformed_df[['lin_pos']]) # scale lin_pos position separately
        if 'speed' in included_columns:
            transformed_df[['speed']] = z_scaler.fit_transform(transformed_df[['speed']]) # scale speed position separately
        return transformed_df


    @classmethod
    def build_pre_spatial_to_spatial_mappings(cls, destination_plot, *args, **kwargs):
        """ builds and returns the mappings from the pre-spatial values to the spatial values, frequently using information from the destination_plot and passed-in variables. """
        if destination_plot.time_curve_render_dimensionality == 2:
            # SpikeRaster2D needs different x_map_fn than the 3D plots:
            x_map_fn = lambda t: t
        else:            
            x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)

        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells)
        z_map_fn = lambda v_main: v_main
        return [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
            {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
        ]


    @classmethod
    def build_render_time_curves_datasource(cls, plot_df, pre_spatial_to_spatial_mappings, **kwargs):
        """ CONSTANT: typically shouldn't need to be overriden, just set up this way for customizability """
        data_series_pre_spatial_list = cls.data_series_pre_spatial_list()
        active_plot_curve_dataframe = cls.prepare_dataframe(plot_df)
        general_curve_interval_datasource = CurveDatasource(active_plot_curve_dataframe, data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))
        return general_curve_interval_datasource

    @classmethod
    def add_render_time_curves(cls, curr_sess, destination_plot, **kwargs):
        """ CONSTANT: directly-called method 
        destination_plot should implement add_rendered_intervals
        destination_plot.add_3D_time_curves(...)
        
        ## TODO: figure out how data should be provided to enable maximum generality. It seems that all datasources are currently dataframe based. 
        
        curr_sess: The session containing the data to be plotted. 
        
        """
        plot_df = curr_sess.position.to_dataframe()
        data_series_pre_spatial_to_spatial_mappings = cls.build_pre_spatial_to_spatial_mappings(destination_plot)
        active_plot_curve_datasource = cls.build_render_time_curves_datasource(plot_df, data_series_pre_spatial_to_spatial_mappings)
        destination_plot.add_3D_time_curves(curve_datasource=active_plot_curve_datasource) # Add the curves from the datasource
        return active_plot_curve_datasource



##########################################
## MUA (Multi-Unit Activity) Curves
class MUA_RenderTimeCurves(GeneralRenderTimeCurves):
    """ builds the MUA (Multi-Unit Activity) 3D Curves and adds them to the spike_raster_plot
    Usage:
        active_mua_plot_curve_datasource = Specific3DTimeCurvesHelper.build_MUA_3D_time_curves(curr_sess, spike_raster_plt_3d)
    """
    default_datasource_name = 'MUA_TimeCurves'
    
    @classmethod
    def build_render_time_curves_datasource(cls, plot_df, pre_spatial_to_spatial_mappings, **kwargs):
        # additional properties:
        data_series_pre_spatial_list = [{'name':'mua_firing_rate','t':'t','v_alt':None,'v_main':'mua_firing_rate','color_name':'white', 'line_width':2.0, 'z_scaling_factor':1.0},
                                        {'name':'mua_spike_counts','t':'t','v_alt':None,'v_main':'mua_spike_counts','color_name':'grey', 'line_width':0.5, 'z_scaling_factor':1.0}
                                       ]
        
        # a value scalar for the z-axis
        z_scaler = MinMaxScaler()
        active_plot_curve_dataframe = plot_df[['t','mua_firing_rate', 'mua_spike_counts']].copy()
        active_plot_curve_dataframe[['mua_firing_rate']] = z_scaler.fit_transform(plot_df[['mua_firing_rate']]) # scale mua_firing_rate separately
        active_plot_curve_dataframe[['mua_spike_counts']] = z_scaler.fit_transform(plot_df[['mua_spike_counts']]) # scale mua_spike_counts separately
        general_curve_interval_datasource = CurveDatasource(active_plot_curve_dataframe, data_series_specs=RenderDataseries.init_from_pre_spatial_data_series_list(data_series_pre_spatial_list, pre_spatial_to_spatial_mappings))
        return general_curve_interval_datasource

    @classmethod
    def add_render_time_curves(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)
        mua_plot_df = pd.DataFrame({'t': curr_sess.mua.time, 'mua_firing_rate': curr_sess.mua.firing_rate, 'mua_spike_counts': curr_sess.mua.spike_counts}).copy()
        # Mappings from the pre-spatial values to the spatial values:
        x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells) # This is what places all values along the back wall
        z_map_fn = lambda v_main: v_main # returns the un-transformed primary value
        data_series_pre_spatial_to_spatial_mappings = [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn},
                                {'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}
                                ]
        active_plot_curve_datasource = cls.build_render_time_curves_datasource(mua_plot_df, data_series_pre_spatial_to_spatial_mappings)
        destination_plot.add_3D_time_curves(curve_datasource=active_plot_curve_datasource) # Add the curves from the datasource
        return active_plot_curve_datasource
    








# ==================================================================================================================== #
# Animal RelativeEntropySurprise Curves                                                                                               #
# ==================================================================================================================== #
@metadata_attributes(short_name=None, tags=['surprise', 'relative_entropy', 'render_time_curve'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-26 13:53', related_items=[])
class RelativeEntropySurpriseRenderTimeCurves(BasePositionDataframeRenderTimeCurves):
    """ 
    add_render_time_curves
        build_pre_spatial_to_spatial_mappings
        build_render_time_curves_datasource
            prepare_dataframe
            data_series_pre_spatial_list
    
    """
    default_datasource_name = 'RelativeEntropySurpriseTimeCurves'
    
    @classmethod
    def data_series_pre_spatial_list(cls, *args, **kwargs):
        """ returns the pre_spatial list for the dataseries. Usually just returns a constant, only a function in case a class wants to do separate setup based on a class property. """
        return [{'name':'speed','t':'t','v_alt':None,'v_main':'speed','color_name':'orange', 'line_width':1.25, 'z_scaling_factor':1.0}]
         
    @classmethod
    def prepare_dataframe(cls, plot_df, *args, **kwargs):
        """ preforms and pre-processing of the dataframe needed (such as scaling/renaming columns/etc and returns a COPY """
        z_scaler = MinMaxScaler()
        transformed_df = plot_df[['t','speed']].copy()
        transformed_df[['speed']] = z_scaler.fit_transform(transformed_df[['speed']]) # scale speed position separately
        return transformed_df


    @classmethod
    def build_pre_spatial_to_spatial_mappings(cls, destination_plot, *args, **kwargs):
        """ builds and returns the mappings from the pre-spatial values to the spatial values, frequently using information from the destination_plot and passed-in variables. """
        if destination_plot.time_curve_render_dimensionality == 2:
            # SpikeRaster2D needs different x_map_fn than the 3D plots:
            x_map_fn = lambda t: t
        else:            
            x_map_fn = lambda t: destination_plot.temporal_to_spatial(t)
        y_map_fn = lambda v: np.full_like(v, -destination_plot.n_half_cells)
        z_map_fn = lambda v_main: v_main
        return [{'name':'name','x':'t','y':'v_alt','z':'v_main','x_map_fn':x_map_fn,'y_map_fn':y_map_fn,'z_map_fn':z_map_fn}]





