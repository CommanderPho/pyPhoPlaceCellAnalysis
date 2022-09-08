import numpy as np
import pandas as pd

from neuropy.core.laps import Laps
from neuropy.core.epoch import Epoch
from neuropy.core.session.dataSession import DataSession


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource

""" 
A general epochs_dataframe_formatter takes a dataframe and adds the required columns

"""
class General2DRenderTimeEpochs(object):
    """docstring for General2DRenderTimeEpochs."""
    def __init__(self):
        super(General2DRenderTimeEpochs, self).__init__()
    
    default_datasource_name = 'GeneralEpochs'
    
    @classmethod
    def _add_missing_df_columns(cls, active_df, y_location, height, pen_color, brush_color, **kwargs):
        ## Add the missing parameters to the dataframe:
            ## y_location:
            if isinstance(y_location, (list, tuple)):
                active_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', [a_y_location for a_y_location in y_location])
            else:
                # Scalar value assignment:
                active_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
            ## series_height:
            if isinstance(height, (list, tuple)):
                active_df['series_height'] = kwargs.setdefault('series_height', [a_height for a_height in height])
            else:
                # Scalar value assignment:
                active_df['series_height'] = kwargs.setdefault('series_height', height)
                
            ## pen_color:
            if isinstance(pen_color, (list, tuple)):
                active_df['pen'] = kwargs.setdefault('pen', [pg.mkPen(a_pen_color) for a_pen_color in pen_color])
            else:
                # Scalar value assignment:
                active_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
            ## brush_color:
            if isinstance(brush_color, (list, tuple)):
                active_df['brush'] = kwargs.setdefault('brush', [pg.mkBrush(a_color) for a_color in brush_color])  
            else:
                # Scalar value assignment:
                active_df['brush'] = kwargs.setdefault('brush', pg.mkBrush(brush_color))
            
            return active_df #, kwargs
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_df)[0]
            
            ## parameters:
            y_location = 0.0
            height = 1.0
            pen_color = pg.mkColor('red')
            brush_color = pg.mkColor('red')

            ## parameters:            
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch

    @classmethod
    def build_render_time_epochs_datasource(cls, active_epochs_obj, **kwargs):
        general_epochs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_epochs_obj, cls.build_epochs_dataframe_formatter(**kwargs), datasource_name='intervals_datasource_from_general_Epochs_obj')
        return general_epochs_interval_datasource

    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        if isinstance(curr_sess, DataSession):
            active_Epochs = curr_sess.epochs # <Epoch> object
        elif isinstance(curr_sess, Epoch):
            active_Epochs = curr_sess  # <Epoch> object passed directly
        else:
            raise NotImplementedError
        interval_datasource = cls.build_render_time_epochs_datasource(active_epochs_obj=active_Epochs, **kwargs)
        out_rects = destination_plot.add_rendered_intervals(interval_datasource, name=kwargs.setdefault('name', cls.default_datasource_name), debug_print=True)
        
##########################################
## General Epochs
class SessionEpochs2DRenderTimeEpochs(General2DRenderTimeEpochs):
    """docstring for SessionEpochs2DRenderTimeEpochs."""
    default_datasource_name = 'SessionEpochs'
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_df)[0]
            ## parameters:
            y_location = 0.0
            height = 1.0
            # pen_color = pg.mkColor('red')
            # brush_color = pg.mkColor('red')

            ## parameters:
            pen_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            brush_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
    
##########################################
## Laps
class Laps2DRenderTimeEpochs(General2DRenderTimeEpochs):
    """docstring for Laps2DRenderTimeEpochs."""
    default_datasource_name = 'Laps'

    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_df)[0]
            ## parameters:
            y_location = 0.0
            height = 1.0
            pen_color = pg.mkColor('red')
            brush_color = pg.mkColor('red')
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
    
    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        if isinstance(curr_sess, DataSession):
            active_Epochs = curr_sess.laps.as_epoch_obj() # <Epoch> object
        elif isinstance(curr_sess, Laps):
            active_Epochs = curr_sess.as_epoch_obj()
        elif isinstance(curr_sess, Epoch):
            active_Epochs = curr_sess
        else:
            raise NotImplementedError
        interval_datasource = cls.build_render_time_epochs_datasource(active_epochs_obj=active_Epochs, **kwargs)
        out_rects = destination_plot.add_rendered_intervals(interval_datasource, name=kwargs.setdefault('name', cls.default_datasource_name), debug_print=True)
        
##########################################
## PBE (Population Burst Events)
class PBE_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    """docstring for PBE_2DRenderTimeEpochs."""
    default_datasource_name = 'PBEs'
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_df)[0]
            ## parameters:
            y_location = 0.0
            height = 2.5
            pen_color = pg.mkColor('w')
            brush_color = pg.mkColor('grey')
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
      
      

##########################################
## Replays
class Replays_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    default_datasource_name = 'Replays'
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            ## parameters:
            y_location = 0.0
            height = 5.5
            pen_color = pg.mkColor('orange')
            brush_color = pg.mkColor('orange')
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
    
    
##########################################
## Ripples
class Ripples_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    default_datasource_name = 'Ripples'
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            ## parameters:
            y_location = 10.0
            height = 8.5
            pen_color = pg.mkColor('blue')
            brush_color = pg.mkColor('blue')
            ## Add the missing parameters to the dataframe:
            active_df = cls._add_missing_df_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
        
""" HISTORICAL NOTE: Specific2DRenderTimeEpochsHelper has been removed in favor of a class-based approach """