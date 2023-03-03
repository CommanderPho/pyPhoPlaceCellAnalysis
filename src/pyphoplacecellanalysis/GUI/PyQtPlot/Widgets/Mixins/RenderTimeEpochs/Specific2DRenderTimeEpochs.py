import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available

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
    def _update_df_visualization_columns(cls, active_df, y_location=None, height=None, pen_color=None, brush_color=None, **kwargs):
        """ updates the columns of the provided active_df given the values specified. If values aren't provided, they aren't changed. """        
        # Update only the provided columns while leaving the others intact
        if y_location is not None:
            ## y_location:
            if isinstance(y_location, (list, tuple)):
                active_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', [a_y_location for a_y_location in y_location])
            else:
                # Scalar value assignment:
                active_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
                
        if height is not None:
            ## series_height:
            if isinstance(height, (list, tuple)):
                active_df['series_height'] = kwargs.setdefault('series_height', [a_height for a_height in height])
            else:
                # Scalar value assignment:
                active_df['series_height'] = kwargs.setdefault('series_height', height)

        if pen_color is not None:
            ## pen_color:
            if isinstance(pen_color, (list, tuple)):
                active_df['pen'] = kwargs.setdefault('pen', [pg.mkPen(a_pen_color) for a_pen_color in pen_color])
            else:
                # Scalar value assignment:
                active_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
            
        if brush_color is not None:
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
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch

    @classmethod
    def build_render_time_epochs_datasource(cls, active_epochs_obj, **kwargs):
        """ allows specifying a custom formatter function """
        custom_epochs_df_formatter = kwargs.pop('epochs_dataframe_formatter', None)
        
        if custom_epochs_df_formatter is None:
            active_epochs_df_formatter = cls.build_epochs_dataframe_formatter(**kwargs)
        else:
            print(f'overriding default epochs_df_formatter...')
            active_epochs_df_formatter = custom_epochs_df_formatter(cls, **kwargs)
        
        if isinstance(active_epochs_obj, Epoch):
            general_epochs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_epochs_obj, active_epochs_df_formatter, datasource_name='intervals_datasource_from_general_Epochs_obj')
            
        elif isinstance(active_epochs_obj, pd.DataFrame):
            ## NOTE that build_epochs_dataframe_formatter is never called if a dataframe is passed in directly, and the dataframe's columns must be named exactly correctly
            general_epochs_interval_datasource = IntervalsDatasource(active_epochs_obj, datasource_name='intervals_datasource_from_general_dataframe_obj')
            
        elif isinstance(active_epochs_obj, tuple):
            assert len(active_epochs_obj) == 3
            # raise NotImplementedError # These do not work because they don't get the required columns added via cls.build_epochs_dataframe_formatter(**kwargs)
            # must be a tuple containing (t_starts, t_durations, optional_values/ids)
            general_epochs_interval_datasource = IntervalsDatasource.init_from_times_values(*active_epochs_obj, active_epochs_df_formatter, datasource_name='intervals_datasource_from_general_times_tuple_obj')
            
            
        else:
            raise NotImplementedError
        return general_epochs_interval_datasource

    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        if isinstance(curr_sess, DataSession):
            active_Epochs = curr_sess.epochs # <Epoch> object
        elif isinstance(curr_sess, (Epoch, pd.DataFrame, tuple)):
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
            y_location = -1.0
            height = 0.9
            # pen_color = pg.mkColor('red')
            # brush_color = pg.mkColor('red')

            ## parameters:
            pen_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            brush_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
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
            y_location = -2.0
            height = 0.9
            pen_color = pg.mkColor('red')
            brush_color = pg.mkColor('red')
            brush_color.setAlphaF(0.5)
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            
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
            y_location = -3.0
            height = 0.9
            pen_color = pg.mkColor('w')
            pen_color.setAlphaF(0.8)
            brush_color = pg.mkColor('grey')
            brush_color.setAlphaF(0.5)
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
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
            y_location = -4.0
            height = 0.9
            pen_color = pg.mkColor('orange')
            brush_color = pg.mkColor('orange')
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
    
    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        if isinstance(curr_sess, DataSession):
            active_Epochs = curr_sess.epochs # <Epoch> object
        elif isinstance(curr_sess, Epoch):
            active_Epochs = curr_sess  # <Epoch> object passed directly
        elif isinstance(curr_sess, pd.DataFrame):
            active_Epochs = (curr_sess['start'].to_numpy(), curr_sess['duration'].to_numpy(), curr_sess['flat_replay_idx'].to_numpy()) 
        else:
            raise NotImplementedError
        interval_datasource = cls.build_render_time_epochs_datasource(active_epochs_obj=active_Epochs, **kwargs)
        out_rects = destination_plot.add_rendered_intervals(interval_datasource, name=kwargs.setdefault('name', cls.default_datasource_name), debug_print=True)
        
        
    
##########################################
## Ripples
class Ripples_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    default_datasource_name = 'Ripples'
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            ## parameters:
            y_location = -5.0
            height = 0.9
            pen_color = pg.mkColor('blue')
            brush_color = pg.mkColor('blue')
            brush_color.setAlphaF(0.5)            
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df
        return _add_interval_dataframe_visualization_columns_general_epoch
        

##########################################
## New Ripples
class NewRipples_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    default_datasource_name = 'NewRipples'
    
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            ## parameters:
            y_location = 0.0
            height = 2.0
            pen_color = pg.mkColor('cyan')
            brush_color = pg.mkColor('cyan')
            brush_color.setAlphaF(0.5)
            
            ## Add the missing parameters to the dataframe:
            active_df = cls._update_df_visualization_columns(active_df, y_location, height, pen_color, brush_color, **kwargs)
            return active_df

        return _add_interval_dataframe_visualization_columns_general_epoch
        

def inline_mkColor(color, alpha=1.0):
    """ helps build a new QColor for a pen/brush in an inline (single-line) way. """
    out_color = pg.mkColor(color)
    out_color.setAlphaF(alpha)
    return out_color

""" HISTORICAL NOTE: Specific2DRenderTimeEpochsHelper has been removed in favor of a class-based approach """