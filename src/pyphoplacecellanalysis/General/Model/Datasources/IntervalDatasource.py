import numpy as np
import pandas as pd

from neuropy.core import Epoch

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Datasources.Datasources import BaseDatasource, DataframeDatasource


class IntervalsDatasource(BaseDatasource):
    """ TODO: a datasource for interval data

    Contains a dataframe.
        
    Signals:
        source_data_changed_signal = QtCore.pyqtSignal(object) # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end)
        
    Usage:
        from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource

        _test_fn = Specific2DRenderTimeEpochsHelper.build_Laps_formatter_datasource(debug_print=False)
        test_laps_interval_datasource = IntervalsDatasource.init_from_epoch_object(sess.laps.as_epoch_obj(), _test_fn, datasource_name='intervals_datasource_from_laps_epoch_obj')
        test_laps_interval_datasource

        active_laps_interval_rects_item = Specific2DRenderTimeEpochsHelper.build_Laps_2D_render_time_epochs(sess, series_vertical_offset=43.0, series_height=2.0)
        

    """
    
    # _required_interval_time_columns = ['t_start', 't_duration', 't_end']
    _required_interval_time_columns = ['t_start', 't_duration']
    _all_interval_time_columns = ['t_start', 't_duration', 't_end']
    
    _required_interval_visualization_columns = ['t_start', 't_duration', 'series_vertical_offset', 'series_height', 'pen', 'brush']
    
    
    @property
    def time_column_names(self):
        """ the name of the relevant time column. Defaults to ['t_start', 't_duration', 't_end'] """
        return ['t_start', 't_duration', 't_end']
    
    @property
    def time_column_values(self):
        """ the values of only the relevant time columns """
        return self._df[self.time_column_names] # get only the relevant time column
    
    @property
    def data_column_names(self):
        """ the names of only the non-time columns """
        return np.setdiff1d(self._df.columns, np.array(self.time_column_names)) # get only the non-time columns
    
    @property
    def data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.data_column_names]
    
    @property
    def datasource_UIDs(self):
        """The datasource_UID property.
        """
        # return [f'{self.custom_datasource_name}.{col_name}' for col_name in self.data_column_values]
        return [f'{self.custom_datasource_name}']
    
    
    ## Active-Only versions of data_column_names, data_column_values, and datasource_UIDs that can be overriden to enable only a subset of the values
    @property
    def active_data_column_names(self):
        """ the names of only the non-time columns """
        return self.data_column_names
    
    @property
    def active_data_column_values(self):
        """ The values of only the non-time columns """
        return self._df[self.active_data_column_names]
    
    @property
    def active_datasource_UIDs(self):
        """The datasource_UID property."""
        return [f'{self.custom_datasource_name}']
    
    @property
    def total_datasource_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        return self.total_df_start_end_times
        
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest times in the total df """
        # earliest_df_time = np.nanmin(self.df[self.time_column_names[0]])
        # latest_df_time = np.nanmax(self.df[self.time_column_names[1]])
        df_timestamps = self.df[self.time_column_names].to_numpy()
        earliest_df_time = df_timestamps[0,0]
        latest_df_time = df_timestamps[-1,-1]
        return (earliest_df_time, latest_df_time)
    
    
    ##### Get/Set Properties ####:
    @property
    def df(self):
        """The df property."""
        return self._df
    @df.setter
    def df(self, value):
        self._df = value
        self.source_data_changed_signal.emit(self)
        

    def __init__(self, df, datasource_name='default_intervals_datasource'):
        # Initialize the datasource as a BaseDatasource
        BaseDatasource.__init__(self, datasource_name=datasource_name)
        self._df = df
        ## Validate that it has all required columns:
        assert np.isin(IntervalsDatasource._required_interval_time_columns, df.columns).all(), f"dataframe is missing required columns:\n Required: {IntervalsDatasource._required_interval_time_columns}, current: {df.columns} "
        
        
    def update_visualization_properties(self, dataframe_vis_columns_function):
        """ called to update the current visualization columns of the df by applying the provided function
        
        Usage:
        
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs

            def _updated_custom_interval_dataframe_visualization_columns_general_epoch(active_df, **kwargs):
                num_intervals = np.shape(active_df)[0]
                ## parameters:
                y_location = 0.0
                height = 40.5
                pen_color = pg.mkColor('white')
                pen_color.setAlphaF(0.8)

                brush_color = pg.mkColor('white')
                brush_color.setAlphaF(0.5)

                ## Update the dataframe's visualization columns:
                active_df = General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=y_location, height=height, pen_color=pen_color, brush_color=brush_color, **kwargs)
                return active_df

            # get the existing dataframe to be updated:
            datasource_to_update = active_2d_plot.interval_datasources.Ripples
            datasource_to_update.update_visualization_properties(_updated_custom_interval_dataframe_visualization_columns_general_epoch)

        """
        self._df = dataframe_vis_columns_function(self._df)
        self.source_data_changed_signal.emit(self) # Emit the data changed signal

        
        
    @classmethod
    def add_missing_reciprocal_columns_if_needed(cls, df):
        """ computes the missing column ('t_duration' or 't_end') from the other two columns. """
        if np.isin(['t_start', 't_duration', 't_end'], df.columns).all():
            return df # no changes needed
        else:
            # at least one is missing
            if 't_end' not in df.columns:
                # if t_end is the one missing, compute it
                df['t_end'] =  df['t_start'] +  df['t_duration']
                return df
            elif 't_duration' not in df.columns:
                # if t_duration is the one missing, compute it
                df['t_duration'] =  df['t_end'] - df['t_start']
                return df
            else:
                raise NotImplementedError
            return df
            
            
        
    @classmethod
    def init_from_times_values(cls, t_starts, t_durations, values, dataframe_vis_columns_function, datasource_name='intervals_datasource_from_epoch_obj'):
        plot_df = pd.DataFrame({'t_start': t_starts, 't_duration': t_durations, 't_end': (t_starts + t_durations), 'v': values})
        return cls.init_from_epoch_object(plot_df, dataframe_vis_columns_function=dataframe_vis_columns_function, datasource_name=datasource_name)
        # return cls(plot_df, datasource_name=datasource_name)
        
        
    @classmethod
    def init_from_epoch_object(cls, epochs, dataframe_vis_columns_function, datasource_name='intervals_datasource_from_epoch_obj', debug_print=False):
        """ Builds an appropriate IntervalsDatasource from any Epoch object and a function that is passed the converted dataframe and adds the visualization specific columns: ['series_vertical_offset', 'series_height', 'pen', 'brush']
        
        Input:
            epochs: Either a neuropy.core.Epoch object OR dataframe with the columns ['t_start', 't_duration']
            dataframe_vis_columns_function: callable that takes a pd.DataFrame that adds the remaining required columns to the dataframe if needed.
        
        Returns:
            IntervalsDatasource
        """
        if isinstance(epochs, Epoch):
            # if it's an Epoch, convert it to a dataframe
            raw_df = epochs.to_dataframe()
            active_df = pd.DataFrame({'t_start':raw_df.start.copy(), 't_duration':raw_df.duration.copy()}) # still will need columns ['series_vertical_offset', 'series_height', 'pen', 'brush'] added later
        elif isinstance(epochs, pd.DataFrame):
            # already a dataframe
            active_df = epochs.copy()
        else:
            raise NotImplementedError
        
        active_df = cls.add_missing_reciprocal_columns_if_needed(active_df)
        active_df = dataframe_vis_columns_function(active_df)
        return cls(active_df, datasource_name=datasource_name)
        
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        # return self.df[self.df[self.time_column_names].between(new_start, new_end)]
        # self.df['t_start', 't_duration', 't_end']    
        return self.df[self.df['t_start', 't_end'].between(new_start, new_end)]
    

