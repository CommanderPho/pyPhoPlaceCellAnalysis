from copy import copy, deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray

import numpy as np
import pandas as pd

import neuropy.utils.type_aliases as types
from neuropy.core import Epoch
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Datasources.Datasources import BaseDatasource, DataframeDatasource

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import RectangleRenderTupleHelpers # used for  `get_serialized_data` and `get_deserialized_data`


class IntervalsDatasource(BaseDatasource):
    """ a datasource for interval data

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
    
    _time_column_name_synonyms = {"t_start":{'begin','start','start_t'},
        't_end':['end','stop','stop_t'],
        "t_duration":['duration'],
    }

    
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
        
        if not np.isin(IntervalsDatasource._required_interval_time_columns, df.columns).all():
            ## try to do the rename
            df = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=df, required_columns_synonym_dict=self.__class__._time_column_name_synonyms)

        ## Validate that it has all required columns:
        assert np.isin(IntervalsDatasource._required_interval_time_columns, df.columns).all(), f"dataframe is missing required columns:\n Required: {IntervalsDatasource._required_interval_time_columns}, current: {df.columns} "
        self._df = df


        
    def update_visualization_properties(self, dataframe_vis_columns_function: Union[callable, Dict]):
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
        if isinstance(dataframe_vis_columns_function, dict):
            ## a dict instead of a callable function. Build the callable function from the dict
            an_epoch_formatting_dict = dataframe_vis_columns_function
            dataframe_vis_columns_function = lambda active_df, **kwargs: self.__class__._update_df_visualization_columns(active_df, **(an_epoch_formatting_dict | kwargs))

        self._df = dataframe_vis_columns_function(self._df)
        self.source_data_changed_signal.emit(self) # Emit the data changed signal

    
    def recover_positioning_properties(self):
        """ Tries to recover the positioning properties from each of the interval_datasources of active_2d_plot
        
        Usage:

            all_series_positioning_dfs, all_series_compressed_positioning_dfs = active_2d_plot.extract_interval_bottom_top_area()
            # all_series_positioning_dfs
            all_series_compressed_positioning_dfs

        all_series_compressed_positioning_dfs: {'PBEs': {'y_location': -11.666666666666668, 'height': 4.166666666666667},
        'Ripples': {'y_location': -15.833333333333336, 'height': 4.166666666666667},
        'Replays': {'y_location': -20.000000000000004, 'height': 4.166666666666667},
        'Laps': {'y_location': -7.083333333333334, 'height': 4.166666666666667},
        'SessionEpochs': {'y_location': -2.916666666666667, 'height': 2.0833333333333335}}

        """
        series_positioning_df = self.df[['series_vertical_offset', 'series_height']].copy() # , 'pen', 'brush'
        # # series can render either 'above' or 'below':
        # series_positioning_df['is_series_below'] = (series_positioning_df['series_vertical_offset'] <= 0.0) # all elements less than or equal to zero indicate that it's below the plot, and its height will be added negatively to find the max-y value
        # _curr_active_effective_series_heights = series_positioning_df.series_height.values.copy()
        # _curr_active_effective_series_heights[series_positioning_df['is_series_below'].values] = -1.0 * _curr_active_effective_series_heights[series_positioning_df['is_series_below'].values] # effective heights are negative for series below the y-axis
        # series_positioning_df['effective_series_heights'] = _curr_active_effective_series_heights # curr_df['series_height'].copy()
        # series_positioning_df['effective_series_extreme_vertical_offsets'] = series_positioning_df['effective_series_heights'] + series_positioning_df['series_vertical_offset']
        # Generate a compressed-position representation of curr_df:
        a_compressed_series_positioning_df = series_positioning_df.drop_duplicates(inplace=False)
        series_compressed_positioning_df = a_compressed_series_positioning_df
        series_compressed_positioning_update_dict = None

        if a_compressed_series_positioning_df.shape[0] == 1:
            # only one entry, to be expected
            series_compressed_positioning_update_dict = {k:list(v.values())[0] for k, v in a_compressed_series_positioning_df.to_dict().items() if k in ['series_vertical_offset', 'series_height']}
            ## Rename columns for update outputs:
            series_compressed_positioning_update_dict['y_location'] = series_compressed_positioning_update_dict.pop('series_vertical_offset')
            series_compressed_positioning_update_dict['height'] = series_compressed_positioning_update_dict.pop('series_height')                
        else:
            series_compressed_positioning_update_dict = None

        return series_positioning_df, series_compressed_positioning_df, series_compressed_positioning_update_dict


    def get_serialized_data(self, drop_duplicates=False):
        """ converts the 'pen' and 'brush' columns of self.df to hashable tuples for serialization.
        Fixes # TypeError: unhashable type: 'QPen'
        """
        interval_datasource_df = self.df
        serialized_df = deepcopy(interval_datasource_df)
        if 'pen_tuple' not in serialized_df.columns:
            serialized_df['pen_tuple'] = [RectangleRenderTupleHelpers.QPen_to_tuple(a_pen) for a_pen in serialized_df['pen']] # gets the RgbF values of the QColor returned from the QPen a_pen
        if 'brush_tuple' not in serialized_df.columns:
            serialized_df['brush_tuple'] = [RectangleRenderTupleHelpers.QBrush_to_tuple(a_brush) for a_brush in serialized_df['brush']] # gets the RgbF values of the QColor returned from the QBrush a_brush
        # overwrite:
        serialized_df['pen'] = serialized_df['pen_tuple']
        serialized_df['brush'] = serialized_df['brush_tuple']
        if drop_duplicates:
            return serialized_df[['series_vertical_offset','series_height','pen','brush']].drop_duplicates(inplace=False)
        else:
            return serialized_df[['series_vertical_offset','series_height','pen','brush']] # return all entries


    @classmethod
    def get_deserialized_data(cls, serialized_df):
        """ converts the list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) tuples back to the original (float, float, float, float, QPen, QBrush) list

        Inverse operation of .get_serialized_data(...).

        Usage:
            seralized_tuples_data = RectangleRenderTupleHelpers.get_serialized_data(tuples_data)
            tuples_data = RectangleRenderTupleHelpers.get_deserialized_data(seralized_tuples_data)

            deserialized_df = get_deserialized_data(serialized_df)
            deserialized_df
        """ 
        serialized_df['pen'] = [pg.mkPen(a_pen.color, width=a_pen.width) for a_pen in serialized_df['pen']] # gets the RgbF values of the QColor returned from the QPen a_pen
        serialized_df['brush'] = [pg.mkBrush(a_brush.color) for a_brush in serialized_df['brush']] # gets the RgbF values of the QColor returned from the QBrush a_brush
        return serialized_df



    # ==================================================================================================================== #
    # classmethods                                                                                                         #
    # ==================================================================================================================== #
        
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
        try:
            # Start by assuming it's an Epoch object, and try to convert it to a dataframe
            raw_df = epochs.to_dataframe()
            active_df = pd.DataFrame({'t_start':raw_df.start.copy(), 't_duration':raw_df.duration.copy()}) # still will need columns ['series_vertical_offset', 'series_height', 'pen', 'brush'] added later
            # RESOLVED: is this 'raw_df.durations.copy()' instead of 'raw_df.duration.copy()' a bug? No, this is right for the result of Epoch.to_dataframe()
        except AttributeError:
            # if it's not an Epoch, assume it's a dataframe
            active_df = epochs.copy()
            # TODO: need to rename the columns?
            # active_df = pd.DataFrame({'t_start':raw_df.start.copy(), 't_duration':raw_df.duration.copy()}) # still will need columns ['series_vertical_offset', 'series_height', 'pen', 'brush'] added later
        except Exception as e:
            raise e
        
        active_df = cls.add_missing_reciprocal_columns_if_needed(active_df)
        active_df = dataframe_vis_columns_function(active_df)
        return cls(active_df, datasource_name=datasource_name)
    
    @QtCore.pyqtSlot(float, float)
    def get_updated_data_window(self, new_start, new_end):
        """ called to get the data that should be displayed for the window starting at new_start and ending at new_end """
        # return self.df[self.df[self.time_column_names].between(new_start, new_end)]
        # self.df['t_start', 't_duration', 't_end']    
        return self.df[self.df['t_start', 't_end'].between(new_start, new_end)]
    

