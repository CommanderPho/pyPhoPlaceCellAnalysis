from typing import List, Dict, Optional
from attrs import define, field, Factory
import numpy as np
import pandas as pd
from copy import deepcopy

from neuropy.core.laps import Laps
from neuropy.core.epoch import Epoch
from neuropy.core.session.dataSession import DataSession


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager # for getting colors of session epochs


""" 
A general epochs_dataframe_formatter takes a dataframe and adds the required columns

"""

@define(slots=False)
class General2DRenderTimeEpochs(object):
    """docstring for General2DRenderTimeEpochs."""
    default_datasource_name: str = 'GeneralEpochs' # class variable
    # default_datasource_name: str = field(default='GeneralEpochs')
    
    _required_interval_visualization_columns = ['t_start', 't_duration', 'series_vertical_offset', 'series_height', 'pen', 'brush']
    
    @classmethod
    def _update_df_visualization_columns(cls, active_df, y_location=None, height=None, pen_color=None, brush_color=None, **kwargs):
        """ updates the columns of the provided active_df given the values specified. If values aren't provided, they aren't changed. 
        
        active_df['series_vertical_offset', 'series_height', 'pen', 'brush']
        
        """        
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
        """ allows specifying a custom formatter function. Called from cls.add_render_time_epochs(...)
        """
        custom_epochs_df_formatter = kwargs.pop('epochs_dataframe_formatter', None)
        
        if custom_epochs_df_formatter is None:
            active_epochs_df_formatter = cls.build_epochs_dataframe_formatter(**kwargs)
        else:
            print(f'overriding default epochs_df_formatter...')
            active_epochs_df_formatter = custom_epochs_df_formatter(cls, **kwargs)
        
        if isinstance(active_epochs_obj, Epoch):
            general_epochs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_epochs_obj, active_epochs_df_formatter, datasource_name='intervals_datasource_from_general_Epochs_obj')
            
        elif isinstance(active_epochs_obj, pd.DataFrame):
            ## ensure the time columns are named correctly (['t_start', 't_duration', 't_end'])
            if not np.isin(cls._required_interval_visualization_columns, active_epochs_obj.columns).all():
                ## check if it's missing any viz columns:
                general_epochs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_epochs_obj, active_epochs_df_formatter, datasource_name='intervals_datasource_from_general_Epochs_obj')
            else:
                ## use exactly with existing viz columns (all the dataframe's columns must be named exactly correctly):
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
    def is_render_time_epochs_enabled(cls, curr_sess, **kwargs) -> bool:
        """ takes the exact same arguments as `add_render_time_epochs(...) but returns True if the call would be valid and False otherwise. """
        try:
            if isinstance(curr_sess, DataSession):
                active_Epochs = curr_sess.epochs # <Epoch> object
            elif isinstance(curr_sess, (Epoch, pd.DataFrame, tuple)):
                active_Epochs = curr_sess  # <Epoch> object passed directly
            else:
                return False
            return (active_Epochs is not None)
        except BaseException:
            return False
        
        
    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs): # , curr_pipeline=None
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        # if curr_pipeline is not None:
        #     assert isinstance(curr_sess, NeuropyPipeline)
            
        # if isinstance(curr_sess, NeuropyPipeline):
        #     curr_pipeline = curr_sess
        #     sess = curr_pipeline.sess
        #     active_Epochs = sess.epochs # <Epoch> object
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
        long_short_display_config_manager = LongShortDisplayConfigManager()
        long_epoch_config = long_short_display_config_manager.long_epoch_config #.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config #.as_pyqtgraph_kwargs()

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
            # pen_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            # brush_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            
            pen_color = [pg.mkColor(long_epoch_config.pen.color()), pg.mkColor(short_epoch_config.pen.color())]
            brush_color = [pg.mkColor(long_epoch_config.brush.color()), pg.mkColor(short_epoch_config.brush.color())]
            
            for a_pen_color in pen_color:
                a_pen_color.setAlphaF(0.8)

            for a_brush_color in brush_color:
                a_brush_color.setAlphaF(0.5)

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
    def is_render_time_epochs_enabled(cls, curr_sess, **kwargs) -> bool:
        """ takes the exact same arguments as `add_render_time_epochs(...) but returns True if the call would be valid and False otherwise. """
        try:
            if isinstance(curr_sess, DataSession):
                active_Epochs = curr_sess.laps.as_epoch_obj() # <Epoch> object
            elif isinstance(curr_sess, Laps):
                active_Epochs = curr_sess.as_epoch_obj()
            elif isinstance(curr_sess, Epoch):
                active_Epochs = curr_sess
            else:
                return False
            return (active_Epochs is not None)
        except BaseException:
            return False
        
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
            height = 1.9
            pen_color = pg.mkColor('orange')
            pen_color.setAlphaF(0.8)
            brush_color = pg.mkColor('orange')
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
            active_Epochs = curr_sess.epochs # <Epoch> object
        elif isinstance(curr_sess, Epoch):
            active_Epochs = curr_sess  # <Epoch> object passed directly
        elif isinstance(curr_sess, pd.DataFrame):
            # tries 'flat_replay_idx' column if it exists, otherwise tries 'label' column
            replay_idx_column_name = 'flat_replay_idx'
            if replay_idx_column_name not in curr_sess.columns:
                # replay_idx_column_name = 'label' # try "label" instead
                assert 'label' in curr_sess.columns
                curr_sess[replay_idx_column_name] = curr_sess['label'].copy() ## make the desired column
            # END if replay_idx_column_name not in curr_sess....

            # active_Epochs = (curr_sess['start'].to_numpy(), curr_sess['duration'].to_numpy(), curr_sess[replay_idx_column_name].to_numpy()) ## ... make stupid tuples if it's provided a dataframe :[            
            active_Epochs = curr_sess # pass the dataframe directly
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
        



##########################################
## Spike Burst Intervals - Requires Pipeline
class SpikeBurstIntervals_2DRenderTimeEpochs(General2DRenderTimeEpochs):
    default_datasource_name = 'SpikeBursts'
    
    ##################################################
    ## Spike Events METHODS
    ##################################################
    
    @classmethod
    def _post_process_detected_burst_interval_dict(cls, active_burst_interval_dict, cell_id_to_fragile_linear_neuron_IDX_map, neuron_colors_hex, y_fragile_linear_neuron_IDX_map, included_burst_levels=[1], fixed_series_height=1.0, debug_print=False):
        """ Filters the dict of dataframes for only the relevent intervals and adds computed columns
        Inputs:
            active_burst_interval_dict: a dict of pd.DataFrames with one entry per cell
            included_burst_levels: a list of burst hierarchy levels to include
            fixed_series_height: float - the height common to all series
            
            Old spike_raster_window.spike_raster_plt_2d.* properties:
                cell_id_to_fragile_linear_neuron_IDX_map - a dict with keys of cell_ids and values of fragile_neuron_IDXs
                params.neuron_colors_hex - a dict with keys of fragile_neuron_IDXs
                y_fragile_linear_neuron_IDX_map - a dict with keys of fragile_neuron_IDXs and values of the correct y-offsets
                
        Requires:
            spike_raster_window.spike_raster_plt_2d
            
            -- OR --
            
            cell_id_to_fragile_linear_neuron_IDX_map
            neuron_colors_hex
            y_fragile_linear_neuron_IDX_map
            
            #- neuron_id_qcolors_map # not right now
            
        Usage:
            filtered_burst_intervals = _post_process_detected_burst_interval_dict(active_burst_intervals, included_burst_levels=(1))
            
            filtered_burst_intervals = _post_process_detected_burst_interval_dict(active_burst_intervals, 
                    spike_raster_window.spike_raster_plt_2d.cell_id_to_fragile_linear_neuron_IDX_map,
                    cell_id_to_fragile_linear_neuron_IDX_map.params.neuron_colors_hex,
                    spike_raster_window.spike_raster_plt_2d.y_fragile_linear_neuron_IDX_map,
                    included_burst_levels=(1)
                )
            
        """
        filtered_burst_interval_dict = deepcopy(active_burst_interval_dict)
        for (a_cell_id, curr_pyburst_interval_df) in filtered_burst_interval_dict.items():
            # loop through the cell_ids          
            # Filter to only desired-order bursts:
            curr_pyburst_interval_df = curr_pyburst_interval_df[np.isin(curr_pyburst_interval_df['burst_level'], included_burst_levels)].copy()
            a_fragile_neuron_IDX = cell_id_to_fragile_linear_neuron_IDX_map[a_cell_id]
            if debug_print:
                print(f'aclu: {a_cell_id}, a_fragile_neuron_IDX: {a_fragile_neuron_IDX}')

            # the color is the same for all within the series
            curr_color_hex = neuron_colors_hex[a_fragile_neuron_IDX]
            curr_color = pg.mkColor(curr_color_hex)
            # curr_color = pg.mkColor(neuron_id_qcolors_map[a_cell_id]) # Other version that uses neuron_id_qcolors_map
            # curr_color = neuron_id_qcolors_map[a_cell_id] # copies the color so it's independent object
            curr_color.setAlphaF(1.0)

            curr_color_pen = pg.mkColor(curr_color)
            # curr_color_pen.setAlphaF(0.9) # Great for single non-overlapping intervals
            curr_color_pen.setAlphaF(0.5)
            curr_color_hex_pen = pg.colorStr(curr_color_pen)
            
            curr_color_brush = pg.mkColor(curr_color)
            # curr_color_brush.setAlphaF(0.2) # Great for single non-overlapping intervals
            curr_color_brush.setAlphaF(0.05)
            curr_color_hex_brush = pg.colorStr(curr_color_brush)
            
            # add the 'aclu' column:
            curr_pyburst_interval_df.loc[:, 'aclu'] = a_cell_id
            # add the 'fragile_linear_neuron_IDX' column:
            curr_pyburst_interval_df.loc[:, 'fragile_linear_neuron_IDX'] = a_fragile_neuron_IDX
            # add the 'visualization_raster_y_location' column:
            curr_pyburst_interval_df.loc[:, 'series_vertical_offset'] = [y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in curr_pyburst_interval_df['fragile_linear_neuron_IDX'].to_numpy()]

            ## hierarchical offset: offset increases slightly (up to a max percentage of the fixed_track_height, specified by `hierarchical_level_max_offset_height_portion`) per level
            hierarchical_level_max_offset_height_portion = 0.5 # offset by at most 20% of the fixed_series_height across all levels
            num_levels = len(included_burst_levels)
            offset_step_per_level = (fixed_series_height*hierarchical_level_max_offset_height_portion)/float(num_levels) # get the offset step per level
            # Optionally add the hierarchical offsets to position the different levels of bursts at different heights
            curr_pyburst_interval_df.loc[:, 'series_vertical_offset'] = curr_pyburst_interval_df['series_vertical_offset'] + ((curr_pyburst_interval_df['burst_level']-1.0)*offset_step_per_level) - 0.5
        
            # add the 'series_height' column:
            curr_pyburst_interval_df.loc[:, 'series_height'] = fixed_series_height # the height is the same for all series

            # # add the 'brush_color' column:
            # curr_pyburst_interval_df.loc[:, 'pen_color_hex'] = curr_color_hex_pen
            # curr_pyburst_interval_df.loc[:, 'brush_color_hex'] = curr_color_hex_brush
            
            # add the 'brush_color' column:
            curr_pyburst_interval_df.loc[:, 'pen_color_hex'] = curr_color_hex_pen
            curr_pyburst_interval_df.loc[:, 'brush_color_hex'] = curr_color_hex_brush
            # Set the actual brushes here too:
            curr_pyburst_interval_df.loc[:, 'pen'] = [pg.mkPen(a_pen_hex_color) for a_pen_hex_color in curr_pyburst_interval_df.pen_color_hex]
            curr_pyburst_interval_df.loc[:, 'brush'] = [pg.mkBrush(a_brush_color_hex) for a_brush_color_hex in curr_pyburst_interval_df.brush_color_hex]
            
            filtered_burst_interval_dict[a_cell_id] = curr_pyburst_interval_df
            
        return filtered_burst_interval_dict
            
            
    @classmethod
    def build_interval_datasource_from_active_burst_intervals(cls, active_burst_intervals, datasource_name='active_burst_intervals_datasource'):
        """ Builds an appropriate IntervalsDatasource from a dict of dataframes for each cell 
        Input:
            active_burst_intervals: IntervalsDatasource
        Returns:
            IntervalsDatasource
        """        
        df = pd.concat(active_burst_intervals.values())
        interval_datasource = IntervalsDatasource(df, datasource_name=datasource_name)
        return interval_datasource # IntervalsDatasource
            
    @staticmethod
    def build_interval_rects_data(active_burst_intervals, included_burst_levels=[1]):
        """ Build the data to pass to IntervalRectsItem that enables rendering the burst rects:
            [X] TODO: Set the brush color to the neuron's color for each row of rectangles
            [X] TODO: Use the self.y property of Spike2DRaster to set the offset value (first entry in the tuple)
            [X] TODO: Add to existing 2D plots in Spike2DRaster
            [X] TODO: Enable scrolling in Spike2DRaster's scrollable plot
            
            
            Depends on correctly computed column information in the passed list of dataframes.
            
        """
        data = [] # data specifically for IntervalRectsItem
        for (i, a_cell_id) in enumerate(active_burst_intervals.keys()):
            # loop through the cell_ids
            curr_pyburst_interval_df = active_burst_intervals[a_cell_id]

            # Filter to only desired-order bursts:
            curr_pyburst_interval_df = curr_pyburst_interval_df[np.isin(curr_pyburst_interval_df['burst_level'], included_burst_levels)]
            # curr_series_num_items = len(curr_pyburst_interval_df.t_start)
            
            # Convert vectors to tuples of rect values:
            # curr_series_offset = series_offsets[i]
            # curr_series_offset = i
            
            curr_series_pens = [pg.mkPen(a_pen_hex_color) for a_pen_hex_color in curr_pyburst_interval_df.pen_color_hex]
            curr_series_brushes = [pg.mkBrush(a_brush_color_hex) for a_brush_color_hex in curr_pyburst_interval_df.brush_color_hex]
            ## build the output tuple list: fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).        
            curr_IntervalRectsItem_interval_pairs = list(zip(curr_pyburst_interval_df.t_start, curr_pyburst_interval_df.series_vertical_offset, curr_pyburst_interval_df.t_duration, curr_pyburst_interval_df.series_height, curr_series_pens, curr_series_brushes))
            
            data = data + curr_IntervalRectsItem_interval_pairs
            
        return data
    
    @classmethod
    def build_burst_event_rectangle_datasource(cls, active_2d_plot, active_burst_intervals, datasource_name='active_burst_intervals_datasource', included_burst_levels=[1,2,3,4], debug_print=False) -> IntervalsDatasource:
        """ 
        Inputs:
            active_2d_plot: e.g. spike_raster_window.spike_raster_plt_2d
            active_burst_intervals
            
        Returns:
            IntervalsDatasource
        
        TODO:
            don't need ['interval_pair']
        
        Usage:            
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper

            output_display_items = Render2DEventRectanglesHelper.add_event_rectangles(spike_raster_window.spike_raster_plt_2d, active_burst_intervals) # {'interval_rects_item': active_interval_rects_item}
            active_interval_rects_item = output_display_items['interval_rects_item']

        """        
        # Builds the filtered burst intervals:
        filtered_burst_intervals = cls._post_process_detected_burst_interval_dict(active_burst_intervals, 
                    active_2d_plot.cell_id_to_fragile_linear_neuron_IDX_map,
                    active_2d_plot.params.neuron_colors_hex,
                    active_2d_plot.y_fragile_linear_neuron_IDX_map,
                    included_burst_levels=included_burst_levels
                )
        return cls.build_interval_datasource_from_active_burst_intervals(active_burst_intervals=filtered_burst_intervals, datasource_name=datasource_name)
        


    ##################################################
    ## Main METHODS
    ##################################################
    @classmethod
    def build_epochs_dataframe_formatter(cls, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
            ## parameters:
            raise NotImplementedError
            return active_df
        return _add_interval_dataframe_visualization_columns_general_epoch
    

    @classmethod
    def is_render_time_epochs_enabled(cls, curr_sess, **kwargs) -> bool:
        """ takes the exact same arguments as `add_render_time_epochs(...) but returns True if the call would be valid and False otherwise. """
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # for advanced add_render_time_epochs
        if isinstance(curr_sess, NeuropyPipeline):
            curr_active_pipeline = curr_sess
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            active_config_name = kwargs.pop('active_config_name', global_epoch_name)

            try:
                active_burst_intervals = curr_active_pipeline.computation_results[active_config_name].computed_data['burst_detection']['burst_intervals'] # this works
                return (active_burst_intervals is not None)
                    
            except (KeyError, AttributeError, ValueError):
                return False
                    
        else:
            return False
        


    @classmethod
    def add_render_time_epochs(cls, curr_sess, destination_plot, **kwargs):
        """ directly-called method 
        destination_plot should implement add_rendered_intervals
        """
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # for advanced add_render_time_epochs

        if isinstance(curr_sess, NeuropyPipeline):
            curr_active_pipeline = curr_sess
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            active_config_name = kwargs.pop('active_config_name', global_epoch_name)
            included_burst_levels = kwargs.pop('included_burst_levels', [1])

            active_burst_intervals = curr_active_pipeline.computation_results[active_config_name].computed_data['burst_detection']['burst_intervals'] # this works

            # Builds the filtered burst intervals:
            # filtered_burst_intervals = cls._post_process_detected_burst_interval_dict(active_burst_intervals, 
            #             destination_plot.cell_id_to_fragile_linear_neuron_IDX_map,
            #             destination_plot.params.neuron_colors_hex,
            #             destination_plot.y_fragile_linear_neuron_IDX_map,
            #             included_burst_levels=included_burst_levels
            #         )
                        
            # Builds the render rectangles:
            # data = cls.build_interval_rects_data(filtered_burst_intervals, included_burst_levels=[1,2,3,4]) # all order bursts
            # if debug_print:
            #     print(f'np.shape(data): {np.shape(data)}') # (25097, 3)
            # np.shape(data): (5412, 6)
            
            ## build the mesh:
            # active_interval_rects_item = IntervalRectsItem(data)

            # interval_datasource = cls.build_render_time_epochs_datasource(active_epochs_obj=active_Epochs, **kwargs)
            interval_datasource = cls.build_burst_event_rectangle_datasource(destination_plot, active_burst_intervals=active_burst_intervals, included_burst_levels=included_burst_levels)
            out_rects = destination_plot.add_rendered_intervals(interval_datasource, name=kwargs.setdefault('name', cls.default_datasource_name), debug_print=True)
            

        else:
            ## `SpikeBurstIntervals_2DRenderTimeEpochs` requires a curr_active_pipeline to be passed.
            raise NotImplementedError
        
        




def inline_mkColor(color, alpha=1.0):
    """ helps build a new QColor for a pen/brush in an inline (single-line) way. """
    out_color = pg.mkColor(color)
    out_color.setAlphaF(alpha)
    return out_color

""" HISTORICAL NOTE: Specific2DRenderTimeEpochsHelper has been removed in favor of a class-based approach """