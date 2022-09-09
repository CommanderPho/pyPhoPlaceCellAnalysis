""" Render2DEventRectanglesHelper and Render2DEventRectanglesMixin

"""
from copy import deepcopy
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from indexed import IndexedOrderedDict
# from lazy_property import LazyProperty, LazyWritableProperty

from neuropy.core import Epoch

# from pyphocorehelpers.print_helpers import print_dataframe_memory_usage

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem
from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource



class Render2DEventRectanglesHelper:
    """ Static helper that adds interval/epoch rectangles to 2D raster plots
 
        Also has the full implemention of Bursts (which are plotted as rectangles per-neuron, which hasn't been updated to the new EpochRenderingMixin format yet

    """
    
    ##################################################
    ## Common METHODS
    ##################################################
    _required_interval_visualization_columns = ['t_start', 't_duration', 'series_vertical_offset', 'series_height', 'pen', 'brush']
    
    @classmethod
    def _build_interval_tuple_list_from_dataframe(cls, df):
        """ build the tuple list required for rendering intervals: fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
        Inputs:
            df: a Pandas.DataFrame with the columns ['t_start', 't_duration', 'series_vertical_offset', 'series_height', 'pen', 'brush']
        Returns:
            a list of tuples with fields (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
        """    
        ## Validate that it has all required columns:
        assert np.isin(cls._required_interval_visualization_columns, df.columns).all(), f"dataframe is missing required columns:\n Required: {cls._required_interval_visualization_columns}, current: {df.columns} "
        return list(zip(df.t_start, df.series_vertical_offset, df.t_duration, df.series_height, df.pen, df.brush))
        
    @classmethod
    def build_IntervalRectsItem_from_epoch(cls, epochs: Epoch, dataframe_vis_columns_function, debug_print=False):
        """ Builds an appropriate IntervalRectsItem from any Epoch object and a function that is passed the converted dataframe and adds the visualization specific columns: ['series_vertical_offset', 'series_height', 'pen', 'brush']
        
        Input:
            epochs: Either a neuropy.core.Epoch object OR dataframe with the columns ['t_start', 't_duration']
            dataframe_vis_columns_function: callable that takes a pd.DataFrame that adds the remaining required columns to the dataframe if needed.
        
        Returns:
            IntervalRectsItem
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
        
        active_df = dataframe_vis_columns_function(active_df)
        
        ## build the output tuple list: fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
        curr_IntervalRectsItem_interval_tuples = cls._build_interval_tuple_list_from_dataframe(active_df)
        ## build the IntervalRectsItem
        return IntervalRectsItem(curr_IntervalRectsItem_interval_tuples)
    
    
    # MAIN METHOD to build datasource ____________________________________________________________________________________ #
    @classmethod
    def build_IntervalRectsItem_from_interval_datasource(cls, interval_datasource: IntervalsDatasource):
        """ Builds an appropriate IntervalRectsItem from any IntervalsDatasource object 
        Input:
            interval_datasource: IntervalsDatasource
        Returns:
            IntervalRectsItem
        """        
        active_df = interval_datasource.df
        ## build the output tuple list: fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
        curr_IntervalRectsItem_interval_tuples = cls._build_interval_tuple_list_from_dataframe(active_df)
        ## build the IntervalRectsItem
        return IntervalRectsItem(curr_IntervalRectsItem_interval_tuples)
    
    
    
    
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
    
    ## Debugging rectangles:
    @staticmethod
    def _simple_debugging_rects_data(series_start_offsets):
        """ Generates a simple set of test rectangles
        
        series_start_offsets
        
        """
        # Have series_offsets which are centers and series_start_offsets which are bottom edges:
        curr_border_color = pg.mkColor('r')
        curr_border_color.setAlphaF(0.8)
        
        curr_fill_color = pg.mkColor('w')
        curr_fill_color.setAlphaF(0.2)

        # build pen/brush from color
        curr_series_pen = pg.mkPen(curr_border_color)
        curr_series_brush = pg.mkBrush(curr_fill_color)
        # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
        #     (40.0, 0.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
        #     (41.0, 1.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
        #     (44.0, series_start_offsets[0], 4.0, 1.0, curr_series_pen, curr_series_brush),
        #     (45.0, series_start_offsets[-1], 4.0, 1.0, curr_series_pen, curr_series_brush),
        # ]
        data = []
        step_x_offset = 0.5
        for i in np.arange(len(series_start_offsets)):
            curr_x_pos = (40.0+(step_x_offset*float(i)))
            data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))
        return data

    # data = _simple_debugging_rects_data(series_start_offsets) # overwrites the data with 2 simple debugging rects
    
    ##################################################
    ## MAIN METHODS
    ##################################################
        
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
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper

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
        return Render2DEventRectanglesHelper.build_interval_datasource_from_active_burst_intervals(active_burst_intervals=filtered_burst_intervals, datasource_name=datasource_name)
        
        
    
        
    @classmethod
    def add_event_rectangles(cls, active_2d_plot, active_burst_intervals, included_burst_levels=[1], debug_print=False):
        """ 
        Inputs:
            active_2d_plot: e.g. spike_raster_window.spike_raster_plt_2d
            active_burst_intervals
        Usage:            
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper

            output_display_items = Render2DEventRectanglesHelper.add_event_rectangles(spike_raster_window.spike_raster_plt_2d, active_burst_intervals) # {'interval_rects_item': active_interval_rects_item}
            active_interval_rects_item = output_display_items['interval_rects_item']

        """
        # Build many-order bursts:
        # filtered_burst_intervals = Render2DEventRectanglesHelper._post_process_detected_burst_interval_dict(active_burst_intervals, included_burst_levels=[1,2,3,4])
        
        ## Gets the 2D plot from itself:
        # active_2d_plot = spike_raster_window.spike_raster_plt_2d
        
        # Builds the filtered burst intervals:
        filtered_burst_intervals = cls._post_process_detected_burst_interval_dict(active_burst_intervals, 
                    active_2d_plot.cell_id_to_fragile_linear_neuron_IDX_map,
                    active_2d_plot.params.neuron_colors_hex,
                    active_2d_plot.y_fragile_linear_neuron_IDX_map,
                    included_burst_levels=included_burst_levels
                )
                    
        # Builds the render rectangles:
        data = cls.build_interval_rects_data(filtered_burst_intervals, included_burst_levels=[1,2,3,4]) # all order bursts

        ## First order bursts only:
        # data = build_interval_rects_data(filtered_burst_intervals, included_burst_levels=[1]) # data_first_order_bursts
    
        if debug_print:
            print(f'np.shape(data): {np.shape(data)}') # (25097, 3)
        # np.shape(data): (5412, 6)
        
        ## build the mesh:
        active_interval_rects_item = IntervalRectsItem(data)

        ## Add the active_interval_rects_item to the main_plot_widget: 
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        main_plot_widget.addItem(active_interval_rects_item)
        # return the updated display items:
        return {'interval_rects_item': active_interval_rects_item}


    @classmethod
    def remove_event_rectangles(cls, active_2d_plot, active_interval_rects_item):
        """ Remove the active_interval_rects_item:
        Inputs:
            active_2d_plot: e.g. spike_raster_window.spike_raster_plt_2d
        Usage:
            ## Remove the rectangles:
            Render2DEventRectanglesHelper.remove_event_rectangles(spike_raster_window.spike_raster_plt_2d, active_interval_rects_item)


        """
        ## Remove the active_interval_rects_item:
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        main_plot_widget.removeItem(active_interval_rects_item)
        active_interval_rects_item = None
        
        
    @classmethod
    def add_separate_render_epoch_rects_plot_item(cls, active_2d_plot):
        """ Adds a separate independent plot for epoch time rects to the 2D plot above the others:
        
        Requires:
            active_2d_plot.ui.main_graphics_layout_widget <GraphicsLayoutWidget>
            
        """
        main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        
        ## Test separate epoch rect rendering plot that's linked with the main plot:
        epoch_rect_separate_plot = main_graphics_layout_widget.addPlot(row=0, col=0) # PlotItem
        epoch_rect_separate_plot.setObjectName('epoch_rect_separate_plot')

        # Setup axes bounds for the bottom windowed plot:
        epoch_rect_separate_plot.hideAxis('left')
        epoch_rect_separate_plot.hideAxis('bottom')

        # setup the epoch_rect_separate_plot to have a linked X-axis to the other scroll plot:
        epoch_rect_separate_plot.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)

        return epoch_rect_separate_plot



