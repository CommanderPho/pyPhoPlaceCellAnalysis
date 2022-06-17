""" Render2DEventRectanglesHelper and Render2DEventRectanglesMixin

"""
from copy import deepcopy
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from indexed import IndexedOrderedDict

from pyphocorehelpers.print_helpers import print_dataframe_memory_usage

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem

from lazy_property import LazyProperty, LazyWritableProperty


class Render2DEventRectanglesHelper:
    """ Static helper that adds interval/epoch rectangles to 2D raster plots
 
    """
    
    
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
            curr_pyburst_interval_df.loc[:, 'visualization_series_y_location'] = [y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in curr_pyburst_interval_df['fragile_linear_neuron_IDX'].to_numpy()]
        #     # add the 'visualization_raster_y_location' column:
        #     curr_pyburst_interval_df.loc[:, 'visualization_series_y_location'] = [y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in curr_pyburst_interval_df['fragile_linear_neuron_IDX'].to_numpy()]
        #     # add the 'visualization_raster_y_location' column:
        #     curr_pyburst_interval_df.loc[:, 'visualization_series_y_location'] = [y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in curr_pyburst_interval_df['fragile_linear_neuron_IDX'].to_numpy()]

            ## hierarchical offset: offset increases slightly (up to a max percentage of the fixed_track_height, specified by `hierarchical_level_max_offset_height_portion`) per level
            hierarchical_level_max_offset_height_portion = 0.5 # offset by at most 20% of the fixed_series_height across all levels
            num_levels = len(included_burst_levels)
            offset_step_per_level = (fixed_series_height*hierarchical_level_max_offset_height_portion)/float(num_levels) # get the offset step per level
            # Optionally add the hierarchical offsets to position the different levels of bursts at different heights
            curr_pyburst_interval_df.loc[:, 'visualization_series_y_location'] = curr_pyburst_interval_df['visualization_series_y_location'] + ((curr_pyburst_interval_df['burst_level']-1.0)*offset_step_per_level) - 0.5
        
            # add the 'visualization_series_heights' column:
            curr_pyburst_interval_df.loc[:, 'visualization_series_height'] = fixed_series_height # the height is the same for all series

            # add the 'brush_color' column:
            curr_pyburst_interval_df.loc[:, 'pen_color_hex'] = curr_color_hex_pen
            curr_pyburst_interval_df.loc[:, 'brush_color_hex'] = curr_color_hex_brush

            filtered_burst_interval_dict[a_cell_id] = curr_pyburst_interval_df
            
        return filtered_burst_interval_dict
            
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
            curr_IntervalRectsItem_interval_pairs = list(zip(curr_pyburst_interval_df.t_start, curr_pyburst_interval_df.visualization_series_y_location, curr_pyburst_interval_df.t_duration, curr_pyburst_interval_df.visualization_series_height, curr_series_pens, curr_series_brushes))
            
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
        main_plot_widget = active_2d_plot.ui.main_plot_widget # PlotItem
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
        main_plot_widget = active_2d_plot.ui.main_plot_widget # PlotItem
        main_plot_widget.removeItem(active_interval_rects_item)
        active_interval_rects_item = None
        
        
        

    # ####################################################
    # ### Render2DEventRectanglesMixin: the mixin version that uses the static helper
    # class Render2DEventRectanglesMixin:
    #     """ A Mixin that uses Render2DEventRectanglesHelper to render 2D event rectangles on conforming classes
        
    #     TODO: This is not really a mixin, need to figure out how I want these used.
            
    #     Requires:
    #         active_burst_intervals
        
    #     """
    #     @property
    #     def render_event_rectangles_plots(self):
    #         """The render_event_rectangles_plots property."""
    #         return self._render_event_rectangles_plots
    #     @render_event_rectangles_plots.setter
    #     def render_event_rectangles_plots(self, value):
    #         self._render_event_rectangles_plots = value
        
        
    #     def _get_series_offsets(self):
    #         """ builds the y-axis offsets for each data series (for example each neuron with its row of spikes)
            
    #             For this neuron_IDX example - it gives a list of the y-positions of the bottom edge of each neuron's row.
    #         """
    #         # from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
    #         # num_cells = spike_raster_window.spike_raster_plt_2d.n_cells
    #         # center_mode = spike_raster_window.spike_raster_plt_2d.params.center_mode
    #         # side_bin_margins = spike_raster_window.spike_raster_plt_2d.params.side_bin_margins
    #         # series_offsets = DataSeriesToSpatial.build_series_identity_axis(num_cells, center_mode=center_mode, bin_position_mode='left_edges', side_bin_margins = side_bin_margins)
    #         # series_offsets_lower = DataSeriesToSpatial.build_series_identity_axis(num_cells, center_mode=center_mode, bin_position_mode='left_edges', side_bin_margins = side_bin_margins) / num_cells
    #         # series_offsets_upper = DataSeriesToSpatial.build_series_identity_axis(num_cells, center_mode=center_mode, bin_position_mode='right_edges', side_bin_margins = side_bin_margins) / num_cells
    #         # num_series = len(series_offsets)
    #         # fixed_series_height = (series_offsets[1]-series_offsets[0])
    #         # # series_heights = num_series*[fixed_series_height] # the height is the same for all series

    #         # series_heights = series_offsets_upper - series_offsets_lower
    #         # series_fragile_linear_neuron_IDX_map = dict(zip(spike_raster_window.spike_raster_plt_2d.fragile_linear_neuron_IDXs, series_offsets))
    #         # series_heights


    #         ## Directly from pre-computed y, lower_y, upper_y values:
    #         series_offsets = spike_raster_window.spike_raster_plt_2d.y
    #         series_offsets_lower = spike_raster_window.spike_raster_plt_2d.lower_y
    #         series_offsets_upper = spike_raster_window.spike_raster_plt_2d.upper_y
    #         num_series = len(series_offsets)
    #         # series_heights = series_offsets_upper - series_offsets_lower # this does not work
    #         fixed_series_height = (series_offsets[1]-series_offsets[0])
    #         series_heights = num_series*[fixed_series_height] # the height is the same for all series
    #         series_fragile_linear_neuron_IDX_map = spike_raster_window.spike_raster_plt_2d.y_fragile_linear_neuron_IDX_map

    #         ## Fixed calculations to get series starts not centers:
    #         # Assume we have series_offsets which are centers (e.g. # series_offsets: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5 36.5 37.5 38.5 39.5])
    #         series_start_offsets = series_offsets - (np.array(series_heights)/2.0) # series_start_offsets: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
    #         print(f'series_start_offsets: {series_start_offsets}')

    #         print(f'series_offsets: {series_offsets}')
    #         print(f'series_offsets_lower: {series_offsets_lower}')
    #         print(f'series_offsets_upper: {series_offsets_upper}')
    #         print(f'series_heights: {series_heights}')
    #         print(f'series_fragile_linear_neuron_IDX_map: {series_fragile_linear_neuron_IDX_map}')

    #         # series_offsets: [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5 36.5 37.5 38.5 39.5]
    #         # series_offsets_lower: [0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975]
    #         # series_offsets_upper: [0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975 1]
    #         # series_heights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #         # series_fragile_linear_neuron_IDX_map: {0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: 5.5, 6: 6.5, 7: 7.5, 8: 8.5, 9: 9.5, 10: 10.5, 11: 11.5, 12: 12.5, 13: 13.5, 14: 14.5, 15: 15.5, 16: 16.5, 17: 17.5, 18: 18.5, 19: 19.5, 20: 20.5, 21: 21.5, 22: 22.5, 23: 23.5, 24: 24.5, 25: 25.5, 26: 26.5, 27: 27.5, 28: 28.5, 29: 29.5, 30: 30.5, 31: 31.5, 32: 32.5, 33: 33.5, 34: 34.5, 35: 35.5, 36: 36.5, 37: 37.5, 38: 38.5, 39: 39.5}

    #     ##################################################
    #     ## MAIN METHODS
    #     ##################################################
            
    #     def add_event_rectangles(self, active_2d_plot, active_burst_intervals):
    #         return Render2DEventRectanglesHelper.add_event_rectangles(active_2d_plot, active_burst_intervals)
            
    #     def remove_event_rectangles(self):
    #         ## Remove the active_interval_rects_item:
    #         main_plot_widget.removeItem(active_interval_rects_item)
    #         active_interval_rects_item = None

