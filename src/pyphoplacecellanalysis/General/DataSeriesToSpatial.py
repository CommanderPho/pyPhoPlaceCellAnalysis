import numpy as np
import pandas as pd


class DataSeriesToSpatial:
    """ Helper functions for building the mapping from events to (X,Y) or (X,Y,Z):
    
    """
    
    @classmethod
    def build_data_series_range(cls, num_data_series, side_bin_margins = 0.0, center_mode='zero_centered', bin_position_mode='bin_center', enable_debug_print=True):
        """ Returns the centers of the bins 
        Useful for generating the position data for the axis that represents the number of independent data series, such as neuron_ids

        num_data_series: the number of dataseries to be displayed
        side_bin_margins: space to sides of the first and last cell on the y-axis
        center_mode: either 'starting_at_zero' or 'zero_centered'
        bin_position_mode: whether the 'left_edges', 'bin_center', or 'right_edges' of each bin is returned.

        Usage:
            curr_num_dataseries = len(curr_active_pipeline.sess.spikes_df.spikes.neuron_ids)
            y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = 1.0)
            # y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='left_edges')
            # y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='bin_center')
            # y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='left_edges')

        """
        half_side_bin_margin = side_bin_margins / 2.0 # size of a single margin
        n_half_data_series = np.ceil(float(num_data_series)/2.0)
        n_full_series_grid = 2.0 * n_half_data_series # could be one more than num_data_series
        if enable_debug_print:
            print(f'build_data_series_range(num_data_series: {num_data_series}, side_bin_margins: {side_bin_margins}, (center_mode: {center_mode}, bin_position_mode: {bin_position_mode})):\n n_half_data_series: {n_half_data_series}, n_full_series_grid: {n_full_series_grid}')
        # full_series_axis_range = side_bin_margins + n_full_series_grid

        bin_relative_offset = 0.0
        if bin_position_mode == 'left_edges':
            bin_relative_offset = 0.0
        elif bin_position_mode == 'bin_center':
            bin_relative_offset = 0.5
        elif bin_position_mode == 'right_edges':
            bin_relative_offset = 1.0
        else:
            raise

        # Whether the whole series starts at zero, is centered around zero, etc.
        if center_mode == 'zero_centered':
            # zero_centered:
            return np.linspace(-n_half_data_series, n_half_data_series, num=num_data_series, endpoint=False) + bin_relative_offset + half_side_bin_margin # add half_side_bin_margin so they're centered
        elif center_mode == 'starting_at_zero':
            # starting_at_zero:
            return np.linspace(0, n_full_series_grid, num=num_data_series, endpoint=False) + bin_relative_offset + half_side_bin_margin # add half_side_bin_margin so they're centered
        else:
            raise