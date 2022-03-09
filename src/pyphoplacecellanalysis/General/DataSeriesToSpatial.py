import numpy as np
import pandas as pd


class DataSeriesToSpatial:
    """ Helper functions for building the mapping from events to (X,Y) or (X,Y,Z):
    
    Two of the axes are arbitrarily defined, but fixed lengths:
    
    temporal_axis:  the mapping of event times to spatial position
        fixed length determined by (temporal_zoom_factor * render_window_duration)
    series_identity_axis: the mapping of each series (such as each neuron_id) to spatial position
        fixed length currently hardcoded to be (1.0 * n_cells) + side_bin_margins
        
        
    
    
    """
    
    @classmethod
    def build_series_identity_axis(cls, num_data_series, side_bin_margins = 0.0, center_mode='zero_centered', bin_position_mode='bin_center', enable_debug_print=False):
        """ Returns the centers of the bins 
        Useful for generating the position data for the axis that represents the number of independent data series, such as neuron_ids

        num_data_series: the number of dataseries to be displayed
        side_bin_margins: space to sides of the first and last cell on the y-axis
        center_mode: either 'starting_at_zero' or 'zero_centered'
        bin_position_mode: whether the 'left_edges', 'bin_center', or 'right_edges' of each bin is returned.

        Usage:
            curr_num_dataseries = len(curr_active_pipeline.sess.spikes_df.spikes.neuron_ids)
            y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = 1.0)
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='left_edges')
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='bin_center')
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='left_edges')

        """
        half_side_bin_margin = side_bin_margins / 2.0 # size of a single margin
        n_half_data_series = np.ceil(float(num_data_series)/2.0)
        n_full_series_grid = 2.0 * n_half_data_series # could be one more than num_data_series
        if enable_debug_print:
            print(f'build_series_identity_axis(num_data_series: {num_data_series}, side_bin_margins: {side_bin_margins}, (center_mode: {center_mode}, bin_position_mode: {bin_position_mode})):\n n_half_data_series: {n_half_data_series}, n_full_series_grid: {n_full_series_grid}')
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
        
    @classmethod
    def build_temporal_axis(cls, num_data_series, side_bin_margins = 0.0, center_mode='zero_centered', bin_position_mode='bin_center', enable_debug_print=False):
        """ Returns the centers of the bins 
        Useful for generating the position data for the axis that represents the number of independent data series, such as neuron_ids

        num_data_series: the number of dataseries to be displayed
        side_bin_margins: space to sides of the first and last cell on the y-axis
        center_mode: either 'starting_at_zero' or 'zero_centered'
        bin_position_mode: whether the 'left_edges', 'bin_center', or 'right_edges' of each bin is returned.

        Usage:
            curr_num_dataseries = len(curr_active_pipeline.sess.spikes_df.spikes.neuron_ids)
            y = DataSeriesToSpatial.build_data_series_range(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = 1.0)
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='zero_centered', bin_position_mode='left_edges')
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='bin_center')
            # y = DataSeriesToSpatial.build_series_identity_axis(curr_num_dataseries, center_mode='starting_at_zero', bin_position_mode='left_edges')

        """
        half_side_bin_margin = side_bin_margins / 2.0 # size of a single margin
        n_half_data_series = np.ceil(float(num_data_series)/2.0)
        n_full_series_grid = 2.0 * n_half_data_series # could be one more than num_data_series
        if enable_debug_print:
            print(f'build_series_identity_axis(num_data_series: {num_data_series}, side_bin_margins: {side_bin_margins}, (center_mode: {center_mode}, bin_position_mode: {bin_position_mode})):\n n_half_data_series: {n_half_data_series}, n_full_series_grid: {n_full_series_grid}')
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
        
  
    

class SpikesDataframeMixin(object):
    """docstring for SpikesDataframeMixin.
    
    Requires:
        self._temporal_zoom_factor: float
     """
     
    @property
    def temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_zoom_factor * self.render_window_duration
    @property
    def half_temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_axis_length / 2.0
    
    
     ######  Get/Set Properties ######:
    @property
    def temporal_zoom_factor(self):
        """The time dilation factor that maps spikes in the current window to x-positions along the time axis multiplicatively.
            Increasing this factor will result in a more spatially expanded time axis while leaving the visible window unchanged.
        """
        return self._temporal_zoom_factor
    @temporal_zoom_factor.setter
    def temporal_zoom_factor(self, value):
        self._temporal_zoom_factor = value
        
        
        
        
    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(SpikesDataframeMixin, self).__init__(*args, **kwargs)
        self.params = VisualizationParameters('')
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        self.params.spike_start_z = -10.0
        # self.spike_end_z = 0.1
        self.params.spike_end_z = -6.0
        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self._temporal_zoom_factor = 40.0 / float(self.render_window_duration)      


    