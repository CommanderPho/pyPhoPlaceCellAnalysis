import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available

from pyphocorehelpers.print_helpers import format_seconds_human_readable # for build_minute_x_tick_labels(...)


class DataSeriesToSpatial:
    """ Helper functions for building the mapping from temporal events (t, v0, v1, ...) to (X,Y) or (X,Y,Z):
    
    Two of the axes are arbitrarily defined, but fixed lengths:
    
    temporal_axis:  the mapping of event times to spatial position
        fixed length determined by (temporal_zoom_factor * render_window_duration)
    series_identity_axis: the mapping of each series (such as each neuron_id) to spatial position
        fixed length currently hardcoded to be (1.0 * n_cells) + side_bin_margins
        
    
    """
    
    @classmethod
    def build_series_identity_axis(cls, num_data_series, side_bin_margins = 0.0, center_mode='zero_centered', bin_position_mode='bin_center', enable_debug_print=False):
        """ Useful for generating the position data for the axis that represents the number of independent data series, such as neuron_ids

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

        Returns:
            Returns the centers of the bins
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
    def temporal_to_spatial_map(cls, event_times, active_window_start_time, active_window_end_time, temporal_axis_spatial_length, center_mode='zero_centered', enable_debug_print=False):
        """ Generates the position data for the axis that represents time
        
        
        (t_start) -> (pos_x)
        
        Inputs:
            event_times: times to translate
            active_window_start_time: start of the active window
            active_window_end_time: end of the active window
            temporal_axis_spatial_length: the current spatial length of the temporal_axis determined by (temporal_zoom_factor * render_window_duration)

        Usage:
            
        """
         # Whether the whole series starts at zero, is centered around zero, etc.
        if center_mode == 'zero_centered':
            # zero_centered:
            half_temporal_axis_spatial_length = float(temporal_axis_spatial_length) / 2.0
            temporal_to_spatial_axis_start_end = (-half_temporal_axis_spatial_length, +half_temporal_axis_spatial_length)
        elif center_mode == 'starting_at_zero':
            # starting_at_zero:
            temporal_to_spatial_axis_start_end = (0.0, temporal_axis_spatial_length)
        else:
            raise
        
        # map the current event times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
        return np.interp(event_times, (active_window_start_time, active_window_end_time), temporal_to_spatial_axis_start_end)

    
    @classmethod
    def temporal_to_spatial_transform_computation(cls, epoch_start_times, epoch_durations, active_window_start_time, active_window_end_time, temporal_axis_spatial_length, center_mode='zero_centered', enable_debug_print=False):
        """ Generates the position data for the axis that represents time from a series of start_times and durations.
        
        (t_start, t_duration) -> (pos_x, width)

        Inputs:
            epoch_start_times: times to translate into positions
            epoch_durations: temporal durations to translate into widths
            active_window_start_time: start of the active window
            active_window_end_time: end of the active window
            temporal_axis_spatial_length: the current spatial length of the temporal_axis determined by (temporal_zoom_factor * render_window_duration)
            center_mode: either 'starting_at_zero' or 'zero_centered'
            
            
        Usage:
            DataSeriesToSpatial.temporal_to_spatial_transform_computation(epoch_start_times, epoch_durations, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
    
        """
        window_duration = active_window_end_time - active_window_start_time
        temporal_axis_spatial_scale_factor = temporal_axis_spatial_length / window_duration # recover the temporal scale factor to know how much times should be dilated by to get their position equivs.
        ## ALT: could also return the dilation and translation factors
        
        epoch_spatial_durations = epoch_durations * temporal_axis_spatial_scale_factor
        epoch_window_relative_start_times = epoch_start_times - active_window_start_time # shift so that it's t=0 is at the start of the window (TODO: should it be the center of the window)?
        # transforms in to positions:        
        epoch_window_relative_start_x_positions = (epoch_window_relative_start_times * temporal_axis_spatial_scale_factor)
        
         # Whether the whole series starts at zero, is centered around zero, etc.
        if center_mode == 'zero_centered':
            # zero_centered:
            half_temporal_axis_spatial_length = float(temporal_axis_spatial_length) / 2.0
            epoch_window_relative_start_x_positions = epoch_window_relative_start_x_positions + (-half_temporal_axis_spatial_length) # if zero_centered mode, we previously said that t=active_window_start_time occured at x=0, but in zero_centered mode it actually occurs at x=(-half_temporal_axis_spatial_length). To correct for this, we need to add back (+alf_temporal_axis_spatial_length) to move it to the true zero. 
            
        elif center_mode == 'starting_at_zero':
            # starting_at_zero:
            # temporal_to_spatial_axis_start_end = (0.0, temporal_axis_spatial_length)
            pass
        else:
            raise
            
        return epoch_window_relative_start_x_positions, epoch_spatial_durations


    @classmethod
    def build_minute_x_tick_labels(cls, spike_raster_plt, enable_debug_print=False):
        """ 
        Starts by finding the t-values (times in the global data time frame) that correspond to 60.0 second (1 minute) steps starting from the global window's earliest value.
            this is global_minute_t_ticks
        Then it transforms these global times into x-values using the usual global_minute_x_tick_positions = DataSeriesToSpatial.temporal_to_spatial_map(global_minute_t_ticks, ...) approach.
            this is global_minute_x_tick_positions
        Finally, it generates an appropraite time label string for each tick t-value to be displayed on the tick. Currently this is just HH:MM:SS.sss format.
            this is the returned list of (a_tick_x_pos, a_tick_label_str) pairs to be used as the vedo.Axes's xValuesAndLabels argument.
        
        should return:
            xValuesAndLabels: list of custom tick positions and labels [(pos1, label1), …]
            
        Example:
            from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial # for DataSeriesToSpatial.build_minute_x_tick_labels(...) function
            xValuesAndLabels = DataSeriesToSpatial.build_minute_x_tick_labels(spike_raster_plt_3d_vedo)
            print(f'xValuesAndLabels: {xValuesAndLabels}')

        """
        if enable_debug_print:
            print('build_minute_x_tick_labels(...):')
        global_start_t, global_end_t = spike_raster_plt.spikes_window.total_df_start_end_times
        global_total_data_duration = global_end_t - global_start_t
        if enable_debug_print:
            print(f'\t(global_start_t: {global_start_t}, global_end_t: {global_end_t}), global_total_data_duration: {global_total_data_duration} (seconds)')
        # find the maximum integer number of minutes that the global_total_data_duration can be divided into
        # global_total_data_duration_minutes = np.floor_divide(global_total_data_duration, 60.0)
        # print(f'\ttotal_data_duration_minutes: {global_total_data_duration_minutes}') # 28.0
        
        # Build the time-ticks for each minute over the global data times:
        # minute_t_ticks = np.linspace(global_start_t, global_end_t, num=global_total_data_duration_minutes)
        global_minute_t_ticks = np.arange(global_start_t, global_end_t, 60.0) # steps by 60.0 seconds (1 minute) from global_start_t to global_end_t. Doesn't need the explicit minutes calculation.
        global_minute_x_tick_positions = DataSeriesToSpatial.temporal_to_spatial_map(global_minute_t_ticks,
                                                                                spike_raster_plt.spikes_window.total_data_start_time, spike_raster_plt.spikes_window.total_data_end_time, # spike_raster_plt_3d_vedo.spikes_window.active_window_start_time, spike_raster_plt_3d_vedo.spikes_window.active_window_end_time,
                                                                                spike_raster_plt.temporal_axis_length,
                                                                                center_mode=spike_raster_plt.params.center_mode, enable_debug_print=enable_debug_print)
        
        #  xValuesAndLabels: list of custom tick positions and labels [(pos1, label1), …]
        # Want to add a tick/label at the x-values corresponding to each minute.
        return [(tick_x, format_seconds_human_readable(tick_t)) for tick_x, tick_t in list(zip(global_minute_x_tick_positions, global_minute_t_ticks))]






class DataSeriesToSpatialTransformingMixin:
    """ 
    Provides parameter independent functions to get spatial values from temporal and unit-identity ones.
    
    temporal_to_spatial(temporal_data)
    
    requires that conforming class implements:
    
    Required Properties:
        self.temporal_axis_length
        self.spikes_window.active_window_start_time
        self.spikes_window.active_window_end_time
        
        self.n_cells
        self.params.center_mode
        self.params.bin_position_mode
        self.params.side_bin_margins
        
    """    
    def temporal_to_spatial(self, temporal_data):
        """ transforms the times in temporal_data to a spatial offset (such as the x-positions for a 3D raster plot) """
        return DataSeriesToSpatial.temporal_to_spatial_map(temporal_data, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
    
    def fragile_linear_neuron_IDX_to_spatial(self, fragile_linear_neuron_IDXs):
        """ transforms the fragile_linear_neuron_IDXs in fragile_linear_neuron_IDXs to a spatial offset (such as the y-positions for a 3D raster plot) """
        raise NotImplementedError
    
    def build_series_identity_axis(self):
        # build the position range for each unit along the y-axis:
        return DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)        
        # return DataSeriesToSpatial.temporal_to_spatial_map(temporal_data, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
    
