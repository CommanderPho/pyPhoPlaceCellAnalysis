# LapsVisualizationMixin.py
import numpy as np
import pandas as pd
import pyvista as pv

class LapsVisualizationMixin:
    """ Looks like some of this class is independent of the rendering library and some is VTK and PyVista specific
    
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.LapsVisualizationMixin import LapsVisualizationMixin
    
    """
    
    @staticmethod
    def lines_from_points(points):
        """Given an array of points, make a line set"""
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
        cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
        poly.lines = cells
        return poly

    @staticmethod
    def get_lap_times(session, curr_lap_id):
        return session.laps.get_lap_times(curr_lap_id)
    
    @staticmethod
    def _compute_laps_specific_position_dfs(session):
        """ Just segments the laps into positions 
        curr_position_df, lap_specific_position_dfs = _compute_laps_specific_position_dfs(sess)

        Args:
            session (DataSession): [description]

        Returns:
            [type]: [description]
        """
        curr_position_df = session.compute_position_laps()
        lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in session.laps.lap_id] # dataframes split for each ID:
        return curr_position_df, lap_specific_position_dfs

    @classmethod
    def _compute_laps_position_data(cls, session):
        """ Given the session, extracts all position data by lap and returns it all """ 
        curr_position_df, lap_specific_position_dfs = cls._compute_laps_specific_position_dfs(session)
        return curr_position_df, lap_specific_position_dfs, *cls._lap_specific_positions_from_lap_position_dfs(lap_specific_position_dfs)
    
    @staticmethod
    def _lap_specific_positions_from_lap_position_dfs(lap_specific_position_dfs):
        """ Helper method just extracts the time and position variables from each of the lap_specific_position_dataframes """
        lap_specific_time_ranges = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]
        lap_specific_position_traces = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
        return lap_specific_time_ranges, lap_specific_position_traces
    
    @staticmethod
    def get_lap_position(session, curr_lap_id):
        curr_position_df = session.position.to_dataframe()
        curr_lap_t_start, curr_lap_t_stop = session.laps.get_lap_times(curr_lap_id)
        print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))

        curr_lap_position_df_is_included = curr_position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive='both') # returns a boolean array indicating inclusion in teh current lap
        curr_lap_position_df = curr_position_df[curr_lap_position_df_is_included] 
        # curr_position_df.query('-0.5 <= t < 0.5')
        curr_lap_position_traces = curr_lap_position_df[['x','y']].to_numpy().T
        print('\t {} positions.'.format(np.shape(curr_lap_position_traces)))
        # print('\t {} spikes.'.format(curr_lap_num_spikes))
        return curr_lap_position_traces

    @staticmethod
    def plot_lap_trajectory_path(interactiveDataExplorer, curr_lap_position_traces):
        num_lap_samples = np.shape(curr_lap_position_traces)[1]
        lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], 0.9)
        plot_name = 'lap_location_trail'
        trail_fade_values = None
        size_values = None
        trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
        size_values = np.linspace(0.7, 0.3, num_lap_samples) # fade from a scale of 0.2 to 0.6
        interactiveDataExplorer.perform_plot_location_trail(plot_name, curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z,
                                                    trail_fade_values=trail_fade_values, trail_point_size_values=size_values,
                                                    render=True, color='red')

    # @staticmethod
    # def plot_lap_trajectory_path_spline(p, curr_lap_position_traces):
    #     num_lap_samples = np.shape(curr_lap_position_traces)[1]
    #     lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], 0.9)
    #     curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z))
    #     plot_name = 'lap_location_trail_spline'
    #     trail_fade_values = None
    #     size_values = None
    #     trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
    #     size_values = np.linspace(0.2, 0.6, num_lap_samples) # fade from a scale of 0.2 to 0.6
    #     line = LapsVisualizationMixin.lines_from_points(curr_lap_points)
    #     line["scalars"] = np.arange(line.n_points)
    #     tube = line.tube(radius=0.5)
    #     p.add_mesh(tube, name=plot_name, render_lines_as_tubes=True, show_scalar_bar=False, color='red')

    @staticmethod
    def plot_lap_trajectory_path_spline(p, curr_lap_position_traces, curr_lap_id, lap_start_z=0.9, lap_id_dependent_z_offset=0.45, name=None, color_by_speed=True, **kwargs):
        """[summary]

        Args:
            p ([type]): [description]
            curr_lap_position_traces ([type]): a (3, N) position
            curr_lap_id ([type]): [description]
            lap_id_dependent_z_offset (float, optional): [description]. Defaults to 0.45.
            color_by_speed (bool, optional): If True, colors the spline by speed. If False, colors by time (index). Defaults to True.
        """
        num_lap_samples = np.shape(curr_lap_position_traces)[1]
        lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], lap_start_z + (lap_id_dependent_z_offset * curr_lap_id))
        curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z))
        
        line = LapsVisualizationMixin.lines_from_points(curr_lap_points)

        if color_by_speed:
            # Compute Speed along the path
            # 1. Calculate differences between consecutive points (dx, dy)
            dx = np.diff(curr_lap_position_traces[0, :])
            dy = np.diff(curr_lap_position_traces[1, :])
            
            # 2. Compute Euclidean distance (speed proxy, assuming constant sampling rate)
            # If sampling rate is not constant, you would divide this by dt (time delta)
            speed = np.sqrt(dx**2 + dy**2)
            
            # 3. Pad the array to match the number of points (diff reduces length by 1)
            # We repeat the last speed value to maintain shape
            speed = np.hstack((speed, speed[-1]))
            line["scalars"] = speed
        else:
            # Color by time (index) as per original implementation
            line["scalars"] = np.arange(line.n_points)
        
        # if name is None:
        #     plot_name = 'lap_location_trail_spline[{}]'.format(int(curr_lap_id))
        # else:
        #     plot_name = name
        
        trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
        size_values = np.linspace(0.2, 0.6, num_lap_samples) # fade from a scale of 0.2 to 0.6
        
        tube = line.tube(radius=0.2)
        # tube.plot(smooth_shading=True)
        color_map_name = 'bmy' # old
        color_map_name = 'cividis' # 2023-05-09 and newer
        
        # Note: 'show_scalar_bar': False is set, so you won't see the legend unless changed to True
        p.add_mesh(tube, **({'name': 'lap_location_trail_spline[{}]'.format(int(curr_lap_id)), 'render_lines_as_tubes': False, 'show_scalar_bar': False, 'cmap': color_map_name, 'lighting': False, 'render': False} | kwargs))

