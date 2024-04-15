from copy import deepcopy
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions3D
from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena


class InteractivePyvistaPlotter_MazeRenderingMixin:
    """ 
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.MazeRenderingMixin import InteractivePyvistaPlotter_MazeRenderingMixin

    Renders the 3D track/maze shape on the 3D PyVista plot.

    """

    def perform_plot_maze(self, color=[0.3, 0.3, 0.3]):
        """ called from within implementor's self.plot(...) implementation to add the track/maze 
        Updates:
            self.plots['maze_bg']
            self.plots_data['maze_bg']
            
        """
        if not self.params.get('should_use_linear_track_geometry', False):
            # linear track geometry is not used to build the arena model, meaning for linear tracks it won't look as good as the geometry version.
            ## The track shape will be approximated from the positions and the positions of the spikes:
            self.plots['maze_bg'] = perform_plot_flat_arena(self.p, self.x, self.y, bShowSequenceTraversalGradient=False, smoothing=self.active_config.plotting_config.use_smoothed_maze_rendering)
            self.plots_data['maze_bg'] = {'track_dims': None, 'maze_pdata': None}
            return self.plots['maze_bg'], self.plots_data['maze_bg']
        
        else:
            #TODO 2023-09-13 14:23: - [ ] 2023-09-13 - A superior version for the linear track that uses actually known maze geometry and the user-provided `grid_bin_bounds` used to compute:
            ## Add the 3D Maze Shape
            assert self.active_config.computation_config.pf_params.grid_bin_bounds is not None, f"could not get the grid_bin_bounds from self.params.active_epoch_placefields.config.grid_bin_bounds"
            # a_track_dims = LinearTrackDimensions3D()
            # a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(self.active_config.computation_config.pf_params.grid_bin_bounds, return_geoemtry=True)


            grid_bin_bounds = deepcopy(self.active_config.computation_config.pf_params.grid_bin_bounds)
            long_track_dims = LinearTrackDimensions3D(track_length=170.0)
            short_track_dims = LinearTrackDimensions3D(track_length=100.0)

            long_track_pdata = long_track_dims.build_maze_geometry(position_offset=None, grid_bin_bounds=grid_bin_bounds)
            short_track_pdata = short_track_dims.build_maze_geometry(position_offset=None, grid_bin_bounds=grid_bin_bounds)


            self.plots['long_maze_bg'] = perform_plot_flat_arena(self.p, long_track_pdata, name='long_maze_bg', label='long_idealized_maze', color=color) # [0.3, 0.3, 0.3]
            self.plots_data['long_maze_bg'] = {'track_dims': long_track_dims, 'maze_pdata': long_track_pdata}

            self.plots['short_maze_bg'] = perform_plot_flat_arena(self.p, short_track_pdata, name='short_maze_bg', label='short_idealized_maze', color=color) # [0.3, 0.3, 0.3]
            self.plots_data['short_maze_bg'] = {'track_dims': short_track_dims, 'maze_pdata': short_track_pdata}

            # keys = ['long_maze_bg', 'short_maze_bg']

            # self.plots['maze_bg'] = perform_plot_flat_arena(self.p, ideal_maze_pdata, name='maze_bg', label='idealized_maze', color=color) # [0.3, 0.3, 0.3]
            # self.plots_data['maze_bg'] = {'track_dims': a_track_dims, 'maze_pdata': ideal_maze_pdata}
            
            return (self.plots['short_maze_bg'], self.plots_data['short_maze_bg']), (self.plots['long_maze_bg'], self.plots_data['long_maze_bg'])
    

    def perform_remove_maze_actor(self) -> bool:
        possible_maze_bg_keys = ['maze_bg', 'long_maze_bg', 'short_maze_bg']
        was_any_removed = False
        for a_maze_bg_key in possible_maze_bg_keys:
            maze_bg = self.plots.pop(a_maze_bg_key, None)
            if maze_bg is not None:
                was_removed_curr = self.p.remove_actor(maze_bg)
                if was_removed_curr:
                    maze_bg_data = self.plots_data.pop(a_maze_bg_key, None)
                was_any_removed = (was_any_removed or was_removed_curr)
        return was_any_removed
