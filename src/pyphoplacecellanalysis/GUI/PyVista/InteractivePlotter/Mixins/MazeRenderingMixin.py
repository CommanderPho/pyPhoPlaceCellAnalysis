from copy import deepcopy
from typing import Any, Optional
import param
from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import BasePlotDataParams
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions3D
from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena


class TrackShapePlottingConfig(BasePlotDataParams):
    """ Handles configuration related to plotting the 3D track shapes in the Vedo visualizations
    
    
    NOTE: Upon reviewing many different versions of my plotting implementations, this Param-based one is the most succinct and smooth.

    This class uses the 'param' library to observe changes to its members and perform corresponding updates to the class that holds it when they happen:
    
    From TrackShapePlottingConfig.setup_occupancy_plotting_mixin(self):
        # Setup watchers:    
        self.occupancy_plotting_config.param.watch(self.plot_occupancy_bars, OccupancyPlottingConfig._config_update_watch_labels(), queued=True)
        self.occupancy_plotting_config.param.watch(self.on_occupancy_plot_update_visibility, OccupancyPlottingConfig._config_visibility_watch_labels(), queued=True)
    
    
    Note that _config_update_watch_labels() provides the names/labels of the properties that when updated trigger plot_occupancy_bars(...)
        and _config_visibility_watch_labels() provides those for on_occupancy_plot_update_visibility(...)
    """
    debug_logging = False

    @staticmethod
    def _config_update_watch_labels():
        return ['should_use_linear_track_geometry', 'visible_track_opacity', 'hidden_track_opacity', 't_delta']
    @staticmethod
    def _config_visibility_watch_labels():
        return ['should_use_linear_track_geometry', 'isVisible']
    
    # Overriding defaults from parent
    name = param.String(default='MazeShape')
    isVisible = param.Boolean(default=False, doc="Whether the Maze Shape is visible") # default to False

    # Bar properties:
    should_use_linear_track_geometry = param.Boolean(default=False, doc="Whether to show a single maze reconstructed from all positions or Diba linear mazes.")

    visible_track_opacity = param.Number(default=1.0, bounds=(0.0, 1.0), step=0.1)
    hidden_track_opacity = param.Number(default=0.1, bounds=(0.0, 1.0), step=0.1)

    long_maze_bg_color = param.Color(default="#4c4c4c", doc="the color of the long track mesh")
    short_maze_bg_color = param.Color(default="#4c4c4c", doc="the color of the short track mesh")
    
    # General properties:    
    t_delta = param.Number(default=-666.0)

    # def to_plot_config_dict(self):
    #     issue_labels = {'name': 'OccupancyLabels', 'name': 'Occupancy'}
    #     return {'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity, 'shape': 'rounded_rect', 'visible_track_opacity': self.visible_track_opacity, 'hidden_track_opacity': self.hidden_track_opacity}
    
    # def to_bars_plot_config_dict(self):
    #     return {'name': 'Occupancy', 'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity}
    
    # def to_labels_plot_config_dict(self):
    #     return {'name': 'OccupancyLabels', 'shape': 'rounded_rect', 'visible_track_opacity': self.visible_track_opacity, 'hidden_track_opacity': self.hidden_track_opacity}
    



class InteractivePyvistaPlotter_MazeRenderingMixin:
    """ Allows Implementors to render a 3D track shape
    
    
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.MazeRenderingMixin import InteractivePyvistaPlotter_MazeRenderingMixin

    Renders the 3D track/maze shape on the 3D PyVista plot.

    """
    @property
    def long_maze_bg(self):
        """The long_maze_bg property."""
        return self.plots.get('long_maze_bg', None)

    @property
    def short_maze_bg(self):
        """The short_maze_bg property."""
        return self.plots.get('short_maze_bg', None)
    

    @property
    def t_delta(self) -> Optional[float]:
        return self.params.get('t_delta', None)
    
    @property
    def visible_track_opacity(self) -> float:
        return self.params.get('visible_track_opacity', 1.0)
    @property
    def hidden_track_opacity(self) -> float:
        return self.params.get('hidden_track_opacity', 0.1)


    @property
    def grid_bin_bounds(self) -> Optional[Any]:
        ## prefer self.params..get('hidden_track_opacity', 0.1)
        grid_bin_bounds = self.params.get('grid_bin_bounds', None)
        if grid_bin_bounds is not None:
            return grid_bin_bounds # return this grid_bin_bounds
        
        ## otherwise try to get it from the active config
        grid_bin_bounds = None
        try:
            grid_bin_bounds = self.active_config.computation_config.pf_params.grid_bin_bounds
        except (KeyError, AttributeError) as err:
            # raise e
            print(f'WARNING: could not get the grid_bin_bounds from self.params.active_epoch_placefields.config.grid_bin_bounds')
        except BaseException as err:
            raise err

        return grid_bin_bounds    


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

            assert self.grid_bin_bounds is not None, f"could not get the grid_bin_bounds from self.params.active_epoch_placefields.config.grid_bin_bounds"

            # assert self.active_config.computation_config.pf_params.grid_bin_bounds is not None, f"could not get the grid_bin_bounds from self.params.active_epoch_placefields.config.grid_bin_bounds"
            # a_track_dims = LinearTrackDimensions3D()
            # a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(self.active_config.computation_config.pf_params.grid_bin_bounds, return_geoemtry=True)
            # grid_bin_bounds = deepcopy(self.active_config.computation_config.pf_params.grid_bin_bounds)
            grid_bin_bounds = deepcopy(self.grid_bin_bounds)

            long_track_dims = LinearTrackDimensions3D(track_length=170.0)
            short_track_dims = LinearTrackDimensions3D(track_length=100.0)

            long_track_pdata = long_track_dims.build_maze_geometry(position_offset=None, grid_bin_bounds=grid_bin_bounds)
            short_track_pdata = short_track_dims.build_maze_geometry(position_offset=None, grid_bin_bounds=grid_bin_bounds)

            def _subfn_perform_plot_mazes(a_plotter, a_plotter_id_prefix: str=''):
                long_maze_bg_key: str = f"long_maze_bg"
                short_maze_bg_key: str = f"short_maze_bg"
                if (a_plotter_id_prefix is not None) and (len(a_plotter_id_prefix) > 0):
                    long_maze_bg_key: str = '_'.join([a_plotter_id_prefix, long_maze_bg_key])
                    short_maze_bg_key: str = '_'.join([a_plotter_id_prefix, short_maze_bg_key])

                self.plots[long_maze_bg_key] = perform_plot_flat_arena(a_plotter, long_track_pdata, name=long_maze_bg_key, label='long_idealized_maze', color=color) # [0.3, 0.3, 0.3]
                self.plots[short_maze_bg_key] = perform_plot_flat_arena(a_plotter, short_track_pdata, name=short_maze_bg_key, label='short_idealized_maze', color=color) # [0.3, 0.3, 0.3]

            self.plots_data['long_maze_bg'] = {'track_dims': long_track_dims, 'maze_pdata': long_track_pdata}
            self.plots_data['short_maze_bg'] = {'track_dims': short_track_dims, 'maze_pdata': short_track_pdata}


            is_multiplotter: bool = (hasattr(self.p, '__getitem__') and hasattr(self.p, '_nrows') and hasattr(self.p, '_ncols'))
            if is_multiplotter:
                for row in range(self.p._nrows):
                    for col in range(self.p._ncols):
                        p = self.p[row, col]
                        _subfn_perform_plot_mazes(a_plotter=p, a_plotter_id_prefix=f"p[{row}][{col}]")

            else:
                p = self.p
                _subfn_perform_plot_mazes(a_plotter=p)

            # keys = ['long_maze_bg', 'short_maze_bg']
            # self.plots['maze_bg'] = perform_plot_flat_arena(self.p, ideal_maze_pdata, name='maze_bg', label='idealized_maze', color=color) # [0.3, 0.3, 0.3]
            # self.plots_data['maze_bg'] = {'track_dims': a_track_dims, 'maze_pdata': ideal_maze_pdata}
            # return (self.plots['short_maze_bg'], self.plots_data['short_maze_bg']), (self.plots['long_maze_bg'], self.plots_data['long_maze_bg'])
    

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


    def setup_MazeRenderingMixin(self):
        """ called to setup all required config 
        """
        self.track_shape_plotting_config = TrackShapePlottingConfig()

        ## keys to add
        # keys_to_add = ['long_maze_bg', 'short_maze_bg']
        # for a_key in keys_to_add:
        #     self.plots[a_key] = None
        #     self.plots_data[a_key] = None 
        
        # # Setup watchers:    
        # self.track_shape_plotting_config.param.watch(self.perform_plot_maze, TrackShapePlottingConfig._config_update_watch_labels(), queued=True)
        # self.track_shape_plotting_config.param.watch(self.on_update_current_window_MazeRenderingMixin, TrackShapePlottingConfig._config_visibility_watch_labels(), queued=True)
    


    def on_update_current_window_MazeRenderingMixin(self, new_window_t_start: float, new_window_t_stop: float):
        """ called to update the current window. 
        
        """
        # print(f'.on_update_current_window_MazeRenderingMixin(new_window_t_start: {new_window_t_start}, new_window_t_stop: {new_window_t_stop})')

        if self.t_delta is None:
            print(f'WARNING: on_update_current_window_MazeRenderingMixin(...): no `t_delta`.')
            return # do nothing
        
        if (self.long_maze_bg is None) or (self.short_maze_bg is None):
            print(f'WARNING: on_update_current_window_MazeRenderingMixin(...): no `long_maze_bg` or `short_maze_bg`.')
            return # do nothing without mazes
        
        if new_window_t_start >= self.t_delta:
            ## long track inivisible:
            long_track_opacity: float = self.hidden_track_opacity
            ## short track visible
            short_track_opacity: float = self.visible_track_opacity
        else:
            ## long track visible:
            long_track_opacity: float = self.visible_track_opacity

            if new_window_t_stop < self.t_delta:
                ## short track inivisible
                short_track_opacity: float = self.hidden_track_opacity
            else:                
                ## short track visible
                short_track_opacity: float = self.visible_track_opacity


        ## Now we have: long_track_opacity, short_track_opacity
        ## post-delta:
        # self.long_maze_bg.GetProperty().SetOpacity(long_track_opacity)
        # self.short_maze_bg.GetProperty().SetOpacity(short_track_opacity)

        def _subfn_perform_update_maze_opacity(long_track_opacity: float, short_track_opacity: float, a_plotter_id_prefix: str=''):
            """ captures: long_track_opacity, short_track_opacity
            
            """
            long_maze_bg_key: str = f"long_maze_bg"
            short_maze_bg_key: str = f"short_maze_bg"
            if (a_plotter_id_prefix is not None) and (len(a_plotter_id_prefix) > 0):
                long_maze_bg_key: str = '_'.join([a_plotter_id_prefix, long_maze_bg_key])
                short_maze_bg_key: str = '_'.join([a_plotter_id_prefix, short_maze_bg_key])
            long_maze_bg = self.plots[long_maze_bg_key]
            short_maze_bg = self.plots[short_maze_bg_key]
            long_maze_bg.GetProperty().SetOpacity(long_track_opacity)
            short_maze_bg.GetProperty().SetOpacity(short_track_opacity)
            

        is_multiplotter: bool = (hasattr(self.p, '__getitem__') and hasattr(self.p, '_nrows') and hasattr(self.p, '_ncols'))
        if is_multiplotter:
            for row in range(self.p._nrows):
                for col in range(self.p._ncols):
                    _subfn_perform_update_maze_opacity(long_track_opacity=long_track_opacity, short_track_opacity=short_track_opacity, a_plotter_id_prefix=f"p[{row}][{col}]")

        else:
            _subfn_perform_update_maze_opacity(long_track_opacity=long_track_opacity, short_track_opacity=short_track_opacity)
