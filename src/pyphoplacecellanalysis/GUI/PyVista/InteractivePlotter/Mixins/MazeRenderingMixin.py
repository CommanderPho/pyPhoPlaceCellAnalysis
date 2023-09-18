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
		else:
			#TODO 2023-09-13 14:23: - [ ] 2023-09-13 - A superior version for the linear track that uses actually known maze geometry and the user-provided `grid_bin_bounds` used to compute:
			## Add the 3D Maze Shape
			assert self.active_config.computation_config.pf_params.grid_bin_bounds is not None, f"could not get the grid_bin_bounds from self.params.active_epoch_placefields.config.grid_bin_bounds"
			a_track_dims = LinearTrackDimensions3D()
			a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(self.active_config.computation_config.pf_params.grid_bin_bounds, return_geoemtry=True)
			self.plots['maze_bg'] = perform_plot_flat_arena(self.p, ideal_maze_pdata, name='maze_bg', label='idealized_maze', color=color) # [0.3, 0.3, 0.3]
			self.plots_data['maze_bg'] = {'track_dims': a_track_dims, 'maze_pdata': ideal_maze_pdata}
			
		return self.plots['maze_bg'], self.plots_data['maze_bg']
	