# InteractivePyvistaPlotterBuildIfNeededMixin

# from neuropy
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import pyvista as pv
from qtpy import QtCore, QtGui, QtWidgets

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho3D.PyVista.gui import customize_default_pyvista_theme, get_gradients, print_controls_helper_text
from pyphoplacecellanalysis.PhoPositionalData.import_data import build_spike_positions_list # Used in _unpack_variables
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.InteractivePlotterMixins import InteractivePyvistaPlotter_ObjectManipulationMixin, InteractivePyvistaPlotter_BoxPlottingMixin, InteractivePyvistaPlotter_PointAndPathPlottingMixin, InteractivePyvistaPlotterBuildIfNeededMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecoderRenderingPyVistaMixin
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.MazeRenderingMixin import InteractivePyvistaPlotter_MazeRenderingMixin


######### MIXINS #############
##############################

######### InteractiveDataExplorerBase #############
##############################        
class InteractiveDataExplorerBase(DecoderRenderingPyVistaMixin, InteractivePyvistaPlotter_MazeRenderingMixin, InteractivePyvistaPlotter_BoxPlottingMixin, InteractivePyvistaPlotter_PointAndPathPlottingMixin, InteractivePyvistaPlotterBuildIfNeededMixin, InteractivePyvistaPlotter_ObjectManipulationMixin, QtCore.QObject):
    """The common abstract base class for building an interactive PyVistaQT BackgroundPlotter with extra GUI components and controls.
    
    Function call order:
        __init__
        _setup()
        _setup_variables()
        _setup_visualization()
        _setup_pyvista_theme()

    """
    def __init__(self, active_config, active_session, extant_plotter=None, data_explorer_name='InteractiveDataExplorerBase', **kwargs):        
        active_config_modifiying_kwargs = kwargs.pop('active_config_modifiying_kwargs', {}) # pop the possible modifications
        params_kwargs = kwargs.pop('params_kwargs', {})
        debug_kwargs = kwargs.pop('debug_kwargs', {})
        plots_data_kwargs = kwargs.pop('debug_kwargs', {}) 
        plots_kwargs = kwargs.pop('plots_kwargs', {}) 
        ui_kwargs = kwargs.pop('ui_kwargs', {}) 

        ## add these to `params_kwargs`
        for a_key in ['owning_pipeline', 'active_config_name', 'active_context']:
            a_val = kwargs.pop(a_key, None)
            if a_val is not None:
                assert (a_key not in params_kwargs), f"key '{a_key}' present both in params_kwargs and as a top-level kwarg to this init function!"
                params_kwargs[a_key] = a_val

        QtCore.QObject.__init__(self, **kwargs) # Initialize the QObject - TypeError: 'should_nan_non_visited_elements' is an unknown keyword argument, kwargs['zScalingFactor']
        self.active_config = deepcopy(active_config)

        ## If provided, apply custom `active_config_modifiying_kwargs` to the self.active_config config before setup
        for k, v in active_config_modifiying_kwargs.items():
            curr_subdict = self.active_config.get(k, {})
            for sub_k, sub_v in v.items():
                try:
                    curr_subdict[sub_k] = sub_v # apply the update
                except TypeError as err:
                    # TypeError: 'PlottingConfig' object does not support item assignment
                    setattr(curr_subdict, sub_k, sub_v)
                    

        self.active_session = active_session
        self.p = extant_plotter
        self.data_explorer_name = data_explorer_name
        
        self.z_fixed = None

        # Position variables: t, x, y
        self.pos_df = self.active_session.position.to_dataframe() ## full dataframe storage; t, x, y are computed properties



        # ## OPTION 1: Compute from 'approx_head_dir_degrees':
        # if 'heading_unit_xy' not in self.pos_df.columns:
        #     ## compute it
        #     h: float = 1.0
        #     self.pos_df['heading_unit_xy'] = self.pos_df['approx_head_dir_degrees'].map(lambda approx_head_dir_degrees: ((np.cos(np.radians(approx_head_dir_degrees)) * h), (np.sin(np.radians(approx_head_dir_degrees)) * h)))


        ## OPTION 2: Compute it from smoothed velocities like original AI implemementation -- handles low speeds that would otherwise cause jitter:
        def _subfn_add_heading_unit_xy(df: pd.DataFrame, vx_col: str = 'velocity_x_smooth', vy_col: str = 'velocity_y_smooth', speed_threshold: float = 0.01, fill_method: str = 'ffill') -> pd.DataFrame:
            """
            Compute heading unit vectors (ux, uy) from smoothed velocities.
            Rows with speed <= threshold or non‑finite velocities are marked invalid.
            Invalid headings can be filled forward ('ffill'), backward ('bfill'),
            or left as NaN (fill_method=None).

            Returns df with new column 'heading_unit_xy' containing tuples (ux, uy).

            fill_method # 'ffill', 'bfill', or None (no fill) 

            """
            vx = df[vx_col].to_numpy()
            vy = df[vy_col].to_numpy()
            speed = np.hypot(vx, vy)

            valid = (speed > speed_threshold) & np.isfinite(vx) & np.isfinite(vy)

            ux = np.full(len(df), np.nan, dtype=float)
            uy = np.full(len(df), np.nan, dtype=float)

            # Compute heading only where valid
            ux[valid] = vx[valid] / speed[valid]
            uy[valid] = vy[valid] / speed[valid]

            # Apply requested fill method
            if fill_method == 'ffill':
                ux = pd.Series(ux).ffill().to_numpy()
                uy = pd.Series(uy).ffill().to_numpy()
            elif fill_method == 'bfill':
                ux = pd.Series(ux).bfill().to_numpy()
                uy = pd.Series(uy).bfill().to_numpy()
            # else: leave NaN as is

            # Create tuple column
            df['heading_unit_xy'] = [ (ux[i], uy[i]) for i in range(len(df)) ]
            return df

        # if 'heading_unit_xy' not in self.pos_df.columns:
        self.pos_df = _subfn_add_heading_unit_xy(self.pos_df) # modifies in‑place

        ## add quaternion-derived heading direction
        h = 1.0
        self.pos_df['heading_unit_xy_quat'] = self.pos_df['quat_head_dir_degrees'].map(lambda approx_head_dir_degrees: ((np.cos(np.radians(approx_head_dir_degrees)) * h), (np.sin(np.radians(approx_head_dir_degrees)) * h)))


        # Helper variables
        display_class_name = f'{str(type(self))}{data_explorer_name}'
        self.params = VisualizationParameters(name=display_class_name, **params_kwargs)
        self.debug = DebugHelper(name=display_class_name, **debug_kwargs)
        self.plots_data = RenderPlotsData(name=display_class_name, **plots_data_kwargs)
        self.plots = RenderPlots(name=display_class_name, **plots_kwargs)
        self.ui = PhoUIContainer(name=display_class_name, **ui_kwargs)
        
        self.params.plotter_backgrounds = get_gradients()


    @property
    def t(self) -> NDArray:
        """The t property."""
        return self.pos_df['t'].to_numpy()

    @property
    def x(self) -> NDArray:
        """The x property."""
        return self.pos_df['x'].to_numpy()

    @property
    def y(self) -> NDArray:
        """The y property."""
        return self.pos_df['y'].to_numpy()


    @staticmethod
    def _unpack_variables(active_session):
        """ Unpacks the required variables from the active_session and returns them. Bascally a flexible mapping between active_session's properties and the required variables for the plotter. """
        # Spike variables: num_cells, spike_list, cell_ids, flattened_spikes
        num_cells = active_session.neurons.n_neurons
        spike_list = active_session.neurons.spiketrains
        cell_ids = active_session.neurons.neuron_ids
        # Gets the flattened spikes, sorted in ascending timestamp for all cells. Returns a FlattenedSpiketrains object
        
        ## NOTE: GOOD: No indexing issues here as it uses neuron_ids
        flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        # Get the indicies required to sort the flattened_spike_times
        flattened_sort_indicies = np.argsort(flattened_spike_times)
        t_start = active_session.neurons.t_start
        reverse_cellID_idx_lookup_map = active_session.neurons.reverse_cellID_index_map

        # Position variables: t, x, y
        pos_df = active_session.position.to_dataframe() ## full dataframe storage

        t = active_session.position.time
        x = active_session.position.x
        y = active_session.position.y
        linear_pos = active_session.position.linear_pos
        speeds = active_session.position.speed

        ### Build the flattened spike positions list
        # Determine the x and y positions each spike occured for each cell
        ## new_df style:
        # flattened_spike_positions_list_new = active_session.flattened_spiketrains.spikes_df[["x", "y"]].to_numpy().T
        # print('\n flattened_spike_positions_list_new: {}, {}'.format(np.shape(flattened_spike_positions_list_new), flattened_spike_positions_list_new))
        # flattened_spike_positions_list_new: (2, 17449), [[ nan 0.37450201 0.37450201 ... 0.86633532 0.86632449 0.86632266], [ nan 0.33842111 0.33842111 ... 0.47504852 0.47503917 0.47503759]]

        ## old-style:
        spike_positions_list = build_spike_positions_list(spike_list, t, x, y)
        flattened_spike_positions_list = np.concatenate(tuple(spike_positions_list), axis=1) # needs tuple(...) to conver the list into a tuple, which is the format it expects
        flattened_spike_positions_list = flattened_spike_positions_list[:, flattened_sort_indicies] # ensure the positions are ordered the same as the other flattened items so they line up
        # print('\n flattened_spike_positions_list_old: {}, {}\n\n'.format(np.shape(flattened_spike_positions_list), flattened_spike_positions_list))
        #  flattened_spike_positions_list_old: (2, 17449), [[103.53295196 100.94485182 100.86902972 ... 210.99778204 210.87296572 210.85173243]

        return num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, flattened_spike_positions_list


    def _setup(self):
        self._setup_variables()
        self._setup_visualization()
        self._setup_pyvista_theme()

    def _setup_variables(self):
        raise NotImplementedError
   
    def _setup_visualization(self):
        raise NotImplementedError

    def _setup_pyvista_theme(self):
        customize_default_pyvista_theme() # Sets the default theme values to those specified in my imported file
        # This defines the position of the vertical/horizontal splitting, in this case 40% of the vertical/horizontal dimension of the window
        # pv.global_theme.multi_rendering_splitting_position = 0.40
        pv.global_theme.multi_rendering_splitting_position = 0.80
        
    
    def plot(self, pActivePlotter=None):
        """ must be overriden by child class """
        raise NotImplementedError
    