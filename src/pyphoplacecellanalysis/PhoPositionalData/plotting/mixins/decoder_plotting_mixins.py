from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from matplotlib.collections import PathCollection

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND

from copy import deepcopy
import param
import numpy as np
import pandas as pd
from attrs import define, field, Factory
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from neuropy.utils.indexing_helpers import PandasHelpers

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlotsData, VisualizationParameters

from pyphocorehelpers.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import NeuronConfigOwningMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_placefields2D, update_plotColorsPlacefield2D

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

# ==================================================================================================================== #
# 2024-04-12 - Decoded Trajectory Plotting on Maze (1D & 2D) - Posteriors and Most Likely Position Paths               #
# ==================================================================================================================== #

from itertools import islice
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import LapsVisualizationMixin, LineCollection, _plot_helper_add_arrow # plot_lap_trajectories_2d
from pyphocorehelpers.plotting.heading_angle_helpers import HeadingAngleHelpers


class RenderColoringMode(str, Enum):
    """How to color rendered path elements (e.g. line segments, arrows): by time (colormap), by speed, or by heading angle (ROYGBIV, North=Red)."""
    TIME = 'time'
    SPEED = 'speed'
    ANGLE = 'angle'
    STATIC = 'static' ## not changing as a function of a property


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.geometry_helpers import point_tuple_mid_point, BoundsRect, is_point_in_rect

from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.EpochRenderTimebinSelectorWidget.EpochRenderTimebinSelectorWidget import EpochTimebinningIndexingDatasource # used in `DecodedTrajectoryPlotter` to conform to `EpochTimebinningIndexingDatasource` protocol



# ==================================================================================================================================================================================================================================================================================== #
# TODO 2025-12-16 16:37: - [ ] AI-implemnented attempt to replace Aims to replace `SingleArtistMultiEpochBatchHelpers` with a much more efficient implementation                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #

"""
Optimized viewport-based rendering with image caching and adaptive bin sizing
for decoded trajectory timeline visualization.

This class efficiently renders only visible epochs, caches rendered thumbnails,
and adapts bin size based on zoom level - similar to video editor timeline previews.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Dict, List, Callable, Any
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import numpy as np
import pandas as pd
from copy import deepcopy
from attrs import define, field, Factory
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image rendering

if TYPE_CHECKING:
    import napari
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, BasePositionDecoder
    from pyphoplacecellanalysis.External.peak_prominence2d import PosteriorPeaksPeakProminence2dResult
    from nptyping import NDArray

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter # DecodedTrajectoryPyVistaPlotter
from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_3d_smooth_mesh, plot_point_labels # DecodedTrajectoryPyVistaPlotter

import logging
logger = logging.getLogger(__name__)



@function_attributes(short_name=None, tags=['multi-ax', 'inefficient'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryMatplotlibPlotter'], used_by=[], creation_date='2025-02-18 03:22', related_items=['SingleArtistMultiEpochBatchHelpers'])
def multi_DecodedTrajectoryMatplotlibPlotter_side_by_side(a_result2D: DecodedFilterEpochsResult, a_new_global_decoder2D: BasePositionDecoder, global_session, n_axes: int = 10, posterior_masking_value: float = 0.020, desired_epoch_start_idx:int=0):
    """ Performs the same plotting as `SingleArtistMultiEpochBatchHelpers`, but in a less performant manner that draws each frame as a seperate artist (but unlike `SingleArtistMultiEpochBatchHelpers` computations are clear and it actually works)
        
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter
        from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import multi_DecodedTrajectoryMatplotlibPlotter_side_by_side

        n_axes: int = 10
        posterior_masking_value: float = 0.02 # for 2D
        a_decoded_traj_plotter, (fig, axs, decoded_epochs_pages) = multi_DecodedTrajectoryMatplotlibPlotter_side_by_side(a_result2D=results2D.a_result2D, a_new_global_decoder2D=results2D.a_new_global2D_decoder,
                                                                                                                        global_session=global_session, n_axes=n_axes, posterior_masking_value=posterior_masking_value)


                                                                                                                  
    """
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    

    # posterior_masking_value: float = 0.02 # for 2D

    # n_axes: int = 25
    ## INPUTS: directional_laps_results, decoder_ripple_filter_epochs_decoder_result_dict, a_result2D
    xbin = deepcopy(a_new_global_decoder2D.xbin)
    xbin_centers = deepcopy(a_new_global_decoder2D.xbin_centers)
    ybin_centers = deepcopy(a_new_global_decoder2D.ybin_centers)
    ybin = deepcopy(a_new_global_decoder2D.ybin)
    num_filter_epochs: int = a_result2D.num_filter_epochs
    a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=a_result2D, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, rotate_to_vertical=True)
    # fig, axs, decoded_epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=n_axes, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True, fixed_columns=n_axes)
    fig, axs, decoded_epochs_pages = a_decoded_traj_plotter.plot_decoded_laps_2d(global_session, curr_num_subplots=n_axes, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True, fixed_columns=n_axes)

    # perform_update_title_subtitle(fig=fig, ax=None, title_string="DecodedTrajectoryMatplotlibPlotter - plot_decoded_trajectories_2d") # , subtitle_string="TEST - SUBTITLE"

    # a_decoded_traj_plotter.fig = fig
    # a_decoded_traj_plotter.axs = axes
    ## INPUTS: desired_epoch_start_idx
    # desired_epoch_start_idx: int = 0
    # desired_epoch_start_idx: int = 214
    # desired_epoch_end_idx: int = desired_epoch_start_idx + 10 ## 10 frames before the 8 minute mark
    # desired_epoch_end_idx: int = 20
    # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
    # desired_epoch_start_idx: int = desired_epoch_end_idx - 10 ## 10 frames before the 8 minute mark
    # print(f'desired_epoch_start_idx: {desired_epoch_start_idx}, desired_epoch_end_idx: {desired_epoch_end_idx}')

    for i in np.arange(n_axes):
        print(f'plotting epoch[{i}]')
        ax = a_decoded_traj_plotter.axs[0][i]
        # Disable autoscaling to prevent later additions from changing limits
        # ax.set_autoscale_on(False)
        an_epoch_idx: int = desired_epoch_start_idx + i
        # a_decoded_traj_plotter.plot_epoch(an_epoch_idx=i, include_most_likely_pos_line=None, time_bin_index=None)
        # a_decoded_traj_plotter.plot_epoch(an_epoch_idx=an_epoch_idx, time_bin_index=None, include_most_likely_pos_line=None, override_ax=ax, should_post_hoc_fit_to_image_extent=False, posterior_masking_value=posterior_masking_value, debug_print=False)
        # a_decoded_traj_plotter.plot_epoch(an_epoch_idx=an_epoch_idx, override_plot_linear_idx=i, time_bin_index=0, include_most_likely_pos_line=None, posterior_masking_value=posterior_masking_value, override_ax=ax, should_post_hoc_fit_to_image_extent=False, debug_print=False)
        a_decoded_traj_plotter.plot_epoch(an_epoch_idx=an_epoch_idx, override_plot_linear_idx=i, time_bin_index=None, include_most_likely_pos_line=None, posterior_masking_value=posterior_masking_value, override_ax=ax, should_post_hoc_fit_to_image_extent=False, debug_print=False) ## OVERRIDE Epoch IDX

    a_decoded_traj_plotter.fig.canvas.draw_idle()

    return a_decoded_traj_plotter, (fig, axs, decoded_epochs_pages)




@define(slots=False)
class DecodedTrajectoryPlotter(EpochTimebinningIndexingDatasource):
    """ Abstract Base Class for something that plots a decoded 1D or 2D trajectory. 
    
    """
    curr_epoch_idx: int = field(default=None)
    a_result: DecodedFilterEpochsResult = field(default=None)
    xbin_centers: NDArray = field(default=None)
    ybin_centers: Optional[NDArray] = field(default=None)
    xbin: NDArray = field(default=None)
    ybin: Optional[NDArray] = field(default=None)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)
    

    @property
    def num_filter_epochs(self) -> int:
        """The num_filter_epochs: int property."""
        return self.a_result.num_filter_epochs
    
    @property
    def curr_n_time_bins(self) -> int:
        """The num_filter_epochs: int property."""
        return len(self.a_result.time_bin_containers[self.curr_epoch_idx].centers)


    # ==================================================================================================================== #
    # EpochTimebinningIndexingDatasource Conformances                                                                      #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['EpochTimebinningIndexingDatasource'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 05:15', related_items=['EpochTimebinningIndexingDatasource'])
    def get_epochs(self) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        return np.arange(self.num_filter_epochs)
        
    @function_attributes(short_name=None, tags=['EpochTimebinningIndexingDatasource'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 05:15', related_items=['EpochTimebinningIndexingDatasource'])
    def get_num_epochs(self) -> int:
        """ returns the number of time_bins for the specified epoch index """
        return self.num_filter_epochs
        

    @function_attributes(short_name=None, tags=['EpochTimebinningIndexingDatasource'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 05:15', related_items=['EpochTimebinningIndexingDatasource'])
    def get_time_bins_for_epoch_index(self, an_epoch_idx: int) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        if self.a_result is None:
            return [] # None
        if an_epoch_idx is None:
            return [] # None
            
        time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers
        n_curr_time_bins: int = len(time_bin_centers)
        return np.arange(n_curr_time_bins)
    


@define(slots=False)
class DecodedTrajectoryMatplotlibPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded 1D or 2D trajectory using matplotlib. 

    Usage:    
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter, RenderColoringMode

        ## 2D:
        # Choose the ripple epochs to plot:
        a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(LS_decoder_ripple_filter_epochs_decoder_result_dict)
        a_result: DecodedFilterEpochsResult = a_decoded_filter_epochs_decoder_result_dict['long'] # 2D
        num_filter_epochs: int = a_result.num_filter_epochs
        a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)
        fig, axs, laps_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True)

        integer_slider = a_decoded_traj_plotter.plot_epoch_with_slider_widget(an_epoch_idx=6)
        integer_slider

    """
    ## Artists/Figures/Axes:
    prev_heatmaps: List = field(default=Factory(list))
    artist_line_dict: Dict = field(default=Factory(dict))
    artist_markers_dict: Dict = field(default=Factory(dict))
    
    plots_data_dict_array: List[List[RenderPlotsData]] = field(init=False)
    artist_dict_array: List[List[Dict]] = field(init=False)
    fig: Any = field(default=None)
    axs: NDArray = field(default=None)
    epochs_pages: List = field(default=Factory(list))
    row_column_indicies: NDArray = field(default=None)
    linear_plotter_indicies: NDArray = field(default=None)
    
    # measured_position_df: Optional[pd.DataFrame] = field(default=None)
    rotate_to_vertical: bool = field(default=False, metadata={'desc': 'if False, the track is rendered horizontally along its length, otherwise it is rendered vectically'})
    cmap: Any = field(default='viridis') 
    
    ## Current Visibility State
    curr_epoch_idx: int = field(default=0)
    curr_time_bin_idx: Optional[int] = field(default=None)
    
    ## Widgets
    epoch_slider = field(default=None, init=False)
    time_bin_slider = field(default=None, init=False)
    checkbox = field(default=None, init=False)

    @property
    def is_single_time_bin_mode(self) -> bool:
        """ if True, all the time bins within the curr_epoch_idx are plotted, otherwise, only the time bin specified by curr_time_bin_idx is used."""
        return (self.curr_time_bin_idx is not None)

    def __attrs_post_init__(self):
        # self.params =
        # if self.cmap is not None:
        #     self.params.cmap = deepcopy(self.cmap)
        pass
        

    ## MAIN PLOT FUNCTION:
    @function_attributes(short_name=None, tags=['main', 'plot', 'posterior', 'epoch', 'line', 'trajectory'], input_requires=[], output_provides=[], uses=['self._perform_add_decoded_posterior_and_trajectory'], used_by=['plot_epoch_with_slider_widget'], creation_date='2025-01-29 15:52', related_items=[])
    def plot_epoch(self, an_epoch_idx: int, override_plot_linear_idx: Optional[int]=None, time_bin_index: Optional[int]=None, include_most_likely_pos_line: Optional[bool]=None, override_ax=None, should_post_hoc_fit_to_image_extent: bool = True, posterior_masking_value: float = 0.0025, debug_print:bool = False, **kwargs):
        """ Main plotting function.
             Internally calls `self._perform_add_decoded_posterior_and_trajectory(...)` to do the plotting.
             
            IMPORTANT: setting `override_plot_linear_idx=9` means the plot will occur on ax 9 but `an_epoch_idx=ANYTHING`. Allows plotting epochs on any arbitrary axes.
            
        """
        self.curr_epoch_idx = an_epoch_idx
        self.curr_time_bin_idx = time_bin_index

        if override_plot_linear_idx is not None:
            a_linear_index: int = override_plot_linear_idx
            
        else:
            a_linear_index: int = an_epoch_idx

        try:
            curr_row = self.row_column_indicies[0][a_linear_index]
            curr_col = self.row_column_indicies[1][a_linear_index]
            curr_artist_dict = self.artist_dict_array[curr_row][curr_col]
            curr_plot_data: RenderPlotsData = self.plots_data_dict_array[curr_row][curr_col]

        except IndexError as e:
            print(f'ERROR: IndexError: {e}:\n\n !!! Did you mean to plot an_epoch_idx={an_epoch_idx} but with an overriden `override_plot_linear_idx`?\n\tThis allows decoupling of the plot and epoch_idx, otherwise it always plots the first epochs.\n')
            raise
        except Exception as e:
            raise

        if override_ax is None:
            an_ax = self.axs[curr_row][curr_col] # np.shape(self.axs) - (n_subplots, 2)
        else:
            an_ax = override_ax
            
        # an_ax = self.axs[0][0] # np.shape(self.axs) - (n_subplots, 2)

        assert len(self.xbin_centers) == np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(self.xbin_centers)}"

        a_p_x_given_n = self.a_result.p_x_given_n_list[an_epoch_idx] # (76, 40, n_epoch_t_bins)
        a_most_likely_positions = self.a_result.most_likely_positions_list[an_epoch_idx] # (n_epoch_t_bins, n_pos_dims) 
        a_time_bin_edges = self.a_result.time_bin_edges[an_epoch_idx] # (n_epoch_t_bins+1, )
        a_time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers # (n_epoch_t_bins, )

        has_measured_positions: bool = hasattr(self.a_result, 'measured_positions_list')
        if has_measured_positions:
            a_measured_pos_df: pd.DataFrame = self.a_result.measured_positions_list[an_epoch_idx]
            # assert len(a_measured_pos_df) == len(a_time_bin_centers)
        else:
            a_measured_pos_df = None

        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)

        assert len(a_time_bin_centers) == len(a_most_likely_positions)

        # heatmaps, a_line, _out_markers, _slider_tuple = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
        #                                                                      a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers) # , allow_time_slider=True

        # removing existing:

        # curr_artist_dict = {'prev_heatmaps': [], 'lines': {}, 'markers': {}}
        
        for a_heatmap in curr_artist_dict['prev_heatmaps']:
            a_heatmap.remove()
        curr_artist_dict['prev_heatmaps'].clear()

        for k, a_line in curr_artist_dict['lines'].items(): 
            a_line.remove()

        for k, _out_markers in curr_artist_dict['markers'].items(): 
            _out_markers.remove()
            
        curr_artist_dict['lines'].clear()# = {}
        curr_artist_dict['markers'].clear() # = {}
        
        ## Perform the plot:
        curr_artist_dict['prev_heatmaps'], (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data = self._perform_add_decoded_posterior_and_trajectory(an_ax, xbin_centers=self.xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, a_measured_pos_df=a_measured_pos_df, ybin_centers=self.ybin_centers,
                                                                            include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=time_bin_index, rotate_to_vertical=self.rotate_to_vertical,
                                                                            # should_perform_reshape=True,
                                                                            should_perform_reshape=False,
                                                                            should_post_hoc_fit_to_image_extent=should_post_hoc_fit_to_image_extent,
                                                                            posterior_masking_value=posterior_masking_value, 
                                                                            time_cmap=deepcopy(self.cmap),
                                                                            debug_print=debug_print, line_start_lw=kwargs.pop('line_start_lw', 2), line_end_lw=kwargs.pop('line_end_lw', 2)) # , allow_time_slider=True 


        ## update the plot_data
        curr_plot_data.update(plots_data)
        self.plots_data_dict_array[curr_row][curr_col] = curr_plot_data ## set to the new value
        
        if a_meas_pos_line is not None:
            curr_artist_dict['lines']['meas'] = a_meas_pos_line
        if _meas_pos_out_markers is not None:
            curr_artist_dict['markers']['meas'] = _meas_pos_out_markers
        
        if a_line is not None:
            curr_artist_dict['lines']['most_likely'] = a_line
        if _out_markers is not None:
            curr_artist_dict['markers']['most_likely'] = _out_markers

        self.fig.canvas.draw_idle()


    @function_attributes(short_name=None, tags=['plotting', 'widget', 'interactive'], input_requires=[], output_provides=[], uses=['self.plot_epoch'], used_by=[], creation_date='2025-01-29 15:49', related_items=[])
    def plot_epoch_with_slider_widget(self, an_epoch_idx: int, include_most_likely_pos_line: Optional[bool]=None):
        """ this builds an interactive ipywidgets slider to scroll through the decoded epoch events
        
        Internally calls `self.plot_epoch` to perform posterior and line plotting
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.curr_epoch_idx = an_epoch_idx  # Ensure curr_epoch_idx is set

        def integer_slider(update_func, description, min_val, max_val, initial_val):
            slider = widgets.IntSlider(description=description, min=min_val, max=max_val, value=initial_val)

            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    update_func(change['new'])
            slider.observe(on_slider_change)
            return slider

        def checkbox_widget(update_func, description, initial_val):
            checkbox = widgets.Checkbox(description=description, value=initial_val)

            def on_checkbox_change(change):
                if (change['type'] == 'change') and (change['name'] == 'value'):
                    update_func(change['new'])
            checkbox.observe(on_checkbox_change)
            return checkbox

        def update_epoch_idx(index):            
            # print(f'update_epoch_idx(index: {index}) called')
            time_bin_index = None # default to no time_bin_idx
            # if not self.time_bin_slider.disabled:
            #     print(f'\t(not self.time_bin_slider.disabled)!!')
            #     self.time_bin_slider.value = 0 # reset to 0
            #     time_bin_index = self.time_bin_slider.value
            self.plot_epoch(an_epoch_idx=index, override_plot_linear_idx=0, time_bin_index=time_bin_index, include_most_likely_pos_line=include_most_likely_pos_line)

        # def update_time_bin_idx(index):
        #     print(f'update_time_bin_idx(index: {index}) called')
        #     self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=index, include_most_likely_pos_line=include_most_likely_pos_line)

        # def on_checkbox_change(value):
        #     print(f'on_checkbox_change(value: {value}) called')
        #     if value:
        #         self.time_bin_slider.disabled = True
        #         self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=None, include_most_likely_pos_line=include_most_likely_pos_line)
        #     else:
        #         self.time_bin_slider.disabled = False
        #         self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=self.time_bin_slider.value, include_most_likely_pos_line=include_most_likely_pos_line)

        self.epoch_slider = integer_slider(update_epoch_idx, 'epoch_IDX:', 0, (self.num_filter_epochs-1), an_epoch_idx)
        # self.time_bin_slider = integer_slider(update_time_bin_idx, 'time bin:', 0, (self.curr_n_time_bins-1), 0)
        # self.checkbox = checkbox_widget(on_checkbox_change, 'Disable time bin slider', True)

        self.plot_epoch(an_epoch_idx=an_epoch_idx, override_plot_linear_idx=0, time_bin_index=None, include_most_likely_pos_line=include_most_likely_pos_line)

        display(self.epoch_slider)
        # display(self.checkbox)
        # display(self.time_bin_slider)


    # ==================================================================================================================== #
    # General Fundamental Plot Element Helpers                                                                             #
    # ==================================================================================================================== #
    
    # fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=0)
    @function_attributes(short_name=None, tags=['matplotlib', 'helper', 'gradient', 'curve', 'line'], input_requires=[], output_provides=[], uses=[], used_by=['plot_lap_trajectories_2d'], creation_date='2025-06-18 06:22', related_items=[])
    @classmethod
    def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False, s=20.0, line_start_lw: float = 0.3, line_end_lw: float = 1.0,
                line_color_scheme: Union[RenderColoringMode, str] = RenderColoringMode.TIME, 
                cmap='viridis', **LineCollection_kwargs,
            ):
        """ Adds a gradient line representing a timeseries of (x, y) positions. line_color_scheme: TIME (time colormap), ANGLE (heading ROYGBIV via HeadingAngleHelpers); str 'time'/'angle' also accepted.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        line_start_lw / line_end_lw: line width at trajectory start/end; thickness scales linearly along the trajectory.
        
        
        cls._helper_add_gradient_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        if isinstance(line_color_scheme, str):
            line_color_scheme = RenderColoringMode(line_color_scheme)
        if line_color_scheme == RenderColoringMode.ANGLE:
            return cls._helper_add_gradient_angle_visualizing_line(ax, t, x, y, add_markers=add_markers, s=s, line_start_lw=line_start_lw, line_end_lw=line_end_lw, **LineCollection_kwargs)
        # TIME (or SPEED): time-based colormap on segments
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        norm = plt.Normalize(t.min(), t.max())
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n_segments = len(segments)
        linewidths = np.linspace(line_start_lw, line_end_lw, n_segments) if n_segments else np.array([line_end_lw])
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)  # Choose a colormap
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidths, **LineCollection_kwargs)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_alpha(0.85)
        line = ax.add_collection(lc)

        if add_markers:
            # Builds scatterplot markers (points) along the path
            colors_arr = cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!
            # segments_arr = line.get_segments() # (16, 2, 2)
            # len(a_most_likely_positions) # 17
            _out_markers: PathCollection = ax.scatter(x=x, y=y, s=s, c=colors_arr, marker='D')
            return line, _out_markers
        else:
            return line, None


    @classmethod
    def _helper_add_gradient_angle_visualizing_line(cls, ax, t, x, y, add_markers=False, s=20.0, line_start_lw: float = 0.3, line_end_lw: float = 1.0, **LineCollection_kwargs):
        """Adds a trajectory line colored by heading angle (North=red, ROYGBIV). Same semantics as Vispy create_heading_rainbow_line.

        Line segments are colored by direction of travel; markers (if add_markers=True) are colored by per-vertex heading.
        Parameters t and time_cmap are kept for API compatibility but are not used for coloring.

        add_markers (bool): if True, draws points at each (x, y) position colored by heading at that vertex.
        line_start_lw / line_end_lw: line width at trajectory start/end; thickness scales linearly along the trajectory.

        Example:
            DecodedTrajectoryMatplotlibPlotter._helper_add_gradient_angle_visualizing_line(ax=axs[curr_row][curr_col], t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:])), x=laps_position_traces[curr_lap_id][0,:], y=laps_position_traces[curr_lap_id][1,:])
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        assert len(x) == len(y), f"len(x): {len(x)} != len(y): {len(y)}"
        if len(x) < 2:
            return None, None
        pos = np.column_stack([x, y])
        d = np.diff(pos, axis=0)
        angle_deg = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 360.0) % 360.0
        compass_deg = HeadingAngleHelpers._heading_deg_to_compass_deg(angle_deg)
        segment_colors = HeadingAngleHelpers.heading_angles_to_rainbow_colors(compass_deg, alpha=0.85)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n_segments = len(segments)
        linewidths = np.linspace(line_start_lw, line_end_lw, n_segments) if n_segments else np.array([line_end_lw])
        lc = LineCollection(segments, colors=segment_colors, linewidths=linewidths, alpha=0.85, **LineCollection_kwargs)
        line = ax.add_collection(lc)
        if add_markers:
            vertex_colors = HeadingAngleHelpers._positions_to_vertex_colors(pos)
            _out_markers: PathCollection = ax.scatter(x=x, y=y, s=s, c=vertex_colors, marker='D')
            return line, _out_markers
        return line, None
        


    @function_attributes(short_name=None, tags=['matplotlib', 'helper', 'gradient', 'curve', 'line'], input_requires=[], output_provides=[], uses=[], used_by=['plot_lap_trajectories_2d'], creation_date='2025-10-21 06:29', related_items=[])
    @classmethod
    def _helper_add_markers_to_line(cls, ax, t, x, y, time_cmap='viridis', s=50, marker='D', **scatter_kwargs) -> PathCollection:
        """ Adds a gradient line representing a timeseries of (x, y) positions.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        
        
        cls._helper_add_markers_to_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        # Create a continuous norm to map from data points to colors
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        norm = plt.Normalize(t.min(), t.max())
        if isinstance(time_cmap, str):
            time_cmap = plt.get_cmap(time_cmap)  # Choose a colormap
        # Builds scatterplot markers (points) along the path
        colors_arr = time_cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!

        _out_markers: PathCollection = ax.scatter(x=x, y=y, s=s, c=colors_arr, marker=marker, **scatter_kwargs)
        return _out_markers



    @function_attributes(short_name=None, tags=['matplotlib', 'helper', 'gradient', 'curve', 'line'], input_requires=[], output_provides=[], uses=[], used_by=['plot_lap_trajectories_2d'], creation_date='2025-10-21 07:40', related_items=[])
    @classmethod
    def _helper_add_concentrated_arrows_to_line(cls, ax, t, x, y, speed=None, time_cmap='viridis',
                                                arrow_color_scheme: Union[RenderColoringMode, str] = RenderColoringMode.TIME, arrow_skip: int=20,
                                                mutation_scale_multiplier = 40, mutation_scale_constant = 10, arrow_length_multiplier = 0.2, arrow_length_constant = 0.05, arrow_lw = 0.5,
                                                ) -> List[FancyArrowPatch]:
        """ Adds arrows along a path. arrow_color_scheme: RenderColoringMode.TIME (time_cmap), SPEED (speed), or ANGLE (HeadingAngleHelpers, North=Red, ROYGBIV); str also accepted.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        
        
        cls._helper_add_markers_to_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        if isinstance(arrow_color_scheme, str):
            arrow_color_scheme = RenderColoringMode(arrow_color_scheme)
        # Create a continuous norm to map from data points to colors
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        if arrow_color_scheme == RenderColoringMode.TIME:
            norm = plt.Normalize(t.min(), t.max())
        else:
            norm = None
        if isinstance(time_cmap, str):
            time_cmap = plt.get_cmap(time_cmap)  # Choose a colormap
        # # Builds scatterplot markers (points) along the path
        if speed is None:
            ## compute the total magnitude of speed but computing the vector displacement distance between successive timepoints:
            ## TODO: speed
            # displacement between successive positions
            dx = np.diff(x)
            dy = np.diff(y)
            dist = np.sqrt(dx**2 + dy**2)
            dt = np.diff(t)
            # instantaneous speed magnitude
            speed = np.concatenate([[0], dist / np.maximum(dt, 1e-9)])
            
        assert len(t) == len(speed), f"len(t): {len(t)} != len(speed): {len(speed)}"
        # Use a robust reference speed (e.g. 95th percentile) so one spike doesn't make one arrow massive
        _speed_finite = np.asarray(speed)[np.isfinite(speed)]
        speed_ref = float(np.percentile(_speed_finite, 95)) if len(_speed_finite) > 0 else 1.0
        if speed_ref <= 0:
            speed_ref = 1.0
        # colors_arr = time_cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!
        _out_markers = {}
        # --- Add Arrows along the path ---
        # how many points to skip between arrows
        for i in range(0, len(x)-arrow_skip, arrow_skip):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i+1], y[i+1]
            dx, dy = x1 - x0, y1 - y0
            spd = speed[i]
            spd_percent_max = min((spd / speed_ref) if np.isfinite(spd) else 0.0, 1.0)
            # scale arrow size by speed
            arrow_length = arrow_length_constant + (arrow_length_multiplier * spd_percent_max)
            mutation_scale = mutation_scale_constant + (mutation_scale_multiplier * spd_percent_max)
            if arrow_color_scheme == RenderColoringMode.TIME:
                arrow_color = time_cmap(norm(t[i]))
            elif arrow_color_scheme == RenderColoringMode.SPEED:
                arrow_color = time_cmap(spd_percent_max)
            elif arrow_color_scheme == RenderColoringMode.ANGLE:
                angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                compass_deg = float(np.asarray(HeadingAngleHelpers._heading_deg_to_compass_deg(angle_deg)).flat[0])
                arrow_color = HeadingAngleHelpers.heading_angle_to_rainbow_rgba(compass_deg, alpha=1.0)
            elif arrow_color_scheme == RenderColoringMode.STATIC:
                arrow_color = '#333333BB'
            else:
                raise ValueError(f"arrow_color_scheme must be {list(RenderColoringMode)}, got {arrow_color_scheme!r}")
            arrow = FancyArrowPatch(
                (x0, y0),
                (x0 + dx * arrow_length, y0 + dy * arrow_length),
                arrowstyle='-|>', mutation_scale=mutation_scale,
                color=arrow_color,
                lw=arrow_lw
            )
            _out_markers[i] = arrow
            ax.add_patch(arrow)
        ## END for for i in range(0, len(x)-arrow_skip, arrow_skip)...
        
        return _out_markers
    

    @function_attributes(short_name=None, tags=['AI', 'posterior', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 12:00', related_items=[])
    @classmethod
    def _helper_add_heatmap(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers=None, ybin_centers=None, rotate_to_vertical:bool=False, debug_print:bool=False,
                            posterior_masking_value: float = 0.0025, full_posterior_opacity: float = 1.0,
                            custom_image_extent=None, time_cmap = 'viridis', should_perform_reshape: bool=True, extant_plot_data: Optional[RenderPlotsData]=None):
        """
        Helper that handles all the posterior heatmap plotting (for both 1D and 2D cases).
        
        Arguments:
            an_ax: the matplotlib axes to plot upon.
            xbin_centers: x axis bin centers.
            a_p_x_given_n: the decoded posterior array. If should_perform_reshape is True, its transpose is taken.
            a_time_bin_centers: array of time bin centers. -- Unused if 2D
            ybin_centers: if provided then a 2D posterior is assumed.
            rotate_to_vertical: if True, swap the x and y axes.
            debug_print: if True, prints debug information.
            posterior_masking_value: values below this are masked.
            should_perform_reshape: if True, reshapes the posterior.
            
        Returns:
            heatmaps: list of image handles.
            image_extent: extent (x_min, x_max, y_min, y_max) used for imshow.
            extra_dict: dictionary of additional computed values:
                For 1D: includes 'fake_y_center', 'fake_y_lower_bound', 'fake_y_upper_bound', 'fake_y_arr'.
                For 2D: may include 'y_values' and the flag 'is_2D': True.
        """
        # Reshape the posterior if necessary.
        if should_perform_reshape:
            posterior = deepcopy(a_p_x_given_n).T
        else:
            posterior = deepcopy(a_p_x_given_n)
        if debug_print:
            print('_helper_add_heatmap(...)')
            print(f'\tnp.shape(posterior): {np.shape(posterior)}')
        
        is_2D_dt: bool = (np.ndim(posterior) >= 3)
        is_2D: bool = (np.ndim(posterior) == 2)
        is_1D: bool = (np.ndim(posterior) < 2)

        # Add time dimension if posterior is 2D (spatial 2D without time dimension)
        if is_2D and (not is_2D_dt):
            posterior = posterior[np.newaxis, :, :]  # Shape: (1, n_x_bins, n_y_bins)

        if posterior_masking_value is not None:
            masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)
        else:
            masked_posterior = posterior
            

        if debug_print:
            print(f'\tis_2D: {is_2D}')
        
        x_values = deepcopy(xbin_centers)
        extra_dict = {'is_2D': is_2D}
        
        # if not is_2D:
        if is_1D:
            # 1D: Build fake y-axis values from current axes limits.
            y_min, y_max = an_ax.get_ylim()
            fake_y_width = (y_max - y_min)
            fake_y_center: float = y_min + (fake_y_width / 2.0)
            fake_y_lower_bound: float = fake_y_center - fake_y_width
            fake_y_upper_bound: float = fake_y_center + fake_y_width
            fake_y_num_samples: int = len(a_time_bin_centers)
            fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples)
            extra_dict.update({
                'fake_y_center': fake_y_center,
                'fake_y_lower_bound': fake_y_lower_bound,
                'fake_y_upper_bound': fake_y_upper_bound,
                'fake_y_arr': fake_y_arr,
            })
            # For plotting, use fake_y values.
            y_values = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples)
            extra_dict['y_values'] = y_values ## not needed?
        else:
            # 2D: use provided ybin_centers.
            assert ybin_centers is not None, "For 2D posterior, ybin_centers must be provided."
            y_values = deepcopy(ybin_centers)
            extra_dict['y_values'] = y_values
        
        # Adjust for vertical orientation if requested.
        if rotate_to_vertical:
            ordinate_first_image_extent = (y_values.min(), y_values.max(), x_values.min(), x_values.max())
            # Swap x and y arrays.
            x_values, y_values = y_values, x_values
            if should_perform_reshape:
                if debug_print:
                    print(f'\trotate_to_vertical: swapping axes. Original masked_posterior shape: {np.shape(masked_posterior)}')
                masked_posterior = masked_posterior.swapaxes(-2, -1) ## swap the last two (x, y) axes -- this doesn't work, because
                
            if debug_print:
                print(f'\tPost-swap masked_posterior shape: {np.shape(masked_posterior)}')
        else:
            ordinate_first_image_extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
        
        if custom_image_extent is not None:
            assert len(custom_image_extent) == 4
            print(f'\tusing `custom_image_extent`: prev_image_extent: {ordinate_first_image_extent}, custom_image_extent: {custom_image_extent}')
            ordinate_first_image_extent = deepcopy(custom_image_extent)

        ## set after any swapping:
        extra_dict['x_values'] = x_values
        extra_dict['y_values'] = y_values

        masked_shape = np.shape(masked_posterior)
        
        if a_time_bin_centers is not None:
            n_time_bins: int = len(a_time_bin_centers)
            # Assert.all_equal(n_time_bins, masked_shape[0])
            assert n_time_bins == masked_shape[0], f" masked_shape[0]: { masked_shape[0]} != n_time_bins: {n_time_bins}"
        else:
            n_time_bins: int = masked_shape[0] ## infer from posterior

        extra_dict['n_time_bins'] = n_time_bins
        if extant_plot_data is None:
            plots_data = RenderPlotsData(name='_helper_add_heatmap', ordinate_first_image_extent=deepcopy(ordinate_first_image_extent), **extra_dict)
        else:
            plots_data = extant_plot_data
            plots_data['ordinate_first_image_extent'] = deepcopy(ordinate_first_image_extent)
            plots_data.update(**extra_dict) ## update the existing

        if debug_print:
            print(f"\tfinal parameters: masked_posterior.shape: {np.shape(masked_posterior)}, aspect='auto', ordinate_first_image_extent={ordinate_first_image_extent}, origin='lower', interpolation='none'")            

        heatmaps = []
        # For simplicity, we assume non-single-time-bin mode (as asserted in the calling function).
        # if (not is_2D):
        if is_1D:
            a_heatmap = an_ax.imshow(masked_posterior, aspect='auto', cmap=time_cmap, alpha=full_posterior_opacity,
                                       extent=ordinate_first_image_extent, origin='lower', interpolation='none')
            heatmaps.append(a_heatmap)
        else:
            vmin_global = np.nanmin(posterior)
            vmax_global = np.nanmax(posterior)
            # Give a minimum opacity per time step.
            time_step_opacity: float = max(full_posterior_opacity/float(n_time_bins), 0.2)
            for i in np.arange(n_time_bins):
                a_heatmap = an_ax.imshow(np.squeeze(masked_posterior[i, :, :]), aspect='auto', cmap=time_cmap, alpha=time_step_opacity,
                                           extent=ordinate_first_image_extent, origin='lower', interpolation='none',
                                           vmin=vmin_global, vmax=vmax_global)
                heatmaps.append(a_heatmap)
        return heatmaps, ordinate_first_image_extent, plots_data


    @function_attributes(short_name=None, tags=['BROKEN', 'NOTFULLYWORKING', 'AI', 'posterior', 'helper', 'contours', 'HDR'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-26 13:00', related_items=[])
    @classmethod
    def _helper_add_hdr_contours(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers=None, ybin_centers=None, 
                                 rotate_to_vertical:bool=False, debug_print:bool=False,
                                 posterior_masking_value: float = 0.0025, full_posterior_opacity: float = 1.0,
                                 custom_image_extent=None, time_cmap = 'viridis', should_perform_reshape: bool=True, 
                                 extant_plot_data: Optional[RenderPlotsData]=None,
                                 contour_level_fractions: List[float] = [0.5], filled: bool = False, smoothing_sigma: float = 1.0):
        """
        Drop-in replacement for _helper_add_heatmap that renders Highest Density Region (HDR) contours.
        
        Args:
            filled (bool): If True, uses contourf (shading). If False, uses contour (outlines).
            smoothing_sigma (float): Standard deviation for Gaussian kernel. 
                                     > 0.5 is recommended to prevent "vertex explosion" crashes.
        """
        from scipy.ndimage import gaussian_filter
        import matplotlib.cm as cm
        import warnings

        # ========================================================== #
        # 1. SETUP & RESHAPING                                       #
        # ========================================================== #
        if should_perform_reshape:
            posterior = deepcopy(a_p_x_given_n).T
        else:
            posterior = deepcopy(a_p_x_given_n)
        
        # Determine Dimensionality
        is_2D: bool = (np.ndim(posterior) >= 3)
        
        # Setup Axes/Values
        x_values = deepcopy(xbin_centers)
        extra_dict = {'is_2D': is_2D}
        
        if not is_2D:
            # 1D Fallback Setup
            y_min, y_max = an_ax.get_ylim()
            fake_y_width = (y_max - y_min)
            fake_y_center = y_min + (fake_y_width / 2.0)
            fake_y_lower = fake_y_center - fake_y_width
            fake_y_upper = fake_y_center + fake_y_width
            fake_y_num = len(a_time_bin_centers) if a_time_bin_centers is not None else posterior.shape[1]
            y_values = np.linspace(fake_y_lower, fake_y_upper, fake_y_num)
            extra_dict.update({'fake_y_center': fake_y_center, 'y_values': y_values})
        else:
            # 2D Setup
            assert ybin_centers is not None, "For 2D posterior, ybin_centers must be provided."
            y_values = deepcopy(ybin_centers)
            extra_dict['y_values'] = y_values
        
        # Handle Rotation
        if rotate_to_vertical:
            ordinate_first_image_extent = (y_values.min(), y_values.max(), x_values.min(), x_values.max())
            x_values, y_values = y_values, x_values
            # Swap data axes (Time, X, Y) -> (Time, Y, X)
            posterior = np.swapaxes(posterior, -2, -1)
        else:
            ordinate_first_image_extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
        
        if custom_image_extent is not None:
            ordinate_first_image_extent = deepcopy(custom_image_extent)

        masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)
        n_time_bins = masked_posterior.shape[0]
        extra_dict['n_time_bins'] = n_time_bins
        
        # Plot Data Container
        if extant_plot_data is None:
            plots_data = RenderPlotsData(name='_helper_add_hdr_contours', ordinate_first_image_extent=deepcopy(ordinate_first_image_extent), **extra_dict)
        else:
            plots_data = extant_plot_data
            plots_data['ordinate_first_image_extent'] = deepcopy(ordinate_first_image_extent)
            plots_data.update(**extra_dict)

        # ========================================================== #
        # 2. RENDERING                                               #
        # ========================================================== #
        artists_list = [] 
        
        if isinstance(time_cmap, str):
            cmap_obj = cm.get_cmap(time_cmap)
        else:
            cmap_obj = time_cmap

        # --- 1D CASE: Standard Heatmap Fallback ---
        if not is_2D:
             a_heatmap = an_ax.imshow(masked_posterior, aspect='auto', cmap=cmap_obj, alpha=full_posterior_opacity,
                                     extent=ordinate_first_image_extent, origin='lower', interpolation='none')
             artists_list.append(a_heatmap)
             
        # --- 2D CASE: HDR Contours ---
        else:
            XX, YY = np.meshgrid(x_values, y_values) 
            
            for t in range(n_time_bins):
                # 1. Extract Frame
                frame_data = np.squeeze(masked_posterior[t, :, :])
                
                # 2. Skip if empty
                if np.all(np.ma.getdata(frame_data) < posterior_masking_value) or np.all(np.isnan(frame_data)):
                    continue
                
                # 3. Gaussian Smoothing (CRITICAL for stability)
                # Fills masked values with 0.0 before smoothing to avoid NaN propagation
                if smoothing_sigma > 0:
                    frame_data_filled = np.ma.filled(frame_data, 0.0)
                    frame_data = gaussian_filter(frame_data_filled, sigma=smoothing_sigma)
                
                frame_max = np.nanmax(frame_data)
                if frame_max <= 1e-9: continue

                # 4. Color Calculation
                time_progress = t / max(1, (n_time_bins - 1))
                # Force tuple cast to prevent Matplotlib cycling error
                rgba_color = tuple(cmap_obj(time_progress)) 
                
                # 5. Level Calculation
                current_levels = [frac * frame_max for frac in contour_level_fractions]
                
                # 6. Plotting
                try:
                    if filled:
                        # Filled (Shaded polygons)
                        # Add a cap slightly above max to ensure the center is filled
                        fill_levels = current_levels + [frame_max * 1.05] 
                        cset = an_ax.contourf(XX, YY, frame_data, 
                                            levels=fill_levels, 
                                            colors=[rgba_color], 
                                            alpha=full_posterior_opacity)
                    else:
                        # Outlines (Lines)
                        if np.shape(frame_data.T) == np.shape(XX):
                            frame_data = frame_data.T
                        cset = an_ax.contour(XX, YY, frame_data, 
                                            levels=current_levels, 
                                            colors=[rgba_color], 
                                            linewidths=1.5, 
                                            alpha=full_posterior_opacity)
                    
                    artists_list.append(cset)
                    
                except ValueError as e:
                    if debug_print: print(f"Skipping contour for t={t}: {e}")
                    continue

        return artists_list, ordinate_first_image_extent, plots_data


    @function_attributes(short_name=None, tags=['matplotlib', 'helper', 'grid', 'xbin', 'ybin'], input_requires=[], output_provides=[], uses=[], used_by=['plot_epoch', 'plot_decoded_trajectories_2d'], creation_date='2025-01-XX XX:XX', related_items=[])
    @classmethod
    def _helper_add_bin_grid_lines(cls, an_ax, xbin=None, ybin=None, xbin_centers=None, ybin_centers=None, rotate_to_vertical: bool=False, grid_kwargs: Optional[Dict[str, Any]]=None, should_plot_on_top: bool=False):
        """Adds grid lines at xbin/ybin edge positions to a matplotlib axes.
        
        Uses matplotlib's built-in grid() function with custom tick positions for efficiency.
        This is much more efficient than creating individual axvline/axhline calls, especially
        when called on many axes or with many bins.
        
        Arguments:
            an_ax: the matplotlib axes to add grid lines to.
            xbin: array of x bin edges. If None, will try to infer from xbin_centers.
            ybin: array of y bin edges. If None, will try to infer from ybin_centers.
            xbin_centers: array of x bin centers (used to infer edges if xbin is None).
            ybin_centers: array of y bin centers (used to infer edges if ybin is None).
            rotate_to_vertical: if True, swap x and y axes.
            grid_kwargs: optional dictionary of kwargs to pass to grid() (e.g., {'color': 'gray', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.3}).
            should_plot_on_top: if True, render grid on top of data elements (useful for opaque heatmaps). Default False renders grid behind data.
        
        Returns:
            grid_lines: reference to the grid object (for compatibility, though grid() doesn't return individual lines).
        """
        if grid_kwargs is None:
            grid_kwargs = {'color': 'gray', 'linestyle': '--', 'linewidth': 0.1, 'alpha': 0.3}
        
        # Determine x bin edges
        if xbin is not None:
            x_edges = np.asarray(xbin)
        elif xbin_centers is not None:
            x_centers = np.asarray(xbin_centers)
            # Infer edges from centers (assuming uniform spacing)
            if len(x_centers) > 1:
                bin_width = x_centers[1] - x_centers[0]
                x_edges = np.concatenate([x_centers - bin_width/2, [x_centers[-1] + bin_width/2]])
            else:
                x_edges = None
        else:
            x_edges = None
        
        # Determine y bin edges
        if ybin is not None:
            y_edges = np.asarray(ybin)
        elif ybin_centers is not None:
            y_centers = np.asarray(ybin_centers)
            # Infer edges from centers (assuming uniform spacing)
            if len(y_centers) > 1:
                bin_width = y_centers[1] - y_centers[0]
                y_edges = np.concatenate([y_centers - bin_width/2, [y_centers[-1] + bin_width/2]])
            else:
                y_edges = None
        else:
            y_edges = None
        

        # 1. Control whether grid renders on top of or behind data elements
        an_ax.set_axisbelow(not should_plot_on_top)

        # Add grid lines using matplotlib's efficient grid() function with minor ticks
        # This is much more efficient than creating individual line artists
        if not rotate_to_vertical:
            # Normal orientation: x is horizontal, y is vertical
            if x_edges is not None:
                an_ax.set_xticks(x_edges, minor=True)
            if y_edges is not None:
                an_ax.set_yticks(y_edges, minor=True)
        else:
            # Rotated orientation: x is vertical, y is horizontal
            if x_edges is not None:
                an_ax.set_yticks(x_edges, minor=True)
            if y_edges is not None:
                an_ax.set_xticks(y_edges, minor=True)
        
        # Enable grid on minor ticks only (doesn't interfere with major ticks/labels)
        if should_plot_on_top:
            an_ax.grid(True, which='minor', zorder=10, **grid_kwargs)
        else:
            an_ax.grid(True, which='minor', **grid_kwargs)

        # # 1. Set the locators (where the ticks/grid lines happen)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0)) # x-grid every 0.5
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0)) # y-grid every 1.0

        # # 3. Apply those positions
        # an_ax.set_xticks(xbin)
        # an_ax.set_yticks(ybin)

        # # Turn on minor ticks
        # ax.minorticks_on()

        # an_ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='gray') # Customize major grid
        # ax.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='gray') # # Customize minor grid

        # 4. Hide ticks and labels, but KEEP the grid
        # 'length=0' is another way to hide ticks, but turning them off is more explicit.
        an_ax.tick_params(
            axis='both',       # Apply to both x and y axes
            # which='both',      # Apply to both major and minor ticks
            which='both',      # Apply to both major and minor ticks
            bottom=False,      # Turn off the tick marks on the bottom
            top=False,         # Turn off the tick marks on the top
            left=False,        # Turn off the tick marks on the left
            right=False,       # Turn off the tick marks on the right
            labelbottom=False, # Turn off the text labels on the bottom
            labelleft=False    # Turn off the text labels on the left
        )

        
        return an_ax  # Return axes for compatibility


    # ==================================================================================================================== #
    # Specific Data Extraction and plot wrapping functions                                                                 #
    # ==================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['specific', 'plot_helper'], input_requires=[], output_provides=[], uses=['cls._helper_add_gradient_line'], used_by=['cls._perform_add_decoded_posterior_and_trajectory'], creation_date='2025-02-11 15:40', related_items=[])
    @classmethod
    def _perform_plot_measured_position_line_helper(cls, an_ax, a_measured_pos_df, a_time_bin_centers, fake_y_lower_bound: float, fake_y_upper_bound: float, rotate_to_vertical: bool, debug_print: bool) -> Tuple[Any, Any]:
        """
        Helper function to plot the measured positions line (recorded laps) as a gradient line.
        This extracts the functionality from the original code block (lines 1116-1181) so that it can be reused.
        
        Returns a tuple (a_meas_pos_line, _meas_pos_out_markers) that are produced by the gradient line helper.
        """
        # a_valid_only_measured_pos_df = deepcopy(a_measured_pos_df)
        a_valid_only_measured_pos_df = deepcopy(a_measured_pos_df).dropna(subset=['t','x','y'])

        # Get measured time bins from the dataframe
        a_measured_time_bin_centers: NDArray = np.atleast_1d([np.squeeze(a_valid_only_measured_pos_df['t'].to_numpy())]).astype(float)
        # Determine X and Y positions based on dimensionality.
        if rotate_to_vertical is False:
            # 1D: construct fake y values.
            measured_fake_y_num_samples: int = len(a_valid_only_measured_pos_df)
            measured_fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, measured_fake_y_num_samples)
            x = np.atleast_1d([a_valid_only_measured_pos_df['x'].to_numpy()]).astype(float)
            y = np.atleast_1d([measured_fake_y_arr]).astype(float)
        else:
            # 2D: take columns as is.
            x = np.squeeze(a_valid_only_measured_pos_df['x'].to_numpy()).astype(float)
            y = np.squeeze(a_valid_only_measured_pos_df['y'].to_numpy()).astype(float)
        
        # If in single-time-bin mode, restrict positions to those with t <= current time bin center.
        # n_time_bins: int = len(a_time_bin_centers)
        # Here, the caller is expected to ensure that time_bin_index is valid.
        # (This helper would be called after the check for single-time-bin mode.)
        # In a full implementation, one may pass time_bin_index as an argument.
        # For now, we only handle the non-restricted case.
        
        # Squeeze arrays down to rank 1.
        a_measured_time_bin_centers = np.squeeze(a_measured_time_bin_centers).astype(float)
        x = np.squeeze(x).astype(float)
        y = np.squeeze(y).astype(float)
        if debug_print:
            print(f'\tFinal Shapes:')
            print(f'\tnp.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)}, np.shape(a_measured_time_bin_centers): {np.shape(a_measured_time_bin_centers)}')
        
        # Set pos_kwargs according to orientation.
        if not rotate_to_vertical:
            pos_kwargs = dict(x=x, y=y)
        else:
            pos_kwargs = dict(x=y, y=x)  # swap if vertical
        
        add_markers = True
        colors = [(0, 0.6, 0), (0, 0, 0)]  # first is green, second is black
        # Create a colormap (green to black).
        time_cmap = LinearSegmentedColormap.from_list("GreenToBlack", colors, N=25)
        
        # Use the helper to add a gradient line.
        a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(an_ax, t=a_measured_time_bin_centers, **pos_kwargs, add_markers=add_markers, cmap=time_cmap, zorder=0)
        
        return a_meas_pos_line, _meas_pos_out_markers
    

    @function_attributes(short_name=None, tags=['plot'], input_requires=[], output_provides=[], uses=['cls._helper_add_heatmap', 'cls._perform_plot_measured_position_line_helper'], used_by=['.plot_epoch'], creation_date='2025-01-29 15:53', related_items=[])
    @classmethod
    def _perform_add_decoded_posterior_and_trajectory(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, a_most_likely_positions, ybin_centers=None, a_measured_pos_df: Optional[pd.DataFrame]=None,
                                                        include_most_likely_pos_line: Optional[bool]=None, time_bin_index: Optional[int]=None, rotate_to_vertical:bool=False, debug_print=False, posterior_masking_value: float = 0.0025, should_perform_reshape: bool=True, should_post_hoc_fit_to_image_extent: bool=False,
                                                        time_cmap='viridis', **kwargs): # posterior_masking_value: float = 0.01 -- 1D
        """ Plots the 1D or 2D posterior and most likely position trajectory over the top of an axes created with `fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)`
        
        np.shape(a_time_bin_centers) # 1D & 2D: (12,)
        np.shape(a_most_likely_positions) # 2D: (12, 2)
        np.shape(posterior): 1D: (56, 27);    2D: (12, 6, 57)

        
        time_bin_index: if time_bin_index is not None, only a single time bin will be plotted. Provide this to plot using a slider or programmatically animating.


        Usage:

        # for 1D need to set `ybin_centers = None`
        an_ax = axs[0][0]
        heatmaps, a_line, _out_markers = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers)


        """

        is_single_time_bin_mode: bool = (time_bin_index is not None) and (time_bin_index != -1)
        assert not is_single_time_bin_mode, f"time_bin_index: {time_bin_index}"

        if debug_print:
            if a_measured_pos_df is not None:
                print(f'a_measured_pos_df.shape: {a_measured_pos_df.shape}')
        

        # ==================================================================================================================== #
        # Plot the posterior heatmap                                                                                           #
        # ==================================================================================================================== #
        # _active_plot_fn = cls._helper_add_heatmap
        # _active_plot_fn = cls._helper_add_hdr_contours
        _active_plot_fn = kwargs.pop('active_plot_fn', DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap)
        line_start_lw = kwargs.pop('line_start_lw', 0.3)
        line_end_lw = kwargs.pop('line_end_lw', 1.0)

        
        # Delegate the posterior plotting functionality.
        heatmaps, image_extent, extra_dict = _active_plot_fn(
            an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, ybin_centers=ybin_centers,
            rotate_to_vertical=rotate_to_vertical, debug_print=debug_print, 
            posterior_masking_value=posterior_masking_value, should_perform_reshape=should_perform_reshape,
            time_cmap=time_cmap)
        
        is_2D: bool = extra_dict['is_2D']
        if debug_print:
            print(f'is_single_time_bin_mode: {is_single_time_bin_mode}, is_2D: {is_2D}')
            
        # For 1D case, retrieve fake y values.
        if np.ndim(a_p_x_given_n) < 3:
            fake_y_center = extra_dict['fake_y_center']
            fake_y_arr = extra_dict['fake_y_arr']
            fake_y_lower_bound = extra_dict['fake_y_lower_bound']
            fake_y_upper_bound = extra_dict['fake_y_upper_bound']
            
        else:
            fake_y_center = None
            fake_y_arr = None
            fake_y_lower_bound = None
            fake_y_upper_bound = None

                    
        # # Add colorbar
        # cbar = plt.colorbar(a_heatmap, ax=an_ax)
        # cbar.set_label('Posterior Probability Density')


        # Add Gradiant Measured Position (recorded laps) Line ________________________________________________________________ #         
        if (a_measured_pos_df is not None):
            a_meas_pos_line, _meas_pos_out_markers = cls._perform_plot_measured_position_line_helper(an_ax, a_measured_pos_df, a_time_bin_centers, fake_y_lower_bound, fake_y_upper_bound, rotate_to_vertical=rotate_to_vertical, debug_print=debug_print)
        else:
            a_meas_pos_line = None
            _meas_pos_out_markers = None
            
        # Add Gradient Most Likely Position Line _____________________________________________________________________________ #
        if include_most_likely_pos_line:
            if not is_2D:
                x = np.atleast_1d([a_most_likely_positions[time_bin_index]]) # why time_bin_idx here?
                y = np.atleast_1d([fake_y_arr[time_bin_index]])
            else:
                # 2D:
                x = np.squeeze(a_most_likely_positions[:,0])
                y = np.squeeze(a_most_likely_positions[:,1])
                
            if is_single_time_bin_mode:
                ## restrict to single time bin if is_single_time_bin_mode:
                assert (time_bin_index < n_time_bins)
                a_time_bin_centers = np.atleast_1d([a_time_bin_centers[time_bin_index]])
                x = np.atleast_1d([x[time_bin_index]])
                y = np.atleast_1d([y[time_bin_index]])
                

            if not rotate_to_vertical:
                pos_kwargs = dict(x=x, y=y)
            else:
                # vertical:
                ## swap x and y:
                pos_kwargs = dict(x=y, y=x)
                

            if not is_2D: # 1D case
                # a_line = _helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=a_most_likely_positions, y=np.full_like(a_time_bin_centers, fake_y_center))
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True, line_start_lw=line_start_lw, line_end_lw=line_end_lw)
            else:
                # 2D case
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True, line_start_lw=line_start_lw, line_end_lw=line_end_lw)
        else:
            a_line, _out_markers = None, None
            

        if should_post_hoc_fit_to_image_extent:
            ## set Axes xlims/ylims post-hoc so they fit
            an_ax.set_xlim(image_extent[0], image_extent[1])
            an_ax.set_ylim(image_extent[2], image_extent[3])


        # plot_data = MatplotlibRenderPlots(name='_perform_add_decoded_posterior_and_trajectory')
        # plots = RenderPlots('_perform_add_decoded_posterior_and_trajectory')
        plots_data: RenderPlotsData = deepcopy(extra_dict) # RenderPlotsData(name='_perform_add_decoded_posterior_and_trajectory', image_extent=deepcopy(image_extent))

        return heatmaps, (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data


    @function_attributes(short_name=None, tags=['main', 'plot'], input_requires=[], output_provides=[], uses=[], used_by=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side', 'self.plot_decoded_laps_2d'], creation_date='2025-06-30 12:58', related_items=[])
    def plot_decoded_trajectories_2d(self, curr_position_df: pd.DataFrame, epoch_specific_position_dfs: List[pd.DataFrame], epoch_ids: NDArray, sess=None, curr_num_subplots=10, active_page_index=0, plot_actual_lap_lines:bool=False, fixed_columns: int = 2, use_theoretical_tracks_instead: bool = True, existing_ax=None, axes_inset_locators_list=None, cmap=None,
                                    posteriors=None, plot_mode: str='time_gradient', should_include_trajectory_arrows: bool=False, arrow_concentration_kwargs=None, line_opacity: float = 1.0, track_background_opacity: float = 0.03, trajectory_line_color_scheme: Union[RenderColoringMode, str] = RenderColoringMode.TIME,
                                    override_title_formatter_fn=None, **kwargs):
        """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots
        
        Called to setup the graph.
        
        Great plotting for laps.
        Plots in a paginated manner.
        
        use_theoretical_tracks_instead: bool = True - # if False, renders all positions the animal traversed over the entire session. Otherwise renders the theoretical (idaal) track.

        track_background_opacity: float = 0.03 - Alpha for the maze/track background (0=transparent, 1=opaque). Applied to both theoretical and linear-plot-data backgrounds.

        trajectory_line_color_scheme: RenderColoringMode.TIME (default) or ANGLE. Use ANGLE to color the trajectory by heading (ROYGBIV); makes doubling-back and direction changes much easier to see.

        ISSUE: `fixed_columns: int = 1` doesn't work due to indexing


        History: based off of plot_lap_trajectories_2d
        
        
        
        #TODO 2026-02-06 09:36: - [ ] This should be fully general and not need to relate to DECODED trajectories. I could be factored out display any trajectories without requiring a decoded result, right?
            ## UPDATE: sad, it looks like it used to be: `plot_decoded_trajectories_2d`
        
        

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import DecodedTrajectoryMatplotlibPlotter
        
            fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_position_df, epoch_specific_position_dfs=None, epoch_ids=None, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)

        
        """
        from pyphocorehelpers.geometry_helpers import compute_data_aspect_ratio

        # _active_plot_fn = cls._helper_add_heatmap
        # _active_plot_fn = cls._helper_add_hdr_contours
        _active_posterior_plot_fn = kwargs.pop('active_plot_fn', DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap)

        arrow_opacity: float = kwargs.pop('arrow_opacity', line_opacity)

        if should_include_trajectory_arrows:
            arrow_concentration_kwargs = dict(
                arrow_skip=30, time_cmap='viridis', arrow_color_scheme=RenderColoringMode.ANGLE,
                mutation_scale_multiplier=20, mutation_scale_constant=1, arrow_length_multiplier=0.2, arrow_length_constant=0.05, arrow_lw=0.5, arrow_opacity=arrow_opacity,
            ) | (arrow_concentration_kwargs or {})
            print(f'arrow_concentration_kwargs: {arrow_concentration_kwargs}')
        

        if (self.xbin is not None) and (self.ybin is not None):
            single_ax_aspect_ratio, (single_ax_width, single_ax_height) = compute_data_aspect_ratio(xbin=self.xbin, ybin=self.ybin)
        else:
            single_ax_width = None
            single_ax_height = None

        # try:
        if cmap is None:
            cmap = 'viridis'
        if isinstance(trajectory_line_color_scheme, str):
            trajectory_line_color_scheme = RenderColoringMode(trajectory_line_color_scheme)

        if (use_theoretical_tracks_instead and (sess is not None)):
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(deepcopy(sess.config))

        # except 

        def _subfn_chunks(iterable, size=10):
            iterator = iter(iterable)
            for first in iterator:    # stops when iterator is depleted
                def chunk():          # construct generator for next chunk
                    yield first       # yield element from for loop
                    for more in islice(iterator, size - 1):
                        yield more    # yield more elements from the iterator
                yield chunk()         # in outer generator, yield next chunk
            
        def _subfn_build_epochs_multiplotter(nfields: int, linear_plot_data=None):
            """ builds the figures
             captures: self.rotate_to_vertical, fixed_columns, (long_track_inst, short_track_inst), single_ax_width, single_ax_height
            
            """
            linear_plotter_indicies = np.arange(nfields)
            needed_rows: int = int(np.ceil(nfields / fixed_columns))

            if (single_ax_width is not None) and (single_ax_height is not None):
                all_column_width: float = (single_ax_width * float(fixed_columns))
                all_row_height: float = (single_ax_height * float(needed_rows))

                # (all_column_width, all_row_height)
                scaling_factor: float = 0.01
                figsize = [(scaling_factor * all_column_width), (scaling_factor * all_row_height)]
            else:
                ## OLD:
                figsize = [4*fixed_columns, 14*needed_rows]

            # print(f'[4*fixed_columns, 14*needed_rows]: {[4*fixed_columns, 14*needed_rows]}')
            # print(f'figsize: {figsize}')
            row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
            
            if existing_ax is None:
                ## Create a new axes and figure
                fig, axs = plt.subplots(needed_rows, fixed_columns, sharex=True, sharey=True, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0}) #ndarray (5,2)
                
            elif isinstance(existing_ax, (list, tuple)):
                ## passed axes were a list of axes
                assert len(existing_ax) >= (needed_rows * fixed_columns)
                axs = existing_ax
                fig = axs[0].get_figure()
            elif isinstance(existing_ax, NDArray):
                ## passed axes were a list of axes
                assert np.size(existing_ax) >= (needed_rows * fixed_columns)
                axs = existing_ax
                axs = np.atleast_2d(axs)
                fig = axs[0][0].get_figure() ## get first axis to get the figure

            else:
                ## use the existing axes to plot the subaxes on                
                print(f'using subaxes on the existing axes')
                assert axes_inset_locators_list is not None
                
                fig = existing_ax.get_figure()
                ## convert to relative??
                
                axs = [] ## list
                # for curr_row, a_row_list in enumerate(self.row_column_indicies):
                a_linear_index = 0
                for curr_row in np.arange(needed_rows):
                    a_new_axs_list = []
                    # for curr_col, an_element in enumerate(a_row_list):
                    for curr_col in np.arange(fixed_columns):
                        # Add subaxes at [left, bottom, width, height] in normalized parent coordinates
                        # ax_inset = existing_ax.add_axes([0.2, 0.6, 0.3, 0.3])  # Positioned at 20% left, 60% bottom
                        ax_inset_location = axes_inset_locators_list[a_linear_index]
                        
                        try:
                            ax_inset = existing_ax.inset_axes(ax_inset_location, transform=existing_ax.transData, borderpad=0) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`
                        except AttributeError as e:
                            # AttributeError: Axes.set() got an unexpected keyword argument 'borderpad'
                            ax_inset = existing_ax.inset_axes(ax_inset_location, transform=existing_ax.transData) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`                        
                        except Exception as e:
                            raise
                        
                        a_new_axs_list.append(ax_inset) 
                        a_linear_index += 1 ## increment

                    ## accumulate the lists
                    axs.append(a_new_axs_list)        

                for a_linear_index in linear_plotter_indicies:
                    curr_row = row_column_indicies[0][a_linear_index]
                    curr_col = row_column_indicies[1][a_linear_index]
                    ## format the titles
                    an_ax = axs[curr_row][curr_col]
                    

            axs = np.atleast_2d(axs)
            # mp.set_size_inches(18.5, 26.5)

            background_track_shadings = {}
            for a_linear_index in linear_plotter_indicies:
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                ## format the titles
                an_ax = axs[curr_row][curr_col]
                an_ax.set_xticks([])
                an_ax.set_yticks([])
                
                if not use_theoretical_tracks_instead:
                    background_track_shadings[a_linear_index] = an_ax.plot(linear_plot_data[a_linear_index][0,:], linear_plot_data[a_linear_index][1,:], c='k', alpha=track_background_opacity)
                else:
                    # active_config = curr_active_pipeline.sess.config
                    background_track_shadings[a_linear_index] = _perform_plot_matplotlib_2D_tracks(long_track_inst=long_track_inst, short_track_inst=short_track_inst, ax=an_ax, rotate_to_vertical=self.rotate_to_vertical, track_background_opacity=track_background_opacity)
                
            return fig, axs, linear_plotter_indicies, row_column_indicies, background_track_shadings
        


        def _subfn_add_specific_epoch_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_epochs_ids, epochs_position_traces, epochs_time_ranges, active_plot_mode: str ='time_gradient', **plot_traj_kwargs):
            """ captures: cmap, should_include_trajectory_arrows, override_title_formatter_fn
            """
            # Add the lap trajectory:
            for a_linear_index in linear_plotter_indicies:
                curr_epoch_id = active_page_epochs_ids[a_linear_index]
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                curr_lap_time_range = epochs_time_ranges[curr_epoch_id]
                # if 'override_title_formatter_fn' in plot_traj_kwargs:
                if override_title_formatter_fn is not None:
                    curr_lap_label_text = override_title_formatter_fn(curr_epoch_id)
                else:
                    curr_lap_label_text = 'Epoch[{}]: t({:.2f}, {:.2f})'.format(curr_epoch_id, curr_lap_time_range[0], curr_lap_time_range[1])

                curr_lap_num_points = len(epochs_position_traces[curr_epoch_id][0,:])
                valid_plotting_modes: List[str] = ['time_gradient', 'line', 'scatter']
                # if use_time_gradient_line:
                if active_plot_mode == 'time_gradient':
                    curr_epoch_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(epochs_position_traces[curr_epoch_id][0,:]))
                    line_start_lw = plot_traj_kwargs.get('line_start_lw', 0.3)
                    line_end_lw = plot_traj_kwargs.get('line_end_lw', 1.0)
                    _alpha = plot_traj_kwargs.get('alpha', 0.85)
                    if trajectory_line_color_scheme == RenderColoringMode.ANGLE:
                        a_line, _ = DecodedTrajectoryMatplotlibPlotter._helper_add_gradient_line(axs[curr_row][curr_col], t=curr_epoch_timeseries, x=epochs_position_traces[curr_epoch_id][0,:], y=epochs_position_traces[curr_epoch_id][1,:], add_markers=False, line_color_scheme=RenderColoringMode.ANGLE, line_start_lw=line_start_lw, line_end_lw=line_end_lw, alpha=_alpha)
                    else:
                        norm = plt.Normalize(curr_epoch_timeseries.min(), curr_epoch_timeseries.max())
                        points = np.array([epochs_position_traces[curr_epoch_id][0,:], epochs_position_traces[curr_epoch_id][1,:]]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        n_segments = len(segments)
                        linewidths = np.linspace(line_start_lw, line_end_lw, n_segments) if n_segments else np.array([line_end_lw])
                        lc = LineCollection(segments, cmap=cmap, norm=norm)
                        lc.set_linewidth(linewidths)
                        lc.set_array(curr_epoch_timeseries)
                        lc.set_alpha(_alpha)
                        a_line = axs[curr_row][curr_col].add_collection(lc)

                    if should_include_trajectory_arrows:
                        ## try to add some arrow markers, might be very bad performance
                        _arrow_kwargs = {k: v for k, v in arrow_concentration_kwargs.items() if k != 'arrow_opacity'}
                        _out_markers = DecodedTrajectoryMatplotlibPlotter._helper_add_concentrated_arrows_to_line(ax=axs[curr_row][curr_col], 
                            t=curr_epoch_timeseries, # np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
                            x=epochs_position_traces[curr_epoch_id][0,:],
                            y=epochs_position_traces[curr_epoch_id][1,:], 
                            speed=None,
                            **_arrow_kwargs
                        )
                        _line_alpha = plot_traj_kwargs.get('alpha', 0.85)
                        _arrow_alpha = arrow_concentration_kwargs.get('arrow_opacity', _line_alpha)
                        for _ar in _out_markers.values():
                            _ar.set_alpha(_arrow_alpha)
                    


                    # add_arrow(line)
                elif active_plot_mode == 'line':
                    if 'c' not in plot_traj_kwargs:
                        plot_traj_kwargs['c'] = 'k'
                    if 'alpha' not in plot_traj_kwargs:
                        plot_traj_kwargs['alpha'] = 0.85
                    a_line = axs[curr_row][curr_col].plot(epochs_position_traces[curr_epoch_id][0,:], epochs_position_traces[curr_epoch_id][1,:], **plot_traj_kwargs)
                    # curr_lap_endpoint = curr_lap_position_traces[curr_lap_id][:,-1].T
                    a_start_arrow = _plot_helper_add_arrow(a_line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                    a_middle_arrow = _plot_helper_add_arrow(a_line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                    a_end_arrow = _plot_helper_add_arrow(a_line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                    # add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                    # add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')

                elif active_plot_mode == 'scatter':
                    if 'c' not in plot_traj_kwargs:
                        plot_traj_kwargs['c'] = 'k'
                    if 'alpha' not in plot_traj_kwargs:
                        plot_traj_kwargs['alpha'] = 0.85
                    a_scatter = axs[curr_row][curr_col].scatter(epochs_position_traces[curr_epoch_id][0,:], epochs_position_traces[curr_epoch_id][1,:], **plot_traj_kwargs)

                else:
                    raise NotImplementedError(f'unexpected plotting mode: plot_mode: "{active_plot_mode}", valid options: {valid_plotting_modes}')                    

                # add lap text label
                # Position text above the axes, centered horizontally, using axes coordinates (0-1)
                a_lap_label_text = axs[curr_row][curr_col].text(0.5, 1.02, curr_lap_label_text, horizontalalignment='center', verticalalignment='bottom', size=6, transform=axs[curr_row][curr_col].transAxes)
                # PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

        def _subfn_extract_posterior_and_extent(posterior_item):
            if isinstance(posterior_item, tuple) and (len(posterior_item) == 2):
                return posterior_item[0], posterior_item[1]
            return posterior_item, None

        def _subfn_add_posterior_overlay(ax, posterior_item, default_extent=None, alpha=None, posterior_cmap='gray', posterior_masking_value: float = 0.0025, should_perform_reshape: bool = True):
            """ captures: _active_posterior_plot_fn 
                        # Delegate the posterior plotting functionality.

            """
            if posterior_item is None:
                return None
            
            posterior_data, posterior_extent = _subfn_extract_posterior_and_extent(posterior_item)
            if posterior_data is None:
                return None
            if posterior_extent is None:
                posterior_extent = default_extent
            xbin_centers = self.xbin_centers if (self.xbin_centers is not None) else self.xbin
            ybin_centers = self.ybin_centers if (self.ybin_centers is not None) else self.ybin
            full_posterior_opacity = 1.0 if alpha is None else alpha
            
            # Handle 2D merged posterior (time-collapsed) as a single 2D image
            if (ybin_centers is not None) and (np.ndim(posterior_data) == 2):
                # Direct 2D plotting for merged posteriors
                # Use helper again:
                heatmaps, image_extent, plots_data = _active_posterior_plot_fn(ax, xbin_centers=xbin_centers, ybin_centers=ybin_centers, a_time_bin_centers=None, a_p_x_given_n=posterior_data, rotate_to_vertical=self.rotate_to_vertical, debug_print=False, posterior_masking_value=posterior_masking_value, full_posterior_opacity=full_posterior_opacity, custom_image_extent=posterior_extent, time_cmap=posterior_cmap, should_perform_reshape=should_perform_reshape, extant_plot_data=None)
                return heatmaps, image_extent, plots_data

            else:
                # Use helper for 3D (time-series) or 1D cases
                heatmaps, image_extent, plots_data = _active_posterior_plot_fn(ax, xbin_centers=xbin_centers, ybin_centers=ybin_centers, a_time_bin_centers=None, a_p_x_given_n=posterior_data, rotate_to_vertical=self.rotate_to_vertical, debug_print=False, posterior_masking_value=posterior_masking_value, full_posterior_opacity=full_posterior_opacity, custom_image_extent=posterior_extent, time_cmap=posterior_cmap, should_perform_reshape=should_perform_reshape, extant_plot_data=None)
                return heatmaps, image_extent, plots_data

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Compute required data from session:
        override_rotate_to_vertical: bool = kwargs.pop('override_rotate_to_vertical', None)
        if override_rotate_to_vertical:
            self.rotate_to_vertical = override_rotate_to_vertical
            print(f'override_rotate_to_vertical: {override_rotate_to_vertical} so overriding self.rotate_to_Vertical')

        if self.rotate_to_vertical:
            # vertical
            # x_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("x")]
            # y_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("y")]

            for a_df in epoch_specific_position_dfs:
                a_df['x_temp'] = deepcopy(a_df['x'])
                a_df['x'] = deepcopy(a_df['y'])
                a_df['y'] = deepcopy(a_df['x_temp'])
                # a_df[['x', 'y']] = a_df[['y', 'x']] ## swap the columns order
                
            curr_position_df[['x', 'y']] = curr_position_df[['y', 'x']] ## swap the columns order
            curr_position_df[['x_smooth', 'y_smooth']] = curr_position_df[['y_smooth', 'x_smooth']] ## swap the columns order

            # print(x_columns)

            # laps_position_traces_list = [lap_pos_df[position_col_names].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
            # lap_specific_position_dfs[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order
            
            # lap_specific_position_dfs[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order
            # curr_position_df[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order
        ## END if self.rotate_to_vertical

        epochs_position_traces_list = [epoch_pos_df[['x','y']].to_numpy().T for epoch_pos_df in epoch_specific_position_dfs]
        epochs_time_range_list = [[epoch_pos_df[['t']].to_numpy()[0].item(), epoch_pos_df[['t']].to_numpy()[-1].item()] for epoch_pos_df in epoch_specific_position_dfs]
        
        ## OUTPUTS: epoch_ids, epochs_time_range_list, epochs_position_traces_list, curr_position_df

        # lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in session.laps.lap_id]

        position_col_names = ['x', 'y']
        # epochs_position_traces_list = [lap_pos_df[position_col_names].to_numpy().T for lap_pos_df in epoch_specific_position_dfs]
        # laps_time_range_list = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in epoch_specific_position_dfs]
        # epoch_time_ranges = dict(zip(sess.laps.lap_id, laps_time_range_list))
        # epoch_position_traces = dict(zip(sess.laps.lap_id, epochs_position_traces_list)) ## each lap indexed by lap_id


        ## INPUTS: epoch_ids, epochs_time_range_list, epochs_position_traces_list, curr_position_df
        # num_laps = len(epoch_ids)
        if epoch_ids is None:
            epoch_ids = np.arange(len(epochs_time_range_list))
            
        valid_only_epoch_ids = [v for v in epoch_ids if v > -1] # epoch_ids: array([ 0,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]) -- here only the first 2 are valid
        num_valid_epochs: int = len(valid_only_epoch_ids) ## exclude the -1 entries
        
        # linear_lap_index = np.arange(num_laps)
        epochs_time_ranges = dict(zip(epoch_ids, epochs_time_range_list))
        epochs_position_traces = dict(zip(epoch_ids, epochs_position_traces_list))

        all_maze_positions = curr_position_df[position_col_names].to_numpy().T # (2, 59308)
        # np.shape(all_maze_positions)
        all_maze_data = [all_maze_positions for i in np.arange(curr_num_subplots)] # repeat the maze data for each subplot. (2, 593080)
        
        # Build Figures/Axes/Etc _____________________________________________________________________________________________ #
        self.fig, self.axs, self.linear_plotter_indicies, self.row_column_indicies, background_track_shadings = _subfn_build_epochs_multiplotter(curr_num_subplots, all_maze_data)
        perform_update_title_subtitle(fig=self.fig, ax=None, title_string="DecodedTrajectoryMatplotlibPlotter - plot_decoded_trajectories_2d") # , subtitle_string="TEST - SUBTITLE"
        
        # generate the pages
        epochs_pages = [list(chunk) for chunk in _subfn_chunks(epoch_ids, curr_num_subplots)] ## this is specific to actual laps...
        active_page_epochs_ids = epochs_pages[active_page_index] if (epochs_pages is not None) and (len(epochs_pages) > 0) else []
        

        # Handle psoterior plottings _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if posteriors is None:
            posteriors = kwargs.pop('posteriors', None)

        posterior_alpha = kwargs.pop('posterior_alpha', None)
        posterior_cmap = kwargs.pop('posterior_cmap', 'gray')
        posterior_masking_value = kwargs.pop('posterior_masking_value', 1e-12)
        posterior_should_perform_reshape = kwargs.pop('posterior_should_perform_reshape', True)

        if posteriors is not None:
            if isinstance(posteriors, dict):
                posteriors_by_epoch_id = posteriors
            elif isinstance(posteriors, (list, tuple)) and (len(posteriors) == len(epoch_ids)):
                posteriors_by_epoch_id = dict(zip(epoch_ids, posteriors))
            elif isinstance(posteriors, np.ndarray):
                if np.ndim(posteriors) == 2:
                    ## single posterior for all epochs, duplicate it
                    posteriors_by_epoch_id = {epoch_id:posteriors for epoch_id in epoch_ids}
                    
                elif (np.ndim(posteriors) >= 3) and (len(posteriors) == len(epoch_ids)):
                    posteriors_by_epoch_id = dict(zip(epoch_ids, list(posteriors)))
                else:
                    raise ValueError(f'np.shape(posteriors): {np.shape(posteriors)} is not supported')
            
            else:
                posteriors_by_epoch_id = None
                
            for a_linear_index in self.linear_plotter_indicies:
                if a_linear_index >= len(active_page_epochs_ids):
                    continue
                curr_row = self.row_column_indicies[0][a_linear_index]
                curr_col = self.row_column_indicies[1][a_linear_index]
                curr_epoch_id = active_page_epochs_ids[a_linear_index]
                curr_posterior = (posteriors_by_epoch_id or {}).get(curr_epoch_id, None)
                an_ax = self.axs[curr_row][curr_col]
                if (curr_posterior is None):
                    _subfn_add_posterior_overlay(an_ax, posteriors, default_extent=None, alpha=posterior_alpha, posterior_cmap=posterior_cmap, posterior_masking_value=posterior_masking_value, should_perform_reshape=posterior_should_perform_reshape)
                else:
                    _subfn_add_posterior_overlay(an_ax, curr_posterior, default_extent=None, alpha=posterior_alpha, posterior_cmap=posterior_cmap, posterior_masking_value=posterior_masking_value, should_perform_reshape=posterior_should_perform_reshape)
         
        

        if plot_actual_lap_lines:
            ## IDK what this is sadly, i think it's a reminant of the lap plotter?
            _out_objs = _subfn_add_specific_epoch_trajectory(self.fig, self.axs, linear_plotter_indicies=self.linear_plotter_indicies, row_column_indicies=self.row_column_indicies, active_page_epochs_ids=active_page_epochs_ids, epochs_position_traces=epochs_position_traces, epochs_time_ranges=epochs_time_ranges, active_plot_mode=plot_mode, **{**kwargs, 'alpha': line_opacity})
            # plt.ylim((125, 152))
        else:
            _out_objs = None

        self.epochs_pages = epochs_pages

        ## Build artist holders:
        # MatplotlibRenderPlots
        self.plots_data_dict_array = []
        self.artist_dict_array = [] ## list
        for a_list in self.row_column_indicies:
            a_new_artists_list = []
            a_new_plot_data_list = []
            for an_element in a_list:
                a_new_artists_list.append({'prev_heatmaps': [], 'lines': {}, 'markers': {}}) ## make a new empty dict for each element
                a_new_plot_data_list.append(RenderPlotsData(f"DecodedTrajectoryMatplotlibPlotter.plot_decoded_trajectories_2d", image_extent=None))
            ## accumulate the lists
            self.plots_data_dict_array.append(a_new_plot_data_list)
            self.artist_dict_array.append(a_new_artists_list)                
        ## Access via ` self.artist_dict_array[curr_row][curr_col]`, same as the axes

        # for a_linear_index in self.linear_plotter_indicies:
        #     curr_row = self.row_column_indicies[0][a_linear_index]
        #     curr_col = self.row_column_indicies[1][a_linear_index]
            #   curr_artist_dict = self.artist_dict_array[curr_row][curr_col]

        return self.fig, self.axs, epochs_pages


    @function_attributes(short_name=None, tags=['main', 'factored-out'], input_requires=[], output_provides=[], uses=['self.plot_decoded_trajectories_2d'], used_by=[], creation_date='2025-12-22 13:33', related_items=[])
    def plot_decoded_laps_2d(self, sess, *args, **kwargs):
        """ Helper function that plots specifically the laps
        """
        curr_position_df, epoch_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(sess)
        epoch_ids = deepcopy(sess.laps.lap_id)
        if kwargs.get('use_theoretical_tracks_instead', False):
            ## need to pass sess
            kwargs['sess'] = sess

        return self.plot_decoded_trajectories_2d(curr_position_df=curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=epoch_ids, *args, **kwargs)
        


# ==================================================================================================================================================================================================================================================================================== #
# PyVista/3D                                                                                                                                                                                                                                                                           #
# ==================================================================================================================================================================================================================================================================================== #

@define(slots=False, eq=False)
class DecodedTrajectoryPyVistaPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded trajectory (path) using pyvista in 3D. 
    
    Usage:
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryPyVistaPlotter
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

    
    curr_active_pipeline.prepare_for_display()
    _out = curr_active_pipeline.display(display_function='_display_3d_interactive_custom_data_explorer', active_session_configuration_context=global_epoch_context,
                                        params_kwargs=dict(should_use_linear_track_geometry=True, **{'t_start': t_start, 't_delta': t_delta, 't_end': t_end}),
                                        )
    iplapsDataExplorer: InteractiveCustomDataExplorer = _out['iplapsDataExplorer']
    pActiveInteractiveLapsPlotter = _out['plotter']
    a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=iplapsDataExplorer.p)
    a_decoded_trajectory_pyvista_plotter.build_ui()

    """
    p: Any = field(default=None)
    curr_time_bin_index: int = field(default=0)
    enable_point_labels: bool = field(default=False)
    enable_plot_all_time_bins_in_epoch_mode: bool = field(default=False)

    slider_epoch = field(default=None)
    slider_epoch_time_bin = field(default=None)
    slider_epoch_time_bin_playback_checkbox = field(default=None)
    
    # Qt slider widgets
    qt_slider_epoch: Optional[QtWidgets.QSlider] = field(default=None)
    qt_slider_epoch_time_bin: Optional[QtWidgets.QSlider] = field(default=None)
    qt_slider_epoch_label: Optional[QtWidgets.QLabel] = field(default=None)
    qt_slider_timebin_label: Optional[QtWidgets.QLabel] = field(default=None)
    qt_playback_checkbox: Optional[QtWidgets.QCheckBox] = field(default=None)
    qt_slider_bar_widget: Optional[QtWidgets.QWidget] = field(default=None)
    
    interactive_plotter: PhoInteractivePlotter = field(default=None)
    plotActors = field(default=None)
    data_dict = field(default=None)
    plotActors_CenterLabels = field(default=None)
    data_dict_CenterLabels = field(default=None)

    active_plot_fn: Callable = field(default=plot_3d_binned_bars) # like [plot_3d_binned_bars, plot_3d_stem_points]
    animation_callback_interval_ms: int = field(default=200) # 200ms per time bin

    # Peak prominence fields
    peak_prominence_result: Optional["PosteriorPeaksPeakProminence2dResult"] = field(default=None, repr=False)
    peak_prominence_actors = field(default=None, repr=False)
    peak_prominence_data = field(default=None, repr=False)
    peak_prominence_kwargs: Dict[str, Any] = field(default=Factory(dict), repr=False)

    # Callback blocking and execution guards to prevent freezing
    _updating_slider_programmatically: bool = field(default=False, init=False, repr=False)
    _update_in_progress: bool = field(default=False, init=False, repr=False)


    def build_ui(self):
        """ builds the Qt slider widgets in a bar at the bottom of the window
        """

        assert self.p is not None
        if self.curr_epoch_idx is None:
            self.curr_epoch_idx = 0
        
        # Build Qt slider bar instead of PyVista sliders
        self._build_qt_slider_bar()
        
        # Note: Interactive plotter is no longer needed for VTK sliders, but we keep it for compatibility
        # Playback is now handled directly by the Qt checkbox and timer


    def _build_qt_slider_bar(self):
        """Builds a Qt slider bar at the bottom of the plotter window with epoch and timebin sliders plus playback checkbox."""
        assert self.p is not None
        
        # Check if plotter has app_window (BackgroundPlotter from pyvistaqt)
        if not hasattr(self.p, 'app_window') or self.p.app_window is None:
            # Fallback: try to get window from plotter
            print("Warning: Plotter does not have app_window attribute. Qt sliders cannot be created.")
            return
        
        app_window = self.p.app_window
        
        # Get or create the slider bar widget
        if self.qt_slider_bar_widget is None:
            # Create a horizontal widget bar
            self.qt_slider_bar_widget = QtWidgets.QWidget()
            slider_layout = QtWidgets.QHBoxLayout(self.qt_slider_bar_widget)
            slider_layout.setContentsMargins(10, 5, 10, 5)
            slider_layout.setSpacing(10)
            
            # Set fixed height for the bar
            self.qt_slider_bar_widget.setFixedHeight(45)
            
            # Epoch slider section
            epoch_label = QtWidgets.QLabel("Epoch Idx:")
            epoch_label.setMinimumWidth(70)
            slider_layout.addWidget(epoch_label)
            
            self.qt_slider_epoch = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.qt_slider_epoch.setMinimum(0)
            self.qt_slider_epoch.setMaximum(max(0, self.num_filter_epochs - 1))
            self.qt_slider_epoch.setValue(0)
            self.qt_slider_epoch.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.qt_slider_epoch.setTickInterval(1)
            slider_layout.addWidget(self.qt_slider_epoch, stretch=1)
            
            self.qt_slider_epoch_label = QtWidgets.QLabel("0")
            self.qt_slider_epoch_label.setMinimumWidth(30)
            self.qt_slider_epoch_label.setAlignment(QtCore.Qt.AlignCenter)
            slider_layout.addWidget(self.qt_slider_epoch_label)
            
            # Add spacing
            slider_layout.addSpacing(20)
            
            # Timebin slider section (only if not in plot_all_time_bins mode)
            if not self.enable_plot_all_time_bins_in_epoch_mode:
                timebin_label = QtWidgets.QLabel("Timebin IDX:")
                timebin_label.setMinimumWidth(80)
                slider_layout.addWidget(timebin_label)
                
                self.qt_slider_epoch_time_bin = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.qt_slider_epoch_time_bin.setMinimum(0)
                curr_num_epoch_time_bins = self.curr_n_time_bins if self.curr_n_time_bins is not None else 0
                self.qt_slider_epoch_time_bin.setMaximum(max(0, curr_num_epoch_time_bins - 1))
                self.qt_slider_epoch_time_bin.setValue(0)
                self.qt_slider_epoch_time_bin.setTickPosition(QtWidgets.QSlider.TicksBelow)
                self.qt_slider_epoch_time_bin.setTickInterval(1)
                slider_layout.addWidget(self.qt_slider_epoch_time_bin, stretch=1)
                
                self.qt_slider_timebin_label = QtWidgets.QLabel("0")
                self.qt_slider_timebin_label.setMinimumWidth(30)
                self.qt_slider_timebin_label.setAlignment(QtCore.Qt.AlignCenter)
                slider_layout.addWidget(self.qt_slider_timebin_label)
                
                # Add spacing
                slider_layout.addSpacing(20)
                
                # Playback checkbox
                self.qt_playback_checkbox = QtWidgets.QCheckBox("Playback")
                slider_layout.addWidget(self.qt_playback_checkbox)
            
            # Add the slider bar to the window
            # Get the central widget (should exist for BackgroundPlotter)
            central_widget = app_window.centralWidget()
            if central_widget is None:
                # Create a central widget if it doesn't exist
                central_widget = QtWidgets.QWidget()
                app_window.setCentralWidget(central_widget)
            
            # Get or create the main layout
            main_layout = central_widget.layout()
            if main_layout is None:
                # No layout exists, create one
                main_layout = QtWidgets.QVBoxLayout(central_widget)
                main_layout.setContentsMargins(0, 0, 0, 0)
                main_layout.setSpacing(0)
                
                # The render widget should already be a child of central_widget
                # Add all existing widgets to the layout (except our slider bar)
                for child in central_widget.children():
                    if isinstance(child, QtWidgets.QWidget) and child != self.qt_slider_bar_widget:
                        # Remove from parent and add to layout
                        child.setParent(None)
                        main_layout.addWidget(child, stretch=1)
            
            # Check if slider bar is already in the layout
            if self.qt_slider_bar_widget.parent() != central_widget or main_layout.indexOf(self.qt_slider_bar_widget) == -1:
                # Add slider bar at the bottom (no stretch, fixed height)
                main_layout.addWidget(self.qt_slider_bar_widget)
            
            # Connect signals
            self._connect_qt_slider_signals()
        
        # Update slider ranges if they've changed
        self._update_qt_slider_ranges()
    
    def _connect_qt_slider_signals(self):
        """Connect Qt slider signals to callback methods."""
        if self.qt_slider_epoch is not None:
            # Use a wrapper to maintain the same callback logic
            def _on_qt_slider_epoch_changed(value):
                if not hasattr(_on_qt_slider_epoch_changed, "last_value"):
                    _on_qt_slider_epoch_changed.last_value = value
                if value != _on_qt_slider_epoch_changed.last_value:
                    self.on_update_slider_epoch_idx(int(value))
                    _on_qt_slider_epoch_changed.last_value = value
                    # Update label
                    if self.qt_slider_epoch_label is not None:
                        self.qt_slider_epoch_label.setText(str(value))
            
            self.qt_slider_epoch.valueChanged.connect(_on_qt_slider_epoch_changed)
            # Update label initially
            if self.qt_slider_epoch_label is not None:
                self.qt_slider_epoch_label.setText(str(self.qt_slider_epoch.value()))
        
        if self.qt_slider_epoch_time_bin is not None:
            def _on_qt_slider_timebin_changed(value):
                # Skip callback if programmatic update is in progress
                if self._updating_slider_programmatically:
                    return
                if not hasattr(_on_qt_slider_timebin_changed, "last_value"):
                    _on_qt_slider_timebin_changed.last_value = value
                if value != _on_qt_slider_timebin_changed.last_value:
                    self.on_update_slider_epoch_time_bin(int(value))
                    _on_qt_slider_timebin_changed.last_value = value
                    # Update label
                    if self.qt_slider_timebin_label is not None:
                        self.qt_slider_timebin_label.setText(str(value))
            
            self.qt_slider_epoch_time_bin.valueChanged.connect(_on_qt_slider_timebin_changed)
            # Update label initially
            if self.qt_slider_timebin_label is not None:
                self.qt_slider_timebin_label.setText(str(self.qt_slider_epoch_time_bin.value()))
        
        if self.qt_playback_checkbox is not None:
            self.qt_playback_checkbox.stateChanged.connect(self._on_playback_checkbox_changed)
    
    def _update_qt_slider_ranges(self):
        """Update Qt slider ranges based on current data."""
        if self.qt_slider_epoch is not None:
            max_epoch = max(0, self.num_filter_epochs - 1)
            self.qt_slider_epoch.setMaximum(max_epoch)
        
        if self.qt_slider_epoch_time_bin is not None and self.curr_n_time_bins is not None:
            max_timebin = max(0, self.curr_n_time_bins - 1)
            self._updating_slider_programmatically = True
            try:
                self.qt_slider_epoch_time_bin.setMaximum(max_timebin)
                self.qt_slider_epoch_time_bin.setValue(0)
                if self.qt_slider_timebin_label is not None:
                    self.qt_slider_timebin_label.setText("0")
            finally:
                self._updating_slider_programmatically = False
    
    def _on_playback_checkbox_changed(self, state):
        """Handle playback checkbox state changes."""
        is_checked = state == QtCore.Qt.Checked
        if self.interactive_plotter is not None:
            # Update the interactive plotter's animation state
            self.interactive_plotter.interface_properties.animation_state = is_checked
        # If interactive_plotter doesn't exist yet, we'll create it when needed
        # For now, we can implement basic playback functionality
        if is_checked and self.qt_slider_epoch_time_bin is not None:
            # Start playback timer
            if not hasattr(self, '_playback_timer'):
                self._playback_timer = QtCore.QTimer()
                self._playback_timer.timeout.connect(self._playback_step)
            self._playback_timer.start(self.animation_callback_interval_ms)
        else:
            # Stop playback
            if hasattr(self, '_playback_timer'):
                self._playback_timer.stop()
    
    def _playback_step(self):
        """Step forward in playback mode."""
        if self.qt_slider_epoch_time_bin is not None:
            current_value = self.qt_slider_epoch_time_bin.value()
            max_value = self.qt_slider_epoch_time_bin.maximum()
            if current_value < max_value:
                self._updating_slider_programmatically = True
                try:
                    self.qt_slider_epoch_time_bin.setValue(current_value + 1)
                finally:
                    self._updating_slider_programmatically = False
            else:
                # Reached end, stop playback
                if self.qt_playback_checkbox is not None:
                    self.qt_playback_checkbox.setChecked(False)


    def update_ui(self):
        """ called to update the epoch_time_bin slider when the epoch_index slider is changed. 
        """
        # Update Qt slider ranges and values
        self._update_qt_slider_ranges()


    def perform_programmatic_slider_epoch_update(self, value):
        """ called to programmatically update the epoch_idx slider. """
        if self.qt_slider_epoch is not None:
            print(f'updating slider_epoch index to : {int(value)}')
            self._updating_slider_programmatically = True
            try:
                self.qt_slider_epoch.setValue(int(value))
                if self.qt_slider_epoch_label is not None:
                    self.qt_slider_epoch_label.setText(str(int(value)))
            finally:
                self._updating_slider_programmatically = False
            self.on_update_slider_epoch_idx(value=int(value))
            print(f'\tdone.')

    def on_update_slider_epoch_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
        # Prevent nested execution to avoid freezing
        if self._update_in_progress:
            return
        self._update_in_progress = True
        try:
            # print(f'.on_update_slider_epoch(value: {value})')
            self.curr_epoch_idx = int(value) ## Update `curr_epoch_idx`
            if not self.enable_plot_all_time_bins_in_epoch_mode:
                self.curr_time_bin_index = 0 # change to 0
            else:
                ## otherwise default to a range
                self.curr_time_bin_index = np.arange(self.curr_n_time_bins)

            self.update_ui() # called to update the dependent time_bin slider

            if not self.enable_plot_all_time_bins_in_epoch_mode:
                self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
            else:
                ## otherwise default to a range
                self.perform_update_plot_epoch_time_bin_range(self.curr_time_bin_index)

            # Removed problematic double-update code that used potentially stale data_dict
            # The main update above already handles the plotting correctly
        finally:
            self._update_in_progress = False



    def on_update_slider_epoch_time_bin(self, value: int):
        """ called when the epoch_time_bin within a given epoch_idx slider changes 
        """
        # Prevent nested execution to avoid freezing
        if self._update_in_progress:
            return
        self._update_in_progress = True
        try:
            # print(f'.on_update_slider_epoch_time_bin(value: {value})')
            self.perform_update_plot_single_epoch_time_bin(value=value)
        finally:
            self._update_in_progress = False
        


    @function_attributes(short_name=None, tags=['main_plot_update', 'single_time_bin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:03', related_items=[])
    def perform_update_plot_single_epoch_time_bin(self, value: int):
        """ single-time-bin plotting:
        Note: This method is called from guarded entry points (on_update_slider_epoch_idx, on_update_slider_epoch_time_bin),
        so it doesn't need its own guard to prevent blocking legitimate nested calls.
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        self.curr_time_bin_index = int(value) # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=self.curr_time_bin_index)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_fn(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels, active_plot_fn=self.active_plot_fn)
        
        ## Render peak prominence if result is set:
        if self.peak_prominence_result is not None and self.data_dict is not None:
            # Get the posterior mesh from data_dict (first entry should contain 'grid')
            posterior_pdata = None
            for plot_name, plot_data in self.data_dict.items():
                if 'grid' in plot_data:
                    posterior_pdata = plot_data['grid']
                    break
            
            if posterior_pdata is not None:
                from pyphoplacecellanalysis.Pho3D.PyVista.peak_prominences import _render_posterior_peak_prominence_2d_results_on_pyvista_plotter
                
                # Get visibility of the posterior to match peak visibility
                posterior_is_visible = 1
                if self.plotActors is not None and len(self.plotActors) > 0:
                    first_actor_key = list(self.plotActors.keys())[0]
                    if 'main' in self.plotActors[first_actor_key]:
                        posterior_is_visible = self.plotActors[first_actor_key]['main'].GetVisibility()
                
                # Create a copy of kwargs without debug_print to avoid duplicate argument error
                peak_prominence_kwargs_copy = self.peak_prominence_kwargs.copy()
                peak_prominence_kwargs_copy.pop('debug_print', None)
                
                # multiplier_factor = an_extra_rendering_info.get('multiplier_factor', 1.0)
                
                all_peaks_data, all_peaks_actors = _render_posterior_peak_prominence_2d_results_on_pyvista_plotter(
                    self.p,
                    posterior_pdata,
                    self.peak_prominence_result,
                    self.curr_epoch_idx,
                    self.curr_time_bin_index,
                    render=False,
                    debug_print=self.peak_prominence_kwargs.get('debug_print', False),
                    **peak_prominence_kwargs_copy
                )
                
                self.peak_prominence_data = all_peaks_data
                self.peak_prominence_actors = all_peaks_actors
                
                # Set visibility to match posterior
                if self.peak_prominence_actors is not None:
                    self.peak_prominence_actors.SetVisibility(posterior_is_visible)
        

    @function_attributes(short_name=None, tags=['main_plot_update', 'multi_time_bins', 'epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:04', related_items=[])
    def perform_update_plot_epoch_time_bin_range(self, value: Optional[NDArray]=None):
        """ multi-time-bin plotting:
        Note: This method is called from guarded entry points (on_update_slider_epoch_idx),
        so it doesn't need its own guard to prevent blocking legitimate nested calls.
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        if value is None:
            value = np.arange(self.curr_n_time_bins)
        self.curr_time_bin_index = value # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=value)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_fn(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                time_bin_centers=a_time_bin_centers, posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels, active_plot_fn=self.active_plot_fn)


    def perform_clear_existing_decoded_trajectory_plots(self):
        """ Remove existing actors and clear data dictionaries to prevent stale state.
        Ensures proper cleanup before new actors are created.
        """
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import clear_3d_binned_bars_plots
        from pyphocorehelpers.gui.PyVista.CascadingDynamicPlotsList import CascadingDynamicPlotsList

        # Clear peak prominence actors first (before clearing posterior actors)
        if self.peak_prominence_actors is not None:
            # Remove all peak prominence actors from plotter
            if isinstance(self.peak_prominence_actors, CascadingDynamicPlotsList):
                # Iterate through all nested actors
                for category_name, category_actors in self.peak_prominence_actors.items():
                    if isinstance(category_actors, CascadingDynamicPlotsList):
                        for actor_name, actor in category_actors.items():
                            if actor is not None:
                                try:
                                    self.p.remove_actor(actor)
                                except Exception as e:
                                    pass  # Actor may already be removed
                    elif category_actors is not None:
                        try:
                            self.p.remove_actor(category_actors)
                        except Exception as e:
                            pass  # Actor may already be removed
            self.peak_prominence_actors = None
        
        # Clear peak prominence data
        if self.peak_prominence_data is not None:
            self.peak_prominence_data = None

        # Clear main plot actors
        if self.plotActors is not None:
            clear_3d_binned_bars_plots(p=self.p, plotActors=self.plotActors)
            self.plotActors.clear()
        
        # Clear data_dict to remove any stale update functions or references
        if self.data_dict is not None:
            # Explicitly clear any update functions stored in data_dict to prevent stale references
            self.data_dict.clear()

        # Remove center label actors from plotter before clearing dict
        if self.plotActors_CenterLabels is not None:
            # plotActors_CenterLabels has same structure as plotActors: dict with 'main' key
            for k, v in self.plotActors_CenterLabels.items():
                if isinstance(v, dict) and 'main' in v:
                    self.p.remove_actor(v['main'])
                elif v is not None:
                    # Handle case where v is directly an actor
                    self.p.remove_actor(v)
            self.plotActors_CenterLabels.clear()
        
        # Clear center labels data dict
        if self.data_dict_CenterLabels is not None:
            self.data_dict_CenterLabels.clear()


    def set_peak_prominence_result(self, peak_prominence_result: "PosteriorPeaksPeakProminence2dResult", promenence_plot_threshold: float = 0.2, included_level_indicies: List[int] = [1], include_contour_bounding_box: bool = False, include_text_labels: bool = False, active_curve_color: Optional[Tuple[float, float, float]] = None, debug_print: bool = False, **kwargs):
        """ Sets the peak prominence result and triggers re-render if plotter is already built.
        
        Args:
            peak_prominence_result: PosteriorPeaksPeakProminence2dResult object
            promenence_plot_threshold: Minimum prominence threshold for plotting
            included_level_indicies: List of level indices to include
            include_contour_bounding_box: Whether to include bounding boxes
            include_text_labels: Whether to include text labels
            active_curve_color: Color for contours/boxes/text (default: white)
            debug_print: Whether to print debug info
            **kwargs: Additional arguments passed to rendering functions
        """
        self.peak_prominence_result = peak_prominence_result
        self.peak_prominence_kwargs = dict(
            promenence_plot_threshold=promenence_plot_threshold,
            included_level_indicies=included_level_indicies,
            include_contour_bounding_box=include_contour_bounding_box,
            include_text_labels=include_text_labels,
            debug_print=debug_print,
            **kwargs
        )
        if active_curve_color is not None:
            self.peak_prominence_kwargs['active_curve_color'] = active_curve_color
        
        # Trigger re-render if plotter is already built and has data
        if self.p is not None and self.data_dict is not None and len(self.data_dict) > 0:
            # Re-render the current time bin to show peaks
            self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
            self.p.render()


    def get_curr_posterior(self, an_epoch_idx: int = 0, time_bin_index:Union[int, NDArray]=0):
        a_posterior_p_x_given_n, a_time_bin_centers = self._perform_get_curr_posterior(a_result=self.a_result, an_epoch_idx=an_epoch_idx, time_bin_index=time_bin_index)
        n_epoch_timebins: int = len(a_time_bin_centers)

        if np.ndim(a_posterior_p_x_given_n) > 2:
            assert np.ndim(a_posterior_p_x_given_n) == 3, f"np.ndim(a_posterior_p_x_given_n) should be either 2 or 3, but it is {np.ndim(a_posterior_p_x_given_n)}"
            n_xbins, n_ybins, actual_n_epoch_timebins = np.shape(a_posterior_p_x_given_n) # (5, 312)
            assert n_epoch_timebins == actual_n_epoch_timebins, f"n_epoch_timebins: {n_epoch_timebins} != actual_n_epoch_timebins: {actual_n_epoch_timebins} from np.shape(a_posterior_p_x_given_n) ({np.shape(a_posterior_p_x_given_n)})"
        else:
            a_posterior_p_x_given_n = np.atleast_2d(a_posterior_p_x_given_n) #.T # (57, 1) ## There was an error being induced by the transpose for non 1D matricies passed in. Transpose seems like it should only be done for the (N, 1) case.

            if np.shape(a_posterior_p_x_given_n)[0] == 1:
                a_posterior_p_x_given_n = a_posterior_p_x_given_n.T 

            required_n_y_bins: int = len(self.ybin_centers) # passing an arbitrary amount of y-bins? Currently it's 6, which I don't get. Oh, I guess that comes from the 2D decoder that's passed in.
            n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n) # (5, 312)

            ## for a 1D posterior
            if (n_ybins < required_n_y_bins) and (n_ybins == 1):
                print(f'building 2D plotting data from 1D posterior.')

                # fill solid across all y-bins
                a_posterior_p_x_given_n = np.tile(a_posterior_p_x_given_n, (1, required_n_y_bins)) # (57, 6)
                
                ## fill only middle 2 bins.
                # a_posterior_p_x_given_n = np.tile(a_posterior_p_x_given_n, (1, required_n_y_bins)) # (57, 6) start ny filling all

                # find middle bin:
                # mid_bin_idx = np.rint(float(required_n_y_bins) / 2.0)
                # a_posterior_p_x_given_n[:, 1:] = np.nan
                # a_posterior_p_x_given_n[:, 3:-1] = np.nan
                

                n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n) # update again with new matrix

        assert n_xbins == np.shape(self.xbin_centers)[0], f"n_xbins: {n_xbins} != np.shape(xbin_centers)[0]: {np.shape(self.xbin_centers)}"
        assert n_ybins == np.shape(self.ybin_centers)[0], f"n_ybins: {n_ybins} != np.shape(ybin_centers)[0]: {np.shape(self.ybin_centers)}"
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        return a_posterior_p_x_given_n, a_time_bin_centers
    

    @classmethod
    def _perform_get_curr_posterior(cls, a_result, an_epoch_idx: int = 0, time_bin_index: Union[int, NDArray]=0, desired_max_height: float = 50.0):
        """ gets the current posterior for the specified epoch_idx and time_bin_index within the epoch."""
        # a_result.time_bin_containers
        a_posterior_p_x_given_n_all_t = a_result.p_x_given_n_list[an_epoch_idx]
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        # a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx]
        a_most_likely_positions = a_result.most_likely_positions_list[an_epoch_idx]
        # a_time_bin_edges = a_result.time_bin_edges[an_epoch_idx]
        a_time_bin_centers = a_result.time_bin_containers[an_epoch_idx].centers
        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)
        assert len(a_time_bin_centers) == len(a_most_likely_positions), f"len(a_time_bin_centers): {len(a_time_bin_centers)} != len(a_most_likely_positions): {len(a_most_likely_positions)}"
        # print(f'np.shape(a_posterior_p_x_given_n): {np.shape(a_posterior_p_x_given_n)}') # : (58, 5, 312) - (n_xbins, n_ybins, n_epoch_timebins)
        # 

        min_v = np.nanmin(a_posterior_p_x_given_n_all_t)
        max_v = np.nanmax(a_posterior_p_x_given_n_all_t)
        # print(f'min_v: {min_v}, max_v: {max_v}')
        multiplier_factor: float = desired_max_height / (float(max_v) - float(min_v))
        # print(f'multiplier_factor: {multiplier_factor}')
        # extra_rendering_info = dict(min_v=min_v, max_v=max_v, multiplier_factor=multiplier_factor)


        ## get the specific time_bin_index posterior:
        if np.ndim(a_posterior_p_x_given_n_all_t) > 2:
            ## multiple time bins case (3D)
            # n_xbins, n_ybins, n_epoch_timebins = np.shape(a_posterior_p_x_given_n_all_t)
            a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, :, time_bin_index])
        else:
            ## single time bin case (2D)
            # n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n_all_t) ???
            a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, time_bin_index])
        a_posterior_p_x_given_n = a_posterior_p_x_given_n * multiplier_factor # multiply by the desired multiplier factor
        return a_posterior_p_x_given_n, a_time_bin_centers



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@metadata_attributes(short_name=None, tags=['pyvista', 'mixin', 'decoder', '3D', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-27 14:38', related_items=['DecodedTrajectoryPyVistaPlotter'])
class DecoderRenderingPyVistaMixin:
    """ Implementors render decoded positions and decoder info with PyVista 
    
    Requires:
        self.params
        
    Provides:
    
        Adds:
            ... More?
            
            
    Known Uses:
        InteractivePlaceCellTuningCurvesDataExplorer
    """

    def add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, debug_print=False):
        """ Adds a red position indicator callback for the current decoded position

        Usage:
            active_one_step_decoder = global_results.pf2D_Decoder
            _update_nearest_decoded_most_likely_position_callback, _conn = add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, _debug_print = False)

        """
        def _update_nearest_decoded_most_likely_position_callback(start_t, end_t):
            """ Only uses end_t
            Implicitly captures: self, _get_nearest_decoded_most_likely_position_callback
            
            Usage:
                _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0])
                _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

            """
            def _get_nearest_decoded_most_likely_position_callback(t):
                """ A callback that when passed a visualization timestamp (the current time to render) returns the most likely predicted position provided by the active_two_step_decoder
                Implicitly captures:
                    active_one_step_decoder, active_two_step_decoder
                Usage:
                    _get_nearest_decoded_most_likely_position_callback(9000.1)
                """
                active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
                active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
                # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
                assert np.shape(active_time_window_variable)[0] == np.shape(active_most_likely_positions)[1], f"timestamps and num positions must be the same but np.shape(active_time_window_variable): {np.shape(active_time_window_variable)} and np.shape(active_most_likely_positions): {np.shape(active_most_likely_positions)}!"
                last_window_index = np.searchsorted(active_time_window_variable, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
                # TODO: CORRECTNESS: why is it returning an index that corresponds to a time later than the current time?
                # for current time t=9000.0
                #     last_window_index: 1577
                #     last_window_time: 9000.5023
                # EH: close enough
                last_window_time = active_time_window_variable[last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
                displayed_time_offset = t - last_window_time # negative value if the window time being displayed is in the future
                if debug_print:
                    print(f'for current time t={t}\n\tlast_window_index: {last_window_index}\n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}')
                return (last_window_time, *list(np.squeeze(active_most_likely_positions[:, last_window_index]).copy()))

            t = end_t # the t under consideration should always be the end_t. This is written this way just for compatibility with the self.sigOnUpdateMeshes (float, float) signature
            curr_t, curr_x, curr_y = _get_nearest_decoded_most_likely_position_callback(t)
            curr_debug_point = [curr_x, curr_y, self.z_fixed[-1]]
            if debug_print:
                print(f'tcurr_debug_point: {curr_debug_point}') # \n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}
            self.perform_plot_location_point('decoded_position_point_plot', curr_debug_point, color='r', render=True)
            return curr_debug_point

        _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0]) # initialize by calling the callback with the current time
        # _conn = pg.SignalProxy(self.sigOnUpdateMeshes, rateLimit=14, slot=_update_nearest_decoded_most_likely_position_callback)
        _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

        # TODO: need to assign these results to somewhere in self. Not sure if I need to retain a reference to `active_one_step_decoder`
        # self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor']

        return _update_nearest_decoded_most_likely_position_callback, _conn # return the callback and the connection

    
    @property
    def decoded_trajectory_pyvista_plotter(self) -> DecodedTrajectoryPyVistaPlotter:
        """The decoded_trajectory_pyvista_plotter property."""
        return self.params['decoded_trajectory_pyvista_plotter']


    @function_attributes(short_name=None, tags=['probability'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryPyVistaPlotter'], used_by=[], creation_date='2025-01-29 07:35', related_items=[])
    def add_decoded_posterior_bars(self, a_result: DecodedFilterEpochsResult, xbin: NDArray, xbin_centers: NDArray, ybin: Optional[NDArray], ybin_centers: Optional[NDArray], enable_plot_all_time_bins_in_epoch_mode:bool=True, active_plot_fn=None) -> "DecodedTrajectoryPyVistaPlotter":
        """ adds the decoded posterior to the PyVista plotter
         
          
        Usage:

            a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = iplapsDataExplorer.add_decoded_posterior_bars(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)

        """
        
        a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=self.p, curr_epoch_idx=0, curr_time_bin_index=0, enable_plot_all_time_bins_in_epoch_mode=enable_plot_all_time_bins_in_epoch_mode,
                                                                                                                active_plot_fn=active_plot_fn)
        a_decoded_trajectory_pyvista_plotter.build_ui()
        self.params['decoded_trajectory_pyvista_plotter'] = a_decoded_trajectory_pyvista_plotter
        return a_decoded_trajectory_pyvista_plotter
    

    def clear_all_added_decoded_posterior_plots(self, clear_ui_elements_also: bool = False):
        """ clears the plotted posterior actors and optionally the control sliders
        
        """
        if ('decoded_trajectory_pyvista_plotter' in self.params) and (self.decoded_trajectory_pyvista_plotter is not None):
            self.decoded_trajectory_pyvista_plotter.perform_clear_existing_decoded_trajectory_plots()
            
            ## can remove the UI (sliders and such) via:
            if clear_ui_elements_also:
                if self.decoded_trajectory_pyvista_plotter.slider_epoch is not None:
                    self.decoded_trajectory_pyvista_plotter.slider_epoch.RemoveAllObservers()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch.Off()
                    # a_decoded_trajectory_pyvista_plotter.slider_epoch.FastDelete()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch = None


                if self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin is not None:
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.RemoveAllObservers()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.Off()
                    # a_decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.FastDelete()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin = None
                    

                self.decoded_trajectory_pyvista_plotter.p.clear_slider_widgets()

            self.decoded_trajectory_pyvista_plotter.p.update()
            self.decoded_trajectory_pyvista_plotter.p.render()



    @classmethod
    def perform_plot_posterior_fn(
        cls, p, xbin, ybin, xbin_centers, ybin_centers, posterior_p_x_given_n, time_bin_centers=None,
        enable_point_labels: bool = True, point_labeling_function=None, point_masking_function=None,
        posterior_name='P_x_given_n', active_plot_fn=None, **kwargs
    ):
        """ called to perform the mesh generation and add_mesh calls
        
        Looks like it switches between 3 different potential plotting functions, all imported directly below

        ## Defaults to `plot_3d_binned_bars` if nothing else is provided        
        
        """
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_3d_smooth_mesh, plot_point_labels

        drop_below_threshold = kwargs.pop('drop_below_threshold', None)
        opacity = kwargs.pop('opacity', 0.75)

        if active_plot_fn is None:
            ## Defaults to `plot_3d_binned_bars` if nothing else is provided     
            active_plot_fn = plot_3d_binned_bars
            # active_plot_fn = plot_3d_stem_points
        
        if active_plot_fn.__name__ == plot_3d_stem_points.__name__:
            active_xbins = xbin_centers
            active_ybins = ybin_centers
        else:
            # required for `plot_3d_binned_bars`
            active_xbins = xbin
            active_ybins = ybin

        is_single_time_bin_posterior_plot: bool = (np.ndim(posterior_p_x_given_n) < 3)
        if is_single_time_bin_posterior_plot:
            plotActors, data_dict = active_plot_fn(
                p, active_xbins, active_ybins, posterior_p_x_given_n, name=posterior_name,
                drop_below_threshold=drop_below_threshold, opacity=opacity, **kwargs
            )

            # , **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75} | kwargs)

            if point_labeling_function is None:
                # The full point shown:
                # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
                # Only the z-values
                point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'

            if point_masking_function is None:
                if drop_below_threshold is not None:
                    # point_masking_function = lambda points: points[:, 2] > 20.0
                    point_masking_function = lambda points: points[:, 2] > drop_below_threshold
                else:
                    point_masking_function = lambda points: points[:, 2] > -1

            if enable_point_labels:
                plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(
                    p, xbin_centers, ybin_centers, posterior_p_x_given_n, 
                    point_labels=point_labeling_function, 
                    point_mask=point_masking_function,
                    shape='rounded_rect', shape_opacity=0.5, show_points=False, name=f'{posterior_name}Labels'
                )
            else:
                plotActors_CenterLabels, data_dict_CenterLabels = None, None

        else:
            ## multi-time bin plot:
            from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars_timeseries

            assert np.ndim(posterior_p_x_given_n) == 3

            plotActors, data_dict = plot_3d_binned_bars_timeseries(
                p=p, xbin=active_xbins, ybin=active_ybins, t_bins=time_bin_centers, data=posterior_p_x_given_n, name=posterior_name,
                drop_below_threshold=drop_below_threshold, opacity=opacity, active_plot_fn=active_plot_fn, **kwargs
            )
            
            if enable_point_labels:
                print(f'WARN: enable_point_labels is not currently implemented for multi-time-bin plotting mode.')

            plotActors_CenterLabels, data_dict_CenterLabels = None, None

        return (plotActors, data_dict), (plotActors_CenterLabels, data_dict_CenterLabels)


    @classmethod
    def perform_plot_filled_contours(cls, p, xbin_centers, ybin_centers, posterior_p_x_given_n, levels=10, cmap='viridis', opacity=0.75, name='PosteriorFilledContours', contour_extrude_z=None, enable_contour_fill: bool=False, **kwargs):
        """ Plots filled contours of the posterior probability using PyVista.

        Args:
            p: The PyVista plotter instance to plot on.
            xbin_centers (np.ndarray): array of x centers
            ybin_centers (np.ndarray): array of y centers
            posterior_p_x_given_n (np.ndarray): 2D array of posterior values (shape: [len(xbin_centers), len(ybin_centers)])
            levels (int or list): Number of contour levels or list of scalar values.
            cmap (str): Colormap name.
            opacity (float): Opacity of the filled contours.
            name (str): Name for the mesh/actor.
            contour_extrude_z (float, optional): If set, extrude the contour line upward along the z-axis by this amount to form a vertical surface (curtain).
            **kwargs: Additional keyword args for PyVista `add_mesh`.

        Returns:
            (dict, dict): ({'contours': actor}, {'contours': mesh})
        """
        import numpy as np
        import pyvista as pv
        from matplotlib import cm as mpl_cm

        is_3d: bool = False
        if np.ndim(posterior_p_x_given_n) == 3:
            n_x_bins, n_y_bins, n_t_bins = posterior_p_x_given_n.shape
            is_3d = True
        elif np.ndim(posterior_p_x_given_n) == 2:
            n_x_bins, n_y_bins = posterior_p_x_given_n.shape
            n_t_bins = 1
            is_3d = False
        else:
            raise ValueError(f'np.shape(posterior_p_x_given_n) should be 2 or 3, but it is: {np.shape(posterior_p_x_given_n)} which is unsupported.')



        def plot_single_t_bin_contour(mask_2d, sub_name: str, t_idx: int, n_t_bins: int):
            """One contour per time bin, with a unique color and translucent fill. Captures: p, xbin_centers, ybin_centers, levels, cmap, opacity, name, kwargs."""
            # Prepare regular grid for StructuredGrid
            nx, ny = len(xbin_centers), len(ybin_centers)
            X, Y = np.meshgrid(xbin_centers, ybin_centers, indexing='ij')
            Z = np.zeros_like(X)

            grid = pv.StructuredGrid(X, Y, Z)
            grid["posterior"] = mask_2d.astype(float).flatten(order='F')

            # Single contour level per time bin: one isosurface at the mid value
            vmin, vmax = np.nanmin(mask_2d), np.nanmax(mask_2d)
            if isinstance(levels, int):
                single_level = (vmin + vmax) / 2.0
            else:
                lev_arr = np.asarray(levels)
                single_level = float(lev_arr[len(lev_arr) // 2]) if len(lev_arr) else (vmin + vmax) / 2.0
            contour_levels = np.array([single_level])

            contours = grid.contour(isosurfaces=contour_levels, scalars="posterior")

            # Unique color for this time bin from the colormap
            cmap_lut = mpl_cm.get_cmap(cmap, max(n_t_bins, 1))
            t_frac = t_idx / max(n_t_bins - 1, 1)
            color_rgb = np.array(cmap_lut(t_frac)[:3])

            filled = None
            fill_actor = None
            if enable_contour_fill:
                # Translucent fill: threshold grid to region >= single_level, then add as shaded mesh
                filled = grid.threshold(value=single_level, scalars="posterior")                
                if filled.n_cells > 0:
                    fill_actor = p.add_mesh(filled, color=color_rgb, opacity=opacity, name=sub_name + "_fill", **kwargs)
                    
            # Optional: extrude contour upward along z to form a vertical surface (curtain)
            contour_surface_actor, contour_surface_mesh = None, None
            if contour_extrude_z is not None and contours.n_cells > 0:
                contour_surface_mesh = contours.extrude([0, 0, float(contour_extrude_z)], capping=False)
                contour_surface_actor = p.add_mesh(contour_surface_mesh, color=color_rgb, opacity=0.2, name=sub_name + "_contour_surface", edge_color=color_rgb, **kwargs) # "blue" show_edges=False, edge_opacity=1.0, 
                
            # Contour line on top (same color, full opacity so boundary is visible)
            line_actor = p.add_mesh(contours, color=color_rgb, opacity=1.0, name=sub_name, **kwargs)
            return contours, fill_actor, line_actor, filled, contour_surface_actor, contour_surface_mesh


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        plotActors = {'contours': {}}
        data_dict = {'contours': {}}

        for t_idx in range(n_t_bins):
            # Extract the 2D boolean mask for the current time bin
            if is_3d:
                mask_2d = posterior_p_x_given_n[:, :, t_idx]
                sub_name: str = f'{name}[{t_idx}]'
            else:
                mask_2d = posterior_p_x_given_n
                sub_name: str = name

            # Skip if the mask is empty (no True values) to avoid errors
            if not np.any(mask_2d):
                continue

            contours, fill_actor, line_actor, filled, contour_surface_actor, contour_surface_mesh = plot_single_t_bin_contour(mask_2d=mask_2d, sub_name=sub_name, t_idx=t_idx, n_t_bins=n_t_bins)
            data_dict['contours'][sub_name] = contours
            plotActors['contours'][sub_name] = line_actor
            if fill_actor is not None:
                data_dict['contours'][sub_name + '_fill'] = filled
                plotActors['contours'][sub_name + '_fill'] = fill_actor
            if contour_surface_actor is not None and contour_surface_mesh is not None:
                data_dict['contours'][sub_name + '_contour_surface'] = contour_surface_mesh
                plotActors['contours'][sub_name + '_contour_surface'] = contour_surface_actor

        return plotActors, data_dict



    def plot_decoded_PBE_matching_past_future_results(self, a_ds, an_epoch_idx: int = 4):
        """Plot a single decoded PBE (population burst event) with matching past/future trajectory segments in a PyVista 3D view."""
        import pyvista as pv
        import pyvistaqt as pvqt
        import numpy as np
        from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena

        def plot_any_spline_literal(p, curr_lap_points, name=None, line_scalars=None, plot_data=None, **kwargs):
            """ plots the position line 
            
            num_lap_samples = np.shape(curr_lap_position_traces)[1]
            curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], curr_lap_position_traces[2, :])) # (N, 3)
            plot_data, plot = plot_any_spline_literal(p, curr_lap_points, name=None, line_scalars=None, plot_data=None, )
            
            """
            line = LapsVisualizationMixin.lines_from_points(curr_lap_points)
            if line_scalars is not None:
                line["scalars"] = line_scalars
            else:      
                # Color by time (index) as per original implementation
                line["scalars"] = np.arange(line.n_points)

            plot_data = (plot_data or {}) | {'name': name, 'curr_lap_points': curr_lap_points}

            tube = line.tube(radius=0.1)
            plot_data['tube'] = tube
            
            # Note: 'show_scalar_bar': False is set, so you won't see the legend unless changed to True
            plot = p.add_mesh(tube, **({'name': name, 'render_lines_as_tubes': False, 'show_scalar_bar': False, 'lighting': True, 'render': False} | kwargs))
            return plot_data, plot


        def plot_any_spline(p, curr_lap_position_traces, lap_start_z=0.9, time_to_z_range: float=10.0, name=None, color_by_speed=True, render_kwargs_dict=None, **kwargs):
            """ plots the position line """
            num_lap_samples = np.shape(curr_lap_position_traces)[1]
            curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], curr_lap_position_traces[2, :])) # (N, 3)

            ts = deepcopy(curr_lap_points[:, 2])
            earliest_t: float = np.nanmin(ts)
            time_range: float = np.ptp(ts)
            print(f'time_range: {time_range}')
            time_axis_scaling_factor: float = time_to_z_range / time_range
            print(f'time_axis_scaling_factor: {time_axis_scaling_factor}')
            times_to_z_pos_fn = lambda ts: ((ts - earliest_t) * time_axis_scaling_factor) + lap_start_z
            curr_lap_points[:, 2] = ((ts - earliest_t) * time_axis_scaling_factor) + lap_start_z # np.array([0.9, 0.900204, 0.900409, ..., 100.9, 100.9, 100.9])

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

            plot_data = {'name': name, 'times_to_z_pos_fn': times_to_z_pos_fn, 'time_range': time_range, 'earliest_t': earliest_t, 'num_lap_samples':num_lap_samples, 'curr_lap_position_traces': curr_lap_position_traces, 'curr_lap_points': curr_lap_points}
            tube = line.tube(radius=0.1)
            plot_data['tube'] = tube
            
            if (render_kwargs_dict is None) or (len(render_kwargs_dict) == 0):
                # tube.plot(smooth_shading=True)
                # color_map_name = 'bmy' # old
                color_map_name = 'cividis' # 2023-05-09 and newer
                # 'cmap': color_map_name, 
                kwargs['cmap'] = color_map_name
            else:
                for k, v in render_kwargs_dict.items():
                    kwargs[k] = v


            # Note: 'show_scalar_bar': False is set, so you won't see the legend unless changed to True
            plot = p.add_mesh(tube, **({'name': name, 'render_lines_as_tubes': False, 'show_scalar_bar': False, 'lighting': False, 'render': False} | kwargs))
            return plot_data, plot


        # Section 1: Datasource and position data
        # a_ds = self
        curr_position_df: pd.DataFrame = deepcopy(a_ds.curr_position_df)
        x = curr_position_df['x'].to_numpy()
        y = curr_position_df['y'].to_numpy()

        # Section 2: Plotter and base scene
        plots = {}
        plots_data = {}
        plotter = pvqt.BackgroundPlotter()
        plotter.background_color = pv.Color('paraview')
        plots['maze_bg'] = perform_plot_flat_arena(plotter, x, y, bShowSequenceTraversalGradient=False, smoothing=False)
        plots_data['maze_bg'] = {'track_dims': None, 'maze_pdata': None}

        # Section 3: Time–Z config and full trajectory
        lap_start_z = 0.9
        time_to_z_range = 100.0
        position_stop_z: float = lap_start_z + time_to_z_range
        render_kwargs_dict = {'color': [0.1, 0.1, 0.1], 'pbr': True, 'metallic': 0.8, 'roughness': 0.5, 'diffuse': 1, 'opacity': 0.01, 'render': True}
        xyt = curr_position_df[['x', 'y', 't']].to_numpy().T
        plot_data, plot_tube = plot_any_spline(plotter, curr_lap_position_traces=xyt, name='all_positions', lap_start_z=lap_start_z, time_to_z_range=time_to_z_range, color_by_speed=False, render_kwargs_dict=render_kwargs_dict)
        plots_data['all_positions'] = plot_data
        plots['all_positions'] = plot_tube
        times_to_z_pos_fn = plots_data['all_positions']['times_to_z_pos_fn']

        # Section 4: Epoch helpers and run
        def _cleanup_epoch_linear_idx_actors(plotter, plots_data=None, plots=None):
            """Removes the actors added by _on_update_epoch_linear_idx (PosteriorFilledContours and PBE path segments).
            plots_data, plots = _cleanup_epoch_linear_idx_actors(plotter=plotter, plots_data=plots_data, plots=plots)
            """
            if plots_data is None:
                plots_data = {}
            if plots is None:
                plots = {}

            # 1) Remove PosteriorFilledContours (structure: {'contours': {sub_name: actor, ...}})
            contour_plot = plots.pop('PosteriorFilledContours', None)
            plots_data.pop('PosteriorFilledContours', None)
            if contour_plot is not None and isinstance(contour_plot, dict) and 'contours' in contour_plot:
                for sub_name, actor in contour_plot['contours'].items():
                    if actor is not None:
                        try:
                            plotter.remove_actor(actor)
                        except Exception:
                            pass

            # 2) Remove all PBE path segment actors (keys like PBE[0][past][0], PBE[0][future][1])
            keys_to_remove = [k for k in plots if isinstance(k, str) and k.startswith('PBE[')]
            for key in keys_to_remove:
                actor_or_struct = plots.pop(key, None)
                plots_data.pop(key, None)
                if actor_or_struct is None:
                    continue
                try:
                    if isinstance(actor_or_struct, dict) and 'main' in actor_or_struct:
                        plotter.remove_actor(actor_or_struct['main'])
                    else:
                        plotter.remove_actor(actor_or_struct)
                except Exception:
                    pass

            return plots_data, plots

        def _on_update_epoch_linear_idx(plotter, a_ds, an_epoch_idx: int, times_to_z_pos_fn, plots_data=None, plots=None):
            """Plot for a given PBE: posterior contours and all matching past/future path segments."""
            if plots_data is None:
                plots_data = {}
            if plots is None:
                plots = {}

            a_ds.filter_epochs['original_epoch_idx'] = a_ds.filter_epochs['original_epoch_idx'].astype(int)
            an_active_PBE_epoch_row = a_ds.filter_epochs.iloc[an_epoch_idx]
            # an_active_PBE_epoch_row.original_epoch_idx = int(an_active_PBE_epoch_row.original_epoch_idx)

            an_epoch_mask = a_ds.epoch_t_bins_high_prob_pos_masks[an_epoch_idx]
            xbin_centers = a_ds.xbin_centers
            ybin_centers = a_ds.ybin_centers
            active_plot_key: str = 'PosteriorFilledContours'
            plots[active_plot_key], plots_data[active_plot_key] = DecoderRenderingPyVistaMixin.perform_plot_filled_contours(p=plotter, xbin_centers=xbin_centers, ybin_centers=ybin_centers, posterior_p_x_given_n=an_epoch_mask, levels=1, cmap='viridis', opacity=0.95, contour_extrude_z=position_stop_z, name=f'PosteriorFilledContours')

            
            a_row_out_dict = a_ds._prepare_epoch_data(an_epoch_idx=an_active_PBE_epoch_row.original_epoch_idx, minimum_included_matching_sequence_length=4)
            curr_matching_past_future_positions_df_dict: Dict[types.PastFutureCategory, Dict[types.epoch_index, pd.DataFrame]] = a_row_out_dict['curr_matching_past_future_positions_df_dict']
            if 'past' not in curr_matching_past_future_positions_df_dict:
                curr_matching_past_future_positions_df_dict['past'] = {}
            if 'future' not in curr_matching_past_future_positions_df_dict:
                curr_matching_past_future_positions_df_dict['future'] = {}
            
            for a_past_future_key, past_matching_position_df_dict in curr_matching_past_future_positions_df_dict.items():
                for a_matched_position_segment_idx, a_matched_pos_segment_df in past_matching_position_df_dict.items():
                    a_matched_pos_segment_df['z'] = a_matched_pos_segment_df['t'].apply(times_to_z_pos_fn)
                    render_kwargs_dict = {'color': [0.9, 0.2, 0.2], 'pbr': False, 'metallic': 0.1, 'roughness': 0.8, 'diffuse': 1, 'render': True, 'lighting': True, 'render_lines_as_tubes': True, 'line_width': 100.0}
                    plot_seg_key: str = f'PBE[{an_active_PBE_epoch_row.original_epoch_idx}][{a_past_future_key}][{a_matched_position_segment_idx}]'
                    print(f'plotting: "{plot_seg_key}"')
                    xyz = a_matched_pos_segment_df[['x', 'y', 'z']].to_numpy().T
                    curr_lap_points = np.column_stack((xyz[0,:], xyz[1,:], xyz[2, :]))
                    plots_data[plot_seg_key], plots[plot_seg_key] = plot_any_spline_literal(p=plotter, curr_lap_points=curr_lap_points, name=plot_seg_key, line_scalars=None, plot_data=dict(), **render_kwargs_dict)
            return plots_data, plots

        
        plots_data, plots = _cleanup_epoch_linear_idx_actors(plotter=plotter, plots_data=plots_data, plots=plots)
        plots_data, plots = _on_update_epoch_linear_idx(plotter=plotter, a_ds=a_ds, an_epoch_idx=an_epoch_idx, times_to_z_pos_fn=times_to_z_pos_fn, plots_data=plots_data, plots=plots)
        plotter.show_bounds()
        return plots_data, plots