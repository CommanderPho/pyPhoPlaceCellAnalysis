from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from matplotlib.collections import PathCollection

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, BasePositionDecoder
    from pyphoplacecellanalysis.External.peak_prominence2d import PosteriorPeaksPeakProminence2dResult
    from nptyping import NDArray

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

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

# ==================================================================================================================== #
# 2024-04-12 - Decoded Trajectory Plotting on Maze (1D & 2D) - Posteriors and Most Likely Position Paths               #
# ==================================================================================================================== #

from pyphocorehelpers.plotting.heading_angle_helpers import HeadingAngleHelpers


class RenderColoringMode(str, Enum):
    """How to color rendered path elements (e.g. line segments, arrows): by time (colormap), by speed, or by heading angle (ROYGBIV, North=Red)."""
    TIME = 'time'
    SPEED = 'speed'
    ANGLE = 'angle'


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.geometry_helpers import point_tuple_mid_point, BoundsRect, is_point_in_rect

# ==================================================================================================================================================================================================================================================================================== #
# TODO 2025-12-16 16:37: - [ ] AI-implemnented attempt to replace Aims to replace `SingleArtistMultiEpochBatchHelpers` with a much more efficient implementation                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #

"""
Optimized viewport-based rendering with image caching and adaptive bin sizing
for decoded trajectory timeline visualization.

This class efficiently renders only visible epochs, caches rendered thumbnails,
and adapts bin size based on zoom level - similar to video editor timeline previews.
"""

from typing import Optional, Tuple, Dict, List, Callable, Any
import numpy as np
import pandas as pd
from copy import deepcopy
from attrs import define, field, Factory
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image rendering

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots

import logging
logger = logging.getLogger(__name__)


@metadata_attributes(short_name=None, tags=['OLD', '2D_timeseries', '2D_posteriors', 'frames', 'UNFINISHED', 'KINDA-WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-19 00:00', related_items=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side'])
@define(slots=False, eq=False)
class SingleArtistMultiEpochBatchHelpers:
    """ Handles draw
    Consider a decoded posterior computed from 2D placefields. You get a separate 2D position posterior for each time bin, which is difficult to view except in 3D.
    To present this data in a 2D interface, as a SpikeRasterWindow (SpikeRaster2D) timeline track, for example, it needs to be framed into "snapshot_periods" of reasonable scale given the current display window
        - this process I call "subdividing" and is done by adding a 'subidvision_idx' column to the dataframe
    These "snapshot_periods" need to then be rendered as 2D artists next to each other along the x-axis (time). 
        (x_min, ..., x_max) | (x_min, ..., x_max), | ... | (x_min, ..., x_max) ## where there are `n_frames` repeats
        
    
        #  each containing `n_frame_division_samples`
        
    1. compute all-time (erroniously called "continuous" throughout the codebase) decoding, which always contains a single epoch (referring to the entire global epoch)
    2. frame_divide this single epoch into frame_divisions of fixed duration: `frame_divide_bin_size`
        There will be `n_frame_division_epochs`: 
        ```
        frame_divide_bin_size: float = 0.5
        n_frame_division_epochs: int = int(round(total_global_time_duration / frame_divide_bin_size))
        ```
        
    
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import SingleArtistMultiEpochBatchHelpers
    
    
    
    
    USAGE:
    
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import SingleArtistMultiEpochBatchHelpers
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode

        track_name: str = 'SingleArtistMultiEpochBatchTrack'
        spike_raster_plt_2d: Spike2DRaster = spike_raster_window.spike_raster_plt_2d
        track_ts_widget, track_fig, track_ax_list = spike_raster_plt_2d.add_new_matplotlib_render_plot_widget(name=track_name)
        track_ax = track_ax_list[0]
        desired_epoch_start_idx: int = 0
        # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
        desired_epoch_end_idx: Optional[int] = None

        ## INPUTS: frame_divide_bin_size, results2D
        batch_plot_helper: SingleArtistMultiEpochBatchHelpers = SingleArtistMultiEpochBatchHelpers(results2D=results2D, active_ax=track_ax, frame_divide_bin_size=frame_divide_bin_size, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)
        plots_data = batch_plot_helper.add_all_track_plots(global_session=global_session)
        
    
            
    Usage -- Individual Components:
        desired_epoch_start_idx: int = 0
        # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
        desired_epoch_end_idx: Optional[int] = None

        ## INPUTS: frame_divide_bin_size, results2D
        batch_plot_helper: SingleArtistMultiEpochBatchHelpers = SingleArtistMultiEpochBatchHelpers(results2D=results2D, active_ax=track_ax, frame_divide_bin_size=frame_divide_bin_size, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)

        batch_plot_helper.shared_build_flat_stacked_data(force_recompute=True, debug_print=True)

        track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session, override_ax=None) ## does not seem to successfully synchronize to window
        # track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session, override_ax=track_shapes_dock_track_ax) ## does not seem to successfully synchronize to window

        measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions(override_ax=None)
        # measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions(override_ax=measured_pos_dock_track_ax)

        curr_artist_dict, image_extent, plots_data = batch_plot_helper.add_position_posteriors(posterior_masking_value=0.0025, override_ax=None, debug_print=True, defer_draw=False)

    
    History 2025-02-20 08:58:
        # In EpochComputationFunctions.py:
        subdivided_epochs_results -> frame_divided_epochs_results
        subdivided_epochs_df -> frame_divided_epochs_df
        global_subivided_epochs_obj -> global_frame_divided_epochs_obj 
        global_subivided_epochs_df -> global_frame_divided_epochs_df
        subdivided_epochs_specific_decoded_results_dict -> frame_divided_epochs_specific_decoded_results_dict

        # In decoder_plotting_mixins.py:
        subdivide_bin_size -> frame_divide_bin_size


    """
    # results2D: "DecodingResultND" = field()
    frame_divided_epochs_result: "DecodedFilterEpochsResult" = field()
    decoder: "BasePositionDecoder" = field()
    pos_df: pd.DataFrame = field()


    active_ax: Any = field()
    frame_divide_bin_size: float = field()
    rotate_to_vertical: bool = field(default=True)
    
    desired_epoch_start_idx: int = field(default=0)
    desired_epoch_end_idx: Optional[int] = field(default=None)

    stacked_flat_global_pos_df: pd.DataFrame = field(default=None, init=False)

    has_data_been_built: bool = field(default=False)
    active_epoch_name: str = field(default='global')
    

    @property
    def num_filter_epochs(self) -> int:
        """number of frame_division epochs."""
        return self.a_result2D.num_filter_epochs

    @property
    def num_horizontal_repeats(self) -> int:
        """number of repeats along the absecessa."""
        return (self.num_filter_epochs-1)

    @property
    def a_result2D(self) -> DecodedFilterEpochsResult:
        # return self.results2D.frame_divided_epochs_results[self.active_epoch_name]
        return self.frame_divided_epochs_result

    @property
    def a_new_global2D_decoder(self) -> BasePositionDecoder:
        # return self.results2D.decoders[self.active_epoch_name]
        return self.decoder

    @property
    def desired_start_time_seconds(self) -> float:
        return self.desired_epoch_start_idx * self.frame_divide_bin_size
    
    @property
    def desired_end_time_seconds(self) -> float:
        if self.desired_epoch_end_idx is not None:
            return self.frame_divide_bin_size * self.desired_epoch_end_idx
        else:
            return self.frame_divide_bin_size * (self.num_filter_epochs-1)
        
    @property
    def desired_time_duration(self) -> float:
        return self.desired_end_time_seconds - self.desired_start_time_seconds


    def __attrs_post_init__(self):
        # Add post-init logic here
        # if self.desired_epoch_end_idx is None:
        #     ## determine the correct end-index            
        pass
    

    @function_attributes(short_name=None, tags=['data'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 16:07', related_items=[])
    def shared_build_flat_stacked_data(self, debug_print=False, should_expand_first_dim: bool=True, force_recompute:bool=False, desired_epoch_start_idx=None, desired_epoch_end_idx=None, **kwargs):
        """ finalize building the data for single-artist plotting (does not plot anything)
        
        From `#### 2025-02-14 - Perform plotting of Measured Positions (using `stacked_flat_global_pos_df['global_frame_division_x_data_offset']`)`
        Uses:
            self.a_new_global2D_decoder
            self.stacked_flat_global_pos_df
            self.a_result2D
        Updates:
            self.stacked_flat_global_pos_df
            
        Outputs:
            (self.n_xbins, self.n_ybins, self.n_tbins), (self.flattened_n_xbins, self.flattened_n_ybins, self.flattened_n_tbins), (self.stacked_p_x_given_n, self.stacked_flat_time_bin_centers, self.stacked_flat_xbin_centers, self.stacked_flat_ybin_centers)
            (self.xbin_edges, self.ybin_edges)
            (self.x0_offset, self.y0_offset, self.x1_offset, self.y1_offset)
            
            
        Usage:
        
            batch_plot_helper.shared_build_flat_stacked_data(debug_print=True)
            
        """
        # stacked_flat_global_pos_df = self.stacked_flat_global_pos_df
        # desired_time_duration = self.desired_time_duration
        # desired_start_time_seconds = self.desired_start_time_seconds
        # desired_end_time_seconds = self.desired_end_time_seconds

        # raise NotImplementedError(f'2025-02-14_TO_REFACTOR_FROM_NOTEBOOK')
        ## INPUTS: a_result2D, a_new_global2D_decoder
        
        # rotate_to_vertical: bool = False
        if desired_epoch_start_idx is not None:
            self.desired_epoch_start_idx = desired_epoch_start_idx
            
        if desired_epoch_end_idx is not None:
            self.desired_epoch_end_idx = desired_epoch_end_idx

        pos_col_names = ['x', 'y']
        binned_col_names = ['binned_x', 'binned_y']

        if debug_print:
            print(f'desired_epoch_start_idx: {self.desired_epoch_start_idx}, desired_epoch_end_idx: {self.desired_epoch_end_idx}')
            print(f'desired_start_time_seconds: {self.desired_start_time_seconds}, desired_end_time_seconds: {self.desired_end_time_seconds}')

        ## finalize building the data for single-artist plotting (does not plot anything)
        (self.n_xbins, self.n_ybins, self.n_tbins), (self.flattened_n_xbins, self.flattened_n_ybins, self.flattened_n_tbins), (self.stacked_p_x_given_n, self.stacked_flat_time_bin_centers, self.stacked_flat_xbin_centers, self.stacked_flat_ybin_centers) = self.complete_build_stacked_flat_arrays(a_result=self.a_result2D, a_new_global_decoder=self.a_new_global2D_decoder,
                                                                                                                                                                                                                    desired_epoch_start_idx=self.desired_epoch_start_idx, desired_epoch_end_idx=self.desired_epoch_end_idx,
                                                                                                                                                                                                                    rotate_to_vertical=self.rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)
        


        if force_recompute is True:
            print(f'force_recompute == True, so `self.stacked_flat_global_pos_df` will be rebuilt from scratch from `self.pos_df`...')
            self.has_data_been_built = False
    
        if (self.stacked_flat_global_pos_df is None) or force_recompute:
            self.stacked_flat_global_pos_df = deepcopy(self.pos_df)

        ## slice `stacked_flat_global_pos_df` by desired start/end indicies too:
        if (self.desired_epoch_end_idx is not None):
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[np.logical_and((self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx), (self.stacked_flat_global_pos_df['global_frame_division_idx'] < self.desired_epoch_end_idx))]
        else:
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[(self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx)]


        # Validate that filtering didn't result in empty dataframe
        if len(self.stacked_flat_global_pos_df) == 0:
            # Get available indices for better error message
            if hasattr(self, 'results2D') and self.pos_df is not None and 'global_frame_division_idx' in self.pos_df.columns:
                available_indices = sorted(self.pos_df['global_frame_division_idx'].unique())
                min_idx, max_idx = available_indices[0], available_indices[-1] if len(available_indices) > 0 else (None, None)
            else:
                available_indices = []
                min_idx, max_idx = None, None
            
            error_msg = (
                f"No data found for epoch range [start={self.desired_epoch_start_idx}, end={self.desired_epoch_end_idx}). "
            )
            if available_indices:
                error_msg += f"Available global_frame_division_idx range: [{min_idx}, {max_idx}] (values: {available_indices[:10]}{'...' if len(available_indices) > 10 else ''})"
            else:
                error_msg += "No global_frame_division_idx values found in source data."
            
            raise ValueError(error_msg)



        # (self.n_xbins, self.n_ybins, self.n_tbins), (self.flattened_n_xbins, self.flattened_n_ybins, self.flattened_n_tbins)
        # np.shape(stacked_p_x_given_n) # (1, 171, 6)
        self.xbin_edges = deepcopy(self.a_new_global2D_decoder.xbin)
        self.ybin_edges = deepcopy(self.a_new_global2D_decoder.ybin)
        # xmin, xmax, ymin, ymax = self.xbin_edges[0], self.xbin_edges[-1], self.ybin_edges[0], self.ybin_edges[-1]
        self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df.position.adding_binned_position_columns(xbin_edges=self.xbin_edges, ybin_edges=self.ybin_edges)
        ## OUTPUTS: (desired_epoch_start_idx, desired_epoch_end_idx), (desired_start_time_seconds, desired_end_time_seconds), desired_time_duration

        ## INPUTS: stacked_flat_global_pos_df, active_ax
        self.inverse_bin_width: float = np.ptp(self.xbin_edges) ## data_coords scale
        self.inverse_bin_height: float = np.ptp(self.ybin_edges)
        if debug_print:
            print(f".xbin: {self.xbin_edges}")
            print(f".ybin: {self.ybin_edges}")

        self.x0_offset: float =  self.xbin_edges[0]
        self.x1_offset: float =  self.xbin_edges[-1]
        
        self.y0_offset: float =  self.ybin_edges[0]       
        self.y1_offset: float =  self.ybin_edges[-1]

        if debug_print:
            print(f'x0_offset: {self.x0_offset}, y0_offset: {self.y0_offset}')

        # (np.nanmin(self.stacked_flat_global_pos_df['x']), np.nanmax(self.stacked_flat_global_pos_df['x']))
        # (np.nanmin(self.stacked_flat_global_pos_df['y']), np.nanmax(self.stacked_flat_global_pos_df['y']))
        
        ## INPUTS: (desired_epoch_start_idx, desired_epoch_end_idx), (desired_start_time_seconds, desired_end_time_seconds), desired_time_duration
        if debug_print:
            print(f'desired_time_duration: {self.desired_time_duration}, (desired_start_time_seconds: {self.desired_start_time_seconds}, desired_end_time_seconds: {self.desired_end_time_seconds})')
        ## INPUTS: x0_offset, y0_offset
        # custom_image_extent = np.array([0.0, 1.0, 0.0, 1.0])
        max_global_frame_division_idx: int = np.nanmax(self.stacked_flat_global_pos_df['global_frame_division_idx']) ## TODO: could allow not starting on zero, but let's not
        active_num_global_frame_divisions: int = max_global_frame_division_idx + 1   
        if debug_print:
            print(f'active_num_global_frame_divisions: {active_num_global_frame_divisions}')
        single_global_frame_division_axes_coords_width: float = 1.0 / float(active_num_global_frame_divisions)
        single_global_frame_division_axes_coords_duration: float = float(self.desired_time_duration) / float(active_num_global_frame_divisions) ## how "long" a single frame spans along the t axis (in seconds)


        assert 'frame_division_epoch_start_t' in self.stacked_flat_global_pos_df
        if debug_print:
            print(f'single_global_frame_division_axes_coords_width: {single_global_frame_division_axes_coords_width}\nsingle_global_frame_division_axes_coords_duration: {single_global_frame_division_axes_coords_duration}')
        self.stacked_flat_global_pos_df['global_frame_division_x_unit_offset'] = (self.stacked_flat_global_pos_df['global_frame_division_idx'].astype(float) * single_global_frame_division_axes_coords_width) # in unit coordinates
        # stacked_flat_global_pos_df['global_frame_division_x_data_offset'] = (stacked_flat_global_pos_df['global_frame_division_x_unit_offset'].astype(float) * single_global_frame_division_axes_coords_duration) # in data coordinates (along the t-axis)
        self.stacked_flat_global_pos_df['global_frame_division_x_data_offset'] = (self.stacked_flat_global_pos_df['global_frame_division_x_unit_offset'].astype(float) * self.stacked_flat_global_pos_df['frame_division_epoch_start_t'].astype(float)) # in data coordinates (along the t-axis)

        ## Actually update 'x' or 'y' inplace in the dataframe:
        if self.rotate_to_vertical:
            # stacked_flat_global_pos_df['x'] -= a_new_global_decoder.xbin[0] ## zero-out the x0 by subtracting out the minimal xbin_edge
            # stacked_flat_global_pos_df['x'] += stacked_flat_global_pos_df['global_frame_division_x_offset']
            
            # ## As imported pre-2025-02-16:
            # self.stacked_flat_global_pos_df['x'] -= self.x0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge
            # self.stacked_flat_global_pos_df['x_smooth'] -= self.x0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge
            
            # self.stacked_flat_global_pos_df['y'] -= self.y0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge

            ## As imported 2025-02-17:
            # self.stacked_flat_global_pos_df['x'] -= self.x0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge
            # self.stacked_flat_global_pos_df['x_smooth'] -= self.x0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge
            # self.stacked_flat_global_pos_df['y'] -= self.y0_offset ## zero-out the x0 by subtracting out the minimal xbin_edge

            self.stacked_flat_global_pos_df['x_scaled'] = (self.stacked_flat_global_pos_df['x'] - self.y0_offset) / (self.y1_offset - self.y0_offset)
            # self.stacked_flat_global_pos_df['x_smooth_scaled'] = (self.stacked_flat_global_pos_df['x_smooth'] - self.y0_offset) / (self.y1_offset - self.y0_offset)
            self.stacked_flat_global_pos_df['y_scaled'] = (self.stacked_flat_global_pos_df['y'] - self.x0_offset) / (self.x1_offset - self.x0_offset)

            # ## scale-down to [0.0, 1.0] scale
            # # stacked_flat_global_pos_df['x'] *= inverse_normalization_factor_width ## scale to [0, 1]
            # stacked_flat_global_pos_df['x'] *= inverse_full_ax_factor_width ## scale to [0, 1]/num_sub_epochs
            # stacked_flat_global_pos_df['x_smooth'] *= inverse_full_ax_factor_width ## scale to [0, 1]/num_sub_epochs
            
            # stacked_flat_global_pos_df['y'] *= inverse_normalization_factor_height ## scale to [0, 1]

            # stacked_flat_global_pos_df['x'] += stacked_flat_global_pos_df['global_frame_division_x_offset']



            # ==================================================================================================================== #
            # 2025-02-18 01:21 New from Notebook                                                                                   #
            # ==================================================================================================================== #
            # self.stacked_flat_global_pos_df = deepcopy(batch_plot_helper.stacked_flat_global_pos_df)



            # stacked_flat_global_pos_df['y_scaled'] = (stacked_flat_global_pos_df['y'] - batch_plot_helper.y0_offset) / (batch_plot_helper.y1_offset - batch_plot_helper.y0_offset)
            # stacked_flat_global_pos_df['x_smooth_scaled'] = (stacked_flat_global_pos_df['x_smooth'] - batch_plot_helper.y0_offset) / (batch_plot_helper.y1_offset - batch_plot_helper.y0_offset)
            # stacked_flat_global_pos_df['x_scaled'] = (stacked_flat_global_pos_df['x'] - batch_plot_helper.x0_offset) / (batch_plot_helper.x1_offset - batch_plot_helper.x0_offset)




            ## swap axes:
            self.stacked_flat_global_pos_df['y_temp'] = deepcopy(self.stacked_flat_global_pos_df['y'])
            self.stacked_flat_global_pos_df['y'] = deepcopy(self.stacked_flat_global_pos_df['x'])
            self.stacked_flat_global_pos_df['x'] = deepcopy(self.stacked_flat_global_pos_df['y_temp'])
            self.stacked_flat_global_pos_df.drop(columns=['y_temp'], inplace=True)

            self.stacked_flat_global_pos_df['y_scaled_temp'] = deepcopy(self.stacked_flat_global_pos_df['y_scaled'])
            self.stacked_flat_global_pos_df['y_scaled'] = deepcopy(self.stacked_flat_global_pos_df['x_scaled'])
            self.stacked_flat_global_pos_df['x_scaled'] = deepcopy(self.stacked_flat_global_pos_df['y_scaled_temp'])
            self.stacked_flat_global_pos_df.drop(columns=['y_scaled_temp'], inplace=True)
            # self.stacked_flat_global_pos_df = PandasHelpers.swap_columns(self.stacked_flat_global_pos_df, lhs_col_name='x', rhs_col_name='y') 
            # self.stacked_flat_global_pos_df = PandasHelpers.swap_columns(self.stacked_flat_global_pos_df, lhs_col_name='x_scaled', rhs_col_name='y_scaled') 

        else:
            raise NotImplementedError()
            self.stacked_flat_global_pos_df['y'] += self.stacked_flat_global_pos_df['global_frame_division_x_offset']
            self.stacked_flat_global_pos_df['y_scaled'] = (self.stacked_flat_global_pos_df['y'] - self.y0_offset) / (self.y1_offset - self.y0_offset)


        ## OUTPUTS: single_global_frame_division_axes_coords_width, single_global_frame_division_axes_coords_duration
        ## UPDATES: stacked_flat_global_pos_df['global_frame_division_x_data_offset']
        



        self.has_data_been_built = True 


    # ==================================================================================================================== #
    # Track Position                                                                                                       #
    # ==================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['ALMOST_FINISHED', 'NOT_YET_FINISHED', '2025-02-14_TO_REFACTOR_FROM_NOTEBOOK'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-14 22:02', related_items=[])
    def add_track_positions(self, override_ax=None, debug_print=False, defer_draw:bool=False, **kwargs):
        """ Add the measured positions 
        
        From `#### 2025-02-14 - Perform plotting of Measured Positions (using `stacked_flat_global_pos_df['global_frame_division_x_data_offset']`)`
        Uses:
            self.inverse_xbin_width
            self.stacked_flat_global_pos_df
            self.a_result2D
        Updates:
            self.stacked_flat_global_pos_df
            
        Outputs:
            (self.n_xbins, self.n_ybins, self.n_tbins), (self.flattened_n_xbins, self.flattened_n_ybins, self.flattened_n_tbins), (self.stacked_p_x_given_n, self.stacked_flat_time_bin_centers, self.stacked_flat_xbin_centers, self.stacked_flat_ybin_centers)
            (self.xbin_edges, self.ybin_edges)
            
            
        Usage:
        
            measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions()
            
            
        """
        if override_ax is None:
            active_ax = self.active_ax
        else:
            active_ax = override_ax        

        if not self.has_data_been_built:
            ## finalize building the data for single-artist plotting (does not plot anything)
            self.shared_build_flat_stacked_data(debug_print=debug_print, should_expand_first_dim=True, **kwargs)

        if debug_print:
            print(f'desired_epoch_start_idx: {self.desired_epoch_start_idx}, desired_epoch_end_idx: {self.desired_epoch_end_idx}')
            print(f'desired_start_time_seconds: {self.desired_start_time_seconds}, desired_end_time_seconds: {self.desired_end_time_seconds}')


        assert 'global_frame_division_x_data_offset' in self.stacked_flat_global_pos_df
        
        # ==================================================================================================================== #
        # Old (non-working) pre 2025-02-17                                                                                     #
        # ==================================================================================================================== #
        # # y_axis_col_name: str = 'y'
        # y_axis_col_name: str = 'y_scaled'

        # assert y_axis_col_name in self.stacked_flat_global_pos_df
        
        # ## Perform the real plotting:
        # x = self.stacked_flat_global_pos_df['global_frame_division_x_data_offset'].to_numpy()
        # # y = self.stacked_flat_global_pos_df[y_axis_col_name].to_numpy() / self.inverse_xbin_width ## needs to be inversely mapped from 0, 1
        # y = self.stacked_flat_global_pos_df[y_axis_col_name].to_numpy() ## needs to be inversely mapped from 0, 1        

        # measured_pos_line_artist = active_ax.plot(x, y, color='r', label='measured_pos')[0]


        # ==================================================================================================================== #
        # New 2025-02-18 01:17                                                                                                 #
        # ==================================================================================================================== #
        
        time_cmap_start_end_colors = [(0, 0.6, 0), (0, 0, 0)]  # first is green, second is black
        time_cmap = LinearSegmentedColormap.from_list("GreenToBlack", time_cmap_start_end_colors, N=25) # Create a colormap (green to black).

        self.stacked_flat_global_pos_df = SingleArtistMultiEpochBatchHelpers.add_color_over_global_frame_division_idx_positions_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=self.stacked_flat_global_pos_df, time_cmap=time_cmap)
        
        # ensure the 'y_scaled' actually are scaled between [0.0, 1.0]
        self.stacked_flat_global_pos_df["y_scaled"] = (self.stacked_flat_global_pos_df["y_scaled"] - self.stacked_flat_global_pos_df["y_scaled"].min()) / (self.stacked_flat_global_pos_df["y_scaled"].max() - self.stacked_flat_global_pos_df["y_scaled"].min())
        
        # stacked_flat_global_pos_df
        new_stacked_flat_global_pos_df = SingleArtistMultiEpochBatchHelpers.add_nan_masked_rows_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=self.stacked_flat_global_pos_df)
        # new_stacked_flat_global_pos_df, color_formatting_dict = add_nan_masked_rows_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=stacked_flat_global_pos_df)

        # active_stacked_flat_global_pos_df = deepcopy(stacked_flat_global_pos_df)
        active_stacked_flat_global_pos_df = deepcopy(new_stacked_flat_global_pos_df)
        # extracted_colors_arr_flat: NDArray = active_stacked_flat_global_pos_df['color'].to_numpy()
        # extracted_colors_arr: NDArray = np.array(active_stacked_flat_global_pos_df['color'].to_list()).astype(float) # .shape # (16299, 4)

        # extracted_colors_arr.T.shape # (16299,)
        # a_time_bin_centers = deepcopy(active_stacked_flat_global_pos_df['t'].to_numpy().astype(float))
        # a_time_bin_centers

        measured_pos_dock_track_ax = active_ax
        # measured_pos_dock_track_ax.set_facecolor('white')
        measured_pos_dock_track_ax.set_facecolor('#333333')
        


        
        measured_pos_line_artist = measured_pos_dock_track_ax.scatter(active_stacked_flat_global_pos_df["global_frame_division_x_data_offset"], active_stacked_flat_global_pos_df["y_scaled"], color=active_stacked_flat_global_pos_df["color"].tolist())
        measured_pos_line_artist.set_alpha(0.85)
        measured_pos_line_artist.set_sizes([14])

        y_axis_kwargs = dict(ymin=0.0, ymax=1.0)
        # y_axis_kwargs = dict(ymin=self.xbin_edges[0], ymax=self.xbin_edges[-1])
        frame_division_epoch_separator_vlines = active_ax.vlines(self.results2D.frame_divided_epochs_df['start'].to_numpy(), **y_axis_kwargs, colors='white', linestyles='solid', label='frame_division_epoch_separator_vlines') # , data=None

        if not defer_draw:
            if override_ax is None:
                self.redraw()
            else:
                override_ax.get_figure().canvas.draw_idle()

        return (measured_pos_line_artist, frame_division_epoch_separator_vlines)


    # ==================================================================================================================== #
    # Decoded Position Posteriors                                                                                          #
    # ==================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['WORKING', '2025-02-14_TO_REFACTOR_FROM_NOTEBOOK'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-14 22:02', related_items=[])
    def add_position_posteriors(self, override_ax=None, posterior_masking_value=0.0025, debug_print=False, defer_draw:bool=False, **kwargs):
        """ add the decoded posteriors as heatmaps

        Corresponding to `#### 2025-02-14 - Perform plotting of Decoded Posteriors` in notebook
        
        
        curr_artist_dict, image_extent, plots_data = batch_plot_helper.add_position_posteriors(posterior_masking_value=0.0025, debug_print=True, defer_draw=False)
        """
        _active_plot_fn = kwargs.pop('active_plot_fn', DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap)
        # _active_plot_fn = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap
        # _active_plot_fn = DecodedTrajectoryMatplotlibPlotter._helper_add_hdr_contours

        if override_ax is None:
            active_ax = self.active_ax
        else:
            active_ax = override_ax           

        if not self.has_data_been_built:
            ## finalize building the data for single-artist plotting (does not plot anything)
            self.shared_build_flat_stacked_data(should_expand_first_dim=True, **kwargs)


        # raise NotImplementedError(f'2025-02-14_TO_REFACTOR_FROM_NOTEBOOK')
        # ==================================================================================================================== #
        # Perform Plotting of Posteriors                                                                                       #
        # ==================================================================================================================== #
        
        ## INPUTS: stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers
        a_xbin_centers = deepcopy(self.stacked_flat_xbin_centers)
        a_ybin_centers = deepcopy(self.stacked_flat_ybin_centers)
        a_p_x_given_n = deepcopy(self.stacked_p_x_given_n)
        # a_p_x_given_n = deepcopy(stacked_p_x_given_n).swapaxes(-2, -1)
        if debug_print:
            print(f'np.shape(a_p_x_given_n): {np.shape(a_p_x_given_n)}')

        ## restrict to subrange
        # ==================================================================================================================== #
        # Plot the posterior heatmap                                                                                           #
        # ==================================================================================================================== #
        # custom_image_extent = [0.0, 1.0, 0.0, 1.0]
        custom_image_extent = [self.desired_start_time_seconds, self.desired_end_time_seconds, 0.0, 1.0] ## n
        # (desired_epoch_start_idx, desired_epoch_end_idx), (desired_start_time_seconds, desired_end_time_seconds)

        curr_artist_dict = {}
        ## Perform the plot:
        # curr_artist_dict['prev_heatmaps'], (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data = DecodedTrajectoryMatplotlibPlotter._perform_add_decoded_posterior_and_trajectory(active_ax, xbin_centers=a_xbin_centers, a_p_x_given_n=a_p_x_given_n,
        #                                                                     a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, a_measured_pos_df=a_measured_pos_df, ybin_centers=a_ybin_centers,
        #                                                                     include_most_likely_pos_line=None, time_bin_index=None, rotate_to_vertical=True, should_perform_reshape=False, should_post_hoc_fit_to_image_extent=False, debug_print=True) # , allow_time_slider=True

        # Delegate the posterior plotting functionality.
        curr_artist_dict['prev_heatmaps'], image_extent, plots_data = _active_plot_fn(active_ax,
                                                        xbin_centers=a_xbin_centers, ybin_centers=a_ybin_centers, a_time_bin_centers=None, a_p_x_given_n=a_p_x_given_n,
                                                        posterior_masking_value=posterior_masking_value, rotate_to_vertical=False, debug_print=True, should_perform_reshape=False, custom_image_extent=custom_image_extent, extant_plot_data=kwargs.get('extant_plot_data', None))


        if not defer_draw:
            if override_ax is None:
                self.redraw()
            else:
                override_ax.get_figure().canvas.draw_idle()
                
        return curr_artist_dict, image_extent, plots_data

    # ==================================================================================================================== #
    # Track Shape Plotting                                                                                                 #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['track_shapes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 11:31', related_items=[])
    def add_track_shapes(self, global_session, override_ax=None, debug_print:bool=True, defer_draw:bool=False):
        """ 
        global_session: needed to build track shapes
        
        
        Uses:
        
            self.track_all_normalized_rect_arr_dict
            self.inverse_normalized_track_all_rect_arr_dict
        
            
            
        track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session)
        
        Usage:
            track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session) ## does not seem to successfully synchronize to window
        
        """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, long_short_display_config_manager
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

        if override_ax is None:
            active_ax = self.active_ax
        else:
            active_ax = override_ax

        is_filtered: bool = False
        ## INPUTS: track_ax, rotate_to_vertical, perform_autoscale
        # frame_divide_bin_size: float = self.frame_divide_bin_size
        ## Slice a subset of the data epochs:
        desired_epoch_start_idx: int = self.desired_epoch_start_idx
        # desired_epoch_end_idx: int = 20
        # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
        if self.desired_epoch_end_idx is not None:
            desired_epoch_end_idx: int = self.desired_epoch_end_idx
            filtered_epoch_range: NDArray = np.arange(start=desired_epoch_start_idx, stop=desired_epoch_end_idx)
            filtered_num_horizontal_repeats: int = len(filtered_epoch_range)
                        
            is_filtered = (filtered_num_horizontal_repeats < self.num_horizontal_repeats)
        else:
            # raise NotImplementedError('oops')
            # desired_epoch_end_idx: int = self.num_filter_epochs
            desired_epoch_end_idx: int = None
            filtered_num_horizontal_repeats: int = self.num_horizontal_repeats
            
        is_filtered = (filtered_num_horizontal_repeats < self.num_horizontal_repeats)
        print(f'desired_epoch_start_idx: {desired_epoch_start_idx}, desired_epoch_end_idx: {desired_epoch_end_idx}')
        # filtered_num_output_rect_total_elements: int = filtered_num_horizontal_repeats * 3 # 3 parts to each track plot
        ## OUTPUTS: filtered_epoch_range, filtered_num_horizontal_repeats, filtered_num_output_rect_total_elements
        # if debug_print:
        #     print(f'filtered_num_output_rect_total_elements: {filtered_num_output_rect_total_elements}')
        
        ## Update `batch_plot_helper.custom_image_extent`
        self.custom_image_extent = [self.desired_start_time_seconds, self.desired_end_time_seconds, 0.0, 1.0] ## n
        num_horizontal_repeats: int = self.num_horizontal_repeats
        
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(deepcopy(global_session.config))

        long_short_display_config_manager = LongShortDisplayConfigManager()
        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        long_kwargs = deepcopy(long_epoch_matplotlib_config)
        long_kwargs = overriding_dict_with(lhs_dict=long_kwargs, **dict(linewidth=2, zorder=-99, alpha=0.5, facecolor='#0099ff07', edgecolor=long_kwargs['facecolor'], linestyle='dashed'))
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
        short_kwargs = deepcopy(short_epoch_matplotlib_config)
        short_kwargs = overriding_dict_with(lhs_dict=short_kwargs, **dict(linewidth=2, zorder=-98, alpha=0.5, facecolor='#f5161607', edgecolor=short_kwargs['facecolor'], linestyle='dashed'))
        track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}

        # BEGIN PLOTTING _____________________________________________________________________________________________________ #
        # long_out_tuple = long_track_inst.plot_rects(plot_item=track_ax, matplotlib_rect_kwargs_override=long_kwargs, rotate_to_vertical=rotate_to_vertical, offset=None)
        # short_out_tuple = short_track_inst.plot_rects(plot_item=track_ax, matplotlib_rect_kwargs_override=short_kwargs, rotate_to_vertical=rotate_to_vertical, offset=None)
        # long_combined_item, long_rect_items, long_rects = long_out_tuple
        # short_combined_item, short_rect_items, short_rects = short_out_tuple

        long_rects = long_track_inst.build_rects(include_rendering_properties=False, rotate_to_vertical=self.rotate_to_vertical)
        short_rects = short_track_inst.build_rects(include_rendering_properties=False, rotate_to_vertical=self.rotate_to_vertical)
        self.track_single_rects_dict = {'long': long_rects, 'short': short_rects}

        # long_path = _build_track_1D_verticies(platform_length=22.0, track_length=170.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=long_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=-1.0, debug_print=True)
        # short_path = _build_track_1D_verticies(platform_length=22.0, track_length=100.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=short_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=1.0, debug_print=True)

        # ## Plot the tracks:
        # long_patch = patches.PathPatch(long_path, **long_track_color, alpha=0.5, lw=2)
        # ax.add_patch(long_patch)

        # short_patch = patches.PathPatch(short_path, **short_track_color, alpha=0.5, lw=2)
        # ax.add_patch(short_patch)
        # if perform_autoscale:
        #     track_ax.autoscale()
        
        # x_offset: float = -131.142
        # long_rect_arr = SingleArtistMultiEpochBatchHelpers.rect_tuples_to_NDArray(long_rects, x_offset=x_offset)
        # short_rect_arr = SingleArtistMultiEpochBatchHelpers.rect_tuples_to_NDArray(short_rects, x_offset=x_offset)


        # num_horizontal_repeats: int = 20 ## hardcoded
        self.track_all_normalized_rect_arr_dict = SingleArtistMultiEpochBatchHelpers.track_dict_all_stacked_rect_arr_normalization(self.track_single_rects_dict, num_horizontal_repeats=num_horizontal_repeats)
        ## INPUTS: filtered_num_horizontal_repeats
        # self.inverse_normalized_track_all_rect_arr_dict = SingleArtistMultiEpochBatchHelpers.track_dict_all_stacked_rect_arr_inverse_normalization(self.track_all_normalized_rect_arr_dict, ax=active_ax, num_active_horizontal_repeats=num_horizontal_repeats)
        self.inverse_normalized_track_all_rect_arr_dict = SingleArtistMultiEpochBatchHelpers.track_dict_all_stacked_rect_arr_inverse_normalization_from_custom_extent(self.track_all_normalized_rect_arr_dict, custom_image_extent=self.custom_image_extent, num_active_horizontal_repeats=num_horizontal_repeats)


        ## OUTPUTS: track_all_normalized_rect_arr_dict, inverse_normalized_track_all_rect_arr_dict
        # track_all_normalized_rect_arr_dict

        # ## Slice a subset of the data epochs:
        if is_filtered:
            # desired_epoch_start_idx: int = 0
            # # desired_epoch_end_idx: int = 20
            # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
            # print(f'desired_epoch_start_idx: {desired_epoch_start_idx}, desired_epoch_end_idx: {desired_epoch_end_idx}')

            track_all_rect_arr_dict = {k:v[(desired_epoch_start_idx*3):(desired_epoch_end_idx*3), :] for k, v in self.track_all_normalized_rect_arr_dict.items()}
            # track_all_rect_arr_dict = {k:v[desired_epoch_start_idx:desired_epoch_end_idx, :] for k, v in track_all_rect_arr_dict.items()}
            # track_all_rect_arr_dict

            ## INPUTS: filtered_num_horizontal_repeats
            # self.inverse_normalized_track_all_rect_arr_dict = SingleArtistMultiEpochBatchHelpers.track_dict_all_stacked_rect_arr_inverse_normalization(track_all_rect_arr_dict, ax=active_ax, num_active_horizontal_repeats=filtered_num_horizontal_repeats)
            self.inverse_normalized_track_all_rect_arr_dict = SingleArtistMultiEpochBatchHelpers.track_dict_all_stacked_rect_arr_inverse_normalization_from_custom_extent(track_all_rect_arr_dict, custom_image_extent=self.custom_image_extent, num_active_horizontal_repeats=filtered_num_horizontal_repeats)
            ## OUTPUTS: inverse_normalized_track_all_rect_arr_dict
            

        ## INPUTS: track_kwargs_dict, inverse_normalized_track_all_rect_arr_dict
        track_shape_patch_collection_artists = SingleArtistMultiEpochBatchHelpers.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=self.inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict) # start (x0: 0.0, 20 of them span to exactly x=1.0)
        # track_shape_patch_collection_artists = SingleArtistMultiEpochBatchHelpers.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict, transform=ax.transData) # start (x0: 31.0, 20 of them span to about x=1000.0)
        # track_shape_patch_collection_artists = SingleArtistMultiEpochBatchHelpers.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict, transform=ax.transAxes) # start (x0: 31.0, 20 of them span to about x=1000.0)
        
        if not defer_draw:
            if override_ax is None:
                self.redraw()
            else:
                override_ax.get_figure().canvas.draw_idle()

        return track_shape_patch_collection_artists



    @function_attributes(short_name=None, tags=['MAIN', 'WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 02:29', related_items=[])
    def add_all_track_plots(self, global_session, override_ax=None, posterior_masking_value=0.0025, debug_print=False, defer_draw:bool=False, **kwargs) -> RenderPlotsData:
        """ performs all plotting on the same axes """
        
        self.shared_build_flat_stacked_data(force_recompute=True, debug_print=debug_print)
        
        # plot_data = MatplotlibRenderPlots(name='_perform_add_decoded_posterior_and_trajectory')
        # plots = RenderPlots('_perform_add_decoded_posterior_and_trajectory')
        plots_data: RenderPlotsData = RenderPlotsData(name='SingleArtistMultiEpochBatchHelpers', image_extent=None, curr_artist_dict=None,
                                                      track_shape_patch_collection_artists=None,
                                                      measured_pos_line_artist=None, frame_division_epoch_separator_vlines=None,
                                                       ) #deepcopy(extra_dict) # RenderPlotsData(name='_perform_add_decoded_posterior_and_trajectory', image_extent=deepcopy(image_extent))




        try:
            plots_data.track_shape_patch_collection_artists = self.add_track_shapes(global_session=global_session, override_ax=override_ax, defer_draw=True, debug_print=debug_print) ## does not seem to successfully synchronize to window
        except KeyError as e:
            # KeyError: 'long_xlim', for non kdiba tracks
            print(f'WARN: non-kdiba track, cannot draw analytical track shape due to exception e: {e}')
        except Exception as e:
            raise e


        # track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session, override_ax=track_shapes_dock_track_ax) ## does not seem to successfully synchronize to window
        plots_data.curr_artist_dict, plots_data.image_extent, plots_data = self.add_position_posteriors(posterior_masking_value=posterior_masking_value, override_ax=override_ax, debug_print=debug_print, defer_draw=True, extant_plot_data=plots_data)

        measured_pos_line_artist, frame_division_epoch_separator_vlines = self.add_track_positions(override_ax=override_ax, debug_print=debug_print, defer_draw=True)
        # measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions(override_ax=measured_pos_dock_track_ax)
        plots_data.measured_pos_line_artist = measured_pos_line_artist
        plots_data.frame_division_epoch_separator_vlines = frame_division_epoch_separator_vlines
        
        # plots_data.curr_artist_dict['measured_pos_line_artist'] = measured_pos_line_artist
        # plots_data.curr_artist_dict['frame_division_epoch_separator_vlines'] = frame_division_epoch_separator_vlines
        
        # plot_obj = RasterPlots()
        
        if not defer_draw:
            if override_ax is None:
                self.redraw()
            else:
                override_ax.get_figure().canvas.draw_idle()

        return plots_data

    # ==================================================================================================================== #
    # Utility                                                                                                              #
    # ==================================================================================================================== #

    def redraw(self):
        """ re-draws the attached axes """
        self.active_ax.get_figure().canvas.draw_idle()
        
    def clear_all_artists(self):
        """ clears all added artists. """
        self.active_ax.clear()
        self.redraw()
        

    @function_attributes(short_name=None, tags=['reshape', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 05:57', related_items=[])
    @classmethod
    def reshape_p_x_given_n_for_single_artist_display(cls, updated_timebins_p_x_given_n: NDArray, rotate_to_vertical: bool = True, should_expand_first_dim: bool=True, debug_print=False) -> NDArray:
        """ 
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import reshape_p_x_given_n_for_single_artist_display
        
        """
        stacked_p_x_given_n = deepcopy(updated_timebins_p_x_given_n) # drop the last epoch
        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (76, 40, 33008)
        stacked_p_x_given_n = np.moveaxis(stacked_p_x_given_n, -1, 0) # move the n_t dimension/axis (which starts as last) to be first (0th)
        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (33008, 76, 40)

        n_xbins, n_ybins, n_tbins = np.shape(stacked_p_x_given_n) # (76, 40, 29532)        
        if not rotate_to_vertical:
            stacked_p_x_given_n = np.row_stack(stacked_p_x_given_n) # .shape: (99009, 39) - ((n_xbins*n_tbins), n_ybins)
            # stacked_p_x_given_n = np.swapaxes(stacked_p_x_given_n, 1, 2).reshape((-1, n_ybins))
        else:
            ## display with y-axis along the primary axis=1
            stacked_p_x_given_n = np.column_stack(stacked_p_x_given_n) # .shape: (n_xbins, (n_ybins*n_tbins))
            stacked_p_x_given_n = stacked_p_x_given_n.T.T
            # stacked_p_x_given_n = stacked_p_x_given_n.reshape(stacked_p_x_given_n.shape[0], stacked_p_x_given_n.shape[1] * stacked_p_x_given_n.shape[2]) # .shape: (n_xbins, (n_ybins*n_tbins))

        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (2508608, 40)
            
        if should_expand_first_dim:
            stacked_p_x_given_n = np.expand_dims(stacked_p_x_given_n, axis=0)
            if debug_print:
                print(np.shape(stacked_p_x_given_n)) # (1, 2508608, 40)
        return stacked_p_x_given_n

    @classmethod
    def _slice_to_epoch_range(cls, flat_timebins_p_x_given_n, flat_time_bin_centers, desired_epoch_start_idx: int = 0, desired_epoch_end_idx: int = 15):
        """ trims down to a specific epoch range """
        flat_timebins_p_x_given_n = flat_timebins_p_x_given_n[:, :, desired_epoch_start_idx:desired_epoch_end_idx]
        flat_time_bin_centers = flat_time_bin_centers[desired_epoch_start_idx:desired_epoch_end_idx]
        return flat_timebins_p_x_given_n, flat_time_bin_centers


    @classmethod
    def complete_build_stacked_flat_arrays(cls, a_result: "DecodedFilterEpochsResult", a_new_global_decoder, desired_epoch_start_idx:int=0, desired_epoch_end_idx: Optional[int] = None, rotate_to_vertical: bool = True, should_expand_first_dim: bool=True):
        """ 
        a_result: DecodedFilterEpochsResult = frame_divided_epochs_specific_decoded_results_dict['global']
        a_new_global_decoder = new_decoder_dict['global']
        # delattr(a_result, 'measured_positions_list')
        a_result.measured_positions_list = deepcopy([global_pos_df[global_pos_df['global_frame_division_idx'] == epoch_idx] for epoch_idx in np.arange(a_result.num_filter_epochs)]) ## add a List[pd.DataFrame] to plot as the measured positions
        rotate_to_vertical: bool = True
        should_expand_first_dim: bool=True
        (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers) = SingleArtistMultiEpochBatchHelpers.complete_build_stacked_flat_arrays(a_result=a_result, a_new_global_decoder=a_new_global_decoder, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)

        
        # Example 2: Filtering to epochs: [0, 20]
        rotate_to_vertical: bool = True
        should_expand_first_dim: bool=True
        (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers) = SingleArtistMultiEpochBatchHelpers.complete_build_stacked_flat_arrays(a_result=a_result, a_new_global_decoder=a_new_global_decoder,
                                                                                                                                                                                                                                                                                desired_epoch_end_idx=20, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)
                                                                                                                                                                                                                                                                                
        """
        n_timebins, flat_time_bin_containers, flat_timebins_p_x_given_n = a_result.flatten()
        flat_time_bin_containers = flat_time_bin_containers.tolist()
        flat_time_bin_centers: NDArray = np.hstack([v.centers for v in flat_time_bin_containers])

        # np.shape(flat_time_bin_containers) # (1738,)
        timebins_p_x_given_n_shape = np.shape(flat_timebins_p_x_given_n) # (76, 40, 29532)
        n_xbins, n_ybins, n_tbins = timebins_p_x_given_n_shape
        # (n_xbins, n_ybins, n_tbins)
        # np.shape(flat_time_bin_centers) # (29532,)

        
        if desired_epoch_end_idx is not None:
            ## Filter if desired:
            flat_timebins_p_x_given_n, flat_time_bin_centers = cls._slice_to_epoch_range(flat_timebins_p_x_given_n=flat_timebins_p_x_given_n, flat_time_bin_centers=flat_time_bin_centers, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)
        
        flattened_timebins_p_x_given_n_shape = np.shape(flat_timebins_p_x_given_n) # (76, 40, 29532)
        n_xbins, n_ybins, n_tbins = flattened_timebins_p_x_given_n_shape ## MUST BE UPDATED POST SLICE
        # (n_xbins, n_ybins, n_tbins)

        # flattened_n_xbins, flattened_n_ybins, flattened_n_tbins = flattened_timebins_p_x_given_n_shape
        # (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins)
        # np.shape(flat_time_bin_centers) # (29532,)
        ## OUTPUTS: flat_p_x_given_n, flat_time_bin_centers, 
        stacked_p_x_given_n = cls.reshape_p_x_given_n_for_single_artist_display(flat_timebins_p_x_given_n, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim) # (1, 57, 90)
        
        # np.shape(stacked_p_x_given_n) # (1, 2244432, 40)
        

        xbin_centers = deepcopy(a_new_global_decoder.xbin_centers)
        ybin_centers = deepcopy(a_new_global_decoder.ybin_centers)

        if not rotate_to_vertical:
            stacked_flat_time_bin_centers = flat_time_bin_centers.repeat(n_xbins) # ((n_xbins*n_tbins), ) -- both are original sizes
            stacked_flat_xbin_centers = deepcopy(xbin_centers).repeat(n_tbins)  
            stacked_flat_ybin_centers = deepcopy(ybin_centers)         
        else:
            # vertically-oriented tracks (default)
            stacked_flat_time_bin_centers = flat_time_bin_centers.repeat(n_ybins) # ((n_ybins*n_tbins), ) -- both are original sizes
            stacked_flat_xbin_centers = deepcopy(xbin_centers)
            stacked_flat_ybin_centers = deepcopy(ybin_centers).repeat(n_tbins) ## these will lay along the x-axis

        flattened_n_xbins = len(stacked_flat_xbin_centers)
        flattened_n_ybins = len(stacked_flat_ybin_centers)
        flattened_n_tbins = len(stacked_flat_time_bin_centers)
        # (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins)

        if should_expand_first_dim:
            stacked_flat_time_bin_centers = np.expand_dims(stacked_flat_time_bin_centers, axis=0) # (1, (n_xbins*n_tbins)) or (1, (n_ybins*n_tbins)) -- both are original sizes

        # np.shape(stacked_flat_time_bin_centers) # (1, (n_ybins*n_tbins))
        ## OUPTUTS: (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers)
        return (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers)


    @classmethod
    @function_attributes(short_name=None, tags=['masked_rows', 'nan', 'position_lines', 'stacked_flat_global_pos_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 23:56', related_items=[])
    def add_nan_masked_rows_to_stacked_flat_global_pos_df(cls, stacked_flat_global_pos_df: pd.DataFrame) -> pd.DataFrame:
        """ seperates each 'global_frame_division_idx' change in the df by adding two NaN rows with ['is_masked_bin'] = True 

        stacked_flat_global_pos_df['global_frame_division_idx'] ## find rows in the dataframe where the 'global_frame_division_idx' column changes values
        ## insert a new row into the dataframe between the two changing rows: where the new row's 't' = (prev_row_t + 1e-6)

        Usage:
        
            new_stacked_flat_global_pos_df = SingleArtistMultiEpochBatchHelpers.add_nan_masked_rows_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=stacked_flat_global_pos_df)
            new_stacked_flat_global_pos_df
        """        
        new_stacked_flat_global_pos_df = deepcopy(stacked_flat_global_pos_df)
        # print(list(new_stacked_flat_global_pos_df.columns))
        column_names_to_copy = ['t', 'global_frame_division_idx', 'frame_division_epoch_start_t']
        column_names_to_update = ['t', 'dt', 'is_masked_bin']
        nan_column_names = ['x', 'y', 'lin_pos', 'speed', 'lap', 'lap_dir', 'velocity_x', 'acceleration_x', 'velocity_y', 'acceleration_y', 'x_smooth', 'y_smooth', 'velocity_x_smooth', 'acceleration_x_smooth', 'velocity_y_smooth', 'acceleration_y_smooth', 'binned_x', 'binned_y', 'global_frame_division_x_unit_offset', 'global_frame_division_x_data_offset', 'x_scaled', 'x_smooth_scaled', 'y_scaled']
        # nan_column_names = ['x', 'y', 'lin_pos', 'speed', 'lap', 'lap_dir', 'x_smooth', 'y_smooth', 'binned_x', 'binned_y', 'global_frame_division_x_unit_offset', 'global_frame_division_x_data_offset', 'x_scaled', 'x_smooth_scaled', 'y_scaled']
        included_nan_column_names = [k for k in nan_column_names if k in new_stacked_flat_global_pos_df.columns]

        new_stacked_flat_global_pos_df['is_masked_bin'] = False
        
        # bad_color = '#000000'
        bad_color = (0.0, 0.0, 0.0, 0.0)
        color_formatting_dict = {}

        dfs = []
        prev = None
        for _, row in new_stacked_flat_global_pos_df.iterrows():
            # is_global_frame_division_idx_changing: bool = (row['global_frame_division_idx'] != prev['global_frame_division_idx'])
            if (prev is not None) and (row['global_frame_division_idx'] != prev['global_frame_division_idx']):
                new_row = prev.copy()
                new_row['t'] = prev['t'] + 1e-6
                new_row[included_nan_column_names] = np.nan
                new_row['is_masked_bin'] = True
                new_row['color'] = deepcopy(bad_color)
                dfs.append(new_row.to_frame().T) 
                ## add following row - I'd also like to add a duplicate of the next_row but with new_row['t'] = next['t'] - 1e-6
                new_next = row.copy()
                new_next['t'] = row['t'] - 1e-6
                new_next[included_nan_column_names] = np.nan
                new_next['is_masked_bin'] = True
                new_next['color'] = deepcopy(bad_color)
                dfs.append(new_next.to_frame().T)

            dfs.append(row.to_frame().T)
            prev = row
            
        new_stacked_flat_global_pos_df = pd.concat(dfs, ignore_index=True).infer_objects()
        ## convert columns back from 'object' to 'float64'
        return new_stacked_flat_global_pos_df

    @classmethod
    @function_attributes(short_name=None, tags=['masked_rows', 'nan', 'position_lines', 'stacked_flat_global_pos_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 23:56', related_items=[])
    def add_color_over_global_frame_division_idx_positions_to_stacked_flat_global_pos_df(cls, stacked_flat_global_pos_df: pd.DataFrame, time_cmap='viridis') -> pd.DataFrame:
        """ seperates each 'global_frame_division_idx' change in the df by adding two NaN rows with ['is_masked_bin'] = True 
        Usage:
        
            stacked_flat_global_pos_df = SingleArtistMultiEpochBatchHelpers.add_color_over_global_frame_division_idx_positions_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=stacked_flat_global_pos_df, time_cmap='viridis')
            stacked_flat_global_pos_df
        """
        if isinstance(time_cmap, str):
            time_cmap = plt.get_cmap(time_cmap)  # Choose a colormap

        group_min = stacked_flat_global_pos_df.groupby('global_frame_division_idx')['t'].transform('min')
        group_max = stacked_flat_global_pos_df.groupby('global_frame_division_idx')['t'].transform('max')
        normed = (stacked_flat_global_pos_df['t'] - group_min) / (group_max - group_min)
        stacked_flat_global_pos_df['color'] = normed.apply(lambda x: time_cmap(x)) ## updates the 'color' column
        return stacked_flat_global_pos_df


    # ==================================================================================================================== #
    # Batch Track Shape Plotting                                                                                           #
    # ==================================================================================================================== #
    @classmethod
    def rect_tuples_to_NDArray(cls, rects, x_offset:float=0.0) -> NDArray:
        """ .shape (3, 4) """
        return np.vstack([[x+x_offset, y, w, h] for x, y, w, h, *args in rects])
        
    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=['cls.all_stacked_rect_arr_normalization'], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def rect_arr_normalization(cls, a_rect_arr, debug_print=False) -> NDArray:
        """ Normalizes the offsets and size to [0, 1]
        .shape (3, 4)
        
        Usage:
            Example 1:        
                normalized_long_rect_arr, ((x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total) = SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(long_rect_arr)
                normalized_long_rect_arr

            Example 2:
                track_single_rect_arr_dict = {'long': long_rect_arr, 'short': short_rect_arr}
                track_single_rect_arr_dict
                track_single_normalized_rect_arr_dict = {k:SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(v)[0] for k, v in track_single_rect_arr_dict.items()}
                track_normalization_tuple_dict = {k:SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(v)[1] for k, v in track_single_rect_arr_dict.items()}
                track_single_normalized_rect_arr_dict
                track_normalization_tuple_dict

        """
        if debug_print:
            print(f'a_rect_arr: {a_rect_arr}, np.shape(a_rect_arr): {np.shape(a_rect_arr)}')
            
        x0_offset: float = a_rect_arr[0, 0]
        y0_offset: float = a_rect_arr[0, 1]
        w0_multiplier: float = a_rect_arr[0, 2]
        h0_total: float = np.sum(a_rect_arr, axis=0)[3]

        if debug_print:
            print(f'x0_offset: {x0_offset}, y0_offset: {y0_offset}, w0_multiplier: {w0_multiplier}, h0_total: {h0_total}')
            
        ## normalize plotting by these values:
        normalized_long_rect_arr = deepcopy(a_rect_arr)
        normalized_long_rect_arr[:, 2] /= w0_multiplier
        normalized_long_rect_arr[:, 3] /= h0_total
        normalized_long_rect_arr[:, 0] /= w0_multiplier
        normalized_long_rect_arr[:, 1] /= h0_total
        if debug_print:
            print(f'normalized_long_rect_arr: {normalized_long_rect_arr}')

        normalized_x0_offset: float = normalized_long_rect_arr[0, 0]
        normalized_y0_offset: float = normalized_long_rect_arr[0, 1]
        if debug_print:
            print(f'normalized_x0_offset: {normalized_x0_offset}, normalized_y0_offset: {normalized_y0_offset}')
        
        ## only after scaling should we apply the translational offset
        normalized_long_rect_arr[:, 0] -= normalized_x0_offset
        normalized_long_rect_arr[:, 1] -= normalized_y0_offset

        # ## raw tanslational offset
        # normalized_long_rect_arr[:, 0] -= x0_offset
        # normalized_long_rect_arr[:, 1] -= y0_offset

        return normalized_long_rect_arr, ((x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total)


    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=['cls.rect_tuples_to_NDArray', 'cls.rect_arr_normalization'], used_by=['cls.track_dict_all_stacked_rect_arr_normalization'], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def all_stacked_rect_arr_normalization(cls, built_track_rects, num_horizontal_repeats: int, x_offset: float = 0.0) -> NDArray:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        a_track_rect_arr = cls.rect_tuples_to_NDArray(built_track_rects, x_offset=x_offset)
        # x0s = a_track_rect_arr[:, 0] # x0
        # widths = a_track_rect_arr[:, 2] # w
        # heights = a_track_rect_arr[:, 3] # h

        ## INPUTS: track_single_normalized_rect_arr_dict, track_normalization_tuple_dict

        # active_track_name: str = 'long'
        track_single_normalized_rect_arr, track_normalization_tuple = SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(a_track_rect_arr)
        (x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total = track_normalization_tuple ## unpack track_normalization_tuple

        single_subdiv_normalized_width = 1.0
        single_subdiv_normalized_height = 1.0
        single_subdiv_normalized_offset_x = 1.0

        test_arr = []
        for epoch_idx in np.arange(num_horizontal_repeats):
            an_arr = deepcopy(track_single_normalized_rect_arr)
            an_arr[:, 0] += (epoch_idx * single_subdiv_normalized_offset_x) ## set offset 
            test_arr.append(an_arr)
            
        test_arr = np.vstack(test_arr)
        # np.shape(test_arr) # (5211, 4)
        return test_arr
            

    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=['cls.all_stacked_rect_arr_normalization'], used_by=[], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def track_dict_all_stacked_rect_arr_normalization(cls, built_track_rects_dict, num_horizontal_repeats: int) -> Dict[str, NDArray]:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        track_all_normalized_rect_arr_dict = {}
        for active_track_name, built_track_rects in built_track_rects_dict.items():
            track_all_normalized_rect_arr_dict[active_track_name] = cls.all_stacked_rect_arr_normalization(built_track_rects=built_track_rects, num_horizontal_repeats=num_horizontal_repeats)

        ## OUTPUTS: track_all_normalized_rect_arr_dict
        return track_all_normalized_rect_arr_dict

    @function_attributes(short_name=None, tags=['NEWEST', 'active', 'inverse', 'extent'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 22:02', related_items=[])
    @classmethod
    def track_dict_all_stacked_rect_arr_inverse_normalization_from_custom_extent(cls, track_all_rect_arr_dict, custom_image_extent: List[float], num_active_horizontal_repeats: int) -> Dict[str, NDArray]:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        assert len(custom_image_extent), f"custom_image_extent: {custom_image_extent} but should be of the form: [x0, y0, width, height]"
        # ax_width: float = custom_image_extent[2] ## how wide the current window is
        # ax_height: float = custom_image_extent[3]
        # x0, y0, ax_width, ax_height = custom_image_extent
        
        x0, x1, y0, y1 = custom_image_extent
        ax_width: float = x1 - x0
        ax_height: float = y1 - y0
        
        # assert x0 == 0.0, f"x0 should be equal to zero (no offsets allowed) but instead it is equal to {x0}"
        # assert y0 == 0.0, f"y0 should be equal to zero (no offsets allowed) but instead it is equal to {y0}"
        
        # (xlim, ylim)
        # (ax_width, ax_height)

        inverse_normalization_factor_width: float = ax_width / num_active_horizontal_repeats
        inverse_normalization_factor_height: float = 1.0 / ax_height

        # (inverse_normalization_factor_width, inverse_normalization_factor_height)
        
        ## OUTPUTS: inverse_normalization_factor_width, inverse_normalization_factor_height

        # ax.get_width()
        inverse_normalized_track_all_rect_arr_dict = {}

        for k, test_arr in track_all_rect_arr_dict.items():
            new_test_arr = deepcopy(test_arr)
            # ## subtract out the offset
            # new_test_arr[:, 0] -= x0
            # new_test_arr[:, 1] -= y0
            
            new_test_arr[:, 2] *= inverse_normalization_factor_width # scale by the width
            new_test_arr[:, 0] *= inverse_normalization_factor_width

            new_test_arr[:, 3] *= inverse_normalization_factor_height # scale by the width
            new_test_arr[:, 1] *= inverse_normalization_factor_height

            inverse_normalized_track_all_rect_arr_dict[k] = new_test_arr
            
        return inverse_normalized_track_all_rect_arr_dict
    
    @function_attributes(short_name=None, tags=['main', 'new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 09:16', related_items=[])
    @classmethod
    def add_batch_track_shapes(cls, ax, inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict, transform=None):
        """ 
        
        track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}
        track_shape_patch_collection_artists = SingleArtistMultiEpochBatchHelpers.add_batch_track_shapes(ax=ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict)
        fig.canvas.draw_idle()
        
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        # import matplotlib.patches as patches
        assert track_kwargs_dict is not None

        extra_transform_kwargs = {}
        if transform is not None:
            extra_transform_kwargs['transform'] = transform
        
        track_names_list = ['long', 'short']
        # track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}
        track_shape_patch_collection_artists = {'long': None, 'short': None}

        for active_track_name in track_names_list:
            # matplotlib_rect_kwargs_override = long_kwargs # {'linewidth': 2, 'edgecolor': '#0099ff42', 'facecolor': '#0099ff07'}

            matplotlib_rect_kwargs = track_kwargs_dict[active_track_name] # {'linewidth': 2, 'edgecolor': '#0099ff42', 'facecolor': '#0099ff07'}
            # active_all_rect_arr = track_all_rect_arr_dict[active_track_name]
            active_all_rect_arr = inverse_normalized_track_all_rect_arr_dict[active_track_name]

            # matplotlib ax was passed
            data = deepcopy(active_all_rect_arr)
            # rect_patches = [Rectangle((x, y), w, h) for x, y, w, h in data]
            rect_patches = [Rectangle((x, y), w, h, **matplotlib_rect_kwargs, **extra_transform_kwargs) for x, y, w, h in data] # , transform=ax.transData, transform=ax.transData
            
            # ## legacy patch-based way
            # rect = patches.Rectangle((x, y), w, h, **matplotlib_rect_kwargs)
            # plot_item.add_patch(rect)    

            # pc = PatchCollection(patches, edgecolors='k', facecolors='none')
            if track_shape_patch_collection_artists.get(active_track_name, None) is not None:
                # remove extant
                print(f'removing existing artist.')
                track_shape_patch_collection_artists[active_track_name].remove()
                track_shape_patch_collection_artists[active_track_name] = None

            # pc = PatchCollection(rect_patches, edgecolors=matplotlib_rect_kwargs.get('edgecolor', '#0099ff42'), facecolors=matplotlib_rect_kwargs.get('facecolor', '#0099ff07'))
            pc = PatchCollection(rect_patches, match_original=True) #, transform=ax.transAxes , transform=ax.transData
            track_shape_patch_collection_artists[active_track_name] = pc
            ax.add_collection(pc)
        ## END for active_track_name in track_names_list:

        # plt.gca().add_collection(pc)
        # plt.show()
        # ax.get_figure()
        # fig.canvas.draw_idle()
        
        return track_shape_patch_collection_artists
