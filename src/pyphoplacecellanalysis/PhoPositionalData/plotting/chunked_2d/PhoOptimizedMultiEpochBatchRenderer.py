from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, BasePositionDecoder
    from nptyping import NDArray

from copy import deepcopy
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

import numpy as np
import pyqtgraph
import pyqtgraph as pg
from PyQt5 import QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlotsData, VisualizationParameters
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter

from pyphocorehelpers.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets
from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import Colormap1DEditorWidget, EditableColormap2DEditorWidget

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

# ==================================================================================================================== #
# 2024-04-12 - Decoded Trajectory Plotting on Maze (1D & 2D) - Posteriors and Most Likely Position Paths               #
# ==================================================================================================================== #

class RenderColoringMode(str, Enum):
    """How to color rendered path elements (e.g. line segments, arrows): by time (colormap), by speed, or by heading angle (ROYGBIV, North=Red)."""
    TIME = 'time'
    SPEED = 'speed'
    ANGLE = 'angle'


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult

from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs


# ==================================================================================================================================================================================================================================================================================== #
# TODO 2025-12-16 16:37: - [ ] AI-implemnented attempt to replace Aims to replace `PhoOptimizedMultiEpochBatchRenderer` with a much more efficient implementation                                                                                                                       #
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


# ==================================================================================================================================================================================================================================================================================== #
# 3D CMap Generation Helpers Temp                                                                                                                                                                                                                                                      #
# ==================================================================================================================================================================================================================================================================================== #

from pyphocorehelpers.gui.Qt.color_helpers import create_3d_lut_saturation, create_3d_lut_cmaps_interp, apply_3d_colormap, composite_stack


# ==================================================================================================================================================================================================================================================================================== #
# PhoOptimizedMultiEpochBatchRenderer - main class                                                                                                                                                                                                                                     #
# ==================================================================================================================================================================================================================================================================================== #

@metadata_attributes(short_name=None, tags=['OLD', '2D_timeseries', '2D_posteriors', 'frames', 'UNFINISHED', 'KINDA-WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-19 00:00', related_items=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side'])
@define(slots=False, eq=False)
class PhoOptimizedMultiEpochBatchRenderer:
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
        
    
    from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PhoOptimizedMultiEpochBatchRenderer import PhoOptimizedMultiEpochBatchRenderer
    
    
    
    
    USAGE:
    
        from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
        from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PhoOptimizedMultiEpochBatchRenderer import PhoOptimizedMultiEpochBatchRenderer
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode

        ## INPUTS: frame_divide_bin_size, frame_divided_epochs_result, decoder, pos_df (or use init_from_results2D)

        track_name: str = 'CustomBatch2Dto1DTimeline'
        spike_raster_plt_2d: Spike2DRaster = spike_raster_window.spike_raster_plt_2d
        a_time_sync_pyqtgraph_widget, track_root_graphics_layout_widget, track_plot_item, dDisplayItem = spike_raster_plt_2d.add_new_embedded_pyqtgraph_render_plot_widget(name=track_name, dockSize=(500,50), sync_mode=SynchronizedPlotMode.TO_WINDOW)

        an_epoch_name: str = 'roam'
        a_decoder = masked_container.pf1D_Decoder_dict[an_epoch_name]
        a_result = masked_container.epochs_decoded_result_cache_dict[0.025][an_epoch_name]

        subdivide_bin_size: float = 5.0  # seconds
        split_column_name: str = 'global_frame_division_idx'
        pos_df, subdivided_epochs_df, maze_bounds_t, pos_tspace_df, (xt, yt) = PhoOptimizedMultiEpochBatchRenderer.build_transforms_for_frames(a_decoder=a_decoder, pos_df=pos_df, subdivide_bin_size=subdivide_bin_size, split_column_name=split_column_name)
        # subdivided_epochs_df
        # pos_tspace_df

        ## INPUTS: pos_df, subdivided_epochs_df, maze_bounds_t, (xt, yt)
        _out_dict = PhoOptimizedMultiEpochBatchRenderer.plot_all(subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                                                                    pos_tspace_df=pos_tspace_df,# xt=xt, yt=yt,
                                                                    a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result,
                                                                    track_plot_item=track_plot_item,
                                                                )



    Pre 2026:
        ## INPUTS: frame_divide_bin_size, frame_divided_epochs_result, decoder, pos_df (or use init_from_results2D)
        batch_plot_helper: PhoOptimizedMultiEpochBatchRenderer = PhoOptimizedMultiEpochBatchRenderer.init_from_results2D(results2D, active_ax=track_ax, frame_divide_bin_size=frame_divide_bin_size, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)
        plots_data = batch_plot_helper.add_all_track_plots(global_session=global_session)
        
    
            
    Usage -- Individual Components:
        desired_epoch_start_idx: int = 0
        # desired_epoch_end_idx: int = int(round(1/frame_divide_bin_size)) * 60 * 8 # 8 minutes
        desired_epoch_end_idx: Optional[int] = None

        ## INPUTS: frame_divide_bin_size, frame_divided_epochs_result, decoder, pos_df (or use init_from_results2D)
        batch_plot_helper: PhoOptimizedMultiEpochBatchRenderer = PhoOptimizedMultiEpochBatchRenderer.init_from_results2D(results2D, active_ax=track_ax, frame_divide_bin_size=frame_divide_bin_size, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)

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
    decoder: BasePositionDecoder = field()

    pos_df: pd.DataFrame = field()

    # active_ax: Any = field()
    frame_divide_bin_size: float = field()
    # rotate_to_vertical: bool = field(default=True)
    
    # desired_epoch_start_idx: int = field(default=0)
    # desired_epoch_end_idx: Optional[int] = field(default=None)

    # stacked_flat_global_pos_df: pd.DataFrame = field(default=None, init=False)
    frame_divided_epochs_result: DecodedFilterEpochsResult = field(default=None)

    has_data_been_built: bool = field(default=False)
    # active_epoch_name: str = field(default='global')
    

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
        return self.frame_divided_epochs_result


    @property
    def a_new_global2D_decoder(self) -> BasePositionDecoder:
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
    

    @classmethod
    def init_from_results2D(cls, results2D: "DecodingResultND", active_epoch_name: str = "global", **kwargs) -> "PhoOptimizedMultiEpochBatchRenderer":
        key = DecoderName(active_epoch_name)
        return cls(frame_divided_epochs_result=results2D.frame_divided_epochs_results[key], decoder=results2D.decoders[key], pos_df=results2D.pos_df, **kwargs)

    @classmethod
    def init_from_modern(cls, a_decoder, pos_df, subdivide_bin_size: float = 5.0, split_column_name: str = 'global_frame_division_idx',
                                        x_padding_pct: float = 0.05, y_padding_pct: float = 0.05,
                                ):

        pos_df, subdivided_epochs_df, maze_bounds_t, pos_tspace_df, (xt, yt) = cls.build_transforms_for_frames(a_decoder=a_decoder, pos_df=pos_df, subdivide_bin_size=subdivide_bin_size, split_column_name=split_column_name)
        # ==================================================================================================================================================================================================================================================================================== #
        # MARK: Compute for subdivided_epochs_df:                                                                                                                                                                                                                                                              #
        # ==================================================================================================================================================================================================================================================================================== #
        ## INPUTS: subdivided_epochs_df
        ## Decode: laps_df
        decoding_time_bin_size: float = 0.050 # 50ms
        # decoding_time_bin_size: float = 0.250 # 250ms
        # decoding_time_bin_size: float = 0.075 # 75ms
        a_decoded_subdivided_epochs_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(spikes_df=a_decoder.spikes_df, filter_epochs=subdivided_epochs_df, decoding_time_bin_size=decoding_time_bin_size)

        obj = cls(frame_divided_epochs_result=a_decoded_subdivided_epochs_result, decoder=decoder, pos_df=pos_df, active_ax=None, frame_divide_bin_size=subdivide_bin_size)


        return obj


    @function_attributes(short_name=None, tags=['MAIN', 'pure', 'static'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-28 22:01', related_items=[])
    @classmethod
    def build_transforms_for_frames(cls, a_decoder, pos_df, subdivide_bin_size: float = 5.0, split_column_name: str = 'global_frame_division_idx',
                                        x_padding_pct: float = 0.05, y_padding_pct: float = 0.05,
                                    ):
        """ 

        subdivided_epochs_df, maze_bounds_t, x, y

        subdivide_bin_size: float = 5.0  # seconds

        Usage:

            from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
            from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PhoOptimizedMultiEpochBatchRenderer import PhoOptimizedMultiEpochBatchRenderer

            an_epoch_name: str = 'roam'
            a_decoder = masked_container.pf1D_Decoder_dict[an_epoch_name]
            a_result = masked_container.epochs_decoded_result_cache_dict[0.025][an_epoch_name]

            subdivide_bin_size: float = 5.0  # seconds
            split_column_name: str = 'global_frame_division_idx'
            pos_df, subdivided_epochs_df, maze_bounds_t, pos_tspace_df, (xt, yt) = PhoOptimizedMultiEpochBatchRenderer.build_transforms_for_frames(a_decoder=a_decoder, pos_df=pos_df, subdivide_bin_size=subdivide_bin_size, split_column_name=split_column_name)
            subdivided_epochs_df
            pos_tspace_df


        """
        pos_df, subdivided_time_windows, subdivided_epochs_df = pos_df.time_point_event.adding_fixed_length_chunk_columns(subdivide_bin_size=subdivide_bin_size,
                                                                            split_column_name=split_column_name, interval_start_t_col_name='frame_division_epoch_start_t', interval_stop_t_col_name='frame_division_epoch_stop_t',
                                                                        )

        ## INPUTS: a_decoder, subdivided_epochs_df, pos_df
        xmin: float = a_decoder.xbin[-1]
        xmax: float = a_decoder.xbin[0]
        ymin: float = a_decoder.ybin[-1]
        ymax: float = a_decoder.ybin[0]

        ## Add the multiplicitive padding factor:
        x_width: float = np.abs(xmax - xmin)
        y_height: float = np.abs(ymax - ymin)

        x_padding: float = x_width * x_padding_pct
        y_padding: float = y_height * y_padding_pct

        x_width_full: float = x_width + x_padding
        y_height_full: float = y_height + y_padding

        maze_bounds_xy = ((x_padding/2.0), (y_padding/2.0), x_width, y_height) # (x0, y0, width_x, height_y)

        ## build offsets

        t_axis_single_window_span_duration: float = subdivide_bin_size ## how wide (in 1D) the window is. Default to subdivide_bin_size
        yt_axis_y_span: float = 1.0 ## span from [0.0, 1.0] to make it easy 

        ## get grid_bin_bounds mapping
        # subdivided_epochs_df: pd.DataFrame = deepcopy(subdivided_epochs_df)
        tmin: float = np.nanmin(subdivided_epochs_df['start'].to_numpy())
        subdivided_epochs_df['start_t_rel'] = subdivided_epochs_df['start'] - tmin
        subdivided_epochs_df['stop_t_rel'] = subdivided_epochs_df['stop'] - tmin
        subdivided_epochs_df

        ## OUTPUTS: subdivided_epochs_df

        x_width # 198.334580944969
        y_height # 195.06978042055803

        (xmin, xmax)
        (ymin, ymax)

        ## Compute transform scaling factors:
        x_to_t_scale: float = (t_axis_single_window_span_duration / x_width_full) # d_xt
        y_to_yt_scale: float = yt_axis_y_span / y_height_full

        # ==================================================================================================================================================================================================================================================================================== #
        # Compute maze_bounds shape in xt/yt coords                                                                                                                                                                                                                                            #
        # ==================================================================================================================================================================================================================================================================================== #
        ## INPUTS: maze_bounds_xy # (x0, y0, width_x, height_y)
        maze_bounds_t = np.array(maze_bounds_xy) #(x_width
        maze_bounds_t[0] = (maze_bounds_t[0] * x_to_t_scale)
        maze_bounds_t[1] = (maze_bounds_t[1] * y_to_yt_scale)
        maze_bounds_t[2] = (maze_bounds_t[2] * x_to_t_scale)
        maze_bounds_t[3] = (maze_bounds_t[3] * y_to_yt_scale)
        ## OUTPUTS: maze_bounds_t # array([0.119048, 0.0238095, 4.7619, 0.952381]) - # (x0, y0, width_x, height_y)

        # Performed 1 aggregation grouped on column: 'global_frame_division_idx'
        # pos_df = pos_df.groupby(['global_frame_division_idx']).agg(t_count=('t', 'count')).reset_index()
        global_frame_split_row_indicies = np.cumsum(pos_df[split_column_name].value_counts().to_numpy()).astype(int) # [600, 1200, 1800, ...] - the indicies at which to insert np.nan rows

        # # flip x/y position before transforming so the line segments are plotted correctly ___________________________________________________________________________________________________________________________________________________________________________________________________ #
        # def subfn_swap_variables(x, y):
        #     ## swap x/y so thee plotted line is correct with the heatmap
        #     # xt_copy = xt.copy()
        #     # xt = yt.copy()
        #     # yt = xt_copy.copy()
        #     # return xt.copy(), yt.copy()
        #     return y.copy(), x.copy()

        # xmin, ymin = subfn_swap_variables(xmin, ymin) ## swap xmin/ymin
        # x_to_t_scale, y_to_yt_scale = subfn_swap_variables(x_to_t_scale, y_to_yt_scale) ## swap xmin/ymin

        ## flip x/y position before transforming:
        pos_df['_x_copy'] = pos_df['x'].copy()
        pos_df['x'] = pos_df['y'].copy()
        pos_df['y'] = pos_df['_x_copy'].copy()
        pos_df = pos_df.drop(columns=['_x_copy'], inplace=False)

        ## convert to relative position components (offsets)
        pos_df['xt'] = (np.abs(pos_df['x'] - xmin) * x_to_t_scale)
        pos_df['yt'] = (np.abs(pos_df['y'] - ymin) * y_to_yt_scale)

        pos_df['xt'] = pos_df['xt'] + pos_df['frame_division_epoch_start_t'] + tmin ## offset by the origin of each frame start, and then by the global tmin
        # pos_df['yt'] = pos_df['yt']

        ## Adds columns: ['xt', 'yt'] which will be plot

        # ==================================================================================================================================================================================================================================================================================== #
        # insert NaNs between rows                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        ## insert NaN rows into pos_df at split indices (same positions as xt/yt for consistent length):
        pos_space_col_names: List[str] = ['t', 'x', 'y', 'xt', 'yt', split_column_name, 'frame_division_epoch_start_t', 'frame_division_epoch_stop_t']
        # pos_tspace_df = pos_df[pos_space_col_names].values
        vals = pos_df[pos_space_col_names].values
        # Insert the NaNs into the underlying array __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # (Note: axis=0 for rows)
        inserted_vals = np.insert(vals.astype(float), global_frame_split_row_indicies, np.nan, axis=0) ## converts all to float
        # Reconstruct the DataFrame
        pos_tspace_df: pd.DataFrame = pd.DataFrame(inserted_vals, columns=pos_space_col_names)
        xt = pos_tspace_df['xt'].to_numpy()
        yt = pos_tspace_df['yt'].to_numpy()

        # xt = pos_df['xt'].to_numpy()
        # yt = pos_df['yt'].to_numpy()

        # ## insert the np.nan values to split the lines at each frame index split row:
        # xt = np.insert(xt, global_frame_split_row_indicies, np.nan)
        # yt = np.insert(yt, global_frame_split_row_indicies, np.nan)     

        
        ## OUTPUTS: subdivided_epochs_df, maze_bounds_t, x, y
        return pos_df, subdivided_epochs_df, maze_bounds_t, pos_tspace_df, (xt, yt)


    @function_attributes(short_name=None, tags=['helper', 'pyqtgraph', 'display', 'renderer', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=['cls.plot_all'], creation_date='2026-03-01 07:13', related_items=[])
    @classmethod
    def plot_decoded_posteriors_for_frames(cls, a_decoded_subdivided_epochs_result, subdivided_epochs_df, maze_bounds_t, 
            track_plot_item: Optional[pg.PlotItem]=None, extant_posterior_image_items=None, **kwargs,
        ):
        """ 
        vaguely based off of `pyqtplot_plot_image_array`

        Note: posterior_img_cmap (and any EditableColormap2DEditorWidget) applies only when
        use_advanced_3D_cmap=False. When use_advanced_3D_cmap=True (default), posteriors are
        precomputed RGBA and ImageItem colormap has no effect.

        Usage:

            _out_dict = PhoOptimizedMultiEpochBatchRenderer.plot_decoded_posteriors_for_frames(a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result,
                                                                        subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                                                                        pos_tspace_df=pos_tspace_df, #xt=xt, yt=yt,
                                                                        extant_posterior_image_items=_out_dict.get('extant_posterior_image_items', None), track_plot_item=track_plot_item,
                                                                    )

        """
        drop_below_threshold = kwargs.pop('drop_below_threshold', 0.0025)
        # shared_axis_order = 'col-major' ## Was working but didn't match pos traj
        shared_axis_order = 'row-major'
        posterior_img_opacity: float = kwargs.pop('posterior_img_opacity', 0.8)
        # posterior_img_composition_mode = kwargs.pop('posterior_img_composition_mode', QtGui.QPainter.CompositionMode_Plus)
        posterior_img_composition_mode = kwargs.pop('posterior_img_composition_mode', QtGui.QPainter.CompositionMode_SourceOver)
        posterior_img_cmap = kwargs.pop('posterior_img_cmap', pg.colormap.get('viridis','matplotlib'))
        use_advanced_3D_cmap: bool = kwargs.pop('use_advanced_3D_cmap', True)
        custom_cmap1 = kwargs.pop('custom_cmap1', None)
        custom_cmap2 = kwargs.pop('custom_cmap2', None)

        global_max_v: float = np.nanmax([np.nanmax(v) for v in a_decoded_subdivided_epochs_result.p_x_given_n_list]) ## across all possible time bins
        print(f'global_max_v: {global_max_v}')

        # QtGui.QPainter.CompositionMode_SourceOver   # default alpha blending
        # QtGui.QPainter.CompositionMode_Plus         # additive (great for heatmaps)
        # QtGui.QPainter.CompositionMode_Multiply     # darkens overlap
        # QtGui.QPainter.CompositionMode_Screen       # lightens overlap
        # QtGui.QPainter.CompositionMode_Overlay      # contrast-based blend
        if use_advanced_3D_cmap:
            if custom_cmap1 is None or custom_cmap2 is None:
                # --- Define Custom Alpha-Only Colormaps ---
                # Positions range from 0.0 to 1.0 (representing the v_idx mapping)
                pos = np.array([0.0, 1.0])
                # pos = np.array([0.5, global_max_v])
                # min_cmap_occupancy: int = 0
                min_cmap_occupancy: int = 100
                max_cmap_occupancy: int = 255
                # Custom "Alpha Red": R=255, G=0, B=0, Alpha mapping from 0 to 255
                colors_red = np.array([[255, 0, 0, min_cmap_occupancy], [255, 0, 0, max_cmap_occupancy]], dtype=np.ubyte)
                custom_cmap1 = pg.ColorMap(pos, colors_red)
                # Custom "Alpha Green": R=0, G=255, B=0, Alpha mapping from 0 to 255
                colors_green = np.array([[0, 255, 0, min_cmap_occupancy], [0, 255, 0, max_cmap_occupancy]], dtype=np.ubyte)
                custom_cmap2 = pg.ColorMap(pos, colors_green)


        have_existing_img_items: bool = False
        
        # from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent, pyqtplot_common_setup
        _out_dict = {}

        # Interpret image data as row-major instead of col-major
        # pg.setConfigOptions(imageAxisOrder='row-major')
        num_decoded_epochs: int = a_decoded_subdivided_epochs_result.num_filter_epochs

        if extant_posterior_image_items is not None:
            assert len(extant_posterior_image_items) == num_decoded_epochs, f"len(extant_posterior_image_items): {len(extant_posterior_image_items)}, num_decoded_epochs: {num_decoded_epochs}"
            # _out_dict['posterior_image_items'] = extant_posterior_image_items
            have_existing_img_items = True
        else:
            extant_posterior_image_items = []
            # _out_dict['posterior_image_items'] = extant_posterior_image_items

        ## restrict to subrange
        # ==================================================================================================================== #
        # Plot the posterior heatmap                                                                                           #
        # ==================================================================================================================== #
        # custom_image_extent = [0.0, 1.0, 0.0, 1.0]
        maze_bounds_t_arr = np.tile(maze_bounds_t, (num_decoded_epochs, 1)) # np.repmat(maze_bounds_t, shape=(num_decoded_epochs,))
        rect_xt_positions = maze_bounds_t[0] + subdivided_epochs_df['start'].to_numpy()
        maze_bounds_t_arr[:, 0] = rect_xt_positions
        # maze_bounds_t_arr[:, 1] = maze_bounds_t_arr[:, 0] + maze_bounds_t_arr[:, 2]

        for epoch_idx in np.arange(num_decoded_epochs):
            epoch_n_t_bins: int = a_decoded_subdivided_epochs_result.nbins[epoch_idx]
            img = a_decoded_subdivided_epochs_result.p_x_given_n_list[epoch_idx] # (n_x_bins, n_y_bins, epoch_n_t_bins)
            if drop_below_threshold is not None:
                # Create a masked array, masking values below the threshold
                img = np.ma.masked_less(img, drop_below_threshold)

            image_bounds_extent = np.squeeze(maze_bounds_t_arr[epoch_idx, :]) ## single bounds

            if not use_advanced_3D_cmap:
                ## need to collapse it over all epoch time bins:
                img = np.nansum(img, axis=-1) / float(epoch_n_t_bins) ## (n_x_bins, n_y_bins)
            else:
                # 1. Create the specialized 3D LUT
                # lut_3d = create_3d_lut_saturation(n_t_bins=epoch_n_t_bins, cmap_name='magma')
                lut_3d = create_3d_lut_cmaps_interp(n_t_bins=epoch_n_t_bins, cmap1_name=custom_cmap1, cmap2_name=custom_cmap2)

                # 2. Apply color mapping instantly using indexing
                # print("Applying advanced color mapping...")
                rgba_volume = apply_3d_colormap(img, lut_3d)
                # 3. Composite into a single flat image
                # print("Compositing stack...")
                img = composite_stack(rgba_volume)


            if not have_existing_img_items:
                ## create the new img_item:
                img_item = pg.ImageItem(img)
                if posterior_img_opacity is not None:
                    img_item.setOpacity(posterior_img_opacity)  # Set transparency for overlay
                if posterior_img_composition_mode is not None:
                    img_item.setCompositionMode(posterior_img_composition_mode)

                if use_advanced_3D_cmap:
                    pass
                else:
                    if posterior_img_cmap is not None:
                        # Set the color map:
                        img_item.setColorMap(posterior_img_cmap)

            else:
                img_item = extant_posterior_image_items[epoch_idx]

            img_item.setImage(img, rect=image_bounds_extent, autoLevels=False, axisOrder=shared_axis_order) # rect: [x, y, w, h] # , axisOrder='row-major'
            # img_item.setZValue(1000)

            if not have_existing_img_items:
                track_plot_item.addItem(img_item, defaultPadding=0.0)
                extant_posterior_image_items.append(img_item) # ['image_item']

        ## END for epoch_idx in np.arange(num_decoded_bins)...
        _out_dict['posterior_image_items'] = extant_posterior_image_items
        return _out_dict


    # def pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    #     """ Plots an array of images provided in 'images' argument
    #     images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
    #         or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
            
    #     NOTES:
    #         2022-09-29 - Extracted from Notebook
    #             🚧 Needs subplot labels changed from Cell[i] to the appropriate standardized titles. Needs other minor refinements.
    #             🚧 pyqtplot_plot_image_array needs major improvements to achieve feature pairity with display_all_pf_2D_pyqtgraph_binned_image_rendering, so probably just use display_all_pf_2D_pyqtgraph_binned_image_rendering.
            
    #     Example:
    #         # Get flat list of images:
    #         images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
    #         # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
    #         occupancy = active_one_step_decoder.ratemap.occupancy

    #         app, win, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy)
    #         win.show()
            
            
    #     # 🚧 TODO: COMPATIBILITY: replace compute_paginated_grid_config with standardized `_determine_best_placefield_2D_layout` block (see below):
    #     ```
    #     from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout
    #     nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons),
    #         **overriding_dict_with(lhs_dict={'subplots': (40, 3), 'fig_column_width': 8.0, 'fig_row_height': 1.0, 'resolution_multiplier': 1.0, 'max_screen_figure_size': (None, None), 'last_figure_subplots_same_layout': True, 'debug_print': True}, **figure_format_config)) 

    #     print(f'nfigures: {nfigures}\ndata_aspect_ratio: {data_aspect_ratio}')
    #     # Loop through each page/figure that's required:
    #     for page_fig_ind, page_fig_size, page_grid_size in zip(np.arange(nfigures), page_figure_sizes, page_grid_sizes):
    #         print(f'\tpage_fig_ind: {page_fig_ind}, page_fig_size: {page_fig_size}, page_grid_size: {page_grid_size}')
    #         # print(f'\tincluded_combined_indicies_pages: {included_combined_indicies_pages}\npage_grid_sizes: {page_grid_sizes}\npage_figure_sizes: {page_figure_sizes}')
    #     ```
            
            
            
    #     """
    #     root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'pyqtplot_plot_image_array: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget) ## 🚧 TODO: BUG: this makes a new QMainWindow to hold this item, which is inappropriate if it's to be rendered as a child of another control
        
    #     pg.setConfigOptions(imageAxisOrder='col-major') # this causes the placefields to be rendered horizontally, like they were in _temp_pyqtplot_plot_image_array
        
    #     # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    #     cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map

    #     image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    #     # image_aspect_ratio, image_width_height_tuple = compute_data_aspect_ratio(x_range, y_range)
    #     # print(f'image_aspect_ratio: {image_aspect_ratio} - xScale/yScale: {float(image_width_height_tuple.width) / float(image_width_height_tuple.height)}')
        
    #     # Compute Images:
    #     included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
    #     nMapsToShow = len(included_unit_indicies)

    #     # Paging Management: Constrain the subplots values to just those that you need
    #     subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
    #     page_idx = 0 # page_idx is zero here because we only have one page:
        
    #     img_item_array = []
    #     other_components_array = []
    #     plot_array = []

    #     for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
    #         # Need to convert to page specific:
    #         curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
    #         curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
    #         curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
    #         is_first_column = (curr_page_relative_col == 0)
    #         is_first_row = (curr_page_relative_row == 0)
    #         is_last_column = (curr_page_relative_col == (page_grid_sizes[page_idx].num_columns-1))
    #         is_last_row = (curr_page_relative_row == (page_grid_sizes[page_idx].num_rows-1))
    #         if debug_print:
    #             print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

    #         neuron_IDX = curr_included_unit_index
    #         curr_cell_identifier_string = f'Cell[{neuron_IDX}]'
    #         curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'

    #         # # Pre-filter the data:
    #         image = _scale_current_placefield_to_acceptable_range(np.squeeze(images[a_linear_index,:,:]), occupancy=occupancy, drop_below_threshold=drop_below_threshold)

    #         # Build the image item:
    #         img_item = pg.ImageItem(image=image, levels=(0,1))
    #         #     # Viewbox version:
    #         #     # vb = layout.addViewBox(lockAspect=False)
    #         #     # # Build the ImageItem (which I'm guessing is like pg.ImageView) to add the image
    #         #     # imv = pg.ImageItem() # Create it with the current image
    #         #     # vb.addItem(imv) # add the item to the view box: why do we need the wrapping view box?
    #         #     # vb.autoRange()
            
    #         # # plot mode:
    #         curr_plot = root_render_widget.addPlot(row=curr_row, col=curr_col, title=curr_cell_identifier_string) # , name=curr_plot_identifier_string 
    #         curr_plot.setObjectName(curr_plot_identifier_string)
    #         curr_plot.showAxes(False)
    #         if is_last_row:
    #             curr_plot.showAxes('x', True)
    #             curr_plot.showAxis('bottom', show=True)
    #         else:
    #             curr_plot.showAxes('x', False)
    #             curr_plot.showAxis('bottom', show=False)
                
    #         if is_first_column:
    #             curr_plot.showAxes('y', True)
    #             curr_plot.showAxis('left', show=True)
    #         else:
    #             curr_plot.showAxes('y', False)
    #             curr_plot.showAxis('left', show=False)
            
    #         curr_plot.hideButtons() # Hides the auto-scale button
            
    #         curr_plot.addItem(img_item, defaultPadding=0.0)  # add ImageItem to PlotItem
    #         # curr_plot.setAspectLocked(lock=True, ratio=image_aspect_ratio)
    #         # curr_plot.showAxes(True)
    #         # curr_plot.showGrid(True, True, 0.7)
    #         # curr_plot.setLabel('bottom', "Label to test offset")
            
    #         # # Overlay cell identifier text:
    #         # curr_label = pg.TextItem(f'Cell[{neuron_IDX}]', color=(230, 230, 230))
    #         # curr_label.setPos(30, 60)
    #         # curr_label.setParentItem(img_item)
    #         # # curr_plot.addItem(curr_label, ignoreBounds=True)
    #         # curr_plot.addItem(curr_label)

    #         # Update the image:
    #         img_item.setImage(image, rect=image_bounds_extent, autoLevels=False) # rect: [x, y, w, h]
    #         img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)

    #         # curr_plot.set
    #         # margin = 2.0
    #         # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
    #         # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
    #         # curr_plot.setXRange(*x_range)
    #         # curr_plot.setYRange(*y_range)
    #         curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
    #         # Sets only the panning limits:
    #         curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
    #         # Link Axes to previous item:
    #         if a_linear_index > 0:
    #             prev_plot_item = plot_array[a_linear_index-1]
    #             curr_plot.setXLink(prev_plot_item)
    #             curr_plot.setYLink(prev_plot_item)
                
                
    #         # Interactive Color Bar:
    #         bar = pg.ColorBarItem(values= (0, 1), colorMap=cmap, width=5, interactive=False) # prepare interactive color bar
    #         # Have ColorBarItem control colors of img and appear in 'plot':
    #         bar.setImageItem(img_item, insert_in=curr_plot)

    #         img_item_array.append(img_item)
    #         plot_array.append(curr_plot)
    #         other_components_array.append({'color_bar':bar}) # note this is a list of Dicts, one for every image
            
    #     # Post images loop:
    #     enable_show = False
        
    #     if parent_root_widget is not None:
    #         if enable_show:
    #             parent_root_widget.show()
            
    #         parent_root_widget.setWindowTitle('pyqtplot image array')

    #     # pg.exec()
    #     return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array



    @function_attributes(short_name=None, tags=['helper', 'renderer', 'maze'], input_requires=[], output_provides=[], uses=[], used_by=['cls.plot_all'], creation_date='2026-02-29 00:01', related_items=[])
    @classmethod
    def plot_all_track_shapes(cls, subdivided_epochs_df, maze_bounds_t, track_plot_item: Optional[pg.PlotItem]=None):
        """ plots all track shapes

        track_shape_rects_item, maze_boundaries_path = plot_all_track_shapes(offsets_epochs_df=offsets_epochs_df, maze_bounds_t=maze_bounds_t, track_plot_item=track_plot_item)

        """
        def build_rect_paths(x_positions, width, height, y0=0.0) -> QtGui.QPainterPath:
            """ efficiently builds all maze rectangles 
            """
            path = QtGui.QPainterPath()
            
            for x in x_positions:
                path.addRect(x, y0, width, height)

            return path

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        # Example
        ## INPUTS: offsets_epochs_df, maze_bounds_t # (x0, y0, width_x, height_y)
        if track_plot_item is None:
            track_plot_item = pg.plot()

        rect_xt_positions = maze_bounds_t[0] + subdivided_epochs_df['start'].to_numpy() ## offset by the origin of each frame start, and then by the global tmin
        num_rects: int = len(rect_xt_positions)
        print(f'len(rect_xt_positions): {len(rect_xt_positions)}')
        # max_num_rects: int = 1500
        max_num_rects: int = 6000

        assert num_rects < max_num_rects, f'num_rects: {num_rects} should be less than {max_num_rects} as not to lock up the viewer...'
        maze_boundaries_path: QtGui.QPainterPath = build_rect_paths(rect_xt_positions, y0=maze_bounds_t[1], width=maze_bounds_t[2], height=maze_bounds_t[3])

        # plot = pg.plot()
        track_shape_rects_item: pg.QtWidgets.QGraphicsPathItem = pg.QtWidgets.QGraphicsPathItem(maze_boundaries_path)
        # item.setPen(pg.mkPen(None))      # no border (fast)
        # item.setBrush(pg.mkBrush('w'))   # filled rectangles

        track_shape_rects_item.setPen(pg.mkPen(None))      # no border (fast)
        track_shape_rects_item.setBrush(pg.mkBrush('#FFFFFF77')) # filled grey rectangles

        track_plot_item.addItem(track_shape_rects_item)

        return track_shape_rects_item, maze_boundaries_path


    @function_attributes(short_name=None, tags=['helper', 'renderer', 'maze'], input_requires=[], output_provides=[], uses=[], used_by=['cls.plot_all'], creation_date='2026-02-29 00:01', related_items=['plot_all_animal_position_gradient_segments'])
    @classmethod
    def plot_all_animal_position_segments(cls, xt, yt, track_plot_item: Optional[pg.PlotItem]=None):
        """ Plots the animal positions for each frame

            animal_position_segments_item, animal_position_segments_path = PhoOptimizedMultiEpochBatchRenderer.plot_all_animal_position_segments(x=x, y=y, track_plot_item=track_plot_item)

        """
        ## INPUTS: track_plot_item
        # ## swap x/y so thee plotted line is correct with the heatmap
        # xt_copy = xt.copy()
        # xt = yt.copy()
        # yt = xt_copy.copy()


        from pyqtgraph.functions import arrayToQPath
        animal_position_segments_path = arrayToQPath(xt, yt, connect='finite')
        animal_position_segments_item: pg.QtWidgets.QGraphicsPathItem = pg.QtWidgets.QGraphicsPathItem(animal_position_segments_path)
        animal_position_segments_item.setPen(pg.mkPen('#FFFFFFBB'))
        animal_position_segments_item.setZValue(9) ## in foreground

        if track_plot_item is None:
            track_plot_item = pg.plot()
        track_plot_item.addItem(animal_position_segments_item)

        return animal_position_segments_item, animal_position_segments_path

    @function_attributes(short_name=None, tags=['helper', 'openGL', 'camera'], input_requires=[], output_provides=[], uses=[], used_by=['plot_all_animal_position_gradient_segments'], creation_date='2026-03-01', related_items=[])
    @classmethod
    def gl_zoom_to_points(cls, gl_view, xt, yt, padding=1.1, min_distance=1.0):
        """Frame the GL view on the given XY points (z assumed 0). Set opts['center'] before setCameraPosition so the camera actually looks at the data."""
        xt = np.asarray(xt)
        yt = np.asarray(yt)
        xmin, xmax = xt.min(), xt.max()
        ymin, ymax = yt.min(), yt.max()
        x_range = xmax - xmin
        y_range = ymax - ymin
        max_range = max(x_range, y_range, 1e-9)
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        distance = max(max_range * padding, min_distance)
        gl_view.opts['center'] = pg.Vector(center_x, center_y, 0)
        gl_view.setCameraPosition(pos=None, distance=distance, elevation=90, azimuth=0)
        gl_view.update()

    @function_attributes(short_name=None, tags=['helper', 'ALT', 'position', 'gradiant', 'openGL'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-03-01 09:50', related_items=['plot_all_animal_position_segments'])
    @classmethod
    def plot_all_animal_position_gradient_segments(cls, t, xt, yt, track_plot_item: Optional[pg.opengl.GLViewWidget]=None):
        """ 
        Plots the animal positions using a per-vertex timestamp colormap
        rendered with GLLinePlotItem.

        animal_position_segments_item = PhoOptimizedMultiEpochBatchRenderer.plot_all_animal_position_gradient_segments(
            t=t, xt=xt, yt=yt, # track_plot_item=track_plot_item
        )
        """
        # import pyqtgraph.opengl as gl
        import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl

        # Ensure numpy arrays
        xt = np.asarray(xt)
        yt = np.asarray(yt)

        assert xt.shape == yt.shape, "xt and yt must have same shape"

        n_points = xt.shape[0]
        if n_points < 2:
            raise ValueError("Need at least 2 points to draw a line.")

        print(f'n_points: {n_points}')
        # ------------------------------------------------------------------
        # Timestamp normalization (using index as time proxy)
        # ------------------------------------------------------------------
        if t is None:
            t = np.arange(n_points, dtype=float)

        assert len(t) == len(xt), f"len(t): {len(t)}, len(xt): {len(xt)}"

        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)

        # ------------------------------------------------------------------
        # Colormap (viridis default, change if desired)
        # ------------------------------------------------------------------
        # cmap = pg.colormap.get('viridis')
        # colors = cmap.map(t_norm, mode='float')  # (N,4) RGBA float32

        colors = (1.0,1.0,1.0,1.0) # pg.mkColor('w')
        # ------------------------------------------------------------------
        # Build 3D positions (z=0)
        # ------------------------------------------------------------------
        pos = np.column_stack((xt, yt, np.zeros_like(xt))) # (N,3) array of floats specifying point locations
        print(f'np.shape(pos): {np.shape(pos)}')

        animal_position_segments_item = gl.GLLinePlotItem(
            pos=pos,
            color=colors,
            width=2.0,
            mode='line_strip',
            antialias=True
        )

        # Foreground ordering (OpenGL depth-based, but still useful)
        # animal_position_segments_item.setGLOptions('opaque')

        # ------------------------------------------------------------------
        # Create GL view if needed
        # ------------------------------------------------------------------
        if track_plot_item is None:
            track_plot_item = gl.GLViewWidget()
            track_plot_item.show()

        track_plot_item.addItem(animal_position_segments_item)

        cls.gl_zoom_to_points(track_plot_item, xt, yt)

        return animal_position_segments_item


    @function_attributes(short_name=None, tags=['MAIN', 'figure', 'plot', 'render', 'pyqtgraph'], input_requires=[], output_provides=[], uses=['cls.plot_all_track_shapes', 'cls.plot_decoded_posteriors_for_frames', 'cls.plot_all_animal_position_segments'], used_by=[], creation_date='2026-03-01 07:14', related_items=[])
    @classmethod
    def plot_all(cls, subdivided_epochs_df, maze_bounds_t, pos_tspace_df: pd.DataFrame, a_decoded_subdivided_epochs_result=None, track_plot_item: Optional[pg.PlotItem]=None, **kwargs):
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
        from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PhoOptimizedMultiEpochBatchRenderer import PhoOptimizedMultiEpochBatchRenderer

        an_epoch_name: str = 'roam'
        a_decoder = masked_container.pf1D_Decoder_dict[an_epoch_name]
        a_result = masked_container.epochs_decoded_result_cache_dict[0.025][an_epoch_name]

        subdivide_bin_size: float = 5.0  # seconds
        split_column_name: str = 'global_frame_division_idx'
        pos_df, subdivided_epochs_df, maze_bounds_t, (xt, yt) = PhoOptimizedMultiEpochBatchRenderer.build_transforms_for_frames(a_decoder=a_decoder, pos_df=pos_df, subdivide_bin_size=subdivide_bin_size, split_column_name=split_column_name)
        subdivided_epochs_df


        track_name: str = 'CustomBatch2Dto1DTimeline'
        spike_raster_plt_2d: Spike2DRaster = spike_raster_window.spike_raster_plt_2d
        # track_ts_widget, track_fig, track_ax_list, dDisplayItem = spike_raster_plt_2d.add_new_matplotlib_render_plot_widget(name=track_name)
        a_time_sync_pyqtgraph_widget, track_root_graphics_layout_widget, track_plot_item, dDisplayItem = spike_raster_plt_2d.add_new_embedded_pyqtgraph_render_plot_widget(name=track_name, dockSize=(500,50), sync_mode=SynchronizedPlotMode.TO_WINDOW)
        track_plot_item

        ## INPUTS: pos_df, subdivided_epochs_df, maze_bounds_t, (xt, yt)
        ## Pass colormap_editor_container so the posterior colormap editor is added to the track dock and visible.
        _out_dict = PhoOptimizedMultiEpochBatchRenderer.plot_all(subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                                                                    pos_tspace_df=pos_tspace_df,# xt=xt, yt=yt, 
                                                                    a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result, track_plot_item=track_plot_item,
                                                                    colormap_editor_container=a_time_sync_pyqtgraph_widget,
                                                                )
        ## Editor is created by default (create_colormap_editor=True) and added when colormap_editor_container is passed.
        ## Only affects 2D mode (use_advanced_3D_cmap=False); when True, posteriors are RGBA and the editor has no effect.


        """
        _out_dict = {}

        # pos_tspace_df: pd.DataFrame = pd.DataFrame(inserted_vals, columns=pos_space_col_names)
        xt = pos_tspace_df['xt'].to_numpy()
        yt = pos_tspace_df['yt'].to_numpy()


        # ## swap x/y so thee plotted line is correct with the heatmap
        # xt = pos_tspace_df['yt'].to_numpy()
        # yt = pos_tspace_df['xt'].to_numpy()

        ## INPUTS: track_plot_item
        track_shape_rects_item, maze_boundaries_path = cls.plot_all_track_shapes(subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t, track_plot_item=track_plot_item)
        _out_dict.update(track_shape_rects_item=track_shape_rects_item, maze_boundaries_path=maze_boundaries_path)

        if a_decoded_subdivided_epochs_result is not None:
            extant_posterior_image_items = kwargs.pop('extant_posterior_image_items', None)
            create_colormap_editor = kwargs.pop('create_colormap_editor', True)
            posterior_colormap_initial_cmap = kwargs.pop('posterior_colormap_initial_cmap', None)
            colormap_editor_container = kwargs.pop('colormap_editor_container', None)
            use_advanced_3D_cmap = kwargs.pop('use_advanced_3D_cmap', True)
            plotted_posterior_items_dict = cls.plot_decoded_posteriors_for_frames(a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result,
                                                                        subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                                                                        extant_posterior_image_items=extant_posterior_image_items, track_plot_item=track_plot_item,
                                                                        use_advanced_3D_cmap=use_advanced_3D_cmap,
                                                                    )
            if plotted_posterior_items_dict is not None:
                _out_dict['plotted_posterior_items_dict'] = plotted_posterior_items_dict
                posterior_image_items = plotted_posterior_items_dict.get('posterior_image_items')
                if create_colormap_editor and posterior_image_items:
                    if use_advanced_3D_cmap:
                        editor = EditableColormap2DEditorWidget()
                        _out_dict['posterior_colormap_editor'] = editor

                        def _reapply_advanced_colormap():
                            cls.plot_decoded_posteriors_for_frames(a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result,
                                    subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                                    extant_posterior_image_items=posterior_image_items, track_plot_item=track_plot_item,
                                    use_advanced_3D_cmap=True, custom_cmap1=editor.getCmap1(), custom_cmap2=editor.getCmap2())

                        editor.sigAdvancedColormapChanged.connect(_reapply_advanced_colormap)
                    else:
                        #TODO 2026-03-02 19:30: - [ ] WARN: Untested, this is just a weird 1D editor
                        editor = Colormap1DEditorWidget(image_items=posterior_image_items, initial_cmap=posterior_colormap_initial_cmap)
                        _out_dict['posterior_colormap_editor'] = editor
                    if colormap_editor_container is not None:
                        if isinstance(colormap_editor_container, QtWidgets.QLayout):
                            colormap_editor_container.addWidget(editor)
                        elif isinstance(colormap_editor_container, QtWidgets.QWidget):
                            layout = colormap_editor_container.layout()
                            if layout is None:
                                layout = QtWidgets.QVBoxLayout(colormap_editor_container)
                                colormap_editor_container.setLayout(layout)
                            layout.addWidget(editor)


        animal_position_segments_item, animal_position_segments_path = cls.plot_all_animal_position_segments(xt=xt, yt=yt, track_plot_item=track_plot_item)
        _out_dict.update(animal_position_segments_item=animal_position_segments_item, animal_position_segments_path=animal_position_segments_path)
        return _out_dict

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

        self.stacked_flat_global_pos_df = PhoOptimizedMultiEpochBatchRenderer.add_color_over_global_frame_division_idx_positions_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=self.stacked_flat_global_pos_df, time_cmap=time_cmap)
        
        # ensure the 'y_scaled' actually are scaled between [0.0, 1.0]
        self.stacked_flat_global_pos_df["y_scaled"] = (self.stacked_flat_global_pos_df["y_scaled"] - self.stacked_flat_global_pos_df["y_scaled"].min()) / (self.stacked_flat_global_pos_df["y_scaled"].max() - self.stacked_flat_global_pos_df["y_scaled"].min())
        
        # stacked_flat_global_pos_df
        new_stacked_flat_global_pos_df = PhoOptimizedMultiEpochBatchRenderer.add_nan_masked_rows_to_stacked_flat_global_pos_df(stacked_flat_global_pos_df=self.stacked_flat_global_pos_df)
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
        frame_division_epoch_separator_vlines = active_ax.vlines(self.frame_divided_epochs_result.filter_epochs['start'].to_numpy(), **y_axis_kwargs, colors='white', linestyles='solid', label='frame_division_epoch_separator_vlines') # , data=None

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
        # long_rect_arr = PhoOptimizedMultiEpochBatchRenderer.rect_tuples_to_NDArray(long_rects, x_offset=x_offset)
        # short_rect_arr = PhoOptimizedMultiEpochBatchRenderer.rect_tuples_to_NDArray(short_rects, x_offset=x_offset)


        # num_horizontal_repeats: int = 20 ## hardcoded
        self.track_all_normalized_rect_arr_dict = PhoOptimizedMultiEpochBatchRenderer.track_dict_all_stacked_rect_arr_normalization(self.track_single_rects_dict, num_horizontal_repeats=num_horizontal_repeats)
        ## INPUTS: filtered_num_horizontal_repeats
        # self.inverse_normalized_track_all_rect_arr_dict = PhoOptimizedMultiEpochBatchRenderer.track_dict_all_stacked_rect_arr_inverse_normalization(self.track_all_normalized_rect_arr_dict, ax=active_ax, num_active_horizontal_repeats=num_horizontal_repeats)
        self.inverse_normalized_track_all_rect_arr_dict = PhoOptimizedMultiEpochBatchRenderer.track_dict_all_stacked_rect_arr_inverse_normalization_from_custom_extent(self.track_all_normalized_rect_arr_dict, custom_image_extent=self.custom_image_extent, num_active_horizontal_repeats=num_horizontal_repeats)


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
            # self.inverse_normalized_track_all_rect_arr_dict = PhoOptimizedMultiEpochBatchRenderer.track_dict_all_stacked_rect_arr_inverse_normalization(track_all_rect_arr_dict, ax=active_ax, num_active_horizontal_repeats=filtered_num_horizontal_repeats)
            self.inverse_normalized_track_all_rect_arr_dict = PhoOptimizedMultiEpochBatchRenderer.track_dict_all_stacked_rect_arr_inverse_normalization_from_custom_extent(track_all_rect_arr_dict, custom_image_extent=self.custom_image_extent, num_active_horizontal_repeats=filtered_num_horizontal_repeats)
            ## OUTPUTS: inverse_normalized_track_all_rect_arr_dict
            

        ## INPUTS: track_kwargs_dict, inverse_normalized_track_all_rect_arr_dict
        track_shape_patch_collection_artists = PhoOptimizedMultiEpochBatchRenderer.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=self.inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict) # start (x0: 0.0, 20 of them span to exactly x=1.0)
        # track_shape_patch_collection_artists = PhoOptimizedMultiEpochBatchRenderer.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict, transform=ax.transData) # start (x0: 31.0, 20 of them span to about x=1000.0)
        # track_shape_patch_collection_artists = PhoOptimizedMultiEpochBatchRenderer.add_batch_track_shapes(ax=active_ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict, transform=ax.transAxes) # start (x0: 31.0, 20 of them span to about x=1000.0)
        
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
        plots_data: RenderPlotsData = RenderPlotsData(name='PhoOptimizedMultiEpochBatchRenderer', image_extent=None, curr_artist_dict=None,
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
        

    
    @function_attributes(short_name=None, tags=['main', 'new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 09:16', related_items=[])
    @classmethod
    def add_batch_track_shapes(cls, ax, inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict, transform=None):
        """ 
        
        track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}
        track_shape_patch_collection_artists = PhoOptimizedMultiEpochBatchRenderer.add_batch_track_shapes(ax=ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict)
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
