from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND

from copy import deepcopy
import param
import numpy as np
import pandas as pd
from attrs import define, field, Factory
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
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.geometry_helpers import point_tuple_mid_point, BoundsRect, is_point_in_rect

from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.EpochRenderTimebinSelectorWidget.EpochRenderTimebinSelectorWidget import EpochTimebinningIndexingDatasource # used in `DecodedTrajectoryPlotter` to conform to `EpochTimebinningIndexingDatasource` protocol

@metadata_attributes(short_name=None, tags=['2D_timeseries', '2D_posteriors', 'frames', 'UNFINISHED', 'KINDA-WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-19 00:00', related_items=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side'])
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
    results2D: "DecodingResultND" = field()

    active_ax = field()
    frame_divide_bin_size: float = field()
    rotate_to_vertical: bool = field(default=True)
    
    desired_epoch_start_idx: int = field(default=0)
    desired_epoch_end_idx: Optional[int] = field(default=None)

    stacked_flat_global_pos_df: pd.DataFrame = field(default=None, init=False)

    has_data_been_built: bool = field(default=False)

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
        return self.results2D.frame_divided_epochs_results['global']

    @property
    def a_new_global2D_decoder(self) -> BasePositionDecoder:
        return self.results2D.decoders['global']

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
    def shared_build_flat_stacked_data(self, debug_print=False, should_expand_first_dim: bool=True, force_recompute:bool=False, **kwargs):
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
            print(f'force_recompute == True, so `self.stacked_flat_global_pos_df` will be rebuilt from scratch from `self.results2D.pos_df`...')
            self.has_data_been_built = False
    
        if (self.stacked_flat_global_pos_df is None) or force_recompute:
            self.stacked_flat_global_pos_df = deepcopy(self.results2D.pos_df)

        ## slice `stacked_flat_global_pos_df` by desired start/end indicies too:
        if (self.desired_epoch_end_idx is not None):
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[np.logical_and((self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx), (self.stacked_flat_global_pos_df['global_frame_division_idx'] < self.desired_epoch_end_idx))]
        else:
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[(self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx)]


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
        curr_artist_dict['prev_heatmaps'], image_extent, plots_data = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(active_ax,
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


        plots_data.track_shape_patch_collection_artists = self.add_track_shapes(global_session=global_session, override_ax=override_ax, defer_draw=True, debug_print=debug_print) ## does not seem to successfully synchronize to window
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
    fig, axs, decoded_epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=n_axes, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True, fixed_columns=n_axes)
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
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter

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
    artist_line_dict = field(default=Factory(dict))
    artist_markers_dict = field(default=Factory(dict))
    
    plots_data_dict_array: List[List[RenderPlotsData]] = field(init=False)
    artist_dict_array: List[List[Dict]] = field(init=False)
    fig = field(default=None)
    axs: NDArray = field(default=None)
    laps_pages: List = field(default=Factory(list))
    row_column_indicies: NDArray = field(default=None)
    linear_plotter_indicies: NDArray = field(default=None)
    
    # measured_position_df: Optional[pd.DataFrame] = field(default=None)
    rotate_to_vertical: bool = field(default=False, metadata={'desc': 'if False, the track is rendered horizontally along its length, otherwise it is rendered vectically'})
    
    
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


    ## MAIN PLOT FUNCTION:
    @function_attributes(short_name=None, tags=['main', 'plot', 'posterior', 'epoch', 'line', 'trajectory'], input_requires=[], output_provides=[], uses=['self._perform_add_decoded_posterior_and_trajectory'], used_by=['plot_epoch_with_slider_widget'], creation_date='2025-01-29 15:52', related_items=[])
    def plot_epoch(self, an_epoch_idx: int, override_plot_linear_idx: Optional[int]=None, time_bin_index: Optional[int]=None, include_most_likely_pos_line: Optional[bool]=None, override_ax=None, should_post_hoc_fit_to_image_extent: bool = True, posterior_masking_value: float = 0.0025, debug_print:bool = False):
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
                                                                            include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=time_bin_index, rotate_to_vertical=self.rotate_to_vertical, should_perform_reshape=True, should_post_hoc_fit_to_image_extent=should_post_hoc_fit_to_image_extent,
                                                                            posterior_masking_value=posterior_masking_value, debug_print=debug_print) # , allow_time_slider=True


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
            self.plot_epoch(an_epoch_idx=index, time_bin_index=time_bin_index, include_most_likely_pos_line=include_most_likely_pos_line)

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

        self.plot_epoch(an_epoch_idx=an_epoch_idx, time_bin_index=None, include_most_likely_pos_line=include_most_likely_pos_line)

        display(self.epoch_slider)
        # display(self.checkbox)
        # display(self.time_bin_slider)


    # ==================================================================================================================== #
    # General Fundamental Plot Element Helpers                                                                             #
    # ==================================================================================================================== #
    
    # fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=0)
    @function_attributes(short_name=None, tags=['matplotlib', 'helper', 'gradient', 'curve', 'line'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-18 06:22', related_items=[])
    @classmethod
    def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False, time_cmap='viridis', **LineCollection_kwargs):
        """ Adds a gradient line representing a timeseries of (x, y) positions.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        
        
        cls._helper_add_gradient_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        # Create a continuous norm to map from data points to colors
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        norm = plt.Normalize(t.min(), t.max())
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if isinstance(time_cmap, str):
            time_cmap = plt.get_cmap(time_cmap)  # Choose a colormap
        lc = LineCollection(segments, cmap=time_cmap, norm=norm, **LineCollection_kwargs)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)
        lc.set_alpha(0.85)
        line = ax.add_collection(lc)

        if add_markers:
            # Builds scatterplot markers (points) along the path
            colors_arr = time_cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!
            # segments_arr = line.get_segments() # (16, 2, 2)
            # len(a_most_likely_positions) # 17
            _out_markers = ax.scatter(x=x, y=y, s=50, c=colors_arr, marker='D')
            return line, _out_markers
        else:
            return line, None

    @function_attributes(short_name=None, tags=['AI', 'posterior', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 12:00', related_items=[])
    @classmethod
    def _helper_add_heatmap(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers=None, ybin_centers=None, rotate_to_vertical:bool=False, debug_print:bool=False,
                            posterior_masking_value: float = 0.0025, full_posterior_opacity: float = 1.0,
                            custom_image_extent=None, cmap = 'viridis', should_perform_reshape: bool=True, extant_plot_data: Optional[RenderPlotsData]=None):
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
            print(f'np.shape(posterior): {np.shape(posterior)}')
        
        masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)
        is_2D: bool = (np.ndim(posterior) >= 3)
        if debug_print:
            print(f'is_2D: {is_2D}')
        
        x_values = deepcopy(xbin_centers)
        extra_dict = {'is_2D': is_2D}
        
        if not is_2D:
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
                    print(f'rotate_to_vertical: swapping axes. Original masked_posterior shape: {np.shape(masked_posterior)}')
                masked_posterior = masked_posterior.swapaxes(-2, -1) ## swap the last two (x, y) axes -- this doesn't work, because
                
            if debug_print:
                print(f'Post-swap masked_posterior shape: {np.shape(masked_posterior)}')
        else:
            ordinate_first_image_extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
        
        if custom_image_extent is not None:
            assert len(custom_image_extent) == 4
            print(f'using `custom_image_extent`: prev_image_extent: {ordinate_first_image_extent}, custom_image_extent: {custom_image_extent}')
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
            

        heatmaps = []
        # For simplicity, we assume non-single-time-bin mode (as asserted in the calling function).
        if not is_2D:
            a_heatmap = an_ax.imshow(masked_posterior, aspect='auto', cmap=cmap, alpha=full_posterior_opacity,
                                       extent=ordinate_first_image_extent, origin='lower', interpolation='none')
            heatmaps.append(a_heatmap)
        else:
            vmin_global = np.nanmin(posterior)
            vmax_global = np.nanmax(posterior)
            # Give a minimum opacity per time step.
            time_step_opacity: float = max(full_posterior_opacity/float(n_time_bins), 0.2)
            for i in np.arange(n_time_bins):
                a_heatmap = an_ax.imshow(np.squeeze(masked_posterior[i, :, :]), aspect='auto', cmap=cmap, alpha=time_step_opacity,
                                           extent=ordinate_first_image_extent, origin='lower', interpolation='none',
                                           vmin=vmin_global, vmax=vmax_global)
                heatmaps.append(a_heatmap)
        return heatmaps, ordinate_first_image_extent, plots_data


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
        a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(an_ax, t=a_measured_time_bin_centers, **pos_kwargs, add_markers=add_markers, time_cmap=time_cmap, zorder=0)
        
        return a_meas_pos_line, _meas_pos_out_markers
    

    @function_attributes(short_name=None, tags=['plot'], input_requires=[], output_provides=[], uses=['cls._helper_add_heatmap', 'cls._perform_plot_measured_position_line_helper'], used_by=['.plot_epoch'], creation_date='2025-01-29 15:53', related_items=[])
    @classmethod
    def _perform_add_decoded_posterior_and_trajectory(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, a_most_likely_positions, ybin_centers=None, a_measured_pos_df: Optional[pd.DataFrame]=None,
                                                        include_most_likely_pos_line: Optional[bool]=None, time_bin_index: Optional[int]=None, rotate_to_vertical:bool=False, debug_print=False, posterior_masking_value: float = 0.0025, should_perform_reshape: bool=True, should_post_hoc_fit_to_image_extent: bool=False): # posterior_masking_value: float = 0.01 -- 1D
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
        # Delegate the posterior plotting functionality.
        heatmaps, image_extent, extra_dict = cls._helper_add_heatmap(
            an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, ybin_centers=ybin_centers,
            rotate_to_vertical=rotate_to_vertical, debug_print=debug_print, 
            posterior_masking_value=posterior_masking_value, should_perform_reshape=should_perform_reshape)
        
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
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True)
            else:
                # 2D case
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True)
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



    @function_attributes(short_name=None, tags=['plot'], input_requires=[], output_provides=[], uses=[], used_by=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side'], creation_date='2025-06-30 12:58', related_items=[])
    def plot_decoded_trajectories_2d(self, sess, curr_num_subplots=10, active_page_index=0, plot_actual_lap_lines:bool=False, fixed_columns: int = 2, use_theoretical_tracks_instead: bool = True, existing_ax=None, axes_inset_locators_list=None):
        """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots
        
        Called to setup the graph.
        
        Great plotting for laps.
        Plots in a paginated manner.
        
        use_theoretical_tracks_instead: bool = True - # if False, renders all positions the animal traversed over the entire session. Otherwise renders the theoretical (idaal) track.

        ISSUE: `fixed_columns: int = 1` doesn't work due to indexing


        History: based off of plot_lap_trajectories_2d

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_decoded_trajectories_2d
        
            fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)

        
        """

        if use_theoretical_tracks_instead:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(deepcopy(sess.config))


        def _subfn_chunks(iterable, size=10):
            iterator = iter(iterable)
            for first in iterator:    # stops when iterator is depleted
                def chunk():          # construct generator for next chunk
                    yield first       # yield element from for loop
                    for more in islice(iterator, size - 1):
                        yield more    # yield more elements from the iterator
                yield chunk()         # in outer generator, yield next chunk
            
        def _subfn_build_epochs_multiplotter(nfields, linear_plot_data=None):
            """ builds the figures
             captures: self.rotate_to_vertical, fixed_columns, (long_track_inst, short_track_inst)
            
            """
            linear_plotter_indicies = np.arange(nfields)
            needed_rows = int(np.ceil(nfields / fixed_columns))
            row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
            
            if existing_ax is None:
                ## Create a new axes and figure
                fig, axs = plt.subplots(needed_rows, fixed_columns, sharex=True, sharey=True, figsize=[4*fixed_columns,14*needed_rows], gridspec_kw={'wspace': 0, 'hspace': 0}) #ndarray (5,2)
                
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
                        ax_inset = existing_ax.inset_axes(ax_inset_location, transform=existing_ax.transData, borderpad=0) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`
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
                    background_track_shadings[a_linear_index] = an_ax.plot(linear_plot_data[a_linear_index][0,:], linear_plot_data[a_linear_index][1,:], c='k', alpha=0.2)
                else:
                    # active_config = curr_active_pipeline.sess.config
                    background_track_shadings[a_linear_index] = _perform_plot_matplotlib_2D_tracks(long_track_inst=long_track_inst, short_track_inst=short_track_inst, ax=an_ax, rotate_to_vertical=self.rotate_to_vertical)
                
            return fig, axs, linear_plotter_indicies, row_column_indicies, background_track_shadings
        
        def _subfn_add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, lap_position_traces, lap_time_ranges, use_time_gradient_line=True):
            # Add the lap trajectory:
            for a_linear_index in linear_plotter_indicies:
                curr_lap_id = active_page_laps_ids[a_linear_index]
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                curr_lap_time_range = lap_time_ranges[curr_lap_id]
                curr_lap_label_text = 'Lap[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[0], curr_lap_time_range[1])
                curr_lap_num_points = len(lap_position_traces[curr_lap_id][0,:])
                if use_time_gradient_line:
                    # Create a continuous norm to map from data points to colors
                    curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(lap_position_traces[curr_lap_id][0,:]))
                    norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
                    # needs to be (numlines) x (points per line) x 2 (for x and y)
                    points = np.array([lap_position_traces[curr_lap_id][0,:], lap_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap='viridis', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(curr_lap_timeseries)
                    lc.set_linewidth(2)
                    lc.set_alpha(0.85)
                    a_line = axs[curr_row][curr_col].add_collection(lc)
                    # add_arrow(line)
                else:
                    a_line = axs[curr_row][curr_col].plot(lap_position_traces[curr_lap_id][0,:], lap_position_traces[curr_lap_id][1,:], c='k', alpha=0.85)
                    # curr_lap_endpoint = curr_lap_position_traces[curr_lap_id][:,-1].T
                    a_start_arrow = _plot_helper_add_arrow(a_line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                    a_middle_arrow = _plot_helper_add_arrow(a_line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                    a_end_arrow = _plot_helper_add_arrow(a_line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                    # add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                    # add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')
                # add lap text label
                a_lap_label_text = axs[curr_row][curr_col].text(250, 126, curr_lap_label_text, horizontalalignment='right', size=12)
                # PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Compute required data from session:
        curr_position_df, lap_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(sess)
        
        # lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in session.laps.lap_id]

        if self.rotate_to_vertical:
            # vertical
            # x_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("x")]
            # y_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("y")]

            for a_df in lap_specific_position_dfs:
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

        position_col_names = ['x', 'y']
        laps_position_traces_list = [lap_pos_df[position_col_names].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
        
        laps_time_range_list = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]
        
        num_laps = len(sess.laps.lap_id)
        linear_lap_index = np.arange(num_laps)
        lap_time_ranges = dict(zip(sess.laps.lap_id, laps_time_range_list))
        lap_position_traces = dict(zip(sess.laps.lap_id, laps_position_traces_list)) ## each lap indexed by lap_id
        
        all_maze_positions = curr_position_df[position_col_names].to_numpy().T # (2, 59308)
        # np.shape(all_maze_positions)
        all_maze_data = [all_maze_positions for i in np.arange(curr_num_subplots)] # repeat the maze data for each subplot. (2, 593080)
        
        # Build Figures/Axes/Etc _____________________________________________________________________________________________ #
        self.fig, self.axs, self.linear_plotter_indicies, self.row_column_indicies, background_track_shadings = _subfn_build_epochs_multiplotter(curr_num_subplots, all_maze_data)
        perform_update_title_subtitle(fig=self.fig, ax=None, title_string="DecodedTrajectoryMatplotlibPlotter - plot_decoded_trajectories_2d") # , subtitle_string="TEST - SUBTITLE"
        
        # generate the pages
        epochs_pages = [list(chunk) for chunk in _subfn_chunks(sess.laps.lap_id, curr_num_subplots)] ## this is specific to actual laps...
         
        if plot_actual_lap_lines:
            ## IDK what this is sadly, i think it's a reminant of the lap plotter?
            active_page_laps_ids = epochs_pages[active_page_index]
            _subfn_add_specific_lap_trajectory(self.fig, self.axs, linear_plotter_indicies=self.linear_plotter_indicies, row_column_indicies=self.row_column_indicies, active_page_laps_ids=active_page_laps_ids, lap_position_traces=lap_position_traces, lap_time_ranges=lap_time_ranges, use_time_gradient_line=True)
            # plt.ylim((125, 152))
            
        self.laps_pages = epochs_pages



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



from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_point_labels


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
    p = field(default=None)
    curr_time_bin_index: int = field(default=0)
    enable_point_labels: bool = field(default=False)
    enable_plot_all_time_bins_in_epoch_mode: bool = field(default=False)


    slider_epoch = field(default=None)
    slider_epoch_time_bin = field(default=None)
    slider_epoch_time_bin_playback_checkbox = field(default=None)
    
    interactive_plotter: PhoInteractivePlotter = field(default=None)
    plotActors = field(default=None)
    data_dict = field(default=None)
    plotActors_CenterLabels = field(default=None)
    data_dict_CenterLabels = field(default=None)

    active_plot_fn: Callable = field(default=plot_3d_stem_points) # like [plot_3d_binned_bars, plot_3d_stem_points]
    animation_callback_interval_ms: int = field(default=200) # 200ms per time bin

    def build_ui(self):
        """ builds the slider vtk widgets 
        """

        assert self.p is not None
        if self.curr_epoch_idx is None:
            self.curr_epoch_idx = 0
        
        num_filter_epochs: int = self.num_filter_epochs
        curr_num_epoch_time_bins: int = self.curr_n_time_bins

        slider_epoch_kwargs = dict()
        if self.enable_plot_all_time_bins_in_epoch_mode:
            slider_epoch_kwargs = slider_epoch_kwargs | dict(event_type="always")

        if self.slider_epoch is None:
            def _on_slider_value_did_change_epoch_idx(value):
                """ only called when the value actually changes from the previous one (or there wasn't a previous one). """
                self.on_update_slider_epoch_idx(int(value))


            def _on_slider_callback_epoch_idx(value):
                """ checks whether the value has changed from the previous one before re-updating. 
                """
                if not hasattr(_on_slider_callback_epoch_idx, "last_value"):
                    _on_slider_callback_epoch_idx.last_value = value
                if value != _on_slider_callback_epoch_idx.last_value:
                    _on_slider_value_did_change_epoch_idx(value)
                    _on_slider_callback_epoch_idx.last_value = value


            self.slider_epoch = self.p.add_slider_widget(
                # callback=lambda value: self.on_update_slider_epoch_idx(int(value)), #storage_engine('epoch', int(value)), # triggering .__call__(self, param='epoch', value)....
                callback=lambda value: _on_slider_callback_epoch_idx(int(value)),
                rng=[0, num_filter_epochs-1],
                value=0,
                title="Epoch Idx",
                pointa=(0.64, 0.2),
                pointb=(0.94, 0.2),
                style='modern',
                fmt='%0.0f',
                **slider_epoch_kwargs,
            )


        if not self.enable_plot_all_time_bins_in_epoch_mode:
            if self.slider_epoch_time_bin is None:
                def _on_slider_value_did_change_epoch_time_bin(value):
                    """ only called when the value actually changes from the previous one (or there wasn't a previous one). """
                    self.on_update_slider_epoch_time_bin(int(value))


                def _on_slider_callback_epoch_time_bin(value):
                    """ checks whether the value has changed from the previous one before re-updating. This might not be the best approach because it should be forcibly re-updated when the epoch_idx changes even if the time_bin_idx stays the same (like it's sitting at 0 while scrolling through epochs)
                    """
                    if not hasattr(_on_slider_callback_epoch_time_bin, "last_value"):
                        _on_slider_callback_epoch_time_bin.last_value = value
                    if value != _on_slider_callback_epoch_time_bin.last_value:
                        _on_slider_value_did_change_epoch_time_bin(value)
                        _on_slider_callback_epoch_time_bin.last_value = value

                self.slider_epoch_time_bin = self.p.add_slider_widget(
                    # callback=lambda value: self.on_update_slider_epoch_time_bin(int(value)), #storage_engine('time_bin', value),
                    callback=lambda value: _on_slider_callback_epoch_time_bin(int(value)),
                    rng=[0, curr_num_epoch_time_bins-1],
                    value=0,
                    title="Timebin IDX",
                    pointa=(0.74, 0.12),
                    pointb=(0.94, 0.12),
                    style='modern',
                    # fmt="%d",
                    event_type="always",
                    fmt='%0.0f',
                )

            if (self.interactive_plotter is None) or (self.slider_epoch_time_bin_playback_checkbox is None):
                self.interactive_plotter = PhoInteractivePlotter.init_from_plotter_and_slider(pyvista_plotter=self.p, interactive_timestamp_slider_actor=self.slider_epoch_time_bin, step_size=1, animation_callback_interval_ms=self.animation_callback_interval_ms) # 500ms per time bin
                self.slider_epoch_time_bin_playback_checkbox = self.interactive_plotter.interactive_checkbox_actor


    def update_ui(self):
        """ called to update the epoch_time_bin slider when the epoch_index slider is changed. 
        """
        if (self.slider_epoch_time_bin is not None) and (self.curr_n_time_bins is not None):
            self.slider_epoch_time_bin.GetRepresentation().SetMaximumValue((self.curr_n_time_bins-1))
            self.slider_epoch_time_bin.GetRepresentation().SetValue(self.slider_epoch_time_bin.GetRepresentation().GetMinimumValue()) # set to 0


    def perform_programmatic_slider_epoch_update(self, value):
        """ called to programmatically update the epoch_idx slider. """
        if (self.slider_epoch is not None):
            print(f'updating slider_epoch index to : {int(value)}')
            self.slider_epoch.GetRepresentation().SetValue(int(value)) # set to 0
            self.on_update_slider_epoch_idx(value=int(value))
            print(f'\tdone.')

    def on_update_slider_epoch_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
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

        ## shouldn't be here:
        # update_plot_fn = self.data_dict.get('plot_3d_binned_bars[55.63197815967686]', {}).get('update_plot_fn', None)
        update_plot_fn = self.data_dict.get('plot_3d_stem_points_P_x_given_n', {}).get('update_plot_fn', None)
        if update_plot_fn is not None:
            update_plot_fn(self.curr_time_bin_index)



    def on_update_slider_epoch_time_bin(self, value: int):
        """ called when the epoch_time_bin within a given epoch_idx slider changes 
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        self.perform_update_plot_single_epoch_time_bin(value=value)
        


    @function_attributes(short_name=None, tags=['main_plot_update', 'single_time_bin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:03', related_items=[])
    def perform_update_plot_single_epoch_time_bin(self, value: int):
        """ single-time-bin plotting:
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
        

    @function_attributes(short_name=None, tags=['main_plot_update', 'multi_time_bins', 'epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:04', related_items=[])
    def perform_update_plot_epoch_time_bin_range(self, value: Optional[NDArray]=None):
        """ multi-time-bin plotting:
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
        ## remove existing actors
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import clear_3d_binned_bars_plots

        if self.plotActors is not None:
            clear_3d_binned_bars_plots(p=self.p, plotActors=self.plotActors)
            self.plotActors.clear()
        if self.data_dict is not None:
            self.data_dict.clear()

        if self.plotActors_CenterLabels is not None:
            self.plotActors_CenterLabels.clear()
        if self.data_dict_CenterLabels is not None:
            self.data_dict_CenterLabels.clear()




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
    def add_decoded_posterior_bars(self, a_result: DecodedFilterEpochsResult, xbin: NDArray, xbin_centers: NDArray, ybin: Optional[NDArray], ybin_centers: Optional[NDArray], enable_plot_all_time_bins_in_epoch_mode:bool=True, active_plot_fn=None):
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
    def perform_plot_posterior_fn(cls, p, xbin, ybin, xbin_centers, ybin_centers, posterior_p_x_given_n, time_bin_centers=None, enable_point_labels: bool = True, point_labeling_function=None, point_masking_function=None, posterior_name='P_x_given_n', active_plot_fn=None):
        """ called to perform the mesh generation and add_mesh calls
        
        Looks like it switches between 3 different potential plotting functions, all imported directly below

        ## Defaults to `plot_3d_binned_bars` if nothing else is provided        
        
        """
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_point_labels

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
        
            # plotActors, data_dict = active_plot_fn(p, xbin, ybin, posterior_p_x_given_n, drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)
            plotActors, data_dict = active_plot_fn(p, active_xbins, active_ybins, posterior_p_x_given_n, drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)

            # , **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75} | kwargs)

            if point_labeling_function is None:
                # The full point shown:
                # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
                # Only the z-values
                point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'

            if point_masking_function is None:
                # point_masking_function = lambda points: points[:, 2] > 20.0
                point_masking_function = lambda points: points[:, 2] > 1E-6

            if enable_point_labels:
                plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(p, xbin_centers, ybin_centers, posterior_p_x_given_n, 
                                                                                    point_labels=point_labeling_function, 
                                                                                    point_mask=point_masking_function,
                                                                                    shape='rounded_rect', shape_opacity= 0.5, show_points=False, name=f'{posterior_name}Labels')
            else:
                plotActors_CenterLabels, data_dict_CenterLabels = None, None

        else:
            ## multi-time bin plot:
            from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars_timeseries

            assert np.ndim(posterior_p_x_given_n) == 3

            plotActors, data_dict = plot_3d_binned_bars_timeseries(p=p, xbin=active_xbins, ybin=active_ybins, t_bins=time_bin_centers, data=posterior_p_x_given_n,
                                           drop_below_threshold=1E-6, name=posterior_name, opacity=0.75, active_plot_fn=active_plot_fn)
            
            if enable_point_labels:
                print(f'WARN: enable_point_labels is not currently implemented for multi-time-bin plotting mode.')

            plotActors_CenterLabels, data_dict_CenterLabels = None, None



        return (plotActors, data_dict), (plotActors_CenterLabels, data_dict_CenterLabels)


