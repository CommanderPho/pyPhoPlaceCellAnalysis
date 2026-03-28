from copy import deepcopy
from pathlib import Path
from warnings import warn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Sequence
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from typing import NewType

import attrs
from attrs import asdict, define, field, Factory, astuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from datetime import datetime, timedelta

from attrs import define, field, asdict, evolve
import neuropy.utils.type_aliases as types # import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior, determine_event_interval_identity, determine_event_interval_is_included # numba acceleration
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor

from neuropy.utils.misc import build_shuffled_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from neuropy.utils.misc import split_array
from neuropy.core.position import PositionAccessor, Position, PositionComputedDataMixin
import pyphoplacecellanalysis.General.type_aliases as types

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert


from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
## NEW: filtering by whether decoded posterior in each t_bin is "position-like"
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring


# ==================================================================================================================================================================================================================================================================================== #
# 2026-01-21 - Vispy                                                                                                                                                                                                                                                                   #
# ==================================================================================================================================================================================================================================================================================== #
from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import TableManager
import colorsys
## vispy
import vispy
import vispy as vp
from vispy import scene
# from vispy import app, scene
# from vispy import app, gloo, visuals
# from vispy.scene.visuals import Arrow, Markers, Line
import vispy.scene.visuals as vz
from vispy.color import Colormap, Color, get_colormap
from qtpy import QtWidgets, QtCore

# # Vispy - Extreme Debugging __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
# # Set the logging level to DEBUG or INFO
# vp.set_log_level('DEBUG')
# # Optional: specifically tell VisPy to be more talkative about config
# vp.sys_info()
# # Enable full debug mode for the gloo layer
# vp.config.update(debug=True, check_errors=True)


# Vispy - Normal Running/Development __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
# Set the logging level to DEBUG or INFO
vp.set_log_level('WARNING')
# Optional: specifically tell VisPy to be more talkative about config
# vp.sys_info()
# Enable full debug mode for the gloo layer
# vp.config.update(debug=False, check_errors=True)

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import MatchingPastFuturePositionsResult, MaskDataSource

from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import VispyHelpers, ContourItem, contours_from_masks, create_contour_line_visuals, VispySceneTreeWidget, _ensure_closed_pos, _triangulate_polygon_2d
from pyphoplacecellanalysis.Pho2D.vispy.predictive_decoding_central_view import render_central_view as render_predictive_decoding_central_view
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig


## missing imports: MatchingPastFuturePositionsResult, BasePositionDecoder, MaskDataSource

@metadata_attributes(short_name=None, tags=['vispy', 'rendering', 'standalone'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21', related_items=[])
@define(slots=False, repr=False, eq=False)
class PredictiveDecodingVispyWidget:
    """Vispy-based widget that renders predictive decoding data (same data as PredictiveDecodingDisplayWidget).

    Keyboard: Left/Right arrow to change epoch. Use init_from_list or init_from_datasource to create;
    render_predictive_decoding_with_vispy is a thin wrapper that returns (main_window, canvas, state).
    """
    # Data / config
    epoch_flat_mask_future_past_result: Optional[List[MatchingPastFuturePositionsResult]] = field(default=None)
    a_decoded_filter_epochs_df: Optional[pd.DataFrame] = field(default=None)
    a_flat_matching_results_list_ds: Optional['MaskDataSource'] = field(default=None)
    curr_position_df: pd.DataFrame = field(default=None)
    pf_decoder: BasePositionDecoder = field(default=None)
    decoded_result: DecodedFilterEpochsResult = field(default=None)
    active_epoch_idx: int = field(default=0)
    
    current_traj_seconds_pre_post_extension: float = field(default=0.750)
    past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = field(default=(0.4, 1.0))
    start_end_extension_max_opacity: float = field(default=0.4)
    show_full_position_background: bool = field(default=False)
    
    require_angle_match: bool = field(default=False)
    color_matches_by_matching_angle: bool = field(default=False)
    color_matches_by_merged_epoch_t_bin_idx: bool = field(default=False)
    
    enable_debug_plot_trajectory_average_angle_arrows: bool = field(default=False)
    minimum_included_matching_sequence_length: Optional[int] = field(default=None)
    

    enable_full_vispy_debug_mode: bool = field(default=False)
    enable_line_render_debug_logging: bool = field(default=True)

    max_time_bins_to_show: int = field(default=12)
    enable_table_widgets: bool = field(default=False)

    enable_multi_epoch_overview_display_mode: bool = field(default=False)
    multi_epoch_overview_container_render_dict_list: Optional[Dict] = field(default=None, metadata={'desc': "only used when `enable_multi_epoch_overview_display_mode == True`"}) ## only used when `enable_multi_epoch_overview_display_mode == True`
    MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER: int = field(default=6, metadata={'desc': "only used when `enable_multi_epoch_overview_display_mode == True`"})

    # Derived in setup()
    xbin: Optional[Any] = field(default=None)
    ybin: Optional[Any] = field(default=None)
    num_epochs: int = field(default=0)
    recording_t_min: float = field(default=0.0)
    recording_t_max: float = field(default=1.0)
    past_future_trajectory_start_extension_seconds: float = field(default=0.0)
    past_future_trajectory_end_extension_seconds: float = field(default=0.0)
    

    # UI / vispy (created in buildUI)
    canvas: Any = field(default=None)
    main_window: Any = field(default=None)
    grid: vispy.scene.widgets.grid.Grid = field(default=None)
    past_view: Any = field(default=None)
    posterior_2d_view: Any = field(default=None)
    future_view: Any = field(default=None)
    time_bin_grid: vispy.scene.widgets.grid.Grid = field(default=None)
    time_bin_views: List[Any] = field(default=Factory(list))
    time_bin_raster: Any = field(default=None)

    combined_timeline_view: Any = field(default=None)
    colorbar_view: Any = field(default=None)
    epoch_slider: Any = field(default=None)
    epoch_value_label: Any = field(default=None)
    epoch_table_manager: Optional[TableManager] = field(default=None)
    current_epoch_idx: int = field(default=0)
    
    # Mutable visual lists (cleared/repopulated in update_epoch_display)
    past_lines: List[Any] = field(default=Factory(list))
    future_lines: List[Any] = field(default=Factory(list))
    time_bin_images: List[Any] = field(default=Factory(list))
    time_bin_labels: List[Any] = field(default=Factory(list))
    past_mask_contours: List[Any] = field(default=Factory(list))
    posterior_mask_contours: List[Any] = field(default=Factory(list))
    future_mask_contours: List[Any] = field(default=Factory(list))
    colorbar_rects: List[Any] = field(default=Factory(list))
    colorbar_texts: List[Any] = field(default=Factory(list))
    centroid_dots: List[Any] = field(default=Factory(list))
    centroid_arrows: List[Any] = field(default=Factory(list))
    trajectory_debug_arrows: Dict[types.PastFutureCategory, List[Any]] = field(default=Factory(dict))
    render_data_dict_list_dict: Dict[types.PastFutureCategory, List[Dict]] = field(default=Factory(dict))

    full_position_background_line: List[Any] = field(default=Factory(list))
    timeline_ticks: List[Any] = field(default=Factory(list))
    trajectory_arrows: List[Any] = field(default=Factory(list))
    posterior_img: Any = field(default=None)
    epoch_info_text: Any = field(default=None)
    current_position_line: Any = field(default=None)
    timeline_bar: Optional[vz.Rectangle] = field(default=None)
    timeline_epoch_rect: Optional[vz.Rectangle] = field(default=None)
    timeline_epoch_triangle: Optional[vz.Polygon] = field(default=None)
    _last_trajectory_epoch_data: Optional[Dict[types.PastFutureCategory, Any]] = field(default=None)


    def __attrs_post_init__(self):
        
        if self.enable_full_vispy_debug_mode:
            # Vispy - Extreme Debugging __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # Set the logging level to DEBUG or INFO
            # vp.set_log_level('DEBUG')
            vp.set_log_level('WARNING')
            # # Optional: specifically tell VisPy to be more talkative about config
            # vp.sys_info()
            # Enable full debug mode for the gloo layer
            vp.config.update(gl_debug=True)
            # vp.config.update(gl_debug=False) # gl_debug=True, logging_level='DEBUG'
            
        else:
            # Vispy - Normal Running/Development __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # Set the logging level to DEBUG or INFO
            vp.set_log_level('WARNING')
            # Optional: specifically tell VisPy to be more talkative about config
            # vp.sys_info()
            # Enable full debug mode for the gloo layer
            vp.config.update(gl_debug=False) # , check_errors=True

        if self.a_flat_matching_results_list_ds is None and self.epoch_flat_mask_future_past_result is not None and self.a_decoded_filter_epochs_df is not None:
            self.a_flat_matching_results_list_ds = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result=self.epoch_flat_mask_future_past_result, filter_epochs=self.a_decoded_filter_epochs_df)
        self.setup()
        self.buildUI()


    @classmethod
    def init_from_list(cls, epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], a_decoded_filter_epochs_df: pd.DataFrame, curr_position_df: pd.DataFrame, pf_decoder: BasePositionDecoder, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0,
        current_traj_seconds_pre_post_extension: float = 0.750, past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = (0.4, 1.0), start_end_extension_max_opacity: float = 0.4, show_full_position_background: bool = False,
        require_angle_match: bool = False, color_matches_by_matching_angle: bool = False, enable_debug_plot_trajectory_average_angle_arrows: bool = False, minimum_included_matching_sequence_length: Optional[int] = None, **kwargs) -> "PredictiveDecodingVispyWidget":
        a_flat_matching_results_list_ds = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result, filter_epochs=a_decoded_filter_epochs_df)
        
        return cls(
            epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result,
            a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
            a_flat_matching_results_list_ds=a_flat_matching_results_list_ds,
            curr_position_df=curr_position_df,
            pf_decoder=pf_decoder,
            decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx,
            current_traj_seconds_pre_post_extension=current_traj_seconds_pre_post_extension,
            past_future_trajectory_extension_seconds=past_future_trajectory_extension_seconds,
            start_end_extension_max_opacity=start_end_extension_max_opacity,
            show_full_position_background=show_full_position_background,
            require_angle_match=require_angle_match,
            color_matches_by_matching_angle=color_matches_by_matching_angle,
            enable_debug_plot_trajectory_average_angle_arrows=enable_debug_plot_trajectory_average_angle_arrows,
            minimum_included_matching_sequence_length=minimum_included_matching_sequence_length,
            **kwargs)

    @classmethod
    def init_from_datasource(cls, datasource: 'MaskDataSource', curr_position_df: pd.DataFrame, pf_decoder: Any, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0, **kwargs) -> "PredictiveDecodingVispyWidget":
        return cls(
            epoch_flat_mask_future_past_result=None,
            a_decoded_filter_epochs_df=datasource.filter_epochs,
            a_flat_matching_results_list_ds=datasource,
            curr_position_df=curr_position_df,
            pf_decoder=pf_decoder,
            decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx)


    def setup(self):
        if self.pf_decoder is None:
            raise ValueError("pf_decoder must be provided")
        self.xbin = deepcopy(self.pf_decoder.xbin)
        self.ybin = deepcopy(self.pf_decoder.ybin)
        self.num_epochs = len(self.a_flat_matching_results_list_ds.p_x_given_n_list)
        if self.curr_position_df is not None and 't' in self.curr_position_df.columns:
            self.recording_t_min = self.curr_position_df['t'].min()
            self.recording_t_max = self.curr_position_df['t'].max()
        elif self.a_decoded_filter_epochs_df is not None:
            self.recording_t_min = self.a_decoded_filter_epochs_df['start'].min() if 'start' in self.a_decoded_filter_epochs_df.columns else 0.0
            self.recording_t_max = self.a_decoded_filter_epochs_df['stop'].max() if 'stop' in self.a_decoded_filter_epochs_df.columns else 1.0
        ext = self.past_future_trajectory_extension_seconds
        if isinstance(ext, (int, float)):
            self.past_future_trajectory_start_extension_seconds = float(ext)
            self.past_future_trajectory_end_extension_seconds = float(ext)
        elif isinstance(ext, (tuple, list)) and len(ext) == 2:
            self.past_future_trajectory_start_extension_seconds, self.past_future_trajectory_end_extension_seconds = float(ext[0]), float(ext[1])
        elif isinstance(ext, dict):
            self.past_future_trajectory_start_extension_seconds = float(ext.get('start', 0.0))
            self.past_future_trajectory_end_extension_seconds = float(ext.get('end', 0.0))
        else:
            self.past_future_trajectory_start_extension_seconds = 0.0
            self.past_future_trajectory_end_extension_seconds = 0.0


    def buildUI(self):
        # from vispy import app, scene
        # from qtpy import QtWidgets, QtCore
        self.current_epoch_idx = self.active_epoch_idx

        # Main Canvas Init ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1920, 1080), title='Predictive Decoding Display - Vispy')
        canvas = scene.SceneCanvas(show=False, size=(1920, 1080), title='Predictive Decoding Display - Vispy',
            # keys='interactive',
            autoswap=False, resizable=True, decorate=True, fullscreen=False,
            # parent=self,
        )
        self.canvas = canvas


        # Build UI ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle('Predictive Decoding Display - Vispy')
        self.main_window = main_window
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add Native Canvas Widget ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        main_layout.addWidget(canvas.native, stretch=1)
        

        if not self.enable_multi_epoch_overview_display_mode:
            slider_widget = QtWidgets.QWidget()
            slider_layout = QtWidgets.QHBoxLayout(slider_widget)
            slider_label = QtWidgets.QLabel("Epoch:")
            epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            epoch_slider.setMinimum(0)
            epoch_slider.setMaximum(max(0, self.num_epochs - 1))
            epoch_slider.setValue(min(self.active_epoch_idx, self.num_epochs - 1))
            epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            epoch_slider.setTickInterval(1)
            epoch_value_label = QtWidgets.QLabel(f"{self.active_epoch_idx}/{self.num_epochs}")
            epoch_value_label.setMinimumWidth(60)
            self.epoch_slider = epoch_slider
            self.epoch_value_label = epoch_value_label
            slider_layout.addWidget(slider_label)
            slider_layout.addWidget(epoch_slider, stretch=1)
            slider_layout.addWidget(epoch_value_label)
            main_layout.addWidget(slider_widget)


        if self.enable_table_widgets:
            from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import TableManager
            table_container = QtWidgets.QWidget()
            table_container.setMaximumHeight(450)
            visible_columns_dict = {
                'curr_merged_segment_epochs': ['start', 'stop', 'is_future_present_past', 'epoch_t_idx', 'label', 'duration', 'num_epoch_t_bins', 'is_reversely_replayed', 'pre_merged_epoch_label'],
                'curr_merged_pos_epochs': ['start', 'stop', 'is_future_present_past', 'label', 'duration'],
            }
            epoch_table_manager = TableManager(table_container, visible_columns_dict=visible_columns_dict)
            self.epoch_table_manager = epoch_table_manager
            
            main_layout.addWidget(table_container)
            

        main_window.setCentralWidget(central_widget)
        main_window.resize(1400, 950)
        main_window.show()
        
        grid: vispy.scene.widgets.grid.Grid = canvas.central_widget.add_grid() # vispy.scene.widgets.grid.Grid
        self.grid = grid
        


        if not self.enable_multi_epoch_overview_display_mode:
            # ==================================================================================================================================================================================================================================================================================== #
            # SINGLE EPOCH VIEW WITH SLIDER                                                                                                                                                                                                                                                        #
            # ==================================================================================================================================================================================================================================================================================== #
            # Default single epoch plotting mode with slider to control active epoch _____________________________________________________________________________________________________________________________________________________________________________________________________________ #
            self.past_view = grid.add_view(row=0, col=0, col_span=1, row_span=2, border_color='red')
            self.future_view = grid.add_view(row=0, col=2, col_span=1, row_span=2, border_color='blue')
    
            self.posterior_2d_view = grid.add_view(row=0, col=1, col_span=1, border_color='gray')
            
            curr_row: int = 1
            self.time_bin_grid: vispy.scene.widgets.grid.Grid = grid.add_grid(row=curr_row, col=1, col_span=1, border_color='gray')
            self.time_bin_grid.height_max = 120
            curr_row = curr_row + 1

            ## add raster
            self.time_bin_raster = grid.add_view(row=curr_row, col=1, col_span=1, border_color='gray')
            self.time_bin_raster.height_max = 120
            curr_row = curr_row + 1

            self.combined_timeline_view = grid.add_view(row=curr_row, col=0, col_span=3, border_color='gray')
            self.combined_timeline_view.height_max = 40
            curr_row = curr_row + 1

            self.colorbar_view = grid.add_view(row=curr_row, col=0, col_span=3, border_color='gray') # curr_row: 3
            self.colorbar_view.height_max = 60


            # Only when single-epoch mode ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            if hasattr(canvas.events, 'key_press'):
                canvas.events.key_press.connect(self.on_key_press)

            for view in [self.past_view, self.posterior_2d_view, self.future_view]:
                view.camera = scene.PanZoomCamera(aspect=1)
                vz.GridLines(parent=view.scene)
                
            self.colorbar_view.camera = scene.PanZoomCamera(aspect=1)
            x_min, x_max = self.xbin[0], self.xbin[-1]
            y_min, y_max = self.ybin[0], self.ybin[-1]
            bbox_vertices = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]], dtype=np.float32)
            for view in [self.past_view, self.posterior_2d_view, self.future_view]:
                bbox_line = vz.Line(pos=bbox_vertices, color='white', width=2, parent=view.scene)
                self._debug_log_line_visual(bbox_line, context='buildUI_bbox', pos=bbox_vertices)
            extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
            for view in [self.past_view, self.future_view]:
                view.camera.set_range(x=(extent[0], extent[1]), y=(extent[2], extent[3]))
            y_range = extent[3] - extent[2]
            self.posterior_2d_view.camera.set_range(x=(extent[0], extent[1]), y=(extent[2] - y_range * 0.05, extent[3] + y_range * 0.2))
            timeline_bar_height = 1.0
            recording_duration = self.recording_t_max - self.recording_t_min
            self.combined_timeline_view.camera = scene.PanZoomCamera()
            self.combined_timeline_view.camera.set_range(x=(self.recording_t_min, self.recording_t_max), y=(0, timeline_bar_height))
            self.update_epoch_display(self.active_epoch_idx)

            assert epoch_slider is not None
            epoch_slider.valueChanged.connect(self.on_slider_value_changed)
            epoch_slider.sliderReleased.connect(self.on_slider_released)
            
            # all_views = [self.past_view, self.future_view, self.posterior_2d_view, self.combined_timeline_view, self.colorbar_view]

        else:
            # ==================================================================================================================================================================================================================================================================================== #
            # MANY EPOCH VIEW AS STACKED GRID                                                                                                                                                                                                                                                      #
            # ==================================================================================================================================================================================================================================================================================== #

            ## build all data at once
            n_epochs: int = self.a_flat_matching_results_list_ds.num_epochs
            epoch_indicies = np.arange(n_epochs)
            epoch_data_list: List[Dict] = [self.a_flat_matching_results_list_ds._prepare_epoch_data(an_epoch_idx=new_epoch_idx, minimum_included_matching_sequence_length=self.minimum_included_matching_sequence_length) for new_epoch_idx in epoch_indicies]
            # epoch_data_list
            
            filter_epochs = self.a_flat_matching_results_list_ds.filter_epochs
            
            row_max_height: int = 640
            
            ## get standard data:            
            x_min, x_max = self.xbin[0], self.xbin[-1]
            y_min, y_max = self.ybin[0], self.ybin[-1]
        
            multi_epoch_overview_container_render_dict_list = [] # {}
            
            # if multi_epoch_overview_container is None:
            
        
            # self.posterior_2d_container_view = grid.add_view(row=0, col=0, col_span=1, border_color='gray')
            # self.posterior_2d_container_view_grid = grid.add_view(row=0, col=0, col_span=1, border_color='gray')
            for idx, epoch_data in enumerate(epoch_data_list):
                ## make a new local only temp container dict
                a_multi_epoch_overview_container = {'a_posterior_2d_container_view_grid': None, 'a_posterior_2d_view': None, 'a_time_bin_grid': None, 'a_time_bin_raster': None, 'an_update_dict': {}} #[]
                
                if idx >= self.MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER:
                    # print(f'MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER: {MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER}')
                    continue
                else:
                    print(f'idx: {idx} < MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER: {self.MAX_NUM_OVERVIEW_EPOCHS_TO_RENDER}, so adding...')
                
                n_time_bins: int = self.a_flat_matching_results_list_ds.num_epoch_time_bins[idx]
                # epoch_data
                # Get posterior data
                p_x_given_n = self.a_flat_matching_results_list_ds.p_x_given_n_list[idx]
                p_x_given_n = np.ascontiguousarray(p_x_given_n, dtype=np.float32)
                
                posterior_2d = np.sum(p_x_given_n, axis=2)
                posterior_2d = np.ascontiguousarray(posterior_2d, dtype=np.float32)

                # Generate time bin colors for use in trajectory and centroid coloring
                n_time_bins: int = p_x_given_n.shape[2]
                time_bin_colors = self._time_bin_colors(n_time_bins, alpha=0.9)
                
                if idx < len(filter_epochs):
                    epoch_row = filter_epochs.iloc[idx]
                    epoch_start_t = epoch_row['start'] if 'start' in epoch_row else epoch_row.get('t_start', None)
                    epoch_end_t = epoch_row['stop'] if 'stop' in epoch_row else epoch_row.get('t_stop', None)
                else:
                    epoch_start_t = None
                    epoch_end_t = None

                # ==================================================================================================================================================================================================================================================================================== #
                # Build Views                                                                                                                                                                                                                                                                          #
                # ==================================================================================================================================================================================================================================================================================== #
                a_posterior_2d_container_view_grid: vispy.scene.widgets.grid.Grid = grid.add_grid(row=idx, col=0, col_span=(1+n_time_bins), border_color='white') ## holds the epoch's two views
                a_posterior_2d_view = a_posterior_2d_container_view_grid.add_view(row=0, col=0, col_span=1, border_color='gray')
                a_posterior_2d_view.height_max = row_max_height
                curr_row: int = 0

                a_time_bin_grid: vispy.scene.widgets.grid.Grid = a_posterior_2d_container_view_grid.add_grid(row=curr_row, col=1, col_span=n_time_bins, border_color='gray') # curr_row: 0
                a_time_bin_grid.height_max = row_max_height
                curr_row = curr_row + 1
                a_time_bin_raster = a_posterior_2d_container_view_grid.add_view(row=curr_row, col=1, col_span=n_time_bins, border_color='gray')
                a_time_bin_raster.height_max = 120
                curr_row = curr_row + 1



                a_posterior_2d_container_view_grid.height_max = row_max_height
                a_multi_epoch_overview_container['a_posterior_2d_container_view_grid'] = a_posterior_2d_container_view_grid
                a_multi_epoch_overview_container['a_posterior_2d_view'] = a_posterior_2d_view
                a_multi_epoch_overview_container['a_time_bin_grid'] = a_time_bin_grid
                a_multi_epoch_overview_container['a_time_bin_raster'] = a_time_bin_raster

                _common_render_kwargs = dict(time_bin_colors=time_bin_colors, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, new_epoch_idx=idx)
                
                # ==================================================================================================================================================================================================================================================================================== #
                # Begin Render                                                                                                                                                                                                                                                                         #
                # ==================================================================================================================================================================================================================================================================================== #
                # if idx >= len(multi_epoch_overview_container_render_dict_list):
                #     ## create the empty dict
                #     _an_update_dict = dict(
                #         # centroid_dots=self.centroid_dots, centroid_arrows=self.centroid_arrows,
                #         # current_position_line=self.current_position_line, trajectory_arrows=self.trajectory_arrows, epoch_info_text=self.epoch_info_text,
                #         # time_bin_views=a_time_bin_grid, #self.multi_epoch_overview_container['a_time_bin_grid'],
                #         # time_bin_labels=self.time_bin_labels, time_bin_images=self.time_bin_images,
                #         # past_mask_contours=self.past_mask_contours, posterior_mask_contours=self.posterior_mask_contours, future_mask_contours=self.future_mask_contours,  
                #         posterior_2d_view=a_posterior_2d_view, time_bin_grid=a_time_bin_grid,
                #         # past_view=None, future_view=None,
                #         # past_mask_contours=[], posterior_mask_contours=[], future_mask_contours=[],
                #     )
                #     # _an_update_dict = {}
                #     multi_epoch_overview_container_render_dict_list.append(_an_update_dict)
                # else:
                #     _an_update_dict = multi_epoch_overview_container_render_dict_list[idx]
                
                _an_update_dict = dict(
                    # centroid_dots=self.centroid_dots, centroid_arrows=self.centroid_arrows,
                    # current_position_line=self.current_position_line, trajectory_arrows=self.trajectory_arrows, epoch_info_text=self.epoch_info_text,
                    # time_bin_views=a_time_bin_grid, #self.multi_epoch_overview_container['a_time_bin_grid'],
                    # time_bin_labels=self.time_bin_labels, time_bin_images=self.time_bin_images,
                    # past_mask_contours=self.past_mask_contours, posterior_mask_contours=self.posterior_mask_contours, future_mask_contours=self.future_mask_contours,  
                    posterior_2d_view=a_posterior_2d_view, time_bin_grid=a_time_bin_grid, time_bin_raster=a_time_bin_raster,
                    # past_view=None, future_view=None,
                    # past_mask_contours=[], posterior_mask_contours=[], future_mask_contours=[],
                )
            
                _an_update_dict = self._render_central_view(p_x_given_n=p_x_given_n, posterior_2d=posterior_2d,
                                        epoch_start_t=epoch_start_t, epoch_end_t=epoch_end_t,
                                        **_common_render_kwargs,
                                        allow_use_self_properties=False, needs_clear_owned_views=False, 
                                        _update_dict=_an_update_dict, #multi_epoch_overview_container_render_dict_list[idx],
                )
                # multi_epoch_overview_container_render_dict_list[idx] = _an_update_dict ## update the existing
                a_multi_epoch_overview_container['an_update_dict'] = _an_update_dict
                
            
                # multi_epoch_overview_container_render_dict_list.append(_update_dict)
                # for _k, _v in _update_dict.items():
                #     setattr(self, _k, _v)
                    
                multi_epoch_overview_container_render_dict_list.append(a_multi_epoch_overview_container)




            ## END for idx, epoch_data in enumerate(epoch_data_list)...
            print(f'\t finally updating self.multi_epoch_overview_container_render_dict_list... len(multi_epoch_overview_container_render_dict_list): {len(multi_epoch_overview_container_render_dict_list)}')
            self.multi_epoch_overview_container_render_dict_list = multi_epoch_overview_container_render_dict_list ## finally update self
            print(f'\tdone.')
                






    # ==================================================================================================================================================================================================================================================================================== #
    # Helper/Rendering Functions                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def _clear_epoch_visuals(self):
        # self._detach_and_clear_visual_lists(
        list_attr_names = [
            'past_lines', 'future_lines',
            'time_bin_images', 'time_bin_labels', ## required
            'past_mask_contours', 'posterior_mask_contours', 'future_mask_contours',
            'colorbar_rects', 'colorbar_texts', 'centroid_dots', 'centroid_arrows', 'timeline_ticks',
            'trajectory_debug_arrows', # 'render_data_dict_list_dict', ## dict-of-lists
        ] #,
        single_ref_attr_names = ['posterior_img', 'epoch_info_text', 'timeline_bar', 'timeline_epoch_rect', 'timeline_epoch_triangle'] #,
        # )
        
        for name in list_attr_names:
            lst = getattr(self, name)
            if isinstance(lst, (list, tuple)):
                for item in lst:
                    if item is not None:
                        item.parent = None
                lst.clear()
                
            elif isinstance(lst, dict):
                ## handle dict of lists
                for k, sub_list in lst.items():
                    for item in sub_list:
                        if item is not None:
                            try:
                                ## try the typical widget removal process
                                item.parent = None
                            except AttributeError as e:
                                ## for clearing a data-dict such as 'render_data_dict_list_dict'
                                ignored_variable_names = ['render_data_dict_list_dict']
                                if (name in ignored_variable_names) or (k in ignored_variable_names):
                                    pass
                                else:
                                    print(f'WARN: AttributeError e {e}')
                                    # raise
                            except Exception as e:
                                raise
                    ## END for item in sub_list...
                    
                    sub_list.clear() ## clear the sublist
                ## END for k, sub_list in lst.items()...
                lst.clear()
            else:
                raise ValueError(f'Unknown type type(lst): {type(lst)}, lst: {lst}')
            
        if single_ref_attr_names:
            for name in single_ref_attr_names:
                ref = getattr(self, name)
                if ref is not None:
                    ref.parent = None
                setattr(self, name, None)
                

    def _time_bin_colors(self, n_bins: int, alpha: float = 0.9) -> np.ndarray:
        """Return (n_bins, 4) float32 array of RGBA colors for time bins (hue cycled)."""
        out = np.zeros((n_bins, 4), dtype=np.float32)
        for t_idx in range(n_bins):
            hue = (t_idx / max(n_bins, 1)) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            out[t_idx] = (rgb[0], rgb[1], rgb[2], alpha)
        return out

    def _debug_log_line_visual(self, line: Any, context: str, pos: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Logs line id and finite stats so crash addresses can be mapped to source visuals."""
        if not (self.enable_line_render_debug_logging or self.enable_full_vispy_debug_mode):
            return
        parts: List[str] = [f'context={context}', f'line_id={hex(id(line)) if line is not None else "None"}']
        if pos is not None:
            pos_arr = np.asarray(pos)
            parts.append(f'pos_shape={pos_arr.shape}')
            parts.append(f'pos_finite={bool(np.all(np.isfinite(pos_arr)))}')
            if pos_arr.size > 0:
                parts.append(f'pos_min={float(np.nanmin(pos_arr)):.6f}')
                parts.append(f'pos_max={float(np.nanmax(pos_arr)):.6f}')
        if colors is not None:
            color_arr = np.asarray(colors)
            parts.append(f'color_shape={color_arr.shape}')
            parts.append(f'color_finite={bool(np.all(np.isfinite(color_arr)))}')
        if extra is not None:
            for k, v in extra.items():
                parts.append(f'{k}={v}')
        print('[VISPY_LINE_DEBUG] ' + ' | '.join(parts))

    def _segment_row_to_time_bin_idx(self, segment_row_idx: int, epoch_idx: int, mode='centroids') -> Optional[int]:
        """Resolve segment row index to time bin index for the given epoch."""
        if self.epoch_flat_mask_future_past_result is None or epoch_idx >= len(self.epoch_flat_mask_future_past_result):
            return None
        curr_epoch_result: MatchingPastFuturePositionsResult = self.epoch_flat_mask_future_past_result[epoch_idx]
        
        valid_modes_list = ['centroids', 'merged_segments']
        assert mode in valid_modes_list, f'mode: "{mode}" not in valid_modes_list: {valid_modes_list}'
        if mode == 'centroids':
            if curr_epoch_result is None or not hasattr(curr_epoch_result, 'centroids_df') or curr_epoch_result.centroids_df is None:
                return None
            if not hasattr(curr_epoch_result, 'a_centroids_search_segments_df') or curr_epoch_result.a_centroids_search_segments_df is None:
                return None
            search_df = curr_epoch_result.a_centroids_search_segments_df
            if segment_row_idx >= len(search_df):
                return None
            actual_segment_idx = search_df.iloc[segment_row_idx]['segment_idx']
            matching_t_bins = curr_epoch_result.centroids_df[curr_epoch_result.centroids_df['segment_idx'] == actual_segment_idx].index
            
        elif mode == 'merged_segments':
            raise NotImplementedError(f'mode: "merged_segments" not implemented!')
            
            # if curr_epoch_result is None or not hasattr(curr_epoch_result, 'centroids_df') or curr_epoch_result.centroids_df is None:
            #     return None
            # if not hasattr(curr_epoch_result, 'a_centroids_search_segments_df') or curr_epoch_result.a_centroids_search_segments_df is None:
            #     return None
            # search_df = curr_epoch_result.a_centroids_search_segments_df
            # if segment_row_idx >= len(search_df):
            #     return None
            # actual_segment_idx = search_df.iloc[segment_row_idx]['segment_idx']
            # matching_t_bins = curr_epoch_result.centroids_df[curr_epoch_result.centroids_df['segment_idx'] == actual_segment_idx].index
            # pass
        
        else:
            raise NotImplementedError(f'mode: "{mode}" not implemented!')

        return int(matching_t_bins[0]) if len(matching_t_bins) > 0 else None
    
    def _extend_trajectory_xy_opacity(self, x_valid: np.ndarray, y_valid: np.ndarray, opacity: np.ndarray, t_valid: np.ndarray, traj_t_min: float, traj_t_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply start/end trajectory extensions from curr_position_df; return (x_valid, y_valid, opacity)."""
        if self.past_future_trajectory_start_extension_seconds > 0 and self.curr_position_df is not None and 't' in self.curr_position_df.columns:
            ext_start_t = traj_t_min - self.past_future_trajectory_start_extension_seconds
            ext_mask = (self.curr_position_df['t'] >= ext_start_t) & (self.curr_position_df['t'] < traj_t_min)
            ext_positions = self.curr_position_df[ext_mask]
            if (len(ext_positions) > 0) and ('x' in ext_positions.columns) and ('y' in ext_positions.columns) and ('t' in ext_positions.columns):
                ext_x = np.asarray(ext_positions['x'].to_numpy(), dtype=np.float64)
                ext_y = np.asarray(ext_positions['y'].to_numpy(), dtype=np.float64)
                ext_t = np.asarray(ext_positions['t'].to_numpy(), dtype=np.float64)
                
                ext_valid_mask = np.isfinite(ext_x) & np.isfinite(ext_y) & np.isfinite(ext_t)
                if np.any(ext_valid_mask):
                    ext_x_valid = ext_x[ext_valid_mask]
                    ext_y_valid = ext_y[ext_valid_mask]
                    ext_t_valid = ext_t[ext_valid_mask]
                    t_valid = np.concatenate([ext_t_valid, t_valid]) ## important that it updates 't_valid'
                    x_valid = np.concatenate([ext_x_valid, x_valid])
                    y_valid = np.concatenate([ext_y_valid, y_valid])
                    ext_opacity = np.ones(len(ext_x_valid)) * self.start_end_extension_max_opacity
                    opacity = np.concatenate([ext_opacity, opacity])
                    
        if self.past_future_trajectory_end_extension_seconds > 0 and self.curr_position_df is not None and ('t' in self.curr_position_df.columns):
            ext_end_t = traj_t_max + self.past_future_trajectory_end_extension_seconds
            ext_mask = (self.curr_position_df['t'] > traj_t_max) & (self.curr_position_df['t'] <= ext_end_t)
            ext_positions = self.curr_position_df[ext_mask]
            if (len(ext_positions) > 0) and ('x' in ext_positions.columns) and ('y' in ext_positions.columns) and ('t' in ext_positions.columns):
                ext_x = np.asarray(ext_positions['x'].to_numpy(), dtype=np.float64)
                ext_y = np.asarray(ext_positions['y'].to_numpy(), dtype=np.float64)
                ext_t = np.asarray(ext_positions['t'].to_numpy(), dtype=np.float64)
                ext_valid_mask = np.isfinite(ext_x) & np.isfinite(ext_y) & np.isfinite(ext_t)
                if np.any(ext_valid_mask):
                    ext_x_valid = ext_x[ext_valid_mask]
                    ext_y_valid = ext_y[ext_valid_mask]
                    ext_t_valid = ext_t[ext_valid_mask]
                    t_valid = np.concatenate([t_valid, ext_t_valid])  ## important that it updates 't_valid'
                    x_valid = np.concatenate([x_valid, ext_x_valid])
                    y_valid = np.concatenate([y_valid, ext_y_valid])
                    ext_opacity = self.start_end_extension_max_opacity * (1.0 - (ext_t_valid - traj_t_max) / self.past_future_trajectory_end_extension_seconds)
                    opacity = np.concatenate([opacity, ext_opacity])
        return t_valid, x_valid, y_valid, opacity


    @function_attributes(short_name=None, tags=['CENTER_COL', 'update', 'pure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-03 13:12', related_items=[])
    def _render_central_view(self, p_x_given_n: pd.DataFrame, posterior_2d: pd.DataFrame, time_bin_colors: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, 
                             new_epoch_idx: int, epoch_start_t: float, epoch_end_t: float,
                             use_new_centroid_arrows: bool = True, use_single_arrows_object: bool = False,
                             _update_dict=None,
                             allow_use_self_properties: bool=True, needs_clear_owned_views: bool=True, 
                             **kwargs,
                             ):
        """Updates the center view with posteriors and time bins. Delegates to standalone render_predictive_decoding_central_view."""
        if _update_dict is None:
            _update_dict = {}
        if allow_use_self_properties:
            if _update_dict.get('posterior_2d_view') is None:
                _update_dict['posterior_2d_view'] = self.posterior_2d_view
            if _update_dict.get('past_view') is None:
                _update_dict['past_view'] = self.past_view
            if _update_dict.get('future_view') is None:
                _update_dict['future_view'] = self.future_view
            if _update_dict.get('time_bin_grid') is None:
                _update_dict['time_bin_grid'] = self.time_bin_grid
            if _update_dict.get('current_position_line') is None:
                _update_dict['current_position_line'] = self.current_position_line
        fallback_mask_2d_for_shape = None
        if (posterior_2d is None or getattr(posterior_2d, 'size', 1) == 0) and hasattr(self.a_flat_matching_results_list_ds, 'epoch_high_prob_pos_masks') and self.a_flat_matching_results_list_ds is not None and self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks is not None and new_epoch_idx < len(self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks):
            mask_2d = self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks[new_epoch_idx]
            if mask_2d is not None and getattr(mask_2d, 'size', 0) > 0:
                fallback_mask_2d_for_shape = mask_2d

        ## Raster:
        pf_decoder = kwargs.pop('pf_decoder', self.pf_decoder)
        active_aclus = np.array(pf_decoder.ratemap.neuron_ids)
        n_active_aclus: int = len(active_aclus)
        print(f'n_active_aclus: {n_active_aclus}')

        # unit_colors_list = None # default rainbow of colors for the raster plots
        neuron_qcolors_list = [pg.mkColor('black') for aclu in active_aclus] # solid green for all
        unit_colors_list = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)
        if (active_epochs_df is None or getattr(active_epochs_df, 'size', 1) == 0) and hasattr(self.a_flat_matching_results_list_ds, 'filter_epochs') and self.a_flat_matching_results_list_ds is not None and self.a_flat_matching_results_list_ds.filter_epochs is not None and new_epoch_idx < len(self.a_flat_matching_results_list_ds.filter_epochs):
            # active_epochs_df: pd.DataFrame = a_flat_matching_results_list_ds.filter_epochs[a_flat_matching_results_list_ds.filter_epochs['original_epoch_idx'] < 3]
            active_epochs_df: pd.DataFrame = a_flat_matching_results_list_ds.filter_epochs.iloc[new_epoch_idx]
        actIve_filter_epochs_spikes_df: pd.DataFrame = pf_decoder.spikes_df

        _raster_kwargs = dict(active_epochs_df=active_epochs_df, active_aclus=active_aclus, actIve_filter_epochs_spikes_df=actIve_filter_epochs_spikes_df, )

        _out = render_predictive_decoding_central_view(p_x_given_n=p_x_given_n, posterior_2d=posterior_2d, time_bin_colors=time_bin_colors, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, new_epoch_idx=new_epoch_idx, epoch_start_t=epoch_start_t, epoch_end_t=epoch_end_t, epoch_flat_mask_future_past_result=self.epoch_flat_mask_future_past_result,
            curr_position_df=self.curr_position_df, current_traj_seconds_pre_post_extension=self.current_traj_seconds_pre_post_extension, num_epochs=self.num_epochs, max_time_bins_to_show=self.max_time_bins_to_show, fallback_mask_2d_for_shape=fallback_mask_2d_for_shape,
            use_new_centroid_arrows=use_new_centroid_arrows, use_single_arrows_object=use_single_arrows_object, _update_dict=_update_dict, needs_clear_owned_views=needs_clear_owned_views,
            **_raster_kwargs, #active_epochs_df=active_epochs_df, actIve_filter_epochs_spikes_df=actIve_filter_epochs_spikes_df,
        )


        return _out


    def _render_trajectory_side(self, positions_dict: dict, epoch_anchor_t: Optional[float], default_hue: float, view: Any, trajectory_colors_and_times_out: list, max_time_distance: float, time_bin_colors: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, new_epoch_idx: int,
                                 lines_list: Optional[List]=None, 
                                 trajectory_debug_arrows: Optional[List]=None,
                                 render_data_dict_list: Optional[List]=None,
                                 ):
        """Render past or future trajectories into view; append to lines_list and trajectory_colors_and_times_out."""
        # from vispy import scene
        # from vispy.color import Colormap
        enable_debug_logging: bool = False
        
        if lines_list is None:
            lines_list = []
        if trajectory_debug_arrows is None:
            trajectory_debug_arrows = [] ## new list
        if render_data_dict_list is None:
            render_data_dict_list = []
        

        for epoch_id, positions_df in list(positions_dict.items()):
            _curr_render_data_dict = dict(epoch_id=epoch_id)
            
            if self.require_angle_match and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns and not (positions_df['centroid_pos_traj_matching_angle_idx'] >= 0).any():
                render_data_dict_list.append(_curr_render_data_dict)
                continue
            
            custom_cmap: Optional[Colormap] = None
            
            if len(positions_df) > 0 and 'x' in positions_df.columns and 'y' in positions_df.columns:
                x_coords, y_coords = np.asarray(positions_df['x'].to_numpy(), dtype=np.float64), np.asarray(positions_df['y'].to_numpy(), dtype=np.float64)
                valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
                if np.count_nonzero(valid_mask) >= 2:
                    x_valid, y_valid = x_coords[valid_mask], y_coords[valid_mask]
                    

                    # Color Modes ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                    if self.color_matches_by_matching_angle and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
                        matching_idx_values = positions_df['centroid_pos_traj_matching_angle_idx'].values
                        valid_match_indices = matching_idx_values[matching_idx_values >= 0]
                        if len(valid_match_indices) > 0:
                            segment_row_idx = int(valid_match_indices[0])
                            matched_t_idx = self._segment_row_to_time_bin_idx(segment_row_idx, new_epoch_idx, mode='centroids')
                            base_rgb = tuple(time_bin_colors[matched_t_idx][:3]) if (matched_t_idx is not None and matched_t_idx < len(time_bin_colors)) else colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                        else:
                            base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                            
                    elif self.color_matches_by_merged_epoch_t_bin_idx and 'matching_found_relevant_pos_epoch' in positions_df.columns:
                        matching_idx_values = positions_df['matching_found_relevant_pos_epoch'].values
                        valid_match_indices = matching_idx_values[matching_idx_values >= 0]
                        valid_rel_match_indices = valid_match_indices - np.nanmin(valid_match_indices) ## get the count of each value
                        n_total_valid_indicies: int = len(valid_rel_match_indices)
                        
                        valid_rel_match_indices_counts = {}
                        valid_rel_match_indices_start_idxs = {}
                        
                        for i, v in enumerate(valid_rel_match_indices):
                            if v not in valid_rel_match_indices_counts:
                                valid_rel_match_indices_counts[v] = 1 ## initialize to 1
                                valid_rel_match_indices_start_idxs[v] = i
                            else:
                                valid_rel_match_indices_counts[v] = valid_rel_match_indices_counts.get(v, 0) + 1 ## increment


                        valid_rel_match_indices_REL_counts = {k:(float(v)/float(n_total_valid_indicies)) for k, v in valid_rel_match_indices_counts.items()}
                        valid_rel_match_indices_REL_start_idxs = {k:(float(v)/float(n_total_valid_indicies-1)) for k, v in valid_rel_match_indices_start_idxs.items()} # -1 to get the last index
                        if enable_debug_logging:
                            print(F'epoch_id: {epoch_id}')
                        n_time_bin_colors: int = np.shape(time_bin_colors)[0] #  np.shape(time_bin_colors): (6, 4)
                        unique_valid_rel_match_indices: NDArray = np.unique(valid_rel_match_indices)
                        n_unique_valid_rel_match_indices: int = len(unique_valid_rel_match_indices)
                        if enable_debug_logging:
                            print(f'\ttime_bin_colors: {time_bin_colors}')
                        colors_from_NDArray: List[NDArray] = [time_bin_colors[i][:3] for i in np.arange(n_time_bin_colors)]
                        if enable_debug_logging:
                            print(f'\tcolors_from_NDArray: {colors_from_NDArray}')
                        controls = None
                        # controls = list(valid_rel_match_indices_REL_start_idxs.values()) ## just the acending counts
                        # print(f'controls: {controls}')
                        # controls = np.interp(np.linspace(0, 1.0, num=n_total_valid_indicies), controls, np.linspace(0, 1.0, num=n_time_bin_colors))
                        # controls = controls[:n_time_bin_colors] + [1.0] ## the last has to be 1.0
                        
                        if controls is not None:
                            if enable_debug_logging:
                                print(f'\tcontrols: {controls}, len(controls): {len(controls)}')
                            # assert len(controls) == n_time_bin_colors, f"len(controls): {len(controls)} != n_time_bin_colors: {n_time_bin_colors}"
                            assert len(controls) == (n_time_bin_colors+1), f"len(controls): {len(controls)} != (n_time_bin_colors+1): {(n_time_bin_colors+1)}"
                            custom_cmap = Colormap(colors=colors_from_NDArray, controls=controls, interpolation='zero') # , controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        else:
                            custom_cmap = Colormap(colors=colors_from_NDArray)
                            
                        if enable_debug_logging:
                            print(f'\tcustom_cmap: {custom_cmap}')
                        #TODO 2026-02-03 01:37: - [ ] Set controls from the correct values
                        base_rgb = None
                        # assert n_unique_valid_rel_match_indices <= n_time_bin_colors, f"n_unique_valid_rel_match_indices: {n_unique_valid_rel_match_indices}, n_time_bin_colors: {n_time_bin_colors}, unique_valid_rel_match_indices: {unique_valid_rel_match_indices}" 
                        # if len(valid_rel_match_indices) > 0:                            
                        #     # Sample the colormap at each vertex (0 to 1 along the line)
                        #     # t = np.linspace(0.0, 1.0, N)
                        #     # t_coords = positions_df['t'].values[valid_mask]
                        #     # n_points: int = len(x_valid)
                        #     segment_row_idx = int(valid_match_indices[0])
                        #     matched_t_idx = self._segment_row_to_time_bin_idx(segment_row_idx, new_epoch_idx, mode='merged_segments')
                        #     matched_t_idx = valid_match_indices
                        #     # base_rgb = tuple(time_bin_colors[matched_t_idx][:3]) if ((matched_t_idx is not None) and (matched_t_idx < len(time_bin_colors))) else colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                        # else:
                        #     base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)

                    else:
                        base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9) ## all the same (default) hue
                        custom_cmap = None


                    ## OUTPUTS: custom_cmap, base_rgb
                    if base_rgb is not None:
                        _curr_render_data_dict.update(base_rgb=base_rgb)
                    
                    if epoch_anchor_t is not None and 't' in positions_df.columns:
                        t_coords = np.asarray(positions_df['t'].to_numpy(), dtype=np.float64)[valid_mask]
                        valid_time_mask = np.isfinite(t_coords)
                        if np.count_nonzero(valid_time_mask) >= 2:
                            t_coords = t_coords[valid_time_mask]
                            x_valid = x_valid[valid_time_mask]
                            y_valid = y_valid[valid_time_mask]
                            mean_time = float(np.mean(t_coords))
                            if np.isfinite(mean_time):
                                trajectory_colors_and_times_out.append((colorsys.hsv_to_rgb(default_hue, 0.8, 0.9), mean_time))
                            time_rel = t_coords - epoch_anchor_t
                            time_distance = np.abs(time_rel)
                            opacity = (1.0 - (time_distance / max_time_distance) * 0.8) if max_time_distance > 0 else np.ones(len(x_valid)) * 0.8
                            traj_t_min, traj_t_max = float(np.min(t_coords)), float(np.max(t_coords))
                            t_valid, x_valid, y_valid, opacity = self._extend_trajectory_xy_opacity(x_valid, y_valid, opacity, t_coords, traj_t_min, traj_t_max)
                        else:
                            opacity = np.ones(len(x_valid)) * 0.8
                            t_valid = np.linspace(0.0, 1.0, num=len(x_valid))
                    else:
                        opacity = np.ones(len(x_valid)) * 0.8
                        t_valid = np.linspace(0.0, 1.0, num=len(x_valid))
                        

                    n_points: int = len(x_valid) ## changes after extension
                    _curr_render_data_dict.update(t_valid=t_valid, x_valid=x_valid, y_valid=y_valid, opacity=opacity, n_points=n_points)


                    colors = np.ones((n_points, 4), dtype=np.float32)
                    if custom_cmap is None:
                        colors[:, 0], colors[:, 1], colors[:, 2] = base_rgb[0], base_rgb[1], base_rgb[2]
                        colors[:, 3] = np.clip(opacity, 0.0, 1.0)
                    else:
                        ## have a valid colormap
                        assert t_valid is not None
                        t_rel_valid = deepcopy(t_valid) - t_valid[0]
                        t_rel_valid_span = np.ptp(t_rel_valid)
                        if np.isfinite(t_rel_valid_span) and (t_rel_valid_span > 0.0):
                            t_rel_valid = t_rel_valid / t_rel_valid_span ## scale between 0.0 and 1.0
                        else:
                            t_rel_valid = np.zeros_like(t_rel_valid, dtype=np.float64)
                        _curr_render_data_dict.update(t_rel_valid=t_rel_valid)
                        if enable_debug_logging:
                            print(f'\tt_rel_valid: {t_rel_valid}')
                        vertex_colors = np.array(custom_cmap.map(t_rel_valid), dtype=np.float32) # (n_points, 4)
                        if enable_debug_logging:
                            print(f'\tvertex_colors: {vertex_colors}')
                        Assert.same_shape(vertex_colors, colors)
                        colors[:, :3] = vertex_colors[:, :3]
                        colors[:, 3] = vertex_colors[:, 3]
                        ## overwrite with opacity values
                        colors[:, 3] = np.clip(opacity, 0.0, 1.0)
                    ## OUTPUTS: colors

                    _curr_render_data_dict.update(colors=colors) #, t_valid=t_valid, x_valid=x_valid, y_valid=y_valid, opacity=opacity, n_points=n_points)
                    
                    render_data_dict_list.append(_curr_render_data_dict)
                    # ==================================================================================================================================================================================================================================================================================== #
                    # Build the visuals to render                                                                                                                                                                                                                                                          #
                    # ==================================================================================================================================================================================================================================================================================== #
                    ## INPUTS: colors, x_valid, y_valid
                    trajectory_pos = np.ascontiguousarray(np.column_stack([x_valid, y_valid]), dtype=np.float32)
                    trajectory_colors = np.ascontiguousarray(colors, dtype=np.float32)
                    if (trajectory_pos.shape[0] < 2) or (trajectory_colors.shape[0] != trajectory_pos.shape[0]) or (not np.all(np.isfinite(trajectory_pos))) or (not np.all(np.isfinite(trajectory_colors))):
                        if self.enable_line_render_debug_logging or self.enable_full_vispy_debug_mode:
                            print(f'[VISPY_LINE_DEBUG] context=trajectory_skipped | reason=invalid_line_buffers | pos_shape={trajectory_pos.shape} | color_shape={trajectory_colors.shape} | epoch_id={epoch_id} | new_epoch_idx={new_epoch_idx}')
                        continue
                    line: vz.Line = vz.Line(pos=trajectory_pos, color=trajectory_colors, width=2, method='gl', parent=view.scene)
                    line.order = 1
                    # line.set_gl_state(blend=True, blend_func=('src_alpha', 'one'))
                    # line.push_gl_state('additive', depth_test=True)
                    # line.push_gl_state('translucent', depth_test=True)
                    line.set_gl_state('translucent', depth_test=True)
                    self._debug_log_line_visual(line, context='trajectory_line', pos=trajectory_pos, colors=trajectory_colors, extra={'epoch_id': epoch_id, 'new_epoch_idx': new_epoch_idx, 'n_points': n_points})
                    
                    lines_list.append(line)
                else:
                    if self.enable_line_render_debug_logging or self.enable_full_vispy_debug_mode:
                        print(f'[VISPY_LINE_DEBUG] context=trajectory_skipped | reason=insufficient_valid_xy | valid_count={int(np.count_nonzero(valid_mask))} | total_count={len(valid_mask)} | epoch_id={epoch_id} | new_epoch_idx={new_epoch_idx}')
                    
                    if self.enable_debug_plot_trajectory_average_angle_arrows and 'segment_Vp_deg' in positions_df.columns:
                        segment_angles = positions_df['segment_Vp_deg'].values
                        valid_angles = segment_angles[np.isfinite(segment_angles)]
                        if len(valid_angles) > 0:
                            mean_angle_deg = np.degrees(np.arctan2(np.mean(np.sin(np.radians(valid_angles))), np.mean(np.cos(np.radians(valid_angles)))))
                            mean_angle_rad = np.radians(mean_angle_deg)
                            center_idx = len(x_valid) // 2
                            x_center, y_center = x_valid[center_idx], y_valid[center_idx]
                            data_scale = max(x_max - x_min, y_max - y_min)
                            arrow_head_size = data_scale * 0.04
                            arrow_length = arrow_head_size * 0.5
                            x_end = x_center + arrow_length * np.cos(mean_angle_rad)
                            y_end = y_center + arrow_length * np.sin(mean_angle_rad)
                            debug_arrow = vz.Arrow(pos=np.array([[x_center, y_center], [x_end, y_end]]), arrows=np.array([[x_center, y_center, x_end, y_end]]), arrow_type='triangle_30', arrow_size=arrow_head_size, color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9), arrow_color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9), width=2.0, method='agg', parent=view.scene)
                            debug_arrow.order = 5
                            trajectory_debug_arrows.append(debug_arrow)
        ## END for epoch_id, positions_df in list(positions_dict....
        
        return lines_list, trajectory_debug_arrows, render_data_dict_list


    @function_attributes(short_name=None, tags=['selection', 'highlight', 'clear'], input_requires=[], output_provides=[], uses=[], used_by=['_apply_trajectory_highlight_for_selected_row'], creation_date='2026-02-03 04:42', related_items=[])
    def _clear_trajectory_highlight(self, has_valid_selection: bool=False) -> None:
        """Reset all trajectory lines to default width; no-op on exception per line."""
        if has_valid_selection:
            line_data_kwargs = dict(width=0.1)
            tick_data_kwargs = dict(width=0.05)
        else:
            ## no selection, default display
            line_data_kwargs = dict(width=1.0)
            tick_data_kwargs = dict(width=1.0)
            
        for line in (self.past_lines or []) + (self.future_lines or []):
            if line is not None:
                # ## try restore colors data:
                # _backup_colors_data = getattr(line, '_backup_colors_data', None)
                # if (_backup_colors_data is not None):
                #     ## resture the colors
                #     line_data_kwargs['color'] = _backup_colors_data
                #     ## clear the backup
                try:
                    line.set_data(**line_data_kwargs)
                    line.pop_gl_state()
                    if line.order != 1:
                        line.order = 1 ## restore default order                
                    # if (_backup_colors_data is not None):
                    #     ## clear the backup
                    #     delattr(line, '_backup_colors_data')
                except Exception:
                    pass
                
        if self.timeline_ticks is not None:            
            for tick in (self.timeline_ticks or []):
                if tick is not None:
                    try:
                        tick.set_data(**tick_data_kwargs)
                        tick.pop_gl_state()
                        if tick.order != 1:
                            tick.order = 1 ## restore default order
                    except Exception:
                        pass
            ## end for tick in ...
            


    @function_attributes(short_name=None, tags=['selection', 'highlight'], input_requires=[], output_provides=[], uses=['_clear_trajectory_highlight'], used_by=[], creation_date='2026-02-03 04:42', related_items=[])
    def _apply_trajectory_highlight_for_selected_row(self, use_raw_idx: bool=True) -> None:
        """Highlight the trajectory line corresponding to the currently selected curr_merged_segment_epochs table row; clear highlight if no valid selection. Fails gracefully."""
        if not getattr(self, 'enable_table_widgets', False) or self.epoch_table_manager is None:
            return
        if self._last_trajectory_epoch_data is None or 'curr_matching_past_future_positions_df_dict' not in self._last_trajectory_epoch_data:
            return
        try:
            table, dDisplayItem, model = self.epoch_table_manager.find_table('curr_merged_segment_epochs')
            # table, dDisplayItem, model = self.epoch_table_manager.find_table('curr_merged_pos_epochs')
        except Exception:
            self._clear_trajectory_highlight()
            return
        # from qtpy import QtCore
        selected = table.selectionModel().selectedIndexes()
        if not selected:
            self._clear_trajectory_highlight()
            return
        row = selected[0].row()
        has_valid_selection: bool = not (row < 0 or ((model.rowCount() is not None) and (row >= model.rowCount())))
        if not has_valid_selection:
            self._clear_trajectory_highlight()
            return
        ## otherwise we have has_valid_selection
        print(f'has_valid_selection: {has_valid_selection}')
        
        ## get the 'label' and 'category' columns from the dataframe/model for the currently selected row(s)
        label_col, is_future_col = None, None
        for col in range(model.columnCount()):
            h = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
            if h is not None and str(h).strip() == 'label':
                label_col = col
            if h is not None and str(h).strip() == 'is_future_present_past':
                is_future_col = col
        if label_col is None or is_future_col is None:
            self._clear_trajectory_highlight()
            return
        # Use either the label index or the raw idx:
        if use_raw_idx:
            label_val = row # use raw index
        else:
            label_val = model.index(row, label_col).data()

        category_val = model.index(row, is_future_col).data()
        if label_val is None:
            self._clear_trajectory_highlight()
            return
        category_str = str(category_val).strip().lower() if category_val is not None else ''
        category: Optional[str] = None
        if 'future' in category_str:
            category = 'future'
        elif 'past' in category_str:
            category = 'past'
        else:
            self._clear_trajectory_highlight()
            return
        positions_dict = self._last_trajectory_epoch_data['curr_matching_past_future_positions_df_dict'].get(category)
        if positions_dict is None:
            self._clear_trajectory_highlight()
            return
        

        if use_raw_idx:
            label_val = row # use raw index
            label_idx = row
        else:    
            ordered_labels = list(positions_dict.keys()) # [2, 11, 14, 17, 21]
            try:
                label_idx = ordered_labels.index(label_val)
            except (ValueError, TypeError):
                self._clear_trajectory_highlight()
                return
            
        ## OUTPUTS: label_idx
            
        lines_list = self.past_lines if category == 'past' else self.future_lines
        has_valid_label_index: bool = not ((label_idx < 0) or (label_idx >= len(lines_list)))
        if not has_valid_label_index:
            self._clear_trajectory_highlight()
            return
        
        # _backup_colors_data = self.render_data_dict_list_dict.get(category)[label_idx].get('colors', None)
        # if _backup_colors_data is not None:        
        #     self._last_trajectory_epoch_data['_backup_colors'] = _backup_colors_data

        _selected_line_kwargs = dict(width=5,
                                    #  color='#FFFFFFFF',
                                      )
        
        _selected_tick_kwargs = dict(width=4,
                                      )
        
        print(f'\tabout to highlight with label_idx: {label_idx}')
        self._clear_trajectory_highlight(has_valid_selection=has_valid_selection)
        
        try:
            line: vz.Line = lines_list[label_idx]
            if line is not None:
                # if getattr(line, '_backup_colors_data', None) is None:             
                #     ## don't allow overwrite
                #     ## get existing colors
                #     # if _backup_colors_data is None:
                #     _backup_colors_data = deepcopy(line.colors) # copy from line            
                #     ## set the backup data to the new value
                #     setattr(line, '_backup_colors_data', deepcopy(_backup_colors_data))

                # line._backup_colors_data = deepcopy(_backup_colors_data)
                # _backup_colors_data = getattr(line, '_backup_colors_data', None)
                # if _backup_colors_data is not None:
                #     ## resture the colors
                ## set/replace the data
                line.set_data(**_selected_line_kwargs)
                # line.push_gl_state('additive', depth_test=True)
                line.push_gl_state('opaque')
                line.order = 6
                
        except Exception:
            pass

        #TODO 2026-02-03 19:06: - [ ] try to update the ticks on the timeline to show the curr selection
        if self.timeline_ticks is not None:
            try:
                tick: vz.Line = self.timeline_ticks[label_idx]
                if tick is not None:
                    ## set/replace the data
                    tick.set_data(**_selected_tick_kwargs)
                    # tick.push_gl_state(blend=True, blend_func=('src_alpha', 'one'))
                    # tick.push_gl_state('additive', depth_test=True)
                    tick.push_gl_state('opaque')
                    # self.set_gl_state('additive', cull_face=False)              
                    tick.order = 6
                    
            except Exception:
                pass
        

    # ==================================================================================================================================================================================================================================================================================== #
    # Main Update Function                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    def update_epoch_display(self, new_epoch_idx: int):
        """Update the display to show a different epoch."""
        if (new_epoch_idx < 0) or (new_epoch_idx >= self.num_epochs):
            return

        self.current_epoch_idx = new_epoch_idx
        self.epoch_slider.blockSignals(True)
        self.epoch_slider.setValue(new_epoch_idx)
        # self.epoch_slider.blockSignals(False)
        self.epoch_value_label.setText(f"{new_epoch_idx}/{self.num_epochs}")
        self._clear_epoch_visuals() ## clear existing
        
        ## Get the epoch data (this performs the filtering by `minimum_included_matching_sequence_length` if set, etc
        epoch_data = self.a_flat_matching_results_list_ds._prepare_epoch_data(an_epoch_idx=new_epoch_idx, minimum_included_matching_sequence_length=self.minimum_included_matching_sequence_length)
        try:
            self._last_trajectory_epoch_data = {'curr_matching_past_future_positions_df_dict': epoch_data.get('curr_matching_past_future_positions_df_dict')} if epoch_data else None
        except Exception:
            self._last_trajectory_epoch_data = None
        filter_epochs = self.a_flat_matching_results_list_ds.filter_epochs        
        if new_epoch_idx < len(filter_epochs):
            epoch_row = filter_epochs.iloc[new_epoch_idx]
            epoch_start_t = epoch_row['start'] if 'start' in epoch_row else epoch_row.get('t_start', None)
            epoch_end_t = epoch_row['stop'] if 'stop' in epoch_row else epoch_row.get('t_stop', None)
        else:
            epoch_start_t = None
            epoch_end_t = None

        ## get standard data:            
        x_min, x_max = self.xbin[0], self.xbin[-1]
        y_min, y_max = self.ybin[0], self.ybin[-1]
        
        # Get posterior data
        p_x_given_n = self.a_flat_matching_results_list_ds.p_x_given_n_list[new_epoch_idx]
        p_x_given_n = np.ascontiguousarray(p_x_given_n, dtype=np.float32)
        
        posterior_2d = np.sum(p_x_given_n, axis=2)
        posterior_2d = np.ascontiguousarray(posterior_2d, dtype=np.float32)

        # Generate time bin colors for use in trajectory and centroid coloring
        n_time_bins: int = p_x_given_n.shape[2]
        time_bin_colors = self._time_bin_colors(n_time_bins, alpha=0.9)


        # ==================================================================================================================================================================================================================================================================================== #
        # Common PAST/Future Properties                                                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        ## INPUTS: epoch_data (filtered data)
        curr_matching_past_future_positions_df_dict = {k: v for k, v in epoch_data['curr_matching_past_future_positions_df_dict'].items()}
        curr_matching_good_merged_segment_epochs_df = epoch_data.get('curr_matching_good_merged_segment_epochs_df', None)
        if curr_matching_good_merged_segment_epochs_df is not None:
            ## use this 
            curr_matching_good_merged_segment_epochs_df = curr_matching_good_merged_segment_epochs_df.reset_index(drop=True, inplace=False)
            num_good_epochs: int = len(curr_matching_good_merged_segment_epochs_df)
            print(f'curr_matching_good_merged_segment_epochs_df - num_good_epochs: {num_good_epochs}')

            num_good_epochs_past_future = {k:len(v) for k, v in curr_matching_past_future_positions_df_dict.items()}
            num_total_good_epochs_past_future: int = np.sum(list(num_good_epochs_past_future.values()))
            print(f'\tnum_good_epochs_past_future: {num_good_epochs_past_future},\n\tnum_total_good_epochs_past_future: {num_total_good_epochs_past_future}')            
            assert num_good_epochs == num_total_good_epochs_past_future, f'num_total_good_epochs_past_future: {num_total_good_epochs_past_future} != num_good_epochs: {num_good_epochs}'


        all_time_distances = []
        if 'past' in curr_matching_past_future_positions_df_dict and epoch_start_t is not None:
            for epoch_id, positions_df in curr_matching_past_future_positions_df_dict['past'].items():
                if len(positions_df) > 0 and 't' in positions_df.columns:
                    t_coords = np.asarray(positions_df['t'].to_numpy(), dtype=np.float64)
                    valid_mask = np.isfinite(t_coords)
                    if np.any(valid_mask):
                        time_rel = t_coords[valid_mask] - epoch_start_t
                        all_time_distances.extend(np.abs(time_rel).tolist())
        if 'future' in curr_matching_past_future_positions_df_dict and epoch_end_t is not None:
            for epoch_id, positions_df in curr_matching_past_future_positions_df_dict['future'].items():
                if len(positions_df) > 0 and 't' in positions_df.columns:
                    t_coords = np.asarray(positions_df['t'].to_numpy(), dtype=np.float64)
                    valid_mask = np.isfinite(t_coords)
                    if np.any(valid_mask):
                        time_rel = t_coords[valid_mask] - epoch_end_t
                        all_time_distances.extend(np.abs(time_rel).tolist())
                        

        ## updates: self.colorbar_rects, self.colorbar_texts, self.colorbar_view
        max_time_distance = max(all_time_distances) if all_time_distances else 1.0
        print(f'max_time_distance: {max_time_distance}')
        if max_time_distance > 0:
            colorbar_width, colorbar_height, num_segments = 800, 40, 200
            time_range = np.linspace(-max_time_distance, max_time_distance, num_segments)
            segment_width = colorbar_width / num_segments
            for i, time_val in enumerate(time_range):
                time_distance = np.abs(time_val)
                distance_normalized = time_distance / max_time_distance
                opacity = np.clip(1.0 - distance_normalized * 0.8, 0.2, 1.0)
                hue = 0.0 if time_val < 0 else 0.5
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = (rgb[0], rgb[1], rgb[2], opacity)
                x_pos = i * segment_width
                rect = vz.Rectangle(center=(x_pos + segment_width/2, colorbar_height/2), width=segment_width, height=colorbar_height, color=color, parent=self.colorbar_view.scene)
                self.colorbar_rects.append(rect)
            label_times = [-max_time_distance, -max_time_distance/2, 0, max_time_distance/2, max_time_distance]
            label_positions = np.linspace(0, colorbar_width, len(label_times))
            for time_val, x_pos in zip(label_times, label_positions):
                text = vz.Text(f'{time_val:.2f}s', pos=(x_pos, colorbar_height + 10), color='white', font_size=10, parent=self.colorbar_view.scene)
                self.colorbar_texts.append(text)
            title_past = vz.Text('Past (time from start)', pos=(colorbar_width/4, -20), color='white', font_size=12, parent=self.colorbar_view.scene)
            title_future = vz.Text('Future (time from end)', pos=(3*colorbar_width/4, -20), color='white', font_size=12, parent=self.colorbar_view.scene)
            title_opacity = vz.Text('Opacity: 1.0 (close) → 0.2 (distant)', pos=(colorbar_width/2, colorbar_height + 25), color='white', font_size=11, parent=self.colorbar_view.scene)
            self.colorbar_texts.extend([title_past, title_future, title_opacity])
            self.colorbar_view.camera = scene.PanZoomCamera(aspect=1)
            self.colorbar_view.camera.set_range(x=(-50, colorbar_width + 50), y=(-50, colorbar_height + 50))
            

        if self.show_full_position_background and self.curr_position_df is not None and 'x' in self.curr_position_df.columns and 'y' in self.curr_position_df.columns:
            bg_x = np.asarray(self.curr_position_df['x'].to_numpy(), dtype=np.float64)
            bg_y = np.asarray(self.curr_position_df['y'].to_numpy(), dtype=np.float64)
            bg_valid_mask = np.isfinite(bg_x) & np.isfinite(bg_y)
            if np.count_nonzero(bg_valid_mask) >= 2:
                bg_x_valid, bg_y_valid = bg_x[bg_valid_mask], bg_y[bg_valid_mask]
                n_bg_points = len(bg_x_valid)
                bg_colors = np.ones((n_bg_points, 4), dtype=np.float32)
                bg_colors[:, :3] = 0.5
                bg_colors[:, 3] = 0.2
                for bg_line in self.full_position_background_line:
                    if bg_line is not None:
                        bg_line.parent = None
                self.full_position_background_line.clear()
                bg_pos = np.column_stack([bg_x_valid, bg_y_valid])
                for view in [self.past_view, self.posterior_2d_view, self.future_view]:
                    line = vz.Line(pos=bg_pos, color=bg_colors, width=1, method='gl', parent=view.scene)
                    line.order = 0
                    self._debug_log_line_visual(line, context='full_position_background_line', pos=bg_pos, colors=bg_colors, extra={'new_epoch_idx': new_epoch_idx})
                    self.full_position_background_line.append(line)
                    



        _common_past_future_render_trajectory_side_kwargs = dict(max_time_distance=max_time_distance, time_bin_colors=time_bin_colors, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, new_epoch_idx=new_epoch_idx)
        ## common outputs: _common_past_future_render_trajectory_side_kwargs


        # ==================================================================================================================================================================================================================================================================================== #
        # LEFT PANE: PAST                                                                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #

        # Render Past Trajectories and collect data for timeline
        past_trajectory_colors_and_times = []
        if 'past' in curr_matching_past_future_positions_df_dict:
            curr_past_future_key_name: types.PastFutureCategory = 'past'
            self.render_data_dict_list_dict[curr_past_future_key_name] = [] ## clear manually
            self.past_lines, self.trajectory_debug_arrows[curr_past_future_key_name], self.render_data_dict_list_dict[curr_past_future_key_name] = self._render_trajectory_side(positions_dict=curr_matching_past_future_positions_df_dict[curr_past_future_key_name], epoch_anchor_t=epoch_start_t, default_hue=0.0, view=self.past_view, trajectory_colors_and_times_out=past_trajectory_colors_and_times, **_common_past_future_render_trajectory_side_kwargs,
                                                               lines_list=self.past_lines, trajectory_debug_arrows=self.trajectory_debug_arrows.get(curr_past_future_key_name, []), render_data_dict_list=self.render_data_dict_list_dict.get(curr_past_future_key_name, []),
                                                           )


        # ==================================================================================================================================================================================================================================================================================== #
        # CENTER PANE: CURRENT PBE                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        _update_dict = self._render_central_view(p_x_given_n=p_x_given_n, posterior_2d=posterior_2d,
                                  epoch_start_t=epoch_start_t, epoch_end_t=epoch_end_t,
                                  **_common_past_future_render_trajectory_side_kwargs,
                                  _update_dict = dict(
                                        centroid_dots=self.centroid_dots, centroid_arrows=self.centroid_arrows,
                                        current_position_line=self.current_position_line, trajectory_arrows=self.trajectory_arrows, epoch_info_text=self.epoch_info_text,
                                        time_bin_views=self.time_bin_views, time_bin_labels=self.time_bin_labels, time_bin_images=self.time_bin_images, time_bin_raster=self.time_bin_raster,
                                        past_mask_contours=self.past_mask_contours, posterior_mask_contours=self.posterior_mask_contours, future_mask_contours=self.future_mask_contours,    
                                    ),  
        )
        for _k, _v in _update_dict.items():
            setattr(self, _k, _v)



        # ==================================================================================================================================================================================================================================================================================== #
        # RIGHT PANE: FUTURE                                                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        # Render Future Trajectories and collect data for timeline
        future_trajectory_colors_and_times = []
        if 'future' in curr_matching_past_future_positions_df_dict:
            curr_past_future_key_name: types.PastFutureCategory = 'future'
            self.render_data_dict_list_dict[curr_past_future_key_name] = [] ## clear manually
            self.future_lines, self.trajectory_debug_arrows[curr_past_future_key_name], self.render_data_dict_list_dict[curr_past_future_key_name] = self._render_trajectory_side(positions_dict=curr_matching_past_future_positions_df_dict[curr_past_future_key_name], epoch_anchor_t=epoch_end_t, default_hue=0.5, view=self.future_view, trajectory_colors_and_times_out=future_trajectory_colors_and_times, **_common_past_future_render_trajectory_side_kwargs,
                                                                lines_list=self.future_lines, trajectory_debug_arrows=self.trajectory_debug_arrows.get(curr_past_future_key_name, []), render_data_dict_list=self.render_data_dict_list_dict.get(curr_past_future_key_name, []),
                                                             )
            

        # ==================================================================================================================================================================================================================================================================================== #
        # Bottom/Common Panes                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        
        # Render Combined Timeline Bar (full width, shows all trajectory ticks and current epoch)
        timeline_bar_height = 1.0
        recording_duration = self.recording_t_max - self.recording_t_min
        if recording_duration > 0:
            bar_fill: vz.Rectangle = vz.Rectangle(center=((self.recording_t_min + self.recording_t_max) / 2, timeline_bar_height / 2), width=recording_duration, height=timeline_bar_height, color=(0.15, 0.15, 0.15, 1.0), border_color=(0.4, 0.4, 0.4, 1.0), parent=self.combined_timeline_view.scene)
            self.timeline_bar = bar_fill
            if epoch_start_t is not None and epoch_end_t is not None:
                epoch_duration = epoch_end_t - epoch_start_t
                epoch_center_t = (epoch_start_t + epoch_end_t) / 2
                epoch_rect: vz.Rectangle = vz.Rectangle(center=(epoch_center_t, timeline_bar_height / 2), width=epoch_duration, height=timeline_bar_height, color=(1.0, 1.0, 1.0, 0.3), border_color=(1.0, 1.0, 1.0, 1.0), border_width=2, parent=self.combined_timeline_view.scene)
                self.timeline_epoch_rect = epoch_rect
                triangle_height = timeline_bar_height * 0.35
                triangle_half_width = recording_duration * 0.008
                triangle_top_y = timeline_bar_height + triangle_height * 0.3
                triangle_bottom_y = timeline_bar_height - triangle_height * 0.3
                triangle_vertices = np.array([[epoch_center_t - triangle_half_width, triangle_top_y], [epoch_center_t + triangle_half_width, triangle_top_y], [epoch_center_t, triangle_bottom_y]], dtype=np.float32)
                epoch_triangle: vz.Polygon = vz.Polygon(pos=np.asarray(triangle_vertices, dtype=np.float32), color=(1.0, 1.0, 1.0, 0.5), border_color=(1.0, 1.0, 1.0, 1.0), border_width=1, parent=self.combined_timeline_view.scene)
                self.timeline_epoch_triangle = epoch_triangle
            for base_rgb, mean_time in (past_trajectory_colors_and_times + future_trajectory_colors_and_times):
                if not np.isfinite(mean_time):
                    continue
                tick_pos = np.array([[mean_time, 0], [mean_time, timeline_bar_height]], dtype=np.float32)
                tick: vz.Line = vz.Line(pos=tick_pos, color=(base_rgb[0], base_rgb[1], base_rgb[2], 1.0), width=1.0, method='agg', parent=self.combined_timeline_view.scene)
                self._debug_log_line_visual(tick, context='timeline_tick', pos=tick_pos, extra={'new_epoch_idx': new_epoch_idx, 'mean_time': float(mean_time)})
                self.timeline_ticks.append(tick)
            self.combined_timeline_view.camera = scene.PanZoomCamera()
            self.combined_timeline_view.camera.set_range(x=(self.recording_t_min, self.recording_t_max), y=(0, timeline_bar_height))
            


        self.canvas.title = f'Predictive Decoding Display - Vispy (Epoch {new_epoch_idx + 1}/{self.num_epochs})'
        # self.canvas.update()
        # QApplication.processEvents()
        

            # 'curr_matching_epochs_df': curr_matching_epochs_df,
            # 'curr_matching_positions_df': curr_matching_positions_df,
            # 'curr_matching_epochs_df_dict': curr_matching_epochs_df_dict,
            # 'curr_matching_merged_segment_epochs_df_dict': curr_matching_merged_segment_epochs_df_dict, 
            # 'curr_matching_past_future_positions_df_dict': curr_matching_past_future_positions_df_dict,
            # 'curr_matching_past_future_positions_df_list': curr_matching_past_future_positions_df_list,
        
        if self.enable_table_widgets:
            def _perform_async_deferred_update_table_widgets():
                """ captures: self, new_epoch_idx, epoch_data 
                """
                self.perform_update_table_widgets(new_epoch_idx=new_epoch_idx, epoch_data=epoch_data)
                

            # END def _perform_async_de...
            print(F'scheduling `_perform_async_deferred_update_table_widgets(...)` on timer')
            QtCore.QTimer.singleShot(0, _perform_async_deferred_update_table_widgets)


        # self.canvas.title = f'Predictive Decoding Display - Vispy (Epoch {new_epoch_idx + 1}/{self.num_epochs})'
        self.canvas.update()
        # QApplication.processEvents()

        ## unblock the epoch_slider       
        self.epoch_slider.blockSignals(False)


    def perform_update_table_widgets(self, new_epoch_idx: Optional[int]=None, epoch_data: Optional[Dict]=None):
        """ 
        """
        if new_epoch_idx is None:
            new_epoch_idx = self.current_epoch_idx

        if epoch_data is None:    
            ## Get the epoch data (this performs the filtering by `minimum_included_matching_sequence_length` if set, etc
            epoch_data = self.a_flat_matching_results_list_ds._prepare_epoch_data(an_epoch_idx=self.current_epoch_idx, minimum_included_matching_sequence_length=self.minimum_included_matching_sequence_length)
        assert epoch_data is not None

        if (self.epoch_table_manager is not None) and (epoch_data is not None):
            # QApplication.processEvents()
            try:
                print(f'trying to update self.epoch_table_manager tables for new_epoch_idx: {new_epoch_idx}...')
                table_update_sources = {}                    
                curr_matching_epochs_df = epoch_data.get('curr_matching_epochs_df', None)
                curr_matching_good_merged_segment_epochs_df = epoch_data.get('curr_matching_good_merged_segment_epochs_df', None)
            
                if (curr_matching_epochs_df is None):
                    print(f'\tERROR: new_epoch_idx: {new_epoch_idx} curr_matching_epochs_df is None')
                else:
                    a_matching_pos_merged_segment_epochs_df: pd.DataFrame = curr_matching_epochs_df # self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list[new_epoch_idx]
                    if (a_matching_pos_merged_segment_epochs_df is not None) and (len(a_matching_pos_merged_segment_epochs_df) > 0):
                        # table_update_sources['curr_merged_segment_epochs'] = a_matching_pos_merged_segment_epochs_df
                        table_update_sources['curr_merged_pos_epochs'] = a_matching_pos_merged_segment_epochs_df
                        

                if (curr_matching_good_merged_segment_epochs_df is None):
                    print(f'\tERROR: new_epoch_idx: {new_epoch_idx} curr_matching_good_merged_segment_epochs_df is None')
                else:
                    a_matching_pos_epochs_df: pd.DataFrame = curr_matching_good_merged_segment_epochs_df # self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list[new_epoch_idx]
                    if (a_matching_pos_epochs_df is not None) and len(a_matching_pos_epochs_df) > 0:
                        # table_update_sources['curr_merged_pos_epochs'] = a_matching_pos_epochs_df
                        table_update_sources['curr_merged_segment_epochs'] = a_matching_pos_epochs_df
                        
                if table_update_sources:
                    visible_columns_dict = {
                        'curr_merged_segment_epochs': ['start', 'stop', 'is_future_present_past', 'epoch_t_idx', 'label', 'duration', 'num_epoch_t_bins', 'is_reversely_replayed', 'pre_merged_epoch_label'],
                        'curr_merged_pos_epochs': ['start', 'stop', 'is_future_present_past', 'label', 'duration'],
                    }
                    print(f'\tperforming update_tables: len(table_update_sources): {len(table_update_sources)}, table_update_sources.keys(): {list(table_update_sources.keys())}, visible_columns_dict: {visible_columns_dict}')
                    self.epoch_table_manager.update_tables(table_update_sources, visible_columns_dict=visible_columns_dict)
                    try:
                        table, dDisplayItem, model = self.epoch_table_manager.find_table('curr_merged_segment_epochs')
                        # table, dDisplayItem, model = self.epoch_table_manager.find_table('curr_merged_pos_epochs')
                        try:
                            table.selectionModel().selectionChanged.disconnect(self._apply_trajectory_highlight_for_selected_row)
                        except Exception:
                            pass
                        table.selectionModel().selectionChanged.connect(self._apply_trajectory_highlight_for_selected_row)
                        self._apply_trajectory_highlight_for_selected_row()
                    except Exception:
                        pass
                else:
                    print(f'\tWARN: no table_update_sources (empty)')

            except Exception as e:
                print(f'\tERROR: encountered exception {e} while trying to update table widgets for new_epoch_idx: {new_epoch_idx}!')
                # raise e
                pass

    # ==================================================================================================================================================================================================================================================================================== #
    # UI Events                                                                                                                                                                                                                                                                            #
    # ==================================================================================================================================================================================================================================================================================== #
    
    def on_slider_value_changed(self, value):
        print(f'on_slider_value_changed(value: {value})')
        self.epoch_value_label.setText(f"{value}/{self.num_epochs}")

    def on_slider_released(self):
        print(f'on_slider_released()')
        self.update_epoch_display(self.epoch_slider.value())


    def on_key_press(self, event):
        print(f'on_key_press(value: {event})')
        proposed_new_epoch: int = self.current_epoch_idx - 1
        if (proposed_new_epoch < 0) or (proposed_new_epoch > (self.num_epochs-1)):
            print('invalid index would be selected be key press. Skipping.')
            return
        else:                
            if event.key == 'Left':
                self.update_epoch_display(self.current_epoch_idx - 1)
            elif event.key == 'Right':
                self.update_epoch_display(self.current_epoch_idx + 1)



    # EXPORT TO IMAGES ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    @function_attributes(short_name=None, tags=['vispy', 'export', 'screenshot', 'high-resolution'], input_requires=[], output_provides=[], uses=['render_predictive_decoding_with_vispy'], used_by=[], creation_date='2026-01-22', related_items=['render_predictive_decoding_with_vispy'])
    def export_vispy_viewer_epochs(self, export_folder: Union[str, Path], export_individual_views: bool = False, epoch_indices: Optional[List[int]] = None,
                                delay_between_epochs: float = 0.15, progress_print: bool = True) -> List[Path]:
        """Export high-resolution renderings of all epoch views from the vispy predictive decoding viewer.
        
        Programmatically iterates through epoch indices, updates the display, and exports high-resolution 
        screenshots of all displayed views to an export folder.
        
        Args:
            viewer_tuple: Tuple (main_window, canvas, state) from render_predictive_decoding_with_vispy, or a PredictiveDecodingVispyWidget instance.
            export_folder: Path to folder where images will be saved. Created if it doesn't exist.
            resolution_scale: Scale factor for high-res rendering (default: 2.0). Higher values produce larger images.
            export_individual_views: If True, export individual views (past, posterior, future) separately (default: False)
            epoch_indices: Optional list of specific epoch indices to export. If None, exports all epochs.
            delay_between_epochs: Delay in seconds after updating epoch display to allow rendering to stabilize (default: 0.15)
            progress_print: If True, print progress messages during export (default: True)
            
        Returns:
            List of Path objects for all exported image files
            
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import render_predictive_decoding_with_vispy, export_vispy_viewer_epochs
            
            viewer = render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result, 
                                                            a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
                                                            curr_position_df=container.decoding_locality.pos_df, 
                                                            pf_decoder=a_decoder, decoded_result=a_decoded_result)
            
            exported_files = export_vispy_viewer_epochs(viewer, export_folder='./exports')
        """
        import time
        from pathlib import Path
        from qtpy.QtWidgets import QApplication
        
        # Try to import imageio for saving, fall back to PIL if not available
        try:
            from imageio import imwrite as save_image
            _use_imageio = True
        except ImportError:
            from PIL import Image
            _use_imageio = False
            def save_image(path, img_array):
                """Save image array using PIL."""
                Image.fromarray(img_array).save(path)
        

        resolution_scale: float = 1.0 ## doesn't work at all

        # Accept either (main_window, canvas, state) tuple or PredictiveDecodingVispyWidget instance
        # if hasattr(widget_container, 'as_viewer_tuple'):
        #     widget_container = widget_container.as_viewer_tuple()

        # def as_viewer_tuple(self) -> tuple:
        #     return (self.main_window, self.canvas, self.get_state())
        
        main_window = self.main_window
        canvas = self.canvas
        state = {
                'num_epochs': self.num_epochs,
                'epoch_slider': self.epoch_slider,
                'epoch_value_label': self.epoch_value_label,
                'update_epoch_display': self.update_epoch_display,
            }
        
        
        # Validate and create export folder
        export_folder = Path(export_folder)
        export_folder.mkdir(parents=True, exist_ok=True)
        
        # Get number of epochs and determine which indices to export
        num_epochs = state['num_epochs']
        if epoch_indices is None:
            epoch_indices = list(range(num_epochs))
        else:
            # Validate provided epoch indices
            epoch_indices = [idx for idx in epoch_indices if 0 <= idx < num_epochs]
        
        if len(epoch_indices) == 0:
            print("Warning: No valid epoch indices to export.")
            return []
        
        # Get the update function from state
        update_epoch_display = state.get('update_epoch_display')
        if update_epoch_display is None:
            raise ValueError("update_epoch_display function not found in state. Ensure you're using an updated version of render_predictive_decoding_with_vispy.")
        
        # Get canvas size for high-res rendering
        canvas_width, canvas_height = canvas.size
        high_res_width = int(canvas_width * resolution_scale)
        high_res_height = int(canvas_height * resolution_scale)
        
        exported_files = []
        total_epochs = len(epoch_indices)
        
        if progress_print:
            print(f"Exporting {total_epochs} epochs to {export_folder} at {resolution_scale}x resolution ({high_res_width}x{high_res_height})...")
        
        for i, epoch_idx in enumerate(epoch_indices):
            try:
                if progress_print:
                    print(f"  Exporting epoch {epoch_idx + 1}/{num_epochs} ({i + 1}/{total_epochs})...", end='', flush=True)
                
                # Update the epoch display programmatically
                # Block slider signals to avoid recursive updates
                state['epoch_slider'].blockSignals(True)
                state['epoch_slider'].setValue(epoch_idx)
                state['epoch_slider'].blockSignals(False)
                
                # Call the update function directly
                update_epoch_display(epoch_idx)
                
                # Process Qt events to ensure rendering completes
                QApplication.processEvents()
                
                # Small delay to allow rendering to stabilize
                time.sleep(delay_between_epochs)
                
                # Process events again after delay
                # QApplication.processEvents()
                
                # Ensure canvas is updated
                canvas.update()
                # QApplication.processEvents()
                
                # Render high-resolution screenshot
                # vispy's render() returns RGBA numpy array with shape (height, width, 4)
                img_array = canvas.render(size=(high_res_width, high_res_height))
                
                # Convert RGBA to RGB by dropping alpha channel (optional, keeps file size smaller)
                # img_array = img_array[:, :, :3]
                
                # Flip vertically if needed (vispy may return origin at bottom-left)
                # Check if image appears upside down and flip
                # img_array = np.flipud(img_array) ## do not flip, otherwise it IS upside down
                
                # Save full canvas screenshot
                full_filename = f"epoch_{epoch_idx:04d}_full.png"
                full_path = export_folder / full_filename
                save_image(str(full_path), img_array)
                exported_files.append(full_path)
                
                if progress_print:
                    print(f" saved to {full_filename}")
                
                # Export individual views if requested
                if export_individual_views:
                    # Note: Individual view export requires rendering each view separately
                    # This is more complex and may require accessing view.scene directly
                    # For now, we export the full canvas; individual view export can be added later
                    pass
                    
            except Exception as e:
                if progress_print:
                    print(f" ERROR: {e}")
                # Continue with next epoch even if this one fails
                continue
        
        if progress_print:
            print(f"Export complete. {len(exported_files)} files saved to {export_folder}")
        
        return exported_files




@function_attributes(short_name=None, tags=['vispy', 'rendering', 'standalone'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21', related_items=[])
def render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], a_decoded_filter_epochs_df: pd.DataFrame, curr_position_df: pd.DataFrame, pf_decoder: BasePositionDecoder, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0,
    current_traj_seconds_pre_post_extension: float = 0.750, 
    past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = (0.4, 1.0), 
    start_end_extension_max_opacity: float = 0.4, show_full_position_background: bool = False, 
    require_angle_match: bool = False, color_matches_by_matching_angle: bool=False, enable_debug_plot_trajectory_average_angle_arrows: bool=False,
    minimum_included_matching_sequence_length: Optional[int] = None,
    **kwargs) -> PredictiveDecodingVispyWidget:
    """Standalone function that renders predictive decoding data using vispy instead of the widget.
    
    Takes the same inputs as PredictiveDecodingDisplayWidget.init_from_datasource but uses vispy
    to render all the data in an interactive visualization.
    
    Keyboard controls:
        Left Arrow: Navigate to previous epoch
        Right Arrow: Navigate to next epoch
    
    Args:
        epoch_flat_mask_future_past_result: List of MatchingPastFuturePositionsResult objects
        a_decoded_filter_epochs_df: DataFrame with filter epochs
        curr_position_df: DataFrame with current position data
        pf_decoder: BasePositionDecoder instance
        decoded_result: DecodedFilterEpochsResult instance
        active_epoch_idx: Initial epoch index to display (default: 0)
        current_traj_seconds_pre_post_extension: Seconds to extend current epoch trajectory before/after epoch bounds (default: 0.750). Positions outside epoch are rendered with 0.2 alpha.
        past_future_trajectory_extension_seconds: Seconds to extend past/future trajectories beyond their computed bounds (default: 0.0).
            Can be a single float (applies to both start and end) or a tuple (start_extension, end_extension).
            Start extensions are rendered with solid start_end_extension_max_opacity. End extensions fade out from 
            start_end_extension_max_opacity to 0.0, visually indicating trajectory direction.
        start_end_extension_max_opacity: Maximum opacity for trajectory extensions (default: 0.2). Start extensions use this as solid opacity,
            end extensions fade from this value to 0.0.
        show_full_position_background: If True, renders the entire position dataframe as a faint (0.2 alpha) grey line behind past/future trajectories (default: False).
        require_angle_match: If True, only display trajectories whose direction aligns with the decoded posterior centroid direction (centroid_pos_traj_matching_angle_idx >= 0). Default: False.
        color_matches_by_matching_angle: If True, trajectories that have a valid angle match (centroid_pos_traj_matching_angle_idx >= 0) 
            will be colored using the corresponding time bin's color instead of the default red (past) or cyan (future). Default: True.
        enable_debug_plot_trajectory_average_angle_arrows: If True, draws small arrows at the temporal center of each 
            past/future trajectory indicating the trajectory's representative direction (circular mean of segment_Vp_deg). Default: True.
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (main_window, canvas, state) for compatibility with export_vispy_viewer_epochs.

    Usage:

        from pyphoplacecellanalysis.Pho2D.vispy.predicitive_decoding_vispy import render_predictive_decoding_with_vispy, PredictiveDecodingVispyWidget

        viewer: PredictiveDecodingVispyWidget = render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result, a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
                                                        curr_position_df=container.decoding_locality.pos_df, pf_decoder=a_decoder, decoded_result=a_decoded_result)

    Implemented via PredictiveDecodingVispyWidget.init_from_list(...); returns (main_window, canvas, state) for compatibility.
    """
    widget = PredictiveDecodingVispyWidget.init_from_list(
        epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result,
        a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
        curr_position_df=curr_position_df,
        pf_decoder=pf_decoder,
        decoded_result=decoded_result,
        active_epoch_idx=active_epoch_idx,
        current_traj_seconds_pre_post_extension=current_traj_seconds_pre_post_extension,
        past_future_trajectory_extension_seconds=past_future_trajectory_extension_seconds,
        start_end_extension_max_opacity=start_end_extension_max_opacity,
        show_full_position_background=show_full_position_background,
        require_angle_match=require_angle_match,
        color_matches_by_matching_angle=color_matches_by_matching_angle,
        enable_debug_plot_trajectory_average_angle_arrows=enable_debug_plot_trajectory_average_angle_arrows,
        minimum_included_matching_sequence_length=minimum_included_matching_sequence_length,
        **kwargs)
    return widget # widget.as_viewer_tuple()



from pyphoplacecellanalysis.Pho2D.vispy.vispy_cameras import CustomTurntableCamera # Used in `Volumentric2DTimeSeriesPlotter`
from pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets import VispySceneWrappingWidget, VispyCanvasContainingWindow

# Volumetric 2D time-series plotter using vispy
_VOLUMETRIC_TURNTABLE_FOV: float = 45.0
_VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION: float = 30.0
_VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH: float = 135.0
_VOLUMETRIC_CAMERA_VIEW_PRESETS: Tuple[Tuple[str, float, float], ...] = (
    ("Top", 90.0, 0.0),
    ("Left", 0.0, -90.0),
    ("Right", 0.0, 90.0),
    ("Perspective", _VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION, _VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH),
)

# Single-slot epoch overlay: at most one epoch's meshes in the scene; swap on `set_active_epoch`.
_SINGLE_SLOT_CONTOUR_ID: str = "contour[active]"
_SINGLE_SLOT_PLANE_ID: str = "plane[active]"


@metadata_attributes(short_name=None, tags=['vispy', 'qt', '3D', 'Bapun', 'ACTIVE'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-03-18 05:41', related_items=[])
@define(slots=False, repr=False, eq=False)
class Volumentric2DTimeSeriesPlotter:
    """plots a 3D volume that represents a rat in a 2D open-field arena (x-, y- axis) over time (z-axis) 
    It renders:
    - The animal's 2D position over time as a curve
    - plotting 2D decoded position posteriors at a certain time bin as a plane
    
    It features
    - highlighting of certain time ranges - highlights the volume along the z-axis
        - can color the curve according to the range color
        - can add labels along the x=0, y=0 planes visually indicate the region in question
        
        
    Usage:
    
        from pyphoplacecellanalysis.Pho2D.vispy.predicitive_decoding_vispy import Volumentric2DTimeSeriesPlotter
    
        viewer_3d: Volumentric2DTimeSeriesPlotter = Volumentric2DTimeSeriesPlotter.init_from_position_and_decoder(curr_position_df=curr_position_df, xbin=xbin, ybin=ybin, p_x_given_n=p_x_given_n, t_bin_edges=t_bin_edges, highlight_epochs=highlight_epochs)
    """

    curr_position_df: pd.DataFrame = field(default=None)
    xbin: NDArray = field(default=None)
    ybin: NDArray = field(default=None)
    p_x_given_n: Optional[NDArray] = field(default=None)
    t_bin_edges: Optional[NDArray] = field(default=None)
    highlight_epochs: Optional[pd.DataFrame] = field(default=None)
    active_t_bin_idx: int = field(default=0)

    t_min: float = field(default=0.0)
    t_max: float = field(default=1.0)
    z_scale: float = field(default=1.0)
    pos3d: Optional[NDArray] = field(default=None)

    canvas: scene.SceneCanvas = field(default=None)
    main_window: VispyCanvasContainingWindow = field(default=None)
    view: scene.widgets.ViewBox = field(default=None)
    scene_tree_widget: VispySceneTreeWidget = field(default=None)

    position_line: vz.Line = field(default=None)
    posterior_plane: Any = field(default=None)
    decoded_posteriors_by_key: Dict[str, Dict[str, Any]] = field(default=Factory(dict))
    decoded_posterior_counter: int = field(default=0)
    highlight_boxes: List[Any] = field(default=Factory(list))
    highlight_labels: List[Any] = field(default=Factory(list))
    t_bin_slider: Optional[Any] = field(default=None)
    t_bin_value_label: Optional[Any] = field(default=None)
    arena_wireframe_lines: List[Any] = field(default=Factory(list))
    coordinate_axes_lines: List[Any] = field(default=Factory(list))
    coordinate_axes_labels: List[Any] = field(default=Factory(list))
    debug_crosshair_lines: List[Any] = field(default=Factory(list))
    debug_is_shift_hover_enabled: bool = field(default=False)
    debug_nearest_pos3d_idx: Optional[int] = field(default=None)
    debug_crosshair_snap_max_pixel_distance: float = field(default=16.0)

    posterior_contours_by_key: Dict[str, Dict[str, Any]] = field(default=Factory(dict))
    posterior_contours_counter: int = field(default=0)

    emphasis_planes_by_key: Dict[str, Dict[str, Any]] = field(default=Factory(dict))
    emphasis_planes_counter: int = field(default=0)

    contour_extrusion_visuals_by_key: Dict[str, Dict[str, Any]] = field(default=Factory(dict))

    epoch_visual_groups: Dict[int, Dict[str, List[str]]] = field(default=Factory(dict))
    epoch_group_nodes: Dict[int, Any] = field(default=Factory(dict))
    active_epoch_idx: Optional[int] = field(default=None)
    epoch_slider: Optional[QtWidgets.QSlider] = field(default=None)
    epoch_value_label: Optional[QtWidgets.QLabel] = field(default=None)

    epoch_flat_mask_future_past_result: Optional[List[MatchingPastFuturePositionsResult]] = field(default=None)
    epoch_visual_extrude: bool = field(default=False)
    epoch_visual_contour_kwargs: Optional[Dict[str, Any]] = field(default=None)
    epoch_visual_plane_kwargs: Optional[Dict[str, Any]] = field(default=None)
    epoch_visual_extrusion_kwargs: Optional[Dict[str, Any]] = field(default=None)
    _single_slot_rendered_epoch_idx: Optional[int] = field(default=None)
    epoch_single_slot_parent_node: Any = field(default=None)

    _epoch_mask_datasource: Optional[MaskDataSource] = field(default=None)
    epoch_visual_filter_epochs: Optional[pd.DataFrame] = field(default=None)
    minimum_included_matching_sequence_length: Optional[int] = field(default=None)
    show_match_lines: bool = field(default=True)
    match_line_visuals: List[Any] = field(default=Factory(list))
    _match_lines_built_for_epoch_idx: Optional[int] = field(default=None)

    reward_zone_extrusion_visuals: List[Any] = field(default=Factory(list))

    debug_xyz_axes: vz.XYZAxis = field(default=None)
    gridlines: vz.GridLines = field(default=None)

    def __attrs_post_init__(self):
        self.setup()
        self.buildUI()


    @classmethod
    def init_from_position_and_decoder(cls, curr_position_df: pd.DataFrame, xbin: NDArray, ybin: NDArray, p_x_given_n: Optional[NDArray]=None, t_bin_edges: Optional[NDArray]=None, highlight_epochs: Optional[pd.DataFrame] = None, **kwargs) -> "Volumentric2DTimeSeriesPlotter":
        return cls(curr_position_df=curr_position_df, xbin=xbin, ybin=ybin, p_x_given_n=p_x_given_n, t_bin_edges=t_bin_edges, highlight_epochs=highlight_epochs, **kwargs)


    @property
    def n_t_bins(self) -> int:
        if self.p_x_given_n is None:
            return 0
        return int(np.shape(self.p_x_given_n)[2])


    @property
    def z_max(self) -> float:
        return float((self.t_max - self.t_min) * self.z_scale)


    def setup(self):
        if self.curr_position_df is None:
            raise ValueError("curr_position_df must be provided")
        missing_cols = {'t', 'x', 'y'} - set(self.curr_position_df.columns)
        if len(missing_cols) > 0:
            raise ValueError(f"curr_position_df missing required columns: {missing_cols}")
        if self.xbin is None or self.ybin is None:
            raise ValueError("xbin and ybin must be provided")

        self.xbin = np.ascontiguousarray(np.asarray(self.xbin, dtype=np.float32))
        self.ybin = np.ascontiguousarray(np.asarray(self.ybin, dtype=np.float32))

        self.setup_position_trajectory_curves()

        if self.p_x_given_n is not None:
            self.p_x_given_n = np.ascontiguousarray(np.asarray(self.p_x_given_n, dtype=np.float32))
            if self.p_x_given_n.ndim != 3:
                raise ValueError("p_x_given_n must be 3D with shape (n_xbins, n_ybins, n_tbins)")
            if self.t_bin_edges is None:
                self.t_bin_edges = np.linspace(self.t_min, self.t_max, self.n_t_bins + 1, dtype=np.float32)
        if self.t_bin_edges is not None:
            self.t_bin_edges = np.ascontiguousarray(np.asarray(self.t_bin_edges, dtype=np.float32))
        if self.n_t_bins > 0:
            self.active_t_bin_idx = self._clamp_t_bin_idx(self.active_t_bin_idx)



    def buildUI(self):
        title = 'Volumetric 2D Time-Series Viewer'
        canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1400, 900), title=title, autoswap=False, resizable=True, decorate=True, fullscreen=False)
        self.canvas = canvas
        self.view = canvas.central_widget.add_view()
        # self.view.camera = scene.TurntableCamera(fov=_VOLUMETRIC_TURNTABLE_FOV, elevation=_VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION, azimuth=_VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH)
        self.view.camera = CustomTurntableCamera(fov=_VOLUMETRIC_TURNTABLE_FOV, elevation=_VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION, azimuth=_VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH)
        root_dockAreaWindow, _app = DockAreaWrapper.build_default_dockAreaWindow(title=title, defer_show=True)
        self.main_window = root_dockAreaWindow
        self._build_camera_view_menu(main_window=root_dockAreaWindow)
        viewer_central_widget = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_central_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        scene_wrap = VispySceneWrappingWidget(canvas=canvas, parent=viewer_central_widget, show_scene_tree=True, tree_on_right=True, tree_minimum_width=200, column_renderers=None, splitter_sizes=(700, 300))
        assert scene_wrap.scene_tree_widget is not None
        self.scene_tree_widget = scene_wrap.scene_tree_widget
        viewer_layout.addWidget(scene_wrap, stretch=1)

        if self.n_t_bins > 0:
            slider_widget = QtWidgets.QWidget()
            slider_layout = QtWidgets.QHBoxLayout(slider_widget)
            slider_layout.addWidget(QtWidgets.QLabel("t-bin:"))
            t_bin_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            t_bin_slider.setMinimum(0)
            t_bin_slider.setMaximum(max(0, self.n_t_bins - 1))
            t_bin_slider.setValue(self.active_t_bin_idx)
            t_bin_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            t_bin_slider.setTickInterval(1)
            t_bin_value_label = QtWidgets.QLabel(f"{self.active_t_bin_idx}/{max(0, self.n_t_bins - 1)}")
            t_bin_value_label.setMinimumWidth(90)
            slider_layout.addWidget(t_bin_slider, stretch=1)
            slider_layout.addWidget(t_bin_value_label)
            viewer_layout.addWidget(slider_widget, stretch=0)
            self.t_bin_slider = t_bin_slider
            self.t_bin_value_label = t_bin_value_label
            t_bin_slider.valueChanged.connect(self.on_slider_value_changed)

        epoch_slider_widget = QtWidgets.QWidget()
        epoch_slider_layout = QtWidgets.QHBoxLayout(epoch_slider_widget)
        epoch_slider_layout.addWidget(QtWidgets.QLabel("epoch:"))
        epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        epoch_slider.setMinimum(0)
        epoch_slider.setMaximum(0)
        epoch_slider.setValue(0)
        epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        epoch_slider.setTickInterval(1)
        epoch_slider.setEnabled(False)
        epoch_value_label = QtWidgets.QLabel("0/0")
        epoch_value_label.setMinimumWidth(90)
        epoch_slider_layout.addWidget(epoch_slider, stretch=1)
        epoch_slider_layout.addWidget(epoch_value_label)
        viewer_layout.addWidget(epoch_slider_widget, stretch=0)
        self.epoch_slider = epoch_slider
        self.epoch_value_label = epoch_value_label
        epoch_slider.valueChanged.connect(self.on_epoch_slider_value_changed)

        viewer_display_config = CustomDockDisplayConfig(showCloseButton=False, showTimelineSyncModeButton=False, showCollapseButton=False, custom_get_colors_callback_fn=CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color="#448aaa", border_color="#338199"))
        _, _ = root_dockAreaWindow.add_display_dock("Viewer", dockSize=(1100, 900), widget=viewer_central_widget, dockAddLocationOpts=['left'], display_config=viewer_display_config)
        root_dockAreaWindow.resize(1400, 950)
        
        # Something to give 3D context (axis from 0 to 1)
        self.debug_xyz_axes = vz.XYZAxis(parent=self.view.scene)
        self.gridlines = vz.GridLines(parent=self.view.scene, color=(0.4, 0.4, 0.4, 0.4))
        self._build_coordinate_axes()

        self._build_arena_wireframe()
        ## Graphics
        self.position_line = vz.Line(pos=self.pos3d, color=(0.22, 0.22, 0.22, 0.6), width=1.0, parent=self.view.scene, name='Pos<x,y,t>')        
        self._build_debug_crosshairs()

        if self.highlight_epochs is not None and len(self.highlight_epochs) > 0:
            self._build_highlight_bands()

        if self.n_t_bins > 0:
            self.update_active_t_bin(self.active_t_bin_idx)

        if hasattr(canvas.events, 'on_resize'):
            canvas.events.on_resize.connect(self.on_resize)
        if hasattr(canvas.events, 'on_close'):
            canvas.events.on_close.connect(self.on_close)
                        

        if hasattr(canvas.events, 'key_press'):
            canvas.events.key_press.connect(self.on_key_press)
        if hasattr(canvas.events, 'key_release'):
            canvas.events.key_release.connect(self.on_key_release)
        if hasattr(canvas.events, 'mouse_move'):
            canvas.events.mouse_move.connect(self.on_mouse_move)
        if hasattr(canvas.events, 'mouse_leave'):
            canvas.events.mouse_leave.connect(self.on_mouse_leave)
            

        

        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        self.view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max), z=(0.0, self.z_max))
        self.scene_tree_widget.rebuild()
        root_dockAreaWindow.show()


    def _apply_turntable_camera_preset(self, elevation: float, azimuth: float) -> None:
        if self.view is None:
            return
        cam = self.view.camera
        if not isinstance(cam, scene.TurntableCamera):
            return
        cam.elevation = float(elevation)
        cam.azimuth = float(azimuth)
        cam.roll = 0.0
        if self.canvas is not None:
            self.canvas.update()


    def _build_camera_view_menu(self, main_window: Any) -> None:
        mb = main_window.menuBar()
        view_menu = mb.addMenu("View")
        for label, elev, azim in _VOLUMETRIC_CAMERA_VIEW_PRESETS:
            act = QtWidgets.QAction(label, main_window)
            act.triggered.connect(lambda *_e, e=elev, a=azim: self._apply_turntable_camera_preset(e, a))
            view_menu.addAction(act)


    def _build_arena_wireframe(self):
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        z0, z1 = 0.0, self.z_max
        lower = np.array([[x_min, y_min, z0], [x_max, y_min, z0], [x_max, y_max, z0], [x_min, y_max, z0], [x_min, y_min, z0]], dtype=np.float32)
        upper = np.array([[x_min, y_min, z1], [x_max, y_min, z1], [x_max, y_max, z1], [x_min, y_max, z1], [x_min, y_min, z1]], dtype=np.float32)
        self.arena_wireframe_lines.append(vz.Line(pos=lower, color=(0.8, 0.8, 0.8, 0.7), width=1.5, parent=self.view.scene, name=f'Arena[lower]'))
        self.arena_wireframe_lines.append(vz.Line(pos=upper, color=(0.8, 0.8, 0.8, 0.7), width=1.5, parent=self.view.scene, name=f'Arena[upper]'))
        for xy in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
            edge = np.array([[xy[0], xy[1], z0], [xy[0], xy[1], z1]], dtype=np.float32)
            self.arena_wireframe_lines.append(vz.Line(pos=edge, color=(0.8, 0.8, 0.8, 0.5), width=1.0, parent=self.view.scene)) ## TODO: name? , name=f'Arena[lower]'


    def _build_coordinate_axes(self):
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        z0, z1 = 0.0, self.z_max
        x_range = float(max(x_max - x_min, 1e-6))
        y_range = float(max(y_max - y_min, 1e-6))
        z_range = float(max(z1 - z0, 1e-6))
        origin = np.array([x_min, y_min, z0], dtype=np.float32)

        x_axis = np.vstack((origin, np.array([x_max, y_min, z0], dtype=np.float32)))
        y_axis = np.vstack((origin, np.array([x_min, y_max, z0], dtype=np.float32)))
        z_axis = np.vstack((origin, np.array([x_min, y_min, z1], dtype=np.float32)))
        self.coordinate_axes_lines.append(vz.Line(pos=x_axis, color=(1.0, 0.25, 0.25, 1.0), width=2.5, parent=self.view.scene, name='Axes<X>'))
        self.coordinate_axes_lines.append(vz.Line(pos=y_axis, color=(0.25, 1.0, 0.25, 1.0), width=2.5, parent=self.view.scene, name='Axes<Y>'))
        self.coordinate_axes_lines.append(vz.Line(pos=z_axis, color=(0.25, 0.5, 1.0, 1.0), width=2.5, parent=self.view.scene, name='Axes<Z>'))

        x_offset = 0.02 * x_range
        y_offset = 0.02 * y_range
        z_offset = 0.02 * z_range
        x_label_pos = np.array([[x_max + x_offset, y_min, z0]], dtype=np.float32)
        y_label_pos = np.array([[x_min, y_max + y_offset, z0]], dtype=np.float32)
        z_label_pos = np.array([[x_min, y_min, z1 + z_offset]], dtype=np.float32)
        self.coordinate_axes_labels.append(vz.Text(text='X', color=(1.0, 0.25, 0.25, 1.0), pos=x_label_pos, font_size=12, anchor_x='left', anchor_y='center', parent=self.view.scene, name='Axes<X>_lbl'))
        self.coordinate_axes_labels.append(vz.Text(text='Y', color=(0.25, 1.0, 0.25, 1.0), pos=y_label_pos, font_size=12, anchor_x='left', anchor_y='center', parent=self.view.scene, name='Axes<Y>_lbl'))
        self.coordinate_axes_labels.append(vz.Text(text='Z', color=(0.25, 0.5, 1.0, 1.0), pos=z_label_pos, font_size=12, anchor_x='left', anchor_y='center', parent=self.view.scene, name='Axes<Z>_lbl'))


    def setup_position_trajectory_curves(self):
        """ builds the 2D+t position trajectory of the animal using self.curr_position_df
        Uses:
            self.curr_position_df
        Updates:
            self.t_min, self.t_max, self.z_scale, self.pos3d
            
        """
        t_vals = np.asarray(self.curr_position_df['t'].to_numpy(), dtype=np.float32)
        x_vals = np.asarray(self.curr_position_df['x'].to_numpy(), dtype=np.float32)
        y_vals = np.asarray(self.curr_position_df['y'].to_numpy(), dtype=np.float32)
        valid_mask = np.isfinite(t_vals) & np.isfinite(x_vals) & np.isfinite(y_vals)
        t_vals = t_vals[valid_mask]
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        if t_vals.size < 2:
            raise ValueError("curr_position_df must contain at least two valid samples")

        new_t_min: float = float(np.nanmin(t_vals))
        self.t_min = min((self.t_min or 0.0), new_t_min) ## update t_max if needed

        new_t_max: float = float(np.nanmax(t_vals))
        self.t_max = max((self.t_max or 1.0), new_t_max) ## update t_max if needed
        
        t_duration = max(self.t_max - self.t_min, 1e-6)
        self.z_scale = float((self.xbin[-1] - self.xbin[0]) / t_duration)
        
        z_vals = (t_vals - self.t_min) * self.z_scale
        self.pos3d = np.ascontiguousarray(np.column_stack((x_vals, y_vals, z_vals)), dtype=np.float32)


    def _t_to_z(self, t: NDArray) -> NDArray:
        """Map session time (seconds) to scene z using the same scale as `setup_position_trajectory_curves`."""
        t_arr = np.asarray(t, dtype=np.float64)
        return np.asarray((t_arr - float(self.t_min)) * float(self.z_scale), dtype=np.float32)


    # Reward Zone Shape Extrusions
    def clear_reward_zone_extrusions(self) -> None:
        for vis in self.reward_zone_extrusion_visuals:
            if vis is not None:
                vis.parent = None
        self.reward_zone_extrusion_visuals.clear()
        if self.canvas is not None:
            self.canvas.update()


    @staticmethod
    def _segment_to_dashed_line_pos(a: NDArray, b: NDArray, dash_len: float, gap_len: float) -> NDArray:
        """Return (2*k, 3) float32 positions for vz.Line(..., connect='segments'): dash-gap pattern along [a,b]."""
        aa = np.asarray(a, dtype=np.float64).reshape(3)
        bb = np.asarray(b, dtype=np.float64).reshape(3)
        d = bb - aa
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            return np.zeros((0, 3), dtype=np.float32)
        u = d / L
        pos_list: List[NDArray] = []
        t = 0.0
        drawing = True
        while t < L - 1e-9:
            if drawing:
                t1 = min(t + float(dash_len), L)
                if t1 - t > 1e-9:
                    pos_list.append(aa + u * t)
                    pos_list.append(aa + u * t1)
                t = t1
            else:
                t = min(t + float(gap_len), L)
            drawing = not drawing
        if not pos_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.ascontiguousarray(np.vstack(pos_list), dtype=np.float32)


    @staticmethod
    def _prism_outline_dashed_pos(ring_xy: NDArray, z0: float, z1: float, dash_len: float, gap_len: float) -> NDArray:
        """Closed XY ring (n,2), unique vertices; dashed bottom loop, top loop, and verticals at each vertex."""
        ring = np.asarray(ring_xy, dtype=np.float32)
        if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        if np.allclose(ring[0], ring[-1], atol=1e-5):
            ring = ring[:-1]
        n = int(ring.shape[0])
        if n < 3:
            return np.zeros((0, 3), dtype=np.float32)
        z0f, z1f = float(z0), float(z1)
        chunks: List[NDArray] = []
        for i in range(n):
            j = (i + 1) % n
            p0 = np.array([ring[i, 0], ring[i, 1], z0f], dtype=np.float64)
            p1 = np.array([ring[j, 0], ring[j, 1], z0f], dtype=np.float64)
            q0 = np.array([ring[i, 0], ring[i, 1], z1f], dtype=np.float64)
            q1 = np.array([ring[j, 0], ring[j, 1], z1f], dtype=np.float64)
            chunks.append(Volumentric2DTimeSeriesPlotter._segment_to_dashed_line_pos(p0, p1, dash_len, gap_len))
            chunks.append(Volumentric2DTimeSeriesPlotter._segment_to_dashed_line_pos(q0, q1, dash_len, gap_len))
            chunks.append(Volumentric2DTimeSeriesPlotter._segment_to_dashed_line_pos(p0, q0, dash_len, gap_len))
        out = [c for c in chunks if c.size > 0]
        if not out:
            return np.zeros((0, 3), dtype=np.float32)
        return np.ascontiguousarray(np.vstack(out), dtype=np.float32)


    @staticmethod
    def _extruded_prism_vertices_faces(ring_xy: NDArray, z0: float, z1: float) -> Tuple[NDArray, NDArray]:
        """Extrude closed XY polygon along z from z0 to z1; returns (vertices (2n,3), faces (m,3))."""
        pts = np.asarray(ring_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
        if np.allclose(pts[0], pts[-1], atol=1e-5):
            pts = pts[:-1]
        n = int(pts.shape[0])
        if n < 3:
            return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
        z0f, z1f = float(z0), float(z1)
        bottom = np.column_stack([pts, np.full(n, z0f, dtype=np.float32)])
        top = np.column_stack([pts, np.full(n, z1f, dtype=np.float32)])
        verts = np.ascontiguousarray(np.vstack([bottom, top]), dtype=np.float32)
        closed_2d = _ensure_closed_pos(pts)
        tri = _triangulate_polygon_2d(closed_2d)
        if tri is None or len(tri) < 1:
            return (verts, np.zeros((0, 3), dtype=np.int32))
        faces_bottom = np.asarray(tri, dtype=np.int32)
        faces_top = faces_bottom.copy() + np.int32(n)
        faces_top = faces_top[:, [0, 2, 1]]
        side_faces: List[NDArray] = []
        for i in range(n):
            j = (i + 1) % n
            bi, bj, ti, tj = i, j, i + n, j + n
            side_faces.append(np.array([[bi, bj, tj], [bi, tj, ti]], dtype=np.int32))
        faces = np.ascontiguousarray(np.vstack([faces_bottom, faces_top] + side_faces), dtype=np.int32)
        return (verts, faces)


    def plot_reward_zones_as_extruded_prisms(self, reward_zones: Dict[str, Polygon], *, clear_existing: bool = True, fill_rgba: Tuple[float, float, float, float] = (1.0, 0.55, 0.12, 0.28), outline_rgba: Tuple[float, float, float, float] = (0.95, 0.42, 0.05, 0.92), outline_width: float = 2.0, dash_length: Optional[float] = None, gap_length: Optional[float] = None) -> None:
        """Draw each 2D shapely polygon as a prism along scene z from 0 to z_max (full time axis).

        Fills use translucent orange by default; outlines are orange dashes (segment-based; VisPy has no 3D stipple).
        Interior rings (holes) are ignored; only ``polygon.exterior`` is extruded.
        """
        if self.view is None:
            return
        if clear_existing:
            self.clear_reward_zone_extrusions()
        x_span = float(self.xbin[-1] - self.xbin[0]) if self.xbin is not None and len(self.xbin) > 1 else 1.0
        y_span = float(self.ybin[-1] - self.ybin[0]) if self.ybin is not None and len(self.ybin) > 1 else 1.0
        span = max(x_span, y_span, 1e-6)
        dlen = float(dash_length) if dash_length is not None else max(span * 0.025, 1e-4)
        glen = float(gap_length) if gap_length is not None else max(dlen * 0.45, 1e-4)
        z0, z1 = 0.0, float(self.z_max)
        parent = self.view.scene
        for name, poly in reward_zones.items():
            if poly is None or poly.is_empty:
                continue
            if not isinstance(poly, Polygon):
                continue
            try:
                xy = np.asarray(poly.exterior.coords, dtype=np.float32)
            except Exception:
                continue
            if xy.shape[0] < 4:
                continue
            verts, faces = self._extruded_prism_vertices_faces(xy, z0, z1)
            if verts.shape[0] < 6 or faces.shape[0] < 1:
                continue
            mesh = vz.Mesh(vertices=verts, faces=faces, color=fill_rgba, parent=parent, name=f'RewardZoneFill[{name}]')
            mesh.order = 19
            mesh.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
            self.reward_zone_extrusion_visuals.append(mesh)
            dash_pos = self._prism_outline_dashed_pos(xy, z0, z1, dlen, glen)
            if dash_pos.shape[0] >= 2:
                ol = vz.Line(pos=dash_pos, color=outline_rgba, width=float(outline_width), connect='segments', method='gl', parent=parent, name=f'RewardZoneOutline[{name}]')
                ol.order = 20
                self.reward_zone_extrusion_visuals.append(ol)
        if self.scene_tree_widget is not None:
            self.scene_tree_widget.rebuild()
        if self.canvas is not None:
            self.canvas.update()


    @staticmethod
    def _make_dummy_filter_epochs(n: int) -> pd.DataFrame:
        return pd.DataFrame({'original_epoch_idx': np.arange(int(n), dtype=np.int64), 'start': np.full(n, np.nan, dtype=np.float64), 'stop': np.full(n, np.nan, dtype=np.float64)})


    def _clear_match_line_visuals(self) -> None:
        for lv in self.match_line_visuals:
            if lv is not None:
                lv.parent = None
        self.match_line_visuals.clear()


    def _build_match_lines_for_epoch(self, epoch_idx: int, parent_node: Any) -> None:
        """3D past/future match trajectories under `parent_node`, z = (t - t_min) * z_scale."""
        if self._epoch_mask_datasource is None:
            self._match_lines_built_for_epoch_idx = int(epoch_idx)
            return
        try:
            epoch_data = self._epoch_mask_datasource._prepare_epoch_data(int(epoch_idx), minimum_included_matching_sequence_length=self.minimum_included_matching_sequence_length)
        except Exception:
            self._match_lines_built_for_epoch_idx = int(epoch_idx)
            return
        pos_dict = epoch_data.get('curr_matching_past_future_positions_df_dict', None)
        if not pos_dict:
            self._match_lines_built_for_epoch_idx = int(epoch_idx)
            return
        for category, default_hue in (('past', 0.0), ('future', 0.5)):
            sub = pos_dict.get(category, None)
            if not sub:
                continue
            rgb = colorsys.hsv_to_rgb(default_hue, 0.85, 0.95)
            rgba = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.92)
            for _epoch_id, positions_df in sub.items():
                if positions_df is None or len(positions_df) < 2:
                    continue
                if 't' not in positions_df.columns or 'x' not in positions_df.columns or 'y' not in positions_df.columns:
                    continue
                x_coords = np.asarray(positions_df['x'].to_numpy(), dtype=np.float64)
                y_coords = np.asarray(positions_df['y'].to_numpy(), dtype=np.float64)
                t_coords = np.asarray(positions_df['t'].to_numpy(), dtype=np.float64)
                vm = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(t_coords)
                if np.count_nonzero(vm) < 2:
                    continue
                xv, yv, tv = x_coords[vm], y_coords[vm], t_coords[vm]
                zv = self._t_to_z(tv)
                pos3d = np.ascontiguousarray(np.column_stack([xv, yv, zv]), dtype=np.float32)
                line = vz.Line(pos=pos3d, color=rgba, width=2.5, method='gl', parent=parent_node, name=f'Match3D[{category}]_{_epoch_id}')
                line.order = 8
                self.match_line_visuals.append(line)
        self._match_lines_built_for_epoch_idx = int(epoch_idx)


    # ==================================================================================================================================================================================================================================================================================== #
    # Debug Crosshairs                                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #
    def _build_debug_crosshairs(self):
        """Builds hidden XYZ crosshair lines used for Shift-hover debugging."""
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        x_half = float(max((x_max - x_min) * 0.02, 1e-3))
        y_half = float(max((y_max - y_min) * 0.02, 1e-3))
        z_half = float(max(self.z_max * 0.02, 1e-3))
        center = np.array([x_min, y_min, 0.0], dtype=np.float32)
        x_line = np.array([[center[0] - x_half, center[1], center[2]], [center[0] + x_half, center[1], center[2]]], dtype=np.float32)
        y_line = np.array([[center[0], center[1] - y_half, center[2]], [center[0], center[1] + y_half, center[2]]], dtype=np.float32)
        z_line = np.array([[center[0], center[1], center[2] - z_half], [center[0], center[1], center[2] + z_half]], dtype=np.float32)
        self.debug_crosshair_lines.append(vz.Line(pos=x_line, color=(1.0, 0.2, 0.2, 0.95), width=2.5, parent=self.view.scene))
        self.debug_crosshair_lines.append(vz.Line(pos=y_line, color=(0.2, 1.0, 0.2, 0.95), width=2.5, parent=self.view.scene))
        self.debug_crosshair_lines.append(vz.Line(pos=z_line, color=(0.2, 0.55, 1.0, 0.95), width=2.5, parent=self.view.scene))
        self._hide_debug_crosshairs()


    def _set_debug_crosshairs_position(self, center_xyz: NDArray):
        if len(self.debug_crosshair_lines) != 3:
            return
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        x_half = float(max((x_max - x_min) * 0.02, 1e-3))
        y_half = float(max((y_max - y_min) * 0.02, 1e-3))
        z_half = float(max(self.z_max * 0.02, 1e-3))
        x, y, z = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
        self.debug_crosshair_lines[0].set_data(pos=np.array([[x - x_half, y, z], [x + x_half, y, z]], dtype=np.float32))
        self.debug_crosshair_lines[1].set_data(pos=np.array([[x, y - y_half, z], [x, y + y_half, z]], dtype=np.float32))
        self.debug_crosshair_lines[2].set_data(pos=np.array([[x, y, z - z_half], [x, y, z + z_half]], dtype=np.float32))
        for a_line in self.debug_crosshair_lines:
            a_line.visible = True


    def _hide_debug_crosshairs(self):
        for a_line in self.debug_crosshair_lines:
            a_line.visible = False
        self.debug_nearest_pos3d_idx = None


    def _event_has_shift_modifier(self, event) -> bool:
        modifiers = list(getattr(event, 'modifiers', []) or [])
        return any(str(a_mod) == 'Shift' for a_mod in modifiers)


    def _nearest_pos3d_idx_from_canvas_pos(self, canvas_pos: NDArray) -> Optional[int]:
        if self.pos3d is None or np.shape(self.pos3d)[0] == 0:
            return None
        if self.view is None or self.canvas is None:
            return None
        try:
            scene_to_canvas = self.view.scene.node_transform(self.canvas.scene)
            mapped = np.asarray(scene_to_canvas.map(self.pos3d), dtype=np.float32)
        except Exception:
            return None
        if np.ndim(mapped) != 2 or np.shape(mapped)[0] == 0:
            return None
        if np.shape(mapped)[1] < 2:
            return None
        mapped_xy = mapped[:, :2]
        finite_mask = np.isfinite(mapped_xy[:, 0]) & np.isfinite(mapped_xy[:, 1])
        if not np.any(finite_mask):
            return None
        valid_xy = mapped_xy[finite_mask]
        valid_indices = np.flatnonzero(finite_mask)
        dx = valid_xy[:, 0] - float(canvas_pos[0])
        dy = valid_xy[:, 1] - float(canvas_pos[1])
        distances_sq = (dx * dx) + (dy * dy)
        nearest_local_idx = int(np.argmin(distances_sq))
        if float(distances_sq[nearest_local_idx]) > float(self.debug_crosshair_snap_max_pixel_distance * self.debug_crosshair_snap_max_pixel_distance):
            return None
        return int(valid_indices[nearest_local_idx])


    def _clamp_t_bin_idx(self, t_bin_idx: int) -> int:
        if self.n_t_bins <= 0:
            return 0
        return int(np.clip(int(t_bin_idx), 0, self.n_t_bins - 1))


    def _t_bin_center(self, t_bin_idx: int) -> float:
        idx = self._clamp_t_bin_idx(t_bin_idx)
        if self.t_bin_edges is not None and np.size(self.t_bin_edges) >= (idx + 2):
            return float((self.t_bin_edges[idx] + self.t_bin_edges[idx + 1]) * 0.5)
        if self.n_t_bins > 0:
            return float(np.linspace(self.t_min, self.t_max, self.n_t_bins, dtype=np.float32)[idx])
        return float(self.t_min)


    # ==================================================================================================================================================================================================================================================================================== #
    # Contour Peak Promenences                                                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    def _next_decoded_posterior_key(self) -> str:
        while True:
            self.decoded_posterior_counter = int(self.decoded_posterior_counter) + 1
            key = f"decoded_posterior_{self.decoded_posterior_counter:04d}"
            if key not in self.decoded_posteriors_by_key:
                return key


    def _next_posterior_contour_key(self) -> str:
        while True:
            self.posterior_contours_counter = int(self.posterior_contours_counter) + 1
            key = f"posterior_contour_{self.posterior_contours_counter:04d}"
            if key not in self.posterior_contours_by_key:
                return key


    def _build_posterior_contours_3d(self, per_t_bin_mask: NDArray, t_bin_edges_for_contours: Optional[NDArray] = None, line_width: float = 2.0, contour_alpha: float = 0.7, level: float = 0.5, fill: bool = True, fill_alpha: float = 0.25, scene_parent: Optional[Any] = None) -> Tuple[List[Any], List[Any], List[Dict]]:
        """Build 3D contour line visuals (and optional filled meshes) from a per-time-bin boolean mask array.

        per_t_bin_mask: (n_xbins, n_ybins, n_tbins) boolean/float mask array.
        Each time-bin slice is contoured in 2D then lifted to the corresponding z-height.
        When fill=True, each contour is also rendered as a translucent vz.Mesh using the
        same RGB as the line with alpha=fill_alpha (default 0.25).
        Returns (line_visuals, fill_visuals, contour_geometries) where contour_geometries is a
        list of dicts with keys 'polygon_2d', 'z_val', 'rgba', 't_idx' for downstream use.
        """
        parent = self.view.scene if scene_parent is None else scene_parent
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        n_tbins = int(per_t_bin_mask.shape[2])
        line_visuals: List[Any] = []
        fill_visuals: List[Any] = []
        contour_geometries: List[Dict] = []
        for t_idx in range(n_tbins):
            if t_bin_edges_for_contours is not None and len(t_bin_edges_for_contours) >= (t_idx + 2):
                t_center = float((t_bin_edges_for_contours[t_idx] + t_bin_edges_for_contours[t_idx + 1]) * 0.5)
            elif self.t_bin_edges is not None and len(self.t_bin_edges) >= (t_idx + 2):
                t_center = float((self.t_bin_edges[t_idx] + self.t_bin_edges[t_idx + 1]) * 0.5)
            else:
                t_center = float(self.t_min + (t_idx + 0.5) * ((self.t_max - self.t_min) / max(n_tbins, 1)))
            z_val = float((t_center - self.t_min) * self.z_scale)
            hue = (t_idx / max(n_tbins, 1)) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            t_color = (rgb[0], rgb[1], rgb[2], contour_alpha)
            mask_2d = per_t_bin_mask[:, :, t_idx].T
            contour_items: List[ContourItem] = contours_from_masks([mask_2d], x_bounds=(x_min, x_max), y_bounds=(y_min, y_max), colors=[t_color], level=level, return_per_mask=False)
            for pos_2d, rgba in contour_items:
                if len(pos_2d) < 2:
                    continue
                contour_geometries.append({'polygon_2d': np.array(pos_2d, dtype=np.float32), 'z_val': z_val, 'rgba': rgba, 't_idx': t_idx})
                pos_3d = np.column_stack([pos_2d, np.full(len(pos_2d), z_val, dtype=np.float32)]).astype(np.float32)
                line = vz.Line(pos=pos_3d, color=rgba, width=line_width, parent=parent, name=f'Contour[t={t_idx}]')
                line.order = 22
                line_visuals.append(line)
                if fill and len(pos_2d) >= 3:
                    faces = _triangulate_polygon_2d(pos_2d)
                    if faces is not None and len(faces) >= 1:
                        closed_2d = _ensure_closed_pos(pos_2d)
                        verts_3d = np.column_stack([closed_2d, np.full(len(closed_2d), z_val, dtype=np.float32)]).astype(np.float32)
                        fill_rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(fill_alpha))
                        mesh = vz.Mesh(vertices=verts_3d, faces=faces, color=fill_rgba, parent=parent, name=f'ContourFill[t={t_idx}]')
                        mesh.order = 21
                        mesh.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
                        fill_visuals.append(mesh)
        return (line_visuals, fill_visuals, contour_geometries)


    # ==================================================================================================================================================================================================================================================================================== #
    # Posterior Planes                                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #
    def _normalize_posterior_2d(self, decoded_posterior_2d: NDArray) -> NDArray:
        posterior_2d = np.asarray(decoded_posterior_2d, dtype=np.float32)
        if posterior_2d.ndim != 2:
            raise ValueError(f"decoded_posterior_2d must be 2D with shape (n_xbins, n_ybins) or (n_ybins, n_xbins), got shape: {np.shape(posterior_2d)}")
        posterior_2d = np.nan_to_num(posterior_2d, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(posterior_2d, dtype=np.float32)


    def _posterior_2d_to_rgba(self, posterior_2d: NDArray) -> NDArray:
        img = np.asarray(posterior_2d.T, dtype=np.float32)
        nan_mask = np.isnan(img)                              # 1. capture NaN positions BEFORE replacing
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)
        cmap = get_colormap('viridis')
        rgba = np.ascontiguousarray(cmap.map(img_norm), dtype=np.float32).reshape(*img_norm.shape, 4)  # cmap.map flattens input; restore (H, W, 4)
        rgba[nan_mask, 3] = 0.0                               # 2. zero alpha where NaN was
        return rgba


    def _build_posterior_plane_from_img(self, img: NDArray, time_value: float, visual_name: str = "posterior", **vzImageKwargs) -> vz.Image:
        posterior_plane = vz.Image(data=img, parent=self.view.scene, name=visual_name, **vzImageKwargs)
        posterior_plane.order = 20
        posterior_plane.set_gl_state('translucent', depth_test=True, depth_mask=False)
        n_y, n_x = img.shape[:2]
        x_scale = float((self.xbin[-1] - self.xbin[0]) / max(n_x, 1))
        y_scale = float((self.ybin[-1] - self.ybin[0]) / max(n_y, 1))
        z_val = float((float(time_value) - self.t_min) * self.z_scale)
        transform = scene.transforms.MatrixTransform()
        transform.scale((x_scale, y_scale, 1.0))
        transform.translate((float(self.xbin[0]), float(self.ybin[0]), z_val))
        posterior_plane.transform = transform
        return posterior_plane


    def _build_posterior_plane_from_2d(self, decoded_posterior_2d: NDArray, time_value: float, visual_name: str = "posterior") -> vz.Image:
        # posterior_2d = self._normalize_posterior_2d(decoded_posterior_2d=decoded_posterior_2d)
        img_rgba = self._posterior_2d_to_rgba(posterior_2d=decoded_posterior_2d)
        return self._build_posterior_plane_from_img(img=img_rgba, time_value=time_value, visual_name=visual_name)
        # ## raw
        # return self._build_posterior_plane_from_img(img=decoded_posterior_2d, time_value=time_value, visual_name=visual_name, cmap='viridis')

    
        



    def _refresh_scene_tree(self):
        if self.scene_tree_widget is not None:
            self.scene_tree_widget.rebuild()


    def _build_posterior_plane(self, t_bin_idx: int):
        if self.p_x_given_n is None or self.n_t_bins <= 0:
            return
        idx = self._clamp_t_bin_idx(t_bin_idx)
        t_bin_center = self._t_bin_center(idx)
        posterior_2d = np.asarray(self.p_x_given_n[:, :, idx], dtype=np.float32)
        posterior_plane = self._build_posterior_plane_from_2d(decoded_posterior_2d=posterior_2d, time_value=t_bin_center, visual_name='posterior')
        self.posterior_plane = posterior_plane


    def update_active_t_bin(self, t_bin_idx: int):
        if self.n_t_bins <= 0:
            return
        idx = self._clamp_t_bin_idx(t_bin_idx)
        if self.posterior_plane is not None:
            self.posterior_plane.parent = None
            self.posterior_plane = None
        self._build_posterior_plane(idx)
        self.active_t_bin_idx = idx
        if self.t_bin_value_label is not None:
            self.t_bin_value_label.setText(f"{idx}/{max(0, self.n_t_bins - 1)}")


    def add_decoded_posterior(self, decoded_posterior_2d: NDArray, time_value: float, unique_identifier: Optional[str] = None, visible: bool = True, replace_if_exists: bool = True) -> str:
        identifier = str(unique_identifier) if unique_identifier is not None else self._next_decoded_posterior_key()
        if len(identifier) == 0:
            identifier = self._next_decoded_posterior_key()
        # posterior_2d = self._normalize_posterior_2d(decoded_posterior_2d=decoded_posterior_2d)
        posterior_2d = decoded_posterior_2d.copy()
        existing_item = self.decoded_posteriors_by_key.get(identifier, None)
        if existing_item is not None:
            if not bool(replace_if_exists):
                raise KeyError(f"decoded posterior key already exists: '{identifier}'")
            self.remove_decoded_posterior(unique_identifier=identifier)
        posterior_plane = self._build_posterior_plane_from_2d(decoded_posterior_2d=posterior_2d, time_value=float(time_value), visual_name=f"posterior[{identifier}]")
        posterior_plane.visible = bool(visible)
        self.decoded_posteriors_by_key[identifier] = {'unique_identifier': identifier, 'time_value': float(time_value), 'decoded_posterior_2d': posterior_2d, 'posterior_plane': posterior_plane, 'visible': bool(visible)}
        self._refresh_scene_tree()
        return identifier


    def get_decoded_posterior(self, unique_identifier: str) -> Optional[Dict[str, Any]]:
        return self.decoded_posteriors_by_key.get(str(unique_identifier), None)


    def list_decoded_posterior_keys(self) -> List[str]:
        return list(self.decoded_posteriors_by_key.keys())


    def set_decoded_posterior_visibility(self, unique_identifier: str, is_visible: bool) -> bool:
        item = self.get_decoded_posterior(unique_identifier=unique_identifier)
        if item is None:
            return False
        posterior_plane = item.get('posterior_plane', None)
        if posterior_plane is None:
            return False
        item['visible'] = bool(is_visible)
        posterior_plane.visible = bool(is_visible)
        return True


    def remove_decoded_posterior(self, unique_identifier: str) -> bool:
        item = self.decoded_posteriors_by_key.pop(str(unique_identifier), None)
        if item is None:
            return False
        posterior_plane = item.get('posterior_plane', None)
        if posterior_plane is not None:
            posterior_plane.parent = None
        self._refresh_scene_tree()
        return True


    def clear_decoded_posteriors(self):
        if len(self.decoded_posteriors_by_key) == 0:
            return
        for item in self.decoded_posteriors_by_key.values():
            posterior_plane = item.get('posterior_plane', None)
            if posterior_plane is not None:
                posterior_plane.parent = None
        self.decoded_posteriors_by_key = {}
        self._refresh_scene_tree()


    # ==================================================================================================================================================================================================================================================================================== #
    # Posterior Contours (3D)                                                                                                                                                                                                                                                               #
    # ==================================================================================================================================================================================================================================================================================== #

    def add_posterior_contours(self, per_t_bin_mask: NDArray, t_bin_edges: Optional[NDArray] = None, unique_identifier: Optional[str] = None, visible: bool = True, replace_if_exists: bool = True, line_width: float = 2.0, contour_alpha: float = 0.7, level: float = 0.5, fill: bool = True, fill_alpha: float = 0.25, scene_parent: Optional[Any] = None) -> str:
        """Add 3D contour lines (and optional filled meshes) from a per-time-bin boolean mask (n_xbins, n_ybins, n_tbins). Returns the assigned key."""
        identifier = str(unique_identifier) if unique_identifier is not None else self._next_posterior_contour_key()
        if len(identifier) == 0:
            identifier = self._next_posterior_contour_key()
        existing = self.posterior_contours_by_key.get(identifier, None)
        if existing is not None:
            if not bool(replace_if_exists):
                raise KeyError(f"posterior contour key already exists: '{identifier}'")
            self.remove_posterior_contours(unique_identifier=identifier)
        resolved_parent = self.view.scene if scene_parent is None else scene_parent
        line_visuals, fill_visuals, contour_geometries = self._build_posterior_contours_3d(per_t_bin_mask=np.asarray(per_t_bin_mask), t_bin_edges_for_contours=t_bin_edges, line_width=line_width, contour_alpha=contour_alpha, level=level, fill=fill, fill_alpha=fill_alpha, scene_parent=resolved_parent)
        for lv in line_visuals:
            lv.visible = bool(visible)
        for fv in fill_visuals:
            fv.visible = bool(visible)
        self.posterior_contours_by_key[identifier] = {'unique_identifier': identifier, 'line_visuals': line_visuals, 'fill_visuals': fill_visuals, 'contour_geometries': contour_geometries, 't_bin_edges': (np.array(t_bin_edges, dtype=np.float32) if t_bin_edges is not None else None), 'visible': bool(visible), 'scene_parent': resolved_parent}
        self._refresh_scene_tree()
        return identifier


    def get_posterior_contours(self, unique_identifier: str) -> Optional[Dict[str, Any]]:
        return self.posterior_contours_by_key.get(str(unique_identifier), None)


    def list_posterior_contour_keys(self) -> List[str]:
        return list(self.posterior_contours_by_key.keys())


    def set_posterior_contours_visibility(self, unique_identifier: str, is_visible: bool) -> bool:
        item = self.get_posterior_contours(unique_identifier=unique_identifier)
        if item is None:
            return False
        item['visible'] = bool(is_visible)
        for lv in item.get('line_visuals', []):
            if lv is not None:
                lv.visible = bool(is_visible)
        for fv in item.get('fill_visuals', []):
            if fv is not None:
                fv.visible = bool(is_visible)
        self.set_contour_extrusions_visibility(unique_identifier=unique_identifier, is_visible=is_visible)
        return True


    def remove_posterior_contours(self, unique_identifier: str) -> bool:
        self.remove_contour_extrusions(unique_identifier=unique_identifier)
        item = self.posterior_contours_by_key.pop(str(unique_identifier), None)
        if item is None:
            return False
        for lv in item.get('line_visuals', []):
            if lv is not None:
                lv.parent = None
        for fv in item.get('fill_visuals', []):
            if fv is not None:
                fv.parent = None
        self._refresh_scene_tree()
        return True


    def clear_posterior_contours(self):
        self.clear_contour_extrusions()
        if len(self.posterior_contours_by_key) == 0:
            return
        for item in self.posterior_contours_by_key.values():
            for lv in item.get('line_visuals', []):
                if lv is not None:
                    lv.parent = None
            for fv in item.get('fill_visuals', []):
                if fv is not None:
                    fv.parent = None
        self.posterior_contours_by_key = {}
        self._refresh_scene_tree()


    # ==================================================================================================================================================================================================================================================================================== #
    # Contour Extrusion + Position-Line Intersection Highlighting                                                                                                                                                                                                                          #
    # ==================================================================================================================================================================================================================================================================================== #

    def _build_extrusion_wall_mesh(self, polygon_2d: NDArray, z_bottom: float, z_top: float, rgba: Tuple[float, float, float, float], wall_alpha: float = 0.1, name: str = 'ExtrusionWall', scene_parent: Optional[Any] = None) -> vz.Mesh:
        """Create a translucent mesh representing the extruded walls of a 2D polygon between z_bottom and z_top."""
        closed = _ensure_closed_pos(polygon_2d)
        n = len(closed)
        bottom_verts = np.column_stack([closed, np.full(n, z_bottom, dtype=np.float32)]).astype(np.float32)
        top_verts = np.column_stack([closed, np.full(n, z_top, dtype=np.float32)]).astype(np.float32)
        vertices = np.vstack([bottom_verts, top_verts]).astype(np.float32)  # shape (2*n, 3)
        faces_list = []
        for i in range(n - 1):
            b0, b1 = i, i + 1
            t0, t1 = i + n, i + 1 + n
            faces_list.append([b0, b1, t1])
            faces_list.append([b0, t1, t0])
        faces = np.array(faces_list, dtype=np.uint32)
        wall_rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(wall_alpha))
        parent = self.view.scene if scene_parent is None else scene_parent
        mesh = vz.Mesh(vertices=vertices, faces=faces, color=wall_rgba, parent=parent, name=name)
        mesh.set_gl_state('translucent', depth_test=True, cull_face=False)
        mesh.order = 19
        return mesh


    def _find_pos3d_segments_inside_contour(self, polygon_2d: NDArray, z_bottom: float, z_top: float) -> List[Tuple[int, int]]:
        """Return list of (start_idx, end_idx) contiguous segments of self.pos3d inside the extruded contour volume."""
        from matplotlib.path import Path as MplPath
        if self.pos3d is None or len(self.pos3d) == 0:
            return []
        closed = _ensure_closed_pos(polygon_2d)
        mpl_path = MplPath(closed[:, :2])
        xy_inside = mpl_path.contains_points(self.pos3d[:, :2])
        z_vals = self.pos3d[:, 2]
        z_inside = (z_vals >= z_bottom) & (z_vals <= z_top)
        inside = xy_inside & z_inside
        segments: List[Tuple[int, int]] = []
        i = 0
        n = len(inside)
        while i < n:
            if inside[i]:
                start = i
                while i < n and inside[i]:
                    i += 1
                segments.append((start, i))
            else:
                i += 1
        return segments


    def _build_intersection_tube(self, segment_points: NDArray, rgba: Tuple[float, float, float, float], tube_radius: float = 1.5, tube_alpha: float = 0.3, name: str = 'IntersectionTube', scene_parent: Optional[Any] = None) -> Optional[Any]:
        """Create a translucent vz.Tube highlighting a segment of the position line."""
        if len(segment_points) < 2:
            return None
        tube_color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(tube_alpha))
        parent = self.view.scene if scene_parent is None else scene_parent
        tube = vz.Tube(points=segment_points, radius=tube_radius, color=tube_color, tube_points=8, shading='smooth', parent=parent, name=name)
        tube.set_gl_state('translucent', depth_test=True, cull_face=False)
        tube.order = 18
        return tube


    def build_contour_extrusions(self, unique_identifier: str, z_half_extent: Optional[float] = None, tube_radius: float = 1.5, tube_alpha: float = 0.3, wall_alpha: float = 0.15) -> bool:
        """Build volumetric extrusions for an existing set of contours and highlight intersecting position-line segments.

        z_half_extent: distance in z-units to extrude in each direction from the contour's z_val.
                       None = extend from z=0 to z=z_max (full z-axis range).
        """
        contour_entry = self.posterior_contours_by_key.get(str(unique_identifier), None)
        if contour_entry is None:
            return False
        geometries = contour_entry.get('contour_geometries', [])
        if len(geometries) == 0:
            return False
        self.remove_contour_extrusions(unique_identifier=unique_identifier)
        scene_parent = contour_entry.get('scene_parent', self.view.scene)
        wall_visuals: List[Any] = []
        tube_visuals: List[Any] = []
        for geom in geometries:
            polygon_2d = geom['polygon_2d']
            z_val = float(geom['z_val'])
            rgba = geom['rgba']
            t_idx = int(geom['t_idx'])
            if z_half_extent is not None:
                z_bot = z_val - float(z_half_extent)
                z_top = z_val + float(z_half_extent)
            else:
                z_bot = 0.0
                z_top = float(self.z_max)
            wall_mesh = self._build_extrusion_wall_mesh(polygon_2d=polygon_2d, z_bottom=z_bot, z_top=z_top, rgba=rgba, wall_alpha=wall_alpha, name=f'ExtWall[{unique_identifier},t={t_idx}]', scene_parent=scene_parent)
            wall_visuals.append(wall_mesh)
            segments = self._find_pos3d_segments_inside_contour(polygon_2d=polygon_2d, z_bottom=z_bot, z_top=z_top)
            for seg_i, (s_start, s_end) in enumerate(segments):
                seg_pts = np.ascontiguousarray(self.pos3d[s_start:s_end], dtype=np.float32)
                tube = self._build_intersection_tube(segment_points=seg_pts, rgba=rgba, tube_radius=tube_radius, tube_alpha=tube_alpha, name=f'IntTube[{unique_identifier},t={t_idx},s={seg_i}]', scene_parent=scene_parent)
                if tube is not None:
                    tube_visuals.append(tube)
        self.contour_extrusion_visuals_by_key[str(unique_identifier)] = {'unique_identifier': str(unique_identifier), 'wall_visuals': wall_visuals, 'tube_visuals': tube_visuals, 'visible': True}
        self._refresh_scene_tree()
        return True


    def remove_contour_extrusions(self, unique_identifier: str) -> bool:
        """Remove extrusion wall meshes and intersection tubes for the given contour identifier."""
        item = self.contour_extrusion_visuals_by_key.pop(str(unique_identifier), None)
        if item is None:
            return False
        for v in item.get('wall_visuals', []):
            if v is not None:
                v.parent = None
        for v in item.get('tube_visuals', []):
            if v is not None:
                v.parent = None
        self._refresh_scene_tree()
        return True


    def set_contour_extrusions_visibility(self, unique_identifier: str, is_visible: bool) -> bool:
        item = self.contour_extrusion_visuals_by_key.get(str(unique_identifier), None)
        if item is None:
            return False
        item['visible'] = bool(is_visible)
        for v in item.get('wall_visuals', []):
            if v is not None:
                v.visible = bool(is_visible)
        for v in item.get('tube_visuals', []):
            if v is not None:
                v.visible = bool(is_visible)
        return True


    def clear_contour_extrusions(self):
        """Remove all contour extrusion visuals."""
        if len(self.contour_extrusion_visuals_by_key) == 0:
            return
        for item in self.contour_extrusion_visuals_by_key.values():
            for v in item.get('wall_visuals', []):
                if v is not None:
                    v.parent = None
            for v in item.get('tube_visuals', []):
                if v is not None:
                    v.parent = None
        self.contour_extrusion_visuals_by_key = {}
        self._refresh_scene_tree()


    def _to_rgba(self, color_value: Any, alpha: float = 0.15) -> Tuple[float, float, float, float]:
        if color_value is None:
            return (0.2, 0.8, 1.0, alpha)
        try:
            rgba = Color(color_value).rgba
            return (float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha)
        except Exception:
            return (0.2, 0.8, 1.0, alpha)


    def _build_highlight_bands(self):
        """ builds bands for intervals defined in self.highlight_epochs
        
        """
        if self.highlight_epochs is None or len(self.highlight_epochs) == 0:
            return
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        x_width = float(max(x_max - x_min, 1e-6))
        y_width = float(max(y_max - y_min, 1e-6))
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        for idx, row in self.highlight_epochs.iterrows():
            if ('start' not in row) or ('stop' not in row):
                continue
            start_t = float(row['start'])
            stop_t = float(row['stop'])
            curr_label: str = row.get('label', f"range_{idx}")
            curr_epoch_identifier: str = f'Epoch[{curr_label}]'

            z0 = float((start_t - self.t_min) * self.z_scale)
            z1 = float((stop_t - self.t_min) * self.z_scale)
            z_low, z_high = float(min(z0, z1)), float(max(z0, z1))
            z_size = float(max(z_high - z_low, 1e-4))
            z_center = float(0.5 * (z_low + z_high))
            rgba = self._to_rgba(row.get('color', None), alpha=0.15)
            edge_rgba = (rgba[0], rgba[1], rgba[2], 0.35)

            # box = vz.Box(width=x_width, height=y_width, depth=z_depth, color=rgba, edge_color=edge_rgba, parent=self.view.scene)
            box = vz.Box(width=x_width, height=z_size, depth=y_width, color=rgba, edge_color=edge_rgba, parent=self.view.scene, name=curr_epoch_identifier)
            # box.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
            box.set_gl_state('translucent', depth_test=False, cull_face=False)
            box.order = 10
            transform = scene.transforms.MatrixTransform()
            transform.translate((x_center, y_center, z_center))
            # transform.translate((x_center, z_center, y_center))
            box.transform = transform
            self.highlight_boxes.append(box)

            label_text = curr_label
            label_pos = np.array([[x_min, y_min, z_center]], dtype=np.float32)
            # label_pos = np.array([[x_center, y_center, z_center]], dtype=np.float32) ## GOOD
            # label_pos = np.array([[x_center, y_center, z_center]], dtype=np.float32)
            
            # label_pos = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
            text_rgba = self._to_rgba(row.get('color', None), alpha=0.8)
            # text_rgba = 'white'
            # label_pos = np.array([[x_min, z_center, y_min]], dtype=np.float32)
            # label = vz.Text(text=label_text, color=edge_rgba, pos=label_pos, font_size=10, anchor_x='left', anchor_y='center', parent=self.view.scene)
            # label = vz.Text(text=label_text, color=text_rgba, pos=label_pos, font_size=12, anchor_x='left', anchor_y='bottom', parent=self.view.scene, name=f'{curr_epoch_identifier}_lbl')
            print(f'label_text: "{label_text}", label_pos: {label_pos}')
            # label = vz.Text(text=label_text, color=text_rgba, pos=label_pos, font_size=12, anchor_x='left', anchor_y='bottom', parent=self.view.scene, name=f'{curr_epoch_identifier}_lbl')
            label = vz.Text(text=label_text, color=text_rgba, pos=label_pos, rotation=-90, font_size=3000, parent=self.view.scene, name=f'{curr_epoch_identifier}_lbl')
            # label_transform = scene.transforms.MatrixTransform()
            # label_transform.translate((x_center, y_center, z_center))
            # label.transform = label_transform
            self.highlight_labels.append(label)


    # ==================================================================================================================================================================================================================================================================================== #
    # Fresh Emphasis/Rendering Helpers                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #
    def add_emphasis_volume(self, start_t: float, stop_t: Optional[float]=None, curr_label: Optional[str]=None):
        """

        Usage:
            epoch_result_for_contours: MatchingPastFuturePositionsResult = epoch_flat_mask_future_past_result[an_epoch_idx] if (epoch_flat_mask_future_past_result and an_epoch_idx < len(epoch_flat_mask_future_past_result)) else None
            per_t_bin_mask = epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask
            # np.shape(per_t_bin_mask) # (45, 63, 5)
            # epoch_result_for_contours.epoch_t_idx_col_name
            time_bin_edges = epoch_result_for_contours.decoded_epoch_result.time_bin_edges

            ## OUTPUTS: per_t_bin_mask, time_bin_edges

            unique_identifier = f'contour[{an_epoch_idx}]'
            _out_contour_img_objs[unique_identifier] = viewer_3d.add_posterior_contours(per_t_bin_mask=per_t_bin_mask, t_bin_edges=time_bin_edges, unique_identifier=unique_identifier)
                
            unique_identifier = f'vol[{an_epoch_idx}]'
            _out_contour_img_objs[unique_identifier] = viewer_3d.add_emphasis_volume(start_t=time_bin_edges[0], stop_t=time_bin_edges[-1], curr_label=unique_identifier)

            
        """
        #TODO 2026-03-19 06:20: - [ ] do it
        from vispy.color import BaseColormap
        class TransGray(BaseColormap):
            glsl_map = """
            vec4 translucent_gray(float t) {
                return vec4(t, t, t, t * 0.06);
            }
            """

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        x_width = float(max(x_max - x_min, 1e-6))
        y_width = float(max(y_max - y_min, 1e-6))
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        if stop_t is None:
            stop_t = start_t + 0.100 # 100ms default duration for an event
            
        curr_label: str = curr_label or 'UNTITLED'
        curr_epoch_identifier: str = f'Vol[{curr_label}]'

        z0 = float((start_t - self.t_min) * self.z_scale)
        z1 = float((stop_t - self.t_min) * self.z_scale)
        z_min, z_max = float(min(z0, z1)), float(max(z0, z1))
        z_size = float(max(z_max - z_min, 1e-4))
        z_center = float(0.5 * (z_min + z_max))

        # box = vz.Box(width=x_width, height=z_size, depth=y_width, color=rgba, edge_color=edge_rgba, parent=self.view.scene, name=curr_epoch_identifier)

        # 2. Generate dummy volume data (Shape: Z, Y, X)
        # Creating a 50x100x200 array
        # nz, ny, nx = 50, 100, 200
        nz, ny, nx = 3, 10, 10
        vol_data: NDArray = np.random.normal(size=(nz, ny, nx), loc=0.5, scale=0.2).astype(np.float32) # shape is (z, y, x)

        # 3. Define the target bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]
        # x_min, x_max = -100, 100
        # y_min, y_max = 0, 50
        # z_min, z_max = -25, 25

        # 4. Calculate scale factors
        scale_x = (x_max - x_min) / nx
        scale_y = (y_max - y_min) / ny
        scale_z = (z_max - z_min) / nz

        # 5. Create the Volume visual
        box_vol: scene.visuals.Volume = scene.visuals.Volume(
            vol_data,
            parent=self.view.scene,
            method='translucent',
            cmap=TransGray(), # cmap='grays'
            interpolation='linear',
            relative_step_size=0.8, name=curr_epoch_identifier,
        )
        box_vol.order = 10
        
        # transform = scene.transforms.MatrixTransform()
        # transform.translate((x_center, y_center, z_center))
        # # transform.translate((x_center, z_center, y_center))
        # box_vol.transform = transform
        
        # 6. Apply the Scale and Translate (STTransform)
        box_vol.transform = scene.transforms.STTransform(
            scale=(scale_x, scale_y, scale_z),
            # translate=(x_center, z_center, y_center), ## ceertainly wrong
            translate=(x_min, y_min, z_min),
        )
        
        self.highlight_boxes.append(box_vol)

        return box_vol


    def add_emphasis_plane(self, time_value: float, curr_label: Optional[str]=None, color: Any=None, alpha: float=0.25, edge_alpha: float=0.5, scene_parent: Optional[Any]=None):
        """Adds a flat, translucent emphasis plane (quad) at a specific time value, spanning the full arena XY extent.
            plane_mesh, edge_line
            
            unique_identifier = f'plane[{an_epoch_idx}]'
            _out_contour_img_objs[unique_identifier] = viewer_3d.add_emphasis_plane(time_value=time_bin_edges[0], color='white', curr_label=unique_identifier, alpha=0.05, edge_alpha=0.5)

        """
        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])

        curr_label: str = curr_label or 'UNTITLED'
        curr_epoch_identifier: str = f'Plane[{curr_label}]'
        plane_parent = self.view.scene if scene_parent is None else scene_parent

        z_val = float((float(time_value) - self.t_min) * self.z_scale)

        rgba = self._to_rgba(color, alpha=alpha)
        edge_rgba = (rgba[0], rgba[1], rgba[2], float(edge_alpha))

        # 1. Build quad vertices (four corners of the arena at z_val)
        vertices = np.array([[x_min, y_min, z_val], [x_max, y_min, z_val], [x_max, y_max, z_val], [x_min, y_max, z_val]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        # 2. Create the filled Mesh plane visual
        plane_mesh = vz.Mesh(vertices=vertices, faces=faces, color=rgba, parent=plane_parent, name=curr_epoch_identifier)
        plane_mesh.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
        plane_mesh.order = 10

        # 3. Create an edge outline around the plane for visibility
        edge_positions = np.array([[x_min, y_min, z_val], [x_max, y_min, z_val], [x_max, y_max, z_val], [x_min, y_max, z_val], [x_min, y_min, z_val]], dtype=np.float32)
        edge_line = vz.Line(pos=edge_positions, color=edge_rgba, width=1.5, parent=plane_parent, name=f'{curr_epoch_identifier}_edge')
        edge_line.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
        edge_line.order = 11

        self.highlight_boxes.append(plane_mesh)
        self.highlight_boxes.append(edge_line)

        identifier = str(curr_label)
        self.emphasis_planes_by_key[identifier] = {'unique_identifier': identifier, 'plane_mesh': plane_mesh, 'edge_line': edge_line, 'visible': True}

        return (plane_mesh, edge_line)


    def get_emphasis_plane(self, unique_identifier: str) -> Optional[Dict[str, Any]]:
        return self.emphasis_planes_by_key.get(str(unique_identifier), None)


    def list_emphasis_plane_keys(self) -> List[str]:
        return list(self.emphasis_planes_by_key.keys())


    def set_emphasis_plane_visibility(self, unique_identifier: str, is_visible: bool) -> bool:
        item = self.get_emphasis_plane(unique_identifier=unique_identifier)
        if item is None:
            return False
        item['visible'] = bool(is_visible)
        plane_mesh = item.get('plane_mesh', None)
        if plane_mesh is not None:
            plane_mesh.visible = bool(is_visible)
        edge_line = item.get('edge_line', None)
        if edge_line is not None:
            edge_line.visible = bool(is_visible)
        return True


    def remove_emphasis_plane(self, unique_identifier: str) -> bool:
        """Detach and remove a registered emphasis plane (mesh + edge) from the scene."""
        item = self.emphasis_planes_by_key.pop(str(unique_identifier), None)
        if item is None:
            return False
        plane_mesh = item.get('plane_mesh', None)
        edge_line = item.get('edge_line', None)
        if plane_mesh is not None:
            plane_mesh.parent = None
        if edge_line is not None:
            edge_line.parent = None
        self.highlight_boxes = [x for x in self.highlight_boxes if x is not plane_mesh and x is not edge_line]
        self._refresh_scene_tree()
        return True


    # ==================================================================================================================================================================================================================================================================================== #
    # Epoch Group Management                                                                                                                                                                                                                                                               #
    # ==================================================================================================================================================================================================================================================================================== #

    @property
    def n_epoch_groups(self) -> int:
        if self.epoch_flat_mask_future_past_result is not None:
            return len(self.epoch_flat_mask_future_past_result)
        return len(self.epoch_visual_groups)


    def _iter_epoch_nav_indices(self) -> List[int]:
        if self.epoch_flat_mask_future_past_result is not None:
            return list(range(len(self.epoch_flat_mask_future_past_result)))
        return sorted(self.epoch_visual_groups.keys())


    def _get_or_create_epoch_group_node(self, epoch_idx: int) -> Any:
        epoch_idx = int(epoch_idx)
        existing = self.epoch_group_nodes.get(epoch_idx, None)
        if existing is not None:
            return existing
        node = scene.Node(parent=self.view.scene, name=f'Epoch {epoch_idx}')
        self.epoch_group_nodes[epoch_idx] = node
        return node


    def _get_or_create_single_slot_epoch_node(self, epoch_idx: int) -> Any:
        """Parent node for single-slot epoch overlays; name tracks the logical epoch index."""
        epoch_idx = int(epoch_idx)
        if self.epoch_single_slot_parent_node is None:
            self.epoch_single_slot_parent_node = scene.Node(parent=self.view.scene, name=f'Epoch {epoch_idx}')
        else:
            self.epoch_single_slot_parent_node.name = f'Epoch {epoch_idx}'
        return self.epoch_single_slot_parent_node


    def _clear_active_slot_visuals(self) -> None:
        """Remove single-slot contour, extrusions, emphasis plane, and match lines from the scene."""
        self._clear_match_line_visuals()
        self._match_lines_built_for_epoch_idx = None
        if self.get_posterior_contours(_SINGLE_SLOT_CONTOUR_ID) is not None:
            self.remove_posterior_contours(_SINGLE_SLOT_CONTOUR_ID)
        if self.get_emphasis_plane(_SINGLE_SLOT_PLANE_ID) is not None:
            self.remove_emphasis_plane(_SINGLE_SLOT_PLANE_ID)
        self._single_slot_rendered_epoch_idx = None


    def _add_epoch_visuals_single_slot(self, epoch_idx: int, per_t_bin_mask, t_bin_edges, extrude: bool, contour_kwargs: Optional[Dict[str, Any]], plane_kwargs: Optional[Dict[str, Any]], extrusion_kwargs: Optional[Dict[str, Any]]) -> None:
        """Build contours (+ optional extrusion) + emphasis plane under fixed keys; does not register legacy epoch_visual_groups."""
        contour_kw = dict(line_width=2.0, contour_alpha=0.7, level=0.5, fill=True, fill_alpha=0.25)
        if contour_kwargs is not None:
            contour_kw.update(contour_kwargs)
        plane_kw = dict(color='white', alpha=0.05, edge_alpha=0.5)
        if plane_kwargs is not None:
            plane_kw.update(plane_kwargs)
        epoch_node = self._get_or_create_single_slot_epoch_node(epoch_idx)
        self.add_posterior_contours(per_t_bin_mask=per_t_bin_mask, t_bin_edges=t_bin_edges, unique_identifier=_SINGLE_SLOT_CONTOUR_ID, scene_parent=epoch_node, **contour_kw)
        if extrude:
            _ext = dict(z_half_extent=None, tube_radius=1.2, tube_alpha=0.3, wall_alpha=0.15)
            if extrusion_kwargs is not None:
                _ext.update(extrusion_kwargs)
            self.build_contour_extrusions(unique_identifier=_SINGLE_SLOT_CONTOUR_ID, z_half_extent=_ext.get('z_half_extent', None), tube_radius=float(_ext.get('tube_radius', 1.2)), tube_alpha=float(_ext.get('tube_alpha', 0.3)), wall_alpha=float(_ext.get('wall_alpha', 0.15)))
        self.add_emphasis_plane(time_value=float(t_bin_edges[0]), curr_label=_SINGLE_SLOT_PLANE_ID, scene_parent=epoch_node, **plane_kw)
        if self.show_match_lines and self._epoch_mask_datasource is not None:
            self._build_match_lines_for_epoch(epoch_idx, epoch_node)


    def set_epoch_visual_source(self, epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], *, extrude: bool = False, contour_kwargs: Optional[Dict[str, Any]] = None, plane_kwargs: Optional[Dict[str, Any]] = None, extrusion_kwargs: Optional[Dict[str, Any]] = None, initial_epoch_idx: int = 0, filter_epochs: Optional[pd.DataFrame] = None, minimum_included_matching_sequence_length: Optional[int] = None, show_match_lines: bool = True) -> None:
        """Store per-epoch `MatchingPastFuturePositionsResult` list and show one epoch at a time (rebuild on change).

        Replaces the pattern of looping `add_epoch_visuals` for every epoch and toggling visibility.
        CPU cost: meshes are rebuilt when switching epochs; memory: only one epoch's scene objects exist.

        filter_epochs: optional decode filter epochs DataFrame aligned 1:1 with `epoch_flat_mask_future_past_result`.
            If omitted, a dummy index-only frame is used (same masking as the 2D viewer still applies via `MaskDataSource`).
        minimum_included_matching_sequence_length: passed to `MaskDataSource._prepare_epoch_data` when building match lines.
        show_match_lines: if False, only contours/plane/extrusion are drawn for each epoch.

        Usage:

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import MatchingPastFuturePositionsResult

            ## INPUTS: _container_container
            decoder_name = 'roam'
            container = _container_container.masked_container

            epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult] = (
                container.debug_computed_dict
                .get(decoder_name, {})
                .get("prominence_future_past_analysis", {})
                .get("_out_epoch_flat_mask_future_past_result")
            )
            assert epoch_flat_mask_future_past_result is not None

            viewer_3d.set_epoch_visual_source(epoch_flat_mask_future_past_result, extrude=True, extrusion_kwargs={"tube_radius": 1.1, "tube_alpha": 0.6}, initial_epoch_idx=35, filter_epochs=a_decoded_filter_epochs_df)
            viewer_3d.set_epoch_visual_source(epoch_flat_mask_future_past_result, show_match_lines=False)


        """
        self.epoch_flat_mask_future_past_result = epoch_flat_mask_future_past_result
        self.epoch_visual_extrude = bool(extrude)
        self.epoch_visual_contour_kwargs = contour_kwargs
        self.epoch_visual_plane_kwargs = plane_kwargs
        self.epoch_visual_extrusion_kwargs = extrusion_kwargs
        self.minimum_included_matching_sequence_length = minimum_included_matching_sequence_length
        self.show_match_lines = bool(show_match_lines)
        self._epoch_mask_datasource = None
        self.epoch_visual_filter_epochs = None
        n = len(epoch_flat_mask_future_past_result)
        if n > 0:
            fe = filter_epochs
            if fe is not None and len(fe) != n:
                warn(f"filter_epochs length {len(fe)} != epoch list length {n}; using dummy filter_epochs", stacklevel=2)
                fe = None
            if fe is None:
                fe = self._make_dummy_filter_epochs(n)
            self.epoch_visual_filter_epochs = fe
            self._epoch_mask_datasource = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result, fe)
        self._update_epoch_slider_range()
        if n == 0:
            self.set_active_epoch(None)
            return
        idx = max(0, min(n - 1, int(initial_epoch_idx)))
        self.set_active_epoch(idx)


    def register_epoch_visual(self, epoch_idx: int, visual_type: str, unique_identifier: str):
        epoch_idx = int(epoch_idx)
        group = self.epoch_visual_groups.get(epoch_idx, None)
        if group is None:
            group = {'contour_keys': [], 'plane_keys': []}
            self.epoch_visual_groups[epoch_idx] = group
        key_list_name = 'contour_keys' if visual_type == 'contour' else 'plane_keys'
        if unique_identifier not in group[key_list_name]:
            group[key_list_name].append(unique_identifier)
        self._update_epoch_slider_range()


    def add_epoch_visuals(self, epoch_idx: int, per_t_bin_mask, t_bin_edges, contour_kwargs: Optional[Dict] = None, plane_kwargs: Optional[Dict] = None, extrude: bool = False, extrusion_kwargs: Optional[Dict] = None) -> Dict[str, str]:
        contour_kw = dict(line_width=2.0, contour_alpha=0.7, level=0.5, fill=True, fill_alpha=0.25)
        if contour_kwargs is not None:
            contour_kw.update(contour_kwargs)
        plane_kw = dict(color='white', alpha=0.05, edge_alpha=0.5)
        if plane_kwargs is not None:
            plane_kw.update(plane_kwargs)

        epoch_node = self._get_or_create_epoch_group_node(epoch_idx)
        contour_id = f'contour[{epoch_idx}]'
        self.add_posterior_contours(per_t_bin_mask=per_t_bin_mask, t_bin_edges=t_bin_edges, unique_identifier=contour_id, scene_parent=epoch_node, **contour_kw)
        self.register_epoch_visual(epoch_idx=epoch_idx, visual_type='contour', unique_identifier=contour_id)

        if extrude:
            _ext = dict(z_half_extent=None, tube_radius=1.2, tube_alpha=0.3, wall_alpha=0.15)
            if extrusion_kwargs is not None:
                _ext.update(extrusion_kwargs)
            self.build_contour_extrusions(unique_identifier=contour_id, z_half_extent=_ext.get('z_half_extent', None), tube_radius=float(_ext.get('tube_radius', 1.2)), tube_alpha=float(_ext.get('tube_alpha', 0.3)), wall_alpha=float(_ext.get('wall_alpha', 0.15)))

        plane_id = f'plane[{epoch_idx}]'
        self.add_emphasis_plane(time_value=float(t_bin_edges[0]), curr_label=plane_id, scene_parent=epoch_node, **plane_kw)
        self.register_epoch_visual(epoch_idx=epoch_idx, visual_type='plane', unique_identifier=plane_id)

        return {'contour': contour_id, 'plane': plane_id}


    def _set_active_epoch_single_source(self, epoch_idx: Optional[int]) -> None:
        """Single-slot mode: remove previous overlay meshes and rebuild for `epoch_idx` (or clear if None / empty list)."""
        src = self.epoch_flat_mask_future_past_result
        assert src is not None
        n = len(src)
        if epoch_idx is not None:
            epoch_idx = int(epoch_idx)
        self.active_epoch_idx = epoch_idx
        if n == 0:
            self._clear_active_slot_visuals()
            if self.epoch_slider is not None:
                self.epoch_slider.blockSignals(True)
                self.epoch_slider.setValue(0)
                self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText("0/0")
            self._refresh_scene_tree()
            if self.canvas is not None:
                self.canvas.update()
            return
        if epoch_idx is None:
            self._clear_active_slot_visuals()
            if self.epoch_slider is not None:
                self.epoch_slider.blockSignals(True)
                self.epoch_slider.setValue(0)
                self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText(f"—/{max(0, n - 1)}")
            self._refresh_scene_tree()
            if self.canvas is not None:
                self.canvas.update()
            return
        if epoch_idx < 0 or epoch_idx >= n:
            epoch_idx = max(0, min(n - 1, epoch_idx))
        need_match_lines = self.show_match_lines and self._epoch_mask_datasource is not None
        match_lines_ok = (not need_match_lines) or (self._match_lines_built_for_epoch_idx == epoch_idx)
        if self._single_slot_rendered_epoch_idx == epoch_idx and self.get_posterior_contours(_SINGLE_SLOT_CONTOUR_ID) is not None and match_lines_ok:
            if self.epoch_slider is not None and self.epoch_slider.value() != epoch_idx:
                self.epoch_slider.blockSignals(True)
                self.epoch_slider.setValue(epoch_idx)
                self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText(f"{epoch_idx}/{max(0, n - 1)}")
            self._refresh_scene_tree()
            if self.canvas is not None:
                self.canvas.update()
            return
        self._clear_active_slot_visuals()
        er = src[epoch_idx]
        per_t_bin_mask = er.epoch_t_bins_high_prob_pos_mask
        t_bin_edges = er.decoded_epoch_result.time_bin_edges
        self._add_epoch_visuals_single_slot(epoch_idx, per_t_bin_mask, t_bin_edges, self.epoch_visual_extrude, self.epoch_visual_contour_kwargs, self.epoch_visual_plane_kwargs, self.epoch_visual_extrusion_kwargs)
        self._single_slot_rendered_epoch_idx = epoch_idx
        if self.epoch_slider is not None and self.epoch_slider.value() != epoch_idx:
            self.epoch_slider.blockSignals(True)
            self.epoch_slider.setValue(epoch_idx)
            self.epoch_slider.blockSignals(False)
        if self.epoch_value_label is not None:
            self.epoch_value_label.setText(f"{epoch_idx}/{max(0, n - 1)}")
        self._refresh_scene_tree()
        if self.canvas is not None:
            self.canvas.update()


    def set_active_epoch(self, epoch_idx: Optional[int]):
        if self.epoch_flat_mask_future_past_result is not None:
            self._set_active_epoch_single_source(epoch_idx)
            return
        if epoch_idx is not None:
            epoch_idx = int(epoch_idx)
        self.active_epoch_idx = epoch_idx
        sorted_indices = sorted(self.epoch_visual_groups.keys())
        for eidx in sorted_indices:
            group = self.epoch_visual_groups[eidx]
            is_active = (epoch_idx is None) or (eidx == epoch_idx)
            for key in group.get('contour_keys', []):
                self.set_posterior_contours_visibility(unique_identifier=key, is_visible=is_active)
            for key in group.get('plane_keys', []):
                self.set_emphasis_plane_visibility(unique_identifier=key, is_visible=is_active)
        if self.epoch_slider is not None and epoch_idx is not None:
            if self.epoch_slider.value() != epoch_idx:
                self.epoch_slider.blockSignals(True)
                self.epoch_slider.setValue(epoch_idx)
                self.epoch_slider.blockSignals(False)
        if self.epoch_value_label is not None:
            n = self.n_epoch_groups
            lbl = f"{epoch_idx}/{max(0, n - 1)}" if epoch_idx is not None else f"all/{max(0, n - 1)}"
            self.epoch_value_label.setText(lbl)
        self._refresh_scene_tree()
        if self.canvas is not None:
            self.canvas.update()


    def _update_epoch_slider_range(self):
        if self.epoch_slider is None:
            return
        n = self.n_epoch_groups
        self.epoch_slider.setMaximum(max(0, n - 1))
        self.epoch_slider.setEnabled(n > 0)


    def on_epoch_slider_value_changed(self, value: int):
        self.set_active_epoch(int(value))


    # ==================================================================================================================================================================================================================================================================================== #
    # Interaction/UI Events                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #
    def on_close(self, event):
        print(f'Volumentric2DTimeSeriesPlotter.on_close(event: {event}): closing!')

    def on_resize(self, event):
        print(f'Volumentric2DTimeSeriesPlotter.on_resize(event:(event.size: {event.size})')
        print(f'\tevent.physical_size: {event.physical_size}')
        # Tell the OpenGL context the new physical dimensions
        vp = (0, 0, event.physical_size[0], event.physical_size[1])
        self.context.set_viewport(*vp)
        
        # If you have custom visuals, you may also need to update transforms:
        # self.visual.transforms.configure(canvas=self, viewport=vp)
        

    def on_key_press(self, event):
        key_name = str(event.key)
        if key_name == 'Shift':
            self.debug_is_shift_hover_enabled = True
            return
        if key_name in {'Up', 'Down'} and self.n_epoch_groups > 0:
            sorted_indices = self._iter_epoch_nav_indices()
            if self.active_epoch_idx is None:
                next_epoch = sorted_indices[0] if key_name == 'Down' else sorted_indices[-1]
            else:
                try:
                    cur_pos = sorted_indices.index(self.active_epoch_idx)
                except ValueError:
                    cur_pos = 0
                step = 1 if key_name == 'Down' else -1
                next_pos = max(0, min(len(sorted_indices) - 1, cur_pos + step))
                next_epoch = sorted_indices[next_pos]
            self.set_active_epoch(next_epoch)
            return
        if key_name not in {'Left', 'Right'}:
            return
        if self.n_t_bins <= 0:
            return
        step = -1 if key_name == 'Left' else 1
        next_idx = self._clamp_t_bin_idx(self.active_t_bin_idx + step)
        if self.t_bin_slider is not None:
            self.t_bin_slider.setValue(next_idx)
        else:
            self.update_active_t_bin(next_idx)


    def on_key_release(self, event):
        key_name = str(event.key)
        if key_name != 'Shift':
            return
        self.debug_is_shift_hover_enabled = False
        self._hide_debug_crosshairs()
        if self.canvas is not None:
            self.canvas.update()


    def on_mouse_move(self, event):
        if self.canvas is None:
            return
        if self.pos3d is None:
            return
        if not (self.debug_is_shift_hover_enabled or self._event_has_shift_modifier(event)):
            if self.debug_nearest_pos3d_idx is not None:
                self._hide_debug_crosshairs()
                self.canvas.update()
            return
        event_pos = getattr(event, 'pos', None)
        if event_pos is None:
            return
        canvas_pos = np.asarray(event_pos, dtype=np.float32)
        if np.shape(canvas_pos)[0] < 2:
            return
        nearest_idx = self._nearest_pos3d_idx_from_canvas_pos(canvas_pos=canvas_pos)
        if nearest_idx is None:
            if self.debug_nearest_pos3d_idx is not None:
                self._hide_debug_crosshairs()
                self.canvas.update()
            return
        self.debug_nearest_pos3d_idx = int(nearest_idx)
        self._set_debug_crosshairs_position(center_xyz=self.pos3d[self.debug_nearest_pos3d_idx, :])
        self.canvas.update()


    def on_mouse_leave(self, event):
        self._hide_debug_crosshairs()
        if self.canvas is not None:
            self.canvas.update()


    def on_slider_value_changed(self, value: int):
        self.update_active_t_bin(value)
