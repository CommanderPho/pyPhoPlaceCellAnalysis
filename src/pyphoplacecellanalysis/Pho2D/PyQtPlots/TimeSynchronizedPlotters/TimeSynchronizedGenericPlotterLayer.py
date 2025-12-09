from neuropy.core.user_annotations import function_attributes
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from attrs import define, field, Factory
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr, shape_only_repr

# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphocorehelpers.assertion_helpers import Assert
from neuropy.utils.mixins.indexing_helpers import get_dict_subset

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.AnimalTrajectoryPlottingMixin import AnimalTrajectoryPlottingMixin
from attrs import define, field, Factory
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.UserEditableROIMixin import UserEditableROIMixin, Rois

import param
import panel as pn
from panel.viewable import Viewer
from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import BasePlotDataParams, ExtendedPlotDataParams


class LayerDisplayConfig(BasePlotDataParams):
    """ This class uses the 'param' library to observe changes to its members 
    and perform corresponding updates to the class that holds it when they happen.
    """
    # Overriding defaults from parent
    name = param.String(default='SessionEpochs', doc='Name of the layer')
    isVisible = param.Boolean(default=True, doc="Whether the layer is visible")
    cmap = param.String(default='matplotlib.jet', doc='The cmap to use')
    opacity = param.Number(default=0.5, bounds=(0.0, 1.0), step=0.1)   
    drop_below_threshold = param.Number(default=1e-3, bounds=(1e-27, 0.1), step=1e-3)

    
    @staticmethod
    def _config_update_watch_labels():
        """Returns list of parameter names that trigger full updates"""
        return ['cmap', 'opacity', 'drop_below_threshold', 'isVisible']
    
    @staticmethod
    def _config_visibility_watch_labels():
        """Returns list of parameter names that trigger visibility updates"""
        return ['isVisible']
    

    # a_stack_item.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map


@define(slots=False)
class TimeSynchronizedGenericPlotterLayer:
    """ A lightweight component (layer) to add to an existing PyQtGraphp lot hierarchy. The layer renders something (generic) at a given moment in time. 
    Uses pyqtgraph to render the relevant items
    Its inherited `self.on_window_changed_rate_limited(...)` is called to perform updates

    Contains, for example, a single `pg.ImageItem` as its contents or something


    Usage

        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedGenericPlotterLayer import TimeSynchronizedGenericPlotterLayer, LayerDisplayConfig

# an_epoch_name: str = 'roam'
# an_epoch_idx: int = 0 ## 0 or 1

an_epoch_name: str = 'sprinkle'
an_epoch_idx: int = 1 ## 0 or 1
a_plotter = sync_plotters[an_epoch_name]
a_plotter
        a_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name=f"{an_epoch_name}_hist", parent=a_plotter, contents={}, data={'time_window_centers': deepcopy(time_window_centers),
                                                                                                                                                            'main': deepcopy(a_moving_avg), 
                                                                                                                                                            #    'pos_df': deepcopy(pos_df),
                                                                                                                                                    })
        a_layer

    """
    name: str = field()
    parent: TimeSynchronizedPlotterBase = field(repr=keys_only_repr)
    contents: Dict[str, Any] = field(default=Factory(Dict), repr=keys_only_repr)
    data: Optional[Dict] = field(default=None, repr=keys_only_repr)
    _params: Optional[VisualizationParameters] = field(default=None)
    gui_params: LayerDisplayConfig = field(default=None, init=False)
    last_window_time: float = field(default=None, init=False)
    last_window_index: int = field(default=None, init=False)


    @property
    def is_layer(self) -> bool:
        """The is_layer property."""
        return True

    @property
    def time_window_centers(self):
        """The time_window_centers property."""
        time_window_centers = (self.data or {}).get('time_window_centers', None)
        if time_window_centers is not None:
            return time_window_centers
        else:
            ## return parent's time_window_centers and assume they're the same:
            return self.parent.time_window_centers # get time window centers (n_time_window_centers,)
    

    @property
    def last_t(self) -> float:
        """for AnimalTrajectoryPlottingMixin"""
        return (self.last_window_time) or 0.0
    

    @property
    def params(self) -> VisualizationParameters:
        """The params property."""
        a_params = getattr(self, '_params', None)
        if a_params is not None:
            return a_params
        else:
            return self.parent.params
    @params.setter
    def params(self, value: VisualizationParameters):
        self._params = value
        

    # ==================================================================================================================================================================================================================================================================================== #
    # Init                                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    def __attrs_post_init__(self):
        self.setup()
        self._buildGraphics()

    def setup(self):
        """ will setup params from parent's self.params 
        """
        assert self.parent is not None
        assert self.parent.params is not None
        # self.params = deepcopy(self.parent.params) ## #TODO 2025-12-09 14:49: - [ ] Ideally we'd only copy the relevant params from the parent
        relevant_only_params_keys = ['name', 'cmap', 'image_margins', 'image_bounds_extent', 'x_range', 'y_range', 'debug_print', 'shared_axis_order', 'drop_below_threshold']
        parent_params = get_dict_subset(self.parent.params.to_dict(), subset_includelist=relevant_only_params_keys) 
        parent_params['name'] = f"{parent_params['name']}_{self.name}" ## append "_{self.name}" to parent's name
        self.params = VisualizationParameters.init_from_dict(parent_params)
        self.params.opacity = 1.0
        
        self.gui_params = LayerDisplayConfig()
        self.gui_params.cmap = f'matplotlib.{self.params.cmap.name}'
        if self.params.drop_below_threshold is not None:
            self.gui_params.drop_below_threshold = self.params.drop_below_threshold
        else:
            self.gui_params.drop_below_threshold = 1e-90 ## insanely small so it's effectively None
            

        # Setup watchers in your implementor class:
        self.gui_params.param.watch(
            self.on_gui_params_update, 
            LayerDisplayConfig._config_update_watch_labels(), 
            queued=True
        )

        # self.gui_params.param.watch(
        #     self.on_gui_params_update, 
        #     LayerDisplayConfig._config_visibility_watch_labels(), 
        #     queued=True
        # )


    def _buildGraphics(self):
        ## More Involved Mode:
        imv = self.contents.get('main', None)
        if imv is None:
            ## create one if it doesn't exist:
            self.contents['main'] = pg.ImageItem()
            imv: pg.ImageItem = self.contents['main']
            # add ImageItem to parent's PlotItem
            self.parent.ui.root_plot.addItem(imv, defaultPadding=0.0)  
        else:
            print(f'WARNING: already had main imv in self.contents: {self.contents}')            

        if self.name not in self.parent.ui.plot_stack:
            print(f'adding to parent plot_stack with name "{self.name}"')
            self.parent.ui.plot_stack[self.name] = self
        else:
            print(f'WARNING: item already exists in parent plot_stack with name "{self.name}".\n\tself.parent.ui.plot_stack.keys(): {list(self.parent.ui.plot_stack.keys())}. Continuing and assuming it was already added elsewhere.')
        
        # Set the color map:
        imv.setColorMap(self.params.cmap)


    def on_gui_params_update(self, updated_params):
        """ called when the user changes the GUI params object. Needs to update the internal self.params
        """
        print(f'on_gui_params_update(updated_params: {updated_params}):')
        did_change: bool = False
        
        cmap_str: str = self.gui_params.cmap
        if '.' in cmap_str:
            ## split
            src_path, src_rel_name = cmap_str.split('.', maxsplit=1)
            print(f'src_path: {src_path}, src_rel_name: {src_rel_name}')
            cmap = pg.colormap.get(src_rel_name, src_path)
        else:
            cmap = pg.colormap.get(cmap_str) ## specifically named colormap
        did_change = did_change or (self.params.cmap != cmap)
        self.params.cmap = cmap

        drop_below_threshold: float = self.gui_params.drop_below_threshold
        if drop_below_threshold < 1e-89:
            drop_below_threshold = None
            
        did_change = did_change or (self.params.drop_below_threshold != drop_below_threshold)
        
        opacity: float = self.gui_params.opacity
        did_change = did_change or (self.params.opacity != opacity)
        self.params.opacity = opacity

        isVisible: bool = self.gui_params.isVisible
        prev_was_visible: bool = (self.params.opacity > 0.0)
        did_change = did_change or (prev_was_visible != isVisible)        
        # if isVisible:
        #     did_change = did_change or (self.params.opacity > 0.0)        
        # else:
        #     did_change = did_change or (prev_was_visible != isVisible)
            
        if did_change:
            print(f'\tchange occured!')
            self.on_params_update()
            
        print('\tdone.')



    def on_params_update(self):
        """ updates """
        print(f'on_params_update():')
        self.contents['main'].setColorMap(self.params.cmap)
        self.contents['main'].setOpacity(self.params.opacity)
        print('\tdone.')

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #

    def on_window_changed(self, start_t, end_t, defer_render=True):
        # called when the window is updated
        if self.params.debug_print:
            print(f'TimeSynchronizedGenericPlotterLayer.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        self.update(start_t, defer_render=defer_render)
        if self.params.debug_print:
            print('\tFinished calling _update_plots()')


    def update(self, t, defer_render=True):
        # Finds the nearest previous decoded position for the time t:
        self.last_window_index = np.searchsorted(self.time_window_centers, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        self.last_window_time = self.time_window_centers[self.last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        # Update the plots:
        if not defer_render:
            self._update_plots()
            pass


    def _update_plots(self):
        if self.params.debug_print:
            print(f'TimeSynchronizedGenericPlotterLayer._update_plots()')
            
        # Update the existing one:
        
        # Update the plots:
        curr_time_window_index = self.last_window_index
        curr_t = self.last_window_time
        
        if (curr_time_window_index is None) or (curr_t is None):
            print(f'WARN: TimeSynchronizedGenericPlotterLayer._update_plots: curr_time_window_index: {curr_time_window_index}')
            return # return without updating
        
        main_data: NDArray = self.data['main']
        
        # assert np.shape(main_data)[-1] == len(
        image = np.squeeze(main_data[:, :, curr_time_window_index]).copy()
        # image_title = f'{self.name}'
    
        if self.params.drop_below_threshold is not None:
            image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the low values if needed
        

        ## get the image item to draw:
        imv: pg.ImageItem = self.contents['main']

        # self.ui.imv.setImage(image, xvals=self.active_time_dependent_placefields.xbin)
        if self.params.shared_axis_order is None:
            imv.setImage(image, rect=self.params.image_bounds_extent)
        else:
            imv.setImage(image, rect=self.params.image_bounds_extent, axisOrder=self.params.shared_axis_order)
        
        


    # @function_attributes(short_name=None, tags=['video', 'export', 'mp4', 'avi', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-11-24 23:09', related_items=[])
    # def export_video(self, output_path: str, start_t: Optional[float] = None, end_t: Optional[float] = None, fps: float = 30.0, width: Optional[int] = None, height: Optional[int] = None, progress_print: bool = True, debug_print: bool = False):
    #     """Efficiently export a video from the TimeSynchronizedGenericPlotterLayer instance (faster than real-time playback)
        
    #     This method iterates through time points, updates the plotter, captures frames using
    #     pyqtgraph's ImageExporter, and saves them as a video using OpenCV.
        
    #     Args:
    #         output_path: Path to save the output video file (e.g., 'output/videos/decoder_video.avi')
    #         start_t: Start time for video export. If None, uses the first available time window center.
    #         end_t: End time for video export. If None, uses the last available time window center.
    #         fps: Frames per second for the output video (default: 30.0)
    #         width: Width of exported frames in pixels. If None, uses current widget width.
    #         height: Height of exported frames in pixels. If None, uses current widget height.
    #         progress_print: Whether to print progress messages (default: True)
    #         debug_print: Whether to print debug information (default: False)
            
    #     Returns:
    #         Path: Path to the saved video file
            
    #     Usage:
    #         plotter = TimeSynchronizedGenericPlotterLayer(...)
    #         video_path = plotter.export_video('output/videos/decoder.avi', start_t=100.0, end_t=200.0, fps=30.0)
    #     """
    #     from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
    #     from pyphoplacecellanalysis.External.pyqtgraph import functions as fn
    #     from pathlib import Path
    #     import cv2
    #     import sys
        
    #     # Get time window centers
    #     time_window_centers = self.time_window_centers
    #     if len(time_window_centers) == 0:
    #         raise ValueError("No time window centers available for video export")
        
    #     # Determine time range
    #     if start_t is None:
    #         start_t = float(time_window_centers[0])
    #     if end_t is None:
    #         end_t = float(time_window_centers[-1])
        
    #     # Find valid time indices
    #     start_idx = np.searchsorted(time_window_centers, start_t, side='left')
    #     end_idx = np.searchsorted(time_window_centers, end_t, side='right')
        
    #     if start_idx >= end_idx:
    #         raise ValueError(f"Invalid time range: start_t={start_t}, end_t={end_t}. No valid frames found.")
        
    #     # Subsample frame indices based on fps to reduce processing
    #     # Calculate desired time step between frames (in seconds)
    #     desired_time_step = 1.0 / fps if fps > 0 else float('inf')
        
    #     # Get all candidate frame indices
    #     all_frame_indices = np.arange(start_idx, end_idx)
    #     all_frame_times = time_window_centers[all_frame_indices]
        
    #     # Subsample frames based on desired time step
    #     if desired_time_step < float('inf') and len(all_frame_indices) > 1:
    #         # Start with the first frame
    #         subsampled_indices = [all_frame_indices[0]]
    #         last_selected_time = all_frame_times[0]
            
    #         # Select frames that are at least desired_time_step apart
    #         for i in range(1, len(all_frame_indices)):
    #             current_time = all_frame_times[i]
    #             time_since_last = current_time - last_selected_time
                
    #             if time_since_last >= desired_time_step:
    #                 subsampled_indices.append(all_frame_indices[i])
    #                 last_selected_time = current_time
            
    #         frame_indices = np.array(subsampled_indices)
    #     else:
    #         # If fps is 0 or invalid, use all frames
    #         frame_indices = all_frame_indices
        
    #     n_frames: int = len(frame_indices)
    #     if n_frames == 0:
    #         raise ValueError(f"No frames to export after subsampling at {fps} fps")
        
    #     if progress_print:
    #         total_available_frames = len(all_frame_indices)
    #         print(f'Exporting video: {n_frames} frames (from {total_available_frames} available) from t={time_window_centers[start_idx]:.2f} to t={time_window_centers[end_idx-1]:.2f} at {fps} fps')
        
    #     # Get widget dimensions
    #     if width is None or height is None:
    #         widget_size = self.ui.root_graphics_layout_widget.size()
    #         if width is None:
    #             width = widget_size.width()
    #         if height is None:
    #             height = widget_size.height()
        
    #     # Disable debug printing during export for performance
    #     original_debug_print = self.params.debug_print
    #     self.params.debug_print = debug_print
        
    #     # Helper to convert QImage to BGR array for OpenCV (contiguous uint8 for compatibility)
    #     def qimage_to_bgr(qimage):
    #         img_array = fn.ndarray_from_qimage(qimage)
    #         # Handle ARGB32 format conversion based on byte order
    #         if img_array.shape[2] == 4:
    #             # ARGB32 format - extract RGB channels based on byte order
    #             if sys.byteorder == 'little':
    #                 # Little-endian: channels are [B, G, R, A] in memory
    #                 bgr = img_array[:, :, :3]  # B, G, R (first 3 channels)
    #             else:
    #                 # Big-endian: channels are [A, R, G, B] in memory
    #                 bgr = img_array[:, :, [3, 2, 1]]  # B, G, R from indices 3,2,1
    #         elif img_array.shape[2] == 3:
    #             # Already RGB format, convert to BGR for OpenCV
    #             bgr = img_array[:, :, ::-1]
    #         else:
    #             raise ValueError(f"Unexpected image format with {img_array.shape[2]} channels")
    #         # Ensure contiguous uint8 array for OpenCV compatibility
    #         return np.ascontiguousarray(bgr, dtype=np.uint8)
        
    #     out = None  # Initialize to None for proper cleanup
    #     try:
    #         # Create ImageExporter for the root plot
    #         exporter = ImageExporter(self.ui.root_plot)
    #         exporter.parameters()['width'] = width
    #         exporter.parameters()['height'] = height
    #         exporter.parameters()['antialias'] = True
            
    #         # Process events to ensure widget is rendered
    #         QtWidgets.QApplication.processEvents()
            
    #         # Capture first frame to get actual output dimensions (may differ from requested)
    #         first_frame_idx = frame_indices[0]
    #         self.update(time_window_centers[first_frame_idx], defer_render=False)
    #         QtWidgets.QApplication.processEvents()
    #         first_qimage = exporter.export(toBytes=True)
    #         first_bgr = qimage_to_bgr(first_qimage)
    #         actual_height, actual_width = first_bgr.shape[:2]
    #         del first_qimage  # Free memory
            
    #         # Set up output path and directory
    #         video_filepath = Path(output_path).resolve()
    #         video_parent_path = video_filepath.parent
    #         if not video_parent_path.exists():
    #             if progress_print:
    #                 print(f'Creating output directory: {video_parent_path}')
    #             video_parent_path.mkdir(parents=True, exist_ok=True)
            
    #         # Initialize video writer with actual frame dimensions
    #         fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #         out = cv2.VideoWriter(str(video_filepath), fourcc, fps, (actual_width, actual_height), isColor=True)
            
    #         if not out.isOpened():
    #             raise RuntimeError(f"Failed to open video writer for {video_filepath}")
            
    #         # Write the first frame we already captured
    #         out.write(first_bgr)
    #         del first_bgr  # Free memory
            
    #         # Streaming capture and write: process remaining frames one at a time
    #         progress_print_every_n_frames = max(1, n_frames // 20)
    #         for i, frame_idx in enumerate(frame_indices):
    #             if i == 0:
    #                 # First frame already written
    #                 if progress_print:
    #                     print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
    #                 continue
                    
    #             if progress_print and (i % progress_print_every_n_frames == 0 or i == n_frames - 1):
    #                 print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                
    #             # Update plotter to current time
    #             t = time_window_centers[frame_idx]
    #             self.update(t, defer_render=False)
                
    #             # Process events to ensure rendering
    #             QtWidgets.QApplication.processEvents()
                
    #             # Capture frame and write directly to video
    #             qimage = exporter.export(toBytes=True)
    #             bgr_array = qimage_to_bgr(qimage)
    #             out.write(bgr_array)
    #     finally:
    #         # Always close video writer (if opened) and restore debug print setting
    #         if out is not None:
    #             out.release()
    #         self.params.debug_print = original_debug_print
        
    #     if progress_print:
    #         print(f'Video exported successfully to: {video_filepath}')
        
    #     return video_filepath

