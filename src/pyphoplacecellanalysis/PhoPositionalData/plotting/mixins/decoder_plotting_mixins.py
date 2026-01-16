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



@dataclass
class Viewport:
    """Represents the visible time range in the viewport"""
    start_time: float
    end_time: float
    width_pixels: int
    height_pixels: int
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def pixels_per_second(self) -> float:
        return self.width_pixels / self.duration if self.duration > 0 else 0.0


@dataclass
class ThumbnailCacheEntry:
    """Cached thumbnail entry"""
    image: np.ndarray  # RGB image array
    epoch_start_idx: int
    epoch_end_idx: int
    bin_size: float
    resolution_level: int
    timestamp: float  # For cache invalidation


@define(slots=False, eq=False)
class OptimizedViewportRenderer:
    """ #TODO 2025-12-16 16:36: - [ ] Aims to replace `SingleArtistMultiEpochBatchHelpers` with a much more efficient implementation
    Efficient viewport-based renderer with image caching and adaptive bin sizing.
    
    Key optimizations:
    1. Viewport-based rendering: Only processes visible epochs + buffer
    2. Image caching: Caches rendered thumbnails to avoid re-rendering
    3. Adaptive bin size: Adjusts epoch size based on zoom level
    4. Multi-resolution support: Can render at different detail levels
    
    Usage:
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import OptimizedViewportRenderer, ThumbnailCacheEntry, Viewport

        renderer = OptimizedViewportRenderer(
            results2D=results2D,
            active_ax=track_ax,
            base_frame_divide_bin_size=0.5,
            min_thumbnail_width_px=50,
            max_thumbnails_per_viewport=100
        )
        
        # Render for current viewport
        viewport = Viewport(start_time=10.0, end_time=20.0, width_pixels=800, height_pixels=200)
        artists, extent = renderer.render_viewport(viewport, posterior_masking_value=0.0025)
    """
    results2D: "DecodingResultND" = field()
    active_ax = field()
    base_frame_divide_bin_size: float = field(default=0.5)
    rotate_to_vertical: bool = field(default=True)
    active_epoch_name: str = field(default='global')
    
    # Adaptive bin sizing parameters
    min_thumbnail_width_px: int = field(default=50)  # Minimum width per thumbnail in pixels
    max_thumbnails_per_viewport: int = field(default=100)  # Maximum thumbnails to show
    viewport_buffer_epochs: int = field(default=2)  # Extra epochs to render outside viewport
    
    # Caching parameters
    max_cache_size: int = field(default=500)  # Maximum cached thumbnails
    cache_resolution_levels: List[int] = field(default=Factory(lambda: [0, 1, 2]))  # Multi-resolution levels
    
    # Internal state
    _thumbnail_cache: OrderedDict = field(default=Factory(OrderedDict), init=False, repr=False)
    _epoch_index: Optional[Dict] = field(default=None, init=False, repr=False)
    _total_time_duration: Optional[float] = field(default=None, init=False, repr=False)
    _prev_composite_artist: Optional[Any] = field(default=None, init=False, repr=False)
    
    def __attrs_post_init__(self):
        """Initialize internal data structures"""
        self._build_epoch_index()
    
    @property
    def a_result2D(self) -> "DecodedFilterEpochsResult":
        """Get the decoded result for the active epoch"""
        return self.results2D.frame_divided_epochs_results[self.active_epoch_name]
    
    @property
    def a_new_global2D_decoder(self) -> "BasePositionDecoder":
        """Get the decoder for the active epoch"""
        return self.results2D.decoders[self.active_epoch_name]
    
    @property
    def num_filter_epochs(self) -> int:
        """Total number of frame division epochs"""
        return self.a_result2D.num_filter_epochs
    
    def _build_epoch_index(self):
        """Build index mapping time ranges to epoch indices for fast lookup"""
        if self._epoch_index is not None:
            return
        
        # Get time bin information from the result
        n_timebins, flat_time_bin_containers, _ = self.a_result2D.flatten()
        flat_time_bin_containers = flat_time_bin_containers.tolist()
        
        # FIRST: Compute total time duration from actual time bin data
        # This must be done before building the index to avoid circular dependency
        all_time_values = []
        for tb in flat_time_bin_containers:
            if hasattr(tb, 'centers') and len(tb.centers) > 0:
                all_time_values.extend(tb.centers)
            elif hasattr(tb, 'edges') and len(tb.edges) > 0:
                # Use edges if centers not available
                all_time_values.extend(tb.edges)
        
        if all_time_values:
            self._total_time_duration = max(all_time_values) - min(all_time_values)
            time_offset = min(all_time_values)  # Track offset for absolute time
        else:
            # Fallback: estimate from num_filter_epochs
            self._total_time_duration = self.num_filter_epochs * self.base_frame_divide_bin_size
            time_offset = 0.0
        
        # Build index: epoch_idx -> (start_time, end_time)
        self._epoch_index = {}
        
        # Try to access epoch-organized time bins directly from a_result2D
        if hasattr(self.a_result2D, 'time_bin_containers') and self.a_result2D.time_bin_containers:
            # Direct epoch access - most reliable method
            # Note: time_bin_containers[epoch_idx] returns a single BinningContainer, not a list
            for epoch_idx in range(min(self.num_filter_epochs, len(self.a_result2D.time_bin_containers))):
                epoch_tb_container = self.a_result2D.time_bin_containers[epoch_idx]
                epoch_times = []
                
                # epoch_tb_container is a single BinningContainer, not an iterable
                if hasattr(epoch_tb_container, 'centers') and len(epoch_tb_container.centers) > 0:
                    epoch_times.extend(epoch_tb_container.centers)
                elif hasattr(epoch_tb_container, 'edges') and len(epoch_tb_container.edges) > 1:
                    # Use edges if centers not available
                    epoch_times.extend(epoch_tb_container.edges)
                elif hasattr(epoch_tb_container, 'edges') and len(epoch_tb_container.edges) > 0:
                    # Use midpoint of edges if only one edge
                    epoch_times.append(epoch_tb_container.edges[0])
                
                if epoch_times:
                    epoch_start = min(epoch_times)
                    epoch_end = max(epoch_times)
                else:
                    # Fallback: compute from epoch index
                    epoch_start = time_offset + epoch_idx * self.base_frame_divide_bin_size
                    epoch_end = time_offset + (epoch_idx + 1) * self.base_frame_divide_bin_size
                
                self._epoch_index[epoch_idx] = (epoch_start, epoch_end)
        else:
            # Fallback: map flat time bins to epochs by time value
            # Group time bins by which epoch they belong to based on their time value
            for epoch_idx in range(self.num_filter_epochs):
                epoch_start_time = time_offset + epoch_idx * self.base_frame_divide_bin_size
                epoch_end_time = time_offset + (epoch_idx + 1) * self.base_frame_divide_bin_size
                
                # Find time bins that fall within this epoch's time range
                epoch_times = []
                for tb in flat_time_bin_containers:
                    if hasattr(tb, 'centers') and len(tb.centers) > 0:
                        for center in tb.centers:
                            if epoch_start_time <= center < epoch_end_time:
                                epoch_times.append(center)
                
                if epoch_times:
                    epoch_start = min(epoch_times)
                    epoch_end = max(epoch_times)
                else:
                    # No time bins found in this range - use calculated times
                    epoch_start = epoch_start_time
                    epoch_end = epoch_end_time
                
                self._epoch_index[epoch_idx] = (epoch_start, epoch_end)
        
        # Ensure total duration is properly set (update if we found a larger range)
        if self._epoch_index:
            max_end = max(end for _, end in self._epoch_index.values())
            min_start = min(start for start, _ in self._epoch_index.values())
            computed_duration = max_end - min_start
            if computed_duration > 0:
                self._total_time_duration = computed_duration
    
    def _get_epoch_idx_for_timebin(self, timebin_idx: int) -> int:
        """Get epoch index for a given timebin index"""
        # This method is primarily used as a fallback if direct epoch access isn't available
        # It estimates epoch based on timebin index and known structure
        
        # Ensure _total_time_duration is computed if not already set
        if self._total_time_duration is None:
            self._build_epoch_index()
        
        # Type check to satisfy linter
        total_duration = self._total_time_duration
        if total_duration is not None and total_duration > 0 and self.num_filter_epochs > 0:
            # Estimate based on proportion of total time
            time_per_epoch = total_duration / self.num_filter_epochs
            # Estimate which epoch this timebin belongs to
            estimated_epoch = int(timebin_idx * time_per_epoch / self.base_frame_divide_bin_size)
            return max(0, min(estimated_epoch, self.num_filter_epochs - 1))
        else:
            # Simple fallback: assume roughly equal distribution
            if self.num_filter_epochs > 0:
                # Estimate based on total timebins
                n_timebins, _, _ = self.a_result2D.flatten()
                if n_timebins > 0:
                    timebins_per_epoch = max(1, n_timebins // self.num_filter_epochs)
                    return min(timebin_idx // timebins_per_epoch, self.num_filter_epochs - 1)
            return 0
    
    def compute_adaptive_bin_size(self, viewport: Viewport) -> float:
        """
        Compute optimal bin size based on viewport dimensions and constraints.
        
        Ensures thumbnails are readable (min width) but don't exceed max count.
        """
        if viewport.duration <= 0:
            return self.base_frame_divide_bin_size
        
        # Calculate based on minimum thumbnail width
        min_bin_size_from_width = viewport.duration / (viewport.width_pixels / self.min_thumbnail_width_px)
        
        # Calculate based on maximum thumbnail count
        max_bin_size_from_count = viewport.duration / self.max_thumbnails_per_viewport
        
        # Use the larger (coarser) bin size to satisfy both constraints
        adaptive_bin_size = max(min_bin_size_from_width, max_bin_size_from_count)
        
        # Round to reasonable precision and ensure it's at least base size
        adaptive_bin_size = max(adaptive_bin_size, self.base_frame_divide_bin_size)
        
        # Optionally snap to base_bin_size multiples for better caching
        if adaptive_bin_size > self.base_frame_divide_bin_size:
            adaptive_bin_size = self.base_frame_divide_bin_size * np.round(adaptive_bin_size / self.base_frame_divide_bin_size)
        
        return float(adaptive_bin_size)
    
    def get_visible_epoch_range(self, viewport: Viewport, bin_size: float) -> Tuple[int, int]:
        """
        Get the range of epoch indices that are visible in the viewport.
        Returns (start_epoch_idx, end_epoch_idx) with buffer.
        """
        if self._epoch_index is None:
            self._build_epoch_index()
        
        # Map viewport absolute times to epoch indices using the epoch index
        start_epoch_idx = None
        end_epoch_idx = None
        
        # Find epochs that overlap with viewport time range
        for epoch_idx, (epoch_start, epoch_end) in self._epoch_index.items():
            # Check if this epoch overlaps with viewport
            if epoch_start <= viewport.end_time and epoch_end >= viewport.start_time:
                if start_epoch_idx is None:
                    start_epoch_idx = epoch_idx
                end_epoch_idx = epoch_idx
        
        # Fallback: if no epochs found, use time-based calculation
        if start_epoch_idx is None:
            # Try to find the first epoch that starts before or at viewport start
            for epoch_idx, (epoch_start, epoch_end) in sorted(self._epoch_index.items()):
                if epoch_start <= viewport.start_time:
                    start_epoch_idx = epoch_idx
                    break
            
            # If still None, use time-based fallback
            if start_epoch_idx is None:
                # Estimate based on time offset
                if self._epoch_index:
                    first_epoch_start = min(start for start, _ in self._epoch_index.values())
                    time_offset = viewport.start_time - first_epoch_start
                    start_epoch_idx = max(0, int(np.floor(time_offset / bin_size)))
                else:
                    start_epoch_idx = 0
            
            # Find end epoch
            for epoch_idx, (epoch_start, epoch_end) in sorted(self._epoch_index.items(), reverse=True):
                if epoch_end >= viewport.end_time:
                    end_epoch_idx = epoch_idx
                    break
            
            if end_epoch_idx is None:
                if self._epoch_index:
                    first_epoch_start = min(start for start, _ in self._epoch_index.values())
                    time_offset = viewport.end_time - first_epoch_start
                    end_epoch_idx = min(self.num_filter_epochs, int(np.ceil(time_offset / bin_size)))
                else:
                    end_epoch_idx = self.num_filter_epochs
        
        # Add buffer
        start_epoch_idx = max(0, start_epoch_idx - self.viewport_buffer_epochs)
        end_epoch_idx = min(self.num_filter_epochs, end_epoch_idx + self.viewport_buffer_epochs)
        
        return start_epoch_idx, end_epoch_idx
    
    def _get_cache_key(self, epoch_start_idx: int, epoch_end_idx: int, bin_size: float, resolution_level: int = 0) -> str:
        """Generate cache key for thumbnail"""
        key_data = f"{epoch_start_idx}_{epoch_end_idx}_{bin_size:.6f}_{resolution_level}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_or_create_thumbnail(self, epoch_start_idx: int, epoch_end_idx: int, bin_size: float, 
                                 resolution_level: int = 0, render_fn: Optional[Callable] = None) -> Optional[np.ndarray]:
        """
        Get thumbnail from cache or create it if missing.
        
        Returns None if epoch range is invalid or empty.
        """
        cache_key = self._get_cache_key(epoch_start_idx, epoch_end_idx, bin_size, resolution_level)
        
        # Check cache
        if cache_key in self._thumbnail_cache:
            # Move to end (LRU)
            entry = self._thumbnail_cache.pop(cache_key)
            self._thumbnail_cache[cache_key] = entry
            return entry.image
        
        # Cache miss - need to render
        if render_fn is None:
            return None
        
        try:
            image = render_fn(epoch_start_idx, epoch_end_idx, bin_size, resolution_level)
            
            if image is not None:
                # Add to cache
                entry = ThumbnailCacheEntry(
                    image=image,
                    epoch_start_idx=epoch_start_idx,
                    epoch_end_idx=epoch_end_idx,
                    bin_size=bin_size,
                    resolution_level=resolution_level,
                    timestamp=0.0  # Could use time.time() for invalidation
                )
                self._thumbnail_cache[cache_key] = entry
                
                # Enforce cache size limit (LRU eviction)
                while len(self._thumbnail_cache) > self.max_cache_size:
                    self._thumbnail_cache.popitem(last=False)  # Remove oldest
                
                return image
        except Exception as e:
            print(f"Error rendering thumbnail for epochs [{epoch_start_idx}, {epoch_end_idx}]: {e}")
            return None
        
        return None
    
    def _render_epoch_thumbnail(self, epoch_start_idx: int, epoch_end_idx: int, bin_size: float,
                                resolution_level: int, posterior_masking_value: float = 0.0025,
                                width_px: int = 200, height_px: int = 200) -> Optional[np.ndarray]:
        """
        Render a single epoch thumbnail to an image array.
        
        This is the core rendering function that creates the actual thumbnail image.
        """
        try:
            # Get data for this epoch range
            a_result = self.a_result2D
            a_decoder = self.a_new_global2D_decoder
            
            # Extract data for this epoch range
            # The flattened array's third dimension is organized by epoch index
            # Similar to how _slice_to_epoch_range works in the original code
            n_timebins, flat_time_bin_containers, flat_timebins_p_x_given_n = a_result.flatten()
            
            # Slice directly by epoch indices (third dimension is epochs)
            # flat_timebins_p_x_given_n shape is (n_xbins, n_ybins, n_epochs)
            if epoch_end_idx > flat_timebins_p_x_given_n.shape[2]:
                epoch_end_idx = flat_timebins_p_x_given_n.shape[2]
            
            if epoch_start_idx >= epoch_end_idx or epoch_start_idx >= flat_timebins_p_x_given_n.shape[2]:
                return None
            
            epoch_timebins_p_x_given_n = flat_timebins_p_x_given_n[:, :, epoch_start_idx:epoch_end_idx]
            
            if epoch_timebins_p_x_given_n.size == 0:
                return None
            
            # Reshape for display (similar to original code)
            if self.rotate_to_vertical:
                # Stack vertically
                stacked = np.column_stack([epoch_timebins_p_x_given_n[:, :, i] for i in range(epoch_timebins_p_x_given_n.shape[2])])
                stacked = stacked.T
            else:
                # Stack horizontally
                stacked = np.row_stack([epoch_timebins_p_x_given_n[:, :, i] for i in range(epoch_timebins_p_x_given_n.shape[2])])
            
            # Create figure and render
            dpi = 100
            fig_width = width_px / dpi
            fig_height = height_px / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            ax.set_axis_off()
            
            # Render as heatmap
            xbin_centers = a_decoder.xbin_centers
            ybin_centers = a_decoder.ybin_centers
            
            # Apply masking
            masked_data = np.where(stacked < posterior_masking_value, np.nan, stacked)
            
            # Create extent using actual epoch time ranges
            if epoch_start_idx in self._epoch_index and epoch_end_idx > 0:
                time_start, _ = self._epoch_index[epoch_start_idx]
                if epoch_end_idx - 1 in self._epoch_index:
                    _, time_end = self._epoch_index[epoch_end_idx - 1]
                else:
                    time_end = time_start + (epoch_end_idx - epoch_start_idx) * bin_size
            else:
                time_start = epoch_start_idx * bin_size
                time_end = epoch_end_idx * bin_size
            
            if self.rotate_to_vertical:
                extent = [time_start, time_end, ybin_centers[0], ybin_centers[-1]]
            else:
                extent = [xbin_centers[0], xbin_centers[-1], time_start, time_end]
            
            # Render heatmap
            im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent, 
                          cmap='viridis', interpolation='bilinear')
            
            # Render to image
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            # Get actual canvas dimensions instead of assuming they match width_px/height_px
            actual_width = int(fig.canvas.get_width_height()[0])
            actual_height = int(fig.canvas.get_width_height()[1])

            # Reshape using actual dimensions
            image = buf.reshape((actual_height, actual_width, 3))

            # Resize to desired dimensions if they don't match
            if actual_width != width_px or actual_height != height_px:
                from PIL import Image
                img = Image.fromarray(image)
                img = img.resize((width_px, height_px), Image.Resampling.LANCZOS)
                image = np.array(img)

            plt.close(fig)
            
            return image
                
        except Exception as e:
            print(f"Error in _render_epoch_thumbnail: {e}")
            import traceback
            traceback.print_exc()
            return None



    
    def render_viewport(self, viewport: Viewport, posterior_masking_value: float = 0.0025,
                       debug_print: bool = False) -> Tuple[Dict[str, Any], Tuple[float, float, float, float]]:
        """
        Render the visible viewport efficiently using cached thumbnails.
        
        Returns:
            (artists_dict, image_extent) where:
            - artists_dict: Dictionary of matplotlib artists
            - image_extent: [x0, x1, y0, y1] extent of the rendered image
        """
        # Compute adaptive bin size
        adaptive_bin_size = self.compute_adaptive_bin_size(viewport)
        
        # Get visible epoch range
        epoch_start_idx, epoch_end_idx = self.get_visible_epoch_range(viewport, adaptive_bin_size)
        
        if debug_print:
            print(f"Viewport: {viewport.start_time:.2f}-{viewport.end_time:.2f}s "
                  f"({viewport.duration:.2f}s), {viewport.width_pixels}x{viewport.height_pixels}px")
            print(f"Adaptive bin_size: {adaptive_bin_size:.3f}s")
            print(f"Visible epoch range: [{epoch_start_idx}, {epoch_end_idx})")
            print(f"Total epochs: {self.num_filter_epochs}")
            if self._epoch_index:
                print(f"Epoch index sample (first 3): {dict(list(self._epoch_index.items())[:3])}")
        
        # Validate epoch range
        if epoch_start_idx >= epoch_end_idx or epoch_start_idx >= self.num_filter_epochs:
            if debug_print:
                print(f"Empty epoch range - no data to render (start={epoch_start_idx}, end={epoch_end_idx}, total={self.num_filter_epochs})")
            return {}, (viewport.start_time, viewport.end_time, 0.0, 1.0)
        
        # Calculate thumbnail dimensions
        num_thumbnails = epoch_end_idx - epoch_start_idx
        thumbnail_width_px = max(self.min_thumbnail_width_px, viewport.width_pixels // num_thumbnails)
        thumbnail_height_px = viewport.height_pixels
        
        # Render thumbnails (using cache when possible)
        thumbnails = []
        render_fn = lambda e_start, e_end, b_size, res_level: self._render_epoch_thumbnail(
            e_start, e_end, b_size, res_level, posterior_masking_value, 
            thumbnail_width_px, thumbnail_height_px
        )
        
        for epoch_idx in range(epoch_start_idx, epoch_end_idx):
            thumbnail = self._get_or_create_thumbnail(
                epoch_idx, epoch_idx + 1, adaptive_bin_size, 
                resolution_level=0, render_fn=render_fn
            )
            if thumbnail is not None:
                thumbnails.append((epoch_idx, thumbnail))
        
        if not thumbnails:
            if debug_print:
                print(f"No thumbnails generated - empty data (tried epochs {epoch_start_idx} to {epoch_end_idx})")
                print(f"Checking if epochs exist in index: start={epoch_start_idx in self._epoch_index if self._epoch_index else False}, end={epoch_end_idx-1 in self._epoch_index if self._epoch_index else False}")
            return {}, (viewport.start_time, viewport.end_time, 0.0, 1.0)
        
        # Composite thumbnails into single image
        composite_width = len(thumbnails) * thumbnail_width_px
        composite_height = thumbnail_height_px
        composite_image = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        for i, (epoch_idx, thumbnail) in enumerate(thumbnails):
            x_start = i * thumbnail_width_px
            x_end = x_start + thumbnail_width_px
            # Resize thumbnail if needed
            if thumbnail.shape[1] != thumbnail_width_px or thumbnail.shape[0] != thumbnail_height_px:
                from PIL import Image
                img = Image.fromarray(thumbnail)
                img = img.resize((thumbnail_width_px, thumbnail_height_px), Image.Resampling.LANCZOS)
                thumbnail = np.array(img)
            composite_image[:, x_start:x_end, :] = thumbnail


        if debug_print:
            print(f"Composite image shape: {composite_image.shape}")
            print(f"Composite image min/max: {composite_image.min()}/{composite_image.max()}")
            print(f"Composite image non-zero pixels: {np.count_nonzero(composite_image)}")


        # Display composite image on axis
        # Use actual time ranges from epoch index
        if self._epoch_index and epoch_start_idx in self._epoch_index:
            time_start, _ = self._epoch_index[epoch_start_idx]
            if epoch_end_idx > 0 and (epoch_end_idx - 1) in self._epoch_index:
                _, time_end = self._epoch_index[epoch_end_idx - 1]
            else:
                time_end = time_start + (epoch_end_idx - epoch_start_idx) * adaptive_bin_size
        else:
            time_start = epoch_start_idx * adaptive_bin_size
            time_end = epoch_end_idx * adaptive_bin_size
        # extent = [time_start, time_end, 0.0, 1.0]

        # Display composite image on axis
        # CRITICAL: Use viewport times for extent to ensure image is visible
        extent = [viewport.start_time, viewport.end_time, 0.0, 1.0]

        
        im = self.active_ax.imshow(composite_image, aspect='auto', origin='lower', 
                                   extent=extent, interpolation='bilinear')
        # Fix 2: Ensure axis limits include the extent
        self.active_ax.set_xlim(viewport.start_time, viewport.end_time)
        self.active_ax.set_ylim(0.0, 1.0)


        artists_dict = {'composite_image': im}
        
        if debug_print:
            print(f"Rendered {len(thumbnails)} thumbnails, cache size: {len(self._thumbnail_cache)}")
        
        return artists_dict, extent
    
    def on_window_changed(self, start_t: float, end_t: float, debug_print: bool = False):
        """
        Window update callback method conforming to the window update protocol.
        
        This method is called when the visible time window changes (e.g., when user scrolls).
        It automatically renders the viewport for the new time range.
        
        Args:
            start_t: Start time of the visible window (absolute time in seconds)
            end_t: End time of the visible window (absolute time in seconds)
            debug_print: Whether to print debug information
        
        This method conforms to the same protocol as TimeSynchronizedPlotterBase.on_window_changed()
        and can be used as a subscriber to window update signals.
        
        Usage:
            # Connect to window update signal
            window_scrolled.connect(renderer.on_window_changed)
            
            # Or call directly
            renderer.on_window_changed(start_t=9293.5, end_t=9295.5)
        """
        # Clear previous artist if it exists
        if hasattr(self, '_prev_composite_artist') and self._prev_composite_artist is not None:
            try:
                self._prev_composite_artist.remove()
            except:
                pass
        
        # Get current axis dimensions
        ax = self.active_ax
        bbox = ax.get_window_extent()
        width_pixels = int(bbox.width)
        height_pixels = int(bbox.height)
        
        # Create viewport for current visible window
        viewport = Viewport(
            start_time=start_t,
            end_time=end_t,
            width_pixels=width_pixels,
            height_pixels=height_pixels
        )
        
        # Render viewport
        artists, extent = self.render_viewport(viewport, debug_print=debug_print)
        
        # Store artist for cleanup
        if 'composite_image' in artists:
            self._prev_composite_artist = artists['composite_image']
        
        # Ensure axis limits match viewport
        ax.set_xlim(start_t, end_t)
        ax.set_ylim(0.0, 1.0)
        
        # Force redraw
        ax.figure.canvas.draw_idle()
        
        return artists, extent
    
    def clear_cache(self):
        """Clear the thumbnail cache"""
        self._thumbnail_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._thumbnail_cache),
            'max_cache_size': self.max_cache_size,
            'cache_keys': list(self._thumbnail_cache.keys())[:10]  # First 10 keys
        }
    
























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
    results2D: "DecodingResultND" = field()

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
        return self.results2D.frame_divided_epochs_results[self.active_epoch_name]

    @property
    def a_new_global2D_decoder(self) -> BasePositionDecoder:
        return self.results2D.decoders[self.active_epoch_name]

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
            print(f'force_recompute == True, so `self.stacked_flat_global_pos_df` will be rebuilt from scratch from `self.results2D.pos_df`...')
            self.has_data_been_built = False
    
        if (self.stacked_flat_global_pos_df is None) or force_recompute:
            self.stacked_flat_global_pos_df = deepcopy(self.results2D.pos_df)

        ## slice `stacked_flat_global_pos_df` by desired start/end indicies too:
        if (self.desired_epoch_end_idx is not None):
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[np.logical_and((self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx), (self.stacked_flat_global_pos_df['global_frame_division_idx'] < self.desired_epoch_end_idx))]
        else:
            self.stacked_flat_global_pos_df = self.stacked_flat_global_pos_df[(self.stacked_flat_global_pos_df['global_frame_division_idx'] >= self.desired_epoch_start_idx)]


        # Validate that filtering didn't result in empty dataframe
        if len(self.stacked_flat_global_pos_df) == 0:
            # Get available indices for better error message
            if hasattr(self, 'results2D') and self.results2D.pos_df is not None and 'global_frame_division_idx' in self.results2D.pos_df.columns:
                available_indices = sorted(self.results2D.pos_df['global_frame_division_idx'].unique())
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
                                                                            include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=time_bin_index, rotate_to_vertical=self.rotate_to_vertical,
                                                                            # should_perform_reshape=True,
                                                                            should_perform_reshape=False,
                                                                            should_post_hoc_fit_to_image_extent=should_post_hoc_fit_to_image_extent,
                                                                            posterior_masking_value=posterior_masking_value, 
                                                                            time_cmap=deepcopy(self.cmap),
                                                                            debug_print=debug_print) # , allow_time_slider=True


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
    def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False, s=20.0, time_cmap='viridis', **LineCollection_kwargs):
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
            _out_markers: PathCollection = ax.scatter(x=x, y=y, s=s, c=colors_arr, marker='D')
            return line, _out_markers
        else:
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
    def _helper_add_concentrated_arrows_to_line(cls, ax, t, x, y, speed=None, time_cmap='viridis', arrow_skip: int=20,
                                                mutation_scale_multiplier = 40, mutation_scale_constant = 10, arrow_length_multiplier = 0.2, arrow_length_constant = 0.05, arrow_lw = 0.5,
                                                ) -> List[FancyArrowPatch]:
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
        # norm = plt.Normalize(t.min(), t.max())
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
            speed = np.concatenate([[0], dist / dt])
            
        assert len(t) == len(speed), f"len(t): {len(t)} != len(speed): {len(speed)}"
        # colors_arr = time_cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!
        _out_markers = {}
        # --- Add Arrows along the path ---
        # how many points to skip between arrows
        for i in range(0, len(x)-arrow_skip, arrow_skip):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i+1], y[i+1]
            dx, dy = x1 - x0, y1 - y0
            spd = speed[i]
            spd_percent_max = (spd / np.max(speed))
            # scale arrow size by speed
            arrow_length = arrow_length_constant + (arrow_length_multiplier * spd_percent_max)
            mutation_scale = mutation_scale_constant + (mutation_scale_multiplier * spd_percent_max)
            
            arrow = FancyArrowPatch(
                (x0, y0),
                (x0 + dx * arrow_length, y0 + dy * arrow_length),
                arrowstyle='-|>', mutation_scale=mutation_scale,
                color=time_cmap(spd_percent_max),
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
            print(f'np.shape(posterior): {np.shape(posterior)}')
        
        is_2D_dt: bool = (np.ndim(posterior) >= 3)
        is_2D: bool = (np.ndim(posterior) == 2)

        # Add time dimension if posterior is 2D (spatial 2D without time dimension)
        if is_2D and (not is_2D_dt):
            posterior = posterior[np.newaxis, :, :]  # Shape: (1, n_x_bins, n_y_bins)

        masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)

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
        if (not is_2D):
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


    @function_attributes(short_name=None, tags=['main', 'plot'], input_requires=[], output_provides=[], uses=[], used_by=['multi_DecodedTrajectoryMatplotlibPlotter_side_by_side', 'self.plot_decoded_laps_2d'], creation_date='2025-06-30 12:58', related_items=[])
    def plot_decoded_trajectories_2d(self, curr_position_df: pd.DataFrame, epoch_specific_position_dfs: List[pd.DataFrame], epoch_ids: NDArray, sess=None, curr_num_subplots=10, active_page_index=0, plot_actual_lap_lines:bool=False, fixed_columns: int = 2, use_theoretical_tracks_instead: bool = True, existing_ax=None, axes_inset_locators_list=None, cmap=None,
                                    posteriors=None, plot_mode: str='time_gradient', **kwargs):
        """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots
        
        Called to setup the graph.
        
        Great plotting for laps.
        Plots in a paginated manner.
        
        use_theoretical_tracks_instead: bool = True - # if False, renders all positions the animal traversed over the entire session. Otherwise renders the theoretical (idaal) track.

        ISSUE: `fixed_columns: int = 1` doesn't work due to indexing


        History: based off of plot_lap_trajectories_2d

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_decoded_trajectories_2d
        
            fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_position_df, epoch_specific_position_dfs=None, epoch_ids=None, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)

        
        """
        from pyphocorehelpers.geometry_helpers import compute_data_aspect_ratio

        # _active_plot_fn = cls._helper_add_heatmap
        # _active_plot_fn = cls._helper_add_hdr_contours
        _active_posterior_plot_fn = kwargs.pop('active_plot_fn', DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap)
        

        if (self.xbin is not None) and (self.ybin is not None):
            single_ax_aspect_ratio, (single_ax_width, single_ax_height) = compute_data_aspect_ratio(xbin=self.xbin, ybin=self.ybin)
        else:
            single_ax_width = None
            single_ax_height = None

        # try:
        if cmap is None:
            cmap = 'viridis'
        
    
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
        


        def _subfn_add_specific_epoch_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_epochs_ids, epochs_position_traces, epochs_time_ranges, active_plot_mode: str ='time_gradient', **plot_traj_kwargs):
            """ captures: cmap 
            """


            # Add the lap trajectory:
            for a_linear_index in linear_plotter_indicies:
                curr_lap_id = active_page_epochs_ids[a_linear_index]
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                curr_lap_time_range = epochs_time_ranges[curr_lap_id]
                curr_lap_label_text = 'Epoch[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[0], curr_lap_time_range[1])
                curr_lap_num_points = len(epochs_position_traces[curr_lap_id][0,:])
                valid_plotting_modes: List[str] = ['time_gradient', 'line', 'scatter']
                # if use_time_gradient_line:
                if active_plot_mode == 'time_gradient':
                    # Create a continuous norm to map from data points to colors
                    curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(epochs_position_traces[curr_lap_id][0,:]))
                    norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
                    # needs to be (numlines) x (points per line) x 2 (for x and y)
                    points = np.array([epochs_position_traces[curr_lap_id][0,:], epochs_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(curr_lap_timeseries)
                    lc.set_linewidth(plot_traj_kwargs.get('linewidth', 2))
                    lc.set_alpha(plot_traj_kwargs.get('alpha', 0.85))
                    a_line = axs[curr_row][curr_col].add_collection(lc)
                    # add_arrow(line)
                elif active_plot_mode == 'line':
                    if 'c' not in plot_traj_kwargs:
                        plot_traj_kwargs['c'] = 'k'
                    if 'alpha' not in plot_traj_kwargs:
                        plot_traj_kwargs['alpha'] = 0.85
                    a_line = axs[curr_row][curr_col].plot(epochs_position_traces[curr_lap_id][0,:], epochs_position_traces[curr_lap_id][1,:], **plot_traj_kwargs)
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
                    a_scatter = axs[curr_row][curr_col].scatter(epochs_position_traces[curr_lap_id][0,:], epochs_position_traces[curr_lap_id][1,:], **plot_traj_kwargs)

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
                # if should_perform_reshape:
                #     posterior_data = deepcopy(posterior_data).T
                # else:
                #     posterior_data = deepcopy(posterior_data)
                    
                # if posterior_masking_value is not None:
                #     masked_posterior = np.ma.masked_less(posterior_data, posterior_masking_value)
                # else:
                #     masked_posterior = posterior_data
                # x_values = deepcopy(xbin_centers)
                # y_values = deepcopy(ybin_centers)
                # if self.rotate_to_vertical:
                #     ordinate_first_image_extent = (y_values.min(), y_values.max(), x_values.min(), x_values.max())
                #     masked_posterior = masked_posterior.T
                # else:
                #     ordinate_first_image_extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
                # if posterior_extent is not None:
                #     ordinate_first_image_extent = deepcopy(posterior_extent)
                # a_heatmap = ax.imshow(masked_posterior, aspect='auto', cmap=posterior_cmap, alpha=full_posterior_opacity, extent=ordinate_first_image_extent, origin='lower', interpolation='none')
                # return [a_heatmap], ordinate_first_image_extent, None
            
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
            _out_objs = _subfn_add_specific_epoch_trajectory(self.fig, self.axs, linear_plotter_indicies=self.linear_plotter_indicies, row_column_indicies=self.row_column_indicies, active_page_epochs_ids=active_page_epochs_ids, epochs_position_traces=epochs_position_traces, epochs_time_ranges=epochs_time_ranges, active_plot_mode=plot_mode, **kwargs)
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



# ==================================================================================================================================================================================================================================================================================== #
# Napari/3D                                                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #
@define(slots=False, eq=False)
class DecodedTrajectoryNapariPlotter(DecodedTrajectoryPlotter):
    """ plots decoded posteriors using Napari with sliders for epoch and time-bin.

    Builds a 4D volume over (epoch, time_bin, xbin, ybin) and shows it as a Napari image layer.

    Usage (example):

        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryNapariPlotter
        napari_plotter = DecodedTrajectoryNapariPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)
        viewer, layer = napari_plotter.build_ui()

    """

    viewer: Optional["napari.viewer.Viewer"] = field(default=None)
    image_layer: Any = field(default=None)
    peak_contours_layer: Any = field(default=None)
    peak_prominence_result: Optional[Any] = field(default=None)
    custom_logging_widget: Optional[Any] = field(default=None, metadata={'desc': 'Custom dock widget for logging messages'})

    posterior_volume: Optional[NDArray] = field(default=None)
    time_bin_centers_matrix: Optional[NDArray] = field(default=None)

    curr_epoch_idx: int = field(default=0)
    curr_time_bin_index: int = field(default=0)

    enable_plot_all_time_bins_in_epoch_mode: bool = field(default=False, metadata={'desc': 'if True, Napari time axis spans all time bins for all epochs; otherwise still the same but semantics may differ in callers.'})
    
    create_logging_dock: bool = field(default=False, metadata={'desc': 'If True, automatically creates a custom logging dock widget when build_ui() is called'})
    logging_dock_area: str = field(default='bottom', metadata={'desc': 'Dock area placement for logging widget: left, right, top, or bottom'})
    logging_dock_name: str = field(default='Custom Log', metadata={'desc': 'Name/title for the logging dock widget'})

    @function_attributes(short_name=None, tags=['napari', 'posterior', 'volume', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=['DecodedTrajectoryNapariPlotter.build_ui'], creation_date='2025-12-23 08:00', related_items=['DecodedTrajectoryPyVistaPlotter'])
    def build_posterior_volume(self, desired_max_height: float = 50.0) -> Tuple[NDArray, NDArray]:
        """Builds a 4D volume over (epoch, time_bin, xbin, ybin) suitable for Napari.

        Uses the same logic as `_perform_get_curr_posterior` on `DecodedTrajectoryPyVistaPlotter`
        to ensure scaling and 1D/2D handling matches the PyVista implementation.

        Returns:
            posterior_volume: np.ndarray with shape (num_epochs, max_num_time_bins, n_xbins, n_ybins)
            time_bin_centers_matrix: np.ndarray with shape (num_epochs, max_num_time_bins)
        """
        assert self.a_result is not None, "DecodedTrajectoryNapariPlotter requires `a_result`."
        assert self.xbin_centers is not None, "DecodedTrajectoryNapariPlotter requires `xbin_centers`."

        num_epochs: int = self.num_filter_epochs
        epoch_time_bin_counts: List[int] = []
        for an_epoch_idx in np.arange(num_epochs):
            time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers
            epoch_time_bin_counts.append(len(time_bin_centers))

        self.epoch_time_bin_counts = epoch_time_bin_counts
        max_num_time_bins: int = int(np.max(epoch_time_bin_counts))
        n_xbins: int = len(self.xbin_centers)
        if self.ybin_centers is not None:
            n_ybins: int = len(self.ybin_centers)
        else:
            # treat as 1D in y if not provided
            n_ybins = 1

        posterior_volume = np.zeros((num_epochs, max_num_time_bins, n_xbins, n_ybins), dtype=float)
        time_bin_centers_matrix = np.full((num_epochs, max_num_time_bins), np.nan, dtype=float)

        for an_epoch_idx in np.arange(num_epochs):
            # replicate core of DecodedTrajectoryPyVistaPlotter._perform_get_curr_posterior
            a_posterior_p_x_given_n_all_t = self.a_result.p_x_given_n_list[an_epoch_idx]
            a_most_likely_positions = self.a_result.most_likely_positions_list[an_epoch_idx]
            a_time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers
            assert len(a_time_bin_centers) == len(a_most_likely_positions), f"len(a_time_bin_centers): {len(a_time_bin_centers)} != len(a_most_likely_positions): {len(a_most_likely_positions)}"

            min_v = np.nanmin(a_posterior_p_x_given_n_all_t)
            max_v = np.nanmax(a_posterior_p_x_given_n_all_t)
            multiplier_factor: float = desired_max_height / (float(max_v) - float(min_v))

            n_time_bins_for_epoch: int = epoch_time_bin_counts[an_epoch_idx]

            for time_bin_index in np.arange(n_time_bins_for_epoch):
                # slice per time bin using same ndim-conditional logic
                if np.ndim(a_posterior_p_x_given_n_all_t) > 2:
                    a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, :, int(time_bin_index)])
                else:
                    a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, int(time_bin_index)])

                a_posterior_p_x_given_n = a_posterior_p_x_given_n * multiplier_factor

                # ensure 2D matrix (n_xbins, n_ybins)
                if np.ndim(a_posterior_p_x_given_n) == 1:
                    a_posterior_p_x_given_n = np.atleast_2d(a_posterior_p_x_given_n).T

                n_x, n_y = np.shape(a_posterior_p_x_given_n)
                assert n_x == n_xbins, f"epoch {an_epoch_idx}, time_bin_index {time_bin_index}: n_x ({n_x}) != len(xbin_centers) ({n_xbins})"

                if n_y != n_ybins:
                    if (n_y == 1) and (n_ybins > 1):
                        # tile single y across all available y-bins
                        a_posterior_p_x_given_n = np.tile(a_posterior_p_x_given_n, (1, n_ybins))
                        n_x, n_y = np.shape(a_posterior_p_x_given_n)
                    else:
                        raise AssertionError(f"epoch {an_epoch_idx}, time_bin_index {time_bin_index}: n_y ({n_y}) != len(ybin_centers) ({n_ybins}) and cannot be safely broadcast.")

                posterior_volume[an_epoch_idx, int(time_bin_index), :, :] = a_posterior_p_x_given_n
                time_bin_centers_matrix[an_epoch_idx, int(time_bin_index)] = a_time_bin_centers[int(time_bin_index)]

        self.posterior_volume = posterior_volume
        self.time_bin_centers_matrix = time_bin_centers_matrix
        return posterior_volume, time_bin_centers_matrix


    @function_attributes(short_name=None, tags=['napari', 'posterior', 'viewer', 'ui'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryNapariPlotter.build_posterior_volume'], used_by=[], creation_date='2025-12-23 08:05', related_items=['DecodedTrajectoryPyVistaPlotter', 'napari_from_layers_dict'])
    def build_ui(self, viewer: Optional["napari.viewer.Viewer"] = None, layer_name: str = 'decoded_posterior', title: str = 'Decoded Posterior', create_logging_dock: Optional[bool] = None, logging_dock_area: Optional[str] = None, logging_dock_name: Optional[str] = None, **viewer_kwargs) -> Tuple["napari.viewer.Viewer", Any]:
        """Builds the Napari viewer and image layer showing the decoded posterior.

        If `viewer` is None, a new `napari.Viewer` is created. Otherwise the provided viewer is used.

        Args:
            viewer: Optional Napari viewer instance. If None, a new viewer is created.
            layer_name: Name for the image layer (default: 'decoded_posterior')
            title: Title for the Napari viewer window (default: 'Decoded Posterior')
            create_logging_dock: If True, automatically creates a custom logging dock widget. 
                                If None, uses the instance field value (default: None, uses self.create_logging_dock)
            logging_dock_area: Dock area for logging widget ('left', 'right', 'top', 'bottom').
                              If None, uses the instance field value (default: None, uses self.logging_dock_area)
            logging_dock_name: Name for the logging dock widget.
                              If None, uses the instance field value (default: None, uses self.logging_dock_name)
            **viewer_kwargs: Additional keyword arguments passed to napari.Viewer()

        Returns:
            viewer: the Napari viewer instance
            image_layer: the created image layer containing the posterior volume
        """
        # local import to avoid hard dependency if Napari is not installed in non-Napari contexts
        import napari

        posterior_volume, _ = self.build_posterior_volume()

        if viewer is None:
            viewer = napari.Viewer(title=title, **viewer_kwargs)

        image_layer = viewer.add_image(posterior_volume, name=layer_name, colormap='viridis', blending='additive', interpolation='nearest')

        # axes: (epoch, time_bin, xbin, ybin)
        viewer.dims.axis_labels = ('epoch', 'time_bin', 'xbin', 'ybin')
        # Ensure the epoch slider appears above the time_bin slider in the dims panel.
        # In napari, the visual order of sliders is controlled by dims.order.
        # Swapping the first two entries orders the corresponding sliders (epoch, time_bin).
        viewer.dims.order = (1, 0, 2, 3)

        # initialize current step
        epoch_idx = int(self.curr_epoch_idx) if self.curr_epoch_idx is not None else 0
        time_idx = int(self.curr_time_bin_index) if self.curr_time_bin_index is not None else 0
        viewer.dims.current_step = (epoch_idx, time_idx, 0, 0)

        self.viewer = viewer
        self.image_layer = image_layer

        # def _on_current_step_change(event):
        #     """Keep internal indices synchronized with Napari sliders."""
        #     if event is None:
        #         return
        #     if not hasattr(event, 'value'):
        #         return
        #     curr_step = event.value
        #     if len(curr_step) >= 2:
        #         self.curr_epoch_idx = int(curr_step[0])
        #         self.curr_time_bin_index = int(curr_step[1])

        # viewer.dims.events.current_step.connect(_on_current_step_change)

        
        viewer.dims.events.current_step.connect(self.on_current_step_change)

        # Optionally create custom logging dock widget
        # Use instance field values if parameters are None
        should_create_dock = create_logging_dock if create_logging_dock is not None else self.create_logging_dock
        dock_area = logging_dock_area if logging_dock_area is not None else self.logging_dock_area
        dock_name = logging_dock_name if logging_dock_name is not None else self.logging_dock_name
        
        if should_create_dock:
            self.add_custom_logging_dock(area=dock_area, name=dock_name)

        return viewer, image_layer


    def _log_to_console(self, message: str):
        """Output message to both stdout and Napari Console (if available).
        
        Args:
            message: The message string to output
        """
        # Always output to stdout (maintains existing behavior)
        print(message)
        
        # Also output to Napari Console if viewer exists and console is available
        if self.viewer is not None:
            try:
                # Try to access the Napari console widget
                if hasattr(self.viewer, 'window') and self.viewer.window is not None:
                    qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
                    if qt_viewer is not None:
                        dock_console = getattr(qt_viewer, 'dockConsole', None)
                        if dock_console is not None:
                            # Get the console widget
                            console_widget = getattr(dock_console, 'widget', None)
                            if console_widget is not None:
                                # Try to get the IPython console kernel
                                kernel = getattr(console_widget, 'kernel', None)
                                if kernel is not None:
                                    # Execute print statement in the console
                                    kernel.execute(f"print({repr(message)})")
                                else:
                                    # Fallback: try to write directly to console output
                                    console_output = getattr(console_widget, 'console', None)
                                    if console_output is not None:
                                        console_output.write(message + '\n')
            except Exception:
                # Silently fail if console access doesn't work - stdout output is sufficient
                pass
        
        # Also output to custom logging widget if it exists
        if self.custom_logging_widget is not None:
            try:
                # Try to call append_log method (for LogViewer-style widgets)
                if hasattr(self.custom_logging_widget, 'append_log'):
                    self.custom_logging_widget.append_log(message)
                # Fallback: try write_to_log method (for LogViewer from codebase)
                elif hasattr(self.custom_logging_widget, 'write_to_log'):
                    self.custom_logging_widget.write_to_log(message)
                # Fallback: try append method (for QTextEdit/QPlainTextEdit)
                elif hasattr(self.custom_logging_widget, 'append'):
                    self.custom_logging_widget.append(message)
            except Exception:
                # Silently fail if custom widget write doesn't work - stdout output is sufficient
                pass


    @function_attributes(short_name=None, tags=['napari', 'logging', 'dock', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-05 00:00', related_items=[])
    def add_custom_logging_dock(self, area: str = 'bottom', name: str = 'Custom Log') -> Optional[Any]:
        """Adds a custom scrolling text/console dock widget to the Napari window for logging.
        
        Creates a dock widget containing a read-only scrolling text area that displays log messages.
        The widget is integrated with `_log_to_console()` method to automatically receive log messages.
        
        Args:
            area: Dock area placement ('left', 'right', 'top', 'bottom'). Default is 'bottom'.
            name: Name/title for the dock widget. Default is 'Custom Log'.
            
        Returns:
            The created logging widget if successful, None if creation fails.
            
        Raises:
            None - all errors are handled gracefully and logged as warnings.
            
        Usage:
            logging_widget = napari_plotter.add_custom_logging_dock(area='bottom', name='Debug Log')
            if logging_widget is not None:
                print("Custom logging dock created successfully")
        """
        if self.viewer is None:
            logger.warning("add_custom_logging_dock: viewer is None, cannot create dock widget")
            return None
        
        try:
            # Try to import Qt widgets
            try:
                from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore
            except ImportError:
                # Fallback: try direct Qt import
                try:
                    from qtpy.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton
                    from qtpy.QtCore import QTimer
                    QtWidgets = type('QtWidgets', (), {
                        'QWidget': QWidget,
                        'QVBoxLayout': QVBoxLayout,
                        'QTextEdit': QTextEdit,
                        'QPushButton': QPushButton
                    })()
                    QtCore = type('QtCore', (), {'QTimer': QTimer})()
                except ImportError:
                    logger.warning("add_custom_logging_dock: Qt widgets not available, cannot create dock widget")
                    return None
            
            # Create custom logging widget (similar to LogViewer but simpler)
            class CustomLoggingWidget(QtWidgets.QWidget):
                """Simple scrolling text widget for displaying log messages."""
                def __init__(self, parent=None):
                    super().__init__(parent)
                    layout = QtWidgets.QVBoxLayout(self)
                    layout.setContentsMargins(0, 0, 0, 0)
                    
                    # Create read-only text edit for log display
                    self.log_display = QtWidgets.QTextEdit(self)
                    self.log_display.setReadOnly(True)
                    layout.addWidget(self.log_display)
                    
                    # Optional: Add clear button
                    clear_button = QtWidgets.QPushButton('Clear Log', self)
                    clear_button.clicked.connect(self.clear_log)
                    layout.addWidget(clear_button)
                
                def append_log(self, message: str):
                    """Append a message to the log display."""
                    self.log_display.append(message)
                    # Auto-scroll to bottom
                    scrollbar = self.log_display.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
                
                def clear_log(self):
                    """Clear the log display."""
                    self.log_display.clear()
            
            # Create the widget instance
            logging_widget = CustomLoggingWidget()
            
            # Add as dock widget to Napari window
            if not hasattr(self.viewer, 'window') or self.viewer.window is None:
                logger.warning("add_custom_logging_dock: viewer.window is not available")
                return None
            
            try:
                self.viewer.window.add_dock_widget(logging_widget, area=area, name=name)
                self.custom_logging_widget = logging_widget
                logger.debug(f"add_custom_logging_dock: Successfully created dock widget '{name}' in area '{area}'")
                return logging_widget
            except Exception as e:
                logger.warning(f"add_custom_logging_dock: Failed to add dock widget: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"add_custom_logging_dock: Failed to create custom logging dock widget: {e}")
            return None


    def on_current_step_change(self, event):
        """Update Napari sliders to match internal state.
        
        This method should be called if internal state (curr_epoch_idx, curr_time_bin_index)
        is modified programmatically to ensure the sliders reflect the current state.
        
        Returns:
            bool: True if sliders were updated, False if viewer doesn't exist

        Updates:
            self.curr_epoch_idx, self.curr_time_bin_index

        Called by `viewer.dims.events.current_step.connect(self.on_current_step_change)`
        
        """
        either_did_change: bool = False
        if event is None:
            return either_did_change
        if not hasattr(event, 'value'):
            return either_did_change
        curr_step = event.value
        if len(curr_step) >= 2:
            old_epoch_idx = deepcopy(self.curr_epoch_idx)
            old_time_bin_idx = deepcopy(self.curr_time_bin_index)

            new_epoch_idx = int(curr_step[0])
            new_time_bin_index = int(curr_step[1])
            
            epoch_idx_did_change: bool = (old_epoch_idx != new_epoch_idx)
            time_bin_index_did_change: bool = (old_time_bin_idx != new_time_bin_index)

            either_did_change = (epoch_idx_did_change or time_bin_index_did_change)
            
            # self.curr_epoch_idx = int(curr_step[0])
            # self.curr_time_bin_index = int(curr_step[1])

            ## apply the change
            self.curr_epoch_idx = new_epoch_idx
            self.curr_time_bin_index = new_time_bin_index

        if either_did_change:
            self.sync_sliders_from_state(block_updates=True)

        return either_did_change



    @function_attributes(short_name=None, tags=['napari', 'sync', 'state'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-05 00:00', related_items=[])
    def sync_sliders_from_state(self, block_updates:bool=True):
        """Update Napari sliders to match internal state.
        
        This method should be called if internal state (curr_epoch_idx, curr_time_bin_index)
        is modified programmatically to ensure the sliders reflect the current state.
        
        Args:
            block_updates: If True, temporarily blocks all event callbacks to prevent recursion.
                          Default is True.
        
        Returns:
            bool: True if sliders were updated, False if viewer doesn't exist
        """
        logger.debug(f"sync_sliders_from_state called: block_updates={block_updates}, viewer={'exists' if self.viewer is not None else 'None'}")
        
        if self.viewer is not None:
            # Get current viewer state for comparison
            current_viewer_step = tuple(self.viewer.dims.current_step) if hasattr(self.viewer.dims, 'current_step') else None
            target_step = (
                self.curr_epoch_idx if self.curr_epoch_idx is not None else 0,
                self.curr_time_bin_index if self.curr_time_bin_index is not None else 0,
                0,
                0
            )
            
            logger.debug(f"sync_sliders_from_state: current internal state (epoch={self.curr_epoch_idx}, time_bin={self.curr_time_bin_index})")
            logger.debug(f"sync_sliders_from_state: current viewer step={current_viewer_step}, target step={target_step}")
            
            if block_updates:
                # Block all callbacks temporarily to prevent recursion
                logger.debug("sync_sliders_from_state: blocking current_step events to prevent recursion")
                with self.viewer.dims.events.current_step.blocker():
                    logger.debug(f"sync_sliders_from_state: setting viewer.dims.current_step to {target_step}")
                    try:
                        self.viewer.dims.current_step = target_step
                        
                        # Verify the change was applied
                        new_viewer_step = tuple(self.viewer.dims.current_step)
                        if new_viewer_step[:2] == target_step[:2]:
                            logger.debug(f"sync_sliders_from_state: successfully updated viewer step to {new_viewer_step}")
                        else:
                            logger.warning(f"sync_sliders_from_state: viewer step mismatch! Expected {target_step[:2]}, got {new_viewer_step[:2]}")
                    except Exception as e:
                        logger.error(f"sync_sliders_from_state: error setting viewer.dims.current_step: {e}", exc_info=True)
                        raise
                logger.debug("sync_sliders_from_state: unblocked current_step events")
            else:
                # Don't block, just set the value
                logger.debug(f"sync_sliders_from_state: setting viewer.dims.current_step to {target_step} (not blocking)")
                try:
                    self.viewer.dims.current_step = target_step
                    
                    # Verify the change was applied
                    new_viewer_step = tuple(self.viewer.dims.current_step)
                    if new_viewer_step[:2] == target_step[:2]:
                        logger.debug(f"sync_sliders_from_state: successfully updated viewer step to {new_viewer_step}")
                    else:
                        logger.warning(f"sync_sliders_from_state: viewer step mismatch! Expected {target_step[:2]}, got {new_viewer_step[:2]}")
                except Exception as e:
                    logger.error(f"sync_sliders_from_state: error setting viewer.dims.current_step: {e}", exc_info=True)
                    raise
            
            return True
        else:
            logger.debug("sync_sliders_from_state: viewer is None, returning False")
        return False


    @function_attributes(short_name=None, tags=['napari', 'peak-counts', 'posterior', 'layer'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryNapariPlotter.build_posterior_volume'], used_by=[], creation_date='2026-01-05 00:00', related_items=['PosteriorPeaksPeakProminence2dResult'])
    def add_peak_counts_layer(self, peak_prominence_result: "PosteriorPeaksPeakProminence2dResult", layer_name: str = 'peak_counts', colormap: str = 'plasma', blending: str = 'additive') -> Any:
        """Adds the peak_counts.raw counter map from a PosteriorPeaksPeakProminence2dResult as a new Napari image layer.

        The peak_counts.raw is a 2D array (n_xbins, n_ybins) that gets broadcast to match the posterior_volume
        shape (num_epochs, max_num_time_bins, n_xbins, n_ybins) by repeating across all epochs and time bins.

        Args:
            peak_prominence_result: PosteriorPeaksPeakProminence2dResult object containing peak_counts
            layer_name: Name for the Napari layer (default: 'peak_counts')
            colormap: Colormap to use for the peak counts layer (default: 'plasma')
            blending: Blending mode for the layer (default: 'additive')

        Returns:
            The created Napari image layer containing the peak counts volume

        Raises:
            AssertionError: If viewer or posterior_volume hasn't been built yet, or if coordinate dimensions don't match

        Usage:
            peak_counts_layer = napari_plotter.add_peak_counts_layer(peak_prominence_result=a_result_posterior_peaks)
            peak_counts_layer
        """
        # local import to avoid hard dependency if Napari is not installed in non-Napari contexts
        import napari
        from pyphoplacecellanalysis.External.peak_prominence2d import PosteriorPeaksPeakProminence2dResult, DecodedEpochIndex, DecodedEpochTimeBinIndex

        # Ensure posterior_volume has been built
        if self.posterior_volume is None:
            self.build_posterior_volume()

        # Ensure viewer exists
        if self.viewer is None:
            # If no viewer exists, create one (though typically build_ui should be called first)
            self.viewer = napari.Viewer(title='Decoded Posterior')

        # Extract peak_counts.raw (2D array: n_xbins, n_ybins)
        peak_counts_2d = peak_prominence_result.peak_counts.raw

        # Verify coordinate alignment
        assert peak_prominence_result.xx is not None, "peak_prominence_result.xx (xbin_centers) is required"
        assert peak_prominence_result.yy is not None, "peak_prominence_result.yy (ybin_centers) is required"
        
        # Check that dimensions match
        posterior_shape = self.posterior_volume.shape  # (num_epochs, max_num_time_bins, n_xbins, n_ybins)
        n_xbins_posterior = posterior_shape[2]
        n_ybins_posterior = posterior_shape[3]
        
        peak_counts_shape = peak_counts_2d.shape
        n_xbins_peaks = peak_counts_shape[0]
        n_ybins_peaks = peak_counts_shape[1] if len(peak_counts_shape) > 1 else 1

        assert n_xbins_peaks == n_xbins_posterior, f"xbin dimension mismatch: peak_counts has {n_xbins_peaks} xbins but posterior_volume has {n_xbins_posterior}"
        assert n_ybins_peaks == n_ybins_posterior, f"ybin dimension mismatch: peak_counts has {n_ybins_peaks} ybins but posterior_volume has {n_ybins_posterior}"

        # Broadcast 2D peak_counts to 4D shape: (num_epochs, max_num_time_bins, n_xbins, n_ybins)
        num_epochs = posterior_shape[0]
        max_num_time_bins = posterior_shape[1]
        
        # Use np.tile to repeat the 2D array across epochs and time bins
        peak_counts_4d = np.tile(peak_counts_2d[np.newaxis, np.newaxis, :, :], (num_epochs, max_num_time_bins, 1, 1))

        # Add as new Napari image layer
        peak_counts_layer = self.viewer.add_image(peak_counts_4d, name=layer_name, colormap=colormap, blending=blending, interpolation='nearest')

        return peak_counts_layer


    @function_attributes(short_name=None, tags=['napari', 'peak-counts', 'posterior', 'layer'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryNapariPlotter.build_posterior_volume'], used_by=[], creation_date='2026-01-05 00:00', related_items=['PosteriorPeaksPeakProminence2dResult'])
    def add_peak_contours_layer(self, peak_prominence_result: "PosteriorPeaksPeakProminence2dResult", layer_name: str = 'peak_contours', 
                                edge_color: str = 'transparent', face_color: str = '#ffaaff78', edge_width: float = 0.005,
                                # edge_color: str = 'white', face_color: str = 'transparent', edge_width: float = 1.0,
                                ) -> Any:
        """Adds peak contours as a Napari shapes layer that updates dynamically when epoch and time_bin sliders change.

        The contours are extracted from the peak_prominence_result and displayed as shapes that update
        based on the current epoch and time_bin indices in the Napari viewer.

        Args:
            peak_prominence_result: PosteriorPeaksPeakProminence2dResult object containing peak contours
            layer_name: Name for the Napari shapes layer (default: 'peak_contours')
            edge_color: Color for contour edges (default: 'red')
            face_color: Color for contour faces, use 'transparent' for no fill (default: 'transparent')
            edge_width: Width of contour edges (default: 2.0)

        Returns:
            The created Napari shapes layer containing the peak contours

        Usage:

            contours_layer = napari_plotter.add_peak_contours_layer(peak_prominence_result=a_result_posterior_peaks)

        """
        # local import to avoid hard dependency if Napari is not installed in non-Napari contexts
        import napari
        from pyphoplacecellanalysis.External.peak_prominence2d import PosteriorPeaksPeakProminence2dResult, DecodedEpochIndex, DecodedEpochTimeBinIndex, DecodedEpochTimeBinIndexTuple

        # Ensure posterior_volume has been built
        if self.posterior_volume is None:
            self.build_posterior_volume()

        # Ensure viewer exists
        if self.viewer is None:
            # If no viewer exists, create one (though typically build_ui should be called first)
            self.viewer = napari.Viewer(title='Decoded Posterior')

        # Store reference to peak_prominence_result for use in callback
        self.peak_prominence_result = peak_prominence_result
        self._log_to_console(f"[DEBUG] add_peak_contours_layer: Stored peak_prominence_result with {len(peak_prominence_result.results)} result entries")

        # Capture log method reference for use in nested functions
        log_to_console = self._log_to_console
        
        # Helper function to extract contours from peaks_dict and convert to Napari shape format
        def extract_contours_from_peaks_dict(peaks_dict: Dict) -> List[NDArray]:
            """Extract all contours from peaks_dict and convert matplotlib Path objects to Napari shape format.
            
            Converts world coordinates (xbin_centers, ybin_centers) to pixel coordinates that match
            the image layer's coordinate system. Uses the plotter's xbin_centers/ybin_centers to ensure
            coordinate system alignment.
            
            Args:
                peaks_dict: Dictionary of peaks, each containing 'level_slices' with contour information
                
            Returns:
                List of vertex arrays, each representing a contour shape for Napari in pixel coordinates
            """
            shapes_data = []
            
            # Use plotter's coordinate system to ensure alignment with image layer
            xbin_centers_plotter = self.xbin_centers
            ybin_centers_plotter = self.ybin_centers if self.ybin_centers is not None else np.array([0.0])
            
            # Helper function to convert world coordinate to pixel index using linear interpolation
            def world_to_pixel_coord(world_coords: NDArray, bin_centers: NDArray) -> NDArray:
                """Convert world coordinates to pixel indices using linear interpolation."""
                if len(bin_centers) == 0:
                    return np.zeros_like(world_coords)
                if len(bin_centers) == 1:
                    return np.zeros_like(world_coords)
                
                # Use searchsorted to find the bin index, then interpolate
                # This handles both uniform and non-uniform bin spacing
                indices = np.searchsorted(bin_centers, world_coords, side='right') - 1
                indices = np.clip(indices, 0, len(bin_centers) - 2)  # -2 because we need at least 2 points for interpolation
                
                # Linear interpolation within the bin
                lower_bounds = bin_centers[indices]
                upper_bounds = bin_centers[indices + 1]
                bin_widths = upper_bounds - lower_bounds
                
                # Avoid division by zero
                bin_widths = np.where(bin_widths > 0, bin_widths, 1.0)
                
                # Interpolate position within bin (0.0 to 1.0)
                frac = (world_coords - lower_bounds) / bin_widths
                frac = np.clip(frac, 0.0, 1.0)
                
                # Convert to pixel coordinates (indices + fractional position within bin)
                pixel_coords = indices.astype(float) + frac
                return pixel_coords
            
            for peak_id, peak_info in peaks_dict.items():
                level_slices = peak_info.get('level_slices', {})
                log_to_console(f"[DEBUG] Peak {peak_id}: level_slices keys: {list(level_slices.keys())}")
                for probe_lvl, slice_info in level_slices.items():
                    contour = slice_info.get('contour', None)
                    if contour is not None:
                        # Convert matplotlib Path to vertices array
                        # Path.vertices is Nx2 array of (x, y) coordinates in world coordinates
                        vertices_world = contour.vertices
                        log_to_console(f"[DEBUG] Peak {peak_id}, level {probe_lvl}: contour has {len(vertices_world)} vertices")
                        if len(vertices_world) > 0:
                            # Convert from world coordinates to pixel coordinates
                            vertices_pixel = np.zeros_like(vertices_world)
                            vertices_pixel[:, 0] = world_to_pixel_coord(vertices_world[:, 0], xbin_centers_plotter)
                            vertices_pixel[:, 1] = world_to_pixel_coord(vertices_world[:, 1], ybin_centers_plotter)
                            
                            # Ensure closed contour by adding first point at end if not already closed
                            if len(vertices_pixel) > 1 and not np.allclose(vertices_pixel[0], vertices_pixel[-1], atol=1e-6):
                                vertices_pixel = np.vstack([vertices_pixel, vertices_pixel[0:1]])
                            shapes_data.append(vertices_pixel)
                            log_to_console(f"[DEBUG] Added contour shape with {len(vertices_pixel)} vertices")
                        else:
                            log_to_console(f"[DEBUG] Peak {peak_id}, level {probe_lvl}: contour has 0 vertices, skipping")
                    else:
                        log_to_console(f"[DEBUG] Peak {peak_id}, level {probe_lvl}: no contour found")
            log_to_console(f"[DEBUG] extract_contours_from_peaks_dict returning {len(shapes_data)} shapes")
            return shapes_data

        # Helper function to update contours based on current epoch and time_bin
        def update_contours_for_current_indices(epoch_idx: int, time_bin_idx: int):
            """Update the shapes layer with contours for the specified epoch and time_bin indices."""
            log_to_console(f"[DEBUG] update_contours_for_current_indices called: epoch_idx={epoch_idx}, time_bin_idx={time_bin_idx}")
            if self.peak_contours_layer is None:
                log_to_console(f"[DEBUG] peak_contours_layer is None, returning early")
                return
            # active_time_bin_id: int = (time_bin_idx + 1) 
            active_time_bin_id: int = time_bin_idx # TODO: this is the right one
            log_to_console(f"[DEBUG] active_time_bin_id = {active_time_bin_id} (time_bin_idx + 1)")
            a_peaks_results: Dict[DecodedEpochTimeBinIndexTuple, Dict] = self.peak_prominence_result.results
            a_epoch_t_bin_tuple: DecodedEpochTimeBinIndexTuple = (epoch_idx, active_time_bin_id)
            log_to_console(f"[DEBUG] Looking for key: {a_epoch_t_bin_tuple}")
            log_to_console(f"[DEBUG] Available keys in results (first 10): {list(a_peaks_results.keys())[:10]}")
            log_to_console(f"[DEBUG] Total number of keys in results: {len(a_peaks_results)}")
            
            # Check if results exist for this epoch/time_bin combination
            if a_epoch_t_bin_tuple not in a_peaks_results:
                # No peaks for this combination, clear the layer
                log_to_console(f"[DEBUG] Key {a_epoch_t_bin_tuple} NOT found in results. Clearing layer.")
                self.peak_contours_layer.data = []
                return
            
            log_to_console(f"[DEBUG] Key {a_epoch_t_bin_tuple} found in results!")
            an_epoch_t_bin_peaks_result: Dict = a_peaks_results[a_epoch_t_bin_tuple]
            peaks_dict = an_epoch_t_bin_peaks_result.get('peaks', {})
            log_to_console(f"[DEBUG] peaks_dict keys: {list(peaks_dict.keys())}")
            log_to_console(f"[DEBUG] Number of peaks in peaks_dict: {len(peaks_dict)}")
            
            if len(peaks_dict) == 0:
                # Empty peaks dict, clear the layer
                log_to_console(f"[DEBUG] peaks_dict is empty. Clearing layer.")
                self.peak_contours_layer.data = []
                return
            
            # Extract and convert contours
            log_to_console(f"[DEBUG] Extracting contours from {len(peaks_dict)} peaks...")
            shapes_data = extract_contours_from_peaks_dict(peaks_dict)
            log_to_console(f"[DEBUG] Extracted {len(shapes_data)} contour shapes")
            if len(shapes_data) > 0:
                log_to_console(f"[DEBUG] First shape has {len(shapes_data[0])} vertices")
            self.peak_contours_layer.data = shapes_data
            log_to_console(f"[DEBUG] Updated peak_contours_layer.data with {len(shapes_data)} shapes")


        def _on_current_step_change_contours(event):
            """Update peak contours when epoch or time_bin slider changes.
            
            Uses internal state (self.curr_epoch_idx, self.curr_time_bin_index) which is
            synchronized by the main event handler in build_ui, ensuring consistency.

            Captures: update_contours_for_current_indices

            """
            log_to_console(f"[DEBUG] _on_current_step_change_contours callback triggered")
            if event is None:
                log_to_console(f"[DEBUG] event is None, returning")
                return
            if not hasattr(event, 'value'):
                log_to_console(f"[DEBUG] event has no 'value' attribute, returning")
                return
            curr_step = event.value
            log_to_console(f"[DEBUG] curr_step = {curr_step}, len = {len(curr_step) if hasattr(curr_step, '__len__') else 'N/A'}")
            if len(curr_step) >= 2:
                epoch_idx = int(curr_step[0])
                time_bin_idx = int(curr_step[1])
                log_to_console(f"[DEBUG] Extracted epoch_idx={epoch_idx}, time_bin_idx={time_bin_idx} from curr_step")
                update_contours_for_current_indices(epoch_idx, time_bin_idx)
            else:
                log_to_console(f"[DEBUG] curr_step length < 2, not updating. \n\tcurr_step: {curr_step}")

            # # Use internal state that is synchronized by the main handler
            # epoch_idx = self.curr_epoch_idx if self.curr_epoch_idx is not None else 0
            # time_bin_idx = self.curr_time_bin_index if self.curr_time_bin_index is not None else 0
            # print(f"[DEBUG] Using synchronized state: epoch_idx={epoch_idx}, time_bin_idx={time_bin_idx}")
            # update_contours_for_current_indices(epoch_idx, time_bin_idx)


        ## END def _on_current_step_change_contours(event)...


        # Create initial empty shapes layer
        shapes_layer = self.viewer.add_shapes(
            data=[],
            shape_type='path',
            name=layer_name,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=edge_width
        )
        # Configure shapes layer to work with 4D volume
        shapes_layer.editable = False  # Make non-editable since it's dynamically updated
        # Ensure shapes layer uses the same coordinate system as the image layer
        # Shapes will automatically be displayed in the current 2D slice (epoch, time_bin)
        self.peak_contours_layer = shapes_layer

        # Create callback function to update contours when sliders change

        # Connect callback to slider changes
        self.viewer.dims.events.current_step.connect(_on_current_step_change_contours)

        # Display initial contours for current epoch/time_bin
        # Read from actual slider positions to ensure sync with viewer state
        curr_step = self.viewer.dims.current_step
        epoch_idx = int(curr_step[0]) if len(curr_step) >= 1 else 0
        time_bin_idx = int(curr_step[1]) if len(curr_step) >= 2 else 0
        self._log_to_console(f"[DEBUG] Initial display from slider: epoch_idx={epoch_idx}, time_bin_idx={time_bin_idx}")
        # Update internal state to match slider positions
        self.curr_epoch_idx = epoch_idx
        self.curr_time_bin_index = time_bin_idx
        update_contours_for_current_indices(epoch_idx, time_bin_idx)

        return shapes_layer, update_contours_for_current_indices
