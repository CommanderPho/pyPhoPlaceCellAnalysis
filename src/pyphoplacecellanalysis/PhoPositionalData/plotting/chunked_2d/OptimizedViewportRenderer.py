from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from matplotlib.collections import PathCollection

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

from itertools import islice
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import LapsVisualizationMixin, LineCollection, _plot_helper_add_arrow # plot_lap_trajectories_2d
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

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots

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
    active_ax: Any = field()
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