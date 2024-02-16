import sys
from copy import deepcopy
from typing import Tuple, Optional, List, Dict
from enum import Enum # for TrackPositionClassification
from attrs import define, field, Factory
from collections import namedtuple
import numpy as np
import pandas as pd

from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.geometry_helpers import point_tuple_mid_point, BoundsRect, is_point_in_rect
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui
from pyphoplacecellanalysis.External.pyqtgraph import PlotItem

# Define the named tuple
ScaleFactors = namedtuple("ScaleFactors", ["major", "minor"])

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.path import Path
import matplotlib.patches as patches # for matplotlib version of the plot
from matplotlib.collections import PatchCollection

import pyvista as pv # for 3D support in `LinearTrackDimensions3D`



class TrackPositionClassification(Enum):
    """ classifying various x-positions as belonging to outside the outside_maze, the track_endcaps, or the track_body

        # TrackPositionClassification.TRACK_ENDCAPS
        # TrackPositionClassification.TRACK_BODY
        # TrackPositionClassification.OUTSIDE_MAZE
    """
    OUTSIDE_MAZE = "outside_maze"
    TRACK_ENDCAPS = "track_endcaps"
    TRACK_BODY = "track_body"

    @property
    def is_on_maze(self) -> bool:
        """ returns True if the point is anywhere on the track (including endcaps) """
        return self.value != TrackPositionClassification.OUTSIDE_MAZE.value

    @property
    def is_endcap(self) -> bool:
        return self.value == TrackPositionClassification.TRACK_ENDCAPS.value


def classify_test_point(test_point, rects) -> "TrackPositionClassification":
    """ 
        rects = [
            (-107.0, -0.125, 22.0, 0.25),
            (-85.0, -0.05, 170.0, 0.1),
            (85.0, -0.125, 22.0, 0.25)
        ]
    """
    assert len(rects) == 3, f"rects should contain three elements for (left_platform, track_body, right_platform). {rects}"
    if is_point_in_rect(test_point, rects[0]) or is_point_in_rect(test_point, rects[2]):
        return TrackPositionClassification.TRACK_ENDCAPS
    elif is_point_in_rect(test_point, rects[1]):
        return TrackPositionClassification.TRACK_BODY
    else:
        return TrackPositionClassification.OUTSIDE_MAZE

def classify_x_position(x, rects) -> "TrackPositionClassification":
    return classify_test_point((x, None), rects)




@function_attributes(short_name=None, tags=['graphics', 'track', 'shape', 'rendering'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-07 12:13', related_items=[])
@define(slots=False)
class LinearTrackDimensions:
    """ represents a linear track comprised of two equally-sized square end platforms connected by a thin (`track_width`) straight track of length `track_length`.
        The goal is to enable plotting a graphical representation of the track along the axes of the plots using pyqtgraph. 
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions
        long_track_dims = LinearTrackDimensions(track_length=100.0)
        short_track_dims = LinearTrackDimensions(track_length=70.0)

        # Variant:> With Alignment Axis specified:
        axis_kwargs = dict(alignment_axis='y')

        # Variant:> With custom `axis_scale_factors`:
        # axis_kwargs = dict(alignment_axis='y', axis_scale_factors=(1.0, 1.0/22.0))

        # Variant:> Compute axis scale factors from grid_bin_bounds
        grid_bin_bounds = long_pf1D.config.grid_bin_bounds # ((29.16, 261.7), (130.23, 150.99))

        ((x, y), (x2, y2)) = grid_bin_bounds #TODO BUG 2023-09-13 13:19: - [ ] incorrect: `((x, y), (x2, y2)) = grid_bin_bounds` -> correction: `((x, x2), (y, y2)) = grid_bin_bounds` I think this is incorrect interpretation of grid_bin_bounds everywhere I used it in the LinearTrackDimensions and LinearTrackDimensions3D, maybe explains why `LinearTrackDimensions` wasn't working right either.
        _height = abs(y2 - y)
        _width = abs(x2 - x)
        print(f'_height: {_height}, _width: {_width}')

        axis_scale_factors = (1.0/_width, 1.0/_height)
        print(f'axis_scale_factors: {axis_scale_factors}
        axis_kwargs = dict(alignment_axis='y', axis_scale_factors=axis_scale_factors)

        long_track_dims = LinearTrackDimensions(track_length=100.0, **axis_kwargs)
        short_track_dims = LinearTrackDimensions(track_length=70.0, **axis_kwargs)


    Example:

        _out = curr_active_pipeline.display('_display_grid_bin_bounds_validation', curr_active_pipeline.get_session_context(), defer_render=False, save_figure=True)
        fig = _out.figures[0]
        ax = _out.axes[0]
        long_track_dims.plot_rects(ax, offset=(0.1, 100.0))
        plt.draw()

    Example 2: Adding as new subplot
        import matplotlib.gridspec as gridspec

        ## Adds a new subplot to an existing (fig, ax) without requiring modifications in the original code!
        _out = curr_active_pipeline.display('_display_grid_bin_bounds_validation', curr_active_pipeline.get_session_context(), defer_render=False, save_figure=True)
        fig = _out.figures[0]
        ax = _out.axes[0]

        # Get the current gridspec from ax
        gs = ax.get_subplotspec().get_gridspec()

        # Create a new gridspec with an additional column
        gs_new = gridspec.GridSpec(1, 2, width_ratios=[1, 0.5]) # new column is half the width of the current one

        # Reposition the existing ax using the new gridspec
        ax.set_position(gs_new[0, 0].get_position(fig))

        # Add a new subplot in the new column
        ax2 = fig.add_subplot(gs_new[0, 1])


        ax2.plot(np.cos(np.linspace(0, 10, 100)))
        # ax2.cla()
        # long_track_dims.plot_rects(ax2)

        # long_track_dims.plot_rects(ax2, offset=(0.1, 0.0))

        plt.tight_layout()
        plt.show()


    """
    # all units in [cm]
    track_width: float = 6.2
    track_length: float = 100.0
    platform_side_length: float = 22.0
    minor_axis_platform_side_width: Optional[float] = None

    # grid_bin_bounds: Optional[BoundsRect] = None #TODO 2023-09-20 12:33: - [ ] Allow storing grid_bin_bounds to help with offset computations

    alignment_axis: str = field(default='x')
    axis_scale_factors: ScaleFactors = field(default=ScaleFactors(1.0, 1.0))  # Major and minor axis scale factors

    XPositions = namedtuple("XPositions", ["platform_start_x", "track_start_x", "track_midpoint_x", "track_end_x", "platform_stop_x"])
    YPositions = namedtuple("YPositions", ["platform_start_y", "track_start_y", "track_center_y", "track_end_y", "platform_stop_y"])

    @property
    def total_length(self) -> float:
        # unscaled total length including both end platforms
        return (self.track_length + (2.0 * self.platform_side_length))
    
    @property
    def total_width(self) -> float:
        # unscaled total width including both end platforms
        return max(self.track_width, (self.minor_axis_platform_side_width or self.platform_side_length))


    @property
    def scaled_track_length(self) -> float:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        return major_axis_factor * self.track_length

    @property
    def scaled_track_width(self) -> float:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        return minor_axis_factor * self.track_width

    @property
    def scaled_platform_size(self) -> tuple:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        minor_axis_width = (self.minor_axis_platform_side_width or self.platform_side_length)
        return ((major_axis_factor * self.platform_side_length), (minor_axis_factor * minor_axis_width))

    @property
    def scaled_total_length(self) -> float:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        return major_axis_factor * self.total_length

    @property
    def scaled_total_width(self) -> float:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        return minor_axis_factor * self.total_width

    @property
    def notable_x_positions(self) -> XPositions: # np.ndarray:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        track_midpoint_x = major_axis_factor * (self.total_length / 2.0)
        track_end_x = major_axis_factor * (self.platform_side_length + self.track_length)
        platform_end_x = self.scaled_total_length # major_axis_factor * self.total_length
        # return np.array((0.0, self.scaled_platform_size[0], track_midpoint_x, track_end_x, platform_end_x))
        return self.XPositions(0.0, self.scaled_platform_size[0], track_midpoint_x, track_end_x, platform_end_x)

    @property
    def notable_y_positions(self) -> YPositions: # np.ndarray:
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        track_center_y = max(self.scaled_platform_size[1], self.scaled_track_width) / 2.0
        track_origin_y = track_center_y - (self.scaled_track_width / 2.0)
        track_top_y = track_center_y + (self.scaled_track_width / 2.0)
        # return np.array((0.0, track_origin_y, track_center_y, track_top_y, self.scaled_platform_size[1]))
        return self.YPositions(0.0, track_origin_y, track_center_y, track_top_y, self.scaled_platform_size[1])

    def get_center_point(self):
        return (self.total_length/2.0,
                self.total_width/2.0)
    
    def compute_position_offset(self, grid_bin_bounds):
        # This mode computes the correct position_offset point from the grid_bin_bounds provided and self's center properties
        assert grid_bin_bounds is not None
        assert len(grid_bin_bounds) == 2, f"{grid_bin_bounds} should be a tuple of length 2"
        grid_bin_bounds_center_point = (point_tuple_mid_point(grid_bin_bounds[0]), point_tuple_mid_point(grid_bin_bounds[1]))
        a_position_offset = grid_bin_bounds_center_point #- a_track_dims.get_center_point()
        a_center_correction_offset = ((self.platform_side_length/2.0)-self.get_center_point()[0], 0.0)
        a_position_offset = np.array(a_position_offset) + np.array(a_center_correction_offset)
        assert len(a_position_offset) == 2, f"{a_position_offset} should be of length 2"
        return a_position_offset

    @classmethod
    def compute_offset_notable_positions(cls, notable_x_positions, notable_y_positions, is_zero_centered:bool=False, offset_point=None):
        """ computes offset

        notable_x_positions, notable_y_positions = self.compute_offset_notable_positions(notable_x_positions, notable_y_positions, is_zero_centered=is_zero_centered, offset_point=offset_point)

        factored out of `_build_component_notable_positions`
        """
        if offset_point is not None:
            assert len(offset_point) == 2, f"offset_point should be a point like (offset_x, offset_y) but was {offset_point}"
            # assert is_zero_centered == True, f"is_zero_centered should always be True when using offset_point!"
            is_zero_centered = True # always uses zero-centered

        if is_zero_centered:
            # returns: array([-107, -85, 0, 85, 107])
            notable_x_positions = notable_x_positions - (notable_x_positions[-1]/2.0)
            notable_y_positions = notable_y_positions - (notable_y_positions[-1]/2.0)
            
        if offset_point is not None:
            notable_x_positions = notable_x_positions + offset_point[0]
            notable_y_positions = notable_y_positions + offset_point[1]

        return notable_x_positions, notable_y_positions

    @classmethod
    def compute_offset_rects(cls, total_length, total_width, rects, is_zero_centered:bool=False, offset_point=None):
        """ 

        rects = compute_offset_rects(self.total_length, self.total_width, rects, is_zero_centered=is_zero_centered, offset_point=offset_point)
        
        factored out of `_build_component_rectangles`

        """
        if offset_point is not None:
            assert len(offset_point) == 2, f"offset_point should be a point like (offset_x, offset_y) but was {offset_point}"
            # assert is_zero_centered == True, f"is_zero_centered should always be True when using offset_point!"
            is_zero_centered = True # always uses zero-centered
            

        x_extent_midpoint: float = (total_length/2.0) # must capture these before updating them
        y_extent_midpoint: float = (total_width/2.0) # must capture these before updating them

        # x_extent_midpoint: float = (notable_x_positions[-1]/2.0) # must capture these before updating them
        # y_extent_midpoint: float = (notable_y_positions[-1]/2.0) # must capture these before updating them

        if is_zero_centered:
            offset_rects = []
            for a_rect in rects:
                an_offset_rect = list(a_rect)
                an_offset_rect[0] = a_rect[0] - x_extent_midpoint
                an_offset_rect[1] = a_rect[1] - y_extent_midpoint
                offset_rects.append(tuple(an_offset_rect))
            rects = offset_rects

        if offset_point is not None:
            offset_rects = []
            for a_rect in rects:
                an_offset_rect = list(a_rect)
                an_offset_rect[0] = a_rect[0] + offset_point[0]
                an_offset_rect[1] = a_rect[1] + offset_point[1]
                offset_rects.append(tuple(an_offset_rect))
            rects = offset_rects

        return rects


    @classmethod
    def init_from_grid_bin_bounds(cls, grid_bin_bounds, debug_print=False):
        """ Builds the object and the maze mesh data from the grid_bin_bounds provided.
        
        ## Add the 3D Maze Shape
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions, LinearTrackDimensions3D
            from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena

            a_track_dims = LinearTrackDimensions3D()
            a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(grid_bin_bounds, return_geoemtry=True)
            ipspikesDataExplorer.plots['maze_bg_ideal'] = perform_plot_flat_arena(pActiveSpikesBehaviorPlotter, ideal_maze_pdata, name='idealized_maze_bg', label='idealized_maze', color=[1.0, 0.3, 0.3]) # [0.3, 0.3, 0.3]


        """
        ((x, x2), (y, y2)) = grid_bin_bounds
        _length, _width = abs(x2 - x), abs(y2 - y)
        if debug_print:
            print(f'_length: {_length}, _width: {_width}')
        _obj = cls()
        deduced_track_length: float = _length - (_obj.platform_side_length * 2.0)
        if debug_print:
            print(f'deduced_track_length: {deduced_track_length}')
        _obj.track_length = deduced_track_length
        ## TODO for now just keep the track_width and platform_side_lengths fixed, ignoring the grid_bin_bounds, since we know those from physical dimension measurements        
        # grid_bin_bounds_extents = (x, y, _length, _width)
        # axis_scale_factors = (1.0/_length, 1.0/_width)
        
        return _obj
        

    def _build_component_notable_positions(self, major_axis_factor:float=1.0, minor_axis_factor:float=1.0, is_zero_centered:bool=False, offset_point=None):
        """ builds 1D dimension lines 
        Allows specifying arbitrary `major_axis_factor:float=1.0, minor_axis_factor:float=1.0` unlike the `self` version
        """
        if self.alignment_axis == 'x':
            scaled_track_length = (major_axis_factor * self.track_length)
            scaled_track_width = (minor_axis_factor * self.track_width)
            scaled_platform_size = ((major_axis_factor * self.platform_side_length), (minor_axis_factor * (self.minor_axis_platform_side_width or self.platform_side_length)))
            
            track_center_y = minor_axis_factor * (self.platform_side_length / 2.0)
            track_origin_y = track_center_y - (scaled_track_width / 2.0) # find the bottom of the track rectangle
            track_top_y = track_center_y + (scaled_track_width / 2.0)
            
            # Aims to position the bottom-left corner of each rect appropriately
            track_midpoint_x: float = (major_axis_factor * (self.total_length/2.0))
            track_end_x: float = (major_axis_factor * (self.platform_side_length + self.track_length))
            platform_end_x: float = (major_axis_factor * self.total_length)
            # (platform_start, track_start, track_midpoint, track_end, platform_stop)
            notable_x_positions = np.array((0.0, scaled_platform_size[0], track_midpoint_x, track_end_x, platform_end_x)) # (platform_start_x, track_start_x, track_midpoint_x, track_end_x, platform_stop_x)
            notable_y_positions = np.array((0.0, track_origin_y, track_center_y, track_top_y, scaled_platform_size[1])) # (platform_start_y, track_start_y, track_center_y, track_end_y, platform_stop_y)

            # TODO 2023-09-20 - Use new self.notable_*_positions properties:
            # notable_x_positions = np.array(self.notable_x_positions) # (platform_start_x, track_start_x, track_midpoint_x, track_end_x, platform_stop_x)
            # notable_y_positions = np.array(self.notable_y_positions) # (platform_start_y, track_start_y, track_center_y, track_end_y, platform_stop_y)

        elif self.alignment_axis == 'y':
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported alignment_axis: {self.alignment_axis}")
            
        notable_x_positions, notable_y_positions = self.compute_offset_notable_positions(notable_x_positions, notable_y_positions, is_zero_centered=is_zero_centered, offset_point=offset_point)
        return notable_x_positions, notable_y_positions

    def _build_component_rectangles(self, is_zero_centered:bool=False, offset_point=None, include_rendering_properties:bool=True):
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        if include_rendering_properties:
            pen = pg.mkPen({'color': "#FF0", 'width': 2})
            brush = pg.mkBrush("#FF0")
            rendering_properties_tuple = (pen, brush)
        else:
            # rendering_properties_tuple = tuple() # omit entirely?
            rendering_properties_tuple = (None, None) # include two None values to allow for loop unwrapping

        if self.alignment_axis == 'x':
            # self.scaled_track_length = (major_axis_factor * self.track_length)
            # self.scaled_track_width = (minor_axis_factor * self.track_width)
            # self.scaled_platform_size = ((major_axis_factor * self.platform_side_length), (minor_axis_factor * (self.minor_axis_platform_side_width or self.platform_side_length)))
            
            track_center_y = (max(self.scaled_platform_size[1], self.scaled_track_width) / 2.0) # assumes platform is thicker than track.
            track_origin_y = track_center_y - (self.scaled_track_width / 2.0) # find the bottom of the track rectangle
            # track_top_y = track_center_y + (self.scaled_track_width / 2.0)

            # # Aims to position the bottom-left corner of each rect appropriately
            # track_midpoint_x: float = (major_axis_factor * (self.total_length/2.0))
            track_end_x: float = (major_axis_factor * (self.platform_side_length + self.track_length))
            # platform_end_x: float = self.scaled_total_length
            # notable_x_positions = np.array((0.0, self.scaled_platform_size[0], track_midpoint_x, track_end_x, platform_end_x)) # (platform_start_x, track_start_x, track_midpoint_x, track_end_x, platform_stop_x)
            # notable_y_positions = np.array((0.0, track_origin_y, track_center_y, track_top_y, self.scaled_platform_size[1])) # (platform_start_y, track_start_y, track_center_y, track_end_y, platform_stop_y)

            notable_x_positions = np.array(self.notable_x_positions) # (platform_start_x, track_start_x, track_midpoint_x, track_end_x, platform_stop_x)
            notable_y_positions = np.array(self.notable_y_positions) # (platform_start_y, track_start_y, track_center_y, track_end_y, platform_stop_y)

            # a_rect: [x, y, length, width, *rendering_properties_tuple]
            rects = [
                (0, 0, *self.scaled_platform_size, *rendering_properties_tuple),
                (self.scaled_platform_size[0], track_origin_y, self.scaled_track_length, self.scaled_track_width, *rendering_properties_tuple),
                (track_end_x, 0, *self.scaled_platform_size, *rendering_properties_tuple)
            ]

        elif self.alignment_axis == 'y':
            raise NotImplementedError # never converted to `(self.minor_axis_platform_side_width or self.platform_side_length)` or maintained
            track_center_x = major_axis_factor * self.platform_side_length / 2.0
            track_origin_x = track_center_x - minor_axis_factor * self.track_width / 2.0
            self.scaled_platform_size = ((minor_axis_factor * self.platform_side_length), (major_axis_factor * self.platform_side_length))
            rects = [
                (0, 0, *self.scaled_platform_size, *rendering_properties_tuple),
                (track_origin_x, major_axis_factor * self.platform_side_length,
                    (minor_axis_factor * self.track_width), 
                    (major_axis_factor * self.track_length),
                    *rendering_properties_tuple),
                (0, major_axis_factor * (self.platform_side_length + self.track_length),*self.scaled_platform_size, *rendering_properties_tuple)
            ]
        else:
            raise ValueError(f"Unsupported alignment_axis: {self.alignment_axis}")

            
        rects = self.compute_offset_rects(self.total_length, self.total_width, rects, is_zero_centered=is_zero_centered, offset_point=offset_point)

        return rects
        
    def plot_rects(self, plot_item, offset=None, matplotlib_rect_kwargs_override=None):
        """ main function to plot 

        
        combined_item, rect_items, rects = item.plot_rect(ax, offset=None)
        """
        rects = self._build_component_rectangles(is_zero_centered=True, offset_point=offset, include_rendering_properties=True)
        rect_items = [] # probably do not need
        for x, y, w, h, pen, brush in rects:
            if isinstance(plot_item, PlotItem):
                rect_item = QtGui.QGraphicsRectItem(x, y, w, h)
                rect_item.setPen(pen)
                rect_item.setBrush(brush)
                rect_items.append(rect_item)
                plot_item.addItem(rect_item)
            elif isinstance(plot_item, matplotlib.axes.Axes):
                import matplotlib.patches as patches
                # matplotlib ax was passed
                if matplotlib_rect_kwargs_override is not None:
                    matplotlib_rect_kwargs = matplotlib_rect_kwargs_override
                else:
                    matplotlib_rect_kwargs = ColorFormatConverter.convert_pen_brush_to_matplot_kwargs(pen, brush) # linewidth=2, edgecolor='red', facecolor='red'                
                rect = patches.Rectangle((x, y), w, h, **matplotlib_rect_kwargs)
                plot_item.add_patch(rect)                
                rect_items.append(rect)
            else:
                raise ValueError("Unsupported plot item type.")

        if isinstance(plot_item, PlotItem):
            plot_item.setAspectLocked()
            plot_item.setClipToView(True)
            combined_item = None
        elif isinstance(plot_item, matplotlib.axes.Axes):
            # Combine patches into a PatchCollection
            # patches = [rect1, rect2, rect3]
            # combined_item = PatchCollection(rect_items, linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
            # plot_item.set_aspect('equal', 'box')
            combined_item = None

        else:
            raise ValueError("Unsupported plot item type.")
        return combined_item, rect_items, rects

    def plot_line_collections(self, plot_item):
        long_notable_x_positions, _long_notable_y_positions = self._build_component_notable_positions(offset_point=(x_midpoint, y_midpoint))
        # Omit the midpoint
        long_notable_x_platform_positions = long_notable_x_positions[[0,1,3,4]]
        long_track_line_collection: matplotlib.collections.LineCollection = plt.vlines(long_notable_x_platform_positions, label='long_track_x_pos_lines', ymin=plot_item.get_ybound()[0], ymax=plot_item.get_ybound()[1], colors='#0000FFAA', linestyles='dashed') # matplotlib.collections.LineCollection
        return long_track_line_collection

# ==================================================================================================================== #
# 3D Support                                                                                                           #
# ==================================================================================================================== #

def get_bounds(center, x_length, y_length, z_length):
    xMin, xMax = center[0] - x_length/2, center[0] + x_length/2
    yMin, yMax = center[1] - y_length/2, center[1] + y_length/2
    zMin, zMax = center[2] - z_length/2, center[2] + z_length/2
    return (xMin, xMax, yMin, yMax, zMin, zMax)


@define(slots=False)
class LinearTrackDimensions3D(LinearTrackDimensions):
    """ extends the linear track shape to 3D for use in pyvista 3D plots 
     
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions3D

    a_track_dims = LinearTrackDimensions3D()
    merged_boxes_pdata = a_track_dims.build_maze_geometry()

    ## Plotting:
    plotter = pv.Plotter()

    # # Add boxes to the plotter
    # plotter.add_mesh(platform1, color="blue")
    # plotter.add_mesh(platform2, color="red")
    # plotter.add_mesh(track_body, color="green")

    # Add the merged geometry to the plotter
    plotter.add_mesh(merged_boxes_pdata, color="lightgray")

    # Show the plot
    plotter.show()

     
    """
    box_thickness: float = field(default=1.0) #= size/10.0
    track_thickness: float = field(default=1.0) # = size/10.0

    def get_center_point(self):
        return (self.total_length/2.0,
                self.total_width/2.0,
                self.box_thickness/2.0)
    
    def compute_position_offset(self, grid_bin_bounds):
        # This mode computes the correct position_offset point from the grid_bin_bounds provided and self's center properties
        assert grid_bin_bounds is not None
        assert len(grid_bin_bounds) == 2, f"{grid_bin_bounds} should be a tuple of length 2"
        grid_bin_bounds_center_point = (point_tuple_mid_point(grid_bin_bounds[0]), point_tuple_mid_point(grid_bin_bounds[1]))
        a_position_offset = (*grid_bin_bounds_center_point, 0.0) #- a_track_dims.get_center_point()
        a_center_correction_offset = ((self.platform_side_length/2.0)-self.get_center_point()[0], 0.0, 0.0)
        a_position_offset = np.array(a_position_offset) + np.array(a_center_correction_offset)
        return a_position_offset


    @classmethod
    def init_from_grid_bin_bounds(cls, grid_bin_bounds, return_geoemtry=True, debug_print=False):
        """ Builds the object and the maze mesh data from the grid_bin_bounds provided.
        
        ## Add the 3D Maze Shape
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions, LinearTrackDimensions3D
            from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena

            a_track_dims = LinearTrackDimensions3D()
            a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(grid_bin_bounds, return_geoemtry=True)
            ipspikesDataExplorer.plots['maze_bg_ideal'] = perform_plot_flat_arena(pActiveSpikesBehaviorPlotter, ideal_maze_pdata, name='idealized_maze_bg', label='idealized_maze', color=[1.0, 0.3, 0.3]) # [0.3, 0.3, 0.3]


        """
        ((x, x2), (y, y2)) = grid_bin_bounds #TODO BUG 2023-09-13 13:19: - [ ] incorrect: `((x, y), (x2, y2)) = grid_bin_bounds` -> correction: `((x, x2), (y, y2)) = grid_bin_bounds` I think this is incorrect interpretation of grid_bin_bounds everywhere I used it in the LinearTrackDimensions and LinearTrackDimensions3D, maybe explains why `LinearTrackDimensions` wasn't working right either.
        _length, _width = abs(x2 - x), abs(y2 - y)
        if debug_print:
            print(f'_length: {_length}, _width: {_width}')
        _obj = cls()
        deduced_track_length: float = _length - (_obj.platform_side_length * 2.0)
        if debug_print:
            print(f'deduced_track_length: {deduced_track_length}')
        _obj.track_length = deduced_track_length
        ## TODO for now just keep the track_width and platform_side_lengths fixed, ignoring the grid_bin_bounds, since we know those from physical dimension measurements        
        # grid_bin_bounds_extents = (x, y, _length, _width)
        # axis_scale_factors = (1.0/_length, 1.0/_width)
        
        # BUILD THE GEOMETRY
        if return_geoemtry:
            maze_pdata = _obj.build_maze_geometry(grid_bin_bounds=grid_bin_bounds)
            return _obj, maze_pdata
        else:
            return _obj
        
    def build_maze_geometry(self, position_offset=None, grid_bin_bounds=None):
        """ builds the maze geometry for use with pyvista.
        
        """
        size = self.platform_side_length
        
        if grid_bin_bounds is not None:
            # This mode computes the correct position_offset point from the grid_bin_bounds provided and self's center properties
            assert position_offset is None, f"grid_bin_bounds is provided and in this mode position_offset will not be used!"
            a_position_offset: np.array = self.compute_position_offset(grid_bin_bounds=grid_bin_bounds)
            # grid_bin_bounds_center_point = (point_tuple_mid_point(grid_bin_bounds[0]), point_tuple_mid_point(grid_bin_bounds[1]))
            # a_position_offset = (*grid_bin_bounds_center_point, 0.0) #- a_track_dims.get_center_point()
            # a_center_correction_offset = ((self.platform_side_length/2.0)-self.get_center_point()[0], 0.0, 0.0)
            # a_position_offset = np.array(a_position_offset) + np.array(a_center_correction_offset)
            return self.build_maze_geometry(position_offset=a_position_offset, grid_bin_bounds=None) # call `build_maze_geometry` in simple position_offset mode
            
        else:
            if position_offset is None:
                position_offset = (0, 0, -0.01) # set default
                
            # Create two square boxes
            platform1 = pv.Box(bounds=get_bounds([0, 0, 0], size, size, self.box_thickness))
            platform2 = pv.Box(bounds=get_bounds([(self.platform_side_length + self.track_length), 0, 0], size, size, self.box_thickness))

            # Create connecting box (rectangular)
            track_body_center = [(platform1.bounds[1] + platform2.bounds[0])/2, 0, 0]
            track_body = pv.Box(bounds=get_bounds(track_body_center, self.track_length, self.track_width, self.track_thickness))
            

            # Merge the three boxes into a single pv.PolyData object
            merged_boxes_pdata = platform1 + platform2 + track_body
            merged_boxes_pdata['occupancy heatmap'] = np.arange(np.shape(merged_boxes_pdata.points)[0])
            merged_boxes_pdata.translate(position_offset)

            # Separate meshes:
            # _out_tuple = (platform1, platform2, track_body)
            # _out_tuple = [a_mesh.translate((0, 0, -0.01)) for a_mesh in _out_tuple]
            # return merged_boxes_pdata , _out_tuple # usage: `merged_boxes_pdata, (platform1, platform2, track_body) = a_track_dims.build_maze_geometry()`
                                    
            return merged_boxes_pdata
     


@define(slots=False)
class LinearTrackInstance:
    """ Aims to combine the dimensions specified by `track_dimensions: LinearTrackDimensions` and the position/bounds information specified by `grid_bin_bounds: BoundsRect`
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance
    
    """
    track_dimensions: LinearTrackDimensions
    grid_bin_bounds: BoundsRect #= None #TODO 2023-09-20 12:33: - [ ] Allow storing grid_bin_bounds to help with offset computations
    
    @property
    def rects(self):
        offset_point = self.grid_bin_bounds.center_point # (self.grid_bin_bounds.center_point[0], 0.75)
        return self.track_dimensions._build_component_rectangles(is_zero_centered=True, offset_point=offset_point, include_rendering_properties=False)

    @classmethod
    def init_from_grid_bin_bounds(cls, grid_bin_bounds: BoundsRect, debug_print=False):
        """ Builds the object and the maze mesh data from the grid_bin_bounds provided.
        
        ## Add the 3D Maze Shape
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions, LinearTrackDimensions3D
            from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena

            a_track_dims = LinearTrackDimensions3D()
            a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(grid_bin_bounds, return_geoemtry=True)
            ipspikesDataExplorer.plots['maze_bg_ideal'] = perform_plot_flat_arena(pActiveSpikesBehaviorPlotter, ideal_maze_pdata, name='idealized_maze_bg', label='idealized_maze', color=[1.0, 0.3, 0.3]) # [0.3, 0.3, 0.3]


        """
        _obj = cls(LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds), grid_bin_bounds=grid_bin_bounds)
        return _obj

    def classify_point(self, test_point) -> "TrackPositionClassification":
        return classify_test_point(test_point, self.rects)
    
    def classify_x_position(self, x) -> "TrackPositionClassification":
        return self.classify_point((x, None))
    
    # TODO: Note that these currently take only x-positions, not real points
    def is_on_maze(self, points):
        return np.array([self.classify_x_position(test_x).is_on_maze for test_x in points])

    def is_on_endcap(self, points):
        return np.array([self.classify_x_position(test_x).is_endcap for test_x in points])


# ==================================================================================================================== #
# Test Plots                                                                                                           #
# ==================================================================================================================== #

def test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims=None, short_track_dims=None):
    """ 
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _test_LinearTrackDimensions_2D

        app, w, cw, (long_track_dims, long_rect_items, long_rects), (short_track_dims, short_rect_items, short_rects) = test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims, short_track_dims)

    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets

    ## Built Long/Short Test Track Dims:
    if long_track_dims is None:
        long_track_dims = LinearTrackDimensions(track_length=170.0)
    if short_track_dims is None:
        short_track_dims = LinearTrackDimensions(track_length=100.0)

    app = pg.mkQApp("Track Graphic via custom LinearTrackDimensions object which builds `QtGui.QGraphicsRectItem` objects")
    w = QtWidgets.QMainWindow()
    cw = pg.GraphicsLayoutWidget()
    w.show()
    w.resize(400,600)
    w.setCentralWidget(cw)
    w.setWindowTitle('LinearTrackDimensions pyqtgraph example: _test_LinearTrackDimensions_2D')

    ax0 = cw.addPlot(row=0, col=0, name="LongTrack")
    ax1 = cw.addPlot(row=1, col=0, name="ShortTrack")

    ax1.setXLink(ax0)
    ax1.setYLink(ax0)
    # p.setRange(QtCore.QRectF(-20, -10, 60, 20))

    long_track_combined_collection, long_rect_items, long_rects = long_track_dims.plot_rects(ax0)
    short_track_combined_collection, short_rect_items, short_rects = short_track_dims.plot_rects(ax1)
    
    return app, w, cw, (ax0, ax1), (long_track_dims, long_rect_items, long_rects), (short_track_dims, short_rect_items, short_rects)


def test_LinearTrackDimensions_2D_Matplotlib(long_track_dims=None, short_track_dims=None, long_offset=None, short_offset=None):
    """ 
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import test_LinearTrackDimensions_2D_Matplotlib
        fig, ax1, ax2 = test_LinearTrackDimensions_2D_Matplotlib()
    """

    # Built Long/Short Test Track Dims:
    if long_track_dims is None:
        long_track_dims = LinearTrackDimensions(track_length=170.0)
    if short_track_dims is None:
        short_track_dims = LinearTrackDimensions(track_length=100.0)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
    
    long_track_dims.plot_rects(ax1, offset=long_offset)
    short_track_dims.plot_rects(ax2, offset=short_offset)
    # [ax.axis('equal') for ax in (ax1, ax2)]
    # Auto scale views
    ax1.autoscale_view()
    ax2.autoscale_view()

    ax1.set_title("LongTrack")
    ax2.set_title("ShortTrack")
    fig.suptitle('LinearTrackDimensions Matplotlib example: test_LinearTrackDimensions_2D_Matplotlib')

    plt.tight_layout()
    plt.show()

    return fig, ax1, ax2


# ==================================================================================================================== #
# General Functions                                                                                                    #
# ==================================================================================================================== #


def add_vertical_track_bounds_lines(grid_bin_bounds, ax=None, include_long:bool=True, include_short:bool=True):
    """ Plots eight vertical lines across ax representing the (start, stop) of each platform (long_left, short_left, short_right, long_right)
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines
        grid_bin_bounds = deepcopy(long_pf2D.config.grid_bin_bounds)
        long_track_line_collection, short_track_line_collection = add_vertical_track_bounds_lines(grid_bin_bounds=grid_bin_bounds, ax=None)

    """
    long_track_dims = LinearTrackDimensions(track_length=170.0)
    short_track_dims = LinearTrackDimensions(track_length=100.0)

    # Find center from `grid_bin_bounds` using `point_tuple_mid_point`
    x_midpoint, y_midpoint = (point_tuple_mid_point(grid_bin_bounds[0]), point_tuple_mid_point(grid_bin_bounds[1])) # grid_bin_bounds_center_point: (145.43, 140.61)

    long_notable_x_positions, _long_notable_y_positions = long_track_dims._build_component_notable_positions(offset_point=(x_midpoint, y_midpoint))
    short_notable_x_positions, _short_notable_y_positions = short_track_dims._build_component_notable_positions(offset_point=(x_midpoint, y_midpoint))

    # Omit the midpoint
    long_notable_x_platform_positions = long_notable_x_positions[[0,1,3,4]]
    short_notable_x_platform_positions = short_notable_x_positions[[0,1,3,4]]

    ## Adds to current axes:
    if ax is None:
        fig = plt.gcf()
        axs = fig.get_axes()
        ax = axs[0]
        
    
    long_short_display_config_manager = LongShortDisplayConfigManager()
    
    
    if include_long:
        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        long_kwargs = deepcopy(long_epoch_matplotlib_config)
        long_track_line_collection: matplotlib.collections.LineCollection = plt.vlines(long_notable_x_platform_positions, label='long_track_x_pos_lines', ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1], colors=long_kwargs.get('edgecolor', '#0000FFAA'), linewidths=long_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection
    else:
        long_track_line_collection = None
        
    if include_short:
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
        short_kwargs = deepcopy(short_epoch_matplotlib_config)
        short_track_line_collection: matplotlib.collections.LineCollection = plt.vlines(short_notable_x_platform_positions, label='short_track_x_pos_lines', ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1], colors=short_kwargs.get('edgecolor', '#FF0000AA'), linewidths=short_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection
    else:
        short_track_line_collection = None

    return long_track_line_collection, short_track_line_collection


def add_track_shapes(grid_bin_bounds, ax=None, include_long:bool=True, include_short:bool=True):
    """ Plots the two track shapes on the plot. Kinda inflexible right now. 
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines
        grid_bin_bounds = deepcopy(long_pf2D.config.grid_bin_bounds)
        long_track_line_collection, short_track_line_collection = add_vertical_track_bounds_lines(grid_bin_bounds=grid_bin_bounds, ax=None)

    """
    if not isinstance(grid_bin_bounds, BoundsRect):
        grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(grid_bin_bounds)
    
    # long_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)
    # short_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)
    long_track_dims = LinearTrackDimensions(track_length=170.0)
    short_track_dims = LinearTrackDimensions(track_length=100.0)

    ## Overrides for 1D
    common_1D_platform_height = 0.25
    common_1D_track_height = 0.1
    long_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    long_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    short_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    short_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    # Centered above and below the y=0.0 line:
    long_offset = (grid_bin_bounds.center_point[0], 0.75)
    short_offset = (grid_bin_bounds.center_point[0], -0.75)
        
    ## Adds to current axes:
    if ax is None:
        fig = plt.gcf()
        axs = fig.get_axes()
        ax = axs[0]
        
    long_short_display_config_manager = LongShortDisplayConfigManager()
    
    if include_long:
        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        # long_kwargs = deepcopy(long_epoch_matplotlib_config)
        # long_kwargs = dict(edgecolor='#0000FFFF', facecolor='#0000FFFF')
        long_kwargs = dict(edgecolor='#000000ff', facecolor='#000000ff')
        long_rects_outputs = long_track_dims.plot_rects(ax, offset=long_offset, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=long_kwargs, **dict(linewidth=2, zorder=-99)))
    else:
        long_rects_outputs = None
        
    if include_short:
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
        # short_kwargs = deepcopy(short_epoch_matplotlib_config)
        # short_kwargs = dict(edgecolor='#FF0000FF', facecolor='#FF0000FF')
        short_kwargs = dict(edgecolor='#000000ff', facecolor='#000000ff')
        short_rects_outputs = short_track_dims.plot_rects(ax, offset=short_offset, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=short_kwargs, **dict(linewidth=2, zorder=-99)))
    else:
        short_rects_outputs = None

    return long_rects_outputs, short_rects_outputs




def _build_track_1D_verticies(platform_length: float = 22.0, track_length: float = 70.0, track_1D_height: float = 1.0, platform_1D_height: float = 1.1, track_center_midpoint_x = 135.0, track_center_midpoint_y = 0.0, debug_print=False) -> Path:
    """ 2023-10-12 - a hyper-simplified 1D plot of the linear track using info from Kamran about the actual midpoint of the track (x=135.0).

    COMPLETELY INDEPENDENT OF ALL OTHER VERSIONS ABOVE.
    Confirmed to be valid for a simple 1D track with a simple x-coord offset
        
    track_center_midpoint_x: float, default: 135.0 # in cm coordinates, according to Kamran on 2023-10-12
    track_center_midpoint_y: float, default: 0.0 # not relevant for 1D track plots


    Usage:
    
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.path import Path
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _build_track_1D_verticies
        
        path = _build_track_1D_verticies(platform_length=22.0, track_length=70.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=135.0, track_center_midpoint_y=0.0, debug_print=True)


        fig, ax = plt.subplots()
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
        ax.autoscale()
        plt.show()

    
    
    
    """
    
    
    # track_width: float = 6.0
    total_track_length: float = platform_length + track_length + platform_length

    # display(total_track_length)

    track_center_relative_point_x = total_track_length / 2.0
    track_center_offset_x = track_center_midpoint_x - track_center_relative_point_x
    # display(track_center_relative_point_x)


    relative_points_array = np.array([[0.0, platform_length], [platform_length, (platform_length+track_length)], [(platform_length+track_length), (platform_length+track_length+platform_length)]]) # still grouped in [[start_x, end_x], ...] pairs
    # relative_points_array.shape # (3, 2)

    bottom_points_x = relative_points_array.flatten() + track_center_offset_x
    bottom_points_y = np.zeros_like(bottom_points_x)
    # For the top points, clone the bottom points to start
    top_points_x = deepcopy(bottom_points_x)
    top_points_y = deepcopy(bottom_points_y)

    # Adjust the important top points
    top_points_y[[True, True, False, False, True, True]] = platform_1D_height
    top_points_y[[False, False, True, True, False, False]] = track_1D_height

    # remove redundant bottom points:
    bottom_points_x = bottom_points_x[[0,-1]]
    bottom_points_y = bottom_points_y[[0,-1]]

    num_verticies = len(bottom_points_x) + len(top_points_x) + 1
    _bottom_point_tuples = [(x, y) for x,y in zip(bottom_points_x, bottom_points_y)]
    _top_point_tuples = [(x, y) for x,y in zip(top_points_x, top_points_y)]
    if debug_print:
        print(_bottom_point_tuples)
        print(_top_point_tuples)

    _all_point_tuples = deepcopy(_top_point_tuples)
    _all_point_tuples.insert(0, _bottom_point_tuples[0])
    _all_point_tuples.append(_bottom_point_tuples[-1])
    _all_point_tuples.append((0.0, 0.0)) # append the extra vertex used to close the polygon


    ## Matplotlib-specific part here:	
    verts = np.array(_all_point_tuples)
    num_verticies = np.shape(verts)[0]
    codes = np.full(num_verticies, Path.LINETO, dtype=int)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    path = Path(verts, codes)

    return path



# ==================================================================================================================== #
# Napari                                                                                                               #
# ==================================================================================================================== #

def convert_QPoint_to_xy_tuple(a_point):
    return np.array([a_point.x(), a_point.y()])

def convert_QGraphicsRectItem_list_to_napari_poly_verticies(a_rect_items):
    """ 
    
    returns: 
        [array([[44.7092, 161.421],
                [44.7092, 124.579],
                [81.5516, 124.579],
                [81.5516, 161.421]]),
        array([[78.4814, 149.294],
                [78.4814, 136.706],
                [342.519, 136.706],
                [342.519, 149.294]]),
        array([[339.448, 161.421],
                [339.448, 124.579],
                [376.291, 124.579],
                [376.291, 161.421]])]

    """
    # long_rect_items # list of QGraphicsRectItem 
    an_extracted_poly_verticies = []

    for a_graphics_rect_item in a_rect_items:
        a_rect = a_graphics_rect_item.mapRectToScene(a_graphics_rect_item.boundingRect()) # QRectF
        # a_rect_coords = a_rect.getRect() # for QRect
        # a_rect_coords = np.array(a_rect.getRect())
        top_left = convert_QPoint_to_xy_tuple(a_rect.topLeft())
        top_right = convert_QPoint_to_xy_tuple(a_rect.topRight())
        bottom_left = convert_QPoint_to_xy_tuple(a_rect.bottomLeft())
        bottom_right = convert_QPoint_to_xy_tuple(a_rect.bottomRight())
        
        a_poly_coords = np.array([bottom_left, top_left, top_right, bottom_right])
        an_extracted_poly_verticies.append(a_poly_coords)
        
    # the order is LeftPlatform, MidTrack, RightPlatform, and we want MidTrack to be in the back, so we move it to the end of the list:
    LeftPlatform, MidTrack, RightPlatform, = an_extracted_poly_verticies
    return [MidTrack, RightPlatform, LeftPlatform]



def add_napari_track_shapes_layer(viewer, long_rect_items, short_rect_items):
    """ 2024-02-05 - Plots the long and short track as Napari shape layers in the `viewer`
    
    """
    # long_rect_items # list of QGraphicsRectItem 
    
    long_extracted_poly_verticies = convert_QGraphicsRectItem_list_to_napari_poly_verticies(long_rect_items)
    short_extracted_poly_verticies = convert_QGraphicsRectItem_list_to_napari_poly_verticies(short_rect_items)

    a_display_config_man = LongShortDisplayConfigManager()
    

    long_rectangles_poly_shapes_layer = viewer.add_shapes(long_extracted_poly_verticies, shape_type='polygon', edge_width=3, edge_color='class', face_color=a_display_config_man.long_epoch_config.mpl_color, text='Long Track', name='LongTrack')
    # change some attributes of the layer
    long_rectangles_poly_shapes_layer.opacity = 1
    long_rectangles_poly_shapes_layer.editable = False
    
    short_rectangles_poly_shapes_layer = viewer.add_shapes(short_extracted_poly_verticies, shape_type='polygon', edge_width=3, edge_color='class', face_color=a_display_config_man.short_epoch_config.mpl_color, text='Short Track', name='ShortTrack')
    # change some attributes of the layer
    short_rectangles_poly_shapes_layer.opacity = 1
    short_rectangles_poly_shapes_layer.editable = False
    
    return long_rectangles_poly_shapes_layer, short_rectangles_poly_shapes_layer


if __name__ == '__main__':

    app, w, cw, (long_track_dims, long_rect_items, long_rects), (short_track_dims, short_rect_items, short_rects) = test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims=None, short_track_dims=None)    
    sys.exit(app.exec_())


