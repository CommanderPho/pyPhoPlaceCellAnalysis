import sys
from copy import deepcopy
from typing import Tuple, Optional, List, Dict, Union
from nptyping import NDArray
from enum import Enum # for TrackPositionClassification
from attrs import define, field, Factory
from collections import namedtuple
import numpy as np
import pandas as pd

from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for safely_accepts_kwargs
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

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Iterable
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin # for NotableTrackPositions
from neuropy.core.session.Formats.SessionSpecifications import SessionConfig


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
# from matplotlib.path import Path
import matplotlib.path as mpl_path


import matplotlib.patches as patches # for matplotlib version of the plot
from matplotlib.collections import PatchCollection

import pyvista as pv # for 3D support in `LinearTrackDimensions3D`



@define(slots=False, repr=True)
class NotableTrackPositions(UnpackableMixin):
    """ 2024-04-10 - Just holds the outer/inner x-positions of the platforms which entirely describes one configuration (long/short) of the track.
    
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import NotableTrackPositions
    
    """
    left_platform_outer: float = field()
    left_platform_inner: float = field()
    right_platform_inner: float = field()
    right_platform_outer: float = field()

    # Computed properties
    @property
    def outer_width(self) -> float:
        """total track (including platform) width"""
        return np.abs(self.right_platform_outer - self.left_platform_outer) # total track (including platform) width
    
    @property
    def inner_width(self) -> float:
        """track (non-platform) width"""
        return np.abs(self.right_platform_inner - self.left_platform_inner) # track (non-platform) width
    
    @classmethod
    def init_x_and_y_notable_positions(cls, long_xlim, long_ylim, short_xlim, short_ylim, platform_side_length: float = 22.0):
        """

        Usage:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import NotableTrackPositions
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import perform_add_vertical_track_bounds_lines

            (long_notable_x_platform_positions, short_notable_x_platform_positions), (long_notable_y_platform_positions, short_notable_y_platform_positions) = NotableTrackPositions.init_notable_track_points_from_session_config(curr_active_pipeline.sess.config)

            LR_long_track_line_collection, LR_short_track_line_collection = perform_add_vertical_track_bounds_lines(long_notable_x_platform_positions=tuple(long_notable_x_platform_positions),
                                                                                                                short_notable_x_platform_positions=tuple(short_notable_x_platform_positions),
                                                                                                                ax=ax_LR)
            RL_long_track_line_collection, RL_short_track_line_collection = perform_add_vertical_track_bounds_lines(long_notable_x_platform_positions=tuple(long_notable_x_platform_positions),
                                                                                                                short_notable_x_platform_positions=tuple(short_notable_x_platform_positions),
                                                                                                                ax=ax_RL)
                                                                                                            
        """
        # XLIM:
        long_notable_x_platform_positions: NotableTrackPositions = cls(left_platform_outer=(long_xlim[0]-platform_side_length), left_platform_inner=long_xlim[0], right_platform_inner=long_xlim[1], right_platform_outer=(long_xlim[1]+platform_side_length))
        short_notable_x_platform_positions: NotableTrackPositions = cls(left_platform_outer=(short_xlim[0]-platform_side_length), left_platform_inner=short_xlim[0], right_platform_inner=short_xlim[1], right_platform_outer=(short_xlim[1]+platform_side_length))

        # YLIM: NOTE: for y-axis the names of the `NotableTrackPositions` class doesn't make a ton of sense
        # long_notable_y_platform_positions: NotableTrackPositions = cls(left_platform_outer=(long_ylim[0]-long_track_dims.platform_side_length), left_platform_inner=long_ylim[0], right_platform_inner=long_ylim[1], right_platform_outer=(long_ylim[1]+long_track_dims.platform_side_length))
        # short_notable_y_platform_positions: NotableTrackPositions = cls(left_platform_outer=(short_ylim[0]-short_track_dims.platform_side_length), left_platform_inner=short_ylim[0], right_platform_inner=short_ylim[1], right_platform_outer=(short_ylim[1]+short_track_dims.platform_side_length))
        long_notable_y_platform_positions: NotableTrackPositions = cls(left_platform_outer=long_ylim[0], left_platform_inner=long_ylim[0], right_platform_inner=long_ylim[1], right_platform_outer=long_ylim[1]) # NOTE: no track width
        short_notable_y_platform_positions: NotableTrackPositions = cls(left_platform_outer=short_ylim[0], left_platform_inner=short_ylim[0], right_platform_inner=short_ylim[1], right_platform_outer=short_ylim[1])  # NOTE: no track width

        return (long_notable_x_platform_positions, short_notable_x_platform_positions), (long_notable_y_platform_positions, short_notable_y_platform_positions) 

    @classmethod
    def init_notable_track_points_from_session_config(cls, a_sess_config: Union[SessionConfig, Dict], platform_side_length:float=22.0) -> Tuple[Tuple["NotableTrackPositions", "NotableTrackPositions"], Tuple["NotableTrackPositions", "NotableTrackPositions"]]:
        """ Builds the two tracks (long/short) objects from the session config provided.
        
        ## Usage:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(curr_active_pipeline.sess.config)

        """
        loaded_track_limits = deepcopy(a_sess_config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
        long_xlim = loaded_track_limits['long_xlim']
        long_ylim = loaded_track_limits['long_ylim']
        ## if we have short, build that one too:
        short_xlim = loaded_track_limits['short_xlim']
        short_ylim = loaded_track_limits['short_ylim']
        return cls.init_x_and_y_notable_positions(long_xlim=long_xlim, long_ylim=long_ylim, short_xlim=short_xlim, short_ylim=short_ylim, platform_side_length=platform_side_length)
    




class TrackPositionClassification(Enum):
    """ classifying various x-positions as belonging to outside the outside_maze, the track_endcaps, or the track_straightaway

        # TrackPositionClassification.TRACK_ENDCAPS
        # TrackPositionClassification.TRACK_BODY
        # TrackPositionClassification.OUTSIDE_MAZE
    """
    OUTSIDE_MAZE = "outside_maze"
    TRACK_ENDCAPS = "track_endcaps"
    TRACK_STRAIGHTAWAY = "track_straightaway"

    @property
    def is_on_maze(self) -> bool:
        """ returns True if the point is anywhere on the track (including endcaps) """
        return self.value != TrackPositionClassification.OUTSIDE_MAZE.value
    

    @property
    def is_track_straightaway(self) -> bool:
        """ returns True if the point is anywhere on the track (including endcaps) """
        return self.value == TrackPositionClassification.TRACK_STRAIGHTAWAY.value

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
    assert len(rects) == 3, f"rects should contain three elements for (left_platform, track_straightaway, right_platform). {rects}"
    if is_point_in_rect(test_point, rects[0]) or is_point_in_rect(test_point, rects[2]):
        return TrackPositionClassification.TRACK_ENDCAPS
    elif is_point_in_rect(test_point, rects[1]):
        return TrackPositionClassification.TRACK_STRAIGHTAWAY
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
    track_width: float = field(default=6.2)
    track_length: float = field(default=100.0)
    platform_side_length: float = field(default=22.0)
    minor_axis_platform_side_width: Optional[float] = field(default=None)

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

    def _build_component_rectangles(self, is_zero_centered:bool=False, offset_point=None, include_rendering_properties:bool=True, rotate_to_vertical:bool=False):
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
        if rotate_to_vertical:
            # swap (x,y) and (w, h)
            rects = [(y, x, h, w, pen, brush) for x, y, w, h, pen, brush in rects]

        return rects
        
    def plot_rects(self, plot_item, offset=None, matplotlib_rect_kwargs_override=None, rotate_to_vertical:bool=False):
        """ main function to plot 

        
        combined_item, rect_items, rects = item.plot_rect(ax, offset=None)
        """
        rects = self._build_component_rectangles(is_zero_centered=True, offset_point=offset, include_rendering_properties=True, rotate_to_vertical=rotate_to_vertical)

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
    track_dimensions: Union[LinearTrackDimensions, LinearTrackDimensions3D] = field()
    grid_bin_bounds: BoundsRect = field() #= None #TODO 2023-09-20 12:33: - [ ] Allow storing grid_bin_bounds to help with offset computations
    
    @property
    def rects(self):
        offset_point = self.grid_bin_bounds.center_point # (self.grid_bin_bounds.center_point[0], 0.75)
        return self.track_dimensions._build_component_rectangles(is_zero_centered=True, offset_point=offset_point, include_rendering_properties=False)

    @classmethod
    def init_from_grid_bin_bounds(cls, grid_bin_bounds: Union[BoundsRect, Tuple, List, NDArray], debug_print=False):
        """ Builds the object and the maze mesh data from the grid_bin_bounds provided.
        
        ## Add the 3D Maze Shape
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions, LinearTrackDimensions3D
            from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena

            a_track_dims = LinearTrackDimensions3D()
            a_track_dims, ideal_maze_pdata = LinearTrackDimensions3D.init_from_grid_bin_bounds(grid_bin_bounds, return_geoemtry=True)
            ipspikesDataExplorer.plots['maze_bg_ideal'] = perform_plot_flat_arena(pActiveSpikesBehaviorPlotter, ideal_maze_pdata, name='idealized_maze_bg', label='idealized_maze', color=[1.0, 0.3, 0.3]) # [0.3, 0.3, 0.3]


        """
        if isinstance(grid_bin_bounds, (Tuple, List, NDArray)):
            grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(grid_bin_bounds)

        _obj = cls(LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds), grid_bin_bounds=grid_bin_bounds)
        return _obj
    


    @classmethod
    def init_tracks_from_session_config(cls, a_sess_config: Union[SessionConfig, Dict], platform_side_length:float=22.0) -> Tuple["LinearTrackInstance", "LinearTrackInstance"]:
        """ Builds the two tracks (long/short) objects from the session config provided.
        
        ## Usage:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(curr_active_pipeline.sess.config)

        """
        loaded_track_limits = deepcopy(a_sess_config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
        x_midpoint: float = a_sess_config.x_midpoint
        pix2cm: float = a_sess_config.pix2cm

        long_xlim = loaded_track_limits['long_xlim']
        long_ylim = loaded_track_limits['long_ylim']

        ## if we have short, build that one too:
        short_xlim = loaded_track_limits['short_xlim']
        short_ylim = loaded_track_limits['short_ylim']
        
        LONG_from_mat_lims_grid_bin_bounds = BoundsRect(xmin=(long_xlim[0]-platform_side_length), xmax=(long_xlim[1]+platform_side_length), ymin=long_ylim[0], ymax=long_ylim[1])
        SHORT_from_mat_lims_grid_bin_bounds = BoundsRect(xmin=(short_xlim[0]-platform_side_length), xmax=(short_xlim[1]+platform_side_length), ymin=short_ylim[0], ymax=short_ylim[1])

        LONG_obj = cls(LinearTrackDimensions.init_from_grid_bin_bounds(LONG_from_mat_lims_grid_bin_bounds), grid_bin_bounds=LONG_from_mat_lims_grid_bin_bounds)
        SHORT_obj = cls(LinearTrackDimensions.init_from_grid_bin_bounds(SHORT_from_mat_lims_grid_bin_bounds), grid_bin_bounds=SHORT_from_mat_lims_grid_bin_bounds)

        return LONG_obj, SHORT_obj
    


    def classify_point(self, test_point) -> "TrackPositionClassification":
        return classify_test_point(test_point, self.rects)
    
    def classify_x_position(self, x) -> "TrackPositionClassification":
        return self.classify_point((x, None))
    
    # TODO: Note that these currently take only x-positions, not real points
    def is_on_maze(self, points):
        return np.array([self.classify_x_position(test_x).is_on_maze for test_x in points])

    def is_on_endcap(self, points):
        return np.array([self.classify_x_position(test_x).is_endcap for test_x in points])

    def build_x_position_classification_df(self, x_arr: NDArray) -> "TrackPositionClassification":
        """ Builds a df with a row for every position passed in x_arr that classifies it in relation to the track
        
        Usage:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(curr_active_pipeline.sess.config)
            long_track_inst
            # track_templates.get_track_length_dict()

            pos_bin_edges = deepcopy(track_templates.get_decoders_dict()['long_LR'].xbin_centers)
            pos_bin_classification_df: pd.DataFrame = long_track_inst.build_x_position_classification_df(x_arr=pos_bin_edges)
            pos_bin_classification_df
        """
        is_pos_bin_endcap = [self.classify_x_position(x).is_endcap for x in x_arr]
        is_pos_bin_on_maze = [self.classify_x_position(x).is_on_maze for x in x_arr]

        return pd.DataFrame({'x': deepcopy(x_arr), 'is_endcap': is_pos_bin_endcap, 'is_on_maze': is_pos_bin_on_maze})


    def plot_rects(self, plot_item, matplotlib_rect_kwargs_override=None, rotate_to_vertical:bool=False):
        """ main function to plot 

        
        combined_item, rect_items, rects = item.plot_rect(ax, offset=None)
        """
        offset_point = self.grid_bin_bounds.center_point # (self.grid_bin_bounds.center_point[0], 0.75)
        return self.track_dimensions.plot_rects(plot_item=plot_item, offset=offset_point, matplotlib_rect_kwargs_override=matplotlib_rect_kwargs_override, rotate_to_vertical=rotate_to_vertical)


    def build_rects(self, include_rendering_properties:bool=False, rotate_to_vertical:bool=False):
        offset_point = self.grid_bin_bounds.center_point # (self.grid_bin_bounds.center_point[0], 0.75)
        return self.track_dimensions._build_component_rectangles(is_zero_centered=True, offset_point=offset_point, include_rendering_properties=include_rendering_properties, rotate_to_vertical=rotate_to_vertical)
        


def get_track_length_dict(long_grid_bin_bounds, short_grid_bin_bounds) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Gets the actual track lengths from the grid_bin_bounds
    
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import get_track_length_dict

    long_grid_bin_bounds = ((22.397021260868584, 245.3970212608686), (133.66465594522782, 155.97244934208123)),
    short_grid_bin_bounds = ((22.397021260868584, 245.3970212608686), (133.66465594522782, 155.97244934208123))

    actual_track_length_dict, idealized_track_length_dict = get_track_length_dict(long_pf2D.config.grid_bin_bounds, short_pf2D.config.grid_bin_bounds)
    actual_track_length_dict
    idealized_track_length_dict

    >>
        {'long': 223.0, 'short': 223.0}
        {'long': 214.0, 'short': 144.0}

    """
    ## Theoretical ideals:
    long_ideal_track_dims = LinearTrackDimensions(track_length=170.0)
    short_ideal_track_dims = LinearTrackDimensions(track_length=100.0)

    long_linear_track, short_linear_track = [LinearTrackInstance.init_from_grid_bin_bounds(grid_bin_bounds=deepcopy(a_grid_bin_bounds), debug_print=True) for a_grid_bin_bounds in (long_grid_bin_bounds, short_grid_bin_bounds)]
    # {'long': long_linear_track, 'short': short_linear_track}
    long_linear_track.track_dimensions.total_length
    short_linear_track.track_dimensions.total_length

    long_ideal_track_dims.total_length
    short_ideal_track_dims.total_length

    return {'long': long_linear_track.track_dimensions.total_length, 'short': short_linear_track.track_dimensions.total_length}, {'long': long_ideal_track_dims.total_length, 'short': short_ideal_track_dims.total_length}



# ==================================================================================================================== #
# Test Plots                                                                                                           #
# ==================================================================================================================== #

def test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims=None, short_track_dims=None):
    """ 
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import test_LinearTrackDimensions_2D_pyqtgraph
        
        app, w, cw, (ax0, ax1), (long_track_dims, long_rect_items, long_rects), (short_track_dims, short_rect_items, short_rects) = test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims, short_track_dims)

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


def test_LinearTrackDimensions_2D_Matplotlib(long_track_dims=None, short_track_dims=None, long_offset=None, short_offset=None, rotate_to_vertical:bool=False):
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

    if rotate_to_vertical:
        figsize=(8,4)
    else:
        figsize=(4,8)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    
    long_track_dims.plot_rects(ax1, offset=long_offset, rotate_to_vertical=rotate_to_vertical)
    short_track_dims.plot_rects(ax2, offset=short_offset, rotate_to_vertical=rotate_to_vertical)
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

@function_attributes(short_name=None, tags=['matplotlib', 'grid_bin_bounds'], input_requires=[], output_provides=[], uses=['perform_add_1D_track_bounds_lines'], used_by=['perform_add_1D_track_bounds_lines'], creation_date='2025-01-20 08:06', related_items=[])
def perform_add_vertical_track_bounds_lines(long_notable_x_platform_positions=None, short_notable_x_platform_positions=None, ax=None, include_long:bool=True, include_short:bool=True):
    """ Plots eight vertical lines across ax representing the (start, stop) of each platform (long_left, short_left, short_right, long_right)
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import perform_add_vertical_track_bounds_lines
        grid_bin_bounds = deepcopy(long_pf2D.config.grid_bin_bounds)
        long_track_line_collection, short_track_line_collection = perform_add_vertical_track_bounds_lines(grid_bin_bounds=grid_bin_bounds, ax=None)

    """
    return perform_add_1D_track_bounds_lines(long_notable_x_platform_positions=long_notable_x_platform_positions, short_notable_x_platform_positions=short_notable_x_platform_positions, ax=ax, include_long=include_long, include_short=include_short, is_vertical=True)


@function_attributes(short_name=None, tags=['grid_bin_bounds', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=['perform_add_vertical_track_bounds_lines'], creation_date='2025-01-20 07:50', related_items=[])
def perform_add_1D_track_bounds_lines(long_notable_x_platform_positions=None, short_notable_x_platform_positions=None, ax=None, include_long:bool=True, include_short:bool=True, is_vertical:bool=True):
    """ Plots eight vertical lines across ax representing the (start, stop) of each platform (long_left, short_left, short_right, long_right)
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import perform_add_1D_track_bounds_lines
        grid_bin_bounds = deepcopy(long_pf2D.config.grid_bin_bounds)
        long_track_line_collection, short_track_line_collection = perform_add_1D_track_bounds_lines(grid_bin_bounds=grid_bin_bounds, ax=None, is_vertical=True)

    """
    ## Adds to current axes:
    if ax is None:
        fig = plt.gcf()
        axs = fig.get_axes()
        ax = axs[0]
        
    long_short_display_config_manager = LongShortDisplayConfigManager()
    
    
    if is_vertical:
       common_ax_bound_kwargs = dict(ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1]) 
    else:
        ## horizontal lines:
        common_ax_bound_kwargs = dict(xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1])

    if (include_long and (long_notable_x_platform_positions is not None)):
        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        long_kwargs = deepcopy(long_epoch_matplotlib_config)
        if is_vertical:
            # long_track_line_collection: matplotlib.collections.LineCollection = plt.vlines(long_notable_x_platform_positions, label='long_track_x_pos_lines', ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1], colors=long_kwargs.get('edgecolor', '#0000FFAA'), linewidths=long_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection
            long_track_line_collection: matplotlib.collections.LineCollection = ax.vlines(long_notable_x_platform_positions, label='long_track_x_pos_lines', **common_ax_bound_kwargs, colors=long_kwargs.get('edgecolor', '#0000FFAA'), linewidths=long_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection
        else:
            ## horizontal lines:
            long_track_line_collection: matplotlib.collections.LineCollection = ax.hlines(long_notable_x_platform_positions, label='long_track_x_pos_lines', **common_ax_bound_kwargs, colors=long_kwargs.get('edgecolor', '#0000FFAA'), linewidths=long_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection

    else:
        long_track_line_collection = None
        
    if (include_short and (short_notable_x_platform_positions is not None)):
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
        short_kwargs = deepcopy(short_epoch_matplotlib_config)
        if is_vertical:
            short_track_line_collection: matplotlib.collections.LineCollection = ax.vlines(short_notable_x_platform_positions, label='short_track_x_pos_lines', **common_ax_bound_kwargs, colors=short_kwargs.get('edgecolor', '#FF0000AA'), linewidths=short_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection
        else:
            # Horizontal
            short_track_line_collection: matplotlib.collections.LineCollection = ax.hlines(short_notable_x_platform_positions, label='short_track_x_pos_lines', **common_ax_bound_kwargs, colors=short_kwargs.get('edgecolor', '#FF0000AA'), linewidths=short_kwargs.get('linewidth', 1.0), linestyles='dashed', zorder=-98) # matplotlib.collections.
            
    else:
        short_track_line_collection = None

    return long_track_line_collection, short_track_line_collection



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
        
    long_track_line_collection, short_track_line_collection = perform_add_vertical_track_bounds_lines(long_notable_x_platform_positions=long_notable_x_platform_positions,
                                                                                                      short_notable_x_platform_positions=short_notable_x_platform_positions,
                                                                                                      ax=ax,
                                                                                                      include_long=include_long, include_short=include_short)

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




def _build_track_1D_verticies(platform_length: float = 22.0, track_length: float = 70.0, track_1D_height: float = 1.0, platform_1D_height: float = 1.1, track_center_midpoint_x = 135.0, track_center_midpoint_y = 0.0, debug_print=False) -> mpl_path.Path:
    """ 2023-10-12 - a hyper-simplified 1D plot of the linear track using info from Kamran about the actual midpoint of the track (x=135.0).

    COMPLETELY INDEPENDENT OF ALL OTHER VERSIONS ABOVE.
    Confirmed to be valid for a simple 1D track with a simple x-coord offset
        
    track_center_midpoint_x: float, default: 135.0 # in cm coordinates, according to Kamran on 2023-10-12
    track_center_midpoint_y: float, default: 0.0 # not relevant for 1D track plots


    Usage:
    
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        # from matplotlib.path import Path
        import matplotlib.path as mpl_path

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
    codes = np.full(num_verticies, mpl_path.Path.LINETO, dtype=int)
    codes[0] = mpl_path.Path.MOVETO
    codes[-1] = mpl_path.Path.CLOSEPOLY
    path = mpl_path.Path(verts, codes)

    return path



@function_attributes(short_name=None, tags=['matplotlib', 'track_plotting', '2D', 'ax'], input_requires=[], output_provides=[], uses=[], used_by=['DecodedTrajectoryMatplotlibPlotter'], creation_date='2024-04-16 16:51', related_items=[])
def _perform_plot_matplotlib_2D_tracks(long_track_inst: LinearTrackInstance, short_track_inst: LinearTrackInstance, ax=None, perform_autoscale: bool = True, rotate_to_vertical:bool=False):
    """ Plots both the long and the short track on a single matplotlib axes.
    
    Usage:
        from pyphocorehelpers.geometry_helpers import BoundsRect
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _perform_plot_matplotlib_2D_tracks, LinearTrackInstance

        # active_config = curr_active_pipeline.sess.config
        active_config = global_session.config

        fig = plt.figure('test track vertical', clear=True)
        an_ax = plt.gca()

        rotate_to_vertical: bool = True
        long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(active_config)
        long_out = _perform_plot_matplotlib_2D_tracks(long_track_inst=long_track_inst, short_track_inst=short_track_inst, ax=an_ax, rotate_to_vertical=rotate_to_vertical)
        if not rotate_to_vertical:
            an_ax.set_xlim(long_track_inst.grid_bin_bounds.xmin, long_track_inst.grid_bin_bounds.xmax)
            an_ax.set_ylim(long_track_inst.grid_bin_bounds.ymin, long_track_inst.grid_bin_bounds.ymax)
            fig.set_size_inches(30.5161, 1.13654)
        else:
            an_ax.set_ylim(long_track_inst.grid_bin_bounds.xmin, long_track_inst.grid_bin_bounds.xmax)
            an_ax.set_xlim(long_track_inst.grid_bin_bounds.ymin, long_track_inst.grid_bin_bounds.ymax)
            fig.set_size_inches(1.13654, 30.5161)

        ax.set_aspect('auto')  # Adjust automatically based on data limits
        ax.set_adjustable('datalim')  # Ensure the aspect ratio respects the data limits
        ax.autoscale()  # Autoscale the view to fit data

    
    """
    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
    long_kwargs = deepcopy(long_epoch_matplotlib_config)
    long_kwargs = overriding_dict_with(lhs_dict=long_kwargs, **dict(linewidth=2, zorder=-99, alpha=0.5, facecolor='#0099ff07', edgecolor=long_kwargs['facecolor'], linestyle='dashed'))
    short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
    short_kwargs = deepcopy(short_epoch_matplotlib_config)
    short_kwargs = overriding_dict_with(lhs_dict=short_kwargs, **dict(linewidth=2, zorder=-98, alpha=0.5, facecolor='#f5161607', edgecolor=short_kwargs['facecolor'], linestyle='dashed'))
        
    # BEGIN PLOTTING _____________________________________________________________________________________________________ #
    long_out_tuple = long_track_inst.plot_rects(plot_item=ax, matplotlib_rect_kwargs_override=long_kwargs, rotate_to_vertical=rotate_to_vertical)
    short_out_tuple = short_track_inst.plot_rects(plot_item=ax, matplotlib_rect_kwargs_override=short_kwargs, rotate_to_vertical=rotate_to_vertical)

    # long_path = _build_track_1D_verticies(platform_length=22.0, track_length=170.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=long_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=-1.0, debug_print=True)
    # short_path = _build_track_1D_verticies(platform_length=22.0, track_length=100.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=short_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=1.0, debug_print=True)

    # ## Plot the tracks:
    # long_patch = patches.PathPatch(long_path, **long_track_color, alpha=0.5, lw=2)
    # ax.add_patch(long_patch)

    # short_patch = patches.PathPatch(short_path, **short_track_color, alpha=0.5, lw=2)
    # ax.add_patch(short_patch)
    if perform_autoscale:
        ax.autoscale()
    
    return long_out_tuple, short_out_tuple





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


@function_attributes(short_name=None, tags=['napari', 'plot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-05 00:00', related_items=[])
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


from enum import Enum

class AclusYOffsetMode(Enum):
    RandomJitter = "random_jitter"
    CountBased = "count_based"


@function_attributes(short_name=None, tags=['matplotlib', 'patches', 'track', 'long-short'], input_requires=[], output_provides=[], uses=['pyphoplacecellanalysis.Pho2D.track_shape_drawing._build_track_1D_verticies'], used_by=['_plot_track_remapping_diagram'], creation_date='2024-06-12 12:57', related_items=[])
def _plot_helper_add_track_shapes(grid_bin_bounds: Union[Tuple[Tuple[float, float], Tuple[float, float]], BoundsRect], is_dark_mode: bool = True, debug_print=False):
    """ Prepares the final matplotlib patch objects to represent the 1D long and short tracks, which can be immediately added to an axis
    
    Usage:

    ## TRACK PLOTTING:
    (long_patch, long_path), (short_patch, short_path) = _plot_helper_add_track_shapes(grid_bin_bounds=grid_bin_bounds, ax=ax, defer_render=defer_render, is_dark_mode=is_dark_mode, debug_print=debug_print)
    # Draw the long/short track shapes: __________________________________________________________________________________ #
    if ax is not None:
        ax.add_patch(long_patch)
        ax.add_patch(short_patch)
        ax.autoscale()

    History:

        Factored out of `_plot_track_remapping_diagram` on 2024-06-12

    """
  # BUILDS TRACK PROPERTIES ____________________________________________________________________________________________ #
    import matplotlib as mpl
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.transforms import Affine2D
    import matplotlib.patheffects as path_effects

    from neuropy.utils.matplotlib_helpers import build_or_reuse_figure

    from pyphocorehelpers.geometry_helpers import BoundsRect
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

    
    if is_dark_mode:
        _default_bg_color = 'white'
        _default_fg_color = 'black'
        _default_edgecolors = '#CCCCCC33' # light gray

    else:
        _default_bg_color = 'black'
        _default_fg_color = 'white'
        _default_edgecolors = '#5a5a5a33'
        
    base_1D_height: float = 1.0
    # base_1D_height: float = 0.5
    base_platform_additive_height: float = 0.1

    long_height_multiplier: float = 1.0
    # long_height_multiplier: float = 0.5 # this renders the long track half-height

    # long_y_baseline: float = 0.1
    # short_y_baseline: float = 0.75

    ## smarter, all are calculated in terms of 1.0 being the height of the total subplot axes:
    top_bottom_padding: float = 0.025
    intra_track_y_spacing: float = 0.05 # spacing in between the long/short tracks

    total_track_y_space: float = 1.0 - (intra_track_y_spacing + (2.0 * top_bottom_padding)) # amount of total space for the tracks
    track_y_height: float = total_track_y_space / 2.0

    long_y_baseline: float = top_bottom_padding
    short_y_baseline: float = long_y_baseline + track_y_height + intra_track_y_spacing

    # long_y_height: float = (short_y_baseline - intra_track_y_spacing)
    # short_y_top: float = (1.0-0.1) # 0.9
    
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    ## Get the track configs for the colors:
    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
    short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

    long_track_color = dict(facecolor=long_epoch_config['facecolor'])
    short_track_color = dict(facecolor=short_epoch_config['facecolor'])

    if isinstance(grid_bin_bounds, (tuple, list)):
        grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(grid_bin_bounds)
    else:
        assert isinstance(grid_bin_bounds, BoundsRect)


    # display(grid_bin_bounds)

    # long_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)
    # short_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)

    long_track_dims = LinearTrackDimensions(track_length=170.0)
    short_track_dims = LinearTrackDimensions(track_length=100.0)

    common_1D_platform_height = 0.25
    common_1D_track_height = 0.1
    long_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    long_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    short_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    short_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    # instances:
    long_track = LinearTrackInstance(long_track_dims, grid_bin_bounds=grid_bin_bounds)
    short_track = LinearTrackInstance(short_track_dims, grid_bin_bounds=grid_bin_bounds)

    # BEGIN PLOTTING _____________________________________________________________________________________________________ #

    track_1D_height=1.0*base_1D_height
    platform_1D_height=1.0*base_1D_height + base_platform_additive_height # want same (additive) height offset even when scaling.

    long_path = _build_track_1D_verticies(platform_length=22.0, track_length=long_track_dims.track_length, track_1D_height=(track_1D_height * long_height_multiplier), platform_1D_height=((track_1D_height * long_height_multiplier) + base_platform_additive_height), track_center_midpoint_x=long_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=-1.0, debug_print=debug_print)
    # long_path = _build_track_1D_verticies(platform_length=22.0, track_length=long_track_dims.track_length, track_1D_height=(track_1D_height * long_height_multiplier), platform_1D_height=(platform_1D_height * long_height_multiplier), track_center_midpoint_x=long_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=-1.0, debug_print=True)
    short_path = _build_track_1D_verticies(platform_length=22.0, track_length=short_track_dims.track_length, track_1D_height=track_1D_height, platform_1D_height=platform_1D_height, track_center_midpoint_x=short_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=1.0, debug_print=debug_print)
    
    # Define the transformation: squish along y-axis by 0.5 and translate up by 0.5 units
    # long_transformation = Affine2D().scale(1, -0.5).translate(0, long_y_baseline)
    # short_transformation = Affine2D().scale(1, 0.5).translate(0, short_y_baseline)
    track_to_baseline_padding: float = 0.05
    # long_transformation = Affine2D().scale(1, 1.0).translate(0, long_y_baseline-track_to_baseline_padding)
    # short_transformation = Affine2D().scale(1, 1.0).translate(0, short_y_baseline-track_to_baseline_padding)

    long_transformation = Affine2D().scale(1, track_y_height).translate(0, (long_y_baseline-track_to_baseline_padding))
    short_transformation = Affine2D().scale(1, track_y_height).translate(0, (short_y_baseline-track_to_baseline_padding))

    # Apply the transformation to the Path
    long_path = long_path.transformed(long_transformation)
    short_path = short_path.transformed(short_transformation)

    # Draw the long/short track shapes: __________________________________________________________________________________ #
    long_patch = patches.PathPatch(long_path, **long_track_color, alpha=0.5, lw=2)
    short_patch = patches.PathPatch(short_path, **short_track_color, alpha=0.5, lw=2)
    
    return (long_patch, long_path), (short_patch, short_path)

@function_attributes(short_name=None, tags=['matplotlib', 'track', 'remapping', 'good', 'working'], input_requires=[], output_provides=[], uses=['_plot_helper_add_track_shapes'], used_by=['plot_bidirectional_track_remapping_diagram'], creation_date='2024-02-22 11:12', related_items=[])
def _plot_track_remapping_diagram(a_dir_decoder_aclu_MAX_peak_maps_df: pd.DataFrame, grid_bin_bounds: Union[Tuple[Tuple[float, float], Tuple[float, float]], BoundsRect], long_column_name:str='long_LR', short_column_name:str='short_LR', ax=None, defer_render: bool=False, enable_interactivity:bool=True, draw_point_aclu_labels:bool=False, enable_adjust_overlapping_text: bool=False, is_dark_mode: bool = True, aclus_y_offset_mode:AclusYOffsetMode=AclusYOffsetMode.CountBased, debug_print=False, **kwargs):
    """ Plots a single figure containing the long and short track outlines (flattened, overlayed) with single points on each corresponding to the peak location in 1D

    🔝🖼️🎨
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _plot_track_remapping_diagram
    # grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(global_pf2D.config.grid_bin_bounds)
    fix, ax, _outputs_tuple = _plot_track_remapping_diagram(LR_only_decoder_aclu_MAX_peak_maps_df, long_peak_x, short_peak_x, peak_x_diff, grid_bin_bounds=long_pf2D.config.grid_bin_bounds)

    
    Usage:

        from matplotlib.gridspec import GridSpec
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, perform_update_title_subtitle
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _plot_track_remapping_diagram
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _get_directional_pf_peaks_dfs

        LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df = _get_directional_pf_peaks_dfs(track_templates, drop_aclu_if_missing_long_or_short=True)

        ## Make a single figure for both LR/RL remapping cells:
        kwargs = {}
        fig = build_or_reuse_figure(fignum='Track Remapping', fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True, clear=True) # , clear=True
        gs = GridSpec(2, 1, figure=fig)
        ax_LR = plt.subplot(gs[0])
        ax_RL = plt.subplot(gs[1])

        fig_LR_RL, ax_LR, _outputs_tuple_LR = _plot_track_remapping_diagram(LR_only_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds=long_pf2D.config.grid_bin_bounds, long_column_name='long_LR', short_column_name='short_LR', ax=ax_LR)
        perform_update_title_subtitle(fig=fig_LR_RL, ax=ax_LR, title_string=None, subtitle_string=f"LR Track Remapping - {len(LR_only_decoder_aclu_MAX_peak_maps_df)} aclus")
        fig_LR_RL, ax_RL, _outputs_tuple_RL = _plot_track_remapping_diagram(RL_only_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds=long_pf2D.config.grid_bin_bounds, long_column_name='long_RL', short_column_name='short_RL', ax=ax_RL)
        perform_update_title_subtitle(fig=fig_LR_RL, ax=ax_RL, title_string=None, subtitle_string=f"RL Track Remapping - {len(RL_only_decoder_aclu_MAX_peak_maps_df)} aclus")

    """
    # BUILDS TRACK PROPERTIES ____________________________________________________________________________________________ #
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.patheffects as path_effects

    from neuropy.utils.matplotlib_helpers import build_or_reuse_figure

    if is_dark_mode:
        _default_bg_color = 'white'
        _default_fg_color = 'black'
        _default_edgecolors = '#CCCCCC33' # light gray

    else:
        _default_bg_color = 'black'
        _default_fg_color = 'white'
        _default_edgecolors = '#5a5a5a33'

    # aclus_y_offset_mode: AclusYOffsetMode = AclusYOffsetMode.CountBased
    # aclus_y_offset_mode: AclusYOffsetMode = AclusYOffsetMode.RandomJitter
    # aclus_y_offset_mode = 'random_jitter'
    # aclus_y_offset_mode = 'count_based'

    aclus_y_offset_mode_POSSIBLE_OPTIONS = ['random_jitter', 'count_based']
    assert aclus_y_offset_mode.value in aclus_y_offset_mode_POSSIBLE_OPTIONS, f"aclus_y_offset_mode must be in {aclus_y_offset_mode_POSSIBLE_OPTIONS} but aclus_y_offset_mode: {aclus_y_offset_mode}"
    unit_id_colors_map = kwargs.pop('unit_id_colors_map', None)

    if aclus_y_offset_mode.value == AclusYOffsetMode.CountBased.value:
        enable_single_y_point_arrows: bool = True # if True all arrows are started from the top aclu for long and bottom/baseline for short. If False each arrow starts from its correct aclu dot, which can appear more busy.
    else:
        enable_single_y_point_arrows: bool = False
        
    base_1D_height: float = 1.0
    # base_1D_height: float = 0.5
    base_platform_additive_height: float = 0.1

    long_height_multiplier: float = 1.0
    # long_height_multiplier: float = 0.5 # this renders the long track half-height

    # long_y_baseline: float = 0.1
    # short_y_baseline: float = 0.75

    ## smarter, all are calculated in terms of 1.0 being the height of the total subplot axes:
    top_bottom_padding: float = 0.025
    intra_track_y_spacing: float = 0.05 # spacing in between the long/short tracks

    total_track_y_space: float = 1.0 - (intra_track_y_spacing + (2.0 * top_bottom_padding)) # amount of total space for the tracks
    track_y_height: float = total_track_y_space / 2.0

    long_y_baseline: float = top_bottom_padding
    short_y_baseline: float = long_y_baseline + track_y_height + intra_track_y_spacing

    # long_y_height: float = (short_y_baseline - intra_track_y_spacing)
    # short_y_top: float = (1.0-0.1) # 0.9
    
    scatter_point_size: float = 15.0

    # Text label options:
    if is_dark_mode:
        aclu_labels_text_color='black'
    else:
        aclu_labels_text_color='white'

    # aclu_labels_fontsize = 6
    aclu_labels_fontsize = 3
    aclu_labels_text_path_effects = [path_effects.Stroke(linewidth=0.1, foreground='darkgrey'), path_effects.Normal()]

    ## Selection (only for interactivity)
    selection_color = (1, 0, 0, 1)  # Red color in RGBA format
    scatter_point_selected_size: float = scatter_point_size + 2.0
    scatter_edgecolors_selection_color = (0.35, 0, 0, 1)  # Red color in RGBA format

    selection_text_path_effects = [path_effects.Stroke(linewidth=0.2, foreground='red'), path_effects.Normal()]

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

    ## Extract the quantities needed from the DF passed
    active_aclus = a_dir_decoder_aclu_MAX_peak_maps_df.index.to_numpy()
    long_peak_x = a_dir_decoder_aclu_MAX_peak_maps_df[long_column_name].to_numpy()
    short_peak_x = a_dir_decoder_aclu_MAX_peak_maps_df[short_column_name].to_numpy()
    # peak_x_diff = LR_only_decoder_aclu_MAX_peak_maps_df['peak_diff'].to_numpy()

    assert (len(long_peak_x) == len(short_peak_x)), f"len(long_peak_x): {len(long_peak_x)} != len(short_peak_x): {len(short_peak_x)}"
    assert (len(long_peak_x) == len(active_aclus)), f"len(long_peak_x): {len(long_peak_x)} != len(active_aclus): {len(active_aclus)}"
    
    ## Find the points missing from long or short:
    disappearing_long_to_short_indicies = np.where(np.isnan(short_peak_x))[0] # missing peak from short
    appearing_long_to_short_indicies = np.where(np.isnan(long_peak_x))[0] # missing peak from long
    disappearing_long_to_short_aclus = active_aclus[disappearing_long_to_short_indicies] # missing peak from short
    appearing_long_to_short_aclus = active_aclus[appearing_long_to_short_indicies] # missing peak from long

    is_aclu_in_both = np.logical_and(np.logical_not(np.isnan(short_peak_x)), np.logical_not(np.isnan(long_peak_x))) # both are non-NaN
    assert (len(is_aclu_in_both) == len(active_aclus)), f"len(is_aclu_in_both): {len(is_aclu_in_both)} != len(active_aclus): {len(active_aclus)}"
    both_aclus = active_aclus[is_aclu_in_both]

    if len(disappearing_long_to_short_aclus) > 0:
        print(f'disappearing_long_to_short_aclus: {disappearing_long_to_short_aclus}')
    if len(appearing_long_to_short_aclus) > 0:
        print(f'appearing_long_to_short_aclus: {appearing_long_to_short_aclus}')


    ## Create the remapping figure:
    ## Figure Setup:
    if ax is None:
        ## Build a new figure:
        from matplotlib.gridspec import GridSpec
        fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
        gs = GridSpec(1, 1, figure=fig)
        ax = plt.subplot(gs[0])

    else:
        # otherwise get the figure from the passed axis
        fig = ax.get_figure()

    ##########################

    ## TRACK PLOTTING:
    (long_patch, long_path), (short_patch, short_path) = _plot_helper_add_track_shapes(grid_bin_bounds=grid_bin_bounds, is_dark_mode=is_dark_mode, debug_print=debug_print)
    # Draw the long/short track shapes: __________________________________________________________________________________ #
    if ax is not None:
        ax.add_patch(long_patch)
        ax.add_patch(short_patch)
        ax.autoscale()

        

    ## INPUTS: LR_only_decoder_aclu_MAX_peak_maps_df, long_peak_x, short_peak_x, peak_x_diff

    # Define a colormap to map your unique integer indices to colors
    # colormap = plt.cm.viridis  # or any other colormap

    if unit_id_colors_map is None:
        # Create a constant colormap with only white color
        if is_dark_mode:
            colormap = mcolors.ListedColormap(['white'])
        else:
            colormap = mcolors.ListedColormap(['black'])

        if isinstance(active_aclus[0], str):
            # string aclus:
            unit_id_colors_map = {}
            color = [unit_id_colors_map.get(an_aclu, _default_bg_color) for an_aclu in active_aclus]
            get_aclu_color_fn = lambda an_aclu: unit_id_colors_map.get(an_aclu, _default_bg_color)

        else:
            normalize = mcolors.Normalize(vmin=active_aclus.min(), vmax=active_aclus.max())
            scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
            color = scalar_map.to_rgba(active_aclus)
            ## constant:
            # color = 'white'
            get_aclu_color_fn = lambda an_aclu: scalar_map.to_rgba(an_aclu)
        
    else:
        ## use the provided `unit_id_colors_map`:
        # color = [unit_id_colors_map[an_aclu] for an_aclu in active_aclus]
        color = [unit_id_colors_map.get(an_aclu, _default_bg_color) for an_aclu in active_aclus]
        get_aclu_color_fn = lambda an_aclu: unit_id_colors_map.get(an_aclu, _default_bg_color)


    ## Count-based offsets: If there are three aclus sharing a bin, offset the repeated aclus by a scaled y-factor:
    if (aclus_y_offset_mode.value == AclusYOffsetMode.CountBased.value):
        offset_scaling_factor: float = 0.1

        # Add a new column with the count of repeated entries
        a_dir_decoder_aclu_MAX_peak_maps_df['long_repeated_count'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(long_column_name).cumcount()
        a_dir_decoder_aclu_MAX_peak_maps_df['short_repeated_count'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(short_column_name).cumcount()

        # a_dir_decoder_aclu_MAX_peak_maps_df['long_repeated_count'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(long_column_name).agg(['cumcount', ''])
        # a_dir_decoder_aclu_MAX_peak_maps_df['short_repeated_count'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(short_column_name).cumcount()

        # Use `transform` to broadcast the maximum count to all rows in the original DataFrame
        a_dir_decoder_aclu_MAX_peak_maps_df['long_MaxRepeat'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(long_column_name)['long_repeated_count'].transform('max')
        a_dir_decoder_aclu_MAX_peak_maps_df['short_MaxRepeat'] = a_dir_decoder_aclu_MAX_peak_maps_df.groupby(short_column_name)['short_repeated_count'].transform('max')


        long_y_offsets = a_dir_decoder_aclu_MAX_peak_maps_df['long_repeated_count'].to_numpy() * offset_scaling_factor
        short_y_offsets = a_dir_decoder_aclu_MAX_peak_maps_df['short_repeated_count'].to_numpy() * offset_scaling_factor

        # long arrows should start at the top of the stack (maximum y)
        long_y_arrow_offsets = a_dir_decoder_aclu_MAX_peak_maps_df['long_MaxRepeat'].to_numpy() * offset_scaling_factor
        # long_y_arrow_val = (np.full_like(long_peak_x, long_y_baseline)) * base_1D_height
        long_y_arrow_val = (np.full_like(long_peak_x, long_y_baseline) + long_y_arrow_offsets) * base_1D_height
        # short arrows should end at the bottom of the stack (minimum y)
        short_y_arrow_val = (np.full_like(short_peak_x, short_y_baseline)) * base_1D_height

        long_y = (np.full_like(long_peak_x, long_y_baseline) + long_y_offsets) * base_1D_height
        short_y = (np.full_like(short_peak_x, short_y_baseline) + short_y_offsets) * base_1D_height

    elif (aclus_y_offset_mode.value == AclusYOffsetMode.RandomJitter.value):
        ## Random Jitter-based offsets:
        random_y_jitter = np.random.ranf((np.shape(active_aclus)[0], )) * 0.05
        # random_y_jitter = np.random.ranf((np.shape(active_aclus)[0], )) * 0.1
        # random_y_jitter = np.zeros((np.shape(active_aclus)[0], )) # no jitter
        long_y = (np.full_like(long_peak_x, long_y_baseline)+random_y_jitter) * base_1D_height
        short_y = (np.full_like(short_peak_x, short_y_baseline)+random_y_jitter) * base_1D_height
    else:
        raise NotImplementedError(f"aclus_y_offset_mode: {aclus_y_offset_mode} not implemented.")
    
    
    # Draw the circle points _____________________________________________________________________________________________ #
    # circle_points_kwargs = dict(alpha=0.9, picker=enable_interactivity, s=30.0, c=color)
    # circle_points_kwargs = dict(alpha=0.9, picker=enable_interactivity, s=25.0, edgecolors=color, c='#AAAAAA33', marker='o', plotnonfinite=False)
    # circle_points_kwargs = dict(alpha=0.9, picker=enable_interactivity, s=np.full_like(active_aclus, fill_value=scatter_point_size), edgecolors=color, facecolors=(['#CCCCCC33'] * len(active_aclus)), marker='o', plotnonfinite=False)
    circle_points_kwargs = dict(alpha=0.9, picker=enable_interactivity, s=np.full((len(active_aclus),), fill_value=scatter_point_size), edgecolors=([_default_edgecolors] * len(active_aclus)), facecolors=color, marker='o', plotnonfinite=False)

    _out_long_points = ax.scatter(long_peak_x, y=long_y, label='long_peak_x', **circle_points_kwargs)
    _out_short_points = ax.scatter(short_peak_x, y=short_y, label='short_peak_x', **circle_points_kwargs)

    ## OUTPUT Variables:
    _output_dict = {'long_scatter': _out_long_points, 'short_scatter': _out_short_points}
    _output_by_aclu_dict = {} # keys are integer aclus, values are dictionaries of returned graphics objects:

    # Draw arrows from the first set of points to the second set _________________________________________________________ #
    # arrowprops_kwargs = dict(arrowstyle="->", alpha=0.6)
    # arrowprops_kwargs = dict(arrowstyle="simple", alpha=0.7)
    arrowprops_kwargs = dict(arrowstyle="fancy, head_length=0.25, head_width=0.25, tail_width=0.05", alpha=0.6)
    # , mutation_scale=10

    ## need to take both to str, or both to int
    both_aclus = [str(v) for v in both_aclus]

    for idx, aclu_val in enumerate(active_aclus):
        # aclu_val: int = int(aclu_val)
        aclu_val: str = str(aclu_val)
        if aclu_val not in _output_by_aclu_dict:
            _output_by_aclu_dict[aclu_val] = {}

        if aclu_val in both_aclus:
            # Starting point coordinates
            start_x = long_peak_x[idx]
            # start_y = 0.1 + random_y_jitter[idx]
            start_y = long_y[idx]
            # End point coordinates
            end_x = short_peak_x[idx]
            # end_y = 0.75 + random_y_jitter[idx]
            end_y = short_y[idx]

            ## override the y-positions to ensure that all arrows neatly attach only to the bottom point when using `(aclus_y_offset_mode.value == AclusYOffsetMode.CountBased.value)` mode
            if ((aclus_y_offset_mode.value == AclusYOffsetMode.CountBased.value) and enable_single_y_point_arrows):
                start_y = long_y_arrow_val[idx]
                end_y = short_y_arrow_val[idx]

            # Calculate the change in x and y for the arrow
            # dx = end_x - start_x
            # dy = end_y - start_y

            # Get the corresponding color for the current index using the colormap
            arrow_color = get_aclu_color_fn(active_aclus[idx])
            
            # Annotate the plot with arrows; adjust the properties according to your needs
            _output_by_aclu_dict[aclu_val]['long_to_short_arrow'] = ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(**arrowprops_kwargs, color=arrow_color), label=str(active_aclus[idx]))
            # _output_by_aclu_dict[aclu_val]['long_to_short_arrow'].arrowprops: {'arrowstyle': '->', 'color': (0.267004, 0.004874, 0.329415, 1.0), 'alpha': 0.6}
            # _output_by_aclu_dict[aclu_val]['long_to_short_arrow'].arrow_patch # mpl.patches.FancyArrowPatch
            # _output_by_aclu_dict[aclu_val]['long_to_short_arrow'].arrow_patch.set_color('')
        else:
            _output_by_aclu_dict[aclu_val]['long_to_short_arrow'] = None


    # ACLU Point Text Labels _____________________________________________________________________________________________ #
    if draw_point_aclu_labels:
        
        text_kwargs = dict(color=aclu_labels_text_color, fontsize=aclu_labels_fontsize, ha='center', va='center')
        # Add text labels to scatter points
        for i, aclu_val in enumerate(active_aclus):
            # aclu_val: int = int(aclu_val)
            aclu_val: str = str(aclu_val)

            if aclu_val not in appearing_long_to_short_aclus:
                a_long_text = ax.text(long_peak_x[i], long_y[i], str(aclu_val), **text_kwargs)
                a_long_text.set_path_effects(aclu_labels_text_path_effects)
            else:
                a_long_text = None

            if aclu_val not in disappearing_long_to_short_aclus:
                a_short_text = ax.text(short_peak_x[i], short_y[i], str(aclu_val), **text_kwargs)
                a_short_text.set_path_effects(aclu_labels_text_path_effects)
            else:
                a_short_text = None

            if aclu_val not in _output_by_aclu_dict:
                _output_by_aclu_dict[aclu_val] = {}
            _output_by_aclu_dict[aclu_val]['long_text'] = a_long_text
            _output_by_aclu_dict[aclu_val]['short_text'] = a_short_text

        if enable_adjust_overlapping_text:
            from adjustText import adjust_text
            # Call adjust_text function to adjust the positions of text labels
            adjust_text([v['long_text'] for v in _output_by_aclu_dict.values()], ax=ax)
            adjust_text([v['short_text'] for v in _output_by_aclu_dict.values()], ax=ax)

            # adjust_text([v['long_text'] for v in _output_by_aclu_dict.values()] + [v['short_text'] for v in _output_by_aclu_dict.values()], ax=ax,
            #             #  expand=(1.2, 2), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
            #             arrowprops=dict(arrowstyle='-', color='gray', alpha=.5), # ensure the labeling is clear by adding small arrows indicating which point a text lable belongs to (these are different than the remapping arrows)
            #             # avoid_self=False,
            #             # force_text=(0.5, 0),# Since the movements are so contrained, high force speeds up the process a lot
            #             # expand=(1, 1), # We want them to be quite compact, so reducing expansion makes sense
            #             # only_move='x-', #Only allow movement to the left
            #             # only_move='y', #Only allow movement on y
            #             # max_move=None,
            #             # autoalign=True
            # )


        


    if enable_interactivity:
        ## Build the interactivity callbacks:
        previous_selected_indices = []

        if not isinstance(active_aclus, NDArray):
            active_aclus = np.array(active_aclus)

        index_is_selected = np.full_like(active_aclus, fill_value=False, dtype=bool)
        

        # if did_reclick_same_selection:

        def _perform_update_scatter_point_color(_out_points, an_index: int, new_color):
            # _out_points._facecolors[an_index] = new_color # scalar_map.to_rgba(active_aclus[index])
            _out_points._edgecolors[an_index] = new_color
            

        def _perform_update_aclu_is_selected(an_index: int, an_aclu: int, is_selected: bool):
            """ Updates the selection state for the specific aclu

            captures: _out_long_points, _out_short_points, _output_by_aclu_dict, 
                scatter_edgecolors_selection_color, selection_color, aclu_labels_fontsize, scalar_map,
                aclu_labels_text_path_effects, selection_text_path_effects, get_aclu_color_fn
            
            _perform_update_scatter_point_is_selected(an_index=an_index, an_aclu=active_aclus[an_index], is_selected=True)
            """
            if is_selected:
                # selected
                # original_color = scalar_map.to_rgba(an_aclu)
                original_color = get_aclu_color_fn(an_aclu)
                active_color = original_color
                # active_color = scatter_edgecolors_selection_color

                active_arrow_color = selection_color
                active_aclu_labels_text_color = selection_color
                active_aclu_labels_fontsize = (aclu_labels_fontsize+0.1)
                active_aclu_labels_text_path_effects = selection_text_path_effects
                active_scatter_point_size = scatter_point_selected_size
                
            else:
                # not-selected
                # original_color = scalar_map.to_rgba(an_aclu)
                original_color = get_aclu_color_fn(an_aclu)
                active_color = original_color
                active_arrow_color = original_color
                active_aclu_labels_text_color = aclu_labels_text_color
                active_aclu_labels_fontsize = aclu_labels_fontsize
                active_aclu_labels_text_path_effects = aclu_labels_text_path_effects
                active_scatter_point_size = scatter_point_size


            # _perform_update_scatter_point_color(_out_long_points, an_index=an_index, new_color=active_color)
            # _perform_update_scatter_point_color(_out_short_points, an_index=an_index, new_color=active_color)

            for _out_points in [_out_long_points, _out_short_points]:
                _perform_update_scatter_point_color(_out_points, an_index=an_index, new_color=active_color)
                _out_points._sizes[an_index] = active_scatter_point_size

            # _out_long_points.get_paths()[an_index].set_path_effects(active_aclu_labels_text_path_effects)
            # _out_short_points.get_paths()[an_index].set_path_effects(active_aclu_labels_text_path_effects)


            # Change the arrow selection:
            a_paired_arrow_container_Text_obj = _output_by_aclu_dict.get(an_aclu, {}).get('long_to_short_arrow', None)
            if a_paired_arrow_container_Text_obj is not None:
                a_paired_arrow: mpl.patches.FancyArrowPatch = a_paired_arrow_container_Text_obj.arrow_patch
                a_paired_arrow.set_color(active_arrow_color)  # Change arrow color to blue

            if draw_point_aclu_labels:
                _output_by_aclu_dict[an_aclu]['long_text'].set_color(active_aclu_labels_text_color)
                _output_by_aclu_dict[an_aclu]['short_text'].set_color(active_aclu_labels_text_color)

                _output_by_aclu_dict[an_aclu]['long_text'].set_fontsize(active_aclu_labels_fontsize)
                _output_by_aclu_dict[an_aclu]['short_text'].set_fontsize(active_aclu_labels_fontsize)
    
                _output_by_aclu_dict[an_aclu]['long_text'].set_path_effects(active_aclu_labels_text_path_effects)
                _output_by_aclu_dict[an_aclu]['short_text'].set_path_effects(active_aclu_labels_text_path_effects)

                if not is_selected:
                    _output_by_aclu_dict[an_aclu]['long_text'].set_zorder(10)  # Choose a z-order value higher than other objects
                    _output_by_aclu_dict[an_aclu]['short_text'].set_zorder(10)
                


        def on_scatter_point_pick(event):
            """ 
            Captures: active_aclus, scalar_map, _out_long_points, _out_short_points, _output_by_aclu_dict, long_peak_x, long_y, selection_color, aclu_labels_fontsize, get_aclu_color_fn
            """
            nonlocal previous_selected_indices, index_is_selected
            newly_selected_ind = event.ind
            if debug_print:
                print(f'on_scatter_point_pick(event: {event}):\n\t', newly_selected_ind, long_peak_x[newly_selected_ind], long_y[newly_selected_ind])
            # Check which subplot the event originated from
            artist = event.artist
            if event.artist not in ax.collections:  # Check if the event originated from the scatter plot in ax1
                # Your code for handling pick events in the first subplot
                # print(f'\t not intended for this ax. Skipping.')
                return

            if len(newly_selected_ind)>1:
                print(f'WARN: multiple indicies selected: {newly_selected_ind} -- selecting only the first item.')
                newly_selected_ind = np.array([newly_selected_ind[0]]) # only get the first item if multiple are selected.


            # did_reclick_same_selection: bool = np.all(newly_selected_ind == previous_selected_indices) # lists are the same
            index_is_selected[newly_selected_ind] = np.logical_not(index_is_selected[newly_selected_ind]) # negate the selection
            any_changed_idxs = np.union1d(previous_selected_indices, newly_selected_ind)
            if debug_print:
                print(f'\tprevious_selected_indices: {previous_selected_indices}')
                print(f'\tnewly_selected_ind: {newly_selected_ind}')
                print(f'\tany_changed_idxs: {any_changed_idxs}')

            for an_index in any_changed_idxs:
                an_index: int = int(an_index)
                if debug_print:
                    print(f'\tan_index: {an_index}, type(an_index): {type(an_index)}, active_aclus: {active_aclus}, type(active_aclus): {type(active_aclus)} ')
                aclu_changed: int = int(active_aclus[an_index])
                if debug_print:
                    print(f'\taclu_changed: {aclu_changed}')
                _perform_update_aclu_is_selected(an_index=an_index, an_aclu=aclu_changed, is_selected=index_is_selected[an_index])


            previous_selected_indices = np.nonzero(index_is_selected==True)

            plt.draw()  # Update the plot

        _mpl_pick_event_handle_idx: int = fig.canvas.mpl_connect('pick_event', on_scatter_point_pick)


        _output_dict['get_aclu_color_fn'] = get_aclu_color_fn
        _output_dict['scatter_select_function'] = on_scatter_point_pick
        _output_dict['_scatter_select_mpl_pick_event_handle_idx'] = _mpl_pick_event_handle_idx

    ## format tha axes:
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Show the plot
    if not defer_render:
        plt.show()
    return fig, ax, (_output_dict, _output_by_aclu_dict)


@function_attributes(short_name='bidir_track_remap', tags=['figure', 'remap', 'track'], input_requires=[], output_provides=[], uses=['_get_directional_pf_peaks_dfs', '_plot_track_remapping_diagram', 'build_shared_sorted_neuron_color_maps'], used_by=[], creation_date='2024-04-29 10:23', related_items=[])
def plot_bidirectional_track_remapping_diagram(track_templates, grid_bin_bounds, active_context=None, perform_write_to_file_callback=None, defer_render: bool=False,
                                                enable_interactivity:bool=True, is_dark_mode:bool=True, aclus_y_offset_mode: AclusYOffsetMode = AclusYOffsetMode.RandomJitter,
                                                use_separate_plot_for_each_direction:bool=True, use_unique_aclu_colors:bool=False, drop_aclu_if_missing_long_or_short:bool=False,
                                                **kwargs):   
    """ 
    Usage:
    
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import plot_bidirectional_track_remapping_diagram

        collector = plot_bidirectional_track_remapping_diagram(track_templates, grid_bin_bounds=long_pf2D.config.grid_bin_bounds, active_context=curr_active_pipeline.build_display_context_for_session(display_fn_name='plot_bidirectional_track_remapping_diagram'))

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText

    from matplotlib.gridspec import GridSpec
    from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, perform_update_title_subtitle
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _plot_track_remapping_diagram
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import build_shared_sorted_neuron_color_maps
    
    if active_context is not None:
            display_context = active_context.adding_context('display_fn', display_fn_name='bidir_track_remap')
        
    with mpl.rc_context({'figure.figsize': (10, 4), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, }):
        # Create a FigureCollector instance
        with FigureCollector(name='plot_bidirectional_track_remapping_diagram', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22>Track <weight:bold>Remapping</></>'):
                """ Captures:

                t_split
                """
                a_collector.contexts.append(sub_context)
                
                # `flexitext` version:
                text_formatter = FormattedFigureText()
                # ax.set_title('')
                fig.suptitle('')
                text_formatter.setup_margins(fig)
                title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin,
                                        title,
                                        va="bottom", xycoords="figure fraction")
                footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                                            text_formatter._build_footer_string(active_context=sub_context),
                                            va="top", xycoords="figure fraction")
            
                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)


            # BEGIN FUNCTION BODY
            (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df = track_templates.get_directional_pf_maximum_peaks_dfs(drop_aclu_if_missing_long_or_short=drop_aclu_if_missing_long_or_short)


            # AnyDir_decoder_aclu_MAX_peak_maps_df.aclu.to_numpy()
            neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in track_templates.get_decoders_dict().values()] # [A, B, C, D, ...]
            # _unit_qcolors_map, unit_colors_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)


            if use_unique_aclu_colors:
                unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists, return_255_array=False)
                # _by_LR = LR_only_decoder_aclu_MAX_peak_maps_df.sort_values(by=['long_LR'], inplace=False)
                _by_ANY: pd.DataFrame = AnyDir_decoder_aclu_MAX_peak_maps_df.sort_values(by=['long_LR', 'long_RL'], inplace=False) ## sort by peak location on Long track
                long_peak_sorted_unit_colors_ndarray_map = dict(zip(_by_ANY.index.to_numpy(), list(_unit_colors_ndarray_map.values())))
                unit_id_colors_map = long_peak_sorted_unit_colors_ndarray_map

            else:
                unit_id_colors_map = None

            
            # kwargs = dict(draw_point_aclu_labels=True, enable_interactivity=False, enable_adjust_overlapping_text=False, unit_id_colors_map=_unit_colors_ndarray_map)
            kwargs = dict(draw_point_aclu_labels=True, enable_interactivity=enable_interactivity, enable_adjust_overlapping_text=False, unit_id_colors_map=unit_id_colors_map, is_dark_mode=is_dark_mode, aclus_y_offset_mode=aclus_y_offset_mode)

            ## Either way, make a single figure for both LR/RL remapping cells:
            if use_separate_plot_for_each_direction:
                ## Make two separate axes for LR/RL remapping cells:
                fig, axs = collector.subplots(nrows=2, ncols=1, sharex=True, sharey=True, num='Track Remapping', figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True, clear=True)
                assert len(axs) == 2, f"{len(axs)}"
                ax_dict = {'ax_LR': axs[0], 'ax_RL': axs[1]}

                fig, ax_LR, _outputs_tuple_LR = _plot_track_remapping_diagram(LR_only_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds=grid_bin_bounds, long_column_name='long_LR', short_column_name='short_LR', ax=ax_dict['ax_LR'], defer_render=defer_render, **kwargs)
                perform_update_title_subtitle(fig=fig, ax=ax_LR, title_string=None, subtitle_string=f"LR Track Remapping - {len(LR_only_decoder_aclu_MAX_peak_maps_df)} neurons")
                fig, ax_RL, _outputs_tuple_RL = _plot_track_remapping_diagram(RL_only_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds=grid_bin_bounds, long_column_name='long_RL', short_column_name='short_RL', ax=ax_dict['ax_RL'], defer_render=defer_render, **kwargs)
                perform_update_title_subtitle(fig=fig, ax=ax_RL, title_string=None, subtitle_string=f"RL Track Remapping - {len(RL_only_decoder_aclu_MAX_peak_maps_df)} neurons")

                setup_common_after_creation(collector, fig=fig, axes=[ax_LR, ax_RL], sub_context=display_context.adding_context('subplot', subplot_name='Track Remapping'))
            else:
                ## plot both LR/RL cells on a single combined axes:
                fig, axs = collector.subplots(nrows=1, ncols=1, sharex=True, sharey=True, num='Track Remapping', figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True, clear=True)
                # assert len(axs) == 1, f"{len(axs)}"
                ax = axs

                fig, ax, _outputs_tuple = _plot_track_remapping_diagram(AnyDir_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds=grid_bin_bounds, long_column_name='long_LR', short_column_name='short_LR', ax=ax, defer_render=defer_render, **kwargs)
                perform_update_title_subtitle(fig=fig, ax=ax, title_string=None, subtitle_string=f"LR+RL Track Remapping - {len(LR_only_decoder_aclu_MAX_peak_maps_df)} neurons")

                setup_common_after_creation(collector, fig=fig, axes=[ax, ], sub_context=display_context.adding_context('subplot', subplot_name='Track Remapping'))


    return collector



if __name__ == '__main__':

    app, w, cw, (long_track_dims, long_rect_items, long_rects), (short_track_dims, short_rect_items, short_rects) = test_LinearTrackDimensions_2D_pyqtgraph(long_track_dims=None, short_track_dims=None)    
    sys.exit(app.exec_())


