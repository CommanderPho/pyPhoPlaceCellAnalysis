from typing import Tuple
from attrs import define, field, Factory
from collections import namedtuple
import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui
from pyphoplacecellanalysis.External.pyqtgraph import PlotItem

# Define the named tuple
ScaleFactors = namedtuple("ScaleFactors", ["major", "minor"])


import matplotlib.axes
import matplotlib.patches as patches # for matplotlib version of the plot

import pyvista as pv # for 3D support in `LinearTrackDimensions3D`


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
    alignment_axis: str = field(default='x')
    axis_scale_factors: ScaleFactors = field(default=ScaleFactors(1.0, 1.0))  # Major and minor axis scale factors

    @property
    def total_length(self) -> float:
        # unscaled total length including both end platforms
        return (self.track_length + (2.0 * self.platform_side_length))
    
    @property
    def total_width(self) -> float:
        # unscaled total width including both end platforms
        return max(self.track_width, self.platform_side_length)


    def _build_component_rectangles(self):
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        pen = pg.mkPen({'color': "#FF0", 'width': 2})
        brush = pg.mkBrush("#FF0")


        if self.alignment_axis == 'x':
            scaled_track_length = (major_axis_factor * self.track_length)
            scaled_track_width = (minor_axis_factor * self.track_width)
            scaled_platform_size = ((major_axis_factor * self.platform_side_length), (minor_axis_factor * self.platform_side_length))

            track_center_y = major_axis_factor * self.platform_side_length / 2.0
            track_origin_y = track_center_y - (scaled_track_width / 2.0)
            
            rects = [
                (0, 0, *scaled_platform_size, pen, brush),
                (scaled_platform_size[0], track_origin_y, scaled_track_length, scaled_track_width, pen, brush),
                (major_axis_factor * (self.platform_side_length + self.track_length), 0, *scaled_platform_size, pen, brush)
            ]

        elif self.alignment_axis == 'y':
            track_center_x = major_axis_factor * self.platform_side_length / 2.0
            track_origin_x = track_center_x - minor_axis_factor * self.track_width / 2.0
            scaled_platform_size = ((minor_axis_factor * self.platform_side_length), (major_axis_factor * self.platform_side_length))
            rects = [
                (0, 0, *scaled_platform_size, pen, brush),
                (track_origin_x, major_axis_factor * self.platform_side_length,
                    (minor_axis_factor * self.track_width), 
                    (major_axis_factor * self.track_length),
                    pen, brush),
                (0, major_axis_factor * (self.platform_side_length + self.track_length),*scaled_platform_size, pen, brush)
            ]
        else:
            raise ValueError(f"Unsupported alignment_axis: {self.alignment_axis}")

        return rects
        
    def plot_rects(self, plot_item, offset=None):
        """ main function to plot """
        rects = self._build_component_rectangles()
        rect_items = [] # probably do not need
        if offset is not None:
            offset_rects = []# [list(rect) for rect in rects]
            # for an_offset_rect in offset_rects:
            for a_rect in rects:
                an_offset_rect = list(a_rect)
                an_offset_rect[0] = a_rect[0] + offset[0]
                an_offset_rect[1] = a_rect[1] + offset[1]
                offset_rects.append(tuple(an_offset_rect))
            rects = offset_rects

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
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='red')
                plot_item.add_patch(rect)                
                rect_items.append(rect)
            else:
                raise ValueError("Unsupported plot item type.")

        if isinstance(plot_item, PlotItem):
            plot_item.setAspectLocked()
            plot_item.setClipToView(True)
        elif isinstance(plot_item, matplotlib.axes.Axes):
            plot_item.set_aspect('equal', 'box')    
        else:
            raise ValueError("Unsupported plot item type.")
        return rect_items, rects


# ==================================================================================================================== #
# 3D Support                                                                                                           #
# ==================================================================================================================== #

from pyphocorehelpers.geometry_helpers import point_tuple_mid_point


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
    

    @classmethod
    def init_from_grid_bin_bounds(cls, grid_bin_bounds, return_geoemtry=True):
        ((x, x2), (y, y2)) = grid_bin_bounds #TODO BUG 2023-09-13 13:19: - [ ] incorrect: `((x, y), (x2, y2)) = grid_bin_bounds` -> correction: `((x, x2), (y, y2)) = grid_bin_bounds` I think this is incorrect interpretation of grid_bin_bounds everywhere I used it in the LinearTrackDimensions and LinearTrackDimensions3D, maybe explains why `LinearTrackDimensions` wasn't working right either.
        _length, _width = abs(x2 - x), abs(y2 - y)
        print(f'_length: {_length}, _width: {_width}')
        _obj = cls()
        

        deduced_track_length: float = _length - (_obj.platform_side_length * 2.0)
        print(f'deduced_track_length: {deduced_track_length}')
        _obj.track_length = deduced_track_length
        ## TODO for now just keep the track_width and platform_side_lengths fixed, ignoring the grid_bin_bounds, since we know those from physical dimension measurements        

        grid_bin_bounds_extents = (x, y, _length, _width)
        axis_scale_factors = (1.0/_length, 1.0/_width)
        
        # BUILD THE GEOMETRY
        if return_geoemtry:
            # grid_bin_bounds_center_point = (point_tuple_mid_point(grid_bin_bounds[0]), point_tuple_mid_point(grid_bin_bounds[1]))
            # a_position_offset = (*grid_bin_bounds_center_point, 0.0) #- a_track_dims.get_center_point()
            # a_center_correction_offset = ((_obj.platform_side_length/2.0)-_obj.get_center_point()[0], 0.0, 0.0)
            # a_position_offset = np.array(a_position_offset) + np.array(a_center_correction_offset)
            maze_pdata = _obj.build_maze_geometry(position_offset=a_position_offset, grid_bin_bounds=grid_bin_bounds)
            return _obj, maze_pdata
        else:
            return _obj
        
    def build_maze_geometry(self, position_offset=(0, 0, -0.01), from_grid_bin_bounds=None):
        """ builds the maze geometry for use with pyvista.
        
        """
        size = self.platform_side_length
        
        if from_grid_bin_bounds is not None:
            # This mode computes the correct position_offset point from the grid_bin_bounds provided and self's center properties
            assert position_offset is None, f"from_grid_bin_bounds is provided and in this mode position_offset will not be used!"
            grid_bin_bounds_center_point = (point_tuple_mid_point(from_grid_bin_bounds[0]), point_tuple_mid_point(from_grid_bin_bounds[1]))
            a_position_offset = (*grid_bin_bounds_center_point, 0.0) #- a_track_dims.get_center_point()
            a_center_correction_offset = ((self.platform_side_length/2.0)-self.get_center_point()[0], 0.0, 0.0)
            a_position_offset = np.array(a_position_offset) + np.array(a_center_correction_offset)
            return self.build_maze_geometry(position_offset=a_position_offset, from_grid_bin_bounds=None) # call `build_maze_geometry` in simple position_offset mode
            
        else:
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
     
