from typing import Tuple
from attrs import define, field, Factory
from collections import namedtuple


from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui
from pyphoplacecellanalysis.External.pyqtgraph import PlotItem

# Define the named tuple
ScaleFactors = namedtuple("ScaleFactors", ["major", "minor"])


import matplotlib.axes
import matplotlib.patches as patches # for matplotlib version of the plot




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

        ((x, y), (x2, y2)) = grid_bin_bounds
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
                plot_item.addItem(rect_item)
            elif isinstance(plot_item, matplotlib.axes.Axes):
                import matplotlib.patches as patches
                # matplotlib ax was passed
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='red')
                plot_item.add_patch(rect)                
            else:
                raise ValueError("Unsupported plot item type.")

        if isinstance(plot_item, PlotItem):
            plot_item.setAspectLocked()
            plot_item.setClipToView(True)
        elif isinstance(plot_item, matplotlib.axes.Axes):
            plot_item.set_aspect('equal', 'box')    
        else:
            raise ValueError("Unsupported plot item type.")
                
