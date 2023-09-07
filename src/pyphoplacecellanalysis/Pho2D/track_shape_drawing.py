from ast import Tuple
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui
from pyphoplacecellanalysis.External.pyqtgraph import PlotItem

from attrs import define, field, Factory
from collections import namedtuple

# Define the named tuple
ScaleFactors = namedtuple("ScaleFactors", ["major", "minor"])


import matplotlib.axes
import matplotlib.patches as patches # for matplotlib version of the plot

@define(slots=False)
class LinearTrackDimensions:
    """ represents a linear track comprised of two equally-sized square end platforms connected by a thin (`track_width`) straight track of length `track_length`.
        The goal is to enable plotting a graphical representation of the track along the axes of the plots using pyqtgraph. 
    
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions
    long_track_dims = LinearTrackDimensions(track_length=100.0)
    short_track_dims = LinearTrackDimensions(track_length=70.0)

    # long_track_rois = long_track_dims._build_component_rect_ROIs()
    # short_track_rois = short_track_dims._build_component_rect_ROIs()
    """
    # all units in [cm]
    track_width: float = 6.2
    track_length: float = 100.0
    platform_side_length: float = 22.0
    alignment_axis: str = field(default='x')
    axis_scale_factors: ScaleFactors = field(default=ScaleFactors(1.0, 1.0))  # Major and minor axis scale factors

    def _build_component_rectangles(self):
        major_axis_factor, minor_axis_factor = self.axis_scale_factors
        pen = pg.mkPen({'color': "#FF0", 'width': 2})
        brush = pg.mkBrush("#FF0")


        if self.alignment_axis == 'x':
            track_center_y = major_axis_factor * self.platform_side_length / 2.0
            track_origin_y = track_center_y - minor_axis_factor * self.track_width / 2.0

            cap_size = ((major_axis_factor * self.platform_side_length), (minor_axis_factor * self.platform_side_length))
            rects = [
                (0, 0, *cap_size, pen, brush),
                (major_axis_factor * self.platform_side_length, track_origin_y, major_axis_factor * self.track_length, minor_axis_factor * self.track_width, pen, brush),
                (major_axis_factor * (self.platform_side_length + self.track_length), 0, *cap_size, pen, brush)
            ]

        elif self.alignment_axis == 'y':
            track_center_x = major_axis_factor * self.platform_side_length / 2.0
            track_origin_x = track_center_x - minor_axis_factor * self.track_width / 2.0
            cap_size = ((minor_axis_factor * self.platform_side_length), (major_axis_factor * self.platform_side_length))
            rects = [
                (0, 0, *cap_size, pen, brush),
                (track_origin_x, major_axis_factor * self.platform_side_length,
                    (minor_axis_factor * self.track_width), 
                    (major_axis_factor * self.track_length),
                    pen, brush),
                (0, major_axis_factor * (self.platform_side_length + self.track_length),*cap_size, pen, brush)
            ]
        else:
            raise ValueError(f"Unsupported alignment_axis: {self.alignment_axis}")

        return rects
        
    def plot_rects(self, plot_item):
        """ main function to plot """
        rects = self._build_component_rectangles()        
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
                
