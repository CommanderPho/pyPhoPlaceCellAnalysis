"""
VisPy Line with Heading-to-Hue Mapping (North = Red)

Uses scene.SceneCanvas and scene.visuals.Line with CPU-side heading-to-color
to avoid custom gloo shaders (which fail on some Windows/drivers).

Heading angle mapping:
- 0° (North/Up) = Red
- 90° (East/Right) = Cyan
- 180° (South/Down) = Green
- 270° (West/Left) = Magenta
- 360° wraps back to Red
"""

import numpy as np
from vispy import app, scene
from vispy.scene import visuals

from pyphocorehelpers.plotting.heading_angle_helpers import HeadingAngleHelpers

# class AngularrColoredLine(scene.visuals.Line):
#     def __init__(self, **kwargs):
# 		t = np.linspace(0, 4 * np.pi, 1000)
#         x = t * np.cos(t) * 0.1
#         y = t * np.sin(t) * 0.1
    
#         self._positions = np.c_[x, y].astype(np.float32)
#         self._colors = HeadingAngleHelpers._positions_to_vertex_colors(self._positions)
#         self._view = self.central_widget.add_view(camera='panzoom')
#         self._view.camera.aspect = 1
#         self._line = scene.visuals.Line(pos=self._positions, color=self._colors, width=2.0, parent=self._view.scene)
#         self._line.set_gl_state('translucent', depth_test=False)


class AngleColoredLineVisual(scene.visuals.Line):
    """ A direct drop-in replacement for scene.Line
    from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import HeadingColoredLine, CompassDemo, InteractiveHeadingLine
    from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import AngleColoredLineVisual

    line = AngleColoredLineVisual(pos=pos, color=vertex_colors, method='gl')
    line.parent = view.scene

    """
    def __init__(self, *args, **kwargs):
        scene.visuals.Line.__init__(self, *args, **kwargs)
        self.unfreeze()
        _colors = HeadingAngleHelpers._positions_to_vertex_colors(self.pos)
        self.set_data(pos=self.pos, color=_colors)
        # # initialize point markers
        # self.markers = scene.visuals.Markers(parent=self)
        # self.marker_colors = np.ones((len(self.pos), 4), dtype=np.float32)
        # self.markers.set_data(pos=self.pos, symbol="s", edge_color="red", size=6)
        # self.selected_point = None
        # self.selected_index = -1
        # # snap grid size
        # self.gridsize = 10
        self.set_gl_state('translucent', depth_test=False)
        self.freeze()
        



class HeadingColoredLine(scene.SceneCanvas):
    """ 
    from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import HeadingColoredLine, CompassDemo, InteractiveHeadingLine
    """
    def __init__(self, **kwargs):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 600), **kwargs)
        self.unfreeze()
        t = np.linspace(0, 4 * np.pi, 1000)
        x = t * np.cos(t) * 0.1
        y = t * np.sin(t) * 0.1
        self._positions = np.c_[x, y].astype(np.float32)
        self._colors = HeadingAngleHelpers._positions_to_vertex_colors(self._positions)
        self._view = self.central_widget.add_view(camera='panzoom')
        self._view.camera.aspect = 1
        self._line = scene.visuals.Line(pos=self._positions, color=self._colors, width=2.0, parent=self._view.scene)
        self._line.set_gl_state('translucent', depth_test=False)
        self._view.camera.set_range(x=(self._positions[:, 0].min() - 0.1, self._positions[:, 0].max() + 0.1), y=(self._positions[:, 1].min() - 0.1, self._positions[:, 1].max() + 0.1))
        self.show()



class CompassLegendItem:
    """ from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import CompassLegendItem
    
    """
    def __init__(self, view: scene.ViewBox, center = np.array([0.0, 0.0]), length = 0.6, 
                 line_points: int = 20, line_width=2.0, **kwargs):

        self._data_dict = dict()


        angles = np.linspace(0, 2 * np.pi, 9)[:-1]

        all_positions = []
        all_tangents = []

        for i, angle in enumerate(angles):
            line_length = length if i % 2 == 0 else length * 0.5
            end = center + line_length * np.array([np.cos(angle), np.sin(angle)])

            
            t_ = np.linspace(0, 1, line_points)
            x = center[0] + t_ * (end[0] - center[0])
            y = center[1] + t_ * (end[1] - center[1])

            positions = np.c_[x, y]
            tangent = (end - center) / np.linalg.norm(end - center)
            tangents = np.tile(tangent, (line_points, 1))

            all_positions.append(positions)
            all_tangents.append(tangents)

            # break line
            all_positions.append(np.array([[np.nan, np.nan]]))
            all_tangents.append(np.array([[0.0, 0.0]]))

        positions = np.vstack(all_positions).astype(np.float32)
        tangents = np.vstack(all_tangents)

        angle_rad = np.arctan2(tangents[:, 1], tangents[:, 0])
        angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0
        compass_deg = HeadingAngleHelpers._heading_deg_to_compass_deg(angle_deg)

        colors = HeadingAngleHelpers.heading_angles_to_rainbow_colors(compass_deg, alpha=1.0)

        self.line = scene.visuals.Line(pos=positions, color=colors, width=line_width, parent=view.scene)
        self.line.set_gl_state('translucent', depth_test=False)

        # self.circle = scene.visuals.Ellipse(center=(float(center[0]), float(center[1])), radius=(circle_radius, circle_radius), color=None, border_width=int(circle_border_width), border_color=circle_color, parent=view.scene)
        # self.circle.set_gl_state('translucent', depth_test=False)


        self._data_dict = dict(pos=positions, tangents=tangents,
                               angle_deg=angle_deg, compass_deg=compass_deg,
                                colors=colors)


        
    
class CompassDemo(scene.SceneCanvas):
    """
    Demonstrates the heading-to-color mapping with lines pointing in cardinal directions
    """
    def __init__(self, **kwargs):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 800), **kwargs)
        self.unfreeze()
        center = np.array([0.0, 0.0])
        length = 0.6
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        all_positions = []
        all_tangents = []
        for i, angle in enumerate(angles):
            line_length = length if i % 2 == 0 else length * 0.5
            end = center + line_length * np.array([np.cos(angle), np.sin(angle)])
            line_points = 20
            t_ = np.linspace(0, 1, line_points)
            x = center[0] + t_ * (end[0] - center[0])
            y = center[1] + t_ * (end[1] - center[1])
            positions = np.c_[x, y]
            tangent = (end - center) / np.linalg.norm(end - center)
            tangents = np.tile(tangent, (line_points, 1))
            all_positions.append(positions)
            all_tangents.append(tangents)
            all_positions.append(np.array([[np.nan, np.nan]]))
            all_tangents.append(np.array([[0.0, 0.0]]))
        positions = np.vstack(all_positions).astype(np.float32)
        tangents = np.vstack(all_tangents)
        angle_rad = np.arctan2(tangents[:, 1], tangents[:, 0])
        angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0
        compass_deg = HeadingAngleHelpers._heading_deg_to_compass_deg(angle_deg)
        colors = HeadingAngleHelpers.heading_angles_to_rainbow_colors(compass_deg, alpha=1.0)
        self._view = self.central_widget.add_view(camera='panzoom')
        self._view.camera.aspect = 1
        self._line = scene.visuals.Line(pos=positions, color=colors, width=5.0, parent=self._view.scene)
        self._line.set_gl_state('translucent', depth_test=False)
        self._view.camera.set_range(x=(-1, 1), y=(-1, 1))
        self.show()
        print("\nCompass Rose Color Mapping:")
        print("North (↑, 0°): Red")
        print("Northeast (↗, 45°): Orange/Yellow")
        print("East (→, 90°): Cyan")
        print("Southeast (↘, 135°): Blue")
        print("South (↓, 180°): Green")
        print("Southwest (↙, 225°): Yellow-Green")
        print("West (←, 270°): Magenta")
        print("Northwest (↖, 315°): Red-Magenta")


class InteractiveHeadingLine(scene.SceneCanvas):
    """
    Interactive version where you can draw lines with the mouse
    """
    def __init__(self, **kwargs):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 600), **kwargs)
        self.unfreeze()
        self.positions = []
        self.drawing = False
        self._view = self.central_widget.add_view(camera='panzoom')
        self._view.camera.aspect = 1
        self._view.camera.set_range(x=(-1, 1), y=(-1, 1))
        self._line = None
        self.show()
        print("Click and drag to draw lines. Press 'C' to clear.")
        print("Draw upward for red, rightward for cyan, downward for green, leftward for magenta")


    def on_mouse_press(self, event):
        self.drawing = True
        x = 2 * event.pos[0] / self.size[0] - 1
        y = 1 - 2 * event.pos[1] / self.size[1]
        self.positions = [[x, y]]


    def on_mouse_move(self, event):
        if self.drawing:
            x = 2 * event.pos[0] / self.size[0] - 1
            y = 1 - 2 * event.pos[1] / self.size[1]
            self.positions.append([x, y])
            self._update_line()
            self.update()


    def on_mouse_release(self, event):
        self.drawing = False


    def on_key_press(self, event):
        if event.key == 'C':
            self.positions = []
            if self._line is not None:
                self._line.parent = None
                self._line = None
            self.update()


    def _update_line(self):
        if len(self.positions) < 2:
            return
        pos = np.array(self.positions, dtype=np.float32)
        colors = HeadingAngleHelpers._positions_to_vertex_colors(pos)
        if self._line is None:
            self._line = scene.visuals.Line(pos=pos, color=colors, width=3.0, parent=self._view.scene)
            self._line.set_gl_state('translucent', depth_test=False)
        else:
            self._line.set_data(pos=pos, color=colors)


if __name__ == '__main__':
    print("1. Spiral example")
    print("2. Compass rose (shows color mapping)")
    print("3. Interactive drawing")
    choice = input("Choose (1, 2, or 3): ").strip()
    if choice == '2':
        canvas = CompassDemo()
    elif choice == '3':
        canvas = InteractiveHeadingLine()
    else:
        canvas = HeadingColoredLine()
    app.run()
