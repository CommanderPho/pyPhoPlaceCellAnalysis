import numpy as np

from vispy.util import transforms
from vispy.util.quaternion import Quaternion
from vispy.visuals.transforms import MatrixTransform
from vispy.scene.cameras.perspective import Base3DRotationCamera


class CustomArcballCamera(Base3DRotationCamera):
    """3D camera class that orbits around a center point while
    maintaining a view on a center point.

    For this camera, the ``scale_factor`` indicates the zoom level, and
    the ``center`` indicates the position to put at the center of the
    view.

    Parameters
    ----------
    fov : float
        Field of view. Zero (default) means orthographic projection.
    distance : float | None
        The distance of the camera from the rotation point (only makes sense
        if fov > 0). If None (default) the distance is determined from the
        scale_factor and fov.
    translate_speed : float
        Scale factor on translation speed when moving the camera center point.
    **kwargs : dict
        Keyword arguments to pass to `BaseCamera`.

    Notes
    -----
    Interaction:

        * LMB: orbits the view around its center point.
        * RMB or scroll: change scale_factor (i.e. zoom level)
        * MMB drag: translate the center point (same as SHIFT + LMB)
        * SHIFT + LMB: translate the center point
        * SHIFT + RMB: change FOV

    Replaces: 'view.camera = scene.cameras.ArcballCamera(fov=45)'
    
    Usage:
    
    	
		import numpy as np
		from vispy import app, scene
		from vispy.color import BaseColormap
		from pyphoplacecellanalysis.Pho2D.vispy.vispy_cameras import CustomArcballCamera

		vol = np.random.rand(64, 64, 64).astype(np.float32)  # shape is (z, y, x)

		canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
		view = canvas.central_widget.add_view()

		view.camera = CustomArcballCamera(fov=45)

		v = scene.visuals.Volume(
			vol,
			parent=view.scene,
			method='translucent',
			cmap='grey',
			interpolation='linear',
			relative_step_size=0.8,
		)

		app.run()

        
    """

    _state_props = Base3DRotationCamera._state_props + ('_quaternion',)

    def __init__(self, fov=45.0, distance=None, translate_speed=1.0, **kwargs):
        super(CustomArcballCamera, self).__init__(fov=fov, **kwargs)

        # Set camera attributes
        self._quaternion = Quaternion()
        self.distance = distance  # None means auto-distance
        self.translate_speed = translate_speed


    def viewbox_mouse_event(self, event):
        super(CustomArcballCamera, self).viewbox_mouse_event(event)
        if not self.interactive:
            return
        if event.type != 'mouse_move' or event.press_event is None:
            return
        if 1 in event.buttons and 2 in event.buttons:
            return
        if 3 not in event.buttons or event.mouse_event.modifiers:
            return
        norm = np.mean(self._viewbox.size)
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        dist = (p1 - p2) / norm * self._scale_factor
        dist[1] *= -1
        dx, dy, dz = self._dist_to_trans(dist)
        ff = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
        c = self._event_value
        self.center = c[0] + dx, c[1] + dy, c[2] + dz


    def _update_rotation(self, event):
        """Update rotation parmeters based on mouse movement"""
        p2 = event.mouse_event.pos
        if self._event_value is None:
            self._event_value = p2
        wh = self._viewbox.size
        self._quaternion = (Quaternion(*_arcball(p2, wh)) *
                            Quaternion(*_arcball(self._event_value, wh)) *
                            self._quaternion)
        self._event_value = p2
        self.view_changed()

    def _get_rotation_tr(self):
        """Return a rotation matrix based on camera parameters"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        return transforms.rotate(180 * rot / np.pi, (x, z, y))

    def _dist_to_trans(self, dist):
        """Convert mouse x, y movement into x, y, z translations"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        tr = MatrixTransform()
        tr.rotate(180 * rot / np.pi, (x, y, z))
        dx, dz, dy = np.dot(tr.matrix[:3, :3],
                            (dist[0], dist[1], 0.)) * self.translate_speed
        return dx, dy, dz

    def _get_dim_vectors(self):
        # Override vectors, camera has no sense of "up"
        return np.eye(3)[::-1]


def _arcball(xy, wh):
    """Convert x,y coordinates to w,x,y,z Quaternion parameters

    Adapted from:

    linalg library

    Copyright (c) 2010-2015, Renaud Blanch <rndblnch at gmail dot com>
    Licence at your convenience:
    GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
    BSD new <http://opensource.org/licenses/BSD-3-Clause>
    """
    x, y = xy
    w, h = wh
    r = (w + h) / 2.
    x, y = -(2. * x - w) / r, (2. * y - h) / r
    h = np.sqrt(x*x + y*y)
    return (0., x/h, y/h, 0.) if h > 1. else (0., x, y, np.sqrt(1. - h*h))



class CustomTurntableCamera(Base3DRotationCamera):
    """3D camera class that orbits around a center point while
    maintaining a view on a center point.

    For this camera, the ``scale_factor`` indicates the zoom level, and
    the ``center`` indicates the position to put at the center of the
    view.

    When ``elevation`` and ``azimuth`` are set to 0, the camera
    points along the +y axis.

    Parameters
    ----------
    fov : float
        Field of view. 0.0 means orthographic projection,
        default is 45.0 (some perspective)
    elevation : float
        Elevation angle in degrees. The elevation angle represents a
        rotation of the camera around the current scene x-axis. The
        camera points along the x-y plane when the angle is 0.
    azimuth : float
        Azimuth angle in degrees. The azimuth angle represents a
        rotation of the camera around the scene z-axis according to the
        right-hand screw rule. The camera points along the y-z plane when
        the angle is 0.
    roll : float
        Roll angle in degrees. The roll angle represents a rotation of
        the camera around the current scene y-axis.
    distance : float | None
        The distance of the camera from the rotation point (only makes sense
        if fov > 0). If None (default) the distance is determined from the
        scale_factor and fov.
    translate_speed : float
        Scale factor on translation speed when moving the camera center point.
    **kwargs : dict
        Keyword arguments to pass to `BaseCamera`.

    Notes
    -----
    Interaction:

        * LMB: orbits the view around its center point.
        * RMB or scroll: change scale_factor (i.e. zoom level)
        * MMB drag: translate the center point (same as SHIFT + LMB)
        * SHIFT + LMB: translate the center point
        * SHIFT + RMB: change FOV

    Replaces: scene.TurntableCamera
    
    Usage:
    
    	from pyphoplacecellanalysis.Pho2D.vispy.vispy_cameras import CustomTurntableCamera
    
    """

    _state_props = Base3DRotationCamera._state_props + ("elevation", "azimuth", "roll")

    def __init__(self, fov=45.0, elevation=30.0, azimuth=30.0, roll=0.0, distance=None, translate_speed=1.0, **kwargs):
        super(CustomTurntableCamera, self).__init__(fov=fov, **kwargs)

        # Set camera attributes
        self.azimuth = azimuth
        self.elevation = elevation
        self.roll = roll
        self.distance = distance  # None means auto-distance
        self.translate_speed = translate_speed


    def viewbox_mouse_event(self, event):
        super(CustomTurntableCamera, self).viewbox_mouse_event(event)
        if not self.interactive:
            return
        if event.type != 'mouse_move' or event.press_event is None:
            return
        if 1 in event.buttons and 2 in event.buttons:
            return
        if 3 not in event.buttons or event.mouse_event.modifiers:
            return
        norm = np.mean(self._viewbox.size)
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        dist = (p1 - p2) / norm * self._scale_factor
        dist[1] *= -1
        dx, dy, dz = self._dist_to_trans(dist)
        ff = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
        c = self._event_value
        self.center = c[0] + dx, c[1] + dy, c[2] + dz


    @property
    def elevation(self):
        """Get the camera elevation angle in degrees.
        
        The camera points along the x-y plane when the angle is 0.
        """
        return self._elevation
    @elevation.setter
    def elevation(self, elev):
        elev = float(elev)
        self._elevation = min(90, max(-90, elev))
        self.view_changed()


    @property
    def azimuth(self):
        """Get the camera azimuth angle in degrees.
        
        The camera points along the y-z plane when the angle is 0.
        """
        return self._azimuth
    @azimuth.setter
    def azimuth(self, azim):
        azim = float(azim)
        while azim < -180:
            azim += 360
        while azim > 180:
            azim -= 360
        self._azimuth = azim
        self.view_changed()


    @property
    def roll(self):
        """Get the camera roll angle in degrees."""
        return self._roll
    @roll.setter
    def roll(self, roll):
        roll = float(roll)
        while roll < -180:
            roll += 360
        while roll > 180:
            roll -= 360
        self._roll = roll
        self.view_changed()


    def orbit(self, azim, elev):
        """Orbits the camera around the center position.

        Parameters
        ----------
        azim : float
            Angle in degrees to rotate horizontally around the center point.
        elev : float
            Angle in degrees to rotate vertically around the center point.
        """
        self.azimuth += azim
        self.elevation = np.clip(self.elevation + elev, -90, 90)
        self.view_changed()

    def _update_rotation(self, event):
        """Update rotation parmeters based on mouse movement"""
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        if self._event_value is None:
            self._event_value = self.azimuth, self.elevation
        self.azimuth = self._event_value[0] - (p2 - p1)[0] * 0.5
        self.elevation = self._event_value[1] + (p2 - p1)[1] * 0.5

    def _get_rotation_tr(self):
        """Return a rotation matrix based on camera parameters"""
        up, forward, right = self._get_dim_vectors()
        matrix = (
            transforms.rotate(self.elevation, -right)
            .dot(transforms.rotate(self.azimuth, up))
            .dot(transforms.rotate(self.roll, forward))
        )
        return matrix

    def _dist_to_trans(self, dist):
        """Convert mouse x, y movement into x, y, z translations"""
        rae = np.array([self.roll, self.azimuth, self.elevation]) * np.pi / 180
        sro, saz, sel = np.sin(rae)
        cro, caz, cel = np.cos(rae)
        d0, d1 = dist[0], dist[1]
        dx = (+ d0 * (cro * caz + sro * sel * saz)
              + d1 * (sro * caz - cro * sel * saz)) * self.translate_speed
        dy = (+ d0 * (cro * saz - sro * sel * caz)
              + d1 * (sro * saz + cro * sel * caz)) * self.translate_speed
        dz = (- d0 * sro * cel + d1 * cro * cel) * self.translate_speed
        return dx, dy, dz

