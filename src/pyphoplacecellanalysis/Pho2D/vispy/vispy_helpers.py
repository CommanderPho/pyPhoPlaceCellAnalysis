import numpy as np
# from vispy import scene, visuals
from vispy import scene
from vispy.scene import visuals
# from vispy.scene import Node
from vispy.scene.node import Node
from vispy.visuals.transforms import STTransform

from vispy.color import Color
from vispy.util.transforms import translate
from typing import List, Optional, Sequence, Union, Tuple

# Optional dependencies for contour extraction
try:
    from skimage import measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Optional Matplotlib colormap support
try:
    import matplotlib
    from matplotlib import cm
    from matplotlib.colors import to_rgba
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# --- Utility: Ramer-Douglas-Peucker polyline simplification (fast, pure-Python) ---
def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Simplify polyline `points` using RDP algorithm. points shape: (N,2)."""
    if eps <= 0 or len(points) < 3:
        return points
    # recursive implementation
    def _split(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        seg = b - a
        if np.allclose(seg, 0):
            dists = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(np.dot(pts - a, seg) / np.dot(seg, seg), 0.0, 1.0)
            proj = a + np.outer(t, seg)
            dists = np.linalg.norm(pts - proj, axis=1)
        idx = np.argmax(dists)
        maxd = dists[idx]
        if maxd > eps:
            left = _split(pts[:idx+1])
            right = _split(pts[idx:])
            return np.vstack((left[:-1], right))
        else:
            return np.vstack((a, b))
    return _split(points)


# --- Color helpers ---
def _color_to_rgba_tuple(c: Union[str, Tuple[float,float,float], Sequence], alpha: Optional[float]=None):
    """
    Return (r,g,b,a) tuple scaled 0..1. Accepts vispy color strings, matplotlib colors or RGB tuples.
    """
    # vispy Color supports many inputs
    try:
        col = Color(c)
        rgba = tuple(col.rgba)
        if alpha is not None:
            rgba = (rgba[0], rgba[1], rgba[2], alpha)
        return rgba
    except Exception:
        pass

    if _HAS_MPL:
        try:
            rgba = to_rgba(c)
            if alpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], alpha)
            return rgba
        except Exception:
            pass

    # fallback: assume tuple-like
    arr = np.asarray(c, dtype=float)
    if arr.size >= 3:
        r, g, b = arr[:3]
        a = alpha if alpha is not None else (arr[3] if arr.size >= 4 else 1.0)
        return (float(r), float(g), float(b), float(a))
    raise ValueError(f"Could not interpret color: {c}")


def _colormap_colors(n: int, cmap_name: str = "viridis", alpha: float = 1.0):
    """Return n RGBA tuples from matplotlib cmap (0..1 floats)."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required when passing a cmap name. Install matplotlib.")
    cmap = cm.get_cmap(cmap_name)
    arr = [cmap(i / max(1, n - 1)) for i in range(n)]
    arr = [(r, g, b, alpha) for (r, g, b, a) in arr]
    return arr


# --- Contour extraction wrappers ---
def _extract_contours_from_mask(mask: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    """
    Return list of Nx2 arrays (y, x) coordinates of contours for binary mask.
    Uses skimage.measure.find_contours if available, otherwise cv2.findContours.
    Coordinates returned are float points in image coordinate space (x=cols, y=rows).
    """
    if _HAS_SKIMAGE:
        contours = measure.find_contours(mask.astype(float), level=level)
        # skimage returns (row, col) coordinates; convert to (x, y)
        return [np.vstack((c[:, 1], c[:, 0])).T for c in contours]
    elif _HAS_CV2:
        # cv2.findContours expects uint8 single-channel image
        im = (mask > 0).astype(np.uint8) * 255
        # OpenCV returns integer coordinates. Use RETR_LIST and CHAIN_APPROX_NONE for full fidelity
        cnts, _ = cv2.findContours(im, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cnts:
            pts = c.reshape(-1, 2).astype(float)
            # cv2 coords are (x, y)
            out.append(pts)
        return out
    else:
        raise RuntimeError("No contour extraction backend available. Install scikit-image or opencv-python.")


# --- Main renderer ---
# class MaskContourRenderer(scene.Node):
class MaskContourRenderer(Node):
    """
    A fast VisPy renderer for lists of binary masks that draws filled contours (low alpha)
    and outline curves. Instantiate attached to any existing scene (e.g. view.scene).

    Example:
    
        from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import MaskContourRenderer
        renderer = MaskContourRenderer(parent=view.scene, fill_alpha=0.2, contour_alpha=0.9)
        renderer.update_masks(list_of_masks, colors=['red','blue'])
    """
    def __init__(self, parent=None, fill_alpha: float = 0.2, contour_alpha: float = 0.9, contour_width: float = 1.0, simplify_eps: float = 0.8, z: float = 0.0, transform=None):
        # Install a transform system BEFORE parenting
        # self.transforms = TransformSystem()

        super().__init__(parent=parent)

        self._fill_alpha = float(fill_alpha)
        self._contour_alpha = float(contour_alpha)
        self._contour_width = float(contour_width)
        self._simplify_eps = float(simplify_eps)
        self._z = float(z)

        # assign a simple transform so your visual children
        # inherit scene coordinates
        self.transform = STTransform()        

        if transform is not None:
            self.transform = transform
        else:
            self.transform = STTransform()

        self._visuals = []

        
    def clear(self):
        """Remove all visuals previously added."""
        for v in self._visuals:
            v.parent = None
        self._visuals.clear()



    def set_colors(self, colors: Optional[Sequence[Union[str, Tuple]]]=None, cmap: Optional[str]=None):
        """Provide explicit colors or a cmap name to generate colors later."""
        if colors is not None:
            # convert to rgba tuples without applying alpha yet
            self._colors = [_color_to_rgba_tuple(c, alpha=None) for c in colors]
        elif cmap is not None:
            if not _HAS_MPL:
                raise RuntimeError("matplotlib is required to use cmap names.")
            self._colors = _colormap_colors(1, cmap_name=cmap, alpha=1.0)  # will expand later
            self._cmap_name = cmap
        else:
            self._colors = None
            self._cmap_name = None

    def update_masks(self, masks: Sequence[np.ndarray], colors: Optional[Sequence[Union[str, Tuple]]] = None, cmap: Optional[str] = None, simplify_eps: Optional[float] = None, z: Optional[float] = None): 
        """
        Render a list of binary masks.

        masks: sequence of 2D boolean/0-1 arrays (ny, nx).
        colors: optional explicit colors (one per mask) -- overrides any previously set colors.
        cmap: optional matplotlib colormap name if matplotlib is available.
        simplify_eps: epsilon for vertex simplification (larger -> fewer points). If None uses object's default.
        z: optional z-order for visuals (useful when layering).
        """
        if simplify_eps is None:
            simplify_eps = self._simplify_eps
        else:
            simplify_eps = float(simplify_eps)

        if z is not None:
            self._z = float(z)

        # prepare colors
        n = len(masks)
        if colors is not None:
            self._colors = [_color_to_rgba_tuple(c, alpha=None) for c in colors]
        elif cmap is not None:
            if not _HAS_MPL:
                raise RuntimeError("matplotlib is required to use cmap names.")
            self._colors = _colormap_colors(n, cmap_name=cmap, alpha=1.0)
        elif self._colors is None:
            # default: use viridis
            if _HAS_MPL:
                self._colors = _colormap_colors(n, cmap_name="viridis", alpha=1.0)
            else:
                # simple fallback palette
                fallback = [(0.9, 0.3, 0.3, 1.0), (0.3, 0.6, 0.9, 1.0), (0.3, 0.9, 0.5, 1.0)]
                self._colors = [fallback[i % len(fallback)] for i in range(n)]
        # ensure length matches n, expand if necessary
        if len(self._colors) < n:
            if _HAS_MPL and hasattr(self, "_cmap_name"):
                self._colors = _colormap_colors(n, cmap_name=self._cmap_name, alpha=1.0)
            else:
                # repeat last color
                last = self._colors[-1]
                extra = [last] * (n - len(self._colors))
                self._colors = list(self._colors) + extra

        # remove previous visuals
        self.clear()

        # iterate masks and render contours
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            # ensure boolean
            arr = np.asarray(mask)
            if arr.ndim != 2:
                raise ValueError(f"Each mask must be 2D; got shape {arr.shape}")
            if arr.size == 0:
                continue

            # extract contours as sequences of (x,y)
            contours = _extract_contours_from_mask(arr, level=0.5)
            if not contours:
                continue

            color_base = self._colors[i]
            fill_rgba = (color_base[0], color_base[1], color_base[2], self._fill_alpha)
            stroke_rgba = (color_base[0], color_base[1], color_base[2], self._contour_alpha)

            for contour_pts in contours:
                if contour_pts.shape[0] < 3:
                    continue
                # Optionally simplify to reduce vertex count
                if simplify_eps and contour_pts.shape[0] > 200:
                    contour_pts = rdp(contour_pts, simplify_eps)

                # create a filled polygon visual (use vispy.scene.visuals.Polygon if available)
                try:
                    poly = visuals.Polygon(np.asarray(contour_pts, dtype=np.float32),
                                           color=fill_rgba,
                                           border_color=stroke_rgba,
                                           border_width=self._contour_width,
                                           parent=self)
                    # set a z-position if requested
                    if self._z != 0.0:
                        poly.transform = scene.transforms.STTransform(translate=(0, 0, self._z))
                    self._visuals.append(poly)
                except Exception:
                    # fallback if Polygon visual not available: draw a Mesh (triangulate)
                    # quick fallback triangulation using simple fan from centroid (works for convex-ish polygons)
                    pts = np.asarray(contour_pts, dtype=np.float32)
                    centroid = pts.mean(axis=0)
                    # build triangles (centroid, i, i+1)
                    npts = len(pts)
                    verts = np.vstack((centroid, pts))
                    tris = []
                    for vi in range(1, npts):
                        tris.append((0, vi, vi+1 if vi+1 <= npts-1 else 1))
                    tris = np.asarray(tris, dtype=np.uint32)
                    mesh = visuals.Mesh(vertices=verts.astype(np.float32), faces=tris, color=fill_rgba, parent=self)
                    line = visuals.Line(pts, color=stroke_rgba, width=self._contour_width, method='gl', parent=self)
                    self._visuals.extend([mesh, line])
                    continue

                # also create a line stroke (separate visual gives more control)
                line = visuals.Line(np.asarray(contour_pts, dtype=np.float32),
                                    color=stroke_rgba,
                                    width=self._contour_width,
                                    method='gl',
                                    connect='strip',
                                    parent=self)
                self._visuals.append(line)

    # convenience
    def update_and_center(self, masks: Sequence[np.ndarray], **kwargs):
        """
        Update masks and center the node's transform to fit the masks bounding box.
        Useful when placing into a 2D scene where you want the data to fit.
        """
        self.update_masks(masks, **kwargs)
        # compute bounding box over masks (find first non-empty mask)
        minx = np.inf; miny = np.inf; maxx = -np.inf; maxy = -np.inf
        any_nonempty = False
        for m in masks:
            if m is None:
                continue
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue
            any_nonempty = True
            minx = min(minx, xs.min()); maxx = max(maxx, xs.max())
            miny = min(miny, ys.min()); maxy = max(maxy, ys.max())
        if not any_nonempty:
            return
        # set transform to place mask in 0..width coordinates (no-op unless you want scaling)
        w = maxx - minx + 1
        h = maxy - miny + 1
        # center: translate by -minx, -miny so mask origin at (0,0)
        self.transform = scene.transforms.STTransform(translate=(-minx, -miny, 0.0))



if __name__ == '__main__':

    # from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import MaskContourRenderer
    
    def make_random_gaussian_masks(
        n_masks: int = 5,
        shape: tuple = (40, 60),
        n_spots_range=(1, 4),
        sigma_range=(2.0, 6.0),
        threshold: float = 0.5,
        seed: int = 0,
    ):
        """
        Generate binary masks containing random Gaussian spots.

        Returns
        -------
        masks : list[np.ndarray]
            List of (ny, nx) boolean masks
        """
        rng = np.random.default_rng(seed)
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        masks = []

        for _ in range(n_masks):
            img = np.zeros((ny, nx), dtype=np.float32)
            n_spots = rng.integers(n_spots_range[0], n_spots_range[1] + 1)

            for _ in range(n_spots):
                cx = rng.uniform(0, nx)
                cy = rng.uniform(0, ny)
                sigma = rng.uniform(*sigma_range)
                amp = rng.uniform(0.8, 1.2)

                img += amp * np.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
                )

            # Normalize per-mask then threshold → binary
            img /= img.max() + 1e-9
            mask = img > threshold
            masks.append(mask)

        return masks


    # --- Example usage ---
    masks_list = make_random_gaussian_masks(
        n_masks=5,
        shape=(40, 60),
        seed=42,
    )

    # Now pass masks_list to the renderer:
    # renderer.update_masks(masks_list, cmap="plasma")



    from vispy import app
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'  # or any 2D camera

    # suppose masks_list is a list of 2D numpy boolean arrays
    renderer = MaskContourRenderer(parent=view.scene, fill_alpha=0.2, contour_alpha=0.9, simplify_eps=1.2)
    renderer.update_masks(masks_list, cmap='viridis')

    app.run()
