from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Sequence, cast
import numpy as np
import nptyping as ND
from nptyping import NDArray

# from vispy import scene, visuals
from vispy import scene
from vispy.scene import visuals
# from vispy.scene import Node
from vispy.scene.node import Node
from vispy.visuals.transforms import STTransform

from vispy.color import Color
from vispy.util.transforms import translate
from typing import List, Optional, Sequence, Union, Tuple

import colorsys
## vispy
import vispy
import vispy as vp
from vispy import scene
# from vispy import app, scene
# from vispy import app, gloo, visuals
# from vispy.scene.visuals import Arrow, Markers, Line
import vispy.scene.visuals as vz
from vispy.color import Colormap
from qtpy import QtWidgets, QtCore


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


from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert


# # --- Utility: Ramer-Douglas-Peucker polyline simplification (fast, pure-Python) ---
# def rdp(points: np.ndarray, eps: float) -> np.ndarray:
#     """Simplify polyline `points` using RDP algorithm. points shape: (N,2)."""
#     if eps <= 0 or len(points) < 3:
#         return points
#     # recursive implementation
#     def _split(pts):
#         if len(pts) < 3:
#             return pts
#         a, b = pts[0], pts[-1]
#         seg = b - a
#         if np.allclose(seg, 0):
#             dists = np.linalg.norm(pts - a, axis=1)
#         else:
#             t = np.clip(np.dot(pts - a, seg) / np.dot(seg, seg), 0.0, 1.0)
#             proj = a + np.outer(t, seg)
#             dists = np.linalg.norm(pts - proj, axis=1)
#         idx = np.argmax(dists)
#         maxd = dists[idx]
#         if maxd > eps:
#             left = _split(pts[:idx+1])
#             right = _split(pts[idx:])
#             return np.vstack((left[:-1], right))
#         else:
#             return np.vstack((a, b))
#     return _split(points)


# # --- Color helpers ---
# def _color_to_rgba_tuple(c: Union[str, Tuple[float,float,float], Sequence], alpha: Optional[float]=None):
#     """
#     Return (r,g,b,a) tuple scaled 0..1. Accepts vispy color strings, matplotlib colors or RGB tuples.
#     """
#     # vispy Color supports many inputs
#     try:
#         col = Color(c)
#         rgba = tuple(col.rgba)
#         if alpha is not None:
#             rgba = (rgba[0], rgba[1], rgba[2], alpha)
#         return rgba
#     except Exception:
#         pass

#     if _HAS_MPL:
#         try:
#             rgba = to_rgba(c)
#             if alpha is not None:
#                 rgba = (rgba[0], rgba[1], rgba[2], alpha)
#             return rgba
#         except Exception:
#             pass

#     # fallback: assume tuple-like
#     arr = np.asarray(c, dtype=float)
#     if arr.size >= 3:
#         r, g, b = arr[:3]
#         a = alpha if alpha is not None else (arr[3] if arr.size >= 4 else 1.0)
#         return (float(r), float(g), float(b), float(a))
#     raise ValueError(f"Could not interpret color: {c}")


# def _colormap_colors(n: int, cmap_name: str = "viridis", alpha: float = 1.0):
#     """Return n RGBA tuples from matplotlib cmap (0..1 floats)."""
#     if not _HAS_MPL:
#         raise RuntimeError("matplotlib is required when passing a cmap name. Install matplotlib.")
#     cmap = cm.get_cmap(cmap_name)
#     arr = [cmap(i / max(1, n - 1)) for i in range(n)]
#     arr = [(r, g, b, alpha) for (r, g, b, a) in arr]
#     return arr


# # --- Contour extraction wrappers ---
# def _extract_contours_from_mask(mask: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
#     """
#     Return list of Nx2 arrays (y, x) coordinates of contours for binary mask.
#     Uses skimage.measure.find_contours if available, otherwise cv2.findContours.
#     Coordinates returned are float points in image coordinate space (x=cols, y=rows).
#     """
#     if _HAS_SKIMAGE:
#         contours = measure.find_contours(mask.astype(float), level=level)
#         # skimage returns (row, col) coordinates; convert to (x, y)
#         return [np.vstack((c[:, 1], c[:, 0])).T for c in contours]
#     elif _HAS_CV2:
#         # cv2.findContours expects uint8 single-channel image
#         im = (mask > 0).astype(np.uint8) * 255
#         # OpenCV returns integer coordinates. Use RETR_LIST and CHAIN_APPROX_NONE for full fidelity
#         cnts, _ = cv2.findContours(im, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
#         out = []
#         for c in cnts:
#             pts = c.reshape(-1, 2).astype(float)
#             # cv2 coords are (x, y)
#             out.append(pts)
#         return out
#     else:
#         raise RuntimeError("No contour extraction backend available. Install scikit-image or opencv-python.")


ContourItem = Tuple[NDArray, Tuple[float, float, float, float]]


def _extract_contours_from_mask(mask: np.ndarray, level: float = 0.5) -> List[NDArray]:
    """Return list of (N, 2) arrays in (row, col) for binary mask. Requires scikit-image."""
    if not _HAS_SKIMAGE:
        raise RuntimeError("Contour extraction requires scikit-image. Install scikit-image.")
    from skimage import measure as _measure
    mask_float = np.asarray(mask, dtype=np.float64)
    contours = _measure.find_contours(mask_float, level=level)
    return [np.asarray(c, dtype=np.float64) for c in contours]


def _contour_pixel_to_world(contour_rc: NDArray, n_rows: int, n_cols: int, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> NDArray:
    """Map contour points from pixel (row, col) to world (x, y). contour_rc is (N, 2) with [:,0]=row, [:,1]=col."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    col = contour_rc[:, 1]
    row = contour_rc[:, 0]
    x_world = x_min + (col / float(n_cols)) * (x_max - x_min)
    y_world = y_min + (row / float(n_rows)) * (y_max - y_min)
    return np.column_stack([x_world, y_world]).astype(np.float32)


def _ensure_closed_pos(pos: NDArray) -> NDArray:
    """Return pos unchanged if first and last points are close; otherwise append first point to close the polygon."""
    if len(pos) < 2:
        return pos
    if np.allclose(pos[0], pos[-1], atol=1e-6):
        return pos
    return np.vstack([pos, pos[0:1]]).astype(np.float32)


def _color_to_rgba_tuple(c: Union[str, Tuple[float, float, float], Sequence], alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Return (r,g,b,a) tuple scaled 0..1. Accepts vispy color strings, matplotlib colors or RGB tuples."""
    try:
        col = Color(c)  # type: ignore[arg-type]
        rgba = tuple(col.rgba)
        if alpha is not None:
            rgba = (rgba[0], rgba[1], rgba[2], alpha)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    except Exception:
        pass
    if _HAS_MPL:
        try:
            from matplotlib.colors import to_rgba as _to_rgba
            rgba = _to_rgba(c)  # type: ignore[arg-type]
            if alpha is not None:
                rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))
            else:
                rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]) if len(rgba) > 3 else 1.0)
            return rgba
        except Exception:
            pass
    arr = np.asarray(c, dtype=float)
    if arr.size >= 3:
        r, g, b = arr[0], arr[1], arr[2]
        a = alpha if alpha is not None else (float(arr[3]) if arr.size >= 4 else 1.0)
        return (float(r), float(g), float(b), float(a))
    raise ValueError(f"Could not interpret color: {c}")


def _colormap_colors(n: int, cmap_name: str = "viridis", alpha: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """Return n RGBA tuples from matplotlib cmap (0..1 floats)."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required when passing a cmap name. Install matplotlib.")
    from matplotlib import cm as _cm
    cmap_obj = _cm.get_cmap(cmap_name)
    arr = [cmap_obj(i / max(1, n - 1)) for i in range(n)]
    return [(float(r), float(g), float(b), float(alpha)) for (r, g, b, a) in arr]


def _contour_colors_for_masks(n: int, color: Optional[Union[Tuple, str]] = None, colors: Optional[Sequence] = None, cmap: Optional[str] = None, cmap_alpha: float = 0.7) -> List[Tuple[float, float, float, float]]:
    """Return list of n RGBA tuples. One of color, colors, or cmap must be provided."""
    if color is not None:
        rgba = _color_to_rgba_tuple(color)
        return [rgba] * n
    if colors is not None:
        return [_color_to_rgba_tuple(c) for c in colors]
    if cmap is not None:
        return _colormap_colors(n, cmap_name=cmap, alpha=cmap_alpha)
    return _colormap_colors(n, cmap_name="viridis", alpha=cmap_alpha)


def contours_from_masks(masks: Union[Sequence[NDArray], NDArray], x_bounds: Tuple[float, float] = (0.0, 1.0), y_bounds: Tuple[float, float] = (0.0, 1.0), color: Optional[Union[Tuple, str]] = None, colors: Optional[Sequence] = None, cmap: Optional[str] = None, cmap_alpha: float = 0.7, level: float = 0.5, return_per_mask: bool = False) -> Union[List[ContourItem], List[List[ContourItem]]]:
    """Return flat list of (pos, rgba) or, if return_per_mask=True, one list per mask index. masks: list of 2D binary arrays or 3D (n_rows, n_cols, n_masks)."""
    arr = np.asarray(masks)
    if arr.ndim == 3:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            n_masks, n_rows, n_cols = arr.shape[0], arr.shape[1], arr.shape[2]
            mask_list = [arr[i, :, :] for i in range(n_masks)]
        else:
            n_rows, n_cols, n_masks = arr.shape[0], arr.shape[1], arr.shape[2]
            mask_list = [arr[:, :, i] for i in range(n_masks)]
    else:
        mask_list = list(masks)
        if not mask_list:
            return [] if not return_per_mask else []
        n_rows, n_cols = np.asarray(mask_list[0]).shape[:2]
    n_masks = len(mask_list)
    color_list = _contour_colors_for_masks(n_masks, color=color, colors=colors, cmap=cmap, cmap_alpha=cmap_alpha)
    out: List[ContourItem] = []
    per_mask_out: List[List[ContourItem]] = [[] for _ in range(n_masks)]
    for idx, mask in enumerate(mask_list):
        m = np.asarray(mask, dtype=np.float64)
        nr, nc = m.shape[0], m.shape[1]
        if not np.any(m):
            continue
        contour_arrays = _extract_contours_from_mask(m, level=level)
        rgba = color_list[idx]
        for contour_rc in contour_arrays:
            pos = _contour_pixel_to_world(contour_rc, nr, nc, x_bounds, y_bounds)
            item = (pos, rgba)
            out.append(item)
            per_mask_out[idx].append(item)
    return per_mask_out if return_per_mask else out


def create_contour_line_visuals(contour_data: List[Tuple[NDArray, Tuple]], parent: Node, line_width: float = 2.0, order: int = 10, fill: bool = False, fill_alpha: Optional[float] = 0.3) -> Tuple[List, List]:
    """Create vispy Line visuals from contour data and attach to parent. When fill=True, adds translucent Polygon fill (same RGB as line) behind each contour. Returns (lines, polygons) so callers can clear both on epoch change; polygons is empty when fill=False."""
    lines: List = []
    polygons: List = []
    alpha = fill_alpha if fill_alpha is not None else 0.3
    for pos, rgba in contour_data:
        if fill and len(pos) >= 3:
            pos_closed = _ensure_closed_pos(pos)
            fill_rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))
            polygon = vz.Polygon(pos=pos_closed, color=fill_rgba, border_width=0, parent=parent)  # type: ignore[call-arg]
            polygon.order = order - 1
            polygons.append(polygon)
        line = vz.Line(pos=pos, color=rgba, width=line_width, parent=parent)  # type: ignore[call-arg]
        line.order = order
        lines.append(line)
    return (lines, polygons)


@metadata_attributes(short_name=None, tags=['VispyHelpers', 'vispy'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-04 11:28', related_items=[])
class VispyHelpers:
    """ helpers for vispy
    
    from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import VispyHelpers
    
    """
    


    @classmethod
    def render_contours(cls, masks: Union[Sequence[NDArray], NDArray], x_bounds: Tuple[float, float] = (0.0, 1.0), y_bounds: Tuple[float, float] = (0.0, 1.0), color: Optional[Union[Tuple, str]] = None, colors: Optional[Sequence] = None, cmap: Optional[str] = None, cmap_alpha: float = 0.7, level: float = 0.5, parents: Optional[Sequence[Node]] = None, line_width: float = 2.0, order: int = 10, fill: bool = False, fill_alpha: Optional[float] = 0.3) -> Union[List[ContourItem], Tuple[List[ContourItem], List[List[Any]]]]:
        """Decoupled contour rendering: takes masks and optional bounds/color/cmap; returns contour data and optionally creates Line (and optional Polygon fill) visuals on given parents."""
        contour_data = cast(List[ContourItem], contours_from_masks(masks, x_bounds=x_bounds, y_bounds=y_bounds, color=color, colors=colors, cmap=cmap, cmap_alpha=cmap_alpha, level=level, return_per_mask=False))
        if parents is None:
            return contour_data
        line_lists = [create_contour_line_visuals(contour_data, parent, line_width=line_width, order=order, fill=fill, fill_alpha=fill_alpha)[0] for parent in parents]
        return (contour_data, line_lists)
    





# ==================================================================================================================================================================================================================================================================================== #
# Examples                                                                                                                                                                                                                                                                             #
# ==================================================================================================================================================================================================================================================================================== #
if __name__ == '__main__':

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


    from vispy import app

    masks_list = make_random_gaussian_masks(n_masks=5, shape=(40, 60), seed=42)
    contour_data = cast(List[ContourItem], contours_from_masks(masks_list, cmap='viridis'))
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    scene_parent = view.scene
    if scene_parent is not None:
        _lines, _polygons = create_contour_line_visuals(contour_data, scene_parent, line_width=2.0, order=10, fill=True, fill_alpha=0.3)
    app.run()
