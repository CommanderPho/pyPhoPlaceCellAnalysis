from __future__ import annotations
from functools import partial
import time
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Sequence, cast
import numpy as np
import pandas as pd
import nptyping as ND
from nptyping import NDArray

# from vispy import scene, visuals
from vispy import scene
from vispy.scene import visuals
# from vispy.scene import Node
from vispy.scene.node import Node
from vispy.visuals.transforms import STTransform, NullTransform, MatrixTransform

from vispy.color import Color
from vispy.util.transforms import translate
from typing import List, Optional, Sequence, Union, Tuple

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
from pyphocorehelpers.plotting.heading_angle_helpers import HeadingAngleHelpers
from pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets import VispySceneTreeWidget

# from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import AngleColoredLineVisual

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


def _triangulate_polygon_2d(pos_2d: NDArray) -> Optional[NDArray]:
    """Triangulate a closed 2D polygon, returning (M, 3) int32 face indices or None on failure.

    Tries vispy's built-in ear-clipping triangulation first; falls back to a simple fan
    triangulation (works correctly for convex contours, approximate for concave ones).
    pos_2d must be (N, 2) with N >= 3. The polygon should be closed (first == last) or open;
    the function handles both cases.
    """
    pts = np.asarray(pos_2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return None
    if np.allclose(pts[0], pts[-1], atol=1e-6):
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return None
    try:
        from vispy.geometry.triangulation import triangulate as _vispy_triangulate
        vertices_out, faces_out = _vispy_triangulate(pts)
        if faces_out is not None and len(faces_out) >= 1:
            return np.asarray(faces_out, dtype=np.int32)
    except Exception:
        pass
    faces = np.column_stack([np.zeros(n - 2, dtype=np.int32), np.arange(1, n - 1, dtype=np.int32), np.arange(2, n, dtype=np.int32)])
    return faces


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


# ==================================================================================================================================================================================================================================================================================== #
# Heading Angles (HeadingAngleHelpers from pyphocorehelpers.plotting.heading_angle_helpers)                                                                                                                                                                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #


def create_contour_line_visuals(contour_data: List[Tuple[NDArray, Tuple]], parent: Node, line_width: float = 2.0, order: int = 10, fill: bool = False, fill_alpha: Optional[float] = 0.3, name: str='Contour') -> Tuple[List, List]:
    """Create vispy Line visuals from contour data and attach to parent. When fill=True, adds translucent Polygon fill (same RGB as line) behind each contour. Returns (lines, polygons) so callers can clear both on epoch change; polygons is empty when fill=False."""
    lines: List = []
    polygons: List = []
    alpha = fill_alpha if fill_alpha is not None else 0.3
    for pos, rgba in contour_data:
        if fill and len(pos) >= 3:
            pos_closed = _ensure_closed_pos(pos)
            fill_rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))
            polygon = vz.Polygon(pos=pos_closed, color=fill_rgba, border_width=0, parent=parent, name=f'{name}.Poly')  # type: ignore[call-arg]
            polygon.order = order - 1
            polygons.append(polygon)
        line = vz.Line(pos=pos, color=rgba, width=line_width, parent=parent, name=f'{name}.Border')  # type: ignore[call-arg]
        line.order = order
        lines.append(line)
    return (lines, polygons)


def _extract_trajectory_segment_positions(segments: List[pd.DataFrame], x_col: str = 'x', y_col: str = 'y') -> Tuple[List[NDArray], List[pd.DataFrame]]:
    """Extract (n_points, 2) float32 position arrays from each DataFrame. Drops NaN rows; omits segments with fewer than 2 points. Returns (pos_list, df_list) where df_list[i] is the DataFrame that produced pos_list[i]."""
    pos_list: List[NDArray] = []
    df_list: List[pd.DataFrame] = []
    for df in segments:
        x = df[x_col].values
        y = df[y_col].values
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        x = np.asarray(x[mask], dtype=np.float32)
        y = np.asarray(y[mask], dtype=np.float32)
        pos = np.column_stack([x, y]).astype(np.float32)
        if len(pos) >= 2:
            pos_list.append(pos)
            df_list.append(df)
    return (pos_list, df_list)


@metadata_attributes(short_name=None, tags=['vispy', 'scene', 'renderer', 'trajectory', 'laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-11 15:44', related_items=[])
class TrajectorySegmentsVisual(Node):
    """Renders a list of 2D trajectory segments (List[pd.DataFrame]) on a single canvas with configurable per-segment styling. Each segment is a DataFrame with at least x and y columns (configurable via x_col, y_col). When all segments share the same line_width and method, uses a single vispy Line with a connect array for one draw call; when width or method differ per segment, uses one Line per segment under this node. Styling can be global (color, line_width, method) or per-segment via colors, line_widths, or segment_style(idx, df) returning a dict with color, width, method. Callers can set set_gl_state on .line (single-Line mode) or on each item in .lines (multi-Line mode). Example: seg_visual = TrajectorySegmentsVisual(segments, parent=view.scene, colors=['r','g','b'], line_width=1.5); seg_visual.set_data(new_segments) to update."""
    def __init__(self, segments: List[pd.DataFrame], parent: Optional[Node] = None, *, x_col: str = 'x', y_col: str = 'y', color: Optional[Union[str, Tuple[float, float, float, float], Sequence]] = None, colors: Optional[Sequence] = None, line_width: float = 2.0, line_widths: Optional[Sequence[float]] = None, method: str = 'gl', segment_style: Optional[Callable[[int, pd.DataFrame], dict]] = None, order: int = 10) -> None:
        super().__init__(parent=parent)
        self._x_col = x_col
        self._y_col = y_col
        self._line_width = line_width
        self._line_widths = line_widths
        self._method = method
        self._segment_style = segment_style
        self._order = order
        self._single_line_mode: Optional[bool] = None
        self._line: Optional[Any] = None
        self._lines: List[Any] = []
        self._segments: List[pd.DataFrame] = []
        self._segment_dfs: List[pd.DataFrame] = []
        self._pos_list: List[NDArray] = []
        self._last_color = color
        self._last_colors = colors
        self._build_styles_and_visuals(segments, color=color, colors=colors)


    def _resolve_per_segment_styles(self, n: int, color: Optional[Union[str, Tuple, Sequence]] = None, colors: Optional[Sequence] = None) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[str]]:
        """Return (rgba_list, width_list, method_list) of length n. Uses segment_style callable if set; else color/colors and line_width/line_widths."""
        rgba_list: List[Tuple[float, float, float, float]] = []
        width_list: List[float] = []
        method_list: List[str] = []
        if self._segment_style is not None and self._segment_dfs:
            for i in range(n):
                df = self._segment_dfs[i] if i < len(self._segment_dfs) else pd.DataFrame()
                style = self._segment_style(i, df)
                c = style.get('color')
                rgba_list.append(_color_to_rgba_tuple(c) if c is not None else (1.0, 1.0, 1.0, 1.0))
                width_list.append(style.get('width', self._line_width))
                method_list.append(style.get('method', self._method))
            return (rgba_list, width_list, method_list)
        if colors is not None:
            rgba_list = [_color_to_rgba_tuple(c) for c in colors]
            while len(rgba_list) < n:
                rgba_list.append(rgba_list[-1] if rgba_list else (1.0, 1.0, 1.0, 1.0))
        else:
            base = _color_to_rgba_tuple(color) if color is not None else (1.0, 1.0, 1.0, 1.0)
            rgba_list = [base] * n
        if self._line_widths is not None:
            width_list = list(self._line_widths)
            while len(width_list) < n:
                width_list.append(width_list[-1] if width_list else self._line_width)
        else:
            width_list = [self._line_width] * n
        method_list = [self._method] * n
        return (rgba_list[:n], width_list[:n], method_list[:n])


    def _build_styles_and_visuals(self, segments: List[pd.DataFrame], color: Optional[Union[str, Tuple, Sequence]] = None, colors: Optional[Sequence] = None) -> None:
        self._segments = list(segments)
        self._pos_list, self._segment_dfs = _extract_trajectory_segment_positions(self._segments, self._x_col, self._y_col)
        n = len(self._pos_list)
        if n == 0:
            self._single_line_mode = False
            return
        rgba_list, width_list, method_list = self._resolve_per_segment_styles(n, color=color, colors=colors)
        uniform_width = all(w == width_list[0] for w in width_list)
        uniform_method = all(m == method_list[0] for m in method_list)
        use_single_line = uniform_width and uniform_method
        self._single_line_mode = use_single_line
        if use_single_line:
            all_pos = np.vstack(self._pos_list).astype(np.float32)
            connect_list: List[NDArray] = []
            offset = 0
            for pos in self._pos_list:
                ni = len(pos)
                for j in range(ni - 1):
                    connect_list.append(np.array([[offset + j, offset + j + 1]], dtype=np.int32))
                offset += ni
            connect = np.vstack(connect_list).astype(np.int32) if connect_list else np.empty((0, 2), dtype=np.int32)
            vertex_colors = np.zeros((len(all_pos), 4), dtype=np.float32)
            offset = 0
            for i, pos in enumerate(self._pos_list):
                r, g, b, a = rgba_list[i]
                vertex_colors[offset:offset + len(pos), :] = (r, g, b, a)
                offset += len(pos)
            self._line = vz.Line(pos=all_pos, color=vertex_colors, width=width_list[0], method=method_list[0], connect=connect, parent=self)
            self._line.order = self._order
            self._lines = []
        else:
            self._line = None
            self._lines = []
            for i, pos in enumerate(self._pos_list):
                r, g, b, a = rgba_list[i]
                line = vz.Line(pos=pos, color=(r, g, b, a), width=width_list[i], method=method_list[i], parent=self)
                line.order = self._order
                self._lines.append(line)


    def set_data(self, segments: List[pd.DataFrame]) -> None:
        """Update segments and refresh the Line(s) without creating a new visual. Preserves styling from construction."""
        self._clear_visuals()
        self._build_styles_and_visuals(segments, color=self._last_color, colors=self._last_colors)


    def _clear_visuals(self) -> None:
        if self._line is not None:
            self._line.parent = None
            self._line = None
        for line in self._lines:
            line.parent = None
        self._lines = []


    @property
    def line(self) -> Optional[Any]:
        """In single-Line mode, the single vz.Line; else None."""
        return self._line


    @property
    def lines(self) -> List[Any]:
        """In multi-Line mode, the list of vz.Line children; else empty."""
        return self._lines



# ==================================================================================================================================================================================================================================================================================== #
# VispySceneTreeWidget - A tree widget that allows interactive customization of the vispy view hiearchy                                                                                                                                                                                #
# ==================================================================================================================================================================================================================================================================================== #

def _format_transform_vector(values: Any, max_dims: int = 3, precision: int = 2) -> str:
    """Format transform vector values for compact tree display."""
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return '(?)'
    if arr.size <= 0:
        return '()'
    clipped = arr[:max_dims]
    joined = ', '.join(f'{float(v):0.{precision}f}' for v in clipped)
    return f'({joined})'


def _extract_matrix_translation(matrix_obj: Any) -> Optional[np.ndarray]:
    """Extract x/y/z translation from a 4x4 affine matrix."""
    try:
        matrix = np.asarray(matrix_obj, dtype=np.float64)
    except Exception:
        return None
    if matrix.shape != (4, 4):
        return None
    col_translation = matrix[:3, 3]
    row_translation = matrix[3, :3]
    if np.linalg.norm(col_translation) > 0.0 or np.allclose(row_translation, 0.0):
        return col_translation
    return row_translation


def render_transform_column(node: Node) -> str:
    """Default Transform-column renderer with location summary."""
    transform_obj = getattr(node, 'transform', None)
    if transform_obj is None:
        return ''
    if isinstance(transform_obj, NullTransform):
        return 'NullTransform (identity)'
    if isinstance(transform_obj, STTransform):
        translate_text = _format_transform_vector(getattr(transform_obj, 'translate', None))
        scale_text = _format_transform_vector(getattr(transform_obj, 'scale', None))
        return f'STTransform t{translate_text} s{scale_text}'
    if isinstance(transform_obj, MatrixTransform):
        matrix = getattr(transform_obj, 'matrix', None)
        if matrix is not None:
            matrix_arr = np.asarray(matrix, dtype=np.float64)
            if matrix_arr.shape == (4, 4) and np.allclose(matrix_arr, np.eye(4)):
                return 'MatrixTransform (identity)'
            translation = _extract_matrix_translation(matrix_arr)
            if translation is not None:
                return f'MatrixTransform t{_format_transform_vector(translation)}'
        return 'MatrixTransform'
    return transform_obj.__class__.__name__


class _BlendPresetDelegate(QtWidgets.QStyledItemDelegate):  # type: ignore[misc]
    """Item delegate that shows a QComboBox for the GL Blend column."""

    _BLEND_PRESETS = ('', 'opaque', 'translucent', 'additive')

    def createEditor(self, parent: Any, option: Any, index: Any) -> Any:
        combo = QtWidgets.QComboBox(parent)
        for preset in self._BLEND_PRESETS:
            combo.addItem(preset)
        return combo


    def setEditorData(self, editor: Any, index: Any) -> None:
        current_text = str(index.data() or '')
        idx = cast(Any, editor).findText(current_text)
        if idx >= 0:
            cast(Any, editor).setCurrentIndex(idx)
        else:
            cast(Any, editor).setCurrentIndex(0)


    def setModelData(self, editor: Any, model: Any, index: Any) -> None:
        model.setData(index, cast(Any, editor).currentText())





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


    @classmethod
    def create_scene_tree_widget(cls, canvas: scene.SceneCanvas, parent: Optional[Any] = None) -> VispySceneTreeWidget:
        """Create a Qt scene-tree inspector widget rooted at `canvas.scene`."""
        return VispySceneTreeWidget(root_node=canvas.scene, canvas=canvas, parent=parent)


    @classmethod
    def create_viewport_overlay_text(cls, canvas: scene.SceneCanvas, text: str = 'Overlay', color: Union[str, Tuple[float, float, float, float]] = 'white', font_size: float = 12.0, bold: bool = False, anchor_x: str = 'left', anchor_y: str = 'top', margin: Tuple[float, float] = (12.0, 12.0), order: int = 10_000, parent: Optional[Node] = None) -> Any:
        """Create viewport-fixed text in pixel space (top-left by default) that does not move with camera pan/zoom. Returns the Text visual."""
        overlay_parent = canvas.scene if parent is None else parent
        overlay_text = vz.Text(text=text, pos=(0.0, 0.0), color=color, font_size=font_size, bold=bold, anchor_x=anchor_x, anchor_y=anchor_y, parent=overlay_parent)  # type: ignore[call-arg]
        overlay_text.order = order

        def _update_overlay_text_position(event=None):
            width, height = canvas.size
            margin_x, margin_y = margin
            if anchor_x == 'left':
                x_pos = float(margin_x)
            elif anchor_x == 'center':
                x_pos = (float(width) * 0.5) + float(margin_x)
            else:
                x_pos = float(width) - float(margin_x)
            if anchor_y == 'top':
                y_pos = float(height) - float(margin_y)
            elif anchor_y == 'center':
                y_pos = (float(height) * 0.5) + float(margin_y)
            else:
                y_pos = float(margin_y)
            overlay_text.pos = np.array([x_pos, y_pos], dtype=np.float32)

        def _disconnect_overlay_resize_handler():
            try:
                canvas.events.resize.disconnect(_update_overlay_text_position)  # type: ignore[attr-defined]
            except Exception:
                pass

        canvas.events.resize.connect(_update_overlay_text_position)  # type: ignore[attr-defined]
        _update_overlay_text_position()
        overlay_text._viewport_overlay_update_position = _update_overlay_text_position
        overlay_text._viewport_overlay_disconnect = _disconnect_overlay_resize_handler
        return overlay_text
    


    @function_attributes(short_name=None, tags=['angle', 'heading', 'color', 'MAIN'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-09 10:27', related_items=[])
    @classmethod
    def create_heading_rainbow_line(cls, pos: NDArray, parent: Optional[Node] = None, headings_deg: Optional[NDArray] = None, line_width: float = 2.0, order: int = 10, alpha: float = 1.0, method: str = 'gl') -> Any:
        """Create a vispy Line colored by heading: 0°=red, ROYGBIV, 359°≈violet. If headings_deg is None, headings are computed from pos (segment directions). Returns a vispy.scene.visuals.Line.

        from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import VispyHelpers

        line, data_dict = VispyHelpers.create_heading_rainbow_line(pos=pos, parent=scene_parent, line_width=1.0, order=10)
        line.set_gl_state('translucent', depth_test=False)


        """
        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 2)
        if headings_deg is None:
            headings_deg = HeadingAngleHelpers.headings_from_positions(pos)
        else:
            headings_deg = np.asarray(headings_deg, dtype=np.float64)
            
        # colors = HeadingAngleHelpers.heading_angles_to_rainbow_colors(headings_deg, alpha=alpha)
        colors = HeadingAngleHelpers._positions_to_vertex_colors(pos)

        data_dict = dict(pos=pos, headings_deg=headings_deg, alpha=alpha, vertex_colors=colors)

        line = vz.Line(pos=pos, color=colors, width=line_width, method=method, parent=parent)  # type: ignore[call-arg]
        # line = AngleColoredLineVisual(pos=pos, color=vertex_colors, method='gl')
        line.order = order
        return line, data_dict


    # ==================================================================================================================================================================================================================================================================================== #
    # Minor Helpers                                                                                                                                                                                                                                                                        #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def build_line_pos(cls, x, y):
        N = len(x)
        assert len(y) == N
        pos = np.zeros((N, 2), dtype=np.float32)
        # Base x trajectory
        pos[:, 0] = x
        # Base downward linear trend
        pos[:, 1] = y
        return pos

    @classmethod
    def set_view_camera(cls, view, pos, padding: float = 0.05):
        xmin, xmax = np.nanmin(pos[:, 0]), np.nanmax(pos[:, 0])
        ymin, ymax = np.nanmin(pos[:, 1]), np.nanmax(pos[:, 1])

        pad_x = padding * (xmax - xmin)
        pad_y = padding * (ymax - ymin)

        view.camera.set_range(
            x=(xmin - pad_x, xmax + pad_x),
            y=(ymin - pad_y, ymax + pad_y),
        )
        return view



    @classmethod
    def generate_angular_shading_legend(cls, x_center: Tuple[float, float] = (0.0, 0.0), radius: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a 2D circle for showing the angle->color mapping 

        Returns
        -------
        pos : (N, 2) float32 ndarray
            Vertex positions
        t : (N,) float32 ndarray
            Normalized [0, 1] parameter (useful for colormaps)

        Usage:

            legend_pos, legend_headings_deg, legend_t = generate_angular_shading_legend(x_center=(0, 0), radius=20)

        """
        N = 360
        loop_center_frac: float = 0.5
        loop_width = 2.0 * radius
        pos = np.zeros((N, 2), dtype=np.float32)

        # # Base x trajectory
        # pos[:, 0] = np.linspace(x_start, x_end, N)

        # # Base downward linear trend
        # pos[:, 1] = pos[:, 0]

        # Loop placement
        loop_center_idx = int(loop_center_frac * (N - 1))
        loop_width = min(loop_width, N)
        half_width = loop_width // 2

        loop_start = int(round(max(0, loop_center_idx - half_width)))
        loop_end = int(round(min(N, loop_start + loop_width)))
        loop_width = loop_end - loop_start  # recompute in case clipped
        loop_width = int(round(loop_width, ndigits=0))

        # Parametric loop
        theta = np.linspace(0.0, (2.0 * np.pi), loop_width, endpoint=True)

        pos[loop_start:loop_end, 0] = radius * np.cos(theta)
        pos[loop_start:loop_end, 1] = radius * np.sin(theta)

        pos[loop_start:loop_end, 0] += x_center[0]
        pos[loop_start:loop_end, 1] += x_center[1] 

        # pos[loop_start:loop_end, 0] -= radius

        # Colormap parameter
        t = np.linspace(0.0, 1.0, N, dtype=np.float32)
        # headings_deg = np.linspace(0.0, 360.0, N, dtype=np.float32)
        
        headings_deg = np.linspace(0.0, 360.0, N, dtype=np.float32) - 90.0

        return pos, headings_deg, t


    @classmethod
    def generate_loop_de_loop_line(cls, N: int = 200, x_start: float = 10.0, x_end: float = 390.0, slope: float = -0.6, loop_center_frac: float = 0.5, loop_width: int = 80, loop_radius: float = 40.0, noise_scale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a 2D line that trends downward, performs a loop-de-loop,
        and then continues downward.

        Returns
        -------
        pos : (N, 2) float32 ndarray
            Vertex positions
        t : (N,) float32 ndarray
            Normalized [0, 1] parameter (useful for colormaps)

        Usage:

            pos, t = generate_loop_de_loop_line(N=300, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=2.0)

        """
        pos = np.zeros((N, 2), dtype=np.float32)

        # Base x trajectory
        pos[:, 0] = np.linspace(x_start, x_end, N)

        # Base downward linear trend
        pos[:, 1] = slope * pos[:, 0]

        # Loop placement
        loop_center_idx = int(loop_center_frac * (N - 1))
        loop_width = min(loop_width, N)
        half_width = loop_width // 2

        loop_start = max(0, loop_center_idx - half_width)
        loop_end = min(N, loop_start + loop_width)
        loop_width = loop_end - loop_start  # recompute in case clipped

        # Parametric loop
        theta = np.linspace(0.0, 2.0 * np.pi, loop_width, endpoint=True)

        pos[loop_start:loop_end, 0] += loop_radius * np.cos(theta)
        pos[loop_start:loop_end, 1] += loop_radius * np.sin(theta)

        pos[loop_start:loop_end, 0] -= loop_radius

        # Optional noise
        if noise_scale > 0.0:
            pos[:, 1] += np.random.normal(scale=noise_scale, size=N)

        # Colormap parameter
        t = np.linspace(0.0, 1.0, N, dtype=np.float32)

        return pos, t




def example_trajectory_segments_visual():
    """Example: render 2D trajectory segments from List[pd.DataFrame] with per-segment colors. Run with: python -c \"from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import example_trajectory_segments_visual; example_trajectory_segments_visual()\"."""
    from vispy import app
    t1 = np.linspace(0, 2 * np.pi, 80)
    df1 = pd.DataFrame({'x': 0.2 * np.cos(t1), 'y': 0.2 * np.sin(t1)})
    t2 = np.linspace(0, 2 * np.pi, 50)
    df2 = pd.DataFrame({'x': 0.15 * np.cos(t2) + 0.3, 'y': 0.15 * np.sin(t2)})
    df3 = pd.DataFrame({'x': np.linspace(-0.25, 0.25, 40), 'y': np.linspace(-0.2, 0.2, 40)})
    segments = [df1, df2, df3]
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    seg_visual = TrajectorySegmentsVisual(segments, parent=view.scene, colors=['red', 'green', 'blue'], line_width=2.0, order=10)
    if seg_visual.line is not None:
        seg_visual.line.set_gl_state('translucent', depth_test=False)
    else:
        for line in seg_visual.lines:
            line.set_gl_state('translucent', depth_test=False)
    VispyHelpers.set_view_camera(view, np.vstack([df1[['x', 'y']].values, df2[['x', 'y']].values, df3[['x', 'y']].values]), padding=0.15)
    app.run()


def example_viewport_overlay_text():
    """Example: draw viewport-fixed top-left text that stays in place while panning/zooming. Run with: python -c \"from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import example_viewport_overlay_text; example_viewport_overlay_text()\"."""
    from vispy import app
    t = np.linspace(0, 6 * np.pi, 600)
    x = 0.18 * t * np.cos(t)
    y = 0.18 * t * np.sin(t)
    pos = np.column_stack([x, y]).astype(np.float32)
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    line = vz.Line(pos=pos, color=(0.2, 0.8, 1.0, 1.0), width=2.0, method='gl', parent=view.scene)  # type: ignore[call-arg]
    line.order = 10
    line.set_gl_state('translucent', depth_test=False)
    VispyHelpers.set_view_camera(view, pos, padding=0.15)
    _overlay_text = VispyHelpers.create_viewport_overlay_text(canvas=canvas, text='Overlay: fixed to viewport top-left', color='white', font_size=12.0, bold=True, margin=(14.0, 14.0))
    app.run()


def example_scene_tree_widget():
    """Example: show a Qt scene-tree inspector for a vispy scene. Run with: python -c \"from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import example_scene_tree_widget; example_scene_tree_widget()\"."""
    from vispy import app
    canvas = scene.SceneCanvas(keys='interactive', size=(1100, 700), show=True, title='Vispy Scene Tree Example')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    axis = vz.XYZAxis(parent=view.scene)
    axis.order = 5
    sphere = vz.Sphere(radius=0.55, method='latitude', parent=view.scene)
    sphere.color = Color((0.4, 0.8, 1.0, 0.9)).rgba
    sphere.order = 10
    label = vz.Text(text='Sphere', pos=(0.0, 0.0, 0.75), color='white', font_size=12.0, parent=view.scene)  # type: ignore[call-arg]
    label.order = 20
    tree_widget = VispyHelpers.create_scene_tree_widget(canvas=canvas)
    tree_widget.setWindowTitle('Vispy Scene Tree')
    tree_widget.resize(700, 520)
    tree_widget.show()
    canvas._scene_tree_widget = tree_widget
    app.run()


# ==================================================================================================================================================================================================================================================================================== #
# Examples                                                                                                                                                                                                                                                                             #
# ==================================================================================================================================================================================================================================================================================== #
if __name__ == '__main__':

    def make_random_gaussian_masks(n_masks: int = 5, shape: tuple = (40, 60), n_spots_range=(1, 4), sigma_range=(2.0, 6.0), threshold: float = 0.5, seed: int = 0):
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


    def example_heading_rainbow_line():
        """Example: draw a path colored by heading (0°=red, ROYGBIV, 359°=violet). Run with: python -c \"from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import example_heading_rainbow_line; example_heading_rainbow_line()\"."""
        from vispy import app
        t = np.linspace(0, 4 * np.pi, 200)
        x = 0.3 * t * np.cos(t)
        y = 0.3 * t * np.sin(t)
        pos = np.column_stack([x, y]).astype(np.float32)
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'panzoom'
        scene_parent = view.scene
        if scene_parent is not None:
            line = create_heading_rainbow_line(pos, parent=scene_parent, line_width=3.0, order=10)
            line.set_gl_state('translucent', depth_test=False)
        app.run()




    from vispy import app

    # masks_list = make_random_gaussian_masks(n_masks=5, shape=(40, 60), seed=42)
    # contour_data = cast(List[ContourItem], contours_from_masks(masks_list, cmap='viridis'))
    # canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    # view = canvas.central_widget.add_view()
    # view.camera = 'panzoom'
    # scene_parent = view.scene
    # if scene_parent is not None:
    #     _lines, _polygons = create_contour_line_visuals(contour_data, scene_parent, line_width=2.0, order=10, fill=True, fill_alpha=0.3)
    #     # example_trajectory_segments_visual()


    # app.run()

    example_trajectory_segments_visual()