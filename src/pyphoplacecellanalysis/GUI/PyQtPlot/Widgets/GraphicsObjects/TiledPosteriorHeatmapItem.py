import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject


class TiledPosteriorHeatmapItem(GraphicsObject):
    """Pre-baked tiled RGBA heatmap item with 1x/4x/16x max-pooled temporal mipmaps for 0 ms PyQtGraph scrolling.

    Designed specifically for docked `ContinuousDecode_*` and `marginal_*` posterior tracks:
    - Evaluates the exact Matplotlib colormap (`viridis`, `Oranges`, custom colormaps, including `bad`/`NaN` -> black,
      `under` -> white, `over` -> red) ONCE into a 259-entry `uint8` RGBA LUT on track creation.
    - Builds `1x`, `4x`, and `16x` max-pooled (`np.fmax`, NaN-safe, zero warnings) RGBA mipmaps sliced along time into
      `<= 4096`-px `QImage` tiles (staying well within Qt5's `32,767`-px `QImage` limit).
    - Blits only the `1-3` tiles intersecting the current `ViewBox` `[min_t, max_t]` via `QPainter.drawImage` (`< 0.15 ms`).
    - Optionally renders horizontal bin separator lines and viewport-pinned left-edge `y_bin_labels` (e.g. `['long', 'short']`)
      directly during `paint()` with zero signal overhead during `setXLink` scrolling.
    """

    MAX_TILE_WIDTH: int = 4096

    def __init__(self, posterior: Optional[np.ndarray] = None, time_window_centers: Optional[np.ndarray] = None, xbin: Optional[np.ndarray] = None, cmap: Any = 'viridis', vmin: float = 0.0, vmax: float = 1.0, bad_color: str = 'black', under_color: str = 'white', over_color: str = 'red', use_original_bounds: bool = False, y_bin_labels: Optional[Sequence[str]] = None):
        super().__init__()
        self._levels_tiles: Dict[int, List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]]] = {}
        self._x_bounds: Optional[Tuple[float, float]] = None
        self._y_bounds: Optional[Tuple[float, float]] = None
        self._bounding_rect: QtCore.QRectF = QtCore.QRectF()
        self._base_dt: float = 0.025
        self._y_bin_labels: Optional[List[str]] = list(y_bin_labels) if y_bin_labels is not None else None
        self._enable_separator_lines: bool = y_bin_labels is not None

        self._sep_pen: QtGui.QPen = QtGui.QPen(QtGui.QColor(0, 0, 0, 190), 1.2)
        self._sep_pen.setCosmetic(True)
        self._label_font: QtGui.QFont = QtGui.QFont('Sans Serif', 9)
        self._label_font.setBold(True)
        self._label_text_pen: QtGui.QPen = QtGui.QPen(QtGui.QColor(245, 245, 245, 235))
        self._label_bg_brush: QtGui.QBrush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 145))
        self._paint_count: int = 0

        if (posterior is not None) and (time_window_centers is not None):
            self.setPosteriorData(
                posterior=posterior,
                time_window_centers=time_window_centers,
                xbin=xbin,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                bad_color=bad_color,
                under_color=under_color,
                over_color=over_color,
                use_original_bounds=use_original_bounds,
                y_bin_labels=y_bin_labels,
            )


    @classmethod
    def compute_extent(cls, time_window_centers: np.ndarray, xbin: np.ndarray, n_xbins: int, use_original_bounds: bool = False) -> Tuple[float, float, float, float, float]:
        """Matches `DecoderPredictionError._subfn_try_get_approximate_recovered_t_pos` extent `(xmin, xmax, ymin, ymax, dt)`."""
        centers = np.asarray(time_window_centers, dtype=np.float64).ravel()
        if len(centers) >= 2:
            diffs = np.diff(centers)
            finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            dt = float(np.median(finite_diffs)) if len(finite_diffs) > 0 else 0.025
        else:
            dt = 0.025
        dt = max(dt, 1e-6)

        if len(centers) == 0:
            xmin, xmax = 0.0, dt
        elif use_original_bounds and len(centers) >= 2:
            xmin, xmax = float(centers[0]), float(centers[-1])
        else:
            xmin = float(centers[0] - 0.5 * dt)
            xmax = float(centers[-1] + 0.5 * dt)

        if xmax <= xmin:
            xmax = xmin + dt

        if xbin is not None and len(xbin) >= 2:
            xb = np.asarray(xbin, dtype=np.float64).ravel()
            ymin, ymax = float(xb[0]), float(xb[-1])
        else:
            ymin, ymax = 0.0, float(max(1, n_xbins))

        if abs(ymax - ymin) < 1e-9:
            ymin -= 0.5
            ymax += 0.5
        elif ymax < ymin:
            ymin, ymax = ymax, ymin

        return xmin, xmax, ymin, ymax, dt


    @classmethod
    def build_rgba_lut_259(cls, cmap: Any = 'viridis', bad_color: str = 'black', under_color: str = 'white', over_color: str = 'red') -> np.ndarray:
        """Builds a `(259, 4)` `uint8` RGBA table: `[0..255]` ramp, `[256]` under, `[257]` over, `[258]` bad/NaN."""
        resolved_cmap = cmap
        if resolved_cmap is None:
            resolved_cmap = 'Oranges'
        if isinstance(resolved_cmap, str):
            try:
                from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
                resolved_cmap = get_heatmap_cmap(cmap=resolved_cmap, bad_color=bad_color, under_color=under_color, over_color=over_color)
            except Exception:
                import matplotlib as mpl
                resolved_cmap = mpl.colormaps[resolved_cmap].copy()
                resolved_cmap.set_bad(bad_color)
                resolved_cmap.set_under(under_color)
                resolved_cmap.set_over(over_color)

        lut = np.zeros((259, 4), dtype=np.uint8)
        lut[0:256] = np.asarray(resolved_cmap(np.linspace(0.0, 1.0, 256), bytes=True), dtype=np.uint8)
        special_rgba = np.asarray(resolved_cmap(np.array([-1.0, 2.0, np.nan]), bytes=True), dtype=np.uint8)
        lut[256:259] = special_rgba[0:3]
        return np.ascontiguousarray(lut)


    @classmethod
    def _map_float_matrix_to_rgba(cls, mat_f32: np.ndarray, lut_u32: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Maps `(n_xbins, n_tbins)` float32 matrix to `(n_xbins, n_tbins, 4)` `uint8` RGBA via 32-bit LUT gather."""
        h, w = mat_f32.shape
        span = float(vmax - vmin)
        inv_span = 255.0 / span if abs(span) > 1e-12 else 0.0
        nan_mask = ~np.isfinite(mat_f32)

        with np.errstate(invalid='ignore'):
            scaled = (mat_f32 - float(vmin)) * inv_span
            idx = np.clip(scaled, 0.0, 255.0).astype(np.uint16)
            idx[mat_f32 < vmin] = 256
            idx[mat_f32 > vmax] = 257
        idx[nan_mask] = 258

        rgba_u32 = lut_u32[idx]
        return rgba_u32.view(np.uint8).reshape((h, w, 4))


    @classmethod
    def _max_pool_time_4x(cls, mat_f32: np.ndarray) -> np.ndarray:
        """NaN-safe 4x temporal max-pool using `np.fmax` (preserves narrow replay peaks without All-NaN warnings)."""
        h, w = mat_f32.shape
        rem = w % 4
        if rem != 0:
            pad_cols = 4 - rem
            padded = np.pad(mat_f32, ((0, 0), (0, pad_cols)), mode='edge')
        else:
            padded = mat_f32
        m = padded.shape[1] // 4
        blk = padded.reshape((h, m, 4))
        return np.fmax(np.fmax(blk[:, :, 0], blk[:, :, 1]), np.fmax(blk[:, :, 2], blk[:, :, 3]))


    @classmethod
    def _slice_rgba_into_tiles(cls, rgba: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]]:
        h, w, _ = rgba.shape
        if w == 0 or h == 0:
            return []
        col_w = (xmax - xmin) / float(w)
        y_height = ymax - ymin
        tiles: List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]] = []
        for c0 in range(0, w, cls.MAX_TILE_WIDTH):
            c1 = min(w, c0 + cls.MAX_TILE_WIDTH)
            tw = c1 - c0
            tile_arr = np.ascontiguousarray(rgba[:, c0:c1, :])
            qimg = QtGui.QImage(tile_arr.data, tw, h, tw * 4, QtGui.QImage.Format_RGBA8888).copy()
            rect = QtCore.QRectF(xmin + c0 * col_w, ymin, tw * col_w, y_height)
            tiles.append((rect, qimg, tile_arr))
        ## END for c0 in range(0, w, cls.MAX_TILE_WIDTH)....

        return tiles


    def setPosteriorData(self, posterior: np.ndarray, time_window_centers: np.ndarray, xbin: Optional[np.ndarray] = None, cmap: Any = 'viridis', vmin: float = 0.0, vmax: float = 1.0, bad_color: str = 'black', under_color: str = 'white', over_color: str = 'red', use_original_bounds: bool = False, y_bin_labels: Optional[Sequence[str]] = None):
        """Pre-bakes 1x/4x/16x RGBA tiles for a continuous `(n_xbins, n_tbins)` posterior matrix."""
        self.prepareGeometryChange()
        self._levels_tiles.clear()
        if y_bin_labels is not None:
            self._y_bin_labels = list(y_bin_labels)
            self._enable_separator_lines = True

        mat = np.asarray(posterior, dtype=np.float32)
        if mat.ndim != 2 or mat.size == 0:
            self._x_bounds = None
            self._y_bounds = None
            self._bounding_rect = QtCore.QRectF()
            self.informViewBoundsChanged()
            self.update()
            return

        n_xbins, n_tbins = mat.shape
        xmin, xmax, ymin, ymax, dt = self.compute_extent(time_window_centers, xbin, n_xbins=n_xbins, use_original_bounds=use_original_bounds)
        self._base_dt = max(1e-6, (xmax - xmin) / float(max(1, n_tbins)))

        lut_u8 = self.build_rgba_lut_259(cmap=cmap, bad_color=bad_color, under_color=under_color, over_color=over_color)
        lut_u32 = np.ascontiguousarray(lut_u8).view(np.uint32).ravel()

        rgba_1x = self._map_float_matrix_to_rgba(mat, lut_u32, vmin=vmin, vmax=vmax)
        self._levels_tiles[1] = self._slice_rgba_into_tiles(rgba_1x, xmin, xmax, ymin, ymax)

        if n_tbins >= 512:
            mat_4x = self._max_pool_time_4x(mat)
            rgba_4x = self._map_float_matrix_to_rgba(mat_4x, lut_u32, vmin=vmin, vmax=vmax)
            self._levels_tiles[4] = self._slice_rgba_into_tiles(rgba_4x, xmin, xmax, ymin, ymax)

            if n_tbins >= 2048:
                mat_16x = self._max_pool_time_4x(mat_4x)
                rgba_16x = self._map_float_matrix_to_rgba(mat_16x, lut_u32, vmin=vmin, vmax=vmax)
                self._levels_tiles[16] = self._slice_rgba_into_tiles(rgba_16x, xmin, xmax, ymin, ymax)

        self._x_bounds = (xmin, xmax)
        self._y_bounds = (ymin, ymax)
        self._bounding_rect = QtCore.QRectF(xmin, ymin, max(1e-6, xmax - xmin), max(1e-6, ymax - ymin))
        self.informViewBoundsChanged()
        self.update()


    def setPosteriorSlices(self, slices_posteriors: Sequence[np.ndarray], slices_time_window_centers: Sequence[np.ndarray], xbin: Optional[np.ndarray] = None, cmap: Any = 'viridis', vmin: float = 0.0, vmax: float = 1.0, bad_color: str = 'black', under_color: str = 'white', over_color: str = 'red', use_original_bounds: bool = True, y_bin_labels: Optional[Sequence[str]] = None):
        """Pre-bakes 1x RGBA tiles for a list of epoch-sliced posteriors."""
        self.prepareGeometryChange()
        self._levels_tiles.clear()
        if y_bin_labels is not None:
            self._y_bin_labels = list(y_bin_labels)
            self._enable_separator_lines = True

        lut_u8 = self.build_rgba_lut_259(cmap=cmap, bad_color=bad_color, under_color=under_color, over_color=over_color)
        lut_u32 = np.ascontiguousarray(lut_u8).view(np.uint32).ravel()

        tiles_1x: List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]] = []
        global_xmin = np.inf
        global_xmax = -np.inf
        global_ymin = np.inf
        global_ymax = -np.inf

        for post_i, centers_i in zip(slices_posteriors, slices_time_window_centers):
            if post_i is None or centers_i is None or len(centers_i) == 0:
                continue
            mat_i = np.asarray(post_i, dtype=np.float32)
            if mat_i.ndim != 2 or mat_i.size == 0:
                continue
            xmin, xmax, ymin, ymax, _ = self.compute_extent(centers_i, xbin, n_xbins=mat_i.shape[0], use_original_bounds=use_original_bounds)
            rgba_i = self._map_float_matrix_to_rgba(mat_i, lut_u32, vmin=vmin, vmax=vmax)
            tiles_1x.extend(self._slice_rgba_into_tiles(rgba_i, xmin, xmax, ymin, ymax))
            global_xmin = min(global_xmin, xmin)
            global_xmax = max(global_xmax, xmax)
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)
        ## END for post_i, centers_i in zip(slices_posteriors, slices_time_window_centers)....

        if not tiles_1x or not np.isfinite(global_xmin):
            self._x_bounds = None
            self._y_bounds = None
            self._bounding_rect = QtCore.QRectF()
        else:
            self._levels_tiles[1] = tiles_1x
            self._x_bounds = (float(global_xmin), float(global_xmax))
            self._y_bounds = (float(global_ymin), float(global_ymax))
            self._bounding_rect = QtCore.QRectF(global_xmin, global_ymin, max(1e-6, global_xmax - global_xmin), max(1e-6, global_ymax - global_ymin))

        self.informViewBoundsChanged()
        self.update()


    def setYBinLabels(self, y_bin_labels: Optional[Sequence[str]], enable_separator_lines: bool = True):
        self._y_bin_labels = list(y_bin_labels) if y_bin_labels is not None else None
        self._enable_separator_lines = bool(enable_separator_lines)
        self.update()


    def boundingRect(self) -> QtCore.QRectF:
        return self._bounding_rect


    def dataBounds(self, ax: int, frac: float = 1.0, orthoRange=None):
        if ax == 0:
            return self._x_bounds if self._x_bounds is not None else (None, None)
        elif ax == 1:
            return self._y_bounds if self._y_bounds is not None else (None, None)
        return (None, None)


    def paint(self, p: QtGui.QPainter, option, widget=None):
        self._paint_count += 1
        if not self._levels_tiles or self._x_bounds is None or self._y_bounds is None:
            return

        vb = self.getViewBox()
        view_rect = vb.viewRect() if vb is not None else self.viewRect()
        if (view_rect is None) or (view_rect.width() <= 0.0):
            view_x0, view_x1 = self._x_bounds
        else:
            view_x0 = float(view_rect.left())
            view_x1 = float(view_rect.right())
            if view_x1 < view_x0:
                view_x0, view_x1 = view_x1, view_x0

        vis_duration = max(1e-9, view_x1 - view_x0)
        tr = p.transform()
        px_per_sec = abs(float(tr.m11()))
        viewport_px_w = max(64.0, vis_duration * px_per_sec) if px_per_sec > 1e-9 else 1200.0
        bins_per_px = (vis_duration / self._base_dt) / viewport_px_w

        if (bins_per_px >= 12.0) and (16 in self._levels_tiles):
            active_tiles = self._levels_tiles[16]
        elif (bins_per_px >= 3.0) and (4 in self._levels_tiles):
            active_tiles = self._levels_tiles[4]
        else:
            active_tiles = self._levels_tiles[1]

        # 1. Blit only visible pre-baked RGBA QImage tiles
        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
            if len(active_tiles) > 8:
                # Binary search first overlapping tile when many tiles exist
                tile_rights = [t[0].right() for t in active_tiles]
                import bisect
                start_idx = max(0, bisect.bisect_left(tile_rights, view_x0))
                for idx in range(start_idx, len(active_tiles)):
                    tile_rect, qimg, _ = active_tiles[idx]
                    if tile_rect.left() > view_x1:
                        break
                    p.drawImage(tile_rect, qimg)
                ## END for idx in range(start_idx, len(active_tiles))....

            else:
                for tile_rect, qimg, _ in active_tiles:
                    if (tile_rect.right() < view_x0) or (tile_rect.left() > view_x1):
                        continue
                    p.drawImage(tile_rect, qimg)
                ## END for tile_rect, qimg, _ in active_tiles....

        finally:
            p.restore()

        # 2. Optional separator lines & viewport-pinned left-edge marginal labels
        if self._y_bin_labels and len(self._y_bin_labels) > 0:
            n_ybins = len(self._y_bin_labels)
            ymin, ymax = self._y_bounds
            y_span = ymax - ymin
            bin_h = y_span / float(n_ybins)

            if self._enable_separator_lines and n_ybins > 1:
                p.save()
                try:
                    p.setPen(self._sep_pen)
                    for k in range(1, n_ybins):
                        y_sep = ymin + k * bin_h
                        p.drawLine(QtCore.QPointF(view_x0, y_sep), QtCore.QPointF(view_x1, y_sep))
                    ## END for k in range(1, n_ybins)....

                finally:
                    p.restore()

            p.save()
            try:
                p.resetTransform()
                p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
                p.setFont(self._label_font)
                fm = QtGui.QFontMetrics(self._label_font)
                for i, lbl_text in enumerate(self._y_bin_labels):
                    y_center = ymin + (i + 0.5) * bin_h
                    pt_dev = tr.map(QtCore.QPointF(view_x0, y_center))
                    tw = fm.horizontalAdvance(str(lbl_text))
                    th = fm.height()
                    rect_bg = QtCore.QRectF(pt_dev.x() + 5.0, pt_dev.y() - 0.5 * th, tw + 8.0, th)
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(self._label_bg_brush)
                    p.drawRoundedRect(rect_bg, 3.0, 3.0)
                    p.setPen(self._label_text_pen)
                    p.drawText(rect_bg, int(QtCore.Qt.AlignmentFlag.AlignCenter), str(lbl_text))
                ## END for i, lbl_text in enumerate(self._y_bin_labels)....

            finally:
                p.restore()
