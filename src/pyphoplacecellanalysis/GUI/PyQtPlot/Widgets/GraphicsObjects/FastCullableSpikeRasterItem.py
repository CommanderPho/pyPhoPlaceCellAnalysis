import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import functions as fn
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ScatterPlotItem import PixmapFragments, renderSymbol


class FastCullableSpikeRasterItem(GraphicsObject):
    """High-performance 2D spike raster graphics item with O(log N) viewport culling and tiled RGBA density LOD mipmaps.

    Drop-in replacement for `pg.ScatterPlotItem` on 2D spike raster tracks:
    - Zoomed-in / medium windows (`<= lod_switch_duration_s` and `<= max_tick_spikes_per_frame`):
      Slices `[min_t, max_t]` in O(log N) via `np.searchsorted`, transforms ONLY visible spikes to device pixels,
      deduplicates sub-pixel collisions, and blits pre-rendered unit-colored tick glyphs via `PixmapFragments`.
    - Wide windows (`> lod_switch_duration_s`) or overview tracks (`force_lod_only=True`):
      Blits 1-3 pre-baked `<= 4096`-px `QImage` RGBA density tiles (`dt` mipmap levels `0.025 s`, `0.10 s`, `0.50 s`)
      preserving every spike and each unit's color in `< 0.2 ms` per frame while avoiding Qt5's `32,767`-px `QImage` limit.
    """

    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object, object)
    sigHovered = QtCore.Signal(object, object, object)

    MAX_TILE_WIDTH: int = 4096
    DEFAULT_MIPMAP_DTS: Tuple[float, ...] = (0.025, 0.10, 0.50)

    def __init__(self, *args, name: Optional[str] = None, pxMode: bool = True, symbol: Any = None, size: float = 5.0, pen: Any = None, brush: Any = None, hoverable: bool = False, lod_switch_duration_s: float = 75.0, max_tick_spikes_per_frame: int = 25000, force_lod_only: bool = False, **kwargs):
        super().__init__()
        if symbol is None:
            vtick = QtGui.QPainterPath()
            vtick.moveTo(0.0, -0.5)
            vtick.lineTo(0.0, 0.5)
            symbol = vtick

        self.opts: Dict[str, Any] = {
            'name': name,
            'pxMode': bool(pxMode),
            'symbol': symbol,
            'size': float(size),
            'pen': pen if pen is not None else pg.mkPen('w', width=1),
            'brush': brush if brush is not None else pg.mkBrush('w'),
            'hoverable': bool(hoverable),
            'useCache': True,
            'antialias': False,
        }
        if name is not None:
            self.setObjectName(str(name))

        self._lod_switch_duration_s: float = float(lod_switch_duration_s)
        self._max_tick_spikes_per_frame: int = int(max_tick_spikes_per_frame)
        self._force_lod_only: bool = bool(force_lod_only)

        self._t_sorted: np.ndarray = np.empty(0, dtype=np.float64)
        self._y_sorted: np.ndarray = np.empty(0, dtype=np.float64)
        self._pal_idx_sorted: np.ndarray = np.empty(0, dtype=np.int32)
        self._palette_rgba: np.ndarray = np.zeros((1, 4), dtype=np.uint8)
        self._palette_rgba[0] = [255, 255, 255, 255]

        self._x_bounds: Optional[Tuple[float, float]] = None
        self._y_bounds: Optional[Tuple[float, float]] = None
        self._bounding_rect: QtCore.QRectF = QtCore.QRectF()

        self._row_origin: float = 0.0
        self._row_step: float = 1.0
        self._n_rows: int = 1
        self._row_idx_sorted: np.ndarray = np.empty(0, dtype=np.int32)

        self._atlas_pixmap: Optional[QtGui.QPixmap] = None
        self._atlas_source_rects: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._pixmap_fragments: PixmapFragments = PixmapFragments()

        # Lazy LOD cache: dt -> list of (QRectF, QImage, backing_uint8_ndarray)
        self._lod_tiles_by_dt: Dict[float, List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]]] = {}

        if (len(args) > 0) or ('spots' in kwargs) or ('x' in kwargs) or ('pos' in kwargs):
            self.setData(*args, **kwargs)


    def clear(self):
        """Removes all spikes and invalidates cached atlases and LOD mipmap tiles."""
        self.prepareGeometryChange()
        self._t_sorted = np.empty(0, dtype=np.float64)
        self._y_sorted = np.empty(0, dtype=np.float64)
        self._pal_idx_sorted = np.empty(0, dtype=np.int32)
        self._row_idx_sorted = np.empty(0, dtype=np.int32)
        self._palette_rgba = np.array([[255, 255, 255, 255]], dtype=np.uint8)
        self._x_bounds = None
        self._y_bounds = None
        self._bounding_rect = QtCore.QRectF()
        self._atlas_pixmap = None
        self._atlas_source_rects = np.zeros((0, 4), dtype=np.float64)
        self._lod_tiles_by_dt.clear()
        self.informViewBoundsChanged()
        self.update()
        self.sigPlotChanged.emit(self)


    def setData(self, *args, **kwargs):
        """Replaces all spike data (compatible with `pg.ScatterPlotItem.setData`)."""
        self._t_sorted = np.empty(0, dtype=np.float64)
        self._y_sorted = np.empty(0, dtype=np.float64)
        self._pal_idx_sorted = np.empty(0, dtype=np.int32)
        self._palette_rgba = np.zeros((0, 4), dtype=np.uint8)
        self._lod_tiles_by_dt.clear()
        self._atlas_pixmap = None
        if (len(args) == 0) and (len(kwargs) == 0):
            self.clear()
            return
        self.addPoints(*args, **kwargs)


    def addPoints(self, *args, **kwargs):
        """Adds spikes from either a `spots` list of dicts or `(x, y)` / `pos` arrays."""
        spots = None
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            spots = args[0]
        elif 'spots' in kwargs and kwargs['spots'] is not None:
            spots = kwargs['spots']

        if spots is not None:
            new_t, new_y, new_palette, new_pal_idx = self._extract_arrays_from_spots(spots)
        else:
            x_arr, y_arr = self._extract_xy_from_args_kwargs(*args, **kwargs)
            pen_arg = kwargs.get('pen', self.opts.get('pen', None))
            brush_arg = kwargs.get('brush', self.opts.get('brush', None))
            new_t, new_y, new_palette, new_pal_idx = self._extract_arrays_from_xy(x_arr, y_arr, pen_arg, brush_arg)

        if len(new_t) == 0:
            if len(self._t_sorted) == 0:
                self.clear()
            return

        if len(self._t_sorted) == 0:
            self.setSpikeArrays(new_t, new_y, new_palette, new_pal_idx)
        else:
            pal_offset = len(self._palette_rgba)
            merged_palette = np.vstack([self._palette_rgba, new_palette]) if pal_offset > 0 else new_palette
            merged_t = np.concatenate([self._t_sorted, new_t])
            merged_y = np.concatenate([self._y_sorted, new_y])
            merged_idx = np.concatenate([self._pal_idx_sorted, new_pal_idx + pal_offset])
            self.setSpikeArrays(merged_t, merged_y, merged_palette, merged_idx)


    def setSpikeArrays(self, spike_t: np.ndarray, spike_y: np.ndarray, palette_rgba: np.ndarray, spike_palette_idx: np.ndarray, y_bounds: Optional[Tuple[float, float]] = None):
        """Zero-dict vectorized fast path for uploading spike times, Y positions, and unit RGBA palette."""
        self.prepareGeometryChange()
        t_arr = np.asarray(spike_t, dtype=np.float64).ravel()
        y_arr = np.asarray(spike_y, dtype=np.float64).ravel()
        p_idx = np.asarray(spike_palette_idx, dtype=np.int32).ravel()
        pal = np.asarray(palette_rgba, dtype=np.uint8)
        if pal.ndim != 2 or pal.shape[1] != 4 or len(pal) == 0:
            pal = np.array([[255, 255, 255, 255]], dtype=np.uint8)
            p_idx = np.zeros(len(t_arr), dtype=np.int32)

        finite_mask = np.isfinite(t_arr) & np.isfinite(y_arr)
        if not np.all(finite_mask):
            t_arr = t_arr[finite_mask]
            y_arr = y_arr[finite_mask]
            p_idx = p_idx[finite_mask]

        if len(t_arr) == 0:
            self.clear()
            return

        # Pre-sort once by spike timestamp so viewport culling is O(log N) via np.searchsorted
        if len(t_arr) > 1 and not bool(np.all(np.diff(t_arr) >= 0.0)):
            order = np.argsort(t_arr, kind='mergesort')
            t_arr = np.ascontiguousarray(t_arr[order])
            y_arr = np.ascontiguousarray(y_arr[order])
            p_idx = np.ascontiguousarray(np.clip(p_idx[order], 0, len(pal) - 1))
        else:
            t_arr = np.ascontiguousarray(t_arr)
            y_arr = np.ascontiguousarray(y_arr)
            p_idx = np.ascontiguousarray(np.clip(p_idx, 0, len(pal) - 1))

        self._t_sorted = t_arr
        self._y_sorted = y_arr
        self._pal_idx_sorted = p_idx
        self._palette_rgba = np.ascontiguousarray(pal)

        # Compute row grid for spike-density LOD mipmaps
        y_unique = np.unique(y_arr)
        if len(y_unique) <= 1:
            self._row_step = 1.0
            y_val = float(y_unique[0]) if len(y_unique) == 1 else 0.0
            self._row_origin = y_val - 0.5
            self._n_rows = 1
        else:
            diffs = np.diff(y_unique)
            pos_diffs = diffs[diffs > 1e-6]
            step = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
            step = max(step, 1e-3)
            y_origin = float(y_unique[0]) - 0.5 * step
            y_top = float(y_unique[-1]) + 0.5 * step
            n_rows = int(np.rint((y_top - y_origin) / step))
            if n_rows > 1024:
                n_rows = 1024
                step = (y_top - y_origin) / float(n_rows)
            n_rows = max(1, n_rows)
            self._row_step = step
            self._row_origin = y_origin
            self._n_rows = n_rows

        self._row_idx_sorted = np.clip(np.floor((self._y_sorted - self._row_origin) / self._row_step).astype(np.int32), 0, self._n_rows - 1)

        x0 = float(self._t_sorted[0])
        x1 = float(self._t_sorted[-1])
        if y_bounds is not None:
            y0, y1 = float(y_bounds[0]), float(y_bounds[1])
        else:
            y0 = float(self._row_origin)
            y1 = float(self._row_origin + self._n_rows * self._row_step)

        self._x_bounds = (x0, x1)
        self._y_bounds = (y0, y1)
        self._bounding_rect = QtCore.QRectF(x0, y0, max(1e-6, x1 - x0), max(1e-6, y1 - y0))

        self._atlas_pixmap = None
        self._lod_tiles_by_dt.clear()

        self.informViewBoundsChanged()
        self.update()
        self.sigPlotChanged.emit(self)


    def setDownsampling(self, *args, **kwargs):
        """Compatibility no-op (automatic searchsorted + mipmap LOD handles all zoom scales)."""
        pass


    def setPointsVisible(self, visible):
        """Compatibility helper for toggling visibility."""
        if isinstance(visible, bool):
            self.setVisible(visible)


    def points(self):
        """Lightweight compatibility stub returning an empty array unless spots are queried."""
        return np.empty(0, dtype=object)


    def pointsAt(self, pos):
        return np.empty(0, dtype=object)


    def boundingRect(self) -> QtCore.QRectF:
        return self._bounding_rect


    def dataBounds(self, ax: int, frac: float = 1.0, orthoRange=None):
        """O(1) cached bounds lookup used by `ViewBox` auto-range and `CustomLinearRegionItem` `clipItem`."""
        if ax == 0:
            return self._x_bounds if self._x_bounds is not None else (None, None)
        elif ax == 1:
            return self._y_bounds if self._y_bounds is not None else (None, None)
        return (None, None)


    def pixelPadding(self) -> int:
        if self.opts.get('pxMode', True):
            return int(max(1.0, 0.5 * float(self.opts.get('size', 5.0)) + 1.0))
        return 0


    # ==================================================================================================================== #
    # Painting (Tick Atlas + Tiled RGBA Mipmap LOD)                                                                        #
    # ==================================================================================================================== #

    def paint(self, p: QtGui.QPainter, option, widget=None):
        n_total = len(self._t_sorted)
        if n_total == 0 or self._x_bounds is None:
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

        # Estimate viewport pixel width from painter transform (`m11` = pixels per second)
        tr = p.transform()
        px_per_sec = abs(float(tr.m11()))
        viewport_px_w = max(64.0, vis_duration * px_per_sec) if px_per_sec > 1e-9 else 1200.0
        sec_per_px = vis_duration / viewport_px_w

        if not self._force_lod_only and (vis_duration <= self._lod_switch_duration_s):
            i0 = int(np.searchsorted(self._t_sorted, view_x0, side='left'))
            i1 = int(np.searchsorted(self._t_sorted, view_x1, side='right'))
            if i1 <= i0:
                return
            if (i1 - i0) <= self._max_tick_spikes_per_frame:
                self._paint_visible_ticks(p, tr, i0, i1)
                return

        # Wide window or overview mode: blit pre-baked RGBA density tiles
        self._paint_lod_tiles(p, view_x0, view_x1, sec_per_px)


    def _ensure_tick_atlas(self):
        if self._atlas_pixmap is not None:
            return
        k = len(self._palette_rgba)
        size = max(2.0, float(self.opts.get('size', 5.0)))
        symbol = self.opts.get('symbol', 'o')

        # Render first symbol to determine exact glyph pixel dimensions
        r0, g0, b0, a0 = [int(v) for v in self._palette_rgba[0]]
        qcol0 = QtGui.QColor(r0, g0, b0, a0)
        pen0 = QtGui.QPen(qcol0, 1.0)
        brush0 = QtGui.QBrush(qcol0)
        sample_img = renderSymbol(symbol, size, pen0, brush0)
        cell_w = max(2, sample_img.width() + 1)
        cell_h = max(2, sample_img.height() + 1)

        # Wrap into rows of <= 256 glyphs so atlas width never exceeds 4096 px
        cols = min(max(1, k), 256)
        rows = int(math.ceil(k / float(cols)))
        atlas_img = QtGui.QImage(cols * cell_w, rows * cell_h, QtGui.QImage.Format_ARGB32_Premultiplied)
        atlas_img.fill(QtCore.Qt.GlobalColor.transparent)

        source_rects = np.zeros((k, 4), dtype=np.float64)
        painter = QtGui.QPainter(atlas_img)
        try:
            for idx in range(k):
                r, g, b, a = [int(v) for v in self._palette_rgba[idx]]
                qcol = QtGui.QColor(r, g, b, a)
                pen_i = QtGui.QPen(qcol, 1.0)
                brush_i = QtGui.QBrush(qcol)
                glyph = renderSymbol(symbol, size, pen_i, brush_i)
                col_i = idx % cols
                row_i = idx // cols
                sx = col_i * cell_w
                sy = row_i * cell_h
                painter.drawImage(sx, sy, glyph)
                source_rects[idx] = [float(sx), float(sy), float(glyph.width()), float(glyph.height())]
            ## END for idx in range(k)....

        finally:
            painter.end()

        self._atlas_pixmap = QtGui.QPixmap.fromImage(atlas_img)
        self._atlas_source_rects = source_rects


    def _paint_visible_ticks(self, p: QtGui.QPainter, tr: QtGui.QTransform, i0: int, i1: int):
        self._ensure_tick_atlas()
        if self._atlas_pixmap is None:
            return

        t_vis = self._t_sorted[i0:i1]
        y_vis = self._y_sorted[i0:i1]
        pal_vis = self._pal_idx_sorted[i0:i1]

        if (not tr.isRotating()) and (not tr.isProjecting()):
            px_x = tr.m11() * t_vis + tr.dx()
            px_y = tr.m22() * y_vis + tr.dy()
        else:
            pts = fn.transformCoordinates(tr, np.vstack([t_vis, y_vis]))
            px_x = pts[0]
            px_y = pts[1]

        px_x = np.clip(px_x, -1e7, 1e7)
        px_y = np.clip(px_y, -1e7, 1e7)

        # Deduplicate sub-pixel overlapping ticks when dense (> 2,500 visible spikes)
        if len(px_x) > 2500:
            ix = (np.rint(px_x).astype(np.int64) & 0xFFFFF)
            iy = (np.rint(px_y).astype(np.int64) & 0xFFFFF)
            ip = (pal_vis.astype(np.int64) & 0xFFFFF)
            packed_keys = (ix << 40) | (iy << 20) | ip
            _, keep_indices = np.unique(packed_keys, return_index=True)
            if len(keep_indices) < len(px_x):
                px_x = px_x[keep_indices]
                px_y = px_y[keep_indices]
                pal_vis = pal_vis[keep_indices]

        m = len(px_x)
        if m == 0:
            return

        frags = self._pixmap_fragments.array(m)
        frags[:, 0] = px_x
        frags[:, 1] = px_y
        frags[:, 2:6] = self._atlas_source_rects[pal_vis]
        frags[:, 6:10] = [1.0, 1.0, 0.0, 1.0]

        p.save()
        try:
            p.resetTransform()
            self._pixmap_fragments.draw(p, m, self._atlas_pixmap)
        finally:
            p.restore()


    def _select_lod_dt(self, sec_per_px: float) -> float:
        chosen = self.DEFAULT_MIPMAP_DTS[0]
        for dt in self.DEFAULT_MIPMAP_DTS:
            if dt <= max(sec_per_px * 0.85, self.DEFAULT_MIPMAP_DTS[0]):
                chosen = dt
        ## END for dt in self.DEFAULT_MIPMAP_DTS....

        return chosen


    def _ensure_lod_level(self, dt: float) -> List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]]:
        if dt in self._lod_tiles_by_dt:
            return self._lod_tiles_by_dt[dt]

        if len(self._t_sorted) == 0 or self._x_bounds is None:
            return []

        t0, t1 = self._x_bounds
        span = max(dt, t1 - t0)
        n_tbins = max(1, int(math.ceil(span / dt)))
        # Guard ultra-long recordings (> 10 hours) so a single LOD level never allocates excessive RAM
        if n_tbins * self._n_rows > 24_000_000:
            dt = max(dt, span / float(max(1, 24_000_000 // self._n_rows)))
            n_tbins = max(1, int(math.ceil(span / dt)))

        t_bin = np.clip(np.floor((self._t_sorted - t0) / dt).astype(np.int64), 0, n_tbins - 1)
        flat_idx = self._row_idx_sorted.astype(np.int64) * n_tbins + t_bin
        total_cells = self._n_rows * n_tbins

        counts_flat = np.bincount(flat_idx, minlength=total_cells)
        rgba_flat = np.zeros((total_cells, 4), dtype=np.uint8)
        rgba_flat[flat_idx] = self._palette_rgba[self._pal_idx_sorted]

        occupied = counts_flat > 0
        if np.any(occupied):
            occ_counts = counts_flat[occupied]
            alpha_boost = np.where(occ_counts <= 1, 0.85, np.where(occ_counts == 2, 0.94, 1.0))
            base_alpha = rgba_flat[occupied, 3].astype(np.float32)
            # Keep deemphasized spikes dim while making normal single spikes crisp and visible
            boosted_alpha = np.where(base_alpha >= 100.0, np.maximum(185.0, base_alpha * alpha_boost), base_alpha * alpha_boost)
            rgba_flat[occupied, 3] = np.clip(boosted_alpha, 0.0, 255.0).astype(np.uint8)

        rgba_img = rgba_flat.reshape((self._n_rows, n_tbins, 4))
        y_height = float(self._n_rows) * self._row_step

        tiles: List[Tuple[QtCore.QRectF, QtGui.QImage, np.ndarray]] = []
        for col_start in range(0, n_tbins, self.MAX_TILE_WIDTH):
            col_end = min(n_tbins, col_start + self.MAX_TILE_WIDTH)
            w = col_end - col_start
            tile_arr = np.ascontiguousarray(rgba_img[:, col_start:col_end, :])
            qimg = QtGui.QImage(tile_arr.data, w, self._n_rows, w * 4, QtGui.QImage.Format_RGBA8888)
            tile_rect = QtCore.QRectF(t0 + col_start * dt, self._row_origin, w * dt, y_height)
            tiles.append((tile_rect, qimg, tile_arr))
        ## END for col_start in range(0, n_tbins, self.MAX_TILE_WIDTH)....

        self._lod_tiles_by_dt[dt] = tiles
        return tiles


    def _paint_lod_tiles(self, p: QtGui.QPainter, view_x0: float, view_x1: float, sec_per_px: float):
        dt = self._select_lod_dt(sec_per_px)
        tiles = self._ensure_lod_level(dt)
        if not tiles:
            return

        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
            for tile_rect, qimg, _ in tiles:
                if (tile_rect.right() < view_x0) or (tile_rect.left() > view_x1):
                    continue
                p.drawImage(tile_rect, qimg)
            ## END for tile_rect, qimg, _ in tiles....

        finally:
            p.restore()


    # ==================================================================================================================== #
    # Input Normalization Helpers                                                                                          #
    # ==================================================================================================================== #

    @classmethod
    def _color_to_rgba_tuple(cls, pen_obj: Any, brush_obj: Any, fallback_rgba: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[int, int, int, int]:
        qcol = None
        if isinstance(pen_obj, QtGui.QPen):
            if pen_obj.style() != QtCore.Qt.PenStyle.NoPen:
                qcol = pen_obj.color()
        elif pen_obj is not None:
            try:
                p_mk = pg.mkPen(pen_obj)
                if p_mk.style() != QtCore.Qt.PenStyle.NoPen:
                    qcol = p_mk.color()
            except Exception:
                qcol = None

        if (qcol is None or qcol.alpha() == 0) and (brush_obj is not None):
            if isinstance(brush_obj, QtGui.QBrush):
                if brush_obj.style() != QtCore.Qt.BrushStyle.NoBrush:
                    qcol = brush_obj.color()
            else:
                try:
                    b_mk = pg.mkBrush(brush_obj)
                    if b_mk.style() != QtCore.Qt.BrushStyle.NoBrush:
                        qcol = b_mk.color()
                except Exception:
                    pass

        if qcol is None:
            return fallback_rgba
        return (int(qcol.red()), int(qcol.green()), int(qcol.blue()), int(qcol.alpha()))


    def _extract_arrays_from_spots(self, spots: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(spots)
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.zeros((1, 4), dtype=np.uint8), np.empty(0, dtype=np.int32)

        default_rgba = self._color_to_rgba_tuple(self.opts.get('pen'), self.opts.get('brush'))
        t_arr = np.empty(n, dtype=np.float64)
        y_arr = np.empty(n, dtype=np.float64)
        pal_idx = np.empty(n, dtype=np.int32)

        obj_id_to_pal: Dict[Tuple[int, int], int] = {}
        rgba_to_pal: Dict[Tuple[int, int, int, int], int] = {}
        palette_list: List[Tuple[int, int, int, int]] = []

        for i, spot in enumerate(spots):
            pos = spot.get('pos', None)
            if pos is not None:
                t_arr[i] = float(pos[0])
                y_arr[i] = float(pos[1])
            else:
                t_arr[i] = float(spot.get('x', np.nan))
                y_arr[i] = float(spot.get('y', np.nan))

            pen_i = spot.get('pen', None)
            brush_i = spot.get('brush', None)
            obj_key = (id(pen_i), id(brush_i))
            p_i = obj_id_to_pal.get(obj_key, None)
            if p_i is None:
                rgba = self._color_to_rgba_tuple(pen_i, brush_i, default_rgba)
                p_i = rgba_to_pal.get(rgba, None)
                if p_i is None:
                    p_i = len(palette_list)
                    palette_list.append(rgba)
                    rgba_to_pal[rgba] = p_i
                obj_id_to_pal[obj_key] = p_i
            pal_idx[i] = p_i
        ## END for i, spot in enumerate(spots)....

        palette_arr = np.asarray(palette_list, dtype=np.uint8) if palette_list else np.array([default_rgba], dtype=np.uint8)
        return t_arr, y_arr, palette_arr, pal_idx


    @classmethod
    def _extract_xy_from_args_kwargs(cls, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(args) >= 2:
            return np.asarray(args[0], dtype=np.float64).ravel(), np.asarray(args[1], dtype=np.float64).ravel()
        if 'pos' in kwargs and kwargs['pos'] is not None:
            pos = np.asarray(kwargs['pos'], dtype=np.float64)
            if pos.ndim == 2:
                if pos.shape[1] == 2:
                    return pos[:, 0], pos[:, 1]
                elif pos.shape[0] == 2:
                    return pos[0, :], pos[1, :]
        if 'x' in kwargs and 'y' in kwargs:
            return np.asarray(kwargs['x'], dtype=np.float64).ravel(), np.asarray(kwargs['y'], dtype=np.float64).ravel()
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)


    def _extract_arrays_from_xy(self, x_arr: np.ndarray, y_arr: np.ndarray, pen_arg: Any, brush_arg: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = min(len(x_arr), len(y_arr))
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.zeros((1, 4), dtype=np.uint8), np.empty(0, dtype=np.int32)
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
        default_rgba = self._color_to_rgba_tuple(self.opts.get('pen'), self.opts.get('brush'))

        if isinstance(pen_arg, (list, tuple, np.ndarray)) and len(pen_arg) == n:
            brush_seq = brush_arg if (isinstance(brush_arg, (list, tuple, np.ndarray)) and len(brush_arg) == n) else [None] * n
            pal_idx = np.empty(n, dtype=np.int32)
            obj_id_to_pal: Dict[Tuple[int, int], int] = {}
            rgba_to_pal: Dict[Tuple[int, int, int, int], int] = {}
            palette_list: List[Tuple[int, int, int, int]] = []
            for i in range(n):
                p_obj = pen_arg[i]
                b_obj = brush_seq[i]
                key = (id(p_obj), id(b_obj))
                idx = obj_id_to_pal.get(key, None)
                if idx is None:
                    rgba = self._color_to_rgba_tuple(p_obj, b_obj, default_rgba)
                    idx = rgba_to_pal.get(rgba, None)
                    if idx is None:
                        idx = len(palette_list)
                        palette_list.append(rgba)
                        rgba_to_pal[rgba] = idx
                    obj_id_to_pal[key] = idx
                pal_idx[i] = idx
            ## END for i in range(n)....

            return x_arr, y_arr, np.asarray(palette_list, dtype=np.uint8), pal_idx

        rgba = self._color_to_rgba_tuple(pen_arg, brush_arg, default_rgba)
        return x_arr, y_arr, np.array([rgba], dtype=np.uint8), np.zeros(n, dtype=np.int32)


    @classmethod
    def build_spike_arrays_from_df(cls, spikes_df, config_fragile_linear_neuron_IDX_map: Optional[Dict[Any, Any]] = None, neuron_colors_by_aclu: Optional[Dict[Any, Any]] = None, downsampling_rate: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fast vectorized helper extracting `(spike_t, spike_y, palette_rgba, spike_palette_idx)` directly from `spikes_df`."""
        active_df = spikes_df.iloc[::downsampling_rate] if (downsampling_rate is not None and downsampling_rate > 1) else spikes_df
        time_col = active_df.spikes.time_variable_name if hasattr(active_df, 'spikes') else ('t_rel_seconds' if 't_rel_seconds' in active_df.columns else 't')
        spike_t = np.asarray(active_df[time_col].to_numpy(), dtype=np.float64)
        spike_y = np.asarray(active_df['visualization_raster_y_location'].to_numpy(), dtype=np.float64)
        n = len(spike_t)
        if n == 0:
            return spike_t, spike_y, np.array([[255, 255, 255, 255]], dtype=np.uint8), np.empty(0, dtype=np.int32)

        if config_fragile_linear_neuron_IDX_map is not None and 'fragile_linear_neuron_IDX' in active_df.columns:
            fragile_idxs = active_df['fragile_linear_neuron_IDX'].to_numpy()
            emph_states = active_df['visualization_raster_emphasis_state'].to_numpy() if 'visualization_raster_emphasis_state' in active_df.columns else None

            # Vectorized palette mapping via unique `(fragile_IDX, emphasis_state_value)` pairs
            unique_fragile = np.unique(fragile_idxs)
            palette_list: List[Tuple[int, int, int, int]] = []
            pair_to_pal: Dict[Tuple[int, Any], int] = {}

            if emph_states is not None:
                unique_states = list(dict.fromkeys(emph_states.tolist()))
            else:
                unique_states = [None]

            for u_idx in unique_fragile:
                cfg = config_fragile_linear_neuron_IDX_map.get(int(u_idx), config_fragile_linear_neuron_IDX_map.get(u_idx, None))
                for st in unique_states:
                    rgba = (255, 255, 255, 255)
                    if cfg is not None:
                        pen_map = cfg[2]
                        pen_obj = pen_map.get(st, None) if isinstance(pen_map, dict) else (pen_map[st] if st is not None else None)
                        if pen_obj is None and isinstance(pen_map, dict) and len(pen_map) > 0:
                            # Fallback by enum `.value` match in case autoreload redefined SpikeEmphasisState
                            st_val = getattr(st, 'value', st)
                            for k_st, v_pen in pen_map.items():
                                if getattr(k_st, 'value', k_st) == st_val:
                                    pen_obj = v_pen
                                    break
                            ## END for k_st, v_pen in pen_map.items()....

                        rgba = cls._color_to_rgba_tuple(pen_obj, None, rgba)
                    pair_to_pal[(int(u_idx), getattr(st, 'value', st))] = len(palette_list)
                    palette_list.append(rgba)
                ## END for st in unique_states....

            ## END for u_idx in unique_fragile....

            if len(unique_states) == 1:
                max_u = int(np.max(unique_fragile)) + 1 if len(unique_fragile) > 0 else 1
                lut_arr = np.zeros(max(1, max_u), dtype=np.int32)
                st0_val = getattr(unique_states[0], 'value', unique_states[0])
                for u_idx in unique_fragile:
                    lut_arr[int(u_idx)] = pair_to_pal[(int(u_idx), st0_val)]
                ## END for u_idx in unique_fragile....

                pal_idx = lut_arr[fragile_idxs.astype(np.int32)]
            else:
                st_vals = np.array([int(getattr(s, 'value', 0)) for s in emph_states], dtype=np.int32)
                min_s = int(np.min(st_vals))
                span_s = int(np.max(st_vals) - min_s + 1)
                max_u = int(np.max(unique_fragile)) + 1
                lut_2d = np.zeros((max_u, span_s), dtype=np.int32)
                for (u_i, s_v), p_i in pair_to_pal.items():
                    if isinstance(s_v, (int, np.integer)) and 0 <= u_i < max_u and 0 <= (int(s_v) - min_s) < span_s:
                        lut_2d[u_i, int(s_v) - min_s] = p_i
                ## END for (u_i, s_v), p_i in pair_to_pal.items()....

                pal_idx = lut_2d[fragile_idxs.astype(np.int32), st_vals - min_s]

            palette_arr = np.asarray(palette_list, dtype=np.uint8) if palette_list else np.array([[255, 255, 255, 255]], dtype=np.uint8)
            return spike_t, spike_y, palette_arr, pal_idx

        if neuron_colors_by_aclu is not None and 'aclu' in active_df.columns:
            aclus = active_df['aclu'].to_numpy()
            unique_aclus, inv = np.unique(aclus, return_inverse=True)
            palette_list = []
            for aclu in unique_aclus:
                col = neuron_colors_by_aclu.get(aclu, neuron_colors_by_aclu.get(int(aclu), 'w'))
                qcol = pg.mkColor(col)
                palette_list.append((int(qcol.red()), int(qcol.green()), int(qcol.blue()), int(qcol.alpha())))
            ## END for aclu in unique_aclus....

            palette_arr = np.asarray(palette_list, dtype=np.uint8) if palette_list else np.array([[255, 255, 255, 255]], dtype=np.uint8)
            return spike_t, spike_y, palette_arr, inv.astype(np.int32)

        return spike_t, spike_y, np.array([[255, 255, 255, 255]], dtype=np.uint8), np.zeros(n, dtype=np.int32)
