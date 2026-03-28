"""Vispy GPU-backed spike raster (multi-epoch) using shader Markers; mirrors `plot_multiple_raster_plot` data prep."""

from __future__ import annotations

from collections import namedtuple
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

from vispy import app, scene
import vispy.scene.visuals as vz
from vispy.geometry import Rect
from vispy.scene.node import Node

from qtpy.QtGui import QPen

from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlotsData
from pyphocorehelpers.function_helpers import function_attributes

from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import (
    _build_scatter_plotting_managers,
    _prepare_spikes_df_from_filter_epochs,
    build_scatter_plot_kwargs,
)
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import (
    Render2DScrollWindowPlotMixin,
)


def _ensure_spikes_df_canonical_time_t(spikes_df: pd.DataFrame) -> pd.DataFrame:
    """Rename the active time column to ``t`` and sync ``spikes`` accessor (matches `build_spikes_data_values_from_df`, avoids cross-epoch class-state mismatch)."""
    active_time_variable_name = spikes_df.spikes.time_variable_name
    if active_time_variable_name == 't':
        return spikes_df
    spikes_df = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(spikes_df, required_columns_synonym_dict={"t": {active_time_variable_name, 't_rel_seconds', 't_seconds'}})
    if active_time_variable_name in spikes_df.columns and active_time_variable_name != 't':
        spikes_df = spikes_df.drop(columns=[active_time_variable_name], inplace=False)
    spikes_df.spikes.set_time_variable_name('t')
    return spikes_df


VispyMultiRasterPlotTuple = namedtuple('VispyMultiRasterPlotTuple', ['canvas', 'plots', 'plots_data'])


def qpen_list_to_rgba_array(pens: Sequence[QPen]) -> np.ndarray:
    """Convert a sequence of QPen to (N, 4) float32 RGBA (linear 0..1)."""
    n = len(pens)
    out = np.empty((n, 4), dtype=np.float32)
    for i, p in enumerate(pens):
        c = p.color()
        _rgbf = cast(Sequence[float], c.getRgbF())
        r, g, b, a = float(_rgbf[0]), float(_rgbf[1]), float(_rgbf[2]), float(_rgbf[3])
        out[i, 0] = r
        out[i, 1] = g
        out[i, 2] = b
        out[i, 3] = a
    return out


class VispyRasterVisual(Node):
    """Single-epoch spike raster: child `Markers` visual (GPU) with vertical bar glyphs per spike.
    
    from pyphoplacecellanalysis.Pho2D.vispy.vispy_raster import VispyRasterVisual, plot_multiple_raster_plot_vispy, VispyMultiRasterPlotTuple
    
    """

    def __init__(self, parent: Optional[Node] = None, *, symbol: str = 'vbar', marker_size: float = 6.0, scaling: bool = True, edge_width: float = 0.0, order: int = 5) -> None:
        super().__init__(parent=parent)
        self._symbol = symbol
        self._marker_size = marker_size
        self._scaling = scaling
        self._edge_width = edge_width
        self._markers = vz.Markers(parent=self, symbol=symbol, size=marker_size, scaling=scaling, edge_width=edge_width)  # type: ignore[call-arg]
        self._markers.order = order
        self._markers.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))


    @property
    def markers(self) -> vz.Markers:
        return self._markers


    def set_spike_arrays(self, t: np.ndarray, y: np.ndarray, rgba: np.ndarray, *, symbol: Optional[str] = None, marker_size: Optional[float] = None, edge_width: Optional[float] = None) -> None:
        """Update all spike positions and per-point RGBA. Arrays length N; rgba shape (N, 4) float32."""
        sym = symbol if symbol is not None else self._symbol
        ms = marker_size if marker_size is not None else self._marker_size
        ew = edge_width if edge_width is not None else self._edge_width
        t = np.asarray(t, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        rgba = np.asarray(rgba, dtype=np.float32)
        n = int(t.shape[0])
        if y.shape[0] != n or rgba.shape[0] != n:
            raise ValueError(f't, y, rgba length mismatch: {n}, {y.shape[0]}, {rgba.shape[0]}')
        if n == 0:
            pos = np.zeros((0, 2), dtype=np.float32)
        else:
            pos = np.column_stack([t, y]).astype(np.float32, copy=False)
        self._markers.set_data(pos=pos, size=ms, face_color=rgba, edge_width=ew, symbol=sym)  # type: ignore[arg-type]


    def set_from_build_result(self, curr_spike_t: np.ndarray, curr_spike_y: np.ndarray, curr_spike_pens: Sequence[QPen], **kwargs: Any) -> None:
        """Convenience: convert `build_spikes_data_values_from_df` pens to RGBA and call `set_spike_arrays`."""
        rgba = qpen_list_to_rgba_array(curr_spike_pens)
        self.set_spike_arrays(curr_spike_t, curr_spike_y, rgba, **kwargs)



def _unit_grid_line_visual(x0: float, x1: float, n_cells: int, parent: Node) -> Optional[vz.Line]:
    """Weak horizontal rules at integer y (same spirit as PyQt Graph `GridItem`); skip if n_cells < 2."""
    if n_cells < 2:
        return None
    ys = np.arange(1.0, float(n_cells), dtype=np.float32)
    if ys.size == 0:
        return None
    nseg = int(ys.size)
    pos = np.empty((nseg * 2, 2), dtype=np.float32)
    pos[0::2, 0] = x0
    pos[1::2, 0] = x1
    pos[0::2, 1] = ys
    pos[1::2, 1] = ys
    connect = np.arange(nseg * 2, dtype=np.uint32).reshape(nseg, 2)
    line = vz.Line(pos=pos, connect=connect, color=(0.53, 0.53, 0.53, 0.55), width=1.0, method='gl', parent=parent)  # type: ignore[call-arg]
    line.order = -55
    return line


def _time_bin_edge_vertical_lines(edge_times: np.ndarray, y0: float, y1: float, parent: Node, *, rgba: Tuple[float, float, float, float] = (0.92, 0.92, 0.98, 0.6), line_width: float = 1.0) -> Optional[vz.Line]:
    """Low-alpha vertical lines at decoded time-bin edges (world time on x); behind unit grid (order -50) and spikes (order 5)."""
    t_arr = np.asarray(edge_times, dtype=np.float32).ravel()
    if t_arr.size < 2:
        return None
    n = int(t_arr.size)
    pos = np.empty((n * 2, 2), dtype=np.float32)
    pos[0::2, 0] = t_arr
    pos[0::2, 1] = float(y0)
    pos[1::2, 0] = t_arr
    pos[1::2, 1] = float(y1)
    connect = np.arange(n * 2, dtype=np.uint32).reshape(n, 2)
    line = vz.Line(pos=pos, connect=connect, color=rgba, width=line_width, method='gl', parent=parent)  # type: ignore[call-arg]
    line.order = -50
    return line


def _marker_style_from_pg_kwargs(pg_kw: Dict[str, Any], *, fallback_size: float = 6.0) -> Tuple[str, float, bool]:
    """Map pyqtgraph-oriented scatter kwargs to vispy Markers: (symbol, size, scaling)."""
    px_mode = bool(pg_kw.get('pxMode', True))
    size = float(pg_kw.get('size', fallback_size))
    sym = pg_kw.get('symbol', 'vbar')
    if isinstance(sym, str):
        return (sym, size, px_mode)
    return ('vbar', size, px_mode)


@function_attributes(short_name=None, tags=['vispy', 'raster', '2D', 'gpu'], input_requires=[], output_provides=[], uses=['_prepare_spikes_df_from_filter_epochs', '_build_scatter_plotting_managers', 'Render2DScrollWindowPlotMixin'], used_by=[], creation_date='2026-03-28', related_items=['plot_multiple_raster_plot'])
def plot_multiple_raster_plot_vispy(filter_epochs_df: pd.DataFrame, spikes_df: pd.DataFrame, included_neuron_ids=None, unit_sort_order=None, unit_colors_list=None, scatter_plot_kwargs=None, epoch_id_key_name='temp_epoch_id', scatter_app_name: str = 'Pho Stacked Replays', defer_show: bool = False, active_context=None, *, draw_unit_grid: bool = True, bgcolor: str = 'white', time_bin_raster_view: Any = None, clear_host_scene: bool = True, time_bin_edges: Optional[np.ndarray] = None, num_epoch_time_bins: Optional[int] = None, **kwargs) -> VispyMultiRasterPlotTuple:
    """Multi-row spike rasters in one `SceneCanvas` (one view per epoch), or embedded into an existing `ViewBox` via `time_bin_raster_view`. Same data arguments as `plot_multiple_raster_plot`.

    Returns `VispyMultiRasterPlotTuple(canvas, plots, plots_data)`:
    - `plots.views` / `plots.raster_visuals` / `plots.grid_lines` / `plots.time_bin_edge_lines` are dicts keyed by epoch index (``an_epoch.Index``).
    - When `time_bin_raster_view` is set, `plots.grid` is ``None``, every `plots.views[k]` is that host view, and `canvas` is the host view's canvas.
    - `active_context` is accepted for API parity with the PyQtGraph helper (stored on `plots_data` when provided).
    - Optional `time_bin_edges` (absolute times, same axis as spikes) draws faint vertical guides at those edges; when set it overrides `num_epoch_time_bins` for line placement.
    - Optional `num_epoch_time_bins` otherwise uses `linspace` across each epoch's [start, stop] (aligned with the time-bin row when counts match).

    Usage:

        from pyphoplacecellanalysis.Pho2D.vispy.vispy_raster import VispyRasterVisual, plot_multiple_raster_plot_vispy, VispyMultiRasterPlotTuple

        a_track_name: str = 'roam'
        a_decoder = masked_container.pf1D_Decoder_dict[a_track_name]
        active_aclus = np.array(a_decoder.ratemap.neuron_ids)
        n_active_aclus: int = len(active_aclus)
        print(f'n_active_aclus: {n_active_aclus}')

        # unit_colors_list = None # default rainbow of colors for the raster plots
        neuron_qcolors_list = [pg.mkColor('black') for aclu in active_aclus] # solid green for all
        unit_colors_list = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)

        # active_epochs_df: pd.DataFrame = a_flat_matching_results_list_ds.filter_epochs[a_flat_matching_results_list_ds.filter_epochs['original_epoch_idx'] < 3]
        active_epochs_df: pd.DataFrame = a_flat_matching_results_list_ds.filter_epochs[a_flat_matching_results_list_ds.filter_epochs['original_epoch_idx'] < 10]
        actIve_filter_epochs_spikes_df: pd.DataFrame = a_decoder.spikes_df
        new_all_aclus_sort_indicies = None
        defer_show = False
        save_figure = False

        # pen = {'color': 'white', 'width': 1}
        # override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=5, pen=None) ## small
        override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=6, pen=None) ## mid
        # override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=10, pen=None) ## big
        # override_scatter_plot_kwargs = dict(pxMode=True, symbol='vbar', size=0.001, pen=None)

        _out_vispy_raster: VispyMultiRasterPlotTuple = plot_multiple_raster_plot_vispy(filter_epochs_df=active_epochs_df, spikes_df=actIve_filter_epochs_spikes_df,
                                                            included_neuron_ids=active_aclus,
                                                            # unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_L, 
                                                            scatter_plot_kwargs=override_scatter_plot_kwargs,
                                            epoch_id_key_name='replay_epoch_id', scatter_app_name=f"{a_track_name} Decoded Example Replays", defer_show=defer_show,
                                            active_context=curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track=a_track_name, epoch='example_replays'))



    """
    _ = kwargs
    rebuild_spikes_df_anyway = True
    if rebuild_spikes_df_anyway:
        spikes_df = spikes_df.copy()
        filter_epochs_df = filter_epochs_df.copy()
    if rebuild_spikes_df_anyway or (epoch_id_key_name not in spikes_df.columns):
        spikes_df = _prepare_spikes_df_from_filter_epochs(spikes_df, filter_epochs=filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name=epoch_id_key_name, debug_print=False)

    plots_data = RenderPlotsData(scatter_app_name)
    if active_context is not None:
        plots_data.active_context = active_context

    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)
    spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(spikes_df, overwrite_existing=True)
    spikes_df = _ensure_spikes_df_canonical_time_t(spikes_df)

    merged_pg_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)
    if time_bin_raster_view is None:
        print(f'scatter_plot_kwargs: {scatter_plot_kwargs}\nmerged_pg_kwargs: {merged_pg_kwargs}')
    sym, msize, scaling = _marker_style_from_pg_kwargs(merged_pg_kwargs)
    if time_bin_raster_view is None:
        print(f'sym: {sym}, msize: {msize}, scaling: {scaling}')

    views: Dict[Any, Any] = {}
    raster_visuals: Dict[Any, VispyRasterVisual] = {}
    grid_lines: Dict[Any, Optional[vz.Line]] = {}
    time_bin_edge_lines: Dict[Any, Optional[vz.Line]] = {}

    n_cells = int(plots_data.n_cells)

    host: Any = time_bin_raster_view
    grid: Any = None
    canvas: Any = None

    if host is not None:
        if clear_host_scene:
            for _child in list(host.scene.children):
                _child.parent = None
        canvas = cast(Any, host).canvas
        host_pz = scene.PanZoomCamera(aspect=None)
        host.camera = host_pz
        host_pz.interactive = False
        if hasattr(host, 'bgcolor'):
            host.bgcolor = bgcolor
        gx0 = float(filter_epochs_df['start'].min())
        gx1 = float(filter_epochs_df['stop'].max())
        host_pz.rect = Rect((gx0, 0.0), ((gx1 - gx0), max(float(n_cells - 1), 1.0)))
    else:
        canvas = scene.SceneCanvas(keys='interactive', show=(not defer_show), title=scatter_app_name, size=(1000, max(400, 80 * len(filter_epochs_df))), bgcolor=bgcolor, resizable=True)
        grid = canvas.central_widget.add_grid()
        grid.spacing = 0

    for an_epoch in filter_epochs_df.itertuples():
        row = int(an_epoch.Index)
        if host is not None:
            view = host
            views[an_epoch.Index] = host
        else:
            assert grid is not None
            view = grid.add_view(row=row, col=0, camera='panzoom', bgcolor=bgcolor)
            row_pz = scene.PanZoomCamera(aspect=None)
            view.camera = row_pz
            row_pz.interactive = False
            views[an_epoch.Index] = view

        scene_parent = cast(Node, view.scene)
        a_vispy_raster_visual: VispyRasterVisual = VispyRasterVisual(parent=scene_parent, symbol=sym, marker_size=msize, scaling=scaling, edge_width=0.0, order=5)
        raster_visuals[an_epoch.Index] = a_vispy_raster_visual

        _active_epoch_spikes_df = spikes_df[spikes_df[epoch_id_key_name] == an_epoch.Index]
        curr_spike_t, curr_spike_y, curr_spike_pens, _, _, curr_n = Render2DScrollWindowPlotMixin.build_spikes_data_values_from_df(_active_epoch_spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=False)
        if curr_n > 0:
            a_vispy_raster_visual.set_from_build_result(curr_spike_t, curr_spike_y, curr_spike_pens)
        else:
            a_vispy_raster_visual.set_spike_arrays(np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 4), dtype=np.float32))

        x0, x1 = float(an_epoch.start), float(an_epoch.stop)
        if host is None:
            assert isinstance(view.camera, scene.PanZoomCamera)
            view.camera.rect = Rect((x0, 0.0), ((x1 - x0), max(float(n_cells - 1), 1.0)))

        if draw_unit_grid:
            grid_lines[an_epoch.Index] = _unit_grid_line_visual(x0, x1, n_cells, scene_parent)
        else:
            grid_lines[an_epoch.Index] = None
        y_hi = max(float(n_cells - 1), 1.0)
        edges_vis: Optional[np.ndarray] = None
        if time_bin_edges is not None:
            te = np.asarray(time_bin_edges, dtype=np.float32).ravel()
            if te.size >= 2:
                edges_vis = te
        if edges_vis is None and num_epoch_time_bins is not None and int(num_epoch_time_bins) > 0:
            n_tb = int(num_epoch_time_bins)
            edges_vis = np.linspace(x0, x1, n_tb + 1, dtype=np.float32)
        if edges_vis is not None:
            time_bin_edge_lines[an_epoch.Index] = _time_bin_edge_vertical_lines(edges_vis, 0.0, y_hi, scene_parent)
        else:
            time_bin_edge_lines[an_epoch.Index] = None
    ## END for an_epoch in filter_epochs_df.itertuples()...
    
    plots = SimpleNamespace(canvas=canvas, grid=grid, views=views, raster_visuals=raster_visuals, grid_lines=grid_lines, time_bin_edge_lines=time_bin_edge_lines, scatter_plot_kwargs_merged=merged_pg_kwargs, filter_epochs_df=filter_epochs_df)

    return VispyMultiRasterPlotTuple(canvas, plots, plots_data)


def raster_scrolling_lines_example() -> None:
    """Original vispy gallery-style `ScrollingLines` demo (separate from spike rasters)."""
    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(1024, 768))
    g = canvas.central_widget.add_grid()
    view = g.add_view(0, 0)
    view.camera = scene.MagnifyCamera(mag=1, size_factor=0.5, radius_ratio=0.6)
    yax = scene.AxisWidget(orientation='left')
    yax.stretch = (0.05, 1)
    g.add_widget(yax, 0, 0)
    yax.link_view(view)
    xax = scene.AxisWidget(orientation='bottom')
    xax.stretch = (1, 0.05)
    g.add_widget(xax, 1, 0)
    xax.link_view(view)
    N = 4900
    M = 2000
    cols = int(N**0.5)
    view.camera.rect = (0, 0, cols, N / cols)
    lines = scene.ScrollingLines(n_lines=N, line_size=M, columns=cols, dx=0.8 / M, cell_size=(1, 8), parent=view.scene)  # type: ignore[call-arg]
    lines.transform = scene.STTransform(scale=(1, 1 / 8.0))

    def update(_ev) -> None:
        m = 50
        data = np.random.normal(size=(N, m), scale=0.3)
        data[data > 1] += 4
        lines.roll_data(data)
        canvas.context.flush()

    timer = app.Timer(connect=update, interval=0)  # type: ignore[arg-type]
    timer.start()


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        raster_scrolling_lines_example()
        app.run()
