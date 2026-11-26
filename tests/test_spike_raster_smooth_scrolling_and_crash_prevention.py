import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

tests_folder = Path(os.path.dirname(__file__)).resolve()
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters

from pyphoplacecellanalysis.General.Model.TimeWindow import TimeWindow
from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow
from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource, SpikesDataframeDatasource
from pyphoplacecellanalysis.General.Model.LiveWindowedData import LiveWindowedData
from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.FastCullableSpikeRasterItem import FastCullableSpikeRasterItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.TiledPosteriorHeatmapItem import TiledPosteriorHeatmapItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster, SynchronizedPlotMode
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPositionDecoderPlotter import _included_posterior_bin_indices_for_viewport, _decoder_bin_left_right_edges
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedGenericPlotterLayer import TimeSynchronizedGenericPlotterLayer
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _build_synthetic_spikes_df(n_spikes: int = 250000, n_neurons: int = 108, t_start: float = 10.0, t_duration: float = 2100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_vals = np.sort(rng.uniform(t_start, t_start + t_duration, size=n_spikes)).astype(np.float64)
    neuron_idxs = rng.integers(0, n_neurons, size=n_spikes, dtype=np.int32)
    aclu_ids = (neuron_idxs + 2).astype(np.int32)
    y_vals = (neuron_idxs.astype(np.float64) + 0.5)
    spikes_df = pd.DataFrame({
        'flat_spike_idx': np.arange(n_spikes, dtype=np.int64),
        't_rel_seconds': t_vals,
        't': t_vals.copy(),
        't_seconds': t_vals.copy(),
        'aclu': aclu_ids,
        'fragile_linear_neuron_IDX': neuron_idxs,
        'neuron_IDX': neuron_idxs.copy(),
        'visualization_raster_y_location': y_vals,
        'neuron_type': np.full(n_spikes, 'pyr', dtype=object),
    })
    spikes_df.spikes.set_time_variable_name('t_rel_seconds')
    return spikes_df


def _render_graphics_view_to_qimage(view: QtWidgets.QGraphicsView, width: int = 1000, height: int = 240) -> QtGui.QImage:
    view.resize(width, height)
    img = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
    img.fill(QtGui.QColor(0, 0, 0, 255))
    painter = QtGui.QPainter(img)
    try:
        view.render(painter)
    finally:
        painter.end()
    return img


class TestCrashAndOOMPrevention(unittest.TestCase):
    """Tests Phase 1 crash, bounds, QTime overflow, OpenGL default, and OOM guards."""

    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_qapp()


    def test_time_window_and_spikes_window_bounds_clamping(self):
        tw = TimeWindow(window_duration=15.0, window_start_time=10.0, min_duration=0.005, max_duration=2200.0)
        tw.window_duration = 0.00001
        self.assertGreaterEqual(tw.window_duration, 0.005)

        tw.window_duration = -50.0
        self.assertGreaterEqual(tw.window_duration, 0.005)

        prev_dur = tw.window_duration
        tw.window_duration = float('nan')
        self.assertEqual(tw.window_duration, prev_dur)

        tw.window_duration = float('inf')
        self.assertEqual(tw.window_duration, prev_dur)

        tw.window_duration = 999999.0
        self.assertLessEqual(tw.window_duration, 2200.0)

        # Inverted start/end in update_window_start_end should reorder cleanly instead of corrupting state
        tw.update_window_start_end(500.0, 200.0)
        self.assertLess(tw.active_window_start_time, tw.active_window_end_time)
        self.assertAlmostEqual(tw.active_window_start_time, 200.0, places=5)
        self.assertAlmostEqual(tw.active_window_end_time, 500.0, places=5)

        # SpikesDataframeWindow derives max_duration from total_df_start_end_times
        df = _build_synthetic_spikes_df(n_spikes=2000, n_neurons=10, t_start=0.0, t_duration=100.0)
        sw = SpikesDataframeWindow(df, window_duration=15.0, window_start_time=0.0)
        sw.window_duration = 50000.0
        self.assertLessEqual(sw.window_duration, 106.0)
        sw.window_duration = 0.0
        self.assertGreaterEqual(sw.window_duration, 0.005)


    def test_playback_control_bar_qtime_clamping_and_signal_blocking(self):
        extreme_inputs = [-1000.0, -0.001, 0.0, 3661.25, 86399.999, 86400.0, 250000.0, float('nan'), float('inf'), float('-inf')]
        for val in extreme_inputs:
            hours, minutes, seconds, milliseconds = Spike3DRasterBottomPlaybackControlBar.decompose_fractional_seconds(val)
            self.assertTrue(0 <= hours <= 23, f"Invalid hours={hours} for val={val}")
            self.assertTrue(0 <= minutes <= 59, f"Invalid minutes={minutes} for val={val}")
            self.assertTrue(0 <= seconds <= 59, f"Invalid seconds={seconds} for val={val}")
            self.assertTrue(0 <= milliseconds <= 999, f"Invalid milliseconds={milliseconds} for val={val}")
            qtime = QtCore.QTime(hours, minutes, seconds, milliseconds)
            self.assertTrue(qtime.isValid(), f"Invalid QTime constructed for val={val}")
        ## END for val in extreme_inputs...

        bar = Spike3DRasterBottomPlaybackControlBar()
        emitted_timestamps: List[float] = []
        bar.jump_specific_time.connect(lambda t: emitted_timestamps.append(t))
        time_edit_signals: List[object] = []
        bar.time_edit.timeChanged.connect(lambda qt: time_edit_signals.append(qt))
        bar.time_fractional_seconds = 123.456
        bar.time_fractional_seconds = -50.0
        bar.time_fractional_seconds = 999999.0
        bar.time_fractional_seconds = float('nan')
        # Programmatic setter must block signals so feedback loops never fire
        self.assertEqual(len(emitted_timestamps), 0)
        self.assertEqual(len(time_edit_signals), 0)
        bar.deleteLater()


    def test_generic_plotter_layer_zero_copy_chunked_aggregation_and_binary_search(self):
        n_xbins, n_ybins, n_tbins = 48, 32, 20000
        t_centers = np.linspace(0.0, 2000.0, n_tbins, dtype=np.float64)
        rng = np.random.default_rng(7)
        posterior_3d = rng.random((n_xbins, n_ybins, n_tbins), dtype=np.float32)

        t0 = time.perf_counter()
        included = _included_posterior_bin_indices_for_viewport(None, t_centers, 50.0, 1950.0)
        dt_search_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(dt_search_ms, 5.0)

        left_edges, right_edges = _decoder_bin_left_right_edges(None, t_centers)
        expected_mask = np.flatnonzero((left_edges >= 50.0) & (right_edges <= 1950.0))
        np.testing.assert_array_equal(included, expected_mask)

        # Verify TimeSynchronizedGenericPlotterLayer._update_plots chunked aggregation matches exact normalized sum
        sub_idx = _included_posterior_bin_indices_for_viewport(None, t_centers, 100.0, 600.0)
        raw_sum = np.nansum(posterior_3d[:, :, sub_idx[0]:sub_idx[-1] + 1].astype(np.float64), axis=-1)
        expected_norm = raw_sum / float(np.nansum(raw_sum))

        captured = {}

        class _DummyImageItem:
            def setCompositionMode(self, mode):
                pass

            def isVisible(self):
                return True

            def show(self):
                pass

            def hide(self):
                pass

            def setImage(self, img, rect=None, axisOrder=None):
                captured['image'] = img

        dummy_layer = object.__new__(TimeSynchronizedGenericPlotterLayer)
        dummy_layer._params = VisualizationParameters(name='test_layer', debug_print=False, drop_below_threshold=None, image_bounds_extent=QtCore.QRectF(0, 0, 10, 10), shared_axis_order=None)
        dummy_layer.data = {'main': posterior_3d, 'time_window_centers': t_centers}
        dummy_layer.contents = {'main': _DummyImageItem()}
        dummy_layer.parent = SimpleNamespace(last_included_posterior_bin_indices=sub_idx, time_window_centers=t_centers)
        dummy_layer.last_window_index = int(sub_idx[0])
        dummy_layer.last_window_time = float(t_centers[int(sub_idx[0])])

        t_agg_0 = time.perf_counter()
        dummy_layer._update_plots()
        dt_agg_ms = (time.perf_counter() - t_agg_0) * 1000.0
        self.assertIn('image', captured)
        self.assertLess(dt_agg_ms, 50.0)
        np.testing.assert_allclose(captured['image'], expected_norm, rtol=1e-5, atol=1e-6)


class TestSignalDeduplicationAndDataPath(unittest.TestCase):
    """Tests Phase 2 signal deduplication, O(log N) datasource slicing, lazy LiveWindowedData, and XLink."""

    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_qapp()


    def test_datasource_binary_search_window_and_lazy_live_windowed_data(self):
        spikes_df = _build_synthetic_spikes_df(n_spikes=100000, n_neurons=64, t_start=5.0, t_duration=1500.0)
        ds = SpikesDataframeDatasource(spikes_df)

        # Cached start/end must be O(1) and exact
        s0, e0 = ds.total_df_start_end_times
        t_col_name = ds.time_column_name
        self.assertAlmostEqual(s0, float(spikes_df[t_col_name].iloc[0]), places=9)
        self.assertAlmostEqual(e0, float(spikes_df[t_col_name].iloc[-1]), places=9)

        # Binary search slice must match pandas .between(new_start, new_end) reference
        t_lo, t_hi = 345.6, 567.8
        t_start = time.perf_counter()
        sliced = ds.get_updated_data_window(t_lo, t_hi)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.assertLess(elapsed_ms, 5.0)

        ref = spikes_df[spikes_df[t_col_name].between(t_lo, t_hi)]
        self.assertEqual(len(sliced), len(ref))
        np.testing.assert_array_equal(sliced[t_col_name].to_numpy(), ref[t_col_name].to_numpy())

        # LiveWindowedData must NOT eagerly slice on window updates by default
        sw = SpikesDataframeWindow(spikes_df, window_duration=20.0, window_start_time=10.0)
        self.assertFalse(sw.enable_eager_window_slicing)

        slice_call_count = [0]
        orig_get_window = sw.dataSource.get_updated_data_window

        def _counting_get_window(new_start, new_end):
            slice_call_count[0] += 1
            return orig_get_window(new_start, new_end)

        sw.dataSource.get_updated_data_window = _counting_get_window
        slice_call_count[0] = 0

        # Scroll 50 times without accessing `.active_windowed_df` -> 0 DataFrame slices!
        for step_idx in range(50):
            sw.update_window_start(20.0 + float(step_idx) * 2.5)
        ## END for step_idx in range(50)...

        self.assertEqual(slice_call_count[0], 0, "LiveWindowedData performed eager DataFrame slicing during scroll!")

        # Accessing `.active_windowed_df` on demand computes the slice cleanly
        df_win_1 = sw.active_windowed_df
        self.assertEqual(slice_call_count[0], 1)
        self.assertGreater(len(df_win_1), 0)


    def test_custom_linear_region_item_blocks_duplicate_line_signals(self):
        region = CustomLinearRegionItem(values=(10.0, 25.0), movable=True)
        emissions: List[Tuple[float, float]] = []
        region.sigRegionChanged.connect(lambda item: emissions.append(tuple(item.getRegion())))

        region.setRegion((20.0, 35.0))
        self.assertEqual(len(emissions), 1)
        # Setting the exact same region again must be a no-op (0 extra emissions)
        region.setRegion((20.0, 35.0))
        self.assertEqual(len(emissions), 1)


class TestFastGraphicsItemsAndBenchmark(unittest.TestCase):
    """Tests Phase 3, Phase 4, and Phase 5 rendering accuracy + <5ms scroll frame times across 15s, 235s, and 2000s."""

    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_qapp()


    def test_fast_cullable_spike_raster_item_tick_and_lod_modes(self):
        rng = np.random.default_rng(123)
        n_spikes, n_neurons = 150000, 80
        t_arr = np.sort(rng.uniform(0.0, 2000.0, size=n_spikes)).astype(np.float64)
        y_arr = (rng.integers(0, n_neurons, size=n_spikes).astype(np.float64) + 0.5)
        pal_idx = (y_arr.astype(np.int32) % n_neurons)
        palette = np.zeros((n_neurons, 4), dtype=np.uint8)
        palette[:, 0] = np.linspace(60, 255, n_neurons).astype(np.uint8)
        palette[:, 1] = np.linspace(120, 220, n_neurons).astype(np.uint8)
        palette[:, 2] = 240
        palette[:, 3] = 230

        pw = pg.PlotWidget()
        pw.resize(1000, 300)
        raster_item = FastCullableSpikeRasterItem(
            spike_times=t_arr,
            spike_y=y_arr,
            spike_palette_idx=pal_idx,
            palette_rgba=palette,
            y_min=0.0,
            y_max=float(n_neurons),
            n_rows=n_neurons,
        )
        pw.addItem(raster_item)
        pw.setYRange(0.0, float(n_neurons), padding=0.0)

        # Mode A: 15s window -> crisp tick atlas
        pw.setXRange(100.0, 115.0, padding=0.0)
        img_15s = _render_graphics_view_to_qimage(pw, 1000, 300)
        ptr_15 = img_15s.bits()
        ptr_15.setsize(img_15s.byteCount())
        arr_15 = np.frombuffer(ptr_15, dtype=np.uint8).reshape((300, 1000, 4))
        self.assertGreater(int(np.count_nonzero(arr_15[:, :, :3].sum(axis=2) > 30)), 100, "Mode A tick render produced blank image!")

        # Mode B: 2000s window -> tiled RGBA spike-density LOD mipmap
        pw.setXRange(0.0, 2000.0, padding=0.0)
        img_2000s = _render_graphics_view_to_qimage(pw, 1000, 300)
        ptr_2000 = img_2000s.bits()
        ptr_2000.setsize(img_2000s.byteCount())
        arr_2000 = np.frombuffer(ptr_2000, dtype=np.uint8).reshape((300, 1000, 4))
        self.assertGreater(int(np.count_nonzero(arr_2000[:, :, :3].sum(axis=2) > 30)), 500, "Mode B LOD mipmap render produced blank image!")

        pw.close()
        pw.deleteLater()


    def test_tiled_posterior_heatmap_item_mipmaps_nan_black_and_labels(self):
        n_pos_bins, n_time_bins = 58, 40000
        posterior = np.zeros((n_pos_bins, n_time_bins), dtype=np.float32)
        t_idx = np.arange(n_time_bins)
        peak_bins = ((np.sin(t_idx * 0.01) * 0.45 + 0.5) * (n_pos_bins - 1)).astype(np.int32)
        posterior[peak_bins, t_idx] = 0.85
        # Inject NaN PBE gap columns -> must render as bad_color (black = 0,0,0)
        posterior[:, 1000:1500] = np.nan

        t_centers = np.linspace(0.0, 2000.0, n_time_bins, dtype=np.float64)
        xbin = np.linspace(0.0, 280.0, n_pos_bins, dtype=np.float64)

        pw = pg.PlotWidget()
        pw.resize(1000, 160)
        pw.getPlotItem().hideAxis('left')
        pw.getPlotItem().hideAxis('bottom')
        heatmap = TiledPosteriorHeatmapItem(
            posterior=posterior,
            time_window_centers=t_centers,
            xbin=xbin,
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            bad_color='black',
            y_bin_labels=['long', 'short'],
        )
        pw.addItem(heatmap)
        pw.setYRange(0.0, 280.0, padding=0.0)

        # Verify 1x (15s), 4x (235s), and 16x (2000s) zoom levels render cleanly
        for win_start, win_end in [(0.0, 15.0), (100.0, 335.0), (0.0, 2000.0)]:
            pw.setXRange(win_start, win_end, padding=0.0)
            img = _render_graphics_view_to_qimage(pw, 1000, 160)
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((160, 1000, 4))
            self.assertGreater(int(np.count_nonzero(arr[:, :, :3].sum(axis=2) > 40)), 200)
        ## END for win_start, win_end in [(0.0, 15.0), (100.0, 335.0), (0.0, 2000.0)]...

        # Check pure NaN viewport (t in [52.0, 72.0] falls strictly inside bins 1040..1440) renders black interior
        heatmap_no_lbl = TiledPosteriorHeatmapItem(
            posterior=posterior,
            time_window_centers=t_centers,
            xbin=xbin,
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            bad_color='black',
        )
        pw2 = pg.PlotWidget()
        pw2.setBackground('k')
        pw2.getPlotItem().hideAxis('left')
        pw2.getPlotItem().hideAxis('bottom')
        pw2.addItem(heatmap_no_lbl)
        pw2.setYRange(0.0, 280.0, padding=0.0)
        pw2.setXRange(52.0, 72.0, padding=0.0)
        img_nan = _render_graphics_view_to_qimage(pw2, 400, 120)
        ptr_nan = img_nan.bits()
        ptr_nan.setsize(img_nan.byteCount())
        arr_nan = np.frombuffer(ptr_nan, dtype=np.uint8).reshape((120, 400, 4))
        interior = arr_nan[20:100, 40:360, :3]
        self.assertEqual(int(np.max(interior)), 0, "NaN columns in TiledPosteriorHeatmapItem did not map to black!")

        pw.close()
        pw.deleteLater()
        pw2.close()
        pw2.deleteLater()


    def test_matplotlib_fallback_throttled_draw_idle(self):
        mpl_widget = MatplotlibTimeSynchronizedWidget()
        ax = mpl_widget.getFigure().add_subplot(111)
        ax.plot([0.0, 100.0], [0.0, 1.0])
        draw_idle_calls = [0]
        orig_draw_idle = mpl_widget.ui.canvas.draw_idle

        def _counted_draw_idle(*args, **kwargs):
            draw_idle_calls[0] += 1
            return orig_draw_idle(*args, **kwargs)

        mpl_widget.ui.canvas.draw_idle = _counted_draw_idle
        # Rapid burst of 25 scroll updates within <33ms must coalesce into 1 immediate draw_idle + 1 pending timer
        for idx in range(25):
            mpl_widget.on_window_changed(10.0 + float(idx), 25.0 + float(idx))
        ## END for idx in range(25)...

        self.assertLessEqual(draw_idle_calls[0], 2, f"Expected coalesced Matplotlib draw_idle, got {draw_idle_calls[0]} calls!")
        self.assertTupleEqual(ax.get_xlim(), (34.0, 49.0))
        mpl_widget.close()
        mpl_widget.deleteLater()


    def test_end_to_end_spike_raster_window_scroll_benchmark_and_duration_sweep(self):
        """End-to-end benchmark of Spike3DRasterWindowWidget (2D mode) + Spike2DRaster + 4 ContinuousDecode tracks + 1 Marginal track + 2D curves."""
        spikes_df = _build_synthetic_spikes_df(n_spikes=250000, n_neurons=108, t_start=10.0, t_duration=2100.0, seed=99)

        raster_win = Spike3DRasterWindowWidget(
            curr_spikes_df=spikes_df,
            window_duration=15.0,
            window_start_time=30.0,
            neuron_colors=None,
            neuron_sort_order=None,
            application_name='TestSmoothSpikeRasterWindow',
            type_of_3d_plotter=None,
            params_kwargs=dict(use_docked_pyqtgraph_plots=True),
        )
        raster_win.resize(1280, 900)
        spike_raster_plt_2d: Spike2DRaster = raster_win.spike_raster_plt_2d

        # 1. Verify OpenGL is safely OFF by default
        self.assertFalse(spike_raster_plt_2d.params.useOpenGL)
        self.assertFalse(spike_raster_plt_2d.params.enableExperimental)

        # 2. Verify both preview and windowed rasters use FastCullableSpikeRasterItem (single item, no 2x-3x duplicates)
        self.assertIsInstance(spike_raster_plt_2d.plots.preview_overview_scatter_plot, FastCullableSpikeRasterItem)
        self.assertIsInstance(spike_raster_plt_2d.plots.scatter_plot, FastCullableSpikeRasterItem)
        self.assertTrue(spike_raster_plt_2d.plots.preview_overview_scatter_plot._force_lod_only)

        # 3. Add static 2D time curves (50,000 samples x 3 position channels)
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import PositionRenderTimeCurves
        n_pos_samples = 50000
        pos_t = np.linspace(10.0, 2110.0, n_pos_samples, dtype=np.float64)
        pos_df = pd.DataFrame({
            't': pos_t,
            'lin_pos': 140.0 + 120.0 * np.sin(pos_t * 0.03),
            'speed': 15.0 + 10.0 * np.cos(pos_t * 0.05),
            'x': 140.0 + 120.0 * np.sin(pos_t * 0.03),
            'y': 142.0 + 3.0 * np.cos(pos_t * 0.05),
        })
        curve_mappings = PositionRenderTimeCurves.build_pre_spatial_to_spatial_mappings(spike_raster_plt_2d)
        curve_ds = PositionRenderTimeCurves.build_render_time_curves_datasource(pos_df.copy(), curve_mappings)
        spike_raster_plt_2d.add_3D_time_curves(curve_datasource=curve_ds)
        self.assertGreaterEqual(len(spike_raster_plt_2d.plots.time_curves), 1)

        # 4. Add 4 ContinuousDecode_* posterior tracks (58 x 80,000 bins) + red lin_pos overlay, and 1 marginal_over_track_ID track (2 x 80,000 bins)
        n_decode_bins = 80000
        decode_t_centers = np.linspace(10.0, 2110.0, n_decode_bins, dtype=np.float64)
        xbin_centers = np.linspace(0.0, 280.0, 58, dtype=np.float64)

        rng = np.random.default_rng(2026)
        base_posterior = np.zeros((58, n_decode_bins), dtype=np.float32)
        col_idx = np.arange(n_decode_bins)
        ridge = ((np.sin(col_idx * 0.008) * 0.45 + 0.5) * 57).astype(np.int32)
        base_posterior[ridge, col_idx] = rng.uniform(0.5, 1.0, size=n_decode_bins).astype(np.float32)
        base_posterior[:, 5000:5500] = np.nan

        decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        created_posterior_widgets = []
        for dec_name in decoder_names:
            track_id, widget, root_gfx, plot_item, dock_item = spike_raster_plt_2d.add_docked_decoded_posterior_track(
                name=f'ContinuousDecode_{dec_name} - 0.025',
                a_1D_posterior=base_posterior,
                time_window_centers=decode_t_centers,
                xbin=xbin_centers,
                measured_position_df=pos_df,
            )
            created_posterior_widgets.append((track_id, widget, root_gfx, plot_item))
            # Verify TO_WINDOW tracks are X-linked directly to main_plot_widget
            self.assertIs(plot_item.vb.linkedView(plot_item.vb.XAxis), spike_raster_plt_2d.plots.main_plot_widget.vb)
        ## END for dec_name in decoder_names...

        marginal_posterior = np.vstack([
            0.5 + 0.45 * np.sin(col_idx * 0.002),
            0.5 - 0.45 * np.sin(col_idx * 0.002),
        ]).astype(np.float32)
        marg_id, marg_widget, marg_gfx, marg_plot_item, _ = spike_raster_plt_2d.add_docked_marginal_track(
            name='ContinuousDecode_marginal_over_track_ID - 0.025',
            a_1D_posterior=marginal_posterior,
            time_window_centers=decode_t_centers,
            xbin=np.array([0.0, 1.0]),
            xbin_labels=['long', 'short'],
            measured_position_df=None,
        )
        self.assertIs(marg_plot_item.vb.linkedView(marg_plot_item.vb.XAxis), spike_raster_plt_2d.plots.main_plot_widget.vb)
        raster_win.update_scrolling_event_filters()
        QtWidgets.QApplication.processEvents()

        # Collect all QGraphicsViews across the raster window + overview + 4 posterior tracks + 1 marginal track
        all_views: List[QtWidgets.QGraphicsView] = []
        for _, _, rgfx, _ in created_posterior_widgets:
            all_views.append(rgfx)
        ## END for _, _, rgfx, _ in created_posterior_widgets...

        all_views.append(marg_gfx)
        if 'rasters[raster_window]' in spike_raster_plt_2d.plots.matplotlib_view_widgets:
            rw_gfx = getattr(spike_raster_plt_2d.plots.matplotlib_view_widgets['rasters[raster_window]'].ui, 'root_graphics_layout_widget', None)
            if rw_gfx is not None:
                all_views.append(rw_gfx)
        if 'rasters[raster_overview]' in spike_raster_plt_2d.plots.matplotlib_view_widgets:
            ro_gfx = getattr(spike_raster_plt_2d.plots.matplotlib_view_widgets['rasters[raster_overview]'].ui, 'root_graphics_layout_widget', None)
            if ro_gfx is not None:
                all_views.append(ro_gfx)

        # 5. Benchmark scroll + scene paint frame time across 15s, 235s, and 2000s window durations!
        benchmark_durations = [15.0, 235.0, 2000.0]

        for dur in benchmark_durations:
            raster_win.update_duration_Controls(dur)
            spike_raster_plt_2d._flush_coalesced_window_scroll()
            # Warmup 2 frames at this duration (builds LOD level once lazily on first wide zoom)
            for warm_i in range(2):
                spike_raster_plt_2d.shift_animation_frame_val(1)
                spike_raster_plt_2d._flush_coalesced_window_scroll()
                for v in all_views:
                    _render_graphics_view_to_qimage(v, 900, 90)
                ## END for v in all_views...
            ## END for warm_i in range(2)...

            step_signal_times_ms: List[float] = []
            step_full_paint_times_ms: List[float] = []
            n_bench_steps = 20
            max_start = max(35.0, 2080.0 - dur)
            step_starts = np.linspace(20.0, max_start, n_bench_steps)

            for s_val in step_starts:
                t0 = time.perf_counter()
                spike_raster_plt_2d.animation_active_time_window.update_window_start(float(s_val))
                spike_raster_plt_2d._flush_coalesced_window_scroll()
                t1 = time.perf_counter()
                for v in all_views:
                    v.viewport().repaint()
                ## END for v in all_views...
                t2 = time.perf_counter()
                step_signal_times_ms.append((t1 - t0) * 1000.0)
                step_full_paint_times_ms.append((t2 - t0) * 1000.0)
            ## END for s_val in step_starts...

            med_sig_ms = float(np.median(step_signal_times_ms))
            med_total_ms = float(np.median(step_full_paint_times_ms))
            print(f"\n[BENCHMARK] window_duration={dur:6.1f}s | median_scroll_update={med_sig_ms:.3f} ms | median_scroll_plus_7_track_repaint={med_total_ms:.3f} ms")
            self.assertLess(med_total_ms, 5.0, f"Scroll + repaint exceeded 5.0ms budget at {dur}s window: {med_total_ms:.3f}ms")
        ## END for dur in benchmark_durations...

        # 6. Stress-test rapid window duration sweep (0.001s -> 5000s -> negative -> NaN -> rapid scroll)
        sweep_durations = [0.001, 0.05, 1.0, 15.0, 75.0, 235.0, 800.0, 2000.0, 5000.0, 0.0001, -10.0, float('nan'), float('inf'), 30.0]
        for sweep_dur in sweep_durations:
            raster_win.on_render_window_duration_valueChanged(sweep_dur)
            spike_raster_plt_2d.shift_animation_frame_val(2)
            spike_raster_plt_2d._flush_coalesced_window_scroll()
        ## END for sweep_dur in sweep_durations...

        # Also test jump to extreme timestamps on bottom playback bar
        raster_win.perform_jump_specific_timestamp(-500.0)
        raster_win.perform_jump_specific_timestamp(999999.0)
        raster_win.perform_jump_specific_timestamp(float('nan'))
        raster_win.perform_jump_specific_timestamp(250.0)
        self.assertTrue(np.isfinite(spike_raster_plt_2d.animation_active_time_window.active_window_start_time))

        raster_win._delayed_gui_timer.stop()
        raster_win.close()
        raster_win.deleteLater()
        QtWidgets.QApplication.processEvents()


if __name__ == '__main__':
    unittest.main()
