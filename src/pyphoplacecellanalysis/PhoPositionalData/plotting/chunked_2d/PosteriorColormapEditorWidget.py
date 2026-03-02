"""
Performant pyqtgraph colormap editor for decoded posterior heatmaps.

Updates posterior ImageItems' LUT/levels only (no re-decoding). Applies only when
use_advanced_3D_cmap=False in the renderer; when use_advanced_3D_cmap=True, posteriors
are precomputed RGBA and this editor has no effect.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import GradientEditorItem

from pyphocorehelpers.gui.Qt.color_helpers import create_3d_lut_saturation, create_3d_lut_cmaps_interp, apply_3d_colormap, composite_stack


# Default preset names (matplotlib source)
DEFAULT_PRESET_NAMES = ('viridis', 'magma', 'plasma', 'inferno', 'jet', 'cividis', 'turbo')

# Debounce delay for live gradient preview (ms)
GRADIENT_DEBOUNCE_MS = 60


def _get_cmap(name: str, source: str = 'matplotlib'):
    try:
        return pg.colormap.get(name, source)
    except Exception:
        return pg.colormap.get('viridis', 'matplotlib')


# Presets for advanced 2D (value x time) colormap: cmap1 = early t, cmap2 = late t
ADVANCED_CMAP_PRESET_NAMES = ('Alpha Red', 'Alpha Green', 'Reds', 'Greens', 'Blues', 'viridis', 'magma')


def _make_alpha_red_cmap(min_alpha: int = 100, max_alpha: int = 255):
    pos = np.array([0.0, 1.0])
    colors = np.array([[255, 0, 0, min_alpha], [255, 0, 0, max_alpha]], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


def _make_alpha_green_cmap(min_alpha: int = 100, max_alpha: int = 255):
    pos = np.array([0.0, 1.0])
    colors = np.array([[0, 255, 0, min_alpha], [0, 255, 0, max_alpha]], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


def _get_advanced_cmap_preset(name: str):
    if name == 'Alpha Red':
        return _make_alpha_red_cmap()
    if name == 'Alpha Green':
        return _make_alpha_green_cmap()
    try:
        return pg.colormap.get(name, 'matplotlib')
    except Exception:
        return _make_alpha_red_cmap()


# ==================================================================================================================================================================================================================================================================================== #
# Standard 1D Colormaps                                                                                                                                                                                                                                                                #
# ==================================================================================================================================================================================================================================================================================== #
class PosteriorColormapEditorWidget(QtWidgets.QWidget):
    """
    Widget combining a ColorBarItem (interactive level range) and a preset dropdown
    to drive the colormap of posterior heatmap ImageItems. Optionally includes a
    GradientEditorItem for full gradient editing. Performant: only updates
    LUT/levels on existing ImageItems and triggers a single view redraw.
    """

    def __init__(self, image_items: Optional[List] = None, initial_cmap=None, values: Tuple[float, float] = (0.0, 1.0),
                 orientation: str = 'vertical', label: Optional[str] = None, preset_names: Optional[Tuple[str, ...]] = None,
                 use_gradient_editor: bool = False, parent=None):
        super().__init__(parent)
        self._image_items = list(image_items) if image_items else []
        self._values = values
        self._orientation = orientation
        self._preset_names = preset_names or DEFAULT_PRESET_NAMES
        self._use_gradient_editor = use_gradient_editor
        if initial_cmap is None:
            initial_cmap = _get_cmap(self._preset_names[0])
        if isinstance(initial_cmap, str):
            initial_cmap = _get_cmap(initial_cmap)
        self._initial_cmap = initial_cmap

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._combo = QtWidgets.QComboBox(self)
        self._combo.setToolTip("Posterior colormap preset")
        for name in self._preset_names:
            self._combo.addItem(name)
        self._combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self._combo)

        self._gl_widget = pg.GraphicsLayoutWidget(parent=self)
        self._color_bar = pg.ColorBarItem(values=values, width=20, colorMap=initial_cmap, label=label or 'posterior',
                                          interactive=True, orientation=orientation)
        self._gl_widget.addItem(self._color_bar, 0, 0)
        row_next = 1
        self._gradient_editor = None
        self._gradient_debounce_timer = None
        if use_gradient_editor:
            self._gradient_editor = GradientEditorItem(orientation='bottom')
            self._gradient_editor.setColorMap(initial_cmap)
            self._gl_widget.addItem(self._gradient_editor, row_next, 0)
            self._gradient_editor.sigGradientChangeFinished.connect(self._apply_gradient_to_posteriors)
            self._gradient_debounce_timer = QtCore.QTimer(self)
            self._gradient_debounce_timer.setSingleShot(True)
            self._gradient_debounce_timer.timeout.connect(self._apply_gradient_to_posteriors)
            self._gradient_editor.sigGradientChanged.connect(self._on_gradient_changed_debounce)
            row_next += 1
        layout.addWidget(self._gl_widget, 1)

        if self._image_items:
            self._color_bar.setImageItem(self._image_items)
        self._sync_combo_from_cmap(initial_cmap)

    def _sync_combo_from_cmap(self, cmap):
        try:
            name = getattr(cmap, 'name', None)
            if name and name in self._preset_names:
                idx = self._combo.findText(name)
                if idx >= 0:
                    self._combo.blockSignals(True)
                    self._combo.setCurrentIndex(idx)
                    self._combo.blockSignals(False)
        except Exception:
            pass

    def _on_preset_changed(self, name: str):
        cmap = _get_cmap(name)
        self._color_bar.setColorMap(cmap)
        if self._gradient_editor is not None:
            self._gradient_editor.setColorMap(cmap)
        self._trigger_view_redraw()


    def _on_gradient_changed_debounce(self, _gradient_item):
        if self._gradient_debounce_timer is not None:
            self._gradient_debounce_timer.stop()
            self._gradient_debounce_timer.start(GRADIENT_DEBOUNCE_MS)


    def _apply_gradient_to_posteriors(self):
        if self._gradient_editor is None:
            return
        lut = self._gradient_editor.getLookupTable(256)
        for img in self._image_items:
            img.setLookupTable(lut)
        if hasattr(self._color_bar, 'bar') and self._color_bar.bar is not None:
            self._color_bar.bar.setLookupTable(lut)
        self._trigger_view_redraw()

    def _trigger_view_redraw(self):
        for img in self._image_items:
            if img.scene() is not None and img.scene().views():
                for v in img.scene().views():
                    v.update()
                break
            break

    def setImageItems(self, image_items: List):
        self._image_items = list(image_items)
        self._color_bar.setImageItem(self._image_items)


    def gradientEditorItem(self) -> Optional[GradientEditorItem]:
        return self._gradient_editor

    def imageItems(self):
        return list(self._image_items)

    def colorBarItem(self):
        return self._color_bar

    def setColorMap(self, colorMap: Union[str, object]):
        if isinstance(colorMap, str):
            colorMap = _get_cmap(colorMap)
        self._color_bar.setColorMap(colorMap)
        if self._gradient_editor is not None:
            self._gradient_editor.setColorMap(colorMap)
        self._sync_combo_from_cmap(colorMap)
        self._trigger_view_redraw()

    def colorMap(self):
        return self._color_bar.colorMap()

    def setLevels(self, values=None, low=None, high=None):
        self._color_bar.setLevels(values=values, low=low, high=high)

    def levels(self):
        return self._color_bar.levels()


# ==================================================================================================================================================================================================================================================================================== #
# 2D Colormaps                                                                                                                                                                                                                                                                         #
# ==================================================================================================================================================================================================================================================================================== #
class PosteriorColormap2DEditorWidget(QtWidgets.QWidget):
    """
    Widget for editing the advanced_3D_cmap format: shows a 2D gradient preview
    (value x time -> RGBA) and two 1D colormap selectors (cmap1 = early t, cmap2 = late t).
    Emits sigAdvancedColormapChanged(cmap1, cmap2) so the renderer can re-apply without re-decoding.
    Does not import the renderer; preview is built via the callable preview_lut_builder.
    """

    sigAdvancedColormapChanged = QtCore.Signal(object, object)

    def __init__(self, preview_lut_builder: Optional[Callable] = None, n_t_bins_preview: int = 16,
                 initial_cmap1=None, initial_cmap2=None, parent=None):
        super().__init__(parent)
        self._preview_lut_builder = preview_lut_builder
        self._n_t_bins_preview = n_t_bins_preview
        if initial_cmap1 is None:
            initial_cmap1 = _get_advanced_cmap_preset(ADVANCED_CMAP_PRESET_NAMES[0])
        if initial_cmap2 is None:
            initial_cmap2 = _get_advanced_cmap_preset(ADVANCED_CMAP_PRESET_NAMES[1])
        if isinstance(initial_cmap1, str):
            initial_cmap1 = _get_advanced_cmap_preset(initial_cmap1)
        if isinstance(initial_cmap2, str):
            initial_cmap2 = _get_advanced_cmap_preset(initial_cmap2)
        self._cmap1 = initial_cmap1
        self._cmap2 = initial_cmap2

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Cmap 1 (early t):"))
        self._combo1 = QtWidgets.QComboBox(self)
        self._combo1.setToolTip("Colormap for early time bins")
        for name in ADVANCED_CMAP_PRESET_NAMES:
            self._combo1.addItem(name)
        self._combo1.currentTextChanged.connect(self._on_cmap1_changed)
        row1.addWidget(self._combo1)
        row1.addWidget(QtWidgets.QLabel("Cmap 2 (late t):"))
        self._combo2 = QtWidgets.QComboBox(self)
        self._combo2.setToolTip("Colormap for late time bins")
        for name in ADVANCED_CMAP_PRESET_NAMES:
            self._combo2.addItem(name)
        self._combo2.currentTextChanged.connect(self._on_cmap2_changed)
        row1.addWidget(self._combo2)
        layout.addLayout(row1)

        if preview_lut_builder is not None:
            self._n_t_spin = QtWidgets.QSpinBox(self)
            self._n_t_spin.setRange(8, 64)
            self._n_t_spin.setValue(n_t_bins_preview)
            self._n_t_spin.setSuffix(" t-bins")
            self._n_t_spin.valueChanged.connect(self._on_n_t_preview_changed)
            row1.addWidget(self._n_t_spin)

        self._gl_widget = pg.GraphicsLayoutWidget(parent=self)
        self._plot_item = self._gl_widget.addPlot(0, 0, title="Value × time (2D LUT preview)")
        self._preview_image = pg.ImageItem()
        self._plot_item.addItem(self._preview_image)
        self._plot_item.setLabel('left', 'value')
        self._plot_item.setLabel('bottom', 'time bin')
        layout.addWidget(self._gl_widget, 1)

        self._sync_combos_from_cmaps()
        self._refresh_preview()


    def _sync_combos_from_cmaps(self):
        idx1 = self._combo1.findText(self._preset_name_for_cmap(self._cmap1))
        if idx1 >= 0:
            self._combo1.blockSignals(True)
            self._combo1.setCurrentIndex(idx1)
            self._combo1.blockSignals(False)
        idx2 = self._combo2.findText(self._preset_name_for_cmap(self._cmap2))
        if idx2 >= 0:
            self._combo2.blockSignals(True)
            self._combo2.setCurrentIndex(idx2)
            self._combo2.blockSignals(False)


    def _preset_name_for_cmap(self, cmap) -> str:
        name = getattr(cmap, 'name', None)
        if name and name in ADVANCED_CMAP_PRESET_NAMES:
            return name
        return ADVANCED_CMAP_PRESET_NAMES[0]


    def _on_cmap1_changed(self, name: str):
        self._cmap1 = _get_advanced_cmap_preset(name)
        self._refresh_preview()
        self.sigAdvancedColormapChanged.emit(self._cmap1, self._cmap2)


    def _on_cmap2_changed(self, name: str):
        self._cmap2 = _get_advanced_cmap_preset(name)
        self._refresh_preview()
        self.sigAdvancedColormapChanged.emit(self._cmap1, self._cmap2)


    def _on_n_t_preview_changed(self, value: int):
        self._n_t_bins_preview = value
        self._refresh_preview()


    def _refresh_preview(self):
        if self._preview_lut_builder is None:
            return
        try:
            # Preferred path: builder with named cmap args (e.g. create_3d_lut_cmaps_interp)
            try:
                lut = self._preview_lut_builder(n_t_bins=self._n_t_bins_preview, cmap1_name=self._cmap1, cmap2_name=self._cmap2)
            except TypeError:
                # Fallback for simple signature: (n_t_bins, cmap1, cmap2)
                lut = self._preview_lut_builder(self._n_t_bins_preview, self._cmap1, self._cmap2)
        except Exception:
            return
        if lut is None or lut.size == 0:
            return
        lut = np.asarray(lut)
        if lut.ndim != 3 or lut.shape[2] != 4:
            return
        self._preview_image.setImage(lut, autoLevels=False, levels=(0, 255), axisOrder='row-major')


    def getCmap1(self):
        return self._cmap1


    def getCmap2(self):
        return self._cmap2


    def setCmap1(self, cmap, emit: bool = True):
        if isinstance(cmap, str):
            cmap = _get_advanced_cmap_preset(cmap)
        self._cmap1 = cmap
        self._sync_combos_from_cmaps()
        self._refresh_preview()
        if emit:
            self.sigAdvancedColormapChanged.emit(self._cmap1, self._cmap2)


    def setCmap2(self, cmap, emit: bool = True):
        if isinstance(cmap, str):
            cmap = _get_advanced_cmap_preset(cmap)
        self._cmap2 = cmap
        self._sync_combos_from_cmaps()
        self._refresh_preview()
        if emit:
            self.sigAdvancedColormapChanged.emit(self._cmap1, self._cmap2)



# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Pho Custom Widget based off of Silx's `ColormapDialogExample` -- featuring of a :mod:`~silx.gui.dialog.ColormapDialog`.
"""

import functools
import numpy
import scipy

from silx.gui import qt
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.colors import Colormap
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.colors import Colormap as SilxColormap


class _HashableLUTColormap(Colormap):
    """Silx Colormap wrapper that returns LUT as list of tuples so ColormapNameComboBox's icon cache (tuple(colors)) is hashable."""

    def getColormapLUT(self, copy=True):
        lut = super().getColormapLUT(copy=copy)
        if lut is None:
            return None
        lut = np.asarray(lut)
        if lut.ndim == 1:
            lut = lut.reshape(-1, 4)
        return [tuple(int(x) for x in row) for row in lut]


def _pg_colormap_to_silx_colormap(pg_cmap) -> Colormap:
    """Convert pyqtgraph ColorMap to silx.gui.colors.Colormap for ColormapDialog.setColormap()."""
    lut = pg_cmap.getLookupTable(start=0.0, stop=1.0, nPts=256, alpha=True, mode=pg_cmap.BYTE)
    return _HashableLUTColormap(name=None, colors=lut)


def pg_to_silx_colormap_dense(pg_colormap: pg.ColorMap,
                              n_colors: int = 256,
                              vmin: float = None,
                              vmax: float = None) -> SilxColormap:
    # Sample dense LUT from pg
    lut = pg_colormap.getLookupTable(nPts=n_colors, mode='byte')

    lut = np.asarray(lut, dtype=np.uint8)

    silx_cmap = SilxColormap(
        name=None,
        colors=lut,
        vmin=vmin,
        vmax=vmax,
        normalization='linear'
    )

    return silx_cmap


def get_default_cmaps(reapply_advanced_colormap_fn=None, **kwargs):
    """
    captures: create_3d_lut_cmaps_interp

    def _reapply_advanced_colormap():
        cls.plot_decoded_posteriors_for_frames(a_decoded_subdivided_epochs_result=a_decoded_subdivided_epochs_result,
                subdivided_epochs_df=subdivided_epochs_df, maze_bounds_t=maze_bounds_t,
                extant_posterior_image_items=posterior_image_items, track_plot_item=track_plot_item,
                use_advanced_3D_cmap=True, custom_cmap1=editor.getCmap1(), custom_cmap2=editor.getCmap2())

    reapply_advanced_colormap_fn = _reapply_advanced_colormap
    _out_dict = 

    """
    custom_cmap1 = kwargs.pop('custom_cmap1', None)
    custom_cmap2 = kwargs.pop('custom_cmap2', None)
    _out_dict = {}

    if custom_cmap1 is None or custom_cmap2 is None:
        # --- Define Custom Alpha-Only Colormaps ---
        # Positions range from 0.0 to 1.0 (representing the v_idx mapping)
        pos = np.array([0.0, 1.0])
        # pos = np.array([0.5, global_max_v])
        # min_cmap_occupancy: int = 0
        min_cmap_occupancy: int = 100
        max_cmap_occupancy: int = 255
        # Custom "Alpha Red": R=255, G=0, B=0, Alpha mapping from 0 to 255
        colors_red = np.array([[255, 0, 0, min_cmap_occupancy], [255, 0, 0, max_cmap_occupancy]], dtype=np.ubyte)
        custom_cmap1 = pg.ColorMap(pos, colors_red)
        # Custom "Alpha Green": R=0, G=255, B=0, Alpha mapping from 0 to 255
        colors_green = np.array([[0, 255, 0, min_cmap_occupancy], [0, 255, 0, max_cmap_occupancy]], dtype=np.ubyte)
        custom_cmap2 = pg.ColorMap(pos, colors_green)

    _out_dict['custom_cmap1'] = custom_cmap1
    _out_dict['custom_cmap2'] = custom_cmap2

    ## have custom_cmap1, custom_cmap2:
    editor = PosteriorColormap2DEditorWidget(preview_lut_builder=create_3d_lut_cmaps_interp, n_t_bins_preview=16)
    _out_dict['posterior_colormap_editor'] = editor

    if reapply_advanced_colormap_fn is None:
        def on_reapply_advanced_colormap_fn():
            print(f'reapply_advanced_colormap_fn():')

        reapply_advanced_colormap_fn = on_reapply_advanced_colormap_fn

    ## set callback:
    editor.sigAdvancedColormapChanged.connect(reapply_advanced_colormap_fn)

    return _out_dict


class EditableColormap2DEditorWidget(QtWidgets.QMainWindow):
    """EditableColormap2DEditorWidget presents a 2D colormap in a child PosteriorColormap2DEditorWidget, with two pg GradientWidgets for editing the 1D colormaps (early t / late t).

        from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import EditableColormap2DEditorWidget

        editable_editor: EditableColormap2DEditorWidget = EditableColormap2DEditorWidget()
        editable_editor.show()

    """

    def __init__(self, custom_cmap1=None, custom_cmap2=None, parent=None):
        super(EditableColormap2DEditorWidget, self).__init__(parent)

        _out_dict = get_default_cmaps(custom_cmap1=custom_cmap1, custom_cmap2=custom_cmap2)
        self.colorEditor: PosteriorColormap2DEditorWidget = _out_dict['posterior_colormap_editor']
        custom_cmap1 = _out_dict['custom_cmap1']
        custom_cmap2 = _out_dict['custom_cmap2']

        self.setWindowTitle("EditableColormap2DEditorWidget")

        self.colormap_1D_list = [custom_cmap1, custom_cmap2]
        self.gradient_widgets: List[pg.GradientWidget] = []

        mainWidget = QtWidgets.QWidget(self)
        mainLayout = QtWidgets.QHBoxLayout()
        mainWidget.setLayout(mainLayout)
        mainLayout.addWidget(self.colorEditor)
        mainLayout.addSpacing(10)

        cmap_panel = QtWidgets.QWidget(self)
        cmap_panel.setLayout(QtWidgets.QVBoxLayout())
        for i, (label_text, cmap) in enumerate(zip(("Cmap 1 (early t)", "Cmap 2 (late t)"), self.colormap_1D_list)):
            container = QtWidgets.QWidget(self)
            container.setLayout(QtWidgets.QVBoxLayout())
            container.layout().addWidget(QtWidgets.QLabel(label_text))
            gw = pg.GradientWidget(self, orientation='bottom')
            gw.setColorMap(cmap)
            container.layout().addWidget(gw)
            cmap_panel.layout().addWidget(container)
            self.gradient_widgets.append(gw)
            gw.sigGradientChangeFinished.connect(functools.partial(self._on_gradient_finished, i))
        mainLayout.addWidget(cmap_panel)
        self.setCentralWidget(mainWidget)
        self.colorEditor.sigAdvancedColormapChanged.connect(self._sync_1d_widgets_from_editor)


    def _sync_1d_widgets_from_editor(self, cmap1, cmap2):
        """Update the two 1D GradientWidgets and colormap_1D_list when the 2D editor dropdowns change."""
        self.colormap_1D_list[0] = cmap1
        self.colormap_1D_list[1] = cmap2
        for gw in self.gradient_widgets:
            gw.blockSignals(True)
        try:
            self.gradient_widgets[0].setColorMap(cmap1)
            self.gradient_widgets[1].setColorMap(cmap2)
        finally:
            for gw in self.gradient_widgets:
                gw.blockSignals(False)


    def _on_gradient_finished(self, index: int, _gradient_item=None):
        cmap = self.gradient_widgets[index].colorMap()
        self.colormap_1D_list[index] = cmap
        if index == 0:
            self.colorEditor.setCmap1(cmap, emit=True)
        else:
            self.colorEditor.setCmap2(cmap, emit=True)



# ==================================================================================================================================================================================================================================================================================== #
# Simple Silx Example                                                                                                                                                                                                                                                                  #
# ==================================================================================================================================================================================================================================================================================== #
class ColormapDialogExample(qt.QMainWindow):
    """PlotWidget with an ad hoc toolbar and a colorbar"""

    def __init__(self, parent=None):
        super(ColormapDialogExample, self).__init__(parent)
        self.setWindowTitle("Colormap dialog example")

        self.colormap1 = Colormap("viridis")
        self.colormap2 = Colormap("gray")

        self.colorBar = ColorBarWidget(self)

        self.colorDialogs = []

        options = qt.QWidget(self)
        options.setLayout(qt.QVBoxLayout())
        self.createOptions(options.layout())

        mainWidget = qt.QWidget(self)
        mainWidget.setLayout(qt.QHBoxLayout())
        mainWidget.layout().addWidget(options)
        mainWidget.layout().addWidget(self.colorBar)
        self.mainWidget = mainWidget

        self.setCentralWidget(mainWidget)
        self.createColorDialog()

    def createOptions(self, layout):
        button = qt.QPushButton("Create a new dialog")
        button.clicked.connect(self.createColorDialog)
        layout.addWidget(button)

        layout.addSpacing(10)

        button = qt.QPushButton("Set editable")
        button.clicked.connect(self.setEditable)
        layout.addWidget(button)
        button = qt.QPushButton("Set non-editable")
        button.clicked.connect(self.setNonEditable)
        layout.addWidget(button)

        layout.addSpacing(10)

        button = qt.QPushButton("Set no colormap")
        button.clicked.connect(self.setNoColormap)
        layout.addWidget(button)
        button = qt.QPushButton("Set colormap 1")
        button.clicked.connect(self.setColormap1)
        layout.addWidget(button)
        button = qt.QPushButton("Set colormap 2")
        button.clicked.connect(self.setColormap2)
        layout.addWidget(button)
        button = qt.QPushButton("Create new colormap")
        button.clicked.connect(self.setNewColormap)
        layout.addWidget(button)

        layout.addSpacing(10)

        button = qt.QPushButton("No histogram")
        button.clicked.connect(self.setNoHistogram)
        layout.addWidget(button)
        button = qt.QPushButton("Positive histogram")
        button.clicked.connect(self.setPositiveHistogram)
        layout.addWidget(button)
        button = qt.QPushButton("Neg-pos histogram")
        button.clicked.connect(self.setNegPosHistogram)
        layout.addWidget(button)
        button = qt.QPushButton("Negative histogram")
        button.clicked.connect(self.setNegativeHistogram)
        layout.addWidget(button)

        layout.addSpacing(10)

        button = qt.QPushButton("No range")
        button.clicked.connect(self.setNoRange)
        layout.addWidget(button)
        button = qt.QPushButton("Positive range")
        button.clicked.connect(self.setPositiveRange)
        layout.addWidget(button)
        button = qt.QPushButton("Neg-pos range")
        button.clicked.connect(self.setNegPosRange)
        layout.addWidget(button)
        button = qt.QPushButton("Negative range")
        button.clicked.connect(self.setNegativeRange)
        layout.addWidget(button)

        layout.addSpacing(10)

        button = qt.QPushButton("No data")
        button.clicked.connect(self.setNoData)
        layout.addWidget(button)
        button = qt.QPushButton("Zero to positive")
        button.clicked.connect(self.setSheppLoganPhantom)
        layout.addWidget(button)
        button = qt.QPushButton("Negative to positive")
        button.clicked.connect(self.setDataFromNegToPos)
        layout.addWidget(button)
        button = qt.QPushButton("With non finite values")
        button.clicked.connect(self.setDataWithNonFinite)
        layout.addWidget(button)

        layout.addStretch()

    def createColorDialog(self):
        newDialog = ColormapDialog(self)
        newDialog.finished.connect(functools.partial(self.removeColorDialog, newDialog))
        self.colorDialogs.append(newDialog)
        self.mainWidget.layout().addWidget(newDialog)

    def removeColorDialog(self, dialog, result):
        self.colorDialogs.remove(dialog)

    def setNoColormap(self):
        self.colorBar.setColormap(None)
        for dialog in self.colorDialogs:
            dialog.setColormap(None)

    def setColormap1(self):
        self.colorBar.setColormap(self.colormap1)
        for dialog in self.colorDialogs:
            dialog.setColormap(self.colormap1)

    def setColormap2(self):
        self.colorBar.setColormap(self.colormap2)
        for dialog in self.colorDialogs:
            dialog.setColormap(self.colormap2)

    def setEditable(self):
        for dialog in self.colorDialogs:
            colormap = dialog.getColormap()
            if colormap is not None:
                colormap.setEditable(True)

    def setNonEditable(self):
        for dialog in self.colorDialogs:
            colormap = dialog.getColormap()
            if colormap is not None:
                colormap.setEditable(False)

    def setNewColormap(self):
        self.colormap = Colormap("inferno")
        self.colorBar.setColormap(self.colormap)
        for dialog in self.colorDialogs:
            dialog.setColormap(self.colormap)

    def setNoHistogram(self):
        for dialog in self.colorDialogs:
            dialog.setHistogram()

    def setPositiveHistogram(self):
        histo = [5, 10, 50, 10, 5]
        pos = 1
        edges = list(range(pos, pos + len(histo)))
        for dialog in self.colorDialogs:
            dialog.setHistogram(histo, edges)

    def setNegPosHistogram(self):
        histo = [5, 10, 50, 10, 5]
        pos = -2
        edges = list(range(pos, pos + len(histo)))
        for dialog in self.colorDialogs:
            dialog.setHistogram(histo, edges)

    def setNegativeHistogram(self):
        histo = [5, 10, 50, 10, 5]
        pos = -30
        edges = list(range(pos, pos + len(histo)))
        for dialog in self.colorDialogs:
            dialog.setHistogram(histo, edges)

    def setNoRange(self):
        for dialog in self.colorDialogs:
            dialog.setDataRange()

    def setPositiveRange(self):
        for dialog in self.colorDialogs:
            dialog.setDataRange(1, 1, 10)

    def setNegPosRange(self):
        for dialog in self.colorDialogs:
            dialog.setDataRange(-10, 1, 10)

    def setNegativeRange(self):
        for dialog in self.colorDialogs:
            dialog.setDataRange(-10, float("nan"), -1)

    def setNoData(self):
        for dialog in self.colorDialogs:
            dialog.setData(None)

    def setSheppLoganPhantom(self):
        from silx.image import phantomgenerator
        data = phantomgenerator.PhantomGenerator.get2DPhantomSheppLogan(256)
        data = data * 1000
        if scipy is not None:
            from scipy import ndimage
            data = ndimage.gaussian_filter(data, sigma=20)
        data = numpy.random.poisson(data)
        self.data = data
        for dialog in self.colorDialogs:
            dialog.setData(self.data)

    def setDataFromNegToPos(self):
        data = numpy.ones((50,50))
        data = numpy.random.poisson(data)
        self.data = data - 0.5
        for dialog in self.colorDialogs:
            dialog.setData(self.data)

    def setDataWithNonFinite(self):
        from silx.image import phantomgenerator
        data = phantomgenerator.PhantomGenerator.get2DPhantomSheppLogan(256)
        data = data * 1000
        if scipy is not None:
            from scipy import ndimage
            data = ndimage.gaussian_filter(data, sigma=20)
        data = numpy.random.poisson(data).astype(numpy.float32)
        data[10] = float("nan")
        data[50] = float("+inf")
        data[100] = float("-inf")
        self.data = data
        for dialog in self.colorDialogs:
            dialog.setData(self.data)


# ==================================================================================================================================================================================================================================================================================== #
# Examples:                                                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #




def example_pyqtgraph_colormap_widget_2D_main():
    """

    from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import ColormapDialogExample, example_silx_colormap_widget_2D_main

    """
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    _out_dict = get_default_cmaps()
    editor = _out_dict['posterior_colormap_editor']
    editor.show()
    app.exec()


def example_combined_editable_colormap_widget_2D_main():
    """

    from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import ColormapDialogExample, example_silx_colormap_widget_2D_main
    from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import EditableColormap2DEditorWidget

    editable_editor: EditableColormap2DEditorWidget = EditableColormap2DEditorWidget()
    editable_editor.show()

    """
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    editable_editor: EditableColormap2DEditorWidget = EditableColormap2DEditorWidget()
    editable_editor.show()

    app.exec()





def example_silx_colormap_widget_2D_main():
    """

    from pyphoplacecellanalysis.PhoPositionalData.plotting.chunked_2d.PosteriorColormapEditorWidget import ColormapDialogExample, example_silx_colormap_widget_2D_main

    """
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    example = ColormapDialogExample()
    example.show()

    app.exec()


if __name__ == '__main__':
    # example_silx_colormap_widget_2D_main() ## Silx
    # example_pyqtgraph_colormap_widget_2D_main() ## PyQtGraph
    example_combined_editable_colormap_widget_2D_main() # combined


__all__ = ['PosteriorColormapEditorWidget', 'PosteriorColormap2DEditorWidget', 'DEFAULT_PRESET_NAMES',
           'ADVANCED_CMAP_PRESET_NAMES', 'GRADIENT_DEBOUNCE_MS', '_get_cmap']
