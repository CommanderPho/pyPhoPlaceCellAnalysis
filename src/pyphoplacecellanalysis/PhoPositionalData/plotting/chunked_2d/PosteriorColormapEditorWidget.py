"""
Performant pyqtgraph colormap editor for decoded posterior heatmaps.

Updates posterior ImageItems' LUT/levels only (no re-decoding). Applies only when
use_advanced_3D_cmap=False in the renderer; when use_advanced_3D_cmap=True, posteriors
are precomputed RGBA and this editor has no effect.
"""

from typing import List, Optional, Tuple, Union

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import GradientEditorItem

# Default preset names (matplotlib source)
DEFAULT_PRESET_NAMES = ('viridis', 'magma', 'plasma', 'inferno', 'jet', 'cividis', 'turbo')

# Debounce delay for live gradient preview (ms)
GRADIENT_DEBOUNCE_MS = 60


def _get_cmap(name: str, source: str = 'matplotlib'):
    try:
        return pg.colormap.get(name, source)
    except Exception:
        return pg.colormap.get('viridis', 'matplotlib')


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


__all__ = ['PosteriorColormapEditorWidget', 'DEFAULT_PRESET_NAMES', 'GRADIENT_DEBOUNCE_MS', '_get_cmap']
