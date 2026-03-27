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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import PhoDockAreaContainingWindow


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
class VispySceneTreeWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Qt tree widget that displays a vispy scene graph hierarchy.
    
    from pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets import VispySceneTreeWidget
    
    from pyphoplacecellanalysis.Pho2D.vispy.predicitive_decoding_vispy import Volumentric2DTimeSeriesPlotter
    viewer_3d = Volumentric2DTimeSeriesPlotter.init_from_position_and_decoder(curr_position_df=curr_position_df, xbin=xbin, ybin=ybin, p_x_given_n=p_x_given_n, t_bin_edges=t_bin_edges, highlight_epochs=highlight_epochs)
    scene_tree_widget = VispySceneTreeWidget(root_node=viewer_3d.canvas.scene, canvas=viewer_3d.canvas)
    scene_tree_widget.show()

    # The default 'Transform' column renderer includes location details for
    # NullTransform, STTransform, and MatrixTransform.

    # Optional custom renderer registration:
    # scene_tree_widget.register_column_renderer('Transform', render_transform_column)
    # scene_tree_widget.register_column_renderer('Name', lambda n: f"{n.__class__.__name__}:{getattr(n, 'name', '')}")

    Leading single-child wrapper chains (e.g. canvas.scene down to the main SubScene) are collapsed so their descendants appear as top-level rows without extra horizontal indent.
    # Type and Transform columns are hidden by default; show with e.g.
    # scene_tree_widget.tree.setColumnHidden(VispySceneTreeWidget._COL_TYPE, False)
    # scene_tree_widget.tree.setColumnHidden(VispySceneTreeWidget._COL_TRANSFORM, False)


    """

    _COL_TYPE = 1
    _COL_TRANSFORM = 6


    def __init__(self, root_node: Node, canvas: Optional[scene.SceneCanvas] = None, parent: Optional[Any] = None, column_renderers: Optional[Dict[str, Callable[[Node], str]]] = None):
        super().__init__(parent=parent)
        self._root_node = root_node
        self._canvas = canvas
        self._is_rebuilding = False
        self._column_headers = ['Name', 'Type', 'Visible', 'Order', 'Opacity', 'GL Blend', 'Transform']
        self._column_visibility = [True] * len(self._column_headers)
        self._user_column_renderers = dict(column_renderers or {})
        self._user_role = getattr(QtCore.Qt, 'UserRole', QtCore.Qt.ItemDataRole.UserRole)
        self._checked_state = getattr(QtCore.Qt, 'Checked', QtCore.Qt.CheckState.Checked)
        self._unchecked_state = getattr(QtCore.Qt, 'Unchecked', QtCore.Qt.CheckState.Unchecked)
        self._item_is_user_checkable = getattr(QtCore.Qt, 'ItemIsUserCheckable', QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        self._item_is_enabled = getattr(QtCore.Qt, 'ItemIsEnabled', QtCore.Qt.ItemFlag.ItemIsEnabled)
        self._item_is_editable = getattr(QtCore.Qt, 'ItemIsEditable', QtCore.Qt.ItemFlag.ItemIsEditable)
        self._init_ui()
        self.setWindowTitle('VispySceneTreeWidget')
        self.rebuild()


    def _init_ui(self) -> None:
        expanding = getattr(QtWidgets.QSizePolicy, 'Expanding', QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        controls_layout = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        controls_layout.addWidget(cast(Any, self.refresh_button))
        controls_layout.addStretch(1)
        layout.addLayout(cast(Any, controls_layout))

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(self._column_headers)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(False)
        header = self.tree.header()
        if header is not None:
            interactive_mode = getattr(QtWidgets.QHeaderView, 'Interactive', QtWidgets.QHeaderView.ResizeMode.Interactive)
            header_any = cast(Any, header)
            header_any.setStretchLastSection(False)
            header_any.setSectionResizeMode(0, interactive_mode)
            header_any.setSectionResizeMode(1, interactive_mode)
            header_any.setSectionResizeMode(2, interactive_mode)
            header_any.setSectionResizeMode(3, interactive_mode)
            header_any.setSectionResizeMode(4, interactive_mode)
            header_any.setSectionResizeMode(5, interactive_mode)
            header_any.setSectionResizeMode(6, interactive_mode)
        self.tree.setItemDelegateForColumn(5, _BlendPresetDelegate(self.tree))
        self.tree.setColumnHidden(self._COL_TYPE, True)
        self.tree.setColumnHidden(self._COL_TRANSFORM, True)
        layout.addWidget(cast(Any, self.tree), stretch=1)

        self.refresh_button.clicked.connect(self.rebuild)
        self.tree.itemChanged.connect(self._on_item_changed)




    def _set_column_visible(self, column: int, visible: bool) -> None:
        if column < 0 or column >= len(self._column_headers):
            return
        self._column_visibility[column] = visible
        self.tree.setColumnHidden(column, not visible)


    def _get_default_column_renderers(self) -> Dict[str, Callable[[Node], str]]:
        def _render_type(node: Node) -> str:
            return node.__class__.__name__

        def _render_name(node: Node) -> str:
            node_name = getattr(node, 'name', None)
            return '' if node_name is None else str(node_name)

        def _render_order(node: Node) -> str:
            return str(getattr(node, 'order', ''))

        def _render_opacity(node: Node) -> str:
            node_opacity_val = getattr(node, 'opacity', None)
            if isinstance(node_opacity_val, (float, int)):
                return f'{float(node_opacity_val):0.2f}'
            return ''

        def _render_gl_blend(node: Node) -> str:
            vshare = getattr(node, '_vshare', None)
            if vshare is None:
                return ''
            gl_state = getattr(vshare, 'gl_state', None)
            if not isinstance(gl_state, dict):
                return ''
            return str(gl_state.get('preset', '') or '')

        return {'Type': _render_type, 'Name': _render_name, 'Order': _render_order, 'Opacity': _render_opacity, 'GL Blend': _render_gl_blend, 'Transform': render_transform_column}


    def _get_cell_text(self, column_name: str, node: Node) -> str:
        renderers = self._get_default_column_renderers()
        renderers.update(self._user_column_renderers)
        renderer = renderers.get(column_name, None)
        if renderer is None:
            return ''
        try:
            return str(renderer(node))
        except Exception:
            return ''


    def register_column_renderer(self, column_name: str, renderer: Callable[[Node], str]) -> None:
        """Register or override a renderer for a text column name."""
        self._user_column_renderers[column_name] = renderer
        self.rebuild()


    def rebuild(self) -> None:
        self._is_rebuilding = True
        self.tree.blockSignals(True)
        self.tree.clear()
        effective = self._effective_display_root(self._root_node)
        effective_children = list(effective.children)
        if len(effective_children) == 0:
            self._populate(node=effective, parent_item=None)
        else:
            for child in effective_children:
                self._populate(node=child, parent_item=None)
        self.tree.expandToDepth(3)
        for col in range(len(self._column_headers)):
            if self._column_visibility[col]:
                self.tree.resizeColumnToContents(col)
        self.tree.blockSignals(False)
        self._is_rebuilding = False


    def _node_has_gl_blend(self, node: Node) -> bool:
        """True when the node supports set_gl_state (Visual subclasses)."""
        return hasattr(node, 'set_gl_state') and hasattr(node, '_vshare')


    _MAX_SINGLE_CHILD_DESCENT = 256

    def _effective_display_root(self, node: Node) -> Node:
        """First node along `node` where `len(children) != 1`, or last node after max descent steps."""
        current = node
        for _ in range(self._MAX_SINGLE_CHILD_DESCENT):
            kids = list(current.children)
            if len(kids) != 1:
                return current
            current = kids[0]
        return current


    def _populate(self, node: Node, parent_item: Optional[Any]) -> None:
        node_visible = bool(getattr(node, 'visible', True))
        node_type = self._get_cell_text(column_name='Type', node=node)
        node_name = self._get_cell_text(column_name='Name', node=node)
        node_order = self._get_cell_text(column_name='Order', node=node)
        node_opacity = self._get_cell_text(column_name='Opacity', node=node)
        gl_blend_text = self._get_cell_text(column_name='GL Blend', node=node)
        transform_text = self._get_cell_text(column_name='Transform', node=node)
        item = QtWidgets.QTreeWidgetItem([node_name, node_type, '', node_order, node_opacity, gl_blend_text, transform_text])
        item.setData(0, self._user_role, node)
        base_flags = item.flags() | self._item_is_user_checkable | self._item_is_enabled
        if self._node_has_gl_blend(node):
            base_flags = base_flags | self._item_is_editable
        item.setFlags(base_flags)
        item.setCheckState(2, self._checked_state if node_visible else self._unchecked_state)
        if parent_item is None:
            self.tree.addTopLevelItem(item)
        else:
            cast(Any, parent_item).addChild(item)
        for child in list(node.children):
            self._populate(node=child, parent_item=item)


    _VALID_BLEND_PRESETS = ('opaque', 'translucent', 'additive')

    def _on_item_changed(self, item: Any, column: int) -> None:
        if self._is_rebuilding:
            return
        if column not in (2, 5):
            return
        node = item.data(0, self._user_role)
        if node is None:
            return
        if column == 2:
            is_checked = (item.checkState(2) == self._checked_state)
            try:
                node.visible = bool(is_checked)
                if self._canvas is not None:
                    self._canvas.update()
            except Exception:
                pass
        elif column == 5:
            new_text = str(item.text(5)).strip()
            if new_text not in self._VALID_BLEND_PRESETS or not self._node_has_gl_blend(node):
                return
            try:
                extra_kwargs = {k: v for k, v in node._vshare.gl_state.items() if k != 'preset'}
                node.set_gl_state(new_text, **extra_kwargs)
                if self._canvas is not None:
                    self._canvas.update()
            except Exception:
                pass



class VispyCanvasContainingWindow(PhoDockAreaContainingWindow):
    """Dock-area main window for composing vispy UIs alongside other docks.
		a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
        pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper.PhoDockAreaContainingWindow
        
        Inherited Properties: .area
    Subclasses ``PhoDockAreaContainingWindow`` (central ``DockArea``, ``.area``). Embed a
    ``SceneCanvas`` via ``canvas.native``, or a ``VispySceneWrappingWidget``, using
    ``add_display_dock`` from ``DynamicDockDisplayAreaContentMixin``. For generic dock-only
    patterns see ``DockAreaWrapper`` / ``PhoDockAreaContainingWindow``.
    """

    # ==================================================================================================================== #
    # Initializers                                                                                                         #
    # ==================================================================================================================== #

    def __init__(self, title='VispyCanvasContainingWindow', *args, **kwargs):
        super(VispyCanvasContainingWindow, self).__init__(title, *args, **kwargs)
        self.setWindowTitle(title)


class VispySceneWrappingWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Composite Qt widget: vispy ``SceneCanvas.native`` plus an optional ``VispySceneTreeWidget``.

    For embedding in a parent layout or dock (no main window or volumetric logic here). The tree
    matches ``VispySceneTreeWidget`` (visibility toggles, GL blend column, refresh). When
    ``show_scene_tree`` is True, ``column_renderers`` is forwarded to ``VispySceneTreeWidget``.

    If ``canvas`` is None, a minimal ``SceneCanvas`` with one default turntable view is created for
    standalone use. If ``canvas`` is provided, the caller must configure views and cameras; this
    widget only embeds ``canvas.native``.
    """

    def __init__(self, canvas: Optional[scene.SceneCanvas] = None, parent: Optional[Any] = None, *, show_scene_tree: bool = True, tree_on_right: bool = True, tree_minimum_width: int = 200, column_renderers: Optional[Dict[str, Callable[[Node], str]]] = None, splitter_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__(parent=parent)
        self.scene_tree_widget: Optional[VispySceneTreeWidget] = None
        if canvas is None:
            _canvas = scene.SceneCanvas(keys='interactive', show=False, size=(800, 600), autoswap=False, resizable=True, decorate=False)
            _view = _canvas.central_widget.add_view()
            _view.camera = 'turntable'
            self.canvas = _canvas
        else:
            self.canvas = canvas
        expanding = getattr(QtWidgets.QSizePolicy, 'Expanding', QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
        self.setWindowTitle('VispySceneWrappingWidget')
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        native = self.canvas.native
        if not show_scene_tree:
            outer.addWidget(cast(Any, native), stretch=1)
        else:
            self.scene_tree_widget = VispySceneTreeWidget(root_node=cast(Node, self.canvas.scene), canvas=self.canvas, parent=self, column_renderers=column_renderers)
            self.scene_tree_widget.setMinimumWidth(tree_minimum_width)
            splitter = QtWidgets.QSplitter(self)
            _qt = QtCore.Qt
            _horiz = getattr(_qt, 'Horizontal', None)
            if _horiz is None:
                _horiz = getattr(cast(Any, _qt).Orientation, 'Horizontal', 1)
            splitter.setOrientation(cast(Any, _horiz))
            first, second = (native, self.scene_tree_widget) if tree_on_right else (self.scene_tree_widget, native)
            splitter.addWidget(cast(Any, first))
            splitter.addWidget(cast(Any, second))
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)
            sizes = list(splitter_sizes) if splitter_sizes is not None else [700, 300]
            if len(sizes) == 2:
                splitter.setSizes(sizes)
            outer.addWidget(cast(Any, splitter), stretch=1)


    def rebuild(self) -> None:
        if self.scene_tree_widget is not None:
            self.scene_tree_widget.rebuild()



    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self.canvas is not None:
            QtCore.QTimer.singleShot(10, self.canvas.update)




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