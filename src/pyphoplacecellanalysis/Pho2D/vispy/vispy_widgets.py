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

@metadata_attributes(short_name=None, tags=['paginated', 'multi-decoder', 'epochs', 'widget', 'window', 'ui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-23 13:54', related_items=[])
class PhoPaginatedMultiDecoderDecodedEpochsWindow(PhoDockAreaContainingWindow):
    """ a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
        pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper.PhoDockAreaContainingWindow
        
        Inherited Properties: .area

    Usage:
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import PhoPaginatedMultiDecoderDecodedEpochsWindow

    ## Ripples:
    pagination_controller_dict =  PhoPaginatedMultiDecoderDecodedEpochsWindow._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epochs_name='ripple', included_epoch_indicies=None, defer_render=False, save_figure=False)
    app, root_dockAreaWindow = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_pagination_controller_dict(pagination_controller_dict) # Combine to a single figure
    root_dockAreaWindow.add_data_overlays(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
            
    ## Laps:
    laps_pagination_controller_dict =  PhoPaginatedMultiDecoderDecodedEpochsWindow._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epochs_name='laps', included_epoch_indicies=None, defer_render=False, save_figure=False)
    laps_app, laps_root_dockAreaWindow = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_pagination_controller_dict(laps_pagination_controller_dict) # Combine to a single figure
    laps_root_dockAreaWindow.add_data_overlays(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)

    """
    @property
    def contents(self):
        return self.ui._contents
    
    @property
    def pagination_controllers(self) -> Dict[types.DecoderName, DecodedEpochSlicesPaginatedFigureController]:
        return self.contents.pagination_controllers

    @property
    def paginator_controller_widget(self) -> PaginationControlWidget:
        """ the widget that goes left and right by pages in the bottom of the left plot. """
        assert self.isPaginatorControlWidgetBackedMode
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        paginator_controller_widget = a_controlling_pagination_controller.ui.mw.ui.paginator_controller_widget
        return paginator_controller_widget
    
    @property
    def isPaginatorControlWidgetBackedMode(self) -> bool:
        """ whether it's isPaginatorControlWidgetBackedMode """
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        return a_controlling_pagination_controller.params.isPaginatorControlWidgetBackedMode
        

    @property
    def paginated_widgets(self) -> Dict[types.DecoderName, MatplotlibTimeSynchronizedWidget]:
        """ the list of plotting child widgets. """
        return {a_decoder_name:a_pagination_controller.ui.mw for a_decoder_name, a_pagination_controller in self.contents.pagination_controllers.items()}


    @property
    def debug_print(self):
        """The debug_print property."""
        return np.all([v.params.debug_print for a_name, v in self.pagination_controllers.items()])
    @debug_print.setter
    def debug_print(self, value):
        for a_name, v in self.pagination_controllers.items():
            v.params.debug_print = value

    @property
    def figure_ctx_dict(self) -> Dict[str, IdentifyingContext]:
        """ the list of plotting child widgets. """
        return {a_name:v.params.active_identifying_figure_ctx for a_name, v in self.pagination_controllers.items()} 

        
    @property
    def global_thin_button_bar_widget(self) -> ThinButtonBarWidget:
        """The global_thin_button_bar_widget property."""
        return self.ui._contents.global_thin_button_bar_widget
    
    @property
    def global_paginator_controller_widget(self) -> PaginationControlWidget:
        """The global_thin_button_bar_widget property."""
        return self.global_thin_button_bar_widget.ui.paginator_controller_widget
    

    # Attached Widgets ___________________________________________________________________________________________________ #
    @property
    def attached_ripple_rasters_widget(self) -> Optional[RankOrderRastersDebugger]:
        """The global_thin_button_bar_widget property."""
        return self.ui.attached_ripple_rasters_widget
    
    @property
    def attached_yellow_blue_marginals_viewer_widget(self) -> Optional[DecodedEpochSlicesPaginatedFigureController]:
        """The attached_yellow_blue_marginals_viewer_widget property."""
        return self.ui.attached_yellow_blue_marginals_viewer_widget
    
    @property
    def attached_directional_template_pfs_debugger(self) -> Optional[TemplateDebugger]:
        """The global_thin_button_bar_widget property."""
        if self.ui.attached_ripple_rasters_widget is None:
            return None
        return self.ui.attached_ripple_rasters_widget.ui.controlled_references.get('directional_template_pfs_debugger', {}).get('obj', None)
 

    # Pass-through properties ____________________________________________________________________________________________ #
    @property
    def decoder_filter_epochs_decoder_result_dict(self) -> Dict[types.DecoderName, DecodedFilterEpochsResult]:
        """ each child has a `.filter_epochs_decoder_result` property """
        return self.get_children_props(prop_path='plots_data.filter_epochs_decoder_result')
 

    # ==================================================================================================================== #
    # Initializers                                                                                                         #
    # ==================================================================================================================== #

    def __init__(self, title='PhoPaginatedMultiDecoderDecodedEpochsWindow', *args, **kwargs):
        super(PhoPaginatedMultiDecoderDecodedEpochsWindow, self).__init__(*args, **kwargs)
        self.ui._contents = None
        self.ui.attached_ripple_rasters_widget = None
        self.ui.attached_yellow_blue_marginals_viewer_widget = None
        # self.highlighted_epoch_time_bin_idx = None
            
        # self.setup()
        # self.buildUI()

    @function_attributes(short_name=None, tags=['ui', 'buttons', 'button_bar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 11:06', related_items=[])
    @classmethod
    def _build_globally_controlled_pagination(cls, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict):
        """ 2024-07-31: Connects the all four plotter's pagination controls to a newly-instantiated global paginator so that they are directly driven.
        
        paginated_multi_decoder_decoded_epochs_window.ui._contents.global_thin_button_bar_widget.ui.paginator_controller_widget
        
        
        """
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget
        from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget, PaginationControlWidgetState
        # from PyQt5 import QtWidgets
        from qtpy import QtWidgets
        
        ## Gets the global bar
        global_thin_button_bar_widget: ThinButtonBarWidget = paginated_multi_decoder_decoded_epochs_window.ui._contents.global_thin_button_bar_widget

        ## Get the current page idx and things from the first pagination_controller:
        a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        # copied_paginator_controller_widget: PaginationControlWidget = a_controlling_pagination_controller.paginator_controller_widget
        curr_page_idx: int = int(a_controlling_pagination_controller.paginator_controller_widget.current_page_idx)
        n_pages: int = int(a_controlling_pagination_controller.paginator_controller_widget.get_total_pages())

        ## INPUTS: n_pages, curr_page_idx, global_thin_button_bar_widget
        ## Creates a new PaginationControlWidget for the global (bottom bar) shared context:
        global_paginator_controller_widget = PaginationControlWidget(n_pages=n_pages)
        global_paginator_controller_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        global_thin_button_bar_widget.horizontalLayout.insertWidget(0, global_paginator_controller_widget)
        global_paginator_controller_widget.setFixedHeight(21)
        ## Update the page_idx:
        # global_paginator_controller_widget.update_page_idx(updated_page_idx=curr_page_idx)
        global_paginator_controller_widget.update_page_idx(curr_page_idx) # throws a fit about positional arguments if you pass it as a kwarg
        ## assign it so that it's internal to the `global_thin_button_bar_widget`
        global_thin_button_bar_widget.ui.paginator_controller_widget = global_paginator_controller_widget

        ## all four controllers are controlled:
        controlled_pagination_controllers_list = (pagination_controller_dict['long_LR'], pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])
        new_connections_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow._perform_convert_decoder_pagination_controller_dict_to_controlled(a_controlling_pagination_controller_widget=global_paginator_controller_widget,
                                                                                                                                            controlled_pagination_controllers_list=controlled_pagination_controllers_list)

        ## Bind
        global_thin_button_bar_widget.sigLoadSelections.connect(lambda *args, **kwargs: paginated_multi_decoder_decoded_epochs_window.restore_selections_from_user_annotations())  # this only successfully works when using a lambda functiton, otherwise it raises memory access errors.

        return new_connections_dict


    @classmethod
    def init_from_pagination_controller_dict(cls, pagination_controller_dict, name = 'CombinedDirectionalDecoderDecodedEpochsWindow', title='Pho Combined Directional Decoder Decoded Epochs', defer_show=False):
        """ 2024-02-14 - Copied from `RankOrderRastersDebugger`'s approach. Merges the four separate decoded epoch windows into single figure with a separate dock for each decoder.
        [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261)

        Usage:
            app, root_dockAreaWindow, _out_dock_widgets, dock_configs = merge_single_window(pagination_controller_dict)

        """
        import inspect
        from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        
        from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget

        ## Convert to controlled first
        # new_connections_dict = cls.convert_decoder_pagination_controller_dict_to_controlled(pagination_controller_dict)

        # pagination_controller_dict = _obj.plots.rasters_display_outputs
        all_widgets = {a_decoder_name:a_pagination_controller.ui.mw for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_windows = {a_decoder_name:a_pagination_controller.ui.mw.window() for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots = {a_decoder_name:a_pagination_controller.plots for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_plots_data = {a_decoder_name:a_pagination_controller.plots_data for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        all_separate_params = {a_decoder_name:a_pagination_controller.params for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}

        main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        
        # all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # # Extract the plot/renderable items
        # all_separate_root_plots = {a_decoder_name:a_pagination_controller.plots.root_plot for a_decoder_name, a_pagination_controller in pagination_controller_dict.items()}
        # all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}
        # all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in pagination_controller_dict.items()}

        # Embedding in docks:
        
        # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Combined Directional Decoder Decoded Epochs')
        
        # Instantiate the class ______________________________________________________________________________________________ #
        # root_dockAreaWindow = PhoDockAreaContainingWindow(title=title)
        root_dockAreaWindow = cls(title=title)
        root_dockAreaWindow.setWindowTitle(f'{title}: dockAreaWindow')
        app = root_dockAreaWindow.app
        
        icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/paginated_multi_decoder_decoded_epochs.ico")
        if icon is not None:
            root_dockAreaWindow.setWindowIcon(icon)

        ## Build Dock Widgets:
        _out_dock_widgets = {}
        dock_configs = dict(zip(('long_LR', 'long_RL',
                                 'short_LR', 'short_RL'),
                            (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                            CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))

        for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(430,780), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name], autoOrientation=False)

        ## Enable a global (ThinButtonBarWidget) footer widget spanning across the entire bottom of the window:
        utility_footer_name: str = 'Utility'
        dock_configs[utility_footer_name] = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)        

        global_thin_button_bar_widget: ThinButtonBarWidget = ThinButtonBarWidget()
        global_thin_button_bar_widget.setObjectName("global_thin_button_bar_widget")
        global_thin_button_bar_widget.setFixedHeight(21)
        global_thin_button_bar_widget.label_message = "<shared>"
        
        possible_mouse_actions_dict: Dict[str, Callable] = {method[0]:method[1] for method in inspect.getmembers(ClickActionCallbacks, predicate=inspect.isfunction)}
        # root_dockAreaWindow.update_params(possible_mouse_actions_dict = possible_mouse_actions_dict, # { 'copy_axis_image_to_clipboard_callback': ClickActionCallbacks.copy_axis_image_to_clipboard_callback, 'copy_click_time_to_clipboard_callback': ClickActionCallbacks.copy_click_time_to_clipboard_callback, 'copy_epoch_times_to_clipboard_callback': ClickActionCallbacks.copy_epoch_times_to_clipboard_callback, 'log_clicked_epoch_times_to_message_box_callback': ClickActionCallbacks.log_clicked_epoch_times_to_message_box_callback },
        # )
        
        _out_dock_widgets[utility_footer_name] = root_dockAreaWindow.add_display_dock(identifier=utility_footer_name, widget=global_thin_button_bar_widget, dockSize=(1200, 30), dockAddLocationOpts=['bottom'], display_config=dock_configs[utility_footer_name], autoOrientation=False)

        # ## Build final .plots and .plots_data:
        root_dockAreaWindow.ui._contents = PhoUIContainer(name=name, names=main_plot_identifiers_list, pagination_controllers=pagination_controller_dict, 
                                                    dock_widgets=_out_dock_widgets, dock_configs=dock_configs,
                                                    widgets=all_widgets, windows=all_windows, plots=all_separate_plots, plots_data=all_separate_plots_data, params=all_separate_params,
                                                    global_thin_button_bar_widget=global_thin_button_bar_widget,
                                                    possible_mouse_actions_dict=possible_mouse_actions_dict) # do I need this extracted data or is it redundant?
        

        ## Convert to controlled by global paginator:
        new_connections_dict = cls._build_globally_controlled_pagination(paginated_multi_decoder_decoded_epochs_window=root_dockAreaWindow, pagination_controller_dict=pagination_controller_dict)
        root_dockAreaWindow._init_UI_additional_mouse_click_action_selection_combos()

        # Add functions ______________________________________________________________________________________________________ #

        root_dockAreaWindow.ui.print = print

        if not defer_show:
            root_dockAreaWindow.show()
            
        return app, root_dockAreaWindow


    @function_attributes(short_name=None, tags=['settings', 'ui', 'private'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-11 20:20', related_items=[])
    def _init_UI_additional_mouse_click_action_selection_combos(self, debug_print: bool=False):
        """ adds 3 GUI controls for selecting the desired mouse button actions - 3 separate controls to be added to a thin horizontal toolbar: 1. allows choosing the LMB action from a list `methods_list` of 4 items, 2. allows choosing the MMB action from the same list, 3. allows choosing the RMB action from the same list. All 3 controls should be independent.
        
        """
        # from qtpy import QtCore, QtWidgets
        from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtWidgets
                
        def update_callbacks(selected_combo_box_text: str, action_key: str):
            """ captures self
            """
            methods_list = list(self.ui._contents.possible_mouse_actions_dict.keys())
            possible_mouse_actions_dict = self.ui._contents.possible_mouse_actions_dict
            assert selected_combo_box_text in possible_mouse_actions_dict
            selected_callback_fn = possible_mouse_actions_dict[selected_combo_box_text]

            on_click_item_callbacks_dict = self.get_children_props(prop_path=action_key) # 'params.on_middle_click_item_callbacks'
            
            for a_name, a_callback_dict in on_click_item_callbacks_dict.items():
                if debug_print:
                    print(f'a_name: {a_name}')
                ## remove old method names
                for old_method_name in methods_list:
                    a_callback_dict.pop(old_method_name, None)
        
                a_callback_dict[selected_combo_box_text] = selected_callback_fn
                # a_callback_dict.update(selected_combo_box_text=selected_callback_fn)
                
            if debug_print:
                print(f'action_key: {action_key} - new action: {selected_combo_box_text}')
            
            # if not hasattr(self.params, 'on_left_click_item_callbacks'):
            #     self.params.params['on_left_click_item_callbacks'] = {}
            # if not hasattr(self.params, 'on_middle_click_item_callbacks'):
            #     self.params.params['on_middle_click_item_callbacks'] = {}
            # if not hasattr(self.params, 'on_secondary_click_item_callbacks'):
            #     self.params.params['on_secondary_click_item_callbacks'] = {}

            # getattr(self.params.params, action_key)[selected_text] = selected_callback

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        # parent_bar_add_widget_fn = lambda x: global_thin_button_bar_widget.addWidget(x)
        parent_bar_add_widget_fn = lambda x: global_thin_button_bar_widget.horizontalLayout.insertWidget(-1, x)
        
        methods_list = list(self.ui._contents.possible_mouse_actions_dict.keys())
        global_thin_button_bar_widget = self.global_thin_button_bar_widget
        
        # # Create and add the left mouse button (LMB) combo box
        # self.ui._contents.lmb_action_combo = QtWidgets.QComboBox()
        # self.ui._contents.lmb_action_combo.addItems(methods_list)
        # lmb_action = QtGui.QAction("LMB Action", self)
        # global_thin_button_bar_widget.addAction(lmb_action)
        # global_thin_button_bar_widget.horizontalLayout.insertWidget(2, self.ui._contents.lmb_action_combo)
        # self.ui._contents.lmb_action_combo.currentIndexChanged.connect(lambda selected_index: update_callbacks(self.ui._contents.lmb_action_combo.itemText(selected_index), 'params.on_left_click_item_callbacks'))
        # self.ui._contents.lmb_action_combo.setCurrentIndex(methods_list.index('copy_click_time_to_clipboard_callback'))
        
        # Create and add the middle mouse button (MMB) combo box
        self.ui._contents.mmb_action_combo = QtWidgets.QComboBox()
        self.ui._contents.mmb_action_combo.addItems(methods_list)
        mmb_action = QtGui.QAction("MMB Action", self)
        global_thin_button_bar_widget.addAction(mmb_action)
        global_thin_button_bar_widget.horizontalLayout.insertWidget(2, self.ui._contents.mmb_action_combo)
        self.ui._contents.mmb_action_combo.currentIndexChanged.connect(lambda selected_index: update_callbacks(self.ui._contents.mmb_action_combo.itemText(selected_index), 'params.on_middle_click_item_callbacks'))
        self.ui._contents.mmb_action_combo.setCurrentIndex(methods_list.index('copy_axis_image_to_clipboard_callback')) # copy_click_time_to_clipboard_callback # Set default index for MMB combo box


        # Create and add the right mouse button (RMB) combo box
        self.ui._contents.rmb_action_combo = QtWidgets.QComboBox()
        self.ui._contents.rmb_action_combo.addItems(methods_list)
        rmb_action = QtGui.QAction("RMB Action", self)
        global_thin_button_bar_widget.addAction(rmb_action)
        global_thin_button_bar_widget.horizontalLayout.insertWidget(3, self.ui._contents.rmb_action_combo)
        self.ui._contents.rmb_action_combo.currentIndexChanged.connect(lambda selected_index: update_callbacks(self.ui._contents.rmb_action_combo.itemText(selected_index), 'params.on_secondary_click_item_callbacks'))
        self.ui._contents.rmb_action_combo.setCurrentIndex(methods_list.index('log_clicked_epoch_times_to_message_box_callback')) # Set default index for RMB combo box



    @classmethod
    def init_from_track_templates(cls, curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict, epochs_name:str ='laps', included_epoch_indicies=None,
                                   name='CombinedDirectionalDecoderDecodedEpochsWindow', title='Pho Combined Directional Decoder Decoded Epochs', defer_show=False, **kwargs):
        """ 2024-02-28 - Combines the previously separate ._subfn_prepare_plot_multi_decoders_stacked_epoch_slices +  .init_from_pagination_controller_dict approaches. 
        Usage:
            ## Example 1 Ripples:
            app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epochs_name='ripple', included_epoch_indicies=None)

            ## Example 2 Laps:
            laps_app, laps_paginated_multi_decoder_decoded_epochs_window, laps_pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epochs_name='laps', included_epoch_indicies=None)

        """
        # 'enable_update_window_title_on_page_change'
        pagination_controller_dict =  cls._subfn_prepare_plot_multi_decoders_stacked_epoch_slices(curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict=decoder_decoded_epochs_result_dict, epochs_name=epochs_name, included_epoch_indicies=included_epoch_indicies, defer_render=True, save_figure=False, **kwargs)
        app, paginated_multi_decoder_decoded_epochs_window = cls.init_from_pagination_controller_dict(pagination_controller_dict, name=name, title=title, defer_show=defer_show) # Combine to a single figure
        build_extra_programmatic_buttons(paginated_multi_decoder_decoded_epochs_window)
    
        # paginated_multi_decoder_decoded_epochs_window.params['track_length_cm_dict'] = track_templates.get_track_length_dict()
        return app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict
    

    ## Add a jump to page function
    def jump_to_page(self, page_idx: int):
        if self.isPaginatorControlWidgetBackedMode:
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            # updates the embedded pagination widget
            # self.paginator_controller_widget.programmatically_update_page_idx(page_idx, block_signals=False) # don't block signals and then we don't have to call updates.
            self.paginator_controller_widget.programmatically_update_page_idx(page_idx, block_signals=True) # don't block signals and then we don't have to call updates.
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            #TODO 2024-03-06 08:16: - [ ] If we add a footer pagination widget to the window we would update it here.
            pass
        
        ## Call programmatically_update_page_idx on the children
        for a_name, a_paginated_controller in self.pagination_controllers.items():
            a_paginated_controller.programmatically_update_page_idx(updated_page_idx=page_idx, block_signals=False) # should ensure a_paginated_controller.current_page_idx is updated
            assert (a_paginated_controller.current_page_idx == page_idx), f"a_paginated_controller.current_page_idx: {a_paginated_controller.current_page_idx} does not equal the desired page index: {page_idx}"
            a_paginated_controller.perform_update_selections(defer_render=False) # update selections
            
        if self.attached_ripple_rasters_widget is not None:
            self.attached_ripple_rasters_widget.clear_highlighting_indicator_regions()
            
        if self.attached_directional_template_pfs_debugger is not None:
            self.attached_directional_template_pfs_debugger.reset_cell_emphasis()

        self.draw()
        
        
    @function_attributes(short_name=None, tags=['data-overlays', 'add'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-12 00:00', related_items=['remove_data_overlays'])
    def add_data_overlays(self, included_columns=None, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data and adds them to the plot.
        
        REFINEMENT: note that it only plots either 'laps' or 'ripple', not both, so it doesn't need all this data.
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import WeightedCorrelationPaginatedPlotDataProvider
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DecodedPositionsPlotDataProvider, DecodedSequenceAndHeuristicsPlotDataProvider
        
        ## Choose which columns from the filter_epochs dataframe to include on the plot.
        if included_columns is None:
            included_columns = []

        decoder_decoded_epochs_result_dict = deepcopy(self.decoder_filter_epochs_decoder_result_dict)
        
        ## Add the overlays to each of the four figures:
        for a_name, a_pagination_controller in self.pagination_controllers.items():          
            # a_pagination_controller.params.xbin 
            a_pagination_controller.add_data_overlays(decoder_decoded_epochs_result=decoder_decoded_epochs_result_dict[a_name], included_columns=included_columns, defer_refresh=True)

        if not defer_refresh:
            self.refresh_current_page()
            


    @function_attributes(short_name=None, tags=['data-overlays', 'remove'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-12 00:00', related_items=['add_data_overlays'])
    def remove_data_overlays(self, defer_refresh=False):
        """ builds the Radon Transforms and Weighted Correlation data for this decoder and adds them to the plot.
        """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.remove_data_overlays(defer_refresh=True)

        if not defer_refresh:
            self.refresh_current_page()





    @classmethod
    def _subfn_prepare_plot_multi_decoders_stacked_epoch_slices(cls, curr_active_pipeline, track_templates, decoder_decoded_epochs_result_dict: Dict[str, DecodedFilterEpochsResult], epochs_name:str ='laps', included_epoch_indicies=None, defer_render=True, save_figure=True, **kwargs):
        """ 2024-02-14 - Adapted from the function that plots the Long/Short decoded epochs side-by-side for comparsion and updated to work with the multi-decoder track templates.
        
        ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
        
        Usage:
            (pagination_controller_L, pagination_controller_S), (fig_L, fig_S), (ax_L, ax_S), (final_context_L, final_context_S), (active_out_figure_paths_L, active_out_figure_paths_S) = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(curr_active_pipeline, defer_render=False)
        """
        debug_print = kwargs.get('debug_print', False)

        ## Extract params_kwargs
        params_kwargs = kwargs.pop('params_kwargs', {})
        params_kwargs = dict(skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, isPaginatorControlWidgetBackedMode=True, epochs_name=epochs_name) | params_kwargs
        # print(f'params_kwargs: {params_kwargs}')
        max_subplots_per_page: int = kwargs.pop('max_subplots_per_page', params_kwargs.pop('max_subplots_per_page', 8)) # kwargs overrides params_kwargs
        
        decoder_names: List[str] = track_templates.get_decoder_names()
        
        track_length_dict = track_templates.get_track_length_dict()
        
        controlling_pagination_item_name: str = decoder_names[0] # first item # 'long_LR'
        # controlled_pagination_controller_names_list = decoder_names[1:]
        pagination_controller_dict = {}
        for i, (a_name, a_decoder) in enumerate(track_templates.get_decoders_dict().items()):
            is_controlling_widget: bool = (a_name == controlling_pagination_item_name)

            curr_params_kwargs = deepcopy(params_kwargs)
            curr_params_kwargs['is_controlled_widget'] = (not is_controlling_widget)
            if ('disable_y_label' not in curr_params_kwargs):
                # If user didn't provide an explicit 'disable_y_label' option, use the defaults which is to hide labels on all the but the controlling widget
                if is_controlling_widget:
                    curr_params_kwargs['disable_y_label'] = False
                else:
                    curr_params_kwargs['disable_y_label'] = True
                    
            curr_params_kwargs['track_length_cm'] = track_length_dict[a_name] ## get the track length in cm

            # a_name: str = 
            a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_decoded_epochs_result_dict[a_name] # DecodedFilterEpochsResult
            pagination_controller_dict[a_name] = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(a_decoder_decoded_epochs_result.filter_epochs,
                                                                                                filter_epochs_decoder_result=a_decoder_decoded_epochs_result,
                                                                                                xbin=a_decoder.xbin, global_pos_df=curr_active_pipeline.sess.position.df,
                                                                                                a_name=f'DecodedEpochSlices[{a_name}]', active_context=curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name),
                                                                                                max_subplots_per_page=max_subplots_per_page, debug_print=debug_print, included_epoch_indicies=included_epoch_indicies, params_kwargs=curr_params_kwargs) # , save_figure=save_figure


        # Constrains each of the plotters at least to the minimum height:
        for a_name, a_pagination_controller in pagination_controller_dict.items():
            # a_pagination_controller.params.all_plots_height
            # resize to minimum height
            a_widget = a_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
            screen = a_widget.screen()
            screen_size = screen.size()

            target_height = a_pagination_controller.params.get('scrollAreaContents_MinimumHeight', None)
            if target_height is None:
                target_height = (a_pagination_controller.params.all_plots_height + 30)
            desired_final_height = int(min(target_height, screen_size.height())) # don't allow the height to exceed the screen height.
            print(f'target_height: {target_height}, {  desired_final_height = }')
            # a_widget.size()
            a_widget.setMinimumHeight(desired_final_height) # the 30 is for the control bar

        return pagination_controller_dict


    @classmethod
    def _perform_convert_decoder_pagination_controller_dict_to_controlled(cls, a_controlling_pagination_controller_widget: PaginationControlWidget, controlled_pagination_controllers_list):
        """
        
        """
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget
        
        new_connections_dict = []

        for a_controlled_pagination_controller in controlled_pagination_controllers_list:
            # hide the pagination widget:
            a_controlled_widget = a_controlled_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget

            if a_controlled_pagination_controller.params.get('isPaginatorControlWidgetBackedMode', True):
                # a_controlled_widget.on_paginator_control_widget_jump_to_page(page_idx=0)
                a_connection = a_controlling_pagination_controller_widget.jump_to_page.connect(a_controlled_pagination_controller.paginator_controller_widget.update_page_idx)
                new_connections_dict.append(a_connection)
                # a_controlled_widget.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection
                a_controlled_widget.ui.paginator_controller_widget.hide()

                ## Enable a equally sized (ThinButtonBarWidget) placeholder widget instead:
                a_controlled_widget.ui.thin_button_bar_widget = ThinButtonBarWidget()
                a_controlled_widget.ui.root_vbox.addWidget(a_controlled_widget.ui.thin_button_bar_widget) # add the pagination control widget
                a_controlled_widget.ui.thin_button_bar_widget.setFixedHeight(21)
                
                a_controlled_widget.ui.thin_button_bar_widget.label_message = "<controlled>"

                ## Build connections to buttons:
                # a_controlled_widget.ui.thin_button_bar_widget.sigCopySelections.connect() # TODO
                # a_controlled_widget.ui.thin_button_bar_widget.sigLoadSelections.connect(lambda: a_controlled_pagination_controller.restore_selections_from_user_annotations())  # this only successfully works when using a lambda functiton, otherwise it raises memory access errors.

        return new_connections_dict
    

    @classmethod
    def convert_decoder_pagination_controller_dict_to_controlled(cls, pagination_controller_dict):
        """
        
        """
        ## Connects the first plotter's pagination controls to the other three controllers so that they are directly driven, by the first.
        a_controlling_pagination_controller = pagination_controller_dict['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        # a_controlling_widget = a_controlling_pagination_controller.ui.mw # MatplotlibTimeSynchronizedWidget
        a_controlling_pagination_controller_widget: PaginationControlWidget = a_controlling_pagination_controller.paginator_controller_widget
        
        # controlled widgets
        controlled_pagination_controllers_list = (pagination_controller_dict['long_RL'], pagination_controller_dict['short_LR'], pagination_controller_dict['short_RL'])

        return cls._perform_convert_decoder_pagination_controller_dict_to_controlled(a_controlling_pagination_controller_widget=a_controlling_pagination_controller_widget, controlled_pagination_controllers_list=controlled_pagination_controllers_list)


    ## ==================================================================================================================== #
    #region Selections/Annotations                                                                                               
    # ==================================================================================================================== #
        
    # User Selections/Annotations ________________________________________________________________________________________ #
    def save_selections(self) -> Dict[str, EpochSelectionsObject]:
        """ Capture current user selections for each child controller 
        Usage:
            saved_selections_dict: Dict[str, SelectionsObject] = self.save_selections()
        """
        saved_selections_dict: Dict[str, EpochSelectionsObject] = {a_name:a_ctrlr.save_selection() for a_name, a_ctrlr in self.pagination_controllers.items()}
        return saved_selections_dict



    def print_user_annotations(self, should_copy_to_clipboard=True, use_new_concise_nested_context_format = True):
        """ Builds user annotations and outputs them. 

        use_new_concise_nested_context_format = True # 2024-03-04 - Concise 

        >>> Prints Output Like:
        Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_LR',user_annotation='selections')] = np.array([])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_RL',user_annotation='selections')] = np.array([array([120.645, 120.862]), array([169.956, 170.16])])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_LR',user_annotation='selections')] = np.array([array([105.4, 105.563]), array([125.06, 125.21]), array([132.511, 132.791]), array([149.959, 150.254]), array([169.956, 170.16])])
            user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_RL',user_annotation='selections')] = np.array([array([125.06, 125.21]), array([149.959, 150.254])])
        
        """
        def _subfn_listify(arr):
            return [list(a) for a in arr]
        
        def _sub_subfn_wrapped_in_brackets(s: str, bracket_strings = ("[", "]")) -> str:
                return bracket_strings[0] + s + bracket_strings[1]
            
        def _sub_subfn_format_nested_list(arr, precision:int=3, num_sep=", ", array_sep=', ') -> str:
            """
            Converts a nested list of floats into a single string,
            with each float formatted to the specified precision.
            
            arr = np.array([[491.798, 492.178], [940.016, 940.219]])
            _sub_subfn_format_nested_list(arr)

            >> '[[491.798, 492.178], [940.016, 940.219]]'

            arr = np.array([[785.738, 785.923]])
            _sub_subfn_format_nested_list(arr)
            >> '[[785.738, 785.923]]'
            """
            return _sub_subfn_wrapped_in_brackets(array_sep.join([_sub_subfn_wrapped_in_brackets(num_sep.join([f"{num:.{precision}f}" for num in row])) for row in arr]))
            


        def _subfn_build_new_nested_context_str(common_context, user_annotations):
            """ Builds a nested hierarchy of annotations like:
                with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
                    with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                        user_annotations[ctx + Ctx(decoder='long_results_obj')] = [5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123]
                        user_annotations[ctx + Ctx(decoder='short_results_obj')] = [ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131]
                    with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                        user_annotations[ctx + Ctx(decoder='long_LR')] = [[292.624, 292.808], [304.44, 304.656], [380.746, 380.904], [873.001, 873.269], [953.942, 954.258], [2212.47, 2212.54], [2214.24, 2214.44], [2214.65, 2214.68], [2219.73, 2219.87], [2422.6, 2422.82], [2451.06, 2451.23], [2452.07, 2452.22], [2453.38, 2453.55], [2470.82, 2470.97], [2473, 2473.15]]
                        user_annotations[ctx + Ctx(decoder='long_RL')] = [[487.205, 487.451], [518.52, 518.992], [802.912, 803.114], [803.592, 803.901], [804.192, 804.338], [831.621, 831.91], [893.989, 894.103], [982.605, 982.909], [1034.82, 1034.86], [1035.12, 1035.31], [1200.7, 1200.9], [1273.35, 1273.54], [1274.12, 1274.44], [1380.75, 1380.89], [1448.17, 1448.34], [1746.25, 1746.43], [1871, 1871.22], [2050.89, 2050.99], [2051.25, 2051.68]]
                        user_annotations[ctx + Ctx(decoder='short_LR')] = [[876.27, 876.452], [950.183, 950.448], [953.942, 954.258], [1044.95, 1045.45], [1129.65, 1129.84], [1259.29, 1259.44], [1259.72, 1259.88], [1511.2, 1511.43], [1511.97, 1512.06], [1549.24, 1549.37], [1558.47, 1558.68], [1560.66, 1560.75], [1561.31, 1561.41], [1561.82, 1561.89], [1655.99, 1656.21], [1730.89, 1731.07], [1734.81, 1734.95], [1861.41, 1861.53], [1909.78, 1910.04], [1967.74, 1968.09], [2036.97, 2037.33], [2038.03, 2038.27], [2038.53, 2038.73], [2042.39, 2042.64], [2070.82, 2071.03], [2153.03, 2153.14], [2191.26, 2191.39], [2192.12, 2192.36], [2193.78, 2193.99], [2194.56, 2194.76], [2200.65, 2200.8], [2201.85, 2202.03], [2219.73, 2219.87], [2248.61, 2248.81], [2249.7, 2249.92], [2313.89, 2314.06], [2422.6, 2422.82], [2462.67, 2462.74], [2482.13, 2482.61], [2484.41, 2484.48], [2530.72, 2530.92], [2531.22, 2531.3], [2556.11, 2556.38], [2556.6, 2556.92]]
                        user_annotations[ctx + Ctx(decoder='short_RL')] = [[66.6616, 66.779], [888.227, 888.465], [890.87, 891.037], [910.571, 911.048], [1014.1, 1014.28], [1200.7, 1200.9], [1211.21, 1211.33], [1214.61, 1214.83], [1317.71, 1318.22], [1333.49, 1333.69], [1380.75, 1380.89], [1381.96, 1382.32], [1448.17, 1448.34], [1499.59, 1499.71], [1744.34, 1744.59], [1798.64, 1798.77], [1970.81, 1970.95], [1994.07, 1994.25], [2050.89, 2050.99], [2051.25, 2051.68], [2132.66, 2132.98], [2203.73, 2203.82], [2204.54, 2204.66], [2317.03, 2317.12], [2330.01, 2330.16], [2331.84, 2331.96], [2403.11, 2403.41], [2456.24, 2456.33], [2456.47, 2456.57], [2457.49, 2458.01]]

            """
            def _indent_str(an_indent_level: int) -> str:
                return "\t" * an_indent_level
            
            def _with_block_template(an_indent_level: int, ctxt):
                # global indent_level
                return f"{_indent_str(an_indent_level)}with {ctxt.get_initialization_code_string(class_name_override='Ctx')} as ctx:"
            def _sub_ctxt_block_template(an_indent_level: int, ctxt):
                # global indent_level
                # indent_level = indent_level + 1
                return f"{_indent_str(an_indent_level)}with (ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}) as ctx:"
            def _leaf_ctxt_assignment_template(an_indent_level: int, ctxt, value):
                # indent_level = indent_level + 1
                return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {value}"
                # return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {list(value)}"
                # return f"{_indent_str(an_indent_level)}user_annotations[ctx + {ctxt.get_initialization_code_string(class_name_override='Ctx')}] = {_sub_subfn_format_nested_list(value)}"
            
            
            indent_level: int = 0
            code_strs: List[str] = []
            code_strs.append(_with_block_template(indent_level, common_context))
            indent_level = indent_level + 1
            common_context_user_annotations = IdentifyingContext.converting_to_relative_contexts(common_context, user_annotations)
            for k, v in common_context_user_annotations.items():
                code_strs.append(_leaf_ctxt_assignment_template(indent_level, k, v))

            # code_str = code_str + '\n'.join(code_strs)
            return code_strs


        if should_copy_to_clipboard:
            from pyphocorehelpers.programming_helpers import copy_to_clipboard
            
        from neuropy.core.user_annotations import UserAnnotationsManager
        annotations_man = UserAnnotationsManager()
        user_annotations = annotations_man.get_user_annotations()
        saved_selections_dict: Dict[str, SelectionsObject] = self.save_selections()
        saved_selections_context_dict = {a_name:v.figure_ctx.adding_context_if_missing(user_annotation='selections') for a_name, v in saved_selections_dict.items()}
        
        for a_name, a_saved_selection in saved_selections_dict.items():
            a_context = saved_selections_context_dict[a_name]
            # user_annotations[a_context] = a_saved_selection.flat_all_data_indicies[a_saved_selection.is_selected]
            user_annotations[a_context] = a_saved_selection.epoch_times

        # Updates the context. Needs to generate the code.

        # ## Generate code to insert int user_annotations:
        self.ui.print('Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:')

        if use_new_concise_nested_context_format:
            # Post 2024-03-04 method of nested strings:
            # active_annotations_dict = {a_context:user_annotations[a_context] for a_name, a_context in saved_selections_context_dict.items()}
            # active_annotations_dict = {a_context:saved_selections_dict[a_name].epoch_times for a_name, a_context in saved_selections_context_dict.items()}
            active_annotations_dict = {a_context:_sub_subfn_format_nested_list(saved_selections_dict[a_name].epoch_times) for a_name, a_context in saved_selections_context_dict.items()} # active_annotations_strs_dict
            common_context = IdentifyingContext.find_longest_common_context(active_annotations_dict)
            code_strings: List[str] = _subfn_build_new_nested_context_str(common_context, user_annotations=active_annotations_dict)

        else:
            # Pre 2024-03-04 method of explicit string representations:
            code_strings: List[str] = []
            for a_name, a_saved_selection in saved_selections_dict.items():
                a_context = saved_selections_context_dict[a_name]
                if use_new_concise_nested_context_format:
                    pass
                else:
                    # a_string = f"user_annotations[{a_context.get_initialization_code_string()}] = {a_saved_selection.epoch_times}"
                    # a_string = f"user_annotations[{a_context.get_initialization_code_string()}] = array({list(a_saved_selection.epoch_times)})"
                    a_string = f"user_annotations[{a_context.get_initialization_code_string(class_name_override='Ctx')}] = {list(a_saved_selection.epoch_times)}"

                code_strings.append(a_string)
                # print(a_string)
        

        code_string: str = '\n'.join(code_strings)
        code_string = f"\n{code_string}\n" # make it easier to copy by adding newlines before and after it

        if should_copy_to_clipboard:
            copy_to_clipboard(code_string, message_print=True)
        else:
            self.ui.print(code_string)
        return code_strings
    

    def restore_selections_from_user_annotations(self, user_annotations: Optional[Dict]=None, defer_render:bool=False, **additional_selections_context):
        """
        # , source='pho_algo'
        , source='diba_evt_file' # source='diba_evt_file': # gets the annotations for the kdiba-evt file exported ripples, consistent with his 2009 paper

        """
        if user_annotations is None:
            from neuropy.core.user_annotations import UserAnnotationsManager
            annotations_man = UserAnnotationsManager()
            user_annotations = annotations_man.get_user_annotations()
        
        # Uses: paginated_multi_decoder_decoded_epochs_window, user_annotations
        # figure_ctx_dict = {a_name:v.params.active_identifying_figure_ctx for a_name, v in self.pagination_controllers.items()} 
        # figure_ctx_dict = self.figure_ctx_dict
        # loaded_selections_context_dict = {a_name:a_figure_ctx.adding_context_if_missing(user_annotation='selections', **additional_selections_context) for a_name, a_figure_ctx in figure_ctx_dict.items()}
        # loaded_selections_dict = {a_name:user_annotations.get(a_selections_ctx, None) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}

        new_selections_dict = {a_decoder_name:a_pagination_controller.restore_selections_from_user_annotations(user_annotations, defer_render=defer_render, **additional_selections_context) for a_decoder_name, a_pagination_controller in self.pagination_controllers.items()}
        

        enable_all_row_selection_sync: bool = True
        if enable_all_row_selection_sync:
            children_is_epoch_selected: NDArray = np.vstack([deepcopy(a_pagination_controller.is_selected) for a_name, a_pagination_controller in self.pagination_controllers.items()]) #.shape (4, 136)
            any_child_epoch_is_selected: NDArray = np.any(children_is_epoch_selected, axis=0) # (136,)
            ## assign to all 
            for a_decoder_name, a_pagination_controller in self.pagination_controllers.items():
                # a_pagination_controller.is_selected = deepcopy(any_child_epoch_is_selected) ## make it independent  # params.update(**updated_values)
                # a_pagination_controller.params.is_selected

                # Replace values in the dictionary with new_values
                Assert.same_length(a_pagination_controller.params.is_selected, any_child_epoch_is_selected)
                a_pagination_controller.params.is_selected = {k: v for k, v in zip(a_pagination_controller.params.is_selected.keys(), any_child_epoch_is_selected)}
                a_pagination_controller.perform_update_selections(defer_render=False)
                

        # self.draw()
        return new_selections_dict
    

    @property
    def any_good_selected_epoch_times(self) -> NDArray:
        """ returns the selected epoch times for any of the self.pagination_controllers 
        """
        concatenated_selected_epoch_times = np.concatenate([a_ctrlr.selected_epoch_times for a_name, a_ctrlr in self.pagination_controllers.items()], axis=0)
        any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
        return any_good_selected_epoch_times


    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray) -> NDArray:
        """ returns the matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        All the self.pagination_controllers should be displaying the same epochs, so searching each controller for the times should result in the same returned indicies.

        Uses:
            self.pagination_controllers
        """
        from pyphocorehelpers.indexing_helpers import NumpyHelpers
        any_good_epoch_idxs_list = [a_ctrlr.find_data_indicies_from_epoch_times(epoch_times) for a_name, a_ctrlr in self.pagination_controllers.items()]
        assert NumpyHelpers.all_array_equal(any_good_epoch_idxs_list), f"all indicies should be identical, but they are not! any_good_epoch_idxs_list: {any_good_epoch_idxs_list}"
        any_good_epoch_idxs: NDArray = any_good_epoch_idxs_list[0]
        return any_good_epoch_idxs

    #endregion Selections/Annotations ______________________________________________________________________________________________________ #


    ## ==================================================================================================================== #
    #region Export/Output                                                                                              
    # ==================================================================================================================== #
    
    # Export/Output ______________________________________________________________________________________________________ #
    @function_attributes(short_name=None, tags=['export'], input_requires=[], output_provides=[], uses=[], used_by=['export_all_pages'], creation_date='2024-08-13 13:05', related_items=[])
    def export_decoder_pagination_controller_figure_page(self, curr_active_pipeline, **kwargs):
        """ exports each pages single-decoder figures separately

        Usage:
            export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

        """
        import matplotlib as mpl
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper

        output_figure_kwargs = dict(write_vector_format=True, write_png=True) | kwargs
        pagination_controller_dict = self.pagination_controllers

        out_fig_paths_dict = {}

        for a_name, a_pagination_controller in pagination_controller_dict.items():
            display_context = a_pagination_controller.params.get('active_identifying_figure_ctx', IdentifyingContext())

            # Get context for current page of items:
            current_page_idx: int = int(a_pagination_controller.current_page_idx)
            a_paginator = a_pagination_controller.paginator
            total_num_pages = int(a_paginator.num_pages)
            page_context = display_context.overwriting_context(page=current_page_idx, num_pages=total_num_pages)
            self.ui.print(page_context)

            ## Get the figure/axes:
            a_plots = a_pagination_controller.plots # RenderPlots
            # a_params = a_pagination_controller.params
            
            with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs({'figure.figsize': (16.8, 4.8), 'figure.dpi': '420', })):
                figs = a_plots.fig
                # axs = a_plots.axs
                active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=page_context, fig=figs, **output_figure_kwargs)
                out_fig_paths_dict[final_context] = active_out_figure_paths

        # end for

        return out_fig_paths_dict
    

    @function_attributes(short_name=None, tags=['export', 'combine', 'image'], input_requires=[], output_provides=[], uses=[], used_by=['export_all_pages'], creation_date='2024-12-17 10:32', related_items=[])
    @classmethod
    def build_combined_all_pages_image(cls, out_fig_paths_dict_list: Dict, combined_image_basename: str = 'combined'):
        """ builds a concatenated image from the individually exported decoded epochs produced by `out_fig_paths_dict_list = export_all_pages(...)` """
        from PIL import Image
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack, image_grid

        _flat_png_list = []
        _flat_pdf_list = []
        
        for a_page_idx, a_page_out_dict in out_fig_paths_dict_list.items():
            print(f'page[{a_page_idx}]: a_page_out_dict: {a_page_out_dict}')
            for a_final_context, an_output_list in a_page_out_dict.items():
                # final_context
                _flat_pdf_list.append(an_output_list[0]) ## pdf
                _flat_png_list.append(an_output_list[1]) ## png

        ## handle PNGs                    
        out_parent_path = _flat_png_list[0].parent.resolve()
        _flat_raster_imgs = [Image.open(i) for i in _flat_png_list] ## open the images from disk
        split_list = [_flat_raster_imgs[i:i + 4] for i in range(0, len(_flat_raster_imgs), 4)]
        _out_combined_img = vertical_image_stack([horizontal_image_stack(a_row, padding=0) for a_row in split_list], padding=4)

        combined_img_out_path = out_parent_path.joinpath(f'{combined_image_basename}.png')
        _out_combined_img.save(combined_img_out_path)
        
        return (_flat_png_list, _out_combined_img, combined_img_out_path)
    

    @function_attributes(short_name=None, tags=['DEPRIcATED', 'export'], input_requires=[], output_provides=[], uses=['export_decoder_pagination_controller_figure_page', 'build_combined_all_pages_image'], used_by=[], creation_date='2024-08-13 13:05', related_items=[])
    def export_all_pages(self, curr_active_pipeline, write_vector_format=True, write_png=True, enable_export_combined_img: bool=False, combined_image_basename: str = 'combined', **kwargs):
        """ exports each pages single-decoder figures separately (as a page with stacked decoders). Does NOT export the rasters or marginals.

        Usage:
            export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

            _out_paths, (_out_combined_img, combined_img_out_path) = paginated_multi_decoder_decoded_epochs_window.export_all_pages(curr_active_pipeline, enable_export_combined_img=True, combined_image_basename=f'{DAY_DATE_TO_USE}_combined_All_Epochs')
            
        """
        output_figure_kwargs = dict(write_vector_format=write_vector_format, write_png=write_png) | kwargs

        # assert self.isPaginatorControlWidgetBackedMode
        a_controlling_pagination_controller = self.contents.pagination_controllers['long_LR'] # DecodedEpochSlicesPaginatedFigureController
        a_paginator = a_controlling_pagination_controller.paginator
        total_num_pages = int(a_paginator.num_pages)
        page_idx_sweep = np.arange(total_num_pages)
        page_num_sweep = page_idx_sweep + 1 # switch to 1-indexed
        # page_num_sweep
        print(f'export_all_pages(...): preparing to export {total_num_pages} pages from 4 decoders:')

        out_fig_paths_dict_list = {}

        for a_page_idx, a_page_num in zip(page_idx_sweep, page_num_sweep):
            print(f'switching to page: a_page_idx: {a_page_idx}, a_page_num: {a_page_num} of total_num_pages: {total_num_pages}')
            # a_pagination_controller.on_paginator_control_widget_jump_to_page(page_idx=a_page_idx)
            # a_pagination_controller.ui.mw.draw()
            # export_decoder_pagination_controller_figure_page(pagination_controller_dict, curr_active_pipeline)

            self.jump_to_page(page_idx=a_page_idx)
            self.draw()
            out_fig_paths_dict_list[a_page_idx] = self.export_decoder_pagination_controller_figure_page(curr_active_pipeline=curr_active_pipeline, **output_figure_kwargs)

        print(f'\tdone.')

        if enable_export_combined_img:
            (_flat_png_list, _out_combined_img, combined_img_out_path) = self.build_combined_all_pages_image(out_fig_paths_dict_list=out_fig_paths_dict_list, combined_image_basename=combined_image_basename)
            return out_fig_paths_dict_list, (_out_combined_img, combined_img_out_path)
        else:
            return out_fig_paths_dict_list
    
    
    #endregion Export/Output ______________________________________________________________________________________________________ #

    # ==================================================================================================================== #
    # MatplotlibTimeSynchronizedWidget Wrappers                                                                            #
    # ==================================================================================================================== #
                
    # def getFigure(self):
    #     return self.plots.fig
        
    

    def draw(self):
        """ Calls .draw() on all children MatplotlibTimeSynchronizedWidget items. 
        Successfully redraws items.

        """
        #TODO 2023-07-06 15:05: - [ ] PERFORMANCE - REDRAW
        for a_name, a_child_paginated_widget in self.paginated_widgets.items():
            # a_child_paginated_widget.ui.canvas.draw()
            a_child_paginated_widget.draw()
        
    

    def refresh_current_page(self):
        """ called to refresh the currently selected page for all controllers (to redraw the data widgets or axes).
        """
        # if self.debug_print:
        #     self.ui.print(f'PhoPaginatedMultiDecoderDecodedEpochsWindow.refresh_current_page():') # for page_idx == max_index this is called but doesn't continue
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.refresh_current_page()

    def update_params(self, **updated_values):
        """ called to change the .params on all of the child controllers simultaneously.
         
          
        
            paginated_multi_decoder_decoded_epochs_window.update_params(posterior_heatmap_imshow_kwargs = dict(vmin=0.0))
            paginated_multi_decoder_decoded_epochs_window.refresh_current_page()


        """
        # if self.debug_print:
        #     self.ui.print(f'PhoPaginatedMultiDecoderDecodedEpochsWindow.refresh_current_page():') # for page_idx == max_index this is called but doesn't continue
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.params.update(**updated_values)
        
    # ==================================================================================================================== #
    # Passthrough methods/properties                                                                                       #
    # ==================================================================================================================== #

    # def get_children_props(self, prop_name):
    #     # return [getattr(child, prop_name) for child in self.findChildren(QWidget)]
    #     return {a_name:getattr(a_pagination_controller, prop_name) for a_name, a_pagination_controller in self.pagination_controllers.items()}

    @function_attributes(short_name=None, tags=['USEFUL', 'children', 'simplification'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-04 05:18', related_items=[])
    def get_children_props(self, prop_path: str):
        """ 
        paginated_multi_decoder_decoded_epochs_window.get_children_props(prop_path='plots_data.epoch_slices')
        paginated_multi_decoder_decoded_epochs_window.get_children_props(prop_path='plots_data.filter_epochs_decoder_result')
        
        paginated_multi_decoder_decoded_epochs_window.get_children_props(prop_path='plots')
        paginated_multi_decoder_decoded_epochs_window.get_children_props(prop_path='plots.axs')
        
        """
        def get_nested_prop(obj, prop_path):
            attrs = prop_path.split(".")
            for attr in attrs:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            return obj

        return {a_name:get_nested_prop(a_pagination_controller, prop_path) for a_name, a_pagination_controller in self.pagination_controllers.items()}


    def set_children_props(self, prop_path: str, value):
        """ sets the property from a path for each child object 
        """
        # def get_nested_prop(obj, prop_path):
        #     attrs = prop_path.split(".")
        #     for attr in attrs:
        #         obj = getattr(obj, attr, None)
        #         if obj is None:
        #             break
        #     return obj
                
        prop_name_parts = prop_path.split('.')
        if len(prop_name_parts) >= 2:
            ## at least two parts
            container_prop_name = '.'.join(prop_name_parts[:-1]) ## all but the last prop
            final_property_name: str = prop_name_parts[-1]
            
            child_container_props_dict = self.get_children_props(prop_path=container_prop_name)
            
            for a_name, a_pagination_controller in self.pagination_controllers.items():
                a_container = child_container_props_dict[a_name]                
                setattr(a_container, final_property_name, value) ## hopefully updates in-place?                
                # get_nested_prop(a_pagination_controller, container_prop_name)
                
        else:
            raise NotImplementedError(f'')
            
        # for child in self.findChildren(QWidget):
        #     setattr(child, prop_name, value)



    def show_message(self, message: str, durationMs:int=4000):
        """ show a toast message """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            a_pagination_controller.show_message(message=message, durationMs=durationMs)



    def update_titles(self, window_title: str, children_titles: Optional[Dict[str, Optional[str]]] = None):
        """ sets the suptitle and window title for the figure """
        # Set the window title:
        self.setWindowTitle(window_title)

        ## Update embedded figures:
        if children_titles is not None:
            for a_name, a_pagination_controller in self.pagination_controllers.items():
                desired_child_title = children_titles.get(a_name, None)
                if desired_child_title is not None:
                    a_pagination_controller.ui.mw.fig.suptitle(desired_child_title, wrap=True) # set the plot suptitle
                    a_pagination_controller.ui.mw.draw()
        

    def enable_middle_click_selected_epoch_times_to_clipboard(self, is_enabled:bool=True):
        """ sets the copying of epoch times to the clipboard """
        for a_name, a_pagination_controller in self.pagination_controllers.items():
            # a_pagination_controller.params.debug_print = True
            if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
                a_pagination_controller.params['on_middle_click_item_callbacks'] = {}

            if is_enabled:
                a_pagination_controller.params.on_middle_click_item_callbacks['copy_epoch_times_to_clipboard_callback'] = ClickActionCallbacks.copy_epoch_times_to_clipboard_callback
            else:
                a_pagination_controller.params.on_middle_click_item_callbacks.pop('copy_epoch_times_to_clipboard_callback', None)

    @function_attributes(short_name=None, tags=['spike_raster', 'attached'], input_requires=[], output_provides=[], uses=['_build_attached_raster_viewer', '_apply_xticks_to_pyqtgraph_plotitem'], used_by=[], creation_date='2024-09-25 15:50', related_items=[])
    def build_attached_raster_viewer_widget(self, track_templates, active_spikes_df: pd.DataFrame, filtered_epochs_df: pd.DataFrame,  enable_adding_to_embedded_dockarea: bool=True) -> Tuple["RankOrderRastersDebugger", Callable]:
        """ Plots a synchronized raster_viewer_widget for the epochs in 
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
            
            __out_ripple_rasters, update_attached_raster_viewer_epoch_callback = paginated_multi_decoder_decoded_epochs_window.build_attached_raster_viewer_widget(track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_epochs_df=long_like_during_post_delta_only_filter_epochs_df) # Long-like-during-post-delta

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
        from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers, DesiredWidgetLocation, WidgetGeometryInfo

        print(f'Middle-click any epoch to adjust the Attached Raster Window to that epoch.')
        
        _out_ripple_rasters: RankOrderRastersDebugger = _build_attached_raster_viewer(self, track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df=filtered_epochs_df)


        ## Get the time bin within the clicked epoch
        @function_attributes(short_name=None, tags=['callback', 'selection', 'time_bin_selection'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-10 09:06', related_items=[])
        def update_clicked_epoch_time_bin_selection_callback(a_pagination_controller, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
            """ gets the time_bin within the clicked epoch
            
            captures: attached_ripple_rasters_widget, attached_directional_template_pfs_debugger
            """
            from matplotlib.backend_bases import MouseButton, MouseEvent, LocationEvent, PickEvent
            print(f'update_clicked_epoch_time_bin_selection_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
            print(f'\tevent: {event}\n\ttype(event): {type(event)}\n') # event: button_press_event: xy=(245, 359) xydata=(65.00700367785453, 156.55817377538108) button=3 dblclick=False inaxes=Axes(0.0296913,0.314173;0.944584x0.0753216)
            # type(event): <class 'matplotlib.backend_bases.MouseEvent'>
            if clicked_epoch_start_stop_time is not None:
                if len(clicked_epoch_start_stop_time) == 2:
                    start_t, end_t = clicked_epoch_start_stop_time
                    print(f'clicked widget at {clicked_ax}. [{start_t}, {end_t}]')
                    found_time_bin_idx = None
                    if isinstance(event, MouseEvent):
                        # matplotlib mouse event
                        if event.inaxes:                   
                            event_dict = {               
                                'data_x':event.xdata,
                                'data_y':event.ydata,
                                'pixel_x':event.x,
                                'pixel_y':event.y,
                            }
                            clicked_t_seconds: float = float(event.xdata)
                            found_time_bin_idx, (found_time_bin_start_t, found_time_bin_stop_t) = a_pagination_controller.try_get_clicked_epoch_time_bin_idx(clicked_data_index=clicked_data_index, clicked_t_seconds=clicked_t_seconds)
                            a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx = {
                                'clicked_data_index': clicked_data_index, 'clicked_epoch_start_stop_time': clicked_epoch_start_stop_time,
                                'found_time_bin_idx': found_time_bin_idx, 'found_time_bin_start_t': found_time_bin_start_t, 'found_time_bin_stop_t': found_time_bin_stop_t,
                                'active_time_bin_spikes_df': None, 'active_time_bin_unique_active_aclus': None,
                            }
                            if found_time_bin_idx is not None:
                                print(f'found_time_bin_idx: {found_time_bin_idx} for clicked time: {clicked_t_seconds}')
                                _out_ripple_rasters.clear_highlighting_indicator_regions() ## only allow a single selection
                                _out_ripple_rasters.add_highlighting_indicator_regions(t_start=found_time_bin_start_t, t_stop=found_time_bin_stop_t, identifier=f"TestTimeBinSelection[{clicked_data_index}, {found_time_bin_idx}]")
                                active_time_bin_spikes_df: pd.DataFrame = deepcopy(_out_ripple_rasters.get_active_epoch_spikes_df().spikes.time_sliced(found_time_bin_start_t, found_time_bin_stop_t)) ## active spikes
                                active_time_bin_unique_active_aclus = np.unique(active_time_bin_spikes_df['aclu'].to_numpy()) ## active time-bin aclus
                                a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx['active_time_bin_spikes_df'] = deepcopy(active_time_bin_spikes_df)                                
                                a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx['active_time_bin_unique_active_aclus'] = deepcopy(active_time_bin_unique_active_aclus)
                                                                
                                print(f'active_time_bin_unique_active_aclus: {active_time_bin_unique_active_aclus}')
                                print(f'a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx: {a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx}')
                                a_pagination_controller.ui.print(f'active_time_bin_unique_active_aclus: {active_time_bin_unique_active_aclus}')
                                a_pagination_controller.ui.print(f'a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx: {a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx}')

                                # a_pagination_controller.attached_directional_template_pfs_debugger
                                attached_directional_template_pfs_debugger = _out_ripple_rasters.attached_directional_template_pfs_debugger
                                if attached_directional_template_pfs_debugger is not None:
                                    if isinstance(attached_directional_template_pfs_debugger, dict):
                                        attached_directional_template_pfs_debugger = attached_directional_template_pfs_debugger['obj']
                                    attached_directional_template_pfs_debugger.update_cell_emphasis(active_time_bin_unique_active_aclus.tolist()) ## update the emphasis to the clicked bin only
                                else:
                                    print(f'attached_directional_template_pfs_debugger is None!')
                                    
                                self.ui.highlighted_epoch_time_bin_idx = deepcopy(a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx)
                                
                                print(f'done!')
                                
                            else:
                                print(f'could not find time bin for clicked time: {clicked_t_seconds}')

                        else:
                            print('event out of axes!')
                            
                    else:
                        pass

                    print(f'done.')


        ## Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior.
        @function_attributes(short_name=None, tags=['callback', 'selection', 'epoch_selection', 'raster'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-29 17:13', related_items=[])
        def update_attached_raster_viewer_epoch_callback(a_pagination_controller, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
            """ Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior. 
            called when the user middle-clicks an epoch 
            
            captures: _out_ripple_rasters
            """
            print(f'update_attached_raster_viewer_epoch_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
            _did_update_selected_epoch: bool = False
            if clicked_epoch_start_stop_time is not None:
                if len(clicked_epoch_start_stop_time) == 2:
                    start_t, end_t = clicked_epoch_start_stop_time
                    print(f'start_t: {start_t}')
                    try:
                        _out_ripple_rasters.programmatically_update_epoch_IDX_from_epoch_start_time(start_t)
                        _did_update_selected_epoch = True
                    except Exception as e:
                        print(f'could not update selected epoch: {e}.')
                        # raise e

            if _did_update_selected_epoch:
                ## update the grid to match the epoch bins
                print(f'_did_update_selected_epoch: True, clicked_data_index: {clicked_data_index}')
                included_page_data_indicies, (curr_page_active_filter_epochs, curr_page_epoch_labels, curr_page_time_bin_containers, curr_page_posterior_containers) = a_pagination_controller.plots_data.paginator.get_page_data(page_idx=a_pagination_controller.current_page_idx)
                # page_rel_clicked_ax_index = included_page_data_indicies.index(clicked_data_index)
                page_rel_clicked_ax_index = clicked_data_index - included_page_data_indicies[0]
                print(f'\tpage_rel_clicked_ax_index: {page_rel_clicked_ax_index}')
                # [clicked_ax]
                # self.plots.axs
                a_binning_container = curr_page_time_bin_containers[page_rel_clicked_ax_index] # BinningContainer 
                curr_epoch_bin_edges: NDArray = deepcopy(a_binning_container.edges)
                # curr_epoch_bin_edges
                
                _out_ripple_rasters.clear_highlighting_indicator_regions()
                
                ## Get the plot to modify on the raster_plot_widget
                # a_render_plots_container = _out_ripple_rasters.plots['all_separate_plots']['Long_LR'] # RenderPlots
                for a_decoder_name, a_render_plots_container in _out_ripple_rasters.plots['all_separate_plots'].items():         
                    plot_item = a_render_plots_container['root_plot']
                    # Define custom ticks at desired x-values
                    # Each tick is a tuple of (position, label)
                    # custom_ticks = [(pos, str(pos)) for pos in curr_epoch_bin_edges]
                    custom_ticks = [(pos, '') for pos in curr_epoch_bin_edges]
                    _apply_xticks_to_pyqtgraph_plotitem(plot_item=plot_item, custom_ticks=custom_ticks)
                    # Update the PlotItem and its scene
                    plot_item.update()
                    plot_item.scene().update()

                    
                print(f'done.')
                
        # ==================================================================================================================================================================================================================================================================================== #
        # Begin function body                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        for a_name, a_pagination_controller in self.pagination_controllers.items():
            # a_pagination_controller.params.debug_print = True
            a_pagination_controller.plots_data.highlighted_epoch_time_bin_idx = {} ## initialize plots_data
                
            if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
                a_pagination_controller.params['on_middle_click_item_callbacks'] = {}
            
            a_pagination_controller.params.on_middle_click_item_callbacks['update_attached_raster_viewer_epoch_callback'] = update_attached_raster_viewer_epoch_callback
        
            if not a_pagination_controller.params.has_attr('on_secondary_click_item_callbacks'):
                    a_pagination_controller.params['on_secondary_click_item_callbacks'] = {}
                
            ## epoch change with middle click, time bin with right click
            a_pagination_controller.params.on_secondary_click_item_callbacks['update_attached_raster_viewer_epoch_callback'] = update_attached_raster_viewer_epoch_callback # need to update epoch first
            a_pagination_controller.params.on_secondary_click_item_callbacks['get_click_time_epoch_time_bin_callback'] = update_clicked_epoch_time_bin_selection_callback # then update time bin


        _out_ripple_rasters.setWindowTitle(f'Template Rasters <Controlled by DecodedEpochSlices window>')
        ## Align the windows:
        target_window = self.window()
        a_controlled_widget = _out_ripple_rasters.root_dockAreaWindow
        WidgetPositioningHelpers.align_window_edges(target_window, a_controlled_widget.window(), relative_position='above', resize_to_main=(1.0, None)) # resize to same width, no change to height

        ## Store raster viewer internally
        self.ui.attached_ripple_rasters_widget = None
        self.ui.attached_ripple_rasters_widget = _out_ripple_rasters
        self.ui.update_attached_raster_viewer_epoch_callback = update_attached_raster_viewer_epoch_callback
        

        if enable_adding_to_embedded_dockarea:
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

            ## Transfer the four items or embed the whole window?
            a_win = _out_ripple_rasters.root_dockAreaWindow
            print(f'moving RankOrderRastersDebugger attached window into main window dock...')
            rankOrderRastersDebugger_dock_name: str = 'RankOrderRastersDebugger'
            self.contents.dock_configs[rankOrderRastersDebugger_dock_name] = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)
            self.contents.dock_widgets[rankOrderRastersDebugger_dock_name] = self.add_display_dock(identifier=rankOrderRastersDebugger_dock_name, widget=a_win, dockSize=(430,780), dockAddLocationOpts=['top'],
                                                                                      display_config=self.contents.dock_configs[rankOrderRastersDebugger_dock_name], autoOrientation=False)

        return _out_ripple_rasters, update_attached_raster_viewer_epoch_callback


    @function_attributes(short_name=None, tags=['yellow-blue', 'matplotlib', 'attached'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices'], used_by=[], creation_date='2024-10-04 07:23', related_items=[])
    def build_attached_yellow_blue_track_identity_marginal_window(self, directional_merged_decoders_result, global_session, 
                                                                   filter_epochs=None, filter_epochs_decoder_result: DecodedFilterEpochsResult=None, name: str ='TrackIdentity_Marginal_Ripples', active_context: IdentifyingContext=None, 
                                                                   enable_adding_to_embedded_dockarea: bool=True, **kwargs) -> RenderPlots:
        """ Attaches a stack of yellow-blue trackID marginal plots to the right side of the window. Currently they do not update.
        
        Uses: global_session.position, global_session.replay
        
        HARDCODED TO RIPPLES RN
        
        
        yellow_blue_trackID_marginals_plot_tuple = paginated_multi_decoder_decoded_epochs_window.build_attached_yellow_blue_track_identity_marginal_window(directional_merged_decoders_result, global_session, ripple_decoding_time_bin_size)


        ## Caller should really pass {'single_plot_fixed_height': 35.0, 'max_num_lap_epochs': 25, 'max_num_ripple_epochs': 45, 'size': (8, 55), 'dpi': 72} with the same values it has so they line up.

        """
        ## INPUTS: paginated_multi_decoder_decoded_epochs_window, directional_merged_decoders_result

        # directional_merged_decoders_result # all_directional_ripple_filter_epochs_decoder_result, ripple_track_identity_marginals_tuple
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
        assert (filter_epochs is not None)
        assert (filter_epochs_decoder_result is not None)
        
        debug_print = kwargs.get('debug_print', False)
        
        # TrackID ____________________________________________________________________________________________________________ #
        marginal_y_bin_labels = kwargs.get('marginal_y_bin_labels', ['long', 'short'])
        active_marginal_fn =  kwargs.get('active_marginal_fn', lambda a_filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(a_filter_epochs_decoder_result))

        # # All-four ___________________________________________________________________________________________________________ #
        # marginal_y_bin_labels = kwargs.get('marginal_y_bin_labels', ['long_LR', 'long_RL', 'short_LR', 'short_RL'])
        # active_marginal_fn = kwargs.get('active_marginal_fn', lambda a_filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(a_filter_epochs_decoder_result)) ## IMPORTANT: `active_marginal_fn` is what makes this a yellow-blue plot



        ## Extract params_kwargs
        params_kwargs = kwargs.pop('params_kwargs', {})
        params_kwargs = dict(skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, isPaginatorControlWidgetBackedMode=True) | params_kwargs ## merge 
        params_kwargs = {'max_subplots_per_page': 10, 'scrollable_figure': False, 'use_AnchoredCustomText': False,
                'should_suppress_callback_exceptions': False, 'isPaginatorControlWidgetBackedMode': True,
                'enable_update_window_title_on_page_change': False, 'build_internal_callbacks': True, 'debug_print': True, 'skip_plotting_measured_positions': True,  'enable_decoded_most_likely_position_curve': False, 
                                                                                                    'enable_decoded_sequence_and_heuristics_curve': False, 'show_pre_merged_debug_sequences': False, 'show_heuristic_criteria_filter_epoch_inclusion_status': False,
                                                                                                     'enable_radon_transform_info': False, 'enable_weighted_correlation_info': False, 'enable_weighted_corr_data_provider_modify_axes_rect': False, 'enable_marginal_labels': True, 'marginal_y_bin_labels': marginal_y_bin_labels} | params_kwargs
        
        print(f'params_kwargs: {params_kwargs}')
        
        # print(f'params_kwargs: {params_kwargs}')
        max_subplots_per_page: int = kwargs.pop('max_subplots_per_page', params_kwargs.pop('max_subplots_per_page', 10)) # kwargs overrides params_kwargs
        is_controlling_widget = False ## always false for YellowBlue plot
        
        curr_params_kwargs = deepcopy(params_kwargs)
        curr_params_kwargs['is_controlled_widget'] = (not is_controlling_widget)
        if ('disable_y_label' not in curr_params_kwargs):
            # If user didn't provide an explicit 'disable_y_label' option, use the defaults which is to hide labels on all the but the controlling widget
            if is_controlling_widget:
                curr_params_kwargs['disable_y_label'] = False
            else:
                curr_params_kwargs['disable_y_label'] = True


        active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
        # long_short_marginals: List[NDArray] = [x.p_x_given_n for x in DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(all_directional_ripple_filter_epochs_decoder_result)] # these work if I want all of them

        first_controller = list(self.pagination_controllers.values())[0]
        # Ripple Track-identity (Long/Short) Marginal:
        ## INPUTS: all_directional_ripple_filter_epochs_decoder_result, global_session, ripple_decoding_time_bin_size
        # _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'TrackID', 't_bin': decoding_time_bin_size}
        # _main_context = IdentifyingContext(**{'decoded_epochs': 'Ripple', 'Marginal': 'TrackID', 't_bin': round(decoding_time_bin_size, ndigits=5)})
        correct_num_filter_epochs: int = np.shape(filter_epochs)[0] ## correct
        # np.shape(filter_epochs_decoder_result.filter_epochs) ## incorrect
        # filter_epochs_decoder_result.filtered_by_epochs(filter_epochs)
        filter_epochs_decoder_result = deepcopy(filter_epochs_decoder_result).filtered_by_epoch_times(included_epoch_start_times=filter_epochs['start'].to_numpy())
        post_filter_num_filter_epochs: int = np.shape(filter_epochs_decoder_result.filter_epochs)[0]
        assert post_filter_num_filter_epochs == correct_num_filter_epochs, f"post_filter_num_filter_epochs: {post_filter_num_filter_epochs}, correct_num_filter_epochs: {correct_num_filter_epochs}"
        
        # ==================================================================================================================== #
        # 2024-10-09 - `DecodedEpochSlicesPaginatedFigureController`-based mode                                                #
        # ==================================================================================================================== #
        a_yellow_blue_controller: DecodedEpochSlicesPaginatedFigureController = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(filter_epochs_decoder_result.filter_epochs, # filter_epochs_decoder_result.filter_epochs,
                                                                                            filter_epochs_decoder_result=filter_epochs_decoder_result,
                                                                                            xbin=active_decoder.xbin, global_pos_df=global_session.position.to_dataframe(),
                                                                                            a_name=f'YellowBlueMarginalEpochSlices', active_context=active_context,
                                                                                            active_marginal_fn=active_marginal_fn, ## IMPORTANT: `active_marginal_fn` is what makes this a yellow-blue plot
                                                                                            # active_marginal_fn=None,
                                                                                            max_subplots_per_page=max_subplots_per_page, debug_print=debug_print,
                                                                                            # included_epoch_indicies=curr_page_epoch_labels, ## This is what broke rendering on every page except the first one
                                                                                            params_kwargs=curr_params_kwargs) # , save_figure=save_figure
        
        # Post-plot call:
        # Constrains each of the plotters at least to the minimum height:
        # a_pagination_controller.params.all_plots_height
        # resize to minimum height
        a_widget = a_yellow_blue_controller.ui.mw # MatplotlibTimeSynchronizedWidget
        screen = a_widget.screen()
        screen_size = screen.size()

        target_height = a_yellow_blue_controller.params.get('scrollAreaContents_MinimumHeight', None)
        if target_height is None:
            target_height = (a_yellow_blue_controller.params.all_plots_height + 30)
        desired_final_height = int(min(target_height, screen_size.height())) # don't allow the height to exceed the screen height.
        if debug_print:
            print(f'target_height: {target_height}, {  desired_final_height = }')
        # a_widget.size()
        a_widget.setMinimumHeight(desired_final_height) # the 30 is for the control bar
        mw = a_yellow_blue_controller.ui.mw # MatplotlibTimeSynchronizedWidget
        yellow_blue_attached_render_plot = a_yellow_blue_controller
        

        ## Align the windows:
        # target_window = paginated_multi_decoder_decoded_epochs_window.window()
        target_window = first_controller.ui.mw
        a_controlled_widget = mw
        WidgetPositioningHelpers.align_window_edges(target_window, a_controlled_widget.window(), relative_position='right_of', resize_to_main=(None, 1.0)) # resize to same height, no change to width
        # TODO: hold a reference to it? Update function for changing pages?
        



        # Finish Setup _______________________________________________________________________________________________________ #
        a_win = a_yellow_blue_controller.ui.mw.window()
        ## TODO: add to dock area?
        
        if enable_adding_to_embedded_dockarea:
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors
            print(f'moving yellow-blue marginals attached window into main window dock...')
            yellowBlueMarginal_dock_name: str = 'yellowBlueMarginal'
            self.contents.dock_configs[yellowBlueMarginal_dock_name] = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)
            self.contents.dock_widgets[yellowBlueMarginal_dock_name] = self.add_display_dock(identifier=yellowBlueMarginal_dock_name, widget=a_win, dockSize=(430,780), dockAddLocationOpts=['right'],
                                                                                      display_config=self.contents.dock_configs[yellowBlueMarginal_dock_name], autoOrientation=False)

        else:
            ## separate window        
            icon = try_get_icon(icon_path=":/Render/Icons/graphics/yellow_blue_plot_icon.png")
            if icon is not None:
                a_win.setWindowIcon(icon)

        ## Gets the global bar and sets up pagination/control
        global_thin_button_bar_widget: ThinButtonBarWidget = self.ui._contents.global_thin_button_bar_widget
        global_paginator_controller_widget = global_thin_button_bar_widget.ui.paginator_controller_widget ## need paginator control widget
        new_connections_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow._perform_convert_decoder_pagination_controller_dict_to_controlled(a_controlling_pagination_controller_widget=global_paginator_controller_widget,
                                                                                                                                    controlled_pagination_controllers_list=(a_yellow_blue_controller, ))


        ## Store yellow-blue viewer internally
        self.ui.attached_yellow_blue_marginals_viewer_widget = None
        self.ui.attached_yellow_blue_marginals_viewer_widget = a_yellow_blue_controller
        # self.ui.connections['attached_yellow_blue_marginals_viewer_widget'] = new_connections_dict
        
        # extant_marginal_label_artists_dict = self.ui.attached_yellow_blue_marginals_viewer_widget.plots.get('marginal_label_artists_dict', {})
        # ## can remove them by
        # for decoder_name, inner_output_dict in extant_marginal_label_artists_dict.items():
        #     for a_name, an_artist in inner_output_dict.items():
        #         an_artist.remove()
        
        # marginal_label_artists_dict = {}

        # for i, ax in enumerate(self.ui.attached_yellow_blue_marginals_viewer_widget.plots.axs):
        #     marginal_label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL'], enable_draw_decoder_colored_lines=False, should_use_ax_fraction_positioning=True) ## use this because we used `DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(a_filter_epochs_decoder_result)` up above
        #     # marginal_label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long', 'short'], enable_draw_decoder_colored_lines=False)
        #     # marginal_label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['LR', 'RL'], enable_draw_decoder_colored_lines=False)

        # self.ui.attached_yellow_blue_marginals_viewer_widget.plots['marginal_label_artists_dict'] = marginal_label_artists_dict

        a_yellow_blue_controller.add_data_overlays(decoder_decoded_epochs_result=filter_epochs_decoder_result, included_columns=[])
        
        return yellow_blue_attached_render_plot



    @function_attributes(short_name=None, tags=['export', 'image', 'marginal'], input_requires=[], output_provides=[], uses=['PosteriorExporting._perform_export_current_epoch_marginal_and_raster_images'], used_by=['cls.export_all_epochs_to_images'], creation_date='2024-10-09 16:29', related_items=[])
    def export_current_epoch_marginal_and_raster_images(self, directional_merged_decoders_result, root_export_path: Path, active_context: Optional[IdentifyingContext]=None):
        """ Export Marginal Pseudo2D posteriors and rasters for middle-clicked epochs
        

        Usage:        
            # DirectionalMergedDecoders: Get the result after computation:
            directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders'] # uses `DirectionalMergedDecoders`.

            # root_export_path: Path = Path(r"/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image").resolve() # Lab
            root_export_path.mkdir(exist_ok=True)
            Assert.path_exists(root_export_path)

            complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
            epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path) = paginated_multi_decoder_decoded_epochs_window.export_current_epoch_marginal_and_raster_images(directional_merged_decoders_result=directional_merged_decoders_result, root_export_path=root_export_path, active_context=complete_session_context)

            file_uri_from_path(epoch_specific_folder)
            fullwidth_path_widget(a_path=epoch_specific_folder, file_name_label="epoch_specific_folder:")
        
        """
        from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
        # root_export_path = Path(r"E:\Dropbox (Personal)\Active\Kamran Diba Lab\Pho-Kamran-Meetings\2024-05-01 - Pseudo2D Again\array_as_image").resolve() # Apogee
        # root_export_path = Path('/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image').resolve() # Lab
        # root_export_path = Path(r"E:\Dropbox (Personal)\Active\Kamran Diba Lab\Pho-Kamran-Meetings\2024-09-25 - Time bin considerations\array_as_image").resolve() # Apogee
        # root_export_path: Path = Path(r"/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image").resolve() # Lab

        if (not hasattr(self.ui, 'attached_ripple_rasters_widget') or (self.ui.attached_ripple_rasters_widget is None)):
            raise ValueError(f"self.ui.attached_ripple_rasters_widget is None! Is there an attached raster_widget yet?")
        
        if active_context is None:
            active_context = IdentifyingContext('display_fn', 'export_current_epoch_marginal_and_raster_images')
        
        ## get the ripple name from the context of the first controller, all four will be the same.
        epoch_id_identifier_str: str = list(self.pagination_controllers.values())[0].params.active_identifying_figure_ctx.epochs
        
        
        epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path) = PosteriorExporting._perform_export_current_epoch_marginal_and_raster_images(_out_ripple_rasters=self.ui.attached_ripple_rasters_widget, directional_merged_decoders_result=directional_merged_decoders_result, 
            # filtered_decoder_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epoch_id_identifier_str='ripple',
            filtered_decoder_filter_epochs_decoder_result_dict=self.decoder_filter_epochs_decoder_result_dict, epoch_id_identifier_str=epoch_id_identifier_str,
            active_session_context=active_context, 
            root_export_path = root_export_path,
        )
        print(f"exported to '{epoch_specific_folder}'")
        return epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path)


    @function_attributes(short_name=None, tags=['export', 'batch', 'marginals', 'rasters'], input_requires=[], output_provides=[], uses=['export_current_epoch_marginal_and_raster_images'], used_by=[], creation_date='2025-06-03 00:47', related_items=[])
    @classmethod
    def perform_export_all_epochs_to_images(cls, paginated_multi_decoder_decoded_epochs_window: "PhoPaginatedMultiDecoderDecodedEpochsWindow", directional_merged_decoders_result, root_export_path: Path, active_context: Optional[IdentifyingContext]=None, **kwargs):
        """ programmatically iterates through all epochs and exports them to file, including their posteriors, rasters, and marginals 

        Usage:
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import PhoPaginatedMultiDecoderDecodedEpochsWindow

            # DirectionalMergedDecoders: Get the result after computation:
            directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders'] # uses `DirectionalMergedDecoders`.

            # root_export_path: Path = Path(r"/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image").resolve() # Lab
            root_export_path: Path = Path(r'K:/scratch/collected_outputs/figures/array_as_image').resolve()
            root_export_path.mkdir(exist_ok=True)
            Assert.path_exists(root_export_path)

            complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
            _out_path_tuples_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.perform_export_all_epochs_to_images(paginated_multi_decoder_decoded_epochs_window=paginated_multi_decoder_decoded_epochs_window, directional_merged_decoders_result=directional_merged_decoders_result, root_export_path=root_export_path, active_context=complete_session_context)

        """

        if (not hasattr(paginated_multi_decoder_decoded_epochs_window.ui, 'attached_ripple_rasters_widget') or (paginated_multi_decoder_decoded_epochs_window.ui.attached_ripple_rasters_widget is None)):
            raise ValueError(f"paginated_multi_decoder_decoded_epochs_window.ui.attached_ripple_rasters_widget is None! Is there an attached raster_widget yet?")

        # attached_yellow_blue_marginals_viewer_widget: DecodedEpochSlicesPaginatedFigureController = paginated_multi_decoder_decoded_epochs_window.attached_yellow_blue_marginals_viewer_widget
        attached_ripple_rasters_widget: RankOrderRastersDebugger = paginated_multi_decoder_decoded_epochs_window.attached_ripple_rasters_widget
        # attached_directional_template_pfs_debugger: TemplateDebugger = paginated_multi_decoder_decoded_epochs_window.attached_directional_template_pfs_debugger
        assert attached_ripple_rasters_widget is not None, f"attached_ripple_rasters_widget is required to export rasters!"

        ## INPUTS: paginated_multi_decoder_decoded_epochs_window, attached_ripple_rasters_widget, root_export_path, directional_merged_decoders_result, complete_session_context
        _out_path_tuples_dict = {}
        n_epochs: int = deepcopy(attached_ripple_rasters_widget.n_epochs)

        for an_epoch_idx in np.arange(n_epochs):
            print(f'processing an_epoch_idx: {an_epoch_idx}/{n_epochs}...')
            attached_ripple_rasters_widget.programmatically_update_epoch_IDX(an_epoch_idx=an_epoch_idx)
            _an_out_paths = paginated_multi_decoder_decoded_epochs_window.export_current_epoch_marginal_and_raster_images(directional_merged_decoders_result=directional_merged_decoders_result, root_export_path=root_export_path, active_context=active_context, **kwargs)
            # epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path) = _an_out_paths
            _out_path_tuples_dict[an_epoch_idx] = (_an_out_paths[0], *_an_out_paths[1]) ## build simple 4-tuple of outputs

        print(f'done with {n_epochs} epochs.')
        # OUTPUTS: _out_path_tuples_dict
        return _out_path_tuples_dict



    
    @function_attributes(short_name=None, tags=['export', 'batch', 'rasters', 'image', 'marginal'], input_requires=[], output_provides=[], uses=['.perform_export_all_epochs_to_images'], used_by=[], creation_date='2025-06-03 01:47', related_items=[])
    def export_all_epoch_marginal_and_raster_images(self, directional_merged_decoders_result, root_export_path: Path, active_context: Optional[IdentifyingContext]=None, **kwargs):
        """ programmatically iterates through all epochs and exports them to file, including their posteriors, rasters, and marginals 

        Usage:
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import PhoPaginatedMultiDecoderDecodedEpochsWindow

            # DirectionalMergedDecoders: Get the result after computation:
            directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders'] # uses `DirectionalMergedDecoders`.

            # root_export_path: Path = Path(r"/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image").resolve() # Lab
            root_export_path: Path = Path(r'K:/scratch/collected_outputs/figures/array_as_image').resolve()
            # root_export_path: Path = Path(r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/array_as_image').resolve()
            root_export_path.mkdir(exist_ok=True)
            Assert.path_exists(root_export_path)

            complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
            _out_path_tuples_dict = paginated_multi_decoder_decoded_epochs_window.export_all_epoch_marginal_and_raster_images(directional_merged_decoders_result=directional_merged_decoders_result, root_export_path=root_export_path, active_context=complete_session_context)
            
            
        """
        return self.perform_export_all_epochs_to_images(paginated_multi_decoder_decoded_epochs_window=self, directional_merged_decoders_result=directional_merged_decoders_result, root_export_path=root_export_path, active_context=active_context, **kwargs)



    @function_attributes(short_name=None, tags=['multi-window', 'widget', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-09 14:19', related_items=[])
    @classmethod
    def plot_full_paginated_decoded_epochs_window(cls, curr_active_pipeline, track_templates, active_spikes_df,
                                                active_decoder_decoded_epochs_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], directional_decoders_epochs_decode_result: "DecoderDecodedEpochsResult", 
                                                active_filter_epochs_df: pd.DataFrame, known_epochs_type='ripple', title='Long-like post-Delta Ripples Only', **kwargs):
        """ 
        Plots 3 connected windows: the main decoded position posteriors, the track identity posteriors, and the rasters

        """
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import co_filter_epochs_and_spikes
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

        ## INPUTS: curr_active_pipeline, track_templates, active_spikes_df, active_decoder_decoded_epochs_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], known_epochs_type='ripple', title='Long-like post-Delta Ripples Only'
        assert known_epochs_type in ['ripple', 'laps'], f"known_epochs_type: '{known_epochs_type}' should be either 'ripple' or 'laps'"
        global_epoch_name = curr_active_pipeline.find_Global_epoch_name()
        
        active_spikes_df = get_proper_global_spikes_df(curr_active_pipeline)
        # active_filter_epochs_df = deepcopy(decoder_laps_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)
        active_filter_epochs_df = deepcopy(active_decoder_decoded_epochs_result_dict['long_LR'].filter_epochs)
        _co_filter_epochs_and_spikes_kwargs_DICT = {'ripple': dict(epoch_id_key_name='ripple_epoch_id'),
            'laps': dict(epoch_id_key_name='lap_id')
        }
        active_co_filter_epochs_and_spikes_kwargs = _co_filter_epochs_and_spikes_kwargs_DICT[known_epochs_type] # resolve for the specific known_epochs_type ('ripple'/'laps')
        
        active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.333333333)
        active_filter_epochs_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=active_spikes_df, active_epochs_df=active_filter_epochs_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, **active_co_filter_epochs_and_spikes_kwargs, no_interval_fill_value=-1, add_unique_aclus_list_column=True, drop_non_epoch_spikes=True)
        
        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders'] # DirectionalPseudo2DDecodersResult, pull from global computations


        
        _shared_plotting_kwargs = {                # 'debug_print': True,
                'max_subplots_per_page': kwargs.get('params_kwargs', {}).get('max_subplots_per_page', 3),
                'scrollable_figure': kwargs.get('params_kwargs', {}).get('scrollable_figure', False),
                # 'scrollable_figure': True,
                # 'posterior_heatmap_imshow_kwargs': dict(vmin=0.0075),
                'use_AnchoredCustomText': kwargs.get('params_kwargs', {}).get('use_AnchoredCustomText', False),
                'should_suppress_callback_exceptions': kwargs.get('params_kwargs', {}).get('should_suppress_callback_exceptions', False),
                # 'build_fn': 'insets_view',
                'should_draw_time_bin_boundaries': kwargs.get('params_kwargs', {}).get('should_draw_time_bin_boundaries', True),
                 'time_bin_edges_display_kwargs': kwargs.get('params_kwargs', {}).get('time_bin_edges_display_kwargs', dict(color='grey', alpha=0.5, linewidth=1.5)),
        }
        
        params_kwargs = {'known_epochs_type': known_epochs_type,
                         'enable_per_epoch_action_buttons': False,
                'skip_plotting_most_likely_positions': True, 'skip_plotting_measured_positions': True, 
                'enable_decoded_most_likely_position_curve': False, 'enable_decoded_sequence_and_heuristics_curve': False, 'enable_radon_transform_info': False, 'enable_weighted_correlation_info': False,
                # 'enable_radon_transform_info': False, 'enable_weighted_correlation_info': False,
                # 'disable_y_label': True,
                'isPaginatorControlWidgetBackedMode': True,
                'enable_update_window_title_on_page_change': False, 'build_internal_callbacks': True,
                'posterior_heatmap_imshow_kwargs': dict(cmap=get_heatmap_cmap(cmap='Oranges', bad_color='black', under_color='white', over_color='red')),
                # 'debug_print': True,
                **_shared_plotting_kwargs,
        } | kwargs.pop('params_kwargs', {})
        
        # Build main Decoded Posterior Window ________________________________________________________________________________ #
        ## uses `active_decoder_decoded_epochs_result_dict`
        app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates,
            decoder_decoded_epochs_result_dict=active_decoder_decoded_epochs_result_dict, epochs_name=known_epochs_type, title=title,
            included_epoch_indicies=None, debug_print=False,
            params_kwargs = params_kwargs,
            **kwargs,
        )
        
        # Build Raster Widget ________________________________________________________________________________________________ #
        ripple_rasters_plot_tuple = paginated_multi_decoder_decoded_epochs_window.build_attached_raster_viewer_widget(track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_epochs_df=active_filter_epochs_df) 
        _out_ripple_rasters, update_attached_raster_viewer_epoch_callback = ripple_rasters_plot_tuple    
        ## Attach TemplateViewer to raster:
        _out_directional_template_pfs_debugger, debug_update_paired_directional_template_pfs_debugger = _out_ripple_rasters.plot_attached_directional_templates_pf_debugger(curr_active_pipeline=curr_active_pipeline)
        # Accessible via `directional_template_pfs_debugger = paginated_multi_decoder_decoded_epochs_window.ui.attached_ripple_rasters_widget.ui.controlled_references['directional_template_pfs_debugger']`

        # Build Yellow-Blue Marginal Widget __________________________________________________________________________________ #        
        _build_attached_yellow_blue_track_identity_marginal_window_kwargs_DICT = {'ripple': dict(decoding_time_bin_size=directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size, name='TrackIdentity_Marginal_Ripples', filter_epochs_decoder_result=deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result)),
            'laps': dict(decoding_time_bin_size=directional_decoders_epochs_decode_result.laps_decoding_time_bin_size, name='TrackIdentity_Marginal_Laps', filter_epochs_decoder_result=deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)),
        }
        active_build_attached_yellow_blue_track_identity_marginal_window_kwargs = _build_attached_yellow_blue_track_identity_marginal_window_kwargs_DICT[known_epochs_type] # resolve for the specific known_epochs_type ('ripple'/'laps')
        yellow_blue_plot_context = IdentifyingContext(**{'decoded_epochs': known_epochs_type.title(), 'Marginal': 'TrackID', 't_bin': round(active_build_attached_yellow_blue_track_identity_marginal_window_kwargs['decoding_time_bin_size'], ndigits=5)})
        
        # directional_merged_decoders_result.filtered_by_epoch_times()
        yellow_blue_trackID_marginals_plot_tuple = paginated_multi_decoder_decoded_epochs_window.build_attached_yellow_blue_track_identity_marginal_window(directional_merged_decoders_result, global_session=global_session, filter_epochs=deepcopy(active_filter_epochs_df), epochs_name=known_epochs_type, 
                                                                                                                                                           **active_build_attached_yellow_blue_track_identity_marginal_window_kwargs, **_shared_plotting_kwargs,
                                                                                                                                                           active_context=yellow_blue_plot_context)

        return (app, paginated_multi_decoder_decoded_epochs_window, pagination_controller_dict), ripple_rasters_plot_tuple, yellow_blue_trackID_marginals_plot_tuple




class VispySceneWrappingWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Composite Qt widget: vispy ``SceneCanvas.native`` plus an optional ``VispySceneTreeWidget``.

    Use for a single parent that shows the GL view and the scene graph inspector without a dock
    layout. The tree is identical to ``VispySceneTreeWidget`` (visibility toggles, GL blend column,
    refresh). ``column_renderers`` are forwarded when ``show_scene_tree`` is True.
    

    from pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets import VispySceneWrappingWidget


    """

    def __init__(self, canvas: scene.SceneCanvas, parent: Optional[Any] = None, *, show_scene_tree: bool = True, tree_on_right: bool = True, tree_minimum_width: int = 200, column_renderers: Optional[Dict[str, Callable[[Node], str]]] = None, splitter_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__(parent=parent)
        self.canvas = canvas
        self.scene_tree_widget: Optional[VispySceneTreeWidget] = None
        expanding = getattr(QtWidgets.QSizePolicy, 'Expanding', QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
        self.setWindowTitle('VispySceneWrappingWidget')
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        native = canvas.native
        if not show_scene_tree:
            outer.addWidget(cast(Any, native), stretch=1)
            return
        self.scene_tree_widget = VispySceneTreeWidget(root_node=canvas.scene, canvas=canvas, parent=self, column_renderers=column_renderers)
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
        outer.addWidget(splitter, stretch=1)
        self.buildUI()
        

    def buildUI(self):
        title = 'Volumetric 2D Time-Series Viewer'
        canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1400, 900), title=title, autoswap=False, resizable=True, decorate=True, fullscreen=False)
        self.canvas = canvas
        self.view = canvas.central_widget.add_view()
        # self.view.camera = scene.TurntableCamera(fov=_VOLUMETRIC_TURNTABLE_FOV, elevation=_VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION, azimuth=_VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH)
        self.view.camera = CustomTurntableCamera(fov=_VOLUMETRIC_TURNTABLE_FOV, elevation=_VOLUMETRIC_CAMERA_PERSPECTIVE_ELEVATION, azimuth=_VOLUMETRIC_CAMERA_PERSPECTIVE_AZIMUTH)
        
        
        self.scene_tree_widget = VispySceneTreeWidget(root_node=self.canvas.scene, canvas=self.canvas)
        self.scene_tree_widget.setMinimumWidth(200)
        root_dockAreaWindow, _app = DockAreaWrapper.build_default_dockAreaWindow(title=title, defer_show=True)
        self.main_window = root_dockAreaWindow
        self._build_camera_view_menu(main_window=root_dockAreaWindow)
        viewer_central_widget = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_central_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(canvas.native, stretch=1)

        if self.n_t_bins > 0:
            slider_widget = QtWidgets.QWidget()
            slider_layout = QtWidgets.QHBoxLayout(slider_widget)
            slider_layout.addWidget(QtWidgets.QLabel("t-bin:"))
            t_bin_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            t_bin_slider.setMinimum(0)
            t_bin_slider.setMaximum(max(0, self.n_t_bins - 1))
            t_bin_slider.setValue(self.active_t_bin_idx)
            t_bin_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            t_bin_slider.setTickInterval(1)
            t_bin_value_label = QtWidgets.QLabel(f"{self.active_t_bin_idx}/{max(0, self.n_t_bins - 1)}")
            t_bin_value_label.setMinimumWidth(90)
            slider_layout.addWidget(t_bin_slider, stretch=1)
            slider_layout.addWidget(t_bin_value_label)
            viewer_layout.addWidget(slider_widget, stretch=0)
            self.t_bin_slider = t_bin_slider
            self.t_bin_value_label = t_bin_value_label
            t_bin_slider.valueChanged.connect(self.on_slider_value_changed)

        epoch_slider_widget = QtWidgets.QWidget()
        epoch_slider_layout = QtWidgets.QHBoxLayout(epoch_slider_widget)
        epoch_slider_layout.addWidget(QtWidgets.QLabel("epoch:"))
        epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        epoch_slider.setMinimum(0)
        epoch_slider.setMaximum(0)
        epoch_slider.setValue(0)
        epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        epoch_slider.setTickInterval(1)
        epoch_slider.setEnabled(False)
        epoch_value_label = QtWidgets.QLabel("0/0")
        epoch_value_label.setMinimumWidth(90)
        epoch_slider_layout.addWidget(epoch_slider, stretch=1)
        epoch_slider_layout.addWidget(epoch_value_label)
        viewer_layout.addWidget(epoch_slider_widget, stretch=0)
        self.epoch_slider = epoch_slider
        self.epoch_value_label = epoch_value_label
        epoch_slider.valueChanged.connect(self.on_epoch_slider_value_changed)

        viewer_display_config = CustomDockDisplayConfig(showCloseButton=False, showTimelineSyncModeButton=False, showCollapseButton=False, custom_get_colors_callback_fn=CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color="#448aaa", border_color="#338199"))
        _, viewer_dock_item = root_dockAreaWindow.add_display_dock("Viewer", dockSize=(1100, 900), widget=viewer_central_widget, dockAddLocationOpts=['left'], display_config=viewer_display_config)
        
        _custom_dock_coloring_fn = CustomDockDisplayConfig.build_custom_get_colors_fn(fg_color='#ffffff', bg_color="#aaa344", border_color="#998A33")
        scene_tree_display_config = CustomDockDisplayConfig(showCloseButton=False, showTimelineSyncModeButton=False, showCollapseButton=False, custom_get_colors_callback_fn=_custom_dock_coloring_fn)
        _, _scene_tree_dock_item = root_dockAreaWindow.add_display_dock("Scene Tree", dockSize=(300, 900), widget=self.scene_tree_widget, dockAddLocationOpts=['right', viewer_dock_item], display_config=scene_tree_display_config)
        root_dockAreaWindow.resize(1400, 950)
        
        # Something to give 3D context (axis from 0 to 1)
        self.debug_xyz_axes = vz.XYZAxis(parent=self.view.scene)
        self.gridlines = vz.GridLines(parent=self.view.scene, color=(0.4, 0.4, 0.4, 0.4))
        self._build_coordinate_axes()

        self._build_arena_wireframe()
        ## Graphics
        self.position_line = vz.Line(pos=self.pos3d, color=(0.22, 0.22, 0.22, 0.6), width=1.0, parent=self.view.scene, name='Pos<x,y,t>')        
        self._build_debug_crosshairs()

        if self.highlight_epochs is not None and len(self.highlight_epochs) > 0:
            self._build_highlight_bands()

        if self.n_t_bins > 0:
            self.update_active_t_bin(self.active_t_bin_idx)

        if hasattr(canvas.events, 'key_press'):
            canvas.events.key_press.connect(self.on_key_press)
        if hasattr(canvas.events, 'key_release'):
            canvas.events.key_release.connect(self.on_key_release)
        if hasattr(canvas.events, 'mouse_move'):
            canvas.events.mouse_move.connect(self.on_mouse_move)
        if hasattr(canvas.events, 'mouse_leave'):
            canvas.events.mouse_leave.connect(self.on_mouse_leave)

        x_min, x_max = float(self.xbin[0]), float(self.xbin[-1])
        y_min, y_max = float(self.ybin[0]), float(self.ybin[-1])
        self.view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max), z=(0.0, self.z_max))
        self.scene_tree_widget.rebuild()
        root_dockAreaWindow.show()
        

    def rebuild(self) -> None:
        if self.scene_tree_widget is not None:
            self.scene_tree_widget.rebuild()



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