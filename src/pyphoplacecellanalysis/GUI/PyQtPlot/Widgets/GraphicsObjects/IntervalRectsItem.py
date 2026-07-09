"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""
import copy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from attr import has
import numpy as np

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from attrs import asdict, astuple, define, field, Factory
from qtpy import QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.color_helpers import ColorDataframeColumnHelpers # replacing `RectangleRenderTupleHelpers`
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.LegendItem import ItemSample, LegendItem # for custom legend
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.ReprPrintableWidgetMixin import ReprPrintableItemMixin
from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.TextItem.AlignableTextItem import CustomRectBoundedTextItem


# @metadata_attributes(short_name=None, tags=['data'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-10 08:55', related_items=['pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins.EpochDisplayConfig'])
@define(slots=False, repr=True)
class IntervalRectsItemData(UnpackableMixin):
    """ incremental progress towards more flexible self.data for `IntervalRectsItem` while maintaining drop-in compatibility with pre 2025-12-10 tuple-based approach via `UnpackableMixin`.
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItemData
    
    rect_data_tuple: IntervalRectsItemData = IntervalRectsItemData(*rect_data_tuple) ## init from raw tuple object
    start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple ## unpack just like tuple
    
    
    """
    start_t: float = field()
    series_vertical_offset: float = field()
    duration_t: float = field()
    series_height: float = field()
    pen: QtGui.QPen = field()
    brush: QtGui.QBrush = field()
    label: Optional[str] = field(default=None)
    

    def UnpackableMixin_unpacking_includes(self) -> Optional[List]:
        """ Items to be included (allowlist) from unpacking. 
        """
        return [self.__attrs_attrs__.start_t, self.__attrs_attrs__.series_vertical_offset, self.__attrs_attrs__.duration_t, self.__attrs_attrs__.series_height, self.__attrs_attrs__.pen, self.__attrs_attrs__.brush]
    


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)
# @function_attributes(short_name=None, tags=['GraphicsObject', 'item', 'renderable'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-28 16:36', related_items=[])
class IntervalRectsItem(ReprPrintableItemMixin, pg.GraphicsObject):
    """ Created to render the 2D Intervals as rectangles in a pyqtgraph 
    
        Based on pyqtgraph's CandlestickItem example
       
    Rectangle Item Specification: 
        Renders rectangles, with each specified by a tuple of the form:
            (start_t, series_vertical_offset, duration_t, series_height, pen, brush)

        Note that this is analagous to the position arguments of `QRectF`:
            (left, top, width, height) and (pen, brush)
            
            
    TODO: BUG: Right click currently invokes the custom example context menu that allows you to select between blue/green etc. This is triggered even when you right click on an area that's between the actual interval rect items (when you click in the blank-space between rects).
        Want this to only be triggered when on an interval. And pass through to its parent otherwise.     
        
    #2025-07-22 18:18: - [x] Custom hover info tooltip text currently works, and the custom formatting function can be set via `self.format_item_tooltip_fn = _custom_format_tooltip_for_rect_data`. An example is provided in 
            
    Usage:
        Example 1 (basic):
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, main
            active_interval_rects_item = IntervalRectsItem(data)
            
            ## Add the active_interval_rects_item to the main_plot_widget: 
            main_plot_widget = spike_raster_window.spike_raster_plt_2d.plots.main_plot_widget # PlotItem
            main_plot_widget.addItem(active_interval_rects_item)

            ## Remove the active_interval_rects_item:
            main_plot_widget.removeItem(active_interval_rects_item)

            
        Example 2 (with custom tooltip function):
        
            def _custom_format_tooltip_for_rect_data(rect_index: int, rect_data_tuple: Tuple) -> str:
                start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
                ## get the optional label field if `rect_data_tuple` is a `IntervalRectsItemData` instead of a plain tuple
                a_label = None
                if not isinstance(rect_data_tuple, Tuple):
                    a_label = rect_data_tuple.label
                
                end_t = start_t + duration_t
                if a_label:
                    tooltip_text = f"{a_label}\n{name}[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}"
                else:
                    tooltip_text = f"{name}[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}"
                return tooltip_text


            # Build the rendered interval item:
            new_interval_rects_item: IntervalRectsItem = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource, format_tooltip_fn=deepcopy(_custom_format_tooltip_for_rect_data))
            new_interval_rects_item._current_hovered_item_tooltip_format_fn = deepcopy(_custom_format_tooltip_for_rect_data)
        
    """
    pressed = False
    clickable = True
    hoverEnter = QtCore.pyqtSignal()
    hoverExit = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()
    ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
    # sigDragged = QtCore.Signal(object)
    # sigPositionChangeFinished = QtCore.Signal(object)
    # sigPositionChanged = QtCore.Signal(object)
    # sigClicked = QtCore.Signal(object, object)
    
    

    def __init__(self, data, format_tooltip_fn=None, format_label_fn=None, debug_print=False, labels_min_pixel_width: float=4.0, labels_min_pixel_height: float=0.9, labels_padding_px: float=0.2, max_visible_labels: int=128, label_update_debounce_ms: int=40):
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        # note that the use of super() is often avoided because Qt does not 
        # allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
        self.generatePicture()
        self.setAcceptHoverEvents(True)
        self._current_hovered_rect = None  # Track which rectangle is currently hovered
        self._current_hovered_item_tooltip_format_fn = None
        if format_tooltip_fn is None:
            format_tooltip_fn = self._default_format_tooltip_for_rect_data
        if format_label_fn is None:
            # format_label_fn = self._default_format_tooltip_for_rect_data            
            pass ## no labels when not explicitly set


        self._current_hovered_item_tooltip_format_fn = format_tooltip_fn
        self._item_label_format_fn = format_label_fn

        self.labels_min_pixel_width = labels_min_pixel_width
        self.labels_min_pixel_height = labels_min_pixel_height
        self.labels_padding_px = labels_padding_px
        self.max_visible_labels = max_visible_labels
        self.label_update_debounce_ms = label_update_debounce_ms
        self._labels = []
        self._active_label_items = {}
        self._label_viewbox = None
        self._label_viewbox_connection = None
        self._label_metadata_needs_rebuild = True
        self._label_start_t = np.array([], dtype=float)
        self._label_end_t = np.array([], dtype=float)
        self._label_y = np.array([], dtype=float)
        self._label_height = np.array([], dtype=float)
        self._label_text = np.array([], dtype=object)
        self._label_sort_order = np.array([], dtype=int)
        self._label_max_duration_t = 0.0
        self._label_update_timer = QtCore.QTimer()
        self._label_update_timer.setSingleShot(True)
        self._label_update_timer.timeout.connect(self._refresh_visible_labels_from_viewbox)
        self.rebuild_label_items()
        


    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        
        # White background bars:
        p.setPen(pg.mkPen('w'))
        p.setBrush(pg.mkBrush('r'))
        
        # for (series_offset, start_t, duration_t) in self.data:
            # # QRectF: (left, top, width, height)
            # p.drawRect(QtCore.QRectF(start_t, series_offset-series_height, duration_t, series_height))
            
        for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in self.data:
            p.setPen(pen)
            p.setBrush(brush) # filling of the rectangles by a passed color:
            # p.drawRect(QtCore.QRectF(start_t, series_vertical_offset-series_height, duration_t, series_height)) # QRectF: (left, top, width, height)
            p.drawRect(QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height)) # QRectF: (left, top, width, height)

        p.end()
    
    def update_data(self, new_data):
        """Update the data in-place and regenerate the picture
        
        Args:
            new_data: List of tuples or IntervalRectsItemData objects with format:
                (start_t, series_vertical_offset, duration_t, series_height, pen, brush) or
                IntervalRectsItemData(...)
        """
        self.data = new_data
        self.generatePicture()
        self.update()
        self.rebuild_label_items()
    
    def paint(self, p, *args):
        self._ensure_label_viewbox_connection()
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())


    @property
    def format_item_tooltip_fn(self) -> Callable:
        """The format_item_tooltip_fn property."""
        return self._current_hovered_item_tooltip_format_fn
    @format_item_tooltip_fn.setter
    def format_item_tooltip_fn(self, value: Callable):
        is_changing: bool = (self._current_hovered_item_tooltip_format_fn != value)
        self._current_hovered_item_tooltip_format_fn = value
        # if is_changing:
        #     self.rebuild_label_items()


    @property
    def item_label_format_fn(self):
        """The item_label_format_fn property."""
        return self._item_label_format_fn
    @item_label_format_fn.setter
    def item_label_format_fn(self, value):
        is_changing: bool = (self._item_label_format_fn != value)
        self._item_label_format_fn = value
        if is_changing:
            self.rebuild_label_items()
        
        

    def rebuild_label_items(self, debug_print: bool=False):
        """Rebuilds lightweight label metadata without creating one text item per interval."""
        if debug_print:
            print(f'IntervalRectsItem.rebuild_label_items(...): rebuilding label metadata.')
        self._clear_active_label_items()
        self._label_metadata_needs_rebuild = True
        self._build_label_metadata()
        self._schedule_label_refresh()

        if debug_print:
            print(f'\tdone.')


    def _format_label_text_for_rect_data(self, rect_index: int, rect_data_tuple: Tuple) -> Optional[str]:
        """Return the display label for a rectangle, preferring a custom formatter when provided."""
        label_text = self.item_label_format_fn(rect_index=rect_index, rect_data_tuple=rect_data_tuple) if self.item_label_format_fn is not None else getattr(rect_data_tuple, 'label', None)
        if label_text is None:
            return None
        try:
            if np.isnan(label_text):
                return None
        except TypeError:
            pass
        label_text = str(label_text)
        if len(label_text) == 0:
            return None
        return label_text


    def _build_label_metadata(self):
        """Build array metadata for label culling while keeping QGraphicsItem count bounded."""
        if not self._label_metadata_needs_rebuild:
            return
        starts, ends, ys, heights, labels = [], [], [], [], []
        for rect_index, rect_data_tuple in enumerate(self.data):
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
            label_text = self._format_label_text_for_rect_data(rect_index=rect_index, rect_data_tuple=rect_data_tuple)
            if label_text is None:
                continue
            starts.append(float(start_t))
            ends.append(float(start_t + duration_t))
            ys.append(float(series_vertical_offset))
            heights.append(float(series_height))
            labels.append(label_text)
        self._label_start_t = np.asarray(starts, dtype=float)
        self._label_end_t = np.asarray(ends, dtype=float)
        self._label_y = np.asarray(ys, dtype=float)
        self._label_height = np.asarray(heights, dtype=float)
        self._label_text = np.asarray(labels, dtype=object)
        self._label_sort_order = np.argsort(self._label_start_t) if len(self._label_start_t) > 0 else np.array([], dtype=int)
        durations = self._label_end_t - self._label_start_t
        self._label_max_duration_t = float(np.nanmax(durations)) if len(durations) > 0 else 0.0
        self._label_metadata_needs_rebuild = False


    def _clear_active_label_items(self):
        """Hide all active labels while keeping the small item pool available for reuse."""
        for a_text_item in self._active_label_items.values():
            a_text_item.setVisible(False)
        self._active_label_items = {}


    def _label_font(self):
        if len(self._labels) > 0:
            return self._labels[0].textItem.font()
        app = QtWidgets.QApplication.instance()
        return app.font() if app is not None else QtGui.QFont()


    def _acquire_label_item(self) -> Optional[CustomRectBoundedTextItem]:
        active_items = set(self._active_label_items.values())
        for a_text_item in self._labels:
            if a_text_item not in active_items:
                return a_text_item
        if len(self._labels) >= self.max_visible_labels:
            return None
        a_text_item = CustomRectBoundedTextItem(rect=QtCore.QRectF(), text='', parent=self)
        a_text_item.setVisible(False)
        self._labels.append(a_text_item)
        return a_text_item


    def _assign_label_item(self, metadata_index: int):
        a_text_item = self._active_label_items.get(metadata_index, None)
        if a_text_item is None:
            a_text_item = self._acquire_label_item()
            if a_text_item is None:
                return
            self._active_label_items[metadata_index] = a_text_item
        a_rect = QtCore.QRectF(float(self._label_start_t[metadata_index]), float(self._label_y[metadata_index]), float(self._label_end_t[metadata_index] - self._label_start_t[metadata_index]), float(self._label_height[metadata_index]))
        label_text = str(self._label_text[metadata_index])
        a_text_item.desired_text_rect = a_rect
        a_text_item.original_text = label_text
        a_text_item.setText(label_text)
        a_text_item.updatePosition()
        a_text_item.setVisible(True)
        if hasattr(a_text_item, 'updateTransform'):
            a_text_item.updateTransform(force=True)
        a_text_item.update()


    def _visible_label_metadata_indices(self, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
        if len(self._label_start_t) == 0:
            return np.array([], dtype=int)
        x_min, x_max = sorted([float(x_range[0]), float(x_range[1])])
        y_min, y_max = sorted([float(y_range[0]), float(y_range[1])])
        sorted_starts = self._label_start_t[self._label_sort_order]
        start_lower_bound = x_min - max(0.0, self._label_max_duration_t)
        sorted_i0 = int(np.searchsorted(sorted_starts, start_lower_bound, side='left'))
        sorted_i1 = int(np.searchsorted(sorted_starts, x_max, side='right'))
        candidate_indices = self._label_sort_order[sorted_i0:sorted_i1]
        if len(candidate_indices) == 0:
            return candidate_indices
        label_y0 = np.minimum(self._label_y[candidate_indices], self._label_y[candidate_indices] + self._label_height[candidate_indices])
        label_y1 = np.maximum(self._label_y[candidate_indices], self._label_y[candidate_indices] + self._label_height[candidate_indices])
        visible_mask = (self._label_end_t[candidate_indices] >= x_min) & (self._label_start_t[candidate_indices] <= x_max) & (label_y1 >= y_min) & (label_y0 <= y_max)
        return candidate_indices[visible_mask]


    def refresh_visible_labels(self, canvas_width_px: int, canvas_height_px: int, x_range: Optional[Tuple[float, float]]=None, y_range: Optional[Tuple[float, float]]=None, immediate: bool=True, force_render_all: bool=False):
        """Show labels for the given canvas size, optionally bypassing text-fit culling for export."""
        self._build_label_metadata()
        if len(self._label_start_t) == 0:
            self._clear_active_label_items()
            return
        if (x_range is None) or (y_range is None):
            view_box = self._resolve_label_viewbox()
            if view_box is None:
                self._clear_active_label_items()
                return
            view_range = view_box.viewRange()
            x_range = view_range[0] if x_range is None else x_range
            y_range = view_range[1] if y_range is None else y_range
        canvas_width_px_f = max(1.0, float(canvas_width_px))
        canvas_height_px_f = max(1.0, float(canvas_height_px))
        x_min, x_max = sorted([float(x_range[0]), float(x_range[1])])
        y_min, y_max = sorted([float(y_range[0]), float(y_range[1])])
        x_span = x_max - x_min
        y_span = y_max - y_min
        if (x_span <= 0.0) or (y_span <= 0.0):
            self._clear_active_label_items()
            return
        metrics = QtGui.QFontMetricsF(self._label_font())
        fitting_candidates = []
        for metadata_index in self._visible_label_metadata_indices((x_min, x_max), (y_min, y_max)):
            label_text = str(self._label_text[metadata_index])
            rect_width_px = abs((self._label_end_t[metadata_index] - self._label_start_t[metadata_index]) / x_span) * canvas_width_px_f
            rect_height_px = abs(self._label_height[metadata_index] / y_span) * canvas_height_px_f
            if (rect_width_px < self.labels_min_pixel_width) or (rect_height_px < self.labels_min_pixel_height):
                continue
            if force_render_all:
                fitting_candidates.append((rect_width_px, int(metadata_index)))
                continue
            text_rect = metrics.boundingRect(label_text)
            if ((text_rect.width() + self.labels_padding_px) <= rect_width_px) and ((text_rect.height() + self.labels_padding_px) <= rect_height_px):
                fitting_candidates.append((rect_width_px, int(metadata_index)))
        fitting_candidates.sort(reverse=True)
        desired_indices = [metadata_index for _, metadata_index in fitting_candidates[:self.max_visible_labels]]
        desired_set = set(desired_indices)
        for metadata_index, a_text_item in list(self._active_label_items.items()):
            if metadata_index not in desired_set:
                a_text_item.setVisible(False)
                del self._active_label_items[metadata_index]
        for metadata_index in desired_indices:
            self._assign_label_item(metadata_index)


    def _refresh_visible_labels_from_viewbox(self):
        view_box = self._resolve_label_viewbox()
        if view_box is None:
            self._clear_active_label_items()
            return
        view_range = view_box.viewRange()
        self.refresh_visible_labels(canvas_width_px=max(1, int(view_box.width())), canvas_height_px=max(1, int(view_box.height())), x_range=view_range[0], y_range=view_range[1], immediate=True)


    def _schedule_label_refresh(self):
        if len(self.data) == 0:
            self._clear_active_label_items()
            return
        if self.label_update_debounce_ms <= 0:
            self._refresh_visible_labels_from_viewbox()
        else:
            self._label_update_timer.start(self.label_update_debounce_ms)


    def _on_label_viewbox_range_changed(self, *args):
        self._schedule_label_refresh()


    def _disconnect_label_viewbox(self):
        if (self._label_viewbox is not None) and (self._label_viewbox_connection is not None):
            try:
                self._label_viewbox.sigRangeChanged.disconnect(self._label_viewbox_connection)
            except (TypeError, RuntimeError):
                pass
        self._label_viewbox = None
        self._label_viewbox_connection = None


    def _resolve_label_viewbox(self):
        """Find the actual ViewBox for label range/size work; `getViewBox()` may fall back to the GraphicsView during scene changes."""
        parent_item = self.parentItem()
        while parent_item is not None:
            if hasattr(parent_item, 'implements') and parent_item.implements('ViewBox') and hasattr(parent_item, 'sigRangeChanged') and hasattr(parent_item, 'viewRange'):
                return parent_item
            parent_item = parent_item.parentItem()
        view_box = self.getViewBox()
        if (view_box is not None) and hasattr(view_box, 'sigRangeChanged') and hasattr(view_box, 'viewRange'):
            return view_box
        return None


    def _ensure_label_viewbox_connection(self):
        view_box = self._resolve_label_viewbox()
        if view_box is None:
            return
        if self._label_viewbox is view_box:
            return
        self._disconnect_label_viewbox()
        self._label_viewbox = view_box
        self._label_viewbox_connection = self._on_label_viewbox_range_changed
        view_box.sigRangeChanged.connect(self._label_viewbox_connection)
        self._schedule_label_refresh()


    def itemChange(self, change, value):
        if not hasattr(self, '_label_viewbox'):
            return pg.GraphicsObject.itemChange(self, change, value)
        try:
            scene_has_changed = QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged
        except AttributeError:
            scene_has_changed = QtWidgets.QGraphicsItem.ItemSceneHasChanged
        if change == scene_has_changed:
            if value is None:
                self._disconnect_label_viewbox()
                self._clear_active_label_items()
            else:
                self._ensure_label_viewbox_connection()
                self._schedule_label_refresh()
        return pg.GraphicsObject.itemChange(self, change, value)



    ## Copy Constructors:
    def __copy__(self):
        independent_data_copy = ColorDataframeColumnHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy, format_tooltip_fn=self.format_item_tooltip_fn, format_label_fn=self.item_label_format_fn, labels_min_pixel_width=self.labels_min_pixel_width, labels_min_pixel_height=self.labels_min_pixel_height, labels_padding_px=self.labels_padding_px, max_visible_labels=self.max_visible_labels, label_update_debounce_ms=self.label_update_debounce_ms)
    
    def __deepcopy__(self, memo):
        independent_data_copy = ColorDataframeColumnHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy, format_tooltip_fn=copy.deepcopy(self.format_item_tooltip_fn, memo), format_label_fn=copy.deepcopy(self.item_label_format_fn, memo), labels_min_pixel_width=self.labels_min_pixel_width, labels_min_pixel_height=self.labels_min_pixel_height, labels_padding_px=self.labels_padding_px, max_visible_labels=self.max_visible_labels, label_update_debounce_ms=self.label_update_debounce_ms)
        # return IntervalRectsItem(copy.deepcopy(self.data, memo))


    # ==================================================================================================================== #
    # Events Copied from https://github.com/CommanderPho/pyqt-xcode/blob/master/menurect.py                                #
    # ==================================================================================================================== #

    def hoverEnterEvent(self, event):
        if self.clickable:
            self.hoverEnter.emit()


    def hoverMoveEvent(self, event):
        """Handle hover move events to show tooltips for individual rectangles."""
        if not self.clickable:
            return
            
        # Get the position in item coordinates
        pos = event.pos()
        
        # Find which rectangle (if any) contains this position
        hovered_rect_index = self._get_rect_at_position(pos)
        
        if hovered_rect_index != self._current_hovered_rect:
            self._current_hovered_rect = hovered_rect_index
            
            if hovered_rect_index is not None:
                # Show tooltip for this rectangle
                global_pos = event.screenPos()
                self._show_tooltip_for_rect(hovered_rect_index, QtCore.QPoint(int(global_pos.x()), int(global_pos.y())))
            else:
                # Hide tooltip when not over any rectangle
                QtWidgets.QToolTip.hideText()

    def hoverLeaveEvent(self, event):
        if self.clickable:
            self.hoverExit.emit()
            # Hide tooltip when leaving the item
            QtWidgets.QToolTip.hideText()
            self._current_hovered_rect = None
            

    def mousePressEvent(self, event):
        if self.clickable:
            pressed = True


    def mouseReleaseEvent(self, event):
        if self.clickable:
            pressed = False
            self.clicked.emit()

    # ==================================================================================================================================================================================================================================================================================== #
    # Hover Event Handlers                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    def _get_rect_at_position(self, pos):
        """
        Find which rectangle (if any) contains the given position.
        Returns the index of the rectangle, or None if no rectangle contains the position.
        
        Args:
            pos: QtCore.QPointF in item coordinates
            
        Returns:
            int or None: Index of the rectangle containing the position, or None
        """
        for i, (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in enumerate(self.data):
            rect = QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height)
            if rect.contains(pos):
                return i
        return None
    
    @classmethod
    def _default_format_tooltip_for_rect_data(cls, rect_index: int, rect_data_tuple: Tuple) -> str:
        """ rect_data_tuple = self.data[rect_index]
        start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
        """
        start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
        ## get the optional label field if `rect_data_tuple` is a `IntervalRectsItemData` instead of a plain tuple
        a_label = None
        if not isinstance(rect_data_tuple, Tuple):
            a_label = rect_data_tuple.label
        
        end_t = start_t + duration_t
        if a_label:
            tooltip_text = f"{a_label}\nItem[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}"
        else:
            tooltip_text = f"Item[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}"
        return tooltip_text



    def _show_tooltip_for_rect(self, rect_index, global_pos):
        """
        Show tooltip for the specified rectangle.
        
        Args:
            rect_index: Index of the rectangle in self.data
            global_pos: Global screen position for tooltip
        """
        if rect_index is None or rect_index >= len(self.data):
            return
        rect_data_tuple = self.data[rect_index]
        assert self._current_hovered_item_tooltip_format_fn is not None, f"self._current_hovered_item_tooltip_format_fn is None!"
        # tooltip_text: str = self._default_format_tooltip_for_rect_data(rect_index=rect_index, rect_data_tuple=rect_data_tuple)
        tooltip_text: str = self._current_hovered_item_tooltip_format_fn(rect_index=rect_index, rect_data_tuple=rect_data_tuple)        
        QtWidgets.QToolTip.showText(global_pos, tooltip_text)
        

    def setToolTip(self, text):
        """
        Override setToolTip to provide custom behavior.
        
        Args:
            text: Tooltip text. If None or empty, enables per-rectangle tooltips.
                  If provided, shows this static text for the entire item.
        """
        print(f'WARNING: EpochRenderingMixin.setTooltip(text: "{text}") was called, but this would set a single, static tooltip for the entire graphics item and is very unlikely to be what you want to do!')
        raise NotImplementedError(f'WARNING: EpochRenderingMixin.setTooltip(text: "{text}") was called, but this would set a single, static tooltip for the entire graphics item and is very unlikely to be what you want to do!')
        # self._custom_tooltip = text
        
        # if text:
        #     # If tooltip text is provided, disable custom per-rectangle tooltips
        #     self._use_custom_tooltips = False
        #     # Call parent implementation to set static tooltip
        #     super().setToolTip(text)
        # else:
        #     # If no text provided, enable custom per-rectangle tooltips
        #     self._use_custom_tooltips = True
        #     # Clear any existing static tooltip
        #     super().setToolTip("")
        

    # ==================================================================================================================== #
    # Context Menu and Interaction Handling                                                                                #
    # ==================================================================================================================== #
    def mouseShape(self):
        """
        Return a QPainterPath representing the clickable shape of the curve

        """
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape
    
    


    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        print(f'IntervalRectsItem.mouseClickEvent(ev: {ev})')
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            # if self.mouseShape().contains(ev.pos()):
            #     ev.accept()
            #     self.sigClicked.emit(self, ev)
                
            
            if self.raiseContextMenu(ev):
                ev.accept() # note that I think this means it won't pass the right click along to its parent view, might messup widget-wide menus

    def raiseContextMenu(self, ev):
        """ works to spawn the context menu in the appropriate location """
        print(f'IntervalRectsItem.raiseContextMenu(ev: {ev})')
        menu = self.getContextMenus()
        
        # Let the scene add on to the end of our context menu
        # (this is optional)
        # menu = self.scene().addParentContextMenus(self, menu, ev)
        
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        """ builds the context menus as needed """
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            # self.menu.setTitle(self.name+ " options..")
            self.menu.setTitle("IntervalRectItem options..")
            
            green = QtGui.QAction("Turn green", self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            
            blue = QtGui.QAction("Turn blue", self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            
            alpha = QtWidgets.QWidgetAction(self.menu)
            alphaSlider = QtWidgets.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    # Define context menu callbacks
    def setGreen(self):
        # self.pen = pg.mkPen('g')
        print(f'.setGreen()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('g')
            override_brush = pg.mkBrush('g')
            # self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
            a_label = None
            if not isinstance(a_tuple, Tuple):
                a_label = a_tuple.label
            self.data[i] = IntervalRectsItemData(start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush, label=a_label)

        
        # Need to regenerate picture
        self.generatePicture()
        # inform Qt that this item must be redrawn.
        self.update()

    def setBlue(self):
        # self.pen = pg.mkPen('b')
        # override_pen = pg.mkPen('b')
        print(f'.setBlue()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('b')
            override_brush = pg.mkBrush('b')
            # self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
            a_label = None
            if not isinstance(a_tuple, Tuple):
                a_label = a_tuple.label
            self.data[i] = IntervalRectsItemData(start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush, label=a_label)


        # Need to regenerate picture
        self.generatePicture()
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)
        


class CustomLegendItemSample(ReprPrintableItemMixin, ItemSample):
    """ A ItemSample that can render a legend item for `IntervalRectsItem`
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import CustomLegendItemSample
    
    legend = pg.LegendItem(offset=(-10, -10))
    legend.setParentItem(plt.graphicsItem())
    legend.setSampleType(CustomLegendItemSample)  

    """
    def __init__(self, item):
        super().__init__(item)
        self.item = item

    def paint(self, p, *args):
        # print(f'CustomItemSample.paint(self, p, *args)')
        if not isinstance(self.item, IntervalRectsItem):
            ## Call superclass paint
            # print(f'\t calling superclass, as type(self.item): {type(self.item)}')
            super().paint(p, *args)
        else:
            # Custom Implementation
            # print(f'\t calling custom implementation!')
            if not self.item.isVisible():
                p.setPen(pg.mkPen('w'))
                p.drawLine(0, 11, 20, 11) # draw flat white line
                return

            # Define the size of the rectangle
            rect_width = 20
            rect_height = 8

            # Calculate the top-left corner coordinates to center the rectangle
            top_left_x = (self.boundingRect().width() - rect_width) / 2
            top_left_y = (self.boundingRect().height() - rect_height) / 2

            ## start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data
            # print(f'len(self.item.data): {len(self.item.data)}')

            # The first item is representitive of all items, don't draw the item over-and-over
            use_only_first_items: bool = True

            for rect_data in self.item.data:
                pen, brush = rect_data[4], rect_data[5]
                if (pen is not None) or (brush is not None):                   
                    p.setPen(pen)
                    p.setBrush(brush)
                    # p.drawRect(QtCore.QRectF(2, 2, 16, 16))
                    p.drawRect(QtCore.QRectF(top_left_x, top_left_y, rect_width, rect_height))
                    if use_only_first_items:
                        return # break, only needed to draw one item

        # print(f'done.')



# ==================================================================================================================== #
# MAIN TESTING                                                                                                         #
# ==================================================================================================================== #
def main():
    # data = [  ## fields are (series_offset, start_t, duration_t).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
    
    
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
        
    series_start_offsets = [1, 5, 7]
    
    # Have series_offsets which are centers and series_start_offsets which are bottom edges:
    curr_border_color = pg.mkColor('r')
    curr_border_color.setAlphaF(0.8)

    curr_fill_color = pg.mkColor('w')
    curr_fill_color.setAlphaF(0.2)

    # build pen/brush from color
    curr_series_pen = pg.mkPen(curr_border_color)
    curr_series_brush = pg.mkBrush(curr_fill_color)
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (40.0, 0.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (41.0, 1.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (44.0, series_start_offsets[0], 4.0, 1.0, curr_series_pen, curr_series_brush),
    #     (45.0, series_start_offsets[-1], 4.0, 1.0, curr_series_pen, curr_series_brush),
    # ]
    data = []
    step_x_offset = 0.5
    for i in np.arange(len(series_start_offsets)):
        curr_x_pos = (40.0+(step_x_offset*float(i)))
        data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))
        
    
    item = IntervalRectsItem(data)

    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')
    # # Adjust the left margin
    # plt.getPlotItem().layout.setContentsMargins(100, 10, 10, 10)  # left, top, right, bottom


    # Add custom legend
    legend = pg.LegendItem(offset=(-10, -10))
    legend.setParentItem(plt.graphicsItem())
    legend.setSampleType(CustomLegendItemSample)    
    legend.addItem(item, 'Custom Rects')

    

    # series_start_offsets = [1, 5, 7]
    # curr_border_color = pg.mkColor('r')
    # curr_border_color.setAlphaF(0.8)
    # curr_fill_color = pg.mkColor('w')
    # curr_fill_color.setAlphaF(0.2)
    # curr_series_pen = pg.mkPen(curr_border_color)
    # curr_series_brush = pg.mkBrush(curr_fill_color)
    # data = []
    # step_x_offset = 0.5
    # for i in np.arange(len(series_start_offsets)):
    #     curr_x_pos = (40.0 + (step_x_offset * float(i)))
    #     data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))

    # item = IntervalRectsItem(data)
    # item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    # plt = pg.plot()
    # plt.addItem(item)
    # plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')

    # # Add custom legend
    # legend = CustomLegendItem(offset=(-10, -10))
    # legend.setParentItem(plt.graphicsItem())
    # legend.addItem(item, 'Custom Rects')

def main2():
    series_start_offsets = [1, 5, 7]
    curr_border_color = pg.mkColor('r')
    curr_border_color.setAlphaF(0.8)
    curr_fill_color = pg.mkColor('w')
    curr_fill_color.setAlphaF(0.2)
    curr_series_pen = pg.mkPen(curr_border_color)
    curr_series_brush = pg.mkBrush(curr_fill_color)
    data = []
    step_x_offset = 0.5
    for i in np.arange(len(series_start_offsets)):
        curr_x_pos = (40.0 + (step_x_offset * float(i)))
        data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))

    item = IntervalRectsItem(data)
    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')
    # Adjust the left margin
    # plt.getPlotItem().layout.setContentsMargins(100, 10, 10, 10)  # left, top, right, bottom
    plt.getPlotItem().layout.setContentsMargins(10, 10, 100, 10)  # left, top, right, bottom

    # Add custom legend in the right margin
    legend = LegendItem(offset=(100, -10))  # Adjust the x-offset as needed
    legend.setParentItem(plt.graphicsItem())
    legend.addItem(CustomLegendItemSample(item), 'Custom Rects')
    return plt, item, legend

if __name__ == '__main__':
    
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    # main()
    plt, item, legend = main2()
    pg.exec()
    