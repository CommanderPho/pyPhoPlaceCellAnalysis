from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from attrs import Factory, define, field

from qtpy import QtCore, QtWidgets

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.GraphicsScene.GraphicsScene import GraphicsScene
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ROI import Handle as PGROIHandle
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ROI import ROI as PGROI


@define(slots=False)
class Rois:
    """Container/manager for pg ROI instances living in a ViewBox.

    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.UserEditableROIMixin import UserEditableROIMixin, Rois
    """

    rois: List[PGROI] = field(default=Factory(list))
    vb: Optional[pg.ViewBox] = field(default=None)
    on_roi_added_callback: Optional[Callable[[PGROI], None]] = None
    on_roi_removed_callback: Optional[Callable[[PGROI], None]] = None
    on_roi_update_callback: Optional[Callable[[PGROI], None]] = None

    def __attrs_post_init__(self):
        self.rebind_existing_rois()

    # ------------------------------------------------------------------------------------------
    @classmethod
    def add_Bapun_maze_ROIs(cls, vb, maze_name: str, maze_cross_boundaries: bool=False):
        """Convenience helper that adds a predefined set of Maze ROIs and registers them."""
        rois = []

        if maze_name in ['maze1', 'n']:
            if maze_cross_boundaries:
                rois.append(pg.LineROI((-80.0, 50.0), (-56.38816352628667, 49.75440987840928), width=5, pen=(1,9)))
                rois.append(pg.LineROI((-72.72778253330875, -23.25280628626046), (-57.88707874532102, -23.461573517913862), width=5, pen=(1,9)))
                rois.append(pg.LineROI((-53.2, -27.9), (-44.698415860398285, -35.85425336824297), width=5, pen=(1,9)))
                rois.append(pg.LineROI((-8.85336044562508, 46.55120269411165), (7.126017351691745, 36.38273324555965), width=5, pen=(1,9)))
                rois.append(pg.LineROI((16.699999999999996, 34.0), (35.507062318788684, 40.82021219041976), width=5, pen=(1,9)))
                rois.append(pg.LineROI((50.900000000000006, -46.849999999999994), (74.6907601353054, -33.97113624293468), width=5, pen=(1,9)))
            else:
                ## full maze ROIs:
                rois.append(pg.EllipseROI((-21.1697, 67.3724), (50.1406, 44.9173), pen=(255, 255, 0)))
                rois.append(pg.EllipseROI((-55.3671, -89.3382), (49.5413, 49.9337), pen=(255, 255, 0)))
                rois.append(pg.EllipseROI((94.5122, -88.7958), (38.088, 48.3889), pen=(255, 255, 0)))
                rois.append(pg.EllipseROI((-87.9834, 106.658), (51.6054, 37.7826), pen=(255, 255, 0)))
                rois.append(pg.RectROI((-72.4633, -30.6643), (11.4325, 85.5815), pen=(255, 255, 0)))

        elif maze_name.lower() in ['maze2', 'u']:
            if maze_cross_boundaries:
                rois.append(pg.LineROI((-70.0, 40.0), (-40, 40), width=5, pen=(1,9)))
                rois.append(pg.LineROI((58.55488015485466, 47.21734016790171), (82.7729169919969, 46.717373983958375), width=5, pen=(1,9)))
                rois.append(pg.LineROI((0, -82), (0, -40), width=5, pen=(1,9)))
            else:
                rois.append(pg.EllipseROI((-82.0021, 43.3586), (49.0323, 54.0987), pen=(255, 255, 0)))
                rois.append(pg.EllipseROI((42.2661, 111.873), (52.88, 53.7065), pen=(255, 255, 0)))
                
        else:
            raise NotImplementedError(f'Unexpected maze identifier - maze_name: "{maze_name}"')

        _obj = cls(rois=rois, vb=vb)
        _obj.add_all_rois()
        return _obj

    @classmethod
    def from_view_box(cls, vb: pg.ViewBox):
        """Builds a Rois manager from existing ROI graphics already inside the ViewBox."""
        discovered: List[PGROI] = []
        for item in getattr(vb, 'addedItems', []):
            if isinstance(item, PGROI) and item not in discovered:
                discovered.append(item)
        return cls(rois=discovered, vb=vb)

    # ------------------------------------------------------------------------------------------
    def set_callbacks(
        self,
        on_added: Optional[Callable[[PGROI], None]] = None,
        on_removed: Optional[Callable[[PGROI], None]] = None,
        on_updated: Optional[Callable[[PGROI], None]] = None,
    ):
        self.on_roi_added_callback = on_added
        self.on_roi_removed_callback = on_removed
        self.on_roi_update_callback = on_updated

    def rebind_existing_rois(self):
        """Ensures internal ROI references are hooked to update notifications."""
        for roi in list(self.rois):
            self._connect_roi_signals(roi)

    def set_view_box(self, vb: Optional[pg.ViewBox]):
        self.vb = vb
        return self

    def add_roi(self, roi: PGROI, add_to_view: bool = True):
        """Registers and optionally adds ROI to the managed ViewBox."""
        if roi is None:
            return None
        was_new = False
        if roi not in self.rois:
            self.rois.append(roi)
            was_new = True
        self._connect_roi_signals(roi)
        if add_to_view and self.vb is not None and roi not in getattr(self.vb, 'addedItems', []):
            self.vb.addItem(roi)
        if was_new and self.on_roi_added_callback is not None:
            self.on_roi_added_callback(roi)
        return roi

    def remove_roi(self, roi: PGROI, remove_from_view: bool = True):
        if roi is None or roi not in self.rois:
            return
        try:
            roi.sigRegionChanged.disconnect(self.on_roi_update)
        except (TypeError, RuntimeError):
            pass

        self.rois.remove(roi)
        if remove_from_view and self.vb is not None:
            try:
                self.vb.removeItem(roi)
            except Exception:
                pass

        if self.on_roi_removed_callback is not None:
            self.on_roi_removed_callback(roi)

    def add_all_rois(self):
        for roi in list(self.rois):
            self.add_roi(roi, add_to_view=True)

    def remove_all_rois(self):
        for roi in list(self.rois):
            self.remove_roi(roi, remove_from_view=True)

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def get_roi_coordinates(roi):
        handles = roi.getHandles()
        points = []
        for h in handles:
            mapped_point = roi.mapToParent(h.pos())
            points.append((mapped_point.x(), mapped_point.y()))
        return points


    def print_maze_rois(self) -> List[str]:
        """Emit python initializers that recreate each ROI."""
        def _format_tuple(pt):
            return f"({float(pt[0]):.8f}, {float(pt[1]):.8f})"

        lines = []
        for roi in self.rois:
            roi_type = type(roi).__name__
            state = roi.saveState()

            if roi_type == "LineROI":
                handles = self.get_roi_coordinates(roi)
                p0, p1 = handles[0], handles[-1]
                lines.append(
                    f"rois.append(pg.LineROI({_format_tuple(p0)}, {_format_tuple(p1)}, "
                    f"width={getattr(roi, 'currentPen', roi.pen).width()}, pen={roi.pen.color().getRgb()[:3]}))"
                )
            elif roi_type == "RectROI":
                pos = tuple(state["pos"])
                size = tuple(state["size"])
                lines.append(
                    f"rois.append(pg.RectROI({_format_tuple(pos)}, {_format_tuple(size)}, "
                    f"pen={roi.pen.color().getRgb()[:3]}))"
                )
            elif roi_type == "EllipseROI":
                pos = tuple(state["pos"])
                size = tuple(state["size"])
                lines.append(
                    f"rois.append(pg.EllipseROI({_format_tuple(pos)}, {_format_tuple(size)}, "
                    f"pen={roi.pen.color().getRgb()[:3]}))"
                )
            else:
                # Fallback: dump generic ROI state so you can restore manually later.
                lines.append(
                    f"# Unsupported ROI type {roi_type}; state = {state!r}"
                )

        print("\n".join(lines))
        return lines



    def on_roi_update(self, an_roi):
        if self.on_roi_update_callback is not None:
            self.on_roi_update_callback(an_roi)

    def _connect_roi_signals(self, roi: PGROI):
        if roi is None:
            return
        try:
            roi.sigRegionChanged.disconnect(self.on_roi_update)
        except (TypeError, RuntimeError):
            pass
        roi.sigRegionChanged.connect(self.on_roi_update)

    # ------------------------------------------------------------------------------------------
    @classmethod
    def find_single_roi_crossings(cls, roi, x_data, y_data):
        """
        Returns indices i where the segment (x[i],y[i]) -> (x[i+1],y[i+1])
        intersects the LineROI.
        """
        # 1. Get ROI Handle positions in Plot Coordinates
        handles = roi.getHandles()
        p1 = roi.mapToParent(handles[0].pos())
        p2 = roi.mapToParent(handles[1].pos())
        
        # ROI Vector (A -> B)
        rx1, ry1 = p1.x(), p1.y()
        rx2, ry2 = p2.x(), p2.y()
        
        # 2. Vector math preparation
        # ROI directional vector
        roi_vec = np.array([rx2 - rx1, ry2 - ry1])
        roi_len_sq = np.dot(roi_vec, roi_vec) # Length squared for normalization
        
        # Normal vector to the ROI (for calculating distance from line)
        # If ROI vector is (dx, dy), Normal is (-dy, dx)
        roi_norm = np.array([-(ry2 - ry1), (rx2 - rx1)])
        
        # Stack data into (N, 2) array
        data_points = np.column_stack((x_data, y_data))
        
        # Vector from ROI start to Data points
        # shape: (N, 2)
        vec_to_data = data_points - np.array([rx1, ry1])
        
        # 3. Calculate "Signed Distance" (proportional)
        # Dot product with normal vector tells us which side of the line we are on
        signed_dists = np.dot(vec_to_data, roi_norm)
        
        # Find where signs change (crossing the infinite line)
        # signal[i] * signal[i+1] < 0 implies a crossing
        cross_candidates = np.where(np.diff(np.signbit(signed_dists)))[0]
        
        valid_crossings = []
        
        # 4. Filter candidates to ensure they are within ROI segment bounds
        for i in cross_candidates:
            # We need the exact intersection point of the two segments:
            # Segment 1: Data[i] to Data[i+1]
            # Segment 2: ROI_Start to ROI_End
            
            # We know they cross the infinite line, now check the "along-line" position.
            # Simple projection approach:
            
            # Get fraction 't' along the data segment where crossing occurs
            d1 = signed_dists[i]
            d2 = signed_dists[i+1]
            
            # Avoid division by zero if exact overlap (unlikely in floats)
            if (d2 - d1) == 0: continue
                
            t = d1 / (d1 - d2) # Linear interpolation for zero-crossing
            
            # Calculate the exact intersection point coordinates
            intersect_pt = data_points[i] + t * (data_points[i+1] - data_points[i])
            
            # Project intersection point onto ROI vector to see if it's on the segment
            # Projection P = (Intersect - ROI_Start) . ROI_Vec
            proj_vec = intersect_pt - np.array([rx1, ry1])
            dot_prod = np.dot(proj_vec, roi_vec)
            
            # Check if 0 <= projection <= |ROI|^2
            if 0 <= dot_prod <= roi_len_sq:
                valid_crossings.append(i)
                
        return valid_crossings


    def find_all_roi_crossings(self, x_data, y_data):
        """ finds the crossings where the positions given by x_data/y_data cross each of the ROIs 
        """
        found_crossings = []
        for roi in self.rois:
            found_crossings.append(self.find_single_roi_crossings(roi, x_data=x_data, y_data=y_data))
        return found_crossings
    
            


ROIConstructor = Callable[[QtCore.QPointF], PGROI]


class UserEditableROIMixin:
    """Mixin that wires a `pg.PlotItem` to accept user-created ROIs via context menus.

    When enabled, right-clicking the associated plot item opens a context menu that allows the
    user to select which ROI primitive to insert at the cursor location. Right-clicking any
    ROI created through the mixin opens a removal prompt.

    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.UserEditableROIMixin import UserEditableROIMixin, Rois


    """

    _user_roi_menu: Optional[QtWidgets.QMenu] = None
    _user_roi_parent_plot: Optional[pg.PlotItem] = None
    _user_roi_view_box: Optional[pg.ViewBox] = None
    _user_roi_scene: Optional[GraphicsScene] = None
    _user_roi_factories: Optional[Dict[str, ROIConstructor]] = None
    _user_rois: Optional[List[PGROI]] = None
    _user_rois_manager: Optional[Rois] = None

    # Public API ---------------------------------------------------------------------------------
    def enable_user_editable_rois(
        self,
        parent_plot_item: pg.PlotItem,
        roi_factories: Optional[Dict[str, ROIConstructor]] = None,
        rois_manager: Optional[Rois] = None,
        adopt_existing_scene_rois: bool = True,
    ) -> None:
        """Attach ROI editing behavior to `parent_plot_item`.

        Args:
            parent_plot_item: The plot item that should receive right-click menus.
            roi_factories: Optional mapping that describes which ROI types are offered. Each
                factory receives the cursor position (in view coordinates) and must return
                a configured ROI instance.
        """
        if parent_plot_item is None:
            raise ValueError("parent_plot_item is required to enable user-editable ROIs.")

        self.disable_user_editable_rois(remove_existing=False)

        scene = parent_plot_item.scene()
        if scene is None:
            raise RuntimeError("Cannot enable ROI editing before the plot item belongs to a scene.")

        self._user_roi_parent_plot = parent_plot_item
        self._user_roi_view_box = parent_plot_item.getViewBox()
        self._user_roi_scene = scene
        self._user_roi_factories = roi_factories or self._build_default_roi_factories()

        self._initialize_rois_manager(rois_manager=rois_manager, adopt_existing_scene_rois=adopt_existing_scene_rois)
        scene.sigMouseClicked.connect(self._handle_user_roi_mouse_click)

    def disable_user_editable_rois(self, remove_existing: bool = True) -> None:
        """Detach ROI editing behavior and optionally remove existing ROIs."""
        if self._user_roi_scene is not None:
            try:
                self._user_roi_scene.sigMouseClicked.disconnect(self._handle_user_roi_mouse_click)
            except (TypeError, RuntimeError):
                pass

        if remove_existing:
            self._remove_all_user_rois()

        self._user_roi_parent_plot = None
        self._user_roi_view_box = None
        self._user_roi_scene = None
        self._user_roi_factories = None
        self._user_rois = None
        self._user_rois_manager = None
        self._clear_active_roi_menu()

    @property
    def user_defined_rois(self) -> List[PGROI]:
        """Return the list of ROIs created by the user."""
        if self._user_rois_manager is not None:
            return list(self._user_rois_manager.rois)
        return list(self._user_rois or [])

    # Hooks for implementors --------------------------------------------------------------------
    def UserEditableROIMixin_on_roi_added(self, roi: PGROI) -> None:  # noqa: N802 (API parity)
        """Called when a new ROI is added. Implementors can override."""

    def UserEditableROIMixin_on_roi_removed(self, roi: PGROI) -> None:  # noqa: N802
        """Called after an ROI has been removed. Implementors can override."""

    def UserEditableROIMixin_on_roi_updated(self, roi: PGROI) -> None:  # noqa: N802
        """Called whenever a managed ROI finishes moving/scaling. Implementors can override."""

    # Internal helpers --------------------------------------------------------------------------
    def _initialize_rois_manager(self, rois_manager: Optional[Rois], adopt_existing_scene_rois: bool):
        if rois_manager is not None:
            rois_manager.set_view_box(self._user_roi_view_box)
            rois_manager.rebind_existing_rois()
            manager = rois_manager
        elif adopt_existing_scene_rois:
            manager = self._discover_existing_rois_manager()
            if manager is None:
                manager = Rois(vb=self._user_roi_view_box)
        else:
            manager = Rois(vb=self._user_roi_view_box)

        manager.set_callbacks(on_updated=self.UserEditableROIMixin_on_roi_updated)
        self._user_rois_manager = manager
        self._user_rois = manager.rois
        for roi in list(self._user_rois):
            self._attach_roi_event_listeners(roi, add_to_plot=False)

    def _discover_existing_rois_manager(self) -> Optional[Rois]:
        if self._user_roi_view_box is None:
            return None
        manager = Rois.from_view_box(self._user_roi_view_box)
        if len(manager.rois) == 0:
            return None
        return manager

    def _handle_user_roi_mouse_click(self, mouse_event) -> None:
        """Scene-level mouse handler that routes right clicks to ROI creation/removal."""
        if (
            self._user_roi_scene is None
            or self._user_roi_parent_plot is None
            or mouse_event.button() != QtCore.Qt.MouseButton.RightButton
        ):
            return

        scene_pos = mouse_event.scenePos()
        if not self._is_scene_pos_on_parent_plot(scene_pos):
            return

        roi = self._find_roi_at_scene_pos(scene_pos)
        if roi is not None:
            mouse_event.accept()
            self._prompt_roi_removal(roi, mouse_event.screenPos())
            return

        view_point = self._user_roi_view_box.mapSceneToView(scene_pos)
        mouse_event.accept()
        self._prompt_roi_creation(view_point, mouse_event.screenPos())

    def _prompt_roi_creation(self, view_point: QtCore.QPointF, screen_point: QtCore.QPointF) -> None:
        if not self._user_roi_factories:
            return

        menu_parent = self if isinstance(self, QtWidgets.QWidget) else None
        menu = QtWidgets.QMenu(menu_parent)
        menu.setTitle("Insert ROI")
        for roi_name, factory in self._user_roi_factories.items():
            action = menu.addAction(f"Add {roi_name}")
            action.triggered.connect(
                lambda checked=False, name=roi_name, pos=view_point: self._create_roi(name, pos)
            )

        menu.aboutToHide.connect(self._clear_active_roi_menu)
        self._user_roi_menu = menu
        menu.popup(self._qt_point_from_point(screen_point))

    def _prompt_roi_removal(self, roi: PGROI, screen_point: QtCore.QPointF) -> None:
        menu_parent = self if isinstance(self, QtWidgets.QWidget) else None
        menu = QtWidgets.QMenu(menu_parent)
        remove_action = menu.addAction("Remove ROI")
        remove_action.triggered.connect(lambda checked=False, target=roi: self._remove_roi(target))

        menu.aboutToHide.connect(self._clear_active_roi_menu)
        self._user_roi_menu = menu
        menu.popup(self._qt_point_from_point(screen_point))

    def _create_roi(self, roi_name: str, view_point: QtCore.QPointF) -> Optional[PGROI]:
        if (
            self._user_roi_factories is None
            or roi_name not in self._user_roi_factories
            or self._user_roi_parent_plot is None
        ):
            return None

        roi = self._user_roi_factories[roi_name](view_point)
        if roi is None:
            return None

        self._attach_roi_event_listeners(roi, add_to_plot=True)

        if self._user_rois_manager is not None:
            self._user_rois_manager.add_roi(roi, add_to_view=False)
            self._user_rois = self._user_rois_manager.rois
        else:
            if self._user_rois is None:
                self._user_rois = []
            self._user_rois.append(roi)

        self.UserEditableROIMixin_on_roi_added(roi)
        return roi

    def _remove_roi(self, roi: PGROI) -> None:
        if self._user_rois_manager is not None:
            managed_list = self._user_rois_manager.rois
        else:
            managed_list = self._user_rois or []

        if roi not in managed_list:
            return

        self._detach_roi_event_listeners(roi)
        if self._user_roi_parent_plot is not None:
            try:
                self._user_roi_parent_plot.removeItem(roi)
            except Exception:
                pass

        if self._user_rois_manager is not None:
            self._user_rois_manager.remove_roi(roi, remove_from_view=False)
        else:
            managed_list.remove(roi)

        self.UserEditableROIMixin_on_roi_removed(roi)

    def _remove_all_user_rois(self) -> None:
        managed_rois = self.user_defined_rois
        if not managed_rois:
            return
        for roi in managed_rois:
            self._remove_roi(roi)

    def _on_roi_region_updated(self, roi: PGROI) -> None:
        self.UserEditableROIMixin_on_roi_updated(roi)

    def _attach_roi_event_listeners(self, roi: PGROI, add_to_plot: bool):
        if roi is None:
            return
        try:
            roi.sigRegionChangeFinished.disconnect(self._on_roi_region_updated)
        except (TypeError, RuntimeError):
            pass
        roi.sigRegionChangeFinished.connect(self._on_roi_region_updated)

        try:
            roi.sigRemoveRequested.disconnect(self._remove_roi)
        except (TypeError, RuntimeError):
            pass
        roi.sigRemoveRequested.connect(self._remove_roi)

        if add_to_plot and self._user_roi_parent_plot is not None and roi.scene() is None:
            self._user_roi_parent_plot.addItem(roi)

    def _detach_roi_event_listeners(self, roi: PGROI):
        if roi is None:
            return
        try:
            roi.sigRegionChangeFinished.disconnect(self._on_roi_region_updated)
        except (TypeError, RuntimeError):
            pass
        try:
            roi.sigRemoveRequested.disconnect(self._remove_roi)
        except (TypeError, RuntimeError):
            pass

    def _build_default_roi_factories(self) -> Dict[str, ROIConstructor]:
        """Return a set of reasonable default ROI constructors centered on the cursor."""
        base_pen = pg.mkPen({"color": (255, 255, 0, 190), "width": 2})
        rect_size = QtCore.QSizeF(30.0, 18.0)
        line_half_length = 15.0

        def _line_roi(center: QtCore.QPointF) -> PGROI:
            cx = float(center.x())
            cy = float(center.y())
            return pg.LineROI(
                [cx - line_half_length, cy],
                [cx + line_half_length, cy],
                width=3,
                pen=base_pen,
                hoverPen=(255, 255, 0, 255),
            )

        def _rect_roi(center: QtCore.QPointF) -> PGROI:
            cx = float(center.x())
            cy = float(center.y())
            size = QtCore.QSizeF(rect_size.width(), rect_size.height())
            pos = QtCore.QPointF(cx - size.width() / 2.0, cy - size.height() / 2.0)
            return pg.RectROI(
                [pos.x(), pos.y()],
                [size.width(), size.height()],
                pen=base_pen,
                hoverPen=(255, 255, 0, 255),
            )

        def _ellipse_roi(center: QtCore.QPointF) -> PGROI:
            cx = float(center.x())
            cy = float(center.y())
            size = QtCore.QSizeF(rect_size.width(), rect_size.height())
            pos = QtCore.QPointF(cx - size.width() / 2.0, cy - size.height() / 2.0)
            return pg.EllipseROI(
                [pos.x(), pos.y()],
                [size.width(), size.height()],
                pen=base_pen,
                hoverPen=(255, 255, 0, 255),
            )

        return {
            "Line": _line_roi,
            "Rectangle": _rect_roi,
            "Ellipse": _ellipse_roi,
        }

    def _find_roi_at_scene_pos(self, scene_pos: QtCore.QPointF) -> Optional[PGROI]:
        if self._user_roi_scene is None or not self._user_rois:
            return None

        qt_point = QtCore.QPointF(float(scene_pos.x()), float(scene_pos.y()))
        for item in self._user_roi_scene.items(qt_point):
            roi = self._normalize_item_to_roi(item)
            if roi is not None and roi in self._user_rois:
                return roi
        return None

    @staticmethod
    def _normalize_item_to_roi(item) -> Optional[PGROI]:
        if item is None:
            return None
        if isinstance(item, PGROI):
            return item
        if isinstance(item, PGROIHandle):
            return item.rois[0] if item.rois else None

        parent_item = item.parentItem() if hasattr(item, "parentItem") else None
        while parent_item is not None:
            if isinstance(parent_item, PGROI):
                return parent_item
            parent_item = parent_item.parentItem()
        return None

    def _is_scene_pos_on_parent_plot(self, scene_pos: QtCore.QPointF) -> bool:
        if self._user_roi_parent_plot is None:
            return False
        bounding_rect = self._user_roi_parent_plot.sceneBoundingRect()
        return bounding_rect.contains(QtCore.QPointF(float(scene_pos.x()), float(scene_pos.y())))

    def _clear_active_roi_menu(self) -> None:
        if self._user_roi_menu is not None:
            try:
                self._user_roi_menu.deleteLater()
            except Exception:
                pass
        self._user_roi_menu = None

    @staticmethod
    def _qt_point_from_point(point: QtCore.QPointF) -> QtCore.QPoint:
        return QtCore.QPoint(int(point.x()), int(point.y()))


