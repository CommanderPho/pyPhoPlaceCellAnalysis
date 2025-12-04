import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF, QTimer, QEvent
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


@metadata_attributes(short_name=None, tags=['painter', 'renderer', 'trapazoid', 'region', 'timeline', 'track'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-03 15:41', related_items=[])
class TrapezoidOverlay(QWidget):
    """ a callout that illustrates a specific dock item is a subset of the above (parent) dock item

    from pyphoplacecellanalysis.External.pyqtgraph_extensions.trapezoid_callout import TrapezoidOverlay, SpacerDock


        ## Customization
        self.brush
        self.pen
        self.top_bottom_edges_pen

    """


    def __init__(self, parent=None, overview_widget=None, overview_zoomed_region_item=None, zoomed_widget=None):
        super().__init__(parent)
        # Make this widget transparent to mouse clicks and background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Ensure widget is visible and can receive paint events
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, False)  # Use normal painting
        
        self.source_plot = overview_widget  # The top PlotWidget
        self.region = overview_zoomed_region_item       # The LinearRegionItem (selection box)
        self.target_plot = zoomed_widget  # The bottom PlotWidget (zoomed view)
        

        # Styles: DashLine, DotLine, DashDotLine, DashDotDotLine
        # dashed_pen_style = pg.Qt.PenStyle.DashLine

        # Color configuration
        if (self.region is not None) and (self.region.brush is not None):
            ## copy from parent so that it's the a matching color. Ideally we could reduce the opacity tho.
            self.brush = pg.mkBrush(self.region.brush)
            # border_color = self.brush.color()
            border_color = QColor(250, 250, 250, 250) ## light gray
            self.pen = pg.mkPen(border_color, width=1)

        else:
            fill_color = QColor(0, 255, 255, 30)  # Cyan with low alpha
            border_color = QColor(0, 255, 255, 80)

            self.brush = QBrush(fill_color)
            self.pen = QPen(border_color, 1)

        # Set pattern: [pixels_on, pixels_off, pixels_on, pixels_off, ...]
        # Example: 10px line, 5px space
        self.pen.setDashPattern([2, 2])
        
        # Pen for highlighting top and bottom edges
        # Use a solid, brighter pen for the edges
        top_bottom_edge_color = QColor(255, 255, 255, 0)  # White, clear
        self.top_bottom_edges_pen = QPen(top_bottom_edge_color, 2)  # Thicker, solid line
        
        # Or if you want to match the region color but brighter:
        # if (self.region is not None) and (self.region.brush is not None):
        #     edge_color = self.brush.color()
        #     edge_color.setAlpha(255)  # Fully opaque
        #     self.top_bottom_edges_pen = QPen(edge_color, 2)
        # else:
        #     self.top_bottom_edges_pen = QPen(QColor(255, 255, 255, 255), 2)

        
        # Hook into parent's resize event to redraw the overlay when the parent/containing widget size changes
        if parent is not None:
            parent.installEventFilter(self)
            # Resize overlay to match parent initially
            self.resize(parent.size())
            # Position at top-left corner of parent
            self.move(0, 0)
            # Make sure the overlay is visible and on top
            self.show()
            self.raise_()
            # Ensure parent is shown first
            if parent.isVisible():
                self.setParent(parent)  # Re-parent to ensure proper stacking
            # Trigger initial paint with delay to ensure layout is complete
            QTimer.singleShot(100, self.update)
        
        # Connect region signals to force updates when region changes
        if self.region is not None:
            def force_overlay_update():
                self.update()
            
            self.region.sigRegionChanged.connect(force_overlay_update)
            self.region.sigRegionChangeFinished.connect(force_overlay_update)






    def _get_plot_item(self, widget):
        """Helper method to get PlotItem from either PlotWidget or PyqtgraphTimeSynchronizedWidget."""
        # Check if it's a PyqtgraphTimeSynchronizedWidget (or similar wrapper)
        if hasattr(widget, 'getRootPlotItem'):
            return widget.getRootPlotItem()
        # Otherwise assume it's a standard PlotWidget
        elif hasattr(widget, 'plotItem'):
            return widget.plotItem
        else:
            raise AttributeError(f"Widget {type(widget)} does not have plotItem or getRootPlotItem method")
    

    def _get_scene_mapping_widget(self, widget):
        """Helper method to get the widget that supports mapFromScene.
        For PyqtgraphTimeSynchronizedWidget, returns the GraphicsLayoutWidget.
        For standard PlotWidget, returns the widget itself."""
        # Check if it's a PyqtgraphTimeSynchronizedWidget (or similar wrapper)
        if hasattr(widget, 'getRootGraphicsLayoutWidget'):
            return widget.getRootGraphicsLayoutWidget()
        # Otherwise assume it's a standard PlotWidget that supports mapFromScene
        elif hasattr(widget, 'mapFromScene'):
            return widget
        else:
            # Fallback: try to find a GraphicsLayoutWidget child
            for child in widget.findChildren(type(widget).__class__):
                if hasattr(child, 'mapFromScene'):
                    return child
            raise AttributeError(f"Widget {type(widget)} does not have mapFromScene or getRootGraphicsLayoutWidget method")


    def paintEvent(self, event):
        # Add comprehensive validation checks
        if not (self.source_plot and self.region and self.target_plot):
            return
        
        # Check if widgets are still valid (not deleted)
        try:
            if not (self.source_plot.isVisible() and self.target_plot.isVisible()):
                return
            
            # Validate that widgets have valid geometry
            if self.source_plot.width() <= 0 or self.source_plot.height() <= 0:
                return
            if self.target_plot.width() <= 0 or self.target_plot.height() <= 0:
                return
        except RuntimeError:
            # Widget has been deleted
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            # --- 1. Get Source Coordinates (Overview Plot with Region) ---
            plot_item_source = self._get_plot_item(self.source_plot)
            vb_source = plot_item_source.vb
            scene_mapping_widget_source = self._get_scene_mapping_widget(self.source_plot)

            # Get the bounding rect of the ViewBox (the grid area) in scene coordinates
            source_view_rect = vb_source.mapRectToScene(vb_source.boundingRect())

            # Map the "Region" X values to Scene X coordinates
            x_min, x_max = self.region.getRegion()
            p1_x = vb_source.mapViewToScene(QPointF(x_min, 0)).x()
            p2_x = vb_source.mapViewToScene(QPointF(x_max, 0)).x()

            # --- 2. Get Target Coordinates (Zoomed Plot) ---
            plot_item_target = self._get_plot_item(self.target_plot)
            vb_target = plot_item_target.vb
            scene_mapping_widget_target = self._get_scene_mapping_widget(self.target_plot)
            target_view_rect = vb_target.mapRectToScene(vb_target.boundingRect())

            # --- 3. Determine relative position of target vs source ---
            # Compare global Y positions to determine if target is above or below source
            source_center_global = self.source_plot.mapToGlobal(self.source_plot.rect().center())
            target_center_global = self.target_plot.mapToGlobal(self.target_plot.rect().center())
            target_is_above = target_center_global.y() < source_center_global.y()

            if target_is_above:
                # Target (zoomed) is ABOVE source (overview)
                source_y_anchor = source_view_rect.top()  # Source anchor: top of ViewBox
                target_anchor_point_left = target_view_rect.bottomLeft()  # Target anchor: bottom of ViewBox
                target_anchor_point_right = target_view_rect.bottomRight()
            else:
                # Target (zoomed) is BELOW source (overview) - original behavior
                source_y_anchor = source_view_rect.bottom()  # Source anchor: bottom of ViewBox
                target_anchor_point_left = target_view_rect.topLeft()  # Target anchor: top of ViewBox
                target_anchor_point_right = target_view_rect.topRight()

            # Convert to Global Screen coordinates
            # Map scene coordinates to widget-local coordinates, then to global
            top_left_local = scene_mapping_widget_source.mapFromScene(QPointF(p1_x, source_y_anchor))
            top_left_global = scene_mapping_widget_source.mapToGlobal(top_left_local)
            
            top_right_local = scene_mapping_widget_source.mapFromScene(QPointF(p2_x, source_y_anchor))
            top_right_global = scene_mapping_widget_source.mapToGlobal(top_right_local)

            # Use the ViewBox width for the corners
            bottom_left_local = scene_mapping_widget_target.mapFromScene(target_anchor_point_left)
            bottom_left_global = scene_mapping_widget_target.mapToGlobal(bottom_left_local)
            
            bottom_right_local = scene_mapping_widget_target.mapFromScene(target_anchor_point_right)
            bottom_right_global = scene_mapping_widget_target.mapToGlobal(bottom_right_local)


        except (RuntimeError, AttributeError, ValueError) as e:
            # Widgets may be in invalid state during resize - log for debugging
            import traceback
            print(f"TrapezoidOverlay.paintEvent error: {e}")
            print(traceback.format_exc())
            return

        # Common Drawing code
        top_left = self.mapFromGlobal(top_left_global)
        top_right = self.mapFromGlobal(top_right_global)

        bottom_left = self.mapFromGlobal(bottom_left_global)
        bottom_right = self.mapFromGlobal(bottom_right_global)

        # --- 3. Draw the Trapezoid ---
        polygon = QPolygonF()
        polygon.append(top_left)
        polygon.append(top_right)
        polygon.append(bottom_right)
        polygon.append(bottom_left)

        # Draw filled polygon without border
        painter.setBrush(self.brush)
        painter.setPen(Qt.PenStyle.NoPen)  # No border for the fill
        painter.drawPolygon(polygon)
        
        # Draw edges with appropriate pens
        # Top and bottom edges with highlighted pen
        painter.setPen(self.top_bottom_edges_pen)
        painter.drawLine(top_left, top_right)  # Top edge
        painter.drawLine(bottom_left, bottom_right)  # Bottom edge
        
        # Left and right edges with regular pen
        painter.setPen(self.pen)
        painter.drawLine(top_left, bottom_left)  # Left edge
        painter.drawLine(top_right, bottom_right)  # Right edge



    def eventFilter(self, obj, event):
        """Event filter to catch resize events from the parent widget."""
        if obj == self.parent() and event.type() == QEvent.Type.Resize:
            # Critical: Check if parent is still valid
            parent = self.parent()
            if parent is None:
                return super().eventFilter(obj, event)
            
            # Prevent infinite recursion by checking if we're already resizing
            # Use a flag or check if size actually changed
            new_size = parent.size()
            if new_size != self.size():
                # Use QTimer to defer resize to avoid recursion
                QTimer.singleShot(0, lambda: self._safe_resize(new_size))
        return super().eventFilter(obj, event)
    

    def _safe_resize(self, new_size):
        """Safely resize the overlay with validation."""
        try:
            if self.parent() is not None:
                self.resize(new_size)
                QTimer.singleShot(0, self.update)
        except RuntimeError:
            # Parent may have been deleted
            pass


    @function_attributes(short_name=None, tags=['timeline', 'tracks', 'SpikeRaster2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-03 15:41', related_items=[])
    @classmethod
    def add_overview_indicator_trapazoids_to_timeline(cls, active_2d_plot):
        """
        from pyphoplacecellanalysis.External.pyqtgraph_extensions.trapezoid_callout import TrapezoidOverlay, SpacerDock
        
        _out_overlays: Dict[Tuple, TrapezoidOverlay] = TrapezoidOverlay.add_overview_indicator_trapazoids_to_timeline(active_2d_plot=active_2d_plot)
        
        """

        flat_dock_item_tuples_dict = active_2d_plot.get_flat_dock_item_tuple_dict()

        overview_child_dock_identifer_pair_list = [('rasters[raster_overview]', 'rasters[raster_window]'),
                                                ('interval_overview', 'intervals'),
                                                ]

        _out_overlays = {}
        for (an_overview_id, a_zoomed_id) in overview_child_dock_identifer_pair_list:
            ov_d, ov_widget = flat_dock_item_tuples_dict[an_overview_id]
            zoomed_d, zoomed_widget = flat_dock_item_tuples_dict[a_zoomed_id]
            ## timeline common scroll item:
            overview_zoomed_region_item = active_2d_plot.ui.scroll_window_region
            display_dock_area = active_2d_plot.dock_manager_widget.displayDockArea
            
            # print_keys_if_possible('ov_widget', ov_widget, max_depth=2)
            # print_keys_if_possible('zoomed_widget', zoomed_widget, max_depth=2)
            # We must parent the overlay to the central widget (DockArea) 
            # so it covers all docks.
            overlay = TrapezoidOverlay(
                parent=display_dock_area, ## this will always be this value 
                overview_widget=ov_widget, 
                # overview_zoomed_region_item=ov_widget.ui.region, 
                overview_zoomed_region_item=overview_zoomed_region_item,
                zoomed_widget=zoomed_widget,
            )
            _out_overlays[(an_overview_id, a_zoomed_id)] = overlay

        # END for (an....    
        return _out_overlays


class SpacerDock(Dock):
    """
    A custom Dock designed to act as a fixed-height visual separator.
    It has no title bar and contains a transparent (or colored) widget.
    """
    def __init__(self, height=50, name="Spacer", color=None):
        """
        Args:
            height (int): The vertical size of the gap in pixels.
            name (str): Unique name for the dock (required by pg.Dock).
            color (str, optional): Hex code (e.g., "#000") if you want a visible background.
                                   Defaults to None (transparent/theme background).
        """
        # Initialize the Dock with specific flags to strip functionality
        # size=(1, height) sets the initial relative size hint
        super().__init__(name, size=(1, height), autoOrientation=False)

        # 1. Hide the Title Bar (removes the name and the drag handle)
        self.hideTitleBar()

        # 2. Create the internal "Shim" widget
        self.shim = QWidget()
        
        # 3. Enforce the Spacer Height
        # Setting fixed height on the internal widget forces the Dock 
        # to respect this size, preventing it from collapsing or expanding.
        self.shim.setFixedHeight(height)

        # 4. Optional Styling
        if color:
            self.shim.setStyleSheet(f"background-color: {color}; border: none;")
        else:
            # Transparent/No-border style
            self.shim.setStyleSheet("background-color: transparent; border: none;")

        # 5. Add the shim to the Dock
        self.addWidget(self.shim)

    def set_height(self, height):
        """Dynamically change the spacer height."""
        self.shim.setFixedHeight(height)
        


class TrapezoidTestingMainWindow(QMainWindow):
    """ just a concrete test window to display the effect of the trapezoid stuff
    """
    def __init__(self, use_SpacerDock_approach: bool=False, zoomed_above: bool=False):
        super().__init__()
        self.use_SpacerDock_approach = use_SpacerDock_approach
        self.zoomed_above = zoomed_above
        
        self.resize(1000, 600)
        self.setStyleSheet("background-color: #222;")

        # Setup DockArea
        self.area = DockArea()
        self.setCentralWidget(self.area)

        if zoomed_above:
            # --- Zoomed View on Top ---
            self.d2 = Dock("Zoomed View", size=(1000, 400))
            self.area.addDock(self.d2, 'top')
            self.w2_zoomed = pg.PlotWidget(title="Detailed View")
            self.w2_zoomed.plot([2, 4, 3, 6, 2], pen='y')  # Dummy zoomed data
            self.d2.addWidget(self.w2_zoomed)

            if use_SpacerDock_approach:
                spacer_dock = SpacerDock(height=60, name="my_spacer")
                self.area.addDock(spacer_dock, 'bottom', self.d2)

            # --- Overview on Bottom ---
            self.d1 = Dock("Overview", size=(1000, 200))
            self.area.addDock(self.d1, 'bottom')
            self.w1_overview = pg.PlotWidget(title="Overview")
            self.w1_overview.plot([1, 5, 2, 4, 3, 6, 2, 5, 8, 3, 1], pen='w')
            self.d1.addWidget(self.w1_overview)
        else:
            # --- Overview on Top (original behavior) ---
            self.d1 = Dock("Overview", size=(1000, 200))
            self.area.addDock(self.d1, 'top')
            self.w1_overview = pg.PlotWidget(title="Overview")
            self.w1_overview.plot([1, 5, 2, 4, 3, 6, 2, 5, 8, 3, 1], pen='w')
            self.d1.addWidget(self.w1_overview)
            
            if use_SpacerDock_approach:
                spacer_dock = SpacerDock(height=60, name="my_spacer")
                self.area.addDock(spacer_dock, 'bottom', self.d1)

            # --- Zoomed View on Bottom ---
            self.d2 = Dock("Zoomed View", size=(1000, 400))
            self.area.addDock(self.d2, 'bottom')
            self.w2_zoomed = pg.PlotWidget(title="Detailed View")
            self.w2_zoomed.plot([2, 4, 3, 6, 2], pen='y')  # Dummy zoomed data
            self.d2.addWidget(self.w2_zoomed)

        # Add the Region Selection (The "Window") to overview plot
        self.region = pg.LinearRegionItem([2, 4])
        self.region.setBrush(QColor(0, 255, 255, 50)) 
        self.w1_overview.addItem(self.region)

        # if not use_SpacerDock_approach:     
        # --- The Overlay Logic ---
        # We must parent the overlay to the central widget (DockArea) 
        # so it covers all docks.
        self.overlay = TrapezoidOverlay(
            parent=self.area, 
            overview_widget=self.w1_overview, 
            overview_zoomed_region_item=self.region, 
            zoomed_widget=self.w2_zoomed
        )


        overlay = self.overlay
        # display_dock_area = self.region
        display_dock_area = self.area
        
        def regionResizeEvent(event):
            """ captures: display_dock_area, overlay
            """
            # Resize the overlay to match the window size
            print(f'debug: resizeEvent(event): event: {event}')
            print(f'\tdisplay_dock_area.size(): {display_dock_area.size()}')
            # if not self.use_SpacerDock_approach:
            overlay.resize(display_dock_area.size())
            # Schedule an update to repaint the overlay with new region coordinates
            QTimer.singleShot(0, overlay.update)


        # Connect signals to trigger repaints
        # # Use deferred update to ensure scene transforms are fully updated before redrawing
        # self.region.sigRegionChanged.connect(self._schedule_overlay_update)
        # # Important: Repaint when the user manually resizes the docks/window. We use a timer to hook into the resize event of the main loop easily, or you can override resizeEvent
        # self.region.sigRegionChangeFinished.connect(self.update_zoom)

        ## connect local function
        # Note: The TrapezoidOverlay now automatically hooks into parent's resize event via eventFilter
        self.region.sigRegionChanged.connect(regionResizeEvent)
        self.region.sigRegionChangeFinished.connect(regionResizeEvent)

        # After creating the overlay, ensure it's properly shown and raised
        overlay.show()
        overlay.raise_()
        # Force an update after a short delay to ensure everything is laid out
        QTimer.singleShot(200, overlay.update)

        # Also connect region signals to force updates
        def force_overlay_update():
            overlay.update()
            
        self.region.sigRegionChanged.connect(force_overlay_update)
        self.region.sigRegionChangeFinished.connect(force_overlay_update)



    def resizeEvent(self, event):
        # Resize the overlay to match the window size
        # if not self.use_SpacerDock_approach:
        self.overlay.resize(self.area.size())
        self._schedule_overlay_update()
            
        super().resizeEvent(event)


    def _schedule_overlay_update(self):
        """Defer overlay update to next event loop iteration to ensure scene transforms are current."""
        QTimer.singleShot(0, self.overlay.update)


    def update_zoom(self):
        # Your logic to update the data in the bottom plot
        # ...
        # if not self.use_SpacerDock_approach:
        self._schedule_overlay_update()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrapezoidTestingMainWindow()
    win.show()
    sys.exit(app.exec())