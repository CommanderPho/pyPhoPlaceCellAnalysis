import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF, QTimer, QEvent
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock


class TrapezoidOverlay(QWidget):
    """ a callout that illustrates a specific dock item is a subset of the above (parent) dock item

    from pyphoplacecellanalysis.External.pyqtgraph_extensions.trapezoid_callout import TrapezoidOverlay, SpacerDock

    """
    def __init__(self, parent=None, overview_widget=None, overview_zoomed_region_item=None, zoomed_widget=None):
        super().__init__(parent)
        # Make this widget transparent to mouse clicks and background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.source_plot = overview_widget  # The top PlotWidget
        self.region = overview_zoomed_region_item       # The LinearRegionItem (selection box)
        self.target_plot = zoomed_widget  # The bottom PlotWidget (zoomed view)
        
        # Color configuration
        self.fill_color = QColor(0, 255, 255, 30)  # Cyan with low alpha
        self.border_color = QColor(0, 255, 255, 80)
        
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
            # Trigger initial paint
            QTimer.singleShot(0, self.update)


    def paintEvent(self, event):
        if not (self.source_plot and self.region and self.target_plot):
            return

        should_inset_for_visibility: bool = True
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if should_inset_for_visibility:

            # --- 1. Get Source Coordinates (Overview Plot with Region) ---
            vb_source = self.source_plot.plotItem.vb

            # Get the bounding rect of the ViewBox (the grid area) in scene coordinates
            source_view_rect = vb_source.mapRectToScene(vb_source.boundingRect())

            # Map the "Region" X values to Scene X coordinates
            x_min, x_max = self.region.getRegion()
            p1_x = vb_source.mapViewToScene(QPointF(x_min, 0)).x()
            p2_x = vb_source.mapViewToScene(QPointF(x_max, 0)).x()

            # --- 2. Get Target Coordinates (Zoomed Plot) ---
            vb_target = self.target_plot.plotItem.vb
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
            top_left_global = self.source_plot.mapToGlobal(self.source_plot.mapFromScene(QPointF(p1_x, source_y_anchor)))
            top_right_global = self.source_plot.mapToGlobal(self.source_plot.mapFromScene(QPointF(p2_x, source_y_anchor)))

            # Use the ViewBox width for the corners
            bottom_left_global = self.target_plot.mapToGlobal(self.target_plot.mapFromScene(target_anchor_point_left))
            bottom_right_global = self.target_plot.mapToGlobal(self.target_plot.mapFromScene(target_anchor_point_right))


        else:
            # --- 1. Get Coordinates of the Top Selection (Source) ---
            # Get the current X values of the region [min, max]
            min_x, max_x = self.region.getRegion()
            
            # Map these data values to the ViewBox's internal coordinate system
            vb_source = self.source_plot.plotItem.vb
            
            # Map data x to view pixel coordinates (0 to width of view)
            p1_view = vb_source.mapViewToDevice(QPointF(min_x, 0))
            p2_view = vb_source.mapViewToDevice(QPointF(max_x, 0))
            
            # We need the X position relative to the Source Widget
            # Note: We take the Bottom Y of the source widget for the anchor
            source_y_bottom = self.source_plot.height()
            
            # Convert to Global Screen coordinates, then back to this Overlay's local coordinates
            top_left_global = self.source_plot.mapToGlobal(QPointF(p1_view.x(), source_y_bottom).toPoint())
            top_right_global = self.source_plot.mapToGlobal(QPointF(p2_view.x(), source_y_bottom).toPoint())

            # --- 2. Get Coordinates of the Bottom Widget (Target) ---
            # We want the top-left and top-right corners of the bottom plot
            target_rect = self.target_plot.rect()
            
            bottom_left_global = self.target_plot.mapToGlobal(target_rect.topLeft())
            bottom_right_global = self.target_plot.mapToGlobal(target_rect.topRight())
            

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

        painter.setBrush(QBrush(self.fill_color))
        painter.setPen(QPen(self.border_color, 1))
        painter.drawPolygon(polygon)

    def eventFilter(self, obj, event):
        """Event filter to catch resize events from the parent widget."""
        if obj == self.parent() and event.type() == QEvent.Type.Resize:
            # Resize the overlay to match the parent widget size
            self.resize(self.parent().size())
            # Schedule an update to redraw the overlay
            QTimer.singleShot(0, self.update)
        return super().eventFilter(obj, event)


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