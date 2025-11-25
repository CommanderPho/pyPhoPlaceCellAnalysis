import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock

class TrapezoidOverlay(QWidget):
    """ a callout that illustrates a specific dock item is a subset of the above (parent) dock item

    from pyphoplacecellanalysis.External.pyqtgraph_extensions.trapezoid_callout import TrapezoidOverlay

    """
    def __init__(self, parent=None, source_plot=None, region_item=None, target_plot=None):
        super().__init__(parent)
        # Make this widget transparent to mouse clicks and background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.source_plot = source_plot  # The top PlotWidget
        self.region = region_item       # The LinearRegionItem (selection box)
        self.target_plot = target_plot  # The bottom PlotWidget (zoomed view)
        
        # Color configuration
        self.fill_color = QColor(0, 255, 255, 30)  # Cyan with low alpha
        self.border_color = QColor(0, 255, 255, 80)

    def paintEvent(self, event):
        if not (self.source_plot and self.region and self.target_plot):
            return

        should_inset_for_visibility: bool = True
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if should_inset_for_visibility:

            # --- 1. Get Source Coordinates (Top Plot) ---
            vb_source = self.source_plot.plotItem.vb

            # Get the bounding rect of the ViewBox (the grid area) in scene coordinates
            source_view_rect = vb_source.mapRectToScene(vb_source.boundingRect())

            # Map the "Region" X values to Scene X coordinates
            x_min, x_max = self.region.getRegion()
            p1_x = vb_source.mapViewToScene(QPointF(x_min, 0)).x()
            p2_x = vb_source.mapViewToScene(QPointF(x_max, 0)).x()

            # USE THIS Y: The bottom of the ViewBox (grid), not the Widget
            source_y_anchor = source_view_rect.bottom() 

            # Convert to Global Screen coordinates
            top_left_global = self.source_plot.mapToGlobal(self.source_plot.mapFromScene(QPointF(p1_x, source_y_anchor)))
            top_right_global = self.source_plot.mapToGlobal(self.source_plot.mapFromScene(QPointF(p2_x, source_y_anchor)))


            # --- 2. Get Target Coordinates (Bottom Plot) ---
            vb_target = self.target_plot.plotItem.vb
            target_view_rect = vb_target.mapRectToScene(vb_target.boundingRect())

            # USE THIS Y: The top of the ViewBox (grid)
            target_y_anchor = target_view_rect.top()

            # Use the ViewBox width for the corners
            bottom_left_global = self.target_plot.mapToGlobal(self.target_plot.mapFromScene(target_view_rect.topLeft()))
            bottom_right_global = self.target_plot.mapToGlobal(self.target_plot.mapFromScene(target_view_rect.topRight()))


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




class TrapezoidTestingMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1000, 600)
        self.setStyleSheet("background-color: #222;")

        # Setup DockArea
        self.area = DockArea()
        self.setCentralWidget(self.area)

        # --- Top Dock (Overview) ---
        self.d1 = Dock("Overview", size=(1000, 200))
        self.area.addDock(self.d1, 'top')
        self.w1 = pg.PlotWidget(title="Overview")
        self.w1.plot([1, 5, 2, 4, 3, 6, 2, 5, 8, 3, 1], pen='w')
        self.d1.addWidget(self.w1)
        
        # Add the Region Selection (The "Window")
        self.region = pg.LinearRegionItem([2, 4])
        self.region.setBrush(QColor(0, 255, 255, 50)) 
        self.w1.addItem(self.region)

        # --- Bottom Dock (Zoomed) ---
        self.d2 = Dock("Zoomed View", size=(1000, 400))
        self.area.addDock(self.d2, 'bottom')
        self.w2 = pg.PlotWidget(title="Detailed View")
        self.w2.plot([2, 4, 3, 6, 2], pen='y') # Dummy zoomed data
        self.d2.addWidget(self.w2)

        # --- The Overlay Logic ---
        # We must parent the overlay to the central widget (DockArea) 
        # so it covers all docks.
        self.overlay = TrapezoidOverlay(
            parent=self.area, 
            source_plot=self.w1, 
            region_item=self.region, 
            target_plot=self.w2
        )

        # Connect signals to trigger repaints
        self.region.sigRegionChanged.connect(self.overlay.update)
        
        # Important: Repaint when the user manually resizes the docks/window
        # We use a timer to hook into the resize event of the main loop easily,
        # or you can override resizeEvent
        self.region.sigRegionChanged.connect(self.update_zoom)

    def resizeEvent(self, event):
        # Resize the overlay to match the window size
        self.overlay.resize(self.area.size())
        self.overlay.update()
        super().resizeEvent(event)

    def update_zoom(self):
        # Your logic to update the data in the bottom plot
        # ...
        self.overlay.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrapezoidTestingMainWindow()
    win.show()
    sys.exit(app.exec())