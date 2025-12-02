"""
Example demonstrating DockPlanningHelperWidget - a widget for planning and 
modifying dock properties interactively in a dock area layout.

This example shows how to use DockPlanningHelperWidget to create and configure
docks with custom titles, identifiers, colors, and orientations.
"""

import sys
import os

# Add the parent directory to the path to allow importing from pyphoplacecellanalysis
# Path from: pyphoplacecellanalysis/External/pyqtgraph/examples/ -> pyphoplacecellanalysis/
examples_dir = os.path.dirname(os.path.abspath(__file__))
pypho_root = os.path.abspath(os.path.join(examples_dir, '../../..'))
if pypho_root not in sys.path:
    sys.path.insert(0, pypho_root)

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget
from pyphoplacecellanalysis.External.pyqtgraph.dockarea import DockArea, Dock

app = pg.mkQApp("DockPlanningHelperWidget Example")

if __name__ == '__main__':
    # Create a dock area window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle('DockPlanningHelperWidget Example')
    win.resize(1000, 600)
    
    # Create the dock area
    dock_area = DockArea()
    win.setCentralWidget(dock_area)
    
    # Create the first dock with a planning helper widget
    dock1 = Dock("Dock 1", size=(400, 300))
    widget1 = DockPlanningHelperWidget(
        dock_title='My First Dock',
        dock_id='first_dock',
        color=QtGui.QColor('#FF6B6B'),
        defer_show=True
    )
    dock1.addWidget(widget1)
    dock_area.addDock(dock1, 'left')
    
    # Create a second dock
    dock2 = Dock("Dock 2", size=(400, 300))
    widget2 = DockPlanningHelperWidget(
        dock_title='My Second Dock',
        dock_id='second_dock',
        color=QtGui.QColor('#4ECDC4'),
        defer_show=True
    )
    dock2.addWidget(widget2)
    dock_area.addDock(dock2, 'right', dock1)
    
    # Create a third dock below
    dock3 = Dock("Dock 3", size=(800, 200))
    widget3 = DockPlanningHelperWidget(
        dock_title='Bottom Dock',
        dock_id='bottom_dock',
        color=QtGui.QColor('#95E1D3'),
        defer_show=True
    )
    dock3.addWidget(widget3)
    dock_area.addDock(dock3, 'bottom')
    
    # Connect signals to demonstrate functionality
    def on_dock_config_changed(widget):
        print(f"Dock config changed for: {widget.identifier}")
        config = widget.rebuild_config()
        print(f"  Config: {config}")
    
    def on_create_new_dock(widget, location):
        print(f"Create new dock request: {widget.identifier} -> {location}")
        # In a real application, you would create a new dock here
        new_dock = Dock(f"New Dock ({location})", size=(300, 200))
        new_widget = DockPlanningHelperWidget(
            dock_title=f'New Dock from {widget.identifier}',
            dock_id=f'new_dock_{len(dock_area.docks)}',
            color=QtGui.QColor('#FFE66D'),
            defer_show=True
        )
        new_dock.addWidget(new_widget)
        
        # Add to appropriate location
        if location == 'right':
            dock_area.addDock(new_dock, 'right', widget.embedding_dock_item)
        elif location == 'left':
            dock_area.addDock(new_dock, 'left', widget.embedding_dock_item)
        elif location == 'bottom':
            dock_area.addDock(new_dock, 'bottom', widget.embedding_dock_item)
        elif location == 'top':
            dock_area.addDock(new_dock, 'top', widget.embedding_dock_item)
    
    def on_save(widget_id, title, out_string):
        print(f"Save requested for dock '{title}' (id: {widget_id})")
        print(f"  Output: {out_string}")
    
    # Connect signals for all widgets
    for widget in [widget1, widget2, widget3]:
        widget.sigDockConfigChanged.connect(on_dock_config_changed)
        widget.sigCreateNewDock.connect(on_create_new_dock)
        widget.sigSave.connect(on_save)
    
    win.show()
    pg.exec()

