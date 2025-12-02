"""
Example demonstrating DockPlanningHelperWindow - a window for interactively 
planning and managing dock layouts with multiple DockPlanningHelperWidget instances.

This example shows how to use DockPlanningHelperWindow to create and manage
a dock area with multiple planning helper widgets that can be dynamically
created and configured.
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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.DockPlanningHelperWindow import DockPlanningHelperWindow

app = pg.mkQApp("DockPlanningHelperWindow Example")

if __name__ == '__main__':
    # Create a DockPlanningHelperWindow with 3 initial dock planning helper widgets
    window = DockPlanningHelperWindow.init_dock_area_builder(
        n_dock_planning_helper_widgets=3
    )
    
    # Create an additional dock widget programmatically
    a_dock_helper_widget, a_dock_config, a_dock_widget = window.perform_create_new_dock_widget(
        dock_id_str='custom_dock',
        active_dock_add_location='right',
        dockSize=(400, 500)
    )
    
    # Set custom properties on the new dock
    a_dock_helper_widget.title = 'Custom Added Dock'
    a_dock_helper_widget.identifier = 'custom_dock'
    
    print(f"Created DockPlanningHelperWindow with {len(window.dock_helper_widgets)} dock widgets")
    print(f"Dock IDs: {list(window.dock_helper_widgets.keys())}")
    
    # The window is automatically shown by init_dock_area_builder
    pg.exec()

