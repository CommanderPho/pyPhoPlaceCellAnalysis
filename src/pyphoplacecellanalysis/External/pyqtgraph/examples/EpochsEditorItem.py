"""
Example demonstrating EpochsEditor - a widget for editing epoch intervals 
using PyQtGraph with customizable linear region items.

This example shows how to use EpochsEditor to display and edit epoch intervals
(such as laps) with interactive drag-and-drop functionality, context menus,
and visual feedback.
"""

import sys
import os

# Add the parent directory to the path to allow importing from pyphoplacecellanalysis
# Path from: pyphoplacecellanalysis/External/pyqtgraph/examples/ -> pyphoplacecellanalysis/
examples_dir = os.path.dirname(os.path.abspath(__file__))
pypho_root = os.path.abspath(os.path.join(examples_dir, '../../..'))
if pypho_root not in sys.path:
    sys.path.insert(0, pypho_root)

import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import EpochsEditor

app = pg.mkQApp("EpochsEditor Example")

if __name__ == '__main__':
    # Create example position data
    pos_data = {
        't': np.linspace(0, 100, 500),
        'x_smooth': np.sin(np.linspace(0, 10, 500)) * 50 + 50,  # Position oscillating around 50
        'velocity_x_smooth': np.cos(np.linspace(0, 10, 500)) * 5,
        'acceleration_x_smooth': np.sin(np.linspace(0, 10, 500)) * np.cos(np.linspace(0, 10, 500)) * 2
    }
    pos_df = pd.DataFrame(pos_data)

    # Create example laps/epochs data
    laps_data = {
        'start': [10, 30, 50, 70, 90],
        'stop': [20, 40, 60, 80, 100],
        'lap_id': [1, 2, 3, 4, 5],
        'label': ['Lap 1', 'Lap 2', 'Lap 3', 'Lap 4', 'Lap 5'],
        'lap_dir': [1, -1, 1, -1, 1],
        'is_LR_dir': [True, False, True, False, True]
    }
    curr_laps_df = pd.DataFrame(laps_data)

    # Initialize the epochs editor
    epochs_editor = EpochsEditor.init_laps_diagnoser(
        pos_df, 
        curr_laps_df, 
        include_velocity=True, 
        include_accel=True
    )
    
    # Ensure the window is shown
    epochs_editor.plots.win.show()
    
    pg.exec()

