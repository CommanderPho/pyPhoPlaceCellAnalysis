"""
Example demonstrating IntervalRectsItem - a custom graphics item for rendering 
rectangular intervals with customizable tooltips and context menus.

This example shows how to use IntervalRectsItem to display rectangular intervals
with hover tooltips, custom context menus, and legend support.
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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import main2

app = pg.mkQApp("IntervalRectsItem Example")

if __name__ == '__main__':
    plt, item, legend = main2()
    pg.exec()

