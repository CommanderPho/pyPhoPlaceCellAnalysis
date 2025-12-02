"""
Example demonstrating ParameterTreeWidget - a widget for displaying and editing
parameter trees with save/restore functionality.

This example shows how to use ParameterTreeWidget to create an interactive
parameter tree interface with various parameter types.
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
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree

app = pg.mkQApp("ParameterTreeWidget Example")

if __name__ == '__main__':
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ParameterTreeWidget import create_parameter_tree_widget
    
    # Create example parameters
    params = [
        {'name': 'Basic Parameters', 'type': 'group', 'children': [
            {'name': 'Integer', 'type': 'int', 'value': 10, 'limits': (0, 100)},
            {'name': 'Float', 'type': 'float', 'value': 3.14159, 'step': 0.01, 'siPrefix': True},
            {'name': 'String', 'type': 'str', 'value': 'Hello World'},
            {'name': 'Boolean', 'type': 'bool', 'value': True},
        ]},
        {'name': 'List Parameters', 'type': 'group', 'children': [
            {'name': 'Choice', 'type': 'list', 'values': ['Option A', 'Option B', 'Option C'], 'value': 'Option A'},
            {'name': 'Color', 'type': 'color', 'value': 'FF0000'},
        ]},
        {'name': 'Advanced', 'type': 'group', 'children': [
            {'name': 'Angle', 'type': 'float', 'value': 45, 'suffix': 'deg', 'siPrefix': False},
            {'name': 'Range', 'type': 'float', 'value': 0.5, 'limits': (0, 1), 'step': 0.1},
        ]},
    ]
    
    # Create the Parameter tree
    p = Parameter.create(name='params', type='group', children=params)
    
    # Connect to parameter changes
    def on_parameter_change(param, changes):
        print("Parameter changed:")
        for param, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print(f"  {childName}: {change} -> {data}")
    
    p.sigTreeStateChanged.connect(on_parameter_change)
    
    # Create the widget
    win, param_tree = create_parameter_tree_widget(parameters=p, debug_print=False)
    
    pg.exec()

