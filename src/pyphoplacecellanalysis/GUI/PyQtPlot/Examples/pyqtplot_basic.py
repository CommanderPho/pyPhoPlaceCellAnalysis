# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance

import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
# from dataclasses import dataclass

# @dataclass
# class BasicPyQtPlotApp(object):
#     """Docstring for BasicPyQtPlotApp."""
#     app: QtGui.QApplication
#     win: QtGui.QMainWindow
#     # w: pg.GraphicsLayoutWidget


def pyqtplot_common_setup(a_title, app=None, parent_root_widget=None, root_render_widget=None):
    """[summary]

    Args:
        a_title ([type]): [description]
        app ([type], optional): [description]. Defaults to None.
        parent_root_widget ([type], optional): the parent root widget that contains the root_render_widget. If None, will create a new QMainWindow and add root_render_widget to that. Defaults to None.
        root_render_widget ([type], optional): The actual widget to render in, a child of parent_root_widget. If None, a new GraphicsLayoutWidget will be created and added to parent_root_widget. Defaults to None.

    Returns:
        [type]: [description]
    """
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    pg.setConfigOptions(antialias = True)
    
    if app is None:
        app = pg.mkQApp(a_title)
    # print(f'type(app): {type(app)}')
    
    if root_render_widget is not None:
        # already have a valid root_render_widget, so we don't care about parent_root_widget either way.
        pass
    else:        
        # Create window to hold the image:
        if parent_root_widget is None:
            parent_root_widget = QtGui.QMainWindow()
            parent_root_widget.resize(1600, 1600)
        # Creating a GraphicsLayoutWidget as the central widget
        if root_render_widget is None:
            root_render_widget = pg.GraphicsLayoutWidget()
            parent_root_widget.setCentralWidget(root_render_widget)
    
    return root_render_widget, parent_root_widget, app