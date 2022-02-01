"""
Simple use of DataTreeWidget to display a structure of nested dicts, lists, and arrays

Example:
    d = {
        'active_sess_config':curr_active_pipeline.active_sess_config.__dict__,
        'active_configs':curr_active_pipeline.active_configs,
        'active_session_computation_configs':active_session_computation_configs[0].__dict__
    }
    # d = {
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }
    # d = {
    #     'active_session_computation_configs':active_session_computation_configs,
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }

    from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_DataTreeWidget import plot_dataTreeWidget
    tree, app = plot_dataTreeWidget(data=d, title='PhoOutputDataTreeApp')

"""

# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


def plot_dataTreeWidget(data, title='PhoOutputDataTreeApp'):
    app = pg.mkQApp(title)
    # d = {
    # 	'a list': [1,2,3,4,5,6, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
    # 	'a dict': {
    # 		'x': 1,
    # 		'y': 2,
    # 		'z': 'three'
    # 	},
    # 	'an array': np.random.randint(10, size=(40,10)),
    # 	'a traceback': some_func1(),
    # 	'a function': some_func1,
    # 	'a class': pg.DataTreeWidget,
    # }
    tree = pg.DataTreeWidget(data=data)
    tree.show()
    tree.setWindowTitle(f'PhoOutputDataTreeApp: pyqtgraph DataTreeWidget: {title}')
    tree.resize(800,600)
    return tree, app



if __name__ == '__main__':
    pg.exec()
