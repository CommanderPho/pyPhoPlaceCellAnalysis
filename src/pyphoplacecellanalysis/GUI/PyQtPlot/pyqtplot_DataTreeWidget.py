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
from collections import OrderedDict

__all__ = ['CustomFormattingDataTreeWidget']

class CustomFormattingDataTreeWidget(pg.DataTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """
    def __init__(self, parent=None, data=None):
        pg.DataTreeWidget.__init__(self, parent=parent, data=data)
        # super(CustomFormattingDataTreeWidget, self).__init__(self, parent, data)
        
        
    # def setData(self, data, hideRoot=False):
    #     """data should be a dictionary."""
    #     self.clear()
    #     self.widgets = []
    #     self.nodes = {}
    #     self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
    #     self.expandToDepth(3)
    #     self.resizeColumnToContents(0)
        
    # def buildTree(self, data, parent, name='', hideRoot=False, path=()):
    #     if hideRoot:
    #         node = parent
    #     else:
    #         node = QtWidgets.QTreeWidgetItem([name, "", ""])
    #         parent.addChild(node)
        
    #     # record the path to the node so it can be retrieved later
    #     # (this is used by DiffTreeWidget)
    #     self.nodes[path] = node

    #     typeStr, desc, childs, widget = self.parse(data)
    #     node.setText(1, typeStr)
    #     node.setText(2, desc)
            
    #     # Truncate description and add text box if needed
    #     if len(desc) > 100:
    #         desc = desc[:97] + '...'
    #         if widget is None:
    #             widget = QtWidgets.QPlainTextEdit(str(data))
    #             widget.setMaximumHeight(200)
    #             widget.setReadOnly(True)
        
    #     # Add widget to new subnode
    #     if widget is not None:
    #         self.widgets.append(widget)
    #         subnode = QtWidgets.QTreeWidgetItem(["", "", ""])
    #         node.addChild(subnode)
    #         self.setItemWidget(subnode, 0, widget)
    #         subnode.setFirstColumnSpanned(True)
            
    #     # recurse to children
    #     for key, data in childs.items():
    #         self.buildTree(data, node, str(key), path=path+(key,))

    # def parse(self, data):
    #     """
    #     Given any python object, return:
    #       * type
    #       * a short string representation
    #       * a dict of sub-objects to be parsed
    #       * optional widget to display as sub-node
    #     """
    #     # defaults for all objects
    #     typeStr = type(data).__name__
    #     if typeStr == 'instance':
    #         typeStr += ": " + data.__class__.__name__
    #     widget = None
    #     desc = ""
    #     childs = {}
        
    #     # type-specific changes
    #     if isinstance(data, dict):
    #         desc = "length=%d" % len(data)
    #         if isinstance(data, OrderedDict):
    #             childs = data
    #         else:
    #             try:
    #                 childs = OrderedDict(sorted(data.items()))
    #             except TypeError: # if sorting falls
    #                 childs = OrderedDict(data.items())
    #     elif isinstance(data, (list, tuple)):
    #         desc = "length=%d" % len(data)
    #         childs = OrderedDict(enumerate(data))
    #     elif HAVE_METAARRAY and (hasattr(data, 'implements') and data.implements('MetaArray')):
    #         childs = OrderedDict([
    #             ('data', data.view(np.ndarray)),
    #             ('meta', data.infoCopy())
    #         ])
    #     elif isinstance(data, np.ndarray):
    #         desc = "shape=%s dtype=%s" % (data.shape, data.dtype)
    #         table = TableWidget()
    #         table.setData(data)
    #         table.setMaximumHeight(200)
    #         widget = table
    #     elif isinstance(data, types.TracebackType):  ## convert traceback to a list of strings
    #         frames = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))
    #         #childs = OrderedDict([
    #             #(i, {'file': child[0], 'line': child[1], 'function': child[2], 'code': child[3]})
    #             #for i, child in enumerate(frames)])
    #         #childs = OrderedDict([(i, ch) for i,ch in enumerate(frames)])
    #         widget = QtWidgets.QPlainTextEdit('\n'.join(frames))
    #         widget.setMaximumHeight(200)
    #         widget.setReadOnly(True)
    #     else:
    #         desc = str(data)
        
    #     return typeStr, desc, childs, widget
        




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
    # tree = pg.DataTreeWidget(data=data)
    tree = CustomFormattingDataTreeWidget(data=data)
    tree.show()
    tree.setWindowTitle(f'PhoOutputDataTreeApp: pyqtgraph DataTreeWidget: {title}')
    tree.resize(800,600)
    return tree, app



if __name__ == '__main__':
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
    
    pg.exec()
