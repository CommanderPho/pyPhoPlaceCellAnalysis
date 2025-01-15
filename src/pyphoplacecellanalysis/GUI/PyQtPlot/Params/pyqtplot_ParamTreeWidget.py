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

from pyphoplacecellanalysis.GUI.PyQtPlot.Params.BuildParamTypes import makeAllParamTypes

import pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes as pTypes
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
from pyphoplacecellanalysis.GUI.PyQtPlot.Params.SaveRestoreStateParamHelpers import default_parameters_save_restore_state_button_children

def plot_paramTreeWidget(title='PhoParamTreeApp'):
    app = pg.mkQApp(title)
    # Build parameter tree:
    params = [
        makeAllParamTypes(),
        default_parameters_save_restore_state_button_children(),
        {'name': 'Custom context menu', 'type': 'group', 'children': [
            {'name': 'List contextMenu', 'type': 'float', 'value': 0, 'context': [
                'menu1',
                'menu2'
            ]},
            {'name': 'Dict contextMenu', 'type': 'float', 'value': 0, 'context': {
                'changeName': 'Title',
                'internal': 'What the user sees',
            }},
        ]},
    ]

    ## Create tree of Parameter objects
    p = Parameter.create(name='params', type='group', children=params)

    ## If anything changes in the tree, print a message
    def change(param, changes):
        print("tree changes:")
        for param, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
        
    p.sigTreeStateChanged.connect(change)


    def valueChanging(param, value):
        print("Value changing (not finalized): %s %s" % (param, value))
        
    # Too lazy for recursion:
    for child in p.children():
        child.sigValueChanging.connect(valueChanging)
        for ch2 in child.children():
            ch2.sigValueChanging.connect(valueChanging)
            

    def save():
        global state
        state = p.saveState()

    def restore():
        global state
        add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
        rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
        p.restoreState(state, addChildren=add, removeChildren=rem)
    p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
    p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)

    ## Build the actual ParameterTree widget, the core GUI
    paramTree = ParameterTree()
    paramTree.setParameters(p, showTop=False)
    paramTree.show()
    paramTree.setWindowTitle(f'PhoParamTreeApp: pyqtgraph ParameterTree: {title}')
    paramTree.resize(800,600)
    return paramTree, app



if __name__ == '__main__':
    _out = plot_paramTreeWidget()
    pg.exec()
