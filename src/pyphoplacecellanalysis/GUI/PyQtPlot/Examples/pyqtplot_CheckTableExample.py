import importlib
import sys
from pathlib import Path
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

from pyphoplacecellanalysis.External.pyqtgraph.widgets.FeedbackButton import FeedbackButton

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.core.neuron_identities import NeuronType


def checkTableWidgetExample(title='PhoCheckTableWidgetExampleApp'):
    app = pg.mkQApp(title)
    # w = pg.CheckTable(['Column 1','Column 2','Column 3'])
    # col_labels = ['pre', 'maze1', 'post1', 'maze2', 'post2']
    # col_labels = ['pre', 'maze1', 'post1', 'maze2', 'post2']
    # col_labels = NeuronType.longClassNames()
    
    col_labels = NeuronType.__members__
    
    
    w = pg.CheckTable(col_labels)
    w.layout.setSpacing(10)

    def on_add_row_clicked(evt):
        w.addRow('New')
        
    def on_table_check_changed(row, col, state):
        # note row: int, col: str, state: 0 for unchecked or 2 for checked
        print(f'on_table_check_changed(row: {row}, col: {col}, state: {state})')
       
    w.sigStateChanged.connect(on_table_check_changed)
    
     
    window = QtWidgets.QWidget()
    layout = QtGui.QVBoxLayout()
    layout.addWidget(w)

    addRowBtn = QtWidgets.QPushButton('Add Row')
    addRowBtn.setObjectName("addRowBtn")
    addRowBtn.clicked.connect(on_add_row_clicked)
        
    layout.addWidget(addRowBtn)
    window.setLayout(layout)

    # w.show()
    window.show()
    window.resize(500,500)
    window.setWindowTitle('pyqtgraph example: CheckTable')

    # w.resize(500,500)
    # w.setWindowTitle('pyqtgraph example: CheckTable')

    rows_data = [f'row[{i}]' for i in np.arange(8)]
    w.updateRows(rows_data)

    return window, app



if __name__ == '__main__':
    win, app = checkTableWidgetExample()
    pg.exec()
