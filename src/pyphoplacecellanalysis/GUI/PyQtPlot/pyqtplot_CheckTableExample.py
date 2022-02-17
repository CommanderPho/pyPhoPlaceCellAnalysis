import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

from pyqtgraph.widgets.FeedbackButton import FeedbackButton

def checkTableWidgetExample(title='PhoCheckTableWidgetExampleApp'):
    app = pg.mkQApp(title)
    # w = pg.CheckTable(['Column 1','Column 2','Column 3'])
    col_labels = ['pre', 'maze1', 'post1', 'maze2', 'post2']
    w = pg.CheckTable(col_labels)
    w.layout.setSpacing(10)

    def on_add_row_clicked(evt):
        w.addRow('New')
        
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
