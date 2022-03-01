"""
This example demonstrates the use of ColorBarItem, which displays a simple interactive color bar.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphoplacecellanalysis.GUI.Qt.PhoUIContainer import PhoUIContainer
# from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionWidget import PlacefieldVisualSelectionWidget
from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControlWidget import PlacefieldVisualSelectionWidget


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = PhoUIContainer()
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        self.ui.backgroundWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.ui.backgroundWidget)
        self.setWindowTitle('Pho testing PlacefieldVisualSelectionWidget')
        self.resize(800,700)
        self.show()

        self.ui.pf_layout = QtWidgets.QHBoxLayout()
        self.ui.pf_layout.setSpacing(0)
        self.ui.pf_layout.setObjectName("horizontalLayout")
        
        self.ui.backgroundWidget.setLayout(self.ui.pf_layout)
        
        self.pf_widgets = []
        for i in np.arange(30):
            curr_pf_string = f"pf[{i}]"
            curr_widget = PlacefieldVisualSelectionWidget()
            curr_widget.setObjectName(curr_pf_string)
            curr_widget.name = curr_pf_string # be sure to set the name
            # TODO: set the color and such too
            self.ui.pf_layout.addWidget(curr_widget)
            self.pf_widgets.append(curr_widget)
        
        
        

mkQApp("PlacefieldVisualSelectionWidget Example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    pg.exec()
