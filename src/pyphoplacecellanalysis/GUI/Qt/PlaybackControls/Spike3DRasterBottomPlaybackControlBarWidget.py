# Spike3DRasterBottomPlaybackControlBar.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlaybackControls\Spike3DRasterBottomPlaybackControlBar.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
from enum import Enum

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.PlaybackControls import Spike3DRasterBottomPlaybackControlBar

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget  # Generated file from .ui

class Spike3DRasterBottomPlaybackControlBar(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_RootWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        pass


    def __str__(self):
         return 





## Start Qt event loop
# if __name__ == '__main__':
#     app = mkQApp("PlacefieldVisualSelectionWidget Example")
#     widget = PlacefieldVisualSelectionWidget()
#     widget.show()
#     pg.exec()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    rootForm = QtWidgets.QWidget()
    ui = Spike3DRasterBottomPlaybackControlBar()
    ui.setupUi(rootForm)
    rootForm.show()
    sys.exit(app.exec_())
