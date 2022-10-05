# IdentifyingContextSelectorWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\IdentifyingContextSelector\IdentifyingContextSelectorWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from .Uic_AUTOGEN_IdentifyingContextSelectorWidget import Ui_Form

class IdentifyingContextSelectorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.


        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = IdentifyingContextSelectorWidget()
    widget.show()
    sys.exit(app.exec_())
