# FigureFormatConfigControls.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\FigureFormatConfigControls\FigureFormatConfigControls.ui automatically by PhoPyQtClassGenerator VSCode Extension
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
# from pyPhoPlaceCellAnalysis.GUI.Qt.FigureFormatConfigControls  import FigureFormatConfigControls
from pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls.Uic_AUTOGEN_FigureFormatConfigControls import Ui_Form


def pair_optional_value_widget(checkBox, valueWidget):
    self.checkBox.toggled['bool'].connect(self.spinBox.setEnabled) # type: ignore
    

class FigureFormatConfigControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
  
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.

        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        self.checkBox_3
        self.spinBox_3
        
    def on_update_values(self):
        print('on_update_values')
        
        


    def __str__(self):
         return 

"""
lblPropertyName

"""

