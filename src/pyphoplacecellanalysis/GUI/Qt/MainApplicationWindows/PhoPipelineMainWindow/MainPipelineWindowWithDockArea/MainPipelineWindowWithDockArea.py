# MainPipelineWindowWithDockArea.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Windows\MainPipelineWindowWithDockArea.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from datetime import datetime, timezone, timedelta
import numpy as np
from enum import Enum

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.PyQtPlot.Windows import MainPipelineWindowWithDockArea
path = os.path.dirname(os.path.abspath(__file__))
# uiFile = os.path.join(path, 'MainPipelineWindow.ui')
uiFile = os.path.join(path, 'MainPipelineWindowWithDockArea.ui') # mostly empty


class MainPipelineWindowWithDockArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        #Load the UI Page
        uic.loadUi(uiFile, self) # load from the ui file
          # self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/MainApplicationWindows/PhoPipelineMainWindow/MainPipelineWindowWithDockArea/MainPipelineWindowWithDockArea.ui", self) # Load the .ui file
  
        # self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Windows/MainPipelineWindowWithDockArea.ui", self) # Load the .ui file
        # self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/MainApplicationWindows/PhoPipelineMainWindow/MainPipelineWindowWithDockArea/MainPipelineWindowWithDockArea.ui", self) # Load the .ui file
        # 'GUI/Qt/MainApplicationWindows/PhoPipelineMainWindow/MainPipelineWindowWithDockArea/MainPipelineWindowWithDockArea.ui'

        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        pass


    def __str__(self):
        return 
