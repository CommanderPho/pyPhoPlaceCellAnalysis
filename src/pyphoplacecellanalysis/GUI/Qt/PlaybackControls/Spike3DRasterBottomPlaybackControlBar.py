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


class Spike3DRasterBottomPlaybackControlBar(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent=parent) # Call the inherited classes __init__ method
		self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/PlaybackControls/Spike3DRasterBottomPlaybackControlBar.ui", self) # Load the .ui file


		self.initUI()
		self.show() # Show the GUI


	def initUI(self):
		pass


	def __str__(self):
 		return 
