# PlacefieldVisualSelectionWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlacefieldVisualSelectionWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import importlib
import sys
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# from PyQt5 import QtGui, QtWidgets, uic
# from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QToolButton
# from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
# from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
# from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir


## IMPORTS:
from pyqtgraph.widgets.ColorButton import ColorButton

from pyphoplacecellanalysis.GUI.Qt.PhoUIContainer import PhoUIContainer
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt import PlacefieldVisualSelectionWidget


class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
	""" Aims to serve the same purpose of the Panel widget.
	Usage Example:
		def placefieldSelectionWidgetExample(title='PhoPfSelectionWidgetExampleApp'):
			app = pg.mkQApp(title)
			
			w = PlacefieldVisualSelectionWidget()
			
			window = QtWidgets.QWidget()
			layout = QtGui.QVBoxLayout()
			layout.addWidget(w)

			window.setLayout(layout)

			window.show()
			window.resize(500,500)
			window.setWindowTitle('pho example: PfSelectionWidget')

			return window, app

		if __name__ == '__main__':
			win, app = placefieldSelectionWidgetExample()
			pg.exec()


	"""
	def __init__(self, parent=None):
		super().__init__(parent=parent) # Call the inherited classes __init__ method
		# self.ui = uic.loadUi("PlacefieldVisualSelectionWidget.ui", self) # Load the .ui file
		# self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/PlacefieldVisualSelectionWidget.ui", self) # Load the .ui file
		self.ui = PhoUIContainer()

		self.initUI()
		self.show() # Show the GUI


	def initUI(self):
		self.ui.btnTitle = QtWidgets.QPushButton('Title')
		self.ui.btnTitle.setObjectName("btnTitle")
  
		self.ui.btnColorButton = ColorButton(self)
		self.ui.btnColorButton.setObjectName("btnColorButton")
		
		self.ui.chkbtnPlacefield = QtWidgets.QToolButton()
		self.ui.chkbtnPlacefield.setObjectName("chkbtnPlacefield")  
		self.ui.chkbtnPlacefield.setText('pf')
		self.ui.chkbtnPlacefield.setCheckable(True)
		self.ui.chkbtnPlacefield.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

		self.ui.chkbtnSpikes = QtWidgets.QToolButton()
		self.ui.chkbtnSpikes.setObjectName("chkbtnSpikes")  
		self.ui.chkbtnSpikes.setText('spikes')
		self.ui.chkbtnSpikes.setCheckable(True)
		self.ui.chkbtnSpikes.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

		## Create a grid layout to manage the widgets size and position
		self.layout = QtGui.QGridLayout()
		self.setLayout(self.layout)
  
		self.layout.addWidget(self.ui.btnTitle, 0, 0, 1, 2)   # button goes in upper-left. Spans 1 row and 2 columns
		self.layout.addWidget(self.ui.btnColorButton, 1, 0, 1, 2) # spans 2 columns
		self.layout.addWidget(self.ui.chkbtnPlacefield, 2, 0)
		self.layout.addWidget(self.ui.chkbtnSpikes, 3, 0)



	def __str__(self):
 		return 



