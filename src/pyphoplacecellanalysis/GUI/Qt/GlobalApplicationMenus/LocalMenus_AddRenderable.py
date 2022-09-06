# LocalMenus_AddRenderable.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\LocalMenus_AddRenderable.ui automatically by PhoPyQtClassGenerator VSCode Extension
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.Uic_AUTOGEN_LocalMenus_AddRenderable import Ui_LocalMenus_AddRenderable

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus import LocalMenus_AddRenderable


# class LocalMenus_AddRenderable(QtWidgets.QWidget):
class LocalMenus_AddRenderable(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent=parent) # Call the inherited classes __init__ method
		# self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/GlobalApplicationMenus/LocalMenus_AddRenderable.ui", self) # Load the .ui file
		# AUTOGEN version:
		self.ui = Ui_LocalMenus_AddRenderable()
		self.ui.setupUi(self) # builds the design from the .ui onto this widget.
		self.initUI()
		self.show() # Show the GUI


	def initUI(self):
		pass


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("LocalMenus_AddRenderable Example")
    widget = LocalMenus_AddRenderable()
    widget.show()
    pg.exec()