# ProgrammaticPipelineWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ProgrammaticPipelineWidget\ProgrammaticPipelineWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphoplacecellanalysis.GUI.Qt.Widgets.ProgrammaticPipelineWidget.Uic_AUTOGEN_ProgrammaticPipelineWidget import Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin

class ProgrammaticPipelineWidget(PipelineOwningMixin, QWidget):
    def __init__(self, parent=None, owning_pipeline=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method

        ## Set member properties:
        self._owning_pipeline = owning_pipeline

        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.initUI()
        self.show() # Show the GUI
        self.updateUi()

    def initUI(self):
        # self.ui.cmbIdentifyingContext.set = self.all_filtered_session_keys
        # self.ui.btnConfirm.clicked.
        # self.ui.contextSelectorWidget._owning_pipeline = self.owning_pipeline
        # self.updateUi()
        pass

    def updateUi(self):
        # Update UI for children controls:
        self.ui.contextSelectorWidget.updateUi()


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = ProgrammaticPipelineWidget()
    widget.show()
    sys.exit(app.exec_())
