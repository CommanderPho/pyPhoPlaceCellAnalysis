# ThinButtonBarWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ThinButtonBar\ThinButtonBarWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import documentation_tags, metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'ThinButtonBarWidget.ui')


@metadata_attributes(short_name=None, tags=['ui', 'widget', 'button-bar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 08:30', related_items=[])
class ThinButtonBarWidget(QWidget):
    """ 
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget

    
    """
    sigCopySelections = QtCore.pyqtSignal()
    sigRefresh = QtCore.pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.ui.btnCopySelectedEpochs.pressed.connect(self.on_copy_selections)
        self.ui.btnRefresh.pressed.connect(self.on_perform_refresh)
        # currentTextChanged.connect(self.on_jump_combo_series_changed)
        pass

    @pyqtExceptionPrintingSlot()
    def on_copy_selections(self):
        """ 
        """
        self.sigCopySelections.emit()
        
    @pyqtExceptionPrintingSlot()
    def on_perform_refresh(self):
        """ 
        """
        self.sigRefresh.emit()



## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = ThinButtonBarWidget()
    widget.show()
    sys.exit(app.exec_())
