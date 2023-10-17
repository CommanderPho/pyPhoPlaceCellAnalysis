# NeuronVisualSelectionControlsWidgetBase.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\NeuronVisualSelectionControls\NeuronVisualSelectionControlsWidgetBase.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from pathlib import Path
from pyphocorehelpers.print_helpers import DocumentationFilePrinter

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

from pyphocorehelpers.gui.Qt.TopLevelWindowHelper import print_widget_hierarchy
## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'NeuronVisualSelectionControlsWidgetBase.ui')

# from PyQt5.QtWidgets import QWidget

class NeuronVisualSelectionControlsWidgetBase(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.ui.chkbtnPlacefield.setVisible(False)
        self.ui.chkbtnSpikes.setVisible(False)
        # self.ui.horizontalLayout_BottomRow.set #setVisible(False)
        # doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='NeuronVisualSelectionControlsWidgetBase')
        # print(f'self.ui: {self.ui}')
        # print(f'self.ui.children(): {self.ui.children()}')  # QWidget
        print_widget_hierarchy(self.ui)
        # doc_printer.save_documentation('NeuronVisualSelectionControlsWidgetBase', self.ui, non_expanded_item_keys=[], skip_print=False)
        # doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = NeuronVisualSelectionControlsWidgetBase()
    widget.ui.btnTitle.setText('02')
    widget.ui.btnTitle.setToolTip('aclu: 02')
    widget.show()
    sys.exit(app.exec_())
