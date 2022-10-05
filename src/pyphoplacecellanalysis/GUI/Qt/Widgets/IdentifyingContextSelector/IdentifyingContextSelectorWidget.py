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
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.CtrlNodeMixins import ComboBoxCtrlOwnerMixin
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'IdentifyingContextSelectorWidget.ui')

# ==================================================================================================================== #
# IdentifyingContextSelectorWidget                                                                                     #
# ==================================================================================================================== #
class IdentifyingContextSelectorWidget(ComboBoxCtrlOwnerMixin, PipelineOwningMixin, QWidget): 
    """_summary_

    pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget.IdentifyingContextSelectorWidget
    
    Usage:
        from pyphoplacecellanalysis.GUI.Qt.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget

    """
    def __init__(self, parent=None, owning_pipeline=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        ## Set member properties:
        self._owning_pipeline = owning_pipeline

        self.initUI()
        # self.show() # Show the GUI

    def initUI(self):
        # self.ui.cmbIdentifyingContext.set = self.all_filtered_session_keys
        # self.ui.btnConfirm.clicked.
        # self.updateUi()
        pass


    def updateUi(self):
        ## Update Combo box items:
        ## Freeze signals:
        curr_combo_box = self.ui.cmbIdentifyingContext # QComboBox 
        curr_combo_box.blockSignals(True)
        
        ## Capture the previous selection:
        selected_index, selected_item_text = self.get_current_combo_item_selection(curr_combo_box)

        # Build updated list:
        # active_list_items = self.all_filtered_session_keys
        active_list_items = self.all_filtered_session_context_descriptions
        self.num_known_types = len(active_list_items)
        ## Build updated list:
        updated_list = active_list_items
        # updated_list.append('Custom...')

        self.replace_combo_items(curr_combo_box, updated_list)
        
        ## Re-select the previously selected item if possible:
        found_desired_index = self.try_select_combo_item_with_text(curr_combo_box, selected_item_text)
        ## Unblock the signals:
        curr_combo_box.blockSignals(False)




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = IdentifyingContextSelectorWidget()
    widget.show()
    sys.exit(app.exec_())
