# PaginationControlWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PaginationCtrl\PaginationControlWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'PaginationControlWidget.ui')

""" 


btnJumpToPrevious
comboActiveJumpTargetSeries
btnJumpToNext



"""

from attrs import define

@define(slots=False)
class PaginationControlWidgetState:
    """docstring for PaginationControlWidgetState."""
    n_pages: int = 0
    current_page_idx: int = 0

    @property
    def _max_valid_index(self):
        """The maximum valid index that current_page_idx can take."""
        return (self.n_pages-1)


    @property
    def can_move_left(self):
        """The can_move_left property."""
        proposed_index = self.current_page_idx - 1
        return (proposed_index >= 0) and (proposed_index <= self._max_valid_index)

    @property
    def can_move_right(self):
        """The can_move_left property."""
        proposed_index = self.current_page_idx + 1
        return (proposed_index >= 0) and (proposed_index <= self._max_valid_index)

    


class PaginationControlWidget(QWidget):
    # Jump Target Items
    jump_target_left = QtCore.pyqtSignal(str)
    jump_target_right = QtCore.pyqtSignal(str)
    jump_series_selection_changed = QtCore.pyqtSignal(str)
    

    def __init__(self, n_pages:int = 9, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.state = PaginationControlWidgetState(n_pages=9, current_page_idx=0)

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):

        # self.ui.spinBoxPage
        self.page_spin_box.setMinimum(1)
        self.page_spin_box.setMaximum(self.get_total_pages())
        self.page_spin_box.setValue(self.current_page+1)
        self.page_spin_box.valueChanged.connect(self.go_to_page)



        # Jump-to:
        self.ui.btnJumpToPrevious.pressed.connect(self.on_jump_prev_series_item)
        self.ui.btnJumpToNext.pressed.connect(self.on_jump_next_series_item)
        # currentTextChanged.connect(self.on_jump_combo_series_changed)
        self._update_series_action_buttons()

    def _update_series_action_buttons(self):
        """ conditionally update whether the buttons are enabled based on whether we have a valid series selection. """        
        self.ui.btnJumpToPrevious.setEnabled(self.state.can_move_left)
        self.ui.btnJumpToNext.setEnabled(self.state.can_move_right)
        # self.ui.btnCurrentIntervals_Remove.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Customize.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Extra.setEnabled(has_valid_series_selection)


    @QtCore.pyqtSlot()
    def on_jump_next_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_jump_next_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_right.emit(curr_jump_series_name)

    @QtCore.pyqtSlot()
    def on_jump_prev_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_jump_prev_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_left.emit(curr_jump_series_name)




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = PaginationControlWidget()
    widget.show()
    sys.exit(app.exec_())
