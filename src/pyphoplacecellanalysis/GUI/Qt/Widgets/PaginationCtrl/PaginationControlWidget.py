# PaginationControlWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PaginationCtrl\PaginationControlWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import documentation_tags, metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

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
    """Contains the current state (the current_page_idx)."""
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
    

@metadata_attributes(short_name=None, tags=['pagination'], input_requires=[], output_provides=[],
    uses=[], used_by=[],
    related_items=['pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget.PaginationControlWidgetState'], pyqt_signals_emitted=['jump_previous_page()','jump_next_page()','jump_to_page(int)'],
    creation_date='2023-05-08 14:28')
class PaginationControlWidget(QWidget):
    """ 2023-05-08 - Provides a basic pagination management widget with a left arrow button, a integer spin-box, and a right arrow button to control the active pages
    

    ## Simple example:

        def on_paginator_control_widget_jump_to_page(page_idx: int):
            print(f'on_paginator_control_widget_jump_to_page(page_idx: {page_idx})')

        a_paginator_controller_widget.jump_to_page.connect(on_paginator_control_widget_jump_to_page)


    ## Practical Example of using PaginationControlWidget to update a plot:
    from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget


    """
    # Jump Target Items
    jump_previous_page = QtCore.pyqtSignal()
    jump_next_page = QtCore.pyqtSignal()
    jump_to_page = QtCore.pyqtSignal(int)
    
    def get_total_pages(self):
        # return (len(self.data) + self.page_size - 1) // self.page_size
        return self.state.n_pages
    
    @property
    def current_page_idx(self):
        """ the 0-based index of the current page. """
        return self.state.current_page_idx
    @current_page_idx.setter
    def current_page_idx(self, value):
        print(f'current_page_idx setter should be depricated!')
        self.state.current_page_idx = value



    def __init__(self, n_pages:int = 9, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.state = PaginationControlWidgetState(n_pages=n_pages, current_page_idx=0)
        
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        # self.ui.spinBoxPage
        self.ui.spinBoxPage.setMinimum(1)
        self.ui.spinBoxPage.setMaximum(self.get_total_pages())
        self.ui.spinBoxPage.setValue(self.current_page_idx+1)
        self.ui.spinBoxPage.valueChanged.connect(self.go_to_page)

        # Jump-to:
        self.ui.btnJumpToPrevious.pressed.connect(self.on_jump_prev_page)
        self.ui.btnJumpToNext.pressed.connect(self.on_jump_next_page)
        # currentTextChanged.connect(self.on_jump_combo_series_changed)

        self._on_update_pagination()
        self._update_series_action_buttons()

        self.jump_to_page.connect(lambda page_idx: self._update_series_action_buttons())


    @pyqtExceptionPrintingSlot()
    def _on_update_pagination(self):
        """ called when the number of pages is updated. """
        # self.ui.spinBoxPage
        self.ui.spinBoxPage.setMinimum(1)
        self.ui.spinBoxPage.setMaximum(self.get_total_pages())
        current_page_number = self.current_page_idx + 1 # page number (1-based) is always one greater than the page_index (0-based)
        self.ui.spinBoxPage.setValue(current_page_number)
        self.ui.spinBoxPage.setSuffix(f"/{self.get_total_pages()}")


    @pyqtExceptionPrintingSlot()
    def _update_series_action_buttons(self):
        """ conditionally update whether the buttons are enabled based on whether we have a valid series selection. """        
        self.ui.btnJumpToPrevious.setEnabled(self.state.can_move_left)
        self.ui.btnJumpToNext.setEnabled(self.state.can_move_right)
        # self.ui.btnCurrentIntervals_Remove.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Customize.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Extra.setEnabled(has_valid_series_selection)


    @pyqtExceptionPrintingSlot()
    def on_jump_next_page(self):
        """ 
        """
        updated_page_idx = self.current_page_idx+1
        updated_page_number = updated_page_idx + 1 # page number (1-based) is always one greater than the page_index (0-based)
        self.state.current_page_idx = updated_page_idx ## update the state
        self.ui.spinBoxPage.blockSignals(True)
        self.ui.spinBoxPage.setValue(updated_page_number)
        self.jump_next_page.emit()
        self.jump_to_page.emit(updated_page_idx)
        self.ui.spinBoxPage.blockSignals(False)
        
    @pyqtExceptionPrintingSlot()
    def on_jump_prev_page(self):
        """ 
        """
        updated_page_idx = self.current_page_idx-1
        updated_page_number = updated_page_idx + 1 # page number (1-based) is always one greater than the page_index (0-based)
        self.state.current_page_idx = updated_page_idx ## update the state
        # self.ui.spinBoxPage.valueChanged.
        self.ui.spinBoxPage.blockSignals(True)
        self.ui.spinBoxPage.setValue(updated_page_number) # +1 because it's supposed to reflect the page_number instead of the page_index
        self.jump_previous_page.emit()
        self.jump_to_page.emit(updated_page_idx)
        self.ui.spinBoxPage.blockSignals(False)

    @pyqtExceptionPrintingSlot(int)
    def go_to_page(self, page_number):
        """ one-based page_number """
        if page_number > 0 and page_number <= self.get_total_pages():
            updated_page_idx = page_number - 1 # convert the page number to a page index
            self.state.current_page_idx = updated_page_idx ## update the state
            self.jump_to_page.emit(updated_page_idx)
            

    @pyqtExceptionPrintingSlot(int)
    def update_page_idx(self, updated_page_idx: int):
        """ this value is safe to bind to. """
        return self.programmatically_update_page_idx(updated_page_idx=updated_page_idx, block_signals=False)


    def programmatically_update_page_idx(self, updated_page_idx: int, block_signals:bool=False):
        """ Programmatically updates the spinBoxPage with the zero-based page_number 
        page number (1-based) is always one greater than the page_index (0-based)
        """
        updated_page_number = updated_page_idx + 1 # page number (1-based) is always one greater than the page_index (0-based)
        assert ((updated_page_number > 0) and (updated_page_number <= self.get_total_pages())), f"programmatically_update_page_idx(updated_page_idx: {updated_page_idx}) is invalid! updated_page_number: {updated_page_number}, total_pages: {self.get_total_pages()}"
        self.state.current_page_idx = updated_page_idx ## update the state
        self.ui.spinBoxPage.blockSignals(True)
        self.ui.spinBoxPage.setValue(updated_page_number) # +1 because it's supposed to reflect the page_number instead of the page_index
        if not block_signals:
            self.jump_to_page.emit(updated_page_idx)
        self.ui.spinBoxPage.blockSignals(False)

            




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = PaginationControlWidget(n_pages=9)
    widget.show()
    sys.exit(app.exec_())
