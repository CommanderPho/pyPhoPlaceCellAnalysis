# IdentifyingContextSelectorWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\IdentifyingContextSelector\IdentifyingContextSelectorWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import sys
import os
import numpy as np

# from pyphoplacecellanalysis.External.pyqtgraph import QtWidgets, QtCore, QtGui


from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QCheckBox
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.GUI.Qt.Mixins.ComboBoxMixins import KeysListAccessingMixin, ComboBoxCtrlOwningMixin
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin

from pyqt_checkbox_table_widget.checkBoxTableWidget import CheckBoxTableWidget


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'IdentifyingContextSelectorWidget.ui')

# ==================================================================================================================== #
# IdentifyingContextSelectorWidget                                                                                     #
# ==================================================================================================================== #
class IdentifyingContextSelectorWidget(ComboBoxCtrlOwningMixin, PipelineOwningMixin, QWidget): 
    """ Allows selecting an IdentifyingContext from a dropdown list
    
    Usage:
        from pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget

        widget = IdentifyingContextSelectorWidget(owning_pipeline=curr_active_pipeline)
        widget.show()

    """

    sigContextChanged = pyqtSignal(object, object) # newKey: str, newContext: IdentifyingContext
    sigMultiContextChanged = pyqtSignal(dict) #contexts: dict<str:IdentifyingContext> 
    
    # ==================================================================================================================== #
    # def __init__(self, parent=None, owning_pipeline=None, enable_multi_context_select:bool=False):
    #     super().__init__(parent=parent) # Call the inherited classes __init__ method
    #     self.ui = uic.loadUi(uiFile, self) # Load the .ui file

    #     ## Set member properties:
    #     self._enable_multi_context_select = enable_multi_context_select
    #     self._owning_pipeline = owning_pipeline

    #     self.initUI()
    #     # self.show() # Show the GUI

    @property
    def has_valid_pipeline(self) -> bool:
        """ Whether there is a currently an item selected or not. """
        if self.owning_pipeline is None:
            return False
        else:
            return True
        

    def __init__(self, parent=None): # owning_pipeline=None, enable_multi_context_select:bool=False
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        ## Set member properties:
        self._enable_multi_context_select = False
        self._owning_pipeline = None
        self._last_context_table_rows = None
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        # self.show() # Show the GUI

    def initUI(self):
        # self.ui.cmbIdentifyingContext.set = self.all_filtered_session_keys
        self.ui.cmbIdentifyingContext.currentIndexChanged.connect(self.on_selected_context_index_changed)
        # self.ui.btnConfirm.clicked.

        ## Setup checktable:
        if self._enable_multi_context_select:
            self._programmaticallyBuildCheckTable()

        if self._owning_pipeline is not None:
            self.updateUi()
    

    def updateUi(self):
        if self._owning_pipeline is not None:
            self._tryUpdateComboItemsUi()
            if self._enable_multi_context_select:
                self._tryUpdateCheckTableUi()

    
    def build_for_pipeline(self, curr_active_pipeline):
        self._owning_pipeline = curr_active_pipeline
        if self._owning_pipeline is not None:
            self.updateUi()
        

    def enable_context_selection(self, is_enabled: bool):
        self.ui.groupBox.setEnabled(is_enabled)
        action_buttons_list = (self.ui.btnConfirm, self.ui.btnRefresh, self.ui.btnRevert)
        for a_btn in action_buttons_list:
            a_btn.setEnabled(is_enabled)
        self.ui.cmbIdentifyingContext.setEnabled(is_enabled)

    # ==================================================================================================================== #
    # Single-context ComboBox Dropdown                                                                                         #
    # ==================================================================================================================== #
    
    @property
    def current_selected_context_key(self):
        """The current_selected_context property."""
        ## Capture the previous selection:
        if not hasattr(self, 'ui'):
            return None

        active_keys_list = self.all_filtered_session_keys.copy()
        selected_index, selected_item_text = self.get_current_combo_item_selection(self.ui.cmbIdentifyingContext)
        if selected_index < 0:
            return None # no selection currently
        else:
            curr_selected_key = active_keys_list[selected_index]
            return curr_selected_key
    
    @property
    def has_valid_selection(self):
        """ Whether there is a currently an item selected or not. """
        if self.current_selected_context_key is None:
            return False
        else:
            return True

    @property
    def current_selected_context(self):
        """The IdentifyingContext that's currently selected, or None."""
        if self.current_selected_context_key is None:
            return None
        else:
            return self.all_filtered_session_contexts[self.current_selected_context_key]

    def _tryUpdateComboItemsUi(self):
        """ tries to update the combo box items. If an item was previously selected before the update, it tries to re-select the same item. """

        ## Update Combo box items:
        curr_combo_box = self.ui.cmbIdentifyingContext # QComboBox 

        ## Freeze signals:
        curr_combo_box.blockSignals(True)
        
        ## Capture the previous selection:
        selected_index, selected_item_text = self.get_current_combo_item_selection(curr_combo_box)
        had_previous_selected_item = (selected_item_text is not None)

        # Build updated list:
        if self.has_valid_pipeline:
            # active_list_items = self.all_filtered_session_keys
            active_list_items = self.all_filtered_session_context_descriptions
            ## Build updated list:
            updated_list = active_list_items
            # updated_list.append('Custom...')
        else:
            updated_list = []

        self.replace_combo_items(curr_combo_box, updated_list)
        
        ## Re-select the previously selected item if possible:
        if not had_previous_selected_item:
            # no previously selected item. Instead, select the first item.
            self._trySelectFirstComboItem()
        found_desired_index = self.try_select_combo_item_with_text(curr_combo_box, selected_item_text)
        ## Unblock the signals:
        curr_combo_box.blockSignals(False)

    def _trySelectFirstComboItem(self):
        """ tries to select the first item (index 0) if possible. Otherwise, fails gracefully.
        Internally calls self.try_select_combo_item_with_text(...)
         """
        # no previously selected item. Instead, select the first item.
        current_list = self.all_filtered_session_context_descriptions
        if (len(current_list) > 0):
            selected_item_text = current_list[0] # get the first item text to try and select.
            found_desired_index = self.try_select_combo_item_with_text(self.ui.cmbIdentifyingContext, selected_item_text)
        else:
            print(f'WARNING: could not select any default items because the list was empty.')
            found_desired_index = None
        return found_desired_index

    @pyqtExceptionPrintingSlot(int)
    def on_selected_context_index_changed(self, new_index):
        if new_index < 0:
            new_key = None
            new_context = None
        else:
            new_key = self.all_filtered_session_keys[new_index]
            new_description = self.all_filtered_session_context_descriptions[new_index]
            new_context = self.all_filtered_session_contexts[new_key]
        print(f'on_selected_context_index_changed: {new_index}, {new_key}, {new_description}, {new_context}')
        self.sigContextChanged.emit(new_key, new_context)

    # ==================================================================================================================== #
    # Multi-context Checkbox Table                                                                                         #
    # ==================================================================================================================== #

    @property
    def check_table_ctrl(self):
        """ The multi-context checkbox table widget """
        if self._enable_multi_context_select:
            return self.ui.checkTable
        else:
            return None

    @property
    def current_selected_multi_context_indicies(self):
        """The indicies of the currently selected contexts (in the multi-context checkbox list).
            e.g. [0, 1]
        """
        if self.check_table_ctrl is None:
            ## pre-init, probably during the `self.ui = uic.loadUi(uiFile, self)` phase which for some reason tries to resolve all of the properties
            return None
        return self.check_table_ctrl.getCheckedRows()

    @property
    def current_selected_multi_context_descriptions(self):
        """The description strings for each contexts (that act as an indentifier).
            e.g. ['kdiba_2006-6-13_14-42-6_maze1_PYR', 'kdiba_2006-6-13_14-42-6_maze2_PYR']
        """
        if self._last_context_table_rows is None:
            return None
        return [self._last_context_table_rows[i] for i in self.current_selected_multi_context_indicies]
    
    @property
    def current_selected_multi_contexts(self):
        """A dictionary of the actual context items.
            e.g. {'maze1_PYR': <neuropy.utils.result_context.IdentifyingContext at 0x2d11d4d5640>,
            'maze2_PYR': <neuropy.utils.result_context.IdentifyingContext at 0x2d11d4d56d0>}

        """
        return {list(self.owning_pipeline.filtered_contexts.keys())[i]:list(self.owning_pipeline.filtered_contexts.values())[i] for i in self.current_selected_multi_context_indicies}

    def _programmaticallyBuildCheckTable(self, col_labels=['compute'], rows=[]):
        """ builds a multi-context selection checkbox table to enable the user to specify multiple relevent contexts to operate on. """
        ## Build Context Table
        self._last_context_table_rows = None

        self.ui.selectAllChkBox = QCheckBox('Check all')
        self.ui.checkTable = CheckBoxTableWidget()

        curr_active_context_descriptions = self.all_filtered_session_context_descriptions
        self._tryRebuildCheckTableUi(curr_active_context_descriptions, checked_contexts=[]) # None checked by default
        self.ui.selectAllChkBox.stateChanged.connect(self.ui.checkTable.toggleState) # if allChkBox is checked, tablewidget checkboxes will also be checked 
        
        self.ui.verticalLayout.addWidget(self.ui.selectAllChkBox)
        self.ui.verticalLayout.addWidget(self.ui.checkTable)

        self.ui.checkTable.checkedSignal.connect(self.on_checktable_checked_state_changed) # checkedSignal = pyqtSignal(int, Qt.CheckState)

    def _tryRebuildCheckTableUi(self, curr_active_context_descriptions=[], checked_contexts=[]):
        """ rebuilds the entire checkbox table from the updated list of contexts """
        _curr_context_table_rows = self._last_context_table_rows
        if _curr_context_table_rows is not None:
            self.ui.checkTable.clearContents() # clears all extant rows
            _curr_selected_table_rows = deepcopy(_curr_context_table_rows[np.array(self.check_table_ctrl.getCheckedRows())]) # gets the indicies of the checked rows
            print(f'_curr_selected_table_rows: {_curr_selected_table_rows}')
        else:
            _curr_selected_table_rows = None

        # Backup the new values that we'll be using
        self._last_context_table_rows = deepcopy(curr_active_context_descriptions)
        num_rows = len(self._last_context_table_rows)
        self.ui.checkTable.setRowCount(num_rows)
        self.ui.checkTable.stretchEveryColumnExceptForCheckBox() # stretch every section of tablewidget except for check box section
        for i in range(self.ui.checkTable.rowCount()):
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter) # align
            item.setText(self._last_context_table_rows[i])
            self.ui.checkTable.setItem(i, 1, item)
            self.ui.checkTable.setCheckedAt(idx=i, f=True)

        # Set the last row to unchecked by default:
        self.ui.checkTable.setCheckedAt(idx=num_rows-1, f=False)


        # Restore previously checked items after updating:
        if _curr_selected_table_rows is not None:
            # TODO: do selection
            print(f'_curr_selected_table_rows: {_curr_selected_table_rows}')
        else:
            # by default select all but the last one
            # _curr_selected_table_rows = [self._last_context_table_rows[i] for i in np.arange(num_rows-1)]
            # for i in np.arange(num_rows-1):
            #     self.ui.checkTable.setCheckedAt(idx=i, f=True)
            # self.ui.checkTable.setCheckedAt(idx=num_rows, f=False)
            pass

    def _tryUpdateCheckTableUi(self):
        """ tries to update the combo box items. If an item was previously selected before the update, it tries to re-select the same item. """
        curr_checked_row_indicies = self.check_table_ctrl.getCheckedRows() # gets the indicies of the checked rows
        print(f'_tryUpdateCheckTableUi():\n\curr_checked_row_indicies: {curr_checked_row_indicies}')
        curr_context_rows = self._last_context_table_rows
        if curr_context_rows is not None:
            _curr_selected_table_rows = [curr_context_rows[i] for i in curr_checked_row_indicies] # gets the values of the checked rows
            print(f'\t_curr_selected_table_rows: {_curr_selected_table_rows}')
        else:
            print(f'\tno rows selected.')
            _curr_selected_table_rows = None

        return
        # curr_active_contexts = self.all_filtered_session_context_descriptions
        # num_rows = len(curr_active_contexts)
        # self.ui.checkTable.setRowCount(num_rows)
        
        # self.check_table_ctrl.setCheckedAt(idx: int, f: bool)
        # for i in range(self.ui.checkTable.rowCount()):
        #     item = self.ui.checkTable.getItem(i, 1)
        #     item.setText(curr_active_contexts[i])
            
    # checkedSignal = pyqtSignal(int, Qt.CheckState) # (rowIndex: int, flag: Qt.CheckState)
    @pyqtExceptionPrintingSlot(int, Qt.CheckState)
    def on_checktable_checked_state_changed(self, row, state):
        print(f'on_checktable_checked_state_changed(row:{row}, state:{state})')
        # Emit sigMultiContextChanged = pyqtSignal(dict) #contexts: dict<str:IdentifyingContext> 
        self.sigMultiContextChanged.emit(self.current_selected_multi_contexts) # emit the currently selected multi-contexts when any one of them change


    # sigMultiContextChanged = pyqtSignal(list, dict) #names: list<str>, contexts: dict<str:IdentifyingContext> 


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    from pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget

    widget = IdentifyingContextSelectorWidget(owning_pipeline=curr_active_pipeline)
    widget.show()

    sys.exit(app.exec_())
