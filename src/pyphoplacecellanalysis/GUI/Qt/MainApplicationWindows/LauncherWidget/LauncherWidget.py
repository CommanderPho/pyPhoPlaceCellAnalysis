# LauncherWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\MainApplicationWindows\LauncherWidget\LauncherWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import traceback
import types
import os
from typing import Optional
from functools import wraps

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import Plot
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'LauncherWidget.ui')

class LauncherWidget(QWidget):
    """ a programmatic launcher widget that displays a tree that can be programmatically updated.
    
    
    Currently trying to use it for programmatic access to the display functions.
    
    Usage:    
        from pyphoplacecellanalysis.External.pyqtgraph import QtWidgets, QtCore, QtGui
        from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.LauncherWidget.LauncherWidget import LauncherWidget

        widget = LauncherWidget()
        treeWidget = widget.mainTreeWidget # QTreeWidget
        widget.build_for_pipeline(curr_active_pipeline=curr_active_pipeline)
        widget.show()

        
    TODO:
        curr_fcn = curr_active_pipeline.registered_display_function_dict['_display_2d_placefield_result_plot_ratemaps_2D']
        print(str(curr_fcn.__code__.co_varnames)) # PyFunction_GetCode # ('computation_result', 'active_config', 'enable_saving_to_disk', 'kwargs', 'display_outputs', 'plot_variable_name', 'active_figure', 'active_pf_computation_params', 'session_identifier', 'fig_label', 'active_pf_2D_figures', 'should_save_to_disk')

     """
    _curr_active_pipeline_ref = None #  : Optional[Plot]

    @property
    def treeWidget(self):
        """The treeWidget property."""
        return self.mainTreeWidget # QTreeWidget
    
    @property
    def curr_active_pipeline(self):
        """The treeWidget property."""
        return self._curr_active_pipeline_ref
    

    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self._curr_active_pipeline_ref = None
        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        # Connect the itemDoubleClicked signal to the on_tree_item_double_clicked slot
        self.treeWidget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)

    # Define a function to be executed when a tree widget item is double-clicked
    # @QtCore.Slot(object, int)
    @pyqtExceptionPrintingSlot(object)
    def on_tree_item_double_clicked(self, item, column):
        print(f"Item double-clicked: {item}, column: {column}\n\t", item.text(column))
        # print(f'\titem.data: {item.data}')
        # raise NotImplementedError
        item_data = item.data(column, 0) # ItemDataRole 
        print(f'\titem_data: {item_data}')
        a_fn_handle = self.curr_active_pipeline.plot.__getattr__(item_data)
        return a_fn_handle()
        

    def build_for_pipeline(self, curr_active_pipeline):
        self._curr_active_pipeline_ref = curr_active_pipeline
        curr_active_pipeline.reload_default_display_functions()
        
        # Add root item
        displayFunctionTreeItem = QtWidgets.QTreeWidgetItem(["Display Functions"])
        # si.gitem = 
        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem)

        for a_fcn_name, a_fcn in curr_active_pipeline.registered_display_function_dict.items():
            # extract the info from the function:
            if hasattr(a_fcn, 'short_name') and a_fcn.short_name is not None:
                active_name = a_fcn.short_name or a_fcn_name
            else:
                active_name = a_fcn_name
                
            print(f'adding {active_name}')
            childDisplayFunctionTreeItem = QtWidgets.QTreeWidgetItem([active_name])
            # childDisplayFunctionTreeItem.setText(0, active_name)
            # Set the tooltip for the item
            active_tooltip_text = curr_active_pipeline.registered_display_function_docs_dict.get(a_fcn_name, "No tooltip")
                        
            childDisplayFunctionTreeItem.setToolTip(0, active_tooltip_text)
            
            childDisplayFunctionTreeItem.setData(0, QtCore.Qt.UserRole, a_fcn_name) # "Child 1 custom data"
            # childDisplayFunctionTreeItem.setIcon(0, QtGui.QIcon("child_1_icon.png"))
            # childDisplayFunctionTreeItem
            # displayFunctionTreeItem.addChild(childDisplayFunctionTreeItem)
            self.treeWidget.addTopLevelItem(childDisplayFunctionTreeItem) # add top level

        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem)



## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = LauncherWidget()
    widget.show()
    sys.exit(app.exec_())
