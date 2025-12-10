# ThinButtonBarWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ThinButtonBar\ThinButtonBarWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from types import MethodType
from attrs import asdict, astuple, define, field, Factory

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons, silx_resources_rc
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import documentation_tags, metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


# extraButtons_uiFile = os.path.join(path, 'ExtraButtonsWidget.ui') # file with extra buttons

# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import Ui_ExtraButtonsWidget
# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.Uic_AUTOGEN_ExtraButtonsWidget import Ui_ExtraButtonsWidget as Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import add_buttons_to_existing_form
from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget, ProgrammaticButtonConfig, build_programmatic_buttons, _create_on_click_fn, build_button, set_trimmed_text


# Add this new class or function to ThinButtonBarWidget.py

# def create_custom_thin_button_bar_widget(parent=None, button_names: List[str] = ['reload', 'copy-all', 'apply', 'config']) -> ThinButtonBarWidget:
#     """
#     Creates a new ThinButtonBarWidget with custom buttons.
    
#     Usage:
#         custom_widget = create_custom_thin_button_bar_widget(parent=some_parent, button_names=['reload', 'copy-all', 'apply', 'config'])
#     """
#     # Map button names to their icon paths
#     button_icon_map = {
#         'reload': ':/png/gui/icons/view-refresh.png',
#         'copy-all': ':/png/gui/icons/edit-copy.png',
#         'apply': ':/png/gui/icons/document-save.png',  # or ':/png/gui/icons/dialog-ok-apply.png' if available
#         'config': ':/png/gui/icons/preferences-system.png',  # or ':/png/gui/icons/configure.png' if available
#     }
    
#     # Create the widget
#     widget = ThinButtonBarWidget(parent=parent)
    
#     # Clear existing buttons
#     widget.clear_all_buttons()
    
#     # Build button configuration
#     button_config_list = [
#         dict(icon_path=button_icon_map.get(name, ':/png/gui/icons/crosshair.png'), name=name)
#         for name in button_names
#     ]
#     button_config_dict = {v['name']: v for v in button_config_list}
    
#     # Build the programmatic buttons
#     new_buttons_config_dict, new_buttons_dict = build_programmatic_buttons(
#         widget, 
#         button_config_dict=button_config_dict, 
#         clear_all_existing=True
#     )
    
#     return widget


# Alternative: Create a subclass with the buttons pre-configured

class IntervalConfigListExtraButtonsWidget(ThinButtonBarWidget):
    """ 
    A ThinButtonBarWidget with custom buttons: ['reload', 'copy-all', 'apply', 'config']
    
    Usage:
		from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.IntervalConfigListExtraButtonsWidget import IntervalConfigListExtraButtonsWidget

        custom_widget = IntervalConfigListExtraButtonsWidget(parent=some_parent)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        # Clear the default buttons
        self.clear_all_buttons()
        
        # Define button configurations
        button_config_list = [
            dict(icon_path=':/png/gui/icons/view-refresh.png', name='reload'),
            dict(icon_path=':/png/gui/icons/edit-copy.png', name='copy-all'),
            dict(icon_path=':/png/gui/icons/document-save.png', name='apply'),
            dict(icon_path=':/png/gui/icons/preferences-system.png', name='config'),
        ]
        button_config_dict = {v['name']: v for v in button_config_list}
        
        # Build the programmatic buttons
        self.new_buttons_config_dict, self.new_buttons_dict = build_programmatic_buttons(
            self, 
            button_config_dict=button_config_dict, 
            clear_all_existing=True
        )
        
        # Connect custom callbacks if needed
        self._connect_custom_callbacks()
    
    def _connect_custom_callbacks(self):
        """Override this method to connect custom button callbacks."""
        # Example: connect reload button
        if 'reload' in self.new_buttons_dict:
            self.new_buttons_dict['reload'].pressed.disconnect()
            self.new_buttons_dict['reload'].pressed.connect(self.on_click_reload)
        
        if 'copy-all' in self.new_buttons_dict:
            self.new_buttons_dict['copy-all'].pressed.disconnect()
            self.new_buttons_dict['copy-all'].pressed.connect(self.on_click_copy_all)
        
        if 'apply' in self.new_buttons_dict:
            self.new_buttons_dict['apply'].pressed.disconnect()
            self.new_buttons_dict['apply'].pressed.connect(self.on_click_apply)
        
        if 'config' in self.new_buttons_dict:
            self.new_buttons_dict['config'].pressed.disconnect()
            self.new_buttons_dict['config'].pressed.connect(self.on_click_config)
    
    @pyqtExceptionPrintingSlot()
    def on_click_reload(self):
        """Callback for reload button."""
        print(f'on_click_reload()')
        self.label_message = 'Reload clicked'
    
    @pyqtExceptionPrintingSlot()
    def on_click_copy_all(self):
        """Callback for copy-all button."""
        print(f'on_click_copy_all()')
        self.label_message = 'Copy All clicked'
    
    @pyqtExceptionPrintingSlot()
    def on_click_apply(self):
        """Callback for apply button."""
        print(f'on_click_apply()')
        self.label_message = 'Apply clicked'
    
    @pyqtExceptionPrintingSlot()
    def on_click_config(self):
        """Callback for config button."""
        print(f'on_click_config()')
        self.label_message = 'Config clicked'




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = IntervalConfigListExtraButtonsWidget()
    # Set the stylesheet
    widget.setStyleSheet("""
/*
background-color: rgb(71, 58, 46);
border-color: rgb(207, 207, 207);
background-color: rgba(71, 65, 60, 180);
color: rgb(244, 244, 244);
border-color: rgb(0, 0, 0);
*/
QWidget {
    background-color: rgb(71, 58, 46);
    color: rgb(244, 244, 244);
}
QToolTip {
    background-color: rgb(71, 58, 46);
    color: rgb(244, 244, 244);
    border: 1px solid rgb(207, 207, 207);
}
QStatusBar {
    background-color: rgb(71, 58, 46);
    color: rgb(244, 244, 244);
}
        */
    """)
    
    # widget.label_message = "TEST TEXT LINE!"
    # widget.ui.txtLineEdit.text = "TEST TEXT LINE!"
    widget.show()
    print(f"widget.label_message: {widget.label_message}")
    sys.exit(app.exec_())
