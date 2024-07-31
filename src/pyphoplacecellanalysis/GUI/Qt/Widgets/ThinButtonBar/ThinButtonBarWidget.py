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
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons, silx_resources_rc
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import documentation_tags, metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'ThinButtonBarWidget.ui')

# extraButtons_uiFile = os.path.join(path, 'ExtraButtonsWidget.ui') # file with extra buttons

# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import Ui_ExtraButtonsWidget
# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.Uic_AUTOGEN_ExtraButtonsWidget import Ui_ExtraButtonsWidget as Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import add_buttons_to_existing_form

@metadata_attributes(short_name=None, tags=['ui', 'widget', 'button-bar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 08:30', related_items=[])
class ThinButtonBarWidget(QWidget):
    """ 
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget

    
    """
    sigCopySelections = QtCore.pyqtSignal()
    sigRefresh = QtCore.pyqtSignal()

    @property
    def label_message(self) -> str:
        """The label_message property."""
        try:
            # return self.ui.txtLineEdit.text
            return str(self.ui.txtLineEdit.text)
        except BaseException as e:
            print(f'WARN: no text box yet. err: {e}')
            return ""
    @label_message.setter
    def label_message(self, value: str):
        try:
            # self.ui.txtLineEdit.text = value
            self.ui.txtLineEdit.setText(value)
        except BaseException as e:
            print(f'WARN: no text box yet. err: {e}')
            pass
            # raise e


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.ui.buttons_list = []
        
        # _temp_ui_buttons = uic.loadUi(extraButtons_uiFile, self)
        # print(f"_temp_ui_buttons: {_temp_ui_buttons}")

        self.ui, self.ui.buttons_list = add_buttons_to_existing_form(self.ui)
        

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        # self.ui.btnUnusedButton.hide()
        self.ui.btnUnusedButton.setVisible(False)
        self.ui.btnCopySelectedEpochs.pressed.connect(self.on_copy_selections)
        self.ui.btnRefresh.pressed.connect(self.on_perform_refresh)

        # self.ui.toolButton_Printer.pressed.connect(self.on_click_print)
        
        _button_callbacks_list = (self.on_click_print, self.on_click_brush, self.on_click_pencil, self.on_click_eraser)
        assert len(_button_callbacks_list) == len(self.ui.buttons_list), f"len(_button_callbacks_list): {len(_button_callbacks_list)} != len(self.ui.buttons_list): {len(self.ui.buttons_list)}"
        for a_btn, a_fn in zip(self.ui.buttons_list, _button_callbacks_list):
            a_btn.pressed.connect(a_fn)
            
        # all_buttons = [self.ui.btnUnusedButton, self.ui.btnCopySelectedEpochs, self.ui.btnRefresh]
        # for a_btn in all_buttons:
        #     a_btn.setEnabled(False)
        #     a_btn.hide()

        # self.ui.horizontalSpacer.hide()

        # self.ui.txtLineEdit.
        # currentTextChanged.connect(self.on_jump_combo_series_changed)
        pass

    @pyqtExceptionPrintingSlot()
    def on_copy_selections(self):
        """ 
        """
        print(f'on_copy_selections()')
        self.sigCopySelections.emit()
        
    @pyqtExceptionPrintingSlot()
    def on_perform_refresh(self):
        """ 
        """
        print(f'on_perform_refresh()')
        self.sigRefresh.emit()
        

    @pyqtExceptionPrintingSlot()
    def on_click_print(self):
        """ 
        """
        print(f'on_click_print()')


    @pyqtExceptionPrintingSlot()
    def on_click_brush(self):
        """ 
        """
        print(f'on_click_brush()')
        
    @pyqtExceptionPrintingSlot()
    def on_click_pencil(self):
        """ 
        """
        print(f'on_click_pencil()')
        
    @pyqtExceptionPrintingSlot()
    def on_click_eraser(self):
        """ 
        """
        print(f'on_click_eraser()')
        
        



## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = ThinButtonBarWidget()
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
    
    widget.label_message = "TEST TEXT LINE!"
    # widget.ui.txtLineEdit.text = "TEST TEXT LINE!"
    widget.show()
    print(f"widget.label_message: {widget.label_message}")
    sys.exit(app.exec_())
