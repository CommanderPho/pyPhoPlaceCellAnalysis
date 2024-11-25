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

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'ThinButtonBarWidget.ui')

# extraButtons_uiFile = os.path.join(path, 'ExtraButtonsWidget.ui') # file with extra buttons

# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import Ui_ExtraButtonsWidget
# from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.Uic_AUTOGEN_ExtraButtonsWidget import Ui_ExtraButtonsWidget as Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import add_buttons_to_existing_form

# ==================================================================================================================== #
# Programmatic Button Helpers                                                                                          #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['button', 'ui', 'programmatic', 'toolbar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-31 09:32', related_items=[])
def build_button(icon_path: str=':/png/gui/icons/crosshair.png', name: str="Refresh", parent=None) -> QtWidgets.QToolButton:
    """ builds a simple toolbar tool button programmatically from the icon_path and the name. """
    if parent is not None:
        new_btn = QtWidgets.QToolButton(parent)
    else:
        new_btn = QtWidgets.QToolButton()
        
    new_btn.setToolTip(name)
    new_btn.setStatusTip(name)
    new_btn.setText(name)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    new_btn.setIcon(icon)
    new_btn.setAutoRaise(True)
    new_btn.setObjectName(f"toolButton_{name}")
    return new_btn


@define(slots=True, eq=False)
class ProgrammaticButtonConfig:
    """ holds the basic info needed to build a tool button.
    """
    icon_path: str = field()
    name: str = field()
    callback: Optional[Callable] = field(default=None)

    @property
    def lower_name(self) -> str:
        """The lower_name property."""
        lower_name: str = self.name.lower()
        lower_name = lower_name.replace(' ', '_')
        return lower_name
    
    @property
    def fn_name(self) -> str:
        """The fn_name property."""
        fn_name: str = f'on_click_{self.lower_name}'
        return fn_name

    def to_dict(self) -> Dict:
        return asdict(self, filter=(lambda an_attr, attr_value: an_attr.name in ['icon_path', 'name']))


def _create_on_click_fn(a_fn_name, a_fn_callback=None):
    def an_on_click_fn(self, *args, **kwargs):
        """ captures: a_fn_name """
        print(f'{a_fn_name}(*args: {args or []}, **kwargs: {kwargs or {}})')
        self.label_message = f'{a_fn_name}()'
        if a_fn_callback is not None:
            a_fn_callback(self, *args, **kwargs)
            
    return an_on_click_fn



def build_programmatic_buttons(global_thin_button_bar_widget, button_config_dict: Union[Dict, List], clear_all_existing:bool=False):
    """
    Usage:
        
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import build_programmatic_buttons, ProgrammaticButtonConfig
        
        global_thin_button_bar_widget: ThinButtonBarWidget = paginated_multi_decoder_decoded_epochs_window.global_thin_button_bar_widget

        button_config_list = [
            dict(icon_path=':/png/gui/icons/document-open.png', name="OpenFile"),
            dict(icon_path=':/png/gui/icons/document-save.png', name="SaveFile"),
            dict(icon_path=':/png/gui/icons/crosshair.png', name="Crosshairs"),
            dict(icon_path=':/png/gui/icons/crop.png', name="Crop"),
            dict(icon_path=':/png/gui/icons/selected.png', name="Selections"),
            dict(icon_path=':/png/gui/icons/view-raw.png', name="CopyAsArray"),
            #  dict(icon_path=':/png/gui/icons/crop.png', name="Crop"),
            #  dict(icon_path=':/png/gui/icons/crop.png', name="Crop"),
        ]
        button_config_dict = {v['name']:v for v in button_config_list}

        new_buttons_config_dict, new_buttons_dict = build_programmatic_buttons(global_thin_button_bar_widget, button_config_dict=button_config_dict, clear_all_existing=True)


    """
    if isinstance(button_config_dict, (list, tuple)):
        button_config_dict = {v['name']:v for v in button_config_dict} # try converting to dict assuming a list
    
    if clear_all_existing:
        global_thin_button_bar_widget.clear_all_buttons()
        
    new_buttons_config_dict = {} # will always start fresh
    # new_buttons_list = []
    new_buttons_dict = global_thin_button_bar_widget.ui.buttons_dict
    

    for k, a_btn_config_dict in button_config_dict.items():    
        a_btn_config: ProgrammaticButtonConfig = ProgrammaticButtonConfig(**a_btn_config_dict)
        a_fn_name: str = a_btn_config.fn_name
        a_fn_callback = a_btn_config.callback
        
        new_buttons_config_dict[k] = a_btn_config

        extant_btn = global_thin_button_bar_widget.ui.buttons_dict.get(k, None)
        if extant_btn is None:
            ## add the widget:
            a_btn = build_button(**a_btn_config.to_dict(), parent=global_thin_button_bar_widget)
            global_thin_button_bar_widget.horizontalLayout_ButtonContainer.addWidget(a_btn)
            new_buttons_dict[k] = a_btn
        else:
            print(f'button: "{k}" already exists!')
            a_btn = extant_btn
            a_btn.pressed.disconnect()

        ## add callback function to `global_thin_button_bar_widget`
        on_click_fn = _create_on_click_fn(a_fn_name, a_fn_callback=a_fn_callback)
        ## Bind the method to the instance
        bound_method = MethodType(on_click_fn, global_thin_button_bar_widget) # add to type instead of instance
        # print(f"bound_method: {bound_method}, type(bound_method): {type(bound_method)}")
        setattr(global_thin_button_bar_widget, a_fn_name, bound_method)
        a_btn.pressed.connect(getattr(global_thin_button_bar_widget, a_fn_name)) ## connect button
        
        
    ## update the buttons properties:
    # global_thin_button_bar_widget.ui.buttons_list = new_buttons_list
    global_thin_button_bar_widget.ui.buttons_dict = new_buttons_dict

    if (len(global_thin_button_bar_widget.ui.buttons_list) != len(global_thin_button_bar_widget.ui.buttons_dict)):
        # rebuild the list from the dict:
        global_thin_button_bar_widget.ui.buttons_list = list(global_thin_button_bar_widget.ui.buttons_dict.values())

    return new_buttons_config_dict, new_buttons_dict


def set_trimmed_text(line_edit, full_text: str): # , max_length: int
    """
    line_edit = QLineEdit()
    full_text = "This is a very long text that needs to be trimmed to fit in the QLineEdit."
    set_trimmed_text(line_edit, full_text, 30)

    """
    # trimmed_text = full_text if len(full_text) <= max_length else full_text[:max_length] + '...'
    # line_edit.setText(trimmed_text)
    # line_edit.setToolTip(full_text)
    line_edit.setToolTip(full_text)
    
    fm = line_edit.fontMetrics()
    # available_width = line_edit.width() - 2  # Subtracting a small margin
    available_width = int(round(line_edit.width() * 0.8))  - 2  # Subtracting a small margin
    elided_text = fm.elidedText(full_text, Qt.ElideRight, available_width)
    line_edit.setText(elided_text)
    
    
    

# ==================================================================================================================== #
# Main ThinButtonBarWidget widget                                                                                      #
# ==================================================================================================================== #
@metadata_attributes(short_name=None, tags=['ui', 'widget', 'button-bar'], input_requires=[], output_provides=[], uses=['add_buttons_to_existing_form'], used_by=[], creation_date='2024-03-01 08:30', related_items=[])
class ThinButtonBarWidget(QWidget):
    """ 
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget

    Uses `add_buttons_to_existing_form` from `ExtraButtonsWidget` to add buttons
    
    """
    sigCopySelections = QtCore.pyqtSignal()
    sigLoadSelections = QtCore.pyqtSignal()
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
            # self.ui.txtLineEdit.setText(value)            
            # set_trimmed_text(line_edit=self.ui.txtLineEdit, full_text=value, 30)
            set_trimmed_text(line_edit=self.ui.txtLineEdit, full_text=str(value))
            
        except BaseException as e:
            print(f'WARN: no text box yet. err: {e}')
            pass
            # raise e


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.ui.buttons_list = []
        self.ui.buttons_dict = {}
        self.ui, self.ui.buttons_list, self.ui.buttons_dict = add_buttons_to_existing_form(self.ui)
        print(f'self.ui.buttons_dict.keys(): {list(self.ui.buttons_dict.keys())}') # self.ui.buttons_dict.keys(): ['Refresh', 'Clipboard', 'Copy Selections', 'Printer', 'Brush', 'Pencil', 'Eraser']
        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.ui.btnUnusedButton.setVisible(False)
        self.ui.btnLoadUserSelectedFromAnnotations.setVisible(True)
        
        _button_callbacks_list = (self.on_perform_refresh, self.on_click_clipboard, self.on_copy_selections, self.on_click_load_selections, self.on_click_print, self.on_click_brush, self.on_click_pencil, self.on_click_eraser)
        assert len(_button_callbacks_list) == len(self.ui.buttons_list), f"len(_button_callbacks_list): {len(_button_callbacks_list)} != len(self.ui.buttons_list): {len(self.ui.buttons_list)}"

        # hidden_buttons_list = ['Refresh', 'Clipboard', 'Copy Selections', 'Printer', 'Brush', 'Pencil', 'Eraser']
        hidden_buttons_list = ['Clipboard', 'Brush', 'Pencil', 'Eraser']

        for a_btn_name, a_btn in self.ui.buttons_dict.items():
            if a_btn_name in hidden_buttons_list:
                a_btn.setVisible(False)

        is_spacer_visible: bool = self.perform_update_tool_spacer_visibility(self.ui)
        print(f'is_spacer_visible: {is_spacer_visible}')
        
        ## connect buttons:
        for a_btn, a_fn in zip(self.ui.buttons_list, _button_callbacks_list):
            a_btn.pressed.connect(a_fn)
            

    @classmethod
    def clear_layout(cls, a_layout):
        """ works to remove the programmatically added buttons """
        while a_layout.count():
            item = a_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                a_layout.removeItem(item)
            


    def clear_all_buttons(self):
        """ works to remove the programmatically added buttons """
        self.clear_layout(a_layout=self.horizontalLayout_ButtonContainer)
        ## remove stored buttons:
        self.ui.buttons_list = []
        self.ui.buttons_dict = {}
        
            
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
    def on_click_clipboard(self):
        """ 
        """
        print(f'on_click_clipboard()')

    @pyqtExceptionPrintingSlot()
    def on_copy_selections(self):
        """ 
        """
        print(f'on_copy_selections()')
        self.sigCopySelections.emit()
        
    @pyqtExceptionPrintingSlot()
    def on_click_load_selections(self):
        """ 
        """
        print(f'on_click_load_selections()')
        self.sigLoadSelections.emit()
        

    # Tools ______________________________________________________________________________________________________________ #
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
    
    # widget.label_message = "TEST TEXT LINE!"
    # widget.ui.txtLineEdit.text = "TEST TEXT LINE!"
    widget.show()
    print(f"widget.label_message: {widget.label_message}")
    sys.exit(app.exec_())
