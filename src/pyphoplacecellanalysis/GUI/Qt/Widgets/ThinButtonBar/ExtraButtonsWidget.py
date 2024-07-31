# ExtraButtonsWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ThinButtonBar\ExtraButtonsWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
# from .Uic_AUTOGEN_ExtraButtonsWidget import Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.Uic_AUTOGEN_ExtraButtonsWidget import Ui_ExtraButtonsWidget as Ui_Form
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons, silx_resources_rc


def add_buttons_to_existing_form(existing_form):
    """
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ExtraButtonsWidget import add_buttons_to_existing_form
    
    
    """
    # self = existing_form
    # _extra_buttons_ui = Ui_Form()
    existing_form.horizontalLayout_ButtonContainer = QtWidgets.QHBoxLayout()
    existing_form.horizontalLayout_ButtonContainer.setContentsMargins(0, -1, -1, -1)
    existing_form.horizontalLayout_ButtonContainer.setSpacing(2)
    existing_form.horizontalLayout_ButtonContainer.setObjectName("horizontalLayout_ButtonContainer")
    
    existing_form.btnRefresh = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.btnRefresh.setToolTip("Refresh")
    existing_form.btnRefresh.setStatusTip("Refresh")
    existing_form.btnRefresh.setText("Refresh")
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(":/png/gui/icons/view-refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.btnRefresh.setIcon(icon)
    existing_form.btnRefresh.setAutoRaise(True)
    existing_form.btnRefresh.setObjectName("btnRefresh")
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.btnRefresh)
    existing_form.toolButton_Clipboard = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Clipboard.setToolTip("Clipboard")
    existing_form.toolButton_Clipboard.setText("Clipboard")
    iconClipboard = QtGui.QIcon()
    iconClipboard.addPixmap(QtGui.QPixmap(":/png/gui/icons/clipboard.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Clipboard.setIcon(iconClipboard)
    existing_form.toolButton_Clipboard.setObjectName("toolButton_Clipboard")
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.toolButton_Clipboard)
    existing_form.btnCopySelectedEpochs = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.btnCopySelectedEpochs.setEnabled(True)
    existing_form.btnCopySelectedEpochs.setToolTip("Copy Selections")
    existing_form.btnCopySelectedEpochs.setStatusTip("CopySelections")
    existing_form.btnCopySelectedEpochs.setText("Copy Selections")
    icon1 = QtGui.QIcon()
    icon1.addPixmap(QtGui.QPixmap(":/png/gui/icons/edit-copy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.btnCopySelectedEpochs.setIcon(icon1)
    existing_form.btnCopySelectedEpochs.setAutoRaise(True)
    existing_form.btnCopySelectedEpochs.setObjectName("btnCopySelectedEpochs")
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.btnCopySelectedEpochs)
    existing_form.toolButton_Printer = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Printer.setToolTip("Printer")
    existing_form.toolButton_Printer.setText("Printer")
    icon2 = QtGui.QIcon()
    icon2.addPixmap(QtGui.QPixmap(":/png/gui/icons/document-print.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Printer.setIcon(icon2)
    existing_form.toolButton_Printer.setAutoRaise(True)
    existing_form.toolButton_Printer.setObjectName("toolButton_Printer")
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.toolButton_Printer)
    
    existing_form.horizontalSpacer_fixedSmall = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
    existing_form.horizontalLayout_ButtonContainer.addItem(existing_form.horizontalSpacer_fixedSmall)
    
    existing_form.toolButton_Brush = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Brush.setToolTip("Brush")
    existing_form.toolButton_Brush.setText("Brush")
    icon3 = QtGui.QIcon()
    icon3.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-brush.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Brush.setIcon(icon3)
    existing_form.toolButton_Brush.setCheckable(True)
    existing_form.toolButton_Brush.setAutoExclusive(True)
    existing_form.toolButton_Brush.setObjectName("toolButton_Brush")
    existing_form.buttonGroup_ActiveTool = QtWidgets.QButtonGroup(existing_form)
    existing_form.buttonGroup_ActiveTool.setObjectName("buttonGroup_ActiveTool")
    existing_form.buttonGroup_ActiveTool.addButton(existing_form.toolButton_Brush)
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.toolButton_Brush)
    existing_form.toolButton_Pencil = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Pencil.setToolTip("Pencil")
    existing_form.toolButton_Pencil.setText("Pencil")
    icon4 = QtGui.QIcon()
    icon4.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-pencil.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Pencil.setIcon(icon4)
    existing_form.toolButton_Pencil.setCheckable(True)
    existing_form.toolButton_Pencil.setAutoExclusive(True)
    existing_form.toolButton_Pencil.setObjectName("toolButton_Pencil")
    existing_form.buttonGroup_ActiveTool.addButton(existing_form.toolButton_Pencil)
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.toolButton_Pencil)
    existing_form.toolButton_Eraser = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Eraser.setToolTip("Eraser")
    existing_form.toolButton_Eraser.setText("Eraser")
    icon5 = QtGui.QIcon()
    icon5.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-rubber.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Eraser.setIcon(icon5)
    existing_form.toolButton_Eraser.setCheckable(True)
    existing_form.toolButton_Eraser.setAutoExclusive(True)
    existing_form.toolButton_Eraser.setObjectName("toolButton_Eraser")
    existing_form.buttonGroup_ActiveTool.addButton(existing_form.toolButton_Eraser)
    existing_form.horizontalLayout_ButtonContainer.addWidget(existing_form.toolButton_Eraser)
    
    ## add the buttonContainer layout to the existing layout:
    existing_form.horizontalLayout.addLayout(existing_form.horizontalLayout_ButtonContainer)
    existing_form.horizontalLayout.setStretch(0, 1)

    buttons_list = (existing_form.btnRefresh, existing_form.toolButton_Clipboard, existing_form.btnCopySelectedEpochs, existing_form.toolButton_Printer, existing_form.toolButton_Brush, existing_form.toolButton_Pencil, existing_form.toolButton_Eraser)
    # buttons_dict = {a_btn.objectName():a_btn for a_btn in buttons_list}
    buttons_dict = {str(a_btn.text()):a_btn for a_btn in buttons_list}
    
    def _perform_update_tool_spacer_visibility(existing_form) -> bool:
        """ 
        captures: existing_form, existing_form
        """
        _tool_buttons = (existing_form.toolButton_Brush, existing_form.toolButton_Pencil, existing_form.toolButton_Eraser)
        has_any_tools: bool = np.any([v.isVisible() for v in _tool_buttons])
        if has_any_tools:
            _spacer_args = (10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        else:
            _spacer_args = (0, 0, QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # QSizePolicy.Policy
        existing_form.horizontalSpacer_fixedSmall.changeSize(*_spacer_args)
        return has_any_tools
    
    existing_form.perform_update_tool_spacer_visibility = _perform_update_tool_spacer_visibility
    existing_form.perform_update_tool_spacer_visibility(existing_form)

    return existing_form, buttons_list, buttons_dict


