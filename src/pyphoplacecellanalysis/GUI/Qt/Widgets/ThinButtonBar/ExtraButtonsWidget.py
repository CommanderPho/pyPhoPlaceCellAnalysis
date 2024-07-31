# ExtraButtonsWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ThinButtonBar\ExtraButtonsWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

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
    # icon = QtGui.QIcon()
    # icon.addPixmap(QtGui.QPixmap(":/png/gui/icons/view-refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    # existing_form.btnRefresh.setIcon(icon)
    # existing_form.btnRefresh.setObjectName("btnRefresh")
    # existing_form.horizontalLayout.addWidget(existing_form.btnRefresh)
    # existing_form.btnCopySelectedEpochs = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    # existing_form.btnCopySelectedEpochs.setEnabled(True)
    # icon1 = QtGui.QIcon()
    # icon1.addPixmap(QtGui.QPixmap(":/png/gui/icons/edit-copy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    # existing_form.btnCopySelectedEpochs.setIcon(icon1)
    # existing_form.btnCopySelectedEpochs.setObjectName("btnCopySelectedEpochs")
    # existing_form.horizontalLayout.addWidget(existing_form.btnCopySelectedEpochs)
    existing_form.toolButton_Printer = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Printer.setToolTip("Printer")
    existing_form.toolButton_Printer.setText("Printer")
    icon2 = QtGui.QIcon()
    icon2.addPixmap(QtGui.QPixmap(":/png/gui/icons/document-print.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Printer.setIcon(icon2)
    existing_form.toolButton_Printer.setObjectName("toolButton_Printer")
    existing_form.horizontalLayout.addWidget(existing_form.toolButton_Printer)
    spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
    existing_form.horizontalLayout.addItem(spacerItem1)
    existing_form.toolButton_Brush = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Brush.setToolTip("Brush")
    existing_form.toolButton_Brush.setText("Brush")
    icon3 = QtGui.QIcon()
    icon3.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-brush.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Brush.setIcon(icon3)
    existing_form.toolButton_Brush.setObjectName("toolButton_Brush")
    existing_form.horizontalLayout.addWidget(existing_form.toolButton_Brush)
    existing_form.toolButton_Pencil = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Pencil.setToolTip("Pencil")
    existing_form.toolButton_Pencil.setText("Pencil")
    icon4 = QtGui.QIcon()
    icon4.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-pencil.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Pencil.setIcon(icon4)
    existing_form.toolButton_Pencil.setObjectName("toolButton_Pencil")
    existing_form.horizontalLayout.addWidget(existing_form.toolButton_Pencil)
    existing_form.toolButton_Eraser = QtWidgets.QToolButton(existing_form.thinButtonBarWidget)
    existing_form.toolButton_Eraser.setToolTip("Eraser")
    existing_form.toolButton_Eraser.setText("Eraser")
    icon5 = QtGui.QIcon()
    icon5.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-rubber.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    existing_form.toolButton_Eraser.setIcon(icon5)
    existing_form.toolButton_Eraser.setObjectName("toolButton_Eraser")
    existing_form.horizontalLayout.addWidget(existing_form.toolButton_Eraser)

    buttons_list = (existing_form.toolButton_Printer, existing_form.toolButton_Brush, existing_form.toolButton_Pencil, existing_form.toolButton_Eraser)
    
    style_sheet: str = """
background-color: rgb(71, 58, 46);
border-color: rgb(207, 207, 207);
background-color: rgba(71, 65, 60, 180);
color: rgb(244, 244, 244);
border-color: rgb(0, 0, 0);
QToolTip {
    background-color: #2a2a2a;
    color: #ffffff;
    border: 1px solid #3a3a3a;
};
QStatusBar {
    background-color: #2a2a2a;
    color: #ffffff;
};
    """
    return existing_form, buttons_list


class ExtraButtonsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.


        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = ExtraButtonsWidget()
    # Set the stylesheet
    widget.setStyleSheet("""
        background-color: rgba(71, 65, 60, 180);
        color: rgb(244, 244, 244);
        border-color: rgb(0, 0, 0);

        QToolTip {
            background-color: rgb(71, 58, 46);
            color: rgb(244, 244, 244);
            border: 1px solid rgb(207, 207, 207);
        }

        QStatusBar {
            background-color: rgb(71, 58, 46);
            color: rgb(244, 244, 244);
        }
    """)

    widget.show()
    sys.exit(app.exec_())
