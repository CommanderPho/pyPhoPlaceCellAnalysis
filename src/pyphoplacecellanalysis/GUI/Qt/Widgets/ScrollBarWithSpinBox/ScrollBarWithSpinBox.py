# ScrollBarWithSpinBox.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ScrollBarWithSpinBox\ScrollBarWithSpinBox.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from typing import Optional, Dict, List, Tuple
import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.External.pyqtgraph.widgets.SpinBox import SpinBox

## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'ScrollBarWithSpinBox.ui')

class ScrollBarWithSpinBox(QWidget):
    """ 
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox
    
    Can be safely updated programmatically by calling .setValue
    
    _a_ScrollBarWithSpinBox.setValue(45)

    """
    sigValueChanged = pg.QtCore.Signal(object)  # (self)

    def __init__(self, parent=None, val: int = 0, disable_emit_changed:bool=False):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.val = val
        self.disable_emit_changed = disable_emit_changed
        self.initUI()
        self.show() # Show the GUI

    
    def initUI(self):
        self.spinBox.setOpts(int=True, step=1, minStep=1, wrapping=False)
        self.spinBox.valueChanged.connect(self.updateSlider)
        self.slider.valueChanged.connect(self.updateSpinBox)
        self.disable_emit_changed = False
        self.val = 0

    @pyqtExceptionPrintingSlot(int, int)
    def update_range(self, low:int, high:int):
        self.disable_emit_changed = True
        self.spinBox.setRange(low, high)
        self.slider.setMinimum(low)
        self.slider.setMaximum(high)
        self.disable_emit_changed = False

    @pyqtExceptionPrintingSlot(int)
    def updateSpinBox(self, value):
        self.val = int(value)
        self.label.setText(str(value))
        self.spinBox.blockSignals(True)
        self.spinBox.setValue(value)
        self.spinBox.blockSignals(False)
        self.emitChanged()

    @pyqtExceptionPrintingSlot(int)
    def updateSlider(self, value):
        self.val = int(value)
        self.label.setText(str(value))
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)
        self.emitChanged()
        
    @pyqtExceptionPrintingSlot(int, bool)
    def setValue(self, value, emit_changed:bool=True):
        """ can be safely called programmatically. 
        """
        self.disable_emit_changed = True
        self.updateSpinBox(value)
        self.updateSlider(value)
        self.disable_emit_changed = False
        if emit_changed:
            self.emitChanged()


    def emitChanged(self):
        assert self.val is not None
        if (not self.disable_emit_changed):
            # self.lastValEmitted = self.val
            # self.valueChanged.emit(float(self.val))
            # print(f'emitChanged(): self.val: {self.val}')
            self.sigValueChanged.emit(int(self.val))
        else:
            print('WARN: self.disable_emit_changed = True')


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = ScrollBarWithSpinBox()
    widget.show()
    sys.exit(app.exec_())
