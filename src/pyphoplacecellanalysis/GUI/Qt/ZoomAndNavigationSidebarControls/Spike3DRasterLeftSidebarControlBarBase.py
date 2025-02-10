# Spike3DRasterLeftSidebarControlBarBase.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\ZoomAndNavigationSidebarControls\Spike3DRasterLeftSidebarControlBarBase.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'Spike3DRasterLeftSidebarControlBarBase.ui')

class Spike3DRasterLeftSidebarControlBarBase(QWidget):
    
    animation_time_step_changed = pyqtSignal(float) # returns bool indicating whether is_playing
    temporal_zoom_factor_changed = pyqtSignal(float)
    render_window_duration_changed = pyqtSignal(float)
        
    crosshair_trace_toggled = pyqtSignal(bool)
    

    # @property
    # def lblCrosshairTraceValue(self):
    #     """The lblCrosshairTraceValue property."""
    #     return self.ui.lblCrosshairTraceValue


    @property
    def crosshair_trace_time(self) -> float:
        """The crosshair_trace_time property."""
        return self._crosshair_trace_time
    @crosshair_trace_time.setter
    def crosshair_trace_time(self, value: float):
        if value is not None:
            self.ui.lblCrosshairTraceValue.setText(f"{value}")
            self.ui.lblCrosshairTraceStaticLabel.setVisible(True)
            self.ui.lblCrosshairTraceValue.setVisible(True)
        else:
            ## Hide it
            self.ui.lblCrosshairTraceStaticLabel.setVisible(False)
            self.ui.lblCrosshairTraceValue.setVisible(False)


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = Spike3DRasterLeftSidebarControlBarBase()
    widget.show()
    sys.exit(app.exec_())
