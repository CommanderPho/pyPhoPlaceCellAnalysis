# LoggingOutputWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\Testing\LoggingOutputWidget\LoggingOutputWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.connections_container import ConnectionsContainer

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'LoggingOutputWidget.ui')

@metadata_attributes(short_name=None, tags=['logging', 'window', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=['Spike3DRasterBottomPlaybackControlBar'], creation_date='2025-01-06 12:04', related_items=[])
class LoggingOutputWidget(QWidget):
    """ A standalone window that has a single large textedit to contain log data.
    
    from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.LoggingOutputWidget.LoggingOutputWidget import LoggingOutputWidget
    
    """
    def __init__(self, parent=None):
        # super().__init__(parent=parent) # Call the inherited classes __init__ method
        QWidget.__init__(self, parent=parent)
        
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.ui.connections = ConnectionsContainer() 
        self.params = VisualizationParameters(name='LoggingOutputWidget', debug_print=False)
        self.setWindowTitle('Logging Output Window')


    @pyqtExceptionPrintingSlot(object)
    def on_log_updated(self, logger):
        if self.params.debug_print:
            print(f'LoggingOutputWidget.on_log_updated(logger: {logger})')
        # logger: LoggingBaseClass
        target_text: str = logger.get_flattened_log_text(flattening_delimiter='\n', limit_to_n_most_recent=None)
        self.ui.logTextEdit.setText(target_text)
        

    @pyqtExceptionPrintingSlot()
    def on_log_update_finished(self):
        if self.params.debug_print:
            print(f'LoggingOutputWidget.on_log_update_finished()')
        # logger: LoggingBaseClass
        # target_text: str = logger.get_flattened_log_text(flattening_delimiter='\n', limit_to_n_most_recent=None)
        # self.ui.logTextEdit.setText(target_text)
        pass
    
## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = LoggingOutputWidget()
    widget.show()
    sys.exit(app.exec_())
