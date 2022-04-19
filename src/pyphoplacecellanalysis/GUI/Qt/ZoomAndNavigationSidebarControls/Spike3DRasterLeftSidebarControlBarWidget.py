import numpy as np

from qtpy import QtCore, QtWidgets
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarBase import Ui_leftSideToolbarWidget # Generated file from .ui



class Spike3DRasterLeftSidebarControlBar(QtWidgets.QWidget):
    """ A controls bar with buttons loaded from a Qt .ui file. """
    
    # TODO: add signals here:
    
    
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_leftSideToolbarWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        pass
                
    def __str__(self):
         return
     
     
     
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterLeftSidebarControlBar()
    # testWidget.show()
    sys.exit(app.exec_())
