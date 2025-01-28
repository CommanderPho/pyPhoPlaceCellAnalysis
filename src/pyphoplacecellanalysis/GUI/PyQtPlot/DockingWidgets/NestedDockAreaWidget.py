from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaContentMixin

class NestedDockAreaWidget(DynamicDockDisplayAreaContentMixin, QtWidgets.QWidget):
    """ a custom QWidget subclass that contains a DockArea as its central view and allows adding nested dock items dynamically
    
    NOTE: the nesting doesn't quite work. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget
  
        pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget.NestedDockAreaWidget
        
    """
    sigDockClosed = QtCore.Signal(object)
    # sigDocksModified = QtCore.Signal(object)

    @property
    def area(self) -> DockArea:
        return self.ui.area

    def __init__(self, *args, **kwargs):
        # self._app = pg.mkQApp(title) # makes a new QApplication or gets the reference to an existing one.
        self.ui = PhoUIContainer()
        self.DynamicDockDisplayAreaContentMixin_on_init()
        super(NestedDockAreaWidget, self).__init__(*args, **kwargs)
        self.setup()
        self.buildUI()
        

    def setup(self):
        self.ui.area = DockArea()
        # Use self.ui.area as central widget:
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setContentsMargins(0,0,0,0)
        self.ui.layout.setVerticalSpacing(2)
        self.setLayout(self.ui.layout)
        self.ui.layout.addWidget(self.ui.area, 0, 0)
        self.DynamicDockDisplayAreaContentMixin_on_setup()
        
        
    def buildUI(self):
        self.DynamicDockDisplayAreaContentMixin_on_buildUI()
        
    
    def closeEvent(self, event):
        # Enables closing all secondary windows when this (main) window is closed.
        self.DynamicDockDisplayAreaContentMixin_on_destroy()
        self.sigDockClosed.emit(self)
        
            
            