# MainWindowWrapper
from qtpy import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.GlobalConnectionManager import GlobalConnectionManager, GlobalConnectionManagerAccessingMixin
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.Resources import ActionIcons
from pyphoplacecellanalysis.Resources import GuiResources

""" PhoMainAppWindowBase

"""


class PhoMainAppWindowBase(GlobalConnectionManagerAccessingMixin, QtWidgets.QMainWindow):
    """ a custom QMainWindow subclass that all custom main windows should inherit from .
    
    Usage:
        from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMainAppWindowBase import PhoMainAppWindowBase
    
    """
    @property
    def app(self):
        """The app property."""
        return self._app
    
    
    def __init__(self, *args, **kwargs):
        self._app = pg.mkQApp() # makes a new QApplication or gets the reference to an existing one.
        self.ui = PhoUIContainer()
        super(PhoMainAppWindowBase, self).__init__(*args, **kwargs)
            
        self.GlobalConnectionManagerAccessingMixin_on_init(owning_application=self.app) # initializes self._connection_man
        self.setup()
        self.buildUI()
        

    def setup(self):
        pass
        
    
    def buildUI(self):
        pass
