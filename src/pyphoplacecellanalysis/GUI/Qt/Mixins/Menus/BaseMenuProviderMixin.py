from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper


class BaseMenuCommand:
    """
    An abstract base command to be executed from a Menu item
    """
    def __init__(self) -> None:
        pass

    @property
    def is_visible(self):
        return True
    
    @property
    def is_enabled(self):
        return True
        
    def execute(self, filename: str) -> None:
        """ Implicitly captures spike_raster_window """
        raise NotImplementedError # implementors must override        
    
    def __call__(self, *args, **kwds):
        return self.execute('')

    

def BaseMenuProviderMixin(object):
    """ a mixin class that provides one ore more QActions and QMenu items
    
        A best practice is to create actions as children of the window in which you’re going to use them.
    
    Implementors Require:
        self._root_window
    
    """
    
    @property
    def root_window(self):
        """The global_window property."""
        return self._root_window
        
    @property
    def root_menu_bar(self):
        """The root_menu_bar property."""
        return self.root_window.menuBar()
    
    # @property
    # def BaseMenuProviderMixin_actionsDict(self):
    #     raise NotImplementedError # implementor must override with the dictionary name
    #     return self.root_window.ui.menuDebugActionsDict
    # @BaseMenuProviderMixin_actionsDict.setter
    # def BaseMenuProviderMixin_actionsDict(self, value):
    #     raise NotImplementedError # implementor must override with the dictionary name
    #     self.root_window.ui.menuDebugActionsDict = value
    
    @QtCore.pyqtSlot()
    def BaseMenuProviderMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        # Assumes that self is a QWidget subclass:
        self._root_window = PhoMenuHelper.try_get_menu_window(self)
        # Define dictionary:
        # self.root_window.ui.menuDebugActionsDict = {}

    
    @QtCore.pyqtSlot()
    def BaseMenuProviderMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    
    def _BaseMenuProviderMixin_build_actions(self):
        """ build QActions """
        ## Add the dynamic menu entries:
        pass
    
    
    def _BaseMenuProviderMixin_build_menus(self):
        """ build QMenus """
        # an_action_key, active_debug_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Debug", name='actionMenuDebug', parent_menu=self.root_menu_bar, menu_actions_dict=self.root_window.ui.menuDebugActionsDict)
        
        # an_action_key, active_drivers_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivers", name='actionMenuDebugMenuActiveDrivers', parent_menu=active_debug_menu, menu_actions_dict=self.root_window.ui.menuDebugActionsDict)
        # # active_drivers_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivers']
        # an_action_key, active_drivables_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivables", name='actionMenuDebugMenuActiveDrivables', parent_menu=active_debug_menu, menu_actions_dict=self.root_window.ui.menuDebugActionsDict)
        # # active_drivables_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivables']
        # an_action_key, active_connections_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Connections", name='actionMenuDebugMenuActiveConnections', parent_menu=active_debug_menu, menu_actions_dict=self.root_window.ui.menuDebugActionsDict)
        # active_connections_menu = self.root_window.ui['actionMenuDebugMenuActiveConnections']
        pass
    

    @QtCore.pyqtSlot()
    def BaseMenuProviderMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self._BaseMenuProviderMixin_build_actions()
        self._BaseMenuProviderMixin_build_menus()
    

    @QtCore.pyqtSlot()
    def BaseMenuProviderMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass


    @QtCore.pyqtSlot()
    def BaseMenuProviderMixin_on_menus_update(self):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass

