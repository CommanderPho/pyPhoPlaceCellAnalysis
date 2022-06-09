from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin

def DebugMenuProviderMixin(BaseMenuProviderMixin):

    @property
    def DebugMenuProviderMixin_actionsDict(self):
        return self.root_window.ui.menuDebugActionsDict
    @DebugMenuProviderMixin_actionsDict.setter
    def DebugMenuProviderMixin_actionsDict(self, value):
        self.root_window.ui.menuDebugActionsDict = value
        
    # @property
    # def connection_man(self):
    # 	"""The connection_man property."""
    # 	return self._connection_man

    
    @QtCore.pyqtSlot()
    def DebugMenuProviderMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        BaseMenuProviderMixin.BaseMenuProviderMixin_on_init(self)
        # Define dictionary:
        self.root_window.ui.menuDebugActionsDict = {}
    
    @QtCore.pyqtSlot()
    def DebugMenuProviderMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    
    def _DebugMenuProviderMixin_build_actions(self):
        """ build QActions """
        ## Add the dynamic menu entries:
        connection_man = self.connection_man
        
        ## Update Drivers Menu:
        curr_drivers_items = list(connection_man.registered_available_drivers.keys())
        for a_driver_key in curr_drivers_items:
            self.root_window.ui.active_drivers_menu.addAction(a_driver_key)
        self.root_window.ui.active_drivers_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivers.get(action.text(), f'Driver KeyNotFound: {action.text()}')))

        ## Update Drivable Menu:
        curr_drivable_items = list(connection_man.registered_available_drivables.keys())
        for a_driveable_key in curr_drivable_items:
            self.root_window.ui.active_drivables_menu.addAction(a_driveable_key)
        self.root_window.ui.active_drivables_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivables.get(action.text(), f'Drivable KeyNotFound: {action.text()}')))
  
        ## Update Connections Menu:
        curr_connections_items = list(connection_man.active_connections.keys())
        for a_connection_key in curr_connections_items:
            self.root_window.ui.active_connections_menu.addAction(a_connection_key)
        self.root_window.ui.active_connections_menu.triggered.connect(lambda action: print(connection_man.active_connections.get(action.text(), f'Connection KeyNotFound: {action.text()}')))
    
    
    def _DebugMenuProviderMixin_build_menus(self):
        """ build QMenus """
        an_action_key, self.root_window.ui.active_debug_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Debug", name='actionMenuDebug', parent_menu=self.root_menu_bar, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        
        an_action_key, self.root_window.ui.active_drivers_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivers", name='actionMenuDebugMenuActiveDrivers', parent_menu=self.root_window.ui.active_debug_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        # active_drivers_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivers']
        an_action_key, self.root_window.ui.active_drivables_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivables", name='actionMenuDebugMenuActiveDrivables', parent_menu=self.root_window.ui.active_debug_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        # active_drivables_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivables']
        an_action_key, self.root_window.ui.active_connections_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Connections", name='actionMenuDebugMenuActiveConnections', parent_menu=self.root_window.ui.active_debug_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        
        
        
        active_connections_menu = self.root_window.ui['actionMenuDebugMenuActiveConnections']
        
    

    @QtCore.pyqtSlot()
    def DebugMenuProviderMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self._DebugMenuProviderMixin_build_menus()
        self._DebugMenuProviderMixin_build_actions() # the actions actually depend on the existance of the menus for this dynamic wmenu case
    

    @QtCore.pyqtSlot()
    def DebugMenuProviderMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## Remove Debug Menu:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.DebugMenuProviderMixin_actionsDict

        curr_menubar.removeAction(curr_actions_dict['actionMenuDebug'])
        curr_window.ui.actionMenuDebug = None
        
        self.root_window.ui.active_drivers_menu = None
        self.root_window.ui.active_drivables_menu = None
        self.root_window.ui.active_connections_menu = None
        
        # curr_window.ui.menuDebugActionsDict = {} # Empty the dict of actions
        self.DebugMenuProviderMixin_actionsDict = {}


    @QtCore.pyqtSlot()
    def DebugMenuProviderMixin_on_menus_update(self):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc.
        TODO: finish implementation
        """
        ## Update Drivers Menu:
        curr_drivers_items = list(connection_man.registered_available_drivers.keys())
        for a_driver_key in curr_drivers_items:
            self.root_window.ui.active_drivers_menu.addAction(a_driver_key)
        ## Update Drivable Menu:
        curr_drivable_items = list(connection_man.registered_available_drivables.keys())
        for a_driveable_key in curr_drivable_items:
            self.root_window.ui.active_drivables_menu.addAction(a_driveable_key)
        ## Update Connections Menu:
        curr_connections_items = list(connection_man.active_connections.keys())
        for a_connection_key in curr_connections_items:
            self.root_window.ui.active_connections_menu.addAction(a_connection_key)
            