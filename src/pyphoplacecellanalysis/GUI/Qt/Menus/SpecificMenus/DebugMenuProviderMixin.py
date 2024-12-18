from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin

class DebugMenuProviderMixin(BaseMenuProviderMixin):
    """ 
    
    .ui.menus.global_window_menus.debug_menu_provider.top_level_menu
    
    .ui.menus.global_window_menus.debug.actions_dict
    .ui.menus.global_window_menus.debug.actions_dict
    
    
    Can be used in two forms:
        1. Via inherting the desired Window widget class to DebugMenuProviderMixin
        2. Via initializing via the __init__(...) method: DebugMenuProviderMixin(render_widget)
            from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.DebugMenuProviderMixin import DebugMenuProviderMixin
            # Debug Menu
            _debug_menu_provider = DebugMenuProviderMixin(render_widget=spike_raster_window)
            _debug_menu_provider.DebugMenuProviderMixin_on_init()
            _debug_menu_provider.DebugMenuProviderMixin_on_buildUI()
            ...
            # Set the returned provider object:
            spike_raster_window.main_menu_window.ui.menus.global_window_menus.debug.menu_provider_obj = _debug_menu_provider
            
            
    """
    action_name = 'Debug'
    top_level_menu_name = 'actionMenuDebug'
    
    
    @property
    def activeMenuReference(self):
        """The reference to the top-level PhoUIContainer for this menu where references are stored to the ui elements and their actions."""
        return self.root_window.ui.menus.global_window_menus.debug
    @activeMenuReference.setter
    def activeMenuReference(self, value):
        self.root_window.ui.menus.global_window_menus.debug = value
        
    @property
    def DebugMenuProviderMixin_actionsDict(self):
        return self.activeMenuReference.actions_dict
    @DebugMenuProviderMixin_actionsDict.setter
    def DebugMenuProviderMixin_actionsDict(self, value):
        self.activeMenuReference.actions_dict = value
        
    @property
    def connection_man(self):
        """The connection_man property."""
        return self.root_window.connection_man


    # __init__ ___________________________________________________________________________________________________________ #
    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(DebugMenuProviderMixin, self).__init__(render_widget=render_widget, parent=parent, **kwargs)
        # Setup member variables:
        self.DebugMenuProviderMixin_on_init()
        self.DebugMenuProviderMixin_on_setup()
        self.DebugMenuProviderMixin_on_buildUI()
        
        
    @pyqtExceptionPrintingSlot()
    def DebugMenuProviderMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        BaseMenuProviderMixin.BaseMenuProviderMixin_on_init(self)
        # Define the core object
        self.activeMenuReference = PhoUIContainer.init_from_dict({'top_level_menu': None, 'actions_dict': {}, 'menu_provider_obj': None})        
    
    @pyqtExceptionPrintingSlot()
    def DebugMenuProviderMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    # build QActions _____________________________________________________________________________________________________ #
    def _DebugMenuProviderMixin_build_actions(self):
        """ build QActions """
        ## Add the dynamic menu entries:
        connection_man = self.connection_man
        
        ## Update Drivers Menu:
        curr_drivers_items = list(connection_man.registered_available_drivers.keys())
        for a_driver_key in curr_drivers_items:
            self.activeMenuReference.active_drivers_menu.addAction(a_driver_key)
        self.activeMenuReference.active_drivers_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivers.get(action.text(), f'Driver KeyNotFound: {action.text()}')))

        ## Update Drivable Menu:
        curr_drivable_items = list(connection_man.registered_available_drivables.keys())
        for a_driveable_key in curr_drivable_items:
            self.activeMenuReference.active_drivables_menu.addAction(a_driveable_key)
        self.activeMenuReference.active_drivables_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivables.get(action.text(), f'Drivable KeyNotFound: {action.text()}')))
  
        ## Update Connections Menu:
        curr_connections_descriptions = list([a_conn_ref.description for a_conn_ref in connection_man.active_connections.values()])
        for a_connection_key in curr_connections_descriptions:
            self.activeMenuReference.active_connections_menu.addAction(a_connection_key)
        self.activeMenuReference.active_connections_menu.triggered.connect(lambda action: print((connection_man.find_active_connection(action.text()) or f'Connection KeyNotFound: {action.text()}')))
        
        
    # build QMenus _______________________________________________________________________________________________________ #
    def _DebugMenuProviderMixin_build_menus(self):
        """ build QMenus """
        an_action_key, self.activeMenuReference.top_level_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Debug", name=self.top_level_menu_name, parent_menu=self.root_menu_bar, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        
        an_action_key, self.activeMenuReference.active_drivers_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivers", name='actionMenuDebugMenuActiveDrivers', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        # active_drivers_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivers']
        an_action_key, self.activeMenuReference.active_drivables_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Drivables", name='actionMenuDebugMenuActiveDrivables', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        # active_drivables_menu = self.root_window.ui['actionMenuDebugMenuActiveDrivables']
        an_action_key, self.activeMenuReference.active_connections_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Active Connections", name='actionMenuDebugMenuActiveConnections', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=self.DebugMenuProviderMixin_actionsDict)
        # active_connections_menu = self.root_window.ui['actionMenuDebugMenuActiveConnections']
        
    @pyqtExceptionPrintingSlot()
    def DebugMenuProviderMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self._DebugMenuProviderMixin_build_menus()
        self._DebugMenuProviderMixin_build_actions() # the actions actually depend on the existance of the menus for this dynamic menu case
    
    
    def _DebugMenuProviderMixin_clear_menu_actions(self):
        """ empties all submenus of their actions so that they can be freshly updated via self.DebugMenuProviderMixin_on_menus_update() """
        actions_to_remove = self.activeMenuReference.active_drivers_menu.actions()
        for an_old_action in actions_to_remove:
            an_old_action.triggered.disconnect()
            self.activeMenuReference.active_drivers_menu.removeAction(an_old_action)
            
        actions_to_remove = self.activeMenuReference.active_drivables_menu.actions()
        for an_old_action in actions_to_remove:
            an_old_action.triggered.disconnect()
            self.activeMenuReference.active_drivables_menu.removeAction(an_old_action)
            
        actions_to_remove = self.activeMenuReference.active_connections_menu.actions()
        for an_old_action in actions_to_remove:
            an_old_action.triggered.disconnect()
            self.activeMenuReference.active_connections_menu.removeAction(an_old_action)
            

    @pyqtExceptionPrintingSlot()
    def DebugMenuProviderMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## Remove Debug Menu:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.DebugMenuProviderMixin_actionsDict

        curr_menubar.removeAction(curr_actions_dict[self.top_level_menu_name])
        curr_window.ui.actionMenuDebug = None
        
        self.activeMenuReference.active_drivers_menu = None
        self.activeMenuReference.active_drivables_menu = None
        self.activeMenuReference.active_connections_menu = None
        
        # curr_window.ui.menus.global_window_menus.debug.actions_dict = {} # Empty the dict of actions
        self.DebugMenuProviderMixin_actionsDict = {}


    @pyqtExceptionPrintingSlot()
    def DebugMenuProviderMixin_on_menus_update(self):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc.
        TODO: finish implementation
        """
        self._DebugMenuProviderMixin_clear_menu_actions() # clear the existing menu actions
        
        ## Update Drivers Menu:
        curr_drivers_items = list(self.connection_man.registered_available_drivers.keys())
        for a_driver_key in curr_drivers_items:
            self.activeMenuReference.active_drivers_menu.addAction(a_driver_key)
        ## Update Drivable Menu:
        curr_drivable_items = list(self.connection_man.registered_available_drivables.keys())
        for a_driveable_key in curr_drivable_items:
            self.activeMenuReference.active_drivables_menu.addAction(a_driveable_key)
        ## Update Connections Menu:
        curr_connections_descriptions = list([a_conn_ref.description for a_conn_ref in self.connection_man.active_connections.values()])
        for a_connection_key in curr_connections_descriptions:
            self.activeMenuReference.active_connections_menu.addAction(a_connection_key)
            