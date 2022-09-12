from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper


def initialize_global_menu_ui_variables(a_main_window):
    """ 
    sets up a_main_window.ui.menus.global_window_menus as needed for the menu providers if needed
    """
    if isinstance(a_main_window.ui, DynamicParameters):            
        # Need this workaround because hasattr fails for DynamicParameters/PhoUIContainer right now:
        a_main_window.ui.setdefault('menus', PhoUIContainer.init_from_dict({}))
    else:
        if not hasattr(a_main_window.ui, 'menus'):
            a_main_window.ui.menus = PhoUIContainer.init_from_dict({})
        
    # a_main_window.ui.menus.setdefault('global_window_menus', PhoUIContainer.init_from_dict({}))
    if not a_main_window.ui.menus.has_attr('global_window_menus'):
        a_main_window.ui.menus.global_window_menus = PhoUIContainer.init_from_dict({})
        
            
            
            
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


class BaseMenuProviderMixin(QtCore.QObject):
    """ a mixin class that provides one ore more QActions and QMenu items
    
        A best practice is to create actions as children of the window in which you’re going to use them.
    
    Implementors Require:
        self._root_window
    
    """
    
    @property
    def has_root_window(self):
        if not hasattr(self, '_root_window'):
            return False
        else:
            return (self._root_window is not None)
        
            
    @property
    def root_window(self):
        """The global_window property."""
        if not hasattr(self, '_root_window'):
            return None
        else:
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
    
    
    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(BaseMenuProviderMixin, self).__init__(parent)
        
        # Setup member variables:
        # Assumes that self is a QWidget subclass:
        self._render_widget = render_widget # do we really need a reference to this?
        self._root_window = PhoMenuHelper.try_get_menu_window(render_widget)
    
    
    @QtCore.Slot()
    def BaseMenuProviderMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        # Assumes that self is a QWidget subclass:
        if not self.has_root_window:
            self._root_window = PhoMenuHelper.try_get_menu_window(self)
    
        initialize_global_menu_ui_variables(self._root_window) # sets up the .ui.menus.global_window_menus property
        

    
    @QtCore.Slot()
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
    

    @QtCore.Slot()
    def BaseMenuProviderMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self._BaseMenuProviderMixin_build_actions()
        self._BaseMenuProviderMixin_build_menus()
    

    @QtCore.Slot()
    def BaseMenuProviderMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass


    @QtCore.Slot()
    def BaseMenuProviderMixin_on_menus_update(self):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass

    
            