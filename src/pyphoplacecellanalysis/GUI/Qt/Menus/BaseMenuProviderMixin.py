from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
# from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper


def initialize_global_menu_ui_variables_if_needed(a_main_window):
    """ 
    sets up a_main_window.ui.menus.global_window_menus as needed for the menu providers if needed
    """
    return PhoMenuHelper.initialize_global_menu_ui_variables_if_needed(a_main_window)
    
         
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
        
    def execute(self, *args, **kwargs) -> None:
        """ Implicitly captures spike_raster_window """
        raise NotImplementedError # implementors must override        
    
    def __call__(self, *args, **kwds):
        return self.execute(*args, **kwds)


class BaseMenuProviderMixin(QtCore.QObject):
    """ a mixin class that provides one ore more QActions and QMenu items
    
    Classes that inherit from `BaseMenuProviderMixin` be used in two forms:
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
            

    Notes:
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

    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(BaseMenuProviderMixin, self).__init__(parent)
        
        # Setup member variables:
        # Assumes that self is a QWidget subclass:
        self._render_widget = render_widget # do we really need a reference to this?
        self._root_window = PhoMenuHelper.try_get_menu_window(render_widget)
    
    
    @pyqtExceptionPrintingSlot()
    def BaseMenuProviderMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        # Assumes that self is a QWidget subclass:
        if not self.has_root_window:
            self._root_window = PhoMenuHelper.try_get_menu_window(self)
    
        initialize_global_menu_ui_variables_if_needed(self._root_window) # sets up the .ui.menus.global_window_menus property
        

    
    @pyqtExceptionPrintingSlot()
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
    

    @pyqtExceptionPrintingSlot()
    def BaseMenuProviderMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self._BaseMenuProviderMixin_build_actions()
        self._BaseMenuProviderMixin_build_menus()
    

    @pyqtExceptionPrintingSlot()
    def BaseMenuProviderMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass


    @pyqtExceptionPrintingSlot()
    def BaseMenuProviderMixin_on_menus_update(self):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass

    
            