from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.CreateNewConnectedWidgetMenuMixin import CreateNewConnectedWidgetMenuMixin
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.ConnectionControlsMenuMixin import ConnectionControlsMenuMixin

from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.DebugMenuProviderMixin import DebugMenuProviderMixin

class GlobalMenubarPresentingMixin(object):
    """docstring for GlobalMenubarPresentingMixin."""
    
    @property
    def root_window(self):
        """The global_window property."""
        return self._root_window
        
    @property
    def root_menu_bar(self):
        """The root_menu_bar property."""
        return self.root_window.menuBar()
    
    
    def __init__(self):
        super(GlobalMenubarPresentingMixin, self).__init__()
        
        # Assumes that self is a QWidget subclass:
        self._root_window = PhoMenuHelper.try_get_menu_window(self)
        self.setup()
        self.build_ui()
        
        
    
    def setup(self):
        pass
    
    def build_ui(self):
        menuCreateNewConnected, actions_dict = cls._build_create_new_connected_widget_menu(curr_window)
        
        # curr_window, menuConnections, actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(spike_raster_window)
        # curr_window, menuCreateNewConnected, actions_dict = CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(a_content_widget=spike_raster_window)
        
        curr_window, menuConnections, actions_dict = ConnectionControlsMenuMixin.bu(spike_raster_window)
        curr_window, menuCreateNewConnected, actions_dict = CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(a_content_widget=spike_raster_window)
        
        