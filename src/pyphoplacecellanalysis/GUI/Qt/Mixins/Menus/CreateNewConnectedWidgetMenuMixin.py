# CreateNewConnectedWidgetMenuMixin
from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
# GuiResources_rc


class CreateNewConnectedWidgetMenuMixin(object):
    """Adds a dynamically generated menubar to a QMainWindow for the purpose of connecting various separate windows
    
    Requirements:
        Implementor must be a QWidget class with:
            .window() property
            
    Main Functions:
        try_add_create_new_connected_widget_menu(...)
        try_remove_create_new_connected_widget_menu(...)
    
    Example:
        from pyphoplacecellanalysis.GUI.Qt.MainWindowWrapper import PhoBaseMainWindow

        curr_content_widget = spike_raster_window.window()
        curr_window = PhoBaseMainWindow(content_widget=curr_content_widget)
        menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window)
        
        from pyphoplacecellanalysis.GUI.Qt.Mixins.CreateNewConnectedWidgetMenuMixin import CreateNewConnectedWidgetMenuMixin
        curr_window, menuCreateNewConnectedWidget, actions_dict = CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(spike_raster_window)

    
    """
    def build_create_new_connected_widget_menu(self):
        return CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(self)


    def remove_create_new_connected_widget_menu(self):
        """ Works to remove the menu created with menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window) """
        return CreateNewConnectedWidgetMenuMixin.try_remove_create_new_connected_widget_menu(self)


    @classmethod
    def try_add_create_new_connected_widget_menu(cls, a_content_widget):
        """ 
        Usage:
            curr_window, menuCreateNewConnected, actions_dict = _try_add_create_new_connected_widget_menu(spike_raster_window)
        """
        curr_window = PhoMenuHelper.try_get_menu_window(a_content_widget)
        menuCreateNewConnected, actions_dict = cls._build_create_new_connected_widget_menu(curr_window)    
        return curr_window, menuCreateNewConnected, actions_dict

    @classmethod
    def try_remove_create_new_connected_widget_menu(cls, a_content_widget):
        """ Works to remove the menu created with menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window) """
        curr_window = a_content_widget.window()
        curr_actions_dict = curr_window.ui.createNewConnectedWidgetMenuActionsDict
        curr_menubar = curr_window.menuBar()
        # remove the menu:
        curr_menubar.removeAction(curr_actions_dict['actionMenuCreateNewConnectedWidget'])
        curr_window.ui.menuCreateNewConnectedWidget = None
        curr_window.ui.createNewConnectedWidgetMenuActionsDict = {} # Empty the dict of actions
            
        
        
    @classmethod
    def _build_create_new_connected_widget_menu(cls, a_main_window, debug_print=False):
        a_main_window.ui.menubar = a_main_window.menuBar()
        found_extant_menu = a_main_window.ui.menubar.findChild(QtWidgets.QMenu, "menuCreateNewConnectedWidget") #"menuCreateNewConnectedWidget"
        if found_extant_menu is not None:
            if debug_print:
                print(f'existing create new connected widget menu found. Returning without creating.')
            return a_main_window.ui.menuCreateNewConnectedWidget, a_main_window.ui.createNewConnectedWidgetMenuActionsDict
        else:
            # menuCreateNewConnectedWidget = menubar.addMenu('&Connections')
            a_main_window.ui.menuCreateNewConnectedWidget = QtWidgets.QMenu(a_main_window.ui.menubar) # A QMenu
            a_main_window.ui.actionMenuCreateNewConnectedWidget = a_main_window.ui.menubar.addMenu(a_main_window.ui.menuCreateNewConnectedWidget) # Used to remove the menu, a QAction
            # a_main_window.ui.menuCreateNewConnectedWidget.setTearOffEnabled(True)
            a_main_window.ui.menuCreateNewConnectedWidget.setObjectName("menuCreateNewConnectedWidget")
            a_main_window.ui.menuCreateNewConnectedWidget.setTitle("Create Connected Widget")

            # Define dictionary:
            a_main_window.ui.createNewConnectedWidgetMenuActionsDict = {'actionMenuCreateNewConnectedWidget':a_main_window.ui.actionMenuCreateNewConnectedWidget}
            # Define actions/menu items:
                # if want the item added can do:
                #   curr_action = a_main_window.ui[curr_action_key]
            
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "New Connected 3D Raster (PyQtGraph)", name="actionNewConnected3DRaster_PyQtGraph", tooltip="Create a new PyQtGraph 3D Raster plotter and connect it to this window", icon_path=":/Icons/Icons/SpikeRaster3DIcon.ico",
                                                        parent_menu=a_main_window.ui.menuCreateNewConnectedWidget, menu_actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            if debug_print:
                print(f'curr_action_key: {curr_action_key}')
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "New Connected 3D Raster (Vedo)", name="actionNewConnected3DRaster_Vedo", tooltip="Create a new Vedo 3D Raster plotter and connect it to this window", icon_path=":/Icons/Icons/SpikeRaster3D_VedoIcon.ico",
                                                        parent_menu=a_main_window.ui.menuCreateNewConnectedWidget, menu_actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            if debug_print:
                print(f'curr_action_key: {curr_action_key}')
            
            
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "New Connected Tuning Curves Explorer (ipcDataExplorer)", name="actionNewConnectedDataExplorer_ipc", tooltip="Create a new 3D Interactive Tuning Curve Data Explorer Plotter and connect it to this window", icon_path=":/Icons/Icons/InteractivePlaceCellDataExplorerIconWithLabel.ico",
                                                        parent_menu=a_main_window.ui.menuCreateNewConnectedWidget, menu_actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "New Connected Spikes+Behavior Explorer (ipspikesDataExplorer)", name="actionNewConnectedDataExplorer_ipspikes", tooltip="Create a new 3D Interactive Spike and Behavior Plotter and connect it to this window", icon_path=":/Icons/Icons/TuningMapDataExplorerIconWithLabel.ico",
                                                        parent_menu=a_main_window.ui.menuCreateNewConnectedWidget, menu_actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            
            
            return a_main_window.ui.menuCreateNewConnectedWidget, a_main_window.ui.createNewConnectedWidgetMenuActionsDict


