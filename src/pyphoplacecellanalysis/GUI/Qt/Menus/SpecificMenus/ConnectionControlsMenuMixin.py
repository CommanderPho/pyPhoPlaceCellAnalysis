from qtpy import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import initialize_global_menu_ui_variables_if_needed

class ConnectionControlsMenuMixin(object):
    """Adds a dynamically generated menubar to a QMainWindow for the purpose of connecting various separate windows
    
    Requirements:
        Implementor must be a QWidget class with:
            .window() property
            
    Main Functions:
        try_add_connections_menu(...)
        try_remove_connections_menu(...)
    
    Example:
        from pyphoplacecellanalysis.GUI.Qt.MainWindowWrapper import PhoBaseMainWindow

        curr_content_widget = spike_raster_window.window()
        curr_window = PhoBaseMainWindow(content_widget=curr_content_widget)
        menuConnections, actions_dict = build_menu(curr_window)
    
        from pyphoplacecellanalysis.GUI.Qt.Mixins.ConnectionControlsMenuMixin import ConnectionControlsMenuMixin
        curr_window, menuConnections, actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(spike_raster_window)
        
    Example 2:
        ## Note that curr_main_menu_window is usually not the same as spike_raster_window, instead curr_main_menu_window wraps it and produces the final output window
        curr_main_menu_window, menuConnections, connections_actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(spike_raster_window)
        spike_raster_window.main_menu_window = curr_main_menu_window # to retain the changes
        
        
    """
    top_level_menu_action_name = 'actionMenuConnections'
    top_level_menu_item_name = "menuConnections"

    # ## Mixin properties:    
    # def build_connections_menu(self):
    #     return ConnectionControlsMenuMixin.try_add_connections_menu(self)


    # def remove_connections_menu(self):
    #     """ Works to remove the menu created with menuConnections, actions_dict = build_menu(curr_window) """
    #     return ConnectionControlsMenuMixin.try_remove_connections_menu(self)


    @classmethod
    def try_add_connections_menu(cls, a_content_widget):
        curr_content_widget = a_content_widget.window()
        curr_window = PhoMenuHelper.try_get_menu_window(curr_content_widget)    
        menuConnections, actions_dict = ConnectionControlsMenuMixin._build_connections_menu(curr_window)
        return curr_window, menuConnections, actions_dict

    @classmethod
    def try_remove_connections_menu(cls, a_content_widget):
        """ Works to remove the menu created with menuConnections, actions_dict = build_menu(curr_window) """
        curr_window = a_content_widget.window()
        curr_actions_dict = curr_window.ui.menus.global_window_menus.menuConnections.actions_dict
        curr_menubar = curr_window.menuBar()
        # remove the menu:
        curr_menubar.removeAction(curr_actions_dict[cls.top_level_menu_action_name])
        curr_window.ui.menus.global_window_menus.menuConnections.top_level_menu = None
        curr_window.ui.menus.global_window_menus.menuConnections.actions_dict = {} # Empty the dict of actions

    @classmethod
    def _build_connections_menu(cls, a_main_window):
        a_main_window.ui.menubar = a_main_window.menuBar()
        found_extant_menu = a_main_window.ui.menubar.findChild(QtWidgets.QMenu, cls.top_level_menu_item_name) #cls.top_level_menu_item_name
        if found_extant_menu is not None:
            print(f'existing connections menu found. Returning without creating.')
            return a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu, a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict
        
        else:
            PhoMenuHelper.set_menu_default_stylesheet(a_main_window.ui.menubar) # Sets the default menu stylesheet
            # Initialize the .ui.menus.global_window_menus properties
            initialize_global_menu_ui_variables_if_needed(a_main_window)
            a_main_window.ui.menus.global_window_menus.menuConnections = PhoUIContainer.init_from_dict({'top_level_menu': None, 'actions_dict': {}})
            
            ## Only creates the QActions now, no QMenus:
            # Connect Child Item:
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "Connect Child...", name="actionConnect_Child", tooltip="Connect a child widget to another widget", icon_path=":/Icons/chain--arrow.png", actions_dict=a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict)
            # Disconnect from Driver item:
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "Disconnect from driver", name="actionDisconnect_from_driver", tooltip="Disconnects the item from the current driver", icon_path=":/Icons/chain--minus.png", actions_dict=a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict)

            ## Now Create the Menus for each QAction:
            # menuConnections = menubar.addMenu('&Connections')
            a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu = QtWidgets.QMenu(a_main_window.ui.menubar) # A QMenu
            a_main_window.ui.actionMenuConnections = a_main_window.ui.menubar.addMenu(a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu) # Used to remove the menu, a QAction
            # a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setTearOffEnabled(True)
            a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setObjectName(cls.top_level_menu_item_name)
            a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setTitle("Connections")
            
            ## Add the actions to the QMenu item:
            a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.addActions(a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict.values())
                        
            ## TODO: is this even needed? I think it's done to remove it, but can't I just use a_main_window.ui.actionMenuConnections directly?
            a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict[cls.top_level_menu_action_name] = a_main_window.ui.actionMenuConnections
            
            return a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu, a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict
