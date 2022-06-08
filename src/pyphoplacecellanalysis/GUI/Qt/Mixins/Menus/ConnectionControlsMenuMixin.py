from qtpy import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
# GuiResources_rc


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
        
    """
    def build_connections_menu(self):
        return ConnectionControlsMenuMixin.try_add_connections_menu(self)


    def remove_connections_menu(self):
        """ Works to remove the menu created with menuConnections, actions_dict = build_menu(curr_window) """
        return ConnectionControlsMenuMixin.try_remove_connections_menu(self)


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
        curr_actions_dict = curr_window.ui.connectionsMenuActionsDict
        curr_menubar = curr_window.menuBar()
        # remove the menu:
        curr_menubar.removeAction(curr_actions_dict['actionMenuConnections'])
        curr_window.ui.menuConnections = None
        curr_window.ui.connectionsMenuActionsDict = {} # Empty the dict of actions
            
        
        
    @classmethod
    def _build_connections_menu(cls, a_main_window):
        a_main_window.ui.menubar = a_main_window.menuBar()
        found_extant_menu = a_main_window.ui.menubar.findChild(QtWidgets.QMenu, "menuConnections") #"menuConnections"
        if found_extant_menu is not None:
            print(f'existing connections menu found. Returning without creating.')
            return a_main_window.ui.menuConnections, a_main_window.ui.connectionsMenuActionsDict
        
            # ## Removing existing:
            # print(f'existing connections menu found. Removing and rebuilding...')
            # # menubar.removeAction(menuConnections)
            # # menubar.removeAction(found_extant_menu)
            # a_main_window.ui.menubar.removeAction(actions_dict['actionMenuConnections'])
            # a_main_window.ui.menuConnections = None
            # a_main_window.ui.connectionsMenuActionsDict = {} # Empty the dict of actions:
        else:
            # menuConnections = menubar.addMenu('&Connections')
            a_main_window.ui.menuConnections = QtWidgets.QMenu(a_main_window.ui.menubar) # A QMenu
            a_main_window.ui.actionMenuConnections = a_main_window.ui.menubar.addMenu(a_main_window.ui.menuConnections) # Used to remove the menu, a QAction
            # a_main_window.ui.menuConnections.setTearOffEnabled(True)
            # icon1 = QtGui.QIcon()
            # icon1.addPixmap(QtGui.QPixmap(":/Icons/Icons/chain.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            # a_main_window.ui.menuConnections.setIcon(icon1)
            a_main_window.ui.menuConnections.setObjectName("menuConnections")
            a_main_window.ui.menuConnections.setTitle("Connections")
            
            a_main_window.ui.connectionsMenuActionsDict = {'actionMenuConnections': a_main_window.ui.actionMenuConnections}
                                               
            # Define actions/menu items:
            # if want the item added can do:
            #   curr_action = a_main_window.ui[curr_action_key]
            
            # Connect Child Item:
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "Connect Child...", name="actionConnect_Child", tooltip="Connect a child widget to another widget", icon_path=":/Icons/chain--arrow.png", parent_menu=a_main_window.ui.menuConnections, menu_actions_dict=a_main_window.ui.connectionsMenuActionsDict)
            # Disconnect from Driver item:
            curr_action_key = PhoMenuHelper.add_menu_action_item(a_main_window, "Disconnect from driver", name="actionDisconnect_from_driver", tooltip="Disconnects the item from the current driver", icon_path=":/Icons/chain--minus.png", parent_menu=a_main_window.ui.menuConnections, menu_actions_dict=a_main_window.ui.connectionsMenuActionsDict)

            return a_main_window.ui.menuConnections, a_main_window.ui.connectionsMenuActionsDict
