from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin

from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.LocalMenus_AddRenderable import LocalMenus_AddRenderable

class CreateLinkedWidget_MenuProvider(BaseMenuProviderMixin):
    """ 
    
    .ui.menus.global_window_menus.debug_menu_provider.top_level_menu
    
    .ui.menus.global_window_menus.debug.actions_dict
    .ui.menus.global_window_menus.debug.actions_dict
    
    """
    top_level_menu_name = 'actionMenuCreateLinkedWidget'
    
    @property
    def activeMenuReference(self):
        """The reference to the top-level PhoUIContainer for this menu where references are stored to the ui elements and their actions."""
        return self.root_window.ui.menus.global_window_menus.create_linked_widget
    @activeMenuReference.setter
    def activeMenuReference(self, value):
        self.root_window.ui.menus.global_window_menus.create_linked_widget = value
        
    @property
    def CreateLinkedWidget_MenuProvider_actionsDict(self):
        return self.activeMenuReference.actions_dict
    @CreateLinkedWidget_MenuProvider_actionsDict.setter
    def CreateLinkedWidget_MenuProvider_actionsDict(self, value):
        self.activeMenuReference.actions_dict = value


    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(CreateLinkedWidget_MenuProvider, self).__init__(render_widget=render_widget, parent=parent, **kwargs)
        # Setup member variables:
        pass
        
        
    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_init(self):
        """ perform any parameters setting/checking during init """
        BaseMenuProviderMixin.BaseMenuProviderMixin_on_init(self)
        
        assert self.has_root_window, "No root window!"
        # Define the core object
        self.activeMenuReference = PhoUIContainer.init_from_dict({'top_level_menu': None, 'actions_dict': {}})
    
    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    
    # def _CreateLinkedWidget_MenuProvider_build_actions(self):
    #     """ build QActions """
    #     ## Add the dynamic menu entries:
    #     # connection_man = self.connection_man
        
    #     # ## Update Drivers Menu:
    #     # curr_drivers_items = list(connection_man.registered_available_drivers.keys())
    #     # for a_driver_key in curr_drivers_items:
    #     #     self.activeMenuReference.active_drivers_menu.addAction(a_driver_key)
    #     # self.activeMenuReference.active_drivers_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivers.get(action.text(), f'Driver KeyNotFound: {action.text()}')))

    #     # ## Update Drivable Menu:
    #     # curr_drivable_items = list(connection_man.registered_available_drivables.keys())
    #     # for a_driveable_key in curr_drivable_items:
    #     #     self.activeMenuReference.active_drivables_menu.addAction(a_driveable_key)
    #     # self.activeMenuReference.active_drivables_menu.triggered.connect(lambda action: print(connection_man.registered_available_drivables.get(action.text(), f'Drivable KeyNotFound: {action.text()}')))
  
    #     # ## Update Connections Menu:
    #     # curr_connections_descriptions = list([a_conn_ref.description for a_conn_ref in connection_man.active_connections.values()])
    #     # for a_connection_key in curr_connections_descriptions:
    #     #     self.activeMenuReference.active_connections_menu.addAction(a_connection_key)
    #     # # self.activeMenuReference.active_connections_menu.triggered.connect(lambda action: print(connection_man.active_connections.get(action.text(), f'Connection KeyNotFound: {action.text()}')))
    #     # self.activeMenuReference.active_connections_menu.triggered.connect(lambda action: print((connection_man.find_active_connection(action.text()) or f'Connection KeyNotFound: {action.text()}')))
    #     pass

    # def _CreateLinkedWidget_MenuProvider_build_menus(self):
    #     """ build QMenus from UI
        
    #     self.ui.menuCreate_Paired_Widget
    #     self.ui.actionTimeSynchronizedOccupancyPlotter
    #     self.ui.actionTimeSynchronizedPlacefieldsPlotter
        
    #     """
        
    #     widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
    #     menu_item = widget.ui.menuCreate_Paired_Widget
        
    #     ## Time Intervals/Epochs:
    #     submenu_addTimeIntervals = [widget.ui.actionTimeSynchronizedOccupancyPlotter,
    #                                 widget.ui.actionTimeSynchronizedPlacefieldsPlotter,
    #                                 ]
    #     submenu_addTimeIntervalCallbacks = [lambda evt=None: print(f'actionTimeSynchronizedOccupancyPlotter callback'),
    #                                         lambda evt=None: print(f'actionTimeSynchronizedPlacefieldsPlotter callback'),
    #                                         ]
    #     submenu_addTimeIntervals_Connections = []
    #     for an_action, a_callback in zip(submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks):
    #         _curr_conn = an_action.triggered.connect(a_callback)
    #         submenu_addTimeIntervals_Connections.append(_curr_conn)
            

    # def _CreateLinkedWidget_MenuProvider_build_menus(self):
    #     """ build QMenus """
        
    #     widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
    #     renderable_menu = widget.ui.menuAdd_Renderable
        
        
    #     an_action_key, self.activeMenuReference.top_level_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Create Paired Widget", name=self.top_level_menu_name, parent_menu=self.root_menu_bar, menu_actions_dict=self.CreateLinkedWidget_MenuProvider_actionsDict)
        
    #     # Adds submenus:
    #     # an_action_key, self.activeMenuReference.active_drivers_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="TimeSynchronizedOccupancyPlotter", name='actionMenuDebugMenuActiveDrivers', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=self.CreateLinkedWidget_MenuProvider_actionsDict)
                
    #     an_action_key = PhoMenuHelper.add_action_item(self.root_window, text="TimeSynchronizedOccupancyPlotter", icon_path=':/Render/Icons/actions/bar-chart_2@1x.png', actions_dict=self.CreateLinkedWidget_MenuProvider_actionsDict)
                
    #     # an_action_key, self.activeMenuReference.active_drivables_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="TimeSynchronizedPlacefieldsPlotter", name='actionMenuDebugMenuActiveDrivables', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=self.CreateLinkedWidget_MenuProvider_actionsDict)
        
    #     an_action_key = PhoMenuHelper.add_action_item(self.root_window, text="TimeSynchronizedPlacefieldsPlotter", icon_path=':/Render/Icons/actions/wifi-channel_2@1x.png', actions_dict=self.CreateLinkedWidget_MenuProvider_actionsDict)
        
    #     ## Add the actions to the QMenu item:
    #     self.activeMenuReference.top_level_menu.addActions(self.CreateLinkedWidget_MenuProvider_actionsDict.values())
                    
    #     ## TODO: is this even needed? I think it's done to remove it, but can't I just use a_main_window.ui.actionMenuConnections directly?
    #     # self.activeMenuReference.actions_dict['actionMenuConnections'] = self.root_window.ui.actionMenuConnections

    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc)
        
        build QMenus from UI
        
        self.ui.menuCreate_Paired_Widget
        self.ui.actionTimeSynchronizedOccupancyPlotter
        self.ui.actionTimeSynchronizedPlacefieldsPlotter
        
        
        """
        # self._CreateLinkedWidget_MenuProvider_build_menus()
        # self._CreateLinkedWidget_MenuProvider_build_actions() # the actions actually depend on the existance of the menus for this dynamic menu case
        
        widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
        renderable_menu = widget.ui.menuCreate_Paired_Widget
        
        ## Time Intervals/Epochs:
        submenu_menuItems = [widget.ui.actionTimeSynchronizedOccupancyPlotter,
                                    widget.ui.actionTimeSynchronizedPlacefieldsPlotter,
                                    ]
        submenu_menuCallbacks = [lambda evt=None: print(f'actionTimeSynchronizedOccupancyPlotter callback'),
                                            lambda evt=None: print(f'actionTimeSynchronizedPlacefieldsPlotter callback'),
                                            ]
        submenu_menu_Connections = []
        for an_action, a_callback in zip(submenu_menuItems, submenu_menuCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_menu_Connections.append(_curr_conn)

        active_2d_plot_renderable_menus = widget, renderable_menu, (submenu_menuItems, submenu_menuCallbacks, submenu_menu_Connections)
        
        
        # widget_2d_menu = active_2d_plot_renderable_menus[0]
        # renderable_menu = widget_2d_menu.ui.menuCreate_Paired_Widget
        # renderable_menu = widget.ui.menuCreate_Paired_Widget
        
        # curr_window = PhoMenuHelper.try_get_menu_window(a_content_widget=a_content_widget)
        # curr_menubar = PhoMenuHelper.try_get_menu_bar(a_content_widget=a_content_widget)
    
        # print(f'renderable_menu: {renderable_menu}') 
        # curr_menubar.addMenu(renderable_menu) # add it to the menubar

        
        
        ## Add menu to the main menu bar:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.CreateLinkedWidget_MenuProvider_actionsDict
        self.activeMenuReference.top_level_menu = renderable_menu
        self.activeMenuReference.actions_dict = curr_actions_dict
        
        self.activeMenuReference.top_level_menu.setObjectName("menuCreateLinkedWidget")
        curr_window.ui.actionMenuCreateLinkedWidget = curr_menubar.addMenu(renderable_menu)  # add it to the menubar

        # Save references in the curr_window
        self.activeMenuReference.actions_dict['actionMenuCreateLinkedWidget'] = curr_window.ui.actionMenuCreateLinkedWidget 
        
        
        # Debugging Reference
        self.activeMenuReference.all_refs = active_2d_plot_renderable_menus
    
        return self.activeMenuReference.top_level_menu, self.activeMenuReference.actions_dict
        
        # return self.activeMenuReference.top_level_menu, self.activeMenuReference.actions_dict, self.activeMenuReference.all_refs
    
        

    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## Remove Debug Menu:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.CreateLinkedWidget_MenuProvider_actionsDict

        curr_menubar.removeAction(curr_actions_dict[self.top_level_menu_name])
        curr_window.ui.actionMenuDebug = None
        
        # self.activeMenuReference.active_drivers_menu = None
        # self.activeMenuReference.active_drivables_menu = None
        # self.activeMenuReference.active_connections_menu = None
        
        # curr_window.ui.menus.global_window_menus.debug.actions_dict = {} # Empty the dict of actions
        self.CreateLinkedWidget_MenuProvider_actionsDict = {}

    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_menus_update(self):
        """ called to update menus dynamically. Only needed if the menu items themselves change dynamically.
        """
        pass