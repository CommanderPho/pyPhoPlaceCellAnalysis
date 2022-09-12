import re # regular expression for PhoMenuHelper
from qtpy import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.MainWindowWrapper import PhoBaseMainWindow
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

class PhoMenuHelper(object):
    """ A static helper for building QMenu items, QActions and adding them to a window """
    
    @staticmethod
    def try_get_menu_object_name_from_text(menu_text):
        # Remove all non-word characters (everything except numbers and letters)
        menu_text = re.sub(r"[^\w\s]", '', menu_text)
        # Replace all runs of whitespace with a single underscore
        menu_text = re.sub(r"\s+", '_', menu_text)
        return menu_text

    @staticmethod
    def setup_action_item(action_item, text, name=None, tooltip=None, icon_path=None):
        """
        action_item: a QtWidgets.QAction
        text (str): this is required, and is the text to display for the menu item
        name (str, Optional):
        tooltip (str, Optional): a string for the tooltip like "Connect a child widget to another widget"
        icon_path (str, Optional): a path like ":/Icons/chain--arrow.png"

        Usage:
            a_main_window.ui.actionConnect_Child = QtWidgets.QAction(a_main_window)
            PhoMenuHelper.setup_menu_item(a_main_window.ui.actionConnect_Child, "Connect Child...", name="actionConnect_Child", tooltip="Connect a child widget to another widget", icon_path=":/Icons/chain--arrow.png")
            a_main_window.ui.menus.global_window_menus.create_new_connected_widget.top_level_menu.addAction(a_main_window.ui.actionConnect_Child)

        """
        action_item.setText(text)
        
        if name is None:
            # if no name is provided, build it from the text
            # text: "Connect Child..."
            # name: "actionConnect_Child"
            name = f'action{PhoMenuHelper.try_get_menu_object_name_from_text(text)}' #.replace(" ", "_")
            
        action_item.setObjectName(name)
        
        if icon_path is not None:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            action_item.setIcon(icon)

        if tooltip is not None:
            action_item.setToolTip(tooltip)
            
        return name # return the object name
    
    @classmethod
    def add_action_item(cls, a_main_window, text, name=None, tooltip=None, icon_path=None, actions_dict=None):
        """Builds a new QAction and adds it to the provided actions_dict and sets a_main_window.ui.{curr_action_key} to the action.
            Internally calls cls.setup_action_item(...) to configure the action before adding it.    
            
            NOTE: this does not create/add the Menu, only the QAction
        Args:
            a_main_window (_type_): _description_
            text (_type_): _description_
            name (_type_, optional): _description_. Defaults to None.
            tooltip (_type_, optional): _description_. Defaults to None.
            icon_path (_type_, optional): _description_. Defaults to None.
        """
        curr_action = QtWidgets.QAction(a_main_window)
        curr_action_key = cls.setup_action_item(curr_action, text, name=name, tooltip=tooltip, icon_path=icon_path)
        a_main_window.ui[curr_action_key] = curr_action # add the action to the main window's .ui:
        # add to actions dictionary:
        if actions_dict is None:
            raise NotImplementedError
        actions_dict[curr_action_key] = a_main_window.ui[curr_action_key] # add to actions dictionary
        return curr_action_key
        
    @classmethod
    def add_menu_action(cls, a_main_window, text, name=None, tooltip=None, icon_path=None, parent_menu=None, actions_dict=None):
        """Builds a new QAction and adds it to the provided actions_dict and sets a_main_window.ui.{curr_action_key} to the action.
            Internally calls cls.setup_menu_item(...) to configure the action before adding it.    
        Args:
            a_main_window (_type_): _description_
            text (_type_): _description_
            name (_type_, optional): _description_. Defaults to None.
            tooltip (_type_, optional): _description_. Defaults to None.
            icon_path (_type_, optional): _description_. Defaults to None.
            
            
        # Separate setup:
            # Connect Child Item:
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "Connect Child...", name="actionConnect_Child", tooltip="Connect a child widget to another widget", icon_path=":/Icons/chain--arrow.png", actions_dict=a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict)
            
            
            
        """
        if parent_menu is None:
            raise NotImplementedError
        parent_menu.addAction(a_main_window.ui[curr_action_key]) # Add to menu
        
        # ## Add the actions to the QMenu item:
        # a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.addActions(a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict.values())
                    
        # ## TODO: is this even needed? I think it's done to remove it, but can't I just use a_main_window.ui.actionMenuConnections directly?
        # a_main_window.ui.menus.global_window_menus.menuConnections.actions_dict['actionMenuConnections'] = a_main_window.ui.actionMenuConnections
        
        
        
        
    @classmethod
    def add_menu(cls, a_main_window, text, name=None, parent_menu=None, tooltip=None, icon_path=None, menu_actions_dict=None):
        """ Modifies `a_main_window.ui[name]`
        
        a_main_window.ui.menus.global_window_menus.create_new_connected_widget.top_level_menu
        
        parent_menu: a QMenu parent or the root menuBar
        """
        # menuCreateNewConnectedWidget = menubar.addMenu('&Connections')
        curr_menu = QtWidgets.QMenu(parent_menu) # A QMenu
        curr_menu.setTitle(text)
        if name is None:
            # if no name is provided, build it from the text
            # text: "Create Connected Widget"
            # name: "menuCreateNewConnectedWidget"
            name = f'menu{PhoMenuHelper.try_get_menu_object_name_from_text(text)}' #.replace(" ", "_")
        # curr_menu.setTearOffEnabled(True)
        curr_menu.setObjectName(name)
        if tooltip is not None:
            curr_menu.setToolTip(tooltip)
        
        a_main_window.ui[name] = parent_menu.addMenu(curr_menu) # Used to remove the menu, a QAction
        if menu_actions_dict is None:
            raise NotImplementedError
        menu_actions_dict[name] = a_main_window.ui[name] # add to actions dictionary
        return name, curr_menu
            
    @classmethod
    def try_get_menu_window(cls, a_content_widget):
        curr_content_widget = a_content_widget.window()
        if not isinstance(curr_content_widget, QtWidgets.QMainWindow):
            # doesn't have a valid QMainWindow window, so wrap it in one using PhoBaseMainWindow(...)
            curr_window = PhoBaseMainWindow(content_widget=curr_content_widget)
        else:
            # already has a valid QMainWindow window
            curr_window = curr_content_widget
            # Make sure curr_window has a .ui property:
            if not hasattr(curr_window, 'ui'):
                # if the window has no .ui property, create one:
                setattr(curr_window, 'ui', PhoUIContainer())
        return curr_window
     
    @classmethod
    def try_get_menu_bar(cls, a_content_widget):
        """ Returns the main window's root menuBar
        can get the root menuBar children via:
            root_children = [a_child for a_child in menubar.children() if isinstance(a_child, pg.QtWidgets.QMenu)] # .title
            root_children
        """
        curr_window = cls.try_get_menu_window(a_content_widget)
        menubar = curr_window.menuBar()
        return menubar    
    
    @classmethod
    def set_menu_default_stylesheet(cls, root_menu_bar):
        """ styles a QMenuBar with my preferred menu stylesheet """
        custom_theme_multiline_string = """QMenuBar
        {
            background-color: #31363b;
            color: #eff0f1;
        }

        QMenuBar::item
        {
            background: transparent;
        }

        QMenuBar::item:selected
        {
            background: transparent;
        }

        QMenuBar::item:disabled
        {
            color: #76797c;
        }

        QMenuBar::item:pressed
        {
            background-color: #3daee9;
            color: #eff0f1;
            margin-bottom: -0.09em;
            padding-bottom: 0.09em;
        }

        QMenu
        {
            color: #eff0f1;
            margin: 0.09em;
        }

        QMenu::icon
        {
            margin: 0.23em;
        }

        QMenu::item
        {
            /* Add extra padding on the right for the QMenu arrow */
            padding: 0.23em 1.5em 0.23em 1.3em;
            border: 0.09em solid transparent;
            background: transparent;
        }

        QMenu::item:selected
        {
            color: #eff0f1;
            background-color: #3daee9;
        }

        QMenu::item:selected:disabled
        {
            background-color: #31363b;
        }

        QMenu::item:disabled
        {
            color: #76797c;
        }

        QMenu::indicator
        {
            width: 0.8em;
            height: 0.8em;
            /* To align with QMenu::icon, which has a 0.23em margin. */
            margin-left: 0.3em;
            subcontrol-position: center left;
        }

        QMenu::indicator:non-exclusive:unchecked
        {
            border-image: url(:/dark/checkbox_unchecked_disabled.svg);
        }

        QMenu::indicator:non-exclusive:unchecked:selected
        {
            border-image: url(:/dark/checkbox_unchecked_disabled.svg);
        }

        QMenu::indicator:non-exclusive:checked
        {
            border-image: url(:/dark/checkbox_checked.svg);
        }

        QMenu::indicator:non-exclusive:checked:selected
        {
            border-image: url(:/dark/checkbox_checked.svg);
        }

        QMenu::indicator:exclusive:unchecked
        {
            border-image: url(:/dark/radio_unchecked_disabled.svg);
        }

        QMenu::indicator:exclusive:unchecked:selected
        {
            border-image: url(:/dark/radio_unchecked_disabled.svg);
        }

        QMenu::indicator:exclusive:checked
        {
            border-image: url(:/dark/radio_checked.svg);
        }

        QMenu::indicator:exclusive:checked:selected
        {
            border-image: url(:/dark/radio_checked.svg);
        }

        QMenu::right-arrow
        {
            margin: 0.23em;
            border-image: url(:/dark/right_arrow.svg);
            width: 0.5em;
            height: 0.8em;
        }

        QMenu::right-arrow:disabled
        {
            border-image: url(:/dark/right_arrow_disabled.svg);
        }
        QAbstractItemView
        {
            alternate-background-color: #31363b;
            color: #eff0f1;
            border: 0.09em solid #31363b;
            border-radius: 0.09em;
        }

        QMenuBar::item:focus:!disabled
        {
            border: 0.04em solid #3daee9;
        }
        QMenu::separator
        {
            height: 0.09em;
            background-color: #76797c;
            padding-left: 0.2em;
            margin-top: 0.2em;
            margin-bottom: 0.2em;
            margin-left: 0.41em;
            margin-right: 0.41em;
        }
        """
    
        # old_custom_theme_multiline_string = """QMenuBar {
        # background-color: transparent;
        # }

        # QMenuBar::item {
        # color : white;
        # margin-top:4px;
        # spacing: 3px;
        # padding: 1px 10px;
        # background: transparent;
        # border-radius: 4px;
        # }

        # QMenuBar::item:selected { /* when selected using mouse or keyboard */
        # background: #a8a8a8;
        # }

        # QMenuBar::item:pressed {
        # background: #888888;
        # }
        # """
        # Set the stylesheet:
        root_menu_bar.setStyleSheet(custom_theme_multiline_string)
        
