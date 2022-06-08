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
    def setup_menu_item(menu_item, text, name=None, tooltip=None, icon_path=None):
        """
        menu_item: a QtWidgets.QAction
        text (str): this is required, and is the text to display for the menu item
        name (str, Optional):
        tooltip (str, Optional): a string for the tooltip like "Connect a child widget to another widget"
        icon_path (str, Optional): a path like ":/Icons/chain--arrow.png"

        Usage:
            a_main_window.ui.actionConnect_Child = QtWidgets.QAction(a_main_window)
            PhoMenuHelper.setup_menu_item(a_main_window.ui.actionConnect_Child, "Connect Child...", name="actionConnect_Child", tooltip="Connect a child widget to another widget", icon_path=":/Icons/chain--arrow.png")
            a_main_window.ui.menuCreateNewConnectedWidget.addAction(a_main_window.ui.actionConnect_Child)

        """
        menu_item.setText(text)
        
        if name is None:
            # if no name is provided, build it from the text
            # text: "Connect Child..."
            # name: "actionConnect_Child"
            name = f'action{PhoMenuHelper.try_get_menu_object_name_from_text(text)}' #.replace(" ", "_")
            
        menu_item.setObjectName(name)
        
        if icon_path is not None:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            menu_item.setIcon(icon)

        if tooltip is not None:
            menu_item.setToolTip("Connect a child widget to another widget")
            
        return name # return the object name
    
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
    
