# MainWindowWrapper
from qtpy import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.Resources import ActionIcons
from pyphoplacecellanalysis.Resources import GuiResources

from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.PhoMainAppWindowBase import PhoMainAppWindowBase



class PhoBaseMainWindow(PhoMainAppWindowBase):
    """ a custom QMainWindow subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
    Usage:
        from pyphoplacecellanalysis.GUI.Qt.MainWindowWrapper import PhoBaseMainWindow

        curr_content_widget = spike_raster_window.window()
        curr_window = PhoBaseMainWindow(content_widget=curr_content_widget)
    
    """
    
    
    def __init__(self, title='PhoBaseMainWindow', content_widget=None, defer_show=False, *args, **kwargs):
        # self._app = pg.mkQApp(title) # makes a new QApplication or gets the reference to an existing one.
        # self.ui = PhoUIContainer()
        super(PhoBaseMainWindow, self).__init__(*args, **kwargs)
        if content_widget is not None:
            self.ui.main_widget = content_widget
        else:
            self.ui.main_widget = QtWidgets.QWidget() # make an empty widget
            
        self.setup()
        self.buildUI()
        
        if not defer_show:
            self.show()


    def setup(self):
        # Use existing central widget: 
        # Use self.ui.main_widget as central widget:        
        if self.ui.main_widget is not None:
            main_widget_geom = self.ui.main_widget.window().geometry() # get the QTCore PyRect object
            main_x, main_y, main_width, main_height = main_widget_geom.getRect() # Note: dx & dy refer to width and height
            self.resize(main_width, main_height) # resize self to the same size as the main widget
            self.move(main_x, main_y)
            
            # Copy properties from child widget:
            main_widget_name = self.ui.main_widget.windowTitle()
            main_widget_icon = self.ui.main_widget.windowIcon()
    
            if main_widget_name is not None:
                self.setWindowTitle(main_widget_name)
    
            if main_widget_icon is not None:
                self.setWindowIcon(main_widget_icon)
                
            self.setStyleSheet(self.ui.main_widget.styleSheet())
            self.setStyle(self.ui.main_widget.style())
            self.setWindowFlags(self.ui.main_widget.windowFlags())
            self.setWindowFilePath(self.ui.main_widget.windowFilePath())

            # Set the central widget:
            self.setCentralWidget(self.ui.main_widget)
        

    def buildUI(self):
        pass
        
        
    @classmethod
    def build_menu(cls, a_main_window):
        a_main_window.ui.menubar = a_main_window.menuBar()
        # found_extant_menu = menubar.findChild(QtWidgets.QMenu, "menuConnections") #"menuConnections"
        # if found_extant_menu is not None:
        #     menubar.removeAction(menuConnections)
        #     menubar.removeAction(found_extant_menu)
        
        # menuConnections = menubar.addMenu('&Connections')
        a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu = QtWidgets.QMenu(a_main_window.ui.menubar) # A QMenu
        a_main_window.ui.actionMenuConnections = a_main_window.ui.menubar.addMenu(a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu) # Used to remove the menu, a QAction
        
        # a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setTearOffEnabled(True)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Icons/Icons/chain.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setIcon(icon1)
        a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.setObjectName("menuConnections")
        # a_main_window.setMenuBar(menubar)
        
        # Define actions
        a_main_window.ui.actionConnect_Child = QtWidgets.QAction(a_main_window)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Icons/chain--arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        a_main_window.ui.actionConnect_Child.setIcon(icon2)
        a_main_window.ui.actionConnect_Child.setObjectName("actionConnect_Child")
        a_main_window.ui.actionConnect_Child.setText("Connect Child...")
        a_main_window.ui.actionConnect_Child.setToolTip("Connect a child widget to another widget")
        
        a_main_window.ui.actionDisconnect_from_driver = QtWidgets.QAction(a_main_window)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Icons/chain--minus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        a_main_window.ui.actionDisconnect_from_driver.setIcon(icon3)
        a_main_window.ui.actionDisconnect_from_driver.setObjectName("actionDisconnect_from_driver")
        a_main_window.ui.actionDisconnect_from_driver.setText("Disconnect from driver")
        a_main_window.ui.actionDisconnect_from_driver.setToolTip("Disconnects the item from the current driver")

        # Add to connections menu:
        a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.addAction(a_main_window.ui.actionConnect_Child)
        a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu.addAction(a_main_window.ui.actionDisconnect_from_driver)

        return a_main_window.ui.menus.global_window_menus.menuConnections.top_level_menu, {'actionMenuConnections':a_main_window.ui.actionMenuConnections, 'actionConnect_Child':a_main_window.ui.actionConnect_Child, 'actionDisconnect_from_driver':a_main_window.ui.actionDisconnect_from_driver}

        
    def closeEvent(self, event):
        # Enables closing all secondary windows when this (main) window is closed.
        # for window in QtWidgets.QApplication.topLevelWidgets():
        #     window.close()
        pass
            
            

