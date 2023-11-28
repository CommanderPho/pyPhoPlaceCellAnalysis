from typing import Tuple
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea

from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.PhoMainAppWindowBase import PhoMainAppWindowBase

from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, DynamicDockDisplayAreaContentMixin

    
# ==================================================================================================================== #
class PhoDockAreaContainingWindow(DynamicDockDisplayAreaContentMixin, PhoMainAppWindowBase):
    """ a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
    """
    @property
    def area(self) -> DockArea:
        return self.ui.area

    def __init__(self, title='PhoDockAreaContainingWindow', *args, **kwargs):
        # self._app = pg.mkQApp(title) # makes a new QApplication or gets the reference to an existing one.
        # self.ui = PhoUIContainer()
        self.DynamicDockDisplayAreaContentMixin_on_init()
        super(PhoDockAreaContainingWindow, self).__init__(*args, **kwargs)
        self.setup()
        self.buildUI()
        

    def setup(self):
        self.ui.area = DockArea()
        # Use self.ui.area as central widget:        
        self.setCentralWidget(self.ui.area)    
        self.DynamicDockDisplayAreaContentMixin_on_setup()
        self.GlobalConnectionManagerAccessingMixin_on_setup()
        
    def buildUI(self):
        self.DynamicDockDisplayAreaContentMixin_on_buildUI()
        
    def closeEvent(self, event):
        # Enables closing all secondary windows when this (main) window is closed.
        print(f'PhoDockAreaContainingWindow.closeEvent(event: {event})')
        
        self.GlobalConnectionManagerAccessingMixin_on_destroy()
        self.DynamicDockDisplayAreaContentMixin_on_destroy()
        
        for window in QtWidgets.QApplication.topLevelWidgets():
            window.close()
            
    ########################################################
    ## For GlobalConnectionManagerAccessingMixin conformance:
    ########################################################
    
    def try_register_widget_if_control(self, a_widget, debug_print=False):
        if self.connection_man.is_known_driver(a_widget):
            print(f'\t\ta_widget: {a_widget} is registering as a driver...')
            self.connection_man.register_driver(a_widget)
        if self.connection_man.is_known_drivable(a_widget):
            print(f'\t\ta_widget: {a_widget} is registering as a drivable...')
            self.connection_man.register_drivable(a_widget)
    
    # @QtCore.pyqtSlot()
    def try_register_any_control_widgets(self):
        """ called whenever the widgets are updated to try and register widgets as controls """
        print(f'PhoDockAreaContainingWindow.try_register_any_control_widgets()')
        flat_widgets_list = self.get_flat_widgets_list()
        print(f'\tflat_widgets_list contains {len(flat_widgets_list)} items')
        for a_widget in flat_widgets_list:
            self.try_register_widget_if_control(a_widget)

    
    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_setup(self):
        """ perfrom registration of drivers/drivables:"""
        ## TODO: register children
        print(f'PhoDockAreaContainingWindow.GlobalConnectionManagerAccessingMixin_on_setup()')
        self.try_register_any_control_widgets()
        
    
    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## TODO: unregister children
        print(f'PhoDockAreaContainingWindow.GlobalConnectionManagerAccessingMixin_on_destroy()')
        flat_widgets_list = self.get_flat_widgets_list()
        print(f'\tflat_widgets_list contains {len(flat_widgets_list)} items')
        for a_widget in flat_widgets_list:
            self.connection_man.unregister_object(a_widget, debug_print=False)

            
            
# ==================================================================================================================== #

            
    
class DockAreaWrapper(object):
    """ Responsible for wrapping several children in Dock items and installing them in a central DockArea
    Primary method is DockAreaWrapper.wrap_with_dockAreaWindow(...)
    
    Known Usage:
        ## In DefaultDisplayFunctions._display_3d_interactive_tuning_curves_plotter(...): to combine the ipcDataExplorer and its controlling placefieldControlsContainerWidget into a single window with each widget wrapped in a dock.
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
            # Wrap:
            active_root_main_widget = ipcDataExplorer.p.window()
            root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(active_root_main_widget, placefieldControlsContainerWidget, title=ipcDataExplorer.data_explorer_name)
            pane = (root_dockAreaWindow, placefieldControlsContainerWidget, pf_widgets)
        
    """

    @classmethod
    def build_default_dockAreaWindow(cls, title='_test_PhoDockAreaWidgetApp', defer_show=False) -> Tuple[PhoDockAreaContainingWindow, QtWidgets.QApplication]:
        """ builds a simple PhoDockAreaContainingWindow, empty 
        
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')
        
        """
        win = PhoDockAreaContainingWindow(title=title)
        win.setWindowTitle(f'{title}: dockAreaWindow')
        app = win.app
        
        if not defer_show:
            win.show()
            
        return win, app
    
    
        
    @classmethod
    def wrap_with_dockAreaWindow(cls, main_window, auxilary_controls_window, title='_test_PhoDockAreaWidgetApp') -> Tuple[PhoDockAreaContainingWindow, QtWidgets.QApplication]:
        """ Combine The Separate Windows into a common DockArea window:
        
        
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper

            # active_root_main_widget = ipcDataExplorer.p.parentWidget()
            active_root_main_widget = ipcDataExplorer.p.window()
            win, app = DockAreaWrapper.wrap_with_dockAreaWindow(active_root_main_widget, placefieldControlsContainerWidget)

        """        
        # build a win of type PhoDockAreaContainingWindow
        win, app = cls._build_default_dockAreaWindow(title=title, defer_show=True)
        
        # curr_main_window_size = main_window.size()
        main_win_geom = main_window.window().geometry() # get the QTCore PyRect object
        main_x, main_y, main_width, main_height = main_win_geom.getRect() # Note: dx & dy refer to width and height
        
        if auxilary_controls_window is not None:
            second_win_geom = auxilary_controls_window.window().geometry()
            secondary_x, secondary_y, secondary_width, secondary_height = second_win_geom.getRect() # Note: dx & dy refer to width and height
            
            combined_width = max(main_width, secondary_width)
            combined_height = main_height + secondary_height
            # main_window.size()[0]
            # win.resize(1000,500)
            win.resize(combined_width, combined_height)
        
        
        # Build Using 
        display_config2 = CustomDockDisplayConfig(showCloseButton=False)
        _, dDisplayItem2 = win.add_display_dock("Dock2 - Content", dockSize=(main_width, main_height), widget=main_window, dockAddLocationOpts=['bottom'], display_config=display_config2)
        
        if auxilary_controls_window is not None:
            display_config1 = CustomDockDisplayConfig(showCloseButton=False)
            _, dDisplayItem1 = win.add_display_dock("Dock1 - Controls", dockSize=(secondary_width, secondary_height), widget=auxilary_controls_window, dockAddLocationOpts=['top', dDisplayItem2], display_config=display_config1)

        # _, dDisplayItem1 = win.add_display_dock("Dock1 - Controls", dockSize=(secondary_width, secondary_height), widget=auxilary_controls_window, dockAddLocationOpts=['bottom'])
        win.show()

        if auxilary_controls_window is not None:
            win.area.moveDock(dDisplayItem1, 'top', dDisplayItem2)     ## move d4 to top edge of d2
    
        # dDisplayItem1.hideTitleBar()
        # dDisplayItem2.hideTitleBar()
        
        return win, app

