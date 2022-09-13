from collections import OrderedDict
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.External.pyqtgraph.console import ConsoleWidget

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMainAppWindowBase import PhoMainAppWindowBase

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

# DockAreaWrapper


class DynamicDockDisplayAreaContentMixin:
    """ Conformers are able to dynamically add/remove Dock items and their widgets to the root self.area (a DockArea) item.
    
    Requires at minimum:
        'self.area': a pg.Dock(...) object containing the root items
    
    Creates: 
        self.displayDockArea: a pg.Dock(...) object containing dynamically created Docks/Widgets for display of display nodes.
        
    Known Usages:
        PhoDockAreaContainingWindow **ONLY** right now 
    
    """
    
    @property
    def dynamic_display_dict(self):
        """The dynamic_display_dict property."""
        return self._dynamic_display_output_dict
    @dynamic_display_dict.setter
    def dynamic_display_dict(self, value):
        self._dynamic_display_output_dict = value
    
    @property
    def displayDockArea(self):
        """The displayDockArea property."""
        return self.ui.area
    @displayDockArea.setter
    def displayDockArea(self, value):
        self.ui.area = value
    
    @QtCore.pyqtSlot()
    def DynamicDockDisplayAreaContentMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        self._dynamic_display_output_dict = OrderedDict() # for DynamicDockDisplayAreaContentMixin

    @QtCore.pyqtSlot()
    def DynamicDockDisplayAreaContentMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass


    @QtCore.pyqtSlot()
    def DynamicDockDisplayAreaContentMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        ## TODO: currently temporary
        self.ui.dock_helper_widgets = [] # required for holding references to dynamically created dock_helper_Widgets.
        
    
    @QtCore.pyqtSlot()
    def DynamicDockDisplayAreaContentMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        self.clear_all_display_docks()



    def get_flat_widgets_list(self, debug_print=False):
        """ extracts the 'widget' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        all_collected_widgets = []
        for an_id, an_item in self.dynamic_display_dict.items():
            if debug_print:
                print(f'an_id: {an_id}, an_item: {an_item}')
            for a_sub_id, a_sub_item in an_item.items():
                if debug_print:
                    print(f'\ta_sub_id: {a_sub_id}, a_sub_item: {a_sub_item}')
                a_widget = a_sub_item.get('widget', None)
                all_collected_widgets.append(a_widget)
                
        return all_collected_widgets
    
    def add_display_dock(self, identifier = None, widget = None, dockSize=(300,200), dockIsClosable=True, dockAddLocationOpts=['bottom']):
        """ adds a dynamic display dock with an appropriate widget of type 'viewContentsType' to the dock area container on the main window. """
        # Add the sample display dock items to the nested dynamic display dock:
        display_dock_area = self.displayDockArea
        # curr_display_dock_items = display_dock_area.children()
        # curr_num_display_dock_items = len(curr_display_dock_items)

        if identifier is None:
            identifier = 'item'
        
        extant_group_items = self.dynamic_display_dict.get(identifier, None) # tries to find extant items with this identifier in the dict of extant plots
        if extant_group_items is not None:
            # Item was found with this identifier, implement one of the strategies
            curr_extant_group_item_count = len(extant_group_items)
            unique_identifier = f'{identifier}-{curr_extant_group_item_count}'
        else:
            # no extant items found
            unique_identifier = identifier

        # Build the new dock item:        
        dDisplayItem = Dock(unique_identifier, size=dockSize, closable=dockIsClosable, widget=widget) # add the new display item
        
        if len(dockAddLocationOpts) < 1:
            dockAddLocationOpts = [dDisplayItem, 'bottom']
        elif len(dockAddLocationOpts) == 1:
            if isinstance(dockAddLocationOpts[0], str):
               relative_string = dockAddLocationOpts[0]
               dockAddLocationOpts = [dDisplayItem, relative_string]
            else:
                raise NotImplementedError            
            
        elif len(dockAddLocationOpts) == 2:
            if isinstance(dockAddLocationOpts[0], Dock):
               # starts with the Dock item, add current dock item to the end of the list
               relative_string = dockAddLocationOpts[1]
               relative_dock_item = dockAddLocationOpts[0]
               dockAddLocationOpts = [relative_dock_item, relative_string, dDisplayItem]
            elif isinstance(dockAddLocationOpts[1], Dock):
                relative_string = dockAddLocationOpts[0]
                relative_dock_item = dockAddLocationOpts[1]
                dockAddLocationOpts = [dDisplayItem, relative_string, relative_dock_item]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        # print(f'dockAddLocationOpts: {dockAddLocationOpts}')
        
        # display_dock_area.addDock(dDisplayItem, *dockAddLocationOpts)
        display_dock_area.addDock(*dockAddLocationOpts)
        
        # Set the dock item's widget to the new_view_widget
        # if widget is not None:
        #     dDisplayItem.addWidget(widget)
        
        if extant_group_items is not None:
            # Item was found with this identifier, implement one of the strategies
            extant_group_items[unique_identifier] = {"dock":dDisplayItem, "widget":widget} # add the unique item to the group's dict
            self.dynamic_display_dict[identifier] = extant_group_items # update the extant group's dict
        else:
            self.dynamic_display_dict[identifier] = OrderedDict() # initialize an empty group for the dict
            self.dynamic_display_dict[identifier][unique_identifier] = {"dock":dDisplayItem, "widget":widget}
            
        # self.dynamic_display_dict[identifier] = {"dock":dDisplayItem, "widget":new_view_widget}        
        return widget, dDisplayItem
    
    def find_display_dock(self, identifier):
        """ returns the first found Dock with the specified title equal to the identifier , or None if it doesn't exist. """
        curr_display_dock_items = self.displayDockArea.findChildren(Dock) # find all dock-type children
        for a_dock_item in curr_display_dock_items:
            if a_dock_item.title() == identifier:
                return a_dock_item #found the correct item, return it
        return None # if never found, return None        
        # dock_item_titles = [a_dock_item.title() for a_dock_item in curr_display_dock_items]
        
    
    def rename_display_dock(self, original_identifier, new_identifier):
        """ renames an existing dock. Searches for Dock-type children instead of using the self.dynamic_display_dict because of the nestedness introduced by the nested dock widgets. """
        # extant_group_items = self.dynamic_display_dict.get(original_identifier, None) # tries to find extant items with this identifier in the dict of extant plots
        # assert extant_group_items is not None, f"original_identifier: {original_identifier} -- {self.dynamic_display_dict}"
        extant_dock_item = self.find_display_dock(original_identifier)
        assert extant_dock_item is not None, f"original_identifier: {original_identifier} -- {[a_dock_item.title() for a_dock_item in self.displayDockArea.findChildren(Dock)]}"
        extant_dock_items_widgets = extant_dock_item.widgets # list, e.g. [<pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget.DockPlanningHelperWidget at 0x1d6b2a1b820>]
        # Perform the update:
        extant_dock_item.setTitle(new_identifier)
        ## TODO: update the self.dynamic_display_dict
        
        
    
    def remove_display_dock(self, identifier):
        """ removes a group of dynamic display widgets with identifier 'identifier'. """
        extant_group_items = self.dynamic_display_dict.get(identifier, None) # tries to find extant items with this identifier in the dict of extant plots
        if extant_group_items is not None:
            num_found_group_items = len(extant_group_items)
            if num_found_group_items > 0:
                # Item was found with this identifier
                print(f'Found a group with the identifier {identifier} containing {num_found_group_items} items. Removing all...')
                for (unique_identifier, item_dict) in extant_group_items.items():
                    # loop through the dictionary and remove the children items:
                    # item_dict['widget'].close() # this shouldn't be needed because the 'dock' is the parent, meaning it should properly close the widget as well.
                    item_dict["dock"].close() # close the dock
                    # del extant_group_items[unique_identifier]
                
                # once done with all children, remove the extant_group_items group:
                del self.dynamic_display_dict[identifier]
                
            else:
                # group was found and valid but already empty prior to remove:
                ## TODO: remove group entirely
                del self.dynamic_display_dict[identifier] # remove the empty dict

        else:
            # no extant items found
            print(f'No extant groups/items found with name {identifier}')
            return
        
        
    def clear_all_display_docks(self):
        """ removes all display docks """
        for group_identifier, extant_group_items in self.dynamic_display_dict.items():
            self.remove_display_dock(group_identifier)
            # for unique_identifier in extant_group_items.keys():
            #     self.remove_display_dock(unique_identifier)
        # TODO: Persistance:
        # self.plotDict[name] = {"dock":dock, "widget":widget, "view":view}
    
    
    def on_dock_rename(self, old_name, new_name):
        """ called when a dock item is renamed """
        pass

    def create_planning_helper_dock(self, identifier='New Test Dock Widget', dockAddLocationOpts=['bottom']):
        """ creates a new planning helper dock relative to an existing dock item 
        
        create_planning_helper_dock(identifier='New Test Dock Widget', dockAddLocationOpts=['bottom'])
        """
        test_dock_planning_widget = DockPlanningHelperWidget(dock_title=identifier, dock_id=identifier, defer_show=True) # don't show yet
        test_dock_planning_widget, dDisplayItem = self.add_display_dock(identifier=test_dock_planning_widget.identifier, widget=test_dock_planning_widget, dockAddLocationOpts=dockAddLocationOpts, dockIsClosable=True)
        # connect the helper widget's add relative widget signal to the perform_create_new_relative_dock function
        test_dock_planning_widget.action_create_new_dock.connect(self.perform_create_new_relative_dock) 
        
        return test_dock_planning_widget, dDisplayItem
        
    # @QtCore.pyqtSlot(object, str)
    # def perform_create_new_relative_dock(self, dock_item, relative_position_string):
    @QtCore.pyqtSlot(object, str)
    def perform_create_new_relative_dock(self, calling_widget, relative_position_string):
        """ NOTE: captures win """
        print(f'perform_create_new_relative_dock(calling_widget: {calling_widget}, relative_position_string: {relative_position_string})') ## Getting called with calling_widget == NONE for some reason.
        
        dock_item = calling_widget.embedding_dock_item
        if dock_item is not None:
            returned_helper_widget, returned_dock = self.create_planning_helper_dock(dockAddLocationOpts=[calling_widget, relative_position_string]) # create the new item
        else:
            print(f'WARNING: dock_item is None for {calling_widget}! Creating using ONLY the position string (not relative to the dock item since it cannot be found!')
            returned_helper_widget, returned_dock = self.create_planning_helper_dock(dockAddLocationOpts=[relative_position_string]) # create the new item

        ## TODO: must hold a reference to the returned widgets else they're garbage collected            
        self.ui.dock_helper_widgets.append((returned_helper_widget, returned_dock))
        return returned_helper_widget, returned_dock
 
        
    
# ==================================================================================================================== #
class PhoDockAreaContainingWindow(DynamicDockDisplayAreaContentMixin, PhoMainAppWindowBase):
    """ a custom PhoMainAppWindowBase (QMainWindow) subclass that contains a DockArea as its central view.
    
        Can be used to dynamically create windows composed of multiple separate widgets programmatically.
    
    """
    @property
    def area(self):
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
class NestedDockAreaWidget(DynamicDockDisplayAreaContentMixin, QtWidgets.QWidget):
    """ a custom QWidget subclass that contains a DockArea as its central view and allows adding nested dock items dynamically
    """
    @property
    def area(self):
        return self.ui.area

    def __init__(self, *args, **kwargs):
        # self._app = pg.mkQApp(title) # makes a new QApplication or gets the reference to an existing one.
        self.ui = PhoUIContainer()
        self.DynamicDockDisplayAreaContentMixin_on_init()
        super(NestedDockAreaWidget, self).__init__(*args, **kwargs)
        self.setup()
        self.buildUI()
        

    def setup(self):
        self.ui.area = DockArea()
        # Use self.ui.area as central widget:        
        self.ui.layout = QtWidgets.QGridLayout()
        self.setLayout(self.ui.layout)
        self.ui.layout.addWidget(self.ui.area, 0, 0)        
        self.DynamicDockDisplayAreaContentMixin_on_setup()
        
        
    def buildUI(self):
        self.DynamicDockDisplayAreaContentMixin_on_buildUI()
        
    
    def closeEvent(self, event):
        # Enables closing all secondary windows when this (main) window is closed.
        self.DynamicDockDisplayAreaContentMixin_on_destroy()
        
            
            
            
    
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
    def _build_default_dockAreaWindow(cls, title='_test_PhoDockAreaWidgetApp', defer_show=False):
        """ builds a simple PhoDockAreaContainingWindow """
        win = PhoDockAreaContainingWindow(title=title)
        win.setWindowTitle(f'{title}: dockAreaWindow')
        app = win.app
        
        if not defer_show:
            win.show()
            
        return win, app
    
    
        
    @classmethod
    def wrap_with_dockAreaWindow(cls, main_window, auxilary_controls_window, title='_test_PhoDockAreaWidgetApp'):
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
        
        second_win_geom = auxilary_controls_window.window().geometry()
        secondary_x, secondary_y, secondary_width, secondary_height = second_win_geom.getRect() # Note: dx & dy refer to width and height
        
        combined_width = max(main_width, secondary_width)
        combined_height = main_height + secondary_height
        # main_window.size()[0]
        # win.resize(1000,500)
        win.resize(combined_width, combined_height)
        
        
        # Build Using 
        _, dDisplayItem2 = win.add_display_dock("Dock2 - Content", dockSize=(main_width, main_height), dockIsClosable=False, widget=main_window, dockAddLocationOpts=['bottom'])
        _, dDisplayItem1 = win.add_display_dock("Dock1 - Controls", dockSize=(secondary_width, secondary_height), dockIsClosable=False, widget=auxilary_controls_window, dockAddLocationOpts=['top', dDisplayItem2])
        # _, dDisplayItem1 = win.add_display_dock("Dock1 - Controls", dockSize=(secondary_width, secondary_height), dockIsClosable=False, widget=auxilary_controls_window, dockAddLocationOpts=['bottom'])

        win.show()

        win.area.moveDock(dDisplayItem1, 'top', dDisplayItem2)     ## move d4 to top edge of d2
        # dDisplayItem1.hideTitleBar()
        # dDisplayItem2.hideTitleBar()
        
        return win, app

