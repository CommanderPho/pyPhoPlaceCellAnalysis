from collections import OrderedDict
# from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget


class CustomDockDisplayConfig(DockDisplayConfig):
    """docstring for DockDisplayConfig."""
    def __init__(self, showCloseButton=True, fontSize='12px', corner_radius='3px'):
        super(CustomDockDisplayConfig, self).__init__(showCloseButton=showCloseButton, fontSize=fontSize, corner_radius=corner_radius)

    def get_colors(self, orientation, is_dim):
        # Common to all:
        if is_dim:
            fg_color = '#aaa' # Grey
        else:
            fg_color = '#fff' # White
        
        # # Blue/Purple-based:
        # if is_dim:
        #     bg_color = '#4444aa' # Dark Blue - (240°, 60, 66.66)
        #     border_color = '#339' # More Vibrant Dark Blue - (240°, 66.66, 60)
        # else:
        #     bg_color = '#6666cc' # Default Purple Color - (240°, 50, 80)
        #     border_color = '#55B' # Similar Purple Color - (240°, 54.54, 73.33)
            
        # Green-based:
        if is_dim:
            bg_color = '#44aa44' # (120°, 60%, 67%)
            border_color = '#339933' # (120°, 67%, 60%)
        else:
            bg_color = '#66cc66' # (120°, 50, 80)
            border_color = '#54ba54' # (120°, 55%, 73%)
            
        # # Red-based:
        # if is_dim:
        #     bg_color = '#aa4444' # (0°, 60%, 67%)
        #     border_color = '#993232' # (0°, 67%, 60%)
        # else:
        #     bg_color = '#cc6666' # (0°, 50, 80)
        #     border_color = '#ba5454' # (0°, 55%, 73%)
 
        return fg_color, bg_color, border_color
    

    # def get_stylesheet(self, orientation, is_dim):
    #     """ Gets the appropriate stylesheet for the given state. This method can be overriden to customize the appearance 
        
    #     Usage:
    #         updated_stylesheet = config.get_stylesheet(self, orientation=self.orientation, is_dim=self.dim)
            
    #     """ 
    #     fg_color, bg_color, border_color = self.get_colors(orientation, is_dim)

    #     if orientation == 'vertical':
    #         return """DockLabel {
    #             background-color : %s;
    #             color : %s;
    #             border-top-right-radius: 0px;
    #             border-top-left-radius: %s;
    #             border-bottom-right-radius: 0px;
    #             border-bottom-left-radius: %s;
    #             border-width: 0px;
    #             border-right: 2px solid %s;
    #             padding-top: 3px;
    #             padding-bottom: 3px;
    #             font-size: %s;
    #         }""" % (bg_color, fg_color, self.corner_radius, self.corner_radius, border_color, self.fontSize)
            
    #     else:
    #         return """DockLabel {
    #             background-color : %s;
    #             color : %s;
    #             border-top-right-radius: %s;
    #             border-top-left-radius: %s;
    #             border-bottom-right-radius: 0px;
    #             border-bottom-left-radius: 0px;
    #             border-width: 0px;
    #             border-bottom: 2px solid %s;
    #             padding-left: 3px;
    #             padding-right: 3px;
    #             font-size: %s;
    #         }""" % (bg_color, fg_color, self.corner_radius, self.corner_radius, border_color, self.fontSize)

    


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
    
    def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None):
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
        # dDisplayItem = Dock(unique_identifier, size=dockSize, closable=dockIsClosable, widget=widget, display_config=CustomDockDisplayConfig()) # add the new display item
        
        if display_config is None:
            display_config = CustomDockDisplayConfig()
            
        dDisplayItem = Dock(unique_identifier, size=dockSize, widget=widget, display_config=display_config) # add the new display item
        
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
        
        display_config = CustomDockDisplayConfig(showCloseButton=True)
        test_dock_planning_widget = DockPlanningHelperWidget(dock_title=identifier, dock_id=identifier, defer_show=True) # don't show yet
        test_dock_planning_widget, dDisplayItem = self.add_display_dock(identifier=test_dock_planning_widget.identifier, widget=test_dock_planning_widget, dockAddLocationOpts=dockAddLocationOpts, display_config=display_config)
        # connect the helper widget's add relative widget signal to the perform_create_new_relative_dock function
        test_dock_planning_widget.action_create_new_dock.connect(self.perform_create_new_relative_dock) 
        
        return test_dock_planning_widget, dDisplayItem
        
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
 
        