from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
from enum import Enum

# import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea

# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

""" 
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig

"""

class CustomDockDisplayConfig(DockDisplayConfig):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc.

    custom_get_colors_callback, if provided, is used to get the colors. This function must be of the form:
        get_colors(self, orientation, is_dim) -> return fg_color, bg_color, border_color
    """

    @property
    def custom_get_colors_callback(self):
        """The custom_get_colors_callback property."""
        return self._custom_get_colors_callback_fn
    @custom_get_colors_callback.setter
    def custom_get_colors_callback(self, value):
        self._custom_get_colors_callback_fn = value


    @property
    def orientation(self) -> str:
        """The orientation property."""
        return self._orientation or 'auto'    
    @orientation.setter
    def orientation(self, value):
        self._orientation = value

    def __init__(self, showCloseButton=True, fontSize='12px', corner_radius='3px', custom_get_colors_callback_fn=None, orientation=None):
        super(CustomDockDisplayConfig, self).__init__(showCloseButton=showCloseButton, fontSize=fontSize, corner_radius=corner_radius)
        self._custom_get_colors_callback_fn = custom_get_colors_callback_fn
        self._orientation = orientation


    def get_colors(self, orientation, is_dim):
        if self.custom_get_colors_callback is not None:
            # Use the custom function instead
            return self.custom_get_colors_callback(orientation, is_dim)

        else:
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


## Build Dock Widgets:
def get_utility_dock_colors(orientation, is_dim):
    """ used for CustomDockDisplayConfig for non-specialized utility docks """
    # Common to all:
    if is_dim:
        fg_color = '#aaa' # Grey
    else:
        fg_color = '#fff' # White
        
    # a purplish-royal-blue 
    if is_dim:
        bg_color = '#d8d8d8' 
        border_color = '#717171' 
    else:
        bg_color = '#9d9d9d' 
        border_color = '#3a3a3a' 

    return fg_color, bg_color, border_color

    
NamedColorScheme = Enum('NamedColorScheme', 'blue green red grey')
# NamedColorScheme.blue  # returns <Animal.ant: 1>
# NamedColorScheme['blue']  # returns <Animal.ant: 1> (string lookup)
# NamedColorScheme.blue.name  # returns 'ant' (inverse lookup)

class CustomCyclicColorsDockDisplayConfig(CustomDockDisplayConfig):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc.

    custom_get_colors_callback, if provided, is used to get the colors. This function must be of the form:
        get_colors(self, orientation, is_dim) -> return fg_color, bg_color, border_color
    """
    @property
    def named_color_scheme(self):
        """The named_color_scheme property."""
        return self._named_color_scheme
    @named_color_scheme.setter
    def named_color_scheme(self, value):
        self._named_color_scheme = value
    
    def __init__(self, showCloseButton=True, fontSize='12px', corner_radius='3px', orientation=None, named_color_scheme=NamedColorScheme.red):
        super(CustomCyclicColorsDockDisplayConfig, self).__init__(showCloseButton=showCloseButton, fontSize=fontSize, corner_radius=corner_radius, orientation=orientation)
        self._named_color_scheme = named_color_scheme

    def get_colors(self, orientation, is_dim):
        # Common to all:
        if is_dim:
            fg_color = '#aaa' # Grey
        else:
            fg_color = '#fff' # White

        if self._named_color_scheme.name == NamedColorScheme.blue.name:
            # Blue/Purple-based:
            if is_dim:
                bg_color = '#4444aa' # Dark Blue - (240°, 60, 66.66)
                border_color = '#339' # More Vibrant Dark Blue - (240°, 66.66, 60)
            else:
                bg_color = '#6666cc' # Default Purple Color - (240°, 50, 80)
                border_color = '#55B' # Similar Purple Color - (240°, 54.54, 73.33)
        elif self._named_color_scheme.name == NamedColorScheme.green.name:
            # Green-based:
            if is_dim:
                bg_color = '#44aa44' # (120°, 60%, 67%)
                border_color = '#339933' # (120°, 67%, 60%)
            else:
                bg_color = '#66cc66' # (120°, 50, 80)
                border_color = '#54ba54' # (120°, 55%, 73%)
        elif self._named_color_scheme.name == NamedColorScheme.red.name:
            # Red-based:
            if is_dim:
                bg_color = '#aa4444' # (0°, 60%, 67%)
                border_color = '#993232' # (0°, 67%, 60%)
            else:
                bg_color = '#cc6666' # (0°, 50, 80)
                border_color = '#ba5454' # (0°, 55%, 73%)
        elif self._named_color_scheme.name == NamedColorScheme.grey.name:
            # Grey-based:
            if is_dim:
                bg_color = '#d8d8d8' 
                border_color = '#717171' 
            else:
                bg_color = '#9d9d9d' 
                border_color = '#3a3a3a' 
        else:
            raise NotImplementedError

        return fg_color, bg_color, border_color




class DynamicDockDisplayAreaContentMixin:
    """ Conformers are able to dynamically add/remove Dock items and their widgets to the root self.area (a DockArea) item.
    
    Requires at minimum:
        'self.area': a pg.Dock(...) object containing the root items
    
    Creates: 
        self.displayDockArea: a pg.Dock(...) object containing dynamically created Docks/Widgets for display of display nodes.
        
    Known Usages:
        PhoDockAreaContainingWindow, NestedDockAreaWidget, Spike2DRaster
    
    """
    
    @property
    def dynamic_display_dict(self) -> OrderedDict:
        """The dynamic_display_dict property."""
        return self._dynamic_display_output_dict
    @dynamic_display_dict.setter
    def dynamic_display_dict(self, value):
        self._dynamic_display_output_dict = value
    
    @property
    def displayDockArea(self) -> "DockArea":
        """The displayDockArea property."""
        return self.ui.area
    @displayDockArea.setter
    def displayDockArea(self, value):
        self.ui.area = value
    
    @pyqtExceptionPrintingSlot()
    def DynamicDockDisplayAreaContentMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        self._dynamic_display_output_dict = OrderedDict() # for DynamicDockDisplayAreaContentMixin

    @pyqtExceptionPrintingSlot()
    def DynamicDockDisplayAreaContentMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass


    @pyqtExceptionPrintingSlot()
    def DynamicDockDisplayAreaContentMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        ## TODO: currently temporary
        self.ui.dock_helper_widgets = [] # required for holding references to dynamically created dock_helper_Widgets.
        
    
    @pyqtExceptionPrintingSlot()
    def DynamicDockDisplayAreaContentMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        self.clear_all_display_docks()


    def get_flat_dockitems_list(self, debug_print=False):
        """ extracts the 'dock' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        all_collected_dock_items = []
        for an_id, an_item in self.dynamic_display_dict.items():
            if debug_print:
                print(f'an_id: {an_id}, an_item: {an_item}')
            for a_sub_id, a_sub_item in an_item.items():
                if debug_print:
                    print(f'\ta_sub_id: {a_sub_id}, a_sub_item: {a_sub_item}')
                a_dock_item = a_sub_item.get('dock', None)
                all_collected_dock_items.append(a_dock_item)
                
        return all_collected_dock_items

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
    
    def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None, **kwargs):
        """ adds a dynamic display dock with an appropriate widget of type 'viewContentsType' to the dock area container on the main window. 

        Input:

        dockAddLocationOpts: []
            Available options are:
            ['bottom', 'top', 'left', 'right', 'above', 'below']. 
                ['above', 'below']: refer to adding in a tab

        """
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

        if (display_config.orientation is not None):
            if display_config.orientation == 'auto':
                kwargs['autoOrientation'] = True
            else:
                kwargs['autoOrientation'] = False
            
        # {'autoOrientation':True}
        dDisplayItem = Dock(unique_identifier, size=dockSize, widget=widget, display_config=display_config, **kwargs) # add the new display item
        if isinstance(dockAddLocationOpts, str):
            print(f'WARN: dockAddLocationOpts should be a tuple containing a string (like `("left", )`), not a string itself! Interpretting dockAddLocationOpts: "{dockAddLocationOpts}" as `dockAddLocationOpts = ("{dockAddLocationOpts}", )`')
            dockAddLocationOpts = (dockAddLocationOpts, )
            assert isinstance(dockAddLocationOpts, tuple)
    
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
        

        if (display_config.orientation is not None) and (display_config.orientation != 'auto'):
            assert display_config.orientation in ['horizontal', 'vertical'], f"display_config.orientation should be either ['horizontal', 'vertical'] but display_config.orientation: '{display_config.orientation}'"
            dDisplayItem.setOrientation(o=display_config.orientation, force=True)

        if extant_group_items is not None:
            # Item was found with this identifier, implement one of the strategies
            extant_group_items[unique_identifier] = {"dock":dDisplayItem, "widget":widget} # add the unique item to the group's dict
            self.dynamic_display_dict[identifier] = extant_group_items # update the extant group's dict
        else:
            self.dynamic_display_dict[identifier] = OrderedDict() # initialize an empty group for the dict
            self.dynamic_display_dict[identifier][unique_identifier] = {"dock":dDisplayItem, "widget":widget}
            

        ## Respond to the close signal so that we can remove the item from the dynamic_display_dict when it is closed.
        dDisplayItem.sigClosed.connect(self.on_dock_closed)


        # self.dynamic_display_dict[identifier] = {"dock":dDisplayItem, "widget":new_view_widget}        
        return widget, dDisplayItem
    
    def find_display_dock(self, identifier) -> Optional[Dock]:
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
                try:
                    del self.dynamic_display_dict[identifier]
                except KeyError as e:
                    """ 
                    on_dock_closed(closing_dock: <Dock long_LR (430, 700)>)
                        closing_dock_identifier: long_LR
                        found by simple title identifier and removed!
                    Uncaught Exception in slot
                    Traceback (most recent call last):
                    line 260, in DynamicDockDisplayAreaContentMixin_on_destroy
                        self.clear_all_display_docks()
                    line 429, in clear_all_display_docks
                        self.remove_display_dock(group_identifier)
                    line 413, in remove_display_dock
                        del self.dynamic_display_dict[identifier]
                    KeyError: 'long_LR'
                    """
                    # seems to always happen, not sure why
                    print(f'WARNING: identifier: {identifier} not found in dynamic_display_dict.keys(): {list(self.dynamic_display_dict.keys())}')
                except Exception as e:
                    # Unhandled exception
                    raise e
                
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
        
    @pyqtExceptionPrintingSlot(object, str)
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
 

    @pyqtExceptionPrintingSlot(object)
    def on_dock_closed(self, closing_dock_item):
        print(f'on_dock_closed(closing_dock: {closing_dock_item})') ## Getting called with calling_widget == NONE for some reason.
        closing_dock_identifier = closing_dock_item.title()
        print(f'\t closing_dock_identifier: {closing_dock_identifier}')

        _removed_item = self.dynamic_display_dict.pop(closing_dock_identifier, None)
        if _removed_item is not None:
            # Try to find simply by the title identifier
            print('\t found by simple title identifier and removed!')
            return
        else:
            # Continue searching for item
            found_id = None
            found_sub_id = None
            for an_id, an_item in self.dynamic_display_dict.items():
                for a_sub_id, a_sub_item in an_item.items():
                    a_dock_item = a_sub_item.get('dock', None)
                    if a_dock_item == closing_dock_item:
                        # Found!
                        found_id = an_id
                        found_sub_id = a_sub_id
                        print(f'\t FOUND closing dock item through more complex search! found_id: {found_id}, found_sub_id: {found_sub_id}')
                        print(f'\t removing item...')
                        _found_item = self.dynamic_display_dict[found_id].pop(found_sub_id, None)
                        if len(self.dynamic_display_dict[found_id]) < 1:
                            # if the item is now empty, remove the entire item
                            self.dynamic_display_dict.pop(found_id, None)
                        print(f'done.')
                        return
            print(f'\t WARNING: searched all items and could not find the closing_dock_item!!')
 
        