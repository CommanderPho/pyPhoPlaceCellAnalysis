from typing import Callable, Optional, Dict, List, Tuple, Union
from collections import OrderedDict
from enum import Enum
from attrs import define, field, Factory
import numpy as np

from neuropy.utils.colors_util import ColorsUtil
from neuropy.utils.mixins.dynamic_conformance_updating_mixin import BaseDynamicInstanceConformingMixin

# import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea

# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

""" 
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig

"""

@define(slots=False)
class DockDisplayColors:
    """ 
    Usage:
    
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig

        dim_config = DockDisplayColors(fg_color='#aaa', bg_color='#44aa44', border_color='#339933')
        regular_config = DockDisplayColors(fg_color='#fff', bg_color='#66cc66', border_color='#54ba54')
        custom_get_colors_dict = {   False: DockDisplayColors(fg_color='#aaa', bg_color='#44aa44', border_color='#339933'),
            True: DockDisplayColors(fg_color='#fff', bg_color='#66cc66', border_color='#54ba54')
        }


    """
    fg_color: str = field(default='#aaa') # Grey
    bg_color: str = field(default='#66cc66') # (120°, 50, 80)
    border_color: str = field(default='#54ba54') # (120°, 55%, 73%)


    @classmethod
    def _subfn_get_random_dock_colors_for_key(cls, key, orientation, is_dim):
        """Generate consistent random colors for a dock based on its key."""
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum

        # Generate a unique background color based on the key
        bg_color = ColorsUtil.generate_unique_hex_color_from_hashable((key, 'bg'))
        
        # Create a darker border color based on the background color
        border_color = DisplayColorsEnum.apply_dock_border_color_adjustment(bg_color)
        
        # Choose a contrasting foreground color (white or black) based on background brightness
        # Simple algorithm: if R+G+B > 384 (out of 765 max), use black text, otherwise white
        r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
        fg_color = '#000' if (r + g + b) > 384 else '#fff'
        
        # Apply dimming if requested
        if is_dim:
            bg_color, fg_color = DisplayColorsEnum.apply_dock_dimming_adjustment(bg_color, fg_color)
            border_color = DisplayColorsEnum.apply_dock_dimming_adjustment(border_color)
        
        return fg_color, bg_color, border_color
    


    @classmethod
    def get_random_dock_colors_for_key_fn(cls, key) -> Callable:
        """Returns a valid `custom_get_colors_callback_fn` that generates consistent random colors for a dock based on its initialization-time key/identity.
        Usage:
            custom_get_colors_callback_fn = DockDisplayColors.get_random_dock_colors_for_key_fn(key=a_key)
        
        """
        from functools import partial
        custom_get_colors_callback_fn = partial(cls._subfn_get_random_dock_colors_for_key, key)
        return custom_get_colors_callback_fn




@define(slots=False)
class CustomDockDisplayConfig(DockDisplayConfig):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc.

    custom_get_colors_callback, if provided, is used to get the colors. This function must be of the form:
        get_colors(self, orientation, is_dim) -> return fg_color, bg_color, border_color
    """
    custom_get_colors_dict: Optional[Dict] = field(default=None)
    _custom_get_colors_callback_fn: Optional[Callable] = field(default=None, alias='custom_get_colors_callback_fn')
    dock_group_names: List[str] = field(default=Factory(list), metadata={'desc': 'a list of conceptual "groups" that the dock specified by this config belongs to. Allows closing, moving, etc multiple docks at a time.'})
    # additional_metadata: Dict = field(default=Factory(dict)) ## optional metadata
    

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
        return (self._orientation or 'auto' )   
    @orientation.setter
    def orientation(self, value):
        self._orientation = value

    def __attrs_post_init__(self):
      if self.custom_get_colors_dict is None:
            self.custom_get_colors_dict = {False: DockDisplayColors(fg_color='#111', bg_color='#66cc66', border_color='#54ba54'),
                True: DockDisplayColors(fg_color='#333', bg_color='#44aa44', border_color='#339933'),
            }

    # def __init__(self, showCloseButton=True, fontSize='10px', corner_radius='2px', custom_get_colors_callback_fn=None, orientation=None, custom_get_colors_dict=None):
    #     super(CustomDockDisplayConfig, self).__init__(showCloseButton=showCloseButton, fontSize=fontSize, corner_radius=corner_radius)
    #     self._custom_get_colors_callback_fn = custom_get_colors_callback_fn
    #     self._orientation = orientation
    #     if custom_get_colors_dict is None:
    #         custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color='#66cc66', border_color='#54ba54'),
    #             True: DockDisplayColors(fg_color='#aaa', bg_color='#44aa44', border_color='#339933'),
    #         }
                        
    #     self.custom_get_colors_dict = custom_get_colors_dict

        

    def get_colors(self, orientation, is_dim):
        if self.custom_get_colors_callback is not None:
            # Use the custom function instead
            return self.custom_get_colors_callback(orientation, is_dim)

        else:
            if self.custom_get_colors_dict is not None:
                ## otherwise use the `self.custom_get_colors_dict`
                active_colors_dict = self.custom_get_colors_dict[is_dim]
                fg_color = active_colors_dict.fg_color
                bg_color = active_colors_dict.bg_color
                border_color = active_colors_dict.border_color      

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
    
    @classmethod
    def build_custom_get_colors_fn(cls, bg_color: str = '#44aa44', border_color: str = '#339933', fg_color: str = '#aaa'):
            """ Builds the custom callback function from some colors:
            
            self.custom_get_colors_callback = CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color='#44aa44', border_color='#339933')
            
            """
            def _subfn_custom_get_colors(self, orientation, is_dim):
                """ captures: bg_color, border_color, fg_color 
                """
                if self.custom_get_colors_callback is not None:
                    # Use the custom function instead
                    return self.custom_get_colors_callback(orientation, is_dim)

                else:
                    # Common to all:
                    if is_dim:
                        fg_color = fg_color or '#aaa' # Grey
                    else:
                        fg_color = fg_color or '#fff' # White
                    
                    # Green-based by default, but custom-color if provided:
                    if is_dim:
                        bg_color = bg_color or '#44aa44' # (120°, 60%, 67%)
                        border_color = border_color or '#339933' # (120°, 67%, 60%)
                    else:
                        bg_color = bg_color or '#66cc66' # (120°, 50, 80)
                        border_color = border_color or '#54ba54' # (120°, 55%, 73%)
                        
                    return fg_color, bg_color, border_color

            ## end def _subfn_custom_get_colors(...)
            return _subfn_custom_get_colors
    

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

@define(slots=False)
class CustomCyclicColorsDockDisplayConfig(CustomDockDisplayConfig):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc.

    custom_get_colors_callback, if provided, is used to get the colors. This function must be of the form:
        get_colors(self, orientation, is_dim) -> return fg_color, bg_color, border_color
    """
    named_color_scheme: NamedColorScheme = field(default=NamedColorScheme.red, alias='named_color_scheme')

    def __attrs_post_init__(self):
      if self.custom_get_colors_dict is None:
            pass # don't update it here, default to None

    # def __init__(self, showCloseButton=True, fontSize='10px', corner_radius='2px', orientation=None, named_color_scheme=NamedColorScheme.red, custom_get_colors_callback_fn=None, custom_get_colors_dict=None):
    #     super(CustomCyclicColorsDockDisplayConfig, self).__init__(showCloseButton=showCloseButton, fontSize=fontSize, corner_radius=corner_radius, orientation=orientation, custom_get_colors_callback_fn=custom_get_colors_callback_fn)
    #     self.named_color_scheme = named_color_scheme
    #     if custom_get_colors_dict is None:
    #         # custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color='#66cc66', border_color='#54ba54'),
    #         #     True: DockDisplayColors(fg_color='#aaa', bg_color='#44aa44', border_color='#339933'),
    #         # }
    #         pass                        
    #     self.custom_get_colors_dict = custom_get_colors_dict
        

    def get_colors(self, orientation, is_dim):
        if self.custom_get_colors_callback is not None:
            # Use the custom function instead
            return self.custom_get_colors_callback(orientation, is_dim)

        else:
            if self.custom_get_colors_dict is not None:
                ## otherwise use the `self.custom_get_colors_dict`
                active_colors_dict = self.custom_get_colors_dict[is_dim]
                fg_color = active_colors_dict.fg_color
                bg_color = active_colors_dict.bg_color
                border_color = active_colors_dict.border_color      

            else:
                # Common to all:
                if is_dim:
                    fg_color = '#aaa' # Grey
                else:
                    fg_color = '#fff' # White

                if self.named_color_scheme.name == NamedColorScheme.blue.name:
                    # Blue/Purple-based:
                    if is_dim:
                        bg_color = '#4444aa' # Dark Blue - (240°, 60, 66.66)
                        border_color = '#339' # More Vibrant Dark Blue - (240°, 66.66, 60)
                    else:
                        bg_color = '#6666cc' # Default Purple Color - (240°, 50, 80)
                        border_color = '#55B' # Similar Purple Color - (240°, 54.54, 73.33)
                elif self.named_color_scheme.name == NamedColorScheme.green.name:
                    # Green-based:
                    if is_dim:
                        bg_color = '#44aa44' # (120°, 60%, 67%)
                        border_color = '#339933' # (120°, 67%, 60%)
                    else:
                        bg_color = '#66cc66' # (120°, 50, 80)
                        border_color = '#54ba54' # (120°, 55%, 73%)
                elif self.named_color_scheme.name == NamedColorScheme.red.name:
                    # Red-based:
                    if is_dim:
                        bg_color = '#aa4444' # (0°, 60%, 67%)
                        border_color = '#993232' # (0°, 67%, 60%)
                    else:
                        bg_color = '#cc6666' # (0°, 50, 80)
                        border_color = '#ba5454' # (0°, 55%, 73%)
                elif self.named_color_scheme.name == NamedColorScheme.grey.name:
                    # Grey-based:
                    if is_dim:
                        bg_color = '#d8d8d8' 
                        border_color = '#717171' 
                    else:
                        bg_color = '#9d9d9d' 
                        border_color = '#3a3a3a' 
                else:
                    raise NotImplementedError
            # END else:
            return fg_color, bg_color, border_color

@define(slots=False)
class FigureWidgetDockDisplayConfig(CustomDockDisplayConfig):
    """docstring for FigureWidgetDockDisplayConfig."""

    def get_colors(self, orientation, is_dim):
        # Common to all:
        if is_dim:
            fg_color = '#aaa' # Grey
        else:
            fg_color = '#fff' # White
            
        # Red-based:
        if is_dim:
            bg_color = '#aa4444' # (0°, 60%, 67%)
            border_color = '#993232' # (0°, 67%, 60%)
        else:
            bg_color = '#cc6666' # (0°, 50, 80)
            border_color = '#ba5454' # (0°, 55%, 73%)
 
        return fg_color, bg_color, border_color
    
    def __attrs_post_init__(self):
      self.fontSize = '10px'
      self.corner_radius = '3px'
      if self.custom_get_colors_dict is None:
            self.custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color='#cc6666', border_color='#ba5454'),
                True: DockDisplayColors(fg_color='#aaa', bg_color='#aa4444', border_color='#993232'),
            }
            




# ==================================================================================================================== #
# BEGIN MAIN MIXIN CLASS                                                                                               #
# ==================================================================================================================== #
class DynamicDockDisplayAreaContentMixin(BaseDynamicInstanceConformingMixin):
    """ Conformers are able to dynamically add/remove Dock items and their widgets to the root self.area (a DockArea) item.
    
    Requires at minimum:
        'self.area': a pg.Dock(...) object containing the root items
    
    Creates: 
        self.displayDockArea: a pg.Dock(...) object containing dynamically created Docks/Widgets for display of display nodes.
        self._dynamic_display_output_dict
        self.ui.dock_helper_widgets
        
        
    Known Usages:
        PhoDockAreaContainingWindow, NestedDockAreaWidget, Spike2DRaster
    
    Signals:
        # DynamicDockDisplayAreaContentMixin Conformance Signals _____________________________________________________________ #
        sigDockAdded = QtCore.Signal(object, object) # (parent, Dock)
        sigDockModified = QtCore.Signal(object, object, object) # (parent, Dock, action)
        sigDockClosed = QtCore.Signal(object, object) # (parent, Dock)

        
    """
    
    @property
    def dynamic_display_dict(self) -> Dict[str, Dict[str, Dict]]:
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
        self._dynamic_display_output_dict = dict() # for DynamicDockDisplayAreaContentMixin

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


    def get_flat_dockitems_list(self, debug_print=False) -> List[Dock]:
        """ extracts the 'dock' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        all_collected_dock_items = []
        for an_id, an_item in self.dynamic_display_dict.items():
            # an_item can be either a Dock itself of a DockGroup
            if debug_print:
                print(f'an_id: {an_id}, an_item: {an_item}')
            for a_sub_id, a_sub_item in an_item.items():
                if debug_print:
                    print(f'\ta_sub_id: {a_sub_id}, a_sub_item: {a_sub_item}')
                a_dock_item = a_sub_item.get('dock', None)
                all_collected_dock_items.append(a_dock_item)
                
        return all_collected_dock_items

    def get_flat_widgets_list(self, debug_print=False) -> List["QtWidgets.QWidget"]:
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
    

    
    def get_flat_dock_identifiers_list(self, debug_print=False) -> List[str]:
        """ extracts the 'dock' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        all_collected_dock_items_identifiers = []
        for an_id, an_item in self.dynamic_display_dict.items():
            if debug_print:
                print(f'an_id: {an_id}, an_item: {an_item}')
            for a_sub_id, a_sub_item in an_item.items():
                if debug_print:
                    print(f'\ta_sub_id: {a_sub_id}, a_sub_item: {a_sub_item}')
                a_dock_item = a_sub_item.get('dock', None)
                all_collected_dock_items_identifiers.append(a_dock_item.name())
        return all_collected_dock_items_identifiers
    
    def get_flat_dock_item_tuple_dict(self, debug_print=False) -> Dict[str, Tuple[Dock, Optional["QtWidgets.QWidget"]]]:
        """ extracts the 'dock' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        # all_collected_dock_items_identifiers = self.get_flat_dock_identifiers_list()
        out_dict = {}
        for an_id, an_item in self.dynamic_display_dict.items():
            for a_sub_id, a_sub_item in an_item.items():
                a_dock_item = a_sub_item.get('dock', None)
                a_widget = a_sub_item.get('widget', None)
                assert (a_sub_id == a_dock_item.name()), f"a_dock_item.name(): '{a_dock_item.name()}' != a_sub_id: '{a_sub_id}'"
                out_dict[a_dock_item.name()] = (a_dock_item, a_widget)
        return out_dict
    
    
    def get_leaf_only_flat_dock_identifiers_list(self) -> List[str]:
        """the dock_identifiers only for the leaf (non-group) items"""
        flat_dock_item_tuple_dict: Dict[str, Tuple[Dock, Optional["QtWidgets.QWidget"]]] = self.get_flat_dock_item_tuple_dict()
        leaf_only_flat_dockwidgets_dict = {k:a_widget for k, (a_dock, a_widget) in flat_dock_item_tuple_dict.items() if ('LEAF' == a_dock.config.additional_metadata.get('type', 'LEAF')) }
        return list(leaf_only_flat_dockwidgets_dict.keys())
    
    def get_group_only_flat_dock_identifiers_list(self) -> List[str]:
        """the dock_identifiers only for the dock-group items"""
        flat_dock_item_tuple_dict: Dict[str, Tuple[Dock, Optional["QtWidgets.QWidget"]]] = self.get_flat_dock_item_tuple_dict()
        group_only_flat_dockwidgets_dict = {k:a_widget for k, (a_dock, a_widget) in flat_dock_item_tuple_dict.items() if ('GROUP' == a_dock.config.additional_metadata.get('type', 'LEAF')) }
        return list(group_only_flat_dockwidgets_dict.keys())
    



    # ==================================================================================================================== #
    # dockGroup                                                                                                            #
    # ==================================================================================================================== #
    # ## Dock all Grouped results from `'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView'`
    # ## INPUTS: active_2d_plot
    # nested_dock_items, nested_dynamic_docked_widget_container_widgets = active_2d_plot.ui.dynamic_docked_widget_container.layout_dockGroups()
    # grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
    # ## OUTPUTS: nested_dock_items, nested_dynamic_docked_widget_container_widgets
    
    @function_attributes(short_name=None, tags=['dockGroup'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=[])
    def get_dockGroup_dock_dict(self, debug_print=False) -> Dict[str, List[Dock]]:
        """ extracts the 'widget' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        flat_dockitems_list = self.get_flat_dockitems_list() ## get the non-grouped dockitems
        grouped_dock_items_dict: Dict[str, List[Dock]] = {}
        # ungrouped_dock_items_list: List[Dock] = []
        for a_dock in flat_dockitems_list:
            ## have a dock
            if len(a_dock.config.dock_group_names) == 0:
                ## ungrouped items
                # grouped_dock_items_dict.append(a_dock)
                pass
            else:
                for a_group_name in a_dock.config.dock_group_names: # a dock can belong to multiple groups
                    if a_group_name not in grouped_dock_items_dict:
                        grouped_dock_items_dict[a_group_name] = [] ## initialize to empty list
                    grouped_dock_items_dict[a_group_name].append(a_dock) ## add the dock to the group
                
        return grouped_dock_items_dict

    @function_attributes(short_name=None, tags=['dockGroup'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=[])
    def get_dockGroup_dock_tree_dict(self, debug_print=False) -> Tuple[List[Union[Dock, Dict[str, List[Dock]]]], Dict]:
        """ extracts the 'widget' property that is the contents of each added dock item from the self.dynamic_display_dict and returns it as a flat list """
        flat_dockitems_list = self.get_flat_dockitems_list()
        tree_out_dock_items_list = []
        non_grouped_dock_items_dict: List[Dock] = []
        
        group_meta_item_dict = {}
        grouped_dock_items_dict: Dict[str, List[Dock]] = {}
        # ungrouped_dock_items_list: List[Dock] = []
        for a_dock in flat_dockitems_list:
            ## have a dock
            if len(a_dock.config.dock_group_names) == 0:
                ## ungrouped items
                parsed_name: str = str(a_dock.name())
                is_group_meta_item: bool = (parsed_name.startswith('GROUP[') and parsed_name.endswith(']'))
                if not is_group_meta_item:
                    non_grouped_dock_items_dict.append(a_dock)
                else:
                    print(f'WARNING: encountered group meta-item for a_dock: {a_dock}, parsed_name: "{parsed_name}", parsed_name: {parsed_name}. Skipping this item')    
                    group_meta_item_identifier: str = (parsed_name.removeprefix('GROUP[').removesuffix(']'))
                    print(f'\tgroup_meta_item_identifier: "{group_meta_item_identifier}"')
                    group_meta_item_dict[group_meta_item_identifier] = a_dock
            else:
                for a_group_name in a_dock.config.dock_group_names: # a dock can belong to multiple groups
                    if a_group_name not in grouped_dock_items_dict:
                        grouped_dock_items_dict[a_group_name] = [] ## initialize to empty list
                    grouped_dock_items_dict[a_group_name].append(a_dock) ## add the dock to the group
                
        tree_out_dock_items_list.extend(non_grouped_dock_items_dict)
        tree_out_dock_items_list.append(grouped_dock_items_dict)
        # for k, v in grouped_dock_items_dict.items():
        #     tree_out_dock_items_list.append(grouped_dock_items_dict)
        return tree_out_dock_items_list, group_meta_item_dict


    @function_attributes(short_name=None, tags=['dockGroup', 'layout', 'sizing'], input_requires=[], output_provides=[], uses=['get_dockGroup_dock_dict', 'build_wrapping_nested_dock_area'], used_by=[], creation_date='2025-02-17 10:26', related_items=[])
    def layout_dockGroups(self):
        """ fetches the dockGroup items and perform layout """
        grouped_dock_items_dict: Dict[str, List[Dock]] = self.get_dockGroup_dock_dict()
        nested_dock_items = {}
        nested_dynamic_docked_widget_container_widgets = {}
        for dock_group_name, flat_group_dockitems_list in grouped_dock_items_dict.items():
            # Skip if this group already has a container
            if hasattr(self, 'nested_dock_items') and (dock_group_name in self.nested_dock_items):
                continue
            # else:
            ## create a new item
            dDisplayItem, nested_dynamic_docked_widget_container = self.build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
            nested_dock_items[dock_group_name] = dDisplayItem # Dock
            nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container # nested_dynamic_docked_widget_container

        # Store the nested items for future reference
        if not hasattr(self, 'nested_dock_items'):
            self.nested_dock_items = {}
        self.nested_dock_items.update(nested_dock_items)

        return nested_dock_items, nested_dynamic_docked_widget_container_widgets
    

    # ==================================================================================================================== #
    # Main Creation/Find/Deletion Functions                                                                                #
    # ==================================================================================================================== #
    
    def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None, **kwargs) -> Tuple["QtWidgets.QWidget", Dock]:
        """ adds a dynamic display dock with an appropriate widget of type 'viewContentsType' to the dock area container on the main window. 

        Input:

        dockAddLocationOpts: []
            Available options are:
            ['bottom', 'top', 'left', 'right', 'above', 'below']. 
                ['above', 'below']: refer to adding in a tab

        """
        # Add the sample display dock items to the nested dynamic display dock:
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


        if display_config is None:
            display_config = CustomDockDisplayConfig()

        if (display_config.orientation is not None):
            if display_config.orientation == 'auto':
                kwargs['autoOrientation'] = True
            else:
                kwargs['autoOrientation'] = False
            
        # Build the new dock item:
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
        self.displayDockArea.addDock(*dockAddLocationOpts)
        
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
            if ((a_dock_item.title() == identifier) or (a_dock_item.name() == identifier)):
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
                print(f'remove_display_dock(identifier="{identifier}"): Found a group with the identifier "{identifier}" containing {num_found_group_items} items. Removing all...')
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
                    print(f'remove_display_dock(identifier="{identifier}"): WARNING: identifier: "{identifier}" not found in dynamic_display_dict.keys(): {list(self.dynamic_display_dict.keys())}')
                except Exception as e:
                    # Unhandled exception
                    raise
                
            else:
                # group was found and valid but already empty prior to remove:
                ## TODO: remove group entirely
                del self.dynamic_display_dict[identifier] # remove the empty dict

        else:
            # no extant items found
            print(f'remove_display_dock(identifier="{identifier}"): No extant groups/items found with name "{identifier}"')
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

    @function_attributes(short_name=None, tags=['docks', 'nested', 'wrapping'], input_requires=[], output_provides=[], uses=['self.add_display_dock(...)', 'NestedDockAreaWidget','CustomDockDisplayConfig'], used_by=['layout_dockGroups'], creation_date='2025-01-14 03:41', related_items=[])        
    def build_wrapping_nested_dock_area(self, flat_group_dockitems_list: List[Dock], dock_group_name: str = 'ContinuousDecode_ - t_bin_size: 0.025'):
        """ 
        Builds a wrapping dock area containing several pre-existing dock items
        
        Usage:
        
            grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
            dock_group_name: str = 'ContinuousDecode_ - t_bin_size: 0.05'
            flat_group_dockitems_list = grouped_dock_items_dict[dock_group_name]
            dDisplayItem, nested_dynamic_docked_widget_container = build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
            nested_dock_items[dock_group_name] = dDisplayItem
            nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container

            
        Example with nesting all dock groups:
        
            ## INPUTS: active_2d_plot
            grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
            nested_dock_items = {}
            nested_dynamic_docked_widget_container_widgets = {}
            for dock_group_name, flat_group_dockitems_list in grouped_dock_items_dict.items():
                dDisplayItem, nested_dynamic_docked_widget_container = active_2d_plot.ui.dynamic_docked_widget_container.build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
                nested_dock_items[dock_group_name] = dDisplayItem
                nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container

            ## OUTPUTS: nested_dock_items, nested_dynamic_docked_widget_container_widgets

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget

        num_child_docks: int = len(flat_group_dockitems_list)
        total_height: float = np.sum([a_dock.height() for a_dock in flat_group_dockitems_list])

        name=f'GROUP[{dock_group_name}]'
        dockSize=(500, total_height)
        dockAddLocationOpts=['bottom']

        display_config = CustomDockDisplayConfig(showCloseButton=True, showCollapseButton=True, showGroupButton=True, showOrientationButton=True, orientation='horizontal', corner_radius='0px', fontSize='15px',
                                                custom_get_colors_dict = {False: DockDisplayColors(fg_color='#5bf', bg_color='#0d001a', border_color='#5467ba'),
                                                                          True: DockDisplayColors(fg_color='#aaa', bg_color='#35265f', border_color='#423399'),
            })
        
        display_config.additional_metadata = {'type': 'GROUP', 'creation_fn': 'build_wrapping_nested_dock_area',            
        }
        
        ## Add the container to hold dynamic matplotlib plot widgets:
        nested_dynamic_docked_widget_container = NestedDockAreaWidget()
        nested_dynamic_docked_widget_container.setObjectName("nested_dynamic_docked_widget_container")
        nested_dynamic_docked_widget_container.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        nested_dynamic_docked_widget_container.setMinimumHeight(total_height)
        nested_dynamic_docked_widget_container.setContentsMargins(0, 0, 0, 0)
        _, dDisplayItem = self.add_display_dock(name, dockSize=dockSize, display_config=display_config, widget=nested_dynamic_docked_widget_container, dockAddLocationOpts=dockAddLocationOpts, autoOrientation=False)
        dDisplayItem.setOrientation('horizontal', force=True)
        dDisplayItem.updateStyle()
        dDisplayItem.update()

        ## Setup children:
        for a_dock in flat_group_dockitems_list:
            a_dock_identifier: str = a_dock.name()
            # a_dock_identifier: str = a_dock.identifier
            print(f'\ta_dock_identifier: "{a_dock_identifier}"')
            ## format nested child docks:
            a_dock.config.showCloseButton = False
            a_dock.config.showCollapseButton = False
            a_dock.config.showGroupButton = False
            a_dock.config.corner_radius='0px'
            a_dock.updateStyle()
            nested_dynamic_docked_widget_container.displayDockArea.addDock(dock=a_dock) ## move the dock items as children to the new container
            
        return dDisplayItem, nested_dynamic_docked_widget_container
    

    def unwrap_docks_in_nested_dock_area(self, dock_group_name: str='ContinuousDecode_0.03'):
        """
        Removes an existing wrapping dock area containing several pre-existing dock items and returns the dock items back to their parent
        
        Reciprocal of `build_wrapping_nested_dock_area`
        
        Usage:
            active_2d_plot.dock_manager_widget.unwrap_docks_in_nested_dock_area(dock_group_name='ContinuousDecode_0.03')

        """
        group_dock_ids_list = self.get_group_only_flat_dock_identifiers_list()
        # leaf_dock_ids_list = self.get_leaf_only_flat_dock_identifiers_list()
        
        group_dock_raw_identifiers_list = [v.lstrip('GROUP[').rstrip(']') for v in group_dock_ids_list] # 'GROUP[ContinuousDecode_0.03]' -> 'ContinuousDecode_0.03'

        assert dock_group_name in group_dock_raw_identifiers_list, f"dock_group_name: '{dock_group_name}' was not found in group_dock_raw_identifiers_list: {group_dock_raw_identifiers_list}"
        grouped_dock_items_dict: Dict[str, List[Dock]] = self.get_dockGroup_dock_dict()
        # {'ContinuousDecode_ - t_bin_size: 0.025': [<Dock ContinuousDecode_long_LR - t_bin_size: 0.025 (65, 200)>,
        #   <Dock ContinuousDecode_long_RL - t_bin_size: 0.025 (65, 200)>,
        #   <Dock ContinuousDecode_short_LR - t_bin_size: 0.025 (65, 200)>,
        #   <Dock ContinuousDecode_short_RL - t_bin_size: 0.025 (65, 200)>],
        #  'ContinuousDecode_0.03': [<Dock DirectionalDecodersDecoded[long_LR]0.03 (65, 200)>,
        #   <Dock DirectionalDecodersDecoded[long_RL]0.03 (65, 200)>,
        #   <Dock DirectionalDecodersDecoded[short_LR]0.03 (65, 200)>,
        #   <Dock DirectionalDecodersDecoded[short_RL]0.03 (65, 200)>]}

        found_group_idx: int = group_dock_raw_identifiers_list.index(dock_group_name)
        assert found_group_idx != -1
        a_group_container_id: str = group_dock_ids_list[found_group_idx]
        a_group_id: str = group_dock_raw_identifiers_list[found_group_idx]
        flat_group_dockitems_list: List[Dock] = grouped_dock_items_dict[a_group_id]

        for a_dock in flat_group_dockitems_list:
            # a_dock_identifier: str = a_dock.name()
            # ## format nested child docks:
            # a_dock.config.showCloseButton = False
            # a_dock.config.showCollapseButton = False
            # a_dock.config.showGroupButton = False
            # a_dock.config.corner_radius='0px'
            # a_dock.updateStyle()
            self.displayDockArea.addDock(dock=a_dock) ## move the dock items as children to the new container
            
        ## remove the group        
        try:
            self.remove_display_dock(identifier=a_group_container_id)
        except KeyError as e:
            # seems to always happen, not sure why
            print(f'unwrap_docks_in_nested_dock_area(dock_group_name: "{dock_group_name}"): WARNING: self.remove_display_dock(identifier: "{a_group_container_id}") failed!')
            pass
        except Exception as e:
            # Unhandled exception
            raise        

        try:
            del self.nested_dock_items[a_group_id] ## remove from the self.nested_dock_items
        except KeyError as e:
            # seems to always happen, not sure why
            print(f'unwrap_docks_in_nested_dock_area(dock_group_name: "{dock_group_name}"): WARNING: del self.nested_dock_items["{a_group_id}"]: "{a_group_id}" not found in nested_dock_items.keys(): {list(self.nested_dock_items.keys())}')
            pass
        except Exception as e:
            # Unhandled exception
            raise



    def unwrap_docks_in_all_nested_dock_area(self):
        """
        Removes all existing wrapping dock areas (groups) and returns the dock items back to their parent
        
        Reciprocal of `build_wrapping_nested_dock_area`
        
        Usage:
            active_2d_plot.dock_manager_widget.unwrap_docks_in_all_nested_dock_area()

        """
        # group_dock_ids_list = self.get_group_only_flat_dock_identifiers_list()
        # leaf_dock_ids_list = self.get_leaf_only_flat_dock_identifiers_list()        
        # group_dock_raw_identifiers_list = [v.lstrip('GROUP[').rstrip(']') for v in group_dock_ids_list] # 'GROUP[ContinuousDecode_0.03]' -> 'ContinuousDecode_0.03'
        grouped_dock_items_dict: Dict[str, List[Dock]] = self.get_dockGroup_dock_dict()

        for a_group_id, a_flat_group_dockitems_list in grouped_dock_items_dict.items():
            a_group_container_id: str = f'GROUP[{a_group_id}]'
            # a_flat_group_dockitems_list: List[Dock]
            for a_dock in a_flat_group_dockitems_list:
                # a_dock_identifier: str = a_dock.name()
                # ## format nested child docks:
                # a_dock.config.showCloseButton = False
                # a_dock.config.showCollapseButton = False
                # a_dock.config.showGroupButton = False
                # a_dock.config.corner_radius='0px'
                # a_dock.updateStyle()
                self.displayDockArea.addDock(dock=a_dock) ## move the dock items as children to the new container
            # END for a_dock in a_flat_group_dockitems_list
            ## remove the group        
            try:
                self.remove_display_dock(identifier=a_group_container_id)
            except KeyError as e:
                # seems to always happen, not sure why
                print(f'unwrap_docks_in_all_nested_dock_area(): WARNING: self.remove_display_dock(identifier: "{a_group_container_id}") failed!')
                pass
            except Exception as e:
                # Unhandled exception
                raise        

            try:
                del self.nested_dock_items[a_group_id] ## remove from the self.nested_dock_items
            except KeyError as e:
                # seems to always happen, not sure why
                print(f'unwrap_docks_in_all_nested_dock_area(): WARNING: del self.nested_dock_items["{a_group_id}"]: "{a_group_id}" not found in nested_dock_items.keys(): {list(self.nested_dock_items.keys())}')
                pass
            except Exception as e:
                # Unhandled exception
                raise
            
        # END for a_group_id, a_flat_group_dockitems_list in grouped_dock_items_dict.items()
    



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
 
        


@metadata_attributes(short_name=None, tags=['widget', 'dock', 'dockarea'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-13 18:21', related_items=['PipelineDynamicDockDisplayAreaMixin'])
class DynamicDockDisplayAreaOwningMixin(BaseDynamicInstanceConformingMixin):
    """A mixin for widgets that own/contain a child widget implementing DynamicDockDisplayAreaContentMixin.
    Provides convenient access to dock management functions by delegating to the child widget.
    
    Requirements:
        - Must implement property `dock_manager_widget` that returns the child widget implementing DynamicDockDisplayAreaContentMixin
        
        
    from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
    from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaOwningMixin, DynamicDockDisplayAreaContentMixin
    
    
    """
    # ==================================================================================================================== #
    # DynamicDockDisplayAreaOwningMixin Conformances                                                                       #
    # ==================================================================================================================== #
    @property 
    def dock_manager_widget(self) -> DynamicDockDisplayAreaContentMixin:
        """Must be implemented by subclasses to return the widget that manages the docks"""
        raise NotImplementedError
        
    def find_display_dock(self, identifier) -> Optional[Dock]:
        """Delegates to child widget's find_display_dock

        Usage:        
            from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget

            active_2d_plot: Spike2DRaster = active_2d_plot
            a_dock = active_2d_plot.find_display_dock(identifier='new_curves_separate_plot')
            widget: PyqtgraphTimeSynchronizedWidget = a_dock.widgets[0]

            widget.getRootPlotItem().setXRange(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time, padding=0) ## global frame
            widget.update(None)

        """
        return self.dock_manager_widget.find_display_dock(identifier)
        
    def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], **kwargs):
        """Delegates to child widget's add_display_dock"""
        return self.dock_manager_widget.add_display_dock(identifier, widget, dockSize, dockAddLocationOpts, **kwargs)
        
    def remove_display_dock(self, identifier):
        """Delegates to child widget's remove_display_dock"""
        return self.dock_manager_widget.remove_display_dock(identifier)
        
    def rename_display_dock(self, original_identifier, new_identifier):
        """Delegates to child widget's rename_display_dock"""
        return self.dock_manager_widget.rename_display_dock(original_identifier, new_identifier)
        
    def clear_all_display_docks(self):
        """Delegates to child widget's clear_all_display_docks"""
        return self.dock_manager_widget.clear_all_display_docks()

    def get_flat_dock_identifiers_list(self, debug_print=False) -> List[str]:
        """Delegates to child widget's get_flat_widgets_list"""
        return self.dock_manager_widget.get_flat_dock_identifiers_list(debug_print=debug_print)


    def get_leaf_only_flat_dock_identifiers_list(self) -> List[str]:
        """Delegates to child widget's get_flat_widgets_list"""
        flat_dock_item_tuple_dict: Dict[str, Tuple] = self.get_flat_dock_item_tuple_dict()
        leaf_only_flat_dockwidgets_dict = {k:a_widget for k, (a_dock, a_widget) in flat_dock_item_tuple_dict.items() if ('LEAF' == a_dock.config.additional_metadata.get('type', 'LEAF')) }
        return list(leaf_only_flat_dockwidgets_dict.keys())
    
    def get_group_only_flat_dock_identifiers_list(self) -> List[str]:
        """Delegates to child widget's get_flat_widgets_list"""
        flat_dock_item_tuple_dict: Dict[str, Tuple] = self.get_flat_dock_item_tuple_dict()
        group_only_flat_dockwidgets_dict = {k:a_widget for k, (a_dock, a_widget) in flat_dock_item_tuple_dict.items() if ('GROUP' == a_dock.config.additional_metadata.get('type', 'LEAF')) }
        return list(group_only_flat_dockwidgets_dict.keys())


    def get_flat_dockitems_list(self, debug_print=False) -> List[Dock]:
        """Delegates to child widget's get_flat_dockitems_list"""
        return self.dock_manager_widget.get_flat_dockitems_list(debug_print=debug_print)

    def get_flat_widgets_list(self, debug_print=False) -> List["QtWidgets.QWidget"]:
        """Delegates to child widget's get_flat_widgets_list"""
        return self.dock_manager_widget.get_flat_widgets_list(debug_print=debug_print)
    
    def get_flat_dock_item_tuple_dict(self, debug_print=False) -> Dict[str, Tuple[Dock, Optional["QtWidgets.QWidget"]]]:
        """ 
        
        
        Usage:
        
        get_flat_dock_item_tuple_dict(
        """
        return self.dock_manager_widget.get_flat_dock_item_tuple_dict(debug_print=debug_print)
    
    def find_dock_item_tuple(self, identifier) -> Tuple[Dock, Optional["QtWidgets.QWidget"]]:
        """ returns a tuple containing the dockItem and its main widget
        
        Usage:        
            from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget

            active_2d_plot: Spike2DRaster = active_2d_plot
            a_dock, widget = active_2d_plot.find_dock_item_tuple(identifier='new_curves_separate_plot')
            widget.getRootPlotItem().setXRange(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time, padding=0) ## global frame
            widget.update(None)

        """
        return self.dock_manager_widget.get_flat_dock_item_tuple_dict()[identifier]


    def get_dockGroup_dock_dict(self, debug_print=False) -> Dict[str, List[Dock]]:
        """Delegates to child widget's get_dockGroup_dock_dict"""
        return self.dock_manager_widget.get_dockGroup_dock_dict(debug_print)

    # ==================================================================================================================== #
    # dockGroups                                                                                                           #
    # ==================================================================================================================== #
    def dissolve_all_dockGroups(self):
        """ fetches the dockGroup items and perform layout """
        return self.dock_manager_widget.unwrap_docks_in_all_nested_dock_area()
    
    def layout_dockGroups(self):
        """ fetches the dockGroup items and perform layout """
        return self.dock_manager_widget.layout_dockGroups()
    