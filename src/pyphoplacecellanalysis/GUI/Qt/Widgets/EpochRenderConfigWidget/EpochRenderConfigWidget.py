# EpochRenderConfigWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\EpochRenderConfigWidget\EpochRenderConfigWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import sys
import os
from typing import Optional, List, Dict, Callable, Union

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic
from pyphoplacecellanalysis.External.pyqtgraph.widgets.LayoutWidget import LayoutWidget
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## IMPORTS:
from attrs import define, field, Factory
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.Param_to_PyQt_Binding import ParamToPyQtBinding
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig, _get_default_epoch_configs
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp
# from pyphoplacecellanalysis.External.pyqtgraph.widgets.ColorButton import ColorButton

# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'EpochRenderConfigWidget.ui')




# @define(slots=False, auto_detect=True) # , init=False
@metadata_attributes(short_name=None, tags=['epoch', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 00:00', related_items=[])
class EpochRenderConfigWidget(pg.Qt.QtWidgets.QWidget):
    """ a widget that allows graphically configuring a single data series for rendering Epoch rectangles 
        EpochDisplayConfig
    """
    sigConfigChanged = QtCore.Signal(object)
    sigRemoveRequested = QtCore.Signal(object)  # New signal for remove requests
    
    # sigCollapseClicked = QtCore.Signal(object)
    # sigGroupClicked = QtCore.Signal(object)
    config: EpochDisplayConfig = field(default=Factory(EpochDisplayConfig))
        
    ## Attrs manual __init__
    def __init__(self, config: Optional[EpochDisplayConfig]=None, parent=None): # 
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.config = config or EpochDisplayConfig()
        # self.__attrs_init__(EpochDisplayConfig()) # Call the __attrs_init__(...) generated by attrs
        self.initUI()
        self.show() # Show the GUI        


    def initUI(self):
        self.update_from_config(self.config)
        ## TODO: Setup Bindings:
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        

        ## Setup connections:
        self.ui.connections = {}
        self.ui.connections['chkbtnVisible'] = self.ui.chkbtnVisible.toggled.connect(self.on_update_config)    
        self.ui.connections['btnFillColor'] = self.ui.btnFillColor.sigColorChanging.connect(self.on_update_config)
        self.ui.connections['btnPenColor'] = self.ui.btnPenColor.sigColorChanging.connect(self.on_update_config)
        self.ui.connections['doubleSpinBoxHeight'] = self.ui.doubleSpinBoxHeight.valueChanged.connect(self.on_update_config)
        self.ui.connections['doubleSpinBoxOffset'] = self.ui.doubleSpinBoxOffset.valueChanged.connect(self.on_update_config)
        self.ui.connections['btnTitle'] = self.ui.btnTitle.pressed.connect(self.on_update_config)
        
        # for a_config_property, a_widget in zip(bound_config_value_list, ui_element_list):
        #     a_widget_type = type(a_widget)
        #     # print(f'a_widget_type: {a_widget_type.__name__}')
        #     # found_binding = param_to_pyqt_binding_dict.get(type(a_widget), None)
        #     found_binding = param_to_pyqt_binding_dict.get(type(a_widget).__name__, None)
        #     if found_binding is not None:
        #         # print(f'found_binding: {found_binding}')
        #         desired_value = a_config_property(self.config)
        #         # print(f'\t{desired_value}')
        #         # curr_value = found_binding.get_value(a_widget)
        #         # print(f'\t{curr_value}')
        #         found_binding.set_value(a_widget, desired_value)
        #     else:
        #         print(f'no binding for {a_widget} of type: {type(a_widget)}')

        self.setupContextMenu()  # Add this line



    def get_ui_element_list(self):
        return [self.ui.btnTitle, self.ui.btnPenColor, self.ui.btnFillColor, self.ui.doubleSpinBoxHeight, self.ui.doubleSpinBoxOffset, self.ui.chkbtnVisible]

    def get_bound_config_value_list(self):
        return [lambda a_config: a_config.name, lambda a_config: a_config.pen_QColor, lambda a_config: a_config.brush_QColor, lambda a_config: a_config.height, lambda a_config: a_config.y_location, lambda a_config: a_config.isVisible]

    def get_config_property_names_list(self) -> List[str]:
        return ['name', 'pen_QColor', 'brush_QColor', 'height', 'y_location', 'isVisible']


    def on_update_config(self, *args, **kwargs):
        print(f'EpochRenderConfigWidget.on_update_config(*args: {args}, **kwargs: {kwargs})')
        self.sigConfigChanged.emit(self)
        

    ## Programmatic Update/Retrieval:    
    def update_from_config(self, config: EpochDisplayConfig):
        """ called to programmatically update the config """
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        config_property_names_list = self.get_config_property_names_list()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        
        for a_config_property_name, a_config_property, a_widget in zip(config_property_names_list, bound_config_value_list, ui_element_list):
            a_widget_type = type(a_widget)
            # print(f'a_widget_type: {a_widget_type.__name__}')
            # found_binding = param_to_pyqt_binding_dict.get(type(a_widget), None)
            found_binding = param_to_pyqt_binding_dict.get(type(a_widget).__name__, None)
            if found_binding is not None:
                # print(f'found_binding: {found_binding}')
                desired_value = a_config_property(config)
                # print(f'\t{desired_value}')
                # curr_value = found_binding.get_value(a_widget)
                # print(f'\t{curr_value}')
                found_binding.set_value(a_widget, desired_value)
            else:
                print(f'no binding for {a_widget} of type: {type(a_widget)}')
                

    def config_from_state(self) -> EpochDisplayConfig:
        """ called to retrieve a valid config from the UI's properties... this means it could have just held a config as its model. """
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        config_property_names_list = self.get_config_property_names_list()

        a_config = deepcopy(self.config)
        
        for a_config_property_name, a_config_property, a_widget in zip(config_property_names_list, bound_config_value_list, ui_element_list):
            found_binding = param_to_pyqt_binding_dict.get(type(a_widget).__name__, None)
            # print(f'a_config_property: {a_config_property}, a_widget: {a_widget}')
            if found_binding is not None:
                # print(f'found_binding: {found_binding}')
                # desired_value = a_config_property(config)
                # print(f'\t{desired_value}')
                curr_value = found_binding.get_value(a_widget)
                # print(f'\tcurr_value: {curr_value}')
                # print(f'\ta_config_property(a_config): {a_config_property(a_config)}')
                did_change: bool = (a_config_property(a_config) != curr_value)
                if did_change:
                    # print(f'\t value changed!')
                    setattr(a_config, a_config_property_name, curr_value)
                    # a_config[a_config_property_name] = curr_value
                    # a_config_property(a_config) = curr_value
                    # found_binding.set_value(a_widget, curr_value)

                    # a_config[
                # a_config_property(a_config) = curr_value # update to current value
            else:
                print(f'\tno binding for {a_widget} of type: {type(a_widget)}')



        # if self.enable_debug_print:
        #     print(f'config_from_state(...): name={self.name}, isVisible={self.isVisible}, color={self.color}, spikesVisible={self.spikesVisible}')
        #     print(f'\tself.color: {self.color} - self.color.name(): {self.color.name()}')
        
        # How to convert a QColor into a HexRGB String:
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        # color_hex_str = self.color.name(QtGui.QColor.HexRgb) 
        # if self.enable_debug_print:
        #     print(f'\thex: {color_hex_str}')
        
        # also I think the original pf name was formed by adding crap...
        ## see 
        # ```curr_pf_string = f'pf[{render_config.name}]'````
        ## UPDATE: this doesn't seem to be a problem. The name is successfully set to the ACLU value in the current state.
        # return SingleNeuronPlottingExtended(name=self.name, isVisible=self.isVisible, color=color_hex_str, spikesVisible=self.spikesVisible)
        return a_config

    # ================================================================================================================================================================================ #
    # Context Menu Methods                                                                                                                                                             #
    # ================================================================================================================================================================================ #
    def setupContextMenu(self):
        """Setup the context menu for the widget"""
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        
        # Create the action if it's not already part of the UI
        if not hasattr(self.ui, 'actionRemoveEpochSeries'):
            self.ui.actionRemoveEpochSeries = QtWidgets.QAction("Remove Epoch Series", self)
            # You can set the icon if available in your resources
            # self.ui.actionRemoveEpochSeries.setIcon(QtGui.QIcon(":/Icons/Icons/actions/pencil--minus.png"))
        
        # Connect the action to a slot
        self.ui.actionRemoveEpochSeries.triggered.connect(self.onRemoveEpochSeries)
    
    def showContextMenu(self, position):
        """Show the context menu at the requested position"""
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.ui.actionRemoveEpochSeries)
        menu.exec_(self.mapToGlobal(position))
    
    def onRemoveEpochSeries(self):
        """Handler for the remove action"""
        self.sigRemoveRequested.emit(self)
        



def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()



# @define(slots=False, auto_detect=True) # , init=False
@metadata_attributes(short_name=None, tags=['epoch', 'widget'], input_requires=[], output_provides=[], uses=['EpochRenderConfigWidget'], used_by=[], creation_date='2023-10-17 00:00', related_items=[])
class EpochRenderConfigsListWidget(pg.Qt.QtWidgets.QWidget):
    """ a widget that contains a vertical list of `EpochRenderConfigWidget`. Allowing graphically configuring the rendering Epochs 

    History: based off of function `build_containing_epoch_display_configs_root_widget`


    from pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget import EpochRenderConfigsListWidget

    
    """
    sigAnyConfigChanged = QtCore.Signal(object)
    # sigSpecificConfigChanged = QtCore.Signal(object, object)
    
    ui: PhoUIContainer = field(init=False, default=None)
    configs: Dict[str, Union[EpochDisplayConfig, List[EpochDisplayConfig]]] = field(init=False, default=None)
    # out_render_config_widgets_dict: Dict[str, pg.Qt.QtWidgets.QWidget] = field(init=False, default=None)

    ## Attrs manual __init__
    def __init__(self, configs: Optional[Dict[str, Union[EpochDisplayConfig, List[EpochDisplayConfig]]]]=None, parent=None): # 
        self.ui = PhoUIContainer()
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.configs = configs or {}
        # self.out_render_config_widgets_dict = 
        if parent is not None:
            parent.addWidget(self)
        self.initUI()


    def _build_children_widgets(self, configs: Union[EpochDisplayConfig, List[EpochDisplayConfig]]):
        """
        updates:
            self.ui.config_widget_layout
            self.ui.out_render_config_widgets_dict

        """
        assert self.ui.config_widget_layout is not None
        assert self.ui.out_render_config_widgets_dict is not None
        assert len(self.ui.out_render_config_widgets_dict) == 0

        added_widget_dict = {}
        _connections_dict = {}
        
        # self.ui.out_render_config_widgets_dict = {}
        for a_config_name, a_config in configs.items():
            if isinstance(a_config, (list, tuple)):
                # a list of configs
                if len(a_config) == 1:
                    curr_widget = build_single_epoch_display_config_widget(a_config[0]) # 
                    self.ui.config_widget_layout.addWidget(curr_widget)
                    self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget
                    added_widget_dict[a_config[0]] = [curr_widget]
                    
                else:
                    ## extract all items
                    a_sub_config_widget_layout = pg.Qt.QtWidgets.QHBoxLayout()
                    a_sub_config_widget_layout.setSpacing(0)
                    a_sub_config_widget_layout.setContentsMargins(0, 0, 0, 0)
                    a_sub_config_widget_layout.setObjectName(f"horizontalLayout[{a_config_name}]")
                    
                    self.ui.out_render_config_widgets_dict[a_config_name] = [] # start with an empty list
                    # added_widget_dict[a_config] = []


                    for i, a_sub_config in enumerate(a_config):
                        a_sub_curr_widget = build_single_epoch_display_config_widget(a_sub_config)
                        a_sub_curr_widget.setObjectName(f"config[{a_config_name}][{i}]")
                        a_sub_config_widget_layout.addWidget(a_sub_curr_widget)
                        # self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget
                        self.ui.out_render_config_widgets_dict[a_config_name].append(a_sub_curr_widget)
                        # added_widget_dict[a_config].append(a_sub_curr_widget)
                        added_widget_dict[a_sub_config] = [a_sub_curr_widget]

                    # curr_widget = a_sub_config_widget_layout
                    self.ui.config_widget_layout.addLayout(a_sub_config_widget_layout)

            else:
                # Otherwise a straight-up config
                curr_widget = build_single_epoch_display_config_widget(a_config)
                self.ui.config_widget_layout.addWidget(curr_widget)
                self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget
                added_widget_dict[a_config] = [curr_widget]

            ## Connect signals to widgets
            for a_config, a_widget_list in added_widget_dict.items():
                _connections_dict[a_config] = []
                for a_widget in a_widget_list:
                    # self.ui.out_render_config_widgets_dict[a_config_name]
                    _connections_dict[a_config].append(a_widget.sigConfigChanged.connect(self.on_config_ui_updated))
                    _connections_dict[a_config].append(a_widget.sigRemoveRequested.connect(self.on_remove_epoch_series))
                    

    def initUI(self):
        # self.ui.rootWidget = pg.Qt.QtWidgets.QWidget()
        # self.addWidget(self.ui.rootWidget)
        self.ui.connections = {}
        self.ui.config_widget_layout = pg.Qt.QtWidgets.QVBoxLayout()
        self.ui.config_widget_layout.setSpacing(0)
        self.ui.config_widget_layout.setContentsMargins(0, 0, 0, 0)
        # config_widget_layout.setObjectName("horizontalLayout")
        self.ui.config_widget_layout.setObjectName("verticalLayout")

        self.ui.out_render_config_widgets_dict = {}
        # self.ui.out_render_config_widgets_dict = self._build_children_widgets(configs=self.configs)
        self._build_children_widgets(configs=self.configs)
        self.setLayout(self.ui.config_widget_layout)

                
    def clear_all_child_widgets(self):
        clear_layout(self.ui.config_widget_layout)
        self.ui.out_render_config_widgets_dict.clear()

    ## Programmatic Update/Retrieval:    
    def update_from_configs(self, configs: Dict[str, Union[EpochDisplayConfig, List[EpochDisplayConfig]]]):
        """ called to programmatically update the config """
        self.clear_all_child_widgets()
        self.configs = configs ## update self.configs
        self._build_children_widgets(configs=self.configs)


    def configs_from_states(self, as_EpochDisplayConfig_obj: bool=True) -> Dict[str, Union[EpochDisplayConfig, List[EpochDisplayConfig]]]:
        """ called to retrieve a valid config from the UI's properties... this means it could have just held a config as its model. """
        _out_configs = {}
        assert self.ui.out_render_config_widgets_dict is not None, f"self.ui.out_render_config_widgets_dict is None!"
        for a_config_name, a_widget_or_widget_list in self.ui.out_render_config_widgets_dict.items():
            if isinstance(a_widget_or_widget_list, (list, tuple)):
                # a list of configs
                if len(a_widget_or_widget_list) == 1:
                    curr_widget = a_widget_or_widget_list[0]
                    _out_configs[a_config_name] = curr_widget.config_from_state()
                else:
                    ## extract all items                    
                    _out_configs[a_config_name] = [] # start with an empty list
                    for i, a_sub_curr_widget in enumerate(a_widget_or_widget_list):
                        _out_configs[a_config_name].append(a_sub_curr_widget.config_from_state())
            else:
                # Otherwise a straight-up config
                curr_widget = a_widget_or_widget_list
                _out_configs[a_config_name] = curr_widget.config_from_state()
                
        ## END FOR
        if as_EpochDisplayConfig_obj:
            return _out_configs
        else:
            ## convert the EpochDisplayConfig objects to dicts
            update_dict = {}
            for k, v in _out_configs.items():
                if not isinstance(v, (list, tuple)):
                    update_dict[k] = v.to_dict()
                else:
                    update_dict[k] = [sub_v.to_dict() for sub_v in v] ## get the sub-items in the list
            return update_dict
        

    def config_dicts_from_states(self) -> Dict[str, Union[Dict, List[Dict]]]:
        """ called to retrieve a valid config from the UI's properties... this means it could have just held a config as its model. """
        return self.configs_from_states(as_EpochDisplayConfig_obj=False) # type: ignore

    def on_config_ui_updated(self, *args, **kwargs):
        print(f'EpochRenderConfigsListWidget.on_config_ui_updated(*args: {args}, **kwargs: {kwargs})')
        self.sigAnyConfigChanged.emit(self)


    # Add this method to handle the remove request
    def on_remove_epoch_series(self, widget):
        """Handle a request to remove an epoch series"""
        print(f'EpochRenderConfigsListWidget.on_remove_epoch_series(widget: {widget})')
        
        # Find the config name for this widget
        config_name = None
        widget_or_list = None
        
        for name, widget_or_widget_list in self.ui.out_render_config_widgets_dict.items():
            if isinstance(widget_or_widget_list, list):
                if widget in widget_or_widget_list:
                    config_name = name
                    widget_or_list = widget_or_widget_list
                    break
            elif widget_or_widget_list == widget:
                config_name = name
                widget_or_list = widget_or_widget_list
                break
        
        if config_name is not None:
            # Remove the widget from the layout
            if isinstance(widget_or_list, list):
                # If it's in a list, remove just that one widget
                idx = widget_or_list.index(widget)
                self.ui.config_widget_layout.removeWidget(widget)
                widget.deleteLater()
                widget_or_list.pop(idx)
                
                # If the list is now empty, remove the whole entry
                if len(widget_or_list) == 0:
                    del self.ui.out_render_config_widgets_dict[config_name]
                    del self.configs[config_name]
            else:
                # Remove the single widget
                self.ui.config_widget_layout.removeWidget(widget)
                widget.deleteLater()
                del self.ui.out_render_config_widgets_dict[config_name]
                del self.configs[config_name]
            
            # Emit signal to notify of the change
            self.sigAnyConfigChanged.emit(self)
            



def build_single_epoch_display_config_widget(render_config: EpochDisplayConfig) -> EpochRenderConfigWidget:
    """ builds a simple EpochRenderConfigWidget widget from a simple EpochDisplayConfig
    
    Called in build_containing_epoch_display_configs_root_widget(...) down below.
    
    """
    curr_epoch_config_string = f'{render_config.name}'
    curr_widget = EpochRenderConfigWidget(config=render_config) # new widget type
    curr_widget.setObjectName(curr_epoch_config_string)
    
    # curr_widget.update_from_config(render_config) # is this the right type of config? I think it is.

    return curr_widget





## Start Qt event loop
if __name__ == '__main__':
    app = pg.mkQApp('test EpochRenderConfigWidget')
    app.setStyleSheet("""
        QToolTip {
            background-color: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 2px;
            font-size: 12px;
        }
    """)
    test_config = EpochDisplayConfig()
    widget = EpochRenderConfigWidget(config=test_config)
    # widget = EpochRenderConfigWidget()
    widget.show()
    sys.exit(app.exec_())
