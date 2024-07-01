# EpochRenderConfigWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\EpochRenderConfigWidget\EpochRenderConfigWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import sys
import os
from typing import Optional, List, Dict, Callable

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
    """ a widget that allows graphically configuring the rendering Epochs 
        EpochDisplayConfig
    """
    config: EpochDisplayConfig = field(default=Factory(EpochDisplayConfig))
        
    ## Attrs manual __init__
    def __init__(self, config: Optional[EpochDisplayConfig]=None, parent=None): # 
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.config = config or EpochDisplayConfig()
        # self.__attrs_init__(EpochDisplayConfig()) # Call the __attrs_init__(...) generated by attrs
        self.initUI()
        self.show() # Show the GUI        

    # def __init__(self, config: EpochDisplayConfig, parent=None):
    #     super().__init__(parent=parent) # Call the inherited classes __init__ method
    #     self.ui = uic.loadUi(uiFile, self) # Load the .ui file

    #     self.initUI()
    #     self.show() # Show the GUI

    def initUI(self):
        self.update_from_config(self.config)
        ## TODO: Setup Bindings:
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        
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
                

    def get_ui_element_list(self):
        return [self.ui.btnTitle, self.ui.btnPenColor, self.ui.btnFillColor, self.ui.doubleSpinBoxHeight, self.ui.doubleSpinBoxOffset, self.ui.chkbtnVisible]

    def get_bound_config_value_list(self):
        return [lambda a_config: a_config.name, lambda a_config: a_config.pen_QColor, lambda a_config: a_config.brush_QColor, lambda a_config: a_config.height, lambda a_config: a_config.y_location, lambda a_config: a_config.isVisible]

    def get_config_property_names_list(self) -> List[str]:
        return ['name', 'pen_QColor', 'brush_QColor', 'height', 'y_location', 'isVisible']


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
        
def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

# @define(slots=False, auto_detect=True) # , init=False
@metadata_attributes(short_name=None, tags=['epoch', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 00:00', related_items=[])
class EpochRenderConfigsListWidget(pg.Qt.QtWidgets.QWidget):
    """ a widget that contains a vertical list of `EpochRenderConfigWidget`. Allowing graphically configuring the rendering Epochs 

    History: based of of function `build_containing_epoch_display_configs_root_widget`


    from pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget import EpochRenderConfigsListWidget

    
    """
    ui: PhoUIContainer = field(init=False, default=None)
    configs: Dict[str, EpochDisplayConfig] = field(init=False, default=None)
    # out_render_config_widgets_dict: Dict[str, pg.Qt.QtWidgets.QWidget] = field(init=False, default=None)

    ## Attrs manual __init__
    def __init__(self, configs: Optional[Dict[str, EpochDisplayConfig]]=None, parent=None): # 
        self.ui = PhoUIContainer()
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.configs = configs or {}
        # self.out_render_config_widgets_dict = 
        if parent is not None:
            parent.addWidget(self)
        self.initUI()


    def _build_children_widgets(self, configs):
        """
        updates:
            self.ui.config_widget_layout
            self.ui.out_render_config_widgets_dict

        """
        assert self.ui.config_widget_layout is not None
        assert self.ui.out_render_config_widgets_dict is not None
        assert len(self.ui.out_render_config_widgets_dict) == 0

        # self.ui.out_render_config_widgets_dict = {}
        for a_config_name, a_config in configs.items():
            if isinstance(a_config, (list, tuple)):
                if len(a_config) == 1:
                    curr_widget = build_single_epoch_display_config_widget(a_config[0]) # 
                    self.ui.config_widget_layout.addWidget(curr_widget)
                    self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget

                else:
                    ## extract all items
                    a_sub_config_widget_layout = pg.Qt.QtWidgets.QHBoxLayout()
                    a_sub_config_widget_layout.setSpacing(0)
                    a_sub_config_widget_layout.setContentsMargins(0, 0, 0, 0)
                    a_sub_config_widget_layout.setObjectName(f"horizontalLayout[{a_config_name}]")
                    
                    self.ui.out_render_config_widgets_dict[a_config_name] = []


                    for i, a_sub_config in enumerate(a_config):
                        a_sub_curr_widget = build_single_epoch_display_config_widget(a_sub_config)
                        a_sub_curr_widget.setObjectName(f"config[{a_config_name}][{i}]")
                        a_sub_config_widget_layout.addWidget(a_sub_curr_widget)
                        # self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget
                        self.ui.out_render_config_widgets_dict[a_config_name].append(a_sub_curr_widget)
                        
                    # curr_widget = a_sub_config_widget_layout
                    self.ui.config_widget_layout.addLayout(a_sub_config_widget_layout)

            else:
                # Otherwise a straight-up config
                curr_widget = build_single_epoch_display_config_widget(a_config)
                self.ui.config_widget_layout.addWidget(curr_widget)
                self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget


    def initUI(self):
        # self.ui.rootWidget = pg.Qt.QtWidgets.QWidget()
        # self.addWidget(self.ui.rootWidget)
        self.ui.config_widget_layout = pg.Qt.QtWidgets.QVBoxLayout()
        self.ui.config_widget_layout.setSpacing(0)
        self.ui.config_widget_layout.setContentsMargins(0, 0, 0, 0)
        # config_widget_layout.setObjectName("horizontalLayout")
        self.ui.config_widget_layout.setObjectName("verticalLayout")

        self.ui.out_render_config_widgets_dict = {}
        for a_config_name, a_config in self.configs.items():
            if isinstance(a_config, (list, tuple)):
                if len(a_config) == 1:
                    curr_widget = build_single_epoch_display_config_widget(a_config[0]) # 
                    self.ui.config_widget_layout.addWidget(curr_widget)
                    self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget

                else:
                    ## extract all items
                    a_sub_config_widget_layout = pg.Qt.QtWidgets.QHBoxLayout()
                    a_sub_config_widget_layout.setSpacing(0)
                    a_sub_config_widget_layout.setContentsMargins(0, 0, 0, 0)
                    a_sub_config_widget_layout.setObjectName(f"horizontalLayout[{a_config_name}]")
                    
                    self.ui.out_render_config_widgets_dict[a_config_name] = []


                    for i, a_sub_config in enumerate(a_config):
                        a_sub_curr_widget = build_single_epoch_display_config_widget(a_sub_config)
                        a_sub_curr_widget.setObjectName(f"config[{a_config_name}][{i}]")
                        a_sub_config_widget_layout.addWidget(a_sub_curr_widget)
                        # self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget
                        self.ui.out_render_config_widgets_dict[a_config_name].append(a_sub_curr_widget)
                        
                    # curr_widget = a_sub_config_widget_layout
                    self.ui.config_widget_layout.addLayout(a_sub_config_widget_layout)

            else:
                # Otherwise a straight-up config
                curr_widget = build_single_epoch_display_config_widget(a_config)
                self.ui.config_widget_layout.addWidget(curr_widget)
                self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget


            # config_widget_layout.addWidget(curr_widget)
            # self.ui.out_render_config_widgets_dict[a_config_name] = curr_widget

        # out_render_config_widgets_dict
        # self.ui.rootWidget.setLayout(self.ui.config_widget_layout)
        self.setLayout(self.ui.config_widget_layout)

                
    def clear_all_child_widgets(self):
        clear_layout(self.ui.config_widget_layout)
        self.ui.out_render_config_widgets_dict.clear()

    ## Programmatic Update/Retrieval:    
    def update_from_configs(self, configs: Dict[str, EpochDisplayConfig]):
        """ called to programmatically update the config """
        self.clear_all_child_widgets()
        self.configs = configs
        self._build_children_widgets(configs=self.configs)




    def configs_from_states(self) -> Dict[str, EpochDisplayConfig]:
        """ called to retrieve a valid config from the UI's properties... this means it could have just held a config as its model. """
        _out_configs = {}
        for a_name, a_widget in self.ui.out_render_config_widgets_dict.items():
            # for a_config_name, a_config in self.configs.items():
            a_config = self.configs[a_name]
            _out_configs[a_name] = a_widget.config_from_state()
        # raise NotImplementedError
        return _out_configs
        



def build_single_epoch_display_config_widget(render_config: EpochDisplayConfig) -> EpochRenderConfigWidget:
    """ builds a simple EpochRenderConfigWidget widget from a simple EpochDisplayConfig
    
    Called in build_containing_epoch_display_configs_root_widget(...) down below.
    
    """
    curr_epoch_config_string = f'{render_config.name}'
    curr_widget = EpochRenderConfigWidget(config=render_config) # new widget type
    curr_widget.setObjectName(curr_epoch_config_string)
    
    # curr_widget.update_from_config(render_config) # is this the right type of config? I think it is.

    return curr_widget


# def build_containing_epoch_display_configs_root_widget(epoch_display_configs, parent=None):
#     """ Renders a list cf config widgets for each epoch into the right sidebar of the Spike3DRasterWindow

#         Usage:
#             from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig, _get_default_epoch_configs
#             from pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget import EpochRenderConfigWidget, build_containing_epoch_display_configs_root_widget


#             rightSideContainerWidget = spike_raster_window.ui.rightSideContainerWidget # pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget.Spike3DRasterRightSidebarWidget
#             a_layout_widget = rightSideContainerWidget.ui.layout_widget
#             rightSideContainerWidget.setVisible(True) # shows the sidebar

#             epoch_display_configs = _get_default_epoch_configs()
#             rootWidget, out_render_config_widgets_dict = build_containing_epoch_display_configs_root_widget(epoch_display_configs, parent=a_layout_widget)
#             rootWidget.show()

#     """
#     if parent is None:
#         rootWidget = pg.Qt.QtWidgets.QWidget()
#     else:
#         rootWidget = pg.Qt.QtWidgets.QWidget()
#         # self.addWidget(rootWidget)
#         parent.addWidget(rootWidget)

#     # self.addWidget(rootWidget)

#     # config_widget_layout = pg.Qt.QtWidgets.QHBoxLayout()
#     config_widget_layout = pg.Qt.QtWidgets.QVBoxLayout()
#     config_widget_layout.setSpacing(0)
#     config_widget_layout.setContentsMargins(0, 0, 0, 0)
#     # config_widget_layout.setObjectName("horizontalLayout")
#     config_widget_layout.setObjectName("verticalLayout")

#     out_render_config_widgets_dict = {}
#     for a_config_name, a_config in epoch_display_configs.items():
#         if isinstance(a_config, (list, tuple)):
#             if len(a_config) == 1:
#                 curr_widget = build_single_epoch_display_config_widget(a_config[0]) # 
#                 config_widget_layout.addWidget(curr_widget)
#             else:
#                 ## extract all items
#                     a_sub_config_widget_layout = pg.Qt.QtWidgets.QHBoxLayout()
#                     a_sub_config_widget_layout.setSpacing(0)
#                     a_sub_config_widget_layout.setContentsMargins(0, 0, 0, 0)
#                     a_sub_config_widget_layout.setObjectName(f"horizontalLayout[{a_config_name}]")
                    
#                     for i, a_sub_config in enumerate(a_config):
#                         a_sub_curr_widget = build_single_epoch_display_config_widget(a_sub_config)
#                         a_sub_curr_widget.setObjectName(f"config[{a_config_name}][{i}]")
#                         a_sub_config_widget_layout.addWidget(a_sub_curr_widget)
                        
#                     curr_widget = a_sub_config_widget_layout
#                     config_widget_layout.addLayout(a_sub_config_widget_layout)

#         else:
#             # Otherwise a straight-up config
#             curr_widget = build_single_epoch_display_config_widget(a_config)
#             config_widget_layout.addWidget(curr_widget)

#         # config_widget_layout.addWidget(curr_widget)
#         out_render_config_widgets_dict[a_config_name] = curr_widget

#     # out_render_config_widgets_dict

#     rootWidget.setLayout(config_widget_layout)
#     return rootWidget, out_render_config_widgets_dict




## Start Qt event loop
if __name__ == '__main__':
    app = pg.mkQApp('test EpochRenderConfigWidget')
    test_config = EpochDisplayConfig()
    widget = EpochRenderConfigWidget(config=test_config)
    # widget = EpochRenderConfigWidget()
    widget.show()
    sys.exit(app.exec_())
