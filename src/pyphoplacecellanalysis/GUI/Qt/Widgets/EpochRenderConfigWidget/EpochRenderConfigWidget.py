# EpochRenderConfigWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\EpochRenderConfigWidget\EpochRenderConfigWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import sys
import os
from typing import Optional, List, Dict, Callable

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir


## IMPORTS:
from attrs import define, field, Factory
from pyphocorehelpers.gui.Qt.Param_to_PyQt_Binding import ParamToPyQtBinding
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig, _get_default_epoch_configs
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp
# from pyphoplacecellanalysis.External.pyqtgraph.widgets.ColorButton import ColorButton

# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'EpochRenderConfigWidget.ui')


     

# @define(slots=False, auto_detect=True) # , init=False
class EpochRenderConfigWidget(QWidget):
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
        ## Setup Bindings:
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        
        for a_config_property, a_widget in zip(bound_config_value_list, ui_element_list):
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
                

    def get_ui_element_list(self):
        return [self.ui.btnTitle, self.ui.btnPenColor, self.ui.btnFillColor, self.ui.doubleSpinBoxHeight, self.ui.doubleSpinBoxOffset, self.ui.chkbtnVisible]

    def get_bound_config_value_list(self):
        return [lambda a_config: a_config.name, lambda a_config: a_config.pen_QColor, lambda a_config: a_config.brush_QColor, lambda a_config: a_config.height, lambda a_config: a_config.y_location, lambda a_config: a_config.isVisible]


    ## Programmatic Update/Retrieval:    
    def update_from_config(self, config: EpochDisplayConfig):
        """ called to programmatically update the config """
        param_to_pyqt_binding_dict = ParamToPyQtBinding.param_to_pyqt_binding_dict()
        ui_element_list = self.get_ui_element_list()
        bound_config_value_list = self.get_bound_config_value_list()
        
        for a_config_property, a_widget in zip(bound_config_value_list, ui_element_list):
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
        
        a_config = deepcopy(self.config)
        
        for a_config_property, a_widget in zip(bound_config_value_list, ui_element_list):
            found_binding = param_to_pyqt_binding_dict.get(type(a_widget).__name__, None)
            if found_binding is not None:
                # print(f'found_binding: {found_binding}')
                # desired_value = a_config_property(config)
                # print(f'\t{desired_value}')
                curr_value = found_binding.get_value(a_widget)
                print(f'\t{curr_value}')
                print(f'a_config_property(a_config): {a_config_property(a_config)}')
                # a_config_property(a_config) = curr_value # update to current value
            else:
                print(f'no binding for {a_widget} of type: {type(a_widget)}')


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
        



## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    test_config = EpochDisplayConfig()
    widget = EpochRenderConfigWidget(config=test_config)
    # widget = EpochRenderConfigWidget()
    widget.show()
    sys.exit(app.exec_())
