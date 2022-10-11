# ProgrammaticPipelineWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ProgrammaticPipelineWidget\ProgrammaticPipelineWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphoplacecellanalysis.GUI.Qt.Widgets.ProgrammaticPipelineWidget.Uic_AUTOGEN_ProgrammaticPipelineWidget import Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin
from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks



class ProgrammaticPipelineWidget(PipelineOwningMixin, QWidget):
    """ 

        'tooltab_Visualization'
            'tooltab_Visualization_Layout'
        'tooltab_Utilities'
            'tooltab_Utilities_Layout'
        'tooltab_Display'
            'tooltab_Display_Layout'
            'tooltab_ProgrammaticDisplayLayout'
    """
    def __init__(self, parent=None, owning_pipeline=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method

        ## Set member properties:
        self._owning_pipeline = owning_pipeline
        
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.initUI()
        self.show() # Show the GUI
        self.updateUi()

    def initUI(self):
        # self.ui.cmbIdentifyingContext.set = self.all_filtered_session_keys
        # self.ui.btnConfirm.clicked.
        # self.ui.contextSelectorWidget._owning_pipeline = self.owning_pipeline
        # self.updateUi()
        # self.ui.btnProgrammaticDisplayConfig.clicked.connect(self.onShowProgrammaticDisplayConfig)
        # self.ui.btnProgrammaticDisplay.pressed.connect(self.onProgrammaticDisplay)
        self.ui.active_figure_format_config_widget = None
        self.ui.dynamicButtonsList = []
        

    def updateUi(self):
        # Update UI for children controls:
        self.ui.contextSelectorWidget.updateUi()
        if self.owning_pipeline is not None:
            self.programmatically_add_display_function_buttons()
        self.updateButtonsEnabled(False) # disable all buttons to start
        self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)

    @pyqtSlot(bool)
    def updateButtonsEnabled(self, isEnabled: bool):
        """ updates whether the buttons are enabled/disabled"""
        self.ui.btnProgrammaticDisplay.setEnabled(isEnabled)
        self.ui.btnProgrammaticDisplayConfig.setEnabled(isEnabled)
        
        for a_button in self.ui.buttonGroup_Outer_Vis.buttons():
            a_button.setEnabled(isEnabled)
        for a_button in self.ui.buttonGroup_Vis_Maps.buttons():
            a_button.setEnabled(isEnabled)
        for a_button in self.ui.buttonGroup_Vis_Raster.buttons():
            a_button.setEnabled(isEnabled)
        for a_button in self.ui.dynamicButtonsList:
            a_button.setEnabled(isEnabled)


    @pyqtSlot(object, object)
    def on_context_changed(self, new_context_key, new_context):
        print(f'on_context_changed(self, new_context_key: {new_context_key}, new_context: {new_context})')
        has_valid_context = new_context_key is not None
        # TODO: Disable all the action buttons if the new context is None
        self.updateButtonsEnabled(has_valid_context)


    @pyqtSlot(bool)
    def onShowProgrammaticDisplayConfig(self, is_shown:bool):
        print(f'on_programmatic_display_settings_clicked(is_shown: {is_shown})')
        self.on_build_programmatic_display_config()


    @pyqtSlot(bool)
    def onProgrammaticDisplay(self, is_shown:bool):
        print(f'on_programmatic_display_clicked(is_shown: {is_shown})')


    @pyqtSlot()
    def on_finalize_figure_format_config(self):
        # Called when figure_format_config is updated
        assert self.ui.active_figure_format_config_widget is not None
        ## Get the figure_format_config from the figure_format_config widget:
        figure_format_config = self.ui.active_figure_format_config_widget.figure_format_config
        print(f'on_finalize_figure_format_config(): {figure_format_config}')
        # Update the current display config
        active_config_name = self.ui.contextSelectorWidget.current_selected_context_key
        assert active_config_name is not None
        self.owning_pipeline.active_configs[active_config_name] = figure_format_config # update the figure format config for this context
        print(f'config at owning_pipeline.active_configs[{active_config_name}] has been updated from GUI.')


    def on_build_programmatic_display_config(self):
        """ builds the programmatic display format config GUI panel, or displays the existing one if already shown. """
        if self.ui.active_figure_format_config_widget is None:
            # Create a new one:
            # curr_selected_context = self.ui.contextSelectorWidget.current_selected_context
            active_config_name = self.ui.contextSelectorWidget.current_selected_context_key
            curr_active_config = self.owning_pipeline.active_configs[active_config_name] # Get default config for this config name
            # print(f'active_config_name: {active_config_name}, curr_active_config: {curr_active_config}')
            self.ui.active_figure_format_config_widget = FigureFormatConfigControls(config=curr_active_config)
            self.ui.active_figure_format_config_widget.figure_format_config_finalized.connect(self.on_finalize_figure_format_config)
            self.ui.active_figure_format_config_widget.show() # even without .show() being called, the figure still appears

            ## Get the figure_format_config from the figure_format_config widget:
            figure_format_config = self.ui.active_figure_format_config_widget.figure_format_config
        else:
            print(f'figure GUI already exists. Just showing again.')
            self.ui.active_figure_format_config_widget.show()


    # ==================================================================================================================== #
    # Helper Functions                                                                                                     #
    # ==================================================================================================================== #
    def add_dynamic_button(self, label="", tooltip="", icon_string=None):
        """ Adds a dynamically generated button to the "display" tab interface

        icon_string: ":/Render/Icons/Icon/Occupancy.png"

        """
        newToolButton = QtWidgets.QToolButton(self.ui.UtilitiesContent)
        newToolButton.setObjectName(f"toolButton_dynamic_{len(self.ui.dynamicButtonsList)}")
        newToolButton.setText(label)
        newToolButton.setToolTip(tooltip)
        if icon_string is not None:
            newIcon = QtGui.QIcon()
            newIcon.addPixmap(QtGui.QPixmap(icon_string), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            newToolButton.setIcon(newIcon)
            newToolButton.setIconSize(QtCore.QSize(30, 30))
        # self.ui.tooltab_Utilities_Layout.addWidget(newToolButton)
        self.ui.tooltab_ProgrammaticDisplayLayout.addWidget(newToolButton)
        self.ui.dynamicButtonsList.append(newToolButton)
        return newToolButton # return the newly created button


    def programmatically_add_display_function_buttons(self):
        all_display_functions_list = self.owning_pipeline.registered_display_function_names
        all_display_functions_readable_names_list = [name.replace('_display_', '') for name in all_display_functions_list]
        for (readable_name, function_name) in zip(all_display_functions_readable_names_list, all_display_functions_list):
            newToolButton = self.add_dynamic_button(label=f'{readable_name}', tooltip=function_name)
            # print(f'function_name: {function_name}')
            # _newToolFunction = lambda bound_function_name=function_name: self._perform_run_display_function(bound_function_name)
            _newToolFunction = lambda isChecked, bound_function_name=function_name: self._perform_run_display_function(bound_function_name)
            # print(f'_newToolFunction: {_newToolFunction}')
            newToolButton.clicked.connect(_newToolFunction)
            

    def _perform_run_display_function(self, curr_display_fcn):
        # has_valid_selection
        # if self.display_results is not None:
        #     custom_args = self.display_results.get('kwargs', {})
        # else:
        #     custom_args = {} # no custom args, just pass empty dictionary
        custom_args = {} # TODO
        # print(f'_perform_run_display_function(curr_display_fcn: {curr_display_fcn}): context: {self.ui.contextSelectorWidget.current_selected_context}')
        display_outputs = self.owning_pipeline.display(curr_display_fcn, self.ui.contextSelectorWidget.current_selected_context, **custom_args)
        return display_outputs




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    self = ProgrammaticPipelineWidget()
    self.show()
    sys.exit(app.exec_())
