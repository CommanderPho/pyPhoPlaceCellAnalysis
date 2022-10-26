# ProgrammaticPipelineWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ProgrammaticPipelineWidget\ProgrammaticPipelineWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from PyQt5 import QtGui, QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphoplacecellanalysis.GUI.Qt.Widgets.ProgrammaticPipelineWidget.Uic_AUTOGEN_ProgrammaticPipelineWidget import Ui_Form
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin
from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons

class ProgrammaticPipelineWidget(PipelineOwningMixin, QWidget):
    """ Displays a UI that allows selecting a filter/computation context from a dropdown list, launching a plots configuration panel, and plotting any of the display functions for the currently selected context.

        'tooltab_Visualization'
            'tooltab_Visualization_Layout'
        'tooltab_Utilities'
            'tooltab_Utilities_Layout'
        'tooltab_Display'
            'tooltab_Display_Layout'
            'tooltab_ProgrammaticDisplayLayout'
    """



    @property
    def active_figure_format_config(self):
        """The figure_Format that overrides the defaults if it exists."""
        if self.ui.active_figure_format_config_widget is None:
            return None # No active override config
        else:
            # Otherwise we have a config widget:
            figure_format_config = self.ui.active_figure_format_config_widget.figure_format_config
            return figure_format_config


    @property
    def last_added_display_output(self):
        """The last_added_display_output property."""
        last_added_context = list(self.owning_pipeline.display_output.keys())[-1]
        last_added_display_output = self.owning_pipeline.display_output[last_added_context]
        return last_added_display_output



    def __init__(self, parent=None, owning_pipeline=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method

        ## Set member properties:
        self._owning_pipeline = owning_pipeline
        
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.initUI()
        self.show() # Show the GUI

        ## This used to be in updateUi(), but that seems very wrong.
        self.ui.contextSelectorWidget.updateUi()
        if self.owning_pipeline is not None:
            self._programmatically_add_display_function_buttons()
        self.updateButtonsEnabled(False) # disable all buttons to start
        self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)
        self.ui.contextSelectorWidget.sigMultiContextChanged.connect(self.on_multi_context_changed)

        self.updateUi()

        ## Schedule delayed_gui_timer
        ## Starts the delayed_gui_itemer which will run after 0.5-seconds to update the GUI:
        self._delayed_gui_timer = QtCore.QTimer(self)
        self._delayed_gui_timer.timeout.connect(self._run_delayed_gui_load_code)
        #Set the interval and start the timer.
        self._delayed_gui_timer.start(500)


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
        # if self.owning_pipeline is not None:
        #     self._programmatically_add_display_function_buttons()
        # self.updateButtonsEnabled(False) # disable all buttons to start
        # self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)
        pass


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


    @pyqtSlot(dict)
    def on_multi_context_changed(self, selected_contexts_dict):
        print(f'on_multi_context_changed(self, selected_contexts_dict: {selected_contexts_dict})')
        has_valid_context = selected_contexts_dict is not None
        # TODO: Disable all the action buttons if the new context is None
        # self.updateButtonsEnabled(has_valid_context)
        curr_any_context_neurons, curr_string_code_rep = self._build_multi_context_any_neurons_id_list()



    def _build_multi_context_any_neurons_id_list(self):
        """ requires from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons
        """
        contextSelectorWidget = self.ui.contextSelectorWidget
        curr_active_contexts = contextSelectorWidget.current_selected_multi_contexts # get the contexts dict from the contextSelectorWidget
        curr_any_context_neurons = _find_any_context_neurons(*[self.owning_pipeline.computation_results[k].computed_data.pf2D.ratemap.neuron_ids for k in list(curr_active_contexts.keys())])
        
        # 'included_unit_neuron_IDs': [2, 3, 4, 5, 8, 10, 11, 13, 14, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 36, 37, 41, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 73, 74, 75, 76, 78, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 96, 98, 100, 102, 105, 108, 109]
        curr_string_code_rep = "'included_unit_neuron_IDs': {curr_any_context_neurons},"
        print(f'curr_string_code_rep: {curr_string_code_rep}')
        return curr_any_context_neurons, curr_string_code_rep




    @pyqtSlot(bool)
    def onShowProgrammaticDisplayConfig(self, is_shown:bool):
        print(f'on_programmatic_display_settings_clicked(is_shown: {is_shown})')
        self._on_build_programmatic_display_config_gui()


    @pyqtSlot(bool)
    def onProgrammaticDisplay(self, is_shown:bool):
        print(f'on_programmatic_display_clicked(is_shown: {is_shown})')


    @pyqtSlot()
    def on_finalize_figure_format_config(self):
        """ called only when figure_format_config is finalized by clicking the Apply button. """
        # Called when figure_format_config is updated
        assert self.ui.active_figure_format_config_widget is not None
        ## Get the figure_format_config from the figure_format_config widget:
        figure_format_config = self.ui.active_figure_format_config_widget.figure_format_config
        print(f'on_finalize_figure_format_config(): {figure_format_config}')
        # Update the current display config
        active_config_name = self.ui.contextSelectorWidget.current_selected_context_key
        assert active_config_name is not None
        # self.owning_pipeline.active_configs[active_config_name] = figure_format_config # update the figure format config for this context
        # print(f'config at owning_pipeline.active_configs[{active_config_name}] has been updated from GUI.')
        ## TODO: update the GUI defaults perminantly? Currently the plot functions will use the overriden values anyway, but not all of the functions accept these kwargs


    @pyqtSlot()
    def _run_delayed_gui_load_code(self):
        """ called when the self._delayed_gui_timer QTimer fires. """
        #Stop the timer.
        self._delayed_gui_timer.stop()
        print(f'_run_delayed_gui_load_code() called!')
        # Try to select the first combo item after they've loaded
        self.ui.contextSelectorWidget._trySelectFirstComboItem()


    # def closeEvent(self, event):
    #     # super().closeEvent(self, event)
    #     if self.ui.active_figure_format_config_widget is not None:
    #         self.ui.active_figure_format_config_widget.close() # close the spanwed widget
    #     # self.sigClosed.emit(self)

    # ==================================================================================================================== #
    # Helper Functions                                                                                                     #
    # ==================================================================================================================== #

    def _on_build_programmatic_display_config_gui(self):
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




    def _add_dynamic_button(self, label="", tooltip="", icon_string=None):
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


    def _programmatically_add_display_function_buttons(self):
        all_display_functions_list = self.owning_pipeline.registered_display_function_names
        all_display_functions_readable_names_list = [name.replace('_display_', '') for name in all_display_functions_list]
        for (readable_name, function_name) in zip(all_display_functions_readable_names_list, all_display_functions_list):
            newToolButton = self._add_dynamic_button(label=f'{readable_name}', tooltip=function_name)
            # print(f'function_name: {function_name}')
            # _newToolFunction = lambda bound_function_name=function_name: self._perform_run_display_function(bound_function_name)
            _newToolFunction = lambda isChecked, bound_function_name=function_name: self._perform_run_display_function(bound_function_name)
            # print(f'_newToolFunction: {_newToolFunction}')
            newToolButton.clicked.connect(_newToolFunction)
            

    def _perform_run_display_function(self, curr_display_fcn, debug_print=False):
        # custom_args = {} # TODO
        custom_args = self.active_figure_format_config or {}
        if debug_print:
            print(f'custom_args: {custom_args}')
        # 'optional_kwargs'
        # print(f'_perform_run_display_function(curr_display_fcn: {curr_display_fcn}): context: {self.ui.contextSelectorWidget.current_selected_context}')
        display_outputs = self.owning_pipeline.display(curr_display_fcn, self.ui.contextSelectorWidget.current_selected_context, **custom_args)
        return display_outputs




## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    self = ProgrammaticPipelineWidget()
    self.show()
    sys.exit(app.exec_())
