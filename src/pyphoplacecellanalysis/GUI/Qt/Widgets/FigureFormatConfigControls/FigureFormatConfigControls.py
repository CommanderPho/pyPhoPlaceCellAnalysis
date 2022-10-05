# FigureFormatConfigControls.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\FigureFormatConfigControls\FigureFormatConfigControls.ui automatically by PhoPyQtClassGenerator VSCode Extension

from datetime import datetime
from pathlib import Path
import warnings # for getting the current date to set the ouptut folder name
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

## IMPORTS:
# from pyPhoPlaceCellAnalysis.GUI.Qt.FigureFormatConfigControls  import FigureFormatConfigControls
from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.Uic_AUTOGEN_FigureFormatConfigControls import Ui_Form

# def pair_optional_value_widget(checkBox, valueWidget):
#     self.checkBox.toggled['bool'].connect(self.spinBox.setEnabled) # type: ignore
    

class FigureFormatConfigControls(QtWidgets.QWidget):
    _debug_print = False
    
    ## Signals
    figure_format_config_updated = QtCore.pyqtSignal(object)
    figure_format_config_finalized = QtCore.pyqtSignal(object) # called only when the changes to the config are finalized by clicking 'Apply' or 'Ok' Button
    
    @property
    def enable_saving_to_disk(self):
        """The enable_saving_to_disk property."""
        return self.ui.chkEnableSavingToDisk.isChecked()
    @enable_saving_to_disk.setter
    def enable_saving_to_disk(self, value):
        self.ui.chkEnableSavingToDisk.setChecked(value)
        
    @property
    def enable_spike_overlay(self):
        """The enable_saving_to_disk property."""
        return self.ui.chkEnableSpikeOverlay.isChecked()
    @enable_spike_overlay.setter
    def enable_spike_overlay(self, value):
        self.ui.chkEnableSpikeOverlay.setChecked(value)
        
    @property
    def enable_debug_print(self):
        return self.ui.chkDebugPrint.isChecked()
    @enable_debug_print.setter
    def enable_debug_print(self, value):
        self.ui.chkDebugPrint.setChecked(value)
        
    @property
    def figure_format_config(self):
        figure_format_config = {self.ui.tupleCtrl_0.control_name:self.ui.tupleCtrl_0.tuple_values,
                    self.ui.tupleCtrl_1.control_name:self.ui.tupleCtrl_1.tuple_values,
                    # self.ui.tupleCtrl_2.control_name:self.ui.tupleCtrl_2.tuple_values,
        }
        ## Add explicit column/row widths to fix window sizing issue:
        figure_format_config = (dict(fig_column_width=self.ui.tupleCtrl_2.tuple_values[0], fig_row_height=self.ui.tupleCtrl_2.tuple_values[1]) | figure_format_config)
        figure_format_config = (dict(enable_spike_overlay=self.enable_spike_overlay, debug_print=self.enable_debug_print, enable_saving_to_disk=self.enable_saving_to_disk) | figure_format_config)
        return figure_format_config
    @figure_format_config.setter
    def figure_format_config(self, value):
        """ update the ui properties and controls from the passed in figure_format_config dict. Not all properties required to be present, and those not specified will be unchanged. """
        if value.get('subplots', None) is not None:
            self.ui.tupleCtrl_0.tuple_values = value.get('subplots', self.ui.tupleCtrl_0.tuple_values)
        if value.get('max_screen_figure_size', None) is not None:
            self.ui.tupleCtrl_1.tuple_values = value.get('max_screen_figure_size', self.ui.tupleCtrl_1.tuple_values)
        # Row/Col Width/Height:
        curr_tuple_values = self.ui.tupleCtrl_2.tuple_values
        self.ui.tupleCtrl_2.tuple_values = (value.get('fig_column_width', curr_tuple_values[0]), value.get('fig_row_height', curr_tuple_values[1]))
        self.enable_spike_overlay = value.get('enable_spike_overlay', self.enable_spike_overlay) 
        self.enable_debug_print = value.get('debug_print', self.enable_debug_print)
        self.enable_saving_to_disk = value.get('enable_saving_to_disk', self.enable_saving_to_disk)
            
            
    def __init__(self, parent=None, config=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
  
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.

        self.config = config # InteractivePlaceCellConfig
        
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        self.ui.tupleCtrl_0.control_name = 'subplots'
        self.ui.tupleCtrl_0.tuple_values = (20, 8)
        self.ui.tupleCtrl_0.tuple_values = (None, 8)
        
        self.ui.tupleCtrl_1.control_name = 'max_screen_figure_size'
        self.ui.tupleCtrl_1.tuple_values = (2256, 1868)
        self.ui.tupleCtrl_1.tuple_values = (None, 1868)
        
        self.ui.tupleCtrl_2.control_name = 'col_width/row_height'
        self.ui.tupleCtrl_2.tuple_values = (5, 5)
        self.ui.tupleCtrl_2.tuple_values = (None, None)
        
        # self.ui.filepkr_FigureOutputPath.path
        try:
            curr_fig_save_path = self.config.plotting_config.get_figure_save_path()
        except AttributeError as e:
            warnings.warn('No config! using default')
            curr_fig_save_path = r'W:\Data\Output'
        except Exception as e:
            raise e
        
        if self._debug_print:
            print(f'Old Individual Plotting Function Figure Output path: {str(curr_fig_save_path)}')
        self.ui.filepkr_FigureOutputPath.path = curr_fig_save_path
        ## Connect Buttons:
        # self.ui.btnOpenFigureExportPathInSystemFileBrowser.pressed.connect(self.perform_open_in_system_filebrowser)
        
        
        # self.ui.filepkr_ProgrammaticDisplayFcnOutputPath:
        programmatic_display_function_testing_output_parent_path = None # currently always start with the hardcoded default below, the user can always change it
        if programmatic_display_function_testing_output_parent_path is None:   
            programmatic_display_function_testing_output_parent_path = Path(r'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Screenshots\ProgrammaticDisplayFunctionTesting')
        else:
            programmatic_display_function_testing_output_parent_path = Path(programmatic_display_function_testing_output_parent_path) # make sure it's a Path
        # programmatic_display_function_testing_output_parent_path.mkdir(exist_ok=True)
        out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
        programmatic_display_function_testing_output_parent_path = programmatic_display_function_testing_output_parent_path.joinpath(out_day_date_folder_name)
        if self._debug_print:
            print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')
        # Set the path control to this path:
        self.ui.filepkr_ProgrammaticDisplayFcnOutputPath.path = programmatic_display_function_testing_output_parent_path
        
        ## Connect signals
        self.ui.tupleCtrl_0.value_changed.connect(self.on_update_values) # type: ignore
        self.ui.tupleCtrl_1.value_changed.connect(self.on_update_values) # type: ignore
        self.ui.tupleCtrl_2.value_changed.connect(self.on_update_values) # type: ignore
        
        # self.ui.buttonBox.clicked.connect(self.on_general_buttonbox_action) # called on any action for the button itself
        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.on_apply)
        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(self.on_cancel)
        # self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.on_revert)
        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.on_revert)
        # self.ui.buttonBox.rejected.connect(self.on_cancel)
        
        
       
    @QtCore.pyqtSlot()
    def on_update_values(self):
        figure_format_config = self.figure_format_config
        if self._debug_print:
            print('on_update_values')
            print(f'\t {figure_format_config}')
        # Made accessible by self.figure_format_config access.
        self.figure_format_config_updated.emit(figure_format_config) # emit the signal
        
    # def __str__(self):
    #      return 


    # ==================================================================================================================== #
    # Figure Output Section                                                                                                #
    # ==================================================================================================================== #
    @QtCore.pyqtSlot()
    def perform_open_in_system_filebrowser(self):
        print('perform_open_in_system_filebrowser()')
        curr_fig_save_path = self.ui.filepkr_FigureOutputPath.path
        print(f'\t{str(curr_fig_save_path)}')
        reveal_in_system_file_manager(curr_fig_save_path)

    # ==================================================================================================================== #
    # Slots for actions button at bottom of widget                                                                         #
    # ==================================================================================================================== #

    # @QtCore.pyqtSlot(object)
    def on_general_buttonbox_action(self, button):
        print('on_general_buttonbox_action({button})')
    
    @QtCore.pyqtSlot()
    def on_apply(self):
        print('on_apply()')
        self.figure_format_config_finalized.emit(self.figure_format_config) # emit the finalized signal
        
    @QtCore.pyqtSlot()
    def on_cancel(self):
        print('on_cancel()')
        # self.figure_format_config_updated.emit(self.figure_format_config) # emit the signal


    @QtCore.pyqtSlot()
    def on_revert(self):
        print('on_revert()')
        # self.figure_format_config_updated.emit(self.figure_format_config) # emit the signal


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("FigureFormatConfigControls Example")
    widget = FigureFormatConfigControls()
    widget.show()
    pg.exec()
