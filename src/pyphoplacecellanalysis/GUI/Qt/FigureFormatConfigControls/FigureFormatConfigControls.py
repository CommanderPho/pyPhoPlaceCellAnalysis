# FigureFormatConfigControls.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\FigureFormatConfigControls\FigureFormatConfigControls.ui automatically by PhoPyQtClassGenerator VSCode Extension
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

## IMPORTS:
# from pyPhoPlaceCellAnalysis.GUI.Qt.FigureFormatConfigControls  import FigureFormatConfigControls
from pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls.Uic_AUTOGEN_FigureFormatConfigControls import Ui_Form

# def pair_optional_value_widget(checkBox, valueWidget):
#     self.checkBox.toggled['bool'].connect(self.spinBox.setEnabled) # type: ignore
    

class FigureFormatConfigControls(QtWidgets.QWidget):
    
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
        

    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
  
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.

        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        self.ui.tupleCtrl_0.control_name = 'Subplots'
        self.ui.tupleCtrl_0.tuple_values = (20, 8)
        self.ui.tupleCtrl_0.tuple_values = (None, 8)
        
        self.ui.tupleCtrl_1.control_name = 'max_screen_figure_size'
        self.ui.tupleCtrl_1.tuple_values = (2256, 1868)
        self.ui.tupleCtrl_1.tuple_values = (None, 1868)
        
        self.ui.tupleCtrl_2.control_name = 'col_width/row_height'
        self.ui.tupleCtrl_2.tuple_values = (5.0, 5.0)
        self.ui.tupleCtrl_2.tuple_values = (None, None)
        
        # enable_spike_overlay
 
        ## Connect signals
        self.ui.tupleCtrl_0.value_changed.connect(self.on_update_values) # type: ignore
        self.ui.tupleCtrl_1.value_changed.connect(self.on_update_values) # type: ignore
        self.ui.tupleCtrl_2.value_changed.connect(self.on_update_values) # type: ignore
        
 
        
    @QtCore.pyqtSlot()
    def on_update_values(self):
        print('on_update_values')
        
        figure_format_config = {self.ui.tupleCtrl_0.control_name:self.ui.tupleCtrl_0.tuple_values,
                    self.ui.tupleCtrl_1.control_name:self.ui.tupleCtrl_1.tuple_values,
                    # self.ui.tupleCtrl_2.control_name:self.ui.tupleCtrl_2.tuple_values,
        }
        
        ## Add explicit column/row widths to fix window sizing issue:
        figure_format_config = (dict(fig_column_width=self.ui.tupleCtrl_2.tuple_values[0], fig_row_height=self.ui.tupleCtrl_2.tuple_values[1]) | figure_format_config)
        
        figure_format_config = (dict(enable_spike_overlay=self.enable_spike_overlay, debug_print=self.enable_debug_print, enable_saving_to_disk=self.enable_saving_to_disk) | figure_format_config)
        
        print(f'\t {figure_format_config}')
        # TODO: OUTPUT figure_format_config
        
        
        # self.ui.check
        # chkEnableSpikeOverlay
        # chkDebugPrint
        # chkEnableSavingToDisk

    def __str__(self):
         return 

"""
lblPropertyName

"""

## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("FigureFormatConfigControls Example")
    widget = FigureFormatConfigControls()
    widget.show()
    pg.exec()
