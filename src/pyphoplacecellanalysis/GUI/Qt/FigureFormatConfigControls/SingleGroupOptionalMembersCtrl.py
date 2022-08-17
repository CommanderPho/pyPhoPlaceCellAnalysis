# SingleGroupOptionalMembersCtrl.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\FigureFormatConfigControls\SingleGroupOptionalMembersCtrl.ui automatically by PhoPyQtClassGenerator VSCode Extension
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls import SingleGroupOptionalMembersCtrl
from pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls.Uic_AUTOGEN_SingleGroupOptionalMembersCtrl import Ui_SingleGroupOptionalMembersCtrl


class SingleGroupOptionalMembersCtrl(QtWidgets.QWidget):
    """ A widget that represents a tuple with optional-valued members 
    
    """
    value_changed = QtCore.pyqtSignal(object,bool,int,bool,int)
    _debug_print = False
    
    @property
    def control_name(self):
        """The control_name property."""
        return self.ui.lblPropertyName.text()
    @control_name.setter
    def control_name(self, value):
        self.ui.lblPropertyName.setText(value)

    @property
    def tuple_values(self):
        """The values property."""
        if self.ui.checkBox_0.isChecked():
            v1 = self.ui.spinBox_0.value()
        else:
            v1 = None
        if self.ui.checkBox_1.isChecked():
            v2 = self.ui.spinBox_1.value()
        else:
            v2 = None
        return (v1, v2)
    @tuple_values.setter
    def tuple_values(self, value):
        assert len(value) == 2
        v1, v2 = value    
        self.ui.checkBox_0.setChecked(v1 is not None)
        if v1 is not None:
            self.ui.spinBox_0.setValue(v1)
        self.ui.checkBox_1.setChecked(v2 is not None)
        if v2 is not None:
            self.ui.spinBox_1.setValue(v2)
    
    def __init__(self, parent=None, control_name:str = 'test_ctrl'):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_SingleGroupOptionalMembersCtrl()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.ui.lblPropertyName.setText(control_name)        
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        pass

    @QtCore.pyqtSlot()
    def on_update_values(self):
        if self._debug_print:
            print('on_update_values')
            print(f'\t {(self.ui.checkBox_0.isChecked(), self.ui.spinBox_0.value())}')
            print(f'\t {(self.ui.checkBox_1.isChecked(), self.ui.spinBox_1.value())}')
        self.value_changed.emit(self.control_name, self.ui.checkBox_0.isChecked(), self.ui.spinBox_0.value(), self.ui.checkBox_1.isChecked(), self.ui.spinBox_1.value())
        # value_changed(str,bool,int,bool,int)
        
    def __str__(self):
         return 





## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("PlacefieldVisualSelectionWidget Example")
    widget = SingleGroupOptionalMembersCtrl(control_name='Subplots')
    widget.show()
    pg.exec()


