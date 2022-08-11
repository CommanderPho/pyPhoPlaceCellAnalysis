# DockPlanningHelperWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockPlanningHelperWidget\DockPlanningHelperWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp
## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget import DockPlanningHelperWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.Uic_AUTOGEN_DockPlanningHelperWidget import Ui_Form

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea


class DockPlanningHelperWidget(QtWidgets.QWidget):
    """ This widget is meant to be embedded in a pyqtgraph.dockarea.Dock to easily prototype/modify its properties. Allows you to create a layout interactively and then save it.
    
    
    """
    action_save = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    action_log = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial
    
    action_create_new_dock = QtCore.pyqtSignal(object, str) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    
    
    close_signal = QtCore.pyqtSignal() # Called when the window is closing. 
    
    
    def __init__(self, dock_title='Position Decoder', dock_id='Position Decoder', defer_show=False, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.

        self.initUI()
        self.title = dock_title
        self.identifier = dock_id
        if not defer_show:
            self.show() # Show the GUI
        self.rebuild_output()

    @property
    def infoLabel(self):
        """The infoLabel property."""
        return self.ui.lblInfoTextLine.text()
    @infoLabel.setter
    def infoLabel(self, value):
        self.ui.lblInfoTextLine.setText(value)
    
    @property
    def extendedLabel(self):
        """The extendedLabel property."""
        return self.ui.txtExtendedLabel.toPlainText()
    @extendedLabel.setter
    def extendedLabel(self, value):
        self.ui.txtExtendedLabel.setPlainText(value)
       
    @property
    def identifier(self):
        """The identifier property."""
        return self.ui.txtDockIdentifier.text()
    @identifier.setter
    def identifier(self, value):
        self.ui.txtDockIdentifier.setText(value)
        
    @property
    def title(self):
        """The title property."""
        return self.ui.txtDockTitle.text()
    @title.setter
    def title(self, value):
        self.ui.txtDockTitle.setText(value)
        
        
    @property
    def dock_orientation(self):
        """The dock_orientation property."""
        if self.ui.radBtnOrientation_Auto.isChecked():
            return 'auto'
        elif self.ui.radBtnOrientation_Vertical.isChecked():
            return 'vertical'
        else:
            return 'horizontal'
        
    @dock_orientation.setter
    def dock_orientation(self, value):
        # ['auto', 'horizontal', 'vertical']
        if value == 'auto':
            self.ui.radBtnOrientation_Auto.setChecked(True)        
        elif value == 'vertical':
            self.ui.radBtnOrientation_Vertical.setChecked(True)
        elif value == 'horizontal':
            self.ui.radBtnOrientation_Horizontal.setChecked(True)
        else:
            raise NotImplementedError


        
        
    
    @property
    def embedding_dock_item(self):
        """Tries to get the Dock that embeds this widget."""
        try:
            potential_dock_parent = self.parentWidget().parentWidget() # Should be two widgets up since Dock creates a container before inserting the widget
            if isinstance(potential_dock_parent, Dock):
                return potential_dock_parent
            else:
                # raise TypeError
                return None # parent is of wrong type.
        except Exception as e:
            # could have no parent, etc.
            return None
        
        
        
    def initUI(self):
        """ 
            txtDockIdentifier
            txtDockTitle

            radBtnOrientation_Auto
            radBtnOrientation_Horizontal
            radBtnOrientation_Vertical

            lblInfoTextLine
            txtExtendedLabel


            btnLog
            btnSave
            
            btnAddWidgetRight
            btnAddWidgetBelow
            
            
        """
        self.ui.txtDockIdentifier.textChanged.connect(self.on_values_updated)
        self.ui.txtDockIdentifier.textChanged.connect(self.on_values_updated)
        
        self.ui.radBtnOrientation_Auto.toggled.connect(self.on_orientation_updated)
        self.ui.radBtnOrientation_Horizontal.toggled.connect(self.on_orientation_updated)
        self.ui.radBtnOrientation_Vertical.toggled.connect(self.on_orientation_updated)
        
        self.ui.btnLog.clicked.connect(self.on_log)
        self.ui.btnSave.clicked.connect(self.on_save)
        
        self.ui.btnAddWidgetRight.clicked.connect(self.on_click_create_new_dock_right)
        self.ui.btnAddWidgetBelow.clicked.connect(self.on_click_create_new_dock_below)
        
        self.rebuild_output()
        
        # self.ui.lblInfoTextLine.setText()
        # self.ui.txtExtendedLabel.setPlainText(value)
        
        
    def rebuild_output(self, debug_print=False):
        log_string = f'title: "{self.title}"\nid: "{self.identifier}"\n'
        log_string = log_string + str(self.geometry())
        
        self.extendedLabel = log_string
        if debug_print:
            print(f'rebuild_output: {log_string}')
        return log_string
        
    def on_values_updated(self):
        print(f'on_values_updated()')
        self.rebuild_output()
        
    def on_orientation_updated(self, evt):
        """ evt should be a boolean indicating whether the radio button was toggled or not """
        print(f'on_orientation_updated({evt})')
        if evt == False:
            return # return without doing anything, because this happens twice for each True update
        if self.embedding_dock_item is not None:
            self.embedding_dock_item.setOrientation(self.dock_orientation)
        
    @QtCore.pyqtSlot()
    def on_log(self):
        log_string = self.rebuild_output()
        print(f'on_log: {log_string}')
        self.action_log.emit(self.identifier, self.title, log_string)
    
    @QtCore.pyqtSlot()
    def on_save(self):
        # TODO: finish
        out_string = self.rebuild_output()
        self.action_save.emit(self.identifier, self.title, out_string)
        
    @QtCore.pyqtSlot()
    def on_click_create_new_dock_below(self):
        # [self.embedding_dock_item, 'bottom']
        print(f'on_click_create_new_dock_below()')
        self.action_create_new_dock.emit(self.embedding_dock_item, 'bottom')

    @QtCore.pyqtSlot()        
    def on_click_create_new_dock_right(self):
        # [self.embedding_dock_item, 'right']
        print(f'on_click_create_new_dock_right()')
        self.action_create_new_dock.emit(self.embedding_dock_item, 'right')

    def __str__(self):
         return 


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("DockPlanningHelperWidget Example")
    widget = DockPlanningHelperWidget()
    widget.show()
    pg.exec()


