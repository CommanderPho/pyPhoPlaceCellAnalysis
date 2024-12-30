# DockPlanningHelperWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockPlanningHelperWidget\DockPlanningHelperWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from typing import Dict, Union
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp
## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget import DockPlanningHelperWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.Uic_AUTOGEN_DockPlanningHelperWidget import Ui_Form
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.Uic_AUTOGEN_TinyDockPlanningHelperWidget import Ui_Form

from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'DockPlanningHelperWidget.ui')

class DockPlanningHelperWidget(QtWidgets.QWidget):
    """ This widget is meant to be embedded in a pyqtgraph.dockarea.Dock to easily prototype/modify its properties. Allows you to create a layout interactively and then save it.
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget
    
    """
    sigSave = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    sigLog = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial
    sigRefresh = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial
    
    sigCreateNewDock = QtCore.pyqtSignal(object, str) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    # sigCreateNewDock = QtCore.pyqtSignal(object, Union[str, tuple[str, str]]) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    
    sigClose = QtCore.pyqtSignal() # Called when the window is closing. 
    
    
    def __init__(self, dock_title='Position Decoder', dock_id='PositionDecoder', defer_show=False, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

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
    def dockHeight(self) -> int:
        return self.ui.spinBox_Height.value()
    @dockHeight.setter
    def dockHeight(self, value):
        self.ui.spinBox_Height.setValue(int(round(value)))
        
    @property
    def dockWidth(self) -> int:
        return self.ui.spinBox_Width.value()
    @dockWidth.setter
    def dockWidth(self, value):
        self.ui.spinBox_Width.setValue(int(round(value)))

    @property
    def contentsFrameWidget(self):
        """The contentsFrameWidget property."""
        return self.ui.contentsFrame
    # @extendedLabel.setter
    # def extendedLabel(self, value):
    #     self.ui.contentsFrame = value


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
        self.ui.btnRefresh.clicked.connect(self.on_refresh)
        
        self.ui.btnAddWidgetRight.clicked.connect(self.on_click_create_new_dock_right)
        self.ui.btnAddWidgetBelow.clicked.connect(self.on_click_create_new_dock_below)
        self.ui.btnAddWidgetAbove.clicked.connect(self.on_click_create_new_dock_tab_on_top_of)
        
        # self.ui.spinBox_Width.valueChanged.connect(self.on_values_updated)

        self.ui.spinBox_Width.editingFinished.connect(self.on_values_updated)        
        self.ui.spinBox_Height.editingFinished.connect(self.on_values_updated)
        
        ## hide non-needed widgets:
        self.ui.txtExtendedLabel.setEnabled(False)
        self.ui.txtExtendedLabel.setHidden(True)
        self.ui.btnLog.setEnabled(False)
        self.ui.btnLog.setHidden(True)
        self.ui.lblInfoTextLine.setHidden(True)
        self.ui.line.setHidden(True)
        self.ui.groupBox.setHidden(True)
        self.ui.groupBoxActions.setHidden(True)                
        self.ui.txtDockIdentifier.setHidden(True)
        self.ui.label_2.setHidden(True)
        self.ui.lblInfoTextLine.setHidden(True)
        
        # self.ui.label_3.setHidden(True)
        # self.ui.horizontalLayout_Size.
        for v in (self.ui.label_3, self.ui.spinBox_Width, self.ui.spinBox_Height): # self.ui.horizontalSpacer_2
            v.setHidden(True)
            

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


    @function_attributes(short_name=None, tags=['TODO'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-12 20:00', related_items=[])
    def rebuild_config(self) -> Dict:
        """ 
        returns the kwarg parameters that would be passed to `parent.add_display_dock(...)`
        `root_dockAreaWindow.add_display_dock(identifier='LongShortColumnsInfo_dock', widget=long_short_info_layout, dockSize=(600,60), dockAddLocationOpts=['top'], display_config=CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False, corner_radius='0px'))`
        
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        
        _custom_display_config = CustomDockDisplayConfig(showCloseButton=False, corner_radius='0px') # custom_get_colors_callback_fn=get_utility_dock_colors,
        # width = self.ui.spinBox_Width.value
        # height = self.ui.spinBox_Height.value
        # _geometry = self.geometry()
        width = int(self.width())
        height = int(self.height())
        # missing keys: dict(widget=self, dockAddLocationOpts=None)
        # return dict(identifier=self.identifier, widget=self, dockSize=(width,height), dockAddLocationOpts=None, display_config=_custom_display_config, autoOrientation=False)
        return dict(identifier=self.identifier, dockSize=(width,height), display_config=_custom_display_config, autoOrientation=False)
           

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
        

    pyqtExceptionPrintingSlot()
    def on_log(self):
        log_string = self.rebuild_output()
        print(f'on_log: {log_string}')
        print(f'DockPlanningHelperWidget.on_log(...):\n\tout_string: {log_string}')
        self.sigLog.emit(self.identifier, self.title, log_string)
    
    pyqtExceptionPrintingSlot()
    def on_save(self):
        # TODO: finish
        out_string = self.rebuild_output()
        print(f'DockPlanningHelperWidget.on_save(...):\n\tout_string: {out_string}')
        self.sigSave.emit(self.identifier, self.title, out_string)
        

    @pyqtExceptionPrintingSlot()
    def on_refresh(self, *args, **kwargs):
        out_string = self.rebuild_output()
        print(f'DockPlanningHelperWidget.on_refresh(...):\n\tout_string: {out_string}')
        self.sigRefresh.emit(self.identifier, self.title, out_string)
        

    # @pyqtExceptionPrintingSlot(object)
    # @QtCore.Slot()
    def on_click_create_new_dock_below(self):
        # [self.embedding_dock_item, 'bottom']
        print(f'DockPlanningHelperWidget.on_click_create_new_dock_below()')
        # self.action_create_new_dock.emit(self.embedding_dock_item, 'bottom')
        self.sigCreateNewDock.emit(self, 'bottom')

    # @pyqtExceptionPrintingSlot(object)
    # @QtCore.
    def on_click_create_new_dock_right(self):
        # [self.embedding_dock_item, 'right']
        print(f'DockPlanningHelperWidget.on_click_create_new_dock_right()')
        # self.action_create_new_dock.emit(self.embedding_dock_item, 'right')
        self.sigCreateNewDock.emit(self, 'right')


    def on_click_create_new_dock_tab_on_top_of(self):
        # [self.embedding_dock_item, 'bottom']
        print(f'DockPlanningHelperWidget.on_click_create_new_dock_tab_on_top_of()')
        # self.action_create_new_dock.emit(self.embedding_dock_item, 'bottom')
        
        # self.sigCreateNewDock.emit(self, ('above', self.identifier,))
        self.sigCreateNewDock.emit(self, f'above, {self.identifier}')
        
    # def __str__(self):
    #      return 


    def resizeEvent(self, event):
        """ default resize event 
        """
        new_size = event.size()
        # Handle the resize event here
        # print(f"Widget resized to: {new_size.width()}x{new_size.height()}")
        
        self.dockHeight = new_size.height()
        self.dockWidth = new_size.width()
        
        super().resizeEvent(event)
        


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("DockPlanningHelperWidget Example")
    widget = DockPlanningHelperWidget()
    widget.show()
    pg.exec()


