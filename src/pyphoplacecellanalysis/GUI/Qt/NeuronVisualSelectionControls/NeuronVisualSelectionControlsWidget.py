# NeuronVisualSelectionControlsWidget.py
# Generated from NeuronVisualSelectionControlsWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic
# from PyQt5 import QtGui, QtWidgets, uic
from matplotlib.colors import to_hex # required for QColor conversion to hex

## IMPORTS:
# For compatibility with the panel ui version:
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'NeuronVisualSelectionControlsWidget.ui')


@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-18 15:04', related_items=[])
class NeuronVisualSelectionControlsWidget(QtWidgets.QWidget):
    """ Copied from PlacefieldVisualSelectionWidget on 2023-10-18 
    
    Change .ui.groupBox to .ui.mainFrame
    """
    # spike_config_changed = QtCore.pyqtSignal(bool) # change_unit_spikes_included(self, neuron_IDXs=None, cell_IDs=None, are_included=True)
    # tuning_curve_display_config_changed = QtCore.pyqtSignal(list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    
    # Send a SingleNeuronPlottingExtended config state instead of a QtGui.QColor for easy access.
    sig_neuron_color_changed = QtCore.pyqtSignal(object) # send the updated color as a QtGui.QColor
    
    # update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
    # finish_signal = QtCore.pyqtSignal(float, float)
 
    # enable_debug_print = True
    enable_debug_print = False
    show_groupbox_title = False
            
    @property
    def name(self):
        """The name property."""
        if self._name is None:
            self._name = self.ui.btnTitle.text
        return self._name 
    @name.setter
    def name(self, value):
        self._name = value
        # if NeuronVisualSelectionControlsWidget.show_groupbox_title:        
        #     self.ui.groupBox.setTitle(self._name) # set a new value like "pf[i]"  
        self.ui.btnTitle.setText(self._name)
  
    @property
    def spikesVisible(self):
        """The spikesVisible property."""
        if self._spikesVisible is None:
            self._spikesVisible = self.ui.chkbtnSpikes.checked
        return self._spikesVisible
    @spikesVisible.setter
    def spikesVisible(self, value):
        self._spikesVisible = value
        self.ui.chkbtnSpikes.setChecked(self._spikesVisible)
        

    @property
    def isVisible(self):
        """The isVisible property."""
        if self._isVisible is None:
            self._isVisible = self.ui.chkbtnPlacefield.checked
        return self._isVisible 
    @isVisible.setter
    def isVisible(self, value):
        self._isVisible = value
        self.ui.chkbtnPlacefield.setChecked(self._isVisible)
        
    @property
    def color(self):
        """The color property."""
        if self._color is None:
            self._color = self.ui.btnColorButton.color()
        return self._color
    @color.setter
    def color(self, value):
        self._color = value
        self.ui.btnColorButton.setColor(self._color, finished=True)
        



    def __init__(self, *args, config=None, parent=None, **kwargs):
        self.enable_debug_print = NeuronVisualSelectionControlsWidget.enable_debug_print
        self._name = None
        self._color = None
        self._isVisible = None
        self._spikesVisible = None
        super().__init__(*args, parent=parent, **kwargs) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()

        if config is not None:
            self.update_from_config(config)

        
        # Setup self.ui.btnColorButton:
        # def change(btn):
        #     if self.enable_debug_print:
        #         print("btnColorButton change", btn.color())
        #     self._color = btn.color()
        # def done(btn):
        #     if self.enable_debug_print:
        #         print("btnColorButton done", btn.color())
        #     self._color = btn.color()
        #     self.sig_neuron_color_changed.emit(self.config_from_state()) # Only call when done with selecting color, but return the whole state.
        #     # self.sig_neuron_color_changed.emit(btn.color()) # Only call when done with selecting color.
        
        
        # self.ui.btnColorButton.sigColorChanging.connect(change)
        # self.ui.btnColorButton.sigColorChanged.connect(done)
        

        ## Hide second row buttons:
        self.ui.chkbtnPlacefield.hide()
        self.ui.chkbtnSpikes.hide()



        # Setup self.ui.chkbtnPlacefield:
        self.ui.chkbtnPlacefield.toggled.connect(self.togglePlacefieldVisibility)
        self.ui.chkbtnSpikes.toggled.connect(self.toggleSpikeVisibility)
  
        # Connect the color button:
        self.ui.btnColorButton.sigColorChanging.connect(self.on_color_button_changing)
        self.ui.btnColorButton.sigColorChanged.connect(self.on_color_button_changed)  

        self.show() # Show the GUI



    def initUI(self):
        self.ui.btnColorButton.setEnabled(True)


    @QtCore.pyqtSlot(object)
    def on_color_button_changing(self, btn):
        self._color = btn.color()
        if self.enable_debug_print:
            print(f'on_color_button_changing(value: {self._color})')

    @QtCore.pyqtSlot(object)
    def on_color_button_changed(self, btn):
        self._color = btn.color()
        if self.enable_debug_print:
            print(f'on_color_button_changed(value: {self._color})')
        
        # Since we're done, emit the sig_neuron_color_changed signal:
        
        try:
            curr_config = self.config_from_state()
        except AttributeError as e:
            if self.enable_debug_print:
                print(f'\t on_color_button_changed(...) encountered AttributeError, gracefully returning: {e}')
            return
            # raise e
        except Exception as e:
            raise e
            
        if self.enable_debug_print:
            print(f'\t self.config_from_state(): {curr_config}')
        self.sig_neuron_color_changed.emit(curr_config) # Only call when done with selecting color, but return the whole state.
        # self.sig_neuron_color_changed.emit(btn.color()) # Only call when done with selecting color.
    
  
    @QtCore.pyqtSlot(bool)
    def togglePlacefieldVisibility(self, value):
        if self.enable_debug_print:
            print(f'_on_toggle_plot_visible_changed(value: {value})')
        self._isVisible = value
        self.tuning_curve_display_config_changed.emit([self.config_from_state()]) # emit signal
    
    @QtCore.pyqtSlot(bool)
    def toggleSpikeVisibility(self, value):
        if self.enable_debug_print:
            print(f'_on_toggle_spikes_visible_changed(value: {value})')
        self._spikesVisible = bool(value)
        self.spike_config_changed.emit(bool(self.spikesVisible)) # emit signal

    ## Programmatic Update/Retrieval:    
    def update_from_config(self, config):
        """ called to programmatically update the config """
        self.name = config.name
        self.color = config.color
        self.isVisible = config.isVisible
        self.spikesVisible = config.spikesVisible

    def config_from_state(self):
        """ called to retrieve a valid config from the UI's properties... this means it could have just held a config as its model. """
        if self.enable_debug_print:
            print(f'config_from_state(...): name={self.name}, isVisible={self.isVisible}, color={self.color}, spikesVisible={self.spikesVisible}')
            print(f'\tself.color: {self.color} - self.color.name(): {self.color.name()}')
        
        # How to convert a QColor into a HexRGB String:
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        color_hex_str = self.color.name(QtGui.QColor.HexRgb) 
        if self.enable_debug_print:
            print(f'\thex: {color_hex_str}')
        
        # also I think the original pf name was formed by adding crap...
        ## see 
        # ```curr_pf_string = f'pf[{render_config.name}]'````
        ## UPDATE: this doesn't seem to be a problem. The name is successfully set to the ACLU value in the current state.
        return SingleNeuronPlottingExtended(name=self.name, isVisible=self.isVisible, color=color_hex_str, spikesVisible=self.spikesVisible)
        

    def add_ui_push_button(self, name):
        button_name = f'btn{name}'
        btnNew = QtWidgets.QPushButton(self.groupBox)
        btnNew.setText(_translate("rootForm", "pf[i]"))
        btnNew.setObjectName(button_name)
        self.verticalLayout.addWidget(btnNew)
        
        return btnNew

    # def add_ui_color_button(self):
    #     btnNewColorButton = ColorButton(self.groupBox)
    #     btnNewColorButton.setEnabled(False)
    #     btnNewColorButton.setMinimumSize(QtCore.QSize(24, 24))
    #     btnNewColorButton.setText("")
    #     btnNewColorButton.setObjectName("btnColorButton")
    #     self.verticalLayout.addWidget(btnNewColorButton)
    #     return btnNewColorButton
    
    @classmethod
    def build_ui_toggle_button(cls, name='chkbtnNewButton', text='spikes', parent=None):
        """ 
        Builds a simple new toggle button widget
        
        chkbtnNewButton = self.build_ui_toggle_button(name=name, text=text, parent=self.groupBox)
        self.verticalLayout.addWidget(chkbtnNewButton)
        
        """
        chkbtnNewButton = QtWidgets.QToolButton(parent)
        chkbtnNewButton.setMinimumSize(QtCore.QSize(48, 25))
        chkbtnNewButton.setStyleSheet(cls.css_toolButton())
        chkbtnNewButton.setCheckable(True)
        chkbtnNewButton.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        chkbtnNewButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        chkbtnNewButton.setText(text)
        chkbtnNewButton.setObjectName(name)
        return chkbtnNewButton
        
    def add_ui_toggle_button(self, name='chkbtnNewButton', text='spikes'):
        """ adds a simple toggle chkbtn """
        # chkbtnNewButton = QtWidgets.QToolButton(self.groupBox)
        chkbtnNewButton = self.build_ui_toggle_button(name=name, text=text, parent=self.groupBox)
        self.verticalLayout.addWidget(chkbtnNewButton)
        setattr(self.ui, name, chkbtnNewButton) # add to self.ui to keep a reference to the widget
        
        return chkbtnNewButton
        
    
    @staticmethod
    def css_pushButton():
        css = '''
                QPushButton {
                                font-size: 10px;
                                background-color: green;
                                color: black;
                                border: 2px green;
                                border-radius: 22px;
                                border-style: outset;
                                    }
                QPushButton:hover {
                                background: qradialgradient(
                                    cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
                                    radius: 1.35, stop: 0 grey, stop: 1 lightgray
                                    );
                                }
                QPushButton:enabled{
                                color: black;
                                font:  10px;
                                background: green;
                                background-color: red;
                                border: 1px black;
                                border-style: outset;
                                }
                QPushButton:pressed {
                                color: white;
                                background: yellow;
                                }
                QPushButton:disabled {
                                color: gray;
                                background-color: gray;
                                border: 1px black;
                                border-style: outset;                
                            }
                QPushButton:checked{
                            color: black; 
                            font:  12px;   
                            font: bold;
                            background-color: red;
                            border: 1px black;
                            border-style: outset;
                            }
                QPushButton:!checked{
                            color: black; 
                            font:  12px;   
                            font: bold;
                            background-color: green;
                            border: 1px black;
                            border-style: outset;
                }
                
                    '''
        return css
    
    @staticmethod
    def css_toolButton():
        
        css = ''' 
        QToolButton {
            /*font-size: 10px;
            background-color: green;
            color: rgb(244, 244, 244);
            border: 2px green;
            border-radius: 22px;
            border-style: outset;*/
        }
        QToolButton:disabled {
            color: gray;
            background-color: gray;
            border: 1px black;
            border-style: outset;                
        }
        QToolButton:checked{
            color: rgb(255, 170, 0); 
            /*font:  12px; */  
            font: bold;
            /*background-color: red;*/
            border: 1px white;
            border-style: outset;
        }
        QToolButton:!checked{
            /*color: black; 
            font:  12px;   
            font: bold;
            background-color: green;
            border: 1px black;
            border-style: outset;*/
        }
        
        '''
        # css_full = '''
        #         QToolButton {
        #                         font-size: 10px;
        #                         background-color: green;
        #                         color: black;
        #                         border: 2px green;
        #                         border-radius: 22px;
        #                         border-style: outset;
        #                             }
        #         QToolButton:hover {
        #                         background: qradialgradient(
        #                             cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
        #                             radius: 1.35, stop: 0 grey, stop: 1 lightgray
        #                             );
        #                         }
        #         QToolButton:enabled{
        #                         color: black;
        #                         font:  10px;
        #                         background: green;
        #                         background-color: red;
        #                         border: 1px black;
        #                         border-style: outset;
        #                         }
        #         QToolButton:pressed {
        #                         color: white;
        #                         background: yellow;
        #                         }
        #         QToolButton:disabled {
        #                         color: gray;
        #                         background-color: gray;
        #                         border: 1px black;
        #                         border-style: outset;                
        #                     }
        #         QToolButton:checked{
        #                     color: black; 
        #                     font:  12px;   
        #                     font: bold;
        #                     background-color: red;
        #                     border: 1px black;
        #                     border-style: outset;
        #                     }
        #         QToolButton:!checked{
        #                     color: black; 
        #                     font:  12px;   
        #                     font: bold;
        #                     background-color: green;
        #                     border: 1px black;
        #                     border-style: outset;
        #         }
        # '''
        return css
    


# ==================================================================================================================== #
# Container Class                                                                                                      #
# ==================================================================================================================== #

@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-18 15:03', related_items=[])
class NeuronWidgetContainer(QtWidgets.QWidget):
    """ Renders a vertical list that displays the color and properties of each neuron
    
    Usage:
        from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
        from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended
        from pyphoplacecellanalysis.GUI.Qt.NeuronVisualSelectionControls.NeuronVisualSelectionControlsWidget import NeuronVisualSelectionControlsWidget, NeuronWidgetContainer

        neuron_ids = active_2d_plot.neuron_ids
        neuron_plotting_configs_list: List[SingleNeuronPlottingExtended] = [SingleNeuronPlottingExtended(name=str(aclu), isVisible=False, color=ColorFormatConverter.qColor_to_hexstring(color, include_alpha=False), spikesVisible=False) for aclu, color in zip(neuron_ids, neuron_qcolors_list)]
        # Standalone:
        # neuron_widget_container = NeuronWidgetContainer(neuron_plotting_configs_list)
        # neuron_widget_container.show()

        ## Render in right sidebar:
        neuron_widget_container = NeuronWidgetContainer(neuron_plotting_configs_list, parent=spike_raster_window.right_sidebar_contents_container)
        spike_raster_window.right_sidebar_contents_container.addWidget(neuron_widget_container)
        
        
    Feature: Adding Buttons Dynamically:
    chkbtnNewButtonTest = a_widget.add_ui_toggle_button(name='chkbtnNewButtonTest', text='test1')
    
    """
    sigRefresh = QtCore.pyqtSignal(object)
    

    def __init__(self, neuron_plotting_configs_list, parent=None):
        self.ui = PhoUIContainer()
        super(NeuronWidgetContainer, self).__init__(parent)
        
        # Create a QWidget for the scroll area content
        self.ui.scroll_content = QtWidgets.QWidget()
        
        self.ui.main_widgets_list_layout = QtWidgets.QVBoxLayout()
        self.ui.main_widgets_list_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.main_widgets_list_layout.setSpacing(2)

        self.ui.widgets_list = []
        for neuron_config in neuron_plotting_configs_list:
            widget = NeuronVisualSelectionControlsWidget(config=neuron_config)
            self.ui.widgets_list.append(widget)
            self.ui.main_widgets_list_layout.addWidget(widget)
        
        # Set the layout to the scroll area content widget
        self.ui.scroll_content.setLayout(self.ui.main_widgets_list_layout)

        # Create scroll area and set its content
        self.ui.scroll_area = QtWidgets.QScrollArea()
        self.ui.scroll_area.setWidgetResizable(True)
        self.ui.scroll_area.setWidget(self.ui.scroll_content)

        # Create a main layout for self and add the scroll area
        self.ui.main_layout = QtWidgets.QVBoxLayout(self)
        self.ui.main_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.main_layout.setSpacing(0)
        self.ui.main_layout.addWidget(self.ui.scroll_area)
        self.setLayout(self.ui.main_layout)


    # ==================================================================================================================== #
    # Container Properties to access child widgets:                                                                        #
    # ==================================================================================================================== #
    @property
    def config_widgets(self):
        """The pf_widgets property."""
        return self.ui.widgets_list
    @config_widgets.setter
    def config_widgets(self, value):
        self.ui.widgets_list = value
        self.rebuild_neuron_id_to_widget_map()
        
    @property
    def neuron_id_config_widgets_map(self):
        """The neuron_id_pf_widgets_map property."""
        return self._neuron_id_config_widgets_map
    @neuron_id_config_widgets_map.setter
    def neuron_id_config_widgets_map(self, value):
        self._neuron_id_config_widgets_map = value
    
    def rebuild_neuron_id_to_widget_map(self):
        """ must be called after changing self.widgets_list """
        self._neuron_id_config_widgets_map = dict()
        for a_widget in self.ui.widgets_list:
            curr_widget_config = a_widget.config_from_state()
            self._neuron_id_config_widgets_map[curr_widget_config.neuron_id] = a_widget 
        

    # @QtCore.pyqtSlot()
    # def onRefreshAction(self):
    #     print(f'PlacefieldVisualSelectionControlsBarWidget.onRefreshAction()')
    #     self.sigRefresh.emit(self)
    #     # self.done(QtCore.Qt.WA_DeleteOnClose)

    @QtCore.pyqtSlot(object)
    def applyUpdatedConfigs(self, active_configs_map):
        """ Updates the placefield Qt widgets provided in the neuron_id_pf_widgets_map from the active_configs_map
        
        Inputs:
            Both maps should have keys of neuron_id <int>
        
        Usage:
            ipcDataExplorer.neuron_id_pf_widgets_map = _build_id_index_configs_dict(pf_widgets)
            apply_updated_configs_to_pf_widgets(ipcDataExplorer.neuron_id_pf_widgets_map, active_configs_map)
        """
        print(f'PlacefieldVisualSelectionControlsBarWidget.applyUpdatedConfigs(active_configs_map: {active_configs_map})')
        ## Update placefield selection GUI widgets from updated configs:
        for neuron_id, updated_config in active_configs_map.items():
            """ Update the placefield selection GUI widgets from the updated configs using the .update_from_config(render_config) fcn """
            # update the widget:
            self.neuron_id_config_widgets_map[neuron_id].update_from_config(updated_config)


    def configsFromStates(self):
        """ gets the current config from the state of each child pf_widget (a list of SingleNeuronPlottingExtended) """
        return [a_widget.config_from_state() for a_widget in self.ui.widgets_list]
        
    def configMapFromChildrenWidgets(self):
        """ returns a map with keys of neuron_id and values of type SingleNeuronPlottingExtended """
        return {a_config.neuron_id:a_config for a_config in self.configsFromStates()}

    

    @classmethod
    def apply_updated_configs_to_neuron_widgets(cls, neuron_id_config_widgets_map, active_configs_map):
        """ Updates the placefield Qt widgets provided in the neuron_id_pf_widgets_map from the active_configs_map
            
        Inputs:
            Both maps should have keys of neuron_id <int>
        
        Usage:
            ipcDataExplorer.neuron_id_pf_widgets_map = _build_id_index_configs_dict(pf_widgets)
            apply_updated_configs_to_pf_widgets(ipcDataExplorer.neuron_id_pf_widgets_map, active_configs_map)
        """
        ## Update placefield selection GUI widgets from updated configs:
        for neuron_id, updated_config in active_configs_map.items():
            """ Update the placefield selection GUI widgets from the updated configs using the .update_from_config(render_config) fcn """
            # update the widget:
            neuron_id_config_widgets_map[neuron_id].update_from_config(updated_config)




## Start Qt event loop
if __name__ == '__main__':
    # app = QApplication([])
    app = pg.mkQApp('NeuronVisualSelectionControlsWidget_test')
    widget = NeuronVisualSelectionControlsWidget()
    widget.show()
    sys.exit(app.exec_())
