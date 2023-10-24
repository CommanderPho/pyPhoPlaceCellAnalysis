# pyuic5 PlacefieldVisualSelectionWidget.ui -o PlacefieldVisualSelectionWidget.py -x
# PlacefieldVisualSelectionWidgetBase
# pyuic5 PlacefieldVisualSelectionWidgetBase.ui -o PlacefieldVisualSelectionWidgetBase.py -x

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from matplotlib.colors import to_hex # required for QColor conversion to hex


from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.PlacefieldVisualSelectionWidgetBase import Ui_rootForm # Generated file from .ui

# For compatibility with the panel ui version:
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended


class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionWidget."""
 
    # spike_config_changed = QtCore.pyqtSignal(list, list, bool) # change_unit_spikes_included(self, neuron_IDXs=None, cell_IDs=None, are_included=True)
    # tuning_curve_display_config_changed = QtCore.pyqtSignal(list, list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    spike_config_changed = QtCore.pyqtSignal(bool) # change_unit_spikes_included(self, neuron_IDXs=None, cell_IDs=None, are_included=True)
    tuning_curve_display_config_changed = QtCore.pyqtSignal(list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    
    # Send a SingleNeuronPlottingExtended config state instead of a QtGui.QColor for easy access.
    sig_neuron_color_changed = QtCore.pyqtSignal(object) # send the updated color as a QtGui.QColor
    
    # update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
    # finish_signal = QtCore.pyqtSignal(float, float)
 
    # enable_debug_print = True
    enable_debug_print = False
    
            
    @property
    def name(self):
        """The name property."""
        if self._name is None:
            self._name = self.ui.btnTitle.text
        return self._name 
    @name.setter
    def name(self, value):
        self._name = value
        self.ui.groupBox.setTitle(self._name) # set a new value like "pf[i]"
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



    def __init__(self, *args, config=None, **kwargs):
        # initialize member variables
        self.enable_debug_print = PlacefieldVisualSelectionWidget.enable_debug_print
        self._name = None
        self._color = None
        self._isVisible = None
        self._spikesVisible = None
        super(PlacefieldVisualSelectionWidget, self).__init__(*args, **kwargs)

        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        # Final UI Refinements:
        self.initUI()


        # If an initial config is provided, set it up using that:
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
        


        # Setup self.ui.chkbtnPlacefield:
        self.ui.chkbtnPlacefield.toggled.connect(self.togglePlacefieldVisibility)
        self.ui.chkbtnSpikes.toggled.connect(self.toggleSpikeVisibility)
  
        # Connect the color button:
        self.ui.btnColorButton.sigColorChanging.connect(self.on_color_button_changing)
        self.ui.btnColorButton.sigColorChanged.connect(self.on_color_button_changed)  
  
  
        
  
    def initUI(self):
        self.ui.btnTitle.hide() # set the title button as hidden, since it's redundent
        self.ui.btnColorButton.setEnabled(True)
        
        
        # Setup the button style properties:
        # self.ui.chkbtnPlacefield.setStyleSheet(PlacefieldVisualSelectionWidget.css_toolButton())
        # self.ui.chkbtnSpikes.setStyleSheet(PlacefieldVisualSelectionWidget.css_toolButton())
        # Disable Changing the color button:
        # self.ui.btnColorButton.SetEnabled
  
  
    
  
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


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("PlacefieldVisualSelectionWidget Example")
    widget = PlacefieldVisualSelectionWidget()
    widget.show()
    pg.exec()


# hjhg

# QToolButton {
#     font-size: 10px;
#     background-color: green;
#     color: black;
#     border: 2px green;
#     border-radius: 22px;
#     border-style: outset;
# }
# QToolButton:hover {
#     background: qradialgradient(
#         cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,
#         radius: 1.35, stop: 0 grey, stop: 1 lightgray
#         );
# }
# QToolButton:enabled{
#     color: black;
#     font:  10px;
#     background: green;
#     background-color: red;
#     border: 1px black;
#     border-style: outset;
# }
# QToolButton:pressed {
#     color: white;
#     background: yellow;
# }
# QToolButton:disabled {
#     color: gray;
#     background-color: gray;
#     border: 1px black;
#     border-style: outset;                
# }
# QToolButton:checked{
#     color: black; 
#     font:  12px;   
#     font: bold;
#     background-color: red;
#     border: 1px black;
#     border-style: outset;
# }
# QToolButton:!checked{
#     color: black; 
#     font:  12px;   
#     font: bold;
#     background-color: green;
#     border: 1px black;
#     border-style: outset;
# }
