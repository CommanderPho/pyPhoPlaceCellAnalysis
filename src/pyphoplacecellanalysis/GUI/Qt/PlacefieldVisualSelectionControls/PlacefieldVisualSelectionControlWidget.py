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
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended



class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionWidget."""
 
    # spike_config_changed = QtCore.pyqtSignal(list, list, bool) # change_unit_spikes_included(self, cell_IDXs=None, cell_IDs=None, are_included=True)
    # tuning_curve_display_config_changed = QtCore.pyqtSignal(list, list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    spike_config_changed = QtCore.pyqtSignal(bool) # change_unit_spikes_included(self, cell_IDXs=None, cell_IDs=None, are_included=True)
    tuning_curve_display_config_changed = QtCore.pyqtSignal(list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    # update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
    # finish_signal = QtCore.pyqtSignal(float, float)
 
    enable_debug_print = False
    
    def __init__(self, *args, **kwargs):
        super(PlacefieldVisualSelectionWidget, self).__init__(*args, **kwargs)
        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        # Final UI Refinements:
        self.initUI()
        
        # initialize member variables
        self.enable_debug_print = PlacefieldVisualSelectionWidget.enable_debug_print
        self._name = None
        self._color = None
        self._isVisible = None
        self._spikesVisible = None
        
        # Setup self.ui.btnColorButton:
        def change(btn):
            if self.enable_debug_print:
                print("btnColorButton change", btn.color())
            self._color = btn.color()
        def done(btn):
            if self.enable_debug_print:
                print("btnColorButton done", btn.color())
            self._color = btn.color()
        
        self.ui.btnColorButton.sigColorChanging.connect(change)
        self.ui.btnColorButton.sigColorChanged.connect(done)

        # Setup self.ui.chkbtnPlacefield:
        self.ui.chkbtnPlacefield.toggled.connect(self.togglePlacefieldVisibility)
        self.ui.chkbtnSpikes.toggled.connect(self.toggleSpikeVisibility)
  
    def initUI(self):
        self.ui.btnTitle.hide() # set the title button as hidden, since it's redundent
        
        # Setup the button style properties:
        # self.ui.chkbtnPlacefield.setStyleSheet(PlacefieldVisualSelectionWidget.css_toolButton())
        # self.ui.chkbtnSpikes.setStyleSheet(PlacefieldVisualSelectionWidget.css_toolButton())
        # Disable Changing the color button:
        # self.ui.btnColorButton.SetEnabled
  
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
            print(f'self.color: {self.color} - self.color.name(): {self.color.name()}')
        
        # How to convert a QColor into a HexRGB String:
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        color_hex_str = self.color.name(QtGui.QColor.HexRgb) 
        if self.enable_debug_print:
            print(f'    hex: {color_hex_str}')
        
        # also I think the original pf name was formed by adding crap...
        ## see 
        # ```curr_pf_string = f'pf[{render_config.name}]'````
        ## UPDATE: this doesn't seem to be a problem. The name is successfully set to the ACLU value in the current state.
        return SingleNeuronPlottingExtended(name=self.name, isVisible=self.isVisible, color=color_hex_str, spikesVisible=self.spikesVisible)


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