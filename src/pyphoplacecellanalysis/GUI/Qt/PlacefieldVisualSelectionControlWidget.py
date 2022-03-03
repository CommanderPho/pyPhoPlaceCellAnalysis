# pyuic5 PlacefieldVisualSelectionWidget.ui -o PlacefieldVisualSelectionWidget.py -x
# PlacefieldVisualSelectionWidgetBase
# pyuic5 PlacefieldVisualSelectionWidgetBase.ui -o PlacefieldVisualSelectionWidgetBase.py -x

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp


from .PlacefieldVisualSelectionWidgetBase import Ui_rootForm # Generated file from .ui

# For compatibility with the panel ui version:
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended



class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionWidget."""
 
    def __init__(self, *args, **kwargs):
        super(PlacefieldVisualSelectionWidget, self).__init__(*args, **kwargs)
        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        # self.show() # Show the GUI
        
        # Setup self.ui.btnColorButton:
        def change(btn):
            print("change", btn.color())
        def done(btn):
            print("done", btn.color())
        self.ui.btnColorButton.sigColorChanging.connect(change)
        self.ui.btnColorButton.sigColorChanged.connect(done)

        # Setup self.ui.chkbtnPlacefield:
        # def pf_check_changed(btn):
        #     print(f'pf_check_changed({btn})')
        # self.ui.chkbtnPlacefield.toggled.connect(pf_check_changed)
        self.ui.chkbtnPlacefield.toggled.connect(self.togglePlacefieldVisibility)
        self.ui.chkbtnSpikes.toggled.connect(self.toggleSpikeVisibility)
  
    @QtCore.pyqtSlot(bool)
    def togglePlacefieldVisibility(self, value):
        print(f'_on_toggle_plot_visible_changed(value: {value})')
        if self._callbacks is not None:
            self._callbacks['pf'](self.config_from_state()) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for pf value changes!')
    
    
    @QtCore.pyqtSlot(bool)
    def toggleSpikeVisibility(self, value):
        print(f'_on_toggle_spikes_visible_changed(value: {value})')
        if self._callbacks is not None:
            updated_config = self.spikesVisible
            self._callbacks['spikes'](bool(self.spikesVisible)) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for spikes value changes!')
        
  
  
    @property
    def name(self):
        """The name property."""
        return self.ui.btnTitle.text
    @name.setter
    def name(self, value):
        self.ui.groupBox.setTitle(value) # set a new value like "pf[i]"
        self.ui.btnTitle.setText(value)
  
    @property
    def spikesVisible(self):
        """The spikesVisible property."""
        return self.ui.chkbtnSpikes.checked
    @spikesVisible.setter
    def spikesVisible(self, value):
        self.ui.chkbtnSpikes.setChecked(value)
        
    @property
    def isVisible(self):
        """The isVisible property."""
        return self.ui.chkbtnPlacefield.checked
    @isVisible.setter
    def isVisible(self, value):
        self.ui.chkbtnPlacefield.setChecked(value)
        
    @property
    def color(self):
        """The color property."""
        return self.ui.btnColorButton.color()
    @color.setter
    def color(self, value):
        self.ui.btnColorButton.setColor(value, finished=True)


 
    def update_from_config(self, config):
        self.name = config.name
        self.color = config.color
        self.isVisible = config.isVisible
        self.spikesVisible = config.spikesVisible

    
    def config_from_state(self):
        return SingleNeuronPlottingExtended(name=self.name, isVisible=self.isVisible, color=self.color, spikesVisible=self.spikesVisible)



## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("PlacefieldVisualSelectionWidget Example")
    widget = PlacefieldVisualSelectionWidget()
    widget.show()
    pg.exec()


