# pyuic5 PlacefieldVisualSelectionWidget.ui -o PlacefieldVisualSelectionWidget.py -x
# PlacefieldVisualSelectionWidgetBase
# pyuic5 PlacefieldVisualSelectionWidgetBase.ui -o PlacefieldVisualSelectionWidgetBase.py -x

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphoplacecellanalysis.GUI.Qt.PhoUIContainer import PhoUIContainer
from .PlacefieldVisualSelectionWidgetBase import Ui_Form # Generated file from .ui


class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionWidget."""
 
    def __init__(self, *args, **kwargs):
        super(PlacefieldVisualSelectionWidget, self).__init__(*args, **kwargs)
        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        # self.show() # Show the GUI
        
        
  
  
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


