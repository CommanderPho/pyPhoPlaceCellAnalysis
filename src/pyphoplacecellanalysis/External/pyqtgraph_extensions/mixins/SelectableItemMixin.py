import sys
from copy import deepcopy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
# import pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg

__all__ = ['SelectableItemMixin']

class SelectableItemMixin:
    """ 
    from pyphoplacecellanalysis.External.pyqtgraph_extensions.mixins.SelectableItemMixin import SelectableItemMixin
    
    """
    sigSelectedChanged = pyqtSignal(object, bool) ## emitted when mouse is clicked. Check for event.isAccepted() to see whether the event has already been acted on.
    
    def init_UI_SelectableItemMixin(self, is_selected:bool=False):
        self.is_selected = is_selected  # Track selection state
        self.setFlag(self.ItemIsSelectable, True)
        self.setFlag(self.ItemIsFocusable, True)
        self.perform_update_selected(new_is_selected=self.is_selected, force_update=True)


    def mousePressEvent(self, ev):
        """ enables togglging selection status on a mousePress event. """
        if ev.button() == QtCore.Qt.LeftButton:
            self.perform_update_selected(new_is_selected=(not self.is_selected))
            ev.accept()
        else:
            super().mousePressEvent(ev)
            

    def perform_update_selected(self, new_is_selected:bool, force_update:bool=False):
        """ programmatically updates the internal `self.is_selected` property and calls update/signals if change occured. 
        
        """
        was_selected: bool = deepcopy(self.is_selected)
        did_selected_status_change: bool = (was_selected != new_is_selected)
        
        if (did_selected_status_change or force_update):
            # emit changed signal
            self.is_selected = new_is_selected
            self.sigSelectedChanged.emit(self, new_is_selected)
            self.updateSelection()
        

    def updateSelection(self):
        """ updates the visual displays based on the value of self.is_selected """
        raise NotImplementedError(f'Implementors must override this method')



