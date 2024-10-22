import sys
from copy import deepcopy
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
# import pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg

__all__ = ['SelectableLabelItem']


class SelectableLabelItem(pg.LabelItem):
    """ A PlotItem subclass that can be selected/deselected via left-click
    
    
    from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.LabelItem.ClickableLabelItem import SelectableLabelItem
        
    # curr_plot: pg.PlotItem = root_render_widget.addPlot(row=(curr_row + plots_start_row_idx), col=curr_col, title=formatted_title) # pg.PlotItem
    # SelectablePlotItem version:
    curr_plot: SelectablePlotItem = SelectablePlotItem(title=formatted_title, is_selected=False)
    root_render_widget.addItem(curr_plot, row=(curr_row + plots_start_row_idx), col=curr_col)
    
    """
    sigSelectedChanged = pyqtSignal(object, bool) ## emitted when mouse is clicked. Check for event.isAccepted() to see whether the event has already been acted on.
    
    def __init__(self, *args, is_selected:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_selected = is_selected  # Track selection state
        self.init_UI()

    def init_UI(self):
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
        plain_text: str = deepcopy(self.text)
        if self.is_selected:
            self.setText(plain_text, size='16pt', bold=True) 
        else:
            self.setText(plain_text, size='12pt', bold=False) 



# if __name__ == "__main__":
#     app = pg.mkQApp('Test SelectablePlotItem')

#     root_render_widget = pg.GraphicsLayoutWidget(show=True)

#     # Variables for demonstration
#     plots_start_row_idx = 0
#     formatted_title = "My Plot Title"

#     # Example grid positions
#     for curr_row in range(3):
#         for curr_col in range(2):
#             curr_plot = SelectablePlotItem(title=formatted_title)
#             root_render_widget.addItem(curr_plot, row=(curr_row + plots_start_row_idx), col=curr_col)
#             curr_plot.plot([1, 2, 3], [4, 5, 6])
            
#     ## select only the last one
#     curr_plot.perform_update_selected(new_is_selected=True)
#     sys.exit(app.exec_())