import sys
from copy import deepcopy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
# import pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph_extensions.mixins.SelectableItemMixin import SelectableItemMixin

__all__ = ['SelectableTextItem']


class SelectableTextItem(SelectableItemMixin, pg.TextItem):
    """ 
    from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.SelectableTextItem import SelectableTextItem
    
    
    """
    sigSelectedChanged = pyqtSignal(object, bool) ## emitted when mouse is clicked. Check for event.isAccepted() to see whether the event has already been acted on.
    

    def __init__(self, text='', color=(200, 200, 200), anchor=(1, 0), is_selected:bool=False, should_use_parent_width:bool=True, **kwargs):
        super().__init__(text=text, color=color, anchor=anchor, **kwargs)
        self.is_selected = is_selected  # Track selection state
        self.should_use_parent_width = should_use_parent_width
        self.init_UI()

    def init_UI(self):
        # # Create a rectangle item as a child
        self.rect_item = pg.QtGui.QGraphicsRectItem(self)
        # Set the pen (border) and brush (fill) properties
        self.rect_item.setPen(pg.mkPen('w', width=1))  # White border
        self.rect_item.setBrush(pg.mkBrush(None))      # No fill
        self.rect_item.hide() # Initially hidden
        self.updateRect()
        self.init_UI_SelectableItemMixin(is_selected=self.is_selected)


    def setSelected(self, selected):
        if selected:
            self.rect_item.show()
        else:
            self.rect_item.hide()

    def updateRect(self):
        # Update the rectangle to match the text bounding rectangle
        # rect = self.textItem.boundingRect()
        rect = self.boundingRect() # QRectF(-14.0, 0.0, 14.0, 21.0) # looks good
        if self.should_use_parent_width:
            if self.parentWidget() is not None:
                parent_width = self.parentWidget().width()
                if parent_width > rect.width():
                    rect.setWidth(parent_width)
                
        self.rect_item.setRect(rect)

    ## Override from `SelectableItemMixin`
    def updateSelection(self):
        """ updates the visual displays based on the value of self.is_selected """
        ## rect mode:
        self.updateRect()
        self.setSelected(selected=self.is_selected)
        # print(f'self.is_selected: {self.is_selected}')
        # self.update()  # Trigger a repaint
        

    # def paint(self, p, *args):
    #     # Draw the text using the parent class method
    #     super().paint(p, *args)

    #     if self.is_selected:
    #         # Get the bounding rectangle of the text
    #         br = self.textItem.boundingRect()
    #         # self.textItem
    #         # Calculate the position adjustment based on the anchor
    #         anchor_x = -br.left() - br.width() * self.anchor[0]
    #         anchor_y = -br.top() - br.height() * self.anchor[1]

    #         # Save the painter's state
    #         p.save()
    #         # Translate the painter to the text position
    #         p.translate(anchor_x, anchor_y)

    #         # Set the pen for the rectangle border
    #         p.setPen(pg.mkPen('w', width=1))  # White border
    #         p.setBrush(QtCore.Qt.NoBrush)     # No fill

    #         # Draw the rectangle around the text bounding rect
    #         # p.drawRoundedRect(br, 5, 5)  # Rounded rectangle with radii 5
    #         p.drawRoundedRect(br, 1, 1)  # Rounded rectangle with radii 5

    #         # Restore the painter's state
    #         p.restore()
            

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