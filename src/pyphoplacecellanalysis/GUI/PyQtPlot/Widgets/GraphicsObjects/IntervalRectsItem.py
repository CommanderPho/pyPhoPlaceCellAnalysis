"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""
import copy
from typing import Callable, Tuple
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import RectangleRenderTupleHelpers
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.LegendItem import ItemSample, LegendItem # for custom legend

## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)
class IntervalRectsItem(pg.GraphicsObject):
    """ Created to render the 2D Intervals as rectangles in a pyqtgraph 
    
        Based on pyqtgraph's CandlestickItem example
       
    Rectangle Item Specification: 
        Renders rectangles, with each specified by a tuple of the form:
            (start_t, series_vertical_offset, duration_t, series_height, pen, brush)

        Note that this is analagous to the position arguments of `QRectF`:
            (left, top, width, height) and (pen, brush)
            
            
    TODO: BUG: Right click currently invokes the custom example context menu that allows you to select between blue/green etc. This is triggered even when you right click on an area that's between the actual interval rect items (when you click in the blank-space between rects).
        Want this to only be triggered when on an interval. And pass through to its parent otherwise.     
        
    #2025-07-22 18:18: - [x] Custom hover info tooltip text currently works, and the custom formatting function can be set via `self.format_item_tooltip_fn = _custom_format_tooltip_for_rect_data`. An example is provided in 
            
    Usage:
        Example 1 (basic):
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, main
            active_interval_rects_item = IntervalRectsItem(data)
            
            ## Add the active_interval_rects_item to the main_plot_widget: 
            main_plot_widget = spike_raster_window.spike_raster_plt_2d.plots.main_plot_widget # PlotItem
            main_plot_widget.addItem(active_interval_rects_item)

            ## Remove the active_interval_rects_item:
            main_plot_widget.removeItem(active_interval_rects_item)

            
        Example 2 (with custom tooltip function):
        
            def _custom_format_tooltip_for_rect_data(rect_index: int, rect_data_tuple: Tuple) -> str:
                start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
                end_t = start_t + duration_t
                tooltip_text = f"{name}[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}" # The tooltip is set generically here to 'PBEs', 'Replays' or whatever the dataseries name is
                return tooltip_text


            # Build the rendered interval item:
            new_interval_rects_item: IntervalRectsItem = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource, format_tooltip_fn=deepcopy(_custom_format_tooltip_for_rect_data))
            new_interval_rects_item._current_hovered_item_tooltip_format_fn = deepcopy(_custom_format_tooltip_for_rect_data)
        
    """
    pressed = False
    clickable = True
    hoverEnter = QtCore.pyqtSignal()
    hoverExit = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()
    ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush


    def __init__(self, data, format_tooltip_fn=None):
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        # note that the use of super() is often avoided because Qt does not 
        # allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
        self.generatePicture()
        self.setAcceptHoverEvents(True)
        self._current_hovered_rect = None  # Track which rectangle is currently hovered
        self._current_hovered_item_tooltip_format_fn = None
        if format_tooltip_fn is None:
            format_tooltip_fn = self._default_format_tooltip_for_rect_data
        self._current_hovered_item_tooltip_format_fn = format_tooltip_fn


    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        
        # White background bars:
        p.setPen(pg.mkPen('w'))
        p.setBrush(pg.mkBrush('r'))
        
        # for (series_offset, start_t, duration_t) in self.data:
            # # QRectF: (left, top, width, height)
            # p.drawRect(QtCore.QRectF(start_t, series_offset-series_height, duration_t, series_height))
            
        for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in self.data:
            p.setPen(pen)
            p.setBrush(brush) # filling of the rectangles by a passed color:
            # p.drawRect(QtCore.QRectF(start_t, series_vertical_offset-series_height, duration_t, series_height)) # QRectF: (left, top, width, height)
            p.drawRect(QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height)) # QRectF: (left, top, width, height)

        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())


    @property
    def format_item_tooltip_fn(self) -> Callable:
        """The format_item_tooltip_fn property."""
        return self._current_hovered_item_tooltip_format_fn
    @format_item_tooltip_fn.setter
    def format_item_tooltip_fn(self, value: Callable):
        self._current_hovered_item_tooltip_format_fn = value

    ## Copy Constructors:
    def __copy__(self):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy)
    
    def __deepcopy__(self, memo):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy)
        # return IntervalRectsItem(copy.deepcopy(self.data, memo))


    # ==================================================================================================================== #
    # Events Copied from https://github.com/CommanderPho/pyqt-xcode/blob/master/menurect.py                                #
    # ==================================================================================================================== #

    def hoverEnterEvent(self, event):
        if self.clickable:
            self.hoverEnter.emit()


    def hoverMoveEvent(self, event):
        """Handle hover move events to show tooltips for individual rectangles."""
        if not self.clickable:
            return
            
        # Get the position in item coordinates
        pos = event.pos()
        
        # Find which rectangle (if any) contains this position
        hovered_rect_index = self._get_rect_at_position(pos)
        
        if hovered_rect_index != self._current_hovered_rect:
            self._current_hovered_rect = hovered_rect_index
            
            if hovered_rect_index is not None:
                # Show tooltip for this rectangle
                global_pos = event.screenPos()
                self._show_tooltip_for_rect(hovered_rect_index, QtCore.QPoint(int(global_pos.x()), int(global_pos.y())))
            else:
                # Hide tooltip when not over any rectangle
                QtWidgets.QToolTip.hideText()

    def hoverLeaveEvent(self, event):
        if self.clickable:
            self.hoverExit.emit()
            # Hide tooltip when leaving the item
            QtWidgets.QToolTip.hideText()
            self._current_hovered_rect = None
            

    def mousePressEvent(self, event):
        if self.clickable:
            pressed = True


    def mouseReleaseEvent(self, event):
        if self.clickable:
            pressed = False
            self.clicked.emit()

    # ==================================================================================================================================================================================================================================================================================== #
    # Hover Event Handlers                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    def _get_rect_at_position(self, pos):
        """
        Find which rectangle (if any) contains the given position.
        Returns the index of the rectangle, or None if no rectangle contains the position.
        
        Args:
            pos: QtCore.QPointF in item coordinates
            
        Returns:
            int or None: Index of the rectangle containing the position, or None
        """
        for i, (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in enumerate(self.data):
            rect = QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height)
            if rect.contains(pos):
                return i
        return None
    
    @classmethod
    def _default_format_tooltip_for_rect_data(cls, rect_index: int, rect_data_tuple: Tuple) -> str:
        """ rect_data_tuple = self.data[rect_index]
        start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
        """
        start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data_tuple
        end_t = start_t + duration_t
        tooltip_text = f"Item[{rect_index}]\nStart: {start_t:.3f}\nEnd: {end_t:.3f}\nDuration: {duration_t:.3f}"
        return tooltip_text

    def _show_tooltip_for_rect(self, rect_index, global_pos):
        """
        Show tooltip for the specified rectangle.
        
        Args:
            rect_index: Index of the rectangle in self.data
            global_pos: Global screen position for tooltip
        """
        if rect_index is None or rect_index >= len(self.data):
            return
        rect_data_tuple = self.data[rect_index]
        assert self._current_hovered_item_tooltip_format_fn is not None, f"self._current_hovered_item_tooltip_format_fn is None!"
        # tooltip_text: str = self._default_format_tooltip_for_rect_data(rect_index=rect_index, rect_data_tuple=rect_data_tuple)
        tooltip_text: str = self._current_hovered_item_tooltip_format_fn(rect_index=rect_index, rect_data_tuple=rect_data_tuple)        
        QtWidgets.QToolTip.showText(global_pos, tooltip_text)
        
    def setToolTip(self, text):
        """
        Override setToolTip to provide custom behavior.
        
        Args:
            text: Tooltip text. If None or empty, enables per-rectangle tooltips.
                  If provided, shows this static text for the entire item.
        """
        print(f'WARNING: EpochRenderingMixin.setTooltip(text: "{text}") was called, but this would set a single, static tooltip for the entire graphics item and is very unlikely to be what you want to do!')
        raise NotImplementedError(f'WARNING: EpochRenderingMixin.setTooltip(text: "{text}") was called, but this would set a single, static tooltip for the entire graphics item and is very unlikely to be what you want to do!')
        # self._custom_tooltip = text
        
        # if text:
        #     # If tooltip text is provided, disable custom per-rectangle tooltips
        #     self._use_custom_tooltips = False
        #     # Call parent implementation to set static tooltip
        #     super().setToolTip(text)
        # else:
        #     # If no text provided, enable custom per-rectangle tooltips
        #     self._use_custom_tooltips = True
        #     # Clear any existing static tooltip
        #     super().setToolTip("")
        

    # ==================================================================================================================== #
    # Context Menu and Interaction Handling                                                                                #
    # ==================================================================================================================== #
    def mouseShape(self):
        """
        Return a QPainterPath representing the clickable shape of the curve

        """
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape
    
    


    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        print(f'IntervalRectsItem.mouseClickEvent(ev: {ev})')
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            # if self.mouseShape().contains(ev.pos()):
            #     ev.accept()
            #     self.sigClicked.emit(self, ev)
                
            
            if self.raiseContextMenu(ev):
                ev.accept() # note that I think this means it won't pass the right click along to its parent view, might messup widget-wide menus

    def raiseContextMenu(self, ev):
        """ works to spawn the context menu in the appropriate location """
        print(f'IntervalRectsItem.raiseContextMenu(ev: {ev})')
        menu = self.getContextMenus()
        
        # Let the scene add on to the end of our context menu
        # (this is optional)
        # menu = self.scene().addParentContextMenus(self, menu, ev)
        
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        """ builds the context menus as needed """
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            # self.menu.setTitle(self.name+ " options..")
            self.menu.setTitle("IntervalRectItem options..")
            
            green = QtGui.QAction("Turn green", self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            
            blue = QtGui.QAction("Turn blue", self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            
            alpha = QtWidgets.QWidgetAction(self.menu)
            alphaSlider = QtWidgets.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    # Define context menu callbacks
    def setGreen(self):
        # self.pen = pg.mkPen('g')
        print(f'.setGreen()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('g')
            override_brush = pg.mkBrush('g')
            self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
        
        # Need to regenerate picture
        self.generatePicture()
        # inform Qt that this item must be redrawn.
        self.update()

    def setBlue(self):
        # self.pen = pg.mkPen('b')
        # override_pen = pg.mkPen('b')
        print(f'.setBlue()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('b')
            override_brush = pg.mkBrush('b')
            self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
            
        # Need to regenerate picture
        self.generatePicture()
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)
        


class CustomLegendItemSample(ItemSample):
    """ A ItemSample that can render a legend item for `IntervalRectsItem`
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import CustomLegendItemSample
    
    legend = pg.LegendItem(offset=(-10, -10))
    legend.setParentItem(plt.graphicsItem())
    legend.setSampleType(CustomLegendItemSample)  

    """
    def __init__(self, item):
        super().__init__(item)
        self.item = item

    def paint(self, p, *args):
        # print(f'CustomItemSample.paint(self, p, *args)')
        if not isinstance(self.item, IntervalRectsItem):
            ## Call superclass paint
            # print(f'\t calling superclass, as type(self.item): {type(self.item)}')
            super().paint(p, *args)
        else:
            # Custom Implementation
            # print(f'\t calling custom implementation!')
            if not self.item.isVisible():
                p.setPen(pg.mkPen('w'))
                p.drawLine(0, 11, 20, 11) # draw flat white line
                return

            # Define the size of the rectangle
            rect_width = 20
            rect_height = 8

            # Calculate the top-left corner coordinates to center the rectangle
            top_left_x = (self.boundingRect().width() - rect_width) / 2
            top_left_y = (self.boundingRect().height() - rect_height) / 2

            ## start_t, series_vertical_offset, duration_t, series_height, pen, brush = rect_data
            # print(f'len(self.item.data): {len(self.item.data)}')

            # The first item is representitive of all items, don't draw the item over-and-over
            use_only_first_items: bool = True

            for rect_data in self.item.data:
                pen, brush = rect_data[4], rect_data[5]
                if (pen is not None) or (brush is not None):                   
                    p.setPen(pen)
                    p.setBrush(brush)
                    # p.drawRect(QtCore.QRectF(2, 2, 16, 16))
                    p.drawRect(QtCore.QRectF(top_left_x, top_left_y, rect_width, rect_height))
                    if use_only_first_items:
                        return # break, only needed to draw one item

        # print(f'done.')



# ==================================================================================================================== #
# MAIN TESTING                                                                                                         #
# ==================================================================================================================== #
def main():
    # data = [  ## fields are (series_offset, start_t, duration_t).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
    
    
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
        
    series_start_offsets = [1, 5, 7]
    
    # Have series_offsets which are centers and series_start_offsets which are bottom edges:
    curr_border_color = pg.mkColor('r')
    curr_border_color.setAlphaF(0.8)

    curr_fill_color = pg.mkColor('w')
    curr_fill_color.setAlphaF(0.2)

    # build pen/brush from color
    curr_series_pen = pg.mkPen(curr_border_color)
    curr_series_brush = pg.mkBrush(curr_fill_color)
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (40.0, 0.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (41.0, 1.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (44.0, series_start_offsets[0], 4.0, 1.0, curr_series_pen, curr_series_brush),
    #     (45.0, series_start_offsets[-1], 4.0, 1.0, curr_series_pen, curr_series_brush),
    # ]
    data = []
    step_x_offset = 0.5
    for i in np.arange(len(series_start_offsets)):
        curr_x_pos = (40.0+(step_x_offset*float(i)))
        data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))
        
    
    item = IntervalRectsItem(data)

    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')
    # # Adjust the left margin
    # plt.getPlotItem().layout.setContentsMargins(100, 10, 10, 10)  # left, top, right, bottom


    # Add custom legend
    legend = pg.LegendItem(offset=(-10, -10))
    legend.setParentItem(plt.graphicsItem())
    legend.setSampleType(CustomLegendItemSample)    
    legend.addItem(item, 'Custom Rects')

    

    # series_start_offsets = [1, 5, 7]
    # curr_border_color = pg.mkColor('r')
    # curr_border_color.setAlphaF(0.8)
    # curr_fill_color = pg.mkColor('w')
    # curr_fill_color.setAlphaF(0.2)
    # curr_series_pen = pg.mkPen(curr_border_color)
    # curr_series_brush = pg.mkBrush(curr_fill_color)
    # data = []
    # step_x_offset = 0.5
    # for i in np.arange(len(series_start_offsets)):
    #     curr_x_pos = (40.0 + (step_x_offset * float(i)))
    #     data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))

    # item = IntervalRectsItem(data)
    # item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    # plt = pg.plot()
    # plt.addItem(item)
    # plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')

    # # Add custom legend
    # legend = CustomLegendItem(offset=(-10, -10))
    # legend.setParentItem(plt.graphicsItem())
    # legend.addItem(item, 'Custom Rects')

def main2():
    series_start_offsets = [1, 5, 7]
    curr_border_color = pg.mkColor('r')
    curr_border_color.setAlphaF(0.8)
    curr_fill_color = pg.mkColor('w')
    curr_fill_color.setAlphaF(0.2)
    curr_series_pen = pg.mkPen(curr_border_color)
    curr_series_brush = pg.mkBrush(curr_fill_color)
    data = []
    step_x_offset = 0.5
    for i in np.arange(len(series_start_offsets)):
        curr_x_pos = (40.0 + (step_x_offset * float(i)))
        data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))

    item = IntervalRectsItem(data)
    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')
    # Adjust the left margin
    # plt.getPlotItem().layout.setContentsMargins(100, 10, 10, 10)  # left, top, right, bottom
    plt.getPlotItem().layout.setContentsMargins(10, 10, 100, 10)  # left, top, right, bottom

    # Add custom legend in the right margin
    legend = LegendItem(offset=(100, -10))  # Adjust the x-offset as needed
    legend.setParentItem(plt.graphicsItem())
    legend.addItem(CustomLegendItemSample(item), 'Custom Rects')


if __name__ == '__main__':
    
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    # main()
    main2()
    pg.exec()
    