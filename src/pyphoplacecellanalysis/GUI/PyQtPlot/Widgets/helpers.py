### Graphics Items Children Discovery - Programmatically getting rows/columns of GraphicsLayoutWidget
from collections import namedtuple
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.graphicsItems as graphicsItems
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem import PlotItem #, PlotCurveItem
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem #, PlotCurveItem
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout import GraphicsLayout

# """ 



# plot_items = [a_child for a_child in main_graphics_layout_widget.items() if isinstance(a_child, (PlotItem))] # ScatterPlotItem
# plot_items

# graphics_layout = main_graphics_layout_widget.ci # <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>

# graphics_layout.rows
# # {1: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb9dc10>},
# #  2: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb164c0>}}

# graphics_layout.items ## item: [(row, col), (row, col), ...]  lists all cells occupied by the item


# """


def recover_graphics_layout_widget_item_indicies(graphics_layout_widget, debug_print=False):
    """ ✅WORKS✅ Recovers the row/column indicies for the items of a graphics_layout_widget 
    
    Example:
        
        found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col) = recover_graphics_layout_widget_item_indicies(main_graphics_layout_widget, debug_print=True)
        
    Output:
    
    Found Items:
        row_indicies: [1 4]
        indicies: [0 0]
        items: [<pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem object at 0x000001B5C917B160>, <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem object at 0x000001B5C93B2D30>]
    (found_unique_rows: [1 4], found_unique_cols: [0])
    (found_max_row: 4, found_max_col: 0)

    """
    # Need graphics_layout to be a GraphicsLayout object:
    if isinstance(graphics_layout_widget, GraphicsLayout):
        graphics_layout = graphics_layout_widget # input is already a GraphicsLayout object. <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>
    else:
        # for GraphicsLayoutWidget or GraphicsWindow:
        graphics_layout = graphics_layout_widget.ci

    assert isinstance(graphics_layout, GraphicsLayout), f"type(graphics_layout): {type(graphics_layout)}"
    plot_items_dict = graphics_layout.items ## item: [(row, col), (row, col), ...]  lists all cells occupied by the item
    num_plot_items = len(plot_items_dict)
    found_items_list = list(plot_items_dict.keys())
    plot_items_index_tuple_list = np.array([list(a_index_tuple[0]) for a_index_tuple in list(plot_items_dict.values())])
    plot_items_row_indicies = plot_items_index_tuple_list[:,0]
    plot_items_col_indicies = plot_items_index_tuple_list[:,1]
    
    found_item_rows = np.unique(plot_items_row_indicies)
    found_item_cols = np.unique(plot_items_col_indicies)
    
    num_found_rows = len(found_item_rows)
    num_found_cols = len(found_item_cols)
    
    found_max_row = max(found_item_rows)
    found_max_col = max(found_item_cols)
    if debug_print:
        print(f'Found Items:\n\trow_indicies: {plot_items_row_indicies}\n\tindicies: {plot_items_col_indicies}\n\titems: {found_items_list}\n(found_unique_rows: {found_item_rows}, found_unique_cols: {found_item_cols})\n(found_max_row: {found_max_row}, found_max_col: {found_max_col})')
        # print(f'plot_items_row_indicies: {plot_items_row_indicies}\nplot_items_col_indicies: {plot_items_col_indicies}\nfound_items_list: {found_items_list}\n(found_unique_rows: {found_item_rows}\nfound_unique_cols: {found_item_cols})\n(found_max_row: {found_max_row}, found_max_col: {found_max_col})')
    return found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col)




# ==================================================================================================================== #
# RectangleRenderTupleHelpers                                                                                          #
# ==================================================================================================================== #
QPenTuple = namedtuple('QPenTuple', ['color', 'width'])
QBrushTuple = namedtuple('QBrushTuple', ['color'])

class RectangleRenderTupleHelpers:
    """ class for use in copying, serializing, etc the list of tuples used by IntervalRectsItem

    Refactored out of `pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem.IntervalRectsItem` on 2022-12-05 

    Usage:

        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import RectangleRenderTupleHelpers

    Known Usages:
        Used in `IntervalRectsItem` and `CustomIntervalRectsItem` to copy themselves

        # Copy Constructors: _________________________________________________________________________________________________ #
        def __copy__(self):
            independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
            return CustomIntervalRectsItem(independent_data_copy)
        
        def __deepcopy__(self, memo):
            independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
            return CustomIntervalRectsItem(independent_data_copy)
            # return CustomIntervalRectsItem(copy.deepcopy(self.data, memo))


    """
    @staticmethod
    def QPen_to_dict(a_pen):
        return {'color': pg.colorStr(a_pen.color()),'width':a_pen.widthF()}

    @staticmethod
    def QBrush_to_dict(a_brush):
        return {'color': pg.colorStr(a_brush.color())} # ,'gradient':a_brush.gradient()


    @staticmethod
    def QPen_to_tuple(a_pen):
        return QPenTuple(color=pg.colorStr(a_pen.color()), width=a_pen.widthF())

    @staticmethod
    def QBrush_to_tuple(a_brush):
        return QBrushTuple(color=pg.colorStr(a_brush.color()))


    
    @classmethod
    def get_serialized_data(cls, tuples_data):
        """ converts the list of (float, float, float, float, QPen, QBrush) tuples into a list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) for serialization. """            
        return [(start_t, series_vertical_offset, duration_t, series_height, cls.QPen_to_dict(pen), cls.QBrush_to_dict(brush)) for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in tuples_data]

    
    @staticmethod
    def get_deserialized_data(seralized_tuples_data):
        """ converts the list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) tuples back to the original (float, float, float, float, QPen, QBrush) list
        
        Inverse operation of .get_serialized_data(...).
        
        Usage:
            seralized_tuples_data = RectangleRenderTupleHelpers.get_serialized_data(tuples_data)
            tuples_data = RectangleRenderTupleHelpers.get_deserialized_data(seralized_tuples_data)
        """ 
        return [(start_t, series_vertical_offset, duration_t, series_height, pg.mkPen(pen_color_hex), pg.mkBrush(**brush_color_hex)) for (start_t, series_vertical_offset, duration_t, series_height, pen_color_hex, brush_color_hex) in seralized_tuples_data]

    @classmethod
    def copy_data(cls, tuples_data):
        seralized_tuples_data = cls.get_serialized_data(tuples_data).copy()
        return cls.get_deserialized_data(seralized_tuples_data)




