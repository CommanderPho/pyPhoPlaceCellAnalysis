### Graphics Items Children Discovery - Programmatically getting rows/columns of GraphicsLayoutWidget
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph.graphicsItems as graphicsItems
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem import PlotItem #, PlotCurveItem
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem #, PlotCurveItem

""" 



plot_items = [a_child for a_child in main_graphics_layout_widget.items() if isinstance(a_child, (PlotItem))] # ScatterPlotItem
plot_items

graphics_layout = main_graphics_layout_widget.ci # <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>

graphics_layout.rows
# {1: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb9dc10>},
#  2: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb164c0>}}

graphics_layout.items ## item: [(row, col), (row, col), ...]  lists all cells occupied by the item


"""


# def recover_graphics_layout_widget_item_indicies(graphics_layout_widget, debug_print=False):
#     """ Recovers the row/column indicies for the items of a graphics_layout_widget 
    
#     Example:
#         found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col) = recover_graphics_layout_widget_item_indicies(main_graphics_layout_widget)
        
#     """
#     # num_plot_items = len(plot_items)
#     num_plot_items = len(graphics_layout_widget.items())
#     max_rows = num_plot_items
#     max_cols = num_plot_items
    
#     # found_item_indicies = []
#     found_item_rows = []
#     found_item_cols = []
#     found_items_list = []

#     for row_i in np.arange(max_rows):
#         for col_i in np.arange(max_cols):
#             found_item = graphics_layout_widget.getItem(row_i, col_i)
#             if found_item is not None:
#                 found_item_rows.append(row_i)
#                 found_item_cols.append(col_i)
#                 found_items_list.append(found_item)
#                 if debug_print:
#                     print(f'found_item[row_i:{row_i}, col_i:{col_i}]: {found_item}')
#         # try:
#         #     print(f'main_graphics_layout_widget.itemIndex(): {main_graphics_layout_widget.itemIndex(a_child)}')
#         # except Exception as e:
#         #     pass

#     found_max_row = max(found_item_rows)
#     found_max_col = max(found_item_cols)
#     if debug_print:
#         print(f'found_max_row: {found_max_row}, found_max_col: {found_max_col}')
#     return found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col)

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
    graphics_layout = graphics_layout_widget.ci # <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>
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
