from enum import Enum
from pathlib import Path
import numpy as np

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.PhoMainAppWindowBase import PhoMainAppWindowBase # for pyqtplot_plot_image


def pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin = 2.0, debug_print=False):
    """ Returns the proper bounds for the image, and the proper x_range and y_range given the margin.
    Used by pyqtplot_plot_image_array(...) to plot binned data.

    Usage:
    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
    

        # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
    
    """
    global_min_x = np.nanmin(xbin_edges)
    global_max_x = np.nanmax(xbin_edges)

    global_min_y = np.nanmin(ybin_edges)
    global_max_y = np.nanmax(ybin_edges)

    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y

    if debug_print:
        print(f'global_min_x: {global_min_x}, global_max_x: {global_max_x}, global_min_y: {global_min_y}, global_max_y: {global_max_y}\nglobal_width: {global_width}, global_height: {global_height}')
    # Get rect image extent in the form [x, y, w, h]:
    image_bounds_extent = [global_min_x, global_min_y, global_width, global_height]

    x_range = (global_min_x-margin, global_max_x+margin)
    y_range = (global_min_y-margin, global_max_y+margin)

    return image_bounds_extent, x_range, y_range
    

def pyqtplot_plot_image(xbin_edges, ybin_edges, image, enable_LUT_Histogram=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Single image plot using pyqtplot: 
    Holy crap! It actually works to plot the maze, and the adjustable slider works as well!
    
    # Example: test single image plot:
        curr_im = np.squeeze(active_one_step_decoder.ratemap.normalized_tuning_curves[0,:,:]) # (43, 63, 63)
        app, win, imv = pyqtplot_plot_image(active_one_step_decoder.xbin, active_one_step_decoder.ybin, curr_im)
        win.show()
    """
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    if app is None:
        app = pg.mkQApp("pyqtplot_plot_image Figure")
        
        
    # image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    
        
    if root_render_widget is None:
        if parent_root_widget is None:
            # Create window to hold the image:
            
            # parent_root_widget = QtGui.QMainWindow()
            parent_root_widget = PhoMainAppWindowBase()
            parent_root_widget.resize(800,800)
        
        # Build a single image view to display the image:
        root_render_widget = pg.ImageView()
        parent_root_widget.setCentralWidget(root_render_widget)
        # imv.setImage(image, xvals=np.linspace(1., 3., data.shape[0]))
        parent_root_widget.show()
        parent_root_widget.setWindowTitle('pyqtplot image')

    ## Display the data and assign each frame a time value from 1.0 to 3.0
    root_render_widget.setImage(image, xvals=xbin_edges)
    # Set the color map:
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
    root_render_widget.setColorMap(cmap)
    
    # if enable_LUT_Histogram:
    #     lut = pg.HistogramLUTItem(orientation="horizontal")
    #     imv.addItem(lut)
    #     imv.setLookupTable(lut, autoLevel=True)
    #     h = imv.getHistogram()
    #     lut.plot.setData(*h)

    # bar = pg.ColorBarItem( values= (0, 20_000), cmap=cm ) # prepare interactive color bar
    # Have ColorBarItem control colors of img and appear in 'plot':
    # bar.setImageItem(image, insert_in=imv) 

    return app, parent_root_widget, root_render_widget
 
 
# ==================================================================================================================== #
# UI Building Helpers                                                                                                  #
# ==================================================================================================================== #

from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

class LayoutScrollability(ExtendedEnum):
    """Whether the layout is scrollable or not. Used by """
    NON_SCROLLABLE = "non_scrollable"
    SCROLLABLE = "scrollable"
    
    @property
    def is_scrollable(self):
        return LayoutScrollability.is_scrollableList()[self]

    # Static properties
    @classmethod
    def is_scrollableList(cls):
        return cls.build_member_value_dict([False, True])


def _perform_build_root_graphics_layout_widget_ui(ui:PhoUIContainer, is_scrollable: bool = True):
    """ just adds the widgets required to make the main graphics layoutr scrollable

    """
    ui.graphics_layout = pg.GraphicsLayoutWidget()

    if is_scrollable:
        ui.graphics_layout.setFixedWidth(1000)
        # ui.graphics_layout.setMinimumWidth(1000)
        ui.graphics_layout.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        # Builds QScrollArea:
        ui.scrollAreaWidget = QtWidgets.QScrollArea()
        ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        ui.scrollAreaWidget.setWidget(ui.graphics_layout)
    else:
        ui.scrollAreaWidget = None

    return ui


def build_root_graphics_layout_widget_ui(name, window_title=None, ui=None):
    """ Updates or builds the ui properties to display a GraphicsLayoutWidget:
    ## **Non-Scrollable** Version of `build_scrollable_graphics_layout_widget_ui`
    Usage:
    ## Build non-scrollable UI version:
    ui = build_root_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    
    """
    if ui is None:
        ui = PhoUIContainer(name=name)
        ui.connections = PhoUIContainer(name=name)
        
    if window_title is None:
        window_title = name
    
    ## Plot Version:
    # ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
    ui = _perform_build_root_graphics_layout_widget_ui(ui, is_scrollable=False)

    ui.graphics_layout.setWindowTitle(window_title)
    ui.graphics_layout.resize(1000, 800)
    # lw.ci.setBorder((50, 50, 100))
    return ui

def build_scrollable_graphics_layout_widget_ui(name, window_title=None, ui=None):
    """ Updates or builds the ui properties to display a GraphicsLayoutWidget with scrollable rows:
    ## **Scrollable** Version of `build_root_graphics_layout_widget_ui`
    Usage:
    ## Build scrollable UI version:
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ui.rootWindow.show()
    
    """
    if ui is None:
        ui = PhoUIContainer(name=name)
        ui.connections = PhoUIContainer(name=name)
        
    if window_title is None:
        window_title = name
    
    ui.rootWindow = QtWidgets.QMainWindow()
    ui.rootWindow.resize(1000, 800)

    ui = _perform_build_root_graphics_layout_widget_ui(ui, is_scrollable=True)

    ui.rootWindow.setCentralWidget(ui.scrollAreaWidget)
    ui.rootWindow.setWindowTitle(window_title)
    
    return ui

def build_scrollable_graphics_layout_widget_with_nested_viewbox_ui(name, window_title=None, ui=None):
    """ Updates or builds the ui properties to display a GraphicsLayoutWidget with scrollable rows:
    Usage:
    ## Build scrollable UI version:
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=params.window_title, ui=ui)
    ui.rootWindow.show()
    
    """
    if ui is None:
        ui = PhoUIContainer(name=name)
        ui.connections = PhoUIContainer(name=name)
        
    if window_title is None:
        window_title = name
    
    ui = build_scrollable_graphics_layout_widget_ui(name, window_title=window_title, ui=ui)
    ## Adds the root_viewbox to the graphics layout
    # ui.root_viewbox = ui.graphics_layout.addViewBox(enableMouse=False) # lockAspect=True
    
    # ui.root_viewbox = ui.graphics_layout.addViewBox(enableMouse=False, defaultPadding=0.0, enableMenu=False, border='r') # lockAspect=True
    # pg.mkColor('r')
    # ui.root_viewbox.setBackgroundColor('r')
    
    # ui.root_viewbox = ui.graphics_layout.addLayout(enableMouse=False, defaultPadding=0.0, enableMenu=False, border='r') # lockAspect=True
    
    ui.nested_graphics_layout = ui.graphics_layout.addLayout(border=(50,0,0))
    ui.nested_graphics_layout.setContentsMargins(10, 10, 10, 10)
    return ui



# ==================================================================================================================== #
# Exporting Helpers                                                                                                    #
# ==================================================================================================================== #




    
    
# def build_vertically_scrollable_graphics_area():
#     """ copied from pyphoplacecellanalysis.External.pyqtgraph.examples.colorMaps example """
#     app = pg.mkQApp()
    
#     ui = PhoUIContainer('')

#     ui.rootWindow = QtWidgets.QMainWindow()
#     ui.rootWindow.resize(1000,800)

#     ui.graphics_layout = pg.GraphicsLayoutWidget()
#     ui.graphics_layout.setFixedWidth(1000)
#     ui.graphics_layout.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

#     ui.scrollAreaWidget = QtWidgets.QScrollArea()
#     ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#     ui.scrollAreaWidget.setWidget(ui.graphics_layout)
#     ui.rootWindow.setCentralWidget(ui.scrollAreaWidget)
#     ui.rootWindow.setWindowTitle('pyqtgraph example: Color maps')
#     ui.rootWindow.show()

#     # bar_width = 32
#     # bar_data = pg.colormap.modulatedBarData(width=bar_width)

#     # num_bars = 0

#     # def add_heading(lw, name):
#     #     global num_bars
#     #     lw.addLabel('=== '+name+' ===')
#     #     num_bars += 1
#     #     lw.nextRow()

#     # def add_bar(lw, name, cm):
#     #     global num_bars
#     #     lw.addLabel(name)
#     #     imi = pg.ImageItem( bar_data )
#     #     imi.setLookupTable( cm.getLookupTable(alpha=True) )
#     #     vb = lw.addViewBox(lockAspect=True, enableMouse=False)
#     #     vb.addItem( imi )
#     #     num_bars += 1
#     #     lw.nextRow()

#     # # Run the setup:
#     # add_heading(lw, 'local color maps')
#     # list_of_maps = pg.colormap.listMaps()
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.swapcase() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name)
#     #     add_bar(lw, map_name, cm)

#     # add_heading(lw, 'Matplotlib import')
#     # list_of_maps = pg.colormap.listMaps('matplotlib')
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.lower() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name, source='matplotlib', skipCache=True)
#     #     if cm is not None:
#     #         add_bar(lw, map_name, cm)

#     # add_heading(lw, 'ColorCET import')
#     # list_of_maps = pg.colormap.listMaps('colorcet')
#     # list_of_maps = sorted( list_of_maps, key=lambda x: x.lower() )
#     # for map_name in list_of_maps:
#     #     cm = pg.colormap.get(map_name, source='colorcet', skipCache=True)
#     #     if cm is not None:
#     #         add_bar(lw, map_name, cm)

#     # ui.graphics_layout.setFixedHeight(num_bars * (bar_width+5) )
#     # return ui, add_heading, add_bar
#     return ui


