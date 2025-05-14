from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.PhoMainAppWindowBase import PhoMainAppWindowBase # for pyqtplot_plot_image


# Function to set small titles with minimal padding
@function_attributes(short_name=None, tags=['title', 'IMPORTANT', 'pyqtgraph', 'title', 'spacing'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-05 16:43', related_items=[])
def set_small_title(plotItem: pg.PlotItem, title_row_fixed_height: int = 10):
    """ Adjusts the size and padding of the plotItem's title. Removes the excessive margins that are present by default.
    title_row_fixed_height: int = 10 - the height of the title row in px, (default 30)
    
    Usage:
        
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import set_small_title

        title_row_fixed_height: int = 10 #the height of the title row in px, (default 30)
        for newPlotItem in _a_trial_by_trial_window.plots['plot_array']:
            set_small_title(newPlotItem, title_row_fixed_height)


    """
    plotItem.layout.setRowMinimumHeight(0, title_row_fixed_height)
    plotItem.layout.setRowMaximumHeight(0, title_row_fixed_height)
    plotItem.layout.setRowFixedHeight(0, title_row_fixed_height) # 0 is the fixed row index in the plotItem's layout
    plotItem.layout.setRowPreferredHeight(0, title_row_fixed_height)






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
    # cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
    cmap = pg.colormap.get('Oranges','matplotlib') # prepare a linear color map
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
 

@function_attributes(short_name=None, tags=['images', 'pyqtgraph', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-21 00:00', related_items=['transition_matrix'])
def visualize_multiple_image_items(images: list, threshold=1e-3) -> None:
    """ Sample multiple pg.ImageItems overlayed on one another

    # Example usage:
    image1 = np.random.rand(100, 100) * 100  # Example image 1
    image2 = np.random.rand(100, 100) * 100  # Example image 2
    image3 = np.random.rand(100, 100) * 100  # Example image 3

    image1
    # Define the threshold

    _out = visualize_multiple_image_items([image1, image2, image3], threshold=50)

    
    History:
        Plotting Generated Transition Matrix Sequences  
    """
    app = pg.mkQApp('visualize_multiple_image_items')  # Initialize the Qt application
    win = pg.GraphicsLayoutWidget(show=True)
    view = win.addViewBox()
    view.setAspectLocked(True)

    for img in images:
        if threshold is not None:
            # Create a masked array, masking values below the threshold
            img = np.ma.masked_less(img, threshold)

        image_item = pg.ImageItem(img)
        view.addItem(image_item)

    # QtGui.QApplication.instance().exec_()
    return app, win, view




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


@function_attributes(short_name=None, tags=['pyqtgraph', 'build'], input_requires=[], output_provides=[], uses=['pg.GraphicsLayoutWidget'], used_by=['BasicBinnedImageRenderingWindow'], creation_date='2023-04-18 00:00', related_items=[])
def _perform_build_root_graphics_layout_widget_ui(ui:PhoUIContainer, is_scrollable: bool = True) -> PhoUIContainer:
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


def build_root_graphics_layout_widget_ui(name, window_title=None, ui=None) -> PhoUIContainer:
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

def build_scrollable_graphics_layout_widget_ui(name, window_title=None, ui=None) -> PhoUIContainer:
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

def build_scrollable_graphics_layout_widget_with_nested_viewbox_ui(name, window_title=None, ui=None) -> PhoUIContainer:
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
# Plotting Helpers                                                                                                     #
# ==================================================================================================================== #
@function_attributes(short_name='build_pyqtgraph_epoch_indicator_regions', tags=['pyqtgraph','epoch','render','plot','CustomLinearRegionItem'], input_requires=[], output_provides=[],
    uses=['pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem', 'pg.InfLineLabel'], used_by=[], creation_date='2023-04-18 08:37')
def build_pyqtgraph_epoch_indicator_regions(win: pg.PlotWidget, t_start:float, t_stop:float, epoch_label:str = 'short', movable=False, removable=True, **kwargs):
    """ 2023-04-17 - Build a CustomLinearRegionItem that sits behind the data in a pyqtgraph PlotItem that indicates the timerange of the current epoch. 

    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
    
        epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(win, t_start=curr_active_pipeline.filtered_epochs[long_epoch_name].t_start, t_stop=curr_active_pipeline.filtered_epochs[long_epoch_name].t_stop, epoch_label='long', **dict(pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00')))
        epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(win, t_start=curr_active_pipeline.filtered_epochs[short_epoch_name].t_start, t_stop=curr_active_pipeline.filtered_epochs[short_epoch_name].t_stop, epoch_label='short', **dict(pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00')))
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem # used in `plot_kourosh_activity_style_figure`

    # Add the linear region overlay:
    epoch_linear_region:CustomLinearRegionItem = CustomLinearRegionItem(**(dict(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'))|kwargs), movable=movable, removable=removable) #, clipItem=plots['difference']  bound the LinearRegionItem to the plotted data
    epoch_linear_region.setObjectName(f'epoch[{epoch_label}]')
    epoch_linear_region.setZValue(-3) # put it in the back
    epoch_region_label:pg.InfLineLabel = pg.InfLineLabel(epoch_linear_region.lines[0], f"{epoch_label}", position=0.95, rotateAxis=(1,0), anchor=(1, 1)) # add the label for the short epoch
    # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
    win.addItem(epoch_linear_region, ignoreBounds=True)
    # Set the position:
    epoch_linear_region.setRegion([t_start, t_stop]) # adjust scroll control
    return epoch_linear_region, epoch_region_label



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



# ==================================================================================================================== #
# 2023-12-19 PyQtGraphCrosshairs                                                                                       #
# ==================================================================================================================== #

"""
Demonstrates some customized mouse interaction by drawing a crosshair that follows 
the mouse.
"""

from attrs import define, field
import pyphoplacecellanalysis.External.pyqtgraph as pg

@define(slots=False, repr=False)
class PyQtGraphCrosshairs:
    """ a class wrapper for the simple hover crosshairs shown in the pyqtgraph examples
    
    """
    vLine: pg.InfiniteLine = field()
    hLine: pg.InfiniteLine = field()
    proxy: pg.SignalProxy = field(init=False) 
    p1: pg.PlotItem = field(init=False)
    label: Optional[pg.LabelItem] = field(init=False)
        
    @classmethod
    def init_from_plot_item(cls, p1, a_label):
        _obj = cls(vLine=pg.InfiniteLine(angle=90, movable=False), 
             hLine=pg.InfiniteLine(angle=0, movable=False))
        _obj.p1 = p1
        _obj.label = a_label
        # _obj.vLine = pg.InfiniteLine(angle=90, movable=False)
        # _obj.hLine = pg.InfiniteLine(angle=0, movable=False)
        p1.addItem(_obj.vLine, ignoreBounds=True)
        p1.addItem(_obj.hLine, ignoreBounds=True)
        _obj.proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=_obj.mouseMoved)
        return _obj
  

    def mouseMoved(self, evt):
        """ captures `label` """
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        vb = self.p1.vb
        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            # if index > 0 and index < len(data1):
            #     print(f"<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
            #     if self.label is not None:
            #         self.label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())



# #generate layout
# app = pg.mkQApp("Crosshair Example")
# win = pg.GraphicsLayoutWidget(show=True)
# win.setWindowTitle('pyqtgraph example: crosshair')
# label = pg.LabelItem(justify='right')
# win.addItem(label)
# p1 = win.addPlot(row=1, col=0)
# # customize the averaged curve that can be activated from the context menu:
# p1.avgPen = pg.mkPen('#FFFFFF')
# p1.avgShadowPen = pg.mkPen('#8080DD', width=10)

# p2 = win.addPlot(row=2, col=0)

# region = pg.LinearRegionItem()
# region.setZValue(10)
# # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
# # item when doing auto-range calculations.
# p2.addItem(region, ignoreBounds=True)

# #pg.dbg()
# p1.setAutoVisible(y=True)

# #create numpy arrays
# #make the numbers large to show that the range shows data from 10000 to all the way 0
# data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
# data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)

# p1.plot(data1, pen="r")
# p1.plot(data2, pen="g")

# p2d = p2.plot(data1, pen="w")
# # bound the LinearRegionItem to the plotted data
# region.setClipItem(p2d)

# def update():
#     region.setZValue(10)
#     minX, maxX = region.getRegion()
#     p1.setXRange(minX, maxX, padding=0)    

# region.sigRegionChanged.connect(update)

# def updateRegion(window, viewRange):
#     rgn = viewRange[0]
#     region.setRegion(rgn)

# p1.sigRangeChanged.connect(updateRegion)

# region.setRegion([1000, 2000])

# #cross hair
# vLine = pg.InfiniteLine(angle=90, movable=False)
# hLine = pg.InfiniteLine(angle=0, movable=False)
# p1.addItem(vLine, ignoreBounds=True)
# p1.addItem(hLine, ignoreBounds=True)
# vb = p1.vb

# a_crosshairs = PyQtGraphCrosshairs.init_from_plot_item(p1=p1, a_label=label)

# #p1.scene().sigMouseMoved.connect(mouseMoved)

