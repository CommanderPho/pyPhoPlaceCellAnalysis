import operator
import weakref
from collections import OrderedDict

import copy
from typing import Optional, Dict, List, Tuple, Callable
from attrs import define, field, Factory
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum


__all__ = ['EpochsEditor']

@define(slots=False)
class EpochsEditor:
    """ EpochsEditor to allow user modification of epoch intervals using PyQtGraph and multiple custom linear rect items

    """
    pos_df: pd.DataFrame = field()
    curr_laps_df: pd.DataFrame = field()
    on_epoch_region_updated_callback: Optional[Callable] = field()

    plots: RenderPlots = field(init=False)
    changed_laps_df: pd.DataFrame = field(init=False)


    @classmethod
    def perform_plot_laps_diagnoser(cls, pos_df: pd.DataFrame, curr_laps_df: pd.DataFrame, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=None):
        """
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import perform_plot_laps_diagnoser

        """
        ## Begin Plotting
        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle('Lap Overlap Debugger')
        win.resize(2000, 400)
        win.ci.setBorder((50, 50, 100))

        position_plots = []
        sub_layouts = []
        viewboxes = []

        sub1 = win.addLayout()
        sub1.addLabel("Position (x)")
        # sub1.addLabel(axis='left', text='Recorded Position Data')
        # sub1.addLabel(axis='bottom', text='time')

        sub1.nextRow()
        v1 = sub1.addViewBox()
        ax_pos = pg.PlotDataItem(x=pos_df.t.to_numpy(), y=pos_df.x.to_numpy(), pen='white', name=f'pos_x')

        v1.setMouseEnabled(x=True, y=False)
        # v1.enableAutoRange(x=False, y=True)
        # v1.setXRange(300, 450)
        # v1.setAutoVisible(x=False, y=True)
        # v1.PanMode

        v1.addItem(ax_pos)

        # legend_size = (80,60) # fixed size legend
        legend_size = None # auto-sizing legend to contents
        legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
        legend.setParentItem(v1.graphicsItem())


        # ax_pos = win.plot(x=pos_df.t.to_numpy(), y=pos_df.x.to_numpy(), pen='white', name=f'pos_x') #  , symbol='o', symbolBrush=pg.intColor(0,6,maxValue=128), symbolBrush=pg.intColor(i,6,maxValue=128)
        legend.addItem(ax_pos, f'pos_x')
        sub_layouts.append(sub1)
        viewboxes.append(v1)
        position_plots.append(ax_pos)
        # if include_velocity:
        # 	ax_velocity = win.plot(x=pos_df.t.to_numpy(), y=pos_df['velocity_x_smooth'].to_numpy(), pen='white', name=f'velocity_x_smooth') #  symbolBrush=pg.intColor(i,6,maxValue=128)
        # 	legend.addItem(ax_velocity, f'velocity_x_smooth')
        # 	position_plots.append(ax_velocity)
            
        # win.graphicsItem().setLabel(axis='left', text='Recorded Position Data')
        # win.graphicsItem().setLabel(axis='bottom', text='time')

        lap_epoch_widgets = {}
        lap_epoch_labels = {}

        for a_lap in curr_laps_df.itertuples():
            epoch_linear_region = lap_epoch_widgets.get(a_lap.label, None)
            if epoch_linear_region is None:
                ## Create a new one:
                # add alpha
                epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(v1, t_start=a_lap.start, t_stop=a_lap.stop, epoch_label=a_lap.label, movable=True, **dict(pen=pg.mkPen(f'{a_lap.lap_color}d6', width=1.0), brush=pg.mkBrush(f"{a_lap.lap_color}42"), hoverBrush=pg.mkBrush(f"{a_lap.lap_color}a8"), hoverPen=pg.mkPen('#00ff00de', width=2.5)), custom_bound_data=a_lap.Index)
                
                lap_epoch_widgets[a_lap.label] = epoch_linear_region
                lap_epoch_labels[a_lap.label] = epoch_region_label
                if on_epoch_region_updated_callback is not None:
                    epoch_linear_region.sigRegionChangeFinished.connect(on_epoch_region_updated_callback)
                
        
        plots = RenderPlots('lap_debugger_plot', win=win,
            sub_layouts=sub_layouts, viewboxes=viewboxes, position_plots=position_plots,
            lap_epoch_widgets=lap_epoch_widgets, lap_epoch_labels=lap_epoch_labels,
            legend=legend)
        return plots

    @classmethod
    def init_laps_diagnoser(cls, pos_df: pd.DataFrame, curr_laps_df: pd.DataFrame, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=None):
        _obj = cls(pos_df=pos_df, curr_laps_df=curr_laps_df, on_epoch_region_updated_callback=on_epoch_region_updated_callback)
        _obj.changed_laps_df = _obj.curr_laps_df.iloc[:0,:].copy() # should be in attrs_post_init
        _obj.plots = cls.perform_plot_laps_diagnoser(pos_df, curr_laps_df, include_velocity=include_velocity, include_accel=include_accel, on_epoch_region_updated_callback=_obj.on_epoch_region_updated)
        return _obj


    # @QtCore.pyqtSlot(object)
    def on_epoch_region_updated(self, epoch_region_item):
        print(f'epoch_region_item: {epoch_region_item}, epoch_region_item.custom_bound_data: {epoch_region_item.custom_bound_data}')
        epoch_index = epoch_region_item.custom_bound_data
        
        prev_start, prev_stop = self.curr_laps_df.loc[epoch_index].start, self.curr_laps_df.loc[epoch_index].stop
        # curr_laps_df.loc[epoch_index]
        new_start, new_stop = epoch_region_item.getRegion()
        change_delta = ((prev_start- new_start), (prev_stop-new_stop))
        start_changed = np.logical_not(np.isclose(prev_start, new_start))
        stop_changed = np.logical_not(np.isclose(prev_stop, new_stop))
        either_changed = stop_changed or start_changed
        if either_changed:
            print(f'either_changed! change_delta: {change_delta}')
            self.changed_laps_df.loc[epoch_index, :] = self.curr_laps_df.loc[epoch_index] # copy the existing row
            self.changed_laps_df.loc[epoch_index, 'start'] =  new_start
            self.changed_laps_df.loc[epoch_index, 'stop'] =  new_stop
            self.changed_laps_df.loc[epoch_index, 'duration'] =  (new_stop - new_start)

        if self.on_epoch_region_updated_callback is not None:
            self.on_epoch_region_updated_callback(epoch_region_item) # pass through the change


# class EpochsEditorItem(pg.GraphicsObject):
#     ## public class
#     """**Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
#     A rectangular item with tick marks along its length that can (optionally) be moved by the user.

#     Usage:

#     # pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
#     position_obj = deepcopy(sess.position)
#     position_obj.compute_higher_order_derivatives()
#     pos_df = position_obj.compute_smoothed_position_info(N=20) ## Smooth the velocity curve to apply meaningful logic to it
#     pos_df = position_obj.to_dataframe()
#     # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
#     pos_df = pos_df.dropna(subset=['t', 'velocity_x_smooth', 'acceleration_x_smooth', 'velocity_y_smooth'])


#     curr_laps_df = sess.laps.to_dataframe()

#     """
    
#     sigTicksChanged = QtCore.Signal(object)
#     sigTicksChangeFinished = QtCore.Signal(object)
    
#     def __init__(self, pos_df: pd.DataFrame, curr_laps_df: pd.DataFrame, include_velocity=True, include_accel=True, allowAdd=True, allowRemove=True, **kargs):
#         """
#         ==============  =================================================================================
#         **Arguments:**
#         orientation     Set the orientation of the gradient. Options are: 'left', 'right'
#                         'top', and 'bottom'.
#         allowAdd        Specifies whether the user can add ticks.
#         allowRemove     Specifies whether the user can remove new ticks.
#         tickPen         Default is white. Specifies the color of the outline of the ticks.
#                         Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`
#         ==============  =================================================================================
#         """
#         ## public
#         pg.GraphicsObject.__init__(self)

#         # Add 'lap_color' column to curr_laps_df
#         curr_laps_df['lap_color'] = DisplayColorsEnum.Laps.even
#         curr_laps_df.loc[(curr_laps_df['lap_dir'] > 0), 'lap_color'] = DisplayColorsEnum.Laps.odd
#         curr_laps_df




#         self.pos_df = pos_df
#         self.curr_laps_df = curr_laps_df

#         self.tickSize = 15
#         self.ticks = {}
#         self.maxDim = 20
#         self.allowAdd = allowAdd
#         self.allowRemove = allowRemove
#         if 'tickPen' in kargs:
#             self.tickPen = fn.mkPen(kargs['tickPen'])
#         else:
#             self.tickPen = fn.mkPen('w')
            
#         self.orientations = {
#             'left': (90, 1, 1), 
#             'right': (90, 1, 1), 
#             'top': (0, 1, -1), 
#             'bottom': (0, 1, 1)
#         }
        
#         self.setOrientation(orientation)
#         #self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame | QtWidgets.QFrame.Shadow.Plain)
#         #self.setBackgroundRole(QtGui.QPalette.ColorRole.NoRole)
#         #self.setMouseTracking(True)
        
#     #def boundingRect(self):
#         #return self.mapRectFromParent(self.geometry()).normalized()
        
#     #def shape(self):  ## No idea why this is necessary, but rotated items do not receive clicks otherwise.
#         #p = QtGui.QPainterPath()
#         #p.addRect(self.boundingRect())
#         #return p
        

#     def initUI(self):
#         ## Begin Plotting
#         win = pg.GraphicsLayoutWidget(show=True)
#         win.setWindowTitle('Lap Overlap Debugger')
#         win.resize(2000, 400)
#         win.ci.setBorder((50, 50, 100))

#         position_plots = []

#         sub1 = win.addLayout()
#         sub1.addLabel("Position (x)")
#         # sub1.addLabel(axis='left', text='Recorded Position Data')
#         # sub1.addLabel(axis='bottom', text='time')

#         sub1.nextRow()
#         v1 = sub1.addViewBox()
#         ax_pos = pg.PlotDataItem(x=pos_df.t.to_numpy(), y=pos_df.x.to_numpy(), pen='white', name=f'pos_x')

#         v1.setMouseEnabled(x=True, y=False)
#         # v1.enableAutoRange(x=False, y=True)
#         # v1.setXRange(300, 450)
#         # v1.setAutoVisible(x=False, y=True)
#         # v1.PanMode

#         v1.addItem(ax_pos)

#         # legend_size = (80,60) # fixed size legend
#         legend_size = None # auto-sizing legend to contents
#         legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
#         legend.setParentItem(v1.graphicsItem())


#         # ax_pos = win.plot(x=pos_df.t.to_numpy(), y=pos_df.x.to_numpy(), pen='white', name=f'pos_x') #  , symbol='o', symbolBrush=pg.intColor(0,6,maxValue=128), symbolBrush=pg.intColor(i,6,maxValue=128)
#         legend.addItem(ax_pos, f'pos_x')
#         position_plots.append(ax_pos)
#         # if include_velocity:
#         # 	ax_velocity = win.plot(x=pos_df.t.to_numpy(), y=pos_df['velocity_x_smooth'].to_numpy(), pen='white', name=f'velocity_x_smooth') #  symbolBrush=pg.intColor(i,6,maxValue=128)
#         # 	legend.addItem(ax_velocity, f'velocity_x_smooth')
#         # 	position_plots.append(ax_velocity)
            
#         # win.graphicsItem().setLabel(axis='left', text='Recorded Position Data')
#         # win.graphicsItem().setLabel(axis='bottom', text='time')


#         # sub1.graphicsItem().setLabel(axis='left', text='Recorded Position Data')
#         # sub1.graphicsItem().setLabel(axis='bottom', text='time')


#         lap_epoch_widgets = {}
#         lap_epoch_labels = {}

#         for a_lap in curr_laps_df.itertuples():
#             epoch_linear_region = lap_epoch_widgets.get(a_lap.label, None)
#             if epoch_linear_region is None:
#                 ## Create a new one:
#                 # add alpha
#                 epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(v1, t_start=a_lap.start, t_stop=a_lap.stop, epoch_label=a_lap.label, movable=True, **dict(pen=pg.mkPen(f'{a_lap.lap_color}d6', width=1.0), brush=pg.mkBrush(f"{a_lap.lap_color}42"), hoverBrush=pg.mkBrush(f"{a_lap.lap_color}a8"), hoverPen=pg.mkPen('#00ff00de', width=2.5)), custom_bound_data=a_lap.Index)
                
#                 lap_epoch_widgets[a_lap.label] = epoch_linear_region
#                 lap_epoch_labels[a_lap.label] = epoch_region_label
#                 epoch_linear_region.sigRegionChangeFinished.connect(self.on_epoch_region_updated)
                

#             epoch_linear_region


#     @QtCore.pyqtSlot(object)
#     def on_epoch_region_updated(self, epoch_region_item):
        
#         print(f'epoch_region_item: {epoch_region_item}, epoch_region_item.custom_bound_data: {epoch_region_item.custom_bound_data}')
#         epoch_index = epoch_region_item.custom_bound_data
        
#         prev_start, prev_stop = curr_laps_df.loc[epoch_index].start, curr_laps_df.loc[epoch_index].stop
#         # curr_laps_df.loc[epoch_index]
#         new_start, new_stop = epoch_region_item.getRegion()
#         change_delta = ((prev_start- new_start), (prev_stop-new_stop))
#         start_changed = np.logical_not(np.isclose(prev_start, new_start))
#         stop_changed = np.logical_not(np.isclose(prev_stop, new_stop))
#         either_changed = stop_changed or start_changed
#         if either_changed:
#             print(f'either_changed! change_delta: {change_delta}')


#     # @QtCore.pyqtSlot()
#     # def _Render2DScrollWindowPlot_on_linear_region_item_update(self) -> None:
#     #     """self when the region moves.zoom_Change plotter area"""
#     #     # self.ui.scroll_window_region.setZValue(10) # bring to the front
#     #     min_x, max_x = self.ui.scroll_window_region.getRegion() # get the current region
#     #     self.window_scrolled.emit(min_x, max_x) # emit this mixin's own window_scrolled function
        
        

#     # @QtCore.pyqtSlot()
#     # def Render2DScrollWindowPlot_on_init():
#     #     """ perform any parameters setting/checking during init """
#     #     pass

#     # @QtCore.pyqtSlot()
#     # def Render2DScrollWindowPlot_on_setup():
#     #     """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
#     #     # Connect the signals for the zoom region and the LinearRegionItem
#     #     # self.ui.scroll_window_region.sigRegionChanged.connect(self.update_zoom_plotter)
#     #     pass

#     # @QtCore.pyqtSlot()
#     # def Render2DScrollWindowPlot_on_destroy():
#     #     """ perfrom teardown/destruction of anything that needs to be manually removed or released """
#     #     pass


#     # def paint(self, p, opt, widget):
#     #     #p.setPen(fn.mkPen('g', width=3))
#     #     #p.drawRect(self.boundingRect())
#     #     return
        
#     # def keyPressEvent(self, ev):
#     #     ev.ignore()

    

