import operator
from collections import OrderedDict

from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

__all__ = ['PhoBaseContainerTool']


@metadata_attributes(short_name=None, tags=['gui'], input_requires=[], output_provides=[], uses=['RenderPlots', 'RenderPlotsData', 'PhoUIContainer'], used_by=[], creation_date='2023-11-17 19:59', related_items=[])
@define(slots=False, eq=False)
class PhoBaseContainerTool:
    """ a tool in a container:
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import PhoBaseContainerTool
    
    
    """
    plots: RenderPlots = field(default=Factory(PyqtgraphRenderPlots, 'plotter'))
    plots_data: RenderPlotsData = field(default=Factory(RenderPlotsData, 'plotter'), repr=False)
    ui: PhoUIContainer = field(default=Factory(PhoUIContainer, 'plotter'), repr=False)
    params: VisualizationParameters = field(default=Factory(VisualizationParameters, 'plotter'), repr=keys_only_repr)




@metadata_attributes(short_name=None, tags=['unused', 'container', 'pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-17 20:06', related_items=[])
@define(slots=False, eq=False)
class GenericPyQtGraphContainer:
    """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer

    """
    name: str = field(default='plot')
    plots: PyqtgraphRenderPlots = field(default=Factory(PyqtgraphRenderPlots, 'plotter'))
    plot_data: RenderPlotsData = field(default=Factory(RenderPlotsData, 'plotter'))
    ui: PhoUIContainer = field(default=Factory(PhoUIContainer, 'plotter'))
    params: VisualizationParameters = field(default=Factory(VisualizationParameters, 'plotter'), repr=keys_only_repr)




# @metadata_attributes(short_name=None, tags=['unused', 'container', 'pyqtgraph', 'interactive', 'scatterplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-17 20:06', related_items=[])
# @define(slots=False)
# class GenericPyQtGraphScatterClicker:
#     """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

#     from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import GenericPyQtGraphScatterClicker

#     """
#     lastClickedDict: Dict = field(default=Factory(dict))


#     def on_scatter_plot_clicked(self, plot, evt):
#         """ captures `lastClicked` 
#         plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
#         clicked points <MouseClickEvent (78.6115,-2.04825) button=1>

#         """
#         # global lastClicked  # Declare lastClicked as a global variable
#         if plot not in self.lastClickedDict:
#             self.lastClickedDict[plot] = None

#         # for p in self.lastClicked:
#         # 	p.resetPen()
#         # print(f'plot: {plot}') # plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
#         # print(f'\tevt: {evt}')	
#         # print("clicked points", evt.pos()) # clicked points <MouseClickEvent (48.2713,1.32425) button=1>
#         # print(f'args: {args}')
#         pt_x, pt_y = evt.pos()
#         idx_x = int(round(pt_x))
#         print(f'\tidx_x: {idx_x}')
#         # pts = plot.pointsAt(evt.pos())
#         # print(f'pts: {pts}')
#         # for p in points:
#         # 	p.setPen(clickedPen)
#         # self.lastClicked = idx_x
#         self.lastClickedDict[plot] = idx_x




# lastClicked = []
# def _test_scatter_plot_clicked(plot, evt):
# 	""" captures `lastClicked` 
# 	plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
# 	clicked points <MouseClickEvent (78.6115,-2.04825) button=1>

# 	"""
# 	global lastClicked  # Declare lastClicked as a global variable
# 	# for p in lastClicked:
# 	# 	p.resetPen()
# 	# print(f'plot: {plot}') # plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
# 	# print(f'\tevt: {evt}')	
# 	# print("clicked points", evt.pos()) # clicked points <MouseClickEvent (48.2713,1.32425) button=1>
# 	# print(f'args: {args}')
# 	pt_x, pt_y = evt.pos()
# 	idx_x = int(round(pt_x))
# 	print(f'\tidx_x: {idx_x}')
# 	# pts = plot.pointsAt(evt.pos())
# 	# print(f'pts: {pts}')
# 	# for p in points:
# 	# 	p.setPen(clickedPen)
# 	lastClicked = idx_x



