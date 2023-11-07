import operator
import weakref
from collections import OrderedDict

from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable
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

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum


from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import _debug_plot_directional_template_rasters
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import build_selected_spikes_df
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import add_selected_spikes_df_points_to_scatter_plot

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import TrackTemplates, RankOrderAnalyses, ShuffleHelper, Zscorer


__all__ = ['RankOrderDebugger']


@define(slots=False)
class GenericPyQtGraphContainer:
    """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderDebugger import GenericPyQtGraphContainer

    """
    name: str = field(default='plot')
    params: VisualizationParameters = field(default=Factory(VisualizationParameters, 'plotter'))
    ui: PhoUIContainer = field(default=Factory(PhoUIContainer, 'plotter'))
    plots: PyqtgraphRenderPlots = field(default=Factory(PyqtgraphRenderPlots, 'plotter'))
    plot_data: RenderPlotsData = field(default=Factory(RenderPlotsData, 'plotter'))




@define(slots=False)
class GenericPyQtGraphScatterClicker:
    """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderDebugger import GenericPyQtGraphScatterClicker

    """
    lastClickedDict: Dict = field(default=Factory(dict))


    def on_scatter_plot_clicked(self, plot, evt):
        """ captures `lastClicked` 
        plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
        clicked points <MouseClickEvent (78.6115,-2.04825) button=1>

        """
        # global lastClicked  # Declare lastClicked as a global variable
        if plot not in self.lastClickedDict:
            self.lastClickedDict[plot] = None

        # for p in self.lastClicked:
        # 	p.resetPen()
        # print(f'plot: {plot}') # plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
        # print(f'\tevt: {evt}')	
        # print("clicked points", evt.pos()) # clicked points <MouseClickEvent (48.2713,1.32425) button=1>
        # print(f'args: {args}')
        pt_x, pt_y = evt.pos()
        idx_x = int(round(pt_x))
        print(f'\tidx_x: {idx_x}')
        # pts = plot.pointsAt(evt.pos())
        # print(f'pts: {pts}')
        # for p in points:
        # 	p.setPen(clickedPen)
        # self.lastClicked = idx_x
        self.lastClickedDict[plot] = idx_x




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







@define(slots=False)
class RankOrderDebugger:
    """ RankOrderDebugger displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)

    """
    global_spikes_df: pd.DataFrame = field()
    active_epochs_df: pd.DataFrame = field()
    track_templates: TrackTemplates = field()
    even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict = field()
    odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict = field()
    
    plots: RenderPlots = field(init=False)
    

    @classmethod
    def init_rank_order_debugger(cls, global_spikes_df: pd.DataFrame, global_epochs_df: pd.DataFrame, track_templates: TrackTemplates, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict):
        """ NOT YET FINISHED!
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()
          
        """
        _obj = cls(global_spikes_df=global_spikes_df, active_epochs_df=global_epochs_df.copy(), track_templates=track_templates,
             even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


        odd_display_outputs, even_display_outputs = _debug_plot_directional_template_rasters(global_spikes_df, self.active_epochs_df, self.track_templates)
        odd_app, odd_win, odd_plots, odd_plots_data, odd_on_update_active_epoch, odd_on_update_active_scatterplot_kwargs = odd_display_outputs
        even_app, even_win, even_plots, even_plots_data, even_on_update_active_epoch, even_on_update_active_scatterplot_kwargs = even_display_outputs




        

        ## Build the selected spikes df:

        (even_selected_spike_df, even_neuron_id_to_new_IDX_map), (odd_selected_spike_df, odd_neuron_id_to_new_IDX_map) = build_selected_spikes_df(self.track_templates, self.active_epochs_df, self.even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, self.odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

        ## Add the spikes
        add_selected_spikes_df_points_to_scatter_plot(plots_data=odd_plots_data, plots=odd_plots, selected_spikes_df=deepcopy(odd_selected_spike_df), _active_plot_identifier = 'long_odd')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=odd_plots_data, plots=odd_plots, selected_spikes_df=deepcopy(odd_selected_spike_df), _active_plot_identifier = 'short_odd')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=even_plots_data, plots=even_plots, selected_spikes_df=deepcopy(even_selected_spike_df), _active_plot_identifier = 'long_even')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=even_plots_data, plots=even_plots, selected_spikes_df=deepcopy(even_selected_spike_df), _active_plot_identifier = 'short_even')





        return _obj



    def on_update_active_epoch(self, an_epoch_idx, an_epoch):
        """ captures: odd_on_update_active_epoch, even_on_update_active_epoch """
        odd_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
        even_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)


    def on_update_epoch_IDX(self, an_epoch_idx):
        """ captures on_update_active_epoch, active_epochs_df to extract the epoch time range and call `on_update_active_epoch` """
        # curr_epoch_spikes = spikes_df[(spikes_df.new_lap_IDX == an_epoch_idx)]
        curr_epoch_df = active_epochs_df[(active_epochs_df.lap_id == (an_epoch_idx+1))]
        curr_epoch = list(curr_epoch_df.itertuples())[0]

        self.on_update_active_epoch(an_epoch_idx, curr_epoch)
                    


    

