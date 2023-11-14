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
    RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Dict = field()
    LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Dict = field()
    
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False)


    @classmethod
    def init_rank_order_debugger(cls, global_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict):
        """ NOT YET FINISHED!
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()
          
        """
        _obj = cls(global_spikes_df=global_spikes_df, active_epochs_df=active_epochs_df.copy(), track_templates=track_templates,
             RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

        name:str = 'RankOrderDebugger'
        LR_display_outputs, RL_display_outputs = _debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates)
        LR_app, LR_win, LR_plots, LR_plots_data, LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs = LR_display_outputs
        RL_app, RL_win, RL_plots, RL_plots_data, RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs = RL_display_outputs

        _obj.plots = RenderPlots(name=name, LR_app=LR_app, LR_win=LR_win, LR_plots=LR_plots, RL_app=RL_app, RL_win=RL_win, RL_plots=RL_plots, )
        _obj.plots_data = RenderPlotsData(name=name, LR_plots_data=LR_plots_data, LR_on_update_active_epoch=LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs=LR_on_update_active_scatterplot_kwargs,
                                           RL_plots_data=RL_plots_data, RL_on_update_active_epoch=RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs=RL_on_update_active_scatterplot_kwargs)
        
        ## Build the selected spikes df:
        (_obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map), (_obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map) = build_selected_spikes_df(_obj.track_templates, _obj.active_epochs_df,
                                                                                                                                             _obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict,
                                                                                                                                            _obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict)

        ## Add the spikes
        add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'long_LR')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'short_LR')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'long_RL')
        add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'short_RL')

        return _obj



    def on_update_active_epoch(self, an_epoch_idx, an_epoch):
        """ captures: LR_on_update_active_epoch, RL_on_update_active_epoch """
        self.plots_data.LR_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
        self.plots_data.RL_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)


    def on_update_epoch_IDX(self, an_epoch_idx):
        """ captures on_update_active_epoch, active_epochs_df to extract the epoch time range and call `on_update_active_epoch` """
        # curr_epoch_spikes = spikes_df[(spikes_df.new_lap_IDX == an_epoch_idx)]
        curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.lap_id == (an_epoch_idx+1))]
        curr_epoch = list(curr_epoch_df.itertuples())[0]
        self.on_update_active_epoch(an_epoch_idx, curr_epoch)
                    


    

