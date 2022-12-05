# LocalMenus_AddRenderable.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\LocalMenus_AddRenderable.ui automatically by PhoPyQtClassGenerator VSCode Extension

from functools import partial
from benedict import benedict

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.Uic_AUTOGEN_LocalMenus_AddRenderable import Ui_LocalMenus_AddRenderable

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Replays_2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs, SessionEpochs2DRenderTimeEpochs, PBE_2DRenderTimeEpochs, Laps2DRenderTimeEpochs # Time Intervals/Epochs
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves, PositionRenderTimeCurves, VelocityRenderTimeCurves, MUA_RenderTimeCurves ## Time Curves

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import AddNewDecodedPosition_MatplotlibPlotCommand ## MatplotlibSubplots

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus import LocalMenus_AddRenderable

class LocalMenus_AddRenderable(QtWidgets.QMainWindow):
    """ A context menu that adds renderables such as interval rectangles and plots to SpikeRaster2D
    
    
    menuCreate_Paired_Widget
        actionTimeSynchronizedOccupancyPlotter
        actionTimeSynchronizedPlacefieldsPlotter


    menuAdd_Matplotlib_Plot
        actionAddMatplotlibPlot_DecodedPosition
        actionClear_all_Matplotlib_Plots
    
    
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        # self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/GlobalApplicationMenus/LocalMenus_AddRenderable.ui", self) # Load the .ui file
        # AUTOGEN version:
        self.programmatic_actions_dict = None # PhoUIContainer.init_from_dict({})
        self.ui = Ui_LocalMenus_AddRenderable()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.initUI()
        # Do not call show, as this is just to add menus, and should be destroyed when done
        # self.show() # Show the GUI

    def initUI(self):
        pass


    @classmethod
    def _build_renderable_menu(cls, destination_plot, curr_active_pipeline, active_config_name):
        """ Builds the callbacks needed and connects them to the QActions and QMenus for the specific destination_plot to be used as context menus. 
        sess: Session
        destination_plot: e.g. active_2d_plot, active_3d_plot
        """
        # Extract sess from the more general curr_active_pipeline:
        computation_result = curr_active_pipeline.computation_results[active_config_name]
        sess = computation_result.sess
        
        
        
        widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
        widget.programmatic_actions_dict = benedict() # PhoUIContainer.init_from_dict({})
        renderable_menu = widget.ui.menuAdd_Renderable
        
        ## Time Intervals/Epochs:
        submenu_addTimeIntervals = [widget.ui.actionAddTimeIntervals_Laps,
                                    widget.ui.actionAddTimeIntervals_PBEs,
                                    widget.ui.actionAddTimeIntervals_Session_Epochs,
                                    widget.ui.actionAddTimeIntervals_Ripples,
                                    widget.ui.actionAddTimeIntervals_Replays,
                                    widget.ui.actionAddTimeIntervals_Custom]
        submenu_addTimeIntervalCallbacks = [lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: PBE_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.pbe, destination_plot=destination_plot),
                                            lambda evt=None: SessionEpochs2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.epochs, destination_plot=destination_plot),
                                            lambda evt=None: Ripples_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.ripple, destination_plot=destination_plot),
                                            lambda evt=None: Replays_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.replay, destination_plot=destination_plot),
                                            lambda evt=None: print(f'actionAddTimeIntervals_Custom not yet supported')]
        
        submenu_addTimeIntervals_Connections = []
        for an_action, a_callback in zip(submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeIntervals_Connections.append(_curr_conn)
            extracted_menu_path = PhoMenuHelper.parse_QAction_for_menu_path(an_action)
            widget.programmatic_actions_dict['.'.join(extracted_menu_path)] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed

        # Set enabled state
        widget.ui.actionAddTimeIntervals_PBEs.setEnabled(sess.pbe is not None)
        widget.ui.actionAddTimeIntervals_Laps.setEnabled(sess.laps is not None)
        widget.ui.actionAddTimeIntervals_Ripples.setEnabled(sess.ripple is not None)
        widget.ui.actionAddTimeIntervals_Replays.setEnabled(sess.has_replays)
        
        ## Time Curves:
        submenu_addTimeCurves = [widget.ui.actionAddTimeCurves_Position, widget.ui.actionAddTimeCurves_Velocity, widget.ui.actionAddTimeCurves_Random, widget.ui.actionAddTimeCurves_Custom]
        submenu_addTimeCurvesCallbacks = [lambda evt=None: PositionRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: VelocityRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: MUA_RenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            # lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: print(f'actionAddTimeCurves_Custom not yet supported')]
        submenu_addTimeCurves_Connections = []
        for an_action, a_callback in zip(submenu_addTimeCurves, submenu_addTimeCurvesCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeCurves_Connections.append(_curr_conn)
            extracted_menu_path = PhoMenuHelper.parse_QAction_for_menu_path(an_action)
            widget.programmatic_actions_dict['.'.join(extracted_menu_path)] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed
        
        ## Matplotlib Plots:
        # self.ui.menuAddRenderable_Matplotlib_Plot        
        submenu_addMatplotlibPlot = [widget.ui.actionAddMatplotlibPlot_DecodedPosition, widget.ui.actionAddMatplotlibPlot_Custom]
        submenu_addMatplotlibPlotCallbacks = [lambda evt=None: AddNewDecodedPosition_MatplotlibPlotCommand(destination_plot, curr_active_pipeline, active_config_name), # DecodedPositionMatplotlibSubplotRenderer.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot), 
                                            lambda evt=None: print(f'actionaddMatplotlibPlot_Custom not yet supported')]
        submenu_addMatplotlibPlot_Connections = []
        for an_action, a_callback in zip(submenu_addMatplotlibPlot, submenu_addMatplotlibPlotCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addMatplotlibPlot_Connections.append(_curr_conn)
            extracted_menu_path = PhoMenuHelper.parse_QAction_for_menu_path(an_action)
            widget.programmatic_actions_dict['.'.join(extracted_menu_path)] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed
        
        # Connect Clear actions:
        clear_actions = [widget.ui.actionClear_all_Time_Curves, widget.ui.actionClear_all_Time_Intervals, widget.ui.actionClear_all_Matplotlib_Plots, widget.ui.actionClear_all_Renderables]

        widget.ui.actionClear_all_Time_Curves.triggered.connect(destination_plot.clear_all_3D_time_curves)
        widget.ui.actionClear_all_Time_Intervals.triggered.connect(destination_plot.clear_all_rendered_intervals)
        widget.ui.actionClear_all_Matplotlib_Plots.triggered.connect(destination_plot.clear_all_matplotlib_plots)
        
        def _clear_all_both():
            destination_plot.clear_all_3D_time_curves()
            destination_plot.clear_all_rendered_intervals()
            destination_plot.clear_all_matplotlib_plots()
            
        widget.ui.actionClear_all_Renderables.triggered.connect(_clear_all_both)

        for an_action in clear_actions:
            widget.programmatic_actions_dict['.'.join(PhoMenuHelper.parse_QAction_for_menu_path(an_action))] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed
        
        return widget, renderable_menu, (submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks, submenu_addTimeIntervals_Connections), (submenu_addTimeCurves, submenu_addTimeCurvesCallbacks, submenu_addTimeCurves_Connections), (submenu_addMatplotlibPlot, submenu_addMatplotlibPlotCallbacks, submenu_addMatplotlibPlot_Connections)


    @classmethod
    def add_renderable_context_menu(cls, active_2d_plot, curr_active_pipeline, active_config_name):
        """ 
        Usage:
            active_2d_plot = spike_raster_window.spike_raster_plt_2d 
            menuAdd_Renderable = LocalMenus_AddRenderable.add_renderable_context_menu(active_2d_plot, sess)
            
        """
        def _subFn_append_custom_menu_to_context_menu(parent_widget, additional_menu, debug_print=False):
            """ Adds the custom menu, such as one loaded from a .ui file, to the end of the context menu
            parent_widget: PlotItem
            additional_menu: QMenu
            
            Example:
                from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
                from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.LocalMenus_AddRenderable import LocalMenus_AddRenderable
                widget = LocalMenus_AddRenderable()
                append_custom_menu_to_context_menu(main_plot_widget, widget.ui.menuAdd_Renderable)
                append_custom_menu_to_context_menu(background_static_scroll_plot_widget, widget.ui.menuAdd_Renderable)
            """
            
            plot_options_context_menu = parent_widget.getContextMenus(None) # This gets the "Plot Options" menu
            # top_level_parent_context_menu = parent_context_menus.parent()
            top_level_parent_context_menu = parent_widget.vb.menu # ViewBoxMenu
         
            active_parent_menu = top_level_parent_context_menu
            active_parent_menu.addSeparator()
            active_parent_menu.addMenu(additional_menu)
            if debug_print:
                print(f'parent_context_menus.actions: {[an_action.text() for an_action in active_parent_menu.actions()]}') # parent_context_menus.actions: ['Transforms', 'Downsample', 'Average', 'Alpha', 'Grid', 'Points']
                
         ## Build `partial` versions of the functions specific to each raster plotter that can be called with no arguments (capturing the destination plotter and the session
        # build_renderable_menu_to_Spike2DRaster = partial(cls.build_renderable_menu, active_2d_plot, sess) # destination_plot
        # active_2d_plot_renderable_menus = cls._build_renderable_menu(active_2d_plot, sess)
        
        active_2d_plot_renderable_menus = cls._build_renderable_menu(active_2d_plot, curr_active_pipeline, active_config_name)
        widget_2d_menu = active_2d_plot_renderable_menus[0]
        menuAdd_Renderable = widget_2d_menu.ui.menuAdd_Renderable
        programmatic_actions_dict = widget_2d_menu.programmatic_actions_dict

        ## Specific to SpikeRaster2D:        
        ## Add the custom menu to the context menus of the plots in SpikeRaster2D:        
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
        _subFn_append_custom_menu_to_context_menu(main_plot_widget, menuAdd_Renderable)
        _subFn_append_custom_menu_to_context_menu(background_static_scroll_plot_widget, menuAdd_Renderable)
        
        # Add the reference to the context menus to owner, so it isn't released:
        ## TODO: currently replaces the dict entry, which we might want to use for other menus
        active_2d_plot.ui.menus = PhoUIContainer.init_from_dict({'custom_context_menus': PhoUIContainer.init_from_dict({'add_renderables': active_2d_plot_renderable_menus})})

        # # Build final programmatic dict from nested PhoUIContainers:
        # out_final = PhoUIContainer.init_from_dict({})
        # for k, v in programmatic_actions_dict.items_sorted_by_keys(reverse=False):
        #     out_final[k] = PhoUIContainer.init_from_dict(v)
        #     # print(k, v)
        # widget_2d_menu.programmatic_actions_dict = out_final


        return menuAdd_Renderable # try returning just the menu and not the stupid references to everything # Works when we hold a reference


    @classmethod
    def perform_build_manual_paired_Widget_menu(cls, action_parent, menu_parent):
        """ does everything locally, does not apply it to any .ui.* paths """
        actions_dict = PhoUIContainer.init_from_dict({})
                
        # Actions:
        actions_dict.actionCreate_paired_time_synchronized_widget = QtWidgets.QAction(action_parent)
        actions_dict.actionCreate_paired_time_synchronized_widget.setObjectName("actionCreate_paired_time_synchronized_widget")
        actions_dict.actionTimeSynchronizedOccupancyPlotter = QtWidgets.QAction(action_parent)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/bar-chart_2@1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        actions_dict.actionTimeSynchronizedOccupancyPlotter.setIcon(icon2)
        actions_dict.actionTimeSynchronizedOccupancyPlotter.setText("TimeSynchronizedOccupancyPlotter")
        actions_dict.actionTimeSynchronizedOccupancyPlotter.setObjectName("actionTimeSynchronizedOccupancyPlotter")
        
        actions_dict.actionTimeSynchronizedPlacefieldsPlotter = QtWidgets.QAction(action_parent)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/wifi-channel_2@1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        actions_dict.actionTimeSynchronizedPlacefieldsPlotter.setIcon(icon3)
        actions_dict.actionTimeSynchronizedPlacefieldsPlotter.setText("TimeSynchronizedPlacefieldsPlotter")
        actions_dict.actionTimeSynchronizedPlacefieldsPlotter.setObjectName("actionTimeSynchronizedPlacefieldsPlotter")
        
        actions_dict.actionCombineTimeSynchronizedPlotterWindow = QtWidgets.QAction(action_parent)
        actions_dict.actionCombineTimeSynchronizedPlotterWindow.setText("Combined Time Syncrhonized Plotter")
        actions_dict.actionCombineTimeSynchronizedPlotterWindow.setObjectName("actionCombineTimeSynchronizedPlotterWindow")
        
        actions_dict.actionTimeSynchronizedDecoderPlotter = QtWidgets.QAction(action_parent)
        actions_dict.actionTimeSynchronizedDecoderPlotter.setText("TimeSynchronizedDecoderPlotter")
        actions_dict.actionTimeSynchronizedDecoderPlotter.setObjectName("actionTimeSynchronizedDecoderPlotter")

        ## Menu:        
        top_level_menu_item = QtWidgets.QMenu(menu_parent)
        top_level_menu_item.setTitle("Create Paired Widget")
        top_level_menu_item.setObjectName("menuCreate_Paired_Widget")
        top_level_menu_item.addAction(actions_dict.actionTimeSynchronizedOccupancyPlotter)
        top_level_menu_item.addAction(actions_dict.actionTimeSynchronizedPlacefieldsPlotter)
        top_level_menu_item.addAction(actions_dict.actionTimeSynchronizedDecoderPlotter)
        top_level_menu_item.addSeparator()
        top_level_menu_item.addAction(actions_dict.actionCombineTimeSynchronizedPlotterWindow)
        
        create_linked_widget = PhoUIContainer.init_from_dict({'top_level_menu': top_level_menu_item, 'actions_dict': actions_dict})
        return create_linked_widget
    

## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("LocalMenus_AddRenderable Example")
    widget = LocalMenus_AddRenderable()
    widget.show()
    pg.exec()