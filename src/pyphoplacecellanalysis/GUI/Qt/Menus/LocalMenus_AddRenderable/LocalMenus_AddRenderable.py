# LocalMenus_AddRenderable.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\LocalMenus_AddRenderable.ui automatically by PhoPyQtClassGenerator VSCode Extension

from functools import partial
from benedict import benedict

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from neuropy.core.user_annotations import function_attributes
from typing_extensions import TypeAlias
from nptyping import NDArray
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.Uic_AUTOGEN_LocalMenus_AddRenderable import Ui_LocalMenus_AddRenderable

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Replays_2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs, SessionEpochs2DRenderTimeEpochs, PBE_2DRenderTimeEpochs, Laps2DRenderTimeEpochs, SpikeBurstIntervals_2DRenderTimeEpochs, NewNonPBE_2DRenderTimeEpochs, NewNonPBEEndcaps_2DRenderTimeEpochs # Time Intervals/Epochs
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves, PositionRenderTimeCurves, VelocityRenderTimeCurves, ConfigurableRenderTimeCurves, MUA_RenderTimeCurves, RelativeEntropySurpriseRenderTimeCurves ## Time Curves

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import AddNewDecodedPosition_MatplotlibPlotCommand ## MatplotlibSubplots

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus import LocalMenus_AddRenderable

class LocalMenus_AddRenderable(QtWidgets.QMainWindow):
    """ A dummy-QMainWindow class that is defined in a .ui file (so it can be edited with QtDesigner GUI) and then only added to other windows (never shown)
    
    A context menu that adds renderables such as interval rectangles and plots to SpikeRaster2D
    
    Main Function: `add_renderable_context_menu`
    
    menuCreate_Paired_Widget
        actionTimeSynchronizedOccupancyPlotter
        actionTimeSynchronizedPlacefieldsPlotter


    menuAdd_Matplotlib_Plot
        actionAddMatplotlibPlot_DecodedPosition
        actionClear_all_Matplotlib_Plots
    
        
    actionDecoded_Epoch_Slices_NonPBEs
    actionAddTimeIntervals_NonPBEs
    actionSpike3DLauncher
    NewNonPBEEndcaps_2DRenderTimeEpochs
    
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
        """ *MAIN*: Builds the callbacks needed and connects them to the QActions and QMenus for the specific destination_plot to be used as context menus. 
        sess: Session
        destination_plot: e.g. active_2d_plot, active_3d_plot
        
        Adds: Time Intervals/Epochs, Time Curves (pyqtgraph), Matplotlib Plots, 
        """
        # Extract sess from the more general curr_active_pipeline:
        computation_result = curr_active_pipeline.computation_results[active_config_name]
        sess = computation_result.sess
        

        widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
        widget.programmatic_actions_dict = benedict() # PhoUIContainer.init_from_dict({})
        renderable_menu = widget.ui.menuAdd_Renderable

        # Each section updates:      
        ## Updates: `widget.programmatic_actions_dict`, 

        # Time Intervals/Epochs: _____________________________________________________________________________________________ #
        ## Creates: `submenu_addTimeIntervals_Connections`
        ## Updates: `widget.programmatic_actions_dict`, 

        submenu_addTimeIntervals: List[QtWidgets.QAction] = [widget.ui.actionAddTimeIntervals_Laps,
                                    widget.ui.actionAddTimeIntervals_PBEs,
                                    widget.ui.actionAddTimeIntervals_NonPBEs,
                                    widget.ui.actionAddTimeIntervals_NonPBEEndcaps,
                                    widget.ui.actionAddTimeIntervals_SessionEpochs,
                                    widget.ui.actionAddTimeIntervals_Ripples,
                                    widget.ui.actionAddTimeIntervals_Replays,
                                    widget.ui.actionAddTimeIntervals_Bursts,
                                    widget.ui.actionAddTimeIntervals_Custom]
        submenu_addTimeIntervalCallbacks: List[Callable] = [lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: PBE_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.pbe, destination_plot=destination_plot),
                                            lambda evt=None: NewNonPBE_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.non_pbe, destination_plot=destination_plot),
                                            lambda evt=None: NewNonPBEEndcaps_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.non_pbe_endcaps, destination_plot=destination_plot),
                                            lambda evt=None: SessionEpochs2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.epochs, destination_plot=destination_plot),
                                            lambda evt=None: Ripples_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.ripple, destination_plot=destination_plot),
                                            lambda evt=None: Replays_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.replay, destination_plot=destination_plot),
                                            lambda evt=None: SpikeBurstIntervals_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=curr_active_pipeline, destination_plot=destination_plot, active_config_name=active_config_name),
                                            lambda evt=None: print(f'actionAddTimeIntervals_Custom not yet supported')]
        
        submenu_addTimeIntervals_Connections = []
        for an_action, a_callback in zip(submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeIntervals_Connections.append(_curr_conn)
            extracted_menu_path: List[str] = PhoMenuHelper.parse_QAction_for_menu_path(an_action) # extracted_menu_path: ['AddTimeIntervals', 'Laps']
            widget.programmatic_actions_dict['.'.join(extracted_menu_path)] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed

        # Set enabled state
        # widget.ui.actionAddTimeIntervals_PBEs.setEnabled(sess.pbe is not None)
        # widget.ui.actionAddTimeIntervals_Laps.setEnabled(sess.laps is not None)
        # widget.ui.actionAddTimeIntervals_Ripples.setEnabled(sess.ripple is not None)
        # widget.ui.actionAddTimeIntervals_Replays.setEnabled(sess.has_replays)
        widget.ui.actionAddTimeIntervals_Laps.setEnabled(Laps2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.laps))
        widget.ui.actionAddTimeIntervals_Ripples.setEnabled(Ripples_2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.ripple))
        widget.ui.actionAddTimeIntervals_PBEs.setEnabled(PBE_2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.pbe))
        widget.ui.actionAddTimeIntervals_NonPBEs.setEnabled(NewNonPBE_2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.non_pbe))
        widget.ui.actionAddTimeIntervals_NonPBEEndcaps.setEnabled(NewNonPBEEndcaps_2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.non_pbe_endcaps))
        widget.ui.actionAddTimeIntervals_Replays.setEnabled(Replays_2DRenderTimeEpochs.is_render_time_epochs_enabled(sess.replay))
        widget.ui.actionAddTimeIntervals_Bursts.setEnabled(SpikeBurstIntervals_2DRenderTimeEpochs.is_render_time_epochs_enabled(curr_sess=curr_active_pipeline, active_config_name=active_config_name)) # disable by default        

        # Time Curves: _______________________________________________________________________________________________________ #
        submenu_addTimeCurves = [widget.ui.actionAddTimeCurves_Position, widget.ui.actionAddTimeCurves_Velocity, widget.ui.actionAddTimeCurves_Random, widget.ui.actionAddTimeCurves_RelativeEntropySurprise, widget.ui.actionAddTimeCurves_Custom]
        submenu_addTimeCurvesCallbacks = [lambda evt=None: PositionRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: VelocityRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: MUA_RenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: RelativeEntropySurpriseRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            # lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: ConfigurableRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            # lambda evt=None: print(f'actionAddTimeCurves_Custom not yet supported')
            ]
        submenu_addTimeCurves_Connections = []
        for an_action, a_callback in zip(submenu_addTimeCurves, submenu_addTimeCurvesCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeCurves_Connections.append(_curr_conn)
            extracted_menu_path = PhoMenuHelper.parse_QAction_for_menu_path(an_action)
            widget.programmatic_actions_dict['.'.join(extracted_menu_path)] = an_action # have to use a string keypath because `out_command_dict[*extracted_menu_path]` is not allowed
        
        # Matplotlib Plots: __________________________________________________________________________________________________ #
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



    # ==================================================================================================================================================================================================================================================================================== #
    # Adding right-click Context Menus                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #




    @classmethod
    def _show_qwidget_context_menu(cls, widget, position):
        """ Helper method to show the context menu for any QWidget """
        context_menu = getattr(widget, '_pho_custom_context_menu', None)
        if context_menu is not None:
            # Convert the position to global coordinates
            global_pos = widget.mapToGlobal(position)

            # Check if there are any existing actions to add first
            if hasattr(widget, '_pho_original_context_menu_event'):
                # If the widget had original context menu functionality, 
                # we could try to extract those actions, but this is complex
                # For now, just show our custom menu
                pass

            # Show the context menu
            context_menu.exec_(global_pos)

    @function_attributes(short_name=None, tags=['private', 'main'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-13 07:39', related_items=[])
    @classmethod
    def _helper_append_custom_menu_to_widget_context_menu_universal(cls, parent_widget, additional_menu, debug_print=False):
        """ Universal helper that works with both pyqtgraph widgets and general QWidgets
        parent_widget: QWidget or pyqtgraph widget
        additional_menu: QMenu
        """

        def _subfn_append_custom_menu_to_QWidget_context_menu(parent_widget, additional_menu, debug_print=False):
            """ 
            """

            # Ensure the widget has custom context menu policy enabled
            if parent_widget.contextMenuPolicy() == QtCore.Qt.NoContextMenu:
                parent_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            elif parent_widget.contextMenuPolicy() == QtCore.Qt.DefaultContextMenu:
                # Switch from default to custom to allow our menu additions
                parent_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

            # Get or create the context menu
            existing_context_menu = getattr(parent_widget, '_pho_custom_context_menu', None)

            if existing_context_menu is None:
                # Create a new context menu
                parent_widget._pho_custom_context_menu = QtWidgets.QMenu(parent_widget)
                existing_context_menu = parent_widget._pho_custom_context_menu

                # Check if we need to preserve any existing context menu actions
                if hasattr(parent_widget, 'contextMenuEvent'):
                    # Store the original contextMenuEvent method if it exists
                    original_context_menu_event = parent_widget.contextMenuEvent
                    parent_widget._pho_original_context_menu_event = original_context_menu_event

                # Connect the customContextMenuRequested signal
                if not hasattr(parent_widget, '_pho_context_menu_connected'):
                    parent_widget.customContextMenuRequested.connect(lambda pos, widget=parent_widget: cls._show_qwidget_context_menu(widget, pos))
                    parent_widget._pho_context_menu_connected = True

            # Add separator and the additional menu
            existing_context_menu.addSeparator()
            existing_context_menu.addMenu(additional_menu)

            if debug_print:
                print(f'QWidget context menu actions: {[an_action.text() for an_action in existing_context_menu.actions()]}')


        # @function_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-13 07:40', related_items=[])
        def _subfn_append_custom_menu_to_PyQtGraph_based_widget_context_menu(parent_widget, additional_menu, debug_print=False):
            """ For PyQtGraph-based widgets with an existing context menu (the default one by PyQtGraph)
            """
            # plot_options_context_menu = parent_widget.getContextMenus(None) # This gets the "Plot Options" menu
            # top_level_parent_context_menu = parent_context_menus.parent()
            top_level_parent_context_menu = parent_widget.vb.menu # ViewBoxMenu
            if top_level_parent_context_menu is not None:
                active_parent_menu = top_level_parent_context_menu
                active_parent_menu.addSeparator()
                active_parent_menu.addMenu(additional_menu)
                if debug_print:
                    print(f'parent_context_menus.actions: {[an_action.text() for an_action in active_parent_menu.actions()]}') # parent_context_menus.actions: ['Transforms', 'Downsample', 'Average', 'Alpha', 'Grid', 'Points']



        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Try pyqtgraph approach first (for PlotItem, ViewBox, etc.)
        try:
            if hasattr(parent_widget, 'vb') and hasattr(parent_widget.vb, 'menu'):
                # This is likely a pyqtgraph PlotItem
                _subfn_append_custom_menu_to_PyQtGraph_based_widget_context_menu(parent_widget, additional_menu, debug_print)
                return
            elif hasattr(parent_widget, 'menu'):
                # This might be a pyqtgraph ViewBox or similar
                active_parent_menu = parent_widget.menu
                active_parent_menu.addSeparator()
                active_parent_menu.addMenu(additional_menu)
                if debug_print:
                    print(f'pyqtgraph widget menu actions: {[an_action.text() for an_action in active_parent_menu.actions()]}')
                return
        except Exception as e:
            if debug_print:
                print(f'Pyqtgraph approach failed: {e}, falling back to QWidget approach')

        # Fall back to general QWidget approach
        _subfn_append_custom_menu_to_QWidget_context_menu(parent_widget, additional_menu, debug_print)





    @function_attributes(short_name=None, tags=['context-menu', 'right-click'], input_requires=[], output_provides=['active_2d_plot.ui.menus'], uses=['._helper_append_custom_menu_to_widget_context_menu_universal'], used_by=['_build_additional_spikeRaster2D_menus'], creation_date='2022-01-01 00:00', related_items=[])
    @classmethod
    def initialize_renderable_context_menu(cls, active_2d_plot, curr_active_pipeline, active_config_name, debug_print=False) -> QtWidgets.QMenu:
        """ Creates the context menus that display when right-clicking a SpikeRaster2D plot showing the actions: add_epochs, add_graph, etc
        ** ONLY FOR THE two hard-coded pyqtgraph widgets! Must be called before trying to add elsewhere

        Usage:
            active_2d_plot = spike_raster_window.spike_raster_plt_2d 
            menuAdd_Renderable = LocalMenus_AddRenderable.add_renderable_context_menu(active_2d_plot, sess)
            
        """
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        
         ## Build `partial` versions of the functions specific to each raster plotter that can be called with no arguments (capturing the destination plotter and the session
        # build_renderable_menu_to_Spike2DRaster = partial(cls.build_renderable_menu, active_2d_plot, sess) # destination_plot
        # active_2d_plot_renderable_menus = cls._build_renderable_menu(active_2d_plot, sess)
        
        active_2d_plot_renderable_menus = cls._build_renderable_menu(active_2d_plot, curr_active_pipeline, active_config_name)
        widget_2d_menu = active_2d_plot_renderable_menus[0]
        menuAdd_Renderable = widget_2d_menu.ui.menuAdd_Renderable
        # programmatic_actions_dict = widget_2d_menu.programmatic_actions_dict

        if debug_print:
            print(f'menuAdd_Renderable: {menuAdd_Renderable}, type(menuAdd_Renderable): {type(menuAdd_Renderable)}')
            # print(f'menuAdd_Renderable: {menuAdd_Renderable}, type(menuAdd_Renderable): {type(menuAdd_Renderable)}')


        # Add the reference to the context menus to owner, so it isn't released:
        ## TODO: currently replaces the dict entry, which we might want to use for other menus
        active_2d_plot.ui.menus = PhoUIContainer.init_from_dict({'custom_context_menus': PhoUIContainer.init_from_dict({'add_renderables': active_2d_plot_renderable_menus})})


        ## Specific to SpikeRaster2D:        
        ## Add the custom menu to the context menus of the plots in SpikeRaster2D:        
        if menuAdd_Renderable is not None:
            main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
            background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
            if main_plot_widget is not None:
                cls._helper_append_custom_menu_to_widget_context_menu_universal(parent_widget=main_plot_widget, additional_menu=menuAdd_Renderable, debug_print=debug_print)
            if background_static_scroll_plot_widget is not None:
                cls._helper_append_custom_menu_to_widget_context_menu_universal(parent_widget=background_static_scroll_plot_widget, additional_menu=menuAdd_Renderable, debug_print=debug_print)

            ## Setup for dock-items:
            for a_track_widget in active_2d_plot.dock_manager_widget.get_flat_widgets_list():
                LocalMenus_AddRenderable._helper_append_custom_menu_to_widget_context_menu_universal(parent_widget=a_track_widget, additional_menu=menuAdd_Renderable, debug_print=debug_print)


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