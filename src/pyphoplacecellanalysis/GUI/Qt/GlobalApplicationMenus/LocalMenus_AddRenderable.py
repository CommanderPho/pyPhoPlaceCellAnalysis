# LocalMenus_AddRenderable.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\LocalMenus_AddRenderable.ui automatically by PhoPyQtClassGenerator VSCode Extension

from functools import partial

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.Uic_AUTOGEN_LocalMenus_AddRenderable import Ui_LocalMenus_AddRenderable

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, SessionEpochs2DRenderTimeEpochs, PBE_2DRenderTimeEpochs, Laps2DRenderTimeEpochs # Time Intervals/Epochs
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves, PositionRenderTimeCurves, MUA_RenderTimeCurves ## Time Curves


## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus import LocalMenus_AddRenderable


# class LocalMenus_AddRenderable(QtWidgets.QWidget):
class LocalMenus_AddRenderable(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        # self.ui = uic.loadUi("../pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/GlobalApplicationMenus/LocalMenus_AddRenderable.ui", self) # Load the .ui file
        # AUTOGEN version:
        self.ui = Ui_LocalMenus_AddRenderable()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        self.initUI()
        # Do not call show, as this is just to add menus, and should be destroyed when done
        # self.show() # Show the GUI

    def initUI(self):
        pass


    @classmethod
    def build_renderable_menu(cls, destination_plot, sess):
        """ Builds the callbacks needed and connects them to the QActions and QMenus for the specific destination_plot to be used as context menus. 
        sess: Session
        destination_plot: e.g. active_2d_plot, active_3d_plot
        """
        widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
        renderable_menu = widget.ui.menuAdd_Renderable
        
        ## Time Intervals/Epochs:
        submenu_addTimeIntervals = [widget.ui.actionAddTimeIntervals_Laps, widget.ui.actionAddTimeIntervals_PBEs, widget.ui.actionAddTimeIntervals_Session_Epochs, widget.ui.actionAddTimeIntervals_Custom]
        submenu_addTimeIntervalCallbacks = [lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: PBE_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.pbe, destination_plot=destination_plot),
                                            lambda evt=None: SessionEpochs2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.epochs, destination_plot=destination_plot),
                                            lambda evt=None: print(f'actionAddTimeIntervals_Custom not yet supported')]
        submenu_addTimeIntervals_Connections = []
        for an_action, a_callback in zip(submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeIntervals_Connections.append(_curr_conn)

        ## Time Curves:
        submenu_addTimeCurves = [widget.ui.actionAddTimeCurves_Position, widget.ui.actionAddTimeCurves_Random, widget.ui.actionAddTimeCurves_Custom]
        submenu_addTimeCurvesCallbacks = [lambda evt=None: PositionRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            lambda evt=None: MUA_RenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),
                                            # lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
                                            lambda evt=None: print(f'actionAddTimeCurves_Custom not yet supported')]
        submenu_addTimeCurves_Connections = []
        for an_action, a_callback in zip(submenu_addTimeCurves, submenu_addTimeCurvesCallbacks):
            _curr_conn = an_action.triggered.connect(a_callback)
            submenu_addTimeCurves_Connections.append(_curr_conn)
            
        # Connect Clear actions:
        widget.ui.actionClear_all_Time_Curves.triggered.connect(destination_plot.clear_all_3D_time_curves)
        widget.ui.actionClear_all_Time_Intervals.triggered.connect(destination_plot.clear_all_rendered_intervals)
        def _clear_all_both():
            destination_plot.clear_all_3D_time_curves()
            destination_plot.clear_all_rendered_intervals()
        widget.ui.actionClear_all_Renderables.triggered.connect(_clear_all_both)
        
        return widget, renderable_menu, (submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks, submenu_addTimeIntervals_Connections), (submenu_addTimeCurves, submenu_addTimeCurvesCallbacks, submenu_addTimeCurves_Connections)


    @classmethod
    def add_renderable_context_menu(cls, active_2d_plot, sess):
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
            
            # parent_context_menus = parent_widget.getMenu() #. .getViewWidget() # GraphicsLayoutWidget 
            # parent_context_menus = parent_widget.getContextMenus() #. .getViewWidget() # GraphicsLayoutWidget 
            # parent_context_menus
            # print(f'parent_context_menus.actions: {parent_context_menus.actions()}')
            # print(f'parent_context_menus.actions: {[an_action.text() for an_action in parent_context_menus.actions()]}') # parent_context_menus.actions: ['Transforms', 'Downsample', 'Average', 'Alpha', 'Grid', 'Points']
            
            # active_parent_menu = parent_context_menus
            active_parent_menu = top_level_parent_context_menu
            active_parent_menu.addSeparator()
            active_parent_menu.addMenu(additional_menu)
            if debug_print:
                print(f'parent_context_menus.actions: {[an_action.text() for an_action in active_parent_menu.actions()]}') # parent_context_menus.actions: ['Transforms', 'Downsample', 'Average', 'Alpha', 'Grid', 'Points']
                
         ## Build `partial` versions of the functions specific to each raster plotter that can be called with no arguments (capturing the destination plotter and the session
        # build_renderable_menu_to_Spike2DRaster = partial(cls.build_renderable_menu, active_2d_plot, sess) # destination_plot
        active_2d_plot_renderable_menus = cls.build_renderable_menu(active_2d_plot, sess)
        widget_2d_menu = active_2d_plot_renderable_menus[0]
        menuAdd_Renderable = widget_2d_menu.ui.menuAdd_Renderable
        
        ## Specific to SpikeRaster2D:        
        ## Add the custom menu to the context menus of the plots in SpikeRaster2D:        
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
        _subFn_append_custom_menu_to_context_menu(main_plot_widget, menuAdd_Renderable)
        _subFn_append_custom_menu_to_context_menu(background_static_scroll_plot_widget, menuAdd_Renderable)
        
        # Add the reference to the context menus to owner, so it isn't released:
        ## TODO: currently replaces the dict entry, which we might want to use for other menus
        active_2d_plot.ui.menus = PhoUIContainer.init_from_dict({'custom_context_menus': PhoUIContainer.init_from_dict({'add_renderables': active_2d_plot_renderable_menus})})        
        return menuAdd_Renderable # try returning just the menu and not the stupid references to everything # Works when we hold a reference

    

## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("LocalMenus_AddRenderable Example")
    widget = LocalMenus_AddRenderable()
    widget.show()
    pg.exec()