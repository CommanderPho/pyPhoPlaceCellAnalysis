# addRenderableActions
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.actions.UNUSED_plotAction import PlotAction

from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable
# from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.Uic_AUTOGEN_LocalMenus_AddRenderable import Ui_LocalMenus_AddRenderable

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, SessionEpochs2DRenderTimeEpochs, PBE_2DRenderTimeEpochs, Laps2DRenderTimeEpochs # Time Intervals/Epochs
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves, PositionRenderTimeCurves, MUA_RenderTimeCurves ## Time Curves

def connect_renderable_menu(sess, destination_plot):
    """ Hook up appropriate signals to add Renderable menu:
    sess: Session
    destination_plot: e.g. active_2d_plot, active_3d_plot
    """
    widget = LocalMenus_AddRenderable() # get the UI widget containing the menu items:
    renderable_menu = widget.ui.menuAdd_Renderable
    
    ## Time Intervals/Epochs:
    submenu_addTimeIntervals = [widget.ui.actionAddTimeIntervals_Laps, widget.ui.actionAddTimeIntervals_PBEs, widget.ui.actionAddTimeIntervals_Session_Epochs, widget.ui.actionAddTimeIntervals_Custom]
    submenu_addTimeIntervalCallbacks = [lambda evt=None: SessionEpochs2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.epochs, destination_plot=destination_plot),
                                        lambda evt=None: PBE_2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.pbe, destination_plot=destination_plot),
                                        lambda evt=None: Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=sess.laps, destination_plot=destination_plot),
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
    return widget, renderable_menu, (submenu_addTimeIntervals, submenu_addTimeIntervalCallbacks, submenu_addTimeIntervals_Connections), (submenu_addTimeCurves, submenu_addTimeCurvesCallbacks, submenu_addTimeCurves_Connections)



class AddRenderableTimeEpochsAction(PlotAction):
    """QAction controlling TimeEpochs on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        tooltip = 'Show plot axis when checked, otherwise hide them'
        PlotAction.__init__(self,
                            plot,
                            icon='axis',
                            text='show axis',
                            tooltip=tooltip,
                            triggered=self._actionTriggered,
                            checkable=False,
                            parent=parent)
        # self.setChecked(self.plot.isAxesDisplayed())
        # plot._sigAxesVisibilityChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
       	Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=self.sess.laps, destination_plot=self.plot)
        
        
class AddRenderableLapsTimeEpochsAction(AddRenderableTimeEpochsAction):
    """QAction controlling TimeEpochs on a :class:`.PlotWidget`.
    """
    def __init__(self, plot, parent=None):
        tooltip = 'Show plot axis when checked, otherwise hide them'
        AddRenderableTimeEpochsAction.__init__(self, plot, icon='axis', text='Laps', tooltip=tooltip, triggered=self._actionTriggered, checkable=True, parent=parent)

    def _actionTriggered(self, checked=False):
        Laps2DRenderTimeEpochs.add_render_time_epochs(curr_sess=self.sess.laps, destination_plot=self.plot)
        
        
        
        
class AddRenderableTimeCurveAction(PlotAction):
    """QAction controlling TimeCurve on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        tooltip = 'Show plot axis when checked, otherwise hide them'
        PlotAction.__init__(self,
                            plot,
                            icon='axis',
                            text='show axis',
                            tooltip=tooltip,
                            triggered=self._actionTriggered,
                            checkable=True,
                            parent=parent)
        self.setChecked(self.plot.isAxesDisplayed())
        plot._sigAxesVisibilityChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        Laps2DRenderTimeEpochs
        
        self.plot.setAxesDisplayed(checked)
        
        
        
# class ShowAxisAction(PlotAction):
#     """QAction controlling axis visibility on a :class:`.PlotWidget`.

#     :param plot: :class:`.PlotWidget` instance on which to operate
#     :param parent: See :class:`QAction`
#     """

#     def __init__(self, plot, parent=None):
#         tooltip = 'Show plot axis when checked, otherwise hide them'
#         PlotAction.__init__(self,
#                             plot,
#                             icon='axis',
#                             text='show axis',
#                             tooltip=tooltip,
#                             triggered=self._actionTriggered,
#                             checkable=True,
#                             parent=parent)
#         self.setChecked(self.plot.isAxesDisplayed())
#         plot._sigAxesVisibilityChanged.connect(self.setChecked)

#     def _actionTriggered(self, checked=False):
#         self.plot.setAxesDisplayed(checked)