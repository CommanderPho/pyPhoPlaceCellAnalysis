# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
from neuropy.utils.mixins.dict_representable import get_dict_subset # used to filter kwargs down to proper Figure inputs

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget

from mpl_multitab import MplMultiTab, MplMultiTab2D ## Main imports
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# from neuropy.utils.matplotlib_helpers import TabbedMatplotlibFigures


__all__ = ['CustomMplMultiTab', 'CustomMatplotlibTabbedWidget']

class CustomMplMultiTab(MplMultiTab):
    """ 
        from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibTabbedWidget import CustomMplMultiTab
    """
    # @property
    # def fig(self):
    #     """The main figure."""
    #     return self.getFigure()
    
    # @property
    # def axes(self):
    #     """The axes that have been added to the figure (via add_subplot(111) or similar)."""
    #     return self.plots.fig.get_axes()
    
    # @property
    # def ax(self):
    #     """The first axes property."""
    #     if len(self.axes) > 0:
    #         return self.axes[0]
    #     else:
    #         return None

    @property
    def tabs_dict(self) -> Dict:
        """The child tabs."""
        return self.tabs._items
         
    @property
    def windowTitle(self):
        """The windowTitle property."""
        return self.params.window_title
    @windowTitle.setter
    def windowTitle(self, value):
        self.params.window_title = value
        if self.window().isVisible():
            print(f'updating the window title!!')
            self.window().setWindowTitle(self.params.window_title)


    def __init__(self, name='CustomMplMultiTab', plot_function_name=None, disable_toolbar=True, scrollable_figure=True, size=(5.0, 4.0), dpi=72, figures=(), title=None, parent=None, **kwargs):
        super().__init__(figures=figures, title=title, parent=parent) # MplMultiTab.__init__(self)
        
        ## Init containers:
        self.params = VisualizationParameters(name=name, plot_function_name=plot_function_name, debug_print=False)
        self.plots_data = RenderPlotsData(name=name)
        self.plots = RenderPlots(name=name)
        self.ui = PhoUIContainer(name=name)
        self.ui.connections = PhoUIContainer(name=name)

        self.params.name = name
        if plot_function_name is not None:
            self.params.window_title = f" - ".join([name, plot_function_name]) # name should be first so window title is rendered reasonably. kwargs.pop('plot_function_name', name)
        else:
            # TypeError: sequence item 1: expected str instance, NoneType found
            self.params.window_title = f"{name}"
            
        self.params.disable_toolbar = disable_toolbar
        self.params.scrollable_figure = scrollable_figure
        self.params.scrollAreaContents_MinimumHeight = kwargs.pop('scrollAreaContents_MinimumHeight', None)
        self.params.verticalScrollBarPolicy = kwargs.pop('verticalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.params.horizontalScrollBarPolicy = kwargs.pop('horizontalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Extract figure_kwargs:
        self.params.figure_kwargs = get_dict_subset(kwargs, ['figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth', 'frameon', 'subplotpars', 'tight_layout', 'constrained_layout', 'layout']) 
        self.params.figure_kwargs['figsize'] = size
        self.params.figure_kwargs['dpi'] = dpi
        
        self.setup()
        self.buildUI()
        
    def setup(self):
        pass
    
    def buildUI(self):
        ## Init Figure and components
        pass
        # # Default matplotlib toolbar:
        # self.ui.toolbar = None
        # if not self.params.disable_toolbar:
        #     self._buildUI_default_toolbar()
        
        # if self.params.scrollable_figure:
        #     self._buildUI_buildScrollableWidget()
        # else:
        #     self._buildUI_buildNonScrollableWidget()

        # self.ui.statusBar = None
        # # self._buildUI_setup_statusbar()


    # def getFigure(self):
    #     return self.plots.fig
        
    def show(self):
        # This is needed so the initial plot is done when launching the gui
        return super().show()


    def draw(self):
        """ Redraws the current figure.
        This is what is taking a ton of time with the .Agg backend at least and complex figures. Need a way to freeze it out so it isn't called until it is needed. 
        """
        #TODO 2023-07-06 15:05: - [ ] PERFORMANCE - REDRAW
        for k, v in self.tabs_dict.items():
            # print(f"k: {k}, v: {v}")
            v.canvas.draw()
            




@function_attributes(short_name=None, tags=['UNFINISHED', 'NOT_YET_WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-12 01:27', related_items=[])
class CustomMatplotlibTabbedWidget(CustomMatplotlibWidget):
    """ Extends CustomMatplotlibWidget with time-synchronization properties 
    
    #TODO 2024-01-06 06:13: - [ ] What does the "time-synchronization" refer to?  It seems to be used in stacks of pre-computed epoch posteriors only.
        Oh right, it's  also used somewhere in SpikeRaster2D to add interactive matplotlib-based plots below I think    

    ?? There's also `TimeSynchronizedPlotterBase`, which this class does not inherit from but the majority of the actually time-synchronized plotting subclasses do.
    
    
    Example::

        from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibTabbedWidget import CustomMatplotlibTabbedWidget
        mw = CustomMatplotlibTabbedWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, scrollAreaContents_MinimumHeight=params.all_plots_height)
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, disable_toolbar=True, size=(5.0, 4.0), dpi=72, **kwargs):
        super(CustomMatplotlibTabbedWidget, self).__init__(disable_toolbar=disable_toolbar, size=size, dpi=dpi, **kwargs)
        
    @property
    def root_multi_tab(self) -> MplMultiTab:
        """The root_multi_tab property."""
        return self.ui.root_multi_tab

        
    @classmethod
    def init_from_tab_names(cls, tab_name_list: List[str], **kwargs):
        _obj = cls(**kwargs)

        
        return _obj
        

    def setup(self):
        pass
    
    def buildUI(self):
        ## Init Figure and components
        # self.plots.fig = Figure(**self.params.figure_kwargs)
        # self.ui.canvas = FigureCanvas(self.plots.fig)
        # self.ui.canvas.setParent(self)

        self.ui.root_multi_tab = MplMultiTab(parent=self, title=self.params.window_title)
        self.ui.root_vbox.addWidget(self.ui.root_multi_tab)
        
        # Default matplotlib toolbar:
        self.ui.toolbar = None
        if not self.params.disable_toolbar:
            self._buildUI_default_toolbar()
        
        if self.params.scrollable_figure:
            self._buildUI_buildScrollableWidget()
        else:
            self._buildUI_buildNonScrollableWidget()

        self.ui.statusBar = None
        # self._buildUI_setup_statusbar()
        self.toast = None
        self._init_ToastShowingWidgetMixin()
        

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        ## Update all children axes:
        for curr_ax in self.axes:
            curr_ax.set_xlim(start_t, end_t)
        self.draw()
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        
        
    
        