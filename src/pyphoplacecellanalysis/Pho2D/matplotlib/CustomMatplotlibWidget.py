from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.function_helpers import function_attributes



__all__ = ['CustomMatplotlibWidget']

@function_attributes(short_name=None, tags=['matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-08 14:20')
class CustomMatplotlibWidget(QtWidgets.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Based off of pyqtgraphs's MatplotlibWidget (pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget)
    Example::

        from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget
        mw = CustomMatplotlibWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, scrollAreaContents_MinimumHeight=params.all_plots_height)
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, name='CustomMatplotlibWidget', plot_function_name=None, disable_toolbar=True, scrollable_figure=True, size=(5.0, 4.0), dpi=72, **kwargs):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to 'CustomMatplotlibWidget'.
            disable_toolbar (bool, optional): _description_. Defaults to True.
            scrollable_figure (bool, optional): If True, the figure (the canvas that renders it) is embedded in a QScrollArea to allow the user to scroll. Defaults to True.
            size (tuple, optional): _description_. Defaults to (5.0, 4.0).
            dpi (int, optional): _description_. Defaults to 72.
        """
        QtWidgets.QWidget.__init__(self)
        
        ## Init containers:
        self.params = VisualizationParameters(name=name, plot_function_name=plot_function_name, debug_print=False)
        self.plots_data = RenderPlotsData(name=name)
        self.plots = RenderPlots(name=name)
        self.ui = PhoUIContainer(name=name)
        self.ui.connections = PhoUIContainer(name=name)

        self.params.name = name
        self.params.window_title = f" - ".join([name, plot_function_name]) # name should be first so window title is rendered reasonably. kwargs.pop('plot_function_name', name)
        self.params.disable_toolbar = disable_toolbar
        self.params.scrollable_figure = scrollable_figure
        self.params.scrollAreaContents_MinimumHeight = kwargs.pop('scrollAreaContents_MinimumHeight', None)
        
        # Extract figure_kwargs:
        self.params.figure_kwargs = kwargs
        self.params.figure_kwargs['figsize'] = size
        self.params.figure_kwargs['dpi'] = dpi
        
        self.setup()
        self.buildUI()
        
    def setup(self):
        pass
    
    def buildUI(self):
        ## Init Figure and components
        self.plots.fig = Figure(**self.params.figure_kwargs)
        self.ui.canvas = FigureCanvas(self.plots.fig)
        self.ui.canvas.setParent(self)
        if not self.params.disable_toolbar:
            self.ui.toolbar = NavigationToolbar(self.ui.canvas, self)
        else:
            self.ui.toolbar = None
            
        if self.params.scrollable_figure:
            self._buildUI_buildScrollableWidget()
        else:
            self._buildUI_buildNonScrollableWidget()

        # self.setWindowTitle(self.params.window_title) # sets the window's title


    def _buildUI_buildNonScrollableWidget(self):
        """ sets up the widget to contain a basic layout with no scrollability """
        self.ui.root_vbox = QtWidgets.QVBoxLayout()
        self.ui.root_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.root_vbox.setObjectName('root_vbox_layout')
        ## Non-scrollable version:
        target_vbox = self.ui.root_vbox
        ## Add the real widgets:
        if not self.params.disable_toolbar:
            target_vbox.addWidget(self.ui.toolbar)
        target_vbox.addWidget(self.ui.canvas)
        ## Common:
        self.setLayout(self.ui.root_vbox)
        

    def _buildUI_buildScrollableWidget(self):
        """ sets up the widget to contain a QScrollArea that contains the main figure """
        self.ui.root_vbox = QtWidgets.QVBoxLayout()
        self.ui.root_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.root_vbox.setObjectName('root_vbox')
        
        ### Scrollable Version:
        ## Build the contents widget and inner_contents_vbox:
        self.ui.scrollAreaContentsWidget = QtWidgets.QWidget()
        self.ui.scrollAreaContentsWidget.setObjectName('scrollAreaContentsWidget')
        self.ui.inner_contents_vbox = QtWidgets.QVBoxLayout()
        self.ui.inner_contents_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.inner_contents_vbox.setObjectName('inner_contents_vbox')
        target_vbox = self.ui.inner_contents_vbox
        ## Add the real widgets:
        if not self.params.disable_toolbar:
            target_vbox.addWidget(self.ui.toolbar)
        target_vbox.addWidget(self.ui.canvas)
        self.ui.scrollAreaContentsWidget.setLayout(self.ui.inner_contents_vbox)
        
        ## Optional Scroll Area Widget:
        self.ui.scrollAreaWidget = QtWidgets.QScrollArea() # Scroll Area which contains the widgets, set as the centralWidget
        
        #Scroll Area Properties
        self.ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn) #  Qt.ScrollBarAlwaysOn
        self.ui.scrollAreaWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Qt.ScrollBarAlwaysOff
        self.ui.scrollAreaWidget.setWidgetResizable(True)
        # self.ui.scrollAreaContentsWidget = widget # Widget that contains the collection of Vertical Box
        if self.ui.scrollAreaContentsWidget is not None:
            # Set contents widget if we have it:
            self.ui.scrollAreaWidget.setWidget(self.ui.scrollAreaContentsWidget)
        self.ui.root_vbox.addWidget(self.ui.scrollAreaWidget)

        self.setLayout(self.ui.root_vbox)
        # Set the minimumHeight to the
        if self.params.scrollAreaContents_MinimumHeight is not None:
            self.ui.scrollAreaContentsWidget.setMinimumHeight(self.params.scrollAreaContents_MinimumHeight)
        
        
    @property
    def fig(self):
        """The main figure."""
        return self.getFigure()
    
    @property
    def axes(self):
        """The axes that have been added to the figure (via add_subplot(111) or similar)."""
        return self.plots.fig.get_axes()
    
    @property
    def ax(self):
        """The first axes property."""
        if len(self.axes) > 0:
            return self.axes[0]
        else:
            return None
         

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


    def getFigure(self):
        return self.plots.fig
        
    def draw(self):
        """ This is what is taking a ton of time with the .Agg backend at least and complex figures. Need a way to freeze it out so it isn't called until it is needed. 
        
        """
        #TODO 2023-07-06 15:05: - [ ] PERFORMANCE - REDRAW
        self.ui.canvas.draw()
        

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    def showEvent(self, event):
        """ called only when the widget is shown for the first time. """
        if self.params.debug_print:
        print(f'showEvent(self, event: {event})')
        print(f'\tevent.spontaneous(): {event.spontaneous()}')
        if event.spontaneous():
            if self.params.debug_print:
            print(f'\tfirst show!')
            # Check if the event is spontaneous to identify the first time the widget is shown
            # self.on_first_show()

        if self.isVisible():
            # IMPORTANT: It seems that the title must not be updated until after ui.mw.show() is called or else it crashes the Jupyter kernel! This is why we check for the first show and make sure that the window is visible
            self.window().setWindowTitle(self.params.window_title)
    
        # Call the base class implementation
        super().showEvent(event)