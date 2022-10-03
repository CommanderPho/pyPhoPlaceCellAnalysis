from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


__all__ = ['CustomMatplotlibWidget']

class CustomMatplotlibWidget(QtWidgets.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Based off of pyqtgraphs's MatplotlibWidget (pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget)
    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, name='CustomMatplotlibWidget', disable_toolbar=True, size=(5.0, 4.0), dpi=72, **kwargs):
        QtWidgets.QWidget.__init__(self)
        
        ## Init containers:
        self.params = VisualizationParameters(name=name)
        self.plots_data = RenderPlotsData(name=name)
        self.plots = RenderPlots(name=name)
        self.ui = PhoUIContainer(name=name)
        self.ui.connections = PhoUIContainer(name=name)

        self.params.name = name
        self.params.window_title = kwargs.pop('plot_function_name', name)
        self.params.disable_toolbar = disable_toolbar
        self.params.figure_kwargs = kwargs
        self.params.figure_kwargs['figsize'] = size
        self.params.figure_kwargs['dpi'] = dpi
        
        self.setup()
        self.buildUI()
        
        
    def setup(self):
        pass
    
    def buildUI(self):
        ## Init Figure and components
        # self.fig = Figure(size, dpi=dpi, **kwargs)
        self.plots.fig = Figure(**self.params.figure_kwargs)
        
        self.ui.canvas = FigureCanvas(self.plots.fig)
        self.ui.canvas.setParent(self)
        
        if not self.params.disable_toolbar:
            self.ui.toolbar = NavigationToolbar(self.ui.canvas, self)
        else:
            self.ui.toolbar = None
            
        self.buildMainContentWidget()



    def buildMainContentWidget(self):
        self.ui.vbox = QtWidgets.QVBoxLayout()
        self.ui.vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.vbox.setObjectName('root_vbox')
        
        # ## Non-scrollable version:
        # target_vbox = self.ui.vbox
        # ## Add the real widgets:
        # if not self.params.disable_toolbar:
        #     target_vbox.addWidget(self.ui.toolbar)
        # target_vbox.addWidget(self.ui.canvas)

        ## Scrollable Version:
        
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
        
        self.ui.vbox.addWidget(self.ui.scrollAreaWidget)
        self.setLayout(self.ui.vbox)
        
        
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
         
    def getFigure(self):
        return self.plots.fig
        
    def draw(self):
        self.ui.canvas.draw()
        

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    