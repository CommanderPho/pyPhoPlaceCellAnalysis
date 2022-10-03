from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..Qt import QtWidgets

__all__ = ['MatplotlibWidget']

class MatplotlibWidget(QtWidgets.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, disable_toolbar=True, size=(5.0, 4.0), dpi=72, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.fig = Figure(size, dpi=dpi, **kwargs)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        
        if not disable_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
        else:
            self.toolbar = None
        
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)
        
        if not disable_toolbar:
            self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)
        self.setLayout(self.vbox)
        
        
    @property
    def axes(self):
        """The axes that have been added to the figure (via add_subplot(111) or similar)."""
        return self.fig.get_axes()
    
    @property
    def ax(self):
        """The first axes property."""
        if len(self.axes) > 0:
            return self.axes[0]
        else:
            return None
         
    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()
