"""
The class :class:`.PlotAction` help the creation of a qt.QAction associated
with a :class:`.PlotWidget`.
"""


import weakref
# from silx.gui import icons
# from silx.gui import qt

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

from pyphoplacecellanalysis.External.pyqtgraph.icons import GraphIcon


class PlotAction(QtGui.QAction):
    """Base class for QAction that operates on a PlotWidget.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param icon: QIcon or str name of icon to use
    :param str text: The name of this action to be used for menu label
    :param str tooltip: The text of the tooltip
    :param triggered: The callback to connect to the action's triggered
                      signal or None for no callback.
    :param bool checkable: True for checkable action, False otherwise (default)
    :param parent: See :class:`QAction`.
    
    
    ====== Copied from the "Silx" project by Pho Hale on 2022-09-06
    
    
    """

    def __init__(self, plot, icon, text, tooltip=None, triggered=None, checkable=False, parent=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        if not isinstance(icon, QtGui.QIcon):
            # Try with icon as a string and load corresponding icon
            # icon = icons.getQIcon(icon)            
            ## TODO: do I even want/need this?
            icon = GraphIcon(icon) # icon: "tiny.png"

        super(PlotAction, self).__init__(icon, text, parent)

        if tooltip is not None:
            self.setToolTip(tooltip)

        self.setCheckable(checkable)

        if triggered is not None:
            self.triggered[bool].connect(triggered)

    @property
    def plot(self):
        """The :class:`.PlotWidget` this action group is controlling."""
        return self._plotRef()