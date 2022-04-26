from pyphoplacecellanalysis.External.pyqtgraph.flowchart import Flowchart, Node
import pyphoplacecellanalysis.External.pyqtgraph.flowchart.library as fclib
from pyphoplacecellanalysis.External.pyqtgraph.flowchart.library.common import CtrlNode
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.AssociatedOutputWidgetNodeMixin import AssociatedOutputWidgetNodeMixin

## At this point, we need some custom Node classes since those provided in the library
## are not sufficient. Each node will define a set of input/output terminals, a 
## processing function, and optionally a control widget (to be displayed in the 
## flowchart control panel)

class ImageViewNode(AssociatedOutputWidgetNodeMixin, Node):
    """Node that displays image data in a view (ImageView widget)"""
    nodeName = 'ImageView'

    def __init__(self, name):
        # Initialize the associated view
        self.view = None
        self.on_remove_function = None
        ## Initialize node with only a single input terminal
        Node.__init__(self, name, terminals={'data': {'io':'in'}})
        
    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        if display and self.view is not None:
            ## the 'data' argument is the value given to the 'data' terminal
            if data is None:
                self.view.setImage(np.zeros((1,1))) # give a blank array to clear the view
            else:
                self.view.setImage(data)

