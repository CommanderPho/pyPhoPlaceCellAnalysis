from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

## At this point, we need some custom Node classes since those provided in the library
## are not sufficient. Each node will define a set of input/output terminals, a 
## processing function, and optionally a control widget (to be displayed in the 
## flowchart control panel)

class ImageViewNode(Node):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'ImageView'
    
    def __init__(self, name):
        self.view = None
        ## Initialize node with only a single input terminal
        Node.__init__(self, name, terminals={'data': {'io':'in'}})
        
    def setView(self, view):  ## setView must be called by the program
        self.view = view
        
    def process(self, data, display=True):
        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        
        if display and self.view is not None:
            ## the 'data' argument is the value given to the 'data' terminal
            if data is None:
                self.view.setImage(np.zeros((1,1))) # give a blank array to clear the view
            else:
                self.view.setImage(data)




## To make our custom node classes available in the flowchart context menu,
## we can either register them with the default node library or make a
## new library.

        
## Method 1: Register to global default library:
#fclib.registerNodeType(ImageViewNode, [('Display',)])
#fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

# ## Method 2: If we want to make our custom node available only to this flowchart,
# ## then instead of registering the node type globally, we can create a new 
# ## NodeLibrary:
# library = fclib.LIBRARY.copy() # start with the default node set
# library.addNodeType(ImageViewNode, [('Display',)])
# # Add the unsharp mask node to two locations in the menu to demonstrate
# # that we can create arbitrary menu structures
# library.addNodeType(UnsharpMaskNode, [('Image',), 
#                                       ('Submenu_test','submenu2','submenu3')])
# fc.setLibrary(library)

