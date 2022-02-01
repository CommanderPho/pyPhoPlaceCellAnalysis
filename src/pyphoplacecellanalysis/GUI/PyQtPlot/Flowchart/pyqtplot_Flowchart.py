from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


# Import the custom nodes:
from .CustomNodes.ImageViewNode import ImageViewNode
from .CustomNodes.UnsharpMaskNode import UnsharpMaskNode
from .CustomNodes.PipelineInputDataNode import PipelineInputDataNode, PipelineFilteringDataNode
from .CustomNodes.PipelineDisplayNode import PipelineDisplayNode

def plot_flowchartWidget(title='PhoFlowchartApp'):
    app = pg.mkQApp(title)

    ## Create main window with a grid layout inside
    win = QtGui.QMainWindow()
    win.setWindowTitle(f'PhoFlowchartApp: pyqtgraph FlowchartCustomNodes: {title}')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    ## Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={
        'dataIn': {'io': 'in'},
        'dataOut': {'io': 'out'}    
    })
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    ## Create two ImageView widgets to display the raw and processed data with contrast
    ## and color control.
    v1 = pg.ImageView()
    v2 = pg.ImageView()
    layout.addWidget(v1, 0, 1)
    layout.addWidget(v2, 1, 1)

    win.show()

    ## generate random input data
    data = np.random.normal(size=(100,100))
    data = 25 * pg.gaussianFilter(data, (5,5))
    data += np.random.normal(size=(100,100))
    data[40:60, 40:60] += 15.0
    data[30:50, 30:50] += 15.0
    #data += np.sin(np.linspace(0, 100, 1000))
    #data = metaarray.MetaArray(data, info=[{'name': 'Time', 'values': np.linspace(0, 1.0, len(data))}, {}])

    ## Set the raw data as the input value to the flowchart
    fc.setInput(dataIn=data)

    ## To make our custom node classes available in the flowchart context menu,
    ## we can either register them with the default node library or make a
    ## new library.

    ## Method 1: Register to global default library:
    #fclib.registerNodeType(ImageViewNode, [('Display',)])
    #fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

    ## Method 2: If we want to make our custom node available only to this flowchart,
    ## then instead of registering the node type globally, we can create a new 
    ## NodeLibrary:
    library = fclib.LIBRARY.copy() # start with the default node set
    library.addNodeType(ImageViewNode, [('Display',)])
    # Add the unsharp mask node to two locations in the menu to demonstrate
    # that we can create arbitrary menu structures
    library.addNodeType(UnsharpMaskNode, [('Image',)])
    
    library.addNodeType(PipelineInputDataNode, [('Data',), 
                                        ('Pho Pipeline','Input')])
    library.addNodeType(PipelineFilteringDataNode, [('Filters',), 
                                        ('Pho Pipeline','Filtering')])
    
    library.addNodeType(PipelineDisplayNode, [('Display',), 
                                        ('Pho Pipeline','Display')])

    fc.setLibrary(library)


    ## Now we will programmatically add nodes to define the function of the flowchart.
    ## Normally, the user will do this manually or by loading a pre-generated
    ## flowchart file.

    v1Node = fc.createNode('ImageView', pos=(0, -150))
    v1Node.setView(v1)

    v2Node = fc.createNode('ImageView', pos=(150, -150))
    v2Node.setView(v2)

    fNode = fc.createNode('UnsharpMask', pos=(0, 0))
    fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
    fc.connectTerminals(fc['dataIn'], v1Node['data'])
    fc.connectTerminals(fNode['dataOut'], v2Node['data'])
    fc.connectTerminals(fNode['dataOut'], fc['dataOut'])

    # end node setup:
    win.resize(800,600)
    return win, app



if __name__ == '__main__':
    pg.exec()
