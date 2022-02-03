from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.widgets import MatplotlibWidget
import pyqtgraph as pg
import numpy as np


# Import the custom nodes:
from .CustomNodes.ImageViewNode import ImageViewNode
from .CustomNodes.UnsharpMaskNode import UnsharpMaskNode
from .CustomNodes.PipelineInputDataNode import PipelineInputDataNode, PipelineFilteringDataNode
from .CustomNodes.PipelineDisplayNode import PipelineDisplayNode




def plot_flowchartWidget(title='PhoFlowchartApp'):
    """ 

    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget
        pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp')
    """
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
    # w = fc.widget() # This is unused?
    # Add the flowchart widget. This is actually not the programmatic programming environment itself, it's the column that lists the nodes and lets you set their parameters.
    layout.addWidget(fc.widget(), 0, 0, 2, 1) # spans 2 rows and 1 column

    ## Result/Visualization Widgets:
    ## build an initial namespace for console commands to be executed in (this is optional;
    ## the user can always import these modules manually)
    namespace = {'pg': pg, 'np': np}

    ## initial text to display in the console
    text = """
    This is an interactive python console. The numpy and pyqtgraph modules have already been imported 
    as 'np' and 'pg'. 

    Go, play.
    """
    console_widget = ConsoleWidget(namespace=namespace, text=text)
    layout.addWidget(console_widget, 0, 1)
    
    # mw = MatplotlibWidget()
    # subplot = mw.getFigure().add_subplot(111)
    # subplot.plot(x,y)
    # mw.draw()
    # # win.show()

    _register_custom_node_types(fc)
    
    _add_pho_pipeline_programmatic_flowchart_nodes(app, fc, layout)
    # _add_default_example_programmatic_flowchart_nodes(fc, layout)    

    # end node setup:
    win.show()
    win.resize(800,600)
    return win, app



def _register_custom_node_types(fc):
    """Register Custom Nodes so they appear in the flowchart context menu"""
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
    



def _add_pho_pipeline_programmatic_flowchart_nodes(app, fc, layout):
    ## Now we will programmatically add nodes to define the function of the flowchart.
    ## Normally, the user will do this manually or by loading a pre-generated
    ## flowchart file.
    """[summary]

    Args:
        fc ([type]): [description]
        layout ([type]): a grid layout to add result/visualization widgets to. 
    """

    def on_remove_widget_fn(widget):
        """ the callback to remove the widget from the layout.
            implicitly used 'layout'.
        """
        item_index = layout.indexOf(widget)
        print(f'on_remove_widget_fn(...): item_index: {item_index}')
        item = layout.itemAt(item_index)
        widget = item.widget() # this should be the same as the passed in widget, but do this just to be sure
        layout.removeWidget(widget)

    ## Result/Visualization Widgets:
    # need app and win
    # new_view_widget = pg.GraphicsWidget()
    
    # Build the new outer container widget to hold the other views:
    new_view_widget = QtGui.QWidget()
    layout.addWidget(new_view_widget, 1, 1) # start at 1 since the console is available at 0
    
    # create a layout for the new container view:
    new_view_layout = QtGui.QGridLayout()
    new_view_widget.setLayout(new_view_layout)
    # build the internal widget
    new_root_render_widget = pg.GraphicsLayoutWidget()
    new_view_layout.addWidget(new_root_render_widget, 1, 1) # add the new view to the new layout
    
    
    # new_view_widget.setCentralWidget(new_root_render_widget)
        
    
    ## Create two ImageView widgets to display the raw and processed data with contrast
    ## and color control.
    # v1 = pg.ImageView()
    # v2 = pg.ImageView()
    # layout.addWidget(v1, 1, 1) # start at 1 since the console is available at 0
    # layout.addWidget(v2, 2, 1)
    
    
    ## Set the raw data as the input value to the flowchart
    fc.setInput(dataIn=None)

    pipeline_input_node = fc.createNode('PipelineInputDataNode', pos=(-200, 50))
    # pipeline_input_node.setView(v1, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node

    pipeline_filter_node = fc.createNode('PipelineFilteringDataNode', pos=(-26, 50))
    # pipeline_filter_node.setView(v2, on_remove_function=on_remove_widget_fn)

    pipeline_display_node = fc.createNode('PipelineDisplayNode', pos=(154, 20))
    pipeline_display_node.setApp(app) # Sets the shared singleton app instance
    pipeline_display_node.setView(new_root_render_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the 
    # pipeline_display_node.setView(new_view_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node

    # Setup connections:
    
    # fc.connectTerminals(fc['dataIn'], pipeline_display_node['dataIn'])
    
    # Input Node Outputs:
    fc.connectTerminals(pipeline_input_node['loaded_pipeline'], pipeline_filter_node['pipeline'])
    fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_filter_node['active_data_mode'])
    
    fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_display_node['active_data_mode'])
    
    # Computation Node Outputs:
    fc.connectTerminals(pipeline_filter_node['filtered_pipeline'], pipeline_display_node['active_pipeline'])
    fc.connectTerminals(pipeline_filter_node['computation_configs'], pipeline_display_node['active_session_computation_configs'])
    fc.connectTerminals(pipeline_filter_node['filter_configurations'], pipeline_display_node['active_session_filter_configurations'])

    fc.connectTerminals(pipeline_filter_node['filtered_pipeline'], fc['dataOut']) # raw pipeline output from computation node
    
    # Display Node Outputs:   
    
    

def _add_default_example_programmatic_flowchart_nodes(fc, layout):
    ## Now we will programmatically add nodes to define the function of the flowchart.
    ## Normally, the user will do this manually or by loading a pre-generated
    ## flowchart file.
    """[summary]

    Args:
        fc ([type]): [description]
        layout ([type]): a grid layout to add result/visualization widgets to. 
    """

    def on_remove_widget_fn(widget):
        """ the callback to remove the widget from the layout.
            implicitly used 'layout'.
        """
        item_index = layout.indexOf(widget)
        print(f'on_remove_widget_fn(...): item_index: {item_index}')
        item = layout.itemAt(item_index)
        widget = item.widget() # this should be the same as the passed in widget, but do this just to be sure
        layout.removeWidget(widget)

    ## Result/Visualization Widgets:
    ## Create two ImageView widgets to display the raw and processed data with contrast
    ## and color control.
    v1 = pg.ImageView()
    v2 = pg.ImageView()
    layout.addWidget(v1, 1, 1) # start at 1 since the console is available at 0
    layout.addWidget(v2, 2, 1)
    
    # layout.addWidget(v1, 0, 1)
    # layout.addWidget(v2, 1, 1)
    
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

    v1Node = fc.createNode('ImageView', pos=(0, -150))
    v1Node.setView(v1, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node

    v2Node = fc.createNode('ImageView', pos=(150, -150))
    v2Node.setView(v2, on_remove_function=on_remove_widget_fn)

    fNode = fc.createNode('UnsharpMask', pos=(0, 0))
    fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
    fc.connectTerminals(fc['dataIn'], v1Node['data'])
    fc.connectTerminals(fNode['dataOut'], v2Node['data'])
    fc.connectTerminals(fNode['dataOut'], fc['dataOut'])



if __name__ == '__main__':
    pg.exec()
