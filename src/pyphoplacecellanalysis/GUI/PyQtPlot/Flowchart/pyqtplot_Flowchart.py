from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.ReloadableNodeLibrary import ReloadableNodeLibrary


from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea


import pyqtgraph as pg
import numpy as np

from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_MainWindow import PhoPipelineMainWindow
from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow

# Import the custom nodes:
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.ImageViewNode import ImageViewNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.UnsharpMaskNode import UnsharpMaskNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineInputDataNode import PipelineInputDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineFilteringDataNode import PipelineFilteringDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineComputationsNode import PipelineComputationsNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineDisplayNode import PipelineDisplayNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PhoPythonEvalNode import PhoPythonEvalNode


"""
TODO:
It looks like CheckTable is what I want in terms of a checkbox list.

from pyqtgraph.widgets.CheckTable import CheckTable





"""
def plot_flowchartWidget(title='PhoFlowchartApp'):
    """ 

    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget
        pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp')
    """

    ## Create main window with a grid layout inside
    # win = QtGui.QMainWindow()
    # cw = QtGui.QWidget()
    # win.setCentralWidget(cw)
    
    # Use the widget defined in the designer as the central widget   
    mainAppWindow = PhoPipelineMainWindow(title)
    mainAppWindow.setWindowTitle(f'PhoFlowchartApp: pyqtgraph FlowchartCustomNodes: {title}')
    
    # get central widget:
    # cw = mainAppWindow.flowchart_controls
    cw = mainAppWindow.centralwidget
    print(f'cw: {cw}')
    
    # setup the main layout of the central widget:
    # layout = QtGui.QGridLayout()
    # cw.setLayout(layout)
    
    layout = QtGui.QVBoxLayout()
    cw.setLayout(layout)
    

    # area = _build_dock_area(mainAppWindow, layout)    
    mainAppWindow.area = DockArea()
    # mainAppWindow.setCentralWidget(area)
    # mainAppWindow.resize(1000,500)
    # layout.addWidget(mainAppWindow.area, 0, 0) # start at 1 since the console is available at 0
    layout.addWidget(mainAppWindow.area) # start at 1 since the console is available at 0

    
    ## Create docks, place them into the window one at a time.
    ## Note that size arguments are only a suggestion; docks will still have to
    ## fill the entire dock area and obey the limits of their internal widgets.
    d1 = Dock("GUI Layout Controls", size=(1, 1))     ## give this dock the minimum possible size
    d2 = Dock("Display Outputs", size=(500,300), closable=True)
    d3 = Dock("Flowchart Configuration Widgets", size=(500,400))
    d4 = Dock("Flowchart", size=(500,200))
    # d5 = Dock("Dock5 - Image", size=(500,200))
    # d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
    mainAppWindow.area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
    mainAppWindow.area.addDock(d2, 'right')     ## place d2 at right edge of dock area
    mainAppWindow.area.addDock(d3, 'bottom', d1)## place d3 at bottom edge of d1
    mainAppWindow.area.addDock(d4, 'right')     ## place d4 at right edge of dock area
    # mainAppWindow.area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
    # mainAppWindow.area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

    # ## Test ability to move docks programatically after they have been placed
    # area.moveDock(d4, 'top', d2)     ## move d4 to top edge of d2
    # area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
    # area.moveDock(d5, 'top', d2)     ## move d5 to top edge of d2
    
    ## Add widgets into each dock
    # _build_dock_save_load(mainAppWindow.area, d1)
     ## first dock gets save/restore buttons
    w1 = pg.LayoutWidget()
    label = QtWidgets.QLabel(""" -- DockArea Example -- 
    This window has 6 Dock widgets in it. Each dock can be dragged
    by its title bar to occupy a different space within the window 
    but note that one dock has its title bar hidden). Additionally,
    the borders between docks may be dragged to resize. Docks that are dragged on top
    of one another are stacked in a tabbed layout. Double-click a dock title
    bar to place it in its own window.
    """)
    saveBtn = QtWidgets.QPushButton('Save dock state')
    restoreBtn = QtWidgets.QPushButton('Restore dock state')
    restoreBtn.setEnabled(False)
    w1.addWidget(label, row=0, col=0)
    w1.addWidget(saveBtn, row=1, col=0)
    w1.addWidget(restoreBtn, row=2, col=0)
    d1.addWidget(w1)
    state = None
    def save():
        global state
        state = mainAppWindow.area.saveState()
        restoreBtn.setEnabled(True)
    def load():
        global state
        mainAppWindow.area.restoreState(state)
    saveBtn.clicked.connect(save)
    restoreBtn.clicked.connect(load)
    
    ## Create an empty flowchart with a single input and output
    mainAppWindow.flowchart = Flowchart(terminals={
        'dataIn': {'io': 'in'},
        'dataOut': {'io': 'out'}    
    })
    
    # Add the flowchart widget. This is actually not the programmatic programming environment itself, it's the column that lists the nodes and lets you set their parameters.
    flowchart_controls_widget = mainAppWindow.flowchart.widget() 
    # layout.addWidget(mainAppWindow.flowchart.widget(), 0, 0, 2, 1) # spans 2 rows and 1 column
    d3.addWidget(flowchart_controls_widget)
    
    
    ## Result/Visualization Widgets:    
    # _setup_console(mainAppWindow)
    # _build_dynamic_results_widgets(mainAppWindow, layout)
    
    ## Create a container to hold all dynamically added widgets.
    new_dynamic_node_view_container_widget = QtGui.QWidget()
    w2 = pg.LayoutWidget()
    w2.addWidget(new_dynamic_node_view_container_widget, row=0, col=0)
    d2.addWidget(w2)
    
    # create a layout for the new container view:
    new_wrapper_container_layout = QtGui.QVBoxLayout()
    new_dynamic_node_view_container_widget.setLayout(new_wrapper_container_layout)
    
    # Get the flowchart window which displays the actual flowchart:
    flowchart_window = flowchart_controls_widget.cwWin
    d4.addWidget(flowchart_window)
    
    
    # Setup the nodes in the flowchart:
    _register_custom_node_types(mainAppWindow.flowchart)
    
    # end node setup:
    mainAppWindow.show()
    mainAppWindow.resize(1920, 1080)
    
    _add_pho_pipeline_programmatic_flowchart_nodes(mainAppWindow.app, mainAppWindow.flowchart, new_wrapper_container_layout) # changed from layout to new_wrapper_container_layout
    # _add_default_example_programmatic_flowchart_nodes(fc, layout)    

    # Expand all pipeline widget items on startup:
    # flowchart_controls_widget.ui.ctrlList is a TreeWidget
    flowchart_controls_widget.ui.ctrlList.expandAll()

    return mainAppWindow, mainAppWindow.app


def _setup_console(mainAppWindow):
    ## build an initial namespace for console commands to be executed in (this is optional;
    ## the user can always import these modules manually)
    namespace = {'pg': pg, 'np': np}

    ## initial text to display in the console
    text = """
    This is an interactive python console. The numpy and pyqtgraph modules have already been imported 
    as 'np' and 'pg'. 

    Go, play.
    """
    # console_widget = ConsoleWidget(namespace=namespace, text=text)
    # layout.addWidget(console_widget, 0, 1)
    mainAppWindow.console.localNamespace = namespace
    mainAppWindow.console.text = text
    
    
def _build_dynamic_results_widgets(mainAppWindow, layout):
    ## Create a container to hold all dynamically added widgets.
    new_dynamic_node_view_container_widget = QtGui.QWidget()
    layout.addWidget(new_dynamic_node_view_container_widget, 0, 0)
    # create a layout for the new container view:
    new_wrapper_container_layout = QtGui.QVBoxLayout()
    new_dynamic_node_view_container_widget.setLayout(new_wrapper_container_layout)
    return new_dynamic_node_view_container_widget
    
    
def _build_dock_area(mainAppWindow, layout):
    area = DockArea()
    # mainAppWindow.setCentralWidget(area)
    # mainAppWindow.resize(1000,500)
    layout.addWidget(area, 0, 0) # start at 1 since the console is available at 0
    # layout.addWidget(area, 1, 1, 2, 4) # start at 1 since the console is available at 0
    return area

def _build_dock_save_load(area, d1):
    ## first dock gets save/restore buttons
    w1 = pg.LayoutWidget()
    label = QtWidgets.QLabel(""" -- DockArea Example -- 
    This window has 6 Dock widgets in it. Each dock can be dragged
    by its title bar to occupy a different space within the window 
    but note that one dock has its title bar hidden). Additionally,
    the borders between docks may be dragged to resize. Docks that are dragged on top
    of one another are stacked in a tabbed layout. Double-click a dock title
    bar to place it in its own window.
    """)
    saveBtn = QtWidgets.QPushButton('Save dock state')
    restoreBtn = QtWidgets.QPushButton('Restore dock state')
    restoreBtn.setEnabled(False)
    w1.addWidget(label, row=0, col=0)
    w1.addWidget(saveBtn, row=1, col=0)
    w1.addWidget(restoreBtn, row=2, col=0)
    d1.addWidget(w1)
    state = None
    def save():
        global state
        state = area.saveState()
        restoreBtn.setEnabled(True)
    def load():
        global state
        area.restoreState(state)
    saveBtn.clicked.connect(save)
    restoreBtn.clicked.connect(load)




def _register_custom_node_types(fc):
    """Register Custom Nodes so they appear in the flowchart context menu"""
    ## Method 1: Register to global default library:
    #fclib.registerNodeType(ImageViewNode, [('Display',)])
    #fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

    ## Method 2: If we want to make our custom node available only to this flowchart,
    ## then instead of registering the node type globally, we can create a new 
    ## NodeLibrary:
    # library = fclib.LIBRARY.copy() # start with the default node set
    library = ReloadableNodeLibrary.from_node_library(fclib.LIBRARY.copy())  # start with the default node set
    
    
    library.addNodeType(ImageViewNode, [('Display',)])
    # Add the unsharp mask node to two locations in the menu to demonstrate
    # that we can create arbitrary menu structures
    library.addNodeType(UnsharpMaskNode, [('Image',)])
    
    # Custom Nodes:
    library.addNodeType(PhoPythonEvalNode, [('Data',), 
                                        ('Pho Pipeline','Eval')])
        
    
    # Pipeline Nodes:
    library.addNodeType(PipelineInputDataNode, [('Data',), 
                                        ('Pho Pipeline','Input')])
    library.addNodeType(PipelineFilteringDataNode, [('Filters',), 
                                        ('Pho Pipeline','Filtering')])
    library.addNodeType(PipelineComputationsNode, [('Data',), 
                                        ('Pho Pipeline','Computation')])
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
        
    def on_add_widget_fn(show_in_separate_window=True):
        """ uses layout implicitly """
        # Matplotlib widget directly:
        new_view_widget = MatplotlibWidget()
        if show_in_separate_window:
            new_widget_window = PhoPipelineSecondaryWindow([new_view_widget])
            new_widget_window.setWindowTitle(f'PhoFlowchartApp: Custom Result Window')
            new_widget_window.show()
            new_widget_window.resize(800,600)
        else:
            new_widget_window = None # no window created
            layout.addWidget(new_view_widget) # now assumes layout is a QVBoxLayout
            # layout.addWidget(new_view_widget, 1, 1) # start at 1 since the console is available at 0
        
        # add example plot to figure
        subplot = new_view_widget.getFigure().add_subplot(111)
        subplot.plot(np.arange(9))
        new_view_widget.draw()
        
        return new_view_widget, new_widget_window
    
        

    ## Result/Visualization Widgets:
    # need app and win
    # new_view_widget = pg.GraphicsWidget()
    
    # # Build the new outer container widget to hold the other views:
    # new_view_widget = QtGui.QWidget()
    # layout.addWidget(new_view_widget, 1, 1) # start at 1 since the console is available at 0
    
    # # create a layout for the new container view:
    # new_view_layout = QtGui.QGridLayout()
    # new_view_widget.setLayout(new_view_layout)
    # # build the internal widget
    # new_root_render_widget = pg.GraphicsLayoutWidget()
    # new_view_layout.addWidget(new_root_render_widget, 1, 1) # add the new view to the new layout
    
    # New Window:
    ## Create main window with a grid layout inside
    # win = QtGui.QMainWindow()
    # win.setWindowTitle(f'PhoFlowchartApp: Custom Result Window')
    # cw = QtGui.QWidget()
    # win.setCentralWidget(cw)
    # # layout = QtGui.QGridLayout()
    # layout = QtGui.QVBoxLayout()
    # cw.setLayout(layout)
    
        
    # # Matplotlib widget directly:
    # new_view_widget = MatplotlibWidget()
    # # layout.addWidget(new_view_widget, 1, 1) # start at 1 since the console is available at 0
    
    # new_widget_window = AnotherWindow([new_view_widget])
    # new_widget_window.setWindowTitle(f'PhoFlowchartApp: Custom Result Window')
    # new_widget_window.show()
    # new_widget_window.resize(800,600)
    
    # subplot = new_view_widget.getFigure().add_subplot(111)
    # subplot.plot(np.arange(9))
    # new_view_widget.draw()
    
    # new_view_widget.setCentralWidget(new_root_render_widget)
        
    
    ## Create two ImageView widgets to display the raw and processed data with contrast
    ## and color control.
    # v1 = pg.ImageView()
    # v2 = pg.ImageView()
    # layout.addWidget(v1, 1, 1) # start at 1 since the console is available at 0
    # layout.addWidget(v2, 2, 1)
    
    
    ## Set the raw data as the input value to the flowchart
    fc.setInput(dataIn='Bapun')
    
    pipeline_input_node = fc.createNode('PipelineInputDataNode', pos=(-300, 50))
    # pipeline_input_node.setView(v1, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    
    pipeline_filter_node = fc.createNode('PipelineFilteringDataNode', pos=(-26, 50))
    # pipeline_filter_node.setView(v2, on_remove_function=on_remove_widget_fn)
    
    pipeline_computation_node = fc.createNode('PipelineComputationsNode', pos=(154, 50))
    

    pipeline_display_node = fc.createNode('PipelineDisplayNode', pos=(280, 120))
    pipeline_display_node.setApp(app) # Sets the shared singleton app instance
    # pipeline_display_node.setView(new_root_render_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the 
    # for direct matploblib widget mode:
    # pipeline_display_node.setView(new_view_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    # dynamic widget building mode:
    pipeline_display_node.setView(on_add_function=on_add_widget_fn, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    
    # Setup connections:
    fc.connectTerminals(fc['dataIn'], pipeline_input_node['known_mode'])
    
    # Input Node Outputs:
    fc.connectTerminals(pipeline_input_node['loaded_pipeline'], pipeline_filter_node['pipeline'])
    fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_filter_node['active_data_mode'])
    
    fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_display_node['mode'])
    
    # Filter Node Outputs:
    fc.connectTerminals(pipeline_filter_node['filtered_pipeline'], pipeline_computation_node['pipeline'])
    fc.connectTerminals(pipeline_filter_node['computation_configs'], pipeline_computation_node['computation_configs'])
    fc.connectTerminals(pipeline_filter_node['filter_configs'], pipeline_display_node['filter_configs'])
    
    # Computation Node Outputs:
    fc.connectTerminals(pipeline_computation_node['computed_pipeline'], pipeline_display_node['pipeline'])
    fc.connectTerminals(pipeline_computation_node['updated_computation_configs'], pipeline_display_node['computation_configs'])

    fc.connectTerminals(pipeline_computation_node['computed_pipeline'], fc['dataOut']) # raw pipeline output from computation node
    
    
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
