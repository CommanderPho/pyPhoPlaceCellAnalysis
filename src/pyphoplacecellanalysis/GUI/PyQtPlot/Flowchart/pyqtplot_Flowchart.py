from pyphoplacecellanalysis.External.pyqtgraph.flowchart import Flowchart, Node
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.ReloadableNodeLibrary import ReloadableNodeLibrary
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets

# Must be called before any figures are created:
import matplotlib
matplotlib.use('qtagg')

from pyphoplacecellanalysis.External.pyqtgraph.console import ConsoleWidget
from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea


import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.PhoPipelineMainWindow.pyqtplot_MainWindow import PhoPipelineMainWindow
from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.DisplayNodeViewHelpers import ProducedViewType


from qtpy import QtWidgets
from qtpy.QtCore import QFile, QTextStream
import pyphoplacecellanalysis.External.breeze_style_sheets.breeze_resources


def plot_flowchartWidget(title='PhoFlowchartApp'):
    """ 
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget
        pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp')
    """
    
    # Use the widget defined in the designer as the central widget   
    mainAppWindow = PhoPipelineMainWindow(title)
    mainAppWindow.setWindowTitle(f'PhoFlowchartApp: pyqtgraph FlowchartCustomNodes: {title}')
    
    # get central widget:
    cw = mainAppWindow.centralwidget
    print(f'cw: {cw}')
    
    # setup the main layout of the central widget:
    layout = QtGui.QVBoxLayout()
    cw.setLayout(layout)
    mainAppWindow.area = DockArea()
    layout.addWidget(mainAppWindow.area) # start at 1 since the console is available at 0

    ## Create docks, place them into the window one at a time.
    ## Note that size arguments are only a suggestion; docks will still have to
    ## fill the entire dock area and obey the limits of their internal widgets.
    d1 = Dock("GUI Layout Controls", size=(1, 1))     ## give this dock the minimum possible size
    # d2 = Dock("Display Outputs", size=(500,300), closable=True)
    d3 = Dock("Flowchart Configuration Widgets", size=(500,400))
    d4 = Dock("Flowchart", size=(500,200))
    # d5 = Dock("Dock5 - Image", size=(500,200))
    # d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
    mainAppWindow.area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
    # mainAppWindow.area.addDock(d2, 'right')     ## place d2 at right edge of dock area
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
    d3.addWidget(flowchart_controls_widget)
    
    ## Result/Visualization Widgets:    
    # _setup_console(mainAppWindow)
    # _build_dynamic_results_widgets(mainAppWindow, layout)
    
    # Old static widget way:
    # new_dynamic_node_view_container_widget, new_wrapper_container_layout = _build_static_display_widget(mainAppWindow)
    
    # New nested Dock area widget way:
    dItem = mainAppWindow._build_dynamic_display_dockarea()
    
    display_dock_area = mainAppWindow.displayDockArea
    
    # Get the flowchart window which displays the actual flowchart:
    flowchart_window = flowchart_controls_widget.cwWin
    d4.addWidget(flowchart_window)
    
    ## Define the dynamic add/remove functions for the display dock widgets:
    def on_remove_widget_fn(identifier):
        """ uses mainAppWindow implicitly. the callback to remove the widgets with the group identifier 'identifier' from the dock areas
        """
        print(f'on_remove_widget_fn({identifier})')
        return mainAppWindow.remove_display_dock(identifier)

    def on_add_widget_fn(identifier, viewContentsType: ProducedViewType):
        """ uses mainAppWindow implicitly """
        # Add the sample display dock items to the nested dynamic display dock:
        print(f'on_add_widget_fn({identifier})')
        new_view_widget, dDisplayItem = mainAppWindow.add_display_dock(identifier=identifier, viewContentsType=viewContentsType)
        return new_view_widget, dDisplayItem
    
    # Setup the nodes in the flowchart:
    ReloadableNodeLibrary.setup_custom_node_library(mainAppWindow.flowchart)
    # _setup_custom_node_library(mainAppWindow.flowchart)
    
    # end node setup:
    mainAppWindow.show()
    mainAppWindow.resize(1920, 1080)
    
    # _add_pho_pipeline_programmatic_flowchart_nodes(mainAppWindow.app, mainAppWindow.flowchart, new_wrapper_container_layout) # changed from layout to new_wrapper_container_layout
    _add_pho_pipeline_programmatic_flowchart_nodes(mainAppWindow.app, mainAppWindow.flowchart, on_add_function=on_add_widget_fn, on_remove_function=on_remove_widget_fn) # dynamic dockarea version
    
    # _add_default_example_programmatic_flowchart_nodes(fc, layout)    

    # Expand all pipeline widget items on startup:
    # flowchart_controls_widget.ui.ctrlList is a TreeWidget
    flowchart_controls_widget.ui.ctrlList.expandAll()

    return mainAppWindow, mainAppWindow.app





def _build_dynamic_display_dockarea(mainAppWindow):
    dItem = Dock("Display Outputs - Dynamic", size=(600,900), closable=True)
    mainAppWindow.area.addDock(dItem, 'right')
    mainAppWindow.displayDockArea = DockArea()
    dItem.addWidget(mainAppWindow.displayDockArea) # add the dynamic nested Dock area to the dItem widget
    return dItem
    
def _build_static_display_widget(mainAppWindow):
    d2 = Dock("Display Outputs", size=(500,300), closable=True)
    mainAppWindow.area.addDock(d2, 'right')     ## place d2 at right edge of dock area
    ## Create a container to hold all dynamically added widgets.
    new_dynamic_node_view_container_widget = QtGui.QWidget()
    w2 = pg.LayoutWidget()
    w2.addWidget(new_dynamic_node_view_container_widget, row=0, col=0)
    d2.addWidget(w2)
    
    # create a layout for the new container view:
    new_wrapper_container_layout = QtGui.QVBoxLayout()
    new_dynamic_node_view_container_widget.setLayout(new_wrapper_container_layout)
    return new_dynamic_node_view_container_widget, new_wrapper_container_layout

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





def _add_pho_pipeline_programmatic_flowchart_nodes(app, fc, on_add_function=None, on_remove_function=None):
    ## Now we will programmatically add nodes to define the function of the flowchart.
    ## Normally, the user will do this manually or by loading a pre-generated
    ## flowchart file.
    """[summary]
    Args:
        fc ([type]): [description]
        layout ([type]): a grid layout to add result/visualization widgets to. 
    """

    # def on_remove_widget_fn(widget):
    #     """ the callback to remove the widget from the layout.
    #         implicitly used 'layout'.
    #     """
    #     item_index = layout.indexOf(widget)
    #     print(f'on_remove_widget_fn(...): item_index: {item_index}')
    #     item = layout.itemAt(item_index)
    #     widget = item.widget() # this should be the same as the passed in widget, but do this just to be sure
    #     layout.removeWidget(widget)
        
    # def on_add_widget_fn(show_in_separate_window=True):
    #     """ uses layout implicitly """
    #     # Matplotlib widget directly:
    #     new_view_widget = MatplotlibWidget()
    #     if show_in_separate_window:
    #         new_widget_window = PhoPipelineSecondaryWindow([new_view_widget])
    #         new_widget_window.setWindowTitle(f'PhoFlowchartApp: Custom Result Window')
    #         new_widget_window.show()
    #         new_widget_window.resize(800,600)
    #     else:
    #         new_widget_window = None # no window created
    #         layout.addWidget(new_view_widget) # now assumes layout is a QVBoxLayout
    #         # layout.addWidget(new_view_widget, 1, 1) # start at 1 since the console is available at 0
        
    #     # add example plot to figure
    #     subplot = new_view_widget.getFigure().add_subplot(111)
    #     subplot.plot(np.arange(9))
    #     new_view_widget.draw()
        
    #     return new_view_widget, new_widget_window
    
    # pipeline_start_x = -400
    pipeline_start_x = 500
    
    ## Set the raw data as the input value to the flowchart
    # fc.setInput(dataIn='Bapun')
    # fc.setInput(dataIn='kdiba')
    fc.setInput(dataIn=None)

    pipeline_input_node = fc.createNode('PipelineInputDataNode', pos=(pipeline_start_x-400, 50))
    # pipeline_input_node.setView(v1, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    
    pipeline_filter_node = fc.createNode('PipelineFilteringDataNode', pos=(pipeline_start_x-26, 50))
    # pipeline_filter_node.setView(v2, on_remove_function=on_remove_widget_fn)
    
    pipeline_computation_node = fc.createNode('PipelineComputationsNode', pos=(pipeline_start_x+154, 50))
    
    pipeline_display_node = fc.createNode('PipelineDisplayNode', pos=(pipeline_start_x+280, 120))
    # pipeline_display_node.setApp(app) # Sets the shared singleton app instance
    # pipeline_display_node.setView(new_root_render_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the 
    # for direct matploblib widget mode:
    # pipeline_display_node.setView(new_view_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    # dynamic widget building mode:
    # pipeline_display_node.setView(on_add_function=on_add_widget_fn, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    # pipeline_display_node.setView(on_add_function=on_add_function, on_remove_function=on_remove_function) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    
    # Pipeline Result Visualization Node:
    pipeline_result_viz_node = fc.createNode('PipelineResultVisNode', pos=(pipeline_start_x+280, 220))
    pipeline_result_viz_node.on_add_function = on_add_function
    pipeline_result_viz_node.on_remove_function = on_remove_function
    
    # Setup connections:
    fc.connectTerminals(fc['dataIn'], pipeline_input_node['known_mode'])
    
    # Input Node Outputs:
    fc.connectTerminals(pipeline_input_node['loaded_pipeline'], pipeline_filter_node['pipeline'])
    fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_filter_node['active_data_mode'])
    
    # fc.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_display_node['mode'])
    
    # Filter Node Outputs:
    fc.connectTerminals(pipeline_filter_node['filtered_pipeline'], pipeline_computation_node['pipeline'])
    fc.connectTerminals(pipeline_filter_node['computation_configs'], pipeline_computation_node['computation_configs'])
    # fc.connectTerminals(pipeline_filter_node['filter_configs'], pipeline_display_node['filter_configs'])
    
    # Computation Node Outputs:
    fc.connectTerminals(pipeline_computation_node['computed_pipeline'], pipeline_display_node['pipeline'])
    fc.connectTerminals(pipeline_computation_node['updated_computation_configs'], pipeline_display_node['computation_configs'])

    fc.connectTerminals(pipeline_computation_node['computed_pipeline'], pipeline_result_viz_node['pipeline'])
    fc.connectTerminals(pipeline_computation_node['updated_computation_configs'], pipeline_result_viz_node['computation_configs'])
    
    fc.connectTerminals(pipeline_computation_node['computed_pipeline'], fc['dataOut']) # raw pipeline output from computation node
    
    # fc.setInput(dataIn='kdiba') # finally set the input data
    
    
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
