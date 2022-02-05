import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np


# from PyQt6 import QtWidgets, uic
from PyQt5 import QtWidgets, uic

from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow

# Import the custom nodes:
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.ImageViewNode import ImageViewNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.UnsharpMaskNode import UnsharpMaskNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineInputDataNode import PipelineInputDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineFilteringDataNode import PipelineFilteringDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.PipelineDisplayNode import PipelineDisplayNode


path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'MainPipelineWindow.ui')


class PhoPipelineMainWindow(QtWidgets.QMainWindow):

    @property
    def flowchart(self):
        """The flowchart property."""
        return self._flowchart
    @flowchart.setter
    def flowchart(self, value):
        self._flowchart = value
        
    @property
    def app(self):
        """The app property."""
        return self._app
    
    @property
    def title(self):
        """The title property."""
        return self._title



    def __init__(self, title='PhoFlowchartApp', *args, **kwargs):
        super(PhoPipelineMainWindow, self).__init__(*args, **kwargs)

		#Load the UI Page
        uic.loadUi(uiFile, self)
        
        self._title = title
        self._app = pg.mkQApp(self._title)
            
        # self.graphWidget = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget)

        # hour = [1,2,3,4,5,6,7,8,9,10]
        # temperature = [30,32,34,32,33,31,29,32,35,45]

        # # plot data: x, y values
        # self.graphWidget.plot(hour, temperature)

    def _initialize_data(self):
        self._flowchart = None
        self._pipeline = None
        
        # ## later on, process data through the node
        # filteredData = filterNode.process(inputTerminal=rawData)
        
    
    def build_flowchart(self):
        """ 

        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget
            pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp')
        """
        
        def _register_custom_node_types():
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
            self.flowchart.setLibrary(library)
            
            
        
        self.setWindowTitle(f'PhoFlowchartApp: pyqtgraph FlowchartCustomNodes: {self.title}')
        layout = QtGui.QGridLayout()
        self.flowchart_controls.setLayout(layout)

        ## Create an empty flowchart with a single input and output
        self.flowchart = Flowchart(terminals={
            'dataIn': {'io': 'in'},
            'dataOut': {'io': 'out'}    
        })
    
        # Add the flowchart widget. This is actually not the programmatic programming environment itself, it's the column that lists the nodes and lets you set their parameters.
        layout.addWidget(self.flowchart.widget(), 0, 0, 2, 1) # spans 2 rows and 1 column

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
        self.console.localNamespace = namespace
        self.console.text = text

        # console_widget = ConsoleWidget(namespace=namespace, text=text)
        # layout.addWidget(console_widget, 0, 1)
        
        ## Create a container to hold all dynamically added widgets.
        new_dynamic_node_view_container_widget = QtGui.QWidget()
        layout.addWidget(new_dynamic_node_view_container_widget, 1, 1) # start at 1 since the console is available at 0
        # create a layout for the new container view:
        new_wrapper_container_layout = QtGui.QVBoxLayout()
        new_dynamic_node_view_container_widget.setLayout(new_wrapper_container_layout)
        
        # mw = MatplotlibWidget()
        # subplot = mw.getFigure().add_subplot(111)
        # subplot.plot(x,y)
        # mw.draw()
        # # win.show()

        _register_custom_node_types(self.flowchart)
        
        self._add_pho_pipeline_programmatic_flowchart_nodes(self.app, self.flowchart, new_wrapper_container_layout) # changed from layout to new_wrapper_container_layout
        # _add_default_example_programmatic_flowchart_nodes(self.flowchart, layout)   
        
        
        
        
            
        
        
        def _add_pho_pipeline_programmatic_flowchart_nodes(self, layout):
            ## Now we will programmatically add nodes to define the function of the flowchart.
            ## Normally, the user will do this manually or by loading a pre-generated
            ## flowchart file.
            """[summary]

            Args:
                self.flowchart ([type]): [description]
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
            self.flowchart.setInput(dataIn='Bapun')
            
            pipeline_input_node = self.flowchart.createNode('PipelineInputDataNode', pos=(-200, 50))
            # pipeline_input_node.setView(v1, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
            
            pipeline_filter_node = self.flowchart.createNode('PipelineFilteringDataNode', pos=(-26, 50))
            # pipeline_filter_node.setView(v2, on_remove_function=on_remove_widget_fn)

            pipeline_display_node = self.flowchart.createNode('PipelineDisplayNode', pos=(154, 20))
            pipeline_display_node.setApp(self.app) # Sets the shared singleton app instance
            # pipeline_display_node.setView(new_root_render_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the 
            
            # for direct matploblib widget mode:
            # pipeline_display_node.setView(new_view_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
            # dynamic widget building mode:
            pipeline_display_node.setView(on_add_function=on_add_widget_fn, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
            
            # Setup connections:
            self.flowchart.connectTerminals(self.flowchart['dataIn'], pipeline_input_node['known_mode'])
            
            # Input Node Outputs:
            self.flowchart.connectTerminals(pipeline_input_node['loaded_pipeline'], pipeline_filter_node['pipeline'])
            self.flowchart.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_filter_node['active_data_mode'])
            
            self.flowchart.connectTerminals(pipeline_input_node['known_data_mode'], pipeline_display_node['active_data_mode'])
            
            # Computation Node Outputs:
            self.flowchart.connectTerminals(pipeline_filter_node['filtered_pipeline'], pipeline_display_node['active_pipeline'])
            self.flowchart.connectTerminals(pipeline_filter_node['computation_configs'], pipeline_display_node['active_session_computation_configs'])
            self.flowchart.connectTerminals(pipeline_filter_node['filter_configurations'], pipeline_display_node['active_session_filter_configurations'])

            self.flowchart.connectTerminals(pipeline_filter_node['filtered_pipeline'], self.flowchart['dataOut']) # raw pipeline output from computation node
            
            # Display Node Outputs:   
        
            
            

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = PhoPipelineMainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()