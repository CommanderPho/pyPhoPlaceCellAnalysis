# from PyQt6 import QtWidgets, uic
from typing import OrderedDict
from PyQt5 import QtWidgets, uic

from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow

# Import the custom nodes:
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.ImageViewNode import ImageViewNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.UnsharpMaskNode import UnsharpMaskNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.BasePipeline.PipelineInputDataNode import PipelineInputDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.BasePipeline.PipelineFilteringDataNode import PipelineFilteringDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.PipelineDisplayNode import PipelineDisplayNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.DisplayNodeViewHelpers import PipelineDynamicDockDisplayAreaMixin


path = os.path.dirname(os.path.abspath(__file__))
# uiFile = os.path.join(path, 'MainPipelineWindow.ui')
uiFile = os.path.join(path, 'MainPipelineWindowWithDockArea.ui') # mostly empty

class PhoPipelineMainWindow(PipelineDynamicDockDisplayAreaMixin, QtWidgets.QMainWindow):
    
    @property
    def app(self):
        """The app property."""
        return self._app
    
    @property
    def flowchart(self):
        """The flowchart property."""
        return self._flowchart
    @flowchart.setter
    def flowchart(self, value):
        self._flowchart = value
        
    @property
    def flowchart_controls_widget(self):
        """ """
        return self.flowchart.widget() # FlowchartCtrlWidget

    @property
    def flowchart_controls_tree_widget(self):
        # flowchart_controls_widget.ui.ctrlList is a TreeWidget
        return self.flowchart_controls_widget.ui.ctrlList

    @property
    def flowchart_window(self):
        """ The window that the flowchart is displayed in. Not the one with the controls by default. """
        return self.flowchart_controls_widget.cwWin
    
    
    ## Specific Flowchart Nodes:
    @property
    def flowchart_nodes(self):
        """ 
        Example:
            {'Input': <Node Input @13fc5877a60>,
            'Output': <Node Output @13fc5877af0>,
            'PipelineInputDataNode.0': <Node PipelineInputDataNode.0 @13fcb1428b0>,
            'PipelineFilteringDataNode.0': <Node PipelineFilteringDataNode.0 @13fcb145ca0>,
            'PipelineComputationsNode.0': <Node PipelineComputationsNode.0 @13fcb14c1f0>,
            'PipelineDisplayNode.0': <Node PipelineDisplayNode.0 @13fcb14eb80>}
        """
        return self.flowchart.nodes()
    
    @property
    def flowchart_input_node(self):
        return self.flowchart_nodes.get('Input', None) # Node or None
    
    
    @property
    def flowchart_output_node(self):
        return self.flowchart_nodes.get('Output', None) # Node or None
    
    
    
    
    



# # curr_node = pipeline_flowchart_nodes['Input']
# # curr_node = pipeline_flowchart_nodes['PipelineFilteringDataNode.0']
# curr_node = pipeline_flowchart_nodes['PipelineComputationsNode.0']

# curr_check_table = curr_node.ctrls['included_configs_table']
# curr_check_table.checked_state # OrderedDict([('maze1', [False, False]), ('maze2', [False, False])])
# curr_check_table.checked_state['maze1']

# curr_check_table.saveState()



    def __init__(self, title='PhoFlowchartApp', *args, **kwargs):
        self._app = pg.mkQApp(title) # makes a new QApplication or gets the reference to an existing one.
        self._initialize_data()
        
        super(PhoPipelineMainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi(uiFile, self) # load from the ui file
        
        # Set window icon:
        

        # icon_path = 'Resources\Icons\ProcessIcon.ico'
        icon_path = r'C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Icons\ProcessIcon.ico'
        self.setWindowIcon(QtGui.QIcon(icon_path))
        
        # ':/Icons/Icons/ProcessIcon.ico' 

        # self.graphWidget = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget)

        # hour = [1,2,3,4,5,6,7,8,9,10]
        # temperature = [30,32,34,32,33,31,29,32,35,45]

        # # plot data: x, y values
        # self.graphWidget.plot(hour, temperature)

    def _initialize_data(self):
        self._flowchart = None
        self._pipeline = None
        
        
        self._dynamic_display_output_dict = OrderedDict() # for PipelineDynamicDockDisplayAreaMixin
        # ## later on, process data through the node
        # filteredData = filterNode.process(inputTerminal=rawData)
        
    
    def closeEvent(self, event):
        # Enables closing all secondary windows when this (main) window is closed.
        for window in QtWidgets.QApplication.topLevelWidgets():
            window.close()
            
            

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = PhoPipelineMainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()