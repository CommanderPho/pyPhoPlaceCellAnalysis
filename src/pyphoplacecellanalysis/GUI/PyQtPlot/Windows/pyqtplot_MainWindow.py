# from PyQt6 import QtWidgets, uic
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


    def __init__(self, title='PhoFlowchartApp', *args, **kwargs):
        self._app = pg.mkQApp(title)
        self._initialize_data()
        
        super(PhoPipelineMainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi(uiFile, self)
        


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
        


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = PhoPipelineMainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()