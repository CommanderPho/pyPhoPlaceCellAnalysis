# DecoderPlotSelectorWidget.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\DecoderPlotSelectorWidget\DecoderPlotSelectorWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

## Matplotlib Imports:
import matplotlib as mpl
mpl.use('Qt5Agg')
# For Matplotlib figures in the GUI:
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer



## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget

from pyphoplacecellanalysis.GUI.Qt.Widgets.DecoderPlotSelectorControls.Uic_AUTOGEN_DecoderPlotSelectorWidget import Ui_Form
# from .Uic_AUTOGEN_DecoderPlotSelectorWidget import Ui_Form

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')


class DecoderPlotSelectorWidget(QtWidgets.QWidget):
    """ Not quite finished

    Usage:    
        from pyphoplacecellanalysis.GUI.Qt.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget

        widget = DecoderPlotSelectorWidget()
        widget.show()

    """
    def __init__(self, name='DecoderPlotSelectorWidget', parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method

        self.ui = Ui_Form()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.plots_data = RenderPlotsData(name=name)
        self.plots = RenderPlots(name=name)
        
        # Final UI Refinements:
        self.initUI()
        self.show() # Show the GUI


    @property
    def variable_name(self):
        """The variable_name property."""
        return self.ui.cmbVariableName.currentText()
    # @variable_name.setter
    # def variable_name(self, value):
    #     self.ui.cmbVariableName.currentText = value

    @property
    def decoder_name(self):
        """The decoder_name property."""
        return self.ui.cmbDecoder.currentText()
    # @decoder_name.setter
    # def decoder_name(self, value):
    #     self._decoder_name = value
    
    def initUI(self):
        """ 
        cmbDecoder
        cmbVariableName
        decoderPlotContainerWidget # widget that the actual plot will be contained in

        """
        # Setup self.ui.chkbtnPlacefield:
        self.ui.cmbDecoder.currentTextChanged.connect(self.onSelectedDecoderNameChanged)
        self.ui.cmbVariableName.currentTextChanged.connect(self.onSelectedVariableNameChanged)
        
        
        # widget that the actual plot will be contained in
        # self.ui.decoderPlotContainerWidget 
        
        
        # self.ui.decoderPlotContainerWidget.layout().addWidget(self.resultplot_figureCanvas)
        # self.ui.fig_container_layout_widget = pg.LayoutWidget(parent=self.ui.decoderPlotContainerWidget)
        
        self.ui.root_plot_grid_layout = QtWidgets.QGridLayout()
        self.ui.decoderPlotContainerWidget.setLayout(self.ui.root_plot_grid_layout)
        
        # self._setup_matplotlib_mode()
        self._setup_pyqtgraph_mode()
        # self.ui.decoderPlotContainerWidget.setLayout()
        

        
        
    def _setup_matplotlib_mode(self):
        """ sets up the widget for matplot-style plotting """
        self.resultplot_figureCanvas = FigureCanvas(Figure(figsize=(15,15), constrained_layout=True))
        self.ui.fig_container_layout_widget.addWidget(self.resultplot_figureCanvas)
        fig = self.resultplot_figureCanvas.figure
        fig.clear()
        self.ax = fig.subplots(ncols=1, nrows=1)
        # Figure 'num' identifier: num=f'debug_two_step_animated: variable_name={self.variable_name}'
        
        
    def _setup_pyqtgraph_mode(self):
        """ sets up the widget for new pyqtgraph-style plotting """
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        # plotItem.invertY(True)           # orient y axis to run top-to-bottom
        # Normal ImageItem():
        # self.plots.imageItem = pg.ImageItem(matrix.T)
        self.ui.imv = pg.ImageView()
        self.ui.root_plot_grid_layout.addWidget(self.ui.imv, 0, 0)
        # self.ui.fig_container_layout_widget.addWidget(self.ui.imv, 0, 0)
        
        
        
    def render(self):
        pass


    def onSelectedDecoderNameChanged(self, decoder_name):
        print(f'onSelectedDecoderNameChanged(decoder_name: "{decoder_name}")')
                
    def onSelectedVariableNameChanged(self, variable_name):
        print(f'onSelectedVariableNameChanged(decoder_name: "{variable_name}")')

    def __str__(self):
         return 


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("DecoderPlotSelectorWidget Example")
    widget = DecoderPlotSelectorWidget()
    widget.show()
    pg.exec()
