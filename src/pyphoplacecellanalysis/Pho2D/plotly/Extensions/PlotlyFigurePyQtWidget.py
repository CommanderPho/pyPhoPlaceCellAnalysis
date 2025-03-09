from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import pyphoplacecellanalysis.External.pyqtgraph as pg
import sys
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

@metadata_attributes(short_name=None, tags=['pyqt5', 'widget', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-09 15:48', related_items=[])
class PlotlyWidget(QtWidgets.QWidget):
    """Custom widget to display Plotly figures in PyQt applications
    
    IMPORTANT NOTE: `QWebEngineView` must be imported prior to any QApplication instance being created, meaning in a jupyter network it must come before %qt5 mmagic
    
		import os
		os.environ['QT_API'] = 'pyqt5'
		os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

		from PyQt5.QtWebEngineWidgets import QWebEngineView ## this must come first, before any QtApplication is made: 'ImportError: QtWebEngineWidgets must be imported or Qt.AA_ShareOpenGLContexts must be set before a QCoreApplication instance is created'

		# required to enable non-blocking interaction:
		%gui qt5


    """
    
    def __init__(self, figure=None, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        
        if figure is not None:
            self.set_figure(figure)
    
    def set_figure(self, figure):
        """Set the plotly figure to display"""
        html = figure.to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html)
    
    def clear(self):
        """Clear the figure"""
        self.web_view.setHtml("")



# ==================================================================================================================== #
# PlotlyDockContainer - A DockArea containing multiple PlotlyWidgets                                                   #
# ==================================================================================================================== #
@metadata_attributes(short_name=None, tags=['plotly', 'pyqt5', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-09 15:48', related_items=[])
class PlotlyDockContainer(QtWidgets.QWidget):
    """Container for multiple Plotly figures displayed in dockable widgets

    Usage:
    
		from pyphoplacecellanalysis.Pho2D.plotly.Extensions.PlotlyFigurePyQtWidget import PlotlyDockContainer, PlotlyWidget
        
        ## INPUTS: _flat_out_figs_dict
		# Create the container
		container = PlotlyDockContainer()
		# Display all figures in the dictionary
		for a_fig_context, fig in _flat_out_figs_dict.items():
			container.add_figure(fig, name=f"{a_fig_context}", position='bottom')

		# Show the container
		container.resize(1200, 800)
		container.show()
        
    Example:    
		
		# Create the container
		container = PlotlyDockContainer()

		# Create sample figures
		fig1 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))
		fig1.update_layout(title="Figure 1")

		fig2 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[2, 5, 3]))
		fig2.update_layout(title="Figure 2")

		fig3 = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
		fig3.update_layout(title="Figure 3")

		# Add figures to docks
		dock1_name = container.add_figure(fig1, name="Scatter 1")
		dock2_name = container.add_figure(fig2, name="Scatter 2", position='right', relativeTo=dock1_name)
		container.add_figure(fig3, name="Bar Chart", position='bottom')

		# Show the container
		container.resize(1200, 800)
		container.show()

    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create dock area
        self.dock_area = DockArea()
        self.layout.addWidget(self.dock_area)
        
        # Track docks and widgets
        self.docks = {}
        self.plotly_widgets = {}
        
    def add_figure(self, fig, name=None, position='bottom', relativeTo=None):
        """Add a plotly figure to a new dock
        
        Args:
            fig (plotly.graph_objects.Figure): Plotly figure to display
            name (str, optional): Name for the dock. Defaults to "Figure N".
            position (str, optional): Position to place dock ('left','right','top','bottom'). Defaults to 'bottom'.
            relativeTo (str, optional): Name of dock to position relative to. Defaults to None.
        
        Returns:
            str: Name of the created dock
        """
        # Generate a name if not provided
        if name is None:
            name = f"Figure {len(self.docks) + 1}"
        
        # Create a new dock
        dock = Dock(name, size=(500, 400))
        self.docks[name] = dock
        
        # Create a PlotlyWidget to display the figure
        plotly_widget = PlotlyWidget(fig)
        self.plotly_widgets[name] = plotly_widget
        
        # Add the widget to the dock
        dock.addWidget(plotly_widget)
        
        # Add the dock to the dock area
        if len(self.docks) == 1:
            # First dock, just add it
            self.dock_area.addDock(dock)
        else:
            # Position relative to existing docks
            ref_dock = self.docks[relativeTo] if relativeTo in self.docks else None
            self.dock_area.addDock(dock, position, ref_dock)
        
        return name
    
    def update_figure(self, name, fig):
        """Update an existing figure
        
        Args:
            name (str): Name of the dock/figure to update
            fig (plotly.graph_objects.Figure): New plotly figure
        """
        if name in self.plotly_widgets:
            self.plotly_widgets[name].set_figure(fig)
    
    def clear_figures(self):
        """Remove all figures"""
        for widget in self.plotly_widgets.values():
            widget.clear()
        
        # Remove all docks
        for dock in self.docks.values():
            dock.close()
        
        self.docks = {}
        self.plotly_widgets = {}




# Usage example
if __name__ == "__main__":
    """ 
	from pyphoplacecellanalysis.Pho2D.plotly.Extensions.PlotlyFigurePyQtWidget import PlotlyWidget


	"""
    app = pg.mkQApp('Test')
    
    # Create a sample figure
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    
    # Create and show the window
    window = PlotlyWidget(fig)
    window.show()
    
    sys.exit(app.exec_())
    
