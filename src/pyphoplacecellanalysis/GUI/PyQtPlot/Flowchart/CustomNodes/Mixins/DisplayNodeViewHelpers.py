# DisplayNodeViewHelpers.py
from enum import Enum
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import pyqtgraph as pg
import numpy as np


from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow



class ProducedViewType(Enum):
	"""Docstring for ProducedViewType."""
	Matplotlib = "Matplotlib"
	Custom = "Custom"
	
 
 
 
 
class DisplayNodeViewHelpers:
    """ Display node is instantiated like so:
    
    pipeline_display_node = fc.createNode('PipelineDisplayNode', pos=(280, 120))
    pipeline_display_node.setApp(app) # Sets the shared singleton app instance
    # pipeline_display_node.setView(new_root_render_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the 
    # for direct matploblib widget mode:
    # pipeline_display_node.setView(new_view_widget, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    # dynamic widget building mode:
    pipeline_display_node.setView(on_add_function=on_add_widget_fn, on_remove_function=on_remove_widget_fn) # Sets the view associated with the node. Note that this is the programmatically instantiated node
    
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
    
    