# DisplayNodeViewHelpers.py
from collections import OrderedDict
from enum import Enum
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.console import ConsoleWidget
from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

# For 3D Plotter Windows:
from pyvistaqt import BackgroundPlotter
from pyvistaqt.plotting import MultiPlotter

# For Matplotlib Windows:
from pyphoplacecellanalysis.GUI.PyQtPlot.Windows.pyqtplot_SecondaryWindow import PhoPipelineSecondaryWindow



class ProducedViewType(Enum):
    """Docstring for ProducedViewType."""
    Matplotlib = 'Matplotlib' # MatplotlibWidget, needs to be passed into display function as "fig=active_fig" argument
    Pyvista = 'Pyvista' # BackgroundPlotter, MultiPlotter: needs to be passed into display function as "extant_plotter=active_plotter" argument
    Custom = 'Custom'
    
    
    
 
 
class PipelineDynamicDockDisplayAreaMixin:
    """ Adds the ability to add/remove dock areas dynamically to a window or widget
    
    Usage:
        PhoPipelineMainWindow only right now 
    
    Requires at minimum:
        'self.area': a pg.Dock(...) object containing the root items
    
    Creates: 
        self.displayDockArea: a pg.Dock(...) object containing dynamically created Docks/Widgets for display of display nodes.
        
    
    """
    
    @property
    def dynamic_display_dict(self):
        """The dynamic_display_dict property."""
        return self._dynamic_display_output_dict
    @dynamic_display_dict.setter
    def dynamic_display_dict(self, value):
        self._dynamic_display_output_dict = value
    
    
    def _build_dynamic_display_dockarea(self):
        dItem = Dock("Display Outputs - Dynamic", size=(600,900), closable=True)
        self.area.addDock(dItem, 'right')
        # Makes a nested DockArea for contents:
        self.displayDockArea = DockArea()
        dItem.addWidget(self.displayDockArea) # add the dynamic nested Dock area to the dItem widget
        self._dynamic_display_output_dict = OrderedDict() # for PipelineDynamicDockDisplayAreaMixin
        return dItem
    
    def add_display_dock(self, identifier = None, viewContentsType: ProducedViewType = ProducedViewType.Matplotlib):
        """ adds a dynamic display dock with an appropriate widget of type 'viewContentsType' to the dock area container on the main window. """
        # Add the sample display dock items to the nested dynamic display dock:
        display_dock_area = self.displayDockArea
        curr_display_dock_items = display_dock_area.children()
        curr_num_display_dock_items = len(curr_display_dock_items)

        if identifier is None:
            identifier = 'item'
        
        extant_group_items = self.dynamic_display_dict.get(identifier, None) # tries to find extant items with this identifier in the dict of extant plots
        if extant_group_items is not None:
            # Item was found with this identifier, implement one of the strategies
            curr_extant_group_item_count = len(extant_group_items)
            unique_identifier = f'{identifier}-{curr_extant_group_item_count}'
        else:
            # no extant items found
            unique_identifier = identifier

        # if identifier is None:
        #     identifier = f'Display Subdock Item {curr_num_display_dock_items}'
        # else:
        #     identifier = f'{identifier}-{curr_num_display_dock_items}'        
        
        dDisplayItem = Dock(unique_identifier, size=(300,200), closable=True) # add the new display item
        display_dock_area.addDock(dDisplayItem, 'bottom')
        
        # Add the widget to the new display item:
        if viewContentsType.value is ProducedViewType.Matplotlib.value:
            new_view_widget = MatplotlibWidget() # Matplotlib widget directly
            # add example plot to figure
            subplot = new_view_widget.getFigure().add_subplot(111)
            subplot.plot(np.arange(9))
            new_view_widget.draw()
        elif viewContentsType.value is ProducedViewType.Pyvista.value:
            new_view_widget = None
            raise NotImplementedError
        else:
            new_view_widget = None
            raise NotImplementedError
    
        # Set the dock item's widget to the new_view_widget    
        dDisplayItem.addWidget(new_view_widget)
        
        
        if extant_group_items is not None:
            # Item was found with this identifier, implement one of the strategies
            extant_group_items[unique_identifier] = {"dock":dDisplayItem, "widget":new_view_widget} # add the unique item to the group's dict
            self.dynamic_display_dict[identifier] = extant_group_items # update the extant group's dict
        else:
            self.dynamic_display_dict[identifier] = OrderedDict() # initialize an empty group for the dict
            self.dynamic_display_dict[identifier][unique_identifier] = {"dock":dDisplayItem, "widget":new_view_widget}
            
        # self.dynamic_display_dict[identifier] = {"dock":dDisplayItem, "widget":new_view_widget}        
        return new_view_widget, dDisplayItem
    
    
    def remove_display_dock(self, identifier):
        """ removes a group of dynamic display widgets with identifier 'identifier'. """
        extant_group_items = self.dynamic_display_dict.get(identifier, None) # tries to find extant items with this identifier in the dict of extant plots
        if extant_group_items is not None:
            num_found_group_items = len(extant_group_items)
            if num_found_group_items > 0:
                # Item was found with this identifier
                print(f'Found a group with the identifier {identifier} containing {num_found_group_items} items. Removing all...')
                for (unique_identifier, item_dict) in extant_group_items.items():
                    # loop through the dictionary and remove the children items:
                    # item_dict['widget'].close() # this shouldn't be needed because the 'dock' is the parent, meaning it should properly close the widget as well.
                    item_dict["dock"].close() # close the dock
                    # del extant_group_items[unique_identifier]
                
                # once done with all children, remove the extant_group_items group:
                del self.dynamic_display_dict[identifier]
                
            else:
                # group was found and valid but already empty prior to remove:
                ## TODO: remove group entirely
                del self.dynamic_display_dict[identifier] # remove the empty dict

        else:
            # no extant items found
            print(f'No extant groups/items found with name {identifier}')
            return
        
    # TODO: Persistance:
    # self.plotDict[name] = {"dock":dock, "widget":widget, "view":view}
    
    
 
 
    #  def _build_debug_test_menu(self):
    #     w1 = pg.LayoutWidget()
    #     label = QtWidgets.QLabel(""" -- DockArea Example -- 
    #     This window has 6 Dock widgets in it. Each dock can be dragged
    #     by its title bar to occupy a different space within the window 
    #     but note that one dock has its title bar hidden). Additionally,
    #     the borders between docks may be dragged to resize. Docks that are dragged on top
    #     of one another are stacked in a tabbed layout. Double-click a dock title
    #     bar to place it in its own window.
    #     """)
    #     saveBtn = QtWidgets.QPushButton('Save dock state')
    #     restoreBtn = QtWidgets.QPushButton('Restore dock state')
    #     restoreBtn.setEnabled(False)
    #     w1.addWidget(label, row=0, col=0)
    #     w1.addWidget(saveBtn, row=1, col=0)
    #     w1.addWidget(restoreBtn, row=2, col=0)
    #     d1.addWidget(w1)
    #     state = None
    #     def save():
    #         global state
    #         state = self.area.saveState()
    #         restoreBtn.setEnabled(True)
    #     def load():
    #         global state
    #         self.area.restoreState(state)
    #     saveBtn.clicked.connect(save)
    #     restoreBtn.clicked.connect(load)
        
        
 
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
    
    
    
    # Matplotlib Widget Mode:
    def on_remove_widget_fn(self, widget, layout):
        """ the callback to remove the widget from the layout.
            implicitly used 'layout'.
        """
        item_index = layout.indexOf(widget)
        print(f'on_remove_widget_fn(...): item_index: {item_index}')
        item = layout.itemAt(item_index)
        widget = item.widget() # this should be the same as the passed in widget, but do this just to be sure
        layout.removeWidget(widget)


    def on_add_widget_fn(self, layout, show_in_separate_window=True):
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
    
    
class DisplayNodeChangeSelectedFunctionMixin:
    
    def on_changed_display_fcn(self, old_fcn, new_fcn):
        """ called when the node changes its display function for any reason.
            - call the self.on_deselect_display_fcn(...) for any previously selected functions.
            - call new display fcn and set any views the function produces as owned by this node. Cache in extant views variable.
        """
        self.on_deselect_display_fcn(old_fcn)
        pass


    def on_deselect_display_fcn(self, old_fcn):
        """ called when the node stops displaying a previously selected display function for any reason (changing to a new function, invalid input, closing, etc)
            - close out extant views it owns
            - null out extant views variable
        
        """
        pass
    
    
    
    
class DisplayMatplotlibWidgetMixin:
    def display_matplotlib_widget(self):
        if (self.view is None):
                # re-open view if we have a function to do so:
                if self.on_add_function is not None:
                    self.on_create_view(None)

        # test plot
        active_fig = self.view.getFigure()
        active_fig.clf()
        self.view.draw()

        # subplot = self.view.getFigure().add_subplot(111)
        # subplot.plot(np.arange(9), np.full((9,), 15))
        
        # active_fig_num = None
        active_fig_num = 1
        # active_fig_num = active_fig.number
                    
        # active_fig_num = self.view.getFigure() # pass the figure itself as the fignum
        # print(f'active_fig_num: {active_fig_num}')
        return {'fignum':active_fig_num, 'fig':active_fig} # could do, but it wouldn't work for 2d functions that didn't accept either of thse parameters.
    
    
    
    # def example_run_3d_pyvista_fcn():
        
    #     if self.display_results is not None:
    #         custom_args = self.display_results.get('kwargs', {})
    #     else:
    #         custom_args = {} # no custom args, just pass empty dictionary

    #     display_outputs = pipeline.display(curr_display_fcn, active_config_name, **custom_args) # extant_plotter=
    #     if display_outputs is dict:
    #         # self.display_results = dict()
    #         self.display_results['outputs'] = display_outputs
    #         # Search for extant_plotter to reuse in the future calls:
    #         active_plotter = display_outputs.get('plotter', None)
    #         # BackgroundPlotter, MultiPlotter
    #         self.display_results['kwargs'] = {'extant_plotter':active_plotter}
            