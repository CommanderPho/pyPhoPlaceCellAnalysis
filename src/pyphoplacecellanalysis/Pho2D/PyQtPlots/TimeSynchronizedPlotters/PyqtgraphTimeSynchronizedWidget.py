from copy import deepcopy
from typing import Optional
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.plotting.mixins.plotting_backend_mixin import PlottingBackendSpecifyingMixin, PlottingBackendType
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.CrosshairsTracingMixin import CrosshairsTracingMixin


@metadata_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 03:42', related_items=['MatplotlibTimeSynchronizedWidget'])
class PyqtgraphTimeSynchronizedWidget(CrosshairsTracingMixin, PlottingBackendSpecifyingMixin, TimeSynchronizedPlotterBase):
    """ Plots the decoded position at a given moment in time. 

    Simple pyqtgraph-based alternative to `MatplotlibTimeSynchronizedWidget`
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget
        TODO: Document

    """
    # Application/Window Configuration Options:
    applicationName = 'PyqtgraphTimeSynchronizedWidgetApp'
    windowName = 'PyqtgraphTimeSynchronizedWidgetWindow'
    
    enable_debug_print = True
    
    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance

    @classmethod
    def get_plot_backing_type(cls) -> PlottingBackendType:
        """PlottingBackendSpecifyingMixin conformance: Implementor should return either [PlottingBackendType.Matplotlib, PlottingBackendType.PyQtGraph]."""
        return PlottingBackendType.PyQtGraph
    

    # @property
    # def time_window_centers(self):
    #     """The time_window_centers property."""
    #     return self.active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,)
    

    # @property
    # def posterior_variable_to_render(self):
    #     """The occupancy_mode_to_render property."""
    #     return self.params.posterior_variable_to_render
    # @posterior_variable_to_render.setter
    # def posterior_variable_to_render(self, value):
    #     self.params.posterior_variable_to_render = value
    #     # on update, be sure to call self._update_plots()
    #     self._update_plots()
    
    @property
    def windowTitle(self):
        """The windowTitle property."""
        return self.params.window_title
    @windowTitle.setter
    def windowTitle(self, value):
        self.params.window_title = value
        if self.window().isVisible():
            print(f'updating the window title!!')
            self.window().setWindowTitle(self.params.window_title)

    

    @property
    def last_t(self):
        raise NotImplementedError(f'Parent property that should not be accessed!')



    @property
    def active_plot_target(self):
        """The active_plot_target property."""
        return self.getRootPlotItem()
    


    def __init__(self, name='PyqtgraphTimeSynchronizedWidget', plot_function_name=None, scrollable_figure=True, application_name=None, window_name=None, parent=None, **kwargs):
        """_summary_
        , disable_toolbar=True, size=(5.0, 4.0), dpi=72
        ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
            occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
        """
        super().__init__(application_name=application_name, window_name=(window_name or PyqtgraphTimeSynchronizedWidget.windowName), parent=parent) # Call the inherited classes __init__ method
            
        ## Init containers:
        self.params = VisualizationParameters(name=name, plot_function_name=plot_function_name, debug_print=False, wants_crosshairs=kwargs.get('wants_crosshairs', False), should_force_discrete_to_bins=kwargs.get('should_force_discrete_to_bins', False))
        self.plots_data = RenderPlotsData(name=name)
        self.plots = PyqtgraphRenderPlots(name=name)
        self.ui = PhoUIContainer(name=name, connections=None)
        self.ui.connections = PhoUIContainer(name=name)

        self.params.name = name
        if plot_function_name is not None:
            self.params.window_title = f" - ".join([name, plot_function_name]) # name should be first so window title is rendered reasonably. kwargs.pop('plot_function_name', name)
        else:
            # TypeError: sequence item 1: expected str instance, NoneType found
            self.params.window_title = f"{name}"
            
        self.params.scrollable_figure = scrollable_figure
        self.params.scrollAreaContents_MinimumHeight = kwargs.pop('scrollAreaContents_MinimumHeight', None)
        self.params.verticalScrollBarPolicy = kwargs.pop('verticalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.params.horizontalScrollBarPolicy = kwargs.pop('horizontalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # self.last_window_index = None
        # self.last_window_time = None
        self.setup()
        
        self.buildUI()
        self._update_plots()
        
    def setup(self):
        assert hasattr(self.ui, 'connections')
        
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        # self.app = pg.mkQApp(self.applicationName)
        # self.params = VisualizationParameters(self.applicationName)
        

        # # Add a trace region (initially hidden)
        # self.trace_region = pg.LinearRegionItem(movable=True, brush=(0, 0, 255, 50))
        # self.trace_region.setZValue(10)  # Ensure it appears above the plot
        # self.trace_region.hide()  # Initially hide the trace region
        # self.plot_widget.addItem(self.trace_region)

        # # Override the PlotWidget's mouse events
        # self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
        # self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        # self.plot_widget.scene().sigMouseReleased.connect(self.mouse_released)
        # self.dragging = False
        # self.start_pos = None
                

        # self.params.shared_axis_order = 'row-major'
        # self.params.shared_axis_order = 'column-major'
        # self.params.shared_axis_order = None
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        # self.params.image_margins = 0.0
        # self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.active_one_step_decoder.xbin, self.active_one_step_decoder.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        pass


    def _buildGraphics(self):
        """ called by self.buildUI() which usually is not overriden. """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomViewBox, CustomGraphicsLayoutWidget

        ## More Involved Mode:
        # self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.ui.root_graphics_layout_widget = CustomGraphicsLayoutWidget()

        # self.ui.root_view = self.ui.root_graphics_layout_widget.addViewBox()
        ## lock the aspect ratio so pixels are always square
        # self.ui.root_view.setAspectLocked(True)

        ## Create image item
        
        # self.ui.imv = pg.ImageItem(border='w')
        # self.ui.root_view.addItem(self.ui.imv)
        # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

        self.ui.root_plot_viewBox = None
        self.ui.root_plot_viewBox = CustomViewBox()
        self.ui.root_plot_viewBox.setObjectName('RootPlotCustomViewBox')
        
        # self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=None) # , name=f'PositionDecoder'
        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=None, viewBox=self.ui.root_plot_viewBox)
        self.ui.root_plot.setObjectName('RootPlot')
        # self.ui.root_plot.addItem(self.ui.imv, defaultPadding=0.0)  # add ImageItem to PlotItem
        ## TODO: add item here
        # self.ui.root_plot.showAxes(True)
        self.ui.root_plot.hideButtons() # Hides the auto-scale button
        
        self.ui.root_plot.showAxes(False)     
        # self.ui.root_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
        # Sets only the panning limits:
        # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

        ## Sets all limits:
        # _x, _y, _width, _height = self.params.image_bounds_extent # [23.923329354140844, 123.85967782096927, 241.7178791533281, 30.256480996256016]
        # self.ui.root_plot.setLimits(minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1],
        #                             minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        
        self.ui.root_plot.setMouseEnabled(x=False, y=False)
        self.ui.root_plot.setMenuEnabled(enableMenu=False)
        
        # ## Optional Interactive Color Bar:
        # bar = pg.ColorBarItem(values= (0, 1), colorMap=self.params.cmap, width=5, interactive=False) # prepare interactive color bar
        # # Have ColorBarItem control colors of img and appear in 'plot':
        # bar.setImageItem(self.ui.imv, insert_in=self.ui.root_plot)
        
        self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # Set the color map:
        # self.ui.imv.setColorMap(self.params.cmap)
        ## Set initial view bounds
        # self.ui.root_view.setRange(QtCore.QRectF(0, 0, 600, 600))

    
    def update(self, t, defer_render=False):
        if self.enable_debug_print:
            print(f'PyqtgraphTimeSynchronizedWidget.update(t: {t})')
    
        # # Finds the nearest previous decoded position for the time t:
        # self.last_window_index = np.searchsorted(self.time_window_centers, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        # self.last_window_time = self.time_window_centers[self.last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        # Update the plots:
        if not defer_render:
            self._update_plots()


    def _update_plots(self):
        if self.enable_debug_print:
            print(f'PyqtgraphTimeSynchronizedWidget._update_plots()')

        # Update the existing one:
        # self.ui.root_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
        # Sets only the panning limits:
        # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

        ## Sets all limits:
        # _x, _y, _width, _height = self.params.image_bounds_extent # [23.923329354140844, 123.85967782096927, 241.7178791533281, 30.256480996256016]
        # self.ui.root_plot.setLimits(minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1],
        #                             minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        
        # Update the plots:
        # curr_time_window_index = self.last_window_index
        # curr_t = self.last_window_time

        # if curr_time_window_index is None or curr_t is None:
        #     return # return without updating

        # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        # self.setWindowTitle(f'PyqtgraphTimeSynchronizedWidget - {image_title} t = {curr_t}')
        pass

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'PyqtgraphTimeSynchronizedWidget.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        # if self.enable_debug_print:
        #     profiler = pg.debug.Profiler(disabled=True, delayed=True)

        if (start_t is not None) and (end_t is not None):            
            self.getRootPlotItem().setXRange(start_t, end_t, padding=0) ## global frame

        self.update(end_t, defer_render=False)
        # if self.enable_debug_print:
        #     profiler('Finished calling _update_plots()')
            


    # def mouse_clicked(self, event):
    #     # Only handle middle mouse button
    #     if event.button() == 2:  # Middle mouse button
    #         pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
    #         self.start_pos = pos.x()
    #         self.dragging = True
    #         self.trace_region.hide()  # Reset trace region visibility
    #         event.accept()

    # def mouse_moved(self, event):
    #     if self.dragging and self.start_pos is not None:
    #         # Update the trace region during dragging
    #         current_pos = self.plot_widget.plotItem.vb.mapSceneToView(event)
    #         x_end = current_pos.x()
    #         self.trace_region.setRegion([min(self.start_pos, x_end), max(self.start_pos, x_end)])
    #         self.trace_region.show()  # Show the trace region as it's being defined

    # def mouse_released(self, event):
    #     # Finalize the trace region definition
    #     if event.button() == 2 and self.dragging:
    #         self.dragging = False
    #         self.start_pos = None
    #         print(f"Trace region set to: {self.trace_region.getRegion()}")
            


    def getRootLayout(self) -> QtWidgets.QGridLayout:
        return self.ui.layout
    
    def getRootGraphicsLayoutWidget(self) -> pg.GraphicsLayoutWidget:
        return self.ui.root_graphics_layout_widget
    
    def getRootPlotItem(self) -> pg.PlotItem:
        return self.ui.root_plot
    
    # ==================================================================================================================== #
    # Misc Functionality                                                                                                   #
    # ==================================================================================================================== #
    

    # ==================================================================================================================== #
    # CrosshairsTracingMixin Conformances                                                                                  #
    # ==================================================================================================================== #
    def on_crosshair_mouse_moved(self, pos, plot_item, vb, vLine, name, matrix=None, xbins=None, ybins=None):
        """Handles crosshair updates when mouse moves
        
        Args:
            pos: Mouse position in scene coordinates
            plot_item: The plot item containing the crosshair
            vb: ViewBox reference
            vLine: The vertical line object
            name: Identifier for this crosshair
            matrix: Optional data matrix
            xbins: Optional x bin edges
            ybins: Optional y bin edges
        """
        # Check if mouse is within the plot area
        if not plot_item.sceneBoundingRect().contains(pos):
            # Hide crosshairs when outside plot
            vLine.setVisible(False)
            if self.params.crosshairs_enable_y_trace and 'crosshairs_hLine' in self.plots[name]:
                self.plots[name]['crosshairs_hLine'].setVisible(False)
            return
            
        # Map scene coordinates to data coordinates
        mousePoint = vb.mapSceneToView(pos)
        x_point = mousePoint.x()
        y_point = mousePoint.y() if self.params.crosshairs_enable_y_trace else 0
        
        # Handle discrete vs continuous formatting
        if self.params.should_force_discrete_to_bins:
            x_point_discrete = float(int(round(x_point)))
            y_point_discrete = float(int(round(y_point))) if self.params.crosshairs_enable_y_trace else 0
            
            # Snap to center of bin for display
            x_point = x_point_discrete + 0.5
            y_point = y_point_discrete + 0.5 if self.params.crosshairs_enable_y_trace else 0
            
            index_x = int(x_point_discrete)
            index_y = int(y_point_discrete) if self.params.crosshairs_enable_y_trace else 0
        else:
            index_x = int(x_point)
            index_y = int(y_point) if self.params.crosshairs_enable_y_trace else 0
        
        # Format value string
        value_str = ""
        if matrix is not None:
            shape = np.shape(matrix)
            valid_x = (index_x >= 0 and index_x < shape[0])
            valid_y = (index_y >= 0 and index_y < shape[1]) if self.params.crosshairs_enable_y_trace else True
            
            if valid_x and valid_y:
                # Position component
                position_str = ""
                if self.params.should_force_discrete_to_bins:
                    if (xbins is not None) and (ybins is not None) and self.params.crosshairs_enable_y_trace:
                        position_str = f"(x[{index_x}]={xbins[index_x]:.3f}, y[{index_y}]={ybins[index_y]:.3f})"
                    else:
                        position_str = f"(x={index_x}, y={index_y if self.params.crosshairs_enable_y_trace else index_x})"
                else:
                    position_str = f"(x={x_point:.1f}, y={y_point:.1f})" if self.params.crosshairs_enable_y_trace else f"(x={x_point:.1f})"
                
                # Value component
                value_component = ""
                if valid_x and valid_y:
                    value_component = f"value={matrix[index_x][index_y if self.params.crosshairs_enable_y_trace else 0]:.3f}"
                
                # Final formatted string for PyQtGraph (can use HTML)
                value_str = f"<span style='font-size: 12pt'>{position_str}, <span style='color: green'>{value_component}</span></span>"
        else:
            # No matrix, just show coordinates
            if self.params.should_force_discrete_to_bins:
                if (xbins is not None) and (ybins is not None) and self.params.crosshairs_enable_y_trace:
                    value_str = f"<span style='font-size: 12pt'>(x[{index_x}]={xbins[index_x]:.3f}, y[{index_y}]={ybins[index_y]:.3f})</span>"
                else:
                    value_str = f"<span style='font-size: 12pt'>(x={index_x}, y={index_y if self.params.crosshairs_enable_y_trace else index_x})</span>"
            else:
                value_str = f"<span style='font-size: 12pt'>(x={x_point:.1f}, y={y_point:.1f})</span>" if self.params.crosshairs_enable_y_trace else f"<span style='font-size: 12pt'>(x={x_point:.1f})</span>"
        
        # Update crosshair positions
        vLine.setPos(x_point)
        vLine.setVisible(True)
        
        if self.params.crosshairs_enable_y_trace and 'crosshairs_hLine' in self.plots[name]:
            self.plots[name]['crosshairs_hLine'].setPos(y_point)
            self.plots[name]['crosshairs_hLine'].setVisible(True)
        
        # Emit signal with formatted value
        self.sigCrosshairsUpdated.emit(self, name, value_str)


    def on_crosshair_enter_view(self, plot_item, name):
        """Called when mouse enters the view"""
        if name in self.plots and 'crosshairs_vLine' in self.plots[name]:
            self.plots[name]['crosshairs_vLine'].setVisible(True)
            if 'crosshairs_hLine' in self.plots[name]:
                self.plots[name]['crosshairs_hLine'].setVisible(True)

    def on_crosshair_leave_view(self, plot_item, name):
        """Called when mouse leaves the view"""
        if name in self.plots and 'crosshairs_vLine' in self.plots[name]:
            self.plots[name]['crosshairs_vLine'].setVisible(False)
            if 'crosshairs_hLine' in self.plots[name]:
                self.plots[name]['crosshairs_hLine'].setVisible(False)




    def add_crosshairs(self, plot_item, name, matrix=None, xbins=None, ybins=None, enable_y_trace:bool=False, should_force_discrete_to_bins:Optional[bool]=True, **kwargs):
        """ adds crosshairs that allow the user to hover a bin and have the label dynamically display the bin (x, y) and value.
        
        Uses:
        self.params.should_force_discrete_to_bins
        
        Updates self.plots[name], self.ui.connections[name]
        
        Emits: self.sigCrosshairsUpdated
        
        """
        print(f'PyqtgraphTimeSynchronizedWidget.add_crosshairs(plot_item: {plot_item}, name: "{name}", ...):')
        try:
            # Create ViewBox and crosshair lines
            vb = plot_item.getViewBox()
            
            # Create vertical line for crosshair
            vLine = pg.InfiniteLine(angle=90, movable=False)
            plot_item.addItem(vLine, ignoreBounds=True)
            self.plots[name]['crosshairs_vLine'] = vLine
            vLine.setVisible(False)
            
            # Create horizontal line if y-trace is enabled
            if enable_y_trace:
                hLine = pg.InfiniteLine(angle=0, movable=False)
                plot_item.addItem(hLine, ignoreBounds=True)
                self.plots[name]['crosshairs_hLine'] = hLine
                hLine.setVisible(False)
            
            # Store parameters
            self.params.crosshairs_enable_y_trace = enable_y_trace
            self.params.should_force_discrete_to_bins = should_force_discrete_to_bins
            
            # Connect signals with better error handling
            self.ui.connections[name] = pg.SignalProxy(plot_item.scene().sigMouseMoved, 
                                                    rateLimit=60, 
                                                    slot=lambda evt: self.on_crosshair_mouse_moved(evt[0], vb, vLine, name, matrix, xbins, ybins))
            
            # Connect enter/leave events
            plot_item.getViewBox().setMouseEnabled(x=True, y=True)
            plot_item.scene().sigMouseEntered.connect(lambda: self.on_crosshair_enter_view(plot_item, name))
            plot_item.scene().sigMouseExited.connect(lambda: self.on_crosshair_leave_view(plot_item, name))
            
            # Success message
            print(f"Successfully added crosshairs for {name}")
        except Exception as e:
            print(f"Failed to add crosshair traces for widget: {self}. Error: {str(e)}")



    def remove_crosshairs(self, plot_item, name=None):
        print(f'PyqtgraphTimeSynchronizedWidget.remove_crosshairs(plot_item: {plot_item}, name: "{name}"):')
        try:
            # Logic to remove specific or all crosshairs
            if name is None:
                # Remove all crosshairs
                for key in list(self.plots.keys()):
                    if 'crosshairs_vLine' in self.plots[key]:
                        self.plots[key]['crosshairs_vLine'].setParent(None)
                        del self.plots[key]['crosshairs_vLine']
                    if 'crosshairs_hLine' in self.plots[key]:
                        self.plots[key]['crosshairs_hLine'].setParent(None)
                        del self.plots[key]['crosshairs_hLine']
                    # Disconnect signals
                    if key in self.ui.connections:
                        self.ui.connections[key].disconnect()
                        del self.ui.connections[key]
            else:
                # Remove specific crosshair
                if name in self.plots:
                    if 'crosshairs_vLine' in self.plots[name]:
                        self.plots[name]['crosshairs_vLine'].setParent(None)
                        del self.plots[name]['crosshairs_vLine']
                    if 'crosshairs_hLine' in self.plots[name]:
                        self.plots[name]['crosshairs_hLine'].setParent(None)
                        del self.plots[name]['crosshairs_hLine']
                    # Disconnect signal
                    if name in self.ui.connections:
                        self.ui.connections[name].disconnect()
                        del self.ui.connections[name]
            
            # Trigger update
            plot_item.update()
        except Exception as e:
            print(f"Error removing crosshairs: {str(e)}")



    def update_crosshair_trace(self, wants_crosshairs_trace: bool):
        """ updates the crosshair trace peferences
        """
        print(f'PyqtgraphTimeSynchronizedWidget.update_crosshair_trace(wants_crosshairs_trace: {wants_crosshairs_trace}):')
        old_value = deepcopy(self.params.wants_crosshairs)
        did_change: bool = (old_value != wants_crosshairs_trace)
        if did_change:
            self.params.wants_crosshairs = wants_crosshairs_trace
            root_plot_item = self.getRootPlotItem()
            if wants_crosshairs_trace:
                print(f'\tadding crosshairs...')
                self.add_crosshairs(plot_item=root_plot_item, name='root_plot_item', matrix=None, xbins=None, ybins=None, enable_y_trace=False)
                print(f'\tdone.')
            else:
                print(f'\tremoving crosshairs...')
                self.remove_crosshairs(plot_item=root_plot_item, name='root_plot_item')
                print(f'\tdone.')
        else:
            print(f'\tno change!')
            

# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = PyqtgraphTimeSynchronizedWidget(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()