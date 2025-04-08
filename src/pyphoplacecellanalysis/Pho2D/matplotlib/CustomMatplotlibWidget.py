from copy import deepcopy
from typing import List, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import io
import sys

import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from neuropy.utils.mixins.dict_representable import get_dict_subset # used to filter kwargs down to proper Figure inputs

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphocorehelpers.gui.Qt.widgets.toast_notification_widget import ToastWidget, ToastShowingWidgetMixin
from pyphocorehelpers.plotting.mixins.plotting_backend_mixin import PlottingBackendSpecifyingMixin, PlottingBackendType
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.CrosshairsTracingMixin import CrosshairsTracingMixin

__all__ = ['CustomMatplotlibWidget']

@metadata_attributes(short_name=None, tags=['matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-08 14:20')
class CustomMatplotlibWidget(CrosshairsTracingMixin, ToastShowingWidgetMixin, PlottingBackendSpecifyingMixin, QtWidgets.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Based off of pyqtgraphs's MatplotlibWidget (pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget)
    Example::

        from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget
        mw = CustomMatplotlibWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, scrollAreaContents_MinimumHeight=params.all_plots_height)
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance
    

    @classmethod
    def get_plot_backing_type(cls) -> PlottingBackendType:
        """PlottingBackendSpecifyingMixin conformance: Implementor should return either [PlottingBackendType.Matplotlib, PlottingBackendType.PyQtGraph]."""
        return PlottingBackendType.Matplotlib


    @property
    def active_plot_target(self):
        """The active_plot_target property."""
        return self.ax
        

    def __init__(self, name='CustomMatplotlibWidget', plot_function_name=None, disable_toolbar=True, scrollable_figure=True, size=(5.0, 4.0), dpi=72, **kwargs):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to 'CustomMatplotlibWidget'.
            disable_toolbar (bool, optional): _description_. Defaults to True.
            scrollable_figure (bool, optional): If True, the figure (the canvas that renders it) is embedded in a QScrollArea to allow the user to scroll. Defaults to True.
            size (tuple, optional): _description_. Defaults to (5.0, 4.0).
            dpi (int, optional): _description_. Defaults to 72.
        """
        QtWidgets.QWidget.__init__(self)
        
        ## Init containers:
        self.params = VisualizationParameters(name=name, plot_function_name=plot_function_name, debug_print=False,
                                               wants_crosshairs=kwargs.get('wants_crosshairs', False), should_force_discrete_to_bins=kwargs.get('should_force_discrete_to_bins', False),
                                               vertical_crosshair_formatting_dict=kwargs.get('vertical_crosshair_formatting_dict', dict(color='#009900', lw=1, ls='--')),
                                               horizontal_crosshair_formatting_dict=kwargs.get('horizontal_crosshair_formatting_dict', dict(color='#00996b', lw=1, ls='--')),
                                               )
        self.plots_data = RenderPlotsData(name=name)
        self.plots = RenderPlots(name=name)
        self.ui = PhoUIContainer(name=name)
        self.ui.connections = PhoUIContainer(name=name)

        self.params.name = name
        if plot_function_name is not None:
            self.params.window_title = f" - ".join([name, plot_function_name]) # name should be first so window title is rendered reasonably. kwargs.pop('plot_function_name', name)
        else:
            # TypeError: sequence item 1: expected str instance, NoneType found
            self.params.window_title = f"{name}"
            
        self.params.disable_toolbar = disable_toolbar
        self.params.scrollable_figure = scrollable_figure
        self.params.scrollAreaContents_MinimumHeight = kwargs.pop('scrollAreaContents_MinimumHeight', None)
        self.params.verticalScrollBarPolicy = kwargs.pop('verticalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.params.horizontalScrollBarPolicy = kwargs.pop('horizontalScrollBarPolicy', pg.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Extract figure_kwargs:
        self.params.figure_kwargs = get_dict_subset(kwargs, ['figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth', 'frameon', 'subplotpars', 'tight_layout', 'constrained_layout', 'layout']) 
        self.params.figure_kwargs['figsize'] = size
        self.params.figure_kwargs['dpi'] = dpi
        
        self.setup()
        self.buildUI()
        
    def setup(self):
        pass
    
    def buildUI(self):
        ## Init Figure and components
        self.plots.fig = Figure(**self.params.figure_kwargs)
        self.ui.canvas = FigureCanvas(self.plots.fig)
        self.ui.canvas.setParent(self)

        # Default matplotlib toolbar:
        self.ui.toolbar = None
        if not self.params.disable_toolbar:
            self._buildUI_default_toolbar()
        
        if self.params.scrollable_figure:
            self._buildUI_buildScrollableWidget()
        else:
            self._buildUI_buildNonScrollableWidget()

        self.ui.statusBar = None
        # self._buildUI_setup_statusbar()
        self.toast = None
        self._init_ToastShowingWidgetMixin()

    def _buildUI_setup_statusbar(self):
        """ builds a status bar added to the bottom of the non-scrollable view.
        """
        
        # Only works on QMainWindow subclasses
        # self.ui.statusBar = QtWidgets.QStatusBar()
        # self.window().setStatusBar(self.ui.statusBar)


        # Add the status label to the layout
        assert self.ui.root_vbox is not None, f"Must be called after proper _buildUI_* function so self.ui.root_vbox exists!"
        
        self.params.statusBarBottom_MinimumHeight = 24
        # Create a status bar-like widget
        self.ui.statusBar = QtWidgets.QLabel("Ready")
        self.ui.statusBar.setFrameStyle(QtWidgets.QFrame.Sunken | QtWidgets.QFrame.Panel)
        self.ui.statusBar.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.ui.statusBar.setMinimumHeight(self.params.statusBarBottom_MinimumHeight)

        self.ui.root_vbox.addWidget(self.ui.statusBar)
        return self.ui.statusBar


    def _buildUI_default_toolbar(self) -> NavigationToolbar:
        """ actually builds and adds a toolbar for the figure
        
        """
        assert self.ui.toolbar is None
        assert self.ui.canvas is not None, f"Requires functional canvas object"
        self.ui.toolbar = NavigationToolbar(self.ui.canvas, self)
        return self.ui.toolbar
            

    def _buildUI_buildNonScrollableWidget(self):
        """ sets up the widget to contain a basic layout with no scrollability """
        self.ui.root_vbox = QtWidgets.QVBoxLayout()
        self.ui.root_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.root_vbox.setObjectName('root_vbox_layout')
        ## Non-scrollable version:
        target_vbox = self.ui.root_vbox
        ## Add the real widgets:
        if not self.params.disable_toolbar:
            target_vbox.addWidget(self.ui.toolbar)
        target_vbox.addWidget(self.ui.canvas)
        ## Common:
        self.setLayout(self.ui.root_vbox)
        

    def _buildUI_buildScrollableWidget(self):
        """ sets up the widget to contain a QScrollArea that contains the main figure """
        self.ui.root_vbox = QtWidgets.QVBoxLayout()
        self.ui.root_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.root_vbox.setObjectName('root_vbox')
        
        ### Scrollable Version:
        ## Build the contents widget and inner_contents_vbox:
        self.ui.scrollAreaContentsWidget = QtWidgets.QWidget()
        self.ui.scrollAreaContentsWidget.setObjectName('scrollAreaContentsWidget')
        self.ui.inner_contents_vbox = QtWidgets.QVBoxLayout()
        self.ui.inner_contents_vbox.setContentsMargins(0, 0, 0, 0)
        self.ui.inner_contents_vbox.setObjectName('inner_contents_vbox')
        target_vbox = self.ui.inner_contents_vbox
        ## Add the real widgets:
        if not self.params.disable_toolbar:
            target_vbox.addWidget(self.ui.toolbar)
        target_vbox.addWidget(self.ui.canvas)
        self.ui.scrollAreaContentsWidget.setLayout(self.ui.inner_contents_vbox)
        
        ## Optional Scroll Area Widget:
        self.ui.scrollAreaWidget = QtWidgets.QScrollArea() # Scroll Area which contains the widgets, set as the centralWidget
        
        #Scroll Area Properties
        self.ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn) #  Qt.ScrollBarAlwaysOn, ScrollBarAsNeeded
        self.ui.scrollAreaWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Qt.ScrollBarAlwaysOff
        self.ui.scrollAreaWidget.setWidgetResizable(True)
        # self.ui.scrollAreaContentsWidget = widget # Widget that contains the collection of Vertical Box
        if self.ui.scrollAreaContentsWidget is not None:
            # Set contents widget if we have it:
            self.ui.scrollAreaWidget.setWidget(self.ui.scrollAreaContentsWidget)
        self.ui.root_vbox.addWidget(self.ui.scrollAreaWidget)

        self.setLayout(self.ui.root_vbox)
        # Set the minimumHeight to the
        if self.params.scrollAreaContents_MinimumHeight is not None:
            self.ui.scrollAreaContentsWidget.setMinimumHeight(self.params.scrollAreaContents_MinimumHeight)
        if self.params.verticalScrollBarPolicy is not None:
            self.ui.scrollAreaWidget.setVerticalScrollBarPolicy(self.params.verticalScrollBarPolicy)
        if self.params.horizontalScrollBarPolicy is not None:
            self.ui.scrollAreaWidget.setHorizontalScrollBarPolicy(self.params.horizontalScrollBarPolicy)
            
        
    @property
    def fig(self):
        """The main figure."""
        return self.getFigure()
    
    @property
    def axes(self):
        """The axes that have been added to the figure (via add_subplot(111) or similar)."""
        return self.plots.fig.get_axes()
    
    @property
    def ax(self):
        """The first axes property."""
        if len(self.axes) > 0:
            return self.axes[0]
        else:
            return None
         

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


    def getFigure(self):
        return self.plots.fig
        
    def draw(self):
        """ Redraws the current figure.
        This is what is taking a ton of time with the .Agg backend at least and complex figures. Need a way to freeze it out so it isn't called until it is needed. 
        
        """
        #TODO 2023-07-06 15:05: - [ ] PERFORMANCE - REDRAW
        self.ui.canvas.draw()
        

    def copy_figure_to_clipboard(self):
        """ Copies its figure to the clipboard as an image. """
        from PIL import Image
        from pyphocorehelpers.programming_helpers import copy_image_to_clipboard

        canvas = self.ui.canvas

        canvas.draw()  # Ensure the canvas has been drawn once before copying the figure        
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)
        # Send the image to the clipboard
        copy_image_to_clipboard(img)
        buf.close()


    def copy_axis_to_clipboard(self, an_ax):
        """ Copies the specified axis to the clipboard as an image. 
        
        """
        canvas = self.ui.canvas

        # Redraw the canvas if it might not be up-to-date
        canvas.draw()

        # Get the bbox of the axis to extract it
        bbox = an_ax.get_tightbbox(canvas.get_renderer()).transformed(self.getFigure().dpi_scale_trans.inverted())

        # Save just the region of the figure that contains the axis to a buffer
        buf = io.BytesIO()
        self.getFigure().savefig(buf, format='png', bbox_inches=bbox)
        buf.seek(0)
        qimage = QtGui.QImage.fromData(buf.getvalue())
        buf.close()
        
        # Convert QImage to QPixmap and copy to clipboard
        qpixmap = QtGui.QPixmap.fromImage(qimage)

        ## Sets the QApplication's clipboard:
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setPixmap(qpixmap)



    def update_status(self, message: str):
        """ Update the text displayed in the fake status bar. """
        assert self.ui.statusBar is not None
        self.ui.statusBar.setText(message)

    def clear_status(self):
        """ Update the text displayed in the fake status bar. """
        self.update_status("")

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    def showEvent(self, event):
        """ called only when the widget is shown for the first time. """
        if self.params.debug_print:
            print(f'showEvent(self, event: {event})')
            print(f'\tevent.spontaneous(): {event.spontaneous()}')
        if event.spontaneous():
            if self.params.debug_print:
                print(f'\tfirst show!')
                # Check if the event is spontaneous to identify the first time the widget is shown
                # self.on_first_show()

        if self.isVisible():
            # IMPORTANT: It seems that the title must not be updated until after ui.mw.show() is called or else it crashes the Jupyter kernel! This is why we check for the first show and make sure that the window is visible
            self.window().setWindowTitle(self.params.window_title)
    
        # Call the base class implementation
        super().showEvent(event)


    # ==================================================================================================================== #
    # CrosshairsTracingMixin Conformances                                                                                  #
    # ==================================================================================================================== #
    def format_crosshair_value_string(self, x_point: float, y_point: Optional[float]=None, matrix=None, xbins=None, ybins=None) -> str:
        """Formats the crosshair value string based on cursor position and data
        
        Args:
            x_point: Current x coordinate
            y_point: Current y coordinate
            index_x: Integer x index
            index_y: Integer y index
            matrix: Optional data matrix
            xbins: Optional x bin values
            ybins: Optional y bin values
            
        Returns:
            Formatted string for crosshair display
        """
        
        index_x = int(x_point)
        if (y_point is not None): index_y = int(y_point)
        value_str_arr: List[str] = []
        value_str: str = ''
        
        # Format position strings
        if matrix is not None:
            shape = np.shape(matrix)
            valid_x = (index_x >= 0 and index_x < shape[0])
            valid_y = (index_y >= 0 and index_y < shape[1]) if (y_point is not None) else True
            
            if valid_x and valid_y:
                # Format position string
                if self.params.should_force_discrete_to_bins:
                    if (xbins is not None) and (ybins is not None) and (y_point is not None):
                        value_str_arr.extend([f"(x[{index_x}]={xbins[index_x]:.3f}", f"y[{index_y}]={ybins[index_y]:.3f}"])
                    else:
                        value_str_arr.extend([f"(x={index_x}", f"y={index_x if not (y_point is not None) else index_y}"])
                else:
                    value_str_arr.extend([f"x={x_point:.1f}", f"y={y_point:.1f}"] if (y_point is not None) else [f"x={x_point:.1f}", ])
                
                # Format data value string
                if self.params.should_force_discrete_to_bins:
                    if (xbins is not None) and (ybins is not None) and (y_point is not None):
                        value_str_arr.append(f"value={matrix[index_x][index_y]:.3f}")
                    else:
                        value_str_arr.append(f"value={matrix[index_x][index_y]:.3f}")
                else:
                    value_str_arr.append(f"value={matrix[index_x][index_y]:.3f}" if (y_point is not None) else f"value={matrix[index_x][0]:.3f}")
        else:
            # No matrix provided, just show the (x/y)
            if self.params.should_force_discrete_to_bins:
                if (xbins is not None) and (ybins is not None) and (y_point is not None):
                    value_str_arr.extend([f"(x[{index_x}]={xbins[index_x]:.3f}", f"y[{index_y}]={ybins[index_y]:.3f}"])
                else:
                    value_str_arr.extend([f"(x={index_x}", f"y={index_x if not (y_point is not None) else index_y}"])
            else:
                value_str_arr.extend([f"x={x_point:.1f}", f"y={y_point:.1f}"] if (y_point is not None) else [f"x={x_point:.1f}", ])
        
        # Join formatted strings
        value_str = self.params.crosshair_value_format_join_symbol.join(value_str_arr) ## build the final value output string from the value_str_arr
        return value_str


    @function_attributes(short_name=None, tags=['callback'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-25 18:24', related_items=[])
    def on_crosshair_mouse_moved(self, event, ax, vLine, name, matrix=None, xbins=None, ybins=None):
        """ Called when mouse moves to update crosshair positions and emit signals
        
        History: factored out of a callback `def mouseMoved(event)` in add_crosshairs(...) on 2025-02-25 18:25 
        
        Args:
            event: matplotlib mouse event
            ax: matplotlib axis
            vLine: vertical line artist
            name: name of the crosshair
            matrix: optional data matrix 
            xbins: optional x bin values
            ybins: optional y bin values
            
            
            
        Updates:
        
        self.plots[name]['crosshairs_hLine']
        self.plots[name]['crosshairs_hLine']
        
        """
        are_crosshairs_currently_visible: bool = vLine.get_visible()
        new_desired_visibility_is_visible: bool = True
        did_visibility_change: bool = False
        did_position_change: bool = False
        
        # crosshair_value_format_join_symbol: str = self.params.setdefault('crosshair_value_format_join_symbol', ', ')
        crosshair_value_format_join_symbol: str = self.params.setdefault('crosshair_value_format_join_symbol', '\n')

        if event.inaxes == ax:
            # Mouse is inside the axes - show and update crosshairs
            x_point = event.xdata
            if self.params.crosshairs_enable_y_trace: y_point = event.ydata
            if self.params.should_force_discrete_to_bins:
                x_point = float(int(round(x_point)))+0.5
                if self.params.crosshairs_enable_y_trace: y_point = float(int(round(y_point)))+0.5
            index_x = int(x_point)
            if self.params.crosshairs_enable_y_trace: index_y = int(y_point)
            
            ## try to retrieve the matrix
            if matrix is None:
                matrix = self.plots_data.get('matrix', None)
                xbins = self.plots_data.get('xbins', None)
                ybins = self.plots_data.get('ybins', None)
                              

            # Replace the value formatting code with a single function call
            value_str: str = self.format_crosshair_value_string(x_point, y_point=y_point, matrix=matrix, xbins=xbins, ybins=ybins)
            
            ## check position changes:
            old_x_point = vLine.get_xdata()
            did_position_change = (did_position_change or np.any(old_x_point != x_point))
            vLine.set_xdata(x_point)
                                
            if self.params.crosshairs_enable_y_trace: 
                old_y_point = self.plots[name]['crosshairs_hLine'].get_ydata()
                did_position_change = (did_position_change or np.any(old_y_point != y_point))
                self.plots[name]['crosshairs_hLine'].set_ydata(y_point)
                
            self.sigCrosshairsUpdated.emit(self, name, value_str) ## emit the `sigCrosshairsUpdated` event
                            
            new_desired_visibility_is_visible = True
        else:
            # Mouse left the axes - hide crosshairs
            new_desired_visibility_is_visible = False
            
        # Check vertical line visibility
        are_crosshairs_currently_visible: bool = vLine.get_visible()
        did_visibility_change: bool = (are_crosshairs_currently_visible != new_desired_visibility_is_visible)

        if did_visibility_change:
            ## update visibility
            vLine.set_visible(new_desired_visibility_is_visible)
            if self.params.crosshairs_enable_y_trace:
                self.plots[name]['crosshairs_hLine'].set_visible(new_desired_visibility_is_visible)
                
        ## END if did_visibility_change
        needs_redraw: bool = (did_position_change or did_visibility_change)
        if needs_redraw:
            ax.figure.canvas.draw_idle()


    @function_attributes(short_name=None, tags=['callback', 'matplotlib', 'ax-level'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-25 18:55', related_items=[])
    def on_axes_enter(self, event):
        """Called when mouse enters an axes to show crosshairs for that axes"""
        if event.inaxes is None:
            return
            
        # Only show crosshairs for the entered axes
        for name, plot_items in self.plots.items():
            if not isinstance(plot_items, dict):
                continue
                
            if 'crosshairs_vLine' in plot_items and plot_items['crosshairs_vLine'].axes == event.inaxes:
                plot_items['crosshairs_vLine'].set_visible(True)
                if self.params.crosshairs_enable_y_trace:
                    plot_items['crosshairs_hLine'].set_visible(True)
        event.canvas.draw_idle()

    @function_attributes(short_name=None, tags=['callback', 'matplotlib', 'ax-level'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-25 18:56', related_items=[])
    def on_axes_leave(self, event):
        """Called when mouse leaves an axes to hide its crosshairs"""
        # if event.inaxes is None:
        #     return
        # Only hide crosshairs for the left axes
        for name, plot_items in self.plots.items():
            if not isinstance(plot_items, dict):
                continue
                
            if 'crosshairs_vLine' in plot_items:
                is_visible: bool = (plot_items['crosshairs_vLine'].axes == event.inaxes) # only if it's equal to the current event axes
                plot_items['crosshairs_vLine'].set_visible(is_visible)
                if self.params.crosshairs_enable_y_trace:
                    plot_items['crosshairs_hLine'].set_visible(is_visible)
                    
            # if 'crosshairs_hLine' in plot_items:
            #     is_hline_visible: bool = (plot_items['crosshairs_hLine'].axes == event.inaxes)
            #     plot_items['crosshairs_hLine'].set_visible(is_hline_visible)
            #     if self.params.crosshairs_enable_y_trace:
            #         plot_items['crosshairs_hLine'].set_visible(is_hline_visible)
                    
        event.canvas.draw_idle()



    # @function_attributes(short_name=None, tags=['callback', 'matplotlib', 'figure-level'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-25 18:40', related_items=[])
    # def on_figure_enter(self, event):
    #     """Called when mouse enters the figure to restore crosshair visibility"""
    #     # Skip if self.plots is not a dictionary or is empty
    #     if not isinstance(self.plots, dict) or not self.plots:
    #         return
            
    #     for name, plot_items in self.plots.items():
    #         # Skip non-dictionary items
    #         if not isinstance(plot_items, dict):
    #             continue
                
    #         if 'crosshairs_vLine' in plot_items:
    #             plot_items['crosshairs_vLine'].set_visible(True)
    #             if self.params.crosshairs_enable_y_trace:
    #                 plot_items['crosshairs_hLine'].set_visible(True)
    #     # Redraw to show changes
    #     event.canvas.draw_idle()

    # @function_attributes(short_name=None, tags=['callback', 'matplotlib', 'figure-level'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-25 18:40', related_items=[])
    # def on_figure_leave(self, event):
    #     """Called when mouse leaves the figure to hide all crosshairs"""
    #     # Skip if self.plots is not a dictionary or is empty
    #     if not isinstance(self.plots, dict) or not self.plots:
    #         return
            
    #     for name, plot_items in self.plots.items():
    #         # Skip non-dictionary items
    #         if not isinstance(plot_items, dict):
    #             continue
                
    #         if 'crosshairs_vLine' in plot_items:
    #             plot_items['crosshairs_vLine'].set_visible(False)
    #             if self.params.crosshairs_enable_y_trace:
    #                 plot_items['crosshairs_hLine'].set_visible(False)
    #     # Redraw to hide changes
    #     event.canvas.draw_idle()







    def add_crosshairs(self, plot_item, name, matrix=None, xbins=None, ybins=None, enable_y_trace:bool=True, should_force_discrete_to_bins:Optional[bool]=True, **kwargs):
        """ adds crosshairs that allow the user to hover a bin and have the label dynamically display the bin (x, y) and value.
        
        Uses:
        self.params.should_force_discrete_to_bins
        
        Updates self.plots[name], self.ui.connections[name]
        
        Emits: self.sigCrosshairsUpdated
        
        
        
        Usage:
        
        
        def on_crosshair_updated_signal(self, name, trace_value):
            # print(f'on_crosshair_updated_signal(self: {self}, name: "{name}", trace_value: "{trace_value}")')
            left_side_bar_controls = spike_raster_window.ui.leftSideToolbarWidget
            left_side_bar_controls.crosshair_trace_time = trace_value
            
            # self.ui.lblCrosshairTraceStaticLabel.setVisible(True)
            # self.ui.lblCrosshairTraceValue.setVisible(True)

        track_ts_widget.update_crosshair_trace(wants_crosshairs_trace=True)
        _crosshairs_updated_conn = track_ts_widget.sigCrosshairsUpdated.connect(on_crosshair_updated_signal)

        
        
        # add_crosshairs
        track_ts_widget.add_crosshairs(track_ax, name='traceHairs', should_force_discrete_to_bins=False, enable_y_trace=True)

        ## cleanup/remove with:
            track_ts_widget.remove_crosshairs(track_ax, name='traceHairs')
            track_ts_widget.sigCrosshairsUpdated.disconnect(_crosshairs_updated_conn)
            
        """
        vertical_crosshair_formatting_dict = deepcopy(self.params.get('vertical_crosshair_formatting_dict', dict(color='green', lw=1, ls='--'))) # dict(color='green', lw=1, ls='--')
        horizontal_crosshair_formatting_dict = deepcopy(self.params.get('horizontal_crosshair_formatting_dict', dict(color='green', lw=1, ls='--'))) # dict(color='green', lw=1, ls='--')
        if should_force_discrete_to_bins is not None:
            self.params.should_force_discrete_to_bins = should_force_discrete_to_bins
        should_force_discrete_to_bins: bool = self.params.should_force_discrete_to_bins
        
        # # crosshair_value_format_join_symbol: str = self.params.setdefault('crosshair_value_format_join_symbol', ', ')
        # crosshair_value_format_join_symbol: str = self.params.setdefault('crosshair_value_format_join_symbol', '\n')
        
        if enable_y_trace is not None:
            self.params.crosshairs_enable_y_trace = enable_y_trace
        else:
            assert self.params.crosshairs_enable_y_trace is not None
        
        ax = plot_item
        print(f'Matplotlib add_crosshairs(ax: {ax}, name: "{name}", ...):')
        plots_dict = self.plots.get(name, {})
        if 'crosshairs_vLine' not in plots_dict:
            if name not in self.plots:
                 self.plots[name] = {'crosshairs_vLine': None} ## create the dictionary to hold the related items
            vLine = ax.axvline(x=0, **vertical_crosshair_formatting_dict, label=f'{name}.crosshairs_vLine')
            self.plots[name]['crosshairs_vLine'] = vLine
            if self.params.crosshairs_enable_y_trace:
                hLine = ax.axhline(y=0, **horizontal_crosshair_formatting_dict, label=f'{name}.crosshairs_hLine')
                self.plots[name]['crosshairs_hLine'] = hLine

            ## connect the callback:
            cid = ax.figure.canvas.mpl_connect('motion_notify_event', lambda event: self.on_crosshair_mouse_moved(event, ax, vLine, name, matrix, xbins, ybins))
            self.ui.connections[name] = {'motion_notify_event': cid,
                                         'axes_enter_event': ax.figure.canvas.mpl_connect('axes_enter_event', self.on_axes_enter),
                                         'axes_leave_event': ax.figure.canvas.mpl_connect('axes_leave_event', self.on_axes_leave),
                                         ## figure-level events
                                        #  'figure_enter_event': ax.figure.canvas.mpl_connect('figure_enter_event', self.on_figure_enter),
                                        #  'figure_leave_event': ax.figure.canvas.mpl_connect('figure_leave_event', self.on_figure_leave),
            }

            # 'figure_enter_event'    # Mouse enters figure
            # 'figure_leave_event'    # Mouse leaves figure
            # 'axes_enter_event'      # Mouse enters axes
            # 'axes_leave_event'      # Mouse leaves axes


        else:
            print(f"already have 'crosshairs_vLine' in plots_dict")



    def remove_crosshairs(self, plot_item, name=None):
        """Removes crosshairs from the specified plot or all plots if name is None
        
        Args:
            plot_item: matplotlib axis
            name: optional name of specific crosshair to remove
        """
        if name is None:
            # Remove all crosshairs
            for key in list(self.plots.keys()):
                for ln in ('crosshairs_vLine','crosshairs_hLine'):
                    if (self.plots[key] is not None) and (ln in self.plots[key]): 
                        self.plots[key][ln].remove()
                if (plot_item is not None) and (key in self.ui.connections):
                    # Disconnect all event types
                    for event_type, cid in self.ui.connections[key].items():
                        plot_item.figure.canvas.mpl_disconnect(cid)
                    del self.ui.connections[key]
                del self.plots[key]
        else:
            # Remove specific named crosshair
            if name in self.plots:
                if self.plots[name] is not None:
                    for ln in ('crosshairs_vLine','crosshairs_hLine'):
                        if (ln in self.plots[name]): 
                            self.plots[name][ln].remove()
                if (plot_item is not None) and (name in self.ui.connections):
                    for event_type, cid in self.ui.connections[name].items():
                        plot_item.figure.canvas.mpl_disconnect(cid)
                    del self.ui.connections[name]                                    
                    # plot_item.figure.canvas.mpl_disconnect(self.ui.connections[name])
                if (self.plots[name] is not None):
                    del self.plots[name]
        
        # Trigger redraw
        plot_item.figure.canvas.draw_idle()


    def update_crosshair_trace(self, wants_crosshairs_trace: bool):
        """ updates the crosshair trace peferences
        """
        print(f'CustomMatplotlibWidget.update_crosshair_trace(wants_crosshairs_trace: {wants_crosshairs_trace}):')
        old_value = deepcopy(self.params.wants_crosshairs)
        did_change: bool = (old_value != wants_crosshairs_trace)
        if did_change:
            self.params.wants_crosshairs = wants_crosshairs_trace
            if self.params.wants_crosshairs:
                print(f'\tadding crosshairs...')
                ybins = self.plots_data.get('ybin', None)
                # has_ybins: bool = (ybins not None)
                has_ybins: bool = True
                self.add_crosshairs(plot_item=self.ax, name='root_plot_item', matrix=self.plots_data.get('matrix', None), xbins=self.plots_data.get('xbin', None), ybins=ybins, enable_y_trace=has_ybins, should_force_discrete_to_bins=None)
            else:
                print(f'\tremoving crosshairs...')
                self.remove_crosshairs(plot_item=self.ax, name='root_plot_item')
                
            print(f'\tdone.')
        else:
            print(f'\tno change!')
            
