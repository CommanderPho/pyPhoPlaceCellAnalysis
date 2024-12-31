import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


@metadata_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 03:42', related_items=['MatplotlibTimeSynchronizedWidget'])
class PyqtgraphTimeSynchronizedWidget(TimeSynchronizedPlotterBase):
    """ Plots the decoded position at a given moment in time. 

    Simple pyqtgraph-based alternative to `MatplotlibTimeSynchronizedWidget`
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget
        TODO: Document

    """
    # Application/Window Configuration Options:
    applicationName = 'PyqtgraphTimeSynchronizedWidgetApp'
    windowName = 'PyqtgraphTimeSynchronizedWidgetWindow'
    
    enable_debug_print = False
        
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

    
    def __init__(self, name='PyqtgraphTimeSynchronizedWidget', plot_function_name=None, scrollable_figure=True, application_name=None, window_name=None, parent=None, **kwargs):
        """_summary_
        , disable_toolbar=True, size=(5.0, 4.0), dpi=72
        ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
            occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
        """
        super().__init__(application_name=application_name, window_name=(window_name or PyqtgraphTimeSynchronizedWidget.windowName), parent=parent) # Call the inherited classes __init__ method
            
        ## Init containers:
        self.params = VisualizationParameters(name=name, plot_function_name=plot_function_name, debug_print=False)
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
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        # self.app = pg.mkQApp(self.applicationName)
        # self.params = VisualizationParameters(self.applicationName)
        pass
        

        # self.params.shared_axis_order = 'row-major'
        # self.params.shared_axis_order = 'column-major'
        # self.params.shared_axis_order = None
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        # self.params.image_margins = 0.0
        # self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.active_one_step_decoder.xbin, self.active_one_step_decoder.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        

    def _buildGraphics(self):
        """ called by self.buildUI() which usually is not overriden. """
        ## More Involved Mode:
        self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        # self.ui.root_view = self.ui.root_graphics_layout_widget.addViewBox()
        ## lock the aspect ratio so pixels are always square
        # self.ui.root_view.setAspectLocked(True)

        ## Create image item
        
        # self.ui.imv = pg.ImageItem(border='w')
        # self.ui.root_view.addItem(self.ui.imv)
        # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=None) # , name=f'PositionDecoder'
        self.ui.root_plot.setObjectName('RootPlot')
        # self.ui.root_plot.addItem(self.ui.imv, defaultPadding=0.0)  # add ImageItem to PlotItem
        ## TODO: add item here
        self.ui.root_plot.showAxes(True)
        self.ui.root_plot.hideButtons() # Hides the auto-scale button
        
        # self.ui.root_plot.showAxes(False)        
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
        # Finds the nearest previous decoded position for the time t:
        self.last_window_index = np.searchsorted(self.time_window_centers, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        self.last_window_time = self.time_window_centers[self.last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        # Update the plots:
        if not defer_render:
            self._update_plots()


    def _update_plots(self):
        if self.enable_debug_print:
            print(f'PyqtgraphTimeSynchronizedWidget._update_plots()')

        # Update the existing one:

        # Update the plots:
        # curr_time_window_index = self.last_window_index
        # curr_t = self.last_window_time

        # if curr_time_window_index is None or curr_t is None:
        #     return # return without updating

        # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        # self.setWindowTitle(f'PyqtgraphTimeSynchronizedWidget - {image_title} t = {curr_t}')
        pass

    def getRootLayout(self) -> QtWidgets.QGridLayout:
        return self.ui.layout
    
    def getRootGraphicsLayoutWidget(self) -> pg.GraphicsLayoutWidget:
        return self.ui.root_graphics_layout_widget
    
    def getRootPlotItem(self) -> pg.PlotItem:
        return self.ui.root_plot
    
    

    

# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = PyqtgraphTimeSynchronizedWidget(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()