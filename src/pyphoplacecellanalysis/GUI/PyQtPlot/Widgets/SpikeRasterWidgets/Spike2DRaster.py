from copy import deepcopy
import time
from typing import Tuple, List, Dict, Optional
import sys
from indexed import IndexedOrderedDict
from attrs import define, field, Factory # used for `FigureWidgetDockDisplayConfig`

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType, TypeVar
from neuropy.utils.indexing_helpers import find_desired_sort_indicies
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types
import numpy as np
import pandas as pd
from neuropy.utils.mixins.indexing_helpers import get_dict_subset

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot


# For Dynamic Plot Widget Adding
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
# from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaOwningMixin, DynamicDockDisplayAreaContentMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget

# For a specific type of dynamic plot widget
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget ## potential import issue due to Qt?

import numpy as np

# import qdarkstyle
from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color # required for the different emphasis states in ._build_cell_configs()

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DNeuronIdentityLinesMixin import Render2DNeuronIdentityLinesMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, SessionEpochs2DRenderTimeEpochs, PBE_2DRenderTimeEpochs, Laps2DRenderTimeEpochs

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin import PyQtGraphSpecificTimeCurvesMixin
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_print_QRect, debug_print_axes_locations, debug_print_temporal_info
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState # required for the different emphasis states in ._build_cell_configs()

from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, DockDisplayColors, FigureWidgetDockDisplayConfig
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper

from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomViewBox
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.SpecificDockWidgetManipulatingMixin import SpecificDockWidgetManipulatingMixin


class SynchronizedPlotMode(ExtendedEnum):
    """Describes the type of file progress actions that can be performed to get the right verbage.
    Used by `print_file_progress_message(...)`
    """
    NO_SYNC = "no_sync" # independent 
    TO_GLOBAL_DATA = "to_global_data" # synchronized only to the global start and end times
    TO_WINDOW = "Generic" # synchronized (via a connection) to the active window, meaning it updates when the slider moves.

    # @property
    # def propertyName(self):
    #     return SynchronizedPlotMode.propertyNameList()[self]

    # # Static properties
    # @classmethod
    # def propertyNameList(cls):
    #     return cls.build_member_value_dict(['from','to',':'])


@metadata_attributes(short_name=None, tags=['raster', 'gui'], input_requires=[], output_provides=[], uses=['LiveWindowedData'], used_by=[], creation_date='2024-12-18 12:45', related_items=[])
class Spike2DRaster(SpecificDockWidgetManipulatingMixin, DynamicDockDisplayAreaOwningMixin, PyQtGraphSpecificTimeCurvesMixin, EpochRenderingMixin, Render2DScrollWindowPlotMixin, SpikeRasterBase):
    """ Displays a 2D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike2DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
        
        
    TODO: FATAL: The Spike2DRaster doesn't make use of the colors set in params or anything where the 3D does! Instead it's unique in that it stores a list of configs for each neuron. While this is a neat idea, it should be scrapped entirely for consistency.
    # self.params.config_items and self._build_cell_configs(...) called from self._buildGraphics(...)
    
    """
    
    # Application/Window Configuration Options:
    applicationName = 'Spike2DRaster'
    windowName = 'Spike2DRaster'
    
    # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False
    Includes2DActiveWindowScatter = True # Includes2DActiveWindowScatter: if True, it displays the main scatter plot for the active window.
    
    ## Scrollable Window Signals
    window_scrolled = QtCore.Signal(float, float) # overriding from Render2DScrollWindowPlotMixin - signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    
    # window_scrolled = QtCore.pyqtSignal(float, float) # signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    sigOnIntervalEnteredWindow = QtCore.Signal(object) # pyqtSignal(object)
    sigOnIntervalExitedindow = QtCore.Signal(object)
    sigRenderedIntervalsListChanged = QtCore.Signal(object) # EpochRenderingMixin conformance: signal emitted whenever the list of rendered intervals changed (add/remove). Added 2023-10-16 to prevent `AttributeError: 'Spike2DRaster' does not have a signal with the signature PyQt_PyObject)`
    # sigEmbeddedWidgetHierarchyChanged = QtCore.Signal(object) # emitted when the hierarchy of nested widgets changes, such as when a new dynamic matplotlib_render_plot_widget is added
    

    # DynamicDockDisplayAreaContentMixin Conformance Signals _____________________________________________________________ #
    sigDockAdded = QtCore.Signal(object, object) # (parent, Dock)
    sigDockModified = QtCore.Signal(object, object, object) # (parent, Dock, action)
    sigDockClosed = QtCore.Signal(object, object) # (parent, Dock)


    sigEmbeddedMatplotlibDockWidgetAdded = QtCore.Signal(object, object, object) # self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name]) -  emitted when a new matplotlib dock widget is added
    sigEmbeddedMatplotlibDockWidgetRemoved = QtCore.Signal(object, object) # (self, identifier: str)

    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance
    
    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration}',
                                                #    f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}']
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    ## FOR EpochRenderingMixin
    @property    
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        if self.params.get('custom_interval_rendering_plots', None) is not None:
            return self.params.custom_interval_rendering_plots
        else:
            return [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget] # for spike_raster_plt_2d
    
    
    ######  Get/Set Properties ######:

    ## FOR TimeCurvesViewMixin
    @property
    def floor_z(self):
        """The offset of the floor in the ordinate-axis. Which is actually the y-axis for a 2D plot """
        return 0
        

    # ==================================================================================================================== #
    # unit_sort_order support from Spike3DRaster                                                                           #
    # ==================================================================================================================== #

    @property
    def series_identity_y_values(self):
        """The series_identity_y_values property."""
        return self._series_identity_y_values

    def update_series_identity_y_values(self, debug_print=False):
        """ updates the fixed self._series_identity_y_values using the DataSeriesToSpatial.build_series_identity_axis(...) function.
        
        Should be called whenever:
        self.n_cells, 
        self.params.center_mode,
        self.params.bin_position_mode
        self.params.side_bin_margins
        self.unit_sort_order

        values change.
        """
        self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self._series_identity_lower_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        self._series_identity_upper_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells

        ## SORT: TODO: This sort condition seems to work and change the sort-order of the cells when self.unit_sort_order is updated... but the colors get all wonky and I'm uncertain if the configs are working correctly. Furthmore, it isn't clear that the spiking is any better aligned.
        # This might be overkill, idk
        # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs), self._series_identity_y_values)) # Using `self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs)` instead of just `self.fragile_linear_neuron_IDXs` should yield sorted results
        if not np.alltrue(self.unit_sort_order == self.fragile_linear_neuron_IDXs):
            if debug_print:
                print(f'update_series_identity_y_values(): building sorted version...')
            # Copy the `self.series_identity_y_values` and sort them according to `self.unit_sort_order`
            _sorted_map_values = self.series_identity_y_values[self.unit_sort_order].copy() # sort the y-values
            # Builds the sorted version by sorting the map values before building:
            self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, _sorted_map_values)) # Old way

        else:
            if debug_print:
                print(f'update_series_identity_y_values(): (self.unit_sort_order == self.fragile_linear_neuron_IDXs) (default sort).')
            self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self._series_identity_y_values)) # Old way 


    ## Required for DataSeriesToSpatialTransformingMixin
    def fragile_linear_neuron_IDX_to_spatial(self, fragile_linear_neuron_IDXs):
        """ transforms the fragile_linear_neuron_IDXs in fragile_linear_neuron_IDXs to a spatial offset (such as the y-positions for a 3D raster plot) """
        if self.series_identity_y_values is None:
            self.update_series_identity_y_values()
        fragile_linear_neuron_IDX_series_indicies = self.unit_sort_order[fragile_linear_neuron_IDXs] # get the appropriate series index for each fragile_linear_neuron_IDX given their sort order
        return self.series_identity_y_values[fragile_linear_neuron_IDX_series_indicies]


    # ==================================================================================================================== #
    # END unit_sort_order                                                                                                  #
    # ==================================================================================================================== #



    @property
    def menu_action_history_list(self) -> List:
        """The menu_action_history_list property."""
        # return self.ui.menus._menu_action_history_list # 2DRaster
        return PhoMenuHelper.try_get_menu_window(self).ui.menus._menu_action_history_list  # Window?
    @menu_action_history_list.setter
    def menu_action_history_list(self, value):
        # self.ui.menus._menu_action_history_list = value
        PhoMenuHelper.try_get_menu_window(self).ui.menus._menu_action_history_list = value
        


    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, **kwargs):
        # SpikeRasterBase.__init__(self=self, params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
        super(Spike2DRaster, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
        
        self.logger.info(f'Spike2DRaster.__init__(...)\t.applicationName: "{self.applicationName}"\n\t.windowName: "{self.windowName}")\n')
        
        # Init the TimeCurvesViewMixin for 3D Line plots:
        ### No plots will actually be added until self.add_3D_time_curves(plot_dataframe) is called with a valid dataframe.
        self.TimeCurvesViewMixin_on_init()
        
         # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.unit_sort_order_changed_signal.connect(self.on_unit_sort_order_changed)

        self.EpochRenderingMixin_on_init()

        if self.enable_show_on_init:
            self.show()
            
        # NOTE: It looks like this didn't work when called before self.show(), but worked when called from the Notebook. Might just be a timeing thing.
        ## Make sure to set the initial linear scroll region size/location to something reasonable and not cut-off so the user can adjust it:
        self._fix_initial_linearRegionLocation() # Implemented in Render2DScrollWindowPlotMixin, since it's the one that creates the Scrollwindow anyways

        ## Starts the delayed_gui_itemer which will run after 1-second to update the GUI:
        self._delayed_gui_timer = QtCore.QTimer(self)
        self._delayed_gui_timer.timeout.connect(self._run_delayed_gui_load_code)
        #Set the interval and start the timer.
        self._delayed_gui_timer.start(1000)
        
    
    def setup(self):
        self.logger.info(f'Spike2DRaster.setup()')
        
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        # self.app = pg.mkQApp("Spike2DRaster")
        self.app = pg.mkQApp(self.applicationName)
        

        # Config        
        """ Adds required params to self.params:
            spike_start_z (default -10.0): the z-offset of the start of the spikes
            spike_end_z (default -6.0): the z-offset of the end of the spikes
            center_mode (allowed: ['starting_at_zero', 'zero_centered']), (default 'zero_centered'): I believe how everything is layed out relative to the origin
            bin_position_mode (allowed: ['bin_center', 'left_edges']), (default 'left_edges'): specifies how the bins are positioned??
            
            axes_walls_z_height (default 20.0): the z-height of the axes plane box that frames the data
            axes_planes_floor_fixed_y_spacing (default 10.0): the spacing of grid lines along the y-axis that subdivide the floor axes plane (blue z-plane)
        
        """
        self.params.setdefault('useOpenGL', True)
        self.params.setdefault('enableExperimental', True)

        # Configure pyqtgraph config:
        try:
            import OpenGL
            # pg.setConfigOption('useOpenGL', True)
            pg.setConfigOption('useOpenGL', self.params.useOpenGL)
            pg.setConfigOption('enableExperimental', self.params.enableExperimental)
        except Exception as e:
            self.logger.error(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            print(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            
        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")
    
        # Config
        # self.params.center_mode = 'zero_centered'
        self.params.center_mode = 'starting_at_zero'
        self.params.bin_position_mode = 'bin_center'
        # self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)
        self.params.temporal_zoom_factor = 1.0        

        # Time Interval (epochs) legends:
        self.params.enable_time_interval_legend_in_right_margin = True
        
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
        # Determine the y-values corresponding to the series identity
        self._series_identity_y_values = None
        self._series_identity_lower_y_values = None
        self._series_identity_upper_y_values = None
        self.update_series_identity_y_values()

        # Build Required SpikesDf fields:
        # print(f'fragile_linear_neuron_IDXs: {self.fragile_linear_neuron_IDXs}, n_cells: {self.n_cells}')
        # self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins) # replaced by self._series_identity_y_values
        # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self._series_identity_y_values)) # Old way 
        # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs), self._series_identity_y_values)) # Using `self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs)` instead of just `self.fragile_linear_neuron_IDXs` should yield sorted results

        # Compute the y for all windows, not just the current one:
        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            self.logger.info('Spike2DRaster.setup(): adding "visualization_raster_y_location" column to spikes_df...')
            all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in self.spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. self.y doesn't change because self.params.center_mode, self.params.bin_position_mode, and self.params.side_bin_margins aren't expected to change. 
            self.logger.info('\tdone.')
            
        self.logger.debug(f'self.spikes_df.columns: {self.spikes_df.columns}')
        if 'visualization_raster_emphasis_state' not in self.spikes_df.columns:
            self.logger.info('Spike2DRaster.setup(): adding "visualization_raster_emphasis_state" column to spikes_df...')
            self.spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
            self.logger.info(f'\tdone.')
        else:
            self.logger.info('\t"visualization_raster_emphasis_state" column already exists.')

        self.EpochRenderingMixin_on_setup()

        # Required for Time Curves:
        self.params.time_curves_datasource = None # required before calling self._update_plot_ranges()
    
       
    @function_attributes(short_name='_build_cell_configs', tags=['config','private'], input_requires=['self.params.neuron_qcolors_map'], output_provides=['self.params.config_items', 'self.config_fragile_linear_neuron_IDX_map'],
        uses=['self.find_cell_ids_from_neuron_IDXs'], used_by=[], creation_date='2023-03-31 18:46')
    def _build_cell_configs(self):
        """ Adds the neuron/cell configurations that are used to color and format the scatterplot spikes and such. 
        Requires:
            self._series_identity_lower_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
            self._series_identity_upper_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        
        NOTE: on self.y vs (self._series_identity_lower_y_values, self._series_identity_upper_y_values): two ndarrays of the same length as self.y but they each express the start/end edges of each series as a ratio of the total.
            this means for example: 
                y:       [0.5, 1.5, 2.5, ..., 65.5, 66.5, 67.5]
                lower_y: [0.0, 0.0147059, 0.0294118, ..., 0.955882, 0.970588, 0.985294]
                upper_y: [0.0147059, 0.0294118, 0.0441176, ..., 0.970588, 0.985294, 1.0]

        Adds:
            self.params.config_items: IndexedOrderedDict
            self.config_fragile_linear_neuron_IDX_map: dict<self.fragile_linear_neuron_IDXs, self.params.config_items>
        
        Known Calls:
            From self._buildGraphics()
            From self.on_neuron_colors_changed(...) and self.on_unit_sort_order_changed(...)
        """
        
        # SpikeEmphasisState
        state_alpha = {SpikeEmphasisState.Hidden: 0.01,
                       SpikeEmphasisState.Deemphasized: 0.1,
                       SpikeEmphasisState.Default: 0.5,
                       SpikeEmphasisState.Emphasized: 1.0,
        }
        
        # state_color_adjust_fcns: functions that take the base color and call build_adjusted_color to get the adjusted color for each state
        state_color_adjust_fcns = {SpikeEmphasisState.Hidden: lambda x: build_adjusted_color(x, alpha_scale=0.01),
                        SpikeEmphasisState.Deemphasized: lambda x: build_adjusted_color(x, saturation_scale=0.35, value_scale=0.8, alpha_scale=0.1),
                        SpikeEmphasisState.Default: lambda x: build_adjusted_color(x, alpha_scale=0.95),
                        SpikeEmphasisState.Emphasized: lambda x: build_adjusted_color(x, value_scale=1.25, alpha_scale=1.0),
        }
        
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = IndexedOrderedDict()
        curr_neuron_ids_list = self.find_cell_ids_from_neuron_IDXs(self.fragile_linear_neuron_IDXs)
        
        # builds one config for each neuron color:
        for i, fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):
            curr_neuron_id = curr_neuron_ids_list[i] # aclu value
            
            curr_state_pen_dict = dict()
            for an_emphasis_state, alpha_value in state_alpha.items():
                curr_color = self.params.neuron_qcolors_map[fragile_linear_neuron_IDX]
                curr_color.setAlphaF(alpha_value)
                curr_color = state_color_adjust_fcns[an_emphasis_state](curr_color)
                curr_pen = pg.mkPen(curr_color)
                curr_state_pen_dict[an_emphasis_state] = curr_pen
            
            # curr_config_item = (i, fragile_linear_neuron_IDX, curr_state_pen_dict, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i]) # config item is just a tuple here

            # TEST: Seems like these other values are unused, and only curr_config_item[2] (containing the curr_state_pen_dict) is ever accessed in the subsequent functions.
            curr_config_item = (None, None, curr_state_pen_dict, None, None) # config item is just a tuple here
            self.params.config_items[curr_neuron_id] = curr_config_item # add the current config item to the config items 


        #!! SORT: TODO: CRITICAL: this is where I think we do the sorting! We leave everything else in the natural order, and then sort the `self.params.config_items.values()` in this map (assuming they're what are used:
        ## ORIGINAL Unsorted version:
        self.config_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self.params.config_items.values()))
        
        # ## Attempted sorted version -- NOTE -- DOES NOT WORK:
        # self.config_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, np.array(list(self.params.config_items.values()))[self.unit_sort_order])) # sort using the `unit_sort_order`

    @function_attributes(short_name=None, tags=['original', 'layout'], input_requires=[], output_provides=[], uses=[], used_by=['._buildGraphics'], creation_date='2025-05-12 17:13', related_items=[])
    def _buildGraphics_InternalLayout(self):
        """ Called only by `self._buildGraphics() and used only when `use_docked_pyqtgraph_plots == False`
        plots.main_plot_widget: 2D display 
            self.plots.scatter_plot: the active 2D display of the current window
        
        plots.background_static_scroll_window_plot: the static plot of the entire data (always shows the entire time range)
            Presents a linear scroll region over the top to allow the user to select the active window.
            
            
        Requires:
            self.ui.main_content_splitter
            self.params.custom_interval_rendering_plots
            
        Creates/Updates:
            self.plots.main_plot_widget
            self.plots.scatter_plot
            self.plots.background_static_scroll_window_plot
            
            
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomGraphicsLayoutWidget
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        
        self.logger.debug(f'\tSpike2DRaster._buildGraphics_InternalLayout()')
        
        ##### Main Raster Plot Content Top ##########
        self.ui.main_graphics_layout_widget = CustomGraphicsLayoutWidget()
        self.ui.main_graphics_layout_widget.setObjectName('main_graphics_layout_widget')
        self.ui.main_graphics_layout_widget.useOpenGL(True)
        self.ui.main_graphics_layout_widget.resize(1000,600)
        
        # add the GLViewWidget to the splitter
        self.ui.main_content_splitter.addWidget(self.ui.main_graphics_layout_widget)
        
        #### Build Graphics Objects ##### 
        # Add debugging widget:
        # self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        # self._series_identity_lower_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        # self._series_identity_upper_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        # self.update_series_identity_y_values()
        # self._build_cell_configs()
        
        ## New: Build a container layout to contain all elements that will represent the active window 
        self.params.main_graphics_active_window_container_layout_rowspan = 1
        self.ui.active_window_container_layout = self.ui.main_graphics_layout_widget.addLayout(row=1, col=0, rowspan=self.params.main_graphics_active_window_container_layout_rowspan, colspan=1)
        self.ui.active_window_container_layout.setObjectName('active_window_container_layout')
                        
        # Custom 2D raster plot:
        self.params.main_graphics_plot_widget_rowspan = 3 # how many rows the main graphics PlotItems should span
        # self.params.setdefault('main_graphics_plot_widget_rowspan', 1) # how many rows the main graphics PlotItems should span
        
        # curr_plot_row = 1
        if self.Includes2DActiveWindowScatter:
            ## Add these active window only plots to the active_window_container_layout

            self.plots.main_plot_widget = self.ui.active_window_container_layout.addPlot(row=1, col=0, rowspan=self.params.main_graphics_plot_widget_rowspan, colspan=1)            
            self.plots.main_plot_widget.setObjectName('main_plot_widget') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
            
            self.plots.main_plot_widget.setMinimumHeight(40.0) # used to be 500.0 and took up too much room
            
            # Common Tick Label
            vtick = QtGui.QPainterPath()
            vtick.moveTo(0, -0.5)
            vtick.lineTo(0, 0.5)

            self.plots.main_plot_widget.setLabel('left', 'Cell ID', units='')
            self.plots.main_plot_widget.setLabel('bottom', 'Time', units='s')
            self.plots.main_plot_widget.setMouseEnabled(x=False, y=False)
            self.plots.main_plot_widget.enableAutoRange(x=False, y=False)
            self.plots.main_plot_widget.setAutoVisible(x=False, y=False)
            self.plots.main_plot_widget.setAutoPan(x=False, y=False)
            self.plots.main_plot_widget.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
            
            # self.plots.main_plot_widget.disableAutoRange()
            self._update_plot_ranges()
            
            ## This scatter plot is the dynamic raster that "zooms" on adjustment of the lienar slider region. It is NOT static background raster that's rendered at the bottom of the window!
            self.plots.scatter_plot = pg.ScatterPlotItem(name='spikeRasterScatterPlotItem', pxMode=True, symbol=vtick, size=10, pen={'color': 'w', 'width': 2})
            self.plots.scatter_plot.setObjectName('scatter_plot')
            self.plots.scatter_plot.opts['useCache'] = True
            self.plots.main_plot_widget.addItem(self.plots.scatter_plot)
            _v_axis_item = Render2DNeuronIdentityLinesMixin.setup_custom_neuron_identity_axis(self.plots.main_plot_widget, self.n_cells)
                
        else:
            self.plots.main_plot_widget = None
            self.plots.scatter_plot = None

        
        # From Render2DScrollWindowPlotMixin:
        self.plots.background_static_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0, rowspan=1, colspan=1) # rowspan=self.params.main_graphics_plot_widget_rowspan
        self.plots.background_static_scroll_window_plot = self.ScrollRasterPreviewWindow_on_BuildUI(self.plots.background_static_scroll_window_plot)

        # self.ScrollRasterPreviewWindow_on_BuildUI()
        if self.Includes2DActiveWindowScatter:
            self.plots.scatter_plot.addPoints(self.plots_data.all_spots)
    
        self.params.custom_interval_rendering_plots.append(self.plots.background_static_scroll_window_plot)

        # custom_interval_rendering_plots = self.params.setdefault('custom_interval_rendering_plots', [self.plots.background_static_scroll_window_plot])
        if self.Includes2DActiveWindowScatter:
            if self.plots.main_plot_widget is not None:
                self.params.custom_interval_rendering_plots.append(self.plots.main_plot_widget)
            


    @function_attributes(short_name=None, tags=['docked', 'tracks'], input_requires=[], output_provides=[], uses=[], used_by=['._buildGraphics'], creation_date='2025-05-12 17:13', related_items=[])
    def _buildGraphics_DockedTracksLayout(self):
            """ Called only by `self._buildGraphics() and used only when `use_docked_pyqtgraph_plots == True`
            plots.main_plot_widget: 2D display 
                self.plots.scatter_plot: the active 2D display of the current window
            
            plots.background_static_scroll_window_plot: the static plot of the entire data (always shows the entire time range)
                Presents a linear scroll region over the top to allow the user to select the active window.
                
                

            
            """
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
            
            self.logger.debug(f'\tSpike2DRaster._buildGraphics_DockedTracksLayout()')
                                    
            #### Build Graphics Objects ##### 

            ## New: Build a container layout to contain all elements that will represent the active window 
            self.params.main_graphics_active_window_container_layout_rowspan = 1
            # self.ui.active_window_container_layout = self.ui.main_graphics_layout_widget.addLayout(row=1, col=0, rowspan=self.params.main_graphics_active_window_container_layout_rowspan, colspan=1)
            # self.ui.active_window_container_layout.setObjectName('active_window_container_layout')
                            
            # Custom 2D raster plot:
            self.plots.main_plot_widget = None
            self.plots.scatter_plot = None
            
            ## Done, ready to add docks
            # Intervals __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            _interval_tracks_out_dict = self.prepare_pyqtgraph_intervalPlot_tracks(enable_interval_overview_track=True, should_link_to_main_plot_widget=False)
            intervals_overview_dock_config, intervals_overview_dock, intervals_overview_time_sync_pyqtgraph_widget, intervals_overview_root_graphics_layout_widget, intervals_overview_plot_item = _interval_tracks_out_dict['interval_overview']
            interval_window_dock_config, intervals_dock, intervals_time_sync_pyqtgraph_widget, intervals_root_graphics_layout_widget, intervals_plot_item = _interval_tracks_out_dict['intervals']
            # _all_outputs_dict['_interval_tracks_out_dict'] = _interval_tracks_out_dict
            self.params.custom_interval_rendering_plots.append(intervals_overview_plot_item)
            self.params.custom_interval_rendering_plots.append(intervals_plot_item)

            # Rasters ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            _raster_overview_tracks_out_dict = self.prepare_pyqtgraph_rasterPlot_track(name_modifier_suffix='raster_overview', should_link_to_main_plot_widget=False, sync_mode=SynchronizedPlotMode.NO_SYNC, downsampling_rate=5)
            raster_overview_dock_config, raster_overview_time_sync_pyqtgraph_widget, raster_overview_root_graphics_layout_widget, raster_overview_plot_item, raster_overview_display_outputs_tuple = _raster_overview_tracks_out_dict['rasters[raster_overview]']
            self.plots.background_static_scroll_window_plot = raster_overview_plot_item
            self.plots.background_static_scroll_window_plot = self.ScrollRasterPreviewWindow_on_BuildUI(self.plots.background_static_scroll_window_plot)
            # self.sync_matplotlib_render_plot_widget('rasters[raster_overview]', sync_mode=SynchronizedPlotMode.NO_SYNC) # disable continued sync
            raster_overview_plot_item.setXRange(self.total_data_start_time, self.total_data_end_time, padding=0) ## global frame
            self.params.custom_interval_rendering_plots.append(self.plots.background_static_scroll_window_plot)

            if self.Includes2DActiveWindowScatter:
                _raster_tracks_out_dict = self.prepare_pyqtgraph_rasterPlot_track(name_modifier_suffix='raster_window', should_link_to_main_plot_widget=False, sync_mode=SynchronizedPlotMode.TO_WINDOW, downsampling_rate=1)
                raster_window_dock_config, raster_window_time_sync_pyqtgraph_widget, raster_window_root_graphics_layout_widget, raster_window_plot_item, raster_window_display_outputs_tuple = _raster_tracks_out_dict['rasters[raster_window]']
                raster_window_app, raster_window_win, raster_window_plots, raster_window_plots_data = raster_window_display_outputs_tuple                
                self.plots.main_plot_widget = raster_window_plot_item
                self.plots.scatter_plot = raster_window_plots.scatter_plot
                # self.params.custom_interval_rendering_plots.append(self.plots.main_plot_widget)
                # _all_outputs_dict['_raster_tracks_out_dict'] = _raster_tracks_out_dict
                if self.plots.scatter_plot is not None:
                    if self.plots_data.all_spots is not None:
                        self.plots.scatter_plot.addPoints(self.plots_data.all_spots)

            # END if use_docked_pyqtgraph_plots...


            
    @function_attributes(short_name=None, tags=['buildGraphics', 'layout'], input_requires=[], output_provides=[], uses=['._buildGraphics_InternalLayout', '._buildGraphics_DockedTracksLayout'], used_by=[], creation_date='2025-05-12 17:13', related_items=[])
    def _buildGraphics(self):
        """ 
        plots.main_plot_widget: 2D display 
            self.plots.scatter_plot: the active 2D display of the current window
        
        plots.background_static_scroll_window_plot: the static plot of the entire data (always shows the entire time range)
            Presents a linear scroll region over the top to allow the user to select the active window.
            
            
        Common:
            self.ui.wrapper_widget
            self.ui.dynamic_docked_widget_container
            self.ui.wrapper_layout
            self.ui.dynamic_docked_widget_container
            
        """
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        
        self.logger.debug(f'Spike2DRaster._buildGraphics()')
        # use_docked_pyqtgraph_plots: bool = self.params.setdefault('use_docked_pyqtgraph_plots', False)
        use_docked_pyqtgraph_plots: bool = self.params.setdefault('use_docked_pyqtgraph_plots', True)

        ## Common
        self.params.custom_interval_rendering_plots = []
        # For this 2D Implementation of TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin
        self.ui.main_time_curves_view_widget = None
        self.ui.main_time_curves_view_legend = None

        # From Render2DScrollWindowPlotMixin:
        self.EpochRenderingMixin_on_buildUI()
        
        self.update_series_identity_y_values()
        self._build_cell_configs()
        # self._update_plot_ranges()
    

        # Build Widgets ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # Create a QWidget to act as a wrapper
        self.ui.wrapper_widget = pg.QtWidgets.QWidget()
        self.ui.wrapper_widget.setObjectName("wrapper_widget")
        if not use_docked_pyqtgraph_plots:
            self.ui.wrapper_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        else:
            self.ui.wrapper_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            

        # Create a layout for the wrapper (you may want a different layout depending on your needs)
        self.ui.wrapper_layout = pg.QtWidgets.QVBoxLayout(self.ui.wrapper_widget)
        self.ui.wrapper_layout.setSpacing(0)
        self.ui.wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        ## Add the container to hold dynamic matplotlib plot widgets:
        self.ui.dynamic_docked_widget_container = NestedDockAreaWidget()
        self.ui.dynamic_docked_widget_container.setObjectName("dynamic_docked_widget_container")
        # self.ui.layout.addWidget(self.ui.dynamic_docked_widget_container, 1, 0) # Add the dynamic container as the second row
        # add the GLViewWidget to the splitter
        # self.ui.main_content_splitter.addWidget(self.ui.dynamic_docked_widget_container)

        # Add the container to the wrapper layout
        self.ui.wrapper_layout.addWidget(self.ui.dynamic_docked_widget_container)


        # add the splitter into your layout
        if not use_docked_pyqtgraph_plots:
            ## Internal Layout
            # create a splitter       
            self.ui.main_content_splitter = pg.QtWidgets.QSplitter(0)
            self.ui.main_content_splitter.setObjectName('main_content_splitter')
            # self.ui.main_content_splitter.setHandleWidth(10)
            self.ui.main_content_splitter.setHandleWidth(10)
            self.ui.main_content_splitter.setOrientation(0) # pg.Qt.Vertical        # Qt.Horizontal

            #TODO 2024-12-19 09:18: - [ ] This is where the handles become huge and RED!!
            self.ui.main_content_splitter.setStyleSheet("""
                    QSplitter::handle {
                        background: rgb(255, 0, 4);
                    }
                    QSplitter::handle:horizontal {
                        width: 2px;
                    }
                    QSplitter::handle:vertical {
                        height: 2px;
                    }
                """)
                    
            # Add the wrapper_widget to the splitter
            self.ui.main_content_splitter.addWidget(self.ui.wrapper_widget)
            self.ui.layout.addWidget(self.ui.main_content_splitter, 0, 0)  # add the splitter to the main layout at 0, 0
            
            # ## New: Build a container layout to contain all elements that will represent the active window 
            # self.params.main_graphics_active_window_container_layout_rowspan = 1
            # self.ui.active_window_container_layout = self.ui.main_graphics_layout_widget.addLayout(row=1, col=0, rowspan=self.params.main_graphics_active_window_container_layout_rowspan, colspan=1)
            # self.ui.active_window_container_layout.setObjectName('active_window_container_layout')
            
        else:
            ## DockedTracksLayout
            self.ui.layout.addWidget(self.ui.wrapper_widget, 0, 0)  # add the wrapper_widget (which contains the docked tracks to the main layout at 0, 0
            


        # From Render2DScrollWindowPlotMixin:
        # self.EpochRenderingMixin_on_buildUI()
        
        # self.Render2DScrollWindowPlot_on_window_update # register with the animation time window for updates for the scroller.
        # Connect the signals for the zoom region and the LinearRegionItem        
        self.rate_limited_signal_scrolled_proxy = pg.SignalProxy(self.window_scrolled, rateLimit=30, slot=self.update_zoomed_plot_rate_limited) # Limit updates to 30 Signals/Second

        # Required for dynamic matplotlib figures (2022-12-23 added, not sure how it relates to above):
        self._setupUI_matplotlib_render_plots()

        self.params.custom_interval_rendering_plots = []
        
        ## specific
        if not use_docked_pyqtgraph_plots:
            self._buildGraphics_InternalLayout()
        else:
            self._buildGraphics_DockedTracksLayout()
        

    
        






    def _run_delayed_gui_load_code(self):
        """ called when the self._delayed_gui_timer QTimer fires. """
        #Stop the timer.
        self._delayed_gui_timer.stop()
        print(f'Spike2DRaster._run_delayed_gui_load_code() called!')
        ## Make sure to set the initial linear scroll region size/location to something reasonable and not cut-off so the user can adjust it:
        self._fix_initial_linearRegionLocation() # Implemented in Render2DScrollWindowPlotMixin, since it's the one that creates the Scrollwindow anyways

        
    ###################################
    #### EVENT HANDLERS
    ##################################
    
    def _update_plot_ranges(self):
        """
        I believe this runs only once to setup the bounds of the plot.
        TODO: TODO-DOC: Figure out when this is called and what its purpose is
        
        """
        # self.plots.main_plot_widget.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
        # self.plots.main_plot_widget.setXRange(0.0, +self.temporal_axis_length, padding=0)
        # self.plots.main_plot_widget.setYRange(self.y[0], self.y[-1], padding=0)
        # self.plots.main_plot_widget.disableAutoRange()
        if self.Includes2DActiveWindowScatter:
            if self.plots.main_plot_widget is not None:
                self.plots.main_plot_widget.disableAutoRange('xy')
                ## TODO: BUG: CONFIRMED: This is for-sure a problem. In the .ScrollRasterPreviewWindow_on_BuildUI(...) where the linear region widget (scroll_window_region) is built, those x-values are definintely timestamps and start slightly negative. This is why the widget is getting cut-off
                """ From the first setup:
                    # Setup range for plot:
                    earliest_t, latest_t = self.spikes_window.total_df_start_end_times
                    background_static_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
                    background_static_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)

                Here it looks like I'm trying to use some sort of reletive x-coordinates (as I noted that I did in self._series_identity_lower_y_values, self._series_identity_upper_y_values?)
                
                OOPS, back-up, this is the main_plot_widget (that should be displaying the contents of the window above), not the same as the static background plot that displays all time.
                """
                        
                # # Get updated time window
                # updated_time_window = self.spikes_window.active_time_window # (30.0, 930.0) ## CHECKL this might actually be invalid at this timepoint, idk
                # earliest_t, latest_t = updated_time_window
                # resolved_start_x = np.nanmin(earliest_t, 0.0)
                # print(f'resolved_start_x: {resolved_start_x}')
                # resolved_end_x = (resolved_start_x+self.temporal_axis_length) # only let it go to the start_x + its appropriate length, otherwise it'll be too long?? Maybe I should actually use the window's end
                # print(f'resolved_end_x: {resolved_end_x}')
                # self.plots.main_plot_widget.setRange(xRange=[resolved_start_x, resolved_end_x], yRange=[self.y[0], self.y[-1]])
                # ## NOW I THINK THIS IS JUST THE ZOOMED PLOT AND NOT THE REASON THE LINEAR SCROLL REGION is cut off
                
                # self.plots.main_plot_widget.setRange(xRange=[0.0, +self.temporal_axis_length], yRange=[self._series_identity_y_values[0], self._series_identity_y_values[-1]]) # After all this, I've concluded that it was indeed correct!
                self.plots.main_plot_widget.setRange(xRange=[0.0, +self.temporal_axis_length], yRange=[min(self._series_identity_y_values), max(self._series_identity_y_values)]) # After all this, I've concluded that it was indeed correct!
                _v_axis_item = Render2DNeuronIdentityLinesMixin.setup_custom_neuron_identity_axis(self.plots.main_plot_widget, self.n_cells)
    
    
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update() # Don't think this does much here
        # self.EpochRenderingMixin_on_window_update()
        
    
    @pyqtExceptionPrintingSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        # print(f'lower_y: {lower_y}\n upper_y: {upper_y}')
        pass


    def _update_plots(self):
        """
        Seems to be called every time the timeline is scrolled at least.

        
        """
        self.logger.debug(f'Spike2DRaster._update_plots()')
        if self.enable_debug_print:
            print(f'Spike2DRaster._update_plots()')
        # assert (len(self.ui.plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.plots): {len(self.ui.plots)} and self.n_cells: {self.n_cells}!"
        # build the position range for each unit along the y-axis:
        # self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self.update_series_identity_y_values()

        # update the current scroll region:
        # self.ui.scroll_window_region.setRegion(updated_time_window)
        
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update() # Don't think this does much here
        

    @pyqtExceptionPrintingSlot(object)
    def update_zoomed_plot_rate_limited(self, evt):
        min_t, max_t = evt ## using signal proxy turns original arguments into a tuple
        self.update_zoomed_plot(min_t, max_t)


    @pyqtExceptionPrintingSlot(float, float)
    def update_zoomed_plot(self, min_t, max_t):
        """ update the zoomed plot, the spikes_window, and update the dependent curves
        
        """
        # Update the main_plot_widget:
        if self.Includes2DActiveWindowScatter:
            self.plots.main_plot_widget.setXRange(min_t, max_t, padding=0)

        # self.render_window_duration = (max_x - min_x) # update the render_window_duration from the slider width
        scroll_window_width = max_t - min_t
        # print(f'min_x: {min_x}, max_x: {max_x}, scroll_window_width: {scroll_window_width}') # min_x: 59.62061245756003, max_x: 76.83228787177144, scroll_window_width: 17.211675414211413

        # Update GUI if we have one:
        if self.WantsRenderWindowControls:
            self.ui.spinTemporalZoomFactor.setValue(1.0)
            self.ui.spinRenderWindowDuration.setValue(scroll_window_width)
            
        # Finally, update the actual spikes_window. This is the part that updates the 3D Raster plot because we bind to this window's signal
        # self.spikes_window.update_window_start(min_t)
        
        # Here is the main problem: The duration and window end-time aren't being updated
        self.spikes_window.update_window_start_end(min_t, max_t)
        
        
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update()
        
        
        
    @pyqtExceptionPrintingSlot(float, float)
    def update_scroll_window_region(self, new_start, new_end, block_signals: bool=True):
        """ called to update the interactive scrolling window control
        
        PUBLIC: primary update function
        
        
        """
        if block_signals:
            self.ui.scroll_window_region.blockSignals(True) # Block signals so it doesn't recursively update
        self.ui.scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
        if block_signals:
            self.ui.scroll_window_region.blockSignals(False)
        


    def update(self, sort_changed=True, colors_changed=True):
        """ refreshes the raster when the colors or sort change. 
        
        """
        if sort_changed:
            # rebuild the position range for each unit along the y-axis:
            self.update_series_identity_y_values()
            all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in self.spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y
            colors_changed = True # colors always changed when sort changes
            
        if colors_changed:
            ## Rebuild Raster Plot Points:
            self._build_cell_configs()
            # ALL Spikes in the preview window:
            self.plots_data.all_spots = self._build_all_spikes_all_spots()
            # Update preview_overview_scatter_plot
            self.update_rasters()
        

        
    # unit_sort_order_changed_signal
    @pyqtExceptionPrintingSlot(object)
    def on_unit_sort_order_changed(self, new_sort_order):
        ## TODO: copied from Spike3DRaster but untested
        print(f'unit_sort_order_changed_signal(new_sort_order: {new_sort_order})')        
        self.update(sort_changed=True, colors_changed=True)
        print('\t done.')


    @pyqtExceptionPrintingSlot(object)
    def on_neuron_colors_changed(self, neuron_id_color_update_dict):
        """ Called when the neuron colors have finished changing (changed) to update the rendered elements.
        
        Inputs:
            neuron_id_color_update_dict: a neuron_id:QColor dictionary
        Updates:
            self.plots_data.all_spots
            
        """
        print(f'Spike2DRaster.neuron_id_color_update_dict: {neuron_id_color_update_dict}')
        self.update(sort_changed=False, colors_changed=True)
        
        
    @function_attributes(short_name=None, tags=['epoch', 'interval', 'find', 'window'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-19 11:11', related_items=[])
    def find_event_intervals_in_active_window(self, included_series_names: Optional[List[str]]=None, debug_print=False) -> Dict[str, pd.DataFrame]:
        """find the events/intervals that are within the currently active render window:
        """
        if included_series_names is None:
            ## include all
            included_series_names = self.rendered_epoch_series_names

        ## Get current time window:
        curr_time_window = self.animation_active_time_window.active_time_window # (45.12114057149739, 60.12114057149739)
        start_t, end_t = curr_time_window
        if debug_print:
            print(f'curr_time_window: {curr_time_window}')

        active_window_series_events_dict: Dict[str, pd.DataFrame] = {}
        for series_name, series_datasource in get_dict_subset(self.interval_datasources, subset_includelist=self.rendered_epoch_series_names).items():
            if debug_print:
                print(f'series_name: {series_name}, series_datasource: {series_datasource}')
            if series_name in included_series_names:
                ## make sure series is included:              
                active_window_series_events_dict[series_name] = series_datasource.get_updated_data_window(new_start=start_t, new_end=end_t)
            
        return active_window_series_events_dict



    # ==================================================================================================================== #
    # State Save/Restore                                                                                                   #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['TODO', 'UNFINISHED', 'save_state', 'renderables', 'restore'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-03 05:23', related_items=[])
    def save_state_active_renderables(self, debug_print=True):
        """ Called to capture the currently added renderables, their customized visual appearance and layout, etc so that they can be restored later by calling `self.perform_restore_renderables(...)` with the output state of this function.
        TODO: not yet complete    

        Usage:

            saved_state_active_renderables = active_2d_plot.save_state_active_renderables(debug_print=True)
            saved_state_active_renderables
            # plots_data: ['name', 'all_spots', 'interval_datasources']

        """
        # # from 'plots' array:
        # epochs = self.plots.rendered_epochs
        # curves = self.plots.time_curves
        
        ## Epoch/Interval Rectangles:
        interval_datasource_names = self.interval_datasource_names # ['CustomPBEs', 'PBEs', 'Ripples', 'Laps', 'Replays', 'SessionEpochs']
        if debug_print:
            print(f'interval_datasource_names: {interval_datasource_names}')
        restore_epoch_menu_commands = [f'AddTimeIntervals.{a_name}' for a_name in interval_datasource_names]
        if debug_print:
            print(f'restore_epoch_menu_commands: {restore_epoch_menu_commands}')
        add_renderables_menu = self.ui.menus.custom_context_menus.add_renderables[0].programmatic_actions_dict
        for a_command in restore_epoch_menu_commands:
            if a_command not in add_renderables_menu:
                print(f"WARNING: command '{a_command}' is not present in add_renderables_menu, so restore will not work for this item!")
            # add_renderables_menu[a_command].trigger()

        # Capture the curve positions and such to restore their position:
        all_series_positioning_dfs, all_series_compressed_positioning_dfs, all_series_compressed_positioning_update_dicts = self.recover_interval_datasources_positioning_properties(debug_print=False)
        # Can be restored with:  
        # all_series_compressed_positioning_update_dicts = { 'SessionEpochs': {'y_location': -2.916666666666667, 'height': 2.0833333333333335},
        # 'Laps': {'y_location': -7.083333333333334, 'height': 4.166666666666667},
        # 'PBEs': {'y_location': -11.666666666666668, 'height': 4.166666666666667},
        # 'Ripples': {'y_location': -15.833333333333336, 'height': 4.166666666666667},
        # 'Replays': {'y_location': -20.000000000000004, 'height': 4.166666666666667}}
        # active_2d_plot.update_rendered_intervals_visualization_properties(all_series_compressed_positioning_update_dicts)

        restore_dict = {'epoch_menu_commands': restore_epoch_menu_commands,
                    'epoch_updating_dicts': all_series_compressed_positioning_update_dicts}
        # return restore_epoch_menu_commands, all_series_compressed_positioning_update_dicts
        
        return restore_dict
            

    def perform_restore_renderables(self, saved_state_active_renderables, debug_print=True):
        """ restore the renderables state saved by `saved_state_active_renderables = save_state_active_renderables(...)` 
        TODO: not yet complete

        """
        add_renderables_menu = self.ui.menus.custom_context_menus.add_renderables[0].programmatic_actions_dict
        restore_epoch_menu_commands = saved_state_active_renderables.get('epoch_menu_commands', [])
        for a_command in restore_epoch_menu_commands:
            if a_command not in add_renderables_menu:
                print(f"WARNING: command '{a_command}' is not present in add_renderables_menu, so restore will not work for this item!")
            else:
                add_renderables_menu[a_command].trigger()
        ## Restore Positions/Visualization Appearance:
        restore_epoch_positions_dict = saved_state_active_renderables.get('epoch_updating_dicts', [])
        self.update_rendered_intervals_visualization_properties(restore_epoch_positions_dict)

    
        


    ######################################################
    # EpochRenderingMixin Convencince methods:
    #####################################################
    def _perform_add_render_item(self, a_plot, a_render_item):
        """Performs the operation of adding the render item from the plot specified

        Args:
            a_render_item (_type_): _description_
            a_plot (_type_): _description_
        """
        a_plot.addItem(a_render_item) # 2D (PlotItem)
        
        
    def _perform_remove_render_item(self, a_plot, a_render_item):
        """Performs the operation of removing the render item from the plot specified

        Args:
            a_render_item (IntervalRectsItem): _description_
            a_plot (PlotItem): _description_
        """
        a_plot.removeItem(a_render_item) # 2D (PlotItem)
        
        
    def add_laps_intervals(self, sess, **kwargs):
        """ Convenince method to add the Laps rectangles to the 2D Plots 
            NOTE: sess can be a DataSession, a Laps object, or an Epoch object containing Laps directly.
            active_2d_plot.add_PBEs_intervals(sess)
        """
        laps_interval_datasource = Laps2DRenderTimeEpochs.build_render_time_epochs_datasource(sess.laps.as_epoch_obj(), **({'series_vertical_offset': 42.0, 'series_height': 1.0} | kwargs))
        self.add_rendered_intervals(laps_interval_datasource, name='Laps', debug_print=False) # removes the rendered intervals
        
    def remove_laps_intervals(self):
        self.remove_rendered_intervals('Laps', debug_print=False)
        
    def add_PBEs_intervals(self, sess, **kwargs):
        """ Convenince method to add the PBE rectangles to the 2D Plots 
            NOTE: sess can be a DataSession, or an Epoch object containing PBEs directly.
        """
        new_PBEs_interval_datasource = PBE_2DRenderTimeEpochs.build_render_time_epochs_datasource(sess.pbe, **({'series_vertical_offset': 43.0, 'series_height': 1.0} | kwargs)) # new_PBEs_interval_datasource
        self.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', debug_print=False) # adds the rendered intervals

    def remove_PBEs_intervals(self):
        self.remove_rendered_intervals('PBEs', debug_print=False)
        

    # ==================================================================================================================== #
    # Legends                                                                                                              #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['legend'], input_requires=[], output_provides=[], uses=[], used_by=['build_or_update_all_epoch_interval_rect_legends'], creation_date='2024-07-01 18:29', related_items=[])
    def _build_or_update_epoch_interval_rect_legend(self, parent_item):
        """ Build a legend for a single plot each of the epoch rects 
    
        parent_item = self.ui.main_plot_widget.graphicsItem()

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import CustomLegendItemSample
        
        # Add legend inside the plot boundaries
        legend_size = None # auto-sizing legend to contents
        legend = pg.LegendItem(legend_size, offset=(-30, -30))  # Negative offset (x, y) from bottom-right corner
        legend.setParentItem(parent_item)
        legend.anchor((1, 1), (1, 1))  # Anchors the legend to the bottom-right corner
        legend.setSampleType(CustomLegendItemSample)
        return legend
    

    @function_attributes(short_name=None, tags=['legend'], input_requires=[], output_provides=[], uses=['_build_or_update_epoch_interval_rect_legend'], used_by=[], creation_date='2024-07-01 18:29', related_items=[])
    def build_or_update_all_epoch_interval_rect_legends(self):
        """ Build a legend for each of the subplots. 

        active_2d_plot.build_or_update_all_epoch_interval_rect_legends()

        """
        from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import RenderedEpochsItemsContainer
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, CustomLegendItemSample

        ## Try to get existing legends:
        legends_dict = self.plots.get('legends', None)
        if legends_dict is None:
            ## create new container
            self.plots['legends'] = RenderPlots(name='legends')
            legends_dict = self.plots['legends']

        assert legends_dict is not None
        ## OUTPUTS: legends_dict

        previously_encountered_plot_items = []
        interval_info = self.get_all_rendered_intervals_dict()
        if self.enable_debug_print:
            print(f'=== BEGIN')
        for a_name, an_intervals_dict in interval_info.items():
            # is_first_iteration_on_plot = (i
            if self.enable_debug_print:
                print(f'a_name: {a_name}:')
            for a_plot_name, a_plotted_intervals in an_intervals_dict.items():
                if self.enable_debug_print:
                    print(f'\ta_plot_name: {a_plot_name}, a_plotted_intervals: {a_plotted_intervals}, type(a_plotted_intervals): {type(a_plotted_intervals)}')                    
                is_included_in_legend: bool = (a_plot_name in self.plots)
                if is_included_in_legend:
                    a_target_plot = self.plots[a_plot_name] # KeyError: 'RootPlot'
                    # if a_plot_name == target_plot_name:
                    ## Here's the object, add it to the legend
                    a_legend = legends_dict.get(a_target_plot, None) ## get the legend for this plot
                    if a_legend is None:
                        ## create a legend:
                        legends_dict[a_target_plot] = self._build_or_update_epoch_interval_rect_legend(a_target_plot.graphicsItem())
                        a_legend = legends_dict[a_target_plot]
                    else:
                        # reuse the legend
                        # legends_dict[a_target_plot].clear() ## clear any existing items
                        pass
                    assert a_legend is not None
                    # end if a_legend  
                    if (a_plot_name not in previously_encountered_plot_items):
                        # first time for this plot
                        a_legend.clear() ## clear

                        needs_legend_margin: bool = (len(interval_info) > 0)

                        ## Increase the right margin:
                        if (self.params.enable_time_interval_legend_in_right_margin and needs_legend_margin):
                            ## Increase the right margin:
                            a_target_plot.layout.setContentsMargins(0, 0, 300, 0)  # left, top, right, bottom
                        else:
                            a_legend_plot.layout.setContentsMargins(0, 0, 0, 0)  # left, top, right, bottom
                                
                    ## END if (a_plot_name not in previously_encountered_plot_items)..
                    a_legend.addItem(a_plotted_intervals, a_name) ## add the item to the legend
                    previously_encountered_plot_items.append(a_plot_name)
                # END if is_included_in_legend
        return legends_dict
    
    @function_attributes(short_name=None, tags=['legend', 'remove_all', 'remove'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-01 18:29', related_items=[])
    def remove_all_epoch_interval_rect_legends(self):
        """ removes any created legends """
        ## Remove the legends
        legends = self.plots.get('legends', None)
        if legends is not None:
            for a_legend_plot, a_legend in legends.data_items():
                if self.enable_debug_print:
                    print(f'a_legend_plot: {a_legend_plot}, a_legend: {a_legend}')
                if a_legend is not None:
                    a_legend.clear() ## clear the legend items
                    try:
                        a_legend.setParentItem(None)
                    except BaseException as err:
                        if self.enable_debug_print:
                            print(f'err: {err}')
                        pass
                    
                    # a_legend_plot.removeItem(a_legend)
                a_legend_plot.layout.setContentsMargins(0, 0, 0, 0)  # left, top, right, bottom

            legends.clear()
            del self.plots['legends']
            


    ######################################################
    # TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin specific overrides for 2D:
    """ 
    As soon as the first 2D Time Curve plot is needed, it creates:
        self.ui.main_time_curves_view_widget - PlotItem by calling add_separate_render_time_curves_plot_item(...)
    
    main_time_curves_view_widget creates new PlotDataItems by calling self.ui.main_time_curves_view_widget.plot(...)
        This .plot(...) command can take either: 
            .plot(x=x, y=y)
            .plot(ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]
            
    """
    
    @property
    def time_curve_render_dimensionality(self) -> int:
        """ the dimensionality of the rendered time curves. (e.g. 2 for SpikeRaster2D, 3 for SpikeRaster3D, SpikeRaster3DVedo """
        return 2
    
    
    #####################################################
    def clear_all_3D_time_curves(self):
        for (aUID, plt) in self.plots.time_curves.items():
            self.ui.main_time_curves_view_widget.removeItem(plt) # this should automatically work for 2D curves as well
            # plt.delete_later() #?
            
        self.ui.main_time_curves_view_legend.clear() # remove all items from the legend
        # Clear the dict
        self.plots.time_curves.clear()
        ## This part might be 3D only, but we do have a working 2D version so maybe just bring that in?
        self.remove_3D_time_curves_baseline_grid_mesh() # from Render3DTimeCurvesBaseGridMixin
        
    def update_3D_time_curves(self):
        """ initialize the graphics objects if needed, or update them if they already exist. """
        if self.params.time_curves_datasource is None:
            return
        elif self.params.time_curves_no_update:
            # don't update because we're in no_update mode
            print(f'')
            return
        else:
            # Common to both:
            # Get current plot items:
            curr_plot3D_active_window_data = self.params.time_curves_datasource.get_updated_data_window(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time) # get updated data for the active window from the datasource # if we want the data from the whole time, we aren't getting that here unfortunately
            
            is_data_series_mode = self.params.time_curves_datasource.has_data_series_specs # True for SpikeRaster2D
            if is_data_series_mode:
                data_series_spaital_values_list = self.params.time_curves_datasource.data_series_specs.get_data_series_spatial_values(curr_plot3D_active_window_data)
                num_data_series = len(data_series_spaital_values_list)
            else:
                # old compatibility mode:
                num_data_series = 1

            # curr_data_series_index = 0
            # Loop through the active data series:                
            for curr_data_series_index in np.arange(num_data_series):
                # Data series mode:
                if is_data_series_mode:
                    # Get the current series:
                    curr_data_series_dict = data_series_spaital_values_list[curr_data_series_index]
                    
                    curr_plot_column_name = curr_data_series_dict.get('name', f'series[{curr_data_series_index}]') # get either the specified name or the generic 'series[i]' name otherwise
                    curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
                    curr_plot_legend_name = self.params.time_curves_datasource.data_column_names[curr_data_series_index] # ['lin_pos', 'x', 'y']
                    
                    # points for the current plot:
                    pts = np.column_stack([curr_data_series_dict['x'], curr_data_series_dict['y'], curr_data_series_dict['z']])
                    
                    # Extra options:
                    # color_name = curr_data_series_dict.get('color_name','white')
                    extra_plot_options_dict = {'color_name':curr_data_series_dict.get('color_name', 'white'),
                                               'color':curr_data_series_dict.get('color', None),
                                               'line_width':curr_data_series_dict.get('line_width', 0.5),
                                               'z_scaling_factor':curr_data_series_dict.get('z_scaling_factor', 0.5),
                                               'legend_name':curr_data_series_dict.get('legend_name', curr_plot_legend_name)
                                               }
                    
                else:
                    raise NotImplementedError # gave up
                
                # outputs of either mode are curr_plot_name, pts
                curr_plt = self._build_or_update_time_curves_plot(curr_plot_name, pts, **extra_plot_options_dict)
                # end for curr_data_series_index in np.arange(num_data_series)

            self.add_3D_time_curves_baseline_grid_mesh() # from Render3DTimeCurvesBaseGridMixin

    def _build_or_update_time_curves_legend(self, parent_item):
        """ Build a legend for each of the curves 
    
        parent_item = self.ui.main_time_curves_view_widget.graphicsItem()

        """
        # legend_size = (80,60) # fixed size legend
        legend_size = None # auto-sizing legend to contents
        legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
        legend.setParentItem(parent_item)

        # desired_series_legend_names = list(self.params.time_curves_datasource.data_column_names) # ['lin_pos', 'x', 'y']
        # for legend_name, (curve_name, curveDataItem) in zip(desired_series_legend_names, self.plots['time_curves'].items()):
        #     print(f'legend_name: {legend_name}, curve_name: {curve_name}')
        #     legend.addItem(curveDataItem, legend_name)
        return legend
    

    def _build_or_update_time_curves_plot(self, plot_name, points, **kwargs):
        """ For 2D
        uses or builds a new self.ui.main_time_curves_view_widget, which the item is added to
        
        """
        if self.ui.main_time_curves_view_widget is None:
            # needs to build the primary 2D time curves plotItem:
            print(f'Spike2DRaster created a new self.ui.main_time_curves_view_widget for TimeCurvesViewMixin plots!')
            # row=0 adds above extant plot
            # row_index = (self.params.main_graphics_plot_widget_rowspan * 2)+1 # row 2 if they were all rowspan 2
            row_index = None # just auto get the next index
            self.ui.main_time_curves_view_widget = self.create_separate_render_plot_item(row=row_index, col=0, rowspan=1, colspan=1, name='new_curves_separate_plot') # PlotItem
            # self.ui.main_time_curves_view_legend = self._build_or_update_time_curves_legend()
        
        # build the plot arguments (color, line thickness, etc)        
        plot_args = ({'color_name':'white','line_width':0.5,'z_scaling_factor':1.0} | kwargs)
        
        curr_plot_legend_name = plot_args.pop('legend_name', None) # See if a legend entry is needed for this plot
        if curr_plot_legend_name is not None:
            if self.ui.main_time_curves_view_legend is None:
                # build the legend if needed
                self.ui.main_time_curves_view_legend = self._build_or_update_time_curves_legend(self.ui.main_time_curves_view_widget.graphicsItem())

        ## Drop the y-value from the 3D version to get the appropriate 2D coordinates (x,y)
        if np.shape(points)[1] == 3:
            # same data from 3D version, drop the y-value accordingly:
            """
                points: (N, 3)
                # t/x, _, 'y' 
                array([[-7.47296, -35, 0.931493],
                    [-7.43977, -35, 0.931998],
                    ...
            """
            points = points[:, [0, 2]]
        assert np.shape(points)[1] == 2, f"points must be (N, 2) but it instead {np.shape(points)}"

        if plot_name in self.plots.time_curves:
            # Plot already exists, update it instead.
            plt = self.plots.time_curves[plot_name]
            plt.setData(points)
            if curr_plot_legend_name is not None:
                # Update the legend entry:
                curr_label = self.ui.main_time_curves_view_legend.getLabel(plt)
                curr_label.setText(curr_plot_legend_name) # update the legend name if needed
                
        else:
            # plot doesn't exist, built it fresh.
            
            line_color = plot_args.get('color', None)
            if line_color is None:
                # if no explicit color value is provided, build a new color from the 'color_name' key, or if that's missing just use white.
                line_color = pg.mkColor(plot_args.setdefault('color_name', 'white'))
                line_color.setAlphaF(0.8)

            # Note .plot(...) seems to allow more options than .addLine(...)
            # curr_plt = self.ui.main_time_curves_view_widget.addLine(x=curr_data_series_dict['x'], y=curr_data_series_dict['y'])
            plt = self.ui.main_time_curves_view_widget.plot(points, pen=line_color, name=plot_name) # TODO: is this the slow version of name =?
            # end for curr_data_series_index in np.arange(num_data_series)
            self.plots.time_curves[plot_name] = plt # add it to the dictionary.
            
            if curr_plot_legend_name is not None:
                # Create the legend entry
                self.ui.main_time_curves_view_legend.addItem(plt, curr_plot_legend_name)
            
            # TODO: set line_width?
            # TODO: scaling like the 3D version?
            
        return plt
    

    @property
    def active_embedded_track_pyqtgraph_time_sync_widgets(self) -> Dict[str, PyqtgraphTimeSynchronizedWidget]:
        """The Dict containing the embedded PyqtgraphTimeSynchronizedWidget belonging to various tracks."""
        return {identifier:active_matplotlib_view_widget for identifier, active_matplotlib_view_widget in self.ui.matplotlib_view_widgets.items() if active_matplotlib_view_widget.is_pyqtgraph_based()}


    @function_attributes(short_name=None, tags=['pyqtgraph', 'render_plot_group'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 04:59', related_items=[])
    def create_separate_render_plot_item(self, row=None, col=None, rowspan=1, colspan=1, name='new_curves_separate_plot'):
        """ Adds a separate pyqtgraph independent plot for epoch time rects to the 2D plot above the others:
        
        Requires:
            active_2d_plot.ui.main_graphics_layout_widget <GraphicsLayoutWidget>
            
        Returns:
         new_curves_separate_plot: a PlotItem
            
        """
        use_docked_pyqtgraph_plots: bool = self.params.use_docked_pyqtgraph_plots
        if use_docked_pyqtgraph_plots:
            a_time_sync_pyqtgraph_widget, root_graphics_layout_widget, plot_item, dDisplayItem = self.add_new_embedded_pyqtgraph_render_plot_widget(name=name, dockSize=(500,50))
            target_graphics_layout_widget = root_graphics_layout_widget
        else:
            # main_graphics_layout_widget = self.ui.main_graphics_layout_widget # GraphicsLayoutWidget
            target_graphics_layout_widget = self.ui.active_window_container_layout # GraphicsLayoutWidget
            
        # self.ui.active_window_container_layout.
        new_curves_separate_plot = target_graphics_layout_widget.addPlot(row=row, col=col, rowspan=rowspan, colspan=colspan) # PlotItem
        new_curves_separate_plot.setObjectName(name)

        # Setup axes bounds for the bottom windowed plot:
        new_curves_separate_plot.hideAxis('left')
        # new_curves_separate_plot.showAxis('left')
        new_curves_separate_plot.hideAxis('bottom') # hide the shared time axis since it's synced with the other plot
        # new_curves_separate_plot.showAxis('bottom')
        
        # Setup axes bounds for the bottom windowed plot:
        # background_static_scroll_window_plot.setLabel('bottom', 'Time', units='s')
        # new_curves_separate_plot.setMouseEnabled(x=False, y=False)
        new_curves_separate_plot.setMouseEnabled(x=False, y=True) ## unusual
        # new_curves_separate_plot.disableAutoRange('xy')
        # background_static_scroll_window_plot.enableAutoRange(x=False, y=False)
        # new_curves_separate_plot.setAutoVisible(x=False, y=False)
        # new_curves_separate_plot.setAutoPan(x=False, y=False)
        
        # Setup range for plot:
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        new_curves_separate_plot.setXRange(earliest_t, latest_t, padding=0)
        # new_curves_separate_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)
        
        
        # # setup the new_curves_separate_plot to have a linked X-axis to the other scroll plot:
        main_plot_widget = self.plots.main_plot_widget # PlotItem
        new_curves_separate_plot.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)
        
        return new_curves_separate_plot
        

    @function_attributes(short_name=None, tags=['pyqtgraph', 'render_plot_group'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 04:59', related_items=[])
    def remove_separate_render_plot_items(self, separate_plot_item):
        """ removes the PlotItem separate_plot_item created by calling self.create_separate_render_plot_item(...) from the layout. """
        use_docked_pyqtgraph_plots: bool = self.params.use_docked_pyqtgraph_plots
        if use_docked_pyqtgraph_plots:
            ## how do I find the plot?
            active_time_sync_pyqtgraph_widgets: Dict[str, PyqtgraphTimeSynchronizedWidget] = {identifier:active_matplotlib_view_widget for identifier, active_matplotlib_view_widget in self.ui.matplotlib_view_widgets.items() if active_matplotlib_view_widget.is_pyqtgraph_based()}
            
            # active_time_sync_pyqtgraph_plot_items = {identifier:active_matplotlib_view_widget.getRootPlotItem() for identifier, active_matplotlib_view_widget in active_time_sync_pyqtgraph_widgets.items()}
            active_time_sync_pyqtgraph_plot_item_to_layout_map = {active_matplotlib_view_widget.getRootPlotItem():active_matplotlib_view_widget.getRootGraphicsLayoutWidget() for identifier, active_matplotlib_view_widget in active_time_sync_pyqtgraph_widgets.items()}
            assert separate_plot_item in active_time_sync_pyqtgraph_plot_item_to_layout_map, f"active_time_sync_pyqtgraph_plot_item_to_layout_map: {active_time_sync_pyqtgraph_plot_item_to_layout_map}, separate_plot_item: {separate_plot_item}, active_time_sync_pyqtgraph_widgets: {active_time_sync_pyqtgraph_widgets}"
            target_graphics_layout_widget = active_time_sync_pyqtgraph_plot_item_to_layout_map[separate_plot_item]
            assert target_graphics_layout_widget is not None            
            # raise NotImplementedError(f'Not finished, how do I find the embedding widget from the plot item?')
            # a_time_sync_pyqtgraph_widget, root_graphics_layout_widget, plot_item = self.add_new_embedded_pyqtgraph_render_plot_widget(name=name, dockSize=(500,50))
            # target_graphics_layout_widget = root_graphics_layout_widget
        else:
            # main_graphics_layout_widget = self.ui.main_graphics_layout_widget # GraphicsLayoutWidget
            target_graphics_layout_widget = self.ui.active_window_container_layout # GraphicsLayoutWidget



        target_graphics_layout_widget.removeItem(separate_plot_item)

        

    # ==================================================================================================================== #
    # DynamicDockDisplayAreaOwningMixin Conformances                                                                       #
    # ==================================================================================================================== #
    @property
    def dock_manager_widget(self) -> DynamicDockDisplayAreaContentMixin:
        """Must be implemented by subclasses to return the widget that manages the docks"""
        return self.ui.dynamic_docked_widget_container


    # matplotlib render subplot __________________________________________________________________________________________ #

    def _setupUI_matplotlib_render_plots(self):
        # performs required setup to enable dynamically added matplotlib render subplots.
        self.ui.matplotlib_view_widgets = {} # empty dictionary
        self.ui.connections['tracks'] = {}
        

    @function_attributes(short_name=None, tags=['matplotlib_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget', 'track'], input_requires=[], output_provides=[], uses=['MatplotlibTimeSynchronizedWidget', 'FigureWidgetDockDisplayConfig'], used_by=[], creation_date='2023-10-17 13:26', related_items=['add_new_embedded_pyqtgraph_render_plot_widget'])
    def add_new_matplotlib_render_plot_widget(self, row=1, col=0, name='matplotlib_view_widget', dockSize=(500,50), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None, sync_mode:Optional[SynchronizedPlotMode]=None) -> Tuple[MatplotlibTimeSynchronizedWidget, Figure, List[Axis], Dock]:
        """ creates a new dynamic MatplotlibTimeSynchronizedWidget, a container widget that holds a matplotlib figure, and adds it as a row to the main layout
        
        emit an event so the parent can call `self.update_scrolling_event_filters()` to add the new item
        
        """
        dDisplayItem: Dock = self.ui.dynamic_docked_widget_container.find_display_dock(identifier=name) # Dock
        if dDisplayItem is None:
            # No extant matplotlib_view_widget and display_dock currently, create a new one:
            ## TODO: hardcoded single-widget: used to be named `self.ui.matplotlib_view_widget`
            self.ui.matplotlib_view_widgets[name] = MatplotlibTimeSynchronizedWidget(name=name) # Matplotlib widget directly
            self.ui.matplotlib_view_widgets[name].setObjectName(name)
            self.ui.matplotlib_view_widgets[name].plots.fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, hspace=0.0, wspace=0.0)

            ## Enable scrollability
            self.ui.matplotlib_view_widgets[name].installEventFilter(self)
            
            ## Add directly to the main grid layout:
            if display_config is None:
                display_config = FigureWidgetDockDisplayConfig(showCloseButton=True)
            
            _, dDisplayItem = self.ui.dynamic_docked_widget_container.add_display_dock(name, dockSize=dockSize, display_config=display_config,
                                                                                    widget=self.ui.matplotlib_view_widgets[name], dockAddLocationOpts=dockAddLocationOpts, autoOrientation=False)
            dDisplayItem.setOrientation('horizontal', force=True)
            dDisplayItem.updateStyle()
            dDisplayItem.update()
            
            ## Add the initial plot:
            fig = self.ui.matplotlib_view_widgets[name].getFigure()
            _single_ax = self.ui.matplotlib_view_widgets[name].getFigure().add_subplot(111) # Adds a single axes to the figure
            ax = self.ui.matplotlib_view_widgets[name].axes # return all axes instead of just the first one
        
            ## emit the signal
            self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name])
            self.sigDockAdded.emit(self, dDisplayItem) ## sigDockAdded signal to indicate new dock has been added ## already has self.sigEmbeddedMatplotlibDockWidgetAdded: `self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name])`
        
        else:
            # Already had the dock and widget
            print(f'already had the valid matplotlib view widget and its display dock. Returning extant.')
            fig = self.ui.matplotlib_view_widgets[name].getFigure()
            ax = self.ui.matplotlib_view_widgets[name].axes # return all axes instead of just the first one
            
        ## Apply the default formatting:
        fig.patch.set_facecolor('black') ## Defines the "no data" color
        fig.patch.set_alpha(0.1)

        for an_ax in ax:
            an_ax.patch.set_facecolor('black')
            an_ax.patch.set_alpha(0.1)
    
        if sync_mode is not None:
            ## sync up the widgets
            self.sync_matplotlib_render_plot_widget(identifier=name, sync_mode=sync_mode)

        return self.ui.matplotlib_view_widgets[name], fig, ax, dDisplayItem
    

    @function_attributes(short_name=None, tags=['pyqtgraph_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget', 'pyqtgraph', 'docked_widget'], input_requires=[], output_provides=[], uses=['PyqtgraphTimeSynchronizedWidget'], used_by=[], creation_date='2024-12-31 03:35', related_items=['add_new_matplotlib_render_plot_widget'])
    def add_new_embedded_pyqtgraph_render_plot_widget(self, name='pyqtgraph_view_widget', dockSize=(500,50), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None, sync_mode:Optional[SynchronizedPlotMode]=None) -> Tuple[PyqtgraphTimeSynchronizedWidget, Any, Any, Dock]:
        """ creates a new dynamic PyqtgraphTimeSynchronizedWidget, a container widget that holds a pyqtgraph-based figure, and adds it as a row to the main layout
        
        based off of `add_new_matplotlib_render_plot_widget`, but to support embedded pyqtgraph plots instead of matplotlib plots
        
        emit an event so the parent can call `self.update_scrolling_event_filters()` to add the new item
        Uses: self.ui.matplotlib_view_widgets
        
        Usage:
        
            a_time_sync_pyqtgraph_widget, root_graphics_layout_widget, plot_item, dDisplayItem = self.add_new_embedded_pyqtgraph_render_plot_widget(name='test_pyqtgraph_view_widget', dockSize=(500,50))
            
        """
        dDisplayItem = self.ui.dynamic_docked_widget_container.find_display_dock(identifier=name) # Dock
        if dDisplayItem is None:
            # No extant matplotlib_view_widget and display_dock currently, create a new one:
            ## TODO: hardcoded single-widget: used to be named `self.ui.matplotlib_view_widget`
            self.ui.matplotlib_view_widgets[name] = PyqtgraphTimeSynchronizedWidget(name=name) # Matplotlib widget directly
            self.ui.matplotlib_view_widgets[name].setObjectName(name)

            if display_config is None:
                display_config = FigureWidgetDockDisplayConfig(showCloseButton=True, showCollapseButton=False, showGroupButton=False)
                
            should_hide_title: bool = getattr(display_config, 'hideTitleBar', False)
            
            _, dDisplayItem = self.ui.dynamic_docked_widget_container.add_display_dock(name, dockSize=dockSize, display_config=display_config,
                                                                                    widget=self.ui.matplotlib_view_widgets[name], dockAddLocationOpts=dockAddLocationOpts, autoOrientation=False, hideTitle=should_hide_title)
            dDisplayItem.setOrientation('horizontal', force=True)
            dDisplayItem.updateStyle()
            dDisplayItem.update()
            
            ## Add the plot:
            root_graphics_layout_widget = self.ui.matplotlib_view_widgets[name].getRootGraphicsLayoutWidget()
            plot_item = self.ui.matplotlib_view_widgets[name].getRootPlotItem()

            ## emit the signal
            self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name])
            self.sigDockAdded.emit(self, dDisplayItem) ## sigDockAdded signal to indicate new dock has been added ## already has self.sigEmbeddedMatplotlibDockWidgetAdded: `self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name])`


        else:
            # Already had the widget
            print(f'already had the valid pyqtgraph view widget and its display dock. Returning extant.')
            root_graphics_layout_widget = self.ui.matplotlib_view_widgets[name].getRootGraphicsLayoutWidget()
            plot_item = self.ui.matplotlib_view_widgets[name].getRootPlotItem()

        if sync_mode is not None:
            ## sync up the widgets
            self.sync_matplotlib_render_plot_widget(identifier=name, sync_mode=sync_mode)
            
        return self.ui.matplotlib_view_widgets[name], root_graphics_layout_widget, plot_item, dDisplayItem
    

    def _perform_remove_embedded_pyqtgraph_render_plot_widget(self, name='pyqtgraph_view_widget') -> Tuple[PyqtgraphTimeSynchronizedWidget, Any, Any]:
        """ creates a new dynamic PyqtgraphTimeSynchronizedWidget, a container widget that holds a matplotlib figure, and adds it as a row to the main layout
        
        based off of `add_new_matplotlib_render_plot_widget`, but to support embedded pyqtgraph plots instead of matplotlib plots
        
        emit an event so the parent can call `self.update_scrolling_event_filters()` to add the new item
        
        Usage:
        
            a_time_sync_pyqtgraph_widget, root_graphics_layout_widget, plot_item = self.add_new_embedded_pyqtgraph_render_plot_widget(name='test_pyqtgraph_view_widget', dockSize=(500,50))

            ## TODO: remove any connections first!
             
        """
        dDisplayItem = self.ui.dynamic_docked_widget_container.find_display_dock(identifier=name) # Dock
        if dDisplayItem is None:
            raise ValueError(f'display item "{name}" does not exist!')
        else:
            # Already had the widget
            print(f'removing the pyqtgraph view widget and its display dock for identifier: "{name}".')
            extant_widget = self.ui.matplotlib_view_widgets.pop(name) ## remove from `self.ui.matplotlib_view_widgets`
            root_graphics_layout_widget = extant_widget.getRootGraphicsLayoutWidget()
            plot_item = extant_widget.getRootPlotItem()
            ## remove here
            extant_widget.deleteLater()
            self.ui.dynamic_docked_widget_container.remove_display_dock(identifer=name)    
            # self.sigEmbeddedMatplotlibDockWidgetAdded.emit(self, dDisplayItem, self.ui.matplotlib_view_widgets[name])
            # self.sigEmbeddedMatplotlibDockWidgetRemoved.emit(self, dDisplayItem, extant_widget)
            self.sigEmbeddedMatplotlibDockWidgetRemoved.emit(self, name) ## get the widget before it fails
            self.sigDockClosed.emit(self, name) ## sigDockClosed signal
            print(f'\tremoved.')

            

    @function_attributes(short_name=None, tags=['matplotlib_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 13:23', related_items=[])
    def find_matplotlib_render_plot_widget(self, identifier, include_dock=False):
        """ finds the existing dynamically added matplotlib_render_plot_widget. 
        returns (widget, fig, ax)
        """
        active_matplotlib_view_widget = self.ui.matplotlib_view_widgets.get(identifier, None)
        if active_matplotlib_view_widget is not None:
            if active_matplotlib_view_widget.is_matplotlib_based():
                ## matplotlib-based:
                if include_dock:
                    dock_item = self.find_display_dock(identifier=identifier)
                    return active_matplotlib_view_widget, active_matplotlib_view_widget.getFigure(), active_matplotlib_view_widget.axes, dock_item # return all axes instead of just the first one
                else:
                    return active_matplotlib_view_widget, active_matplotlib_view_widget.getFigure(), active_matplotlib_view_widget.axes # return all axes instead of just the first one
            elif active_matplotlib_view_widget.is_pyqtgraph_based():
                ## pyqtgraph-based:
                if include_dock:
                    dock_item = self.find_display_dock(identifier=identifier)
                    return active_matplotlib_view_widget, active_matplotlib_view_widget.getRootGraphicsLayoutWidget(), active_matplotlib_view_widget.getRootPlotItem(), dock_item
                else:
                    return active_matplotlib_view_widget, active_matplotlib_view_widget.getRootGraphicsLayoutWidget(), active_matplotlib_view_widget.getRootPlotItem()
            else:
                raise NotImplementedError(f'active_matplotlib_view_widget: {active_matplotlib_view_widget}')
        else:
            print(f'WARNING: active_matplotlib_view_widget with identifier "{identifier}" was not found!')
            if include_dock:
                return None, None, None, None
            else:
                return None, None, None


    @function_attributes(short_name=None, tags=['track', 'pyqtgraph_view_widget', 'matplotlib_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget'], input_requires=[], output_provides=[], uses=['self.find_display_dock'], used_by=[], creation_date='2023-10-17 13:23', related_items=[])
    def find_track(self, identifier) -> Tuple[Optional[Union[MatplotlibTimeSynchronizedWidget, PyqtgraphTimeSynchronizedWidget]], Any, Any, Optional[Dock]]:
        """ finds the existing dynamically added widget (independent of the plotting backend used). 
        returns (widget, fig, ax, dock)
        
        """
        active_widget = self.ui.matplotlib_view_widgets.get(identifier, None)
        if active_widget is not None:
            if active_widget.is_matplotlib_based():
                ## matplotlib-based:
                dock_item = self.find_display_dock(identifier=identifier)
                return active_widget, active_widget.getFigure(), active_widget.axes, dock_item # return all axes instead of just the first one
            
            elif active_widget.is_pyqtgraph_based():
                ## pyqtgraph-based:
                dock_item = self.find_display_dock(identifier=identifier)
                return active_widget, active_widget.getRootGraphicsLayoutWidget(), active_widget.getRootPlotItem(), dock_item

            else:
                raise NotImplementedError(f'active_matplotlib_view_widget: {active_widget}')
        else:
            print(f'WARNING: active_matplotlib_view_widget with identifier "{identifier}" was not found!')
            return None, None, None, None
            

    @function_attributes(short_name=None, tags=['matplotlib_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 13:27', related_items=[])
    def remove_matplotlib_render_plot_widget(self, identifier):
        """ removes the subplot - does not work yet """
        ## TODO: need to remove the display item from self.ui.dynamic_docked_widget_container?
        active_matplotlib_view_widget = self.ui.matplotlib_view_widgets.get(identifier, None)
        if active_matplotlib_view_widget is not None:
            ## TODO: remove the connection from self.ui.connections[identifier]?
            self.sync_matplotlib_render_plot_widget(identifier, sync_mode=SynchronizedPlotMode.NO_SYNC)
            ## Remove the widget itself:
            # self.ui.dynamic_docked_widget_container
            self.ui.layout.removeWidget(active_matplotlib_view_widget) # Remove the matplotlib widget
            active_matplotlib_view_widget.deleteLater()
            active_matplotlib_view_widget = None # Set the matplotlib_view_widget to None ## TODO: this doesn't actually remove it from the UI container does it?
            ## remove from the dictionary
            del self.ui.matplotlib_view_widgets[identifier]
            self.sigEmbeddedMatplotlibDockWidgetRemoved.emit(self, identifier) ## get the widget before it fails
            self.sigDockClosed(self, identifier) ## sigDockClosed signal
            
        else:
            print(f'WARNING: active_matplotlib_view_widget with identifier "{identifier}" was not found!')

    @function_attributes(short_name=None, tags=['matplotlib_render_widget', 'dynamic_ui', 'group_matplotlib_render_plot_widget', 'sync'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 13:27', related_items=[])
    def sync_matplotlib_render_plot_widget(self, identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW):
        """ syncs a matplotlib render plot widget with a specified identifier with either the global window, the active time window, or disables sync with the Spike2DRaster. """
        # Requires specifying the identifier
        active_matplotlib_view_widget = self.ui.matplotlib_view_widgets.get(identifier, None)
        if active_matplotlib_view_widget is not None:
            if sync_mode.name == SynchronizedPlotMode.NO_SYNC.name:
                # disable syncing
                sync_connection = self.ui.connections.get(identifier, None)
                if sync_connection is not None:
                    # have an existing sync connection, need to disconnect it.
                    print(f'disconnecting window_scrolled for "{identifier}"')
                    self.window_scrolled.disconnect(sync_connection)
                    # print(f'WARNING: connection exists!')
                    self.ui.connections[identifier] = None
                    del self.ui.connections[identifier] # remove the connection after disconnecting it.

                return None
            elif sync_mode.name == SynchronizedPlotMode.TO_GLOBAL_DATA.name:
                ## Synchronize just once to the global data:
                # disable active window syncing if it's enabled:
                sync_connection = self.ui.connections.get(identifier, None)
                if sync_connection is not None:
                    # have an existing sync connection, need to disconnect it.
                    print(f'\tdisconnecting window_scrolled for "{identifier}"')
                    self.window_scrolled.disconnect(sync_connection)
                    # print(f'WARNING: connection exists!')
                    self.ui.connections[identifier] = None
                    del self.ui.connections[identifier] # remove the connection after disconnecting it.

                # Perform Initial (one-time) update from source -> controlled:
                active_matplotlib_view_widget.on_window_changed(self.spikes_window.total_df_start_end_times[0], self.spikes_window.total_df_start_end_times[1])
                return None

            elif sync_mode.name == SynchronizedPlotMode.TO_WINDOW.name:
                # Perform Initial (one-time) update from source -> controlled:
                active_matplotlib_view_widget.on_window_changed(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
                sync_connection = self.window_scrolled.connect(active_matplotlib_view_widget.on_window_changed)
                self.ui.connections[identifier] = sync_connection # add the connection to the connections array
                return sync_connection # return the connection
            else:
                raise NotImplementedError

        else:
            print(f'WARNING: active_matplotlib_view_widget with identifier "{identifier}" was not found!')
            return None

    
    def clear_all_matplotlib_plots(self):
        """ required by the menu function """
        print(f'clear_all_matplotlib_plots()')
        raise NotImplementedError
    

    # ==================================================================================================================== #
    # MARK: PyQtGraph                                                                                                      #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['UNFINISHED', 'DOCKs', 'tracks', 'dock'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-14 03:43', related_items=[])
    def perform_remove_dockgroup(self, flat_group_dockitems_list):
        # a_widget_to_remove = {}
        a_dock_to_remove = {}
        for a_dock in flat_group_dockitems_list:
            a_dock_identifier: str = a_dock.name()
            print(f'a_dock_identifier: "{a_dock_identifier}"')
            self.remove_matplotlib_render_plot_widget(identifier=a_dock_identifier)
            # active_2d_plot._perform_remove_embedded_pyqtgraph_render_plot_widget(name=a_dock_identifier)
            
            # for a_child_widget in a_dock.widgets:
            #     a_widget_to_remove[a_dock_identifier] = a_child_widget
            #     a_child_widget.deleteLater()
            a_dock_to_remove[a_dock_identifier] = a_dock
            a_dock.close()
        return a_dock_to_remove


    # ==================================================================================================================== #
    # Time Curves                                                                                                          #
    # ==================================================================================================================== #
    

    # Overrides for Render3DTimeCurvesBaseGridMixin, since this 2D class can't draw a 3D background grid _________________ #
    def init_3D_time_curves_baseline_grid_mesh(self):
        self.params.setdefault('time_curves_enable_baseline_grid', False) # this is False for this class (until it's implemented at least)
        self.params.setdefault('time_curves_baseline_grid_color', 'White')
        self.params.setdefault('time_curves_baseline_grid_alpha', 0.5)
        # BaseGrid3DTimeCurvesHelper.init_3D_time_curves_baseline_grid_mesh(self)
        pass

    def add_3D_time_curves_baseline_grid_mesh(self):
        # TODO: needs to be updated on .on_adjust_temporal_spatial_mapping(...)
        # return BaseGrid3DTimeCurvesHelper.add_3D_time_curves_baseline_grid_mesh(self)
        return False

    def update_3D_time_curves_baseline_grid_mesh(self):
        # BaseGrid3DTimeCurvesHelper.update_3D_time_curves_baseline_grid_mesh(self)
        pass

    def remove_3D_time_curves_baseline_grid_mesh(self):
        return False # nothing to remove
        # return BaseGrid3DTimeCurvesHelper.remove_3D_time_curves_baseline_grid_mesh(self)
    
            
    # Spike Emphasis Functions ___________________________________________________________________________________________ #
    def reset_spike_emphasis(self, defer_render=False):
        """ resets the emphasis state of all spikes to the default (SpikeEmphasisState.Default) and then rebuilds the all_spots """
        self.spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
        # TODO: PERFORMANCE: Rebuild the all_spots for all spikes after the update: (FUTURE) if more efficient, could just modify those that changed
        self.plots_data.all_spots = self._build_all_spikes_all_spots()
            
        # Once the dataframe is updated, rebuild the all_spots and update the plotters
        if not defer_render:
            # Update preview_overview_scatter_plot
            self.update_rasters()
                
    def update_spike_emphasis(self, spike_indicies=None, new_emphasis_state: SpikeEmphasisState=SpikeEmphasisState.Default, defer_render=False):
        """ sets the emphasis state for the spikes specified by spike_indices to new_emphasis_state 
        
        spike_indicies: e.g. np.logical_not(is_spike_included)
        defer_render: if false, the all_spots will be rebuilt after updating the dataframe and the changes rendered out


        Examples:

        from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState

        ## Example 1: De-emphasize spikes excluded from the placefield calculations:
        is_spike_included_in_pf = np.isin(spike_raster_window.spike_raster_plt_2d.spikes_df.index, active_pf_2D.filtered_spikes_df.index)
        spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included_in_pf), SpikeEmphasisState.Deemphasized)

        ## Example 2: De-emphasize spikes that don't have their 'aclu' from a given set of indicies:
        is_spike_included = spike_raster_window.spike_raster_plt_2d.spikes_df.aclu.to_numpy() == 2
        spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included), SpikeEmphasisState.Deemphasized)

        """
        if 'visualization_raster_emphasis_state' not in self.spikes_df.columns:
            print('Spike2DRaster.update_spike_emphasis(): adding "visualization_raster_emphasis_state" column to spikes_df...')
            self.spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default

        if spike_indicies is None:
            # If no particular indicies are specified, change all spikes by default
            # spike_indicies = self.spikes_df.indicies
            # spike_indicies = np.arange(np.shape(self.spikes_df)[0]) # build all indicies
            spike_indicies = np.full((np.shape(self.spikes_df)[0],), True)
        
        # Set the non-included spikes as SpikeEmphasisState.Deemphasized
        self.spikes_df.loc[spike_indicies, 'visualization_raster_emphasis_state'] = new_emphasis_state
        # TODO: PERFORMANCE: Rebuild the all_spots for all spikes after the update: (FUTURE) if more efficient, could just modify those that changed
        self.plots_data.all_spots = self._build_all_spikes_all_spots()
            
        # Once the dataframe is updated, rebuild the all_spots and update the plotters
        if not defer_render:
            self.update_rasters()
        


    # Debug Printing/Logging _____________________________________________________________________________________________ #
    
    def debug_print_spike_raster_2D_specific_plots_info(self, indent_string = '\t'):
        """ Prints a bunch of debugging info related to its specific plots and what they're displaying.
        Output Example:
            spikes_window Properties:
                total_df_start_end_times: (22.3668519082712, 2093.8524703475414)
                active_time_window: (341.96749018175865, 356.96749018175865)
                window_duration: 15.0
            Spatial Properties:
                temporal_axis_length: 15.0
                temporal_zoom_factor: 1.0
                render_window_duration: 15.0
            Time Curves:
                main_time_curves_view_widget.viewRect(): QRectF(x: 341.95783767210617, y: -0.19213325644284424, width: 15.009652509652483, height: 1.1716180255700652)
            UI/Graphics Properties:
                main_plot_widget:
                    x: 341.96749018175865, 356.96749018175865
                    y: -2.789445932897294, 72.7894459328973
                background_static_scroll_plot_widget:
                    x: 22.3668519082712, 2093.8524703475414
                    y: 0.5, 69.5
                ui.scroll_window_region
                    min_x: 341.96749018175865, max_x: 356.96749018175865, x_duration: 15.0
            debug_print_axes_locations(...): Active Window/Local Properties:
                (active_t_start: 341.96749018175865, active_t_end: 356.96749018175865), active_window_t_duration: 15.0
                (active_x_start: 2.3142857142857145, active_x_end: 2.422903412616405), active_x_length: 0.10861769833069035
            debug_print_axes_locations(...): Global (all data) Data Properties:
                (global_start_t: 22.3668519082712, global_end_t: 2093.8524703475414), global_total_data_duration: 2071.48561843927 (seconds)
                total_data_duration_minutes: 34.0
                (global_x_start: 0.0, global_x_end: 15.0), global_total_x_length: 15.0


        """
        # main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        main_plot_widget = self.plots.main_plot_widget # PlotItem
        background_static_scroll_plot_widget = self.plots.background_static_scroll_window_plot # PlotItem
        
        print(f'{indent_string}main_plot_widget:')
        curr_x_min, curr_x_max, curr_y_min, curr_y_max = self.get_plot_view_range(main_plot_widget, debug_print=False)
        print(f'{indent_string}\tx: {curr_x_min}, {curr_x_max}\n{indent_string}\ty: {curr_y_min}, {curr_y_max}')
        
        print(f'{indent_string}background_static_scroll_plot_widget:')
        curr_x_min, curr_x_max, curr_y_min, curr_y_max = self.get_plot_view_range(background_static_scroll_plot_widget, debug_print=False)
        print(f'{indent_string}\tx: {curr_x_min}, {curr_x_max}\n{indent_string}\ty: {curr_y_min}, {curr_y_max}')

        min_x, max_x = self.ui.scroll_window_region.getRegion()
        x_duration = max_x - min_x
        print(f'{indent_string}ui.scroll_window_region\n{indent_string}\tmin_x: {min_x}, max_x: {max_x}, x_duration: {x_duration}') # min_x: 7455.820603311667, max_x: 7532.52160713601, x_duration: 76.70100382434339 -- NOTE: these are the real seconds!
        
        
    def debug_print_spike_raster_timeline_alignments(self, indent_string = '\t'):
        """ dumps debug properties related to alignment of various windows for a spike_raster_window
            Created 2022-09-05 to debug issues with adding Time Curves to spike_raster_2d
        Usage:
            active_2d_plot.debug_print_spike_raster_timeline_alignments()
        
        Example Output:
            spikes_window Properties:
                total_df_start_end_times: (22.3668519082712, 2093.8524703475414)
                active_time_window: (341.96749018175865, 356.96749018175865)
                window_duration: 15.0
            Spatial Properties:
                temporal_axis_length: 15.0
                temporal_zoom_factor: 1.0
                render_window_duration: 15.0
            Time Curves:
                main_time_curves_view_widget.viewRect(): QRectF(x: 341.95783767210617, y: -0.19213325644284424, width: 15.009652509652483, height: 1.1716180255700652)
            UI/Graphics Properties:
                main_plot_widget:
                    x: 341.96749018175865, 356.96749018175865
                    y: -2.789445932897294, 72.7894459328973
                background_static_scroll_plot_widget:
                    x: 22.3668519082712, 2093.8524703475414
                    y: 0.5, 69.5
                ui.scroll_window_region
                    min_x: 341.96749018175865, max_x: 356.96749018175865, x_duration: 15.0
            debug_print_axes_locations(...): Active Window/Local Properties:
                (active_t_start: 341.96749018175865, active_t_end: 356.96749018175865), active_window_t_duration: 15.0
                (active_x_start: 2.3142857142857145, active_x_end: 2.422903412616405), active_x_length: 0.10861769833069035
            debug_print_axes_locations(...): Global (all data) Data Properties:
                (global_start_t: 22.3668519082712, global_end_t: 2093.8524703475414), global_total_data_duration: 2071.48561843927 (seconds)
                total_data_duration_minutes: 34.0
                (global_x_start: 0.0, global_x_end: 15.0), global_total_x_length: 15.0

        """
        
        # Window Properties:
        print(f'spikes_window Properties:')
        self.spikes_window.debug_print_spikes_window(prefix_string='', indent_string=indent_string)
        
        ## Spatial Properties:
        print(f'Spatial Properties:')
        debug_print_temporal_info(self, prefix_string='', indent_string=indent_string)
        
        ## Time Curves: main_time_curves_view_widget:
        if self.ui.main_time_curves_view_widget is not None:
            print(f'Time Curves:')
            main_tc_view_rect = self.ui.main_time_curves_view_widget.viewRect() # PyQt5.QtCore.QRectF(57.847549828567, -0.007193522045074202, 15.76451934295443, 1.0150365839255244)
            debug_print_QRect(main_tc_view_rect, prefix_string='main_time_curves_view_widget.viewRect(): ', indent_string=indent_string)
        else:
            print(f'No Time Curves added.')

        ## UI Properties:
        print(f'UI/Graphics Properties:')
        self.debug_print_spike_raster_2D_specific_plots_info(indent_string = '\t')
        debug_print_axes_locations(self)


    # ==================================================================================================================== #
    # Crosshairs/Tracing Functions                                                                                         #
    # ==================================================================================================================== #

    def toggle_crosshair_traces_enabled(self, are_crosshairs_enabled):
        print(f'SpikeRaster2D.on_crosshair_trace_toggled(is_crosshairs_enabled={are_crosshairs_enabled})')
        if are_crosshairs_enabled:
            # for a_ts_widget in self.get_flat_widgets_list():
            for an_identifier, (a_dock_item, a_ts_widget) in self.get_flat_dock_item_tuple_dict().items():
                if hasattr(a_ts_widget, 'add_crosshairs'):
                    # add_crosshairs
                    try:
                        a_ts_widget.add_crosshairs(a_ts_widget.active_plot_target, name='traceHairs', should_force_discrete_to_bins=False, enable_y_trace=True)
                        print(f'\tan_identifier: "{an_identifier}"')
                        if an_identifier not in self.ui.connections['tracks']:
                            self.ui.connections['tracks'][an_identifier] = {} ## make new dict to hold connections
                            
                        if 'sigCrosshairsUpdated' not in self.ui.connections['tracks'][an_identifier]:
                            ## enable crosshairs callback
                            print(f'\t\tadding crosshairs callback for item: "{an_identifier}"')
                            self.ui.connections['tracks'][an_identifier]['sigCrosshairsUpdated'] = None
                            # self.on_child_crosshair_updated_signal
                            _crosshairs_updated_conn = a_ts_widget.sigCrosshairsUpdated.connect(lambda a_child_widget, an_identifier, a_trace_value: self.on_child_crosshair_updated_signal(an_identifier, a_trace_value))
                            self.ui.connections['tracks'][an_identifier]['sigCrosshairsUpdated'] = _crosshairs_updated_conn ## set just as the raw connection so we can disconnect
                            # self.ui.connections['tracks'][an_identifier]['sigCrosshairsUpdated'] = (_crosshairs_updated_conn, a_ts_widget.sigCrosshairsUpdated) ## set a tuple so we can disconnect
                            print(f'\t\tdone.')
                        else:
                            print(f'\talready have a crosshairs callback for item {an_identifier}')
                            
                    except Exception as e:
                        print(f'\tfailed to add crosshair traces for widget: {a_ts_widget}.\n\tError: {e}\n\tSkipping.')
                    
                    
        else:
            ## disable crosshairs
            # for a_ts_widget in self.get_flat_widgets_list():
            for an_identifier, (a_dock_item, a_ts_widget) in self.get_flat_dock_item_tuple_dict().items():
                if hasattr(a_ts_widget, 'remove_crosshairs'):
                    # remove_crosshairs:
                    try:
                        a_ts_widget.remove_crosshairs(a_ts_widget.active_plot_target, name='traceHairs')
                        an_existing_crosshairs_updated_conn = self.ui.connections['tracks'].get(an_identifier, {}).pop('sigCrosshairsUpdated', None)
                        if an_existing_crosshairs_updated_conn is not None:
                            print(f'\tfound connection to remove for an_identifier: {an_identifier}, an_existing_crosshairs_updated_conn: {an_existing_crosshairs_updated_conn}')
                            a_ts_widget.sigCrosshairsUpdated.disconnect(an_existing_crosshairs_updated_conn) ## disconnect
                            print(f'\t\tconnection removed!')                        
                        
                    except Exception as e:
                        print(f'\tfailed to remove crosshair traces for widget: {a_ts_widget}.\n\tError: {e}\n\tSkipping.')
                        # raise e
                                
            

    def on_child_crosshair_updated_signal(self, child_identifier, trace_value):
        """ called when a child (with crosshairs enabled) updates its crosshairs trace. """
        if self.debug_print:
            print(f'SpikeRaster2D.on_child_crosshair_updated_signal(self: {self}, child_identifier: "{child_identifier}", trace_value: "{trace_value}")')
        self.sigCrosshairsUpdated.emit(self, child_identifier, trace_value)



    
# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike2DRaster()
#     v.animation()
# dfsd