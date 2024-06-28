from copy import deepcopy
from typing import Dict
import numpy as np

from qtpy import QtCore, QtWidgets
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.Qt.GlobalConnectionManager import GlobalConnectionManager, GlobalConnectionManagerAccessingMixin
from pyphocorehelpers.gui.Qt.qevent_lookup_helpers import QEventLookupHelpers

from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Uic_AUTOGEN_Spike3DRasterWindowBase import Ui_RootWidget # Generated file from .ui


from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo

from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController, TimeWindowPlaybackControllerActionsMixin

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import SpikeRasterBottomFrameControlsMixin
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarWidget import Spike3DRasterLeftSidebarControlBar, SpikeRasterLeftSidebarControlsMixin
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget import Spike3DRasterRightSidebarWidget, SpikeRasterRightSidebarOwningMixin

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin

from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers, UnitColoringMode, ColorData
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.GUI.Qt.NeuronVisualSelectionControls.NeuronVisualSelectionControlsWidget import NeuronVisualSelectionControlsWidget, NeuronWidgetContainer, add_neuron_display_config_widget



# remove TimeWindowPlaybackControllerActionsMixin
# class Spike3DRasterWindowWidget(SpikeRasterBottomFrameControlsMixin, TimeWindowPlaybackControllerActionsMixin, TimeWindowPlaybackPropertiesMixin, QtWidgets.QWidget):
class Spike3DRasterWindowWidget(GlobalConnectionManagerAccessingMixin, SpikeRasterRightSidebarOwningMixin, SpikeRasterLeftSidebarControlsMixin, SpikeRasterBottomFrameControlsMixin, SpikesWindowOwningMixin, QtWidgets.QWidget):
    """ A main raster window loaded from a Qt .ui file. 
    
    Manages the main raster views in addition to the shared window-related functions such as menu management, connections, etc.
    
    Properties/Parameters:
    
        self.params.scrollStepMultiplier: (default: 30.0) - The multiplier by which each scroll step is multiplied. Decrease this value to increase scrolling precision (making the same rotation of the mousewheel scroll less in time).
        
    
    Usage:
    
    from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

    spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df)
    
    """
    
    enable_window_close_confirmation = False
    # Application/Window Configuration Options:
    applicationName = 'Spike3DRasterWindow'
    windowName = 'Spike3DRasterWindow'
    
    # enable_debug_print = True
    enable_debug_print = False
    # enable_interaction_events_debug_print = True
    enable_interaction_events_debug_print = False
    
    enable_smooth_scrolling_animation = False # if True, scrolling will be animated
    
    
    # TODO: add signals here:
    
    
    ######## TimeWindowPlaybackPropertiesMixin requirement:
    @property
    def animation_active_time_window(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate."""
        return self.spikes_window
    @animation_active_time_window.setter
    def animation_active_time_window(self, value):
        self.spikes_window = value
    @property
    def spikes_window(self):
        """ Just wraps its embedded spike_raster_plt_2d widget's spikes_window property."""
        return self.spike_raster_plt_2d.spikes_window
    @spikes_window.setter
    def spikes_window(self, value):
        self.spike_raster_plt_2d.spikes_window = value

    @property
    def should_debug_print_interaction_events(self):
        """ Whether to debug print user interaction events like mouse clicks, mouse wheel, key presses, etc. """
        return (self.enable_debug_print and Spike3DRasterWindowWidget.enable_interaction_events_debug_print)
    
    @property
    def temporal_zoom_factor(self):
        """The time dilation factor that maps spikes in the current window to y-positions along the time axis multiplicatively.
            Increasing this factor will result in a more spatially expanded time axis while leaving the visible window unchanged.
        """
        if self.spike_raster_plt_3d is not None:
            return self.spike_raster_plt_3d.temporal_zoom_factor
        elif self.spike_raster_plt_2d is not None:
            return self.spike_raster_plt_2d.temporal_zoom_factor
        else:
            return self.params.temporal_zoom_factor
    @temporal_zoom_factor.setter
    def temporal_zoom_factor(self, value):
        if self.spike_raster_plt_3d is not None:
            self.spike_raster_plt_3d.temporal_zoom_factor = value
        elif self.spike_raster_plt_2d is not None:
            self.spike_raster_plt_2d.temporal_zoom_factor = value
        else:
            self.params.temporal_zoom_factor = value
            # this is the no plotter case.
        # TODO: should this update the temporal_zoom_factor for both the 2D AND the 3D plotter?
        
        
        
    ######## TimeWindowPlaybackPropertiesMixin requirement:
        
    @property
    def animation_time_step(self):
        """ How much to step forward in time at each frame of animation. """
        return self.params.animation_time_step
    @animation_time_step.setter
    def animation_time_step(self, value):
        self.params.animation_time_step = value
        
        
    ## STATE PROPERTIES
    @property
    def is_playback_reversed(self):
        """The is_playback_reversed property."""
        return self.params.is_playback_reversed
    @is_playback_reversed.setter
    def is_playback_reversed(self, value):
        self.params.is_playback_reversed = value
        
    @property
    def animation_playback_direction_multiplier(self):
        """The animation_reverse_multiplier property."""
        if self.params.is_playback_reversed:
            return -1.0
        else:
            return 1.0

    @property
    def playback_update_frequency(self):
        """The rate at which the separate animation thread attempts to update the interface. ReadOnly."""
        return self.params.playback_update_frequency

    @property
    def playback_rate_multiplier(self):
        """ 1x playback (real-time) occurs when self.playback_update_frequency == self.animation_time_step. 
            if self.animation_time_step = 2.0 * self.playback_update_frequency => for each update the window will step double the time_step forward in time than it would be default, meaning a 2.0x playback_rate_multiplier.
        """
        return (self.animation_time_step / self.playback_update_frequency)
    @playback_rate_multiplier.setter
    def playback_rate_multiplier(self, value):
        """ since self.playback_update_frequency is fixed, only self.animation_time_step can be adjusted to set the playback_rate_multiplier. """
        desired_playback_rate_multiplier = value
        self.animation_time_step = self.playback_update_frequency * desired_playback_rate_multiplier

    def compute_animation_frame_shift_duration(self, shift_frames: int):
        """ Computes the equivalent time duration for a specified number of animation shift_frames
            Does not modify any internal animation state.
            extracted from Spike3DRasterWindowWidget.shift_animation_frame_val(...)
        """
        return (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames)) # compute the amount of time equivalent to the shift_frames

    def compute_frame_shifted_start_timestamp(self, shift_frames: int):
        """ Computes the next start timestamp given the specified number of animation shift frames.
            Does not modify any internal animation state.
            extracted from Spike3DRasterWindowWidget.shift_animation_frame_val(...) 
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.compute_frame_shifted_start_timestamp(shift_frames: {shift_frames})')
        curr_start_time = self.animation_active_time_window.active_window_start_time
        shift_time = self.compute_animation_frame_shift_duration(shift_frames) # compute the amount of time equivalent to the shift_frames
        next_start_timestamp = curr_start_time + shift_time
        return next_start_timestamp
        
        
    ## Other Properties:
    
    @property
    def spike_raster_plt_2d(self):
        """The spike_raster_plt_2d property."""
        return self.ui.spike_raster_plt_2d
    
    @property
    def spike_raster_plt_3d(self):
        """The spike_raster_plt_2d property."""
        return self.ui.spike_raster_plt_3d
    
    
    
    def __init__(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, application_name=None, type_of_3d_plotter='pyqtgraph', parent=None):
        """_summary_

        Args:
            curr_spikes_df (_type_): _description_
            core_app_name (str, optional): _description_. Defaults to 'UnifiedSpikeRasterApp'.
            window_duration (float, optional): _description_. Defaults to 15.0.
            window_start_time (float, optional): _description_. Defaults to 30.0.
            neuron_colors (_type_, optional): _description_. Defaults to None.
            neuron_sort_order (_type_, optional): _description_. Defaults to None.
            application_name (_type_, optional): _description_. Defaults to None.
            type_of_3d_plotter (str, optional): specifies which type of 3D plotter to build. Must be {'pyqtgraph', 'vedo', None}. Defaults to 'pyqtgraph'.
            parent (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_RootWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        if application_name is not None:
            self.applicationName = application_name
        # else:
        #     self.applicationName = Spike3DRasterWindowWidget.applicationName
        
        
        self.enable_debug_print = Spike3DRasterWindowWidget.enable_debug_print
        
        app = pg.mkQApp(self.applicationName) # <PyQt5.QtWidgets.QApplication at 0x1d44a4891f0>
        self.GlobalConnectionManagerAccessingMixin_on_init(owning_application=app) # initializes self._connection_man
        
        
        # self.ui.splitter.setSizes([900, 200])
        # self.ui.splitter.setStretchFactor(0, 5) # have the top widget by 3x the height as the bottom widget
        # self.ui.splitter.setStretchFactor(1, 1) # have the top widget by 3x the height as the bottom widget        
        
        
        self.params = VisualizationParameters(self.applicationName)
        self.params.type_of_3d_plotter = type_of_3d_plotter
        
        # Helper Mixins: INIT:
        
        self.SpikeRasterBottomFrameControlsMixin_on_init()
        self.SpikeRasterLeftSidebarControlsMixin_on_init()
        self.SpikeRasterRightSidebarOwningMixin_on_init()

        # Helper Mixins: SETUP:
        self.SpikeRasterBottomFrameControlsMixin_on_setup()
        self.SpikeRasterLeftSidebarControlsMixin_on_setup()
        self.SpikeRasterRightSidebarOwningMixin_on_setup()
        
        self.initUI(curr_spikes_df, core_app_name=application_name, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, type_of_3d_plotter=self.params.type_of_3d_plotter)
        
        # Update the windows once before showing the UI:
        self.spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
        
        self.show() # Show the GUI

    def initUI(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, type_of_3d_plotter='pyqtgraph'):
        # 
        self.ui.spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
        
        if type_of_3d_plotter is None:
            # No 3D plotter:
            self.ui.spike_raster_plt_3d = None 

            
        elif type_of_3d_plotter == 'pyqtgraph':
            self.ui.spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None)
            # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
        elif type_of_3d_plotter == 'vedo':
            # To work around a bug with the vedo plotter with the pyqtgraph 2D controls: we must update the 2D Scroll Region to the initial value, since it only works if the 2D Raster plot (pyqtgraph-based) is created before the Spike3DRaster_Vedo (Vedo-based). This is probably due to the pyqtgraph's instancing of the QtApplication. 
            self.ui.spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
            
            # Build the 3D Vedo Raster plotter
            self.ui.spike_raster_plt_3d = Spike3DRaster_Vedo.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None)
            self.ui.spike_raster_plt_3d.disable_render_window_controls()
            
            # Set the 3D Vedo plots' window to the current values of the 2d plot:
            self.ui.spike_raster_plt_3d.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        
        else:
            # unrecognized command for 3D plotter
            raise NotImplementedError
        
        # Add the plotter widgets to the UI:
        self.ui.v_layout = QtWidgets.QVBoxLayout()
        self.ui.v_layout.setContentsMargins(0,0,0,0)
        if self.ui.spike_raster_plt_3d is not None:
            self.ui.v_layout.addWidget(self.ui.spike_raster_plt_3d)
        self.ui.mainSpike3DRasterWidget.setLayout(self.ui.v_layout)

        if self.ui.spike_raster_plt_3d is None:
            # IF there's NOT any 3D Plotter, collapse `mainSpike3DRasterWidget`, the holder widget for it so the 2D plotter takes up the whole frame
            self.ui.mainSpike3DRasterWidget.setVisible(False)
        
        self.ui.v_layout_secondary = QtWidgets.QVBoxLayout()
        self.ui.v_layout_secondary.setContentsMargins(0,0,0,0)
        self.ui.v_layout_secondary.addWidget(self.ui.spike_raster_plt_2d)
        self.ui.secondarySpikeRasterControlWidget.setLayout(self.ui.v_layout_secondary)
        
        ## TODO: could easily wrap the two plotters (3D and 2D) in Docks if desired here
        
        self.GlobalConnectionManagerAccessingMixin_on_setup()
        
        # self.ui.additional_connections = {} # NOTE: self.ui.additional_connections has been removed in favor of self.connection_man
        
        ## Create the animation properties:
        self.playback_controller = TimeWindowPlaybackController()
        self.playback_controller.setup(self) # pass self to have properties set
        
        ## Connect the UI Controls:
        # Helper Mixins: buildUI:
        self.SpikeRasterLeftSidebarControlsMixin_on_buildUI() # Call this to set the initial values for the UI before signals are connected.
        self.SpikeRasterRightSidebarOwningMixin_on_buildUI()


        # self.ui.bottom_controls_frame, self.ui.bottom_controls_layout = self.SpikeRasterBottomFrameControlsMixin_on_buildUI() # NOTE: do not call for the window as it already has a valid bottom bar widget
        # Connect the signals:
        self.ui.bottom_bar_connections = None 
        self.ui.bottom_bar_connections = self.SpikeRasterBottomFrameControlsMixin_connectSignals(self.ui.bottomPlaybackControlBarWidget)
        
        self.ui.left_side_bar_connections = None
        self.ui.left_side_bar_connections = self.SpikeRasterLeftSidebarControlsMixin_connectSignals(self.ui.leftSideToolbarWidget)

        self.ui.right_side_bar_connections = None
        self.ui.right_side_bar_connections = self.SpikeRasterRightSidebarOwningMixin_connectSignals(self.ui.leftSideToolbarWidget)

        
        ## Setup the right side bar:
        rightSideContainerWidget = self.ui.rightSideContainerWidget
        self.ui.rightSideContainerWidget.setVisible(False) # collapses and hides the sidebar
        # self.ui.rightSideContainerWidget.setVisible(True) # shows the sidebar


        ## Install the event filter in the 2D View to enable scroll wheel events:
        if self.ui.spike_raster_plt_2d is not None:
            # self.ui.spike_raster_plt_2d.installEventFilter(self) # Kinda works, but doesn't show when scrolling over plots
            # self.ui.spike_raster_plt_2d.plots.background_static_scroll_window_plot.installEventFilter(self) # background_static_scroll_window_plot is a PlotItem 
            # if self.ui.spike_raster_plt_2d.plots.main_plot_widget is not None:
            #     # This doesn't work
            #     self.ui.spike_raster_plt_2d.plots.main_plot_widget.installEventFilter(self) # main_plot_widget is a PlotItem 
            if self.ui.spike_raster_plt_2d.plots.scatter_plot is not None:
                # This is the first thing that produces detected event.type() == QtCore.QEvent.GraphicsSceneWheel when the scrolling is done over the 2D active widnow raster plot 
                self.ui.spike_raster_plt_2d.plots.scatter_plot.installEventFilter(self) # scatter_plot is a ScatterPlotItem 
            if self.ui.spike_raster_plt_2d.plots.preview_overview_scatter_plot is not None:
                # This is the first thing that produces detected event.type() == QtCore.QEvent.GraphicsSceneWheel when the scrolling is done over the 2D active widnow raster plot 
                self.ui.spike_raster_plt_2d.plots.preview_overview_scatter_plot.installEventFilter(self) # plots.preview_overview_scatter_plot is a ScatterPlotItem

            # Connect BottomBar's combobox to the 2D plotters rendered intervals
            self.ui.bottom_bar_connections.append(self.ui.spike_raster_plt_2d.sigRenderedIntervalsListChanged.connect(self.ui.bottomPlaybackControlBarWidget.on_rendered_intervals_list_changed))
            # Connect the jump forward/back controls:
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_target_left.connect(self.perform_jump_prev_series_item))
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_target_right.connect(self.perform_jump_next_series_item))
            # Connect the interval series action controls:
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.series_remove_pressed.connect(self.perform_interval_series_remove_item))
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.series_customize_pressed.connect(self.perform_interval_series_customize_item))
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.series_clear_all_pressed.connect(self.perform_interval_series_clear_all))
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.series_add_pressed.connect(self.perform_interval_series_request_add))

        # Set Window Title Options:
        if self.applicationName is not None:
            # self.setWindowFilePath(str(sess.filePrefix.resolve()))
            self.setWindowTitle(f'{self.applicationName}') # f'Spike Raster Window - {secondary_active_config_name} - {str(sess.filePrefix.resolve())}'

        ## Scrolling Properties:
        self._scheduledAnimationSteps = 0
        self.params.scrollStepMultiplier = 30.0 # The multiplier by which each scroll step is multiplied. Decrease this value to increase scrolling precision (making the same rotation of the mousewheel scroll less in time).
        if self.enable_smooth_scrolling_animation:
            ## Add the QPropertyAnimation for smooth scrolling, but do not start it:
            self.ui.scrollAnim = QtCore.QPropertyAnimation(self, b"numScheduledScalings") # the animation will act on the self.numScheduledScalings pyqtProperty
            # self.ui.scrollAnim.setEndValue(0) # Update the end value
            self.ui.scrollAnim.setDuration(250) # set duration in milliseconds
            
            ## QTimeLine-style smooth scrolling:
            self.ui.scrollAnimTimeline = QtCore.QTimeLine(250, parent=self) # Make a new QTimeLine with a 250ms animation duration
            self.ui.scrollAnimTimeline.setUpdateInterval(20)
            self.ui.scrollAnimTimeline.setCurveShape(QtCore.QTimeLine.CurveShape.LinearCurve)
            self.ui.scrollAnimTimeline.setFrameRange(0, 100)
            self.ui.scrollAnimTimeline.frameChanged.connect(self.onScrollingTimelineFired)
            # self.ui.scrollAnimTimeline.valueChanged.connect(self.onScrollingTimelineFired)
            self.ui.scrollAnimTimeline.finished.connect(self.onScrollingTimelineAnimationFinished)
            # self.ui.scrollAnimTimeline.start() # Do not start it

        else:
            self.ui.scrollAnim = None
            self.ui.scrollAnimTimeline = None
            
        
          
    def __str__(self):
         return
      


    def connect_plotter_time_windows(self):
        """ connects the controlled plotter (usually the 3D plotter) to the 2D plotter that controls it. """
        # self.spike_3d_to_2d_window_connection = self.spike_raster_plt_2d.window_scrolled.connect(self.spike_raster_plt_3d.spikes_window.update_window_start_end)        
        # Rate limited version:
        self.spike_3d_to_2d_window_connection = self.connection_man.connect_drivable_to_driver(drivable=self.spike_raster_plt_3d, driver=self.spike_raster_plt_2d,
                                                       custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.spikes_window.update_window_start_end_rate_limited)))
                                                       
        # self.spike_3d_to_2d_window_connection = pg.SignalProxy(self.spike_raster_plt_2d.window_scrolled, delay=0.2, rateLimit=60, slot=self.spike_raster_plt_3d.spikes_window.update_window_start_end_rate_limited) # Limit updates to 60 Signals/Second
        
         
    def connect_additional_controlled_plotter(self, controlled_plt):
        """ try to connect the controlled_plt to the current controller (usually the 2D plot). """
        return self.connection_man.connect_drivable_to_driver(drivable=controlled_plt, driver=self.spike_raster_plt_2d)
        
    def connect_controlled_time_synchronized_plotter(self, controlled_plt):
        """ try to connect the controlled_plt to the current controller (usually the 2D plot). """
        return self.connection_man.connect_drivable_to_driver(drivable=controlled_plt, driver=self.spike_raster_plt_2d,
                                                       custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.spikes_window.update_window_start_end_rate_limited)))
     
    def create_new_connected_widget(self, type_of_3d_plotter='vedo'):
        """ called to create a new/independent widget instance that's connected to this window's driver. """
        
        # self.neuron_colors
        # window_duration = self.render_window_duration
        window_duration = self.spikes_window.window_duration
        window_start_time = self.spikes_window.active_window_start_time
        # window_end_time = self.spikes_window.active_window_end_time
        
        neuron_colors = None
        neuron_sort_order = None
        
        if type_of_3d_plotter is None:
            # No 3D plotter:
            output_widget = None 
            
        elif type_of_3d_plotter == 'pyqtgraph2D':
            # a 2D child widget:
            output_widget = Spike2DRaster.init_from_independent_data(self.spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None)
            # connect_controlled_time_synchronized_plotter(...) should still be good for the new Spike2DRaster            
            
            
        elif type_of_3d_plotter == 'pyqtgraph':
            output_widget = Spike3DRaster.init_from_independent_data(self.spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None)
            # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
        elif type_of_3d_plotter == 'vedo':
            # To work around a bug with the vedo plotter with the pyqtgraph 2D controls: we must update the 2D Scroll Region to the initial value, since it only works if the 2D Raster plot (pyqtgraph-based) is created before the Spike3DRaster_Vedo (Vedo-based). This is probably due to the pyqtgraph's instancing of the QtApplication. 
            # self.ui.spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
            
            # Build the 3D Vedo Raster plotter
            output_widget = Spike3DRaster_Vedo.init_from_independent_data(self.spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=self.applicationName, enable_independent_playback_controller=False, should_show=False, parent=None)
            output_widget.disable_render_window_controls()
            
            # Set the 3D Vedo plots' window to the current values of the 2d plot:
            # output_widget.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        
        else:
            # unrecognized command for 3D plotter
            raise NotImplementedError

        # Connect the output_widget:
        # self.connect_additional_controlled_plotter(output_widget)
        self.connect_controlled_time_synchronized_plotter(output_widget)
        
        return output_widget
        
    
    ###################################
    #### EVENT HANDLERS
    ##################################
    def closeEvent(self, event):
        """closeEvent(self, event): pyqt default event, doesn't have to be registered. Called when the widget will close.
        """
        if self.enable_window_close_confirmation:
            reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        else:
            reply = QtWidgets.QMessageBox.Yes
            
        if reply == QtWidgets.QMessageBox.Yes:
            self.GlobalConnectionManagerAccessingMixin_on_destroy() # call destroy to tear down the registered children for the global connection mannager
            event.accept()
            print('Window closed')
        else:
            event.ignore()
   
    ###################################
    #### EVENT HANDLERS
    ##################################
    

    @pyqtExceptionPrintingSlot(float)
    def update_animation(self, next_start_timestamp: float):
        """ Actually updates the animation given the next_start_timestep
            extracted from Spike3DRasterWindowWidget.shift_animation_frame_val(...)
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.update_animation(next_start_timestamp: {next_start_timestamp})')
        # self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        # Update the windows once before showing the UI:
        self.spike_raster_plt_2d.update_scroll_window_region(next_start_timestamp, next_start_timestamp+self.animation_active_time_window.window_duration, block_signals=True) # self.spike_raster_plt_2d.window_scrolled should be emitted        
        # signal emit:
        self.spike_raster_plt_2d.window_scrolled.emit(next_start_timestamp, next_start_timestamp+self.animation_active_time_window.window_duration)
        # update_scroll_window_region
        # self.ui.spike_raster_plt_3d.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        

    @pyqtExceptionPrintingSlot(int)
    def shift_animation_frame_val(self, shift_frames: int):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.shift_animation_frame_val(shift_frames: {shift_frames})')
        next_start_timestamp = self.animation_active_time_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames)) # Equivalent to self.compute_frame_shifted_start_timestamp(shift_frames)
        self.update_animation(next_start_timestamp)
        

    # Called from SliderRunner's thread when it emits the update_signal:        
    @pyqtExceptionPrintingSlot()
    def increase_animation_frame_val(self):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.increase_animation_frame_val()')
        self.shift_animation_frame_val(1)
        
    ## Update Functions:
    @pyqtExceptionPrintingSlot(bool)
    def play_pause(self, is_playing):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.play_pause(is_playing: {is_playing})')
        if (not is_playing):
            self.animationThread.start()
        else:
            self.animationThread.terminate()
            

    @pyqtExceptionPrintingSlot()
    def on_jump_left(self):
        # Skip back some frames
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_left()')
        self.shift_animation_frame_val(-5)
        
    @pyqtExceptionPrintingSlot()
    def on_jump_right(self):
        # Skip forward some frames
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_right()')
        self.shift_animation_frame_val(5)
        

    @pyqtExceptionPrintingSlot()
    def on_jump_window_left(self):
        """ jumps by the full width of the window, consistent with a PaegUp operation. """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_window_left()')
        time_window = self.animation_active_time_window # SpikesDataframeWindow
        window_duration_sec = time_window.window_duration
        proposed_next_window_start_time = time_window.active_window_start_time - window_duration_sec
        self.update_animation(proposed_next_window_start_time)
        

    @pyqtExceptionPrintingSlot()
    def on_jump_window_right(self):
        """ jumps by the full width of the window, consistent with a PaegDown operation. """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_window_right()')
        time_window = self.animation_active_time_window # SpikesDataframeWindow
        window_duration_sec = time_window.window_duration
        proposed_next_window_start_time = time_window.active_window_start_time + window_duration_sec
        self.update_animation(proposed_next_window_start_time)
        


    @pyqtExceptionPrintingSlot(bool)
    def on_reverse_held(self, is_reversed):
        print(f'Spike3DRasterWindowWidget.on_reverse_held(is_reversed: {is_reversed})')
        pass
    
    ########################################################
    ## For SpikeRasterLeftSidebarControlsMixin conformance:
    ########################################################
    @pyqtExceptionPrintingSlot(float)
    def on_animation_timestep_valueChanged(self, updated_val):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_animation_timestep_valueChanged(updated_val: {updated_val})')
        old_value = self.animation_time_step
        self.animation_time_step = updated_val
        
    @pyqtExceptionPrintingSlot(float)
    def on_temporal_zoom_factor_valueChanged(self, updated_val):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_temporal_zoom_factor_valueChanged(updated_val: {updated_val})')
        old_value = self.temporal_zoom_factor        
        self.temporal_zoom_factor = updated_val
                
    @pyqtExceptionPrintingSlot(float)
    def on_render_window_duration_valueChanged(self, updated_val):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_render_window_duration_valueChanged(updated_val: {updated_val})')
        old_value = self.render_window_duration
        self.render_window_duration = updated_val
        # TODO 2023-03-29 19:14: - [ ] need to set self.render_window_duration.timeWindow.window_duration = updated_val
        


    ########################################################
    ## For SpikeRasterBottomFrameControlsMixin conformance:
    ########################################################

    @pyqtExceptionPrintingSlot(str)
    def perform_jump_next_series_item(self, curr_jump_series_name):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        # curr_jump_series_idx, curr_jump_series_name = self.ui.bottomPlaybackControlBarWidget.get_current_jump_target_series_selection() # (0, 'PBEs')
        # print(f'perform_jump_next_series_item(): curr_jump_series_idx: {curr_jump_series_idx}, curr_jump_series_name: {curr_jump_series_name}')

        ## Get Interval Datasources:
        
        interval_datasources = self.spike_raster_plt_2d.interval_datasources
        assert curr_jump_series_name in interval_datasources, f"curr_jump_series_name: '{curr_jump_series_name}' not in interval_datasources: {interval_datasources}"
        selected_rendered_interval_series_ds = interval_datasources[curr_jump_series_name] # IntervalsDatasource
        # selected_rendered_interval_series.time_column_names # ['t_start', 't_duration', 't_end']
        selected_rendered_interval_series_times_df = selected_rendered_interval_series_ds.time_column_values # ['t_start', 't_duration', 't_end']
        ## Get current time window:
        curr_time_window = self.animation_active_time_window.active_time_window # (45.12114057149739, 60.12114057149739)
        ## Find the events beyond that time:
        filtered_times_df = selected_rendered_interval_series_times_df[(selected_rendered_interval_series_times_df['t_start'].to_numpy() > curr_time_window[0])] #.first(0) #.iat[0,:]
        next_target_jump_time = filtered_times_df['t_start'].to_numpy()[0]
        print(f'curr_time_window: {curr_time_window}, next_target_jump_time: {next_target_jump_time}')
        # jump_change_time = next_target_jump_time - curr_time_window[0]
        # print(f'jump_change_time: {jump_change_time}')
        ## Update the window:
        self.update_animation(next_target_jump_time)


    @pyqtExceptionPrintingSlot(str)
    def perform_jump_prev_series_item(self, curr_jump_series_name):
        """ seeks the current active_time_Window to the start of the previous epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the previous epoch event
        """
        ## Get Interval Datasources:
        interval_datasources = self.spike_raster_plt_2d.interval_datasources
        assert curr_jump_series_name in interval_datasources, f"curr_jump_series_name: '{curr_jump_series_name}' not in interval_datasources: {interval_datasources}"
        selected_rendered_interval_series_ds = interval_datasources[curr_jump_series_name] # IntervalsDatasource
        selected_rendered_interval_series_times_df = selected_rendered_interval_series_ds.time_column_values
        ## Get current time window:
        curr_time_window = self.animation_active_time_window.active_time_window # (45.12114057149739, 60.12114057149739)
        ## Find the events beyond that time:
        filtered_times_df = selected_rendered_interval_series_times_df[(selected_rendered_interval_series_times_df['t_start'].to_numpy() < curr_time_window[0])]
        next_target_jump_time = filtered_times_df['t_start'].to_numpy()[-1]
        print(f'curr_time_window: {curr_time_window}, next_target_jump_time: {next_target_jump_time}')
        # jump_change_time = next_target_jump_time - curr_time_window[0]
        # print(f'jump_change_time: {jump_change_time}')
        ## Update the window:
        self.update_animation(next_target_jump_time)


    @pyqtExceptionPrintingSlot(float, float)
    def perform_jump_specific_timestamp(self, next_start_timestamp: float, window_duration: float=None):
        """ Jumps to a specific time window (needs window size too)
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.perform_jump_specific_timestamp(next_start_timestamp: {next_start_timestamp}, window_duration: {window_duration})')
        
        # Set the window_duration first so it fits the window:
        if window_duration is not None:
            if window_duration != self.animation_active_time_window.window_duration:
                if self.enable_debug_print:
                    print(f'perform_jump_specific_timestamp(): window_duration changed: new_window_duration {window_duration} != self.animation_active_time_window.window_duration: {self.animation_active_time_window.window_duration}')
                self.animation_active_time_window.timeWindow.window_duration = window_duration
                # TODO 2023-03-29 19:18: - [ ] See if anything needs to be updated manually when window duration changes.
        self.update_animation(next_start_timestamp)
        



    @pyqtExceptionPrintingSlot(str)
    def perform_interval_series_remove_item(self, curr_series_name):
        """ Removes the interval series with the name specified by curr_series_name
        """
        ## Get Interval Datasources:
        interval_datasources = self.spike_raster_plt_2d.interval_datasources
        assert curr_series_name in interval_datasources, f"curr_series_name: '{curr_series_name}' not in interval_datasources: {interval_datasources}"
        self.spike_raster_plt_2d.remove_rendered_intervals(name=curr_series_name)

    @pyqtExceptionPrintingSlot(str)
    def perform_interval_series_customize_item(self, curr_series_name):
        """ Launches a customization dialog for the interval series with the name specified by curr_series_name
        """
        ## Get Interval Datasources:
        interval_datasources = self.spike_raster_plt_2d.interval_datasources
        assert curr_series_name in interval_datasources, f"curr_series_name: '{curr_series_name}' not in interval_datasources: {interval_datasources}"
        ## TODO: not yet implemented
        print(f'perform_series_customize_item(curr_series_name: "{curr_series_name}"): NOT YET IMPLEMENTED')


    @pyqtExceptionPrintingSlot()
    def perform_interval_series_clear_all(self):
        """ Removes all rendered interval series
        """
        self.spike_raster_plt_2d.clear_all_rendered_intervals()

    @pyqtExceptionPrintingSlot()
    def perform_interval_series_request_add(self):
        """ Launches a dialog to add new rendered interval series
        """
        ## TODO: not yet implemented
        print(f'perform_interval_series_request_add(): NOT YET IMPLEMENTED')
        # self.spike_raster_plt_2d.clear_all_rendered_intervals()

    ########################################################
    ## For GlobalConnectionManagerAccessingMixin conformance:
    ########################################################
    
    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_setup(self):
        """ perfrom registration of drivers/drivables:"""
        ## register children:
        
        # Register the 2D plotter as both drivable and a driver:
        self.connection_man.register_driver(self.ui.spike_raster_plt_2d)
        self.connection_man.register_drivable(self.ui.spike_raster_plt_2d)
        
        if self.ui.spike_raster_plt_3d is not None:
            self.connection_man.register_drivable(self.ui.spike_raster_plt_3d)
            self.connect_plotter_time_windows()

    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released
        
        TODO: call this at some point
        """
        print(f'GlobalConnectionManagerAccessingMixin_on_destroy()')
        ## unregister children:
        self.connection_man.unregister_object(self.ui.spike_raster_plt_2d)
        if self.ui.spike_raster_plt_3d is not None:
            self.connection_man.unregister_object(self.ui.spike_raster_plt_3d)
        

    ########################################################
    ## For Other conformances:
    ########################################################
    @pyqtExceptionPrintingSlot()
    def on_spikes_df_changed(self):
        """ changes:
            self.fragile_linear_neuron_IDXs
            self.n_full_cell_grid
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_spikes_df_changed()')
        
    @pyqtExceptionPrintingSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        # TODO 2023-03-29 19:03: - [ ] Shouldn't this at least update the plots like on_window_changed does? I know duration changing is more involved than just start_t changing.


    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_t, end_t)
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    
    @pyqtExceptionPrintingSlot(float, float, float, object)
    def on_windowed_data_window_duration_changed(self, start_t, end_t, duration, updated_data_value):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration}, updated_data_value: ...)')

    @pyqtExceptionPrintingSlot(float, float, object)
    def on_windowed_data_window_changed(self, start_t, end_t, updated_data_value):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_changed(start_t: {start_t}, end_t: {end_t}, updated_data_value: ...)')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_t, end_t)
        
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    
    def update_neurons_color_data(self, updated_neuron_render_configs):
        """ Propagates the neuron color updates to any valid children that need these updates.
        """
        if self.spike_raster_plt_2d is not None:
            self.spike_raster_plt_2d.update_neurons_color_data(updated_neuron_render_configs)
            
        if self.spike_raster_plt_3d is not None:
            self.spike_raster_plt_3d.update_neurons_color_data(updated_neuron_render_configs)
        
    

    ########################################################
    ## For Key Press Events:
    ########################################################

    ##-----------------------------------------
    def keyPressEvent(self, event):
        if self.should_debug_print_interaction_events:
            print(f'pressed from Spike3DRasterWindowWidget.keyPressEvent(event): {event.key()}')
            print(f'\t event.modifiers(): {event.modifiers()}')
            # e.Modifiers()
            print('event received @ Spike3DRasterWindowWidget')
        super(Spike3DRasterWindowWidget, self).keyPressEvent(event)
        if self.should_debug_print_interaction_events:
            if event.key() == QtCore.Qt.Key_Space:
                print(f'\t detected event: {event.key()}')
            elif event.key() == QtCore.Qt.Key_0:
                print(f'\t detected event: {event.key()}')
            else:
                print(f'\t undetected event')
        # self.keyPressed.emit(event)

    ##-----------------------------------------
    
    @QtCore.Property(int) # Note that this ia *pyqt*Property, meaning it's available to pyqt
    def scheduledAnimationSteps(self):
        """The scheduledAnimationSteps property."""
        return self._scheduledAnimationSteps
    @scheduledAnimationSteps.setter
    def scheduledAnimationSteps(self, value):
        if self._scheduledAnimationSteps != value:
            # Only update if the value has changed from the previous one:
            self._scheduledAnimationSteps = value
            # TODO: maybe use a rate-limited signal that's emitted instead so this isn't called too often during interpolation?
            # self.shift_animation_frame_val(self._scheduledAnimationSteps) # TODO: this isn't quite right
            
    def onScrollingTimelineAnimationFinished(self):
        """ used for the QTimeline version of the smooth scrolling animation """
        print(f'onScrollingTimelineAnimationFinished()')
        print(f'\t self._scheduledAnimationSteps: {self._scheduledAnimationSteps}')
        self.scheduledAnimationSteps = 0 # updated method that actually zeros out the scheduled scalings        
        print('\t zeroing out.')
        # if self._scheduledAnimationSteps > 0:
        #     self._scheduledAnimationSteps -= 1
        # else:
        #     self._scheduledAnimationSteps += 1
        

    def onScrollingTimelineFired(self, x):
        """ used for the QTimeline version of the smooth scrolling animation 
        
        # OLD VERSION: x appears to be a float between 0.0-1.0 by default that indicates how far along in the animation it is
        
        x is an int indicating the number of frames for the timeline that were set with between 0.0-1.0 by default that indicates how far along in the animation it is
        
        """
        print(f'onScrollingTimelineFired(x: {x})')
        # self.shift_animation_frame_val(x)        
        curr_shifted_next_start_time = self.compute_frame_shifted_start_timestamp(x)
        print(f'\t curr_shifted_next_start_time: {curr_shifted_next_start_time}')
        self.update_animation(curr_shifted_next_start_time)
        self._scheduledAnimationSteps = self._scheduledAnimationSteps - x # subtract off the frames that have been shifted
        
    def eventFilter(self, watched, event):
        """  has to be installed on an item like:
            self.grid = pg.GraphicsLayoutWidget()
            self.top_left = self.grid.addViewBox(row=1, col=1)
            self.top_left.installEventFilter(self)
        
        """
        # print(f'Spike3DRasterWindowWidget.eventFilter(self, watched, event)')
        if event.type() == QtCore.QEvent.GraphicsSceneWheel:
            # QtCore.QEvent.GraphicsSceneWheel
            """ 
            
            event.delta(): (gives values like +/- 120, 240, etc) # Returns the distance that the wheel is rotated, in eighths (1/8s) of a degree. A positive value indicates that the wheel was rotated forwards away from the user; a negative value indicates that the wheel was rotated backwards toward the user.
                Most mouse types work in steps of 15 degrees, in which case the delta value is a multiple of 120 (== 15 * 8).

            event.orientation(): 1 for alternative scroll wheel dir and 2 for primary scroll wheel dir
            
            """
            if self.should_debug_print_interaction_events:
                print(f'Spike3DRasterWindowWidget.eventFilter(...)\n\t detected event.type() == QtCore.QEvent.GraphicsSceneWheel')
                print(f'\twatched: {watched}\n\tevent: {event}')
                print(f'\tevent.delta(): {event.delta()}')
                print(f'\tevent.orientation(): {event.orientation()}')
                # print(f'\tevent.phase(): {event.phase()}')
                # print(f'\tevent.pixelDelta(): {event.pixelDelta()}')
                
            
            numDegrees = event.delta() / 8
            numSteps = numDegrees / 15 # see QWheelEvent documentation
            numSteps = int(round(float(self.params.scrollStepMultiplier) * float(numSteps)))
                       
            updatedNumScheduledScalings = self._scheduledAnimationSteps + numSteps
            if (updatedNumScheduledScalings * numSteps < 0):
                updatedNumScheduledScalings = numSteps # if user moved the wheel in another direction, we reset previously scheduled scalings
            
            if self.enable_smooth_scrolling_animation:
                # ## pyqt Property Animation Method:            
                # self.ui.scrollAnim.setEndValue(updatedNumScheduledScalings) # Update the end value
                # self.ui.scrollAnim.start() # start the animation
                
                ## QTimeline version:
                self._scheduledAnimationSteps = updatedNumScheduledScalings # Set the updated number of scalings:
                self.ui.scrollAnimTimeline.setEndFrame(self._scheduledAnimationSteps)
                self.ui.scrollAnimTimeline.start() # Start the timeline's animation event
            else:
                # No animation, just update directly ("old way")
                self._scheduledAnimationSteps = updatedNumScheduledScalings
                self.shift_animation_frame_val(self._scheduledAnimationSteps)
                self._scheduledAnimationSteps = 0 # New method: zero it out instead of having it compound

            return True
        else:
            if self.should_debug_print_interaction_events:
                print(f'\t unhandled event {QEventLookupHelpers.get_event_string(event)}')
        # If not a particularlly handled case, do the default thing.
        return super().eventFilter(watched, event)
    
    
    ##-----------------------------------------
    def wheelEvent(self, event):
        super(Spike3DRasterWindowWidget, self).wheelEvent(event)
        if self.should_debug_print_interaction_events:
            print(f'Spike3DRasterWindowWidget.wheelEvent(...)')
            # self.x = self.x + event.delta()/120
            # print self.x
            # self.label.setText("Total Steps: "+QString.number(self.x))        
            print(f'\t wheelEvent(event: {event}')
    

    @classmethod
    def find_or_create_if_needed(cls, curr_active_pipeline, force_create_new:bool=False, **kwargs):
        """ Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
        Usage:
        
        from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
        spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline)

        
        """
        # Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
        from pyphocorehelpers.gui.Qt.TopLevelWindowHelper import TopLevelWindowHelper
        import pyphoplacecellanalysis.External.pyqtgraph as pg # Used to get the app for TopLevelWindowHelper.top_level_windows
        ## For searching with `TopLevelWindowHelper.all_widgets(...)`:

        found_spike_raster_windows = TopLevelWindowHelper.all_widgets(pg.mkQApp(), searchType=cls)

        if len(found_spike_raster_windows) < 1:
            # no existing spike_raster_windows. Make a new one
            print(f'no existing SpikeRasterWindow. Creating a new one.')
            # Create a new `SpikeRaster2D` instance using `_display_spike_raster_pyqtplot_2D` and capture its outputs:
            active_2d_plot, active_3d_plot, spike_raster_window = curr_active_pipeline.plot._display_spike_rasters_pyqtplot_2D(**kwargs).values()
        else:
            print(f'found {len(found_spike_raster_windows)} existing Spike3DRasterWindowWidget windows using TopLevelWindowHelper.all_widgets(...). Will use the most recent.')
            if force_create_new:
                print(f'force_create_new=True. Creating a new Spike3DRasterWindowWidget.')
                # Create a new `SpikeRaster2D` instance using `_display_spike_raster_pyqtplot_2D` and capture its outputs:
                active_2d_plot, active_3d_plot, spike_raster_window = curr_active_pipeline.plot._display_spike_rasters_pyqtplot_2D(**kwargs).values()
            else:
                # assert len(found_spike_raster_windows) == 1, f"found {len(found_spike_raster_windows)} Spike3DRasterWindowWidget windows using TopLevelWindowHelper.all_widgets(...) but require exactly one."
                # Get the most recent existing one and reuse that:
                spike_raster_window = None
                # spike_raster_window = found_spike_raster_windows[0]
                for a_canidate_window in found_spike_raster_windows:
                    if spike_raster_window is None:
                        if a_canidate_window.isVisible():
                            spike_raster_window = a_canidate_window
    
                if spike_raster_window is None:
                    raise ValueError(f'WARNING: found no open windows for spike_raster_window out of {len(found_spike_raster_windows)} candidate options!!!')

                # Extras:
                active_2d_plot = spike_raster_window.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
                active_3d_plot = spike_raster_window.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>

        main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem

        return spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget)
    
    

    # ==================================================================================================================== #
    # Neuron Visual Configs Widget                                                                                         #
    # ==================================================================================================================== #

    @property
    def neuron_visual_config_widget_container(self) -> NeuronWidgetContainer:
        try:
            return self.ui.rightSideContainerWidget.ui.neuron_widget_container
        except (AttributeError, KeyError):
            print(f'missing self.ui.rightSideContainerWidget.ui.neuron_widget_container. returning None.')
            return None
        except Exception:
            """ unhandled exception """
            raise


    def on_neuron_color_display_config_changed(self, new_config):
        """ The function called when the neuron color is changed in the widget

        Recieves a SingleNeuronPlottingExtended config

        Usage:
            for a_widget in pf_widgets:
                # Connect the signals to the debugging slots:
                a_widget.spike_config_changed.connect(_on_spike_config_changed)
                a_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        """
        print(f'_on_neuron_color_display_config_changed(new_config: {new_config})')

        if isinstance(new_config, SingleNeuronPlottingExtended):
            # wrap it in a single-element dict before passing:
            new_config = {int(new_config.name):new_config}

        # extracted_neuron_id_updated_colors_map = {int(a_config.name):a_config.color for a_config in new_config}

        # Update the raster when the configs change:
        self.update_neurons_color_data(new_config)



    ### Callbacks for NeuronWidgetContainer and `spike_raster_window`
    def update_neuron_config_widgets_from_raster(self, *arg, block_signals: bool=True, **kwargs):
        """ The function called when the neuron color is changed.
        Implicitly captures spike_raster_window, active_raster_plot, neuron_widget_container

        Recieves a SingleNeuronPlottingExtended config

        Raster -> Configs

        Usage:
            for a_widget in pf_widgets:
                # Connect the signals to the debugging slots:
                a_widget.spike_config_changed.connect(_on_spike_config_changed)
                a_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        """
        print(f'update_neuron_config_widgets_from_raster()')
        # # Set colors from the raster:
        # neuron_plotting_configs_dict: Dict = DataSeriesColorHelpers.build_cell_display_configs(active_2d_plot.neuron_ids, colormap_name='PAL-relaxed_bright', colormap_source=None)

        ## Get 2D or 3D Raster from spike_raster_window
        active_raster_plot = self.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
        if active_raster_plot is None:
            active_raster_plot = self.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
            assert active_raster_plot is not None


        ## Get the configs from the `active_raster_plot` widget's colors:
        neuron_plotting_configs_dict: Dict = DataSeriesColorHelpers.build_cell_display_configs(active_raster_plot.neuron_ids, deepcopy(active_raster_plot.params.neuron_qcolors))
        neuron_widget_container = self.ui.rightSideContainerWidget.ui.neuron_widget_container
        
        # TODO apply to neuron_widget_container
        if block_signals:
            neuron_widget_container.blockSignals(True) # Block signals so it doesn't recursively update
        neuron_widget_container.applyUpdatedConfigs(neuron_plotting_configs_dict)
        if block_signals:
            neuron_widget_container.blockSignals(False)




    def _perform_build_attached_neuron_visual_configs_widget(self, neuron_plotting_configs_dict: Dict):
        # Standalone:
        # neuron_widget_container = NeuronWidgetContainer(neuron_plotting_configs_dict)
        # neuron_widget_container.show()

        assert not hasattr(self.ui.rightSideContainerWidget.ui, 'neuron_widget_container')
        
        ## Render in right sidebar:
        self.ui.rightSideContainerWidget.ui.neuron_widget_container = NeuronWidgetContainer(neuron_plotting_configs_dict, parent=self.right_sidebar_contents_container)
        ## add reference to sidebar.ui.neuron_widget_container
        self.right_sidebar_contents_container.addWidget(self.ui.rightSideContainerWidget.ui.neuron_widget_container)

        ## Connect	
        ## TODO: use `self.connection_man`?

        _connections_list = []
        for curr_widget in self.ui.rightSideContainerWidget.ui.neuron_widget_container.config_widgets:        
            # Connect the signals to the widgets:
            # curr_widget.spike_config_changed.connect(lambda are_included, spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included, cell_id_copy=neuron_id: spikes_config_changed_callback(neuron_IDXs=None, cell_IDs=[cell_id_copy], are_included=are_included))
            # # curr_widget.spike_config_changed.connect(_on_spike_config_changed)
            # curr_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
            _connections_list.append(curr_widget.sig_neuron_color_changed.connect(self.on_neuron_color_display_config_changed))

        self.ui.rightSideContainerWidget.ui.neuron_widget_container.rebuild_neuron_id_to_widget_map()
        
        _connections_list.append(self.ui.rightSideContainerWidget.ui.neuron_widget_container.sigRevert.connect(self.update_neuron_config_widgets_from_raster))
        

        return self.ui.rightSideContainerWidget.ui.neuron_widget_container, _connections_list
    
        
    def build_neuron_visual_configs_widget(self, build_new_neuron_colormap:bool=False):
        """ addds to the right sidebar and connects controls """
        ## Get 2D or 3D Raster from spike_raster_window
        active_raster_plot = self.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
        if active_raster_plot is None:
            active_raster_plot = self.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
            assert active_raster_plot is not None


        ## Backup Existing Colors:
        # _plot_backup_colors = ColorData.backup_raster_colors(active_raster_plot) # note that they are all 0.0-1.0 format. RGBA
        # deepcopy(active_raster_plot.params.neuron_qcolors) #, active_raster_plot.params.neuron_qcolors_map

        # Build updated configs from the raster_plot's colors:
        if build_new_neuron_colormap:
            # builds a new colormap
            print(f'building new colormap')
            neuron_plotting_configs_dict: Dict = DataSeriesColorHelpers.build_cell_display_configs(active_raster_plot.neuron_ids, colormap_name='PAL-relaxed_bright', colormap_source=None)
        else:
            # Uses the existing neuron colors from self:
            neuron_plotting_configs_dict: Dict = DataSeriesColorHelpers.build_cell_display_configs(active_raster_plot.neuron_ids, deepcopy(active_raster_plot.params.neuron_qcolors))
        
        # neuron_widget_container, _connections_list = add_neuron_display_config_widget(self)
         
        self.ui.rightSideContainerWidget.ui.neuron_widget_container, _connections_list = self._perform_build_attached_neuron_visual_configs_widget(neuron_plotting_configs_dict)

        # Display the sidebar:
        self.set_right_sidebar_visibility(is_visible=True)
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterWindowWidget()
    # testWidget.show()
    sys.exit(app.exec_())

