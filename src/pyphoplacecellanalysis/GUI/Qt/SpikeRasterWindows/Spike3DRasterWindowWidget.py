from copy import deepcopy
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets, QtGui
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.Qt.GlobalConnectionManager import GlobalConnectionManager, GlobalConnectionManagerAccessingMixin
from pyphocorehelpers.gui.Qt.qevent_lookup_helpers import QEventLookupHelpers
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowBase import Ui_RootWidget # Generated file from .ui

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo

from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.LoggingOutputWidget.LoggingOutputWidget import LoggingOutputWidget
from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController, TimeWindowPlaybackControllerActionsMixin

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar, SpikeRasterBottomFrameControlsMixin
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarWidget import Spike3DRasterLeftSidebarControlBar, SpikeRasterLeftSidebarControlsMixin
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget import Spike3DRasterRightSidebarWidget, SpikeRasterRightSidebarOwningMixin

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin

from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers, UnitColoringMode, ColorData
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.GUI.Qt.NeuronVisualSelectionControls.NeuronVisualSelectionControlsWidget import NeuronVisualSelectionControlsWidget, NeuronWidgetContainer, add_neuron_display_config_widget

from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass, LoggingBaseClassLoggerOwningMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode

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
    enable_interaction_events_debug_print = True
    # enable_interaction_events_debug_print = False
    
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
        return (self.enable_debug_print and self.enable_interaction_events_debug_print) # note self is accessing `Spike3DRasterWindowWidget.enable_interaction_events_debug_print`
    
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
    

    @property
    def menu_action_history_list(self) -> List:
        """The menu_action_history_list property."""
        # return PhoMenuHelper.try_get_menu_window(self).ui.menus._menu_action_history_list # Window?
        # if not self.params.has_attr('_menu_action_history_list'):
        #     self.params._menu_action_history_list = [] ## a list to show the history
        return self.params._menu_action_history_list ## Spike3DRasterWindowWidget
        # return self.spike_raster_plt_2d.ui._menu_action_history_list ## Spike3DRasterWindowWidget
    @menu_action_history_list.setter
    def menu_action_history_list(self, value):
        # self.spike_raster_plt_2d.ui._menu_action_history_list = value
        self.params._menu_action_history_list = value
        # PhoMenuHelper.try_get_menu_window(self).ui.menus._menu_action_history_list = value # window


    def __init__(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, application_name=None, type_of_3d_plotter='pyqtgraph', params_kwargs=None, parent=None):
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
        QtWidgets.QWidget.__init__(self, parent=parent) # note kwargs (`self=self`), as in `QtWidgets.QWidget.__init__(self=self, parent=parent)` results in `TypeError: descriptor '__init__' of 'sip.simplewrapper' object needs an argument`
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
        
        if params_kwargs is None:
            params_kwargs = {}

        self.params = VisualizationParameters(self.applicationName, _menu_action_history_list=[], type_of_3d_plotter=type_of_3d_plotter, is_crosshair_trace_enabled=False, debug_print=False, **params_kwargs)
        self.params.type_of_3d_plotter = type_of_3d_plotter
        self.params._menu_action_history_list = []
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
        
        ## Starts the delayed_gui_itemer which will run after 1-second to update the GUI:
        self._delayed_gui_timer = QtCore.QTimer(self)
        self._delayed_gui_timer.timeout.connect(self._run_delayed_gui_load_code)
        #Set the interval and start the timer.
        self._delayed_gui_timer.start(1000)

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
        self.ui.additional_connections = {}
        
        ## Create the animation properties:
        self.playback_controller = TimeWindowPlaybackController()
        self.playback_controller.setup(self) # pass self to have properties set
        
        ## Connect the UI Controls:
        # Helper Mixins: buildUI:
        self.SpikeRasterLeftSidebarControlsMixin_on_buildUI() # `self.ui.leftSideToolbarWidget` Call this to set the initial values for the UI before signals are connected.
        self.SpikeRasterRightSidebarOwningMixin_on_buildUI() # `self.ui.rightSideContainerWidget`


        # self.ui.bottom_controls_frame, self.ui.bottom_controls_layout = self.SpikeRasterBottomFrameControlsMixin_on_buildUI() # NOTE: do not call for the window as it already has a valid bottom bar widget
        # Connect the signals:
        self.ui.bottom_bar_connections = None 
        self.ui.bottom_bar_connections = self.SpikeRasterBottomFrameControlsMixin_connectSignals(self.ui.bottomPlaybackControlBarWidget)
        self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.sig_joystick_delta_occured.connect(self.perform_slide_relative_time))
        if self.ui.spike_raster_plt_2d is not None:
            ## connect the btnAddDockedTrack to the 2D plotter's add track function:
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.sigAddDockedTrackRequested.connect(lambda: self.ui.spike_raster_plt_2d.add_new_matplotlib_render_plot_widget(name='newDockedWidget', sync_mode=SynchronizedPlotMode.TO_WINDOW)))

        self.ui.left_side_bar_connections = None
        self.ui.left_side_bar_connections = self.SpikeRasterLeftSidebarControlsMixin_connectSignals(self.ui.leftSideToolbarWidget)

        self.ui.right_side_bar_connections = None
        self.ui.right_side_bar_connections = self.SpikeRasterRightSidebarOwningMixin_connectSignals(self.ui.rightSideContainerWidget)

        
        

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
            
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_specific_time.connect(self.perform_jump_specific_timestamp_only))
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_specific_time_window.connect(lambda start_t, end_t: self.perform_jump_specific_timestamp(start_t, (end_t - start_t))))
            


            # self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_specific_time.connect(self.update_animation)
            # self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.jump_specific_time.connect((lambda new_time: self.update_animation(new_time))))
            self.ui.bottom_bar_connections.append(self.ui.spike_raster_plt_2d.sigEmbeddedMatplotlibDockWidgetAdded.connect(lambda spike_raster_plt_2D, added_dock_item, added_widget: self.update_scrolling_event_filters())) ## not really a bottom_bar_connections, but who cares
                        
            self.ui.bottom_bar_connections.append(self.ui.bottomPlaybackControlBarWidget.sigToggleRightSidebarVisibility.connect(lambda is_visible: self.set_right_sidebar_visibility(is_visible)))
            
            # ## update the jump time when it scrolls
            # self.ui.bottomPlaybackControlBarWidget.time_fractional_seconds
            
            ## update the dynamic event filters if needed
            self.update_scrolling_event_filters()
            
            ## `LiveWindowEventIntervalMonitoringMixin` event setup
            connections = {}
            connections['LiveWindowEventIntervalMonitoringMixin_entered'] = self.ui.spike_raster_plt_2d.sigOnIntervalEnteredWindow.connect(self.on_visible_event_intervals_added)
            connections['LiveWindowEventIntervalMonitoringMixin_exited'] = self.ui.spike_raster_plt_2d.sigOnIntervalExitedindow.connect(self.on_visible_event_intervals_removed)
            self.ui.bottom_bar_connections.append(connections['LiveWindowEventIntervalMonitoringMixin_entered'])
            self.ui.bottom_bar_connections.append(connections['LiveWindowEventIntervalMonitoringMixin_exited'])
            
            
        # END if self.ui.spike_raster_plt_2d is not None
            

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
            
        



    def _run_delayed_gui_load_code(self):
        """ called when the self._delayed_gui_timer QTimer fires. """
        #Stop the timer.
        self._delayed_gui_timer.stop()
        print(f'Spike3DRasterWindowWidget._run_delayed_gui_load_code() called!')
        ## Make sure to set the initial linear scroll region size/location to something reasonable and not cut-off so the user can adjust it:
        bottom_playback_control_bar_widget = self.bottom_playback_control_bar_widget # Spike3DRasterBottomPlaybackControlBar
        if bottom_playback_control_bar_widget is not None:
            # comboActiveJumpTargetSeries = bottom_playback_control_bar_widget.ui.comboActiveJumpTargetSeries # QComboBox 
            bottom_playback_control_bar_widget.current_selected_jump_target_series_name = 'Laps' ## tries to select the "Laps" epochs fromt he jump-to-comobo box

        self.init_left_and_bottom_bar_times_from_active_window() ## Initialize




    @function_attributes(short_name=None, tags=['interactivity', 'event', 'scrolling', 'children'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-18 10:29', related_items=[])
    def update_scrolling_event_filters(self, debug_print=False):
        """ enables scrollability in the added matplotlib views just like the two upper views 
        
        Needs to be done when any new child widget is added, right?
        
        """
        #   .installEventFilter(self)
        ## Install the event filter in the 2D View to enable scroll wheel events:
        if self.ui.spike_raster_plt_2d is not None: 
            if self.ui.spike_raster_plt_2d.ui.matplotlib_view_widgets is not None:
                for a_name, a_time_sync_widget in self.ui.spike_raster_plt_2d.ui.matplotlib_view_widgets.items():
                    if debug_print:
                        print(f'a_name: {a_name}')
                        
                    try:
                        # a_time_sync_widget.installEventFilter(self) # plots.preview_overview_scatter_plot is a ScatterPlotItem ... does it have to be a pyqtgraph subclass to do this? I'm worried it does
                        a_time_sync_widget.ui.canvas.installEventFilter(self)
                    except KeyError as e:
                        ## expect `KeyError: 'canvas'` for pyqtgraph-based sync widgets (as opposed to matplotlib-based ones)
                        pass ## just skip
                    except Exception as e:
                        raise



    def __str__(self):
         return
      

    # ==================================================================================================================== #
    # GlobalConnectionManagerAccessingMixin Conformances                                                                   #
    # ==================================================================================================================== #
    
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
        if hasattr(controlled_plt, 'spikes_window'):
            return self.connection_man.connect_drivable_to_driver(drivable=controlled_plt, driver=self.spike_raster_plt_2d,
                                                custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.spikes_window.update_window_start_end_rate_limited)))

        if hasattr(controlled_plt, 'on_window_changed_rate_limited'):
            return self.connection_man.connect_drivable_to_driver(drivable=controlled_plt, driver=self.spike_raster_plt_2d,
                                                        custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
                
        else:
            raise NotImplementedError(f'') 



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
    

    @pyqtExceptionPrintingSlot(float)
    def update_animation(self, next_start_timestamp: float):
        """ Actually updates the animation given the next_start_timestep
            extracted from Spike3DRasterWindowWidget.shift_animation_frame_val(...)
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.update_animation(next_start_timestamp: {next_start_timestamp})')
        # self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        next_end_timestamp = next_start_timestamp+self.animation_active_time_window.window_duration
        
        # Update the windows once before showing the UI:
        self.spike_raster_plt_2d.update_scroll_window_region(next_start_timestamp, next_end_timestamp, block_signals=True) # self.spike_raster_plt_2d.window_scrolled should be emitted        
        # signal emit:
        self.spike_raster_plt_2d.window_scrolled.emit(next_start_timestamp, next_end_timestamp)
        # update_scroll_window_region
        # self.ui.spike_raster_plt_3d.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        # self.bottom_playback_control_bar_widget.on_window_changed(next_start_timestamp, next_end_timestamp) ## direct
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(next_start_timestamp, next_end_timestamp) ## indirect 
        self.SpikeRasterLeftSidebarControlsMixin_on_window_update(next_start_timestamp, next_end_timestamp)
        

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

        # Add a guard to prevent circular updates
        if abs(self.render_window_duration - updated_val) < 1e-6:
            # Skip if value is essentially the same
            return
            
        old_value = deepcopy(self.render_window_duration)
        self.render_window_duration = updated_val
        
        window_duration = float(updated_val)
        if window_duration is not None:
            if abs(window_duration - self.animation_active_time_window.window_duration) > 1e-6:
                if self.enable_debug_print:
                    print(f'\ton_render_window_duration_valueChanged(updated_val: {updated_val}): window_duration changed: new_window_duration {window_duration} != self.animation_active_time_window.window_duration: {self.animation_active_time_window.window_duration}')
                # Block programmatic updates from triggering more signals
                self._updating_window_programmatically = True
                try:
                    self.animation_active_time_window.timeWindow.window_duration = window_duration
                finally:
                    self._updating_window_programmatically = False
                    

        # old_value = deepcopy(self.render_window_duration)
        # self.render_window_duration = updated_val
        # # self.spikes_window.window_duration = float(updated_val)
        # # self.spikes_window.timeWindow.window_duration = float(updated_val)

        # # TODO 2023-03-29 19:14: - [ ] need to set self.animation_active_time_window.timeWindow.window_duration = updated_val
        # window_duration = float(updated_val)
        # if window_duration is not None:
        #     if window_duration != self.animation_active_time_window.window_duration:
        #         if self.enable_debug_print:
        #             print(f'\ton_render_window_duration_valueChanged(updated_val: {updated_val}): window_duration changed: new_window_duration {window_duration} != self.animation_active_time_window.window_duration: {self.animation_active_time_window.window_duration}')
        #         self.animation_active_time_window.timeWindow.window_duration = window_duration
        #         # TODO 2023-03-29 19:18: - [ ] See if anything needs to be updated manually when window duration changes.



    def init_left_and_bottom_bar_times_from_active_window(self):
        """ Initializes the left and bottom time controls from the actual visable window
        """
        window_duration = self.animation_active_time_window.timeWindow.window_duration
        start_time, end_time = self.animation_active_time_window.timeWindow.active_time_window
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_time, end_time) ## indirect 
        self.SpikeRasterLeftSidebarControlsMixin_on_window_update(start_time, end_time)
        return (start_time, end_time), window_duration



    # ==================================================================================================================== #
    # Crosshairs                                                                                                           #
    # ==================================================================================================================== #
    # @pyqtExceptionPrintingSlot(bool)
    def on_crosshair_trace_toggled(self):
        # def on_crosshair_trace_toggled(self, updated_is_crosshair_trace_enabled):
        updated_is_crosshair_trace_enabled: bool = self.ui.leftSideToolbarWidget.ui.btnToggleCrosshairTrace.isChecked()
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_crosshair_trace_toggled(updated_is_crosshair_trace_enabled: {updated_is_crosshair_trace_enabled})')
        old_value = self.params.is_crosshair_trace_enabled
        did_update: bool = (old_value != updated_is_crosshair_trace_enabled)
        self.params.is_crosshair_trace_enabled = updated_is_crosshair_trace_enabled
        
        if updated_is_crosshair_trace_enabled:
            ## enable crosshairs callback        
            if self.spike_raster_plt_2d is not None:
                ## set SpikeRaster2D's is_crosshair_trace_enabled
                extant_conn = self.ui.additional_connections.get('spike_3d_to_2d_window_crosshair_connection', None)
                
                # self.spike_raster_plt_2d.sigCrosshairsUpdated.connect(self.on_crosshair_updated_signal)        
                # Rate limited version:
                if extant_conn is None:
                    ## create new connection:
                    # spike_3d_to_2d_window_crosshair_connection = self.connection_man.connect_drivable_to_driver(drivable=self, driver=self.spike_raster_plt_2d,
                    #                                                         custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.sigCrosshairsUpdated, delay=0.2, rateLimit=30, slot=drivable.on_crosshair_updated_signal)))
                    # spike_3d_to_2d_window_crosshair_connection = self.spike_raster_plt_2d.sigCrosshairsUpdated.connect(self.on_crosshair_updated_signal) ## create the new connection
                    # spike_3d_to_2d_window_crosshair_connection = self.spike_raster_plt_2d.sigCrosshairsUpdated.connect(lambda a_child_widget, an_identifier, a_trace_value: self.on_crosshair_updated_signal(an_identifier, a_trace_value))
                    spike_3d_to_2d_window_crosshair_connection = self.spike_raster_plt_2d.sigCrosshairsUpdated.connect(lambda a_child_widget, an_identifier, a_trace_value: self.on_owned_child_crosshair_updated_signal(a_child_widget, an_identifier, a_trace_value))
                    self.ui.additional_connections['spike_3d_to_2d_window_crosshair_connection'] = spike_3d_to_2d_window_crosshair_connection
                else:
                    print(f'\talready had extant connection!')
        else:
            print(f'\tshould disable crosshairs.')
            extant_conn = self.ui.additional_connections.pop('spike_3d_to_2d_window_crosshair_connection', None)            
            if extant_conn is not None:
                ## remove it
                # self.connection_man.disconnect(self.spike_3d_to_2d_window_crosshair_connection)
                self.spike_raster_plt_2d.sigCrosshairsUpdated.disconnect(extant_conn)
                # self.spike_3d_to_2d_window_crosshair_connection = None
                # delattr(self, 'spike_3d_to_2d_window_crosshair_connection') ## remove the attribute


        if (self.spike_raster_plt_2d is not None):
            ## inform spike_raster_plt_2d that traces should be enabled
            self.spike_raster_plt_2d.toggle_crosshair_traces_enabled(updated_is_crosshair_trace_enabled)            



    def on_owned_child_crosshair_updated_signal(self, child, name, trace_value):
        """ called when the crosshair is updated"""
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_owned_child_crosshair_updated_signal(child: {child}, name: "{name}", trace_value: "{trace_value}")')
        left_side_bar_controls = self.ui.leftSideToolbarWidget
        left_side_bar_controls.crosshair_trace_time = trace_value
        
        # self.ui.lblCrosshairTraceStaticLabel.setVisible(True)
        # self.ui.lblCrosshairTraceValue.setVisible(True)
        

    # def on_crosshair_updated_signal(self, child, name, trace_value):
    def on_crosshair_updated_signal(self, name, trace_value):
        """ called when the crosshair is updated"""
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_crosshair_updated_signal(name: "{name}", trace_value: "{trace_value}")')
        left_side_bar_controls = self.ui.leftSideToolbarWidget
        left_side_bar_controls.crosshair_trace_time = trace_value
        
        # self.ui.lblCrosshairTraceStaticLabel.setVisible(True)
        # self.ui.lblCrosshairTraceValue.setVisible(True)


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
        if self.enable_debug_print:
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
        if self.enable_debug_print:
            print(f'curr_time_window: {curr_time_window}, next_target_jump_time: {next_target_jump_time}')
        # jump_change_time = next_target_jump_time - curr_time_window[0]
        # print(f'jump_change_time: {jump_change_time}')
        ## Update the window:
        self.update_animation(next_target_jump_time)


    @pyqtExceptionPrintingSlot(float)
    def perform_jump_specific_timestamp_only(self, next_start_timestamp: float):
        """ Jumps to a specific time window
        """
        return self.perform_jump_specific_timestamp(next_start_timestamp)
    

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
        



    @pyqtExceptionPrintingSlot(float, float)
    def perform_slide_relative_time(self, dx, dy):
        """ dx: change in x-direction of joystick, this is the only one that matters currently (maps to relative t updates
        dy: change in y-dir of joystick

        # sig_joystick_delta_occured
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.perform_slide_relative_time(self, dx: {dx}, dy: {dy})')
        
        # speed_adjustment_factor: float = 1e-3

        speed_adjustment_factor: float = 10.0
        desired_dt: float = (dx * speed_adjustment_factor) # map to reasonable timescale
        
        start_t: float = self.animation_active_time_window.timeWindow.active_window_start_time
        next_start_timestamp: float = start_t + desired_dt

        if self.enable_debug_print:
            print(f'\tdesired_dt: {desired_dt}\n\tdx: {dx}\n\tnext_start_timestamp: {next_start_timestamp}')
            
        # call update animation:
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

    @pyqtExceptionPrintingSlot(float)
    def on_window_start_changed(self, start_t):
        """ 2024-12-19 02:09 - doesn't actually seem to always be called on user scrolling? """
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_start_changed(start_t: {start_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_t, None)
        self.SpikeRasterLeftSidebarControlsMixin_on_window_update(start_t, None)
        
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')


    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        """ 2024-12-19 02:09 - doesn't actually seem to always be called on user scrolling? """
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_t, end_t)
        self.SpikeRasterLeftSidebarControlsMixin_on_window_update(start_t, end_t)
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    
    @pyqtExceptionPrintingSlot(float, float, float, object)
    def on_windowed_data_window_duration_changed(self, start_t, end_t, duration, updated_data_value):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration}, updated_data_value: ...)')

    @pyqtExceptionPrintingSlot(float, float, object)
    def on_windowed_data_window_changed(self, start_t, end_t, updated_data_value):
        """ 2024-12-19 02:09 - doesn't actually seem to always be called on user scrolling? """
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_changed(start_t: {start_t}, end_t: {end_t}, updated_data_value: ...)')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        self.SpikeRasterBottomFrameControlsMixin_on_window_update(start_t, end_t)
        self.SpikeRasterLeftSidebarControlsMixin_on_window_update(start_t, end_t)
        
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    
    def update_neurons_color_data(self, updated_neuron_render_configs):
        """ Propagates the neuron color updates to any valid children that need these updates.
        """
        if self.spike_raster_plt_2d is not None:
            self.spike_raster_plt_2d.update_neurons_color_data(updated_neuron_render_configs)
            
        if self.spike_raster_plt_3d is not None:
            self.spike_raster_plt_3d.update_neurons_color_data(updated_neuron_render_configs)
        
    


    
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
            
    @QtCore.Property(object) # # Note that this ia *pyqt*Property, meaning it's available to pyqt
    def bottom_playback_control_bar_widget(self) -> Spike3DRasterBottomPlaybackControlBar:
        """The bottom_playback_control_bar_widget property."""
        return self.ui.bottomPlaybackControlBarWidget

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


    @function_attributes(short_name=None, tags=['TODO', 'ACTIVE', 'programmatic', 'scrolling', 'time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-18 12:09', related_items=[])
    def programmatically_scroll_to_time(self, new_time):
        numSteps: int = 3
        updatedNumScheduledScalings = self._scheduledAnimationSteps + numSteps
        if (updatedNumScheduledScalings * numSteps < 0):
            updatedNumScheduledScalings = numSteps # if user moved the wheel in another direction, we reset previously scheduled scalings
        
        if self.enable_smooth_scrolling_animation:
            ## QTimeline version:
            self._scheduledAnimationSteps = updatedNumScheduledScalings # Set the updated number of scalings:
            self.ui.scrollAnimTimeline.setEndFrame(self._scheduledAnimationSteps)
            self.ui.scrollAnimTimeline.start() # Start the timeline's animation event
        else:
            # No animation, just update directly ("old way")
            self._scheduledAnimationSteps = updatedNumScheduledScalings
            self.shift_animation_frame_val(self._scheduledAnimationSteps)
            self._scheduledAnimationSteps = 0 # New method: zero it out instead of having it compound


    
    ##-----------------------------------------

    @classmethod
    def find_or_create_if_needed(cls, curr_active_pipeline, force_create_new:bool=False, **kwargs):
        """ Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
        Usage:
        
        from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
        
        spike_raster_window, (active_2d_plot, active_3d_plot, *_out_args) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline, force_create_new=False)
        main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget = _out_args
        
        """
        # Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
        from pyphocorehelpers.gui.Qt.TopLevelWindowHelper import TopLevelWindowHelper
        import pyphoplacecellanalysis.External.pyqtgraph as pg # Used to get the app for TopLevelWindowHelper.top_level_windows
        ## For searching with `TopLevelWindowHelper.all_widgets(...)`:

        found_spike_raster_windows = TopLevelWindowHelper.all_widgets(pg.mkQApp(), searchType=cls)
        _out_args = []
        # _out_args = {}
        
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
                        else:
                            ## close the window
                            a_canidate_window.deleteLater()


                if spike_raster_window is None:
                    raise ValueError(f'WARNING: found no open windows for spike_raster_window out of {len(found_spike_raster_windows)} candidate options!!!')

                # Extras:
                active_2d_plot = spike_raster_window.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
                active_3d_plot = spike_raster_window.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>


        # if active_3d_plot is not None:
        #     _out_args.append(active_3d_plot)
            
        main_graphics_layout_widget = active_2d_plot.ui.get('main_graphics_layout_widget', None) # GraphicsLayoutWidget
        # if main_graphics_layout_widget is not None:
        _out_args.append(main_graphics_layout_widget)
        main_plot_widget = active_2d_plot.plots.get('main_plot_widget', None) # PlotItem
        # if main_plot_widget is not None:
        _out_args.append(main_plot_widget)
        background_static_scroll_plot_widget = active_2d_plot.plots.get('background_static_scroll_window_plot', None) # PlotItem
        # if background_static_scroll_plot_widget is not None:
        _out_args.append(background_static_scroll_plot_widget)

        # main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        # main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        # background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem

        ## Fix window title to display the session context post-hoc
        complete_session_context, (curr_session_context,  additional_session_context) = curr_active_pipeline.get_complete_session_context()
        spike_raster_window.params.complete_session_context = deepcopy(complete_session_context) # Updates `spike_raster_window.params.complete_session_context`
        complete_session_context_window_title_str: str = complete_session_context.get_description(separator='|', include_property_names=False) # 'kdiba|gor01|two|2006-6-07_16-40-19|normal_computed|[1, 2, 4, 6, 7, 8, 9]|5.0'
        spike_raster_window.params.window_title = f"Spike Raster Window - {complete_session_context_window_title_str}" # Updates `spike_raster_window.params.window_title`
        # spike_raster_window.window().setWindowTitle(spike_raster_window.params.window_title) ## sets the window title
        spike_raster_window.setWindowTitle(spike_raster_window.params.window_title)
        # return spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget)
        return spike_raster_window, (active_2d_plot, active_3d_plot, *_out_args)
    
        # return spike_raster_window, (active_2d_plot, active_3d_plot, *_out_args)


    # ==================================================================================================================== #
    # Visible Epoch Intervals in Viewport Window Widgets                                                                   #
    # ==================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['visible_intervals_info', 'table', 'widget', 'ui', 'right-sidebar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-07 18:53', related_items=[])
    def _perform_build_attached_visible_interval_info_widget(self):
        """ Draws a dynamically updating stack of tables in the right sidebar showing the elements in the active intervals
        Called to update or create the tables
        
        Usage:
            visible_intervals_info_widget_container, visible_intervals_ctrl_layout_widget =  self._perform_build_attached_visible_interval_info_widget()

        """
        print(f'spikes_raster_window._perform_build_attached_visible_interval_info_widget()')
        
        is_visible: bool = self.right_sidebar_widget.isVisible() 
        if not is_visible:
            print(f'\tright_sidebar_widget is not Visible, so skipping `._perform_build_attached_visible_interval_info_widget()` update.')
            return None, None
        else:
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig
            from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import TableManager
                            
            ## get the updated data:
            # included_series_names=['Replays', 'Laps', 'PBEs']
            included_series_names=None
            dataframes_dict: Dict[str, pd.DataFrame] = self.find_event_intervals_in_active_window(included_series_names=included_series_names)

            ## see if widgets need to be build or can just be updated:
            needs_init: bool = False
            
            if (not hasattr(self.ui.rightSideContainerWidget.ui, 'visible_intervals_info_widget_container')):
                needs_init = True
            else:
                if (self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container is None):
                    needs_init = True


            if needs_init:
                if self.debug_print:
                    print(f'\t has no .visible_intervals_info_widget_container so NEEDS INIT!')
                assert not hasattr(self.ui.rightSideContainerWidget.ui, 'visible_intervals_info_widget_container')
                self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container = {} # initialize
                
                ## Render in right sidebar:
                updated_ui_dict = {'ctrl_layout': None, 'dynamic_tables_container_widget': None, 'dynamic_tables_container_VBoxLayout': None, 'bottom_spacer_widget': None, 'manager': None} # , 'tables_dict': {}
                root_ctrl_layout_widget = pg.LayoutWidget() ## ROOT layout widget
                updated_ui_dict['ctrl_layout'] = root_ctrl_layout_widget

                # Main Tables container:
                updated_ui_dict['dynamic_tables_container_widget'] = pg.QtWidgets.QWidget()
                updated_ui_dict['dynamic_tables_container_widget'].setObjectName('tables_container')
                updated_ui_dict['dynamic_tables_container_widget'].setSizePolicy(pg.QtWidgets.QSizePolicy.Expanding, pg.QtWidgets.QSizePolicy.Expanding)

                manager = TableManager(updated_ui_dict['dynamic_tables_container_widget'])
                updated_ui_dict['manager'] = manager

                tables_layout = manager.wrapper_layout
                tables_layout.setObjectName('tables_container_VBoxLayout')
                updated_ui_dict['dynamic_tables_container_VBoxLayout'] = tables_layout


                ## add the table vertical layout:
                root_ctrl_layout_widget.addWidget(updated_ui_dict['dynamic_tables_container_widget'], row=1, col=1)
                # Add permanent expanding spacer at the bottom
                # Create an empty widget and set its size policy to expanding
                bottom_spacer_widget = pg.QtWidgets.QWidget()
                bottom_spacer_widget.setObjectName('bottom_spacer')
                bottom_spacer_widget.setSizePolicy(pg.QtWidgets.QSizePolicy.Expanding, pg.QtWidgets.QSizePolicy.Expanding)

                updated_ui_dict['bottom_spacer_widget'] = bottom_spacer_widget # pg.QtWidgets.QSpacerItem(0, 0, pg.QtWidgets.QSizePolicy.Minimum, pg.QtWidgets.QSizePolicy.Expanding)
                # ctrl_layout.layout.addItem(spacer, ctrl_layout.layout.rowCount(), 0, 1, 1)
                root_ctrl_layout_widget.addWidget(updated_ui_dict['bottom_spacer_widget'], row=2, col=1)

                manager.update_tables(dataframes_dict)

                self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container = updated_ui_dict # {k:v for k, v in updated_ui_dict.items()}
                if self.debug_print:
                    print(f'\t done.')
                # VisibleIntervalTable
                rightSideContainerWidget = self.ui.rightSideContainerWidget # pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget.Spike3DRasterRightSidebarWidget
                right_sidebar_contents_container_dockarea = rightSideContainerWidget.right_sidebar_contents_container_dockarea

                # New Dock-based way _________________________________________________________________________________________________ #
                name: str = 'VisibleWindowIntervalTables'
                display_config = CustomDockDisplayConfig(showCloseButton=True, showCollapseButton=True, orientation='horizontal', custom_get_colors_dict={False: DockDisplayColors(fg_color='#111', bg_color='#c5c5c5', border_color='#a7babd'),
                        True: DockDisplayColors(fg_color='#333', bg_color='#757575', border_color='#424242'),
                    })
                
                # Create new widget
                # No extant table widget and display_dock currently, create a new one:
                dDisplayItem = right_sidebar_contents_container_dockarea.find_display_dock(identifier=name) # Dock
                assert dDisplayItem is None
                
                # Add to dynamic dock container 
                _, dDisplayItem = right_sidebar_contents_container_dockarea.add_display_dock(name, display_config=display_config, widget=root_ctrl_layout_widget, dockAddLocationOpts=['bottom'], autoOrientation=False)
                dDisplayItem.setOrientation('horizontal', force=True)
                dDisplayItem.updateStyle()
                dDisplayItem.update()
                
                rightSideContainerWidget.dock_items[name] = dDisplayItem
            


                # Connect ____________________________________________________________________________________________________________ #
                ## TODO: use `self.connection_man`?
                # Rate limited version:`
                self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container['connections'] = {'update_connection': self.connection_man.connect_drivable_to_driver(drivable=self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container['manager'], driver=self.spike_raster_plt_2d,
                                                                custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.002, rateLimit=24, slot=self.on_update_right_sidebar_visible_interval_info_tables))),
                                                            'rendered_interval_list_changed': self.spike_raster_plt_2d.sigRenderedIntervalsListChanged.connect(lambda interval_list: self.on_update_right_sidebar_visible_interval_info_tables()),
                                                            'interval_entered_window': pg.SignalProxy(self.spike_raster_plt_2d.sigOnIntervalEnteredWindow, delay=0.002, rateLimit=2, slot=(lambda _: self.on_update_right_sidebar_visible_interval_info_tables())),
                                                            'interval_exited_window': pg.SignalProxy(self.spike_raster_plt_2d.sigOnIntervalExitedindow, delay=0.002, rateLimit=2, slot=(lambda _: self.on_update_right_sidebar_visible_interval_info_tables())),
                }
                

                ## Dock all Grouped results from `'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView'`
                ## INPUTS: active_2d_plot
                grouped_dock_items_dict = right_sidebar_contents_container_dockarea.get_dockGroup_dock_dict()
                nested_dock_items = {}
                nested_dynamic_docked_widget_container_widgets = {}
                for dock_group_name, flat_group_dockitems_list in grouped_dock_items_dict.items():
                    dDisplayItem, nested_dynamic_docked_widget_container = right_sidebar_contents_container_dockarea.build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
                    nested_dock_items[dock_group_name] = dDisplayItem
                    nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container


                ## show it
                self.set_right_sidebar_visibility(True)
        
            else:
                if self.debug_print:
                    print(f'\t does not need init, just update')
                extant_dict = self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container
                
                extant_manager = extant_dict['manager']
                extant_manager.update_tables(dataframes_dict)
                

            return self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container, self.ui.rightSideContainerWidget.ui.visible_intervals_info_widget_container['ctrl_layout']
    

    def on_update_right_sidebar_visible_interval_info_tables(self):
        """called to update the tables in the right sidebar with the intervals visible in the current window
        """
        # print(f'on_update_right_sidebar_visible_interval_info_tables()')
        self._perform_build_attached_visible_interval_info_widget()
        

    

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
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig
                
        assert not hasattr(self.ui.rightSideContainerWidget.ui, 'neuron_widget_container')
        
        ## Render in right sidebar:
        rightSideContainerWidget = self.ui.rightSideContainerWidget # pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget.Spike3DRasterRightSidebarWidget
        right_sidebar_contents_container_dockarea = rightSideContainerWidget.right_sidebar_contents_container_dockarea
        
        # New Dock-based way _________________________________________________________________________________________________ #
        name: str = 'NeuronVisualConfigs'
        display_config = CustomDockDisplayConfig(showCloseButton=True, orientation='horizontal', custom_get_colors_dict={False: DockDisplayColors(fg_color='#111', bg_color='#c5c5c5', border_color='#a7babd'),
                True: DockDisplayColors(fg_color='#333', bg_color='#757575', border_color='#424242'),
            }, showCollapseButton=True)
        
        # Create new widget
        # No extant table widget and display_dock currently, create a new one:
        dDisplayItem = right_sidebar_contents_container_dockarea.find_display_dock(identifier=name) # Dock
        assert dDisplayItem is None
        
        self.ui.rightSideContainerWidget.ui.neuron_widget_container = NeuronWidgetContainer(neuron_plotting_configs_dict) ## a widget

        # Add to dynamic dock container 
        _, dDisplayItem = right_sidebar_contents_container_dockarea.add_display_dock(name, display_config=display_config, widget=self.ui.rightSideContainerWidget.ui.neuron_widget_container, dockAddLocationOpts=['bottom'], autoOrientation=False)
        dDisplayItem.setOrientation('horizontal', force=True)
        dDisplayItem.updateStyle()
        dDisplayItem.update()
        
        rightSideContainerWidget.dock_items[name] = dDisplayItem
        
        ## Connect Signals
        ## TODO: use `self.connection_man`?
        _connections_list = []
        for curr_widget in self.ui.rightSideContainerWidget.ui.neuron_widget_container.config_widgets:        
            # Connect the signals to the widgets:
            _connections_list.append(curr_widget.sig_neuron_color_changed.connect(self.on_neuron_color_display_config_changed))

        self.ui.rightSideContainerWidget.ui.neuron_widget_container.rebuild_neuron_id_to_widget_map()
        
        _connections_list.append(self.ui.rightSideContainerWidget.ui.neuron_widget_container.sigRevert.connect(self.update_neuron_config_widgets_from_raster))        

        return self.ui.rightSideContainerWidget.ui.neuron_widget_container, _connections_list
    
    @function_attributes(short_name=None, tags=['widget', 'interactive', 'display', 'config', 'intervals', 'epoch', 'visual'], input_requires=[], output_provides=[], uses=['_perform_build_attached_neuron_visual_configs_widget'], used_by=[], creation_date='2025-01-27 16:27', related_items=[])
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
        self.set_right_sidebar_visibility(True)
    


    @function_attributes(short_name=None, tags=['widget', 'interactive', 'display', 'config', 'intervals', 'epoch', 'visual'], input_requires=[], output_provides=[], uses=['EpochRenderConfigsListWidget'], used_by=[], creation_date='2025-01-27 14:06', related_items=[])
    def build_epoch_intervals_visual_configs_widget(self):
        """ addds to the right sidebar and connects controls 
        
        active_raster_plot.ui.epochs_render_configs_widget
        
        """
        from pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget import EpochRenderConfigWidget, EpochRenderConfigsListWidget
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig

        ## Get 2D or 3D Raster from spike_raster_window
        active_raster_plot = self.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
        if active_raster_plot is None:
            active_raster_plot = self.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
            assert active_raster_plot is not None


        ## Backup Existing Colors:
        # _plot_backup_colors = ColorData.backup_raster_colors(active_raster_plot) # note that they are all 0.0-1.0 format. RGBA
        # deepcopy(active_raster_plot.params.neuron_qcolors) #, active_raster_plot.params.neuron_qcolors_map

        rightSideContainerWidget = self.ui.rightSideContainerWidget # pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget.Spike3DRasterRightSidebarWidget
        right_sidebar_contents_container_dockarea = rightSideContainerWidget.right_sidebar_contents_container_dockarea

        # New Dock-based way _________________________________________________________________________________________________ #
        name: str = 'EpochIntervalsVisualConfigs'
        needs_init: bool = True
        
        # Create new widget
        # No extant table widget and display_dock currently, create a new one:
        dDisplayItem = right_sidebar_contents_container_dockarea.find_display_dock(identifier=name) # Dock
        an_extant_epochs_display_list_widget = active_raster_plot.ui.get('epochs_render_configs_widget', None)
        
        if (dDisplayItem is not None) and (an_extant_epochs_display_list_widget is not None):
            needs_init = False
            an_epochs_display_list_widget: EpochRenderConfigsListWidget = an_extant_epochs_display_list_widget
            assert an_epochs_display_list_widget is not None
            

        if needs_init:
            ## INITIALIZE:
            assert dDisplayItem is None
            assert an_extant_epochs_display_list_widget is None
            epoch_display_configs = active_raster_plot.extract_interval_display_config_lists()
            an_epochs_display_list_widget:EpochRenderConfigsListWidget = EpochRenderConfigsListWidget(epoch_display_configs)
            # Add to dynamic dock container
            display_config = CustomDockDisplayConfig(showCloseButton=True, showCollapseButton=True, orientation='horizontal', custom_get_colors_dict={False: DockDisplayColors(fg_color='#111', bg_color='#c5c5c5', border_color='#a7babd'),
                    True: DockDisplayColors(fg_color='#333', bg_color='#757575', border_color='#424242'),
                })
            _, dDisplayItem = right_sidebar_contents_container_dockarea.add_display_dock(name, display_config=display_config, widget=an_epochs_display_list_widget, dockAddLocationOpts=['bottom'], autoOrientation=False)
            dDisplayItem.setOrientation('horizontal', force=True)
            dDisplayItem.updateStyle()
            dDisplayItem.update()
            
            rightSideContainerWidget.dock_items[name] = dDisplayItem    
            active_raster_plot.ui.epochs_render_configs_widget = an_epochs_display_list_widget
        
            ## Build Connections to signals:
            _connections_list = []
            def _on_update_rendered_intervals(active_2d_plot):
                print(f'_on_update_rendered_intervals(...)')
                _legends_dict = active_2d_plot.build_or_update_all_epoch_interval_rect_legends()
                epoch_display_configs = active_2d_plot.extract_interval_display_config_lists()
                an_epochs_display_list_widget = active_2d_plot.ui.get('epochs_render_configs_widget', None)
                if an_epochs_display_list_widget is None:
                    # create a new one:    
                    an_epochs_display_list_widget:EpochRenderConfigsListWidget = EpochRenderConfigsListWidget(epoch_display_configs)
                    active_2d_plot.ui.epochs_render_configs_widget = an_epochs_display_list_widget
                else:
                    an_epochs_display_list_widget.update_from_configs(configs=epoch_display_configs)
            # END def _on_update_rendered_intervals(active_2d_plot)...


            _a_connection = active_raster_plot.sigRenderedIntervalsListChanged.connect(_on_update_rendered_intervals)
            _connections_list.append(_a_connection)
            

            ## Connect the update signal
            _a_sigAnyConfigChanged_connection = an_epochs_display_list_widget.sigAnyConfigChanged.connect(lambda an_updated_epochs_display_list: active_raster_plot.update_epochs_from_configs_widget())
            _connections_list.append(_a_sigAnyConfigChanged_connection)

        ## END if needs_init...
        # self.ui.rightSideContainerWidget.ui.neuron_widget_container, _connections_list = self._perform_build_attached_neuron_visual_configs_widget(neuron_plotting_configs_dict)

        # Display the sidebar:
        self.set_right_sidebar_visibility(True)


    # @property
    # def dock_tree_sidebar_widget(self) -> Optional[DockAreaDocksTree]:
    #     """The dock_tree_sidebar_widget property."""
    #     ## Get 2D or 3D Raster from spike_raster_window
    #     active_raster_plot = self.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
    #     if active_raster_plot is None:
    #         active_raster_plot = self.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
    #         if active_raster_plot is None:
    #             ## no available raster plots
    #             return None

    #     return active_raster_plot.ui.dockarea_dock_managing_tree_widget
    


    @function_attributes(short_name=None, tags=['widget', 'dock_area_managing_tree', 'interactive', 'right-sidebar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-28 07:22', related_items=['DockAreaDocksTree'])
    def build_dock_area_managing_tree_widget(self):
        """ adds a tree widget containing all added Dock items to the right sidebar and connects its controls
        """
        from pyphoplacecellanalysis.GUI.Qt.Widgets.DockAreaDocksTree.DockAreaDocksTree import DockAreaDocksTree
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DockDisplayColors, CustomDockDisplayConfig

        ## Get 2D or 3D Raster from spike_raster_window
        active_raster_plot = self.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
        if active_raster_plot is None:
            active_raster_plot = self.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
            assert active_raster_plot is not None


        rightSideContainerWidget = self.ui.rightSideContainerWidget # pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterRightSidebarWidget.Spike3DRasterRightSidebarWidget
        right_sidebar_contents_container_dockarea = rightSideContainerWidget.right_sidebar_contents_container_dockarea

        # New Dock-based way _________________________________________________________________________________________________ #
        name: str = 'DockAreaManagingTree'
        widget_name: str = 'dockarea_dock_managing_tree_widget'
        needs_init: bool = True
        
        # Create new widget
        # No extant table widget and display_dock currently, create a new one:
        dDisplayItem = right_sidebar_contents_container_dockarea.find_display_dock(identifier=name) # Dock
        an_extant_widget = active_raster_plot.ui.get(widget_name, None)
        
        if (dDisplayItem is not None) and (an_extant_widget is not None):
            needs_init = False
            a_docks_tree_widget: DockAreaDocksTree = an_extant_widget
            assert a_docks_tree_widget is not None
            print(f'found existing Docks display tree widget for right sidebar. Updating existing instead of creating new.')
            
            

        if needs_init:
            ## INITIALIZE:
            assert dDisplayItem is None
            assert an_extant_widget is None
            epoch_display_configs = active_raster_plot.extract_interval_display_config_lists()
            # active_raster_plot
            a_docks_tree_widget:DockAreaDocksTree = DockAreaDocksTree()
            
            dynamic_docked_widget_container = active_raster_plot.ui.dynamic_docked_widget_container # NestedDockAreaWidget
            dock_tree_list, group_meta_item_dict = dynamic_docked_widget_container.get_dockGroup_dock_tree_dict()
            a_docks_tree_widget.rebuild_dock_tree_items(dock_tree_list=dock_tree_list)
            

            # Add to dynamic dock container
            display_config = CustomDockDisplayConfig(showCloseButton=True, showCollapseButton=True, orientation='horizontal', custom_get_colors_dict={False: DockDisplayColors(fg_color='#111', bg_color='#c5c5c5', border_color='#a7babd'),
                    True: DockDisplayColors(fg_color='#333', bg_color='#757575', border_color='#424242'),
                })
            _, dDisplayItem = right_sidebar_contents_container_dockarea.add_display_dock(name, display_config=display_config, widget=a_docks_tree_widget, dockAddLocationOpts=['bottom'], autoOrientation=False)
            dDisplayItem.setOrientation('horizontal', force=True)
            dDisplayItem.updateStyle()
            dDisplayItem.update()
            
            rightSideContainerWidget.dock_items[name] = dDisplayItem    
            active_raster_plot.ui.dockarea_dock_managing_tree_widget = a_docks_tree_widget
        
            ## Build Connections to signals:
            _connections_list = []
            def _on_update_dock_items(active_2d_plot):
                print(f'_on_update_dock_items(...)')
                dock_tree_list, group_meta_item_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_tree_dict()
                a_dock_tree_widget = active_2d_plot.ui.get(widget_name, None)
                if a_dock_tree_widget is None:
                    # create a new one:    
                    a_dock_tree_widget:DockAreaDocksTree = DockAreaDocksTree()
                    active_2d_plot.ui.dockarea_dock_managing_tree_widget = a_dock_tree_widget
                    a_docks_tree_widget.rebuild_dock_tree_items(dock_tree_list=dock_tree_list)
                else:
                    a_docks_tree_widget.rebuild_dock_tree_items(dock_tree_list=dock_tree_list)
            # END def _on_update_dock_items(active_2d_plot)...


            _a_connection = active_raster_plot.sigEmbeddedMatplotlibDockWidgetAdded.connect(lambda active_2d_plot, dock, widget: _on_update_dock_items(active_2d_plot=active_2d_plot))
            _connections_list.append(_a_connection)
            
            _a_sigDockAdded_connection = active_raster_plot.sigDockAdded.connect(lambda active_2d_plot, a_dock: _on_update_dock_items(active_2d_plot=active_2d_plot))
            _connections_list.append(_a_sigDockAdded_connection)
            
            _a_sigDockModified_connection = active_raster_plot.sigDockModified.connect(lambda active_2d_plot, a_dock, an_action: _on_update_dock_items(active_2d_plot=active_2d_plot)) # (parent, Dock, action)
            _connections_list.append(_a_sigDockModified_connection)
            

            _a_removed_connection = active_raster_plot.sigEmbeddedMatplotlibDockWidgetRemoved.connect(lambda active_2d_plot, removed_identifier: _on_update_dock_items(active_2d_plot=active_2d_plot))
            _connections_list.append(_a_removed_connection)
            
            _a_sigDockClosed_connection = active_raster_plot.sigDockClosed.connect(lambda active_2d_plot, removed_identifier: _on_update_dock_items(active_2d_plot=active_2d_plot))
            _connections_list.append(_a_sigDockClosed_connection)
            
            # ## Connect the update signal
            # _a_sigAnyConfigChanged_connection = a_docks_tree_widget.sigDockConfigChanged.connect(lambda an_updated_epochs_display_list: active_raster_plot.update_epochs_from_configs_widget())
            # _connections_list.append(_a_sigAnyConfigChanged_connection)
            #TODO 2025-04-03 18:43: - [ ] Nothing is done with `_connections_list`. It should be appended somewhere at least
            

        ## END if needs_init...
        # self.ui.rightSideContainerWidget.ui.neuron_widget_container, _connections_list = self._perform_build_attached_neuron_visual_configs_widget(neuron_plotting_configs_dict)

        # Display the sidebar:
        self.set_right_sidebar_visibility(True)
        

# DockAreaDocksTree



    @function_attributes(short_name=None, tags=['menus', 'actions'], input_requires=[], output_provides=[], uses=['PhoMenuHelper.build_all_programmatic_menu_command_dict'], used_by=[], creation_date='2024-12-18 16:29', related_items=[])
    def build_all_menus_actions_dict(self, wants_flat_actions_dict: bool=True, **kwargs) -> Tuple[Dict, Dict[str, QtWidgets.QAction]]:
        """ gets absolutely all of the possible actions (from the menus, both global and context) and returns them 
        
        Usage:
            all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
            all_global_menus_actionsDict
         
        """
        from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper

        return PhoMenuHelper.build_all_programmatic_menu_command_dict(spike_raster_window=self, wants_flat_actions_dict=wants_flat_actions_dict, **kwargs)


    @function_attributes(short_name=None, tags=['event', 'intervals', 'epochs', 'window', 'active-window'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-19 11:31', related_items=[])
    def find_event_intervals_in_active_window(self, included_series_names: Optional[List[str]]=None) -> Dict[str, pd.DataFrame]:
        """find the events/intervals that are within the currently active render window:
        Usage:
            included_series_names=['Replays', 'Laps', 'PBEs']
            curr_intervals_dict: Dict[str, pd.DataFrame] = active_2d_plot.find_event_intervals_in_active_window(included_series_names=included_series_names)
        """
        return self.spike_raster_plt_2d.find_event_intervals_in_active_window(included_series_names=included_series_names, debug_print=False)



    @pyqtExceptionPrintingSlot(object)
    def on_visible_event_intervals_added(self, added_rows):
        print(f'Spike3DRasterWindowWidget.on_visible_event_intervals_added(added_rows: {added_rows})')
        self.bottom_playback_control_bar_logger.add_log_line(f'visible_event_intervals_added(added_rows: {added_rows})')
        
    @pyqtExceptionPrintingSlot(object)
    def on_visible_event_intervals_removed(self, removed_rows):
        print(f'Spike3DRasterWindowWidget.visible_event_intervals_removed(removed_rows: {removed_rows})')
        self.bottom_playback_control_bar_logger.add_log_line(f'visible_event_intervals_removed(removed_rows: {removed_rows})')



    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
        
    def eventFilter(self, watched, event):
        """  has to be installed on an item like:
            self.grid = pg.GraphicsLayoutWidget()
            self.top_left = self.grid.addViewBox(row=1, col=1)
            self.top_left.installEventFilter(self)
        
        """
        # print(f'Spike3DRasterWindowWidget.eventFilter(self, watched, event)')
        delta = None
        if (event.type() == QtCore.QEvent.GraphicsSceneWheel):
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
                
            delta = event.delta()
        
        
        elif (event.type() == QtCore.QEvent.Wheel): # the second case (QtGui.QWheelEvent) doesn't even exist I don't think. IDK why ChatGPT said to use it.
            """ the event is an instance of `QtGui.QWheelEvent`, but the event's .type() is NEVER QtGui.QWheelEvent, that's not even a possible type. """
            if self.should_debug_print_interaction_events:
                print(f'Spike3DRasterWindowWidget.eventFilter(...)\n\t detected event.type() == QtCore.QEvent.Wheel')
                print(f'\twatched: {watched}\n\tevent: {event}')
                print(f'\tevent.angleDelta(): {event.angleDelta()}')
                
            delta = event.angleDelta().x()
            if delta == 0:
                delta = event.angleDelta().y()
            
        else:
            delta = None
            if self.should_debug_print_interaction_events:
                print(f'\t unhandled event {QEventLookupHelpers.get_event_string(event)}')
                
        if (delta is not None) and (abs(delta) > 0):
            ## do the scroll
            if self.should_debug_print_interaction_events:
                print(f'\tperofmring scroll with delta: {delta}')

            numDegrees = delta / 8
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
        # END if (delta is not None) a....
        else:
            # Unknown event type
            if self.should_debug_print_interaction_events:
                print(f'\t unhandled event {QEventLookupHelpers.get_event_string(event)}')

        # If not a particularlly handled case, do the default thing.
        return super().eventFilter(watched, event)



    def wheelEvent(self, event):
        super(Spike3DRasterWindowWidget, self).wheelEvent(event)
        if self.should_debug_print_interaction_events:
            print(f'Spike3DRasterWindowWidget.wheelEvent(...)')
            # self.x = self.x + event.delta()/120
            # print self.x
            # self.label.setText("Total Steps: "+QString.number(self.x))        
            print(f'\t wheelEvent(event: {event}')
    

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
            self.deleteLater() # schedule the object for deletion
            event.accept()
            print('Window closed')
        else:
            event.ignore()
            


    def setWindowTitle(self, desired_window_title: str):
        """ passthrough support for setting the window title
        
        Prevents having to use `spike_raster_window.window().setWindowTitle(desired_window_title)`.
        
        Usage:
            ## Fix window title to display the session context post-hoc
            desired_window_title: str = curr_active_pipeline.get_complete_session_identifier_string() # 'kdiba_gor01_two_2006-6-07_16-40-19__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_5.0'
            spike_raster_window.setWindowTitle(desired_window_title)
            
        """
        # self.window().setWindowTitle(desired_window_title) ## causes kernel crashes
        win = self.window()        
        if win is self:
            super().setWindowTitle(desired_window_title) 
        else:
            win.setWindowTitle(desired_window_title)



# ==================================================================================================================== #
# __name__ == "__main__"                                                                                               #
# ==================================================================================================================== #
if __name__ == "__main__":
    """ 
            assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
        ## Build Pickle Path:
        finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve() # Use Default


        # BEGIN FUNCTION BODY
        global_batch_run = _try_load_global_batch_result()
    
        
        
        & c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/python.exe c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py
        
        & c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/python.exe c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py
        
        # [--multiprocess] [--show-pqi-stack]
        & c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/python.exe -m PyQtInspect --direct --qt-support=pyqt5 --file c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py
    """
    import argparse
    import sys
    from pathlib import Path
    from typing import Optional, List, Dict
    from neuropy.utils.result_context import IdentifyingContext
    from pyphocorehelpers.assertion_helpers import Assert
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData
    from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows
    from pyphoplacecellanalysis.GUI.Qt.Widgets.DebugWidgetStylesheetInspector import ConnectStyleSheetInspector
    from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, BatchResultDataframeAccessor, run_diba_batch, SessionBatchProgress, main
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence

    def _subfn_run_main(pkl_path, debug_print:bool=True, debug_mode: bool=True):
        """ run main function to perform batch processing. """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import _setup_spike_raster_window_for_debugging
        
        print(f'pkl_path: {pkl_path}')
        
        def _try_load_global_batch_result():
            if debug_print:
                print(f'pkl_path: {pkl_path}')
            # try to load an existing batch result:
            try:
                curr_active_pipeline = loadData(pkl_path, debug_print=debug_print)
                
            except NotImplementedError:
                # Fixes issue with pickled POSIX_PATH on windows for path.                    
                with set_posix_windows():
                    curr_active_pipeline = loadData(pkl_path, debug_print=debug_print) # Fails this time if it still throws an error

            except (FileNotFoundError, TypeError):
                # loading failed
                print(f'Failure loading {pkl_path}.')
                curr_active_pipeline = None
                
            return curr_active_pipeline
        
        curr_active_pipeline = _try_load_global_batch_result()
        assert curr_active_pipeline is not None
        print(f'loaded `curr_active_pipeline`, building Spike3DRasterWindowWidget...')
        
        ## Loads `curr_active_pipeline` from the provided path
        if debug_mode:
            Spike3DRasterWindowWidget.enable_interaction_events_debug_print = True    
        # Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
        spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline, force_create_new=True)
        print(f'built `Spike3DRasterWindowWidget`, launching...')
        if debug_mode:
            spike_raster_window.enable_debug_print = True
            Spike3DRasterWindowWidget.enable_interaction_events_debug_print = True
            print(f'\tspike_raster_window.should_debug_print_interaction_events: {spike_raster_window.should_debug_print_interaction_events}')
    

        # included_session_contexts: Optional[List[IdentifyingContext]] = [IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54')]


        # global_batch_run, result_handler, across_sessions_instantaneous_fr_dict, output_filenames_tuple = main(active_result_suffix=active_result_suffix, 
        #                                                                                                     included_session_contexts=included_session_contexts,
        #                                                                                                     num_processes=num_processes, 
        #                                                                                                     should_force_reload_all=should_force_reload_all, 
        #                                                                                                     should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, 
        #                                                                                                     debug_print=debug_print)
        
        all_global_menus_actionsDict, global_flat_action_dict, _all_outputs_dict = _setup_spike_raster_window_for_debugging(spike_raster_window, debug_print=debug_print)

        return spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget), curr_active_pipeline
        

    """ Usage:
    
    python scripts/runSingleBatch.py --active_result_suffix "2023-08-08_bApogee" --num_processes 4 --should_force_reload_all --debug_print
    python scripts/runSingleBatch.py --active_result_suffix "2023-08-08_LNX00052" --num_processes 4 --should_force_reload_all --debug_print
    
    --should_perform_figure_generation_to_file
    --should_perform_figure_generation_to_file
    
    """
    # parser = argparse.ArgumentParser(description='Perform batch processing.')
    # parser.add_argument('--active_result_suffix', required=True, help='Suffix used for filenames throughout the notebook.')
    # parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use.')
    # parser.add_argument('--should_force_reload_all', action='store_true', help='Force reload all data.')
    # parser.add_argument('--should_perform_figure_generation_to_file', action='store_true', help='Perform figure generation to file.')
    # parser.add_argument('--debug_print', action='store_true', help='Enable debug printing.')

    # args = parser.parse_args()
    
    pkl_path = Path('W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/loadedSessPickle.pkl')
    # pkl_path = Path('W:/Data/KDIBA/vvp01/one/2006-4-09_17-29-30/loadedSessPickle.pkl')
    # pkl_path = Path('W:/Data/KDIBA/vvp01/one/2006-4-09_17-29-30/loadedSessPickle.pkl')
    # pkl_path = Path('W:/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/loadedSessPickle.pkl')
    
    Assert.path_exists(pkl_path)

    app = QtWidgets.QApplication(sys.argv)
    # Define stylesheet
    app.setStyleSheet("""
        VContainer {
            background-color: #00e1ff; /* Light gray background */
            color: #ff0095;           /* Text color */
            border: 1px solid #4400ff; /* Border around the container */
        }
        VContainer QLabel {
            color: #ffd000; /* Red text inside VContainer's labels */
        }
        QSplitter::handle { height: 2px; width: 2px; }
    """)

    # testWidget = Spike3DRasterWindowWidget()



    spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget), curr_active_pipeline = _subfn_run_main(pkl_path)
    spike_raster_window.show()

    # testWidget.show()
    
    print(f'waiting...')
    # ConnectStyleSheetInspector(main_window=spike_raster_window, shortcut=QKeySequence(Qt.CTRL + Qt.SHIFT + Qt.Key_F12)) # Connects the global stylesheet inspector to the window


    sys.exit(app.exec_())

