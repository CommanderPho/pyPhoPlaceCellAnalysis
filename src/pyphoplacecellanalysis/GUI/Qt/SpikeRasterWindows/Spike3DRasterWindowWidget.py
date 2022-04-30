import numpy as np

from qtpy import QtCore, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.Qt.SyncedTimelineWindowLink import connect_additional_controlled_plotter, connect_controlled_time_synchornized_plotter

from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowBase import Ui_RootWidget # Generated file from .ui

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo

from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController, TimeWindowPlaybackControllerActionsMixin

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import SpikeRasterBottomFrameControlsMixin
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarWidget import Spike3DRasterLeftSidebarControlBar, SpikeRasterLeftSidebarControlsMixin

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin

# remove TimeWindowPlaybackControllerActionsMixin
# class Spike3DRasterWindowWidget(SpikeRasterBottomFrameControlsMixin, TimeWindowPlaybackControllerActionsMixin, TimeWindowPlaybackPropertiesMixin, QtWidgets.QWidget):
class Spike3DRasterWindowWidget(SpikeRasterLeftSidebarControlsMixin, SpikeRasterBottomFrameControlsMixin, SpikesWindowOwningMixin, QtWidgets.QWidget):
    """ A main raster window loaded from a Qt .ui file. 
    
    Usage:
    
    from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

    spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df)
    
    """
    
    enable_window_close_confirmation = False
    # Application/Window Configuration Options:
    applicationName = 'Spike3DRasterWindow'
    windowName = 'Spike3DRasterWindow'
    
    enable_debug_print = False
    
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
        
        # self.ui.splitter.setSizes([900, 200])
        # self.ui.splitter.setStretchFactor(0, 5) # have the top widget by 3x the height as the bottom widget
        # self.ui.splitter.setStretchFactor(1, 1) # have the top widget by 3x the height as the bottom widget        
        
        
        self.params = VisualizationParameters(self.applicationName)
        self.params.type_of_3d_plotter = type_of_3d_plotter
        
        # Helper Mixins: INIT:
        self.SpikeRasterBottomFrameControlsMixin_on_init()
        self.SpikeRasterLeftSidebarControlsMixin_on_init()
        
        # Helper Mixins: SETUP:
        self.SpikeRasterBottomFrameControlsMixin_on_setup()
        self.SpikeRasterLeftSidebarControlsMixin_on_setup()
        
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
        
        self.ui.v_layout_secondary = QtWidgets.QVBoxLayout()
        self.ui.v_layout_secondary.setContentsMargins(0,0,0,0)
        self.ui.v_layout_secondary.addWidget(self.ui.spike_raster_plt_2d)
        self.ui.secondarySpikeRasterControlWidget.setLayout(self.ui.v_layout_secondary)
        
        
        if self.ui.spike_raster_plt_3d is not None:
            self.connect_plotter_time_windows()
        
        self.ui.additional_connections = {}
        
        # self.spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
        # self.spike_3d_to_2d_window_connection = self.spike_raster_plt_2d.window_scrolled.connect(self.spike_raster_plt_3d.spikes_window.update_window_start_end)
        # self.spike_raster_plt_3d.disable_render_window_controls()
        # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
        # self.spike_raster_plt_3d.setWindowTitle('Main 3D Raster Window')
        
        ## Create the animation properties:
        self.playback_controller = TimeWindowPlaybackController()
        self.playback_controller.setup(self) # pass self to have properties set
        
        
        ## Connect the UI Controls:
        # Helper Mixins: buildUI:
        self.SpikeRasterLeftSidebarControlsMixin_on_buildUI() # Call this to set the initial values for the UI before signals are connected.
        # self.ui.bottom_controls_frame, self.ui.bottom_controls_layout = self.SpikeRasterBottomFrameControlsMixin_on_buildUI() # NOTE: do not call for the window as it already has a valid bottom bar widget
        # Connect the signals:
        self.ui.bottom_bar_connections = None 
        self.ui.bottom_bar_connections = self.SpikeRasterBottomFrameControlsMixin_connectSignals(self.ui.bottomPlaybackControlBarWidget)
        
        self.ui.left_side_bar_connections = None
        self.ui.left_side_bar_connections = self.SpikeRasterLeftSidebarControlsMixin_connectSignals(self.ui.leftSideToolbarWidget)
        
        

    def connect_plotter_time_windows(self):
        """ connects the controlled plotter (usually the 3D plotter) to the 2D plotter that controls it. """
        # self.spike_3d_to_2d_window_connection = self.spike_raster_plt_2d.window_scrolled.connect(self.spike_raster_plt_3d.spikes_window.update_window_start_end)        
        # Rate limited version:
        self.spike_3d_to_2d_window_connection = pg.SignalProxy(self.spike_raster_plt_2d.window_scrolled, delay=0.2, rateLimit=60, slot=self.spike_raster_plt_3d.spikes_window.update_window_start_end_rate_limited) # Limit updates to 60 Signals/Second
        
         
    def connect_additional_controlled_plotter(self, controlled_plt):
        """ try to connect the controlled_plt to the current controller (usually the 2D plot). """
        extant_connection = self.ui.additional_connections.get(controlled_plt, None)
        if extant_connection is None:
            new_connection_obj = connect_additional_controlled_plotter(self.spike_raster_plt_2d, controlled_plt=controlled_plt)
            self.ui.additional_connections[controlled_plt] = new_connection_obj # add the connection object to the self.ui.additional_connections array
            return self.ui.additional_connections[controlled_plt]
        else:
            print(f'connection already existed!')
            return extant_connection
        
        
        
    def connect_controlled_time_synchronized_plotter(self, controlled_plt):
        """ try to connect the controlled_plt to the current controller (usually the 2D plot). """
        extant_connection = self.ui.additional_connections.get(controlled_plt, None)
        if extant_connection is None:
            new_connection_obj = connect_controlled_time_synchornized_plotter(self.spike_raster_plt_2d, controlled_plt=controlled_plt)
            self.ui.additional_connections[controlled_plt] = new_connection_obj # add the connection object to the self.ui.additional_connections array
            return self.ui.additional_connections[controlled_plt]
        else:
            print(f'connect_controlled_time_synchronized_plotter(...): connection already existed!')
            return extant_connection
        

          
          
      
    def __str__(self):
         return
     
     
    
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
            self.onClose() # ensure onClose() is called
            event.accept()
            print('Window closed')
        else:
            event.ignore()
   
    ###################################
    #### EVENT HANDLERS
    ##################################
    @QtCore.Slot(int)
    def shift_animation_frame_val(self, shift_frames: int):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.shift_animation_frame_val(shift_frames: {shift_frames})')
        # if self.spike_raster_plt_2d is not None:
        #     self.spike_raster_plt_2d.shift_an
        next_start_timestamp = self.animation_active_time_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames))
        # self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        # Update the windows once before showing the UI:
        self.spike_raster_plt_2d.update_scroll_window_region(next_start_timestamp, next_start_timestamp+self.animation_active_time_window.window_duration, block_signals=True) # self.spike_raster_plt_2d.window_scrolled should be emitted
        
        # signal emit:
        # self.spike_raster_plt_2d.window_scrolled.emit(next_start_timestamp, next_start_timestamp+self.animation_active_time_window.window_duration)

        # update_scroll_window_region
        # self.ui.spike_raster_plt_3d.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        
        

    # Called from SliderRunner's thread when it emits the update_signal:        
    @QtCore.Slot()
    def increase_animation_frame_val(self):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.increase_animation_frame_val()')
        self.shift_animation_frame_val(1)
        
        
    ## Update Functions:
    @QtCore.Slot(bool)
    def play_pause(self, is_playing):
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.play_pause(is_playing: {is_playing})')
        if (not is_playing):
            self.animationThread.start()
        else:
            self.animationThread.terminate()
            

    @QtCore.Slot()
    def on_jump_left(self):
        # Skip back some frames
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_left()')
        self.shift_animation_frame_val(-5)
        
    @QtCore.Slot()
    def on_jump_right(self):
        # Skip forward some frames
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_jump_right()')
        self.shift_animation_frame_val(5)
        

    @QtCore.Slot(bool)
    def on_reverse_held(self, is_reversed):
        print(f'Spike3DRasterWindowWidget.on_reverse_held(is_reversed: {is_reversed})')
        pass
    
    ########################################################
    ## For SpikeRasterLeftSidebarControlsMixin conformance:
    ########################################################
    @QtCore.Slot(float)
    def on_animation_timestep_valueChanged(self, updated_val):
        old_value = self.animation_time_step
        self.animation_time_step = updated_val
        
    @QtCore.Slot(float)
    def on_temporal_zoom_factor_valueChanged(self, updated_val):
        old_value = self.temporal_zoom_factor        
        self.temporal_zoom_factor = updated_val
                
    @QtCore.Slot(float)
    def on_render_window_duration_valueChanged(self, updated_val):
        old_value = self.render_window_duration
        self.render_window_duration = updated_val
        

        
        
        
        
    ########################################################
    ## For Other conformances:
    ########################################################
    @QtCore.Slot()
    def on_spikes_df_changed(self):
        """ changes:
            self.unit_ids
            self.n_full_cell_grid
        """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_spikes_df_changed()')
        
    @QtCore.Slot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')


    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
        
        
    
    @QtCore.Slot(float, float, float, object)
    def on_windowed_data_window_duration_changed(self, start_t, end_t, duration, updated_data_value):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration}, updated_data_value: ...)')

    @QtCore.Slot(float, float, object)
    def on_windowed_data_window_changed(self, start_t, end_t, updated_data_value):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRasterWindowWidget.on_windowed_data_window_changed(start_t: {start_t}, end_t: {end_t}, updated_data_value: ...)')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    





     
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterWindowWidget()
    # testWidget.show()
    sys.exit(app.exec_())

