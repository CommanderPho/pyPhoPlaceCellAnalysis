""" TimeWindowPlaybackMixin

Extends a TimeWindow to support realtime playback capabilities.

"""
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets



class TimeWindowPlaybackPropertiesMixin:
    
    @property
    def animation_active_time_window(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate.
        IMPLEMENTORS MUST OVERRIDE.
        """
        raise NotImplementedError
    
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
        return self._playback_update_frequency

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

    @QtCore.pyqtSlot(int)
    def shift_animation_frame_val(self, shift_frames: int):
        next_start_timestamp = self.animation_active_time_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames))
        self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        
    # Called from SliderRunner's thread when it emits the update_signal:
    @QtCore.pyqtSlot()
    def increase_animation_frame_val(self):
        self.shift_animation_frame_val(1)
        

class TimeWindowPlaybackController(QtCore.QObject):
    """docstring for TimeWindowPlaybackController.
    
    """
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
        
    # def __init__(self):
    #     super(TimeWindowPlaybackController, self).__init__()
        
    def __init__(self):
        QtCore.QObject.__init__(self)
        

    
    def setup(self, root_TimeWindowOwner):
        """ sets up the passed root_TimeWindow object's required parameters and state variables. """
        root_TimeWindowOwner._playback_update_frequency = TimeWindowPlaybackController.PlaybackUpdateFrequency
        root_TimeWindowOwner.speedBurstPlaybackRate = TimeWindowPlaybackController.SpeedBurstPlaybackRate
        root_TimeWindowOwner.params.is_speed_burst_mode_active = False
        root_TimeWindowOwner.params.is_playback_reversed = False
        root_TimeWindowOwner.params.animation_time_step = 0.04
        
        # Slider update thread:        
        root_TimeWindowOwner.animationThread = UpdateRunner(update_frequency=root_TimeWindowOwner.playback_update_frequency)
        # self.sliderThread.update_signal.connect(self.increase_slider_val)
        root_TimeWindowOwner.animationThread.update_signal.connect(root_TimeWindowOwner.increase_animation_frame_val)


        


class UpdateRunner(QtCore.QThread):
    update_signal = QtCore.pyqtSignal()

    def __init__(self, update_frequency=0.04):
        self.update_frequency = update_frequency
        QtCore.QThread.__init__(self)

    def run(self):
        while(True):
            self.update_signal.emit()
            # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            time.sleep(self.update_frequency) # 40.0 ms = 25 FPS
            # time.sleep(.32) # 320ms
            # time.sleep(0.05) # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            
            
            
            