""" TimeWindowPlaybackMixin

Extends a TimeWindow to support realtime playback capabilities.

"""
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets



# class SimplePlayPauseWithExternalAppMixin:
    
#     # @property
#     # def animationThread(self):
#     #     """The animationThread property."""
#     #     return self.playback_controller
    
#     @property
#     def animationThread(self):
#         """The animationThread property."""
#         return self.playback_controller.animationThread
    


class TimeWindowPlaybackPropertiesMixin:
    """ TimeWindowPlaybackPropertiesMixin: Properties that implementors that want to control time-window playback should implement.
    
    Implementors must override:
        animation_active_time_window
        
    Required Properties:
        self.params
    """
    
    @property
    def animation_active_time_window(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate.
        IMPLEMENTORS MUST OVERRIDE.
        """
        raise NotImplementedError
    
#     @property
#     def animation_time_step(self):
#         """ How much to step forward in time at each frame of animation. """
#         return self.params.animation_time_step
#     @animation_time_step.setter
#     def animation_time_step(self, value):
#         self.params.animation_time_step = value
        
        
#     ## STATE PROPERTIES
#     @property
#     def is_playback_reversed(self):
#         """The is_playback_reversed property."""
#         return self.params.is_playback_reversed
#     @is_playback_reversed.setter
#     def is_playback_reversed(self, value):
#         self.params.is_playback_reversed = value
        
#     @property
#     def animation_playback_direction_multiplier(self):
#         """The animation_reverse_multiplier property."""
#         if self.params.is_playback_reversed:
#             return -1.0
#         else:
#             return 1.0

#     @property
#     def playback_update_frequency(self):
#         """The rate at which the separate animation thread attempts to update the interface. ReadOnly."""
#         return self.params.playback_update_frequency

#     @property
#     def playback_rate_multiplier(self):
#         """ 1x playback (real-time) occurs when self.playback_update_frequency == self.animation_time_step. 
#             if self.animation_time_step = 2.0 * self.playback_update_frequency => for each update the window will step double the time_step forward in time than it would be default, meaning a 2.0x playback_rate_multiplier.
#         """
#         return (self.animation_time_step / self.playback_update_frequency)
#     @playback_rate_multiplier.setter
#     def playback_rate_multiplier(self, value):
#         """ since self.playback_update_frequency is fixed, only self.animation_time_step can be adjusted to set the playback_rate_multiplier. """
#         desired_playback_rate_multiplier = value
#         self.animation_time_step = self.playback_update_frequency * desired_playback_rate_multiplier

#     @QtCore.pyqtSlot(int)
#     def shift_animation_frame_val(self, shift_frames: int):
#         next_start_timestamp = self.animation_active_time_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames))
#         self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        
#     # Called from SliderRunner's thread when it emits the update_signal:
#     @QtCore.pyqtSlot()
#     def increase_animation_frame_val(self):
#         self.shift_animation_frame_val(1)
        



class TimeWindowPlaybackController(QtCore.QObject):
    """
        TimeWindowPlaybackController
        
    Usage:
    
        # Setup the animation playback object for the time window:
        self.playback_controller = TimeWindowPlaybackController()
        # self.playback_controller.setup(self._spikes_window)
        self.playback_controller.setup(self) # pass self to have properties set
        
        
    Known Uses:
        SpikeRasterBase
        
    """
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
        
    def __init__(self):
        QtCore.QObject.__init__(self)
        

    def setup(self, root_TimeWindowOwner):
        """ sets up the passed root_TimeWindow object's required parameters and state variables. """
        root_TimeWindowOwner.params.playback_update_frequency = TimeWindowPlaybackController.PlaybackUpdateFrequency
        root_TimeWindowOwner.params.speedBurstPlaybackRate = TimeWindowPlaybackController.SpeedBurstPlaybackRate
        root_TimeWindowOwner.params.is_speed_burst_mode_active = False
        root_TimeWindowOwner.params.is_playback_reversed = False
        root_TimeWindowOwner.params.animation_time_step = 0.04
        
        # Slider update thread:        
        root_TimeWindowOwner.animationThread = UpdateRunner(update_frequency=root_TimeWindowOwner.playback_update_frequency)
        # self.sliderThread.update_signal.connect(self.increase_slider_val)
        root_TimeWindowOwner.animationThread.update_signal.connect(root_TimeWindowOwner.increase_animation_frame_val)


        
class TimeWindowPlaybackControllerActionsMixin:
    """ Defines the callback functions to respond to UI events such as play/pause button presses, skip/jumps, etc.
    
    Required Properties:
        .animationThread
    
    Required Functions:
        @QtCore.pyqtSlot(int)
        shift_animation_frame_val(self, shift_frames: int)
        
        @QtCore.pyqtSlot()
        increase_animation_frame_val(self)
        
    """
    pass

#     ## Update Functions:
#     @QtCore.pyqtSlot(bool)
#     def play_pause(self, is_playing):
#         print(f'TimeWindowPlaybackControllerActionsMixin.play_pause(is_playing: {is_playing})')
#         if (not is_playing):
#             self.animationThread.start()
#         else:
#             self.animationThread.terminate()

#     @QtCore.pyqtSlot()
#     def on_jump_left(self):
#         # Skip back some frames
#         print(f'TimeWindowPlaybackControllerActionsMixin.on_jump_left()')
#         self.shift_animation_frame_val(-5)
        
#     @QtCore.pyqtSlot()
#     def on_jump_right(self):
#         # Skip forward some frames
#         print(f'TimeWindowPlaybackControllerActionsMixin.on_jump_right()')
#         self.shift_animation_frame_val(5)
        

#     @QtCore.pyqtSlot(bool)
#     def on_reverse_held(self, is_reversed):
#         print(f'TimeWindowPlaybackControllerActionsMixin.on_reverse_held(is_reversed: {is_reversed})')
#         pass
    

class UpdateRunner(QtCore.QThread):
    """ 
        A QThread that calls update_signal at a fixed interval determined by update_frequency.
    """
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
            
            
            
            