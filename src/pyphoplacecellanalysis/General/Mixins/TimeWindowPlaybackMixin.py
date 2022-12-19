""" TimeWindowPlaybackMixin

Extends a TimeWindow to support realtime playback capabilities.

"""
import time
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets


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
            
            
            
            