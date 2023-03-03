import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
from dataclasses import dataclass


@dataclass
class VisualizationWindow(object):
    """

    Used in:
        InteractivePlaceCellDataExplorer: to hold the fixed-duration time windows for each path
    
    Usage:
        ## Simplified with just two windows:
        self.params.longer_spikes_window = VisualizationWindow(duration_seconds=1024.0, sampling_rate=self.active_session.position.sampling_rate) # have it start clearing spikes more than 30 seconds old
        self.params.curr_view_window_length_samples = self.params.longer_spikes_window.duration_num_frames # number of samples the window should last
        print('longer_spikes_window - curr_view_window_length_samples - {}'.format(self.params.curr_view_window_length_samples))

        self.params.recent_spikes_window = VisualizationWindow(duration_seconds=10.0, sampling_rate=self.active_session.position.sampling_rate) # increasing this increases the length of the position tail
        self.params.curr_view_window_length_samples = self.params.recent_spikes_window.duration_num_frames # number of samples the window should last
        print('recent_spikes_window - curr_view_window_length_samples - {}'.format(self.params.curr_view_window_length_samples))

        ## Build the sliding windows:

        # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
        self.params.active_epoch_position_linear_indicies = np.arange(np.size(self.active_session.position.time))
        self.params.pre_computed_window_sample_indicies = self.params.recent_spikes_window.build_sliding_windows(self.params.active_epoch_position_linear_indicies)
        # print('pre_computed_window_sample_indicies: {}\n shape: {}'.format(pre_computed_window_sample_indicies, np.shape(pre_computed_window_sample_indicies)))


    """
    duration_seconds: float = None
    sampling_rate: float = None
    duration_num_frames: int = None

    def __init__(self, duration_seconds=None, sampling_rate=None, duration_num_frames=None):
        self.duration_seconds = duration_seconds # Update every frame
        if (sampling_rate is not None):
            self.sampling_rate = sampling_rate # number of updates per second (Hz)
        else:
            print('Sampling rate is none!')
            self.sampling_rate = None
            
        if (duration_num_frames is not None):
            self.duration_num_frames = duration_num_frames
        else:
            self.duration_num_frames = VisualizationWindow.compute_window_samples(self.duration_seconds, self.sampling_rate)

    def build_sliding_windows(self, times):
        return VisualizationWindow.compute_sliding_windows(times, self.duration_num_frames)


    @staticmethod
    def compute_window_samples(window_duration_seconds, sampling_rate):
        return int(np.floor(window_duration_seconds * sampling_rate))
        
    @staticmethod
    def compute_sliding_windows(times, num_window_frames):
        # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(times, num_window_frames)

    
