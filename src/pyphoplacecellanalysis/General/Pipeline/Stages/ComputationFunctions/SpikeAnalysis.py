import numpy as np
import pandas as pd
import itertools

from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from neurodsp.plts.time_series import plot_time_series, plot_bursts # for plotting results


from pyphoplacecellanalysis.General.Mixins.AllFunctionEnumeratingMixin import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.ComputationResults import ComputationResult
from pyphoplacecellanalysis.General.Decoder.decoder_result import build_position_df_resampled_to_time_windows


class SpikeAnalysisComputations(AllFunctionEnumeratingMixin):
    
    def _perform_spike_burst_detection_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes periods when the cells are firing in bursts """
        
        # Get sampling rate:
        sampling_rate = computation_result.sess.recinfo.dat_sampling_rate # 32552
        # sampling_rate = computation_result.sess.recinfo.eeg_sampling_rate # 1252
        
        
        
        is_burst = detect_bursts_dual_threshold(sig, fs=sampling_rate, dual_thresh=(1, 2), f_range=(8, 12))
        
        
        
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
        computation_result.computed_data['burst_detection'] = {
         'is_burst': is_burst
        }
        """ 
        Access via ['burst_detection']['is_burst']
        Example:
            active_burst_info = curr_active_pipeline.computation_results['maze1'].computed_data['burst_detection']
            active_burst_info
        """
        return computation_result
    
    
    
    
    