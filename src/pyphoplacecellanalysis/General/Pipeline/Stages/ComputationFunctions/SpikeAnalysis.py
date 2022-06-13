import numpy as np
import pandas as pd
from indexed import IndexedOrderedDict
import itertools

# from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
# from neurodsp.plts.time_series import plot_time_series, plot_bursts # for plotting results

# PyBursts:
from pybursts import pybursts

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


class SpikeAnalysisComputations(AllFunctionEnumeratingMixin):
    
    def _perform_spike_burst_detection_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes periods when the cells are firing in bursts """
        
        
        """
        
        # To Access results from previous computation stages:
        computation_result.computed_data['pf2D_Decoder'].neuron_IDs
        
        
        # Important/Common Properties accessible by default:
        
            computation_result.sess # The filtered session object for this series of computations
            computation_result.computation_config  # The computation parameters for this series of computations
            computation_result
        
        
        """
        
        # # Get sampling rate:
        # sampling_rate = computation_result.sess.recinfo.dat_sampling_rate # 32552
        # # sampling_rate = computation_result.sess.recinfo.eeg_sampling_rate # 1252
        
        
        
        # is_burst = detect_bursts_dual_threshold(sig, fs=sampling_rate, dual_thresh=(1, 2), f_range=(8, 12))
        
        
        
        # time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.time_bin_size) # TimedeltaIndexResampler
        # time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
        # computation_result.computed_data['burst_detection'] = {
        #  'is_burst': is_burst
        # }
        
        
        max_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to
        kleinberg_parameters = DynamicParameters(s=2, gamma=0.1)
        
        out_pyburst_intervals = IndexedOrderedDict()
        
        # Build the progress bar:
        p_bar = tqdm(computation_result.sess.spikes_df.spikes.neuron_ids) # Progress bar version
        
        for (i, a_cell_id) in enumerate(p_bar):
            # loop through the cell_ids  
            if i == 0 or True:
                # print(f'computing burst intervals for {a_cell_id}...')
                curr_df = computation_result.sess.spikes_df.groupby('aclu').get_group(a_cell_id)
                curr_spike_train = curr_df[curr_df.spikes.time_variable_name].to_numpy()
                num_curr_spikes = len(curr_spike_train)
                # print(f'\tnum_curr_spikes: {num_curr_spikes}')
        
                # For testing, limit to the first 1000 spikes:
                num_curr_spikes = min(num_curr_spikes, max_num_spikes_per_neuron)
                # print(f'\ttruncating to 1000...: {num_curr_spikes}')
                curr_spike_train = curr_spike_train[:num_curr_spikes]
        
                # Update the progress bar:
                p_bar.set_description(f'computing burst intervals for "{a_cell_id}" | \tnum_curr_spikes: {num_curr_spikes} | \ttruncating to {max_num_spikes_per_neuron}...: {num_curr_spikes}:')
        
                # Perform the computation:
                curr_pyburst_intervals = pybursts.kleinberg(curr_spike_train, s=kleinberg_parameters.s, gamma=kleinberg_parameters.gamma)
                
                curr_pyburst_interval_df = pd.DataFrame(curr_pyburst_intervals, columns=['burst_level', 't_start', 't_end'])
                curr_pyburst_interval_df['t_duration'] = curr_pyburst_interval_df.t_end - curr_pyburst_interval_df.t_start
                
                # Convert vectors to tuples of (t_start, t_duration) pairs:
                curr_pyburst_interval_df['interval_pair'] = list(zip(curr_pyburst_interval_df.t_start, curr_pyburst_interval_df.t_duration)) # pairs like # [(33, 4), (76, 16), (76, 1)]
                # print(f'interval_pairs: {interval_pairs}') 
                out_pyburst_intervals[a_cell_id] = curr_pyburst_interval_df
                
            
            
        """ 
        Access via ['burst_detection']['is_burst']
        Example:
            active_burst_info = curr_active_pipeline.computation_results['maze1'].computed_data['burst_detection']
            active_burst_info
        """
        return computation_result
    
    
    
    
    