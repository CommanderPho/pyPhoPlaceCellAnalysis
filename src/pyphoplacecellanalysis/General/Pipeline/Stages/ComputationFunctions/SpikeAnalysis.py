import numpy as np
import pandas as pd
from indexed import IndexedOrderedDict
import itertools

# from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
# from neurodsp.plts.time_series import plot_time_series, plot_bursts # for plotting results

# PyBursts:
from pybursts import pybursts

# For Progress bars:
from tqdm.notebook import tqdm


from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


class SpikeAnalysisComputations(AllFunctionEnumeratingMixin):
    
    _computationGroupName = 'burst_detection'
    
    def _perform_spike_burst_detection_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes periods when the cells are firing in bursts
        
        Model Goal:
            "Thus, the combined goal is to track the sequence of gaps as well as possible without changing state too much."
        
        Model Parameters:            
            s: the base of the exponential distribution that is used for modeling the event frequencies
                scaling parameter `s`: controls the "resolution" with which the discrete rate values of the states are able to track the real-valued gaps; 
            gamma: coefficient for the transition costs between states
                parameter `gamma`: (default 1.0) controls the ease with which the automaton can change states

        """
        def _compute_pybursts_burst_interval_detection(sess, max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicParameters(s=2, gamma=0.1), use_progress_bar=False, debug_print=False):
            out_pyburst_intervals = IndexedOrderedDict()

            # Build the progress bar:
            if use_progress_bar:
                p_bar = tqdm(sess.spikes_df.spikes.neuron_ids) # Progress bar version
            else:
                p_bar = sess.spikes_df.spikes.neuron_ids # Non-progress bar version, just wrap the iterable

            for (i, a_cell_id) in enumerate(p_bar):
                # loop through the cell_ids  
                if i == 0 or True:
                    if debug_print:
                        print(f'computing burst intervals for {a_cell_id}...')
                    curr_df = sess.spikes_df.groupby('aclu').get_group(a_cell_id)
                    curr_spike_train = curr_df[curr_df.spikes.time_variable_name].to_numpy()
                    num_curr_spikes = len(curr_spike_train)
                    if debug_print:
                        print(f'\tnum_curr_spikes: {num_curr_spikes}')

                    # For testing, limit to the first 1000 spikes:
                    num_curr_spikes = min(num_curr_spikes, max_num_spikes_per_neuron)
                    if debug_print:
                        print(f'\ttruncating to 1000...: {num_curr_spikes}')
                    curr_spike_train = curr_spike_train[:num_curr_spikes]

                    # Update the progress bar:
                    if use_progress_bar:
                        p_bar.set_description(f'computing burst intervals for "{a_cell_id}" | \tnum_curr_spikes: {num_curr_spikes} | \ttruncating to {max_num_spikes_per_neuron}...: {num_curr_spikes}:')

                    # Perform the computation:
                    curr_pyburst_intervals = pybursts.kleinberg(curr_spike_train, s=kleinberg_parameters.s, gamma=kleinberg_parameters.gamma)
                    # Convert ouptut intervals to dataframe
                    curr_pyburst_interval_df = pd.DataFrame(curr_pyburst_intervals, columns=['burst_level', 't_start', 't_end'])
                    curr_pyburst_interval_df['t_duration'] = curr_pyburst_interval_df.t_end - curr_pyburst_interval_df.t_start

                    # Convert vectors to tuples of (t_start, t_duration) pairs:
                    curr_pyburst_interval_df['interval_pair'] = list(zip(curr_pyburst_interval_df.t_start, curr_pyburst_interval_df.t_duration)) # pairs like # [(33, 4), (76, 16), (76, 1)]
                    # print(f'interval_pairs: {interval_pairs}') 
                    out_pyburst_intervals[a_cell_id] = curr_pyburst_interval_df
            
            return out_pyburst_intervals
            
        """
        
        # To Access results from previous computation stages:
        computation_result.computed_data['pf2D_Decoder'].neuron_IDs
        
        
        # Important/Common Properties accessible by default:
        
            computation_result.sess # The filtered session object for this series of computations
            computation_result.computation_config  # The computation parameters for this series of computations
            computation_result.sess.position.to_dataframe()
        
        
        """
        
        # computation_config = computation_result.computation_config.spike_analysis
        # {
        #     'max_num_spikes_per_neuron': 20000,
        #     'kleinberg_parameters': DynamicParameters(s=2, gamma=0.1),
        #     'use_progress_bar': False,
        #     'debug_print': False
        # }
        # max_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to
        # kleinberg_parameters = DynamicParameters(s=2, gamma=0.1)
        # use_progress_bar = False # whether to use a tqdm progress bar
        # debug_print = False # whether to print debug-level progress using traditional print(...) statements


        default_spike_analysis_config = DynamicParameters(max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicParameters(s=2, gamma=0.1), use_progress_bar=False, debug_print=False)
        
        active_spike_analysis_config = computation_result.computation_config.get('spike_analysis', default_spike_analysis_config)
        active_spike_analysis_config = (default_spike_analysis_config | active_spike_analysis_config) # augment the actual values of the analysis config with the defaults if they're unavailable. This allows the user to pass only partially complete parameters in .spike_analysis
        
        ## Set the config values from the defaults to ensure we have access to them later:
        computation_result.computation_config['spike_analysis'] = active_spike_analysis_config
        
        # print(f'computation_result.computation_config: {computation_result.computation_config.}')
        
        # # Get sampling rate:
        # sampling_rate = computation_result.sess.recinfo.dat_sampling_rate # 32552
        # # sampling_rate = computation_result.sess.recinfo.eeg_sampling_rate # 1252
        # is_burst = detect_bursts_dual_threshold(sig, fs=sampling_rate, dual_thresh=(1, 2), f_range=(8, 12))
        # time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.time_bin_size) # TimedeltaIndexResampler
        # time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
        # computation_result.computed_data['burst_detection'] = {
        #  'is_burst': is_burst
        # }
        
                
        out_pyburst_intervals = _compute_pybursts_burst_interval_detection(computation_result.sess, **active_spike_analysis_config)        
        computation_result.computed_data[SpikeAnalysisComputations._computationGroupName] = {'burst_intervals': out_pyburst_intervals}
            
        """ 
        Access via ['burst_detection']['burst_intervals']
        Example:
            active_burst_info = curr_active_pipeline.computation_results['maze1'].computed_data['burst_detection']
            active_burst_info
        """
        return computation_result
    
    
    
    
    