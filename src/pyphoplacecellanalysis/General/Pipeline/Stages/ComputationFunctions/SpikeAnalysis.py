from attrs import define, field, asdict
import numpy as np
import pandas as pd
from indexed import IndexedOrderedDict
import itertools

# from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
# from neurodsp.plts.time_series import plot_time_series, plot_bursts # for plotting results

# PyBursts:
from pybursts import pybursts

# 2022-11-08 Firing Rate Calculations
from elephant.statistics import mean_firing_rate, instantaneous_rate, time_histogram
from quantities import ms, s, Hz
from neo.core.spiketrain import SpikeTrain
from neo.core.analogsignal import AnalogSignal
from elephant.kernels import GaussianKernel

# For Progress bars:
from tqdm.notebook import tqdm

from neuropy.utils.misc import safe_pandas_get_group # for _compute_pybursts_burst_interval_detection
from neuropy.utils.mixins.binning_helpers import BinningContainer # used in _perform_firing_rate_trends_computation
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_firing_rate_trends_computation

import multiprocessing




@custom_define(slots=False)
class SpikeRateTrends(HDFMixin, AttrsBasedClassHelperMixin):
    """ Computes instantaneous firing rates for each cell
    
    """
    epoch_agg_inst_fr_list: np.ndarray = serialized_field(is_computable=True, init=False) # .shape (n_epochs, n_cells)
    cell_agg_inst_fr_list: np.ndarray = serialized_field(is_computable=True, init=False) # .shape (n_cells,)
    all_agg_inst_fr: float = serialized_attribute_field(is_computable=True, init=False) # the scalar value that results from aggregating over ALL (timebins, epochs, cells)
    
    # Add aclu values:


    """ holds information relating to the firing rates of cells across time. 
    
    In general I'd want access to:
    filter_epochs
    instantaneous_time_bin_size_seconds=0.5, kernel=GaussianKernel(200*ms), t_start=0.0, t_stop=1000.0, included_neuron_ids=None
    
    a `inst_fr_df` is a df with time bins along the rows and aclu values along the columns in the style of `unit_specific_binned_spike_counts`
    """
    #TODO 2023-07-31 08:36: - [ ] both of these properties would ideally be serialized to HDF, but they can't be right now.`
    inst_fr_df_list: list[pd.DataFrame] = non_serialized_field() # a list containing a inst_fr_df for each epoch. 
    inst_fr_signals_list: list[AnalogSignal] = non_serialized_field()
    included_neuron_ids: np.ndarray = serialized_field(is_computable=False) # .shape (n_cells,)
    filter_epochs_df: pd.DataFrame = serialized_field(is_computable=False) # .shape (n_epochs, ...)
    
    instantaneous_time_bin_size_seconds: float = serialized_attribute_field(default=0.01, is_computable=False)
    kernel_width_ms: float = serialized_attribute_field(default=10.0, is_computable=False)
    
    
    @classmethod
    def init_from_spikes_and_epochs(cls, spikes_df: pd.DataFrame, filter_epochs, included_neuron_ids=None, instantaneous_time_bin_size_seconds=0.01, kernel=GaussianKernel(10*ms)) -> "SpikeRateTrends":
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids
        if isinstance(filter_epochs, pd.DataFrame):
            filter_epochs_df = filter_epochs
        else:
            filter_epochs_df = filter_epochs.to_dataframe()
            
        epoch_inst_fr_df_list, epoch_inst_fr_signal_list, epoch_agg_firing_rates_list = cls.compute_epochs_unit_avg_inst_firing_rates(spikes_df=spikes_df, filter_epochs=filter_epochs_df, included_neuron_ids=included_neuron_ids, instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel=kernel)
        _out = cls(inst_fr_df_list=epoch_inst_fr_df_list, inst_fr_signals_list=epoch_inst_fr_signal_list, included_neuron_ids=included_neuron_ids, filter_epochs_df=filter_epochs_df,
                    instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel_width_ms=kernel.sigma.magnitude)
        _out.recompute_on_update()
        return _out

    def recompute_on_update(self):
        """ called after update to self.inst_fr_df_list or self.inst_fr_signals_list to update all of the aggregate properties. 

        """
        n_epochs = len(self.inst_fr_df_list)
        assert n_epochs > 0        
        n_cells = self.inst_fr_df_list[0].shape[1]
        epoch_agg_firing_rates_list = np.vstack([a_signal.max(axis=0).magnitude for a_signal in self.inst_fr_signals_list]) # find the peak within each epoch (for all cells) using `.max(...)`
        assert epoch_agg_firing_rates_list.shape == (n_epochs, n_cells)
        self.epoch_agg_inst_fr_list = epoch_agg_firing_rates_list # .shape (n_epochs, n_cells)
        cell_agg_firing_rates_list = epoch_agg_firing_rates_list.mean(axis=0) # find the peak over all epochs (for all cells) using `.max(...)` --- OOPS, what about the zero epochs? Should those actually effect the rate? Should they be excluded?
        assert cell_agg_firing_rates_list.shape == (n_cells,)
        self.cell_agg_inst_fr_list = cell_agg_firing_rates_list # .shape (n_cells,)
        self.all_agg_inst_fr = cell_agg_firing_rates_list.mean() # .magnitude.item() # scalar


    @classmethod
    def compute_simple_time_binned_firing_rates_df(cls, active_spikes_df, time_bin_size_seconds=0.5, debug_print=False):
        """ This simple function computes the firing rates for each time bin. 
        Captures: debug_print
        """
        unit_specific_binned_spike_counts_df, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(active_spikes_df.copy(), max_time_bin_size=time_bin_size_seconds, debug_print=debug_print)
        ## Convert to firing rates in Hz for each bin by dividing by the time bin size
        unit_specific_binned_spike_rate_df = unit_specific_binned_spike_counts_df.astype('float') / time_bin_size_seconds
        return unit_specific_binned_spike_rate_df, unit_specific_binned_spike_counts_df, time_window_edges, time_window_edges_binning_info

    @classmethod
    def compute_instantaneous_time_firing_rates(cls, active_spikes_df, time_bin_size_seconds=0.5, kernel=GaussianKernel(200*ms), t_start=0.0, t_stop=1000.0, included_neuron_ids=None):
        """ I think the error is actually occuring when: `time_bin_size_seconds > (t_stop - t_start)` """
        is_smaller_than_single_bin = (time_bin_size_seconds > (t_stop - t_start))
        assert not is_smaller_than_single_bin, f"ERROR: time_bin_size_seconds ({time_bin_size_seconds}) > (t_stop - t_start) ({t_stop - t_start}). Reduce the bin size or exclude this epoch."
        
        if included_neuron_ids is None:
            included_neuron_ids = np.unique(active_spikes_df.aclu)
        # unit_split_spiketrains = [SpikeTrain(t_start=computation_result.sess.t_start, t_stop=computation_result.sess.t_stop, times=spiketrain_times, units=s) for spiketrain_times in computation_result.sess.spikes_df.spikes.time_sliced(t_start=computation_result.sess.t_start, t_stop=computation_result.sess.t_stop).spikes.get_unit_spiketrains()]
        unit_split_spiketrains = [SpikeTrain(t_start=t_start, t_stop=t_stop, times=spiketrain_times, units=s) for spiketrain_times in active_spikes_df.spikes.time_sliced(t_start=t_start, t_stop=t_stop).spikes.get_unit_spiketrains(included_neuron_ids=included_neuron_ids)]
        # len(unit_split_spiketrains) # 52
        # is_spiketrain_empty = [np.size(spiketrain) == 0 for spiketrain in unit_split_spiketrains]
        # neo.core.spiketrain.SpikeTrain
        # rate : neo.AnalogSignal
        #         2D matrix that contains the rate estimation in unit hertz (Hz) of shape
        #         ``(time, len(spiketrains))`` or ``(time, 1)`` in case of a single
        #         input spiketrain. `rate.times` contains the time axis of the rate
        #         estimate: the unit of this property is the same as the resolution that
        #         is given via the argument `sampling_period` to the function.

        # elephant.statistics.instantaneous_rate

        inst_rate = instantaneous_rate(unit_split_spiketrains, sampling_period=time_bin_size_seconds*s, kernel=kernel) # ValueError: `bins` must be positive, when an integer
            # Raises `TypeError: The input must be a list of SpikeTrain` when the unit_split_spiketrains are empty, which occurs at least when included_neuron_ids is empty
        # AnalogSignal
        # print(type(inst_rate), f"of shape {inst_rate.shape}: {inst_rate.shape[0]} samples, {inst_rate.shape[1]} channel")
        # print('sampling rate:', inst_rate.sampling_rate)
        # print('times (first 10 samples): ', inst_rate.times[:10])
        # print('instantaneous rate (first 10 samples):', inst_rate.T[0, :10])
        # neuron_IDXs = np.arange(len(included_neuron_ids))
        instantaneous_unit_specific_spike_rate_values = pd.DataFrame(inst_rate.magnitude, columns=included_neuron_ids) # builds a df with times along the rows and aclu values along the columns in the style of unit_specific_binned_spike_counts
        return instantaneous_unit_specific_spike_rate_values, inst_rate, unit_split_spiketrains

    @classmethod
    def compute_epochs_unit_avg_inst_firing_rates(cls, spikes_df: pd.DataFrame, filter_epochs, included_neuron_ids=None, instantaneous_time_bin_size_seconds=0.02, kernel=GaussianKernel(20*ms), debug_print=False):
        """Computes the average firing rate for each neuron (unit) in each epoch. 
            Usage:
            epoch_inst_fr_df_list, epoch_avg_firing_rates_list = SpikeRateTrends.compute_epochs_unit_avg_inst_firing_rates(spikes_df=filter_epoch_spikes_df_L, filter_epochs=epochs_df_L, included_neuron_ids=EITHER_subset.track_exclusive_aclus, debug_print=True)
        """
        epoch_inst_fr_df_list = []
        epoch_inst_fr_signal_list = []
        epoch_avg_firing_rates_list = []
        
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids

        if isinstance(filter_epochs, pd.DataFrame):
            filter_epochs_df = filter_epochs
        else:
            filter_epochs_df = filter_epochs.to_dataframe()
            
        if debug_print:
            print(f'filter_epochs: {filter_epochs.epochs.n_epochs}')
        
        for epoch_id in np.arange(np.shape(filter_epochs_df)[0]):
            epoch_start = filter_epochs_df.start.values[epoch_id]
            epoch_end = filter_epochs_df.stop.values[epoch_id]
            epoch_spikes_df = spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)

            unit_specific_inst_spike_rate_values_df, unit_specific_inst_spike_rate_signal, _unit_split_spiketrains = SpikeRateTrends.compute_instantaneous_time_firing_rates(epoch_spikes_df, time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel=kernel,
                                                                                                                                                                            t_start=epoch_start, t_stop=epoch_end, included_neuron_ids=included_neuron_ids)
            epoch_inst_fr_df_list.append(unit_specific_inst_spike_rate_values_df)
            epoch_inst_fr_signal_list.append(unit_specific_inst_spike_rate_signal)

            # Compute average firing rate for each neuron
            unit_avg_firing_rates = np.nanmean(unit_specific_inst_spike_rate_signal.magnitude, axis=0)
            epoch_avg_firing_rates_list.append(unit_avg_firing_rates)
            
        epoch_avg_firing_rates_list = np.vstack(epoch_avg_firing_rates_list)

        return epoch_inst_fr_df_list, epoch_inst_fr_signal_list, epoch_avg_firing_rates_list





class SpikeAnalysisComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'burst_detection'
    _computationPrecidence = 4
    _is_global = False

    @function_attributes(short_name=None, tags=['spikes','burst'], input_requires=[], output_provides=[], uses=['safe_pandas_get_group','pybursts.kleinberg'], used_by=[], creation_date='2023-09-12 17:27', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['burst_detection'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['burst_detection']['burst_intervals']), is_global=False)
    def _perform_spike_burst_detection_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes periods when the cells are firing in bursts in a hierarchical manner
        
        Requires:
            computation_result.sess
            
        Optional Requires:
            computation_result.computation_config['spike_analysis']
        
        Provides:
        computation_result.computed_data['burst_detection']
            ['burst_detection']['burst_intervals']
    
        Model Goal:
            "Thus, the combined goal is to track the sequence of gaps as well as possible without changing state too much."
        
        Model Parameters:            
            s: the base of the exponential distribution that is used for modeling the event frequencies
                scaling parameter `s`: controls the "resolution" with which the discrete rate values of the states are able to track the real-valued gaps; 
            gamma: coefficient for the transition costs between states
                parameter `gamma`: (default 1.0) controls the ease with which the automaton can change states

        """
        def _compute_pybursts_burst_interval_detection(sess, max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicParameters(s=2, gamma=0.1), use_progress_bar=False, debug_print=False):
            """ Computes spike bursts in a hierarchical manner """
            # Build the progress bar:
            if use_progress_bar:
                p_bar = tqdm(sess.spikes_df.spikes.neuron_ids) # Progress bar version
            else:
                p_bar = sess.spikes_df.spikes.neuron_ids # Non-progress bar version, just wrap the iterable

            # # Parallel version:
            # num_processes = 8
            # with multiprocessing.Pool(processes=num_processes) as pool:
            #     # results = pool.map(_compute_pybursts_burst_interval_detection_single_cell, elements)
            #     results = pool.starmap(_compute_pybursts_burst_interval_detection_single_cell, [(sess.spikes_df, a_cell_id, max_num_spikes_per_neuron, kleinberg_parameters.s, kleinberg_parameters.gamma) for a_cell_id in p_bar])

            # return IndexedOrderedDict({a_cell_id: result for a_cell_id, result in zip(p_bar, results)})

            # Non-parallel version:
            out_pyburst_intervals = IndexedOrderedDict()
            for (i, a_cell_id) in enumerate(p_bar):
                # loop through the cell_ids  
                if i == 0 or True:
                    if debug_print:
                        print(f'computing burst intervals for {a_cell_id}...')
                    curr_df = safe_pandas_get_group(sess.spikes_df.groupby('aclu'), a_cell_id)
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
        default_spike_analysis_config = DynamicParameters(max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicParameters(s=2, gamma=0.1), use_progress_bar=False, debug_print=False)
        
        active_spike_analysis_config = computation_result.computation_config.get('spike_analysis', default_spike_analysis_config)
        active_spike_analysis_config = (default_spike_analysis_config | active_spike_analysis_config) # augment the actual values of the analysis config with the defaults if they're unavailable. This allows the user to pass only partially complete parameters in .spike_analysis
        
        ## Set the config values from the defaults to ensure we have access to them later:
        computation_result.computation_config['spike_analysis'] = active_spike_analysis_config
        out_pyburst_intervals = _compute_pybursts_burst_interval_detection(computation_result.sess, **active_spike_analysis_config)        
        computation_result.computed_data[SpikeAnalysisComputations._computationGroupName] = DynamicParameters.init_from_dict({'burst_intervals': out_pyburst_intervals})
            
        """ 
        Access via ['burst_detection']['burst_intervals']
        Example:
            active_burst_info = curr_active_pipeline.computation_results['maze1'].computed_data['burst_detection']
            active_burst_info
        """
        return computation_result
    
    

    @function_attributes(short_name='firing_rate_trends', tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-31 00:00', related_items=[],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['time_binned_position_df']), is_global=False)
    def _perform_firing_rate_trends_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes trends and time-courses of each neuron's firing rate. 
        
        Requires:
            ['pf2D']
            
        Provides:
            computation_result.computed_data['firing_rate_trends']
                ['firing_rate_trends']['time_bin_size_seconds']
                
                ['firing_rate_trends']['all_session_spikes']:
                    ['firing_rate_trends']['all_session_spikes']['time_binning_container']
                    ['firing_rate_trends']['all_session_spikes']['time_window_edges']
                    ['firing_rate_trends']['all_session_spikes']['time_window_edges_binning_info']
                    ['firing_rate_trends']['all_session_spikes']['time_binned_unit_specific_binned_spike_rate']
                    ['firing_rate_trends']['all_session_spikes']['min_spike_rates']
                    ['firing_rate_trends']['all_session_spikes']['mean_spike_rates']
                    ['firing_rate_trends']['all_session_spikes']['median_spike_rates']
                    ['firing_rate_trends']['all_session_spikes']['max_spike_rates']
                    ['firing_rate_trends']['all_session_spikes']['instantaneous_unit_specific_spike_rate']
                    ['firing_rate_trends']['all_session_spikes']['instantaneous_unit_specific_spike_rate_values_df']
                    
                ['firing_rate_trends']['pf_included_spikes_only']:
                    ['firing_rate_trends']['pf_included_spikes_only']['time_binning_container']
                    ['firing_rate_trends']['pf_included_spikes_only']['time_window_edges']
                    ['firing_rate_trends']['pf_included_spikes_only']['time_window_edges_binning_info']
                    ['firing_rate_trends']['pf_included_spikes_only']['time_binned_unit_specific_binned_spike_rate']
                    ['firing_rate_trends']['pf_included_spikes_only']['min_spike_rates']
                    ['firing_rate_trends']['pf_included_spikes_only']['mean_spike_rates']
                    ['firing_rate_trends']['pf_included_spikes_only']['median_spike_rates']
                    ['firing_rate_trends']['pf_included_spikes_only']['max_spike_rates']
        
        """

        time_bin_size_seconds = 0.5
        
        ## Compute for all the session spikes first:
        active_session_spikes_df = computation_result.sess.spikes_df.copy()
        sess_unit_specific_binned_spike_rate_df, sess_unit_specific_binned_spike_counts_df, sess_time_window_edges, sess_time_window_edges_binning_info = SpikeRateTrends.compute_simple_time_binned_firing_rates_df(active_session_spikes_df, time_bin_size_seconds=time_bin_size_seconds)
        sess_time_binning_container = BinningContainer(edges=sess_time_window_edges, edge_info=sess_time_window_edges_binning_info)

        sess_min_spike_rates = sess_unit_specific_binned_spike_rate_df.min()
        sess_mean_spike_rates = sess_unit_specific_binned_spike_rate_df.mean()
        sess_median_spike_rates = sess_unit_specific_binned_spike_rate_df.median()
        sess_max_spike_rates = sess_unit_specific_binned_spike_rate_df.max()
            
        # Instantaneous versions:
        sess_unit_specific_inst_spike_rate_values_df, sess_unit_specific_inst_spike_rate, sess_unit_split_spiketrains = SpikeRateTrends.compute_instantaneous_time_firing_rates(active_session_spikes_df, time_bin_size_seconds=time_bin_size_seconds, t_start=computation_result.sess.t_start, t_stop=computation_result.sess.t_stop)
        if debug_print:
            print(f'sess_unit_specific_inst_spike_rate: {sess_unit_specific_inst_spike_rate}')

        # Compute for only the placefield included spikes as well:
        active_pf_2D = computation_result.computed_data['pf2D']
        active_pf_included_spikes_only_spikes_df = active_pf_2D.filtered_spikes_df.copy()
        pf_only_unit_specific_binned_spike_rate_df, pf_only_unit_specific_binned_spike_counts_df, pf_only_time_window_edges, pf_only_time_window_edges_binning_info = SpikeRateTrends.compute_simple_time_binned_firing_rates_df(active_pf_included_spikes_only_spikes_df)
        pf_only_time_binning_container = BinningContainer(edges=pf_only_time_window_edges, edge_info=pf_only_time_window_edges_binning_info)
        pf_only_min_spike_rates = pf_only_unit_specific_binned_spike_rate_df.min()
        pf_only_mean_spike_rates = pf_only_unit_specific_binned_spike_rate_df.mean()
        pf_only_median_spike_rates = pf_only_unit_specific_binned_spike_rate_df.median()
        pf_only_max_spike_rates = pf_only_unit_specific_binned_spike_rate_df.max()

        computation_result.computed_data['firing_rate_trends'] = DynamicParameters.init_from_dict({
            'time_bin_size_seconds': time_bin_size_seconds,
            'all_session_spikes': DynamicParameters.init_from_dict({
                'time_binning_container': sess_time_binning_container,
                'time_window_edges': sess_time_window_edges,
                'time_window_edges_binning_info': sess_time_window_edges_binning_info,
                'time_binned_unit_specific_binned_spike_rate': sess_unit_specific_binned_spike_rate_df,
                'time_binned_unit_specific_binned_spike_counts': sess_unit_specific_binned_spike_counts_df,
                'min_spike_rates': sess_min_spike_rates,
                'mean_spike_rates': sess_mean_spike_rates,
                'median_spike_rates': sess_median_spike_rates,
                'max_spike_rates': sess_max_spike_rates,
                'instantaneous_unit_specific_spike_rate': sess_unit_specific_inst_spike_rate,
                'instantaneous_unit_specific_spike_rate_values_df': sess_unit_specific_inst_spike_rate_values_df,
            }),
            'pf_included_spikes_only': DynamicParameters.init_from_dict({
                'time_binning_container': pf_only_time_binning_container,
                'time_window_edges': pf_only_time_window_edges,
                'time_window_edges_binning_info': pf_only_time_window_edges_binning_info,
                'time_binned_unit_specific_binned_spike_rate': pf_only_unit_specific_binned_spike_rate_df,
                'time_binned_unit_specific_binned_spike_counts': pf_only_unit_specific_binned_spike_counts_df,
                'min_spike_rates': pf_only_min_spike_rates,
                'mean_spike_rates': pf_only_mean_spike_rates,
                'median_spike_rates': pf_only_median_spike_rates,
                'max_spike_rates': pf_only_max_spike_rates,                
            }),
        })
        return computation_result
        # can access via:
        # active_firing_rate_trends = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('firing_rate_trends', None)
        # active_time_bin_size_seconds = active_firing_rate_trends['time_bin_size_seconds']
        # active_all_session_spikes = active_firing_rate_trends['all_session_spikes']
        # active_pf_included_spikes_only = active_firing_rate_trends['pf_included_spikes_only']
        # active_time_binning_container, active_time_window_edges, active_time_window_edges_binning_info, active_time_binned_unit_specific_binned_spike_rate, active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binning_container'], pf_included_spikes_only['time_window_edges'], pf_included_spikes_only['time_window_edges_binning_info'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_rate'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']

        # active_time_binning_container = pf_included_spikes_only['time_binning_container']
        # active_time_window_edges = pf_included_spikes_only['time_window_edges']
        # active_time_window_edges_binning_info = pf_included_spikes_only['time_window_edges_binning_info']
        # active_time_binned_unit_specific_binned_spike_rate = pf_included_spikes_only['time_binned_unit_specific_binned_spike_rate']
        # active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']


    # def _perform_instantaneous_firing_rates_for_epochs_computation(computation_result: ComputationResult, debug_print=False):
    #     """ Computes trends and time-courses of each neuron's firing rate. 
        
    #     Requires:
    #         ['pf2D']
            
    #     Provides:
    #         computation_result.computed_data['firing_rate_trends']
    #             ['firing_rate_trends']['time_bin_size_seconds']
                
    #             ['firing_rate_trends']['all_session_spikes']:
    #                 ['firing_rate_trends']['all_session_spikes']['time_binning_container']
    #                 ['firing_rate_trends']['all_session_spikes']['time_window_edges']
    #                 ['firing_rate_trends']['all_session_spikes']['time_window_edges_binning_info']
    #                 ['firing_rate_trends']['all_session_spikes']['time_binned_unit_specific_binned_spike_rate']
    #                 ['firing_rate_trends']['all_session_spikes']['min_spike_rates']
    #                 ['firing_rate_trends']['all_session_spikes']['mean_spike_rates']
    #                 ['firing_rate_trends']['all_session_spikes']['median_spike_rates']
    #                 ['firing_rate_trends']['all_session_spikes']['max_spike_rates']
    #                 ['firing_rate_trends']['all_session_spikes']['instantaneous_unit_specific_spike_rate']
    #                 ['firing_rate_trends']['all_session_spikes']['instantaneous_unit_specific_spike_rate_values_df']
                    
    #             ['firing_rate_trends']['pf_included_spikes_only']:
    #                 ['firing_rate_trends']['pf_included_spikes_only']['time_binning_container']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['time_window_edges']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['time_window_edges_binning_info']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['time_binned_unit_specific_binned_spike_rate']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['min_spike_rates']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['mean_spike_rates']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['median_spike_rates']
    #                 ['firing_rate_trends']['pf_included_spikes_only']['max_spike_rates']
        
    #     """


    #     long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    #     long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session
    #     (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)
        

    #     long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
    #     x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
    #     active_context = long_short_fr_indicies_analysis_results['active_context']
    #     long_laps, long_replays, short_laps, short_replays, global_laps, global_replays = [long_short_fr_indicies_analysis_results[k] for k in ['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays']]


    #     # Replays: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_replays`, `short_replays`
    #     # LxC: `long_exclusive.track_exclusive_aclus`
    #     # ReplayDeltaMinus: `long_replays`
    #     LxC_ReplayDeltaMinus = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus)
    #     # ReplayDeltaPlus: `short_replays`
    #     LxC_ReplayDeltaPlus = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus)

    #     # SxC: `short_exclusive.track_exclusive_aclus`
    #     # ReplayDeltaMinus: `long_replays`
    #     SxC_ReplayDeltaMinus = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus)
    #     # ReplayDeltaPlus: `short_replays`
    #     SxC_ReplayDeltaPlus = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus)

    #     # Note that in general LxC and SxC might have differing numbers of cells.
    #     Fig2_Replay_FR = [(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std()) for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus)]
    #     Fig2_Replay_FR

    #     # Laps/Theta: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_laps`, `short_laps`
    #     # LxC: `long_exclusive.track_exclusive_aclus`
    #     # ThetaDeltaMinus: `long_laps`
    #     LxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus)
    #     # ThetaDeltaPlus: `short_laps`
    #     LxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus)

    #     # SxC: `short_exclusive.track_exclusive_aclus`
    #     # ThetaDeltaMinus: `long_laps`
    #     SxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus)
    #     # ThetaDeltaPlus: `short_laps`
    #     SxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus)

    #     # Note that in general LxC and SxC might have differing numbers of cells.
    #     Fig2_Laps_FR = [(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std()) for v in (LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus)]
    #     Fig2_Laps_FR


    #     computation_result.computed_data['firing_rate_trends'] = DynamicParameters.init_from_dict({
    #         'time_bin_size_seconds': time_bin_size_seconds,
    #         'all_session_spikes': DynamicParameters.init_from_dict({
    #             'time_binning_container': sess_time_binning_container,
    #             'time_window_edges': sess_time_window_edges,
    #             'time_window_edges_binning_info': sess_time_window_edges_binning_info,
    #             'time_binned_unit_specific_binned_spike_rate': sess_unit_specific_binned_spike_rate_df,
    #             'time_binned_unit_specific_binned_spike_counts': sess_unit_specific_binned_spike_counts_df,
    #             'min_spike_rates': sess_min_spike_rates,
    #             'mean_spike_rates': sess_mean_spike_rates,
    #             'median_spike_rates': sess_median_spike_rates,
    #             'max_spike_rates': sess_max_spike_rates,
    #             'instantaneous_unit_specific_spike_rate': sess_unit_specific_inst_spike_rate,
    #             'instantaneous_unit_specific_spike_rate_values_df': sess_unit_specific_inst_spike_rate_values_df,
    #         }),
    #         'pf_included_spikes_only': DynamicParameters.init_from_dict({
    #             'time_binning_container': pf_only_time_binning_container,
    #             'time_window_edges': pf_only_time_window_edges,
    #             'time_window_edges_binning_info': pf_only_time_window_edges_binning_info,
    #             'time_binned_unit_specific_binned_spike_rate': pf_only_unit_specific_binned_spike_rate_df,
    #             'time_binned_unit_specific_binned_spike_counts': pf_only_unit_specific_binned_spike_counts_df,
    #             'min_spike_rates': pf_only_min_spike_rates,
    #             'mean_spike_rates': pf_only_mean_spike_rates,
    #             'median_spike_rates': pf_only_median_spike_rates,
    #             'max_spike_rates': pf_only_max_spike_rates,                
    #         }),
    #     })
    #     return computation_result
    #     # can access via:
    #     # active_firing_rate_trends = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('firing_rate_trends', None)
    #     # active_time_bin_size_seconds = active_firing_rate_trends['time_bin_size_seconds']
    #     # active_all_session_spikes = active_firing_rate_trends['all_session_spikes']
    #     # active_pf_included_spikes_only = active_firing_rate_trends['pf_included_spikes_only']
    #     # active_time_binning_container, active_time_window_edges, active_time_window_edges_binning_info, active_time_binned_unit_specific_binned_spike_rate, active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binning_container'], pf_included_spikes_only['time_window_edges'], pf_included_spikes_only['time_window_edges_binning_info'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_rate'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']

    #     # active_time_binning_container = pf_included_spikes_only['time_binning_container']
    #     # active_time_window_edges = pf_included_spikes_only['time_window_edges']
    #     # active_time_window_edges_binning_info = pf_included_spikes_only['time_window_edges_binning_info']
    #     # active_time_binned_unit_specific_binned_spike_rate = pf_included_spikes_only['time_binned_unit_specific_binned_spike_rate']
    #     # active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']



# ==================================================================================================================== #
# Private Functions                                                                                                    #
# ==================================================================================================================== #
def _compute_pybursts_burst_interval_detection_single_cell(spikes_df, a_cell_id, max_num_spikes_per_neuron=20000, s=2, gamma=0.1):
    """ computes the burst intervals for a single cell using pybursts. Written this way so that it can be parallelized."""
    curr_df = safe_pandas_get_group(spikes_df.groupby('aclu'), a_cell_id)
    curr_spike_train = curr_df[curr_df.spikes.time_variable_name].to_numpy()
    num_curr_spikes = len(curr_spike_train)

    # For testing, limit to the first 1000 spikes:
    num_curr_spikes = min(num_curr_spikes, max_num_spikes_per_neuron)
    curr_spike_train = curr_spike_train[:num_curr_spikes]

    # Perform the computation:
    curr_pyburst_intervals = pybursts.kleinberg(curr_spike_train, s=s, gamma=gamma)
    # Convert ouptut intervals to dataframe
    curr_pyburst_interval_df = pd.DataFrame(curr_pyburst_intervals, columns=['burst_level', 't_start', 't_end'])
    curr_pyburst_interval_df['t_duration'] = curr_pyburst_interval_df.t_end - curr_pyburst_interval_df.t_start

    # Convert vectors to tuples of (t_start, t_duration) pairs:
    curr_pyburst_interval_df['interval_pair'] = list(zip(curr_pyburst_interval_df.t_start, curr_pyburst_interval_df.t_duration)) # pairs like # [(33, 4), (76, 16), (76, 1)]
    return curr_pyburst_interval_df