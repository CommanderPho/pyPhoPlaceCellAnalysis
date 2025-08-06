from copy import deepcopy
from attrs import define, field, asdict, Factory
from neuropy.analyses.decoders import BinningInfo
from nptyping import NDArray
import numpy as np
import pandas as pd
from indexed import IndexedOrderedDict
from typing import Any, Optional, Dict, List, Tuple, Union
import nptyping as ND
from nptyping import NDArray
import itertools

# from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
# from neurodsp.plts.time_series import plot_time_series, plot_bursts # for plotting results

# PyBursts:
from pybursts import pybursts

# 2022-11-08 Firing Rate Calculations
from elephant.statistics import mean_firing_rate, instantaneous_rate, time_histogram
import quantities as pq
from quantities import ms, s, Hz
from neo.core.spiketrain import SpikeTrain
from neo.core.analogsignal import AnalogSignal
from elephant.kernels import GaussianKernel

# For Progress bars:
from tqdm.notebook import tqdm

from neuropy.utils.misc import safe_pandas_get_group # for _compute_pybursts_burst_interval_detection
from neuropy.core.epoch import ensure_dataframe
from neuropy.utils.mixins.binning_helpers import BinningContainer # used in _perform_firing_rate_trends_computation
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol
from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_firing_rate_trends_computation


@custom_define(slots=False)
class SpikeRateTrends(HDFMixin, NeuronUnitSlicableObjectProtocol, AttrsBasedClassHelperMixin):
    """ Computes instantaneous firing rates for each cell
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends

    
    """
    epoch_agg_inst_fr_list: NDArray[ND.Shape["N_EPOCHS, N_CELLS"], Any] = serialized_field(is_computable=True, init=False) # .shape (n_epochs, n_cells)
    cell_agg_inst_fr_list: NDArray[ND.Shape["N_CELLS"], Any] = serialized_field(is_computable=True, init=False) # .shape (n_cells,)
    all_agg_inst_fr: float = serialized_attribute_field(is_computable=True, init=False) # the scalar value that results from aggregating over ALL (timebins, epochs, cells)
    
    # Add aclu values:


    """ holds information relating to the firing rates of cells across time. 
    
    In general I'd want access to:
    filter_epochs
    instantaneous_time_bin_size_seconds=0.5, kernel=GaussianKernel(200*ms), t_start=0.0, t_stop=1000.0, included_neuron_ids=None
    
    a `inst_fr_df` is a df with time bins along the rows and aclu values along the columns in the style of `unit_specific_binned_spike_counts`
    """
    #TODO 2023-07-31 08:36: - [ ] both of these properties would ideally be serialized to HDF, but they can't be right now.`
    inst_fr_df_list: List[pd.DataFrame] = non_serialized_field() # a list containing an`inst_fr_df` for each epoch. 
    inst_fr_signals_list: List[AnalogSignal] = non_serialized_field()
    
    included_neuron_ids: Optional[NDArray[ND.Shape["N_CELLS"], Any]] = serialized_field(default=None, is_computable=False) # .shape (n_cells,)
    filter_epochs_df: pd.DataFrame = serialized_field(is_computable=False) # .shape (n_epochs, ...)
    
    instantaneous_time_bin_size_seconds: float = serialized_attribute_field(default=0.01, is_computable=False)
    kernel_width_ms: float = serialized_attribute_field(default=10.0, is_computable=False)

    spike_counts_df_list: Optional[List[pd.DataFrame]] = non_serialized_field(default=None, metadata={}) # a list containing a df for each epoch with its rows as the number of spikes in each time bin and one column for each neuron. 
    epoch_unit_fr_df_list: Optional[List[pd.DataFrame]] = non_serialized_field(default=None, metadata={})
    
    per_aclu_additional_properties_dict: Dict[str, Union[List, Dict]] = non_serialized_field(default=Factory(dict), metadata={'field_added':'2025.08.06_0'})
    
    @classmethod
    def init_from_spikes_and_epochs(cls, spikes_df: pd.DataFrame, filter_epochs: pd.DataFrame, included_neuron_ids=None, instantaneous_time_bin_size_seconds:float=0.01, kernel=GaussianKernel(10*ms),
                                    use_instantaneous_firing_rate:bool=False, epoch_handling_mode:str='DropShorterMode', **kwargs) -> "SpikeRateTrends":
        """ the main called function
        
        epoch_handling_mode='DropShorterMode' - the default mode prior to 2024-09-12, drops any epochs shorter than the time_bin_size so binning works appropriately.
        epoch_handling_mode='UseAllEpochsMode' - if you pass a time_bin_size larger than the duration of an epoch, that epoch will have only one bin (with the bin duration the length of the Epoch, meaning each epoch can have variable bin sizes. Introduced 2024-09-12.
                `instantaneous_time_bin_size_seconds` then effectively becomes a MAXIMUM time_bin_size duration as a smaller bin size (the length of one epoch) can be used if needed
        
        """
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids
        if len(included_neuron_ids)>0:
            filter_epochs_df = ensure_dataframe(filter_epochs) 
            assert epoch_handling_mode in ['DropShorterMode', 'UseAllEpochsMode']

            if epoch_handling_mode == 'DropShorterMode':
                minimum_event_duration: float = instantaneous_time_bin_size_seconds # allow direct use            
                ## Drop those less than the time bin duration
                print(f'DropShorterMode:')
                pre_drop_n_epochs = len(filter_epochs_df)
                if minimum_event_duration is not None:                
                    filter_epochs_df = filter_epochs_df[filter_epochs_df['duration'] > minimum_event_duration]
                    post_drop_n_epochs = len(filter_epochs_df)
                    n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                    print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
                else:
                    filter_epochs_df = filter_epochs_df[filter_epochs_df['duration'] > instantaneous_time_bin_size_seconds]
                    post_drop_n_epochs = len(filter_epochs_df)
                    n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                    print(f'\tdropping {n_dropped_epochs} that are shorter than our instantaneous_time_bin_size_seconds of {instantaneous_time_bin_size_seconds}', end='\t') 

                print(f'{post_drop_n_epochs} remain.')
            elif epoch_handling_mode == 'UseAllEpochsMode':
                # Don't drop any epochs
                print(f'UseAllEpochsMode')
                pass
            else:
                raise NotImplementedError(f'epoch_handling_mode: "{epoch_handling_mode}" is unsupported.')
            
            epoch_inst_fr_df_list, epoch_inst_fr_signal_list, epoch_agg_firing_rates_list, epoch_results_list_dict = cls.compute_epochs_unit_avg_inst_firing_rates(spikes_df=spikes_df, filter_epochs=filter_epochs_df, included_neuron_ids=included_neuron_ids, instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel=kernel,
                                                                                                                                          use_instantaneous_firing_rate=use_instantaneous_firing_rate, **kwargs)
            _out = cls(inst_fr_df_list=epoch_inst_fr_df_list, inst_fr_signals_list=epoch_inst_fr_signal_list, included_neuron_ids=included_neuron_ids, filter_epochs_df=filter_epochs_df,
                        instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel_width_ms=kernel.sigma.magnitude,
                        spike_counts_df_list=epoch_results_list_dict.pop('spike_counts', None),
                        epoch_unit_fr_df_list=epoch_results_list_dict.pop('epoch_unit_fr', None),
                        )
            _out.per_aclu_additional_properties_dict.update(**epoch_results_list_dict) ## add the rest of the returned properties
            _out.recompute_on_update()
        else:
            _out = None # return None if included_neuron_ids are empty

        return _out


    def recompute_on_update(self):
        """ called after update to self.inst_fr_df_list or self.inst_fr_signals_list to update all of the aggregate properties. 

        """
        n_epochs: int = len(self.inst_fr_df_list)
        assert n_epochs > 0        
        n_cells: int = self.inst_fr_df_list[0].shape[1]
        
        ## Recompute for all time bins within each epoch to get at epoch_aggregated firing rates (one for each epoch):
        is_non_instantaneous = np.all([a_signal is None for a_signal in self.inst_fr_signals_list])
        if is_non_instantaneous:
            epoch_agg_firing_rates_list = np.vstack([np.nanmean(a_df.to_numpy(), axis=0) for a_df in self.inst_fr_df_list])
        else:
            # use instantaneous version
            epoch_agg_firing_rates_list = np.vstack([a_signal.max(axis=0).magnitude for a_signal in self.inst_fr_signals_list]) # find the peak within each epoch (for all cells) using `.max(...)` - # (226, 66) - (n_epochs, n_cells)
            # epoch_agg_firing_rates_list = epoch_agg_firing_rates_list.mean(axis=0)
            

        assert epoch_agg_firing_rates_list.shape == (n_epochs, n_cells)
        self.epoch_agg_inst_fr_list = epoch_agg_firing_rates_list # .shape (n_epochs, n_cells)

        ## OLD WAY that abused directional cells:
        # cell_agg_firing_rates_list = epoch_agg_firing_rates_list.mean(axis=0) # find the peak over all epochs (for all cells) using `.max(...)` --- OOPS, what about the zero epochs? Should those actually effect the rate? Should they be excluded?

        ## 2025-08-01 07:29 NEW WAY that permits LR/RL/ALL consideration separately so purely directional cells don't get averaged acrossed periods they aren't supposed to be active:
        ## INPUTS: epoch_agg_inst_fr_list # (N_EPOCHS, N_ACLUS) in period

        # an_inst_fr_list = self.epoch_agg_inst_fr_list # (N_EPOCHS, N_ACLUS) in period
        # epoch_agg_firing_rates_list = deepcopy(epoch_agg_firing_rates_list) # (N_EPOCHS, N_ACLUS) in period
        # an_inst_fr_list = np.squeeze(a_pre_post_period_result.epoch_agg_inst_fr_list[:, target_aclu_idx]) # (N_EPOCHS) in period
        # print(f'an_inst_fr_list.shape: {np.shape(an_inst_fr_list)}') # an_inst_fr_list.shape: (39, 20)
        LR_an_inst_fr_list = epoch_agg_firing_rates_list[::2, :] ## even epochs only, all aclus
        RL_an_inst_fr_list = epoch_agg_firing_rates_list[1::2, :] ## odd epochs only, all aclus
        # print(f'LR_an_inst_fr_list.shape: {np.shape(LR_an_inst_fr_list)}') # LR_an_inst_fr_list.shape: (20, 20)
        # print(f'RL_an_inst_fr_list.shape: {np.shape(RL_an_inst_fr_list)}') # RL_an_inst_fr_list.shape: (19, 20)
        a_period_directional_inst_fr_list = [LR_an_inst_fr_list, RL_an_inst_fr_list, epoch_agg_firing_rates_list] # LR, RL, ALL
        a_period_epoch_agg_frs_list = np.vstack([np.nanmean(a_fr_list, axis=0) for a_fr_list in a_period_directional_inst_fr_list]) ## average over epochs, output (3, N_ACLUS)
        # print(f'a_period_epoch_agg_frs_list.shape: {np.shape(a_period_epoch_agg_frs_list)}') # a_period_epoch_agg_frs_list.shape: (3, 20)
        cell_agg_firing_rates_list: float = np.nanmax(a_period_epoch_agg_frs_list, axis=0) ## get the highest fr in any the LR/RL/ALL only
        # print(f'a_period_epoch_agg_fr.shape: {np.shape(a_period_epoch_agg_fr)}') # a_period_epoch_agg_fr.shape: (20,)
    
        ## OVERWRITE cell_agg_inst_fr_list
        assert cell_agg_firing_rates_list.shape == (n_cells,)
        self.cell_agg_inst_fr_list = deepcopy(cell_agg_firing_rates_list) # .shape (n_cells,)
        ## update the all agg result
        self.all_agg_inst_fr = cell_agg_firing_rates_list.mean() # .magnitude.item() # scalar




    @classmethod
    def compute_simple_time_binned_firing_rates_df(cls, active_spikes_df: pd.DataFrame, time_bin_size_seconds: float=0.5, debug_print=False) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray, BinningInfo]:
        """ This simple function computes the firing rates for each time bin. 
        Captures: debug_print
        """
        unit_specific_binned_spike_counts_df, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(active_spikes_df.copy(), max_time_bin_size=time_bin_size_seconds, debug_print=debug_print)
        ## Convert to firing rates in Hz for each bin by dividing by the time bin size
        unit_specific_binned_spike_rate_df = unit_specific_binned_spike_counts_df.astype('float') / time_bin_size_seconds
        return unit_specific_binned_spike_rate_df, unit_specific_binned_spike_counts_df, time_window_edges, time_window_edges_binning_info

    @classmethod
    def compute_single_epoch_instantaneous_time_firing_rates(cls, active_spikes_df: pd.DataFrame, time_bin_size_seconds:float=0.5, kernel=GaussianKernel(200*ms), epoch_t_start:float=0.0, epoch_t_stop:float=1000.0, included_neuron_ids=None, **kwargs) -> Tuple[pd.DataFrame, Any, List[SpikeTrain]]:
        """ I think the error is actually occuring when: `time_bin_size_seconds > (t_stop - t_start)`
        
        2025-07-25 07:43 Confirmed that changing `sampling_rate: pq.Quantity = (1252 * Hz)` has no effect on the output instantaneous rates
        
        """
        
        is_smaller_than_single_bin = (time_bin_size_seconds > (epoch_t_stop - epoch_t_start))
        assert not is_smaller_than_single_bin, f"ERROR: time_bin_size_seconds ({time_bin_size_seconds}) > (t_stop - t_start) ({epoch_t_stop - epoch_t_start}). Reduce the bin size or exclude this epoch."
        
        if included_neuron_ids is None:
            included_neuron_ids = np.unique(active_spikes_df.aclu)
            
        active_spikes_df = deepcopy(active_spikes_df).spikes.time_sliced(t_start=epoch_t_start, t_stop=epoch_t_stop)
        
        # unit_split_spiketrains = [SpikeTrain(t_start=computation_result.sess.t_start, t_stop=computation_result.sess.t_stop, times=spiketrain_times, units=s) for spiketrain_times in computation_result.sess.spikes_df.spikes.time_sliced(t_start=computation_result.sess.t_start, t_stop=computation_result.sess.t_stop).spikes.get_unit_spiketrains()]
        unit_split_spiketrains = [SpikeTrain(t_start=epoch_t_start, t_stop=epoch_t_stop, times=spiketrain_times, units=s) for spiketrain_times in active_spikes_df.spikes.get_unit_spiketrains(included_neuron_ids=included_neuron_ids)]
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

        inst_rate = instantaneous_rate(unit_split_spiketrains, t_start=epoch_t_start*s, t_stop=epoch_t_stop*s, sampling_period=time_bin_size_seconds*s, kernel=kernel) # ValueError: `bins` must be positive, when an integer
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
    def compute_epochs_unit_avg_inst_firing_rates(cls, spikes_df: pd.DataFrame, filter_epochs, included_neuron_ids=None, instantaneous_time_bin_size_seconds:float=0.02, kernel=GaussianKernel(20*ms), use_instantaneous_firing_rate: bool=False, debug_print=False, **kwargs):
        """Computes the average firing rate for each neuron (unit) in each epoch. 
            Usage:
            epoch_inst_fr_df_list, epoch_inst_fr_signal_list, epoch_avg_firing_rates_list, epoch_results_list_dict = SpikeRateTrends.compute_epochs_unit_avg_inst_firing_rates(spikes_df=filter_epoch_spikes_df_L, filter_epochs=epochs_df_L, included_neuron_ids=EITHER_subset.track_exclusive_aclus, debug_print=True)
        """
        epoch_inst_fr_df_list = []
        epoch_inst_fr_signal_list = []
        epoch_avg_firing_rates_list = []
        epoch_results_list_dict = {'spike_counts': [], 'epoch_avg_spike_counts': [], 'epoch_unit_fr': [], 'epoch_is_participating_dict': []}
        
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids

        if isinstance(filter_epochs, pd.DataFrame):
            filter_epochs_df = filter_epochs
        else:
            filter_epochs_df = filter_epochs.to_dataframe()
            
        if debug_print:
            print(f'filter_epochs: {filter_epochs.epochs.n_epochs}')
        
        n_epochs: int = np.shape(filter_epochs_df)[0]
        for epoch_id in np.arange(n_epochs):
            epoch_start = filter_epochs_df.start.values[epoch_id]
            epoch_end = filter_epochs_df.stop.values[epoch_id]
            epoch_duration: float = epoch_end - epoch_start
            epoch_spikes_df = spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)
            
            if use_instantaneous_firing_rate:
                # True Instantaneous Firing Rate _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                unit_specific_inst_spike_rate_values_df, unit_specific_inst_spike_rate_signal, _unit_split_spiketrains = SpikeRateTrends.compute_single_epoch_instantaneous_time_firing_rates(epoch_spikes_df, time_bin_size_seconds=instantaneous_time_bin_size_seconds, kernel=kernel, epoch_t_start=epoch_start, epoch_t_stop=epoch_end, included_neuron_ids=included_neuron_ids, **kwargs)
                # times accessible via `unit_specific_inst_spike_rate_signal.times`
                epoch_inst_fr_df_list.append(unit_specific_inst_spike_rate_values_df)
                epoch_inst_fr_signal_list.append(unit_specific_inst_spike_rate_signal)

                # Compute average firing rate for each neuron
                unit_avg_firing_rates = np.nanmean(unit_specific_inst_spike_rate_signal.magnitude, axis=0)
                epoch_avg_firing_rates_list.append(unit_avg_firing_rates)
                
                ## spike counts:
                epoch_spike_counts_dict = deepcopy(epoch_spikes_df['aclu']).value_counts().to_dict()            
                epoch_value_counts = []
                for aclu in included_neuron_ids:          
                    epoch_value_counts.append(epoch_spike_counts_dict.get(aclu, 0))
                    # epoch_value_counts.append({k:v.get(aclu, 0) for k, v in epoch_spike_counts_dict.items()})
                    
                epoch_units_total_num_spikes_df: pd.DataFrame = pd.DataFrame(epoch_value_counts, index=deepcopy(included_neuron_ids))
                # unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df=epoch_spikes_df, active_indicies, debug_print=debug_print)
                epoch_results_list_dict['spike_counts'].append(epoch_units_total_num_spikes_df)
                
                unit_specific_binned_spike_counts_df = deepcopy(epoch_units_total_num_spikes_df) ## just to make sure it works
                
            else:
                # Non-instantaneous rate _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                # 2025-07-25 08:51 IMPORTANT: since this is calculated by dividing the number of spikes in each time bin by the duration of each bin, when the duration exceeds the length of the epoch it will be divided by too large of a bin size
                if instantaneous_time_bin_size_seconds > epoch_duration:
                    epoch_instantaneous_time_bin_size_seconds: float = epoch_duration
                else:
                    epoch_instantaneous_time_bin_size_seconds: float = instantaneous_time_bin_size_seconds
                                    
                unit_specific_binned_spike_rate_df, unit_specific_binned_spike_counts_df, time_window_edges, time_window_edges_binning_info = SpikeRateTrends.compute_simple_time_binned_firing_rates_df(epoch_spikes_df, time_bin_size_seconds=epoch_instantaneous_time_bin_size_seconds, debug_print=debug_print) # returns dfs containing only the relevant entries
                n_epoch_time_bins: int = np.shape(unit_specific_binned_spike_rate_df)[0]
                ## Convert the returned df to a "full" representation: containing a column for each aclu in `included_neuron_ids` (which will be all zeros for aclus not active in this epoch)
                unit_specific_binned_spike_rate_dict = unit_specific_binned_spike_rate_df.to_dict('list')
                unit_specific_binned_spike_rate_dict = {aclu:unit_specific_binned_spike_rate_dict.get(aclu, np.zeros((n_epoch_time_bins, ))) for aclu in included_neuron_ids} # add entries for missing aclus
                unit_specific_binned_spike_rate_df = pd.DataFrame(unit_specific_binned_spike_rate_dict) ## Same as the incomming `unit_specific_binned_spike_rate_df` four lines up except it now has columns for EVERY aclu with zeros as needed
                
                epoch_inst_fr_df_list.append(unit_specific_binned_spike_rate_df)
                epoch_inst_fr_signal_list.append(None)

                # Compute average firing rate for each neuron
                unit_avg_firing_rates = np.nanmean(unit_specific_binned_spike_rate_df.to_numpy(), axis=0) # (n_neurons, )
                epoch_avg_firing_rates_list.append(unit_avg_firing_rates)
                
                ## Spike Counts
                ## Convert the returned df to a "full" representation: containing a column for each aclu in `included_neuron_ids` (which will be all zeros for aclus not active in this epoch)
                unit_specific_binned_spike_counts_dict = unit_specific_binned_spike_counts_df.to_dict('list')
                unit_specific_binned_spike_counts_dict = {aclu:unit_specific_binned_spike_counts_dict.get(aclu, np.zeros((n_epoch_time_bins, ))) for aclu in included_neuron_ids} # add entries for missing aclus
                unit_specific_binned_spike_counts_df = pd.DataFrame(unit_specific_binned_spike_counts_dict)
                ## OUTPUT unit_specific_binned_spike_counts_df to spike_counts
                epoch_results_list_dict['spike_counts'].append(unit_specific_binned_spike_counts_df) # one column for every aclu, and one row for every time bin in the epoch
                # unit_avg_spike_counts = np.nanmean(unit_specific_binned_spike_counts_df.to_numpy(), axis=0) # (n_neurons, )
                # epoch_results_list_dict['epoch_avg_spike_counts'].append(unit_avg_spike_counts)
                
            # Either instantaneous or non-instantaneous
            # unit_specific_has_epoch_participation_dict = {aclu:(unit_specific_binned_spike_counts_dict.get(aclu, np.zeros((n_epoch_time_bins, ))) > 0) for aclu in included_neuron_ids}
            # epoch_results_list_dict['epoch_is_participating_dict'].append(unit_specific_has_epoch_participation_dict)

            ## Sum up the spikes per epoch for each cell, and then divide each by the epoch duration to get the epoch firing rate
            unit_approximate_entire_epoch_fr_df: pd.DataFrame = deepcopy(unit_specific_binned_spike_counts_df.sum(axis='index', skipna=True, numeric_only=True)).astype(float) / epoch_duration # np.nanmean(unit_specific_binned_spike_counts_df.to_numpy(), axis=0) # (n_neurons, )
            epoch_results_list_dict['epoch_unit_fr'].append(unit_approximate_entire_epoch_fr_df)
                
        ## END for epoch_id in np.arange(n_epochs)...
        has_epoch_participation: NDArray = np.vstack([(unit_specific_binned_spike_counts_df.T[0].to_numpy() > 0.0) for unit_specific_binned_spike_counts_df in epoch_results_list_dict['spike_counts']]) # has_epoch_participation # .shape # (39, 20) - (n_epochs, n_aclus)
        n_participating_epochs: NDArray = has_epoch_participation.sum(axis=0) # .shape (N_ACLUS)
        assert len(included_neuron_ids) == len(n_participating_epochs), f"len(included_neuron_ids): {len(included_neuron_ids)} != len(n_participating_epochs): {len(n_participating_epochs)}"
        epoch_results_list_dict['epoch_is_participating_dict'] = deepcopy(has_epoch_participation)


        epoch_avg_firing_rates_list = np.vstack(epoch_avg_firing_rates_list)
        # epoch_results_list_dict['spike_counts'] = np.vstack(epoch_results_list_dict['spike_counts'])

        return epoch_inst_fr_df_list, epoch_inst_fr_signal_list, epoch_avg_firing_rates_list, epoch_results_list_dict

    @function_attributes(short_name=None, tags=['private', 'participation', 'additional'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-06 05:33', related_items=[])
    @classmethod
    def _perform_compute_participation_stats(cls, a_pre_post_period_result: "SpikeRateTrends", a_session_ctxt: Optional[IdentifyingContext]=None):
        """ Uses: `a_pre_post_period_result.spike_counts_df_list`
        """
        active_ACLUS = deepcopy(a_pre_post_period_result.included_neuron_ids)
        if a_session_ctxt is not None:
            session_uid: str = a_session_ctxt.get_description_as_session_global_uid()
            active_neuron_UIDs: List[str] = [f"{session_uid}|{aclu}" for aclu in active_ACLUS]
            active_ACLUS = deepcopy(active_neuron_UIDs) ## Use the neuron UIDs instead of the simple aclus if they're available

        per_aclu_additional_properties_dict = {}
        n_epochs: int = len(a_pre_post_period_result.filter_epochs_df) ## total number of possible epochs
        
        has_epoch_participation: NDArray = np.vstack([(v.T[0].to_numpy() > 0.0) for v in a_pre_post_period_result.spike_counts_df_list]) # has_epoch_participation # .shape # (39, 20) - (n_epochs, n_aclus)
        n_participating_epochs: NDArray = has_epoch_participation.sum(axis=0) # .shape (N_ACLUS)
        assert len(active_ACLUS) == len(n_participating_epochs), f"len(a_pre_post_period_result.included_neuron_ids): {len(active_ACLUS)} != len(n_participating_epochs): {len(n_participating_epochs)}"
        n_participating_epochs_dict = dict(zip(active_ACLUS, n_participating_epochs))
        
        per_aclu_additional_properties_dict['n_epochs'] = n_epochs # .shape - (n_aclus,)

        ## Compute ratio of participating epochs
        ratio_participating_epochs = n_participating_epochs.astype(float) / float(n_epochs)
        per_aclu_additional_properties_dict['ratio_participating_epochs'] = ratio_participating_epochs # .shape - (n_aclus,)
        
        return n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict


    @function_attributes(short_name=None, tags=['pure','participation', 'additional'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-06 05:33', related_items=[])
    def compute_participation_stats(self, a_session_ctxt: Optional[IdentifyingContext]=None, should_update_self: bool=False, **kwargs):
        """ Uses: `a_pre_post_period_result.spike_counts_df_list`
        
        self.per_aclu_additional_properties_dict['n_participating_epochs'] # .shape - (n_aclus,)
        self.per_aclu_additional_properties_dict['n_participating_epochs_dict']
        
        """
        n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict =  self._perform_compute_participation_stats(a_pre_post_period_result=self, a_session_ctxt=a_session_ctxt, **kwargs)
        if should_update_self:
            if not hasattr(self, 'per_aclu_additional_properties_dict'):
                self.per_aclu_additional_properties_dict = {} ## Initialize to a new dict
            ## apply the update
            self.per_aclu_additional_properties_dict['has_epoch_participation'] = deepcopy(has_epoch_participation) # .shape - (n_epochs, n_aclus)
            self.per_aclu_additional_properties_dict['n_participating_epochs'] = deepcopy(n_participating_epochs) # .shape - (n_aclus,)
            self.per_aclu_additional_properties_dict['n_participating_epochs_dict'] = deepcopy(n_participating_epochs_dict) # Dict len n_aclus,
            self.per_aclu_additional_properties_dict['included_neuron_uids'] = np.array(deepcopy(list(n_participating_epochs_dict.keys()))) # .shape - (n_aclus,)
            for k, v in per_aclu_additional_properties_dict.items():
                self.per_aclu_additional_properties_dict[k] = deepcopy(v) 
            
        return n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids: NDArray[ND.Shape["N_CELLS"], Any]) -> 'SpikeRateTrends':
        """Returns object with neuron_ids equal to ids"""
        _obj = deepcopy(self)        
        if _obj.included_neuron_ids is not None:
            is_neuron_id_included = np.isin(_obj.included_neuron_ids, ids)
            _obj.included_neuron_ids = np.intersect1d(_obj.included_neuron_ids, ids)
            
        _obj.epoch_agg_inst_fr_list = _obj.epoch_agg_inst_fr_list[:, ids]
        _obj.cell_agg_inst_fr_list = _obj.cell_agg_inst_fr_list[ids]
        # _obj.epoch_agg_inst_fr_list = _obj.epoch_agg_inst_fr_list[:, ids]
        _obj.inst_fr_signals_list = _obj.inst_fr_signals_list[ids]

        flattened_spiketrains = deepcopy(self)
        return _obj
    





class SpikeAnalysisComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'burst_detection'
    _computationPrecidence = 4
    _is_global = False

    @function_attributes(short_name='spike_burst_detection', tags=['spikes','burst'],
                         input_requires=["computation_result.sess.spikes_df", "computation_result.sess.position"],
                         output_provides=["computation_result.computation_config['spike_analysis']", "computation_result.computed_data['burst_detection']"],
                         uses=['safe_pandas_get_group','pybursts.kleinberg'], used_by=[], creation_date='2023-09-12 17:27', related_items=[],
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
    
    

    @function_attributes(short_name='firing_rate_trends', tags=[''],
                         input_requires=["computation_result.sess.spikes_df", "computation_result.computed_data['pf2D']"], output_provides=["computation_result.computed_data['firing_rate_trends']"],
                         uses=[], used_by=[], creation_date='2023-08-31 00:00', related_items=[],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends']['pf_included_spikes_only']), is_global=False)
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
        sess_unit_specific_inst_spike_rate_values_df, sess_unit_specific_inst_spike_rate, sess_unit_split_spiketrains = SpikeRateTrends.compute_single_epoch_instantaneous_time_firing_rates(active_session_spikes_df, time_bin_size_seconds=time_bin_size_seconds, epoch_t_start=computation_result.sess.t_start, epoch_t_stop=computation_result.sess.t_stop)
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