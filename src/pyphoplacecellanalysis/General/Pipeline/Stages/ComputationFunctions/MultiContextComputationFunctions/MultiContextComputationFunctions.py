from neuropy.core.epoch import NamedTimerange
import numpy as np
import pandas as pd
from functools import wraps

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_pf_dt_sequential_surprise


def _wrap_multi_context_computation_function(global_comp_fcn):
    """ captures global_comp_fcn and unwraps its arguments: owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False """
    @wraps(global_comp_fcn) # @wraps ensures that the functions name, docs, etc are accessible in the wrapped version of the function.
    def _(x, **kwargs):
        assert len(x) > 4, f"looks like it ensures we have more than four (at least 5) positional arguments provided. {x}"
        x[1] = global_comp_fcn(*x, **kwargs) # update global_computation_results
        return x
    return _


class MultiContextComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'multi_context'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='PBE_stats', tags=['PBE', 'stats'], input_requires=[], output_provides=[], uses=['_perform_PBE_stats'], used_by=[], creation_date='2023-09-12 17:37', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['PBE_stats_analyses'], curr_active_pipeline.global_computation_results.computed_data['pbe_analyses_result_df']), is_global=True)
    def _perform_PBE_stats_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ 
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['PBE_stats_analyses']
                ['PBE_stats_analyses']['pbe_analyses_result_df']
                ['PBE_stats_analyses']['all_epochs_info']
        
        """
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
        
        pbe_analyses_result_df, all_epochs_info = _perform_PBE_stats(owning_pipeline_reference, include_includelist=include_includelist, debug_print=debug_print)

        global_computation_results.computed_data['PBE_stats_analyses'] = DynamicParameters.init_from_dict({
            'pbe_analyses_result_df': pbe_analyses_result_df,
            'all_epochs_info': all_epochs_info,
        })
        return global_computation_results


# ==================================================================================================================== #
# PBE Stats                                                                                                            #
# ==================================================================================================================== #
def _perform_PBE_stats(owning_pipeline_reference, include_includelist=None, debug_print = False):
    """ # Analyze PBEs by looping through the filtered epochs:
        This whole implementation seems silly and inefficient        
        Can't I use .agg(['count', 'mean']) or something? 
        
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _perform_PBE_stats
        pbe_analyses_result_df, [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] = _perform_PBE_stats(curr_active_pipeline, debug_print=False) # all_epochs_n_pbes: [206, 31, 237], all_epochs_mean_pbe_durations: [0.2209951456310722, 0.23900000000001073, 0.22335021097046923], all_epochs_cummulative_pbe_durations: [45.52500000000087, 7.409000000000333, 52.934000000001205], all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, 1910.1600048116618]
        pbe_analyses_result_df

    """
    if include_includelist is None:
        include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

    all_epochs_labels = []
    all_epochs_total_durations = []
    all_epochs_n_pbes = []
    all_epochs_pbe_duration_lists = []
    all_epochs_cummulative_pbe_durations = []
    all_epochs_mean_pbe_durations = []
    all_epochs_full_pbe_spiketrain_lists = []
    all_epochs_pbe_num_spikes_lists = []
    all_epochs_intra_pbe_interval_lists = []
    
    for (name, filtered_sess) in owning_pipeline_reference.filtered_sessions.items():
        if name in include_includelist:
            # interested in analyzing both the filtered_sess.pbe and the filtered_sess.spikes_df (as they relate to the PBEs)
            all_epochs_labels.append(name)
            curr_named_time_range = owning_pipeline_reference.sess.epochs.get_named_timerange(name) # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
            
            if not np.isscalar(curr_named_time_range.duration):
                # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
                curr_named_time_range = NamedTimerange(name='maze', start_end_times=[owning_pipeline_reference.sess.epochs['maze1'][0], owning_pipeline_reference.sess.epochs['maze2'][1]])
            
            curr_epoch_duration = curr_named_time_range.duration
            all_epochs_total_durations.append(curr_epoch_duration) # TODO: this should be in seconds (or at least the same units as the PBE durations)... actually this might be right.
            # Computes the intervals between each PBE:
            curr_intra_pbe_intervals = filtered_sess.pbe.starts[1:] - filtered_sess.pbe.stops[:-1]
            all_epochs_intra_pbe_interval_lists.append(curr_intra_pbe_intervals)
            all_epochs_n_pbes.append(filtered_sess.pbe.n_epochs)
            all_epochs_pbe_duration_lists.append(filtered_sess.pbe.durations)
            all_epochs_cummulative_pbe_durations.append(np.sum(filtered_sess.pbe.durations))
            all_epochs_mean_pbe_durations.append(np.nanmean(filtered_sess.pbe.durations))
            # filtered_sess.spikes_df.PBE_id
            curr_pbe_only_spikes_df = filtered_sess.spikes_df[filtered_sess.spikes_df.PBE_id > -1].copy()
            unique_PBE_ids = np.unique(curr_pbe_only_spikes_df['PBE_id'])
            flat_PBE_ids = [int(id) for id in unique_PBE_ids]
            num_unique_PBE_ids = len(flat_PBE_ids)
            # groups the spikes_df by PBEs:
            curr_pbe_grouped_spikes_df = curr_pbe_only_spikes_df.groupby(['PBE_id'])
            curr_spiketrains = list()
            curr_PBE_spiketrain_num_spikes = list()
            for i in np.arange(num_unique_PBE_ids):
                curr_PBE_id = flat_PBE_ids[i] # actual cell ID
                #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
                curr_PBE_dataframe = curr_pbe_grouped_spikes_df.get_group(curr_PBE_id)
                curr_PBE_num_spikes = np.shape(curr_PBE_dataframe)[0] # the number of spikes in this PBE
                curr_PBE_spiketrain_num_spikes.append(curr_PBE_num_spikes)
                curr_spiketrains.append(curr_PBE_dataframe['t'].to_numpy())

            curr_PBE_spiketrain_num_spikes = np.array(curr_PBE_spiketrain_num_spikes)
            all_epochs_pbe_num_spikes_lists.append(curr_PBE_spiketrain_num_spikes)
            curr_spiketrains = np.array(curr_spiketrains, dtype='object')
            all_epochs_full_pbe_spiketrain_lists.append(curr_spiketrains)
            if debug_print:
                print(f'name: {name}, filtered_sess.pbe: {filtered_sess.pbe}')

    if debug_print:
        print(f'all_epochs_n_pbes: {all_epochs_n_pbes}, all_epochs_mean_pbe_durations: {all_epochs_mean_pbe_durations}, all_epochs_cummulative_pbe_durations: {all_epochs_cummulative_pbe_durations}, all_epochs_total_durations: {all_epochs_total_durations}')
        # all_epochs_n_pbes: [3152, 561, 1847, 832, 4566], all_epochs_mean_pbe_durations: [0.19560881979695527, 0.22129233511594312, 0.19185056848946497, 0.2333112980769119, 0.1987152869032212]

    all_epochs_pbe_occurance_rate = [(float(all_epochs_total_durations[i]) / float(all_epochs_n_pbes[i])) for i in np.arange(len(all_epochs_n_pbes))]
    all_epochs_pbe_percent_duration = [(float(all_epochs_total_durations[i]) / float(all_epochs_cummulative_pbe_durations[i])) for i in np.arange(len(all_epochs_n_pbes))]    
    all_epoch_mean_num_pbe_spikes = [np.nanmean(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [3151, 561, 1847, 831, 4563]
    all_epoch_std_num_pbe_spikes = [np.nanstd(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [11.638970035733648, 15.013817202645336, 15.5123897729991, 15.113395025612247, 11.473087401691878]
    # [20.429704855601397, 27.338680926916222, 23.748781808337846, 25.673886883273166, 20.38614946307254]
    # Build the final output result dataframe:
    pbe_analyses_result_df = pd.DataFrame({'n_pbes':all_epochs_n_pbes, 'mean_pbe_durations': all_epochs_mean_pbe_durations, 'cummulative_pbe_durations':all_epochs_cummulative_pbe_durations, 'epoch_total_duration':all_epochs_total_durations,
                'pbe_occurance_rate':all_epochs_pbe_occurance_rate, 'pbe_percent_duration':all_epochs_pbe_percent_duration,
                'mean_num_pbe_spikes':all_epoch_mean_num_pbe_spikes, 'stddev_num_pbe_spikes':all_epoch_std_num_pbe_spikes}, index=all_epochs_labels)
    # temporary: this isn't how the returns work for other computation functions:
    all_epochs_info = [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] # list version
    # all_epochs_info = {'all_epochs_full_pbe_spiketrain_lists':all_epochs_full_pbe_spiketrain_lists, 'all_epochs_pbe_num_spikes_lists':all_epochs_pbe_num_spikes_lists, 'all_epochs_intra_pbe_interval_lists':all_epochs_intra_pbe_interval_lists} # dict version
    return pbe_analyses_result_df, all_epochs_info

