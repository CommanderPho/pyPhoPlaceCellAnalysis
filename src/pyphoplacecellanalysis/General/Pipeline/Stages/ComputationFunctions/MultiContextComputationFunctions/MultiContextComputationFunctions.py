import numpy as np
import pandas as pd

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_relative_entropy_analyses


def _wrap_multi_context_computation_function(global_comp_fcn):
    """ captures global_comp_fcn and unwraps its arguments: owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False """
    def _(x):
        assert len(x) > 4, f"{x}"
        x[1] = global_comp_fcn(*x) # update global_computation_results
        return x
    return _


class MultiContextComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'multi_context'
    _computationPrecidence = 1000
    _is_global = True


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


    # def _perform_relative_entropy_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
    #     """ NOTE: 2022-12-14 - this mirrors the non-global version at `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_time_dependent_pf_sequential_surprise_computation` that I just modified except it only uses the global epoch.
        
    #     Requires:
    #         ['firing_rate_trends']
    #         pf1D_dt (or it can build a new one)
            
    #     Provides:
    #         computation_result.computed_data['relative_entropy_analyses']
    #             ['relative_entropy_analyses']['short_long_neurons_diff']
    #             ['relative_entropy_analyses']['poly_overlap_df']
        
    #     """
    #     # use_extant_pf1D_dt_mode = TimeDependentPlacefieldSurpriseMode.STATIC_METHOD_ONLY
    #     use_extant_pf1D_dt_mode = TimeDependentPlacefieldSurpriseMode.USING_EXTANT # reuse the existing pf1D_dt

    #     if include_includelist is None:
    #         include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

    #     # Epoch dataframe stuff:
    #     global_epoch_name = include_includelist[-1] # 'maze_PYR'
        
    #     computation_result = computation_results[global_epoch_name]
    #     global_results_data = computation_result['computed_data']

    #     ## Get the time-binning from `firing_rate_trends`:
    #     active_firing_rate_trends = global_results_data['firing_rate_trends']
    #     time_bin_size_seconds, all_session_spikes, pf_included_spikes_only = active_firing_rate_trends['time_bin_size_seconds'], active_firing_rate_trends['all_session_spikes'], active_firing_rate_trends['pf_included_spikes_only']

    #     active_time_binning_container, active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binning_container'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']
    #     ZhangReconstructionImplementation._validate_time_binned_spike_rate_df(active_time_binning_container.centers, active_time_binned_unit_specific_binned_spike_counts)

    #     ## Use appropriate pf_1D_dt:
    #     active_session, pf_computation_config = computation_result.sess, computation_result.computation_config.pf_params
    #     active_session_spikes_df, active_pos, computation_config, included_epochs = active_session.spikes_df, active_session.position, pf_computation_config, pf_computation_config.computation_epochs
        
    #     ## Get existing `pf1D_dt`:
    #     if not use_extant_pf1D_dt_mode.needs_build_new:
    #         ## Get existing `pf1D_dt`:
    #         active_pf_1D_dt = global_results_data.pf1D_dt
    #     else:
    #         # note even in TimeDependentPlacefieldSurpriseMode.STATIC_METHOD_ONLY a PfND_TimeDependent object is used to access its properties for the Static Method (although it isn't modified)
    #         active_pf_1D_dt = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
    #                                             speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
    #                                             grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

    #     out_pair_indicies = build_pairwise_indicies(np.arange(active_time_binning_container.edge_info.num_bins))
    #     time_intervals = active_time_binning_container.edges[out_pair_indicies] # .shape # (4153, 2)
    #     # time_intervals

    #     ## Entirely independent computations for binned_times:
    #     if use_extant_pf1D_dt_mode.use_pf_dt_obj:
    #         active_pf_1D_dt.reset()

    #     out_list_t = []
    #     out_list = []
    #     for start_t, end_t in time_intervals:
    #         out_list_t.append(end_t)
            
    #         ## Inline version that reuses active_pf_1D_dt directly:
    #         if use_extant_pf1D_dt_mode.use_pf_dt_obj:
    #             out_list.append(active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=False))
    #         else:
    #             # Static version that calls PfND_TimeDependent.perform_time_range_computation(...) itself using just the computed variables of `active_pf_1D_dt`:
    #             out_list.append(PfND_TimeDependent.perform_time_range_computation(active_pf_1D_dt.all_time_filtered_spikes_df, active_pf_1D_dt.all_time_filtered_pos_df, position_srate=active_pf_1D_dt.position_srate,
    #                                                                         xbin=active_pf_1D_dt.xbin, ybin=active_pf_1D_dt.ybin,
    #                                                                         start_time=start_t, end_time=end_t,
    #                                                                         included_neuron_IDs=active_pf_1D_dt.included_neuron_IDs, active_computation_config=active_pf_1D_dt.config, override_smooth=active_pf_1D_dt.smooth))

    #     # out_list # len(out_list) # 4153
    #     out_list_t = np.array(out_list_t)
    #     historical_snapshots = {float(t):v for t, v in zip(out_list_t, out_list)} # build a dict<float:PlacefieldSnapshot>
    #     # {1.9991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x16c2b74fb20>, 2.4991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x168acfb3bb0>, ...}

    #     post_update_times, snapshot_differences_result_dict, flat_relative_entropy_results, flat_jensen_shannon_distance_results = compute_snapshot_relative_entropy_surprise_differences(historical_snapshots)
    #     relative_entropy_result_dicts_list = [a_val_dict['relative_entropy_result_dict'] for a_val_dict in snapshot_differences_result_dict]
    #     long_short_rel_entr_curves_list = [a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list] # [0].shape # (108, 63) = (n_neurons, n_xbins)
    #     short_long_rel_entr_curves_list = [a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]
    #     long_short_rel_entr_curves_frames = np.stack([a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
    #     short_long_rel_entr_curves_frames = np.stack([a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)

    #     global_computation_results.computed_data['relative_entropy_analyses'] = DynamicParameters.init_from_dict({
    #         'time_bin_size_seconds': time_bin_size_seconds,
    #         'historical_snapshots': historical_snapshots,
    #         'post_update_times': post_update_times,
    #         'snapshot_differences_result_dict': snapshot_differences_result_dict, 'time_intervals': time_intervals,
    #         'long_short_rel_entr_curves_frames': long_short_rel_entr_curves_frames, 'short_long_rel_entr_curves_frames': short_long_rel_entr_curves_frames,
    #         'flat_relative_entropy_results': flat_relative_entropy_results, 'flat_jensen_shannon_distance_results': flat_jensen_shannon_distance_results
    #     })
    #     return global_computation_results



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

