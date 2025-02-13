from copy import deepcopy
import numpy as np
import pandas as pd

from pyphocorehelpers.indexing_helpers import build_pairwise_indicies
from scipy import stats # for compute_relative_entropy_divergence_overlap
from scipy.special import rel_entr # alternative for compute_relative_entropy_divergence_overlap


from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from neuropy.core.position import build_position_df_resampled_to_time_windows

# from neuropy.analyses.laps import _build_new_lap_and_intra_lap_intervals # for _perform_time_dependent_pf_sequential_surprise_computation

# For _perform_pf_dt_sequential_surprise
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

class TimeDependentPlacefieldSurpriseMode(ExtendedEnum):
    """for _perform_pf_dt_sequential_surprise """
    STATIC_METHOD_ONLY = "static_method_only"
    USING_EXTANT = "using_extant"
    BUILD_NEW = "build_new"

    @property
    def needs_build_new(self):
        return TimeDependentPlacefieldSurpriseMode.needs_build_newList()[self]

    @property
    def use_pf_dt_obj(self):
        return TimeDependentPlacefieldSurpriseMode.use_pf_dt_objList()[self]

    # Static properties
    @classmethod
    def use_pf_dt_objList(cls):
        return cls.build_member_value_dict([False, True, True])

    @classmethod
    def needs_build_newList(cls):
        return cls.build_member_value_dict([False, False, True])





class ExtendedStatsComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'extended_stats'
    _computationPrecidence = 3
    _is_global = False

    @function_attributes(short_name='extended_stats', tags=['position', 'resample', 'time_binned', 'statistics'], 
        input_requires=["computation_result.sess.position", "computation_result.computation_config.pf_params.time_bin_size"], 
        output_provides=["computation_result.computed_data['extended_stats']['time_binned_positioned_resampler']", "computation_result.computed_data['extended_stats']['time_binned_position_df']", "computation_result.computed_data['extended_stats']['time_binned_position_mean']", "computation_result.computed_data['extended_stats']['time_binned_position_covariance']"],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['time_binned_position_df']), is_global=False)
    def _perform_extended_statistics_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes extended statistics regarding firing rates and such from the various dataframes.
        
        Requires:
            computation_result.sess.position
            computation_result.computation_config.pf_params.time_bin_size
            
        Provides:
            computation_result.computed_data['extended_stats']
                ['extended_stats']['time_binned_positioned_resampler']
                ['extended_stats']['time_binned_position_df']
                ['extended_stats']['time_binned_position_mean']
                ['extended_stats']['time_binned_position_covariance']
                
        
        """
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.pf_params.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
        computation_result.computed_data['extended_stats'] = DynamicParameters.init_from_dict({
         'time_binned_positioned_resampler': time_binned_position_resampler, # this might be the unpicklable object? 
         'time_binned_position_df': time_binned_position_df,
         'time_binned_position_mean': time_binned_position_df.resample("1min").mean(), # 3 minutes
         'time_binned_position_covariance': time_binned_position_df.cov(min_periods=12)
        })
        """ 
        Access via ['extended_stats']['time_binned_position_df']
        Example:
            active_extended_stats = curr_active_pipeline.computation_results['maze1'].computed_data['extended_stats']
            time_binned_pos_df = active_extended_stats['time_binned_position_df']
            time_binned_pos_df
        """
        return computation_result
    


    @function_attributes(short_name='pf_dt_sequential_surprise', tags=['surprise', 'time_dependent_pf'], 
        uses=['compute_snapshot_relative_entropy_surprise_differences'], used_by=[], related_items=[], conforms_to=[],
        computation_precidence=9.01,
        input_requires=["computation_result.computed_data['firing_rate_trends']", "computation_result.computed_data['pf1D_dt']", "computation_result.sess.position", "computation_result.computation_config.pf_params.time_bin_size"], 
        output_provides=["computation_result.computed_data['extended_stats']['time_binned_positioned_resampler']", "computation_result.computed_data['extended_stats']['time_binned_position_df']", "computation_result.computed_data['extended_stats']['time_binned_position_mean']", "computation_result.computed_data['extended_stats']['time_binned_position_covariance']"],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['pf_dt_sequential_surprise'], np.sum(curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['pf_dt_sequential_surprise']['flat_relative_entropy_results'], axis=1),  np.sum(curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['pf_dt_sequential_surprise']['flat_jensen_shannon_distance_results'], axis=1)),
        # validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (np.sum(curr_active_pipeline.global_computation_results.computed_data['pf_dt_sequential_surprise']['flat_relative_entropy_results'], axis=1), np.sum(curr_active_pipeline.global_computation_results.computed_data['pf_dt_sequential_surprise']['flat_jensen_shannon_distance_results'], axis=1)),
        is_global=False)
    def _perform_time_dependent_pf_sequential_surprise_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes extended statistics regarding firing rates and such from the various dataframes.
        NOTE: 2022-12-14 - previously this version only did laps, but now it does the binned times for the entire epoch from ['firing_rate_trends']

        Requires:
            computed_data['firing_rate_trends']
            computed_data['pf1D_dt']
            computation_result.sess.position
            computation_result.computation_config.pf_params.time_bin_size
            
        Provides:
            computation_result.computed_data['extended_stats']
                ['extended_stats']['time_binned_positioned_resampler']
                ['extended_stats']['time_binned_position_df']
                ['extended_stats']['time_binned_position_mean']
                ['extended_stats']['time_binned_position_covariance']                
                
        
        """
        # use_extant_pf1D_dt_mode = TimeDependentPlacefieldSurpriseMode.STATIC_METHOD_ONLY
        use_extant_pf1D_dt_mode = TimeDependentPlacefieldSurpriseMode.USING_EXTANT # reuse the existing pf1D_dt

        # ==================================================================================================================== #
        # prev version using batch-snapshotting and the laps:
        # sess = computation_result.sess
        # sess, combined_records_list = _build_new_lap_and_intra_lap_intervals(sess) # from PendingNotebookCode

        # difference_snapshots = active_pf_1D_dt.batch_snapshotting(combined_records_list, reset_at_start=True, debug_print=debug_print)
        # # post_update_times, pf_overlap_results, flat_relative_entropy_results, flat_jensen_shannon_distance_results = compute_snapshot_relative_entropy_surprise_differences(difference_snapshots) # this fails, use the active_pf_1D_dt.historical_snapshots instead
        # post_update_times, pf_overlap_results, flat_relative_entropy_results, flat_jensen_shannon_distance_results = compute_snapshot_relative_entropy_surprise_differences(active_pf_1D_dt.historical_snapshots)


        # ==================================================================================================================== #
        # _perform_time_dependent_pf_sequential_surprise_computation - using time binning from computation_result.computed_data['firing_rate_trends']
        active_pf_1D_dt = computation_result.computed_data['pf1D_dt']
        
        ## Get the time-binning from `firing_rate_trends`:
        active_firing_rate_trends = computation_result.computed_data['firing_rate_trends']
        time_bin_size_seconds, pf_included_spikes_only = active_firing_rate_trends['time_bin_size_seconds'], active_firing_rate_trends['pf_included_spikes_only']

        active_time_binning_container, active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binning_container'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']
        # ZhangReconstructionImplementation._validate_time_binned_spike_rate_df(active_time_binning_container.centers, active_time_binned_unit_specific_binned_spike_counts)

        ## Use appropriate pf_1D_dt:
        active_session, pf_computation_config = computation_result.sess, computation_result.computation_config.pf_params
        active_session_spikes_df, active_pos, computation_config, included_epochs = active_session.spikes_df, active_session.position, pf_computation_config, pf_computation_config.computation_epochs
        
        ## Get existing `pf1D_dt`:
        if not use_extant_pf1D_dt_mode.needs_build_new:
            ## Get existing `pf1D_dt`:
            active_pf_1D_dt = computation_result.computed_data.pf1D_dt
        else:
            # NOTE: even in TimeDependentPlacefieldSurpriseMode.STATIC_METHOD_ONLY a PfND_TimeDependent object is used to access its properties for the Static Method (although it isn't modified)
            active_pf_1D_dt = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                                speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                                grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

        out_pair_indicies = build_pairwise_indicies(np.arange(active_time_binning_container.edge_info.num_bins))
        time_intervals = active_time_binning_container.edges[out_pair_indicies] # .shape # (4153, 2)

        ## Entirely independent computations for binned_times:
        if use_extant_pf1D_dt_mode.use_pf_dt_obj:
            active_pf_1D_dt.reset()

        if not use_extant_pf1D_dt_mode.use_pf_dt_obj:
            historical_snapshots = {} # build a dict<float:PlacefieldSnapshot>

        for start_t, end_t in time_intervals:
            
            ## Inline version that reuses active_pf_1D_dt directly:
            if use_extant_pf1D_dt_mode.use_pf_dt_obj:
                active_pf_1D_dt.update(end_t, should_snapshot=True) # use this because it correctly integrates over [0, end_t] instead of [start_t, end_t]
                # active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=True, should_snapshot=True)
                # historical_snapshots[float(end_t)] = active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=False)
            else:
                # Static version that calls PfND_TimeDependent.perform_time_range_computation(...) itself using just the computed variables of `active_pf_1D_dt`:
                historical_snapshots[float(end_t)] = PfND_TimeDependent.perform_time_range_computation(active_pf_1D_dt.all_time_filtered_spikes_df, active_pf_1D_dt.all_time_filtered_pos_df, position_srate=active_pf_1D_dt.position_srate,
                                                                            xbin=active_pf_1D_dt.xbin, ybin=active_pf_1D_dt.ybin,
                                                                            start_time=start_t, end_time=end_t,
                                                                            included_neuron_IDs=active_pf_1D_dt.included_neuron_IDs, active_computation_config=active_pf_1D_dt.config, override_smooth=active_pf_1D_dt.smooth)

        # {1.9991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x16c2b74fb20>, 2.4991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x168acfb3bb0>, ...}
        if use_extant_pf1D_dt_mode.use_pf_dt_obj:
            historical_snapshots = active_pf_1D_dt.historical_snapshots

        post_update_times, snapshot_differences_result_dict, flat_relative_entropy_results, flat_jensen_shannon_distance_results = compute_snapshot_relative_entropy_surprise_differences(historical_snapshots)
        relative_entropy_result_dicts_list = [a_val_dict['relative_entropy_result_dict'] for a_val_dict in snapshot_differences_result_dict]
        long_short_rel_entr_curves_list = [a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list] # [0].shape # (108, 63) = (n_neurons, n_xbins)
        short_long_rel_entr_curves_list = [a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]
        long_short_rel_entr_curves_frames = np.stack([a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
        short_long_rel_entr_curves_frames = np.stack([a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)

        
        # ==================================================================================================================== #
        ## Save Outputs:
        if 'extended_stats' not in computation_result.computed_data:
            computation_result.computed_data['extended_stats'] = DynamicParameters() # new 'extended_stats' dict
 
        # Here is where we use `DynamicParameters`
        computation_result.computed_data['extended_stats']['pf_dt_sequential_surprise'] = DynamicParameters.init_from_dict({
            'time_bin_size_seconds': time_bin_size_seconds,
            'historical_snapshots': historical_snapshots,
            'post_update_times': post_update_times,
            'snapshot_differences_result_dict': snapshot_differences_result_dict, 'time_intervals': time_intervals,
            'long_short_rel_entr_curves_frames': long_short_rel_entr_curves_frames, 'short_long_rel_entr_curves_frames': short_long_rel_entr_curves_frames,
            'flat_relative_entropy_results': flat_relative_entropy_results, 'flat_jensen_shannon_distance_results': flat_jensen_shannon_distance_results
        })
        """ 
        Access via ['extended_stats']['pf_dt_sequential_surprise']
        Example:
            active_extended_stats = curr_active_pipeline.computation_results['maze'].computed_data['extended_stats']
            active_relative_entropy_results = active_extended_stats['pf_dt_sequential_surprise']
            post_update_times = active_relative_entropy_results['post_update_times']
            snapshot_differences_result_dict = active_relative_entropy_results['snapshot_differences_result_dict']
            time_intervals = active_relative_entropy_results['time_intervals']
            long_short_rel_entr_curves_frames = active_relative_entropy_results['long_short_rel_entr_curves_frames']
            short_long_rel_entr_curves_frames = active_relative_entropy_results['short_long_rel_entr_curves_frames']
            flat_relative_entropy_results = active_relative_entropy_results['flat_relative_entropy_results']
            flat_jensen_shannon_distance_results = active_relative_entropy_results['flat_jensen_shannon_distance_results']
        """

        # computation_result.computed_data['extended_stats']['sequential_surprise'] = DynamicParameters.init_from_dict({
        #  'post_update_times': post_update_times,
        #  'pf_overlap_results': pf_overlap_results,
        #  'flat_relative_entropy_results': flat_relative_entropy_results,
        #  'flat_jensen_shannon_distance_results': flat_jensen_shannon_distance_results,
        #  'difference_snapshots': difference_snapshots
        # })
        """ 
        Access via ['extended_stats']['sequential_surprise']
        Example:
            active_extended_stats = curr_active_pipeline.computation_results['maze'].computed_data['extended_stats']
            sequential_surprise = active_extended_stats['sequential_surprise']
            post_update_times = sequential_surprise['post_update_times']
            pf_overlap_results = sequential_surprise['pf_overlap_results']
            flat_relative_entropy_results = sequential_surprise['flat_relative_entropy_results']
            difference_snapshots = sequential_surprise['difference_snapshots']
        """
        return computation_result
    






@function_attributes(short_name=None, tags=['surprise', 'snapshot', 'pfdt', 'relative_entropy'], input_requires=[], output_provides=[], uses=['compute_surprise_relative_entropy_divergence'], used_by=[], creation_date='2023-09-22 07:18', related_items=[])
def compute_snapshot_relative_entropy_surprise_differences(historical_snapshots_dict):
    """
    Computes the surprise between consecutive pairs of placefield snapshots extracted from a computed `active_pf_1D_dt`

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats import compute_snapshot_relative_entropy_surprise_differences
        pf_overlap_results, flat_relative_entropy_results = compute_snapshot_relative_entropy_surprise_differences(active_pf_1D_dt)


    """

    # Subfunctions _______________________________________________________________________________________________________ #
    def compute_surprise_relative_entropy_divergence(long_curve, short_curve):
        """ Pre 2023-03-10 Refactoring:
        Given two tuning maps, computes the surprise (in terms of the KL-divergence a.k.a. relative entropy) between the two
        Returns a dictionary containing the results in both directions

        TODO 2023-03-08 02:41: - [ ] Convert naming convention from long_, short_ to lhs_, rhs_ to be general
        TODO 2023-03-08 02:47: - [ ] Convert output dict to a dataclass

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats import compute_surprise_relative_entropy_divergence

        """
        long_short_rel_entr_curve = rel_entr(long_curve, short_curve)
        long_short_relative_entropy = sum(long_short_rel_entr_curve) 
        short_long_rel_entr_curve = rel_entr(short_curve, long_curve)
        short_long_relative_entropy = sum(short_long_rel_entr_curve)
        # Jensen-Shannon distance is an average of KL divergence:
        mixture_distribution = 0.5 * (long_curve + short_curve)
        jensen_shannon_distance = 0.5 * (sum(rel_entr(mixture_distribution, long_curve)) + sum(rel_entr(mixture_distribution, short_curve))) # is this right? I'm confused by sum(...)

        return dict(long_short_rel_entr_curve=long_short_rel_entr_curve, long_short_relative_entropy=long_short_relative_entropy, short_long_rel_entr_curve=short_long_rel_entr_curve, short_long_relative_entropy=short_long_relative_entropy,
                jensen_shannon_distance=jensen_shannon_distance)


    # Begin Function Body ________________________________________________________________________________________________ #
    # Lists with one entry per snapshot in historical_snapshots_dict
    pf_overlap_results = []
    flat_relative_entropy_results = []
    flat_jensen_shannon_distance_results = []

    n_snapshots = len(historical_snapshots_dict)
    snapshot_times = list(historical_snapshots_dict.keys())
    snapshots = list(historical_snapshots_dict.values())
    snapshot_indicies = np.arange(n_snapshots) # [0, 1, 2, 3, 4]

    post_update_times = snapshot_times[1:] # all but the first snapshot

    snapshot_pair_indicies = build_pairwise_indicies(snapshot_indicies) # [(0, 1), (1, 2), (2, 3), ... , (146, 147), (147, 148), (148, 149)]
    for earlier_snapshot_idx, later_snapshot_idx in snapshot_pair_indicies:
        ## Extract the two sequential snapshots for this period:
        earlier_snapshot, later_snapshot = snapshots[earlier_snapshot_idx], snapshots[later_snapshot_idx]
        earlier_snapshot_t, later_snapshot_t = snapshot_times[earlier_snapshot_idx], snapshot_times[later_snapshot_idx]

        ## Proof of concept, compute surprise between the two snapshots:
        # relative_entropy_overlap_dict, relative_entropy_overlap_scalars_df = compute_relative_entropy_divergence_overlap(earlier_snapshot, later_snapshot, debug_print=False)
        # print(earlier_snapshot['occupancy_weighted_tuning_maps_matrix'].shape) # (108, 63)
        # print(later_snapshot['occupancy_weighted_tuning_maps_matrix'].shape) # (108, 63)
        # relative_entropy_result_dict = compute_surprise_relative_entropy_divergence(earlier_snapshot['occupancy_weighted_tuning_maps_matrix'], later_snapshot['occupancy_weighted_tuning_maps_matrix'])
        relative_entropy_result_dict = compute_surprise_relative_entropy_divergence(earlier_snapshot.occupancy_weighted_tuning_maps_matrix, later_snapshot.occupancy_weighted_tuning_maps_matrix)

        # 'long_short_relative_entropy'

        # aclu_keys = [k for k,v in relative_entropy_result_dict.items() if v is not None] # len(aclu_keys) # 101
        # short_long_rel_entr_curves = np.vstack([v['short_long_rel_entr_curve'] for k,v in relative_entropy_result_dict.items() if v is not None])

        # np.vstack(relative_entropy_result_dict['short_long_relative_entropy'])


        # short_long_rel_entr_curves # .shape # (101, 63)
        # print(f"{relative_entropy_result_dict['short_long_rel_entr_curve'].shape}") # (108, 63)
        # print(f"{relative_entropy_result_dict['short_long_relative_entropy'].shape}") # (63,)

        flat_relative_entropy_results.append(relative_entropy_result_dict['short_long_relative_entropy'])
        flat_jensen_shannon_distance_results.append(relative_entropy_result_dict['jensen_shannon_distance'])

        pf_overlap_results.append({'t': (earlier_snapshot_t, later_snapshot_t),
                                   'snapshots': (earlier_snapshot, later_snapshot),
                                   'relative_entropy_result_dict': relative_entropy_result_dict,
            # 'short_long_rel_entr_curves': short_long_rel_entr_curves,
            # 'relative_entropy_overlap_scalars_df': relative_entropy_overlap_scalars_df,        
        })

    # flatten the relevent results:
    post_update_times = np.array(post_update_times)
    flat_jensen_shannon_distance_results = np.vstack(flat_jensen_shannon_distance_results) # flatten the list
    flat_relative_entropy_results = np.vstack(flat_relative_entropy_results) # flatten the list
    
    # relative_entropy_result_dicts_list = [a_val_dict['relative_entropy_result_dict'] for a_val_dict in pf_overlap_results]
    # # long_short_rel_entr_curves_list = [a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list] # [0].shape # (108, 63) = (n_neurons, n_xbins)
    # # short_long_rel_entr_curves_list = [a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]
    # long_short_rel_entr_curves_frames = np.stack([a_val_dict['long_short_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
    # short_long_rel_entr_curves_frames = np.stack([a_val_dict['short_long_rel_entr_curve'] for a_val_dict in relative_entropy_result_dicts_list]) # build a 3D array (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)

    return post_update_times, pf_overlap_results, flat_relative_entropy_results, flat_jensen_shannon_distance_results