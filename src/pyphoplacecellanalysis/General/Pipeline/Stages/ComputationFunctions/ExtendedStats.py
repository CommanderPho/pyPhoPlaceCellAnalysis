import numpy as np
import pandas as pd

from pyphocorehelpers.indexing_helpers import build_pairwise_indicies
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import compute_relative_entropy_divergence_overlap
from scipy import stats # for compute_relative_entropy_divergence_overlap
from scipy.special import rel_entr # alternative for compute_relative_entropy_divergence_overlap


from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import build_position_df_resampled_to_time_windows, build_position_df_time_window_idx


from neuropy.analyses.laps import _build_new_lap_and_intra_lap_intervals # for _perform_time_dependent_pf_sequential_surprise_computation


class ExtendedStatsComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'extended_stats'
    _computationPrecidence = 3
    _is_global = False


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
         'time_binned_positioned_resampler': time_binned_position_resampler,
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
    
    

    def _perform_time_dependent_pf_sequential_surprise_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes extended statistics regarding firing rates and such from the various dataframes.
        
        Requires:
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
        # _perform_time_dependent_pf_sequential_surprise_computation
        active_pf_1D_dt = computation_result.computed_data['pf1D_dt']
        # active_pf_2D_dt = computation_result.computed_data['pf2D_dt']

        sess = computation_result.sess
        sess, combined_records_list = _build_new_lap_and_intra_lap_intervals(sess) # from PendingNotebookCode

        difference_snapshots = active_pf_1D_dt.batch_snapshotting(combined_records_list, reset_at_start=True, debug_print=debug_print)
        post_update_times, pf_overlap_results, flat_relative_entropy_results, flat_jensen_shannon_distance_results = compute_snapshot_differences(active_pf_1D_dt)
        flat_jensen_shannon_distance_results = np.vstack(flat_jensen_shannon_distance_results) # flatten the list
        flat_relative_entropy_results = np.vstack(flat_relative_entropy_results) # flatten the list

        if 'extended_stats' not in computation_result.computed_data:
            computation_result.computed_data['extended_stats'] = DynamicParameters() # new 'extended_stats' dict
 
        computation_result.computed_data['extended_stats']['sequential_surprise'] = DynamicParameters.init_from_dict({
         'post_update_times': np.array(post_update_times),
         'pf_overlap_results': pf_overlap_results,
         'flat_relative_entropy_results': flat_relative_entropy_results,
         'flat_jensen_shannon_distance_results': flat_jensen_shannon_distance_results,
         'difference_snapshots': difference_snapshots
        })
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
    



def compute_surprise_relative_entropy_divergence(long_curve, short_curve):
    """
    Given two tuning maps, computes the surprise (in terms of the KL-divergence a.k.a. relative entropy) between the two
    Returns a dictionary containing the results in both directions
    """
    long_short_rel_entr_curve = rel_entr(long_curve, short_curve)
    long_short_relative_entropy = sum(long_short_rel_entr_curve) 
    short_long_rel_entr_curve = rel_entr(short_curve, long_curve)
    short_long_relative_entropy = sum(short_long_rel_entr_curve)
    # Jensen-Shannon distance is an average of KL divergence:
    mixture_distribution = 0.5 * (long_curve + short_curve)
    jensen_shannon_distance = 0.5 * (sum(rel_entr(mixture_distribution, long_curve)) + sum(rel_entr(mixture_distribution, short_curve)))


    return dict(long_short_rel_entr_curve=long_short_rel_entr_curve, long_short_relative_entropy=long_short_relative_entropy, short_long_rel_entr_curve=short_long_rel_entr_curve, short_long_relative_entropy=short_long_relative_entropy,
            jensen_shannon_distance=jensen_shannon_distance)


def compute_snapshot_differences(active_pf_1D_dt):
    """
    Computes the surprise between consecutive pairs of placefield snapshots extracted from a computed `active_pf_1D_dt`

    Usage:

        pf_overlap_results, flat_relative_entropy_results = compute_snapshot_differences(active_pf_1D_dt)


    """
    pf_overlap_results = []
    flat_relative_entropy_results = []
    flat_jensen_shannon_distance_results = []

    n_snapshots = len(active_pf_1D_dt.historical_snapshots)
    snapshot_times = list(active_pf_1D_dt.historical_snapshots.keys())
    snapshots = list(active_pf_1D_dt.historical_snapshots.values())
    snapshot_indicies = np.arange(n_snapshots) # [0, 1, 2, 3, 4]

    post_update_times = snapshot_times[1:] # all but the first snapshot

    snapshot_pair_indicies = build_pairwise_indicies(snapshot_indicies) # [(0, 1), (1, 2), (2, 3), ... , (146, 147), (147, 148), (148, 149)]
    for earlier_snapshot_idx, later_snapshot_idx in snapshot_pair_indicies:
        ## Extract the two sequential snapshots for this period:
        # earlier_snapshot, later_snapshot = active_pf_1D_dt.historical_snapshots[earlier_snapshot_idx], active_pf_1D_dt.historical_snapshots[later_snapshot_idx]
        earlier_snapshot, later_snapshot = snapshots[earlier_snapshot_idx], snapshots[later_snapshot_idx]
        earlier_snapshot_t, later_snapshot_t = snapshot_times[earlier_snapshot_idx], snapshot_times[later_snapshot_idx]

        ## Proof of concept, comute surprise between the two snapshots:
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
        
    
    return post_update_times, pf_overlap_results, flat_relative_entropy_results, flat_jensen_shannon_distance_results