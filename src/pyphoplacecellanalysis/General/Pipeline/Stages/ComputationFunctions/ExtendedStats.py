import numpy as np
import pandas as pd
import itertools

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import build_position_df_resampled_to_time_windows


class ExtendedStatsComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'extended_stats'
    _computationPrecidence = 3
    
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
    
    
    def _perform_placefield_overlap_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes the pairwise overlap between every pair of placefields. 
        
        TODO: Move to PlacefieldDensityAnalysisComputationFunctions
        
        
        Requires:
            pf2D_Decoder
            
        Provides:
            ['placefield_overlap']
        
        """
        """ all_pairwise_neuron_IDXs_combinations: (np.shape: (903, 2))
        array([[ 0,  1],
            [ 0,  2],
            [ 0,  3],
            ...,
            [40, 41],
            [40, 42],
            [41, 42]])
        """
        all_pairwise_neuron_IDs_combinations = np.array(list(itertools.combinations(computation_result.computed_data['pf2D_Decoder'].neuron_IDs, 2)))
        list_of_unit_pfs = [computation_result.computed_data['pf2D_Decoder'].pf.ratemap.normalized_tuning_curves[i,:,:] for i in computation_result.computed_data['pf2D_Decoder'].neuron_IDXs]
        all_pairwise_pfs_combinations = np.array(list(itertools.combinations(list_of_unit_pfs, 2)))
        # np.shape(all_pairwise_pfs_combinations) # (903, 2, 63, 63)
        all_pairwise_overlaps = np.squeeze(np.prod(all_pairwise_pfs_combinations, axis=1)) # multiply over the dimension containing '2' (multiply each pair of pfs).
        # np.shape(all_pairwise_overlaps) # (903, 63, 63)
        total_pairwise_overlaps = np.sum(all_pairwise_overlaps, axis=(1, 2)) # sum over all positions, finding the total amount of overlap for each pair
        # np.shape(total_pairwise_overlaps) # (903,)

        # np.max(total_pairwise_overlaps) # 31.066909225698513
        # np.min(total_pairwise_overlaps) # 1.1385978466010813e-07

        # Sort the pairs by their total overlap to potentially elminate redundant pairs:
        pairwise_overlap_sort_order = np.flip(np.argsort(total_pairwise_overlaps))

        # Sort the returned quantities:
        all_pairwise_neuron_IDs_combinations = all_pairwise_neuron_IDs_combinations[pairwise_overlap_sort_order,:] # get the identities of the maximally overlapping placefields
        total_pairwise_overlaps = total_pairwise_overlaps[pairwise_overlap_sort_order]
        all_pairwise_overlaps = all_pairwise_overlaps[pairwise_overlap_sort_order,:,:]

        computation_result.computed_data['placefield_overlap'] = DynamicParameters.init_from_dict({
         'all_pairwise_neuron_IDs_combinations': all_pairwise_neuron_IDs_combinations,
         'total_pairwise_overlaps': total_pairwise_overlaps,
         'all_pairwise_overlaps': all_pairwise_overlaps,
        })
        """ 
        Access via ['placefield_overlap']['all_pairwise_overlaps']
        Example:
            active_pf_overlap_results = curr_active_pipeline.computation_results[active_config_name].computed_data['placefield_overlap']
            all_pairwise_neuron_IDs_combinations = active_pf_overlap_results['all_pairwise_neuron_IDs_combinations']
            total_pairwise_overlaps = active_pf_overlap_results['total_pairwise_overlaps']
            all_pairwise_overlaps = active_pf_overlap_results['all_pairwise_overlaps']
            all_pairwise_overlaps
        """
        return computation_result
    
    

        
    