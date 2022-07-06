import sys
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


"""-------------- Specific Computation Functions to be registered --------------"""

class PlacefieldDensityAnalysisComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """Performs analyeses related to placefield densities and overlap across spactial bins. Includes analyses for Eloy from 07-2022. """
    _computationPrecidence = 2 # must be done after PlacefieldComputations and DefaultComputationFunctions

    def _perform_velocity_vs_pf_density_computation(computation_result: ComputationResult, debug_print=False):
            """ Builds the analysis to test Eloy's Pf-Density/Velocity Hypothesis for 2D Placefields
            
            Requires:
                computed_data['pf1D']
                computed_data['pf2D']
                computed_data['pf2D_TwoStepDecoder']['avg_speed_per_pos']
                
            Provides:
                computed_data['EloyAnalysis']
                computed_data['EloyAnalysis']['pdf_normalized_pf_1D']: 
                computed_data['EloyAnalysis']['pf_overlapDensity_1D']: 
                computed_data['EloyAnalysis']['avg_1D_speed_per_pos']: 
                computed_data['EloyAnalysis']['avg_1D_speed_sort_idxs']: 
                computed_data['EloyAnalysis']['sorted_1D_avg_speed_per_pos']: 
                computed_data['EloyAnalysis']['sorted_PFoverlapDensity_1D']: 
                
                
                computed_data['EloyAnalysis']['pdf_normalized_pf_2D']: 
                computed_data['EloyAnalysis']['pf_overlapDensity_2D']: 
                computed_data['EloyAnalysis']['avg_2D_speed_per_pos']:
                computed_data['EloyAnalysis']['avg_2D_speed_sort_idxs']: 
                computed_data['EloyAnalysis']['sorted_avg_2D_speed_per_pos']: 
                computed_data['EloyAnalysis']['sorted_PFoverlapDensity_2D']: 
                
            """
            # Disable 1D:
            # pdf_normalized_pf_1D = None
            # pf_overlapDensity_1D = None
            # sorted_1D_avg_speed_per_pos = None
            # sorted_PFoverlapDensity_1D = None

            active_pf_1D = computation_result.computed_data['pf1D']
            active_pf_2D = computation_result.computed_data['pf2D']
            
            pdf_normalized_pf_1D = active_pf_1D.ratemap.pdf_normalized_tuning_curves
            pdf_normalized_pf_2D = active_pf_2D.ratemap.pdf_normalized_tuning_curves

            if debug_print:
                print(f'np.shape(pdf_normalized_pf_1D): {np.shape(pdf_normalized_pf_1D)}') # np.shape(_test_1D_AOC_normalized_pdf) # (39, 59)
                print(f'np.shape(pdf_normalized_pf_2D): {np.shape(pdf_normalized_pf_2D)}') # np.shape(pdf_normalized_pf_2D) # (39, 59, 21)

            ## Compute the PFoverlapDensity by summing over all cells:
            pf_overlapDensity_1D = np.sum(pdf_normalized_pf_1D, 0) # should be same size as positions
            pf_overlapDensity_2D = np.sum(pdf_normalized_pf_2D, 0) # should be same size as positions
            if debug_print:
                print(f'pf_overlapDensity_1D.shape: {pf_overlapDensity_1D.shape}') # (59, 21)
                print(f'pf_overlapDensity_2D.shape: {pf_overlapDensity_2D.shape}') # (39, 59, 21)

            ## Renormalize by dividing by the number of placefields (i)
            pf_overlapDensity_1D = pf_overlapDensity_1D / float(active_pf_1D.ratemap.n_neurons)
            pf_overlapDensity_2D = pf_overlapDensity_2D / float(active_pf_2D.ratemap.n_neurons)

            ## Order the bins in a flat array by ascending speed values:

            ## 1D: Average velocity per position bin:
            
                # TODO: should be xbin_centers or xbin?
                    # active_pf_1D.xbin
                    # avg_speed_per_pos = _compute_avg_speed_at_each_position_bin(computation_result.sess.position.to_dataframe(), computation_result.computation_config, prev_one_step_bayesian_decoder.xbin_centers, None, debug_print=debug_print)
                    # avg_1D_speed_per_pos = _compute_avg_speed_at_each_position_bin(active_pf_1D.filtered_pos_df, active_one_step_decoder.xbin_centers, None, debug_print=False)
                    # avg_1D_speed_per_pos = _compute_avg_1D_speed_at_each_position_bin(active_pf_1D.filtered_pos_df, active_pf_1D.xbin, debug_print=True)
            ## Hardcoded-implementation:
            position_bin_dependent_specific_average_velocities = active_pf_1D.filtered_pos_df.groupby(['binned_x'])['speed'].agg([np.nansum, np.nanmean, np.nanmin, np.nanmax]).reset_index()
            avg_1D_speed_per_pos = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy()
            if debug_print:
                print(f'avg_1D_speed_per_pos.shape: {np.shape(avg_1D_speed_per_pos)}') # avg_speed_per_pos_1D.shape # (64,)
            
            avg_1D_speed_sort_idxs = np.argsort(avg_1D_speed_per_pos, axis=None) # axis=None means the array is flattened before sorting
            ## Apply the same ordering to the PFoverlapDensities
            sorted_1D_avg_speed_per_pos = avg_1D_speed_per_pos.flat[avg_1D_speed_sort_idxs]
            sorted_PFoverlapDensity_1D = pf_overlapDensity_1D.flat[avg_1D_speed_sort_idxs]
            if debug_print:
                print(f'avg_1D_speed_sort_idxs.shape: {np.shape(avg_1D_speed_sort_idxs)}')
                print(f'sorted_1D_avg_speed_per_pos.shape: {np.shape(sorted_1D_avg_speed_per_pos)}')
                print(f'sorted_PFoverlapDensity_1D.shape: {np.shape(sorted_PFoverlapDensity_1D)}')
                
    
            ## 2D: Average velocity per position bin:
            avg_2D_speed_per_pos = computation_result.computed_data['pf2D_TwoStepDecoder']['avg_speed_per_pos']
            if debug_print:
                print(f'avg_2D_speed_per_pos.shape: {avg_2D_speed_per_pos.shape}') # (59, 21)
            avg_2D_speed_sort_idxs = np.argsort(avg_2D_speed_per_pos, axis=None) # axis=None means the array is flattened before sorting
            ## Apply the same ordering to the PFoverlapDensities
            sorted_avg_2D_speed_per_pos = avg_2D_speed_per_pos.flat[avg_2D_speed_sort_idxs]
            sorted_PFoverlapDensity_2D = pf_overlapDensity_2D.flat[avg_2D_speed_sort_idxs]
            # sorted_avg_speed_per_pos            

            computation_result.computed_data['EloyAnalysis'] = DynamicParameters.init_from_dict({'pdf_normalized_pf_1D': pdf_normalized_pf_1D, 'pdf_normalized_pf_2D': pdf_normalized_pf_2D,
                                                                                                 'pf_overlapDensity_1D': pf_overlapDensity_1D, 'pf_overlapDensity_2D': pf_overlapDensity_2D,
                                                                                                 'avg_1D_speed_per_pos': avg_1D_speed_per_pos, 'avg_2D_speed_per_pos': avg_2D_speed_per_pos,
                                                                                                 'avg_1D_speed_sort_idxs': avg_1D_speed_sort_idxs, 'avg_2D_speed_sort_idxs': avg_2D_speed_sort_idxs,
                                                                    'sorted_1D_avg_speed_per_pos': sorted_1D_avg_speed_per_pos, 'sorted_avg_2D_speed_per_pos': sorted_avg_2D_speed_per_pos, 
                                                                    'sorted_PFoverlapDensity_1D': sorted_PFoverlapDensity_1D, 'sorted_PFoverlapDensity_2D': sorted_PFoverlapDensity_2D                                                                   
                                                                   })
            return computation_result
        
        
    def _perform_velocity_vs_pf_simplified_count_density_computation(computation_result: ComputationResult, debug_print=False):
            """ Builds the simplified density analysis suggested by Kamran at the 2022-07-06 lab meeting related analysis to test Eloy's Pf-Density/Velocity Hypothesis for 2D Placefields
            
            Requires:
                # computed_data['pf1D']
                computed_data['pf2D']
                
                computed_data['EloyAnalysis']['avg_2D_speed_per_pos']
                computed_data['EloyAnalysis']['avg_2D_speed_sort_idxs']
                
            Provides:
                computed_data['SimplerNeuronMeetingThresholdFiringAnalysis']
                computed_data['SimplerNeuronMeetingThresholdFiringAnalysis']['n_neurons_meeting_firing_critiera_by_position_bins_2D']: 
                computed_data['SimplerNeuronMeetingThresholdFiringAnalysis']['sorted_n_neurons_meeting_firing_critiera_by_position_bins_2D']: 
                
            """
            
            # active_pf_1D = computation_result.computed_data['pf1D']
            active_pf_2D = computation_result.computed_data['pf2D']
            
            avg_2D_speed_per_pos = computation_result.computed_data['EloyAnalysis']['avg_2D_speed_per_pos']
            avg_2D_speed_sort_idxs = computation_result.computed_data['EloyAnalysis']['avg_2D_speed_sort_idxs']
            
            ## From active_pf_2D:
            meets_firing_threshold_indicies = np.where((active_pf_2D.ratemap.tuning_curves > 1.0)) # ((2287,), (2287,), (2287,)) # tuple of indicies for each axis of values that meet the critiera
            n_xbins = len(active_pf_2D.xbin) - 1 # the -1 is to get the counts for the centers only
            n_ybins = len(active_pf_2D.ybin) - 1 # the -1 is to get the counts for the centers only
            n_neurons = active_pf_2D.ratemap.n_neurons
            
            ## From active_pf_2D_dt:
            ## uses curr_occupancy_weighted_tuning_maps_matrix: the firing rate of each neuron in a given position bin, hopefully in Hz
            # meets_firing_threshold_indicies = np.where((active_pf_2D_dt.curr_occupancy_weighted_tuning_maps_matrix > 1.0)) 
            # n_xbins = len(active_pf_2D_dt.xbin) - 1 # the -1 is to get the counts for the centers only
            # n_ybins = len(active_pf_2D_dt.ybin) - 1 # the -1 is to get the counts for the centers only
            # n_neurons = active_pf_2D_dt.n_fragile_linear_neuron_IDXs

            # meets_firing_threshold_indicies: ((2287,), (2287,), (2287,)) # tuple of indicies for each axis of values that meet the critiera
            n_neurons_meeting_firing_critiera_by_position_bins_2D = np.zeros((n_neurons, n_xbins, n_ybins), dtype=int) # create an accumulator matrix

            for (fragile_linear_neuron_IDX, xbin_index, ybin_index) in zip(meets_firing_threshold_indicies[0], meets_firing_threshold_indicies[1], meets_firing_threshold_indicies[2]):
                n_neurons_meeting_firing_critiera_by_position_bins_2D[fragile_linear_neuron_IDX, xbin_index, ybin_index] += 1
                            
            n_neurons_meeting_firing_critiera_by_position_bins_2D = np.sum(n_neurons_meeting_firing_critiera_by_position_bins_2D, 0)
            assert np.shape(n_neurons_meeting_firing_critiera_by_position_bins_2D) == (n_xbins, n_ybins), f"should reduce to number of bins, with each bin containing the count of the number of neurons that meet the 1Hz firing criteria"
            # _n_neurons_meeting_firing_critiera_by_position_bins.shape # (64, 29) should reduce to number of bins, with each bin containing the count of the number of neurons that meet the 1Hz firing criteria
            
            ## 2D: Average velocity per position bin:
            ## Apply the same ordering to the n_neurons_meeting_firing_critiera_by_position_bins_2D
            # sorted_avg_2D_speed_per_pos = avg_2D_speed_per_pos.flat[avg_2D_speed_sort_idxs]
            sorted_n_neurons_meeting_firing_critiera_by_position_bins_2D = n_neurons_meeting_firing_critiera_by_position_bins_2D.flat[avg_2D_speed_sort_idxs]
            
            computation_result.computed_data['SimplerNeuronMeetingThresholdFiringAnalysis'] = DynamicParameters.init_from_dict({'n_neurons_meeting_firing_critiera_by_position_bins_2D': n_neurons_meeting_firing_critiera_by_position_bins_2D, 'sorted_n_neurons_meeting_firing_critiera_by_position_bins_2D': sorted_n_neurons_meeting_firing_critiera_by_position_bins_2D})
            return computation_result
        