import sys
from warnings import warn
import numpy as np
import pandas as pd
import itertools # for _perform_placefield_overlap_computation

import matplotlib
# configure backend here
matplotlib.use('Agg')
import matplotlib.pyplot as plt # required for _perform_pf_analyze_results_peak_prominence2d_computation to build Path objects. Nothing is plotted though

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.External.peak_prominence2d import compute_prominence_contours # Required for _perform_pf_find_ratemap_peaks_peak_prominence2d_computation

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder

from scipy.ndimage.filters import uniform_filter, gaussian_filter # for _perform_pf_find_ratemap_peaks_peak_prominence2d_computation

from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns # for _perform_pf_find_ratemap_peaks_peak_prominence2d_computation/_build_filtered_summits_analysis_results

"""-------------- Specific Computation Functions to be registered --------------"""

modify_dict_mode = True # if True, writes the dict


class FindpeaksHelpers:
    
    def ratemap_find_placefields(ratemap, debug_print=False):
        """ Uses the `findpeaks` library for finding local maxima of TuningMaps
        Input:
        ratemap: a 2D ratemap
        
        Returns:
        
        
        """
        from findpeaks import findpeaks # for _perform_pf_find_ratemap_peaks_computation. Install with pip install findpeaks. Specifically used in `ratemap_find_placefields` subfunction
        
        def _ratemap_compute_peaks(X, debug_print):
            if debug_print:
                print(f'np.shape(X): {np.shape(X)}') # np.shape(X): (60, 8)
            
            if debug_print:
                verboosity_level = 3 # info
            else:
                # verboosity_level = 2 # warnings only
                verboosity_level = 1 # errors only
            # Initialize
            
            fp_mask = findpeaks(method='mask', verbose=verboosity_level)
            # Fit
            results_mask = fp_mask.fit(X)
            # Initialize
            fp_topo = findpeaks(method='topology', verbose=verboosity_level)
            # Fit
            results_topo = fp_topo.fit(X)
            return fp_mask, results_mask, fp_topo, results_topo

        def _expand_mask_results(results_mask):
            """ Expands the results from peak detection with the 'mask' method and returns a dataframe """
            # list(results_mask.keys()) # ['Xraw', 'Xproc', 'Xdetect', 'Xranked']
            ranked_peaks = results_mask['Xranked'] # detected peaks with respect the input image. Elements are the ranked peaks (1=best).
            peak_scores = results_mask['Xdetect'] # the scores
            peak_indicies = np.argwhere(ranked_peaks) # [[ 3  7],[ 7 14],[ 7 15],[13 17],[13 18],[25 17],[29 11],[44 15],[59 22]]
            if debug_print:
                print(peak_indicies) 
            peak_ranks = ranked_peaks[peak_indicies[:,0], peak_indicies[:,1]] # array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            if debug_print:
                print(f'peak_ranks: {peak_ranks}')
            peak_scores = peak_scores[peak_indicies[:,0], peak_indicies[:,1]]
            if debug_print:
                print(f'peak_scores: {peak_ranks}')
            peak_values = results_mask['Xraw'][peak_indicies[:,0], peak_indicies[:,1]]
            if debug_print:
                print(f'peak_values: {peak_values}')
            # scipy.ndimage.measurements.label # to label the found peaks?
            return pd.DataFrame({'rank': peak_ranks, 'peak_score': peak_scores, 'xbin_idx':peak_indicies[:,0], 'ybin_idx':peak_indicies[:,1], 'value': peak_values})
        
        def _expand_topo_results(results_topo):
            """ Expands the results from peak detection with the 'topology' method and returns a dataframe """
            # list(results_mask.keys()) # results_topo.keys(): ['Xraw', 'Xproc', 'Xdetect', 'Xranked', 'persistence', 'groups0']
            ranked_peaks = results_topo['Xranked'] # detected peaks with respect the input image. Elements are the ranked peaks (1=best).
            peak_scores = results_topo['Xdetect'] # the scores (higher is better)
            peak_indicies = np.argwhere(ranked_peaks) # [[ 3  7],[ 7 14],[ 7 15],[13 17],[13 18],[25 17],[29 11],[44 15],[59 22]]
            peak_ranks = ranked_peaks[peak_indicies[:,0], peak_indicies[:,1]] # array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            # is_peak = results_topo['persistence'].peak.to_numpy() # Bool array, True if it is a peak, false otherwise
            is_peak = peak_ranks > 0 # Bool array, True if it is a peak, false otherwise
            peak_indicies = peak_indicies[is_peak] # only get the peak_indicies corresponding to peaks (not valleys)
            peak_ranks = peak_ranks[is_peak] # remove non-peak indicies
            if debug_print:
                print(peak_indicies)
            if debug_print:
                print(f'peak_ranks: {peak_ranks}')
            peak_scores = peak_scores[peak_indicies[:,0], peak_indicies[:,1]]
            if debug_print:
                print(f'peak_scores: {peak_ranks}')
            peak_values = results_topo['Xraw'][peak_indicies[:,0], peak_indicies[:,1]]
            if debug_print:
                print(f'peak_values: {peak_values}')
            if debug_print:
                print(f"Xproc: {results_topo['Xproc']}, groups0: {results_topo['groups0']}") # , persistence: {results_topo['persistence']}
            
            out_df = pd.DataFrame({'rank': peak_ranks, 'peak_score': peak_scores, 'xbin_idx':peak_indicies[:,0], 'ybin_idx':peak_indicies[:,1], 'value': peak_values})
            out_df.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
            
            return (out_df, results_topo['persistence'])
        
        fp_mask, results_mask, fp_topo, results_topo = _ratemap_compute_peaks(ratemap, debug_print=debug_print)
        if debug_print:
            print(f'results_topo.keys(): {list(results_topo.keys())}')
        topo_results_df, topo_persistence_df =_expand_topo_results(results_topo)
        mask_results_df = _expand_mask_results(results_mask)
        return fp_mask, mask_results_df, fp_topo, topo_results_df, topo_persistence_df
        
        
    def _filter_found_peaks_by_exclusion_threshold(df_list, peak_xy_points_pos_list, peak_score_inclusion_percent_threshold=0.25, debug_print=False):
        peak_score_inclusion_threshold = peak_score_inclusion_percent_threshold * 255.0 # it must be at least 1/4 of the promenance of the largest peak (which is always 255.0)
        peaks_are_included_list = [(result_df['peak_score'] > peak_score_inclusion_threshold) for result_df in df_list]
        
        ## filter by the peaks_are_included_list:
        filtered_df_list = [a_df[peak_is_included] for a_df, peak_is_included in zip(df_list, peaks_are_included_list)] # Also works
        filtered_peak_xy_points_pos_list = [a_xy_points_pos[:, peak_is_included] for a_xy_points_pos, peak_is_included in zip(peak_xy_points_pos_list, peaks_are_included_list)]

        if debug_print:
            print(f'peaks_are_included_list: {peaks_are_included_list}')
            print(f'filtered_df_list: {filtered_df_list}')
            print(f'filtered_peak_xy_points_pos_list: {filtered_peak_xy_points_pos_list}')

        return peaks_are_included_list, filtered_df_list, filtered_peak_xy_points_pos_list
        


            
class PlacefieldDensityAnalysisComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """Performs analyeses related to placefield densities and overlap across spactial bins. Includes analyses for Eloy from 07-2022. 
    
    active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
    active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
    active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
    active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
    
    provides: ['EloyAnalysis', 'SimplerNeuronMeetingThresholdFiringAnalysis', 'RatemapPeaksAnalysis', 'placefield_overlap']

    """
    _computationPrecidence = 2 # must be done after PlacefieldComputations and DefaultComputationFunctions
    _is_global = False


    @function_attributes(short_name='EloyAnalysis', tags=['pf_density','velocity'],
                         input_requires=["computation_result.computed_data['pf1D']", "computation_result.computed_data['pf2D']", "computation_result.computed_data['pf2D_TwoStepDecoder']['avg_speed_per_pos']"],
                         output_provides=["computation_result.computed_data['EloyAnalysis']"],
                         uses=[], used_by=[], creation_date='2023-09-12 17:11', related_items=[], 
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['EloyAnalysis'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['EloyAnalysis']['sorted_PFoverlapDensity_2D']), is_global=False)
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
        

    @function_attributes(short_name='velocity_vs_pf_simplified_count_density', tags=['pf_density', 'velocity', 'simplified'],
                        input_requires=[],
                        output_provides=["computation_result.computed_data['SimplerNeuronMeetingThresholdFiringAnalysis']"],
                        uses=[], used_by=[], creation_date='2022-07-06 00:00', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['SimplerNeuronMeetingThresholdFiringAnalysis'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['SimplerNeuronMeetingThresholdFiringAnalysis']['sorted_n_neurons_meeting_firing_critiera_by_position_bins_2D']), is_global=False)
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
        
    @function_attributes(short_name='_DEP_ratemap_peaks', tags=['pf','ratemap', 'peaks', 'DEPRICATED'],
                          input_requires=[],
                          output_provides=["computation_result.computed_data['RatemapPeaksAnalysis']"],
                          uses=['findpeaks'], used_by=[], creation_date='2023-09-12 17:24', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['RatemapPeaksAnalysis'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['RatemapPeaksAnalysis']['final_filtered_results']), is_global=False)
    def _DEP_perform_pf_find_ratemap_peaks_computation(computation_result: ComputationResult, debug_print=False, peak_score_inclusion_percent_threshold=0.25):
            """ Uses the `findpeaks` library to compute the topographical peak locations and information with the intent of doing an extended pf size/density analysis.
                Not really used as the `peak_prominence2d` seems to work much better.
                        
            Requires:
                computed_data['pf2D']                
                
            Provides:
                computed_data['RatemapPeaksAnalysis']
                # computed_data['RatemapPeaksAnalysis']['tuning_curve_findpeaks_results']: peaks_outputs: fp_mask, mask_results_df, fp_topo, topo_results_df, topo_persistence_df
                
                computed_data['RatemapPeaksAnalysis']['mask_results']: 
                    computed_data['RatemapPeaksAnalysis']['mask_results']['fp_list']:
                    computed_data['RatemapPeaksAnalysis']['mask_results']['df_list']:
                    
                computed_data['RatemapPeaksAnalysis']['topo_results']:
                    computed_data['RatemapPeaksAnalysis']['topo_results']['fp_list']:
                    computed_data['RatemapPeaksAnalysis']['topo_results']['df_list']:
                    computed_data['RatemapPeaksAnalysis']['topo_results']['persistence_df_list']:
                    computed_data['RatemapPeaksAnalysis']['topo_results']['peak_xy_points_pos_list']: the actual (x, y) positions of the peak points for each neuron
                    
                computed_data['RatemapPeaksAnalysis']['final_filtered_results']:
                    computed_data['RatemapPeaksAnalysis']['final_filtered_results']['peaks_are_included_list']: a list of bools into the original raw arrays ('topo_results') that was used to filter down to the final peaks based on the promenences
                    computed_data['RatemapPeaksAnalysis']['final_filtered_results']['df_list']:
                    computed_data['RatemapPeaksAnalysis']['final_filtered_results']['peak_xy_points_pos_list']:
                    
            """            

            # ==================================================================================================================== #
            # BEGIN MAIN FUNCTION BODY                                                                                             #
            # ==================================================================================================================== #
            active_pf_1D = computation_result.computed_data['pf1D']
            active_pf_2D = computation_result.computed_data['pf2D']
            fp_mask_list, mask_results_df_list, fp_topo_list, topo_results_df_list, topo_persistence_df_list = tuple(zip(*[FindpeaksHelpers.ratemap_find_placefields(a_tuning_curve.copy(), debug_print=debug_print) for a_tuning_curve in active_pf_2D.ratemap.pdf_normalized_tuning_curves]))
            topo_results_peak_xy_pos_list = [np.vstack((active_pf_2D.xbin[curr_topo_result_df['xbin_idx'].to_numpy()], active_pf_2D.ybin[curr_topo_result_df['ybin_idx'].to_numpy()])) for curr_topo_result_df in topo_results_df_list]
            peaks_are_included_list, filtered_df_list, filtered_peak_xy_points_pos_list = FindpeaksHelpers._filter_found_peaks_by_exclusion_threshold(topo_results_df_list, topo_results_peak_xy_pos_list, peak_score_inclusion_percent_threshold=peak_score_inclusion_percent_threshold)
            
            computation_result.computed_data['RatemapPeaksAnalysis'] = DynamicParameters(mask_results=DynamicParameters(fp_list=fp_mask_list, df_list=mask_results_df_list),
                                                                                         topo_results=DynamicParameters(fp_list=fp_topo_list, df_list=topo_results_df_list, persistence_df_list=topo_persistence_df_list, peak_xy_points_pos_list=topo_results_peak_xy_pos_list),
                                                                                         final_filtered_results=DynamicParameters(df_list=filtered_df_list, peak_xy_points_pos_list=filtered_peak_xy_points_pos_list, peaks_are_included_list=peaks_are_included_list)
                                                                                         )
            
            return computation_result

    @function_attributes(short_name='ratemap_peaks_prominence2d', tags=['pf', 'peaks', 'promienence', '2d', 'ratemap', 'Eloy'],
                          input_requires=["computation_result.computed_data['pf2D']"],
                          output_provides=["computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D']"],
                          uses=["compute_prominence_contours"], used_by=[], creation_date='2023-09-12 17:21', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['RatemapPeaksAnalysis'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['RatemapPeaksAnalysis']['PeakProminence2D']), is_global=False)
    def _perform_pf_find_ratemap_peaks_peak_prominence2d_computation(computation_result: ComputationResult, step=0.01, peak_height_multiplier_probe_levels=(0.5, 0.9), minimum_included_peak_height = 0.2, uniform_blur_size = 3, gaussian_blur_sigma = 3, debug_print=False):
            """ Uses the peak_prominence2d package to find the peaks and promenences of 2D placefields
            
            Independent of the other peak-computing computation functions above
            
            Inputs:
                peak_height_multiplier_probe_levels = (0.5, 0.9) # 50% and 90% of the peak height
                gaussian_blur_sigma: input to gaussian_filter for blur of peak counts
                uniform_blur_size: inputt to uniform_filter for blur of peak counts
                
                
            Requires:
                computed_data['pf2D']
                
            Provides:
                computed_data['RatemapPeaksAnalysis']['PeakProminence2D']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['xx']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['yy']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['neuron_extended_ids']
                    
                    
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['result_tuples']: (slab, peaks, idmap, promap, parentmap)
                    
                    flat_peaks_df
                    filtered_flat_peaks_df
                    
                    peak_counts
                        raw
                        uniform_blurred
                        gaussian_blurred
                    
            """            
            
            matplotlib.use('Agg') # require use of non-interactive backend to prevent the stupid figure from showing up

            def _find_contours_at_levels(xbin_centers, ybin_centers, slab, peak_probe_point, probe_levels):
                """ finds the contours containing the peak_probe_point at the specified probe_levels.
                    performs slicing through desired z-values (1/2 prominence, etc) using contourf
                    
                    
                Inputs:
                    peak_probe_point: a point (x, y) to use to validate or exclude found contours. This allows us to only get the contour the encloses a peak at a given level, not any others that may happen to be at that level as well.
                    probe_levels: a list of z-values to slice at to find the contours
                    
                Returns:
                    a dict with keys of the probe_levels and values containing a list of their corresponding contours
                """
                vmax = np.nanmax(slab)
                fig, ax = plt.subplots()
                included_computed_contours = DynamicParameters.init_from_dict({}) 
                #---------------Loop through levels---------------
                for ii, levii in enumerate(probe_levels[::-1]):
                    # Note that contourf requires at least 2 levels, hence the use of the vmax+1.0 term and accessing only the first item in the collection. Otherwise: "ValueError: Filled contours require at least 2 levels."
                    csii = ax.contourf(xbin_centers, ybin_centers, slab, [levii, vmax+1.0]) ## Heavy-lifting code here. levii is the level
                    csii = csii.collections[0]
                    # ax.cla() ## TODO: this is the most computationally expensive part of the code, and it doesn't seem necissary
                    #--------------Loop through contours at level--------------
                    # find only the ones containing the peak_probe_point
                    included_computed_contours[levii] = [contjj for jj, contjj in enumerate(csii.get_paths()) if contjj.contains_point(peak_probe_point)]
                    n_contours = len(included_computed_contours[levii])
                    assert n_contours <= 1, f"n_contours is supposed to be equal to be either 0 or 1 but len(included_computed_contours[levii]): {len(included_computed_contours[levii])}!"
                    # assert n_contours == 1, f"contour_stats is supposed to be equal to 1 but len(included_computed_contours[levii]): {len(included_computed_contours[levii])}!"
                    if n_contours == 0:
                        warn( f"n_contours is 0 for level: {levii}")
                        included_computed_contours[levii] = None # set to None
                    else:                   
                        included_computed_contours[levii] = included_computed_contours[levii][0] # unwrapped from the list format, it's just the single Path/Curve now
                    
                plt.close(fig) # close the figure when done generating the contours to prevent an empty figure from showing
                return included_computed_contours

            def _build_filtered_summits_analysis_results(xbin, ybin, xbin_labels, ybin_labels, flat_peaks_df, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=0.5, debug_print=False):
                """ builds the filtered summits analysis results dataframe and flat counts matrix 
                
                Usage:
                    filtered_summits_analysis_df, pf_peak_counts_map = build_filtered_summits_analysis_results(active_pf_2D.xbin, active_pf_2D.ybin, active_pf_2D.xbin_labels, active_pf_2D.ybin_labels,
                                                                                                    active_peak_prominence_2d_results, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=1.0, debug_print = False)
                                                                                                    
                """
                ## Find which position bin each peak falls in and add it to the flat_peaks_df:
                filtered_summits_analysis_df = flat_peaks_df[flat_peaks_df['peak_height'] >= minimum_included_peak_height].copy() # filter for peaks greater than 1.0Hz
                
                ## IMPORTANT: Filter by only one of the slice_levels before continuing, otherwise you're double-counting:
                filtered_summits_analysis_df = filtered_summits_analysis_df[filtered_summits_analysis_df['slice_level_multiplier'] == slice_level_multiplier].copy()

                ## Build outputs:
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
                pf_peak_counts_map = np.zeros((n_xbins, n_ybins), dtype=int) # create an initially zero matrix

                current_bin_counts = filtered_summits_analysis_df.value_counts(subset=['peak_center_binned_x', 'peak_center_binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # current_bin_counts: a series with a MultiIndex index for each bin that has nonzero counts
                if debug_print:
                    print(f'np.shape(current_bin_counts): {np.shape(current_bin_counts)}') # (247,)
                for (xbin_label, ybin_label), count in current_bin_counts.iteritems():
                    if debug_print:
                        print(f'xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
                    try:
                        pf_peak_counts_map[xbin_label-1, ybin_label-1] += count #if it's already a label, why are we subtracting 1?
                    except IndexError as e:
                        print(f'e: {e}\n filtered_summits_analysis_df: {np.shape(filtered_summits_analysis_df)}, current_bin_counts: {np.shape(current_bin_counts)}\n pf_peak_counts_map: {np.shape(pf_peak_counts_map)}')
                        raise e
                    
                return filtered_summits_analysis_df, pf_peak_counts_map
            
            def _compute_distances_from_peaks_to_boundary(active_pf_2D, filtered_flat_peaks_df, debug_print = True):
                """ Computes the distance to boundary by computing the distance to the nearest never-occupied bin
                        For any given peak location, the distance to the boundary in each of the four directions can be computed.
                        
                    TODO: this function currently uses the binned peak positions and computes distances to the boundaries in terms of bins in each dimension. Could use a continuous position measure as well.
                    

                # filtered_flat_peaks_df

                # Required Input Columns:
                # ['peak_center_binned_x', 'peak_center_binned_y']

                # Output Columns:
                # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate

                # ['peak_nearest_directional_boundary_bins', 'peak_nearest_directional_boundary_displacements', 'peak_nearest_directional_boundary_distances'] # combined tuple columns
                
                
                TODO: I should have just used actual continuous position values instead of counting bins :[
                    
                Usage:
                    peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = _compute_distances_from_peaks_to_boundary(active_pf_2D, filtered_summits_analysis_df, debug_print=debug_print)

                """
                # Build the boundary mask from the NaN speeds, which correspond to never-occupied cells:
                # boundary_mask_indicies = ~np.isfinite(active_eloy_analysis.avg_2D_speed_per_pos)
                boundary_mask_indicies = active_pf_2D.never_visited_occupancy_mask.copy() # True if value is never-occupied, False otherwise

                ## Add a padding of size 1 of True values around the edge, ensuring a border of never-visited bins on all sides:
                # boundary_mask_indicies = np.pad(boundary_mask_indicies, 1, 'constant', constant_values=(True, True)) ## BUG: this changes the indicies and doesn't completely fix the problem

                ## Get just the True indicies. A 2-tuple of 1D np.array vectors containing the true indicies
                boundary_mask_true_indicies = np.vstack(np.where(boundary_mask_indicies)).T
                # boundary_mask_true_indicies.shape # (235, 2)
                # boundary_mask_true_indicies

                ## Compute the extrema to deal with border effects:
                # active_pf_2D.bin_info
                xbin_indicies = active_pf_2D.xbin_labels -1
                xbin_outer_extrema = (xbin_indicies[0]-1, xbin_indicies[-1]+1) # if indicies [0, 59] are valid, the outer_extrema for this axis should be (-1, 60)
                ybin_indicies = active_pf_2D.ybin_labels -1
                ybin_outer_extrema = (ybin_indicies[0]-1, ybin_indicies[-1]+1) # if indicies [0, 7] are valid, the outer_extrema for this axis should be (-1, 8)

                if debug_print:
                    print(f'xbin_indicies: {xbin_indicies}\nxbin_outer_extrema: {xbin_outer_extrema}\nybin_indicies: {ybin_indicies}\nybin_outer_extrema: {ybin_outer_extrema}')
                
                peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = list(), list(), list()

                for a_peak_row in filtered_flat_peaks_df[['peak_center_binned_x', 'peak_center_binned_y']].itertuples():
                    peak_x_bin_idx, peak_y_bin_idx = (a_peak_row.peak_center_binned_x-1), (a_peak_row.peak_center_binned_y-1)
                    if debug_print:
                        print(f'peak_x_bin_idx: {peak_x_bin_idx}, peak_y_bin_idx: {peak_y_bin_idx}')
                    # For a given (x_idx, y_idx):
                    ## Perform vertical line scan (across y-values) by first getting all matching x-values:
                    matching_vertical_scan_y_idxs = boundary_mask_true_indicies[(boundary_mask_true_indicies[:,0]==peak_x_bin_idx), 1] # the [*, 1] is because we only need the y-values
                    # matching_vertical_scan_y_idxs # array([0, 1, 2, 6, 7], dtype=int64)
                    if debug_print:
                        print(f'\tmatching_vertical_scan_y_idxs: {matching_vertical_scan_y_idxs}')

                    if len(matching_vertical_scan_y_idxs) == 0:
                        # both min and max ends missing. Should be set to the bin just outside the minimum and maximum bin in that dimension
                        warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 0: setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                        matching_vertical_scan_y_idxs = ybin_outer_extrema
                    elif len(matching_vertical_scan_y_idxs) == 1:
                        # only one end missing, need to determine which end it is and replace the missing end with the appropriate extrema
                        if (matching_vertical_scan_y_idxs[0] > peak_y_bin_idx):
                            # add the lower extrema
                            warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 1: missing lower extrema, adding ybin_outer_extrema[0] = {ybin_outer_extrema[0]} to matching_vertical_scan_y_idxs')
                            matching_vertical_scan_y_idxs = np.insert(matching_vertical_scan_y_idxs, 0, ybin_outer_extrema[0])
                            # matching_horizontal_scan_x_idxs.insert(xbin_outer_extrema[0], 0)
                        elif (matching_vertical_scan_y_idxs[0] < peak_y_bin_idx):
                            # add the upper extrema
                            warn(f'\tWARNING: len(matching_vertical_scan_y_idxs) == 1: missing upper extrema, adding ybin_outer_extrema[1] = {ybin_outer_extrema[1]} to matching_vertical_scan_y_idxs')
                            matching_vertical_scan_y_idxs = np.append(matching_vertical_scan_y_idxs, [ybin_outer_extrema[1]])
                        else:
                            # # EQUAL CONDITION SHOULDN'T HAPPEN!
                            # raise NotImplementedError
                            # This condition should only happen when peak_y_bin_idx is right against the boundary itself (e.g. (peak_y_bin_idx == 7) or (peak_y_bin_idx == 0)
                            if (peak_y_bin_idx == ybin_indicies[0]):
                                # matching_vertical_scan_y_idxs[0] = ybin_outer_extrema[0] ## replace the duplicated value with the lower extreme
                                warn(f'\tWARNING: peak_y_bin_idx ({peak_y_bin_idx}) == ybin_indicies[0] ({ybin_indicies[0]}): setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                                matching_vertical_scan_y_idxs = ybin_outer_extrema
                            elif (peak_y_bin_idx == ybin_indicies[-1]):
                                # matching_vertical_scan_y_idxs[0] = ybin_outer_extrema[1] ## replace the duplicated value with the upper extreme
                                warn(f'\tWARNING: peak_y_bin_idx ({peak_y_bin_idx}) == ybin_indicies[-1] ({ybin_indicies[-1]}): setting matching_vertical_scan_y_idxs = {ybin_outer_extrema}')
                                matching_vertical_scan_y_idxs = ybin_outer_extrema
                            else:
                                warn(f'\tWARNING: This REALLY should not happen! peak_y_bin_idx: {peak_y_bin_idx}, matching_vertical_scan_y_idxs: {matching_vertical_scan_y_idxs}!!')
                                raise NotImplementedError
                                
                    ## Partition on the peak_y_bin_idx:
                    found_start_indicies = np.searchsorted(matching_vertical_scan_y_idxs, peak_y_bin_idx, side='left')
                    found_end_indicies = np.searchsorted(matching_vertical_scan_y_idxs, peak_y_bin_idx, side='right') # find the end of the range
                    out = np.hstack((found_start_indicies, found_end_indicies))
                    if debug_print:     
                        print(f'\tfound_start_indicies: {found_start_indicies}, found_end_indicies: {found_end_indicies}, out: {out}')
                    split_vertical_scan_y_idxs = np.array_split(matching_vertical_scan_y_idxs, [found_start_indicies]) # need to pass in found_start_indicies as a list containing the scalar value because this functionality is different than if the scalar itself is passed in.
                    if debug_print:
                        print(f'\tsplit_vertical_scan_y_idxs: {split_vertical_scan_y_idxs}')

                    """ Encountering IndexError with split_vertical_scan_y_idxs[0][-1], says len(split_vertical_scan_y_idxs[0]) == 0
                    peak_x_bin_idx: 1, peak_y_bin_idx: 0
                        matching_vertical_scan_y_idxs: [6 7]
                        found_start_indicies: 0, found_end_indicies: 0, out: [0 0]
                        split_vertical_scan_y_idxs: [array([], dtype=int64), array([6, 7], dtype=int64)]

                    """
                    lower_list, upper_list = split_vertical_scan_y_idxs[0], split_vertical_scan_y_idxs[1]
                    if len(lower_list)==0:
                        # if the lower list is empty get the ybin_outer_extrema[0]
                        below_bound = ybin_outer_extrema[0]
                    else:
                        below_bound = lower_list[-1] # get the last (maximum) of the lower list

                    if len(upper_list)==0:
                        # if the upper list is empty get the ybin_outer_extrema[1]
                        above_bound = ybin_outer_extrema[1]
                    else:
                        above_bound = upper_list[0] # get the first (minimum) of the upper list
                    vertical_scan_result = (below_bound, above_bound) # get the last (maximum) of the lower list, and the first (minimum) of the upper list.
                    if debug_print:
                        print(f'\tvertical_scan_result: {vertical_scan_result}') # vertical_scan_result: (2, 6)


                    ## Perform horizontal line scan (across x-values):
                    matching_horizontal_scan_x_idxs = boundary_mask_true_indicies[(boundary_mask_true_indicies[:,1]==peak_y_bin_idx), 0] # the [*, 0] is because we only need the x-values
                    # matching_horizontal_scan_x_idxs # array([0, 1, 2, 6, 7], dtype=int64)
                    if debug_print:
                        print(f'\tmatching_horizontal_scan_x_idxs: {matching_horizontal_scan_x_idxs}')

                    if len(matching_horizontal_scan_x_idxs) == 0:
                        # both min and max ends missing. Should be set to the bin just outside the minimum and maximum bin in that dimension
                        warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 0: setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                        matching_horizontal_scan_x_idxs = xbin_outer_extrema
                    elif len(matching_horizontal_scan_x_idxs) == 1:
                        # only one end missing, need to determine which end it is and replace the missing end with the appropriate extrema
                        if (matching_horizontal_scan_x_idxs[0] > peak_x_bin_idx):
                            # add the lower extrema
                            warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 1: missing lower extrema, adding xbin_outer_extrema[0] = {xbin_outer_extrema[0]} to matching_horizontal_scan_x_idxs')
                            matching_horizontal_scan_x_idxs = np.insert(matching_horizontal_scan_x_idxs, 0, xbin_outer_extrema[0])
                            # matching_horizontal_scan_x_idxs.insert(xbin_outer_extrema[0], 0)
                        elif (matching_horizontal_scan_x_idxs[0] < peak_x_bin_idx):
                            # add the upper extrema
                            warn(f'\tWARNING: len(matching_horizontal_scan_x_idxs) == 1: missing upper extrema, adding xbin_outer_extrema[1] = {xbin_outer_extrema[1]} to matching_horizontal_scan_x_idxs')
                            matching_horizontal_scan_x_idxs = np.append(matching_horizontal_scan_x_idxs, [xbin_outer_extrema[1]])
                        else:
                            # # EQUAL CONDITION SHOULDN'T HAPPEN!
                            # raise NotImplementedError
                            # This condition should only happen when peak_x_bin_idx is right against the boundary itself (e.g. (peak_x_bin_idx == 7) or (peak_x_bin_idx == 0)
                            if (peak_x_bin_idx == xbin_indicies[0]):
                                # matching_horizontal_scan_x_idxs[0] = xbin_outer_extrema[0] ## replace the duplicated value with the lower extreme
                                warn(f'\tWARNING: peak_x_bin_idx ({peak_x_bin_idx}) == xbin_indicies[0] ({xbin_indicies[0]}): setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                                matching_horizontal_scan_x_idxs = xbin_outer_extrema
                            elif (peak_x_bin_idx == xbin_indicies[-1]):
                                # matching_horizontal_scan_x_idxs[0] = xbin_outer_extrema[1] ## replace the duplicated value with the upper extreme
                                warn(f'\tWARNING: peak_x_bin_idx ({peak_x_bin_idx}) == xbin_indicies[-1] ({xbin_indicies[-1]}): setting matching_horizontal_scan_x_idxs = {xbin_outer_extrema}')
                                matching_horizontal_scan_x_idxs = xbin_outer_extrema
                            else:
                                warn(f'\tWARNING: This REALLY should not happen! peak_x_bin_idx: {peak_x_bin_idx}, matching_horizontal_scan_x_idxs: {matching_horizontal_scan_x_idxs}!!')
                                raise NotImplementedError
                                
                    # Otherwise we're good

                    ### Partition on the peak_x_bin_idx
                    found_start_indicies = np.searchsorted(matching_horizontal_scan_x_idxs, peak_x_bin_idx, side='left')
                    found_end_indicies = np.searchsorted(matching_horizontal_scan_x_idxs, peak_x_bin_idx, side='right') # find the end of the range
                    out = np.hstack((found_start_indicies, found_end_indicies))
                    if debug_print:
                        print(f'\tfound_start_indicies: {found_start_indicies}, found_end_indicies: {found_end_indicies}, out: {out}')
                    split_horizontal_scan_x_idxs = np.array_split(matching_horizontal_scan_x_idxs, [found_start_indicies]) # need to pass in found_start_indicies as a list containing the scalar value because this functionality is different than if the scalar itself is passed in.
                    if debug_print:
                        print(f'\tsplit_horizontal_scan_x_idxs: {split_horizontal_scan_x_idxs}')

                    lower_list, upper_list = split_horizontal_scan_x_idxs[0], split_horizontal_scan_x_idxs[1]
                    if len(lower_list)==0:
                        # if the lower list is empty get the xbin_outer_extrema[0]
                        below_bound = xbin_outer_extrema[0]
                    else:
                        below_bound = lower_list[-1] # get the last (maximum) of the lower list
                    if len(upper_list)==0:
                        # if the upper list is empty get the xbin_outer_extrema[1]
                        above_bound = xbin_outer_extrema[1]
                    else:
                        above_bound = upper_list[0] # get the first (minimum) of the upper list
                    horizontal_scan_result = (below_bound, above_bound) # get the last (maximum) of the lower list, and the first (minimum) of the upper list.
                    if debug_print:
                        print(f'\thorizontal_scan_result: {horizontal_scan_result}') # horizontal_scan_result: (0, 60)
                    
                    ## Build final four directional boundary bins:
                    final_four_boundary_bin_tuples = [(peak_x_bin_idx, boundary_y) for boundary_y in vertical_scan_result] # [(46, 2), (46, 7)]
                    final_four_boundary_bin_tuples += [(boundary_x, peak_y_bin_idx) for boundary_x in horizontal_scan_result] # [(0, 4), (60, 4)]
                    # final_four_boundary_bin_tuples # [(46, 2), (46, 7), (0, 4), (60, 4)]
                    # Add to outputs:
                    peak_nearest_directional_boundary_bins.append(final_four_boundary_bin_tuples)
                    final_four_boundary_bins = np.array(final_four_boundary_bin_tuples) # convert to a (4, 2) np.array
                    if debug_print:
                        print(f'\tfinal_four_boundary_bins: {final_four_boundary_bins}')
                    ## Compute displacements from current point to each boundary:
                    final_four_boundary_displacements = final_four_boundary_bins - [peak_x_bin_idx, peak_y_bin_idx]
                    if debug_print:
                        print(f'\tfinal_four_boundary_displacements: {final_four_boundary_displacements}')

                    # Add to outputs:
                    peak_nearest_directional_boundary_displacements.append([(final_four_boundary_displacements[row_idx,0], final_four_boundary_displacements[row_idx,1]) for row_idx in np.arange(final_four_boundary_displacements.shape[0])])

                    # Compute distances from current point to each boundary:
                    # Flatten down to the pure distances in each component axis, form is (down, up, left, right)
                    final_four_boundary_distances = np.max(np.abs(final_four_boundary_displacements), axis=1) # array([ 2,  2, 47, 14], dtype=int64)
                    # final_four_boundary_distances # again a (4, 2) np.array
                    if debug_print:
                        print(f'\tfinal_four_boundary_distances: {final_four_boundary_distances}')
                    peak_nearest_directional_boundary_distances.append(final_four_boundary_distances)
                
                return peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances


            # ==================================================================================================================== #
            # begin main function body ___________________________________________________________________________________________ #
            active_pf_2D = computation_result.computed_data['pf2D']
            n_neurons = active_pf_2D.ratemap.n_neurons
            
            ## TODO: change to amap with keys of active_pf_2D.neuron_ids
            
            # active_tuning_curves = active_pf_2D.ratemap.tuning_curves # Raw Tuning Curves
            active_tuning_curves = active_pf_2D.ratemap.unit_max_tuning_curves # Unit-max scaled tuning curves
            # tuning_curve_peak_firing_rates = active_pf_2D.ratemap.tuning_curve_peak_firing_rates # the peak firing rates of each tuning curve
            tuning_curve_peak_firing_rates = active_pf_2D.ratemap.tuning_curve_unsmoothed_peak_firing_rates
             
            #  Build the results:
            out_results = {}
            out_cell_peak_dfs_list = []
            n_slices = len(peak_height_multiplier_probe_levels)

            for neuron_idx in np.arange(n_neurons):
                neuron_id = active_pf_2D.neuron_extended_ids[neuron_idx].id #  Inner exception: 'NeuronExtendedIdentity' object has no attribute 'id'
                neuron_tuning_curve_peak_firing_rate = tuning_curve_peak_firing_rates[neuron_idx]
                slab = active_tuning_curves[neuron_idx].T
                _, _, slab, cell_peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(xbin_centers=active_pf_2D.xbin_centers, ybin_centers=active_pf_2D.ybin_centers, slab=slab, step=step, min_area=None, min_depth=0.2, include_edge=True, verbose=False)
                #""" Analyze all peaks of a given cell/ratemap """
                n_peaks = len(cell_peaks_dict)
                
                ## Neuron:
                n_total_cell_slice_results = n_slices * n_peaks                
                neuron_id_arr = np.full((n_total_cell_slice_results,), neuron_id) # repeat the neuron_id many times for the datatable
                neuron_peak_curve_rate_arr = np.full((n_total_cell_slice_results,), neuron_tuning_curve_peak_firing_rate) # repeat the neuron_id many times for the datatable
                
                ## Peak
                summit_slice_peak_id_arr = np.zeros((n_peaks, n_slices), dtype=np.int16) # same summit/peak id for all in the slice
                summit_slice_peak_level_multiplier_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_level_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_height_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_slice_peak_prominence_arr = np.zeros((n_peaks, n_slices), dtype=float) # same summit/peak id for all in the slice
                summit_peak_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_peak_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)
                
                ## Slice
                summit_slice_idx_arr = np.tile(np.arange(n_slices), n_peaks).astype('int') # array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                summit_slice_x_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_y_side_length_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_x_arr = np.zeros((n_peaks, n_slices), dtype=float)
                summit_slice_center_y_arr = np.zeros((n_peaks, n_slices), dtype=float)

                for peak_idx, (peak_id, a_peak_dict) in enumerate(cell_peaks_dict.items()):
                    if debug_print:
                        print(f'computing contours for peak_id: {peak_id}...')                    
                    summit_slice_peak_height_arr[peak_idx, :] = a_peak_dict['height']
                    summit_slice_peak_prominence_arr[peak_idx, :] = a_peak_dict['prominence']
                    summit_peak_center_x_arr[peak_idx, :] = a_peak_dict['center'][0]
                    summit_peak_center_y_arr[peak_idx, :] = a_peak_dict['center'][1]
                    
                    ## This is where we would loop through each desired slice/probe levels:
                    a_peak_dict['probe_levels'] = np.array([a_peak_dict['height']*multiplier for multiplier in peak_height_multiplier_probe_levels]).astype('float') # specific probe levels
                    summit_slice_peak_level_multiplier_arr[peak_idx, :] = np.array(peak_height_multiplier_probe_levels).astype('float')
                    summit_slice_peak_level_arr[peak_idx, :] = a_peak_dict['probe_levels']
                    included_computed_contours = _find_contours_at_levels(active_pf_2D.xbin_centers, active_pf_2D.ybin_centers, slab, a_peak_dict['center'], a_peak_dict['probe_levels']) # DONE: efficiency: This would be more efficient to do for all peaks at once I believe. CONCLUSION: No, this needs to be done separately for each peak as they each have separate prominences which determine the levels they should be sliced at.
                    ## Build the dict that contains the output level slices
                    a_peak_dict['level_slices'] = {probe_lvl:{'contour':contour, 'bbox':contour.get_extents(), 'size':contour.get_extents().size} for probe_lvl, contour in included_computed_contours.items() if (contour is not None)} # if contour is None, it looks like the 'build flat output' step fails below

                    if debug_print:
                        print(f"probe_levels: {a_peak_dict['probe_levels']}")

                    ## Build flat output:
                    for lvl_idx, probe_lvl in enumerate(a_peak_dict['probe_levels']):
                        # a_slice = a_peak_dict['level_slices'][probe_lvl]
                        a_slice = a_peak_dict['level_slices'].get(probe_lvl, None) # allow missing entries. This will occur when (contour is None) above.
                        #TODO 2023-09-25 23:59: - [ ] Do I need to do anything else to handle this case, like remove the invalid curve from 
                        if a_slice is None:
                            print(f'WARNING: a_slice is None. 2023-09-25 - Unsure if this is okay, used to be a fatal error.') # a_peak_dict: {a_peak_dict}
                        else:
                            slice_bbox = a_slice['bbox']
                            (x0, y0, width, height) = slice_bbox.bounds        
                            summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                            summit_slice_x_side_length_arr[peak_idx, lvl_idx] = width
                            summit_slice_y_side_length_arr[peak_idx, lvl_idx] = height
                            # summit_slice_center_x_arr[peak_idx, lvl_idx] = slice_bbox.center[0]
                            # summit_slice_center_y_arr[peak_idx, lvl_idx] = slice_bbox.center[1]
                            summit_slice_center_x_arr[peak_idx, lvl_idx] = float(x0) + (0.5 * float(width))
                            summit_slice_center_y_arr[peak_idx, lvl_idx] = float(y0) + (0.5 * float(height))
                    
                if debug_print:
                    print(f'building peak_df for neuron[{neuron_idx}] with {n_peaks}...')
                cell_peaks_df = pd.DataFrame({'neuron_id': neuron_id_arr, 'neuron_peak_firing_rate': neuron_peak_curve_rate_arr, 'summit_idx': summit_slice_peak_id_arr.flatten(), 'summit_slice_idx': summit_slice_idx_arr.flatten(),
                                             'slice_level_multiplier': summit_slice_peak_level_multiplier_arr.flatten(), 'summit_slice_level': summit_slice_peak_level_arr.flatten(),
                                             'peak_relative_height': summit_slice_peak_height_arr.flatten(), 'peak_prominence': summit_slice_peak_prominence_arr.flatten(),
                                             'peak_center_x': summit_peak_center_x_arr.flatten(), 'peak_center_y': summit_peak_center_y_arr.flatten(),
                                             'summit_slice_x_width': summit_slice_x_side_length_arr.flatten(), 'summit_slice_y_width': summit_slice_y_side_length_arr.flatten(),
                                             'summit_slice_center_x': summit_slice_center_x_arr.flatten(), 'summit_slice_center_y': summit_slice_center_y_arr.flatten()
                                             })
                cell_peaks_df['peak_height'] = cell_peaks_df['peak_relative_height'] * neuron_peak_curve_rate_arr
                
                out_cell_peak_dfs_list.append(cell_peaks_df)
                
                                           
                if debug_print:
                    print(f'done.') # END Analyze peaks
                    
                out_results[neuron_id] = {'peaks': cell_peaks_dict, 'slab': slab, 'id_map':id_map, 'prominence_map':prominence_map, 'parent_map':parent_map} 
    
            # Build final concatenated dataframe:
            if debug_print:
                print(f'building final concatenated cell_peaks_df for {n_neurons} total neurons...')
            cell_peaks_df = pd.concat(out_cell_peak_dfs_list)
            ## Find which position bin each peak falls in and add it to the flat_peaks_df:
            cell_peaks_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(cell_peaks_df, bin_values=(active_pf_2D.xbin, active_pf_2D.ybin), position_column_names=('peak_center_x', 'peak_center_y'), binned_column_names=('peak_center_binned_x', 'peak_center_binned_y'), active_computation_config=None, force_recompute=False, debug_print=debug_print)

            ## Find the avg velocity corresponding to each position bin containing a peak value:
            active_eloy_analysis = computation_result.computed_data.get('EloyAnalysis', None)            
            if active_eloy_analysis is not None:
                ## Extract the previously computed results:
                avg_2D_speed_per_pos = active_eloy_analysis.avg_2D_speed_per_pos # (60, 8)
                # avg_2D_speed_sort_idxs = active_eloy_analysis.avg_2D_speed_sort_idxs
                # assert np.shape(avg_2D_speed_per_pos) == np.shape(pf_peak_counts_map), f"the shape of the active_eloy_analysis.avg_2D_speed_per_pos ({np.shape(avg_2D_speed_per_pos)}) and the shape of the newly built pf_peak_counts_map ({np.shape(pf_peak_counts_map)}) must match!"
                _temp_peak_center_bin_label_xy = cell_peaks_df[['peak_center_binned_x', 'peak_center_binned_y']].to_numpy()-1 # (26, 2)
                ## Add the column to matrix:
                cell_peaks_df['peak_center_avg_speed'] = avg_2D_speed_per_pos[_temp_peak_center_bin_label_xy[:,0], _temp_peak_center_bin_label_xy[:,1]] # array([33.2289, 33.7748, 35.9964, 0, 59.3887, 37.354, 16.1506, 0, 49.8418, 33.2289, 0, 14.9843, 33.8302, 29.8891, 37.354, 10.6905, 0, 33.7748, nan, 53.9505, 34.003, 33.7748, 22.7252, nan, 11.8898, 58.9018])                
            

            ## Filter the summits, compute velocities, etc:            
            filtered_summits_analysis_df, pf_peak_counts_map = _build_filtered_summits_analysis_results(active_pf_2D.xbin, active_pf_2D.ybin, active_pf_2D.xbin_labels, active_pf_2D.ybin_labels, cell_peaks_df, active_eloy_analysis, slice_level_multiplier=0.5, minimum_included_peak_height=minimum_included_peak_height, debug_print = debug_print)
            
            pf_peak_counts_map_blurred = uniform_filter(pf_peak_counts_map.astype('float'), size=uniform_blur_size, mode='constant')
            pf_peak_counts_map_blurred_gaussian = gaussian_filter(pf_peak_counts_map.astype('float'), sigma=gaussian_blur_sigma)
            pf_peak_counts_results = DynamicParameters(raw=pf_peak_counts_map, uniform_blurred=pf_peak_counts_map_blurred, gaussian_blurred=pf_peak_counts_map_blurred_gaussian)

            try:
                ## Add distance to boundary by computing the distance to the nearest never-occupied bin
                peak_nearest_directional_boundary_bins, peak_nearest_directional_boundary_displacements, peak_nearest_directional_boundary_distances = _compute_distances_from_peaks_to_boundary(active_pf_2D, filtered_summits_analysis_df, debug_print=debug_print)

                ## Add the output columns to the peaks dataframe:
                # Output Columns:
                # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate
                # ['peak_nearest_directional_boundary_bins', 'peak_nearest_directional_boundary_displacements', 'peak_nearest_directional_boundary_distances'] # combined tuple columns
                filtered_summits_analysis_df['peak_nearest_directional_boundary_bins'] = peak_nearest_directional_boundary_bins
                filtered_summits_analysis_df['peak_nearest_directional_boundary_displacements'] = peak_nearest_directional_boundary_displacements
                filtered_summits_analysis_df['peak_nearest_directional_boundary_distances'] = peak_nearest_directional_boundary_distances
                filtered_summits_analysis_df['nearest_directional_boundary_direction_idx'] = np.argmin(peak_nearest_directional_boundary_distances, axis=1) # an index [0,1,2,3] corresponding to the direction of travel to the nearest index. Corresponds to (down, up, left, right)
                filtered_summits_analysis_df['nearest_directional_boundary_direction_distance'] = np.min(peak_nearest_directional_boundary_distances, axis=1) # the distance in the minimal dimension towards the nearest boundary

                # ['peak_nearest_boundary_bin_negX', 'peak_nearest_boundary_bin_posX', 'peak_nearest_boundary_bin_negY', 'peak_nearest_boundary_bin_posY'] # separate
                distances = np.vstack([np.asarray(a_tuple) for a_tuple in peak_nearest_directional_boundary_distances])
                x_distances = np.min(distances[:,3:], axis=1) # find the distance to nearest wall vertically
                y_distances = np.min(distances[:,:2], axis=1) # find the distance to nearest wall horizontally

                filtered_summits_analysis_df['nearest_x_boundary_distance'] = x_distances # the distance in the minimal dimension towards the nearest x boundary
                filtered_summits_analysis_df['nearest_y_boundary_distance'] = y_distances # the distance in the minimal dimension towards the nearest y boundary
            except (BaseException, NotImplementedError) as err:
                print(f'could not find distances to nearest boundary in `ratemap_peaks_prominence2d`. Some columns will be missing from the output dataframe. Error: {err}')


            ## Build function output:
            if 'RatemapPeaksAnalysis' not in computation_result.computed_data:
                computation_result.computed_data['RatemapPeaksAnalysis'] = DynamicParameters() # get the existing RatemapPeaksAnalysis output or create a new one if needed
                            
            computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D'] = DynamicParameters(xx=active_pf_2D.xbin_centers, yy=active_pf_2D.ybin_centers, neuron_extended_ids=active_pf_2D.neuron_extended_ids, results=out_results,
                                                                                                             flat_peaks_df=cell_peaks_df, filtered_flat_peaks_df=filtered_summits_analysis_df, peak_counts=pf_peak_counts_results)

            return computation_result

    @function_attributes(short_name='placefield_overlap', tags=['overlap'],
        input_requires=["computation_result.computed_data['pf2D_Decoder']"],
        output_provides=["computation_result.computed_data['placefield_overlap']"],
        uses=[], used_by=[], creation_date='2023-09-12 17:20', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['placefield_overlap'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['placefield_overlap']['all_pairwise_overlaps']), is_global=False)
    def _perform_placefield_overlap_computation(computation_result: ComputationResult, debug_print=False):
            """ Computes the pairwise overlap between every pair of placefields. 
            
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

