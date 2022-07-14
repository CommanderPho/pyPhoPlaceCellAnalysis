import sys
from warnings import warn
import numpy as np
import pandas as pd
from findpeaks import findpeaks # for _perform_pf_find_ratemap_peaks_computation. Install with pip install findpeaks 


import matplotlib
# configure backend here
matplotlib.use('Agg')
import matplotlib.pyplot as plt # required for _perform_pf_analyze_results_peak_prominence2d_computation to build Path objects. Nothing is plotted though


from pyphoplacecellanalysis.External.peak_prominence2d import compute_prominence_contours # Required for _perform_pf_find_ratemap_peaks_peak_prominence2d_computation

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


"""-------------- Specific Computation Functions to be registered --------------"""

modify_dict_mode = True # if True, writes the dict


            
            
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
        
        
        
    def _perform_pf_find_ratemap_peaks_computation(computation_result: ComputationResult, debug_print=False, peak_score_inclusion_percent_threshold=0.25):
            """ Builds the simplified density analysis suggested by Kamran at the 2022-07-06 lab meeting related analysis to test Eloy's Pf-Density/Velocity Hypothesis for 2D Placefields
            
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
            def ratemap_find_placefields(ratemap, debug_print=False):
                """ Uses the `findpeaks` library for finding local maxima of TuningMaps
                Input:
                ratemap: a 2D ratemap
                
                Returns:
                
                
                """
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
                
                
            def _filter_found_peaks_by_exclusion_threshold(df_list, peak_xy_points_pos_list, peak_score_inclusion_percent_threshold=0.25):
                peak_score_inclusion_threshold = peak_score_inclusion_percent_threshold * 255.0 # it must be at least 1/4 of the promenance of the largest peak (which is always 255.0)
                peaks_are_included_list = [(result_df['peak_score'] > peak_score_inclusion_threshold) for result_df in df_list]
                
                # inclusion_filter_function = lambda result_list: [a_result[peak_is_included, :] for a_result, peak_is_included in zip(result_list, peaks_are_included_list)]
                ## filter by the peaks_are_included_list:
                # filtered_df_list = [result_df[(result_df['peak_score'] > peak_score_inclusion_threshold)] for result_df in df_list]  # WORKS
                # filtered_df_list = [df_list[i][peak_is_included] for i, peak_is_included in enumerate(peaks_are_included_list)] # Also works
                # filtered_peak_xy_points_pos_list = [peak_xy_points_pos_list[i][:, peak_is_included] for i, peak_is_included in enumerate(peaks_are_included_list)]
                
                filtered_df_list = [a_df[peak_is_included] for a_df, peak_is_included in zip(df_list, peaks_are_included_list)] # Also works
                filtered_peak_xy_points_pos_list = [a_xy_points_pos[:, peak_is_included] for a_xy_points_pos, peak_is_included in zip(peak_xy_points_pos_list, peaks_are_included_list)]

                # print(f'peaks_are_included_list: {peaks_are_included_list}')
                # print(f'filtered_df_list: {filtered_df_list}')
                # print(f'filtered_peak_xy_points_pos_list: {filtered_peak_xy_points_pos_list}')

                return peaks_are_included_list, filtered_df_list, filtered_peak_xy_points_pos_list
                
                
            # active_pf_1D = computation_result.computed_data['pf1D']
            active_pf_2D = computation_result.computed_data['pf2D']
            # n_xbins = len(active_pf_2D.xbin) - 1 # the -1 is to get the counts for the centers only
            # n_ybins = len(active_pf_2D.ybin) - 1 # the -1 is to get the counts for the centers only
            # n_neurons = active_pf_2D.ratemap.n_neurons
            
            # peaks_outputs: fp_mask, mask_results_df, fp_topo, topo_results_df, topo_persistence_df
            # peaks_outputs = [ratemap_find_placefields(a_tuning_curve.copy(), debug_print=debug_print) for a_tuning_curve in active_pf_2D.ratemap.pdf_normalized_tuning_curves]
            # fp_mask_list, mask_results_df_list, fp_topo_list, topo_results_df_list, topo_persistence_df_list = tuple(zip(*findpeaks_results))
            fp_mask_list, mask_results_df_list, fp_topo_list, topo_results_df_list, topo_persistence_df_list = tuple(zip(*[ratemap_find_placefields(a_tuning_curve.copy(), debug_print=debug_print) for a_tuning_curve in active_pf_2D.ratemap.pdf_normalized_tuning_curves]))
            topo_results_peak_xy_pos_list = [np.vstack((active_pf_2D.xbin[curr_topo_result_df['xbin_idx'].to_numpy()], active_pf_2D.ybin[curr_topo_result_df['ybin_idx'].to_numpy()])) for curr_topo_result_df in topo_results_df_list]
            # peak_xy_pos_shapes = [np.shape(a_xy_pos) for a_xy_pos in topo_results_peak_xy_pos_list]

            peaks_are_included_list, filtered_df_list, filtered_peak_xy_points_pos_list = _filter_found_peaks_by_exclusion_threshold(topo_results_df_list, topo_results_peak_xy_pos_list, peak_score_inclusion_percent_threshold=peak_score_inclusion_percent_threshold)
            
            # computation_result.computed_data['RatemapPeaksAnalysis'] = DynamicParameters(tuning_curve_findpeaks_results=peaks_outputs)
            computation_result.computed_data['RatemapPeaksAnalysis'] = DynamicParameters(mask_results=DynamicParameters(fp_list=fp_mask_list, df_list=mask_results_df_list),
                                                                                         topo_results=DynamicParameters(fp_list=fp_topo_list, df_list=topo_results_df_list, persistence_df_list=topo_persistence_df_list, peak_xy_points_pos_list=topo_results_peak_xy_pos_list),
                                                                                         final_filtered_results=DynamicParameters(df_list=filtered_df_list, peak_xy_points_pos_list=filtered_peak_xy_points_pos_list, peaks_are_included_list=peaks_are_included_list)
                                                                                         )
            
            return computation_result


    def _perform_pf_find_ratemap_peaks_peak_prominence2d_computation(computation_result: ComputationResult, step=0.01, peak_height_multiplier_probe_levels = (0.5, 0.9), debug_print=False):
            """ Uses the peak_prominence2d package to find the peaks and promenences of 2D placefields
            
            Independent of the other peak-computing computation functions above
            
            Inputs:
                peak_height_multiplier_probe_levels = (0.5, 0.9) # 50% and 90% of the peak height
                
            Requires:
                computed_data['pf2D']
                
            Provides:
                computed_data['RatemapPeaksAnalysis']['PeakProminence2D']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['xx']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['yy']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['neuron_extended_ids']
                    
                    
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['result_tuples']: (slab, peaks, idmap, promap, parentmap)
                    
                    flat_peaks_df
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


            # begin main function body ___________________________________________________________________________________________ #
            active_pf_2D = computation_result.computed_data['pf2D']
            n_neurons = active_pf_2D.ratemap.n_neurons
            
            ## TODO: change to amap with keys of active_pf_2D.neuron_ids
            # out_result_tuples = []
            # active_tuning_curves = active_pf_2D.ratemap.tuning_curves # Raw Tuning Curves
            active_tuning_curves = active_pf_2D.ratemap.unit_max_tuning_curves # Unit-max scaled tuning curves
            #  Build the results:
            out_results = {}
            out_cell_peak_dfs_list = []
            n_slices = len(peak_height_multiplier_probe_levels)
            

            for neuron_idx in np.arange(n_neurons):
                neuron_id = active_pf_2D.neuron_extended_ids[neuron_idx].id               
                slab = active_tuning_curves[neuron_idx].T
                _, _, slab, cell_peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(xbin_centers=active_pf_2D.xbin_centers, ybin_centers=active_pf_2D.ybin_centers, slab=slab, step=step, min_area=None, min_depth=0.2, include_edge=True, verbose=False)
                #""" Analyze all peaks of a given cell/ratemap """
                n_peaks = len(cell_peaks_dict)
                
                ## Neuron:
                n_total_cell_slice_results = n_slices * n_peaks                
                neuron_id_arr = np.full((n_total_cell_slice_results,), neuron_id) # repeat the neuron_id many times for the datatable
                
                ## Peak
                summit_slice_peak_id_arr = np.zeros((n_peaks, n_slices)) # same summit/peak id for all in the slice
                summit_slice_peak_level_multiplier_arr = np.zeros((n_peaks, n_slices), dtype=np.float16) # same summit/peak id for all in the slice
                summit_slice_peak_level_arr = np.zeros((n_peaks, n_slices), dtype=np.float16) # same summit/peak id for all in the slice
                summit_slice_peak_height_arr = np.zeros((n_peaks, n_slices), dtype=np.float16) # same summit/peak id for all in the slice
                summit_slice_peak_prominence_arr = np.zeros((n_peaks, n_slices), dtype=np.float16) # same summit/peak id for all in the slice
                summit_peak_center_x_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)
                summit_peak_center_y_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)
                
                
                ## Slice
                summit_slice_idx_arr = np.tile(np.arange(n_slices), n_peaks) # array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                summit_slice_x_side_length_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)
                summit_slice_y_side_length_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)
                summit_slice_center_x_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)
                summit_slice_center_y_arr = np.zeros((n_peaks, n_slices), dtype=np.float16)

                for peak_idx, (peak_id, a_peak_dict) in enumerate(cell_peaks_dict.items()):
                    if debug_print:
                        print(f'computing contours for peak_id: {peak_id}...')                    
                    
                    summit_slice_peak_height_arr[peak_idx, :] = a_peak_dict['height']
                    summit_slice_peak_prominence_arr[peak_idx, :] = a_peak_dict['prominence']
                    summit_peak_center_x_arr[peak_idx, :] = a_peak_dict['center'][0]
                    summit_peak_center_y_arr[peak_idx, :] = a_peak_dict['center'][1]
                    
                    
                    
                    ## This is where we would loop through each desired slice/probe levels:
                    a_peak_dict['probe_levels'] = np.array([a_peak_dict['height']*multiplier for multiplier in peak_height_multiplier_probe_levels]).astype('float') # specific probe levels
                    summit_slice_peak_level_arr[peak_idx, :] = np.array(peak_height_multiplier_probe_levels).astype('float')
                    summit_slice_peak_level_multiplier_arr[peak_idx, :] = a_peak_dict['probe_levels']
                    included_computed_contours = _find_contours_at_levels(active_pf_2D.xbin_centers, active_pf_2D.ybin_centers, slab, a_peak_dict['center'], a_peak_dict['probe_levels']) # DONE: efficiency: This would be more efficient to do for all peaks at once I believe. CONCLUSION: No, this needs to be done separately for each peak as they each have separate prominences which determine the levels they should be sliced at.
                    ## Build the dict that contains the output level slices
                    a_peak_dict['level_slices'] = {probe_lvl:{'contour':contour, 'bbox':contour.get_extents(), 'size':contour.get_extents().size} for probe_lvl, contour in included_computed_contours.items() if (contour is not None)} # 

                    if debug_print:
                        print(f"probe_levels: {a_peak_dict['probe_levels']}")

                    ## Build flat output:
                    
                    for lvl_idx, probe_lvl in enumerate(a_peak_dict['probe_levels']):
                        a_slice = a_peak_dict['level_slices'][probe_lvl]
                        slice_bbox = a_slice['bbox']
                        (x0, y0, width, height) = slice_bbox.bounds        
                        summit_slice_peak_id_arr[peak_idx, lvl_idx] = peak_id
                        summit_slice_x_side_length_arr[peak_idx, lvl_idx] = width
                        summit_slice_y_side_length_arr[peak_idx, lvl_idx] = height
                        # summit_slice_center_x_arr[peak_idx, lvl_idx] = slice_bbox.center[0]
                        # summit_slice_center_y_arr[peak_idx, lvl_idx] = slice_bbox.center[1]
                        summit_slice_center_x_arr[peak_idx, lvl_idx] = float(x0) + (0.5 * float(width))
                        summit_slice_center_y_arr[peak_idx, lvl_idx] = float(y0) + (0.5 * float(height))
                        
                    
                # if debug_print:
                print(f'building peak_df for neuron[{neuron_idx}] with {n_peaks}...')    
                cell_peaks_df = pd.DataFrame({'neuron_id': neuron_id_arr, 'summit_idx': summit_slice_peak_id_arr.flatten(), 'summit_slice_idx': summit_slice_idx_arr.flatten(),
                                             'slice_level_multiplier': summit_slice_peak_level_multiplier_arr.flatten(), 'summit_slice_level': summit_slice_peak_level_arr.flatten(),
                                             'peak_height': summit_slice_peak_height_arr.flatten(), 'peak_prominence': summit_slice_peak_prominence_arr.flatten(),
                                             'peak_center_x': summit_peak_center_x_arr.flatten(), 'peak_center_y': summit_peak_center_y_arr.flatten(),
                                             'summit_slice_x_width': summit_slice_x_side_length_arr.flatten(), 'summit_slice_y_width': summit_slice_y_side_length_arr.flatten(),
                                             'summit_slice_center_x': summit_slice_center_x_arr.flatten(), 'summit_slice_center_y': summit_slice_center_y_arr.flatten()
                                             })
                out_cell_peak_dfs_list.append(cell_peaks_df)
                
                                           
                if debug_print:
                    print(f'done.') # END Analyze peaks
                    
                out_results[neuron_id] = {'peaks': cell_peaks_dict, 'slab': slab, 'id_map':id_map, 'prominence_map':prominence_map, 'parent_map':parent_map} 
    
            # Build final concatenated dataframe:
            print(f'building final concatenated cell_peaks_df for {n_neurons} total neurons...')
            cell_peaks_df = pd.concat(out_cell_peak_dfs_list)

            ## Build function output:
            computation_result.computed_data.setdefault('RatemapPeaksAnalysis', DynamicParameters()) # get the existing RatemapPeaksAnalysis output or create a new one if needed
            computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D'] = DynamicParameters(xx=active_pf_2D.xbin_centers, yy=active_pf_2D.ybin_centers, neuron_extended_ids=active_pf_2D.neuron_extended_ids, results=out_results, flat_peaks_df=cell_peaks_df)
            
            return computation_result




