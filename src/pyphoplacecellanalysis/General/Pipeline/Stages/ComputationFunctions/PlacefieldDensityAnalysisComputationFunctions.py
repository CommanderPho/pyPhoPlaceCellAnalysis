import sys
import numpy as np
import pandas as pd
from findpeaks import findpeaks # for _perform_pf_find_ratemap_peaks_computation. Install with pip install findpeaks 
from pyphoplacecellanalysis.External.peak_prominence2d import getProminence, plot_Prominence # Required for _perform_pf_find_ratemap_peaks_peak_prominence2d_computation

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




    def _perform_pf_find_ratemap_peaks_peak_prominence2d_computation(computation_result: ComputationResult, step=0.2, debug_print=False):
            """ Uses the peak_prominence2d package to find the peaks and promenences of 2D placefields
            
            Independent of the other peak-computing computation functions above
            
            Requires:
                computed_data['pf2D']
                
            Provides:
                computed_data['RatemapPeaksAnalysis']['PeakProminence2D']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['xx']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['yy']:
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['neuron_extended_ids']
                    
                    
                    computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['result_tuples']: (slab, peaks, idmap, promap, parentmap)
                    
                    
            """            
            

            def _perform_compute_prominence_contours(xbin_centers, ybin_centers, slab, step=0.2):
                """
                xbin_centers and ybin_centers should be like *bin_labels not *bin
                slab should usually be transposed: tuning_curves[i].T
                
                Usage:        
                    step = 0.2
                    i = 0
                    xx, yy, slab, peaks, idmap, promap, parentmap = perform_compute_prominence_contours(active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, active_pf_2D.ratemap.tuning_curves[i].T, step=step)
                    
                    # Test plot the promenence result
                    figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=False)

                """
                peaks, idmap, promap, parentmap = getProminence(slab, step, ybin_centers=ybin_centers, xbin_centers=xbin_centers, min_area=None, include_edge=True, verbose=False)
                return xbin_centers, ybin_centers, slab, peaks, idmap, promap, parentmap



            active_pf_2D = computation_result.computed_data['pf2D']
            n_neurons = active_pf_2D.ratemap.n_neurons
            
            out_result_tuples = []

            #  Build the results first:
            for i in np.arange(n_neurons):
                xx, yy, slab, peaks, idmap, promap, parentmap = _perform_compute_prominence_contours(active_pf_2D.xbin_labels, active_pf_2D.ybin_labels, active_pf_2D.ratemap.tuning_curves[i].T, step=step)
                out_result_tuples.append((slab, peaks, idmap, promap, parentmap))
    
            computation_result.computed_data.setdefault('RatemapPeaksAnalysis', DynamicParameters()) # get the existing RatemapPeaksAnalysis output or create a new one if needed
            computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D'] = DynamicParameters(xx=active_pf_2D.xbin_labels, yy=active_pf_2D.ybin_labels, neuron_extended_ids=active_pf_2D.neuron_extended_ids, result_tuples=out_result_tuples)
            return computation_result
