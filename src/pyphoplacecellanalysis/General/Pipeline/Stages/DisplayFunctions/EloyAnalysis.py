import numpy as np
import pandas as pd


## All matplotlib-related stuff is for _display_pf_peak_prominence2d_plots
import matplotlib
# configure backend here
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm



from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.External.peak_prominence2d import plot_Prominence # required for _plot_promenence_peaks

class EloyAnalysisDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing results related to Pho's 2022 Analysis of Placefield Density and Animal Speed for Eloy """
    
    # def _display_speed_vs_PFoverlapDensity_plots(computation_result, active_config, enable_saving_to_disk=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, **kwargs):
    def _display_speed_vs_PFoverlapDensity_plots(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
        """ Plot the 1D and 2D sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends
        """
        active_eloy_analysis = computation_result.computed_data.get('EloyAnalysis', None)
        # root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'_display_speed_vs_PFoverlapDensity_plots', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget)

        ## 1D:
        ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
        out_plot_1D = pg.plot(active_eloy_analysis.sorted_1D_avg_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_1D, pen=None, symbol='o', title='Sorted 1D AVG Speed per Pos vs. Sorted 1D PFOverlapDensity', left='Sorted 1D PFOverlapDensity', bottom='Sorted 1D AVG Speed per Pos bin (x)') ## setting pen=None disables line drawing
        # out_plot_1D = root_render_widget.addPlot(row=curr_row, col=curr_col, name=curr_plot_identifier_string, title=curr_cell_identifier_string)
        
        ## 2D:
        ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
        out_plot_2D = pg.plot(active_eloy_analysis.sorted_avg_2D_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_2D, pen=None, symbol='o', title='Sorted AVG 2D Speed per Pos vs. Sorted 2D PFOverlapDensity', left='Sorted 2D PFOverlapDensity', bottom='Sorted AVG 2D Speed per Pos bin (x,y)') ## setting pen=None disables line drawing
        
        return out_plot_1D, out_plot_2D
        # return app, parent_root_widget, root_render_widget




    def _display_pf_peak_prominence2d_default_quadrant_plots(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
            """ Plots the 4-quadrant figure generated by default from peak_rpominence2d to show the found prominence peaks
            
            Usage:
                curr_display_function_name = '_display_pf_peak_prominence2d_plots'
                out_figs, out_axes, out_idxs = curr_active_pipeline.display(curr_display_function_name, active_config_name) 
                curr_display_function_name = 'plot_Prominence'
                built_pdf_metadata, curr_pdf_save_path = _build_pdf_pages_output_info(curr_display_function_name)
                with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=built_pdf_metadata) as pdf:
                    for an_idx, a_fig in zip(active_peak_prominence_2d_results.neuron_extended_ids, out_figs):
                        a_fig.suptitle(f'neuron: {an_idx.id}', fontsize=16)
                        pdf.savefig(a_fig)
            """
            ## Now should have out_results
            def _plot_prominence_peaks(xx, yy, out_results, n_contour_levels=5, debug_print=False):
                """ Plots the 4-quadrant figure generated by default from peak_rpominence2d to show the found prominence peaks
                                
                Usage:
                    
                    out_figs, out_axes, out_idxs = _plot_promenence_peaks(active_peak_prominence_2d_results.xx, active_peak_prominence_2d_results.yy, active_peak_prominence_2d_results.neuron_extended_ids, active_peak_prominence_2d_results.result_tuples, debug_print=False)
                """
                out_figs = []
                out_axes = []
                out_idxs = [] # the neuron_IDXs (not ids) corresponding to the actual output plots

                # for i, a_result in enumerate(out_results):
                for curr_neuron_id, a_result in out_results.items():
                    # Test plot the promenence result
                    try:
                        figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, a_result['slab'], a_result['peaks'], a_result['id_map'], a_result['prominence_map'], a_result['parent_map'], n_contour_levels=n_contour_levels, debug_print=debug_print)
                        figure.suptitle(f'neuron: {curr_neuron_id}', fontsize=16)
                    except ValueError as e:
                        print(f'e: {e} for neuron_id: {curr_neuron_id}. Skipping')
                    else:
                        out_idxs.append(curr_neuron_id)
                        out_figs.append(figure)
                        out_axes.append((ax1, ax2, ax3, ax4))
                return out_figs, out_axes, out_idxs

            # ==================================================================================================================== #
            # Begin Function Body                                                                                                  #
            # ==================================================================================================================== #
            active_peak_prominence_2d_results = computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D']            
            ## Simple _plot_prominence_peaks
            n_contour_levels = kwargs.get('n_contour_levels', 9)
            out_figs, out_axes, out_idxs = _plot_prominence_peaks(active_peak_prominence_2d_results.xx, active_peak_prominence_2d_results.yy, active_peak_prominence_2d_results.results, n_contour_levels=n_contour_levels, debug_print=debug_print)
            return out_figs, out_axes, out_idxs
        
            
            
    def _display_pf_peak_prominence2d_plots(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
            """ Plots a the custom placefield width/height results for each peak belonging to a single neuron/ratemap.
                Uses the prominence2d results
            Usage:
                curr_display_function_name = '_display_pf_peak_prominence2d_plots'
                figure, ax = curr_active_pipeline.display(curr_display_function_name, active_config_name, neuron_id=5) 

            """
            def _plot_single_neuron_result(neuron_id, a_result, promenence_plot_threshold = 1.0, included_level_indicies=[1], debug_print=False):
                ## Create figure:
                figure = plt.figure(figsize=(12,10), dpi=100)
                ax = figure.add_subplot(1,1,1)
                slab, peaks, idmap, promap, parentmap = a_result['slab'], a_result['peaks'], a_result['id_map'], a_result['prominence_map'], a_result['parent_map']
                
                n_peaks = len(peaks)
                if debug_print:
                    print(f'neruon_id: {neuron_id} - : {n_peaks} peaks:')
                colors = iter(cm.rainbow(np.linspace(0, 1, n_peaks)))

                peak_locations = np.empty((n_peaks, 3), dtype=float) # x, y, z for each peak
                prominence_array = np.empty((n_peaks,), dtype=float)
                is_included_array = np.empty((n_peaks,), dtype=bool)

                for i, (peak_id, a_peak) in enumerate(peaks.items()):
                    # loop through each of the peaks and plot them
                    if debug_print:
                        print(f'peak_id: {peak_id}')
                    prominence = a_peak['prominence']
                    
                    if prominence >= promenence_plot_threshold:
                        if debug_print:
                            print(f'\tprominence: {prominence}')
                            print(f'\t# contours: {len(computed_contours)}')
                        
                        curr_slices = a_peak['level_slices']
                        # print(list(curr_slices.keys())) # [4.680000000000001, 2.6]
                        levels_list = list(curr_slices.keys())
                        if included_level_indicies is not None:
                            filtered_levels_list = [levels_list[i] for i in included_level_indicies]  
                        else:
                            filtered_levels_list = levels_list
                        
                        # computed_contours = a_peak['computed_contours']
                        
                        curr_color = next(colors)
                        peak_center = a_peak['center']
                        peak_height = a_peak['height']                        
                        peak_locations[i,[0,1]] = a_peak['center']
                        peak_locations[i,2] = a_peak['height']
                        
                        if debug_print:
                            print(f"\tcenter: {peak_center}")
                            print(f"\theight: {peak_height}")
                        ax.scatter(peak_center[0], peak_center[1], color=curr_color) # probably just accumulate these                                                
                        for level_value in filtered_levels_list:
                            curr_slice = curr_slices[level_value]    
                            curr_contour = curr_slice['contour']
                            if curr_contour is not None:
                                ax.plot(curr_contour.vertices[:,0], curr_contour.vertices[:,1],':', color=curr_color)
                                bbox = curr_slice['bbox']
                                (x0, y0, width, height) = bbox.bounds
                                # Add the patch to the Axes
                                ax.add_patch(Rectangle((x0, y0), width, height, linewidth=1, edgecolor=curr_color, facecolor='none'))
                            else:
                                print(f"contour missing for neuron_id: {neuron_id} - peak_id: {peak_id} - slice[{level_value}]. Skipping.")
                    else:
                        print(f'\tskipping neuron_id: {neuron_id} - peak_id: {peak_id} because prominence: {prominence} is too low.')
                    
                return figure, ax
                    
            # ==================================================================================================================== #
            # Begin Function Body                                                                                                  #
            # ==================================================================================================================== #
            # active_pf_2D = computation_result.computed_data['pf2D']
            active_peak_prominence_2d_results = computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D']
            
            valid_neuron_id = kwargs.get('neuron_id', 2)
            assert valid_neuron_id in active_peak_prominence_2d_results.results, f"neuron_id {valid_neuron_id} must be in the results keys, but it is not. results keys: {list(active_peak_prominence_2d_results.results.keys())}"
            promenence_plot_threshold = kwargs.get('promenence_plot_threshold', 0.2)
            figure, ax = _plot_single_neuron_result(valid_neuron_id, active_peak_prominence_2d_results.results[valid_neuron_id], promenence_plot_threshold=promenence_plot_threshold, included_level_indicies=[1], debug_print=debug_print)
            return figure, ax