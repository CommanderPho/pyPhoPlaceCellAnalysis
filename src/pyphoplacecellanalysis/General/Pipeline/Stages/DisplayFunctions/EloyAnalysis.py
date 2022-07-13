import numpy as np
import pandas as pd

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, pyqtplot_common_setup

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




    def _display_pf_peak_prominence2d_plots(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
            """ TODO
            """
            ## Now should have out_results
            def _plot_promenence_peaks(xx, yy, neuron_extended_ids, out_results, debug_print=False):
                """ Plots the promenence peaks
                                
                Usage:
                    
                    out_figs, out_axes, out_idxs = _plot_promenence_peaks(active_peak_prominence_2d_results.xx, active_peak_prominence_2d_results.yy, active_peak_prominence_2d_results.neuron_extended_ids, active_peak_prominence_2d_results.result_tuples, debug_print=False)
                """
                out_figs = []
                out_axes = []
                out_idxs = [] # the neuron_IDXs (not ids) corresponding to the actual output plots

                for i, a_result in enumerate(out_results):
                    # Test plot the promenence result
                    try:
                        slab, peaks, idmap, promap, parentmap = a_result # unwrap the result
                        figure, (ax1, ax2, ax3, ax4) = plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, debug_print=debug_print)
                        curr_neuron_id = neuron_extended_ids[i]
                        figure.suptitle(f'neuron: {curr_neuron_id.id}', fontsize=16)
                    except ValueError as e:
                        print(f'e: {e} for item {i}. Skipping')
                    else:
                        out_idxs.append(i)
                        out_figs.append(figure)
                        out_axes.append((ax1, ax2, ax3, ax4))
                return out_figs, out_axes, out_idxs


            def _plot_single_neuron_result():
                included_level_indicies=[1]
                promenence_plot_threshold = 1.0
                i = 0

                ## Create figure:
                figure = plt.figure(figsize=(12,10), dpi=100)
                ax = figure.add_subplot(1,1,1)

                extended_neuron_id = active_peak_prominence_2d_results.neuron_extended_ids[i]
                slab, peaks, idmap, promap, parentmap = active_peak_prominence_2d_results.result_tuples[i]
                neuron_id = extended_neuron_id.id
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
                        computed_contours = a_peak['computed_contours']
                        levels_list = list(computed_contours.keys())
                        if included_level_indicies is not None:
                            filtered_levels_list = [levels_list[i] for i in included_level_indicies]  
                        else:
                            filtered_levels_list = levels_list
                        
                        curr_color = next(colors)
                        peak_center = a_peak['center']
                        peak_height = a_peak['height']
                        
                        peak_locations[i,[0,1]] = a_peak['center']
                        peak_locations[i,2] = a_peak['height']
                        
                        if debug_print:
                            print(f"\tcenter: {peak_center}")
                            print(f"\theight: {peak_height}")
                        ax.scatter(peak_center[0], peak_center[1], color=curr_color) # probably just accumulate these
                        bboxes = a_peak['pf_bbox']
                        sizes = a_peak['pf_size']
                        # fig, ax = plot_computed_contours(computed_contours, bboxes, included_level_indicies=[1], figure=figure, ax1=ax)
                        for level_value in filtered_levels_list:
                            contour_list = computed_contours[level_value]
                            for a_contour in contour_list:
                                ax.plot(a_contour.vertices[:,0], a_contour.vertices[:,1],':', color=curr_color)
                            bbox = bboxes[level_value]
                            # Add the patch to the Axes
                            (x0, y0, width, height) = bbox.bounds
                            ax.add_patch(Rectangle((x0, y0), width, height, linewidth=1, edgecolor=curr_color, facecolor='none'))
                    else:
                        print(f'\tskipping because prominence too low. prominence: {prominence}')
                    
                    
            # ==================================================================================================================== #
            # Begin Function Body                                                                                                  #
            # ==================================================================================================================== #
            active_pf_2D = computation_result.computed_data['pf2D']
            active_peak_prominence_2d_results = computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D']
            
            
            out_figs, out_axes, out_idxs = _plot_promenence_peaks(active_peak_prominence_2d_results.xx, active_peak_prominence_2d_results.yy, active_peak_prominence_2d_results.neuron_extended_ids, active_peak_prominence_2d_results.result_tuples, debug_print=False)
            
            curr_display_function_name = 'plot_Prominence'
            built_pdf_metadata, curr_pdf_save_path = _build_pdf_pages_output_info(curr_display_function_name)
            with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=built_pdf_metadata) as pdf:
                for an_idx, a_fig in zip(active_peak_prominence_2d_results.neuron_extended_ids, out_figs):
                    a_fig.suptitle(f'neuron: {an_idx.id}', fontsize=16)
                    pdf.savefig(a_fig)
            
            
            
            ## 1D:
            ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
            out_plot_1D = pg.plot(active_eloy_analysis.sorted_1D_avg_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_1D, pen=None, symbol='o', title='Sorted 1D AVG Speed per Pos vs. Sorted 1D PFOverlapDensity', left='Sorted 1D PFOverlapDensity', bottom='Sorted 1D AVG Speed per Pos bin (x)') ## setting pen=None disables line drawing
            # out_plot_1D = root_render_widget.addPlot(row=curr_row, col=curr_col, name=curr_plot_identifier_string, title=curr_cell_identifier_string)
            
            ## 2D:
            ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
            out_plot_2D = pg.plot(active_eloy_analysis.sorted_avg_2D_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_2D, pen=None, symbol='o', title='Sorted AVG 2D Speed per Pos vs. Sorted 2D PFOverlapDensity', left='Sorted 2D PFOverlapDensity', bottom='Sorted AVG 2D Speed per Pos bin (x,y)') ## setting pen=None disables line drawing
            
            return out_plot_1D, out_plot_2D
            # return app, parent_root_widget, root_render_widget