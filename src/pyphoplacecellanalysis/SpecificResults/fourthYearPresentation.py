""" 
Contains code related to Pho Hale's 4th Year PhD Presentation on 2023-09-25

"""
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import TruncationCheckingResults
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult


from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_short_v_long_pf1D_comparison
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines, add_track_shapes
from neuropy.plotting.ratemaps import plot_ratemap_1D
from neuropy.core.epoch import Epoch
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

from neuropy.utils.matplotlib_helpers import draw_epoch_regions

from neuropy.utils.matplotlib_helpers import FormattedFigureText
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_long_short_surprise_difference_plot, plot_long_short, plot_long_short_any_values
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

long_short_display_config_manager = LongShortDisplayConfigManager()
long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

from flexitext import flexitext ## flexitext for formatted matplotlib text
from flexitext.parser.make_texts import make_texts
# from flexitext.flexitext.text import Text
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
from neuropy.utils.matplotlib_helpers import FormattedFigureText

@function_attributes(short_name=None, tags=['display'], input_requires=[], output_provides=[], uses=['_display_long_short_pf1D_comparison'], used_by=[], creation_date='2024-06-10 21:40', related_items=[])
def fig_example_nontopo_remap(curr_active_pipeline):
    """Specific Figure: Example of non-neighbor preserving remapping
    Usage:
        from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_example_nontopo_remap

        graphics_output_dict = fig_example_nontopo_remap(curr_active_pipeline)
    """
    example_aclus = [7, 38] # 95 was BAAAAD
    # # flat_stack_mode: all placefields are stacked up (z-wise) on top of each other on a single axis with no offsets:
    example_shared_kwargs = dict(pad=1, active_context=curr_active_pipeline.get_session_context(), plot_zero_baselines=True, skip_figure_titles=True, use_flexitext_titles=True, flat_stack_mode=True)
    example_top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=True, should_plot_linear_track_shapes=True) # Renders the linear track shape on the maze. Assumes `flat_stack_mode=True`

    # (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, example_aclus, reuse_axs_tuple=None, single_figure=True, title_string="Example Non-Neighbor Preserving Remapping Cells", subtitle_string=f"2 Example Cells {example_aclus}", shared_kwargs=example_shared_kwargs, **example_top_level_shared_kwargs)
    return curr_active_pipeline.display('_display_long_short_pf1D_comparison', curr_active_pipeline.get_session_context(), included_any_context_neuron_ids=example_aclus, reuse_axs_tuple=None, single_figure=True, title_string="Example Non-Neighbor Preserving Remapping Cells", subtitle_string=f"2 Example Cells {example_aclus}", shared_kwargs=example_shared_kwargs, **example_top_level_shared_kwargs)
     

@function_attributes(short_name=None, tags=['display'], input_requires=[], output_provides=[], uses=['_display_long_short_pf1D_comparison'], used_by=[], creation_date='2024-06-10 21:39', related_items=[])
def fig_remapping_cells(curr_active_pipeline, **kwargs):
    """

    from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_remapping_cells

    graphics_output_dict = graphics_output_dict | fig_remapping_cells(curr_active_pipeline)


    """
    # Extract Results to Display _________________________________________________________________________________________ #
    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
    # (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    ## long_short_endcap_analysis:
    truncation_checking_result: TruncationCheckingResults = curr_active_pipeline.global_computation_results.computed_data.long_short_endcap

    disappearing_endcap_aclus = truncation_checking_result.disappearing_endcap_aclus
    disappearing_endcap_aclus

    trivially_remapping_endcap_aclus = truncation_checking_result.minor_remapping_endcap_aclus
    trivially_remapping_endcap_aclus

    significant_distant_remapping_endcap_aclus = truncation_checking_result.significant_distant_remapping_endcap_aclus
    significant_distant_remapping_endcap_aclus

    appearing_aclus = jonathan_firing_rate_analysis_result.neuron_replay_stats_df[jonathan_firing_rate_analysis_result.neuron_replay_stats_df['track_membership'] == SplitPartitionMembership.RIGHT_ONLY].index
    appearing_aclus


    active_context = curr_active_pipeline.get_session_context()

    # Display: ___________________________________________________________________________________________________________ #
    curr_active_pipeline.reload_default_display_functions()

    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
    short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

    long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
    short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

    shared_kwargs = dict(pad=1, cmap='hsv', active_context=active_context, plot_zero_baselines=True, skip_figure_titles=True, use_flexitext_titles=True, flat_stack_mode=False)
    top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=True, sortby='peak_long')
    # top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=False, should_plot_linear_track_shapes=True) # Renders the linear track shape on the maze. Assumes `flat_stack_mode=True`

    use_flexitext_titles = False

    # # flat_stack_mode: all placefields are stacked up (z-wise) on top of each other on a single axis with no offsets:
    # shared_kwargs = dict(pad=1, active_context=curr_active_pipeline.get_session_context(), plot_zero_baselines=True, skip_figure_titles=True, use_flexitext_titles=True, flat_stack_mode=True)
    # top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=False, should_plot_linear_track_shapes=True) # Renders the linear track shape on the maze. Assumes `flat_stack_mode=True`

    # graphics_output_dict = {}

    # long_results = curr_active_pipeline.computation_results['maze1_PYR'].computed_data
    # short_results = curr_active_pipeline.computation_results['maze2_PYR'].computed_data
    # curr_any_context_neurons = _find_any_context_neurons(*[curr_active_pipeline.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in ['maze1_PYR', 'maze2_PYR']])

    if active_context is not None:
        display_context = active_context.adding_context('display_fn', display_fn_name='fig_remapping_cells')
        
    perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: curr_active_pipeline.output_figure(final_context, fig)))
    graphics_output_dict: Dict[str, MatplotlibRenderPlots] = {}

    with mpl.rc_context({'figure.figsize': (12.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42,
                          "axes.spines.left": False, "axes.spines.right": False, "axes.spines.bottom": False, "axes.spines.top": False,
                          "axes.edgecolor": "none", "xtick.bottom": False, "xtick.top": False, "ytick.left": False, "ytick.right": False}):
        # Create a FigureCollector instance
        with FigureCollector(name='fig_remapping_cells', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, out_container: MatplotlibRenderPlots, sub_context, title=f'<size:22>Track <weight:bold>Remapping</></>'):
                """ Captures:

                t_split
                """
                a_collector.contexts.append(sub_context)
                
                fig = out_container.figures[0]
                ax = out_container.axes[0]
            
                # title_string = f'1D Placemaps {title_substring}'
                # subtitle_string = f'({len(ratemap.neuron_ids)} good cells)'

                # 

                if use_flexitext_titles:
                    perform_update_title_subtitle(fig=fig, ax=ax, title_string=None, subtitle_string=title, active_context=out_container.context, use_flexitext_titles=False)

                    # `flexitext` version:
                    text_formatter = FormattedFigureText()
                    # ax.set_title('')
                    fig.suptitle('')
                    text_formatter.setup_margins(fig)
                    title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin,
                                            title,
                                            va="bottom", xycoords="figure fraction")
                    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                                                text_formatter._build_footer_string(active_context=sub_context),
                                                va="top", xycoords="figure fraction")
                    
                else:
                    ## strip text:
                    if title is not None:
                        ## strip formatting to convert back to plaintext string:
                        title_texts_obj: List = make_texts(title) # Text
                        plaintext_title: str = ''.join([v.string for v in title_texts_obj])
                        # plaintext_title: str = make_plaintext_string(formatted_string=title)
                    # if subtitle_string is not None:
                    #     title_texts_obj = make_texts(subtitle_string)
                    #     plaintext_subtitle: str = title_texts_obj.string
                    print(f'plaintext_title: {plaintext_title}')
                    perform_update_title_subtitle(fig=fig, ax=ax, title_string=plaintext_title, subtitle_string=None, active_context=out_container.context, use_flexitext_titles=True)

                collector.post_hoc_append(figs=out_container.figures, axes=out_container.axes, contexts=out_container.context)

                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)


        ## outputs are all `MatplotlibRenderPlots`
        max_num_cells: int = int(max(len(disappearing_endcap_aclus), max(len(significant_distant_remapping_endcap_aclus), len(trivially_remapping_endcap_aclus))))
        print(f'max_num_cells: {max_num_cells}')

        #TODO 2024-06-10 21:57: - [ ] Want the outputs to be a fixed height


    
        if len(disappearing_endcap_aclus) > 0:
            percent_max_height = float(len(disappearing_endcap_aclus)) / float(max_num_cells)
            with mpl.rc_context({'figure.figsize': ((12.4*percent_max_height), 4.8)}):
                # (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, disappearing_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Disappearing Cells", subtitle_string=None, **top_level_shared_kwargs)
                graphics_output_dict['disappearing_endcap_aclus'] = curr_active_pipeline.display('_display_long_short_pf1D_comparison',active_context.adding_context_if_missing(cell_subset='disappear_endcap'), included_any_context_neuron_ids=disappearing_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Disappearing Cells", subtitle_string=None, **top_level_shared_kwargs)
                # perform_update_title_subtitle(fig=fig, ax=ax_RL, title_string=None, subtitle_string=f"RL Track Remapping - {len(RL_only_decoder_aclu_MAX_peak_maps_df)} neurons")
                setup_common_after_creation(collector, out_container=graphics_output_dict['disappearing_endcap_aclus'], sub_context=display_context.adding_context('subplot', subplot_name='Disappearing Cells'), 
                                            title=f'<size:22>Remapping: <weight:bold>Disappearing</> cells</>')
        

        if len(significant_distant_remapping_endcap_aclus) > 0:
            percent_max_height = float(len(significant_distant_remapping_endcap_aclus)) / float(max_num_cells)
            with mpl.rc_context({'figure.figsize': ((12.4*percent_max_height), 4.8)}):
                # (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, significant_distant_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Significant Distance Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
                graphics_output_dict['significant_distant_remapping_endcap_aclus'] = curr_active_pipeline.display('_display_long_short_pf1D_comparison',active_context.adding_context_if_missing(cell_subset='sig_remap_endcap'), included_any_context_neuron_ids=significant_distant_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Significant Distance Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
                setup_common_after_creation(collector, out_container=graphics_output_dict['significant_distant_remapping_endcap_aclus'], sub_context=display_context.adding_context('subplot', subplot_name='Significantly Remapping Cells'), 
                                            title=f'<size:22>Remapping: <weight:bold>Significant Distance</> cells</>')
            
        if len(trivially_remapping_endcap_aclus) > 0:
            percent_max_height = float(len(trivially_remapping_endcap_aclus)) / float(max_num_cells)
            with mpl.rc_context({'figure.figsize': ((12.4*percent_max_height), 4.8)}):
                # (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, trivially_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Trivially Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
                graphics_output_dict['trivially_remapping_endcap_aclus'] = curr_active_pipeline.display('_display_long_short_pf1D_comparison', active_context.adding_context_if_missing(cell_subset='triv_remap_endcap'), included_any_context_neuron_ids=trivially_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Trivially Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
                setup_common_after_creation(collector, out_container=graphics_output_dict['trivially_remapping_endcap_aclus'], sub_context=display_context.adding_context('subplot', subplot_name='Simple Remapping Cells'), 
                                            title=f'<size:22>Remapping: <weight:bold>Simple Translation</> cells</>')
                
    return collector
    # return graphics_output_dict


def fig_example_handpicked_pho_jonathan_active_set_cells(curr_active_pipeline, save_figure=False, included_LxC_example_neuron_IDs=[4, 58], included_SxC_example_neuron_IDs=[2]):
    # 2023-09-07 - Build Example LxC/SxC cells from handpicked examples: aclus = [4, 58]
    # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import build_extra_cell_info_label_string
    curr_active_pipeline.reload_default_display_functions()
    _LxC_out = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', n_max_plot_rows=2, save_figure=save_figure, included_unit_neuron_IDs=included_LxC_example_neuron_IDs, active_context=curr_active_pipeline.get_session_context().adding_context_if_missing(example='short_exclusive')) # , included_unit_neuron_IDs=[4, 58]
    _SxC_out = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', n_max_plot_rows=2, save_figure=save_figure, included_unit_neuron_IDs=included_SxC_example_neuron_IDs, active_context=curr_active_pipeline.get_session_context().adding_context_if_missing(example='long_exclusive')) # handpicked long-exclusive
    return _LxC_out, _SxC_out




def fig_surprise_results(curr_active_pipeline):
    """ 2023-09-10 - Plots the flat_jensen_Shannon Distance across all positions over time
    

    from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_surprise_results


    """
    def _helper_prepare_epoch_df_for_draw_epoch_regions(active_filter_epochs) -> Epoch:
        """	Prepare active_filter_epochs:
            
        Usage:
            active_filter_epochs = curr_active_pipeline.sess.replay
            active_filter_epoch_obj: Epoch = _helper_prepare_epoch_df_for_draw_epoch_regions(active_filter_epochs)
        """
        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        return active_filter_epoch_obj


    active_context = curr_active_pipeline.get_session_context().adding_context('display_fn', display_fn_name='fig_surprise_results')
    

    # epoch_region_facecolor=('red','cyan')
    epoch_region_facecolor=[a_kwargs['facecolor'] for a_kwargs in (long_epoch_config, short_epoch_config)]
    

    # Prepare active_filter_epochs:
    active_filter_epochs = curr_active_pipeline.sess.replay
    active_filter_epoch_obj: Epoch = _helper_prepare_epoch_df_for_draw_epoch_regions(active_filter_epochs) # these are the replays!
    
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_results = curr_active_pipeline.computation_results[global_epoch_name].computed_data
    active_extended_stats = global_results['extended_stats']
    active_relative_entropy_results = active_extended_stats['pf_dt_sequential_surprise']
    post_update_times = active_relative_entropy_results['post_update_times'] # (4152,) = (n_post_update_times,)
    snapshot_differences_result_dict = active_relative_entropy_results['snapshot_differences_result_dict']
    time_intervals = active_relative_entropy_results['time_intervals']
    long_short_rel_entr_curves_frames = active_relative_entropy_results['long_short_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
    short_long_rel_entr_curves_frames = active_relative_entropy_results['short_long_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
    flat_relative_entropy_results = active_relative_entropy_results['flat_relative_entropy_results'] # (149, 63) - (nSnapshots, nXbins)
    flat_jensen_shannon_distance_results = active_relative_entropy_results['flat_jensen_shannon_distance_results'] # (149, 63) - (nSnapshots, nXbins)
    flat_jensen_shannon_distance_across_all_positions = np.sum(np.abs(flat_jensen_shannon_distance_results), axis=1) # sum across all position bins # (4152,) - (nSnapshots)
    flat_surprise_across_all_positions = np.sum(np.abs(flat_relative_entropy_results), axis=1) # sum across all position bins # (4152,) - (nSnapshots)



    def plot_data_and_epochs(x_data, y_data, xlabel, ylabel, title, epochs, laps_epochs, filter_epochs, epoch_region_facecolor, defer_render=True, debug_print=False, save_figure=True):
        final_context = active_context.adding_context('title', title=title)
        print(f'final_context: {final_context}')


        fig, ax = plt.subplots(figsize=(16, 3), dpi=120) # fignum=str(final_context)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        time_range = (np.nanmin(x_data), np.nanmax(x_data))
        ax.plot(x_data, y_data, label=title, zorder=100) # plot line in front

        lap_labels_kwargs = None
        # lap_labels_kwargs = {'y_offset': -16.0, 'size': 8}
        
        epochs_collection, epoch_labels = draw_epoch_regions(epochs, ax, facecolor=epoch_region_facecolor, alpha=0.1, 
                        edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 12}, 
                        defer_render=defer_render, debug_print=debug_print, zorder=-20)

        
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(laps_epochs, ax, facecolor='#33FF00', 
                        edgecolors=None,#'black', 
                        labels_kwargs=lap_labels_kwargs, 
                        defer_render=defer_render, debug_print=debug_print, zorder=-10)
        
        track_epochs_collection, track_epoch_labels = draw_epoch_regions(filter_epochs, ax, facecolor='orange', edgecolors=None, 
                        labels_kwargs=None, defer_render=defer_render, debug_print=debug_print, zorder=-9)

        ax.set_xlim(*time_range)
        fig.suptitle(title)
        plt.subplots_adjust(top=0.847, bottom=0.201, left=0.045, right=0.972, hspace=0.2, wspace=0.2)
        fig.show()
        # fig.save

        
        
        def _perform_write_to_file_callback():
            ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            return curr_active_pipeline.output_figure(final_context, fig)


        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='fig_surprise_results', figures=(fig), axes=(ax), plot_data={}, context=final_context) # saved_figures=active_out_figure_paths
        # graphics_output_dict['plot_data'] = {'included_any_context_neuron_ids': included_any_context_neuron_ids, 'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

        return graphics_output_dict

    
    # Change the two track epoch labels to ['maze1', 'maze2'] before preceeding:
    track_epochs = curr_active_pipeline.sess.epochs
    assert len(track_epochs._df['label']) == 2, f"labels should be ['maze1', 'maze2']!"
    track_epochs._df['label'] = ['long', 'short'] # access private methods to set the proper labels
    # track_epochs

    laps_epochs = curr_active_pipeline.sess.laps.as_epoch_obj()
    filter_epochs = active_filter_epoch_obj

    graphics_outputs = [
            # plot_data_and_epochs(post_update_times, flat_surprise_across_all_positions, 
            # 			't (seconds)', 'Relative Entropy across all positions', 
            # 			'flat_surprise_across_all_positions', 
            # 			track_epochs, laps_epochs, filter_epochs, epoch_region_facecolor),
            plot_data_and_epochs(post_update_times, flat_jensen_shannon_distance_across_all_positions, 
                        't (seconds)', 'J-S Distance across all positions', 
                        'flat_jensen_shannon_distance_across_all_positions', 
                        track_epochs, laps_epochs, filter_epochs, epoch_region_facecolor),
            # plot_data_and_epochs(post_update_times, flat_relative_entropy_results, 
            # 					't (seconds)', 'Relative Entropy', 
            # 					'Relative Entropy vs Time', 
            # 					track_epochs, laps_epochs, filter_epochs, epoch_region_facecolor, defer_render=False)
    ]

    return graphics_outputs


@function_attributes(short_name=None, tags=['surprise', 'video', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-27 20:54', related_items=[])
def export_active_relative_entropy_results_videos(active_relative_entropy_results, active_context):
    """ 

    from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import export_active_relative_entropy_results_videos

    video_output_parent_path = export_active_relative_entropy_results_videos(active_relative_entropy_results, active_context=curr_active_pipeline.get_session_context())

    """
    import cv2
    from pyphocorehelpers.plotting.media_output_helpers import save_array_as_video

    video_output_parent_path = Path('output/videos').resolve()
    video_output_dict = {}
    video_properties_dict = {
        'snapshot_occupancy_weighted_tuning_maps': dict(isColor=False),
    #  'flat_jensen_shannon_distance_results': dict(blending='additive', colormap='gray'),
        # 'long_short_rel_entr_curves_frames': dict(isColor=False),
        # 'short_long_rel_entr_curves_frames': dict(isColor=False),
    }

    for a_name, video_properties in video_properties_dict.items():
        # image_layer_dict[a_name] = viewer.add_image(active_relative_entropy_results_xr_dict[a_name].to_numpy().astype(float), name=a_name)
        video_out_path = video_output_parent_path.joinpath(f'{active_context.get_description()}_{a_name}.avi')
        video_output_dict[a_name] = save_array_as_video(array=active_relative_entropy_results[a_name], video_filename=video_out_path, isColor=False)
        print(f'video_output_dict[a_name]: {video_output_dict[a_name]}')

    # 
    return video_output_parent_path