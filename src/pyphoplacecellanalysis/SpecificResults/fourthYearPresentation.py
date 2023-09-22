""" 
Contains code related to Pho Hale's 4th Year PhD Presentation on 2023-09-25

"""
import numpy as np
import pandas as pd

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
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines, add_track_shapes

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_long_short_surprise_difference_plot, plot_long_short, plot_long_short_any_values
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots

long_short_display_config_manager = LongShortDisplayConfigManager()
long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()


def fig_example_nontopo_remap(curr_active_pipeline):
	"""Specific Figure: Example of non-neighbor preserving remapping
	Usage:
		from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_example_nontopo_remap

		graphics_output_dict = fig_example_nontopo_remap(curr_active_pipeline)
	"""
	example_aclus = [7, 95]
	# # flat_stack_mode: all placefields are stacked up (z-wise) on top of each other on a single axis with no offsets:
	example_shared_kwargs = dict(pad=1, active_context=curr_active_pipeline.get_session_context(), plot_zero_baselines=True, skip_figure_titles=True, use_flexitext_titles=True, flat_stack_mode=True)
	example_top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=True, should_plot_linear_track_shapes=True) # Renders the linear track shape on the maze. Assumes `flat_stack_mode=True`

	# (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, example_aclus, reuse_axs_tuple=None, single_figure=True, title_string="Example Non-Neighbor Preserving Remapping Cells", subtitle_string=f"2 Example Cells {example_aclus}", shared_kwargs=example_shared_kwargs, **example_top_level_shared_kwargs)
	return curr_active_pipeline.display('_display_short_long_pf1D_comparison', curr_active_pipeline.get_session_context(), included_any_context_neuron_ids=example_aclus, reuse_axs_tuple=None, single_figure=True, title_string="Example Non-Neighbor Preserving Remapping Cells", subtitle_string=f"2 Example Cells {example_aclus}", shared_kwargs=example_shared_kwargs, **example_top_level_shared_kwargs)
	 

def fig_remapping_cells(curr_active_pipeline):
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


	# # flat_stack_mode: all placefields are stacked up (z-wise) on top of each other on a single axis with no offsets:
	# shared_kwargs = dict(pad=1, active_context=curr_active_pipeline.get_session_context(), plot_zero_baselines=True, skip_figure_titles=True, use_flexitext_titles=True, flat_stack_mode=True)
	# top_level_shared_kwargs = dict(should_plot_vertical_track_bounds_lines=False, should_plot_linear_track_shapes=True) # Renders the linear track shape on the maze. Assumes `flat_stack_mode=True`

	graphics_output_dict = {}

	# long_results = curr_active_pipeline.computation_results['maze1_PYR'].computed_data
	# short_results = curr_active_pipeline.computation_results['maze2_PYR'].computed_data
	# curr_any_context_neurons = _find_any_context_neurons(*[curr_active_pipeline.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in ['maze1_PYR', 'maze2_PYR']])

	# (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, disappearing_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Disappearing Cells", subtitle_string=None, **top_level_shared_kwargs)
	graphics_output_dict['disappearing_endcap_aclus'] = curr_active_pipeline.display('_display_short_long_pf1D_comparison',active_context.adding_context_if_missing(cell_subset='disappear_endcap'), included_any_context_neuron_ids=disappearing_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Disappearing Cells", subtitle_string=None, **top_level_shared_kwargs)

	# (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, significant_distant_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Significant Distance Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
	graphics_output_dict['significant_distant_remapping_endcap_aclus'] = curr_active_pipeline.display('_display_short_long_pf1D_comparison',active_context.adding_context_if_missing(cell_subset='sig_remap_endcap'), included_any_context_neuron_ids=significant_distant_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Significant Distance Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)

	# (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, trivially_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Trivially Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)
	graphics_output_dict['trivially_remapping_endcap_aclus'] = curr_active_pipeline.display('_display_short_long_pf1D_comparison', active_context.adding_context_if_missing(cell_subset='triv_remap_endcap'), included_any_context_neuron_ids=trivially_remapping_endcap_aclus, reuse_axs_tuple=None, single_figure=True, shared_kwargs=shared_kwargs, title_string="Trivially Remapping Cells", subtitle_string="1D Placefields", **top_level_shared_kwargs)

	return graphics_output_dict


def fig_example_handpicked_pho_jonathan_active_set_cells(curr_active_pipeline, save_figure=False, included_LxC_example_neuron_IDs=[4, 58], included_SxC_example_neuron_IDs=[2]):
	# 2023-09-07 - Build Example LxC/SxC cells from handpicked examples: aclus = [4, 58]
	# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import build_extra_cell_info_label_string
	curr_active_pipeline.reload_default_display_functions()
	_LxC_out = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', n_max_plot_rows=2, save_figure=save_figure, included_unit_neuron_IDs=included_LxC_example_neuron_IDs, active_context=curr_active_pipeline.get_session_context().adding_context_if_missing(example='short_exclusive')) # , included_unit_neuron_IDs=[4, 58]
	_SxC_out = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', n_max_plot_rows=2, save_figure=save_figure, included_unit_neuron_IDs=included_SxC_example_neuron_IDs, active_context=curr_active_pipeline.get_session_context().adding_context_if_missing(example='long_exclusive')) # handpicked long-exclusive
	return _LxC_out, _SxC_out




def fig_surprise_results(curr_active_pipeline):
	""" 

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
	active_filter_epoch_obj: Epoch = _helper_prepare_epoch_df_for_draw_epoch_regions(active_filter_epochs)

	global_results = curr_active_pipeline.computation_results['maze'].computed_data
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
		
		draw_epoch_regions(epochs, ax, facecolor=epoch_region_facecolor, alpha=0.1, 
						edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 12}, 
						defer_render=defer_render, debug_print=debug_print, zorder=-20)

		draw_epoch_regions(laps_epochs, ax, facecolor='red', edgecolors='black', 
						labels_kwargs=lap_labels_kwargs, 
						defer_render=defer_render, debug_print=debug_print, zorder=-10)

		draw_epoch_regions(filter_epochs, ax, facecolor='orange', edgecolors=None, 
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

	

	epochs = curr_active_pipeline.sess.epochs
	laps_epochs = curr_active_pipeline.sess.laps.as_epoch_obj()
	filter_epochs = active_filter_epoch_obj

	graphics_outputs = [plot_data_and_epochs(post_update_times, flat_surprise_across_all_positions, 
						't (seconds)', 'Relative Entropy across all positions', 
						'flat_surprise_across_all_positions', 
						epochs, laps_epochs, filter_epochs, epoch_region_facecolor),
	plot_data_and_epochs(post_update_times, flat_jensen_shannon_distance_across_all_positions, 
						't (seconds)', 'J-S Distance across all positions', 
						'flat_jensen_shannon_distance_across_all_positions', 
						epochs, laps_epochs, filter_epochs, epoch_region_facecolor),
	plot_data_and_epochs(post_update_times, flat_relative_entropy_results, 
						't (seconds)', 'Relative Entropy', 
						'Relative Entropy vs Time', 
						epochs, laps_epochs, filter_epochs, epoch_region_facecolor, defer_render=False)]

	# for (fig, ax) in graphics_outputs:


	# fig, ax = plt.subplots()
	# ax.plot(post_update_times, flat_surprise_across_all_positions)
	# ax.set_ylabel('Relative Entropy across all positions')
	# ax.set_xlabel('t (seconds)')
	# epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=epoch_region_facecolor, alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
	# laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
	# replays_epochs_collection, replays_epoch_labels = draw_epoch_regions(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
	# fig.suptitle('flat_surprise_across_all_positions')
	# fig.show()

	# fig, ax = plt.subplots()
	# ax.plot(post_update_times, flat_jensen_shannon_distance_across_all_positions, label='JS_Distance')
	# ax.set_ylabel('J-S Distance across all positions')
	# ax.set_xlabel('t (seconds)')
	# epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=epoch_region_facecolor, alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
	# laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
	# replays_epochs_collection, replays_epoch_labels = draw_epoch_regions(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
	# fig.suptitle('flat_jensen_shannon_distance_across_all_positions')
	# fig.show()

	# # Show basic relative entropy vs. time plot:
	# fig, ax = plt.subplots()
	# ax.plot(post_update_times, flat_relative_entropy_results)
	# ax.set_ylabel('Relative Entropy')
	# ax.set_xlabel('t (seconds)')
	# epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=epoch_region_facecolor, alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=False, debug_print=False)
	# fig.show()

	# outputs = (fig, ax)
	# outputs = (None, None)
	# return outputs

	return graphics_outputs