""" 
Contains code related to Pho Hale's 4th Year PhD Presentation on 2023-09-25

"""
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import TruncationCheckingResults
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult


from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_short_v_long_pf1D_comparison
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines, add_track_shapes
from neuropy.plotting.ratemaps import plot_ratemap_1D
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager




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
