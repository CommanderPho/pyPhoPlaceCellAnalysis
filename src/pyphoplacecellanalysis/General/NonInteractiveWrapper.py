import importlib
import sys
from pathlib import Path
import numpy as np
from enum import unique # SessionBatchProgress

## MATPLOTLIB Imports:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends import backend_pdf


# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.core.epoch import NamedTimerange
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.misc import compute_paginated_grid_config # for paginating shared aclus

# pyphocorehelpers
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # required for SessionBatchProgress
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.indexing_helpers import compute_position_grid_size

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership # needed for batch_programmatic_figures
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, session_context_to_relative_path, programmatic_display_to_PDF
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_metadata_from_display_context, create_daily_programmatic_display_function_testing_folder_if_needed # newer version of build_pdf_export_metadata
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # for batch_load_session

@unique
class SessionBatchProgress(ExtendedEnum):
    """Indicates the progress state for a given session in a batch processing queue """
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"

 
""" 

filters should be checkable to express whether we want to build that one or not



"""

class NonInteractiveWrapper(object):
	"""A wrapper class that performs a non-interactive version of the jupyter-lab notebook for loading and processing the pipeline. """
	def __init__(self, enable_saving_to_disk=False):
		super(NonInteractiveWrapper, self).__init__()
		self.enable_saving_to_disk = enable_saving_to_disk
		# common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\Final Placemaps 2021-01-14')
		# self.common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\2022-01-16')
		self.common_parent_foldername = Path(r'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\output')

		
	
	@staticmethod
	def compute_position_grid_bin_size(x, y, num_bins=(64,64), debug_print=False):
		""" Compute Required Bin size given a desired number of bins in each dimension
		Usage:
			active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64)
		"""
		out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(x, y, num_bins=num_bins)
		active_grid_bin = tuple(out_grid_bin_size)
		if debug_print:
			print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
		return active_grid_bin

	# WARNING! TODO: Changing the smooth values from (1.5, 1.5) to (0.5, 0.5) was the difference between successful running and a syntax error!
	# try:
	#     active_grid_bin
	# except NameError as e:
	#     print('setting active_grid_bin = None')
	#     active_grid_bin = None
	# finally:
	#     # active_session_computation_config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=active_grid_bin, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5) # if active_grid_bin is missing, figure out the name
	#     active_session_computation_config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=active_grid_bin, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5) # if active_grid_bin is missing, figure out the name

	## Dynamic mode:
	@staticmethod
	def _build_active_computation_configs(sess):
		""" _get_computation_configs(curr_kdiba_pipeline.sess) 
			# From Diba:
			# (3.777, 1.043) # for (64, 64) bins
			# (1.874, 0.518) # for (128, 128) bins

		"""
		return [DynamicParameters(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=NonInteractiveWrapper.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None))]
		
	@staticmethod
	def perform_computation(pipeline, active_computation_configs, enabled_filter_names=None):
		pipeline.perform_computations(active_computation_configs[0], enabled_filter_names) # unuses the first config
		pipeline.prepare_for_display() # TODO: pass a display config
		return pipeline

	@staticmethod
	def perform_filtering(pipeline, filter_configs):
		pipeline.filter_sessions(filter_configs)
		return pipeline


		
  
  
	@staticmethod
	def bapun_format_all(pipeline):
		active_session_computation_configs, active_session_filter_configurations = NonInteractiveWrapper.bapun_format_define_configs(pipeline)
		pipeline = NonInteractiveWrapper.perform_filtering(pipeline, active_session_filter_configurations)
		pipeline = NonInteractiveWrapper.perform_computation(pipeline, active_session_computation_configs)
		return pipeline, active_session_computation_configs, active_session_filter_configurations

     
     
     
	@staticmethod
	def bapun_format_define_configs(curr_bapun_pipeline):
		# curr_bapun_pipeline = NeuropyPipeline(name='bapun_pipeline', session_data_type='bapun', basedir=known_data_session_type_dict['bapun'].basedir, load_function=known_data_session_type_dict['bapun'].load_function)
		# curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
		active_session_computation_configs = NonInteractiveWrapper._build_active_computation_configs(curr_bapun_pipeline.sess)
		# active_session_computation_config.grid_bin = compute_position_grid_bin_size(curr_bapun_pipeline.sess.position.x, curr_bapun_pipeline.sess.position.y, num_bins=(64, 64))
				
		# Bapun/DataFrame style session filter functions:
		def build_bapun_any_maze_epochs_filters(sess):
			# all_filters = build_custom_epochs_filters(sess)
			# # print(f'all_filters: {all_filters}')
			# maze_only_filters = dict()
			# for (name, filter_fcn) in all_filters.items():
			#     if 'maze' in name:
			#         maze_only_filters[name] = filter_fcn
			# maze_only_filters = build_custom_epochs_filters(sess, epoch_name_whitelist=['maze1','maze2'])
			# { key:value for (key,value) in dictOfNames.items() if key % 2 == 0}
			# dict(filter(lambda elem: len(elem[1]) == 6,dictOfNames.items()))
			# maze_only_name_filter_fn = lambda dict: dict(filter(lambda elem: 'maze' in elem[0], dict.items()))
			maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names)) # include only maze tracks
			# print(f'callable(maze_only_name_filter_fn): {callable(maze_only_name_filter_fn)}')
			# print(maze_only_name_filter_fn(['pre', 'maze1', 'post1', 'maze2', 'post2']))
			# lambda elem: elem[0] % 2 == 0
			maze_only_filters = build_custom_epochs_filters(sess, epoch_name_whitelist=maze_only_name_filter_fn)
			# print(f'maze_only_filters: {maze_only_filters}')
			return maze_only_filters


		active_session_filter_configurations = build_bapun_any_maze_epochs_filters(curr_bapun_pipeline.sess)
		for i in np.arange(len(active_session_computation_configs)):
			active_session_computation_configs[i].computation_epochs = None  # set the placefield computation epochs to None, using all epochs.
		return active_session_computation_configs, active_session_filter_configurations


	@staticmethod
	def kdiba_format_all(pipeline):
		active_session_computation_configs, active_session_filter_configurations = NonInteractiveWrapper.kdiba_format_define_configs(pipeline)
		pipeline = NonInteractiveWrapper.perform_filtering(pipeline, active_session_filter_configurations)
		pipeline = NonInteractiveWrapper.perform_computation(pipeline, active_session_computation_configs)
		return pipeline, active_session_computation_configs, active_session_filter_configurations


	@staticmethod
	def kdiba_format_define_configs(curr_kdiba_pipeline):
		## Data must be pre-processed using the MATLAB script located here: 
		# R:\data\KDIBA\gor01\one\IIDataMat_Export_ToPython_2021_11_23.m
		# From pre-computed .mat files:
		## 07: 
		# basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
		# # ## 08:
		# basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
		# curr_kdiba_pipeline = NeuropyPipeline(name='kdiba_pipeline', session_data_type='kdiba', basedir=known_data_session_type_dict['kdiba'].basedir, load_function=known_data_session_type_dict['kdiba'].load_function)
		# curr_kdiba_pipeline = NeuropyPipeline.init_from_known_data_session_type('kdiba', known_data_session_type_dict['kdiba'])
		# active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
		# active_session_computation_config.grid_bin = active_grid_bin
		active_session_computation_configs = NonInteractiveWrapper._build_active_computation_configs(curr_kdiba_pipeline.sess)
		
		def build_any_maze_epochs_filters(sess):
			sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
			active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')) } # just maze 1
			# active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
			#                                     'maze2': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2')),
			#                                     'maze': lambda x: (x.filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]))
			#                                    }
			return active_session_filter_configurations

		active_session_filter_configurations = build_any_maze_epochs_filters(curr_kdiba_pipeline.sess)
		for i in np.arange(len(active_session_computation_configs)):
			active_session_computation_configs[i].computation_epochs = None # add the laps epochs to all of the computation configs.

		return active_session_computation_configs, active_session_filter_configurations
		
		# # set curr_pipeline for testing:
		# curr_pipeline = curr_kdiba_pipeline
		


		# Pyramidal and Lap-Only:
		def build_pyramidal_epochs_filters(sess):
			sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
			active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
												'maze2': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2')),
												'maze': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]))
											}
			return active_session_filter_configurations

		active_session_filter_configurations = build_pyramidal_epochs_filters(curr_kdiba_pipeline.sess)

		lap_specific_epochs = curr_kdiba_pipeline.sess.laps.as_epoch_obj()
		any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(curr_kdiba_pipeline.sess.laps.lap_id))])
		even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(curr_kdiba_pipeline.sess.laps.lap_id), 2)])
		odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(curr_kdiba_pipeline.sess.laps.lap_id), 2)])

		# Copy the active session_computation_config:
		for i in np.arange(len(active_session_computation_configs)):
			active_session_computation_configs[i].computation_epochs = any_lap_specific_epochs # add the laps epochs to all of the computation configs.

		return active_session_computation_configs, active_session_filter_configurations
	

# ==================================================================================================================== #
# 2022-12-07 - batch_load_session - Computes Entire Pipeline                                                           #
# ==================================================================================================================== #

def batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, force_reload=False, **kwargs):
    """Loads and runs the entire pipeline for a session folder located at the path 'basedir'.

    Args:
        global_data_root_parent_path (_type_): _description_
        active_data_mode_name (_type_): _description_
        basedir (_type_): _description_

    Returns:
        _type_: _description_
    """

    epoch_name_whitelist = kwargs.get('epoch_name_whitelist', ['maze1','maze2','maze'])
    debug_print = kwargs.get('debug_print', False)
    skip_save = kwargs.get('skip_save', False)
    active_pickle_filename = kwargs.get('active_pickle_filename', 'loadedSessPickle.pkl')

    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
        override_basepath=Path(basedir), override_post_load_functions=[], force_reload=force_reload, active_pickle_filename=active_pickle_filename, skip_save=True)

    active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_whitelist=epoch_name_whitelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
    if debug_print:
        print(f'active_session_filter_configurations: {active_session_filter_configurations}')
    
    curr_active_pipeline.filter_sessions(active_session_filter_configurations, changed_filters_ignore_list=['maze1','maze2','maze'], debug_print=True)

    # ## Compute shared grid_bin_bounds for all epochs from the global positions:
    # global_unfiltered_session = curr_active_pipeline.sess
    # # ((22.736279243974774, 261.696733348342), (49.989466271998936, 151.2870218547401))
    # first_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[0]]
    # # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
    # second_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[1]]
    # # ((71.67666779621361, 224.37820920766043), (110.51617463644946, 151.2870218547401))

    # grid_bin_bounding_session = first_filtered_session
    # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(grid_bin_bounding_session.position.x, grid_bin_bounding_session.position.y)

    ## OR use no grid_bin_bounds meaning they will be determined dynamically for each epoch:
    grid_bin_bounds = None

    active_session_computation_configs = active_data_mode_registered_class.build_default_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=0.03333, grid_bin_bounds=grid_bin_bounds) #1.0/30.0 # decode at 30fps to match the position sampling frequency

    # Whitelist Mode:
    computation_functions_name_whitelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                        '_perform_position_decoding_computation', 
                                        '_perform_firing_rate_trends_computation',
                                        '_perform_pf_find_ratemap_peaks_computation',
                                        # '_perform_two_step_position_decoding_computation',
                                        # '_perform_recursive_latent_placefield_decoding'
                                     ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'
    computation_functions_name_blacklist=None

    # # Blacklist Mode:
    # computation_functions_name_whitelist=None
    # computation_functions_name_blacklist=['_perform_spike_burst_detection_computation','_perform_recursive_latent_placefield_decoding']

    curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=True, debug_print=debug_print) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)

    # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_blacklist=['_perform_spike_burst_detection_computation'], debug_print=False, fail_on_exception=False) # whitelist: ['_perform_baseline_placefield_computation']

    curr_active_pipeline.prepare_for_display(root_output_dir=global_data_root_parent_path.joinpath('Output'), should_smooth_maze=True) # TODO: pass a display config

    if not skip_save:
        curr_active_pipeline.save_pipeline()
    else:
        print(f'skip_save == True, so not saving at the end of batch_load_session')
    return curr_active_pipeline

# ==================================================================================================================== #
# Batch Programmatic Figures - 2022-12-08 Batch Programmatic Figures (Currently only Jonathan-style)                                                                                           #
# ==================================================================================================================== #

def batch_programmatic_figures(curr_active_pipeline):
    """ programmatically generates and saves the batch figures 2022-12-07 
        curr_active_pipeline is the pipeline for a given session with all computations done.

    ## TODO: curr_session_parent_out_path


    """
    ## 🗨️🟢 2022-10-26 - Jonathan Firing Rate Analyses
    # Perform missing global computations                                                                                  #
    curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_jonathan_replay_firing_rate_analyses', '_perform_short_long_pf_overlap_analyses'], fail_on_exception=True, debug_print=True)

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']

    # ==================================================================================================================== #
    # Batch Output of Figures                                                                                              #
    # ==================================================================================================================== #
    ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
    # %matplotlib qt
    short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
    short_only_aclus = short_only_df.index.values.tolist()
    long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
    long_only_aclus = long_only_df.index.values.tolist()
    shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
    shared_aclus = shared_df.index.values.tolist()
    print(f'shared_aclus: {shared_aclus}')
    print(f'long_only_aclus: {long_only_aclus}')
    print(f'short_only_aclus: {short_only_aclus}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
    figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
    active_session_figures_out_path = session_context_to_relative_path(figures_parent_out_path, active_identifying_session_ctx)
    print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
    active_session_figures_out_path.mkdir(parents=True, exist_ok=True) # make folder if needed


    # ==================================================================================================================== #
    # Output Figures to File                                                                                               #
    # ==================================================================================================================== #
    ## PDF Output
    # %matplotlib qtagg
    import matplotlib
    # configure backend here
    # matplotlib.use('Qt5Agg')
    # backend_qt5agg
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

    n_max_page_rows = 10
    _batch_plot_kwargs_list = BatchPhoJonathanFiguresHelper._build_batch_plot_kwargs(long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=n_max_page_rows)
    active_out_figures_list = BatchPhoJonathanFiguresHelper._perform_batch_plot(curr_active_pipeline, _batch_plot_kwargs_list, figures_parent_out_path=active_session_figures_out_path, write_pdf=False, write_png=True, progress_print=True, debug_print=False)



    return active_identifying_session_ctx, active_session_figures_out_path, active_out_figures_list

def batch_extended_programmatic_figures(curr_active_pipeline):
    _bak_rcParams = mpl.rcParams.copy()
    mpl.rcParams['toolbar'] = 'None' # disable toolbars
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!
    # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefields', debug_print=False) # 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations') # , filter_name=active_config_name 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear. Moderate visual improvements can still be made (titles overlap and stuff). Works with %%capture
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_result_plot_ratemaps_2D') #  🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.

class BatchPhoJonathanFiguresHelper(object):
	"""Private methods that help with batch figure generator for ClassName.

	2022-12-08 - Batch Programmatic Figures (Currently only Jonathan-style) 
	2022-12-01 - Automated programmatic output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`

	"""

	@classmethod
	def _subfn_batch_plot_automated(cls, curr_active_pipeline, included_unit_neuron_IDs=None, active_identifying_ctx=None, fignum=None, fig_idx=0, n_max_page_rows=10):
		""" the a programmatic wrapper for automated output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`. The specific plot function called. 
		Called ONLY by `_perform_batch_plot(...)`

		"""
		# size_dpi = 100.0,
		# single_subfigure_size_px = np.array([1920.0, 220.0])
		single_subfigure_size_inches = np.array([19.2,  2.2])

		num_cells = len(included_unit_neuron_IDs)
		desired_figure_size_inches = single_subfigure_size_inches.copy()
		desired_figure_size_inches[1] = desired_figure_size_inches[1] * num_cells
		graphics_output_dict = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', active_identifying_ctx,
															n_max_plot_rows=n_max_page_rows, included_unit_neuron_IDs=included_unit_neuron_IDs,
															show_inter_replay_frs=False, spikes_color=(0.1, 0.0, 0.1), spikes_alpha=0.5, fignum=fignum, fig_idx=fig_idx, figsize=desired_figure_size_inches)
		fig, subfigs, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['subfigs'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
		fig.suptitle(active_identifying_ctx.get_description()) # 'kdiba_2006-6-08_14-26-15_[4, 13, 36, 58, 60]'
		return fig

	@classmethod
	def _build_batch_plot_kwargs(cls, long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=10):
		""" builds the list of kwargs for all aclus """
		_batch_plot_kwargs_list = [] # empty list to start
		## {long_only, short_only} plot configs (doesn't include the shared_aclus)
		if len(long_only_aclus) > 0:        
			_batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=long_only_aclus,
			active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
				display_fn_name='batch_plot_test', plot_result_set='long_only', aclus=f"{long_only_aclus}"
			),
			fignum='long_only', n_max_page_rows=len(long_only_aclus)))
		else:
			print(f'WARNING: long_only_aclus is empty, so not adding kwargs for these.')
		
		if len(short_only_aclus) > 0:
			_batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=short_only_aclus,
			active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
				display_fn_name='batch_plot_test', plot_result_set='short_only', aclus=f"{short_only_aclus}"
			),
			fignum='short_only', n_max_page_rows=len(short_only_aclus)))
		else:
			print(f'WARNING: short_only_aclus is empty, so not adding kwargs for these.')

		## Build Pages for Shared ACLUS:    
		nAclusToShow = len(shared_aclus)
		if nAclusToShow > 0:        
			# Paging Management: Constrain the subplots values to just those that you need
			subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nAclusToShow, max_num_columns=1, max_subplots_per_page=n_max_page_rows, data_indicies=shared_aclus, last_figure_subplots_same_layout=True)
			num_pages = len(included_combined_indicies_pages)
			## paginated outputs for shared cells
			included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
			paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
				active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name='batch_plot_test', plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{curr_included_unit_indicies}"),
				fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]
			_batch_plot_kwargs_list.extend(paginated_shared_cells_kwarg_list) # add paginated_shared_cells_kwarg_list to the list
		else:
			print(f'WARNING: shared_aclus is empty, so not adding kwargs for these.')
		return _batch_plot_kwargs_list

	@classmethod
	def _perform_batch_plot(cls, curr_active_pipeline, active_kwarg_list, figures_parent_out_path=None, subset_whitelist=None, subset_blacklist=None, write_pdf=False, write_png=True, progress_print=True, debug_print=False):
		""" Plots everything using the kwargs provided in `active_kwarg_list`

		Args:
			active_kwarg_list (_type_): generated by `_build_batch_plot_kwargs(...)`
			figures_parent_out_path (_type_, optional): _description_. Defaults to None.
			write_pdf (bool, optional): _description_. Defaults to False.
			write_png (bool, optional): _description_. Defaults to True.
			progress_print (bool, optional): _description_. Defaults to True.
			debug_print (bool, optional): _description_. Defaults to False.

		Returns:
			_type_: _description_
		"""
		if figures_parent_out_path is None:
			figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
		active_out_figures_list = [] # empty list to hold figures
		num_pages = len(active_kwarg_list)
		for i, curr_batch_plot_kwargs in enumerate(active_kwarg_list):
			curr_active_identifying_ctx = curr_batch_plot_kwargs['active_identifying_ctx']
			# print(f'curr_active_identifying_ctx: {curr_active_identifying_ctx}')
			active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx, subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist)
			# print(f'active_pdf_save_filename: {active_pdf_save_filename}')
			curr_pdf_save_path = figures_parent_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)
			# One plot at a time to PDF:
			if write_pdf:
				with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
					a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
					active_out_figures_list.append(a_fig)
					# Save out PDF page:
					pdf.savefig(a_fig)
			else:
				a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
				active_out_figures_list.append(a_fig)

			# Also save .png versions:
			if write_png:
				# curr_page_str = f'pg{i+1}of{num_pages}'
				fig_png_out_path = curr_pdf_save_path.with_suffix('.png')
				# fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
				a_fig.savefig(fig_png_out_path)
				if progress_print:
					print(f'\t saved {fig_png_out_path}')

		return active_out_figures_list



