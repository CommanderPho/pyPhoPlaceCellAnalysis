from copy import deepcopy
import shutil
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from neuropy.analyses import Epoch
from neuropy.core.epoch import ensure_dataframe
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

from pathlib import Path
import inspect
from jinja2 import Template
from neuropy.utils.result_context import IdentifyingContext
from nptyping import NDArray
import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from neuropy.utils.mixins.indexing_helpers import get_dict_subset


# ---------------------------------------------------------------------------- #
#      2024-06-25 - Diba 2009-style Replay Detection via Quiescent Period      #
# ---------------------------------------------------------------------------- #
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import helper_perform_pickle_pipeline

class BatchCompletionHelpers:
	""" helpers
	
	from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_completion_helpers import BatchCompletionHelpers
	
	
	"""
	@function_attributes(short_name=None, tags=['replay', 'epochs'], input_requires=[], output_provides=[], uses=[], used_by=['overwrite_replay_epochs_and_recompute'], creation_date='2024-06-26 21:10', related_items=[])
	@classmethod
	def replace_replay_epochs(cls, curr_active_pipeline, new_replay_epochs: Epoch):
		""" 
		Replaces each session's replay epochs and their `preprocessing_parameters.epoch_estimation_parameters.replays` config
		
		from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import replace_replay_epochs


		"""
		_backup_session_replay_epochs = {}
		_backup_session_configs = {}
		
		if isinstance(new_replay_epochs, pd.DataFrame):
			new_replay_epochs = Epoch.from_dataframe(new_replay_epochs) # ensure it is an epoch object
			
		new_replay_epochs = new_replay_epochs.get_non_overlapping() # ensure non-overlapping

		## Get the estimation parameters:
		replay_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
		assert replay_estimation_parameters is not None
		_bak_replay_estimation_parameters = deepcopy(replay_estimation_parameters) ## backup original
		## backup original values:
		_backup_session_replay_epochs['sess'] = deepcopy(_bak_replay_estimation_parameters)
		_backup_session_configs['sess'] = deepcopy(curr_active_pipeline.sess.replay)

		## Check if they changed
		did_change: bool = False
		did_change = did_change or np.any(ensure_dataframe(_backup_session_configs['sess']).to_numpy() != ensure_dataframe(new_replay_epochs).to_numpy())

		## Set new:
		replay_estimation_parameters.epochs_source = new_replay_epochs.metadata.get('epochs_source', None)
		# replay_estimation_parameters.require_intersecting_epoch = None # don't actually purge these as I don't know what they are used for
		replay_estimation_parameters.min_inclusion_fr_active_thresh = new_replay_epochs.metadata.get('minimum_inclusion_fr_Hz', 1.0)
		replay_estimation_parameters.min_num_unique_aclu_inclusions = new_replay_epochs.metadata.get('min_num_active_neurons', 5)

		did_change = did_change or (get_dict_subset(_bak_replay_estimation_parameters, ['epochs_source', 'min_num_unique_aclu_inclusions', 'min_inclusion_fr_active_thresh']) != get_dict_subset(replay_estimation_parameters, ['epochs_source', 'min_num_unique_aclu_inclusions', 'min_inclusion_fr_active_thresh']))
		## Assign the new parameters:
		curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays = deepcopy(replay_estimation_parameters)

		assert curr_active_pipeline.sess.basepath.exists()
		## assign the new replay epochs:
		curr_active_pipeline.sess.replay = deepcopy(new_replay_epochs)
		for k, a_filtered_session in curr_active_pipeline.filtered_sessions.items():
			## backup original values:
			_backup_session_replay_epochs[k] = deepcopy(a_filtered_session.config.preprocessing_parameters.epoch_estimation_parameters.replays)
			_backup_session_configs[k] = deepcopy(a_filtered_session.replay)

			## assign the new replay epochs:
			a_filtered_session.replay = deepcopy(new_replay_epochs).time_slice(a_filtered_session.t_start, a_filtered_session.t_stop)
			assert curr_active_pipeline.sess.basepath.exists()
			a_filtered_session.config.basepath = deepcopy(curr_active_pipeline.sess.basepath)
			assert a_filtered_session.config.basepath.exists()
			a_filtered_session.config.preprocessing_parameters.epoch_estimation_parameters.replays = deepcopy(replay_estimation_parameters)

			# print(a_filtered_session.replay)
			# a_filtered_session.start()

		print(f'did_change: {did_change}')

		return did_change, _backup_session_replay_epochs, _backup_session_configs

	@classmethod
	def _get_custom_suffix_replay_epoch_source_name(cls, epochs_source: str) -> str:
		valid_epochs_source_values = ['compute_diba_quiescent_style_replay_events', 'diba_evt_file', 'initial_loaded', 'normal_computed']
		assert epochs_source in valid_epochs_source_values, f"epochs_source: '{epochs_source}' is not in valid_epochs_source_values: {valid_epochs_source_values}"
		to_filename_conversion_dict = {'compute_diba_quiescent_style_replay_events':'_withNewComputedReplays', 'diba_evt_file':'_withNewKamranExportedReplays', 'initial_loaded': '_withOldestImportedReplays', 'normal_computed': '_withNormalComputedReplays'}
		return to_filename_conversion_dict[epochs_source]
		# if epochs_source == 'compute_diba_quiescent_style_replay_events':
		# 	return '_withNewComputedReplays'
		# elif epochs_source == 'diba_evt_file':
		# 	return '_withNewKamranExportedReplays'
		# 	# qclu = new_replay_epochs.metadata.get('qclu', "[1,2]") # Diba export files are always qclus [1, 2]

		# elif epochs_source == 'initial_loaded':
		# 	return '_withOldestImportedReplays'

		# elif epochs_source == 'normal_computed':
		# 	return '_withNormalComputedReplays'
			
		# else:
		# 	raise NotImplementedError(f'epochs_source: {epochs_source} is of unknown type or is missing metadata.')    



	@function_attributes(short_name=None, tags=['dataframe', 'filename', 'metadata'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-28 12:40', related_items=[])
	@classmethod
	def _get_custom_suffix_for_replay_filename(cls, new_replay_epochs: Epoch, *extras_strings) -> str:
		""" Uses metadata stored in the replays dataframe to determine an appropriate filename
		
		
		from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _get_custom_suffix_for_replay_filename
		custom_suffix = _get_custom_suffix_for_replay_filename(new_replay_epochs=new_replay_epochs)

		print(f'custom_suffix: "{custom_suffix}"')

		"""
		assert new_replay_epochs.metadata is not None
		metadata = deepcopy(new_replay_epochs.metadata)
		extras_strings = []

		epochs_source = metadata.get('epochs_source', None)
		assert epochs_source is not None
		# print(f'epochs_source: {epochs_source}')

		valid_epochs_source_values = ['compute_diba_quiescent_style_replay_events', 'diba_evt_file', 'initial_loaded', 'normal_computed']
		assert epochs_source in valid_epochs_source_values, f"epochs_source: '{epochs_source}' is not in valid_epochs_source_values: {valid_epochs_source_values}"

		custom_suffix: str = cls._get_custom_suffix_replay_epoch_source_name(epochs_source=epochs_source)
		
		if epochs_source == 'compute_diba_quiescent_style_replay_events':
			# qclu = new_replay_epochs.metadata.get('qclu', "[1,2]")
			custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])

		elif epochs_source == 'diba_evt_file':
			custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata.get('minimum_inclusion_fr_Hz', 5.0):.1f}", *extras_strings])
			# qclu = new_replay_epochs.metadata.get('qclu', "[1,2]") # Diba export files are always qclus [1, 2]
		elif epochs_source == 'initial_loaded':
			custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', 'XX')}", f"frateThresh_{metadata.get('minimum_inclusion_fr_Hz', 0.1):.1f}", *extras_strings])

		elif epochs_source == 'normal_computed':
			custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
		else:
			raise NotImplementedError(f'epochs_source: {epochs_source} is of unknown type or is missing metadata.')    

		# with np.printoptions(precision=1, suppress=True, threshold=5):
		#     # score_text = f"score: " + str(np.array([epoch_score])).lstrip("[").rstrip("]") # output is just the number, as initially it is '[0.67]' but then the [ and ] are stripped.   
		#     return '-'.join([f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
		#     # return '-'.join([f"replaySource_{metadata['epochs_source']}", f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
			
		return custom_suffix

	@classmethod
	def _custom_replay_str_for_filename(cls, new_replay_epochs: Epoch, *extras_strings):
		assert new_replay_epochs.metadata is not None
		with np.printoptions(precision=1, suppress=True, threshold=5):
			metadata = deepcopy(new_replay_epochs.metadata)
			# score_text = f"score: " + str(np.array([epoch_score])).lstrip("[").rstrip("]") # output is just the number, as initially it is '[0.67]' but then the [ and ] are stripped.   
			return '-'.join([f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
			# return '-'.join([f"replaySource_{metadata['epochs_source']}", f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
			



	@function_attributes(short_name=None, tags=['replay', 'ALT_REPLAYS', 'new_replay', 'top'], input_requires=[], output_provides=[],
					   uses=['replace_replay_epochs', 'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'], used_by=[], creation_date='2024-06-25 22:49', related_items=['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])
	@classmethod
	def overwrite_replay_epochs_and_recompute(cls, curr_active_pipeline, new_replay_epochs: Epoch, ripple_decoding_time_bin_size: float = 0.025, 
											num_wcorr_shuffles: int=25, fail_on_exception=True,
											enable_save_pipeline_pkl: bool=True, enable_save_global_computations_pkl: bool=False, enable_save_h5: bool = False, user_completion_dummy=None, drop_previous_result_and_compute_fresh:bool=True):
		""" Recomputes the replay epochs using a custom implementation of the criteria in Diba 2007.

		, included_qclu_values=[1,2], minimum_inclusion_fr_Hz=5.0


		#TODO 2024-07-04 10:52: - [ ] Need to add the custom processing suffix to `BATCH_DATE_TO_USE`

		
		If `did_change` == True,
			['merged_directional_placefields', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring']
			['wcorr_shuffle_analysis']

			are updated

		Otherwise:
			['wcorr_shuffle_analysis'] can be updated

		Usage:
			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import overwrite_replay_epochs_and_recompute

			did_change, custom_save_filenames, custom_save_filepaths = overwrite_replay_epochs_and_recompute(curr_active_pipeline=curr_active_pipeline, new_replay_epochs=evt_epochs)

		"""
		from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
		from neuropy.utils.debug_helpers import parameter_sweeps
		from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function
		from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import finalize_output_shuffled_wcorr

		# 'epochs_source'
		custom_suffix: str = cls._get_custom_suffix_for_replay_filename(new_replay_epochs=new_replay_epochs) # correct
		print(f'custom_suffix: "{custom_suffix}"')

		assert (user_completion_dummy is not None), f"2024-07-04 - `user_completion_dummy` must be provided with a modified .BATCH_DATE_TO_USE to include the custom suffix!"

		additional_session_context = None
		try:
			if custom_suffix is not None:
				additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
				print(f'Using custom suffix: "{custom_suffix}" - additional_session_context: "{additional_session_context}"')
		except NameError as err:
			additional_session_context = None
			print(f'NO CUSTOM SUFFIX.')    
			

		## OUTPUTS: new_replay_epochs, new_replay_epochs_df
		did_change, _backup_session_replay_epochs, _backup_session_configs = cls.replace_replay_epochs(curr_active_pipeline=curr_active_pipeline, new_replay_epochs=new_replay_epochs)

		custom_save_filenames = {
			'pipeline_pkl':f'loadedSessPickle{custom_suffix}.pkl',
			'global_computation_pkl':f"global_computation_results{custom_suffix}.pkl",
			'pipeline_h5':f'pipeline{custom_suffix}.h5',
		}
		print(f'custom_save_filenames: {custom_save_filenames}')
		custom_save_filepaths = {k:v for k, v in custom_save_filenames.items()}

		if not did_change:
			print(f'no changes!')
			curr_active_pipeline.reload_default_computation_functions()
		
			## wcorr shuffle:
			curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['wcorr_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles, 'drop_previous_result_and_compute_fresh': drop_previous_result_and_compute_fresh}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

		else:
			print(f'replay epochs changed!')

			curr_active_pipeline.reload_default_computation_functions()
			
			should_skip_laps: bool = False

			metadata = deepcopy(new_replay_epochs.metadata)
			minimum_inclusion_fr_Hz = metadata.get('minimum_inclusion_fr_Hz', None)
			included_qclu_values = metadata.get('included_qclu_values', None)
			curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields','perform_rank_order_shuffle_analysis'],
															computation_kwargs_list=[{'laps_decoding_time_bin_size': None, 'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size},
																						{'num_shuffles': num_wcorr_shuffles, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz, 'included_qclu_values': included_qclu_values, 'skip_laps': should_skip_laps}],
															enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
			
			# '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
			# curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['perform_rank_order_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz, 'included_qclu_values': included_qclu_values, 'skip_laps': True}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation

			global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['DirectionalDecodersEpochsEvaluations'], debug_print=True)

			curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_evaluate_epochs',  'directional_decoders_epoch_heuristic_scoring'],
							computation_kwargs_list=[{'should_skip_radon_transform': False}, {}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
			
			## Export these new computations to .csv for across-session analysis:
			# Uses `perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function` to compute the new outputs:

			# BEGIN normal data Export ___________________________________________________________________________________________ #
			return_full_decoding_results: bool = False
			# desired_laps_decoding_time_bin_size = [None] # doesn't work
			desired_laps_decoding_time_bin_size = [1.5] # large so it doesn't take long
			desired_ripple_decoding_time_bin_size = [0.010, 0.025]

			custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size,
																						desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size,
																					use_single_time_bin_per_epoch=[False],
																					minimum_event_duration=[desired_ripple_decoding_time_bin_size[-1]])


			## make sure that the exported .csv and .h5 files have unique names based on the unique replays used. Also avoid unduely recomputing laps each time.
			_across_session_results_extended_dict = {}
			## Combine the output of `perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function` into two dataframes for the laps, one per-epoch and one per-time-bin
			_across_session_results_extended_dict = _across_session_results_extended_dict | perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(user_completion_dummy, None,
															curr_session_context=curr_active_pipeline.get_session_context(), curr_session_basedir=curr_active_pipeline.sess.basepath.resolve(), curr_active_pipeline=curr_active_pipeline,
															across_session_results_extended_dict=_across_session_results_extended_dict, return_full_decoding_results=return_full_decoding_results,
															save_hdf=enable_save_h5, save_csvs=True,
															# desired_shared_decoding_time_bin_sizes = np.linspace(start=0.030, stop=0.5, num=4),
															custom_all_param_sweep_options=custom_all_param_sweep_options, # directly provide the parameter sweeps
															additional_session_context=additional_session_context
														)
			# with `return_full_decoding_results == False`
			out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple = _across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']
			(several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple

			_out_file_paths_dict = {
				'ripple_h5_out_path': out_path,
				'ripple_csv_out_path': ripple_out_path,
				'ripple_csv_time_bin_marginals': ripple_time_bin_marginals_out_path,
				
				## Laps:
				'laps_csv_out_path': laps_out_path,
				'laps_csv_time_bin_marginals_out_path': laps_time_bin_marginals_out_path,
			}

			for a_name, a_path in _out_file_paths_dict.items():
				custom_save_filepaths[a_name] = a_path

			# custom_save_filepaths['csv_out_path'] = out_path # ends up being the .h5 path for some reason
			# custom_save_filepaths['csv_out_path'] = out_path # ends up being the .h5 path for some reason
			# custom_save_filepaths['ripple_csv_out_path'] = ripple_out_path

			# END Normal data Export _____________________________________________________________________________________________ #

			## Long/Short Stuff:
			# curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['long_short_decoding_analyses','long_short_fr_indicies_analyses','jonathan_firing_rate_analysis',
			#             'long_short_post_decoding','long_short_inst_spike_rate_groups','long_short_endcap_analysis'], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

			## Rank-Order Shuffle
			## try dropping result and recomputing:
			# global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['SequenceBased'], debug_print=True) # Now use , drop_previous_result_and_compute_fresh:bool=True

			# curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['rank_order_shuffle_analysis',], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
			# curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['rank_order_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': 10, 'skip_laps': True}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

			## Pickle first thing after changes:
			# custom_save_filepaths = helper_perform_pickle_pipeline(curr_active_pipeline=curr_active_pipeline, custom_save_filepaths=custom_save_filepaths)

			## wcorr shuffle:
			curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['wcorr_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles, 'drop_previous_result_and_compute_fresh': drop_previous_result_and_compute_fresh}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

			## Pickle again after recomputing:
			custom_save_filepaths = helper_perform_pickle_pipeline(a_curr_active_pipeline=curr_active_pipeline, custom_save_filenames=custom_save_filenames, custom_save_filepaths=custom_save_filepaths,
																	enable_save_pipeline_pkl=enable_save_pipeline_pkl, enable_save_global_computations_pkl=enable_save_global_computations_pkl, enable_save_h5=enable_save_h5)

		try:
			decoder_names = deepcopy(TrackTemplates.get_decoder_names())
			wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path) = finalize_output_shuffled_wcorr(a_curr_active_pipeline=curr_active_pipeline, decoder_names=decoder_names, custom_suffix=custom_suffix)
			custom_save_filepaths['standalone_wcorr_pkl'] = standalone_pkl_filepath
			custom_save_filepaths['standalone_mat_pkl'] = standalone_mat_filepath
			print(f'completed overwrite_replay_epochs_and_recompute(...). custom_save_filepaths: {custom_save_filepaths}\n')
			custom_save_filenames['standalone_wcorr_pkl'] = standalone_pkl_filepath.name
			custom_save_filenames['standalone_mat_pkl'] = standalone_mat_filepath.name
			
		except Exception as e:
			print(f'failed doing `finalize_output_shuffled_wcorr(...)` with error: {e}')
			if user_completion_dummy.fail_on_exception:
				print(f'did_change: {did_change}, custom_save_filenames: {custom_save_filenames}, custom_save_filepaths: {custom_save_filepaths}')
				raise e


		return did_change, custom_save_filenames, custom_save_filepaths


	# Replay Loading/Estimation Methods __________________________________________________________________________________ #

	@function_attributes(short_name=None, tags=['replay', 'ALT_REPLAYS', 'epochs', 'import', 'diba_evt_file'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-26 21:06', related_items=[])
	@classmethod
	def try_load_neuroscope_EVT_file_epochs(cls, curr_active_pipeline, ext:str='bst') -> Optional[Epoch]:
		""" loads the replay epochs from an exported .evt file

		Usage:

			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import try_load_neuroscope_EVT_file_epochs

			evt_epochs = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline)

			## load a previously exported to .ebt computed replays:
			evt_epochs = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline, ext='PHONEW')
			evt_epochs.metadata['epochs_source'] = 'compute_diba_quiescent_style_replay_events'

		"""
		## FROM .evt file
		evt_filepath = curr_active_pipeline.sess.basepath.joinpath(f'{curr_active_pipeline.session_name}.{ext}.evt').resolve()
		evt_epochs = None
		if evt_filepath.exists():
			# assert evt_filepath.exists(), f"evt_filepath: '{evt_filepath}' does not exist!"
			evt_epochs: Epoch = Epoch.from_neuroscope(in_filepath=evt_filepath, metadata={'epochs_source': 'diba_evt_file'}).get_non_overlapping()
			evt_epochs.filename = str(evt_filepath) ## set the filepath
		return evt_epochs


	@function_attributes(short_name=None, tags=['helper'], input_requires=[], output_provides=[], uses=[], used_by=['compute_diba_quiescent_style_replay_events'], creation_date='2024-06-27 22:16', related_items=[])
	@classmethod
	def check_for_and_merge_overlapping_epochs(cls, quiescent_periods: pd.DataFrame, debug_print=False) -> pd.DataFrame:
		"""
		Checks for overlaps in the quiescent periods and merges them if necessary.

		Parameters:
		quiescent_periods (pd.DataFrame): DataFrame containing quiescent periods with 'start' and 'stop' columns.

		Returns:
		pd.DataFrame: DataFrame with non-overlapping quiescent periods.

		Usage:

			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_diba_quiescent_style_replay_events, find_quiescent_windows, check_for_overlaps
			from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

			quiescent_periods = find_quiescent_windows(active_spikes_df=get_proper_global_spikes_df(curr_active_pipeline), silence_duration=0.06)
			quiescent_periods

		"""
		non_overlapping_periods = []
		last_stop = -float('inf')

		for idx, row in quiescent_periods.iterrows():
			if (last_stop is not None):        
				if (row['start'] > last_stop):
					non_overlapping_periods.append(row)
					last_stop = row['stop']
				else:
					# Optionally, you can log or handle the overlapping intervals here
					if debug_print:
						print(f"Overlap detected: {row['start']} - {row['stop']} overlaps with last stop {last_stop}")
					non_overlapping_periods[-1]['stop'] = row['stop'] # update the last event, don't add a new one
					last_stop = row['stop']

			else:
				non_overlapping_periods.append(row)
				last_stop = row['stop']

		non_overlapping_periods_df = pd.DataFrame(non_overlapping_periods)
		non_overlapping_periods_df["time_diff"] = non_overlapping_periods_df["stop"] - non_overlapping_periods_df["start"]
		non_overlapping_periods_df["duration"] = non_overlapping_periods_df["stop"] - non_overlapping_periods_df["start"]
		non_overlapping_periods_df = non_overlapping_periods_df.reset_index(drop=True)
		non_overlapping_periods_df["label"] = non_overlapping_periods_df.index.astype('str', copy=True)

		return non_overlapping_periods_df


	@function_attributes(short_name=None, tags=['replay', 'ALT_REPLAYS', 'compute', 'compute_diba_quiescent_style_replay_events'], input_requires=[], output_provides=[], uses=['check_for_and_merge_overlapping_epochs'], used_by=[], creation_date='2024-06-25 12:54', related_items=[])
	@classmethod
	def compute_diba_quiescent_style_replay_events(cls, curr_active_pipeline, spikes_df, included_qclu_values=[1,2], minimum_inclusion_fr_Hz=5.0, silence_duration:float=0.06, firing_window_duration:float=0.3,
				enable_export_to_neuroscope_EVT_file:bool=True):
		""" Method to find putative replay events similar to the Diba 2007 paper: by finding quiet periods and then getting the activity for 300ms after them.

		if 'included_qclu_values' and 'minimum_inclusion_fr_Hz' don't change, the templates and directional lap results aren't required it seems. 
		All of this is just in service of getting the properly filtered `active_spikes_df` to determine the quiescent periods.

		Usage:
			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_diba_quiescent_style_replay_events

		"""
		# ==================================================================================================================== #
		# BEGIN SUBFUNCTIONS                                                                                                   #
		# ==================================================================================================================== #
		from neuropy.core.epoch import Epoch, ensure_dataframe
		from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates

		def find_quiescent_windows(active_spikes_df: pd.DataFrame, silence_duration:float=0.06) -> pd.DataFrame:
			"""
			# Define the duration for silence and firing window
			silence_duration = 0.06  # 60 ms
			firing_window_duration = 0.3  # 300 ms
			min_unique_neurons = 14

			CAPTURES NOTHING
			"""
			## INPUTS: active_spikes_df

			# Ensure the DataFrame is sorted by the event times
			spikes_df = deepcopy(active_spikes_df)[['t_rel_seconds']].sort_values(by='t_rel_seconds').reset_index(drop=True).drop_duplicates(subset=['t_rel_seconds'], keep='first')

			# Drop rows with duplicate values in the 't_rel_seconds' column, keeping the first occurrence
			spikes_df = spikes_df.drop_duplicates(subset=['t_rel_seconds'], keep='first')

			# Calculate the differences between consecutive event times
			spikes_df['time_diff'] = spikes_df['t_rel_seconds'].diff()

			# Find the indices where the time difference is greater than 60ms (0.06 seconds)
			quiescent_periods = spikes_df[spikes_df['time_diff'] > silence_duration]

			# Extract the start and end times of the quiescent periods
			# quiescent_periods['start'] = spikes_df['t_rel_seconds'].shift(1)
			quiescent_periods['stop'] = quiescent_periods['t_rel_seconds']
			quiescent_periods['start'] = quiescent_periods['stop'] - quiescent_periods['time_diff']

			# Drop the NaN values that result from the shift operation
			quiescent_periods = quiescent_periods.dropna(subset=['start'])

			# Select the relevant columns
			quiescent_periods = quiescent_periods[['start', 'stop', 'time_diff']]
			# quiescent_periods["label"] = quiescent_periods.index.astype('str', copy=True)
			# quiescent_periods["duration"] = quiescent_periods["stop"] - quiescent_periods["start"] 
			quiescent_periods = cls.check_for_and_merge_overlapping_epochs(quiescent_periods=quiescent_periods)
			# print(quiescent_periods)
			return quiescent_periods

		def find_active_epochs_preceeded_by_quiescent_windows(active_spikes_df, silence_duration:float=0.06, firing_window_duration:float=0.3, min_unique_neurons:int=14):
			"""
			# Define the duration for silence and firing window
			silence_duration = 0.06  # 60 ms
			firing_window_duration = 0.3  # 300 ms
			min_unique_neurons = 14

			CAPTURES NOTHING
			"""
			## INPUTS: active_spikes_df

			# Ensure the DataFrame is sorted by the event times
			spikes_df = deepcopy(active_spikes_df).sort_values(by='t_rel_seconds').reset_index(drop=True)
			# Calculate the differences between consecutive event times
			spikes_df['time_diff'] = spikes_df['t_rel_seconds'].diff()

			## INPUTS: quiescent_periods
			quiescent_periods = find_quiescent_windows(active_spikes_df=active_spikes_df, silence_duration=silence_duration)

			# List to hold the results
			results = []

			# Variable to keep track of the end time of the last valid epoch
			# last_epoch_end = -float('inf')

			# Iterate over each quiescent period
			for idx, row in quiescent_periods.iterrows():
				silence_end = row['stop']
				window_start = silence_end
				window_end = silence_end + firing_window_duration
				
				# Check if there's another quiescent period within the current window
				if (idx + 1) < len(quiescent_periods):
					next_row = quiescent_periods.iloc[idx + 1]
					next_quiescent_start = next_row['start']
					if next_quiescent_start < window_end:
						window_end = next_quiescent_start
						# break
			
				# Filter events that occur in the 300-ms window after the quiescent period
				window_events = spikes_df[(spikes_df['t_rel_seconds'] >= window_start) & (spikes_df['t_rel_seconds'] <= window_end)]
				
				# Count unique neurons firing in this window
				unique_neurons = window_events['aclu'].nunique()
				
				# Check if at least 14 unique neurons fired in this window
				if unique_neurons >= min_unique_neurons:
					results.append({
						'quiescent_start': row['start'],
						'quiescent_end': silence_end,
						'window_start': window_start,
						'window_end': window_end,
						'unique_neurons': unique_neurons
					})
					# Variable to keep track of the end time of the last valid epoch
					# last_epoch_end = window_end


			# Convert results to a DataFrame
			results_df = pd.DataFrame(results)
			results_df["label"] = results_df.index.astype('str', copy=True)
			results_df["duration"] = results_df["window_end"] - results_df["window_start"] 
			return results_df, quiescent_periods

		# ==================================================================================================================== #
		# BEGIN FUNCTION BODY                                                                                                  #
		# ==================================================================================================================== #

		## INPUTS: curr_active_pipeline, directional_laps_results, rank_order_results
		# track_templates.determine_decoder_aclus_filtered_by_frate(5.0)
		# qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
		qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values)
		# qclu_included_aclus
		
		directional_laps_results: DirectionalLapsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'])
		modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(qclu_included_aclus)
		active_track_templates: TrackTemplates = deepcopy(modified_directional_laps_results.get_templates(minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)) # Here is where the firing rate matters
		# active_track_templates

		any_decoder_neuron_IDs = deepcopy(active_track_templates.any_decoder_neuron_IDs)
		n_neurons: int = len(any_decoder_neuron_IDs)
		# min_num_active_neurons: int = max(int(round(0.3 * float(n_neurons))), 5)
		min_num_active_neurons: int = active_track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline) # smarter, considers the minimum template

		print(f'n_neurons: {n_neurons}, min_num_active_neurons: {min_num_active_neurons}')
		# get_templates(5.0)
		active_spikes_df: pd.DataFrame = deepcopy(spikes_df)
		active_spikes_df = active_spikes_df.spikes.sliced_by_neuron_id(any_decoder_neuron_IDs)
		# active_spikes_df

		## OUTPUTS: active_spikes_df
		new_replay_epochs_df, quiescent_periods = find_active_epochs_preceeded_by_quiescent_windows(active_spikes_df, silence_duration=silence_duration, firing_window_duration=firing_window_duration, min_unique_neurons=min_num_active_neurons)
		new_replay_epochs_df = new_replay_epochs_df.rename(columns={'window_start': 'start', 'window_end': 'stop',})

		new_replay_epochs: Epoch = Epoch.from_dataframe(new_replay_epochs_df, metadata={'epochs_source': 'compute_diba_quiescent_style_replay_events',
																						'included_qclu_values': included_qclu_values, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
																						'silence_duration': silence_duration, 'firing_window_duration': firing_window_duration,
																						'qclu_included_aclus': qclu_included_aclus, 'min_num_active_neurons': min_num_active_neurons})
		
		if enable_export_to_neuroscope_EVT_file:
			## Save computed epochs out to a neuroscope .evt file:
			filename = f"{curr_active_pipeline.session_name}"
			filepath = curr_active_pipeline.get_output_path().joinpath(filename).resolve()
			## set the filename of the Epoch:
			new_replay_epochs.filename = filepath
			filepath = new_replay_epochs.to_neuroscope(ext='PHONEW')
			assert filepath.exists()
			print(F'saved out newly computed epochs to "{filepath}".')

		return (qclu_included_aclus, active_track_templates, active_spikes_df, quiescent_periods), (new_replay_epochs_df, new_replay_epochs)

	@function_attributes(short_name=None, tags=['MAIN', 'ALT_REPLAYS', 'replay'], input_requires=[], output_provides=[], uses=['compute_diba_quiescent_style_replay_events', 'try_load_neuroscope_EVT_file_epochs', 'try_load_neuroscope_EVT_file_epochs'], used_by=['compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function'], creation_date='2024-07-03 06:12', related_items=[])
	@classmethod
	def compute_all_replay_epoch_variations(cls, curr_active_pipeline, included_qclu_values = [1,2,4,6,7,9], minimum_inclusion_fr_Hz=5.0, suppress_exceptions: bool = True) -> Dict[str, Epoch]:
		""" Computes alternative replays (such as loading them from Diba-exported files, computing using the quiescent periods before the event, etc)
		
		suppress_exceptions: bool - allows some alternative replay computations to fail
		
		Usage:
			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_replay_epoch_variations

			replay_epoch_variations = compute_all_replay_epoch_variations(curr_active_pipeline)
			replay_epoch_variations

		"""
		from neuropy.core.epoch import ensure_Epoch
		from pyphocorehelpers.exception_helpers import ExceptionPrintingContext

		# print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
		# print(f'compute_all_replay_epoch_variations(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
		
		# ==================================================================================================================== #
		# Compute Alternative Replays: `replay_epoch_variations`                                                               #
		# ==================================================================================================================== #

		## Compute new epochs: 
		replay_epoch_variations = {}

		# with ExceptionPrintingContext(suppress=True, exception_print_fn=(lambda formatted_exception_str: print(f'\t"initial_loaded" failed with error: {formatted_exception_str}. Skipping.'))):
		#     replay_epoch_variations.update({
		#         'initial_loaded': ensure_Epoch(deepcopy(curr_active_pipeline.sess.replay_backup), metadata={'epochs_source': 'initial_loaded'}),
		#     })
		
		with ExceptionPrintingContext(suppress=suppress_exceptions, exception_print_fn=(lambda formatted_exception_str: print(f'\t"normal_computed" failed with error: {formatted_exception_str}. Skipping.'))):
			## Get the estimation parameters:
			replay_estimation_parameters = deepcopy(curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays)
			assert replay_estimation_parameters is not None

			## get the epochs computed normally:
			replay_epoch_variations.update({
				'normal_computed': ensure_Epoch(deepcopy(curr_active_pipeline.sess.replay), metadata={'epochs_source': 'normal_computed',
																							'minimum_inclusion_fr_Hz': replay_estimation_parameters['min_inclusion_fr_active_thresh'],
																							'min_num_active_neurons': replay_estimation_parameters['min_num_unique_aclu_inclusions'],
																								'included_qclu_values': deepcopy(included_qclu_values)
																								}),
			})

		# with ExceptionPrintingContext(suppress=suppress_exceptions, exception_print_fn=(lambda formatted_exception_str: print(f'\t"diba_quiescent_method_replay_epochs" failed with error: {formatted_exception_str}. Skipping.'))):
		#     ## Compute new epochs:
		#     spikes_df = get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
		#     (qclu_included_aclus, active_track_templates, active_spikes_df, quiescent_periods), (diba_quiescent_method_replay_epochs_df, diba_quiescent_method_replay_epochs) = compute_diba_quiescent_style_replay_events(curr_active_pipeline=curr_active_pipeline,
		#                                                                                                                                                                                 included_qclu_values=included_qclu_values, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, spikes_df=spikes_df)
		#     ## OUTPUTS: diba_quiescent_method_replay_epochs
		#     replay_epoch_variations.update({
		#         'diba_quiescent_method_replay_epochs': deepcopy(diba_quiescent_method_replay_epochs),
		#     })
			
		# with ExceptionPrintingContext(suppress=True, exception_print_fn=(lambda formatted_exception_str: print(f'\t"diba_evt_file" failed with error: {formatted_exception_str}. Skipping.'))):
		#     ## FROM .evt file
		#     ## Load exported epochs from a neuroscope .evt file:
		#     diba_evt_file_replay_epochs: Epoch = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline)
		#     if diba_evt_file_replay_epochs is not None:
		#         replay_epoch_variations.update({
		#             'diba_evt_file': deepcopy(diba_evt_file_replay_epochs),
		#         })
			
		print(f'completed replay extraction, have: {list(replay_epoch_variations.keys())}')
		
		## OUTPUT: replay_epoch_variations
		return replay_epoch_variations    

