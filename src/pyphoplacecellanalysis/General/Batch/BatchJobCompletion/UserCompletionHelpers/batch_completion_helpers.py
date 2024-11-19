from copy import deepcopy
import shutil
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType
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
from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str

from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names
ShuffleIdx = NewType('ShuffleIdx', int)


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
					   uses=['replace_replay_epochs', 'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function', 'cls.finalize_output_shuffled_wcorr'], used_by=[], creation_date='2024-06-25 22:49', related_items=['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])
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
															additional_session_context=additional_session_context,
															# additional_session_context=IdentifyingContext(custom_suffix=None),
														)
			# with `return_full_decoding_results == False`
			out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple, output_saved_individual_sweep_files_dict = _across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']
			(several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple

			_out_file_paths_dict = {
				'ripple_h5_out_path': out_path, # this seems to duplicate parameters '2024-11-18-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_time_bin_marginals_df).csv'
				'ripple_csv_out_path': ripple_out_path,
				'ripple_csv_time_bin_marginals': ripple_time_bin_marginals_out_path,
				
				## Laps:
				'laps_csv_out_path': laps_out_path,
				'laps_csv_time_bin_marginals_out_path': laps_time_bin_marginals_out_path,
			}

			for a_name, a_path in _out_file_paths_dict.items():
				custom_save_filepaths[a_name] = a_path


			for an_export_file_type, a_path_list in output_saved_individual_sweep_files_dict.items():
				custom_save_filepaths[an_export_file_type] = deepcopy(a_path_list) # a list of paths
				

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
			wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path) = cls.finalize_output_shuffled_wcorr(a_curr_active_pipeline=curr_active_pipeline, decoder_names=decoder_names, custom_suffix=custom_suffix)
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
			
			if minimum_inclusion_fr_Hz is not None:
				## apply the specified `minimum_inclusion_fr_Hz` to the replay_estimation_parameters
				replay_estimation_parameters['min_inclusion_fr_active_thresh'] = minimum_inclusion_fr_Hz
				


			## get the epochs computed normally:
			replay_epoch_variations.update({
				'normal_computed': ensure_Epoch(deepcopy(curr_active_pipeline.sess.replay), metadata={'epochs_source': 'normal_computed',
																							# 'minimum_inclusion_fr_Hz': replay_estimation_parameters['min_inclusion_fr_active_thresh'],
																							'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
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



	# Output Methods _____________________________________________________________________________________________________ #
	
	@function_attributes(short_name=None, tags=['output'], input_requires=[], output_provides=[], uses=[], used_by=['cls.overwrite_replay_epochs_and_recompute'], creation_date='2024-06-27 00:00', related_items=[])
	@classmethod
	def finalize_output_shuffled_wcorr(cls, a_curr_active_pipeline, decoder_names, custom_suffix: str):
		"""
		Gets the shuffled wcorr results and outputs the final histogram for this session

		Usage:
			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import finalize_output_shuffled_wcorr

			decoder_names = deepcopy(track_templates.get_decoder_names())
			wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path) = finalize_output_shuffled_wcorr(curr_active_pipeline=curr_active_pipeline,
																																			decoder_names=decoder_names, custom_suffix=custom_suffix)
		"""
		from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer

		from neuropy.utils.mixins.indexing_helpers import get_dict_subset

		from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme



		wcorr_shuffle_results: SequenceBasedComputationsContainer = a_curr_active_pipeline.global_computation_results.computed_data.get('SequenceBased', None)
		if wcorr_shuffle_results is not None:    
			wcorr_shuffles: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
			wcorr_shuffles: WCorrShuffle = WCorrShuffle(**get_dict_subset(wcorr_shuffles.to_dict(), subset_excludelist=['_VersionedResultMixin_version']))
			a_curr_active_pipeline.global_computation_results.computed_data.SequenceBased.wcorr_ripple_shuffle = wcorr_shuffles
			filtered_epochs_df: pd.DataFrame = deepcopy(wcorr_shuffles.filtered_epochs_df)
			print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles}')
		else:
			print(f'SequenceBased is not computed.')
			wcorr_shuffles = None
			raise ValueError(f'SequenceBased is not computed.')
		
		# wcorr_ripple_shuffle: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline, enable_saving_entire_decoded_shuffle_result=True)

		n_epochs: int = wcorr_shuffles.n_epochs
		print(f'n_epochs: {n_epochs}')
		n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles
		print(f'n_completed_shuffles: {n_completed_shuffles}')
		wcorr_shuffles.compute_shuffles(num_shuffles=2, curr_active_pipeline=a_curr_active_pipeline)
		n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles
		print(f'n_completed_shuffles: {n_completed_shuffles}')
		desired_ripple_decoding_time_bin_size: float = wcorr_shuffle_results.wcorr_ripple_shuffle.all_templates_decode_kwargs['desired_ripple_decoding_time_bin_size']
		print(f'{desired_ripple_decoding_time_bin_size = }')
		# filtered_epochs_df

		# 7m - 200 shuffles
		(_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), _out_shuffle_wcorr_arr = wcorr_shuffles.post_compute(decoder_names=deepcopy(decoder_names))
		wcorr_ripple_shuffle_all_df, all_shuffles_wcorr_df = wcorr_shuffles.build_all_shuffles_dataframes(decoder_names=deepcopy(decoder_names))
		## Prepare for plotting in histogram:
		wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['start', 'stop'], how='any', inplace=False)
		wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL'], how='all', inplace=False)
		wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.convert_dtypes()
		# {'long_best_dir_decoder_IDX': int, 'short_best_dir_decoder_IDX': int}
		wcorr_ripple_shuffle_all_df
		## Gets the absolutely most extreme value from any of the four decoders and uses that
		best_wcorr_max_indices = np.abs(wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].values).argmax(axis=1)
		wcorr_ripple_shuffle_all_df[f'abs_best_wcorr'] = [wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].values[i, best_idx] for i, best_idx in enumerate(best_wcorr_max_indices)] #  np.where(direction_max_indices, wcorr_ripple_shuffle_all_df['long_LR'].filter_epochs[a_column_name].to_numpy(), wcorr_ripple_shuffle_all_df['long_RL'].filter_epochs[a_column_name].to_numpy())
		
		## Add the worst direction for comparison (testing):
		_out_worst_dir_indicies = []
		_LR_indicies = [0, 2]
		_RL_indicies = [1, 3]

		for an_is_most_likely_direction_LR in wcorr_ripple_shuffle_all_df['is_most_likely_direction_LR']:
			if an_is_most_likely_direction_LR:
				_out_worst_dir_indicies.append(_RL_indicies)
			else:
				_out_worst_dir_indicies.append(_LR_indicies)

		_out_worst_dir_indicies = np.vstack(_out_worst_dir_indicies)
		# _out_best_dir_indicies

		wcorr_ripple_shuffle_all_df['long_worst_dir_decoder_IDX'] = _out_worst_dir_indicies[:,0]
		wcorr_ripple_shuffle_all_df['short_worst_dir_decoder_IDX'] = _out_worst_dir_indicies[:,1]

		best_decoder_index = wcorr_ripple_shuffle_all_df['long_best_dir_decoder_IDX'] ## Kamran specified to restrict to the long-templates only for now
		worst_decoder_index = wcorr_ripple_shuffle_all_df['long_worst_dir_decoder_IDX']

		## INPUTS: wcorr_ripple_shuffle_all_df, best_decoder_index
		## MODIFIES: wcorr_ripple_shuffle_all_df
		curr_score_col_decoder_col_names = [f"wcorr_{a_decoder_name}" for a_decoder_name in ['long_LR', 'long_RL', 'short_LR', 'short_RL']]
		wcorr_ripple_shuffle_all_df['wcorr_most_likely'] = [wcorr_ripple_shuffle_all_df[curr_score_col_decoder_col_names].to_numpy()[epoch_idx, a_decoder_idx] for epoch_idx, a_decoder_idx in zip(np.arange(np.shape(wcorr_ripple_shuffle_all_df)[0]), best_decoder_index.to_numpy())]
		wcorr_ripple_shuffle_all_df['abs_most_likely_wcorr'] = np.abs(wcorr_ripple_shuffle_all_df['wcorr_most_likely'])
		wcorr_ripple_shuffle_all_df['wcorr_least_likely'] = [wcorr_ripple_shuffle_all_df[curr_score_col_decoder_col_names].to_numpy()[epoch_idx, a_decoder_idx] for epoch_idx, a_decoder_idx in zip(np.arange(np.shape(wcorr_ripple_shuffle_all_df)[0]), worst_decoder_index.to_numpy())]
		# wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].max(axis=1, skipna=True)

		## OUTPUTS: wcorr_ripple_shuffle_all_df
		wcorr_ripple_shuffle_all_df


		all_shuffles_only_best_decoder_wcorr_df = pd.concat([all_shuffles_wcorr_df[np.logical_and((all_shuffles_wcorr_df['epoch_idx'] == epoch_idx), (all_shuffles_wcorr_df['decoder_idx'] == best_idx))] for epoch_idx, best_idx in enumerate(best_wcorr_max_indices)])

		## OUTPUTS: wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df


		## INPUTS: wcorr_ripple_shuffle, a_curr_active_pipeline
		def a_subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
			""" captures `a_curr_active_pipeline`
			"""
			out_path, out_filename, out_basename = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(rounded_minutes=10), data_identifier_str=data_identifier_str,
																													parent_output_path=parent_output_path, out_extension='.csv')
			export_df.to_csv(out_path)
			return out_path 
		

		#TODO 2024-11-19 03:33: - [ ] `custom_export_df_to_csv_fn`
		custom_export_df_to_csv_fn = a_subfn_custom_export_df_to_csv		

		# standalone save
		out_path, standalone_pkl_filename, standalone_pkl_filepath = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(), data_identifier_str='standalone_wcorr_ripple_shuffle_data_only', parent_output_path=a_curr_active_pipeline.get_output_path(), out_extension='.pkl', suffix_string=f'_{wcorr_shuffles.n_completed_shuffles}')		
		_prev_standalone_pkl_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_wcorr_ripple_shuffle_data_only_{wcorr_shuffles.n_completed_shuffles}.pkl' 
		if _prev_standalone_pkl_filename != standalone_pkl_filename:
			print(f'standalone_pkl_filename:\n\t"{standalone_pkl_filename}"')
			print(f'_prev_standalone_pkl_filename:\n\t"{_prev_standalone_pkl_filename}"')
		# standalone_pkl_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_pkl_filename).resolve() # Path("W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\2024-05-30_0925AM_standalone_wcorr_ripple_shuffle_data_only_1100.pkl")
		print(f'saving to "{standalone_pkl_filepath}"...')
		wcorr_shuffles.save_data(standalone_pkl_filepath)
		## INPUTS: wcorr_ripple_shuffle
		_prev_standalone_mat_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_all_shuffles_wcorr_array.mat' 
		# standalone_mat_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve() # r"W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\2024-06-03_0400PM_standalone_all_shuffles_wcorr_array.mat"
		out_path, standalone_mat_filename, standalone_mat_filepath = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(), data_identifier_str='standalone_all_shuffles_wcorr_array', parent_output_path=a_curr_active_pipeline.get_output_path(), out_extension='.mat')
		if _prev_standalone_mat_filename != standalone_mat_filename:
			print(f'standalone_mat_filepath:\n\t"{standalone_mat_filepath}"')
			print(f'_prev_standalone_mat_filename:\n\t"{_prev_standalone_mat_filename}"')
		#TODO 2024-11-19 03:25: - [ ] Previously used `custom_suffix`
		wcorr_shuffles.save_data_mat(filepath=standalone_mat_filepath, **{'session': a_curr_active_pipeline.get_session_context().to_dict()})

		try:
			# active_context = a_curr_active_pipeline.get_session_context()
			# additional_session_context = a_curr_active_pipeline.get_session_additional_parameters_context()
			complete_session_context, (session_context, additional_session_context) = a_curr_active_pipeline.get_complete_session_context()
			active_context = complete_session_context
			session_name: str = a_curr_active_pipeline.session_name
			export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=a_curr_active_pipeline.get_output_path().resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=a_curr_active_pipeline,
														#    source='diba_evt_file',
														source='compute_diba_quiescent_style_replay_events',
														custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
														)
			ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
			print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
			# callback_outputs['ripple_WCorrShuffle_df_export_CSV_path'] = ripple_WCorrShuffle_df_export_CSV_path
		except Exception as e:
			raise e
			# exception_info = sys.exc_info()
			# err = CapturedException(e, exception_info)
			# print(f"ERROR: encountered exception {err} while trying to perform wcorr_ripple_shuffle.export_csvs(parent_output_path='{self.collected_outputs_path.resolve()}', ...) for {curr_session_context}")
			# ripple_WCorrShuffle_df_export_CSV_path = None # set to None because it failed.
			# if self.fail_on_exception:
			#     raise err.exc
			
		# wcorr_ripple_shuffle.discover_load_and_append_shuffle_data_from_directory(save_directory=curr_active_pipeline.get_output_path().resolve())
		# active_context = curr_active_pipeline.get_session_context()
		# session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
		# session_name: str = curr_active_pipeline.session_name
		# export_files_dict = wcorr_ripple_shuffle.export_csvs(parent_output_path=collected_outputs_path.resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=curr_active_pipeline)
		# export_files_dict

		return wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path)


	@function_attributes(short_name=None, tags=['plotting', 'histogram', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=['compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function'], creation_date='2024-06-27 22:43', related_items=[])
	@classmethod
	def plot_replay_wcorr_histogram(cls, df: pd.DataFrame, plot_var_name: str, all_shuffles_only_best_decoder_wcorr_df: Optional[pd.DataFrame]=None, footer_annotation_text=None):
		""" Create horizontal histogram Takes outputs of finalize_output_shuffled_wcorr to plot a histogram like the Diba 2007 paper

		Usage:

			from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_replay_wcorr_histogram
			plot_var_name: str = 'abs_best_wcorr'
			footer_annotation_text = f'{curr_active_pipeline.get_session_context()}<br>{params_description_str}'

			fig = plot_replay_wcorr_histogram(df=wcorr_ripple_shuffle_all_df, plot_var_name=plot_var_name,
				all_shuffles_only_best_decoder_wcorr_df=all_shuffles_only_best_decoder_wcorr_df, footer_annotation_text=footer_annotation_text)

			# Save figure to disk:
			_out_result = curr_active_pipeline.output_figure(a_fig_context, fig=fig)
			_out_result

			# Show the figure
			fig.show()

		"""
		import plotly.io as pio
		import plotly.express as px
		import plotly.graph_objects as go
		from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers


		resolution_multiplier = 1
		fig_size_kwargs = {'width': resolution_multiplier*1650, 'height': resolution_multiplier*480}
		is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
		pio.templates.default = template

		# fig = px.histogram(df, x=plot_var_name) # , orientation='h'
		df = deepcopy(df) # pd.DataFrame(data)
		df = df.dropna(subset=[plot_var_name], how='any', inplace=False)

		histogram_kwargs = dict(histnorm='percent', nbinsx=30)
		fig = go.Figure()
		wcorr_ripple_hist_trace = fig.add_trace(go.Histogram(x=df[plot_var_name], name='Observed Replay', **histogram_kwargs))

		if all_shuffles_only_best_decoder_wcorr_df is not None:
			shuffle_trace = fig.add_trace(go.Histogram(x=all_shuffles_only_best_decoder_wcorr_df['shuffle_wcorr'], name='Shuffle', **histogram_kwargs))
		
		# Overlay both histograms
		fig = fig.update_layout(barmode='overlay')
		# Reduce opacity to see both histograms
		fig = fig.update_traces(opacity=0.75)

		# Update layout for better visualization
		fig = fig.update_layout(
			title=f'Horizontal Histogram of "{plot_var_name}"',
			xaxis_title=plot_var_name,
			yaxis_title='Percent Count',
			# yaxis_title='Count',
		)

		## Add the metadata for the replays being plotted:
		# new_replay_epochs.metadata
		if footer_annotation_text is None:
			footer_annotation_text = ''

		# Add footer text annotation
		fig = fig.update_layout(
			annotations=[
				dict(
					x=0,
					y=-0.25,
					xref='paper',
					yref='paper',
					text=footer_annotation_text,
					showarrow=False,
					xanchor='left',
					yanchor='bottom'
				)
			]
		)

		fig = fig.update_layout(fig_size_kwargs)
		return fig



