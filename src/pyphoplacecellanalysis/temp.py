from pathlib import Path
import numpy as np
import pandas as pd
from neuropy.core import Epoch
from copy import deepcopy

from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # for decode_specific_epochs
from neuropy.analyses.decoders import epochs_spkcount # for decode_specific_epochs
from neuropy.core.session.dataSession import DataSession # for `pipeline_complete_compute_long_short_fr_indicies`
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter # for `plot_long_short_firing_rate_indicies`

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import KnownFilterEpochs # for `pipeline_complete_compute_long_short_fr_indicies`

import matplotlib as mpl
import matplotlib.pyplot as plt

def _epoch_unit_avg_firing_rates(spikes_df, filter_epochs, included_neuron_ids=None, debug_print=False):
	"""Computes the average firing rate for each neuron (unit) in each epoch.

	Args:
		spikes_df (_type_): _description_
		filter_epochs (_type_): _description_
		included_neuron_ids (_type_, optional): _description_. Defaults to None.
		debug_print (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_

	TODO: very inefficient.

	"""
	epoch_avg_firing_rate = {}
	# .spikes.get_unit_spiketrains()
	# .spikes.get_split_by_unit(included_neuron_ids=None)
	# Add add_epochs_id_identity

	if included_neuron_ids is None:
		included_neuron_ids = spikes_df.spikes.neuron_ids

	if isinstance(filter_epochs, pd.DataFrame):
		filter_epochs_df = filter_epochs
	else:
		filter_epochs_df = filter_epochs.to_dataframe()
		
	if debug_print:
		print(f'filter_epochs: {filter_epochs.n_epochs}')
	## Get the spikes during these epochs to attempt to decode from:
	filter_epoch_spikes_df = deepcopy(spikes_df)
	## Add the epoch ids to each spike so we can easily filter on them:
	# filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
	if debug_print:
		print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
	# filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] != -1] # Drop all non-included spikes
	if debug_print:
		print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

	# for epoch_start, epoch_end in filter_epochs:
	for epoch_id in np.arange(np.shape(filter_epochs_df)[0]):
		epoch_start = filter_epochs_df.start.values[epoch_id]
		epoch_end = filter_epochs_df.stop.values[epoch_id]
		epoch_spikes_df = spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)
		# epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] == epoch_id]
		for aclu, unit_epoch_spikes_df in zip(included_neuron_ids, epoch_spikes_df.spikes.get_split_by_unit(included_neuron_ids=included_neuron_ids)):
			if aclu not in epoch_avg_firing_rate:
				epoch_avg_firing_rate[aclu] = []
			epoch_avg_firing_rate[aclu].append((float(np.shape(unit_epoch_spikes_df)[0]) / (epoch_end - epoch_start)))

	return epoch_avg_firing_rate, {aclu:np.mean(unit_epoch_avg_frs) for aclu, unit_epoch_avg_frs in epoch_avg_firing_rate.items()}

def _fr_index(long_fr, short_fr):
	return ((long_fr - short_fr) / (long_fr + short_fr))

def compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays, save_path=None):
	"""A computation for the long/short firing rate index that Kamran and I discussed as one of three metrics during our meeting on 2023-01-19.

	Args:
		spikes_df (_type_): _description_
		long_laps (_type_): _description_
		long_replays (_type_): _description_
		short_laps (_type_): _description_
		short_replays (_type_): _description_

	Returns:
		_type_: _description_


	The backups saved with this function can be loaded via:

	# Load previously computed from data:
	long_mean_laps_frs, long_mean_replays_frs, short_mean_laps_frs, short_mean_replays_frs, x_frs_index, y_frs_index = loadData("data/temp_2023-01-20_results.pkl").values()

	"""
	long_mean_laps_all_frs, long_mean_laps_frs = _epoch_unit_avg_firing_rates(spikes_df, long_laps)
	long_mean_replays_all_frs, long_mean_replays_frs = _epoch_unit_avg_firing_rates(spikes_df, long_replays)

	short_mean_laps_all_frs, short_mean_laps_frs = _epoch_unit_avg_firing_rates(spikes_df, short_laps)
	short_mean_replays_all_frs, short_mean_replays_frs = _epoch_unit_avg_firing_rates(spikes_df, short_replays)

	all_results_dict = dict(zip(['long_mean_laps_frs', 'long_mean_replays_frs', 'short_mean_laps_frs', 'short_mean_replays_frs'], [long_mean_laps_frs, long_mean_replays_frs, short_mean_laps_frs, short_mean_replays_frs])) # all variables
	all_results_dict.update(dict(zip(['long_mean_laps_all_frs', 'long_mean_replays_all_frs', 'short_mean_laps_all_frs', 'short_mean_replays_all_frs'], [long_mean_laps_all_frs, long_mean_replays_all_frs, short_mean_laps_all_frs, short_mean_replays_all_frs]))) # all variables

	y_frs_index = {aclu:_fr_index(long_mean_laps_frs[aclu], short_mean_laps_frs[aclu]) for aclu in long_mean_laps_frs.keys()}
	x_frs_index = {aclu:_fr_index(long_mean_replays_frs[aclu], short_mean_replays_frs[aclu]) for aclu in long_mean_replays_frs.keys()}

	all_results_dict.update(dict(zip(['x_frs_index', 'y_frs_index'], [x_frs_index, y_frs_index]))) # all variables
	# long_mean_laps_all_frs, long_mean_replays_all_frs, short_mean_laps_all_frs, short_mean_replays_all_frs = [np.array(list(fr_dict.values())) for fr_dict in [long_mean_laps_all_frs, long_mean_replays_all_frs, short_mean_laps_all_frs, short_mean_replays_all_frs]]	

	# Save a backup of the data:
	if save_path is not None:
		# save_path: e.g. 'temp_2023-01-20_results.pkl'
		# backup_results_dict = dict(zip(['long_mean_laps_frs', 'long_mean_replays_frs', 'short_mean_laps_frs', 'short_mean_replays_frs', 'x_frs_index', 'y_frs_index'], [long_mean_laps_frs, long_mean_replays_frs, short_mean_laps_frs, short_mean_replays_frs, x_frs_index, y_frs_index])) # all variables
		backup_results_dict = all_results_dict # really all of the variables
		saveData(save_path, backup_results_dict)

	return x_frs_index, y_frs_index, all_results_dict


# from pyphoplacecellanalysis.temp import pipeline_complete_compute_long_short_fr_indicies, compute_long_short_firing_rate_indicies, plot_long_short_firing_rate_indicies

def pipeline_complete_compute_long_short_fr_indicies(curr_active_pipeline, temp_save_filename=None):
	""" wraps `compute_long_short_firing_rate_indicies(...)` to compute the long_short_fr_index for the complete pipeline

	Args:
		curr_active_pipeline (_type_): _description_
		temp_save_filename (_type_, optional): If None, disable caching the `compute_long_short_firing_rate_indicies` results. Defaults to None.

	Returns:
		_type_: _description_
	"""
	active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06' # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
	long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
	# long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
	# long_computation_results, short_computation_results, global_computation_results = [curr_active_pipeline.computation_results[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
	# long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # *_results just shortcut for computation_result['computed_data']

	active_context = active_identifying_session_ctx.adding_context(collision_prefix='fn', fn_name='long_short_firing_rate_indicies')

	spikes_df = curr_active_pipeline.sess.spikes_df
	long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
	
	# try:
	#     long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # NOTE: this includes a few overlapping   epochs since the function to remove overlapping ones seems to be broken
	# except (AttributeError, KeyError) as e:
		# print(f'e: {e}')
	# AttributeError: 'DataSession' object has no attribute 'replay'. Fallback to PBEs?
	# filter_epochs = a_session.pbe # Epoch object
	filter_epoch_replacement_type = KnownFilterEpochs.PBE

	# filter_epochs = a_session.ripple # Epoch object
	# filter_epoch_replacement_type = KnownFilterEpochs.RIPPLE

	print(f'missing .replay epochs, using {filter_epoch_replacement_type} as surrogate replays...')
	active_context = active_context.adding_context(collision_prefix='replay_surrogate', replays=filter_epoch_replacement_type.name)

	## Working:
	# long_replays, short_replays, global_replays = [KnownFilterEpochs.perform_get_filter_epochs_df(sess=a_computation_result.sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration) for a_computation_result in [long_computation_results, short_computation_results, global_computation_results]] # returns Epoch objects
	# New sess.compute_estimated_replay_epochs(...) based method:
	long_replays, short_replays, global_replays = [DataSession.compute_estimated_replay_epochs(curr_active_pipeline.filtered_sessions[an_epoch_name]) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # NOTE: this includes a few overlapping epochs since the function to remove overlapping ones seems to be broken

	## Build the output results dict:
	all_results_dict = dict(zip(['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays'], [long_laps, long_replays, short_laps, short_replays, global_laps, global_replays])) # all variables


	# temp_save_filename = f'{active_context.get_description()}_results.pkl'
	if temp_save_filename is not None:
		print(f'temp_save_filename: {temp_save_filename}')

	x_frs_index, y_frs_index, updated_all_results_dict = compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays, save_path=temp_save_filename) # 'temp_2023-01-24_results.pkl'

	all_results_dict.update(updated_all_results_dict) # append the results dict

	# all_results_dict.update(dict(zip(['x_frs_index', 'y_frs_index'], [x_frs_index, y_frs_index]))) # append the indicies to the results dict

	return x_frs_index, y_frs_index, active_context, all_results_dict # TODO: add to computed_data instead



def plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, fig_save_parent_path=None, debug_print=False):
	""" Plot long|short firing rate index 
	Each datapoint is a neuron.
	"""
	fig = plt.figure(figsize=(8.5, 7.25), num=f'long|short fr indicies_{active_context.get_description(separator="/")}', clear=True)
	plt.scatter(x_frs_index.values(), y_frs_index.values())
	plt.xlabel('$\\frac{L_{R}-S_{R}}{L_{R} + S_{R}}$', fontsize=16)
	plt.ylabel('$\\frac{L_{\\theta}-S_{\\theta}}{L_{\\theta} + S_{\\theta}}$', fontsize=16)
	plt.title('Computed long ($L$)|short($S$) firing rate indicies')
	plt.suptitle(f'{active_context.get_description(separator="/")}')
	# fig = plt.gcf()
	fig.set_size_inches([8.5, 7.25]) # size figure so the x and y labels aren't cut off

	temp_fig_filename = f'{active_context.get_description()}.png'
	if debug_print:
		print(f'temp_fig_filename: {temp_fig_filename}')
	if fig_save_parent_path is None:
		fig_save_parent_path = Path.cwd()

	_temp_full_fig_save_path = fig_save_parent_path.joinpath(temp_fig_filename)

	with ProgressMessagePrinter(_temp_full_fig_save_path, 'Saving', 'plot_long_short_firing_rate_indicies results'):
		fig.savefig(fname=_temp_full_fig_save_path, transparent=True)
	fig.show()
	return fig, _temp_full_fig_save_path



if __name__ == "__main__":
	# saveData('temp_2023-01-20.pkl', backup_dict)
	# dict(zip(['spikes_df', 'long_laps', 'short_laps', 'global_laps', 'long_replays', 'short_replays', 'global_replays'], [spikes_df, long_laps, short_laps, global_laps, long_replays, short_replays, global_replays]))
	backup_dict = loadData(r"C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\temp_2023-01-20.pkl")
	spikes_df, long_laps, short_laps, global_laps, long_replays, short_replays, global_replays = backup_dict.values()
	x_frs_index, y_frs_index, active_context, all_results_dict = compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays, save_path='temp_2023-01-20_results.pkl')
	print(f'x_frs_index: {x_frs_index}, y_frs_index: {y_frs_index}')

