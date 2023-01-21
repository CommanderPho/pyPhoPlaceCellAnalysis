import numpy as np
import pandas as pd
from neuropy.core import Epoch
from copy import deepcopy

from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # for decode_specific_epochs
from neuropy.analyses.decoders import epochs_spkcount # for decode_specific_epochs

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData


def epoch_unit_avg_firing_rates(spikes_df, filter_epochs, included_neuron_ids=None, debug_print=False):
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

		# epoch_spikes_df = spikes_df[epoch_start <= spikes_df[spikes_df.spikes.time_variable_name] <= epoch_end]
		epoch_spikes_df = spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)
		# filter_epoch_spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)
		# epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] == epoch_id]

		for aclu, unit_epoch_spikes_df in zip(included_neuron_ids, epoch_spikes_df.spikes.get_split_by_unit(included_neuron_ids=included_neuron_ids)):
			if aclu not in epoch_avg_firing_rate:
				epoch_avg_firing_rate[aclu] = []
			epoch_avg_firing_rate[aclu].append((float(np.shape(unit_epoch_spikes_df)[0]) / (epoch_end - epoch_start)))

	return epoch_avg_firing_rate,	{aclu:np.mean(unit_epoch_avg_frs) for aclu, unit_epoch_avg_frs in epoch_avg_firing_rate.items()}


def fr_index(long_fr, short_fr):
	return ((long_fr - short_fr) / (long_fr + short_fr))

def compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays):

	_, long_mean_laps_frs = epoch_unit_avg_firing_rates(spikes_df, long_laps)
	_, long_mean_replays_frs = epoch_unit_avg_firing_rates(spikes_df, long_replays)

	_, short_mean_laps_frs = epoch_unit_avg_firing_rates(spikes_df, short_laps)
	_, short_mean_replays_frs = epoch_unit_avg_firing_rates(spikes_df, short_replays)

	backup_results_dict = dict(zip(['long_mean_laps_frs', 'long_mean_replays_frs', 'short_mean_laps_frs', 'short_mean_replays_frs'], [long_mean_laps_frs, long_mean_replays_frs, short_mean_laps_frs, short_mean_replays_frs]))
	saveData('temp_2023-01-20_results.pkl', backup_results_dict)

	print(f'reached this point!')
	y_frs_index = {aclu:fr_index(long_mean_laps_frs[aclu], short_mean_laps_frs[aclu]) for aclu in long_mean_laps_frs.keys()}
	x_frs_index = {aclu:fr_index(long_mean_replays_frs[aclu], short_mean_replays_frs[aclu]) for aclu in long_mean_replays_frs.keys()}


	return x_frs_index, y_frs_index

if __name__ == "__main__":
	# x_frs_index, y_frs_index = compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays)

	# saveData('temp_2023-01-20.pkl', backup_dict)
	# dict(zip(['spikes_df', 'long_laps', 'short_laps', 'global_laps', 'long_replays', 'short_replays', 'global_replays'], [spikes_df, long_laps, short_laps, global_laps, long_replays, short_replays, global_replays]))

	

	backup_dict = loadData(r"C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\temp_2023-01-20.pkl")
	spikes_df, long_laps, short_laps, global_laps, long_replays, short_replays, global_replays = backup_dict.values()
	x_frs_index, y_frs_index = compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays)
	print(f'x_frs_index: {x_frs_index}, y_frs_index: {y_frs_index}')

