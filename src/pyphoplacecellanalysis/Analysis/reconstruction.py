# methods of reconstruction/decoding

import numpy as np
import pandas as pd

from pyphocorehelpers.indexing_helpers import build_spanning_bins

# cut_bins = np.linspace(59200, 60800, 9)
# pd.cut(df['column_name'], bins=cut_bins)

# # just want counts of number of occurences of each?
# df['column_name'].value_counts(bins=8, sort=False)


def compute_time_binned_spiking_activity(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
    """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

    Args:
        spikes_df ([type]): [description]
        time_bin_size ([type]): [description]
        debug_print (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'

    out_digitized_variable_bins, out_binning_info = build_spanning_bins(spikes_df[time_variable_name].to_numpy(), max_bin_size=max_time_bin_size, debug_print=debug_print) # np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]
    if debug_print:
        print(f'spikes_df[time_variable_name]: {np.shape(spikes_df[time_variable_name])}\nout_digitized_variable_bins: {np.shape(out_digitized_variable_bins)}')
        # assert (np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]), f'np.shape(out_digitized_variable_bins)[0]: {np.shape(out_digitized_variable_bins)[0]} should equal np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]}'
        print(out_binning_info)

    # spikes_df[time_variable_name].value_counts(bins=out_binning_info.num_bins, sort=False) # fast way to get the binned counts across all cells

    # cut_bins = np.linspace(59200, 60800, 9)
    # pd.cut(df[time_variable_name], bins=cut_bins)
    # pd.cut(df['ext price'], bins=4)
    # time_binned_spikes_df = pd.cut(spikes_df[time_variable_name].to_numpy(), bins=out_digitized_variable_bins)
    spikes_df['binned_time'] = pd.cut(spikes_df[time_variable_name].to_numpy(), bins=out_digitized_variable_bins, include_lowest=True, labels=out_binning_info.bin_indicies[1:]) # same shape as the input data (time_binned_spikes_df: (69142,))
    # print(f'time_binned_spikes_df: {np.shape(time_binned_spikes_df)}, type(time_binned_spikes_df): {type(time_binned_spikes_df)}')
    # spikes_df['binned_time'] = time_binned_spikes_df
    # spikes_df[time_variable_name].to_numpy()[time_binned_spikes_df]

    # df.groupby(bins)['Value'].agg(['count', 'sum'])
    # spikes_df.groupby(time_binned_spikes_df)[time_variable_name].agg('count')

    # any_unit_spike_counts = spikes_df.groupby(['binned_time'])[time_variable_name].agg('count') # unused any cell spike counts

    unit_specific_bin_specific_spike_counts = spikes_df.groupby(['aclu','binned_time'])[time_variable_name].agg('count')
    active_aclu_binned_time_multiindex = unit_specific_bin_specific_spike_counts.index
    active_unique_aclu_values = np.unique(active_aclu_binned_time_multiindex.get_level_values('aclu'))
    unit_specific_binned_spike_counts = np.array([unit_specific_bin_specific_spike_counts[aclu].values for aclu in active_unique_aclu_values]).T # (85841, 40)
    if debug_print:
        print(f'np.shape(unit_specific_spike_counts): {np.shape(unit_specific_binned_spike_counts)}') # np.shape(unit_specific_spike_counts): (40, 85841)

    # unit_specific_spike_counts.get_group(2)

    # spikes_df.groupby(['binned_time']).agg('count')

    # for name, group in spikes_df.groupby(['aclu','binned_time']):
    #     print(f'name: {name}, group: {group}') 

    # neuron_ids, neuron_specific_spikes_dfs = partition(spikes_df, 'aclu')
    # spikes_df.groupby(['aclu','binned_time'])
    # groups.size().unstack()
    # spikes_df._obj.groupby(['aclu'])
    # neuron_ids, neuron_specific_spikes_dfs = partition(spikes_df, 'aclu')

    return unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info




class PlacemapPositionDecoder(object):
    """docstring for PlacemapPositionDecoder."""
    def __init__(self, arg):
        super(PlacemapPositionDecoder, self).__init__()
        self.arg = arg
        
  
        # """Get binned spike counts

        # Parameters
        # ----------
        # bin_size : float, optional
        #     bin size in seconds, by default 0.25

        # Returns
        # -------
        # neuropy.core.BinnedSpiketrains

        # """

        # bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        # spike_counts = np.asarray([np.histogram(_, bins=bins)[0] for _ in self.spiketrains])