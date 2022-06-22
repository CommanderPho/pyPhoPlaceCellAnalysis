import numpy as np
from pyphocorehelpers.indexing_helpers import build_pairwise_indicies
from scipy.ndimage import gaussian_filter1d

# plotting:
import matplotlib.pyplot as plt



def _compute_single_lap_reliability(curr_lap_filtered_spikes_df, variable_extents_array, min_subdivision_resolution:float = 0.01, spike_blurring:float = 80.0, span_width:int=None, debug_print=False):
    """ """
    # for now, just do x (first variable)
    curr_variable_extents = variable_extents_array[0]
    num_subdivisions = int(np.ceil((curr_variable_extents[1] - curr_variable_extents[0])/min_subdivision_resolution))
    actual_subdivision_step_size = (curr_variable_extents[1] - curr_variable_extents[0]) / float(num_subdivisions) # the actual exact size of the bin
    
    if debug_print:
        print(f'for min_subdivision_resolution: {min_subdivision_resolution} -> num_subdivisions: {num_subdivisions}, actual_subdivision_step_size: {actual_subdivision_step_size}')
    out_indicies = np.arange(num_subdivisions)
    out_digitized_position_bins = np.linspace(curr_variable_extents[0], curr_variable_extents[1], num_subdivisions, dtype=float)#.astype(float)
    out_within_lap_spikes_overlap = np.zeros_like(out_digitized_position_bins, dtype=float)

    curr_digitized_variable = np.digitize(curr_lap_filtered_spikes_df['x'].to_numpy(), out_digitized_position_bins) # these are indicies
    # perform span_width: a span is a fixed width for each spike instead of a single bin wide delta function (using a rectangle function instead)
    if (span_width is not None) and (span_width > 0.0):
        span_range = np.arange(1, span_width)
        # span_ranges = [i-span_range for i in curr_digitized_variable]
        for i, value in enumerate(curr_digitized_variable):
            out_within_lap_spikes_overlap[value-span_range] += 5.0 # set spikes to 1.0
            out_within_lap_spikes_overlap[value] += 10.0 # set spikes to 1.0
            out_within_lap_spikes_overlap[value+span_range] += 5.0 # set spikes to 1.0
    else:
        out_within_lap_spikes_overlap[curr_digitized_variable] = 10.0 # set spikes to 1.0

    # perform spike_blurring:
    if (spike_blurring is not None) and (spike_blurring > 0.0):
        # convert spike_blurring from real units (which is how it's input) to bins
        spike_blurring_step_units = (spike_blurring / actual_subdivision_step_size)
        if debug_print:
            print(f'spike_blurring: {spike_blurring}, spike_blurring_step_units: {spike_blurring_step_units}')
        out_within_lap_spikes_overlap = gaussian_filter1d(out_within_lap_spikes_overlap, sigma=spike_blurring_step_units)
    else:
        if debug_print:
            print('spike blurring disabled because spike_blurring is set to None or 0.0')

    # np.convolve(out[curr_digitized_variable], np.
    return out_indicies, out_digitized_position_bins, out_within_lap_spikes_overlap

def compute_lap_to_lap_reliability(active_pf, filtered_spikes_df, lap_ids, cellind, min_subdivision_resolution:float = 0.01, plot_results=False, plot_horizontal=True, debug_print=True):
    """ Computes the reliability of a placecell from lap-to-lap
    
    Example:    
        curr_result_label = 'maze1'
        sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
        # sess = curr_kdiba_pipeline.sess

        curr_neuron_IDX = 2 
        # curr_neuron_IDX = 3 # good for end platform analysis
        curr_cell_ID = sess.spikes_df.spikes.neuron_ids[curr_neuron_IDX]
        print(f'curr_neuron_IDX: {curr_neuron_IDX}, curr_cell_ID: {curr_cell_ID}')

        # pre-filter by spikes that occur in one of the included laps for the filtered_spikes_df
        filtered_spikes_df = sess.spikes_df.copy()
        time_variable_name = filtered_spikes_df.spikes.time_variable_name # 't_rel_seconds'

        lap_ids = sess.laps.lap_id
        # lap_flat_idxs = sess.laps.get_lap_flat_indicies(lap_ids)

        out_indicies, out_digitized_position_bins, out, all_laps_reliability = compute_lap_to_lap_reliability(curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf2D'], filtered_spikes_df, lap_ids, curr_neuron_IDX, debug_print=False);


    """
    time_variable_name = filtered_spikes_df.spikes.time_variable_name # 't_rel_seconds'

    if active_pf.ndim < 2:
        variable_array = [active_pf.x]
        label_array = ["X position (cm)"]
    else:
        variable_array = [active_pf.x, active_pf.y]
        label_array = ["X position (cm)", "Y position (cm)"]
        
    # compute extents:
    variable_extents_array = [(np.nanmin(a_var), np.nanmax(a_var)) for a_var in variable_array]
    # for now, just do x (first variable)
    curr_variable_extents = variable_extents_array[0]
    num_subdivisions = int(np.ceil((curr_variable_extents[1] - curr_variable_extents[0])/min_subdivision_resolution))
    if debug_print:
        print(f'for min_subdivision_resolution: {min_subdivision_resolution} -> num_subdivisions: {num_subdivisions}')
    # Pre-allocate output variables:
    out_indicies = np.arange(num_subdivisions)
    out_digitized_position_bins = np.linspace(curr_variable_extents[0], curr_variable_extents[1], num_subdivisions, dtype=float)#.astype(float)
    out_within_lap_spikes_overlap = np.zeros([num_subdivisions, len(lap_ids)], dtype=float)

    # all spike times and positions for the specified cellind:
    spk_pos_, spk_t_ = active_pf.spk_pos[cellind], active_pf.spk_t[cellind]
    
    # filtered_spikes_df = filtered_spikes_df[np.isin(filtered_spikes_df['lap'], included_lap_ids)] # get only the spikes that occur in one of the included laps for the filtered_spikes_df
    if debug_print:
        print('filtering spikes by times in pf2D', end=' ')
    filtered_spikes_df = filtered_spikes_df[np.isin(filtered_spikes_df[time_variable_name].to_numpy(), spk_t_)] # get only the spikes that occur in one of the included laps for the filtered_spikes_df
    if debug_print:
        print('done.')

    # testing only:
    # lap_ids = [lap_ids[0], lap_ids[1]] # TODO: TEST ONLY FIRST ELEMENT
    flat_lap_idxs = np.arange(len(lap_ids))

    should_share_non_common_axes_lims = False
    if plot_results:
        if plot_horizontal:
            fig, axs = plt.subplots(1, len(lap_ids), sharex=should_share_non_common_axes_lims, sharey=True, figsize=(40, 24))
        else:
            # vertical
            fig, axs = plt.subplots(len(lap_ids), 1, sharex=True, sharey=should_share_non_common_axes_lims, figsize=(24, 40))

    for lap_idx, lap_ID in zip(flat_lap_idxs, lap_ids):
        # for each lap
        curr_lap_filtered_spikes_df = filtered_spikes_df[filtered_spikes_df['lap'] == lap_ID] # get only the spikes that occur in one of the included laps for the filtered_spikes_df
        if debug_print:
            print(f'{lap_idx},{lap_ID}: spikes {np.shape(curr_lap_filtered_spikes_df)[0]}')
        out_indicies, out_digitized_position_bins, out_within_lap_spikes_overlap[:, lap_idx] = _compute_single_lap_reliability(curr_lap_filtered_spikes_df, variable_extents_array, min_subdivision_resolution=min_subdivision_resolution, spike_blurring=5.0, span_width=None, debug_print=debug_print)
        # Debug Plotting to test the produced output:
        if plot_results:
            if plot_horizontal:
                axs[lap_idx].plot(out_within_lap_spikes_overlap[:, lap_idx], out_digitized_position_bins)
            else:
                # vertical
                axs[lap_idx].plot(out_digitized_position_bins, out_within_lap_spikes_overlap[:, lap_idx])

    # Actual Computations of Reliability:
    out_pairwise_pair_results = np.zeros_like(out_within_lap_spikes_overlap)
    
    # do simple diff:
    laps_spikes_overlap_diff = np.diff(out_within_lap_spikes_overlap, axis=1) # the element-wise diff of the overlap. Shows changes.
    out_pairwise_pair_results[:, 1:] = laps_spikes_overlap_diff
    # out_pairwise_pair_results[:, -1] = np.zeros_like(out_within_lap_spikes_overlap[:,0])
    
    # do custom pairwise operation:
#     for first_item_lap_idx, next_item_lap_idx in list(out_pairwise_flat_lap_indicies):
#         first_item = out_within_lap_spikes_overlap[:, first_item_lap_idx]
#         next_item = out_within_lap_spikes_overlap[:, next_item_lap_idx]
#         out_pairwise_pair_results[:, next_item_lap_idx] = (first_item * next_item) # the result should be stored in the index of the second item, if we're doing the typical backwards style differences.
#         # print(f'np.max(out_pairwise_pair_results[:, next_item_lap_idx]): {np.max(out_pairwise_pair_results[:, next_item_lap_idx])}')
        
    if debug_print: 
        print(f'max out: {np.max(out_pairwise_pair_results)}')
        
    # add to the extant plot as a new color:
    if plot_results:
        for lap_idx, lap_ID in zip(flat_lap_idxs, lap_ids):
            # curr_lap_alt_ax = axs[lap_idx]
            if plot_horizontal:
                curr_lap_alt_ax = axs[lap_idx].twiny()
                curr_lap_alt_ax.plot(out_pairwise_pair_results[:, lap_idx], out_digitized_position_bins, '--r')
            else:
                # vertical
                curr_lap_alt_ax = axs[lap_idx].twinx()
                curr_lap_alt_ax.plot(out_digitized_position_bins, out_pairwise_pair_results[:, lap_idx], '--r')
            
    cum_laps_reliability = np.cumprod(out_within_lap_spikes_overlap, axis=1)
    all_laps_reliability = np.prod(out_within_lap_spikes_overlap, axis=1, keepdims=True)
    
    if plot_results:
        fig_result, axs_result = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(24, 40))
        axs_result[0].plot(out_digitized_position_bins, all_laps_reliability, 'r')
        axs_result[1].plot(out_digitized_position_bins, cum_laps_reliability, 'r')
    
    return out_indicies, out_digitized_position_bins, out_within_lap_spikes_overlap
    


# def compute_reliability_metrics(out_indicies, out_digitized_position_bins, out_within_lap_spikes_overlap, debug_print=False, plot_results=False):
#     """ Takes input from compute_lap_to_lap_reliability(...) to build the actual reliability metrics """
#     # Actual Computations of Reliability:
#     out_pairwise_pair_results = np.zeros_like(out_within_lap_spikes_overlap)
    
#     # do simple diff:
#     laps_spikes_overlap_diff = np.diff(out_within_lap_spikes_overlap, axis=1) # the element-wise diff of the overlap. Shows changes.
#     out_pairwise_pair_results[:, 1:] = laps_spikes_overlap_diff
#     # out_pairwise_pair_results[:, -1] = np.zeros_like(out_within_lap_spikes_overlap[:,0])
    
#     # do custom pairwise operation:
# #     for first_item_lap_idx, next_item_lap_idx in list(out_pairwise_flat_lap_indicies):
# #         first_item = out_within_lap_spikes_overlap[:, first_item_lap_idx]
# #         next_item = out_within_lap_spikes_overlap[:, next_item_lap_idx]
# #         out_pairwise_pair_results[:, next_item_lap_idx] = (first_item * next_item) # the result should be stored in the index of the second item, if we're doing the typical backwards style differences.
# #         # print(f'np.max(out_pairwise_pair_results[:, next_item_lap_idx]): {np.max(out_pairwise_pair_results[:, next_item_lap_idx])}')

#     if debug_print: 
#         print(f'max out: {np.max(out_pairwise_pair_results)}')
        
#     lap_ids 
#     flat_lap_idxs = np.arange(len(lap_ids))
    
    
#     # add to the extant plot as a new color:
#     if plot_results:
#         for lap_idx, lap_ID in zip(flat_lap_idxs, lap_ids):
#             # curr_lap_alt_ax = axs[lap_idx]
#             if plot_horizontal:
#                 curr_lap_alt_ax = axs[lap_idx].twiny()
#                 curr_lap_alt_ax.plot(out_pairwise_pair_results[:, lap_idx], out_digitized_position_bins, '--r')
#             else:
#                 # vertical
#                 curr_lap_alt_ax = axs[lap_idx].twinx()
#                 curr_lap_alt_ax.plot(out_digitized_position_bins, out_pairwise_pair_results[:, lap_idx], '--r')
            
#     cum_laps_reliability = np.cumprod(out_within_lap_spikes_overlap, axis=1)
#     all_laps_reliability = np.prod(out_within_lap_spikes_overlap, axis=1, keepdims=True)
    
#     if plot_results:
#         fig_result, axs_result = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(24, 40))
#         axs_result[0].plot(out_digitized_position_bins, all_laps_reliability, 'r')
#         axs_result[1].plot(out_digitized_position_bins, cum_laps_reliability, 'r')

    
    