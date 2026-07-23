 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
from copy import deepcopy
import numpy as np
import pandas as pd
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from attrs import define, field, Factory

import neuropy.utils.type_aliases as types
from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

from pyphocorehelpers.indexing_helpers import build_pairwise_indicies
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix

# plotting:

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    ## typehinting only imports here
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes



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

@function_attributes(short_name=None, tags=['original'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2022-01-01 00:00', related_items=[])
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
    # for first_item_lap_idx, next_item_lap_idx in list(out_pairwise_flat_lap_indicies):
    #     first_item = out_within_lap_spikes_overlap[:, first_item_lap_idx]
    #     next_item = out_within_lap_spikes_overlap[:, next_item_lap_idx]
    #     out_pairwise_pair_results[:, next_item_lap_idx] = (first_item * next_item) # the result should be stored in the index of the second item, if we're doing the typical backwards style differences.
    #     # print(f'np.max(out_pairwise_pair_results[:, next_item_lap_idx]): {np.max(out_pairwise_pair_results[:, next_item_lap_idx])}')

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

    
    

# ==================================================================================================================== #
# 2024-02-02 - Trial-by-trial Correlation Matrix C                                                                     #
# ==================================================================================================================== #



@metadata_attributes(short_name=None, tags=['trial-by-trial', 'lap-stability'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-12 00:00', related_items=[])
@define(slots=False)
class TrialByTrialActivity:
    """ 2024-02-12 - Computes lap-by-lap placefields and helps display correlation matricies and such.
    
    """
    active_epochs_df: pd.DataFrame = field()
    C_trial_by_trial_correlation_matrix: NDArray[ND.Shape["N_ACLUS, N_EPOCHS, N_EPOCHS"], Any] = field(metadata={'shape':('n_neurons', 'n_epochs', 'n_epochs')})
    z_scored_tuning_map_matrix: NDArray[ND.Shape["N_EPOCHS, N_ACLUS, N_POS_BINS"], Any] = field(metadata={'shape':('n_epochs', 'n_neurons', 'n_pos_bins')})
    aclu_to_matrix_IDX_map: Dict = field() # factory=Factory(dict)
    neuron_ids: NDArray = field(metadata={'shape':('n_neurons',)})
    
    @property 
    def stability_score(self) -> NDArray:
        """ nanmedian(C, axis=(1,2)) # Over the two epochs dimensions... is this a double counting issue that would effect the median?"""
        return np.nanmedian(self.C_trial_by_trial_correlation_matrix, axis=(1,2))
    
    @property 
    def aclu_to_stability_score_dict(self) -> Dict[int, NDArray]:
        return dict(zip(self.neuron_ids, self.stability_score))
    

    def sliced_by_neuron_id(self, included_neuron_ids: NDArray) -> "TrialByTrialActivity":
        _obj = deepcopy(self)
        assert np.all([(v in _obj.neuron_ids) for v in included_neuron_ids]), f"All included_neuron_ids must already exist in the object: included_neuron_ids: {included_neuron_ids}\n\t_obj.neuron_ids: {_obj.neuron_ids}"
        n_aclus = len(included_neuron_ids)
        # is_neuron_id_included = np.isin(included_neuron_ids, _obj.neuron_ids)
        is_neuron_id_included = np.where(np.isin(included_neuron_ids, _obj.neuron_ids))[0]
        _obj.z_scored_tuning_map_matrix = _obj.z_scored_tuning_map_matrix[:, is_neuron_id_included, :]
        _obj.C_trial_by_trial_correlation_matrix = _obj.C_trial_by_trial_correlation_matrix[is_neuron_id_included, :, :]
        _obj.aclu_to_matrix_IDX_map = dict(zip(included_neuron_ids, np.arange(n_aclus)))
        _obj.neuron_ids = deepcopy(included_neuron_ids)
        # z_scored_tuning_map_matrix = deepcopy(z_scored_tuning_map_matrix)
        return _obj
    

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"
    

    @classmethod
    def compute_spatial_binned_activity_via_pfdt(cls, active_pf_dt: PfND_TimeDependent, epochs_df: pd.DataFrame, included_neuron_IDs=None):
        """ 2024-02-01 - Use pfND_dt to compute spatially binned activity during the epochs.
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_spatial_binned_activity_via_pfdt
            
            if 'pf1D_dt' not in curr_active_pipeline.computation_results[global_epoch_name].computed_data:
                # if `KeyError: 'pf1D_dt'` recompute
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)


            active_pf_1D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'])
            active_pf_2D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt'])


            laps_df = deepcopy(global_any_laps_epochs_obj.to_dataframe())
            n_laps = len(laps_df)

            active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_1D_dt)
            # active_pf_dt = deepcopy(active_pf_2D_dt) # 2D
            historical_snapshots = compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=laps_df)

        """
        use_pf_dt_obj = False

        if included_neuron_IDs is None:
            included_neuron_IDs = deepcopy(active_pf_dt.included_neuron_IDs) # this may be under-included. Is there like an "all-times-neuron_IDs?"
            
        
        if isinstance(epochs_df, (pd.DataFrame, Epoch)):
            epochs_df = ensure_dataframe(epochs_df)
            # dataframes are treated weird by PfND_dt, convert to basic numpy array of shape (n_epochs, 2)
            time_intervals = epochs_df[['start', 'stop']].to_numpy() # .shape # (n_epochs, 2)
        else:
            time_intervals = epochs_df # assume already a numpy array
            
        assert np.shape(time_intervals)[-1] == 2
        n_epochs: int = np.shape(time_intervals)[0]
            
        ## Entirely independent computations for binned_times:
        if use_pf_dt_obj:
            active_pf_dt.reset()

        # if included_neuron_IDs is not None:
        #     # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        
        if not use_pf_dt_obj:
            historical_snapshots = {} # build a dict<float:PlacefieldSnapshot>

        for start_t, end_t in time_intervals:
            ## Inline version that reuses active_pf_1D_dt directly:
            if use_pf_dt_obj:
                # active_pf_1D_dt.update(end_t, should_snapshot=True) # use this because it correctly integrates over [0, end_t] instead of [start_t, end_t]
                # active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=True, should_snapshot=True)
                historical_snapshots[float(end_t)] = active_pf_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=False, should_snapshot=False) # Integrates each [start_t, end_t] independently
            else:
                # Static version that calls PfND_TimeDependent.perform_time_range_computation(...) itself using just the computed variables of `active_pf_1D_dt`:
                all_time_filtered_spikes_df: pd.DataFrame = deepcopy(active_pf_dt.all_time_filtered_spikes_df).spikes.sliced_by_neuron_id(included_neuron_IDs)
                historical_snapshots[float(end_t)] = PfND_TimeDependent.perform_time_range_computation(all_time_filtered_spikes_df, active_pf_dt.all_time_filtered_pos_df, position_srate=active_pf_dt.position_srate,
                                                                            xbin=active_pf_dt.xbin, ybin=active_pf_dt.ybin,
                                                                            start_time=start_t, end_time=end_t,
                                                                            included_neuron_IDs=included_neuron_IDs, active_computation_config=active_pf_dt.config, override_smooth=active_pf_dt.smooth)

        # {1.9991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x16c2b74fb20>, 2.4991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x168acfb3bb0>, ...}
        if use_pf_dt_obj:
            historical_snapshots = active_pf_dt.historical_snapshots

        epoch_pf_results_dict = {'historical_snapshots': historical_snapshots}
        epoch_pf_results_dict['num_position_samples_occupancy'] = np.stack([placefield_snapshot.num_position_samples_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['seconds_occupancy'] = np.stack([placefield_snapshot.seconds_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['normalized_occupancy'] = np.stack([placefield_snapshot.normalized_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['spikes_maps_matrix'] = np.stack([placefield_snapshot.spikes_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        # active_lap_pf_results_dict['snapshot_occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in active_lap_pf_results_dict['historical_snapshots'].values()])

        # len(historical_snapshots)
        return epoch_pf_results_dict


    @classmethod
    def compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, occupancy_weighted_tuning_maps_matrix: NDArray[ND.Shape["N_ACLUS, N_TRIALS, N_XBINS"], Any], included_neuron_IDs=None, epsilon_value: float = 1e-12) -> Tuple[NDArray, NDArray, Dict]:
        """ 2024-02-02 - computes the Trial-by-trial Correlation Matrix C 
        
        Returns:
            C_trial_by_trial_correlation_matrix: .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
            z_scored_tuning_map_matrix

        Usage:
            from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix = TrialByTrialActivity.compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix)

        """
        if included_neuron_IDs is None:
            neuron_ids = deepcopy(np.array(active_pf_dt.ratemap.neuron_ids))
        else:
            neuron_ids = np.array(included_neuron_IDs)
            

        n_aclus = len(neuron_ids)
        n_xbins = len(active_pf_dt.xbin_centers)

        assert np.shape(occupancy_weighted_tuning_maps_matrix)[1] == n_aclus
        assert np.shape(occupancy_weighted_tuning_maps_matrix)[2] == n_xbins

        
        # Assuming 'occupancy_weighted_tuning_maps_matrix' is your dataset with shape (trials, positions)
        # Z-score along the position axis (axis=1)
        position_axis_idx: int = 2 ## 
        z_scored_tuning_map_matrix: NDArray[ND.Shape["N_TRIALS, N_ACLUS, N_XBINS"], Any] = (occupancy_weighted_tuning_maps_matrix - np.nanmean(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True)) / ((np.nanstd(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True))+epsilon_value)

        # trial-by-trial correlation matrix C
        M = float(n_xbins)
        C_list = []
        for i, aclu in enumerate(neuron_ids):
            A_i = np.squeeze(z_scored_tuning_map_matrix[:,i,:])
            C_i = (1/(M-1)) * (A_i @ A_i.T) # Perform matrix multiplication using the @ operator
            # C_i.shape # (n_epochs, n_epochs) - (84, 84) - gives the correlation between each epoch and the others
            C_list.append(C_i)
        # occupancy_weighted_tuning_maps_matrix

        C_trial_by_trial_correlation_matrix: NDArray[ND.Shape["N_ACLUS, N_EPOCHS, N_EPOCHS"], Any] = np.stack(C_list, axis=0) # .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
        # outputs: C_trial_by_trial_correlation_matrix

        # n_laps: int = len(laps_unique_ids)
        aclu_to_matrix_IDX_map: Dict[int, int] = dict(zip(neuron_ids, np.arange(n_aclus)))

        return C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map


    ## MAIN CALL:
    @classmethod
    def directional_compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, directional_lap_epochs_dict: Dict[types.DecoderName, Epoch], included_neuron_IDs=None) -> Dict[types.DecoderName, "TrialByTrialActivity"]:
        """ Computes the trial-by-trial (lap-by-lap) correlation for each cell

        
        2024-02-02 - 10pm - Have global version working but want seperate directional versions. Seperately do `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`:
        
        Usage:
            from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
            from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity


            ## INPUTS: curr_active_pipeline, track_templates, global_epoch_name, (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)
            any_decoder_neuron_IDs = deepcopy(track_templates.any_decoder_neuron_IDs)
            any_decoder_neuron_IDs

            # track_templates.shared_LR_aclus_only_neuron_IDs
            # track_templates.shared_RL_aclus_only_neuron_IDs

            ## Directional Trial-by-Trial Activity:
            if 'pf1D_dt' not in curr_active_pipeline.computation_results[global_epoch_name].computed_data:
                # if `KeyError: 'pf1D_dt'` recompute
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

            active_pf_1D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'])
            active_pf_2D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt'])

            active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_1D_dt)
            # active_pf_dt.res
            # Limit only to the placefield aclus:
            active_pf_dt = active_pf_dt.get_by_id(ids=any_decoder_neuron_IDs)

            # active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_2D_dt) # 2D
            long_LR_name, long_RL_name, short_LR_name, short_RL_name = track_templates.get_decoder_names()

            directional_lap_epochs_dict = dict(zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)))
            directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = TrialByTrialActivity.directional_compute_trial_by_trial_correlation_matrix(active_pf_dt=active_pf_dt, directional_lap_epochs_dict=directional_lap_epochs_dict, included_neuron_IDs=any_decoder_neuron_IDs)

            ## OUTPUTS: directional_active_lap_pf_results_dicts


        """
        directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = {}

        # # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        # if included_neuron_IDs is not None:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()


        # Seperately do each decoder as they represent laps from each direction and track (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj):
        for an_epoch_name, active_laps_epoch in directional_lap_epochs_dict.items():
            active_laps_df = deepcopy(active_laps_epoch.to_dataframe()) # ensure_dataframe
            active_lap_pf_results_dict = cls.compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=active_laps_df, included_neuron_IDs=included_neuron_IDs)
            # Unpack the variables:
            historical_snapshots = active_lap_pf_results_dict['historical_snapshots']
            occupancy_weighted_tuning_maps_matrix = active_lap_pf_results_dict['occupancy_weighted_tuning_maps'] # .shape: (n_epochs, n_aclus, n_xbins) - (84, 80, 56)
            # 2024-02-02 - Trial-by-trial Correlation Matrix C
            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map = cls.compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix, included_neuron_IDs=included_neuron_IDs)
            neuron_ids = np.array(list(aclu_to_matrix_IDX_map.keys()))
            
            # directional_active_lap_pf_results_dicts[an_epoch_name] = (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) # currently discards: occupancy_weighted_tuning_maps_matrix, historical_snapshots, active_lap_pf_results_dict, active_laps_df
            directional_active_lap_pf_results_dicts[an_epoch_name] = TrialByTrialActivity(active_epochs_df=active_laps_df, C_trial_by_trial_correlation_matrix=C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix=z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map=aclu_to_matrix_IDX_map, neuron_ids=neuron_ids)
            
        return directional_active_lap_pf_results_dicts


    @classmethod
    def plot_napari_trial_by_trial_correlation_matrix(cls, directional_active_lap_pf_results_dicts: Dict[types.DecoderName, "TrialByTrialActivity"], include_trial_by_trial_correlation_matrix:bool=True):
        """ Produces 5 Napari windows to display the trial-by-trial correlation matricies for each of the decoders.

        aTbyT:TrialByTrialActivity = a_trial_by_trial_result.directional_active_lap_pf_results_dicts['long_LR']
        aTbyT.C_trial_by_trial_correlation_matrix.shape # (40, 21, 21)
        aTbyT.z_scored_tuning_map_matrix.shape # (21, 40, 57) (n_epochs, n_neurons, n_pos_bins)

        (directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict) = aTbyT.plot_napari_trial_by_trial_correlation_matrix(directional_active_lap_pf_results_dicts=a_trial_by_trial_result.directional_active_lap_pf_results_dicts)
        """
        import napari
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_plot_directional_trial_by_trial_activity_viz, napari_trial_by_trial_activity_viz, napari_export_image_sequence

        ## Directional
        directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict = napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts, include_trial_by_trial_correlation_matrix=include_trial_by_trial_correlation_matrix)
    
        for a_decoder_name, a_result in directional_active_lap_pf_results_dicts.items():
            ## Global:
            viewer, image_layer_dict = napari_trial_by_trial_activity_viz(a_result.z_scored_tuning_map_matrix, a_result.C_trial_by_trial_correlation_matrix, title=f'Trial-by-trial Correlation Matrix C - Decoder {a_decoder_name}', axis_labels=('aclu', 'lap', 'xbin')) # GLOBAL
            
        ## Global:
        # viewer, image_layer_dict = napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix, title='Trial-by-trial Correlation Matrix C', axis_labels=('aclu', 'lap', 'xbin')) # GLOBAL

        return (directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict)


# ==================================================================================================================== #
# 2024-02-01 - Spatial Information                                                                                     #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

def _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy):
    """ function to calculate Spatial Information (SI) score
    
    # f_i is the trial-averaged activity per position bin i -- sounds like the average number of spikes in each position bin within the trial

    # f is the mean activity rate over the whole session, computed as the sum of f_i * p_i over all N (position) bins

    ## What they call "p_i" - "occupancy probability per position bin per trial" ([Sosa et al., 2023, p. 23](zotero://select/library/items/I5FLMP5R)) ([pdf](zotero://open-pdf/library/items/C3Y8AKEB?page=23&annotation=GAHX9PYH))
    occupancy_probability = a_spikes_bin_counts_mat.copy()
    occupancy_probability = occupancy_probability / occupancy_probability.sum(axis=1, keepdims=True) # quotient is "total number of samples in each trial"
    occupancy_probability

    # We then summed the occupancy probabilities across trials and divided by the total per session to get an occupancy probability per position bin per session

    # To get the spatial “tuning curve” over the session, we averaged the activity in each bin across trials

    Usage:    
    SI = calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy)
    """
    ## SI Calculator: fi/<f>
    p_i = probability_normalized_occupancy.copy()

    # f_rate_over_all_session = global_all_spikes_counts['rate_Hz'].to_numpy()
    # f_rate_over_all_session
    check_f = np.nansum((p_i *  epoch_averaged_activity_per_pos_bin), axis=-1) # a check for f (rate over all session)
    f_rate_over_all_session = check_f # temporarily use check_f instead of the real f_rate

    fi_over_mean_f = epoch_averaged_activity_per_pos_bin / f_rate_over_all_session.reshape(-1, 1) # the `.reshape(-1, 1)` fixes the broadcasting

    log_base_2_of_fi_over_mean_f = np.log2(fi_over_mean_f) ## Here is where some entries become -np.inf

    _summand = (p_i * fi_over_mean_f * log_base_2_of_fi_over_mean_f) # _summand.shape # (77, 56)

    SI = np.nansum(_summand, axis=1)
    return SI


@function_attributes(short_name=None, tags=['spatial-information'], input_requires=[], output_provides=[], uses=['_perform_calc_SI'], used_by=[], creation_date='2024-05-28 15:24', related_items=[])
def compute_spatial_information(all_spikes_df: pd.DataFrame, an_active_pf: PfND, global_session_duration:float):
    """ Calculates the spatial information (SI) for each cell and returns all intermediates.

    Usage: 
        global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df).drop(columns=['neuron_type'], inplace=False)
        an_active_pf = deepcopy(global_pf1D)
        SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts = compute_spatial_information(all_spikes_df=global_spikes_df, an_active_pf=an_active_pf, global_session_duration=global_session.duration)


    """
    from neuropy.core.flattened_spiketrains import SpikesAccessor
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns

    #  Inputs: global_spikes_df: pd.DataFrame, an_active_pf: PfND, 
    # Build the aclu indicies:
    # neuron_IDs = global_spikes_df.aclu.unique()
    # n_aclus = global_spikes_df.aclu.nunique()
    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    all_spikes_df = deepcopy(all_spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # global_spikes_df


    # Get <f> for each sell, the rate over the entire session.
    global_all_spikes_counts = all_spikes_df.groupby(['aclu']).agg(t_count=('t', 'count')).reset_index()
    global_all_spikes_counts['rate_Hz'] = global_all_spikes_counts['t_count'] / global_session_duration
    # global_all_spikes_counts

    assert len(global_all_spikes_counts) == n_aclus
    
    ## Next need epoch-averaged activity per position bin:

    # Build the full matrix:
    global_per_position_bin_spikes_counts = all_spikes_df.groupby(['aclu', 'binned_x', 'binned_y']).agg(t_count=('t', 'count')).reset_index()
    a_spikes_df_bin_grouped = global_per_position_bin_spikes_counts.groupby(['aclu', 'binned_x']).agg(t_count_sum=('t_count', 'sum')).reset_index() ## for 1D plotting mode, collapse over all y-bins
    # a_spikes_df_bin_grouped

    assert n_aclus is not None
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)

    print(f'{n_aclus = }, {n_xbins = }')

    # a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell
    epoch_averaged_activity_per_pos_bin = np.zeros((n_aclus, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        # lap = int(row['lap'])
        aclu = int(row['aclu'])
        neuron_fragile_IDX: int = neuron_id_to_new_IDX_map[aclu]
        binned_x = int(row['binned_x'])
        count = row['t_count_sum']
        # a_spikes_bin_counts_mat[lap - 1][binned_x - 1] = count
        epoch_averaged_activity_per_pos_bin[neuron_fragile_IDX - 1][binned_x - 1] = count

    # an_active_pf.occupancy.shape # (n_xbins,) - (56,)
    # epoch_averaged_activity_per_pos_bin.shape # (n_aclus, n_xbins) - (77, 56)
    assert np.shape(an_active_pf.occupancy)[0] == np.shape(epoch_averaged_activity_per_pos_bin)[1]
        
    ## Compute actual Spatial Information for each cell:
    SI = _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy=an_active_pf.ratemap.probability_normalized_occupancy)

    return SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts


@function_attributes(short_name=None, tags=['UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-01 00:00', related_items=[])
def permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100):
    """ Not yet implemented. 2024-02-01
    
    Based off of the following quote:
    To determine the significance of the SI scores, we created a null distribution by circularly permuting the position data relative to the timeseries of each cell, by a random amount of at least 1 sec and a maximum amount of the length of the trial, independently on each trial. SI was calculated from the trial-averaged activity of each shuffle, and this shuffle procedure was repeated 100 times per cell. A cell’s true SI was considered significant if it exceeded 95% of the SI scores from all shuffles within animal (i.e. shuffled scores were pooled across cells within animal to produce this threshold, which is more stringent than comparing to the shuffle of each individual cell
    
    Usage:
        # True place field rate maps for all cells
        rate_maps = np.array('your rate maps')
        # True occupancy maps for all cells
        occupancy_maps = np.array('your occupancy maps')
        # Your position data
        position_data = np.array('your position data')

        # Call the permutation test function with the given number of permutations
        sig_cells = permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100)

        print(f'Indices of cells with significant SI: {sig_cells}')

    
    """
    # function to calculate Spatial Information (SI) score
    def calc_SI(rate_map, occupancy):
        # Place your existing SI calculation logic here
        pass

    # function to calculate rate map for given position data
    def calc_rate_map(position_data):
        # logic to calculate rate map
        pass

    # function to calculate occupancy map for given position data
    def calc_occupancy_map(position_data):
        # logic to calculate occupancy map
        pass

    n_cells = rate_maps.shape[0]  # number of cells
    si_scores = np.empty((n_cells, n_permutations))  # Initialize container for SI scores per cell per permutation
    true_si_scores = np.empty(n_cells)  # Initialize container for true SI scores per cell
   
    for cell_idx in range(n_cells):
        true_si_scores[cell_idx] = calc_SI(rate_maps[cell_idx], occupancy_maps[cell_idx])
        
        for perm_idx in range(n_permutations):
            shift_val = np.random.randint(1, len(position_data))  # A random shift amount
            shuffled_position_data = np.roll(position_data, shift_val)  # Shift the position data
        
            shuffled_rate_map = calc_rate_map(shuffled_position_data)
            shuffled_occupancy_map = calc_occupancy_map(shuffled_position_data)

            si_scores[cell_idx][perm_idx] = calc_SI(shuffled_rate_map, shuffled_occupancy_map)
   
    pooled_scores = si_scores.flatten() # Pool scores within animal
    threshold = np.percentile(pooled_scores, 95)  # Get the 95th percentile of the pooled scores

    return np.where(true_si_scores > threshold)  # Return indices where true SI scores exceed 95 percentile


@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=['compute_spatially_binned_activity'], creation_date='2024-01-31 00:00', related_items=[])
def compute_activity_by_lap_by_position_bin_matrix(a_spikes_df: pd.DataFrame, lap_id_to_matrix_IDX_map: Dict, n_xbins: int): # , an_active_pf: Optional[PfND] = None
    """ 2024-01-31 - Note that this does not take in position tracking information, so it cannot compute real occupancy. 
    
    Plots for a single neuron.
    
    an_active_pf: is just so we have access to the placefield's properties later
    
    
    Currently plots raw spikes counts (in number of spikes).
    
    """
    # Filter rows based on column: 'binned_x'
    a_spikes_df = a_spikes_df[a_spikes_df['binned_x'].astype("string").notna()]
    # a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y', 'lap']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    # a_spikes_df_bin_grouped

    ## for 1D plotting mode, collapse over all y-bins:
    a_spikes_df_bin_grouped = a_spikes_df_bin_grouped.groupby(['binned_x', 'lap']).agg(t_seconds_count_sum=('t_seconds_count', 'sum')).reset_index()
    # a_spikes_df_bin_grouped
    assert n_xbins is not None
    assert lap_id_to_matrix_IDX_map is not None
    n_laps: int = len(lap_id_to_matrix_IDX_map)
    
    a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        lap_id = int(row['lap'])
        lap_IDX = lap_id_to_matrix_IDX_map[lap_id]
        
        binned_x = int(row['binned_x'])
        count = row['t_seconds_count_sum']
        a_spikes_bin_counts_mat[lap_IDX][binned_x - 1] = count
        
    # active_out_matr = occupancy_probability
    
    # active_out_matr = a_spikes_bin_counts_mat
    # “calculated the occupancy (number of imaging samples) in each bin on each trial, and divided this by the total number of samples in each trial to get an occupancy probability per position bin per trial” 
    return a_spikes_bin_counts_mat


@function_attributes(short_name=None, tags=['spatial_information', 'binned', 'pos'], input_requires=[], output_provides=[], uses=['compute_activity_by_lap_by_position_bin_matrix'], used_by=[], creation_date='2024-01-31 00:00', related_items=[])
def compute_spatially_binned_activity(an_active_pf: PfND): # , global_any_laps_epochs_obj
    """ 
        from pyphoplacecellanalysis.Analysis.reliability import compute_spatially_binned_activity
        
        # a_spikes_df = None
        # a_spikes_df: pd.DataFrame = deepcopy(long_one_step_decoder_1D.spikes_df) #.drop(columns=['neuron_type'], inplace=False)

        # an_active_pf = deepcopy(global_pf2D)
        # an_active_pf = deepcopy(global_pf1D)
        # an_active_pf.linear_pos_obj

        # an_active_pf = active_pf_2D_dt
        an_active_pf = active_pf_1D_dt
        position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map) = compute_spatially_binned_activity(an_active_pf)
        # 14.8s
    """
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
    # from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # needed to add laps column

    ## need global laps positions now.

    # # Position:
    # position_df: pd.DataFrame = deepcopy(an_active_pf.filtered_pos_df) # .drop(columns=['neuron_type'], inplace=False)
    # position_df, (xbin,), bin_infos = build_df_discretized_binned_position_columns(position_df, bin_values=(an_active_pf.xbin,), position_column_names=('lin_pos',), binned_column_names=('binned_x',), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # if 'lap' not in position_df:
    #     position_df = add_epochs_id_identity(position_df, epochs_df=deepcopy(global_any_laps_epochs_obj.to_dataframe()), epoch_id_key_name='lap', epoch_label_column_name='lap_id', no_interval_fill_value=-1, override_time_variable_name='t')
    #     # drop the -1 indicies because they are below the speed:
    #     position_df = position_df[position_df['lap'] != -1] # Drop all non-included spikes
    # position_df

    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    # all_spikes_df: pd.DataFrame = deepcopy(all_spikes_df) # require passed-in value
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df)
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.filtered_spikes_df)
    all_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df) # Use placefields all spikes 
    all_spikes_df = all_spikes_df.spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df = all_spikes_df[all_spikes_df['lap'] > -1] # get only the spikes within a lap
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)

    split_spikes_dfs_list = all_spikes_df.spikes.get_split_by_unit()
    split_spikes_df_dict = dict(zip(neuron_IDs, split_spikes_dfs_list))
    
    laps_unique_ids = all_spikes_df.lap.unique()
    n_laps: int = len(laps_unique_ids)
    lap_id_to_matrix_IDX_map = dict(zip(laps_unique_ids, np.arange(n_laps)))

    # n_laps: int = position_df.lap.nunique()
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)
    
    # idx: int = 9
    # aclu: int = neuron_IDs[idx]
    # print(f'aclu: {aclu}')
    
    position_binned_activity_matr_dict = {}

    # for a_spikes_df in split_spikes_dfs:
    for aclu, a_spikes_df in split_spikes_df_dict.items():
        # split_spikes_df_dict[aclu], (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(a_spikes_df.drop(columns=['neuron_type'], inplace=False), bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
        a_position_binned_activity_matr = compute_activity_by_lap_by_position_bin_matrix(a_spikes_df=a_spikes_df, lap_id_to_matrix_IDX_map=lap_id_to_matrix_IDX_map, n_xbins=n_xbins)
        position_binned_activity_matr_dict[aclu] = a_position_binned_activity_matr
        
    # output: split_spikes_df_dict
    return position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map)


from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import polars as pl
from neuropy.utils.mixins.binning_helpers import compute_spanning_bins
from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo # for epochs_spkcount getting the correct time bins
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.core.flattened_spiketrains import SpikesAccessor


class CellIndividualReliabilityMatrix:
    """
        from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

        pfs = curr_active_pipeline.computation_results[maze_name].computed_data['pf2D']
        ratemaps = pfs.ratemap
        neuron_ids = deepcopy(pfs.ratemap.neuron_ids)
        n_neuron_ids: int = len(neuron_ids)
        spikes_df = deepcopy(pfs.filtered_spikes_df).spikes.sliced_by_neuron_id(neuron_ids)

        _fake_reliability_df, in_field_masks = CellIndividualReliabilityMatrix._partial_compute_reliability_matrix(
            spikes_df=spikes_df,
            active_peak_prominence_2d_results=active_peak_prominence_2d_results,
            ratemaps=ratemaps,
            n_top_peaks=3,
            # slice_level_multiplier=0.9,
            slice_level_multiplier=0.2,
            fn_tn_mode='occupancy_seconds',  # or 'occupied_bins'
        )

        t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse = CellIndividualReliabilityMatrix.compute_reliability_matrix(
            spikes_df=spikes_df,
            ratemaps=ratemaps,
            pfs=pfs,
            in_field_masks=in_field_masks,
            neuron_ids=neuron_ids,
            time_bin_size_seconds=0.050,
            max_t_idx = 1000,
        )
        t_bin_aclus_reliability_df

        ## OUTPUTS: _fake_reliability_df, in_field_masks, t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse


    """
    @function_attributes(short_name=None, tags=['MAIN', 'STAGE_2'], input_requires=[], output_provides=[], uses=['perform_compute_confusion_matrix'], used_by=[], creation_date='2026-07-22 19:47', related_items=[])
    @classmethod
    def compute_reliability_matrix(cls, spikes_df: pd.DataFrame, ratemaps, pfs, in_field_masks: Dict[int, np.ndarray], neuron_ids=None, time_bin_size_seconds: float = 0.050, **kwargs):
        """Compute per-aclu TP/FP/TN/FN reliability counts from time-binned spikes vs in-field masks.

        Parameters
        ----------
        spikes_df : filtered spikes with at least ['aclu','x','y'] (and a spikes time column).
        ratemaps : 2D Ratemap (provides xbin/ybin; neuron_ids used if `neuron_ids` is None).
        pfs : PfND / pf2D object (provides `filtered_pos_df` for interpolating animal position per t-bin).
        in_field_masks : Dict[aclu, np.ndarray[bool]] shaped like ratemap occupancy (nx, ny), 0-based.
        neuron_ids : optional explicit neuron id order; defaults to `ratemaps.neuron_ids`.
        time_bin_size_seconds : temporal bin width used for t_bin_idx / position alignment.

        Returns
        -------
        t_bin_aclus_reliability_df : DataFrame indexed by aclu with true_pos/true_neg/false_pos/false_neg.
        per_tbin_aclu_spike_counts_df : long DataFrame with columns ['aclu', 't_bin_idx', 'n_spikes'] (nonzero bins only; spike t_bin_idx is 1-based).
        time_bin_info_df : per-time-bin animal position with 0-based t_bin_idx.
        per_tbin_aclu_spike_counts_sparse : csr_matrix shape (n_aclus, n_t_bins), dtype int32.
            Rows follow `neuron_ids` order; columns are 0-based time bins aligned with `time_bin_info_df['t_bin_idx']`.
            Zero entries mean no spikes in that (aclu, t_bin).
        """
        # ==================================================================================================================================================================================================================================================================================== #
        # Main Compute Block                                                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #

        if neuron_ids is None:
            neuron_ids = np.asarray(ratemaps.neuron_ids)
        else:
            neuron_ids = np.asarray(neuron_ids)

        # ratemaps = curr_active_pipeline.computation_results[maze_name].computed_data['pf2D'].ratemap
        # spikes_df = deepcopy(curr_active_pipeline.computation_results[maze_name].computed_data['pf2D'].filtered_spikes_df)
        if 't_bin_idx' in spikes_df.columns:
            spikes_df = spikes_df.drop(columns=['t_bin_idx'], inplace=False)

        ## INPUTS: spikes_df, ratemaps
        # spikes_df should already have 'x' and 'y' (e.g. active_pf_2D.filtered_spikes_df)

        # spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(spikes_df, bin_values=(ratemaps.xbin, ratemaps.ybin), position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=False)
        spikes_df = spikes_df.spikes.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)

        if spikes_df.spikes.time_variable_name not in spikes_df.columns:
            if 't_seconds' in spikes_df.columns:
                spikes_df.spikes.set_time_variable_name('t_seconds')

        time_bin_edges, time_bin_edges_binning_info = compute_spanning_bins(spikes_df.spikes.times, bin_size=time_bin_size_seconds)
        bin_container = BinningContainer.init_from_edges(edges=time_bin_edges, edge_info=time_bin_edges_binning_info)
        n_t_bins: int = len(bin_container.centers) # 1427041

        spikes_df = spikes_df.spikes.add_binned_time_column(time_bin_edges, time_bin_edges_binning_info)
        spikes_df.rename(columns={'binned_time': 't_bin_idx'}, inplace=True)
        spikes_df['t_bin_idx'] = spikes_df['t_bin_idx'].astype(int)

        ## Positions:
        active_pos_df: pd.DataFrame = deepcopy(pfs.filtered_pos_df)
        # active_pos_df
        active_pos_df = active_pos_df.position.add_binned_time_column(time_bin_edges, time_bin_edges_binning_info)
        active_pos_df.rename(columns={'binned_time': 't_bin_idx'}, inplace=True)
        active_pos_df['t_bin_idx'] = spikes_df['t_bin_idx'].astype(int)
        active_pos_df = active_pos_df.dropna(subset=['binned_x', 'binned_y', 't_bin_idx']) # Drop rows with missing data in columns: 'binned_x', 'binned_y', 't_bin_idx'
        active_pos_df

        # pos_df = pfs.filtered_pos_df  # or sess.position.to_dataframe()

        time_bin_info_df: pd.DataFrame = pd.DataFrame({'t': bin_container.centers, 't_bin_idx': np.arange(bin_container.num_bins),
            'x': np.interp(bin_container.centers, pfs.filtered_pos_df['t'], pfs.filtered_pos_df['x']),
            'y': np.interp(bin_container.centers, pfs.filtered_pos_df['t'], pfs.filtered_pos_df['y']),
        })

        # time_bin_info_df.position.add
        time_bin_info_df = time_bin_info_df.position.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)
        time_bin_info_df

        ## OUTPUTS: spikes_df, active_pos_df, time_bin_info_df
        # spikes_df, active_pos_df, time_bin_info_df


        # ==================================================================================================================================================================================================================================================================================== #
        # Build in_field LUT (aclu, binned_x, binned_y) for Polars joins                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        # in_field_masks: Dict[aclu, np.ndarray[bool] shape (nx, ny)]  # 0-based array indexing
        rows = []
        for aclu, mask in in_field_masks.items():
            ix, iy = np.nonzero(mask)  # 0-based
            for bx, by in zip(ix + 1, iy + 1):  # match spikes_df binned_x/y labels
                rows.append({"aclu": int(aclu), "binned_x": int(bx), "binned_y": int(by), "is_in_field": True})
            ## END for bx, by in zip(ix + 1, iy + 1)...
        ## END for aclu, mask in in_field_masks.items()...

        in_field_lut = pl.DataFrame(rows).with_columns([
            pl.col("aclu").cast(pl.Int64),
            pl.col("binned_x").cast(pl.Int64),
            pl.col("binned_y").cast(pl.Int64),
        ])  # only True cells; absent = out-of-field / unknown spatial bin

        # ==================================================================================================================================================================================================================================================================================== #
        # Polars: per-(aclu, t_bin) spike counts                                                                                                                                                                                                                                               #
        # ==================================================================================================================================================================================================================================================================================== #
        spikes_pl = pl.from_pandas(spikes_df[["t_bin_idx", "aclu", "binned_x", "binned_y"]]).with_columns([
            pl.col("binned_x").cast(pl.Int64),
            pl.col("binned_y").cast(pl.Int64),
            pl.col("aclu").cast(pl.Int64),
            pl.col("t_bin_idx").cast(pl.Int64),
        ])

        per_tbin_aclu_spike_counts_df = (
            spikes_pl
            .group_by(["aclu", "t_bin_idx"])
            .agg([pl.len().alias("n_spikes")])
        ).to_pandas()

        # Sparse (n_aclus, n_t_bins) spike counts from COO nonzero entries (no dense allocate).
        # Spike t_bin_idx labels are 1-based; matrix columns / time_bin_info_df use 0-based indices.
        n_aclus: int = len(neuron_ids)
        aclu_arr = per_tbin_aclu_spike_counts_df['aclu'].to_numpy()
        t_bin_arr = per_tbin_aclu_spike_counts_df['t_bin_idx'].to_numpy().astype(np.int64)
        n_spikes_arr = per_tbin_aclu_spike_counts_df['n_spikes'].to_numpy().astype(np.int32)
        row_i = pd.Categorical(aclu_arr, categories=list(neuron_ids)).codes.astype(np.int64)
        col_j = t_bin_arr - 1
        valid = (row_i >= 0) & (col_j >= 0) & (col_j < n_t_bins)
        per_tbin_aclu_spike_counts_sparse = csr_matrix((n_spikes_arr[valid], (row_i[valid], col_j[valid])), shape=(n_aclus, n_t_bins), dtype=np.int32)

        # ==================================================================================================================================================================================================================================================================================== #
        # Compute Reliability Matrix                                                                                                                                                                                                                                                           #
        # ==================================================================================================================================================================================================================================================================================== #
        t_bin_aclus_reliability_df = cls.perform_compute_confusion_matrix(per_tbin=per_tbin_aclu_spike_counts_df, time_bin_info_df=time_bin_info_df, neuron_ids=neuron_ids, in_field_lut=in_field_lut, **kwargs)

        ## OUTPUTS: t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse
        return t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse


    @function_attributes(short_name=None, tags=['confusion_matrix', 'reliability'], input_requires=[], output_provides=[], uses=[], used_by=['compute_reliability_matrix'], creation_date='2026-07-22 19:39', related_items=[])
    @classmethod
    def perform_compute_confusion_matrix(cls, per_tbin: pd.DataFrame, time_bin_info_df: pd.DataFrame, neuron_ids,
                                         in_field_lut: pl.DataFrame, max_t_idx: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Accumulate per-aclu TP/FP/TN/FN from animal position per t-bin vs which cells' fields cover that bin.

        Polars join/agg implementation (no per-t-bin Python loop).

        Parameters
        ----------
        per_tbin : DataFrame with columns ['aclu', 't_bin_idx', 'n_spikes'].
        time_bin_info_df : per-time-bin animal position with ['t_bin_idx', 'binned_x', 'binned_y'] (1-based labels).
        neuron_ids : ordered neuron ids (row order of output).
        in_field_lut : Polars DataFrame with columns ['aclu', 'binned_x', 'binned_y'] (in-field spatial bins only).
        max_t_idx : if set, only process rows with t_bin_idx < max_t_idx (debug/partial runs).

        Returns
        -------
        t_bin_aclus_reliability_df : indexed by aclu with true_pos/true_neg/false_pos/false_neg.
            TP/FP are spike counts normalized by each cell's total spikes (TP+FP).
            TN/FN are silent time-bin counts normalized by each cell's opportunity counts:
            true_neg = TN / n_out_of_field_tbins, false_neg = FN / n_in_field_tbins.

        Notes
        -----
        Spatial bins absent from ``in_field_lut`` are "unknown": they contribute to ``n_computed_bins``
        but not to per-cell opportunities / TP/FP/TN/FN (matches prior empty-dict behavior).
        For known bins: ``n_out = n_known_tbins - n_in``, ``FN = n_in - infield_spike_tbins``,
        ``TN = n_out - outfield_spike_tbins``.
        """
        neuron_ids = np.asarray(neuron_ids)
        neuron_ids_i64 = neuron_ids.astype(np.int64)

        pos_cols = ['t_bin_idx', 'binned_x', 'binned_y']
        assert all(c in time_bin_info_df.columns for c in pos_cols), f"time_bin_info_df missing {pos_cols}"

        pos = (
            pl.from_pandas(time_bin_info_df[pos_cols])
            .with_columns([
                pl.col('t_bin_idx').cast(pl.Int64),
                pl.col('binned_x').cast(pl.Int64),
                pl.col('binned_y').cast(pl.Int64),
            ])
            .filter(pl.col('binned_x').is_not_null() & pl.col('binned_y').is_not_null())
        )
        if max_t_idx is not None:
            pos = pos.filter(pl.col('t_bin_idx') < int(max_t_idx))
        n_computed_bins: int = pos.height
        print(f"n_tbins={len(time_bin_info_df)}, n_valid={n_computed_bins}, n_nan={len(time_bin_info_df) - n_computed_bins}")

        lut = (
            in_field_lut
            .select(['aclu', 'binned_x', 'binned_y'])
            .with_columns([
                pl.col('aclu').cast(pl.Int64),
                pl.col('binned_x').cast(pl.Int64),
                pl.col('binned_y').cast(pl.Int64),
            ])
            .unique()
        )
        ## restrict LUT to requested neuron_ids
        lut = lut.filter(pl.col('aclu').is_in(neuron_ids_i64.tolist()))

        known_keys = lut.select(['binned_x', 'binned_y']).unique()
        known_pos = pos.join(known_keys, on=['binned_x', 'binned_y'], how='inner')
        n_known_tbins: int = known_pos.height

        ## n_in_field per aclu = # known visits whose animal bin is in that cell's field
        n_in_df = (
            known_pos
            .join(lut, on=['binned_x', 'binned_y'], how='inner')
            .group_by('aclu')
            .agg(pl.len().alias('n_in_field_tbins'))
        )

        base = pl.DataFrame({
            'aclu': neuron_ids_i64,
            'neuron_IDX': np.arange(len(neuron_ids), dtype=np.int64),
        }).with_columns([
            pl.lit(n_known_tbins).alias('n_known_tbins'),
            pl.lit(n_computed_bins).alias('n_computed_bins'),
        ])
        base = (
            base
            .join(n_in_df, on='aclu', how='left')
            .with_columns(pl.col('n_in_field_tbins').fill_null(0))
            .with_columns((pl.col('n_known_tbins') - pl.col('n_in_field_tbins')).alias('n_out_of_field_tbins'))
        )

        ## spikes only at known animal-position bins
        spikes = (
            pl.from_pandas(per_tbin[['aclu', 't_bin_idx', 'n_spikes']])
            .with_columns([
                pl.col('aclu').cast(pl.Int64),
                pl.col('t_bin_idx').cast(pl.Int64),
                pl.col('n_spikes').cast(pl.Float64),
            ])
        )
        sp = (
            spikes
            .join(known_pos.select(['t_bin_idx', 'binned_x', 'binned_y']), on='t_bin_idx', how='inner')
            .join(lut.with_columns(pl.lit(True).alias('is_in_field')), on=['aclu', 'binned_x', 'binned_y'], how='left')
            .with_columns(pl.col('is_in_field').fill_null(False))
        )

        spike_aggs = sp.group_by('aclu').agg([
            pl.col('n_spikes').filter(pl.col('is_in_field')).sum().fill_null(0).alias('true_pos_n_spikes'),
            pl.col('n_spikes').filter(~pl.col('is_in_field')).sum().fill_null(0).alias('false_pos_n_spikes'),
            pl.col('t_bin_idx').filter(pl.col('is_in_field')).n_unique().fill_null(0).alias('n_infield_spike_tbins'),
            pl.col('t_bin_idx').filter(~pl.col('is_in_field')).n_unique().fill_null(0).alias('n_outfield_spike_tbins'),
        ])

        out_pl = (
            base
            .join(spike_aggs, on='aclu', how='left')
            .with_columns([
                pl.col('true_pos_n_spikes').fill_null(0),
                pl.col('false_pos_n_spikes').fill_null(0),
                pl.col('n_infield_spike_tbins').fill_null(0),
                pl.col('n_outfield_spike_tbins').fill_null(0),
            ])
            .with_columns([
                (pl.col('n_in_field_tbins') - pl.col('n_infield_spike_tbins')).alias('false_neg_n_tbins'),
                (pl.col('n_out_of_field_tbins') - pl.col('n_outfield_spike_tbins')).alias('true_neg_n_tbins'),
            ])
            .with_columns([
                (pl.col('true_pos_n_spikes') + pl.col('false_pos_n_spikes')).alias('n_total_spikes'),
            ])
            .with_columns([
                pl.when(pl.col('n_total_spikes') > 0).then(pl.col('true_pos_n_spikes') / pl.col('n_total_spikes')).otherwise(None).alias('true_pos'),
                pl.when(pl.col('n_total_spikes') > 0).then(pl.col('false_pos_n_spikes') / pl.col('n_total_spikes')).otherwise(None).alias('false_pos'),
                pl.when(pl.col('n_out_of_field_tbins') > 0).then(pl.col('true_neg_n_tbins') / pl.col('n_out_of_field_tbins')).otherwise(None).alias('true_neg'),
                pl.when(pl.col('n_in_field_tbins') > 0).then(pl.col('false_neg_n_tbins') / pl.col('n_in_field_tbins')).otherwise(None).alias('false_neg'),
            ])
            .sort('neuron_IDX')
        )

        t_bin_aclus_reliability_df: pd.DataFrame = out_pl.to_pandas().set_index('aclu', drop=True, inplace=False)
        ## drop helper cols not in prior schema
        t_bin_aclus_reliability_df = t_bin_aclus_reliability_df.drop(columns=['n_known_tbins', 'n_infield_spike_tbins', 'n_outfield_spike_tbins'], errors='ignore')

        ## OUTPUTS: t_bin_aclus_reliability_df
        return t_bin_aclus_reliability_df


    @function_attributes(short_name=None, tags=['promence', 'PeakPromenence', 'mask'], input_requires=[], output_provides=[], uses=[], used_by=['_partial_compute_reliability_matrix'], creation_date='2026-07-22 19:26', related_items=[])
    @classmethod
    def _build_top_peak_90pct_masks(cls, active_peak_prominence_2d_results, n_top_peaks: int = 3, slice_level_multiplier: float = 0.9) -> Dict[int, np.ndarray]:
        """Build per-neuron boolean masks (ny, nx) = union of top-N peak contours at `slice_level_multiplier` * peak height.

        Uses precomputed `level_slices` when present; otherwise recomputes the contour from the stored `slab`.
        """
        from matplotlib.path import Path
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence

        xx = np.asarray(active_peak_prominence_2d_results.xx)
        yy = np.asarray(active_peak_prominence_2d_results.yy)
        XX, YY = np.meshgrid(xx, yy, indexing='xy')  # (ny, nx) — matches prominence `slab` (.T of tuning curve)
        points = np.column_stack([XX.ravel(), YY.ravel()])

        def _contour_to_path(contour):
            if contour is None:
                return None
            if isinstance(contour, tuple):
                contour = contour[0]
            if isinstance(contour, np.ndarray):
                return Path(contour)
            return contour  # already a Path

        def _lookup_precomputed_slice(a_peak, lvl: float):
            level_slices = a_peak.get('level_slices', {}) or {}
            # exact / float-key match
            slice_info = level_slices.get(lvl)
            if slice_info is not None:
                return slice_info
            for k, v in level_slices.items():
                if np.isclose(float(k), lvl, rtol=1e-5, atol=1e-8):
                    return v
            ## END for k, v in level_slices.items()...
            # closest precomputed probe level (by multiplier), if any
            probe_levels = np.asarray(a_peak.get('probe_levels', []), dtype=float)
            if len(probe_levels) == 0 or float(a_peak.get('height', 0.0)) <= 0:
                return None
            mults = probe_levels / float(a_peak['height'])
            lvl_idx = int(np.argmin(np.abs(mults - slice_level_multiplier)))
            if np.isclose(mults[lvl_idx], slice_level_multiplier, atol=1e-3):
                nearest_lvl = float(probe_levels[lvl_idx])
                for k, v in level_slices.items():
                    if np.isclose(float(k), nearest_lvl, rtol=1e-5, atol=1e-8):
                        return v
                    ## END for k, v in level_slices.items()...
            return None

        def _recompute_contour(a_peak, slab, lvl: float):
            if slab is None:
                return None
            peak_center = np.asarray(a_peak['center'], dtype=float)
            included = PeakPromenence._find_contours_at_levels(xx, yy, slab, peak_center, np.asarray([lvl], dtype=float))
            # keys are the probe level floats used; match with isclose
            for k, contour in included.items():
                if np.isclose(float(k), lvl, rtol=1e-5, atol=1e-8):
                    return contour
                ## END for k, contour in included.items()...
            # single-level call → at most one entry
            if len(included) == 1:
                return next(iter(included.values()))
            return None

        masks_by_neuron: Dict[int, np.ndarray] = {}
        for neuron_id, a_result in active_peak_prominence_2d_results.results.items():
            peaks = a_result['peaks']
            slab = a_result.get('slab', None)
            top_peaks = sorted(peaks.items(), key=lambda kv: kv[1]['prominence'], reverse=True)[:n_top_peaks]
            union_mask = np.zeros(XX.shape, dtype=bool)

            for _peak_id, a_peak in top_peaks:
                lvl = float(a_peak['height'] * slice_level_multiplier)
                slice_info = _lookup_precomputed_slice(a_peak, lvl)
                contour = None
                if slice_info is not None:
                    contour = _contour_to_path(slice_info.get('contour'))
                if contour is None:
                    contour = _contour_to_path(_recompute_contour(a_peak, slab, lvl))
                if contour is None:
                    continue
                union_mask |= contour.contains_points(points).reshape(XX.shape)
            ## END for _peak_id, a_peak in top_peaks...

            masks_by_neuron[int(neuron_id)] = union_mask
        ## END for neuron_id, a_result in active_peak_prominence_2d_results.results.items()...

        return masks_by_neuron


    @function_attributes(short_name=None, tags=['INCOMPLETE'], input_requires=[], output_provides=[], uses=['_build_top_peak_90pct_masks'], used_by=[], creation_date='2026-07-22 19:22', related_items=[])
    @classmethod
    def _partial_compute_reliability_matrix(cls, spikes_df: pd.DataFrame, active_peak_prominence_2d_results, ratemaps, n_top_peaks: int = 3, slice_level_multiplier: float = 0.9, fn_tn_mode: str = 'occupancy_seconds') -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
        """Per-cell placefield reliability confusion counts using 90% prominence field masks.

        Definitions (condition = in-field; detection = spike):
        TP: # spikes inside field
        FP: # spikes outside field (visited maze only)
        FN: in-field occupancy with no spikes (bins or occupancy-seconds)
        TN: out-of-field occupancy with no spikes (bins or occupancy-seconds)

        Parameters
        ----------
        spikes_df : must include ['aclu','x','y'] (or ['binned_x','binned_y']).
        active_peak_prominence_2d_results : PeakProminence2D DynamicParameters.
        ratemaps : neuropy Ratemap (2D) with occupancy, spikes_maps, xbin/ybin, neuron_ids.
        fn_tn_mode : 'occupancy_seconds' | 'occupied_bins'

        Returns
        -------
        reliability_df : one row per neuron with TP/FP/FN/TN (+ rates)
        in_field_masks_xy : neuron_id -> bool mask shaped like ratemap occupancy (nx, ny)
        """
        assert fn_tn_mode in ('occupancy_seconds', 'occupied_bins')
        occupancy = np.asarray(ratemaps.occupancy)  # (nx, ny)
        visited = occupancy > 0
        neuron_ids = np.asarray(ratemaps.neuron_ids)
        xbin, ybin = np.asarray(ratemaps.xbin), np.asarray(ratemaps.ybin)
        nx, ny = occupancy.shape

        # Contour masks are (ny, nx); ratemap maps are (nx, ny)
        masks_ny_nx = cls._build_top_peak_90pct_masks(active_peak_prominence_2d_results, n_top_peaks=n_top_peaks, slice_level_multiplier=slice_level_multiplier)
        in_field_masks_xy: Dict[int, np.ndarray] = {nid: m.T for nid, m in masks_ny_nx.items() if m.shape == (ny, nx)}

        # Prefer unsmoothed spike maps when available; else histogram from spikes_df
        spikes_maps = getattr(ratemaps, 'spikes_maps', None)
        if spikes_maps is not None:
            spikes_maps = np.asarray(spikes_maps)

        rows = []
        for neuron_idx, neuron_id in enumerate(neuron_ids):
            neuron_id = int(neuron_id)
            in_field = in_field_masks_xy.get(neuron_id)
            if in_field is None:
                in_field = np.zeros((nx, ny), dtype=bool)
                in_field_masks_xy[neuron_id] = in_field
            in_field = in_field & visited
            out_field = (~in_field) & visited

            # --- TP / FP from spikes ---
            if (spikes_maps is not None) and (spikes_maps.ndim >= 3):
                spikes_map = np.asarray(spikes_maps[neuron_idx])
                assert spikes_map.shape == (nx, ny), f"spikes_map shape {spikes_map.shape} != occupancy {(nx, ny)}"
                TP = float(np.nansum(spikes_map[in_field]))
                FP = float(np.nansum(spikes_map[out_field]))

            else:
                ## have to compute spikes_maps mangually from spikes_df:
                cell_spikes = spikes_df[spikes_df['aclu'] == neuron_id]
                if {'binned_x', 'binned_y'}.issubset(cell_spikes.columns):
                    bx = cell_spikes['binned_x'].to_numpy().astype(int) - 1  # labels are often 1-indexed
                    by = cell_spikes['binned_y'].to_numpy().astype(int) - 1
                    valid = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
                    bx, by = bx[valid], by[valid]
                else:
                    assert {'x', 'y'}.issubset(cell_spikes.columns), "spikes_df needs ['x','y'] or ['binned_x','binned_y']"
                    spikes_map, _, _ = np.histogram2d(cell_spikes['x'].to_numpy(), cell_spikes['y'].to_numpy(), bins=(xbin, ybin))
                    TP = float(np.nansum(spikes_map[in_field])) ## total spikes in the field (over all positions)
                    FP = float(np.nansum(spikes_map[out_field])) ## total spikes outside the field (over all positions)
                    bx = by = None

                if bx is not None:
                    is_in = in_field[bx, by]
                    TP = float(np.sum(is_in))
                    FP = float(np.sum(~is_in))
                    spikes_map = np.histogram2d(cell_spikes['x'].to_numpy(), cell_spikes['y'].to_numpy(), bins=(xbin, ybin))[0] if {'x','y'}.issubset(cell_spikes.columns) else None

            if spikes_maps is not None and spikes_maps.ndim >= 3:
                spikes_map = np.asarray(spikes_maps[neuron_idx])
            elif 'spikes_map' not in locals() or spikes_map is None:
                cell_spikes = spikes_df[spikes_df['aclu'] == neuron_id]
                spikes_map = np.histogram2d(cell_spikes['x'].to_numpy(), cell_spikes['y'].to_numpy(), bins=(xbin, ybin))[0]

            no_spike = (spikes_map == 0) & visited
            fn_mask = in_field & no_spike
            tn_mask = out_field & no_spike

            if fn_tn_mode == 'occupancy_seconds':
                FN = float(np.nansum(occupancy[fn_mask]))
                TN = float(np.nansum(occupancy[tn_mask]))
            else:
                FN = float(np.count_nonzero(fn_mask))
                TN = float(np.count_nonzero(tn_mask))

            denom_pos = TP + FN
            denom_neg = TN + FP
            denom_all = TP + FP + FN + TN
            rows.append({
                'neuron_id': neuron_id,
                'n_field_bins': int(np.count_nonzero(in_field)),
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                'precision': TP / (TP + FP) if (TP + FP) > 0 else np.nan,
                'recall_sensitivity': TP / denom_pos if denom_pos > 0 else np.nan,  # in-field spike capture
                'specificity': TN / denom_neg if denom_neg > 0 else np.nan,
                'false_positive_rate': FP / denom_neg if denom_neg > 0 else np.nan,
                'accuracy': (TP + TN) / denom_all if denom_all > 0 else np.nan,
                'in_field_spike_fraction': TP / (TP + FP) if (TP + FP) > 0 else np.nan,
            })
        ## END for neuron_idx, neuron_id in enumerate(neuron_ids)...

        return pd.DataFrame(rows).set_index('neuron_id'), in_field_masks_xy



    @function_attributes(short_name=None, tags=['matplotlib', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-22 19:20', related_items=[])
    @classmethod
    def plot_in_field_masks_with_spikes(cls, pfs, in_field_masks: Dict[int, np.ndarray],
                                        included_neuron_ids: Optional[Sequence[int]] = None,
                                        color_by_in_field: bool = True, max_n_cells: Optional[int] = None,
                                        subplots: Optional[Tuple[int, int]] = None, figsize_per_cell: float = 2.5,
                                        mask_cmap: str = "Greens", mask_alpha: float = 0.55,
                                        heatmap_cmap: str = "jet", heatmap_alpha: float = 0.7,
                                        spike_s: float = 2.0, spike_alpha: float = 0.3,
                                        use_pcolormesh: bool = True, show_trajectory: bool = False,
                                        trajectory_alpha: float = 0.15) -> Tuple[Figure, np.ndarray]:
        """Plot per-cell placefield heatmap (background) + in-field mask + spike positions.

        Layer order (bottom → top): trajectory (optional) → tuning-curve heatmap → in-field mask → spikes.

        Usage:

            from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

            ## Usage:
            fig, axes = CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes(pfs, in_field_masks)

        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        assert getattr(pfs, "ndim", 2) >= 2, "plot_in_field_masks_with_spikes requires 2D PfND"
        xbin = np.asarray(pfs.xbin)
        ybin = np.asarray(pfs.ybin)
        spikes_df = pfs.filtered_spikes_df
        ratemap = pfs.ratemap
        neuron_ids_rm = np.asarray(ratemap.neuron_ids)
        tuning_curves = np.asarray(ratemap.tuning_curves)  # (n_neurons, nx, ny)
        nx, ny = len(xbin) - 1, len(ybin) - 1
        extent = (xbin[0], xbin[-1], ybin[0], ybin[-1])

        neuron_ids = list(included_neuron_ids) if included_neuron_ids is not None else sorted(in_field_masks.keys())
        if max_n_cells is not None:
            neuron_ids = neuron_ids[:int(max_n_cells)]
        ## END if max_n_cells is not None...

        n = len(neuron_ids)
        assert n > 0, "No neuron_ids to plot"

        if subplots is None:
            n_cols = int(np.ceil(np.sqrt(n)))
            n_rows = int(np.ceil(n / n_cols))
        else:
            n_rows, n_cols = subplots
        ## END if subplots is None...

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows), squeeze=False)
        flat_axes: List[Axes] = list(axes.ravel())

        for ax_i, aclu in enumerate(neuron_ids):
            ax = flat_axes[ax_i]
            aclu = int(aclu)
            mask = np.asarray(in_field_masks.get(aclu, np.zeros((nx, ny), dtype=bool)), dtype=bool)
            if mask.shape != (nx, ny):
                if mask.shape == (ny, nx):
                    mask = mask.T
                else:
                    raise ValueError(f"aclu {aclu}: mask shape {mask.shape} != expected {(nx, ny)}")
                ## END if mask.shape == (ny, nx)...
            ## END if mask.shape != (nx, ny)...

            if show_trajectory and hasattr(pfs, "x") and hasattr(pfs, "y"):
                ax.plot(pfs.x, pfs.y, color="#d3c5c5", alpha=trajectory_alpha, linewidth=0.5, zorder=0)
            ## END if show_trajectory...

            # --- background heatmap (tuning curve) ---
            rm_idx = np.flatnonzero(neuron_ids_rm == aclu)
            if len(rm_idx) > 0:
                pfmap = np.asarray(tuning_curves[int(rm_idx[0])], dtype=float)
                if use_pcolormesh:
                    ax.pcolormesh(xbin, ybin, pfmap.T, cmap=heatmap_cmap, alpha=heatmap_alpha, shading="flat", zorder=1)
                else:
                    plot_pf = np.fliplr(np.rot90(pfmap, k=-1))
                    ax.imshow(plot_pf, origin="lower", extent=extent, cmap=heatmap_cmap, alpha=heatmap_alpha, zorder=1, aspect="auto")
                ## END if use_pcolormesh...
            ## END if len(rm_idx) > 0...

            # --- in-field mask ---
            if use_pcolormesh:
                ax.pcolormesh(xbin, ybin, mask.T.astype(float), cmap=mask_cmap, alpha=mask_alpha, shading="flat", vmin=0, vmax=1, zorder=2)
            else:
                plot_mask = np.fliplr(np.rot90(mask.astype(float), k=-1))
                ax.imshow(plot_mask, origin="lower", extent=extent, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1, zorder=2, aspect="auto")
            ## END if use_pcolormesh...

            # --- spikes ---
            cell_spk = spikes_df[spikes_df["aclu"] == aclu]
            if len(cell_spk) > 0:
                if color_by_in_field and {"binned_x", "binned_y"}.issubset(cell_spk.columns):
                    bx = cell_spk["binned_x"].to_numpy().astype(int) - 1
                    by = cell_spk["binned_y"].to_numpy().astype(int) - 1
                    valid = (bx >= 0) & (by >= 0) & (bx < mask.shape[0]) & (by < mask.shape[1])
                    in_field = np.zeros(len(cell_spk), dtype=bool)
                    in_field[valid] = mask[bx[valid], by[valid]]
                    ax.scatter(cell_spk.loc[~in_field, "x"], cell_spk.loc[~in_field, "y"], s=spike_s, c="0.45", alpha=spike_alpha * 0.7, marker=".", linewidths=0, zorder=3)
                    ax.scatter(cell_spk.loc[in_field, "x"], cell_spk.loc[in_field, "y"], s=spike_s, c="red", alpha=spike_alpha, marker=".", linewidths=0, zorder=4)
                else:
                    ax.scatter(cell_spk["x"], cell_spk["y"], s=spike_s, c="red", alpha=spike_alpha, marker=".", linewidths=0, zorder=3)
                ## END if color_by_in_field...
            ## END if len(cell_spk) > 0...

            ax.set_aspect("equal")
            ax.set_xlim(xbin[0], xbin[-1])
            ax.set_ylim(ybin[0], ybin[-1])
            ax.set_title(f"aclu {aclu}", fontsize=9)
            ax.axis("off")
        ## END for ax_i, aclu in enumerate(neuron_ids)...

        for ax in flat_axes[n:]:
            ax.axis("off")
        ## END for ax in flat_axes[n:]...

        fig.suptitle("PF heatmap + in-field masks + spikes", fontsize=12)
        fig.tight_layout()
        return fig, axes

