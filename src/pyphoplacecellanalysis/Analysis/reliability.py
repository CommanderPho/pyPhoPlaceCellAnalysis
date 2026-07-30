 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
from copy import deepcopy
from enum import Enum, auto
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

from typing import Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import polars as pl
from neuropy.utils.mixins.binning_helpers import compute_spanning_bins
from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo # for epochs_spkcount getting the correct time bins
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.core.flattened_spiketrains import SpikesAccessor

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

        from pyphoplacecellanalysis.Analysis.reliability import compute_spatial_information, _perform_calc_SI


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



# ==================================================================================================================================================================================================================================================================================== #
# Cell Individual Reliability and Confusion                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #

class ReliabilityDecoderModifierMode(Enum):
    """How the reliability information is integrated into the decoder (when reliability is available on the decoder) """
    IGNORE = auto()
    LIKELIHOOD_TEMPERING = auto() ## "Power-prior mode: Raise each cell’s likelihood to the power of the likelihood and re‑normalise."
    # MIXTURE_MODEL = auto()

    def __str__(self):
        return self.name

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]


class ReliabilityEstimationMode(Enum):
    """How per-cell reliability arrays are estimated from confusion-matrix products."""
    PER_CELL = auto()              # (n_neurons,) from confusion rates
    POSITION_DEPENDENT = auto()    # (n_flat_position_bins, n_neurons) from rates × in_field

    def __str__(self):
        return self.name

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]


@metadata_attributes(short_name=None, tags=['reliability'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-23 05:13', related_items=[])
class CellIndividualReliabilityMatrix:
    """
        from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

        pfs = curr_active_pipeline.computation_results[maze_name].computed_data['pf2D']
        ratemaps = pfs.ratemap
        neuron_ids = deepcopy(pfs.ratemap.neuron_ids)
        n_neuron_ids: int = len(neuron_ids)
        spikes_df = deepcopy(pfs.filtered_spikes_df).spikes.sliced_by_neuron_id(neuron_ids)

        # STAGE_1: in-field masks from PeakProminence2D (or from pf via build_in_field_masks_xy_from_pf)
        in_field_masks = CellIndividualReliabilityMatrix.build_in_field_masks_xy(
            active_peak_prominence_2d_results=active_peak_prominence_2d_results,
            ratemaps=ratemaps,
            n_top_peaks=3,
            # slice_level_multiplier=0.9,
            slice_level_multiplier=0.2,
            neuron_ids=neuron_ids,
        )

        # STAGE_2: time-binned TP/FP/TN/FN confusion products
        t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df, per_tbin_aclu_per_lap_xy_spike_counts_df = CellIndividualReliabilityMatrix.compute_reliability_matrix(
            spikes_df=spikes_df,
            ratemaps=ratemaps,
            pfs=pfs,
            in_field_masks=in_field_masks,
            neuron_ids=neuron_ids,
            time_bin_size_seconds=0.050,
            max_t_idx = 1000,
        )
        t_bin_aclus_reliability_df

        ## OUTPUTS: in_field_masks, t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df, per_tbin_aclu_per_lap_xy_spike_counts_df

        # Or use the decoder mixin entrypoint (STAGE_1 + STAGE_2 + reliability_* metrics):
        #   a_dst_decoder2D.compute_unit_confusion_reliability_variables()


    """
    @function_attributes(short_name=None, tags=['MAIN', 'STAGE_2'], input_requires=[], output_provides=[], uses=['perform_compute_confusion_matrix'], used_by=[], creation_date='2026-07-22 19:47', related_items=[])
    @classmethod
    def compute_reliability_matrix(cls, spikes_df: pd.DataFrame, pfs: PfND, in_field_masks: Dict[int, np.ndarray], ratemaps=None, neuron_ids=None, time_bin_size_seconds: float = 0.050,
                                        reliability_estimation_mode: ReliabilityEstimationMode = ReliabilityEstimationMode.POSITION_DEPENDENT,
                                    **kwargs):
        """Compute per-aclu TP/FP/TN/FN reliability counts from time-binned spikes vs in-field masks.

        Parameters
        ----------
        spikes_df : filtered spikes with at least ['aclu','x','y'] (and a spikes time column).
        ratemaps : 2D Ratemap (provides xbin/ybin; neuron_ids used if `neuron_ids` is None).
        pfs : PfND / pf2D object (provides `filtered_pos_df` for interpolating animal position per t-bin).
        in_field_masks : Dict[aclu, np.ndarray[bool]] shaped like ratemap occupancy (nx, ny), 0-based.
        neuron_ids : optional explicit neuron id order; defaults to `ratemaps.neuron_ids`.
        time_bin_size_seconds : temporal bin width used for t_bin_idx / position alignment.
        reliability_estimation_mode : reserved for callers; xy spike counts are always computed (needed for POSITION_DEPENDENT maps).

        Returns
        -------
        t_bin_aclus_reliability_df : DataFrame indexed by aclu with true_pos/true_neg/false_pos/false_neg.
        per_tbin_aclu_spike_counts_df : long DataFrame with columns ['aclu', 't_bin_idx', 'n_spikes'] (nonzero bins only; spike t_bin_idx is 1-based).
        time_bin_info_df : per-time-bin animal position with 0-based t_bin_idx.
        per_tbin_aclu_spike_counts_sparse : csr_matrix shape (n_aclus, n_t_bins), dtype int32.
            Rows follow `neuron_ids` order; columns are 0-based time bins aligned with `time_bin_info_df['t_bin_idx']`.
            Zero entries mean no spikes in that (aclu, t_bin).
        per_tbin_aclu_xy_spike_counts_df : long DataFrame with columns ['aclu', 't_bin_idx', 'binned_x', 'binned_y', 'n_spikes']
            (nonzero spike-location bins only; spike ``t_bin_idx`` is 1-based; ``binned_x``/``binned_y`` are spike positions).
        per_tbin_aclu_per_lap_xy_spike_counts_df : long DataFrame with columns
            ['aclu', 't_bin_idx', 'lap', 'binned_x', 'binned_y', 'n_spikes'] when ``spikes_df`` has a ``lap`` column;
            otherwise ``None``. Spike ``t_bin_idx`` is 1-based.
        """
        # ==================================================================================================================================================================================================================================================================================== #
        # Main Compute Block                                                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        if ratemaps is None:
            ratemaps = pfs.ratemap


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

        # # spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(spikes_df, bin_values=(ratemaps.xbin, ratemaps.ybin), position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=False)
        # spikes_df = spikes_df.spikes.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)

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
        active_pos_df = active_pos_df.position.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)
        active_pos_df = active_pos_df.position.add_binned_time_column(time_bin_edges, time_bin_edges_binning_info)
        active_pos_df.rename(columns={'binned_time': 't_bin_idx'}, inplace=True)
        # active_pos_df['t_bin_idx'] = active_pos_df['t_bin_idx'].astype(int) # #TODO 2026-07-28 19:19: - [ ] 't_bin_idx' was actually the problem, it was being set to the wrong dataframe's column and then forced to int (which np.nan can't go to int)
        active_pos_df.dropna(subset=['binned_x', 'binned_y', 't_bin_idx'], inplace=True) # Drop rows with missing data in columns: 'binned_x', 'binned_y', 't_bin_idx'
        active_pos_df['t_bin_idx'] = active_pos_df['t_bin_idx'].astype(int) ## convert to int
        


        ## Interpolate the spikes positions from the position df:
        # pos_df = pfs.filtered_pos_df  # or sess.position.to_dataframe()
        spikes_df = deepcopy(spikes_df).spikes.interpolate_spike_positions(
            active_pos_df['t'].to_numpy(), active_pos_df['x'].to_numpy(), active_pos_df['y'].to_numpy(),
            replace_existing=True,
        )
        # spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(spikes_df, bin_values=(ratemaps.xbin, ratemaps.ybin), position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=False)
        spikes_df = spikes_df.spikes.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)


        time_bin_info_df: pd.DataFrame = pd.DataFrame({'t': bin_container.centers, 't_bin_idx': np.arange(bin_container.num_bins),
            'x': np.interp(bin_container.centers, active_pos_df['t'], active_pos_df['x']),
            # 'y': np.interp(bin_container.centers, active_pos_df['t'], active_pos_df['y']),
        })

        if 'y' in active_pos_df.columns:
            time_bin_info_df['y'] = np.interp(bin_container.centers, active_pos_df['t'], active_pos_df['y'])

        if 'z' in active_pos_df.columns:
            time_bin_info_df['z'] = np.interp(bin_container.centers, active_pos_df['t'], active_pos_df['z'])


        # time_bin_info_df.position.add
        time_bin_info_df = time_bin_info_df.position.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)
        time_bin_info_df

        ## OUTPUTS: spikes_df, active_pos_df, time_bin_info_df
        # spikes_df, active_pos_df, time_bin_info_df


        # ==================================================================================================================================================================================================================================================================================== #
        # Build in_field LUT (aclu, binned_x, binned_y) for Polars joins                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        # in_field_masks: Dict[aclu, np.ndarray[bool] shape (nx, ny)]  # 0-based array indexing
        in_field_lut = cls.build_in_field_lut(in_field_masks)  # only True cells; absent = out-of-field / unknown spatial bin


        # # right after adding_binned_position_columns, before Polars:
        # print(spikes_df[['y','binned_y']].dtypes)
        # print(spikes_df['binned_y'].isna().mean())
        # print(spikes_df['y'].describe())
        # print(ratemaps.ybin[[0,-1]])

        # ==================================================================================================================================================================================================================================================================================== #
        # Polars: per-(aclu, t_bin, binned_x, binned_y) spike counts (spike locations)                                                                                                                                                                                                           #
        # ==================================================================================================================================================================================================================================================================================== #
        spikes_pl = pl.from_pandas(spikes_df[["t_bin_idx", "aclu", "binned_x", "binned_y"]]).with_columns([
            pl.col("binned_x").cast(pl.Int64),
            pl.col("binned_y").cast(pl.Int64),
            pl.col("aclu").cast(pl.Int64),
            pl.col("t_bin_idx").cast(pl.Int64),
        ])

        per_tbin_aclu_xy_spike_counts_df = (
            spikes_pl
            .group_by(["aclu", "t_bin_idx", "binned_x", "binned_y"])
            .agg([pl.len().alias("n_spikes")])
        ).to_pandas()

        # Optional lap-partitioned counts (same spike locations, split by lap); None when no lap column
        if "lap" in spikes_df.columns:
            spikes_lap_pl = pl.from_pandas(spikes_df[["t_bin_idx", "aclu", "lap", "binned_x", "binned_y"]]).with_columns([
                pl.col("binned_x").cast(pl.Int64),
                pl.col("binned_y").cast(pl.Int64),
                pl.col("aclu").cast(pl.Int64),
                pl.col("t_bin_idx").cast(pl.Int64),
                pl.col("lap").cast(pl.Int64),
            ])
            per_tbin_aclu_per_lap_xy_spike_counts_df = (
                spikes_lap_pl
                .group_by(["aclu", "t_bin_idx", "lap", "binned_x", "binned_y"])
                .agg([pl.len().alias("n_spikes")])
            ).to_pandas()
        else:
            per_tbin_aclu_per_lap_xy_spike_counts_df = None
        ## END if "lap" in spikes_df.columns...

        # Coarse per-(aclu, t_bin) counts for PER_CELL confusion + sparse matrix (sum over spike locations)
        per_tbin_aclu_spike_counts_df = (
            per_tbin_aclu_xy_spike_counts_df
            .groupby(["aclu", "t_bin_idx"], as_index=False)["n_spikes"]
            .sum()
        )

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

        ## OUTPUTS: t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df, per_tbin_aclu_per_lap_xy_spike_counts_df
        return t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df, per_tbin_aclu_per_lap_xy_spike_counts_df


    @classmethod
    def _prepare_visit_polars_frames(cls, per_tbin: pd.DataFrame, time_bin_info_df: pd.DataFrame, neuron_ids, in_field_lut: pl.DataFrame, max_t_idx: Optional[int] = None, spike_t_bin_offset: int = 1) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray]:
        """Cast/filter animal position, in-field LUT, and per-tbin spikes for visit-conditioned aggregations.

        Spike ``t_bin_idx`` labels from ``compute_reliability_matrix`` are 1-based; animal
        ``time_bin_info_df['t_bin_idx']`` is 0-based. Default ``spike_t_bin_offset=1`` subtracts
        that offset so joins align on 0-based time bins.

        Returns
        -------
        pos, lut, spikes, neuron_ids_i64
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

        lut = (
            in_field_lut
            .select(['aclu', 'binned_x', 'binned_y'])
            .with_columns([
                pl.col('aclu').cast(pl.Int64),
                pl.col('binned_x').cast(pl.Int64),
                pl.col('binned_y').cast(pl.Int64),
            ])
            .unique()
            .filter(pl.col('aclu').is_in(neuron_ids_i64.tolist()))
        )

        # Align spike t_bin_idx to 0-based animal time_bin_info_df indices
        spikes = (
            pl.from_pandas(per_tbin[['aclu', 't_bin_idx', 'n_spikes']])
            .with_columns([
                pl.col('aclu').cast(pl.Int64),
                (pl.col('t_bin_idx').cast(pl.Int64) - int(spike_t_bin_offset)).alias('t_bin_idx'),
                pl.col('n_spikes').cast(pl.Float64),
            ])
        )
        return pos, lut, spikes, neuron_ids_i64


    @function_attributes(short_name=None, tags=['confusion_matrix', 'reliability'], input_requires=[], output_provides=[], uses=['_prepare_visit_polars_frames'], used_by=['compute_reliability_matrix'], creation_date='2026-07-22 19:39', related_items=[])
    @classmethod
    def perform_compute_confusion_matrix(cls, per_tbin: pd.DataFrame, time_bin_info_df: pd.DataFrame, neuron_ids,
                                         in_field_lut: pl.DataFrame, max_t_idx: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Accumulate per-aclu TP/FP/TN/FN from animal position per t-bin vs which cells' fields cover that bin.

        Polars join/agg implementation (no per-t-bin Python loop).

        Parameters
        ----------
        per_tbin : DataFrame with columns ['aclu', 't_bin_idx', 'n_spikes'] (spike ``t_bin_idx`` 1-based).
        time_bin_info_df : per-time-bin animal position with 0-based ``t_bin_idx`` and 1-based ``binned_x``/``binned_y``.
        neuron_ids : ordered neuron ids (row order of output).
        in_field_lut : Polars DataFrame with columns ['aclu', 'binned_x', 'binned_y'] (in-field spatial bins only).
        max_t_idx : if set, only process rows with t_bin_idx < max_t_idx (debug/partial runs).

        Returns
        -------
        t_bin_aclus_reliability_df : indexed by aclu with true_pos/true_neg/false_pos/false_neg.
            Position cols: ``n_infield_tbins``, ``n_outfield_tbins``, ``n_total_tbins``.
            Spike cols: ``n_infield_spike_tbins``, ``n_outfield_spike_tbins``, ``n_infield_nonspike_tbins``, ``n_outfield_nonspike_tbins``.
            TP/FP are spike counts normalized by each cell's total spikes (TP+FP).
            TN/FN are silent time-bin counts normalized by each cell's opportunity counts:
            true_neg = n_outfield_nonspike_tbins / n_outfield_tbins, false_neg = n_infield_nonspike_tbins / n_infield_tbins.

        Notes
        -----
        Spatial bins absent from ``in_field_lut`` are "unknown": they contribute to ``n_computed_bins``
        but not to per-cell opportunities / TP/FP/TN/FN (matches prior empty-dict behavior).
        For known bins: ``n_outfield_tbins = n_total_tbins - n_infield_tbins``,
        ``n_infield_nonspike_tbins = n_infield_tbins - n_infield_spike_tbins``,
        ``n_outfield_nonspike_tbins = n_outfield_tbins - n_outfield_spike_tbins``.
        """
        neuron_ids = np.asarray(neuron_ids)
        pos, lut, spikes, neuron_ids_i64 = cls._prepare_visit_polars_frames(per_tbin=per_tbin, time_bin_info_df=time_bin_info_df, neuron_ids=neuron_ids, in_field_lut=in_field_lut, max_t_idx=max_t_idx)
        ## spikes: ['aclu', 't_bin_idx', 'n_spikes']
        ## pos: ['t_bin_idx', 'binned_x', 'binned_y']

        n_computed_bins: int = pos.height
        print(f"n_tbins={len(time_bin_info_df)}, n_valid={n_computed_bins}, n_nan={len(time_bin_info_df) - n_computed_bins}")

        known_pos = pos.join(lut.select(['binned_x', 'binned_y']).unique(), on=['binned_x', 'binned_y'], how='inner') # ['t_bin_idx', 'binned_x', 'binned_y']
        n_total_tbins: int = known_pos.height

        ## n_infield per aclu = # known visits whose animal bin is in that cell's field
        n_in = known_pos.join(lut, on=['binned_x', 'binned_y'], how='inner').group_by('aclu').agg(pl.len().alias('n_infield_tbins'))

        ## spikes only at known animal-position bins
        sp = (
            spikes
            .join(known_pos.select(['t_bin_idx', 'binned_x', 'binned_y']), on='t_bin_idx', how='inner')
            .join(lut.with_columns(pl.lit(True).alias('is_in_field')), on=['aclu', 'binned_x', 'binned_y'], how='left')
            .with_columns(pl.col('is_in_field').fill_null(False))
        )
        spike_aggs = sp.group_by('aclu').agg([
            pl.col('n_spikes').filter(pl.col('is_in_field')).sum().fill_null(0).alias('true_pos_n_spikes'),
            pl.col('n_spikes').filter(~pl.col('is_in_field')).sum().fill_null(0).alias('false_pos_n_spikes'),
            pl.col('t_bin_idx').filter(pl.col('is_in_field')).len().fill_null(0).alias('n_infield_spike_tbins'),
            pl.col('t_bin_idx').filter(~pl.col('is_in_field')).len().fill_null(0).alias('n_outfield_spike_tbins'),
        ])

        out_pl = (
            pl.DataFrame({'aclu': neuron_ids_i64, 'neuron_IDX': np.arange(len(neuron_ids), dtype=np.int64)})
            .with_columns([
                pl.lit(n_total_tbins).alias('n_total_tbins'),
                pl.lit(n_computed_bins).alias('n_computed_bins'),
            ])
            .join(n_in, on='aclu', how='left')
            .join(spike_aggs, on='aclu', how='left')
            .with_columns([
                pl.col('n_infield_tbins').fill_null(0),
                pl.col('true_pos_n_spikes').fill_null(0),
                pl.col('false_pos_n_spikes').fill_null(0),
                pl.col('n_infield_spike_tbins').fill_null(0),
                pl.col('n_outfield_spike_tbins').fill_null(0),
            ])
            .with_columns((pl.col('n_total_tbins') - pl.col('n_infield_tbins')).alias('n_outfield_tbins'))
            .with_columns([
                (pl.col('n_infield_tbins') - pl.col('n_infield_spike_tbins')).alias('n_infield_nonspike_tbins'),
                (pl.col('n_outfield_tbins') - pl.col('n_outfield_spike_tbins')).alias('n_outfield_nonspike_tbins'),
                (pl.col('true_pos_n_spikes') + pl.col('false_pos_n_spikes')).alias('n_total_spikes'),
            ])
            .with_columns([
                pl.when(pl.col('n_total_spikes') > 0).then(pl.col('true_pos_n_spikes') / pl.col('n_total_spikes')).otherwise(None).alias('true_pos'),
                pl.when(pl.col('n_total_spikes') > 0).then(pl.col('false_pos_n_spikes') / pl.col('n_total_spikes')).otherwise(None).alias('false_pos'),
                pl.when(pl.col('n_outfield_tbins') > 0).then(pl.col('n_outfield_nonspike_tbins') / pl.col('n_outfield_tbins')).otherwise(None).alias('true_neg'),
                pl.when(pl.col('n_infield_tbins') > 0).then(pl.col('n_infield_nonspike_tbins') / pl.col('n_infield_tbins')).otherwise(None).alias('false_neg'),
            ])
            .sort('neuron_IDX')
        )

        ## OUTPUTS: t_bin_aclus_reliability_df — position_cols: (n_infield_tbins, n_outfield_tbins, n_total_tbins), spike_cols: (n_infield_spike_tbins, n_outfield_spike_tbins, n_infield_nonspike_tbins, n_outfield_nonspike_tbins)
        t_bin_aclus_reliability_df: pd.DataFrame = out_pl.to_pandas().set_index('aclu', drop=True, inplace=False)
        return t_bin_aclus_reliability_df


    @classmethod
    def build_in_field_lut(cls, in_field_masks: Dict[int, np.ndarray]) -> pl.DataFrame:
        """Convert Dict[aclu, (nx, ny) bool] masks to Polars LUT with 1-based ``binned_x``/``binned_y`` labels."""
        rows = []
        for aclu, mask in in_field_masks.items():
            ix, iy = np.nonzero(mask)  # 0-based
            for bx, by in zip(ix + 1, iy + 1):
                rows.append({"aclu": int(aclu), "binned_x": int(bx), "binned_y": int(by), "is_in_field": True})
            ## END for bx, by in zip(ix + 1, iy + 1)...
        ## END for aclu, mask in in_field_masks.items()...
        if len(rows) == 0:
            return pl.DataFrame(schema={"aclu": pl.Int64, "binned_x": pl.Int64, "binned_y": pl.Int64, "is_in_field": pl.Boolean})
        return pl.DataFrame(rows).with_columns([
            pl.col("aclu").cast(pl.Int64),
            pl.col("binned_x").cast(pl.Int64),
            pl.col("binned_y").cast(pl.Int64),
        ])


    @function_attributes(short_name=None, tags=['confusion_matrix', 'reliability', 'position-dependent'], input_requires=[], output_provides=[], uses=['_prepare_visit_polars_frames'], used_by=['BayesianPlacemapPositionDecoder._compute_reliability_metrics'], creation_date='2026-07-28 17:00', related_items=['perform_compute_confusion_matrix'])
    @classmethod
    def perform_compute_position_dependent_reliability_maps(cls, per_tbin: pd.DataFrame, time_bin_info_df: pd.DataFrame, neuron_ids, in_field_lut: pl.DataFrame, occupancy_shape: Tuple[int, ...], max_t_idx: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Build visit-conditioned reliability maps ``(n_flat_position_bins, n_neurons)`` from animal position per t-bin.

        For each (aclu, animal binned_x, binned_y):
            p_fire = n_active_visits / n_visits
            in-field:  R_active = p_fire, R_silent = p_fire   (= 1 - local FN)
            out-field: R_active = 1 - p_fire, R_silent = 1 - p_fire  (= local TN)

        Unvisited bins remain 0. ``binned_x``/``binned_y`` are 1-based labels matching ``time_bin_info_df``.
        Uses unique visited spatial bins × aclus (not full n_tbins × n_neurons cross).

        Parameters
        ----------
        per_tbin : DataFrame with columns ['aclu', 't_bin_idx', 'n_spikes'] (spike ``t_bin_idx`` 1-based).
        time_bin_info_df : animal position per t-bin with 0-based ``t_bin_idx`` and 1-based ``binned_x``/``binned_y``.
        neuron_ids : ordered neuron ids (column order of output maps).
        in_field_lut : Polars DataFrame ['aclu', 'binned_x', 'binned_y'] (in-field bins only).
        occupancy_shape : ``(nx, ny)`` used to flatten maps in C-order (same as ``F`` / occupancy).

        Returns
        -------
        R_active, R_silent : ndarray shape ``(n_flat, n_neurons)``
        position_aclus_reliability_df : long DataFrame with per-(aclu, binned_x, binned_y) rates
        """
        neuron_ids = np.asarray(neuron_ids)
        nx, ny = int(occupancy_shape[0]), int(occupancy_shape[1])
        n_flat: int = nx * ny
        n_neurons: int = len(neuron_ids)

        pos, lut, spikes, neuron_ids_i64 = cls._prepare_visit_polars_frames(per_tbin=per_tbin, time_bin_info_df=time_bin_info_df, neuron_ids=neuron_ids, in_field_lut=in_field_lut, max_t_idx=max_t_idx)

        # Visit counts per animal spatial bin (shared across aclus)
        visit_counts = pos.group_by(['binned_x', 'binned_y']).agg(pl.len().alias('n_visits'))

        # Active visits: cell fired in a t-bin while animal was at (bx, by)
        active = (
            spikes
            .filter(pl.col('n_spikes') > 0)
            .join(pos, on='t_bin_idx', how='inner')
            .group_by(['aclu', 'binned_x', 'binned_y'])
            .agg(pl.len().alias('n_active_visits'))
        )

        aclus = pl.DataFrame({'aclu': neuron_ids_i64})
        bin_aggs = (
            visit_counts
            .join(aclus, how='cross')
            .join(active, on=['aclu', 'binned_x', 'binned_y'], how='left')
            .with_columns(pl.col('n_active_visits').fill_null(0).cast(pl.Float64))
            .join(lut.with_columns(pl.lit(True).alias('is_in_field')), on=['aclu', 'binned_x', 'binned_y'], how='left')
            .with_columns(pl.col('is_in_field').fill_null(False))
            .with_columns((pl.col('n_active_visits') / pl.col('n_visits')).alias('p_fire'))
            .with_columns([
                (1.0 - pl.col('p_fire')).alias('p_silence'),
                pl.when(pl.col('is_in_field')).then(pl.col('p_fire')).otherwise(1.0 - pl.col('p_fire')).alias('R_active'),
                pl.when(pl.col('is_in_field')).then(pl.col('p_fire')).otherwise(1.0 - pl.col('p_fire')).alias('R_silent'),
            ])
        )

        position_aclus_reliability_df: pd.DataFrame = bin_aggs.to_pandas()

        R_active = np.zeros((n_flat, n_neurons), dtype=float)
        R_silent = np.zeros((n_flat, n_neurons), dtype=float)
        aclu_to_i = {int(a): i for i, a in enumerate(neuron_ids)}
        for row in position_aclus_reliability_df.itertuples(index=False):
            i = aclu_to_i.get(int(row.aclu))
            if i is None:
                continue
            ix = int(row.binned_x) - 1
            iy = int(row.binned_y) - 1
            if (ix < 0) or (iy < 0) or (ix >= nx) or (iy >= ny):
                continue
            flat_idx = ix * ny + iy
            R_active[flat_idx, i] = float(row.R_active)
            R_silent[flat_idx, i] = float(row.R_silent)
        ## END for row in position_aclus_reliability_df.itertuples(index=False)...

        return np.nan_to_num(R_active, nan=0.0), np.nan_to_num(R_silent, nan=0.0), position_aclus_reliability_df


    @function_attributes(short_name=None, tags=['promenece', 'PeakPromenence', 'mask'], input_requires=[], output_provides=[], uses=[], used_by=['build_in_field_masks_xy'], creation_date='2026-07-22 19:26', related_items=[])
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


    @function_attributes(short_name=None, tags=['prominence', 'PeakProminence2D', 'pf'], input_requires=[], output_provides=[], uses=['PeakPromenence.compute_prominence_contours'], used_by=['build_in_field_masks_xy_from_pf', 'BayesianPlacemapPositionDecoderDST.compute_unit_confusion_reliability_variables'], creation_date='2026-07-23 16:00', related_items=['_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'])
    @classmethod
    def compute_peak_prominence_2d_from_pf(cls, pf: PfND, step: float = 0.01, min_considered_promenence: float = 0.2, neuron_ids=None):
        """Build a minimal PeakProminence2D DynamicParameters from a 2D PfND (no pipeline cache required).

        Mirrors the core of ``ratemap_peaks_prominence2d`` (unit-max tuning curves → prominence contours),
        but only stores what ``_build_top_peak_90pct_masks`` / ``build_in_field_masks_xy`` need:
        ``xx``, ``yy``, and per-neuron ``results[nid] = {peaks, slab, id_map, prominence_map, parent_map}``.

        Parameters
        ----------
        pf : PfND
            2D placefield object (``pf.ndim >= 2``).
        step, min_considered_promenence : forwarded to ``PeakPromenence.compute_prominence_contours``.
        neuron_ids : optional subset of neuron ids to include; defaults to all ``pf.ratemap.neuron_ids``.

        Returns
        -------
        DynamicParameters with ``xx``, ``yy``, ``results`` (compatible with ``build_in_field_masks_xy``).
        """
        import matplotlib
        from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
        # from neuropy.utils.dynamic_container import DynamicParameters
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence

        assert getattr(pf, 'ndim', 2) >= 2, "compute_peak_prominence_2d_from_pf requires 2D PfND"
        matplotlib.use('Agg')  # avoid interactive figures from contour helpers

        ratemap = pf.ratemap
        all_neuron_ids = np.asarray(ratemap.neuron_ids)
        active_tuning_curves = ratemap.unit_max_tuning_curves  # (n_neurons, nx, ny)
        if neuron_ids is None:
            included_set = None
        else:
            included_set = {int(nid) for nid in np.asarray(neuron_ids)}

        out_results = {}
        for neuron_idx, neuron_id in enumerate(all_neuron_ids):
            neuron_id = int(neuron_id)
            if (included_set is not None) and (neuron_id not in included_set):
                continue
            slab = np.asarray(active_tuning_curves[neuron_idx]).T
            _, _, slab, cell_peaks_dict, id_map, prominence_map, parent_map = PeakPromenence.compute_prominence_contours(xbin_centers=pf.xbin_centers, ybin_centers=pf.ybin_centers, slab=slab, step=step, min_area=None, min_considered_promenence=min_considered_promenence, include_edge=True, verbose=False)
            out_results[neuron_id] = {'peaks': cell_peaks_dict, 'slab': slab, 'id_map': id_map, 'prominence_map': prominence_map, 'parent_map': parent_map}
        ## END for neuron_idx, neuron_id in enumerate(all_neuron_ids)...

        return DynamicParameters(xx=pf.xbin_centers, yy=pf.ybin_centers, results=out_results)


    @function_attributes(short_name=None, tags=['prominence', 'in_field', 'mask', 'pf'], input_requires=[], output_provides=[], uses=['compute_peak_prominence_2d_from_pf', 'build_in_field_masks_xy'], used_by=['BayesianPlacemapPositionDecoderDST.compute_unit_confusion_reliability_variables'], creation_date='2026-07-23 16:00', related_items=[])
    @classmethod
    def build_in_field_masks_xy_from_pf(cls, pf: PfND, n_top_peaks: int = 3, slice_level_multiplier: float = 0.9, neuron_ids=None, step: float = 0.01, min_considered_promenence: float = 0.2) -> Dict[int, np.ndarray]:
        """Build per-neuron in-field masks (nx, ny) from a 2D PfND by recomputing PeakProminence2D.

        Convenience wrapper: ``compute_peak_prominence_2d_from_pf`` → ``build_in_field_masks_xy``.
        """
        active_peak_prominence_2d_results = cls.compute_peak_prominence_2d_from_pf(pf, step=step, min_considered_promenence=min_considered_promenence, neuron_ids=neuron_ids)
        return cls.build_in_field_masks_xy(active_peak_prominence_2d_results=active_peak_prominence_2d_results, ratemaps=pf.ratemap, n_top_peaks=n_top_peaks, slice_level_multiplier=slice_level_multiplier, neuron_ids=neuron_ids)


    @function_attributes(short_name=None, tags=['prominence', 'in_field', 'mask'], input_requires=[], output_provides=[], uses=['_build_top_peak_90pct_masks'], used_by=['compute_reliability_matrix', 'build_in_field_masks_xy_from_pf', 'CellIndividualReliabilityComputingMixin.compute_unit_confusion_reliability_variables'], creation_date='2026-07-23 04:06', related_items=[])
    @classmethod
    def build_in_field_masks_xy(cls, active_peak_prominence_2d_results, ratemaps, n_top_peaks: int = 3, slice_level_multiplier: float = 0.9, neuron_ids=None) -> Dict[int, np.ndarray]:
        """Build per-neuron in-field boolean masks shaped like ratemap occupancy (nx, ny).

        Contour masks from PeakProminence are (ny, nx); this method transposes matching masks to (nx, ny).
        Neurons without a valid contour mask get an all-False mask of shape (nx, ny).

        Parameters
        ----------
        active_peak_prominence_2d_results : PeakProminence2D DynamicParameters.
        ratemaps : neuropy Ratemap (2D) with occupancy and neuron_ids.
        n_top_peaks, slice_level_multiplier : forwarded to `_build_top_peak_90pct_masks`.
        neuron_ids : optional explicit neuron id order; defaults to `ratemaps.neuron_ids`.

        Returns
        -------
        in_field_masks_xy : Dict[neuron_id, np.ndarray[bool]] shaped like occupancy (nx, ny).
        """
        occupancy = np.asarray(ratemaps.occupancy)  # (nx, ny)
        nx, ny = occupancy.shape
        if neuron_ids is None:
            neuron_ids = np.asarray(ratemaps.neuron_ids)
        else:
            neuron_ids = np.asarray(neuron_ids)

        # Contour masks are (ny, nx); ratemap maps are (nx, ny)
        masks_ny_nx = cls._build_top_peak_90pct_masks(active_peak_prominence_2d_results, n_top_peaks=n_top_peaks, slice_level_multiplier=slice_level_multiplier)
        in_field_masks_xy: Dict[int, np.ndarray] = {nid: m.T for nid, m in masks_ny_nx.items() if m.shape == (ny, nx)}

        for neuron_id in neuron_ids:
            neuron_id = int(neuron_id)
            if neuron_id not in in_field_masks_xy:
                in_field_masks_xy[neuron_id] = np.zeros((nx, ny), dtype=bool)
        ## END for neuron_id in neuron_ids...

        return in_field_masks_xy


    # ==================================================================================================================================================================================================================================================================================== #
    # Cell Reliability Metrics                                                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['private', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=['compute_skaggs_alpha', 'compute_sparsity_alpha', 'compute_dsnr_alpha'], creation_date='2026-07-23 05:28', related_items=[])
    @classmethod
    def _extract_pf_data(cls, pf) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Safely extracts, flattens, and normalizes occupancy and tuning curves from a PfND object.
        
        Args:
            pf: A Neuropy PfND object containing ratemap data.
            
        Returns:
            P_v: (V,) numpy array of normalized spatial occupancy probabilities for valid bins.
            lambda_i: (N, V) numpy array of tuning curves for N neurons over V valid spatial bins.
            N: Integer representing the number of neurons.
        """
        try:
            # Access the ratemap properties native to Neuropy's PfND object
            occupancy = pf.ratemap.occupancy
            tuning_curves = pf.ratemap.tuning_curves
        except AttributeError:
            raise ValueError(
                "Provided object does not have the expected PfND structure. "
                "Ensure it contains `pf.ratemap.occupancy` and `pf.ratemap.tuning_curves`."
            )

        # Flatten the spatial dimensions to generalize across 1D and 2D placefields
        N = tuning_curves.shape[0]
        lambda_i = tuning_curves.reshape(N, -1)
        occupancy = occupancy.reshape(-1)

        # Filter out unvisited bins (occupancy is 0 or NaN)
        valid_bins = (occupancy > 0) & ~np.isnan(occupancy)
        occ_valid = occupancy[valid_bins]
        lambda_i_valid = lambda_i[:, valid_bins]

        # Normalize occupancy to be a strict probability distribution P(v) where sum(P(v)) == 1
        P_v = occ_valid / np.nansum(occ_valid)

        return P_v, lambda_i_valid, N


    @function_attributes(short_name=None, tags=['metric', 'cell-reliability', 'dempster-shafer'], input_requires=[], output_provides=[], uses=['cls._extract_pf_data'], used_by=[], creation_date='2026-07-23 05:26', related_items=['compute_sparsity_alpha', 'compute_dsnr_alpha'])
    @classmethod
    def compute_skaggs_alpha(cls, pf, k: float = 1.0) -> NDArray[ND.Shape["N_NEURONS"], Any]:
        """
        Computes the reliability factor (alpha) based on Skaggs Spatial Information.
        
        Formula:
            I_i = sum_v [ P(v) * (lambda_i(v) / lambda_bar_i) * log2(lambda_i(v) / lambda_bar_i) ]
            alpha_i = 1 - e^(-k * I_i)
            
        Args:
            pf: The PfND placefield object.
            k: Exponential decay threshold mapping bits/spike to a [0, 1) range.
            
        Returns:
            alpha: NDArray[ND.Shape["N_NEURONS"], Any] — per-neuron reliability factors in [0, 1).
        """
        P_v, lambda_i, N = cls._extract_pf_data(pf)

        # Calculate overall mean firing rate for each cell (lambda_bar)
        # Shape: (N, 1) for broadcasting
        lambda_bar = np.nansum(lambda_i * P_v, axis=1, keepdims=True)

        # Initialize the ratio lambda_i(v) / lambda_bar_i
        ratio = np.zeros_like(lambda_i)
        
        # Mask to prevent division by zero or log2 of zero
        valid_mask = (lambda_bar > 0) & (lambda_i > 0)
        
        # Temporarily suppress numpy warnings for the valid indexing block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute the ratio only where valid
            lambda_bar_bc = np.broadcast_to(lambda_bar, lambda_i.shape)
            np.place(ratio, valid_mask, lambda_i[valid_mask] / lambda_bar_bc[valid_mask])

        # Calculate Skaggs Information (I_i) in bits/spike
        log2_ratio = np.zeros_like(ratio)
        np.log2(ratio, out=log2_ratio, where=(ratio > 0))
        I_i = np.nansum(P_v * ratio * log2_ratio, axis=1)

        # Map to Dempster-Shafer mass via exponential decay
        alpha: NDArray[ND.Shape["N_NEURONS"], Any] = 1.0 - np.exp(-k * I_i)
        
        return alpha


    @function_attributes(short_name=None, tags=['metric', 'cell-reliability', 'dempster-shafer'], input_requires=[], output_provides=[], uses=['cls._extract_pf_data'], used_by=[], creation_date='2026-07-23 05:26', related_items=['compute_skaggs_alpha', 'compute_dsnr_alpha'])
    @classmethod
    def compute_sparsity_alpha(cls, pf) -> NDArray[ND.Shape["N_NEURONS"], Any]:
        """
        Computes the reliability factor (alpha) based on Spatial Sparsity.
        A highly tuned cell has sparsity approaching 0. A uniform firer approaches 1.
        
        Formula:
            S_i = (sum_v [ P(v) * lambda_i(v) ])^2 / sum_v [ P(v) * lambda_i(v)^2 ]
            alpha_i = 1 - S_i
            
        Args:
            pf: The PfND placefield object.
            
        Returns:
            alpha: NDArray[ND.Shape["N_NEURONS"], Any] — per-neuron reliability factors in [0, 1).
        """
        P_v, lambda_i, N = cls._extract_pf_data(pf)

        # Numerator: Square of the expected firing rate across the environment
        num = np.nansum(P_v * lambda_i, axis=1)**2
        
        # Denominator: Expected squared firing rate
        den = np.nansum(P_v * (lambda_i**2), axis=1)

        # Initialize Sparsity array
        S_i = np.ones(N)  # Default to 1 (pure ignorance) if cell doesn't fire
        
        # Compute sparsity only for cells with a non-zero denominator
        valid_den = den > 0
        S_i[valid_den] = num[valid_den] / den[valid_den]

        # Map to Dempster-Shafer mass by inverting the sparsity
        alpha: NDArray[ND.Shape["N_NEURONS"], Any] = 1.0 - S_i
        
        return alpha


    @function_attributes(short_name=None, tags=['metric', 'cell-reliability', 'dempster-shafer'], input_requires=[], output_provides=[], uses=['cls._extract_pf_data'], used_by=[], creation_date='2026-07-23 05:26', related_items=['compute_skaggs_alpha', 'compute_sparsity_alpha'])
    @classmethod
    def compute_dsnr_alpha(cls, pf, n_i: Union[NDArray[ND.Shape["N_NEURONS"], Any], NDArray[ND.Shape["N_NEURONS, N_TIME_BINS"], Any], list], tau: float, lambda_bg: Optional[NDArray[ND.Shape["N_NEURONS"], Any]] = None) -> Union[NDArray[ND.Shape["N_NEURONS"], Any], NDArray[ND.Shape["N_NEURONS, N_TIME_BINS"], Any]]:
        """
        Computes the temporally dynamic reliability factor (alpha_i(t)) based on 
        instantaneous Signal-to-Noise ratio.
        
        Formula:
            alpha_i(t) = n_i(t) / (n_i(t) + (lambda_bg * tau))
            
        Args:
            pf: The PfND placefield object (used to estimate lambda_bg if not provided).
            n_i: Spike counts per neuron. NDArray[ND.Shape["N_NEURONS"], Any] for one time bin,
                or NDArray[ND.Shape["N_NEURONS, N_TIME_BINS"], Any] for multiple time bins.
            tau: Time window duration in seconds (e.g., 0.02 for 20ms bins).
            lambda_bg: Optional NDArray[ND.Shape["N_NEURONS"], Any] of baseline out-of-field
                    firing rates (Hz). If None, estimated as the 5th percentile of ratemap activity.
                    
        Returns:
            alpha: NDArray matching ``n_i`` shape — NDArray[ND.Shape["N_NEURONS"], Any] or
                NDArray[ND.Shape["N_NEURONS, N_TIME_BINS"], Any] — dynamic reliability in [0, 1).



        Usage:
            pf2D = curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D']
            an_active_pf = deepcopy(pf2D)
            ## INPUTS: an_active_pf, time_bin_size_seconds, _decoder_per_tbin_aclu_spike_counts_sparse
            alpha_skaggs = CellIndividualReliabilityMatrix.compute_skaggs_alpha(an_active_pf, k=1.0) # array([0.417225, 0.612937, 0.0186054, 0.839156, 0.253242, 0.390859, 0.551637, 0.410431, 0.232258, 0.319258, 0.0831956, 0.500425, 0.439415, 0.40174, 0.460294, 0.507179, 0.467489, 0.487803, 0.262977, 0.316431, 0.499277, 0.356243, 0.758122, 0.133721, 0.649214])
            alpha_sparsity = CellIndividualReliabilityMatrix.compute_sparsity_alpha(an_active_pf)
            alpha_dsnr = CellIndividualReliabilityMatrix.compute_dsnr_alpha(an_active_pf, n_i = _decoder_per_tbin_aclu_spike_counts_sparse.toarray(), tau=time_bin_size_seconds)

            alpha_skaggs
            alpha_sparsity
            alpha_dsnr


        """
        n_i = np.asarray(n_i, dtype=float)
        _, lambda_i, N = cls._extract_pf_data(pf)
        
        if lambda_bg is None:
            # Estimate the out-of-field baseline firing rate as the 5th percentile 
            # of the valid spatial bins. This is a standard proxy for "noise".
            lambda_bg = np.nanpercentile(lambda_i, 5, axis=1)
        else:
            lambda_bg = np.asarray(lambda_bg)
            if lambda_bg.shape[0] != N:
                raise ValueError(f"lambda_bg must have shape ({N},) to match neuron count.")

        # Align shapes for broadcasting if n_i is a 2D array (N, T)
        if n_i.ndim == 2:
            lambda_bg = lambda_bg.reshape(-1, 1)

        # Compute expected baseline spike count in this time window
        expected_bg_spikes = lambda_bg * tau
        
        # Add a tiny epsilon to prevent division by zero in intervals with zero spikes 
        # and zero estimated background rate.
        eps = 1e-12 
        
        alpha: Union[NDArray[ND.Shape["N_NEURONS"], Any], NDArray[ND.Shape["N_NEURONS, N_TIME_BINS"], Any]] = n_i / (n_i + expected_bg_spikes + eps)
        
        return alpha



    # ==================================================================================================================================================================================================================================================================================== #
    # Plotting Display                                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #


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
            fig, axes = CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes(a_dst_decoder2D.pfs, a_dst_decoder2D.in_field_masks)

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


    @function_attributes(short_name=None, tags=['matplotlib', 'figure', 'reliability'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-29 14:35', related_items=['plot_in_field_masks_with_spikes'])
    @classmethod
    def plot_reliability_maps_with_spikes(cls, pfs, reliability_active: np.ndarray, reliability_silent: np.ndarray, neuron_ids,
                                        reliability_estimation_mode: ReliabilityEstimationMode = ReliabilityEstimationMode.PER_CELL,
                                        in_field_masks: Optional[Dict[int, np.ndarray]] = None,
                                        position_aclus_reliability_df: Optional[pd.DataFrame] = None,
                                        per_tbin_aclu_per_lap_xy_spike_counts_df: Optional[pd.DataFrame] = None,
                                        included_neuron_ids: Optional[Sequence[int]] = None,
                                        reliability_variables: str = "active", show_confusion_counts: bool = True,
                                        should_display_lap_by_lap_spike_counts: bool = False,
                                        max_n_cells: Optional[int] = None,
                                        subplots: Optional[Tuple[int, int]] = None, figsize_per_cell: float = 2.5,
                                        mask_cmap: str = "Greens", mask_alpha: float = 0.35,
                                        heatmap_cmap: str = "viridis", heatmap_alpha: float = 0.9,
                                        count_cmap: str = "YlOrRd", count_alpha: float = 0.95,
                                        spike_s: float = 2.0, spike_alpha: float = 0.3, color_by_in_field: bool = True,
                                        use_pcolormesh: bool = True, show_trajectory: bool = False,
                                        trajectory_alpha: float = 0.15, show_count_bin_labels: bool = True,
                                        show_bin_grid: bool = True, count_label_fontsize: Optional[float] = None,
                                        bin_grid_linewidth: float = 0.25, bin_grid_color: str = "0.35",
                                        bin_grid_alpha: float = 0.65) -> Tuple[Figure, np.ndarray]:
        """Plot per-cell reliability maps (active | silent) + optional TP/FP/TN/FN count maps + spikes.

        Layout: for each cell, panels for selected reliability variable(s), and when
        ``position_aclus_reliability_df`` is provided and ``show_confusion_counts`` is True,
        four additional panels of raw visit-event counts: TP / FP / TN / FN.
        When ``should_display_lap_by_lap_spike_counts`` is True, an additional panel shows a
        vertical stack (one row per lap) of spike counts in each marginal_x position bin,
        built from ``per_tbin_aclu_per_lap_xy_spike_counts_df`` (summed over t-bins and y).

        Per-position confusion counts (from visit-conditioned ``position_aclus_reliability_df``):
            TP = n_active_visits when in-field
            FP = n_active_visits when out-of-field
            FN = (n_visits - n_active_visits) when in-field
            TN = (n_visits - n_active_visits) when out-of-field

        Count panels draw a thin bin grid and tiny integer labels on every non-zero bin.
        Each count panel uses its own vmax (not shared across TP/FP/TN/FN) so sparse maps remain visible.
        ``count_label_fontsize=None`` auto-sizes labels from median bin size vs axis display size (clamped).
        Lap×x panels use the same count colormap; y is lap index (not maze y), aspect auto.

        Mode handling:
            PER_CELL: scalar per neuron → constant ``(nx, ny)`` fill; title shows ``R_a`` / ``R_s``.
            POSITION_DEPENDENT: spatial maps ``(*spatial, n_neurons)`` or flat ``(n_flat, n_neurons)``.

        Layer order (bottom → top): trajectory (optional) → heatmap → in-field mask (optional) → spikes
        (count panels: heatmap → bin grid → count labels; no spikes/mask overlay).

        Usage:

            from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

            fig, axes = CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes(
                a_dst_decoder2D.pf, a_dst_decoder2D.reliability_active, a_dst_decoder2D.reliability_silent,
                neuron_ids=a_dst_decoder2D.neuron_IDs,
                reliability_estimation_mode=a_dst_decoder2D.reliability_estimation_mode,
                in_field_masks=a_dst_decoder2D.in_field_masks,
                position_aclus_reliability_df=a_dst_decoder2D.position_aclus_reliability_df,
                per_tbin_aclu_per_lap_xy_spike_counts_df=a_dst_decoder2D.per_tbin_aclu_per_lap_xy_spike_counts_df,
                max_n_cells=9, should_display_lap_by_lap_spike_counts=True,
            )

        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        assert getattr(pfs, "ndim", 2) >= 2, "plot_reliability_maps_with_spikes requires 2D PfND"
        assert reliability_active is not None and reliability_silent is not None, "reliability_active and reliability_silent are required"
        reliability_variables = str(reliability_variables).lower().strip()
        assert reliability_variables in ("both", "active", "silent"), f'reliability_variables must be "both", "active", or "silent"; got {reliability_variables!r}'

        xbin = np.asarray(pfs.xbin)
        ybin = np.asarray(pfs.ybin)
        spikes_df = pfs.filtered_spikes_df
        nx, ny = len(xbin) - 1, len(ybin) - 1
        extent = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        n_flat: int = nx * ny
        x_centers = 0.5 * (xbin[:-1] + xbin[1:])
        y_centers = 0.5 * (ybin[:-1] + ybin[1:])
        median_bin_w = float(np.median(np.diff(xbin))) if nx > 0 else 1.0
        median_bin_h = float(np.median(np.diff(ybin))) if ny > 0 else 1.0
        data_w = float(xbin[-1] - xbin[0])
        data_h = float(ybin[-1] - ybin[0])
        # Approximate displayed axes size (inches) before tight_layout; equal-aspect letterboxing applied below.
        ax_disp_in = float(figsize_per_cell) * 0.78
        data_aspect = data_w / max(data_h, 1e-9)
        if data_aspect >= 1.0:
            disp_w_in, disp_h_in = ax_disp_in, ax_disp_in / data_aspect
        else:
            disp_w_in, disp_h_in = ax_disp_in * data_aspect, ax_disp_in
        ## END if data_aspect >= 1.0...
        pts_per_x = (disp_w_in * 72.0) / max(data_w, 1e-9)
        pts_per_y = (disp_h_in * 72.0) / max(data_h, 1e-9)
        bin_w_pts = median_bin_w * pts_per_x
        bin_h_pts = median_bin_h * pts_per_y

        neuron_ids = np.asarray(neuron_ids)
        n_neurons: int = len(neuron_ids)
        R_active = np.asarray(reliability_active, dtype=float)
        R_silent = np.asarray(reliability_silent, dtype=float)
        assert R_active.shape == R_silent.shape, f'reliability_active shape {R_active.shape} != reliability_silent shape {R_silent.shape}'
        assert R_active.shape[-1] == n_neurons, f'reliability last dim {R_active.shape[-1]} != len(neuron_ids) {n_neurons}'

        estimation_mode = reliability_estimation_mode
        is_position_dependent: bool = (estimation_mode.value == ReliabilityEstimationMode.POSITION_DEPENDENT.value)
        if is_position_dependent:
            if R_active.ndim == 2:
                assert R_active.shape[0] == n_flat, f'flat reliability size {R_active.shape[0]} != n_flat {n_flat}'
            else:
                assert R_active.ndim >= 3, f'POSITION_DEPENDENT reliability expected ndim>=2, got {R_active.ndim}'
                assert int(np.prod(R_active.shape[:-1])) == n_flat, f'spatial reliability size {int(np.prod(R_active.shape[:-1]))} != n_flat {n_flat}'
            ## END if R_active.ndim == 2...
        else:
            assert R_active.ndim == 1, f'PER_CELL reliability expected shape (n_neurons,), got {R_active.shape}'
        ## END if is_position_dependent...

        plot_neuron_ids = list(included_neuron_ids) if included_neuron_ids is not None else list(neuron_ids)
        if max_n_cells is not None:
            plot_neuron_ids = plot_neuron_ids[:int(max_n_cells)]
        ## END if max_n_cells is not None...

        n = len(plot_neuron_ids)
        assert n > 0, "No neuron_ids to plot"
        aclu_to_i = {int(a): i for i, a in enumerate(neuron_ids)}

        reliability_panels: List[str] = ["active", "silent"] if reliability_variables == "both" else [reliability_variables]
        confusion_panels: List[str] = ["TP", "FP", "TN", "FN"] if (show_confusion_counts and (position_aclus_reliability_df is not None)) else []
        lap_panels: List[str] = ["laps"] if should_display_lap_by_lap_spike_counts else []
        panels: List[str] = list(reliability_panels) + list(confusion_panels) + list(lap_panels)
        n_panels: int = len(panels)
        confusion_panel_set = set(confusion_panels)
        lap_panel_set = set(lap_panels)

        # Shared lap identity order across cells from per_tbin_aclu_per_lap_xy_spike_counts_df (valid laps only)
        all_lap_ids: np.ndarray = np.array([], dtype=int)
        lap_id_to_idx: Dict[int, int] = {}
        n_laps: int = 0
        if should_display_lap_by_lap_spike_counts:
            assert per_tbin_aclu_per_lap_xy_spike_counts_df is not None, (
                'should_display_lap_by_lap_spike_counts=True requires per_tbin_aclu_per_lap_xy_spike_counts_df; '
                'call compute_unit_confusion_reliability_variables(...) with spikes that include a "lap" column.'
            )
            required_lap_cols = {'aclu', 'lap', 'binned_x', 'n_spikes'}
            missing = required_lap_cols - set(per_tbin_aclu_per_lap_xy_spike_counts_df.columns)
            assert len(missing) == 0, f'per_tbin_aclu_per_lap_xy_spike_counts_df missing columns: {sorted(missing)}'
            lap_vals = per_tbin_aclu_per_lap_xy_spike_counts_df['lap'].to_numpy()
            valid_lap_mask = pd.notna(lap_vals) & (np.asarray(lap_vals, dtype=float) > -1)
            all_lap_ids = np.sort(np.unique(np.asarray(lap_vals[valid_lap_mask], dtype=int)))
            n_laps = int(len(all_lap_ids))
            lap_id_to_idx = {int(lid): i for i, lid in enumerate(all_lap_ids)}
            if n_laps == 0:
                warnings.warn('should_display_lap_by_lap_spike_counts=True but no valid laps (lap > -1) found in per_tbin_aclu_per_lap_xy_spike_counts_df; lap panels will be empty.')
            ## END if n_laps == 0...
        ## END if should_display_lap_by_lap_spike_counts...

        if subplots is None:
            # One row per cell when showing multiple panel types (reliability both and/or confusion counts / lap stacks)
            if (reliability_variables == "both") or (len(confusion_panels) > 0) or should_display_lap_by_lap_spike_counts:
                n_rows, n_cols_cells = n, 1
            else:
                n_cols_cells = int(np.ceil(np.sqrt(n)))
                n_rows = int(np.ceil(n / n_cols_cells))
            ## END if (reliability_variables == "both") or (len(confusion_panels) > 0) or should_display_lap_by_lap_spike_counts...
        else:
            n_rows, n_cols_cells = subplots
        ## END if subplots is None...

        n_ax_cols: int = n_cols_cells * n_panels
        fig, axes = plt.subplots(n_rows, n_ax_cols, figsize=(figsize_per_cell * n_ax_cols, figsize_per_cell * n_rows), squeeze=False)
        flat_axes: List[Axes] = list(axes.ravel())

        def _map_for_cell(R: np.ndarray, cell_idx: int) -> np.ndarray:
            if R.ndim == 1:
                return np.full((nx, ny), float(R[cell_idx]), dtype=float)
            if R.ndim == 2:
                return np.asarray(R[:, cell_idx], dtype=float).reshape((nx, ny), order='C')
            return np.asarray(R[..., cell_idx], dtype=float).reshape((nx, ny), order='C')


        def _normalize_mask(aclu: int) -> Optional[np.ndarray]:
            if in_field_masks is None:
                return None
            mask = np.asarray(in_field_masks.get(aclu, np.zeros((nx, ny), dtype=bool)), dtype=bool)
            if mask.shape != (nx, ny):
                if mask.shape == (ny, nx):
                    mask = mask.T
                else:
                    raise ValueError(f"aclu {aclu}: mask shape {mask.shape} != expected {(nx, ny)}")
                ## END if mask.shape == (ny, nx)...
            ## END if mask.shape != (nx, ny)...
            return mask


        def _confusion_count_maps_for_aclu(aclu: int) -> Dict[str, np.ndarray]:
            """Build (nx, ny) TP/FP/TN/FN visit-event count maps for one aclu from position_aclus_reliability_df."""
            out = {k: np.zeros((nx, ny), dtype=float) for k in ("TP", "FP", "TN", "FN")}
            if position_aclus_reliability_df is None:
                return out
            cell_df = position_aclus_reliability_df[position_aclus_reliability_df['aclu'].to_numpy() == aclu]
            if len(cell_df) == 0:
                return out
            for row in cell_df.itertuples(index=False):
                ix = int(row.binned_x) - 1
                iy = int(row.binned_y) - 1
                if (ix < 0) or (iy < 0) or (ix >= nx) or (iy >= ny):
                    continue
                n_visits = float(getattr(row, 'n_visits', 0.0))
                n_active = float(getattr(row, 'n_active_visits', 0.0))
                n_silent = max(n_visits - n_active, 0.0)
                is_in_field = bool(getattr(row, 'is_in_field', False))
                if is_in_field:
                    out["TP"][ix, iy] = n_active
                    out["FN"][ix, iy] = n_silent
                else:
                    out["FP"][ix, iy] = n_active
                    out["TN"][ix, iy] = n_silent
                ## END if is_in_field...
            ## END for row in cell_df.itertuples(index=False)...
            return out


        def _lap_by_lap_marginal_x_counts(aclu: int) -> np.ndarray:
            """Build (n_laps, nx) spike-count matrix from per_tbin_aclu_per_lap_xy_spike_counts_df.

            One row per lap, columns = marginal_x bins (sum over t_bin_idx and binned_y).
            """
            n_rows_laps: int = max(n_laps, 1)
            out = np.zeros((n_rows_laps, nx), dtype=float)
            if (n_laps == 0) or (per_tbin_aclu_per_lap_xy_spike_counts_df is None):
                return out
            cell_df = per_tbin_aclu_per_lap_xy_spike_counts_df[per_tbin_aclu_per_lap_xy_spike_counts_df['aclu'].to_numpy() == aclu]
            if len(cell_df) == 0:
                return out
            lap_col = cell_df['lap'].to_numpy()
            valid_lap = pd.notna(lap_col) & (np.asarray(lap_col, dtype=float) > -1)
            bx = cell_df['binned_x'].to_numpy().astype(float)
            valid_bx = np.isfinite(bx) & (bx >= 1) & (bx <= nx)
            valid = valid_lap & valid_bx
            if not np.any(valid):
                return out
            lap_i = np.asarray([lap_id_to_idx[int(lid)] for lid in lap_col[valid]], dtype=int)
            bx_i = bx[valid].astype(int) - 1
            n_spk = cell_df['n_spikes'].to_numpy(dtype=float)[valid]
            np.add.at(out, (lap_i, bx_i), n_spk)
            return out


        def _draw_bin_grid(ax: Axes):
            for xv in xbin:
                ax.axvline(xv, color=bin_grid_color, linewidth=bin_grid_linewidth, alpha=bin_grid_alpha, zorder=4)
            ## END for xv in xbin...
            for yv in ybin:
                ax.axhline(yv, color=bin_grid_color, linewidth=bin_grid_linewidth, alpha=bin_grid_alpha, zorder=4)
            ## END for yv in ybin...


        def _draw_lap_x_bin_grid(ax: Axes, n_rows_laps: int):
            for xv in xbin:
                ax.axvline(xv, color=bin_grid_color, linewidth=bin_grid_linewidth, alpha=bin_grid_alpha, zorder=4)
            ## END for xv in xbin...
            for yi in range(n_rows_laps + 1):
                ax.axhline(float(yi), color=bin_grid_color, linewidth=bin_grid_linewidth, alpha=bin_grid_alpha, zorder=4)
            ## END for yi in range(n_rows_laps + 1)...


        def _bind_lap_format_coord(ax: Axes, count_mat: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, lap_ids: np.ndarray):
            """Restore status-bar hover to show bin count (pcolormesh omits z by default)."""
            def format_coord(x, y):
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    return ""
                if (x < x_edges[0]) or (x > x_edges[-1]) or (y < y_edges[0]) or (y > y_edges[-1]):
                    return f'x={x:.2f}, y={y:.2f}'
                ix = int(np.searchsorted(x_edges, x, side='right') - 1)
                iy = int(np.searchsorted(y_edges, y, side='right') - 1)
                ix = int(np.clip(ix, 0, count_mat.shape[1] - 1))
                iy = int(np.clip(iy, 0, count_mat.shape[0] - 1))
                z = int(count_mat[iy, ix])
                lap_id = int(lap_ids[iy]) if (iy < len(lap_ids)) else iy
                return f'x={x:.2f}, lap_idx={iy} (lap={lap_id}), count={z}'


            ax.format_coord = format_coord


        def _fontsize_for_count(n_digits: int) -> float:
            """Pick a label size that fits inside one median bin (height and digit-width constrained)."""
            if count_label_fontsize is not None:
                return float(count_label_fontsize)
            # ~0.55em per digit width; leave ~20% padding inside the bin
            fit_h = 0.78 * bin_h_pts
            fit_w = (0.88 * bin_w_pts) / max(0.55 * float(max(n_digits, 1)), 1e-6)
            return float(np.clip(min(fit_h, fit_w), 5.5, 11.0))


        def _draw_nonzero_count_labels(ax: Axes, count_map: np.ndarray):
            ixs, iys = np.nonzero(count_map > 0)
            for ix, iy in zip(ixs, iys):
                n_val = int(count_map[ix, iy])
                ax.text(float(x_centers[ix]), float(y_centers[iy]), f"{n_val}",
                        ha="center", va="center", fontsize=_fontsize_for_count(len(str(n_val))),
                        color="0.05", clip_on=True, zorder=5)
            ## END for ix, iy in zip(ixs, iys)...


        for cell_i, aclu in enumerate(plot_neuron_ids):
            aclu = int(aclu)
            cell_idx = aclu_to_i.get(aclu)
            assert cell_idx is not None, f'aclu {aclu} not found in neuron_ids'
            mask = _normalize_mask(aclu)
            r_a = _map_for_cell(R_active, cell_idx)
            r_s = _map_for_cell(R_silent, cell_idx)
            maps_by_panel: Dict[str, np.ndarray] = {"active": r_a, "silent": r_s}
            if len(confusion_panels) > 0:
                maps_by_panel.update(_confusion_count_maps_for_aclu(aclu))
            ## END if len(confusion_panels) > 0...
            if should_display_lap_by_lap_spike_counts:
                maps_by_panel["laps"] = _lap_by_lap_marginal_x_counts(aclu)
            ## END if should_display_lap_by_lap_spike_counts...

            row = cell_i // n_cols_cells
            col_cell = cell_i % n_cols_cells
            for panel_j, panel_name in enumerate(panels):
                ax = axes[row, col_cell * n_panels + panel_j]
                rmap = maps_by_panel[panel_name]
                is_lap_panel: bool = (panel_name in lap_panel_set)
                is_count_panel: bool = (panel_name in confusion_panel_set)
                panel_cmap = "CMRmap" if is_lap_panel else (count_cmap if is_count_panel else heatmap_cmap)
                panel_alpha = count_alpha if (is_count_panel or is_lap_panel) else heatmap_alpha
                if is_count_panel or is_lap_panel:
                    panel_vmax = float(np.nanmax(rmap)) if np.any(np.isfinite(rmap)) else 1.0
                    if (not np.isfinite(panel_vmax)) or (panel_vmax <= 0.0):
                        panel_vmax = 1.0
                    ## END if (not np.isfinite(panel_vmax)) or (panel_vmax <= 0.0)...
                    panel_vmin = 0.0
                else:
                    panel_vmin, panel_vmax = 0.0, 1.0
                ## END if is_count_panel or is_lap_panel...

                if is_lap_panel:
                    # rmap: (n_laps, nx) — rows = laps (bottom→top), cols = marginal_x
                    n_rows_laps = int(rmap.shape[0])
                    lap_y_edges = np.arange(n_rows_laps + 1, dtype=float)
                    if use_pcolormesh:
                        ax.pcolormesh(xbin, lap_y_edges, rmap, cmap=panel_cmap, alpha=panel_alpha, shading="flat", vmin=panel_vmin, vmax=panel_vmax, zorder=1)
                    else:
                        ax.imshow(rmap, origin="lower", extent=(xbin[0], xbin[-1], 0.0, float(n_rows_laps)), cmap=panel_cmap, alpha=panel_alpha, vmin=panel_vmin, vmax=panel_vmax, zorder=1, aspect="auto", interpolation="nearest")
                    ## END if use_pcolormesh...
                    if show_bin_grid:
                        _draw_lap_x_bin_grid(ax, n_rows_laps)
                    ## END if show_bin_grid...
                    _bind_lap_format_coord(ax, rmap, xbin, lap_y_edges, all_lap_ids)
                    ax.set_aspect("auto")
                    ax.set_xlim(xbin[0], xbin[-1])
                    ax.set_ylim(0.0, float(n_rows_laps))
                    title = f"aclu {aclu}  lap×x  n_laps={n_laps}  Σ={int(np.nansum(rmap))}"
                else:
                    if show_trajectory and hasattr(pfs, "x") and hasattr(pfs, "y"):
                        ax.plot(pfs.x, pfs.y, color="#d3c5c5", alpha=trajectory_alpha, linewidth=0.5, zorder=0)
                    ## END if show_trajectory...

                    if use_pcolormesh:
                        ax.pcolormesh(xbin, ybin, rmap.T, cmap=panel_cmap, alpha=panel_alpha, shading="flat", vmin=panel_vmin, vmax=panel_vmax, zorder=1)
                    else:
                        plot_r = np.fliplr(np.rot90(rmap, k=-1))
                        ax.imshow(plot_r, origin="lower", extent=extent, cmap=panel_cmap, alpha=panel_alpha, vmin=panel_vmin, vmax=panel_vmax, zorder=1, aspect="auto")
                    ## END if use_pcolormesh...

                    # Count panels: thin bin grid + integer labels (skip mask/spikes so counts stay readable)
                    if is_count_panel:
                        if show_bin_grid:
                            _draw_bin_grid(ax)
                        ## END if show_bin_grid...
                        if show_count_bin_labels:
                            _draw_nonzero_count_labels(ax, rmap)
                        ## END if show_count_bin_labels...
                    else:
                        if mask is not None:
                            if use_pcolormesh:
                                ax.pcolormesh(xbin, ybin, mask.T.astype(float), cmap=mask_cmap, alpha=mask_alpha, shading="flat", vmin=0, vmax=1, zorder=2)
                            else:
                                plot_mask = np.fliplr(np.rot90(mask.astype(float), k=-1))
                                ax.imshow(plot_mask, origin="lower", extent=extent, cmap=mask_cmap, alpha=mask_alpha, vmin=0, vmax=1, zorder=2, aspect="auto")
                            ## END if use_pcolormesh...
                        ## END if mask is not None...

                        cell_spk = spikes_df[spikes_df["aclu"] == aclu]
                        if len(cell_spk) > 0:
                            if color_by_in_field and (mask is not None) and {"binned_x", "binned_y"}.issubset(cell_spk.columns):
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
                    ## END if is_count_panel...

                    ax.set_aspect("equal")
                    ax.set_xlim(xbin[0], xbin[-1])
                    ax.set_ylim(ybin[0], ybin[-1])
                    if is_count_panel:
                        title = f"aclu {aclu}  {panel_name}  Σ={int(np.nansum(rmap))}"
                    elif not is_position_dependent:
                        title = f"aclu {aclu}  R_a={float(R_active[cell_idx]):.2f}  R_s={float(R_silent[cell_idx]):.2f}  ({panel_name})"
                    else:
                        title = f"aclu {aclu}  ({panel_name})"
                    ## END if is_count_panel...
                ## END if is_lap_panel...
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            ## END for panel_j, panel_name in enumerate(panels)...
        ## END for cell_i, aclu in enumerate(plot_neuron_ids)...

        n_used = n * n_panels
        for ax in flat_axes[n_used:]:
            ax.axis("off")
        ## END for ax in flat_axes[n_used:]...

        mode_label = str(estimation_mode)
        count_suffix = " + TP/FP/TN/FN" if len(confusion_panels) > 0 else ""
        lap_suffix = " + lap×x spike counts" if should_display_lap_by_lap_spike_counts else ""
        fig.suptitle(f"Reliability maps ({mode_label}){count_suffix}{lap_suffix} + spikes", fontsize=12)
        fig.tight_layout()
        return fig, axes



@metadata_attributes(short_name=None, tags=['reliability', 'decoder'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-29 09:04', related_items=[])
class CellIndividualReliabilityComputingMixin:
    """ Implementors compute and use cell individual reliabilities.

    Usage:
        from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityComputingMixin, CellIndividualReliabilityMatrix, ReliabilityDecoderModifierMode, ReliabilityEstimationMode

    Required ``self.`` properties (implementor must provide)
    -------------------------------------------------------
    Core decoder / placefield state:
        pf : PfND
            2D placefield; ``assert self.pf is not None`` in ``_compute_reliability_metrics``.
        ratemap
            Ratemap for bin edges / ``neuron_ids`` fallback (typically ``self.pf.ratemap``).
        neuron_IDs : optional array-like
            Neuron id order; if ``None``, uses ``self.ratemap.neuron_ids``.
        spikes_df : optional DataFrame
            May start as ``None``; filled from ``pf.filtered_spikes_df`` when needed.
        time_bin_size : optional float
            Default temporal bin width; may be written if ``None`` when compute is called.
        F : optional ndarray
            Flat tuning matrix; when not ``None``, ``flat_position_size`` is used for map shape checks.
        flat_position_size : int
            Number of flat position bins (used when ``F is not None``).
        original_position_data_shape : tuple
            Occupancy / map shape (used when ``F is None`` or in-field masks are empty).

    Reliability configuration (read):
        n_top_peaks : int
            Top-N peaks for in-field mask contours.
        slice_level_multiplier : float
            Contour height fraction for in-field masks (e.g. 0.2).
        reliability_estimation_mode : ReliabilityEstimationMode
            ``PER_CELL`` vs ``POSITION_DEPENDENT`` (defaults via ``getattr`` to ``PER_CELL`` if missing).
        should_discount_silence : bool
            If True, ``reliability_silent`` uses confusion/visit rates; else ones.

    Written / updated by this mixin
    -------------------------------
        in_field_masks, t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df,
        per_tbin_aclu_xy_spike_counts_df, per_tbin_aclu_per_lap_xy_spike_counts_df,
        time_bin_info_df, per_tbin_aclu_spike_counts_sparse,
        position_aclus_reliability_df, reliability_active, reliability_silent
        (and optionally ``spikes_df`` / ``time_bin_size`` when they were ``None``).

    """
    # ==================================================================================================================================================================================================================================================================================== #
    # Cell Reliability Computations                                                                                                                                                                                                                                                        #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['UNUSED', 'ALT', 'pho', 'true-positive', 'false-positive', 'reliability'], input_requires=[], output_provides=[], uses=['CellIndividualReliabilityMatrix.build_in_field_masks_xy_from_pf', 'CellIndividualReliabilityMatrix.build_in_field_masks_xy', 'CellIndividualReliabilityMatrix.compute_reliability_matrix', '_compute_reliability_metrics'], used_by=[], creation_date='2026-07-23 09:58', related_items=[])
    def _perform_compute_unit_confusion_reliability_variables(self, active_peak_prominence_2d_results=None, spikes_df: Optional[pd.DataFrame] = None, time_bin_size_seconds: Optional[float] = None, max_t_idx: Optional[int] = None, **kwargs):
        """Compute per-aclu confusion-matrix reliability products and refresh ``reliability_*`` on self.

        After writing confusion products, calls ``_compute_reliability_metrics()`` so
        ``reliability_active`` / ``reliability_silent`` match ``reliability_estimation_mode``
        (``PER_CELL`` or ``POSITION_DEPENDENT``).

        Parameters
        ----------
        active_peak_prominence_2d_results : optional PeakProminence2D results for in-field masks.
            If None, builds masks via ``CellIndividualReliabilityMatrix.build_in_field_masks_xy_from_pf``
            (recomputes PeakProminence2D from ``self.pf``; no pipeline cache required).
            If provided, uses ``build_in_field_masks_xy`` with those results.
        spikes_df : optional spikes override; defaults to `self.spikes_df` sliced to `self.neuron_IDs`.
        time_bin_size_seconds : temporal bin width; defaults to `self.time_bin_size`.
        max_t_idx : optional cap on number of time bins (None = all).

        Uses instance fields ``n_top_peaks``, ``slice_level_multiplier``, ``fn_tn_mode``, and ``reliability_estimation_mode``.

        Returns
        -------
        t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df


        UPDATES:
            self.in_field_masks, self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.per_tbin_aclu_xy_spike_counts_df,
            self.per_tbin_aclu_per_lap_xy_spike_counts_df (when spikes have ``lap``; else None),
            self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse, self.position_aclus_reliability_df (POSITION_DEPENDENT),
            self.reliability_active, self.reliability_silent


        Usage:

            t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse, per_tbin_aclu_xy_spike_counts_df = a_dst_decoder2D.compute_unit_confusion_reliability_variables()
            # also stored: a_dst_decoder2D.per_tbin_aclu_per_lap_xy_spike_counts_df

        """
        pfs = self.pf
        ratemaps = self.ratemap
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else ratemaps.neuron_ids)
        if spikes_df is None:
            if self.spikes_df is None:
                self.spikes_df = deepcopy(pfs.filtered_spikes_df).spikes.sliced_by_neuron_id(neuron_ids)
            spikes_df = deepcopy(self.spikes_df)
            spikes_df = spikes_df.spikes.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True)


        spikes_df = spikes_df.spikes.sliced_by_neuron_id(neuron_ids)
        if time_bin_size_seconds is None:
            time_bin_size_seconds = self.time_bin_size

        if (self.spikes_df is None):
            self.spikes_df = spikes_df

        if (self.time_bin_size is None):
            self.time_bin_size = time_bin_size_seconds

        if active_peak_prominence_2d_results is None:
            self.in_field_masks = CellIndividualReliabilityMatrix.build_in_field_masks_xy_from_pf(
                pf=pfs, n_top_peaks=self.n_top_peaks, slice_level_multiplier=self.slice_level_multiplier, neuron_ids=neuron_ids,
            )
        else:
            self.in_field_masks = CellIndividualReliabilityMatrix.build_in_field_masks_xy(
                active_peak_prominence_2d_results=active_peak_prominence_2d_results, ratemaps=ratemaps,
                n_top_peaks=self.n_top_peaks, slice_level_multiplier=self.slice_level_multiplier,
                neuron_ids=neuron_ids,
            )

        ## add binned:
        spikes_df = spikes_df.spikes.adding_binned_position_columns(xbin_edges=ratemaps.xbin, ybin_edges=ratemaps.ybin, position_column_names=('x', 'y'), binned_column_names=('binned_x', 'binned_y'), force_recompute=True) ## #TODO 2026-07-28 19:47: - [ ] inefficient to do this again and again

        self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse, self.per_tbin_aclu_xy_spike_counts_df, self.per_tbin_aclu_per_lap_xy_spike_counts_df = CellIndividualReliabilityMatrix.compute_reliability_matrix(
            spikes_df=spikes_df, pfs=pfs, ratemaps=ratemaps, in_field_masks=self.in_field_masks, neuron_ids=neuron_ids,
            time_bin_size_seconds=time_bin_size_seconds, max_t_idx=max_t_idx,
            reliability_estimation_mode=self.reliability_estimation_mode, **kwargs,
        )

        self.compute_reliability_metrics()
        return self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse, self.per_tbin_aclu_xy_spike_counts_df


    @function_attributes(short_name=None, tags=['_DEP', 'old', 'simplification'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-30 04:45', related_items=[])
    def _build_position_dependent_reliability_maps(self, true_pos: np.ndarray, false_pos: np.ndarray, true_neg: np.ndarray, false_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: ``(n_flat, n_neurons)`` maps from global per-cell confusion rates × in-field masks.

        Preferred POSITION_DEPENDENT path uses visit-conditioned
        ``CellIndividualReliabilityMatrix.perform_compute_position_dependent_reliability_maps``.

        For hypothesized position x and cell i:
            in-field:  R_active = true_pos[i],  R_silent = 1 - false_neg[i]
            out-field: R_active = 1 - false_pos[i], R_silent = true_neg[i]
        """
        assert self.in_field_masks is not None, 'POSITION_DEPENDENT reliability requires in_field_masks; call compute_unit_confusion_reliability_variables(...) first.'
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else self.ratemap.neuron_ids)
        n_neurons: int = len(neuron_ids)
        n_flat: int = int(self.flat_position_size) if (self.F is not None) else int(np.prod(self.original_position_data_shape))

        in_field_flat = np.zeros((n_flat, n_neurons), dtype=bool)
        for i, nid in enumerate(neuron_ids):
            mask = self.in_field_masks.get(int(nid), None)
            if mask is None:
                continue
            flat_mask = np.asarray(mask).ravel(order='C')
            assert flat_mask.size == n_flat, f'in_field_masks[{nid}] flat size {flat_mask.size} != n_flat_position_bins {n_flat}'
            in_field_flat[:, i] = flat_mask
        ## END for i, nid in enumerate(neuron_ids)...

        R_active = np.where(in_field_flat, true_pos[np.newaxis, :], (1.0 - false_pos)[np.newaxis, :])
        R_silent = np.where(in_field_flat, (1.0 - false_neg)[np.newaxis, :], true_neg[np.newaxis, :])
        return np.nan_to_num(R_active, nan=0.0), np.nan_to_num(R_silent, nan=0.0)


    def compute_reliability_metrics(self, debug_print: bool=False, **kwargs):
        """ Called after main confusion computation
        Builds reliability arrays from confusion-matrix rates (no Skaggs).

        Modes (``self.reliability_estimation_mode``):
            PER_CELL: ``(n_neurons,)`` from ``true_pos`` (default).
            POSITION_DEPENDENT: visit-conditioned maps from
                ``per_tbin_aclu_spike_counts_df`` × animal ``time_bin_info_df`` (falls back to global rates × ``in_field_masks``).
                Stored as ``(*original_position_data_shape, n_neurons)`` for 2D decoders (e.g. ``(nx, ny, n_neurons)``);
                flat ``(n_flat, n_neurons)`` for 1D.

        If ``reliability_modifier_mode == IGNORE``:
            set ``reliability_*`` to ones and return (no discounting; avoids DST decode recursion).

        If ``t_bin_aclus_reliability_df`` is missing (and not IGNORE):
            run ``compute_unit_confusion_reliability_variables`` for both PER_CELL and POSITION_DEPENDENT
            (both need confusion rates), then return (nested metrics already refreshed ``reliability_*``).

        Updates: self.reliability_active, self.reliability_silent

        """
        assert (self.pf is not None)
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else self.ratemap.neuron_ids)
        n_neurons: int = int(len(neuron_ids))
        estimation_mode = getattr(self, 'reliability_estimation_mode', ReliabilityEstimationMode.PER_CELL)

        is_ignore_mode: bool = (self.reliability_modifier_mode.value == ReliabilityDecoderModifierMode.IGNORE.value)
        if is_ignore_mode:
            if debug_print:
                print(f'WARN: ._compute_reliability_metrics(...): called in (self.reliability_modifier_mode == ReliabilityDecoderModifierMode.IGNORE) mode. skipping computations and returning.')

            R_ones = np.ones(n_neurons, dtype=float)
            self.reliability_active = R_ones
            self.reliability_silent = np.ones_like(R_ones)
            return ### ignore, skipping computations

        has_confusion: bool = (self.t_bin_aclus_reliability_df is not None) and ('true_pos' in self.t_bin_aclus_reliability_df.columns)
        should_compute: bool = (not has_confusion) and (not is_ignore_mode)

        if should_compute:
            if debug_print:
                print(f'WARN: ._compute_reliability_metrics(...): reliability requires t_bin_aclus_reliability_df with true_pos; calling .compute_unit_confusion_reliability_variables(...) first...')
            _ = self._perform_compute_unit_confusion_reliability_variables(**kwargs) ## performing compute.
            if debug_print:
                print(f'\tdone.')
            if self.t_bin_aclus_reliability_df is None:
                raise ValueError('reliability requires t_bin_aclus_reliability_df with true_pos; AND CALLING .compute_unit_confusion_reliability_variables(...) resulted in an empty reliability!')
            return  # nested call already refreshed reliability_*

        rel_df = self.t_bin_aclus_reliability_df.reindex(neuron_ids)
        true_pos = np.nan_to_num(rel_df['true_pos'].to_numpy(dtype=float), nan=0.0)
        false_pos = np.nan_to_num(rel_df['false_pos'].to_numpy(dtype=float), nan=0.0) if ('false_pos' in rel_df.columns) else (1.0 - true_pos)
        true_neg = np.nan_to_num(rel_df['true_neg'].to_numpy(dtype=float), nan=0.0) if ('true_neg' in rel_df.columns) else np.zeros_like(true_pos)
        false_neg = np.nan_to_num(rel_df['false_neg'].to_numpy(dtype=float), nan=0.0) if ('false_neg' in rel_df.columns) else np.zeros_like(true_pos)
        assert len(true_pos) == n_neurons, f'Confusion rates length {len(true_pos)} != n_neurons {n_neurons} after reindex by neuron_IDs.'

        if (estimation_mode.value == ReliabilityEstimationMode.POSITION_DEPENDENT.value):
            has_visit_tables: bool = (self.per_tbin_aclu_spike_counts_df is not None) and (self.time_bin_info_df is not None) and (self.in_field_masks is not None)
            if has_visit_tables:
                if len(self.in_field_masks) > 0:
                    sample_mask = next(iter(self.in_field_masks.values()))
                    occupancy_shape = tuple(np.asarray(sample_mask).shape)
                else:
                    occupancy_shape = tuple(np.asarray(self.original_position_data_shape))
                in_field_lut = CellIndividualReliabilityMatrix.build_in_field_lut(self.in_field_masks)
                R_active, R_silent_from_confusion, self.position_aclus_reliability_df = CellIndividualReliabilityMatrix.perform_compute_position_dependent_reliability_maps(
                    per_tbin=self.per_tbin_aclu_spike_counts_df, time_bin_info_df=self.time_bin_info_df, neuron_ids=neuron_ids,
                    in_field_lut=in_field_lut, occupancy_shape=occupancy_shape, **kwargs,
                )
                n_flat: int = int(self.flat_position_size) if (self.F is not None) else int(np.prod(self.original_position_data_shape))
                assert R_active.shape == (n_flat, n_neurons), f'visit-conditioned R_active shape {R_active.shape} != ({n_flat}, {n_neurons})'
            else:
                # Slice / partial state: fall back to global rates × masks
                raise NotImplementedError(f'_build_position_dependent_reliability_maps is off-limits because it is wrong. FIXME!')
                self.position_aclus_reliability_df = None
                R_active, R_silent_from_confusion = self._build_position_dependent_reliability_maps(true_pos=true_pos, false_pos=false_pos, true_neg=true_neg, false_neg=false_neg)

            ## OUTPUTS: R_active, R_silent_from_confusion — reshape to (*spatial, n_neurons) for 2D
            if (int(self.ndim) >= 2) and (R_active.ndim == 2):
                spatial_shape = tuple(np.asarray(self.original_position_data_shape))
                assert R_active.shape[0] == int(np.prod(spatial_shape)), f'R_active flat size {R_active.shape[0]} != prod(spatial_shape) {int(np.prod(spatial_shape))}'
                R_active = R_active.reshape((*spatial_shape, R_active.shape[-1]))
                R_silent_from_confusion = R_silent_from_confusion.reshape((*spatial_shape, R_silent_from_confusion.shape[-1]))

            self.reliability_active = R_active
            if self.should_discount_silence:
                self.reliability_silent = R_silent_from_confusion
            else:
                self.reliability_silent = np.ones_like(R_active)
        else:
            # PER_CELL: position-independent reliability from true_pos
            # self.position_aclus_reliability_df = None ## do not need to clear the per-position reliability
            R_base = true_pos
            self.reliability_active = R_base
            # Map reliability for silence (n_i = 0).
            # Defaults to 1.0 (perfect reliability -> collapses to pure Bayesian) if discounting is disabled.
            if self.should_discount_silence:
                self.reliability_silent = R_base
            else:
                self.reliability_silent = np.ones_like(R_base)


    # ==================================================================================================================================================================================================================================================================================== #
    # Plotting Display                                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['matplotlib', 'figure', 'passthrough'], input_requires=[], output_provides=[], uses=['CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes'], used_by=[], creation_date='2026-07-29 14:35', related_items=[])
    def plot_in_field_masks_with_spikes(self, included_neuron_ids: Optional[Sequence[int]] = None, **kwargs):
        """Passthrough to ``CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes`` using ``self.pf`` / ``self.in_field_masks``.

        Usage:
            fig, axes = a_dst_decoder2D.plot_in_field_masks_with_spikes(max_n_cells=9)
        """
        assert self.in_field_masks is not None, 'in_field_masks is required; call compute_unit_confusion_reliability_variables(...) first.'
        return CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes(
            self.pf, self.in_field_masks,
            included_neuron_ids=included_neuron_ids if included_neuron_ids is not None else (self.neuron_IDs if self.neuron_IDs is not None else None),
            **kwargs,
        )


    @function_attributes(short_name=None, tags=['matplotlib', 'figure', 'passthrough', 'reliability'], input_requires=[], output_provides=[], uses=['CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes'], used_by=[], creation_date='2026-07-29 14:35', related_items=[])
    def plot_reliability_maps_with_spikes(self, included_neuron_ids: Optional[Sequence[int]] = None, **kwargs):
        """Passthrough to ``CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes`` using decoder reliability state.

        Usage:
            fig, axes = a_dst_decoder2D.plot_reliability_maps_with_spikes(max_n_cells=9, should_display_lap_by_lap_spike_counts=True)
        """
        assert self.reliability_active is not None and self.reliability_silent is not None, 'reliability_active and reliability_silent are required; call compute_reliability_metrics(...) first.'
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else self.ratemap.neuron_ids)
        return CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes(
            self.pf, self.reliability_active, self.reliability_silent, neuron_ids,
            reliability_estimation_mode=getattr(self, 'reliability_estimation_mode', ReliabilityEstimationMode.PER_CELL),
            in_field_masks=self.in_field_masks,
            position_aclus_reliability_df=getattr(self, 'position_aclus_reliability_df', None),
            per_tbin_aclu_per_lap_xy_spike_counts_df=getattr(self, 'per_tbin_aclu_per_lap_xy_spike_counts_df', None),
            included_neuron_ids=included_neuron_ids, **kwargs,
        )

