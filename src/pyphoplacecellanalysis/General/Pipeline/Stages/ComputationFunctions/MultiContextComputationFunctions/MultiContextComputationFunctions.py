from enum import Enum # required by `FiringRateActivitySource` enum
from dataclasses import dataclass # required by `SortOrderMetric` class

import numpy as np
import pandas as pd

from neuropy.utils.misc import safe_pandas_get_group # for _compute_pybursts_burst_interval_detection

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_firing_rate_trends_computation

from shapely.geometry import LineString # for compute_polygon_overlap
from shapely.ops import unary_union, polygonize # for compute_polygon_overlap
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons, _compare_computation_results # for compute_polygon_overlap
from scipy.signal import convolve as convolve # compute_convolution_overlap
from scipy import stats # for compute_relative_entropy_divergence_overlap
from scipy.special import rel_entr # alternative for compute_relative_entropy_divergence_overlap

from collections import Counter # Count the Number of Occurrences in a Python list using Counter
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership

from neuropy.analyses import detect_pbe_epochs # used in `_perform_jonathan_replay_firing_rate_analyses(.)` if replays are missing

def _compute_custom_PBEs(sess):
    """ 
        new_pbe_epochs = _compute_custom_PBEs(sess)
    """
    print('computing PBE epochs for session...\n')
    # smth_mua = curr_active_pipeline.sess.mua.get_smoothed(sigma=0.02) # Get the smoothed mua from the session's mua
    # pbe = detect_pbe_epochs(smth_mua, thresh=(0, 3), min_dur=0.1, merge_dur=0.01, max_dur=1.0) # Default
    # pbe.to_dataframe()
    # new_pbe_epochs = detect_pbe_epochs(smth_mua, thresh=(0, 1.5), min_dur=0.06, merge_dur=0.06, max_dur=2.3) # Kamran's Parameters
    smth_mua = sess.mua.get_smoothed(sigma=0.030)
    new_pbe_epochs = detect_pbe_epochs(smth_mua, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.300) # NewPaper's Parameters
    return new_pbe_epochs

def _wrap_multi_context_computation_function(global_comp_fcn):
    """ captures global_comp_fcn and unwraps its arguments: owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False """
    def _(x):
        assert len(x) > 4, f"{x}"
        x[1] = global_comp_fcn(*x) # update global_computation_results
        return x
    return _


class MultiContextComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'multi_context'
    _computationPrecidence = 1000
    _is_global = True


    def _perform_PBE_stats_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False):
        """ 
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['PBE_stats_analyses']
                ['PBE_stats_analyses']['pbe_analyses_result_df']
                ['PBE_stats_analyses']['all_epochs_info']
        
        """
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
        
        pbe_analyses_result_df, all_epochs_info = _perform_PBE_stats(owning_pipeline_reference, include_whitelist=include_whitelist, debug_print=debug_print)

        global_computation_results.computed_data['PBE_stats_analyses'] = DynamicParameters.init_from_dict({
            'pbe_analyses_result_df': pbe_analyses_result_df,
            'all_epochs_info': all_epochs_info,
        })
        return global_computation_results


    def _perform_jonathan_replay_firing_rate_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False):
        """ Ported from Jonathan's `Gould_22-09-29.ipynb` Notebook
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['jonathan_firing_rate_analysis']
                ['jonathan_firing_rate_analysis']['rdf']:
                    ['jonathan_firing_rate_analysis']['rdf']['rdf']
                    ['jonathan_firing_rate_analysis']['rdf']['aclu_to_idx']
                    
                ['jonathan_firing_rate_analysis']['irdf']:
                    ['jonathan_firing_rate_analysis']['irdf']['irdf']
                    ['jonathan_firing_rate_analysis']['irdf']['aclu_to_idx']

                ['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']:
                    ['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_bins']
                    ['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_binned_unit_specific_binned_spike_rate']

                ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']:
                    ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
                    ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']

                ['jonathan_firing_rate_analysis']['neuron_replay_stats_df']
        
        """
        replays_df = None
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        # Epoch dataframe stuff:
        long_epoch_name = include_whitelist[0] # 'maze1_PYR'
        short_epoch_name = include_whitelist[1] # 'maze2_PYR'
        if len(include_whitelist) > 2:
            global_epoch_name = include_whitelist[-1] # 'maze_PYR'
        else:
            print(f'WARNING: no global_epoch detected.')
            global_epoch_name = '' # None

        if debug_print:
            print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')
        pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
        pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
        # pf1d = computation_results[global_epoch_name]['computed_data']['pf1D']

        ## Compute for all the session spikes first:

        # ## Use the filtered spikes from the global_epoch_name: these are those that pass the filtering stage (such as Pyramidal-only). These include spikes and aclus that are not incldued in the placefields.
        # assert global_epoch_name in owning_pipeline_reference.filtered_sessions, f"global_epoch_name: {global_epoch_name} not in owning_pipeline_reference.filtered_sessions.keys(): {list(owning_pipeline_reference.filtered_sessions.keys())}"
        # sess = owning_pipeline_reference.filtered_sessions[global_epoch_name] # get the filtered session with the global_epoch_name (which we assert exists!)

        # I think this is the correct way:
        assert global_epoch_name in computation_results, f"global_epoch_name: {global_epoch_name} not in computation_results.keys(): {list(computation_results.keys())}"
        sess = computation_results[global_epoch_name].sess # should be the same as `owning_pipeline_reference.filtered_sessions[global_epoch_name]`
        assert sess is not None

        ## Unfiltered mode (probably a mistake)
        # sess = owning_pipeline_reference.sess

        ## neuron_IDs used for instantaneous_unit_specific_spike_rate to build the dataframe:
        neuron_IDs = np.unique(sess.spikes_df.aclu) # TODO: make sure standardized

        ## HERE I CAN SPECIFY WHICH REPLAYS TO USE FOR THE ANALYSIS:
        # print(f'replays_df: {replays_df}, type(replays_df): {type(replays_df)}')
        # if replays_df is None:
        # If not replays_df argument is provided, get it from `sess`:
        try:
            replays_df = sess.replay
        except AttributeError as e:
            print(f'session is missing the `sess.replay` property. Falling back to sess.pbe.to_dataframe()...')
            new_pbe_epochs = _compute_custom_PBEs(sess)
            sess.pbe = new_pbe_epochs # copy the detected PBEs to the session
            replays_df = sess.pbe.to_dataframe()
            # replays_df = sess.ripple.to_dataframe()
        except Exception as e:
            raise e
        

        # else:
        #     replays_df = replays_df.copy() # make a copy of the provided df


        rdf, aclu_to_idx, irdf, aclu_to_idx_irdf = _final_compute_jonathan_replay_fr_analyses(sess, replays_df)
        rdf, neuron_replay_stats_df = _compute_neuron_replay_stats(rdf, aclu_to_idx) # neuron_replay_stats_df is joined with `final_jonathan_df` after that is built

        ## time_binned_unit_specific_binned_spike_rate mode:
        try:
            active_firing_rate_trends = computation_results[global_epoch_name]['computed_data']['firing_rate_trends']    
            time_bins = active_firing_rate_trends.all_session_spikes.time_binning_container.centers
            time_binned_unit_specific_binned_spike_rate = active_firing_rate_trends.all_session_spikes.time_binned_unit_specific_binned_spike_rate
        except KeyError:
            time_bins, time_binned_unit_specific_binned_spike_rate = {}, {}
        time_binned_unit_specific_spike_rate_result = DynamicParameters.init_from_dict({
            'time_bins': time_bins.copy(),
            'time_binned_unit_specific_binned_spike_rate': time_binned_unit_specific_binned_spike_rate,           
        })

        ## instantaneous_unit_specific_spike_rate mode:
        try:
            active_firing_rate_trends = computation_results[global_epoch_name]['computed_data']['firing_rate_trends']
            # neuron_IDs = np.unique(computation_results[global_epoch_name].sess.spikes_df.aclu) # TODO: make sure standardized
            instantaneous_unit_specific_spike_rate = active_firing_rate_trends.all_session_spikes.instantaneous_unit_specific_spike_rate
            # instantaneous_unit_specific_spike_rate = computation_results[global_epoch_name]['computed_data']['firing_rate_trends'].all_session_spikes.instantaneous_unit_specific_spike_rate
            instantaneous_unit_specific_spike_rate_values = pd.DataFrame(instantaneous_unit_specific_spike_rate.magnitude, columns=neuron_IDs) # builds a df with times along the rows and aclu values along the columns in the style of unit_specific_binned_spike_counts
            time_bins = instantaneous_unit_specific_spike_rate.times.magnitude # .shape (3429,)
        except KeyError:
            time_bins, instantaneous_unit_specific_spike_rate_values = {}, {}
        instantaneous_unit_specific_spike_rate_result = DynamicParameters.init_from_dict({
            'time_bins': time_bins.copy(),
            'instantaneous_unit_specific_spike_rate_values': instantaneous_unit_specific_spike_rate_values,           
        })

        final_jonathan_df = _subfn_computations_make_jonathan_firing_comparison_df(time_binned_unit_specific_binned_spike_rate, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf)
        final_jonathan_df = final_jonathan_df.join(neuron_replay_stats_df, how='outer')

        # Uses `aclu_to_idx` to add the ['active_aclus', 'is_neuron_active'] columns
        # Uses to add ['num_long_only_neuron_participating', 'num_shared_neuron_participating', 'num_short_only_neuron_participating'] columns
        flat_matrix = make_fr(rdf) # flat_matrix.shape # (116, 52) # (n_replays, n_neurons)
        n_replays = np.shape(flat_matrix)[0] # 743
        is_inactive_mask = np.isclose(flat_matrix, 0.0)
        is_active_mask = np.logical_not(is_inactive_mask) # .shape # (743, 70)

        rdf_aclus = np.array(list(aclu_to_idx.keys()))
        aclu_to_track_membership_map = {aclu:row['track_membership'] for aclu, row in final_jonathan_df.iterrows()} # {2: <SplitPartitionMembership.LEFT_ONLY: 0>, 3: <SplitPartitionMembership.SHARED: 1>, ...}
        is_cell_active_list = []
        active_aclus_list = []
        num_long_only_neuron_participating = []
        num_shared_neuron_participating = []
        num_short_only_neuron_participating = []

        for i, (replay_index, row) in enumerate(rdf.iterrows()):
            active_aclus = rdf_aclus[is_active_mask[i]]
            # get the aclu's long_only/shared/short_only identity
            active_cells_track_membership = [aclu_to_track_membership_map[aclu] for aclu in active_aclus]
            counts = Counter(active_cells_track_membership) # Counter({<SplitPartitionMembership.LEFT_ONLY: 0>: 3, <SplitPartitionMembership.SHARED: 1>: 7, <SplitPartitionMembership.RIGHT_ONLY: 2>: 7})
            num_long_only_neuron_participating.append(counts[SplitPartitionMembership.LEFT_ONLY])
            num_shared_neuron_participating.append(counts[SplitPartitionMembership.SHARED])
            num_short_only_neuron_participating.append(counts[SplitPartitionMembership.RIGHT_ONLY])
            is_cell_active_list.append(is_active_mask[i])
            active_aclus_list.append(active_aclus)

        rdf = rdf.assign(is_neuron_active=is_cell_active_list, active_aclus=active_aclus_list,
                        num_long_only_neuron_participating=num_long_only_neuron_participating,
                        num_shared_neuron_participating=num_shared_neuron_participating,
                        num_short_only_neuron_participating=num_short_only_neuron_participating)

        global_computation_results.computed_data['jonathan_firing_rate_analysis'] = DynamicParameters.init_from_dict({
            'rdf': DynamicParameters.init_from_dict({
                'rdf': rdf,
                'aclu_to_idx': aclu_to_idx, 
            }),
            'irdf': DynamicParameters.init_from_dict({
                'irdf': irdf,
                'aclu_to_idx': aclu_to_idx_irdf,
            }),
            'time_binned_unit_specific_spike_rate': time_binned_unit_specific_spike_rate_result,
            'time_binned_instantaneous_unit_specific_spike_rate': instantaneous_unit_specific_spike_rate_result,
            'neuron_replay_stats_df': final_jonathan_df
        })
        return global_computation_results


    def _perform_short_long_pf_overlap_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False):
        """ 
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['short_long_pf_overlap_analyses']
                ['short_long_pf_overlap_analyses']['short_long_neurons_diff']
                ['short_long_pf_overlap_analyses']['poly_overlap_df']
        
        """
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        # Epoch dataframe stuff:
        long_epoch_name = include_whitelist[0] # 'maze1_PYR'
        short_epoch_name = include_whitelist[1] # 'maze2_PYR'
        if len(include_whitelist) > 2:
            global_epoch_name = include_whitelist[-1] # 'maze_PYR'
        else:
            print(f'WARNING: no global_epoch detected.')
            global_epoch_name = '' # None

        if debug_print:
            print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')

        long_results = computation_results[long_epoch_name]['computed_data']
        short_results = computation_results[short_epoch_name]['computed_data']

        # Compute various forms of 1D placefield overlaps:        
        pf_neurons_diff = _compare_computation_results(long_results.pf1D.ratemap.neuron_ids, short_results.pf1D.ratemap.neuron_ids) # get shared neuron info
        poly_overlap_df = compute_polygon_overlap(long_results, short_results, debug_print=debug_print)
        conv_overlap_dict, conv_overlap_scalars_df = compute_convolution_overlap(long_results, short_results, debug_print=debug_print)
        product_overlap_dict, product_overlap_scalars_df = compute_dot_product_overlap(long_results, short_results, debug_print=debug_print)
        relative_entropy_overlap_dict, relative_entropy_overlap_scalars_df = compute_relative_entropy_divergence_overlap(long_results, short_results, debug_print=debug_print)



        global_computation_results.computed_data['short_long_pf_overlap_analyses'] = DynamicParameters.init_from_dict({
            'short_long_neurons_diff': pf_neurons_diff,
            'poly_overlap_df': poly_overlap_df,
            'conv_overlap_dict': conv_overlap_dict, 'conv_overlap_scalars_df': conv_overlap_scalars_df,
            'product_overlap_dict': product_overlap_dict, 'product_overlap_scalars_df': product_overlap_scalars_df,
            'relative_entropy_overlap_dict': relative_entropy_overlap_dict, 'relative_entropy_overlap_scalars_df': relative_entropy_overlap_scalars_df
        })
        return global_computation_results



# ==================================================================================================================== #
# Jonathan's helper functions                                                                                          #
# ==================================================================================================================== #
def _final_compute_jonathan_replay_fr_analyses(sess, replays_df, debug_print=False):
    """_summary_

    Args:
        sess (_type_): _description_
        replays_df (pd.DataFrame): sess.replay dataframe. Must have [["start", "end"]] columns

    Returns:
        _type_: _description_

    Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _final_compute_jonathan_replay_fr_analyses
            ## Compute for all the session spikes first:
            sess = owning_pipeline_reference.sess
            # BAD DOn'T DO THIS:
            rdf, aclu_to_idx, irdf, aclu_to_idx_irdf = _final_compute_jonathan_replay_fr_analyses(sess)
            pos_df = sess.position.to_dataframe()


    """
    ## Compute for all the session spikes first:
    # assert ["start", "end"] in replays_df.columns,

    if 'end' not in replays_df.columns:
        # Adds the 'end' column if needed
        replays_df['end'] = replays_df['stop']

    ### Make `rdf` (replay dataframe)
    rdf = make_rdf(sess, replays_df) # this creates the replay dataframe variable
    rdf = remove_repeated_replays(rdf)
    rdf, aclu_to_idx = add_spike_counts(sess, rdf)

    rdf = remove_nospike_replays(rdf)
    rdf['duration'] = rdf['end'] - rdf['start']
    if debug_print:
        print(f"RDF has {len(rdf)} rows.")

    ### Make `irdf` (inter-replay dataframe)
    irdf = make_irdf(sess, rdf)
    irdf = remove_repeated_replays(irdf) # TODO: make the removal process more meaningful
    irdf, aclu_to_idx_irdf = add_spike_counts(sess, irdf)
    irdf['duration'] = irdf['end'] - irdf['start']
    assert aclu_to_idx_irdf == aclu_to_idx # technically, these might not match, which would be bad

    return rdf, aclu_to_idx, irdf, aclu_to_idx_irdf


def _subfn_computations_make_jonathan_firing_comparison_df(unit_specific_time_binned_firing_rates, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf, debug_print=False):
    """ the computations that were factored out of _make_jonathan_interactive_plot(...) 
    Historical: used to be called `_subfn_computations_make_jonathan_interactive_plot(...)`
    """
    # ==================================================================================================================== #
    ## Calculating:

    ## The actual firing rate we want:
    
    # unit_specific_time_binned_firing_rates = pf2D_Decoder.unit_specific_time_binned_spike_counts.astype(np.float32) / pf2D_Decoder.time_bin_size
    if debug_print:
        print(f'np.shape(unit_specific_time_binned_firing_rates): {np.shape(unit_specific_time_binned_firing_rates)}')

    # calculations for ax[0,0] ___________________________________________________________________________________________ #
    # below we find where the tuning curve peak was for each cell in each context and store it in a dataframe
    # pf1d_long = computation_results['maze1_PYR']['computed_data']['pf1D']
    long_peaks = [pf1d_long.xbin_centers[np.argmax(x)] for x in pf1d_long.ratemap.tuning_curves] # CONCERN: these correspond to different neurons between the short and long peaks, right?
    long_df = pd.DataFrame(long_peaks, columns=['long_pf_peak_x'], index=pf1d_long.cell_ids) # nevermind, this is okay because we're using the correct cell_ids to build the dataframe
    long_df['has_long_pf'] = True

    # pf1d_short = computation_results['maze2_PYR']['computed_data']['pf1D']
    short_peaks = [pf1d_short.xbin_centers[np.argmax(x)] for x in pf1d_short.ratemap.tuning_curves] 
    short_df = pd.DataFrame(short_peaks, columns=['short_pf_peak_x'], index=pf1d_short.cell_ids)
    short_df['has_short_pf'] = True

    # df keeps most of the interesting data for these plots
    # at this point, it has columns 'long_pf_peak_x' and 'short_pf_peak_x' holding the peak tuning curve positions for each context
    # the index of this dataframe are the ACLU's for each neuron; this is why `how='outer'` works.
    df = long_df.join(short_df, how='outer')
    all_cell_ids = np.array(list(aclu_to_idx.keys()))
    missing_cell_id_mask = np.isin(all_cell_ids, df.index.to_numpy(), invert=True) # invert=True returns True for the things NOT in the existing aclus 
    missing_cell_ids = all_cell_ids[missing_cell_id_mask]
    neither_df = pd.DataFrame(index=missing_cell_ids)
    # Join on the neither df:
    df = df.join(neither_df, how='outer')

    df["has_na"] = df.isna().any(axis=1) # determines if any aclu are missing from either (long and short) ratemap
    # After the join the missing values are NaN instead of False. Fill them.
    df['has_short_pf'] = df['has_short_pf'].fillna(value=False)
    df['has_long_pf'] = df['has_long_pf'].fillna(value=False)

    # Add TrackMembershipMode
    df['track_membership'] = SplitPartitionMembership.SHARED
    df.loc[np.logical_and(df['has_short_pf'], np.logical_not(df['has_long_pf'])),'track_membership'] = SplitPartitionMembership.RIGHT_ONLY
    df.loc[np.logical_and(df['has_long_pf'], np.logical_not(df['has_short_pf'])),'track_membership'] = SplitPartitionMembership.LEFT_ONLY

    # calculations for ax[1,0] ___________________________________________________________________________________________ #
    
    non_replay_long_averages, non_replay_short_averages, non_replay_diff = take_difference_nonzero(irdf)
    replay_long_averages, replay_short_averages, replay_diff  = take_difference_nonzero(rdf)
        
    df["long_non_replay_mean"] = [non_replay_long_averages[aclu_to_idx[aclu]] for aclu in df.index]
    df["short_non_replay_mean"] = [non_replay_short_averages[aclu_to_idx[aclu]] for aclu in df.index]
    df["non_replay_diff"] = [non_replay_diff[aclu_to_idx[aclu]] for aclu in df.index]

    df["long_replay_mean"] = [replay_long_averages[aclu_to_idx[aclu]] for aclu in df.index]
    df["short_replay_mean"] = [replay_short_averages[aclu_to_idx[aclu]] for aclu in df.index]
    df["replay_diff"] = [replay_diff[aclu_to_idx[aclu]] for aclu in df.index]

    # Combined/Both Firing Rates:
    df['long_mean'] = (df['long_replay_mean'] + df['long_non_replay_mean'])/2.0
    df['short_mean'] = (df['short_replay_mean'] + df['short_non_replay_mean'])/2.0
    df['mean_diff'] = df['short_mean'] - df['long_mean']

    ## Compare the number of replay events between the long and the short
    

    return df


# Common _____________________________________________________________________________________________________________ #
def make_fr(rdf):
    """ extracts the firing_rates column from the dataframe and returns a numpy matrix 
        output_dict.shape # (116, 52) # (n_replays, n_neurons)
    """
    return np.vstack(rdf.firing_rates)

def add_spike_counts(sess, rdf):
    """ adds the spike counts vector to the dataframe """
    aclus = np.sort(sess.spikes_df.aclu.unique())
    aclu_to_idx = {aclus[i]:i for i in range(len(aclus))}

    spike_counts_list = []

    for index, row in rdf.iterrows():
        replay_spike_counts = np.zeros(sess.n_neurons)
        mask = (row["start"] < sess.spikes_df.t_rel_seconds) & (sess.spikes_df.t_rel_seconds < row["end"])
        for aclu in sess.spikes_df.loc[mask,"aclu"]:
            replay_spike_counts[aclu_to_idx[aclu]] += 1
        replay_spike_counts /= row["end"] - row["start"] # converts to a firing rate instead of a spike count
        
        if(np.isclose(replay_spike_counts.sum(), 0)):
            print(f"Time window {index} has no spikes." )

        spike_counts_list.append(replay_spike_counts)
    
    rdf = rdf.assign(firing_rates=spike_counts_list)
    return rdf, aclu_to_idx

# Make `rdf` (replay dataframe) ______________________________________________________________________________________ #
def make_rdf(sess, replays_df):
    """ recieves `replays_df`, but uses `sess.paradigm[1][0,0]` """
    rdf = replays_df.copy()[["start", "end"]]
    rdf["short_track"] = rdf["start"] > sess.paradigm[1][0,0]
    return rdf

def remove_nospike_replays(rdf):
    to_drop = rdf.index[make_fr(rdf).sum(axis=1)==0]
    rdf = rdf.drop(to_drop, axis=0)
    return rdf

def remove_low_p_replays(rdf):
    to_drop = rdf.index[rdf["replay_p"] > .1]
    rdf = rdf.drop(to_drop, axis=0)
    return rdf

# Make `irdf` (inter-replay dataframe) _______________________________________________________________________________ #
def make_irdf(sess, rdf):
    starts = [sess.paradigm[0][0,0]]
    ends = []
    for i, row in rdf.iterrows():
        ends.append(row.start)
        starts.append(row.end)
    ends.append(sess.paradigm[1][0,1])
    short_track = [s > sess.paradigm[1][0,0] for s in starts]
    return pd.DataFrame(dict(start=starts, end=ends, short_track=short_track))

def remove_repeated_replays(rdf):
    return rdf.drop_duplicates("start")

def take_difference(df):
    """this compares the average firing rate for each neuron before and after the context switch
    
    This function works on variables like `rdf` and `irdf`."""
    short_fr = make_fr(df[df["short_track"]])
    long_fr = make_fr(df[~df["short_track"]])   
    
    short_averages = np.zeros(short_fr.shape[1])
    for i in np.arange(short_fr.shape[1]):
        row = [x for x in short_fr[:,i] if x >= 0]
        short_averages[i] = np.mean(row)
        
    long_averages = np.zeros(long_fr.shape[1])
    for i in np.arange(long_fr.shape[1]):
        row = [x for x in long_fr[:,i] if x >= 0]
        long_averages[i] = np.mean(row)
        
    return long_averages, short_averages, (short_averages - long_averages)

def take_difference_nonzero(df):
    """this compares the average firing rate for each neuron before and after the context switch
    
    Note that this function compares the nonzero firing rates for each group; this is supposed to 
    correct for differences in participation."""
    
    short_fr = make_fr(df[df["short_track"]])
    long_fr = make_fr(df[~df["short_track"]])
    
    short_averages = np.zeros(short_fr.shape[1])
    for i in np.arange(short_fr.shape[1]):
        row = [x for x in short_fr[:,i] if x > 0] # NOTE: the difference from take_difference(df) seems to be only the `x > 0` instead of `x >= 0`
        short_averages[i] = np.mean(row)
        
    long_averages = np.zeros(long_fr.shape[1])
    for i in np.arange(long_fr.shape[1]):
        row = [x for x in long_fr[:,i] if x > 0] # NOTE: the difference from take_difference(df) seems to be only the `x > 0` instead of `x >= 0`
        long_averages[i] = np.mean(row)
        
    return long_averages, short_averages, (short_averages - long_averages)


# # note: this is defined here, but not used anywhere
# def take_difference_adjust_for_time(sess, df, aclu_to_idx):
#     """this compares the average firing rate for each neuron before and after the context switch
    
#     Note that this function corrects for the length of the intervals before averageing."""
#     short_time = 0
#     short_spikes = np.zeros(len(aclu_to_idx))
#     for index, row in df[df["short_track"]].iterrows():
#         short_time += row.end-row.start
#         t_slice = (row.start < sess.spikes_df.t_rel_seconds) & (sess.spikes_df.t_rel_seconds < row.end)
#         sdf = sess.spikes_df[t_slice]
#         for index, row in sdf.iterrows():
#             short_spikes[aclu_to_idx[row.aclu]] += 1
#     short_averages =short_spikes/ short_time
            
#     long_time = 0
#     long_spikes = np.zeros(len(aclu_to_idx))
#     for index, row in df[~df["short_track"]].iterrows():
#         long_time += row.end-row.start
#         t_slice = (row.start < sess.spikes_df.t_rel_seconds) & (sess.spikes_df.t_rel_seconds < row.end)
#         sdf = sess.spikes_df[t_slice]
#         for index, row in sdf.iterrows():
#             long_spikes[aclu_to_idx[row.aclu]] += 1
#     long_averages = long_spikes/ long_time
        
#     return short_averages  - long_averages



# Aggregate Stats ____________________________________________________________________________________________________ #
class FiringRateActivitySource(Enum):
    """Specifies which type of firing rate statistics should be used to determine sort and partition separations.
        Used as argument to `compute_evening_morning_parition(..., firing_rates_activity_source:FiringRateActivitySource=FiringRateActivitySource.ONLY_REPLAY, ...)`
    """
    BOTH = "BOTH" # uses both replay and non-replay firing rate means
    ONLY_REPLAY = "ONLY_REPLAY" # uses only replay firing rate means
    ONLY_NONREPLAY = "ONLY_NONREPLAY" # uses only non-replay firing rate means
    
    @classmethod
    def get_column_names_dict_list(cls):
        _tmp_active_column_names_list = [{'long':'long_mean', 'short':'short_mean', 'diff':'mean_diff'}, {'long':'long_replay_mean', 'short':'short_replay_mean', 'diff':'replay_diff'}, {'long':'long_non_replay_mean', 'short':'short_non_replay_mean', 'diff':'non_replay_diff'}]
        return {a_type.name:a_dict for a_type, a_dict in zip(list(cls), _tmp_active_column_names_list)}

    @property
    def active_column_names(self):
        """The active_column_names property."""
        return self.__class__.get_column_names_dict_list()[self.name]

@dataclass
class SortOrderMetric(object):
    """Holds return values of the same from from `compute_evening_morning_parition(...)`
    """
    sort_idxs: np.ndarray
    sorted_aclus: np.ndarray
    sorted_column_values: np.ndarray


def _compute_modern_aggregate_short_long_replay_stats(rdf, debug_print=True):
    """ Computes measures across all epochs in rdf: such as the average number of replays in each epoch (long v short) and etc
    Usage:
        (diff_total_num_replays, diff_total_replay_duration, diff_mean_replay_duration, diff_var_replay_duration), (long_total_num_replays, long_total_replay_duration, long_mean_replay_duration, long_var_replay_duration), (short_total_num_replays, short_total_replay_duration, short_mean_replay_duration, short_var_replay_duration) = _compute_modern_aggregate_short_long_replay_stats(rdf)
        print(f'diff_total_num_replays: {diff_total_num_replays}, diff_replay_duration: (total: {diff_total_replay_duration}, mean: {diff_mean_replay_duration}, var: {diff_var_replay_duration})')
    """
    (long_total_replay_duration, long_mean_replay_duration, long_var_replay_duration), (short_total_replay_duration, short_mean_replay_duration, short_var_replay_duration) = rdf.groupby("short_track")['duration'].agg(['sum','mean','var']).to_numpy() #.count()
    # long_total_replay_duration, short_total_replay_duration = rdf.groupby("short_track")['duration'].agg(['sum']).to_numpy() #.count()
    # print(f'long_total_replay_duration: {long_total_replay_duration}, short_total_replay_duration: {short_total_replay_duration}')
    if debug_print:
        print(f'long_replay_duration: (total: {long_total_replay_duration}, mean: {long_mean_replay_duration}, var: {long_var_replay_duration}), short_replay_duration: (total: {short_total_replay_duration}, mean: {short_mean_replay_duration}, var: {short_var_replay_duration})')
    long_total_num_replays, short_total_num_replays = rdf.groupby(by=["short_track"])['start'].agg('count').to_numpy() # array([392, 353], dtype=int64)
    if debug_print:
        print(f'long_total_num_replays: {long_total_num_replays}, short_total_num_replays: {short_total_num_replays}')
    # Differences
    diff_total_num_replays, diff_total_replay_duration, diff_mean_replay_duration, diff_var_replay_duration = (short_total_num_replays-long_total_num_replays), (short_total_replay_duration-long_total_replay_duration), (short_mean_replay_duration-long_mean_replay_duration), (short_var_replay_duration-long_var_replay_duration)

    if debug_print:
        print(f'diff_total_num_replays: {diff_total_num_replays}, diff_replay_duration: (total: {diff_total_replay_duration}, mean: {diff_mean_replay_duration}, var: {diff_var_replay_duration})')

    return (diff_total_num_replays, diff_total_replay_duration, diff_mean_replay_duration, diff_var_replay_duration), (long_total_num_replays, long_total_replay_duration, long_mean_replay_duration, long_var_replay_duration), (short_total_num_replays, short_total_replay_duration, short_mean_replay_duration, short_var_replay_duration)

def _compute_neuron_replay_stats(rdf, aclu_to_idx):
    """ Computes measures regarding replays across all neurons: such as the number of replays a neuron is involved in, etc 

    Usage:
        out_replay_df, out_neuron_df = _compute_neuron_replay_stats(rdf, aclu_to_idx)
        out_neuron_df

    """
    # Find the total number of replays each neuron is active during:

    # could assert np.shape(list(aclu_to_idx.keys())) # (52,) is equal to n_neurons
    # def _subfn_compute_epoch_neuron_replay_stats(epoch_rdf, aclu_to_idx):
    def _subfn_compute_epoch_neuron_replay_stats(epoch_rdf):
        # Extract the firing rates into a flat matrix instead
        flat_matrix = np.vstack(epoch_rdf.firing_rates)
        # flat_matrix.shape # (116, 52) # (n_replays, n_neurons)
        # n_replays = np.shape(flat_matrix)[0]
        n_neurons = np.shape(flat_matrix)[1]
        is_inactive_mask = np.isclose(flat_matrix, 0.0)
        is_active_mask = np.logical_not(is_inactive_mask)

        ## Number of unique replays each neuron participates in:
        neuron_num_active_replays = np.sum(is_active_mask, axis=0)
        assert (neuron_num_active_replays.shape[0] == n_neurons) # neuron_num_active_replays.shape # (52,) # (n_neurons,)
        return neuron_num_active_replays
        # # build output dataframes:
        # return pd.DataFrame({'aclu': aclu_to_idx.keys(), 'neuron_IDX': aclu_to_idx.values(), 'num_replays': neuron_num_active_replays}).set_index('aclu')
    
    ## Begin function body:
    grouped_rdf = rdf.groupby(by=["short_track"])
    long_rdf = grouped_rdf.get_group(False)
    # long_neuron_df = _subfn_compute_epoch_neuron_replay_stats(long_rdf, aclu_to_idx)
    long_neuron_num_active_replays = _subfn_compute_epoch_neuron_replay_stats(long_rdf)
    short_rdf = grouped_rdf.get_group(True)
    # short_neuron_df = _subfn_compute_epoch_neuron_replay_stats(short_rdf, aclu_to_idx)
    short_neuron_num_active_replays = _subfn_compute_epoch_neuron_replay_stats(short_rdf)

    # build output dataframes:
    out_neuron_df = pd.DataFrame({'aclu': aclu_to_idx.keys(), 'neuron_IDX': aclu_to_idx.values(), 'num_replays': (long_neuron_num_active_replays+short_neuron_num_active_replays), 'long_num_replays': long_neuron_num_active_replays, 'short_num_replays': short_neuron_num_active_replays}).set_index('aclu')

    ## Both:
    # Extract the firing rates into a flat matrix instead
    flat_matrix = np.vstack(rdf.firing_rates) # flat_matrix.shape # (116, 52) # (n_replays, n_neurons)
    n_replays = np.shape(flat_matrix)[0]
    # n_neurons = np.shape(flat_matrix)[1]
    is_inactive_mask = np.isclose(flat_matrix, 0.0)
    is_active_mask = np.logical_not(is_inactive_mask)
    ## Number of unique neurons participating in each replay:    
    replay_num_neuron_participating = np.sum(is_active_mask, axis=1)
    assert (replay_num_neuron_participating.shape[0] == n_replays) # num_active_replays.shape # (52,) # (n_neurons,)
    
    out_replay_df = rdf.copy()
    out_replay_df['num_neuron_participating'] = replay_num_neuron_participating
                 
    return out_replay_df, out_neuron_df


def compute_evening_morning_parition(neuron_replay_stats_df, firing_rates_activity_source:FiringRateActivitySource=FiringRateActivitySource.ONLY_REPLAY, debug_print=True):
    """ 2022-11-27 - Computes the cells that are either appearing or disappearing across the transition from the long to short track.
    
    Goal: Detect the cells that either appear or disappear across the transition from the long-to-short track
    
    
    Usage:
        difference_sorted_aclus, evening_sorted_aclus, morning_sorted_aclus = compute_evening_morning_parition(neuron_replay_stats_df, debug_print=True)
        sorted_neuron_replay_stats_df = neuron_replay_stats_df.reindex(difference_sorted_aclus).copy() # This seems to work to re-sort the dataframe by the sort indicies
        sorted_neuron_replay_stats_df
        
    difference_sorted_aclus: [        nan         nan  4.26399584  3.84391289  3.2983088   3.26820908
      2.75093881  2.32313925  2.28524202  2.24443817  1.92526386  1.87876877
      1.71554535  1.48531487  1.18602994  1.04168718  0.81165515  0.7807097
      0.59763511  0.5509481   0.54756479  0.50568564  0.41716005  0.37976643
      0.37645228  0.26027113  0.21105209  0.12519103  0.10830269 -0.03520479
     -0.04286447 -0.15702646 -0.17816494 -0.29196706 -0.31561772 -0.31763809
     -0.32949624 -0.38297539 -0.38715584 -0.40302644 -0.44631645 -0.45664655
     -0.47779662 -0.48631874 -0.60326742 -0.61542106 -0.68274119 -0.69134462
     -0.70242751 -0.7262794  -0.74993767 -0.79563808 -0.83345136 -1.02494536
     -1.0809595  -1.09055803 -1.12411968 -1.27320071 -1.28961086 -1.3305737
     -1.48966833 -1.87966732 -2.04939727 -2.24369668 -2.42700786 -2.59375268
     -2.62661755 -3.06693382 -4.56042725]
     For difference sorted values (difference_sorted_aclus), the first values in the array are likely to be short-specific while the last values are likely to be long-specific
    """
    # active_column_names = {'long':'long_mean', 'short':'short_mean', 'diff':'mean_diff'} # uses both replay and non-replay firing rate means
    # active_column_names = {'long':'long_replay_mean', 'short':'short_replay_mean', 'diff':'replay_diff'} # uses only replay firing rate means
    # active_column_names = {'long':'long_non_replay_mean', 'short':'short_non_replay_mean', 'diff':'non_replay_diff'} # uses only non-replay firing rate means
    active_column_names = firing_rates_activity_source.active_column_names
    out_dict = {}

    # Find "Evening" Cells: which have almost no activity in the 'long' epoch
    curr_long_mean_abs = neuron_replay_stats_df[active_column_names['long']].abs().to_numpy()
    long_nearest_zero_sort_idxs = np.argsort(curr_long_mean_abs)
    evening_sorted_aclus = neuron_replay_stats_df.index.to_numpy()[long_nearest_zero_sort_idxs] # find cells nearest to zero firing for long_mean
    out_dict['evening'] = SortOrderMetric(long_nearest_zero_sort_idxs, evening_sorted_aclus, curr_long_mean_abs[long_nearest_zero_sort_idxs])
    if debug_print:
        print(f'Evening sorted values: {curr_long_mean_abs[long_nearest_zero_sort_idxs]}')
    
    ## Find "Morning" Cells: which have almost no activity in the 'short' epoch
    curr_short_mean_abs = neuron_replay_stats_df[active_column_names['short']].abs().to_numpy()
    short_nearest_zero_sort_idxs = np.argsort(curr_short_mean_abs)
    morning_sorted_aclus = neuron_replay_stats_df.index.to_numpy()[short_nearest_zero_sort_idxs] # find cells nearest to zero firing for short_mean
    out_dict['morning'] = SortOrderMetric(short_nearest_zero_sort_idxs, morning_sorted_aclus, curr_short_mean_abs[short_nearest_zero_sort_idxs])
    if debug_print:
        print(f'Morning sorted values: {curr_short_mean_abs[short_nearest_zero_sort_idxs]}')
    
    # Look at differences method:
    curr_mean_diff = neuron_replay_stats_df[active_column_names['diff']].to_numpy()
    biggest_differences_sort_idxs = np.argsort(curr_mean_diff)[::-1] # sort this one in order of increasing values (most promising differences first)
    difference_sorted_aclus = neuron_replay_stats_df.index.to_numpy()[biggest_differences_sort_idxs]
    out_dict['diff'] = SortOrderMetric(biggest_differences_sort_idxs, difference_sorted_aclus, curr_mean_diff[biggest_differences_sort_idxs])
    # for the difference sorted method, the aclus at both ends of the `difference_sorted_aclus` are more likely to belong to morning/evening respectively
    if debug_print:
        print(f'Difference sorted values: {curr_mean_diff[biggest_differences_sort_idxs]}')
    # return (difference_sorted_aclus, evening_sorted_aclus, morning_sorted_aclus)
    return out_dict





# ==================================================================================================================== #
# Overlap                                                                                                      #
# ==================================================================================================================== #
def extrapolate_short_curve_to_long(long_xbins, short_xbins, short_curve, debug_print=False):
    """ extrapolate the short curve so that it is aligned with long_curve
        
    Usage:
        extrapolated_short_xbins, extrapolated_short_curve = extrapolate_short_curve_to_long(long_xbins, short_xbins, short_curve, debug_print=False)

    Known Uses:
        compute_dot_product_overlap, 
    """
    extrapolated_short_curve = np.interp(long_xbins, short_xbins, short_curve, left=0.0, right=0.0)
    return long_xbins, extrapolated_short_curve



# Polygon Overlap ____________________________________________________________________________________________________ #
def compute_polygon_overlap(long_results, short_results, debug_print=False):
    """ computes the overlap between 1D placefields for all units
    If the placefield is unique to one of the two epochs, a value of zero is returned for the overlap.
    """
    def _subfcn_compute_single_unit_polygon_overlap(avg_coords, model_coords, debug_print=False):
        polygon_points = [] #creates a empty list where we will append the points to create the polygon

        for xyvalue in avg_coords:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

        for xyvalue in model_coords[::-1]:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

        for xyvalue in avg_coords[0:1]:
            polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

        avg_poly = [] 
        model_poly = []

        for xyvalue in avg_coords:
            avg_poly.append([xyvalue[0],xyvalue[1]]) 

        for xyvalue in model_coords:
            model_poly.append([xyvalue[0],xyvalue[1]]) 


        line_non_simple = LineString(polygon_points)
        mls = unary_union(line_non_simple)

        Area_cal =[]

        for polygon in polygonize(mls):
            Area_cal.append(polygon.area)
            if debug_print:
                print(polygon.area)# print area of each section 
            Area_poly = (np.asarray(Area_cal).sum())
        if debug_print:
            print(Area_poly)#print combined area
        return Area_poly

    # get shared neuron info:
    pf_neurons_diff = _compare_computation_results(long_results.pf1D.ratemap.neuron_ids, short_results.pf1D.ratemap.neuron_ids)
    curr_any_context_neurons = pf_neurons_diff.either
    n_neurons = pf_neurons_diff.shared.n_neurons
    shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs
    
    short_xbins = short_results.pf1D.xbin_centers # .shape # (40,)
    # short_curves = short_results.pf1D.ratemap.tuning_curves # .shape # (64, 40)
    short_curves = short_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 40)

    long_xbins = long_results.pf1D.xbin_centers # .shape # (63,)
    # long_curves = long_results.pf1D.ratemap.tuning_curves # .shape # (64, 63)
    long_curves = long_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 63)

    pf_overlap_polys = []
    for i, a_pair in enumerate(pf_neurons_diff.shared.pairs):
        long_idx, short_idx = a_pair
        if long_idx is None or short_idx is None:
            # missing entry, answer is zero
            overlap_poly = 0
        else:        
            long_coords = list(zip(long_xbins, long_curves[long_idx]))
            short_coords = list(zip(short_xbins, short_curves[short_idx]))
            overlap_poly = _subfcn_compute_single_unit_polygon_overlap(short_coords, long_coords)
        pf_overlap_polys.append(overlap_poly)

    # np.array(pf_overlap_polys).shape # (69,)
    # return pf_overlap_polys
    overlap_df = pd.DataFrame(dict(aclu=curr_any_context_neurons, fragile_linear_IDX=shared_fragile_neuron_IDXs, poly_overlap=pf_overlap_polys)).set_index('aclu')
    return overlap_df

# Convolution Overlap ________________________________________________________________________________________________ #
def compute_convolution_overlap(long_results, short_results, debug_print=False):
    """ computes the overlap between 1D placefields for all units
    If the placefield is unique to one of the two epochs, a value of zero is returned for the overlap.
    """
    def _subfcn_compute_single_unit_convolution_overlap(long_xbins, long_curve, short_xbins, short_curve, debug_print=False):
        ### Convolve
        convolved_result_full = convolve(long_curve, short_curve, mode='full') # .shape # (102,)
        ### Define time of convolved data
        # here we'll uses t=long_results
        x_long = long_xbins.copy()
        x_short = short_xbins.copy()
        x_full = np.linspace(x_long[0]+x_short[0],x_long[-1]+x_short[-1],len(convolved_result_full)) # .shape # (102,)
        # t_same = t

        ### Compute the restricted bounds of the output so that it matches the long input function:
        istart = (np.abs(x_full-x_long[0])).argmin()
        iend = (np.abs(x_full-x_long[-1])).argmin()+1
        x_subset = x_full[istart:iend] # .shape # (63,)
        convolved_result_subset = convolved_result_full[istart:iend] # .shape # (63,)

        ### Normalize the discrete convolutions
        convolved_result_area = np.trapz(convolved_result_full, x=x_full)
        normalized_convolved_result_full = convolved_result_full / convolved_result_area
        
        convolved_result_subset_area = np.trapz(convolved_result_subset, x=x_subset)
        normalized_convolved_result_subset = convolved_result_subset / convolved_result_subset_area

        return dict(full=dict(x=x_full, convolved_result=convolved_result_full, normalized_convolved_result=normalized_convolved_result_full, area=convolved_result_area),
            valid_subset=dict(x=x_subset, convolved_result=convolved_result_subset, normalized_convolved_result=normalized_convolved_result_subset, area=convolved_result_subset_area))

    # get shared neuron info:
    pf_neurons_diff = _compare_computation_results(long_results.pf1D.ratemap.neuron_ids, short_results.pf1D.ratemap.neuron_ids)
    curr_any_context_neurons = pf_neurons_diff.either
    n_neurons = pf_neurons_diff.shared.n_neurons
    shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs
    
    short_xbins = short_results.pf1D.xbin_centers # .shape # (40,)
    # short_curves = short_results.pf1D.ratemap.tuning_curves # .shape # (64, 40)
    short_curves = short_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 40)

    long_xbins = long_results.pf1D.xbin_centers # .shape # (63,)
    # long_curves = long_results.pf1D.ratemap.tuning_curves # .shape # (64, 63)
    long_curves = long_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 63)

    pf_overlap_conv_results = []
    for i, a_pair in enumerate(pf_neurons_diff.shared.pairs):
        long_idx, short_idx = a_pair
        if long_idx is None or short_idx is None:
            # missing entry, answer is zero
            overlap_results_dict = None
        else:        
            # long_coords = list(zip(long_xbins, long_curves[long_idx]))
            # short_coords = list(zip(short_xbins, short_curves[short_idx]))
            long_curve = long_curves[long_idx]
            short_curve = short_curves[short_idx]
            overlap_results_dict = _subfcn_compute_single_unit_convolution_overlap(long_xbins, long_curve, short_xbins, short_curve)
        pf_overlap_conv_results.append(overlap_results_dict)

    overlap_dict = {aclu:pf_overlap_conv_results[i] for i, aclu in enumerate(curr_any_context_neurons)}
    # print(f"{[pf_overlap_conv_results[i] for i, aclu in enumerate(curr_any_context_neurons)]}")
    # print(f"{[(pf_overlap_conv_results[i] or {}).get('full', {}).get('area', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]}")    
    overlap_areas = [(pf_overlap_conv_results[i] or {}).get('full', {}).get('area', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]
    overlap_scalars_df = pd.DataFrame(dict(aclu=curr_any_context_neurons, fragile_linear_IDX=shared_fragile_neuron_IDXs, conv_overlap=overlap_areas)).set_index('aclu')

    return overlap_dict, overlap_scalars_df

# Product Overlap ____________________________________________________________________________________________________ #
def compute_dot_product_overlap(long_results, short_results, debug_print=False):
    """ computes the overlap between 1D placefields for all units
    If the placefield is unique to one of the two epochs, a value of zero is returned for the overlap.
    """
    def _subfcn_compute_single_unit_dot_product_overlap(long_xbins, long_curve, short_xbins, short_curve, debug_print=False):
        # extrapolate the short curve so that it is aligned with long_curve
        if len(long_xbins) > len(short_xbins):
            # Need to interpolate:
            extrapolated_short_xbins, extrapolated_short_curve = extrapolate_short_curve_to_long(long_xbins, short_xbins, short_curve, debug_print=False)
        else:
            # They are already using the same xbins:
            extrapolated_short_curve = short_curve

        # extrapolated_short_curve = np.interp(long_xbins, short_xbins, short_curve, left=0.0, right=0.0)
        pf_overlap_dot_product_curve = extrapolated_short_curve * long_curve

        overlap_dot_product_maximum = np.nanmax(pf_overlap_dot_product_curve)

        ### Normalize the discrete convolutions
        overlap_area = np.trapz(pf_overlap_dot_product_curve, x=long_xbins)
        normalized_overlap_dot_product = pf_overlap_dot_product_curve / overlap_area

        return dict(x=long_xbins, overlap_dot_product=pf_overlap_dot_product_curve, normalized_overlap_dot_product=normalized_overlap_dot_product, area=overlap_area, peak_max=overlap_dot_product_maximum, extrapolated_short_curve=extrapolated_short_curve)

    # get shared neuron info:
    pf_neurons_diff = _compare_computation_results(long_results.pf1D.ratemap.neuron_ids, short_results.pf1D.ratemap.neuron_ids)
    curr_any_context_neurons = pf_neurons_diff.either
    n_neurons = pf_neurons_diff.shared.n_neurons
    shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs
    
    short_xbins = short_results.pf1D.xbin_centers # .shape # (40,)
    # short_curves = short_results.pf1D.ratemap.tuning_curves # .shape # (64, 40)
    short_curves = short_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 40)

    long_xbins = long_results.pf1D.xbin_centers # .shape # (63,)
    # long_curves = long_results.pf1D.ratemap.tuning_curves # .shape # (64, 63)
    long_curves = long_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 63)

    pf_overlap_results = []
    for i, a_pair in enumerate(pf_neurons_diff.shared.pairs):
        long_idx, short_idx = a_pair
        if long_idx is None or short_idx is None:
            # missing entry, answer is zero
            overlap_results_dict = None
        else:        
            long_curve = long_curves[long_idx]
            short_curve = short_curves[short_idx]
            overlap_results_dict = _subfcn_compute_single_unit_dot_product_overlap(long_xbins, long_curve, short_xbins, short_curve)
        pf_overlap_results.append(overlap_results_dict)

    overlap_dict = {aclu:pf_overlap_results[i] for i, aclu in enumerate(curr_any_context_neurons)}
    prod_overlap_areas = [(pf_overlap_results[i] or {}).get('area', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]
    prod_overlap_peak_max = [(pf_overlap_results[i] or {}).get('peak_max', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]
    overlap_scalars_df = pd.DataFrame(dict(aclu=curr_any_context_neurons, fragile_linear_IDX=shared_fragile_neuron_IDXs, prod_overlap=prod_overlap_areas, prod_overlap_peak_max=prod_overlap_peak_max)).set_index('aclu')

    return overlap_dict, overlap_scalars_df


# Relative Entropy Divergence Overlap ____________________________________________________________________________________________________ #
def compute_relative_entropy_divergence_overlap(long_results, short_results, debug_print=False):
    """ computes the Compute the relative entropy (KL-Divergence) between each pair of tuning curves between {long, global} (in both directions) 1D placefields for all units
    If the placefield is unique to one of the two epochs, a value of zero is returned for the overlap.
    """
    def _subfcn_compute_single_unit_relative_entropy_divergence_overlap(long_xbins, long_curve, short_xbins, short_curve, debug_print=False):
        # extrapolate the short curve so that it is aligned with long_curve
        if len(long_xbins) > len(short_xbins):
            # Need to interpolate:
            extrapolated_short_xbins, extrapolated_short_curve = extrapolate_short_curve_to_long(long_xbins, short_xbins, short_curve, debug_print=debug_print)
        else:
            # They are already using the same xbins:
            extrapolated_short_curve = short_curve
    
        long_short_rel_entr_curve = rel_entr(long_curve, extrapolated_short_curve)
        long_short_relative_entropy = sum(long_short_rel_entr_curve) 

        short_long_rel_entr_curve = rel_entr(extrapolated_short_curve, long_curve)
        short_long_relative_entropy = sum(short_long_rel_entr_curve) 

        return dict(long_short_rel_entr_curve=long_short_rel_entr_curve, long_short_relative_entropy=long_short_relative_entropy, short_long_rel_entr_curve=short_long_rel_entr_curve, short_long_relative_entropy=short_long_relative_entropy, extrapolated_short_curve=extrapolated_short_curve)

    # get shared neuron info:
    pf_neurons_diff = _compare_computation_results(long_results.pf1D.ratemap.neuron_ids, short_results.pf1D.ratemap.neuron_ids)
    curr_any_context_neurons = pf_neurons_diff.either
    n_neurons = pf_neurons_diff.shared.n_neurons
    shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs

    short_xbins = short_results.pf1D.xbin_centers # .shape # (40,)
    # short_curves = short_results.pf1D.ratemap.tuning_curves # .shape # (64, 40)
    short_curves = short_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 40)

    long_xbins = long_results.pf1D.xbin_centers # .shape # (63,)
    # long_curves = long_results.pf1D.ratemap.tuning_curves # .shape # (64, 63)
    long_curves = long_results.pf1D.ratemap.normalized_tuning_curves # .shape # (64, 63)

    pf_overlap_results = []
    for i, a_pair in enumerate(pf_neurons_diff.shared.pairs):
        long_idx, short_idx = a_pair
        if long_idx is None or short_idx is None:
            # missing entry, answer is zero
            overlap_results_dict = None
        else:        
            long_curve = long_curves[long_idx]
            short_curve = short_curves[short_idx]
            overlap_results_dict = _subfcn_compute_single_unit_relative_entropy_divergence_overlap(long_xbins, long_curve, short_xbins, short_curve)
        pf_overlap_results.append(overlap_results_dict)

    overlap_dict = {aclu:pf_overlap_results[i] for i, aclu in enumerate(curr_any_context_neurons)}
    long_short_relative_entropy = [(pf_overlap_results[i] or {}).get('long_short_relative_entropy', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]
    short_long_relative_entropy = [(pf_overlap_results[i] or {}).get('short_long_relative_entropy', 0.0) for i, aclu in enumerate(curr_any_context_neurons)]
    overlap_scalars_df = pd.DataFrame(dict(aclu=curr_any_context_neurons, fragile_linear_IDX=shared_fragile_neuron_IDXs, long_short_relative_entropy=long_short_relative_entropy, short_long_relative_entropy=short_long_relative_entropy)).set_index('aclu')

    return overlap_dict, overlap_scalars_df



# ==================================================================================================================== #
# PBE Stats                                                                                                            #
# ==================================================================================================================== #
def _perform_PBE_stats(owning_pipeline_reference, include_whitelist=None, debug_print = False):
    """ # Analyze PBEs by looping through the filtered epochs:
        This whole implementation seems silly and inefficient        
        Can't I use .agg(['count', 'mean']) or something? 
        
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _perform_PBE_stats
        pbe_analyses_result_df, [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] = _perform_PBE_stats(curr_active_pipeline, debug_print=False) # all_epochs_n_pbes: [206, 31, 237], all_epochs_mean_pbe_durations: [0.2209951456310722, 0.23900000000001073, 0.22335021097046923], all_epochs_cummulative_pbe_durations: [45.52500000000087, 7.409000000000333, 52.934000000001205], all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, 1910.1600048116618]
        pbe_analyses_result_df

    """
    if include_whitelist is None:
        include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

    all_epochs_labels = []
    all_epochs_total_durations = []
    all_epochs_n_pbes = []
    all_epochs_pbe_duration_lists = []
    all_epochs_cummulative_pbe_durations = []
    all_epochs_mean_pbe_durations = []
    all_epochs_full_pbe_spiketrain_lists = []
    all_epochs_pbe_num_spikes_lists = []
    all_epochs_intra_pbe_interval_lists = []
    
    for (name, filtered_sess) in owning_pipeline_reference.filtered_sessions.items():
        if name in include_whitelist:
            # interested in analyzing both the filtered_sess.pbe and the filtered_sess.spikes_df (as they relate to the PBEs)
            all_epochs_labels.append(name)
            curr_named_time_range = owning_pipeline_reference.sess.epochs.get_named_timerange(name) # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
            
            if not np.isscalar(curr_named_time_range.duration):
                # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
                curr_named_time_range = NamedTimerange(name='maze', start_end_times=[owning_pipeline_reference.sess.epochs['maze1'][0], owning_pipeline_reference.sess.epochs['maze2'][1]])
            
            curr_epoch_duration = curr_named_time_range.duration
            all_epochs_total_durations.append(curr_epoch_duration) # TODO: this should be in seconds (or at least the same units as the PBE durations)... actually this might be right.
            # Computes the intervals between each PBE:
            curr_intra_pbe_intervals = filtered_sess.pbe.starts[1:] - filtered_sess.pbe.stops[:-1]
            all_epochs_intra_pbe_interval_lists.append(curr_intra_pbe_intervals)
            all_epochs_n_pbes.append(filtered_sess.pbe.n_epochs)
            all_epochs_pbe_duration_lists.append(filtered_sess.pbe.durations)
            all_epochs_cummulative_pbe_durations.append(np.sum(filtered_sess.pbe.durations))
            all_epochs_mean_pbe_durations.append(np.nanmean(filtered_sess.pbe.durations))
            # filtered_sess.spikes_df.PBE_id
            curr_pbe_only_spikes_df = filtered_sess.spikes_df[filtered_sess.spikes_df.PBE_id > -1].copy()
            unique_PBE_ids = np.unique(curr_pbe_only_spikes_df['PBE_id'])
            flat_PBE_ids = [int(id) for id in unique_PBE_ids]
            num_unique_PBE_ids = len(flat_PBE_ids)
            # groups the spikes_df by PBEs:
            curr_pbe_grouped_spikes_df = curr_pbe_only_spikes_df.groupby(['PBE_id'])
            curr_spiketrains = list()
            curr_PBE_spiketrain_num_spikes = list()
            for i in np.arange(num_unique_PBE_ids):
                curr_PBE_id = flat_PBE_ids[i] # actual cell ID
                #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
                curr_PBE_dataframe = curr_pbe_grouped_spikes_df.get_group(curr_PBE_id)
                curr_PBE_num_spikes = np.shape(curr_PBE_dataframe)[0] # the number of spikes in this PBE
                curr_PBE_spiketrain_num_spikes.append(curr_PBE_num_spikes)
                curr_spiketrains.append(curr_PBE_dataframe['t'].to_numpy())

            curr_PBE_spiketrain_num_spikes = np.array(curr_PBE_spiketrain_num_spikes)
            all_epochs_pbe_num_spikes_lists.append(curr_PBE_spiketrain_num_spikes)
            curr_spiketrains = np.array(curr_spiketrains, dtype='object')
            all_epochs_full_pbe_spiketrain_lists.append(curr_spiketrains)
            if debug_print:
                print(f'name: {name}, filtered_sess.pbe: {filtered_sess.pbe}')

    if debug_print:
        print(f'all_epochs_n_pbes: {all_epochs_n_pbes}, all_epochs_mean_pbe_durations: {all_epochs_mean_pbe_durations}, all_epochs_cummulative_pbe_durations: {all_epochs_cummulative_pbe_durations}, all_epochs_total_durations: {all_epochs_total_durations}')
        # all_epochs_n_pbes: [3152, 561, 1847, 832, 4566], all_epochs_mean_pbe_durations: [0.19560881979695527, 0.22129233511594312, 0.19185056848946497, 0.2333112980769119, 0.1987152869032212]

    all_epochs_pbe_occurance_rate = [(float(all_epochs_total_durations[i]) / float(all_epochs_n_pbes[i])) for i in np.arange(len(all_epochs_n_pbes))]
    all_epochs_pbe_percent_duration = [(float(all_epochs_total_durations[i]) / float(all_epochs_cummulative_pbe_durations[i])) for i in np.arange(len(all_epochs_n_pbes))]    
    all_epoch_mean_num_pbe_spikes = [np.nanmean(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [3151, 561, 1847, 831, 4563]
    all_epoch_std_num_pbe_spikes = [np.nanstd(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [11.638970035733648, 15.013817202645336, 15.5123897729991, 15.113395025612247, 11.473087401691878]
    # [20.429704855601397, 27.338680926916222, 23.748781808337846, 25.673886883273166, 20.38614946307254]
    # Build the final output result dataframe:
    pbe_analyses_result_df = pd.DataFrame({'n_pbes':all_epochs_n_pbes, 'mean_pbe_durations': all_epochs_mean_pbe_durations, 'cummulative_pbe_durations':all_epochs_cummulative_pbe_durations, 'epoch_total_duration':all_epochs_total_durations,
                'pbe_occurance_rate':all_epochs_pbe_occurance_rate, 'pbe_percent_duration':all_epochs_pbe_percent_duration,
                'mean_num_pbe_spikes':all_epoch_mean_num_pbe_spikes, 'stddev_num_pbe_spikes':all_epoch_std_num_pbe_spikes}, index=all_epochs_labels)
    # temporary: this isn't how the returns work for other computation functions:
    all_epochs_info = [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] # list version
    # all_epochs_info = {'all_epochs_full_pbe_spiketrain_lists':all_epochs_full_pbe_spiketrain_lists, 'all_epochs_pbe_num_spikes_lists':all_epochs_pbe_num_spikes_lists, 'all_epochs_intra_pbe_interval_lists':all_epochs_intra_pbe_interval_lists} # dict version
    return pbe_analyses_result_df, all_epochs_info


