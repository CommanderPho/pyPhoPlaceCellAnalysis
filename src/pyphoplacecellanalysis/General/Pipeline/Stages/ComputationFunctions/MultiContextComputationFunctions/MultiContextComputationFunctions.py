import numpy as np
import pandas as pd

from neuropy.utils.misc import safe_pandas_get_group # for _compute_pybursts_burst_interval_detection

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_firing_rate_trends_computation

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

                ['jonathan_firing_rate_analysis']['final_jonathan_df']
        
        """
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        # ## Compute for all the session spikes first:
        sess = owning_pipeline_reference.sess
        replays_df = sess.replay
        rdf, aclu_to_idx, irdf, aclu_to_idx_irdf = _final_compute_jonathan_replay_fr_analyses(sess, replays_df)

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
        pf1d = computation_results[global_epoch_name]['computed_data']['pf1D']

        try:
            # pf2D_Decoder = computation_results[global_epoch_name]['computed_data']['pf2D_Decoder']
            active_firing_rate_trends = computation_results[global_epoch_name]['computed_data']['firing_rate_trends']

            ## time_binned_unit_specific_binned_spike_rate mode:
            # time_bins = active_firing_rate_trends.all_session_spikes.time_binning_container.centers
            # time_binned_unit_specific_binned_spike_rate = active_firing_rate_trends.all_session_spikes.time_binned_unit_specific_binned_spike_rate

            ## instantaneous_unit_specific_spike_rate mode:
            neuron_IDs = np.unique(computation_results[global_epoch_name].sess.spikes_df.aclu)
            # neuron_IDXs = np.arange(len(neuron_IDs))
            instantaneous_unit_specific_spike_rate = active_firing_rate_trends.all_session_spikes.instantaneous_unit_specific_spike_rate
            # instantaneous_unit_specific_spike_rate = computation_results[global_epoch_name]['computed_data']['firing_rate_trends'].all_session_spikes.instantaneous_unit_specific_spike_rate
            instantaneous_unit_specific_spike_rate_values = pd.DataFrame(instantaneous_unit_specific_spike_rate.magnitude, columns=neuron_IDs) # builds a df with times along the rows and aclu values along the columns in the style of unit_specific_binned_spike_counts
            time_bins = instantaneous_unit_specific_spike_rate.times.magnitude # .shape (3429,)
            time_binned_unit_specific_binned_spike_rate = instantaneous_unit_specific_spike_rate_values # .shape (3429, 71)

        except KeyError:
            # except ValueError:
            # print(f'non-placefield neuron. Skipping.')
            time_bins, time_binned_unit_specific_binned_spike_rate = {}, {}

        final_jonathan_df = _subfn_computations_make_jonathan_firing_comparison_df(time_binned_unit_specific_binned_spike_rate, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf)

        global_computation_results.computed_data['jonathan_firing_rate_analysis'] = DynamicParameters.init_from_dict({
            'rdf': DynamicParameters.init_from_dict({
                'rdf': rdf,
                'aclu_to_idx': aclu_to_idx, 
            }),
            'irdf': DynamicParameters.init_from_dict({
                'irdf': irdf,
                'aclu_to_idx': aclu_to_idx_irdf,           
            }),
            'time_binned_unit_specific_spike_rate': DynamicParameters.init_from_dict({
                'time_bins': time_bins,
                'time_binned_unit_specific_binned_spike_rate': time_binned_unit_specific_binned_spike_rate,           
            }),
            'final_jonathan_df': final_jonathan_df
        })
        return global_computation_results


# ==================================================================================================================== #
# Jonathan's helper functions                                                                                          #
# ==================================================================================================================== #
def _final_compute_jonathan_replay_fr_analyses(sess, replays_df):
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
    print(f"RDF has {len(rdf)} rows.")

    ### Make `irdf` (inter-replay dataframe)
    irdf = make_irdf(sess, rdf)
    irdf = remove_repeated_replays(irdf) # TODO: make the removal process more meaningful
    irdf, aclu_to_idx_irdf = add_spike_counts(sess, irdf)

    assert aclu_to_idx_irdf == aclu_to_idx # technically, these might not match, which would be bad

    return rdf, aclu_to_idx, irdf, aclu_to_idx_irdf


def _subfn_computations_make_jonathan_firing_comparison_df(unit_specific_time_binned_firing_rates, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf):
    """ the computations that were factored out of _make_jonathan_interactive_plot(...) 
    Historical: used to be called `_subfn_computations_make_jonathan_interactive_plot(...)`
    """
    # ==================================================================================================================== #
    ## Calculating:

    ## The actual firing rate we want:
    
    # unit_specific_time_binned_firing_rates = pf2D_Decoder.unit_specific_time_binned_spike_counts.astype(np.float32) / pf2D_Decoder.time_bin_size
    print(f'np.shape(unit_specific_time_binned_firing_rates): {np.shape(unit_specific_time_binned_firing_rates)}')

    # calculations for ax[0,0] ___________________________________________________________________________________________ #
    # below we find where the tuning curve peak was for each cell in each context and store it in a dataframe
    # pf1d_long = computation_results['maze1_PYR']['computed_data']['pf1D']
    long_peaks = [pf1d_long.xbin_centers[np.argmax(x)] for x in pf1d_long.ratemap.tuning_curves]
    long_df = pd.DataFrame(long_peaks, columns=['long'], index=pf1d_long.cell_ids)

    # pf1d_short = computation_results['maze2_PYR']['computed_data']['pf1D']
    short_peaks = [pf1d_short.xbin_centers[np.argmax(x)] for x in pf1d_short.ratemap.tuning_curves]
    short_df = pd.DataFrame(short_peaks, columns=['short'],index=pf1d_short.cell_ids)

    # df keeps most of the interesting data for these plots
    # at this point, it has columns 'long' and 'short' holding the peak tuning curve positions for each context
    # the index of this dataframe are the ACLU's for each neuron; this is why `how='outer'` works.
    df = long_df.join(short_df, how='outer')
    df["has_na"] = df.isna().any(axis=1)

    # calculations for ax[1,0] ___________________________________________________________________________________________ #
    non_replay_diff = take_difference_nonzero(irdf)
    replay_diff = take_difference_nonzero(rdf)
    df["non_replay_diff"] = [non_replay_diff[aclu_to_idx[aclu]] for aclu in df.index]
    df["replay_diff"] = [replay_diff[aclu_to_idx[aclu]] for aclu in df.index]

    return df


# Common _____________________________________________________________________________________________________________ #
def make_fr(rdf):
    return np.vstack(rdf.firing_rates)

def add_spike_counts(sess, rdf):
    """ adds the spike counts vector to the dataframe """
    aclus = np.sort(sess.spikes_df.aclu.unique())
    aclu_to_idx = {aclus[i] : i for i in range(len(aclus))}

    spike_counts_list = []

    for index, row in rdf.iterrows():
        replay_spike_counts = np.zeros(sess.n_neurons)
        mask = (row["start"] < sess.spikes_df.t_rel_seconds) & (sess.spikes_df.t_rel_seconds < row["end"])
        for aclu in sess.spikes_df.loc[mask,"aclu"]:
            replay_spike_counts[aclu_to_idx[aclu]] += 1
        replay_spike_counts /= row["end"] - row["start"]
        
        if(np.isclose(replay_spike_counts.sum(), 0)):
            print(f"Time window {index} has no spikes." )

        spike_counts_list.append(replay_spike_counts)
    
    rdf = rdf.assign(firing_rates=spike_counts_list)
    return rdf, aclu_to_idx

# Make `rdf` (replay dataframe) ______________________________________________________________________________________ #
def make_rdf(sess, replays_df):
    """ uses the `sess.replay` property"""

    rdf = replays_df.copy()[["start", "end"]]
    rdf["short_track"] = rdf["start"] > sess.paradigm[1][0,0]
    return rdf

def remove_nospike_replays(rdf):
    to_drop = np.where(make_fr(rdf).sum(axis=1)==0)[0]
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
        row = [x for x in short_fr[:,i] if x >=0]
        short_averages[i] = np.mean(row)
        
    long_averages = np.zeros(long_fr.shape[1])
    for i in np.arange(long_fr.shape[1]):
        row = [x for x in long_fr[:,i] if x >= 0]
        long_averages[i] = np.mean(row)
        
    return short_averages  - long_averages


def take_difference_nonzero(df):
    """this compares the average firing rate for each neuron before and after the context switch
    
    Note that this function compares the nonzero firing rates for each group; this is supposed to 
    correct for differences in participation."""
    
    short_fr = make_fr(df[df["short_track"]])
    long_fr = make_fr(df[~df["short_track"]])   
    
    short_averages = np.zeros(short_fr.shape[1])
    for i in np.arange(short_fr.shape[1]):
        row = [x for x in short_fr[:,i] if x >0]
        short_averages[i] = np.mean(row)
        
    long_averages = np.zeros(long_fr.shape[1])
    for i in np.arange(long_fr.shape[1]):
        row = [x for x in long_fr[:,i] if x > 0]
        long_averages[i] = np.mean(row)
        
    return short_averages  - long_averages


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


