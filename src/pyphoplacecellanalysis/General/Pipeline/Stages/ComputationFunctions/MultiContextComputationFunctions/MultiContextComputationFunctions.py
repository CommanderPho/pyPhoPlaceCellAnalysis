import numpy as np
import pandas as pd

from neuropy.utils.misc import safe_pandas_get_group # for _compute_pybursts_burst_interval_detection


from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for _perform_firing_rate_trends_computation


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
        
        """
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']


        # ## Compute for all the session spikes first:
        sess = owning_pipeline_reference.sess

        ### Make `rdf` (replay dataframe)
        rdf = make_rdf(sess) # this creates the replay dataframe variable
        rdf = remove_repeated_replays(rdf)
        rdf, aclu_to_idx = add_spike_counts(sess, rdf)

        rdf = remove_nospike_replays(rdf)
        print(f"RDF has {len(rdf)} rows.")

        ### Make `irdf` (inter-replay dataframe)
        irdf = make_irdf(sess, rdf)
        irdf = remove_repeated_replays(irdf) # TODO: make the removal process more meaningful
        irdf, aclu_to_idx_irdf = add_spike_counts(sess, irdf)

        assert aclu_to_idx_irdf == aclu_to_idx # technically, these might not match, which would be bad

        global_computation_results.computed_data['jonathan_firing_rate_analysis'] = DynamicParameters.init_from_dict({
            'rdf': DynamicParameters.init_from_dict({
                'rdf': rdf,
                'aclu_to_idx': aclu_to_idx, 
            }),
            'irdf': DynamicParameters.init_from_dict({
                'irdf': irdf,
                'aclu_to_idx': aclu_to_idx_irdf,           
            }),
        })
        return global_computation_results



# ==================================================================================================================== #
# Jonathan's helper functions                                                                                          #
# ==================================================================================================================== #
def make_fr(rdf):
    return np.vstack(rdf.firing_rates)

def make_rdf(sess):
    rdf = sess.replay.copy()[["start", "end"]]
    rdf["short_track"] = rdf["start"] > sess.paradigm[1][0,0]
    return rdf

def add_spike_counts(sess, rdf):
    
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

def remove_nospike_replays(rdf):
    to_drop = np.where(make_fr(rdf).sum(axis=1)==0)[0]
    rdf = rdf.drop(to_drop, axis=0)
    return rdf

def remove_low_p_replays(rdf):
    to_drop = rdf.index[rdf["replay_p"] > .1]
    rdf = rdf.drop(to_drop, axis=0)
    return rdf

def remove_repeated_replays(rdf):
    return rdf.drop_duplicates("start")

def make_irdf(sess, rdf):
    starts = [sess.paradigm[0][0,0]]
    ends = []
    for i, row in rdf.iterrows():
        ends.append(row.start)
        starts.append(row.end)
    ends.append(sess.paradigm[1][0,1])
    short_track = [s > sess.paradigm[1][0,0] for s in starts]
    return pd.DataFrame(dict(start=starts, end=ends, short_track=short_track))

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


