from copy import deepcopy

import numpy as np
import pandas as pd
from attrs import define, field # used for `JonathanFiringRateAnalysisResult`, `LongShortPipelineTests`
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from neuropy.analyses.placefields import PfND # used in `constrain_to_laps` to construct new objects

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_full_session_leave_one_out_decoding_analysis
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SetPartition
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

@define(slots=False, repr=False)
class LeaveOneOutDecodingAnalysis(ComputedResult):
    """ 2023-05-10 - holds the results of a leave-one-out decoding analysis of the long and short track 
    Usage:
        leave_one_out_decoding_analysis_obj = LeaveOneOutDecodingAnalysis(long_decoder, short_decoder, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj)
    """    
    long_decoder: BayesianPlacemapPositionDecoder
    short_decoder: BayesianPlacemapPositionDecoder
    long_replays: pd.DataFrame
    short_replays: pd.DataFrame
    global_replays: pd.DataFrame
    long_shared_aclus_only_decoder: BasePositionDecoder
    short_shared_aclus_only_decoder: BasePositionDecoder
    shared_aclus: np.ndarray
    long_short_pf_neurons_diff: SetPartition
    n_neurons: int
    long_results_obj: SurpriseAnalysisResult
    short_results_obj: SurpriseAnalysisResult

    is_global: bool = True

class LongShortTrackComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'long_short_track'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='_perform_long_short_decoding_analyses', tags=['long_short', 'short_long','replay', 'decoding', 'computation'], input_requires=[], output_provides=[], uses=['_long_short_decoding_analysis_from_decoders'], used_by=[], creation_date='2023-05-10 15:10')
    def _perform_long_short_decoding_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False, decoding_time_bin_size=None, perform_cache_load=False, always_recompute_replays=True):
        """ 
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['long_short_leave_one_out_decoding_analysis']
                # ['long_short_leave_one_out_decoding_analysis']['short_long_neurons_diff']
                # ['long_short_leave_one_out_decoding_analysis']['poly_overlap_df']
        
        """
        # # New unified `pipeline_complete_compute_long_short_fr_indicies(...)` method for entire pipeline:
        # x_frs_index, y_frs_index, active_context, all_results_dict = pipeline_complete_compute_long_short_fr_indicies(owning_pipeline_reference) # use the all_results_dict as the computed data value
        # global_computation_results.computed_data['long_short_fr_indicies_analysis'] = DynamicParameters.init_from_dict({**all_results_dict, 'active_context': active_context})

        # 2023-05-16 - Correctly initialized pipelines (pfs limited to laps, decoders already long/short constrainted by default, replays already the estimated versions:
        is_certain_properly_constrained = True


        if not is_certain_properly_constrained:
            owning_pipeline_reference = constrain_to_laps(owning_pipeline_reference) # Constrains placefields to laps
            
            (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_long_short_constrained_decoders(owning_pipeline_reference, recalculate_anyway=True)
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_results, short_results, global_results = [owning_pipeline_reference.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
            long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D

            # Compute/estimate replays if missing from session:
            if not global_session.has_replays or always_recompute_replays:
                if not global_session.has_replays:
                    print(f'Replays missing from sessions. Computing replays...')
                else:
                    print(f'Replays exist but `always_recompute_replays` is True, so estimate_replay_epochs will be performed and the old ones will be overwritten.')
                # Backup and replace loaded replays with computed ones:
                long_replays, short_replays, global_replays = [a_session.replace_session_replays_with_estimates(require_intersecting_epoch=None, debug_print=False) for a_session in [long_session, short_session, global_session]]

            # 3m 40.3s
        else:
            print(f'is_certain_properly_constrained: True - Correctly initialized pipelines (pfs limited to laps, decoders already long/short constrainted by default, replays already the estimated versions')
            if always_recompute_replays:
               print(f'\t is_certain_properly_constrained IGNORES always_recompute_replays!')
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_results, short_results, global_results = [owning_pipeline_reference.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_one_step_decoder_1D, short_one_step_decoder_1D  = [deepcopy(results_data.get('pf1D_Decoder', None)) for results_data in (long_results, short_results)]


        if decoding_time_bin_size is None:
            decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333
        else:
            # check if decoding_time_bin_size is the same
            if not (decoding_time_bin_size == long_one_step_decoder_1D.time_bin_size):
                print(f'decoding_time_bin_size different than decoder: decoding_time_bin_size: {decoding_time_bin_size}, long_one_step_decoder_1D.time_bin_size: {long_one_step_decoder_1D.time_bin_size}')
                raise NotImplementedError
                # TODO: invalidate cached
                perform_cache_load = False
                ## Update `long_one_step_decoder_1D.time_bin_size` to the new size? TODO 2023-05-10 - redo computations with this size for `long_one_step_decoder_1D`?
                long_one_step_decoder_1D.time_bin_size = decoding_time_bin_size
                

        leave_one_out_decoding_analysis_obj = _long_short_decoding_analysis_from_decoders(long_one_step_decoder_1D, short_one_step_decoder_1D, long_session, short_session, global_session,
                                                                                           decoding_time_bin_size=decoding_time_bin_size, perform_cache_load=perform_cache_load)
        # TODO 2023-05-10 - need to update existing ['long_short'] if it exists:
        # global_computation_results.computed_data['long_short'] = {
        #     'leave_one_out_decoding_analysis': leave_one_out_decoding_analysis_obj
        # } # end long_short
        global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'] = leave_one_out_decoding_analysis_obj # end long_short
        # TODO 2023-05-10 - Do I want long_one_step_decoder_2D, short_one_step_decoder_2D that I computed?

        """ Getting outputs:
        
        
            ## long_short_decoding_analyses:
            curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
            ## Extract variables from results object:
            long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global


        """
        return global_computation_results









# ==================================================================================================================== #
# 2023-04-07 - `constrain_to_laps`                                                                                     #
#   Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.
# ==================================================================================================================== #
## 2023-04-07 - Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.

def constrain_to_laps(curr_active_pipeline):
    """ 2023-04-07 - Constrains the placefields to just the laps, computing the laps if needed.
    Other laps-related things?
        # ??? pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
        # DataSession.compute_position_laps(self)
        # DataSession.compute_laps_position_df(position_df, laps_df)

    Usage:
        from PendingNotebookCode import constrain_to_laps
        curr_active_pipeline = constrain_to_laps(curr_active_pipeline)
        
        
    MUTATES:
        curr_active_pipeline.computation_results[*].computed_data.pf1D,
        curr_active_pipeline.computation_results[*].computed_data.pf2D,
        Maybe others?
    """
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

    for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):
        # a_sess = estimate_session_laps(a_sess, should_plot_laps_2d=True)
        a_sess = a_sess.replace_session_laps_with_estimates(should_plot_laps_2d=False)
        
        ## The filter the laps specifically for use in the placefields with non-overlapping, duration, constraints:
        curr_laps_obj = a_sess.laps.as_epoch_obj() # set this to the laps object
        curr_laps_obj = curr_laps_obj.get_non_overlapping()
        curr_laps_obj = curr_laps_obj.filtered_by_duration(1.0, 30.0) # the lap must be at least 1 second long and at most 10 seconds long
        # curr_laps_obj = a_sess.estimate_laps().as_epoch_obj()

        ## Check if already the same:
        if curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs == curr_laps_obj:
            print(f'WARNING: constrain_to_laps(...): already had the computations ran with this laps object, so no recomputations are needed.')
            pass
        else:
            # Must recompute since the computation_epochs changed
            print(f'setting new computation epochs because laps changed.')
            curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs = curr_laps_obj # TODO: does this change the config that's used for computations? I think it should. 
            
            # Get existing placefields:
            curr_pf1D, curr_pf2D = a_result.pf1D, a_result.pf2D

            lap_filtered_curr_pf1D = deepcopy(curr_pf1D)
            lap_filtered_curr_pf1D = PfND(spikes_df=lap_filtered_curr_pf1D.spikes_df, position=lap_filtered_curr_pf1D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf1D.config, compute_on_init=True)
            lap_filtered_curr_pf2D = deepcopy(curr_pf2D)
            lap_filtered_curr_pf2D = PfND(spikes_df=lap_filtered_curr_pf2D.spikes_df, position=lap_filtered_curr_pf2D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf2D.config, compute_on_init=True)
            # Replace the result with the lap-filtered variety. This is perminant.
            a_result.pf1D = lap_filtered_curr_pf1D
            a_result.pf2D = lap_filtered_curr_pf2D

        return curr_active_pipeline

def compute_long_short_constrained_decoders(curr_active_pipeline, enable_two_step_decoders:bool = False, recalculate_anyway:bool=True):
    """ 2023-04-14 - Computes both 1D & 2D Decoders constrained to each other's position bins 
    Usage:

        (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_long_short_constrained_decoders(curr_active_pipeline)

        With Two-step Decoders:
        (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D) = compute_long_short_constrained_decoders(curr_active_pipeline, enable_two_step_decoders=True)

    """
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    # 1D Decoders constrained to each other
    def compute_short_long_constrained_decoders_1D(curr_active_pipeline, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 1D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]

        # long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
        long_one_step_decoder_1D, short_one_step_decoder_1D  = [deepcopy(results_data.get('pf1D_Decoder', None)) for results_data in (long_results, short_results)]
        # ds and Decoders conform between the long and the short epochs:
        short_one_step_decoder_1D, did_recompute = short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D, force_recompute=True)

        # ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_1D is None) or (short_two_step_decoder_1D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
                long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
                assert (long_two_step_decoder_1D is not None and short_two_step_decoder_1D is not None)
        else:
            long_two_step_decoder_1D, short_two_step_decoder_1D = None, None

        return long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D

    def compute_short_long_constrained_decoders_2D(curr_active_pipeline, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 2D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]
        # Make the 2D Placefields and Decoders conform between the long and the short epochs:
        long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
        short_one_step_decoder_2D, did_recompute = short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_2D, short_two_step_decoder_2D  = [results_data.get('pf2D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_2D is None) or (short_two_step_decoder_2D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
                long_two_step_decoder_2D, short_two_step_decoder_2D  = [results_data.get('pf2D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
                assert (long_two_step_decoder_2D is not None and short_two_step_decoder_2D is not None)
        else:
            long_two_step_decoder_2D, short_two_step_decoder_2D = None, None
        # Sums are similar:
        # print(f'{np.sum(long_one_step_decoder_2D.marginal.x.p_x_given_n) =},\t {np.sum(long_one_step_decoder_1D.p_x_given_n) = }') # 31181.999999999996 vs 31181.99999999999

        ## Validate:
        assert long_one_step_decoder_2D.marginal.x.p_x_given_n.shape == long_one_step_decoder_1D.p_x_given_n.shape, f"Must equal but: {long_one_step_decoder_2D.marginal.x.p_x_given_n.shape =} and {long_one_step_decoder_1D.p_x_given_n.shape =}"
        assert long_one_step_decoder_2D.marginal.x.most_likely_positions_1D.shape == long_one_step_decoder_1D.most_likely_positions.shape, f"Must equal but: {long_one_step_decoder_2D.marginal.x.most_likely_positions_1D.shape =} and {long_one_step_decoder_1D.most_likely_positions.shape =}"

        ## validate values:
        # assert np.allclose(long_one_step_decoder_2D.marginal.x.p_x_given_n, long_one_step_decoder_1D.p_x_given_n), f"1D Decoder should have an x-posterior equal to its own posterior"
        # assert np.allclose(curr_epoch_result['marginal_x']['most_likely_positions_1D'], curr_epoch_result['most_likely_positions']), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"
        return long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D

    ## BEGIN MAIN FUNCTION BODY:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D = compute_short_long_constrained_decoders_1D(curr_active_pipeline, enable_two_step_decoders=enable_two_step_decoders)
    long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D = compute_short_long_constrained_decoders_2D(curr_active_pipeline, enable_two_step_decoders=enable_two_step_decoders)

    if enable_two_step_decoders:
        return (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D)
    else:
        # Only return the one_step decoders
        return (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D)

# ==================================================================================================================== #
# 2023-05-10 - Long Short Decoding Analysis                                                                            #
# ==================================================================================================================== #

def _long_short_decoding_analysis_from_decoders(long_one_step_decoder_1D, short_one_step_decoder_1D, long_session, short_session, global_session, decoding_time_bin_size = 0.025, perform_cache_load=False):
    """ Uses existing decoders and other long/short variables to run `perform_full_session_leave_one_out_decoding_analysis` on each. """
    # Get existing long/short decoders from the cell under "# 2023-02-24 Decoders"
    long_decoder, short_decoder = deepcopy(long_one_step_decoder_1D), deepcopy(short_one_step_decoder_1D)
    assert np.all(long_decoder.xbin == short_decoder.xbin)

    ## backup existing replay objects
    # long_session.replay_backup, short_session.replay_backup, global_session.replay_backup = [deepcopy(a_session.replay) for a_session in [long_session, short_session, global_session]]
    # null-out the replay objects
    # long_session.replay, short_session.replay, global_session.replay = [None, None, None]

    
    # Extract replays either way:
    long_replays, short_replays, global_replays = [a_session.replay for a_session in [long_session, short_session, global_session]]

    # Prune to the shared aclus in both epochs (short/long):
    long_shared_aclus_only_decoder, short_shared_aclus_only_decoder = [BasePositionDecoder.init_from_stateful_decoder(a_decoder) for a_decoder in (long_decoder, short_decoder)]
    shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff = BasePositionDecoder.prune_to_shared_aclus_only(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder)

    n_neurons = len(shared_aclus)
    # # for plotting purposes, build colors only for the common (present in both, the intersection) neurons:
    # neurons_colors_array = build_neurons_color_map(n_neurons, sortby=None, cmap=None)
    # print(f'{n_neurons = }, {neurons_colors_array.shape =}')

    # with VizTracer(output_file=f"viztracer_{get_now_time_str()}-full_session_LOO_decoding_analysis.json", min_duration=200, tracer_entries=3000000, ignore_frozen=True) as tracer:
    long_results_obj = perform_full_session_leave_one_out_decoding_analysis(global_session, original_1D_decoder=long_shared_aclus_only_decoder, decoding_time_bin_size=decoding_time_bin_size, cache_suffix = '_long', perform_cache_load=perform_cache_load) # , perform_cache_load=False
    short_results_obj = perform_full_session_leave_one_out_decoding_analysis(global_session, original_1D_decoder=short_shared_aclus_only_decoder, decoding_time_bin_size=decoding_time_bin_size, cache_suffix = '_short', perform_cache_load=perform_cache_load) # , perform_cache_load=False

    leave_one_out_decoding_analysis_obj = LeaveOneOutDecodingAnalysis(long_decoder, short_decoder, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global=True)

    ## Dict mode for result    
    # owning_pipeline_reference.global_computation_results.computed_data.long_short = {
    #     'leave_one_out_decoding_analysis': {
    #             'long_decoder': long_decoder,  'short_decoder': short_decoder, 
    #             'long_replays': long_replays,  'short_replays': short_replays,  'global_replays': global_replays,
    #             'long_shared_aclus_only_decoder': long_shared_aclus_only_decoder,  'short_shared_aclus_only_decoder': short_shared_aclus_only_decoder, 
    #             'shared_aclus': shared_aclus,  'long_shared_aclus_only_decoder': long_shared_aclus_only_decoder,  'short_shared_aclus_only_decoder': short_shared_aclus_only_decoder,  'long_short_pf_neurons_diff': long_short_pf_neurons_diff, 
    #             'n_neurons': n_neurons,
    #             'long_results_obj': long_results_obj,  'short_results_obj': short_results_obj
    #     }
    # } # end long_short

    return leave_one_out_decoding_analysis_obj




@define
class LongShortPipelineTests(object):
	"""2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks."""
	curr_active_pipeline: "NeuropyPipeline"

	def validate_placefields(self):
		""" 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
		long_epoch_name, short_epoch_name, global_epoch_name = self.curr_active_pipeline.find_LongShortGlobal_epoch_names()
		long_results, short_results, global_results = [self.curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
		# Assert conformance between the long and short position bins for both the 1D and 2D placefields. This should be the case because pf_params.grid_bin and pf_params.grid_bin_bounds are set to the same for both tracks:
		assert np.all(long_results.pf1D.xbin == short_results.pf1D.xbin), f"long_results.pf1D.xbin: {len(long_results.pf1D.xbin)}, short_results.pf1D.xbin: {len(short_results.pf1D.xbin)}"
		assert np.all(long_results.pf2D.xbin == short_results.pf2D.xbin), f"long_results.pf2D.xbin: {len(long_results.pf2D.xbin)}, short_results.pf2D.xbin: {len(short_results.pf2D.xbin)}"
		assert np.all(long_results.pf2D.ybin == short_results.pf2D.ybin), f"long_results.pf2D.ybin: {len(long_results.pf2D.ybin)}, short_results.pf2D.ybin: {len(short_results.pf2D.ybin)}"

	def validate_decoders(self):
		## Decoders should also conform if placefields do from the onset prior to computations:
		long_epoch_name, short_epoch_name, global_epoch_name = self.curr_active_pipeline.find_LongShortGlobal_epoch_names()
		long_results, short_results, global_results = [self.curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
		assert np.all(long_results.pf1D_Decoder.xbin == short_results.pf1D_Decoder.xbin), f"long_results.pf1D_Decoder.xbin: {len(long_results.pf1D_Decoder.xbin)}, short_results.pf1D_Decoder.xbin: {len(short_results.pf1D_Decoder.xbin)}"
		assert np.all(long_results.pf2D_Decoder.xbin == short_results.pf2D_Decoder.xbin), f"long_results.pf2D_Decoder.xbin: {len(long_results.pf2D_Decoder.xbin)}, short_results.pf2D_Decoder.xbin: {len(short_results.pf2D_Decoder.xbin)}"
		assert np.all(long_results.pf2D_Decoder.ybin == short_results.pf2D_Decoder.ybin), f"long_results.pf2D_Decoder.ybin: {len(long_results.pf2D_Decoder.ybin)}, short_results.pf2D_Decoder.ybin: {len(short_results.pf2D_Decoder.ybin)}"


	def validate(self):
		self.validate_placefields()
		self.validate_decoders()