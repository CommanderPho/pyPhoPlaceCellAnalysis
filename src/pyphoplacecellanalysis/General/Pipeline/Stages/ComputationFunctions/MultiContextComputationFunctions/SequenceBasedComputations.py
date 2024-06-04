# ==================================================================================================================== #
# 2024-05-27 - WCorr Shuffle Stuff                                                                                     #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from typing import NewType

import attrs
from attrs import asdict, define, field, Factory, astuple

import numpy as np
import pandas as pd

from attrs import define, field, asdict, evolve
import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from neuropy.core.epoch import find_data_indicies_from_epoch_times


from neuropy.utils.misc import build_shuffled_ids # used in _SHELL_analyze_leave_one_out_decoding_results

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import compute_weighted_correlations
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

# ==================================================================================================================== #
# 2024-05-24 - Shuffling to show wcorr exceeds shuffles                                                                #
# ==================================================================================================================== #
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType
from nptyping import NDArray

import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates


DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names
ShuffleIdx = NewType('ShuffleIdx', int)


@define(slots=False, repr=False, eq=False)
class WCorrShuffle(ComputedResult):
    """ Performs shufflings to test wcorr
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle

    wcorr_tool: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates,
        directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result, directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result,
        global_epoch_name=global_epoch_name)


    """
    _VersionedResultMixin_version: str = "2024.05.30_0" # to be updated in your IMPLEMENTOR to indicate its version

    curr_active_pipeline = non_serialized_field(default=None, repr=False) # required to continue computations
    track_templates = non_serialized_field(default=None, repr=False) # required to continue computations

    filtered_epochs_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    active_spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)

    real_decoder_ripple_weighted_corr_arr: NDArray = serialized_field(default=None, repr=False, metadata={'shape': ('n_epochs', 'n_decoders')})

    all_templates_decode_kwargs: Dict = non_serialized_field(default=Factory(dict), repr=False)

    output_extracted_result_wcorrs_list: List = serialized_field(default=Factory(list), repr=False)
    output_all_shuffles_decoded_results_list: List[DecodedEpochsResultsDict] = serialized_field(default=Factory(list), repr=False, metadata={'description': 'optionally (depending on `enable_saving_entire_decoded_shuffle_result`) produced decoding results that can allow reuse across multiple computation types (other than wcorr). Takes MUCH more memeory.'}) ## Opt
    
    enable_saving_entire_decoded_shuffle_result: bool = serialized_attribute_field(default=True, is_computable=False, repr=True)

    result_version: str = serialized_attribute_field(default='2024.05.30_0', is_computable=False, repr=False) # this field specfies the version of the result. 

    @property
    def n_epochs(self):
        """The number of epochs property."""
        return len(self.filtered_epochs_df)


    @property
    def n_completed_shuffles(self):
        """The n_completed_shuffles property."""
        return len(self.output_extracted_result_wcorrs_list)

    @property
    def all_shuffles_wcorr_array(self) -> NDArray:
        """The wcorrs for each shuffle. .shape: (n_shuffles, n_epochs, 4) """
        _out_wcorr = []
        total_n_shuffles: int = len(self.output_extracted_result_wcorrs_list)
        assert total_n_shuffles > 0

        ## USES: self.output_extracted_result_wcorrs_list, self.real_decoder_ripple_weighted_corr_arr
        for a_decoder_ripple_weighted_corr_df_dict in self.output_extracted_result_wcorrs_list:
            decoder_ripple_weighted_corr_dict = {k:v['wcorr'].to_numpy() for k, v in a_decoder_ripple_weighted_corr_df_dict.items()}
            a_decoder_ripple_weighted_corr_df = pd.DataFrame(decoder_ripple_weighted_corr_dict) ## (n_epochs, 4)
            a_shuffle_wcorr_arr = a_decoder_ripple_weighted_corr_df.to_numpy()            
            _out_wcorr.append(a_shuffle_wcorr_arr)
        return np.stack(_out_wcorr) # .shape ## (n_shuffles, n_epochs, 4)


    @classmethod
    def build_real_result(cls, track_templates, directional_merged_decoders_result, active_spikes_df, all_templates_decode_kwargs) -> Tuple[Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]], NDArray]:
        real_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)
        # real_output_alt_directional_merged_decoders_result, (real_decoder_laps_filter_epochs_decoder_result_dict, real_decoder_ripple_filter_epochs_decoder_result_dict) = _try_all_templates_decode(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), a_directional_merged_decoders_result=real_directional_merged_decoders_result, shuffled_decoders_dict=real_directional_merged_decoders_result.all_directional_decoder_dict, **a_sweep_dict)
        real_output_alt_directional_merged_decoders_result, (real_decoder_laps_filter_epochs_decoder_result_dict, real_decoder_ripple_filter_epochs_decoder_result_dict) = cls._try_all_templates_decode(spikes_df=active_spikes_df, a_directional_merged_decoders_result=real_directional_merged_decoders_result, shuffled_decoders_dict=track_templates.get_decoders_dict(), 
                                                                                                                                                                                                    skip_merged_decoding=True, **all_templates_decode_kwargs)
        real_decoder_ripple_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(real_decoder_ripple_filter_epochs_decoder_result_dict))
        real_decoder_ripple_weighted_corr_dict = {k:v['wcorr'].to_numpy() for k, v in real_decoder_ripple_weighted_corr_df_dict.items()}
        real_decoder_ripple_weighted_corr_df: pd.DataFrame = pd.DataFrame(real_decoder_ripple_weighted_corr_dict) ## (n_epochs, 4)
        real_decoder_ripple_weighted_corr_arr: NDArray = real_decoder_ripple_weighted_corr_df.to_numpy()
        print(f'real_decoder_ripple_weighted_corr_arr: {np.shape(real_decoder_ripple_weighted_corr_arr)}')
        # return real_directional_merged_decoders_result, real_decoder_ripple_weighted_corr_arr
        return (real_directional_merged_decoders_result, real_decoder_ripple_filter_epochs_decoder_result_dict), real_decoder_ripple_weighted_corr_arr
    

    @classmethod
    def init_from_templates(cls, curr_active_pipeline, enable_saving_entire_decoded_shuffle_result: bool=False, track_templates=None, directional_decoders_epochs_decode_result=None, global_epoch_name=None, debug_print=False) -> "WCorrShuffle":
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle

        wcorr_tool: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates, directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result, global_epoch_name=global_epoch_name)

        """
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #
        ## INPUTS: track_templates, directional_decoders_epochs_decode_result, 

        # BEGIN BLOCK 2 - modernizing from `_perform_compute_custom_epoch_decoding`  ________________________________________________________________________________________________________ #

        ## Copy the default result:
        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
        alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        if directional_decoders_epochs_decode_result is None:
            directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
        
        if track_templates is None:
            directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # used to get track_templates
            rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder'] # only used for `rank_order_results.minimum_inclusion_fr_Hz`
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)


        directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=False)

        ## INPUTS: curr_active_pipeline, global_epoch_name, track_templates

        # 2024-03-04 - Filter out the epochs based on the criteria:
        filtered_epochs_df, filtered_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)

        # print(f'real_decoder_ripple_weighted_corr_arr: {np.shape(real_decoder_ripple_weighted_corr_arr)}')

        ## Adds 'is_most_likely_direction_LR', 'P_LR' to the `filtered_epochs_df` so we can determine which direction is most likely. This uses `directional_decoders_epochs_decode_result`
        ## INPUTS: directional_decoders_epochs_decode_result

        ##Gotta get those ['P_LR', 'P_RL'] columns to determine best directions
        extracted_merged_scores_df: pd.DataFrame = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df()
        extracted_merged_scores_df['is_most_likely_direction_LR'] = (extracted_merged_scores_df['P_LR'] > 0.5)

        ## Find the correct indicies corresponding to the filtered events
        filtered_start_times = deepcopy(filtered_epochs_df['start'].to_numpy())
        filtered_epoch_indicies = find_data_indicies_from_epoch_times(extracted_merged_scores_df, np.squeeze(filtered_start_times), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
        # Constrain the `extracted_merged_scores_df` to match the `filtered_epochs_df`
        extracted_merged_scores_df['is_included_in_filtered'] = False
        extracted_merged_scores_df['is_included_in_filtered'].iloc[filtered_epoch_indicies] = True
        filtered_extracted_merged_scores_df = extracted_merged_scores_df[extracted_merged_scores_df['is_included_in_filtered']]

        included_cols = ['P_LR', 'is_most_likely_direction_LR']
        assert len(filtered_epochs_df) == len(filtered_extracted_merged_scores_df), f"better match in length before we add properties"
        for a_col in included_cols:
            filtered_epochs_df[a_col] = filtered_extracted_merged_scores_df[a_col].to_numpy()

        _out_best_dir_indicies = []
        # LR
        _LR_indicies = [0, 2]
        _RL_indicies = [1, 3]

        for an_is_most_likely_direction_LR in filtered_epochs_df['is_most_likely_direction_LR']:
            if an_is_most_likely_direction_LR:
                _out_best_dir_indicies.append(_LR_indicies)
            else:
                _out_best_dir_indicies.append(_RL_indicies)

        _out_best_dir_indicies = np.vstack(_out_best_dir_indicies)
        # _out_best_dir_indicies

        filtered_epochs_df['long_best_dir_decoder_IDX'] = _out_best_dir_indicies[:,0]
        filtered_epochs_df['short_best_dir_decoder_IDX'] = _out_best_dir_indicies[:,1]

        ## OUTPUTS: filtered_epochs_df['long_best_dir_decoder_IDX'], filtered_epochs_df['short_best_dir_decoder_IDX']

        ## Filter the epochs by minimum duration:
        replay_epochs_df: pd.DataFrame = deepcopy(filtered_epochs_df)
        desired_ripple_decoding_time_bin_size: float = alt_directional_merged_decoders_result.ripple_decoding_time_bin_size
        minimum_event_duration: float = 2.0 * float(alt_directional_merged_decoders_result.ripple_decoding_time_bin_size)
        ## Drop those less than the time bin duration
        if debug_print:
            print(f'DropShorterMode:')
        pre_drop_n_epochs = len(replay_epochs_df)
        if minimum_event_duration is not None:                
            replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] >= minimum_event_duration]
            post_drop_n_epochs = len(replay_epochs_df)
            n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
            if debug_print:
                print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
        else:
            replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > desired_ripple_decoding_time_bin_size]
            post_drop_n_epochs = len(replay_epochs_df)
            n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
            if debug_print:
                print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_ripple_decoding_time_bin_size}', end='\t') 


        # all_templates_decode_kwargs = dict(desired_shared_decoding_time_bin_size=alt_directional_merged_decoders_result.ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=False, minimum_event_duration=alt_directional_merged_decoders_result.ripple_decoding_time_bin_size)
        all_templates_decode_kwargs = dict(desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size,
                            override_replay_epochs_df=replay_epochs_df, ## Use the filtered epochs
                            use_single_time_bin_per_epoch=False, minimum_event_duration=minimum_event_duration)
        

        # ==================================================================================================================== #
        # REAL                                                                                                                 #
        # ==================================================================================================================== #
        active_spikes_df = deepcopy(curr_active_pipeline.sess.spikes_df)
        (real_directional_merged_decoders_result, real_decoder_ripple_filter_epochs_decoder_result_dict), real_decoder_ripple_weighted_corr_arr = cls.build_real_result(track_templates=track_templates, directional_merged_decoders_result=alt_directional_merged_decoders_result, active_spikes_df=active_spikes_df, all_templates_decode_kwargs=all_templates_decode_kwargs)

        # laps_pre_computed_filter_epochs_dict, ripple_pre_computed_filter_epochs_dict = cls._pre_build_all_templates_decoding_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
        #                                                                                                                             a_directional_merged_decoders_result=alt_directional_merged_decoders_result,
        #                                                                                                                             shuffled_decoders_dict=deepcopy(track_templates.get_decoders_dict()),
        #                                                                                                                             **all_templates_decode_kwargs)

        return cls(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates,
            filtered_epochs_df=replay_epochs_df, active_spikes_df=active_spikes_df,
            # alt_directional_merged_decoders_result=alt_directional_merged_decoders_result, real_directional_merged_decoders_result=real_directional_merged_decoders_result,
            real_decoder_ripple_weighted_corr_arr=real_decoder_ripple_weighted_corr_arr,
            all_templates_decode_kwargs=all_templates_decode_kwargs, enable_saving_entire_decoded_shuffle_result=enable_saving_entire_decoded_shuffle_result) # output_extracted_result_wcorrs_list=[], 


    @classmethod
    def compute_z_transformed_scores(cls, data: NDArray) -> NDArray:
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        return z_scores

    @classmethod
    def compute_z_score(cls, data: NDArray, real_v: float) -> float:
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = (real_v - mean) / std_dev
        return z_scores

    @classmethod
    def _shuffle_pf1D_decoder(cls, a_pf1D_Decoder: BasePositionDecoder, shuffle_IDXs: NDArray, shuffle_aclus: NDArray) -> BasePositionDecoder:
        """ Shuffle the neuron_ids for a `alt_directional_merged_decoders_result` - `DirectionalPseudo2DDecodersResult `:
        """
        a_shuffled_decoder = deepcopy(a_pf1D_Decoder)
        # restrict the shuffle_acus to the actual aclus of the ratemap
        is_shuffle_aclu_included = np.isin(shuffle_aclus, a_shuffled_decoder.pf.ratemap.neuron_ids)
        shuffle_aclus = shuffle_aclus[is_shuffle_aclu_included]

        ## find the correct indicies to shuffle by:
        # shuffle_IDXs = [list(shuffle_aclus).index(aclu) for aclu in a_shuffled_decoder.pf.ratemap.neuron_ids]
        shuffle_IDXs = [list(a_shuffled_decoder.pf.ratemap.neuron_ids).index(aclu) for aclu in shuffle_aclus]
        a_shuffled_decoder.pf.ratemap = a_shuffled_decoder.pf.ratemap.get_by_id(shuffle_aclus)
        neuron_indexed_field_names = ['neuron_IDs', 'neuron_IDs']
        for a_field in neuron_indexed_field_names:
            setattr(a_shuffled_decoder, a_field, getattr(a_shuffled_decoder, a_field)[shuffle_IDXs])

        a_shuffled_decoder.F[shuffle_IDXs, :] = a_shuffled_decoder.F[shuffle_IDXs, :] # @TODO - is this needed?

        return a_shuffled_decoder
    



    @classmethod
    def _pre_build_all_templates_decoding_epochs(cls, spikes_df: pd.DataFrame, a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, shuffled_decoders_dict, use_single_time_bin_per_epoch: bool,
                            override_replay_epochs_df: Optional[pd.DataFrame]=None, desired_laps_decoding_time_bin_size: Optional[float]=None, desired_ripple_decoding_time_bin_size: Optional[float]=None, desired_shared_decoding_time_bin_size: Optional[float]=None, minimum_event_duration: Optional[float]=None, debug_print: bool = False) -> Tuple[Dict[types.DecoderName, Any], Dict[types.DecoderName, Any]]: #-> Tuple[None, Tuple[Dict[str, DecodedFilterEpochsResult], Dict[str, DecodedFilterEpochsResult]]]:
        """ 
        
        #TODO 2024-06-03 11:07: - [ ] NOT WORKING, REVERTED TO LONG-FORM METHOD

        History:
            Split `_try_all_templates_decode` into `_pre_build_all_templates_decoding_epochs` and `_all_templates_perform_pre_built_specific_epochs_decoding`

            Split `perform_decode_specific_epochs` into two subfunctions: `_build_decode_specific_epochs_result_shell` and `_perform_decoding_specific_epochs`
            pre_built_epochs_decoding_result = self.pre_build_epochs_decoding_result(neuron_IDs=self.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
            return self.perform_pre_built_specific_epochs_decoding(filter_epochs_decoder_result=pre_built_epochs_decoding_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)

        """
        ripple_decoding_time_bin_size = None
        if desired_shared_decoding_time_bin_size is not None:
            assert desired_laps_decoding_time_bin_size is None
            assert desired_ripple_decoding_time_bin_size is None
            desired_laps_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            desired_ripple_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            
        ## Decode Laps:
        if desired_laps_decoding_time_bin_size is not None:
            laps_epochs_df = deepcopy(a_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs)
            if not isinstance(laps_epochs_df, pd.DataFrame):
                laps_epochs_df = laps_epochs_df.to_dataframe()
            # global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
            min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(laps_epochs_df['duration'].to_numpy())
            min_bounded_laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
            if desired_laps_decoding_time_bin_size < min_bounded_laps_decoding_time_bin_size:
                print(f'WARN: desired_laps_decoding_time_bin_size: {desired_laps_decoding_time_bin_size} < min_bounded_laps_decoding_time_bin_size: {min_bounded_laps_decoding_time_bin_size}... hopefully it works.')
            laps_decoding_time_bin_size: float = desired_laps_decoding_time_bin_size # allow direct use
            if use_single_time_bin_per_epoch:
                laps_decoding_time_bin_size = None

        ## Decode Ripples: ripples are kinda optional (if `desired_ripple_decoding_time_bin_size is None` they are not computed.
        if desired_ripple_decoding_time_bin_size is not None:
            if override_replay_epochs_df is not None:
                replay_epochs_df = deepcopy(override_replay_epochs_df)
            else:
                # global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
                replay_epochs_df = deepcopy(a_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
            
            if not isinstance(replay_epochs_df, pd.DataFrame):
                replay_epochs_df = replay_epochs_df.to_dataframe()
            ripple_decoding_time_bin_size: float = desired_ripple_decoding_time_bin_size # allow direct use            
            ## Drop those less than the time bin duration
            if debug_print:
                print(f'DropShorterMode:')
            pre_drop_n_epochs = len(replay_epochs_df)
            if minimum_event_duration is not None:                
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] >= minimum_event_duration]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                if debug_print:
                    print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
            else:
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > desired_ripple_decoding_time_bin_size]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                if debug_print:
                    print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_ripple_decoding_time_bin_size}', end='\t') 

            if debug_print:
                print(f'{post_drop_n_epochs} remain.')

        laps_pre_computed_filter_epochs_dict: Dict[types.DecoderName, Any] = {}
        ripple_pre_computed_filter_epochs_dict: Dict[types.DecoderName, Any] = {}
                
        ## This does the single 1D versions
        for a_name, a_decoder in shuffled_decoders_dict.items():
            a_directional_ripple_pre_built_epochs_decoding_result = a_decoder.pre_build_epochs_decoding_result(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(replay_epochs_df), decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

            laps_pre_computed_filter_epochs_dict[a_name] = None
            ripple_pre_computed_filter_epochs_dict[a_name] = a_directional_ripple_pre_built_epochs_decoding_result


        ## OUTPUTS: (replay_epochs_df, ), 
        return (laps_pre_computed_filter_epochs_dict, ripple_pre_computed_filter_epochs_dict)

    @classmethod
    def _all_templates_perform_pre_built_specific_epochs_decoding(cls, laps_pre_computed_filter_epochs_dict, ripple_pre_computed_filter_epochs_dict, shuffled_decoders_dict, use_single_time_bin_per_epoch: bool, debug_print: bool = False, **kwargs) -> Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]: #-> Tuple[None, Tuple[Dict[str, DecodedFilterEpochsResult], Dict[str, DecodedFilterEpochsResult]]]:
        """ decodes laps and ripples for a single bin size but for each of the four track templates. 
        
        Split `_try_all_templates_decode` into `_pre_build_all_templates_decoding_epochs` and `_all_templates_perform_pre_built_specific_epochs_decoding`

        #TODO 2024-06-03 11:07: - [ ] NOT WORKING, REVERTED TO LONG-FORM METHOD
        
        History:
            Split `perform_decode_specific_epochs` into two subfunctions: `_build_decode_specific_epochs_result_shell` and `_perform_decoding_specific_epochs`
            pre_built_epochs_decoding_result = self.pre_build_epochs_decoding_result(neuron_IDs=self.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
            return self.perform_pre_built_specific_epochs_decoding(filter_epochs_decoder_result=pre_built_epochs_decoding_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
            

        """

        # laps_pre_computed_filter_epochs_dict, ripple_pre_computed_filter_epochs_dict = cls._pre_build_all_templates_decoding_epochs(spikes_df=spikes_df, )

        decoder_laps_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        decoder_ripple_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        
        ## This does the single 1D versions
        for a_name, a_decoder in shuffled_decoders_dict.items():
            a_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_decoder.perform_pre_built_specific_epochs_decoding(filter_epochs_decoder_result=deepcopy(ripple_pre_computed_filter_epochs_dict[a_name]), use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
            decoder_laps_filter_epochs_decoder_result_dict[a_name] = None
            decoder_ripple_filter_epochs_decoder_result_dict[a_name] = a_directional_ripple_filter_epochs_decoder_result

        return (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        


    ## All templates AND merged decode:
    @classmethod
    def _try_all_templates_decode(cls, spikes_df: pd.DataFrame, a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, shuffled_decoders_dict, use_single_time_bin_per_epoch: bool,
                            override_replay_epochs_df: Optional[pd.DataFrame]=None,
                            desired_laps_decoding_time_bin_size: Optional[float]=None, desired_ripple_decoding_time_bin_size: Optional[float]=None, desired_shared_decoding_time_bin_size: Optional[float]=None, minimum_event_duration: Optional[float]=None,
                            skip_merged_decoding=False, debug_print: bool = False) -> Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]: #-> Tuple[None, Tuple[Dict[str, DecodedFilterEpochsResult], Dict[str, DecodedFilterEpochsResult]]]:
        """ MOSTLY DEPRICATED - used only for `build_real_result(...)`
         
        decodes laps and ripples for a single bin size but for each of the four track templates. 
        
        Added 2024-05-23 04:23 

        desired_laps_decoding_time_bin_size
        desired_ripple_decoding_time_bin_size
        minimum_event_duration: if provided, excludes all events shorter than minimum_event_duration

        Uses:
            .all_directional_pf1D_Decoder - calling `decode_specific_epochs(...)` on it

        Looks like it updates:
            .all_directional_laps_filter_epochs_decoder_result, .all_directional_ripple_filter_epochs_decoder_result, and whatever .perform_compute_marginals() updates

        
        Compared to `_compute_lap_and_ripple_epochs_decoding_for_decoder`, it looks like this only computes for the `*all*_directional_pf1D_Decoder` while `_compute_lap_and_ripple_epochs_decoding_for_decoder` is called for each separate directional pf1D decoder

        Usage:

            from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, _try_all_templates_decode

        
        History:
            Split `perform_decode_specific_epochs` into two subfunctions: `_build_decode_specific_epochs_result_shell` and `_perform_decoding_specific_epochs`
            pre_built_epochs_decoding_result = self.pre_build_epochs_decoding_result(neuron_IDs=self.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
            return self.perform_pre_built_specific_epochs_decoding(filter_epochs_decoder_result=pre_built_epochs_decoding_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
            

        """
        ripple_decoding_time_bin_size = None
        if desired_shared_decoding_time_bin_size is not None:
            assert desired_laps_decoding_time_bin_size is None
            assert desired_ripple_decoding_time_bin_size is None
            desired_laps_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            desired_ripple_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            
        if not skip_merged_decoding:
            # Separate the decoder first so they're all independent:
            a_directional_merged_decoders_result = deepcopy(a_directional_merged_decoders_result)

        ## Decode Laps:
        if desired_laps_decoding_time_bin_size is not None:
            laps_epochs_df = deepcopy(a_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs)
            if not isinstance(laps_epochs_df, pd.DataFrame):
                laps_epochs_df = laps_epochs_df.to_dataframe()
            # global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
            min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(laps_epochs_df['duration'].to_numpy())
            min_bounded_laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
            if desired_laps_decoding_time_bin_size < min_bounded_laps_decoding_time_bin_size:
                print(f'WARN: desired_laps_decoding_time_bin_size: {desired_laps_decoding_time_bin_size} < min_bounded_laps_decoding_time_bin_size: {min_bounded_laps_decoding_time_bin_size}... hopefully it works.')
            laps_decoding_time_bin_size: float = desired_laps_decoding_time_bin_size # allow direct use
            if use_single_time_bin_per_epoch:
                laps_decoding_time_bin_size = None

            if not skip_merged_decoding:
                a_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = a_directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=laps_epochs_df,
                                                                                                                                                                decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

        ## Decode Ripples: ripples are kinda optional (if `desired_ripple_decoding_time_bin_size is None` they are not computed.
        if desired_ripple_decoding_time_bin_size is not None:
            if override_replay_epochs_df is not None:
                replay_epochs_df = deepcopy(override_replay_epochs_df)
            else:
                # global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
                replay_epochs_df = deepcopy(a_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
            
            if not isinstance(replay_epochs_df, pd.DataFrame):
                replay_epochs_df = replay_epochs_df.to_dataframe()
            ripple_decoding_time_bin_size: float = desired_ripple_decoding_time_bin_size # allow direct use            
            ## Drop those less than the time bin duration
            if debug_print:
                print(f'DropShorterMode:')
            pre_drop_n_epochs = len(replay_epochs_df)
            if minimum_event_duration is not None:                
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] >= minimum_event_duration]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                if debug_print:
                    print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
            else:
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > desired_ripple_decoding_time_bin_size]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                if debug_print:
                    print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_ripple_decoding_time_bin_size}', end='\t') 

            if debug_print:
                print(f'{post_drop_n_epochs} remain.')

            if not skip_merged_decoding:
                # returns a `DecodedFilterEpochsResult`
                a_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = a_directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=replay_epochs_df,
                                                                                                                                                                                                decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

        if not skip_merged_decoding:
            a_directional_merged_decoders_result.perform_compute_marginals() # this only works for the pseudo2D decoder, not the individual 1D ones

        # directional_merged_decoders_result_dict: Dict[types.DecoderName, DirectionalPseudo2DDecodersResult] = {}

        decoder_laps_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        decoder_ripple_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        
        ## This does the single 1D versions
        for a_name, a_decoder in shuffled_decoders_dict.items():
            # external-function way:
            # decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)
            a_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(deepcopy(spikes_df), filter_epochs=deepcopy(replay_epochs_df), decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
            decoder_laps_filter_epochs_decoder_result_dict[a_name] = None
            decoder_ripple_filter_epochs_decoder_result_dict[a_name] = a_directional_ripple_filter_epochs_decoder_result

        if skip_merged_decoding:
            a_directional_merged_decoders_result = None

        return a_directional_merged_decoders_result, (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        

    @classmethod
    def _shuffle_and_decode_wcorrs(cls, curr_active_pipeline, track_templates: TrackTemplates, alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, all_templates_decode_kwargs: Dict, num_shuffles: int = 2):
        """ We shuffle the cell idenitities and decodes new posteriors from the shuffled values.

        Only computes for the ripples, not the laps

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import shuffle_and_decode_wcorrs

            num_shuffles: int = 10
            _updated_output_extracted_result_wcorrs_list = shuffle_and_decode_wcorrs(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates, alt_directional_merged_decoders_result=alt_directional_merged_decoders_result, all_templates_decode_kwargs=all_templates_decode_kwargs, num_shuffles=num_shuffles)


        """
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        is_pre_compute_mode: bool = False #TODO 2024-05-30 10:00: - [ ] is_pre_compute_mode seems to produce the same shuffles each time :[

        ## INPUTS: num_shuffles
        
        ## INPUTS: alt_directional_merged_decoders_result, num_shuffles, 
        ## Requires: `output_extracted_result_wcorrs_list`
        # output_extracted_result_wcorrs_list = [] 
        _updated_output_extracted_result_wcorrs_list = []
        _updated_output_extracted_full_decoded_results_list = []

        shuffled_aclus, shuffle_IDXs = build_shuffled_ids(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.neuron_IDs, num_shuffles=num_shuffles, seed=None)

        if is_pre_compute_mode:
            laps_pre_computed_filter_epochs_dict, ripple_pre_computed_filter_epochs_dict = cls._pre_build_all_templates_decoding_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), a_directional_merged_decoders_result=alt_directional_merged_decoders_result, shuffled_decoders_dict=deepcopy(track_templates.get_decoders_dict()), **all_templates_decode_kwargs)


        ## FOR EACH SHUFFLE:
        for i, a_shuffle_IDXs, a_shuffle_aclus in zip(np.arange(num_shuffles), shuffle_IDXs, shuffled_aclus):
            print(f'a_shuffle_IDXs: {a_shuffle_IDXs}, a_shuffle_aclus: {a_shuffle_aclus}')

            ## Shuffle the neuron_ids for a `alt_directional_merged_decoders_result` - `DirectionalPseudo2DDecodersResult `:
            alt_directional_merged_decoders_result.all_directional_pf1D_Decoder = cls._shuffle_pf1D_decoder(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder, shuffle_IDXs=a_shuffle_IDXs, shuffle_aclus=a_shuffle_aclus)

            shuffled_decoder_specific_neuron_ids_dict = dict(zip(track_templates.get_decoder_names(), [a_shuffle_aclus[np.isin(a_shuffle_aclus, v)] for v in track_templates.decoder_neuron_IDs_list]))

            ## Shuffle the four 1D decoders as well so they can be passed in.
            shuffled_decoders_dict: Dict[str, BasePositionDecoder] = {a_name:cls._shuffle_pf1D_decoder(a_decoder, shuffle_IDXs=a_shuffle_IDXs, shuffle_aclus=shuffled_decoder_specific_neuron_ids_dict[a_name]) for a_name, a_decoder in track_templates.get_decoders_dict().items()}

            ## Decode epochs for all four decoders:
        

            if is_pre_compute_mode:
                decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = cls._all_templates_perform_pre_built_specific_epochs_decoding(laps_pre_computed_filter_epochs_dict=deepcopy(laps_pre_computed_filter_epochs_dict),
                                                                                                                                                                                ripple_pre_computed_filter_epochs_dict=deepcopy(ripple_pre_computed_filter_epochs_dict),
                                                                                                                                                                                shuffled_decoders_dict=shuffled_decoders_dict, **all_templates_decode_kwargs)
            
            else:
                _, (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = cls._try_all_templates_decode(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), a_directional_merged_decoders_result=alt_directional_merged_decoders_result, shuffled_decoders_dict=shuffled_decoders_dict,
                                                                                                                                                                        skip_merged_decoding=True, **all_templates_decode_kwargs)


            ## Weighted Correlation
            # decoder_laps_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_laps_filter_epochs_decoder_result_dict))
            decoder_ripple_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_ripple_filter_epochs_decoder_result_dict))

            ## Build the output tuple:
            # output_extracted_result_wcorrs[i] = decoder_ripple_weighted_corr_df_dict # only wcorr
            _updated_output_extracted_result_wcorrs_list.append(decoder_ripple_weighted_corr_df_dict)
            _updated_output_extracted_full_decoded_results_list.append(decoder_ripple_filter_epochs_decoder_result_dict)

        return _updated_output_extracted_result_wcorrs_list, _updated_output_extracted_full_decoded_results_list


    def compute_shuffles(self, num_shuffles: int=100, curr_active_pipeline=None, track_templates=None) -> List:
        """ Computes new shuffles and adds them to `self.output_extracted_result_wcorrs_list`
        """
        if (self.curr_active_pipeline is None):
            if (curr_active_pipeline is None):
                raise NotImplementedError(f"cannot compute because self.curr_active_pipeline is missing and no curr_active_pipeline were provided as kwargs!")
            else:
                # non-None pipeline passed in, use for self
                self.curr_active_pipeline = curr_active_pipeline

        
        if (self.track_templates is None) or isinstance(self.track_templates, dict):
            if (track_templates is None):
                ## recover them from `curr_active_pipeline`
                if self.curr_active_pipeline is not None:
                    directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # used to get track_templates
                    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder'] # only used for `rank_order_results.minimum_inclusion_fr_Hz`
                    minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                    track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
                    self.track_templates = deepcopy(track_templates)
                else:
                    raise NotImplementedError(f"cannot compute because self.track_templates is missing and no track_templates were provided as kwargs!")
            else:
                # non-None pipeline passed in, use for self
                self.track_templates = track_templates

        assert ((self.curr_active_pipeline is not None) and  (self.track_templates is not None))

        _updated_output_extracted_result_wcorrs_list, _updated_output_extracted_full_decoded_results_list = self._shuffle_and_decode_wcorrs(curr_active_pipeline=self.curr_active_pipeline, track_templates=self.track_templates,
                                                                                  alt_directional_merged_decoders_result=deepcopy(self.curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']),
                                                                                  all_templates_decode_kwargs=self.all_templates_decode_kwargs,
                                                                                  num_shuffles=num_shuffles)

        self.output_extracted_result_wcorrs_list.extend(_updated_output_extracted_result_wcorrs_list)
        if self.enable_saving_entire_decoded_shuffle_result:
            self.output_all_shuffles_decoded_results_list.extend(_updated_output_extracted_full_decoded_results_list)


    def get_all_shuffles_wcorr_array(self) -> NDArray:
        """The wcorrs for each shuffle. .shape: (n_shuffles, n_epochs, 4) """
        _out_wcorr = []
        total_n_shuffles: int = len(self.output_extracted_result_wcorrs_list)
        assert total_n_shuffles > 0

        ## USES: self.output_extracted_result_wcorrs_list, self.real_decoder_ripple_weighted_corr_arr
        for a_decoder_ripple_weighted_corr_df_dict in self.output_extracted_result_wcorrs_list:
            decoder_ripple_weighted_corr_dict = {k:v['wcorr'].to_numpy() for k, v in a_decoder_ripple_weighted_corr_df_dict.items()}
            a_decoder_ripple_weighted_corr_df = pd.DataFrame(decoder_ripple_weighted_corr_dict) ## (n_epochs, 4)
            a_shuffle_wcorr_arr = a_decoder_ripple_weighted_corr_df.to_numpy()            
            _out_wcorr.append(a_shuffle_wcorr_arr)
        return np.stack(_out_wcorr) # .shape ## (n_shuffles, n_epochs, 4)
    

    def post_compute(self, curr_active_pipeline=None, debug_print:bool=False):
        """ Called after computing some shuffles.
        """
        ## Want ## (n_shuffles, n_epochs, 4)
                
        ## INPUTS: output_extracted_result_wcorrs_list, real_decoder_ripple_weighted_corr_arr
        if (self.curr_active_pipeline is None):
            if (curr_active_pipeline is None):
                raise NotImplementedError(f"cannot compute because self.curr_active_pipeline is missing and no curr_active_pipeline were provided as kwargs!")
            else:
                # non-None pipeline passed in, use for self
                self.curr_active_pipeline = curr_active_pipeline

        if (self.track_templates is None):
            ## recover them from `curr_active_pipeline`
            if self.curr_active_pipeline is not None:
                directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # used to get track_templates
                rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder'] # only used for `rank_order_results.minimum_inclusion_fr_Hz`
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
                self.track_templates = deepcopy(track_templates)
            else:
                raise NotImplementedError(f"cannot compute because self.track_templates is missing and no track_templates were provided as kwargs!")

        assert ((self.curr_active_pipeline is not None) and  (self.track_templates is not None))


        n_decoders: int = 4
        
        _out_wcorr = []
        _out_shuffle_is_more_extreme = []

        total_n_shuffles: int = len(self.output_extracted_result_wcorrs_list)
        print(f'total_n_shuffles: {total_n_shuffles}')

        ## USES: self.output_extracted_result_wcorrs_list, self.real_decoder_ripple_weighted_corr_arr

        for i, a_decoder_ripple_weighted_corr_df_dict in enumerate(self.output_extracted_result_wcorrs_list):
            decoder_ripple_weighted_corr_dict = {k:v['wcorr'].to_numpy() for k, v in a_decoder_ripple_weighted_corr_df_dict.items()}
            a_decoder_ripple_weighted_corr_df = pd.DataFrame(decoder_ripple_weighted_corr_dict) ## (n_epochs, 4)
            a_shuffle_wcorr_arr = a_decoder_ripple_weighted_corr_df.to_numpy()
            a_shuffle_is_more_extreme = np.abs(a_shuffle_wcorr_arr) > np.abs(self.real_decoder_ripple_weighted_corr_arr)
            
            _out_wcorr.append(a_shuffle_wcorr_arr)
            _out_shuffle_is_more_extreme.append(a_shuffle_is_more_extreme)


        # ==================================================================================================================== #
        # Process Outputs                                                                                                      #
        # ==================================================================================================================== #
        _out_shuffle_wcorr_arr = np.stack(_out_wcorr) # .shape ## (n_shuffles, n_epochs, 4)
        n_epochs: int = np.shape(_out_shuffle_wcorr_arr)[1]
        if debug_print:
            print(f'n_epochs: {n_epochs}')
        
        try:
            assert n_epochs == len(self.filtered_epochs_df), f"n_epochs: {n_epochs} != len(filtered_epochs_df): {len(self.filtered_epochs_df)}"
        except AssertionError:
            desired_ripple_decoding_time_bin_size: float = self.all_templates_decode_kwargs['desired_ripple_decoding_time_bin_size']
            minimum_event_duration: float = self.all_templates_decode_kwargs['minimum_event_duration']
            self.filtered_epochs_df = self.filtered_epochs_df[self.filtered_epochs_df['duration'] >= minimum_event_duration]
            assert n_epochs == len(self.filtered_epochs_df), f"EVEN AFTER TRYING TO FILTER: n_epochs: {n_epochs} != len(filtered_epochs_df): {len(self.filtered_epochs_df)}"


        _out_shuffle_is_more_extreme = np.stack(_out_shuffle_is_more_extreme) # .shape ## (n_shuffles, n_epochs, 4)
        if debug_print:
            print(f'np.shape(_out_shuffle_wcorr_arr): {np.shape(_out_shuffle_wcorr_arr)}')
            print(f'np.shape(_out_shuffle_is_more_extreme): {np.shape(_out_shuffle_is_more_extreme)}')

        total_n_shuffles_more_extreme_than_real = np.sum(_out_shuffle_is_more_extreme, axis=0) # sum only over the number of shuffles # (n_epochs, 4)
        if debug_print:
            print(f'np.shape(total_n_shuffles_more_extreme_than_real): {np.shape(total_n_shuffles_more_extreme_than_real)}')

        valid_shuffle_indicies = np.logical_not(np.isnan(_out_shuffle_wcorr_arr)) ## (n_shuffles, n_epochs, 4)
        n_valid_shuffles = np.sum(valid_shuffle_indicies, axis=0) # sum only over epochs to get n_shuffles for each epoch for each decoder # (n_epochs, 4)
        if debug_print:
            print(f'np.shape(n_valid_shuffles): {np.shape(n_valid_shuffles)}')

        _long_short_keys = ['long', 'short']

        total_n_shuffles_more_extreme_than_real_LSdict = {}

        for k in _long_short_keys:
            total_n_shuffles_more_extreme_than_real_LSdict[k] = np.array([total_n_shuffles_more_extreme_than_real[epoch_idx, decoder_idx] for epoch_idx, decoder_idx in enumerate(self.filtered_epochs_df[f'{k}_best_dir_decoder_IDX'].to_numpy())])

        _out_shuffle_wcorr_Zscore_val = np.zeros((n_epochs, 4)) # (n_epochs, 4)

        for epoch_idx in np.arange(n_epochs):    

            for decoder_idx in np.arange(n_decoders):
                a_single_decoder_epoch_all_shuffles_wcorr = np.squeeze(_out_shuffle_wcorr_arr[:, epoch_idx, decoder_idx]) # all shuffles and decoders for this epoch
                # a_single_decoder_epoch_z_scores = self.compute_z_scores(a_single_decoder_epoch_all_shuffles_wcorr)
                a_single_decoder_epoch_z_score: float = self.compute_z_score(a_single_decoder_epoch_all_shuffles_wcorr, self.real_decoder_ripple_weighted_corr_arr[epoch_idx, decoder_idx])
                # (n_shuffles, n_epochs, 4)
                _out_shuffle_wcorr_Zscore_val[epoch_idx, decoder_idx] = a_single_decoder_epoch_z_score
        # end for

        # compute_z_scores
        if debug_print:
            print(f'np.shape(_out_shuffle_wcorr_Zscore_val): {np.shape(_out_shuffle_wcorr_Zscore_val)}')

        _out_shuffle_wcorr_ZScore_LONG = np.array([_out_shuffle_wcorr_Zscore_val[epoch_idx, decoder_idx] for epoch_idx, decoder_idx in enumerate(self.filtered_epochs_df['long_best_dir_decoder_IDX'].to_numpy())]) # (n_epochs,)
        _out_shuffle_wcorr_ZScore_SHORT = np.array([_out_shuffle_wcorr_Zscore_val[epoch_idx, decoder_idx] for epoch_idx, decoder_idx in enumerate(self.filtered_epochs_df['short_best_dir_decoder_IDX'].to_numpy())]) # (n_epochs,)

        if debug_print:
            print(f'np.shape(_out_shuffle_wcorr_ZScore_LONG): {np.shape(_out_shuffle_wcorr_ZScore_LONG)}')
            print(f'np.shape(_out_shuffle_wcorr_ZScore_SHORT): {np.shape(_out_shuffle_wcorr_ZScore_SHORT)}')

        ## OUTPUTS: _out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT
        ## OUTPUTS: _out_shuffle_wcorr_arr_ZScores_LONG, _out_shuffle_wcorr_arr_ZScores_SHORT

        _out_p = total_n_shuffles_more_extreme_than_real.astype('float') / n_valid_shuffles.astype('float') # (n_epochs, 4)
        if debug_print:
            print(f'np.shape(_out_p): {np.shape(_out_p)}') # (640, 4) - (n_shuffles, 4)

        assert self.track_templates is not None
        total_n_shuffles_more_extreme_than_real_df: pd.DataFrame = pd.DataFrame(total_n_shuffles_more_extreme_than_real, columns=self.track_templates.get_decoder_names())
        total_n_shuffles_more_extreme_than_real_dict = dict(zip(self.track_templates.get_decoder_names(), total_n_shuffles_more_extreme_than_real.T))

        _out_p_dict = dict(zip(self.track_templates.get_decoder_names(), _out_p.T))

        ## INPUTS: filtered_epochs_df

        # epoch_start_t = self.filtered_epochs_df['start'].to_numpy() # ripple start time
        return (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), _out_shuffle_wcorr_arr


    def save_data(self, filepath):
        """ saves the important results to pickle """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

        try:
            # all_shuffles_wcorr_array = self.get_all_shuffles_wcorr_array()
            (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), all_shuffles_wcorr_array = self.post_compute()

        except (BaseException, AssertionError):
            all_shuffles_wcorr_array = []

        saveData(filepath, (self.output_extracted_result_wcorrs_list, self.real_decoder_ripple_weighted_corr_arr, self.output_all_shuffles_decoded_results_list, all_shuffles_wcorr_array))
        

    def save_data_mat(self, filepath, **additional_mat_elements):
        """ tries to export the wcorr_ripple_shuffle results to a .mat MATLAB file.
        Usage:

            standalone_mat_filename: str = f'{get_now_rounded_time_str()}_standalone_all_shuffles_wcorr_array.mat' 
            standalone_mat_filepath = curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve()
            wcorr_ripple_shuffle.save_data_mat(filepath=standalone_mat_filepath, **{'session': curr_active_pipeline.get_session_context().to_dict()})
        """
        from scipy.io import savemat
        if isinstance(filepath, str):
            filepath = Path(filepath).resolve()

        print(f'saving .mat file to "{filepath}"...')
        all_shuffles_wcorr_array: NDArray = deepcopy(self.all_shuffles_wcorr_array)
        mat_dic = {"all_shuffles_wcorr_array": all_shuffles_wcorr_array, "n_epochs": self.n_epochs, **additional_mat_elements}
        savemat(filepath, mat_dic)
        # return mat_dic
        
    


    @classmethod
    def init_from_data_only_file(cls, filepath, curr_active_pipeline, track_templates=None, directional_decoders_epochs_decode_result=None, global_epoch_name=None):
        """ loads previously saved results from a pickle file to rebuild
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle

        wcorr_tool: WCorrShuffle = WCorrShuffle.init_from_data_only_file(filepath=filepath, curr_active_pipeline=curr_active_pipeline, track_templates=track_templates,
                                                directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result,
                                                global_epoch_name=global_epoch_name, filepath='temp22.pkl')
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData

        output_extracted_result_wcorrs_list, real_decoder_ripple_weighted_corr_arr, output_all_shuffles_decoded_results_list = loadData(filepath)
        _obj = cls.init_from_templates(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates,
                                        directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result,
                                        global_epoch_name=global_epoch_name)
        _obj.output_extracted_result_wcorrs_list = output_extracted_result_wcorrs_list
        _obj.real_decoder_ripple_weighted_corr_arr = real_decoder_ripple_weighted_corr_arr
        _obj.output_all_shuffles_decoded_results_list = output_all_shuffles_decoded_results_list        
        return _obj


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        for a_non_pickleable_field in _non_pickled_fields:
            del state[a_non_pickleable_field]

        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
        
        _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
            if a_field_name not in state:
                state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from


    ## Plotting
    def plot_histogram_figure(self, a_decoder_idx: int = 0, selected_epoch_index: int = 0):
        """
        
        
        """
        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histogram_for_z_scores
        import matplotlib.pyplot as plt

        ## Get all shuffles for a single epoch:
        ## INPUTS: all_shuffles_wcorr_array, a_decoder_idx, selected_epoch_index

        ## start with one decoder:
        # a_decoder_name: types.DecoderName = 'long_LR'
        
        # _single_epoch_all_shuffles_wcorr_arr = _out_shuffle_wcorr_arr[:, selected_epoch_index]
        _single_epoch_all_shuffles_wcorr_arr = self.all_shuffles_wcorr_array[:, selected_epoch_index, a_decoder_idx]
        print(f'np.shape(_single_epoch_all_shuffles_wcorr_arr): {np.shape(_single_epoch_all_shuffles_wcorr_arr)}') # (n_shuffles, )

        _single_epoch_real_wcorr: float = self.real_decoder_ripple_weighted_corr_arr[selected_epoch_index, a_decoder_idx]

        a_single_decoder_epoch_z_scored_values: NDArray = self.compute_z_transformed_scores(_single_epoch_all_shuffles_wcorr_arr)
        a_single_decoder_epoch_z_score: float = self.compute_z_score(_single_epoch_all_shuffles_wcorr_arr, _single_epoch_real_wcorr)
        print(f'a_single_decoder_epoch_z_score: {a_single_decoder_epoch_z_score}')
        fig = plt.figure(num=f"histogram_for_z_scores - decoder[{a_decoder_idx}], epoch[{selected_epoch_index}]")
        # List of z-scored values
        z_scores = a_single_decoder_epoch_z_scored_values
        plot_histogram_for_z_scores(z_scores, title_suffix=f': decoder[{a_decoder_idx}], epoch[{selected_epoch_index}]')
        plt.axvline(a_single_decoder_epoch_z_score, color='red', linestyle='--', linewidth=2, label='Actual Value')
        return fig



@define(slots=False, repr=False, eq=False)
class SequenceBasedComputationsContainer(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer

        wcorr_shuffle_results: SequenceBasedComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('SequenceBased', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'SequenceBased is not computed.')
            
    """
    _VersionedResultMixin_version: str = "2024.05.27_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    wcorr_ripple_shuffle: Optional[WCorrShuffle] = serialized_field(default=None, repr=False)
    # RL_ripple: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    # LR_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    # RL_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)

    # ripple_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)
    # laps_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)

    # ripple_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    # ripple_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)
    # # ripple_n_valid_shuffles: Optional[int] = serialized_attribute_field(default=None, repr=False)

    # laps_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    # laps_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)

    # minimum_inclusion_fr_Hz: float = serialized_attribute_field(default=2.0, repr=True)
    # included_qclu_values: Optional[List] = serialized_attribute_field(default=None, repr=True)


    # Utility Methods ____________________________________________________________________________________________________ #

    # def to_dict(self) -> Dict:
    #     # return asdict(self, filter=attrs.filters.exclude((self.__attrs_attrs__.is_global))) #  'is_global'
    #     return {k:v for k, v in self.__dict__.items() if k not in ['is_global']}
    
    # def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, **kwargs):
    #     """ Saves the object to key in the hdf5 file specified by file_path
    #     enable_hdf_testing_mode: bool - default False - if True, errors are not thrown for the first field that cannot be serialized, and instead all are attempted to see which ones work.


    #     Usage:
    #         hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
    #         _pfnd_obj: PfND = long_one_step_decoder_1D.pf
    #         _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
    #     """
    #     super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, **kwargs)
    #     # handle custom properties here

    def __setstate__(self, state):
        # Restore instance attributes

        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        self.__dict__.update(state)




def validate_has_sequence_based_results(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
    """ Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    seq_results: SequenceBasedComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['SequenceBased']
    if seq_results is None:
        return False
    
    wcorr_ripple_shuffle: WCorrShuffle = seq_results.wcorr_ripple_shuffle
    if wcorr_ripple_shuffle is None:
        return False



class SequenceBasedComputationsGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to sequence-based decoding computations. """
    _computationGroupName = 'sequence_based'
    _computationGlobalResultGroupName = 'SequenceBased'
    _computationPrecidence = 1005
    _is_global = True

    @function_attributes(short_name='wcorr_shuffle_analysis', tags=['directional_pf', 'laps', 'wcorr', 'session', 'pf1D'], input_requires=['DirectionalLaps', 'RankOrder'], output_provides=['SequenceBased'], uses=['SequenceBasedComputationsContainer', 'WCorrShuffle'], used_by=[], creation_date='2024-05-27 14:31', related_items=[],
        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders', 'RankOrder', 'DirectionalDecodersEpochsEvaluations'], provides_global_keys=['SequenceBased'],
        validate_computation_test=validate_has_sequence_based_results, is_global=True)
    def perform_wcorr_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, num_shuffles:int=4096):
        """ Performs the computation of the spearman and pearson correlations for the ripple and lap epochs.

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['SequenceBased']
                ['SequenceBased'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps


        """
        if include_includelist is not None:
            print(f'WARN: perform_wcorr_shuffle_analysis(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        print(f'perform_wcorr_shuffle_analysis(..., num_shuffles={num_shuffles})')

        # Needs to store the parameters
        # num_shuffles:int=1000
        # minimum_inclusion_fr_Hz:float=12.0
        # included_qclu_values=[1,2]

        if ('SequenceBased' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'SequenceBased')):
            # initialize
            global_computation_results.computed_data['SequenceBased'] = SequenceBasedComputationsContainer(wcorr_ripple_shuffle=None, is_global=True)

        # global_computation_results.computed_data['SequenceBased'].included_qclu_values = included_qclu_values
        if (not hasattr(global_computation_results.computed_data['SequenceBased'], 'wcorr_ripple_shuffle') or (global_computation_results.computed_data['SequenceBased'].wcorr_ripple_shuffle is None)):
            # initialize a new wcorr result            
            wcorr_tool: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=owning_pipeline_reference, enable_saving_entire_decoded_shuffle_result=False)
            global_computation_results.computed_data['SequenceBased'].wcorr_ripple_shuffle = wcorr_tool
        else:
            ## get the existing one:
            wcorr_tool = global_computation_results.computed_data['SequenceBased'].wcorr_ripple_shuffle
        

        n_completed_shuffles: int = wcorr_tool.n_completed_shuffles

        if n_completed_shuffles < num_shuffles:   
            print(f'n_prev_completed_shuffles: {n_completed_shuffles}.')
            print(f'needed num_shuffles: {num_shuffles}.')
            desired_new_num_shuffles: int = max((num_shuffles - wcorr_tool.n_completed_shuffles), 0)
            print(f'need desired_new_num_shuffles: {desired_new_num_shuffles} more shuffles.')
            ## add some more shuffles to it:
            wcorr_tool.compute_shuffles(num_shuffles=desired_new_num_shuffles, curr_active_pipeline=owning_pipeline_reference)

        # (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict) = wcorr_tool.post_compute(debug_print=False)
        # wcorr_tool.save_data(filepath='temp100.pkl')

        global_computation_results.computed_data['SequenceBased'].wcorr_ripple_shuffle = wcorr_tool
        

        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer

        wcorr_shuffle_results: SequenceBasedComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('SequenceBased', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'SequenceBased is not computed.')
            
        """
        return global_computation_results
    

   


# ==================================================================================================================== #
# Display Function Helpers                                                                                             #
# ==================================================================================================================== #

# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

# # ==================================================================================================================== #
# # Display Functions                                                                                                    #
# # ==================================================================================================================== #

# class SequenceBasedGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
#     """ RankOrderGlobalDisplayFunctions
#     These display functions compare results across several contexts.
#     Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
#     """
