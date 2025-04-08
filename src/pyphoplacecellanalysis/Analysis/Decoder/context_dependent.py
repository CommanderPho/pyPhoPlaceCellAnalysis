from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from attrs import define, field, Factory, asdict, astuple
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Iterable
from typing_extensions import TypeAlias
from numpy.typing import NDArray  # Correct import for NDArray
from typing import NewType
import neuropy.utils.type_aliases as types
# DecoderName = NewType('DecoderName', str)

from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from neuropy.utils.indexing_helpers import NumpyHelpers
from pyphocorehelpers.assertion_helpers import Assert

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.epoch import NamedTimerange, Epoch, ensure_dataframe
from neuropy.core.epoch import find_data_indicies_from_epoch_times
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr, shape_only_repr, array_values_preview_repr
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.utils.indexing_helpers import PandasHelpers, NumpyHelpers, flatten
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs
from neuropy.core.epoch import Epoch, TimeColumnAliasesProtocol, subdivide_epochs, ensure_dataframe, ensure_Epoch
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult # needed in DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult, DirectionalPseudo2DDecodersResult, EpochFilteringMode, _compute_proper_filter_epochs
from pyphocorehelpers.indexing_helpers import partition_df_dict, partition_df
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

# import scipy.stats
# from scipy import ndimage

# from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

# from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability

# decoder_name_str: TypeAlias = str # an string name of a particular decoder, such as 'Long_LR' or 'Short_RL'

# from pyphocorehelpers.programming_helpers import metadata_attributes
# from pyphocorehelpers.function_helpers import function_attributes

# if TYPE_CHECKING:
#     ## typehinting only imports here
#     # from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, SingleEpochDecodedResult #typehinting only
#     # from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
#     from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig


# # DecodedMarginalResultTuple: TypeAlias = Tuple[List[DynamicContainer], NDArray[float], NDArray[int], NDArray[bool]] # used by 
# DecodedMarginalResultTuple: TypeAlias = Tuple[
#     List[DynamicContainer],
#     NDArray[np.float_],
#     NDArray[np.int_],
#     NDArray[np.bool_]
# ]

# # 2025-02-21 - I realized that `TrackTemplates` and `DirectionalLapsResult` are pointlessly hard-coded to require directionality, but a more general solution would be just to allow dicts of decoders.
#TODO 2025-03-07 18:38: - [ ] Generalize `DecoderDecodedEpochsResult`


#TODO 2025-03-08 13:57: - [ ] Final Output will be: a decoded posterior 
# - used to decode:
#   - LapEpochs, RippleEpochs, ContinuousEpochs (not needed)
# - computed using:
#   - RunningDecoder (Only-laps), NonPBEDecoder (Any non-PBE period)
#   - 

""" 

    ##Gotta get those ['P_LR', 'P_RL'] columns to determine best directions
    extracted_merged_scores_df: pd.DataFrame = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df()
    
    extracted_merged_scores_df = extracted_merged_scores_df.loc[:, ~extracted_merged_scores_df.columns.duplicated()] # drops the duplicate columns, keeping only the first instance
    extracted_merged_scores_df['is_most_likely_direction_LR'] = (extracted_merged_scores_df['P_LR'] > 0.5) # ValueError: Cannot set a DataFrame with multiple columns to the single column is_most_likely_direction_LR. Have duplicate columns for 'P_LR' unfortunately.

        
        
    ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates)

    ## run 'directional_decoders_epoch_heuristic_scoring',
    directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=True)

    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    _output_csv_paths = directional_decoders_epochs_decode_result.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=curr_session_name, curr_session_t_delta=t_delta,
                                                                              user_annotation_selections={'ripple': any_good_selected_epoch_times},
                                                                              valid_epochs_selections={'ripple': filtered_valid_epoch_times},
                                                                              custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
                                                                              should_export_complete_all_scores_df=True, export_df_variable_names=[], # `export_df_variable_names=[]` means export no non-complete dfs
                                                                              )
"""


from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
from neuropy.analyses.placefields import PfND
# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPlacefieldGlobalComputationFunctions, DirectionalLapsResult, TrackTemplates, DecoderDecodedEpochsResult

from neuropy.utils.result_context import IdentifyingContext
from typing import Literal
# Define a type that can only be one of these specific strings
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs, GeneralDecoderDictDecodedEpochsDictResult, KnownFilterEpochs, NonPBEDimensionalDecodingResult ## #TODO 2025-03-11 08:15: - [ ] Actually look into this class instead of the literal

import pyphoplacecellanalysis.General.type_aliases as types


""" Possible IdentifyingContext keys:
All entries are {IdentifyingContextKeyName} | {ImplicitButUnenforcedKeyValueType} | Description with possible values
trained_compute_epochs | KnownNamedDecoderTrainedComputeEpochsType | the actual epochs that are used to build the decoder. (e.g. 'laps', 'non_pbe') -- NOT 'pbe' as PBEs were never used to build the decoder. I think these should be actual epochs objects instead of named epochs... but nah they don't have to be.
pfND_ndim | int | number of spatial dimensions the decoder will have (e.g. 1 or 2)
decoder_identifier | types.DecoderName | name of a decoder ['long_LR', 'long', etc]
time_bin_size | float | decoding time bin size (e.g. 0.025, 0.058)
known_named_decoding_epochs_type | KnownNamedDecodingEpochsType | the name of a known set of Epochs (e.g. 'laps', 'replay', 'ripple', 'pbe', 'non_pbe')
masked_time_bin_fill_type | MaskedTimeBinFillType | used in `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(...)` to specify how invalid bins (due to too few spikes) are treated. (e.g. 'ignore', 'last_valid', 'nan_filled', 'dropped')
data_grain | DataTimeGrain | how the data is binned in time (e.g. "data_grain='per_epoch'" or "data_grain='per_time_bin'") Note: used in `pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.SpecificPrePostDeltaScatter._pre_post_delta_scatter_laps_per_time_bin`

"""

# @metadata_attributes(short_name=None, tags=['Generic', 'Improved'], input_requires=[], output_provides=[], uses=[], used_by=['generalized_decode_epochs_dict_and_export_results_completion_function'], creation_date='2025-03-11 07:53', related_items=['GeneralDecoderDictDecodedEpochsDictResult'])
@define(slots=False, eq=False)
class GenericDecoderDictDecodedEpochsDictResult(ComputedResult):
    """ General dict-based class that uses IdentifyingContext (unordered-dict-like) for keys into multiple result dicts. (flat-unordered-tuple-like indicies)
    ** Just accumulates results and extracts them from previous computed results **
    
    Means to store general results and export them easily to .CSV (FAT_df) or separate classical .CSVs
    
    
    from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType
    
    Info:
        ### Makes masked versions of all previous results

        ## Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation:
        _new_results_to_add = {} ## need a temporary entry so we aren't modifying the dict property `a_new_fully_generic_result.filter_epochs_pseudo2D_continuous_specific_decoded_result` while we update it
        for a_context, a_decoded_filter_epochs_result in a_new_fully_generic_result.filter_epochs_pseudo2D_continuous_specific_decoded_result.items():
            a_modified_context = deepcopy(a_context)
            a_masked_decoded_filter_epochs_result, _mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
            a_modified_context = a_modified_context.adding_context_if_missing(masked_time_bin_fill_type='last_valid')
            _new_results_to_add[a_modified_context] = a_masked_decoded_filter_epochs_result
        
        print(f'computed {len(_new_results_to_add)} new results')

        a_new_fully_generic_result.filter_epochs_pseudo2D_continuous_specific_decoded_result.update(_new_results_to_add)

        
    
    COMPLETE EXAMPLE:
    
    
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType

        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult()  # start empty

        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = a_new_fully_generic_result.adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(a_general_decoder_dict_decoded_epochs_dict_result=a_general_decoder_dict_decoded_epochs_dict_result)

        # a_new_fully_generic_result
        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL
        a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)

        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
        a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_merged_decoders_result=directional_merged_decoders_result)

        spikes_df = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        a_new_fully_generic_result = a_new_fully_generic_result.creating_new_spikes_per_t_bin_masked_variants(spikes_df=spikes_df)

        a_new_fully_generic_result
    
    """
    _VersionedResultMixin_version: str = "2025.03.11_0" # to be updated in your IMPLEMENTOR to indicate its version

    ## Flat/Common:
    pos_df: pd.DataFrame = serialized_field(default=None, repr=None)
    spikes_df_dict: Dict[types.GenericResultTupleIndexType, pd.DataFrame] = serialized_field(default=Factory(dict), repr=keys_only_repr, metadata={'field_added': "2025.03.11_0"}) # global

    # decoder_trained_compute_epochs_dict: Dict[GenericResultTupleIndexType, Epoch] = serialized_field(default=None, metadata={'field_added': "2025.03.11_0"}) ## Is this needed, or are they present in the computation_epochs in the decoder/pf?

    # original_pfs_dict: Dict[types.DecoderName, PfND]
    decoders: Dict[types.GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=Factory(dict), repr=keys_only_repr, metadata={'field_added': "2025.03.11_0"}) # Combines both pf1D_Decoder_dict and pseudo2D_decoder-type decoders (both 1D and 2D as well) by indexing via context
    # pf1D_Decoder_dict: Dict[GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=None, metadata={'field_added': "2024.01.16_0"})
    # pseudo2D_decoder: Dict[GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=None, metadata={'field_added': "2024.01.22_0"})
    
    ## Result Keys
    filter_epochs_to_decode_dict: Dict[types.GenericResultTupleIndexType, Epoch] = serialized_field(default=Factory(dict), repr=keys_only_repr)
    filter_epochs_specific_decoded_result: Dict[types.GenericResultTupleIndexType, DecodedFilterEpochsResult] = serialized_field(default=Factory(dict), repr=keys_only_repr) ## why is this labeled as if they have to be continuous or Pseudo2D? They can be any result right?
    filter_epochs_decoded_track_marginal_posterior_df_dict: Dict[types.GenericResultTupleIndexType, pd.DataFrame] = serialized_field(default=Factory(dict), repr=keys_only_repr)


    should_use_flat_context_mode: bool = serialized_attribute_field(default=True, is_computable=False, repr=True)
    

    @property
    def single_FAT_df(self) -> pd.DataFrame:
        """
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType
        
        """
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
        return SingleFatDataframe.build_fat_df(dfs_dict=self.filter_epochs_decoded_track_marginal_posterior_df_dict)

    # ================================================================================================================================================================================ #
    # Additive from old results objects                                                                                                                                                #
    # ================================================================================================================================================================================ #
    @classmethod
    def init_from_old_GeneralDecoderDictDecodedEpochsDictResult(cls, a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ converts the 2025-03-10 result (`GeneralDecoderDictDecodedEpochsDictResult`) to the 2025-03-11 format (`GenericDecoderDictDecodedEpochsDictResult`)

        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType

        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.init_from_old_GeneralDecoderDictDecodedEpochsDictResult(a_general_decoder_dict_decoded_epochs_dict_result)
        a_new_fully_generic_result
        """
        a_new_fully_generic_result = GenericDecoderDictDecodedEpochsDictResult() # start empty
        _shared_context_fragment = {'trained_compute_epochs': 'non_pbe'}
        for a_known_epoch_name, an_epoch in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_to_decode_dict.items():
            a_new_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name)
            a_new_fully_generic_result.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(an_epoch)
            
            a_new_fully_generic_result.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_pseudo2D_continuous_specific_decoded_result[a_known_epoch_name])
            for a_known_t_bin_fill_type, a_posterior_df in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_known_epoch_name].items():
                a_new_joint_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name, masked_time_bin_fill_type=a_known_t_bin_fill_type)
                a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[a_new_joint_identifier] = deepcopy(a_posterior_df)

        return a_new_fully_generic_result



    # @function_attributes(short_name=None, tags=['adding'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 14:40', related_items=[])
    def adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(self, a_general_decoder_dict_decoded_epochs_dict_result, flat_contexts: bool=True, debug_print: bool=True) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ adds the results from a 2025-03-10 result (`GeneralDecoderDictDecodedEpochsDictResult`) to the new 2025-03-11 format (`GenericDecoderDictDecodedEpochsDictResult`), in-place, but returning itself

        
        Updates: .filter_epochs_to_decode_dict, .filter_epochs_specific_decoded_result, .filter_epochs_decoded_track_marginal_posterior_df_dict
        
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType

        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult()  # start empty
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = a_new_fully_generic_result.adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(a_general_decoder_dict_decoded_epochs_dict_result=a_general_decoder_dict_decoded_epochs_dict_result)
        a_new_fully_generic_result
        """
        
        if debug_print:
            print(f'.adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(...):')
            
        _shared_context_fragment = {'trained_compute_epochs': 'non_pbe'}
        for a_known_epoch_name, an_epoch in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_to_decode_dict.items():
            a_new_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name)
            self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(an_epoch)
            self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_pseudo2D_continuous_specific_decoded_result[a_known_epoch_name])
            ## Marginal dataframes:
            for a_known_t_bin_fill_type, a_posterior_df in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_known_epoch_name].items():
                a_new_joint_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name, masked_time_bin_fill_type=a_known_t_bin_fill_type)
                if flat_contexts:
                    self.filter_epochs_to_decode_dict[a_new_joint_identifier] = deepcopy(an_epoch)
                    self.filter_epochs_specific_decoded_result[a_new_joint_identifier] = deepcopy(a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_pseudo2D_continuous_specific_decoded_result[a_known_epoch_name])
                    ## #TODO 2025-03-21 09:19: - [ ] unused contexts now in 
                    # del self.filter_epochs_to_decode_dict[a_new_identifier]
                    # del self.filter_epochs_specific_decoded_result[a_new_identifier]
                    
                self.filter_epochs_decoded_track_marginal_posterior_df_dict[a_new_joint_identifier] = deepcopy(a_posterior_df)

        return self
    

    # @function_attributes(short_name=None, tags=['adding'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 13:16', related_items=[])
    def adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(self, directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult, masked_bin_fill_mode = 'nan_filled', debug_print: bool=True) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL

        # a_new_fully_generic_result = _subfn_filter_by_spikes_per_t_bin_masked_and_add_to_generic_result(a_new_fully_generic_result, directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)
        a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)

        """        
        if debug_print:
            print(f'.adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(...):')
            
        filtered_epochs_df = None ## parameter just to allow an override I think
         
        decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict)
        decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict)

        # ==================================================================================================================== #
        # Reuse the old/previously computed version of the result with the additional properties                               #
        # ==================================================================================================================== #
        ## These are of type `trained_compute_epochs` -- e.g. trained_compute_epochs='laps'  || trained_compute_epochs='non_pbe'
        # trained_compute_epochs_dict_dict = {'laps': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## only laps were ever trained to decode until the non-PBEs
        #                                     # 'pbe': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## PBEs were never used to decode, only laps
        # }

        ## NOTE that this is only done for the "trained_compute_epochs='laps'" context
        decoder_filter_epochs_result_dict_dict = {'laps': deepcopy(decoder_laps_filter_epochs_decoder_result_dict),
                                                  'pbe': deepcopy(decoder_ripple_filter_epochs_decoder_result_dict),
        }

        epochs_decoding_time_bin_size_dict = {'laps': directional_decoders_epochs_decode_result.laps_decoding_time_bin_size,
                                              'pbe': directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size,
        }

        filtered_decoder_filter_epochs_decoder_result_dict_dict = {}

        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result_dict in decoder_filter_epochs_result_dict_dict.items():

            unfiltered_epochs_df = deepcopy(a_decoder_epochs_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)
            if filtered_epochs_df is not None:
                ## filter
                filtered_epochs_df = ensure_dataframe(filtered_epochs_df)
                filtered_decoder_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) for a_name, a_result in a_decoder_epochs_filter_epochs_decoder_result_dict.items()} # working filtered
            else:
                unfiltered_epochs_df = ensure_dataframe(unfiltered_epochs_df)
                filtered_decoder_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(ensure_dataframe(unfiltered_epochs_df)[['start', 'stop']].to_numpy()) for a_name, a_result in a_decoder_epochs_filter_epochs_decoder_result_dict.items()} # working unfiltered

            ## collect results:
            filtered_decoder_filter_epochs_decoder_result_dict_dict[a_known_decoded_epochs_type] = filtered_decoder_filter_epochs_decoder_result_dict
        ## END for a_known_decoded_epochs_type, a_decoder_...
        
        Assert.all_equal(epochs_decoding_time_bin_size_dict.values())
        epochs_decoding_time_bin_size: float = list(epochs_decoding_time_bin_size_dict.values())[0]
        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        if debug_print:
            print(f'{pos_bin_size = }, {epochs_decoding_time_bin_size = }')


        ## OUTPUTS: filtered_decoder_filter_epochs_decoder_result_dict_dict, epochs_decoding_time_bin_size

        ## Perform the decoding and masking as needed for invalid bins:
        
        spikes_df: pd.DataFrame = deepcopy(list(self.spikes_df_dict.values())[0])

        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result_dict in decoder_filter_epochs_result_dict_dict.items():

            for a_decoder_name, a_decoded_epochs_result in a_decoder_epochs_filter_epochs_decoder_result_dict.items():
                ## build the complete identifier
                a_new_identifier: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier=a_decoder_name, time_bin_size=epochs_decoding_time_bin_size, known_named_decoding_epochs_type=a_known_decoded_epochs_type, masked_time_bin_fill_type='ignore')
                if debug_print:
                    print(f'a_new_identifier: "{a_new_identifier}"')
                
                self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_decoded_epochs_result)
                self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(a_decoded_epochs_result.filter_epochs) ## needed? Do I want full identifier as key?

                ## add the filtered versions down here:
                a_decoded_filter_epochs_result = self.filter_epochs_specific_decoded_result[a_new_identifier] ## this line shouldn't have to be in the try if `self.get_matching_contexts(...)` works right, but for now it is
                a_modified_context = deepcopy(a_new_identifier)
                a_spikes_df = deepcopy(spikes_df)
                a_masked_decoded_filter_epochs_result, _mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=a_spikes_df, masked_bin_fill_mode=masked_bin_fill_mode)
                # a_modified_context = a_modified_context.adding_context_if_missing(masked_time_bin_fill_type='last_valid')
                a_modified_context = a_modified_context.overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
                if debug_print:
                    print(f'a_modified_context: "{a_modified_context}"')
                assert a_modified_context != a_new_identifier
                self.filter_epochs_specific_decoded_result[a_modified_context] = deepcopy(a_masked_decoded_filter_epochs_result)
                self.filter_epochs_to_decode_dict[a_modified_context] = deepcopy(a_masked_decoded_filter_epochs_result.filter_epochs) ## needed? Do I want full identifier as key?

                # a_new_fully_generic_result.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_new_identifier] ## #TODO 2025-03-11 11:39: - [ ] must be computed or assigned from prev result
                


        # directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug
                                                                                    
        # extracted_merged_scores_df = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug_print=True)
        # extracted_merged_scores_df
        ## Inputs: a_new_fully_generic_result

        return self



    # @function_attributes(short_name=None, tags=['adding', 'pseudo2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 14:24', related_items=[])
    def adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(self, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, masked_bin_fill_mode = 'nan_filled', debug_print: bool=True) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self

        Usage:
            directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
            a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_merged_decoders_result=directional_merged_decoders_result)

        """
        if debug_print:
            print(f'.adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(...):')
        filtered_epochs_df = None ## parameter just to allow an override I think
        
        # DirectionalMergedDecoders: Get the result after computation:
        
        ## NOTE, HAVE:
        all_directional_pf1D_Decoder: BasePositionDecoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
        # all_directional_decoder_dict: Dict[str, PfND] = directional_merged_decoders_result.all_directional_decoder_dict ## AHHH these are PfND, NOT BasePositionDecoder
        # Posteriors computed via the all_directional decoder:
        all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
        all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result
        
        # # Marginalized posteriors computed from above posteriors:
        # laps_directional_marginals_tuple: Tuple = directional_merged_decoders_result.laps_directional_marginals_tuple # laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
        # laps_track_identity_marginals_tuple: Tuple = directional_merged_decoders_result.laps_track_identity_marginals_tuple
        # # laps_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Laps'})
        
        # ripple_directional_marginals_tuple: Tuple = directional_merged_decoders_result.ripple_directional_marginals_tuple
        # ripple_track_identity_marginals_tuple: Tuple = directional_merged_decoders_result.ripple_track_identity_marginals_tuple
        # # ripple_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Replay'})
    

        # ==================================================================================================================== #
        # Reuse the old/previously computed version of the result with the additional properties                               #
        # ==================================================================================================================== #
        ## These are of type `trained_compute_epochs` -- e.g. trained_compute_epochs='laps'  || trained_compute_epochs='non_pbe'
        # trained_compute_epochs_dict_dict = {'laps': ensure_Epoch(deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs)), ## only laps were ever trained to decode until the non-PBEs
        #                                     # 'pbe': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## PBEs were never used to decode, only laps
        # }

        ## NOTE that this is only done for the "trained_compute_epochs='laps'" context
        decoder_filter_epochs_result_dict = {'laps': deepcopy(all_directional_laps_filter_epochs_decoder_result),
                                                'pbe': deepcopy(all_directional_ripple_filter_epochs_decoder_result),
        }

        epochs_decoding_time_bin_size_dict = {'laps': directional_merged_decoders_result.laps_decoding_time_bin_size,
                                                'pbe': directional_merged_decoders_result.ripple_decoding_time_bin_size,
        }


        decoder_epoch_marginals_df_dict_dict = {'laps': {'per_time_bin': deepcopy(directional_merged_decoders_result.laps_time_bin_marginals_df), 'per_epoch': deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df)},
                                                'pbe': {'per_time_bin': deepcopy(directional_merged_decoders_result.ripple_time_bin_marginals_df), 'per_epoch': deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)},
        }


        # dict(data_grain='per_epoch', known_named_decoding_epochs_type='laps'): deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df),
        # dict(data_grain='per_time_bin', known_named_decoding_epochs_type='laps'): deepcopy(directional_merged_decoders_result.laps_time_bin_marginals_df), 
        # ...

        filtered_decoder_filter_epochs_decoder_result_dict = {}
        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result in decoder_filter_epochs_result_dict.items():

            filtered_decoder_filter_epochs_decoder_result = None
            unfiltered_epochs_df = deepcopy(a_decoder_epochs_filter_epochs_decoder_result.filter_epochs)
            if filtered_epochs_df is not None:
                ## filter
                filtered_epochs_df = ensure_dataframe(filtered_epochs_df)
                filtered_decoder_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_decoder_epochs_filter_epochs_decoder_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) # working filtered
            else:
                unfiltered_epochs_df = ensure_dataframe(unfiltered_epochs_df)
                filtered_decoder_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_decoder_epochs_filter_epochs_decoder_result.filtered_by_epoch_times(ensure_dataframe(unfiltered_epochs_df)[['start', 'stop']].to_numpy()) # working unfiltered

            assert filtered_decoder_filter_epochs_decoder_result is not None
            
            ## collect results:
            filtered_decoder_filter_epochs_decoder_result_dict[a_known_decoded_epochs_type] = filtered_decoder_filter_epochs_decoder_result
        ## END for a_known_decoded_epochs_type, a_....
        
        Assert.all_equal(epochs_decoding_time_bin_size_dict.values())
        epochs_decoding_time_bin_size: float = list(epochs_decoding_time_bin_size_dict.values())[0]
        # pos_bin_size: float = directional_merged_decoders_result.pos_bin_size
        # print(f'{pos_bin_size = }, {epochs_decoding_time_bin_size = }')

        ## OUTPUTS: filtered_decoder_filter_epochs_decoder_result_dict_dict, epochs_decoding_time_bin_size

        ## Perform the decoding and masking as needed for invalid bins:
        a_decoder_name: str = 'pseudo2D' # FIXED FOR THIS ENTIRE FUNCTION
        a_base_identifier: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier=a_decoder_name, time_bin_size=epochs_decoding_time_bin_size)
        self.decoders[a_base_identifier] = deepcopy(all_directional_pf1D_Decoder) ## add 'pseudo2D' decoder
        
        spikes_df: pd.DataFrame = deepcopy(list(self.spikes_df_dict.values())[0])


        ## Updates `self.filter_epochs_specific_decoded_result`, `self.filter_epochs_to_decode_dict`
        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result in decoder_filter_epochs_result_dict.items():
            ## build the complete identifier
            a_new_identifier: IdentifyingContext = a_base_identifier.adding_context_if_missing(known_named_decoding_epochs_type=a_known_decoded_epochs_type, masked_time_bin_fill_type='ignore') # IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier=a_decoder_name, time_bin_size=epochs_decoding_time_bin_size, known_named_decoding_epochs_type=a_known_decoded_epochs_type, masked_time_bin_fill_type='ignore')
            if debug_print:
                print(f'a_new_identifier: "{a_new_identifier}"')
            
            self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_decoder_epochs_filter_epochs_decoder_result)
            # self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(a_decoder_epochs_filter_epochs_decoder_result.filter_epochs) ## needed? Do I want full identifier as key?
            ## use the filtered approach instead:
            self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(filtered_decoder_filter_epochs_decoder_result_dict[a_known_decoded_epochs_type])


            ## Updates `self.decoders` with the four individual decoders
            # for an_individual_decoder_name, an_individual_directional_decoder in all_directional_decoder_dict.items():
            #     a_new_individual_decoder_identifier: IdentifyingContext = deepcopy(a_new_identifier).overwriting_context(decoder_identifier=an_individual_decoder_name) # replace 'decoder_identifier'
            #     if debug_print:
            #         print(f'\t a_new_individual_decoder_identifier: "{a_new_individual_decoder_identifier}"')
            #     assert a_new_individual_decoder_identifier not in self.decoders
            #     self.decoders[a_new_individual_decoder_identifier] = deepcopy(an_individual_directional_decoder) 

            ## Updates `self.filter_epochs_decoded_track_marginal_posterior_df_dict`

            # self.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_new_identifier] ## #TODO 2025-03-11 11:39: - [ ] must be computed or assigned from prev result
            self.decoders[a_new_identifier] = deepcopy(all_directional_pf1D_Decoder) ## this will duplicate this decoder needlessly for each repetation here, but that's okay for now
            for a_known_data_grain, a_decoded_marginals_df in decoder_epoch_marginals_df_dict_dict[a_known_decoded_epochs_type].items():
                a_new_data_grain_identifier: IdentifyingContext = deepcopy(a_new_identifier).overwriting_context(data_grain=a_known_data_grain)
                if debug_print:
                    print(f'\t a_new_data_grain_identifier: "{a_new_data_grain_identifier}"')
                assert a_new_data_grain_identifier not in self.filter_epochs_decoded_track_marginal_posterior_df_dict
                self.filter_epochs_decoded_track_marginal_posterior_df_dict[a_new_data_grain_identifier] = deepcopy(a_decoded_marginals_df) 

                # TODO 2025-03-20 09:00: - [ ] New Version ___________________________________________________________________________ #
                ## INPUTS: a_result, masked_bin_fill_mode
                a_masked_updated_context: IdentifyingContext = deepcopy(a_new_data_grain_identifier).overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
                if debug_print:
                    print(f'a_masked_updated_context: {a_masked_updated_context}')
                
                ## MASKED with NaNs (no backfill):
                a_dropping_masked_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_decoder_epochs_filter_epochs_decoder_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode=masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.
                ## Computes marginals for `dropping_masked_laps_pseudo2D_continuous_specific_decoded_result`
                a_dropping_masked_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_dropping_masked_pseudo2D_continuous_specific_decoded_result, marginal_context=a_masked_updated_context)
                _was_update_success = self.updating_results_for_context(new_context=a_masked_updated_context, a_result=deepcopy(a_dropping_masked_pseudo2D_continuous_specific_decoded_result), a_decoder=deepcopy(all_directional_pf1D_Decoder), a_decoded_marginal_posterior_df=deepcopy(a_dropping_masked_decoded_marginal_posterior_df)) ## update using the result
                if not _was_update_success:
                    print(f'update failed for masked context: {a_masked_updated_context}')
            ## END for a_known_data_g...

            #TODO 2025-03-20 09:16: - [ ] These aren't time_grain specific?!?
            ## add the filtered versions down here:
            a_decoded_filter_epochs_result = self.filter_epochs_specific_decoded_result[a_new_identifier] ## this line shouldn't have to be in the try if `self.get_matching_contexts(...)` works right, but for now it is
            a_modified_context = deepcopy(a_new_identifier)
            a_spikes_df = deepcopy(spikes_df)
            a_masked_decoded_filter_epochs_result, _mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=a_spikes_df, masked_bin_fill_mode=masked_bin_fill_mode)
            # a_modified_context = a_modified_context.adding_context_if_missing(masked_time_bin_fill_type='last_valid')
            a_modified_context = a_modified_context.overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
            if debug_print:
                print(f'a_modified_context: "{a_modified_context}"')
            assert a_modified_context != a_new_identifier
            self.filter_epochs_specific_decoded_result[a_modified_context] = deepcopy(a_masked_decoded_filter_epochs_result)
            self.filter_epochs_to_decode_dict[a_modified_context] = deepcopy(a_masked_decoded_filter_epochs_result.filter_epochs) ## needed? Do I want full identifier as key?
            

            # # TODO 2025-03-20 09:00: - [ ] New Version ___________________________________________________________________________ #
            # ## INPUTS: a_result, masked_bin_fill_mode
            # a_masked_updated_context: IdentifyingContext = deepcopy(a_new_identifier).overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
            # if debug_print:
            #     print(f'a_masked_updated_context: {a_masked_updated_context}')
            
            # ## MASKED with NaNs (no backfill):
            # a_dropping_masked_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode=masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.
            # ## Computes marginals for `dropping_masked_laps_pseudo2D_continuous_specific_decoded_result`
            # a_dropping_masked_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_dropping_masked_pseudo2D_continuous_specific_decoded_result, marginal_context=a_masked_updated_context)
            # _was_update_success = self.updating_results_for_context(new_context=a_masked_updated_context, a_result=deepcopy(a_dropping_masked_pseudo2D_continuous_specific_decoded_result), a_decoder=deepcopy(all_directional_pf1D_Decoder), a_decoded_marginal_posterior_df=deepcopy(a_dropping_masked_decoded_marginal_posterior_df)) ## update using the result
            # if not _was_update_success:
            #     print(f'update failed for masked context: {a_masked_updated_context}')
        ## END for a_known_decode...

        # directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug
                                                                                    
        # extracted_merged_scores_df = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug_print=True)
        # extracted_merged_scores_df
        ## Inputs: a_new_fully_generic_result

        return self
    


    # ================================================================================================================================================================================ #
    # Self-compute                                                                                                                                                  #
    # ================================================================================================================================================================================ #
    
    # @function_attributes(short_name=None, tags=['adding', 'generating', 'masking'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 13:22', related_items=[])
    def creating_new_spikes_per_t_bin_masked_variants(self, spikes_df: pd.DataFrame, a_target_context: Optional[IdentifyingContext]=None) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self
        
        spikes_df = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        a_new_fully_generic_result = a_new_fully_generic_result.creating_new_spikes_per_t_bin_masked_variants(spikes_df=spikes_df)
        """
        _new_results_to_add = {} ## need a temporary entry so we aren't modifying the dict property `a_new_fully_generic_result.filter_epochs_specific_decoded_result` while we update it
        # 
        
        ## get data_grain='per_time_bin' results only
        if (a_target_context is None):
            a_target_context: IdentifyingContext = IdentifyingContext(data_grain='per_time_bin') # , masked_time_bin_fill_type='ignore', decoder_identifier='long_LR'
            
        flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = self.get_results_matching_contexts(context_query=a_target_context, return_multiple_matches=True, debug_print=False)
        flat_context_list

        masked_bin_fill_mode = 'nan_filled'
        for a_context in flat_context_list:
            try:
                a_decoded_filter_epochs_result = flat_result_context_dict[a_context] ## this line shouldn't have to be in the try if `self.get_matching_contexts(...)` works right, but for now it is
                a_modified_context = deepcopy(a_context)
                a_spikes_df = deepcopy(spikes_df)
                a_masked_decoded_filter_epochs_result, _mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=a_spikes_df, masked_bin_fill_mode=masked_bin_fill_mode)
                # a_modified_context = a_modified_context.adding_context_if_missing(masked_time_bin_fill_type='last_valid')
                a_modified_context = a_modified_context.overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
                _new_results_to_add[a_modified_context] = a_masked_decoded_filter_epochs_result
                ## can directly add the others that we aren't iterating over
                self.spikes_df_dict[a_modified_context] = deepcopy(a_spikes_df) ## TODO: reduce the context?                
            except (IndexError, KeyError) as e:
                print(f'IndexError: {e}. Skipping .creating_new_spikes_per_t_bin_masked_variants(...) for a_context: {a_context}.')
                pass
            except Exception as e:
                raise e

        print(f'\tcomputed {len(_new_results_to_add)} new results')

        self.filter_epochs_specific_decoded_result.update(_new_results_to_add)

        return self


    # ==================================================================================================================== #
    # Fresh Compute without relying on extant properties                                                                   #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['UNFINISHED', 'UNTESTED', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 00:00', related_items=[])
    def example_compute_fn(self, curr_active_pipeline, context: IdentifyingContext):
        """ Uses the context to extract proper values from the pipeline, and performs a fresh computation
        
        Usage:
            _out = a_new_fully_generic_result.example_compute_fn(curr_active_pipeline=curr_active_pipeline, context=IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='laps'))
            
        
        History:
            from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.Compute_NonPBE_Epochs.recompute`
        
        """
        initial_context_dict: Dict = deepcopy(context.to_dict())
        final_output_context_dict: Dict = {}
        
        # from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.Compute_NonPBE_Epochs.recompute`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # 'pfND_ndim' ________________________________________________________________________________________________________ #
        pfND_ndim: int = initial_context_dict.pop('pfND_ndim', 1)
        if pfND_ndim == 1:
             ## Uses 1D Placefields
            print(f'Uses 1D Placefields')
            long_pfND, short_pfND, global_pfND = long_results.pf1D, short_results.pf1D, global_results.pf1D
        else:
            ## Uses 2D Placefields
            print(f'Uses 2D Placefields')
            long_pfND, short_pfND, global_pfND = long_results.pf2D, short_results.pf2D, global_results.pf2D
            # long_pfND_decoder, short_pfND_decoder, global_pfND_decoder = long_results.pf2D_Decoder, short_results.pf2D_Decoder, global_results.pf2D_Decoder
        final_output_context_dict['pfND_ndim'] = pfND_ndim


        non_directional_names_to_default_epoch_names_map = dict(zip(['long', 'short', 'global'], [long_epoch_name, short_epoch_name, global_epoch_name]))
        
        original_pfs_dict: Dict[types.DecoderName, PfND] = {'long': deepcopy(long_pfND), 'short': deepcopy(short_pfND), 'global': deepcopy(global_pfND)} ## Uses ND Placefields

        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        single_global_epoch: Epoch = Epoch(single_global_epoch_df)
        # single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

        # # Time-dependent
        # long_pf1D_dt: PfND_TimeDependent = long_results.pf1D_dt
        # long_pf2D_dt: PfND_TimeDependent = long_results.pf2D_dt
        # short_pf1D_dt: PfND_TimeDependent = short_results.pf1D_dt
        # short_pf2D_dt: PfND_TimeDependent = short_results.pf2D_dt
        # global_pf1D_dt: PfND_TimeDependent = global_results.pf1D_dt
        # global_pf2D_dt: PfND_TimeDependent = global_results.pf2D_dt
        
        # 'time_bin_size' ____________________________________________________________________________________________________ #
        time_bin_size: float = initial_context_dict.pop('time_bin_size', 0.025)
        epochs_decoding_time_bin_size: float = time_bin_size
        final_output_context_dict['time_bin_size'] = epochs_decoding_time_bin_size
        
        frame_divide_bin_size = 60.0

        # 'trained_compute_epochs' ________________________________________________________________________________________________________ #
        trained_compute_epochs_name: str = initial_context_dict.pop('trained_compute_epochs', 'laps') # ['laps', 'pbe', 'non_pbe']
        final_output_context_dict['trained_compute_epochs'] = trained_compute_epochs_name
        # assert hasattr(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[global_epoch_name]], trained_compute_epochs_name), f"trained_compute_epochs_name: '{trained_compute_epochs_name}'"
        assert hasattr(global_session, trained_compute_epochs_name), f"trained_compute_epochs_name: '{trained_compute_epochs_name}'"
        #TODO 2025-03-11 09:10: - [X] Get proper compute epochs from the context
        trained_compute_epochs: Epoch = ensure_Epoch(deepcopy(getattr(global_session, trained_compute_epochs_name))) # .non_pbe
        
        new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=trained_compute_epochs) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        # new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(a_new_training_df_dict[a_name])) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        

        # 'known_named_decoding_epochs_type' ____________________________________________________________________________________________________ #
        # types.KnownNamedDecodingEpochsType:  typing.Literal['laps', 'replay', 'ripple', 'pbe', 'non_pbe']
        known_named_decoding_epochs_type: str = initial_context_dict.pop('known_named_decoding_epochs_type', 'laps')
        #TODO 2025-03-11 09:10: - [ ] Get proper decode epochs from the context
        decode_epochs = deepcopy(single_global_epoch)
        
        final_output_context_dict['known_named_decoding_epochs_type'] = known_named_decoding_epochs_type
        

        ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        continuous_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=decode_epochs, decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}

        # from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.Compute_NonPBE_Epochs.compute_all` _________________________________________ #
        # frame_divided_epochs_specific_decoded_results2D_dict = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder2D_dict.items()}


        # From `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationFunctions.perform_compute_non_PBE_epochs` __________________ #

        # ==================================================================================================================== #
        # 2025-03-09 - Compute the Decoded Marginals for the known epochs (laps, ripples, etc)                                 #
        # ==================================================================================================================== #
        ## Common/shared for all decoded epochs:
        unique_decoder_names = ['long', 'short']
        
        # ======================================================================================================================================================================================================================================== #
        ## EXPAND `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationsComputationsContainer._build_merged_joint_placefields_and_decode`
        # ======================================================================================================================================================================================================================================== #
        # non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df) = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline))) # , filter_epochs=deepcopy(global_any_laps_epochs_obj)
        spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        # a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = self.a_new_NonPBE_Epochs_obj
        # results1D: NonPBEDimensionalDecodingResult = self.results1D
        # results2D: NonPBEDimensionalDecodingResult = self.results2D

        # ==================================================================================================================== #
        # extracted from `perform_compute_non_PBE_epochs(...)` global computation function                                     #
        # ==================================================================================================================== #
        training_data_portion: float = 0.0
        skip_training_test_split=True
        a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.init_from_pipeline(curr_active_pipeline=curr_active_pipeline, training_data_portion=training_data_portion, skip_training_test_split=skip_training_test_split)
        # curr_active_pipeline.global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj

        ## apply the new epochs to the session:
        # curr_active_pipeline.filtered_sessions[global_epoch_name].non_PBE = deepcopy(a_new_NonPBE_Epochs_obj.global_epoch_only_non_PBE_epoch_df) ## Only adds to global_epoch? Not even .sess?

        results1D, results2D = a_new_NonPBE_Epochs_obj.compute_all(curr_active_pipeline, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size, compute_1D=True, compute_2D=False, skip_training_test_split=skip_training_test_split)
        # if (results1D is not None) and compute_1D:
        #     curr_active_pipeline.global_computation_results.computed_data['EpochComputations'].results1D = results1D
        # if (results2D is not None) and compute_2D:
        #     curr_active_pipeline.global_computation_results.computed_data['EpochComputations'].results2D = results2D
        # curr_active_pipeline.global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj
    
        # epochs_decoding_time_bin_size = results1D.epochs_decoding_time_bin_size
        # frame_divide_bin_size = results1D.frame_divide_bin_size

        # print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')

        ## INPUTS: results1D, results1D.continuous_results, a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs

        unique_decoder_names: List[str] = ['long', 'short']
        # results1D.pfs
        # results1D.decoders # BasePositionDecoder

        pfs: Dict[types.DecoderName, PfND] = {k:deepcopy(v) for k, v in results1D.pfs.items() if k in unique_decoder_names}
        # decoders: Dict[types.DecoderName, BasePositionDecoder] = {k:deepcopy(v) for k, v in results1D.decoders.items() if k in unique_decoder_names}
        continuous_decoded_results_dict: Dict[str, DecodedFilterEpochsResult] = {k:deepcopy(v) for k, v in results1D.continuous_results.items() if k in unique_decoder_names}
        # DirectionalPseudo2DDecodersResult(

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        non_PBE_all_directional_pf1D: PfND = PfND.build_merged_directional_placefields(pfs, debug_print=False)
        non_PBE_all_directional_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(non_PBE_all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # non_PBE_all_directional_pf1D_Decoder

        if decode_epochs is None:
            # use global epoch
            # single_global_epoch_df: pd.DataFrame = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            single_global_epoch: Epoch = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            decode_epochs = single_global_epoch


        # takes 6.3 seconds
        ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(decode_epochs),
                                                                                                                                                decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)

        ## OUTPUTS: pseudo2D_continuous_specific_decoded_results, non_PBE_all_directional_pf1D, non_PBE_all_directional_pf1D_Decoder
        # 3.3s

        #@ build_generalized_non_marginalized_raw_posteriors
        # ==================================================================================================================== #
        # Compute Marginals over TrackID                                                                                       #
        # ==================================================================================================================== #
        # pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = pseudo2D_continuous_specific_decoded_result
        assert len(pseudo2D_continuous_specific_decoded_result.p_x_given_n_list) == 1

        # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
        non_PBE_marginal_over_track_IDs_list, non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names)
        non_PBE_marginal_over_track_ID = non_PBE_marginal_over_track_IDs_list[0]['p_x_given_n']
        time_bin_containers = pseudo2D_continuous_specific_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)

        ## Build into a marginal df like `all_sessions_laps_df` - uses `time_window_centers`, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID:
        track_marginal_posterior_df : pd.DataFrame = deepcopy(non_PBE_marginal_over_track_ID_posterior_df) # pd.DataFrame({'t':deepcopy(time_window_centers), 'P_Long': np.squeeze(non_PBE_marginal_over_track_ID[0, :]), 'P_Short': np.squeeze(non_PBE_marginal_over_track_ID[1, :]), 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size})
        
        if 'time_bin_size' not in track_marginal_posterior_df.columns:
            track_marginal_posterior_df['time_bin_size'] = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size ## add time_bin_size column if needed

        # track_marginal_posterior_df['delta_aligned_start_t'] = track_marginal_posterior_df['t'] - t_delta ## subtract off t_delta
        

        ## END EXPAND `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationsComputationsContainer._build_merged_joint_placefields_and_decode`
        # ======================================================================================================================================================================================================================================== #


        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        
        ## from dict of filter_epochs to decode:
        global_replays_df: pd.DataFrame = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
        global_any_laps_epochs_obj = curr_active_pipeline.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs # global_session.get
        filter_epochs_to_decode_dict: Dict[types.KnownNamedDecodingEpochsType, Epoch] = {'laps': ensure_Epoch(deepcopy(global_any_laps_epochs_obj)),
                                                                                'pbe': ensure_Epoch(deepcopy(global_session.pbe.get_non_overlapping())),
                                        #  'ripple': ensure_Epoch(deepcopy(global_session.ripple)),
                                        #   'replay': ensure_Epoch(deepcopy(global_replays_df)),
                                        'non_pbe': ensure_Epoch(deepcopy(global_session.non_pbe)),
                                        }
        # filter_epochs_to_decode_dict

        ## constrain all epochs to be at least two decoding time bins long, or drop them entirely:
        filter_epochs_to_decode_dict = {k:_compute_proper_filter_epochs(epochs_df=v, desired_decoding_time_bin_size=epochs_decoding_time_bin_size, minimum_event_duration=(2.0 * epochs_decoding_time_bin_size), mode=EpochFilteringMode.DropShorter)[0] for k, v in filter_epochs_to_decode_dict.items()} # `[0]` gets just the dataframe, as in DropShorter mode the time_bin_size is unchanged

        ## Perform the decoding and masking as needed for invalid bins:
        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()


        # ======================================================================================================================================================================================================================================== #
        # Replaces pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationsComputationsContainer._build_output_decoded_posteriors
        # ======================================================================================================================================================================================================================================== #
        assert epochs_decoding_time_bin_size is not None, f"epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
        # time_bin_size: float = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size
        
        filter_epochs_pseudo2D_continuous_specific_decoded_result: Dict[types.KnownNamedDecodingEpochsType, DecodedFilterEpochsResult] = {}
        filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[types.MaskedTimeBinFillType, pd.DataFrame]] = {}
        
        for a_decoded_epoch_type_name, a_filter_epoch_obj in filter_epochs_to_decode_dict.items():
            # a_decoded_epoch_type_name: like 'laps', 'ripple', or 'non_pbe'
            
            a_filtered_epochs_df = ensure_dataframe(deepcopy(a_filter_epoch_obj)).epochs.filtered_by_duration(min_duration=epochs_decoding_time_bin_size*2)
            # active_filter_epochs = a_filter_epoch_obj
            active_filter_epochs = a_filtered_epochs_df
            a_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(active_filter_epochs), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)
            
            # a_filtered_epochs_df = a_pseudo2D_continuous_specific_decoded_result.filter_epochs.epochs.filtered_by_duration(min_duration=epochs_decoding_time_bin_size*2)
            # a_pseudo2D_continuous_specific_decoded_result = a_pseudo2D_continuous_specific_decoded_result.filtered_by_epoch_times(included_epoch_start_times=a_filtered_epochs_df['start'].to_numpy())
            

            filter_epochs_pseudo2D_continuous_specific_decoded_result[a_decoded_epoch_type_name] = a_pseudo2D_continuous_specific_decoded_result ## add result to outputs dict
            # a_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = a_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            a_non_PBE_marginal_over_track_ID, a_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(a_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']


            
            ## MASKED:
            # masked_pseudo2D_continuous_specific_decoded_result, _mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='last_valid') ## Masks the low-firing bins so they don't confound the analysis.
            masked_laps_pseudo2D_continuous_specific_decoded_result, _mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='last_valid') ## Masks the low-firing bins so they don't confound the analysis.
            # masked_laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = masked_laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            masked_laps_non_PBE_marginal_over_track_ID, masked_laps_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(masked_laps_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']

            ## MASKED with NaNs (no backfill):
            dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='nan_filled') ## Masks the low-firing bins so they don't confound the analysis.
            # dropping_masked_laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = dropping_masked_laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            dropping_masked_laps_non_PBE_marginal_over_track_ID, dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']
            
            # from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

            ## Build into a marginal df like `all_sessions_laps_df`:
            
            # track_marginal_posterior_df_list = [track_marginal_posterior_df, masked_track_marginal_posterior_df]

            # masked_bin_fill_modes: ['ignore', 'last_valid', 'nan_filled', 'dropped']

            # _track_marginal_posterior_df_dict = {'track_marginal_posterior_df':track_marginal_posterior_df, 'masked_track_marginal_posterior_df': masked_track_marginal_posterior_df}
            decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[types.MaskedTimeBinFillType, pd.DataFrame] = {# 'track_marginal_posterior_df':track_marginal_posterior_df,
                                                                                            'ignore': a_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            'last_valid': masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            'nan_filled': dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'a_non_PBE_marginal_over_track_ID_posterior_df': a_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'masked_laps_non_PBE_marginal_over_track_ID_posterior_df': masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df': dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            }

            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            for k in list(decoded_filter_epoch_track_marginal_posterior_df_dict.keys()):
                a_df = decoded_filter_epoch_track_marginal_posterior_df_dict[k]
                a_df['delta_aligned_start_t'] = a_df['t'] - t_delta ## subtract off t_delta    
                a_df = a_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col='t')
                decoded_filter_epoch_track_marginal_posterior_df_dict[k] = a_df
                
            # ['dropping_masked', 'dropping_masked', 'masked', 'dropping_masked']
            filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_decoded_epoch_type_name] = decoded_filter_epoch_track_marginal_posterior_df_dict


            ## UPDATES: track_marginal_posterior_df, masked_track_marginal_posterior_df
            ## UPDATES: laps_non_PBE_marginal_over_track_ID_posterior_df, masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            # masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            # dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            
            ## OUTPUTS: laps_non_PBE_marginal_over_track_ID_posterior_df, dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df, masked_laps_non_PBE_marginal_over_track_ID_posterior_df

        # END for a_...

        # return filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
        a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = GeneralDecoderDictDecodedEpochsDictResult(filter_epochs_to_decode_dict=filter_epochs_to_decode_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result=filter_epochs_pseudo2D_continuous_specific_decoded_result,
                                                         filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict=filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict)
    

        # # filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
        # a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = EpochComputationsComputationsContainer._build_output_decoded_posteriors(non_PBE_all_directional_pf1D_Decoder=non_PBE_all_directional_pf1D_Decoder, # pseudo2D_continuous_specific_decoded_result=pseudo2D_continuous_specific_decoded_result,
        #     filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
        #     unique_decoder_names=unique_decoder_names, spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
        #     session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
        # )
    
        final_output_context: IdentifyingContext = IdentifyingContext(**final_output_context_dict)
        return final_output_context

    @function_attributes(short_name=None, tags=['compute', 'continuous', 'epoch', 'global'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-21 00:00', related_items=[])
    def computing_for_global_epoch(self, curr_active_pipeline, debug_print=True):
        """ Uses the context to extract proper values from the pipeline, and performs a fresh computation
        Computes what are often (misleadinging) called "continuous" epoch computations, meaning they are computed uninterrupted across all time instead of start/ending at specific epochs (like laps or PBEs).
        
        Usage:
            _out = a_new_fully_generic_result.computing_for_global_epoch(curr_active_pipeline=curr_active_pipeline, context=IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='laps'))
            
        
        History:
            from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.Compute_NonPBE_Epochs.recompute`
        
            
        adds keys known_named_decoding_epochs_type='global'
        """
        print(f'GenericDecoderDictDecodedEpochsDictResult.computing_for_global_epoch(...):')
        
        # use global epoch
        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        single_global_epoch: Epoch = Epoch(single_global_epoch_df)
        # single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)
        decode_epochs = single_global_epoch
    
        # ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        # continuous_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=decode_epochs, decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}


        search_context = IdentifyingContext(pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_time_bin') # , data_grain= 'per_time_bin -- not really relevant: ['masked_time_bin_fill_type', 'known_named_decoding_epochs_type', 'data_grain']
        # flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=search_context, return_multiple_matches=True, debug_print=True)
        a_ctxt, a_result, a_decoder, _ = self.get_results_matching_contexts(context_query=search_context, return_multiple_matches=False, debug_print=True)
        # a_decoder
        if debug_print:
            print(f'a_ctxt: {a_ctxt}')

        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        # for a_ctxt, a_result in self.filter_epochs_specific_decoded_result.items():
        # for a_ctxt, a_decoder in self.decoders.items():
        if debug_print:
            print(f'a_ctxt: {a_ctxt}')
            
        a_new_context = a_ctxt.overwriting_context(known_named_decoding_epochs_type='global')
        if debug_print:
            print(f'\ta_new_context: {a_new_context}')

        # 'time_bin_size' ____________________________________________________________________________________________________ #
        time_bin_size: float = a_ctxt.to_dict().get('time_bin_size', 0.025)
        a_new_result = a_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(single_global_epoch), decoding_time_bin_size=time_bin_size, debug_print=False)
        a_new_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_new_result, marginal_context=a_new_context)
        _was_update_success = self.updating_results_for_context(new_context=a_new_context, a_result=a_new_result, a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=a_new_decoded_marginal_posterior_df, an_epoch_to_decode=single_global_epoch, debug_print=debug_print)
        if not _was_update_success:
            print(f'\t\tWARN: update failed for global context: {a_new_context}')

        ## MASKED with NaNs (no backfill):
        masked_bin_fill_mode = 'nan_filled'
        ## INPUTS: a_result, masked_bin_fill_mode
        a_masked_updated_context: IdentifyingContext = deepcopy(a_new_context).overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
        if debug_print:
            print(f'\ta_masked_updated_context: {a_masked_updated_context}')

        a_dropping_masked_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_new_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), masked_bin_fill_mode=masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.
        ## Computes marginals for `dropping_masked_laps_pseudo2D_continuous_specific_decoded_result`
        a_dropping_masked_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_dropping_masked_pseudo2D_continuous_specific_decoded_result, marginal_context=a_masked_updated_context)
        _was_update_success = self.updating_results_for_context(new_context=a_masked_updated_context, a_result=deepcopy(a_dropping_masked_pseudo2D_continuous_specific_decoded_result), a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=deepcopy(a_dropping_masked_decoded_marginal_posterior_df)) ## update using the result
        if not _was_update_success:
            print(f'\t\tWARN: update failed for masked context: {a_masked_updated_context}')


        # #@ build_generalized_non_marginalized_raw_posteriors
        # # ==================================================================================================================== #
        # # Compute Marginals over TrackID                                                                                       #
        # # ==================================================================================================================== #
        # # pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = pseudo2D_continuous_specific_decoded_result
        # assert len(pseudo2D_continuous_specific_decoded_result.p_x_given_n_list) == 1

        # # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
        # non_PBE_marginal_over_track_IDs_list, non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names)
        # non_PBE_marginal_over_track_ID = non_PBE_marginal_over_track_IDs_list[0]['p_x_given_n']
        # time_bin_containers = pseudo2D_continuous_specific_decoded_result.time_bin_containers[0]
        # time_window_centers = time_bin_containers.centers
        # # p_x_given_n.shape # (62, 4, 209389)

        # ## Build into a marginal df like `all_sessions_laps_df` - uses `time_window_centers`, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID:
        # track_marginal_posterior_df : pd.DataFrame = deepcopy(non_PBE_marginal_over_track_ID_posterior_df) # pd.DataFrame({'t':deepcopy(time_window_centers), 'P_Long': np.squeeze(non_PBE_marginal_over_track_ID[0, :]), 'P_Short': np.squeeze(non_PBE_marginal_over_track_ID[1, :]), 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size})
        
        # if 'time_bin_size' not in track_marginal_posterior_df.columns:
        #     track_marginal_posterior_df['time_bin_size'] = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size ## add time_bin_size column if needed

        # # track_marginal_posterior_df['delta_aligned_start_t'] = track_marginal_posterior_df['t'] - t_delta ## subtract off t_delta
        

        # ## END EXPAND `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationsComputationsContainer._build_merged_joint_placefields_and_decode`
        # # ======================================================================================================================================================================================================================================== #

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        
        laps_trained_decoder_search_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_time_bin') # , data_grain= 'per_time_bin -- not really relevant: ['masked_time_bin_fill_type', 'known_named_decoding_epochs_type', 'data_grain']
        # flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=search_context, return_multiple_matches=True, debug_print=True)
        a_laps_trained_decoder_ctxt, a_laps_trained_decoder_result, a_laps_trained_decoder, _ = self.get_results_matching_contexts(context_query=laps_trained_decoder_search_context, return_multiple_matches=False, debug_print=True)
        # a_decoder
        if debug_print:
            print(f'a_laps_trained_decoder_ctxt: {a_laps_trained_decoder_ctxt}')





        ## get the base decoder we'll use for decoding
        print(f'trying to compute for known_named_decoding_epochs_type="non_pbe_endcaps"...')
        # a_base_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore'
        # a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = self.get_results_matching_contexts(a_base_context, return_multiple_matches=False)
        new_desired_decode_epochs_name = 'non_pbe_endcaps'
        assert hasattr(global_session, new_desired_decode_epochs_name), f"must already have the valid non_pbe_endcaps for the global_session. known_named_decoding_epochs_type: '{new_desired_decode_epochs_name}'"
        new_desired_decode_epochs: Epoch = ensure_Epoch(deepcopy(getattr(global_session, new_desired_decode_epochs_name))) # .non_pbe_endcaps
        
        filter_epochs_to_decode_dict = {
            # deepcopy(a_best_matching_context).overwriting_context(known_named_decoding_epochs_type='laps'):deepcopy(laps_df),
            # deepcopy(a_best_matching_context).overwriting_context(known_named_decoding_epochs_type='pbes'):deepcopy(non_pbe_df),
            deepcopy(a_laps_trained_decoder_ctxt).overwriting_context(known_named_decoding_epochs_type=new_desired_decode_epochs_name):deepcopy(new_desired_decode_epochs),
        }
        self = self.perform_decoding(an_all_directional_pf1D_Decoder=deepcopy(a_laps_trained_decoder), filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                                            spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), epochs_decoding_time_bin_size=time_bin_size,
                                            session_name=session_name,
                                            t_start=t_start, t_delta=t_delta, t_end=t_end,
                                        )

        ## OUTPUTS: a_new_fully_generic_result




        # 'trained_compute_epochs' ________________________________________________________________________________________________________ #
        new_desired_trained_compute_epochs_name: str = 'non_pbe' # initial_context_dict.pop('trained_compute_epochs', 'laps') # ['laps', 'pbe', 'non_pbe']
        # final_output_context_dict['trained_compute_epochs'] = new_desired_trained_compute_epochs_name
        
        a_new_context = a_laps_trained_decoder_ctxt.overwriting_context(trained_compute_epochs=new_desired_trained_compute_epochs_name)
        if debug_print:
            print(f'\ta_new_context: {a_new_context}')
            

        # assert hasattr(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[global_epoch_name]], trained_compute_epochs_name), f"trained_compute_epochs_name: '{trained_compute_epochs_name}'"
        assert hasattr(global_session, new_desired_trained_compute_epochs_name), f"trained_compute_epochs_name: '{new_desired_trained_compute_epochs_name}'"
        #TODO 2025-03-11 09:10: - [X] Get proper compute epochs from the context
        trained_compute_epochs: Epoch = ensure_Epoch(deepcopy(getattr(global_session, new_desired_trained_compute_epochs_name))) # .non_pbe
        # new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=trained_compute_epochs) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        a_new_decoder: BasePositionDecoder = deepcopy(a_laps_trained_decoder) #TODO 2025-03-26 12:51: - [ ] This is WRONG as is. # .replacing_computation_epochs(epochs=trained_compute_epochs) ## build new simple decoders BasePositionDecoder(pf=deepcopy(a_laps_trained_decoder.pf))
        ## #TODO 2025-03-26 12:48: - [ ] `a_laps_trained_decoder.pf.position` is None... how does that happen?
        ## add the new decoder
        self.decoders[a_new_context] = a_new_decoder
        
        #TODO 2025-03-26 10:49: - [ ] Decoding using the decoder


        # final_output_context: IdentifyingContext = IdentifyingContext(**final_output_context_dict)
        
        # return final_output_context
        print(f'\tdone.')
        return self



    @function_attributes(short_name=None, tags=['batch', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=['generalized_decode_epochs_dict_and_export_results_completion_function'], creation_date='2025-03-21 00:00', related_items=[])
    @classmethod
    def batch_user_compute_fn(cls, curr_active_pipeline, force_recompute:bool=True, time_bin_size: float = 0.025, debug_print:bool=True) -> 'GenericDecoderDictDecodedEpochsDictResult':
        """ Uses the context to extract proper values from the pipeline, and performs a fresh computation
        
        Usage:
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=curr_active_pipeline, force_recompute=force_recompute, debug_print=debug_print)
            
        History:
            from `generalized_decode_epochs_dict_and_export_results_completion_function`
        
        """
        from typing import Literal
        from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe, ensure_Epoch, TimeColumnAliasesProtocol
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer, NonPBEDimensionalDecodingResult, Compute_NonPBE_Epochs, KnownFilterEpochs, GeneralDecoderDictDecodedEpochsDictResult
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes
        from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode, _compute_proper_filter_epochs
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, DecoderDecodedEpochsResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting, CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, DirectionalPseudo2DDecodersResult
        
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        if force_recompute and ('EpochComputations' in curr_active_pipeline.global_computation_results.computed_data):
            print(f'\t recomputing...')
            del curr_active_pipeline.global_computation_results.computed_data['EpochComputations'] ## drop the old result

        curr_active_pipeline.reload_default_computation_functions()
        ## perform the computation either way:
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['non_PBE_epochs_results'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring', 'non_PBE_epochs_results'],
                                                        computation_kwargs_list=[{'ripple_decoding_time_bin_size': time_bin_size, 'laps_decoding_time_bin_size': time_bin_size}, {'time_bin_size': time_bin_size}, {'should_skip_radon_transform': True},
                                                                                    {'same_thresh_fraction_of_track': 0.05, 'max_ignore_bins': 2, 'use_bin_units_instead_of_realworld': False, 'max_jump_distance_cm': 60.0},
                                                                                     dict(epochs_decoding_time_bin_size=time_bin_size, frame_divide_bin_size=10.0, compute_1D=True, compute_2D=False, drop_previous_result_and_compute_fresh=force_recompute, skip_training_test_split=True, debug_print_memory_breakdown=False)], ## END KWARGS LIST
                                                                                     enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.batch_extended_computations(include_includelist=['non_PBE_epochs_results'], include_global_functions=True, included_computation_filter_names=None, fail_on_exception=True, debug_print=False) ## just checking


        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        
        ## Unpack the results:
        # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        ## Unpack from pipeline:
        nonPBE_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
        # a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = nonPBE_results.a_new_NonPBE_Epochs_obj
        results1D: NonPBEDimensionalDecodingResult = nonPBE_results.results1D
        # results2D: NonPBEDimensionalDecodingResult = nonPBE_results.results2D

        epochs_decoding_time_bin_size = nonPBE_results.epochs_decoding_time_bin_size
        frame_divide_bin_size = nonPBE_results.frame_divide_bin_size

        if debug_print:
            print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')

        assert (results1D is not None)
        # assert (results2D is not None)

        # ==================================================================================================================== #
        # Pre 2025-03-10 Semi-generic (unfinalized) Result                                                                     #
        # ==================================================================================================================== #
        a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = nonPBE_results.a_general_decoder_dict_decoded_epochs_dict_result ## get the pre-decoded result
        assert a_general_decoder_dict_decoded_epochs_dict_result is not None

        # ==================================================================================================================== #
        # New 2025-03-11 Generic Result:                                                                                       #
        # ==================================================================================================================== #
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult(is_global=True)  # start empty

        ## add the 'non_pbe' results:
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = a_new_fully_generic_result.adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(a_general_decoder_dict_decoded_epochs_dict_result=a_general_decoder_dict_decoded_epochs_dict_result)
        a_new_fully_generic_result.spikes_df_dict[curr_active_pipeline.get_session_context()] = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        a_new_fully_generic_result.spikes_df_dict[IdentifyingContext()] = deepcopy(get_proper_global_spikes_df(curr_active_pipeline)) # all contexts
        
        # # ==================================================================================================================== #
        # # Phase 1: build from 'DirectionalDecodersEpochsEvaluations'                                                           #
        # # ==================================================================================================================== #
        # # filtered_epochs_df = None
        # if 'DirectionalDecodersEpochsEvaluations' in curr_active_pipeline.global_computation_results.computed_data:
        #     ## INPUTS: curr_active_pipeline, track_templates, a_decoded_filter_epochs_decoder_result_dict
        #     directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL
        #     ## INPUTS: directional_decoders_epochs_decode_result, filtered_epochs_df
        #     ## Inputs: a_new_fully_generic_result
        #     a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)
        #     ## OUTPUTS: a_new_fully_generic_result
        # else:
        #     print('WARN: missing "DirectionalDecodersEpochsEvaluations" global result. Skipping.')

        # ==================================================================================================================== #
        # Phase 2 - Get Directional Decoded Epochs                                                                             #
        # ==================================================================================================================== #

        #TODO 🚧 2025-03-11 13:01: - [ ] Allow overriding qclu and inclusion_fr values, see other user function
        if 'DirectionalMergedDecoders' in curr_active_pipeline.global_computation_results.computed_data:
            # DirectionalMergedDecoders: Get the result after computation:
            directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
            a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_merged_decoders_result=directional_merged_decoders_result) # , spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        else:
            print('WARN: missing "DirectionalMergedDecoders" global result. Skipping.')
        
        # # ==================================================================================================================== #
        # # 2025-02-20 20:06 New `nonPBE_results._build_merged_joint_placefields_and_decode` method                              #
        # # ==================================================================================================================== #
        # non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
        # masked_pseudo2D_continuous_specific_decoded_result, _mask_index_tuple = pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))


        # ==================================================================================================================== #
        # Phase 2.5 - Add Continuous Results as a `known_named_decoding_epochs_type` --- known_named_decoding_epochs_type='continuous'
        # ==================================================================================================================== #
        known_named_decoding_epochs_type='global'
        a_new_fully_generic_result = a_new_fully_generic_result.computing_for_global_epoch(curr_active_pipeline=curr_active_pipeline, debug_print=debug_print)       



        # ==================================================================================================================== #
        # Phase 2.55 - Add non-PBE decoders to the .decoders dict                                                              #
        # ==================================================================================================================== #
        # # 'trained_compute_epochs' ________________________________________________________________________________________________________ #
        # trained_compute_epochs_name: str = initial_context_dict.pop('trained_compute_epochs', 'laps') # ['laps', 'pbe', 'non_pbe']
        # final_output_context_dict['trained_compute_epochs'] = trained_compute_epochs_name
        # # assert hasattr(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[global_epoch_name]], trained_compute_epochs_name), f"trained_compute_epochs_name: '{trained_compute_epochs_name}'"
        # assert hasattr(global_session, trained_compute_epochs_name), f"trained_compute_epochs_name: '{trained_compute_epochs_name}'"
        # #TODO 2025-03-11 09:10: - [X] Get proper compute epochs from the context
        # trained_compute_epochs: Epoch = ensure_Epoch(deepcopy(getattr(global_session, trained_compute_epochs_name))) # .non_pbe
        
        # new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=trained_compute_epochs) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        
        # ==================================================================================================================== #
        # Phase 3 - `creating_new_spikes_per_t_bin_masked_variants`                                                                      #
        # ==================================================================================================================== #
        spikes_df = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        a_new_fully_generic_result = a_new_fully_generic_result.creating_new_spikes_per_t_bin_masked_variants(spikes_df=spikes_df)
        
        ## ensure all optional fields are present before output:
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        for k in list(a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict.keys()):
            a_df = a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[k]
            ## note in per-epoch mode we use the start of the epoch (because for example laps are long and we want to see as soon as it starts) but for time bins we use the center time.
            time_column_name: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(a_df, col_connonical_name='t', required_columns_synonym_dict={"t":{'t_bin_center', 'lap_start_t', 'ripple_start_t', 'epoch_start_t'}}, should_raise_exception_on_fail=True)
            assert time_column_name in a_df
            a_df['delta_aligned_start_t'] = a_df[time_column_name] - t_delta ## subtract off t_delta
            a_df = a_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col=time_column_name)
            a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[k] = a_df

        # ==================================================================================================================== #
        # Phase 4 - Remdial - Add any missing dataframes directly.                                                             #
        # ==================================================================================================================== #
        ## Build masked versions of important contexts:
        ## INPUTS: time_bin_size, 

        ## Common/shared for all decoded epochs:
        for a_masked_bin_fill_mode in ['dropped']: # , 'last_valid'
            # a_masked_bin_fill_mode = 'nan_filled'

            ## INPUTS: a_new_fully_generic_result
            base_contexts_list = [IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_time_bin'),
                                IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore', data_grain='per_time_bin'),
                                IdentifyingContext(trained_compute_epochs='non_pbe', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_time_bin'),
                                IdentifyingContext(trained_compute_epochs='non_pbe', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')]
            masked_contexts_dict = {}

            for a_base_context in base_contexts_list:

                a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_contexts(a_base_context, return_multiple_matches=False)
                ## `a_decoder` is None for some reason?`
                ## INPUTS: a_result, masked_bin_fill_mode
                a_masked_updated_context: IdentifyingContext = deepcopy(a_best_matching_context).overwriting_context(masked_time_bin_fill_type=a_masked_bin_fill_mode, data_grain='per_time_bin')
                masked_contexts_dict[a_base_context] = a_masked_updated_context
                if debug_print:
                    print(f'a_masked_updated_context: {a_masked_updated_context}')
                
                ## MASKED with NaNs (no backfill):
                a_dropping_masked_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode=a_masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.
                ## Computes marginals for `dropping_masked_laps_pseudo2D_continuous_specific_decoded_result`
                a_dropping_masked_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_dropping_masked_pseudo2D_continuous_specific_decoded_result, marginal_context=a_masked_updated_context)
                a_new_fully_generic_result.updating_results_for_context(new_context=a_masked_updated_context, a_result=deepcopy(a_dropping_masked_pseudo2D_continuous_specific_decoded_result), a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=deepcopy(a_dropping_masked_decoded_marginal_posterior_df)) ## update using the result
                
            ## OUTPUTS: masked_contexts_dict

            
        ## ensure all optional fields are present before output:
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        for k in list(a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict.keys()):
            a_df = a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[k]
            ## note in per-epoch mode we use the start of the epoch (because for example laps are long and we want to see as soon as it starts) but for time bins we use the center time.
            time_column_name: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(a_df, col_connonical_name='t', required_columns_synonym_dict={"t":{'t_bin_center', 'lap_start_t', 'ripple_start_t', 'epoch_start_t'}}, should_raise_exception_on_fail=True)
            assert time_column_name in a_df
            a_df['delta_aligned_start_t'] = a_df[time_column_name] - t_delta ## subtract off t_delta
            a_df = a_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col=time_column_name)
            a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[k] = a_df


        # ==================================================================================================================================================================================================================================================================================== #
        # Phase 5 - Get the corrected 'per_epoch' results from the 'per_time_bin' versions                                                                                                                                                                                                     #
        # ==================================================================================================================================================================================================================================================================================== #
        ## get all non-global, `data_grain= 'per_time_bin'`
        flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=IdentifyingContext(trained_compute_epochs='laps', decoder_identifier='pseudo2D',
                                                                                                                                                                                                                            time_bin_size=time_bin_size,
                                                                                                                                                                                                                            known_named_decoding_epochs_type=['pbe', 'laps', 'non_pbe'],
                                                                                                                                                                                                                            masked_time_bin_fill_type=('ignore', 'dropped'), data_grain= 'per_time_bin'))        

        _newly_updated_values_tuple = a_new_fully_generic_result.compute_all_per_epoch_aggregations_from_per_time_bin_results(flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict)
        # per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict = _newly_updated_values_tuple




        # ==================================================================================================================== #
        # Create and add the output                                                                                            #
        # ==================================================================================================================== #
        
        print('\t\tdone.')        


        return a_new_fully_generic_result



    @function_attributes(short_name=None, tags=['compute', 'decode', 'epochs'], input_requires=[], output_provides=[], uses=['self.updating_results_for_context'], used_by=[], creation_date='2025-04-05 11:51', related_items=[])
    def perform_decoding(self, an_all_directional_pf1D_Decoder: BasePositionDecoder, filter_epochs_to_decode_dict: Dict[types.GenericResultTupleIndexType, Epoch], spikes_df: pd.DataFrame, epochs_decoding_time_bin_size: float,
                                        session_name: str, t_start: float, t_delta: float, t_end: float, unique_decoder_names: Optional[List[str]]=None, debug_print:bool=True):
        """ Uses the provided decoder to decode each filter_epoch in the provided dict. The keys of the dict should be the IdentifyingContext of the resultant decoded values.

        Usage: 
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult

            ## INPUTS: a_new_fully_generic_result
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            spikes_df = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))

            epochs_decoding_time_bin_size = 0.025

            ## get the base decoder we'll use for decoding
            a_base_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore'
            a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_contexts(a_base_context, return_multiple_matches=False)

            filter_epochs_to_decode_dict = {deepcopy(a_best_matching_context).overwriting_context(known_named_decoding_epochs_type='laps'):deepcopy(laps_df),
                                    # deepcopy(a_best_matching_context).overwriting_context(known_named_decoding_epochs_type='pbes'):deepcopy(non_pbe_df),
                                    deepcopy(a_best_matching_context).overwriting_context(known_named_decoding_epochs_type='non_pbe_endcaps'):deepcopy(non_pbe_endcaps_df),
            }

            a_new_fully_generic_result = a_new_fully_generic_result.perform_decoding(an_all_directional_pf1D_Decoder=a_decoder, filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                                                spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                                                session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
                                            )

            ## OUTPUTS: a_new_fully_generic_result
            
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationsComputationsContainer

        ## INPUTS: a_decoder
        if unique_decoder_names is None:
            unique_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            # unique_decoder_names=['long', 'short']
            

        if len(filter_epochs_to_decode_dict) == 0:
            print(f'nothing to decode! Skipping.')
            return self
        else:
            _temp_out_tuple = EpochComputationsComputationsContainer._build_context_general_output_decoded_posteriors(non_PBE_all_directional_pf1D_Decoder=an_all_directional_pf1D_Decoder, filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                unique_decoder_names=unique_decoder_names,
                spikes_df=deepcopy(spikes_df), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
            )

            filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict = _temp_out_tuple
            ## INPUTS: filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict

            for a_new_context, a_filter_epoch_to_decode in filter_epochs_to_decoded_dict.items():
                a_new_result = filter_epochs_pseudo2D_continuous_specific_decoded_result[a_new_context]
                a_decoder = filter_epochs_decoder_dict[a_new_context]
                a_new_decoded_marginal_posterior_df = filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_new_context]

                if debug_print:
                    print(f'\tupdating results for context: {a_new_context}')
                _was_update_success = self.updating_results_for_context(new_context=a_new_context, a_result=a_new_result, a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=a_new_decoded_marginal_posterior_df, an_epoch_to_decode=a_filter_epoch_to_decode, debug_print=debug_print)
                if not _was_update_success:
                    print(f'\t\tWARN: update failed for context: {a_new_context}')

            print(f'done.')
            return self




    # ================================================================================================================================================================================ #
    # Retreval and use                                                                                                                                                                 #
    # ================================================================================================================================================================================ #
    
    @classmethod
    def get_keys_or_elements(cls, obj) -> List:
        """ Handle both dictionaries and tuples """
        if hasattr(obj, 'keys'):
            return list(obj.keys())
        elif isinstance(obj, tuple) and len(obj) > 0:
            return list(obj[0]) # item is a tuple (context, a_data) returned by `_subfn_get_value_with_context_matching`, extract just the context
            # return list(obj)
        else:
            return []
        

    @classmethod
    def _subfn_get_value_with_context_matching(cls, dictionary: Dict, query: IdentifyingContext, item_name="item", return_multiple_matches: bool=True, debug_print:bool=True) -> Union[Dict, Tuple[IdentifyingContext, Any]]:
        """Helper function to get a value from a dictionary with context matching.
        #2025-04-07 11:51: - [X] Fixed for matching empty IdentifyingContext()
        """
        try:
            value = dictionary[query]  # Try exact match first
            if (not return_multiple_matches):
                return query, value # (context, value)
            else:
                ## format as a single-item dictionary with value for compatibility
                return {query: value} # {context:value}-dict <single-item>
        except (KeyError, TypeError):
            ## TypeError: unhashable type: 'list -- occurs when the query contains multiple values
            # Find single best matching context
            if (not return_multiple_matches):
                best_match, max_num_matching_context_attributes = IdentifyingContext.find_best_matching_context(query, dictionary)
                if best_match:
                    if debug_print:
                        print(f"Found best match for {item_name} with {max_num_matching_context_attributes} matching attributes:\t{best_match}\n")
                    return best_match, dictionary[best_match] # (context, value)
                else:
                    if debug_print:
                        print(f"{item_name}: No matches found in the dictionary.")
                    return None, None # (context, value)
            else:
                # Find multiple matching contexts
                matching_contexts, number_matching_context_attributes, max_num_matching_context_attributes = IdentifyingContext.find_best_matching_contexts(query, dictionary)
                if matching_contexts:
                    if debug_print:
                        print(f"Found {len(matching_contexts)} matches for {item_name}")
                    # return [(ctx, dictionary[ctx]) for ctx in matching_contexts]
                    return {ctx:dictionary[ctx] for ctx in matching_contexts} # {context:value}-dict
                else:
                    if debug_print:
                        print(f"{item_name}: No matches found in the dictionary.")
                    return {} # {context:value}-dict
                
        except Exception as e:
            raise e
        

    @function_attributes(short_name=None, tags=['contexts', 'matching'], input_requires=[], output_provides=[], uses=['get_flattened_contexts_for_posteriors_dfs'], used_by=[], creation_date='2025-03-12 11:30', related_items=[])
    def get_matching_contexts(self, context_query: Optional[IdentifyingContext]=None, return_multiple_matches: bool=True, debug_print:bool=True) -> List[IdentifyingContext]: 
        """ contexts only, no results returned.
        This doesn't quite make sense because each results dictionary may have different contexts
        
        
        """
        if context_query is None:
            context_query = IdentifyingContext() ## empty context, returning all matches
            
        if (not return_multiple_matches):
            # ==================================================================================================================== #
            # Find single best matching context                                                                                    #
            # ==================================================================================================================== #
            result_context, a_result = self._subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoder_context, a_decoder = self._subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            posterior_context, a_decoded_marginal_posterior_df = self._subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            
            # Determine the best matching context:
            contexts = [c for c in [result_context, decoder_context, posterior_context] if c is not None]
            if contexts:
                context_lenghts = np.array([len(v.to_dict()) for v in contexts])
                max_context_length = np.max(context_lenghts)
                max_context_length_idx = np.argmax(context_lenghts)
                best_matching_context = contexts[max_context_length_idx]
            else:
                best_matching_context = None

            # Optionally add a warning for different contexts
            if debug_print and len(set(contexts)) > 1:
                print(f"Warning: Different contexts matched: result={result_context}, decoder={decoder_context}, posterior={posterior_context}")

            return best_matching_context
        
        else:
            # ==================================================================================================================== #
            # Find multiple matching contexts                                                                                      #
            # ==================================================================================================================== #
            result_context_dict = self._subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoder_context_dict = self._subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoded_marginal_posterior_df_context_dict = self._subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            
            if isinstance(decoded_marginal_posterior_df_context_dict, dict):
                any_matching_contexts_list = list(set(self.get_keys_or_elements(result_context_dict)).union(set(self.get_keys_or_elements(decoder_context_dict))).union(set(self.get_keys_or_elements(decoded_marginal_posterior_df_context_dict))))
                return any_matching_contexts_list
            else:
                return self.get_flattened_contexts_for_posteriors_dfs(decoded_marginal_posterior_df_context_dict) ## a bit inefficient but there's never that many contexts
            
    @function_attributes(short_name=None, tags=['contexts', 'matching', 'single-context', 'best'], input_requires=[], output_provides=[], uses=['get_matching_contexts'], used_by=[], creation_date='2025-04-07 18:44', related_items=[])
    def get_best_matching_context(self, context_query: Optional[IdentifyingContext]=None, debug_print:bool=True) -> Optional[IdentifyingContext]: 
        """ contexts only, no results returned.
        This doesn't quite make sense because each results dictionary may have different contexts
        
        
        """
        return self.get_matching_contexts(context_query=context_query, return_multiple_matches=False, debug_print=debug_print)
        




    @function_attributes(short_name=None, tags=['contexts', 'matching'], input_requires=[], output_provides=[], uses=['get_flattened_contexts_for_posteriors_dfs'], used_by=[], creation_date='2025-03-12 11:30', related_items=['get_results_matching_context', 'get_matching_contexts'])
    def get_results_matching_contexts(self, context_query: Optional[IdentifyingContext]=None, return_multiple_matches: bool=True, debug_print:bool=True): 
        """ Get a specific contexts
        
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='long_LR', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore')
        
        any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict = 
        
        ## Get all matching results that match the context:
        flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context, return_multiple_matches: bool=True)        
        
        ## Get single set of results best matching the context:
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context, return_multiple_matches=False)


        """
        if context_query is None:
            context_query = IdentifyingContext() ## empty context, returning all matches

        if (not return_multiple_matches):
            # ==================================================================================================================== #
            # Find single best matching context                                                                                    #
            # ==================================================================================================================== #
            result_context, a_result = self._subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoder_context, a_decoder = self._subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            posterior_context, a_decoded_marginal_posterior_df = self._subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            
            # Determine the best matching context
            contexts = [c for c in [result_context, decoder_context, posterior_context] if c is not None]
            if contexts:
                context_lenghts = np.array([len(v.to_dict()) for v in contexts])
                max_context_length = np.max(context_lenghts)
                max_context_length_idx = np.argmax(context_lenghts)
                best_matching_context = contexts[max_context_length_idx]
            else:
                best_matching_context = None

            # Optionally add a warning for different contexts
            if debug_print and len(set(contexts)) > 1:
                print(f"Warning: Different contexts matched: result={result_context}, decoder={decoder_context}, posterior={posterior_context}")

            return best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df
        
        else:
            # ==================================================================================================================== #
            # Find multiple matching contexts                                                                                      #
            # ==================================================================================================================== #
            result_context_dict = self._subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoder_context_dict = self._subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            decoded_marginal_posterior_df_context_dict = self._subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df", return_multiple_matches=return_multiple_matches, debug_print=debug_print)
            
            if isinstance(decoded_marginal_posterior_df_context_dict, dict):
                # decoded_marginal_posterior_df_context_dict =
                # any_matching_contexts_list = list(set(list(result_context_dict.keys())).union(set(list(decoder_context_dict.keys()))).union(set(list(decoded_marginal_posterior_df_context_dict.keys()))))
                any_matching_contexts_list = list(set(self.get_keys_or_elements(result_context_dict)).union(set(self.get_keys_or_elements(decoder_context_dict))).union(set(self.get_keys_or_elements(decoded_marginal_posterior_df_context_dict))))
                return any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict
            else:
                return self.get_flattened_contexts_for_posteriors_dfs(decoded_marginal_posterior_df_context_dict) ## a bit inefficient but there's never that many contexts
            

    @function_attributes(short_name=None, tags=['contexts', 'matching', 'single-context', 'best'], input_requires=[], output_provides=[], uses=['get_results_matching_contexts'], used_by=[], creation_date='2025-04-07 00:00', related_items=['get_results_matching_contexts', 'get_best_matching_context'])
    def get_results_best_matching_context(self, context_query: Optional[IdentifyingContext]=None, debug_print:bool=True): 
        """ Gets the results that best match a specific context
        
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='long_LR', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore')
        
        ## Get single set of results best matching the context:
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)

        
        
        """
        return self.get_results_matching_contexts(context_query=context_query, return_multiple_matches=False, debug_print=debug_print)
      



    @function_attributes(short_name=None, tags=['contexts', 'not-quite-working', 'BUG'], input_requires=[], output_provides=[], uses=[], used_by=['get_matching_contexts'], creation_date='2025-03-12 11:30', related_items=[])
    def get_flattened_contexts_for_posteriors_dfs(self, decoded_marginal_posterior_df_context_dict):
        """ returns 4 flat dicts with the same (full) contexts that the passed `decoded_marginal_posterior_df_context_dict` have
        
        Usage:
        
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore') # , decoder_identifier='long_LR'
            any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_matching_contexts(context_query=a_target_context, return_multiple_matches=True, return_flat_same_length_dicts=False)
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_flattened_contexts_for_posteriors_dfs(decoded_marginal_posterior_df_context_dict)
            flat_context_list

        #TODO 2025-03-20 08:49: - [ ] Doesn't return consistent length results sadly
        
        
        """
        flat_context_list = []
        flat_decoder_context_dict = {}
        flat_result_context_dict = {}

        for a_context, a_df in decoded_marginal_posterior_df_context_dict.items():
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = self.get_results_matching_contexts(context_query=a_context, return_multiple_matches=False, debug_print=False)
            assert best_matching_context == a_context, f"best_matching_context: {best_matching_context}, a_context: {a_context}"
            flat_decoder_context_dict[best_matching_context] = a_decoder
            flat_result_context_dict[best_matching_context] = a_result
            flat_context_list.append(best_matching_context)


        assert len(flat_decoder_context_dict) == len(flat_result_context_dict)
        Assert.same_length(flat_context_list, decoded_marginal_posterior_df_context_dict, flat_decoder_context_dict, flat_result_context_dict)
        return flat_context_list, flat_result_context_dict, flat_decoder_context_dict, decoded_marginal_posterior_df_context_dict


    # ==================================================================================================================== #
    # Updating and Adding                                                                                                  #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['contexts', 'updating', 'adding'], input_requires=[], output_provides=[], uses=['.build_per_time_bin_marginals_df', '.compute_marginals'], used_by=[], creation_date='2025-03-19 15:00', related_items=['self.get_matching_contexts'])
    def updating_results_for_context(self, new_context: IdentifyingContext, a_result: DecodedFilterEpochsResult, a_decoder:BasePositionDecoder, a_decoded_marginal_posterior_df: pd.DataFrame, an_epoch_to_decode: Optional[Epoch]=None, debug_print:bool=True): 
        """ Get a specific contexts
        reciprocal of `self.get_matching_contexts(...)`
        
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='long_LR', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore')
        """
        def _subfn_update_value_with_context_matching(dictionary, updated_context, updated_value: Any) -> bool:
            """Helper function to get a value from a dictionary with context matching.
            
            Captures: return_multiple_matches
            """
            try:
                dictionary[updated_context] = updated_value # Try exact match first
                return True
            except Exception as e:
                raise e
                return False

        # END def _subfn_update_value_wi...
        
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #
        _was_update_success: bool = _subfn_update_value_with_context_matching(self.filter_epochs_specific_decoded_result, new_context, updated_value=a_result)
        _was_update_success = _was_update_success and _subfn_update_value_with_context_matching(self.decoders, new_context, updated_value=a_decoder)
        _was_update_success = _was_update_success and _subfn_update_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, new_context, updated_value=a_decoded_marginal_posterior_df)
        if an_epoch_to_decode is not None:
            _was_update_success = _was_update_success and _subfn_update_value_with_context_matching(self.filter_epochs_to_decode_dict, new_context, updated_value=an_epoch_to_decode)

        return _was_update_success
    
    

    @function_attributes(short_name=None, tags=['private', 'export', 'CSV', 'main'], input_requires=[], output_provides=[], uses=['SingleFatDataframe'], used_by=['self.export_csvs'], creation_date='2025-03-13 07:12', related_items=[])
    @classmethod
    def _perform_export_dfs_dict_to_csvs(cls, extracted_dfs_dict: Dict[IdentifyingContext, pd.DataFrame], parent_output_path: Path, active_context: IdentifyingContext, session_name: str, tbin_values_dict: Dict[str, float],
                                    t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None,
                                    user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None, use_single_FAT_df: bool=True):
        """ Classmethod: export as separate .csv files. 

        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
        from pyphocorehelpers.assertion_helpers import Assert        
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType

        
        active_context = curr_active_pipeline.get_session_context()
        curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
        CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
        print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

        active_context = curr_active_pipeline.get_session_context()
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        session_name: str = curr_active_pipeline.session_name
        earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = curr_active_pipeline.find_LongShortDelta_times()
        histogram_bins = 25
        # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
        delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
        decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline)
        any_good_selected_epoch_indicies = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)

        ## Export to CSVs:

        # parent_output_path = curr_active_pipeline.get_output_path().resolve() ## Session-specific folder:
        parent_output_path = collected_outputs_path.resolve() ## Session-specific folder:
        Assert.path_exists(parent_output_path)

        ## INPUTS: collected_outputs_path
        decoding_time_bin_size: float = 0.025

        complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
        active_context = complete_session_context
        session_name: str = curr_active_pipeline.session_name
        tbin_values_dict={'laps': decoding_time_bin_size, 'pbe': decoding_time_bin_size, 'non_pbe': decoding_time_bin_size, 'FAT': decoding_time_bin_size}

        ## Build the function that uses curr_active_pipeline to build the correct filename and actually output the .csv to the right place
        def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
            output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
            out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
            export_df.to_csv(out_path)
            return out_path 

        # data_identifier_str: str = f'(MICE_marginals_df)'
        data_identifier_str: str = f'(FAT_marginals_df)'
        csv_save_paths_dict = GenericDecoderDictDecodedEpochsDictResult._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict,
                                                    parent_output_path=parent_output_path.resolve(),
                                                    active_context=active_context, session_name=session_name, #curr_active_pipeline=curr_active_pipeline,
                                                    tbin_values_dict=tbin_values_dict,
                                                    use_single_FAT_df=True,
                                                    custom_export_df_to_csv_fn=_subfn_custom_export_df_to_csv,
                                                    )
        csv_save_paths_dict


        #TODO 2024-11-01 07:53: - [ ] Need to pass the correct (full) context, including the qclu/fr_Hz filter values and the replay name. 
        #TODO 2024-11-01 07:54: - [X] Need to add a proper timebin column to the df instead of including it in the filename (if possible)
            - does already add a 'time_bin_size' column, and the suffix is probably so it doesn't get overwritten by different time_bin_sizes, might need to put them together post-hoc
            
        '2024-11-01_1250PM-kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_weighted_corr_merged_df)_tbin-1.5.csv'
            
        """
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
        from pyphocorehelpers.assertion_helpers import Assert

        assert parent_output_path.exists(), f"'{parent_output_path}' does not exist!"
        output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
        
        # active_context.custom_suffix = '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0' # '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
        
        #TODO 2024-03-02 12:12: - [ ] Could add weighted correlation if there is a dataframe for that and it's computed:
        # tbin_values_dict = {'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}
        time_col_name_dict = {'laps': 'lap_start_t', 'ripple': 'ripple_start_t', 'FAT': 't_bin_center'} ## default should be 't_bin_center'
        
        # ================================================================================================================================================================================ #
        # BEGIN SUBN BLOCK                                                                                                                                                                 #
        # ================================================================================================================================================================================ #
        # Export CSVs:

        # def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
        #     """ captures CURR_BATCH_DATE_TO_USE, `curr_active_pipeline`
        #     """
        #     output_date_str: str = deepcopy(CURR_BATCH_DATE_TO_USE)
        #     if (output_date_str is None) or (len(output_date_str) < 1):
        #         output_date_str = get_now_rounded_time_str(rounded_minutes=10)
        #     out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
        #     export_df.to_csv(out_path)
        #     return out_path 
        

        # custom_export_df_to_csv_fn = _subfn_custom_export_df_to_csv
        
        assert custom_export_df_to_csv_fn is not None, f"2025-04-05 - the default `_subfn_export_df_to_csv(...)` implementation provided below produces unparsable filenames, so require one passed for now."
        # if custom_export_df_to_csv_fn is None:
        #     def _subfn_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(single_FAT)', parent_output_path: Path=None):
        #         """ captures `active_context`, 'output_date_str' !! #TODO 2025-04-05 18:25: - [ ] session_identifier_str: seems a little long: 'kdiba_gor01_one_2006-6-09_1-22-43_normal_computed_[1, 2, 4, 6, 7, 9]_5.0' --- `data_identifier_str` is REALLY long tho: '(trained_compute_epochs:non_pbe|known_named_decoding_epochs_type:laps|masked_time_bin_fill_type:ignore)_tbin-0.025'
        #         """
        #         # parent_output_path: Path = Path('output').resolve()
        #         # active_context = curr_active_pipeline.get_session_context()
        #         session_identifier_str: str = active_context.get_description() # 'kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
        #         # session_identifier_str: str = active_context.get_description(subset_excludelist=['custom_suffix']) # no this is just the session
        #         assert output_date_str is not None
        #         out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-11-15_0200PM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays_qclu_[1, 2, 4, 6, 7, 9]_frateThresh_5.0-(ripple_WCorrShuffle_df)_tbin-0.025'
        #         out_filename = f"{out_basename}.csv"
        #         out_path = parent_output_path.joinpath(out_filename).resolve()
        #         export_df.to_csv(out_path)
        #         return out_path     
        #     custom_export_df_to_csv_fn = _subfn_export_df_to_csv

        def _subfn_pre_process_and_export_df(export_df: pd.DataFrame, a_df_identifier: Union[str, IdentifyingContext]):
            """ sets up all the important metadata and then calls `custom_export_df_to_csv_fn(....)` to actually export the CSV
            
            captures: t_start, t_delta, t_end, tbin_values_dict, time_col_name_dict, user_annotation_selections, valid_epochs_selections, custom_export_df_to_csv_fn
            """
            if isinstance(a_df_identifier, str):
                an_epochs_source_name: str = a_df_identifier.split(sep='_', maxsplit=1)[0] # get the first part of the variable names that indicates whether it's for "laps" or "ripple"
            else:
                ## probably an IdentifyingContext
                an_epochs_source_name: str = a_df_identifier.known_named_decoding_epochs_type

            a_tbin_size: float = float(tbin_values_dict[an_epochs_source_name])
            a_time_col_name: str = time_col_name_dict.get(an_epochs_source_name, 't_bin_center')
            
            ## Add t_bin column method
            export_df = export_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=a_tbin_size, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=a_time_col_name) ## #TODO 2025-04-05 18:12: - [ ] what about qclu? FrHz?
            a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
            a_data_identifier_str: str = f'({a_df_identifier})_tbin-{a_tbin_size_str}' ## build the identifier '(laps_weighted_corr_merged_df)_tbin-1.5'
            
            # add in custom columns
            #TODO 2024-03-14 06:48: - [ ] I could use my newly implemented `directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=True)` function, but since this looks at decoder-specific info it's better just to duplicate implementation and do it again here.
            # ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
            # ripple_marginals_df['ripple_start_t'] = ripple_epochs_df['start'].to_numpy()
            if (user_annotation_selections is not None):
                any_good_selected_epoch_times = user_annotation_selections.get(an_epochs_source_name, None) # like ripple
                if any_good_selected_epoch_times is not None:
                    num_valid_epoch_times: int = len(any_good_selected_epoch_times)
                    print(f'num_user_selected_times: {num_valid_epoch_times}')
                    any_good_selected_epoch_indicies = None
                    print(f'adding user annotation column!')

                    if any_good_selected_epoch_indicies is None:
                        try:
                            any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(export_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
                        except AttributeError as e:
                            print(f'ERROR: failed method 2 for {a_df_identifier}. Out of options.')        
                        except Exception as e:
                            print(f'ERROR: failed for {a_df_identifier}. Out of options.')
                        
                    if any_good_selected_epoch_indicies is not None:
                        print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} user selections) for {a_df_identifier}. got {len(any_good_selected_epoch_indicies)} indicies!')
                        export_df['is_user_annotated_epoch'] = False
                        export_df['is_user_annotated_epoch'].iloc[any_good_selected_epoch_indicies] = True
                    else:
                        print(f'\t failed all methods for annotations')

            # adds in column 'is_valid_epoch'
            if (valid_epochs_selections is not None):
                # 2024-03-04 - Filter out the epochs based on the criteria:
                any_good_selected_epoch_times = valid_epochs_selections.get(an_epochs_source_name, None) # like ripple
                if any_good_selected_epoch_times is not None:
                    num_valid_epoch_times: int = len(any_good_selected_epoch_times)
                    print(f'num_valid_epoch_times: {num_valid_epoch_times}')
                    any_good_selected_epoch_indicies = None
                    print(f'adding valid filtered epochs column!')

                    if any_good_selected_epoch_indicies is None:
                        try:
                            any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(export_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
                        except AttributeError as e:
                            print(f'ERROR: failed method 2 for {a_df_identifier}. Out of options.')        
                        except Exception as e:
                            print(f'ERROR: failed for {a_df_identifier}. Out of options.')
                        
                    if any_good_selected_epoch_indicies is not None:
                        print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} valid filter epoch times) for {a_df_identifier}. got {len(any_good_selected_epoch_indicies)} indicies!')
                        export_df['is_valid_epoch'] = False

                        try:
                            export_df['is_valid_epoch'].iloc[any_good_selected_epoch_indicies] = True
                            # a_df['is_valid_epoch'].loc[any_good_selected_epoch_indicies] = True

                        except Exception as e:
                            print(f'WARNING: trying to get whether the epochs are valid FAILED probably, 2024-06-28 custom computed epochs thing: {e}, just setting all to True')
                            export_df['is_valid_epoch'] = True
                    else:
                        print(f'\t failed all methods for selection filter')

            return custom_export_df_to_csv_fn(export_df, data_identifier_str=a_data_identifier_str, parent_output_path=parent_output_path) # this is exporting corr '(ripple_WCorrShuffle_df)_tbin-0.025'
        

        # ================================================================================================================================================================================ #
        # BEGIN FUNCTION BODY                                                                                                                                                              #
        # ================================================================================================================================================================================ #
        export_files_dict = {}
        
        if use_single_FAT_df:
            single_FAT_df: pd.DataFrame = SingleFatDataframe.build_fat_df(dfs_dict=extracted_dfs_dict, additional_common_context=active_context)
            export_files_dict['FAT'] =  _subfn_pre_process_and_export_df(export_df=single_FAT_df, a_df_identifier="FAT")
            
        else:
            for a_df_identifier, a_df in extracted_dfs_dict.items():
                export_files_dict[a_df_identifier] =  _subfn_pre_process_and_export_df(export_df=a_df, a_df_identifier=a_df_identifier) ## I bet `a_df_identifier` is an IdentifyingContext here, but a string in the above FAT case.
            # end for a_df_name, a_df
        # END if use_single_FAT_df
        
        return export_files_dict
    
    @function_attributes(short_name=None, tags=['export', 'CSV', 'main'], input_requires=['self.filter_epochs_decoded_track_marginal_posterior_df_dict'], output_provides=[], uses=['self.filter_epochs_decoded_track_marginal_posterior_df_dict', '_perform_export_dfs_dict_to_csvs'], used_by=[], creation_date='2025-03-13 08:58', related_items=[])
    def export_csvs(self, parent_output_path: Path, active_context: IdentifyingContext, decoding_time_bin_size: float, session_name: str, curr_session_t_delta: Optional[float]=None, user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None, export_df_variable_names=None, use_single_FAT_df=True, tbin_values_dict: Optional[Dict[str, float]]=None, should_export_complete_all_scores_df:bool=True):
        """ export as a single_FAT .csv file or optionally (not yet implemented) separate .csv files.    


            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType

            ## Export to CSVs:
            csv_save_paths = {}

            # parent_output_path = curr_active_pipeline.get_output_path().resolve() ## Session-specific folder:
            parent_output_path = collected_outputs_path.resolve() ## Session-specific folder:
            Assert.path_exists(parent_output_path)

            ## INPUTS: collected_outputs_path
            decoding_time_bin_size: float = 0.025

            complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
            active_context = complete_session_context
            session_name: str = curr_active_pipeline.session_name
            earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = curr_active_pipeline.find_LongShortDelta_times()
            tbin_values_dict={'laps': decoding_time_bin_size, 'pbe': decoding_time_bin_size, 'non_pbe': decoding_time_bin_size, 'FAT': decoding_time_bin_size}

            ## Build the function that uses curr_active_pipeline to build the correct filename and actually output the .csv to the right place
            def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
                output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
                out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
                export_df.to_csv(out_path)
                return out_path 
    
            # csv_save_paths_dict = GenericDecoderDictDecodedEpochsDictResult._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict,
            csv_save_paths_dict = a_new_fully_generic_result.export_csvs(
                                                        parent_output_path=parent_output_path.resolve(),
                                                        active_context=active_context, session_name=session_name, #curr_active_pipeline=curr_active_pipeline,
                                                        decoding_time_bin_size=decoding_time_bin_size,
                                                        curr_session_t_delta=t_delta,
                                                        custom_export_df_to_csv_fn=_subfn_custom_export_df_to_csv,
                                                    )
            csv_save_paths_dict


        """
        export_files_dict = {}
        
        if tbin_values_dict is None:
            tbin_values_dict = {'laps': decoding_time_bin_size, 'pbe': decoding_time_bin_size, 'non_pbe': decoding_time_bin_size, 'FAT': decoding_time_bin_size}

        ## to restrict to specific variables
        # _df_variables_names = ['laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df']
        # if export_df_variable_names is None:
        #     # export all by default
        #     export_df_variable_names = _df_variables_names
            
        extracted_dfs_dict: Dict[IdentifyingContext, pd.DataFrame] = self.filter_epochs_decoded_track_marginal_posterior_df_dict # {a_df_name:getattr(self, a_df_name) for a_df_name in export_df_variable_names}
        if len(extracted_dfs_dict) > 0:
            export_files_dict = export_files_dict | self._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=extracted_dfs_dict, parent_output_path=parent_output_path, tbin_values_dict=tbin_values_dict,
                                                                                          active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=user_annotation_selections, valid_epochs_selections=valid_epochs_selections, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
                                                                                          use_single_FAT_df=use_single_FAT_df)


        # ## try to export the merged all_scores dataframe
        # if should_export_complete_all_scores_df:
        #     extracted_merged_scores_df: pd.DataFrame = self.build_complete_all_scores_merged_df()
        #     if 'time_bin_size' not in extracted_merged_scores_df.columns:
        #         ## add the column
        #         print(f'WARN: adding the time_bin_size columns: {self.ripple_decoding_time_bin_size}')
        #         extracted_merged_scores_df['time_bin_size'] = self.ripple_decoding_time_bin_size

        #     export_df_dict = {'ripple_all_scores_merged_df': extracted_merged_scores_df}
        #     export_files_dict = export_files_dict | self.perform_export_dfs_dict_to_csvs(extracted_dfs_dict=export_df_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)

        return export_files_dict


    def __repr__(self):
        """Custom multi-line representation for BinningContainer
        renders like:
        ```
            GenericDecoderDictDecodedEpochsDictResult(
                filter_epochs_to_decode_dict=<['known_named_decoding_epochs_type:laps', 'known_named_decoding_epochs_type:pbe', 'known_named_decoding_epochs_type:non_pbe']>,
                filter_epochs_pseudo2D_continuous_specific_decoded_result=<['known_named_decoding_epochs_type:laps', 'known_named_decoding_epochs_type:pbe', 'known_named_decoding_epochs_type:non_pbe']>,
                filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict=<['known_named_decoding_epochs_type:laps|masked_time_bin_fill_type:ignore', 'known_named_decoding_epochs_type:laps|masked_time_bin_fill_type:last_valid', 'known_named_decoding_epochs_type:laps|masked_time_bin_fill_type:nan_filled', 'known_named_decoding_epochs_type:pbe|masked_time_bin_fill_type:ignore', 'known_named_decoding_epochs_type:pbe|masked_time_bin_fill_type:last_valid', 'known_named_decoding_epochs_type:pbe|masked_time_bin_fill_type:nan_filled', 'known_named_decoding_epochs_type:non_pbe|masked_time_bin_fill_type:ignore', 'known_named_decoding_epochs_type:non_pbe|masked_time_bin_fill_type:last_valid', 'known_named_decoding_epochs_type:non_pbe|masked_time_bin_fill_type:nan_filled']>
            )
        ```
        """
        def _subfn_format_keys_of_IdentifyingContext(a_dict):
            ## prints just the keys and values
            return f"<{[v.get_description(include_property_names=True, key_value_separator=':', separator='|') for v in list(a_dict.keys())]}>" # like 'known_named_decoding_epochs_type:laps|masked_time_bin_fill_type:last_valid'

        # Get the string representations of each field
        spikes_df_dict_repr = _subfn_format_keys_of_IdentifyingContext(self.spikes_df_dict)
        # decoder_trained_compute_epochs_dict_repr = _subfn_format_keys_of_IdentifyingContext(self.decoder_trained_compute_epochs_dict)
        decoders_repr = _subfn_format_keys_of_IdentifyingContext(self.decoders)
        filter_epochs_to_decode_dict_repr = _subfn_format_keys_of_IdentifyingContext(self.filter_epochs_to_decode_dict)
        filter_epochs_specific_decoded_result_repr = _subfn_format_keys_of_IdentifyingContext(self.filter_epochs_specific_decoded_result)
        filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict_repr = _subfn_format_keys_of_IdentifyingContext(self.filter_epochs_decoded_track_marginal_posterior_df_dict)
        

        # Format with indentation and line breaks
        return (f"GenericDecoderDictDecodedEpochsDictResult(\n"
                f"    spikes_df_dict={spikes_df_dict_repr},\n"
                # f"    decoder_trained_compute_epochs_dict={decoder_trained_compute_epochs_dict_repr},\n"
                f"    decoders={decoders_repr},\n"
                f"    filter_epochs_to_decode_dict={filter_epochs_to_decode_dict_repr},\n"
                f"    filter_epochs_specific_decoded_result={filter_epochs_specific_decoded_result_repr},\n"
                f"    filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict={filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict_repr}\n"
                f")")

    @classmethod
    def _perform_per_epoch_time_bin_aggregation(cls, a_decoded_time_bin_marginal_posterior_df: pd.DataFrame, probabilitY_column_to_aggregate:str='P_Short', n_rolling_avg_window_tbins: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ For a single marginal_df, performs aggregation to get the corresponding 'per_epoch' value.
        
        Usage:        
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult

            a_decoded_per_epoch_marginals_df, a_decoded_time_bin_marginal_posterior_df = GenericDecoderDictDecodedEpochsDictResult._perform_per_epoch_time_bin_aggregation(a_decoded_time_bin_marginal_posterior_df=a_decoded_time_bin_marginal_posterior_df, probabilitY_column_to_aggregate='P_Short', n_rolling_avg_window_tbins=3)

        """
        from neuropy.analyses.time_bin_aggregation import TimeBinAggregation

        ## INPUTS: a_decoded_time_bin_marginal_posterior_df

        
        # Create a copy to avoid modifying the original
        # result_df = a_decoded_time_bin_marginal_posterior_df.copy()
        result_df = deepcopy(a_decoded_time_bin_marginal_posterior_df)
        epoch_partitioned_dfs_dict = a_decoded_time_bin_marginal_posterior_df.pho.partition_df_dict(partitionColumn='parent_epoch_label')

        # Process each partition
        for k, df in epoch_partitioned_dfs_dict.items():
            rolling_avg = TimeBinAggregation.ToPerEpoch.peak_rolling_avg(df=df, column=probabilitY_column_to_aggregate, window=n_rolling_avg_window_tbins)    
            # Calculate the mean of P_Short for this group
            mean_p_short = TimeBinAggregation.ToPerEpoch.mean(df=df, column=probabilitY_column_to_aggregate)

            # Get indices from this partition
            indices = df.index
            # Assign the result to the corresponding rows in the result dataframe
            result_df.loc[indices, f'rolling_avg_{probabilitY_column_to_aggregate}'] = rolling_avg
            result_df.loc[indices, f'mean_{probabilitY_column_to_aggregate}'] = mean_p_short  # Same mean value for all rows in group
            
            # result_df.loc[indices

        ## OUTPUTS: result_df

        # Then keep only the first entry for each 'parent_epoch_label'
        a_decoded_per_epoch_marginals_df = a_decoded_time_bin_marginal_posterior_df.groupby('parent_epoch_label').first().reset_index()
        
        ## fixup `data_grain` column so it's now correct
        a_decoded_per_epoch_marginals_df = TimeBinAggregation.ToPerEpoch.fixup_data_grain_to_per_epoch(df=a_decoded_per_epoch_marginals_df)

        return a_decoded_per_epoch_marginals_df, a_decoded_time_bin_marginal_posterior_df

        ## OUTPUTS: a_decoded_time_bin_marginal_posterior_df, a_decoded_per_epoch_marginals_df
        ## Columns of interest: f'rolling_avg_{probabilitY_column_to_aggregate}' (e.g. 'rolling_avg_P_Short')
        



    @classmethod
    def _perform_all_per_time_bin_to_per_epoch_aggregations(cls, a_new_fully_generic_result: "GenericDecoderDictDecodedEpochsDictResult", flat_decoded_marginal_posterior_df_context_dict: Dict[IdentifyingContext, pd.DataFrame], probabilitY_column_to_aggregate:str='P_Short', n_rolling_avg_window_tbins: int = 3, **kwargs) -> Tuple["GenericDecoderDictDecodedEpochsDictResult", Tuple[Dict[IdentifyingContext, IdentifyingContext], Dict[IdentifyingContext, pd.DataFrame], Dict[IdentifyingContext, pd.DataFrame]]]:
        """ For a single marginal_df, performs aggregation to get the corresponding 'per_epoch' value.
        
        Usage:        
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
            from neuropy.utils.result_context import IdentifyingContext, CollisionOutcome
            from neuropy.analyses.time_bin_aggregation import TimeBinAggregation

            ## get all non-global, `data_grain= 'per_time_bin'`
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=IdentifyingContext(trained_compute_epochs='laps', decoder_identifier='pseudo2D', time_bin_size=0.025,
                                                                                                                                                                                                                                known_named_decoding_epochs_type=['pbe', 'laps'],
                masked_time_bin_fill_type=('ignore', 'dropped'),
                # masked_time_bin_fill_type='dropped',
                data_grain= 'per_time_bin'))        
                
            a_new_fully_generic_result, (per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict) = GenericDecoderDictDecodedEpochsDictResult._perform_all_per_time_bin_to_per_epoch_aggregations(a_new_fully_generic_result=a_new_fully_generic_result,
                    flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict)

        """
        from neuropy.analyses.time_bin_aggregation import TimeBinAggregation

        ## INPUTS: a_new_fully_generic_result
        ## INPUTS: a_decoded_time_bin_marginal_posterior_df

        ## INPUTS: flat_decoded_marginal_posterior_df_context_dict
        per_time_bin_to_per_epoch_context_map_dict = {}
        flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict = {}
        flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict = {}
        for a_per_time_bin_ctxt, a_decoded_time_bin_marginal_posterior_df in flat_decoded_marginal_posterior_df_context_dict.items():
            a_per_epoch_ctxt = TimeBinAggregation.ToPerEpoch.get_per_epoch_ctxt_from_per_time_bin_ctxt(a_per_time_bin_ctxt=a_per_time_bin_ctxt)
            


            a_decoded_per_epoch_marginals_df, a_decoded_time_bin_marginal_posterior_df = cls._perform_per_epoch_time_bin_aggregation(a_decoded_time_bin_marginal_posterior_df=a_decoded_time_bin_marginal_posterior_df, probabilitY_column_to_aggregate=probabilitY_column_to_aggregate, n_rolling_avg_window_tbins=n_rolling_avg_window_tbins, **kwargs)            
            per_time_bin_to_per_epoch_context_map_dict[a_per_time_bin_ctxt] = a_per_epoch_ctxt
            flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict[a_per_time_bin_ctxt] = deepcopy(a_decoded_time_bin_marginal_posterior_df)
            flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict[a_per_epoch_ctxt] = a_decoded_per_epoch_marginals_df

            ## INLINE UPDATE
            a_decoded_time_bin_marginal_posterior_df = deepcopy(a_decoded_time_bin_marginal_posterior_df)
            a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict[a_per_time_bin_ctxt] = a_decoded_time_bin_marginal_posterior_df
            a_best_matching_context, a_result, a_decoder, a_decoded_time_bin_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(a_per_time_bin_ctxt)
            # a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_context(a_per_time_bin_ctxt, return_multiple_matches=False)
            # a_result
            # print(f'updating: "{a_per_epoch_ctxt}"')
            # print(f"\tWARN: TODO 2025-04-07 19:22: - [ ] a_result is wrong, it's the per-time-bin version not the per-epoch version") #TODO 2025-04-07 19:22: - [ ] a_result is wrong, it's the per-time-bin version not the per-epoch version
            ## #TODO 2025-04-08 07:05: - [ ] Actually this is surprisingly just fine, as the incoming `a_result` which is supposed to belong to the 'per_time_bin' context actually has the original 'per_epoch' values that were split up anyway! (e.g. `len(a_result.filter_epochs)`: 84 laps, 
            ## need to get updated a_decoder, a_result
            a_new_fully_generic_result.updating_results_for_context(new_context=deepcopy(a_per_epoch_ctxt), a_result=deepcopy(a_result), a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=deepcopy(a_decoded_per_epoch_marginals_df)) ## update using the result
            

        ## OUTPUTS: per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict
        # flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict
        # flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict
                

        ## INPUTS: per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict
        # flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict
        # flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict


        return a_new_fully_generic_result, (per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict)


    def compute_all_per_epoch_aggregations_from_per_time_bin_results(self, flat_decoded_marginal_posterior_df_context_dict: Dict[IdentifyingContext, pd.DataFrame], **kwargs) -> Tuple[Dict[IdentifyingContext, IdentifyingContext], Dict[IdentifyingContext, pd.DataFrame], Dict[IdentifyingContext, pd.DataFrame]]:
        """ For a single marginal_df, performs aggregation to get the corresponding 'per_epoch' value.
        
        Usage:        
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
            from neuropy.utils.result_context import IdentifyingContext, CollisionOutcome
            from neuropy.analyses.time_bin_aggregation import TimeBinAggregation

            ## get all non-global, `data_grain= 'per_time_bin'`
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=IdentifyingContext(trained_compute_epochs='laps', decoder_identifier='pseudo2D', time_bin_size=0.025,
                                                                                                                                                                                                                                known_named_decoding_epochs_type=['pbe', 'laps'],
                masked_time_bin_fill_type=('ignore', 'dropped'),
                # masked_time_bin_fill_type='dropped',
                data_grain= 'per_time_bin'))        
                
            _newly_updated_values_tuple = a_new_fully_generic_result.compute_all_per_epoch_aggregations_from_per_time_bin_results(flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict)
            per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict = _newly_updated_values_tuple
            
        """
        
        self, _newly_updated_values_tuple = GenericDecoderDictDecodedEpochsDictResult._perform_all_per_time_bin_to_per_epoch_aggregations(a_new_fully_generic_result=self, flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict, **kwargs)
        # per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict = _newly_updated_values_tuple
        return _newly_updated_values_tuple

