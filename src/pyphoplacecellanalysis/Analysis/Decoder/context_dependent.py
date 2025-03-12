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
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult, DirectionalPseudo2DDecodersResult, EpochFilteringMode
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



# @define(slots=False, repr=False)
# class GeneralizedPseudo2DDecodersResult(ComputedResult):
#     """ a container for holding information regarding the computation of merged (pseudo2D) directional placefields.

#     From `DirectionalPseudo2DDecodersResult`
    
#     #TODO 2024-05-22 17:26: - [ ] 'DirectionalMergedDecodersResult' -> 'DirectionalPseudo2DDecodersResult'

#     #TODO 2025-03-04 17:35: - [ ] Limited in the following ways:
#         - Assumes directional (long/short) configurations, not general
#         - Assumes two epochs of interest: (laps/ripples)
#         - Outputs a bunch of separate marginal files which seems excessive
        
    
#     """
#     _VersionedResultMixin_version: str = "2024.10.09_0" # to be updated in your IMPLEMENTOR to indicate its version
    
#     all_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
#     all_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    
#     long_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
#     long_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
#     short_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
#     short_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)

#     # Posteriors computed via the all_directional decoder:
#     all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)
#     all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)

#     # Marginalized posteriors computed from above posteriors:
#     laps_directional_marginals_tuple: Tuple = serialized_field(default=None) # laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
#     laps_track_identity_marginals_tuple: Tuple = serialized_field(default=None)
#     laps_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Laps'})
    
#     ripple_directional_marginals_tuple: Tuple = serialized_field(default=None)
#     ripple_track_identity_marginals_tuple: Tuple = serialized_field(default=None) 
#     ripple_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Replay'})


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

KnownNamedDecoderTrainedComputeEpochsType = Literal['laps', 'non_pbe']

KnownNamedDecodingEpochsType = Literal['laps', 'replay', 'ripple', 'pbe', 'non_pbe']
# Define a type that can only be one of these specific strings
MaskedTimeBinFillType = Literal['ignore', 'last_valid', 'nan_filled', 'dropped'] ## used in `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(...)` to specify how invalid bins (due to too few spikes) are treated.

DataTimeGrain = Literal['per_epoch', 'per_time_bin']
GenericResultTupleIndexType: TypeAlias = IdentifyingContext # an template/stand-in variable that aims to abstract away the unique-hashable index of a single result computed with a given set of parameters. Not yet fully implemented 2025-03-09 17:50 
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

# @metadata_attributes(short_name=None, tags=['Generic', 'Improved'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 07:53', related_items=['GeneralDecoderDictDecodedEpochsDictResult'])
@define(slots=False, eq=False)
class GenericDecoderDictDecodedEpochsDictResult(ComputedResult):
    """ General flat-unordered-tuple-like indicies
    
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
    spikes_df_dict: Dict[GenericResultTupleIndexType, pd.DataFrame] = serialized_field(default=Factory(dict), repr=keys_only_repr, metadata={'field_added': "2025.03.11_0"}) # global

    # decoder_trained_compute_epochs_dict: Dict[GenericResultTupleIndexType, Epoch] = serialized_field(default=None, metadata={'field_added': "2025.03.11_0"}) ## Is this needed, or are they present in the computation_epochs in the decoder/pf?

    # original_pfs_dict: Dict[types.DecoderName, PfND]
    decoders: Dict[GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=Factory(dict), repr=keys_only_repr, metadata={'field_added': "2025.03.11_0"}) # Combines both pf1D_Decoder_dict and pseudo2D_decoder-type decoders (both 1D and 2D as well) by indexing via context
    # pf1D_Decoder_dict: Dict[GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=None, metadata={'field_added': "2024.01.16_0"})
    # pseudo2D_decoder: Dict[GenericResultTupleIndexType, BasePositionDecoder] = serialized_field(default=None, metadata={'field_added': "2024.01.22_0"})
    
    ## Result Keys
    filter_epochs_to_decode_dict: Dict[GenericResultTupleIndexType, Epoch] = serialized_field(default=Factory(dict), repr=keys_only_repr)
    filter_epochs_specific_decoded_result: Dict[GenericResultTupleIndexType, DecodedFilterEpochsResult] = serialized_field(default=Factory(dict), repr=keys_only_repr) ## why is this labeled as if they have to be continuous or Pseudo2D? They can be any result right?
    filter_epochs_decoded_track_marginal_posterior_df_dict: Dict[GenericResultTupleIndexType, pd.DataFrame] = serialized_field(default=Factory(dict), repr=keys_only_repr)



    # ================================================================================================================================================================================ #
    # Additive from old results objects                                                                                                                                                #
    # ================================================================================================================================================================================ #
    @classmethod
    def init_from_old_GeneralDecoderDictDecodedEpochsDictResult(cls, a_general_decoder_dict_decoded_epochs_dict_result) -> "GenericDecoderDictDecodedEpochsDictResult":
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
    def adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(self, a_general_decoder_dict_decoded_epochs_dict_result) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ adds the results from a 2025-03-10 result (`GeneralDecoderDictDecodedEpochsDictResult`) to the new 2025-03-11 format (`GenericDecoderDictDecodedEpochsDictResult`), in-place, but returning itself

        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType

        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult()  # start empty
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = a_new_fully_generic_result.adding_from_old_GeneralDecoderDictDecodedEpochsDictResult(a_general_decoder_dict_decoded_epochs_dict_result=a_general_decoder_dict_decoded_epochs_dict_result)
        a_new_fully_generic_result
        """
        _shared_context_fragment = {'trained_compute_epochs': 'non_pbe'}
        for a_known_epoch_name, an_epoch in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_to_decode_dict.items():
            a_new_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name)
            self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(an_epoch)
            
            self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_pseudo2D_continuous_specific_decoded_result[a_known_epoch_name])
            for a_known_t_bin_fill_type, a_posterior_df in a_general_decoder_dict_decoded_epochs_dict_result.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_known_epoch_name].items():
                a_new_joint_identifier = IdentifyingContext(**_shared_context_fragment, known_named_decoding_epochs_type=a_known_epoch_name, masked_time_bin_fill_type=a_known_t_bin_fill_type)
                self.filter_epochs_decoded_track_marginal_posterior_df_dict[a_new_joint_identifier] = deepcopy(a_posterior_df)

        return self
    

    # @function_attributes(short_name=None, tags=['adding'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 13:16', related_items=[])
    def adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(self, directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL

        # a_new_fully_generic_result = _subfn_filter_by_spikes_per_t_bin_masked_and_add_to_generic_result(a_new_fully_generic_result, directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)
        a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)

        """
        filtered_epochs_df = None ## parameter just to allow an override I think
        

        decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict)
        decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict)

        # ==================================================================================================================== #
        # Reuse the old/previously computed version of the result with the additional properties                               #
        # ==================================================================================================================== #
        ## These are of type `trained_compute_epochs` -- e.g. trained_compute_epochs='laps'  || trained_compute_epochs='non_pbe'
        trained_compute_epochs_dict_dict = {'laps': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## only laps were ever trained to decode until the non-PBEs
                                            # 'pbe': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## PBEs were never used to decode, only laps
        }

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
            
        Assert.all_equal(epochs_decoding_time_bin_size_dict.values())
        epochs_decoding_time_bin_size: float = list(epochs_decoding_time_bin_size_dict.values())[0]
        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        print(f'{pos_bin_size = }, {epochs_decoding_time_bin_size = }')


        ## OUTPUTS: filtered_decoder_filter_epochs_decoder_result_dict_dict, epochs_decoding_time_bin_size

        ## Perform the decoding and masking as needed for invalid bins:

        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result_dict in decoder_filter_epochs_result_dict_dict.items():

            for a_decoder_name, a_decoded_epochs_result in decoder_ripple_filter_epochs_decoder_result_dict.items():
                ## build the complete identifier
                a_new_identifier: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier=a_decoder_name, time_bin_size=epochs_decoding_time_bin_size, known_named_decoding_epochs_type=a_known_decoded_epochs_type, masked_time_bin_fill_type='ignore')
                self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_decoded_epochs_result)
                self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(a_decoded_epochs_result.filter_epochs) ## needed? Do I want full identifier as key?
                # a_new_fully_generic_result.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_new_identifier] ## #TODO 2025-03-11 11:39: - [ ] must be computed or assigned from prev result
                


        # directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug
                                                                                    
        # extracted_merged_scores_df = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug_print=True)
        # extracted_merged_scores_df
        ## Inputs: a_new_fully_generic_result

        return self



    # @function_attributes(short_name=None, tags=['adding', 'pseudo2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 14:24', related_items=[])
    def adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(self, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self

        Usage:
            directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
            a_new_fully_generic_result = a_new_fully_generic_result.adding_directional_pseudo2D_decoder_results_filtered_by_spikes_per_t_bin_masked(directional_merged_decoders_result=directional_merged_decoders_result)

        """
        filtered_epochs_df = None ## parameter just to allow an override I think
        
        # DirectionalMergedDecoders: Get the result after computation:
        
        ## NOTE, HAVE:
        all_directional_pf1D_Decoder: BasePositionDecoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
        all_directional_decoder_dict: Dict[str, BasePositionDecoder] = directional_merged_decoders_result.all_directional_decoder_dict
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
        trained_compute_epochs_dict_dict = {'laps': ensure_Epoch(deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs)), ## only laps were ever trained to decode until the non-PBEs
                                            # 'pbe': ensure_Epoch(deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)), ## PBEs were never used to decode, only laps
        }

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
        
        for a_known_decoded_epochs_type, a_decoder_epochs_filter_epochs_decoder_result in decoder_filter_epochs_result_dict.items():
            ## build the complete identifier
            a_new_identifier: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier=a_decoder_name, time_bin_size=epochs_decoding_time_bin_size, known_named_decoding_epochs_type=a_known_decoded_epochs_type, masked_time_bin_fill_type='ignore')
            self.filter_epochs_specific_decoded_result[a_new_identifier] = deepcopy(a_decoder_epochs_filter_epochs_decoder_result)
            # self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(a_decoder_epochs_filter_epochs_decoder_result.filter_epochs) ## needed? Do I want full identifier as key?            
            ## use the filtered approach instead:
            self.filter_epochs_to_decode_dict[a_new_identifier] = deepcopy(filtered_decoder_filter_epochs_decoder_result_dict[a_known_decoded_epochs_type])

            # self.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_new_identifier] ## #TODO 2025-03-11 11:39: - [ ] must be computed or assigned from prev result
            self.decoders[a_new_identifier] = all_directional_pf1D_Decoder ## this will duplicate this decoder needlessly for each repetation here, but that's okay for now
            for a_known_data_grain, a_decoded_marginals_df in decoder_epoch_marginals_df_dict_dict[a_known_decoded_epochs_type].items():
                a_new_data_grain_identifier: IdentifyingContext = a_new_identifier.adding_context_if_missing(data_grain=a_known_data_grain)
                self.filter_epochs_decoded_track_marginal_posterior_df_dict[a_new_data_grain_identifier] = deepcopy(a_decoded_marginals_df) 

            for an_individual_decoder_name, an_individual_directional_decoder in all_directional_decoder_dict.items():
                a_new_individual_decoder_identifier: IdentifyingContext = a_new_identifier.overwriting_context(decoder_identifier=an_individual_decoder_name) # replace 'decoder_identifier'
                self.decoders[a_new_individual_decoder_identifier] = deepcopy(an_individual_directional_decoder) 

                

        # directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug
                                                                                    
        # extracted_merged_scores_df = directional_decoders_epochs_decode_result.build_complete_all_scores_merged_df(debug_print=True)
        # extracted_merged_scores_df
        ## Inputs: a_new_fully_generic_result

        return self
    


    # ================================================================================================================================================================================ #
    # Self-compute                                                                                                                                                  #
    # ================================================================================================================================================================================ #
    
    # @function_attributes(short_name=None, tags=['adding', 'generating', 'masking'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 13:22', related_items=[])
    def creating_new_spikes_per_t_bin_masked_variants(self, spikes_df: pd.DataFrame) -> "GenericDecoderDictDecodedEpochsDictResult":
        """ Takes the previously computed results and produces versions with each time bin masked by a required number of spike counts/participation.
        Updates in-places, creating new entries, but also returns self
        
        spikes_df = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        a_new_fully_generic_result = a_new_fully_generic_result.creating_new_spikes_per_t_bin_masked_variants(spikes_df=spikes_df)
        """
        _new_results_to_add = {} ## need a temporary entry so we aren't modifying the dict property `a_new_fully_generic_result.filter_epochs_specific_decoded_result` while we update it
        for a_context, a_decoded_filter_epochs_result in self.filter_epochs_specific_decoded_result.items():
            a_modified_context = deepcopy(a_context)
            a_spikes_df = deepcopy(spikes_df)
            a_masked_decoded_filter_epochs_result, _mask_index_tuple = a_decoded_filter_epochs_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=a_spikes_df)
            a_modified_context = a_modified_context.adding_context_if_missing(masked_time_bin_fill_type='last_valid')
            _new_results_to_add[a_modified_context] = a_masked_decoded_filter_epochs_result
            ## can directly add the others that we aren't iterating over
            self.spikes_df_dict[a_modified_context] = deepcopy(a_spikes_df) ## TODO: reduce the context?

        print(f'computed {len(_new_results_to_add)} new results')

        self.filter_epochs_specific_decoded_result.update(_new_results_to_add)
        

        return self

    @function_attributes(short_name=None, tags=['UNFINISHED', 'UNTESTED', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 00:00', related_items=[])
    def example_compute_fn(self, curr_active_pipeline, context: IdentifyingContext):
        """ Uses the context to extract proper values from the pipeline, and performs a fresh computation
        
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

        # t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        # single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        # single_global_epoch: Epoch = Epoch(single_global_epoch_df)

        single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

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


        #TODO 2025-03-11 09:10: - [ ] Get proper compute epochs from the context
        trained_compute_epochs = deepcopy(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[a_name]].non_pbe)
        
        new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=trained_compute_epochs) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        # new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(a_new_training_df_dict[a_name])) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
        
        #TODO 2025-03-11 09:10: - [ ] Get proper decode epochs from the context
        decode_epochs = deepcopy(single_global_epoch)

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
        a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = self.a_new_NonPBE_Epochs_obj
        results1D: NonPBEDimensionalDecodingResult = self.results1D
        # results2D: NonPBEDimensionalDecodingResult = self.results2D

        epochs_decoding_time_bin_size = self.epochs_decoding_time_bin_size
        frame_divide_bin_size = self.frame_divide_bin_size

        print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')

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

        if filter_epochs is None:
            # use global epoch
            # single_global_epoch_df: pd.DataFrame = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            single_global_epoch: Epoch = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            filter_epochs = single_global_epoch


        # takes 6.3 seconds
        ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(filter_epochs),
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
        filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch] = {'laps': ensure_Epoch(deepcopy(global_any_laps_epochs_obj)),
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
        
        filter_epochs_pseudo2D_continuous_specific_decoded_result: Dict[KnownNamedDecodingEpochsType, DecodedFilterEpochsResult] = {}
        filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[KnownNamedDecodingEpochsType, Dict[MaskedTimeBinFillType, pd.DataFrame]] = {}
        
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
            decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[MaskedTimeBinFillType, pd.DataFrame] = {# 'track_marginal_posterior_df':track_marginal_posterior_df,
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


    # ================================================================================================================================================================================ #
    # Retreval and use                                                                                                                                                                 #
    # ================================================================================================================================================================================ #
    def get_matching_contexts(self, context_query: IdentifyingContext, return_multiple_matches: bool=False, debug_print:bool=True):
        """ Get a specific contexts
        
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='long_LR', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore')
        """
        
        def _subfn_get_value_with_context_matching(dictionary, query, item_name="item"):
            """Helper function to get a value from a dictionary with context matching."""
            try:
                value = dictionary[query]  # Try exact match first
                return query, value
            except KeyError:
                # Find single best matching context
                if not return_multiple_matches:
                    best_match, match_count = IdentifyingContext.find_best_matching_context(query, dictionary)
                    if best_match:
                        if debug_print:
                            print(f"Found best match for {item_name} with {match_count} matching attributes:\t{best_match}\n")
                        return best_match, dictionary[best_match]
                    else:
                        if debug_print:
                            print(f"{item_name}: No matches found in the dictionary.")
                        return None, None
                else:
                    # Find multiple matching contexts
                    matching_contexts, match_count = IdentifyingContext.find_best_matching_contexts(query, dictionary)
                    if matching_contexts:
                        if debug_print:
                            print(f"Found {len(matching_contexts)} matches for {item_name}")
                        # return [(ctx, dictionary[ctx]) for ctx in matching_contexts]
                        return {ctx:dictionary[ctx] for ctx in matching_contexts}                        
                    else:
                        if debug_print:
                            print(f"{item_name}: No matches found in the dictionary.")
                        return {}
                    
            except Exception as e:
                raise e



        if (not return_multiple_matches):
            # Find single best matching context
            # Get all values using the helper function - one line per call
            result_context, a_result = _subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result")
            decoder_context, a_decoder = _subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder")
            posterior_context, a_decoded_marginal_posterior_df = _subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df")
            

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
            # Find multiple matching contexts
            # Get all values using the helper function - one line per call
            result_context_dict = _subfn_get_value_with_context_matching(self.filter_epochs_specific_decoded_result, context_query, "a_result")
            decoder_context_dict = _subfn_get_value_with_context_matching(self.decoders, context_query, "a_decoder")
            decoded_marginal_posterior_df_context_dict = _subfn_get_value_with_context_matching(self.filter_epochs_decoded_track_marginal_posterior_df_dict, context_query, "a_decoded_marginal_posterior_df")            
            any_matching_contexts_list = list(set(list(result_context_dict.keys())).union(set(list(decoder_context_dict.keys()))).union(set(list(decoded_marginal_posterior_df_context_dict.keys()))))
            return any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict
 

    def get_flattened_contexts_for_posteriors_dfs(self, decoded_marginal_posterior_df_context_dict):
        """ returns 4 flat dicts with the same (full) contexts that the passed `decoded_marginal_posterior_df_context_dict` have
        
        Usage:
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_flattened_contexts_for_posteriors_dfs(decoded_marginal_posterior_df_context_dict)
            flat_context_list

        """
        flat_context_list = []
        flat_decoder_context_dict = {}
        flat_result_context_dict = {}

        for a_context, a_df in decoded_marginal_posterior_df_context_dict.items():
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = self.get_matching_contexts(context_query=a_context, return_multiple_matches=False, debug_print=False)
            assert best_matching_context == a_context, f"best_matching_context: {best_matching_context}, a_context: {a_context}"
            flat_decoder_context_dict[best_matching_context] = a_decoder
            flat_result_context_dict[best_matching_context] = a_result
            flat_context_list.append(best_matching_context)


        assert len(flat_decoder_context_dict) == len(flat_result_context_dict)
        Assert.same_length(flat_context_list, decoded_marginal_posterior_df_context_dict, flat_decoder_context_dict, flat_result_context_dict)
        return flat_context_list, flat_result_context_dict, flat_decoder_context_dict, decoded_marginal_posterior_df_context_dict





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


# @define(slots=False, repr=False)
# class GeneralDecoderDecodedEpochsResult(ComputedResult):
#     """ Contains Decoded Epochs (such as laps, ripple) for a each of the Decoders.

#     2024-02-15 - Computed by `_decode_and_evaluate_epochs_using_directional_decoders`
    
#     from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
#     """ 

#     _VersionedResultMixin_version: str = "2025.03.07_0" # Initial: 2025.03.07_0
    
#     pos_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)
#     ripple_decoding_time_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)
#     laps_decoding_time_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)

#     decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field(default=None)
#     decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field(default=None)

#     decoder_laps_radon_transform_df_dict: Dict = serialized_field(default=None)
#     decoder_ripple_radon_transform_df_dict: Dict = serialized_field(default=None)
        
#     decoder_laps_radon_transform_extras_dict: Dict = non_serialized_field(default=None) # non-serialized
#     decoder_ripple_radon_transform_extras_dict: Dict = non_serialized_field(default=None) # non-serialized
        
#     laps_weighted_corr_merged_df: pd.DataFrame = serialized_field(default=None)
#     ripple_weighted_corr_merged_df: pd.DataFrame = serialized_field(default=None)
#     decoder_laps_weighted_corr_df_dict: Dict = serialized_field(default=Factory(dict))
#     decoder_ripple_weighted_corr_df_dict: Dict = serialized_field(default=Factory(dict))
    
#     laps_simple_pf_pearson_merged_df: pd.DataFrame = serialized_field(default=None)
#     ripple_simple_pf_pearson_merged_df: pd.DataFrame = serialized_field(default=None)
    
#     @classmethod
#     def compute_matching_best_indicies(cls, a_marginals_df: pd.DataFrame, index_column_name: str = 'most_likely_decoder_index', second_index_column_name: str = 'best_decoder_index', enable_print=True):
#         """ count up the number of rows that the RadonTransform and the most-likely direction agree 
        
#         DecoderDecodedEpochsResult.compute_matching_best_indicies

#         """
#         num_total_epochs: int = len(a_marginals_df)
#         agreeing_rows_count: int = (a_marginals_df[index_column_name] == a_marginals_df[second_index_column_name]).sum()
#         agreeing_rows_ratio = float(agreeing_rows_count)/float(num_total_epochs)
#         if enable_print:
#             print(f'agreeing_rows_count/num_total_epochs: {agreeing_rows_count}/{num_total_epochs}\n\tagreeing_rows_ratio: {agreeing_rows_ratio}')
#         return agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs)


#     @classmethod
#     def add_session_df_columns(cls, df: pd.DataFrame, session_name: str, time_bin_size: float=None, t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None, time_col: str=None, end_time_col_name: Optional[str]=None) -> pd.DataFrame:
#         """ adds session-specific information to the marginal dataframes 
    
#         Added Columns: ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']

#         Usage:
#             # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
#             session_name: str = curr_active_pipeline.session_name
#             t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
#             df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
            
#             a_ripple_df = DecoderDecodedEpochsResult.add_session_df_columns(a_ripple_df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
    
#         """
#         from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
        
#         return df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=time_bin_size, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=time_col, end_time_col_name=end_time_col_name)
    
#     @classmethod
#     @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 22:58', related_items=[])
#     def load_user_selected_epoch_times(cls, curr_active_pipeline, track_templates=None, epochs_name='ripple', **additional_selections_context) -> Tuple[Dict[str, NDArray], NDArray]:
#         """

#         Usage:    
#             decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = load_user_selected_epoch_times(curr_active_pipeline)
#             # Finds the indicies into the dataframe (`filtered_ripple_simple_pf_pearson_merged_df`) from the decoder_user_selected_epoch_times_dict
#             # Inputs: filtered_ripple_simple_pf_pearson_merged_df, decoder_user_selected_epoch_times_dict

#             new_selections_dict = {}
#             for a_name, a_start_stop_arr in decoder_user_selected_epoch_times_dict.items():
#                 # a_pagination_controller = self.pagination_controllers[a_name] # DecodedEpochSlicesPaginatedFigureController
#                 if len(a_start_stop_arr) > 0:
#                     assert np.shape(a_start_stop_arr)[1] == 2, f"input should be start, stop times as a numpy array"
#                     # new_selections_dict[a_name] = filtered_ripple_simple_pf_pearson_merged_df.epochs.find_data_indicies_from_epoch_times(a_start_stop_arr) # return indicies into dataframe
#                     new_selections_dict[a_name] = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(a_start_stop_arr) # return sliced dataframes
                    
#             new_selections_dict

#         """
#         # Inputs: curr_active_pipeline (for curr_active_pipeline.build_display_context_for_session)
#         from neuropy.utils.misc import numpyify_array
#         from neuropy.core.user_annotations import UserAnnotationsManager
#         annotations_man = UserAnnotationsManager()
#         user_annotations = annotations_man.get_user_annotations()

#         if track_templates is None:
#             # Get from the pipeline:
#             directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
#             rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
#             minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
#             included_qclu_values: float = rank_order_results.included_qclu_values
#             track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only
        

#         # loaded_selections_context_dict = {a_name:curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections') for a_name, a_decoder in track_templates.get_decoders_dict().items()}
#         loaded_selections_context_dict = {a_name:curr_active_pipeline.sess.get_context().merging_context('display_', IdentifyingContext(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections', **additional_selections_context)) for a_name, a_decoder in track_templates.get_decoders_dict().items()} ## gets around DisplayPipelineStage being passed for `curr_active_pipeline` sometimes

#         decoder_user_selected_epoch_times_dict = {a_name:np.atleast_2d(numpyify_array(user_annotations.get(a_selections_ctx, []))) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}
#         # loaded_selections_dict
        
#         ## Inputs: loaded_selections_dict, 
#         ## Find epochs that are present in any of the decoders:
#         total_num_user_selections: int = int(np.sum([np.size(v) for v in decoder_user_selected_epoch_times_dict.values()]))
#         if total_num_user_selections > 0:
#             concatenated_selected_epoch_times = NumpyHelpers.safe_concat([a_start_stop_arr for a_name, a_start_stop_arr in decoder_user_selected_epoch_times_dict.items() if np.size(a_start_stop_arr)>0], axis=0) # ` if np.size(a_start_stop_arr)>0` part was added to avoid empty lists causing `ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)`
#             any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
#         else:
#             print(f'WARNING: No user selections for this epoch')
#             any_good_selected_epoch_times: NDArray = np.atleast_2d([]) 
            
#         return decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times


#     @classmethod
#     @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-02 13:28', related_items=[])
#     def merge_decoded_epochs_result_dfs(cls, *dfs_list, should_drop_directional_columns:bool=True, start_t_idx_name='ripple_start_t'):
#         """ filter the ripple results scores by the user annotations. 
        
#         *dfs_list: a series of dataframes to join
#         should_drop_directional_columns:bool - if True, the direction (LR/RL) columns are dropped and only the _best_ columns are left.
#         """   
#         filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df = dfs_list # , additional_columns_merged_df

#         df: Optional[pd.DataFrame] = None

#         if filtered_ripple_simple_pf_pearson_merged_df is not None:
#             if df is None:
#                 df = filtered_ripple_simple_pf_pearson_merged_df.copy()
#                 assert np.all(np.isin(['P_LR', 'P_RL'], df.columns)), f"{list(df.columns)}" # ,'P_Long', 'P_Short'
#                 direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
#                 # track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)

#             direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
#             track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)
#             # Get only the best direction long/short values for each metric:
#             df['long_best_pf_peak_x_pearsonr'] = np.where(direction_max_indices, df['long_LR_pf_peak_x_pearsonr'], df['long_RL_pf_peak_x_pearsonr'])
#             df['short_best_pf_peak_x_pearsonr'] = np.where(direction_max_indices, df['short_LR_pf_peak_x_pearsonr'], df['short_RL_pf_peak_x_pearsonr'])
#             if should_drop_directional_columns:
#                 df = df.drop(columns=['P_LR', 'P_RL','best_decoder_index', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr']) # drop the directional column names

#         # Outputs: df

#         ## Add new weighted correlation results as new columns in existing filter_epochs df:
#         # Inputs: ripple_weighted_corr_merged_df, df from previous step

#         if ripple_weighted_corr_merged_df is not None:
#             if df is None:
#                 df: pd.DataFrame = ripple_weighted_corr_merged_df.copy()
#                 assert np.all(np.isin(['P_LR', 'P_RL'], df.columns)), f"{list(df.columns)}" # ,'P_Long', 'P_Short'
#                 direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
#                 # track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)


#             ## Perfrom a 1D matching of the epoch start times:
#             ## ORDER MATTERS:
#             # elements =  df[start_t_idx_name].to_numpy()
#             # test_elements = ripple_weighted_corr_merged_df[start_t_idx_name].to_numpy()
#             # valid_found_indicies = np.nonzero(np.isclose(test_elements[:, None], elements, atol=1e-3).any(axis=1))[0] #TODO 2024-03-14 09:34: - [ ] ERROR HERE?!?!
#             # hand_selected_ripple_weighted_corr_merged_df = ripple_weighted_corr_merged_df.iloc[valid_found_indicies].reset_index(drop=True) ## NOTE .iloc used here!
#             valid_found_indicies = find_data_indicies_from_epoch_times(ripple_weighted_corr_merged_df, epoch_times=df[start_t_idx_name].to_numpy(), t_column_names=[start_t_idx_name,], atol=1e-3)
#             hand_selected_ripple_weighted_corr_merged_df = ripple_weighted_corr_merged_df.loc[valid_found_indicies].reset_index(drop=True) ## Switched to .loc

#             ## Add the wcorr columns to `df`:
#             wcorr_column_names = ['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']
#             df[wcorr_column_names] = hand_selected_ripple_weighted_corr_merged_df[wcorr_column_names] # add the columns to the dataframe
#             df['long_best_wcorr'] = np.where(direction_max_indices, df['wcorr_long_LR'], df['wcorr_long_RL'])
#             df['short_best_wcorr'] = np.where(direction_max_indices, df['wcorr_short_LR'], df['wcorr_short_RL'])
#             if should_drop_directional_columns:
#                 df = df.drop(columns=['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']) # drop the directional column names
            
#             ## Add differences:
#             df['wcorr_abs_diff'] = df['long_best_wcorr'].abs() - df['short_best_wcorr'].abs()
#             df['pearsonr_abs_diff'] = df['long_best_pf_peak_x_pearsonr'].abs() - df['short_best_pf_peak_x_pearsonr'].abs()

#         return df


#     @classmethod
#     @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 22:58', related_items=[])
#     def filter_epochs_dfs_by_annotation_times(cls, curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size, *dfs_list):
#         """ filter the ripple results scores by the user annotations. 
        
#         *dfs_list: a series of dataframes to join

#         """   
#         # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

#         filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df = dfs_list

#         hand_selected_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
#         # hand_selected_ripple_simple_pf_pearson_merged_df

#         df: pd.DataFrame = cls.merge_decoded_epochs_result_dfs(hand_selected_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df, should_drop_directional_columns=True)

#         # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
#         session_name: str = curr_active_pipeline.session_name
#         t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
#         df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end, time_col='ripple_start_t')
#         df["time_bin_size"] = ripple_decoding_time_bin_size
#         df['is_user_annotated_epoch'] = True # if it's filtered here, it's true

#         return df



#     @classmethod
#     @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_valid_epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-04 00:00', related_items=[])
#     def try_add_is_epoch_boolean_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times: NDArray, new_column_name:str='is_valid_epoch', t_column_names=None, atol:float=0.01, not_found_action='skip_index', debug_print=False) -> bool:
#         """ tries to add a 'new_column_name' column to the dataframe. 
        
#         t_column_names = ['ripple_start_t',]
#         """
#         if (any_good_selected_epoch_times is None):
#             return False
#         any_good_selected_epoch_indicies = None
#         try:
#             # any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=t_column_names, atol=atol, not_found_action=not_found_action, debug_print=debug_print)    
#             any_good_selected_epoch_indicies = a_df.epochs.find_data_indicies_from_epoch_times(epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=t_column_names, atol=atol)
#         except BaseException as e:
#             print(f'ERROR: failed with error {e} while trying to add column "{new_column_name}". Out of options.')

#         if any_good_selected_epoch_indicies is None:
#             return False

#         # print(f'\t succeded at getting indicies! for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
#         a_df[new_column_name] = False
#         # a_df[new_column_name].iloc[any_good_selected_epoch_indicies] = True
#         a_df[new_column_name].loc[any_good_selected_epoch_indicies] = True
#         # a_df[new_column_name].loc[a_df.index.to_numpy()[any_good_selected_epoch_indicies]] = True # IndexError: index 392 is out of bounds for axis 0 with size 390
#         return True


#     @classmethod
#     @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_user_annotated_epoch'], input_requires=[], output_provides=[], uses=['cls.try_add_is_epoch_boolean_column'], used_by=[], creation_date='2024-03-02 13:17', related_items=[])
#     def try_add_is_user_annotated_epoch_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times, t_column_names=['ripple_start_t',]) -> bool:
#         """ tries to add a 'is_user_annotated_epoch' column to the dataframe. """
#         return cls.try_add_is_epoch_boolean_column(a_df=a_df, any_good_selected_epoch_times=any_good_selected_epoch_times, new_column_name='is_user_annotated_epoch', t_column_names=t_column_names, atol=0.01, not_found_action='skip_index', debug_print=False)
    

#     @classmethod
#     @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_valid_epoch'], input_requires=[], output_provides=[], uses=['cls.try_add_is_epoch_boolean_column'], used_by=[], creation_date='2024-03-02 13:17', related_items=[])
#     def try_add_is_valid_epoch_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times, t_column_names=['ripple_start_t',]) -> bool:
#         """ tries to add a 'is_valid_epoch' column to the dataframe. """
#         return cls.try_add_is_epoch_boolean_column(a_df=a_df, any_good_selected_epoch_times=any_good_selected_epoch_times, new_column_name='is_valid_epoch', t_column_names=t_column_names, atol=0.01, not_found_action='skip_index', debug_print=False)


#     @function_attributes(short_name=None, tags=['columns', 'epochs', 'IMPORTANT'], input_requires=[], output_provides=[], uses=['filter_and_update_epochs_and_spikes'], used_by=[], creation_date='2024-03-14 09:22', related_items=[])
#     def add_all_extra_epoch_columns(self, curr_active_pipeline, track_templates: TrackTemplates, required_min_percentage_of_active_cells: float = 0.333333,
#                                      debug_print=False, **additional_selections_context) -> None:
#         """ instead of filtering by the good/user-selected ripple epochs, it adds two columns: ['is_valid_epoch', 'is_user_annotated_epoch'] so they can be later identified and filtered to `self.decoder_ripple_filter_epochs_decoder_result_dict.filter_epochs`
#         Updates `self.decoder_ripple_filter_epochs_decoder_result_dict.filter_epochs` in-place 
#         """
#         ## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict

#         # 2024-03-04 - Filter out the epochs based on the criteria:
#         _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
#         session_name: str = curr_active_pipeline.session_name
#         t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

#         filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline=curr_active_pipeline, global_epoch_name=global_epoch_name, track_templates=track_templates, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, **additional_selections_context)
#         filtered_valid_epoch_times = filtered_epochs_df[['start', 'stop']].to_numpy()

#         ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
#         decoder_user_selected_epoch_times_dict, any_user_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)

#         a_result_dict = self.decoder_ripple_filter_epochs_decoder_result_dict ## Only operates on `self.decoder_ripple_filter_epochs_decoder_result_dict` (ripples)

#         for a_name, a_result in a_result_dict.items():
#             did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(ensure_dataframe(a_result.filter_epochs), any_good_selected_epoch_times=any_user_selected_epoch_times, t_column_names=None)
#             if debug_print:
#                 print(f'did_update_user_annotation_col["{a_name}"]: {did_update_user_annotation_col}')
#             did_update_is_valid = DecoderDecodedEpochsResult.try_add_is_valid_epoch_column(ensure_dataframe(a_result.filter_epochs), any_good_selected_epoch_times=filtered_valid_epoch_times, t_column_names=None)
#             # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
#             a_result.filter_epochs = DecoderDecodedEpochsResult.add_session_df_columns(ensure_dataframe(a_result.filter_epochs), session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end)            
#             if debug_print:
#                 print(f'did_update_is_valid["{a_name}"]: {did_update_is_valid}')
#         if debug_print:
#             print(f'\tdone.')


#     @classmethod
#     def add_score_best_dir_columns(cls, df: pd.DataFrame, col_name: str = 'pf_peak_x_pearsonr', should_drop_directional_columns:bool=False, is_col_name_suffix_mode: bool = False, 
#                                 include_LS_diff_col: bool=True, include_best_overall_score_col: bool=True) -> pd.DataFrame:
#         """ adds in a single "*_diff" and the 'long_best_*', 'short_best_*' columns
#         Generalized from `merge_decoded_epochs_result_dfs`

#         is_col_name_suffix_mode: bool - if True, the variable name (specified by `col_name`)

        
#         Usage:

#             from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

#             directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
#             directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=False)

#         """
#         added_col_names = []
#         direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
#         track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)
#         # Get only the best direction long/short values for each metric:
#         long_best_col_name: str = f'long_best_{col_name}'
#         short_best_col_name: str = f'short_best_{col_name}'
#         overall_best_col_name: str = f'overall_best_{col_name}'
        
        
#         if is_col_name_suffix_mode:
#             long_LR_string = f'long_LR_{col_name}'
#             long_RL_string = f'long_RL_{col_name}'
#             short_LR_string = f'short_LR_{col_name}'
#             short_RL_string = f'short_RL_{col_name}'
#         else:
#             long_LR_string = f'{col_name}_long_LR'
#             long_RL_string = f'{col_name}_long_RL'
#             short_LR_string = f'{col_name}_short_LR'
#             short_RL_string = f'{col_name}_short_RL'
        
#         df[long_best_col_name] = np.where(direction_max_indices, df[long_LR_string], df[long_RL_string])
#         df[short_best_col_name] = np.where(direction_max_indices, df[short_LR_string], df[short_RL_string])
#         added_col_names.append(long_best_col_name)
#         added_col_names.append(short_best_col_name)
        
#         if should_drop_directional_columns:
#             df = df.drop(columns=['P_LR', 'P_RL','best_decoder_index', long_LR_string, long_RL_string, short_LR_string, short_RL_string]) # drop the directional column names

#         ## Add differences:
#         if include_LS_diff_col:
#             LS_diff_col_name: str = f'{col_name}_diff'
#             df[LS_diff_col_name] = df[long_best_col_name].abs() - df[short_best_col_name].abs()
#             added_col_names.append(LS_diff_col_name)
        

#         ## adds an "overall_best_{col_name}" column that includes the maximum between long and short
#         if include_best_overall_score_col and (long_best_col_name in df) and (short_best_col_name in df):
#             df[overall_best_col_name] = df[[long_best_col_name, short_best_col_name]].max(axis=1, skipna=True)
#             added_col_names.append(overall_best_col_name)

#         return df, tuple(added_col_names)


#     @classmethod
#     def get_all_scores_column_names(cls) -> Tuple:
#         from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring
        
#         # Column Names _______________________________________________________________________________________________________ #
#         basic_df_column_names = ['start', 'stop', 'label', 'duration']
#         selection_col_names = ['is_user_annotated_epoch', 'is_valid_epoch']
#         session_identity_col_names = ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']
        
#         # Score Columns (one value for each decoder) _________________________________________________________________________ #
#         decoder_bayes_prob_col_names = ['P_decoder']

#         radon_transform_col_names = ['score', 'velocity', 'intercept', 'speed']
#         weighted_corr_col_names = ['wcorr']
#         pearson_col_names = ['pearsonr']

#         heuristic_score_col_names = HeuristicReplayScoring.get_all_score_computation_col_names()
#         print(f'heuristic_score_col_names: {heuristic_score_col_names}')

#         ## All included columns:
#         all_df_shared_column_names: List[str] = basic_df_column_names + selection_col_names + session_identity_col_names # these are not replicated for each decoder, they're the same for the epoch
#         all_df_score_column_names: List[str] = decoder_bayes_prob_col_names + radon_transform_col_names + weighted_corr_col_names + pearson_col_names + heuristic_score_col_names 
#         all_df_column_names: List[str] = all_df_shared_column_names + all_df_score_column_names ## All included columns, includes the score columns which will not be replicated

#         ## Add in the 'wcorr' metrics:
#         merged_conditional_prob_column_names = ['P_LR', 'P_RL', 'P_Long', 'P_Short']
#         merged_wcorr_column_names = ['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']

#         return (all_df_shared_column_names, all_df_score_column_names, all_df_column_names,
#                     merged_conditional_prob_column_names, merged_wcorr_column_names, heuristic_score_col_names)

#     @function_attributes(short_name=None, tags=['merged', 'all_scores', 'df', 'epochs'], input_requires=[], output_provides=[], uses=['.decoder_ripple_filter_epochs_decoder_result_dict', '_build_merged_score_metric_df'], used_by=[], creation_date='2024-03-14 19:10', related_items=[])
#     def build_complete_all_scores_merged_df(self, debug_print=False) -> pd.DataFrame:
#         """ Builds a single merged dataframe from the four separate .filter_epochs dataframes from the result for each decoder, merging them into a single dataframe with ['_long_LR','_long_RL','_short_LR','_short_RL'] suffixes for the combined columns.
#         2024-03-14 19:04 

#         Usage:
#             extracted_merged_scores_df = build_complete_all_scores_merged_df(directional_decoders_epochs_decode_result)
#             extracted_merged_scores_df


#         #TODO 2024-07-15 18:32: - [ ] Ending up with multiple 'P_LR' columns in the dataframe! Not sure how this can happen.


#         """
#         from neuropy.core.epoch import ensure_dataframe
#         from neuropy.utils.indexing_helpers import flatten, NumpyHelpers, PandasHelpers
#         from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _build_merged_score_metric_df

#         # # Column Names _______________________________________________________________________________________________________ #

#         # ## All included columns:
#         if debug_print:
#             print(f'build_complete_all_scores_merged_df(...):')
#         all_df_shared_column_names, all_df_score_column_names, all_df_column_names, merged_conditional_prob_column_names, merged_wcorr_column_names, heuristic_score_col_names = self.get_all_scores_column_names()

#         ## Extract the concrete dataframes from the results:
#         extracted_filter_epochs_dfs_dict = {k:ensure_dataframe(a_result.filter_epochs) for k, a_result in self.decoder_ripple_filter_epochs_decoder_result_dict.items()} # #TODO 2024-12-17 08:12: - [ ] Debugging: P_Decoder columns already have NaNs in the same place as they do in the next step here!
#         ## Merge the dict of four dataframes, one for each decoder, with column names like ['wcorr', 'travel', 'speed'] to a single merged df with suffixed of the dict keys like ['wcorr_long_LR', 'wcorr_long_RL',  ...., 'travel_long_LR', 'travel_long_RL', 'travel_short_LR', 'travel_short_RL', ...]
#         extracted_merged_scores_df: pd.DataFrame = _build_merged_score_metric_df(extracted_filter_epochs_dfs_dict, columns=all_df_score_column_names, best_decoder_index_column_name=None) ## already has NaN values for the decoder probabilities here :[
#         # extracted_merged_scores_df

#         _ref_df = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0]) # first dataframe is the same as the others, determine which columns are available
#         included_all_df_shared_column_names = [k for k in all_df_shared_column_names if k in _ref_df.columns] # only the included columns

#         # `common_shared_portion_df` the columns of the dataframe that is the same for all four decoders
#         # common_shared_portion_df: pd.DataFrame = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0][all_df_shared_column_names]) # copy it from the first dataframe
#         common_shared_portion_df: pd.DataFrame = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0][included_all_df_shared_column_names]) # copy it from the first dataframe
#         base_shape = np.shape(common_shared_portion_df)

#         included_merge_dfs_list = [common_shared_portion_df]

#         #TODO 2024-07-12 07:06: - [ ] `self.ripple_weighted_corr_merged_df` is the problem it seems, it's of different size (more epochs) than all of the other dataframes

#         ##Gotta get those ['P_LR', 'P_RL'] columns to determine best directions
#         conditional_prob_df = deepcopy(self.ripple_weighted_corr_merged_df[merged_conditional_prob_column_names]) ## just use the columns from this
#         conditional_prob_df_shape = np.shape(conditional_prob_df)
#         if (base_shape[0] != conditional_prob_df_shape[0]):
#             print(f'\tbuild_complete_all_scores_merged_df(...): warning: all dfs should have same number of rows, but conditional_prob_df_shape: {conditional_prob_df_shape} != base_shape: {base_shape}. Skipping adding `conditional_prob_df`.')
#         else:
#             ## add it 
#             included_merge_dfs_list.append(conditional_prob_df)

        
#         ## Re-derive the correct conditional probs:
#         # ['P_LR', 'P_RL']
#         # ['P_Long', 'P_Short']

#         P_decoder_column_names = ['P_decoder_long_LR','P_decoder_long_RL','P_decoder_short_LR','P_decoder_short_RL']
#         P_decoder_marginals_column_names = ['P_LR', 'P_RL', 'P_Long', 'P_Short']

#         # if np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_column_names]):
#         if np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_marginals_column_names]):
#             # needs Marginalized Probability columns: ['P_LR', 'P_RL'], ['P_Long', 'P_Short']
#             if debug_print:
#                 print(f'\tneeds Marginalized Probability columns. adding.')
#             # assert np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_column_names]), f"missing marginals and cannot recompute them because we're also missing the raw probabilities. extracted_merged_scores_df.columns: {list(extracted_merged_scores_df.columns)}"
#             ## They remain normalized because they all already sum to one.
#             extracted_merged_scores_df['P_Long'] = extracted_merged_scores_df['P_decoder_long_LR'] + extracted_merged_scores_df['P_decoder_long_RL']
#             extracted_merged_scores_df['P_Short'] = extracted_merged_scores_df['P_decoder_short_LR'] + extracted_merged_scores_df['P_decoder_short_RL']

#             extracted_merged_scores_df['P_LR'] = extracted_merged_scores_df['P_decoder_long_LR'] + extracted_merged_scores_df['P_decoder_short_LR']
#             extracted_merged_scores_df['P_RL'] = extracted_merged_scores_df['P_decoder_long_RL'] + extracted_merged_scores_df['P_decoder_short_RL']


#         extracted_merged_scores_df_shape = np.shape(extracted_merged_scores_df)
#         if (base_shape[0] != extracted_merged_scores_df_shape[0]):
#             print(f'\tbuild_complete_all_scores_merged_df(...): warning: all dfs should have same number of rows, but extracted_merged_scores_df_shape: {extracted_merged_scores_df_shape} != base_shape: {base_shape}. Skipping adding `extracted_merged_scores_df`.')
#         else:
#             ## add it
#             included_merge_dfs_list.append(extracted_merged_scores_df)

#         # # Weighted correlations:

#         # Build the final merged dataframe with the score columns for each of the four decoders but only one copy of the common columns.
#         extracted_merged_scores_df: pd.DataFrame = pd.concat(included_merge_dfs_list, axis='columns') # (common_shared_portion_df, conditional_prob_df, extracted_merged_scores_df) ### I THINK NaNs come in here
#         # extracted_merged_scores_df: pd.DataFrame = pd.concat((common_shared_portion_df, conditional_prob_df, extracted_merged_scores_df), axis='columns')
#         extracted_merged_scores_df['ripple_start_t'] = extracted_merged_scores_df['start']

#         if np.any([(a_col not in extracted_merged_scores_df) for a_col in merged_wcorr_column_names]):
#             # needs wcorr columns
#             if debug_print:
#                 print(f'\tbuild_complete_all_scores_merged_df(...): needs wcorr columns. adding.')
#             wcorr_columns_df = deepcopy(self.ripple_weighted_corr_merged_df[merged_wcorr_column_names]) ## just use the columns from this
#             assert np.shape(wcorr_columns_df)[0] == np.shape(extracted_merged_scores_df)[0], f"should have same number of columns"
#             extracted_merged_scores_df: pd.DataFrame = pd.concat((extracted_merged_scores_df, wcorr_columns_df), axis='columns')

#         ## Add in the wcorr and pearsonr columns:
#         # self.ripple_simple_pf_pearson_merged_df ## ?? where is it getting "pearsonr_long_LR"?

#         ## add in the "_diff" columns and the 'best_dir_*' columns
#         added_column_names = []
#         # for a_score_col in heuristic_score_col_names:
#         #     extracted_merged_scores_df, curr_added_column_name_tuple = self.add_score_best_dir_columns(extracted_merged_scores_df, col_name=a_score_col, should_drop_directional_columns=False, is_col_name_suffix_mode=False)
#         #     added_column_names.extend(curr_added_column_name_tuple)
#         #     # (long_best_col_name, short_best_col_name, LS_diff_col_name)

#         try:
#             for a_score_col in all_df_score_column_names:
#                 extracted_merged_scores_df, curr_added_column_name_tuple = self.add_score_best_dir_columns(extracted_merged_scores_df, col_name=a_score_col, should_drop_directional_columns=False, is_col_name_suffix_mode=False, include_LS_diff_col=True, include_best_overall_score_col=True)
#                 added_column_names.extend(curr_added_column_name_tuple)
#         except Exception as err:
#             print(f'\tbuild_complete_all_scores_merged_df(...): Encountered ERROR: {err} while trying to add "a_score_col": {a_score_col}, but trying to continue, so close!')


#         extracted_merged_scores_df = extracted_merged_scores_df.rename(columns=dict(zip(['P_decoder_long_LR','P_decoder_long_RL','P_decoder_short_LR','P_decoder_short_RL'], ['P_Long_LR','P_Long_RL','P_Short_LR','P_Short_RL'])), inplace=False)
#         if 'time_bin_size' not in extracted_merged_scores_df.columns:
#             ## add the column
#             extracted_merged_scores_df['time_bin_size'] = self.ripple_decoding_time_bin_size

#         extracted_merged_scores_df = PandasHelpers.dropping_duplicated_df_columns(df=extracted_merged_scores_df)
#         return extracted_merged_scores_df



#     @classmethod
#     def _perform_export_dfs_dict_to_csvs(cls, extracted_dfs_dict: Dict, parent_output_path: Path, active_context: IdentifyingContext, session_name: str, tbin_values_dict: Dict,
#                                         t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None,
#                                         user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None):
#         """ Classmethod: export as separate .csv files. 
#         active_context = curr_active_pipeline.get_session_context()
#         curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
#         CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
#         print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

#         active_context = curr_active_pipeline.get_session_context()
#         session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
#         session_name: str = curr_active_pipeline.session_name
#         earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = curr_active_pipeline.find_LongShortDelta_times()
#         histogram_bins = 25
#         # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
#         delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
#         decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline)
#         any_good_selected_epoch_indicies = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
#         df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
#         df

#         tbin_values_dict={'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}

#         #TODO 2024-11-01 07:53: - [ ] Need to pass the correct (full) context, including the qclu/fr_Hz filter values and the replay name. 
#         #TODO 2024-11-01 07:54: - [X] Need to add a proper timebin column to the df instead of including it in the filename (if possible)
#             - does already add a 'time_bin_size' column, and the suffix is probably so it doesn't get overwritten by different time_bin_sizes, might need to put them together post-hoc
            
#         '2024-11-01_1250PM-kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_weighted_corr_merged_df)_tbin-1.5.csv'
            
#         """

#         assert parent_output_path.exists(), f"'{parent_output_path}' does not exist!"
#         output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
#         # Export CSVs:
#         def export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
#             """ captures `active_context`, 'output_date_str'
#             """
#             # parent_output_path: Path = Path('output').resolve()
#             # active_context = curr_active_pipeline.get_session_context()
#             session_identifier_str: str = active_context.get_description() # 'kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
#             # session_identifier_str: str = active_context.get_description(subset_excludelist=['custom_suffix']) # no this is just the session
#             assert output_date_str is not None
#             out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-11-15_0200PM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays_qclu_[1, 2, 4, 6, 7, 9]_frateThresh_5.0-(ripple_WCorrShuffle_df)_tbin-0.025'
#             out_filename = f"{out_basename}.csv"
#             out_path = parent_output_path.joinpath(out_filename).resolve()
#             export_df.to_csv(out_path)
#             return out_path 

#         if custom_export_df_to_csv_fn is None:
#             # use the default
#             custom_export_df_to_csv_fn = export_df_to_csv


#         # active_context.custom_suffix = '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0' # '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
        
#         #TODO 2024-03-02 12:12: - [ ] Could add weighted correlation if there is a dataframe for that and it's computed:
#         # tbin_values_dict = {'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}
#         time_col_name_dict = {'laps': 'lap_start_t', 'ripple': 'ripple_start_t'} ## default should be 't_bin_center'
    
#         ## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict
#         export_files_dict = {}
        
#         for a_df_name, a_df in extracted_dfs_dict.items():
#             an_epochs_source_name: str = a_df_name.split(sep='_', maxsplit=1)[0] # get the first part of the variable names that indicates whether it's for "laps" or "ripple"

#             a_tbin_size: float = float(tbin_values_dict[an_epochs_source_name])
#             a_time_col_name: str = time_col_name_dict.get(an_epochs_source_name, 't_bin_center')
#             ## Add t_bin column method
#             a_df = cls.add_session_df_columns(a_df, session_name=session_name, time_bin_size=a_tbin_size, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=a_time_col_name)
#             a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
#             a_data_identifier_str: str = f'({a_df_name})_tbin-{a_tbin_size_str}' ## build the identifier '(laps_weighted_corr_merged_df)_tbin-1.5'
            
#             # add in custom columns
#             #TODO 2024-03-14 06:48: - [ ] I could use my newly implemented `directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=True)` function, but since this looks at decoder-specific info it's better just to duplicate implementation and do it again here.
#             # ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
#             # ripple_marginals_df['ripple_start_t'] = ripple_epochs_df['start'].to_numpy()
#             if (user_annotation_selections is not None):
#                 any_good_selected_epoch_times = user_annotation_selections.get(an_epochs_source_name, None) # like ripple
#                 if any_good_selected_epoch_times is not None:
#                     num_valid_epoch_times: int = len(any_good_selected_epoch_times)
#                     print(f'num_user_selected_times: {num_valid_epoch_times}')
#                     any_good_selected_epoch_indicies = None
#                     print(f'adding user annotation column!')

#                     if any_good_selected_epoch_indicies is None:
#                         try:
#                             any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
#                         except AttributeError as e:
#                             print(f'ERROR: failed method 2 for {a_df_name}. Out of options.')        
#                         except Exception as e:
#                             print(f'ERROR: failed for {a_df_name}. Out of options.')
                        
#                     if any_good_selected_epoch_indicies is not None:
#                         print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} user selections) for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
#                         a_df['is_user_annotated_epoch'] = False
#                         a_df['is_user_annotated_epoch'].iloc[any_good_selected_epoch_indicies] = True
#                     else:
#                         print(f'\t failed all methods for annotations')

#             # adds in column 'is_valid_epoch'
#             if (valid_epochs_selections is not None):
#                 # 2024-03-04 - Filter out the epochs based on the criteria:
#                 any_good_selected_epoch_times = valid_epochs_selections.get(an_epochs_source_name, None) # like ripple
#                 if any_good_selected_epoch_times is not None:
#                     num_valid_epoch_times: int = len(any_good_selected_epoch_times)
#                     print(f'num_valid_epoch_times: {num_valid_epoch_times}')
#                     any_good_selected_epoch_indicies = None
#                     print(f'adding valid filtered epochs column!')

#                     if any_good_selected_epoch_indicies is None:
#                         try:
#                             any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
#                         except AttributeError as e:
#                             print(f'ERROR: failed method 2 for {a_df_name}. Out of options.')        
#                         except Exception as e:
#                             print(f'ERROR: failed for {a_df_name}. Out of options.')
                        
#                     if any_good_selected_epoch_indicies is not None:
#                         print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} valid filter epoch times) for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
#                         a_df['is_valid_epoch'] = False

#                         try:
#                             a_df['is_valid_epoch'].iloc[any_good_selected_epoch_indicies] = True
#                             # a_df['is_valid_epoch'].loc[any_good_selected_epoch_indicies] = True

#                         except Exception as e:
#                             print(f'WARNING: trying to get whether the epochs are valid FAILED probably, 2024-06-28 custom computed epochs thing: {e}, just setting all to True')
#                             a_df['is_valid_epoch'] = True
#                     else:
#                         print(f'\t failed all methods for selection filter')

#             export_files_dict[a_df_name] = custom_export_df_to_csv_fn(a_df, data_identifier_str=a_data_identifier_str, parent_output_path=parent_output_path) # this is exporting corr '(ripple_WCorrShuffle_df)_tbin-0.025'
#         # end for a_df_name, a_df
        
#         return export_files_dict
    


#     def perform_export_dfs_dict_to_csvs(self, extracted_dfs_dict: Dict, parent_output_path: Path, active_context, session_name: str, curr_session_t_delta: Optional[float], user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None):
#         """ export as separate .csv files. 
#         active_context = curr_active_pipeline.get_session_context()
#         curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
#         CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
#         print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

#         active_context = curr_active_pipeline.get_session_context()
#         session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
#         session_name: str = curr_active_pipeline.session_name
#         earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = curr_active_pipeline.find_LongShortDelta_times()
#         histogram_bins = 25
#         # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
#         delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
#         decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline)
#         any_good_selected_epoch_indicies = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
#         df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
#         df

#         """
#         return self._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=extracted_dfs_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, tbin_values_dict={'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size},
#                                                      curr_session_t_delta=curr_session_t_delta, user_annotation_selections=user_annotation_selections, valid_epochs_selections=valid_epochs_selections, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)



#     @function_attributes(short_name=None, tags=['export', 'CSV', 'main'], input_requires=[], output_provides=['ripple_all_scores_merged_df.csv'], uses=['self.perform_export_dfs_dict_to_csvs', 'self.build_complete_all_scores_merged_df'], used_by=['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'], creation_date='2024-03-15 10:13', related_items=[])
#     def export_csvs(self, parent_output_path: Path, active_context: IdentifyingContext, session_name: str, curr_session_t_delta: Optional[float], user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None, export_df_variable_names=None, should_export_complete_all_scores_df:bool=True):
#         """ export as separate .csv files. 

#         from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

#         active_context = curr_active_pipeline.get_session_context()
#         curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
#         CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
#         print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

#         active_context = curr_active_pipeline.get_session_context()
#         session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
#         session_name: str = curr_active_pipeline.session_name
#         earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = curr_active_pipeline.find_LongShortDelta_times()
#         histogram_bins = 25
#         # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
#         delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
#         decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline)
#         any_good_selected_epoch_indicies = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
#         df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
#         df


            
#         """
#         export_files_dict = {}
#         _df_variables_names = ['laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df']
#         if export_df_variable_names is None:
#             # export all by default
#             export_df_variable_names = _df_variables_names
            
#         extracted_dfs_dict = {a_df_name:getattr(self, a_df_name) for a_df_name in export_df_variable_names}
#         if len(extracted_dfs_dict) > 0:
#             export_files_dict = export_files_dict | self.perform_export_dfs_dict_to_csvs(extracted_dfs_dict=extracted_dfs_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=user_annotation_selections, valid_epochs_selections=valid_epochs_selections, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)

#         ## try to export the merged all_scores dataframe
#         if should_export_complete_all_scores_df:
#             extracted_merged_scores_df: pd.DataFrame = self.build_complete_all_scores_merged_df()
#             if 'time_bin_size' not in extracted_merged_scores_df.columns:
#                 ## add the column
#                 print(f'WARN: adding the time_bin_size columns: {self.ripple_decoding_time_bin_size}')
#                 extracted_merged_scores_df['time_bin_size'] = self.ripple_decoding_time_bin_size

#             export_df_dict = {'ripple_all_scores_merged_df': extracted_merged_scores_df}
#             export_files_dict = export_files_dict | self.perform_export_dfs_dict_to_csvs(extracted_dfs_dict=export_df_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)

#         return export_files_dict

    




# # # Define the namedtuple
# LongShortDecodersTuple = namedtuple('LongShortDecodersTuple', ['long', 'short'])

# @metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 13:19', related_items=[])
# class GenericTrackTemplatesMixin:
    
#     def get_decoders(self) -> Tuple[BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return LongShortDecodersTuple(list(self.decoders.values()))

#     @classmethod
#     def get_decoder_names(cls) -> Tuple[str]:
#         raise NotImplementedError(f'because this is a class method we must override!')
#         # return tuple(list(self.decoders.keys())) # ('long_LR','long_RL','short_LR','short_RL')
    
#     def get_decoders_dict(self) -> Dict[types.DecoderName, BasePositionDecoder]:
#         return self.decoders
    


# @define(slots=False, repr=False, eq=False)
# class BaseTrackTemplates(HDFMixin, AttrsBasedClassHelperMixin, GenericTrackTemplatesMixin):
#     """ Holds the four directional templates for direction placefield analysis.
#     from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

#     History:
#         Based off of `ShuffleHelper` on 2023-10-27
#         TODO: eliminate functional overlap with `ShuffleHelper`
#         TODO: should be moved into `DirectionalPlacefieldGlobalComputation` instead of RankOrder

#     """
#     decoders: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(repr=False)


#     @property
#     def decoder_neuron_IDs_list(self) -> List[NDArray]:
#         """ a list of the neuron_IDs for each decoder (independently) """
#         return [a_decoder.pf.ratemap.neuron_ids for a_decoder in (list(self.decoders.values()))]
    
#     @property
#     def any_decoder_neuron_IDs(self) -> NDArray:
#         """ a list of the neuron_IDs for each decoder (independently) """
#         return np.sort(union_of_arrays(*self.decoder_neuron_IDs_list)) # neuron_IDs as they appear in any list

#     @property
#     def decoder_peak_location_list(self) -> List[NDArray]:
#         """ a list of the peak_tuning_curve_center_of_masses for each decoder (independently) """
#         return [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses for a_decoder in (list(self.decoders.values()))]
    
#     @property
#     def decoder_peak_rank_list_dict(self) -> Dict[str, NDArray]:
#         """ a dict (one for each decoder) of the rank_lists for each decoder (independently) """
#         return {a_decoder_name:scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
#     @property
#     def decoder_aclu_peak_rank_dict_dict(self) -> Dict[str, Dict[types.aclu_index, float]]:
#         """ a Dict (one for each decoder) of aclu-to-rank maps for each decoder (independently) """
#         return {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method))) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
#     @property
#     def decoder_normalized_tuning_curves_dict_dict(self) -> Dict[str, Dict[types.aclu_index, NDArray]]:
#         """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
#         return {a_name:a_decoder.pf.normalized_tuning_curves_dict for a_name, a_decoder in self.get_decoders_dict().items()}
            

#     @property
#     def decoder_stability_dict_dict(self): # -> Dict[str, Dict[types.aclu_index, NDArray]]:
#         # """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
#         return {a_name:a_decoder.pf.ratemap.spatial_sparcity for a_name, a_decoder in self.get_decoders_dict().items()}
    

#     def get_decoders_tuning_curve_modes(self, peak_mode='peaks', **find_peaks_kwargs) -> Tuple[Dict[decoder_name_str, Dict[types.aclu_index, NDArray]], Dict[decoder_name_str, Dict[types.aclu_index, int]], Dict[decoder_name_str, pd.DataFrame]]:
#         """ 2023-12-19 - Uses `scipy.signal.find_peaks to find the number of peaks or ("modes") for each of the cells in the ratemap. 
#         Can detect bimodal (or multi-modal) placefields.
        
#         Depends on:
#             self.tuning_curves
        
#         Returns:
#             aclu_n_peaks_dict: Dict[int, int] - A mapping between aclu:n_tuning_curve_modes
#         Usage:    
#             decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict = track_templates.get_decoders_tuning_curve_modes()

#         """
#         decoder_peaks_results_tuples_dict = {a_decoder_name:a_decoder.pf.ratemap.compute_tuning_curve_modes(peak_mode=peak_mode, **find_peaks_kwargs) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
#         # each tuple contains: peaks_dict, aclu_n_peaks_dict, peaks_results_df, so unwrap below
        
#         decoder_peaks_dict_dict = {k:v[0] for k,v in decoder_peaks_results_tuples_dict.items()}
#         decoder_aclu_n_peaks_dict_dict = {k:v[1] for k,v in decoder_peaks_results_tuples_dict.items()}
#         decoder_peaks_results_df_dict = {k:v[2] for k,v in decoder_peaks_results_tuples_dict.items()}

#         # return peaks_dict, aclu_n_peaks_dict, unimodal_peaks_dict, peaks_results_dict
#         return decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict
    

#     @function_attributes(short_name=None, tags=['WORKING', 'peak', 'multi-peak', 'decoder', 'pfs'], input_requires=[], output_provides=[], uses=['get_tuning_curve_peak_positions'], used_by=['add_directional_pf_maximum_peaks'], creation_date='2024-05-21 19:00', related_items=[])
#     def get_directional_pf_maximum_peaks_dfs(self, drop_aclu_if_missing_long_or_short: bool = True) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
#         """ The only version that only gets the maximum peaks appropriate for each decoder.

#         # 2024-05-21 - Replaces `.get_decoders_aclu_peak_location_df(...)` for properly getting peak locations. Is correct (which is why the old result was replaced) but has a potential drawback of not currently accepting `, **find_peaks_kwargs`. I only see `width=None` ever passed in like this though.

#         # 2024-04-09 00:36: - [X] Could be refactored into TrackTemplates

#         #TODO 2024-05-21 22:53: - [ ] Noticed that short track always has all-non-NaN peaks (has a value for each peak) and long track is missing values. This doesn't make sense because many of the peaks indicated for short occur only on the long-track, which makes no sense.

#         Usage:

#             (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df = track_templates.get_directional_pf_maximum_peaks_dfs(drop_aclu_if_missing_long_or_short=False)

#             AnyDir_decoder_aclu_MAX_peak_maps_df
#             LR_only_decoder_aclu_MAX_peak_maps_df
#             RL_only_decoder_aclu_MAX_peak_maps_df


#         """
#         # drop_aclu_if_missing_long_or_short: bool = True ## default=True; Drop entire row if either long/short is missing a value
#         # drop_aclu_if_missing_long_or_short: bool = False
#         from neuropy.utils.indexing_helpers import intersection_of_arrays, union_of_arrays
#         from neuropy.utils.indexing_helpers import unwrap_single_item


#         ## Split into LR/RL groups to get proper peak differences:
#         # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
#         LR_decoder_names = self.get_LR_decoder_names() # ['long_LR', 'short_LR']
#         RL_decoder_names = self.get_RL_decoder_names() # ['long_RL', 'short_RL']

#         ## Only the maximums (height=1 items), guaranteed to be a single (or None) location:
#         decoder_aclu_MAX_peak_maps_dict: Dict[types.DecoderName, Dict[types.aclu_index, Optional[float]]] = {types.DecoderName(a_name):{k:unwrap_single_item(v) for k, v in deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)))).items()} for a_name, a_decoder in self.get_decoders_dict().items()}
#         #TODO 2024-05-21 22:59: - [ ] NEed to ensure that `a_decoder.neuron_IDs` and `a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)` are returned in the same order for sure
#             # it should because it's dependent only on `pdf_normalized_tuning_curves`, which is in the neuron_IDs order. The only issue could be if the subpeaks sorting issue happens

#         # decoder_aclu_MAX_peak_maps_dict
#         AnyDir_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in (LR_decoder_names + RL_decoder_names)}) # either direction decoder

#         ## Splits by direction:
#         LR_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in LR_decoder_names})
#         RL_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in RL_decoder_names})

#         ## Drop entire row if either long/short is missing a value:
#         if drop_aclu_if_missing_long_or_short:
#             LR_only_decoder_aclu_MAX_peak_maps_df = LR_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')
#             RL_only_decoder_aclu_MAX_peak_maps_df = RL_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')

#             AnyDir_decoder_aclu_MAX_peak_maps_df = AnyDir_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any') # might need to think this through a little better. Currently only using the `AnyDir_*` result with `drop_aclu_if_missing_long_or_short == False`

#         ## Compute the difference between the Long/Short peaks: I don't follow this:
#         LR_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = LR_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]
#         RL_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = RL_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]

#         AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_LR'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(LR_decoder_names)].diff(axis='columns').to_numpy()[:, -1]
#         AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_RL'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(RL_decoder_names)].diff(axis='columns').to_numpy()[:, -1]

#         # OUTPUTS: LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df
#         return (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df


#     @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
#     def get_decoder_aclu_peak_maps(self, peak_mode='CoM') -> DirectionalDecodersTuple:
#         """ returns a tuple of dicts, each containing a mapping between aclu:peak_pf_x for a given decoder. 
         
#         # Naievely:
#         long_LR_aclu_peak_map = deepcopy(dict(zip(self.long_LR_decoder.neuron_IDs, self.long_LR_decoder.peak_locations)))
#         long_RL_aclu_peak_map = deepcopy(dict(zip(self.long_RL_decoder.neuron_IDs, self.long_RL_decoder.peak_locations)))
#         short_LR_aclu_peak_map = deepcopy(dict(zip(self.short_LR_decoder.neuron_IDs, self.short_LR_decoder.peak_locations)))
#         short_RL_aclu_peak_map = deepcopy(dict(zip(self.short_RL_decoder.neuron_IDs, self.short_RL_decoder.peak_locations)))
        
#         """
#         assert peak_mode in ['peaks', 'CoM']
#         if peak_mode == 'peaks':
#             # return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode=peak_mode)))) for a_decoder in (list(self.decoders.values()))])
#             return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_locations))) for a_decoder in (list(self.decoders.values()))]) ## #TODO 2024-02-16 04:27: - [ ] This uses .peak_locations which are the positions corresponding to the peak position bin (but not continuously the peak from the curve).
#         elif peak_mode == 'CoM':
#             return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_tuning_curve_center_of_masses))) for a_decoder in (list(self.decoders.values()))])
#         else:
#             raise NotImplementedError(f"peak_mode: '{peak_mode}' is not supported.")
    

#     @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
#     def get_decoder_aclu_peak_map_dict(self, peak_mode='CoM') -> Dict[decoder_name_str, Dict]:
#         return dict(zip(self.get_decoder_names(), self.get_decoder_aclu_peak_maps(peak_mode=peak_mode)))


#     def __repr__(self):
#         """ 
#         TrackTemplates(long_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             long_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             short_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             short_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             shared_LR_aclus_only_neuron_IDs: numpy.ndarray,
#             is_good_LR_aclus: NoneType,
#             shared_RL_aclus_only_neuron_IDs: numpy.ndarray,
#             is_good_RL_aclus: NoneType,
#             decoder_LR_pf_peak_ranks_list: list,
#             decoder_RL_pf_peak_ranks_list: list
#         )
#         """
#         content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
#         return f"{type(self).__name__}({content}\n)"



#     # ==================================================================================================================== #
#     # GenericTrackTemplatesMixin                                                                                           #
#     # ==================================================================================================================== #
#     def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return DirectionalDecodersTuple(list(self.decoders.values()))

#     @classmethod
#     def get_decoder_names(cls) -> Tuple[str, str, str, str]:
#         return ('long_LR','long_RL','short_LR','short_RL')
    
    
#     def get_decoders_dict(self) -> Dict[types.DecoderName, BasePositionDecoder]:
#         return {'long_LR': self.long_LR_decoder,
#             'long_RL': self.long_RL_decoder,
#             'short_LR': self.short_LR_decoder,
#             'short_RL': self.short_RL_decoder,
#         }
    


#     # # Init/ClassMethods __________________________________________________________________________________________________ #

#     # @classmethod
#     # def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], rank_method:str='average') -> "TrackTemplates":
#     #     """ 2023-10-31 - Extract from pairs

#     #     """
#     #     long_LR_decoder, short_LR_decoder = LR_decoder_pair
#     #     long_RL_decoder, short_RL_decoder = RL_decoder_pair

#     #     shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
#     #     assert np.all(short_LR_decoder.neuron_IDs == shared_LR_aclus_only_neuron_IDs), f"{short_LR_decoder.neuron_IDs} != {shared_LR_aclus_only_neuron_IDs}"

#     #     shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)
#     #     assert np.all(short_RL_decoder.neuron_IDs == shared_RL_aclus_only_neuron_IDs), f"{short_RL_decoder.neuron_IDs} != {shared_RL_aclus_only_neuron_IDs}"

#     #     # is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
#     #     # shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

#     #     ## 2023-10-11 - Get the long/short peak locations
#     #     # decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
#     #     ## Compute the ranks:
#     #     # decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]

#     #     #TODO 2023-11-21 13:06: - [ ] Note these are in order of the original entries, and do not reflect any sorts or ordering changes.

#     #     return cls(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, None, shared_RL_aclus_only_neuron_IDs, None,
#     #                 decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_LR_decoder, short_LR_decoder)],
#     #                 decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_RL_decoder, short_RL_decoder)],
#     #                 rank_method=rank_method)

#     # @classmethod
#     # def determine_decoder_aclus_filtered_by_frate_and_qclu(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(decoders_dict=track_templates.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

#     #     """
#     #     decoder_names = cls.get_decoder_names() # ('long_LR', 'long_RL', 'short_LR', 'short_RL')
#     #     modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=decoders_dict, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
#     #     # individual_decoder_filtered_aclus_list = list(modified_neuron_ids_dict.values())
#     #     individual_decoder_filtered_aclus_list = [modified_neuron_ids_dict[a_decoder_name] for a_decoder_name in decoder_names]
#     #     assert len(individual_decoder_filtered_aclus_list) == 4, f"len(individual_decoder_filtered_aclus_list): {len(individual_decoder_filtered_aclus_list)} but expected 4!"
#     #     original_decoder_list = [deepcopy(decoders_dict[a_decoder_name]) for a_decoder_name in decoder_names]
#     #     ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
#     #     filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
#     #     filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
#     #     # build the final shared aclus:
#     #     filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
#     #     filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip(original_decoder_list, filtered_direction_shared_aclus_list)]
#     #     return filtered_decoder_list, filtered_direction_shared_aclus_list
    
#     # @classmethod
#     # def _perform_determine_decoder_aclus_filtered_by_qclu_and_frate(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=track_templates.get_decoders_dict())
            
#     #         decoders_dict=self.get_decoders_dict()
            
#     #     """
#     #     # original_neuron_ids_list = [a_decoder.pf.ratemap.neuron_ids for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
#     #     original_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder.pf.ratemap.neuron_ids) for a_decoder_name, a_decoder in decoders_dict.items()}
#     #     if (minimum_inclusion_fr_Hz is not None) and (minimum_inclusion_fr_Hz > 0.0):
#     #         modified_neuron_ids_dict = {a_decoder_name:np.array(a_decoder.pf.ratemap.neuron_ids)[a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz] for a_decoder_name, a_decoder in decoders_dict.items()}
#     #     else:            
#     #         modified_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder_neuron_ids) for a_decoder_name, a_decoder_neuron_ids in original_neuron_ids_dict.items()}
        
#     #     if included_qclu_values is not None:
#     #         # filter by included_qclu_values
#     #         for a_decoder_name, a_decoder in decoders_dict.items():
#     #             # a_decoder.pf.spikes_df
#     #             neuron_identities: pd.DataFrame = deepcopy(a_decoder.pf.filtered_spikes_df).spikes.extract_unique_neuron_identities()
#     #             # filtered_neuron_identities: pd.DataFrame = neuron_identities[neuron_identities.neuron_type == NeuronType.PYRAMIDAL]
#     #             filtered_neuron_identities: pd.DataFrame = deepcopy(neuron_identities)
#     #             filtered_neuron_identities = filtered_neuron_identities[['aclu', 'shank', 'cluster', 'qclu']]
#     #             # filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, original_neuron_ids_dict[a_decoder_name])]
#     #             filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, modified_neuron_ids_dict[a_decoder_name])] # require to match to decoders
#     #             filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.qclu, included_qclu_values)] # drop [6, 7], which are said to have double fields - 80 remain
#     #             final_included_aclus = filtered_neuron_identities['aclu'].to_numpy()
#     #             modified_neuron_ids_dict[a_decoder_name] = deepcopy(final_included_aclus) #.tolist()
                
#     #     return modified_neuron_ids_dict
    

#     # def determine_decoder_aclus_filtered_by_qclu_and_frate(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         modified_neuron_ids_dict = TrackTemplates.determine_decoder_aclus_filtered_by_qclu_and_frate(track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)

#     #     """
#     #     return TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=self.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

#     # @classmethod
#     # def determine_active_min_num_unique_aclu_inclusions_requirement(cls, min_num_unique_aclu_inclusions: int, total_num_cells: int, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = cls.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                                                                                 total_num_cells=len(any_list_neuron_IDs))

#     #     """
#     #     required_min_percentage_of_active_cells = float(required_min_percentage_of_active_cells)
#     #     if debug_print:
#     #         print(f'required_min_percentage_of_active_cells: {required_min_percentage_of_active_cells}') # 20% of active cells
#     #     dynamic_percentage_minimum_num_unique_aclu_inclusions: int = int(round((float(total_num_cells) * required_min_percentage_of_active_cells))) # dynamic_percentage_minimum_num_unique_aclu_inclusions: the percentage-based requirement for the number of active cells
#     #     active_min_num_unique_aclu_inclusions_requirement: int = max(dynamic_percentage_minimum_num_unique_aclu_inclusions, min_num_unique_aclu_inclusions)
#     #     if debug_print:
#     #         print(f'active_min_num_unique_aclu_inclusions_requirement: {active_min_num_unique_aclu_inclusions_requirement}')
#     #     return active_min_num_unique_aclu_inclusions_requirement


#     # def min_num_unique_aclu_inclusions_requirement(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

#     #     """
#     #     smallest_template_n_neurons: int = np.min([len(v) for v in self.decoder_neuron_IDs_list]) # smallest_template_n_neurons: the fewest number of neurons any template has
#     #     ## Compute the dynamic minimum number of active cells from current num total cells and the `curr_active_pipeline.sess.config.preprocessing_parameters` values:`
#     #     return self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                             total_num_cells=smallest_template_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)


#     # def min_num_unique_aclu_inclusions_requirement_dict(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> Dict[str, int]:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

#     #     """
#     #     decoder_neuron_IDs_dict = dict(zip(self.get_decoder_names(), self.decoder_neuron_IDs_list))
#     #     decoder_num_neurons_dict = {k:len(v) for k, v in decoder_neuron_IDs_dict.items()}
#     #     return {k:self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                             total_num_cells=a_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells) for k, a_n_neurons in decoder_num_neurons_dict.items()}


#     # @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=['TransitionMatrixComputations'], used_by=[], creation_date='2024-08-02 07:33', related_items=[])
#     # def compute_decoder_transition_matricies(self, n_powers:int=50, use_direct_observations_for_order:bool=True) -> Dict[types.DecoderName, List[NDArray]]:
#     #     """ Computes the position transition matricies for each of the decoders 
#     #     returns a list of length n_powers for each decoder
        
#     #     Usage:
#     #         binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = track_templates.compute_decoder_transition_matricies(n_powers=50)
        
#     #     """
#     #     from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
        
#     #     ## INPUTS: track_templates
#     #     decoders_dict: Dict[types.DecoderName, BasePositionDecoder] = self.get_decoders_dict()
#     #     binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = {}

#     #     for a_decoder_name, a_decoder in decoders_dict.items():
#     #         a_pf1D = deepcopy(a_decoder.pf)
#     #         binned_x_transition_matrix_higher_order_list_dict[a_decoder_name] = TransitionMatrixComputations._compute_position_transition_matrix(a_pf1D.xbin_labels, binned_x_index_sequence=(a_pf1D.filtered_pos_df['binned_x'].dropna().to_numpy()-1), n_powers=n_powers, use_direct_observations_for_order=use_direct_observations_for_order) # the -1 here is to convert to (binned_x_index_sequence = binned_x - 1)

#     #     # OUTPUTS: binned_x_transition_matrix_higher_order_list_dict
#     #     return binned_x_transition_matrix_higher_order_list_dict






# @define(slots=False, repr=False)
# class DirectionalLapsResult(ComputedResult):
#     """ a container for holding information regarding the computation of directional laps.

#     ## Build a `DirectionalLapsResult` container object to hold the result:
#     directional_laps_result = DirectionalLapsResult()
#     directional_laps_result.directional_lap_specific_configs = directional_lap_specific_configs
#     directional_laps_result.split_directional_laps_dict = split_directional_laps_dict
#     directional_laps_result.split_directional_laps_contexts_dict = split_directional_laps_contexts_dict
#     directional_laps_result.split_directional_laps_config_names = split_directional_laps_config_names
#     directional_laps_result.computed_base_epoch_names = computed_base_epoch_names

#     # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
#     directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D


#     long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_LR_shared_aclus_only_one_step_decoder_1D', 'long_RL_shared_aclus_only_one_step_decoder_1D', 'short_LR_shared_aclus_only_one_step_decoder_1D', 'short_RL_shared_aclus_only_one_step_decoder_1D']]

#     """
#     _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
#     directional_lap_specific_configs: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_dict: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_contexts_dict: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_config_names: List[str] = serialized_field(default=Factory(list))
#     computed_base_epoch_names: List[str] = serialized_field(default=Factory(list))

#     long_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     long_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     short_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     short_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)

#     long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
#     long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
#     short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
#     short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

#     # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D

#     def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return DirectionalDecodersTuple(self.long_LR_one_step_decoder_1D, self.long_RL_one_step_decoder_1D, self.short_LR_one_step_decoder_1D, self.short_RL_one_step_decoder_1D)

#     def get_shared_aclus_only_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()
#         """
#         return DirectionalDecodersTuple(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D)


#     def get_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
#         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_one_step_decoder_1D, self.short_LR_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_one_step_decoder_1D, self.short_RL_one_step_decoder_1D))
#         if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
#             return _obj
#         else:
#             return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
    

#     def get_shared_aclus_only_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
#         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D))
#         if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
#             return _obj
#         else:
#             # return _obj.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
#             return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        

#     def filtered_by_included_aclus(self, included_neuronIDs) -> "DirectionalLapsResult":
#         """ Returns a copy of self with each decoder filtered by the `qclu_included_aclus`
        
#         Usage:
        
#         qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
#         modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(included_neuronIDs=qclu_included_aclus)
#         modified_directional_laps_results

#         """
#         directional_laps_results = deepcopy(self)
        
#         decoders_list = [directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D,
#                          directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D
#                         ]
#         modified_decoders_list = []
#         for a_decoder in decoders_list:
#             # a_decoder = deepcopy(directional_laps_results.long_LR_one_step_decoder_1D)
#             is_aclu_qclu_included_list = np.isin(a_decoder.pf.ratemap.neuron_ids, included_neuronIDs)
#             included_aclus = np.array(a_decoder.pf.ratemap.neuron_ids)[is_aclu_qclu_included_list]
#             modified_decoder = a_decoder.get_by_id(included_aclus)
#             modified_decoders_list.append(modified_decoder)

#         ## Assign the modified decoders:
#         directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D, directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D = modified_decoders_list

#         return directional_laps_results
    
#     ## For serialization/pickling:
#     def __getstate__(self):
#         # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
#         state = self.__dict__.copy()
#         return state

#     def __setstate__(self, state):
#         # Restore instance attributes (i.e., _mapping and _keys_at_init).
#         self.__dict__.update(state)
#         # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
#         # super(DirectionalLapsResult, self).__init__() # TypeError: super(type, obj): obj must be an instance or subtype of type.



#     @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps'], creation_date='2025-02-13 16:46', related_items=[])
#     @classmethod
#     def init_from_pipeline_natural_epochs(cls, curr_active_pipeline, progress_print=False) -> "DirectionalLapsResult":
#         """ 2023-10-31 - 4pm  - Main computation function, simply extracts the diretional laps from the existing epochs.

#         PURE?: Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

#                 ## {"even": "RL", "odd": "LR"}

#         #TODO 2023-11-10 21:00: - [ ] Convert above "LR/RL" notation to new "LR/RL" versions:

        
#         History 2025-02-13 16:52 used to be called 'DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs'
        
#         Uses:
        
#             curr_active_pipeline.computation_results[an_epoch_name].computed_data.get('pf1D_Decoder', None)
        
        
#         """

#         long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names() # ('maze1_any', 'maze2_any', 'maze_any')
#         # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
#         # long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.

#         # Unwrap the naturally produced directional placefields:
#         long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
#         # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
#         # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
#         # (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
#         (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
#         # (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
#         # (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

#         # Unpack all directional variables:
#         long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name # ('maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any')

#         # Validate:
#         assert not (curr_active_pipeline.computation_results[long_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
#         assert not (curr_active_pipeline.computation_results[short_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
#         # Fix the computation epochs to be constrained to the proper long/short intervals:
#         was_modified = DirectionalLapsHelpers.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline) # cls: DirectionalLapsResult
#         was_modified = was_modified or DirectionalLapsHelpers.fixup_directional_pipeline_if_needed(curr_active_pipeline)
#         print(f'DirectionalLapsResult.init_from_pipeline_natural_epochs(...): was_modified: {was_modified}')

#         # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
#         # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

#         ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
#         long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]


#         #TODO 2023-12-07 20:48: - [ ] It looks like I'm still only looking at the intersection here! Do I want this?

#         ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
#         # Kamran says LR and RL sets should be shared
#         ## Odd Laps:
#         LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
#         LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
#         LR_n_neurons = len(LR_shared_aclus)
#         if progress_print:
#             print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

#         ## Even Laps:
#         RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
#         RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
#         RL_n_neurons = len(RL_shared_aclus)
#         if progress_print:
#             print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

#         # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
#         long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
#         long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

#         ## Build a `DirectionalLapsResult` (a `ComputedResult`) container object to hold the result:
#         directional_laps_result = DirectionalLapsResult(is_global=True, result_version=DirectionalLapsResult._VersionedResultMixin_version)
#         directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # directional_lap_specific_configs
#         directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)}  # split_directional_laps_dict
#         directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # split_directional_laps_contexts_dict
#         directional_laps_result.split_directional_laps_config_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name] # split_directional_laps_config_names

#         # use the constrained epochs:
#         directional_laps_result.long_LR_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.long_RL_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_LR_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_RL_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

#         # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
#         directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.long_RL_shared_aclus_only_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_LR_shared_aclus_only_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_RL_shared_aclus_only_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

#         return directional_laps_result



