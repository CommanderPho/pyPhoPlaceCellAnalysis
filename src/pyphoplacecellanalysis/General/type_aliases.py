from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Literal
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from typing import NewType
import nptyping as ND
from nptyping import NDArray
from neuropy.utils.type_aliases import aclu_index, DecoderName ## other imports?
from neuropy.utils.result_context import IdentifyingContext

""" Usage:

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
# import neuropy.utils.type_aliases as types
import pyphoplacecellanalysis.General.type_aliases as types

e.g. Dict[types.FilterContextName, Dict[types.ComputationFunctionName: CapturedException]]
"""

""" from `neuropy.utils.type_aliases`
	aclu_index: TypeAlias = int # an integer index that is an aclu
	DecoderName = NewType('DecoderName', str)
"""
FilterContextName: TypeAlias = str # a string identifier of a specific filtering context -- 'maze1_odd', 'maze2_odd', 'maze_odd' -- used in `.get_failed_computations(...)`
ComputationFunctionName: TypeAlias = str # a string identifier of a computation function -- can be either the long or short name -- '_split_to_directional_laps' -- used in `.get_failed_computations(...)`

KnownNamedDecoderTrainedComputeEpochsType = Literal['laps', 'non_pbe']
KnownNamedDecodingEpochsType = Literal['laps', 'replay', 'ripple', 'pbe', 'non_pbe', 'non_pbe_endcaps', 'global']
# Define a type that can only be one of these specific strings
MaskedTimeBinFillType = Literal['ignore', 'last_valid', 'nan_filled', 'dropped'] ## used in `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(...)` to specify how invalid bins (due to too few spikes) are treated.
DataTimeGrain = Literal['per_epoch', 'per_time_bin']
PrePostDeltaCategory = Literal['pre_delta', 'post_delta']



GenericResultTupleIndexType: TypeAlias = IdentifyingContext # an template/stand-in variable that aims to abstract away the unique-hashable index of a single result computed with a given set of parameters. Not yet fully implemented 2025-03-09 17:50 


type_to_name_mapping: Dict = dict(zip([KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain], ['known_named_decoder_trained_compute_epochs_type', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_time_grain']))
name_to_short_name_dict: Dict = dict(zip(['known_named_decoder_trained_compute_epochs_type', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_time_grain'], ['train', 'decode', 'mfill', 'grain']))


# FilterContextName = NewType('FilterContextName', str) 
# ComputationFunctionName = NewType('ComputationFunctionName', str) 
