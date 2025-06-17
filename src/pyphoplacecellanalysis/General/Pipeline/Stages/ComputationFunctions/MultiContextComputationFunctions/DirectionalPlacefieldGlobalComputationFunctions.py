from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from attrs import define, field, Factory, asdict, astuple
from functools import wraps
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
from datetime import datetime, date, timedelta

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Iterable
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
# from numpy.typing import NDArray  # Correct import for NDArray in this context
from typing import NewType
import neuropy.utils.type_aliases as types
# DecoderName = NewType('DecoderName', str)

from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from pyphocorehelpers.exception_helpers import ExceptionPrintingContext
from neuropy.utils.indexing_helpers import NumpyHelpers
from pyphocorehelpers.assertion_helpers import Assert

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.dict_representable import override_dict # used to build config
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange, Epoch, ensure_dataframe
from neuropy.core.epoch import find_data_indicies_from_epoch_times
from neuropy.utils.indexing_helpers import union_of_arrays # `paired_incremental_sort_neurons`
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.utils.indexing_helpers import PandasHelpers, NumpyHelpers, flatten
from neuropy.utils.misc import capitalize_after_underscore


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult # needed in DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

import scipy.stats

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability

decoder_name_str: TypeAlias = str # an string name of a particular decoder, such as 'Long_LR' or 'Short_RL'

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

if TYPE_CHECKING:
    ## typehinting only imports here
    # from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, SingleEpochDecodedResult #typehinting only
    # from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig


# DecodedMarginalResultTuple: TypeAlias = Tuple[List[DynamicContainer], NDArray[float], NDArray[int], NDArray[bool]] # used by 
DecodedMarginalResultTuple: TypeAlias = Tuple[
    List[DynamicContainer],
    NDArray[ND.Shape["*"], ND.Float],
    NDArray[ND.Shape["*"], ND.Int],
    NDArray[ND.Shape["*"], ND.Bool]
] 

# DecodedMarginalResultTuple = NewType('DecodedMarginalResultTuple', Tuple[List[DynamicContainer], NDArray[float], NDArray[int], NDArray[bool]])

# Assume a1 and a2 are your numpy arrays
# def find_shift(a1, a2):
#     """ 
#     from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import find_shift
#     shift = find_shift(a1, a2)
    
#     """
#     correlation = np.correlate(a1, a2, "full")
#     max_index = np.argmax(correlation)
#     shift = max_index - len(a2) + 1
#     return shift



def find_shift(a1, a2):
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import find_shift
    shift = find_shift(a1, a2)
    
    """
    from scipy import signal
    # correlation = np.correlate(a1, a2, "full")
    correlation = signal.correlate(a1, a2, mode="full")
    lags = signal.correlation_lags(a1.size, a2.size, mode="full")
    lag = lags[np.argmax(correlation)]
    return lag


@function_attributes(short_name=None, tags=['laps', 'groundtruth', 'validate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-19 21:42', related_items=[])
def add_laps_groundtruth_information_to_dataframe(curr_active_pipeline, result_laps_epochs_df: pd.DataFrame) -> pd.DataFrame:
    """ adds the ground_truth to the passed dataframe: `result_laps_epochs_df` and returns it:

    Updates: ['maze_id', 'is_LR_dir', 'is_most_likely_track_identity_Long', 'is_most_likely_direction_LR']

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import add_laps_groundtruth_information_to_dataframe

        laps_weighted_corr_merged_df: pd.DataFrame = add_laps_groundtruth_information_to_dataframe(curr_active_pipeline=curr_active_pipeline, result_laps_epochs_df=laps_weighted_corr_merged_df)
        laps_weighted_corr_merged_df


    """
    from neuropy.core import Laps

    ## Inputs: a_directional_merged_decoders_result, laps_df

    if 'lap_id' not in result_laps_epochs_df.columns:
        assert 'lap_idx' in result_laps_epochs_df.columns, f"`result_laps_epochs_df` dataframe must have either 'lap_id' or 'lap_idx' as a column in order to add the correct groundtruths"
        result_laps_epochs_df['lap_id'] = result_laps_epochs_df['lap_idx'] + 1
    
    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_df = laps_obj.to_dataframe()
    laps_df: pd.DataFrame = Laps._update_dataframe_computed_vars(laps_df=laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=curr_active_pipeline.sess, replace_existing=True) # NOTE: .sess is used because global_session is missing the last two laps

    ## 2024-04-20 - HACK
    print(f"2024-04-20 - HACK -- FIXME: Invert the 'is_LR_dir' column since it is clearly reversed. No clue why.")
    laps_df['is_LR_dir'] = np.logical_not(laps_df['is_LR_dir'])
    is_RL_dir = np.logical_not(laps_df['is_LR_dir'])
    laps_df['lap_dir'] = is_RL_dir.astype(int) # 1 for RL, 0 for LR

    ## 2024-01-17 - Updates the `a_directional_merged_decoders_result.laps_epochs_df` with both the ground-truth values and the decoded predictions
    result_laps_epochs_df['maze_id'] = laps_df['maze_id'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
    ## add the 'is_LR_dir' groud-truth column in:
    result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching

    assert np.all([a_col in result_laps_epochs_df.columns for a_col in ('maze_id', 'is_LR_dir')]), f"result_laps_epochs_df.columns: {list(result_laps_epochs_df.columns)}"

    result_laps_epochs_df['true_decoder_index'] = (result_laps_epochs_df['maze_id'].astype(int) * 2) + np.logical_not(result_laps_epochs_df['is_LR_dir']).astype(int)

    return result_laps_epochs_df




# Define the namedtuple
DirectionalDecodersTuple = namedtuple('DirectionalDecodersTuple', ['long_LR', 'long_RL', 'short_LR', 'short_RL'])

@define(slots=False, repr=False, eq=False)
class TrackTemplates(HDFMixin, AttrsBasedClassHelperMixin):
    """ Holds the four directional templates for direction placefield analysis.
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

    History:
        Based off of `ShuffleHelper` on 2023-10-27
        TODO: eliminate functional overlap with `ShuffleHelper`
        TODO: should be moved into `DirectionalPlacefieldGlobalComputation` instead of RankOrder

    """
    long_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
    long_RL_decoder: BasePositionDecoder = serialized_field(repr=False) # keys_only_repr
    short_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
    short_RL_decoder: BasePositionDecoder = serialized_field(repr=False)

    # ## Computed properties
    shared_LR_aclus_only_neuron_IDs: NDArray = serialized_field(repr=True)
    is_good_LR_aclus: NDArray = serialized_field(repr=False)

    shared_RL_aclus_only_neuron_IDs: NDArray = serialized_field(repr=True)
    is_good_RL_aclus: NDArray = serialized_field(repr=False)

    ## Computed properties
    decoder_LR_pf_peak_ranks_list: List = serialized_field(repr=True)
    decoder_RL_pf_peak_ranks_list: List = serialized_field(repr=True)

    rank_method: str = serialized_attribute_field(default="average", is_computable=False, repr=True)


    @property
    def decoder_neuron_IDs_list(self) -> List[NDArray]:
        """ a list of the neuron_IDs for each decoder (independently) """
        return [a_decoder.pf.ratemap.neuron_ids for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)]
    
    @property
    def any_decoder_neuron_IDs(self) -> NDArray:
        """ a list of the neuron_IDs for each decoder (independently) """
        return np.sort(union_of_arrays(*self.decoder_neuron_IDs_list)) # neuron_IDs as they appear in any list

    @property
    def decoder_peak_location_list(self) -> List[NDArray]:
        """ a list of the peak_tuning_curve_center_of_masses for each decoder (independently) """
        return [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)]
    
    @property
    def decoder_peak_rank_list_dict(self) -> Dict[str, NDArray]:
        """ a dict (one for each decoder) of the rank_lists for each decoder (independently) """
        return {a_decoder_name:scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
    @property
    def decoder_aclu_peak_rank_dict_dict(self) -> Dict[str, Dict[types.aclu_index, float]]:
        """ a Dict (one for each decoder) of aclu-to-rank maps for each decoder (independently) """
        return {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method))) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
    @property
    def decoder_normalized_tuning_curves_dict_dict(self) -> Dict[str, Dict[types.aclu_index, NDArray]]:
        """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
        return {a_name:a_decoder.pf.normalized_tuning_curves_dict for a_name, a_decoder in self.get_decoders_dict().items()}
            

    @property
    def decoder_stability_dict_dict(self): # -> Dict[str, Dict[types.aclu_index, NDArray]]:
        # """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
        return {a_name:a_decoder.pf.ratemap.spatial_sparcity for a_name, a_decoder in self.get_decoders_dict().items()}
    

    def get_decoders_tuning_curve_modes(self, peak_mode='peaks', **find_peaks_kwargs) -> Tuple[Dict[decoder_name_str, Dict[types.aclu_index, NDArray]], Dict[decoder_name_str, Dict[types.aclu_index, int]], Dict[decoder_name_str, pd.DataFrame]]:
        """ 2023-12-19 - Uses `scipy.signal.find_peaks to find the number of peaks or ("modes") for each of the cells in the ratemap. 
        Can detect bimodal (or multi-modal) placefields.
        
        Depends on:
            self.tuning_curves
        
        Returns:
            aclu_n_peaks_dict: Dict[int, int] - A mapping between aclu:n_tuning_curve_modes
        Usage:    
            decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict = track_templates.get_decoders_tuning_curve_modes()

        """
        decoder_peaks_results_tuples_dict = {a_decoder_name:a_decoder.pf.ratemap.compute_tuning_curve_modes(peak_mode=peak_mode, **find_peaks_kwargs) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
        # each tuple contains: peaks_dict, aclu_n_peaks_dict, peaks_results_df, so unwrap below
        
        decoder_peaks_dict_dict = {k:v[0] for k,v in decoder_peaks_results_tuples_dict.items()}
        decoder_aclu_n_peaks_dict_dict = {k:v[1] for k,v in decoder_peaks_results_tuples_dict.items()}
        decoder_peaks_results_df_dict = {k:v[2] for k,v in decoder_peaks_results_tuples_dict.items()}

        # return peaks_dict, aclu_n_peaks_dict, unimodal_peaks_dict, peaks_results_dict
        return decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict
    

    @function_attributes(short_name=None, tags=['WORKING', 'peak', 'multi-peak', 'decoder', 'pfs'], input_requires=[], output_provides=[], uses=['get_tuning_curve_peak_positions'], used_by=['add_directional_pf_maximum_peaks'], creation_date='2024-05-21 19:00', related_items=[])
    def get_directional_pf_maximum_peaks_dfs(self, drop_aclu_if_missing_long_or_short: bool = True) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """ The only version that only gets the maximum peaks appropriate for each decoder.

        # 2024-05-21 - Replaces `.get_decoders_aclu_peak_location_df(...)` for properly getting peak locations. Is correct (which is why the old result was replaced) but has a potential drawback of not currently accepting `, **find_peaks_kwargs`. I only see `width=None` ever passed in like this though.

        # 2024-04-09 00:36: - [X] Could be refactored into TrackTemplates

        #TODO 2024-05-21 22:53: - [ ] Noticed that short track always has all-non-NaN peaks (has a value for each peak) and long track is missing values. This doesn't make sense because many of the peaks indicated for short occur only on the long-track, which makes no sense.

        Usage:

            (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df = track_templates.get_directional_pf_maximum_peaks_dfs(drop_aclu_if_missing_long_or_short=False)

            AnyDir_decoder_aclu_MAX_peak_maps_df
            LR_only_decoder_aclu_MAX_peak_maps_df
            RL_only_decoder_aclu_MAX_peak_maps_df


        """
        # drop_aclu_if_missing_long_or_short: bool = True ## default=True; Drop entire row if either long/short is missing a value
        # drop_aclu_if_missing_long_or_short: bool = False
        # from neuropy.utils.indexing_helpers import intersection_of_arrays, union_of_arrays
        from neuropy.utils.indexing_helpers import unwrap_single_item


        ## Split into LR/RL groups to get proper peak differences:
        # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        LR_decoder_names = self.get_LR_decoder_names() # ['long_LR', 'short_LR']
        RL_decoder_names = self.get_RL_decoder_names() # ['long_RL', 'short_RL']

        ## Only the maximums (height=1 items), guaranteed to be a single (or None) location:
        decoder_aclu_MAX_peak_maps_dict: Dict[types.DecoderName, Dict[types.aclu_index, Optional[float]]] = {types.DecoderName(a_name):{k:unwrap_single_item(v) for k, v in deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)))).items()} for a_name, a_decoder in self.get_decoders_dict().items()}
        #TODO 2024-05-21 22:59: - [ ] NEed to ensure that `a_decoder.neuron_IDs` and `a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)` are returned in the same order for sure
            # it should because it's dependent only on `pdf_normalized_tuning_curves`, which is in the neuron_IDs order. The only issue could be if the subpeaks sorting issue happens

        # decoder_aclu_MAX_peak_maps_dict
        AnyDir_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in (LR_decoder_names + RL_decoder_names)}) # either direction decoder

        ## Splits by direction:
        LR_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in LR_decoder_names})
        RL_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in RL_decoder_names})

        ## Drop entire row if either long/short is missing a value:
        if drop_aclu_if_missing_long_or_short:
            LR_only_decoder_aclu_MAX_peak_maps_df = LR_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')
            RL_only_decoder_aclu_MAX_peak_maps_df = RL_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')

            AnyDir_decoder_aclu_MAX_peak_maps_df = AnyDir_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any') # might need to think this through a little better. Currently only using the `AnyDir_*` result with `drop_aclu_if_missing_long_or_short == False`

        ## Compute the difference between the Long/Short peaks: I don't follow this:
        LR_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = LR_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]
        RL_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = RL_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]

        AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_LR'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(LR_decoder_names)].diff(axis='columns').to_numpy()[:, -1]
        AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_RL'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(RL_decoder_names)].diff(axis='columns').to_numpy()[:, -1]

        # ## UNUSED BLOCK:
        # # maximal_peak_only_decoder_aclu_peak_location_df_merged = deepcopy(decoder_aclu_peak_location_df_merged)[decoder_aclu_peak_location_df_merged['long_LR_peak_height'] == 1.0]

        # LR_height_column_names = ['long_LR_peak_height', 'short_LR_peak_height']

        # # [decoder_aclu_peak_location_df_merged[a_name] == 1.0 for a_name in LR_height_column_names]

        # LR_max_peak_dfs = [deepcopy(decoder_aclu_peak_location_df_merged)[decoder_aclu_peak_location_df_merged[a_name] == 1.0].drop(columns=['subpeak_idx', 'series_idx', 'LR_peak_diff', 'RL_peak_diff', a_name]) for a_name in LR_height_column_names]

        # aclus_with_LR_peaks = intersection_of_arrays(*[a_df.aclu.unique() for a_df in LR_max_peak_dfs])
        # aclus_with_LR_peaks

        # ## Align them now:
        # LR_max_peak_dfs = [a_df[a_df.aclu.isin(aclus_with_LR_peaks)] for a_df in LR_max_peak_dfs]
        # LR_max_peak_dfs

        # maximal_peak_only_decoder_aclu_peak_location_df_merged = deepcopy(decoder_aclu_peak_location_df_merged)[decoder_aclu_peak_location_df_merged[LR_height_column_names] == 1.0]
        # maximal_peak_only_decoder_aclu_peak_location_df_merged

        # OUTPUTS: LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df
        return (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df

    ## WARNING 2024-02-07 - The following all use .peak_tuning_curve_center_of_masses: .get_decoder_aclu_peak_maps, get_decoder_aclu_peak_map_dict, get_decoder_aclu_peak_map_dict
    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
    def get_long_short_decoder_shifts(self):
        """ uses `find_shift` """

        def _subfn_long_short_decoder_shift(long_dir_decoder, short_dir_decoder):
            """ finds the offsets
            """
            long_dir_pf1D = deepcopy(long_dir_decoder.pf.ratemap.pdf_normalized_tuning_curves)
            short_dir_pf1D = deepcopy(short_dir_decoder.pf.ratemap.pdf_normalized_tuning_curves)

            assert np.shape(long_dir_pf1D) == np.shape(short_dir_pf1D)
            assert short_dir_decoder.num_neurons == long_dir_decoder.num_neurons
            neuron_ids = deepcopy(short_dir_decoder.pf.ratemap.neuron_ids)
            # xbin_centers = deepcopy(short_dir_decoder.xbin_centers)

            # shift = find_shift(long_dir_pf1D[:,i], short_dir_pf1D[:,i])
            shift = np.array([find_shift(long_dir_pf1D[i,:], short_dir_pf1D[i,:]) for i in np.arange(long_dir_decoder.num_neurons)])
            shift_x = shift.astype(float) * float(long_dir_decoder.pf.bin_info['xstep']) # In position coordinates
            
            return shift_x, shift, neuron_ids


        LR_shift_x, LR_shift, LR_neuron_ids = _subfn_long_short_decoder_shift(long_dir_decoder=self.long_LR_decoder, short_dir_decoder=self.short_LR_decoder)
        RL_shift_x, RL_shift, RL_neuron_ids = _subfn_long_short_decoder_shift(long_dir_decoder=self.long_RL_decoder, short_dir_decoder=self.short_RL_decoder)

        return (LR_shift_x, LR_shift, LR_neuron_ids), (RL_shift_x, RL_shift, RL_neuron_ids)
    
    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
    def get_decoder_aclu_peak_maps(self, peak_mode='CoM') -> DirectionalDecodersTuple:
        """ returns a tuple of dicts, each containing a mapping between aclu:peak_pf_x for a given decoder. 
         
        # Naievely:
        long_LR_aclu_peak_map = deepcopy(dict(zip(self.long_LR_decoder.neuron_IDs, self.long_LR_decoder.peak_locations)))
        long_RL_aclu_peak_map = deepcopy(dict(zip(self.long_RL_decoder.neuron_IDs, self.long_RL_decoder.peak_locations)))
        short_LR_aclu_peak_map = deepcopy(dict(zip(self.short_LR_decoder.neuron_IDs, self.short_LR_decoder.peak_locations)))
        short_RL_aclu_peak_map = deepcopy(dict(zip(self.short_RL_decoder.neuron_IDs, self.short_RL_decoder.peak_locations)))
        
        """
        assert peak_mode in ['peaks', 'CoM']
        if peak_mode == 'peaks':
            # return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode=peak_mode)))) for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)])
            return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_locations))) for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)]) ## #TODO 2024-02-16 04:27: - [ ] This uses .peak_locations which are the positions corresponding to the peak position bin (but not continuously the peak from the curve).
        elif peak_mode == 'CoM':
            return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_tuning_curve_center_of_masses))) for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)])
        else:
            raise NotImplementedError(f"peak_mode: '{peak_mode}' is not supported.")
    

    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
    def get_decoder_aclu_peak_map_dict(self, peak_mode='CoM') -> Dict[decoder_name_str, Dict]:
        return dict(zip(self.get_decoder_names(), self.get_decoder_aclu_peak_maps(peak_mode=peak_mode)))


    def __repr__(self):
        """ 
        TrackTemplates(long_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            long_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            short_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            short_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            shared_LR_aclus_only_neuron_IDs: numpy.ndarray,
            is_good_LR_aclus: NoneType,
            shared_RL_aclus_only_neuron_IDs: numpy.ndarray,
            is_good_RL_aclus: NoneType,
            decoder_LR_pf_peak_ranks_list: list,
            decoder_RL_pf_peak_ranks_list: list
        )
        """
        content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        return f"{type(self).__name__}({content}\n)"


    def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)

    @classmethod
    def get_decoder_names(cls) -> Tuple[str, str, str, str]:
        return ('long_LR','long_RL','short_LR','short_RL')
    
    @classmethod
    def get_LR_decoder_names(cls) -> Tuple[str, str]:
        return ('long_LR', 'short_LR')
    
    @classmethod
    def get_RL_decoder_names(cls) -> Tuple[str, str]:
        return ('long_RL', 'short_RL')
    
    def get_decoders_dict(self) -> Dict[types.DecoderName, BasePositionDecoder]:
        return {'long_LR': self.long_LR_decoder,
            'long_RL': self.long_RL_decoder,
            'short_LR': self.short_LR_decoder,
            'short_RL': self.short_RL_decoder,
        }
    

    def get_track_length_dict(self) -> Dict[types.DecoderName, float]:
        """ gets the length of the track for all four decoders """
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import get_track_length_dict
        decoder_grid_bin_bounds_dict = {a_name:a_decoder.pf.config.grid_bin_bounds for a_name, a_decoder in self.get_decoders_dict().items()}
        assert NumpyHelpers.all_allclose(list(decoder_grid_bin_bounds_dict.values())), f"all decoders should have the same grid_bin_bounds (independent of whether they are built on long/short, etc but they do not! This violates following assumptions."
        grid_bin_bounds = list(decoder_grid_bin_bounds_dict.values())[0] # tuple
        actual_track_length_dict, idealized_track_length_dict = get_track_length_dict(grid_bin_bounds, grid_bin_bounds)
        # idealized_track_length_dict # {'long': 214.0, 'short': 144.0}
        return {a_name:idealized_track_length_dict[a_name.split('_', maxsplit=1)[0]] for a_name in self.get_decoder_names()} # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}
    
    
    # Slicing/Filtering by Neuron Properties _____________________________________________________________________________ #
    def sliced_by_neuron_id(self, included_neuron_ids: NDArray) -> "TrackTemplates":
        """ refactored out of `self.filtered_by_frate(...)` and `TrackTemplates.determine_decoder_aclus_filtered_by_frate(...)`
        """
        _obj = deepcopy(self) # temporary copy of the object
        # assert np.all([(v in _obj.neuron_ids) for v in included_neuron_ids]), f"All included_neuron_ids must already exist in the object: included_neuron_ids: {included_neuron_ids}\n\t_obj.neuron_ids: {_obj.neuron_ids}"
        # n_aclus = len(included_neuron_ids)
        # is_neuron_id_included = np.isin(included_neuron_ids, _obj.neuron_ids)
        # is_neuron_id_included = np.where(np.isin(included_neuron_ids, _obj.neuron_ids))[0]
        # z_scored_tuning_map_matrix = deepcopy(z_scored_tuning_map_matrix)
        long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = _obj.long_LR_decoder, _obj.long_RL_decoder, _obj.short_LR_decoder, _obj.short_RL_decoder
        # original_neuron_ids_list = [a_decoder.pf.ratemap.neuron_ids for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        # is_aclu_included_list = [np.where(np.isin(included_neuron_ids, a_decoder.pf.ratemap.neuron_ids))[0] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        individual_decoder_filtered_aclus_list = [np.array(a_decoder.pf.ratemap.neuron_ids)[np.where(np.isin(included_neuron_ids, a_decoder.pf.ratemap.neuron_ids))[0]] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]

        ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
        filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
        filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
        # build the final shared aclus:
        filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
        # rebuild the is_aclu_included_list from the shared aclus
        # is_aclu_included_list = [np.isin(an_original_neuron_ids, a_filtered_neuron_ids) for an_original_neuron_ids, a_filtered_neuron_ids in zip(original_neuron_ids_list, filtered_direction_shared_aclus_list)]
        filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip((long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder), filtered_direction_shared_aclus_list)]
        long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = filtered_decoder_list # unpack filtered decoders
        
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_LR_decoder, short_LR_decoder), RL_decoder_pair=(long_RL_decoder, short_RL_decoder), rank_method=_obj.rank_method)
        assert np.all(filtered_direction_shared_aclus_list[0] == _obj.shared_LR_aclus_only_neuron_IDs)
        assert np.all(filtered_direction_shared_aclus_list[1] == _obj.shared_RL_aclus_only_neuron_IDs)
        assert len(filtered_direction_shared_aclus_list[0]) == len(_obj.decoder_LR_pf_peak_ranks_list[0])
        assert len(filtered_direction_shared_aclus_list[1]) == len(_obj.decoder_RL_pf_peak_ranks_list[0])
        return _obj


    def filtered_by_frate(self, minimum_inclusion_fr_Hz: float = 5.0) -> "TrackTemplates":
        """ Does not modify self! Returns a copy! Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)
        Usage:
            minimum_inclusion_fr_Hz: float = 5.0
            filtered_decoder_list = [filtered_by_frate(a_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True) for a_decoder in (track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)]
        """
        return self.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=None)
    

    @function_attributes(short_name=None, tags=['main', 'filter', 'qclu', 'frate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 03:50', related_items=[])
    def filtered_by_frate_and_qclu(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> "TrackTemplates":
        """ Does not modify self! Returns a copy! Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        Usage:
            filtered_track_templates = track_templates.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=5.0, included_qclu_values=[1, 2])
        """
        filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate_and_qclu(decoders_dict=self.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)        
        long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = filtered_decoder_list # unpack
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_LR_decoder, short_LR_decoder), RL_decoder_pair=(long_RL_decoder, short_RL_decoder), rank_method=self.rank_method)
        assert np.all(filtered_direction_shared_aclus_list[0] == _obj.shared_LR_aclus_only_neuron_IDs)
        assert np.all(filtered_direction_shared_aclus_list[1] == _obj.shared_RL_aclus_only_neuron_IDs)
        assert len(filtered_direction_shared_aclus_list[0]) == len(_obj.decoder_LR_pf_peak_ranks_list[0])
        assert len(filtered_direction_shared_aclus_list[1]) == len(_obj.decoder_RL_pf_peak_ranks_list[0])
        return _obj



    # Init/ClassMethods __________________________________________________________________________________________________ #

    @classmethod
    def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], rank_method:str='average') -> "TrackTemplates":
        """ 2023-10-31 - Extract from pairs

        """
        long_LR_decoder, short_LR_decoder = LR_decoder_pair
        long_RL_decoder, short_RL_decoder = RL_decoder_pair

        shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
        assert np.all(short_LR_decoder.neuron_IDs == shared_LR_aclus_only_neuron_IDs), f"{short_LR_decoder.neuron_IDs} != {shared_LR_aclus_only_neuron_IDs}"

        shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)
        assert np.all(short_RL_decoder.neuron_IDs == shared_RL_aclus_only_neuron_IDs), f"{short_RL_decoder.neuron_IDs} != {shared_RL_aclus_only_neuron_IDs}"

        # is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
        # shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

        ## 2023-10-11 - Get the long/short peak locations
        # decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        ## Compute the ranks:
        # decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]

        #TODO 2023-11-21 13:06: - [ ] Note these are in order of the original entries, and do not reflect any sorts or ordering changes.

        return cls(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, None, shared_RL_aclus_only_neuron_IDs, None,
                    decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_LR_decoder, short_LR_decoder)],
                    decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_RL_decoder, short_RL_decoder)],
                    rank_method=rank_method)

    @classmethod
    def determine_decoder_aclus_filtered_by_frate_and_qclu(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

        individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

        Usage:
            filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(decoders_dict=track_templates.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

        """
        decoder_names = cls.get_decoder_names() # ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=decoders_dict, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        # individual_decoder_filtered_aclus_list = list(modified_neuron_ids_dict.values())
        individual_decoder_filtered_aclus_list = [modified_neuron_ids_dict[a_decoder_name] for a_decoder_name in decoder_names]
        assert len(individual_decoder_filtered_aclus_list) == 4, f"len(individual_decoder_filtered_aclus_list): {len(individual_decoder_filtered_aclus_list)} but expected 4!"
        original_decoder_list = [deepcopy(decoders_dict[a_decoder_name]) for a_decoder_name in decoder_names]
        ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
        filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
        filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
        # build the final shared aclus:
        filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
        filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip(original_decoder_list, filtered_direction_shared_aclus_list)]
        return filtered_decoder_list, filtered_direction_shared_aclus_list
    
    @classmethod
    def _perform_determine_decoder_aclus_filtered_by_qclu_and_frate(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

        individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

        Usage:
            modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=track_templates.get_decoders_dict())
            
            decoders_dict=self.get_decoders_dict()
            
        """
        # original_neuron_ids_list = [a_decoder.pf.ratemap.neuron_ids for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        original_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder.pf.ratemap.neuron_ids) for a_decoder_name, a_decoder in decoders_dict.items()}
        if (minimum_inclusion_fr_Hz is not None) and (minimum_inclusion_fr_Hz > 0.0):
            modified_neuron_ids_dict = {a_decoder_name:np.array(a_decoder.pf.ratemap.neuron_ids)[a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz] for a_decoder_name, a_decoder in decoders_dict.items()}
        else:            
            modified_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder_neuron_ids) for a_decoder_name, a_decoder_neuron_ids in original_neuron_ids_dict.items()}
        
        if included_qclu_values is not None:
            # filter by included_qclu_values
            for a_decoder_name, a_decoder in decoders_dict.items():
                # a_decoder.pf.spikes_df
                neuron_identities: pd.DataFrame = deepcopy(a_decoder.pf.filtered_spikes_df).spikes.extract_unique_neuron_identities()
                # filtered_neuron_identities: pd.DataFrame = neuron_identities[neuron_identities.neuron_type == NeuronType.PYRAMIDAL]
                filtered_neuron_identities: pd.DataFrame = deepcopy(neuron_identities)
                filtered_neuron_identities = filtered_neuron_identities[['aclu', 'shank', 'cluster', 'qclu']]
                # filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, original_neuron_ids_dict[a_decoder_name])]
                filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, modified_neuron_ids_dict[a_decoder_name])] # require to match to decoders
                filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.qclu, included_qclu_values)] # drop [6, 7], which are said to have double fields - 80 remain
                final_included_aclus = filtered_neuron_identities['aclu'].to_numpy()
                modified_neuron_ids_dict[a_decoder_name] = deepcopy(final_included_aclus) #.tolist()
                
        return modified_neuron_ids_dict
    

    def determine_decoder_aclus_filtered_by_qclu_and_frate(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

        individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

        Usage:
            modified_neuron_ids_dict = TrackTemplates.determine_decoder_aclus_filtered_by_qclu_and_frate(track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)

        """
        return TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=self.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

    @classmethod
    def determine_active_min_num_unique_aclu_inclusions_requirement(cls, min_num_unique_aclu_inclusions: int, total_num_cells: int, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
        """ 2023-12-21 - Compute the dynamic minimum number of active cells

            active_min_num_unique_aclu_inclusions_requirement: int = cls.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
                                                                                                                                    total_num_cells=len(any_list_neuron_IDs))

        """
        required_min_percentage_of_active_cells = float(required_min_percentage_of_active_cells)
        if debug_print:
            print(f'required_min_percentage_of_active_cells: {required_min_percentage_of_active_cells}') # 20% of active cells
        dynamic_percentage_minimum_num_unique_aclu_inclusions: int = int(round((float(total_num_cells) * required_min_percentage_of_active_cells))) # dynamic_percentage_minimum_num_unique_aclu_inclusions: the percentage-based requirement for the number of active cells
        active_min_num_unique_aclu_inclusions_requirement: int = max(dynamic_percentage_minimum_num_unique_aclu_inclusions, min_num_unique_aclu_inclusions)
        if debug_print:
            print(f'active_min_num_unique_aclu_inclusions_requirement: {active_min_num_unique_aclu_inclusions_requirement}')
        return active_min_num_unique_aclu_inclusions_requirement


    def min_num_unique_aclu_inclusions_requirement(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
        """ 2023-12-21 - Compute the dynamic minimum number of active cells

            active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

        """
        smallest_template_n_neurons: int = np.min([len(v) for v in self.decoder_neuron_IDs_list]) # smallest_template_n_neurons: the fewest number of neurons any template has
        ## Compute the dynamic minimum number of active cells from current num total cells and the `curr_active_pipeline.sess.config.preprocessing_parameters` values:`
        return self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
                                                                                total_num_cells=smallest_template_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)


    def min_num_unique_aclu_inclusions_requirement_dict(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> Dict[str, int]:
        """ 2023-12-21 - Compute the dynamic minimum number of active cells

            active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

        """
        decoder_neuron_IDs_dict = dict(zip(self.get_decoder_names(), self.decoder_neuron_IDs_list))
        decoder_num_neurons_dict = {k:len(v) for k, v in decoder_neuron_IDs_dict.items()}
        return {k:self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
                                                                                total_num_cells=a_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells) for k, a_n_neurons in decoder_num_neurons_dict.items()}


    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=['TransitionMatrixComputations'], used_by=[], creation_date='2024-08-02 07:33', related_items=[])
    def compute_decoder_transition_matricies(self, n_powers:int=50, use_direct_observations_for_order:bool=True) -> Dict[types.DecoderName, List[NDArray]]:
        """ Computes the position transition matricies for each of the decoders 
        returns a list of length n_powers for each decoder
        
        Usage:
            binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = track_templates.compute_decoder_transition_matricies(n_powers=50)
        
        """
        from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
        
        ## INPUTS: track_templates
        decoders_dict: Dict[types.DecoderName, BasePositionDecoder] = self.get_decoders_dict()
        binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = {}

        for a_decoder_name, a_decoder in decoders_dict.items():
            a_pf1D = deepcopy(a_decoder.pf)
            binned_x_transition_matrix_higher_order_list_dict[a_decoder_name] = TransitionMatrixComputations._compute_position_transition_matrix(a_pf1D.xbin_labels, binned_x_index_sequence=(a_pf1D.filtered_pos_df['binned_x'].dropna().to_numpy()-1), n_powers=n_powers, use_direct_observations_for_order=use_direct_observations_for_order) # the -1 here is to convert to (binned_x_index_sequence = binned_x - 1)

        # OUTPUTS: binned_x_transition_matrix_higher_order_list_dict
        return binned_x_transition_matrix_higher_order_list_dict




@function_attributes(short_name=None, tags=['ESSENTIAL', 'filter', 'epoch_selection', 'spikes', 'epochs'], input_requires=[], output_provides=[], uses=[], used_by=['filter_and_update_epochs_and_spikes'], creation_date='2024-10-03 00:00', related_items=[])
def co_filter_epochs_and_spikes(active_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, included_aclus: Optional[NDArray]=None, min_num_unique_aclu_inclusions: Optional[int]=None, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, add_unique_aclus_list_column=False, drop_non_epoch_spikes=True):
    """ #TODO 2025-01-08 11:26: - [ ] TO NUMPY?
    
    Filters epochs and spikes to be consistent with one another based on the specified criteria, and updates the epoch IDs.

    Args:
        active_spikes_df (pd.DataFrame): 
        active_epochs_df (pd.DataFrame): 
        included_aclus (Optional[NDArray]): The neuron ids (aclus) to be included.
        min_num_unique_aclu_inclusions (int): The minimum number of unique ACLUs required for inclusion.
        epoch_id_key_name (str, optional): The name of the epoch ID key. Default is 'ripple_epoch_id'.
        no_interval_fill_value (int, optional): The value to fill for no intervals. Default is -1.

    Returns:
        tuple: A tuple containing the filtered active epochs DataFrame and the updated spikes DataFrame.

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import co_filter_epochs_and_spikes
        
        active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)
        filtered_epochs_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=active_spikes_df, active_epochs_df=active_epochs_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, add_unique_aclus_list_column=False, drop_non_epoch_spikes=True)
        filtered_epochs_df


    """
    # Start Filtering
    active_epochs_df = deepcopy(ensure_dataframe(active_epochs_df))
    active_spikes_df = deepcopy(active_spikes_df)
    if included_aclus is not None:
        active_spikes_df = active_spikes_df.spikes.sliced_by_neuron_id(included_aclus)

    # Update epochs and spikes
    _label_column_type = 'int64'
    assert isinstance(active_epochs_df, pd.DataFrame), f"active_epochs should be a dataframe but it is: {type(active_epochs_df)}"
    active_epochs_df['label'] = active_epochs_df['label'].astype(_label_column_type)

    active_spikes_df = active_spikes_df.spikes.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
                                                                              no_interval_fill_value=no_interval_fill_value, should_replace_existing_column=True, drop_non_epoch_spikes=drop_non_epoch_spikes)

    active_epochs_df = active_epochs_df.epochs.adding_active_aclus_information(spikes_df=active_spikes_df, epoch_id_key_name=epoch_id_key_name, add_unique_aclus_list_column=add_unique_aclus_list_column)
    if min_num_unique_aclu_inclusions is not None:
        active_epochs_df = active_epochs_df[active_epochs_df['n_unique_aclus'] >= min_num_unique_aclu_inclusions]
        print(f'min_num_unique_aclu_inclusions: {min_num_unique_aclu_inclusions}')

    # print(f'len(active_epochs_df): {len(active_epochs_df)}')
    active_epochs_df = active_epochs_df.reset_index(drop=True)

    # Update 'Probe_Epoch_id'
    active_spikes_df = active_spikes_df.spikes.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name='label', override_time_variable_name='t_rel_seconds',
                                                                             no_interval_fill_value=no_interval_fill_value, should_replace_existing_column=True, drop_non_epoch_spikes=drop_non_epoch_spikes)

    return active_epochs_df, active_spikes_df



@function_attributes(short_name=None, tags=['ESSENTIAL', 'filter', 'epoch_selection', 'user-annotations', 'replay'], input_requires=['filtered_sessions[*].replay'], output_provides=[], uses=['co_filter_epochs_and_spikes'], used_by=['add_all_extra_epoch_columns'], creation_date='2024-03-08 13:28', related_items=[])
def filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name: str, track_templates: TrackTemplates, required_min_percentage_of_active_cells: float = 0.333333, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1):
    """ #TODO 2025-01-08 11:26: - [ ] TO NUMPY?
    
    Filters epochs and spikes based on the specified criteria, and updates the epoch IDs. Only seems to be for `.replay` events

    Args:
        curr_active_pipeline (object): The current active pipeline object.
        global_epoch_name (str): The name of the global epoch.
        track_templates (object): The track templates object.
        active_min_num_unique_aclu_inclusions_requirement (int): The minimum number of unique ACLUs required for inclusion.
        epoch_id_key_name (str, optional): The name of the epoch ID key. Default is 'ripple_epoch_id'.
        no_interval_fill_value (int, optional): The value to fill for no intervals. Default is -1.

    Returns:
        tuple: A tuple containing the filtered active epochs DataFrame and the updated spikes DataFrame.

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

        filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, active_min_num_unique_aclu_inclusions_requirement, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
        filtered_epochs_df


    """
    from neuropy.core import Epoch

    global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
    if isinstance(global_replays, pd.DataFrame):
        global_replays = Epoch(global_replays.epochs.get_valid_df())

    global_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
    global_spikes_df = global_spikes_df.spikes.sliced_by_neuron_id(track_templates.any_decoder_neuron_IDs)

    # Start Filtering
    active_epochs_df = deepcopy(global_replays.to_dataframe())
    print(f'len(active_epochs_df): {len(active_epochs_df)}')
    active_spikes_df = deepcopy(global_spikes_df)

    ## Add to config somewhere?
    # curr_active_pipeline.config.pre_proessing_params.required_min_percentage_of_active_cells: float = 0.333333
    
    # required_min_percentage_of_active_cells: float = 0.333333 # 20% of active cells
    active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)
    # Update epochs and spikes
    active_epochs_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=active_spikes_df, active_epochs_df=active_epochs_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, add_unique_aclus_list_column=False, drop_non_epoch_spikes=True)

    return active_epochs_df, active_spikes_df


@define(slots=False, repr=False)
class DirectionalLapsResult(ComputedResult):
    """ a container for holding information regarding the computation of directional laps.

    ## Build a `DirectionalLapsResult` container object to hold the result:
    directional_laps_result = DirectionalLapsResult()
    directional_laps_result.directional_lap_specific_configs = directional_lap_specific_configs
    directional_laps_result.split_directional_laps_dict = split_directional_laps_dict
    directional_laps_result.split_directional_laps_contexts_dict = split_directional_laps_contexts_dict
    directional_laps_result.split_directional_laps_config_names = split_directional_laps_config_names
    directional_laps_result.computed_base_epoch_names = computed_base_epoch_names

    # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
    directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D


    long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_LR_shared_aclus_only_one_step_decoder_1D', 'long_RL_shared_aclus_only_one_step_decoder_1D', 'short_LR_shared_aclus_only_one_step_decoder_1D', 'short_RL_shared_aclus_only_one_step_decoder_1D']]

    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    directional_lap_specific_configs: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_dict: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_contexts_dict: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_config_names: List[str] = serialized_field(default=Factory(list))
    computed_base_epoch_names: List[str] = serialized_field(default=Factory(list))

    long_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    long_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    short_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    short_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)

    long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
    long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
    short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
    short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

    # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D

    def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_one_step_decoder_1D, self.long_RL_one_step_decoder_1D, self.short_LR_one_step_decoder_1D, self.short_RL_one_step_decoder_1D)

    def get_shared_aclus_only_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D)


    def get_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_one_step_decoder_1D, self.short_LR_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_one_step_decoder_1D, self.short_RL_one_step_decoder_1D))
        if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
            return _obj
        else:
            return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
    

    def get_shared_aclus_only_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D))
        if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
            return _obj
        else:
            # return _obj.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
            return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        

    def filtered_by_included_aclus(self, included_neuronIDs) -> "DirectionalLapsResult":
        """ Returns a copy of self with each decoder filtered by the `qclu_included_aclus`
        
        Usage:
        
        qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
        modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(included_neuronIDs=qclu_included_aclus)
        modified_directional_laps_results

        """
        directional_laps_results = deepcopy(self)
        
        decoders_list = [directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D,
                         directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D
                        ]
        modified_decoders_list = []
        for a_decoder in decoders_list:
            # a_decoder = deepcopy(directional_laps_results.long_LR_one_step_decoder_1D)
            is_aclu_qclu_included_list = np.isin(a_decoder.pf.ratemap.neuron_ids, included_neuronIDs)
            included_aclus = np.array(a_decoder.pf.ratemap.neuron_ids)[is_aclu_qclu_included_list]
            modified_decoder = a_decoder.get_by_id(included_aclus)
            modified_decoders_list.append(modified_decoder)

        ## Assign the modified decoders:
        directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D, directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D = modified_decoders_list

        return directional_laps_results
    
    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(DirectionalLapsResult, self).__init__() # TypeError: super(type, obj): obj must be an instance or subtype of type.



    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps'], creation_date='2025-02-13 16:46', related_items=[])
    @classmethod
    def init_from_pipeline_natural_epochs(cls, curr_active_pipeline, progress_print=False) -> "DirectionalLapsResult":
        """ 2023-10-31 - 4pm  - Main computation function, simply extracts the diretional laps from the existing epochs.

        PURE?: Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

                ## {"even": "RL", "odd": "LR"}

        #TODO 2023-11-10 21:00: - [ ] Convert above "LR/RL" notation to new "LR/RL" versions:

        
        History 2025-02-13 16:52 used to be called 'DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs'
        
        Uses:
        
            curr_active_pipeline.computation_results[an_epoch_name].computed_data.get('pf1D_Decoder', None)
        
        
        """

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names() # ('maze1_any', 'maze2_any', 'maze_any')
        # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.

        # Unwrap the naturally produced directional placefields:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        # (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
        # (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
        # (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name # ('maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any')

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        was_modified = DirectionalLapsHelpers.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline) # cls: DirectionalLapsResult
        was_modified = was_modified or DirectionalLapsHelpers.fixup_directional_pipeline_if_needed(curr_active_pipeline)
        print(f'DirectionalLapsResult.init_from_pipeline_natural_epochs(...): was_modified: {was_modified}')

        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]


        #TODO 2023-12-07 20:48: - [ ] It looks like I'm still only looking at the intersection here! Do I want this?

        ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
        # Kamran says LR and RL sets should be shared
        ## Odd Laps:
        LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        LR_n_neurons = len(LR_shared_aclus)
        if progress_print:
            print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

        ## Even Laps:
        RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        RL_n_neurons = len(RL_shared_aclus)
        if progress_print:
            print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

        # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
        long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

        ## Build a `DirectionalLapsResult` (a `ComputedResult`) container object to hold the result:
        directional_laps_result = DirectionalLapsResult(is_global=True, result_version=DirectionalLapsResult._VersionedResultMixin_version)
        directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # directional_lap_specific_configs
        directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)}  # split_directional_laps_dict
        directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # split_directional_laps_contexts_dict
        directional_laps_result.split_directional_laps_config_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name] # split_directional_laps_config_names

        # use the constrained epochs:
        directional_laps_result.long_LR_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
        directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_shared_aclus_only_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_shared_aclus_only_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_shared_aclus_only_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        return directional_laps_result




@metadata_attributes(short_name=None, tags=['helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-24 00:00', related_items=[])
class DirectionalLapsHelpers:
    """ 2023-10-24 - Directional Placefields Computations

    use_direction_dependent_laps

    from neuropy.core.laps import Laps
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

    curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline=curr_active_pipeline, add_created_configs_to_pipeline=True)




    Computing:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

        # Run directional laps and set the global result:
        curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] = DirectionalLapsHelpers.complete_directional_pfs_computations(curr_active_pipeline)


    """


    lap_direction_suffix_list = ['_odd', '_even', '_any'] # ['maze1_odd', 'maze1_even', 'maze1_any', 'maze2_odd', 'maze2_even', 'maze2_any', 'maze_odd', 'maze_even', 'maze_any']
    # lap_direction_suffix_list = ['_odd', '_even', ''] # no '_any' prefix, instead reuses the existing names
    split_directional_laps_name_parts = ['odd_laps', 'even_laps'] # , 'any_laps'

    split_all_laps_name_parts = ['odd_laps', 'even_laps', 'any']
    # ['maze_even_laps', 'maze_odd_laps']

    @classmethod
    def has_duplicated_memory_references(cls, *args) -> bool:
        # Check for duplicated memory references in the configs first:
        memory_ids = [id(a_config) for a_config in args] # YUP, they're different for odd/even but duplicated for long/short
        has_duplicated_reference: bool = len(np.unique(memory_ids)) < len(memory_ids)
        return has_duplicated_reference

    @classmethod
    def deduplicate_memory_references(cls, *args) -> list:
        """ Ensures that all entries in the args list point to unique memory addresses, deduplicating them with `deepcopy` if needed.

        Usage:

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

            args = DirectionalLapsHelpers.deduplicate_memory_references(args)

        """
        has_duplicated_reference: bool = cls.has_duplicated_memory_references(*args)
        if has_duplicated_reference:
            de_deuped_args = [deepcopy(v) for v in args]
            assert not cls.has_duplicated_memory_references(*de_deuped_args), f"duplicate memory references still exist even after de-duplicating with deepcopy!!!"
            return de_deuped_args
        else:
            return args

    @function_attributes(short_name=None, tags=['fixup'], input_requires=[], output_provides=[], uses=[], used_by=['.fixup_directional_pipeline_if_needed'], creation_date='2025-02-19 13:21', related_items=[])
    @classmethod
    def post_fixup_filtered_contexts(cls, curr_active_pipeline, debug_print=False) -> bool:
        """ 2023-10-24 - tries to update misnamed `curr_active_pipeline.filtered_contexts`

            curr_active_pipeline.filtered_contexts with correct filter_names

            Uses: `curr_active_pipeline.filtered_epoch`
            Updates: `curr_active_pipeline.filtered_contexts`

        Still needed for 2023-11-29 to add back in the 'lap_dir' key

        History: factored out of BatchCompletionHandler

        NOTE: works for non-directional contexts as well, fixing `filter_name` as needed.


        """
        was_updated = False
        for a_name, a_named_timerange in curr_active_pipeline.filtered_epochs.items():
            # `desired_filter_name`: the correct name to be set as the .filter_name in the context
            # 2023-11-29 - as of right now, I think the full name including the lap_dir 'maze1_any' (mode 2) should be used as this is literally what the name of the corresponding filtering function is.
            # desired_filter_name:str = a_named_timerange.name # mode 1: uses the period name 'maze1' without the lap_dir part, probably best for compatibility in most places
            desired_filter_name:str = a_name  # mode 2: uses the config_name: 'maze1_any', includes the lap_dir part

            if debug_print:
                print(f'"{a_name}" - desired_filter_name: "{desired_filter_name}"')
            a_filtered_ctxt = curr_active_pipeline.filtered_contexts[a_name]
            ## Parse the name into the parts:
            _split_parts = a_name.split('_')
            if (len(_split_parts) >= 2):
                # also have lap_dir:
                a_split_name, lap_dir, *remainder_list = a_name.split('_') # successfully splits 'maze_odd_laps' into good
                if (a_filtered_ctxt.filter_name != desired_filter_name):
                    was_updated = True
                    print(f"WARNING: filtered_contexts['{a_name}']'s actual context name is incorrect. \n\ta_filtered_ctxt.filter_name: '{a_filtered_ctxt.filter_name}' != desired_filter_name: '{desired_filter_name}'\n\tUpdating it. (THIS IS A HACK)")
                    a_filtered_ctxt = a_filtered_ctxt.overwriting_context(filter_name=desired_filter_name, lap_dir=lap_dir)

                if not a_filtered_ctxt.has_keys('lap_dir'):
                    print(f'WARNING: context {a_name} is missing the "lap_dir" key despite directional laps being detected from the name! Adding missing context key! lap_dir="{lap_dir}"')
                    a_filtered_ctxt = a_filtered_ctxt.adding_context_if_missing(lap_dir=lap_dir) # Add the lap_dir context if it was missing
                    was_updated = True

            else:
                if a_filtered_ctxt.filter_name != desired_filter_name:
                    was_updated = True
                    print(f"WARNING: filtered_contexts['{a_name}']'s actual context name is incorrect. \n\ta_filtered_ctxt.filter_name: '{a_filtered_ctxt.filter_name}' != desired_filter_name: '{desired_filter_name}'\n\tUpdating it. (THIS IS A HACK)")
                    a_filtered_ctxt = a_filtered_ctxt.overwriting_context(filter_name=desired_filter_name)

            if debug_print:
                print(f'\t{a_filtered_ctxt.to_dict()}')
            curr_active_pipeline.filtered_contexts[a_name] = a_filtered_ctxt # correct the context

        # end for
        return was_updated

    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=['.has_duplicated_memory_references'], used_by=['.fixup_directional_pipeline_if_needed'], creation_date='2025-02-19 13:22', related_items=[])
    @classmethod
    def fix_computation_epochs_if_needed(cls, curr_active_pipeline, debug_print=False):
        """2023-11-10 - WORKING NOW - decouples (if they refer to shared memory locations, which was causing a bug) the configs and constrains the computation_epochs to the relevant long/short periods. Will need recomputations if was_modified """
        #TODO 2023-11-10 23:32: - [ ] WORKING NOW!
        # 2023-11-10 21:15: - [X] Not yet finished! Does not work due to shared memory issue. Changes to the first two affect the next two

        was_modified: bool = False
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.
        if debug_print:
            print(f'long_epoch_obj: {long_epoch_obj}, short_epoch_obj: {short_epoch_obj}')
        assert short_epoch_obj.n_epochs > 0
        assert long_epoch_obj.n_epochs > 0

        ## {"even": "RL", "odd": "LR"}
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']

        (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]

        # Check for duplicated memory references in the configs first:
        has_duplicated_reference: bool = cls.has_duplicated_memory_references(long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)
        if has_duplicated_reference:
            long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config = [deepcopy(a_config) for a_config in (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)]
            assert not cls.has_duplicated_memory_references(long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config), f"duplicate memory references still exist even after de-duplicating with deepcopy!!!"
            was_modified = was_modified or True # duplicated references fixed!
            # re-assign:
            for an_epoch_name, a_deduplicated_config in zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)):
                curr_active_pipeline.computation_results[an_epoch_name].computation_config = a_deduplicated_config
            print(f'deduplicated references!')

        original_num_epochs = np.array([curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.n_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)])
        if debug_print:
            print(f'original_num_epochs: {original_num_epochs}')
        assert np.all(original_num_epochs > 0)
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        # relys on: long_epoch_obj, short_epoch_obj
        for an_epoch_name in (long_LR_name, long_RL_name):
            curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs = deepcopy(curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.time_slice(long_epoch_obj.t_start, long_epoch_obj.t_stop))

        for an_epoch_name in (short_LR_name, short_RL_name):
            curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs = deepcopy(curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.time_slice(short_epoch_obj.t_start, short_epoch_obj.t_stop))

        modified_num_epochs = np.array([curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.n_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)])
        if debug_print:
            print(f'modified_num_epochs: {modified_num_epochs}')
        was_modified = was_modified or np.any(original_num_epochs != modified_num_epochs)
        assert np.all(modified_num_epochs > 0)

        return was_modified

    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=['.post_fixup_filtered_contexts', '.fix_computation_epochs_if_needed'], used_by=[], creation_date='2025-02-19 13:22', related_items=[])
    @classmethod
    def fixup_directional_pipeline_if_needed(cls, curr_active_pipeline, debug_print=False) -> bool:
        """2023-11-29 - Updates the filtered context and decouples the configs and constrains the computation_epochs to the relevant long/short periods as needed. Will need recomputations if was_modified """
        #TODO 2023-11-10 23:32: - [ ] WORKING NOW!
        # 2023-11-10 21:15: - [X] Not yet finished! Does not work due to shared memory issue. Changes to the first two affect the next two

        was_modified: bool = False
        was_modified = was_modified or DirectionalLapsHelpers.post_fixup_filtered_contexts(curr_active_pipeline)
        was_modified = was_modified or DirectionalLapsHelpers.fix_computation_epochs_if_needed(curr_active_pipeline)
        return was_modified

    @classmethod
    def update_lap_directions_properties(cls, curr_active_pipeline, debug_print=False) -> bool:
        """2024-01-24 - Updates laps for all filtered and unfiltered sessions with new column definitions session and filtered versions:"""
        was_modified = False
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        curr_sess = curr_active_pipeline.sess
        origin_lap_df = curr_sess.laps.to_dataframe()
        added_column = ('is_LR_dir' not in origin_lap_df.columns)
        was_modified = was_modified or added_column
        curr_sess.laps.update_lap_dir_from_smoothed_velocity(pos_input=curr_sess.position)
        curr_sess.laps.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        
        for an_epoch_name, curr_sess in curr_active_pipeline.filtered_sessions.items():
            origin_lap_df = curr_sess.laps.to_dataframe()
            added_column = ('is_LR_dir' not in origin_lap_df.columns)
            curr_sess.laps.update_lap_dir_from_smoothed_velocity(pos_input=curr_sess.position)
            curr_sess.laps.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            was_modified = was_modified or added_column

        return was_modified # just always assume they changed.


    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps'], creation_date='2025-02-13 16:46', related_items=[])
    @classmethod
    def build_global_directional_result_from_natural_epochs(cls, curr_active_pipeline, progress_print=False) -> "DirectionalLapsResult":
        """ 2023-10-31 - 4pm  - Main computation function, simply extracts the diretional laps from the existing epochs.

        PURE?: Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

                ## {"even": "RL", "odd": "LR"}

        #TODO 2023-11-10 21:00: - [ ] Convert above "LR/RL" notation to new "LR/RL" versions:

        
        Uses:
        
            curr_active_pipeline.computation_results[an_epoch_name].computed_data.get('pf1D_Decoder', None)
        
        
        """

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names() # ('maze1_any', 'maze2_any', 'maze_any')
        # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.

        # Unwrap the naturally produced directional placefields:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
        (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
        (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name # ('maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any')

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        was_modified = cls.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline)
        was_modified = was_modified or DirectionalLapsHelpers.fixup_directional_pipeline_if_needed(curr_active_pipeline)
        print(f'build_global_directional_result_from_natural_epochs(...): was_modified: {was_modified}')

        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]


        #TODO 2023-12-07 20:48: - [ ] It looks like I'm still only looking at the intersection here! Do I want this?

        # # ## Version 2023-10-30 - All four templates with same shared_aclus version:
        # # # Prune to the shared aclus in both epochs (short/long):
        # active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        # # Find only the common aclus amongst all four templates:
        # shared_aclus = np.array(list(set.intersection(*map(set,active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        # n_neurons = len(shared_aclus)
        # print(f'n_neurons: {n_neurons}, shared_aclus: {shared_aclus}')
        # # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

        ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
        # Kamran says LR and RL sets should be shared
        ## Odd Laps:
        LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        LR_n_neurons = len(LR_shared_aclus)
        if progress_print:
            print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

        ## Even Laps:
        RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        RL_n_neurons = len(RL_shared_aclus)
        if progress_print:
            print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

        # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
        long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]


        # ## Encode/Decode from global result:
        # # Unpacking:
        # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # # split_directional_laps_config_names

        ## Build a `DirectionalLapsResult` (a `ComputedResult`) container object to hold the result:
        directional_laps_result = DirectionalLapsResult(is_global=True, result_version=DirectionalLapsResult._VersionedResultMixin_version)
        directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # directional_lap_specific_configs
        directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)}  # split_directional_laps_dict
        directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # split_directional_laps_contexts_dict
        directional_laps_result.split_directional_laps_config_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name] # split_directional_laps_config_names

        # # use the non-constrained epochs:
        # directional_laps_result.long_LR_one_step_decoder_1D = long_LR_laps_one_step_decoder_1D
        # directional_laps_result.long_RL_one_step_decoder_1D = long_RL_laps_one_step_decoder_1D
        # directional_laps_result.short_LR_one_step_decoder_1D = short_LR_laps_one_step_decoder_1D
        # directional_laps_result.short_RL_one_step_decoder_1D = short_RL_laps_one_step_decoder_1D

        # use the constrained epochs:
        directional_laps_result.long_LR_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
        directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_shared_aclus_only_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_shared_aclus_only_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_shared_aclus_only_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        return directional_laps_result


def validate_has_directional_laps(curr_active_pipeline, computation_filter_name='maze'):
    # Unpacking:
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
    directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = directional_laps_results.directional_lap_specific_configs, directional_laps_results.split_directional_laps_dict, directional_laps_results.split_directional_laps_config_names, directional_laps_results.computed_base_epoch_names

    long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
    long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()

    # determine if needs
    has_updated_laps_dirs = ('is_LR_dir' in curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe().columns)
    
    # has_matching_filter_name = (computation_filter_name in split_directional_laps_config_names) # what is this requirement?
    # Does it pass a valid name, or just 'maze'?
        # >> It does pass a valid name: 'maze_any'.
    has_matching_filter_name = True

    # assert (computation_filter_name in computed_base_epoch_names), f'computation_filter_name: {computation_filter_name} is missing from computed_base_epoch_names: {computed_base_epoch_names} '
    return (has_matching_filter_name and has_updated_laps_dirs)
    # return (computation_filter_name in computed_base_epoch_names)

# Avoid division by zero and handle NaN values
_prevent_div_by_zero_epsilon: float = 1e-10

@define(slots=False, repr=False)
class DirectionalPseudo2DDecodersResult(ComputedResult):
    """ a container for holding information regarding the computation of merged (pseudo2D) directional placefields.

    #TODO 2024-05-22 17:26: - [ ] 'DirectionalMergedDecodersResult' -> 'DirectionalPseudo2DDecodersResult'

    #TODO 2025-03-04 17:35: - [ ] Limited in the following ways:
        - Assumes directional (long/short) configurations, not general
        - Assumes two epochs of interest: (laps/ripples)
        - Outputs a bunch of separate marginal files which seems excessive
        
    
    {'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalMergedDecodersResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalPseudo2DDecodersResult'}


    ## Get the result after computation:
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    
    all_directional_decoder_dict_value = directional_merged_decoders_result.all_directional_decoder_dict
    all_directional_pf1D_Decoder_value = directional_merged_decoders_result.all_directional_pf1D_Decoder
    # long_directional_pf1D_Decoder_value = directional_merged_decoders_result.long_directional_pf1D_Decoder
    # long_directional_decoder_dict_value = directional_merged_decoders_result.long_directional_decoder_dict
    # short_directional_pf1D_Decoder_value = directional_merged_decoders_result.short_directional_pf1D_Decoder
    # short_directional_decoder_dict_value = directional_merged_decoders_result.short_directional_decoder_dict

    all_directional_laps_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
    all_directional_ripple_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result

    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = directional_merged_decoders_result.laps_track_identity_marginals_tuple
    ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.ripple_directional_marginals_tuple
    ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = directional_merged_decoders_result.ripple_track_identity_marginals_tuple
    
    ripple_decoding_time_bin_size: float = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.decoding_time_bin_size
    ripple_decoding_time_bin_size
    laps_decoding_time_bin_size: float = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.decoding_time_bin_size
    laps_decoding_time_bin_size

    laps_all_epoch_bins_marginals_df = directional_merged_decoders_result.laps_all_epoch_bins_marginals_df
    ripple_all_epoch_bins_marginals_df = directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df

    """
    _VersionedResultMixin_version: str = "2024.10.09_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    all_directional_decoder_dict: Dict[str, PfND] = serialized_field(default=None)
    all_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_decoder_dict: Dict[str, PfND] = serialized_field(default=None)
    short_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    short_directional_decoder_dict: Dict[str, PfND] = serialized_field(default=None)

    # Posteriors computed via the all_directional decoder:
    all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)
    all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)

    # Marginalized posteriors computed from above posteriors:
    laps_directional_marginals_tuple: Tuple = serialized_field(default=None) # laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
    laps_track_identity_marginals_tuple: Tuple = serialized_field(default=None)
    laps_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Laps'})
    
    ripple_directional_marginals_tuple: Tuple = serialized_field(default=None)
    ripple_track_identity_marginals_tuple: Tuple = serialized_field(default=None) 
    ripple_non_marginalized_decoder_marginals_tuple: Tuple = serialized_field(default=None, metadata={'field_added': "2024.10.09_0"}, hdf_metadata={'epochs': 'Replay'})

    # Computed Properties ________________________________________________________________________________________________ #
    @property
    def laps_epochs_df(self) -> pd.DataFrame:
        a_df = deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs)
        return ensure_dataframe(a_df)
    @laps_epochs_df.setter
    def laps_epochs_df(self, value: pd.DataFrame):
        self.all_directional_laps_filter_epochs_decoder_result.filter_epochs = ensure_dataframe(value)


    @property
    def ripple_epochs_df(self) -> pd.DataFrame:
        return deepcopy(self.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)

    @property
    def laps_decoding_time_bin_size(self) -> float:
        return self.all_directional_laps_filter_epochs_decoder_result.decoding_time_bin_size

    @property
    def ripple_decoding_time_bin_size(self) -> float:
        return self.all_directional_ripple_filter_epochs_decoder_result.decoding_time_bin_size

    # *epoch_bins_marginals_df _____________________________________________________________________________________________ #
    @property
    def laps_all_epoch_bins_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 
        the `*_all_epoch_bins_marginals_df` has a row per epoch. 

        all_epoch_bins_marginal

        """
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = self.laps_track_identity_marginals_tuple

        laps_marginals_df = pd.DataFrame(np.hstack((laps_directional_all_epoch_bins_marginal, laps_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        laps_marginals_df['lap_idx'] = laps_marginals_df.index.to_numpy()
        laps_marginals_df['lap_start_t'] = self.laps_epochs_df['start'].to_numpy()
        
        ## ensure we have the generic columns too (duplicated):
        laps_marginals_df['epoch_idx'] = laps_marginals_df['lap_idx']
        laps_marginals_df['epoch_start_t'] = laps_marginals_df['lap_start_t']

        return laps_marginals_df

    @property
    def ripple_all_epoch_bins_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 
        the `*_all_epoch_bins_marginals_df` has a row per epoch. 
        """
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = self.ripple_directional_marginals_tuple
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = self.ripple_track_identity_marginals_tuple

        ## Ripple marginals_df:
        ripple_marginals_df = pd.DataFrame(np.hstack((ripple_directional_all_epoch_bins_marginal, ripple_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
        ripple_marginals_df['ripple_start_t'] = self.ripple_epochs_df['start'].to_numpy()
        
        ## ensure we have the generic columns too (duplicated):
        ripple_marginals_df['epoch_idx'] = ripple_marginals_df['ripple_idx']
        ripple_marginals_df['epoch_start_t'] = ripple_marginals_df['ripple_start_t']

        return ripple_marginals_df

    # *time_bin_marginals_df _____________________________________________________________________________________________ #
    @property
    def laps_time_bin_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 

        the `*_time_bin_marginals_df` has a row per time bin instead of per epoch. 
        
        """
        (self.laps_directional_marginals_tuple, self.laps_track_identity_marginals_tuple, self.laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = self.all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                            additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = self.laps_track_identity_marginals_tuple
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = self.laps_non_marginalized_decoder_marginals_tuple
        laps_time_bin_marginals_df: pd.DataFrame = self.all_directional_laps_filter_epochs_decoder_result.build_per_time_bin_marginals_df(active_marginals_tuple=(laps_directional_marginals, laps_track_identity_marginals, non_marginalized_decoder_marginals), columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short'], ['long_LR', 'long_RL', 'short_LR', 'short_RL']))
        
        ## #TODO 2024-10-08 19:37: - [ ] Add extra columns
        return laps_time_bin_marginals_df

    @property
    def ripple_time_bin_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 
        the `*_time_bin_marginals_df` has a row per time bin instead of per epoch. 
        """
        (self.ripple_directional_marginals_tuple, self.ripple_track_identity_marginals_tuple, self.ripple_non_marginalized_decoder_marginals_tuple), ripple_marginals_df = self.all_directional_ripple_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='ripple_idx', epoch_start_t_col_name='ripple_start_t',
                                                                                                                                                                        additional_transfer_column_names=['start','stop','label','duration'])
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = self.ripple_directional_marginals_tuple
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = self.ripple_track_identity_marginals_tuple
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = self.ripple_non_marginalized_decoder_marginals_tuple
        ## Build the per-time-bin results:
        ripple_time_bin_marginals_df: pd.DataFrame = self.all_directional_ripple_filter_epochs_decoder_result.build_per_time_bin_marginals_df(active_marginals_tuple=(ripple_directional_marginals, ripple_track_identity_marginals, non_marginalized_decoder_marginals), columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short'], ['long_LR', 'long_RL', 'short_LR', 'short_RL']))
        ## #TODO 2024-10-08 19:37: - [ ] Add extra columns
        
        return ripple_time_bin_marginals_df



    def __attrs_post_init__(self):
        # Computes and initializes the marginal properties:
        # laps_epochs_df = deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs).to_dataframe()
        marginals_needs_purge = True
        if all([v is not None for v in [self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result]]):
            try:
                self.perform_compute_marginals()
                marginals_needs_purge = False
            except AttributeError as e:				
                raise e

        if marginals_needs_purge:
            self.laps_directional_marginals_tuple = None
            self.laps_track_identity_marginals_tuple = None
            self.ripple_directional_marginals_tuple = None
            self.ripple_track_identity_marginals_tuple = None


    def perform_compute_marginals(self):
        """ 
        Called after [self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result] are updated to compute the four marginal tuples

        Requires: self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result

        Updates: 
        (self.laps_directional_marginals_tuple, self.laps_track_identity_marginals_tuple
        self.ripple_directional_marginals_tuple, self.ripple_track_identity_marginals_tuple
        """
        # Computes and initializes the marginal properties:        
        if self.all_directional_laps_filter_epochs_decoder_result is not None:
            (self.laps_directional_marginals_tuple, self.laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = self.all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                                 additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir'])
        if self.all_directional_ripple_filter_epochs_decoder_result is not None:
            (self.ripple_directional_marginals_tuple, self.ripple_track_identity_marginals_tuple, ripple_non_marginalized_decoder_marginals_tuple), ripple_marginals_df = self.all_directional_ripple_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='ripple_idx', epoch_start_t_col_name='ripple_start_t',
                                                                                                                                                                 additional_transfer_column_names=['start','stop','label','duration'])

        

    # Low-level direct marginal computation functions ____________________________________________________________________ #
    @function_attributes(short_name=None, tags=['pseudo2D', 'marginal', 'laps', 'pbe'], input_requires=[], output_provides=[], uses=['cls.build_generalized_non_marginalized_raw_posteriors'], used_by=['GenericDecoderDictDecodedEpochsDictResult'], creation_date='2025-03-20 08:30', related_items=[])
    @classmethod
    def perform_compute_specific_marginals(cls, a_result: DecodedFilterEpochsResult, marginal_context: IdentifyingContext) -> pd.DataFrame:
        """ Computes the appropriate pseudo2D marginals for the result given the context (requires laps/pbe)        
        Usage:

            ## Common/shared for all decoded epochs:
            unique_decoder_names = ['long', 'short']
            masked_bin_fill_mode='nan_filled'

            
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_contexts(IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= 0.025, known_named_decoding_epochs_type='laps', masked_time_bin_fill_type= 'ignore', data_grain= 'per_time_bin'), return_multiple_matches=False)

            ## INPUTS: a_result, masked_bin_fill_mode
            _updated_context = deepcopy(best_matching_context).overwriting_context(masked_time_bin_fill_type=masked_bin_fill_mode)
            _updated_context

            ## MASKED with NaNs (no backfill):
            dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode=masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.
            ## Computes marginals for `dropping_masked_laps_pseudo2D_continuous_specific_decoded_result`
            dropping_masked_laps_decoded_marginal_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, marginal_context=_updated_context)
            a_new_fully_generic_result.updating_results_for_context(new_context=_updated_context, a_result=deepcopy(dropping_masked_laps_pseudo2D_continuous_specific_decoded_result), a_decoder=deepcopy(a_decoder), a_decoded_marginal_posterior_df=deepcopy(dropping_masked_laps_decoded_marginal_posterior_df)) ## update using the result

        """
        data_grain = marginal_context.get('data_grain', None)
        assert data_grain is not None, f"marginal_context: {marginal_context}"
        assert data_grain in ['per_time_bin', 'per_epoch'], f"marginal_context: {marginal_context}"
        
        known_named_decoding_epochs_type = marginal_context.get('known_named_decoding_epochs_type', None)
        assert known_named_decoding_epochs_type is not None
        assert known_named_decoding_epochs_type in ['laps', 'replay', 'ripple', 'pbe', 'non_pbe', 'non_pbe_endcaps', 'global'], f"known_named_decoding_epochs_type: {known_named_decoding_epochs_type}" # ['laps', 'pbe', 'global']

        # ==================================================================================================================== #
        # See if it is a 4-tuple Pseudo2D (all-directional) decoder or a 2-tuple version by checking shape:                    #
        # ==================================================================================================================== #
        p_x_given_n_list = DirectionalPseudo2DDecodersResult.get_proper_p_x_given_n_list(a_result)
        p_x_given_n_list_second_dim_sizes = np.array([np.shape(a_p_x_given_n)[1] for a_p_x_given_n in p_x_given_n_list])
        is_four_tuple_pseudo2D_decoder = np.all((p_x_given_n_list_second_dim_sizes == 4)) # only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        is_track_identity_only_pseudo2D_decoder: bool = np.all((p_x_given_n_list_second_dim_sizes == 2)) # only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 2, corresponding to ['long', 'short']
        
        if is_four_tuple_pseudo2D_decoder:
            unique_decoder_names: List[str] = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            columns_tuple = (['P_LR', 'P_RL'], ['P_Long', 'P_Short'], unique_decoder_names)
        else:            
            assert is_track_identity_only_pseudo2D_decoder, f"only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 2, corresponding to ['long', 'short']"
            unique_decoder_names: List[str] = ['long', 'short']
            columns_tuple = (['P_Long', 'P_Short'])
        ## OUTPUTS: unique_decoder_names, columns_tuple
        


        if known_named_decoding_epochs_type == 'laps':
            # Case: Per-epoch laps marginals
            # Compute the marginals from the result
            epoch_idx_col_name='lap_idx'
            epoch_start_t_col_name='lap_start_t'
            additional_transfer_column_names=['start', 'stop', 'label', 'duration', 'lap_id', 'lap_dir', 'maze_id', 'is_LR_dir']
            
        elif known_named_decoding_epochs_type == 'pbe':
            # Case: Per-epoch PBE/ripple marginals
            epoch_idx_col_name='ripple_idx'
            epoch_start_t_col_name='ripple_start_t'
            additional_transfer_column_names=['start', 'stop', 'label', 'duration']
            
        elif known_named_decoding_epochs_type == 'global':
            # Case: Per-epoch global marginals
            epoch_idx_col_name='epoch_idx'
            epoch_start_t_col_name='epoch_start_t'
            additional_transfer_column_names=['start', 'stop', 'label', 'duration']


        elif known_named_decoding_epochs_type in ('replay', 'ripple', 'non_pbe', 'non_pbe_endcaps'):
            # Case: Per-epoch any marginals
            epoch_idx_col_name='epoch_idx'
            epoch_start_t_col_name='epoch_start_t'
            additional_transfer_column_names=['start', 'stop', 'label', 'duration']
    
        else:
            raise ValueError(f"Unexpected value for known_named_decoding_epochs_type: {known_named_decoding_epochs_type}")

        ## OUTPUTS: epoch_idx_col_name, epoch_start_t_col_name, additional_transfer_column_names

        if data_grain == 'per_time_bin':
            
            additional_column_kwargs = dict(epoch_idx_col_name=epoch_idx_col_name, epoch_start_t_col_name=None, additional_transfer_column_names=None) # set to None because we aren't doing epochs, we're doing time bins , epoch_start_t_col_name=epoch_start_t_col_name, additional_transfer_column_names=additional_transfer_column_names
        
            if (not is_four_tuple_pseudo2D_decoder):
                ## 2-tuple method
                assert is_track_identity_only_pseudo2D_decoder
                # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
                _a_marginal_over_track_IDs_list, a_marginal_over_track_ID_posterior_df = cls.build_generalized_non_marginalized_raw_posteriors(a_result, unique_decoder_names=unique_decoder_names, **additional_column_kwargs) # DirectionalPseudo2DDecodersResult

                ## Build into a marginal df like `all_sessions_laps_df` - uses `time_window_centers`, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID:
                track_marginal_posterior_df : pd.DataFrame = deepcopy(a_marginal_over_track_ID_posterior_df) # pd.DataFrame({'t':deepcopy(time_window_centers), 'P_Long': np.squeeze(non_PBE_marginal_over_track_ID[0, :]), 'P_Short': np.squeeze(non_PBE_marginal_over_track_ID[1, :]), 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size})
                epoch_time_bin_marginals_df = track_marginal_posterior_df
                return epoch_time_bin_marginals_df

            else:
                ## standard (4-tuple) method
                if known_named_decoding_epochs_type == 'laps':
                    # Case: Per-time-bin laps marginals
                    # Compute the marginals from the result
                    (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), _ = a_result.compute_marginals(**additional_column_kwargs)
                    
                    # Expanded implementation of build_laps_time_bin_marginals_df
                    epoch_directional_marginals, _, _, _ = laps_directional_marginals_tuple
                    epoch_track_identity_marginals, _, _, _ = laps_track_identity_marginals_tuple
                    non_marginalized_decoder_marginals, _, _, _ = laps_non_marginalized_decoder_marginals_tuple
                                    
                elif known_named_decoding_epochs_type == 'pbe':
                    # Case: Per-time-bin PBE/ripple marginals
                    # Compute the marginals from the result
                    (ripple_directional_marginals_tuple, ripple_track_identity_marginals_tuple, ripple_non_marginalized_decoder_marginals_tuple), _ = a_result.compute_marginals(
                        epoch_idx_col_name='ripple_idx', 
                        epoch_start_t_col_name='ripple_start_t',
                        additional_transfer_column_names=['start', 'stop', 'label', 'duration']
                    )
                    
                    # Expanded implementation of build_ripple_time_bin_marginals_df
                    epoch_directional_marginals, _, _, _ = ripple_directional_marginals_tuple
                    epoch_track_identity_marginals, _, _, _ = ripple_track_identity_marginals_tuple
                    non_marginalized_decoder_marginals, _, _, _ = ripple_non_marginalized_decoder_marginals_tuple
                    
                elif known_named_decoding_epochs_type == 'global':
                    # Case: Per-time-bin global marginals
                    # Compute the marginals from the result
                    (epoch_directional_marginals_tuple, epoch_track_identity_marginals_tuple, epoch_non_marginalized_decoder_marginals_tuple), _ = a_result.compute_marginals(
                        epoch_idx_col_name='epoch_idx',
                        epoch_start_t_col_name='epoch_start_t',
                        additional_transfer_column_names=['start', 'stop', 'label', 'duration']
                    )
                    
                    # Expanded implementation of build_epoch_time_bin_marginals_df
                    epoch_directional_marginals, _, _, _ = epoch_directional_marginals_tuple
                    epoch_track_identity_marginals, _, _, _ = epoch_track_identity_marginals_tuple
                    non_marginalized_decoder_marginals, _, _, _ = epoch_non_marginalized_decoder_marginals_tuple


                elif known_named_decoding_epochs_type in ('replay', 'ripple', 'non_pbe', 'non_pbe_endcaps'):
                    # Case: Per-time-bin any marginals
                    # Compute the marginals from the result
                    (epoch_directional_marginals_tuple, epoch_track_identity_marginals_tuple, epoch_non_marginalized_decoder_marginals_tuple), _ = a_result.compute_marginals(
                        epoch_idx_col_name=epoch_idx_col_name,
                        epoch_start_t_col_name=epoch_start_t_col_name,
                        additional_transfer_column_names=additional_transfer_column_names
                    )
                    
                    # Expanded implementation of build_epoch_time_bin_marginals_df
                    epoch_directional_marginals, _, _, _ = epoch_directional_marginals_tuple
                    epoch_track_identity_marginals, _, _, _ = epoch_track_identity_marginals_tuple
                    non_marginalized_decoder_marginals, _, _, _ = epoch_non_marginalized_decoder_marginals_tuple
                    
                else:
                    raise ValueError(f"Unexpected value for known_named_decoding_epochs_type: {known_named_decoding_epochs_type}")
                ## END if known_named_decoding...

                # Common (success) marginal code _____________________________________________________________________________________ #
                epoch_time_bin_marginals_df = a_result.build_per_time_bin_marginals_df(
                    active_marginals_tuple=(epoch_directional_marginals, epoch_track_identity_marginals, non_marginalized_decoder_marginals), 
                    columns_tuple=columns_tuple
                )
            ## END if is_track_....
            
            return epoch_time_bin_marginals_df
        

        elif data_grain == 'per_epoch':
            
            additional_column_kwargs = dict(epoch_idx_col_name=epoch_idx_col_name, epoch_start_t_col_name=epoch_start_t_col_name, additional_transfer_column_names=additional_transfer_column_names)
                    
            if (not is_four_tuple_pseudo2D_decoder):
                ## 2-tuple method
                assert is_track_identity_only_pseudo2D_decoder
                # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
                _a_marginal_over_track_IDs_list, a_marginal_over_track_ID_posterior_df = cls.build_generalized_non_marginalized_raw_posteriors(a_result, unique_decoder_names=unique_decoder_names, **additional_column_kwargs) # DirectionalPseudo2DDecodersResult
                track_marginal_posterior_df : pd.DataFrame = deepcopy(a_marginal_over_track_ID_posterior_df)
                epoch_marginals_df = track_marginal_posterior_df
                return epoch_marginals_df
            
            else:
                ## standard (4-tuple) method
                if known_named_decoding_epochs_type == 'laps':
                    # Case: Per-epoch laps marginals
                    # Compute the marginals from the result
                    (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, _), _ = a_result.compute_marginals(**additional_column_kwargs)
                    
                    # Expanded implementation of build_laps_all_epoch_bins_marginals_df
                    epoch_directional_marginals, epoch_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = laps_directional_marginals_tuple
                    epoch_track_identity_marginals, epoch_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = laps_track_identity_marginals_tuple

                    epoch_marginals_df = pd.DataFrame(np.hstack((epoch_directional_all_epoch_bins_marginal, epoch_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
                    epoch_marginals_df['lap_idx'] = epoch_marginals_df.index.to_numpy()
                    epoch_marginals_df['lap_start_t'] = a_result.filter_epochs['start'].to_numpy()
                    ## ensure we have the generic columns too (duplicated):
                    epoch_marginals_df['epoch_idx'] = epoch_marginals_df['lap_idx']
                    epoch_marginals_df['epoch_start_t'] = epoch_marginals_df['lap_start_t']                    


                elif known_named_decoding_epochs_type == 'pbe':
                    # Case: Per-epoch PBE/ripple marginals
                    # Compute the marginals from the result
                    (ripple_directional_marginals_tuple, ripple_track_identity_marginals_tuple, _), _ = a_result.compute_marginals(**additional_column_kwargs)
                    
                    # Expanded implementation of build_ripple_all_epoch_bins_marginals_df
                    epoch_directional_marginals, epoch_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir = ripple_directional_marginals_tuple
                    epoch_track_identity_marginals, epoch_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = ripple_track_identity_marginals_tuple

                    epoch_marginals_df = pd.DataFrame(np.hstack((epoch_directional_all_epoch_bins_marginal, epoch_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
                    epoch_marginals_df['ripple_idx'] = epoch_marginals_df.index.to_numpy()
                    epoch_marginals_df['ripple_start_t'] = a_result.filter_epochs['start'].to_numpy()
                    ## ensure we have the generic columns too (duplicated):
                    epoch_marginals_df['epoch_idx'] = epoch_marginals_df['ripple_idx']
                    epoch_marginals_df['epoch_start_t'] = epoch_marginals_df['ripple_start_t']

                    
                elif known_named_decoding_epochs_type == 'global':
                    # Case: Per-epoch global marginals
                    # Compute the marginals from the result
                    (epoch_directional_marginals_tuple, epoch_track_identity_marginals_tuple, _), _ = a_result.compute_marginals(**additional_column_kwargs)
                    
                    # Expanded implementation of build_epoch_all_epoch_bins_marginals_df
                    epoch_directional_marginals, epoch_directional_all_epoch_bins_marginal, epoch_most_likely_direction_from_decoder, epoch_is_most_likely_direction_LR_dir = epoch_directional_marginals_tuple
                    epoch_track_identity_marginals, epoch_track_identity_all_epoch_bins_marginal, epoch_most_likely_track_identity_from_decoder, epoch_is_most_likely_track_identity_Long = epoch_track_identity_marginals_tuple

                    epoch_marginals_df = pd.DataFrame(np.hstack((epoch_directional_all_epoch_bins_marginal, epoch_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
                    epoch_marginals_df['epoch_idx'] = epoch_marginals_df.index.to_numpy()
                    epoch_marginals_df['epoch_start_t'] = a_result.filter_epochs['start'].to_numpy()
                    
                elif known_named_decoding_epochs_type in ('replay', 'ripple', 'non_pbe', 'non_pbe_endcaps'):
                    # Case: Per-epoch any marginals
                    # Compute the marginals from the result                    
                    overrides_dict = dict(epoch_idx_col_name=epoch_idx_col_name, epoch_start_t_col_name=epoch_start_t_col_name, additional_transfer_column_names=additional_transfer_column_names)
                    
                    (epoch_directional_marginals_tuple, epoch_track_identity_marginals_tuple, _), _ = a_result.compute_marginals(**(additional_column_kwargs | overrides_dict))
                    
                    # Expanded implementation of build_epoch_all_epoch_bins_marginals_df
                    epoch_directional_marginals, epoch_directional_all_epoch_bins_marginal, epoch_most_likely_direction_from_decoder, epoch_is_most_likely_direction_LR_dir = epoch_directional_marginals_tuple
                    epoch_track_identity_marginals, epoch_track_identity_all_epoch_bins_marginal, epoch_most_likely_track_identity_from_decoder, epoch_is_most_likely_track_identity_Long = epoch_track_identity_marginals_tuple

                    epoch_marginals_df = pd.DataFrame(np.hstack((epoch_directional_all_epoch_bins_marginal, epoch_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
                    epoch_marginals_df['epoch_idx'] = epoch_marginals_df.index.to_numpy()
                    epoch_marginals_df['epoch_start_t'] = a_result.filter_epochs['start'].to_numpy()

                else:
                    raise ValueError(f"Unexpected value for known_named_decoding_epochs_type: {known_named_decoding_epochs_type}")
                ## END if known_named_decoding...
            ## END if is_track_....

            # Common (success) marginal code _____________________________________________________________________________________ #
            # epoch_time_bin_marginals_df = a_result.build_per_time_bin_marginals_df(
            #     active_marginals_tuple=(epoch_directional_marginals, epoch_track_identity_marginals, non_marginalized_decoder_marginals), 
            #     columns_tuple=columns_tuple
            # )
            return epoch_marginals_df


        else:
            raise ValueError(f"Unexpected value for data_grain: {data_grain}")



    @classmethod
    def get_proper_p_x_given_n_list(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult]) -> List[DynamicContainer]:
        """ only works for the all-directional coder with the four items
        
        Requires: filter_epochs_decoder_result.p_x_given_n_list
        
        Usage:
           
        """            
        if isinstance(filter_epochs_decoder_result, (List, NDArray)):
            ## assume it's already `p_x_given_n_list` format
            p_x_given_n_list = filter_epochs_decoder_result
        elif (not hasattr(filter_epochs_decoder_result, 'p_x_given_n_list')):
            ## assume it's a SingleEpochDecodedResult and wrap its only entry
            assert hasattr(filter_epochs_decoder_result, 'p_x_given_n'), f"type(filter_epochs_decoder_result): {type(filter_epochs_decoder_result)}, filter_epochs_decoder_result: {filter_epochs_decoder_result}"
            p_x_given_n_list = [filter_epochs_decoder_result.p_x_given_n]

        else:
            ## assume it's in the `DecodedFilterEpochsResult` format, access its `.p_x_given_n_list`
            p_x_given_n_list = filter_epochs_decoder_result.p_x_given_n_list

        return p_x_given_n_list
    

    @classmethod
    def build_top_level_raw_posteriors(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], debug_print=False) -> List[DynamicContainer]:
        """ only works for the all-directional coder with the four items
        
        NOTE OUTPUT WILL BE 3D!!
        
        Requires: filter_epochs_decoder_result.p_x_given_n_list
        """
        
        custom_curr_unit_marginal_list = []
        
        for a_p_x_given_n in filter_epochs_decoder_result.p_x_given_n_list:
            curr_array_shape = np.shape(a_p_x_given_n) # .shape # (62, 4, 236) - (n_pos_bins, 4, n_epoch_t_bins[i])
            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=a_p_x_given_n, most_likely_positions_1D=None)            
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        return custom_curr_unit_marginal_list


    @function_attributes(short_name=None, tags=['ACTIVE', 'general', 'non_PBE'], input_requires=[], output_provides=[], uses=[], used_by=['EpochComputationsComputationsContainer._build_merged_joint_placefields_and_decode', 'DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals', 'EpochComputationsComputationsContainer._build_output_decoded_posteriors'], creation_date='2025-02-20 16:16', related_items=[])
    @classmethod
    def build_generalized_non_marginalized_raw_posteriors(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], unique_decoder_names: List[str], epoch_idx_col_name: Optional[str]=None, epoch_start_t_col_name: Optional[str]=None, additional_transfer_column_names: Optional[List[str]]=None, auto_transfer_all_columns:bool=True, debug_print=False) -> Tuple[List[DynamicContainer], pd.DataFrame]:
        """ works for an all-directional coder with an arbitrary (but specified as `unique_decoder_names`) n_decoders items
        
        Requires: filter_epochs_decoder_result.p_x_given_n_list
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe
            from pyphocorehelpers.indexing_helpers import partition_df_dict, partition_df        
            # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _adding_global_non_PBE_epochs
            # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer, NonPBEDimensionalDecodingResult, Compute_NonPBE_Epochs, KnownFilterEpochs
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import estimate_memory_requirements_bytes
            from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
            from neuropy.analyses.placefields import PfND
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
            from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
            from neuropy.analyses.placefields import PfND
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand, AddNewDecodedEpochMarginal_MatplotlibPlotCommand
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedEpochMarginal_MatplotlibPlotCommand
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers


            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

            ## Unpack from pipeline:
            nonPBE_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
            a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = nonPBE_results.a_new_NonPBE_Epochs_obj
            results1D: NonPBEDimensionalDecodingResult = nonPBE_results.results1D
            results2D: NonPBEDimensionalDecodingResult = nonPBE_results.results2D

            epochs_decoding_time_bin_size = nonPBE_results.epochs_decoding_time_bin_size
            frame_divide_bin_size = nonPBE_results.frame_divide_bin_size


            ## INPUTS: results1D, results1D.continuous_results, a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs

            included_epoch_names: List[str] = ['long', 'short']
            # results1D.pfs
            # results1D.decoders # BasePositionDecoder
            # single_global_epoch_df: pd.DataFrame = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            single_global_epoch: Epoch = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            # PfND

            pfs: Dict[types.DecoderName, PfND] = {k:deepcopy(v) for k, v in results1D.pfs.items() if k in included_epoch_names}
            decoders: Dict[types.DecoderName, BasePositionDecoder] = {k:deepcopy(v) for k, v in results1D.decoders.items() if k in included_epoch_names}
            continuous_decoded_results_dict: Dict[str, DecodedFilterEpochsResult] = {k:deepcopy(v) for k, v in results1D.continuous_results.items() if k in included_epoch_names}
            # DirectionalPseudo2DDecodersResult(

            ## Combine the non-directional PDFs and renormalize to get the directional PDF:
            non_PBE_all_directional_pf1D: PfND = PfND.build_merged_directional_placefields(pfs, debug_print=False)
            non_PBE_all_directional_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(non_PBE_all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
            # non_PBE_all_directional_pf1D_Decoder

            # takes 6.3 seconds
            ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
            pseudo2D_continuous_specific_decoded_results: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)),
                                                                                                                                                                        filter_epochs=deepcopy(single_global_epoch),
                                                                                                                                                                        decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)

                                                                                                                                                                        
            ## OUTPUTS: pseudo2D_continuous_specific_decoded_results, non_PBE_all_directional_pf1D, non_PBE_all_directional_pf1D_Decoder
            # 3.3s

            #@ build_generalized_non_marginalized_raw_posteriors
            # ==================================================================================================================== #
            # Compute Marginals over TrackID                                                                                       #
            # ==================================================================================================================== #
            pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = pseudo2D_continuous_specific_decoded_results
            assert len(pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list) == 1

            unique_decoder_names = ['long', 'short']
            # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
            non_PBE_marginal_over_track_ID, non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(pseudo2D_decoder_continuously_decoded_result, unique_decoder_names=unique_decoder_names)[0]['p_x_given_n']
            # non_marginalized_raw_result = DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']
            # marginal_over_direction = DirectionalPseudo2DDecodersResult.build_custom_marginal_over_direction(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']
            # marginal_over_track_ID = DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']

            time_bin_containers = pseudo2D_decoder_continuously_decoded_result.time_bin_containers[0]
            time_window_centers = time_bin_containers.centers
            # p_x_given_n.shape # (62, 4, 209389)



            ## Main INPUT: continuous_specific_decoded_results_dict
            display_output = {}


            _output_dict = AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_custom_decoder_decoded_epochs(curr_active_pipeline=None, active_2d_plot=active_2d_plot,
                                                                                                                            continuously_decoded_dict=continuous_decoded_results_dict, info_string='non-PBE-pseudo2D marginals', # results1D.continuous_results
                                                                                                                            xbin=deepcopy(results1D.decoders['global'].xbin), debug_print=False)



                                                                                                                            

            # ==================================================================================================================== #
            # Plot the Decodings and their Marginals over TrackID as new Tracks                                                    #
            # ==================================================================================================================== #

            ## Manually call `AddNewDecodedEpochMarginal_MatplotlibPlotCommand` to add the custom marginals track to the active SpikeRaster3DWindow
            # list(curr_active_pipeline.display_output.keys())

            # active_display_fn_identifying_ctx = IdentifyingContext(format_name= 'kdiba', animal= 'gor01', exper_name= 'one', session_name= '2006-6-09_1-22-43', filter_name= 'maze_any', lap_dir= 'any', display_fn_name= 'display_spike_rasters_window')
            # active_display_fn_identifying_ctx = IdentifyingContext(format_name= 'kdiba', animal= 'gor01', exper_name= 'one', session_name= '2006-6-09_1-22-43', filter_name= 'maze_any', lap_dir= 'any', display_fn_name= '_display_spike_rasters_pyqtplot_2D')
            # display_output = curr_active_pipeline.display_output[active_display_fn_identifying_ctx]
            # display_output
            # active_config_name: str = 'maze_any'
            ## INPUTS: active_config_name, active_display_fn_identifying_ctx, display_output


            # output_references = _build_additional_window_menus(spike_raster_window, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx)
            # _docked_menu_provider.DockedWidgets_MenuProvider_on_buildUI(spike_raster_window=spike_raster_window, owning_pipeline_reference=owning_pipeline_reference, context=active_display_fn_identifying_ctx, active_config_name=active_config_name, display_output=owning_pipeline_reference.display_output[active_display_fn_identifying_ctx])

            # _cmd = AddNewDecodedEpochMarginal_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline,
            #                                                          active_config_name=active_config_name, active_context=active_display_fn_identifying_ctx, display_output=display_output, action_identifier='actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView')
            # _cmd

            # ## To begin, the destination plot must have a matplotlib widget plot to render to:
            # # print(f'AddNewDecodedEpochMarginal_MatplotlibPlotCommand.execute(...)')
            # active_2d_plot = _cmd._spike_raster_window.spike_raster_plt_2d
            # enable_rows_config_kwargs = dict(enable_non_marginalized_raw_result=_cmd.enable_non_marginalized_raw_result, enable_marginal_over_direction=_cmd.enable_marginal_over_direction, enable_marginal_over_track_ID=_cmd.enable_marginal_over_track_ID)

            # # output_dict = self.add_pseudo2D_decoder_decoded_epoch_marginals(self._active_pipeline, active_2d_plot, **enable_rows_config_kwargs)
            # output_dict = AddNewDecodedEpochMarginal_MatplotlibPlotCommand.add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epoch_marginals(_cmd._active_pipeline, active_2d_plot, **enable_rows_config_kwargs)

            time_bin_size = epochs_decoding_time_bin_size
            info_string: str = f" - t_bin_size: {time_bin_size}"

            dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)

            output_dict = {}
            a_posterior_name: str = 'non-PBE_marginal_over_track_ID'
            assert non_PBE_marginal_over_track_ID.shape[0] == 2, f"expected the 2 marginalized pseudo-y bins for the decoder in non_PBE_marginal_over_track_ID.shape[1]. but found non_PBE_marginal_over_track_ID.shape: {non_PBE_marginal_over_track_ID.shape}"
            output_dict[a_posterior_name] = AddNewDecodedEpochMarginal_MatplotlibPlotCommand._perform_add_new_decoded_posterior_marginal_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=dock_config,
                                                                                                a_variable_name=a_posterior_name, xbin=np.arange(2), time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)

            ## Draw the "Long", "Short" labels
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
            label_artists_dict = {}
            for i, ax in enumerate(matplotlib_fig_axes):
                label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long', 'short'], enable_draw_decoder_colored_lines=False)
            output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)


        0: LR
        1: RL
        
        """
        from neuropy.utils.misc import capitalize_after_underscore

        p_x_given_n_list = cls.get_proper_p_x_given_n_list(filter_epochs_decoder_result)
        n_decoders: int = len(unique_decoder_names)
        
        custom_curr_unit_marginal_list: List[DynamicContainer] = []
        
        for epoch_i, a_p_x_given_n in enumerate(p_x_given_n_list):
            # an_array = all_directional_laps_filter_epochs_decoder_result.p_x_given_n_list[0] # .shape # (62, 4, 236)
            curr_array_shape = np.shape(a_p_x_given_n) # .shape # (62, 4, 236) - (n_pos_bins, 4, n_epoch_t_bins[i])
            if debug_print:
                print(f'p_x_given_n_list[{epoch_i}]:')
                print(f'\ta_p_x_given_n.shape: {curr_array_shape}')

            if debug_print:
                print(f'\tnp.shape(a_p_x_given_n): {np.shape(curr_array_shape)}')
                
            assert curr_array_shape[1] == n_decoders, f"only works when curr_array_shape[1]: {curr_array_shape[1]} correspond to the unique_decoder_names: {unique_decoder_names}. (typically all-directional decoder)."                
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=a_p_x_given_n, most_likely_positions_1D=None)
            
            if debug_print:
                print(f'\tnp.shape(curr_unit_posterior_list.p_x_given_n): {np.shape(curr_unit_marginal_x.p_x_given_n)}')
            
            # y-axis marginal:
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(a_p_x_given_n, axis=0)) # sum over all x. Result should be (n_decoders, n_epoch_t_bins[i])
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0, keepdims=True) # sum over all four decoders for each time_bin (so there's a normalized distribution at each timestep)

            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_x.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        ## END for epoch_i, a_p_x_given_...
        
        # ## INPUTS: custom_curr_unit_marginal_list
        
        ## All
        epochs_repeated_epoch_index: NDArray = np.hstack([np.full((n_bins, ), i) for i, n_bins in enumerate(filter_epochs_decoder_result.nbins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
        epochs_repeated_sub_epoch_time_bin_index: NDArray = np.hstack([np.arange(n_bins) for n_bins in filter_epochs_decoder_result.nbins]) # np.shape(epochs_t_centers) # (19018,)
        
        ## Redundant but very helpful parent_epoch_* columns
        epochs_repeated_parent_epoch_start_t: NDArray = np.hstack([np.full((n_bins, ), filter_epochs_decoder_result.time_bin_containers[i].edge_info.variable_extents[0]) for i, n_bins in enumerate(filter_epochs_decoder_result.nbins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
        epochs_repeated_parent_epoch_stop_t: NDArray = np.hstack([np.full((n_bins, ), filter_epochs_decoder_result.time_bin_containers[i].edge_info.variable_extents[-1]) for i, n_bins in enumerate(filter_epochs_decoder_result.nbins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)

        # a_masked_decoded_result.time_bin_containers[0].edge_info.variable_extents

        ## Non-masked
        epochs_p_x_given_n: NDArray = np.hstack([v['p_x_given_n'] for v in custom_curr_unit_marginal_list]) # np.shape(epochs_p_x_given_n) # (2, 19018)
        epochs_t_centers: NDArray = np.hstack([t.centers for t in filter_epochs_decoder_result.time_bin_containers]) # np.shape(epochs_t_centers) # (19018,)
        assert np.shape(epochs_p_x_given_n)[-1] == np.shape(epochs_t_centers)[0]

        ## OUTPUTS: epochs_repeated_epoch_index, epochs_repeated_sub_epoch_time_bin_index, epochs_t_centers, epochs_p_x_given_n
        # _common_epoch_df_dict = {'epoch_id': epochs_repeated_epoch_index, 'sub_epoch_time_bin_index': epochs_repeated_sub_epoch_time_bin_index}
        _common_epoch_df_dict = {'epoch_id': epochs_repeated_epoch_index, 'sub_epoch_time_bin_index': epochs_repeated_sub_epoch_time_bin_index}
        _common_epoch_df_dict.update(**{'parent_epoch_id': epochs_repeated_epoch_index, 'parent_epoch_start_t': epochs_repeated_parent_epoch_start_t, 'parent_epoch_end_t': epochs_repeated_parent_epoch_stop_t})
        
        assert np.shape(epochs_p_x_given_n)[0] == len(unique_decoder_names), f"unique_decoder_names: {unique_decoder_names} len(unique_decoder_names) != np.shape(epochs_p_x_given_n)[0]: {np.shape(epochs_p_x_given_n)[0]}"

        # _marginal_prob_var_names = [f"P_{v.capitalize()}" for v in unique_decoder_names] # ['P_Long', 'P_Short']
        _marginal_prob_var_names = [f"P_{capitalize_after_underscore(v, should_capitalize_start=True)}" for v in unique_decoder_names] # ['P_Long', 'P_Short']
        

        _marginal_probs_dict = {a_marginal_prob_col_name:np.squeeze(epochs_p_x_given_n[i, :]) for i, (a_marginal_prob_col_name, a_decoder_name) in enumerate(zip(_marginal_prob_var_names, unique_decoder_names))} # 'P_Long': np.squeeze(epochs_p_x_given_n[0, :]), 'P_Short': np.squeeze(epochs_p_x_given_n[1, :]),
        if debug_print:
            print(f'_marginal_probs_dict: {_marginal_probs_dict}')
        
        ## Build into a marginal df like `all_sessions_epochs_df`:
        track_marginal_posterior_df: pd.DataFrame = pd.DataFrame({'t':deepcopy(epochs_t_centers),
                                                                    **_marginal_probs_dict,
                                                                    # 'P_Long': np.squeeze(epochs_p_x_given_n[0, :]), 'P_Short': np.squeeze(epochs_p_x_given_n[1, :]),
                                                                     **deepcopy(_common_epoch_df_dict)}) # , 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size
        ## OUTPUTS: track_marginal_posterior_df

        ## Build combined marginals df:
        #TODO 2025-03-26 14:27: - [ ] NOTE: the `filter_epochs_decoder_result.filter_epochs` do not match for time_bin grain! They instead only show the epochs!
        tentative_epochs_df: pd.DataFrame = ensure_dataframe(deepcopy(filter_epochs_decoder_result.filter_epochs)) ## get the base filter epochs dataframe
        # epochs_marginals_df = pd.DataFrame(np.hstack((epochs_directional_all_epoch_bins_marginal, epochs_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        # epochs_marginals_df = pd.DataFrame(np.hstack((non_marginalized_decoder_all_epoch_bins_marginal, epochs_directional_all_epoch_bins_marginal, epochs_track_identity_all_epoch_bins_marginal)), columns=['long_LR', 'long_RL', 'short_LR', 'short_RL', 'P_LR', 'P_RL', 'P_Long', 'P_Short'])
        if epoch_idx_col_name is not None:
            track_marginal_posterior_df[epoch_idx_col_name] = track_marginal_posterior_df.index.to_numpy()
            
        if len(tentative_epochs_df) == len(track_marginal_posterior_df):
            if epoch_start_t_col_name is not None:
                assert 'start' in tentative_epochs_df
                track_marginal_posterior_df[epoch_start_t_col_name] = tentative_epochs_df['start'].to_numpy()
            # epochs_marginals_df[epoch_start_t_col_name] = epochs_df['start'].to_numpy()
            # epochs_marginals_df['stop'] = epochs_epochs_df['stop'].to_numpy()
            # epochs_marginals_df['label'] = epochs_epochs_df['label'].to_numpy()
            if auto_transfer_all_columns and (additional_transfer_column_names is None):
                ## if no columns are explcitly specified, transfer all columns
                additional_transfer_column_names = list(tentative_epochs_df.columns)
                
            if additional_transfer_column_names is not None:
                for a_col_name in additional_transfer_column_names:
                    if ((a_col_name in tentative_epochs_df) and (a_col_name not in track_marginal_posterior_df)):
                        track_marginal_posterior_df[a_col_name] = tentative_epochs_df[a_col_name].to_numpy()
                    else:
                        print(f'\tWARN: extra column: "{a_col_name}" was specified but not present in epochs_df (which has columns: {list(tentative_epochs_df.columns)}). Skipping.')
                     
        else:
            if (epoch_start_t_col_name is not None) or (additional_transfer_column_names is not None):
                print(f'\tWARN: len(epochs_df): {len(tentative_epochs_df)} != len(track_marginal_posterior_df): {len(track_marginal_posterior_df)}. This is expected in time_bin grain (see TODO 2025-03-26 14:27), but you probably called this with the wrong arguments!')

        return custom_curr_unit_marginal_list, track_marginal_posterior_df


    @function_attributes(short_name=None, tags=['marginal'], input_requires=[], output_provides=[], uses=[], used_by=['cls.determine_non_marginalized_decoder_likelihoods', 'AddNewDecodedEpochMarginal_MatplotlibPlotCommand.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals', '_display_directional_merged_pf_decoded_epochs'], creation_date='2025-05-02 17:29', related_items=[])
    @classmethod
    def build_non_marginalized_raw_posteriors(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], debug_print=False) -> List[DynamicContainer]:
        """ only works for the all-directional coder with the four items
        
        Requires: filter_epochs_decoder_result.p_x_given_n_list
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            active_decoder = all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: build_custom_marginal_over_direction(filter_epochs_decoder_result),
                                                        )
                                    
                                                        
        0: LR
        1: RL
        
        """
        p_x_given_n_list = cls.get_proper_p_x_given_n_list(filter_epochs_decoder_result)
        
        custom_curr_unit_marginal_list = []
        
        for a_p_x_given_n in p_x_given_n_list:
            # an_array = all_directional_laps_filter_epochs_decoder_result.p_x_given_n_list[0] # .shape # (62, 4, 236)
            curr_array_shape = np.shape(a_p_x_given_n) # .shape # (62, 4, 236) - (n_pos_bins, 4, n_epoch_t_bins[i])
            if debug_print:
                print(f'a_p_x_given_n.shape: {curr_array_shape}')

            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            if debug_print:
                print(f'np.shape(a_p_x_given_n): {np.shape(curr_array_shape)}')
                
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=a_p_x_given_n, most_likely_positions_1D=None)
            
            if debug_print:
                print(f'np.shape(curr_unit_posterior_list.p_x_given_n): {np.shape(curr_unit_marginal_x.p_x_given_n)}')
            
            # y-axis marginal:
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(a_p_x_given_n, axis=0)) # sum over all x. Result should be (4, n_epoch_t_bins[i])
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0, keepdims=True) # sum over all four decoders for each time_bin (so there's a normalized distribution at each timestep)

            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_x.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        return custom_curr_unit_marginal_list

    @function_attributes(short_name=None, tags=['marginal'], input_requires=[], output_provides=[], uses=[], used_by=['determine_directional_likelihoods'], creation_date='2025-05-02 17:30', related_items=[])
    @classmethod
    def build_custom_marginal_over_direction(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], debug_print=False) -> List[DynamicContainer]:
        """ only works for the all-directional coder with the four items
        
        Requires: filter_epochs_decoder_result.p_x_given_n_list

        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            active_decoder = all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: build_custom_marginal_over_direction(filter_epochs_decoder_result),
                                                        )
                                    
                                                        
        0: LR
        1: RL
        
        """
        p_x_given_n_list = cls.get_proper_p_x_given_n_list(filter_epochs_decoder_result)
        
        custom_curr_unit_marginal_list = []
        
        ## Make sure it is the Pseudo2D (all-directional) decoder by checking shape:
        p_x_given_n_list_second_dim_sizes = np.array([np.shape(a_p_x_given_n)[1] for a_p_x_given_n in p_x_given_n_list])
        is_pseudo2D_decoder = np.all((p_x_given_n_list_second_dim_sizes == 4)) # only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        if not is_pseudo2D_decoder:
            raise ValueError(f"this only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL'] but p_x_given_n_list_second_dim_sizes: {p_x_given_n_list_second_dim_sizes} ")
        
        for a_p_x_given_n in p_x_given_n_list:
            # an_array = all_directional_laps_filter_epochs_decoder_result.p_x_given_n_list[0] # .shape # (62, 4, 236)
            curr_array_shape = np.shape(a_p_x_given_n)
            if debug_print:
                print(f'a_p_x_given_n.shape: {curr_array_shape}')

            assert curr_array_shape[1] == 4, f"curr_array_shape: {curr_array_shape} but this only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            ## I don't want to use np.nansum(...) here because I DO want to propgate NaNs if one of the elements are NaN:
            out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]) # LR_marginal = long_LR + short_LR
            out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]) # RL_marginal = long_RL + short_RL

            normalized_out_p_x_given_n = out_p_x_given_n

            input_array = normalized_out_p_x_given_n

            if debug_print:
                print(f'np.shape(input_array): {np.shape(input_array)}')
            # custom marginal over long/short, leaving only LR/RL:
            curr_unit_marginal_y = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_y.p_x_given_n = input_array
            
            # y-axis marginal:
            curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=0)) # sum over all x. Result should be [y_bins x time_bins]

            curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

            if debug_print:
                print(f'np.shape(curr_unit_marginal_y.p_x_given_n): {np.shape(curr_unit_marginal_y.p_x_given_n)}')
            
            ## Ensures that the marginal posterior is at least 2D:
            # print(f"curr_unit_marginal_y.p_x_given_n.ndim: {curr_unit_marginal_y.p_x_given_n.ndim}")
            # assert curr_unit_marginal_y.p_x_given_n.ndim >= 2
            if curr_unit_marginal_y.p_x_given_n.ndim == 0:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_y.p_x_given_n.ndim == 1:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_y.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_y)
        return custom_curr_unit_marginal_list


    @function_attributes(short_name=None, tags=['marginal'], input_requires=[], output_provides=[], uses=[], used_by=['determine_long_short_likelihoods', 'prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals'], creation_date='2025-05-02 17:30', related_items=[])
    @classmethod
    def build_custom_marginal_over_long_short(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], debug_print=False) -> List[DynamicContainer]:
        """ only works for the all-directional coder with the four items
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            active_decoder = all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                                                        )
                                    
                                                        
        0: LR
        1: RL
        
        """
        p_x_given_n_list = cls.get_proper_p_x_given_n_list(filter_epochs_decoder_result)
        
        custom_curr_unit_marginal_list = []
        
        for a_p_x_given_n in p_x_given_n_list:
            # an_array = all_directional_laps_filter_epochs_decoder_result.p_x_given_n_list[0] # .shape # (62, 4, 236)
            curr_array_shape = np.shape(a_p_x_given_n)
            if debug_print:
                print(f'a_p_x_given_n.shape: {curr_array_shape}')
            # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            # (['long', 'long', 'short', 'short'])
            # (n_neurons, is_long, is_LR, pos_bins)
            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            # out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            # out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]) # LR_marginal = long_LR + short_LR
            # out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]) # RL_marginal = long_RL + short_RL

            # Extract the Long/Short items
            out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 1, :]) # Long_marginal = long_LR + long_RL 
            out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 2, :] + a_p_x_given_n[:, 3, :]) # Short_marginal = short_LR + short_RL
            # normalized_out_p_x_given_n = out_p_x_given_n / np.sum(out_p_x_given_n, axis=1) # , keepdims=True
            normalized_out_p_x_given_n = out_p_x_given_n
            # reshaped_p_x_given_n = np.reshape(a_p_x_given_n, (curr_array_shape[0], 2, 2, curr_array_shape[-1]))
            # assert np.array_equiv(reshaped_p_x_given_n[:,0,0,:], a_p_x_given_n[:, 0, :]) # long_LR
            # assert np.array_equiv(reshaped_p_x_given_n[:,1,0,:], a_p_x_given_n[:, 2, :]) # short_LR

            # print(f'np.shape(reshaped_p_x_given_n): {np.shape(reshaped_p_x_given_n)}')

            # input_array = a_p_x_given_n
            # input_array = normalized_reshaped_p_x_given_n
            input_array = normalized_out_p_x_given_n

            if debug_print:
                print(f'np.shape(input_array): {np.shape(input_array)}')
            # custom marginal over long/short, leaving only LR/RL:
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_x.p_x_given_n = input_array
            
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, 1)) # sum over all y. Result should be [x_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        
            # y-axis marginal:
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(input_array, axis=0)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=1, keepdims=True) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=1)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            if debug_print:
                print(f'np.shape(curr_unit_marginal_y.p_x_given_n): {np.shape(curr_unit_marginal_x.p_x_given_n)}')
            
            ## Ensures that the marginal posterior is at least 2D:
            # print(f"curr_unit_marginal_y.p_x_given_n.ndim: {curr_unit_marginal_y.p_x_given_n.ndim}")
            # assert curr_unit_marginal_y.p_x_given_n.ndim >= 2
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_x.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        return custom_curr_unit_marginal_list
        
    # Higher-level likelihood access functions ___________________________________________________________________________ #
    @function_attributes(short_name=None, tags=['marginal', 'MAIN'], input_requires=[], output_provides=[], uses=['cls.build_custom_marginal_over_direction'], used_by=['DecodedFilterEpochsResult.perform_compute_marginals'], creation_date='2025-04-01 00:00', related_items=[])
    @classmethod
    def determine_directional_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult]) -> DecodedMarginalResultTuple:
        """ 

        determine_directional_likelihoods

        directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decode, is_most_likely_direction_LR_dir = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)

        0: LR
        1: RL
        
        """
        directional_marginals: List[DynamicContainer] = cls.build_custom_marginal_over_direction(all_directional_laps_filter_epochs_decoder_result)
        
        # gives the likelihood of [LR, RL] for each epoch using information from both Long/Short:
        directional_all_epoch_bins_marginal = np.stack([np.sum(v.p_x_given_n, axis=-1)/np.sum(v.p_x_given_n, axis=(-2, -1)) for v in directional_marginals], axis=0) # sum over all time-bins within the epoch to reach a consensus
        
        # Compute the likelihood of [LR, RL] for each epoch using information from both Long/Short
        # directional_all_epoch_bins_marginal = np.stack([
        #     np.nansum(v.p_x_given_n, axis=-1) / np.maximum(np.nansum(v.p_x_given_n, axis=(-2, -1)), _prevent_div_by_zero_epsilon)
        #     for v in directional_marginals
        # ], axis=0)  # sum over all time-bins within the epoch to reach a consensus

        # Find the indicies via this method:
        most_likely_direction_from_decoder = np.argmax(directional_all_epoch_bins_marginal, axis=1) # consistent with 'lap_dir' columns. for LR_dir, values become more positive with time
        is_most_likely_direction_LR_dir = np.logical_not(most_likely_direction_from_decoder) # consistent with 'is_LR_dir' column. for LR_dir, values become more positive with time
        return directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decoder, is_most_likely_direction_LR_dir
    
    @function_attributes(short_name=None, tags=['marginal', 'MAIN'], input_requires=[], output_provides=[], uses=['cls.build_custom_marginal_over_direction'], used_by=['DecodedFilterEpochsResult.perform_compute_marginals'], creation_date='2025-04-01 00:00', related_items=[])
    @classmethod
    def determine_long_short_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult]) -> DecodedMarginalResultTuple:
        """ 
        
        laps_track_identity_marginals = DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
        track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long = laps_track_identity_marginals
        
        0: Long
        1: Short
        
        """
        track_identity_marginals: List[DynamicContainer] = cls.build_custom_marginal_over_long_short(all_directional_laps_filter_epochs_decoder_result)
        
        # gives the likelihood of [LR, RL] for each epoch using information from both Long/Short:
        track_identity_all_epoch_bins_marginal = np.stack([np.sum(v.p_x_given_n, axis=-1)/np.sum(v.p_x_given_n, axis=(-2, -1)) for v in track_identity_marginals], axis=0) # sum over all time-bins within the epoch to reach a consensus

        # Find the indicies via this method:
        most_likely_track_identity_from_decoder = np.argmax(track_identity_all_epoch_bins_marginal, axis=1) # consistent with 'lap_dir' columns. for LR_dir, values become more positive with time
        is_most_likely_track_identity_Long = np.logical_not(most_likely_track_identity_from_decoder) # consistent with 'is_LR_dir' column. for LR_dir, values become more positive with time
        return track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long


    @function_attributes(short_name=None, tags=['marginal', 'MAIN'], input_requires=[], output_provides=[], uses=['cls.build_custom_marginal_over_direction'], used_by=['DecodedFilterEpochsResult.perform_compute_marginals'], creation_date='2025-04-01 00:00', related_items=[])
    @classmethod
    def determine_non_marginalized_decoder_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, DecodedFilterEpochsResult], debug_print=False) -> Tuple[List[DynamicContainer], NDArray[float], NDArray[int], pd.DataFrame]:
        """ 
        
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = DirectionalPseudo2DDecodersResult.determine_non_marginalized_decoder_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
        
        
        [(4, 47), (4, 31), (4, 33), ]
        
        """
        non_marginalized_decoder_marginals: List[DynamicContainer] = cls.build_non_marginalized_raw_posteriors(all_directional_laps_filter_epochs_decoder_result, debug_print=debug_print) # each P-x[i].shape: (4, n_epoch_t_bins[i])
        # gives the likelihood of [LR, RL] for each epoch using information from both Long/Short:
        non_marginalized_decoder_all_epoch_bins_marginal = np.stack([np.sum(v.p_x_given_n, axis=-1)/np.sum(v.p_x_given_n, axis=(-2, -1)) for v in non_marginalized_decoder_marginals], axis=0) # sum over all time-bins within the epoch to reach a consensus .shape: (4, )

        # non_marginalized_decoder_all_epoch_bins_marginal.shape: [n_epochs, 4]
        most_likely_decoder_idxs = np.argmax(non_marginalized_decoder_all_epoch_bins_marginal, axis=1) # consistent with 'lap_dir' columns. for LR_dir, values become more positive with time | most_likely_decoder_idxs will always contain valid indices, not NaN.
        # build the dataframe:
        assert (np.shape(non_marginalized_decoder_all_epoch_bins_marginal)[1] == 4), f"shape of non_marginalized_decoder_all_epoch_bins_marginal must be 4 (corresponding to the 4 decoders) but instead np.shape(non_marginalized_decoder_all_epoch_bins_marginal): {np.shape(non_marginalized_decoder_all_epoch_bins_marginal)}"
        non_marginalized_decoder_all_epoch_bins_decoder_probs_df: pd.DataFrame = pd.DataFrame(non_marginalized_decoder_all_epoch_bins_marginal, columns=['long_LR', 'long_RL', 'short_LR', 'short_RL'])
        return non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df



    @function_attributes(short_name=None, tags=['ground-truth', 'laps', 'validation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-05 19:23', related_items=['add_groundtruth_information()'])
    @classmethod
    def validate_lap_dir_estimations(cls, global_session, active_global_laps_df, laps_is_most_likely_direction_LR_dir):
        """ validates the lap direction and track estimations. 
        """
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        active_global_laps_df = Laps._compute_lap_dir_from_smoothed_velocity(active_global_laps_df, global_session=global_session, replace_existing=False) ## NOTE: global_session does not work, use curr_active_pipeline.sess instead (unfiltered session) otherwise it clips the last two laps

        # Validate Laps:
        # ground_truth_lap_dirs = active_global_laps_df['lap_dir'].to_numpy()
        ground_truth_lap_is_LR_dir = active_global_laps_df['is_LR_dir'].to_numpy()
        n_laps = np.shape(active_global_laps_df)[0]
        assert len(laps_is_most_likely_direction_LR_dir) == n_laps
        percent_laps_estimated_correctly = (np.sum(ground_truth_lap_is_LR_dir == laps_is_most_likely_direction_LR_dir) / n_laps)
        print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        return percent_laps_estimated_correctly

    
    @function_attributes(short_name=None, tags=['export', 'csv', 'marginal', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-08 19:01', related_items=[])
    def compute_and_export_marginals_df_csvs(self, parent_output_path: Path, active_context: IdentifyingContext):
        """ Builds the four dataframes from the marginal distributions and exports them to .csv files.
        
        active_context = curr_active_pipeline.get_session_context()
        """
        output_date_str: str = get_now_rounded_time_str()

        # Export CSVs:
        def export_marginals_df_csv(marginals_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)'):
            """ captures `active_context`, `parent_output_path`. 'output_date_str'
            """
            # parent_output_path: Path = Path('output').resolve()
            # active_context = curr_active_pipeline.get_session_context()
            session_identifier_str: str = active_context.get_description()
            assert output_date_str is not None
            out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-01-04|kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
            out_filename = f"{out_basename}.csv"
            out_path = parent_output_path.joinpath(out_filename).resolve()
            marginals_df.to_csv(out_path)
            return out_path 


        ## Laps:
        laps_epochs_df = ensure_dataframe(deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs))        
        # laps_directional_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        # laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
        # laps_track_identity_marginals = DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        # track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long = laps_track_identity_marginals
        laps_decoding_time_bin_size_str: str = f"{round(self.laps_decoding_time_bin_size, ndigits=5)}"
        
        ## Build the per-time-bin results:
        # laps_time_bin_marginals_df: pd.DataFrame = self._build_per_time_bin_marginals(a_decoder_result=self.all_directional_laps_filter_epochs_decoder_result, active_marginals=track_identity_marginals)
        laps_time_bin_marginals_df: pd.DataFrame = self.laps_time_bin_marginals_df.copy()
        laps_time_bin_marginals_out_path = export_marginals_df_csv(laps_time_bin_marginals_df, data_identifier_str=f'(laps_time_bin_marginals_df)_tbin-{laps_decoding_time_bin_size_str}')

        # laps_marginals_df: pd.DataFrame = pd.DataFrame(np.hstack((laps_directional_all_epoch_bins_marginal, track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        # laps_marginals_df['lap_idx'] = laps_marginals_df.index.to_numpy()
        # laps_marginals_df['lap_start_t'] = laps_epochs_df['start'].to_numpy()
        # laps_marginals_df

        (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = self.all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                            additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
        
        # epoch_extracted_posteriors = [a_result['p_x_given_n'] for a_result in track_identity_marginals]
        # n_epoch_time_bins = [np.shape(a_posterior)[-1] for a_posterior in epoch_extracted_posteriors]
        # epoch_idx_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=i) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)])
        # epoch_time_bin_marginals_df = pd.DataFrame(np.hstack((epoch_extracted_posteriors)).T, columns=['P_Long', 'P_Short'])
        # epoch_time_bin_marginals_df['epoch_idx'] = epoch_idx_column
        # epoch_time_bin_marginals_df['t_bin_center'] = flat_time_bin_centers_column
        
        laps_out_path = export_marginals_df_csv(laps_marginals_df, data_identifier_str=f'(laps_marginals_df)_tbin-{laps_decoding_time_bin_size_str}')

        ## Ripples:
        ripple_epochs_df = deepcopy(self.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
        all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = self.all_directional_ripple_filter_epochs_decoder_result
        # ripple_marginals = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(all_directional_ripple_filter_epochs_decoder_result)
        # ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = ripple_marginals
        # ripple_track_identity_marginals = DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods(all_directional_ripple_filter_epochs_decoder_result)
        # ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = ripple_track_identity_marginals
        ripple_decoding_time_bin_size_str: str = f"{round(self.ripple_decoding_time_bin_size, ndigits=5)}"

        ## Build the per-time-bin results:
        # ripple_time_bin_marginals_df: pd.DataFrame = self._build_per_time_bin_marginals(a_decoder_result=self.all_directional_ripple_filter_epochs_decoder_result, active_marginals=ripple_directional_marginals)
        ripple_time_bin_marginals_df: pd.DataFrame = self.ripple_time_bin_marginals_df.copy()
        ripple_time_bin_marginals_out_path = export_marginals_df_csv(ripple_time_bin_marginals_df, data_identifier_str=f'(ripple_time_bin_marginals_df)_tbin-{ripple_decoding_time_bin_size_str}')


        ## Ripple marginals_df:
        # ripple_marginals_df: pd.DataFrame = pd.DataFrame(np.hstack((ripple_directional_all_epoch_bins_marginal, ripple_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        # ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
        # ripple_marginals_df['ripple_start_t'] = ripple_epochs_df['start'].to_numpy()
        # ripple_marginals_df

        ## INPUTS: all_directional_ripple_filter_epochs_decoder_result


        (ripple_directional_marginals_tuple, ripple_track_identity_marginals_tuple, ripple_non_marginalized_decoder_marginals_tuple), ripple_marginals_df = all_directional_ripple_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='ripple_idx', epoch_start_t_col_name='ripple_start_t',
                                                                                                                                                                    additional_transfer_column_names=['start','stop','label','duration','ripple_idx','maze_id']) # ,'lap_dir','is_LR_dir',
        ## OUTPUT: ripple_marginals_df
        
        ripple_out_path = export_marginals_df_csv(ripple_marginals_df, data_identifier_str=f'(ripple_marginals_df)_tbin-{ripple_decoding_time_bin_size_str}')

        return (laps_marginals_df, laps_out_path, laps_time_bin_marginals_df, laps_time_bin_marginals_out_path), (ripple_marginals_df, ripple_out_path, ripple_time_bin_marginals_df, ripple_time_bin_marginals_out_path)

    @function_attributes(short_name=None, tags=['correlation', 'simple_corr', 'spike-times-v-pf-peak-x'], input_requires=[], output_provides=[], uses=['_perform_compute_simple_spike_time_v_pf_peak_x_by_epoch'], used_by=[], creation_date='2024-02-15 18:29', related_items=[])
    def compute_simple_spike_time_v_pf_peak_x_by_epoch(self, track_templates: TrackTemplates, spikes_df: pd.DataFrame) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], List[str]]:
        """ 
        Updates the .filter_epochs property on both the laps and the ripples objects
        adds columns: ['long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr', 'best_decoder_index']
        """
        best_decoder_index_col_name: str = 'best_decoder_index' # _pearsonr
        all_directional_laps_filter_epochs_decoder_result_value = self.all_directional_laps_filter_epochs_decoder_result
        all_directional_ripple_filter_epochs_decoder_result_value = self.all_directional_ripple_filter_epochs_decoder_result

        ## Drop endcap aclus:
        # excluded_endcap_aclus: NDArray = np.array([ 6,  14,  15,  19,  28,  33,  40,  43,  51,  52,  53,  59,  60,  67,  71,  83,  84,  85,  87,  95, 101, 102]) # #TODO 2024-02-22 13:25: - [ ] Hardcoded values for IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')
        spikes_df: pd.DataFrame = deepcopy(spikes_df)
        # unique_aclus = spikes_df.aclu.unique()
        # included_aclus = np.array([aclu for aclu in unique_aclus if (aclu not in excluded_endcap_aclus)]) 
        # spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_aclus)
        # spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

        ## TODO: Now filter included epochs by num_unique_aclus

        for an_epochs_result in (all_directional_laps_filter_epochs_decoder_result_value, all_directional_ripple_filter_epochs_decoder_result_value):
            # spikes_df = deepcopy(curr_active_pipeline.global_computation_results.computed_data['RankOrder'].LR_ripple.selected_spikes_df)
            active_epochs_df: pd.DataFrame = deepcopy(an_epochs_result.filter_epochs)
            corr_df, corr_column_names = self._perform_compute_simple_spike_time_v_pf_peak_x_by_epoch(track_templates=track_templates, spikes_df=spikes_df, active_epochs_df=active_epochs_df, epoch_label_column_name='label') # corr_column_names: ['long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr']
            # corr_df.dropna(subset=corr_column_names)
            # corr_df['n_participating_aclus'] = 

            # Replace None with NaN
            corr_df = corr_df.where(pd.notnull(corr_df), np.nan)
            ## Join the correlations result into the active_epochs_df:
            active_epochs_df = Epoch(active_epochs_df).to_dataframe()
            
            if len(corr_df) > 0:
                ## non-empty corr_df:
                active_epochs_df = active_epochs_df.drop(columns=corr_column_names, errors='ignore', inplace=False) # drop existing columns so they can be replaced
                active_epochs_df = active_epochs_df.join(corr_df)
                active_epochs_df[best_decoder_index_col_name] = active_epochs_df[corr_column_names].fillna(0.0).abs().apply(lambda row: np.argmax(row.values), axis=1) # Computes the highest-valued decoder for this score. Note `.abs()` is important here to consider both directions.'
            else:
                print(f'WARN: corr_df is empty in compute_simple_spike_time_v_pf_peak_x_by_epoch(....). Continuing without adding best_decoder_index_col_name: {best_decoder_index_col_name} or corr_column_names: {corr_column_names}.')


            if isinstance(an_epochs_result.filter_epochs, pd.DataFrame):
                an_epochs_result.filter_epochs = active_epochs_df
            else:
                an_epochs_result.filter_epochs = Epoch(active_epochs_df)

        return (Epoch(all_directional_laps_filter_epochs_decoder_result_value.filter_epochs).to_dataframe(), Epoch(all_directional_ripple_filter_epochs_decoder_result_value.filter_epochs).to_dataframe()), corr_column_names

    @classmethod
    def _perform_compute_simple_spike_time_v_pf_peak_x_by_epoch(cls, track_templates: TrackTemplates, spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, epoch_label_column_name = 'label') -> pd.DataFrame:
        """ Computes one type of epoch (laps, ripple) for all four decoders
        epoch_label_column_name = 'label'

        #TODO 2025-01-16 10:14: - [ ] This is returning and empty dataframe for some reason.
        
        """
        from pyphocorehelpers.indexing_helpers import partition_df # used by _compute_simple_spike_time_v_pf_peak_x_by_epoch
        from scipy.stats import pearsonr # used by _compute_simple_spike_time_v_pf_peak_x_by_epoch

        _NaN_Type = pd.NA
        _label_column_type: str = 'int64'
        
        ## Add the epochs identity column ('Probe_Epoch_id') to spikes_df so that they can be split by epoch:
        ## INPUTS: track_templates, spikes_df, active_epochs_df
        if not isinstance(active_epochs_df, pd.DataFrame):
            active_epochs_df = active_epochs_df.to_dataframe()

        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
        neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        included_neuron_ids = np.array(track_templates.any_decoder_neuron_IDs) # one list for all decoders
        n_neurons = len(included_neuron_ids)
        # print(f'included_neuron_ids: {included_neuron_ids}, n_neurons: {n_neurons}')

        # Get only the spikes for the shared_aclus:
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_neuron_ids)
        if epoch_label_column_name is not None:
            assert epoch_label_column_name in active_epochs_df
            active_epochs_df[epoch_label_column_name] = pd.to_numeric(active_epochs_df[epoch_label_column_name]).astype(int) # 'Int64'

        spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=epoch_label_column_name,
                                                                    should_replace_existing_column=True, drop_non_epoch_spikes=True) # , override_time_variable_name='t_seconds'
        # spikes_df = spikes_df[(spikes_df['Probe_Epoch_id'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
        spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        # spikes_df

        ## Add pf peak locations to each spike: _pf_peak_x_column_names # ['long_LR_pf_peak_x', 'long_RL_pf_peak_x', 'short_LR_pf_peak_x', 'short_RL_pf_peak_x']

        # Inputs: spikes_df
        decoder_aclu_peak_map_dict = track_templates.get_decoder_aclu_peak_map_dict(peak_mode='CoM') # original implementation
        # decoder_aclu_peak_map_dict = track_templates.get_decoder_aclu_peak_map_dict(peak_mode='peaks') # new attempt to improve ripple decoding scores by using the non-CoM positions. 2024-02-16 - actually worse performance
        
        ## Restrict to only the relevant columns, and Initialize the dataframe columns to np.nan:
        spikes_df: pd.DataFrame = deepcopy(spikes_df[['t_rel_seconds', 'aclu', 'Probe_Epoch_id']]).sort_values(['Probe_Epoch_id', 't_rel_seconds', 'aclu']).astype({'Probe_Epoch_id': _label_column_type}) # Sort by columns: 'Probe_Epoch_id' (ascending), 't_rel_seconds' (ascending), 'aclu' (ascending)

        # _pf_peak_x_column_names = ['LR_Long_pf_peak_x', 'RL_Long_pf_peak_x', 'LR_Short_pf_peak_x', 'RL_Short_pf_peak_x']
        _pf_peak_x_column_names = [f'{a_decoder_name}_pf_peak_x' for a_decoder_name in track_templates.get_decoder_names()]
        corr_column_names = [f'{n}_pearsonr' for n in _pf_peak_x_column_names]

        ## Initialize the output dataframe:
        spikes_df[_pf_peak_x_column_names] = pd.DataFrame([[_NaN_Type, _NaN_Type, _NaN_Type, _NaN_Type]], index=spikes_df.index)
        for a_decoder_name, an_aclu_peak_map in decoder_aclu_peak_map_dict.items():
            spikes_df[f'{a_decoder_name}_pf_peak_x'] = spikes_df.aclu.map(an_aclu_peak_map)

        # # NOTE: to shuffle aclus, a more complicated approach (as follows) must be used:
        # unique_Probe_Epoch_IDs = active_selected_spikes_df['Probe_Epoch_id'].unique()
        # for a_probe_epoch_ID in unique_Probe_Epoch_IDs:
        # 	mask = (a_probe_epoch_ID == active_selected_spikes_df['Probe_Epoch_id'])
        # 	for a_decoder_name, an_aclu_peak_map in decoder_aclu_peak_map_dict.items():
        # 		active_selected_spikes_df.loc[mask, 'aclu'] = active_selected_spikes_df.loc[mask, 'aclu'].sample(frac=1).values # Shuffle aclus here
        # 		active_selected_spikes_df.loc[mask, f'{a_decoder_name}_pf_peak_x'] = active_selected_spikes_df.loc[mask, 'aclu'].map(an_aclu_peak_map)

        # spikes_df

        ## Compute the spike-t v. pf_peak_x correlation for each of the decoders
        should_NaN_out_results_for_epochs_with_too_few_cells: bool = False
        required_min_percentage_of_active_cells: float = 0.333333 # 20% of active cells
        active_min_num_unique_aclu_inclusions_requirement: int = 15 # track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)
        min_num_required_unique_aclus_list = [max(5, int(float(len(a_decoder_neuron_IDs)) * 0.333)) for a_decoder_neuron_IDs in neuron_IDs_lists]
            

        _simple_corr_results_dict = {}
        partitioned_dfs: Dict[int, pd.DataFrame] = dict(zip(*partition_df(spikes_df, partitionColumn='Probe_Epoch_id')))
        for an_epoch_idx, an_epoch_spikes_df in partitioned_dfs.items():
            # get the spikes in each epoch
            # print(f'an_epoch_idx: {an_epoch_idx}, np.shape(epoch_spikes_df): {np.shape(an_epoch_spikes_df)}')
           
            # _temp_dfs = []
            _simple_corr_results_dict[an_epoch_idx] = []
            for a_peak_x_col_name, a_decoder_neuron_IDs, min_num_required_unique_aclus in zip(_pf_peak_x_column_names, neuron_IDs_lists, min_num_required_unique_aclus_list):
                ## For each decoder, so we can slice by that decoder's included neuron ids:

                # a_decoder_specific_spikes_df = deepcopy(an_epoch_spikes_df).spikes.sliced_by_neuron_id(a_decoder_neuron_IDs)
                a_decoder_specific_spikes_df = deepcopy(an_epoch_spikes_df)
                a_decoder_specific_spikes_df = a_decoder_specific_spikes_df[a_decoder_specific_spikes_df['aclu'].isin(a_decoder_neuron_IDs)] # filter down to only the decoder-unique entries
                active_epoch_decoder_active_aclus = a_decoder_specific_spikes_df.aclu.unique()

                # min_num_required_unique_aclus = max(5, int(float(len(a_decoder_neuron_IDs)) * 0.333))
                if (should_NaN_out_results_for_epochs_with_too_few_cells and (len(active_epoch_decoder_active_aclus) < min_num_required_unique_aclus)):
                    _simple_corr_results_dict[an_epoch_idx].append(np.nan) ## NaN can be added here to indicate that the curr epoch has too few active aclus for this decoder.
                else:
                    _an_arr = a_decoder_specific_spikes_df[['t_rel_seconds', a_peak_x_col_name]].dropna(subset=['t_rel_seconds', a_peak_x_col_name], inplace=False).to_numpy().T
                    if np.shape(_an_arr)[-1] < 2:
                        ## pearsonr is undefined if there are less than two elements in the list
                        print(f'WARNING: less than two array elements for an_epoch_idx: {an_epoch_idx}, a_pearson_v will be NaN for a_peak_x_col_name: {a_peak_x_col_name}, a_decoder_neuron_IDs: {a_decoder_neuron_IDs}, min_num_required_unique_aclus: {min_num_required_unique_aclus}')
                        a_pearson_v = np.nan               
                    else:
                        a_pearson_v = pearsonr(_an_arr[0], _an_arr[1]).statistic #ValueError: x and y must have length at least 2.
                    if np.isnan(a_pearson_v):
                        print(f'WARNING: hit NaN a_pearson_v: {a_pearson_v} for a_peak_x_col_name: {a_peak_x_col_name}, a_decoder_neuron_IDs: {a_decoder_neuron_IDs}, min_num_required_unique_aclus: {min_num_required_unique_aclus}')
                    _simple_corr_results_dict[an_epoch_idx].append(a_pearson_v) ## how is this returning NaN for some entries? ANS: pearsonr is undefined if there are less than two elements in the list


        ## Convert results dict into a pd.DataFrame            
        corr_df: pd.DataFrame = pd.DataFrame(_simple_corr_results_dict).T
        
        corr_df = corr_df.rename(columns=dict(zip(corr_df.columns, corr_column_names)))
        corr_df.index.name = 'epoch_id'
        return corr_df, corr_column_names

    @function_attributes(short_name=None, tags=['ground-truth', 'laps'], input_requires=[], output_provides=[], uses=['add_laps_groundtruth_information_to_dataframe'], used_by=['add_groundtruth_information'], creation_date='2024-04-19 18:53', related_items=[])
    def _perform_add_groundtruth_information_to_dataframe(self, curr_active_pipeline, result_laps_epochs_df: pd.DataFrame) -> pd.DataFrame:
        """ adds the ground_truth to the passed dataframe: `result_laps_epochs_df` and returns it:

        Updates: ['maze_id', 'is_LR_dir', 'is_most_likely_track_identity_Long', 'is_most_likely_direction_LR']

        Usage:
            a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = directional_merged_decoders_result
            result_laps_epochs_df: pd.DataFrame = a_directional_merged_decoders_result.add_groundtruth_information(curr_active_pipeline)
            result_laps_epochs_df

        """     
        result_laps_epochs_df = add_laps_groundtruth_information_to_dataframe(curr_active_pipeline=curr_active_pipeline, result_laps_epochs_df=result_laps_epochs_df)

        ## Add the decoded results to the laps df:
        ## Get the most likely direction/track from the decoded posteriors:
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = self.laps_directional_marginals_tuple
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = self.laps_track_identity_marginals_tuple
        result_laps_epochs_df['is_most_likely_track_identity_Long'] = laps_is_most_likely_track_identity_Long
        result_laps_epochs_df['is_most_likely_direction_LR'] = laps_is_most_likely_direction_LR_dir

        assert np.all([a_col in result_laps_epochs_df.columns for a_col in ('maze_id', 'is_LR_dir', 'is_most_likely_track_identity_Long', 'is_most_likely_direction_LR')]), f"result_laps_epochs_df.columns: {list(result_laps_epochs_df.columns)}"

        return result_laps_epochs_df
    

    @function_attributes(short_name=None, tags=['ground-truth', 'laps', 'validation'], input_requires=[], output_provides=[], uses=['_perform_add_groundtruth_information_to_dataframe'], used_by=[], creation_date='2024-04-05 19:23', related_items=['cls.validate_lap_dir_estimations()'])
    def add_groundtruth_information(self, curr_active_pipeline):
        """ adds the ground_truth to `self.laps_epochs_df`:

        Updates: ['maze_id', 'is_LR_dir', 'is_most_likely_track_identity_Long', 'is_most_likely_direction_LR']

        Usage:
            a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = directional_merged_decoders_result
            result_laps_epochs_df: pd.DataFrame = a_directional_merged_decoders_result.add_groundtruth_information(curr_active_pipeline)
            result_laps_epochs_df

        """
        ## Update the source:
        self.laps_epochs_df = self._perform_add_groundtruth_information_to_dataframe(curr_active_pipeline=curr_active_pipeline, result_laps_epochs_df=self.laps_epochs_df)

        # self.laps_weighted_corr_merged_df = self._perform_add_groundtruth_information_to_dataframe(curr_active_pipeline=curr_active_pipeline, result_laps_epochs_df=self.laps_weighted_corr_merged_df)
        # self.laps_simple_pf_pearson_merged_df = self._perform_add_groundtruth_information_to_dataframe(curr_active_pipeline=curr_active_pipeline, result_laps_epochs_df=self.laps_simple_pf_pearson_merged_df)

        ## Dict-type:
        # decoder_laps_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_df_dict
        # decoder_laps_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_extras_dict
        # decoder_laps_weighted_corr_df_dict: Dict[str, pd.DataFrame] = directional_decoders_epochs_decode_result.decoder_laps_weighted_corr_df_dict

        return self.laps_epochs_df
    
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





def validate_has_directional_merged_placefields(curr_active_pipeline, computation_filter_name='maze'):
    """ 
        DirectionalPseudo2DDecodersResult.validate_has_directional_merged_placefields
    """
    # Unpacking:
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    
    # extract properties:
    all_directional_decoder_dict_value = directional_merged_decoders_result.all_directional_decoder_dict
    all_directional_pf1D_Decoder_value = directional_merged_decoders_result.all_directional_pf1D_Decoder
    long_directional_pf1D_Decoder_value = directional_merged_decoders_result.long_directional_pf1D_Decoder
    long_directional_decoder_dict_value = directional_merged_decoders_result.long_directional_decoder_dict
    short_directional_pf1D_Decoder_value = directional_merged_decoders_result.short_directional_pf1D_Decoder
    short_directional_decoder_dict_value = directional_merged_decoders_result.short_directional_decoder_dict

    all_directional_laps_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
    all_directional_ripple_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result

    laps_epochs_df = directional_merged_decoders_result.laps_epochs_df
    ripple_epochs_df = directional_merged_decoders_result.ripple_epochs_df

    return True



@define(slots=False, repr=False)
class DirectionalDecodersContinuouslyDecodedResult(ComputedResult):
    """ a container used to cache the results of decoding across all time-bins (continuously) for multiple decoders, and at varying time_bin_sizes
    Also holds the four pf1D_Decoders and the merged Pseudo2D decoder.
    
    Used by: `_decode_continuous_using_directional_decoders` to hold the result of continuous decoding
    #TODO 2024-05-22 17:21: - [ ] This class is very poorly named. It holds a continuous decoding at multiple time_bin_sizes. Strangely it doesn't seemed to be used in the batch function that actually sweeps the time bins, only the continuous decoding computation fn. 

    #TODO 2024-05-22 17:26: - [ ] 'DirectionalDecodersDecodedResult' -> 'DirectionalDecodersContinuouslyDecodedResult'
    'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersDecodedResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersContinuouslyDecodedResult',
    

    #TODO 2025-03-04 17:39: - [ ] This class looks nearly general!
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult

        ## Get the result after computation:
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
    
        
    """
    _VersionedResultMixin_version: str = "2024.01.22_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(default=None, metadata={'field_added': "2024.01.16_0"})
    pseudo2D_decoder: BasePositionDecoder = serialized_field(default=None, metadata={'field_added': "2024.01.22_0"})
    spikes_df: pd.DataFrame = serialized_field(default=None, metadata={'field_added': "2024.01.22_0"}) # global
    
    # Posteriors computed via the all_directional decoder:
    continuously_decoded_result_cache_dict: Dict[float, Dict[types.DecoderName, DecodedFilterEpochsResult]] = serialized_field(default=None, metadata={'field_added': "2024.01.16_0"}) # key is the t_bin_size in seconds
    

    @property
    def most_recent_decoding_time_bin_size(self) -> Optional[float]:
        """Gets the last cached continuously_decoded_dict property."""
        if ((self.continuously_decoded_result_cache_dict is None) or (len(self.continuously_decoded_result_cache_dict or {}) < 1)):
            return None
        else:
            last_time_bin_size: float = list(self.continuously_decoded_result_cache_dict.keys())[-1]
            return last_time_bin_size   
        

    @property
    def most_recent_continuously_decoded_dict(self) -> Optional[Dict[str, DecodedFilterEpochsResult]]:
        """Gets the last cached continuously_decoded_dict property."""
        last_time_bin_size = self.most_recent_decoding_time_bin_size
        if (last_time_bin_size is None):
            return None
        else:
            # otherwise return the result            
            return self.continuously_decoded_result_cache_dict[last_time_bin_size]         


    @property
    def most_recent_continuously_decoded_pseudo2D_decoder(self) -> Optional[DecodedFilterEpochsResult]:
        """Gets the last cached continuously_decoded_dict property.
        
        pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('pseudo2D', None)
        
        """
        last_time_bin_size = self.most_recent_decoding_time_bin_size
        if (last_time_bin_size is None):
            return None
        else:
            # otherwise return the result            
            return self.continuously_decoded_result_cache_dict[last_time_bin_size].get('pseudo2D', None)    


    @property
    def continuously_decoded_pseudo2D_decoder_dict(self) -> Optional[Dict[float, DecodedFilterEpochsResult]]:
        """Gets 'pseudo2D' result only for each time_bin_size
        
        ## Split across the 2nd axis to make 1D posteriors that can be displayed in separate dock rows:
        assert p_x_given_n.shape[1] == 4, f"expected the 4 pseudo-y bins for the decoder in p_x_given_n.shape[1]. but found p_x_given_n.shape: {p_x_given_n.shape}"
        split_pseudo2D_posteriors_dict = {k:np.squeeze(p_x_given_n[:, i, :]) for i, k in enumerate(('long_LR', 'long_RL', 'short_LR', 'short_RL'))}
        
        pseudo2D_decoder_continuously_decoded_results: DecodedFilterEpochsResult = .continuously_decoded_pseudo2D_decoder_dict
        
        """
        if (self.continuously_decoded_result_cache_dict is None):
            return None
        else:
            # otherwise return the result            
            return {k:v.get('pseudo2D', None) for k, v in self.continuously_decoded_result_cache_dict.items()}   


    @function_attributes(short_name=None, tags=['pseudo2D', 'timeline-track', '1D', 'split-to-1D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-26 07:23', related_items=['pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult.split_pseudo2D_result_to_1D_result'])
    def split_pseudo2D_continuous_result_to_1D_continuous_result(self, pseudo2D_decoder_names_list: Optional[str]=None, debug_print=False):
        """ Get 1D representations of the Pseudo2D track (4 decoders) so they can be plotted on seperate tracks and bin-debugged independently.

        This returns the "all-decoders-sum-to-1-across-time' normalization style that Kamran likes
        
        
        Uses: self.continuously_decoded_result_cache_dict

        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult

        
        all_time_bin_sizes_output_dict: : Dict[float, Dict[types.DecoderName, SingleEpochDecodedResult]] = directional_decoders_decode_result.split_pseudo2D_continuous_result_to_1D_continuous_result()
        """
        from neuropy.utils.mixins.indexing_helpers import get_dict_subset
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult

        if pseudo2D_decoder_names_list is None:
            pseudo2D_decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')

        pseudo2D_decoder: BasePositionDecoder = self.pseudo2D_decoder  # merged pseudo2D decoder
        all_directional_pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = self.pf1D_Decoder_dict ## separate 1D decoders

        continuously_decoded_result_cache_dict = self.continuously_decoded_result_cache_dict

        # all_time_bin_sizes_output_dict = {'non_marginalized_raw_result': [], 'marginal_over_direction': [], 'marginal_over_track_ID': []}
        # flat_all_time_bin_sizes_output_tuples_list: List[Tuple] = []

        all_time_bin_sizes_output_dict: Dict[float, Dict[types.DecoderName, SingleEpochDecodedResult]] = {} ## time_bin_size: 1D_tracks
        
        for time_bin_size, a_continuously_decoded_dict in continuously_decoded_result_cache_dict.items():
            ## Each iteration here adds 4 more tracks -- one for each decoding context
            
            # a_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult]
            if debug_print:
                print(f'time_bin_size: {time_bin_size}')

            # info_string: str = f" - t_bin_size: {time_bin_size}"
            
            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict

            # continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_decode_result.most_recent_continuously_decoded_dict)
            assert a_continuously_decoded_dict is not None

            ## Get the separate 1D results, these are ready-to-go:
            all_directional_pf1D_Decoder_continuous_results_dict: Dict[types.DecoderName, SingleEpochDecodedResult] = {k:v.get_result_for_epoch(0) for k, v in get_dict_subset(a_continuously_decoded_dict, subset_includelist=None, subset_excludelist=['pseudo2D']).items()}
            

            ## Extract the Pseudo2D results as separate 1D tracks
            ## INPUTS: most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string
            
            # all_directional_continuously_decoded_dict = most_recent_continuously_decoded_dict or {}
            pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = a_continuously_decoded_dict.get('pseudo2D', None)
            assert len(pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list) == 1
            single_pseudo2D_decoder_continuously_decoded_result: SingleEpochDecodedResult = pseudo2D_decoder_continuously_decoded_result.get_result_for_epoch(0)
            
            p_x_given_n = single_pseudo2D_decoder_continuously_decoded_result.p_x_given_n ## continuous -- meaning single epoch
            # # p_x_given_n = pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list[0]['p_x_given_n']
            # time_bin_container = single_pseudo2D_decoder_continuously_decoded_result.time_bin_container
            # time_window_centers = time_bin_container.centers
            

            # p_x_given_n.shape # (62, 4, 209389)

            ## Split across the 2nd axis to make 1D posteriors that can be displayed in separate dock rows:
            assert p_x_given_n.shape[1] == len(pseudo2D_decoder_names_list), f"for pseudo2D_decoder_names_list: {pseudo2D_decoder_names_list}\n\texpected the len(pseudo2D_decoder_names_list): {len(pseudo2D_decoder_names_list)} pseudo-y bins for the decoder in p_x_given_n.shape[1]. but found p_x_given_n.shape: {p_x_given_n.shape}"
            # split_pseudo2D_posteriors_dict = {k:np.squeeze(p_x_given_n[:, i, :]) for i, k in enumerate(pseudo2D_decoder_names_list)}

            xbin_centers: Optional[NDArray] = None
            ybin_centers: Optional[NDArray] = None
            if pseudo2D_decoder is not None:
                assert xbin_centers is None, f"don't provide both xbin_centers and pseudo2D_decoder"
                assert ybin_centers is None, f"don't provide both ybin_centers and pseudo2D_decoder"
                original_position_data_shape = deepcopy(pseudo2D_decoder.original_position_data_shape[:-1]) ## all but the pseudo dim
                xbin_centers = deepcopy(pseudo2D_decoder.xbin_centers)
                ybin_centers = None

            # Need all_directional_pf1D_Decoder_dict
            # output_dict = {}
            output_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, SingleEpochDecodedResult] = {}

            # for a_decoder_name, a_1D_posterior in split_pseudo2D_posteriors_dict.items():
            for i, a_decoder_name in enumerate(pseudo2D_decoder_names_list):
                ## make separate `SingleEpochDecodedResult` objects
                
                # all_directional_pf1D_Decoder_continuous_results_dict
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name] = deepcopy(single_pseudo2D_decoder_continuously_decoded_result) ## copy the whole pseudo2D result
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n = a_1D_posterior ## or could squish them here
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n = np.squeeze(output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n[:, i, :]) ## or could squish them here
                
                ## Copied from `pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult.split_pseudo2D_result_to_1D_result`
                # #TODO 2025-06-13 14:29: - [X] fix busted-up columns: ['most_likely_position_indicies_list', 'most_likely_positions'
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies = np.squeeze(np.argmax(output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n, axis=0))   # average along time dimension

                if (xbin_centers is not None) and (ybin_centers is not None):
                    ## NOTE: I think 2D position decoding isn't implemented here.
                    # much more efficient than the other implementation. Result is # (85844, 2)
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions = np.vstack((xbin_centers[output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies[0,:]], self.ybin_centers[output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies[1,:]])).T
                elif (xbin_centers is not None):
                    # 1D Decoder case:
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions = np.squeeze(xbin_centers[output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies])
                else:
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions = []
                # #TODO 2025-06-13 14:30: - [ ] Finish recomputing marginals
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].marginal_x = None
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].marginal_y = None
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].compute_marginals()



                # _out_tuple = dict(a_decoder_name=a_decoder_name, a_position_decoder=pseudo2D_decoder, time_window_centers=time_window_centers, a_1D_posterior=a_1D_posterior)
                # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = _out_tuple
                # output_dict[a_decoder_name] = _out_tuple
            ## END for i, a_de...
            all_time_bin_sizes_output_dict[time_bin_size] = output_pseudo2D_split_to_1D_continuous_results_dict
            
        ## END for time_bin_size, a_continuously_decoded_dict in cont
        
        return all_time_bin_sizes_output_dict




        ## OUTPUTS: all_directional_pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder], all_directional_pf1D_Decoder_continuous_results_dict: Dict[types.DecoderName, SingleEpochDecodedResult]
        ## OUTPUTS: pseudo2D_decoder: BasePositionDecoder, single_pseudo2D_decoder_continuously_decoded_result: SingleEpochDecodedResult, 

    

    @classmethod
    def validate_has_directional_decoded_continuous_epochs(cls, curr_active_pipeline, computation_filter_name='maze') -> bool:
        """ Validates that the decoding is complete
        """
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
        if pseudo2D_decoder is None:
            return False
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        if len(continuously_decoded_result_cache_dict) < 1:
            return False
    
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        if time_bin_size is None:
            return False

        continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
        if continuously_decoded_dict is None:
            return False
        pseudo2D_decoder_continuously_decoded_result = continuously_decoded_dict.get('pseudo2D', None)
        if pseudo2D_decoder_continuously_decoded_result is None:
            return False

        return True




def _workaround_validate_has_directional_decoded_continuous_epochs(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """ Validates that the decoding is complete
    """
    directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
    all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
    pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
    if pseudo2D_decoder is None:
        return False
    continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
    if len(continuously_decoded_result_cache_dict) < 1:
        return False

    time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
    if time_bin_size is None:
        return False

    continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
    if continuously_decoded_dict is None:
        return False
    pseudo2D_decoder_continuously_decoded_result = continuously_decoded_dict.get('pseudo2D', None)
    if pseudo2D_decoder_continuously_decoded_result is None:
        return False

    return True



@define(slots=False, repr=False)
class DecoderDecodedEpochsResult(ComputedResult):
    """ Contains Decoded Epochs (such as laps, ripple) for a each of the Decoders.

    2024-02-15 - Computed by `_decode_and_evaluate_epochs_using_directional_decoders`
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

    #TODO 2025-03-11 05:56: - [ ] Seems stupid and wasteful, for example, why doesn't it contain the Pseudo2D decoder or its results? It also has a whole bunch of crap that just accumulated over time


    """ 

    _VersionedResultMixin_version: str = "2024.02.16_0" # to be updated in your IMPLEMENTOR to indicate its version

    pos_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)
    ripple_decoding_time_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)
    laps_decoding_time_bin_size: float = serialized_attribute_field(default=None, is_computable=False, repr=True)

    decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field(default=None)
    decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field(default=None)

    decoder_laps_radon_transform_df_dict: Dict = serialized_field(default=None)
    decoder_ripple_radon_transform_df_dict: Dict = serialized_field(default=None)
        
    decoder_laps_radon_transform_extras_dict: Dict = non_serialized_field(default=None) # non-serialized
    decoder_ripple_radon_transform_extras_dict: Dict = non_serialized_field(default=None) # non-serialized
        
    laps_weighted_corr_merged_df: pd.DataFrame = serialized_field(default=None)
    ripple_weighted_corr_merged_df: pd.DataFrame = serialized_field(default=None)
    decoder_laps_weighted_corr_df_dict: Dict = serialized_field(default=Factory(dict))
    decoder_ripple_weighted_corr_df_dict: Dict = serialized_field(default=Factory(dict))
    
    laps_simple_pf_pearson_merged_df: pd.DataFrame = serialized_field(default=None)
    ripple_simple_pf_pearson_merged_df: pd.DataFrame = serialized_field(default=None)
    
    @classmethod
    def compute_matching_best_indicies(cls, a_marginals_df: pd.DataFrame, index_column_name: str = 'most_likely_decoder_index', second_index_column_name: str = 'best_decoder_index', enable_print=True):
        """ count up the number of rows that the RadonTransform and the most-likely direction agree 
        
        DecoderDecodedEpochsResult.compute_matching_best_indicies

        """
        num_total_epochs: int = len(a_marginals_df)
        agreeing_rows_count: int = (a_marginals_df[index_column_name] == a_marginals_df[second_index_column_name]).sum()
        agreeing_rows_ratio = float(agreeing_rows_count)/float(num_total_epochs)
        if enable_print:
            print(f'agreeing_rows_count/num_total_epochs: {agreeing_rows_count}/{num_total_epochs}\n\tagreeing_rows_ratio: {agreeing_rows_ratio}')
        return agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs)


    @classmethod
    def add_session_df_columns(cls, df: pd.DataFrame, session_name: str, time_bin_size: float=None, t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None, time_col: str=None, end_time_col_name: Optional[str]=None) -> pd.DataFrame:
        """ adds session-specific information to the marginal dataframes 
    
        Added Columns: ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']

        Usage:
            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
            
            a_ripple_df = DecoderDecodedEpochsResult.add_session_df_columns(a_ripple_df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
    
        """
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
        
        return df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=time_bin_size, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=time_col, end_time_col_name=end_time_col_name)
    
    @classmethod
    @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 22:58', related_items=[])
    def load_user_selected_epoch_times(cls, curr_active_pipeline, track_templates=None, epochs_name='ripple', **additional_selections_context) -> Tuple[Dict[str, NDArray], NDArray]:
        """

        Usage:    
            decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = load_user_selected_epoch_times(curr_active_pipeline)
            # Finds the indicies into the dataframe (`filtered_ripple_simple_pf_pearson_merged_df`) from the decoder_user_selected_epoch_times_dict
            # Inputs: filtered_ripple_simple_pf_pearson_merged_df, decoder_user_selected_epoch_times_dict

            new_selections_dict = {}
            for a_name, a_start_stop_arr in decoder_user_selected_epoch_times_dict.items():
                # a_pagination_controller = self.pagination_controllers[a_name] # DecodedEpochSlicesPaginatedFigureController
                if len(a_start_stop_arr) > 0:
                    assert np.shape(a_start_stop_arr)[1] == 2, f"input should be start, stop times as a numpy array"
                    # new_selections_dict[a_name] = filtered_ripple_simple_pf_pearson_merged_df.epochs.find_data_indicies_from_epoch_times(a_start_stop_arr) # return indicies into dataframe
                    new_selections_dict[a_name] = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(a_start_stop_arr) # return sliced dataframes
                    
            new_selections_dict

        """
        # Inputs: curr_active_pipeline (for curr_active_pipeline.build_display_context_for_session)
        from neuropy.utils.misc import numpyify_array
        from neuropy.core.user_annotations import UserAnnotationsManager
        annotations_man = UserAnnotationsManager()
        user_annotations = annotations_man.get_user_annotations()

        if track_templates is None:
            # Get from the pipeline:
            directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
            rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            included_qclu_values: float = rank_order_results.included_qclu_values
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only
        

        # loaded_selections_context_dict = {a_name:curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections') for a_name, a_decoder in track_templates.get_decoders_dict().items()}
        loaded_selections_context_dict = {a_name:curr_active_pipeline.sess.get_context().merging_context('display_', IdentifyingContext(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections', **additional_selections_context)) for a_name, a_decoder in track_templates.get_decoders_dict().items()} ## gets around DisplayPipelineStage being passed for `curr_active_pipeline` sometimes

        decoder_user_selected_epoch_times_dict = {a_name:np.atleast_2d(numpyify_array(user_annotations.get(a_selections_ctx, []))) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}
        # loaded_selections_dict
        
        ## Inputs: loaded_selections_dict, 
        ## Find epochs that are present in any of the decoders:
        total_num_user_selections: int = int(np.sum([np.size(v) for v in decoder_user_selected_epoch_times_dict.values()]))
        if total_num_user_selections > 0:
            concatenated_selected_epoch_times = NumpyHelpers.safe_concat([a_start_stop_arr for a_name, a_start_stop_arr in decoder_user_selected_epoch_times_dict.items() if np.size(a_start_stop_arr)>0], axis=0) # ` if np.size(a_start_stop_arr)>0` part was added to avoid empty lists causing `ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)`
            any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
        else:
            print(f'WARNING: No user selections for this epoch')
            any_good_selected_epoch_times: NDArray = np.atleast_2d([]) 
            
        return decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times


    @classmethod
    @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-02 13:28', related_items=[])
    def merge_decoded_epochs_result_dfs(cls, *dfs_list, should_drop_directional_columns:bool=True, start_t_idx_name='ripple_start_t'):
        """ filter the ripple results scores by the user annotations. 
        
        *dfs_list: a series of dataframes to join
        should_drop_directional_columns:bool - if True, the direction (LR/RL) columns are dropped and only the _best_ columns are left.
        """   
        filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df = dfs_list # , additional_columns_merged_df

        df: Optional[pd.DataFrame] = None

        if filtered_ripple_simple_pf_pearson_merged_df is not None:
            if df is None:
                df = filtered_ripple_simple_pf_pearson_merged_df.copy()
                assert np.all(np.isin(['P_LR', 'P_RL'], df.columns)), f"{list(df.columns)}" # ,'P_Long', 'P_Short'
                direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
                # track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)

            direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
            track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)
            # Get only the best direction long/short values for each metric:
            df['long_best_pf_peak_x_pearsonr'] = np.where(direction_max_indices, df['long_LR_pf_peak_x_pearsonr'], df['long_RL_pf_peak_x_pearsonr'])
            df['short_best_pf_peak_x_pearsonr'] = np.where(direction_max_indices, df['short_LR_pf_peak_x_pearsonr'], df['short_RL_pf_peak_x_pearsonr'])
            if should_drop_directional_columns:
                df = df.drop(columns=['P_LR', 'P_RL','best_decoder_index', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr']) # drop the directional column names

        # Outputs: df

        ## Add new weighted correlation results as new columns in existing filter_epochs df:
        # Inputs: ripple_weighted_corr_merged_df, df from previous step

        if ripple_weighted_corr_merged_df is not None:
            if df is None:
                df: pd.DataFrame = ripple_weighted_corr_merged_df.copy()
                assert np.all(np.isin(['P_LR', 'P_RL'], df.columns)), f"{list(df.columns)}" # ,'P_Long', 'P_Short'
                direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
                # track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)


            ## Perfrom a 1D matching of the epoch start times:
            ## ORDER MATTERS:
            # elements =  df[start_t_idx_name].to_numpy()
            # test_elements = ripple_weighted_corr_merged_df[start_t_idx_name].to_numpy()
            # valid_found_indicies = np.nonzero(np.isclose(test_elements[:, None], elements, atol=1e-3).any(axis=1))[0] #TODO 2024-03-14 09:34: - [ ] ERROR HERE?!?!
            # hand_selected_ripple_weighted_corr_merged_df = ripple_weighted_corr_merged_df.iloc[valid_found_indicies].reset_index(drop=True) ## NOTE .iloc used here!
            valid_found_indicies = find_data_indicies_from_epoch_times(ripple_weighted_corr_merged_df, epoch_times=df[start_t_idx_name].to_numpy(), t_column_names=[start_t_idx_name,], atol=1e-3)
            hand_selected_ripple_weighted_corr_merged_df = ripple_weighted_corr_merged_df.loc[valid_found_indicies].reset_index(drop=True) ## Switched to .loc

            ## Add the wcorr columns to `df`:
            wcorr_column_names = ['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']
            df[wcorr_column_names] = hand_selected_ripple_weighted_corr_merged_df[wcorr_column_names] # add the columns to the dataframe
            df['long_best_wcorr'] = np.where(direction_max_indices, df['wcorr_long_LR'], df['wcorr_long_RL'])
            df['short_best_wcorr'] = np.where(direction_max_indices, df['wcorr_short_LR'], df['wcorr_short_RL'])
            if should_drop_directional_columns:
                df = df.drop(columns=['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']) # drop the directional column names
            
            ## Add differences:
            df['wcorr_abs_diff'] = df['long_best_wcorr'].abs() - df['short_best_wcorr'].abs()
            df['pearsonr_abs_diff'] = df['long_best_pf_peak_x_pearsonr'].abs() - df['short_best_pf_peak_x_pearsonr'].abs()

        return df


    @classmethod
    @function_attributes(short_name=None, tags=['temp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-01 22:58', related_items=[])
    def filter_epochs_dfs_by_annotation_times(cls, curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size, *dfs_list):
        """ filter the ripple results scores by the user annotations. 
        
        *dfs_list: a series of dataframes to join

        """   
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

        filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df = dfs_list

        hand_selected_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
        # hand_selected_ripple_simple_pf_pearson_merged_df

        df: pd.DataFrame = cls.merge_decoded_epochs_result_dfs(hand_selected_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df, should_drop_directional_columns=True)

        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end, time_col='ripple_start_t')
        df["time_bin_size"] = ripple_decoding_time_bin_size
        df['is_user_annotated_epoch'] = True # if it's filtered here, it's true

        return df



    @classmethod
    @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_valid_epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-04 00:00', related_items=[])
    def try_add_is_epoch_boolean_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times: NDArray, new_column_name:str='is_valid_epoch', t_column_names=None, atol:float=0.01, not_found_action='skip_index', debug_print=False) -> bool:
        """ tries to add a 'new_column_name' column to the dataframe. 
        
        t_column_names = ['ripple_start_t',]
        """
        if (any_good_selected_epoch_times is None):
            return False
        any_good_selected_epoch_indicies = None
        try:
            # any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=t_column_names, atol=atol, not_found_action=not_found_action, debug_print=debug_print)    
            any_good_selected_epoch_indicies = a_df.epochs.find_data_indicies_from_epoch_times(epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=t_column_names, atol=atol)
        except BaseException as e:
            print(f'ERROR: failed with error {e} while trying to add column "{new_column_name}". Out of options.')

        if any_good_selected_epoch_indicies is None:
            return False

        # print(f'\t succeded at getting indicies! for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
        a_df[new_column_name] = False
        # a_df[new_column_name].iloc[any_good_selected_epoch_indicies] = True
        a_df[new_column_name].loc[any_good_selected_epoch_indicies] = True
        # a_df[new_column_name].loc[a_df.index.to_numpy()[any_good_selected_epoch_indicies]] = True # IndexError: index 392 is out of bounds for axis 0 with size 390
        return True


    @classmethod
    @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_user_annotated_epoch'], input_requires=[], output_provides=[], uses=['cls.try_add_is_epoch_boolean_column'], used_by=[], creation_date='2024-03-02 13:17', related_items=[])
    def try_add_is_user_annotated_epoch_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times, t_column_names=['ripple_start_t',]) -> bool:
        """ tries to add a 'is_user_annotated_epoch' column to the dataframe. """
        return cls.try_add_is_epoch_boolean_column(a_df=a_df, any_good_selected_epoch_times=any_good_selected_epoch_times, new_column_name='is_user_annotated_epoch', t_column_names=t_column_names, atol=0.01, not_found_action='skip_index', debug_print=False)
    

    @classmethod
    @function_attributes(short_name=None, tags=['user-annotations', 'column', 'epoch', 'is_valid_epoch'], input_requires=[], output_provides=[], uses=['cls.try_add_is_epoch_boolean_column'], used_by=[], creation_date='2024-03-02 13:17', related_items=[])
    def try_add_is_valid_epoch_column(cls, a_df: pd.DataFrame, any_good_selected_epoch_times, t_column_names=['ripple_start_t',]) -> bool:
        """ tries to add a 'is_valid_epoch' column to the dataframe. """
        return cls.try_add_is_epoch_boolean_column(a_df=a_df, any_good_selected_epoch_times=any_good_selected_epoch_times, new_column_name='is_valid_epoch', t_column_names=t_column_names, atol=0.01, not_found_action='skip_index', debug_print=False)


    @function_attributes(short_name=None, tags=['columns', 'epochs', 'IMPORTANT'], input_requires=[], output_provides=[], uses=['filter_and_update_epochs_and_spikes'], used_by=[], creation_date='2024-03-14 09:22', related_items=[])
    def add_all_extra_epoch_columns(self, curr_active_pipeline, track_templates: TrackTemplates, required_min_percentage_of_active_cells: float = 0.333333,
                                     debug_print=False, **additional_selections_context) -> None:
        """ instead of filtering by the good/user-selected ripple epochs, it adds two columns: ['is_valid_epoch', 'is_user_annotated_epoch'] so they can be later identified and filtered to `self.decoder_ripple_filter_epochs_decoder_result_dict.filter_epochs`
        Updates `self.decoder_ripple_filter_epochs_decoder_result_dict.filter_epochs` in-place 
        """
        ## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict

        # 2024-03-04 - Filter out the epochs based on the criteria:
        _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

        filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline=curr_active_pipeline, global_epoch_name=global_epoch_name, track_templates=track_templates, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, **additional_selections_context)
        filtered_valid_epoch_times = filtered_epochs_df[['start', 'stop']].to_numpy()

        ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
        decoder_user_selected_epoch_times_dict, any_user_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)

        a_result_dict = self.decoder_ripple_filter_epochs_decoder_result_dict ## Only operates on `self.decoder_ripple_filter_epochs_decoder_result_dict` (ripples)

        for a_name, a_result in a_result_dict.items():
            did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(ensure_dataframe(a_result.filter_epochs), any_good_selected_epoch_times=any_user_selected_epoch_times, t_column_names=None)
            if debug_print:
                print(f'did_update_user_annotation_col["{a_name}"]: {did_update_user_annotation_col}')
            did_update_is_valid = DecoderDecodedEpochsResult.try_add_is_valid_epoch_column(ensure_dataframe(a_result.filter_epochs), any_good_selected_epoch_times=filtered_valid_epoch_times, t_column_names=None)
            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            a_result.filter_epochs = DecoderDecodedEpochsResult.add_session_df_columns(ensure_dataframe(a_result.filter_epochs), session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end)            
            if debug_print:
                print(f'did_update_is_valid["{a_name}"]: {did_update_is_valid}')
        if debug_print:
            print(f'\tdone.')


    @classmethod
    def add_score_best_dir_columns(cls, df: pd.DataFrame, col_name: str = 'pf_peak_x_pearsonr', should_drop_directional_columns:bool=False, is_col_name_suffix_mode: bool = False, 
                                include_LS_diff_col: bool=True, include_best_overall_score_col: bool=True) -> pd.DataFrame:
        """ adds in a single "*_diff" and the 'long_best_*', 'short_best_*' columns
        Generalized from `merge_decoded_epochs_result_dfs`

        is_col_name_suffix_mode: bool - if True, the variable name (specified by `col_name`)

        
        Usage:

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

            directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
            directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=False)

        """
        added_col_names = []
        direction_max_indices = df[['P_LR', 'P_RL']].values.argmax(axis=1)
        track_identity_max_indices = df[['P_Long', 'P_Short']].values.argmax(axis=1)
        # Get only the best direction long/short values for each metric:
        long_best_col_name: str = f'long_best_{col_name}'
        short_best_col_name: str = f'short_best_{col_name}'
        overall_best_col_name: str = f'overall_best_{col_name}'
        
        
        if is_col_name_suffix_mode:
            long_LR_string = f'long_LR_{col_name}'
            long_RL_string = f'long_RL_{col_name}'
            short_LR_string = f'short_LR_{col_name}'
            short_RL_string = f'short_RL_{col_name}'
        else:
            long_LR_string = f'{col_name}_long_LR'
            long_RL_string = f'{col_name}_long_RL'
            short_LR_string = f'{col_name}_short_LR'
            short_RL_string = f'{col_name}_short_RL'
        
        df[long_best_col_name] = np.where(direction_max_indices, df[long_LR_string], df[long_RL_string])
        df[short_best_col_name] = np.where(direction_max_indices, df[short_LR_string], df[short_RL_string])
        added_col_names.append(long_best_col_name)
        added_col_names.append(short_best_col_name)
        
        if should_drop_directional_columns:
            df = df.drop(columns=['P_LR', 'P_RL','best_decoder_index', long_LR_string, long_RL_string, short_LR_string, short_RL_string]) # drop the directional column names

        ## Add differences:
        if include_LS_diff_col:
            LS_diff_col_name: str = f'{col_name}_diff'
            df[LS_diff_col_name] = df[long_best_col_name].abs() - df[short_best_col_name].abs()
            added_col_names.append(LS_diff_col_name)
        

        ## adds an "overall_best_{col_name}" column that includes the maximum between long and short
        if include_best_overall_score_col and (long_best_col_name in df) and (short_best_col_name in df):
            df[overall_best_col_name] = df[[long_best_col_name, short_best_col_name]].max(axis=1, skipna=True)
            added_col_names.append(overall_best_col_name)

        return df, tuple(added_col_names)


    @classmethod
    def get_all_scores_column_names(cls) -> Tuple:
        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring
        
        # Column Names _______________________________________________________________________________________________________ #
        basic_df_column_names = ['start', 'stop', 'label', 'duration']
        selection_col_names = ['is_user_annotated_epoch', 'is_valid_epoch']
        session_identity_col_names = ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']
        
        # Score Columns (one value for each decoder) _________________________________________________________________________ #
        decoder_bayes_prob_col_names = ['P_decoder']

        radon_transform_col_names = ['score', 'velocity', 'intercept', 'speed']
        weighted_corr_col_names = ['wcorr']
        pearson_col_names = ['pearsonr']

        heuristic_score_col_names = HeuristicReplayScoring.get_all_score_computation_col_names()
        print(f'heuristic_score_col_names: {heuristic_score_col_names}')

        ## All included columns:
        all_df_shared_column_names: List[str] = basic_df_column_names + selection_col_names + session_identity_col_names # these are not replicated for each decoder, they're the same for the epoch
        all_df_score_column_names: List[str] = decoder_bayes_prob_col_names + radon_transform_col_names + weighted_corr_col_names + pearson_col_names + heuristic_score_col_names 
        all_df_column_names: List[str] = all_df_shared_column_names + all_df_score_column_names ## All included columns, includes the score columns which will not be replicated

        ## Add in the 'wcorr' metrics:
        merged_conditional_prob_column_names = ['P_LR', 'P_RL', 'P_Long', 'P_Short']
        merged_wcorr_column_names = ['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']

        return (all_df_shared_column_names, all_df_score_column_names, all_df_column_names,
                    merged_conditional_prob_column_names, merged_wcorr_column_names, heuristic_score_col_names)

    @function_attributes(short_name=None, tags=['merged', 'all_scores', 'df', 'epochs'], input_requires=[], output_provides=[], uses=['.decoder_ripple_filter_epochs_decoder_result_dict', '_build_merged_score_metric_df'], used_by=[], creation_date='2024-03-14 19:10', related_items=[])
    def build_complete_all_scores_merged_df(self, debug_print=False) -> pd.DataFrame:
        """ Builds a single merged dataframe from the four separate .filter_epochs dataframes from the result for each decoder, merging them into a single dataframe with ['_long_LR','_long_RL','_short_LR','_short_RL'] suffixes for the combined columns.
        2024-03-14 19:04 

        Usage:
            extracted_merged_scores_df = build_complete_all_scores_merged_df(directional_decoders_epochs_decode_result)
            extracted_merged_scores_df


        #TODO 2024-07-15 18:32: - [ ] Ending up with multiple 'P_LR' columns in the dataframe! Not sure how this can happen.


        """
        from neuropy.core.epoch import ensure_dataframe
        from neuropy.utils.indexing_helpers import flatten, NumpyHelpers, PandasHelpers
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _build_merged_score_metric_df

        # # Column Names _______________________________________________________________________________________________________ #

        # ## All included columns:
        if debug_print:
            print(f'build_complete_all_scores_merged_df(...):')
        all_df_shared_column_names, all_df_score_column_names, all_df_column_names, merged_conditional_prob_column_names, merged_wcorr_column_names, heuristic_score_col_names = self.get_all_scores_column_names()

        ## Extract the concrete dataframes from the results:
        extracted_filter_epochs_dfs_dict = {k:ensure_dataframe(a_result.filter_epochs) for k, a_result in self.decoder_ripple_filter_epochs_decoder_result_dict.items()} # #TODO 2024-12-17 08:12: - [ ] Debugging: P_Decoder columns already have NaNs in the same place as they do in the next step here!
        ## Merge the dict of four dataframes, one for each decoder, with column names like ['wcorr', 'travel', 'speed'] to a single merged df with suffixed of the dict keys like ['wcorr_long_LR', 'wcorr_long_RL',  ...., 'travel_long_LR', 'travel_long_RL', 'travel_short_LR', 'travel_short_RL', ...]
        extracted_merged_scores_df: pd.DataFrame = _build_merged_score_metric_df(extracted_filter_epochs_dfs_dict, columns=all_df_score_column_names, best_decoder_index_column_name=None) ## already has NaN values for the decoder probabilities here :[
        # extracted_merged_scores_df

        _ref_df = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0]) # first dataframe is the same as the others, determine which columns are available
        included_all_df_shared_column_names = [k for k in all_df_shared_column_names if k in _ref_df.columns] # only the included columns

        # `common_shared_portion_df` the columns of the dataframe that is the same for all four decoders
        # common_shared_portion_df: pd.DataFrame = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0][all_df_shared_column_names]) # copy it from the first dataframe
        common_shared_portion_df: pd.DataFrame = deepcopy(tuple(extracted_filter_epochs_dfs_dict.values())[0][included_all_df_shared_column_names]) # copy it from the first dataframe
        base_shape = np.shape(common_shared_portion_df)

        included_merge_dfs_list = [common_shared_portion_df]

        #TODO 2024-07-12 07:06: - [ ] `self.ripple_weighted_corr_merged_df` is the problem it seems, it's of different size (more epochs) than all of the other dataframes

        ##Gotta get those ['P_LR', 'P_RL'] columns to determine best directions
        conditional_prob_df = deepcopy(self.ripple_weighted_corr_merged_df[merged_conditional_prob_column_names]) ## just use the columns from this
        conditional_prob_df_shape = np.shape(conditional_prob_df)
        if (base_shape[0] != conditional_prob_df_shape[0]):
            print(f'\tbuild_complete_all_scores_merged_df(...): warning: all dfs should have same number of rows, but conditional_prob_df_shape: {conditional_prob_df_shape} != base_shape: {base_shape}. Skipping adding `conditional_prob_df`.')
        else:
            ## add it 
            included_merge_dfs_list.append(conditional_prob_df)

        
        ## Re-derive the correct conditional probs:
        # ['P_LR', 'P_RL']
        # ['P_Long', 'P_Short']

        P_decoder_column_names = ['P_decoder_long_LR','P_decoder_long_RL','P_decoder_short_LR','P_decoder_short_RL']
        P_decoder_marginals_column_names = ['P_LR', 'P_RL', 'P_Long', 'P_Short']

        # if np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_column_names]):
        if np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_marginals_column_names]):
            # needs Marginalized Probability columns: ['P_LR', 'P_RL'], ['P_Long', 'P_Short']
            if debug_print:
                print(f'\tneeds Marginalized Probability columns. adding.')
            # assert np.any([(a_col not in extracted_merged_scores_df) for a_col in P_decoder_column_names]), f"missing marginals and cannot recompute them because we're also missing the raw probabilities. extracted_merged_scores_df.columns: {list(extracted_merged_scores_df.columns)}"
            ## They remain normalized because they all already sum to one.
            extracted_merged_scores_df['P_Long'] = extracted_merged_scores_df['P_decoder_long_LR'] + extracted_merged_scores_df['P_decoder_long_RL']
            extracted_merged_scores_df['P_Short'] = extracted_merged_scores_df['P_decoder_short_LR'] + extracted_merged_scores_df['P_decoder_short_RL']

            extracted_merged_scores_df['P_LR'] = extracted_merged_scores_df['P_decoder_long_LR'] + extracted_merged_scores_df['P_decoder_short_LR']
            extracted_merged_scores_df['P_RL'] = extracted_merged_scores_df['P_decoder_long_RL'] + extracted_merged_scores_df['P_decoder_short_RL']


        extracted_merged_scores_df_shape = np.shape(extracted_merged_scores_df)
        if (base_shape[0] != extracted_merged_scores_df_shape[0]):
            print(f'\tbuild_complete_all_scores_merged_df(...): warning: all dfs should have same number of rows, but extracted_merged_scores_df_shape: {extracted_merged_scores_df_shape} != base_shape: {base_shape}. Skipping adding `extracted_merged_scores_df`.')
        else:
            ## add it
            included_merge_dfs_list.append(extracted_merged_scores_df)

        # # Weighted correlations:

        # Build the final merged dataframe with the score columns for each of the four decoders but only one copy of the common columns.
        extracted_merged_scores_df: pd.DataFrame = pd.concat(included_merge_dfs_list, axis='columns') # (common_shared_portion_df, conditional_prob_df, extracted_merged_scores_df) ### I THINK NaNs come in here
        # extracted_merged_scores_df: pd.DataFrame = pd.concat((common_shared_portion_df, conditional_prob_df, extracted_merged_scores_df), axis='columns')
        extracted_merged_scores_df['ripple_start_t'] = extracted_merged_scores_df['start']

        if np.any([(a_col not in extracted_merged_scores_df) for a_col in merged_wcorr_column_names]):
            # needs wcorr columns
            if debug_print:
                print(f'\tbuild_complete_all_scores_merged_df(...): needs wcorr columns. adding.')
            wcorr_columns_df = deepcopy(self.ripple_weighted_corr_merged_df[merged_wcorr_column_names]) ## just use the columns from this
            assert np.shape(wcorr_columns_df)[0] == np.shape(extracted_merged_scores_df)[0], f"should have same number of columns"
            extracted_merged_scores_df: pd.DataFrame = pd.concat((extracted_merged_scores_df, wcorr_columns_df), axis='columns')

        ## Add in the wcorr and pearsonr columns:
        # self.ripple_simple_pf_pearson_merged_df ## ?? where is it getting "pearsonr_long_LR"?

        ## add in the "_diff" columns and the 'best_dir_*' columns
        added_column_names = []
        # for a_score_col in heuristic_score_col_names:
        #     extracted_merged_scores_df, curr_added_column_name_tuple = self.add_score_best_dir_columns(extracted_merged_scores_df, col_name=a_score_col, should_drop_directional_columns=False, is_col_name_suffix_mode=False)
        #     added_column_names.extend(curr_added_column_name_tuple)
        #     # (long_best_col_name, short_best_col_name, LS_diff_col_name)

        try:
            for a_score_col in all_df_score_column_names:
                extracted_merged_scores_df, curr_added_column_name_tuple = self.add_score_best_dir_columns(extracted_merged_scores_df, col_name=a_score_col, should_drop_directional_columns=False, is_col_name_suffix_mode=False, include_LS_diff_col=True, include_best_overall_score_col=True)
                added_column_names.extend(curr_added_column_name_tuple)
        except Exception as err:
            print(f'\tbuild_complete_all_scores_merged_df(...): Encountered ERROR: {err} while trying to add "a_score_col": {a_score_col}, but trying to continue, so close!')


        extracted_merged_scores_df = extracted_merged_scores_df.rename(columns=dict(zip(['P_decoder_long_LR','P_decoder_long_RL','P_decoder_short_LR','P_decoder_short_RL'], ['P_Long_LR','P_Long_RL','P_Short_LR','P_Short_RL'])), inplace=False)
        if 'time_bin_size' not in extracted_merged_scores_df.columns:
            ## add the column
            extracted_merged_scores_df['time_bin_size'] = self.ripple_decoding_time_bin_size

        extracted_merged_scores_df = PandasHelpers.dropping_duplicated_df_columns(df=extracted_merged_scores_df)
        return extracted_merged_scores_df



    @classmethod
    def _perform_export_dfs_dict_to_csvs(cls, extracted_dfs_dict: Dict, parent_output_path: Path, active_context: IdentifyingContext, session_name: str, tbin_values_dict: Dict,
                                        t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None,
                                        user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None):
        """ Classmethod: export as separate .csv files. 
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
        df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
        df

        tbin_values_dict={'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}

        #TODO 2024-11-01 07:53: - [ ] Need to pass the correct (full) context, including the qclu/fr_Hz filter values and the replay name. 
        #TODO 2024-11-01 07:54: - [X] Need to add a proper timebin column to the df instead of including it in the filename (if possible)
            - does already add a 'time_bin_size' column, and the suffix is probably so it doesn't get overwritten by different time_bin_sizes, might need to put them together post-hoc
            
        '2024-11-01_1250PM-kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_weighted_corr_merged_df)_tbin-1.5.csv'
            
        """

        assert parent_output_path.exists(), f"'{parent_output_path}' does not exist!"
        output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
        # Export CSVs:
        def export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
            """ captures `active_context`, 'output_date_str'
            """
            # parent_output_path: Path = Path('output').resolve()
            # active_context = curr_active_pipeline.get_session_context()
            session_identifier_str: str = active_context.get_description() # 'kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
            # session_identifier_str: str = active_context.get_description(subset_excludelist=['custom_suffix']) # no this is just the session
            assert output_date_str is not None
            out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-11-15_0200PM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays_qclu_[1, 2, 4, 6, 7, 9]_frateThresh_5.0-(ripple_WCorrShuffle_df)_tbin-0.025'
            out_filename = f"{out_basename}.csv"
            out_path = parent_output_path.joinpath(out_filename).resolve()
            export_df.to_csv(out_path)
            return out_path 

        if custom_export_df_to_csv_fn is None:
            # use the default
            custom_export_df_to_csv_fn = export_df_to_csv


        # active_context.custom_suffix = '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0' # '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]'
        
        #TODO 2024-03-02 12:12: - [ ] Could add weighted correlation if there is a dataframe for that and it's computed:
        # tbin_values_dict = {'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}
        time_col_name_dict = {'laps': 'lap_start_t', 'ripple': 'ripple_start_t'} ## default should be 't_bin_center'
    
        ## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict
        export_files_dict = {}
        
        for a_df_name, a_df in extracted_dfs_dict.items():
            an_epochs_source_name: str = a_df_name.split(sep='_', maxsplit=1)[0] # get the first part of the variable names that indicates whether it's for "laps" or "ripple"

            a_tbin_size: float = float(tbin_values_dict[an_epochs_source_name])
            a_time_col_name: str = time_col_name_dict.get(an_epochs_source_name, 't_bin_center')
            ## Add t_bin column method
            a_df = cls.add_session_df_columns(a_df, session_name=session_name, time_bin_size=a_tbin_size, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=a_time_col_name)
            a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
            a_data_identifier_str: str = f'({a_df_name})_tbin-{a_tbin_size_str}' ## build the identifier '(laps_weighted_corr_merged_df)_tbin-1.5'
            
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
                            any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
                        except AttributeError as e:
                            print(f'ERROR: failed method 2 for {a_df_name}. Out of options.')        
                        except Exception as e:
                            print(f'ERROR: failed for {a_df_name}. Out of options.')
                        
                    if any_good_selected_epoch_indicies is not None:
                        print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} user selections) for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
                        a_df['is_user_annotated_epoch'] = False
                        a_df['is_user_annotated_epoch'].iloc[any_good_selected_epoch_indicies] = True
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
                            any_good_selected_epoch_indicies = find_data_indicies_from_epoch_times(a_df, np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t',], atol=0.01, not_found_action='skip_index', debug_print=False)
                        except AttributeError as e:
                            print(f'ERROR: failed method 2 for {a_df_name}. Out of options.')        
                        except Exception as e:
                            print(f'ERROR: failed for {a_df_name}. Out of options.')
                        
                    if any_good_selected_epoch_indicies is not None:
                        print(f'\t succeded at getting {len(any_good_selected_epoch_indicies)} selected indicies (of {num_valid_epoch_times} valid filter epoch times) for {a_df_name}. got {len(any_good_selected_epoch_indicies)} indicies!')
                        a_df['is_valid_epoch'] = False

                        try:
                            a_df['is_valid_epoch'].iloc[any_good_selected_epoch_indicies] = True
                            # a_df['is_valid_epoch'].loc[any_good_selected_epoch_indicies] = True

                        except Exception as e:
                            print(f'WARNING: trying to get whether the epochs are valid FAILED probably, 2024-06-28 custom computed epochs thing: {e}, just setting all to True')
                            a_df['is_valid_epoch'] = True
                    else:
                        print(f'\t failed all methods for selection filter')

            export_files_dict[a_df_name] = custom_export_df_to_csv_fn(a_df, data_identifier_str=a_data_identifier_str, parent_output_path=parent_output_path) # this is exporting corr '(ripple_WCorrShuffle_df)_tbin-0.025'
        # end for a_df_name, a_df
        
        return export_files_dict
    


    def perform_export_dfs_dict_to_csvs(self, extracted_dfs_dict: Dict, parent_output_path: Path, active_context, session_name: str, curr_session_t_delta: Optional[float], user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None):
        """ export as separate .csv files. 
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
        df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
        df

        """
        return self._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=extracted_dfs_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, tbin_values_dict={'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size},
                                                     curr_session_t_delta=curr_session_t_delta, user_annotation_selections=user_annotation_selections, valid_epochs_selections=valid_epochs_selections, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)



    @function_attributes(short_name=None, tags=['export', 'CSV', 'main'], input_requires=[], output_provides=['ripple_all_scores_merged_df.csv'], uses=['self.perform_export_dfs_dict_to_csvs', 'self.build_complete_all_scores_merged_df'], used_by=['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'], creation_date='2024-03-15 10:13', related_items=[])
    def export_csvs(self, parent_output_path: Path, active_context: IdentifyingContext, session_name: str, curr_session_t_delta: Optional[float], user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=None, export_df_variable_names=None, should_export_complete_all_scores_df:bool=True):
        """ export as separate .csv files. 

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

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
        df = filter_epochs_dfs_by_annotation_times(curr_active_pipeline, any_good_selected_epoch_times, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, filtered_ripple_simple_pf_pearson_merged_df, ripple_weighted_corr_merged_df)
        df


            
        """
        export_files_dict = {}
        _df_variables_names = ['laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df']
        if export_df_variable_names is None:
            # export all by default
            export_df_variable_names = _df_variables_names
            
        extracted_dfs_dict = {a_df_name:getattr(self, a_df_name) for a_df_name in export_df_variable_names}
        if len(extracted_dfs_dict) > 0:
            export_files_dict = export_files_dict | self.perform_export_dfs_dict_to_csvs(extracted_dfs_dict=extracted_dfs_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=user_annotation_selections, valid_epochs_selections=valid_epochs_selections, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)

        ## try to export the merged all_scores dataframe
        if should_export_complete_all_scores_df:
            extracted_merged_scores_df: pd.DataFrame = self.build_complete_all_scores_merged_df()
            if 'time_bin_size' not in extracted_merged_scores_df.columns:
                ## add the column
                print(f'WARN: adding the time_bin_size columns: {self.ripple_decoding_time_bin_size}')
                extracted_merged_scores_df['time_bin_size'] = self.ripple_decoding_time_bin_size

            export_df_dict = {'ripple_all_scores_merged_df': extracted_merged_scores_df}
            export_files_dict = export_files_dict | self.perform_export_dfs_dict_to_csvs(extracted_dfs_dict=export_df_dict, parent_output_path=parent_output_path, active_context=active_context, session_name=session_name, curr_session_t_delta=curr_session_t_delta, user_annotation_selections=None, valid_epochs_selections=None, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn)

        return export_files_dict

    
    # ## For serialization/pickling:
    # def __getstate__(self):
    # 	# Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
    # 	state = self.__dict__.copy()
    # 	return state

    # def __setstate__(self, state):
    # 	# Restore instance attributes (i.e., _mapping and _keys_at_init).
    # 	self.__dict__.update(state)
    # 	# Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
    # 	super(DecoderDecodedEpochsResult, self).__init__() # from




def _workaround_validate_has_directional_decoded_epochs_evaluations(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """ Validates that the decoding is complete, workaround to maybe prevent  #TODO 2024-02-16 14:25: - [ ] PicklingError: Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x7fd35e66b700>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell

    
    """
    directional_decoders_decode_epochs_result = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
    if directional_decoders_decode_epochs_result is None:
        return False
    pos_bin_size: float = directional_decoders_decode_epochs_result.pos_bin_size
    if pos_bin_size is None:
        return False
    ripple_decoding_time_bin_size: float = directional_decoders_decode_epochs_result.ripple_decoding_time_bin_size
    if ripple_decoding_time_bin_size is None:
        return False
    laps_decoding_time_bin_size: float = directional_decoders_decode_epochs_result.laps_decoding_time_bin_size
    if laps_decoding_time_bin_size is None:
        return False

    decoder_laps_filter_epochs_decoder_result_dict = directional_decoders_decode_epochs_result.decoder_laps_filter_epochs_decoder_result_dict
    if decoder_laps_filter_epochs_decoder_result_dict is None:
        return False

    decoder_ripple_filter_epochs_decoder_result_dict = directional_decoders_decode_epochs_result.decoder_ripple_filter_epochs_decoder_result_dict
    if decoder_ripple_filter_epochs_decoder_result_dict is None:
        return False

    laps_simple_pf_pearson_merged_df = directional_decoders_decode_epochs_result.laps_simple_pf_pearson_merged_df
    if laps_simple_pf_pearson_merged_df is None:
        return False

    #TODO 2024-02-16 13:52: - [ ] Rest of properties
    return True

# from neuropy.utils.indexing_helpers import MissingColumnsError





def _workaround_validate_has_directional_decoded_epochs_heuristic_scoring(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """ 
    
    KeyError: "['travel', 'coverage', 'jump', 'longest_sequence_length_ratio', 'direction_change_bin_ratio', 'congruent_dir_bins_ratio', 'total_congruent_direction_change'] not in index"

    
    """
    from neuropy.core.epoch import ensure_dataframe
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicsResult

    print_missing_columns: bool = True
    directional_decoders_decode_epochs_result = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
    if directional_decoders_decode_epochs_result is None:
        return False

    decoder_laps_filter_epochs_decoder_result_dict = directional_decoders_decode_epochs_result.decoder_laps_filter_epochs_decoder_result_dict
    if decoder_laps_filter_epochs_decoder_result_dict is None:
        return False

    decoder_ripple_filter_epochs_decoder_result_dict = directional_decoders_decode_epochs_result.decoder_ripple_filter_epochs_decoder_result_dict
    if decoder_ripple_filter_epochs_decoder_result_dict is None:
        return False
    

    all_df_shared_column_names, all_df_score_column_names, all_df_column_names, merged_conditional_prob_column_names, merged_wcorr_column_names, heuristic_score_col_names = DecoderDecodedEpochsResult.get_all_scores_column_names()
    
    # heuristic_score_col_names = ['congruent_dir_bins_ratio', 'coverage', 'direction_change_bin_ratio', 'longest_sequence_length', 'longest_sequence_length_ratio', 'travel']
    
    ripple_has_required_columns = PandasHelpers.require_columns({a_name:ensure_dataframe(a_result.filter_epochs) for a_name, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()},
        required_columns=heuristic_score_col_names, print_missing_columns=print_missing_columns)
    if (ripple_has_required_columns is None) or (not ripple_has_required_columns):
        # raise MissingColumnsError(heuristic_score_col_names)
        return False

    laps_has_required_columns = PandasHelpers.require_columns({a_name:ensure_dataframe(a_result.filter_epochs) for a_name, a_result in decoder_laps_filter_epochs_decoder_result_dict.items()},
        required_columns=heuristic_score_col_names,  print_missing_columns=print_missing_columns)
    if (laps_has_required_columns is None) or (not laps_has_required_columns):
        # raise MissingColumnsError(heuristic_score_col_names)
        return False


    heuristics_result: HeuristicsResult = curr_active_pipeline.global_computation_results.computed_data.get('Heuristics', None)
    if heuristics_result is None:
        return False

    return True




# ---------------------------------------------------------------------------- #
#             2024-03-29 - Rigorous Decoder Performance assessment             #
# ---------------------------------------------------------------------------- #
# Quantify cell contributions to decoders
# Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result

import portion as P # Required for interval search: portion~=2.3.0
import attrs
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df


# 2024-04-04 - Continued Decoder Error Assessment ____________________________________________________________________ #

@define(slots=False, repr=False, eq=False)
class DecodedContextCorrectnessArraysTuple(UnpackableMixin, object):
    is_decoded_track_correct: bool = field()
    is_decoded_dir_correct: bool = field()
    are_both_decoded_properties_correct: bool = field()
    
@define(slots=False, repr=False, eq=False)
class PercentDecodedContextCorrectnessTuple(UnpackableMixin, object):
    percent_laps_track_identity_estimated_correctly: float = field()
    percent_laps_direction_estimated_correctly: float = field()
    percent_laps_estimated_correctly: float = field()

@define(slots=False, repr=False, eq=False)
class CompleteDecodedContextCorrectness(UnpackableMixin, object):
    correctness_arrays_tuple: Tuple = field()
    percent_correct_tuple: Tuple = field()

@define(slots=False, repr=False, eq=False)
class MeasuredDecodedPositionComparison(UnpackableMixin, object):
    measured_positions_dfs_list: List[pd.DataFrame] = field()
    decoded_positions_df_list: List[pd.DataFrame] = field()
    decoded_measured_diff_df: pd.DataFrame = field()
    
@define(slots=False)
class CustomDecodeEpochsResult(UnpackableMixin):
    """ Represents a custom decoding of an Epochs object that also compares the decoded position with the measured position
    
    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, MeasuredDecodedPositionComparison, DecodedFilterEpochsResult

            
        CustomDecodeEpochsResult.build_measured_decoded_position_comparison


    """
    measured_decoded_position_comparion: MeasuredDecodedPositionComparison = field()
    decoder_result: DecodedFilterEpochsResult = field()
    epochs_bin_by_bin_performance_analysis_df: pd.DataFrame = field()
    

    @classmethod
    def init_from_single_decoder_decoding_result_and_measured_pos_df(cls, a_decoder_decoding_result: DecodedFilterEpochsResult, global_measured_position_df: pd.DataFrame, pfND_Decoder: Optional[BasePositionDecoder]=None, debug_print:bool=False) -> "CustomDecodeEpochsResult":
        """ compare the decoded most-likely-positions and the measured positions interpolated to the same time bins.
        
         all_directional_laps_filter_epochs_decoder_custom_result: CustomDecodeEpochsResult = CustomDecodeEpochsResult.init_from_single_decoder_decoding_result_and_measured_pos_df(a_decoder_decoding_result=deepcopy(all_directional_laps_filter_epochs_decoder_result), global_measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']))
         all_directional_laps_filter_epochs_decoder_custom_result
        """
        # all_directional_laps_filter_epochs_decoder_custom_result: MeasuredDecodedPositionComparison = cls.build_single_measured_decoded_position_comparison(a_decoder_decoding_result=deepcopy(all_directional_laps_filter_epochs_decoder_result), global_measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']))
        a_custom_decoder_decoding_result = cls(measured_decoded_position_comparion=cls.build_single_measured_decoded_position_comparison(a_decoder_decoding_result=a_decoder_decoding_result, global_measured_position_df=global_measured_position_df),
                                                                                                 decoder_result=a_decoder_decoding_result,
                                                                                                 epochs_bin_by_bin_performance_analysis_df=None,
        )
        if pfND_Decoder is not None:
            try:
                a_custom_decoder_decoding_result.epochs_bin_by_bin_performance_analysis_df = a_custom_decoder_decoding_result.get_lap_bin_by_bin_performance_analysis_df(active_pf_2D=deepcopy(pfND_Decoder), debug_print=debug_print) # active_pf_2D: used for binning position columns # active_pf_2D: used for binning position columns\
            except ValueError as e:
                # AssertionError: curr_array_shape: (57, 3) but this only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL'] 
                pass # skip this for now
            except Exception as e:
                raise
        return a_custom_decoder_decoding_result
        

    @classmethod
    def build_single_measured_decoded_position_comparison(cls, a_decoder_decoding_result: DecodedFilterEpochsResult, global_measured_position_df: pd.DataFrame, should_drop_epochs_with_no_valid_timebins: bool = True) -> MeasuredDecodedPositionComparison:
        """ compare the decoded most-likely-positions and the measured positions interpolated to the same time bins.
        
        """
        from sklearn.metrics import mean_squared_error

        decoded_time_bin_centers_list = deepcopy([a_cont.centers for a_cont in a_decoder_decoding_result.time_bin_containers]) # this is NOT the same for all decoders because they could have different numbers of test laps because different directions/configs might have different numbers of general laps

        measured_positions_dfs_list = []
        decoded_positions_df_list = [] # one per epoch
        decoded_measured_diff_df = [] # one per epoch

        for epoch_idx, a_sample_times in enumerate(decoded_time_bin_centers_list):
            interpolated_measured_df = TrainTestLapsSplitting.interpolate_positions(global_measured_position_df, a_sample_times)
            measured_positions_dfs_list.append(interpolated_measured_df)

            decoded_positions = a_decoder_decoding_result.most_likely_positions_list[epoch_idx]
            if np.ndim(decoded_positions) > 1:
                ## 2D positions, need to get only the x or get the marginals
                decoded_positions = a_decoder_decoding_result.marginal_x_list[epoch_idx]['most_likely_positions_1D']
                assert np.ndim(decoded_positions) < 2, f" the new decoded positions should now be 1D but instead: np.ndim(decoded_positions): {np.ndim(decoded_positions)}, and np.shape(decoded_positions): {np.shape(decoded_positions)}"
            assert len(a_sample_times) == len(decoded_positions), f"len(a_sample_times): {len(a_sample_times)} == len(decoded_positions): {len(decoded_positions)}"
            

            interpolated_measured_x = interpolated_measured_df['x'].to_numpy()

            timebin_is_valid_for_both: NDArray = np.logical_and(np.isfinite(decoded_positions), np.isfinite(interpolated_measured_x)) # bool array of valid indicies for both
            num_valid_timebins = np.sum(timebin_is_valid_for_both)
            
            a_valid_sample_times = a_sample_times[timebin_is_valid_for_both]
            a_valid_decoded_positions = decoded_positions[timebin_is_valid_for_both]
            a_valid_interpolated_measured_x = interpolated_measured_x[timebin_is_valid_for_both]
            assert len(a_valid_sample_times) == len(a_valid_decoded_positions), f"len(a_valid_sample_times): {len(a_valid_sample_times)} == len(a_valid_decoded_positions): {len(a_valid_decoded_positions)}"
            

            ## one for each decoder:
            # test_decoded_positions_df = pd.DataFrame({'t':a_sample_times, 'x':decoded_positions, 'is_timebin_valid_for_both': timebin_is_valid_for_both})

            ## only get valid bins: do we need all bins?
            test_decoded_positions_df = pd.DataFrame({'t':a_valid_sample_times, 'x':a_valid_decoded_positions})
            center_epoch_time = np.mean(a_sample_times)

            decoded_positions_df_list.append(test_decoded_positions_df)
            # compute the diff error:
            # mean_squared_error(y_true, y_pred)
            

            if (num_valid_timebins == 0):
                print(f'WARN: encountered epoch[{epoch_idx}] with no valid timebins.')
                if not should_drop_epochs_with_no_valid_timebins:
                    print(f'\tskipping because `should_drop_epochs_with_no_valid_timebins = True`. Output will have diff number of epochs than input!')
                    pass
                else:
                    print(f'\tfilling with NaN and continuing.')
                    ## have to fill with something!
                    test_decoded_measured_diff: float = np.nan
                    test_decoded_measured_diff_cm: float = np.nan
                    
            else:
                ## has some timebins that are valid:

                # test_decoded_measured_diff_df = (interpolated_measured_df[['x']] - pd.DataFrame({'x':v.most_likely_positions_list[epoch_idx]})) ## error at each point
                # test_decoded_measured_diff: float = mean_squared_error(interpolated_measured_df['x'].to_numpy(), decoded_positions) # single float error
                test_decoded_measured_diff: float = mean_squared_error(a_valid_interpolated_measured_x, a_valid_decoded_positions) # single float error
                test_decoded_measured_diff_cm: float = np.sqrt(test_decoded_measured_diff)

            decoded_measured_diff_df.append((center_epoch_time, test_decoded_measured_diff, test_decoded_measured_diff_cm))

            ## END FOR
        decoded_measured_diff_df: pd.DataFrame = pd.DataFrame(decoded_measured_diff_df, columns=['t', 'sq_err', 'err_cm']) # convert list of tuples to a single df

        # return measured_positions_dfs_list, decoded_positions_df_list, decoded_measured_diff_df
        return MeasuredDecodedPositionComparison(measured_positions_dfs_list, decoded_positions_df_list, decoded_measured_diff_df)

    @classmethod
    def build_measured_decoded_position_comparison(cls, test_laps_decoder_results_dict: Dict[str, DecodedFilterEpochsResult], global_measured_position_df: pd.DataFrame):
        """ compare the decoded most-likely-positions and the measured positions interpolated to the same time bins.
        
        from sklearn.metrics import mean_squared_error
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult

        # Interpolated measured position DataFrame - looks good
        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']) # computation_result.sess.position.to_dataframe()
        test_measured_positions_dfs_dict, test_decoded_positions_df_dict, test_decoded_measured_diff_df_dict = CustomDecodeEpochsResult.build_measured_decoded_position_comparison(test_laps_decoder_results_dict, global_measured_position_df=global_measured_position_df)
        train_measured_positions_dfs_dict, train_decoded_positions_df_dict, train_decoded_measured_diff_df_dict = CustomDecodeEpochsResult.build_measured_decoded_position_comparison(train_laps_decoder_results_dict, global_measured_position_df=global_measured_position_df)


        """
        test_measured_positions_dfs_dict = {}
        test_decoded_positions_df_dict = {}
        test_decoded_measured_diff_df_dict = {}

        for k, a_decoder_decoding_result in test_laps_decoder_results_dict.items():
            # Using `build_single_measured_decoded_position_comparison`
            measured_positions_dfs_list, decoded_positions_df_list, decoded_measured_diff_df = cls.build_single_measured_decoded_position_comparison(a_decoder_decoding_result, global_measured_position_df=global_measured_position_df)
            test_measured_positions_dfs_dict[k] = measured_positions_dfs_list
            test_decoded_positions_df_dict[k] = decoded_positions_df_list
            test_decoded_measured_diff_df_dict[k] = decoded_measured_diff_df

        return test_measured_positions_dfs_dict, test_decoded_positions_df_dict, test_decoded_measured_diff_df_dict

    @function_attributes(short_name=None, tags=['laps', 'performance', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 11:55', related_items=['plot_estimation_correctness_with_raw_data'])
    @classmethod
    def build_lap_bin_by_bin_performance_analysis_df(cls, test_all_directional_laps_decoder_result: "CustomDecodeEpochsResult", active_pf_2D, should_include_decoded_pos_columns: bool=False, debug_print:bool=False) -> pd.DataFrame:
        """ 
        Adds columns:
            ['P_Long', 'P_Short', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'is_LR_dir', 'start', 'stop', 'duration', 'P_LR', 'P_RL', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'x_meas', 'y_meas', 'binned_x_meas', 'is_Long', 'estimation_correctness_track_ID', 'estimation_correctness_track_dir', 'correct_decoder_idx', 'estimation_correctness_decoder_idx']
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_run_rigorous_decoder_performance_assessment, build_lap_bin_by_bin_performance_analysis_df, plot_estimation_correctness_with_raw_data
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, MeasuredDecodedPositionComparison, DecodedFilterEpochsResult

            ## INPUTS: curr_active_pipeline, all_directional_pf1D_Decoder

            active_laps_decoding_time_bin_size: float = 0.025
            ## INPUTS: active_laps_decoding_time_bin_size: float = 0.025
            _out_subset_decode_results = _perform_run_rigorous_decoder_performance_assessment(curr_active_pipeline=curr_active_pipeline, included_neuron_IDs=None, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size)
            ## extract results:
            complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_laps_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results = _out_subset_decode_results
            (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
            test_all_directional_laps_decoder_result: CustomDecodeEpochsResult = deepcopy(test_all_directional_laps_decoder_result)
            active_pf_2D = deepcopy(all_directional_pf1D_Decoder) # active_pf_2D: used for binning position columns # active_pf_2D: used for binning position columns
            epochs_track_identity_marginal_df = build_lap_bin_by_bin_performance_analysis_df(test_all_directional_laps_decoder_result, active_pf_2D)
            epochs_track_identity_marginal_df

            # Example Plotting Viz:
            plot_estimation_correctness_with_raw_data(epochs_track_identity_marginal_df, 'binned_x_meas', 'estimation_correctness_track_ID')

        
        """
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, MeasuredDecodedPositionComparison, DecodedFilterEpochsResult
        from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
        from pyphocorehelpers.indexing_helpers import reorder_columns_relative
        
        min_number_ybins_to_consider_as_spatial: int = 6 ## prevents detecting pseudo2D y-bins (array([0, 1, 2, 3, 4])) incorrectly as spatial
        is_2D_dim_decoder: bool = ((active_pf_2D.ybin is not None) and (len(active_pf_2D.ybin) > min_number_ybins_to_consider_as_spatial))
        

        _position_col_names = []

        all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(test_all_directional_laps_decoder_result.decoder_result)

        measured_decoded_position_comparion: MeasuredDecodedPositionComparison = deepcopy(test_all_directional_laps_decoder_result.measured_decoded_position_comparion) ## provides actual measured positions at each of these bins
        
        active_filter_epochs: pd.DataFrame = deepcopy(all_directional_laps_filter_epochs_decoder_result.active_filter_epochs)

        Assert.same_length(active_filter_epochs, measured_decoded_position_comparion.measured_positions_dfs_list)
        Assert.same_length(active_filter_epochs, measured_decoded_position_comparion.decoded_positions_df_list)

        measured_positions_dfs_list = [a_measured_pos_df.rename(columns={'x': 'x_meas', 'y': 'y_meas'}) for a_measured_pos_df in measured_decoded_position_comparion.measured_positions_dfs_list]
        if should_include_decoded_pos_columns:
            decoded_positions_df_list = [a_decoded_pos_df.rename(columns={'x': 'x_decode', 'y': 'y_decode'}) for a_decoded_pos_df in measured_decoded_position_comparion.decoded_positions_df_list]

            
        ## INPUTS: test_all_directional_decoder_result
        (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                                        additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])

        ## adds marginal columns:
        epochs_directional_marginals, epochs_directional_all_epoch_bins_marginal, epochs_most_likely_direction_from_decoder, epochs_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
        epochs_directional_marginal_p_x_given_n_list: List[NDArray] = [v['p_x_given_n'] for v in epochs_directional_marginals] ## List[DynamicContainer]
        Assert.same_length(active_filter_epochs, epochs_directional_marginal_p_x_given_n_list) 

        epochs_track_identity_marginals, epochs_track_identity_all_epoch_bins_marginal, epochs_most_likely_track_identity_from_decoder, epochs_is_most_likely_track_identity_Long = laps_track_identity_marginals_tuple
        epochs_track_identity_marginal_p_x_given_n_list: List[NDArray] = [v['p_x_given_n'] for v in epochs_track_identity_marginals] ## List[DynamicContainer]
        Assert.same_length(active_filter_epochs, epochs_track_identity_marginal_p_x_given_n_list)
            
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = laps_non_marginalized_decoder_marginals_tuple
        epochs_non_marginalized_decoder_marginals_p_x_given_n_list: List[NDArray] = [v['p_x_given_n'] for v in non_marginalized_decoder_marginals] ## List[DynamicContainer]
        Assert.same_length(active_filter_epochs, epochs_non_marginalized_decoder_marginals_p_x_given_n_list)
        
        ## NEW COMPUTE MARGINALS:
        PandasHelpers.require_columns(active_filter_epochs, required_columns=['long_LR', 'long_RL', 'short_LR', 'short_RL', 'P_LR', 'P_RL', 'P_Long', 'P_Short'], print_missing_columns=True)


        # DecodedFilterEpochsResult.perform_compute_marginals(filter_epochs_decoder_result=epochs_directional_marginal_p_x_given_n_list, filter_epochs=active_filter_epochs, epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t', additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
        
        ## Begin Building Dataframes:
        lambda_full_correctly_sized_value_fn = lambda p_x_given_n, x: np.full((p_x_given_n.T.shape[0], 1), x)
        epochs_track_identity_marginal_df_list: List[pd.DataFrame] = [pd.DataFrame(np.hstack([track_identity_p_x_given_n.T,
                                                                                            lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, i), lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.lap_id), lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.maze_id),
                                                                                            lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.lap_dir),
                                                                                            lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.is_LR_dir),
                                                                                            lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.start),lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.stop),lambda_full_correctly_sized_value_fn(track_identity_p_x_given_n, a_lap_tuple.duration),
                                                                                            ]), columns=['P_Long', 'P_Short', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'is_LR_dir', 'start', 'stop', 'duration']) for i, (a_lap_tuple, track_identity_p_x_given_n) in enumerate(zip(active_filter_epochs.itertuples(index=True, name='LapTuple'), epochs_track_identity_marginal_p_x_given_n_list))]

        epochs_track_identity_marginal_df_list = [pd.concat([a_df, pd.DataFrame(directional_marginal_p_x_given_n.T, columns=['P_LR', 'P_RL'])], axis='columns', ignore_index=False) for a_df, directional_marginal_p_x_given_n in zip(epochs_track_identity_marginal_df_list, epochs_directional_marginal_p_x_given_n_list)]
        epochs_track_identity_marginal_df_list = [pd.concat([a_df, pd.DataFrame(non_marginalized_p_x_given_n.T, columns=['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL'])], axis='columns', ignore_index=False) for a_df, non_marginalized_p_x_given_n in zip(epochs_track_identity_marginal_df_list, epochs_non_marginalized_decoder_marginals_p_x_given_n_list)]
        

        # epochs_track_identity_marginal_df_list = [pd.concat([a_df, a_measured_pos_df[['x_meas', 'y_meas']]], axis='columns', ignore_index=False) for i, (a_df, a_measured_pos_df, directional_marginal_p_x_given_n) in enumerate(zip(epochs_track_identity_marginal_df_list, measured_positions_dfs_list, epochs_directional_marginal_p_x_given_n_list))]
        epochs_track_identity_marginal_df_list = [pd.concat([a_df, a_measured_pos_df[['x_meas', 'y_meas']]], axis='columns', ignore_index=False) for a_df, a_measured_pos_df in zip(epochs_track_identity_marginal_df_list, measured_positions_dfs_list)]
        _position_col_names.extend(['x_meas', 'y_meas'])
        
        if should_include_decoded_pos_columns:
            if is_2D_dim_decoder:
                decoder_col_names = ['x_decode', 'y_decode']
            else:
                decoder_col_names = ['x_decode']
                 
            epochs_track_identity_marginal_df_list = [pd.concat([a_df, a_decoded_pos_df[decoder_col_names]], axis='columns', ignore_index=False) for a_df, a_decoded_pos_df in zip(epochs_track_identity_marginal_df_list, decoded_positions_df_list)]
            _position_col_names.extend(decoder_col_names)

        # .itertuples(index=True, name='MeasuredPositionTuple')

        Assert.same_length(active_filter_epochs, epochs_track_identity_marginal_df_list)

        # epochs_track_identity_marginal_df_list

        epochs_track_identity_marginal_df: pd.DataFrame = pd.concat(epochs_track_identity_marginal_df_list).reset_index(drop=True) #.astype({'lap_idx': int,'lap_id': int,'maze_id': int,'lap_dir': int,'is_LR_dir':bool})
        epochs_track_identity_marginal_df = epochs_track_identity_marginal_df.astype({'lap_idx': int,'lap_id': int,'maze_id': int,'lap_dir': int,'is_LR_dir':bool})
        # Assert.same_length(active_filter_epochs, epochs_track_identity_marginal_df_list)



        if is_2D_dim_decoder:
            epochs_track_identity_marginal_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(deepcopy(epochs_track_identity_marginal_df), bin_values=(deepcopy(active_pf_2D.xbin), deepcopy(active_pf_2D.ybin)),
                                                                                                                        position_column_names = ('x_meas', 'y_meas'),  binned_column_names = ('binned_x', 'binned_y'),
                                                                                                                        force_recompute=False, debug_print=True)
            _position_col_names.extend(['binned_x_meas', 'binned_y_meas'])
        else:
            # x-only:
            epochs_track_identity_marginal_df, (xbin, ), bin_infos = build_df_discretized_binned_position_columns(deepcopy(epochs_track_identity_marginal_df), bin_values=(deepcopy(active_pf_2D.xbin),),
                                                                                                                    position_column_names = ('x_meas',),  binned_column_names = ('binned_x_meas', ),
                                                                                                                    force_recompute=False, debug_print=True)
            _position_col_names.extend(['binned_x_meas'])


        if should_include_decoded_pos_columns:
            if is_2D_dim_decoder:
                epochs_track_identity_marginal_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(deepcopy(epochs_track_identity_marginal_df), bin_values=(deepcopy(active_pf_2D.xbin), deepcopy(active_pf_2D.ybin)),
                                                                                                                    position_column_names = ('x_decode', 'y_decode'),  binned_column_names = ('binned_x_decode', 'binned_y_decode'),
                                                                                                                    force_recompute=False, debug_print=True)
                _position_col_names.extend(['binned_x_decode', 'binned_y_decode'])
            else:
                # x-only:
                epochs_track_identity_marginal_df, (xbin, ), bin_infos = build_df_discretized_binned_position_columns(deepcopy(epochs_track_identity_marginal_df), bin_values=(deepcopy(active_pf_2D.xbin),),
                                                                                                                            position_column_names = ('x_decode',),  binned_column_names = ('binned_x_decode', ),
                                                                                                                            force_recompute=False, debug_print=True)
                _position_col_names.extend(['binned_x_decode'])
        
        # Add ground-truth/performance comparisons ___________________________________________________________________________ #
        epochs_track_identity_marginal_df['is_Long'] = (epochs_track_identity_marginal_df['lap_dir'] > 0)
        # if epochs_track_identity_marginal_df['is_Long'] assign epochs_track_identity_marginal_df['P_Long'], else assign assign epochs_track_identity_marginal_df['P_Short']
        # epochs_track_identity_marginal_df['estimation_correctness_track_ID'] = epochs_track_identity_marginal_df['P_Long'] * epochs_track_identity_marginal_df['is_Long'].astype(float)
        epochs_track_identity_marginal_df['estimation_correctness_track_ID'] = np.where(
            epochs_track_identity_marginal_df['is_Long'],
            epochs_track_identity_marginal_df['P_Long'],
            epochs_track_identity_marginal_df['P_Short']
        )
        
        epochs_track_identity_marginal_df['estimation_correctness_track_dir'] = np.where(
            epochs_track_identity_marginal_df['is_LR_dir'],
            epochs_track_identity_marginal_df['P_LR'],
            epochs_track_identity_marginal_df['P_RL']
        )
        
        ## switch between the four columns based on the value of 'correct_decoder_idx' using `np.select(...)`:
        epochs_track_identity_marginal_df['correct_decoder_idx'] = (2 * epochs_track_identity_marginal_df['maze_id']) + epochs_track_identity_marginal_df['lap_dir'] # ['correct_decoder_idx'] is [0, 1, 2, 3]

        # Define the conditions
        conditions = [
            epochs_track_identity_marginal_df['correct_decoder_idx'] == 0,
            epochs_track_identity_marginal_df['correct_decoder_idx'] == 1,
            epochs_track_identity_marginal_df['correct_decoder_idx'] == 2,
            epochs_track_identity_marginal_df['correct_decoder_idx'] == 3,
        ]

        # Define the corresponding choices
        choices = [
            epochs_track_identity_marginal_df['P_Long_LR'],
            epochs_track_identity_marginal_df['P_Long_RL'],
            epochs_track_identity_marginal_df['P_Short_LR'],
            epochs_track_identity_marginal_df['P_Short_RL'],
        ]

        # Use np.select to assign values based on the conditions
        epochs_track_identity_marginal_df['estimation_correctness_decoder_idx'] = np.select(conditions, choices, default=np.nan)

        if debug_print:
            print(list(epochs_track_identity_marginal_df.columns)) # ['P_Long', 'P_Short', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'is_LR_dir', 'start', 'stop', 'duration', 'x_meas', 'y_meas', 'binned_x_meas', 'is_Long', 'estimation_correctness_track_ID']
        # ['P_Long', 'P_Short', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'is_LR_dir', 'start', 'stop', 'duration', 'P_LR', 'P_RL', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'x_meas', 'y_meas', 'binned_x_meas', 'is_Long', 'estimation_correctness_track_ID', 'estimation_correctness_track_dir', 'correct_decoder_idx', 'estimation_correctness_decoder_idx']

        if debug_print:
            print(f'_position_col_names: {_position_col_names}')
        
        # move the specified columns to the start of the df:
        epochs_track_identity_marginal_df = reorder_columns_relative(epochs_track_identity_marginal_df, column_names=['start', 'stop', 'duration', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'correct_decoder_idx', *_position_col_names, 'is_LR_dir', 'is_Long'], relative_mode='start')

        # move performance columns to very end:
        # epochs_track_identity_marginal_df = reorder_columns_relative(epochs_track_identity_marginal_df, column_names=list(filter(lambda column: column.startswith('estimation_correctness_'), epochs_track_identity_marginal_df.columns)), relative_mode='end')
        
        return epochs_track_identity_marginal_df
        
    @function_attributes(short_name=None, tags=['laps', 'performance', 'position'], input_requires=[], output_provides=[], uses=['cls.build_lap_bin_by_bin_performance_analysis_df'], used_by=[], creation_date='2024-12-31 11:55', related_items=['plot_estimation_correctness_with_raw_data'])
    def get_lap_bin_by_bin_performance_analysis_df(self, active_pf_2D, should_include_decoded_pos_columns: bool=False, debug_print:bool=False) -> pd.DataFrame:
        """ 
        Adds columns:
            ['P_Long', 'P_Short', 'lap_idx', 'lap_id', 'maze_id', 'lap_dir', 'is_LR_dir', 'start', 'stop', 'duration', 'P_LR', 'P_RL', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'x_meas', 'y_meas', 'binned_x_meas', 'is_Long', 'estimation_correctness_track_ID', 'estimation_correctness_track_dir', 'correct_decoder_idx', 'estimation_correctness_decoder_idx']
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_run_rigorous_decoder_performance_assessment, build_lap_bin_by_bin_performance_analysis_df, plot_estimation_correctness_with_raw_data
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, MeasuredDecodedPositionComparison, DecodedFilterEpochsResult

            ## INPUTS: curr_active_pipeline, all_directional_pf1D_Decoder

            active_laps_decoding_time_bin_size: float = 0.025
            ## INPUTS: active_laps_decoding_time_bin_size: float = 0.025
            _out_subset_decode_results = _perform_run_rigorous_decoder_performance_assessment(curr_active_pipeline=curr_active_pipeline, included_neuron_IDs=None, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size)
            ## extract results:
            complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_laps_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results = _out_subset_decode_results
            (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
            test_all_directional_laps_decoder_result: CustomDecodeEpochsResult = deepcopy(test_all_directional_laps_decoder_result)
            active_pf_2D = deepcopy(all_directional_pf1D_Decoder) # active_pf_2D: used for binning position columns # active_pf_2D: used for binning position columns
            epochs_track_identity_marginal_df = build_lap_bin_by_bin_performance_analysis_df(test_all_directional_laps_decoder_result, active_pf_2D)
            epochs_track_identity_marginal_df

            # Example Plotting Viz:
            plot_estimation_correctness_with_raw_data(epochs_track_identity_marginal_df, 'binned_x_meas', 'estimation_correctness_track_ID')

        
        """
        return self.build_lap_bin_by_bin_performance_analysis_df(test_all_directional_laps_decoder_result=self, active_pf_2D=active_pf_2D, should_include_decoded_pos_columns=should_include_decoded_pos_columns, debug_print=debug_print)
    
        


@function_attributes(short_name='custom_decode_epochs', tags=['USEFUL', 'decode', 'pure'], input_requires=[], output_provides=[], uses=[], used_by=['_do_custom_decode_epochs_dict', '_do_train_test_split_decode_and_evaluate'], creation_date='2024-04-05 11:59', related_items=[])
def _do_custom_decode_epochs(global_spikes_df: pd.DataFrame,  global_measured_position_df: pd.DataFrame, pfND_Decoder: BasePositionDecoder, epochs_to_decode_df: pd.DataFrame, decoding_time_bin_size: float, debug_print=False, **kwargs) -> CustomDecodeEpochsResult: #Tuple[MeasuredDecodedPositionComparison, DecodedFilterEpochsResult]:
    """
    Do a single position decoding using a single decoder for a single set of epochs
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _do_custom_decode_epochs
    
    """
    ## INPUTS: global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size
    decoder_result: DecodedFilterEpochsResult = pfND_Decoder.decode_specific_epochs(spikes_df=deepcopy(global_spikes_df), filter_epochs=deepcopy(epochs_to_decode_df), decoding_time_bin_size=decoding_time_bin_size, debug_print=debug_print)
    # Interpolated measured position DataFrame - looks good
    ## INPUTS: test_all_directional_decoder_result, all_directional_pf1D_Decoder
    test_all_directional_decoder_result: CustomDecodeEpochsResult = CustomDecodeEpochsResult.init_from_single_decoder_decoding_result_and_measured_pos_df(decoder_result, global_measured_position_df=global_measured_position_df,
                                                                                                                                                          pfND_Decoder=deepcopy(pfND_Decoder), debug_print=debug_print,
                                                                                                                                                          )
    # epochs_bin_by_bin_performance_analysis_df = test_all_directional_decoder_result.epochs_bin_by_bin_performance_analysis_df ## UNPACK WITH: 
    
    return test_all_directional_decoder_result


@function_attributes(short_name='custom_decode_epochs_dict', tags=['decode', 'general', 'epoch'], input_requires=[], output_provides=[], uses=['_do_custom_decode_epochs'], used_by=[], creation_date='2024-04-05 11:59', related_items=[])
def _do_custom_decode_epochs_dict(global_spikes_df: pd.DataFrame, global_measured_position_df: pd.DataFrame, pfND_Decoder_dict: Dict[str, BasePositionDecoder], epochs_to_decode_dict: Dict[str, pd.DataFrame], decoding_time_bin_size: float, decoder_and_epoch_keys_independent:bool=True, **kwargs) -> Union[Dict[decoder_name, CustomDecodeEpochsResult], Dict[epoch_split_key, Dict[decoder_name, CustomDecodeEpochsResult]]]:
    """
    Do a position decoding for a each set of epochs (`epochs_to_decode_dict`) and each decoder from a dict of decoders (`pfND_Decoder_dict`)

    
    decoder_and_epoch_keys_independent: bool - if False, it indicates that pf1D_Decoder_dict and epochs_to_decode_dict share the same keys, meaning they are paired. If True, they will be treated as independent and the epochs will be decoded by all provided decoders.
    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _do_custom_decode_epochs_dict

        active_laps_decoding_time_bin_size: float = 0.75

        global_spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline)
        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']) # computation_result.sess.position.to_dataframe()


        ## INPUTS: flat_epochs_to_decode_dict, active_laps_decoding_time_bin_size
        train_decoder_results_dict: Dict[epoch_split_key, Dict[decoder_name, CustomDecodeEpochsResult]] = _do_custom_decode_epochs_dict(global_spikes_df=global_spikes_df, global_measured_position_df=global_measured_position_df,
                                                                                                                                        pf1D_Decoder_dict=train_lap_specific_pf1D_Decoder_dict,
                                                                                                                                        epochs_to_decode_dict=train_epochs_dict,
                                                                                                                                        decoding_time_bin_size=active_laps_decoding_time_bin_size,
                                                                                                                                        decoder_and_epoch_keys_independent=False)


        test_decoder_results_dict: Dict[epoch_split_key, Dict[decoder_name, CustomDecodeEpochsResult]] = _do_custom_decode_epochs_dict(global_spikes_df=global_spikes_df, global_measured_position_df=global_measured_position_df,
                                                                                                                                        pf1D_Decoder_dict=train_lap_specific_pf1D_Decoder_dict,
                                                                                                                                        epochs_to_decode_dict=test_epochs_dict, 
                                                                                                                                        decoding_time_bin_size=active_laps_decoding_time_bin_size,
                                                                                                                                        decoder_and_epoch_keys_independent=False)

                                                                                                                                        

    """
    if (not decoder_and_epoch_keys_independent):
        assert np.all(np.isin(pfND_Decoder_dict.keys(), epochs_to_decode_dict.keys())), f"decoder_and_epoch_keys_independent == False but pf1D_Decoder_dict.keys(): {list(pfND_Decoder_dict.keys())} != epochs_to_decode_dict.keys(): {list(epochs_to_decode_dict.keys())}"


    ## INPUTS: global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size

    ## Output should be {epoch_name, {decoder_name, RESULT}}
    # final_decoder_results_dict: Dict[epoch_split_key, Dict[decoder_name, DecodedFilterEpochsResult]] = {str(an_epoch_name):{} for an_epoch_name in epochs_to_decode_dict.keys()}

    final_decoder_results_dict: Dict[epoch_split_key, Dict[decoder_name, CustomDecodeEpochsResult]] = {str(an_epoch_name):{} for an_epoch_name in epochs_to_decode_dict.keys()}

    for a_decoder_name, a_pf1D_Decoder in pfND_Decoder_dict.items():
        for epoch_name, an_epoch_to_decode_df in epochs_to_decode_dict.items():
            if ((not decoder_and_epoch_keys_independent) and (a_decoder_name != epoch_name)):
                continue # skip the non-matching elements
            else:
                full_decoder_result: CustomDecodeEpochsResult = _do_custom_decode_epochs(global_spikes_df=global_spikes_df, global_measured_position_df=global_measured_position_df,
                    pfND_Decoder=a_pf1D_Decoder, epochs_to_decode_df=an_epoch_to_decode_df,
                    decoding_time_bin_size=decoding_time_bin_size, **kwargs)

                # measured_decoded_position_comparion, decoder_result = decoder_result
                final_decoder_results_dict[epoch_name][a_decoder_name] = full_decoder_result

    if (not decoder_and_epoch_keys_independent):
        final_decoder_results_dict: Dict[decoder_name, CustomDecodeEpochsResult] = {k:v[k] for k, v in final_decoder_results_dict.items()} # flatten down

    return final_decoder_results_dict



@function_attributes(short_name=None, tags=['TrainTestSplit', 'decode'], input_requires=[], output_provides=[],
                      uses=['_do_custom_decode_epochs', 'get_proper_global_spikes_df', '_check_result_laps_epochs_df_performance', 'DirectionalPseudo2DDecodersResult'], used_by=['_perform_run_rigorous_decoder_performance_assessment'], creation_date='2024-10-08 02:35', related_items=[])
def _do_train_test_split_decode_and_evaluate(curr_active_pipeline, active_laps_decoding_time_bin_size: float=1.5, force_recompute_directional_train_test_split_result: bool=False, compute_separate_decoder_results: bool=True, included_neuron_IDs=None, debug_print=False):
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _do_train_test_split_decode_and_evaluate

    # active_laps_decoding_time_bin_size: float = 0.25
    active_laps_decoding_time_bin_size: float = 2.5
    complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, all_directional_laps_filter_epochs_decoder_result = _do_train_test_split_decode_and_evaluate(curr_active_pipeline=curr_active_pipeline, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size, force_recompute_directional_train_test_split_result=False)

    (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
    percent_laps_track_identity_estimated_correctly

    """
    from neuropy.core.session.dataSession import Laps
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import PfND
    # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import compute_weighted_correlations
    # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance
    
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
    
    def _add_extra_epochs_df_columns(epochs_df: pd.DataFrame):
        """ captures: global_session, t_start, t_delta, t_end
        """
        epochs_df = epochs_df.sort_values(['start', 'stop', 'label']).reset_index(drop=True) # Sort by columns: 'start' (ascending), 'stop' (ascending), 'label' (ascending)
        epochs_df = epochs_df.drop_duplicates(subset=['start', 'stop', 'label'])
        epochs_df = epochs_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        epochs_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=epochs_df, global_session=deepcopy(global_session), replace_existing=True)
        return epochs_df
        

    directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data.get('TrainTestSplit', None)
    if (directional_train_test_split_result is None) or force_recompute_directional_train_test_split_result:
        ## recompute
        if debug_print:
            print(f"'TrainTestSplit' not computed, recomputing...")
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_train_test_split'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data['TrainTestSplit']
        assert directional_train_test_split_result is not None, f"faiiled even after recomputation"
        if debug_print:
            print('\tdone.')

    training_data_portion: float = directional_train_test_split_result.training_data_portion
    test_data_portion: float = directional_train_test_split_result.test_data_portion
    if debug_print:
        print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

    test_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.test_epochs_dict
    train_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.train_epochs_dict
    train_lap_specific_pfND_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict


    if included_neuron_IDs is not None:
        _alt_directional_train_test_split_result = directional_train_test_split_result.sliced_by_neuron_id(included_neuron_ids=included_neuron_IDs)
        train_lap_specific_pfND_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = _alt_directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict
        

    # OUTPUTS: train_test_split_laps_df_dict

    # Tuple[Tuple[Dict, Dict], Dict[str, BasePositionDecoder], Dict]

    # OUTPUTS: test_epochs_dict, train_epochs_dict, train_lap_specific_pf1D_Decoder_dict

    # MERGED all_directional decoder _____________________________________________________________________________________ #
    # _all_directional_decoder_pf1D_dict: Dict[str, BasePositionDecoder] = deepcopy(train_lap_specific_pf1D_Decoder_dict) # copy the dictionary
    _all_directional_decoder_pf1D_dict: Dict[str, PfND] = {k:v.pf for k, v in deepcopy(train_lap_specific_pfND_Decoder_dict).items()} # copy the dictionary
    all_directional_pf1D: PfND = PfND.build_merged_directional_placefields(_all_directional_decoder_pf1D_dict, debug_print=False) # `PfND.build_merged_directional_placefields`
    all_directional_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

    ## Get test_epochs that are applicable to any decoder:
    all_test_epochs_df: pd.DataFrame = pd.concat([v for v in test_epochs_dict.values()], axis='index')
    all_test_epochs_df = _add_extra_epochs_df_columns(epochs_df=ensure_dataframe(all_test_epochs_df))
    ## OUTPUTS: alt_directional_merged_decoders_result, all_directional_pf1D_Decoder, all_test_epochs_df

    ## Decode Laps - DecodedFilterEpochsResult
    if compute_separate_decoder_results:    
        # INPUTS: flat_epochs_to_decode_dict, active_laps_decoding_time_bin_size
        test_epochs_dict = {k:_add_extra_epochs_df_columns(epochs_df=ensure_dataframe(v)) for k, v in test_epochs_dict.items()}
        # Decoding of the test epochs (what matters):
        test_decoder_results_dict: Dict[types.DecoderName, CustomDecodeEpochsResult] = _do_custom_decode_epochs_dict(global_spikes_df=get_proper_global_spikes_df(curr_active_pipeline), global_measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']),
                                                                                                                                        pfND_Decoder_dict=train_lap_specific_pfND_Decoder_dict,
                                                                                                                                        epochs_to_decode_dict=deepcopy(test_epochs_dict), 
                                                                                                                                        decoding_time_bin_size=active_laps_decoding_time_bin_size,
                                                                                                                                        decoder_and_epoch_keys_independent=False, debug_print=debug_print)
        train_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {k:v.decoder_result for k, v in test_decoder_results_dict.items()}
        # weighted_corr_data_dict = compute_weighted_correlations(train_decoded_results_dict, debug_print=True)
        # # weighted_corr_data_dict
        # train_decoded_wcorr_df = pd.concat(weighted_corr_data_dict)
        # train_decoded_wcorr_df
        # a_decoded_measured_diff_df: pd.DataFrame = test_decoder_results_dict['long_LR'].measured_decoded_position_comparion.decoded_measured_diff_df
        train_decoded_measured_diff_df_dict: Dict[types.DecoderName, pd.DataFrame] = {k:v.measured_decoded_position_comparion.decoded_measured_diff_df for k, v in test_decoder_results_dict.items()}
        _out_separate_decoder_results = (test_decoder_results_dict, train_decoded_results_dict, train_decoded_measured_diff_df_dict)
    else:
        _out_separate_decoder_results = None


    ## Decoding of the test epochs (what matters) for `all_directional_pf1D_Decoder`:
    test_all_directional_decoder_result: CustomDecodeEpochsResult = _do_custom_decode_epochs(global_spikes_df=get_proper_global_spikes_df(curr_active_pipeline), global_measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']),
                                                            pfND_Decoder=all_directional_pf1D_Decoder, epochs_to_decode_df=deepcopy(all_test_epochs_df),
                                                            decoding_time_bin_size=active_laps_decoding_time_bin_size, debug_print=debug_print)
    all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = test_all_directional_decoder_result.decoder_result
    ## INPUTS: test_all_directional_decoder_result
    (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                                    additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])

    # all_directional_laps_filter_epochs_decoder_result.active_filter_epochs
    # all_directional_laps_filter_epochs_decoder_result.filter_epochs
    
    # laps_marginals_df: pd.DataFrame = deepcopy(laps_marginals_df)
    # result_laps_epochs_df: pd.DataFrame = add_groundtruth_information(curr_active_pipeline, a_directional_merged_decoders_result=alt_directional_merged_decoders_result)
    # result_laps_epochs_df: pd.DataFrame = alt_directional_merged_decoders_result.add_groundtruth_information(curr_active_pipeline)
    if 'is_most_likely_track_identity_Long' not in laps_marginals_df.columns:
        laps_marginals_df['is_most_likely_track_identity_Long'] = (laps_marginals_df['P_Long'] >= 0.5)
    if 'is_most_likely_direction_LR' not in laps_marginals_df.columns:
        laps_marginals_df['is_most_likely_direction_LR'] = (laps_marginals_df['P_LR'] >= 0.5)

    ## Uses only 'result_laps_epochs_df'
    complete_decoded_context_correctness_tuple = _check_result_laps_epochs_df_performance(laps_marginals_df)

    ## reverse update:
    # test_all_directional_decoder_result.decoder_result = all_directional_laps_filter_epochs_decoder_result
    
    # Unpack like:
    # (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
    # complete_decoded_context_correctness_tuple
    # percent_laps_track_identity_estimated_correctly
    # percent_laps_direction_estimated_correctly
    # percent_laps_estimated_correctly
    
    ## OUTPUTS: all_directional_pf1D_Decoder
    return complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results



## INPUTS: output_full_directional_merged_decoders_result, sweep_params_idx: int = -1
@function_attributes(short_name=None, tags=['context', 'param_sweep', 'sweep'], input_requires=[], output_provides=[], uses=['DecodedEpochSlicesPaginatedFigureController'], used_by=[], creation_date='2024-04-03 00:00', related_items=['CustomDecodeEpochsResult', '_do_custom_decode_epochs', '_do_custom_decode_epochs_dict', 'CustomDecodeEpochsResult'])
def _show_sweep_result(output_full_directional_merged_decoders_result=None, global_measured_position_df: pd.DataFrame=None, xbin=None, active_context: IdentifyingContext=None, sweep_params_idx: int = -1, sweep_key_name: str="desired_shared_decoding_time_bin_size", debug_print=False, **kwargs):
    """2024-04-03 - Interactively show the lap decoding performance for a single time bin size:

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _show_sweep_result

        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['lap']) # computation_result.sess.position.to_dataframe()
        # sweep_key_name: str="desired_shared_decoding_time_bin_size"
        sweep_key_name: str="desired_laps_decoding_time_bin_size"
        _out_pagination_controller, (all_swept_measured_positions_dfs_dict, all_swept_decoded_positions_df_dict, all_swept_decoded_measured_diff_df_dict) = _show_sweep_result(output_full_directional_merged_decoders_result, global_measured_position_df=global_measured_position_df, sweep_params_idx=0, sweep_key_name=sweep_key_name)
    """
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController


    a_sweep_params_tuple, a_result = tuple(output_full_directional_merged_decoders_result.items())[sweep_params_idx]
    # convert frozenset back to dict
    a_sweep_params_dict = {s[0]:s[1] for i, s in enumerate(a_sweep_params_tuple)} # {'minimum_event_duration': 0.1, 'desired_shared_decoding_time_bin_size': 0.044, 'use_single_time_bin_per_epoch': False}

    # print_keys_if_possible('a_result', a_result, max_depth=1)

    an_all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(a_result.all_directional_laps_filter_epochs_decoder_result)
    # all_directional_decoder_dict: Dict[str, PfND] = deepcopy(a_result.all_directional_decoder_dict)

    ## OUTPUTS: a_sweep_params_dict, an_all_directional_laps_filter_epochs_decoder_result

    # an_all_directional_laps_filter_epochs_decoder_result
    if debug_print:
        print(f'sweep_params_idx: {sweep_params_idx}')
        print(f'a_sweep_params_dict["desired_shared_decoding_time_bin_size"]: {a_sweep_params_dict[sweep_key_name]}')
        print(f'decoding_time_bin_size: {an_all_directional_laps_filter_epochs_decoder_result.decoding_time_bin_size}')


    # Interpolated measured position DataFrame - looks good
    all_swept_measured_positions_dfs_dict, all_swept_decoded_positions_df_dict, all_swept_decoded_measured_diff_df_dict = CustomDecodeEpochsResult.build_measured_decoded_position_comparison({k:deepcopy(v.all_directional_laps_filter_epochs_decoder_result) for k, v in output_full_directional_merged_decoders_result.items()}, global_measured_position_df=global_measured_position_df)
    

    ## OUTPUTS: all_swept_measured_positions_dfs_dict, all_swept_decoded_positions_df_dict, all_swept_decoded_measured_diff_df_dict
    final_context = active_context.adding_context_if_missing(**dict(t_bin=f"{a_sweep_params_dict[sweep_key_name]}s"))

    #### 2024-04-03 - Interactively show the lap decoding performance for a single time bin size:
    _out_pagination_controller = DecodedEpochSlicesPaginatedFigureController.init_from_decoder_data(an_all_directional_laps_filter_epochs_decoder_result.active_filter_epochs,
                                                                                                an_all_directional_laps_filter_epochs_decoder_result,
                                                                                                xbin=xbin, global_pos_df=global_measured_position_df,
                                                                                                active_context=final_context,
                                                                                                **({'a_name': 'an_all_directional_laps_filter_epochs_decoder_result', 'max_subplots_per_page': 20} | kwargs))
    return _out_pagination_controller, (all_swept_measured_positions_dfs_dict, all_swept_decoded_positions_df_dict, all_swept_decoded_measured_diff_df_dict)





# 2024-04-09 - TrainTestLapsSplitting ________________________________________________________________________________ #

@define(slots=False, repr=False)
class TrainTestSplitResult(ComputedResult):
    """ Represents a rigorous test of decoding performance by splitting the lap epochs into Train/Test periods
    
    Usage:
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult

    """
    _VersionedResultMixin_version: str = "2024.04.09_0" # to be updated in your IMPLEMENTOR to indicate its version

    training_data_portion: float = serialized_attribute_field(default=None, is_computable=False, repr=True)
    test_data_portion: float = serialized_attribute_field(default=None, is_computable=False, repr=False)

    test_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = serialized_field(default=None)
    train_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = serialized_field(default=None)
    train_lap_specific_pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(default=None)

    def sliced_by_neuron_id(self, included_neuron_ids: NDArray) -> "TrainTestSplitResult":
        """ refactored out of `self.filtered_by_frate(...)` and `TrackTemplates.determine_decoder_aclus_filtered_by_frate(...)`
        Only `self.train_lap_specific_pf1D_Decoder_dict` is affected
        
        """
        _obj = deepcopy(self) # temporary copy of the object
        _obj.train_lap_specific_pf1D_Decoder_dict = {k:v.get_by_id(included_neuron_ids) for k, v in _obj.train_lap_specific_pf1D_Decoder_dict.items()}
        return _obj
    

def _workaround_validate_has_directional_train_test_split_result(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult

        directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data.get('TrainTestSplit', None)
        test_epochs_dict = directional_train_test_split_result.test_epochs_dict
        train_epochs_dict = directional_train_test_split_result.train_epochs_dict
        train_lap_specific_pf1D_Decoder_dict = directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict


    """
    directional_train_test_split_result = curr_active_pipeline.global_computation_results.computed_data.get('TrainTestSplit', None)
    if directional_train_test_split_result is None:
        return False

    test_epochs_dict = directional_train_test_split_result.test_epochs_dict
    if (test_epochs_dict is None) or (len(test_epochs_dict) == 0):
        return False

    train_epochs_dict = directional_train_test_split_result.train_epochs_dict
    if (train_epochs_dict is None) or (len(train_epochs_dict) == 0):
        return False
    
    train_lap_specific_pf1D_Decoder_dict = directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict
    if (train_lap_specific_pf1D_Decoder_dict is None) or (len(train_lap_specific_pf1D_Decoder_dict) == 0):
        return False
    
    return True



class TrainTestLapsSplitting:
    
    @classmethod
    def decode_using_new_decoders(cls, global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size: float) -> Dict[str, DecodedFilterEpochsResult]:
        """ 


        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import decode_using_new_decoders

            active_laps_decoding_time_bin_size = 0.75
            # AssertionError: Intervals in start_stop_times_arr must be non-overlapping

            ## INPUTS: global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size
            global_spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline)
            test_laps_decoder_results_dict: Dict[str, DecodedFilterEpochsResult] = decode_using_new_decoders(global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size=active_laps_decoding_time_bin_size)

        """
        ## NOTE: they currently only decode the correct test epochs, as in the test epochs corresponding to their train epochs and not others:
        test_laps_decoder_results_dict: Dict[str, DecodedFilterEpochsResult] = {k:v.decode_specific_epochs(spikes_df=deepcopy(global_spikes_df), filter_epochs=deepcopy(test_epochs_dict[k]), decoding_time_bin_size=laps_decoding_time_bin_size, debug_print=False) for k,v in train_lap_specific_pf1D_Decoder_dict.items()}
        return test_laps_decoder_results_dict

    @function_attributes(short_name=None, tags=['split', 'train-test'], input_requires=[], output_provides=[], uses=['split_laps_training_and_test'], used_by=['_split_train_test_laps_data'], creation_date='2024-03-29 22:14', related_items=[])
    @classmethod
    def compute_train_test_split_laps_decoders(cls, directional_laps_results: DirectionalLapsResult, track_templates: TrackTemplates, training_data_portion: float=5.0/6.0,
                                            debug_output_hdf5_file_path=None, debug_plot: bool = False, debug_print: bool = False) -> TrainTestSplitResult: #Tuple[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]], Dict[str, BasePositionDecoder], Dict[str, DynamicContainer]]:
        """ 
        ## Split the lap epochs into training and test periods.
        ##### Ideally we could test the lap decoding error by sampling randomly from the time bins and omitting 1/6 of time bins from the placefield building (effectively the training data). These missing bins will be used as the "test data" and the decoding error will be computed by decoding them and subtracting the actual measured position during these bins.

        ## Get custom decoder that is only trained on a portion of the laps
        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:

        
        Hints/Ref:
            old_directional_lap_name: str = 'maze1_even'
            a_modern_name: str = 'long_LR'

            decoders_dict['long_LR'] # BasePositionDecoder
            decoders_dict['long_LR'].pf.config # PlacefieldComputationParameters

        Usage:

            training_data_portion: float = 5.0/6.0
            test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration
            print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')
            decoders_dict = deepcopy(track_templates.get_decoders_dict())
            (train_test_split_laps_df_dict, train_test_split_laps_epoch_obj_dict), (split_train_test_lap_specific_pf1D_Decoder_dict, split_train_test_lap_specific_pf1D_dict, split_train_test_lap_specific_configs) = compute_train_test_split_laps_decoders(directional_laps_results, track_templates)
            # train_test_split_laps_df_dict

            # train_lap_specific_pf1D_Decoder_dict = split_train_test_lap_specific_pf1D_Decoder_dict

            ## Get test epochs:
            train_epoch_names = [k for k in train_test_split_laps_df_dict.keys() if k.endswith('_train')]
            test_epoch_names = [k for k in train_test_split_laps_df_dict.keys() if k.endswith('_test')]

            train_lap_specific_pf1D_Decoder_dict: Dict[str,BasePositionDecoder] = {k.split('_train', maxsplit=1)[0]:split_train_test_lap_specific_pf1D_Decoder_dict[k] for k in train_epoch_names} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

            # test_epochs_dict: Dict[str,Epoch] = {k:v for k,v in train_test_split_laps_epoch_obj_dict.items() if k.endswith('_test')}

            test_epochs_dict: Dict[str,Epoch] = {k.split('_test', maxsplit=1)[0]:v for k,v in train_test_split_laps_epoch_obj_dict.items() if k.endswith('_test')} # the `k.split('_test', maxsplit=1)[0]` part just gets the original key like 'long_LR'
            test_epochs_dict

        

        """
        from neuropy.core.epoch import Epoch, ensure_dataframe
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder

        test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

        if debug_print:
            print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')


        decoders_dict = deepcopy(track_templates.get_decoders_dict())

        
        # Converting between decoder names and filtered epoch names:
        # {'long':'maze1', 'short':'maze2'}
        # {'LR':'odd', 'RL':'even'}
        long_LR_name, short_LR_name, long_RL_name, short_RL_name = ['maze1_odd', 'maze2_odd', 'maze1_even', 'maze2_even']
        decoder_name_to_session_context_name: Dict[str,str] = dict(zip(track_templates.get_decoder_names(), (long_LR_name, long_RL_name, short_LR_name, short_RL_name))) # {'long_LR': 'maze1_odd', 'long_RL': 'maze1_even', 'short_LR': 'maze2_odd', 'short_RL': 'maze2_even'}
        # session_context_name_to_decoder_name: Dict[str,str] = dict(zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), track_templates.get_decoder_names())) # {'maze1_odd': 'long_LR', 'maze1_even': 'long_RL', 'maze2_odd': 'short_LR', 'maze2_even': 'short_RL'}
        old_directional_names = list(directional_laps_results.directional_lap_specific_configs.keys()) #['maze1_odd', 'maze1_even', 'maze2_odd', 'maze2_even']
        modern_names_list = list(decoders_dict.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        assert len(old_directional_names) == len(modern_names_list), f"old_directional_names: {old_directional_names} length is not equal to modern_names_list: {modern_names_list}"

        # lap_dir_keys = ['LR', 'RL']
        # maze_id_keys = ['long', 'short']
        training_test_suffixes = ['_train', '_test'] ## used in loop

        _written_HDF5_manifest_keys = []

        train_test_split_laps_df_dict: Dict[str, pd.DataFrame] = {} # analagoues to `directional_laps_results.split_directional_laps_dict`
        train_test_split_laps_epoch_obj_dict: Dict[str, Epoch] = {}

        ## Per-Period Outputs
        split_train_test_lap_specific_configs = {}
        split_train_test_lap_specific_pf1D_dict = {} # analagous to `all_directional_decoder_dict` (despite `all_directional_decoder_dict` having an incorrect name, it's actually pfs)
        split_train_test_lap_specific_pf1D_Decoder_dict = {}

        for a_modern_name in modern_names_list:
            ## Loop through each decoder:
            old_directional_lap_name: str = decoder_name_to_session_context_name[a_modern_name] # e.g. 'maze1_even'
            if debug_print:
                print(f'a_modern_name: {a_modern_name}, old_directional_lap_name: {old_directional_lap_name}')
            a_1D_decoder = deepcopy(decoders_dict[a_modern_name])

            # directional_laps_results # DirectionalLapsResult
            a_config = deepcopy(directional_laps_results.directional_lap_specific_configs[old_directional_lap_name])
            # type(a_config) # DynamicContainer

            # type(a_config['pf_params'].computation_epochs) # Epoch
            # a_config['pf_params'].computation_epochs
            a_laps_df: pd.DataFrame = ensure_dataframe(deepcopy(a_config['pf_params'].computation_epochs))
            # ensure non-overlapping first:
            a_laps_training_df, a_laps_test_df = a_laps_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.
            
            if debug_output_hdf5_file_path is not None:
                # Write out to HDF5 file:
                a_possible_hdf5_file_output_prefix: str = 'provided'
                a_laps_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', format='table')
                a_laps_training_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', format='table')
                a_laps_test_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df', format='table')

                _written_HDF5_manifest_keys.extend([f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df'])


            a_training_test_names = [f"{a_modern_name}{a_suffix}" for a_suffix in training_test_suffixes] # ['long_LR_train', 'long_LR_test']
            a_train_epoch_name: str = a_training_test_names[0] # just the train epoch, like 'long_LR_train'
            a_training_test_split_laps_df_dict: Dict[str, pd.DataFrame] = dict(zip(a_training_test_names, (a_laps_training_df, a_laps_test_df))) # analagoues to `directional_laps_results.split_directional_laps_dict`

            # _temp_a_training_test_split_laps_valid_epoch_df_dict: Dict[str,Epoch] = {k:deepcopy(v).get_non_overlapping() for k, v in a_training_test_split_laps_df_dict.items()} ## NOTE: these lose the associated extra columns like 'lap_id', 'lap_dir', etc.
            a_training_test_split_laps_epoch_obj_dict: Dict[str, Epoch] = {k:Epoch(deepcopy(v)).get_non_overlapping() for k, v in a_training_test_split_laps_df_dict.items()} ## NOTE: these lose the associated extra columns like 'lap_id', 'lap_dir', etc.

            train_test_split_laps_df_dict.update(a_training_test_split_laps_df_dict)
            train_test_split_laps_epoch_obj_dict.update(a_training_test_split_laps_epoch_obj_dict)

            a_valid_laps_training_df, a_valid_laps_test_df = ensure_dataframe(a_training_test_split_laps_epoch_obj_dict[a_training_test_names[0]]), ensure_dataframe(a_training_test_split_laps_epoch_obj_dict[a_training_test_names[1]])

            # ## Check Visually - look fine, barely altered
            if debug_plot:
                # fig, ax = debug_draw_laps_train_test_split_epochs(a_laps_df, a_laps_training_df, a_laps_test_df, fignum=0)
                # fig.show()
                fig2, ax = cls.debug_draw_laps_train_test_split_epochs(a_laps_df, a_valid_laps_training_df, a_valid_laps_test_df, fignum=f'Train/Test Split: {a_modern_name}')
                fig2.show()

            if debug_output_hdf5_file_path is not None:
                # Write out to HDF5 file:
                a_possible_hdf5_file_output_prefix: str = 'valid'
                # a_laps_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', format='table')
                a_valid_laps_training_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', format='table')
                a_valid_laps_test_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df', format='table')

                _written_HDF5_manifest_keys.extend([f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df'])


            # uses `a_modern_name`
            a_lap_period_description: str = a_train_epoch_name
            curr_lap_period_epoch_obj: Epoch = a_training_test_split_laps_epoch_obj_dict[a_train_epoch_name]

            a_config_copy = deepcopy(a_config)
            a_config_copy['pf_params'].computation_epochs = curr_lap_period_epoch_obj
            split_train_test_lap_specific_configs[a_lap_period_description] = a_config_copy
            curr_pf1D = a_1D_decoder.pf
            ## Restrict the PfNDs:
            lap_filtered_curr_pf1D: PfND = curr_pf1D.replacing_computation_epochs(deepcopy(curr_lap_period_epoch_obj))
            split_train_test_lap_specific_pf1D_dict[a_lap_period_description] = lap_filtered_curr_pf1D

            ## apply the lap_filtered_curr_pf1D to the decoder:
            a_sliced_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(lap_filtered_curr_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
            split_train_test_lap_specific_pf1D_Decoder_dict[a_lap_period_description] = a_sliced_pf1D_Decoder
        ## ENDFOR a_modern_name in modern_names_list
            
        if debug_print:
            print(list(split_train_test_lap_specific_pf1D_Decoder_dict.keys())) # ['long_LR_train', 'long_RL_train', 'short_LR_train', 'short_RL_train']
        
        ## OUTPUTS: (train_test_split_laps_df_dict, train_test_split_laps_epoch_obj_dict), (split_train_test_lap_specific_pf1D_Decoder_dict, split_train_test_lap_specific_pf1D_dict, split_train_test_lap_specific_configs)


        ## Get test epochs:
        train_epoch_names: List[str] = [k for k in train_test_split_laps_df_dict.keys() if k.endswith('_train')] # ['long_LR_train', 'long_RL_train', 'short_LR_train', 'short_RL_train']
        test_epoch_names: List[str] = [k for k in train_test_split_laps_df_dict.keys() if k.endswith('_test')] # ['long_LR_test', 'long_RL_test', 'short_LR_test', 'short_RL_test']

        ## train_test_split_laps_df_dict['long_LR_test'] != train_test_split_laps_df_dict['long_LR_train'], which is correct
        ## Only the decoders built with the training epochs make any sense:
        train_lap_specific_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = {k.split('_train', maxsplit=1)[0]:split_train_test_lap_specific_pf1D_Decoder_dict[k] for k in train_epoch_names} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

        # Epoch obj mode (loses associated info)
        # test_epochs_obj_dict: Dict[str,Epoch] = {k.split('_test', maxsplit=1)[0]:v for k,v in train_test_split_laps_epoch_obj_dict.items() if k.endswith('_test')} # the `k.split('_test', maxsplit=1)[0]` part just gets the original key like 'long_LR'
        # train_epochs_obj_dict: Dict[str,Epoch] = {k.split('_train', maxsplit=1)[0]:v for k,v in train_test_split_laps_epoch_obj_dict.items() if k.endswith('_train')} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

        # DF mode so they don't lose the associated info:
        test_epochs_dict: Dict[str, pd.DataFrame] = {k.split('_test', maxsplit=1)[0]:v for k,v in train_test_split_laps_df_dict.items() if k.endswith('_test')} # the `k.split('_test', maxsplit=1)[0]` part just gets the original key like 'long_LR'
        train_epochs_dict: Dict[str, pd.DataFrame] = {k.split('_train', maxsplit=1)[0]:v for k,v in train_test_split_laps_df_dict.items() if k.endswith('_train')} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

        ## Now decode the test epochs using the new decoders:
        # ## INPUTS: global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size
        # global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline)
        # test_laps_decoder_results_dict = decode_using_new_decoders(global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size)
        # test_laps_decoder_results_dict

        if debug_output_hdf5_file_path is not None:
            print(f'successfully wrote out to: "{debug_output_hdf5_file_path}"')
            print(f'\t_written_HDF5_manifest_keys: {_written_HDF5_manifest_keys}\n')
            # print(f'\t_written_HDF5_manifest_keys: {",\n".join(_written_HDF5_manifest_keys)}')
            
        return TrainTestSplitResult(is_global=True, training_data_portion=training_data_portion, test_data_portion=test_data_portion,
                             test_epochs_dict=test_epochs_dict, train_epochs_dict=train_epochs_dict,
                             train_lap_specific_pf1D_Decoder_dict=train_lap_specific_pf1D_Decoder_dict)
        

        # return (train_epochs_dict, test_epochs_dict), train_lap_specific_pf1D_Decoder_dict, split_train_test_lap_specific_configs

    @classmethod
    def interpolate_positions(cls, df: pd.DataFrame, sample_times: NDArray, time_column_name: str = 't') -> pd.DataFrame:
        """
        Interpolates position data to new sample times using SciPy's interp1d.

        Parameters:
        df (pd.DataFrame): Original DataFrame with position columns.
        sample_times (NDArray): Array of new sample times at which to interpolate.
        time_column_name (str): Name of the time column in df.

        Returns:
        pd.DataFrame: New DataFrame with interpolated positional data.
        """
        from scipy.interpolate import interp1d

        # Drop any NaNs in the DataFrame to avoid issues with interpolation
        df = df.dropna(subset=[time_column_name, 'x', 'y'])

        # Extract the column data for interpolation
        times = df[time_column_name].values
        x_positions = df['x'].values
        y_positions = df['y'].values
        # If you have 'z' positions as well, extract them too.

        # Create interpolation functions for each position axis
        fx = interp1d(times, x_positions, kind='linear', bounds_error=False, fill_value='extrapolate')
        fy = interp1d(times, y_positions, kind='linear', bounds_error=False, fill_value='extrapolate')
        # If you have z_positions, create an interpolation function for 'z'.

        # Interpolate at new sample times
        new_x_positions = fx(sample_times)
        new_y_positions = fy(sample_times)
        # If you have z_positions: new_z_positions = fz(sample_times)

        # Create a new dataframe with the interpolated values
        interpolated_df = pd.DataFrame({
            time_column_name: sample_times,
            'x': new_x_positions,
            'y': new_y_positions,
            # If you have z_positions, include 'z': new_z_positions
        })

        return interpolated_df



    # DRAWING/Figures ____________________________________________________________________________________________________ #


    ## INPUTS: laps_df, laps_training_df, laps_test_df
    @function_attributes(short_name=None, tags=['matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-29 15:46', related_items=[])
    @classmethod
    def debug_draw_laps_train_test_split_epochs(cls, laps_df, laps_training_df, laps_test_df, fignum=1, fig=None, ax=None, active_context=None, use_brokenaxes_method: bool = False):
        """ Draws the division of the train/test epochs using a matplotlib figure.

        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestLapsSplitting

            fig, ax = TrainTestLapsSplitting.debug_draw_laps_train_test_split_epochs(laps_df, laps_training_df, laps_test_df, fignum=0)
            fig.show()
        """
        from neuropy.core.epoch import Epoch, ensure_dataframe
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, perform_update_title_subtitle
        from neuropy.utils.matplotlib_helpers import draw_epoch_regions


        def _subfn_prepare_epochs_df(laps_test_df: pd.DataFrame) -> Epoch:
            active_filter_epochs = deepcopy(laps_test_df)

            if not 'stop' in active_filter_epochs.columns:
                # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
                active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
                
            if not 'label' in active_filter_epochs.columns:
                # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
                active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

            active_filter_epoch_obj = Epoch(active_filter_epochs)
            return active_filter_epoch_obj


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        laps_Epoch_obj = _subfn_prepare_epochs_df(laps_df)
        laps_training_df_Epoch_obj = _subfn_prepare_epochs_df(laps_training_df)
        laps_test_df_Epoch_obj = _subfn_prepare_epochs_df(laps_test_df)

        if fignum is None:
            if f := plt.get_fignums():
                fignum = f[-1] + 1
            else:
                fignum = 1

        ## Figure Setup:
        if ax is None:
            fig = build_or_reuse_figure(fignum=fignum, fig=fig, fig_idx=0, figsize=(12, 4.2), dpi=None, clear=True, tight_layout=False)
            gs = GridSpec(1, 1, figure=fig)

            if use_brokenaxes_method:
                # `brokenaxes` method: DOES NOT YET WORK!
                from brokenaxes import brokenaxes ## Main brokenaxes import 
                pad_size: float = 0.1
                # [(a_tuple.start, a_tuple.stop) for a_tuple in a_test_epoch_df.itertuples(index=False, name="EpochTuple")]
                lap_start_stop_tuples_list = [((a_tuple.start - pad_size), (a_tuple.stop + pad_size)) for a_tuple in ensure_dataframe(laps_Epoch_obj).itertuples(index=False, name="EpochTuple")]
                # ax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05, subplot_spec=gs[0])
                ax = brokenaxes(xlims=lap_start_stop_tuples_list, hspace=.05, subplot_spec=gs[0])
            else:
                ax = plt.subplot(gs[0])

        else:
            # otherwise get the figure from the passed axis
            fig = ax.get_figure()

        # epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(laps_Epoch_obj, ax, facecolor='black', edgecolors=None, labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False, label='laps')
        test_epochs_collection, test_epoch_labels = draw_epoch_regions(laps_test_df_Epoch_obj, ax, facecolor='purple', edgecolors='purple', labels_kwargs=None, defer_render=True, debug_print=True, label='test')
        train_epochs_collection, train_epoch_labels = draw_epoch_regions(laps_training_df_Epoch_obj, ax, facecolor='green', edgecolors='green', labels_kwargs=None, defer_render=False, debug_print=True, label='train')
        ax.autoscale()
        fig.legend()
        # plt.title('Lap epochs divided into separate training and test intervals')
        plt.xlabel('time (sec)')
        plt.ylabel('Lap Epochs')

        # Set window title and plot title
        perform_update_title_subtitle(fig=fig, ax=ax, title_string=f'Lap epochs divided into separate training and test intervals', subtitle_string=f'{fignum}', active_context=active_context, use_flexitext_titles=True)
        
        return fig, ax

    @function_attributes(short_name=None, tags=['matplotlib', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-01 11:00', related_items=[])
    @classmethod
    def _show_decoding_result(cls, laps_decoder_results_dict: Dict[str, DecodedFilterEpochsResult], measured_positions_dfs_dict, decoded_positions_df_dict, a_name: str = 'long_LR', epoch_IDX: int = 2, xbin=None):
        """ Plots the decoding of a single lap epoch, with its most-likely positions and actual behavioral measured positions overlayed as lines.
        Plot a single decoder, single epoch comparison of measured v. decoded position

        Captures: `directional_laps_results` for purposes of xbin

        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestLapsSplitting
            
            fig, curr_ax = TrainTestLapsSplitting._show_decoding_result(test_measured_positions_dfs_dict, test_decoded_positions_df_dict, a_name = 'long_LR', epoch_IDX = 2, xbin=xbin)

        """
        import matplotlib.pyplot as plt
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions


        a_decoder = laps_decoder_results_dict[a_name] # depends only on a_name
        active_posterior = a_decoder.p_x_given_n_list[epoch_IDX]

        print(f'for a_name: {a_name}, epoch_IDX: {epoch_IDX}')
        measured_positions_df: pd.DataFrame = measured_positions_dfs_dict[a_name][epoch_IDX]
        decoded_positions_df: pd.DataFrame = decoded_positions_df_dict[a_name][epoch_IDX]

        time_window_centers = decoded_positions_df['t'].to_numpy()
        active_measured_positions = measured_positions_df['x'].to_numpy() # interpolated positions
        active_most_likely_positions = decoded_positions_df['x'].to_numpy()

        active_decoder_evaluation_df = pd.DataFrame({'t': time_window_centers, 'measured': active_measured_positions, 'decoded': active_most_likely_positions})

        # OUTPUTS: active_decoder_evaluation_df, (time_window_centers, active_measured_positions, active_most_likely_positions), active_posterior

        # active_decoder_evaluation_df = global_measured_position_df

        # ## Get the previously created matplotlib_view_widget figure/ax:
        fig, curr_ax, _return_out_artists_dict = plot_1D_most_likely_position_comparsions(active_decoder_evaluation_df, time_window_centers=time_window_centers, xbin=xbin,
                                                            posterior=active_posterior,
                                                            active_most_likely_positions_1D=active_most_likely_positions,
                                                            # variable_name='x',
                                                            variable_name='measured',
                                                            enable_flat_line_drawing=False,
                                                            ax=None, return_created_artists=True,
                                                        )
        plt.title('decoding performance')

        # # test interpolated v. actual measured positions: ____________________________________________________________________ #
        # fig, curr_ax = plot_1D_most_likely_position_comparsions(global_measured_position_df, time_window_centers=time_window_centers, xbin=xbin,
        #                                                     posterior=active_posterior,
        #                                                     active_most_likely_positions_1D=active_measured_positions, # interpolated are gray
        #                                                     variable_name='x',
        #                                                     # variable_name='measured',
        #                                                     enable_flat_line_drawing=False,
        #                                                     ax=None,
        #                                                 )
        # plt.title('Interp. v. Actual Measured Positions')
        return fig, curr_ax

    @function_attributes(short_name=None, tags=['UNUSED', 'matplotlib', 'helper', 'ChatGPT', 'UNVALIDATED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-01 18:26', related_items=[])
    @classmethod
    def wrap_xaxis_in_subplots(cls, data_x, data_y, width, **kwargs):
        """
        Create a matplotlib figure with stacked subplots where the x-axis is wrapped at a given width.

        Parameters:
        data_x: array-like, the x coordinates of the data points.
        data_y: array-like, the y coordinates of the data points, should be the same length as data_x.
        width: int, the fixed width to wrap the x-axis.
        kwargs: extra keyword arguments passed to the `plot` function.


        Example Usage:

            # Example usage:
            # data_x might represent some periodic data
            data_x = np.linspace(0, 30, 300)
            data_y = np.sin(data_x) + 0.1*np.random.randn(data_x.size)

            wrap_xaxis_in_subplots(data_x, data_y, 10)

        """
        import matplotlib.pyplot as plt
        
        # Calculate the number of subplots needed
        max_x = np.max(data_x)
        num_subplots = int(np.ceil(max_x / width))

        # Create the figure and subplots
        fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 2), sharex=True)

        # Check if we only have one subplot (matplotlib returns an Axes object directly instead of a list if so)
        if num_subplots == 1:
            axs = [axs]
        
        for i in range(num_subplots):
            # Determine the start and end of the x-range for this subplot
            start_x = i * width
            end_x = start_x + width

            # Extract the data for this range
            mask = (data_x >= start_x) & (data_x < end_x)
            sub_data_x = data_x[mask] - start_x  # Shift x data to start at 0
            sub_data_y = data_y[mask]

            # Plot on the appropriate subplot
            axs[i].plot(sub_data_x, sub_data_y, **kwargs)
            axs[i].set_xlim(0, width)
            axs[i].set_ylim(min(data_y), max(data_y))
            axs[i].set_title(f'Range: [{start_x}, {end_x})')

        # Label the shared x-axis
        fig.text(0.5, 0.04, 'X-axis wraped at width ' + str(width), ha='center')

        # Adjust spacing between the plots
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        plt.show()



# ==================================================================================================================== #
# 2024-05-28 - TrialByTrialActivity                                                                                    #
# ==================================================================================================================== #
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity


@define(slots=False, repr=False, eq=False)
class TrialByTrialActivityResult(ComputedResult):
    """ Holds the result computed by `_build_trial_by_trial_activity_metrics`
    Usage:
    
        from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult

        directional_trial_by_trial_activity_result: TrialByTrialActivityResult = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)

        if directional_trial_by_trial_activity_result is not None:
            any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
            active_pf_dt: PfND_TimeDependent = directional_trial_by_trial_activity_result.active_pf_dt
            directional_lap_epochs_dict: Dict[str, Epoch] = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
            directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
            ## OUTPUTS: directional_trial_by_trial_activity_result, directional_active_lap_pf_results_dicts
        else:
            print(f'TrialByTrialActivity is not computed.')

    """
    _VersionedResultMixin_version: str = "2024.05.28_0" # to be updated in your IMPLEMENTOR to indicate its version

    any_decoder_neuron_IDs: NDArray = serialized_field(default=None)
    active_pf_dt: PfND_TimeDependent = serialized_field(default=None)
    directional_lap_epochs_dict: Dict[decoder_name, Epoch] =  serialized_field(default=None)
    directional_active_lap_pf_results_dicts: Dict[decoder_name, TrialByTrialActivity] = serialized_field(default=None)


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
    

    def build_combined_decoded_epoch_z_scored_tuning_map_matrix(self) -> Tuple[NDArray, List[types.DecoderName]]:
        """ Builds a properly stacked `_flat_z_scored_tuning_map_matrix` from the decoder-split versions
        
        _flat_z_scored_tuning_map_matrix, _flat_decoder_identity_arr = a_trial_by_trial_result.build_combined_decoded_epoch_z_scored_tuning_map_matrix() # .shape: (n_epochs, n_neurons, n_pos_bins)
        _flat_z_scored_tuning_map_matrix

        """
        # self.TrialByTrialActivityResult
        ## INPUTS: directional_lap_epochs_dict: Dict[types.DecoderName, Epoch]
        directional_lap_epochs_dict: Dict[types.DecoderName, Epoch] = self.directional_lap_epochs_dict
        ## ensure the decoder-split versions are sorted, then add the ['decoder_name', 'decoder_relative_idx'] columns to them:
        for k, v in directional_lap_epochs_dict.items():
            v._df['decoder_name'] = k
            # ensure each is sorted within its own decoder:
            v._df = v._df.sort_values(['start', 'stop', 'label']).reset_index(drop=True)
            v._df['decoder_relative_idx'] = deepcopy(v._df.index)

        # all_combined_epochs_obj: pd.DataFrame = pd.concat((ensure_dataframe(long_LR_epochs_obj), ensure_dataframe(long_RL_epochs_obj), ensure_dataframe(short_LR_epochs_obj), ensure_dataframe(short_RL_epochs_obj)), axis='index')
        all_combined_epochs_df: pd.DataFrame = pd.concat([ensure_dataframe(v) for v in directional_lap_epochs_dict.values()], axis='index')
        # Sort by columns: 'start' (ascending), 'stop' (ascending), 'label' (ascending)
        all_combined_epochs_df = all_combined_epochs_df.sort_values(['start', 'stop', 'label']).reset_index(drop=True)
        # all_combined_epochs_obj
        
        # Builds a properly stacked `_flat_z_scored_tuning_map_matrix` from the decoder-split versions:
        # INPUTS: a_trial_by_trial_result, all_combined_epochs_df
        _flat_z_scored_tuning_map_matrix = []
        _flat_decoder_identity_arr: List[types.DecoderName] = []
        # _flat_C_trial_by_trial_correlation_matrix = []
        for a_row in all_combined_epochs_df.itertuples():
            ## do I need `aclu_to_matrix_IDX_map` to properly map to the matrix indicies?
            curr_decoder_name: str = str(a_row.decoder_name)
            a_matrix_idx: int = int(a_row.decoder_relative_idx) ## direct
            # curr_aclu_to_matrix_IDX_map = a_trial_by_trial_result.directional_active_lap_pf_results_dicts[curr_decoder_name].aclu_to_matrix_IDX_map
            # a_matrix_idx: int = curr_aclu_to_matrix_IDX_map[int(a_row.decoder_relative_idx)] ## aclu-based        
            a_matrix_out_row = self.directional_active_lap_pf_results_dicts[curr_decoder_name].z_scored_tuning_map_matrix[a_matrix_idx, :, :] # .shape (n_epochs, n_neurons, n_pos_bins) 
            _flat_z_scored_tuning_map_matrix.append(a_matrix_out_row)
            _flat_decoder_identity_arr.append(curr_decoder_name)
            
        _flat_z_scored_tuning_map_matrix = np.stack(_flat_z_scored_tuning_map_matrix, axis=0) # .shape (n_epochs, n_neurons, n_pos_bins)
        ## OUTPUTS: _flat_z_scored_tuning_map_matrix
        return _flat_z_scored_tuning_map_matrix, _flat_decoder_identity_arr
    

    def build_separated_nan_filled_decoded_epoch_z_scored_tuning_map_matrix(self, enable_C_trial_by_trial_correlation_matrix: bool=False) -> Dict[types.DecoderName, TrialByTrialActivity]:
        """ Builds a properly stacked `_flat_z_scored_tuning_map_matrix` from the decoder-split versions
        
        _flat_z_scored_tuning_map_matrix, _flat_decoder_identity_arr = a_trial_by_trial_result.build_combined_decoded_epoch_z_scored_tuning_map_matrix() # .shape: (n_epochs, n_neurons, n_pos_bins)
        _flat_z_scored_tuning_map_matrix

        """
        # self.TrialByTrialActivityResult
        ## INPUTS: directional_lap_epochs_dict: Dict[types.DecoderName, Epoch]
        directional_lap_epochs_dict: Dict[types.DecoderName, Epoch] = self.directional_lap_epochs_dict
        ## ensure the decoder-split versions are sorted, then add the ['decoder_name', 'decoder_relative_idx'] columns to them:
        for k, v in directional_lap_epochs_dict.items():
            v._df['decoder_name'] = k
            # ensure each is sorted within its own decoder:
            v._df = v._df.sort_values(['start', 'stop', 'label']).reset_index(drop=True)
            v._df['decoder_relative_idx'] = deepcopy(v._df.index)

        # all_combined_epochs_obj: pd.DataFrame = pd.concat((ensure_dataframe(long_LR_epochs_obj), ensure_dataframe(long_RL_epochs_obj), ensure_dataframe(short_LR_epochs_obj), ensure_dataframe(short_RL_epochs_obj)), axis='index')
        all_combined_epochs_df: pd.DataFrame = pd.concat([ensure_dataframe(v) for v in directional_lap_epochs_dict.values()], axis='index')
        # Sort by columns: 'start' (ascending), 'stop' (ascending), 'label' (ascending)
        all_combined_epochs_df = all_combined_epochs_df.sort_values(['start', 'stop', 'label']).reset_index(drop=True)        
        all_combined_epochs_df['full_idx'] = deepcopy(all_combined_epochs_df.index.values)
        
        # all_combined_epochs_obj
        n_epochs: int = len(all_combined_epochs_df)
        modified_directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = deepcopy(self.directional_active_lap_pf_results_dicts)
        

        decoder_epoch_matrix_idx_dict = {str(a_decoder_name):all_combined_epochs_df[all_combined_epochs_df['decoder_name'] == str(a_decoder_name)]['full_idx'].values for a_decoder_name, a_TbyT in modified_directional_active_lap_pf_results_dicts.items()}
        # Builds a properly stacked `_flat_z_scored_tuning_map_matrix` from the decoder-split versions:
        # INPUTS: a_trial_by_trial_result, all_combined_epochs_df        
        for a_decoder_name, a_TbyT in modified_directional_active_lap_pf_results_dicts.items():
            # z_scored_tuning_map_matrix: ('n_epochs', 'n_neurons', 'n_pos_bins')
            _orig_z_scored_tuning_map_matrix = deepcopy(a_TbyT.z_scored_tuning_map_matrix)
            _orig_n_epochs, _orig_n_neurons, _orig_n_pos_bins = np.shape(_orig_z_scored_tuning_map_matrix)
            a_TbyT.z_scored_tuning_map_matrix = np.full(shape=(n_epochs, _orig_n_neurons, _orig_n_pos_bins), fill_value=np.nan) # fill with NaN
            a_TbyT.z_scored_tuning_map_matrix[decoder_epoch_matrix_idx_dict[a_decoder_name], :, :] = _orig_z_scored_tuning_map_matrix ## assign the matrix to the rows
            if enable_C_trial_by_trial_correlation_matrix:
                # C_trial_by_trial_correlation_matrix: ('n_neurons', 'n_epochs', 'n_epochs')
                _orig_C_trial_by_trial_correlation_matrix = deepcopy(a_TbyT.C_trial_by_trial_correlation_matrix)
                a_TbyT.C_trial_by_trial_correlation_matrix = np.full(shape=(_orig_n_neurons, n_epochs, n_epochs), fill_value=np.nan) # fill with NaN
                a_TbyT.C_trial_by_trial_correlation_matrix[:, np.squeeze(decoder_epoch_matrix_idx_dict[a_decoder_name]), np.squeeze(decoder_epoch_matrix_idx_dict[a_decoder_name])] = _orig_C_trial_by_trial_correlation_matrix ## assign the matrix to the rows
            
        ## OUTPUTS: z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix
        return modified_directional_active_lap_pf_results_dicts
    

    @classmethod    
    def determine_neuron_stability(cls, stability_df: pd.DataFrame, minimum_one_point_stability: float = 0.6, zero_point_stability: float = 0.1):
        """ try to determine the LxC/SxC remapping cells from their stability 
        
        neuron_group_split_stability_dfs_tuple = self.determine_neuron_stability(stability_df=stability_df)
        appearing_stability_df, disappearing_stability_df, appearing_or_disappearing_stability_df, stable_both_stability_df, stable_neither_stability_df, stable_long_stability_df, stable_short_stability_df = neuron_group_split_stability_dfs_tuple
        neuron_group_split_stability_aclus_tuple = [sort(a_df.aclu.values) for a_df in neuron_group_split_stability_dfs_tuple]
        appearing_aclus, disappearing_aclus, appearing_or_disappearing_aclus, stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus = neuron_group_split_stability_aclus_tuple
        
        """
        from numpy import sort
        
        stability_df['long_best'] = stability_df[['long_LR', 'long_RL']].max(axis=1, skipna=True)
        stability_df['short_best'] = stability_df[['short_LR', 'short_RL']].max(axis=1, skipna=True)
        stability_df = stability_df.sort_values(['long_best'], ascending=True)

        ## find effectively zero (unstable/no-pf) cells:    
        unstable_long = (stability_df['long_best'] <= zero_point_stability)
        unstable_short = (stability_df['short_best'] <= zero_point_stability)

        ## NOTE that `stable_*` is not the negation of `unstable_*` because they require a minimum `one_point_stability`
        stable_long = (stability_df['long_best'] >= minimum_one_point_stability)
        stable_short = (stability_df['short_best'] >= minimum_one_point_stability)

        # stable_long = np.logical_not(unstable_long)
        # stable_short = np.logical_not(unstable_short)

        # is_appearing = np.logical_and(unstable_long, np.logical_not(unstable_short))
        is_appearing = np.logical_and(unstable_long, stable_short)
        appearing_stability_df = stability_df[is_appearing]
        appearing_aclus = appearing_stability_df.aclu.values

        is_disappearing = np.logical_and(stable_long, unstable_short)
        disappearing_stability_df = stability_df[is_disappearing]
        disappearing_aclus = disappearing_stability_df.aclu.values

        ## OUTPUTS: appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus
        # appearing_aclus, disappearing_aclus
        # appearing_stability_df
        # disappearing_stability_df

        # unstable_changing = np.logical_xor(unstable_long, unstable_short)
        # unstable_changing

        is_appearing_or_disappearing = np.logical_or(is_appearing, is_disappearing)
        appearing_or_disappearing_stability_df = stability_df[is_appearing_or_disappearing]
        appearing_or_disappearing_aclus = sort(appearing_or_disappearing_stability_df.aclu.values)
        

        # appearing_or_disappearing_aclus = (disappearing_aclus.tolist() + appearing_aclus.tolist())
        # appearing_or_disappearing_aclus = sort(appearing_or_disappearing_aclus)
        # override_active_neuron_IDs.extend(appearing_aclus)
        # override_active_neuron_IDs # [ 3 11 14 15 24 34 35 51 58 67 74 82]
        
        stable_both_stability_df = stability_df[np.logical_and(stable_long, stable_short)]
        stable_both_aclus = stable_both_stability_df.aclu.values
        
        stable_neither_stability_df = stability_df[np.logical_and(unstable_long, unstable_short)]
        stable_neither_aclus = stable_neither_stability_df.aclu.values
        
        stable_long_stability_df = stability_df[stable_long]
        stable_long_aclus = stable_long_stability_df.aclu.values
        
        stable_short_stability_df = stability_df[stable_short]
        stable_short_aclus = stable_short_stability_df.aclu.values

        return (appearing_stability_df, disappearing_stability_df, appearing_or_disappearing_stability_df, stable_both_stability_df, stable_neither_stability_df, stable_long_stability_df, stable_short_stability_df)
        # (appearing_aclus, disappearing_aclus, appearing_or_disappearing_aclus, stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus)
    
        # return appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus, (stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus)
    
        # return appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus, (stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus)


    def get_stability_df(self):
        """ 
        Usage:
        
            stability_df, stability_dict = a_trial_by_trial_result.get_stability_df()
        
        """
        # directional_lap_epochs_dict: Dict[str, Epoch] = self.directional_lap_epochs_dict
        stability_dict = {k:list(v.aclu_to_stability_score_dict.values()) for k,v in self.directional_active_lap_pf_results_dicts.items()}
        stability_df: pd.DataFrame = pd.DataFrame({'aclu': self.any_decoder_neuron_IDs, **stability_dict})
        return stability_df, stability_dict
    
    def get_cell_stability_info(self, minimum_one_point_stability: float = 0.6, zero_point_stability: float = 0.1):
        """ 
        
        Usage:
            stability_df, stability_dict = a_trial_by_trial_result.get_stability_df()
            appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus = a_trial_by_trial_result.get_cell_stability_info(minimum_one_point_stability=0.6, zero_point_stability=0.1)
            override_active_neuron_IDs = deepcopy(appearing_or_disappearing_aclus)
            override_active_neuron_IDs


        """
        stability_df, stability_dict = self.get_stability_df()
        
        neuron_group_split_stability_dfs_tuple = TrialByTrialActivityResult.determine_neuron_stability(stability_df=stability_df, minimum_one_point_stability=minimum_one_point_stability, zero_point_stability=zero_point_stability)
        appearing_stability_df, disappearing_stability_df, appearing_or_disappearing_stability_df, stable_both_stability_df, stable_neither_stability_df, stable_long_stability_df, stable_short_stability_df = neuron_group_split_stability_dfs_tuple
        neuron_group_split_stability_aclus_tuple = [np.sort(a_df.aclu.values) for a_df in neuron_group_split_stability_dfs_tuple]
        appearing_aclus, disappearing_aclus, appearing_or_disappearing_aclus, stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus = neuron_group_split_stability_aclus_tuple
        return (neuron_group_split_stability_dfs_tuple, neuron_group_split_stability_aclus_tuple)

def _workaround_validate_has_directional_trial_by_trial_activity_result(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """ Validates `_build_trial_by_trial_activity_metrics`
    """
    directional_trial_by_trial_activity_result = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)
    if directional_trial_by_trial_activity_result is None:
        return False

    any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
    if (any_decoder_neuron_IDs is None) or (len(any_decoder_neuron_IDs) == 0):
        return False

    directional_lap_epochs_dict = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
    if (directional_lap_epochs_dict is None) or (len(directional_lap_epochs_dict) == 0):
        return False
    
    directional_active_lap_pf_results_dicts = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
    if (directional_active_lap_pf_results_dicts is None) or (len(directional_active_lap_pf_results_dicts) == 0):
        return False
    
    return True



def _workaround_validate_has_extended_pf_peak_info_result(curr_active_pipeline, computation_filter_name='maze') -> bool:
    """ Validates `_add_extended_pf_peak_information`
    """
    jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data.get('jonathan_firing_rate_analysis', None)
    if jonathan_firing_rate_analysis is None:
        return False
    
    neuron_replay_stats_df = jonathan_firing_rate_analysis.neuron_replay_stats_df
    if neuron_replay_stats_df is None:
        return False
    
    if 'long_LR_pf2D_peak_x' not in neuron_replay_stats_df.columns:
        return False

    if 'long_LR_pf1D_peak' not in neuron_replay_stats_df.columns:
        return False

    return True



# ==================================================================================================================== #
# Resume Misc Helper Functions                                                                                         #
# ==================================================================================================================== #

def _check_result_laps_epochs_df_performance(result_laps_epochs_df: pd.DataFrame, debug_print=True) -> CompleteDecodedContextCorrectness:
    """ 2024-01-17 - Validates the performance of the pseudo2D decoder posteriors using the laps data.

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance
    (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)
    
    """
    # Check 'maze_id' decoding accuracy
    n_laps = np.shape(result_laps_epochs_df)[0]
    is_decoded_track_correct = (result_laps_epochs_df['maze_id'] == result_laps_epochs_df['is_most_likely_track_identity_Long'].apply(lambda x: 0 if x else 1))
    percent_laps_track_identity_estimated_correctly = (np.sum(is_decoded_track_correct) / n_laps)
    if debug_print:
        print(f'percent_laps_track_identity_estimated_correctly: {percent_laps_track_identity_estimated_correctly}')
    # Check 'is_LR_dir' decoding accuracy:
    is_decoded_dir_correct = (result_laps_epochs_df['is_LR_dir'].apply(lambda x: 0 if x else 1) == result_laps_epochs_df['is_most_likely_direction_LR'].apply(lambda x: 0 if x else 1))
    percent_laps_direction_estimated_correctly = (np.sum(is_decoded_dir_correct) / n_laps)
    if debug_print:
        print(f'percent_laps_direction_estimated_correctly: {percent_laps_direction_estimated_correctly}')

    # Both should be correct
    are_both_decoded_properties_correct = np.logical_and(is_decoded_track_correct, is_decoded_dir_correct)
    percent_laps_estimated_correctly = (np.sum(are_both_decoded_properties_correct) / n_laps)
    if debug_print:
        print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        
    result_laps_epochs_df['is_decoded_track_correct'] = is_decoded_track_correct
    result_laps_epochs_df['is_decoded_dir_correct'] = is_decoded_dir_correct
    result_laps_epochs_df['are_both_decoded_properties_correct'] = are_both_decoded_properties_correct

    # return (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)
    return CompleteDecodedContextCorrectness(DecodedContextCorrectnessArraysTuple(is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), PercentDecodedContextCorrectnessTuple(percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly))



def _update_decoder_result_active_filter_epoch_columns(a_result_obj: DecodedFilterEpochsResult, a_score_result_df: pd.DataFrame, columns=['score', 'velocity', 'intercept', 'speed'], index_column_names=None):
    """ Joins the radon-transform result into the `a_result_obj.filter_epochs` dataframe.
    
    decoder_laps_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_laps_filter_epochs_decoder_result_dict[a_name], a_radon_transform_df=decoder_laps_radon_transform_df_dict[a_name])
    decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_ripple_filter_epochs_decoder_result_dict[a_name], a_radon_transform_df=decoder_ripple_radon_transform_df_dict[a_name])
    
    If the score result is filtered, the results will be NaN :[

    


    """
    if index_column_names is not None:
        ## use these columns as the ones to join on
        # assert a_result_obj.active_filter_epochs.n_epochs == np.shape(a_radon_transform_df)[0]
        # a_score_result_df['ripple_start_t']

        # from neuropy.core.epoch import find_data_indicies_from_epoch_times

        # selection_start_stop_times = deepcopy(active_epochs_df[['start', 'stop']].to_numpy())
        # print(f'np.shape(selection_start_stop_times): {np.shape(selection_start_stop_times)}')

        # test_epochs_data_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df)
        # print(f'np.shape(test_epochs_data_df): {np.shape(test_epochs_data_df)}')

        # # 2D_search (for both start, end times):
        # found_data_indicies = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=selection_start_stop_times)
        # print(f'np.shape(found_data_indicies): {np.shape(found_data_indicies)}')

        # # 1D_search (only for start times):
        # found_data_indicies_1D_search = find_data_indicies_from_epoch_times(test_epochs_data_df, epoch_times=np.squeeze(selection_start_stop_times[:, 0]))
        # print(f'np.shape(found_data_indicies_1D_search): {np.shape(found_data_indicies_1D_search)}')
        # found_data_indicies_1D_search

        # assert np.array_equal(found_data_indicies, found_data_indicies_1D_search)

        ## INPUTS: a_result_obj.filter_epochs, a_score_result_df
        # ValueError: columns overlap but no suffix specified: Index(['ripple_idx', 'ripple_start_t'], dtype='object')

        # list(a_result_obj.filter_epochs.columns)
        # list(a_score_result_df.columns)
        # a_result_obj.filter_epochs.merge(a_score_result_df)


        if isinstance(a_result_obj.filter_epochs, pd.DataFrame):
            a_result_obj.filter_epochs.drop(columns=columns, inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
            # a_result_obj.filter_epochs = a_result_obj.filter_epochs.join(a_score_result_df) # add the newly computed columns to the Epochs object - ValueError: columns overlap but no suffix specified: Index(['ripple_idx', 'ripple_start_t'], dtype='object')
            a_result_obj.filter_epochs = a_result_obj.filter_epochs.merge(a_score_result_df, how='left', on=index_column_names)            
        else:
            # Otherwise it's an Epoch object
            a_result_obj.filter_epochs._df.drop(columns=columns, inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
            # a_result_obj.filter_epochs._df = a_result_obj.filter_epochs.to_dataframe().join(a_score_result_df) # add the newly computed columns to the Epochs object
            a_result_obj.filter_epochs._df = a_result_obj.filter_epochs.to_dataframe().merge(a_score_result_df, how='left', on=index_column_names)

    else:
        ## must have the same number of entries in each
        assert a_result_obj.num_filter_epochs == np.shape(a_score_result_df)[0], f"a_result_obj.num_filter_epochs: {a_result_obj.num_filter_epochs} != np.shape(a_score_result_df)[0]: {np.shape(a_score_result_df)[0]}" # #TODO 2024-05-23 02:19: - [ ] I don't know the full purpose of this assert, I'm guessing it's to make sure were operating on the same epochs. What's passed in is a flat vector of values so we have no correspondance (like the start_t) if they don't literally correspond
        if isinstance(a_result_obj.filter_epochs, pd.DataFrame):
            a_result_obj.filter_epochs.drop(columns=columns, inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
            a_result_obj.filter_epochs = a_result_obj.filter_epochs.join(a_score_result_df) # add the newly computed columns to the Epochs object - ValueError: columns overlap but no suffix specified: Index(['ripple_idx', 'ripple_start_t'], dtype='object')
        else:
            # Otherwise it's an Epoch object
            a_result_obj.filter_epochs._df.drop(columns=columns, inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
            a_result_obj.filter_epochs._df = a_result_obj.filter_epochs.to_dataframe().join(a_score_result_df) # add the newly computed columns to the Epochs object
            ## ValueError: columns overlap but no suffix specified: Index(['lap_idx', 'lap_start_t'], dtype='object') -- occured after adding in start_t

    return a_result_obj


## INPUTS: decoder_laps_radon_transform_df_dict
def _build_merged_score_metric_df(decoder_epochs_score_metric_df_dict: Dict[str, pd.DataFrame], columns=['score', 'velocity', 'intercept', 'speed'], best_decoder_index_column_name:str='best_decoder_index') ->  pd.DataFrame:
    """Build a single merged dataframe from the cpomputed score metric results for all four decoders.

    Works with radon transform, wcorr, etc

    Creates columns like: score_long_LR, score_short_LR, ...

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _build_merged_score_metric_df, _update_decoder_result_active_filter_epoch_columns
        ## Extract the concrete dataframes from the results:
        extracted_filter_epochs_dfs_dict = {k:ensure_dataframe(a_result.filter_epochs) for k, a_result in directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict.items()}
        ## Merge the dict of four dataframes, one for each decoder, with column names like ['wcorr', 'travel', 'speed'] to a single merged df with suffixed of the dict keys like ['wcorr_long_LR', 'wcorr_long_RL',  ...., 'travel_long_LR', 'travel_long_RL', 'travel_short_LR', 'travel_short_RL', ...]
        extracted_merged_scores_df: pd.DataFrame = _build_merged_score_metric_df(extracted_filter_epochs_dfs_dict, columns=['travel'], best_decoder_index_column_name=None)
        extracted_merged_scores_df

    """
    from neuropy.utils.indexing_helpers import intersection_of_arrays, union_of_arrays

    score_metric_merged_df: pd.DataFrame = None

    ## only get columns that are actually included in each:
    valid_columns = intersection_of_arrays(*[list(a_df.columns) for a_decoder_name, a_df in decoder_epochs_score_metric_df_dict.items()])
    ## de-duplicate columns
    valid_columns = list(dict.fromkeys(valid_columns).keys()) # de-duplicate valid_columns
    columns = list(dict.fromkeys(columns).keys()) # de-duplicate user-provided columns
    valid_columns = [v for v in valid_columns if v in columns] # only include column_names that are in columns


    # filter_columns_fn = lambda df: df[['score']]
    filter_columns_fn = lambda df: df[valid_columns] # KeyError: "['P_decoder', 'score', 'velocity', 'intercept', 'speed', 'pearsonr', 'travel', 'coverage', 'jump', 'longest_sequence_length_ratio', 'direction_change_bin_ratio', 'congruent_dir_bins_ratio', 'total_congruent_direction_change', 'total_variation', 'integral_second_derivative', 'stddev_of_diff'] not in index"
    for a_decoder_name, a_df in decoder_epochs_score_metric_df_dict.items():
        # a_name: str = a_name.capitalize()
        if score_metric_merged_df is None:
            score_metric_merged_df = filter_columns_fn(deepcopy(a_df))
            score_metric_merged_df = score_metric_merged_df.add_suffix(f"_{a_decoder_name}") # suffix the columns so they're unique
        else:
            ## append to the initial_df
            score_metric_merged_df = score_metric_merged_df.join(filter_columns_fn(deepcopy(a_df)).add_suffix(f"_{a_decoder_name}"), lsuffix=None, rsuffix=None)

    # Get the column name with the maximum value for each row
    if best_decoder_index_column_name is not None:
        # initial_df['best_decoder_index'] = initial_df.idxmax(axis=1)
        score_metric_merged_df[best_decoder_index_column_name] = score_metric_merged_df.apply(lambda row: np.argmax(np.abs(row.values)), axis=1) #TODO 2024-03-14 18:09: - [ ] This needs to be changed so that it works with multiple columns. Previously it was hardcoded to just find the best index of a single column.

    ## OUTPUTS: radon_transform_merged_df, decoder_laps_radon_transform_df_dict
    return score_metric_merged_df


@function_attributes(short_name=None, tags=['spikes_df', 'global', 'global_spikes_df'], input_requires=[], output_provides=[], uses=[], used_by=['_decode_continuous_using_directional_decoders', 'compute_train_test_split_laps_decoders'], creation_date='2024-03-29 22:28', related_items=['decode_specific_epochs'])
def get_proper_global_spikes_df(owning_pipeline_reference, minimum_inclusion_fr_Hz: Optional[float]=None, included_qclu_values: Optional[List]=None) -> pd.DataFrame:
    """ Gets the global_spikes_df filtered to the correct cells, etc.

    In the form needed by `decode_specific_epochs(global_spikes_df, ...)`
     Get proper global_spikes_df, requires curr_active_pipeline

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

    spikes_df = get_proper_global_spikes_df(curr_active_pipeline)

    """
    # Get proper global_spikes_df:
    long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
    rank_order_results = owning_pipeline_reference.global_computation_results.computed_data.get('RankOrder', None) # "RankOrderComputationsContainer"
    if rank_order_results is not None:
        if minimum_inclusion_fr_Hz is None:	
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        if included_qclu_values is None:
            included_qclu_values: List[int] = rank_order_results.included_qclu_values
    else:        
        ## get from parameters:
        if minimum_inclusion_fr_Hz is None:
            minimum_inclusion_fr_Hz: float = owning_pipeline_reference.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        if included_qclu_values is None:
            included_qclu_values: List[int] = owning_pipeline_reference.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
    
    directional_laps_results: DirectionalLapsResult = owning_pipeline_reference.global_computation_results.computed_data['DirectionalLaps']
    track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
    any_list_neuron_IDs = track_templates.any_decoder_neuron_IDs # neuron_IDs as they appear in any list
    global_spikes_df = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].spikes_df).spikes.sliced_by_neuron_id(any_list_neuron_IDs) # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:        
    return global_spikes_df


from enum import Enum

class EpochFilteringMode(Enum):
    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode, _compute_proper_filter_epochs
    
    """
    DropShorter = 1 # ## Drop those less than the time bin duration
    ConstrainDecodingTimeBinSizeToMinimum = 2 # constraints the actual time_bin_size used to the minimum of the smallest epoch duration and the user's desired_time_bin_size.


@function_attributes(short_name=None, tags=['filter_epochs', 'time_bin_size'], input_requires=[], output_provides=[], uses=['EpochFilteringMode'], used_by=['_compute_lap_and_ripple_epochs_decoding_for_decoder'], creation_date='2024-06-04 06:43', related_items=[])
def _compute_proper_filter_epochs(epochs_df: pd.DataFrame, desired_decoding_time_bin_size: float, minimum_event_duration: Optional[float]=None, mode:EpochFilteringMode=EpochFilteringMode.DropShorter, debug_print: bool = False) -> Tuple[pd.DataFrame, float]:
    """ Either drops invalid epochs from `replay_epochs_df` or adjusts `ripple_decoding_time_bin_size` (depending on the `mode`) to ensure that no invalid epochs are passed for decoding.

    DropShorterMode

    Generate an enum with 2 modes: 'DropShorter', 'ConstrainDecodingTimeBinSizeToMinimum'
    replay_epochs_df, ripple_decoding_time_bin_size = _compute_proper_filter_epochs(replay_epochs_df=global_replays, desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size, minimum_event_duration=(2.0 * desired_ripple_decoding_time_bin_size), mode="DropShorter")

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_proper_filter_epochs

    global_replays, ripple_decoding_time_bin_size = _compute_proper_filter_epochs(replay_epochs_df=global_replays, desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size, minimum_event_duration=(2.0 * desired_ripple_decoding_time_bin_size), mode=EpochFilteringMode.DropShorter)
    global_replays, ripple_decoding_time_bin_size = _compute_proper_filter_epochs(replay_epochs_df=global_replays, desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size, mode=EpochFilteringMode.ConstrainDecodingTimeBinSizeToMinimum)


    """
    if not isinstance(epochs_df, pd.DataFrame):
        epochs_df = epochs_df.to_dataframe()

    if mode.value == EpochFilteringMode.DropShorter.value:
        ripple_decoding_time_bin_size: float = desired_decoding_time_bin_size # allow direct use            
        ## Drop those less than the time bin duration
        if debug_print:
            print(f'DropShorterMode:')
        pre_drop_n_epochs = len(epochs_df)
        if minimum_event_duration is not None:                
            epochs_df = epochs_df[epochs_df['duration'] >= minimum_event_duration]
            post_drop_n_epochs = len(epochs_df)
            n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
            if debug_print:
                print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
        else:
            epochs_df = epochs_df[epochs_df['duration'] > desired_decoding_time_bin_size]
            post_drop_n_epochs = len(epochs_df)
            n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
            if debug_print:
                print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_decoding_time_bin_size}', end='\t') 

        if debug_print:
            print(f'{post_drop_n_epochs} remain.')
        return epochs_df, ripple_decoding_time_bin_size
    
    elif mode.value == EpochFilteringMode.ConstrainDecodingTimeBinSizeToMinimum.value:
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(epochs_df['duration'].to_numpy())
        ripple_decoding_time_bin_size: float = min(desired_decoding_time_bin_size, min_possible_time_bin_size) # 10ms # 0.002

        return epochs_df, ripple_decoding_time_bin_size # return the unfiltered epochs_df and the constrained 'ripple_decoding_time_bin_size'
    else:
        raise NotImplementedError(f"unhandled mode: '{mode}'")
    




@function_attributes(short_name=None, tags=['replay', 'hardcoded-epochs-laps-and-replays'], input_requires=['filtered_sessions[global_epoch_name].replay'], output_provides=[], uses=['a_directional_pf1D_Decoder.decode_specific_epochs', '_compute_proper_filter_epochs'], used_by=[], creation_date='2024-07-03 15:35', related_items=['_try_single_decode'])
def _compute_arbitrary_epochs_decoding_for_decoder(a_directional_pf1D_Decoder: BasePositionDecoder, spikes_df: pd.DataFrame, decoding_epochs: Epoch, desired_epoch_decoding_time_bin_size: float = 0.1, use_single_time_bin_per_epoch: bool=False, epochs_filtering_mode:EpochFilteringMode=EpochFilteringMode.DropShorter) -> DecodedFilterEpochsResult:
    """ Decodes any arbitrarily specfied epochs at the specified time bin size

    History:
        Factored out of `_compute_lap_and_ripple_epochs_decoding_for_decoder(...)` which did hard-coded laps and replays together so that it could be used on any arbitrary interval


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, _compute_arbitrary_epochs_decoding_for_decoder

        active_decoding_epochs: Epoch = deepcopy(replay_epoch_variations['diba_quiescent_method_replay_epochs'])
        active_decoding_time_bin_size: float = 0.02 # 20ms
        minimum_inclusion_fr_Hz: float = active_decoding_epochs.metadata.get('minimum_inclusion_fr_Hz', 1.0)
        included_qclu_values: List[int] = active_decoding_epochs.metadata.get('included_qclu_values', [1,2])

        directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # spikes_df = curr_active_pipeline.sess.spikes_df # inferior way
        spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        ## Decode epochs for all four decoders:
        new_epochs_decoder_result_dict: Dict[str, Optional["DecodedFilterEpochsResult"]] = {}
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            new_epochs_decoder_result_dict[a_name] = _compute_arbitrary_epochs_decoding_for_decoder(a_decoder, spikes_df=deepcopy(spikes_df), decoding_epochs=active_decoding_epochs, desired_epoch_decoding_time_bin_size=active_decoding_time_bin_size)

        ## OUTPUTS: new_epochs_decoder_result_dict,

    """
    from neuropy.core.epoch import TimeColumnAliasesProtocol
    
    # Modifies alt_directional_merged_decoders_result, a copy of the original result, with new timebins
    a_directional_pf1D_Decoder = deepcopy(a_directional_pf1D_Decoder)

    if use_single_time_bin_per_epoch:
        print(f'WARNING: use_single_time_bin_per_epoch=True so time bin sizes will be ignored.')

    ## Decode Epochs by calling `a_directional_pf1D_Decoder.decode_specific_epochs` plus do some extra pre-processing of the epochs to be safe:
    if desired_epoch_decoding_time_bin_size is not None:
        decoding_epochs = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(decoding_epochs))
        decoding_epochs, epoch_decoding_time_bin_size = _compute_proper_filter_epochs(epochs_df=decoding_epochs, desired_decoding_time_bin_size=desired_epoch_decoding_time_bin_size, minimum_event_duration=(2.0 * desired_epoch_decoding_time_bin_size), mode=epochs_filtering_mode)
        if use_single_time_bin_per_epoch:
            epoch_decoding_time_bin_size = None

        a_decoded_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(spikes_df), filter_epochs=decoding_epochs, decoding_time_bin_size=epoch_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

    else:
        a_decoded_filter_epochs_decoder_result = None


    return a_decoded_filter_epochs_decoder_result #, epoch_decoding_time_bin_size








@function_attributes(short_name=None, tags=['replay', 'hardcoded-epochs-laps-and-replays', 'TODO::hardcoded-epochs_laps_and_replays'], input_requires=['filtered_sessions[global_epoch_name].replay'], output_provides=[], uses=['_compute_proper_filter_epochs'], used_by=[], creation_date='2024-05-22 17:58', related_items=['_try_single_decode'])
def _compute_lap_and_ripple_epochs_decoding_for_decoder(a_directional_pf1D_Decoder: BasePositionDecoder, curr_active_pipeline, desired_laps_decoding_time_bin_size: float = 0.5, desired_ripple_decoding_time_bin_size: float = 0.1, use_single_time_bin_per_epoch: bool=False,
                                                         epochs_filtering_mode:EpochFilteringMode=EpochFilteringMode.DropShorter) -> Tuple[DecodedFilterEpochsResult, Optional[DecodedFilterEpochsResult]]:
    """ Decodes the laps and the ripples and their RadonTransforms using the provided decoder for a single set of time_bin_size values
    ~12.2s per decoder.

    #TODO 2024-07-03 16:17: - [ ] Convert to call new `_compute_arbitrary_epochs_decoding_for_decoder(...)` one for the laps and replays. Should be drop-in equivalent

    """
    from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
    from neuropy.core.epoch import TimeColumnAliasesProtocol
    
    # Modifies alt_directional_merged_decoders_result, a copy of the original result, with new timebins
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    a_directional_pf1D_Decoder = deepcopy(a_directional_pf1D_Decoder)

    if use_single_time_bin_per_epoch:
        print(f'WARNING: use_single_time_bin_per_epoch=True so time bin sizes will be ignored.')
    
    ## Decode Laps:
    if desired_laps_decoding_time_bin_size is not None:
        global_any_laps_epochs_obj = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
        min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(global_any_laps_epochs_obj.to_dataframe()['duration'].to_numpy())
        laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
        if use_single_time_bin_per_epoch:
            laps_decoding_time_bin_size = None

        a_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
        ## new `_compute_arbitrary_epochs_decoding_for_decoder` way (but untested)
        # a_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = _compute_arbitrary_epochs_decoding_for_decoder(a_directional_pf1D_Decoder, deepcopy(curr_active_pipeline.sess.spikes_df), global_any_laps_epochs_obj, laps_decoding_time_bin_size, use_single_time_bin_per_epoch, epochs_filtering_mode)

    else:
        # do not decode laps:
        a_directional_laps_filter_epochs_decoder_result = None

    ## Decode Ripples:
    if desired_ripple_decoding_time_bin_size is not None:
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay))
        global_replays, ripple_decoding_time_bin_size = _compute_proper_filter_epochs(epochs_df=global_replays, desired_decoding_time_bin_size=desired_ripple_decoding_time_bin_size, minimum_event_duration=(2.0 * desired_ripple_decoding_time_bin_size), mode=epochs_filtering_mode) # `ripple_decoding_time_bin_size` is set here! It takes a minimum value
        if use_single_time_bin_per_epoch:
            ripple_decoding_time_bin_size = None

        a_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = a_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
        # ## new `_compute_arbitrary_epochs_decoding_for_decoder` way (but untested)
        # a_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = _compute_arbitrary_epochs_decoding_for_decoder(a_directional_pf1D_Decoder, deepcopy(curr_active_pipeline.sess.spikes_df), global_replays, ripple_decoding_time_bin_size, use_single_time_bin_per_epoch, epochs_filtering_mode)

    else:
        a_directional_ripple_filter_epochs_decoder_result = None
        # ripple_radon_transform_df = None

    ## Post Compute Validations:
    # alt_directional_merged_decoders_result.perform_compute_marginals()
    return a_directional_laps_filter_epochs_decoder_result, a_directional_ripple_filter_epochs_decoder_result #, (laps_radon_transform_df, ripple_radon_transform_df)


@function_attributes(short_name=None, tags=['wcorr', 'correlation'], input_requires=[], output_provides=[], uses=[], used_by=['_compute_all_df_score_metrics', ''], creation_date='2024-04-19 18:07', related_items=[])
def compute_weighted_correlations(decoder_decoded_epochs_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], debug_print=False):
    """ 
    ## Weighted Correlation can only be applied to decoded posteriors, not spikes themselves.
    ### It works by assessing the degree to which a change in position corresponds to a change in time. For a simple diagonally increasing trajectory across the track at early timebins position will start at the bottom of the track, and as time increases the position also increases. The "weighted" part just corresponds to making use of the confidence probabilities of the decoded posterior: instead of relying on only the most-likely position we can include all information returned. Naturally will emphasize sharp decoded positions and de-emphasize diffuse ones.

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import compute_weighted_correlations

        decoder_laps_weighted_corr_df_dict = _compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_laps_filter_epochs_decoder_result_dict))
        decoder_ripple_weighted_corr_df_dict = _compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_ripple_filter_epochs_decoder_result_dict))

        
    History:
        Extracted and renamed from `_subfn_compute_weighted_correlations`

    """
    from neuropy.analyses.decoders import wcorr
    # INPUTS: decoder_decoded_epochs_result_dict

    weighted_corr_data_dict = {}

    # for a_name in track_templates.get_decoder_names():
    for a_name, curr_results_obj in decoder_decoded_epochs_result_dict.items():
        if curr_results_obj is not None:         
            weighted_corr_data = np.array([wcorr(a_P_x_given_n) for a_P_x_given_n in curr_results_obj.p_x_given_n_list]) # each `wcorr(a_posterior)` call returns a float
            if debug_print:
                print(f'a_name: "{a_name}"\n\tweighted_corr_data.shape: {np.shape(weighted_corr_data)}') # (84, ) - (n_epochs, )
            weighted_corr_data_dict[a_name] = pd.DataFrame({'wcorr': weighted_corr_data})

    ## end for
    return weighted_corr_data_dict






# ==================================================================================================================== #
#MARK ComputationFunctions                                                                                               
# ==================================================================================================================== #

## INPUTS: laps_all_epoch_bins_marginals_df, radon_transform_merged_df
def _compute_nonmarginalized_decoder_prob(an_all_epoch_bins_marginals_df: pd.DataFrame) ->  pd.DataFrame:
    """ Convert from the marginalized Long/Short probabilities back to the individual 1D decoder probability.
    """
    ## Get the probability of each decoder:
    a_marginals_df = deepcopy(an_all_epoch_bins_marginals_df)
    a_marginals_df['P_Long_LR'] = a_marginals_df['P_LR'] * a_marginals_df['P_Long']
    a_marginals_df['P_Long_RL'] = a_marginals_df['P_RL'] * a_marginals_df['P_Long']
    a_marginals_df['P_Short_LR'] = a_marginals_df['P_LR'] * a_marginals_df['P_Short']
    a_marginals_df['P_Short_RL'] = a_marginals_df['P_RL'] * a_marginals_df['P_Short']
    assert np.allclose(a_marginals_df[['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']].dropna().sum(axis=1), 1.0) # why are there NaN rows? All four decoders have NaN values for these rows.
    # # an_arr = a_marginals_df[['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']].dropna().sum(axis=1)
    # an_arr = a_marginals_df[['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']].sum(axis=1)
    # invalid_idxs = np.where(np.logical_not(np.isclose(an_arr, 1.0, equal_nan=True)))[0] # , rtol=0.000000001, atol=0.0

    # array([182, 281, 349], dtype=int64) - row 182: `182	0.0	0.0	0.0	0.0	182	761.2960146963596	0.0	0.0	0.0	0.0`, row 281: `281	0.0	0.0	0.0	0.0	281	1221.8660240046447	0.0	0.0	0.0	0.0`

    # Get the column name with the maximum value for each row
    # a_marginals_df['most_likely_decoder_index'] = a_marginals_df[['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']].idxmax(axis=1)
    a_marginals_df['most_likely_decoder_index'] = a_marginals_df[['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']].apply(lambda row: np.argmax(row.values), axis=1)
    return a_marginals_df




def _subfn_compute_arbitrary_epoch_decoding_radon_transform_for_decoder(a_directional_pf1D_Decoder, a_directional_arbitrary_filter_epochs_decoder_result: Optional[DecodedFilterEpochsResult], nlines=4192, margin=4, n_jobs=4):
    """ Decodes any arbitrary decoded epochs and their RadonTransforms using the provided decoder.
    
    #TODO 2024-07-03 17:14: - [ ] replaces `_subfn_compute_epoch_decoding_radon_transform_for_decoder`

    """
    a_directional_pf1D_Decoder = deepcopy(a_directional_pf1D_Decoder)
    pos_bin_size: float = a_directional_pf1D_Decoder.pos_bin_size # 3.793023081021702'
    xbin_centers = deepcopy(a_directional_pf1D_Decoder.xbin_centers)

    ## Decode Epochs:
    if a_directional_arbitrary_filter_epochs_decoder_result is not None:
        epoch_radon_transform_extras = []
        epoch_radon_transform_df, *epoch_radon_transform_extras = a_directional_arbitrary_filter_epochs_decoder_result.compute_radon_transforms(pos_bin_size=pos_bin_size, xbin_centers=xbin_centers, nlines=nlines, margin=margin, n_jobs=n_jobs)
    else:
        epoch_radon_transform_extras = None
        epoch_radon_transform_df = None

    return epoch_radon_transform_df, epoch_radon_transform_extras






@function_attributes(short_name=None, tags=['TODO::hardcoded-epochs_laps_and_replays'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-03 17:10', related_items=[])
def _subfn_compute_epoch_decoding_radon_transform_for_decoder(a_directional_pf1D_Decoder, a_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult, a_directional_ripple_filter_epochs_decoder_result: Optional[DecodedFilterEpochsResult], nlines=4192, margin=4, n_jobs=4):
    """ Decodes the laps and the ripples and their RadonTransforms using the provided decoder.
    ~12.2s per decoder.

    """
    a_directional_pf1D_Decoder = deepcopy(a_directional_pf1D_Decoder)
    pos_bin_size: float = a_directional_pf1D_Decoder.pos_bin_size # 3.793023081021702'
    xbin_centers = deepcopy(a_directional_pf1D_Decoder.xbin_centers)

    laps_radon_transform_extras = []
    laps_radon_transform_df, *laps_radon_transform_extras = a_directional_laps_filter_epochs_decoder_result.compute_radon_transforms(pos_bin_size=pos_bin_size, xbin_centers=xbin_centers, nlines=nlines, margin=margin, n_jobs=n_jobs)

    ## Decode Ripples:
    if a_directional_ripple_filter_epochs_decoder_result is not None:
        ripple_radon_transform_extras = []
        # ripple_radon_transform_df = compute_radon_transforms(a_directional_pf1D_Decoder, a_directional_ripple_filter_epochs_decoder_result)
        ripple_radon_transform_df, *ripple_radon_transform_extras = a_directional_ripple_filter_epochs_decoder_result.compute_radon_transforms(pos_bin_size=pos_bin_size, xbin_centers=xbin_centers, nlines=nlines, margin=margin, n_jobs=n_jobs)
    else:
        ripple_radon_transform_extras = None
        ripple_radon_transform_df = None

    return laps_radon_transform_df, laps_radon_transform_extras, ripple_radon_transform_df, ripple_radon_transform_extras


@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=['_compute_all_df_score_metrics'], creation_date='2024-05-22 18:07', related_items=[])
def _subfn_compute_complete_df_metrics(directional_merged_decoders_result: "DirectionalPseudo2DDecodersResult", track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, decoder_laps_df_dict: Dict[str, pd.DataFrame], decoder_ripple_df_dict: Dict[str, pd.DataFrame],
                                        active_df_columns = ['wcorr'], suppress_exceptions: bool = False):
    """ Called one for each specific score metric (e.g. (Radon Transform, WCorr, PearsonR)) after it is computed to compute its merged dataframes and dataframe dicts. 
    
    Generalized to work with any result dfs not just Radon Transforms
    
    
    Usage:

    # DirectionalMergedDecoders: Get the result after computation:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']

    (laps_radon_transform_merged_df, ripple_radon_transform_merged_df), (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _compute_complete_df_metrics(track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                                            decoder_laps_df_dict=deepcopy(decoder_laps_radon_transform_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_radon_transform_df_dict), active_df_columns = ['score', 'velocity', 'intercept', 'speed'])

    (laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df), (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _compute_complete_df_metrics(track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                                            decoder_laps_df_dict=deepcopy(decoder_laps_weighted_corr_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_weighted_corr_df_dict), active_df_columns = ['wcorr'])


    """
    ## INPUTS: track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, decoder_laps_filter_epochs_decoder_result_dict, decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict
    for a_name, a_decoder in track_templates.get_decoders_dict().items():
        decoder_laps_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_laps_filter_epochs_decoder_result_dict[a_name], a_score_result_df=decoder_laps_df_dict[a_name], columns=active_df_columns)
        decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_ripple_filter_epochs_decoder_result_dict[a_name], a_score_result_df=decoder_ripple_df_dict[a_name], columns=active_df_columns)

    ## Convert from a Dict (decoder_laps_df_dict) to a merged dataframe with columns suffixed with the decoder name:
    laps_metric_merged_df = _build_merged_score_metric_df(decoder_laps_df_dict, columns=active_df_columns)
    ripple_metric_merged_df = _build_merged_score_metric_df(decoder_ripple_df_dict, columns=active_df_columns)
    ## OUTPUTS: laps_metric_merged_df, ripple_metric_merged_df

    # The output CSVs have the base columns from the `ripple_all_epoch_bins_marginals_df`, which is a bit surprising
    ## why is this even needed? The `directional_merged_decoders_result.laps_all_epoch_bins_marginals_df` come in with the marginals but not the original non-marginalized P_decoder values. 
    ## Get the 1D decoder probabilities explicitly and add them as columns to the dfs:
    _laps_all_epoch_bins_marginals_df =  _compute_nonmarginalized_decoder_prob(deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df)) # incomming df has columns: ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'lap_idx', 'lap_start_t']
    _ripple_all_epoch_bins_marginals_df =  _compute_nonmarginalized_decoder_prob(deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)) # incomming ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'ripple_idx', 'ripple_start_t']
    

    ## Merge in the RadonTransform df:
    _mergev_laps_metric_merged_df: pd.DataFrame = deepcopy(_laps_all_epoch_bins_marginals_df).join(deepcopy(laps_metric_merged_df)) #TODO 2025-01-02 13:12: - [ ] !!PITFALL!! `df.join(...)` always seems to mess things up, is this where the problems are happening?
    _mergev_ripple_metric_merged_df: pd.DataFrame = deepcopy(_ripple_all_epoch_bins_marginals_df).join(deepcopy(ripple_metric_merged_df)) # has ['ripple_idx', 'ripple_start_t'] to join on
    
    # from neuropy.utils.indexing_helpers import PandasHelpers
    laps_metric_merged_df = PandasHelpers.adding_additional_df_columns(original_df=_laps_all_epoch_bins_marginals_df, additional_cols_df=laps_metric_merged_df) # update the filter_epochs with the new columns
    ripple_metric_merged_df = PandasHelpers.adding_additional_df_columns(original_df=_ripple_all_epoch_bins_marginals_df, additional_cols_df=ripple_metric_merged_df)

    assert _mergev_laps_metric_merged_df.equals(laps_metric_merged_df) # == does NOT work at all, and it doesn't even make sense. (_mergev_laps_metric_merged_df == laps_metric_merged_df)
    assert _mergev_ripple_metric_merged_df.equals(ripple_metric_merged_df) # == does NOT work at all, and it doesn't even make sense.  (_mergev_ripple_metric_merged_df == ripple_metric_merged_df)

    ## Extract the individual decoder probability into the .active_epochs:
    shared_index_column_names = ['ripple_idx', 'ripple_start_t']
    per_decoder_df_columns = ['P_decoder']
    decoder_name_to_decoder_probability_column_map = dict(zip(track_templates.get_decoder_names(), ['P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL']))
    # for a_name, a_decoder in track_templates.get_decoders_dict().items():
    for a_name in track_templates.get_decoder_names():
        ## Build a single-column dataframe containing only the appropriate column for this decoder
        a_prob_column_name:str = decoder_name_to_decoder_probability_column_map[a_name]
        
        # ## #TODO 2024-07-05 23:22: - [ ] add in the columns needed to align properly
        a_laps_additional_columns = {}
        # a_laps_additional_columns = {k:laps_metric_merged_df[k].to_numpy() for k in ['lap_idx', 'lap_start_t']} # when incldued, I get `ValueError: columns overlap but no suffix specified: Index(['lap_idx', 'lap_start_t'], dtype='object')`
        a_laps_decoder_prob_df: pd.DataFrame = pd.DataFrame({'P_decoder': laps_metric_merged_df[a_prob_column_name].to_numpy(), **a_laps_additional_columns}) # ['lap_idx', 'lap_start_t']

        ## Ripple Setup: special
        # a_ripple_additional_columns = {}
        ripple_additional_column_names = ['ripple_start_t'] # ['ripple_idx', 'ripple_start_t']
        a_ripple_additional_columns = {k:ripple_metric_merged_df[k].to_numpy() for k in ripple_additional_column_names}
        a_ripple_decoder_prob_df: pd.DataFrame = pd.DataFrame({'P_decoder': ripple_metric_merged_df[a_prob_column_name].to_numpy(), **a_ripple_additional_columns}) # ['ripple_idx', 'ripple_start_t']
        a_ripple_decoder_prob_df = a_ripple_decoder_prob_df.rename(columns={'ripple_start_t': 'start'})


        # Example Suppressing Exception:
        with ExceptionPrintingContext(suppress=suppress_exceptions):
            decoder_laps_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_laps_filter_epochs_decoder_result_dict[a_name], a_score_result_df=a_laps_decoder_prob_df, columns=per_decoder_df_columns)
        with ExceptionPrintingContext(suppress=suppress_exceptions):
            decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_ripple_filter_epochs_decoder_result_dict[a_name], a_score_result_df=a_ripple_decoder_prob_df, columns=per_decoder_df_columns, index_column_names=['start']) # ripple_additional_column_names

    return (laps_metric_merged_df, ripple_metric_merged_df), (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)



#TODO 2024-07-03 17:17: - [ ] NOT YET FINISHED
@function_attributes(short_name=None, tags=['NOT_YET_FINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-03 17:18', related_items=[])
def _compute_arbitrary_epochs_all_df_score_metrics(directional_merged_decoders_result: "DirectionalPseudo2DDecodersResult", track_templates, new_epochs_decoder_result_dict: Dict[str, Optional[DecodedFilterEpochsResult]],
                                   spikes_df: pd.DataFrame, should_skip_radon_transform=False, suppress_exceptions: bool=False):
    """ computes for all score metrics (Radon Transform, WCorr, PearsonR) and adds them appropriately. 
    
    spikes_df is needed for Simple Correlation Score calculation.
    
    ## NOTE: To plot the radon transforms the values must be added to the result object's active_filter_epochs dataframe.
        That's why we spend so much trouble to do that
        
        
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_all_df_score_metrics

        decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = _perform_compute_custom_epoch_decoding(curr_active_pipeline, directional_merged_decoders_result, track_templates)

    
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(directional_merged_decoders_result, track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df = merged_df_outputs_tuple
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple

    """
    raise NotImplementedError()

    ## Radon Transform:
    decoder_ripple_radon_transform_df_dict = {}

    # These 'extra' dicts were intended to be used in debugging the Radon Transform calculations, but were never needed.
    decoder_ripple_radon_transform_extras_dict = {}

    if not should_skip_radon_transform:
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            decoder_ripple_radon_transform_df_dict[a_name], decoder_ripple_radon_transform_extras_dict[a_name] = _subfn_compute_arbitrary_epoch_decoding_radon_transform_for_decoder(a_directional_pf1D_Decoder=a_decoder, a_directional_arbitrary_filter_epochs_decoder_result=new_epochs_decoder_result_dict[a_name],
                                                                                                                                                                                    nlines=8192, margin=4.0,
                                                                                                                                                                                    n_jobs=6)
            
        # 6m 19.7s - nlines=8192, margin=16, n_jobs=1
        # 17m 57.6s - nlines=24000, margin=16, n_jobs=1
        # 4m 31.9s -  nlines=8192, margin=16, n_jobs=4
        # Still running 14m later - neighbours: 8 = int(margin: 32 / pos_bin_size: 3.8054171165052444)
            
        ## INPUTS: decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict
        (_, ripple_radon_transform_merged_df), (new_epochs_decoder_result_dict, new_epochs_decoder_result_dict) = _subfn_compute_complete_df_metrics(directional_merged_decoders_result, track_templates, new_epochs_decoder_result_dict, new_epochs_decoder_result_dict,
                                                                                                                                                                                                                    decoder_laps_df_dict=deepcopy(decoder_laps_radon_transform_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_radon_transform_df_dict), active_df_columns = ['score', 'velocity', 'intercept', 'speed'],
                                                                                                                                                                                                                    suppress_exceptions=suppress_exceptions)

        # agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs) = laps_radon_stats
        print(f'\tPerformance: Ripple: Radon Transform:')
        ripple_radon_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_radon_transform_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
    else:
        _, ripple_radon_transform_merged_df = None, None

    ## Weighted Correlation
    decoder_ripple_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(new_epochs_decoder_result_dict))
    (_, ripple_weighted_corr_merged_df), (new_epochs_decoder_result_dict, new_epochs_decoder_result_dict) = _subfn_compute_complete_df_metrics(directional_merged_decoders_result, track_templates, new_epochs_decoder_result_dict, new_epochs_decoder_result_dict,
                                                                                                                                                                                                                decoder_laps_df_dict=deepcopy(decoder_laps_weighted_corr_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_weighted_corr_df_dict), active_df_columns = ['wcorr'],
                                                                                                                                                                                                                suppress_exceptions=suppress_exceptions)
    
    ## Simple Pearson Correlation
    assert spikes_df is not None
    (_, ripple_simple_pf_pearson_merged_df), corr_column_names = directional_merged_decoders_result.compute_simple_spike_time_v_pf_peak_x_by_epoch(track_templates=track_templates, spikes_df=deepcopy(spikes_df))
    ## OUTPUTS: (_, ripple_simple_pf_pearson_merged_df), corr_column_names
    ## Computes the highest-valued decoder for this score:
    best_decoder_index_col_name: str = 'best_decoder_index'
    ripple_simple_pf_pearson_merged_df[best_decoder_index_col_name] = ripple_simple_pf_pearson_merged_df[corr_column_names].abs().apply(lambda row: np.argmax(row.values), axis=1)

    ## Get the 1D decoder probabilities explicitly and add them as columns to the dfs, and finally merge in the results:
    ripple_simple_pf_pearson_merged_df: pd.DataFrame = _compute_nonmarginalized_decoder_prob(deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)).join(ripple_simple_pf_pearson_merged_df)
    
    ## Extract the individual decoder probability into the .active_epochs
    per_decoder_df_column_name = 'pearsonr'
    # for a_name, a_decoder in track_templates.get_decoders_dict().items():
    for a_name, a_simple_pf_column_name in zip(track_templates.get_decoder_names(), corr_column_names):
        ## Build a single-column dataframe containing only the appropriate column for this decoder
        _a_ripple_metric_df: pd.DataFrame = pd.DataFrame({per_decoder_df_column_name: ripple_simple_pf_pearson_merged_df[a_simple_pf_column_name].to_numpy()})     
        with ExceptionPrintingContext():
            new_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=new_epochs_decoder_result_dict[a_name], a_score_result_df=_a_ripple_metric_df, columns=[per_decoder_df_column_name])


    # TEST AGREEMENTS ____________________________________________________________________________________________________ #

    ## count up the number that the RadonTransform and the most-likely direction agree
    # agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs) = laps_radon_stats
    print(f'Performance: Ripple: WCorr')
    ripple_wcorr_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_weighted_corr_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)

    # Test agreement:
    print(f'Performance: Ripple: Simple PF PearsonR')
    ripple_simple_pf_pearsonr_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_simple_pf_pearson_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)

    ## OUTPUTS: laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df

    ## OUTPUTS: decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict
    ## OUTPUTS: laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df
    ## OUTPUTS: (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
    raw_dict_outputs_tuple = (decoder_ripple_radon_transform_df_dict, decoder_ripple_radon_transform_extras_dict, decoder_ripple_weighted_corr_df_dict)
    merged_df_outputs_tuple = (ripple_radon_transform_merged_df, ripple_weighted_corr_merged_df, ripple_simple_pf_pearson_merged_df)
    return (new_epochs_decoder_result_dict, new_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple




@function_attributes(short_name=None, tags=['weighted-correlation', 'radon-transform', 'multiple-decoders', 'main-computation-function', 'TODO::hardcoded-epochs_laps_and_replays'], input_requires=[], output_provides=[],
 uses=['_compute_complete_df_metrics', 'compute_weighted_correlations', '_compute_epoch_decoding_radon_transform_for_decoder', '_compute_matching_best_indicies'],
 used_by=['_decode_and_evaluate_epochs_using_directional_decoders', 'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'], creation_date='2024-02-15 19:55', related_items=[])
def _compute_all_df_score_metrics(directional_merged_decoders_result: "DirectionalPseudo2DDecodersResult", track_templates,
                                   decoder_laps_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, Optional[DecodedFilterEpochsResult]],
                                   spikes_df: pd.DataFrame, should_skip_radon_transform=False, suppress_exceptions: bool=False):
    """ computes for all score metrics (Radon Transform, WCorr, PearsonR) and adds them appropriately. 
    
    spikes_df is needed for Simple Correlation Score calculation.
    
    ## NOTE: To plot the radon transforms the values must be added to the result object's active_filter_epochs dataframe.
        That's why we spend so much trouble to do that
        
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_all_df_score_metrics

        decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = _perform_compute_custom_epoch_decoding(curr_active_pipeline, directional_merged_decoders_result, track_templates)

    
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(directional_merged_decoders_result, track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df = merged_df_outputs_tuple
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple

    """

    ## Radon Transform:
    decoder_laps_radon_transform_df_dict = {}
    decoder_ripple_radon_transform_df_dict = {}

    # These 'extra' dicts were intended to be used in debugging the Radon Transform calculations, but were never needed.
    decoder_laps_radon_transform_extras_dict = {}
    decoder_ripple_radon_transform_extras_dict = {}

    if not should_skip_radon_transform:
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            # decoder_laps_radon_transform_df_dict[a_name], decoder_ripple_radon_transform_df_dict[a_name] = _compute_epoch_decoding_radon_transform_for_decoder(a_decoder, decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name], n_jobs=4)
            decoder_laps_radon_transform_df_dict[a_name], decoder_laps_radon_transform_extras_dict[a_name], decoder_ripple_radon_transform_df_dict[a_name], decoder_ripple_radon_transform_extras_dict[a_name] = _subfn_compute_epoch_decoding_radon_transform_for_decoder(a_directional_pf1D_Decoder=a_decoder, 
                                                                                                                                                                                                                                                                            a_directional_laps_filter_epochs_decoder_result=decoder_laps_filter_epochs_decoder_result_dict[a_name],
                                                                                                                                                                                                                                                                            a_directional_ripple_filter_epochs_decoder_result=decoder_ripple_filter_epochs_decoder_result_dict[a_name],
                                                                                                                                                                                                                                                                            nlines=8192, margin=4.0,
                                                                                                                                                                                                                                                                            n_jobs=6)
            
        # 6m 19.7s - nlines=8192, margin=16, n_jobs=1
        # 17m 57.6s - nlines=24000, margin=16, n_jobs=1
        # 4m 31.9s -  nlines=8192, margin=16, n_jobs=4
        # Still running 14m later - neighbours: 8 = int(margin: 32 / pos_bin_size: 3.8054171165052444)
            
        ## INPUTS: decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict
        (laps_radon_transform_merged_df, ripple_radon_transform_merged_df), (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _subfn_compute_complete_df_metrics(directional_merged_decoders_result, track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                                                    decoder_laps_df_dict=deepcopy(decoder_laps_radon_transform_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_radon_transform_df_dict), active_df_columns = ['score', 'velocity', 'intercept', 'speed'],
                                                                                                                                                                                                                    suppress_exceptions=suppress_exceptions)
        print(f'Performance: Radon Transform:\n\tLaps:')
        ## count up the number that the RadonTransform and the most-likely direction agree
        laps_radon_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(laps_radon_transform_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
        # agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs) = laps_radon_stats
        print(f'\tPerformance: Ripple: Radon Transform:')
        ripple_radon_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_radon_transform_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
    else:
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df = None, None

    ## Weighted Correlation
    decoder_laps_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_laps_filter_epochs_decoder_result_dict))
    decoder_ripple_weighted_corr_df_dict = compute_weighted_correlations(decoder_decoded_epochs_result_dict=deepcopy(decoder_ripple_filter_epochs_decoder_result_dict))
    (laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df), (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _subfn_compute_complete_df_metrics(directional_merged_decoders_result, track_templates, decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                                                decoder_laps_df_dict=deepcopy(decoder_laps_weighted_corr_df_dict), decoder_ripple_df_dict=deepcopy(decoder_ripple_weighted_corr_df_dict), active_df_columns = ['wcorr'],
                                                                                                                                                                                                                suppress_exceptions=suppress_exceptions)
    
    ## Simple Pearson Correlation

    assert spikes_df is not None
    (laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df), corr_column_names = directional_merged_decoders_result.compute_simple_spike_time_v_pf_peak_x_by_epoch(track_templates=track_templates, spikes_df=deepcopy(spikes_df))
    ## OUTPUTS: (laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df), corr_column_names
    try:
        ## Computes the highest-valued decoder for this score:
        best_decoder_index_col_name: str = 'best_decoder_index'
        laps_simple_pf_pearson_merged_df[best_decoder_index_col_name] = laps_simple_pf_pearson_merged_df[corr_column_names].abs().apply(lambda row: np.argmax(row.values), axis=1)
        ripple_simple_pf_pearson_merged_df[best_decoder_index_col_name] = ripple_simple_pf_pearson_merged_df[corr_column_names].abs().apply(lambda row: np.argmax(row.values), axis=1)

        ## Get the 1D decoder probabilities explicitly and add them as columns to the dfs, and finally merge in the results:
        laps_simple_pf_pearson_merged_df: pd.DataFrame = _compute_nonmarginalized_decoder_prob(deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df)).join(laps_simple_pf_pearson_merged_df)
        ripple_simple_pf_pearson_merged_df: pd.DataFrame = _compute_nonmarginalized_decoder_prob(deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)).join(ripple_simple_pf_pearson_merged_df)
        
        ## Extract the individual decoder probability into the .active_epochs
        per_decoder_df_column_name = 'pearsonr'
        # for a_name, a_decoder in track_templates.get_decoders_dict().items():
        for a_name, a_simple_pf_column_name in zip(track_templates.get_decoder_names(), corr_column_names):
            ## Build a single-column dataframe containing only the appropriate column for this decoder
            _a_laps_metric_df: pd.DataFrame = pd.DataFrame({per_decoder_df_column_name: laps_simple_pf_pearson_merged_df[a_simple_pf_column_name].to_numpy()})

            ripple_additional_column_names = ['ripple_start_t'] # ['ripple_idx', 'ripple_start_t']
            a_ripple_additional_columns = {k:ripple_simple_pf_pearson_merged_df[k].to_numpy() for k in ripple_additional_column_names}
            _a_ripple_metric_df: pd.DataFrame = pd.DataFrame({per_decoder_df_column_name: ripple_simple_pf_pearson_merged_df[a_simple_pf_column_name].to_numpy(), **a_ripple_additional_columns})     
            _a_ripple_metric_df = _a_ripple_metric_df.rename(columns={'ripple_start_t': 'start'})
            
            with ExceptionPrintingContext(suppress=suppress_exceptions): # this is causing horrible silent failures   
                decoder_laps_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_laps_filter_epochs_decoder_result_dict[a_name], a_score_result_df=_a_laps_metric_df, columns=[per_decoder_df_column_name])

            with ExceptionPrintingContext(suppress=suppress_exceptions):
                decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _update_decoder_result_active_filter_epoch_columns(a_result_obj=decoder_ripple_filter_epochs_decoder_result_dict[a_name], a_score_result_df=_a_ripple_metric_df, columns=[per_decoder_df_column_name], index_column_names=['start'])
    except KeyError as e:
        print(f"_compute_all_df_score_metrics(...): failed to get '*_pf_peak_x_pearsonr' columns, 'best_decoder_index' and other pearson related stats wont be calculated. Err: {e}")
        pass
    

    # TEST AGREEMENTS ____________________________________________________________________________________________________ #

    ## count up the number that the RadonTransform and the most-likely direction agree
    print(f'Performance: WCorr:\n\tLaps:')
    with ExceptionPrintingContext(suppress=True):
        laps_wcorr_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(laps_weighted_corr_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
    # agreeing_rows_ratio, (agreeing_rows_count, num_total_epochs) = laps_radon_stats
    print(f'Performance: Ripple: WCorr')
    with ExceptionPrintingContext(suppress=True):
        ripple_wcorr_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_weighted_corr_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)

    # Test agreement:
    print(f'Performance: Simple PF PearsonR:\n\tLaps:')
    with ExceptionPrintingContext(suppress=True):
        laps_simple_pf_pearson_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(laps_simple_pf_pearson_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
    print(f'Performance: Ripple: Simple PF PearsonR')
    with ExceptionPrintingContext(suppress=True):
        ripple_simple_pf_pearsonr_stats = DecoderDecodedEpochsResult.compute_matching_best_indicies(ripple_simple_pf_pearson_merged_df, index_column_name='most_likely_decoder_index', second_index_column_name='best_decoder_index', enable_print=True)
    


    ## OUTPUTS: laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df

    ## OUTPUTS: decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict
    ## OUTPUTS: laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df
    ## OUTPUTS: (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
    raw_dict_outputs_tuple = (decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_extras_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict)
    merged_df_outputs_tuple = (laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df, laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df)
    return (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple

# Custom Decoding of Epochs (Laps/Ripple) ____________________________________________________________________________ #

# Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result
@function_attributes(short_name=None, tags=['epochs', 'TODO::hardcoded-epochs_laps_and_replays'], input_requires=[], output_provides=[], uses=['_compute_lap_and_ripple_epochs_decoding_for_decoder'], used_by=['_decode_and_evaluate_epochs_using_directional_decoders'], creation_date='2024-05-22 18:07', related_items=[])
def _perform_compute_custom_epoch_decoding(curr_active_pipeline, directional_merged_decoders_result: "DirectionalPseudo2DDecodersResult", track_templates: "TrackTemplates", epochs_filtering_mode:EpochFilteringMode=EpochFilteringMode.DropShorter) -> Tuple[Dict[str, DecodedFilterEpochsResult], Dict[str, Optional[DecodedFilterEpochsResult]]]:
        """ Custom Decoder Computation:
        2024-02-15 - Appears to be best to refactor to the TrackTemplates object. __________________________________________ #
            # directional_merged_decoders_result mmakes more sense since it has the time_bin_size already

        Currently needs:
            curr_active_pipeline
        
        Pretty slow
        
        Usage:
        
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _perform_compute_custom_epoch_decoding
        
        """
        ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size
        pos_bin_size: float = track_templates.get_decoders()[0].pos_bin_size # 3.793023081021702
        print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}, ripple_decoding_time_bin_size: {ripple_decoding_time_bin_size}, pos_bin_size: {pos_bin_size}')

        ## Decode epochs for all four decoders:
        decoder_laps_filter_epochs_decoder_result_dict: Dict[str, Optional[DecodedFilterEpochsResult]] = {}
        decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, Optional[DecodedFilterEpochsResult]] = {}

        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, epochs_filtering_mode=epochs_filtering_mode)

        # decoder_laps_radon_transform_df_dict ## ~4m
        ## OUTPUTS: decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, decoder_laps_filter_epochs_decoder_result_dict, decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict
        return decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict


@function_attributes(short_name=None, tags=['add_position_columns', 'decoded_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-03 17:14', related_items=[])
def _helper_add_interpolated_position_columns_to_decoded_result_df(a_result: DecodedFilterEpochsResult, a_decoder: BasePositionDecoder, a_decoded_marginal_posterior_df: pd.DataFrame, global_measured_position_df: pd.DataFrame):
    """ adds the measured positions as columns in the decoded epochs/epoch_time_bins df
    
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df

        ## INPUTS: curr_active_pipeline, a_result, a_decoder, a_decoded_marginal_posterior_df

        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        
        a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)
    
    """
    from neuropy.utils.indexing_helpers import PandasHelpers

    ## add in measured position columns
    global_measured_position_df: pd.DataFrame = deepcopy(global_measured_position_df) # computation_result.sess.position.to_dataframe()
    global_all_directional_decoder_result: CustomDecodeEpochsResult = CustomDecodeEpochsResult.init_from_single_decoder_decoding_result_and_measured_pos_df(a_result, global_measured_position_df=global_measured_position_df)

    xbin_edges = deepcopy(a_decoder.xbin)
    # ybin_edges = deepcopy(a_decoder.ybin)
    # xbin_edges = None
    ybin_edges = None
    active_computation_config = deepcopy(a_decoder.pf.config)
    # grid_bin_bounds = deepcopy(a_decoder.pf.config.grid_bin_bounds)

    measured_positions_df: pd.DataFrame = deepcopy(global_all_directional_decoder_result.measured_decoded_position_comparion.measured_positions_dfs_list[0])
    measured_positions_df = measured_positions_df.position.adding_binned_position_columns(xbin_edges=xbin_edges, ybin_edges=ybin_edges, active_computation_config=active_computation_config)
    measured_positions_df = measured_positions_df.rename(columns={'x':'x_meas', 'y':'y_meas', 'binned_x':'binned_x_meas', 'binned_y':'binned_y_meas'}, inplace=False).drop(columns=['t'], inplace=False) ## measured

    decoded_positions_df: pd.DataFrame = deepcopy(global_all_directional_decoder_result.measured_decoded_position_comparion.decoded_positions_df_list[0])
    decoded_positions_df = decoded_positions_df.position.adding_binned_position_columns(xbin_edges=xbin_edges, ybin_edges=ybin_edges, active_computation_config=active_computation_config)
    decoded_positions_df = decoded_positions_df.rename(columns={'x':'x_decoded_most_likely', 'binned_x':'binned_x_decoded_most_likely'}, inplace=False).drop(columns=['t'], inplace=False) ## decoded most likely

    ## OUTPUTS: measured_positions_df, decoded_positions_df

    ## Add directly to `a_decoded_marginal_posterior_df` (concatenating columns to existing rows)
    a_decoded_marginal_posterior_df = PandasHelpers.adding_additional_df_columns(original_df=a_decoded_marginal_posterior_df,
                                                                                additional_cols_df=measured_positions_df,
                                                                                )

    a_decoded_marginal_posterior_df = PandasHelpers.adding_additional_df_columns(original_df=a_decoded_marginal_posterior_df,
                                                                                additional_cols_df=decoded_positions_df,
                                                                                )
    return a_decoded_marginal_posterior_df



# ==================================================================================================================== #
# Main Computation Functions                                                                                           #
# ==================================================================================================================== #
class DirectionalPlacefieldGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'directional_pfs'
    _computationPrecidence = 1000
    _is_global = True

    @function_attributes(short_name='split_to_directional_laps', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'],
                          input_requires=["computation_results[an_epoch_name]['pf1D_Decoder']"], output_provides=[],
                          uses=['DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs'], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
        provides_global_keys=['DirectionalLaps'],
        validate_computation_test=validate_has_directional_laps, is_global=True)
    def _split_to_directional_laps(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Splits the existing laps into directional versions
    
        laps_obj.update_lap_dir_from_smoothed_velocity(pos_input=curr_active_pipeline.sess.position)
        laps_obj.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
    
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['DirectionalLaps']
                ['DirectionalLaps']['directional_lap_specific_configs']
                ['DirectionalLaps']['split_directional_laps_dict']
                ['DirectionalLaps']['split_directional_laps_contexts_dict']
                ['DirectionalLaps']['split_directional_laps_names']
                ['DirectionalLaps']['computed_base_epoch_names']

        """
        if include_includelist is not None:
            print(f'WARN: _split_to_directional_laps(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        # Set the global result:
        global_computation_results.computed_data['DirectionalLaps'] = DirectionalLapsResult.init_from_pipeline_natural_epochs(owning_pipeline_reference) # Uses `curr_active_pipeline.computation_results[an_epoch_name].computed_data.get('pf1D_Decoder', None)`

        ## NOTE: Needs to call `owning_pipeline_reference.prepare_for_display()` before display functions can be used with new directional results

        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        """
        return global_computation_results



    @function_attributes(short_name='merged_directional_placefields', tags=['directional_pf', 'laps', 'epoch', 'replay', 'session', 'pf1D', 'pf2D'],
                       input_requires=['filtered_sessions[global_epoch_name].replay', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz', 'global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values'], output_provides=[],
                       uses=['PfND.build_merged_directional_placefields'], used_by=[], creation_date='2023-10-25 09:33', related_items=['DirectionalPseudo2DDecodersResult'],
        requires_global_keys=['DirectionalLaps'], provides_global_keys=['DirectionalMergedDecoders'],
        validate_computation_test=validate_has_directional_merged_placefields, is_global=True)
    def _build_merged_directional_placefields(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False,
                                                laps_decoding_time_bin_size: float = 0.250, # 250ms
                                                ripple_decoding_time_bin_size: float = 0.025, # 25ms
                                                should_validate_lap_decoding_performance: bool = False,
                                            ):
        """ Merges the computed directional placefields into a Pseudo2D decoder, with the pseudo last axis corresponding to the decoder index.

        NOTE: this builds a **decoder** not just placefields, which is why it depends on the time_bin_sizes (which will later be used for decoding)		

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['DirectionalMergedDecoders']
                ['DirectionalMergedDecoders']['directional_lap_specific_configs']
                ['DirectionalMergedDecoders']['split_directional_laps_dict']
                ['DirectionalMergedDecoders']['split_directional_laps_contexts_dict']
                ['DirectionalMergedDecoders']['split_directional_laps_names']
                ['DirectionalMergedDecoders']['computed_base_epoch_names']


                directional_merged_decoders_result: "DirectionalPseudo2DDecodersResult" = global_computation_results.computed_data['DirectionalMergedDecoders']

        """
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        # global_computation_results is not None, but `global_computation_results.computation_config` is. Why?
        assert global_computation_results.computation_config is not None   
        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz # #TODO 2024-12-10 13:21: - [ ] Failing here --- no clue why, global_computation_results is None
        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        # included_aclus = owning_pipeline_reference.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values, debug_print=True)
        

        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        global_session = owning_pipeline_reference.filtered_sessions[global_epoch_name]
        
        unfiltered_session = deepcopy(owning_pipeline_reference.sess)
        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used in 

        # Unwrap the naturally produced directional placefields:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [owning_pipeline_reference.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        # (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [owning_pipeline_reference.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [owning_pipeline_reference.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
       
        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name

        # Use the four epochs to make to a pseudo-y:
        all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        all_directional_pf_dict = dict(zip(all_directional_decoder_names, [deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D)]))
        # all_directional_decoder_dict = {a_name:a_decoder.get_by_id(ids=included_aclus) for a_name, a_decoder in all_directional_decoder_dict.items()} # slice by included aclus
        all_directional_pf_dict, filtered_direction_shared_aclus_list = PfND.determine_pf_aclus_filtered_by_frate_and_qclu(pf_dict=all_directional_pf_dict, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        

        all_directional_pf1D = PfND.build_merged_directional_placefields(all_directional_pf_dict, debug_print=False)
        all_directional_pf1D_Decoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        # Inputs: long_LR_pf1D, long_RL_pf1D
        long_directional_decoder_names = ['long_LR', 'long_RL']
        # long_directional_decoder_dict = dict(zip(long_directional_decoder_names, [deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D)]))
        long_directional_decoder_dict = {k:deepcopy(all_directional_pf_dict[k]) for k in long_directional_decoder_names}
        long_directional_pf1D = PfND.build_merged_directional_placefields(long_directional_decoder_dict, debug_print=False)
        long_directional_pf1D_Decoder = BasePositionDecoder(long_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        # Inputs: short_LR_pf1D, short_RL_pf1D
        short_directional_decoder_names = ['short_LR', 'short_RL']
        # short_directional_decoder_dict = dict(zip(short_directional_decoder_names, [deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D)]))
        short_directional_decoder_dict = {k:deepcopy(all_directional_pf_dict[k]) for k in short_directional_decoder_names}
        short_directional_pf1D = PfND.build_merged_directional_placefields(short_directional_decoder_dict, debug_print=False)
        short_directional_pf1D_Decoder = BasePositionDecoder(short_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # takes 6.3 seconds

        ## Get or update the global directional_merged_decoders_result:
        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data.get('DirectionalMergedDecoders', DirectionalPseudo2DDecodersResult(all_directional_decoder_dict=all_directional_pf_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder))


        directional_merged_decoders_result.__dict__.update(all_directional_decoder_dict=all_directional_pf_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        
        



        # Decode Epochs (Laps/Ripples) Using the merged all-directional decoder): ____________________________________________ #
        ## this is where the time_bin_size dependent part comes in
        
        ## Decode Laps:
        if (laps_decoding_time_bin_size is not None):
            global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_any_name].computation_config.pf_params.computation_epochs) # global_any_name='maze_any' (? same as global_epoch_name?)
            global_any_laps_epochs_df = ensure_dataframe(global_any_laps_epochs_obj)    
            global_any_laps_epochs_df = global_any_laps_epochs_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            global_any_laps_epochs_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=global_any_laps_epochs_df, global_session=deepcopy(global_session), replace_existing=True)
            directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)),
                                                                                                                                               filter_epochs=deepcopy(global_any_laps_epochs_df), decoding_time_bin_size=laps_decoding_time_bin_size, debug_print=False)
        else:
            print(f'skipping lap recomputation because laps_decoding_time_bin_size == None')
            should_validate_lap_decoding_performance = False


        ## Decode Ripples (only uses the global_epoch_name's ripples):
        if (ripple_decoding_time_bin_size is not None):
            global_replays: pd.DataFrame = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(ensure_dataframe(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay)))
            min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
            # ripple_decoding_time_bin_size: float = min(0.010, min_possible_time_bin_size) # 10ms # 0.002
            if ripple_decoding_time_bin_size < min_possible_time_bin_size:
                print(f'WARN: ripple_decoding_time_bin_size {ripple_decoding_time_bin_size} < min_possible_time_bin_size ({min_possible_time_bin_size}). This used to be enforced but continuing anyway.') 
            global_replays = global_replays.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(get_proper_global_spikes_df(owning_pipeline_reference, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)), filter_epochs=deepcopy(global_replays), decoding_time_bin_size=ripple_decoding_time_bin_size)
        else:
            print(f'skipping ripple recomputation because ripple_decoding_time_bin_size == None')

        ## Post Compute Validations:
        directional_merged_decoders_result.perform_compute_marginals()
        
        # Validate Laps:
        if should_validate_lap_decoding_performance:
            result_laps_epochs_df: pd.DataFrame = directional_merged_decoders_result.add_groundtruth_information(owning_pipeline_reference)
            try:
                laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.laps_directional_marginals_tuple
                percent_laps_estimated_correctly = DirectionalPseudo2DDecodersResult.validate_lap_dir_estimations(unfiltered_session, active_global_laps_df=global_any_laps_epochs_obj.to_dataframe(), laps_is_most_likely_direction_LR_dir=laps_is_most_likely_direction_LR_dir)
                print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
            except (AssertionError, ValueError) as err:
                print(F'fails due to some types thing?')
                print(f'\terr: {err}')
                pass
        
        # Set the global result:
        global_computation_results.computed_data['DirectionalMergedDecoders'] = directional_merged_decoders_result
        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        ripple_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result
        
        """
        return global_computation_results
    

    @function_attributes(short_name='directional_decoders_decode_continuous', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D', 'continuous'], input_requires=[], output_provides=[], uses=['DirectionalDecodersContinuouslyDecodedResult'], used_by=[], creation_date='2024-01-17 09:05', related_items=[],
        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['DirectionalDecodersDecoded'],
        # validate_computation_test=DirectionalDecodersContinuouslyDecodedResult.validate_has_directional_decoded_continuous_epochs,
        validate_computation_test=_workaround_validate_has_directional_decoded_continuous_epochs,
        is_global=True, computation_precidence=(1002.0))
    def _decode_continuous_using_directional_decoders(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, time_bin_size: Optional[float]=None, should_disable_cache: bool = False):
        """ Using the four 1D decoders, decodes continously streams of positions from the neural activity for each.
        
        should_disable_cache: bool = True # when True, always recomputes and does not attempt to use the cache.
        
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['pf1D_Decoder_dict']
                ['DirectionalDecodersDecoded']['directional_lap_specific_configs']
                ['DirectionalDecodersDecoded']['continuously_decoded_result_cache_dict']


                from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult

                directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
                all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
                pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder

                # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
                time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
                print(f'time_bin_size: {time_bin_size}')
                continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
                pseudo2D_decoder_continuously_decoded_result = continuously_decoded_dict.get('pseudo2D', None)



        """
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        from neuropy.core.epoch import Epoch
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder
                
        # directional_decoders_decode_result = global_computation_results.computed_data.get('DirectionalDecodersDecoded', DirectionalDecodersContinuouslyDecodedResult(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict, continuously_decoded_result_cache_dict=continuously_decoded_result_cache_dict))
        # Store all_directional_pf1D_Decoder_dict, all_directional_continuously_decoded_dict
        
        ## Create or update the global directional_decoders_decode_result:            
        directional_decoders_decode_result = global_computation_results.computed_data.get('DirectionalDecodersDecoded', None)
        had_existing_DirectionalDecodersDecoded_result: bool = (directional_decoders_decode_result is not None)

        if should_disable_cache:
            print(f'should_disable_cache == True so setting had_existing_DirectionalDecodersDecoded_result = False')
            had_existing_DirectionalDecodersDecoded_result = False
            directional_decoders_decode_result = None # set to None

        did_update_computations: bool = False
        ## Currently used for both cases to decode:
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        single_global_epoch: Epoch = Epoch(single_global_epoch_df)
        
        if (not had_existing_DirectionalDecodersDecoded_result):
            # Build a new result _________________________________________________________________________________________________ #
            print(f'\thad_existing_DirectionalDecodersDecoded_result == False. New DirectionalDecodersContinuouslyDecodedResult will be built...')
            # # Unpack all directional variables:
            long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
            # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
            (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
            (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [owning_pipeline_reference.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
            (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

            all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = dict(zip(all_directional_decoder_names, [deepcopy(long_LR_pf1D_Decoder), deepcopy(long_RL_pf1D_Decoder), deepcopy(short_LR_pf1D_Decoder), deepcopy(short_RL_pf1D_Decoder)]))

            # DirectionalMergedDecoders: Get the result after computation:
            directional_merged_decoders_result = owning_pipeline_reference.global_computation_results.computed_data['DirectionalMergedDecoders'] # uses `DirectionalMergedDecoders`.

            # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_merged_decoders_result.all_directional_decoder_dict # This does not work, because the values in the returned dictionary are PfND, not 1D decoders
            all_directional_pf1D_Decoder_value = directional_merged_decoders_result.all_directional_pf1D_Decoder

            pseudo2D_decoder: BasePositionDecoder = all_directional_pf1D_Decoder_value
            
            ## Build Epoch object across whole sessions:
            if time_bin_size is None:
                # use default time_bin_size from the previous decoder
                # first_decoder = list(all_directional_pf1D_Decoder_dict.values())[0]
                # time_bin_size = first_decoder.time_bin_size
                time_bin_size = directional_merged_decoders_result.ripple_decoding_time_bin_size


            # time_binning_container: BinningContainer = deepcopy(long_LR_pf1D_Decoder.time_binning_container)
            # time_binning_container.edges # array([31.8648, 31.8978, 31.9308, ..., 1203.56, 1203.6, 1203.63])
            # time_binning_container.centers # array([31.8813, 31.9143, 31.9473, ..., 1203.55, 1203.58, 1203.61])
            print(f'\ttime_bin_size: {time_bin_size}')

            # Get proper global_spikes_df:
            global_spikes_df = get_proper_global_spikes_df(owning_pipeline_reference)

            spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

            # print(f'add_directional_decoder_decoded_epochs(...): decoding continuous epochs for each directional decoder.')
            # t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            # single_global_epoch: Epoch = Epoch(pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})) # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session
            # global_spikes_df, _, _ = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=owning_pipeline_reference, num_shuffles=0) # does not do shuffling
            # spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)
            all_directional_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = {k:v.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(single_global_epoch), decoding_time_bin_size=time_bin_size, debug_print=False) for k,v in all_directional_pf1D_Decoder_dict.items()}
            pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = pseudo2D_decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(single_global_epoch), decoding_time_bin_size=time_bin_size, debug_print=False)
            all_directional_continuously_decoded_dict['pseudo2D'] = pseudo2D_decoder_continuously_decoded_result
            continuously_decoded_result_cache_dict = {time_bin_size:all_directional_continuously_decoded_dict} # result is a single time_bin_size
            print(f'\t computation done. Creating new DirectionalDecodersContinuouslyDecodedResult....')
            directional_decoders_decode_result = DirectionalDecodersContinuouslyDecodedResult(pseudo2D_decoder=pseudo2D_decoder, pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict, spikes_df=deepcopy(global_spikes_df), continuously_decoded_result_cache_dict=continuously_decoded_result_cache_dict)
            did_update_computations = True

        else:
            # had_existing_DirectionalDecodersDecoded_result == True _____________________________________________________________ #
            print(f'\thad_existing_DirectionalDecodersDecoded_result == True. Using existing result and updating.')
            ## Try to get the existing results to reuse:
            all_directional_pf1D_Decoder_dict = directional_decoders_decode_result.pf1D_Decoder_dict
            pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder # pseudo2D_decoder: missing .spikes_df (is None)
            spikes_df = directional_decoders_decode_result.spikes_df
            continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
            previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
            # In future could extract `single_global_epoch` from the previously decoded result:
            # first_decoded_result = continuously_decoded_result_cache_dict[previously_decoded_keys[0]]

            ## Get the current time_bin_size:
            if time_bin_size is None:
                # use default time_bin_size from the previous decoder
                first_decoder = list(all_directional_pf1D_Decoder_dict.values())[0]
                time_bin_size = first_decoder.time_bin_size
                
            print(f'\ttime_bin_size: {time_bin_size}')
            
            needs_recompute: bool = (time_bin_size not in previously_decoded_keys)
            if needs_recompute:
                print(f'\t\trecomputing for time_bin_size: {time_bin_size}...')
                ## Recompute here only:
                all_directional_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = {k:v.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=single_global_epoch, decoding_time_bin_size=time_bin_size, debug_print=False) for k,v in all_directional_pf1D_Decoder_dict.items()}
                pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = pseudo2D_decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=single_global_epoch, decoding_time_bin_size=time_bin_size, debug_print=False)
                all_directional_continuously_decoded_dict['pseudo2D'] = pseudo2D_decoder_continuously_decoded_result
                # directional_decoders_decode_result.__dict__.update(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict)
                directional_decoders_decode_result.continuously_decoded_result_cache_dict[time_bin_size] = all_directional_continuously_decoded_dict # update the entry for this time_bin_size
                did_update_computations = True
                
            else:
                print(f'(time_bin_size == {time_bin_size}) already found in cache. Not recomputing.')

        # Set the global result:
        global_computation_results.computed_data['DirectionalDecodersDecoded'] = directional_decoders_decode_result
        
        if did_update_computations:
            ## #TODO 2025-06-13 13:46: - [ ] try to add to generic results
            # try:
            #     valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
            #     a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result

            # except Exception as e:
            #     print(f'failed to add continuous computation to the generic results with error e: {e}. Skipping.')
            pass
                       
        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult

        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
        
        """
        return global_computation_results

    @function_attributes(short_name='directional_decoders_evaluate_epochs', tags=['directional-decoders', 'epochs', 'decode', 'score', 'weighted-correlation', 'radon-transform', 'multiple-decoders', 'main-computation-function'],
                          input_requires=['global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'], output_provides=[], uses=['_perform_compute_custom_epoch_decoding', '_compute_all_df_score_metrics'], used_by=[], creation_date='2024-02-16 12:49', related_items=['DecoderDecodedEpochsResult'],
                         requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['DirectionalDecodersEpochsEvaluations'],
                         validate_computation_test=_workaround_validate_has_directional_decoded_epochs_evaluations,
                        is_global=True, computation_precidence=(1002.1))
    def _decode_and_evaluate_epochs_using_directional_decoders(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, should_skip_radon_transform=False):
        """ Using the four 1D decoders, performs 1D Bayesian decoding for each of the known epochs (Laps, Ripple) from the neural activity during these peirods.
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['pf1D_Decoder_dict']
                ['DirectionalDecodersEpochsEvaluations']['directional_lap_specific_configs']
                ['DirectionalDecodersEpochsEvaluations']['continuously_decoded_result_cache_dict']


                from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

                directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
                pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
                ripple_decoding_time_bin_size = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
                laps_decoding_time_bin_size = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
                decoder_laps_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
                decoder_ripple_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict
                decoder_laps_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_df_dict
                decoder_ripple_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_df_dict

                # New items:
                decoder_laps_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_extras_dict
                decoder_ripple_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_extras_dict

                # Weighted correlations:
                laps_weighted_corr_merged_df = directional_decoders_epochs_decode_result.laps_weighted_corr_merged_df
                ripple_weighted_corr_merged_df = directional_decoders_epochs_decode_result.ripple_weighted_corr_merged_df
                decoder_laps_weighted_corr_df_dict = directional_decoders_epochs_decode_result.decoder_laps_weighted_corr_df_dict
                decoder_ripple_weighted_corr_df_dict = directional_decoders_epochs_decode_result.decoder_ripple_weighted_corr_df_dict

                # Pearson's correlations:
                laps_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result.laps_simple_pf_pearson_merged_df
                ripple_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result.ripple_simple_pf_pearson_merged_df

                
        Should call:
        
        _perform_compute_custom_epoch_decoding


        """
        # ==================================================================================================================== #
        # BEGIN SUBFN                                                                                                          #
        # ==================================================================================================================== #

        # Calls `_perform_compute_custom_epoch_decoding`, `_compute_all_df_score_metrics`

        # directional_decoders_decode_result = global_computation_results.computed_data.get('DirectionalDecodersDecoded', DirectionalDecodersContinuouslyDecodedResult(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict, continuously_decoded_result_cache_dict=continuously_decoded_result_cache_dict))
        
        # spikes_df = curr_active_pipeline.sess.spikes_df
        # rank_order_results = global_computation_results.computed_data['RankOrder'] # : "RankOrderComputationsContainer"
        # minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        # included_qclu_values: List[int] = rank_order_results.included_qclu_values
        
        if global_computation_results.computation_config is None:
            raise NotImplementedError(f'global_computation_results.computation_config is None!')
        
        minimum_inclusion_fr_Hz = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        # num_shuffles = global_computation_results.computation_config.rank_order_shuffle_analysis.num_shuffles

        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        # print(f'included_qclu_values: {included_qclu_values}')

        # DirectionalMergedDecoders: Get the result after computation:
        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data['DirectionalMergedDecoders']
        ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size
        pos_bin_size: float = track_templates.get_decoders()[0].pos_bin_size

        print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}, ripple_decoding_time_bin_size: {ripple_decoding_time_bin_size}, pos_bin_size: {pos_bin_size}')
        
        # epochs_filtering_mode: EpochFilteringMode = EpochFilteringMode.DropShorter - Matches WCorrShuffles
        epochs_filtering_mode: EpochFilteringMode = EpochFilteringMode.ConstrainDecodingTimeBinSizeToMinimum # Pre 2024-06-03 Way 

        decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = _perform_compute_custom_epoch_decoding(owning_pipeline_reference, directional_merged_decoders_result=directional_merged_decoders_result, track_templates=track_templates, epochs_filtering_mode=epochs_filtering_mode) # Dict[str, Optional[DecodedFilterEpochsResult]]

        ## Recompute the epoch scores/metrics such as radon transform and wcorr:
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(directional_merged_decoders_result, track_templates,
                                                                                                                                                                                            decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df),
                                                                                                                                                                                            should_skip_radon_transform=should_skip_radon_transform, suppress_exceptions=False)
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df, laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df = merged_df_outputs_tuple
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_extras_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple

        #TODO 2024-02-16 13:46: - [ ] Currently always replace
        ## Create or update the global directional_merged_decoders_result:
        # directional_decoders_epochs_decode_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
        # if directional_decoders_epochs_decode_result is not None:
        # directional_decoders_epochs_decode_result.__dict__.update(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
        #                                               long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
        #                                               short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = DecoderDecodedEpochsResult(is_global=True, **{'pos_bin_size': pos_bin_size, 'ripple_decoding_time_bin_size':ripple_decoding_time_bin_size, 'laps_decoding_time_bin_size':laps_decoding_time_bin_size,
                                                                                                'decoder_laps_filter_epochs_decoder_result_dict':decoder_laps_filter_epochs_decoder_result_dict,
            'decoder_ripple_filter_epochs_decoder_result_dict':decoder_ripple_filter_epochs_decoder_result_dict, 'decoder_laps_radon_transform_df_dict':decoder_laps_radon_transform_df_dict, 'decoder_ripple_radon_transform_df_dict':decoder_ripple_radon_transform_df_dict,
            'decoder_laps_radon_transform_extras_dict': decoder_laps_radon_transform_extras_dict, 'decoder_ripple_radon_transform_extras_dict': decoder_ripple_radon_transform_extras_dict,
            'laps_weighted_corr_merged_df': laps_weighted_corr_merged_df, 'ripple_weighted_corr_merged_df': ripple_weighted_corr_merged_df, 'decoder_laps_weighted_corr_df_dict': decoder_laps_weighted_corr_df_dict, 'decoder_ripple_weighted_corr_df_dict': decoder_ripple_weighted_corr_df_dict,
            'laps_simple_pf_pearson_merged_df': laps_simple_pf_pearson_merged_df, 'ripple_simple_pf_pearson_merged_df': ripple_simple_pf_pearson_merged_df,
            })


        # Set the global result:
        global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations'] = directional_decoders_epochs_decode_result
        

        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
        ## UNPACK HERE via direct property access:
        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        ripple_decoding_time_bin_size = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
        decoder_laps_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
        decoder_ripple_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict
        decoder_laps_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_df_dict
        decoder_ripple_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_df_dict

        # New items:
        decoder_laps_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_extras_dict
        decoder_ripple_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_extras_dict

        # Weighted correlations:
        laps_weighted_corr_merged_df = directional_decoders_epochs_decode_result.laps_weighted_corr_merged_df
        ripple_weighted_corr_merged_df = directional_decoders_epochs_decode_result.ripple_weighted_corr_merged_df
        decoder_laps_weighted_corr_df_dict = directional_decoders_epochs_decode_result.decoder_laps_weighted_corr_df_dict
        decoder_ripple_weighted_corr_df_dict = directional_decoders_epochs_decode_result.decoder_ripple_weighted_corr_df_dict

        # Pearson's correlations:
        laps_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result.laps_simple_pf_pearson_merged_df
        ripple_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result.ripple_simple_pf_pearson_merged_df

        
        ## OLD METHOD: UNPACK HERE via direct property access:
        # Extract as a dict to allow unpacking like before:
        directional_decoders_epochs_decode_result_dict = directional_decoders_epochs_decode_result.__dict__.copy()
        pos_bin_size: float = directional_decoders_epochs_decode_result_dict['pos_bin_size']
        ripple_decoding_time_bin_size = directional_decoders_epochs_decode_result_dict['ripple_decoding_time_bin_size']
        laps_decoding_time_bin_size = directional_decoders_epochs_decode_result_dict['laps_decoding_time_bin_size']
        decoder_laps_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result_dict['decoder_laps_filter_epochs_decoder_result_dict']
        decoder_ripple_filter_epochs_decoder_result_dict = directional_decoders_epochs_decode_result_dict['decoder_ripple_filter_epochs_decoder_result_dict']
        decoder_laps_radon_transform_df_dict = directional_decoders_epochs_decode_result_dict['decoder_laps_radon_transform_df_dict']
        decoder_ripple_radon_transform_df_dict = directional_decoders_epochs_decode_result_dict['decoder_ripple_radon_transform_df_dict']
        ## New 2024-02-14 - Noon:
        decoder_laps_radon_transform_extras_dict = directional_decoders_epochs_decode_result_dict['decoder_laps_radon_transform_extras_dict']
        decoder_ripple_radon_transform_extras_dict = directional_decoders_epochs_decode_result_dict['decoder_ripple_radon_transform_extras_dict']
        ## New 2024-02-16 _ Weighted Corr
        laps_weighted_corr_merged_df = directional_decoders_epochs_decode_result_dict['laps_weighted_corr_merged_df']
        ripple_weighted_corr_merged_df = directional_decoders_epochs_decode_result_dict['ripple_weighted_corr_merged_df']
        decoder_laps_weighted_corr_df_dict = directional_decoders_epochs_decode_result_dict['decoder_laps_weighted_corr_df_dict']
        decoder_ripple_weighted_corr_df_dict = directional_decoders_epochs_decode_result_dict['decoder_ripple_weighted_corr_df_dict']

        laps_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result_dict['laps_simple_pf_pearson_merged_df']
        ripple_simple_pf_pearson_merged_df = directional_decoders_epochs_decode_result_dict['ripple_simple_pf_pearson_merged_df']

        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
        
        """
        return global_computation_results


    @function_attributes(short_name='directional_decoders_epoch_heuristic_scoring', tags=['heuristic', 'directional-decoders', 'epochs', 'filter', 'score', 'weighted-correlation', 'radon-transform', 'multiple-decoders', 'main-computation-function'],
                       input_requires=['global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'], output_provides=[],
                       uses=['filter_and_update_epochs_and_spikes', 'HeuristicReplayScoring.compute_all_heuristic_scores', 'HeuristicsResult'], used_by=[], creation_date='2024-03-12 17:23', related_items=[],
        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders', 'DirectionalDecodersDecoded', 'DirectionalDecodersEpochsEvaluations'], provides_global_keys=['Heuristics'],
        validate_computation_test=_workaround_validate_has_directional_decoded_epochs_heuristic_scoring, 
                        is_global=True, computation_precidence=1002.2)
    def _decoded_epochs_heuristic_scoring(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, same_thresh_fraction_of_track: float=0.05, max_ignore_bins:float=2, max_jump_distance_cm: float=60.0, use_bin_units_instead_of_realworld:bool=False):
        """ Using the four 1D decoders, performs 1D Bayesian decoding for each of the known epochs (Laps, Ripple) from the neural activity during these peirods.
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['pf1D_Decoder_dict']
                ['DirectionalDecodersEpochsEvaluations']['directional_lap_specific_configs']
                ['DirectionalDecodersEpochsEvaluations']['continuously_decoded_result_cache_dict']
                
                
                from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicsResult

                a_heuristics_result: HeuristicsResult = curr_active_pipeline.global_computation_results.computed_data['Heuristics']
                
        Should call:
        
        _perform_compute_custom_epoch_decoding

        """
        from neuropy.core.epoch import TimeColumnAliasesProtocol
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes
        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicsResult

        if global_computation_results.computation_config is None:
            raise NotImplementedError(f'global_computation_results.computation_config is None!')

        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

        # rank_order_results = global_computation_results.computed_data['RankOrder'] # : "RankOrderComputationsContainer"
        # minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        # included_qclu_values: List[int] = rank_order_results.included_qclu_values
        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        # print(f'included_qclu_values: {included_qclu_values}')

        decoder_track_length_dict = track_templates.get_track_length_dict() # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}

        # same_thresh_fraction_of_track: float = 0.025 ## up to 2.5% of the track -- notably worse
        # same_thresh_fraction_of_track: float = 0.05 ## up to 5.0% of the track
        
        # same_thresh_fraction_of_track: float = 0.15 ## up to 15% of the track
        same_thresh_cm_dict: Dict[types.DecoderName, float] = {k:(v * same_thresh_fraction_of_track) for k, v in decoder_track_length_dict.items()}
        # same_thresh_n_bin_units: float = {k:(v * same_thresh_fraction_of_track) for k, v in decoder_track_length_dict.items()}

        a_same_thresh_cm: float = same_thresh_cm_dict['long_LR'] # ALWAYS USE THE SAME FOR ALL TRACKS/DECODERS
        # a_same_thresh_cm: float = 0.0
        print(f'same_thresh_cm: {a_same_thresh_cm}')


        # DirectionalMergedDecoders: Get the result after computation:
        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data['DirectionalMergedDecoders']
        ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size

        # DirectionalDecodersEpochsEvaluations
        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
        decoder_laps_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
        decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict


        #TODO 2024-02-16 13:46: - [ ] Currently always replace
        ## Create or update the global directional_merged_decoders_result:
        # directional_decoders_epochs_decode_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
        # if directional_decoders_epochs_decode_result is not None:
        # directional_decoders_epochs_decode_result.__dict__.update(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
        #                                               long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
        #                                               short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        

        # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores` ______________________________________________________________ #
        decoder_ripple_filter_epochs_decoder_result_dict, ripple_out_new_scores, ripple_partition_result_dict = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict,
                                                                    same_thresh_cm=a_same_thresh_cm, max_ignore_bins=max_ignore_bins, max_jump_distance_cm=max_jump_distance_cm, use_bin_units_instead_of_realworld=use_bin_units_instead_of_realworld)

        # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores` ______________________________________________________________ #
        decoder_laps_filter_epochs_decoder_result_dict, _laps_out_new_scores, _laps_partition_result_dict = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict,
                                                                    same_thresh_cm=a_same_thresh_cm, max_ignore_bins=max_ignore_bins, max_jump_distance_cm=max_jump_distance_cm, use_bin_units_instead_of_realworld=use_bin_units_instead_of_realworld)


        a_heuristics_result = HeuristicsResult(is_global=True, heuristic_scores_df_dict=ripple_out_new_scores, partition_result_dict=ripple_partition_result_dict)
        
        # 'Heuristics'
        global_computation_results.computed_data['Heuristics'] = a_heuristics_result

        return global_computation_results


    @function_attributes(short_name='directional_train_test_split', tags=['train-test-split', 'global_computation'],
                        input_requires=['global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'], output_provides=[], uses=['TrainTestLapsSplitting.compute_train_test_split_laps_decoders'], used_by=[], creation_date='2024-04-09 06:09', related_items=[],
                        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['TrainTestSplit'],
                        validate_computation_test=_workaround_validate_has_directional_train_test_split_result, 
                        is_global=True, computation_precidence=(1002.3))
    def _split_train_test_laps_data(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False,
                                    training_data_portion: float = (5.0/6.0), debug_output_hdf5_file_path = None):
        """ Using the four 1D decoders, performs 1D Bayesian decoding for each of the known epochs (Laps, Ripple) from the neural activity during these peirods.
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['TrainTestSplit']
                ['DirectionalDecodersEpochsEvaluations']['directional_lap_specific_configs']
                ['DirectionalDecodersEpochsEvaluations']['continuously_decoded_result_cache_dict']
                a_train_test_result = global_computation_results.computed_data['TrainTestSplit']
        """
        if global_computation_results.computation_config is None:
            raise NotImplementedError(f'global_computation_results.computation_config is None!')

        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        # print(f'included_qclu_values: {included_qclu_values}')

        test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration
        print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

        # decoders_dict = deepcopy(track_templates.get_decoders_dict())

        # debug_output_hdf5_file_path = Path('output', 'laps_train_test_split.h5').resolve()
        debug_output_hdf5_file_path = None
        a_train_test_result: TrainTestSplitResult = TrainTestLapsSplitting.compute_train_test_split_laps_decoders(directional_laps_results=directional_laps_results, track_templates=track_templates, training_data_portion=training_data_portion,
                                                                                                                                    debug_output_hdf5_file_path=debug_output_hdf5_file_path, debug_plot=False, debug_print=debug_print)  # type: Tuple[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, BasePositionDecoder], Any]

        global_computation_results.computed_data['TrainTestSplit'] = a_train_test_result
        

        ## decode:
        

        return global_computation_results


    @function_attributes(short_name='trial_by_trial_metrics', tags=['trial_by_trial', 'global_computation'],
                           input_requires=["owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf1D_dt']",
                            'global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'],
                         output_provides=[], uses=['TrialByTrialActivity','TrialByTrialActivityResult'], used_by=[], creation_date='2024-05-28 00:00', related_items=[],
                        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['TrialByTrialActivity'],
                        validate_computation_test=_workaround_validate_has_directional_trial_by_trial_activity_result, 
                        is_global=True, computation_precidence=(1002.4))
    def _build_trial_by_trial_activity_metrics(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Analyzes the trial-by-trial changes (such as the lap-to-lap correlations in the placefields for each cell)
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['TrialByTrialActivity']
                ['DirectionalDecodersEpochsEvaluations']['directional_lap_specific_configs']
                ['DirectionalDecodersEpochsEvaluations']['continuously_decoded_result_cache_dict']
                a_train_test_result = global_computation_results.computed_data['TrialByTrialActivity']

        Usage:

            from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult

            directional_trial_by_trial_activity_result: TrialByTrialActivityResult = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)
            any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
            active_pf_dt: PfND_TimeDependent = directional_trial_by_trial_activity_result.active_pf_dt
            directional_lap_epochs_dict: Dict[str, Epoch] = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
            directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
            directional_active_lap_pf_results_dicts


    
        """
        from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
        from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

        if global_computation_results.computation_config is None:
            raise NotImplementedError(f'global_computation_results.computation_config is None!')

        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = track_templates.get_decoders()
        minimum_inclusion_fr_Hz = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        # Unpack all directional variables:
        ## {"even": "RL", "odd": "LR"}
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [owning_pipeline_reference.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        
        ## INPUTS: curr_active_pipeline, track_templates, global_epoch_name, (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)
        any_decoder_neuron_IDs: NDArray = deepcopy(track_templates.any_decoder_neuron_IDs)
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()

        # ## Directional Trial-by-Trial Activity:
        if 'pf1D_dt' not in owning_pipeline_reference.computation_results[global_epoch_name].computed_data:
            # if `KeyError: 'pf1D_dt'` recompute
            owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

        active_pf_1D_dt: PfND_TimeDependent = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf1D_dt'])
        # active_pf_2D_dt: PfND_TimeDependent = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computed_data['pf2D_dt'])

        active_pf_dt: PfND_TimeDependent = active_pf_1D_dt
        # Limit only to the placefield aclus:
        active_pf_dt = active_pf_dt.get_by_id(ids=any_decoder_neuron_IDs)

        # active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_2D_dt) # 2D
        long_LR_name, long_RL_name, short_LR_name, short_RL_name = track_templates.get_decoder_names()

        directional_lap_epochs_dict = dict(zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)))
        directional_lap_epochs_dict[global_any_name] = global_any_laps_epochs_obj # add the global epoch as well

        directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = TrialByTrialActivity.directional_compute_trial_by_trial_correlation_matrix(active_pf_dt=active_pf_dt, directional_lap_epochs_dict=directional_lap_epochs_dict, included_neuron_IDs=any_decoder_neuron_IDs)

        ## OUTPUTS: directional_active_lap_pf_results_dicts
        a_trial_by_trial_result: TrialByTrialActivityResult = TrialByTrialActivityResult(any_decoder_neuron_IDs=any_decoder_neuron_IDs,
                                                                                     active_pf_dt=deepcopy(active_pf_dt),
                                                                                     directional_lap_epochs_dict=directional_lap_epochs_dict,
                                                                                     directional_active_lap_pf_results_dicts=directional_active_lap_pf_results_dicts,
                                                                                     is_global=True)  # type: Tuple[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, BasePositionDecoder], Any]

        global_computation_results.computed_data['TrialByTrialActivity'] = a_trial_by_trial_result

        return global_computation_results
    

    @function_attributes(short_name='extended_pf_peak_information', tags=['trial_by_trial', 'global_computation'], input_requires=['global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'],
                             output_provides=[], uses=[], used_by=[], creation_date='2024-05-28 00:00', related_items=[],
                        requires_global_keys=['DirectionalLaps', 'jonathan_firing_rate_analysis', 'RatemapPeaksAnalysis'],# provides_global_keys=['jonathan_firing_rate_analysis'],
                        validate_computation_test=_workaround_validate_has_extended_pf_peak_info_result, 
                        is_global=True, computation_precidence=(1005.4))
    def _add_extended_pf_peak_information(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Analyzes the trial-by-trial changes (such as the lap-to-lap correlations in the placefields for each cell
        
        Requires:
            ['sess']

        Updates:
            global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df

        """
                
        if global_computation_results.computation_config is None:
            raise NotImplementedError(f'global_computation_results.computation_config is None!')
        
        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        
        # rank_order_results = global_computation_results.computed_data['RankOrder'] # : "RankOrderComputationsContainer"
        # minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        # included_qclu_values: List[int] = rank_order_results.included_qclu_values
        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?

        jonathan_firing_rate_analysis_result = global_computation_results.computed_data.jonathan_firing_rate_analysis # JonathanFiringRateAnalysisResult
        neuron_replay_stats_df: pd.DataFrame = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
        
        ## Requires: 
        # owning_pipeline_reference.computation_results[global_context].computed_data['RatemapPeaksAnalysis']['PeakProminence2D']
        try:
            neuron_replay_stats_df, all_pf2D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=owning_pipeline_reference, track_templates=track_templates)
        except KeyError as e:
            # KeyError: 'RatemapPeaksAnalysis'
            print(f"missing ['RatemapPeaksAnalysis']['PeakProminence2D']. skipping.")
            pass
        except BaseException as e:
            raise e
        
        neuron_replay_stats_df, all_pf1D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_directional_pf_maximum_peaks(track_templates=track_templates)
        # both_included_neuron_stats_df = deepcopy(neuron_replay_stats_df[neuron_replay_stats_df['LS_pf_peak_x_diff'].notnull()]).drop(columns=['track_membership', 'neuron_type'])
        global_computation_results.computed_data['jonathan_firing_rate_analysis'] = jonathan_firing_rate_analysis_result

        return global_computation_results



# ==================================================================================================================== #
# Display Functions/Plotting                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks

from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
import plotly.express as px

def plot_all_sessions(directory, save_figures=False, figure_save_extension='.png'):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    # # Example usage:
    directory = '/home/halechr/FastData/collected_outputs/'

    all_session_figures = plot_all_sessions(directory, save_figures=True)

    # Show figures for all sessions
    for fig_laps, fig_ripples in all_session_figures:
        fig_laps.show()
        fig_ripples.show()

    """
    if not isinstance(directory, Path):
        directory = Path(directory).resolve()
    assert directory.exists()
    print(f'plot_all_sessions(directory: {directory})')
    if save_figures:
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder = Path(directory, 'figures')
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: {figures_folder}')
    
    # Get all CSV files in the specified directory
    # all_csv_files = Path(directory).glob('*-(laps|ripple)_marginals_df).csv')
    all_csv_files = sorted(Path(directory).glob('*_marginals_df).csv'))

    # Separate the CSV files into laps and ripple lists
    laps_files = [file for file in all_csv_files if 'laps' in file.stem]
    ripple_files = [file for file in all_csv_files if 'ripple' in file.stem]

    # Create an empty list to store the figures
    all_figures = []

    # Iterate through the pairs and create figures
    for laps_file, ripple_file in zip(laps_files, ripple_files):
        session_name = laps_file.stem.split('-')[3]  # Extract session name from the filename
        print(f'processing session_name: {session_name}')
        
        laps_df = pd.read_csv(laps_file)
        ripple_df = pd.read_csv(ripple_file)

        # SEPERATELY _________________________________________________________________________________________________________ #
        # Create a bubble chart for laps
        fig_laps = px.scatter(laps_df, x='lap_start_t', y='P_Long', title=f"Laps - Session: {session_name}")

        # Create a bubble chart for ripples
        fig_ripples = px.scatter(ripple_df, x='ripple_start_t', y='P_Long', title=f"Ripples - Session: {session_name}")

        if save_figures:
            # Save the figures to the 'figures' subfolder
            print(f'\tsaving figures...')
            fig_laps_name = Path(figures_folder, f"{session_name}_laps_marginal{figure_save_extension}").resolve()
            print(f'\tsaving "{fig_laps_name}"...')
            fig_laps.write_image(fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{session_name}_ripples_marginal{figure_save_extension}").resolve()
            print(f'\tsaving "{fig_ripple_name}"...')
            fig_ripples.write_image(fig_ripple_name)
        
        # Append both figures to the list
        all_figures.append((fig_laps, fig_ripples))
        
        # # COMBINED ___________________________________________________________________________________________________________ #
        # # Create a subplot with laps and ripples stacked vertically
        # fig_combined = px.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        #                                         subplot_titles=[f"Laps - Session: {session_name}", f"Ripples - Session: {session_name}"])

        # # Add scatter traces to the subplots
        # fig_combined.add_trace(px.scatter(laps_df, x='lap_start_t', y='P_Long').data[0], row=1, col=1)
        # fig_combined.add_trace(px.scatter(ripple_df, x='ripple_start_t', y='P_Long').data[0], row=2, col=1)

        # # Update layout for better visualization
        # fig_combined.update_layout(height=600, width=800, title_text=f"Combined Plot - Session: {session_name}")

        # # Save the figure to the 'figures' subfolder
        # figure_filename = Path(figures_folder, f"{session_name}_marginal.png")
        # fig_combined.write_image(figure_filename)
        
        # all_figures.append(fig_combined)
        
    return all_figures

def plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=None, t_split=1000.0, t_end=None, active_context=None, perform_write_to_file_callback=None):
    """ Plots three Matplotlib figures displaying the quantile differences
    
    
    Usage:
    
    ripple_all_epoch_bins_marginals_df
    laps_all_epoch_bins_marginals_df
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_all_epoch_bins_marginal_predictions

    _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    global_epoch = curr_active_pipeline.filtered_epochs[global_epoch_name]
    t_start, t_end = global_epoch.start_end_times
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
    split_time_t: float = short_epoch.t_start
    active_context = curr_active_pipeline.sess.get_context()

    ## Get the result after computation:
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    
    collector = plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=t_start, t_split=split_time_t, t_end=t_end, active_context=active_context, perform_write_to_file_callback=None)

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText
    
    laps_all_epoch_bins_marginals_df = deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df)
    ripple_all_epoch_bins_marginals_df = deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)

    if active_context is not None:
        display_context = active_context.adding_context('display_fn', display_fn_name='plot_all_epoch_bins_marginal_predictions')
        
    # These subset contexts are used to filter out lap/ripple only keys.
    # e.g. active_context=curr_active_pipeline.build_display_context_for_session('directional_merged_pf_decoded_epochs', laps_t_bin=laps_decoding_time_bin_size, ripple_t_bin=ripple_decoding_time_bin_size)
        # only want laps_t_bin on the laps plot and ripple_t_bin on the ripples plot
    laps_only_keys = [item for item in display_context.keys() if 'lap' in item] # items exclusive to laps: ['laps_t_bin']
    ripple_only_keys = [item for item in display_context.keys() if 'ripple' in item]
    laps_display_context = display_context.get_subset(subset_excludelist=ripple_only_keys) # laps specific context filtering out the ripple keys
    ripple_display_context = display_context.get_subset(subset_excludelist=laps_only_keys) # ripple specific context filtering out the laps keys


    with mpl.rc_context({'figure.figsize': (12.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42,
                          "axes.spines.left": False, "axes.spines.right": False, "axes.spines.bottom": False, "axes.spines.top": False,
                          "axes.edgecolor": "none", "xtick.bottom": False, "xtick.top": False, "ytick.left": False, "ytick.right": False}):
        # Create a FigureCollector instance
        with FigureCollector(name='plot_all_epoch_bins_marginal_predictions', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> <weight:bold>Quantile Diff</></>'):
                """ Captures:

                t_split, t_start, t_end)
                """
                a_collector.contexts.append(sub_context)
                
                for ax in (axes if isinstance(axes, Iterable) else [axes]):
                    # Update the xlimits with the new bounds
                    ax.set_ylim(0.0, 1.0)
                    # Add epoch indicators
                    _tmp_output_dict = PlottingHelpers.helper_matplotlib_add_long_short_epoch_indicator_regions(ax=ax, t_split=t_split, t_start=t_start, t_end=t_end)
                    # Update the xlimits with the new bounds
                    ax.set_xlim(t_start, t_end)
                    # Draw a horizontal line at y=0.5
                    ax.axhline(y=0.5, color=(0,0,0,1)) # , linestyle='--'
                    ## This is figure level stuff and only needs to be done once:
                    # `flexitext` version:
                    text_formatter = FormattedFigureText()
                    ax.set_title('')
                    fig.suptitle('')
                    # top=0.84, bottom=0.125, left=0.07, right=0.97,
                    # text_formatter.setup_margins(fig, top_margin=1.0, left_margin=0.0, right_margin=1.0, bottom_margin=0.05)
                    text_formatter.setup_margins(fig, top_margin=0.84, left_margin=0.07, right_margin=0.97, bottom_margin=0.125)
                    # fig.subplots_adjust(top=top_margin, left=left_margin, right=right_margin, bottom=bottom_margin)
                    # title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, title, va="bottom", xycoords="figure fraction")
                    title_text_obj = flexitext(text_formatter.left_margin, 0.98, title, va="top", xycoords="figure fraction") # 0.98, va="top" means the top edge of the title will be aligned to the fig_y=0.98 mark of the figure.
                    # footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                    #                             text_formatter._build_footer_string(active_context=sub_context),
                    #                             va="top", xycoords="figure fraction")

                    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (0.0025), ## (va="bottom", (0.0025)) - this means that the bottom edge of the footer text is aligned with the fig_y=0.0025 in figure space
                                                text_formatter._build_footer_string(active_context=sub_context),
                                                va="bottom", xycoords="figure fraction")
            
                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)
                
            # Plot for BestDir
            fig, ax = collector.subplots(num='Laps_Marginal', clear=True)
            _out_Laps = sns.scatterplot(
                ax=ax,
                data=laps_all_epoch_bins_marginals_df,
                x='lap_start_t',
                y='P_Long',
                # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            )
            setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=laps_display_context.adding_context('subplot', subplot_name='Laps all_epoch_binned Marginals'), 
                                        title=f'<size:22> Laps <weight:bold>all_epoch_binned</> Marginals</>')
            
            fig, ax = collector.subplots(num='Ripple_Marginal', clear=True)
            _out_Ripple = sns.scatterplot(
                ax=ax,
                data=ripple_all_epoch_bins_marginals_df,
                x='ripple_start_t',
                y='P_Long',
                # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            )
            setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=ripple_display_context.adding_context('subplot', subplot_name='Ripple all_epoch_binned Marginals'), 
                            title=f'<size:22> Ripple <weight:bold>all_epoch_binned</> Marginals</>')


            # # Plot for Both Laps/Ripple on the same figure using subplots:
            # fig, axs = collector.subplots(num='all_epoch_binned_Marginals', nrows = 2, ncols = 1, sharex=True, sharey=True, clear=True)
            # _out_Laps = sns.scatterplot(
            #     ax=axs[0],
            #     data=laps_all_epoch_bins_marginals_df,
            #     x='lap_start_t',
            #     y='P_Long',
            #     # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            # )
            
            # # Ripple_Marginal
            # _out_Ripple = sns.scatterplot(
            #     ax=axs[1],
            #     data=ripple_all_epoch_bins_marginals_df,
            #     x='ripple_start_t',
            #     y='P_Long',
            #     # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            # )
            # setup_common_after_creation(collector, fig=fig, axes=axs, sub_context=display_context.adding_context('subplot', subplot_name='Laps and Ripple all_epoch_binned Marginals'), 
            #                             title=f'<size:22> Laps+Ripple <weight:bold>all_epoch_binned</> Marginals</>')




    
    # Access the collected figures outside the context manager
    # result = tuple(collector.created_figures)

    return collector


class DirectionalPlacefieldGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='directional_laps_overview', tags=['directional','laps','overview', 'pyqtgraph'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], requires_global_keys=["global_computation_results.computed_data['DirectionalLaps']"], uses=['EpochsEditor'], used_by=[], creation_date='2023-11-09 12:03', related_items=[], is_global=True)
    def _display_directional_laps_overview(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Renders a window with the position/laps displayed in the middle and the four templates displayed to the left and right of them.

            #TODO 2023-12-07 09:29: - [ ] This function's rasters have not been updated (as `_display_directional_template_debugger` on 2023-12-07) and when filtering the unit sort order and their labels will probably become incorrect.

            """

            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import EpochsEditor # perform_plot_laps_diagnoser
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
            from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
            from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
            from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons, paired_incremental_sort_neurons # _display_directional_template_debugger

            # raise NotImplementedError
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            # assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results = global_computation_results.computed_data.get('RankOrder', None)
            if rank_order_results is not None:
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = rank_order_results.included_qclu_values
            else:        
                ## get from parameters:
                minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
            
            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

            grid_bin_bounds = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.grid_bin_bounds) # ((0.0, 287.7697841726619), (115.10791366906477, 172.66187050359713))
            ## #TODO 2025-02-12 03:31: - [ ] YIKES!! above grid_bin_bounds should be wrong, 
            # correct_grid_bin_bounds = deepcopy(owning_pipeline_reference.active_configs[global_epoch_name].computation_config.pf_params.grid_bin_bounds) ## CORRECT
        

            # uses `global_session`
            epochs_editor = EpochsEditor.init_from_session(global_session, include_velocity=True, include_accel=False, grid_bin_bounds=grid_bin_bounds)
            root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(epochs_editor.plots.win, None, title='Pho Directional Laps Templates')

            decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

            # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`

            # INCRIMENTAL SORTING:
            if use_incremental_sorting:
                ref_decoder_name: str = list(decoders_dict.keys())[0] # name of the reference coder. Should be 'long_LR'
                sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_incremental_sort_neurons(decoders_dict, included_any_context_neuron_ids)
            else:
                # INDIVIDUAL SORTING:
                # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
                sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
                sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids, sortable_values_list_dict=sortable_values_list_dict)

            sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]

            ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            _out_pf1D_heatmaps = {}
            for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
                if use_incremental_sorting:
                    title_str = f'{a_decoder_name}_pf1Ds [sort: {ref_decoder_name}]'
                else:
                    title_str = f'{a_decoder_name}_pf1Ds'

                _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
                # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

                # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
                curr_win, curr_img = _out_pf1D_heatmaps[a_decoder_name] # win, img
                a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

                # Coloring the heatmap data for each row of the 1D heatmap:
                curr_data = deepcopy(sorted_pf_tuning_curves[i])
                if debug_print:
                    print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

                _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

                for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
                    # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
                    text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
                    text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                    curr_win.addItem(text)

                    # modulate heatmap color for this row (`curr_data[i, :]`):
                    heatmap_base_color = pg.mkColor(a_color_vector)
                    out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
                    _temp_curr_out_colors_heatmap_image.append(out_colors_row)

                ## Build the colored heatmap:
                out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
                if debug_print:
                    print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

                # Ensure the data is in the correct range [0, 1]
                out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
                curr_img.updateImage(out_colors_heatmap_image_matrix)
                _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix


            ## Build Dock Widgets:
            # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
            _out_dock_widgets = {}
            dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                            CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
            # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
            dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))

            for i, (a_decoder_name, a_heatmap) in enumerate(_out_pf1D_heatmaps.items()):
                _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_heatmap[0], dockSize=(300,200), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name])


            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': (epochs_editor, _out_dock_widgets), 'plots': _out_pf1D_heatmaps, 'data': _out_data}

            # Saving/Exporting to file ___________________________________________________________________________________________ #
            #TODO 2023-11-16 22:16: - [ ] Figure out how to save

            def save_figure(): # export_file_base_path: Path = Path(f'output').resolve()
                """ captures: epochs_editor, _out_pf1D_heatmaps

                TODO: note output paths are currently hardcoded. Needs to add the animal's context at least. Probably needs to be integrated into pipeline.
                import pyqtgraph as pg
                import pyqtgraph.exporters
                from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
                """
                ## Get main laps plotter:
                # print_keys_if_possible('_out', _out, max_depth=4)
                # plots = _out['plots']

                ## Already have: epochs_editor, _out_pf1D_heatmaps
                epochs_editor = graphics_output_dict['ui'][0]

                shared_output_file_prefix = f'output/2023-11-20'
                # print(list(plots.keys()))
                # pg.GraphicsLayoutWidget
                main_graphics_layout_widget = epochs_editor.plots.win
                export_file_path = Path(f'{shared_output_file_prefix}_test_main_position_laps_line_plot').with_suffix('.svg').resolve()
                export_pyqtgraph_plot(main_graphics_layout_widget, savepath=export_file_path) # works

                _out_pf1D_heatmaps = graphics_output_dict['plots']
                for a_decoder_name, a_decoder_heatmap_tuple in _out_pf1D_heatmaps.items():
                    a_win, a_img = a_decoder_heatmap_tuple
                    # a_win.export_image(f'{a_decoder_name}_heatmap.png')
                    print(f'a_win: {type(a_win)}')

                    # create an exporter instance, as an argument give it the item you wish to export
                    exporter = pg.exporters.ImageExporter(a_win.plotItem)
                    # exporter = pg.exporters.SVGExporter(a_win.plotItem)
                    # set export parameters if needed
                    # exporter.parameters()['width'] = 300   # (note this also affects height parameter)

                    # save to file
                    export_file_path = Path(f'{shared_output_file_prefix}_test_{a_decoder_name}_heatmap').with_suffix('.png').resolve() # '.svg' # .resolve()

                    exporter.export(str(export_file_path)) # '.png'
                    print(f'exporting to {export_file_path}')
                    # .scene()


            #TODO 2023-11-16 22:23: - [ ] The other display functions using matplotlib do things like this:
            # final_context = active_context
            # graphics_output_dict['context'] = final_context
            # graphics_output_dict['plot_data'] |= {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate,
            #     'time_variable_name':time_variable_name, 'fignum':curr_fig_num}

            # def _perform_write_to_file_callback():
            #     ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            #     return owning_pipeline_reference.output_figure(final_context, graphics_output_dict.figures[0])

            # if save_figure:
            #     active_out_figure_paths = _perform_write_to_file_callback()
            # else:
            #     active_out_figure_paths = []

            # graphics_output_dict['saved_figures'] = active_out_figure_paths


            return graphics_output_dict


    @function_attributes(short_name='directional_template_debugger', tags=['directional','template','debug', 'overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-28 10:13', related_items=[], is_global=True)
    def _display_directional_template_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Renders a window with the four template heatmaps displayed to the left and right of center, and the ability to filter the actively included aclus via `included_any_context_neuron_ids`

            enable_cell_colored_heatmap_rows: bool - uses the cell's characteristic assigned color to shade the 1D heatmap row value for that cell. NOTE: there are some perceptual non-uniformities with luminance how it is being applied now.

            use_incremental_sorting: bool = False - incremental sorting refers to the method of sorting where plot A is sorted first, all of those cells retain their position for all subsequent plots, but the B-unique cells are sorted for B, ... and so on.
                The alternative (use_incremental_sorting = False) is *individual* sorting, where each is sorted independently.

            """

            # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger
            
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
            use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
            # _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

            rank_order_results = global_computation_results.computed_data.get('RankOrder', None)
            if rank_order_results is not None:
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = rank_order_results.included_qclu_values
            else:        
                ## get from parameters:
                minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
            
            assert minimum_inclusion_fr_Hz is not None
            if (use_shared_aclus_only_templates):
                track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # shared-only
            else:
                track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only

            template_debugger: TemplateDebugger = TemplateDebugger.init_templates_debugger(track_templates=track_templates, included_any_context_neuron_ids=included_any_context_neuron_ids,
                                                      use_incremental_sorting=use_incremental_sorting, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, use_shared_aclus_only_templates=use_shared_aclus_only_templates,
                                                      figure_name=figure_name, debug_print=debug_print, defer_render=defer_render, **kwargs)

            # decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }


            # # build the window with the dock widget in it:
            # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
            # _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, text_items_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)
            # root_dockAreaWindow.resize(900, 700)

            # _out_data, _out_plots, _out_ui = TemplateDebugger._subfn_buildUI_directional_template_debugger_data(included_any_context_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict)
            # update_callback_fn = (lambda included_neuron_ids: TemplateDebugger._subfn_update_directional_template_debugger_data(included_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict))
            # _out_ui.on_update_callback = update_callback_fn

            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            # graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': _out_ui, 'plots': _out_plots, 'data': _out_data}

            graphics_output_dict = {'win': template_debugger.ui.root_dockAreaWindow, 'app': template_debugger.ui.app,  'ui': template_debugger.ui, 'plots': template_debugger.plots, 'data': template_debugger.plots_data, 'obj': template_debugger}


            

            # def on_update(included_any_context_neuron_ids):
            #     """ call to update when `included_any_context_neuron_ids` changes.

            #      captures: `decoders_dict`, `_out_plots`, 'enable_cell_colored_heatmap_rows'

            #     """
            #     decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
            #     sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)
            #     # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`

            #     ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            #     _out_plots.pf1D_heatmaps = {}
            #     for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            #         _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=f'{a_decoder_name}_pf1Ds [sort: long_RL]', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
            #         # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

            #         # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
            #         curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img

            #         a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

            #         # Coloring the heatmap data for each row of the 1D heatmap:
            #         curr_data = deepcopy(sorted_pf_tuning_curves[i])
            #         if debug_print:
            #             print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

            #         _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

            #         for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
            #             # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
            #             text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
            #             text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
            #             curr_win.addItem(text)

            #             # modulate heatmap color for this row (`curr_data[i, :]`):
            #             heatmap_base_color = pg.mkColor(a_color_vector)
            #             out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
            #             _temp_curr_out_colors_heatmap_image.append(out_colors_row)

            #         ## Build the colored heatmap:
            #         out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            #         if debug_print:
            #             print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

            #         # Ensure the data is in the correct range [0, 1]
            #         out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            #         if enable_cell_colored_heatmap_rows:
            #             curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
            #         _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

            # graphics_output_dict['ui'].on_update_callback = on_update

            #TODO 2023-11-16 22:23: - [ ] The other display functions using matplotlib do things like this:
            # final_context = active_context
            # graphics_output_dict['context'] = final_context
            # graphics_output_dict['plot_data'] |= {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate,
            #     'time_variable_name':time_variable_name, 'fignum':curr_fig_num}

            # def _perform_write_to_file_callback():
            #     ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            #     return owning_pipeline_reference.output_figure(final_context, graphics_output_dict.figures[0])

            # if save_figure:
            #     active_out_figure_paths = _perform_write_to_file_callback()
            # else:
            #     active_out_figure_paths = []

            # graphics_output_dict['saved_figures'] = active_out_figure_paths

            return graphics_output_dict


    @function_attributes(short_name='directional_track_template_pf1Ds', tags=['directional','template','debug', 'overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-22 10:41', related_items=[], is_global=True)
    def _display_directional_track_template_pf1Ds(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Plots each template's pf1Ds side-by-side in four adjacent subplots. 
            Stack of line-curves style, not heatmap-style
            """

            # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger
            from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables
            import matplotlib.pyplot as plt

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text
            from neuropy.utils.matplotlib_helpers import FormattedFigureText, FigureMargins ## flexitext version

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            # from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers

            from matplotlib.gridspec import GridSpec
            from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, perform_update_title_subtitle
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _plot_track_remapping_diagram
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import build_shared_sorted_neuron_color_maps
            from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
            
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
            if active_context is not None:
                    display_context = active_context.adding_context('display_fn', display_fn_name='directional_track_template_pf1Ds')
                
            def _perform_write_to_file_callback(final_context, fig):
                if save_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig)
                else:
                    pass # do nothing, don't save


            with mpl.rc_context({'savefig.transparent': True, 'ps.fonttype': 42, }): # 'figure.dpi': '220', 'figure.figsize': (10, 4), 
                # Create a FigureCollector instance
                with FigureCollector(name='directional_track_template_pf1Ds', base_context=display_context) as collector:
                    fignum = kwargs.pop('fignum', None)
                    if fignum is not None:
                        print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

                    defer_render = kwargs.pop('defer_render', False)
                    debug_print: bool = kwargs.pop('debug_print', False)

                    enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
                    use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

                    figure_name: str = kwargs.pop('figure_name', 'directional_track_template_pf1Ds')
                    # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
                    # _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)

                    # Recover from the saved global result:
                    directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

                    rank_order_results = global_computation_results.computed_data.get('RankOrder', None)
                    if rank_order_results is not None:
                        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                        included_qclu_values: List[int] = rank_order_results.included_qclu_values
                    else:        
                        ## get from parameters:
                        minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                        included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
                
                    assert minimum_inclusion_fr_Hz is not None
                    if (use_shared_aclus_only_templates):
                        track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # shared-only
                    else:
                        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only

                    ## {"even": "RL", "odd": "LR"}
                    long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
                    (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]

                    decoders_dict = track_templates.get_decoders_dict()
                    decoders_dict_keys = list(decoders_dict.keys())
                    decoder_context_dict = dict(zip(decoders_dict_keys, (long_LR_context, long_RL_context, short_LR_context, short_RL_context)))

                    # print(f'decoders_dict_keys: {decoders_dict_keys}')
                    plot_kwargs = {}
                    mosaic = [
                            ["ax_pf_tuning_curve"],
                            ["ax_pf_occupancy"],
                        ]
                    fig = plt.figure() # layout="constrained"
                    # fig = collector.build_or_reuse_figure(fignum=fignum)
                    subfigures_dict = dict(zip(list(decoders_dict.keys()), fig.subfigures(nrows=1, ncols=4)))
                    # subfigures_dict = dict(zip(list(decoders_dict.keys()), collector.create_subfigures(fig=fig, nrows=1, ncols=4))) 
                    # plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
                    # plt.tight_layout()                    
                    display_outputs = {}
                    all_subfigures = []
                    all_axes_flat_list = []
                    # all_axes_dicts = []
                    all_axes_dicts = {}
                    all_display_contexts = []
                    for a_decoder_name, a_subfigure in subfigures_dict.items():
                        axd = a_subfigure.subplot_mosaic(mosaic, sharex=True, height_ratios=[8, 1], gridspec_kw=dict(wspace=0, hspace=0.15))
                        # axd = collector.subplot_mosaic(mosaic, sharex=True, height_ratios=[8, 1], gridspec_kw=dict(wspace=0, hspace=0.15), extant_fig=a_subfigure)
                        a_decoder = decoders_dict[a_decoder_name]
                        active_context = decoder_context_dict[a_decoder_name]
                        active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_ratemaps_1D')
                        # active_display_fn_identifying_ctx = curr_active_pipeline.build_display_context_for_filtered_session(filtered_session_name=a_name, display_fn_name='plot_directional_pf1Ds')
                        # active_display_fn_identifying_ctx
                        ax_pf_1D = a_decoder.pf.plot_ratemaps_1D(ax=axd["ax_pf_tuning_curve"], active_context=active_display_ctx)
                        active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_occupancy_1D')
                        # active_display_ctx_string = active_display_ctx.get_description(separator='|')
                        all_display_contexts.append(active_display_ctx)
                        all_axes_flat_list.append(ax_pf_1D)

                        ## Occupancy:
                        occupancy_ax = a_decoder.pf.plot_occupancy(fig=a_subfigure, ax=axd["ax_pf_occupancy"], active_context=active_display_ctx, use_flexitext_titles=False, skip_update_titles=True, **({} | plot_kwargs))[-1]
                        # display_outputs[a_name] =                  
                        # plot_variable_name = ({'plot_variable': None} | kwargs)
                        plot_variable_name = plot_kwargs.get('plot_variable', enumTuningMap2DPlotVariables.OCCUPANCY).name
                        active_display_ctx = active_display_ctx.adding_context(None, plot_variable=plot_variable_name)
                        all_display_contexts.append(active_display_ctx)
                        all_axes_flat_list.append(occupancy_ax)
                        

                        ## Setup titles:
                        axd["ax_pf_tuning_curve"].set_title(a_decoder_name)
                        axd["ax_pf_occupancy"].set_title('')
                        
                        # Clear the normal text:
                        a_subfigure.suptitle('')
                        
                        ## end both plots:
                        all_subfigures.append(a_subfigure)
                        # all_axes_dicts.append(axd)
                        all_axes_dicts[a_decoder_name] = axd
                        # all_display_contexts.append(active_display_ctx)
                        
                    # a_decoder.pf
                    collector.post_hoc_append(figs=[fig, *all_subfigures], axes=all_axes_dicts, contexts=all_display_contexts)
                    # collector.post_hoc_append(figs=[fig, *all_subfigures], axes=all_axes_flat_list, contexts=all_display_contexts)
                    
                    a_decoder_name: str = 'long_LR'
                    a_decoder = decoders_dict[a_decoder_name]
                    if a_decoder.pf.ndim > 1:
                        is_2D = True
                    else:
                        is_2D = False
                    active_config = deepcopy(a_decoder.pf.config)
                    # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D=is_2D, normal_to_extras_line_sep="\n")}'])
                    # subtitle_string = active_config.str_for_display(is_2D=is_2D, normal_to_extras_line_sep="\n")
                    subtitle_string = active_config.str_for_display(is_2D=is_2D) # , normal_to_extras_line_sep=","
                    # print(f'subtitle_string: {subtitle_string}')

                    ## BUild figure titles:
                    # INPUTS: main_fig
                    fig.suptitle('')
                    text_formatter = FormattedFigureText() # .init_from_margins(left_margin=0.01)
                    # text_formatter.setup_margins(fig, left_margin=0.01) # , left_margin=0.1
                    # text_formatter = FormattedFigureText.init_from_margins(left_margin=0.01) # , top_margin=0.9
                    # text_formatter.setup_margins(fig, left_margin=0.01, top_margin=0.9)
                    text_formatter.setup_margins(fig)
                    for i, a_subfigure in enumerate(all_subfigures):
                        text_formatter.setup_margins(a_subfigure)
                        
                    # active_config = deepcopy(self.config)
                    # active_config.float_precision = 1
                    title_string: str = f"directional_track_template_pf1Ds"
                    # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
                    # header_text_obj = flexitext(text_formatter.left_margin, 0.9, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    header_text_obj = flexitext(0.01, 0.85, f'<size:20><weight:bold>{title_string}</></>\n<size:9>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
                    fig.canvas.manager.set_window_title(title_string) # sets the window's title
                    if ((_perform_write_to_file_callback is not None) and (display_context is not None)):
                        _perform_write_to_file_callback(display_context, fig)
                        
            return collector



    @function_attributes(short_name='track_remapping_diagram', tags=['remapping'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_bidirectional_track_remapping_diagram'], used_by=[], creation_date='2024-04-29 09:24', related_items=[], is_global=True)
    def _display_directional_track_remapping_diagram(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, is_dark_mode:bool=False, **kwargs):
            """ For both directions, plots a subplot showing the cell's location on the long track and the short track with connecting arrows showing their transition. Draws both tracks in the background as reference. 
            """
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import plot_bidirectional_track_remapping_diagram

            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render: bool = kwargs.pop('defer_render', False)
            should_show: bool = (not defer_render)    
            debug_print: bool = kwargs.pop('debug_print', False)

            use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_track_template_pf1Ds')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
            # _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

            # assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results = global_computation_results.computed_data.get('RankOrder', None)
            if rank_order_results is not None:
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = rank_order_results.included_qclu_values
            else:        
                ## get from parameters:
                minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
            
            assert minimum_inclusion_fr_Hz is not None
            if (use_shared_aclus_only_templates):
                track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # shared-only
            else:
                track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only

            decoders_dict = track_templates.get_decoders_dict()
            decoders_dict_keys = list(decoders_dict.keys())
            a_decoder = list(track_templates.get_decoders_dict().values())[-1]
            grid_bin_bounds = deepcopy(a_decoder.pf.config.grid_bin_bounds)

            active_display_ctx = active_context.adding_context('display_fn', display_fn_name='track_remapping_diagram')

            def _perform_write_to_file_callback(final_context, fig):
                if save_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig)
                else:
                    pass # do nothing, don't save
                

            collector = plot_bidirectional_track_remapping_diagram(track_templates, grid_bin_bounds=grid_bin_bounds, active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback, enable_interactivity=True, draw_point_aclu_labels=True, is_dark_mode=is_dark_mode)

            return collector


    @function_attributes(short_name='directional_merged_pfs', tags=['display', 'pyqtgraph', 'heatmap'], input_requires=["global_computation_results.computed_data['DirectionalMergedDecoders']"], output_provides=[], uses=['display_all_pf_2D_pyqtgraph_binned_image_rendering'], used_by=[], creation_date='2024-01-04 03:27', related_items=[], is_global=True)
    def _display_directional_merged_pfs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                        plot_all_directions=True, plot_long_directional=False, plot_short_directional=False, **kwargs):
        """ Plots the merged pseduo-2D pfs/ratemaps. Plots: All-Directions, Long-Directional, Short-Directional in seperate windows. 
        
        History: this is the Post 2022-10-22 display_all_pf_2D_pyqtgraph_binned_image_rendering-based method:

        TODO 2025-01-21 17:49: - [ ] The frame is too thick when there's only 4 vertical bins present, the lines are too thick too.
        
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, display_all_pf_2D_pyqtgraph_binned_image_rendering
        from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow 
        

        defer_render = kwargs.pop('defer_render', False)
        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data['DirectionalMergedDecoders']
        active_merged_pf_plots_data_dict = {} #empty dict
        
        if plot_all_directions:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='All-Directions', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.all_directional_pf1D_Decoder.pf # all-directions
        if plot_long_directional:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='Long-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.long_directional_pf1D_Decoder.pf # Long-only
        if plot_short_directional:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='Short-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.short_directional_pf1D_Decoder.pf # Short-only

        out_plots_dict = {}
        
        for active_context, active_pf_2D in active_merged_pf_plots_data_dict.items():
            # figure_format_config = {} # empty dict for config
            figure_format_config = {'scrollability_mode': LayoutScrollability.NON_SCROLLABLE} # kwargs # kwargs as default figure_format_config
            out_all_pf_2D_pyqtgraph_binned_image_fig: BasicBinnedImageRenderingWindow  = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config) # output is BasicBinnedImageRenderingWindow
        
            # Set the window title from the context
            out_all_pf_2D_pyqtgraph_binned_image_fig.setWindowTitle(f'{active_context.get_description()}')
            out_plots_dict[active_context] = out_all_pf_2D_pyqtgraph_binned_image_fig

            # Tries to update the display of the item:
            names_list = [v for v in list(out_all_pf_2D_pyqtgraph_binned_image_fig.plots.keys()) if v not in ('name', 'context')]
            for a_name in names_list:
                # Adjust the size of the text for the item by passing formatted text
                a_plot: pg.PlotItem = out_all_pf_2D_pyqtgraph_binned_image_fig.plots[a_name].mainPlotItem # PlotItem
                a_plot.invertY(True)  # Inverts the y-axis
                # no clue why 2 is a good value for this...
                a_plot.titleLabel.setMaximumHeight(2)
                a_plot.layout.setRowFixedHeight(0, 2)

            if not defer_render:
                out_all_pf_2D_pyqtgraph_binned_image_fig.show()

        return out_plots_dict


    @function_attributes(short_name='directional_merged_decoder_decoded_epochs', tags=['yellow-blue-plots', 'directional_merged_decoder_decoded_epochs', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices'], used_by=[], creation_date='2024-01-04 02:59', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                                    single_plot_fixed_height=50.0, max_num_lap_epochs: int = 25, max_num_ripple_epochs: int = 45, size=(15,7), dpi=72, constrained_layout=True, scrollable_figure=True,
                                                    skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, **kwargs):
            """ Renders the marignal posteriors as a yellow-blue plot for trackID

            Renders two windows, one with the decoded laps and another with the decoded ripple posteriors, computed using the merged pseudo-2D decoder.

            """
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
            from neuropy.utils.matplotlib_helpers import FormattedFigureText
        

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                display_context = active_context.adding_context('display_fn', display_fn_name='directional_merged_pf_decoded_epochs')
            else:
                active_context = owning_pipeline_reference.sess.get_context()
                # Build the active context directly:
                display_context = owning_pipeline_reference.build_display_context_for_session('directional_merged_pf_decoded_epochs')

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)
            active_config_name: bool = kwargs.pop('active_config_name', None)

            perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))
            # Extract kwargs for figure rendering
            render_merged_pseudo2D_decoder_laps = kwargs.pop('render_merged_pseudo2D_decoder_laps', False)
            
            render_directional_marginal_laps = kwargs.pop('render_directional_marginal_laps', True)
            render_directional_marginal_ripples = kwargs.pop('render_directional_marginal_ripples', False)
            render_track_identity_marginal_laps = kwargs.pop('render_track_identity_marginal_laps', False)
            render_track_identity_marginal_ripples = kwargs.pop('render_track_identity_marginal_ripples', False)

            directional_merged_decoders_result = kwargs.pop('directional_merged_decoders_result', None)
            if directional_merged_decoders_result is not None:
                print("WARN: User provided a custom directional_merged_decoders_result as a kwarg. This will be used instead of the computed result global_computation_results.computed_data['DirectionalMergedDecoders'].")
                
            else:
                directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']


            # get the time bin size from the decoder:
            laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size
            ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size


            # figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})

            # Recover from the saved global result:
            # directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            # directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

            # requires `laps_is_most_likely_direction_LR_dir` from `laps_marginals`
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()

            graphics_output_dict = {}

            # Shared active_decoder, global_session:
            active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
            global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 

            # 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            # 'figure.constrained_layout.use': constrained_layout, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            with mpl.rc_context({'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, }): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name='plot_directional_merged_pf_decoded_epochs', base_context=display_context) as collector:

                    ## Define the overriden plot function that internally calls the normal plot function but also permits doing operations before and after, such as building titles or extracting figures to save them:
                    def _mod_plot_decoded_epoch_slices(*args, **subfn_kwargs):
                        """ implicitly captures: owning_pipeline_reference, collector, perform_write_to_file_callback, save_figure, skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True

                        NOTE: each call requires adding the additional kwarg: `_main_context=_main_context`
                        """
                        assert '_mod_plot_kwargs' in subfn_kwargs
                        _mod_plot_kwargs = subfn_kwargs.pop('_mod_plot_kwargs')
                        assert 'final_context' in _mod_plot_kwargs
                        _main_context = _mod_plot_kwargs['final_context']
                        assert _main_context is not None
                        # Build the rest of the properties:
                        sub_context: IdentifyingContext = owning_pipeline_reference.build_display_context_for_session('directional_merged_pf_decoded_epochs', **_main_context)
                        sub_context_str: str = sub_context.get_description(subset_includelist=['t_bin'], include_property_names=True) # 't-bin_0.5' # str(sub_context.get_description())
                        modified_name: str = subfn_kwargs.pop('name', '')
                        if len(sub_context_str) > 0:
                            modified_name = f"{modified_name}_{sub_context_str}"
                        subfn_kwargs['name'] = modified_name # update the name by appending 't-bin_0.5'
                        
                        # Call the main plot function:
                        out_plot_tuple = plot_decoded_epoch_slices(*args, skip_plotting_measured_positions=skip_plotting_measured_positions, skip_plotting_most_likely_positions=skip_plotting_most_likely_positions, **subfn_kwargs)
                        # Post-plot call:
                        assert len(out_plot_tuple) == 4
                        params, plots_data, plots, ui = out_plot_tuple # [2] corresponds to 'plots' in params, plots_data, plots, ui = laps_plots_tuple
                        # post_hoc_append to collector
                        mw = ui.mw # MatplotlibTimeSynchronizedWidget
                        
                        y_bin_labels = _mod_plot_kwargs.get('y_bin_labels', None)
                        if y_bin_labels is not None:
                            label_artists_dict = {}
                            for i, ax in enumerate(plots.axs):
                                label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False)
                            plots['label_artists_dict'] = label_artists_dict
                            

                        if mw is not None:
                            fig = mw.getFigure()
                            collector.post_hoc_append(figs=mw.fig, axes=mw.axes, contexts=sub_context)
                            title = mw.params.name
                        else:
                            fig = plots.fig
                            collector.post_hoc_append(figs=fig, axes=plots.axs, contexts=sub_context)
                            title = params.name

                        # Recover the proper title:
                        assert title is not None, f"title: {title}"
                        print(f'title: {title}')
                        
                        if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                            if save_figure:
                                perform_write_to_file_callback(sub_context, fig)
                            
                        # Close if defer_render
                        if defer_render:
                            if mw is not None:
                                mw.close()

                        return out_plot_tuple
                    
                    if render_merged_pseudo2D_decoder_laps:
                        # Merged Pseduo2D Decoder Posteriors:
                        _main_context = {'decoded_epochs': 'Laps', 'Pseudo2D': 'Posterior', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        graphics_output_dict['raw_posterior_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Posterior',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL']),
                            **deepcopy(kwargs)
                        )            
                        # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
                        # label_artists_dict = {}
                        # for i, ax in enumerate(matplotlib_fig_axes):
                        #     label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL'], enable_draw_decoder_colored_lines=False)
                        # output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
                        
                        

                    if render_directional_marginal_laps:
                        # Laps Direction (LR/RL) Marginal:
                        _main_context = {'decoded_epochs': 'Laps', 'Marginal': 'Direction', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        graphics_output_dict['directional_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Marginal_LAPS',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['LR', 'RL']),
                            **deepcopy(kwargs)
                        )

                    if render_directional_marginal_ripples:
                        # Ripple Direction (LR/RL) Marginal:
                        _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'Direction', 't_bin': ripple_decoding_time_bin_size}
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
                        graphics_output_dict['directional_ripples_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_replays, directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Marginal_Ripples',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_ripple_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['LR', 'RL']),
                            **deepcopy(kwargs)
                        )

                    if render_track_identity_marginal_laps:
                        # Laps Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Laps', 'Marginal': 'TrackID', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_LAPS',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long', 'short']),
                            **deepcopy(kwargs)
                        )


                    if render_track_identity_marginal_ripples:
                        # Ripple Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'TrackID', 't_bin': ripple_decoding_time_bin_size}
                        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_ripples_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_replays, directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_Ripples',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_ripple_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long', 'short']),
                            **deepcopy(kwargs)
                        )

            graphics_output_dict['collector'] = collector

            return graphics_output_dict



    @function_attributes(short_name='directional_decoded_epochs_marginals', tags=['scatter-plot', 'directional_merged_decoder_decoded_epochs','directional','marginal'], input_requires=[], output_provides=[], uses=['plot_rank_order_epoch_inst_fr_result_tuples'], used_by=[], creation_date='2023-12-15 21:46', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_epochs_marginals(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
        """ Plots two scatter plot figures of the decoded marginals from the merged decoder

        """
        defer_render: bool = kwargs.pop('defer_render', False)
        should_show: bool = (not defer_render)
        
        # #TODO 2024-01-03 05:24: - [ ] Do something to switch the matplotlib backend to 'AGG' if defer_render == True. Currently only adjusts the pyqtgraph-based figures (`plot_rank_order_epoch_inst_fr_result_tuples`)

        active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        directional_merged_decoders_result = kwargs.pop('directional_merged_decoders_result', None)
        if directional_merged_decoders_result is not None:
            print("WARN: User provided a custom `directional_merged_decoders_result` as a kwarg. This will be used instead of the computed result global_computation_results.computed_data['DirectionalMergedDecoders'].")
            
        else:
            directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

        def _perform_write_to_file_callback(final_context, fig):
            if save_figure:
                return owning_pipeline_reference.output_figure(final_context, fig)
            else:
                pass # do nothing, don't save


        # Quantile Diff Figures: _____________________________________________________________________________________________ #
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        all_epoch_bins_marginals_collector = plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=t_start, t_split=t_delta, t_end=t_end, active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback)

        return all_epoch_bins_marginals_collector



    @function_attributes(short_name='directional_decoded_stacked_epoch_slices', tags=['export', 'heatmaps', 'posterior', 'directional_merged_decoder_decoded_epochs','directional','marginal','NON-VISUAL','NO-DISPLAY'],
                          input_requires=['RankOrder', 'DirectionalLaps', 'DirectionalMergedDecoders', 'DirectionalDecodersEpochsEvaluations'], output_provides=[], uses=['PosteriorExporting.perform_export_all_decoded_posteriors_as_images'], used_by=[], creation_date='2024-09-27 17:08', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_stacked_epoch_slices(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                                                    custom_export_formats: Dict[str, "HeatmapExportConfig"]=None, parent_output_folder: Optional[Path] = None, desired_height:int=1200, **kwargs):
        """ Exports all decoded epoch posteriors separately to a folder. NON-VISUAL, never displays.
         Effectively the entire stack of decoded epochs for both the long and short, including their Radon transformed lines if that information is available.


        # '_display_directional_merged_pf_decoded_stacked_epoch_slices' or  directional_decoded_stacked_epoch_slices
        curr_active_pipeline.reload_default_display_functions()
        _out_paths = curr_active_pipeline.display('_display_directional_merged_pf_decoded_stacked_epoch_slices')
        _out_paths

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
        # from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots        
        from neuropy.utils.result_context import IdentifyingContext
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
        from datetime import datetime, date, timedelta
        from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
        from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig
        from pyphocorehelpers.plotting.media_output_helpers import ImagePostRenderFunctionSets, ImageOperationsAndEffects
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import try_discover_default_collected_outputs_dir

        DAY_DATE_STR: str = date.today().strftime("%Y-%m-%d")
        DAY_DATE_TO_USE = f'{DAY_DATE_STR}' # used for filenames throught the notebook
        # print(f'DAY_DATE_STR: {DAY_DATE_STR}, DAY_DATE_TO_USE: {DAY_DATE_TO_USE}')

        # defer_render: bool = kwargs.pop('defer_render', False)
        # should_show: bool = (not defer_render)
        
        # #TODO 2024-01-03 05:24: - [ ] Do something to switch the matplotlib backend to 'AGG' if defer_render == True. Currently only adjusts the pyqtgraph-based figures (`plot_rank_order_epoch_inst_fr_result_tuples`)


        # active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

        rank_order_results = global_computation_results.computed_data.get('RankOrder', None)
        if rank_order_results is not None:
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = rank_order_results.included_qclu_values
        else:        
            ## get from parameters:
            minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

        directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
        track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        
        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
        directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(owning_pipeline_reference, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=False)

        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
        decoder_laps_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
        decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict

        print(f'pos_bin_size: {pos_bin_size}')
        print(f'ripple_decoding_time_bin_size: {ripple_decoding_time_bin_size}')
        print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}')
        


        # Get appropriate context ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        #TODO 2025-05-30 07:52: - [ ] This assumes `ripple_decoding_time_bin_size == laps_decoding_time_bin_size`
        complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()

        assert (ripple_decoding_time_bin_size == laps_decoding_time_bin_size), f"ripple_decoding_time_bin_size: {ripple_decoding_time_bin_size} == laps_decoding_time_bin_size: {laps_decoding_time_bin_size}"
        active_context = kwargs.pop('active_context', None)
        if active_context is not None:
            # Update the existing context:
            active_context = deepcopy(active_context).overwriting_context(time_bin_size=ripple_decoding_time_bin_size)
        else:
            # active_context = owning_pipeline_reference.sess.get_context()
            active_context = deepcopy(complete_session_context).overwriting_context(time_bin_size=ripple_decoding_time_bin_size) # owning_pipeline_reference.sess.get_context()




        ## Make sure it has all the epoch info
        
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        session_name: str = owning_pipeline_reference.session_name
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()

        decoder_laps_filter_epochs_decoder_result_dict = {k:DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end) for k, a_result in decoder_laps_filter_epochs_decoder_result_dict.items()}
        decoder_ripple_filter_epochs_decoder_result_dict = {k:DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end) for k, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()}

        # parent_output_folder: Path = kwargs.pop('parent_output_folder', None)
        
        needs_discover_default_collected_outputs_dir: bool = True
        if parent_output_folder is not None:
            if isinstance(parent_output_folder, str):
                parent_output_folder = Path(parent_output_folder).resolve()
                if parent_output_folder.exists():
                    needs_discover_default_collected_outputs_dir = False # we're good, the provided dir exists
                    
        if needs_discover_default_collected_outputs_dir:
                ## if none is provided it tries to find one in collected_outputs
                collected_outputs_directory = try_discover_default_collected_outputs_dir(known_collected_outputs_paths=[Path(v).resolve() for v in ['/Users/pho/data/collected_outputs',
                                                                            '/Volumes/SwapSSD/Data/collected_outputs', r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                            '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']])
                assert collected_outputs_directory is not None, f"no collected output dir found!"
                assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
                # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
                print(f'collected_outputs_directory: "{collected_outputs_directory}"')
                # Create a 'figures' subfolder if it doesn't exist
                figures_folder: Path = collected_outputs_directory.joinpath('figures', '_temp_individual_posteriors').resolve()
                figures_folder.mkdir(parents=False, exist_ok=True)
                assert figures_folder.exists()
                print(f'\tfigures_folder: "{figures_folder}"')
                ## this is good
                parent_output_folder = figures_folder


        Assert.path_exists(parent_output_folder)
        posterior_out_folder = parent_output_folder.joinpath(DAY_DATE_TO_USE).resolve()
        posterior_out_folder.mkdir(parents=True, exist_ok=True)
        save_path = posterior_out_folder.resolve()
        _parent_save_context: IdentifyingContext = owning_pipeline_reference.build_display_context_for_session('perform_export_all_decoded_posteriors_as_images')
        # _specific_session_output_folder = save_path.joinpath(active_context.get_description(subset_excludelist=['format_name'])).resolve()
        _specific_session_output_folder = save_path.joinpath(active_context.get_description(subset_excludelist=['format_name', 'display_fn_name', 'time_bin_size'])).resolve()


        _specific_session_output_folder.mkdir(parents=True, exist_ok=True)
        print(f'\tspecific_session_output_folder: "{_specific_session_output_folder}"')

        if custom_export_formats is None:
            ## use this default instead:
            common_export_config_kwargs = dict(desired_height = desired_height,
                post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict,
                # post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_mergedColorDecoders_image_export_functions_dict,
            )

            custom_export_formats: Dict[str, HeatmapExportConfig] = {
                'greyscale': HeatmapExportConfig.init_greyscale(**common_export_config_kwargs),
                'color': HeatmapExportConfig(colormap='Oranges', **common_export_config_kwargs), # #TODO 2025-05-30 03:45: - [ ] previous config
                # 'color': HeatmapExportConfig(colormap=additional_cmaps['long_LR']),
                # 'color': HeatmapExportConfig(colormap=cmap1, desired_height=200),
            }

        out_paths, out_custom_formats_dict = PosteriorExporting.perform_export_all_decoded_posteriors_as_images(decoder_laps_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                 _save_context=_parent_save_context, parent_output_folder=_specific_session_output_folder,
                                                                                                                  desired_height=desired_height, custom_export_formats=custom_export_formats)
        # out_paths
        print(_specific_session_output_folder)

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        # pagination_controllers, figs, axs, ctxts, out_figure_paths = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(owning_pipeline_reference, included_epoch_indicies=included_epoch_indicies, defer_render=defer_render, save_figure=save_figure, **kwargs)
        graphics_output_dict = {'export_paths': out_paths,
                                'out_custom_formats_dict': out_custom_formats_dict,
                                'parent_output_folder': parent_output_folder,
                                'parent_specific_session_output_folder': _specific_session_output_folder,
            
        }
        return graphics_output_dict
    


# ==================================================================================================================== #
# Menu Plotting Commands                                                                                               #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuCommand

@metadata_attributes(short_name=None, tags=['track', 'Spike3DRasterWindowWidget', 'MatplotlibPlotCommand', 'epoch', 'decode'],
                      input_requires=['DirectionalDecodersDecoded', 'DirectionalDecodersDecoded.pf1D_Decoder_dict["pseudo2D"]'], output_provides=[], uses=['plot_1D_most_likely_position_comparsions'], used_by=[], creation_date='2024-01-17 00:00', related_items=[])
@define(slots=False)
class AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand(BaseMenuCommand):
    """ 2024-01-17 
    ## #TODO 2025-06-02 10:27: - [ ] BROKEN, Not aligned with positions also plotted

    Adds four rows to the SpikeRaster2D showing the decoded posteriors for the PBE epochs for each of the four 1D decoders
    These are the ones Kamran wants to be normalized differently.
    
    Adds rows: `f'{a_decoder_name}_ContinuousDecode'`
    
    Usage:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand

    """
    _spike_raster_window = field()
    _active_pipeline = field(alias='curr_active_pipeline')
    _active_config_name = field(default=None)
    _context = field(default=None, alias="active_context")
    _display_output = field(default=Factory(dict))


    @classmethod
    def add_directional_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, debug_print=False):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.SpecificDockWidgetManipulatingMixin import SpecificDockWidgetManipulatingMixin #TODO 2025-06-02 10:28: - [ ] Converting to use `SpecificDockWidgetManipulatingMixin`
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult


        output_dict = {}

        ## get the result data:
        try:
            time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
            if debug_print:
                print(f'time_bin_size: {time_bin_size}')

            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            # pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            a_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(time_bin_size, None)
            all_time_bin_sizes_output_dict: Dict[float, Dict[types.DecoderName, SingleEpochDecodedResult]] = directional_decoders_decode_result.split_pseudo2D_continuous_result_to_1D_continuous_result()
            a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict: Dict[types.DecoderName, SingleEpochDecodedResult] = all_time_bin_sizes_output_dict.get(time_bin_size, None)

            assert a_continuously_decoded_dict is not None, f"a_continuously_decoded_dict is None even after recomputing!"
            assert a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict is not None, f"a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict is None even after recomputing!"
            info_string: str = f" - t_bin_size: {time_bin_size:.3f}"

        except (KeyError, AttributeError) as e:
            # KeyError: 'DirectionalDecodersDecoded'
            print(f'add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(...) failed to add any tracks, perhaps because the pipeline is missing any computed "DirectionalDecodersDecoded" global results. Error: "{e}". Skipping.')
            a_continuously_decoded_dict = None
            pseudo2D_decoder = None        
            pass

        except Exception as e:
            raise


        # ==================================================================================================================== #
        # PLOTTING                                                                                                             #
        # ==================================================================================================================== #
        dock_group_sep_character: str = '_'
        showCloseButton = True
        _common_dock_config_kwargs = {'dock_group_names': [dock_group_sep_character.join([f'ContinuousDecode', info_string])], 'showCloseButton': showCloseButton}
        dock_configs: Dict[str, CustomDockDisplayConfig] = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'),
                                (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs),
                                CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs))))

        output_dict = active_2d_plot.add_docked_decoded_results_dict_tracks(name=f'ContinuousDecode', a_decoded_result_dict=a_split_pseudo2D_continuous_result_to_1D_continuous_result_dict, dock_configs=dock_configs, pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict,
                                                                                                    measured_position_df=deepcopy(curr_active_pipeline.sess.position.to_dataframe()),
                                                                                                    extended_dock_title_info=info_string)
        ## layout the dockGroups:
        nested_dock_items, nested_dynamic_docked_widget_container_widgets = active_2d_plot.ui.dynamic_docked_widget_container.layout_dockGroups()
        

        return output_dict
    
    def validate_can_display(self) -> bool:
        """ returns True if the item is enabled, otherwise returns false """
        #TODO 2025-06-02 10:42: - [ ] Fixed, but now redundant, so disable perminantly by default
        return False
        # try:
        #     curr_active_pipeline = self._active_pipeline
        #     # assert curr_active_pipeline is not None
        #     if curr_active_pipeline is None:
        #         raise ValueError("Current active pipeline is None!")
        #     active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        #     # assert active_2d_plot is not None
        #     if active_2d_plot is None:
        #         raise ValueError("active_2d_plot is None!")

        #     return DirectionalDecodersContinuouslyDecodedResult.validate_has_directional_decoded_continuous_epochs(curr_active_pipeline=curr_active_pipeline)
            
        # except Exception as e:
        #     print(f'Exception {e} occured in validate_can_display(), returning False')
        #     return False

    def execute(self, *args, **kwargs) -> None:
        print(f'menu execute(): {self}')
        ## To begin, the destination plot must have a matplotlib widget plot to render to:
        # print(f'AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute(...)')
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        # If no plot to render on, do this:
        output_dict = self.add_directional_decoder_decoded_epochs(self._active_pipeline, active_2d_plot) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    
        self.log_command(*args, **kwargs) # adds this command to the `menu_action_history_list` 

        # Update display output dict:
        for a_decoder_name, an_output_tuple in output_dict.items():
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple
            self._display_output[identifier_name] = an_output_tuple

        print(f'\t AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute() is done.')
        


@metadata_attributes(short_name=None, tags=['track', 'KAMRAN_LIKES_NORMALIZATION', 'Spike3DRasterWindowWidget', 'MatplotlibPlotCommand', 'epochs', 'decoder'],
                      input_requires=['DirectionalDecodersDecoded', 'DirectionalDecodersDecoded.most_recent_continuously_decoded_dict["pseudo2D"]'], output_provides=[], uses=['plot_1D_most_likely_position_comparsions'], used_by=[], creation_date='2024-01-22 00:00', related_items=[])
@define(slots=False)
class AddNewDecodedPosteriors_MatplotlibPlotCommand(BaseMenuCommand):
    """ 2024-01-22 
    ??? Adds four rows to the SpikeRaster2D showing the marginal probabilities 
    continuously decoded posterior for each of the four 1D decoders
    -- this heatmap normalization is one that Kamran likes -- the four contexts "compete" with one another, to add up to 1.0 in each time bin (it seems).
    
    
    Usage:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand

    """
    _spike_raster_window = field()
    _active_pipeline = field(alias='curr_active_pipeline')
    _active_config_name = field(default=None)
    _context = field(default=None, alias="active_context")
    _display_output = field(default=Factory(dict))
    
    @function_attributes(short_name=None, tags=['row', 'posterior'], input_requires=[], output_provides=[], uses=['add_new_matplotlib_render_plot_widget', 'plot_1D_most_likely_position_comparsions'], used_by=[], creation_date='2024-12-18 08:53', related_items=[])
    @classmethod
    def _build_dock_group_id(cls, extended_dock_title_info: Optional[str]=None) -> str:
        # identifier_name: str = f'{a_decoder_name}_ContinuousDecode'
        dock_group_identifier_name: str = f'ContinuousDecode_' # {a_decoder_name}
        if extended_dock_title_info is not None:
            dock_group_identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        # print(f'identifier_name: {identifier_name}')
        return dock_group_identifier_name


    @classmethod
    def _perform_overlay_measured_position(cls, identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, measured_position_df: pd.DataFrame, variable_name = 'x'):
        """ overlays a red measured position line over the heatmap
        
        if a_position_decoder.pf.filtered_pos_df is not None:
            measured_position_df = deepcopy(a_position_decoder.pf.filtered_pos_df)
        else:
            # fallback to session
            measured_position_df = curr_active_pipeline.sess.position.to_dataframe()
            
        """
        # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes
        actual_postion_plot_kwargs = {'color': '#ff000066', 'alpha': 0.35, 'marker': 'none', 'animated': False}
        _out_artists = {}
        for an_ax in matplotlib_fig_axes:                
            line_measured_position = an_ax.plot(measured_position_df['t'].to_numpy(), measured_position_df[variable_name].to_numpy(), label=f'measured {variable_name}', **actual_postion_plot_kwargs) # Opaque RED # , linestyle='dashed', linewidth=2, color='#ff0000ff'
            _out_artists[an_ax] = line_measured_position
        return _out_artists

            

    @function_attributes(short_name=None, tags=['row', 'posterior'], input_requires=[], output_provides=[], uses=['add_new_matplotlib_render_plot_widget', 'plot_1D_most_likely_position_comparsions'], used_by=['prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs'], creation_date='2024-12-18 08:53', related_items=[])
    @classmethod
    def _perform_add_new_decoded_posterior_row(cls, curr_active_pipeline, active_2d_plot, a_dock_config, a_decoder_name: str, a_position_decoder: BasePositionDecoder, time_window_centers, a_1D_posterior, extended_dock_title_info: Optional[str]=None):
        """ used with `add_pseudo2D_decoder_decoded_epochs` - adds a single decoded row to the matplotlib dynamic output
        
        # a_decoder_name: str = "long_LR"
        
        USES: a_position_decoder.xbin
        UNUSED: curr_active_pipeline, 

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import FixedCustomColormaps
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        

        ## ✅ Add a new row for each of the four 1D directional decoders:
        # identifier_name: str = f'{a_decoder_name}_ContinuousDecode'
        identifier_name: str = f'ContinuousDecode_{a_decoder_name}'
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        print(f'identifier_name: {identifier_name}')

        widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(65, 200), display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        # _active_config_name = None
        variable_name: str = a_decoder_name
        
        active_bins = deepcopy(a_position_decoder.xbin)
        # time_window_centers = deepcopy(active_decoder.time_window_centers)
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        # active_posterior = active_marginals.p_x_given_n

        active_posterior = deepcopy(a_1D_posterior)
        
        # most_likely_positions_mode: 'standard'|'corrected'
        # fig, curr_ax = curr_active_pipeline.display('_display_plot_marginal_1D_most_likely_position_comparisons', _active_config_name, variable_name='x', most_likely_positions_mode='corrected', ax=an_ax) # ax=active_2d_plot.ui.matplotlib_view_widget.ax
        posterior_heatmap_imshow_kwargs = dict(
            # cmap=get_heatmap_cmap(cmap='Oranges', bad_color='black', under_color='white', over_color='red'),
            cmap = get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red'),
            # cmap = FixedCustomColormaps.get_custom_orange_with_low_values_dropped_cmap()
            # cmap = FixedCustomColormaps.get_custom_black_with_low_values_dropped_cmap(low_value_cutoff=0.05)
            # cmap = ColormapHelpers.create_colormap_transparent_below_value(active_cmap, low_value_cuttoff=0.1)
            # cmap = FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.25)
            # active_cmap = FixedCustomColormaps.get_custom_orange_with_low_values_dropped_cmap()
        )

        measured_position_df = None # Note: for some reason setting `measured_position_df` to anything other than None here messes up the plotting entirely. Set it to None now, and if we want measured positions plot them after
            
        ## Actual plotting portion:
        fig, an_ax, _return_out_artists_dict = plot_1D_most_likely_position_comparsions(measured_position_df=measured_position_df, time_window_centers=time_window_centers, xbin=active_bins,
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs, return_created_artists=True)

        ## Update the params
        widget.params.variable_name = variable_name
        widget.params.posterior_heatmap_imshow_kwargs = deepcopy(posterior_heatmap_imshow_kwargs)
        widget.params.enable_flat_line_drawing = False
        if extended_dock_title_info is not None:
            widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
            



        ## Update the plots_data - used for crosshairs tracing and other things
        if time_window_centers is not None:
            widget.plots_data.time_window_centers = deepcopy(time_window_centers)
        if active_bins is not None:
            widget.plots_data.xbin = deepcopy(active_bins)
        if active_most_likely_positions is not None:
            widget.plots_data.active_most_likely_positions = deepcopy(active_most_likely_positions)
        widget.plots_data.variable_name = variable_name
        if a_1D_posterior is not None:
            widget.plots_data.matrix = deepcopy(a_1D_posterior)
        if a_position_decoder is not None:
            widget.plots_data.a_decoder = deepcopy(a_position_decoder)
        
        ## Update the .plots
        widget.plots.update({k:v for k, v in _return_out_artists_dict.items() if v is not None})

        ## try to add the measured_positions
        widget.plots_data.measured_position_df = None
        widget.plots.measured_position_artists = None       
        if curr_active_pipeline is not None:
            measured_position_df = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
            if measured_position_df is not None:
                widget.plots_data.measured_position_df = measured_position_df
                _out_artists = cls._perform_overlay_measured_position(identifier_name=identifier_name, widget=widget, matplotlib_fig=fig, matplotlib_fig_axes=[an_ax], measured_position_df=measured_position_df)
                widget.plots.measured_position_artists = _out_artists
                
        widget.draw() # alternative to accessing through full path?
        active_2d_plot.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem
    



    # ==================================================================================================================== #
    # Specific Methods                                                                                                     #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['main'], input_requires=[], output_provides=[], uses=['_perform_add_new_decoded_posterior_row', '_build_dock_group_id'], used_by=['add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs', 'add_pseudo2D_decoder_decoded_epochs'], creation_date='2025-01-29 09:25', related_items=[])
    @classmethod
    def prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string: str, a_pseudo2D_decoder: BasePositionDecoder, debug_print: bool=False, **kwargs):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
        
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict

        # continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_decode_result.most_recent_continuously_decoded_dict)
        assert continuously_decoded_dict is not None

        ## INPUTS: most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string
        
        # all_directional_continuously_decoded_dict = most_recent_continuously_decoded_dict or {}
        pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('pseudo2D', None)
        assert len(pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list) == 1
        p_x_given_n = pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list[0]
        # p_x_given_n = pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list[0]['p_x_given_n']
        time_bin_containers = pseudo2D_decoder_continuously_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)

        ## Split across the 2nd axis to make 1D posteriors that can be displayed in separate dock rows:
        assert p_x_given_n.shape[1] == 4, f"expected the 4 pseudo-y bins for the decoder in p_x_given_n.shape[1]. but found p_x_given_n.shape: {p_x_given_n.shape}"
        split_pseudo2D_posteriors_dict = {k:np.squeeze(p_x_given_n[:, i, :]) for i, k in enumerate(('long_LR', 'long_RL', 'short_LR', 'short_RL'))}


        showCloseButton = True
        _common_dock_config_kwargs = {'dock_group_names': [cls._build_dock_group_id(extended_dock_title_info=info_string)],
                                                           'showCloseButton': showCloseButton,
                                    }
        

        
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, **_common_dock_config_kwargs), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, **_common_dock_config_kwargs))))


        # Need all_directional_pf1D_Decoder_dict
        output_dict = {}

        for a_decoder_name, a_1D_posterior in split_pseudo2D_posteriors_dict.items():
            a_dock_config = dock_configs[a_decoder_name]
            _out_tuple = cls._perform_add_new_decoded_posterior_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=a_dock_config, a_decoder_name=a_decoder_name, a_position_decoder=a_pseudo2D_decoder,
                                                                     time_window_centers=time_window_centers, a_1D_posterior=a_1D_posterior, extended_dock_title_info=info_string)
            # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = _out_tuple
            output_dict[a_decoder_name] = _out_tuple
        
        # OUTPUTS: output_dict
        return output_dict
    

    @function_attributes(short_name=None, tags=['multi-time-bin'], input_requires=[], output_provides=[], uses=['cls.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs'], used_by=[], creation_date='2025-02-26 05:37', related_items=[])
    @classmethod
    def add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, debug_print=False, **kwargs):
        """ adds all computed time_bin_sizes in `curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded'].continuously_decoded_result_cache_dict` from the global_computation_results as new matplotlib plot rows. """
        flat_all_time_bin_sizes_output_tuples_dict: Dict[str, Tuple] = {}
        
        try:
            ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder        
            # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict

            # all_time_bin_sizes_output_dict = {'non_marginalized_raw_result': [], 'marginal_over_direction': [], 'marginal_over_track_ID': []}
            # flat_all_time_bin_sizes_output_tuples_list: List[Tuple] = []
            
            for time_bin_size, a_continuously_decoded_dict in continuously_decoded_result_cache_dict.items():
                ## Each iteration here adds 4 more tracks -- one for each decoding context
                
                # a_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult]
                if debug_print:
                    print(f'time_bin_size: {time_bin_size}')

                info_string: str = f" - t_bin_size: {time_bin_size}"
                # output_dict = _cmd.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=_cmd._active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, **enable_rows_config_kwargs)
                output_dict = cls.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, a_pseudo2D_decoder=pseudo2D_decoder, debug_print=debug_print, **kwargs)
                for a_key, an_output_tuple in output_dict.items():
                    identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple
                    # if a_key not in all_time_bin_sizes_output_dict:
                    #     all_time_bin_sizes_output_dict[a_key] = [] ## init empty list
                    # all_time_bin_sizes_output_dict[a_key].append(an_output_tuple)
                    
                    assert (identifier_name not in flat_all_time_bin_sizes_output_tuples_dict), f"identifier_name: {identifier_name} already in flat_all_time_bin_sizes_output_tuples_dict: {list(flat_all_time_bin_sizes_output_tuples_dict.keys())}"
                    flat_all_time_bin_sizes_output_tuples_dict[identifier_name] = an_output_tuple
        except (KeyError, AttributeError) as e:
            # KeyError: 'DirectionalDecodersDecoded'
            print(f'add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(...) failed to add any tracks, perhaps because the pipeline is missing any computed "DirectionalDecodersDecoded" global results. Error: "{e}". Skipping.')
            pass                   

        except Exception as e:
            raise
            
        ## OUTPUTS: all_time_bin_sizes_output_dict
        return flat_all_time_bin_sizes_output_tuples_dict
    


    @function_attributes(short_name=None, tags=['pseudo2D'], input_requires=[], output_provides=[], uses=['cls.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs'], used_by=[], creation_date='2025-01-29 09:24', related_items=[])
    @classmethod
    def add_pseudo2D_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, debug_print=False, **kwargs):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
    
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        if debug_print:
            print(f'time_bin_size: {time_bin_size}')
            
        info_string: str = f" - t_bin_size: {time_bin_size}"
        most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_decode_result.most_recent_continuously_decoded_dict)
        
        return cls.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=most_recent_continuously_decoded_dict, info_string=info_string, pseudo2D_decoder=pseudo2D_decoder, debug_print=debug_print, **kwargs)


    @function_attributes(short_name=None, tags=['tracks', 'custom_decoder'], input_requires=[], output_provides=[], uses=['cls._perform_add_new_decoded_posterior_row'], used_by=[], creation_date='2025-01-29 12:08', related_items=[])
    @classmethod
    def prepare_and_perform_custom_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string: str, xbin: NDArray, debug_print: bool=False, **kwargs):
        """ adds the 3 tracks (long/short/global) to the spike3DRasterWindow as tracks
        for `AddNewDecodedPosteriors_MatplotlibPlotCommand`
        
        adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. 
        
        Usage:
        
            _output_dict = AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_custom_decoder_decoded_epochs(curr_active_pipeline=None, active_2d_plot=active_2d_plot, continuously_decoded_dict=continuous_specific_decoded_results_dict, info_string='non-PBE-decodings', xbin=deepcopy(new_decoder_dict['global'].xbin), debug_print=False)

        """
        from attrs import make_class
        SimpleDecoderDummy = make_class('SimpleDecoderDummy', attrs=['xbin'])
        
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
        
        assert continuously_decoded_dict is not None

        showCloseButton = True
        _common_dock_config_kwargs = {'dock_group_names': [AddNewDecodedPosteriors_MatplotlibPlotCommand._build_dock_group_id(extended_dock_title_info=info_string)],
                                                            'showCloseButton': showCloseButton,
                                    }
        

        override_dock_group_name: str = kwargs.pop('dock_group_name', None)
        if override_dock_group_name is not None:
            ## set the user-provided dock_group_name first:
            _common_dock_config_kwargs['dock_group_names'] = [*_common_dock_config_kwargs['dock_group_names'], override_dock_group_name]
        
        dock_configs = dict(zip(('long', 'short', 'global'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Epochs.get_long_dock_colors, **_common_dock_config_kwargs),
                                                              CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Epochs.get_short_dock_colors, **_common_dock_config_kwargs),
                                                              CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Epochs.get_global_dock_colors, **_common_dock_config_kwargs))))

        # Need all_directional_pf1D_Decoder_dict
        output_dict = {}

        for a_decoder_name, a_decoded_result in continuously_decoded_dict.items():
            a_dock_config = dock_configs[a_decoder_name]
            assert len(a_decoded_result.p_x_given_n_list) == 1
            p_x_given_n = a_decoded_result.p_x_given_n_list[0]
            # p_x_given_n = a_decoded_result.p_x_given_n_list[0]['p_x_given_n']
            time_bin_containers = a_decoded_result.time_bin_containers[0]
            time_window_centers = time_bin_containers.centers
                    

            _out_tuple = cls._perform_add_new_decoded_posterior_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=a_dock_config, a_decoder_name=a_decoder_name, a_position_decoder=SimpleDecoderDummy(xbin=xbin),
                                                                        time_window_centers=time_window_centers, a_1D_posterior=p_x_given_n, extended_dock_title_info=info_string)
            # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = _out_tuple
            output_dict[a_decoder_name] = _out_tuple
        
        # OUTPUTS: output_dict
        return output_dict


    # ==================================================================================================================== #
    # General Methods                                                                                                      #
    # ==================================================================================================================== #

    def validate_can_display(self) -> bool:
        """ returns True if the item is enabled, otherwise returns false """
        try:
            curr_active_pipeline = self._active_pipeline
            # assert curr_active_pipeline is not None
            if curr_active_pipeline is None:
                raise ValueError("Current active pipeline is None!")
            active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
            # assert active_2d_plot is not None
            if active_2d_plot is None:
                raise ValueError("active_2d_plot is None!")

            return DirectionalDecodersContinuouslyDecodedResult.validate_has_directional_decoded_continuous_epochs(curr_active_pipeline=curr_active_pipeline)
            
        except Exception as e:
            print(f'Exception {e} occured in validate_can_display(), returning False')
            return False

    def execute(self, *args, **kwargs) -> None:
        print(f'menu execute(): {self}')
        self.log_command(*args, **kwargs) # adds this command to the `menu_action_history_list` 

        ## To begin, the destination plot must have a matplotlib widget plot to render to:
        # print(f'AddNewDecodedPosteriors_MatplotlibPlotCommand.execute(...)')
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d

        # output_dict = self.add_pseudo2D_decoder_decoded_epochs(self._active_pipeline, active_2d_plot)
        output_dict = self.add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(self._active_pipeline, active_2d_plot, **kwargs)
        
        # Update display output dict:
        for a_decoder_name, an_output_tuple in output_dict.items():
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple
            self._display_output[identifier_name] = an_output_tuple
        
        print(f'\t AddNewDecodedPosteriors_MatplotlibPlotCommand.execute() is done.')


    def remove(self, *args, **kwargs) -> None:
        """ Removes any added docks/items """
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        # self.log_command(*args, **kwargs) # pops this command from the `menu_action_history_list`
        for identifier_name, an_output_tuple in self._display_output.items():
            # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple
            # widget, matplotlib_fig, matplotlib_fig_axes = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(65, 200), display_config=a_dock_config)
            active_2d_plot.remove_matplotlib_render_plot_widget(identifier_name)


        # raise NotImplementedError # implementors must override      
    



@metadata_attributes(short_name=None, tags=['track', 'Spike3DRasterWindowWidget', 'MatplotlibPlotCommand', 'epoch', 'marginal'],
                      input_requires=['DirectionalDecodersDecoded', 'DirectionalDecodersDecoded.most_recent_continuously_decoded_dict["pseudo2D"]'], output_provides=[], uses=[], used_by=[], creation_date='2024-12-24 15:47', related_items=[])
@define(slots=False)
class AddNewDecodedEpochMarginal_MatplotlibPlotCommand(AddNewDecodedPosteriors_MatplotlibPlotCommand):
    """ 2024-01-23
    Adds THREE rows ('non_marginalized_raw_result', 'marginal_over_direction', 'marginal_over_track_ID') to the SpikeRaster2D showing the marginalized posteriors for the continuously decoded decoder
    These are the 4 x n_total_time_bins grid of context likelihoods. They are normalized okay, maybe how Kamran wants them?
    
    Usage:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand

    """
    _spike_raster_window = field()
    _active_pipeline = field(alias='curr_active_pipeline')
    _active_config_name = field(default=None)
    _context = field(default=None, alias="active_context")
    _display_output = field(default=Factory(dict))
    enable_non_marginalized_raw_result: bool = field(default=True)
    enable_marginal_over_direction: bool = field(default=False)
    enable_marginal_over_track_ID: bool = field(default=True)
    allow_overwrite_enabled_marginals_from_params: bool = field(default=True, metadata={'desc': 'if True, allows overwriting self.enable_* properties with the values in `self._spike_raster_window.spike_raster_plt_2d.params`.'})


    @property
    def all_computed_time_bin_sizes(self) -> List[float]:
        """The all_computed_time_bin_sizes property."""
        assert self._active_pipeline is not None
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = self._active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        return list(continuously_decoded_result_cache_dict.keys())


    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=['plot_1D_most_likely_position_comparsions', 'get_heatmap_cmap'], used_by=['cls.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals'], creation_date='2025-06-02 09:55', related_items=[])
    @classmethod
    def _perform_add_new_decoded_posterior_marginal_row(cls, curr_active_pipeline, active_2d_plot, a_dock_config, a_variable_name: str, time_window_centers, xbin, a_1D_posterior, extended_dock_title_info: Optional[str]=None):
        """ used with `add_pseudo2D_decoder_decoded_epoch_marginals` - adds a single decoded row to the matplotlib dynamic output
        
        # a_decoder_name: str = "long_LR"

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        

        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = f'{a_variable_name}_ContinuousDecode'
        if extended_dock_title_info is not None:
            identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes, dock_crap = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(25, 200), display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        variable_name: str = a_variable_name
        
        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        active_posterior = deepcopy(a_1D_posterior)
        
        posterior_heatmap_imshow_kwargs = dict()	
        
        # most_likely_positions_mode: 'standard'|'corrected'
        # fig, curr_ax = curr_active_pipeline.display('_display_plot_marginal_1D_most_likely_position_comparisons', _active_config_name, variable_name='x', most_likely_positions_mode='corrected', ax=an_ax) # ax=active_2d_plot.ui.matplotlib_view_widget.ax
        ## Actual plotting portion:
        fig, curr_ax, _return_out_artists_dict = plot_1D_most_likely_position_comparsions(None, time_window_centers=time_window_centers, xbin=deepcopy(xbin),
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs, return_created_artists=True)

        ## Update the params
        widget.params.variable_name = variable_name
        widget.params.posterior_heatmap_imshow_kwargs = deepcopy(posterior_heatmap_imshow_kwargs)
        widget.params.enable_flat_line_drawing = False
        if extended_dock_title_info is not None:
            widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
            
        ## Update the plots_data
        if time_window_centers is not None:
            widget.plots_data.time_window_centers = deepcopy(time_window_centers)
        if xbin is not None:
            widget.plots_data.xbin = deepcopy(xbin)
        if active_most_likely_positions is not None:
            widget.plots_data.active_most_likely_positions = deepcopy(active_most_likely_positions)
        widget.plots_data.variable_name = variable_name
        if a_1D_posterior is not None:
            widget.plots_data.matrix = deepcopy(a_1D_posterior)


        widget.draw() # alternative to accessing through full path?
        active_2d_plot.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes
    

    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=['cls._perform_add_new_decoded_posterior_marginal_row', 'DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors', 'PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels'], used_by=['add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epoch_marginals'], creation_date='2025-06-02 09:52', related_items=[])
    @classmethod
    def prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(cls, curr_active_pipeline, active_2d_plot, continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string: str, enable_non_marginalized_raw_result: bool=True, 
                                                                         enable_marginal_over_direction: bool=False, enable_marginal_over_track_ID: bool=True, enable_marginal_labels:bool=True, debug_print: bool=False):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        # pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder        
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict

        # continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_decode_result.most_recent_continuously_decoded_dict)
        assert continuously_decoded_dict is not None

        ## INPUTS: most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string
        


        # all_directional_continuously_decoded_dict = most_recent_continuously_decoded_dict or {}
        pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('pseudo2D', None)
        assert len(pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list) == 1
        non_marginalized_raw_result = DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']
        marginal_over_direction = DirectionalPseudo2DDecodersResult.build_custom_marginal_over_direction(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']
        marginal_over_track_ID = DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(pseudo2D_decoder_continuously_decoded_result)[0]['p_x_given_n']
        # non_marginalized_raw_result.shape # (4, 128672)
        # marginal_over_direction.shape # (2, 128672)
        # marginal_over_track_ID.shape # (2, 128672)
        ## WARN: note that it's typically .shape[1] that we check for the pseudo-y bins, but there's that quirk about the contents of ['p_x_given_n'] being transposed. Hope it's all okay.
        

        # p_x_given_n = pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list[0]
        # p_x_given_n = pseudo2D_decoder_continuously_decoded_result.p_x_given_n_list[0]['p_x_given_n']
        time_bin_containers = pseudo2D_decoder_continuously_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)
        
        ## Split across the 2nd axis to make 1D posteriors that can be displayed in separate dock rows:
        
        # split_pseudo2D_posteriors_dict = {k:np.squeeze(p_x_given_n[:, i, :]) for i, k in enumerate(('long_LR', 'long_RL', 'short_LR', 'short_RL'))}

        # Need all_directional_pf1D_Decoder_dict
        output_dict = {}

        showCloseButton = True
        result_names = ('non_marginalized_raw_result', 'marginal_over_direction', 'marginal_over_track_ID')
        dock_configs = dict(zip(result_names, (CustomCyclicColorsDockDisplayConfig(showCloseButton=showCloseButton, named_color_scheme=NamedColorScheme.grey),
                                               CustomCyclicColorsDockDisplayConfig(showCloseButton=showCloseButton, named_color_scheme=NamedColorScheme.grey),
                                                CustomCyclicColorsDockDisplayConfig(showCloseButton=showCloseButton, named_color_scheme=NamedColorScheme.grey))))

        if enable_non_marginalized_raw_result:
            a_posterior_name: str = 'non_marginalized_raw_result'
            assert non_marginalized_raw_result.shape[0] == 4, f"expected the 4 pseudo-y bins for the decoder in non_marginalized_raw_result.shape[1]. but found non_marginalized_raw_result.shape: {non_marginalized_raw_result.shape}"
            output_dict[a_posterior_name] = cls._perform_add_new_decoded_posterior_marginal_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=dock_configs[a_posterior_name],
                                                                                                a_variable_name=a_posterior_name, xbin=np.arange(4), time_window_centers=time_window_centers, a_1D_posterior=non_marginalized_raw_result, extended_dock_title_info=info_string)
            if enable_marginal_labels:
                identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
                label_artists_dict = {}
                for i, ax in enumerate(matplotlib_fig_axes):
                    label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL'], enable_draw_decoder_colored_lines=False)
                output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
            
        if enable_marginal_over_direction:
            a_posterior_name: str = 'marginal_over_direction'
            assert marginal_over_direction.shape[0] == 2, f"expected the 2 marginalized pseudo-y bins for the decoder in marginal_over_direction.shape[1]. but found marginal_over_direction.shape: {marginal_over_direction.shape}"
            output_dict[a_posterior_name] = cls._perform_add_new_decoded_posterior_marginal_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=dock_configs[a_posterior_name],
                                                                                                a_variable_name=a_posterior_name, xbin=np.arange(2), time_window_centers=time_window_centers, a_1D_posterior=marginal_over_direction, extended_dock_title_info=info_string)
            if enable_marginal_labels:
                identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
                label_artists_dict = {}
                for i, ax in enumerate(matplotlib_fig_axes):
                    label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['LR', 'RL'], enable_draw_decoder_colored_lines=False)
                output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
                
        if enable_marginal_over_track_ID:
            a_posterior_name: str = 'marginal_over_track_ID'
            assert marginal_over_track_ID.shape[0] == 2, f"expected the 2 marginalized pseudo-y bins for the decoder in marginal_over_track_ID.shape[1]. but found marginal_over_track_ID.shape: {marginal_over_track_ID.shape}"
            output_dict[a_posterior_name] = cls._perform_add_new_decoded_posterior_marginal_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=dock_configs[a_posterior_name],
                                                                                                a_variable_name=a_posterior_name, xbin=np.arange(2), time_window_centers=time_window_centers, a_1D_posterior=marginal_over_track_ID, extended_dock_title_info=info_string)
            if enable_marginal_labels:
                identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
                label_artists_dict = {}
                for i, ax in enumerate(matplotlib_fig_axes):
                    label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long', 'short'], enable_draw_decoder_colored_lines=False)
                output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
            
        return output_dict
    


    @classmethod
    def add_pseudo2D_decoder_decoded_epoch_marginals(cls, curr_active_pipeline, active_2d_plot, debug_print=False, **kwargs):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        # pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder        
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        if debug_print:
            print(f'time_bin_size: {time_bin_size}')
        
        info_string: str = f" - t_bin_size: {time_bin_size}"

        most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_decode_result.most_recent_continuously_decoded_dict)
        
        ## INPUTS: most_recent_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult], info_string
        return cls.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=most_recent_continuously_decoded_dict, info_string=info_string, debug_print=debug_print, **kwargs)




    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=['cls.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals'], used_by=[], creation_date='2025-06-02 09:50', related_items=[])
    @classmethod
    def add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epoch_marginals(cls, curr_active_pipeline, active_2d_plot, debug_print=False, **kwargs):
        """ adds all computed time_bin_sizes in `curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded'].continuously_decoded_result_cache_dict` from the global_computation_results as new matplotlib plot rows.

        We might not always want to add all time bin sizes, sometimes they're overwhelming. Better to generate a separate menu option for each size.

        """
        
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        # pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder        
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict

        all_time_bin_sizes_output_dict = {'non_marginalized_raw_result': [], 'marginal_over_direction': [], 'marginal_over_track_ID': []}
        # flat_all_time_bin_sizes_output_tuples_list: List[Tuple] = []
        flat_all_time_bin_sizes_output_tuples_dict: Dict[str, Tuple] = {}
        for time_bin_size, a_continuously_decoded_dict in continuously_decoded_result_cache_dict.items():
            # a_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult]
            if debug_print:
                print(f'time_bin_size: {time_bin_size}')

            info_string: str = f" - t_bin_size: {time_bin_size}"
            # output_dict = _cmd.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=_cmd._active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, **enable_rows_config_kwargs)
            output_dict = cls.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, debug_print=debug_print, **kwargs)
            for a_key, an_output_tuple in output_dict.items():
                identifier_name, *tuple_extras = an_output_tuple
                
                if a_key not in all_time_bin_sizes_output_dict:
                    all_time_bin_sizes_output_dict[a_key] = [] ## init empty list
                all_time_bin_sizes_output_dict[a_key].append(an_output_tuple)
                
                assert (identifier_name not in flat_all_time_bin_sizes_output_tuples_dict), f"identifier_name: {identifier_name} already in flat_all_time_bin_sizes_output_tuples_dict: {list(flat_all_time_bin_sizes_output_tuples_dict.keys())}"
                flat_all_time_bin_sizes_output_tuples_dict[identifier_name] = an_output_tuple
                
            # all_time_bin_sizes_output_dict['non_marginalized_raw_result'].append(output_dict['non_marginalized_raw_result']) #= all_time_bin_sizes_output_dict['non_marginalized_raw_result'] + 
            # all_time_bin_sizes_output_dict['marginal_over_direction'].append(output_dict['marginal_over_direction'])
            # all_time_bin_sizes_output_dict['marginal_over_track_ID'].append(output_dict['marginal_over_track_ID'])
            
        ## OUTPUTS: all_time_bin_sizes_output_dict
        return flat_all_time_bin_sizes_output_tuples_dict


    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=['.add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epoch_marginals'], used_by=[], creation_date='2025-06-02 09:51', related_items=[])
    def execute(self, *args, **kwargs) -> None:
        """ 
        
        
        """
        print(f'menu execute(): {self}')
        self.log_command(*args, **kwargs) # adds this command to the `menu_action_history_list` 
        
        ## To begin, the destination plot must have a matplotlib widget plot to render to:
        # print(f'AddNewDecodedEpochMarginal_MatplotlibPlotCommand.execute(...)')
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        
        # Allow overwriting with `self._spike_raster_window.spike_raster_plt_2d.params`
        if self.allow_overwrite_enabled_marginals_from_params:
            enable_non_marginalized_raw_result = active_2d_plot.params.get('enable_non_marginalized_raw_result', None)
            if enable_non_marginalized_raw_result is not None:
                self.enable_non_marginalized_raw_result = enable_non_marginalized_raw_result
                
            enable_marginal_over_direction = active_2d_plot.params.get('enable_marginal_over_direction', None)
            if enable_marginal_over_direction is not None:
                self.enable_marginal_over_direction = enable_marginal_over_direction
                
            enable_marginal_over_track_ID = active_2d_plot.params.get('enable_marginal_over_track_ID', None)
            if enable_marginal_over_track_ID is not None:
                self.enable_marginal_over_track_ID = enable_marginal_over_track_ID

        enable_rows_config_kwargs = dict(enable_non_marginalized_raw_result=self.enable_non_marginalized_raw_result, enable_marginal_over_direction=self.enable_marginal_over_direction, enable_marginal_over_track_ID=self.enable_marginal_over_track_ID)
        
        output_dict = self.add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epoch_marginals(self._active_pipeline, active_2d_plot, **enable_rows_config_kwargs)
        
        # Update display output dict:
        for a_decoder_name, an_output_tuple in output_dict.items():
            identifier_name, *tuple_extras = an_output_tuple
            self._display_output[identifier_name] = an_output_tuple

        print(f'\t AddNewDecodedEpochMarginal_MatplotlibPlotCommand.execute() is done.')


