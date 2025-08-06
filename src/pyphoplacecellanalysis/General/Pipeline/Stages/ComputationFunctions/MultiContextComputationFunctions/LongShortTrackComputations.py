from copy import deepcopy
from enum import Enum # required by `FiringRateActivitySource` enum
from dataclasses import dataclass
from pathlib import Path # required by `SortOrderMetric` class

import h5py # for to_hdf and read_hdf definitions
import numpy as np
import tables as tb # for to_hdf and read_hdf definitions
import pandas as pd
from attrs import define, field # used for `JonathanFiringRateAnalysisResult`, `LongShortPipelineTests`, `LeaveOneOutDecodingAnalysis`

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.indexing_helpers import join_on_index

from neuropy.analyses.placefields import PfND # used in `constrain_to_laps` to construct new objects
from neuropy.core.epoch import Epoch, ensure_dataframe

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_full_session_leave_one_out_decoding_analysis
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingAnalysisResult
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SetPartition
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

## Private Computation Function Includes:
from shapely.geometry import LineString # for compute_polygon_overlap
from shapely.ops import unary_union, polygonize # for compute_polygon_overlap
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons, _compare_computation_results # for compute_polygon_overlap
from scipy.signal import convolve as convolve # compute_convolution_overlap

from scipy import stats # for compute_relative_entropy_divergence_overlap
from scipy.special import rel_entr # alternative for compute_relative_entropy_divergence_overlap

from collections import Counter # Count the Number of Occurrences in a Python list using Counter
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership
from neuropy.core.neuron_identities import NeuronIdentityDataframeAccessor, NeuronType

from neuropy.analyses import detect_pbe_epochs # used in `_perform_jonathan_replay_firing_rate_analyses(.)` if replays are missing

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData # for `pipeline_complete_compute_long_short_fr_indicies`
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import KnownFilterEpochs # for `pipeline_complete_compute_long_short_fr_indicies`
from neuropy.core.session.dataSession import DataSession # for `pipeline_complete_compute_long_short_fr_indicies`
from pyphoplacecellanalysis.General.Mixins.PickleSerializableMixin import PickleSerializableMixin

from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.special import factorial, logsumexp
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
import nptyping as ND
from nptyping import NDArray, DataFrame, Shape, assert_isinstance, Int, Structure as S
import awkward as ak # `simpler_compute_measured_vs_expected_firing_rates` new Awkward array for ragged arrays

from neuropy.core.user_annotations import UserAnnotationsManager, SessionCellExclusivityRecord
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.result_context import IdentifyingContext


@custom_define(slots=False, eq=False)
class TrackExclusivePartitionSubset(HDFMixin, AttrsBasedClassHelperMixin):
    """ holds information about a subset of aclus, e.g. that contain long-only placefields, etc. """
    is_aclu_pf_track_exclusive: np.ndarray = serialized_field()
    track_exclusive_aclus: np.ndarray = serialized_field()
    track_exclusive_df: pd.DataFrame = serialized_field()
    

    def get_refined_track_exclusive_aclus(self) -> np.ndarray:
        """ 2023-09-28 - Uses the exclusivity definitions refined by the firing rate indicies 
        
        """
        return self.track_exclusive_aclus[self.track_exclusive_df['is_refined_exclusive']]



    #TODO 2023-08-02 05:58: - [ ] These (`to_hdf`, `read_hdf`) were auto-generated and not sure if they work:

    def to_hdf(self, file_path):
        with h5py.File(file_path, 'w') as f:
            for attribute, value in self.__dict__.items():
                if isinstance(value, pd.DataFrame):
                    value.to_hdf(file_path, key=attribute)
                elif isinstance(value, np.ndarray):
                    f.create_dataset(attribute, data=value)
                # ... handle other attribute types as needed ...

    @classmethod
    def read_hdf(cls, file_path):
        with h5py.File(file_path, 'r') as f:
            attrs_dict = {}
            for attribute in cls.__annotations__:
                if attribute in f:
                    if pd.api.types.is_categorical_dtype(f[attribute]):
                        attrs_dict[attribute] = pd.read_hdf(file_path, key=attribute)
                    else:
                        attrs_dict[attribute] = np.array(f[attribute])
                # ... handle other attribute types as needed ...
        return cls(**attrs_dict)


# @custom_define(slots=False)
# class 

@custom_define(slots=False)
class EpochStatsResult(HDFMixin, AttrsBasedClassHelperMixin):
    df: pd.DataFrame = serialized_field()
    aclu_to_idx: Dict = non_serialized_field(is_computable=False)
    
@custom_define(slots=False)
class JonathanFiringRateAnalysisResult(HDFMixin, AttrsBasedClassHelperMixin):
    """ holds the outputs of `_perform_jonathan_replay_firing_rate_analyses` 
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
        jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
        jonathan_firing_rate_analysis_result.neuron_replay_stats_df.to_clipboard()
    """
    rdf: DynamicParameters = serialized_field(is_hdf_handled_custom=True, metadata={'tags':['custom_hdf_implementation', 'is_hdf_handled_custom'], 'description': 'replay data frame'})
    irdf: DynamicParameters = serialized_field(is_hdf_handled_custom=True, metadata={'tags':['custom_hdf_implementation', 'is_hdf_handled_custom'], 'description': 'replay data frame'})
    time_binned_unit_specific_spike_rate: DynamicParameters = non_serialized_field(is_computable=False)
    time_binned_instantaneous_unit_specific_spike_rate: DynamicParameters = non_serialized_field(is_computable=False)
    neuron_replay_stats_df: pd.DataFrame = non_serialized_field() # serialized_field(is_hdf_handled_custom=True, metadata={'tags':['custom_hdf_implementation', 'is_hdf_handled_custom']})
    

    @function_attributes(short_name=None, tags=['cell', 'partition', 'exclusivity'], input_requires=[], output_provides=[], uses=['TrackExclusivePartitionSubset'], used_by=[], creation_date='2023-06-20 00:00', related_items=[])
    def get_cell_track_partitions(self, frs_index_inclusion_magnitude:float=0.5) -> Tuple[pd.DataFrame, TrackExclusivePartitionSubset, TrackExclusivePartitionSubset, TrackExclusivePartitionSubset, TrackExclusivePartitionSubset, TrackExclusivePartitionSubset, TrackExclusivePartitionSubset]:
        """ 2023-06-20 - Partition the `neuron_replay_stats_df` into subsets by seeing whether each aclu has a placefield for the long/short track.
            # Four distinct subgroups are formed:  pf on neither, pf on both, pf on only long, pf on only short
            # L_only_aclus, S_only_aclus

            #TODO 2023-05-23 - Can do more detailed peaks analysis with: `long_results.RatemapPeaksAnalysis` and `short_results.RatemapPeaksAnalysis`

            As a side-effect it also updates `self.neuron_replay_stats_df` with the 'is_refined_exclusive', 'is_refined_LxC', 'is_refined_SxC' column

        Usage:
            ## Refine the LxC/SxC designators using the firing rate index metric:
            frs_index_inclusion_magnitude:float = 0.5

            ## Get global `long_short_fr_indicies_analysis`:
            long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
            long_short_fr_indicies_df = long_short_fr_indicies_analysis_results['long_short_fr_indicies_df']
            jonathan_firing_rate_analysis_result.refine_exclusivity_by_inst_frs_index(long_short_fr_indicies_df, frs_index_inclusion_magnitude=frs_index_inclusion_magnitude)

            neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=frs_index_inclusion_magnitude)
            
        """
        # needs `neuron_replay_stats_df`
        neuron_replay_stats_df = self.neuron_replay_stats_df #.copy()
        # neuron_replay_stats_df = neuron_replay_stats_df.sort_values(by=['long_pf_peak_x'], inplace=False, ascending=True)
        use_refined_aclus = True

        if 'custom_frs_index' not in neuron_replay_stats_df.columns:
            print(f"WARNINGL: neuron_replay_stats_df must have refindments added")
            use_refined_aclus = False

        if use_refined_aclus:
            neuron_replay_stats_df['is_refined_exclusive'] = False # fill all with False to start
            neuron_replay_stats_df['is_refined_LxC'] = False # fill all with False to start
            neuron_replay_stats_df['is_refined_SxC'] = False # fill all with False to start

        ## 2023-05-19 - Get S-only pfs
        is_S_pf_only = np.logical_and(np.logical_not(neuron_replay_stats_df['has_long_pf']), neuron_replay_stats_df['has_short_pf'])
        _is_S_only = neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY
        assert (is_S_pf_only == _is_S_only).all()
        ## Refine based on ['is_rate_extrema'] computed from `custom_frs_index`
        assert 'custom_frs_index' in neuron_replay_stats_df.columns, f"neuron_replay_stats_df must have refindments added"
        if use_refined_aclus:
            _is_refined_S_only = np.logical_and(_is_S_only, (neuron_replay_stats_df['custom_frs_index'] < -frs_index_inclusion_magnitude))
            neuron_replay_stats_df.loc[_is_refined_S_only, 'is_refined_exclusive'] = True
            neuron_replay_stats_df.loc[_is_refined_S_only, 'is_refined_SxC'] = True
            # _is_S_only = _is_refined_S_only
        S_only_aclus = neuron_replay_stats_df.index[_is_S_only].to_numpy()
        S_only_df = neuron_replay_stats_df[is_S_pf_only]

        ## Show L-only pfs stop replaying on S
        is_L_pf_only = np.logical_and(np.logical_not(neuron_replay_stats_df['has_short_pf']), neuron_replay_stats_df['has_long_pf'])
        _is_L_only = neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY
        assert (is_L_pf_only == _is_L_only).all()
        ## Refine based on ['is_rate_extrema'] computed from `custom_frs_index`
        assert 'custom_frs_index' in neuron_replay_stats_df.columns, f"neuron_replay_stats_df must have refindments added"
        if use_refined_aclus:
            _is_refined_L_only = np.logical_and(_is_L_only, (neuron_replay_stats_df['custom_frs_index'] > frs_index_inclusion_magnitude))
            neuron_replay_stats_df.loc[_is_refined_L_only, 'is_refined_exclusive'] = True
            neuron_replay_stats_df.loc[_is_refined_L_only, 'is_refined_LxC'] = True
            # _is_L_only = _is_refined_L_only
        L_only_aclus = neuron_replay_stats_df.index[_is_L_only].to_numpy()
        L_only_df = neuron_replay_stats_df[_is_L_only]

        #TODO 2023-09-28 16:15: - [ ] fix the combination properties. Would work if we directly used the computed _is_L_only and _is_S_only above
        print(f'WARN: 2023-09-28 16:15: - [ ] fix the combination properties. Would work if we directly used the computed _is_L_only and _is_S_only above')

        ## For ('kdiba', 'gor01', 'one', '2006-6-09_1-22-43') - Have L-only cells [24, 98] that have ['short_num_replays'] = [8, 7]. We were hoping that there would be few to no replays on the S-track that involved L-only cells.
        ## 2023-05-23 - Get Common (SHARED) placefields
        ## Goal 1: From the cells with the placefields on both tracks, compute the degree to which they remap in position and sort them according to their distance.
        is_BOTH_pf_only = np.logical_and(neuron_replay_stats_df['has_short_pf'], neuron_replay_stats_df['has_long_pf']) # (63,)
        BOTH_pf_only_aclus = neuron_replay_stats_df.index[is_BOTH_pf_only].to_numpy()

        ## NOTE: is_BOTH_pf_only is a much more stringent requirement (and a strict subset) than `is_BOTH_only`
        _is_BOTH_only = neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED # (99,)
        _BOTH_only_aclus = neuron_replay_stats_df.index[_is_BOTH_only].to_numpy()
        assert _BOTH_only_aclus.shape[0] >= BOTH_pf_only_aclus.shape[0]

        BOTH_pf_only_df = neuron_replay_stats_df[is_BOTH_pf_only].copy()
        BOTH_pf_only_df['long_short_pf_peak_x_displacement'] = BOTH_pf_only_df['long_pf_peak_x'].values - BOTH_pf_only_df['short_pf_peak_x'].values
        BOTH_pf_only_df['long_short_pf_peak_x_distance'] = BOTH_pf_only_df['long_short_pf_peak_x_displacement'].abs()
        BOTH_pf_only_df.sort_values(by=['long_short_pf_peak_x_distance'], inplace=True, ascending=False)

        is_EITHER_pf_only = np.logical_or(neuron_replay_stats_df['has_short_pf'], neuron_replay_stats_df['has_long_pf']) # (63,)
        EITHER_pf_only_aclus = neuron_replay_stats_df.index[is_EITHER_pf_only].to_numpy()
        EITHER_pf_only_df = neuron_replay_stats_df[is_EITHER_pf_only].copy()
        

        is_XOR_pf_only = np.logical_xor(neuron_replay_stats_df['has_short_pf'], neuron_replay_stats_df['has_long_pf'])
        # XOR_pf_only_aclus = np.hstack((L_only_aclus, S_only_aclus))
        XOR_pf_only_aclus = neuron_replay_stats_df.index[is_XOR_pf_only].to_numpy()
        XOR_only_df = neuron_replay_stats_df[is_XOR_pf_only]

        is_NEITHER_pf_only = np.logical_and(np.logical_not(neuron_replay_stats_df['has_short_pf']), np.logical_not(neuron_replay_stats_df['has_long_pf'])) # (63,)
        NEITHER_pf_only_aclus = neuron_replay_stats_df.index[is_NEITHER_pf_only].to_numpy()
        NEITHER_only_df = neuron_replay_stats_df[is_NEITHER_pf_only]

        # is_S_pf_only, is_L_pf_only, is_BOTH_pf_only, is_EITHER_pf_only
        # S_only_aclus, L_only_aclus, BOTH_pf_only_aclus, EITHER_pf_only_aclus
        # S_only_df, L_only_df, BOTH_pf_only_df, neuron_replay_stats_df

        short_exclusive = TrackExclusivePartitionSubset(is_S_pf_only, S_only_aclus, S_only_df)
        long_exclusive = TrackExclusivePartitionSubset(is_L_pf_only, L_only_aclus, L_only_df)
        BOTH_subset = TrackExclusivePartitionSubset(is_BOTH_pf_only, BOTH_pf_only_aclus, BOTH_pf_only_df)
        EITHER_subset = TrackExclusivePartitionSubset(is_EITHER_pf_only, EITHER_pf_only_aclus, EITHER_pf_only_df)
        XOR_subset = TrackExclusivePartitionSubset(is_XOR_pf_only, XOR_pf_only_aclus, XOR_only_df)
        NEITHER_subset = TrackExclusivePartitionSubset(is_NEITHER_pf_only, NEITHER_pf_only_aclus, NEITHER_only_df)
        
        # Sort dataframe by 'long_pf_peak_x' now so the aclus aren't out of order.
        neuron_replay_stats_df = neuron_replay_stats_df.sort_values(by=['long_pf_peak_x'], inplace=False, ascending=True)
        return neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset

    


    # HDFMixin Conformances ______________________________________________________________________________________________ #
    
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        # super().to_hdf(file_path, key=key, **kwargs)
        # Finish for the custom properties
        
        #TODO 2023-08-04 12:09: - [ ] Included outputs_local/global

        # Convert the ['track_membership', 'neuron_type'] columns of self._neuron_replay_stats_df to the categorical type if needed
        
        _neuron_replay_stats_df = deepcopy(self.neuron_replay_stats_df)
        
        active_context = kwargs.pop('active_context', None) # TODO: requiring that the caller pass active_context isn't optimal
        _neuron_replay_stats_df = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(_neuron_replay_stats_df, active_context=active_context, aclu_column_name=None)
        try:
            _neuron_replay_stats_df.to_hdf(file_path, key=f'{key}/neuron_replay_stats_df', format='table', data_columns=True) # TypeError: Cannot serialize the column [neuron_type] because its data contents are not [string] but [mixed] object dtype
        except TypeError:
            _neuron_replay_stats_df.astype(str).to_hdf(file_path, key=f'{key}/neuron_replay_stats_df', format='table', data_columns=True)
        except BaseException:
            raise

        self.rdf.rdf.to_hdf(file_path, key=f'{key}/rdf/df') # , format='table', data_columns=True Can't do 'table' format because `TypeError: Cannot serialize the column [firing_rates] because its data contents are not [string] but [mixed] object dtype`
        aclu_to_idx: Dict = self.rdf.aclu_to_idx
        aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
        aclu_to_idx_df.to_hdf(file_path, key=f'{key}/rdf/aclu_to_idx_df', format='table', data_columns=True)

        # irdf_group[f'{outputs_local_key}/df'] = self.irdf.irdf
        self.irdf.irdf.to_hdf(file_path, key=f'{key}/irdf/df') # , format='table', data_columns=True Can't do 'table' format because `TypeError: Cannot serialize the column [firing_rates] because its data contents are not [string] but [mixed] object dtype`
        aclu_to_idx: Dict = self.irdf.aclu_to_idx
        aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
        aclu_to_idx_df.to_hdf(file_path, key=f'{key}/irdf/aclu_to_idx_df', format='table', data_columns=True)


    def get_df_for_CSV(self, *kwargs) -> pd.DataFrame:
        """ Saves the object to key in the hdf5 file specified by file_path"""    
        _neuron_replay_stats_df: pd.DataFrame = deepcopy(self.neuron_replay_stats_df)
        
        active_context = kwargs.pop('active_context', None) # TODO: requiring that the caller pass active_context isn't optimal
        _neuron_replay_stats_df = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(_neuron_replay_stats_df, active_context=active_context, aclu_column_name=None)

        # _neuron_replay_stats_df.astype(str).to_hdf(file_path, key=f'{key}/neuron_replay_stats_df', format='table', data_columns=True)

        # self.rdf.rdf.to_hdf(file_path, key=f'{key}/rdf/df') # , format='table', data_columns=True Can't do 'table' format because `TypeError: Cannot serialize the column [firing_rates] because its data contents are not [string] but [mixed] object dtype`
        aclu_to_idx: Dict = self.rdf.aclu_to_idx
        aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
        # aclu_to_idx_df.to_hdf(file_path, key=f'{key}/rdf/aclu_to_idx_df', format='table', data_columns=True)

        # irdf_group[f'{outputs_local_key}/df'] = self.irdf.irdf
        # self.irdf.irdf.to_hdf(file_path, key=f'{key}/irdf/df') # , format='table', data_columns=True Can't do 'table' format because `TypeError: Cannot serialize the column [firing_rates] because its data contents are not [string] but [mixed] object dtype`
        aclu_to_idx: Dict = self.irdf.aclu_to_idx
        aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
        # aclu_to_idx_df.to_hdf(file_path, key=f'{key}/irdf/aclu_to_idx_df', format='table', data_columns=True)
        return _neuron_replay_stats_df




    def refine_exclusivity_by_inst_frs_index(self, custom_SpikeRateTrends_df: pd.DataFrame, frs_index_inclusion_magnitude: float = 0.5, override_existing_frs_index_values:bool=False) -> bool: 
        """ 
        inst_frs_index_inclusion_magnitude: float = 0.5 # the magnitude of the value for a candidate LxC/SxC to be included:


        Adds ['custom_frs_index', 'is_refined_exclusive'] to both: (short_exclusive.track_exclusive_df, long_exclusive.track_exclusive_df)

        Returns: bool - Indiciating whether the `custom_SpikeRateTrends_df` was updated or whether it already had all of the needed columns and computation was skipped.

        """
        if (not override_existing_frs_index_values) and np.isin(['aclu','custom_frs_index','is_rate_extrema','is_refined_exclusive','is_refined_LxC','is_refined_SxC'], custom_SpikeRateTrends_df.columns).all():
            # all columns already present. We can skip.
            return False
    
        if 'aclu' not in custom_SpikeRateTrends_df.columns:
            custom_SpikeRateTrends_df['aclu'] = custom_SpikeRateTrends_df.index
        if 'custom_frs_index' not in custom_SpikeRateTrends_df.columns:
            custom_SpikeRateTrends_df['custom_frs_index'] = custom_SpikeRateTrends_df['non_replays_frs_index']
        
        all_aclus = self.neuron_replay_stats_df.index.to_numpy()
        instSpikeRate_values_df = custom_SpikeRateTrends_df[np.isin(custom_SpikeRateTrends_df.aclu, all_aclus)]
        refined_track_exclusive_aclus = instSpikeRate_values_df[(instSpikeRate_values_df.custom_frs_index < -frs_index_inclusion_magnitude)].aclu.to_numpy()
        # assert 'aclu' in self.neuron_replay_stats_df
        self.neuron_replay_stats_df['aclu'] = all_aclus
        self.neuron_replay_stats_df['custom_frs_index'] = instSpikeRate_values_df.custom_frs_index
        self.neuron_replay_stats_df['is_rate_extrema'] = (np.abs(self.neuron_replay_stats_df['custom_frs_index'].to_numpy()) > frs_index_inclusion_magnitude)
        
        ## Setup the 'is_refined_exclusive' column
        self.neuron_replay_stats_df['is_refined_exclusive'] = False # fill all with False to start
        self.neuron_replay_stats_df['is_refined_LxC'] = False # fill all with False to start
        self.neuron_replay_stats_df['is_refined_SxC'] = False # fill all with False to start

        _is_L_only = self.neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY
        _is_refined_L_only = np.logical_and(_is_L_only, (self.neuron_replay_stats_df['custom_frs_index'] > frs_index_inclusion_magnitude))
        self.neuron_replay_stats_df.loc[_is_refined_L_only, 'is_refined_exclusive'] = True
        self.neuron_replay_stats_df.loc[_is_refined_L_only, 'is_refined_LxC'] = True

        # _is_S_only = np.logical_and(np.logical_not(self.neuron_replay_stats_df['has_long_pf']), self.neuron_replay_stats_df['has_short_pf'])
        _is_S_only = self.neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY
        _is_refined_S_only = np.logical_and(_is_S_only, (self.neuron_replay_stats_df['custom_frs_index'] < -frs_index_inclusion_magnitude))
        self.neuron_replay_stats_df.loc[_is_refined_S_only, 'is_refined_exclusive'] = True
        self.neuron_replay_stats_df.loc[_is_refined_S_only, 'is_refined_SxC'] = True

        return True

    @classmethod
    def _perform_add_peak_promenance_pf_peaks(cls, neuron_replay_stats_df: pd.DataFrame, curr_active_pipeline, track_templates):
        """ adds `active_peak_prominence_2d_results` to existing `neuron_replay_stats_df` from `jonathan_firing_rate_analysis_result`, adding the `['long_pf2D_peak_x', 'long_pf2D_peak_y'] + ['short_pf2D_peak_x', 'short_pf2D_peak_y']` columns

        Updated to use directional values not just long/short 

        
        Usage:

            jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
            neuron_replay_stats_df: pd.DataFrame = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)

            long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any'] # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            active_result_epoch_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name]

            neuron_replay_stats_df, all_modified_columns = _add_peak_promenance_pf_peaks(neuron_replay_stats_df=neuron_replay_stats_df,
                                                                                        curr_active_pipeline=curr_active_pipeline,
                                                                                        #  active_result_epoch_names=active_result_epoch_names,
                                                                                        track_templates=track_templates,
                                                                                        )

            print(f'all_modified_columns: {all_modified_columns}') # all_modified_columns: ['maze1_odd_pf2D_peak_x', 'maze1_odd_pf2D_peak_y', 'maze1_even_pf2D_peak_x', 'maze1_even_pf2D_peak_y', 'maze2_odd_pf2D_peak_x', 'maze2_odd_pf2D_peak_y', 'maze2_even_pf2D_peak_x', 'maze2_even_pf2D_peak_y']
            neuron_replay_stats_df

        all_modified_columns: ['long_LR_pf2D_peak_x', 'long_LR_pf2D_peak_y', 'long_RL_pf2D_peak_x', 'long_RL_pf2D_peak_y', 'short_LR_pf2D_peak_x', 'short_LR_pf2D_peak_y', 'short_RL_pf2D_peak_x', 'short_RL_pf2D_peak_y']

        
        History:
            Added 2024-06-11 
        """
        ## INPUTS: neuron_replay_stats_df, curr_active_pipeline, track_templates
        all_modified_columns = []

        neuron_replay_stats_df = deepcopy(neuron_replay_stats_df)

        # Converting between decoder names and filtered epoch names:
        # {'long':'maze1', 'short':'maze2'}
        # {'LR':'odd', 'RL':'even'}
        long_LR_name, short_LR_name, long_RL_name, short_RL_name = ['maze1_odd', 'maze2_odd', 'maze1_even', 'maze2_even']
        decoder_name_to_session_context_name: Dict[str,str] = dict(zip(track_templates.get_decoder_names(), (long_LR_name, long_RL_name, short_LR_name, short_RL_name))) # {'long_LR': 'maze1_odd', 'long_RL': 'maze1_even', 'short_LR': 'maze2_odd', 'short_RL': 'maze2_even'}

        ## try to add the 2D peak information to the cells in `neuron_replay_stats_df`:
        for a_decoder_name, a_computation_result_context_name in decoder_name_to_session_context_name.items():
        
            # columns should be created with the modern encoder names: (e.g. 'long_LR', ...)
            a_peak_x_col_name: str = f'{a_decoder_name}_pf2D_peak_x'
            a_peak_y_col_name: str = f'{a_decoder_name}_pf2D_peak_y'

            a_modified_columns = [a_peak_x_col_name, a_peak_y_col_name]

            # Null out existing columns or initialize them to null:
            neuron_replay_stats_df[a_peak_x_col_name] = pd.NA
            neuron_replay_stats_df[a_peak_y_col_name] = pd.NA
        
            # flat_peaks_df: pd.DataFrame = deepcopy(active_peak_prominence_2d_results['flat_peaks_df']).reset_index(drop=True)
            a_filtered_flat_peaks_df: pd.DataFrame = deepcopy(curr_active_pipeline.computation_results[a_computation_result_context_name].computed_data['RatemapPeaksAnalysis']['PeakProminence2D']['filtered_flat_peaks_df']).reset_index(drop=True)
            neuron_replay_stats_df.loc[np.isin(neuron_replay_stats_df['aclu'].to_numpy(), a_filtered_flat_peaks_df.neuron_id.to_numpy()), a_modified_columns] = a_filtered_flat_peaks_df[['peak_center_x', 'peak_center_y']].to_numpy()

            all_modified_columns.extend(a_modified_columns)

        # end for
        
        return neuron_replay_stats_df, all_modified_columns


    def add_peak_promenance_pf_peaks(self, curr_active_pipeline, track_templates):
        """ modifies `self.neuron_replay_stats_df`, adding the 2D peak information to the cells as new columns from the peak_prominence results

        Adds: ['long_LR_pf2D_peak_x', 'long_LR_pf2D_peak_y', 'long_RL_pf2D_peak_x', 'long_RL_pf2D_peak_y', 'short_LR_pf2D_peak_x', 'short_LR_pf2D_peak_y', 'short_RL_pf2D_peak_x', 'short_RL_pf2D_peak_y']

        neuron_replay_stats_df, all_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        neuron_replay_stats_df

        neuron_replay_stats_df, all_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        neuron_replay_stats_df

        History:
            Added 2024-06-11 

        """
        self.neuron_replay_stats_df, all_modified_columns = JonathanFiringRateAnalysisResult._perform_add_peak_promenance_pf_peaks(neuron_replay_stats_df=self.neuron_replay_stats_df,
                                                                             curr_active_pipeline=curr_active_pipeline,
                                                                            #  active_result_epoch_names=active_result_epoch_names,
                                                                             track_templates=track_templates,
                                                                             )
        print(f'all_modified_columns: {all_modified_columns}') # all_modified_columns: ['maze1_odd_pf2D_peak_x', 'maze1_odd_pf2D_peak_y', 'maze1_even_pf2D_peak_x', 'maze1_even_pf2D_peak_y', 'maze2_odd_pf2D_peak_x', 'maze2_odd_pf2D_peak_y', 'maze2_even_pf2D_peak_x', 'maze2_even_pf2D_peak_y']
        return self.neuron_replay_stats_df, all_modified_columns


    @function_attributes(short_name=None, tags=['maximum-peaks', 'peaks'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-11 18:33', related_items=[])
    def add_directional_pf_maximum_peaks(self, track_templates):
        """ modifies `self.neuron_replay_stats_df`, adding the columns from the peak_prominence results

        Adds: ['long_LR_pf1D_peak', 'long_RL_pf1D_peak', 'short_LR_pf1D_peak', 'short_RL_pf1D_peak', 'peak_diff_LR_pf1D_peak', 'peak_diff_RL_pf1D_peak']

        2024-04-09 - Maximum peaks only for each template. 

        neuron_replay_stats_df, all_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        neuron_replay_stats_df

        neuron_replay_stats_df, all_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        neuron_replay_stats_df
        """
        all_modified_columns = []

        neuron_replay_stats_df = deepcopy(self.neuron_replay_stats_df)
        (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df = track_templates.get_directional_pf_maximum_peaks_dfs(drop_aclu_if_missing_long_or_short=False)
        additive_df_column_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL', 'peak_diff_LR', 'peak_diff_RL']
        target_df_column_names = [f"{v}_pf1D_peak" for v in additive_df_column_names] # columns to add to `self.neuron_replay_stats_df`

        a_modified_columns = target_df_column_names
        # Null out existing columns or initialize them to null:
        neuron_replay_stats_df[target_df_column_names] = pd.NA
    
        # flat_peaks_df: pd.DataFrame = deepcopy(active_peak_prominence_2d_results['flat_peaks_df']).reset_index(drop=True)
        a_filtered_flat_peaks_df: pd.DataFrame = deepcopy(AnyDir_decoder_aclu_MAX_peak_maps_df).reset_index(drop=False, names=['aclu']) # .reset_index(drop=True)
        neuron_replay_stats_df.loc[np.isin(neuron_replay_stats_df['aclu'].to_numpy(), a_filtered_flat_peaks_df['aclu'].to_numpy()), a_modified_columns] = a_filtered_flat_peaks_df[additive_df_column_names].to_numpy()

        all_modified_columns.extend(a_modified_columns)
        self.neuron_replay_stats_df = neuron_replay_stats_df
        print(f'all_modified_columns: {all_modified_columns}') # all_modified_columns: ['maze1_odd_pf2D_peak_x', 'maze1_odd_pf2D_peak_y', 'maze1_even_pf2D_peak_x', 'maze1_even_pf2D_peak_y', 'maze2_odd_pf2D_peak_x', 'maze2_odd_pf2D_peak_y', 'maze2_even_pf2D_peak_x', 'maze2_even_pf2D_peak_y']
        return self.neuron_replay_stats_df, all_modified_columns
         

@custom_define(slots=False, repr=False)
class LeaveOneOutDecodingAnalysis(ComputedResult):
    """ 2023-05-10 - holds the results of a leave-one-out decoding analysis of the long and short track 
    Usage:
        leave_one_out_decoding_analysis_obj = LeaveOneOutDecodingAnalysis(long_decoder, short_decoder, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj)
    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    long_decoder: BayesianPlacemapPositionDecoder = serialized_field()
    short_decoder: BayesianPlacemapPositionDecoder = serialized_field()
    long_replays: pd.DataFrame = serialized_field()
    short_replays: pd.DataFrame = serialized_field()
    global_replays: pd.DataFrame = serialized_field()
    long_shared_aclus_only_decoder: BasePositionDecoder = serialized_field()
    short_shared_aclus_only_decoder: BasePositionDecoder = serialized_field()
    shared_aclus: np.ndarray = serialized_field()
    long_short_pf_neurons_diff: SetPartition = serialized_field()
    n_neurons: int = serialized_attribute_field()
    long_results_obj: LeaveOneOutDecodingAnalysisResult = serialized_field()
    short_results_obj: LeaveOneOutDecodingAnalysisResult = serialized_field()
    is_global: bool = serialized_attribute_field(default=True, kw_only=True)


@custom_define(slots=False, kw_only=True) # NOTE: kw_only=True prevents errors from only assigning some of the attributes with a specific field
class ExpectedVsObservedResult(ComputedResult):
    """ Allows initialization from 
    curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    expected_v_observed_result = curr_long_short_post_decoding.expected_v_observed_result
    expected_v_observed_result_obj = ExpectedVsObservedResult(**expected_v_observed_result.to_dict())
    expected_v_observed_result_obj
    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version

    Flat_epoch_time_bins_mean: np.ndarray = serialized_field()
    Flat_decoder_time_bin_centers: np.ndarray
    num_neurons: int = serialized_attribute_field()
    num_timebins_in_epoch: np.ndarray
    num_total_flat_timebins: np.int64 = serialized_attribute_field()
    decoder_time_bin_centers_LONG: list
    all_epochs_computed_expected_cell_num_spikes_LONG: list
    all_epochs_computed_observed_from_expected_difference_LONG: list
    measured_pos_window_centers_LONG: list
    all_epochs_decoded_epoch_time_bins_mean_LONG: np.ndarray
    all_epochs_computed_expected_cell_firing_rates_mean_LONG: np.ndarray
    all_epochs_computed_expected_cell_firing_rates_stddev_LONG: np.ndarray
    all_epochs_computed_observed_from_expected_difference_maximum_LONG: list
    Flat_decoder_time_bin_centers_LONG: np.ndarray
    Flat_all_epochs_computed_expected_cell_num_spikes_LONG: np.ndarray = serialized_field()
    returned_shape_tuple_LONG: tuple
    observed_from_expected_diff_ptp_LONG: np.ma.core.MaskedArray = serialized_field()
    observed_from_expected_diff_mean_LONG: np.ndarray = serialized_field(is_computable=True)
    observed_from_expected_diff_std_LONG: np.ndarray = serialized_field(is_computable=True)
    # Short properties:
    decoder_time_bin_centers_SHORT: list
    all_epochs_computed_expected_cell_num_spikes_SHORT: list
    all_epochs_computed_observed_from_expected_difference_SHORT: list
    measured_pos_window_centers_SHORT: list
    all_epochs_decoded_epoch_time_bins_mean_SHORT: np.ndarray
    all_epochs_computed_expected_cell_firing_rates_mean_SHORT: np.ndarray
    all_epochs_computed_expected_cell_firing_rates_stddev_SHORT: np.ndarray
    all_epochs_computed_observed_from_expected_difference_maximum_SHORT: list
    Flat_decoder_time_bin_centers_SHORT: np.ndarray
    Flat_all_epochs_computed_expected_cell_num_spikes_SHORT: np.ndarray = serialized_field()
    returned_shape_tuple_SHORT: tuple
    observed_from_expected_diff_ptp_SHORT: np.ma.core.MaskedArray = serialized_field()
    observed_from_expected_diff_mean_SHORT: np.ndarray = serialized_field(is_computable=True)
    observed_from_expected_diff_std_SHORT: np.ndarray = serialized_field(is_computable=True)
    is_short_track_epoch: np.ndarray
    is_long_track_epoch: np.ndarray
    short_short_diff: np.ndarray
    long_long_diff: np.ndarray
    
@custom_define(slots=False, kw_only=True) # NOTE: kw_only=True prevents errors from only assigning some of the attributes with a specific field
class RateRemappingResult(ComputedResult):
    """ 
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import RateRemappingResult

    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    rr_df: pd.DataFrame = serialized_field()
    high_only_rr_df: pd.DataFrame = serialized_field(is_computable=True)
    considerable_remapping_threshold: float = serialized_attribute_field(default=0.7)
    
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        # super().to_hdf(file_path, key=key, **kwargs)
        # Finish for the custom properties
        
        #TODO 2023-08-04 12:09: - [ ] Included outputs_local/global
                 
        # df: pd.DataFrame = self.rdf.rdf
        # aclu_to_idx: Dict = self.rdf.aclu_to_idx
        # aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
        
        # with tb.open_file(file_path, mode='a') as f:
        #     # f.get_node(f'{key}/rdf/df')._f_remove(recursive=True)

        #     try:
        #         rdf_group = f.create_group(key, 'rdf', title='replay data frame', createparents=True)
        #     except tb.exceptions.NodeError:
        #         # Node already exists
        #         pass
        #     except BaseException as e:
        #         raise
        #     # rdf_group[f'{outputs_local_key}/df'] = self.rdf.rdf
        #     try:
        #         irdf_group = f.create_group(key, 'irdf', title='intra-replay data frame', createparents=True)
        #     except tb.exceptions.NodeError:
        #         # Node already exists
        #         pass
        #     except BaseException as e:
        #         raise

        active_context = kwargs.pop('active_context', None) # TODO: requiring that the caller pass active_context isn't optimal
        assert active_context is not None

        ## New
        # session_context = curr_active_pipeline.get_session_context() 
        # session_group_key: str = "/" + session_context.get_description(separator="/", include_property_names=False) # 'kdiba/gor01/one/2006-6-08_14-26-15'
        # session_uid: str = session_context.get_description(separator="|", include_property_names=False)
        ## Global Computations
        # a_global_computations_group_key: str = f"{session_group_key}/global_computations"
        # with tb.open_file(file_path, mode='w') as f:
        #     a_global_computations_group = f.create_group(session_group_key, 'global_computations', title='the result of computations that operate over many or all of the filters in the session.', createparents=True)

        rate_remapping_df = deepcopy(self.rr_df)
        rate_remapping_df = rate_remapping_df[['laps', 'replays', 'skew', 'max_axis_distance_from_center', 'distance_from_center', 'has_considerable_remapping']]
        rate_remapping_df = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(rate_remapping_df, active_context=active_context, aclu_column_name=None)
        # rate_remapping_df.to_hdf(file_path, key=f'{key}/rate_remapping_df', format='table', data_columns=True)
        rate_remapping_df.to_hdf(file_path, key=f'{key}/rate_remapping', format='table', data_columns=True)


@custom_define(slots=False)
class TruncationCheckingResults(ComputedResult):
    """ result for `_perform_long_short_endcap_analysis`
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import TruncationCheckingResults
        
        global_computation_results.computed_data['long_short_endcap'].significant_distant_remapping_endcap_aclus = significant_distant_remapping_endcap_aclus
        global_computation_results.computed_data['long_short_endcap'].minor_remapping_endcap_aclus = minorly_changed_endcap_cells_df.index
        global_computation_results.computed_data['long_short_endcap'].non_disappearing_endcap_aclus = non_disappearing_endcap_cells_df.index
        global_computation_results.computed_data['long_short_endcap'].disappearing_endcap_aclus = disappearing_endcap_cells_df.index
        
    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    disappearing_endcap_aclus: pd.Index = serialized_field()
    non_disappearing_endcap_aclus: pd.Index = serialized_field()
    significant_distant_remapping_endcap_aclus: pd.Index = serialized_field()
    minor_remapping_endcap_aclus: pd.Index = serialized_field()

    # @property
    # def truncation_checking_aclus_dict(self, neuron_replay_stats_df, assign_to_df_column=True) -> Dict:

    def build_truncation_checking_aclus_dict(self, neuron_replay_stats_df, assign_to_df_column: bool=True) -> Dict:
        """The truncation_checking_aclus_dict property.

        Usage:    
            ## long_short_endcap_analysis:
            truncation_checking_result: TruncationCheckingResults = curr_active_pipeline.global_computation_results.computed_data.long_short_endcap
            truncation_checking_aclus_dict, jonathan_firing_rate_analysis_result.neuron_replay_stats_df = truncation_checking_result.build_truncation_checking_aclus_dict(neuron_replay_stats_df=jonathan_firing_rate_analysis_result.neuron_replay_stats_df)

        Integrate 'truncation_checking_aclus_dict' into `neuron_replay_stats_df` as a 'truncation_checking' column:

        """
        from neuropy.utils.indexing_helpers import union_of_arrays

        truncation_checking_result = self
        disappearing_endcap_aclus = truncation_checking_result.disappearing_endcap_aclus.to_numpy()
        non_disappearing_endcap_aclus = truncation_checking_result.non_disappearing_endcap_aclus.to_numpy()
        trivially_remapping_endcap_aclus = truncation_checking_result.minor_remapping_endcap_aclus.to_numpy()
        significant_distant_remapping_endcap_aclus = truncation_checking_result.significant_distant_remapping_endcap_aclus.to_numpy()
        # any_aclus = union_of_arrays(disappearing_endcap_aclus, non_disappearing_endcap_aclus, trivially_remapping_endcap_aclus, significant_distant_remapping_endcap_aclus, appearing_aclus)

        # truncation_checking_aclus_dict = {'disappearing': disappearing_endcap_aclus, 'non_disappearing_endcap_aclus': non_disappearing_endcap_aclus,
        #                            'significant_distant_remapping_endcap_aclus': significant_distant_remapping_endcap_aclus, 'trivially_remapping': trivially_remapping_endcap_aclus}

        truncation_checking_aclus_dict = {'disappearing': disappearing_endcap_aclus, 'non_disappearing_endcap': non_disappearing_endcap_aclus,
                                'significant_distant_remapping_endcap': significant_distant_remapping_endcap_aclus, 'trivially_remapping': trivially_remapping_endcap_aclus}
        

        appearing_aclus = neuron_replay_stats_df[neuron_replay_stats_df['track_membership'] == SplitPartitionMembership.RIGHT_ONLY].index.to_numpy()
        truncation_checking_aclus_dict.update({'appearing': appearing_aclus})

        any_aclus = union_of_arrays(*[v for v in truncation_checking_aclus_dict.values() if len(v) > 0])
        truncation_checking_aclus_dict.update({'any': any_aclus})

        # return {'any': any_aclus, 'disappearing': disappearing_endcap_aclus, 'appearing': appearing_aclus, 'non_disappearing_endcap_aclus': non_disappearing_endcap_aclus,
                                        # 'significant_distant_remapping_endcap_aclus': significant_distant_remapping_endcap_aclus, 'trivially_remapping': trivially_remapping_endcap_aclus}
        
        if assign_to_df_column:
            ## Integrate 'truncation_checking_aclus_dict' into `neuron_replay_stats_df` as a 'truncation_checking' column:
            neuron_replay_stats_df['truncation_checking'] = 'none'

            for k, v in truncation_checking_aclus_dict.items():
                if k != 'any':
                    neuron_replay_stats_df.loc[np.isin(neuron_replay_stats_df['aclu'], truncation_checking_aclus_dict[k]), 'truncation_checking'] = k


            neuron_replay_stats_df


        return truncation_checking_aclus_dict, neuron_replay_stats_df
    


    



@define(slots=False, repr=False)
class LongShortPipelineTests:
    """2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks."""
    curr_active_pipeline: "NeuropyPipeline"

    def validate_placefields(self, override_long_epoch_name:Optional[str]=None, override_short_epoch_name:Optional[str]=None):
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
        long_epoch_name, short_epoch_name, global_epoch_name = self.curr_active_pipeline.find_LongShortGlobal_epoch_names()
        if override_long_epoch_name is not None:
            long_epoch_name = override_long_epoch_name
        if override_short_epoch_name is not None:
            short_epoch_name = override_short_epoch_name
        long_results, short_results, global_results = [self.curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # Assert conformance between the long and short position bins for both the 1D and 2D placefields. This should be the case because pf_params.grid_bin and pf_params.grid_bin_bounds are set to the same for both tracks:
        assert np.all(long_results.pf1D.xbin == short_results.pf1D.xbin), f"long_results.pf1D.xbin: {len(long_results.pf1D.xbin)}, short_results.pf1D.xbin: {len(short_results.pf1D.xbin)}"
        assert np.all(long_results.pf2D.xbin == short_results.pf2D.xbin), f"long_results.pf2D.xbin: {len(long_results.pf2D.xbin)}, short_results.pf2D.xbin: {len(short_results.pf2D.xbin)}"
        assert np.all(long_results.pf2D.ybin == short_results.pf2D.ybin), f"long_results.pf2D.ybin: {len(long_results.pf2D.ybin)}, short_results.pf2D.ybin: {len(short_results.pf2D.ybin)}"

    def validate_decoders(self, override_long_epoch_name:Optional[str]=None, override_short_epoch_name:Optional[str]=None):
        """Decoders should also conform if placefields do from the onset prior to computations:

        Validates the position bins of : pf1D_Decoder, pf2D_Decoder
        """
        long_epoch_name, short_epoch_name, global_epoch_name = self.curr_active_pipeline.find_LongShortGlobal_epoch_names()
        if override_long_epoch_name is not None:
            long_epoch_name = override_long_epoch_name
        if override_short_epoch_name is not None:
            short_epoch_name = override_short_epoch_name
        long_results, short_results, global_results = [self.curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        assert np.all(long_results.pf1D_Decoder.xbin == short_results.pf1D_Decoder.xbin), f"long_results.pf1D_Decoder.xbin: {len(long_results.pf1D_Decoder.xbin)}, short_results.pf1D_Decoder.xbin: {len(short_results.pf1D_Decoder.xbin)}"
        assert np.all(long_results.pf2D_Decoder.xbin == short_results.pf2D_Decoder.xbin), f"long_results.pf2D_Decoder.xbin: {len(long_results.pf2D_Decoder.xbin)}, short_results.pf2D_Decoder.xbin: {len(short_results.pf2D_Decoder.xbin)}"
        assert np.all(long_results.pf2D_Decoder.ybin == short_results.pf2D_Decoder.ybin), f"long_results.pf2D_Decoder.ybin: {len(long_results.pf2D_Decoder.ybin)}, short_results.pf2D_Decoder.ybin: {len(short_results.pf2D_Decoder.ybin)}"

    def validate(self, override_long_epoch_name:Optional[str]=None, override_short_epoch_name:Optional[str]=None) -> bool:
        long_epoch_name, short_epoch_name, global_epoch_name = self.curr_active_pipeline.find_LongShortGlobal_epoch_names()
        if override_long_epoch_name is not None:
            long_epoch_name = override_long_epoch_name
        if override_short_epoch_name is not None:
            short_epoch_name = override_short_epoch_name

        # Run the tests:
        try:
            self.validate_placefields(override_long_epoch_name=long_epoch_name, override_short_epoch_name=short_epoch_name)
            self.validate_decoders(override_long_epoch_name=long_epoch_name, override_short_epoch_name=short_epoch_name)
            return True
        
        except AssertionError:
            return False
            
        except Exception:
            raise # unhandled exception

    def __call__(self, override_long_epoch_name:Optional[str]=None, override_short_epoch_name:Optional[str]=None) -> bool:
        return self.validate(override_long_epoch_name=override_long_epoch_name, override_short_epoch_name=override_short_epoch_name)




# ==================================================================================================================== #
# BEGIN COMPUTATION FUNCTIONS                                                                                          #
# ==================================================================================================================== #
class LongShortTrackComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    
    _computationGroupName = 'long_short_track'
    _computationPrecidence = 1003
    _is_global = True

    @function_attributes(short_name='long_short_decoding_analyses', tags=['long_short', 'short_long', 'leave-one-out', 'replay', 'decoding', 'computation'], input_requires=['sess.replays'], output_provides=['global_computation_results.computed_data.long_short_leave_one_out_decoding_analysis'], uses=['_long_short_decoding_analysis_from_decoders'], used_by=[], creation_date='2023-05-10 15:10',
                         requires_global_keys=[], provides_global_keys=['long_short_leave_one_out_decoding_analysis'],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].long_results_obj, curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].short_results_obj), is_global=True)
    def _perform_long_short_decoding_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, decoding_time_bin_size=None, perform_cache_load=False, always_recompute_replays=False, override_long_epoch_name:Optional[str]=None, override_short_epoch_name:Optional[str]=None):
        """ Performs decoding for replay epochs after ensuring that the long and short placefields are properly constrained to match one another.
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['long_short_leave_one_out_decoding_analysis']
                # ['long_short_leave_one_out_decoding_analysis']['short_long_neurons_diff']
                # ['long_short_leave_one_out_decoding_analysis']['poly_overlap_df']

        


        Call Hierarchy:
            - if not `is_certain_properly_constrained`, calls `compute_long_short_constrained_decoders` to build placefields and decoders constrained to the same position bins


        #TODO 2024-03-29 20:16: - [ ] This doesn't seem to correctly validate after it has been computed, it keeps recomputing every time.

        """
        # # New unified `pipeline_complete_compute_long_short_fr_indicies(...)` method for entire pipeline:
        # x_frs_index, y_frs_index, active_context, all_results_dict = pipeline_complete_compute_long_short_fr_indicies(owning_pipeline_reference) # use the all_results_dict as the computed data value
        # global_computation_results.computed_data['long_short_fr_indicies_analysis'] = DynamicParameters.init_from_dict({**all_results_dict, 'active_context': active_context})

        ## Get the active long/short epoch names:
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        if override_long_epoch_name is not None:
            long_epoch_name = override_long_epoch_name
        if override_short_epoch_name is not None:
            short_epoch_name = override_short_epoch_name
        is_certain_properly_constrained = LongShortPipelineTests(owning_pipeline_reference).validate(override_long_epoch_name=long_epoch_name, override_short_epoch_name=short_epoch_name)
        # is_certain_properly_constrained = False

        """ a properly constrained pipeline has the computation_epochs for its placefields equal to its laps, 
            - equal n_bins between short and long.
            
        """

        # 2023-05-16 - Correctly initialized pipelines (pfs limited to laps, decoders already long/short constrainted by default, replays already the estimated versions:
        # is_certain_properly_constrained = True

        if not is_certain_properly_constrained:
            print(f'WARN: _perform_long_short_decoding_analyses: Not certain if pipeline results are properly constrained. Need to recompute and update.')
            owning_pipeline_reference = constrain_to_laps(owning_pipeline_reference) # Constrains placefields to laps
            
            (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_long_short_constrained_decoders(owning_pipeline_reference, long_epoch_name=long_epoch_name, short_epoch_name=short_epoch_name, recalculate_anyway=True)
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

            # Now we are certain that it's properly constrained. If changes were made, we'll need to save
            #TODO 2023-10-11 12:12: - [ ] Save indicator that it IS properly constrained so long as: 'grid_bin_bounds', 'grid_bin', TODO_MORE don't change. Also store the current date.
            # owning_pipeline_reference # will become invalidated when grid_bin_bounds, grid_bin, etc change.


            # 3m 40.3s
        else:
            print(f'`is_certain_properly_constrained`: True - Correctly initialized pipelines (pfs limited to laps, decoders already long/short constrainted by default, replays already the estimated versions')
            if always_recompute_replays:
                print(f'\t `is_certain_properly_constrained` IGNORES always_recompute_replays!')
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_results, short_results, global_results = [owning_pipeline_reference.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            long_one_step_decoder_1D, short_one_step_decoder_1D  = [deepcopy(results_data.get('pf1D_Decoder', None)) for results_data in (long_results, short_results)]


        if decoding_time_bin_size is None:
            decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333
        else:
            # check if decoding_time_bin_size is the same
            if not (decoding_time_bin_size == long_one_step_decoder_1D.time_bin_size):
                print(f'`decoding_time_bin_size` different than decoder: decoding_time_bin_size: {decoding_time_bin_size}, long_one_step_decoder_1D.time_bin_size: {long_one_step_decoder_1D.time_bin_size}')
                raise NotImplementedError
                # TODO: invalidate cached
                perform_cache_load = False
                ## Update `long_one_step_decoder_1D.time_bin_size` to the new size? TODO 2023-05-10 - redo computations with this size for `long_one_step_decoder_1D`?
                long_one_step_decoder_1D.time_bin_size = decoding_time_bin_size
                


        ## Perform the actual `_long_short_decoding_analysis_from_decoders`:
        leave_one_out_decoding_analysis_obj = _long_short_decoding_analysis_from_decoders(long_one_step_decoder_1D, short_one_step_decoder_1D, long_session, short_session, global_session,
                                                                                           decoding_time_bin_size=decoding_time_bin_size, perform_cache_load=perform_cache_load)


        #TODO 2023-10-25 08:45: - [ ] Need to make separate results based on the passed in long/short epoch names.
        # specifically the directional placefield analyses only need `long_shared_aclus_only_decoder`, `short_shared_aclus_only_decoder` from this analysis

        # TODO 2023-05-10 - need to update existing ['long_short'] if it exists:
        # global_computation_results.computed_data['long_short'] = {
        #     'leave_one_out_decoding_analysis': leave_one_out_decoding_analysis_obj
        # } # end long_short

        if not 'long_short_leave_one_out_decoding_analysis' in global_computation_results.computed_data:
            global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'] = leave_one_out_decoding_analysis_obj # end long_short
        else:
            print(f'WARN: overwriting existing result `_perform_long_short_decoding_analyses`.')
            global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'] = leave_one_out_decoding_analysis_obj
        # TODO 2023-05-10 - Do I want long_one_step_decoder_2D, short_one_step_decoder_2D that I computed?

        if not np.all([hasattr(global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'], 'long_results_obj'), hasattr(global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'], 'short_results_obj')]):
            print(f'WARN: FIXME: `_perform_long_short_decoding_analyses` failed to validate its properties even after fresh computation! FIX THIS. 2024-03-29 20:21!')

        """ Getting outputs:
        
        
            ## long_short_decoding_analyses:
            curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
            ## Extract variables from results object:
            long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global


        """
        return global_computation_results
    

    @function_attributes(short_name='long_short_rate_remapping', tags=['long_short', 'short_long','replay', 'rate_remapping', 'computation', '?BROKEN?'], input_requires=['global_computation_results.computed_data.long_short_leave_one_out_decoding_analysis'], output_provides=['global_computation_results.computed_data.long_short_rate_remapping'], uses=['compute_rate_remapping_stats'], used_by=[], creation_date='2023-05-31 13:57',
                        requires_global_keys=['long_short_fr_indicies_analysis'], provides_global_keys=['long_short_rate_remapping'], is_global=True)
    def _perform_long_short_decoding_rate_remapping_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, decoding_time_bin_size=None, perform_cache_load=False, always_recompute_replays=False):
        """ Computes rate remapping statistics
        
        Requires:
            ['global_computation_results.computed_data.long_short_leave_one_out_decoding_analysis']
            
        Provides:
            computation_result.computed_data['long_short_rate_remapping']
                # ['long_short_rate_remapping']['rr_df']
                # ['long_short_rate_remapping']['high_only_rr_df']
        
        """
        
        from neuropy.core.neurons import NeuronType
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_rate_remapping_stats
        
        ## long_short_decoding_analyses:
        # curr_long_short_decoding_analyses = global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'] # end long_short
        # active_analyses_result = curr_long_short_decoding_analyses
        long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis'] 
        active_analyses_result = long_short_fr_indicies_analysis_results

        ## Extract variables from results object:
        # long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        
        ## Compute Rate Remapping Dataframe:
        rate_remapping_result: RateRemappingResult = compute_rate_remapping_stats(active_analyses_result, global_session.neurons.aclu_to_neuron_type_map, considerable_remapping_threshold=0.7)

        rate_remapping_df: pd.DataFrame = rate_remapping_result.rr_df
        # high_remapping_cells_only = deepcopy(rate_remapping_df.high_remapping_cells_only)
        # high_remapping_cells_only = rate_remapping_df[rate_remapping_df['has_considerable_remapping']]

        # Add to computed results:
        # global_computation_results.computed_data['long_short_rate_remapping'] = ComputedResult(is_global=True, rr_df=rate_remapping_df, high_only_rr_df=high_remapping_cells_only)
        global_computation_results.computed_data['long_short_rate_remapping'] = rate_remapping_result


        """ Getting outputs:
        
            ## long_short_rate_remapping:
            curr_long_short_rr = curr_active_pipeline.global_computation_results.computed_data['long_short_rate_remapping']
            ## Extract variables from results object:
            rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df

        """
        return global_computation_results

    
    @function_attributes(short_name='short_long_pf_overlap_analyses',  tags=['overlap', 'pf'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-05 11:10', related_items=[], 
                         requires_global_keys=[], provides_global_keys=['short_long_pf_overlap_analyses'],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_scalars_df'], curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_dict']), is_global=True)
    def _perform_long_short_pf_overlap_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Computes multiple forms of overlap between the short and the long placefields
        
        Requires:
            ['sess']
            
        Provides:
            global_computation_results.computed_data['short_long_pf_overlap_analyses']
                ['short_long_pf_overlap_analyses']['short_long_neurons_diff']
                ['short_long_pf_overlap_analyses']['poly_overlap_df']
                
        Usage:
        
            ## Unwrapping `short_long_pf_overlap_analyses`:
            short_long_pf_overlap_analyses: DynamicParameters = curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']
            short_long_neurons_diff = short_long_pf_overlap_analyses['short_long_neurons_diff']
            poly_overlap_df = short_long_pf_overlap_analyses['poly_overlap_df']
            conv_overlap_dict = short_long_pf_overlap_analyses['conv_overlap_dict']
            conv_overlap_scalars_df = short_long_pf_overlap_analyses['conv_overlap_scalars_df']
            product_overlap_dict = short_long_pf_overlap_analyses['product_overlap_dict']
            product_overlap_scalars_df = short_long_pf_overlap_analyses['product_overlap_scalars_df']
            relative_entropy_overlap_dict = short_long_pf_overlap_analyses['relative_entropy_overlap_dict']
            relative_entropy_overlap_scalars_df = short_long_pf_overlap_analyses['relative_entropy_overlap_scalars_df']

        
        """
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        # Epoch dataframe stuff:
        long_epoch_name = include_includelist[0] # 'maze1_PYR'
        short_epoch_name = include_includelist[1] # 'maze2_PYR'
        if len(include_includelist) > 2:
            global_epoch_name = include_includelist[-1] # 'maze_PYR'
        else:
            print(f'WARNING: no global_epoch detected.')
            global_epoch_name = '' # None

        if debug_print:
            print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')

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


    @function_attributes(short_name='long_short_fr_indicies_analyses', tags=['short_long','firing_rate', 'computation','laps','replays'], input_requires=['laps', 'replays', 'sess.replay'], output_provides=['long_short_fr_indicies_analysis'], uses=['pipeline_complete_compute_long_short_fr_indicies'], used_by=[], creation_date='2023-04-11 00:00', 
                         requires_global_keys=[], provides_global_keys=['long_short_fr_indicies_analysis'],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis'], curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']['long_short_fr_indicies_df']), is_global=True)
    def _perform_long_short_firing_rate_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Computes the firing rate indicies which is a measure of the changes in firing rate (rate-remapping) between the long and the short track
        
        Requires:
            ['sess']
            
        Provides:
            global_computation_results.computed_data['long_short_fr_indicies_analysis']
                ['long_short_fr_indicies_analysis']['active_context']
                ['long_short_fr_indicies_analysis']['non_replays_frs_index']
                ['long_short_fr_indicies_analysis']['long_short_fr_indicies_df']
        
        """
        # New unified `pipeline_complete_compute_long_short_fr_indicies(...)` method for entire pipeline:
        active_context, all_results_dict = pipeline_complete_compute_long_short_fr_indicies(owning_pipeline_reference) # use the all_results_dict as the computed data value
        global_computation_results.computed_data['long_short_fr_indicies_analysis'] = DynamicParameters.init_from_dict({**all_results_dict, 'active_context': active_context})
        return global_computation_results


    @function_attributes(short_name='jonathan_firing_rate_analysis', tags=['replay'], input_requires=['_perform_long_short_firing_rate_analyses'],
                          requires_global_keys=['long_short_fr_indicies_analysis'], provides_global_keys=['jonathan_firing_rate_analysis'],
                          validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df, curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df['is_refined_exclusive']), is_global=True)
    def _perform_jonathan_replay_firing_rate_analyses(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Ported from Jonathan's `Gould_22-09-29.ipynb` Notebook
        
        Requires:
            ['sess']
            
        Provides:
            computation_result.computed_data['jonathan_firing_rate_analysis']
                ['jonathan_firing_rate_analysis']['rdf']:
                    ['jonathan_firing_rate_analysis']['rdf']['rdf']
                    ['jonathan_firing_rate_analysis'].rdf.aclu_to_idx
                    
                ['jonathan_firing_rate_analysis']['irdf']:
                    ['jonathan_firing_rate_analysis'].irdf.irdf
                    ['jonathan_firing_rate_analysis']['irdf']['aclu_to_idx']

                ['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate:
                    ['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_bins']
                    ['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_binned_unit_specific_binned_spike_rate']

                ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']:
                    ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
                    ['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']

                ['jonathan_firing_rate_analysis'].neuron_replay_stats_df
        
        """
        def _subfn_compute_custom_PBEs(sess):
            """ 
                new_pbe_epochs = _compute_custom_PBEs(sess)
            """
            print('computing PBE epochs for session...\n')
            raise NotImplementedError # this should not happen, we should use the valid sess.replay
            # kamrans_new_parameters = DynamicParameters(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.6) # 2023-10-05 Kamran's imposed Parameters, wants to remove the effect of the max_dur which was previously at 0.300
            # new_pbe_epochs = sess.compute_pbe_epochs(sess, active_parameters=kamrans_new_parameters) # NewPaper's Parameters # , **({'thresh': (0, 1.5), 'min_dur': 0.03, 'merge_dur': 0.1, 'max_dur': 0.3} | kwargs)
            # return new_pbe_epochs

        # BEGIN MAIN FUNCTION:
        replays_df = None
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        # Epoch dataframe stuff:
        # long_epoch_name = include_includelist[0] # 'maze1_PYR'
        # short_epoch_name = include_includelist[1] # 'maze2_PYR'
        # if len(include_includelist) > 2:
        #     global_epoch_name = include_includelist[-1] # 'maze_PYR'
        # else:
        #     print(f'WARNING: no global_epoch detected.')
        #     global_epoch_name = '' # None
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        # t_split = sess.paradigm[0][0,1] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

        if debug_print:
            print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')
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
            new_pbe_epochs = _subfn_compute_custom_PBEs(sess)
            sess.pbe = new_pbe_epochs # copy the detected PBEs to the session
            replays_df = sess.pbe.to_dataframe()
            # replays_df = sess.ripple.to_dataframe()
        except BaseException as e:
            raise e
        # else:
        #     replays_df = replays_df.copy() # make a copy of the provided df

        replays_df = ensure_dataframe(replays_df)
        
        rdf, aclu_to_idx, irdf, aclu_to_idx_irdf = _final_compute_jonathan_replay_fr_analyses(sess, replays_df, t_start=t_start, t_delta=t_delta, t_end=t_end)
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

        final_jonathan_df: pd.DataFrame = _subfn_computations_make_jonathan_firing_comparison_df(time_binned_unit_specific_binned_spike_rate, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf)
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


        final_jonathan_df['neuron_type'] = [sess.neurons.aclu_to_neuron_type_map[aclu] for aclu in final_jonathan_df.index.to_numpy()]
        

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

        # Convert to explicit `JonathanFiringRateAnalysisResult` object:
        jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**global_computation_results.computed_data['jonathan_firing_rate_analysis'].to_dict())
        global_computation_results.computed_data['jonathan_firing_rate_analysis'] = jonathan_firing_rate_analysis_result # set the actual result object
        

        ## Refine the LxC/SxC designators using the firing rate index metric:
        frs_index_inclusion_magnitude:float = 0.5

        ## Get global `long_short_fr_indicies_analysis`:
        long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
        long_short_fr_indicies_df = long_short_fr_indicies_analysis_results['long_short_fr_indicies_df']
        jonathan_firing_rate_analysis_result.refine_exclusivity_by_inst_frs_index(long_short_fr_indicies_df, frs_index_inclusion_magnitude=frs_index_inclusion_magnitude)

        neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=frs_index_inclusion_magnitude)
        ## Update long_exclusive/short_exclusive properties with `long_short_fr_indicies_df`
        # long_exclusive.refine_exclusivity_by_inst_frs_index(long_short_fr_indicies_df, frs_index_inclusion_magnitude=0.5)
        # short_exclusive.refine_exclusivity_by_inst_frs_index(long_short_fr_indicies_df, frs_index_inclusion_magnitude=0.5)

        return global_computation_results


    @function_attributes(short_name='long_short_post_decoding', tags=['long_short', 'short_long','replay', 'decoding', 'computation', 'radon_transforms', 'expected_v_observed'], input_requires=['global_computation_results.computed_data.long_short_leave_one_out_decoding_analysis', 'global_computation_results.computed_data.long_short_fr_indicies_analysis'], output_provides=[],
                          uses=['compute_rate_remapping_stats', 'compute_measured_vs_expected_firing_rates', 'simpler_compute_measured_vs_expected_firing_rates', 'compute_radon_transforms'], used_by=[], creation_date='2023-05-31 13:57',
                          requires_global_keys=['long_short_fr_indicies_analysis', 'long_short_leave_one_out_decoding_analysis'], provides_global_keys=['long_short_post_decoding'],
                          validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding'].rate_remapping.rr_df, is_global=True)
    def _perform_long_short_post_decoding_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """ Must be performed after `_perform_long_short_decoding_analyses` and `_perform_long_short_firing_rate_analyses`
        
        Currently an amalgamation of a bunch of computations that make use of the previous global decoding results to add features like:
            - measured vs. expected firing rates
            - linear fits to replay events via the Radon transform
            - rate remapping properties.
        
        Requires:
            ['global_computation_results.computed_data.long_short_leave_one_out_decoding_analysis', 'global_computation_results.computed_data.long_short_fr_indicies_analysis']
            
        Provides:
            computation_result.computed_data['long_short_rate_remapping']
                # ['long_short_rate_remapping']['rr_df']
                # ['long_short_rate_remapping']['high_only_rr_df']
        
        """
        import scipy.stats
        from neuropy.core.neurons import NeuronType
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_rate_remapping_stats
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_measured_vs_expected_firing_rates
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import simpler_compute_measured_vs_expected_firing_rates
        
        ## long_short_decoding_analyses:
        long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
        # x_frs_dict, y_frs_dict = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
        curr_long_short_decoding_analyses = global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'] 


        ## Extract variables from results object:
        long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        
        ## Common to both Long and Short:
        active_pos_df = global_session.position.to_dataframe()
        assert (long_results_obj.active_filter_epochs.as_array() == short_results_obj.active_filter_epochs.as_array()).all() # ensure that the active_filter_epochs for both are the same.
        active_filter_epochs = long_results_obj.active_filter_epochs
        num_epochs = active_filter_epochs.n_epochs

        ## Long Specific:
        decoder_1D_LONG = long_results_obj.original_1D_decoder
        decoder_result_LONG = long_results_obj.all_included_filter_epochs_decoder_result
        # call `compute_measured_vs_expected_firing_rates`
        decoder_time_bin_centers_LONG, all_epochs_computed_expected_cell_num_spikes_LONG, all_epochs_computed_observed_from_expected_difference_LONG, measured_pos_window_centers_LONG, (all_epochs_decoded_epoch_time_bins_mean_LONG, all_epochs_computed_expected_cell_firing_rates_mean_LONG, all_epochs_computed_expected_cell_firing_rates_stddev_LONG, all_epochs_computed_observed_from_expected_difference_maximum_LONG) = compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, decoder_1D_LONG, decoder_result_LONG)
        # The Flat_* versions exist because it's difficult to plot while all of the metrics are separated in lists.
        Flat_decoder_time_bin_centers_LONG = np.concatenate(decoder_time_bin_centers_LONG) # .shape: (412,)
        Flat_all_epochs_computed_expected_cell_num_spikes_LONG = np.vstack([np.concatenate([all_epochs_computed_observed_from_expected_difference_LONG[decoded_epoch_idx][target_neuron_IDX, :] for decoded_epoch_idx in np.arange(decoder_result_LONG.num_filter_epochs)]) for target_neuron_IDX in decoder_1D_LONG.neuron_IDXs]) #.shape (20, 22)

        # call `simpler_compute_measured_vs_expected_firing_rates`
        returned_shape_tuple_LONG, (observed_from_expected_diff_max_LONG, observed_from_expected_diff_ptp_LONG, observed_from_expected_diff_mean_LONG, observed_from_expected_diff_std_LONG) = simpler_compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, a_decoder_1D=decoder_1D_LONG, a_decoder_result=decoder_result_LONG)
        

        ## Short Specific:
        decoder_1D_SHORT = short_results_obj.original_1D_decoder
        decoder_result_SHORT = short_results_obj.all_included_filter_epochs_decoder_result

        # call `compute_measured_vs_expected_firing_rates`
        decoder_time_bin_centers_SHORT, all_epochs_computed_expected_cell_num_spikes_SHORT, all_epochs_computed_observed_from_expected_difference_SHORT, measured_pos_window_centers_SHORT, (all_epochs_decoded_epoch_time_bins_mean_SHORT, all_epochs_computed_expected_cell_firing_rates_mean_SHORT, all_epochs_computed_expected_cell_firing_rates_stddev_SHORT, all_epochs_computed_observed_from_expected_difference_maximum_SHORT) = compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, decoder_1D_SHORT, decoder_result_SHORT)
        # The Flat_* versions exist because it's difficult to plot while all of the metrics are separated in lists.
        Flat_decoder_time_bin_centers_SHORT = np.concatenate(decoder_time_bin_centers_SHORT) # .shape: (412,)
        Flat_all_epochs_computed_expected_cell_num_spikes_SHORT = np.vstack([np.concatenate([all_epochs_computed_observed_from_expected_difference_SHORT[decoded_epoch_idx][target_neuron_IDX, :] for decoded_epoch_idx in np.arange(decoder_result_SHORT.num_filter_epochs)]) for target_neuron_IDX in decoder_1D_SHORT.neuron_IDXs]) #.shape (20, 22)
        Flat_epoch_time_bins_mean = all_epochs_decoded_epoch_time_bins_mean_SHORT[:,0]

        # call `simpler_compute_measured_vs_expected_firing_rates`
        returned_shape_tuple_SHORT, (observed_from_expected_diff_max_SHORT, observed_from_expected_diff_ptp_SHORT, observed_from_expected_diff_mean_SHORT, observed_from_expected_diff_std_SHORT) = simpler_compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, a_decoder_1D=decoder_1D_SHORT, a_decoder_result=decoder_result_SHORT)

        ## Sanity Checks that the LONG and SHORT decoders and their decoded results aren't identical:
        assert (Flat_decoder_time_bin_centers_SHORT == Flat_decoder_time_bin_centers_LONG).all()
        Flat_decoder_time_bin_centers = Flat_decoder_time_bin_centers_LONG # equivalent
        assert (decoder_1D_LONG.neuron_IDs == decoder_1D_SHORT.neuron_IDs).all()
        assert not (decoder_1D_LONG.P_x == decoder_1D_SHORT.P_x).all() # the occupancies shouldn't be identical between the two encoders, this might indicate an error
        assert not (decoder_1D_LONG.F == decoder_1D_SHORT.F).all() # the placefields shouldn't be identical between the two encoders, this might indicate an error
        if debug_print:
            print(f"returned_shape_tuple_LONG: {returned_shape_tuple_LONG}, returned_shape_tuple_SHORT: {returned_shape_tuple_SHORT}")
        assert (returned_shape_tuple_LONG[0] == returned_shape_tuple_SHORT[0]) and (returned_shape_tuple_LONG[-1] == returned_shape_tuple_SHORT[-1]) , f"returned_shape_tuple_LONG: {returned_shape_tuple_LONG}, returned_shape_tuple_SHORT: {returned_shape_tuple_SHORT}"
        num_neurons, num_timebins_in_epoch, num_total_flat_timebins = returned_shape_tuple_LONG # after the assert they're guaranteed to be the same for short
        # assert not np.array([(Flat_all_epochs_computed_expected_cell_num_spikes_LONG[i] == Flat_all_epochs_computed_expected_cell_num_spikes_SHORT[i]).all() for i in np.arange(active_filter_epochs.n_epochs)]).all(), "all expected number spikes for all cells should not be the same for two different decoders! This likely indicates an error!"

        
        ## Outputs: Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins


        ## More complex outputs:

        # 2023-05-30 9:30pm - just compute the aggregate stats for the short v long expected firing rates: ___________________ #
        ## Get the epochs that occur on the short and the long tracks:
        is_short_track_epoch = (Flat_epoch_time_bins_mean >= short_session.t_start)
        is_long_track_epoch = np.logical_not(is_short_track_epoch)

        # All comparisons should be (native - alternative), so a positive value (> 0) is expected
        # analyze the decodings of the events on the short track:
        short_short_diff = (observed_from_expected_diff_mean_SHORT[:, is_short_track_epoch] - observed_from_expected_diff_mean_LONG[:, is_short_track_epoch])
        # expect that the short_short decoding is better than the long_short decoding, so short_short_diff should be significantly < 0.0

        # analyze the decodings of the events on the long track:
        long_long_diff = (observed_from_expected_diff_mean_LONG[:, is_long_track_epoch] - observed_from_expected_diff_mean_SHORT[:, is_long_track_epoch])

        # np.mean(long_long_diff)
        # np.mean(short_short_diff)

        # ttest_results = scipy.stats.ttest_rel(long_long_diff, short_short_diff)
        if debug_print:
            print(f'long_test: {np.mean(long_long_diff)}')
            print(f'short_test: {np.mean(short_short_diff)}')


        ### 2023-05-25 - Radon Transform approach to finding line of best fit for each replay Epoch. Modifies the original: `long_results_obj`, `short_results_obj`, `curr_long_short_decoding_analyses`
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import compute_radon_transforms

        ## 2023-05-25 - Get the 1D Posteriors for each replay epoch so they can be analyzed via score_posterior(...) with a Radon Transform approch to find the line of best fit (which gives the velocity).
        epochs_linear_fit_df_LONG, *extra_outputs = compute_radon_transforms(long_results_obj.original_1D_decoder, long_results_obj.all_included_filter_epochs_decoder_result)
        assert long_results_obj.active_filter_epochs.n_epochs == np.shape(epochs_linear_fit_df_LONG)[0]
        long_results_obj.active_filter_epochs._df.drop(columns=['score', 'velocity', 'intercept', 'speed'], inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
        long_results_obj.active_filter_epochs._df = long_results_obj.active_filter_epochs.to_dataframe().join(epochs_linear_fit_df_LONG) # add the newly computed columns to the Epochs object
        # epochs_linear_fit_df_LONG

        epochs_linear_fit_df_SHORT, *extra_outputs = compute_radon_transforms(short_results_obj.original_1D_decoder, short_results_obj.all_included_filter_epochs_decoder_result)
        assert short_results_obj.active_filter_epochs.n_epochs == np.shape(epochs_linear_fit_df_SHORT)[0]
        short_results_obj.active_filter_epochs._df.drop(columns=['score', 'velocity', 'intercept', 'speed'], inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
        short_results_obj.active_filter_epochs._df = short_results_obj.active_filter_epochs.to_dataframe().join(epochs_linear_fit_df_SHORT) # add the newly computed columns to the Epochs object
        # epochs_linear_fit_df_SHORT

        # Join the columns of LONG and SHORT dataframes with suffixes
        assert np.shape(epochs_linear_fit_df_LONG)[0] == np.shape(epochs_linear_fit_df_SHORT)[0]
        combined_epochs_linear_fit_df = epochs_linear_fit_df_LONG.join(epochs_linear_fit_df_SHORT, lsuffix='_LONG', rsuffix='_SHORT')
        # combined_epochs_linear_fit_df

        assert active_filter_epochs.n_epochs == np.shape(combined_epochs_linear_fit_df)[0]
        active_filter_epochs._df.drop(columns=['score_LONG', 'velocity_LONG', 'intercept_LONG', 'speed_LONG', 'score_SHORT', 'velocity_SHORT', 'intercept_SHORT', 'speed_SHORT'], inplace=True, errors='ignore') # 'ignore' doesn't raise an exception if the columns don't already exist.
        active_filter_epochs._df = active_filter_epochs.to_dataframe().join(combined_epochs_linear_fit_df) # add the newly computed columns to the Epochs object

        
        # Add to computed results:
        expected_v_observed_result = ExpectedVsObservedResult(Flat_epoch_time_bins_mean=Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers=Flat_decoder_time_bin_centers, num_neurons=num_neurons, num_timebins_in_epoch=num_timebins_in_epoch, num_total_flat_timebins=num_total_flat_timebins, 
                                                    decoder_time_bin_centers_LONG=decoder_time_bin_centers_LONG, all_epochs_computed_expected_cell_num_spikes_LONG=all_epochs_computed_expected_cell_num_spikes_LONG, all_epochs_computed_observed_from_expected_difference_LONG=all_epochs_computed_observed_from_expected_difference_LONG, measured_pos_window_centers_LONG=measured_pos_window_centers_LONG, all_epochs_decoded_epoch_time_bins_mean_LONG=all_epochs_decoded_epoch_time_bins_mean_LONG, all_epochs_computed_expected_cell_firing_rates_mean_LONG=all_epochs_computed_expected_cell_firing_rates_mean_LONG, all_epochs_computed_expected_cell_firing_rates_stddev_LONG=all_epochs_computed_expected_cell_firing_rates_stddev_LONG, all_epochs_computed_observed_from_expected_difference_maximum_LONG=all_epochs_computed_observed_from_expected_difference_maximum_LONG, Flat_decoder_time_bin_centers_LONG=Flat_decoder_time_bin_centers_LONG, Flat_all_epochs_computed_expected_cell_num_spikes_LONG=Flat_all_epochs_computed_expected_cell_num_spikes_LONG, returned_shape_tuple_LONG=returned_shape_tuple_LONG, observed_from_expected_diff_ptp_LONG=observed_from_expected_diff_ptp_LONG, observed_from_expected_diff_mean_LONG=observed_from_expected_diff_mean_LONG, observed_from_expected_diff_std_LONG=observed_from_expected_diff_std_LONG,
                                                    decoder_time_bin_centers_SHORT=decoder_time_bin_centers_SHORT, all_epochs_computed_expected_cell_num_spikes_SHORT=all_epochs_computed_expected_cell_num_spikes_SHORT, all_epochs_computed_observed_from_expected_difference_SHORT=all_epochs_computed_observed_from_expected_difference_SHORT, measured_pos_window_centers_SHORT=measured_pos_window_centers_SHORT, all_epochs_decoded_epoch_time_bins_mean_SHORT=all_epochs_decoded_epoch_time_bins_mean_SHORT, all_epochs_computed_expected_cell_firing_rates_mean_SHORT=all_epochs_computed_expected_cell_firing_rates_mean_SHORT, all_epochs_computed_expected_cell_firing_rates_stddev_SHORT=all_epochs_computed_expected_cell_firing_rates_stddev_SHORT, all_epochs_computed_observed_from_expected_difference_maximum_SHORT=all_epochs_computed_observed_from_expected_difference_maximum_SHORT, Flat_decoder_time_bin_centers_SHORT=Flat_decoder_time_bin_centers_SHORT, Flat_all_epochs_computed_expected_cell_num_spikes_SHORT=Flat_all_epochs_computed_expected_cell_num_spikes_SHORT, returned_shape_tuple_SHORT=returned_shape_tuple_SHORT, observed_from_expected_diff_ptp_SHORT=observed_from_expected_diff_ptp_SHORT, observed_from_expected_diff_mean_SHORT=observed_from_expected_diff_mean_SHORT, observed_from_expected_diff_std_SHORT=observed_from_expected_diff_std_SHORT,
                                                    is_short_track_epoch=is_short_track_epoch, is_long_track_epoch=is_long_track_epoch, short_short_diff=short_short_diff, long_long_diff=long_long_diff,
        )


        # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins
        
        ## Compute Rate Remapping Dataframe:
        rate_remapping_result: RateRemappingResult = compute_rate_remapping_stats(long_short_fr_indicies_analysis_results, global_session.neurons.aclu_to_neuron_type_map, considerable_remapping_threshold=0.7)
        # rr_df: pd.DataFrame = rate_remapping_result.rr_df
        # high_remapping_cells_only = rate_remapping_result.high_only_rr_df
       
        
        # Add to computed results:
        global_computation_results.computed_data['long_short_post_decoding'] = DynamicParameters(expected_v_observed_result=expected_v_observed_result,
                                                                                               rate_remapping=rate_remapping_result
                                                                                            )
        
        """ Getting outputs:
        
            ## long_short_post_decoding:
            curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
            ## Extract variables from results object:
            expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
            rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
            Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff

        """
        return global_computation_results


    # InstantaneousSpikeRateGroupsComputation
    @function_attributes(short_name='long_short_inst_spike_rate_groups', tags=['long_short', 'LxC', 'SxC', 'Figure2','replay', 'decoding', 'computation'], input_requires=['global_computation_results.computed_data.jonathan_firing_rate_analysis', 'global_computation_results.computed_data.long_short_fr_indicies_analysis'], output_provides=['global_computation_results.computed_data.long_short_endcap'], uses=[], used_by=[], creation_date='2023-08-21 16:52', related_items=[],
        requires_global_keys=['jonathan_firing_rate_analysis', 'long_short_fr_indicies_analysis'], provides_global_keys=['long_short_inst_spike_rate_groups'],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': curr_active_pipeline.global_computation_results.computed_data['long_short_inst_spike_rate_groups'], is_global=True)
    def _perform_long_short_instantaneous_spike_rate_groups_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, instantaneous_time_bin_size_seconds: Optional[float]=0.01):
        """ Must be performed after `_perform_jonathan_replay_firing_rate_analyses`
        
        Factoring out of `InstantaneousSpikeRateGroupsComputation`
        
    
        Requires:
            ['global_computation_results.computed_data.jonathan_firing_rate_analysis', 'global_computation_results.computed_data.long_short_fr_indicies_analysis']
            
        Provides:
            global_computation_results.computed_data['long_short_inst_spike_rate_groups']
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends # for `_perform_long_short_instantaneous_spike_rate_groups_analysis`
        from neuropy.utils.dynamic_container import DynamicContainer


        if instantaneous_time_bin_size_seconds is None:
            instantaneous_time_bin_size_seconds = 0.01 # set to default

        # if global_computation_results.computation_config is None:
        #     # Create a DynamicContainer-backed computation_config
        #     print(f'_perform_long_short_instantaneous_spike_rate_groups_analysis is lacking a required computation config parameter! creating a new curr_active_pipeline.global_computation_results.computation_config')
        #     global_computation_results.computation_config.long_short_inst_spike_rate_groups.instantaneous_time_bin_size_seconds = 0.01
        # else:
        #     print(f'have an existing `global_computation_results.computation_config`: {global_computation_results.computation_config}')	
            

        # Could also use `owning_pipeline_reference.global_computation_results.computation_config`
        assert (global_computation_results.computation_config is not None), f"requires `global_computation_results.computation_config.long_short_inst_spike_rate_groups`"
        assert (global_computation_results.computation_config.get('long_short_inst_spike_rate_groups', None) is not None), f"requires `global_computation_results.computation_config.long_short_inst_spike_rate_groups`"

        global_computation_results.computation_config.long_short_inst_spike_rate_groups.instantaneous_time_bin_size_seconds = instantaneous_time_bin_size_seconds # 0.01 # 10ms
        
        ## INPUTS: instantaneous_time_bin_size_seconds
        sess = owning_pipeline_reference.sess 
        # Get the provided context or use the session context:
        active_context = sess.get_context()

        # Build the output result:
        inst_spike_rate_groups_result: InstantaneousSpikeRateGroupsComputation = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds)
        inst_spike_rate_groups_result.compute(owning_pipeline_reference)
        
        # inst_spike_rate_groups_result.Fig2_Laps_FR, inst_spike_rate_groups_result.Fig2_Replay_FR can only be done at the end probably

        # Find instantaneous firing rate for spikes outside of replays

        # Add to computed results:
        global_computation_results.computed_data['long_short_inst_spike_rate_groups'] = inst_spike_rate_groups_result

        """ Getting outputs:
        
            ## long_short_post_decoding:
            inst_spike_rate_groups_result: InstantaneousSpikeRateGroupsComputation = curr_active_pipeline.global_computation_results.computed_data.long_short_inst_spike_rate_groups
            inst_spike_rate_groups_result
            

        """
        return global_computation_results


    @function_attributes(short_name='long_short_endcap_analysis', tags=['long_short_endcap_analysis', 'endcap'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-15 10:37', related_items=[],
                         requires_global_keys=['jonathan_firing_rate_analysis'], provides_global_keys=['long_short_endcap'],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.global_computation_results.computed_data['long_short_endcap']), is_global=True)
    def _perform_long_short_endcap_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """  2023-09-14 - Find cells outside the bounds of the short track

        Must be performed after `_perform_jonathan_replay_firing_rate_analyses`
        
        Requires:
            ['global_computation_results.computed_data.jonathan_firing_rate_analysis', 'global_computation_results.computed_data.long_short_fr_indicies_analysis']
            
        Provides:
            global_computation_results.computed_data['long_short_endcap']
        
        """
        
        loaded_track_limits = deepcopy(owning_pipeline_reference.sess.config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
        x_midpoint: float = owning_pipeline_reference.sess.config.x_midpoint
        pix2cm: float = owning_pipeline_reference.sess.config.pix2cm

        ## INPUTS: loaded_track_limits
        print(f'loaded_track_limits: {loaded_track_limits}')


        long_xlim = loaded_track_limits['long_xlim']
        # long_ylim = loaded_track_limits['long_ylim']
        short_xlim = loaded_track_limits['short_xlim']
        # short_ylim = loaded_track_limits['short_ylim']

        occupancy_midpoint: float = x_midpoint # 142.7512402496278 # 150.0
        left_cap_x_bound: float = (long_xlim[0] - x_midpoint) #-shift by midpoint - 72.0 # on long track
        right_cap_x_bound: float = (long_xlim[1] - x_midpoint) # 72.0 # on long track
        min_significant_remapping_x_distance: float = 40.0 # from long->short track


        ## STATIC:
        # occupancy_midpoint: float = 142.7512402496278 # 150.0
        # left_cap_x_bound: float = -72.0 # on long track
        # right_cap_x_bound: float = 72.0 # on long track
        # min_significant_remapping_x_distance: float = 40.0 # from long->short track


        jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = global_computation_results.computed_data.jonathan_firing_rate_analysis

        # Modifies the `jonathan_firing_rate_analysis_result.neuron_replay_stats_df` in-place instead of creating a copy:
        # neuron_replay_stats_df = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
        neuron_replay_stats_df = jonathan_firing_rate_analysis_result.neuron_replay_stats_df
        # Extract the peaks of the long placefields to find ones that have peaks outside the boundaries
        long_pf_peaks = neuron_replay_stats_df[neuron_replay_stats_df['has_long_pf']]['long_pf_peak_x'] - occupancy_midpoint # this shift of `occupancy_midpoint` is to center the midpoint of the track at 0. 
        is_left_cap = (long_pf_peaks < left_cap_x_bound)
        is_right_cap = (long_pf_peaks > right_cap_x_bound)
        # is_either_cap =  np.logical_or(is_left_cap, is_right_cap)

        # Adds ['is_long_peak_left_cap', 'is_long_peak_right_cap', 'is_long_peak_either_cap'] columns: 
        neuron_replay_stats_df['is_long_peak_left_cap'] = False
        neuron_replay_stats_df['is_long_peak_right_cap'] = False
        neuron_replay_stats_df.loc[is_left_cap.index, 'is_long_peak_left_cap'] = is_left_cap # True
        neuron_replay_stats_df.loc[is_right_cap.index, 'is_long_peak_right_cap'] = is_right_cap # True

        neuron_replay_stats_df['is_long_peak_either_cap'] = np.logical_or(neuron_replay_stats_df['is_long_peak_left_cap'], neuron_replay_stats_df['is_long_peak_right_cap'])

        # adds ['LS_pf_peak_x_diff'] column
        neuron_replay_stats_df['LS_pf_peak_x_diff'] = neuron_replay_stats_df['long_pf_peak_x'] - neuron_replay_stats_df['short_pf_peak_x']

        cap_cells_df = neuron_replay_stats_df[np.logical_and(neuron_replay_stats_df['has_long_pf'], neuron_replay_stats_df['is_long_peak_either_cap'])]
        num_total_endcap_cells = len(cap_cells_df)

        # "Disppearing" cells fall below the 1Hz firing criteria on the short track:
        disappearing_endcap_cells_df = cap_cells_df[np.logical_not(cap_cells_df['has_short_pf'])]
        num_disappearing_endcap_cells = len(disappearing_endcap_cells_df)
        print(f'num_disappearing_endcap_cells/num_total_endcap_cells: {num_disappearing_endcap_cells}/{num_total_endcap_cells}')

        non_disappearing_endcap_cells_df = cap_cells_df[cap_cells_df['has_short_pf']] # "non_disappearing" cells are those with a placefield on the short track as well
        num_non_disappearing_endcap_cells = len(non_disappearing_endcap_cells_df)
        print(f'num_non_disappearing_endcap_cells/num_total_endcap_cells: {num_non_disappearing_endcap_cells}/{num_total_endcap_cells}')

        # display(non_disappearing_endcap_cells_df)
        # non_disappearing_endcap_cells_df['LS_pf_peak_x_diff'] = non_disappearing_endcap_cells_df['long_pf_peak_x'] - non_disappearing_endcap_cells_df['short_pf_peak_x']
        # display(non_disappearing_endcap_cells_df)

        # Classify the non_disappearing cells into two groups:
        # 1. Those that exhibit significant remapping onto somewhere else on the track
        non_disappearing_endcap_cells_df['has_significant_distance_remapping'] = (np.abs(non_disappearing_endcap_cells_df['LS_pf_peak_x_diff']) >= min_significant_remapping_x_distance) # The most a placefield could translate intwards would be (35 + (pf_width/2.0)) I think.
        num_significant_position_remappping_endcap_cells = len(non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == True])
        print(f'num_significant_position_remappping_endcap_cells/num_non_disappearing_endcap_cells: {num_significant_position_remappping_endcap_cells}/{num_non_disappearing_endcap_cells}')

        # 2. Those that seem to remain where they were on the long track, perhaps being "sampling-clipped" or translated adjacent to the platform. These two subcases can be distinguished by a change in the placefield's length (truncated cells would be a fraction of the length, although might need to account for scaling with new track length)
        minorly_changed_endcap_cells_df = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == False]

        significant_distant_remapping_endcap_aclus = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping']].index # Int64Index([3, 5, 7, 11, 14, 38, 41, 53, 57, 61, 62, 75, 78, 79, 82, 83, 85, 95, 98, 100, 102], dtype='int64')
        
        # make sure the result is copied back to the global_computations
        global_computation_results.computed_data.jonathan_firing_rate_analysis.neuron_replay_stats_df = neuron_replay_stats_df # make sure the result is copied back to the global_computations
        
        # # Add to computed results:
        global_computation_results.computed_data['long_short_endcap'] = TruncationCheckingResults(is_global=True)
        # global_computation_results.computed_data['long_short_endcap'].significant_distant_remapping_endcap_aclus = significant_distant_remapping_endcap_aclus
        # global_computation_results.computed_data['long_short_endcap'].minorly_changed_endcap_cells_df = minorly_changed_endcap_cells_df.index
        # global_computation_results.computed_data['long_short_endcap'].non_disappearing_endcap_cells_df = non_disappearing_endcap_cells_df.index
        # global_computation_results.computed_data['long_short_endcap'].disappearing_endcap_cells_df = disappearing_endcap_cells_df.index
        global_computation_results.computed_data['long_short_endcap'].significant_distant_remapping_endcap_aclus = significant_distant_remapping_endcap_aclus.copy()
        global_computation_results.computed_data['long_short_endcap'].minor_remapping_endcap_aclus = minorly_changed_endcap_cells_df.index.copy()
        global_computation_results.computed_data['long_short_endcap'].non_disappearing_endcap_aclus = non_disappearing_endcap_cells_df.index.copy()
        global_computation_results.computed_data['long_short_endcap'].disappearing_endcap_aclus = disappearing_endcap_cells_df.index.copy()

        
        """ Getting outputs:
        
            ## long_short_endcap_analysis:
            truncation_checking_result: TruncationCheckingResults = curr_active_pipeline.global_computation_results.computed_data.long_short_endcap
            truncation_checking_result
            

        """
        return global_computation_results

    


# ==================================================================================================================== #
# BEGIN PRIVATE IMPLEMENTATIONS                                                                                        #
# ==================================================================================================================== #


# ==================================================================================================================== #
# Long/Short Curve Extrapolation and Management                                                                        #
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

# ==================================================================================================================== #
# 2023-04-07 - `constrain_to_laps`                                                                                     #
#   Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.
# ==================================================================================================================== #
## 2023-04-07 - Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`

@function_attributes(short_name=None, tags=['laps', 'constrain', 'placefields'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-07 00:00', related_items=[])
def constrain_to_laps(curr_active_pipeline):
    """ 2023-04-07 - Constrains the placefields to just the laps, computing the laps if needed.
    Other laps-related things?
        # ??? pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
        # DataSession.compute_position_laps(self)
        # DataSession.compute_laps_position_df(position_df, laps_df)

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import constrain_to_laps

        curr_active_pipeline, directional_lap_specific_configs = constrain_to_laps(curr_active_pipeline)
        
        
    MUTATES:
        curr_active_pipeline.computation_results[*].computed_data.pf1D,
        curr_active_pipeline.computation_results[*].computed_data.pf2D,
        Maybe others?
    """

    lap_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
    assert lap_estimation_parameters is not None
    
    use_direction_dependent_laps: bool = lap_estimation_parameters.get('use_direction_dependent_laps', True)
    print(f'constrain_to_laps(...): use_direction_dependent_laps: {use_direction_dependent_laps}')

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

    for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):
        # replace laps with estimates:
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
            curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs = deepcopy(curr_laps_obj) # TODO: does this change the config that's used for computations? I think it should. 
            
            # Get existing placefields:
            curr_pf1D, curr_pf2D = a_result.pf1D, a_result.pf2D

            lap_filtered_curr_pf1D = deepcopy(curr_pf1D)
            lap_filtered_curr_pf1D = PfND(spikes_df=lap_filtered_curr_pf1D.spikes_df, position=lap_filtered_curr_pf1D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf1D.config, compute_on_init=True)
            lap_filtered_curr_pf2D = deepcopy(curr_pf2D)
            lap_filtered_curr_pf2D = PfND(spikes_df=lap_filtered_curr_pf2D.spikes_df, position=lap_filtered_curr_pf2D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf2D.config, compute_on_init=True)
            # Replace the result with the lap-filtered variety. This is perminant.
            a_result.pf1D = lap_filtered_curr_pf1D
            a_result.pf2D = lap_filtered_curr_pf2D


    ## After all top-level computations are done, compute the subsets for direction laps
    directional_lap_specific_configs = {}
    if use_direction_dependent_laps:
        print(f'constrain_to_laps(...) processing for directional laps...')
        split_directional_laps_name_parts = ['odd_laps', 'even_laps', 'any_laps']
        split_directional_laps_config_names = [f'{a_name}_{a_lap_dir_description}' for a_lap_dir_description in split_directional_laps_name_parts]
        print(f'\tsplit_directional_laps_config_names: {split_directional_laps_config_names}')
        for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):
            lap_specific_epochs = a_sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0) # set this to the laps object
            any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(a_sess.laps.lap_id))])
            even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(a_sess.laps.lap_id), 2)])
            odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(a_sess.laps.lap_id), 2)])
            split_directional_laps_dict = {'odd_laps': odd_lap_specific_epochs, 'even_laps': even_lap_specific_epochs, 'any_laps': lap_specific_epochs}

            print(f'lap_specific_epochs: {lap_specific_epochs}\n\tany_lap_specific_epochs: {any_lap_specific_epochs}\n\teven_lap_specific_epochs: {even_lap_specific_epochs}\n\todd_lap_specific_epochs: {odd_lap_specific_epochs}\n')
            for a_lap_dir_description, lap_dir_epochs in split_directional_laps_dict.items():
                new_name = f'{a_name}_{a_lap_dir_description}'
                print(f'\tnew_name: {new_name}')
                active_config_copy = deepcopy(curr_active_pipeline.active_configs[a_name])
                # active_config_copy.computation_config.pf_params.computation_epochs = active_config_copy.computation_config.pf_params.computation_epochs.label_slice(odd_lap_specific_epochs.labels)
                ## Just overwrite directly:
                active_config_copy.computation_config.pf_params.computation_epochs = lap_dir_epochs
                directional_lap_specific_configs[new_name] = active_config_copy
                # curr_active_pipeline.active_configs[new_name] = active_config_copy
            # end loop over split_directional_lap types:
        # end loop over filter epochs:

        print(f'directional_lap_specific_configs: {directional_lap_specific_configs}')


    return curr_active_pipeline, directional_lap_specific_configs


@function_attributes(short_name=None, tags=['long_short', 'decoder', 'constrained', 'important'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-14 00:00', related_items=[])
def compute_long_short_constrained_decoders(curr_active_pipeline, long_epoch_name:str, short_epoch_name:str, enable_two_step_decoders:bool = False, recalculate_anyway:bool=True):
    """ 2023-04-14 - Computes both 1D & 2D Decoders constrained to each other's position bins 
    Usage:
        
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_long_short_constrained_decoders(curr_active_pipeline, long_epoch_name=long_epoch_name, short_epoch_name=short_epoch_name)

        With Two-step Decoders:
        (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D) = compute_long_short_constrained_decoders(curr_active_pipeline, long_epoch_name=long_epoch_name, short_epoch_name=short_epoch_name, enable_two_step_decoders=True)

    """
    

    # 1D Decoders constrained to each other
    def compute_short_long_constrained_decoders_1D(curr_active_pipeline, long_epoch_name:str, short_epoch_name:str, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 1D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]

        # long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
        long_one_step_decoder_1D, short_one_step_decoder_1D  = [deepcopy(results_data.get('pf1D_Decoder', None)) for results_data in (long_results, short_results)]
        # ds and Decoders conform between the long and the short epochs:
        short_one_step_decoder_1D, did_recompute = short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D, force_recompute=True)

        # ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_1D is None) or (short_two_step_decoder_1D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
                long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
                assert (long_two_step_decoder_1D is not None and short_two_step_decoder_1D is not None)
        else:
            long_two_step_decoder_1D, short_two_step_decoder_1D = None, None

        return long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D

    def compute_short_long_constrained_decoders_2D(curr_active_pipeline, long_epoch_name:str, short_epoch_name:str, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 2D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]
        # Make the 2D Placefields and Decoders conform between the long and the short epochs:
        long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
        short_one_step_decoder_2D, did_recompute = short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_2D, short_two_step_decoder_2D  = [results_data.get('pf2D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_2D is None) or (short_two_step_decoder_2D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
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
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D = compute_short_long_constrained_decoders_1D(curr_active_pipeline, long_epoch_name=long_epoch_name, short_epoch_name=short_epoch_name, enable_two_step_decoders=enable_two_step_decoders)
    long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D = compute_short_long_constrained_decoders_2D(curr_active_pipeline, long_epoch_name=long_epoch_name, short_epoch_name=short_epoch_name, enable_two_step_decoders=enable_two_step_decoders)

    if enable_two_step_decoders:
        return (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D)
    else:
        # Only return the one_step decoders
        return (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D)

# ==================================================================================================================== #
# 2023-05-10 - Long Short Decoding Analysis                                                                            #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['decoding', 'loo', 'surprise', 'replay'], input_requires=['long_session.replay', 'short_session.replay', 'global_session.replay'], output_provides=[], uses=['LeaveOneOutDecodingAnalysis', 'perform_full_session_leave_one_out_decoding_analysis'], used_by=[], creation_date='2023-09-21 17:25', related_items=[])
def _long_short_decoding_analysis_from_decoders(long_one_step_decoder_1D, short_one_step_decoder_1D, long_session, short_session, global_session, decoding_time_bin_size = 0.025, perform_cache_load=False) -> LeaveOneOutDecodingAnalysis:
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
    ## This is very verboose but the new version properties mess up the *arg attribute initializer
    # Potentially useful assingment kwargs string: `# long_decoder=long_decoder, short_decoder=short_decoder, long_replays=long_replays, short_replays=short_replays, global_replays=global_replays, long_shared_aclus_only_decoder=long_shared_aclus_only_decoder, short_shared_aclus_only_decoder=short_shared_aclus_only_decoder, shared_aclus=shared_aclus, long_short_pf_neurons_diff=long_short_pf_neurons_diff, n_neurons=n_neurons, long_results_obj=long_results_obj, short_results_obj=short_results_obj`
    leave_one_out_decoding_analysis_obj: LeaveOneOutDecodingAnalysis = LeaveOneOutDecodingAnalysis(is_global=True)
    leave_one_out_decoding_analysis_obj.long_decoder = long_decoder
    leave_one_out_decoding_analysis_obj.short_decoder = short_decoder
    leave_one_out_decoding_analysis_obj.long_replays = long_replays
    leave_one_out_decoding_analysis_obj.short_replays = short_replays
    leave_one_out_decoding_analysis_obj.global_replays = global_replays
    leave_one_out_decoding_analysis_obj.long_shared_aclus_only_decoder = long_shared_aclus_only_decoder
    leave_one_out_decoding_analysis_obj.short_shared_aclus_only_decoder = short_shared_aclus_only_decoder
    leave_one_out_decoding_analysis_obj.shared_aclus = shared_aclus
    leave_one_out_decoding_analysis_obj.long_short_pf_neurons_diff = long_short_pf_neurons_diff
    leave_one_out_decoding_analysis_obj.n_neurons = n_neurons
    leave_one_out_decoding_analysis_obj.long_results_obj = long_results_obj
    leave_one_out_decoding_analysis_obj.short_results_obj = short_results_obj
    
    # long_decoder=long_decoder, short_decoder=short_decoder, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj
    # long_decoder=long_decoder, short_decoder=short_decoder, long_replays=long_replays, short_replays=short_replays, global_replays=global_replays, long_shared_aclus_only_decoder=long_shared_aclus_only_decoder, short_shared_aclus_only_decoder=short_shared_aclus_only_decoder, shared_aclus=shared_aclus, long_short_pf_neurons_diff=long_short_pf_neurons_diff, n_neurons=n_neurons, long_results_obj=long_results_obj, short_results_obj=short_results_obj
    
    return leave_one_out_decoding_analysis_obj

# ==================================================================================================================== #
# Long Short Firing Rate Indicies                                                                                      #
# ==================================================================================================================== #
def _unwrap_aclu_epoch_values_dict_to_array(mean_epochs_all_frs):
    """ unwraps a dictionary with keys of ACLUs and values of np.arrays (vectors) """
    aclus = list(mean_epochs_all_frs.keys())
    values = np.array(list(mean_epochs_all_frs.values())) # 
    return aclus, values # values.shape # (108, 36)

def _epoch_unit_avg_firing_rates(spikes_df, filter_epochs, included_neuron_ids=None, debug_print=False):
    """Computes the average firing rate for each neuron (unit) in each epoch.

    Args:
        spikes_df (_type_): _description_
        filter_epochs (_type_): _description_
        included_neuron_ids (_type_, optional): _description_. Defaults to None.
        debug_print (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    TODO: very inefficient.

    WARNING: NaNs will emerge when the filter_epoch is of length 0.

    """
    epoch_avg_firing_rate = {}
    # .spikes.get_unit_spiketrains()
    # .spikes.get_split_by_unit(included_neuron_ids=None)
    # Add add_epochs_id_identity

    if included_neuron_ids is None:
        included_neuron_ids = spikes_df.spikes.neuron_ids

    if isinstance(filter_epochs, pd.DataFrame):
        filter_epochs_df = filter_epochs
    else:
        filter_epochs_df = filter_epochs.to_dataframe()
        
    if debug_print:
        print(f'filter_epochs: {filter_epochs.n_epochs}')
    ## Get the spikes during these epochs to attempt to decode from:
    filter_epoch_spikes_df = deepcopy(spikes_df)
    ## Add the epoch ids to each spike so we can easily filter on them:
    # filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
    if debug_print:
        print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
    # filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] != -1] # Drop all non-included spikes
    if debug_print:
        print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

    # for epoch_start, epoch_end in filter_epochs:
    for epoch_id in np.arange(np.shape(filter_epochs_df)[0]):
        epoch_start = filter_epochs_df.start.values[epoch_id]
        epoch_end = filter_epochs_df.stop.values[epoch_id]
        epoch_spikes_df = spikes_df.spikes.time_sliced(t_start=epoch_start, t_stop=epoch_end)
        # epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] == epoch_id]
        for aclu, unit_epoch_spikes_df in zip(included_neuron_ids, epoch_spikes_df.spikes.get_split_by_unit(included_neuron_ids=included_neuron_ids)):
            if aclu not in epoch_avg_firing_rate:
                epoch_avg_firing_rate[aclu] = []
            epoch_avg_firing_rate[aclu].append((float(np.shape(unit_epoch_spikes_df)[0]) / (epoch_end - epoch_start)))  #TODO 2023-06-23 11:52: - [ ] This uses the naive method of computating firing rates (num_spikes/epoch_duration) but this doesn't make sense necissarily as the cell isn't supposed to be firing for the whole epoch.

    return epoch_avg_firing_rate, {aclu:np.mean(unit_epoch_avg_frs) for aclu, unit_epoch_avg_frs in epoch_avg_firing_rate.items()}

@function_attributes(short_name='_fr_index', tags=['long_short', 'compute', 'fr_index'], input_requires=[], output_provides=[], uses=[], used_by=['_compute_long_short_firing_rate_indicies'], creation_date='2023-01-19 00:00')
def _fr_index(long_fr, short_fr):
    """ Pho's 2023 firing-rate-index [`fri`] measure."""
    return ((long_fr - short_fr) / (long_fr + short_fr))


@function_attributes(short_name=None, tags=['long_short', 'compute', 'fr_index'], input_requires=[], output_provides=[], uses=['_epoch_unit_avg_firing_rates', 'SpikeRateTrends'], used_by=['pipeline_complete_compute_long_short_fr_indicies'], creation_date='2023-09-07 19:49', related_items=[])
def _generalized_compute_long_short_firing_rate_indicies(spikes_df, instantaneous_time_bin_size_seconds: Optional[float]=None, save_path=None, **kwargs):
    """A computation for the long/short firing rate index that Kamran and I discussed as one of three metrics during our meeting on 2023-01-19.

    Args:
        spikes_df (_type_): _description_
        long_laps (_type_): _description_
        long_replays (_type_): _description_
        short_laps (_type_): _description_
        short_replays (_type_): _description_

    Returns:
        _type_: _description_


    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import _generalized_compute_long_short_firing_rate_indicies

    Aims to replace:

        x_frs_index, y_frs_index, updated_all_results_dict = _compute_long_short_firing_rate_indicies(spikes_df, long_laps, long_replays, short_laps, short_replays, save_path=temp_save_filename) # 'temp_2023-01-24_results.pkl' ## OLD VERSION

    With:
        x_frs_index, y_frs_index, updated_all_results_dict = _generalized_compute_long_short_firing_rate_indicies(spikes_df, **{'laps': (long_laps, short_laps), 'replays': (long_replays, short_replays)}, save_path=temp_save_filename) ## NEW VERSION



    """
    # long_laps, long_replays, short_laps, short_replays = args
    # kwargs={'laps': (long_laps, short_laps), 'replays': (long_replays, short_replays)}
    # if instantaneous_time_bin_size_seconds is not None:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends # for instantaneous-version analysis
    
    all_results_dict = {}
    out_frs_index_list = []
    for key, (long_epochs, short_epochs) in kwargs.items():
        print(f'_generalized_compute_long_short_firing_rate_indicies(...): processing key: "{key}"')
        assert short_epochs.n_epochs > 0, f"No short epochs for '{key}'!\t long: ({key}: {long_epochs.n_epochs > 0}), \t short: ({key}: {short_epochs.n_epochs})"
        assert long_epochs.n_epochs > 0, f"No long epochs for '{key}'!\t long: ({key}: {long_epochs.n_epochs > 0}), \t short: ({key}: {short_epochs.n_epochs})"

        ## The non-instantaneous (niaeve) firing rate computations
        long_mean_epochs_all_frs, long_mean_epochs_frs = _epoch_unit_avg_firing_rates(spikes_df, long_epochs)
        short_mean_epochs_all_frs, short_mean_epochs_frs = _epoch_unit_avg_firing_rates(spikes_df, short_epochs)
    
        all_results_dict.update(dict(zip([f'long_mean_{key}_frs', f'short_mean_{key}_frs'], [long_mean_epochs_frs, short_mean_epochs_frs]))) # all variables
        all_results_dict.update(dict(zip([f'long_mean_{key}_all_frs', f'short_mean_{key}_all_frs'], [long_mean_epochs_all_frs, short_mean_epochs_all_frs]))) # all variables

        a_frs_index = {aclu:_fr_index(long_mean_epochs_frs[aclu], short_mean_epochs_frs[aclu]) for aclu in long_mean_epochs_frs.keys()}
        all_results_dict.update(dict(zip([f'{key}_frs_index'], [a_frs_index]))) # all variables
        out_frs_index_list.append(a_frs_index)

        ## Instantaneous versions (2023-09-08) for comparison:
        if instantaneous_time_bin_size_seconds is not None:
            #TODO 2023-09-08 08:00: - [ ] Not yet fully implemented
            # raise NotImplementedError
            long_custom_InstSpikeRateTrends: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=spikes_df.copy(),
                                                                                                    filter_epochs=long_epochs,
                                                                                                    instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds)

            short_custom_InstSpikeRateTrends: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=spikes_df.copy(),
                                                                                                    filter_epochs=short_epochs,
                                                                                                    instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds)
            

            all_results_dict.update(dict(zip([f'long_mean_{key}_all_inst_frs', f'short_mean_{key}_all_inst_frs'], [long_custom_InstSpikeRateTrends.cell_agg_inst_fr_list, short_custom_InstSpikeRateTrends.cell_agg_inst_fr_list]))) # all variables
            _an_inst_fr_values = _fr_index(long_fr=long_custom_InstSpikeRateTrends.cell_agg_inst_fr_list, short_fr=short_custom_InstSpikeRateTrends.cell_agg_inst_fr_list)
            an_inst_fr_index = dict(zip(long_custom_InstSpikeRateTrends.included_neuron_ids, _an_inst_fr_values))

            # all_results_dict.update(dict(zip([f'{key}_inst_frs_index'], [an_inst_fr_index]))) # Is this update different than a direct assignment (below) is it because it only has a single key?
            all_results_dict[f'{key}_inst_frs_index'] = an_inst_fr_index
            # all_results_dict[f'replays_inst_frs_index'] = an_inst_fr_index

            # custom_InstSpikeRateTrends_df = pd.DataFrame({'aclu': long_custom_InstSpikeRateTrends.included_neuron_ids, 'long_inst_fr': long_custom_InstSpikeRateTrends.cell_agg_inst_fr_list,  'short_inst_fr': short_custom_InstSpikeRateTrends.cell_agg_inst_fr_list})
            # ,  'global_inst_fr': custom_InstSpikeRateTrends.cell_agg_inst_fr_list

            # Compute the single-dimensional firing rate index for the custom epochs and add it as a column to the dataframe:
            # custom_InstSpikeRateTrends_df['custom_frs_index'] = _fr_index(long_fr=long_custom_InstSpikeRateTrends.cell_agg_inst_fr_list, short_fr=short_custom_InstSpikeRateTrends.cell_agg_inst_fr_list)

    # long_mean_laps_all_frs, long_mean_replays_all_frs, short_mean_laps_all_frs, short_mean_replays_all_frs = [np.array(list(fr_dict.values())) for fr_dict in [long_mean_laps_all_frs, long_mean_replays_all_frs, short_mean_laps_all_frs, short_mean_replays_all_frs]]	

    # Save a backup of the data:
    if save_path is not None:
        # save_path: e.g. 'temp_2023-01-20_results.pkl'
        # backup_results_dict = dict(zip(['long_mean_laps_frs', 'long_mean_replays_frs', 'short_mean_laps_frs', 'short_mean_replays_frs', 'x_frs_index', 'y_frs_index'], [long_mean_laps_frs, long_mean_replays_frs, short_mean_laps_frs, short_mean_replays_frs, x_frs_index, y_frs_index])) # all variables
        backup_results_dict = all_results_dict # really all of the variables
        saveData(save_path, backup_results_dict)

    return *out_frs_index_list, all_results_dict



def _compute_epochs_num_aclu_inclusions(all_epochs_frs_mat, min_inclusion_fr_thresh=19.01):
    """Finds the number of unique cells that are included (as measured by their firing rate exceeding the `min_inclusion_fr_thresh`) in each epoch of interest.

    Args:
        all_epochs_frs_mat (_type_): _description_
        min_inclusion_fr_thresh (float, optional): Firing rate threshold in Hz. Defaults to 19.01.
    """
     # Hz
    is_cell_included_in_epoch_mat = all_epochs_frs_mat > min_inclusion_fr_thresh
    # is_cell_included_in_epoch_mat
    # num_cells_included_in_epoch_mat: the num unique cells included in each epoch that mean the min_inclusion_fr_thresh criteria. Should have one value per epoch of interest.
    num_cells_included_in_epoch_mat = np.sum(is_cell_included_in_epoch_mat, 0)
    # num_cells_included_in_epoch_mat
    return num_cells_included_in_epoch_mat

@function_attributes(short_name='pipeline_complete_compute_long_short_fr_indicies', tags=['long_short', 'top_level', 'compute', 'fr_index', 'laps', 'replays'], input_requires=['laps', 'replays', 'filtered_sessions[an_epoch_name].replay'], output_provides=[], uses=['_compute_long_short_firing_rate_indicies'], used_by=[], creation_date='2023-01-19 00:00')
def pipeline_complete_compute_long_short_fr_indicies(curr_active_pipeline, temp_save_filename=None):
    """ wraps `compute_long_short_firing_rate_indicies(...)` to compute the long_short_fr_index for the complete pipeline

    Requires:
        Session Laps
        If the session is missing .replay objects, uses `DataSession.compute_estimated_replay_epochs(...)` to compute them from session PBEs.

    - called in `_perform_long_short_firing_rate_analyses`

    Args:
        curr_active_pipeline (_type_): _description_
        temp_save_filename (_type_, optional): If None, disable caching the `compute_long_short_firing_rate_indicies` results. Defaults to None.

    Returns:
        _type_: _description_
        
        # all_results_dict keys:
            ['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays', 'long_non_replays', 'short_non_replays', 'global_non_replays', # epochs
            'long_mean_laps_frs', 'short_mean_laps_frs', 'long_mean_replays_frs', 'short_mean_replays_frs', 'long_mean_non_replays_frs', 'short_mean_non_replays_frs',  # raw mean firing rate variables 
            'long_mean_laps_all_frs', 'short_mean_laps_all_frs', 'long_mean_replays_all_frs', 'short_mean_replays_all_frs', 'long_mean_non_replays_all_frs', 'short_mean_non_replays_all_frs', # each epoch array of firing rates
            'laps_frs_index', 'replays_frs_index', 'non_replays_frs_index', 'x_frs_index', 'y_frs_index', 'z_frs_index', # fri variables  
            'long_short_fr_indicies_df', # dataframe
            ]



    """
    from neuropy.core.epoch import Epoch
    from neuropy.utils.dynamic_container import DynamicContainer # for instantaneous firing rate versions

    # Instantaneous firing rate config:
    if curr_active_pipeline.global_computation_results.computation_config is None:
        # Create a DynamicContainer-backed computation_config
        print(f'pipeline_complete_compute_long_short_fr_indicies is lacking a required computation config parameter! creating a new curr_active_pipeline.global_computation_results.computation_config')
        # curr_active_pipeline.global_computation_results.computation_config = DynamicContainer(instantaneous_time_bin_size_seconds=0.01)
        curr_active_pipeline.global_computation_results.computation_config = DynamicContainer(instantaneous_time_bin_size_seconds=None) # disable inst frs indexS
    else:
        print(f'have an existing `global_computation_results.computation_config`: {curr_active_pipeline.global_computation_results.computation_config}')	

    # Could also use `owning_pipeline_reference.global_computation_results.computation_config`
    
    
    assert (curr_active_pipeline.global_computation_results.computation_config is not None), f"requires `global_computation_results.computation_config.instantaneous_time_bin_size_seconds`"
    assert (hasattr(curr_active_pipeline.global_computation_results.computation_config.long_short_inst_spike_rate_groups, 'instantaneous_time_bin_size_seconds'))
    ## TODO: get from active_configs or something similar
    instantaneous_time_bin_size_seconds: float = curr_active_pipeline.global_computation_results.computation_config.long_short_inst_spike_rate_groups.get('instantaneous_time_bin_size_seconds', None) # 0.01 # 10ms

    # Setting `instantaneous_time_bin_size_seconds = None` disables instantaneous computations

    ## Begin Original:
    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06' # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in [long_epoch_name, short_epoch_name]]
    
    ## The issue comes in when attempting to label_slice on the session's `.epochs` object, which has labels: ['maze1', 'maze2', 'maze'] and not ['maze1_any', 'maze2_any', 'maze_any']
    if ((long_epoch_obj.n_epochs == 0) or (short_epoch_obj.n_epochs == 0)):
        # if either epoch is missing its info, we should try to use the filtered_epochs instead.
        long_epoch_obj, short_epoch_obj = [curr_active_pipeline.filtered_epochs[an_epoch_name].to_Epoch() for an_epoch_name in [long_epoch_name, short_epoch_name]] # `curr_active_pipeline.filtered_epochs[an_epoch_name]` actually contains a NamedTimerange, but we can get an Epoch if we want
        assert ((long_epoch_obj.n_epochs > 0) or (short_epoch_obj.n_epochs > 0))
    

    active_context = active_identifying_session_ctx.adding_context(collision_prefix='fn', fn_name='long_short_firing_rate_indicies')

    spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df) # TODO: CORRECTNESS: should I be using this spikes_df instead of the filtered ones?

    # Get existing laps from session:
    # long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_laps, short_laps, global_laps = [Epoch.filter_epochs(curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj(), pos_df=curr_active_pipeline.filtered_sessions[an_epoch_name].position.to_dataframe(), spikes_df=curr_active_pipeline.filtered_sessions[an_epoch_name].spikes_df, min_epoch_included_duration=1.0, max_epoch_included_duration=30.0, maximum_speed_thresh=None, min_num_unique_aclu_inclusions=3) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    # TODO 2023-04-11 - Note this doesn't assign these filtered laps objects to the session or anything yet, it just returns them.

    # Get existing replays from session:
    try:
        long_replays, short_replays, global_replays = [DataSession.filter_replay_epochs(curr_active_pipeline.filtered_sessions[an_epoch_name].replay, pos_df=curr_active_pipeline.filtered_sessions[an_epoch_name].position.to_dataframe(), spikes_df=curr_active_pipeline.filtered_sessions[an_epoch_name].spikes_df) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # NOTE: this includes a few overlapping   epochs since the function to remove overlapping ones seems to be broken

    except (AttributeError, KeyError) as e:
        print(f'!!WARN!!: pipeline_complete_compute_long_short_fr_indicies(...): Replays missing, need to compute new ones... e: {e}')
        # AttributeError: 'DataSession' object has no attribute 'replay'. Fallback to PBEs?
        # filter_epochs = a_session.pbe # Epoch object
        filter_epoch_replacement_type = KnownFilterEpochs.PBE

        # filter_epochs = a_session.ripple # Epoch object
        # filter_epoch_replacement_type = KnownFilterEpochs.RIPPLE
        print(f'!!WARN!!: pipeline_complete_compute_long_short_fr_indicies(...): missing .replay epochs, using {filter_epoch_replacement_type} as surrogate replays...')
        active_context = active_context.adding_context(collision_prefix='replay_surrogate', replays=filter_epoch_replacement_type.name)

        ## Working:
        # long_replays, short_replays, global_replays = [KnownFilterEpochs.perform_get_filter_epochs_df(sess=a_computation_result.sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration) for a_computation_result in [long_computation_results, short_computation_results, global_computation_results]] # returns Epoch objects
        # New sess.compute_estimated_replay_epochs(...) based method:
        # raise NotImplementedError(f'estimate_replay_epochs is invalid because it does not properly use the parameters!')
        assert curr_active_pipeline.sess.config.preprocessing_parameters.replays is not None
        raise ValueError(f"2024-07-02 - Overwriting internal replays with estimates, not what was intended!")
        long_replays, short_replays, global_replays = [curr_active_pipeline.filtered_sessions[an_epoch_name].estimate_replay_epochs(*curr_active_pipeline.sess.config.preprocessing_parameters.replays) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]


    # non_running_periods = Epoch.from_PortionInterval(owning_pipeline_reference.sess.laps.as_epoch_obj().to_PortionInterval().complement())
    global_non_replays: Epoch = Epoch(Epoch.from_PortionInterval(global_replays.to_PortionInterval().complement()).time_slice(t_start=long_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop).to_dataframe()[:-1]) #[:-1] # any period except the replay ones, drop the infinite last entry
    long_non_replays: Epoch  = global_non_replays.time_slice(t_start=long_epoch_obj.t_start, t_stop=long_epoch_obj.t_stop) # any period except the replay ones
    short_non_replays: Epoch  = global_non_replays.time_slice(t_start=short_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop) # any period except the replay ones

    ## Now we have replays either way:


    ## Build the output results dict:
    all_results_dict = dict(zip(['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays'], [long_laps, long_replays, short_laps, short_replays, global_laps, global_replays])) # all variables
    # Add the non-replay periods
    all_results_dict.update(dict(zip(['long_non_replays', 'short_non_replays', 'global_non_replays'], [long_non_replays, short_non_replays, global_non_replays])))
    

    # temp_save_filename = f'{active_context.get_description()}_results.pkl'
    if temp_save_filename is not None:
        print(f'temp_save_filename: {temp_save_filename}')


    ## Now have the epoch periods to calculate the firing rates for, next compute the firing rates.
    x_frs_index, y_frs_index, z_frs_index, updated_all_results_dict = _generalized_compute_long_short_firing_rate_indicies(spikes_df, **{'laps': (long_laps, short_laps), 'replays': (long_replays, short_replays), 'non_replays': (long_non_replays, global_non_replays)},
                                                                                                                           instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds, save_path=temp_save_filename)

    all_results_dict.update(updated_all_results_dict) # append the results dict
    
    # Computing consolidated `long_short_fr_indicies_df`
    _curr_aclus = list(all_results_dict['laps_frs_index'].keys()) # extract one set of keys for the aclus
    _curr_frs_indicies_dict = {k:v.values() for k,v in all_results_dict.items() if k in ['laps_frs_index', 'laps_inst_frs_index', 'replays_frs_index', 'replays_inst_frs_index', 'non_replays_frs_index', 'non_replays_inst_frs_index']} # extract the values
    all_results_dict['long_short_fr_indicies_df'] = long_short_fr_indicies_df = pd.DataFrame(_curr_frs_indicies_dict, index=_curr_aclus)
   
    ## Set the backwards compatibility variables:
    all_results_dict.update({'x_frs_index': x_frs_index, 'y_frs_index': y_frs_index, 'z_frs_index': z_frs_index}) # make sure that [x,y]_frs_index key is present for backwards compatibility.
    # long_short_fr_indicies_analysis_results['x_frs_index'] = long_short_fr_indicies_analysis_results['replays_inst_frs_index'].copy()
    # long_short_fr_indicies_analysis_results['y_frs_index'] = long_short_fr_indicies_analysis_results['non_replays_inst_frs_index'].copy()
    # all_results_dict.update(dict(zip(['x_frs_index', 'y_frs_index'], [x_frs_index, y_frs_index]))) # append the indicies to the results dict
    
    curr_active_pipeline.global_computation_results.computation_config.long_short_inst_spike_rate_groups.instantaneous_time_bin_size_seconds = instantaneous_time_bin_size_seconds ## update the property with the used value after computation
    
    return active_context, all_results_dict # TODO: add to computed_data instead

@function_attributes(short_name=None, tags=['rr', 'rate_remapping', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-18 18:58', related_items=[])
def compute_rate_remapping_stats(long_short_fr_indicies_analysis, aclu_to_neuron_type_map, considerable_remapping_threshold:float=0.7) -> RateRemappingResult:
    """ 2023-05-18 - Yet another form of measuring rate-remapping. 
    
    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_rate_remapping_stats
        rate_remapping_df = compute_rate_remapping_stats(curr_active_pipeline.global_computation_results.computed_data.long_short_fr_indicies_analysis, global_session.neurons.aclu_to_neuron_type_map, considerable_remapping_threshold=0.7)

        high_remapping_cells_only = rate_remapping_df[rate_remapping_df['has_considerable_remapping']] 
        high_remapping_cells_only

        ## Extract rr_* variables from rate_remapping_df
        rr_aclus = rate_remapping_df.index.values
        rr_laps, rr_replays, rr_skew, rr_neuron_type = [rate_remapping_df[n].values for n in ['laps', 'replays', 'skew', 'neuron_type']]


        ## Display:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_rr_aclu
        n_debug_limit = 100
        fig, ax = plot_rr_aclu([str(aclu) for aclu in rr_aclus[:n_debug_limit]], rr_laps=rr_laps[:n_debug_limit], rr_replays=rr_replays[:n_debug_limit])

        ## Display Paginated multi-plot
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import RateRemappingPaginatedFigureController
        active_identifying_session_ctx = curr_active_pipeline.sess.get_context()
        _out_rr_pagination_controller = RateRemappingPaginatedFigureController.init_from_rr_data(rr_aclus, rr_laps, rr_replays, rr_neuron_type, max_subplots_per_page=20, a_name='TestRateRemappingPaginatedFigureController', active_context=active_identifying_session_ctx)
        a_paginator = _out_rr_pagination_controller.plots_data.paginator

        
    x_frs_index: rr_replays
    y_frs_index: rr_laps
    
    """
    rr_aclus = np.array(list(long_short_fr_indicies_analysis.y_frs_index.keys())) # uses the .keys()
    rr_neuron_type = np.array([aclu_to_neuron_type_map[aclu] for aclu in rr_aclus])
    rr_laps = np.array(list(long_short_fr_indicies_analysis.y_frs_index.values()))
    rr_replays = np.array(list(long_short_fr_indicies_analysis.x_frs_index.values()))
    rr_skew = rr_laps / rr_replays

    ## Sort on 'rr_replays':
    # sort_indicies = np.argsort(rr_replays)

    ## Sort on 'rr_laps':
    sort_indicies = np.argsort(rr_laps)

    ## Sort all the variables:
    rr_aclus = rr_aclus[sort_indicies]
    rr_neuron_type = rr_neuron_type[sort_indicies] # NeuronType
    rr_laps = rr_laps[sort_indicies]
    rr_replays = rr_replays[sort_indicies]
    rr_skew = rr_skew[sort_indicies]

    # Build dataframe:
    rate_remapping_df = pd.DataFrame({'aclu': rr_aclus, 'neuron_type': rr_neuron_type, 'laps': rr_laps, 'replays': rr_replays, 'skew': rr_skew})
    rate_remapping_df.set_index('aclu', inplace=True)
    rate_remapping_df.dropna('index', subset=['laps','replays'], how='any', inplace=True) # drop any cells that don't have any replays at all.

    ## Add distances from the center to indicate how much remapping that cell engaged in so we can find the cells that remapped the most:
    rate_remapping_df['max_axis_distance_from_center'] =  np.max(np.vstack((np.abs(rate_remapping_df['laps'].to_numpy()), np.abs(rate_remapping_df['replays'].to_numpy()))), axis=0)
    rate_remapping_df['distance_from_center'] = np.sqrt(rate_remapping_df['laps'].to_numpy() ** 2 + rate_remapping_df['replays'].to_numpy() ** 2)

    ## Find only the cells that exhibit considerable remapping:
    # considerable_remapping_threshold = 0.7 # cells exhibit "considerable remapping" when they exceed 0.7
    rate_remapping_df['has_considerable_remapping'] = (rate_remapping_df['max_axis_distance_from_center'] > considerable_remapping_threshold)
    
    # rendering only:
    rate_remapping_df['render_color'] = [a_neuron_type.renderColor for a_neuron_type in rate_remapping_df.neuron_type] # Get color corresponding to each neuron type (`NeuronType`):

    
    high_remapping_cells_only = rate_remapping_df[rate_remapping_df['has_considerable_remapping']]
    rate_remapping_result = RateRemappingResult(rr_df=rate_remapping_df, high_only_rr_df=high_remapping_cells_only, considerable_remapping_threshold=considerable_remapping_threshold, is_global=True)


    return rate_remapping_result

# ==================================================================================================================== #
# Jonathan's helper functions                                                                                          #
# ==================================================================================================================== #
def _final_compute_jonathan_replay_fr_analyses(sess, replays_df: pd.DataFrame, t_start: float, t_delta: float, t_end: float, debug_print=False):
    """_summary_

    Args:
        sess (_type_): _description_
        replays_df (pd.DataFrame): sess.replay dataframe. Must have [["start", "end"]] columns

    Returns:
        _type_: _description_

    Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import _final_compute_jonathan_replay_fr_analyses
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
    rdf: pd.DataFrame = make_rdf(sess, replays_df, t_start=t_start, t_delta=t_delta, t_end=t_end) # this creates the replay dataframe variable
    rdf = remove_repeated_replays(rdf)
    rdf, aclu_to_idx = add_spike_counts(sess, rdf)

    rdf = remove_nospike_replays(rdf)
    rdf['duration'] = rdf['end'] - rdf['start']
    if debug_print:
        print(f"RDF has {len(rdf)} rows.")

    ### Make `irdf` (inter-replay dataframe)
    irdf: pd.DataFrame = make_irdf(sess, rdf, t_start=t_start, t_delta=t_delta, t_end=t_end)
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
    aclu_to_idx: dict[int, int] = {aclus[i]:i for i in range(len(aclus))}

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
def make_rdf(sess, replays_df, t_start: float, t_delta: float, t_end: float):
    """ recieves `replays_df`, but uses `sess.paradigm[1][0,0]` """
    rdf = replays_df.copy()[["start", "end"]]
    rdf["short_track"] = rdf["start"] > t_delta
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
def make_irdf(sess, rdf, t_start: float, t_delta: float, t_end: float):
    starts = [t_start]
    ends = []
    for i, row in rdf.iterrows():
        ends.append(row.start)
        starts.append(row.end)
    ends.append(t_end)
    short_track = [s > t_delta for s in starts]
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

@function_attributes(short_name='compute_evening_morning_parition', tags=['evening_morning', 'compute', 'firing_rates', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-18 13:23')
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

@function_attributes(short_name=None, tags=['measured_vs_expected', 'firing_rate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-26 00:00', related_items=[])
def compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, a_decoder_1D: "BasePositionDecoder", a_decoder_result: "DecodedFilterEpochsResult"):
    """ 2023-05-26 - Goal is to compute the expected and measured firing rates for each cell for each epoch. 

    Want to be able to get a vector of firing rates (one for each cell) for an epoch i.

    """
    all_cells_decoded_epoch_time_bins = {}
    all_cells_decoded_expected_firing_rates = {}
    
    ## for each cell:
    for i, left_out_aclu in enumerate(a_decoder_1D.neuron_IDs):
        # aclu = decoder_1D.neuron_IDs[i]
        left_out_neuron_IDX = a_decoder_1D.neuron_IDXs[i] # should just be i, but just to be safe
        ## TODO: only look at bins where the cell fires (is_cell_firing_time_bin[i])

        ## single cell outputs:
        curr_cell_decoded_epoch_time_bins = [] # will be a list of the time bins in each epoch that correspond to each surprise in the corresponding list in curr_cell_computed_epoch_surprises 
        
        curr_cell_pf_curve = a_decoder_1D.pf.ratemap.tuning_curves[left_out_neuron_IDX]
        
        ## Must pre-allocate each with an empty list:
        all_cells_decoded_expected_firing_rates[left_out_aclu] = [] 
        
        for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs):
            curr_epoch_time_bin_container = a_decoder_result.time_bin_containers[decoded_epoch_idx]
            curr_cell_decoded_epoch_time_bins.append(curr_epoch_time_bin_container)
            curr_time_bins = curr_epoch_time_bin_container.centers
            curr_epoch_p_x_given_n = a_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_cell_pf_curve.shape[0]
            
            ## Need to exclude estimates from bins that didn't have any spikes in them (in general these glitch around):
            curr_total_spike_counts_per_window = np.sum(a_decoder_result.spkcount[decoded_epoch_idx], axis=0) # left_out_decoder_result.spkcount[i].shape # (69, 222) - (nCells, nTimeWindowCenters)
            curr_is_time_bin_non_firing = (curr_total_spike_counts_per_window == 0) # this would mean that no cells fired in this time bin
            curr_most_likely_position_indicies = a_decoder_result.most_likely_position_indicies_list[decoded_epoch_idx] # (n_epoch_time_bins, ) one position for each time bin in the replay
            
            # From the firing map of the placefields for this neuron (`decoder_1D.F.T[left_out_neuron_IDX]`) get the value for each position bin index in the epoch
            curr_epoch_expected_fr = np.squeeze(a_decoder_1D.F.T[left_out_neuron_IDX][curr_most_likely_position_indicies])
            # expected_num_spikes = curr_epoch_expected_fr * decoder_result.decoding_time_bin_size

            # Eqn 1:
            # p_n_given_x = lambda n: (1.0/factorial(n)) * pow(expected_num_spikes, n) * np.exp(-expected_num_spikes) # likelihood function
            
            all_cells_decoded_expected_firing_rates[left_out_aclu].append(curr_epoch_expected_fr)


        ## End loop over decoded epochs
        # assert len(curr_cell_decoded_epoch_time_bins) == len(curr_cell_computed_epoch_surprises)
        all_cells_decoded_epoch_time_bins[left_out_aclu] = curr_cell_decoded_epoch_time_bins

    # ## End loop over cells

    ## Reshape to -for-each-epoch instead of -for-each-cell
    all_epochs_decoded_epoch_time_bins = []
    all_epochs_computed_expected_cell_firing_rates = []
    for decoded_epoch_idx in np.arange(active_filter_epochs.n_epochs):
        all_epochs_decoded_epoch_time_bins.append(np.array([all_cells_decoded_epoch_time_bins[aclu][decoded_epoch_idx].centers for aclu in a_decoder_1D.neuron_IDs])) # these are duplicated (and the same) for each cell
        all_epochs_computed_expected_cell_firing_rates.append(np.array([all_cells_decoded_expected_firing_rates[aclu][decoded_epoch_idx] for aclu in a_decoder_1D.neuron_IDs]))

    ## These are already in the -for-each-epoch form and just need conversion:
    decoder_time_bin_centers = [a_decoder_result.time_bin_containers[decoded_epoch_idx].centers for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs)]
    all_epochs_computed_expected_cell_num_spikes = [(all_epochs_computed_expected_cell_firing_rates[decoded_epoch_idx] * a_decoder_result.decoding_time_bin_size) for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs)]
    all_epochs_computed_observed_from_expected_difference = [(all_epochs_computed_expected_cell_num_spikes[decoded_epoch_idx] - a_decoder_result.spkcount[decoded_epoch_idx]) for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs)]
    # Interpolate the measured positions to the window center times:
    measured_pos_window_centers = [np.interp(curr_time_bins, active_pos_df.t, active_pos_df.lin_pos) for curr_time_bins in decoder_time_bin_centers] # TODO 2023-05-26: do I want .x or .lin_pos?
    
    ## These aggregate over all time bins in each epoch:
        # Note that some of these correspond to values that are still separate by cell
    all_epochs_decoded_epoch_time_bins_mean = np.vstack([np.mean(curr_epoch_time_bins, axis=1) for curr_epoch_time_bins in all_epochs_decoded_epoch_time_bins]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_expected_cell_firing_rates_mean = np.vstack([np.mean(curr_epoch_values, axis=1) for curr_epoch_values in all_epochs_computed_expected_cell_firing_rates]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_expected_cell_firing_rates_stddev = np.vstack([np.std(curr_epoch_values, axis=1) for curr_epoch_values in all_epochs_computed_expected_cell_firing_rates]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)

    # the maximum magnitude difference is found for all timebins within each epoch. This gives 1 value for each epoch
    all_epochs_computed_observed_from_expected_difference_max_index = [np.argmax(np.abs(all_epochs_computed_observed_from_expected_difference[decoded_epoch_idx]), axis=1, keepdims=False) for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs)]
    all_epochs_computed_observed_from_expected_difference_maximum = [np.array([all_epochs_computed_observed_from_expected_difference[decoded_epoch_idx][neuron_IDX, all_epochs_computed_observed_from_expected_difference_max_index[decoded_epoch_idx][neuron_IDX]] for neuron_IDX in np.arange(len(a_decoder_1D.neuron_IDs))]) for decoded_epoch_idx in np.arange(a_decoder_result.num_filter_epochs)]
    
    return decoder_time_bin_centers, all_epochs_computed_expected_cell_num_spikes, all_epochs_computed_observed_from_expected_difference, measured_pos_window_centers, (all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_expected_cell_firing_rates_mean, all_epochs_computed_expected_cell_firing_rates_stddev, all_epochs_computed_observed_from_expected_difference_maximum)

@function_attributes(short_name=None, tags=['measured_vs_expected', 'firing_rate'], input_requires=[], output_provides=[], uses=[], used_by=['_perform_long_short_post_decoding_analysis'], creation_date='2023-05-30 00:00', related_items=[])
def simpler_compute_measured_vs_expected_firing_rates(active_pos_df, active_filter_epochs, a_decoder_1D: "BasePositionDecoder", a_decoder_result: "DecodedFilterEpochsResult", debug_print:bool=False):
    """ 2023-05-30 - Goal is to compute the expected and measured firing rates for each cell for each epoch. 
            Attempting a smarter and more refined implementation.
    Want to be able to get a vector of firing rates (one for each cell) for an epoch i.

    """
    num_neurons = a_decoder_1D.num_neurons
    num_epochs = a_decoder_result.num_filter_epochs
    
    all_cells_decoded_expected_firing_rates_list: List[np.ndarray] = [a_decoder_1D.F[np.squeeze(curr_most_likely_position_indicies),:] for curr_most_likely_position_indicies in a_decoder_result.most_likely_position_indicies_list]
    assert len(all_cells_decoded_expected_firing_rates_list) == a_decoder_result.num_filter_epochs # one for each epoch

    num_timebins_in_epoch: NDArray[Shape["Num_epochs"], Int] = np.array([np.shape(epoch_values)[0] for epoch_values in all_cells_decoded_expected_firing_rates_list])
    num_total_flat_timebins: int = np.sum(num_timebins_in_epoch) # number of timebins across all epochs
    # flat_epoch_idxs: NDArray[Shape["N_total_flat_timebins"], Int] = np.concatenate([np.repeat(i, np.shape(epoch_values)[0]) for i, epoch_values in enumerate(all_cells_decoded_expected_firing_rates_list)]) # for each time bin repeat the epoch_id so we can recover it if needed
    
    # Basic Numpy (Full-array) version:
    # flat_expected_firing_rates: NDArray[Shape["N_total_flat_timebins, N_neurons"], Any] = np.vstack(all_cells_decoded_expected_firing_rates_list)
    # flat_expected_num_spikes: NDArray[Shape["N_total_flat_timebins, N_neurons"], Any] = flat_expected_firing_rates * a_decoder_result.decoding_time_bin_size
    # flat_observed_num_spikes: NDArray[Shape["N_total_flat_timebins, N_neurons"], Any] = np.hstack(a_decoder_result.spkcount).T
    # flat_observed_from_expected_difference: NDArray[Shape["N_total_flat_timebins, N_neurons"], Any] = flat_expected_num_spikes - flat_observed_num_spikes

    ## Awkward Array (Ragged-array) version:
    ragged_expected_firing_rates_arr = ak.Array(all_cells_decoded_expected_firing_rates_list) # awkward array
    num_timebins_in_epoch = ak.num(ragged_expected_firing_rates_arr, axis=1).to_numpy()
    num_total_flat_timebins: int = np.sum(num_timebins_in_epoch)
    if debug_print:
        print(f'num_neurons: {num_neurons}, num_epochs: {num_epochs}, num_total_flat_timebins: {num_total_flat_timebins}')

    ragged_expected_num_spikes_arr = ragged_expected_firing_rates_arr * a_decoder_result.decoding_time_bin_size
    ragged_observed_from_expected_diff = ragged_expected_num_spikes_arr - ak.Array([v.T for v in a_decoder_result.spkcount])


    ## By epoch quantities, this is correct:
    observed_from_expected_diff_max = ak.to_regular(ak.max(ragged_observed_from_expected_diff, axis=1)).to_numpy().T # type: 120 * 30 * float64
    observed_from_expected_diff_ptp = ak.to_regular(ak.ptp(ragged_observed_from_expected_diff, axis=1)).to_numpy().T # type: 120 * 30 * float64
    observed_from_expected_diff_mean = ak.to_regular(ak.mean(ragged_observed_from_expected_diff, axis=1)).to_numpy().T # type: 120 * 30 * float64
    observed_from_expected_diff_std = ak.to_regular(ak.std(ragged_observed_from_expected_diff, axis=1)).to_numpy().T # type: 120 * 30 * float64

    return (num_neurons, num_timebins_in_epoch, num_total_flat_timebins), (observed_from_expected_diff_max, observed_from_expected_diff_ptp, observed_from_expected_diff_mean, observed_from_expected_diff_std)


# ==================================================================================================================== #
# Overlap                                                                                                      #
# ==================================================================================================================== #

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


@custom_define(slots=False)
class SingleBarResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ a simple replacement for the tuple that's current passed """
    mean: float = serialized_attribute_field() # computable
    std: float = serialized_attribute_field() # effectively computable
    values: np.ndarray = serialized_field()
    LxC_aclus: np.ndarray = serialized_field(hdf_metadata={'track_eXclusive_cells': 'LxC'}) # the list of long-eXclusive cell aclus
    SxC_aclus: np.ndarray = serialized_field(hdf_metadata={'track_eXclusive_cells': 'SxC'}) # the list of short-eXclusive cell aclus
    LxC_scatter_props: Optional[Dict] = non_serialized_field(is_computable=False)
    SxC_scatter_props: Optional[Dict] = non_serialized_field(is_computable=False)

    @classmethod
    def combine(cls, *results: 'SingleBarResult') -> 'SingleBarResult':
        """Combines multiple SingleBarResult instances into a new instance.

        Args:
            *results: One or more SingleBarResult instances to combine

        Returns:
            A new SingleBarResult with combined data and recomputed statistics

        Raises:
            ValueError: If no results are provided
        """
        if len(results) == 0:
            raise ValueError("At least one SingleBarResult must be provided")

        if len(results) == 1:
            # Return a copy of the single result
            return deepcopy(results[0])

        # Initialize arrays for combining
        all_values = []
        all_LxC_aclus = []
        all_SxC_aclus = []

        # Initialize scatter props dictionaries
        combined_LxC_scatter_props = {}
        combined_SxC_scatter_props = {}

        # Collect values from all results
        for result in results:
            if result.values is not None and len(result.values) > 0:
                all_values.append(result.values)

            if result.LxC_aclus is not None and len(result.LxC_aclus) > 0:
                all_LxC_aclus.append(result.LxC_aclus)

            if result.SxC_aclus is not None and len(result.SxC_aclus) > 0:
                all_SxC_aclus.append(result.SxC_aclus)

            # Merge scatter properties dictionaries
            if result.LxC_scatter_props:
                for key, value in result.LxC_scatter_props.items():
                    if key in combined_LxC_scatter_props:
                        # If the key exists and values are arrays, concatenate them
                        if isinstance(value, np.ndarray) and isinstance(combined_LxC_scatter_props[key], np.ndarray):
                            combined_LxC_scatter_props[key] = np.concatenate([combined_LxC_scatter_props[key], value])
                        else:
                            # For non-array values, keep the most recent
                            combined_LxC_scatter_props[key] = value
                    else:
                        combined_LxC_scatter_props[key] = value

            if result.SxC_scatter_props:
                for key, value in result.SxC_scatter_props.items():
                    if key in combined_SxC_scatter_props:
                        # If the key exists and values are arrays, concatenate them
                        if isinstance(value, np.ndarray) and isinstance(combined_SxC_scatter_props[key], np.ndarray):
                            combined_SxC_scatter_props[key] = np.concatenate([combined_SxC_scatter_props[key], value])
                        else:
                            # For non-array values, keep the most recent
                            combined_SxC_scatter_props[key] = value
                    else:
                        combined_SxC_scatter_props[key] = value

        # Combine arrays
        combined_values = np.concatenate(all_values) if all_values else np.array([])
        combined_LxC_aclus = np.concatenate(all_LxC_aclus) if all_LxC_aclus else np.array([])
        combined_SxC_aclus = np.concatenate(all_SxC_aclus) if all_SxC_aclus else np.array([])

        # Compute new statistics
        new_mean = np.mean(combined_values) if len(combined_values) > 0 else 0.0
        new_std = np.std(combined_values) if len(combined_values) > 0 else 0.0

        # Create and return a new instance
        return cls(mean=new_mean, std=new_std, values=combined_values, LxC_aclus=combined_LxC_aclus, SxC_aclus=combined_SxC_aclus,
                   LxC_scatter_props=(combined_LxC_scatter_props if combined_LxC_scatter_props else None), SxC_scatter_props=(combined_SxC_scatter_props if combined_SxC_scatter_props else None))




    @function_attributes(short_name=None, tags=['combine', 'add', 'across-sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-12 09:06', related_items=[])
    def add(self, *others: 'SingleBarResult') -> 'SingleBarResult':
        """Combines multiple SingleBarResult instances into a new instance.

        Args:
            *others: One or more SingleBarResult instances to combine with this one

        Returns:
            A new SingleBarResult with combined data and recomputed statistics
        """
        # Create and return a new instance
        return SingleBarResult.combine(self, *others)





def _InstantaneousSpikeRateGroupsComputation_convert_Fig2_ANY_FR_to_hdf_fn(f, key: str, value):
    """ Converts `Fig2_Replay_FR: List[SingleBarResult]` or `Fig2_Laps_FR: List[SingleBarResult]` into something serializable. Unfortunately has to be defined outside the `InstantaneousSpikeRateGroupsComputation` definition so it can be used in the field. value: List[SingleBarResult] """
    assert isinstance(value, list)
    assert len(value) == 4
    if key.endswith('/Fig2_Replay_FR'):
        key = key.removesuffix('/Fig2_Replay_FR')
        Fig2_Replay_FR_key:str = f"{key}/Fig2/Replay/inst_FR_Bars"
        for specific_bar_key, specific_bar_SingleBarResult in zip(['LxC_ReplayDeltaMinus', 'LxC_ReplayDeltaPlus', 'SxC_ReplayDeltaMinus', 'SxC_ReplayDeltaPlus'], value):
            specific_bar_SingleBarResult.to_hdf(f, f'{Fig2_Replay_FR_key}/{specific_bar_key}')
    elif key.endswith('/Fig2_Laps_FR'):
        key = key.removesuffix('/Fig2_Laps_FR')
        Fig2_Laps_FR_key:str = f"{key}/Fig2/Laps/inst_FR_Bars"
        for specific_bar_key, specific_bar_SingleBarResult in zip(['LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus'], value):
            specific_bar_SingleBarResult.to_hdf(f, f'{Fig2_Laps_FR_key}/{specific_bar_key}')
    else:
        raise NotImplementedError



@custom_define(slots=False)
class InstantaneousSpikeRateGroupsComputation(PickleSerializableMixin, HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ class to handle spike rate computations

    Appears to only be for the XxC (LxC, SxC) cells.
    

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation

    """
    instantaneous_time_bin_size_seconds: float = serialized_attribute_field(default=0.01) # 10ms
    active_identifying_session_ctx: IdentifyingContext = serialized_attribute_field(init=False, repr=True, default=None, is_computable=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v.to_dict()))) # , serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v.to_dict()))
    
    LxC_aclus: np.ndarray = serialized_field(init=False, hdf_metadata={'track_eXclusive_cells': 'LxC'}) # the list of long-eXclusive cell aclus
    SxC_aclus: np.ndarray = serialized_field(init=False, hdf_metadata={'track_eXclusive_cells': 'SxC'}) # the list of short-eXclusive cell aclus
    AnyC_aclus: NDArray = serialized_field(init=False, hdf_metadata={'track_eXclusive_cells': 'AnyC'}, metadata={'field_added': "2025.07.23_0"}) # the list of short-eXclusive cell aclus

    Fig2_Replay_FR: List[SingleBarResult] = serialized_field(init=False, is_computable=True, serialization_fn=_InstantaneousSpikeRateGroupsComputation_convert_Fig2_ANY_FR_to_hdf_fn, hdf_metadata={'epochs': 'Replay'}) # a list of the four single-bar results.
    Fig2_Laps_FR: List[SingleBarResult] = serialized_field(init=False, is_computable=True, serialization_fn=_InstantaneousSpikeRateGroupsComputation_convert_Fig2_ANY_FR_to_hdf_fn, hdf_metadata={'epochs': 'Laps'}) # a list of the four single-bar results.

    AnyC_ReplayDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'AnyC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaMinus'}, metadata={'field_added': "2025.07.23_0"})
    AnyC_ReplayDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'AnyC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaPlus'}, metadata={'field_added': "2025.07.23_0"})
    AnyC_ThetaDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'AnyC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaMinus'}, metadata={'field_added': "2025.07.23_0"})
    AnyC_ThetaDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'AnyC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaPlus'}, metadata={'field_added': "2025.07.23_0"})

    LxC_ReplayDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaMinus'})
    LxC_ReplayDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaPlus'})
    SxC_ReplayDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaMinus'})
    SxC_ReplayDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaPlus'})

    LxC_ThetaDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaMinus'})
    LxC_ThetaDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaPlus'})
    SxC_ThetaDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaMinus'})
    SxC_ThetaDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaPlus'})

    ## New
    all_incl_endPlatforms_InstSpikeRateTrends_df: Optional[pd.DataFrame] = non_serialized_field(repr=False, default=None, is_computable=False) # converter=attrs.converters.default_if_none("")


    @classmethod
    def _perform_compute_spike_rate_bars(cls, LxC_aclus: NDArray, SxC_aclus: NDArray, LxC_ThetaDeltaMinus: SpikeRateTrends, LxC_ThetaDeltaPlus: SpikeRateTrends, SxC_ThetaDeltaMinus: SpikeRateTrends, SxC_ThetaDeltaPlus: SpikeRateTrends,
                                          LxC_ReplayDeltaMinus: SpikeRateTrends, LxC_ReplayDeltaPlus: SpikeRateTrends, SxC_ReplayDeltaMinus: SpikeRateTrends, SxC_ReplayDeltaPlus: SpikeRateTrends,
                                          ):
        """ Computing 
        

        (Fig2_Laps_FR, Fig2_Replay_FR) = InstantaneousSpikeRateGroupsComputation._perform_compute_spike_rate_bars(LxC_aclus=LxC_aclus, SxC_aclus=SxC_aclus,
                                             LxC_ThetaDeltaMinus=LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus=LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus=SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus=SxC_ThetaDeltaPlus,
                                             LxC_ReplayDeltaMinus=LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus=LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus=SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus=SxC_ReplayDeltaPlus,
                                          )
        """
        # Common:
        are_LxC_empty: bool = (LxC_aclus is None) or (len(LxC_aclus) == 0)
        are_SxC_empty: bool = (SxC_aclus is None) or (len(SxC_aclus) == 0)
        
        # Note that in general LxC and SxC might have differing numbers of cells.
        if (are_LxC_empty or are_SxC_empty):
            # self.Fig2_Replay_FR = None # None mode
            # initialize with an empty array and None values for the mean and std.
            Fig2_Replay_FR: List[SingleBarResult] = []
            for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus):
                if v is not None:
                    Fig2_Replay_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, LxC_aclus, SxC_aclus, None, None))
                else:
                    Fig2_Replay_FR.append(SingleBarResult(None, None, np.array([], dtype=float), LxC_aclus, SxC_aclus, None, None))
        else:
            Fig2_Replay_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, LxC_aclus, SxC_aclus, None, None) for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus)]
        
        # Note that in general LxC and SxC might have differing numbers of cells.
        if are_LxC_empty or are_SxC_empty:
            # self.Fig2_Laps_FR = None # NONE mode
            Fig2_Laps_FR: List[SingleBarResult] = []
            for v in (LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus):
                if v is not None:
                    Fig2_Laps_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, LxC_aclus, SxC_aclus, None, None))
                else:
                    Fig2_Laps_FR.append(SingleBarResult(None, None, np.array([], dtype=float), LxC_aclus, SxC_aclus, None, None))
                    
        else:
            # Note that in general LxC and SxC might have differing numbers of cells.
            Fig2_Laps_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, LxC_aclus, SxC_aclus, None, None) for v in (LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus)]
            
        return (Fig2_Laps_FR, Fig2_Replay_FR)



    def compute_spike_rate_bars(self):
        """ Computing """
        (self.Fig2_Laps_FR, self.Fig2_Replay_FR) = InstantaneousSpikeRateGroupsComputation._perform_compute_spike_rate_bars(LxC_aclus=self.LxC_aclus, SxC_aclus=self.SxC_aclus,
                                             LxC_ThetaDeltaMinus=self.LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus=self.LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus=self.SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus=self.SxC_ThetaDeltaPlus,
                                             LxC_ReplayDeltaMinus=self.LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus=self.LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus=self.SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus=self.SxC_ReplayDeltaPlus,
                                          )


    def compute(self, curr_active_pipeline, minimum_inclusion_fr_Hz=0.0, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:

        Can access via:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import InstantaneousSpikeRateGroupsComputation

            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

    
        sess = curr_active_pipeline.sess
        # Get the provided context or use the session context:
        active_context = kwargs.get('active_context', sess.get_context())

        epoch_handling_mode:str = kwargs.pop('epoch_handling_mode', 'DropShorterMode')
            
        self.active_identifying_session_ctx = active_context
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session

        long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
        long_laps, long_replays, short_laps, short_replays, global_laps, global_replays = [long_short_fr_indicies_analysis_results[k] for k in ['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays']]

        ## Manual User-annotation mode:
        LxC_aclus = kwargs.pop('LxC_aclus', None)
        SxC_aclus = kwargs.pop('SxC_aclus', None)
        AnyC_aclus = kwargs.pop('AnyC_aclus', None)

        annotation_man: UserAnnotationsManager = UserAnnotationsManager()
        session_cell_exclusivity: SessionCellExclusivityRecord = annotation_man.annotations[self.active_identifying_session_ctx].get('session_cell_exclusivity', None)
        if ((LxC_aclus is None) or (SxC_aclus is None)) and (session_cell_exclusivity is not None):
            print(f'setting LxC_aclus/SxC_aclus from user annotation.')
            self.LxC_aclus = session_cell_exclusivity.LxC
            self.SxC_aclus = session_cell_exclusivity.SxC
        else:
            print(f'WARN: no user annotation for session_cell_exclusivity')

        # Common:
        are_LxC_empty: bool = (self.LxC_aclus is None) or (len(self.LxC_aclus) == 0)
        are_SxC_empty: bool = (self.SxC_aclus is None) or (len(self.SxC_aclus) == 0)

        # spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline) ## this gets too few spikes, should just use the raw spikes maybe
        spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        
        all_spikes_aclus = deepcopy(spikes_df.spikes.neuron_ids)
        
        if AnyC_aclus is None:
            ## build from the LxC and SxCs
            # AnyC_aclus = np.array([aclu for aclu in all_spikes_aclus if ((aclu not in self.LxC_aclus) and (aclu not in self.SxC_aclus))]) ## shared only
            AnyC_aclus = np.array([aclu for aclu in all_spikes_aclus]) # All
            self.AnyC_aclus = AnyC_aclus
            

        # Replays: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_replays`, `short_replays`
        use_instantaneous_firing_rate: bool = kwargs.get('use_instantaneous_firing_rate', False)
        if (self.instantaneous_time_bin_size_seconds >= 5.0):
            print(f'self.instantaneous_time_bin_size_seconds: {self.instantaneous_time_bin_size_seconds} >= 5.0 so not using instantaneous firing rate')
            use_instantaneous_firing_rate = False
            kwargs['use_instantaneous_firing_rate'] = False            

        else:
            print(f'self.instantaneous_time_bin_size_seconds: {self.instantaneous_time_bin_size_seconds} < 5.0 so using instantaneous firing rate')
            use_instantaneous_firing_rate = True ## override True
            kwargs['use_instantaneous_firing_rate'] = True
            
            
        # AnyC: `AnyC.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        self.AnyC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_replays, included_neuron_ids=self.AnyC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ReplayDeltaPlus: `short_replays`
        self.AnyC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_replays, included_neuron_ids=self.AnyC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        
        # LxC: `long_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        self.LxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_replays, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ReplayDeltaPlus: `short_replays`
        self.LxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_replays, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        self.SxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_replays, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ReplayDeltaPlus: `short_replays`
        self.SxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_replays, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # Note that in general LxC and SxC might have differing numbers of cells.
        if (are_LxC_empty or are_SxC_empty):
            # self.Fig2_Replay_FR = None # None mode
            # initialize with an empty array and None values for the mean and std.
            self.Fig2_Replay_FR: List[SingleBarResult] = []
            for v in (self.LxC_ReplayDeltaMinus, self.LxC_ReplayDeltaPlus, self.SxC_ReplayDeltaMinus, self.SxC_ReplayDeltaPlus):
                if v is not None:
                    self.Fig2_Replay_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None))
                else:
                    self.Fig2_Replay_FR.append(SingleBarResult(None, None, np.array([], dtype=float), self.LxC_aclus, self.SxC_aclus, None, None))
        else:
            self.Fig2_Replay_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None) for v in (self.LxC_ReplayDeltaMinus, self.LxC_ReplayDeltaPlus, self.SxC_ReplayDeltaMinus, self.SxC_ReplayDeltaPlus)]
        

        # Laps/Theta: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_laps`, `short_laps`
        # AnyC: `AnyC.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        self.AnyC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_laps, included_neuron_ids=self.AnyC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ThetaDeltaPlus: `short_laps`
        self.AnyC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_laps, included_neuron_ids=self.AnyC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # LxC: `long_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        self.LxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_laps, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ThetaDeltaPlus: `short_laps`
        self.LxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_laps, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        self.SxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_laps, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # ThetaDeltaPlus: `short_laps`
        self.SxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_laps, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # Note that in general LxC and SxC might have differing numbers of cells.
        if are_LxC_empty or are_SxC_empty:
            # self.Fig2_Laps_FR = None # NONE mode
            self.Fig2_Laps_FR: List[SingleBarResult] = []
            for v in (self.LxC_ThetaDeltaMinus, self.LxC_ThetaDeltaPlus, self.SxC_ThetaDeltaMinus, self.SxC_ThetaDeltaPlus):
                if v is not None:
                    self.Fig2_Laps_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None))
                else:
                    self.Fig2_Laps_FR.append(SingleBarResult(None, None, np.array([], dtype=float), self.LxC_aclus, self.SxC_aclus, None, None))
                    
        else:
            # Note that in general LxC and SxC might have differing numbers of cells.
            self.Fig2_Laps_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None) for v in (self.LxC_ThetaDeltaMinus, self.LxC_ThetaDeltaPlus, self.SxC_ThetaDeltaMinus, self.SxC_ThetaDeltaPlus)]



        # ==================================================================================================================================================================================================================================================================================== #
        # Compute the participation stats but they are only needed for the `AnyC_*` result groups because that is sufficient to provide it for the dataframe                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        ### Specifically updates these results by calling `a_pre_post_period_result.compute_participation_stats(..., should_update_self=True)` (which updates the `SpikeRateTrends` result in place)
                
        ## add session info
        # if self.active_identifying_session_ctx is not None:
            ## Add the extended neuron identifiers (like the global neuron_uid, session_uid) columns
            # df_combined = df_combined.neuron_identity.make_neuron_indexed_df_global(self.active_identifying_session_ctx, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
        
        a_sess_pre_post_delta_result_list = (self.AnyC_ThetaDeltaMinus, self.AnyC_ThetaDeltaPlus, self.AnyC_ReplayDeltaMinus, self.AnyC_ReplayDeltaPlus)
        a_sess_pre_post_delta_result_dict = dict(zip(['ThetaDeltaMinus', 'ThetaDeltaPlus', 'ReplayDeltaMinus', 'ReplayDeltaPlus'], a_sess_pre_post_delta_result_list))
        # for a_pre_post_period_result in a_sess_pre_post_delta_result_list:
        for a_period_name, a_pre_post_period_result in a_sess_pre_post_delta_result_dict.items():
            a_result_col_name: str = 'n_participating_epochs'
            n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict = a_pre_post_period_result.compute_participation_stats(a_session_ctxt=self.active_identifying_session_ctx, should_update_self=True)
            # df_combined['lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus'
            assert len(a_pre_post_period_result.included_neuron_ids) == len(n_participating_epochs), f"len(a_pre_post_period_result.included_neuron_ids): {len(a_pre_post_period_result.included_neuron_ids)} != len(n_participating_epochs): {len(n_participating_epochs)}"
            # assert len(df_combined) == len(n_participating_epochs), f"len(df_combined): {len(df_combined)} != len(n_participating_epochs): {len(n_participating_epochs)}"
            # df_combined[f"{a_period_name}_{a_result_col_name}"] = deepcopy(n_participating_epochs) ## add this column to the dataframe



    def get_summary_dataframe(self) -> pd.DataFrame:
        """ Returns a summary datatable for each neuron with one entry for each cell in (self.LxC_aclus + self.SxC_aclus)

        Additional Columns:
            ['ThetaDeltaMinus_n_participating_epochs', 'ThetaDeltaPlus_n_participating_epochs', 'ReplayDeltaMinus_n_participating_epochs', 'ReplayDeltaPlus_n_participating_epochs']
        """
        table_columns = ['aclu', 'lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus', 'active_set_membership']
        
        n_LxC_aclus: int = len(self.LxC_aclus)        
        v_LxC_aclus = [list(self.LxC_aclus)] + [v.cell_agg_inst_fr_list for v in (self.LxC_ThetaDeltaMinus, self.LxC_ThetaDeltaPlus, self.LxC_ReplayDeltaMinus, self.LxC_ReplayDeltaPlus) if v is not None] + [(['LxC'] * n_LxC_aclus)]
        df_LxC_aclus = pd.DataFrame(dict(zip(table_columns, v_LxC_aclus)))

        n_SxC_aclus: int = len(self.SxC_aclus)
        v_SxC_aclus = [list(self.SxC_aclus)] + [v.cell_agg_inst_fr_list for v in (self.SxC_ThetaDeltaMinus, self.SxC_ThetaDeltaPlus, self.SxC_ReplayDeltaMinus, self.SxC_ReplayDeltaPlus) if v is not None] + [(['SxC'] * n_SxC_aclus)]
        df_SxC_aclus = pd.DataFrame(dict(zip(table_columns, v_SxC_aclus)))
        
        n_AnyC_aclus: int = len(self.AnyC_aclus)
        v_AnyC_aclus = [list(self.AnyC_aclus)] + [v.cell_agg_inst_fr_list for v in (self.AnyC_ThetaDeltaMinus, self.AnyC_ThetaDeltaPlus, self.AnyC_ReplayDeltaMinus, self.AnyC_ReplayDeltaPlus) if v is not None] + [(['AnyC'] * n_AnyC_aclus)]
        df_AnyC_aclus = pd.DataFrame(dict(zip(table_columns, v_AnyC_aclus)))

        # Concatenate the two dataframes
        df_combined = pd.concat([df_LxC_aclus, df_SxC_aclus, df_AnyC_aclus], ignore_index=True)
        ## Drop duplicates, keeping the first (corresponding to the SxC/LxC over the AnyC, although all the values are the same so only the 'active_set_membership' column would need to be changed): 
        df_combined = df_combined.drop_duplicates(subset=['aclu'], keep='first', inplace=False)
        df_combined['aclu'] = df_combined['aclu'].astype(int)
        
        ## Add extra columns:
        df_combined['inst_time_bin_seconds'] = float(self.instantaneous_time_bin_size_seconds)        
        ## add session info
        if self.active_identifying_session_ctx is not None:
            ## Add the extended neuron identifiers (like the global neuron_uid, session_uid) columns
            df_combined = df_combined.neuron_identity.make_neuron_indexed_df_global(self.active_identifying_session_ctx, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
        

        # ==================================================================================================================================================================================================================================================================================== #
        # Compute `n_participating_epochs` and add to the returned dataframe as needed.                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        # Adds columns ['ThetaDeltaMinus_n_participating_epochs', 'ThetaDeltaPlus_n_participating_epochs', 'ReplayDeltaMinus_n_participating_epochs', 'ReplayDeltaPlus_n_participating_epochs']

        # a_sess_pre_post_delta_result_list = (self.AnyC_ThetaDeltaMinus, self.AnyC_ThetaDeltaPlus, self.AnyC_ReplayDeltaMinus, self.AnyC_ReplayDeltaPlus,
        #                                      self.LxC_ThetaDeltaMinus, self.LxC_ThetaDeltaPlus, self.LxC_ReplayDeltaMinus, self.LxC_ReplayDeltaPlus,
        #                                      self.SxC_ThetaDeltaMinus, self.SxC_ThetaDeltaPlus, self.SxC_ReplayDeltaMinus, self.SxC_ReplayDeltaPlus)
        
        a_sess_pre_post_delta_result_list = (self.AnyC_ThetaDeltaMinus, self.AnyC_ThetaDeltaPlus, self.AnyC_ReplayDeltaMinus, self.AnyC_ReplayDeltaPlus)
        a_sess_pre_post_delta_result_dict = dict(zip(['ThetaDeltaMinus', 'ThetaDeltaPlus', 'ReplayDeltaMinus', 'ReplayDeltaPlus'], a_sess_pre_post_delta_result_list))
        # for a_pre_post_period_result in a_sess_pre_post_delta_result_list:
        for a_period_name, a_pre_post_period_result in a_sess_pre_post_delta_result_dict.items():
            a_result_col_name: str = 'n_participating_epochs'
            n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict = a_pre_post_period_result.compute_participation_stats(a_session_ctxt=self.active_identifying_session_ctx, should_update_self=True)
            # df_combined['lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus'
            assert len(a_pre_post_period_result.included_neuron_ids) == len(n_participating_epochs), f"len(a_pre_post_period_result.included_neuron_ids): {len(a_pre_post_period_result.included_neuron_ids)} != len(n_participating_epochs): {len(n_participating_epochs)}"
            assert len(df_combined) == len(n_participating_epochs), f"len(df_combined): {len(df_combined)} != len(n_participating_epochs): {len(n_participating_epochs)}"
            df_combined[f"{a_period_name}_{a_result_col_name}"] = deepcopy(n_participating_epochs) ## add this column to the dataframe

            for k, v in per_aclu_additional_properties_dict.items():
                df_combined[f"{a_period_name}_{k}"] = deepcopy(v) 
                

        return df_combined


    @function_attributes(short_name=None, tags=['UNFINISHED', 'export', 'CSV', 'FAT', 'FAT_df'], input_requires=[], output_provides=[], uses=['.get_comprehensive_dataframe'], used_by=[], creation_date='2025-07-17 16:43', related_items=[])
    def export_as_FAT_df_CSV(self, active_export_parent_output_path: Path, owning_pipeline_reference, decoding_time_bin_size: float):
        """  export as a single_FAT .csv file
        active_export_parent_output_path = self.collected_outputs_path.resolve()
        Assert.path_exists(parent_output_path)
        csv_save_paths_dict = a_new_fully_generic_result.export_as_FAT_df_CSV(active_export_parent_output_path=active_export_parent_output_path, owning_pipeline_reference=owning_pipeline_reference, decoding_time_bin_size=decoding_time_bin_size)
        csv_save_paths_dict
        
        History:
            Extracted from `pyphoplacecellanalysis.Analysis.Decoder.context_dependent.GenericDecoderDictDecodedEpochsDictResult`
                                `._perform_export_dfs_dict_to_csvs`
                                `.export_csvs`
                                `.default_export_all_CSVs`
            

        """ 
        from pyphocorehelpers.assertion_helpers import Assert
        from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe, ensure_Epoch, TimeColumnAliasesProtocol
        from pyphocorehelpers.print_helpers import get_now_rounded_time_str
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe


        print(f'WARN: NOT YET IMPLEMENTED 2025-07-17 16:56!!!!!!!!!!!!!!')
        
        ## Unpack from pipeline:
        ## Export to CSVs:
        result_identifier_str: str = f'FAT_inst_frs'

        Assert.path_exists(active_export_parent_output_path)

        ## INPUTS: collected_outputs_path
        # decoding_time_bin_size: float = epochs_decoding_time_bin_size

        complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()
        active_context = complete_session_context
        session_name: str = owning_pipeline_reference.session_name
        earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = owning_pipeline_reference.find_LongShortDelta_times()

        ## Build the function that uses owning_pipeline_reference to build the correct filename and actually output the .csv to the right place
        def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'({result_identifier_str})', parent_output_path: Path=None):
            """ captures `owning_pipeline_reference`
            """
            output_date_str: str = get_now_rounded_time_str(rounded_minutes=10)
            out_path, out_filename, out_basename = owning_pipeline_reference.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
            export_df.to_csv(out_path)
            return out_path 

        custom_export_df_to_csv_fn = _subfn_custom_export_df_to_csv


        def _subfn_pre_process_and_export_df(export_df: pd.DataFrame, a_df_identifier: Union[str, IdentifyingContext]):
            """ sets up all the important metadata and then calls `custom_export_df_to_csv_fn(....)` to actually export the CSV
            
            captures: decoding_time_bin_size, t_start, t_delta, t_end, tbin_values_dict, time_col_name_dict, user_annotation_selections, valid_epochs_selections, custom_export_df_to_csv_fn
            """
            a_tbin_size: float = float(decoding_time_bin_size)
            ## Add t_bin column method
            export_df = export_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=a_tbin_size) ## #TODO 2025-04-05 18:12: - [ ] what about qclu? FrHz?
            a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
            a_data_identifier_str: str = f'({a_df_identifier})_tbin-{a_tbin_size_str}' ## build the identifier '(laps_weighted_corr_merged_df)_tbin-1.5'
            
            return custom_export_df_to_csv_fn(export_df, data_identifier_str=a_data_identifier_str, parent_output_path=active_export_parent_output_path) # this is exporting corr '(ripple_WCorrShuffle_df)_tbin-0.025'
        

        # ================================================================================================================================================================================ #
        # BEGIN FUNCTION BODY                                                                                                                                                              #
        # ================================================================================================================================================================================ #
        
        # tbin_values_dict={'laps': decoding_time_bin_size, 'pbe': decoding_time_bin_size, 'non_pbe': decoding_time_bin_size, 'FAT': decoding_time_bin_size}

        # csv_save_paths_dict = GenericDecoderDictDecodedEpochsDictResult._perform_export_dfs_dict_to_csvs(extracted_dfs_dict=a_new_fully_generic_result.filter_epochs_decoded_track_marginal_posterior_df_dict,
        # csv_save_paths_dict = self.export_csvs(parent_output_path=active_export_parent_output_path.resolve(),
        #                                             active_context=active_context, session_name=session_name, #curr_active_pipeline=owning_pipeline_reference,
        #                                             custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
        #                                             decoding_time_bin_size=decoding_time_bin_size,
        #                                             curr_session_t_delta=t_delta
        #                                             )
        
        csv_save_paths_dict = {}
        df: pd.DataFrame = self.get_comprehensive_dataframe() ## actually convert to a DF
        single_FAT_df: pd.DataFrame = SingleFatDataframe.build_fat_df(dfs_dict={result_identifier_str:df}, additional_common_context=active_context)
        csv_save_paths_dict[result_identifier_str] =  _subfn_pre_process_and_export_df(export_df=single_FAT_df, a_df_identifier="FAT")
    

        # across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function']['csv_save_paths_dict'] = deepcopy(csv_save_paths_dict)
        print(f'csv_save_paths_dict: {csv_save_paths_dict}\n')
        return csv_save_paths_dict



    # ==================================================================================================================================================================================================================================================================================== #
    # `get_comprehensive_dataframe(...)` Private Helpers                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['MAIN', 'to_df', 'FAT', 'FAT_df', 'equiv', 'UNVALIDATED'], input_requires=[], output_provides=[], uses=[], used_by=['.export_as_FAT_df_CSV'], creation_date='2025-01-17 15:30', related_items=['cls.from_comprehensive_dataframe'])
    def get_comprehensive_dataframe(self) -> pd.DataFrame:
        """ Creates a comprehensive DataFrame with ALL InstantaneousSpikeRateGroupsComputation data.
        Each row represents one cell with all conditions as columns.

        INVERSE OF: `cls.from_comprehensive_dataframe`

        Returns:
            pd.DataFrame: Comprehensive DataFrame with all computation data

        Usage:

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import InstantaneousSpikeRateGroupsComputation

            # Example usage of the complete round-trip functionality:

            # 1. Create and compute the original object
            inst_fr_comp = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01)
            inst_fr_comp.compute(curr_active_pipeline=curr_active_pipeline)

            # 2. Get comprehensive DataFrame
            comprehensive_df: pd.DataFrame = inst_fr_comp.get_comprehensive_dataframe()
            print(f"DataFrame shape: {comprehensive_df.shape}")
            print(f"Columns: {comprehensive_df.columns.tolist()}")
            comprehensive_df

        """
        # Handle empty case
        if ((self.LxC_aclus is None or len(self.LxC_aclus) == 0) and (self.SxC_aclus is None or len(self.SxC_aclus) == 0) and (self.AnyC_aclus is None or len(self.AnyC_aclus) == 0)):
            base_columns = ['aclu', 'cell_type', 'cell_index_in_type', 'instantaneous_time_bin_size_seconds']
            fr_columns = ['replay_delta_minus_firing_rate', 'replay_delta_plus_firing_rate', 
                        'theta_delta_minus_firing_rate', 'theta_delta_plus_firing_rate']
            pop_stat_columns = []
            for condition in ['replay_delta_minus', 'replay_delta_plus', 'theta_delta_minus', 'theta_delta_plus']:
                pop_stat_columns.extend([f'{condition}_pop_mean', f'{condition}_pop_std', f'{condition}_pop_n_cells'])
            summary_columns = []
            for condition in ['LxC_ReplayDeltaMinus', 'LxC_ReplayDeltaPlus', 'SxC_ReplayDeltaMinus', 'SxC_ReplayDeltaPlus',
                            'LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus']:
                summary_columns.extend([f'{condition}_summary_mean', f'{condition}_summary_std'])
            return pd.DataFrame(columns=base_columns + fr_columns + pop_stat_columns + summary_columns)

        records = []

        # Session-level metadata (same for all rows)
        session_metadata = {'instantaneous_time_bin_size_seconds': self.instantaneous_time_bin_size_seconds}
        if self.active_identifying_session_ctx is not None:
            session_ctx_dict = self.active_identifying_session_ctx.to_dict()
            session_metadata.update({f'session_{k}': v for k, v in session_ctx_dict.items()})

        # Process each cell type
        for cell_type, aclus in [('LxC', self.LxC_aclus), ('SxC', self.SxC_aclus), ('AnyC', self.AnyC_aclus)]:
            if aclus is None or len(aclus) == 0:
                continue

            # Get the relevant SpikeRateTrends objects for this cell type
            spike_trends = {
                'replay_delta_minus': getattr(self, f'{cell_type}_ReplayDeltaMinus', None),
                'replay_delta_plus': getattr(self, f'{cell_type}_ReplayDeltaPlus', None),
                'theta_delta_minus': getattr(self, f'{cell_type}_ThetaDeltaMinus', None),
                'theta_delta_plus': getattr(self, f'{cell_type}_ThetaDeltaPlus', None),
            }

            # Extract population-level metadata once per cell type
            pop_metadata = {}
            for condition, trend_obj in spike_trends.items():
                if trend_obj is not None:
                    # Extract common attributes
                    for attr in ['n_epochs', 'total_duration', 'mean_epoch_duration', 'n_cells']:
                        if hasattr(trend_obj, attr):
                            pop_metadata[f'{condition}_{attr}'] = getattr(trend_obj, attr)

                    # Add population statistics
                    if hasattr(trend_obj, 'cell_agg_inst_fr_list'):
                        fr_list = trend_obj.cell_agg_inst_fr_list
                        if len(fr_list) > 0:
                            pop_metadata[f'{condition}_pop_mean'] = np.mean(fr_list)
                            pop_metadata[f'{condition}_pop_std'] = np.std(fr_list)
                            pop_metadata[f'{condition}_pop_n_cells'] = len(fr_list)
                        else:
                            pop_metadata[f'{condition}_pop_mean'] = np.nan
                            pop_metadata[f'{condition}_pop_std'] = np.nan
                            pop_metadata[f'{condition}_pop_n_cells'] = 0
            ## END for condition, trend_obj in spike_trends.items()
            

            # Process each cell
            for i, aclu in enumerate(aclus):
                record = {
                    'aclu': aclu,
                    'cell_type': cell_type,
                    'cell_index_in_type': i,
                    **session_metadata,
                    **pop_metadata  # Add population metadata to each row
                }

                # Add individual cell firing rates
                for condition, spike_trend in spike_trends.items():
                    if spike_trend is not None and hasattr(spike_trend, 'cell_agg_inst_fr_list'):
                        if i < len(spike_trend.cell_agg_inst_fr_list):
                            record[f'{condition}_firing_rate'] = spike_trend.cell_agg_inst_fr_list[i]
                        else:
                            record[f'{condition}_firing_rate'] = np.nan
                    else:
                        record[f'{condition}_firing_rate'] = np.nan
                ## END for condition, spike_trend in spike_trends.items()...
                records.append(record)
            ## END for i, aclu in enumerate(aclus)...
        ## END for cell_type, aclus in [('LxC', self.LxC_aclus), ('SxC', self.SxC_aclus), ('AnyC', self.AnyC_aclus)]...
          
        # Create DataFrame
        df = pd.DataFrame(records)

        # df = df.drop_duplicates(subset=['aclu'], keep='last', ignore_index=True, inplace=False) ## drop any duplicate aclus, keep the last (AnyC version)
        df = df.drop_duplicates(subset=['aclu'], keep='first', ignore_index=True, inplace=False) ## keep the first now, so they are still found in LxC and SxCs

        # Add Fig2 summary statistics
        # Fig2_Replay_FR: [LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus]
        replay_conditions = ['LxC_ReplayDeltaMinus', 'LxC_ReplayDeltaPlus', 'SxC_ReplayDeltaMinus', 'SxC_ReplayDeltaPlus']
        if hasattr(self, 'Fig2_Replay_FR') and self.Fig2_Replay_FR and len(self.Fig2_Replay_FR) == 4:
            for condition, result in zip(replay_conditions, self.Fig2_Replay_FR):
                cell_type = 'LxC' if condition.startswith('LxC') else 'SxC'
                mask = df['cell_type'] == cell_type
                df.loc[mask, f'{condition}_summary_mean'] = result.mean if result.mean is not None else np.nan
                df.loc[mask, f'{condition}_summary_std'] = result.std if result.std is not None else np.nan

                # Add scatter properties if they exist
                for scatter_type, scatter_props in [('LxC_scatter', result.LxC_scatter_props), ('SxC_scatter', result.SxC_scatter_props)]:
                    if scatter_props is not None:
                        for key, value in scatter_props.items():
                            column_name = f'{condition}_{scatter_type}_{key}'
                            if isinstance(value, np.ndarray):
                                if len(value) == mask.sum():
                                    df.loc[mask, column_name] = value
                                else:
                                    df.loc[mask, column_name] = str(value)
                            else:
                                df.loc[mask, column_name] = value

        # Fig2_Laps_FR: [LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus]
        laps_conditions = ['LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus']
        if hasattr(self, 'Fig2_Laps_FR') and self.Fig2_Laps_FR and len(self.Fig2_Laps_FR) == 4:
            for condition, result in zip(laps_conditions, self.Fig2_Laps_FR):
                cell_type = 'LxC' if condition.startswith('LxC') else 'SxC'
                mask = df['cell_type'] == cell_type
                df.loc[mask, f'{condition}_summary_mean'] = result.mean if result.mean is not None else np.nan
                df.loc[mask, f'{condition}_summary_std'] = result.std if result.std is not None else np.nan

                # Add scatter properties if they exist
                for scatter_type, scatter_props in [('LxC_scatter', result.LxC_scatter_props), ('SxC_scatter', result.SxC_scatter_props)]:
                    if scatter_props is not None:
                        for key, value in scatter_props.items():
                            column_name = f'{condition}_{scatter_type}_{key}'
                            if isinstance(value, np.ndarray):
                                if len(value) == mask.sum():
                                    df.loc[mask, column_name] = value
                                else:
                                    df.loc[mask, column_name] = str(value)
                            else:
                                df.loc[mask, column_name] = value

        # Add extended neuron identity columns if session context exists
        if self.active_identifying_session_ctx is not None:
            df = df.neuron_identity.make_neuron_indexed_df_global(
                self.active_identifying_session_ctx, 
                add_expanded_session_context_keys=True, 
                add_extended_aclu_identity_columns=True
            )

        return df

    # ==================================================================================================================================================================================================================================================================================== #
    # From comprehensive dataframe                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    @function_attributes(short_name=None, tags=['FAT_df', 'df', 'UNVALIDATED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-17 16:57', related_items=['cls.get_comprehensive_dataframe'])
    @classmethod
    def from_comprehensive_dataframe(cls, df: pd.DataFrame) -> "InstantaneousSpikeRateGroupsComputation":
        """ Reconstruct InstantaneousSpikeRateGroupsComputation from comprehensive DataFrame.
        INVERSE OF: `cls.get_comprehensive_dataframe`
        Args:
            df: DataFrame created by get_comprehensive_dataframe()

        Returns:
            Reconstructed InstantaneousSpikeRateGroupsComputation instance

        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import InstantaneousSpikeRateGroupsComputation

            # 8. Direct reconstruction from DataFrame
            df = inst_fr_comp.get_comprehensive_dataframe()
            reconstructed_comp2 = InstantaneousSpikeRateGroupsComputation.from_comprehensive_dataframe(df)

            # 9. Compare original and reconstructed objects
            print(f"LxC aclus match: {np.array_equal(inst_fr_comp.LxC_aclus, reconstructed_comp2.LxC_aclus)}")
            print(f"SxC aclus match: {np.array_equal(inst_fr_comp.SxC_aclus, reconstructed_comp2.SxC_aclus)}")


        """
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

        if df.empty:
            return cls(instantaneous_time_bin_size_seconds=0.01)

        # Handle column name synonyms
        df = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df, required_columns_synonym_dict={"cell_type":{'neuron_type',}})

        # Extract scalar session-level metadata
        instantaneous_time_bin_size_seconds = df['instantaneous_time_bin_size_seconds'].iloc[0]

        # Create instance
        instance = cls(instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds)

        # Reconstruct session context
        session_columns = [col for col in df.columns if col.startswith('session_')]
        if session_columns:
            session_data = {}
            for col in session_columns:
                key = col.replace('session_', '')
                session_data[key] = df[col].iloc[0]

            try:
                instance.active_identifying_session_ctx = IdentifyingContext.from_dict(session_data)
            except:
                instance.active_identifying_session_ctx = None
        else:
            instance.active_identifying_session_ctx = None

        # Reconstruct cell arrays
        lxc_mask = df['cell_type'] == 'LxC'
        sxc_mask = df['cell_type'] == 'SxC'

        lxc_df = df[lxc_mask].sort_values('cell_index_in_type') if lxc_mask.any() else pd.DataFrame()
        sxc_df = df[sxc_mask].sort_values('cell_index_in_type') if sxc_mask.any() else pd.DataFrame()

        instance.LxC_aclus = lxc_df['aclu'].values if not lxc_df.empty else np.array([])
        instance.SxC_aclus = sxc_df['aclu'].values if not sxc_df.empty else np.array([])

        # Reconstruct SpikeRateTrends objects
        trend_mappings = {
            'LxC': {
                'replay_delta_minus': 'LxC_ReplayDeltaMinus',
                'replay_delta_plus': 'LxC_ReplayDeltaPlus',
                'theta_delta_minus': 'LxC_ThetaDeltaMinus',
                'theta_delta_plus': 'LxC_ThetaDeltaPlus',
            },
            'SxC': {
                'replay_delta_minus': 'SxC_ReplayDeltaMinus',
                'replay_delta_plus': 'SxC_ReplayDeltaPlus',
                'theta_delta_minus': 'SxC_ThetaDeltaMinus',
                'theta_delta_plus': 'SxC_ThetaDeltaPlus',
            }
        }

        for cell_type in ['LxC', 'SxC']:
            cell_mask = df['cell_type'] == cell_type
            cell_df = df[cell_mask].sort_values('cell_index_in_type') if cell_mask.any() else pd.DataFrame()

            if cell_df.empty:
                # Set all trends to None for this cell type
                for condition, attr_name in trend_mappings[cell_type].items():
                    setattr(instance, attr_name, None)
                continue

            for condition, attr_name in trend_mappings[cell_type].items():
                firing_rate_col = f'{condition}_firing_rate'

                if firing_rate_col in cell_df.columns:
                    firing_rates = cell_df[firing_rate_col].values
                    valid_rates = firing_rates[~pd.isna(firing_rates)]

                    if len(valid_rates) > 0:
                        # Create minimal SpikeRateTrends object
                        trend_obj = SpikeRateTrends.__new__(SpikeRateTrends)
                        trend_obj.cell_agg_inst_fr_list = valid_rates
                        trend_obj.instantaneous_time_bin_size_seconds = instance.instantaneous_time_bin_size_seconds

                        # Extract additional metadata if available
                        metadata_cols = [col for col in cell_df.columns if col.startswith(f'{condition}_')]
                        for col in metadata_cols:
                            attr_name_meta = col.replace(f'{condition}_', '')
                            if attr_name_meta not in ['firing_rate'] and not col.endswith('_firing_rate'):
                                if hasattr(trend_obj, attr_name_meta):
                                    value = cell_df[col].iloc[0]
                                    if not pd.isna(value):
                                        setattr(trend_obj, attr_name_meta, value)

                        setattr(instance, attr_name, trend_obj)
                    else:
                        setattr(instance, attr_name, None)
                else:
                    setattr(instance, attr_name, None)

        # Reconstruct Fig2_Replay_FR
        replay_conditions = ['LxC_ReplayDeltaMinus', 'LxC_ReplayDeltaPlus', 'SxC_ReplayDeltaMinus', 'SxC_ReplayDeltaPlus']
        instance.Fig2_Replay_FR = []

        for condition in replay_conditions:
            mean_col = f'{condition}_summary_mean'
            std_col = f'{condition}_summary_std'

            if mean_col in df.columns and std_col in df.columns:
                mean_val = df[mean_col].iloc[0] if not df[mean_col].isna().all() else None
                std_val = df[std_col].iloc[0] if not df[std_col].isna().all() else None

                # Extract values array from individual cell data
                cell_type = 'LxC' if condition.startswith('LxC') else 'SxC'
                firing_rate_condition = 'replay_delta_minus' if 'DeltaMinus' in condition else 'replay_delta_plus'

                cell_mask = df['cell_type'] == cell_type
                firing_rate_col = f'{firing_rate_condition}_firing_rate'

                if firing_rate_col in df.columns and cell_mask.any():
                    values = df[cell_mask][firing_rate_col].dropna().values
                else:
                    values = np.array([])

                result = SingleBarResult(mean=mean_val, std=std_val, values=values, LxC_aclus=instance.LxC_aclus, SxC_aclus=instance.SxC_aclus, LxC_scatter_props=None, SxC_scatter_props=None)  # Could be reconstructed if needed 
                instance.Fig2_Replay_FR.append(result)
            else:
                # Create empty result
                instance.Fig2_Replay_FR.append(SingleBarResult(mean=None, std=None, values=np.array([]), LxC_aclus=instance.LxC_aclus, SxC_aclus=instance.SxC_aclus, LxC_scatter_props=None, SxC_scatter_props=None))

        # Reconstruct Fig2_Laps_FR
        laps_conditions = ['LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus']
        instance.Fig2_Laps_FR = []

        for condition in laps_conditions:
            mean_col = f'{condition}_summary_mean'
            std_col = f'{condition}_summary_std'

            if mean_col in df.columns and std_col in df.columns:
                mean_val = df[mean_col].iloc[0] if not df[mean_col].isna().all() else None
                std_val = df[std_col].iloc[0] if not df[std_col].isna().all() else None

                # Extract values array
                cell_type = 'LxC' if condition.startswith('LxC') else 'SxC'
                firing_rate_condition = 'theta_delta_minus' if 'DeltaMinus' in condition else 'theta_delta_plus'

                cell_mask = df['cell_type'] == cell_type
                firing_rate_col = f'{firing_rate_condition}_firing_rate'

                if firing_rate_col in df.columns and cell_mask.any():
                    values = df[cell_mask][firing_rate_col].dropna().values
                else:
                    values = np.array([])

                result = SingleBarResult(mean=mean_val, std=std_val, values=values, LxC_aclus=instance.LxC_aclus, SxC_aclus=instance.SxC_aclus, LxC_scatter_props=None, SxC_scatter_props=None)
                instance.Fig2_Laps_FR.append(result)
            else:
                # Create empty result
                instance.Fig2_Laps_FR.append(SingleBarResult(mean=None, std=None, values=np.array([]), LxC_aclus=instance.LxC_aclus, SxC_aclus=instance.SxC_aclus, LxC_scatter_props=None, SxC_scatter_props=None))

        return instance




@function_attributes(short_name=None, tags=['merged', 'firing_rate_indicies', 'multi_result', 'neuron_indexed'], input_requires=[], output_provides=[], uses=['pyphocorehelpers.indexing_helpers.join_on_index'], used_by=[], creation_date='2023-09-12 18:10', related_items=[])
def build_merged_neuron_firing_rate_indicies(curr_active_pipeline, enable_display_intermediate_results=False) -> pd.DataFrame:
    """ 2023-09-12 - TODO - merges firing rate indicies computed in several different computations into a single dataframe for comparison.
    
    Combines [long_short_fr_indicies_df, neuron_replay_stats_df, rate_remapping_df] into a single table
    
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import build_merged_neuron_firing_rate_indicies
        joined_neruon_fri_df = build_merged_neuron_firing_rate_indicies(curr_active_pipeline, enable_display_intermediate_results=False)
        joined_neruon_fri_df
        
    """
    should_add_prefix=False    
    # Requires (curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis'], curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis, curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    
    # 'long_short_fr_indicies_analysis'
    curr_long_short_fr_indicies_analysis = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis'] # 'lsfria'
    _curr_aclus = list(curr_long_short_fr_indicies_analysis['laps_frs_index'].keys()) # extract one set of keys for the aclus
    _curr_frs_indicies_dict = {k:v.values() for k,v in curr_long_short_fr_indicies_analysis.items() if k in ['laps_frs_index', 'laps_inst_frs_index', 'replays_frs_index', 'replays_inst_frs_index', 'non_replays_frs_index', 'non_replays_inst_frs_index']} # extract the values

    long_short_fr_indicies_df = pd.DataFrame(_curr_frs_indicies_dict, index=_curr_aclus)
    if should_add_prefix:
        long_short_fr_indicies_df = long_short_fr_indicies_df.add_prefix('lsfria_')

    # columns_to_prefix = ['laps_frs_index', 'laps_inst_frs_index', 'replays_frs_index', 'replays_inst_frs_index', 'non_replays_frs_index', 'non_replays_inst_frs_index']
    # rename_dict = {col:f'lsfria_{col}' for col in columns_to_prefix}
    # df_prefixed = long_short_fr_indicies_df.rename(columns=rename_dict)
    long_short_fr_indicies_df.index.name = 'aclu'
    long_short_fr_indicies_df = long_short_fr_indicies_df.reset_index()
    # long_short_fr_indicies_df_with_prefix = long_short_fr_indicies_df.add_prefix('lsfria_')
    # Rename specific columns to skip the prefix
    # columns_to_skip = ['aclu']
    # for col in columns_to_skip:
    #     df_with_prefix.rename(columns={f'prefix_{col}': col}, inplace=True)
        
    if enable_display_intermediate_results:
        print(long_short_fr_indicies_df)

    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis # 'jfra'
    neuron_replay_stats_df = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
    if should_add_prefix:
        neuron_replay_stats_df = neuron_replay_stats_df.add_prefix('jfra_')
        neuron_replay_stats_df.rename(columns={'jfra_neuron_type':'neuron_type'}, inplace=True)
    else:
        neuron_replay_stats_df.rename(columns={'aclu':'jfra_aclu'}, inplace=True) # neuron_replay_stats_df already has an explicit 'aclu' column which we need to rename to avoid collisions.

    neuron_replay_stats_df.index.name = 'aclu'
    neuron_replay_stats_df = neuron_replay_stats_df.reset_index()
    

    if enable_display_intermediate_results: 
        print(neuron_replay_stats_df)

    ## Get global 'long_short_post_decoding' results:
    curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding'] # 'lspd'
    rate_remapping_df = deepcopy(curr_long_short_post_decoding.rate_remapping.rr_df[['laps', 'replays',	'skew',	'max_axis_distance_from_center', 'distance_from_center', 'has_considerable_remapping']]) # drops ['neuron_type', 'render_color']
    if should_add_prefix:
        rate_remapping_df = rate_remapping_df.add_prefix('lspd_')
    rate_remapping_df.index.name = 'aclu'
    rate_remapping_df = rate_remapping_df.reset_index() 
    if enable_display_intermediate_results:
        print(rate_remapping_df)


    # suffixes_list = ({'suffixes':('_lsfria', '_jfra')}, )
    # suffixes_list = ((None, '_lsfria'), ('_lsfria', '_jfra'), ('_jfra', '_lspd'))
    suffixes_list = (('_lsfria', '_jfra'), ('_jfra', '_lspd'))

    joined_df = join_on_index(long_short_fr_indicies_df, neuron_replay_stats_df, rate_remapping_df, join_index='aclu', suffixes_list=suffixes_list)
    # joined_df = join_on_index(long_short_fr_indicies_df_with_prefix, neuron_replay_stats_df_with_prefix, rate_remapping_df_with_prefix)

    # joined_df = join_on_index(long_short_fr_indicies_df, neuron_replay_stats_df)

    ## Add the extended neuron identifiers (like the global neuron_uid, session_uid) columns
    joined_df = joined_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=False, add_extended_aclu_identity_columns=False)
    joined_df.drop(columns=['jfra_aclu'], inplace=True)


    return joined_df
