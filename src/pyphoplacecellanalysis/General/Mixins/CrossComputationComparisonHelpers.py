# CrossComputationComparisonHelpers
from typing import Dict, List, Optional, Tuple
import numpy as np
from functools import reduce # _find_any_context_neurons
from enum import Enum, unique # used for `SplitPartitionMembership`
from functools import total_ordering # used for `SplitPartitionMembership`
from attrs import define, field, Factory

from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.dict_representable import override_dict, overriding_dict_with, get_dict_subset
from neuropy.utils.misc import safe_item
import pandas as pd
from pandas import CategoricalDtype

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

from neuropy.utils.colors_util import get_neuron_colors # required for build_neurons_color_map 
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDFConvertableEnum, HDFMixin
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

@total_ordering
@unique
class SplitPartitionMembership(HDFConvertableEnum, Enum):
    """ from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership    
    """
    LEFT_ONLY = 0
    SHARED = 1
    RIGHT_ONLY = 2
    # WHERE IS NEITHER!!?

    def __eq__(self, other):
        return self.value == other.value

    def __le__(self, other):
        return self.value < other.value

    def __hash__(self):
        return hash(self.value)
    
    @classmethod
    def hdf_coding_ClassNames(cls):
        return [cls.LEFT_ONLY.name, cls.SHARED.name, cls.RIGHT_ONLY.name]
    
    # HDFConvertableEnum Conformances ____________________________________________________________________________________ #
    @classmethod
    def get_pandas_categories_type(cls) -> CategoricalDtype:
        return CategoricalDtype(categories=list(cls.hdf_coding_ClassNames()), ordered=True)

    @classmethod
    def convert_to_hdf(cls, value) -> str:
        return value.name

    @classmethod
    def from_hdf_coding_string(cls, string_value: str) -> "SplitPartitionMembership":
        string_value = string_value.lower()
        itemindex = np.where(cls.hdf_coding_ClassNames()==string_value)
        return SplitPartitionMembership(itemindex[0])



@custom_define(slots=False)
class SetPartition_SharedPartitionStructure(HDFMixin, AttrsBasedClassHelperMixin):
    """ DynamicContainer(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map) """
    n_neurons: int = serialized_attribute_field()
    shared_fragile_neuron_IDXs: np.ndarray = serialized_field()
    pairs: List[Tuple[Optional[int], Optional[int]]] = non_serialized_field(is_computable=True)
    missing_neuron_IDXs: np.ndarray = serialized_field()
    missing_neuron_ids: np.ndarray = serialized_field()
    aclu_to_shared_fragile_IDX_map: Dict = non_serialized_field(is_computable=False)


@custom_define(slots=False)
class SetPartition(HDFMixin, AttrsBasedClassHelperMixin):
    """ Converted from a one-off structure produced by `_compare_computation_results` as illustrated below:
        pf_neurons_diff = DynamicContainer(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, intersection=pf_neurons_both, either=pf_neurons_either,
             shared=DynamicContainer(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map)) 

        TODO: perhaps bring computations in here?
    """
    lhs_only: np.ndarray = serialized_field()
    rhs_only: np.ndarray = serialized_field()
    intersection: np.ndarray = serialized_field()
    either: np.ndarray = serialized_field()
    shared: SetPartition_SharedPartitionStructure = serialized_field(alias='shared_structure')

    @property
    def n_neurons(self):
        """The n_neurons property."""
        return len(self.either)



@function_attributes(short_name=None, tags=['compare','private','computation_result'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 20:06', related_items=[])
def _compare_computation_results(lhs_computation_results, rhs_computation_results) -> SetPartition:
    """Computes the differences between two separate computation results, such as those computed for different epochs

    Args:
        lhs_computation_results (_type_): _description_
        rhs_computation_results (_type_): _description_

    Returns:
        _type_: _description_

    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results
        pf_neurons_diff = _compare_computation_results(computation_results.maze1_PYR, computation_results.maze2_PYR)
        pf_neurons_diff

    """
    def _subfn_extract_neuron_ids(computation_results):
        if isinstance(computation_results, ComputationResult):
            neuron_ids = computation_results.computed_data.pf2D.ratemap.neuron_ids
        elif isinstance(computation_results, (DynamicParameters, dict)):
            neuron_ids = computation_results['pf2D'].ratemap.neuron_ids # assume computed_data
        elif isinstance(computation_results, (list, np.ndarray)):
            # assume to be the neuron_ids directly
            neuron_ids = computation_results
        else:
            print(f'ERROR: type(computation_results): {type(computation_results)}, is unhandled')
            raise NotImplementedError
        return neuron_ids

    lhs_neuron_ids = _subfn_extract_neuron_ids(lhs_computation_results)
    rhs_neuron_ids = _subfn_extract_neuron_ids(rhs_computation_results)

    pf_neurons_lhs_unique = np.setdiff1d(lhs_neuron_ids, rhs_neuron_ids) # returns neurons present in lhs that are missing from rhs
    pf_neurons_rhs_unique = np.setdiff1d(rhs_neuron_ids, lhs_neuron_ids) # returns neurons present in rhs that are missing from lhs
    pf_neurons_both = np.intersect1d(rhs_neuron_ids, lhs_neuron_ids) # only those common in both (intersection/AND)

    pf_neurons_either = np.union1d(rhs_neuron_ids, lhs_neuron_ids) # those present in either (union/OR)
    n_neurons = len(pf_neurons_either)
    shared_fragile_neuron_IDXs = np.arange(n_neurons) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69]

    aclu_to_shared_fragile_IDX_map = {aclu:idx for idx, aclu in zip(shared_fragile_neuron_IDXs, pf_neurons_either)} # reverse lookup map from aclu to shared fragile index

    pf_neurons_missing_from_any = np.union1d(pf_neurons_lhs_unique, pf_neurons_rhs_unique) # a list of aclus that are missing from at least one of the inputs (meaning they're unique to one of them). e.g. [2, 4, 8, 13, 19, 105, 109]
    all_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, None) for aclu in pf_neurons_missing_from_any if aclu in pf_neurons_either] # the IDXs of all neurons missing from at least one of the inputs (meaning they're unique to one of them). e.g. [0, 2, 4, 7, 11, 67, 69]
    
    # lhs_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, np.nan) for aclu in pf_neurons_rhs_unique if aclu in aclu_to_shared_fragile_IDX_map]
    # rhs_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, np.nan) for aclu in pf_neurons_lhs_unique if aclu in aclu_to_shared_fragile_IDX_map]

    lhs_shared_IDXs_map: List[Optional[int]] = [safe_item(np.squeeze(np.argwhere(aclu == lhs_neuron_ids)), default=None) for aclu in pf_neurons_either] # [0, 1, None, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
    rhs_shared_IDXs_map: List[Optional[int]] = [safe_item(np.squeeze(np.argwhere(aclu == rhs_neuron_ids)), default=None) for aclu in pf_neurons_either] # [None, 0, 1, 2, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, None, 65, None]

    assert len(rhs_shared_IDXs_map) == n_neurons
    assert len(lhs_shared_IDXs_map) == n_neurons
    shared_fragile_neuron_IDXs_to_pairs: List[Tuple[Optional[int], Optional[int]]] = list(zip(lhs_shared_IDXs_map, rhs_shared_IDXs_map)) # [(0, None), (1, 0), (None, 1), (2, 2), (3, None), (4, 3), (5, 4), (None, 5), (6, 6), (7, 7), (8, 8), (None, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, None), (65, 65), (66, None)]
    pf_neurons_diff = SetPartition(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, intersection=pf_neurons_both, either=pf_neurons_either,
        shared_structure=SetPartition_SharedPartitionStructure(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=np.array(all_missing_IDXs), missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map)
        )
    
    return pf_neurons_diff




def _find_any_context_neurons(*args):
    """Given lists of ids/IDXs are arguments, it finds all unique ids/IDXs present in any of the lists.
    Returns:
        _type_: np.array

    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons
        all_results_neuron_ids_lists = [a_result.computed_data.pf2D.ratemap.neuron_ids for a_result in curr_active_pipeline.computation_results.values()]
        _find_any_context_neurons(*all_results_neuron_ids_lists) # array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 32, 33, 34, 36, 37, 38, 41, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 59])
    """
    # reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    return reduce(np.union1d, tuple(args))

@function_attributes(short_name=None, tags=['color', 'neuron'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 20:07', related_items=[])
def build_neurons_color_map(n_neurons:int, sortby=None, cmap=None):
    """ returns the list of colors, an RGBA np.array of shape: 4 x n_neurons. 
    neurons_colors_array = build_neurons_color_map(n_neurons, sortby=shared_fragile_neuron_IDXs, cmap=None) 
    """
    if sortby is None:
        sort_ind = np.arange(n_neurons)
    elif isinstance(sortby, (list, np.ndarray)):
        # use the provided sort indicies
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    # Use the get_neuron_colors function to generate colors for these neurons
    neurons_colors_array = get_neuron_colors(sort_ind, cmap=cmap)
    return neurons_colors_array


# ==================================================================================================================== #
# Custom Scatterplot Markers with Multi-Color Filling                                                                  #
# ==================================================================================================================== #
from enum import Enum

import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath
from matplotlib.colors import Normalize
# from matplotlib.patches import Circle
from matplotlib.transforms import Bbox



class CustomScatterMarkerMode(Enum):
    """Docstring for CustomScatterMarkerMode."""
    NoSplit = "NoSplit"
    TwoSplit = "TwoSplitMode"
    TriSplit = "TriSplitMode"
    

def _build_neuron_type_distribution_color(rdf):
    """
    # The colors for each point indicating the percentage of participating cells that belong to which track.
        - More long_only -> more red
        - More short_only -> more blue

    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _build_neuron_type_distribution_color

        rdf, (_percent_long_only, _percent_shared, _percent_short_only, _percent_short_long_diff) = _build_neuron_type_distribution_color(rdf)
        rdf
    """
    n_replays = np.shape(rdf)[0]
    _percent_long_only = rdf.num_long_only_neuron_participating.values/rdf.num_neuron_participating.values
    _percent_shared = rdf.num_shared_neuron_participating.values/rdf.num_neuron_participating.values
    _percent_short_only = rdf.num_short_only_neuron_participating.values/rdf.num_neuron_participating.values
    # list(zip(_percent_long_only, _percent_shared, _percent_short_only)) # each replay's colors add up to 1.0

    # RGB Mode:
    colors_mat = np.zeros((n_replays, 3))
    # reds_mat = np.repeat(np.atleast_2d([1.0, 0.0, 0.0]), axis=0, repeats=n_replays)
    colors_mat[:,0] = _percent_long_only
    # greens_mat = np.repeat(np.atleast_2d([0.0, 1.0, 0.0]), axis=0, repeats=n_replays)
    # colors_mat[:,1] = _percent_shared
    colors_mat[:,1] = 0.0 # just to make it brighter
    # blues_mat = np.repeat(np.atleast_2d([0.0, 0.0, 1.0]), axis=0, repeats=n_replays)
    colors_mat[:,2] = _percent_short_only
    # colors_mat = reds_mat + greens_mat + blues_mat
    
    # Scalar Output Mode
    # _percent_short_long_diff = _percent_short_only - _percent_long_only
    # _short_long_balance_diff = _percent_short_only - _percent_long_only / (_percent_short_only + _percent_long_only) ## Working but (0, 1, 1) would clip to 0.5 despite (1, 13, 0) going all the way down to -1.0
    _long_to_short_balances = (rdf.num_short_only_neuron_participating.values - rdf.num_long_only_neuron_participating.values) / (rdf.num_short_only_neuron_participating.values + rdf.num_long_only_neuron_participating.values) ## Working but (0, 1, 1) would clip to 0.5 despite (1, 13, 0) going all the way down to -1.0
    # those where (_percent_short_only + _percent_long_only) now have NaN, so call np.nan_to_num(_percent_short_long_diff) to replace these NaNs with zeros to indicate that they are perfectly balanced
    _long_to_short_balances = np.nan_to_num(_long_to_short_balances)
    # colors_mat = _percent_short_long_diff
    rdf['neuron_type_distribution_color_RGB'] = colors_mat.tolist()
    rdf['neuron_type_distribution_color_scalar'] = _long_to_short_balances.tolist()    
    return rdf, (_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances)




ch_rng = lambda x: 2.0*x-1.0 # map from [0, 1] -> [-1, 1] (the space where the paths are built)
# validation: [ch_rng(v) for v in [0, 0.25, 0.5, 0.75, 1.0]] # [-1.0, -0.5, 0.0, 0.5, 1.0]

def build_replays_custom_scatter_markers(rdf, marker_split_mode=CustomScatterMarkerMode.TriSplit, debug_print=False):
    """ Builds all custom scatter markers for the rdf (replay dataframe), one for each replay
    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_replays_custom_scatter_markers, build_custom_scatter_marker
        
        out_plot_kwargs_array = build_replays_custom_scatter_markers(rdf, debug_print=False)
        out_plot_kwargs_array

    Implementation:
        Internally calls `_subfn_build_custom_scatter_marker(...)` for each marker
    """
    n_replays = np.shape(rdf)[0]
    _percent_long_only = rdf.num_long_only_neuron_participating.values/rdf.num_neuron_participating.values
    _percent_shared = rdf.num_shared_neuron_participating.values/rdf.num_neuron_participating.values
    _percent_short_only = rdf.num_short_only_neuron_participating.values/rdf.num_neuron_participating.values

    _long_to_short_balances = (rdf.num_short_only_neuron_participating.values - rdf.num_long_only_neuron_participating.values) / (rdf.num_short_only_neuron_participating.values + rdf.num_long_only_neuron_participating.values) ## Working but (0, 1, 1) would clip to 0.5 despite (1, 13, 0) going all the way down to -1.0
    # those where (_percent_short_only + _percent_long_only) now have NaN, so call np.nan_to_num(_percent_short_long_diff) to replace these NaNs with zeros to indicate that they are perfectly balanced
    _long_to_short_balances = np.nan_to_num(_long_to_short_balances)

    # custom_markers_dict_list = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # scatter_plot_kwargs_list, scatter_markerstyles_list, scatter_marker_paths_list = custom_markers_dict_list['plot_kwargs'], custom_markers_dict_list['markerstyles'], custom_markers_dict_list['paths'] # Extract variables from the `custom_markers_dict_list` dictionary to the local workspace

    custom_markers_tuple_list = [_subfn_build_custom_scatter_marker(long, shared, short, long_to_short_balance, marker_split_mode=marker_split_mode, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    

    out_plot_kwargs_list = [a_tuple[0] for a_tuple in custom_markers_tuple_list]
    # out_plot_kwargs_array = [build_custom_scatter_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False)[0] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # out_plot_kwargs_array

    # scatter_markerstyles_list = [a_tuple[1] for a_tuple in custom_markers_tuple_list]
    # # Break into two parts <list<tuple[2]<MarkerStyle>> -> list<MarkerStyle>, List<MarkerStyle>
    # scatter_markerstyles_0_list = [a_tuple[0] for a_tuple in scatter_markerstyles_list]
    # scatter_markerstyles_1_list = [a_tuple[1] for a_tuple in scatter_markerstyles_list]

    # out_scatter_marker_paths_array = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False)[1] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # out_scatter_marker_paths_array

    return out_plot_kwargs_list
    # return (scatter_markerstyles_0_list, scatter_markerstyles_1_list)


def _subfn_build_custom_scatter_marker(long, shared, short, long_to_short_balance, marker_split_mode=CustomScatterMarkerMode.TriSplit, debug_print=False):
    """ Builds a single custom scatterplot marker representing a replay and its distribution of (long_only, both, short_only) cells 
    Usage:
    
        n_replays = np.shape(rdf)[0]
        _percent_long_only = rdf.num_long_only_neuron_participating.values/rdf.num_neuron_participating.values
        _percent_shared = rdf.num_shared_neuron_participating.values/rdf.num_neuron_participating.values
        _percent_short_only = rdf.num_short_only_neuron_participating.values/rdf.num_neuron_participating.values
        _long_to_short_balances = rdf['neuron_type_distribution_color_scalar'].values

        out_plot_kwargs, out_paths = _build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False)

        _long_to_short_balances = rdf['neuron_type_distribution_color_scalar'].values


        # custom_markers_dict_list = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
        # scatter_plot_kwargs_list, scatter_markerstyles_list, scatter_marker_paths_list = custom_markers_dict_list['plot_kwargs'], custom_markers_dict_list['markerstyles'], custom_markers_dict_list['paths'] # Extract variables from the `custom_markers_dict_list` dictionary to the local workspace

        custom_markers_tuple_list = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
        scatter_markerstyles_list = [a_tuple[1] for a_tuple in custom_markers_tuple_list]

        # Break into two parts <list<tuple[2]<MarkerStyle>> -> list<MarkerStyle>, List<MarkerStyle>
        scatter_markerstyles_0_list = [a_tuple[0] for a_tuple in scatter_markerstyles_list]
        scatter_markerstyles_1_list = [a_tuple[1] for a_tuple in scatter_markerstyles_list]

        len(scatter_markerstyles_0_list) # 743 - n_replays
        # out_plot_kwargs_array = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False)[0] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
        # out_plot_kwargs_array

        # out_scatter_marker_paths_array = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False)[1] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
        # out_scatter_marker_paths_array


    """
    def _subfn_build_marker_path(clip_box=None):
        """ old version that built a marker path for a specific replay idx """
        if clip_box is None:
            # first tuple : coords of box' bottom-left corner, from the figure's bottom-left corner
            # second tuple : coords of box' top-right corner, from the figure's bottom-left corner
            # clip_box = Bbox(((0,0),(0.5, 1.0)))
            clip_box = Bbox(((-1.0, -1.0),(0.0, 1.0))) # half, because the unit paths are centered at (0, 0)

        # marker1 = clipping_rectangle
        # marker2 = clipped_rect

        marker1 = mpath.Path.unit_circle()
        # marker2 = mpath.Path.unit_circle_righthalf()

        ## Clipped circle:
        marker2 = mpath.Path.unit_circle()
        marker2 = marker2.clip_to_bbox(bbox=clip_box, inside=True)

        # concatenate the circle with an internal cutout of the star
        out_path = mpath.Path(
            vertices=np.concatenate([marker1.vertices, marker2.vertices[::-1, ...]]),
            codes=np.concatenate([marker1.codes, marker2.codes]))
        return out_path

    ################ BEGIN FUNCTION BODY
    
    # Tri-split mode:
    if marker_split_mode.value == CustomScatterMarkerMode.TriSplit.value:
        # colors = ['r','g','b'] # should these be reversed too since the two-split version are? I assume so.
        colors = ['b','g','g'] # should these be reversed too since the two-split version are? I assume so.
        cum_long, cum_shared, cum_short = long, (long+shared), (long+shared+short)
        if debug_print:
            print(f'cum_long: {cum_long}, cum_shared: {cum_shared}, cum_short: {cum_short}')
        clip_bboxs = (Bbox(((-1.0, -1.0),(ch_rng(cum_long), 1.0))),
              Bbox(((ch_rng(cum_long), -1.0),(ch_rng(cum_shared), 1.0))),
              Bbox(((ch_rng(cum_shared), -1.0),(ch_rng(cum_short), 1.0))))
        
    elif marker_split_mode.value == CustomScatterMarkerMode.TwoSplit.value:
        # Two-split mode:
        # colors = ['r','b'] # OLD: noted that I had to reverse these to make the plot correct, but I'm honestly not sure why. I'm thinking it has to do with the order they are plotted or something
        colors = ['b','r']
        if debug_print:
            print(f'long_to_short_balance: {long_to_short_balance}')
        clip_bboxs = (Bbox(((-1.0, -1.0),(long_to_short_balance, 1.0))),
                      Bbox(((long_to_short_balance, -1.0),(1.0, 1.0))), )
    else:
        # No-split mode:
        colors = ['g']
        if debug_print:
            print(f'long_to_short_balance: {long_to_short_balance}')
        clip_bboxs = (Bbox(((-1.0, -1.0),(1.0, 1.0))), )

    if debug_print:
        print(f'clip_bboxs: {clip_bboxs}')

    # out_paths = [_subfn_build_marker_path(clip_bbox) for i, clip_bbox in enumerate(clip_bboxs)]
    out_paths = [mpath.Path.unit_circle().clip_to_bbox(bbox=clip_bbox, inside=True) for i, clip_bbox in enumerate(clip_bboxs)]
    # Rotate the markers by 45 degrees
    t = Affine2D().rotate_deg(45)     # ax.plot(x, 0, marker=MarkerStyle('o', 'left', t), **common_style)    
    out_markerstyles = [MarkerStyle(mpath.Path.unit_circle().clip_to_bbox(bbox=clip_bbox, inside=True), 'full', t) for i, clip_bbox in enumerate(clip_bboxs)]
    out_plot_kwargs = [dict(marker=m, color=colors[i], linestyle='none') for i, m in enumerate(out_markerstyles)]

    return (out_plot_kwargs, out_markerstyles, out_paths)

##########################
