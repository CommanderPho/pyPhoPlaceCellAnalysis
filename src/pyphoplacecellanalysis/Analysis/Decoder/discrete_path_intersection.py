from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING


# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from enum import Enum
from neuropy.analyses.decoders import BinningContainer
from neuropy.utils.result_context import IdentifyingContext
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
import nptyping as ND
from nptyping import NDArray
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder #typehinting only


# @define(slots=False, eq=False)
@metadata_attributes(short_name=None, tags=['position', '2D', 'path', 'self-intersection', 'self-avoidance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-04-10 05:08', related_items=[])
class PathIntersectionDetection:
    """ 

    Usage:
        import numpy as np
        from numpy.typing import NDArray
        from pyphoplacecellanalysis.Analysis.Decoder.discrete_path_intersection import PathIntersectionDetection

        # min_n_intra_intersect_time_steps: int = 10
        min_n_intra_intersect_time_steps: int = 100
        max_t_idx: Optional[int] = None
        min_included_duration: Optional[float] = 1.0 ## at least one sec per loop
        intersect_pairs_df, pos_path, pos_df = PathIntersectionDetection.run_all(a_decoder=a_decoder, 
                                        min_n_intra_intersect_time_steps=min_n_intra_intersect_time_steps, min_included_duration=min_included_duration,
                                    )

        ## Filter to only good:
        included_only_intersect_pairs_df: pd.DataFrame = intersect_pairs_df[intersect_pairs_df['is_included_filtered']].drop(columns=['is_included_filtered'], inplace=False)

        # Sort by columns: 'duration' (descending), 'first_t' (ascending)
        max_t_idx: int = 2
        included_only_intersect_pairs_df = included_only_intersect_pairs_df.sort_values(['duration', 'first_t'], ascending=[False, True]).head(max_t_idx)

        ## INPUTS: included_only_intersect_pairs_df, binned_pos_idx_column_labels
        img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D = PathIntersectionDetection.build_img_representations(a_decoder=a_decoder, intersect_pairs_df=included_only_intersect_pairs_df, max_t_idx=max_t_idx)

        fig, imv = ComputerVisionComputations.imshow(RGBA_img, xbin_edges=a_decoder.xbin, ybin_edges=a_decoder.ybin)


    """
    start_repeat_pair_idx_columns = ['first_index', 'repeated_index']
    binned_pos_column_labels = ['binned_x', 'binned_y']

    # binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray] = field(factory=dict)
    # n_powers: int = field(default=20)
    # time_bin_size: float = field(default=None)
    # pos_bin_size: float = field(default=None)
    

    @classmethod
    def self_intersection_pairs_and_coords(cls, positions: NDArray[np.number]) -> tuple[NDArray[np.int64], NDArray[np.number]]:
        """
        Find every self-intersection in a 2D trajectory.

        Parameters
        ----------
        positions : array, shape (n_time_bins, 2)
            positions[t] = [x_t, y_t]

        Returns
        -------
        pairs : array, shape (n_pairs, 2), dtype int64
            Each row is (i, j) with i < j and positions[i] == positions[j].
        coords : array, shape (n_pairs, 2)
            The repeated coordinate for each pair.
        """
        pos = np.asarray(positions)

        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (n_time_bins, 2)")

        if pos.shape[0] < 2:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=pos.dtype)

        # Optional validation for your stated transition constraint:
        # successive bins must differ by at most 1 in each axis.
        if np.any(np.abs(np.diff(pos, axis=0)) > 1):
            raise ValueError(
                "Found an invalid transition; successive bins must lie in {-1, 0, +1} "
                "for each axis."
            )

        # Single-pass grouping of indices by coordinate.
        seen: dict[tuple[float, float], list[int]] = {}
        pair_list: list[tuple[int, int]] = []
        coord_list: list[tuple[float, float]] = []

        for i in range(pos.shape[0]):
            x = pos[i, 0].item()
            y = pos[i, 1].item()
            key = (x, y)

            prev = seen.get(key)
            if prev is None:
                seen[key] = [i]
                continue

            # Every previous visit to the same coordinate forms a self-intersection pair.
            for j in prev:
                pair_list.append((j, i))
                coord_list.append(key)

            prev.append(i)

        if not pair_list:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=pos.dtype)

        return (
            np.asarray(pair_list, dtype=np.int64),
            np.asarray(coord_list, dtype=pos.dtype),
        )


    @classmethod
    def add_intersect_pairs_df_is_filtered_column(cls, intersect_pairs_df: pd.DataFrame, min_n_intra_intersect_time_steps: int = 100, max_t_idx: Optional[int] = None, min_duration: Optional[float]=None) -> pd.DataFrame:
        """ 

        Usage:
            min_n_intra_intersect_time_steps: int = 100
            max_t_idx: Optional[int] = None
            intersect_pairs_df = add_intersect_pairs_df_is_filtered_column(intersect_pairs_df=intersect_pairs_df, min_n_intra_intersect_time_steps=min_n_intra_intersect_time_steps, max_t_idx=max_t_idx)
            intersect_pairs_df

        """    
        if min_n_intra_intersect_time_steps is not None:
            non_adjacent_only_intersection_indicies = (intersect_pairs_df['num_intra_intersect_time_steps'] >= min_n_intra_intersect_time_steps)
            included_intersection_point_indicies = non_adjacent_only_intersection_indicies ## only non-adjacent included
        else:
            included_intersection_point_indicies = np.ones(shape=(len(intersect_pairs_df),), dtype=bool).astype(bool) ## all included

        
        if max_t_idx is not None:
            included_intersection_point_indicies = np.logical_and(included_intersection_point_indicies,
                                                                    np.all((intersect_pairs_df[start_repeat_pair_idx_columns].to_numpy() < max_t_idx), axis=1), ## included intersection points
                                                                )

        if min_duration is not None:
            included_intersection_point_indicies = np.logical_and(included_intersection_point_indicies, (intersect_pairs_df['duration'] >= min_duration))


        intersect_pairs_df['is_included_filtered'] = included_intersection_point_indicies
        return intersect_pairs_df


    @classmethod
    def build_img_representations(cls, a_decoder, intersect_pairs_df: pd.DataFrame, max_t_idx: int = 1005, **kwargs): # pos_path: Union[NDArray, pd.DataFrame], 
        """ 
        max_t_idx: int = 1005 # shows all t_bins previous to this

        Usage:

            max_t_idx: int = 1005

            ## INPUTS: a_decoder, intersect_pairs_df, max_t_idx
            included_only_intersect_pairs_df: pd.DataFrame = intersect_pairs_df[intersect_pairs_df['is_included_filtered']].drop(columns=['is_included_filtered'], inplace=False)
            img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D = PathIntersectionDetection.build_img_representations(a_decoder=a_decoder, intersect_pairs_df=included_only_intersect_pairs_df, max_t_idx=max_t_idx)
            fig, imv = ComputerVisionComputations.imshow(RGBA_img, xbin_edges=a_decoder.xbin, ybin_edges=a_decoder.ybin)

        """
        binned_pos_column_labels = kwargs.get('binned_pos_column_labels', cls.binned_pos_column_labels)
        assert len(binned_pos_column_labels) > 0
        binned_pos_idx_column_labels = kwargs.get('binned_pos_idx_column_labels', [f'{k}_idx' for k in binned_pos_column_labels]) # ['binned_x_idx', 'binned_y_idx']
        assert len(binned_pos_idx_column_labels) > 0
        
        ## Build Image Representations:
        n_x_bins: int = len(a_decoder.xbin) - 1
        n_y_bins: int = len(a_decoder.ybin) - 1

        img_arr = np.zeros(shape=(n_x_bins, n_y_bins), dtype=np.int8)
        RGBA_img = np.zeros(shape=(n_x_bins, n_y_bins, 4), dtype=np.float)
        RGBA_img[:, :, -1] = 1.0 ## full alpha

        # assert (not isinstance(pos_path, pd.DataFrame)), f" should not be dataframe!"
        assert (isinstance(intersect_pairs_df, pd.DataFrame)), f" must be a dataframe!"
        
        pos_path = intersect_pairs_df[binned_pos_idx_column_labels].astype(int).to_numpy() # np.shape(pos_path) # (331555, 2)
        num_included_intersections: int = len(intersect_pairs_df)
        # pos_path = (change_only_pos_df[['binned_x', 'binned_y']].astype(int)-1).to_numpy() # np.shape(pos_path) # (331555, 2)

        # max_t_idx: int = 105 # shows all t_bins previous to this
        # max_t_idx: int = 1005 # shows all t_bins previous to this

        ## 3D version
        # img_3D_dtype = np.bool8
        img_3D_dtype = np.float32
        img_arr_3D = np.zeros(shape=(n_x_bins, n_y_bins, max_t_idx), dtype=img_3D_dtype)
        img_intersection_arr_3D = np.zeros(shape=(n_x_bins, n_y_bins, max_t_idx), dtype=img_3D_dtype)

        for t_idx in np.arange(max_t_idx):
            a_x_idx, a_y_idx = pos_path[t_idx, :]
            img_arr[a_x_idx, a_y_idx] += 1
            img_arr_3D[a_x_idx, a_y_idx, t_idx] += 1
            RGBA_img[a_x_idx, a_y_idx, 1] += 1.0 ## 2nd (green) channel only
            ## add intersection points:
            # intersect_index_pairs


        # Intersections Loop _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        ## rebuild from the `intersect_pairs_df: pd.DataFrame`:
        # included_intersection_point_indicies = np.where(intersect_pairs_df['is_included_filtered'].to_numpy())[0]
        included_intersection_pairs = intersect_pairs_df[['first_index', 'repeated_index']].to_numpy()
        included_intersect_position_pairs = intersect_pairs_df[['binned_x_idx', 'binned_y_idx']].to_numpy()
        # num_included_intersections: int = np.shape(included_intersection_pairs)[0]
        included_intersect_point_t_bin_idxs = intersect_pairs_df['intersect_pair_idx'].to_numpy()


        for i in np.arange(num_included_intersections):
            curr_intersection_idx_pair = included_intersection_pairs[i, :]
            curr_intersection_pos_location = included_intersect_position_pairs[i, :]
            a_x_idx, a_y_idx = curr_intersection_pos_location
            a_t_idx = included_intersect_point_t_bin_idxs[i]
            # print(f'a_t_idx: {a_t_idx}')
            if a_t_idx < max_t_idx:
                img_intersection_arr_3D[a_x_idx, a_y_idx, a_t_idx] += 1
            
            RGBA_img[a_x_idx, a_y_idx, 0] += 1.0

        ## OUTPUTS: img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D
        # return img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D
        ## OUTPUTS: img_arr_3D, img_intersection_arr_3D
        return img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D


    @classmethod
    def run_all(cls, a_decoder: BasePositionDecoder,
            min_n_intra_intersect_time_steps: Optional[int] = 100, max_t_idx: Optional[int] = None, min_included_duration: Optional[float] = None, ## filter parameters
            binned_pos_column_labels = ['binned_x', 'binned_y'],
            # should_build_img_reps: bool=False,
            **kwargs):
        """ runs all

        Usage:

            import numpy as np
            from numpy.typing import NDArray
            from pyphoplacecellanalysis.Analysis.Decoder.discrete_path_intersection import PathIntersectionDetection

            # min_n_intra_intersect_time_steps: int = 10
            min_n_intra_intersect_time_steps: int = 100
            max_t_idx: Optional[int] = None
            min_included_duration: Optional[float] = 1.0 ## at least one sec per loop
            intersect_pairs_df, pos_path, pos_df = PathIntersectionDetection.run_all(a_decoder=a_decoder, 
                                            min_n_intra_intersect_time_steps=min_n_intra_intersect_time_steps, min_included_duration=min_included_duration,
                                        )


            intersect_pairs_df


        """

        ## INPUTS: a_decoder,

        # pf2D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf2D'])
        
        binned_pos_idx_column_labels = [f'{k}_idx' for k in binned_pos_column_labels] # ['binned_x_idx', 'binned_y_idx']
        pos_df = a_decoder.pf.filtered_pos_df.dropna(axis='index', how='any', subset=binned_pos_column_labels)

        # n_time_bins: int = len(pos_df)
        # pos_path = (pos_df[binned_pos_column_labels].astype(int)-1).to_numpy()
        # np.shape(pos_path) # (331555, 2)

        # pos_path_xy_tuples = [(pos_path[i, 0], pos_path[i, 1]) for i in np.arange(n_time_bins)]
        # binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix_2d(a_decoder.xbin_labels, a_decoder.ybin_labels, *[(pos_df[k].to_numpy() - 1) for k in binned_pos_column_labels]) 

        all_desired_columns = ['t'] + binned_pos_column_labels
        pos_bin_point_change_indicies = np.where((pos_df[binned_pos_column_labels].diff().abs() > 0).any(axis=1))[0] ## find indicies where binned positions actually change
        pos_bin_point_change_indicies

        change_only_pos_df: pd.DataFrame = pos_df.iloc[pos_bin_point_change_indicies][all_desired_columns]
        change_only_pos_df[binned_pos_idx_column_labels] = (change_only_pos_df[binned_pos_column_labels].astype(int)-1).to_numpy()
        change_only_pos_df = change_only_pos_df.reset_index(drop=False).rename(columns={'index': 'pos_df_idx'}, inplace=False)
        change_only_pos_df['change_idx'] = change_only_pos_df.index.to_numpy().astype(int)
        n_change_only_time_steps: int = len(change_only_pos_df)
        print(f'n_change_only_time_steps: {n_change_only_time_steps}')
        change_only_pos_df
        # pos_df.iloc[pos_bin_point_change_indicies]

        # pos_bin_point_change_indicies


        ## INPUTS: change_only_pos_df, 
        # pos_path = (pos_df[['binned_x', 'binned_y']].astype(int)-1).to_numpy()
        # pos_path = (change_only_pos_df[['binned_x', 'binned_y']].astype(int)-1).to_numpy() # np.shape(pos_path) # (331555, 2)
        pos_path = change_only_pos_df[binned_pos_idx_column_labels].astype(int).to_numpy() # np.shape(pos_path) # (331555, 2)
        # n_time_bins: int = np.shape(pos_path)[0]
        (intersect_index_pairs, intersect_position_pairs) = cls.self_intersection_pairs_and_coords(positions=pos_path)
        num_intra_intersect_time_steps = np.squeeze(np.diff(intersect_index_pairs, axis=1)) ## num time steps between first and later intersections
        ## OUTPUTS: intersect_index_pairs, 
        ## INPUTS: intersect_index_pairs, num_intra_intersect_time_steps

        # plt.hist(num_intra_intersect_time_steps, bins=25)
        if min_n_intra_intersect_time_steps is not None:
            non_adjacent_only_intersection_indicies = (num_intra_intersect_time_steps >= min_n_intra_intersect_time_steps)
            included_intersection_point_indicies = non_adjacent_only_intersection_indicies ## only non-adjacent included

        else:
            included_intersection_point_indicies = np.ones_like(num_intra_intersect_time_steps).astype(bool) ## all included

        max_t_idx = None
        if max_t_idx is not None:
            included_intersection_point_indicies = np.logical_and(included_intersection_point_indicies,
                                                                    np.all((intersect_index_pairs < max_t_idx), axis=1), ## included intersection points
                                                                )

        ## INPUTS: included_intersection_point_indicies
        ## filter by included
        # included_intersection_pairs = intersect_index_pairs[included_intersection_point_indicies, :]
        # included_intersect_position_pairs = intersect_position_pairs[included_intersection_point_indicies, :]
        # num_included_intersections: int = np.shape(included_intersection_pairs)[0]
        # included_intersect_point_t_bin_idxs = np.where(included_intersection_point_indicies)[0]
        # included_intersect_point_t_bin_idxs

        ## OUTPUTS: intersect_index_pairs, intersect_position_pairs, included_intersection_point_indicies, included_intersection_pairs, included_intersect_position_pairs, included_intersect_point_t_bin_idxs

        intersect_pairs_df: pd.DataFrame = pd.DataFrame({'first_index': intersect_index_pairs[:, 0], 'repeated_index': intersect_index_pairs[:, 1], 'binned_x_idx': intersect_position_pairs[:, 0], 'binned_y_idx': intersect_position_pairs[:, 1], 'num_intra_intersect_time_steps': num_intra_intersect_time_steps}) # , 't_bin_idx': included_intersect_point_t_bin_idxs
        assert len(num_intra_intersect_time_steps) == len(intersect_pairs_df)
        intersect_pairs_df['intersect_pair_idx'] = intersect_pairs_df.index.to_numpy().astype(int)

        # ## rebuild from the `intersect_pairs_df: pd.DataFrame`:
        # included_intersection_pairs = intersect_pairs_df[['first_index', 'repeated_index']].to_numpy()
        # included_intersect_position_pairs = intersect_pairs_df[['binned_x_idx', 'binned_y_idx']].to_numpy()
        # num_included_intersections: int = np.shape(included_intersection_pairs)[0]
        # included_intersect_point_t_bin_idxs = intersect_pairs_df['intersect_pair_idx'].to_numpy()
        

        # intersect_pairs_df['intersect_pair_idx'].to_numpy().astype(int)
        # change_only_pos_df.iloc[intersect_pairs_df['intersect_pair_idx'].to_numpy().astype(int)]['t']
        # intersect_pairs_df['intersect_t'] = change_only_pos_df.iloc[intersect_pairs_df['intersect_pair_idx'].to_numpy().astype(int)]['t']
        intersect_pairs_df['first_t'] = change_only_pos_df.iloc[intersect_pairs_df['first_index'].to_numpy().astype(int)]['t'].to_numpy()
        intersect_pairs_df['repeat_t'] = change_only_pos_df.iloc[intersect_pairs_df['repeated_index'].to_numpy().astype(int)]['t'].to_numpy()
        intersect_pairs_df['duration'] = intersect_pairs_df['repeat_t'] - intersect_pairs_df['first_t']
        ## pos_df idx for easy positions access
        intersect_pairs_df['first_pos_df_idx'] = change_only_pos_df.iloc[intersect_pairs_df['first_index'].to_numpy().astype(int)]['pos_df_idx'].to_numpy()
        intersect_pairs_df['repeat_pos_df_idx'] = change_only_pos_df.iloc[intersect_pairs_df['repeated_index'].to_numpy().astype(int)]['pos_df_idx'].to_numpy()

        intersect_pairs_df = cls.add_intersect_pairs_df_is_filtered_column(intersect_pairs_df=intersect_pairs_df, min_n_intra_intersect_time_steps=min_n_intra_intersect_time_steps, max_t_idx=max_t_idx, min_duration=min_included_duration)
    
        # # EXTRA/not-needed ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # included_only_intersect_pairs_df: pd.DataFrame = intersect_pairs_df[intersect_pairs_df['is_included_filtered']].drop(columns=['is_included_filtered'], inplace=False)
        # ## OUTPUTS: included_only_intersect_pairs_df
        # included_only_intersect_pairs_df


        # # Sort by columns: 'duration' (descending), 'first_t' (ascending)
        # max_t_idx: int = 2
        # included_only_intersect_pairs_df = included_only_intersect_pairs_df.sort_values(['duration', 'first_t'], ascending=[False, True]).head(max_t_idx)
        # included_only_intersect_pairs_df


        # ## INPUTS: included_only_intersect_pairs_df, binned_pos_idx_column_labels
        # # max_t_idx: int = 1005
        # # binned_pos_idx_column_labels = ['binned_x_idx', 'binned_y_idx']
        # pos_path = included_only_intersect_pairs_df[binned_pos_idx_column_labels].astype(int).to_numpy() # np.shape(pos_path) # (331555, 2)
        # img_arr, RGBA_img, img_arr_3D, img_intersection_arr_3D = cls.build_img_representations(a_decoder=a_decoder, pos_path=pos_path, max_t_idx=max_t_idx)

        # fig, imv = ComputerVisionComputations.imshow(RGBA_img, xbin_edges=a_decoder.xbin, ybin_edges=a_decoder.ybin)
        # # fig, imv = ComputerVisionComputations.imshow(img, xbin_edges=a_decoder.xbin, ybin_edges=a_decoder.ybin)


        return intersect_pairs_df, pos_path, pos_df
