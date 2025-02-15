from copy import deepcopy
import param
import numpy as np
import pandas as pd
from attrs import define, field, Factory
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlotsData, VisualizationParameters

from pyphocorehelpers.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import NeuronConfigOwningMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_placefields2D, update_plotColorsPlacefield2D

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

# ==================================================================================================================== #
# 2024-04-12 - Decoded Trajectory Plotting on Maze (1D & 2D) - Posteriors and Most Likely Position Paths               #
# ==================================================================================================================== #

from itertools import islice
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import LapsVisualizationMixin, LineCollection, _plot_helper_add_arrow # plot_lap_trajectories_2d

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


class SingleArtistMultiEpochBatchHelpers:
    """ 
    
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import SingleArtistMultiEpochBatchHelpers
    """
    @function_attributes(short_name=None, tags=['reshape', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 05:57', related_items=[])
    @classmethod
    def reshape_p_x_given_n_for_single_artist_display(cls, updated_timebins_p_x_given_n: NDArray, rotate_to_vertical: bool = True, should_expand_first_dim: bool=True, debug_print=False) -> NDArray:
        """ 
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import reshape_p_x_given_n_for_single_artist_display
        
        """
        stacked_p_x_given_n = deepcopy(updated_timebins_p_x_given_n) # drop the last epoch
        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (76, 40, 33008)
        stacked_p_x_given_n = np.moveaxis(stacked_p_x_given_n, -1, 0) # move the n_t dimension/axis (which starts as last) to be first (0th)
        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (33008, 76, 40)

        n_xbins, n_ybins, n_tbins = np.shape(stacked_p_x_given_n) # (76, 40, 29532)        
        if not rotate_to_vertical:
            stacked_p_x_given_n = np.row_stack(stacked_p_x_given_n) # .shape: (99009, 39) - ((n_xbins*n_tbins), n_ybins)
            # stacked_p_x_given_n = np.swapaxes(stacked_p_x_given_n, 1, 2).reshape((-1, n_ybins))
        else:
            ## display with y-axis along the primary axis=1
            stacked_p_x_given_n = np.column_stack(stacked_p_x_given_n) # .shape: (n_xbins, (n_ybins*n_tbins))
            stacked_p_x_given_n = stacked_p_x_given_n.T.T
            # stacked_p_x_given_n = stacked_p_x_given_n.reshape(stacked_p_x_given_n.shape[0], stacked_p_x_given_n.shape[1] * stacked_p_x_given_n.shape[2]) # .shape: (n_xbins, (n_ybins*n_tbins))

        if debug_print:
            print(np.shape(stacked_p_x_given_n)) # (2508608, 40)
            
        if should_expand_first_dim:
            stacked_p_x_given_n = np.expand_dims(stacked_p_x_given_n, axis=0)
            if debug_print:
                print(np.shape(stacked_p_x_given_n)) # (1, 2508608, 40)
        return stacked_p_x_given_n

    @classmethod
    def _slice_to_epoch_range(cls, flat_timebins_p_x_given_n, flat_time_bin_centers, desired_epoch_start_idx: int = 0, desired_epoch_end_idx: int = 15):
        """ trims down to a specific epoch range """
        flat_timebins_p_x_given_n = flat_timebins_p_x_given_n[:, :, desired_epoch_start_idx:desired_epoch_end_idx]
        flat_time_bin_centers = flat_time_bin_centers[desired_epoch_start_idx:desired_epoch_end_idx]
        return flat_timebins_p_x_given_n, flat_time_bin_centers


    @classmethod
    def complete_build_stacked_flat_arrays(cls, a_result, a_new_global_decoder, desired_epoch_start_idx:int=0, desired_epoch_end_idx: Optional[int] = None, rotate_to_vertical: bool = True, should_expand_first_dim: bool=True):
        """ 
        a_result: DecodedFilterEpochsResult = subdivided_epochs_specific_decoded_results_dict['global']
        a_new_global_decoder = new_decoder_dict['global']
        # delattr(a_result, 'measured_positions_list')
        a_result.measured_positions_list = deepcopy([global_pos_df[global_pos_df['global_subdivision_idx'] == epoch_idx] for epoch_idx in np.arange(a_result.num_filter_epochs)]) ## add a List[pd.DataFrame] to plot as the measured positions
        rotate_to_vertical: bool = True
        should_expand_first_dim: bool=True
        (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers) = SingleArtistMultiEpochBatchHelpers.complete_build_stacked_flat_arrays(a_result=a_result, a_new_global_decoder=a_new_global_decoder, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)

        
        # Example 2: Filtering to epochs: [0, 20]
        rotate_to_vertical: bool = True
        should_expand_first_dim: bool=True
        (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers) = SingleArtistMultiEpochBatchHelpers.complete_build_stacked_flat_arrays(a_result=a_result, a_new_global_decoder=a_new_global_decoder,
                                                                                                                                                                                                                                                                                desired_epoch_end_idx=20, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)
                                                                                                                                                                                                                                                                                
        """
        n_timebins, flat_time_bin_containers, flat_timebins_p_x_given_n = a_result.flatten()
        flat_time_bin_containers = flat_time_bin_containers.tolist()
        flat_time_bin_centers: NDArray = np.hstack([v.centers for v in flat_time_bin_containers])

        # np.shape(flat_time_bin_containers) # (1738,)
        timebins_p_x_given_n_shape = np.shape(flat_timebins_p_x_given_n) # (76, 40, 29532)
        n_xbins, n_ybins, n_tbins = timebins_p_x_given_n_shape
        # (n_xbins, n_ybins, n_tbins)
        # np.shape(flat_time_bin_centers) # (29532,)

        
        if desired_epoch_end_idx is not None:
            ## Filter if desired:
            flat_timebins_p_x_given_n, flat_time_bin_centers = cls._slice_to_epoch_range(flat_timebins_p_x_given_n=flat_timebins_p_x_given_n, flat_time_bin_centers=flat_time_bin_centers, desired_epoch_start_idx=desired_epoch_start_idx, desired_epoch_end_idx=desired_epoch_end_idx)
        
        flattened_timebins_p_x_given_n_shape = np.shape(flat_timebins_p_x_given_n) # (76, 40, 29532)
        n_xbins, n_ybins, n_tbins = flattened_timebins_p_x_given_n_shape ## MUST BE UPDATED POST SLICE
        # (n_xbins, n_ybins, n_tbins)

        # flattened_n_xbins, flattened_n_ybins, flattened_n_tbins = flattened_timebins_p_x_given_n_shape
        # (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins)
        # np.shape(flat_time_bin_centers) # (29532,)
        ## OUTPUTS: flat_p_x_given_n, flat_time_bin_centers, 
        stacked_p_x_given_n = cls.reshape_p_x_given_n_for_single_artist_display(flat_timebins_p_x_given_n, rotate_to_vertical=rotate_to_vertical, should_expand_first_dim=should_expand_first_dim) # (1, 57, 90)
        
        # np.shape(stacked_p_x_given_n) # (1, 2244432, 40)
        

        xbin_centers = deepcopy(a_new_global_decoder.xbin_centers)
        ybin_centers = deepcopy(a_new_global_decoder.ybin_centers)

        if not rotate_to_vertical:
            stacked_flat_time_bin_centers = flat_time_bin_centers.repeat(n_xbins) # ((n_xbins*n_tbins), ) -- both are original sizes
            stacked_flat_xbin_centers = deepcopy(xbin_centers).repeat(n_tbins)  
            stacked_flat_ybin_centers = deepcopy(ybin_centers)         
        else:
            # vertically-oriented tracks (default)
            stacked_flat_time_bin_centers = flat_time_bin_centers.repeat(n_ybins) # ((n_ybins*n_tbins), ) -- both are original sizes
            stacked_flat_xbin_centers = deepcopy(xbin_centers)
            stacked_flat_ybin_centers = deepcopy(ybin_centers).repeat(n_tbins) ## these will lay along the x-axis

        flattened_n_xbins = len(stacked_flat_xbin_centers)
        flattened_n_ybins = len(stacked_flat_ybin_centers)
        flattened_n_tbins = len(stacked_flat_time_bin_centers)
        # (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins)

        if should_expand_first_dim:
            stacked_flat_time_bin_centers = np.expand_dims(stacked_flat_time_bin_centers, axis=0) # (1, (n_xbins*n_tbins)) or (1, (n_ybins*n_tbins)) -- both are original sizes

        # np.shape(stacked_flat_time_bin_centers) # (1, (n_ybins*n_tbins))
        ## OUPTUTS: (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers)
        return (n_xbins, n_ybins, n_tbins), (flattened_n_xbins, flattened_n_ybins, flattened_n_tbins), (stacked_p_x_given_n, stacked_flat_time_bin_centers, stacked_flat_xbin_centers, stacked_flat_ybin_centers)


    # ==================================================================================================================== #
    # Batch Track Shape Plotting                                                                                           #
    # ==================================================================================================================== #
    @classmethod
    def rect_tuples_to_NDArray(cls, rects, x_offset:float=0.0) -> NDArray:
        """ .shape (3, 4) """
        return np.vstack([[x+x_offset, y, w, h] for x, y, w, h, *args in rects])
        
    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=['cls.all_stacked_rect_arr_normalization'], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def rect_arr_normalization(cls, a_rect_arr, debug_print=False) -> NDArray:
        """ Normalizes the offsets and size to [0, 1]
        .shape (3, 4)
        
        Usage:
            Example 1:        
                normalized_long_rect_arr, ((x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total) = SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(long_rect_arr)
                normalized_long_rect_arr

            Example 2:
                track_single_rect_arr_dict = {'long': long_rect_arr, 'short': short_rect_arr}
                track_single_rect_arr_dict
                track_single_normalized_rect_arr_dict = {k:SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(v)[0] for k, v in track_single_rect_arr_dict.items()}
                track_normalization_tuple_dict = {k:SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(v)[1] for k, v in track_single_rect_arr_dict.items()}
                track_single_normalized_rect_arr_dict
                track_normalization_tuple_dict

        """
        if debug_print:
            print(f'a_rect_arr: {a_rect_arr}, np.shape(a_rect_arr): {np.shape(a_rect_arr)}')
            
        x0_offset: float = a_rect_arr[0, 0]
        y0_offset: float = a_rect_arr[0, 1]
        w0_multiplier: float = a_rect_arr[0, 2]
        h0_total: float = np.sum(a_rect_arr, axis=0)[3]

        if debug_print:
            print(f'x0_offset: {x0_offset}, y0_offset: {y0_offset}, w0_multiplier: {w0_multiplier}, h0_total: {h0_total}')
            
        ## normalize plotting by these values:
        normalized_long_rect_arr = deepcopy(a_rect_arr)
        normalized_long_rect_arr[:, 2] /= w0_multiplier
        normalized_long_rect_arr[:, 3] /= h0_total
        normalized_long_rect_arr[:, 0] /= w0_multiplier
        normalized_long_rect_arr[:, 1] /= h0_total
        if debug_print:
            print(f'normalized_long_rect_arr: {normalized_long_rect_arr}')

        normalized_x0_offset: float = normalized_long_rect_arr[0, 0]
        normalized_y0_offset: float = normalized_long_rect_arr[0, 1]
        if debug_print:
            print(f'normalized_x0_offset: {normalized_x0_offset}, normalized_y0_offset: {normalized_y0_offset}')
        
        ## only after scaling should we apply the translational offset
        normalized_long_rect_arr[:, 0] -= normalized_x0_offset
        normalized_long_rect_arr[:, 1] -= normalized_y0_offset

        # ## raw tanslational offset
        # normalized_long_rect_arr[:, 0] -= x0_offset
        # normalized_long_rect_arr[:, 1] -= y0_offset

        return normalized_long_rect_arr, ((x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total)


    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=['cls.rect_tuples_to_NDArray', 'cls.rect_arr_normalization'], used_by=['cls.track_dict_all_stacked_rect_arr_normalization'], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def all_stacked_rect_arr_normalization(cls, built_track_rects, num_horizontal_repeats: int, x_offset: float = 0.0) -> NDArray:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        a_track_rect_arr = cls.rect_tuples_to_NDArray(built_track_rects, x_offset=x_offset)
        # x0s = a_track_rect_arr[:, 0] # x0
        # widths = a_track_rect_arr[:, 2] # w
        # heights = a_track_rect_arr[:, 3] # h

        ## INPUTS: track_single_normalized_rect_arr_dict, track_normalization_tuple_dict

        # active_track_name: str = 'long'
        track_single_normalized_rect_arr, track_normalization_tuple = SingleArtistMultiEpochBatchHelpers.rect_arr_normalization(a_track_rect_arr)
        (x0_offset, y0_offset), (normalized_x0_offset, normalized_y0_offset), w0_multiplier, h0_total = track_normalization_tuple ## unpack track_normalization_tuple

        single_subdiv_normalized_width = 1.0
        single_subdiv_normalized_height = 1.0
        single_subdiv_normalized_offset_x = 1.0

        test_arr = []
        for epoch_idx in np.arange(num_horizontal_repeats):
            an_arr = deepcopy(track_single_normalized_rect_arr)
            an_arr[:, 0] += (epoch_idx * single_subdiv_normalized_offset_x) ## set offset 
            test_arr.append(an_arr)
            
        test_arr = np.vstack(test_arr)
        # np.shape(test_arr) # (5211, 4)
        return test_arr
            

    @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=['cls.all_stacked_rect_arr_normalization'], used_by=[], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def track_dict_all_stacked_rect_arr_normalization(cls, built_track_rects_dict, num_horizontal_repeats: int) -> Dict[str, NDArray]:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        track_all_normalized_rect_arr_dict = {}
        for active_track_name, built_track_rects in built_track_rects_dict.items():
            track_all_normalized_rect_arr_dict[active_track_name] = cls.all_stacked_rect_arr_normalization(built_track_rects=built_track_rects, num_horizontal_repeats=num_horizontal_repeats)

        ## OUTPUTS: track_all_normalized_rect_arr_dict
        return track_all_normalized_rect_arr_dict
    

    @function_attributes(short_name=None, tags=['new', 'active', 'inverse'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 08:41', related_items=[])
    @classmethod
    def track_dict_all_stacked_rect_arr_inverse_normalization(cls, track_all_rect_arr_dict, ax, num_active_horizontal_repeats: int) -> Dict[str, NDArray]:
        """ 
        Usage:
        
            all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
            all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax_width: float = np.diff(xlim)[0]
        ax_height: float = np.diff(ylim)[0]

        # (xlim, ylim)
        # (ax_width, ax_height)

        inverse_normalization_factor_width: float = ax_width / num_active_horizontal_repeats
        inverse_normalization_factor_height: float = 1.0 / ax_height

        # (inverse_normalization_factor_width, inverse_normalization_factor_height)
        
        ## OUTPUTS: inverse_normalization_factor_width, inverse_normalization_factor_height

        # ax.get_width()
        inverse_normalized_track_all_rect_arr_dict = {}

        for k, test_arr in track_all_rect_arr_dict.items():
            new_test_arr = deepcopy(test_arr)
            new_test_arr[:, 2] *= inverse_normalization_factor_width # scale by the width
            new_test_arr[:, 0] *= inverse_normalization_factor_width

            new_test_arr[:, 3] *= inverse_normalization_factor_height # scale by the width
            new_test_arr[:, 1] *= inverse_normalization_factor_height

            inverse_normalized_track_all_rect_arr_dict[k] = new_test_arr
            
        return inverse_normalized_track_all_rect_arr_dict
        ## OUTPUTS: inverse_normalized_track_all_rect_arr_dict
    


    # @function_attributes(short_name=None, tags=['new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-10 00:00', related_items=[])
    # @classmethod
    # def rect_tuples_to_horizontally_stacked_NDArray(cls, built_track_rects, num_horizontal_repeats: int, padding_x: float = 0.0, axes_inset_locators_list=None):
    #     """ 
    #     Usage:
        
    #         all_long_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(long_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))
    #         all_short_rect_arr = rect_tuples_to_horizontally_stacked_NDArray(short_rects, num_horizontal_repeats=(a_result.num_filter_epochs-1))

    #     """
    #     a_track_rect_arr = cls.rect_tuples_to_NDArray(built_track_rects, x_offset=-131.142)
    #     x0s = a_track_rect_arr[:, 0] # x0
    #     widths = a_track_rect_arr[:, 2] # w
    #     heights = a_track_rect_arr[:, 3] # h

    #     if axes_inset_locators_list is None:        
    #         # x1s = x0s + widths
    #         # x0s
    #         # widths
    #         # x1s
    #         single_subdiv_width: float = np.max(widths)
    #         single_subdiv_height: float = np.max(heights)
            
    #         single_subdiv_offset_x: float = single_subdiv_width + padding_x

    #         ## OUTPUTS: single_subdiv_width, single_subdiv_height, single_subdiv_offset_x
    #         return np.vstack(deepcopy([((epoch_idx * single_subdiv_offset_x), 0, single_subdiv_width, single_subdiv_height) for epoch_idx in np.arange(num_horizontal_repeats)])) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`
    #     else:
    #         return np.vstack(deepcopy([((axes_inset_locators_list[epoch_idx, 0] * single_subdiv_offset_x), 0, single_subdiv_width, single_subdiv_height) for epoch_idx in np.arange(num_horizontal_repeats)])) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`

    @function_attributes(short_name=None, tags=['main', 'new', 'active'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 09:16', related_items=[])
    @classmethod
    def add_batch_track_shapes(cls, ax, inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict, transform=None):
        """ 
        
        track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}
        track_shape_patch_collection_artists = SingleArtistMultiEpochBatchHelpers.add_batch_track_shapes(ax=ax, inverse_normalized_track_all_rect_arr_dict=inverse_normalized_track_all_rect_arr_dict, track_kwargs_dict=track_kwargs_dict)
        fig.canvas.draw_idle()
        
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        # import matplotlib.patches as patches
        assert track_kwargs_dict is not None

        extra_transform_kwargs = {}
        if transform is not None:
            extra_transform_kwargs['transform'] = transform
        
        track_names_list = ['long', 'short']
        # track_kwargs_dict = {'long': long_kwargs, 'short': short_kwargs}
        track_shape_patch_collection_artists = {'long': None, 'short': None}

        for active_track_name in track_names_list:
            # matplotlib_rect_kwargs_override = long_kwargs # {'linewidth': 2, 'edgecolor': '#0099ff42', 'facecolor': '#0099ff07'}

            matplotlib_rect_kwargs = track_kwargs_dict[active_track_name] # {'linewidth': 2, 'edgecolor': '#0099ff42', 'facecolor': '#0099ff07'}
            # active_all_rect_arr = track_all_rect_arr_dict[active_track_name]
            active_all_rect_arr = inverse_normalized_track_all_rect_arr_dict[active_track_name]

            # matplotlib ax was passed
            data = deepcopy(active_all_rect_arr)
            # rect_patches = [Rectangle((x, y), w, h) for x, y, w, h in data]
            rect_patches = [Rectangle((x, y), w, h, **matplotlib_rect_kwargs, **extra_transform_kwargs) for x, y, w, h in data] # , transform=ax.transData, transform=ax.transData
            
            # ## legacy patch-based way
            # rect = patches.Rectangle((x, y), w, h, **matplotlib_rect_kwargs)
            # plot_item.add_patch(rect)    

            # pc = PatchCollection(patches, edgecolors='k', facecolors='none')
            if track_shape_patch_collection_artists.get(active_track_name, None) is not None:
                # remove extant
                print(f'removing existing artist.')
                track_shape_patch_collection_artists[active_track_name].remove()
                track_shape_patch_collection_artists[active_track_name] = None

            # pc = PatchCollection(rect_patches, edgecolors=matplotlib_rect_kwargs.get('edgecolor', '#0099ff42'), facecolors=matplotlib_rect_kwargs.get('facecolor', '#0099ff07'))
            pc = PatchCollection(rect_patches, match_original=True) #, transform=ax.transAxes , transform=ax.transData
            track_shape_patch_collection_artists[active_track_name] = pc
            ax.add_collection(pc)
        ## END for active_track_name in track_names_list:

        # plt.gca().add_collection(pc)
        # plt.show()
        # ax.get_figure()
        # fig.canvas.draw_idle()
        
        return track_shape_patch_collection_artists



@define(slots=False)
class DecodedTrajectoryPlotter:
    """ Abstract Base Class for something that plots a decoded 1D or 2D trajectory. 
    
    """
    curr_epoch_idx: int = field(default=None)
    a_result: DecodedFilterEpochsResult = field(default=None)
    xbin_centers: NDArray = field(default=None)
    ybin_centers: Optional[NDArray] = field(default=None)
    xbin: NDArray = field(default=None)
    ybin: Optional[NDArray] = field(default=None)

    @property
    def num_filter_epochs(self) -> int:
        """The num_filter_epochs: int property."""
        return self.a_result.num_filter_epochs
    
    @property
    def curr_n_time_bins(self) -> int:
        """The num_filter_epochs: int property."""
        return len(self.a_result.time_bin_containers[self.curr_epoch_idx].centers)




@define(slots=False)
class DecodedTrajectoryMatplotlibPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded 1D or 2D trajectory using matplotlib. 

    Usage:    
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter

        ## 2D:
        # Choose the ripple epochs to plot:
        a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(LS_decoder_ripple_filter_epochs_decoder_result_dict)
        a_result: DecodedFilterEpochsResult = a_decoded_filter_epochs_decoder_result_dict['long'] # 2D
        num_filter_epochs: int = a_result.num_filter_epochs
        a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)
        fig, axs, laps_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True)

        integer_slider = a_decoded_traj_plotter.plot_epoch_with_slider_widget(an_epoch_idx=6)
        integer_slider

    """
    ## Artists/Figures/Axes:
    prev_heatmaps: List = field(default=Factory(list))
    artist_line_dict = field(default=Factory(dict))
    artist_markers_dict = field(default=Factory(dict))
    
    plots_data_dict_array: List[List[RenderPlotsData]] = field(init=False)
    artist_dict_array: List[List[Dict]] = field(init=False)
    fig = field(default=None)
    axs: NDArray = field(default=None)
    laps_pages: List = field(default=Factory(list))
    row_column_indicies: NDArray = field(default=None)
    linear_plotter_indicies: NDArray = field(default=None)
    
    # measured_position_df: Optional[pd.DataFrame] = field(default=None)
    rotate_to_vertical: bool = field(default=False, metadata={'desc': 'if False, the track is rendered horizontally along its length, otherwise it is rendered vectically'})
    
    
    ## Current Visibility State
    curr_epoch_idx: int = field(default=0)
    curr_time_bin_idx: Optional[int] = field(default=None)
    
    ## Widgets
    epoch_slider = field(default=None, init=False)
    time_bin_slider = field(default=None, init=False)
    checkbox = field(default=None, init=False)

    @property
    def is_single_time_bin_mode(self) -> bool:
        """ if True, all the time bins within the curr_epoch_idx are plotted, otherwise, only the time bin specified by curr_time_bin_idx is used."""
        return (self.curr_time_bin_idx is not None)


    ## MAIN PLOT FUNCTION:
    @function_attributes(short_name=None, tags=['main', 'plot', 'posterior', 'epoch', 'line', 'trajectory'], input_requires=[], output_provides=[], uses=['self._perform_add_decoded_posterior_and_trajectory'], used_by=['plot_epoch_with_slider_widget'], creation_date='2025-01-29 15:52', related_items=[])
    def plot_epoch(self, an_epoch_idx: int, time_bin_index: Optional[int]=None, include_most_likely_pos_line: Optional[bool]=None, override_ax=None, should_post_hoc_fit_to_image_extent: bool = True, debug_print:bool = False):
        """ Main plotting function.
             Internally calls `self._perform_add_decoded_posterior_and_trajectory(...)` to do the plotting.
        """
        self.curr_epoch_idx = an_epoch_idx
        self.curr_time_bin_idx = time_bin_index

        a_linear_index: int = an_epoch_idx
        curr_row = self.row_column_indicies[0][a_linear_index]
        curr_col = self.row_column_indicies[1][a_linear_index]
        curr_artist_dict = self.artist_dict_array[curr_row][curr_col]
        curr_plot_data: RenderPlotsData = self.plots_data_dict_array[curr_row][curr_col]


        if override_ax is None:
            an_ax = self.axs[curr_row][curr_col] # np.shape(self.axs) - (n_subplots, 2)
        else:
            an_ax = override_ax
            
        # an_ax = self.axs[0][0] # np.shape(self.axs) - (n_subplots, 2)

        assert len(self.xbin_centers) == np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(self.xbin_centers)}"

        a_p_x_given_n = self.a_result.p_x_given_n_list[an_epoch_idx] # (76, 40, n_epoch_t_bins)
        a_most_likely_positions = self.a_result.most_likely_positions_list[an_epoch_idx] # (n_epoch_t_bins, n_pos_dims) 
        a_time_bin_edges = self.a_result.time_bin_edges[an_epoch_idx] # (n_epoch_t_bins+1, )
        a_time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers # (n_epoch_t_bins, )

        has_measured_positions: bool = hasattr(self.a_result, 'measured_positions_list')
        if has_measured_positions:
            a_measured_pos_df: pd.DataFrame = self.a_result.measured_positions_list[an_epoch_idx]
            # assert len(a_measured_pos_df) == len(a_time_bin_centers)
        else:
            a_measured_pos_df = None

        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)

        assert len(a_time_bin_centers) == len(a_most_likely_positions)

        # heatmaps, a_line, _out_markers, _slider_tuple = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
        #                                                                      a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers) # , allow_time_slider=True

        # removing existing:

        # curr_artist_dict = {'prev_heatmaps': [], 'lines': {}, 'markers': {}}
        
        for a_heatmap in curr_artist_dict['prev_heatmaps']:
            a_heatmap.remove()
        curr_artist_dict['prev_heatmaps'].clear()

        for k, a_line in curr_artist_dict['lines'].items(): 
            a_line.remove()

        for k, _out_markers in curr_artist_dict['markers'].items(): 
            _out_markers.remove()
            
        curr_artist_dict['lines'].clear()# = {}
        curr_artist_dict['markers'].clear() # = {}
        
        ## Perform the plot:
        curr_artist_dict['prev_heatmaps'], (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data = self._perform_add_decoded_posterior_and_trajectory(an_ax, xbin_centers=self.xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, a_measured_pos_df=a_measured_pos_df, ybin_centers=self.ybin_centers,
                                                                            include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=time_bin_index, rotate_to_vertical=self.rotate_to_vertical, should_perform_reshape=True, should_post_hoc_fit_to_image_extent=should_post_hoc_fit_to_image_extent, debug_print=debug_print) # , allow_time_slider=True


        ## update the plot_data
        curr_plot_data.update(plots_data)
        self.plots_data_dict_array[curr_row][curr_col] = curr_plot_data ## set to the new value
        
        if a_meas_pos_line is not None:
            curr_artist_dict['lines']['meas'] = a_meas_pos_line
        if _meas_pos_out_markers is not None:
            curr_artist_dict['markers']['meas'] = _meas_pos_out_markers
        
        if a_line is not None:
            curr_artist_dict['lines']['most_likely'] = a_line
        if _out_markers is not None:
            curr_artist_dict['markers']['most_likely'] = _out_markers

        self.fig.canvas.draw_idle()


    @function_attributes(short_name=None, tags=['plotting', 'widget', 'interactive'], input_requires=[], output_provides=[], uses=['self.plot_epoch'], used_by=[], creation_date='2025-01-29 15:49', related_items=[])
    def plot_epoch_with_slider_widget(self, an_epoch_idx: int, include_most_likely_pos_line: Optional[bool]=None):
        """ this builds an interactive ipywidgets slider to scroll through the decoded epoch events
        
        Internally calls `self.plot_epoch` to perform posterior and line plotting
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.curr_epoch_idx = an_epoch_idx  # Ensure curr_epoch_idx is set

        def integer_slider(update_func, description, min_val, max_val, initial_val):
            slider = widgets.IntSlider(description=description, min=min_val, max=max_val, value=initial_val)

            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    update_func(change['new'])
            slider.observe(on_slider_change)
            return slider

        def checkbox_widget(update_func, description, initial_val):
            checkbox = widgets.Checkbox(description=description, value=initial_val)

            def on_checkbox_change(change):
                if (change['type'] == 'change') and (change['name'] == 'value'):
                    update_func(change['new'])
            checkbox.observe(on_checkbox_change)
            return checkbox

        def update_epoch_idx(index):            
            # print(f'update_epoch_idx(index: {index}) called')
            time_bin_index = None # default to no time_bin_idx
            # if not self.time_bin_slider.disabled:
            #     print(f'\t(not self.time_bin_slider.disabled)!!')
            #     self.time_bin_slider.value = 0 # reset to 0
            #     time_bin_index = self.time_bin_slider.value
            self.plot_epoch(an_epoch_idx=index, time_bin_index=time_bin_index, include_most_likely_pos_line=include_most_likely_pos_line)

        # def update_time_bin_idx(index):
        #     print(f'update_time_bin_idx(index: {index}) called')
        #     self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=index, include_most_likely_pos_line=include_most_likely_pos_line)

        # def on_checkbox_change(value):
        #     print(f'on_checkbox_change(value: {value}) called')
        #     if value:
        #         self.time_bin_slider.disabled = True
        #         self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=None, include_most_likely_pos_line=include_most_likely_pos_line)
        #     else:
        #         self.time_bin_slider.disabled = False
        #         self.plot_epoch(an_epoch_idx=self.epoch_slider.value, time_bin_index=self.time_bin_slider.value, include_most_likely_pos_line=include_most_likely_pos_line)

        self.epoch_slider = integer_slider(update_epoch_idx, 'epoch_IDX:', 0, (self.num_filter_epochs-1), an_epoch_idx)
        # self.time_bin_slider = integer_slider(update_time_bin_idx, 'time bin:', 0, (self.curr_n_time_bins-1), 0)
        # self.checkbox = checkbox_widget(on_checkbox_change, 'Disable time bin slider', True)

        self.plot_epoch(an_epoch_idx=an_epoch_idx, time_bin_index=None, include_most_likely_pos_line=include_most_likely_pos_line)

        display(self.epoch_slider)
        # display(self.checkbox)
        # display(self.time_bin_slider)


    # ==================================================================================================================== #
    # General Fundamental Plot Element Helpers                                                                             #
    # ==================================================================================================================== #
    
    # fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=0)
    @classmethod
    def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False, time_cmap='viridis', **LineCollection_kwargs):
        """ Adds a gradient line representing a timeseries of (x, y) positions.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        
        
        cls._helper_add_gradient_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        # Create a continuous norm to map from data points to colors
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        norm = plt.Normalize(t.min(), t.max())
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if isinstance(time_cmap, str):
            time_cmap = plt.get_cmap(time_cmap)  # Choose a colormap
        lc = LineCollection(segments, cmap=time_cmap, norm=norm, **LineCollection_kwargs)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)
        lc.set_alpha(0.85)
        line = ax.add_collection(lc)

        if add_markers:
            # Builds scatterplot markers (points) along the path
            colors_arr = time_cmap(norm(t)) # line.get_colors() # (17, 4) -- this is not working!
            # segments_arr = line.get_segments() # (16, 2, 2)
            # len(a_most_likely_positions) # 17
            _out_markers = ax.scatter(x=x, y=y, s=50, c=colors_arr, marker='D')
            return line, _out_markers
        else:
            return line, None

    @function_attributes(short_name=None, tags=['AI', 'posterior', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 12:00', related_items=[])
    @classmethod
    def _helper_add_heatmap(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, ybin_centers=None, rotate_to_vertical:bool=False, debug_print:bool=False, posterior_masking_value: float = 0.0025, full_posterior_opacity: float = 1.0,
                            custom_image_extent=None, cmap = 'viridis', should_perform_reshape: bool=True):
        """
        Helper that handles all the posterior heatmap plotting (for both 1D and 2D cases).
        
        Arguments:
            an_ax: the matplotlib axes to plot upon.
            xbin_centers: x axis bin centers.
            a_p_x_given_n: the decoded posterior array. If should_perform_reshape is True, its transpose is taken.
            a_time_bin_centers: array of time bin centers. -- Unused if 2D
            ybin_centers: if provided then a 2D posterior is assumed.
            rotate_to_vertical: if True, swap the x and y axes.
            debug_print: if True, prints debug information.
            posterior_masking_value: values below this are masked.
            should_perform_reshape: if True, reshapes the posterior.
            
        Returns:
            heatmaps: list of image handles.
            image_extent: extent (x_min, x_max, y_min, y_max) used for imshow.
            extra_dict: dictionary of additional computed values:
                For 1D: includes 'fake_y_center', 'fake_y_lower_bound', 'fake_y_upper_bound', 'fake_y_arr'.
                For 2D: may include 'y_values' and the flag 'is_2D': True.
        """
        # Reshape the posterior if necessary.
        if should_perform_reshape:
            posterior = deepcopy(a_p_x_given_n).T
        else:
            posterior = deepcopy(a_p_x_given_n)
        if debug_print:
            print(f'np.shape(posterior): {np.shape(posterior)}')
        
        masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)
        is_2D: bool = (np.ndim(posterior) >= 3)
        if debug_print:
            print(f'is_2D: {is_2D}')
        
        x_values = deepcopy(xbin_centers)
        extra_dict = {'is_2D': is_2D}
        
        if not is_2D:
            # 1D: Build fake y-axis values from current axes limits.
            y_min, y_max = an_ax.get_ylim()
            fake_y_width = (y_max - y_min)
            fake_y_center: float = y_min + (fake_y_width / 2.0)
            fake_y_lower_bound: float = fake_y_center - fake_y_width
            fake_y_upper_bound: float = fake_y_center + fake_y_width
            fake_y_num_samples: int = len(a_time_bin_centers)
            fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples)
            extra_dict.update({
                'fake_y_center': fake_y_center,
                'fake_y_lower_bound': fake_y_lower_bound,
                'fake_y_upper_bound': fake_y_upper_bound,
                'fake_y_arr': fake_y_arr,
            })
            # For plotting, use fake_y values.
            y_values = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples)
            extra_dict['y_values'] = y_values ## not needed?
        else:
            # 2D: use provided ybin_centers.
            assert ybin_centers is not None, "For 2D posterior, ybin_centers must be provided."
            y_values = deepcopy(ybin_centers)
            extra_dict['y_values'] = y_values
        
        # Adjust for vertical orientation if requested.
        if rotate_to_vertical:
            image_extent = (y_values.min(), y_values.max(), x_values.min(), x_values.max())
            # Swap x and y arrays.
            x_values, y_values = y_values, x_values
            if should_perform_reshape:
                if debug_print:
                    print(f'rotate_to_vertical: swapping axes. Original masked_posterior shape: {np.shape(masked_posterior)}')
                masked_posterior = masked_posterior.swapaxes(-2, -1) ## swap the last two (x, y) axes -- this doesn't work, because
                
            if debug_print:
                print(f'Post-swap masked_posterior shape: {np.shape(masked_posterior)}')
        else:
            image_extent = (x_values.min(), x_values.max(), y_values.min(), y_values.max())
        
        if custom_image_extent is not None:
            assert len(custom_image_extent) == 4
            image_extent = deepcopy(custom_image_extent)

        ## set after any swapping:
        extra_dict['x_values'] = x_values
        extra_dict['y_values'] = y_values

        n_time_bins: int = len(a_time_bin_centers)
        masked_shape = np.shape(masked_posterior)
        # Assert.all_equal(n_time_bins, masked_shape[0])
        assert n_time_bins == masked_shape[0], f" masked_shape[0]: { masked_shape[0]} != n_time_bins: {n_time_bins}"
        
        heatmaps = []
        # For simplicity, we assume non-single-time-bin mode (as asserted in the calling function).
        if not is_2D:
            a_heatmap = an_ax.imshow(masked_posterior, aspect='auto', cmap=cmap, alpha=full_posterior_opacity,
                                       extent=image_extent, origin='lower', interpolation='none')
            heatmaps.append(a_heatmap)
        else:
            vmin_global = np.nanmin(posterior)
            vmax_global = np.nanmax(posterior)
            # Give a minimum opacity per time step.
            time_step_opacity: float = max(full_posterior_opacity/float(n_time_bins), 0.2)
            for i in np.arange(n_time_bins):
                a_heatmap = an_ax.imshow(np.squeeze(masked_posterior[i, :, :]), aspect='auto', cmap=cmap, alpha=time_step_opacity,
                                           extent=image_extent, origin='lower', interpolation='none',
                                           vmin=vmin_global, vmax=vmax_global)
                heatmaps.append(a_heatmap)
        return heatmaps, image_extent, extra_dict


    # ==================================================================================================================== #
    # Specific Data Extraction and plot wrapping functions                                                                 #
    # ==================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['specific', 'plot_helper'], input_requires=[], output_provides=[], uses=['cls._helper_add_gradient_line'], used_by=['cls._perform_add_decoded_posterior_and_trajectory'], creation_date='2025-02-11 15:40', related_items=[])
    @classmethod
    def _perform_plot_measured_position_line_helper(cls, an_ax, a_measured_pos_df, a_time_bin_centers, fake_y_lower_bound: float, fake_y_upper_bound: float, rotate_to_vertical: bool, debug_print: bool) -> Tuple[Any, Any]:
        """
        Helper function to plot the measured positions line (recorded laps) as a gradient line.
        This extracts the functionality from the original code block (lines 1116-1181) so that it can be reused.
        
        Returns a tuple (a_meas_pos_line, _meas_pos_out_markers) that are produced by the gradient line helper.
        """
        # Get measured time bins from the dataframe
        a_measured_time_bin_centers: NDArray = np.atleast_1d([np.squeeze(a_measured_pos_df['t'].to_numpy())])
        # Determine X and Y positions based on dimensionality.
        if rotate_to_vertical is False:
            # 1D: construct fake y values.
            measured_fake_y_num_samples: int = len(a_measured_pos_df)
            measured_fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, measured_fake_y_num_samples)
            x = np.atleast_1d([a_measured_pos_df['x'].to_numpy()])
            y = np.atleast_1d([measured_fake_y_arr])
        else:
            # 2D: take columns as is.
            x = np.squeeze(a_measured_pos_df['x'].to_numpy())
            y = np.squeeze(a_measured_pos_df['y'].to_numpy())
        
        # If in single-time-bin mode, restrict positions to those with t <= current time bin center.
        # n_time_bins: int = len(a_time_bin_centers)
        # Here, the caller is expected to ensure that time_bin_index is valid.
        # (This helper would be called after the check for single-time-bin mode.)
        # In a full implementation, one may pass time_bin_index as an argument.
        # For now, we only handle the non-restricted case.
        
        # Squeeze arrays down to rank 1.
        a_measured_time_bin_centers = np.squeeze(a_measured_time_bin_centers)
        x = np.squeeze(x)
        y = np.squeeze(y)
        if debug_print:
            print(f'\tFinal Shapes:')
            print(f'\tnp.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)}, np.shape(a_measured_time_bin_centers): {np.shape(a_measured_time_bin_centers)}')
        
        # Set pos_kwargs according to orientation.
        if not rotate_to_vertical:
            pos_kwargs = dict(x=x, y=y)
        else:
            pos_kwargs = dict(x=y, y=x)  # swap if vertical
        
        add_markers = True
        colors = [(0, 0.6, 0), (0, 0, 0)]  # first is green, second is black
        # Create a colormap (green to black).
        time_cmap = LinearSegmentedColormap.from_list("GreenToBlack", colors, N=25)
        
        # Use the helper to add a gradient line.
        a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(an_ax, t=a_measured_time_bin_centers, **pos_kwargs, add_markers=add_markers, time_cmap=time_cmap, zorder=0)
        
        return a_meas_pos_line, _meas_pos_out_markers
    

    @function_attributes(short_name=None, tags=['plot'], input_requires=[], output_provides=[], uses=['cls._helper_add_heatmap', 'cls._perform_plot_measured_position_line_helper'], used_by=['.plot_epoch'], creation_date='2025-01-29 15:53', related_items=[])
    @classmethod
    def _perform_add_decoded_posterior_and_trajectory(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, a_most_likely_positions, ybin_centers=None, a_measured_pos_df: Optional[pd.DataFrame]=None,
                                                        include_most_likely_pos_line: Optional[bool]=None, time_bin_index: Optional[int]=None, rotate_to_vertical:bool=False, debug_print=False, posterior_masking_value: float = 0.0025, should_perform_reshape: bool=True, should_post_hoc_fit_to_image_extent: bool=False): # posterior_masking_value: float = 0.01 -- 1D
        """ Plots the 1D or 2D posterior and most likely position trajectory over the top of an axes created with `fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)`
        
        np.shape(a_time_bin_centers) # 1D & 2D: (12,)
        np.shape(a_most_likely_positions) # 2D: (12, 2)
        np.shape(posterior): 1D: (56, 27);    2D: (12, 6, 57)

        
        time_bin_index: if time_bin_index is not None, only a single time bin will be plotted. Provide this to plot using a slider or programmatically animating.


        Usage:

        # for 1D need to set `ybin_centers = None`
        an_ax = axs[0][0]
        heatmaps, a_line, _out_markers = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers)


        """

        is_single_time_bin_mode: bool = (time_bin_index is not None) and (time_bin_index != -1)
        assert not is_single_time_bin_mode, f"time_bin_index: {time_bin_index}"

        if debug_print:
            if a_measured_pos_df is not None:
                print(f'a_measured_pos_df.shape: {a_measured_pos_df.shape}')
        

        # ==================================================================================================================== #
        # Plot the posterior heatmap                                                                                           #
        # ==================================================================================================================== #
        # Delegate the posterior plotting functionality.
        heatmaps, image_extent, extra_dict = cls._helper_add_heatmap(
            an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, ybin_centers=ybin_centers,
            rotate_to_vertical=rotate_to_vertical, debug_print=debug_print, 
            posterior_masking_value=posterior_masking_value, should_perform_reshape=should_perform_reshape)
        
        is_2D: bool = extra_dict['is_2D']
        if debug_print:
            print(f'is_single_time_bin_mode: {is_single_time_bin_mode}, is_2D: {is_2D}')
            

        # For 1D case, retrieve fake y values.
        if np.ndim(a_p_x_given_n) < 3:
            fake_y_center = extra_dict['fake_y_center']
            fake_y_arr = extra_dict['fake_y_arr']
            fake_y_lower_bound = extra_dict['fake_y_lower_bound']
            fake_y_upper_bound = extra_dict['fake_y_upper_bound']
            
        else:
            fake_y_center = None
            fake_y_arr = None
            fake_y_lower_bound = None
            fake_y_upper_bound = None

                    
        # # Add colorbar
        # cbar = plt.colorbar(a_heatmap, ax=an_ax)
        # cbar.set_label('Posterior Probability Density')


        # Add Gradiant Measured Position (recorded laps) Line ________________________________________________________________ #         
        if (a_measured_pos_df is not None):
            a_meas_pos_line, _meas_pos_out_markers = cls._perform_plot_measured_position_line_helper(an_ax, a_measured_pos_df, a_time_bin_centers, fake_y_lower_bound, fake_y_upper_bound, rotate_to_vertical=rotate_to_vertical, debug_print=debug_print)
               
        # if (a_measured_pos_df is not None):
        #     if debug_print:
        #         print(f'plotting measured positions...')
        #     a_measured_time_bin_centers: NDArray = np.atleast_1d([np.squeeze(a_measured_pos_df['t'].to_numpy())]) # np.atleast_1d([np.squeeze(a_measured_pos_df['t'].to_numpy())])                
        #     if not is_2D:
        #         measured_fake_y_num_samples: int = len(a_measured_pos_df)
        #         measured_fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, measured_fake_y_num_samples)
        #         x = np.atleast_1d([a_measured_pos_df['x'].to_numpy()])
        #         y = np.atleast_1d([measured_fake_y_arr])
        #     else:
        #         # 2D:
        #         x = np.squeeze(a_measured_pos_df['x'].to_numpy())
        #         y = np.squeeze(a_measured_pos_df['y'].to_numpy())
                
        #     if is_single_time_bin_mode:
        #         ## restrict to single time bin if is_single_time_bin_mode:
        #         if debug_print:
        #             print(f'\tis_single_time_bin_mode, so restricting to specific time bin: {time_bin_index}')
        #         assert (time_bin_index < n_time_bins)
        #         a_curr_tbin_center: float = a_time_bin_centers[time_bin_index] ## it's a real time
        #         is_measured_t_bin_included = (a_measured_pos_df['t'].to_numpy() <= a_curr_tbin_center) ## find all bins less than the current index
        #         a_measured_time_bin_centers = np.atleast_1d([np.squeeze(a_measured_pos_df['t'].to_numpy()[is_measured_t_bin_included])]) ## could just slice `a_measured_time_bin_centers`, but we don't
        #         x = np.atleast_1d([x[is_measured_t_bin_included]])
        #         y = np.atleast_1d([y[is_measured_t_bin_included]])
                
        #     # if debug_print:
        #     #     print(f'\tnp.shape(a_measured_time_bin_centers): {np.shape(a_measured_time_bin_centers)}')
        #     #     print(f'\tnp.shape(x): {np.shape(x)}')
        #     #     print(f'\tnp.shape(y): {np.shape(y)}')
                
        #     ## squeeze back down so all are rank 1 - (n_epoch_t_bins, )
        #     a_measured_time_bin_centers = np.squeeze(a_measured_time_bin_centers)
        #     x = np.squeeze(x)
        #     y = np.squeeze(y)
            
        #     if debug_print:
        #         print(f'\tFinal Shapes:')
        #         print(f'\tnp.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)}, np.shape(a_measured_time_bin_centers): {np.shape(a_measured_time_bin_centers)}')
                
        #     if not rotate_to_vertical:
        #         pos_kwargs = dict(x=x, y=y)
        #     else:
        #         # vertical:
        #         pos_kwargs = dict(x=y, y=x) ## swap x and y
                
        #     add_markers = True
        #     # time_cmap = 'Reds'
        #     # time_cmap = 'gist_gray'
            
        #     colors = [(0, 0.6, 0), (0, 0, 0)] # first color is black, last is green
        #     time_cmap = LinearSegmentedColormap.from_list("GreenToBlack", colors, N=25)

        #     if not is_2D: # 1D case
        #         # a_line = _helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=a_most_likely_positions, y=np.full_like(a_time_bin_centers, fake_y_center))
        #         a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(an_ax, t=a_measured_time_bin_centers, **pos_kwargs, add_markers=add_markers, time_cmap=time_cmap, zorder=0)
        #     else:
        #         # 2D case
        #         if debug_print:
        #             print(f'a_measured_time_bin_centers: {a_measured_time_bin_centers}')
        #         a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(an_ax, t=a_measured_time_bin_centers, **pos_kwargs, add_markers=add_markers, time_cmap=time_cmap, zorder=0)
                
        #     # _out_markers = ax.scatter(x=x, y=y, c=colors_arr)
            
        # else:
        #     a_meas_pos_line, _meas_pos_out_markers = None, None
            

        # Add Gradient Most Likely Position Line _____________________________________________________________________________ #
        if include_most_likely_pos_line:
            if not is_2D:
                x = np.atleast_1d([a_most_likely_positions[time_bin_index]]) # why time_bin_idx here?
                y = np.atleast_1d([fake_y_arr[time_bin_index]])
            else:
                # 2D:
                x = np.squeeze(a_most_likely_positions[:,0])
                y = np.squeeze(a_most_likely_positions[:,1])
                
            if is_single_time_bin_mode:
                ## restrict to single time bin if is_single_time_bin_mode:
                assert (time_bin_index < n_time_bins)
                a_time_bin_centers = np.atleast_1d([a_time_bin_centers[time_bin_index]])
                x = np.atleast_1d([x[time_bin_index]])
                y = np.atleast_1d([y[time_bin_index]])
                

            if not rotate_to_vertical:
                pos_kwargs = dict(x=x, y=y)
            else:
                # vertical:
                ## swap x and y:
                pos_kwargs = dict(x=y, y=x)
                

            if not is_2D: # 1D case
                # a_line = _helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=a_most_likely_positions, y=np.full_like(a_time_bin_centers, fake_y_center))
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True)
            else:
                # 2D case
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, **pos_kwargs, add_markers=True)
        else:
            a_line, _out_markers = None, None
            

        if should_post_hoc_fit_to_image_extent:
            ## set Axes xlims/ylims post-hoc so they fit
            an_ax.set_xlim(image_extent[0], image_extent[1])
            an_ax.set_ylim(image_extent[2], image_extent[3])


        # plot_data = MatplotlibRenderPlots(name='_perform_add_decoded_posterior_and_trajectory')
        # plots = RenderPlots('_perform_add_decoded_posterior_and_trajectory')
        plots_data = RenderPlotsData(name='_perform_add_decoded_posterior_and_trajectory', image_extent=deepcopy(image_extent))

        return heatmaps, (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data




    def plot_decoded_trajectories_2d(self, sess, curr_num_subplots=10, active_page_index=0, plot_actual_lap_lines:bool=False, fixed_columns: int = 2, use_theoretical_tracks_instead: bool = True, existing_ax=None, axes_inset_locators_list=None):
        """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots
        
        Called to setup the graph.
        
        Great plotting for laps.
        Plots in a paginated manner.
        
        use_theoretical_tracks_instead: bool = True - # if False, renders all positions the animal traversed over the entire session. Otherwise renders the theoretical (idaal) track.

        ISSUE: `fixed_columns: int = 1` doesn't work due to indexing


        History: based off of plot_lap_trajectories_2d

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_decoded_trajectories_2d
        
            fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)

        
        """

        if use_theoretical_tracks_instead:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(deepcopy(sess.config))


        def _subfn_chunks(iterable, size=10):
            iterator = iter(iterable)
            for first in iterator:    # stops when iterator is depleted
                def chunk():          # construct generator for next chunk
                    yield first       # yield element from for loop
                    for more in islice(iterator, size - 1):
                        yield more    # yield more elements from the iterator
                yield chunk()         # in outer generator, yield next chunk
            
        def _subfn_build_epochs_multiplotter(nfields, linear_plot_data=None):
            """ builds the figures
             captures: self.rotate_to_vertical, fixed_columns, (long_track_inst, short_track_inst)
            
            """
            linear_plotter_indicies = np.arange(nfields)
            needed_rows = int(np.ceil(nfields / fixed_columns))
            row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
            
            if existing_ax is None:
                ## Create a new axes and figure
                fig, axs = plt.subplots(needed_rows, fixed_columns, sharex=True, sharey=True, figsize=[4*fixed_columns,14*needed_rows], gridspec_kw={'wspace': 0, 'hspace': 0}) #ndarray (5,2)
                
            else:
                ## use the existing axes to plot the subaxes on
                print(f'using subaxes on the existing axes')
                assert axes_inset_locators_list is not None
                
                fig = existing_ax.get_figure()
                ## convert to relative??
                
                axs = [] ## list
                # for curr_row, a_row_list in enumerate(self.row_column_indicies):
                a_linear_index = 0
                for curr_row in np.arange(needed_rows):
                    a_new_axs_list = []
                    # for curr_col, an_element in enumerate(a_row_list):
                    for curr_col in np.arange(fixed_columns):
                        # Add subaxes at [left, bottom, width, height] in normalized parent coordinates
                        # ax_inset = existing_ax.add_axes([0.2, 0.6, 0.3, 0.3])  # Positioned at 20% left, 60% bottom
                        ax_inset_location = axes_inset_locators_list[a_linear_index]
                        ax_inset = existing_ax.inset_axes(ax_inset_location, transform=existing_ax.transData, borderpad=0) # [x0, y0, width, height], where [x0, y0] is the lower-left corner -- can do data_coords by adding `, transform=existing_ax.transData`
                        a_new_axs_list.append(ax_inset) 
                        a_linear_index += 1 ## increment

                    ## accumulate the lists
                    axs.append(a_new_axs_list)        

                for a_linear_index in linear_plotter_indicies:
                    curr_row = row_column_indicies[0][a_linear_index]
                    curr_col = row_column_indicies[1][a_linear_index]
                    ## format the titles
                    an_ax = axs[curr_row][curr_col]
                    

            axs = np.atleast_2d(axs)
            # mp.set_size_inches(18.5, 26.5)

            background_track_shadings = {}
            for a_linear_index in linear_plotter_indicies:
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                ## format the titles
                an_ax = axs[curr_row][curr_col]
                an_ax.set_xticks([])
                an_ax.set_yticks([])
                
                if not use_theoretical_tracks_instead:
                    background_track_shadings[a_linear_index] = an_ax.plot(linear_plot_data[a_linear_index][0,:], linear_plot_data[a_linear_index][1,:], c='k', alpha=0.2)
                else:
                    # active_config = curr_active_pipeline.sess.config
                    background_track_shadings[a_linear_index] = _perform_plot_matplotlib_2D_tracks(long_track_inst=long_track_inst, short_track_inst=short_track_inst, ax=an_ax, rotate_to_vertical=self.rotate_to_vertical)
                
            return fig, axs, linear_plotter_indicies, row_column_indicies, background_track_shadings
        
        def _subfn_add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, lap_position_traces, lap_time_ranges, use_time_gradient_line=True):
            # Add the lap trajectory:
            for a_linear_index in linear_plotter_indicies:
                curr_lap_id = active_page_laps_ids[a_linear_index]
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                curr_lap_time_range = lap_time_ranges[curr_lap_id]
                curr_lap_label_text = 'Lap[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[0], curr_lap_time_range[1])
                curr_lap_num_points = len(lap_position_traces[curr_lap_id][0,:])
                if use_time_gradient_line:
                    # Create a continuous norm to map from data points to colors
                    curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(lap_position_traces[curr_lap_id][0,:]))
                    norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
                    # needs to be (numlines) x (points per line) x 2 (for x and y)
                    points = np.array([lap_position_traces[curr_lap_id][0,:], lap_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap='viridis', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(curr_lap_timeseries)
                    lc.set_linewidth(2)
                    lc.set_alpha(0.85)
                    a_line = axs[curr_row][curr_col].add_collection(lc)
                    # add_arrow(line)
                else:
                    a_line = axs[curr_row][curr_col].plot(lap_position_traces[curr_lap_id][0,:], lap_position_traces[curr_lap_id][1,:], c='k', alpha=0.85)
                    # curr_lap_endpoint = curr_lap_position_traces[curr_lap_id][:,-1].T
                    a_start_arrow = _plot_helper_add_arrow(a_line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                    a_middle_arrow = _plot_helper_add_arrow(a_line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                    a_end_arrow = _plot_helper_add_arrow(a_line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                    # add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                    # add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')
                # add lap text label
                a_lap_label_text = axs[curr_row][curr_col].text(250, 126, curr_lap_label_text, horizontalalignment='right', size=12)
                # PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Compute required data from session:
        curr_position_df, lap_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(sess)
        
        # lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in session.laps.lap_id]

        if self.rotate_to_vertical:
            # vertical
            # x_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("x")]
            # y_columns = [col for col in lap_specific_position_dfs[0].columns if col.startswith("y")]

            for a_df in lap_specific_position_dfs:
                a_df['x_temp'] = deepcopy(a_df['x'])
                a_df['x'] = deepcopy(a_df['y'])
                a_df['y'] = deepcopy(a_df['x_temp'])
                # a_df[['x', 'y']] = a_df[['y', 'x']] ## swap the columns order
                
            curr_position_df[['x', 'y']] = curr_position_df[['y', 'x']] ## swap the columns order
            curr_position_df[['x_smooth', 'y_smooth']] = curr_position_df[['y_smooth', 'x_smooth']] ## swap the columns order

            # print(x_columns)

            # laps_position_traces_list = [lap_pos_df[position_col_names].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
            # lap_specific_position_dfs[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order
            
            # lap_specific_position_dfs[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order
            # curr_position_df[['x', 'y']] = lap_specific_position_dfs[['y', 'x']] ## swap the columns order

        position_col_names = ['x', 'y']
        laps_position_traces_list = [lap_pos_df[position_col_names].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
        
        laps_time_range_list = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]
        
        num_laps = len(sess.laps.lap_id)
        linear_lap_index = np.arange(num_laps)
        lap_time_ranges = dict(zip(sess.laps.lap_id, laps_time_range_list))
        lap_position_traces = dict(zip(sess.laps.lap_id, laps_position_traces_list)) ## each lap indexed by lap_id
        
        all_maze_positions = curr_position_df[position_col_names].to_numpy().T # (2, 59308)
        # np.shape(all_maze_positions)
        all_maze_data = [all_maze_positions for i in np.arange(curr_num_subplots)] # repeat the maze data for each subplot. (2, 593080)
        
        # Build Figures/Axes/Etc _____________________________________________________________________________________________ #
        self.fig, self.axs, self.linear_plotter_indicies, self.row_column_indicies, background_track_shadings = _subfn_build_epochs_multiplotter(curr_num_subplots, all_maze_data)
        perform_update_title_subtitle(fig=self.fig, ax=None, title_string="DecodedTrajectoryMatplotlibPlotter - plot_decoded_trajectories_2d") # , subtitle_string="TEST - SUBTITLE"
        
        # generate the pages
        epochs_pages = [list(chunk) for chunk in _subfn_chunks(sess.laps.lap_id, curr_num_subplots)] ## this is specific to actual laps...
         
        if plot_actual_lap_lines:
            ## IDK what this is sadly, i think it's a reminant of the lap plotter?
            active_page_laps_ids = epochs_pages[active_page_index]
            _subfn_add_specific_lap_trajectory(self.fig, self.axs, linear_plotter_indicies=self.linear_plotter_indicies, row_column_indicies=self.row_column_indicies, active_page_laps_ids=active_page_laps_ids, lap_position_traces=lap_position_traces, lap_time_ranges=lap_time_ranges, use_time_gradient_line=True)
            # plt.ylim((125, 152))
            
        self.laps_pages = epochs_pages



        ## Build artist holders:
        # MatplotlibRenderPlots
        self.plots_data_dict_array = []
        self.artist_dict_array = [] ## list
        for a_list in self.row_column_indicies:
            a_new_artists_list = []
            a_new_plot_data_list = []
            for an_element in a_list:
                a_new_artists_list.append({'prev_heatmaps': [], 'lines': {}, 'markers': {}}) ## make a new empty dict for each element
                a_new_plot_data_list.append(RenderPlotsData(f"DecodedTrajectoryMatplotlibPlotter.plot_decoded_trajectories_2d", image_extent=None))
            ## accumulate the lists
            self.plots_data_dict_array.append(a_new_plot_data_list)
            self.artist_dict_array.append(a_new_artists_list)                
        ## Access via ` self.artist_dict_array[curr_row][curr_col]`, same as the axes

        # for a_linear_index in self.linear_plotter_indicies:
        #     curr_row = self.row_column_indicies[0][a_linear_index]
        #     curr_col = self.row_column_indicies[1][a_linear_index]
            #   curr_artist_dict = self.artist_dict_array[curr_row][curr_col]

        return self.fig, self.axs, epochs_pages



from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_point_labels


@define(slots=False)
class DecodedTrajectoryPyVistaPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded trajectory (path) using pyvista. 
    
    Usage:
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryPyVistaPlotter
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

    
    curr_active_pipeline.prepare_for_display()
    _out = curr_active_pipeline.display(display_function='_display_3d_interactive_custom_data_explorer', active_session_configuration_context=global_epoch_context,
                                        params_kwargs=dict(should_use_linear_track_geometry=True, **{'t_start': t_start, 't_delta': t_delta, 't_end': t_end}),
                                        )
    iplapsDataExplorer: InteractiveCustomDataExplorer = _out['iplapsDataExplorer']
    pActiveInteractiveLapsPlotter = _out['plotter']
    a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=iplapsDataExplorer.p)
    a_decoded_trajectory_pyvista_plotter.build_ui()

    """
    p = field(default=None)
    curr_time_bin_index: int = field(default=0)
    enable_point_labels: bool = field(default=False)
    enable_plot_all_time_bins_in_epoch_mode: bool = field(default=False)


    slider_epoch = field(default=None)
    slider_epoch_time_bin = field(default=None)
    slider_epoch_time_bin_playback_checkbox = field(default=None)
    
    interactive_plotter: PhoInteractivePlotter = field(default=None)
    plotActors = field(default=None)
    data_dict = field(default=None)
    plotActors_CenterLabels = field(default=None)
    data_dict_CenterLabels = field(default=None)

    active_plot_fn: Callable = field(default=plot_3d_stem_points) # like [plot_3d_binned_bars, plot_3d_stem_points]
    animation_callback_interval_ms: int = field(default=200) # 200ms per time bin

    def build_ui(self):
        """ builds the slider vtk widgets 
        """

        assert self.p is not None
        if self.curr_epoch_idx is None:
            self.curr_epoch_idx = 0
        
        num_filter_epochs: int = self.num_filter_epochs
        curr_num_epoch_time_bins: int = self.curr_n_time_bins

        slider_epoch_kwargs = dict()
        if self.enable_plot_all_time_bins_in_epoch_mode:
            slider_epoch_kwargs = slider_epoch_kwargs | dict(event_type="always")

        

        if self.slider_epoch is None:
            def _on_slider_value_did_change_epoch_idx(value):
                """ only called when the value actually changes from the previous one (or there wasn't a previous one). """
                self.on_update_slider_epoch_idx(int(value))


            def _on_slider_callback_epoch_idx(value):
                """ checks whether the value has changed from the previous one before re-updating. 
                """
                if not hasattr(_on_slider_callback_epoch_idx, "last_value"):
                    _on_slider_callback_epoch_idx.last_value = value
                if value != _on_slider_callback_epoch_idx.last_value:
                    _on_slider_value_did_change_epoch_idx(value)
                    _on_slider_callback_epoch_idx.last_value = value


            self.slider_epoch = self.p.add_slider_widget(
                # callback=lambda value: self.on_update_slider_epoch_idx(int(value)), #storage_engine('epoch', int(value)), # triggering .__call__(self, param='epoch', value)....
                callback=lambda value: _on_slider_callback_epoch_idx(int(value)),
                rng=[0, num_filter_epochs-1],
                value=0,
                title="Epoch Idx",
                pointa=(0.64, 0.2),
                pointb=(0.94, 0.2),
                style='modern',
                fmt='%0.0f',
                **slider_epoch_kwargs,
            )


        if not self.enable_plot_all_time_bins_in_epoch_mode:
            if self.slider_epoch_time_bin is None:
                def _on_slider_value_did_change_epoch_time_bin(value):
                    """ only called when the value actually changes from the previous one (or there wasn't a previous one). """
                    self.on_update_slider_epoch_time_bin(int(value))


                def _on_slider_callback_epoch_time_bin(value):
                    """ checks whether the value has changed from the previous one before re-updating. This might not be the best approach because it should be forcibly re-updated when the epoch_idx changes even if the time_bin_idx stays the same (like it's sitting at 0 while scrolling through epochs)
                    """
                    if not hasattr(_on_slider_callback_epoch_time_bin, "last_value"):
                        _on_slider_callback_epoch_time_bin.last_value = value
                    if value != _on_slider_callback_epoch_time_bin.last_value:
                        _on_slider_value_did_change_epoch_time_bin(value)
                        _on_slider_callback_epoch_time_bin.last_value = value

                self.slider_epoch_time_bin = self.p.add_slider_widget(
                    # callback=lambda value: self.on_update_slider_epoch_time_bin(int(value)), #storage_engine('time_bin', value),
                    callback=lambda value: _on_slider_callback_epoch_time_bin(int(value)),
                    rng=[0, curr_num_epoch_time_bins-1],
                    value=0,
                    title="Timebin IDX",
                    pointa=(0.74, 0.12),
                    pointb=(0.94, 0.12),
                    style='modern',
                    # fmt="%d",
                    event_type="always",
                    fmt='%0.0f',
                )

            if (self.interactive_plotter is None) or (self.slider_epoch_time_bin_playback_checkbox is None):
                self.interactive_plotter = PhoInteractivePlotter.init_from_plotter_and_slider(pyvista_plotter=self.p, interactive_timestamp_slider_actor=self.slider_epoch_time_bin, step_size=1, animation_callback_interval_ms=self.animation_callback_interval_ms) # 500ms per time bin
                self.slider_epoch_time_bin_playback_checkbox = self.interactive_plotter.interactive_checkbox_actor




    def update_ui(self):
        """ called to update the epoch_time_bin slider when the epoch_index slider is changed. 
        """
        if (self.slider_epoch_time_bin is not None) and (self.curr_n_time_bins is not None):
            self.slider_epoch_time_bin.GetRepresentation().SetMaximumValue((self.curr_n_time_bins-1))
            self.slider_epoch_time_bin.GetRepresentation().SetValue(self.slider_epoch_time_bin.GetRepresentation().GetMinimumValue()) # set to 0


    def perform_programmatic_slider_epoch_update(self, value):
        """ called to programmatically update the epoch_idx slider. """
        if (self.slider_epoch is not None):
            print(f'updating slider_epoch index to : {int(value)}')
            self.slider_epoch.GetRepresentation().SetValue(int(value)) # set to 0
            self.on_update_slider_epoch_idx(value=int(value))
            print(f'\tdone.')






    def on_update_slider_epoch_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
        # print(f'.on_update_slider_epoch(value: {value})')
        self.curr_epoch_idx = int(value) ## Update `curr_epoch_idx`
        if not self.enable_plot_all_time_bins_in_epoch_mode:
            self.curr_time_bin_index = 0 # change to 0
        else:
            ## otherwise default to a range
            self.curr_time_bin_index = np.arange(self.curr_n_time_bins)

        self.update_ui() # called to update the dependent time_bin slider

        if not self.enable_plot_all_time_bins_in_epoch_mode:
            self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
        else:
            ## otherwise default to a range
            self.perform_update_plot_epoch_time_bin_range(self.curr_time_bin_index)

        ## shouldn't be here:
        # update_plot_fn = self.data_dict.get('plot_3d_binned_bars[55.63197815967686]', {}).get('update_plot_fn', None)
        update_plot_fn = self.data_dict.get('plot_3d_stem_points_P_x_given_n', {}).get('update_plot_fn', None)
        if update_plot_fn is not None:
            update_plot_fn(self.curr_time_bin_index)



    def on_update_slider_epoch_time_bin(self, value: int):
        """ called when the epoch_time_bin within a given epoch_idx slider changes 
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        self.perform_update_plot_single_epoch_time_bin(value=value)
        


    @function_attributes(short_name=None, tags=['main_plot_update', 'single_time_bin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:03', related_items=[])
    def perform_update_plot_single_epoch_time_bin(self, value: int):
        """ single-time-bin plotting:
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        self.curr_time_bin_index = int(value) # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=self.curr_time_bin_index)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_fn(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels, active_plot_fn=self.active_plot_fn)
        

    @function_attributes(short_name=None, tags=['main_plot_update', 'multi_time_bins', 'epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:04', related_items=[])
    def perform_update_plot_epoch_time_bin_range(self, value: Optional[NDArray]=None):
        """ multi-time-bin plotting:
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        if value is None:
            value = np.arange(self.curr_n_time_bins)
        self.curr_time_bin_index = value # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=value)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_fn(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                time_bin_centers=a_time_bin_centers, posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels, active_plot_fn=self.active_plot_fn)

    def perform_clear_existing_decoded_trajectory_plots(self):
        ## remove existing actors
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import clear_3d_binned_bars_plots

        if self.plotActors is not None:
            clear_3d_binned_bars_plots(p=self.p, plotActors=self.plotActors)
            self.plotActors.clear()
        if self.data_dict is not None:
            self.data_dict.clear()

        if self.plotActors_CenterLabels is not None:
            self.plotActors_CenterLabels.clear()
        if self.data_dict_CenterLabels is not None:
            self.data_dict_CenterLabels.clear()




    def get_curr_posterior(self, an_epoch_idx: int = 0, time_bin_index:Union[int, NDArray]=0):
        a_posterior_p_x_given_n, a_time_bin_centers = self._perform_get_curr_posterior(a_result=self.a_result, an_epoch_idx=an_epoch_idx, time_bin_index=time_bin_index)
        n_epoch_timebins: int = len(a_time_bin_centers)

        if np.ndim(a_posterior_p_x_given_n) > 2:
            assert np.ndim(a_posterior_p_x_given_n) == 3, f"np.ndim(a_posterior_p_x_given_n) should be either 2 or 3, but it is {np.ndim(a_posterior_p_x_given_n)}"
            n_xbins, n_ybins, actual_n_epoch_timebins = np.shape(a_posterior_p_x_given_n) # (5, 312)
            assert n_epoch_timebins == actual_n_epoch_timebins, f"n_epoch_timebins: {n_epoch_timebins} != actual_n_epoch_timebins: {actual_n_epoch_timebins} from np.shape(a_posterior_p_x_given_n) ({np.shape(a_posterior_p_x_given_n)})"
        else:
            a_posterior_p_x_given_n = np.atleast_2d(a_posterior_p_x_given_n) #.T # (57, 1) ## There was an error being induced by the transpose for non 1D matricies passed in. Transpose seems like it should only be done for the (N, 1) case.

            if np.shape(a_posterior_p_x_given_n)[0] == 1:
                a_posterior_p_x_given_n = a_posterior_p_x_given_n.T 

            required_n_y_bins: int = len(self.ybin_centers) # passing an arbitrary amount of y-bins? Currently it's 6, which I don't get. Oh, I guess that comes from the 2D decoder that's passed in.
            n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n) # (5, 312)

            ## for a 1D posterior
            if (n_ybins < required_n_y_bins) and (n_ybins == 1):
                print(f'building 2D plotting data from 1D posterior.')

                # fill solid across all y-bins
                a_posterior_p_x_given_n = np.tile(a_posterior_p_x_given_n, (1, required_n_y_bins)) # (57, 6)
                
                ## fill only middle 2 bins.
                # a_posterior_p_x_given_n = np.tile(a_posterior_p_x_given_n, (1, required_n_y_bins)) # (57, 6) start ny filling all

                # find middle bin:
                # mid_bin_idx = np.rint(float(required_n_y_bins) / 2.0)
                # a_posterior_p_x_given_n[:, 1:] = np.nan
                # a_posterior_p_x_given_n[:, 3:-1] = np.nan
                

                n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n) # update again with new matrix

        assert n_xbins == np.shape(self.xbin_centers)[0], f"n_xbins: {n_xbins} != np.shape(xbin_centers)[0]: {np.shape(self.xbin_centers)}"
        assert n_ybins == np.shape(self.ybin_centers)[0], f"n_ybins: {n_ybins} != np.shape(ybin_centers)[0]: {np.shape(self.ybin_centers)}"
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        return a_posterior_p_x_given_n, a_time_bin_centers
    
    @classmethod
    def _perform_get_curr_posterior(cls, a_result, an_epoch_idx: int = 0, time_bin_index: Union[int, NDArray]=0, desired_max_height: float = 50.0):
        """ gets the current posterior for the specified epoch_idx and time_bin_index within the epoch."""
        # a_result.time_bin_containers
        a_posterior_p_x_given_n_all_t = a_result.p_x_given_n_list[an_epoch_idx]
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        # a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx]
        a_most_likely_positions = a_result.most_likely_positions_list[an_epoch_idx]
        # a_time_bin_edges = a_result.time_bin_edges[an_epoch_idx]
        a_time_bin_centers = a_result.time_bin_containers[an_epoch_idx].centers
        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)
        assert len(a_time_bin_centers) == len(a_most_likely_positions), f"len(a_time_bin_centers): {len(a_time_bin_centers)} != len(a_most_likely_positions): {len(a_most_likely_positions)}"
        # print(f'np.shape(a_posterior_p_x_given_n): {np.shape(a_posterior_p_x_given_n)}') # : (58, 5, 312) - (n_xbins, n_ybins, n_epoch_timebins)
        # 

        min_v = np.nanmin(a_posterior_p_x_given_n_all_t)
        max_v = np.nanmax(a_posterior_p_x_given_n_all_t)
        # print(f'min_v: {min_v}, max_v: {max_v}')
        multiplier_factor: float = desired_max_height / (float(max_v) - float(min_v))
        # print(f'multiplier_factor: {multiplier_factor}')

        ## get the specific time_bin_index posterior:
        if np.ndim(a_posterior_p_x_given_n_all_t) > 2:
            ## multiple time bins case (3D)
            # n_xbins, n_ybins, n_epoch_timebins = np.shape(a_posterior_p_x_given_n_all_t)
            a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, :, time_bin_index])
        else:
            ## single time bin case (2D)
            # n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n_all_t) ???
            a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, time_bin_index])
        a_posterior_p_x_given_n = a_posterior_p_x_given_n * multiplier_factor # multiply by the desired multiplier factor
        return a_posterior_p_x_given_n, a_time_bin_centers



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@metadata_attributes(short_name=None, tags=['pyvista', 'mixin', 'decoder', '3D', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-27 14:38', related_items=['DecodedTrajectoryPyVistaPlotter'])
class DecoderRenderingPyVistaMixin:
    """ Implementors render decoded positions and decoder info with PyVista 
    
    Requires:
        self.params
        
    Provides:
    
        Adds:
            ... More?
            
            
    Known Uses:
        InteractivePlaceCellTuningCurvesDataExplorer
    """

    def add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, debug_print=False):
        """ Adds a red position indicator callback for the current decoded position

        Usage:
            active_one_step_decoder = global_results.pf2D_Decoder
            _update_nearest_decoded_most_likely_position_callback, _conn = add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, _debug_print = False)

        """
        def _update_nearest_decoded_most_likely_position_callback(start_t, end_t):
            """ Only uses end_t
            Implicitly captures: self, _get_nearest_decoded_most_likely_position_callback
            
            Usage:
                _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0])
                _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

            """
            def _get_nearest_decoded_most_likely_position_callback(t):
                """ A callback that when passed a visualization timestamp (the current time to render) returns the most likely predicted position provided by the active_two_step_decoder
                Implicitly captures:
                    active_one_step_decoder, active_two_step_decoder
                Usage:
                    _get_nearest_decoded_most_likely_position_callback(9000.1)
                """
                active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
                active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
                # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
                assert np.shape(active_time_window_variable)[0] == np.shape(active_most_likely_positions)[1], f"timestamps and num positions must be the same but np.shape(active_time_window_variable): {np.shape(active_time_window_variable)} and np.shape(active_most_likely_positions): {np.shape(active_most_likely_positions)}!"
                last_window_index = np.searchsorted(active_time_window_variable, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
                # TODO: CORRECTNESS: why is it returning an index that corresponds to a time later than the current time?
                # for current time t=9000.0
                #     last_window_index: 1577
                #     last_window_time: 9000.5023
                # EH: close enough
                last_window_time = active_time_window_variable[last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
                displayed_time_offset = t - last_window_time # negative value if the window time being displayed is in the future
                if debug_print:
                    print(f'for current time t={t}\n\tlast_window_index: {last_window_index}\n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}')
                return (last_window_time, *list(np.squeeze(active_most_likely_positions[:, last_window_index]).copy()))

            t = end_t # the t under consideration should always be the end_t. This is written this way just for compatibility with the self.sigOnUpdateMeshes (float, float) signature
            curr_t, curr_x, curr_y = _get_nearest_decoded_most_likely_position_callback(t)
            curr_debug_point = [curr_x, curr_y, self.z_fixed[-1]]
            if debug_print:
                print(f'tcurr_debug_point: {curr_debug_point}') # \n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}
            self.perform_plot_location_point('decoded_position_point_plot', curr_debug_point, color='r', render=True)
            return curr_debug_point

        _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0]) # initialize by calling the callback with the current time
        # _conn = pg.SignalProxy(self.sigOnUpdateMeshes, rateLimit=14, slot=_update_nearest_decoded_most_likely_position_callback)
        _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

        # TODO: need to assign these results to somewhere in self. Not sure if I need to retain a reference to `active_one_step_decoder`
        # self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor']

        return _update_nearest_decoded_most_likely_position_callback, _conn # return the callback and the connection

    
    @property
    def decoded_trajectory_pyvista_plotter(self) -> DecodedTrajectoryPyVistaPlotter:
        """The decoded_trajectory_pyvista_plotter property."""
        return self.params['decoded_trajectory_pyvista_plotter']


    @function_attributes(short_name=None, tags=['probability'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryPyVistaPlotter'], used_by=[], creation_date='2025-01-29 07:35', related_items=[])
    def add_decoded_posterior_bars(self, a_result: DecodedFilterEpochsResult, xbin: NDArray, xbin_centers: NDArray, ybin: Optional[NDArray], ybin_centers: Optional[NDArray], enable_plot_all_time_bins_in_epoch_mode:bool=True, active_plot_fn=None):
        """ adds the decoded posterior to the PyVista plotter
         
          
        Usage:

            a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = iplapsDataExplorer.add_decoded_posterior_bars(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)

        """
        
        a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=self.p, curr_epoch_idx=0, curr_time_bin_index=0, enable_plot_all_time_bins_in_epoch_mode=enable_plot_all_time_bins_in_epoch_mode,
                                                                                                                active_plot_fn=active_plot_fn)
        a_decoded_trajectory_pyvista_plotter.build_ui()
        self.params['decoded_trajectory_pyvista_plotter'] = a_decoded_trajectory_pyvista_plotter
        return a_decoded_trajectory_pyvista_plotter
    

    def clear_all_added_decoded_posterior_plots(self, clear_ui_elements_also: bool = False):
        """ clears the plotted posterior actors and optionally the control sliders
        
        """
        if ('decoded_trajectory_pyvista_plotter' in self.params) and (self.decoded_trajectory_pyvista_plotter is not None):
            self.decoded_trajectory_pyvista_plotter.perform_clear_existing_decoded_trajectory_plots()
            
            ## can remove the UI (sliders and such) via:
            if clear_ui_elements_also:
                if self.decoded_trajectory_pyvista_plotter.slider_epoch is not None:
                    self.decoded_trajectory_pyvista_plotter.slider_epoch.RemoveAllObservers()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch.Off()
                    # a_decoded_trajectory_pyvista_plotter.slider_epoch.FastDelete()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch = None


                if self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin is not None:
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.RemoveAllObservers()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.Off()
                    # a_decoded_trajectory_pyvista_plotter.slider_epoch_time_bin.FastDelete()
                    self.decoded_trajectory_pyvista_plotter.slider_epoch_time_bin = None
                    

                self.decoded_trajectory_pyvista_plotter.p.clear_slider_widgets()

            self.decoded_trajectory_pyvista_plotter.p.update()
            self.decoded_trajectory_pyvista_plotter.p.render()



    @classmethod
    def perform_plot_posterior_fn(cls, p, xbin, ybin, xbin_centers, ybin_centers, posterior_p_x_given_n, time_bin_centers=None, enable_point_labels: bool = True, point_labeling_function=None, point_masking_function=None, posterior_name='P_x_given_n', active_plot_fn=None):
        """ called to perform the mesh generation and add_mesh calls
        
        Looks like it switches between 3 different potential plotting functions, all imported directly below

        ## Defaults to `plot_3d_binned_bars` if nothing else is provided        
        
        """
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_3d_stem_points, plot_point_labels

        if active_plot_fn is None:
            ## Defaults to `plot_3d_binned_bars` if nothing else is provided     

            active_plot_fn = plot_3d_binned_bars
            # active_plot_fn = plot_3d_stem_points
        
        if active_plot_fn.__name__ == plot_3d_stem_points.__name__:
            active_xbins = xbin_centers
            active_ybins = ybin_centers
        else:
            # required for `plot_3d_binned_bars`
            active_xbins = xbin
            active_ybins = ybin

        is_single_time_bin_posterior_plot: bool = (np.ndim(posterior_p_x_given_n) < 3)
        if is_single_time_bin_posterior_plot:
        
            # plotActors, data_dict = active_plot_fn(p, xbin, ybin, posterior_p_x_given_n, drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)
            plotActors, data_dict = active_plot_fn(p, active_xbins, active_ybins, posterior_p_x_given_n, drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)

            # , **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75} | kwargs)

            if point_labeling_function is None:
                # The full point shown:
                # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
                # Only the z-values
                point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'

            if point_masking_function is None:
                # point_masking_function = lambda points: points[:, 2] > 20.0
                point_masking_function = lambda points: points[:, 2] > 1E-6

            if enable_point_labels:
                plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(p, xbin_centers, ybin_centers, posterior_p_x_given_n, 
                                                                                    point_labels=point_labeling_function, 
                                                                                    point_mask=point_masking_function,
                                                                                    shape='rounded_rect', shape_opacity= 0.5, show_points=False, name=f'{posterior_name}Labels')
            else:
                plotActors_CenterLabels, data_dict_CenterLabels = None, None

        else:
            ## multi-time bin plot:
            from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars_timeseries

            assert np.ndim(posterior_p_x_given_n) == 3

            plotActors, data_dict = plot_3d_binned_bars_timeseries(p=p, xbin=active_xbins, ybin=active_ybins, t_bins=time_bin_centers, data=posterior_p_x_given_n,
                                           drop_below_threshold=1E-6, name=posterior_name, opacity=0.75, active_plot_fn=active_plot_fn)
            
            if enable_point_labels:
                print(f'WARN: enable_point_labels is not currently implemented for multi-time-bin plotting mode.')

            plotActors_CenterLabels, data_dict_CenterLabels = None, None



        return (plotActors, data_dict), (plotActors_CenterLabels, data_dict_CenterLabels)


