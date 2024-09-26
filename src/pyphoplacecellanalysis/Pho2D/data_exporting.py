from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from typing import NewType
import neuropy.utils.type_aliases as types

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps, ImageFilter # for export_array_as_image


from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult


# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack
from PIL import Image, ImageOps, ImageFilter # for export_array_as_image



@metadata_attributes(short_name=None, tags=['export', 'helper', 'static'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-11 07:35', related_items=[])
class PosteriorExporting:
    """ 2024-01-23 - Writes the posteriors out to file 
    
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
    
    
    
    """
    @classmethod
    def save_posterior_to_video(cls, a_decoder_continuously_decoded_result: DecodedFilterEpochsResult, result_name: str='a_decoder_continuously_decoded_result'):
        """ 
        
        Usage:
            from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
            
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
            all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
            pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
            spikes_df = directional_decoders_decode_result.spikes_df
            continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
            previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
            print(F'previously_decoded time_bin_sizes: {previously_decoded_keys}')
            # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
            time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
            print(f'time_bin_size: {time_bin_size}')

            continuously_decoded_dict = directional_decoders_decode_result.most_recent_continuously_decoded_dict
            pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('pseudo2D', None)
            pseudo2D_decoder_continuously_decoded_result

            a_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('long_LR', None)

            a_decoder_continuously_decoded_result: DecodedFilterEpochsResult = continuously_decoded_dict.get('long_LR', None)
            save_posterior_to_video(a_decoder_continuously_decoded_result=a_decoder_continuously_decoded_result, result_name='continuous_long_LR')

            save_posterior_to_video(a_decoder_continuously_decoded_result=pseudo2D_decoder_continuously_decoded_result, result_name='continuous_pseudo2D')

        """
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_video
        

        a_p_x_given_n = deepcopy(a_decoder_continuously_decoded_result.p_x_given_n_list[0]) # (57, 4, 83755) (n_x_bins, n_decoders, n_time_bins)
        if np.ndim(a_p_x_given_n) > 2:
            n_x_bins, n_decoders, n_time_bins = np.shape(a_p_x_given_n)
            transpose_axes_tuple = (2, 1, 0,)
        else:
            assert np.ndim(a_p_x_given_n) == 2, f"np.ndim(a_p_x_given_n): {np.ndim(a_p_x_given_n)}"
            n_x_bins, n_time_bins = np.shape(a_p_x_given_n)
            a_p_x_given_n = a_p_x_given_n[:, np.newaxis, :]
            assert np.ndim(a_p_x_given_n) == 3, f"np.ndim(a_p_x_given_n): {np.ndim(a_p_x_given_n)}"
            # transpose_axes_tuple = (1, 0,)
            transpose_axes_tuple = (2, 1, 0,)
        
            a_p_x_given_n = np.tile(a_p_x_given_n, (1, 8, 1,))
            # display(a_p_x_given_n)
        # time_window_centers = deepcopy(a_decoder_continuously_decoded_result.time_window_centers[0])

        ## get tiny portion just to test
        # a_p_x_given_n = a_p_x_given_n[:, :, :2000]
        # a_p_x_given_n

        # a_p_x_given_n = np.reshape(a_p_x_given_n, (n_time_bins, n_decoders, n_x_bins))
        a_p_x_given_n = np.transpose(a_p_x_given_n, transpose_axes_tuple)
        # display(a_p_x_given_n)

        decoding_realtime_FPS: float = 1.0 / float(a_decoder_continuously_decoded_result.decoding_time_bin_size)
        print(f'decoding_realtime_FPS: {decoding_realtime_FPS}')
        ## save video
        video_out_path = save_array_as_video(array=a_p_x_given_n, video_filename=f'output/videos/{result_name}.avi', isColor=False, fps=decoding_realtime_FPS)
        print(f'video_out_path: {video_out_path}')
        # reveal_in_system_file_manager(video_out_path)
        return video_out_path



    @function_attributes(short_name=None, tags=['figure', 'save', 'IMPORTANT', 'marginal'], input_requires=[], output_provides=[], uses=[], used_by=['save_marginals_arrays_as_image'], creation_date='2024-01-23 00:00', related_items=[])
    @classmethod
    def save_posterior(cls, raw_posterior_epochs_marginals, epochs_directional_marginals, epochs_track_identity_marginals, collapsed_per_epoch_epoch_marginal_dir_point, collapsed_per_epoch_marginal_track_identity_point,
        parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'lap', epoch_id: int = 9, export_all_raw_marginals_separately:bool = False, base_image_height: float=100, include_value_labels: bool = True, allow_override_aspect_ratio:bool=True, debug_print:bool=True) -> Tuple[Tuple[Image.Image, Path], Tuple[Image.Image, Path], Tuple[Image.Image, Path], Tuple[Image.Image, Path], List[Tuple[Image.Image, Path]]]:
        """ 2024-01-23 - Writes the posteriors out to file 
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import save_posterior

            collapsed_per_lap_epoch_marginal_track_identity_point = laps_marginals_df[['P_Long', 'P_Short']].to_numpy().astype(float)
            collapsed_per_lap_epoch_marginal_dir_point = laps_marginals_df[['P_LR', 'P_RL']].to_numpy().astype(float)

            for epoch_id in np.arange(laps_filter_epochs_decoder_result.num_filter_epochs):
                raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple = save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point,
                                                                                            parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='lap', epoch_id=epoch_id)

        """
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image, get_array_as_image, vertical_image_stack, horizontal_image_stack
        
        assert parent_array_as_image_output_folder.exists()
        
        _out_save_tuples: List[Tuple[Image.Image, Path]] = []
        
        epoch_id_str: str = f"{epoch_id_identifier_str}[{epoch_id}]"
        _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_raw_marginal.png').resolve()

        img_data = raw_posterior_epochs_marginals[epoch_id]
        if not isinstance(img_data, NDArray):
            img_data = img_data['p_x_given_n']
        img_data = img_data.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        # img_data = raw_posterior_laps_marginals[epoch_id]['p_x_given_n'].astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        if debug_print:
            print(f'np.shape(raw_posterior_laps_marginals[{epoch_id}]["p_x_given_n"]): {np.shape(img_data)}')

        base_image_height: int = int(round(base_image_height))
        desired_half_height: int = int(round(base_image_height/2))
        # get_array_as_img_kwargs = dict(desired_width=None, skip_img_normalization=False, include_value_labels=include_value_labels)
        get_array_as_img_kwargs = dict(desired_width=None, skip_img_normalization=True, include_value_labels=include_value_labels) # do not manually normalize the image before output. 
        save_array_as_image_kwargs = dict(include_value_labels=include_value_labels, allow_override_aspect_ratio=allow_override_aspect_ratio)

        if np.ndim(img_data) > 2:
            n_x_bins, n_decoders, n_curr_epoch_time_bins = np.shape(img_data)
            # img_data_array = [np.atleast_2d(np.squeeze(img_data[:,:, i])).T for i in np.arange(n_curr_epoch_time_bins)]
            raw_posterior_array = [np.atleast_2d(np.squeeze(img_data[:,:, i])) for i in np.arange(n_curr_epoch_time_bins)]
                    
            imgs_array = [get_array_as_image(raw_posterior_array[i].T, desired_height=base_image_height, **get_array_as_img_kwargs) for i in np.arange(n_curr_epoch_time_bins)]            
            if export_all_raw_marginals_separately:
                _sub_img_parent_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_raw_marginal').resolve()
                _sub_img_parent_path.mkdir(parents=False, exist_ok=True)
                for i, (a_single_time_bin_raw_posterior, an_img) in enumerate(zip(raw_posterior_array, imgs_array)):
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{epoch_id_str}_raw_marginal_{i}.png').resolve()
                    an_img.save(_sub_img_path) # Save image to file
                    _out_save_tuples.append((an_img, _sub_img_path))
                    
                    if debug_print:
                        print(f'i: {i}, np.shape(a_single_time_bin_raw_posterior): {np.shape(a_single_time_bin_raw_posterior)}') # (n_x_bins, n_decoders)
                    _decoder_prob_arr = np.sum(a_single_time_bin_raw_posterior, axis=0) # get the four-tuple of decoder probabilities ... should this be axis=0? ... no because it's transposed... how confusing
                    # _decoder_prob_arr = _decoder_prob_arr / np.nansum(_decoder_prob_arr) # normalize
                    if debug_print:
                        print(f'\tnp.shape(_decoder_prob_arr): {np.shape(_decoder_prob_arr)}, _decoder_prob_arr: {_decoder_prob_arr}')
                    _decoder_prob_img: Image.Image = get_array_as_image(np.atleast_2d(_decoder_prob_arr).T, desired_height=base_image_height, **get_array_as_img_kwargs)
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{epoch_id_str}_marginal_decoder_{i}.png').resolve()
                    _decoder_prob_img.save(_sub_img_path) # Save image to file
                    _out_save_tuples.append((_decoder_prob_img, _sub_img_path))
                    
                    # _long_arr = np.sum(a_single_time_bin_raw_posterior[[0,1], :], axis=0) # two long rows
                    # _short_arr = np.sum(a_single_time_bin_raw_posterior[[2,3], :], axis=0) # the two short rows
                    # _long_arr = np.sum(a_single_time_bin_raw_posterior[:, [0,1]], axis=1) # two long rows
                    # _short_arr = np.sum(a_single_time_bin_raw_posterior[:, [2,3]], axis=1) # the two short rows
                    
                    # if debug_print:
                    #     print(f'\tnp.shape(_long_arr): {np.shape(_long_arr)}')

                    # _long_arr = _long_arr / np.nansum(_long_arr) # normalize
                    # _short_arr = _short_arr / np.nansum(_short_arr) # normalize

                    ## Compute marginal:
                    _long_any: float = np.sum(a_single_time_bin_raw_posterior[:, [2,3]]) # long are the upper-two rows
                    _short_any: float = np.sum(a_single_time_bin_raw_posterior[:, [0,1]]) # short are the lower-two rows
                    
                    # _long_any: float = np.sum(_long_arr)
                    # _short_any: float = np.sum(_short_arr)

                    # skipping _direction_marginal_2tuple
                    # _track_marginal_2tuple = np.atleast_2d(np.array([_long_any, _short_any])).T
                    _track_marginal_2tuple = np.atleast_2d(np.array([_short_any, _long_any]))
                    # _track_marginal_2tuple = _track_marginal_2tuple / np.nansum(_track_marginal_2tuple) # normalize
                    _temp_img: Image.Image = get_array_as_image(_track_marginal_2tuple.T, desired_height=desired_half_height, **get_array_as_img_kwargs)
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{epoch_id_str}_track_marginal_two_tuple_{i}.png').resolve()
                    _temp_img.save(_sub_img_path) # Save image to file
                    _out_save_tuples.append((_temp_img, _sub_img_path))
                          
                    if debug_print:
                        print(f'\t_long_any: {_long_any}, _short_any: {_short_any}')
                # end for
                
                # ## Build a composite image stack from the separate raw marginals
                # grid = [imgs_array]
                # composite_image = create_image_grid(grid, h_spacing=10, v_spacing=10)
                # _sub_img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_raw_marginal_COMPOSITE.png').resolve()
                # composite_image.save(_sub_img_path)
                
                
            # output_img = get_array_as_image_stack(imgs=imgs_array,
            #                                     offset=25, single_image_alpha_level=0.5,
            #                                     should_add_border=True, border_size=1, border_color=(255, 255, 255),
            #                                     should_add_shadow=True, shadow_offset=1, shadow_color=(255,255,255,100))
            # output_img.save(_img_path)
            # raw_tuple = (output_img, _img_path,)
            # image_raw, path_raw = raw_tuple
        else:
            ## 2D output
            # n_x_bins, n_decoders, n_curr_epoch_time_bins = np.shape(img_data)
            raw_tuple = save_array_as_image(img_data, desired_height=base_image_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)
            _out_save_tuples.append(raw_tuple)
            
            # image_raw, path_raw = raw_tuple
            
        # [2 x n_epoch_t_bins] matrix
        _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir.png').resolve()
        img_data = epochs_directional_marginals[epoch_id]['p_x_given_n'].astype(float)
        marginal_dir_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)
        # image_marginal_dir, path_marginal_dir = marginal_dir_tuple

        # [2 x n_epoch_t_bins] matrix
        _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity.png').resolve()
        img_data = epochs_track_identity_marginals[epoch_id]['p_x_given_n'].astype(float)
        marginal_track_identity_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)
        # image_marginal_track_identity, path_marginal_track_identity = marginal_track_identity_tuple

        # 2-vector
        _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity_point.png').resolve()
        img_data = np.atleast_2d(collapsed_per_epoch_marginal_track_identity_point[epoch_id,:]).T
        marginal_dir_point_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)

        # 2-vector
        _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir_point.png').resolve()
        img_data = np.atleast_2d(collapsed_per_epoch_epoch_marginal_dir_point[epoch_id,:]).T
        marginal_track_identity_point_tuple: Tuple[Image.Image, Path] = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)

        
        return marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple, _out_save_tuples
        


    @function_attributes(short_name=None, tags=['export','marginal', 'pseudo2D', 'IMPORTANT'], input_requires=[], output_provides=[], uses=['save_posterior'], used_by=[], creation_date='2024-09-10 00:06', related_items=[])
    @classmethod
    def save_marginals_arrays_as_image(cls, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'ripple', epoch_ids=None, export_all_raw_marginals_separately: bool=True, include_value_labels:bool=False, allow_override_aspect_ratio:bool=True, debug_print=False):
        """ Exports all the raw_merged pseudo2D posteriors and their marginals out to image files.
        
        Usage: 
        
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import save_marginals_arrays_as_image
            directional_merged_decoders_result.perform_compute_marginals()
            parent_array_as_image_output_folder = Path('output/Exports/array_as_image').resolve()
            parent_array_as_image_output_folder.mkdir(parents=True, exist_ok=True)
            assert parent_array_as_image_output_folder.exists()
            save_marginals_arrays_as_image(directional_merged_decoders_result=directional_merged_decoders_result, parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='ripple', epoch_ids=[31])

        """
        assert epoch_id_identifier_str in ['ripple', 'lap']
        active_marginals_df: pd.DataFrame = deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)
        active_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result)

        raw_posterior_active_marginals: List = deepcopy(active_filter_epochs_decoder_result.p_x_given_n_list)
        if debug_print:
            print(f'len(raw_posterior_active_marginals): {len(raw_posterior_active_marginals)}')
        collapsed_per_epoch_marginal_track_identity_point = active_marginals_df[['P_Long', 'P_Short']].to_numpy().astype(float)
        collapsed_per_epoch_marginal_dir_point = active_marginals_df[['P_LR', 'P_RL']].to_numpy().astype(float)
        if debug_print:
            print(f'collapsed_per_epoch_marginal_track_identity_point.shape: {np.shape(collapsed_per_epoch_marginal_track_identity_point)}')

        
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir = directional_merged_decoders_result.ripple_directional_marginals_tuple
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = directional_merged_decoders_result.ripple_track_identity_marginals_tuple

        # raw_posterior_laps_marginals
        # raw_posterior_active_marginals = directional_merged_decoders_result.build_non_marginalized_raw_posteriors(active_filter_epochs_decoder_result)
        # raw_posterior_active_marginals

        # INPUTS:
        # raw_posterior_laps_marginals = deepcopy(raw_posterior_laps_marginals)
        # active_directional_marginals = deepcopy(laps_directional_marginals)
        # active_track_identity_marginals = deepcopy(laps_track_identity_marginals)
        # raw_posterior_active_marginals = deepcopy(raw_posterior_laps_marginals)
        active_directional_marginals = deepcopy(ripple_directional_marginals)
        active_track_identity_marginals = deepcopy(ripple_track_identity_marginals)

        assert parent_array_as_image_output_folder.exists()

        if epoch_ids is None:
            epoch_ids = np.arange(active_filter_epochs_decoder_result.num_filter_epochs)

        out_tuple_dict = {}
        for epoch_id in epoch_ids:
            # Make epoch folder
            _curr_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_identifier_str}_{epoch_id}').resolve()
            _curr_path.mkdir(exist_ok=True)
                                                     
            out_tuple_dict[epoch_id] = cls.save_posterior(raw_posterior_active_marginals, active_directional_marginals,
                                                                                active_track_identity_marginals, collapsed_per_epoch_marginal_dir_point, collapsed_per_epoch_marginal_track_identity_point,
                                                                                        parent_array_as_image_output_folder=_curr_path, epoch_id_identifier_str=epoch_id_identifier_str, epoch_id=epoch_id,
                                                                                        export_all_raw_marginals_separately=export_all_raw_marginals_separately, include_value_labels=include_value_labels, allow_override_aspect_ratio=allow_override_aspect_ratio,
                                                                                        debug_print=debug_print)
        return out_tuple_dict
        
            
    @classmethod
    def _test_export_marginals_for_figure(cls, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], clicked_epoch: NDArray, context_specific_root_export_path: Path, epoch_specific_folder: Path, epoch_id_identifier_str='ripple', debug_print=True, allow_override_aspect_ratio:bool=True, **kwargs):
        """
        
                epoch_id_identifier_str='ripple'

        """
        
        # root_export_path = Path(r"E:\Dropbox (Personal)\Active\Kamran Diba Lab\Pho-Kamran-Meetings\2024-05-01 - Pseudo2D Again\array_as_image").resolve() # Apogee
        # # root_export_path = Path('/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image').resolve() # Lab
        
        # ## Session-specific folder:
        # context_specific_root_export_path = root_export_path.joinpath(curr_context.get_description(separator='_')).resolve()
        # context_specific_root_export_path.mkdir(exist_ok=True)
        # assert context_specific_root_export_path.exists()

        # # Epoch-specific folder:
        # ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"ripple_{active_epoch_dict['ripple_idx']}").resolve()
        # ripple_specific_folder.mkdir(exist_ok=True)
        # assert ripple_specific_folder.exists()
        
        
        # # ==================================================================================================================== #
        # # Export Rasters:                                                                                                      #
        # # ==================================================================================================================== #
        
        # # Get the clicked epoch from the _out_ripple_rasters GUI _____________________________________________________________ #
        # active_epoch_tuple = deepcopy(_out_ripple_rasters.active_epoch_tuple)
        # active_epoch_dict = {k:getattr(active_epoch_tuple, k) for k in ['start', 'stop', 'ripple_idx', 'Index']} # , 'session_name', 'time_bin_size', 'delta_aligned_start_t' {'start': 1161.0011335673044, 'stop': 1161.274357107468, 'session_name': '2006-6-09_1-22-43', 'time_bin_size': 0.025, 'delta_aligned_start_t': 131.68452480540145}
        # # clicked_epoch = np.array([169.95631618227344, 170.15983607806265])
        # # clicked_epoch = np.array([91.57839279191103, 91.857145929])
        # clicked_epoch = np.array([active_epoch_dict['start'], active_epoch_dict['stop']])
        # # OUTPUTS: clicked_epoch
        
        # # Save out the actual raster-plots ___________________________________________________________________________________ #
        # _out_rasters_save_paths = _out_ripple_rasters.save_figure(export_path=ripple_specific_folder)
        # _out_rasters_save_paths
        
        ## OUTPUTS: ripple_specific_folder, _out_rasters_save_paths

        
        # ==================================================================================================================== #
        # Export Pseudo2D Step-by-Step Raw Posteriors and their Marginalizations over Track Identity                           #
        # ==================================================================================================================== #

        ## INPUTS: directional_merged_decoders_result, context_specific_root_export_path, 
        
        directional_merged_decoders_result.perform_compute_marginals()
        assert context_specific_root_export_path.exists()
        print(f'parent_array_as_image_output_folder: "{context_specific_root_export_path}"')

        ## INPUTS: clicked_epoch, 
        if epoch_id_identifier_str == 'ripple':
            active_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result)
        elif epoch_id_identifier_str == 'lap':
            active_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
        else:
            raise NotImplementedError(f'epoch_id_identifier_str: {epoch_id_identifier_str}')
        
        epoch_ids = active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) # [262], [296]
        if debug_print:
            print(f'epoch_ids: {epoch_ids}')

        if len(epoch_ids) < 1:
            print(f'WARN: no found epoch_ids from the provided clicked_epoch: {clicked_epoch}!!')
        assert (len(epoch_ids) > 0), f"no found epoch_ids from the provided clicked_epoch: {clicked_epoch}"

        ## Sanity check:
        if debug_print:
            curr_epoch_result: SingleEpochDecodedResult = active_filter_epochs_decoder_result.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
            print(f"curr_epoch_result.epoch_info_tuple: {curr_epoch_result.epoch_info_tuple}")
            print(f"\tnbins: {curr_epoch_result.nbins}")
            print(f'\tactive_filter_epochs_decoder_result.decoding_time_bin_size: {active_filter_epochs_decoder_result.decoding_time_bin_size}')
            # curr_epoch_result.time_bin_container

        # active_filter_epochs_decoder_result.all_directional_ripple_filter_epochs_decoder_result
        out_image_save_tuple_dict = cls.save_marginals_arrays_as_image(directional_merged_decoders_result=directional_merged_decoders_result, parent_array_as_image_output_folder=context_specific_root_export_path,
                                                          epoch_id_identifier_str=epoch_id_identifier_str, epoch_ids=epoch_ids, debug_print=True, include_value_labels=False, allow_override_aspect_ratio=allow_override_aspect_ratio,# **kwargs,
                                                        )

        # ==================================================================================================================== #
        # Export Decoded Position Posteriors                                                                                   #
        # ==================================================================================================================== #
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
        ## INPUTS: filtered_decoder_filter_epochs_decoder_result_dict, parent_array_as_image_output_folder, epoch_id_identifier_str='ripple', 

        # assert parent_array_as_image_output_folder.exists()
        assert epoch_specific_folder.exists()

        epoch_id = epoch_ids[0] # assume a single epoch idx
        if debug_print:
            print(f'epoch_idx: {epoch_id}')
        epoch_id_str: str = f"{epoch_id_identifier_str}[{epoch_id}]"
        if debug_print:
            print(f'epoch_id_str: {epoch_id_str}')

        # # Make epoch folder
        # ripple_specific_folder = parent_array_as_image_output_folder.joinpath(f'{epoch_id_identifier_str}_{epoch_id}').resolve()
        # ripple_specific_folder.mkdir(exist_ok=True)

        out_image_paths = {}
        for k, v in filtered_decoder_filter_epochs_decoder_result_dict.items():
            # v: DecodedFilterEpochsResult
            a_result: SingleEpochDecodedResult = v.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
            print(f"{k}: filtered_decoder_filter_epochs_decoder_result_dict[{k}].decoding_time_bin_size: {v.decoding_time_bin_size}") # 0.016!! 
            _img_path = epoch_specific_folder.joinpath(f'{epoch_id_str}_posterior_{k}.png').resolve()
            a_result.save_posterior_as_image(_img_path, colormap='Oranges', allow_override_aspect_ratio=allow_override_aspect_ratio, flip_vertical_axis=True, **kwargs)
            out_image_paths[k] = _img_path

        return out_image_save_tuple_dict, out_image_paths


    @classmethod
    def _perform_export_current_epoch_marginal_and_raster_images(cls, _out_ripple_rasters, directional_merged_decoders_result, filtered_decoder_filter_epochs_decoder_result_dict, active_session_context, root_export_path: Path, epoch_id_identifier_str='lap',
                                                                 desired_width = 2048, desired_height = 720, debug_print=False
                                                                 ):
        """ captures: _out_ripple_rasters, directional_merged_decoders_result, (filtered_decoder_filter_epochs_decoder_result_dict, decoder_laps_filter_epochs_decoder_result_dict)
        """

        # Get the clicked epoch from the _out_ripple_rasters GUI _____________________________________________________________ #
        active_epoch_tuple = deepcopy(_out_ripple_rasters.active_epoch_tuple)
        if debug_print:
            print(f'active_epoch_tuple: {active_epoch_tuple}')
        # active_epoch_dict = {k:getattr(active_epoch_tuple, k) for k in ['start', 'stop', 'ripple_idx', 'Index']} # , 'session_name', 'time_bin_size', 'delta_aligned_start_t' {'start': 1161.0011335673044, 'stop': 1161.274357107468, 'session_name': '2006-6-09_1-22-43', 'time_bin_size': 0.025, 'delta_aligned_start_t': 131.68452480540145}
        active_epoch_dict = {k:getattr(active_epoch_tuple, k) for k in ['start', 'stop', 'Index']} # , 'session_name', 'time_bin_size', 'delta_aligned_start_t' {'start': 1161.0011335673044, 'stop': 1161.274357107468, 'session_name': '2006-6-09_1-22-43', 'time_bin_size': 0.025, 'delta_aligned_start_t': 131.68452480540145}
        # EpochTuple(Index=8, lap_idx=8, lap_start_t=499.299262000015, P_Long_LR=0.83978564760147, P_Long_RL=0.06863305250806279, P_Short_LR=0.08466212906826685, P_Short_RL=0.006919170822200387, most_likely_decoder_index=0, start=499.299262000015, stop=504.80590599996503, label='8', duration=5.506643999950029, lap_id=9, lap_dir=1, long_LR_pf_peak_x_pearsonr=-0.775475552508641, long_RL_pf_peak_x_pearsonr=-0.5082034915116096, short_LR_pf_peak_x_pearsonr=-0.7204573376193804, short_RL_pf_peak_x_pearsonr=-0.4648058215927542, best_decoder_index=0, session_name='2006-6-08_14-26-15', time_bin_size=0.25, delta_aligned_start_t=-712.2588180310559)

        # clicked_epoch = np.array([169.95631618227344, 170.15983607806265])
        # clicked_epoch = np.array([91.57839279191103, 91.857145929])
        if debug_print:
            print(f'clicked_epoch: {active_epoch_dict}')
        clicked_epoch = np.array([active_epoch_dict['start'], active_epoch_dict['stop']])
        # OUTPUTS: clicked_epoch
        if debug_print:
            print(f'clicked_epoch: {clicked_epoch}')

        ## Export Marginal Pseudo2D posteriors and rasters for middle-clicked epochs:

        # epoch_id_identifier_str='ripple'
        

        ## Session-specific folder:
        context_specific_root_export_path = root_export_path.joinpath(active_session_context.get_description(separator='_')).resolve()
        context_specific_root_export_path.mkdir(exist_ok=True)
        assert context_specific_root_export_path.exists()

        # Epoch-specific folder:
        # ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"ripple_{active_epoch_dict['ripple_idx']}").resolve()
        # ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"ripple_{active_epoch_dict['Index']}").resolve()
        ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"{epoch_id_identifier_str}_{active_epoch_dict['Index']}").resolve()
        ripple_specific_folder.mkdir(exist_ok=True)
        assert ripple_specific_folder.exists()
        # file_uri_from_path(ripple_specific_folder)
        # fullwidth_path_widget(a_path=ripple_specific_folder, file_name_label="lap_specific_folder:")

        # clicked_epoch: {'start': 105.40014315512963, 'stop': 105.56255971186329, 'ripple_idx': 8, 'Index': 8}
        # clicked_epoch: [105.4 105.563]
        # ripple_8

        # ==================================================================================================================== #
        # Export Rasters:                                                                                                      #
        # ==================================================================================================================== #

        # Save out the actual raster-plots ___________________________________________________________________________________ #
        _out_rasters_save_paths = _out_ripple_rasters.save_figure(export_path=ripple_specific_folder,
                                                                width=desired_width,
                                                                #    height=desired_height,
                                                                )
        # _out_rasters_save_paths

        # OUTPUTS: ripple_specific_folder, _out_rasters_save_paths
        out_image_save_tuple_dict = cls._test_export_marginals_for_figure(directional_merged_decoders_result=directional_merged_decoders_result,
                                                                                        filtered_decoder_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict, ## laps
                                                                                        clicked_epoch=clicked_epoch,
                                                                                        context_specific_root_export_path=context_specific_root_export_path, epoch_specific_folder=ripple_specific_folder,
                                                                                        epoch_id_identifier_str=epoch_id_identifier_str, debug_print=False, desired_width=desired_width, desired_height=desired_height)
        # out_image_save_tuple_dict

        ## INPUTS: _out_rasters_save_paths, out_image_save_tuple_dict[-1]

        # Open the images
        _raster_imgs = [Image.open(i) for i in _out_rasters_save_paths]
        # Open the images
        _posterior_imgs = [Image.open(i) for i in list(out_image_save_tuple_dict[-1].values())]
            
        # _out_hstack = horizontal_image_stack([vertical_image_stack([a_posterior_img, a_raster_img], padding=5, v_overlap=50) for a_raster_img, a_posterior_img in zip(_raster_imgs, _posterior_imgs)], padding=5)
        _out_all_decoders_posteriors_and_rasters_stack_image = horizontal_image_stack([vertical_image_stack([a_posterior_img, a_raster_img], padding=0) for a_raster_img, a_posterior_img in zip(_raster_imgs, _posterior_imgs)], padding=0) # no overlap
        _out_all_decoders_posteriors_and_rasters_stack_image

        # _out_all_decoders_posteriors_and_rasters_stack_image = horizontal_image_stack([vertical_image_stack([a_posterior_img, a_raster_img], padding=0, v_overlap=a_posterior_img.size[-1]) for a_raster_img, a_posterior_img in zip(_raster_imgs, _posterior_imgs)], padding=5) # posterior is inset to the top of the raster (raster image is taller).

        ## Save merged image:
        merged_img_save_path = _out_rasters_save_paths[0].parent.resolve().joinpath(f'all_decoders_posteriors_and_rasters_stack_image.png').resolve()
        _out_all_decoders_posteriors_and_rasters_stack_image.save(merged_img_save_path) # Save image to file
        print(f'saved image to: "{merged_img_save_path.as_posix()}"')
        # _out_save_tuples.append((_out_all_decoders_posteriors_and_rasters_stack_image, _sub_img_path))

        return ripple_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path)
