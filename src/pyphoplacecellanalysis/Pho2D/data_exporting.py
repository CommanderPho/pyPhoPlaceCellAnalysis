 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from typing import NewType
import neuropy.utils.type_aliases as types

from attrs import define, field, Factory, asdict

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

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Literal, NewType
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types

# Custom Type Definitions ____________________________________________________________________________________________ #
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)
# Define the type alias
KnownEpochsName = Literal['laps', 'ripple', 'other']

import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str

from PIL import Image, ImageOps, ImageFilter # for export_array_as_image
# from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
from pyphocorehelpers.assertion_helpers import Assert
from attrs import define, field, Factory

from enum import Enum, auto


if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

from pyphocorehelpers.plotting.media_output_helpers import ImagePostRenderFunctionSets, ImageStackOrientation  # used in `_subfn_build_combined_output_images`



@metadata_attributes(short_name=None, tags=['enum', 'export_mode'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-14 14:14', related_items=[])
class HeatmapExportKind(Enum):
    """Description of the enum class and its purpose.
    
    from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind
    
    
    """
    GREYSCALE = auto()
    COLORMAPPED = auto()
    RAW_RGBA = auto()

    def __str__(self):
        return self.name

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]




@define(slots=False, eq=False)
class HeatmapExportConfig:
    """ specifies a single configuration for exporitng a heatmap, such as a posterior"""
    colormap: Optional[str] = field()
    export_kind: HeatmapExportKind = field(default=HeatmapExportKind.COLORMAPPED)
    export_folder: Optional[Path] = field(default=None)
    export_grayscale: bool = field(default=False)
    desired_height: Optional[int] = field(default=None) 
    desired_width: Optional[int] = field(default=None)
    skip_img_normalization:bool = field(default=False)
    vmin: Optional[float] = field(default=None, metadata={'desc': 'used for colormap normalization. The min colormap value.'})
    vmax: Optional[float] = field(default=None, metadata={'desc': 'used for colormap normalization. The max colormap value.'})
    allow_override_aspect_ratio:bool = field(default=False)
    post_render_image_functions_builder_fn: Optional[Callable] = field(default=ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict, metadata={'desc': 'a function that builds the post-rendering image modification functions list'}) #List[Dict[str, Callable]]

    # lower_bound_alpha: float= field(default=0.1, metadata={'desc':'the lower bound for the alpha value for the color map. 0 means fully transparent, 1 means fully opaque.'})
    # drop_below_threshold: float = field(default=1e-3, metadata={'desc':'the threshold for the alpha value for the color map. 0 means fully transparent, 1 means fully opaque.'})
    raw_RGBA_only_parameters: Dict[str, Any] = field(default=Factory(dict))


    ## OUTPUTS:
    posterior_saved_image: Optional[Image.Image] = field(default=None, init=False)
    posterior_saved_path: Optional[Path] = field(default=None, init=False)

    def __attrs_post_init__(self):        
        self.export_grayscale = (self.export_kind.value == HeatmapExportKind.GREYSCALE.value) or ((self.export_kind.value != HeatmapExportKind.COLORMAPPED.value) and (self.colormap is None))
        
        # (not (self.export_kind.value == HeatmapExportKind.RAW_RGBA.value)) 
        if (self.colormap is None):
            assert (self.export_kind.value != HeatmapExportKind.COLORMAPPED.value), f"colormap should not be specified when export_grayscale=True"
            

    @classmethod
    def init_for_export_kind(cls, export_kind: HeatmapExportKind, **kwargs):
        """ initializes for a specific export kind """
        
        colormap = kwargs.pop('colormap', None)

        if export_kind.value == HeatmapExportKind.GREYSCALE.value:
            # if export_grayscale:
            return cls.init_greyscale(**kwargs)
            
        elif (export_kind.value == HeatmapExportKind.COLORMAPPED.value):
            ## Color export mode!
            assert (colormap is not None)
            return cls(colormap, export_grayscale=False, skip_img_normalization=kwargs.pop('skip_img_normalization', False), export_kind=HeatmapExportKind.COLORMAPPED, **kwargs)


        elif export_kind.value == HeatmapExportKind.RAW_RGBA.value:
            ## Raw ready to use RGBA image is passed in:
            # `raw_RGBA_only_parameters` dict with keys: ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
            raw_RGBA_only_parameters = kwargs.pop('raw_RGBA_only_parameters', {})
            required_keys = ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
            assert np.all([k in raw_RGBA_only_parameters.keys() for k in required_keys]), f"missing required keys:\n\tequired_keys: {required_keys}\n\tlist(raw_RGBA_only_parameters.keys()): {list(raw_RGBA_only_parameters.keys())}\n"
            
            return cls(colormap=None, export_grayscale=False, skip_img_normalization=True, export_kind=HeatmapExportKind.RAW_RGBA, raw_RGBA_only_parameters=raw_RGBA_only_parameters, **kwargs)
            
        else:
            raise NotImplementedError(f"export_kind: {export_kind}")    
        
    
    @classmethod
    def init_greyscale(cls, **kwargs):
        desired_colormap = kwargs.pop('colormap', None)
        assert desired_colormap is None
        _obj = cls(colormap=None, export_grayscale=True, skip_img_normalization=kwargs.pop('skip_img_normalization', False), export_kind=HeatmapExportKind.GREYSCALE, **kwargs) # is_greyscale=True, 
        return _obj
    
    

    def to_dict(self) -> Dict:
        filter_fn = lambda attr, value: attr.name not in ["export_folder", "posterior_saved_image", "posterior_saved_path"]
        return asdict(deepcopy(self), recurse=False, filter=filter_fn)


@metadata_attributes(short_name=None, tags=['export', 'helper', 'static'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-11 07:35', related_items=[])
class PosteriorExporting:
    """ 2024-01-23 - Writes the posteriors out to file 
    
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
    
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
        ## Exports: "2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(decoded_posteriors).h5"
		print(f'save_hdf == True, so exporting posteriors to HDF file...')
		# parent_output_path = self.collected_outputs_path.resolve()
		save_path: Path = PosteriorExporting.build_custom_export_to_h5_path(curr_active_pipeline, output_date_str=None, data_identifier_str='(decoded_posteriors)', a_tbin_size=directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size, parent_output_path=self.collected_outputs_path.resolve())
		# save_path = Path(f'output/{BATCH_DATE_TO_USE}_newest_all_decoded_epoch_posteriors.h5').resolve()
		complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
		_, _, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
		custom_params_hdf_key: str = custom_suffix.strip('_') # strip leading/trailing underscores
		# _parent_save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5', custom_suffix=custom_suffix)
		_parent_save_context: DisplaySpecifyingIdentifyingContext = deepcopy(session_context).overwriting_context(custom_suffix=custom_params_hdf_key, display_fn_name='save_decoded_posteriors_to_HDF5')
		# _parent_save_context: DisplaySpecifyingIdentifyingContext = complete_session_context.overwriting_context(display_fn_name='save_decoded_posteriors_to_HDF5')
		_parent_save_context.display_dict = {
			'custom_suffix': lambda k, v: f"{v}", # just include the name
			'display_fn_name': lambda k, v: f"{v}", # just include the name
		}
		out_contexts, _flat_all_HDF5_out_paths = PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(decoder_laps_filter_epochs_decoder_result_dict=None,
																					decoder_ripple_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict),
																					_save_context=_parent_save_context.get_raw_identifying_context(), save_path=save_path, should_overwrite_extant_file=(not allow_append_to_session_h5_file))

		_flat_all_HDF5_out_paths = list(dict.fromkeys([v.as_posix() for v in _flat_all_HDF5_out_paths]).keys())
		# _output_HDF5_paths_info_str: str = '\n'.join([f'"{file_uri_from_path(a_path)}"' for a_path in _flat_all_HDF5_out_paths])
		_output_HDF5_paths_info_str: str = '\n'.join([f'"{a_path}"' for a_path in _flat_all_HDF5_out_paths])
		# print(f'\t\t\tHDF5 Paths: {_flat_all_HDF5_out_paths}\n')
		print(f'\t\t\tHDF5 Paths: {_output_HDF5_paths_info_str}\n')
        
    """
    @classmethod
    def build_custom_export_to_h5_path(cls, curr_active_pipeline, output_date_str: Optional[str]=None, data_identifier_str: str = f'(decoded_posteriors)', a_tbin_size: float=None, parent_output_path: Path=None):
        """ captures CURR_BATCH_DATE_TO_USE, `curr_active_pipeline`
        
        based off of `_subfn_build_custom_export_to_h5_path`
        
        Usage:
        
        PosteriorExporting.build_custom_export_to_h5_path(
        
        """
        if (output_date_str is None) or (len(output_date_str) < 1):
            output_date_str = get_now_rounded_time_str(rounded_minutes=10)
            
        if (a_tbin_size is not None):
            ## add optional time bin suffix:
            a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
            a_data_identifier_str: str = f'{data_identifier_str}_tbin-{a_tbin_size_str}' ## build the identifier '(decoded_posteriors)_tbin-1.5'
            
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=a_data_identifier_str, parent_output_path=parent_output_path, out_extension='.h5')
        return out_path 
    

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

        ## Check the parent path exists, as it previously failed silently (after doing all the work) if the directory wasn't present
        
        ## save video
        video_out_path = save_array_as_video(array=a_p_x_given_n, video_filename=f'output/videos/{result_name}.avi', isColor=False, fps=decoding_realtime_FPS)
        print(f'video_out_path: {video_out_path}')
        # reveal_in_system_file_manager(video_out_path)
        return video_out_path

    @function_attributes(short_name=None, tags=['figure', 'save', 'IMPORTANT', 'marginal', 'single-export-format'], input_requires=[], output_provides=[], uses=[], used_by=['save_marginals_arrays_as_image'], creation_date='2024-01-23 00:00', related_items=[])
    @classmethod
    def save_posterior(cls, raw_posterior_epochs_marginals, epochs_directional_marginals, epochs_track_identity_marginals, collapsed_per_epoch_epoch_marginal_dir_point, collapsed_per_epoch_marginal_track_identity_point,
        parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'lap', epoch_IDX: int = 9, complete_epoch_identifier_str: Optional[str]=None, export_all_raw_marginals_separately:bool = False, base_image_height: float=100, export_kind: Optional[HeatmapExportKind] = None, include_value_labels: bool = True, allow_override_aspect_ratio:bool=True, debug_print:bool=False) -> Tuple[Tuple[Image.Image, Path], Tuple[Image.Image, Path], Tuple[Image.Image, Path], Tuple[Image.Image, Path], List[Tuple[Image.Image, Path]]]:
        """ 2024-01-23 - Writes the posteriors out to file 
        
        Usage:
            from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting

            collapsed_per_lap_epoch_marginal_track_identity_point = laps_marginals_df[['P_Long', 'P_Short']].to_numpy().astype(float)
            collapsed_per_lap_epoch_marginal_dir_point = laps_marginals_df[['P_LR', 'P_RL']].to_numpy().astype(float)

            for epoch_id in np.arange(laps_filter_epochs_decoder_result.num_filter_epochs):
                raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple = PosteriorExporting.save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point,
                                                                                            parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='lap', epoch_id=epoch_id)

        """
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image, get_array_as_image, vertical_image_stack, horizontal_image_stack
        
        assert parent_array_as_image_output_folder.exists()
        
        _out_save_tuples: List[Tuple[Image.Image, Path]] = []
        
        if complete_epoch_identifier_str is None:
            complete_epoch_identifier_str: str = f"{epoch_id_identifier_str}[{epoch_IDX}]"


        _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_raw_marginal.png').resolve()

        img_data = raw_posterior_epochs_marginals[epoch_IDX]
        if not isinstance(img_data, NDArray):
            img_data = img_data['p_x_given_n']
        img_data = img_data.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        # img_data = raw_posterior_laps_marginals[epoch_id]['p_x_given_n'].astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        if debug_print:
            print(f'np.shape(raw_posterior_laps_marginals[{epoch_IDX}]["p_x_given_n"]): {np.shape(img_data)}')

        base_image_height: int = int(round(base_image_height))
        desired_half_height: int = int(round(base_image_height/2))
        # get_array_as_img_kwargs = dict(desired_width=None, skip_img_normalization=False, include_value_labels=include_value_labels)
        get_array_as_img_kwargs = dict(desired_width=None, export_kind=export_kind, skip_img_normalization=True, include_value_labels=include_value_labels) # do not manually normalize the image before output. 
        save_array_as_image_kwargs = dict(export_kind=export_kind, include_value_labels=include_value_labels, allow_override_aspect_ratio=allow_override_aspect_ratio)

        if np.ndim(img_data) > 2:
            n_x_bins, n_decoders, n_curr_epoch_time_bins = np.shape(img_data)
            # img_data_array = [np.atleast_2d(np.squeeze(img_data[:,:, i])).T for i in np.arange(n_curr_epoch_time_bins)]
            raw_posterior_array = [np.atleast_2d(np.squeeze(img_data[:,:, i])) for i in np.arange(n_curr_epoch_time_bins)]
                    
            imgs_array = [get_array_as_image(raw_posterior_array[i].T, desired_height=base_image_height, **get_array_as_img_kwargs) for i in np.arange(n_curr_epoch_time_bins)]            
            if export_all_raw_marginals_separately:
                _sub_img_parent_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_raw_marginal').resolve()
                _sub_img_parent_path.mkdir(parents=False, exist_ok=True)
                for i, (a_single_time_bin_raw_posterior, an_img) in enumerate(zip(raw_posterior_array, imgs_array)):
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{complete_epoch_identifier_str}_raw_marginal_{i}.png').resolve()
                    an_img.save(_sub_img_path) # Save image to file
                    _out_save_tuples.append((an_img, _sub_img_path))
                    
                    if debug_print:
                        print(f'i: {i}, np.shape(a_single_time_bin_raw_posterior): {np.shape(a_single_time_bin_raw_posterior)}') # (n_x_bins, n_decoders)
                    _decoder_prob_arr = np.sum(a_single_time_bin_raw_posterior, axis=0) # get the four-tuple of decoder probabilities ... should this be axis=0? ... no because it's transposed... how confusing
                    # _decoder_prob_arr = _decoder_prob_arr / np.nansum(_decoder_prob_arr) # normalize
                    if debug_print:
                        print(f'\tnp.shape(_decoder_prob_arr): {np.shape(_decoder_prob_arr)}, _decoder_prob_arr: {_decoder_prob_arr}')
                    _decoder_prob_img: Image.Image = get_array_as_image(np.atleast_2d(_decoder_prob_arr).T, desired_height=base_image_height, **get_array_as_img_kwargs)
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{complete_epoch_identifier_str}_marginal_decoder_{i}.png').resolve()
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
                    _temp_img: Image.Image = get_array_as_image(_track_marginal_2tuple.T, desired_height=desired_half_height, **get_array_as_img_kwargs) # #TODO 2025-05-14 08:28: - [ ] Potential normalization error here, as when using raw_RGBA mode with a 2-decoder the passed posteriors must be normalized.
                    _sub_img_path = _sub_img_parent_path.joinpath(f'{complete_epoch_identifier_str}_track_marginal_two_tuple_{i}.png').resolve()
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
        _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_marginal_dir.png').resolve()
        img_data = epochs_directional_marginals[epoch_IDX]['p_x_given_n'].astype(float)
        marginal_dir_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)
        # image_marginal_dir, path_marginal_dir = marginal_dir_tuple

        # [2 x n_epoch_t_bins] matrix
        _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_marginal_track_identity.png').resolve()
        img_data = epochs_track_identity_marginals[epoch_IDX]['p_x_given_n'].astype(float)
        marginal_track_identity_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)
        # image_marginal_track_identity, path_marginal_track_identity = marginal_track_identity_tuple

        # 2-vector
        _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_marginal_track_identity_point.png').resolve()
        img_data = np.atleast_2d(collapsed_per_epoch_marginal_track_identity_point[epoch_IDX,:]).T
        marginal_dir_point_tuple = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)

        # 2-vector
        _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}_marginal_dir_point.png').resolve()
        img_data = np.atleast_2d(collapsed_per_epoch_epoch_marginal_dir_point[epoch_IDX,:]).T
        marginal_track_identity_point_tuple: Tuple[Image.Image, Path] = save_array_as_image(img_data, desired_height=desired_half_height, desired_width=None, skip_img_normalization=True, out_path=_img_path, **save_array_as_image_kwargs)

        
        return marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple, _out_save_tuples
        

    @function_attributes(short_name=None, tags=['export','marginal', 'pseudo2D', 'IMPORTANT', 'single-export-format'], input_requires=[], output_provides=[], uses=['.save_posterior'], used_by=['_test_export_marginals_for_figure'], creation_date='2024-09-10 00:06', related_items=[])
    @classmethod
    def save_marginals_arrays_as_image(cls, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'ripple',
                                        epoch_IDXs=None, complete_epoch_identifier_strs: Optional[List[str]]=None, 
                                        export_all_raw_marginals_separately: bool=True, export_kind: Optional[HeatmapExportKind] = None, include_value_labels:bool=False, allow_override_aspect_ratio:bool=True, debug_print=False):
        """ Exports all the raw_merged pseudo2D posteriors and their marginals out to image files.
        
        Usage: 
        
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import save_marginals_arrays_as_image
            directional_merged_decoders_result.perform_compute_marginals()
            parent_array_as_image_output_folder = Path('output/Exports/array_as_image').resolve()
            parent_array_as_image_output_folder.mkdir(parents=True, exist_ok=True)
            assert parent_array_as_image_output_folder.exists()
            save_marginals_arrays_as_image(directional_merged_decoders_result=directional_merged_decoders_result, parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='ripple', epoch_ids=[31])

        """
        assert epoch_id_identifier_str in ['ripple', 'laps'], f"epoch_id_identifier_str: '{epoch_id_identifier_str}' should be either 'ripple' or 'laps'"
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

        if epoch_IDXs is None:
            epoch_IDXs = np.arange(active_filter_epochs_decoder_result.num_filter_epochs)

        if complete_epoch_identifier_strs is None:
            complete_epoch_identifier_strs = [f'{epoch_id_identifier_str}_{epoch_IDX:03d}' for epoch_IDX in epoch_IDXs]

        assert len(complete_epoch_identifier_strs) == len(epoch_IDXs), f"complete_epoch_identifier_strs: {complete_epoch_identifier_strs} should be the same length as epoch_IDXs: {epoch_IDXs}"

        out_tuple_dict = {}
        for i, epoch_IDX in enumerate(epoch_IDXs):
            a_complete_epoch_identifier_str: str = complete_epoch_identifier_strs[i]
            
            # Make epoch folder
            _curr_path = parent_array_as_image_output_folder.joinpath(a_complete_epoch_identifier_str).resolve()
            _curr_path.mkdir(exist_ok=True)
                                                     
            out_tuple_dict[epoch_IDX] = cls.save_posterior(raw_posterior_active_marginals, active_directional_marginals,
                                                                                active_track_identity_marginals, collapsed_per_epoch_marginal_dir_point, collapsed_per_epoch_marginal_track_identity_point,
                                                                                        parent_array_as_image_output_folder=_curr_path,
                                                                                        epoch_id_identifier_str=epoch_id_identifier_str, epoch_IDX=epoch_IDX, complete_epoch_identifier_str=a_complete_epoch_identifier_str,
                                                                                        export_all_raw_marginals_separately=export_all_raw_marginals_separately, export_kind=export_kind, include_value_labels=include_value_labels, allow_override_aspect_ratio=allow_override_aspect_ratio,
                                                                                        debug_print=debug_print)
        return out_tuple_dict
        


    # ==================================================================================================================== #
    # Save/Load                                                                                                            #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['IMPORTANT', 'save', 'export', 'ESSENTIAL', 'posterior', 'export'], input_requires=[], output_provides=[], uses=['SingleEpochDecodedResult.save_posterior_as_image', 'ImageOperationsAndEffects'], used_by=['cls.perform_export_all_decoded_posteriors_as_images'], creation_date='2024-08-05 10:47', related_items=[])
    def export_decoded_posteriors_as_images(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult, # decoder_ripple_filter_epochs_decoder_result_dict: DecoderResultDict,
                                             posterior_out_folder:Path='output/_temp_individual_posteriors',
                                             custom_export_formats: Optional[Dict[str, HeatmapExportConfig]]=None, colormap='Oranges', #'viridis',
                                             desired_height=None, **kwargs): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the decoded posteiors (decoded epochs) into one image file for each format specified in `custom_export_formats`
        
        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.computer_vision import ComputerVisionComputations
            should_export_separate_color_and_greyscale: bool = True
            # a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_laps_filter_epochs_decoder_result_dict['long_LR']
            # epochs_name='laps'
            a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_ripple_filter_epochs_decoder_result_dict['long_LR']
            epochs_name='ripple'

            parent_output_folder = Path(r'output/_temp_individual_posteriors').resolve()
            posterior_out_folder = parent_output_folder.joinpath(DAY_DATE_TO_USE, epochs_name).resolve()

            (posterior_out_folder, posterior_out_folder_greyscale, posterior_out_folder_color), _save_out_paths = PosteriorExporting.export_decoded_posteriors_as_images(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, posterior_out_folder=posterior_out_folder, should_export_separate_color_and_greyscale=should_export_separate_color_and_greyscale)

            if should_export_separate_color_and_greyscale:
                fullwidth_path_widget(posterior_out_folder_greyscale)
                fullwidth_path_widget(posterior_out_folder_color)
            else:
                fullwidth_path_widget(posterior_out_folder)
                
        History:
            Refactored from `ComputerVisionComputations` on 2024-09-30
        """

        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
        from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind
        from pyphocorehelpers.plotting.media_output_helpers import ImagePostRenderFunctionSets
        from neuropy.core.epoch import ensure_dataframe
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
        
        if not isinstance(posterior_out_folder, Path):
            posterior_out_folder = Path(posterior_out_folder).resolve()

        posterior_out_folder.mkdir(parents=True, exist_ok=True)

        if custom_export_formats is None:
            custom_export_formats: Dict[str, HeatmapExportConfig] = {
                'greyscale': HeatmapExportConfig.init_greyscale(desired_height=desired_height, **kwargs),
                'color': HeatmapExportConfig(colormap=colormap, export_kind=HeatmapExportKind.COLORMAPPED, desired_height=desired_height, **kwargs),
                'raw_rgba': HeatmapExportConfig.init_for_export_kind(export_kind=HeatmapExportKind.RAW_RGBA, lower_bound_alpha=0.1, drop_below_threshold=1e-2, desired_height=desired_height, **kwargs),
            }
        else:
            custom_export_formats = deepcopy(custom_export_formats) # make sure they're all independent

        assert custom_export_formats is not None
        
        export_formats_post_render_image_functions_builder_fn_dict: Dict[str, List[Dict[str, Callable]]] = {}
        
        for export_format_name, export_format_config in custom_export_formats.items():
            if export_format_config.export_folder is None:
                export_format_config.export_folder = posterior_out_folder.joinpath(export_format_name).resolve()
            ## create the folder if needed
            export_format_config.export_folder.mkdir(parents=True, exist_ok=True)
                    
            post_render_image_functions_builder_fn = getattr(export_format_config, 'post_render_image_functions_builder_fn', ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict)
            assert post_render_image_functions_builder_fn is not None
            post_render_image_functions_dict_list: List[Dict[str, Callable]] = post_render_image_functions_builder_fn(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result)
            export_formats_post_render_image_functions_builder_fn_dict[export_format_name] = post_render_image_functions_dict_list

        # END for export_format_n...
            

        num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
        # active_filter_epochs: pd.DataFrame = ensure_dataframe(a_decoder_decoded_epochs_result.active_filter_epochs)
        
        # assert Assert.require_columns(active_filter_epochs, required_columns=['maze_id'])
        # is_epoch_pre_post_delta = active_filter_epochs['maze_id'].to_numpy()
        

        # Build post-image-generation callback functions _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        
        # fixed_label_region_height: Optional[int] = 520

        # # font_size = 144
        # # font_size = 96
        # font_size = 72
        
        epoch_id_identifier_str: str = 'p_x_given_n'

        
        _save_out_paths = []
        _save_out_format_results: Dict[str, List] = {}
        for i in np.arange(num_filter_epochs):
            active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=i)

            # Prepare a multi-line, sideways label _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #                                      
            if desired_height is None:
                ## set to 1x
                desired_height = active_captured_single_epoch_result.n_xbins # 1 pixel for each xbin
                
            for export_format_name, export_format_config in custom_export_formats.items():
                if export_format_config.export_folder is None:
                    export_format_config.export_folder = posterior_out_folder.joinpath(export_format_name).resolve()
                if export_format_name not in _save_out_format_results:
                    ## create the empty result:
                    _save_out_format_results[export_format_name] = []
                    
                ## get the post-render functions
                curr_post_render_image_functions_dict = export_formats_post_render_image_functions_builder_fn_dict[export_format_name][i]
                
                ## mode to use
                # active_epoch_data_IDX: int = self.epoch_data_index
                curr_epoch_info_dict = active_captured_single_epoch_result.epoch_info_tuple._asdict()
                active_epoch_id: int = curr_epoch_info_dict.get('label', None)
                if active_epoch_id is not None:
                    active_epoch_id = int(active_epoch_id)
                    complete_epoch_identifier_str = f"{epoch_id_identifier_str}[{active_epoch_id:03d}]" # 2025-06-03 - 'p_x_given_n[067]'
                else:
                    complete_epoch_identifier_str = f"{epoch_id_identifier_str}"

                assert complete_epoch_identifier_str is not None
                _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=export_format_config.export_folder, complete_epoch_identifier_str=complete_epoch_identifier_str, **(kwargs|export_format_config.to_dict()), post_render_image_functions=curr_post_render_image_functions_dict)
            
                _output_export_format_config = deepcopy(export_format_config)
                _output_export_format_config.posterior_saved_path = posterior_save_path
                _output_export_format_config.posterior_saved_image = _posterior_image
                _save_out_paths.append(posterior_save_path)
                # _save_out_format_results[export_format_name].append(export_format_config) # save out the modified v
                _save_out_format_results[export_format_name].append(_output_export_format_config) # save out the modified v
            # END for export_format_n...
                   
        # END for i in np.arange(num_filter_epochs)
        
        return (posterior_out_folder, _save_out_format_results, ), _save_out_paths



    @function_attributes(short_name=None, tags=['private', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=['._subfn_perform_export_single_epochs'], creation_date='2025-05-14 11:00', related_items=[])
    @classmethod
    def _subfn_build_combined_output_images(cls, single_known_epoch_type_dict: Dict[DecoderName, Dict[str, List[HeatmapExportConfig]]], specific_epochs_posterior_out_folder: Path, known_epoch_type_name: str = 'laps', custom_export_format_series_name: str = 'color', joined_export_folder_name: str = 'combined', combined_img_padding=4, combined_img_separator_color=None,
                                            stack_orientations: List[ImageStackOrientation] = [ImageStackOrientation.GRID]):
        """ exports combined images stiched across the seperate decoder's images
        
        """
        from pyphocorehelpers.plotting.media_output_helpers import ImageStackOrientation  # used in `_subfn_build_combined_output_images`
                
        ## INPUTS: out_custom_formats_dict, known_epoch_type_name, custom_export_format_series_name

        # _output_combined_dir = _out['parent_specific_session_output_folder'].joinpath(known_epoch_type_name, joined_export_folder_name, custom_export_format_series_name).resolve()
        _output_combined_dir = specific_epochs_posterior_out_folder.joinpath(joined_export_folder_name, custom_export_format_series_name).resolve()
        _output_combined_dir.mkdir(parents=True, exist_ok=True)

        # Stich acrossed decoders
        out_all_decoders_epochs_list = []
        # _single_epoch_single_series_single_export_type_row = []
        for decoder_name, a_single_export_format_export_result_dict in single_known_epoch_type_dict.items():
            # one for each epoch
            an_epochs_export_result_list: List[HeatmapExportConfig] = a_single_export_format_export_result_dict[custom_export_format_series_name]
            out_all_decoders_epochs_list.append([v.posterior_saved_image for v in an_epochs_export_result_list])
            
        num_exported_epochs: int = len(out_all_decoders_epochs_list[0])
        print(f'num_exported_epochs: {num_exported_epochs}')
        # INPUT: out_all_decoders_epochs_list

        assert len(out_all_decoders_epochs_list) == 4, f"expected the out_all_decoders_epochs_list to be of length 4, with one entry per each decoder_name, but len(out_all_decoders_epochs_list): {len(out_all_decoders_epochs_list)}. out_all_decoders_epochs_list: {out_all_decoders_epochs_list}"
        _single_epoch_single_series_single_export_type_rows_list = []
        _output_combined_image_save_dirs = []
        for i in np.arange(num_exported_epochs):
            ## for a single epoch:
            
            # _single_epoch_combined_img = horizontal_image_stack(_single_epoch_row, padding=combined_img_padding, separator_color=combined_img_separator_color)
            # _single_epoch_combined_img = vertical_image_stack(_single_epoch_row, padding=combined_img_padding, separator_color=combined_img_separator_color)
            
            for an_orientation in stack_orientations:
                if an_orientation.is_grid:
                    _single_epoch_row = [[out_all_decoders_epochs_list[0][i], out_all_decoders_epochs_list[1][i]], [out_all_decoders_epochs_list[2][i], out_all_decoders_epochs_list[3][i]]] # 2 x 2
                else:
                    _single_epoch_row = [out_all_decoders_epochs_list[0][i], out_all_decoders_epochs_list[1][i], out_all_decoders_epochs_list[2][i], out_all_decoders_epochs_list[3][i]] # 4 x 1
                    
                _single_epoch_combined_img = an_orientation.stack_images(_single_epoch_row, padding=combined_img_padding, separator_color=combined_img_separator_color)    
                _single_epoch_single_series_single_export_type_rows_list.append(_single_epoch_combined_img)
                ## Save the image:
                _img_path = _output_combined_dir.joinpath(f'merged{an_orientation.shortname}_{known_epoch_type_name}[{i}].png').resolve() # 'mergedV_laps[0].png'
                _single_epoch_combined_img.save(_img_path)
                _output_combined_image_save_dirs.append(_img_path)

        ## OUTPUTS: _output_combined_dir, out_all_decoders_epochs_list, _single_epoch_single_series_single_export_type_rows_list, _output_combined_image_save_dirs
        return _output_combined_dir, _output_combined_image_save_dirs


    @function_attributes(short_name=None, tags=['private' 'helper'], input_requires=[], output_provides=[], uses=['.export_decoded_posteriors_as_images', '._subfn_build_combined_output_images'], used_by=['.perform_export_all_decoded_posteriors_as_images'], creation_date='2025-05-14 11:03', related_items=[])
    @classmethod
    def _subfn_perform_export_single_epochs_result_set(cls, _active_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], epochs_name: str, a_parent_output_folder: Path, custom_export_formats: Optional[Dict[str, HeatmapExportConfig]]=None, desired_height=None, combined_img_padding=4, combined_img_separator_color=None, **kwargs) -> IdentifyingContext:
        """ saves a single set of named epochs, like 'laps' or 'ripple' 
        """
        n_decoders: int = len(_active_filter_epochs_decoder_result_dict)
        
        out_paths = {}
        out_custom_export_formats_results_dict = {}

        specific_epochs_posterior_out_folder = a_parent_output_folder.joinpath(epochs_name).resolve() # 'K:/scratch/collected_outputs/figures/_temp_individual_posteriors/2024-09-30/gor01_one_2006-6-09_1-22-43/ripple/'
        specific_epochs_posterior_out_folder.mkdir(parents=True, exist_ok=True)

        for a_decoder_name, a_decoder_decoded_epochs_result in _active_filter_epochs_decoder_result_dict.items():
            # _save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5', decoder_name=a_decoder_name, epochs_name=epochs_name)
            # _specific_save_context = deepcopy(a_save_context).overwriting_context(decoder_name=a_decoder_name, epochs_name=epochs_name)
            posterior_out_folder = specific_epochs_posterior_out_folder.joinpath(a_decoder_name).resolve() # 'K:/scratch/collected_outputs/figures/_temp_individual_posteriors/2024-09-30/gor01_one_2006-6-09_1-22-43/ripple/long_RL'
            posterior_out_folder.mkdir(parents=True, exist_ok=True)
            # print(f'a_decoder_name: {a_decoder_name}, _specific_save_context: {_specific_save_context}, posterior_out_folder: {posterior_out_folder}')
            (an_out_posterior_out_folder, a_custom_export_format_results), an_out_flat_save_out_paths = cls.export_decoded_posteriors_as_images(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, posterior_out_folder=posterior_out_folder,
                                                                                                                                                    desired_height=desired_height, custom_export_formats=custom_export_formats, **kwargs) #TODO 2025-05-14 08:55: - [ ] BUG?!? Is it possible to plot the overlaid color image when iterating through the decoders 1-by-1? Don't I need all 4 at once?
            
            out_paths[a_decoder_name] = an_out_posterior_out_folder
            out_custom_export_formats_results_dict[a_decoder_name] = a_custom_export_format_results


            #TODO 2025-06-03 05:47: - [ ] Marginals
            
            
        ## try to export the combined figures right away
        if n_decoders > 1:
            # custom_export_format_series_name: str = list(out_custom_export_formats_results_dict.keys())[0]
            # custom_export_format_series_name: str = list(out_custom_export_formats_results_dict[list(out_custom_export_formats_results_dict.keys())[0]].keys())[0] # the inner key is the decoder_name (like 'long_LR') but the outer key is the custom_export name like 'color'
            try:
                for custom_export_format_series_name in list(out_custom_export_formats_results_dict[list(out_custom_export_formats_results_dict.keys())[0]].keys()):
                    _output_combined_dir, _output_combined_image_save_dirs = cls._subfn_build_combined_output_images(single_known_epoch_type_dict=out_custom_export_formats_results_dict, specific_epochs_posterior_out_folder=specific_epochs_posterior_out_folder,
                                                                                                                known_epoch_type_name=epochs_name, custom_export_format_series_name=custom_export_format_series_name,
                                                                                                                combined_img_padding=combined_img_padding, combined_img_separator_color=combined_img_separator_color)
                    
            except (AssertionError, ValueError) as e:
                print(f'WARN: failed to merge images to combined images at the end with error: {e}')
            except Exception as e:
                print(f'WARN: failed to merge images to combined images at the end with error: {e}')
                raise

            
        return out_paths, out_custom_export_formats_results_dict



        
    @classmethod
    @function_attributes(short_name=None, tags=['MAIN', 'export', 'images', 'ESSENTIAL'], input_requires=[], output_provides=[], uses=['._subfn_perform_export_single_epochs'], used_by=['_display_directional_merged_pf_decoded_stacked_epoch_slices'], creation_date='2024-08-28 08:36', related_items=[])
    def perform_export_all_decoded_posteriors_as_images(cls, decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult],
                                                         _save_context: IdentifyingContext, parent_output_folder: Path, custom_export_formats: Optional[Dict[str, HeatmapExportConfig]]=None, desired_height=None, combined_img_padding=4, combined_img_separator_color=None):
        """ Exports the decoded epoch position posteriors as raw images, also includes functionality to export merged/combined images.
        
        Usage:
        
            
        History:
            Refactored from `ComputerVisionComputations` on 2024-09-30
        """

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        assert parent_output_folder.exists(), f"parent_output_folder: {parent_output_folder} does not exist"
        
        _common_kwargs = dict(desired_height=desired_height, combined_img_padding=combined_img_padding, combined_img_separator_color=combined_img_separator_color)

        out_paths_dict = {'laps': None, 'ripple': None}
        out_custom_formats_results_dict = {'laps': None, 'ripple': None}
        if decoder_laps_filter_epochs_decoder_result_dict is not None:
            out_paths_dict['laps'], out_custom_formats_results_dict['laps'] = cls._subfn_perform_export_single_epochs_result_set(_active_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epochs_name='laps', a_parent_output_folder=parent_output_folder, custom_export_formats=custom_export_formats, **_common_kwargs)
        if decoder_ripple_filter_epochs_decoder_result_dict is not None:
            out_paths_dict['ripple'], out_custom_formats_results_dict['ripple'] = cls._subfn_perform_export_single_epochs_result_set(_active_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict,epochs_name='ripple', a_parent_output_folder=parent_output_folder, custom_export_formats=custom_export_formats, **_common_kwargs)
        return out_paths_dict, out_custom_formats_results_dict




    # ==================================================================================================================== #
    # Save/Load                                                                                                            #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['posterior', 'HDF5', 'output', 'save', 'export'], input_requires=[], output_provides=[], uses=['h5py', 'SingleEpochDecodedResult', 'SingleEpochDecodedResult.to_hdf(...)'], used_by=['perform_save_all_decoded_posteriors_to_HDF5'], creation_date='2024-08-28 02:38', related_items=['load_decoded_posteriors_from_HDF5'])
    def save_decoded_posteriors_to_HDF5(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult, save_path:Path='decoded_epoch_posteriors.h5', allow_append:bool=False, out_context=None, debug_print=False): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the DecodedFilterEpochsResult info to a file
        
        _save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_transition_matricies')
        _save_path = PosteriorExporting.save_decoded_posteriors_to_HDF5(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, out_context=_save_context, save_path='output/transition_matrix_data.h5')
        _save_path

        History:
            Refactored from `ComputerVisionComputations` on 2024-09-30
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        import h5py

        if not isinstance(save_path, Path):
            save_path = Path(save_path).resolve()
            
        if out_context is None:
            out_context = IdentifyingContext()

        if not allow_append:
            file_mode = 'w' 
        else:
            file_mode = 'r+' # 'x' #
            
        # Save to .h5 file
        with h5py.File(save_path, file_mode) as f: #  
            # r Readonly, file must exist (default)
            # r+ Read/write, file must exist 
            # w Create file, truncate if exists 
            # w- or x Create file, fail if exists a Read/write if exists, create otherwise
            if out_context is not None:
                # add context to the file
                if not isinstance(out_context, dict):
                    flat_context_desc: str = out_context.get_description(separator='|') # 'kdiba|gor01|one|2006-6-08_14-26-15|save_transition_matricies'
                    _out_context_dict = out_context.to_dict() | {'session_context_desc': flat_context_desc}
                else:
                    # it is a raw dict
                    _out_context_dict = deepcopy(out_context)
                    
                for k, v in _out_context_dict.items():
                    ## add the context as file-level metadata
                    f.attrs[k] = v
                    
            ## BEGIN MAIN OUTPUT
            num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
            num_required_zero_padding: int = len(str(num_filter_epochs))
            
            for i in np.arange(num_filter_epochs):
                active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=i) ## extracts a `SingleEpochDecodedResult`
                epoch_data_idx_str: str = f"{i:0{num_required_zero_padding}d}"
                # _curr_context = out_context.overwriting_context(epoch_idx=i)
                _curr_context = out_context.overwriting_context(epoch_idx=epoch_data_idx_str)
                _curr_key: str = _curr_context.get_description(separator='/')
                
                # #TODO 2024-11-26 09:03: - [ ] Attempt to extract and combine the relevant keys to make a good HDF_key, but gave up
                # _curr_context_dict = _curr_context.to_dict()
                # _curr_context_dict.subset(_curr_context._get_session_context_keys())
                # _all_keys = _curr_context_dict.keys()
                # _non_session_keys = _all_keys - _curr_context._get_session_context_keys()
                # _last_session_key_index = list(_all_keys).index(_curr_context._get_session_context_keys()[-1])
                # _resume_keypath_index = list(_all_keys).index('display_fn_name')
                # _keys_to_merge = list(_all_keys)[(_last_session_key_index+1):_resume_keypath_index] # one after the last_session_key_index - ['epochs_source', 'included_qclu_values', 'minimum_inclusion_fr_Hz']
                # _values_to_merge = [_curr_context_dict[k] for k in _keys_to_merge]
                # _merged_dict = dict(zip(_keys_to_merge, _values_to_merge))
                # epochs_source: str = _merged_dict.pop('epochs_source', None)
                # custom_suffix_str: str = '-'.join([epochs_source, *['_'.join([k, str(v),]) for k, v in _merged_dict.items()]])
                # custom_suffix_str ## wrong: 'normal_computed-included_qclu_values_[1, 2, 4, 6, 7, 9]-minimum_inclusion_fr_Hz_5.0'                
                # _curr_key: str = _curr_context.get_description(separator='/') ## TODO: not finished

                if not _curr_key.startswith('/'):
                    _curr_key = "/" + _curr_key
                # active_captured_single_epoch_result.to_hdf(save_path, key=_curr_key, debug_print=True, enable_hdf_testing_mode=True)
                active_captured_single_epoch_result.to_hdf(f, key=_curr_key, debug_print=debug_print, enable_hdf_testing_mode=False, required_zero_padding=num_required_zero_padding)
        
        return save_path
    

    @classmethod
    @function_attributes(short_name=None, tags=['MAIN', 'save', 'export', 'HDF5', 'h5'], input_requires=[], output_provides=[], uses=['save_decoded_posteriors_to_HDF5'], used_by=[], creation_date='2024-08-28 08:36', related_items=['LoadedPosteriorContainer.load_batch_hdf5_exports'])
    def perform_save_all_decoded_posteriors_to_HDF5(cls, decoder_laps_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], decoder_ripple_filter_epochs_decoder_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult], _save_context: IdentifyingContext, save_path: Path, should_overwrite_extant_file:bool=True):
        """
        
        Usage:
        
            save_path = Path('output/newest_all_decoded_epoch_posteriors.h5').resolve()
            _parent_save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5')
            out_contexts = PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, _save_context=_parent_save_context, save_path=save_path)
            out_contexts

        History:
            Refactored from `ComputerVisionComputations` on 2024-09-30
        """
        _flat_all_out_paths = []
        
        def _subfn_perform_save_single_epochs(_active_filter_epochs_decoder_result_dict, a_save_context: IdentifyingContext, epochs_name: str, save_path: Path) -> Dict[types.DecoderName, IdentifyingContext]:
            """ saves a single set of named epochs, like 'laps' or 'ripple' 
            Captures/Updates: _flat_all_out_paths,
        
            """
            _sub_out_contexts = {}
            for a_decoder_name, a_decoder_decoded_epochs_result in _active_filter_epochs_decoder_result_dict.items():
                # _save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5', decoder_name=a_decoder_name, epochs_name=epochs_name)
                _specific_save_context = deepcopy(a_save_context).overwriting_context(decoder_name=a_decoder_name, epochs_name=epochs_name)
                print(f'a_decoder_name: {a_decoder_name}, _specific_save_context: {_specific_save_context}')
                if not save_path.exists():
                    print(f'\t file does not exist, so setting allow_append = False')
                    allow_append = False
                else:
                    print(f'\tsave_path exists, so allow_append = True')
                    allow_append = True
                an_out_path = cls.save_decoded_posteriors_to_HDF5(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, out_context=_specific_save_context, save_path=save_path, allow_append=allow_append)
                _sub_out_contexts[a_decoder_name] = _specific_save_context
                _flat_all_out_paths.append(an_out_path)
                
            return _sub_out_contexts


        if save_path.exists() and should_overwrite_extant_file:
            print(f'\tsave_path "{save_path}" exists and should_overwrite_extant_file==True, so removing file...')
            save_path.unlink(missing_ok=False)
            print(f'\t successfully removed.')
            
        out_contexts = {'laps': None, 'ripple': None}
        if decoder_laps_filter_epochs_decoder_result_dict is not None:
            out_contexts['laps'] = _subfn_perform_save_single_epochs(decoder_laps_filter_epochs_decoder_result_dict, a_save_context=_save_context, epochs_name='laps', save_path=save_path)
        if decoder_ripple_filter_epochs_decoder_result_dict is not None:
            out_contexts['ripple'] = _subfn_perform_save_single_epochs(decoder_ripple_filter_epochs_decoder_result_dict, a_save_context=_save_context, epochs_name='ripple', save_path=save_path)
        return out_contexts, _flat_all_out_paths


    @classmethod
    @function_attributes(short_name=None, tags=['posterior', 'HDF5', 'load'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=['save_decoded_posteriors_to_HDF5'])
    def load_decoded_posteriors_from_HDF5(cls, load_path: Path, debug_print=True) -> Dict[types.DecoderName, Dict[KnownEpochsName, Dict]]:
        """
        Load the transition matrix info from a file
        
        Usage:
            load_path = Path('output/2024-11-26_Lab_newest_all_decoded_epoch_posteriors.h5')

            ## used for reconstituting dataset:
            dataset_type_fields = ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']
            decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']

            _out_dict, (session_key_parts, custom_replay_parts) = PosteriorExporting.load_decoded_posteriors_from_HDF5(load_path=load_path, debug_print=True)
            _out_ripple_only_dict = {k:v['ripple'] for k, v in _out_dict.items()} ## cut down to only the laps

            ## build the final ripple data outputs:
            ripple_data_field_dict = {}
            # active_var_key: str = 'p_x_given_n' # dataset_type_fields	

            for active_var_key in dataset_type_fields:
                ripple_data_field_dict[active_var_key] = {
                    a_decoder_name: [v for v in _out_ripple_only_dict[a_decoder_name][active_var_key]] for a_decoder_name in decoder_names
                }


            ripple_img_dict = ripple_data_field_dict['p_x_given_n_grey']
            ripple_img_dict['long_LR'][0]
        
        
        History:
            Refactored from `ComputerVisionComputations` on 2024-09-30
        """
        if not isinstance(load_path, Path):
            load_path = Path(load_path).resolve()

        assert load_path.exists(), f"load_path: '{load_path}' does not exist!"

        dataset_type_fields = ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']
        attribute_type_fields = ['nbins', 'epoch_data_index', 'n_xbins', 'creation_date']
        

        import h5py
        from pyphocorehelpers.Filesystem.HDF5.hdf5_file_helpers import HDF5_Helper

        # Usage
        found_groups = HDF5_Helper.find_groups_by_name(load_path, 'save_decoded_posteriors_to_HDF5')
        if debug_print:
            print(found_groups) # ['kdiba/gor01/one/2006-6-08_14-26-15/save_decoded_posteriors_to_HDF5']
        assert len(found_groups) == 1, f"{found_groups}"
        _save_key: str = found_groups[0]
        if debug_print:
            print(f'_save_key: {_save_key}')

        # leaf_datasets = get_leaf_datasets(load_path)
        # print(leaf_datasets)

        curr_export_result_save_properties = _save_key.split('/') # split into its path parts, like "/kdiba/gor01/one/2006-6-09_1-22-43/normal_computed/[1, 2, 4, 6, 7, 9]/5.0/save_decoded_posteriors_to_HDF5"
        if debug_print:
            print(f'curr_export_result_save_properties: {curr_export_result_save_properties}')
        assert curr_export_result_save_properties[-1] == 'save_decoded_posteriors_to_HDF5', f"last component should equal 'save_decoded_posteriors_to_HDF5' but instead it equals: {curr_export_result_save_properties[-1]}"
        session_key_parts = curr_export_result_save_properties[:4]
        session_key_str: str = '-'.join(session_key_parts)
        print(f'session_key_str: "{session_key_str}"')
        
        custom_replay_parts = curr_export_result_save_properties[4:-1]

        print(f'session_key_parts: {session_key_parts}')
        print(f'custom_replay_parts: {custom_replay_parts}')
        if len(custom_replay_parts) == 3:
            custom_replay_name, fr_Hz, qclus = custom_replay_parts
        elif len(custom_replay_parts) == 1:
            custom_suffix_str = custom_replay_parts[0]
            
        else:
            # raise NotImplementedError(f'could not parse curr_export_result_save_properties: {curr_export_result_save_properties}, ')
            print(f'ERROR: could not parse curr_export_result_save_properties: {curr_export_result_save_properties} for file: "{load_path.as_posix()}". Skipping those properties but loading anyway.')
            pass


        out_dict: Dict = {}

        with h5py.File(load_path, 'r') as f:            
            main_save_group = f[_save_key]
            # if debug_print:
            #     print(f'main_save_group: {main_save_group}')
            
            for decoder_prefix in main_save_group.keys():
                # if debug_print:
                #     print(f'decoder_prefix: {decoder_prefix}')
                if decoder_prefix not in out_dict:
                    out_dict[decoder_prefix] = {}

                decoder_group = main_save_group[decoder_prefix]
                for known_epochs_name in decoder_group.keys():
                    # if debug_print:
                    #     print(f'\tknown_epochs_name: {known_epochs_name}')
                    if known_epochs_name not in out_dict[decoder_prefix]:
                        out_dict[decoder_prefix][known_epochs_name] = {}
                        
                    decoder_epochtype_group = decoder_group[known_epochs_name]
                    
                    ## allocate outputs:
                    out_dict[decoder_prefix][known_epochs_name] = {k:list() for k in dataset_type_fields} # allocate a dict of empty lists for each item in `dataset_type_fields`
                    
                    for dataset_name in decoder_epochtype_group.keys():
                        # if debug_print:
                        #     print(f'\t\tdataset_name: {dataset_name}')
                            
                        ## the lowest-level group before the data itself
                        dataset_final_group = decoder_epochtype_group[dataset_name]
                        # dataset_type_fields
                        
                        for leaf_data_key, leaf_data in dataset_final_group.items():
                            # if debug_print:
                            #     print(f'\t\t\tleaf_data_key: {leaf_data_key}')
                            # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[{dataset_name}]"][()]
                            # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[{dataset_name}]"][()]
                            array = leaf_data[()] # should get the NDArray
                            # if debug_print:
                            #     print(f'\t\t\t\tarray: {type(array)}')
                            
                            out_dict[decoder_prefix][known_epochs_name][leaf_data_key].append(array) #
                            
                        # array = decoder_epochtype_group[dataset_name] #[()]
                        
                        
                        # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[()]"]
                        # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[{dataset_name}]"][()]
                        # if debug_print:
                        #     print(f'\t\t\tarray: {type(array)}')
                        # markov_order = group[dataset_name].attrs['index']
                        # arrays_list.append((markov_order, array))
                        # arrays_list.append(array)
                
                    # arrays_list.sort(key=lambda x: x[0])  # Sort by markov_order
                    # out_dict[decoder_prefix] = [array for _, array in arrays_list]
                    # if decoder_prefix not in out_dict[epochs_name]:
                        # out_dict[epochs_name][decoder_prefix] = arrays_list
                    # out_dict[epochs_name][decoder_prefix] = arrays_list
                    
        # END open

        return out_dict, (session_key_parts, custom_replay_parts)


    # ==================================================================================================================== #
    # Other/Testing                                                                                                        #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['export', 'images', 'marginals', 'testing'], input_requires=[], output_provides=[], uses=['.save_marginals_arrays_as_image'], used_by=['_perform_export_current_epoch_marginal_and_raster_images'], creation_date='2025-05-13 19:45', related_items=[])
    @classmethod
    def _test_export_marginals_for_figure(cls, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], clicked_epoch: NDArray, context_specific_root_export_path: Path, epoch_specific_folder: Path,
                                           epoch_id_identifier_str='ripple', complete_epoch_identifier_str: Optional[str]=None, debug_print=True, export_kind: Optional[HeatmapExportKind] = None, allow_override_aspect_ratio:bool=True, **kwargs):
        """
            Takes a `clicked_epoch` start_t 
            
            
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
        elif epoch_id_identifier_str in ('laps','lap'):
            active_filter_epochs_decoder_result: DecodedFilterEpochsResult = deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
        else:
            raise NotImplementedError(f'epoch_id_identifier_str: {epoch_id_identifier_str}')
        
        epoch_data_IDXs = active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) # [262], [296]
        if debug_print:
            print(f'epoch_ids: {epoch_data_IDXs}')

        if len(epoch_data_IDXs) < 1:
            print(f'WARN: no found epoch_ids from the provided clicked_epoch: {clicked_epoch}!!')
        assert (len(epoch_data_IDXs) > 0), f"no found epoch_ids from the provided clicked_epoch: {clicked_epoch}"
        # Determine the index we'll use ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # an_active_filter_epochs_decoder_result = list(filtered_decoder_filter_epochs_decoder_result_dict.values())[0] #['long_LR']
        # epoch_data_IDXs = an_active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) # [262], [296]
        # {k:an_active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) for k, an_active_filter_epochs_decoder_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
        # active_epoch_id: int = int(epoch_data_IDXs[0])
        active_epoch_data_IDX: int = int(epoch_data_IDXs[0])
        active_epoch_id: int = int(active_filter_epochs_decoder_result.filter_epochs['label'][active_epoch_data_IDX])

        # active_epoch_id: int = int(epoch_data_IDXs[0]) # assume a single epoch idx
        if debug_print:
            print(f'epoch_idx: {active_epoch_id}')
        if complete_epoch_identifier_str is None:
            complete_epoch_identifier_str: str = f"{epoch_id_identifier_str}[{active_epoch_id:03d}]" # format `{active_epoch_dict['Index']}` as a fixed-width integer with leading zeros as needed to make sure the resulting files can be sorted by filename
        else:
            ## use the user provided ones
            pass

        assert complete_epoch_identifier_str is not None
        if debug_print:
            print(f'complete_epoch_identifier_str: {complete_epoch_identifier_str}')


        # if complete_epoch_identifier_str is not None:
            # complete_epoch_identifier_strs = [complete_epoch_identifier_str]
            
        complete_epoch_identifier_strs = [complete_epoch_identifier_str]
            
        ## Sanity check:
        if debug_print:
            curr_epoch_result: SingleEpochDecodedResult = active_filter_epochs_decoder_result.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
            print(f"curr_epoch_result.epoch_info_tuple: {curr_epoch_result.epoch_info_tuple}")
            print(f"\tnbins: {curr_epoch_result.nbins}")
            print(f'\tactive_filter_epochs_decoder_result.decoding_time_bin_size: {active_filter_epochs_decoder_result.decoding_time_bin_size}')
            # curr_epoch_result.time_bin_container

        # active_filter_epochs_decoder_result.all_directional_ripple_filter_epochs_decoder_result
        out_image_save_tuple_dict = cls.save_marginals_arrays_as_image(directional_merged_decoders_result=directional_merged_decoders_result, parent_array_as_image_output_folder=context_specific_root_export_path,
                                                          epoch_id_identifier_str=epoch_id_identifier_str, epoch_IDXs=epoch_data_IDXs, complete_epoch_identifier_strs=complete_epoch_identifier_strs,
                                                          debug_print=True, export_kind=export_kind, include_value_labels=False, allow_override_aspect_ratio=allow_override_aspect_ratio,# **kwargs,
                                                        )

        # ==================================================================================================================== #
        # Export Decoded Position Posteriors                                                                                   #
        # ==================================================================================================================== #
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
        ## INPUTS: filtered_decoder_filter_epochs_decoder_result_dict, parent_array_as_image_output_folder, epoch_id_identifier_str='ripple', 

        # assert parent_array_as_image_output_folder.exists()
        assert epoch_specific_folder.exists()



        # # Make epoch folder
        # ripple_specific_folder = parent_array_as_image_output_folder.joinpath(f'{epoch_id_identifier_str}_{epoch_id}').resolve()
        # ripple_specific_folder.mkdir(exist_ok=True)

        out_image_paths = {}
        for k, v in filtered_decoder_filter_epochs_decoder_result_dict.items():
            # v: DecodedFilterEpochsResult
            a_result: SingleEpochDecodedResult = v.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
            print(f"{k}: filtered_decoder_filter_epochs_decoder_result_dict[{k}].decoding_time_bin_size: {v.decoding_time_bin_size}") # 0.016!! 
            _img_path = epoch_specific_folder.joinpath(f'{complete_epoch_identifier_str}_posterior_{k}.png').resolve()
            a_result.save_posterior_as_image(_img_path, export_kind=HeatmapExportKind.COLORMAPPED, colormap='Oranges', allow_override_aspect_ratio=allow_override_aspect_ratio, flip_vertical_axis=True, **kwargs) # #TODO 2025-05-14 08:30: - [ ] forced orange
            out_image_paths[k] = _img_path

        return out_image_save_tuple_dict, out_image_paths



    @function_attributes(short_name=None, tags=['marginal', 'export'], input_requires=[], output_provides=[], uses=['cls._test_export_marginals_for_figure'], used_by=[], creation_date='2024-09-06 00:00', related_items=[])
    @classmethod
    def _perform_export_current_epoch_marginal_and_raster_images(cls, _out_ripple_rasters: "RankOrderRastersDebugger", directional_merged_decoders_result, filtered_decoder_filter_epochs_decoder_result_dict, active_session_context, root_export_path: Path, epoch_id_identifier_str='lap',
                                                                 desired_width = 2048, desired_height = 720, debug_print=False, **kwargs,
                                                                 ):
        """ Exports all rasters, marginals, and posteriors as images

        Called on `_out_ripple_rasters: RankOrderRastersDebugger`
        
        Usage:        
            from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting

            root_export_path: Path = Path(r"/media/halechr/MAX/cloud/University of Michigan Dropbox/Pho Hale/Pho Diba Paper 2023/array_as_image").resolve() # Lab

            epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path) = PosteriorExporting._perform_export_current_epoch_marginal_and_raster_images(_out_ripple_rasters=_out_ripple_rasters, directional_merged_decoders_result=directional_merged_decoders_result, 
                filtered_decoder_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epoch_id_identifier_str='ripple',
                # filtered_decoder_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict, epoch_id_identifier_str='lap',
                active_session_context=curr_context, 
                root_export_path = root_export_path,
            )


            file_uri_from_path(epoch_specific_folder)
            fullwidth_path_widget(a_path=epoch_specific_folder, file_name_label="epoch_specific_folder:")

        """
        from neuropy.core.epoch import ensure_dataframe
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack

        # Get the clicked epoch from the _out_ripple_rasters GUI _____________________________________________________________ #
        active_epoch_tuple = deepcopy(_out_ripple_rasters.active_epoch_tuple) # EpochTuple(Index=21, start=1193.98785331822, stop=1194.1515601143474, label=284, duration=0.1637067961273715, end=1194.1515601143474, wcorr=0.7139066712389361, P_decoder=0.2918033246241751, pearsonr=-0.521100995923076, mseq_len=5, mseq_len_ignoring_intrusions=5, mseq_len_ignoring_intrusions_and_repeats=5, mseq_len_ratio_ignoring_intrusions_and_repeats=0.7142857142857143, mseq_tcov=0.576271186440678, mseq_dtrav=165.8334349469577, avg_jump_cm=34.83895692162977, travel=0.1899320081397262, coverage=0.3050847457627119, total_distance_traveled=0.8474576271186441, track_coverage_score=0.8474576271186441, longest_sequence_length=5, longest_sequence_length_ratio=0.7142857142857143, direction_change_bin_ratio=0.0, congruent_dir_bins_ratio=1.0, total_congruent_direction_change=243.8726984514084, total_variation=243.8726984514084, integral_second_derivative=5281.281702837474, stddev_of_diff=19.441955441371213, is_user_annotated_epoch=True, is_valid_epoch=True, session_name='2006-6-07_16-40-19', delta_aligned_start_t=-42.27839204540942, pre_post_delta_category='pre-delta', maze_id=0, unique_active_aclus=array([39,  7, 52, 63, 40,  6, 27, 55, 11, 53, 24, 17,  8, 60]), n_unique_aclus=14)
        if debug_print:
            print(f'active_epoch_tuple: {active_epoch_tuple}')
        # active_epoch_dict = {k:getattr(active_epoch_tuple, k) for k in ['start', 'stop', 'ripple_idx', 'Index']} # , 'session_name', 'time_bin_size', 'delta_aligned_start_t' {'start': 1161.0011335673044, 'stop': 1161.274357107468, 'session_name': '2006-6-09_1-22-43', 'time_bin_size': 0.025, 'delta_aligned_start_t': 131.68452480540145}
        active_epoch_dict = {k:getattr(active_epoch_tuple, k) for k in ['start', 'stop', 'Index', 'label']} # , 'session_name', 'time_bin_size', 'delta_aligned_start_t' {'start': 1161.0011335673044, 'stop': 1161.274357107468, 'session_name': '2006-6-09_1-22-43', 'time_bin_size': 0.025, 'delta_aligned_start_t': 131.68452480540145}
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
        
        # Determine the index we'll use ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        an_active_filter_epochs_decoder_result = list(filtered_decoder_filter_epochs_decoder_result_dict.values())[0] #['long_LR']
        epoch_data_IDXs = an_active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) # [262], [296]
        # {k:an_active_filter_epochs_decoder_result.filter_epochs.epochs.find_data_indicies_from_epoch_times(np.atleast_1d(clicked_epoch[0])) for k, an_active_filter_epochs_decoder_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
        # active_epoch_id: int = int(epoch_data_IDXs[0])
        active_epoch_data_IDX: int = int(epoch_data_IDXs[0])
        active_epoch_id: int = int(ensure_dataframe(an_active_filter_epochs_decoder_result.filter_epochs)['label'][active_epoch_data_IDX])
        

        # active_epoch_id: int = int(active_epoch_dict['Index']) #
        ## OUTPUTS: active_epoch_id

        ## Session-specific folder:
        context_specific_root_export_path = root_export_path.joinpath(active_session_context.get_description(separator='_')).resolve()
        context_specific_root_export_path.mkdir(exist_ok=True)
        assert context_specific_root_export_path.exists()

        # Epoch-specific folder:
        complete_epoch_identifier_str: str = f"{epoch_id_identifier_str}_{active_epoch_id:03d}" # format `{active_epoch_dict['Index']}` as a fixed-width integer with leading zeros as needed to make sure the resulting files can be sorted by filename
        # ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"ripple_{active_epoch_dict['ripple_idx']}").resolve()
        # ripple_specific_folder: Path = context_specific_root_export_path.joinpath(f"ripple_{active_epoch_dict['Index']}").resolve()
        a_specific_epoch_type_specific_folder: Path = context_specific_root_export_path.joinpath(complete_epoch_identifier_str).resolve() # like 'ripple' or 'laps'
        
        a_specific_epoch_type_specific_folder.mkdir(exist_ok=True)
        assert a_specific_epoch_type_specific_folder.exists()
        # file_uri_from_path(ripple_specific_folder)
        # fullwidth_path_widget(a_path=ripple_specific_folder, file_name_label="lap_specific_folder:")

        # clicked_epoch: {'start': 105.40014315512963, 'stop': 105.56255971186329, 'ripple_idx': 8, 'Index': 8}
        # clicked_epoch: [105.4 105.563]
        # ripple_8

        # ==================================================================================================================== #
        # Export Rasters:                                                                                                      #
        # ==================================================================================================================== #

        # Save out the actual raster-plots ___________________________________________________________________________________ #
        _out_rasters_save_paths = _out_ripple_rasters.save_figure(export_path=a_specific_epoch_type_specific_folder,
                                                                width=desired_width,
                                                                #    height=desired_height,
                                                                **kwargs,
                                                                )
        # _out_rasters_save_paths

        # OUTPUTS: ripple_specific_folder, _out_rasters_save_paths
        out_image_save_tuple_dict = cls._test_export_marginals_for_figure(directional_merged_decoders_result=directional_merged_decoders_result,
                                                                                        filtered_decoder_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict, ## laps
                                                                                        clicked_epoch=clicked_epoch,
                                                                                        context_specific_root_export_path=context_specific_root_export_path, epoch_specific_folder=a_specific_epoch_type_specific_folder,
                                                                                        epoch_id_identifier_str=epoch_id_identifier_str, complete_epoch_identifier_str=complete_epoch_identifier_str,
                                                                                        debug_print=False, desired_width=desired_width, desired_height=desired_height)
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

        return a_specific_epoch_type_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path)


    @function_attributes(short_name=None, tags=['TEMP', 'export', 'image', 'files', 'merge', 'combine', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-30 14:51', related_items=[])
    @classmethod
    def post_export_build_combined_images(cls, out_custom_formats_dict, custom_merge_layout_dict: Optional[Dict]=None, epoch_name_list = ['laps', 'ripple'], included_epoch_idxs: Optional[List]=None, progress_print:bool=False, should_use_raw_rgba_export_image: bool=True):
        """merges the 4 1D decoders and the multi-color pseudo2D to produce a single combined output image for each epoch

        Responsible for the `_temp_individual_posteriors/2025-08-13/gor01_one_2006-6-09_1-22-43/ripple/combined/multi` images
        
        Usage:
            from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting

            _out_final_merged_image_save_paths, _out_final_merged_images = PosteriorExporting.post_export_build_combined_images(out_custom_formats_dict=out_custom_formats_dict)
        
        """
        from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack, image_grid, ImageOperationsAndEffects
        
        ## INPUTS: out_custom_formats_dict

        # merges the separate 1D and the multiColor merged posteriors into a single image

        # active_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL', 'psuedo2D_ignore']
        active_1D_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        active_1D_decoder_name_labels = ['long <', 'long >', 'short <', 'short >']
        active_1D_decoder_name_to_label_dict = dict(zip(active_1D_decoder_names, active_1D_decoder_name_labels))

        pseudo_2D_decoder_name: str = 'psuedo2D_ignore'

        # epoch_name_list = ['laps', 'ripple']

        _out_final_merged_images = []
        _out_final_merged_image_save_paths: List[Path] = []

        # normalization_column_labels: List[str] = ['individual norm', 'context-weighted norm']
        normalization_column_labels: List[str] = ['indiv.', 'global']
        ## The preferred search order to look for images. Stops after finding the first one:
        export_format_name_options = ['greyscale_shared_norm', 'viridis_shared_norm', 'greyscale']
        separator_color=f'#1b0014' ## for greyscale
        # export_format_name_options = ['viridis_shared_norm', 'greyscale_shared_norm', 'greyscale']
        # separator_color=f'#fae2e2'

        def _subfn_try_find_existing_format(out_custom_formats_dict, export_format_name_options: List[str] = ['greyscale_shared_norm', 'viridis_shared_norm', 'greyscale']) -> Tuple[Optional[str], int]:
            """ tries to find the existing export name from a list of options 
            """
            active_found_export_format_name: str = None
            a_decoder_name = 'long_LR' ## temp
            _a_partial_dict = out_custom_formats_dict[f'{a_decoding_epoch_name}.{a_decoder_name}']
            

            ## find the appropriate `active_found_export_format_name`
            for an_export_format_name in export_format_name_options:
                if (active_found_export_format_name is None):
                    if (an_export_format_name in _a_partial_dict):
                        active_found_export_format_name = an_export_format_name

            num_epochs: int = 0
            if active_found_export_format_name is not None:
                num_epochs = len(_a_partial_dict[active_found_export_format_name])
            
            return active_found_export_format_name, num_epochs



        if custom_merge_layout_dict is None:
            custom_merge_layout_dict = [['greyscale'],
                ['greyscale_shared_norm'],
                # ['psuedo2D_ignore/raw_rgba'], ## Implicitly always appends the pseudo2D_ignore/raw_rgba image at the bottom row
            ]
            

        if not should_use_raw_rgba_export_image:
            _label_kwargs = ImagePostRenderFunctionSets._get_export_color_scheme_kwargs(is_prepare_for_publication=True)
        
        for a_decoding_epoch_name in epoch_name_list:
            ## e.g. 'laps' or 'ripple'
            try:
                ## find the appropriate `active_found_export_format_name`
                active_found_export_format_name, num_epochs = _subfn_try_find_existing_format(out_custom_formats_dict=out_custom_formats_dict, export_format_name_options=export_format_name_options)
                if active_found_export_format_name is None:
                    raise KeyError('skipping')

                ## Iterate through each epoch:
                for epoch_IDX in np.arange(num_epochs):
                    if (included_epoch_idxs is None) or ((included_epoch_idxs is not None) and (epoch_IDX in included_epoch_idxs)):
                        if progress_print:
                            print(f'{a_decoding_epoch_name}[{epoch_IDX}]: processing...')
                            
                        _tmp_curr_merge_layout_raster_imgs = []
                        for row_idx, a_merge_layout_row in enumerate(custom_merge_layout_dict):
                            _tmp_curr_row_raster_imgs = []
                            if progress_print:
                                print(f'epoch_IDX: {epoch_IDX}')
                        
                            if progress_print:
                                print(f'\trow_idx: {row_idx}')
                            for col_idx, a_merge_layout_col in enumerate(a_merge_layout_row):
                                # vertical stack
                                if progress_print:
                                    print(f'\t\tcol_idx: {col_idx}')
                                # _tmp_curr_raster_imgs = []
                                for decoder_IDX, a_decoder_name in enumerate(active_1D_decoder_names):
                                    ## get the single decoder image for this format:
                                    # a_config = out_custom_formats_dict[f'{a_decoding_epoch_name}.{a_decoder_name}'][active_found_export_format_name][epoch_IDX] # a HeatmapExportConfig
                                    a_config = out_custom_formats_dict[f'{a_decoding_epoch_name}.{a_decoder_name}'][a_merge_layout_col][epoch_IDX] # a HeatmapExportConfig
                                    # a_config.posterior_saved_path ## the saved image file
                                    an_active_img = deepcopy(a_config.posterior_saved_image) ## the actual image object
                                    an_active_img = an_active_img.reduce(factor=(1, 4)) ## scale image down by 1/4 in height but leave the original width
                                    # an_active_img = an_active_img.reduce(factor=(4, 1)) ## scale image down by 1/4 in width but leave the original height
                                    curr_img_size = deepcopy(an_active_img.size)
                                    
                                    if progress_print:
                                        print(f'\t\t\tdecoder[{decoder_IDX}]: .size - {an_active_img.size}')
                                        
                                    ## Add overlay text
                                    # an_active_img = ImageOperationsAndEffects.add_overlayed_text(an_active_img, a_decoder_name, font_size=48, text_color="#FF00EACA",
                                    #                                                                     #  inverse_scale_factor=(2, 1),
                                    #                                                                     stroke_width=1, stroke_fill="#000000",
                                    #                                                                      )
                                    ## Decoder label to the left, and only on the first col
                                    if (col_idx == 0) and (row_idx == 0):
                                        ## note, these aren't really the row/col index because they're kinda hardcoded rn.
                                        a_decoder_name_label: str = active_1D_decoder_name_to_label_dict[a_decoder_name] ## a_decoder_name: just the name like 'long_LR'
                                        an_active_img = ImageOperationsAndEffects.add_boxed_adjacent_label(an_active_img, a_decoder_name_label, image_edge='left', font_size=48, text_color="#000000",
                                                                                                            background_color=(255, 255, 255, 0),
                                                                                                            # fixed_label_region_size = [_out_row_stack.width, 62]
                                                                                                            ) ## why is this size unchanged from before adding the label?
                                        curr_img_size = deepcopy(an_active_img.size)
                                        
                                    if progress_print:
                                        print(f'\t\t\t\tpre-append img_size: {an_active_img.size}')
                                    _tmp_curr_row_raster_imgs.append(an_active_img)
                                ## END for decoder_IDX, a_d...
                            ## Build merged row image:
                            # separator_color=f'#ff0000'
                            _out_row_stack = vertical_image_stack(_tmp_curr_row_raster_imgs, padding=5, separator_color=separator_color)
                            
                            # _out_row_stack = horizontal_image_stack(_tmp_curr_row_raster_imgs, padding=5, separator_color=separator_color)

                            ## Add top normalization labels:
                            # [row_idx]
                            
                            if row_idx < len(normalization_column_labels):
                                normalization_label_text: str = normalization_column_labels[row_idx] # 'global'      
                                _out_row_stack = ImageOperationsAndEffects.add_boxed_adjacent_label(_out_row_stack, normalization_label_text, image_edge='top', font_size=48, text_color="#000000",
                                                                                                    background_color=(255, 255, 255, 0),
                                                                                                    fixed_label_region_size = [_out_row_stack.width, 62]
                                                                                                    )


                            _tmp_curr_merge_layout_raster_imgs.append(_out_row_stack)

                        ## END for row_idx, a_merge_layout_row in enumerate(custom_merge_layout_dict)
                        

                        ## Build merged all rows image:
                        # separator_color=f'#66ff00' 
                        _out_vstack = horizontal_image_stack(_tmp_curr_merge_layout_raster_imgs, padding=25, separator_color=separator_color) # , separator_color=separator_color
                        # _out_vstack = vertical_image_stack(_tmp_curr_merge_layout_raster_imgs, padding=35, separator_color=separator_color)
                        _out_vstack = _out_vstack.reduce(factor=(2, 1)) ## scale image down by 1/2 in width but leave the original height
                        _tmp_curr_merge_layout_raster_imgs = [_out_vstack, ] # combined image with both columns concatenated is back
                        if progress_print:
                            print(f'_out_vstack.size: {_out_vstack.size}')
                            
                        ## get the multicolor iamge last:
                        if should_use_raw_rgba_export_image:
                            try:
                                a_config = out_custom_formats_dict[f'{a_decoding_epoch_name}.{pseudo_2D_decoder_name}']['raw_rgba'][epoch_IDX] # a HeatmapExportConfig
                                _tmp_curr_merge_layout_raster_imgs.append(a_config.posterior_saved_image)
                                if progress_print:
                                    print(f'\ta_config.posterior_saved_image.size: {a_config.posterior_saved_image.size}')
                            except KeyError as e:
                                # KeyError: "Invalid keys: '['laps', 'long_LR']'"
                                print(f"\tcould not get multicolor image data for out_custom_formats_dict[f'{a_decoding_epoch_name}.{pseudo_2D_decoder_name}']['raw_rgba'][{epoch_IDX}], key error: {e}\n\tskipping.")    
                                pass
                            except Exception as e:
                                raise

                        else:
                            ## GET Text
                            # post_render_image_functions_dict_list: List[Dict[str, Callable]] = _build_image_export_functions_dict(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result)
                            epoch_id_text: str = f"{a_decoding_epoch_name}[{epoch_IDX}] r{row_idx}" # normalization_column_labels[row_idx] # 'global'      

                            ## INPUTS: _label_kwargs
                            # _out_vstack = ImageOperationsAndEffects.add_boxed_adjacent_label(_out_vstack, epoch_id_text, image_edge='bottom', font_size=24, text_color="#000000",
                            #                                         background_color=(255, 255, 255, 0),
                            #                                         fixed_label_region_size = [_out_vstack.width, _label_kwargs['fixed_label_region_height']]
                            #                                         )
                            # _label_kwargs = ImagePostRenderFunctionSets._get_export_color_scheme_kwargs(is_prepare_for_publication=True)
                            # _out_vstack = ImageOperationsAndEffects.add_bottom_label(_out_vstack, label_text=epoch_id_text, **_label_kwargs)
                            # create_label_function = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_bottom_label, **_label_kwargs) #  text_color=(255, 255, 255), background_color=(66, 66, 66), font_size=font_size, fixed_label_region_height=fixed_label_region_height
                            # create_half_width_rectangle_function = ImageOperationsAndEffects.create_fn_builder(ImageOperationsAndEffects.add_half_width_rectangle, height_fraction = 0.1)
                            _tmp_curr_merge_layout_raster_imgs = [_out_vstack, ]
                            pass


                        # a_config.posterior_saved_image ## the actual image object
                        a_posterior_saved_path: Path = a_config.posterior_saved_path ## the saved image file
                        merged_dir = a_posterior_saved_path.parent.parent.parent.joinpath('combined', 'multi')
                        merged_dir.mkdir(exist_ok=True, parents=True)
                        a_merged_posterior_export_path: Path = merged_dir.joinpath(a_posterior_saved_path.name) # '_temp_individual_posteriors/2025-05-30/gor01_one_2006-6-12_15-55-31/ripple/combined/multi/p_x_given_n[2].png'
                        
                        ## Build merged all rows image:
                        # separator_color=f'#006eff' 
                        # if len(_tmp_curr_merge_layout_raster_imgs) > 1:
                        _out_vstack = vertical_image_stack(_tmp_curr_merge_layout_raster_imgs, padding=10, separator_color=separator_color)
                        # else:
                        #     _out_vstack = _tmp_curr_merge_layout_raster_imgs[0] ## just get the only real image
                        if progress_print:
                            print(f'final _out_vstack.size: {_out_vstack.size}')
                        _out_final_merged_images.append(_out_vstack)

                        ## save it
                        ## a_merged_posterior_export_path, _out_vstack
                        _out_vstack.save(a_merged_posterior_export_path) # Save image to file
                        _out_final_merged_image_save_paths.append(a_merged_posterior_export_path)

                        ## END for col_idx, a_merge_layout_col in enumerate(a_merge_layout_row)...
                        
                ## END for epoch_IDX in np.arange(num_epochs)...
            except KeyError as e:
                # KeyError: "Invalid keys: '['laps', 'long_LR']'"
                print(f'\tcould not get export data for a_decoding_epoch_name: "{a_decoding_epoch_name}", key error: {e}\n\tskipping.')    
                # continue
                raise
            
            except Exception as e:
                raise
            
        ## END for a_decoding_epoch_name in epoch_name_list

        return _out_final_merged_image_save_paths, _out_final_merged_images






@metadata_attributes(short_name=None, tags=['export', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-16 22:28', related_items=[])
@define(slots=False)
class LoadedPosteriorContainer:
    """ Loads Posteriors exported as .h5 files

    Usage:    
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import H5FileAggregator
        from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
        from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
        from pyphoplacecellanalysis.Pho2D.data_exporting import LoadedPosteriorContainer
        from neuropy.utils.indexing_helpers import flatten, flatten_dict
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
                        
        ## INPUTS: parsed_h5_files_df
        decoded_posteriors_parsed_h5_files_df = parsed_h5_files_df[parsed_h5_files_df['file_type'] == 'decoded_posteriors']

        print(decoded_posteriors_parsed_h5_files_df['custom_replay_name'].unique())
        # matching_custom_replay_name_str: str = "withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]"
        # matching_custom_replay_name_str: str = "withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]"
        matching_custom_replay_name_str: str = 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'
        decoded_posteriors_parsed_h5_files_df = decoded_posteriors_parsed_h5_files_df[decoded_posteriors_parsed_h5_files_df['custom_replay_name'] == matching_custom_replay_name_str]
        decoded_posteriors_parsed_h5_files_df
        decoded_posteriors_h5_files = [Path(v.as_posix()).resolve() for v in decoded_posteriors_parsed_h5_files_df['path'].to_list()]
        # decoded_posteriors_h5_files
        ## OUTPUTS: decoded_posteriors_h5_files

        all_sessions_exported_posteriors_dict, all_sessions_exported_posteriors_data_only_dict = LoadedPosteriorContainer.load_batch_hdf5_exports(exported_posterior_data_h5_files=decoded_posteriors_h5_files)
        ## OUTPUTS: all_sessions_exported_posteriors_dict, all_sessions_exported_posteriors_data_only_dict
        
        ## Using the outputs:        
        a_posterior_container: LoadedPosteriorContainer = all_sessions_exported_posteriors_dict['/kdiba/gor01/two/2006-6-12_16-53-46']
        # list(a_posterior_container.ripple_data_field_dict.keys()) # list(a_posterior_container.ripple_data_field_dict.keys()): ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']

        most_likely_positions_dict: Dict[types.DecoderName, List[NDArray]] = a_posterior_container.ripple_data_field_dict['most_likely_positions']
        t_bin_centers: List[NDArray] = list(a_posterior_container.ripple_data_field_dict['t_bin_centers'].values())[0] ## they're all the same for each decoder, so just get the first decoder's values

        a_decoder_name: str = 'long_LR'
        t_bin_centers_flat = np.concatenate(t_bin_centers)
        n_time_bins: int = len(t_bin_centers_flat)
        most_likely_positions_flat = np.concatenate(most_likely_positions_dict[a_decoder_name])
        assert np.shape(t_bin_centers_flat) == np.shape(most_likely_positions_flat)
        print(f'n_time_bins: {n_time_bins}') ## 639? Not so many


    """
    file_path: Optional[Path] = field(default=None)
    session_key_parts: List = field(default=Factory(list))
    custom_replay_parts: List = field(default=Factory(list))
    
    ripple_data_field_dict: Dict = field(default=Factory(dict))

    _decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    

    @property
    def all_parts_tuple(self) -> Tuple:
        """The ripple_img_dict property."""
        return tuple([*self.session_key_parts, *self.custom_replay_parts])
    
    @property
    def full_complete_context_key(self) -> str:
        """The 'kdiba/gor01/one/2006-6-08_14-26-15/withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0' property."""
        return '/' + ('/'.join(list(self.all_parts_tuple))).lstrip('/') ## ensure starting forward-slash
    
    @property
    def ripple_img_dict(self):
        """The ripple_img_dict property."""
        return self.ripple_data_field_dict['p_x_given_n_grey']


    @classmethod
    def init_from_load_path(cls, load_path: Path):
        """ loads """
        Assert.path_exists(load_path)

        ## used for reconstituting dataset:
        dataset_type_fields = ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']
        decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']

        _out_dict, (session_key_parts, custom_replay_parts) = PosteriorExporting.load_decoded_posteriors_from_HDF5(load_path=load_path, debug_print=True)
        _out_ripple_only_dict = {k:v['ripple'] for k, v in _out_dict.items()} ## cut down to only the laps

        ## build the final ripple data outputs:
        ripple_data_field_dict = {}
        # active_var_key: str = 'p_x_given_n' # dataset_type_fields	

        for active_var_key in dataset_type_fields:
            ripple_data_field_dict[active_var_key] = {
                a_decoder_name: [v for v in _out_ripple_only_dict[a_decoder_name][active_var_key]] for a_decoder_name in decoder_names
            }

        _obj = cls(file_path=load_path, 
                   ripple_data_field_dict=ripple_data_field_dict, session_key_parts=session_key_parts, custom_replay_parts=custom_replay_parts)
        # ripple_img_dict = ripple_data_field_dict['p_x_given_n_grey']

        return _obj

    @function_attributes(short_name=None, tags=['MAIN', 'load', 'batch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-17 03:06', related_items=['PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5'])
    @classmethod
    def load_batch_hdf5_exports(cls, exported_posterior_data_h5_files: List[Path]) -> Dict[str, "LoadedPosteriorContainer"]:
        """ 
        
        all_sessions_exported_posteriors_list = LoadedPosteriorContainer.load_batch_hdf5_exports(exported_posterior_data_h5_files=exported_posterior_data_h5_files)
        
        """
        exported_posterior_data_h5_files = [Path(v).resolve() for v in exported_posterior_data_h5_files] ## should parse who name and stuff... but we don't.
        all_sessions_exported_posteriors_list: List["LoadedPosteriorContainer"] = [cls.init_from_load_path(load_path=hdf_load_path) for hdf_load_path in exported_posterior_data_h5_files] ## need to export those globally unique identifiers for each aclu within a session
        # _global_output_dict = {tuple([*v.session_key_parts, *v.custom_replay_parts]):v for v in all_sessions_exported_posteriors_list}
        all_sessions_exported_posteriors_dict = {v.full_complete_context_key:v for v in all_sessions_exported_posteriors_list}
        all_sessions_exported_posteriors_data_only_dict = {v.full_complete_context_key:deepcopy(v.ripple_data_field_dict) for v in all_sessions_exported_posteriors_list}

        return all_sessions_exported_posteriors_dict, all_sessions_exported_posteriors_data_only_dict
    

        # for i, (a_path, a_first_spike_time_tuple) in enumerate(zip(first_spike_activity_data_h5_files, all_sessions_first_spike_activity_tuples)):
        #     all_cells_first_spike_time_df_loaded, global_spikes_df_loaded, global_spikes_dict_loaded, first_spikes_dict_loaded, extra_dfs_dict_loaded = a_first_spike_time_tuple ## unpack


@metadata_attributes(short_name=None, tags=['export', 'posterior', 'datasource', 'plotting'], input_requires=[], output_provides=[], uses=[], used_by=['DataFrameFilter'], creation_date='2024-12-16 22:28', related_items=[])
@define(slots=False, eq=False, repr=False)
class PosteriorPlottingDatasource:
    """ a datasource that provides posteriors
    
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorPlottingDatasource
    
    """
    data_field_dict: Dict = field(default=Factory(dict))
    plot_heatmap_fn: Callable = field(default=None)

    def get_posterior_data(self, session_name: str, custom_replay_name: str, a_variable_name:str='p_x_given_n_grey', a_decoder_name: str='long_LR', last_selected_idx: Optional[int] = 0):
        """ 
        
        """
        '/kdiba_gor01_one_2006-6-09_1-22-43/withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'
        '/kdiba/gor01/one/2006-6-09_1-22-43/withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'

        session_name = session_name.replace('_', '/', 3) # '/kdiba_gor01_one_2006-6-09_1-22-43' -> '/kdiba/gor01/one/2006-6-09_1-22-43'
        # if last_selected_idx is None:
        #     # df_filter.hover_posterior_data.ripple_img_dict
        #     # df_filter.hover_posterior_preview_figure_widget.add_heatmap()
        full_key: str = f'/{session_name}/{custom_replay_name}'
        assert full_key in self.data_field_dict, f"full_key: '{full_key}' was not in self.data_field_dict: {list(self.data_field_dict.keys())}"
        a_heatmap_img = self.data_field_dict[full_key][a_variable_name][a_decoder_name][last_selected_idx]
        return a_heatmap_img

    # def _plot_hoverred_heatmap_preview_posterior(self, session_name: str, custom_replay_name: str, last_selected_idx: Optional[int] = 0):
    #     # if last_selected_idx is None:
    #     #     # df_filter.hover_posterior_data.ripple_img_dict
    #     #     # df_filter.hover_posterior_preview_figure_widget.add_heatmap()
    #     full_key: str = f'/{session_name}/{custom_replay_name}'
    #     assert full_key in self.data_field_dict, f"full_key: '{full_key}' was not in self.data_field_dict: {list(self.data_field_dict.keys())}"
    #     a_heatmap_img = self.data_field_dict[full_key]['long_LR'][last_selected_idx]
        
    #     # a_heatmap_img = df_filter.hover_posterior_data.ripple_img_dict['long_LR'][last_selected_idx]    
        
    #     ## update the plot
    #     if self.plot_heatmap_fn is not None:
    #         self.plot_heatmap_fn(
    #     df_filter.hover_posterior_preview_figure_widget.add_heatmap(z=a_heatmap_img, showscale=False, name='selected_posterior', )
