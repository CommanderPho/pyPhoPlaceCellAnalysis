# ==================================================================================================================== #
# 2024-08-15 - Computer Vision Approches to line recognition                                                           #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from nptyping import NDArray
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder #typehinting only
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult


from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter
from ipywidgets import interact, FloatSlider
from copy import deepcopy
import numpy as np

from ipywidgets import interact, FloatSlider, fixed
from IPython.display import display

# Custom Type Definitions ____________________________________________________________________________________________ #
T = TypeVar('T')
DecoderListDict: TypeAlias = Dict[types.DecoderName, List[T]] # Use like `v: DecoderListDict[NDArray]`


DecoderResultDict: TypeAlias = Dict[types.DecoderName, DecodedFilterEpochsResult] # Use like `v: DecoderListDict[NDArray]`





@define(slots=False, eq=False)
@metadata_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-14 00:00', related_items=[])
class ComputerVisionComputations:
    """ 
    from pyphoplacecellanalysis.Analysis.Decoder.computer_vision import ComputerVisionComputations
    
    # Visualization ______________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    
    out = TransitionMatrixComputations.plot_transition_matricies(decoders_dict=decoders_dict, binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict)
    out

    """
    # ==================================================================================================================== #
    # Save/Load                                                                                                            #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'save', 'export'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=[])
    def export_decoded_posteriors_as_images(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult, # decoder_ripple_filter_epochs_decoder_result_dict: DecoderResultDict,
                                             posterior_out_folder:Path='output/_temp_individual_posteriors',
                                  should_export_separate_color_and_greyscale: bool = True, out_context=None, debug_print=False): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the transitiion matrix info to a file
        
        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.computer_vision import ComputerVisionComputations
            should_export_separate_color_and_greyscale: bool = True
            # a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_laps_filter_epochs_decoder_result_dict['long_LR']
            # epochs_name='laps'
            a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_ripple_filter_epochs_decoder_result_dict['long_LR']
            epochs_name='ripple'

            parent_output_folder = Path(r'output/_temp_individual_posteriors').resolve()
            posterior_out_folder = parent_output_folder.joinpath(DAY_DATE_TO_USE, epochs_name).resolve()

            (posterior_out_folder, posterior_out_folder_greyscale, posterior_out_folder_color), _save_out_paths = ComputerVisionComputations.export_decoded_posteriors_as_images(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, posterior_out_folder=posterior_out_folder, should_export_separate_color_and_greyscale=should_export_separate_color_and_greyscale)

            if should_export_separate_color_and_greyscale:
                fullwidth_path_widget(posterior_out_folder_greyscale)
                fullwidth_path_widget(posterior_out_folder_color)
            else:
                fullwidth_path_widget(posterior_out_folder)
                
                
        """

        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult

        # a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_laps_filter_epochs_decoder_result_dict['long_LR']
        # epochs_name='laps'
        # a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = decoder_ripple_filter_epochs_decoder_result_dict['long_LR']
        # epochs_name='ripple'



        if not isinstance(posterior_out_folder, Path):
            posterior_out_folder = Path(posterior_out_folder).resolve()

        # parent_output_folder = Path(r'output/_temp_individual_posteriors').resolve()
        # posterior_out_folder = parent_output_folder.joinpath(DAY_DATE_TO_USE, epochs_name).resolve()
        posterior_out_folder.mkdir(parents=True, exist_ok=True)

        if should_export_separate_color_and_greyscale:
            posterior_out_folder_greyscale = posterior_out_folder.joinpath('greyscale').resolve()
            posterior_out_folder_color = posterior_out_folder.joinpath('color').resolve()
            posterior_out_folder_greyscale.mkdir(parents=True, exist_ok=True)
            posterior_out_folder_color.mkdir(parents=True, exist_ok=True)

        num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
        
        _save_out_paths = []
        for i in np.arange(num_filter_epochs):
            active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=i)

            if should_export_separate_color_and_greyscale:
                _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=posterior_out_folder_color, export_grayscale=False, skip_img_normalization=False, desired_height=1024)
                _save_out_paths.append(posterior_save_path)
                _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=posterior_out_folder_greyscale, export_grayscale=True, skip_img_normalization=False, desired_height=1024)
                _save_out_paths.append(posterior_save_path)	
            else:
                _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=posterior_out_folder, export_grayscale=True, skip_img_normalization=False, desired_height=1024)
                _save_out_paths.append(posterior_save_path)
                
            # if i > 25 and i < 30:
                # _posterior_image
                
        # 

        if should_export_separate_color_and_greyscale:
            _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=posterior_out_folder_color, export_grayscale=False, skip_img_normalization=False, desired_height=1024)
            _save_out_paths.append(posterior_save_path)
            _posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder=posterior_out_folder_greyscale, export_grayscale=True, skip_img_normalization=False, desired_height=1024)
            return (posterior_out_folder, posterior_out_folder_greyscale, posterior_out_folder_color), _save_out_paths
        else:
            return posterior_out_folder, _save_out_paths
                
        


    @classmethod
    def interactive_image_preview(cls, input_img, blur_v_sigma, blur_h_sigma, hessian_sigma):
        # Perform Gaussian blur
        blurred_img = gaussian_filter(input_img.astype(float), sigma=(blur_v_sigma, blur_h_sigma), mode='nearest')

        # Compute the Hessian matrix
        H_out = hessian_matrix(blurred_img, sigma=hessian_sigma, mode='nearest')

        # Compute the eigenvalues of the Hessian matrix
        lambda1, lambda2 = hessian_matrix_eigvals(H_out)

        # Ridge detection based on eigenvalues
        ridges = np.abs(lambda2)  # Use the second eigenvalue for ridges

        # Display the results (this will work as in your notebook)
        display(blurred_img, ridges, clear=True)
        
        return ridges


    @classmethod
    def run_interactive(cls, input_img):
        interact(cls.interactive_image_preview, 
                 input_img=fixed(input_img),
                 blur_v_sigma=FloatSlider(min=0, max=10, step=0.5, value=0, description='blur_v_sigma'),
                 blur_h_sigma=FloatSlider(min=0, max=10, step=0.5, value=0, description='blur_h_sigma'),		 
                 hessian_sigma=FloatSlider(min=0, max=10, step=0.5, value=0, description='hessian_sigma'))
        

    @classmethod
    def interactive_binarization(cls, ridges, ridge_binarization_threshold):
        ridge_binary = ridges > ridge_binarization_threshold
        # cls._display_binarization(ridge_binarization_threshold, ridge_binary)
        # display(ridge_binary, clear=False)
        
        # # Store these values if needed
        # cls.user_ridge_binarization_threshold = ridge_binarization_threshold
        # cls.user_ridge_binary = ridge_binary

        return ridge_binary
    

    # @staticmethod
    # def _display_binarization(threshold, binary_image):
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(binary_image, cmap='gray')
    #     plt.title(f'Ridge Binarization (Threshold: {threshold})')
    #     plt.axis('off')
    #     plt.show()

    @classmethod
    def run_binarization_interactive(cls, ridges):
        interact(cls.interactive_binarization, 
                 ridges=fixed(ridges),
                 ridge_binarization_threshold=FloatSlider(min=0, max=np.nanmax(ridges), step=np.nanmax(ridges)/100, value=0, description='Binarization Threshold'))
        
