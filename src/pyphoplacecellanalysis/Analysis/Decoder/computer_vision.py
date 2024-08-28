# ==================================================================================================================== #
# 2024-08-15 - Computer Vision Approches to line recognition                                                           #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from enum import Enum
from matplotlib import pyplot as plt
from neuropy.utils.result_context import IdentifyingContext
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar, NewType, Literal
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
from ipywidgets import interact, FloatSlider, IntSlider, fixed
from copy import deepcopy
import numpy as np

import cv2
from PIL import Image

from IPython.display import display

# Custom Type Definitions ____________________________________________________________________________________________ #
T = TypeVar('T')
DecoderListDict: TypeAlias = Dict[types.DecoderName, List[T]] # Use like `v: DecoderListDict[NDArray]`


DecoderResultDict: TypeAlias = Dict[types.DecoderName, DecodedFilterEpochsResult] # Use like `v: DecoderListDict[NDArray]`

aclu_index: TypeAlias = int # an integer index that is an aclu
DecoderName = NewType('DecoderName', str)

# Define the type alias
KnownEpochsName = Literal['laps', 'ripple', 'other']




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
                # Greyscale only:
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
                
        
    # ==================================================================================================================== #
    # Save/Load                                                                                                            #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['posterior', 'HDF5', 'output', 'save', 'export'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-28 02:38', related_items=['load_decoded_posteriors_from_HDF5'])
    def save_decoded_posteriors_to_HDF5(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult, save_path:Path='decoded_epoch_posteriors.h5', allow_append:bool=False, out_context=None, debug_print=False): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the transitiion matrix info to a file
        
        _save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_transition_matricies')
        _save_path = ComputerVisionComputations.save_decoded_posteriors_to_HDF5(a_decoder_decoded_epochs_result=a_decoder_decoded_epochs_result, out_context=_save_context, save_path='output/transition_matrix_data.h5')
        _save_path

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
                active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=i)
                epoch_data_idx_str: str = f"{i:0{num_required_zero_padding}d}"
                # _curr_context = out_context.overwriting_context(epoch_idx=i)
                _curr_context = out_context.overwriting_context(epoch_idx=epoch_data_idx_str)
                _curr_key: str = _curr_context.get_description(separator='/')
                if not _curr_key.startswith('/'):
                    _curr_key = "/" + _curr_key
                # active_captured_single_epoch_result.to_hdf(save_path, key=_curr_key, debug_print=True, enable_hdf_testing_mode=True)
                active_captured_single_epoch_result.to_hdf(f, key=_curr_key, debug_print=debug_print, enable_hdf_testing_mode=False, required_zero_padding=num_required_zero_padding)
        
        return save_path
    

    @classmethod
    @function_attributes(short_name=None, tags=['posterior', 'HDF5', 'load'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=['save_decoded_posteriors_to_HDF5'])
    def load_decoded_posteriors_from_HDF5(cls, load_path: Path, debug_print=True) -> Dict[types.DecoderName, Dict[KnownEpochsName, Dict]]:
        """
        Load the transition matrix info from a file
        
        load_path = Path('output/transition_matrix_data.h5')
        _out_dict = ComputerVisionComputations.load_decoded_posteriors_from_HDF5(load_path=load_path, debug_print=False)
        ripple_0_img = _out_dict['long_LR']['ripple']['p_x_given_n_grey'][0]
        lap_0_img = _out_dict['long_LR']['laps']['p_x_given_n_grey'][0]
        lap_0_img
        
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

        out_dict: Dict = {}

        with h5py.File(load_path, 'r') as f:
            
            main_save_group = f[_save_key]
            if debug_print:
                print(f'main_save_group: {main_save_group}')
            
            for decoder_prefix in main_save_group.keys():
                if debug_print:
                    print(f'decoder_prefix: {decoder_prefix}')
                if decoder_prefix not in out_dict:
                    out_dict[decoder_prefix] = {}

                decoder_group = main_save_group[decoder_prefix]
                for known_epochs_name in decoder_group.keys():
                    if debug_print:
                        print(f'\tknown_epochs_name: {known_epochs_name}')
                    if known_epochs_name not in out_dict[decoder_prefix]:
                        out_dict[decoder_prefix][known_epochs_name] = {}
                        
                    decoder_epochtype_group = decoder_group[known_epochs_name]
                    
                    ## allocate outputs:
                    out_dict[decoder_prefix][known_epochs_name] = {k:list() for k in dataset_type_fields} # allocate a dict of empty lists for each item in `dataset_type_fields`
                    
                    for dataset_name in decoder_epochtype_group.keys():
                        if debug_print:
                            print(f'\t\tdataset_name: {dataset_name}')
                            
                        ## the lowest-level group before the data itself
                        dataset_final_group = decoder_epochtype_group[dataset_name]
                        # dataset_type_fields
                        
                        for leaf_data_key, leaf_data in dataset_final_group.items():
                            if debug_print:
                                print(f'\t\t\tleaf_data_key: {leaf_data_key}')
                            # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[{dataset_name}]"][()]
                            # array = decoder_epochtype_group[dataset_name][f"p_x_given_n[{dataset_name}]"][()]
                            array = leaf_data[()] # should get the NDArray
                            if debug_print:
                                print(f'\t\t\t\tarray: {type(array)}')
                            
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

        return out_dict

    # ==================================================================================================================== #
    # Interactive                                                                                                          #
    # ==================================================================================================================== #
    

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
        
    @classmethod
    def debug_print_img_info(cls, img):
        """ Prints debug info (shape, type, size, etc) for the img
        
        """
        print(f'type(img): {type(img)}')
        print(f'np.shape(img): {np.shape(img)}')
        print(f'img.dtype: {img.dtype}')
        # np.issubdtype(img.dtype, np.integer)
        



    @classmethod
    def top_N_values_binarization(cls, img, top_n:int=3, non_included_index_value=0):
        """ 
        
        """
        _original_img = None
        # original_dtype = img.dtype
        if np.issubdtype(img.dtype, np.integer):
            _original_img = deepcopy(img)
            ## convert to float
            img = (img / 255.0)
            
        # Step 1: Get the sorted indices for each column, ignoring NaNs
        sorted_indices = np.argsort(-np.nan_to_num(img, nan=-np.inf), axis=0)

        # Step 2: Extract the `top_n` indices and values in each column
        top_indices = sorted_indices[:top_n, :]
        top_values = np.take_along_axis(img, top_indices, axis=0)

        # Step 3: Mask NaN values
        top_values[np.isnan(top_values)] = np.nan
        # top_values[np.isnan(top_values)] = -1

        # Step 3: Create a mask initialized to False
        bw_top_values_mask = np.zeros(img.shape, dtype=bool)

        # Step 4: Set the mask to True for the top indices
        for col in range(img.shape[1]):
            valid_indices = top_indices[:, col][~np.isnan(img[top_indices[:, col], col])]
            # valid_indices = top_indices[:, col][(img[top_indices[:, col], col]) != -1]
            bw_top_values_mask[valid_indices, col] = True
            
        # Apply mask to create the masked image
        if _original_img is not None:
            masked_img = deepcopy(_original_img)
        else:
            masked_img = deepcopy(img)

        # masked_img[np.logical_not(bw_top_values_mask)] = non_included_index_value # 0.0
        masked_img[~bw_top_values_mask] = non_included_index_value # 0.0
        
        return bw_top_values_mask, masked_img

    
    

    @classmethod
    def load_image(cls, img_path) -> NDArray:
        """ loads the image from a file """
        if not isinstance(img_path, Path):
            img_path = Path(img_path).resolve()
            
        assert img_path.exists()
        pimg = Image.open(img_path)
        img = np.array(pimg)
        return img


    @classmethod
    def image_moments(cls, img):
        """ Computes the image moments of a 2D image `img` """
        
        ## Moments returned in xy
        def moment_raw(r, i, j):
            _x = np.arange(r.shape[0])**j
            _y = np.arange(r.shape[1])**i
            _XX, _YY = np.meshgrid(_y, _x)
            return (_XX*_YY*r).sum()

        def centroid(M00,M10,M01):
            return M10/(M00 + 1e-5), M01/(M00 + 1e-5)



        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        import cv2
        
        assert np.ndim(img) == 2, f"Image should be a 2D greyscale image, passed as a numpy array. Instead the image is of shape: {np.shape(img)}"
        
        if not isinstance(img, NDArray):
            img = np.array(img) # convert to NDArray
            
        # 'r' refers to the red component only in the example notebook, but the passed image is already greyscale, so it's equivalent.
        r = deepcopy(img)

        ## Compute the moments:
        moments = cv2.moments(r)
        # m_ij: raw moments
        # mu_ij: central moments, translation invariant
        # nu_ij: scale invariants
        hu_moments = cv2.HuMoments(moments)
        # hu moments, rotation invariant

        ## OUTPUTS: moments, hu_moments
        
        # Raw Moments ________________________________________________________________________________________________________ #
        print("Area", np.prod(np.asarray(r.shape)))

        M00 = moment_raw(r, 0,0)
        M10 = moment_raw(r, 1,0)
        M01 = moment_raw(r, 0,1)
        M11 = moment_raw(r, 1,1)
        M20 = moment_raw(r, 2,0)
        M02 = moment_raw(r, 0,2)
        M21 = moment_raw(r, 2,1)
        M12 = moment_raw(r, 1,2)
        M30 = moment_raw(r, 3,0)
        M03 = moment_raw(r, 0,3)
        x_, y_ = centroid(M00, M10, M01)

        M00, M11, M10

        # central moments: translationally invariant _________________________________________________________________________ #
        mu00 = M00 
        mu01 = 0
        mu10 = 0
        mu11 = M11 - x_* M01 # = M11 - y_* M10
        mu20 = M20 - x_ * M10
        mu02 = M02 - y_ * M01
        mu21 = M21 - 2*x_ * M11 - y_ * M20 + 2 * x_**2 * M01
        mu12 = M12 - 2*y_ * M11 - x_ * M02 + 2 * y_**2 * M10

        mu30 = M30 - 3*x_ * M20 + 2 * x_**2 * M10
        mu03 = M03 - 3*y_ * M02 + 2 * y_**2 * M01

        mu11, mu20, mu02, mu21, mu12, mu30, mu03


        # Image Orientation Covariance Matrix from second order central moments ______________________________________________ #
        # mu00 == m00
        mup_20 = mu20/mu00 - x_**2
        mup_02 = mu02/mu00 - y_**2
        mup_11 = mu11/mu00 - x_*y_

        theta = np.arctan(2*mup_11/(mup_20 - mup_02))/2
        print("intensity orientation %.3f deg, centroid (%.2f, %.2f)"%((theta*180/np.pi), x_, y_))

        cov_Ixy = np.array([[mup_20, mup_11],[mup_11, mup_02]])
        cov_Ixy
        

        return moments, hu_moments, (theta, cov_Ixy)
     


    @classmethod
    def imshow(cls, img: NDArray, xbin_edges=None, ybin_edges=None):
        """ visualizes the image using matplotlib's imshow """
        
        img_shape = np.shape(img)
        
        if xbin_edges is None:
            xbin_edges = np.arange(img_shape[0]+1) # + 0.5
            
        if ybin_edges is None:
            ybin_edges = np.arange(img_shape[1]+1) #+ 0.5

        xmin, xmax, ymin, ymax = (xbin_edges[0], xbin_edges[-1], ybin_edges[0], ybin_edges[-1])
        y_first_extent = (ymin, ymax, xmin, xmax) # swapped the order of the extent axes.
        main_plot_kwargs = {
            'cmap': 'viridis',
            'origin':'lower',
            'extent':y_first_extent,
        }

        fig = plt.figure(layout="constrained")
        imv = plt.imshow(img, **main_plot_kwargs)
        
        # ax_dict = fig.subplot_mosaic(
        #     [
        #         ["ax_LONG_pf_tuning_curve", "ax_LONG_activity_v_time", "ax_SHORT_activity_v_time", "ax_SHORT_pf_tuning_curve"],
        #     ],
        #     # set the height ratios between the rows
        #     # height_ratios=[8, 1],
        #     # height_ratios=[1, 1],
        #     # set the width ratios between the columns
        #     width_ratios=[1, 8, 8, 1],
        #     sharey=True,
        #     gridspec_kw=dict(wspace=0, hspace=0.15) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
        # )
        return fig, imv
    
    @classmethod
    def plot_pyqtgraph(cls, img):
        """ plots a single image 
        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        app = pg.mkQApp("ImageView Example")

        ## Create window with ImageView widget
        win = QtWidgets.QMainWindow()
        win.resize(800,800)
        imv = pg.ImageView()
        win.setCentralWidget(imv)
        win.show()
        win.setWindowTitle('ComputerVisionComputations.plot_pyqtgraph: ImageView')

        # ## Create random 3D data set with time varying signals
        # dataRed = np.ones((100, 200, 200)) * np.linspace(90, 150, 100)[:, np.newaxis, np.newaxis]
        # dataRed += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100
        # dataGrn = np.ones((100, 200, 200)) * np.linspace(90, 180, 100)[:, np.newaxis, np.newaxis]
        # dataGrn += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100
        # dataBlu = np.ones((100, 200, 200)) * np.linspace(180, 90, 100)[:, np.newaxis, np.newaxis]
        # dataBlu += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100

        # data = np.concatenate(
        #     (dataRed[:, :, :, np.newaxis], dataGrn[:, :, :, np.newaxis], dataBlu[:, :, :, np.newaxis]), axis=3
        # )
        
        data = deepcopy(img)


        ## Display the data and assign each frame a time value from 1.0 to 3.0
        # imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))
        
        imv.setImage(data)

        # ## Set a custom color map
        # colors = [
        #     (0, 0, 0),
        #     (45, 5, 61),
        #     (84, 42, 55),
        #     (150, 87, 60),
        #     (208, 171, 141),
        #     (255, 255, 255)
        # ]
        # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # imv.setColorMap(cmap)

        # Start up with an ROI
        # imv.ui.roiBtn.setChecked(True)
        # imv.roiClicked()
        
        return app, win, imv