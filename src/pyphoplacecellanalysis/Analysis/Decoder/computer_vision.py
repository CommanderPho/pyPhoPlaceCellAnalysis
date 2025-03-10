from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

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
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, SingleEpochDecodedResult #typehinting only
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter

from copy import deepcopy
import numpy as np

# import cv2
from PIL import Image


# Custom Type Definitions ____________________________________________________________________________________________ #
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
    # Interactive                                                                                                          #
    # ==================================================================================================================== #
    

    @classmethod
    def interactive_image_preview(cls, input_img, blur_v_sigma, blur_h_sigma, hessian_sigma):
        from IPython.display import display
        
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
        from ipywidgets import interact, FloatSlider, IntSlider, fixed
        
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
        from ipywidgets import interact, FloatSlider, IntSlider, fixed
        
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
        


    # ==================================================================================================================== #
    # Binarizations                                                                                                        #
    # ==================================================================================================================== #

    @classmethod
    def smallest_non_zero_values_binarization(cls, img, non_included_index_value=0):
        """ Very inclusive
        
        """
        _original_img = None
        # original_dtype = img.dtype
        if np.issubdtype(img.dtype, np.integer):
            _original_img = deepcopy(img)
            ## convert to float
            img = (img / 255.0)
            

        is_non_zero = np.nonzero(img)
        min_non_zero_value: float = np.nanmin(img[is_non_zero]) # float or int
        smallest_nonzero_bw_mask = (img > min_non_zero_value)
            
        # Apply mask to create the masked image
        if _original_img is not None:
            masked_img = deepcopy(_original_img)
        else:
            masked_img = deepcopy(img)

        # masked_img[np.logical_not(bw_top_values_mask)] = non_included_index_value # 0.0
        masked_img[~smallest_nonzero_bw_mask] = non_included_index_value # 0.0
        
        return smallest_nonzero_bw_mask, masked_img
    

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
    def intensity_orientation(cls, img, moments, figsize=(15,8)):
        
        #m00, m10, m01, m11, m20, m02,
        center = np.array(img.shape[:2])/2
        
        # centroid
        x_, y_ =  moments["m10"]/(moments["m00"] + 1e-5), moments["m01"]/(moments["m00"] + 1e-5)
        
        # second order central moments
        mup_20 = moments["mu20"]/moments["m00"] - x_**2
        mup_02 = moments["mu02"]/moments["m00"] - y_**2
        mup_11 = moments["mu11"]/moments["m00"] - x_*y_

        # angle
        theta = np.arctan(2*mup_11/(mup_20 - mup_02))/2
        print("intensity orientation %.3f deg, centroid (%.2f, %.2f)"%((theta*180/np.pi), x_, y_))
        
        
        # intensity covariance
        cov_Ixy = np.array([[mup_20, mup_11],[mup_11, mup_02]])
        print("cov I(x,y)\n", cov_Ixy)
        
        # eigen vectors and values
        evals, evecs = np.linalg.eig(cov_Ixy)
        
        print("evals\n", evals)
        print("evecs\n", evecs)

        plt.figure(figsize=figsize)
        plt.imshow(img)

        for e, v in zip(evals, evecs):
            plt.plot([x_, np.sqrt(np.abs(e))*v[0]+x_], [y_, np.sqrt(np.abs(e))*v[1]+y_], 'r-', lw=2)

        plt.scatter(x_, y_, color="yellow")
        plt.annotate("I(x,y) centroid",[ x_+15, y_-10], color="yellow" )

        plt.xticks([0, x_, img.shape[1]], [0, int(x_), img.shape[1]])
        plt.yticks([0, y_, img.shape[0]], [0, int(y_), img.shape[0]])

        plt.grid()
        plt.show()
        

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
    












# ==================================================================================================================== #
# ComputerVisionPipeline                                                                                               #
# ==================================================================================================================== #


def remove_small_regions(img, min_size):
    from scipy.ndimage import label
    
    # Step 1: Label connected components
    labeled_img, num_features = label(img)

    # Step 2: Calculate the size of each connected component
    component_sizes = np.bincount(labeled_img.ravel())

    # Step 3: Create a mask for large components
    large_components = component_sizes >= min_size
    large_components_mask = large_components[labeled_img]

    # Step 4: Return the image with small regions removed
    filtered_img = img * large_components_mask

    return filtered_img

from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()


from skimage import data, img_as_float
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT


@define(slots=False, eq=False)
class ComputerVisionPipeline:
    """ 
    from pyphoplacecellanalysis.Analysis.Decoder.computer_vision import ComputerVisionPipeline
    """
    a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = field()
    
    @property
    def num_filter_epochs(self) -> int:
        """The num_filter_epochs: int  property."""
        return self.a_decoder_decoded_epochs_result.num_filter_epochs 


    def _get_data(self, active_epoch_idx: int):
        """ captures: decoder_ripple_filter_epochs_decoder_result_dict, 
        
        """
        # a_decoder_decoded_epochs_result.filter_epochs
        from pyphocorehelpers.plotting.media_output_helpers import img_data_to_greyscale
        
        a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = deepcopy(self.a_decoder_decoded_epochs_result)
        num_filter_epochs: int = a_decoder_decoded_epochs_result.num_filter_epochs
        # active_epoch_idx: int = 6 #28
        active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=active_epoch_idx)
        most_likely_position_indicies = deepcopy(active_captured_single_epoch_result.most_likely_position_indicies)
        most_likely_position_indicies = np.squeeze(most_likely_position_indicies)
        t_bin_centers = deepcopy(active_captured_single_epoch_result.time_bin_container.centers)
        t_bin_indicies = np.arange(len(np.squeeze(most_likely_position_indicies)))
        # most_likely_position_indicies
        p_x_given_n = deepcopy(active_captured_single_epoch_result.marginal_x.p_x_given_n)
        # p_x_given_n_image = active_captured_single_epoch_result.get_posterior_as_image(skip_img_normalization=False, export_grayscale=True)
        p_x_given_n_image = img_data_to_greyscale(p_x_given_n)

        # active_captured_single_epoch_result.epoch_info_tuple # EpochTuple(Index=28, start=971.8437469999772, stop=983.9541530000279, label='28', duration=12.110406000050716, lap_id=29, lap_dir=1, score=0.36769430044232587, velocity=1.6140523749028528, intercept=1805.019565924132, speed=1.6140523749028528, wcorr=-0.9152062701244238, P_decoder=0.6562437078530542, pearsonr=-0.7228173157676305, travel=0.0324318935144031, coverage=0.19298245614035087, jump=0.0005841121495327102, sequential_correlation=16228.563177472019, monotonicity_score=16228.563177472019, laplacian_smoothness=16228.563177472019, longest_sequence_length=22, longest_sequence_length_ratio=0.4583333333333333, direction_change_bin_ratio=0.19148936170212766, congruent_dir_bins_ratio=0.574468085106383, total_congruent_direction_change=257.92556950947574, total_variation=326.1999849678664, integral_second_derivative=7423.7044320722935, stddev_of_diff=8.368982188902695)

        return p_x_given_n, p_x_given_n_image

    @classmethod
    def _process_CV(cls, img):
        ## Find smallest non-zero value
        # p_x_given_n_image

        img = deepcopy(img)

        img_stage_outputs = []
        img_stage_outputs.append(img)

        smallest_nonzero_bw_mask, masked_img = ComputerVisionComputations.smallest_non_zero_values_binarization(img, non_included_index_value=0)
        # smallest_nonzero_bw_mask
        img_stage_outputs.append(smallest_nonzero_bw_mask)
        # masked_img

        bw_top_values_mask, masked_img = ComputerVisionComputations.top_N_values_binarization(img, top_n=3, non_included_index_value=0)
        # bw_top_values_mask
        # masked_img
        img_stage_outputs.append(bw_top_values_mask)

        # # Display the results
        # for col in range(img.shape[1]):
        #     print(f"Column {col}:")
        #     for idx in range(top_n):
        #         if not np.isnan(top_values[idx, col]):
        #             print(f"Index: {top_indices[idx, col]}, Value: {top_values[idx, col]}")
                    

        # Remove small regions
        # filtered_img = remove_small_regions(masked_img, min_size=3)
        filtered_img = remove_small_regions(bw_top_values_mask, min_size=5)
        # filtered_img
        img_stage_outputs.append(filtered_img)

        final_image = deepcopy(filtered_img)

        return final_image, img_stage_outputs


    def _run_all(self, mw, active_epoch_idx: int): # mw: CustomMatplotlibWidget
        # a_decoder_decoded_epochs_result.filter_epochs
        print(f'active_epoch_idx: {active_epoch_idx}')
        p_x_given_n, p_x_given_n_image = self._get_data(active_epoch_idx=active_epoch_idx)
        final_image, img_stage_outputs = self._process_CV(img=p_x_given_n_image)


        # ==================================================================================================================== #
        # SIFT                                                                                                                 #
        # ==================================================================================================================== #
        # descriptor_extractor = SIFT()

        # descriptor_extractor.detect_and_extract(p_x_given_n_image)
        # keypoints1 = descriptor_extractor.keypoints
        # descriptors1 = descriptor_extractor.descriptors

        # descriptor_extractor.detect_and_extract(img2)
        # keypoints2 = descriptor_extractor.keypoints
        # descriptors2 = descriptor_extractor.descriptors

        # descriptor_extractor.detect_and_extract(img3)
        # keypoints3 = descriptor_extractor.keypoints
        # descriptors3 = descriptor_extractor.descriptors

        # matches12 = match_descriptors(
        #     descriptors1, descriptors2, max_ratio=0.6, cross_check=True
        # )
        # matches13 = match_descriptors(
        #     descriptors1, descriptors3, max_ratio=0.6, cross_check=True
        # )
        # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

        # plt.gray()

        # plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12)
        # ax[0, 0].axis('off')
        # ax[0, 0].set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

        # plot_matches(ax[1, 0], img1, img3, keypoints1, keypoints3, matches13)
        # ax[1, 0].axis('off')
        # ax[1, 0].set_title(
        #     "Original Image vs. Transformed Image\n" "(all keypoints and matches)"
        # )

        # plot_matches(
        #     ax[0, 1], img1, img2, keypoints1, keypoints2, matches12[::15], only_matches=True
        # )
        # ax[0, 1].axis('off')
        # ax[0, 1].set_title(
        #     "Original Image vs. Flipped Image\n" "(subset of matches for visibility)"
        # )

        # plot_matches(
        #     ax[1, 1], img1, img3, keypoints1, keypoints3, matches13[::15], only_matches=True
        # )
        # ax[1, 1].axis('off')
        # ax[1, 1].set_title(
        #     "Original Image vs. Transformed Image\n" "(subset of matches for visibility)"
        # )

        # plt.tight_layout()
        # plt.show()


        # ==================================================================================================================== #
        # Ridge operators                                                                                                      #
        # ==================================================================================================================== #

        # def original(image, **kwargs):
        #     """Return the original image, ignoring any kwargs."""
        #     return image

        # image = deepcopy(p_x_given_n_image)
        # cmap = plt.cm.gray

        # plt.rcParams["axes.titlesize"] = "medium"
        # axes = plt.figure(figsize=(10, 4)).subplots(2, 9)
        # for i, black_ridges in enumerate([True, False]):
        #     for j, (func, sigmas) in enumerate(
        #         [
        #             (original, None),
        #             (meijering, [1]),
        #             (meijering, range(1, 5)),
        #             (sato, [1]),
        #             (sato, range(1, 5)),
        #             (frangi, [1]),
        #             (frangi, range(1, 5)),
        #             (hessian, [1]),
        #             (hessian, range(1, 5)),
        #         ]
        #     ):
        #         result = func(image, black_ridges=black_ridges, sigmas=sigmas)
        #         axes[i, j].imshow(result, cmap=cmap)
        #         if i == 0:
        #             title = func.__name__
        #             if sigmas:
        #                 title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
        #             axes[i, j].set_title(title)
        #         if j == 0:
        #             axes[i, j].set_ylabel(f'{black_ridges = }')
        #         axes[i, j].set_xticks([])
        #         axes[i, j].set_yticks([])

        # plt.tight_layout()
        # plt.show()


        # ==================================================================================================================== #
        # Plot                                                                                                                 #
        # ==================================================================================================================== #
        # a, b = detect_ridges(p_x_given_n_image, sigma=3.0)

        # plot_images(p_x_given_n_image, a, b)

        ## OUTPUT:
        num_imgs = len(img_stage_outputs)
        print(f'num_imgs: {num_imgs}')
        ax_dict = mw.plots['ax_dict']
        assert len(ax_dict) == num_imgs
        
        # display(widgets.HBox((p_x_given_n_image, final_image)), clear=True)
        # display(p_x_given_n_image, final_image, clear=True)
        # fig = mw.getFigure()
        # Clear all axes
        # for ax in fig.axes:
        #     ax.clear()
            
        # mw.draw()
        # axs = mw.axes
        # # Clear all axes
        # for ax in fig.axes:
        #     ax.clear()

        nrows, ncols = 1, num_imgs  # Adjust these based on your layout

        for i, an_img in enumerate(img_stage_outputs):
            # subplot = mw.getFigure().add_subplot(1, num_imgs, (i + 1))
            subplot = ax_dict[f"img_stage_outputs[{i}]"]
            # # Calculate the subplot position
            # position = (nrows, ncols, (i + 1))            
            # # Check if the subplot exists
            # # if fig.axes and len(fig.axes) > i:
            # if mw.axes and len(mw.axes) > i:
            #     subplot = mw.axes[i]
            #     subplot.clear()  # Clear existing subplot for new image
            # else:
            #     subplot = fig.add_subplot(*position)  # Create a new subplot
            # print(f"img_stage_outputs[{i}]: an_img: {an_img}\n {subplot}")
            # subplot.clear()  # Clear existing subplot for new image
            
            subplot.imshow(an_img)
            
            # subplot = mw.getFigure().add_subplot(122)
            # subplot.imshow(p_x_given_n_image)

        # subplot.plot(x,y)
        # mw.setWindowTitle(f'ComputerVisionPipeline: active_epoch_idx: {active_epoch_idx}')
        mw.update()
        mw.draw()

        # display(mw.getFigure())
        # mw.show()        
        
