# ==================================================================================================================== #
# 2024-05-27 - WCorr Shuffle Stuff                                                                                     #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from typing import NewType

import attrs
from attrs import asdict, define, field, Factory, astuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from attrs import define, field, asdict, evolve
import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from neuropy.core.epoch import EpochHelpers, ensure_dataframe, find_data_indicies_from_epoch_times
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior, determine_event_interval_identity, determine_event_interval_is_included # numba acceleration
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor

from neuropy.utils.misc import build_shuffled_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from neuropy.utils.misc import split_array

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import compute_weighted_correlations
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df


"""

## ⚓ Decoded 2D Posterior Specificity - Metrics using the promenence mask

## Focality/Diffusivity: Number of bins in in the 90% promenence mask over the total number of bins --> [0.0, 1.0]
    ## definitionally the 90% promenance mask bins must be together/contiguous spatially, as outliers are considered different peaks.

## Sharpness/Peakiness: Number of bins in the 90% promenence mask over the number of bins exceeding 90% of the promenence peak height -- specifically looks at the area of the mean peak compared to the off-peak non-contiguous areas of similar heights

## Modality: The count of detected peaks exceeding a certain promenence -- e.g. 1 if unimodal, 2 if bimodal, ..., N if multi-modal. 



## Decoded Epoch Temporal Sequentiality - detecting spatial sweeps in subsequent time bins
    ## partially dpeends on time bin sizes -- of interest -- a real effect should remain at lower resolution/temporal subdivisions while fake ones can get washed out

## ❌ Sequentiality - dilate the 80% promenence mask of the top peak (? same as using a lower mask percentage??) and compute the mask overlap with the subsequent time bin.
    # 0 indicates disjoint/discontiguous... but it could be a jumpy sequence!!

## compute the 2D change vector between subsequent peak locations (either bin-space or cm)
    ## real trajectories should be roughly aligned and have constrained changes in direction (e.g. momentum)





"""

# ==================================================================================================================== #
# 2024-05-24 - Shuffling to show wcorr exceeds shuffles                                                                #
# ==================================================================================================================== #
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType
import nptyping as ND
from nptyping import NDArray
from scipy.interpolate import interp1d

import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration



@define(slots=False, repr=False, eq=False)
class DecodingLocalityMeasures(ComputedResult): #PickleSerializableMixin, AttrsBasedClassHelperMixin):
    """ Handles computing information about how the current decoded position relates to the present position (e.g. is it local, non-local, etc)
    
    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import DecodingLocalityMeasures, PredictiveDecoding
        
        decoding_locality_measures: DecodingLocalityMeasures = DecodingLocalityMeasures.init_from_decode_result(curr_active_pipeline=curr_active_pipeline, directional_decoders_decode_result=directional_decoders_decode_result, extant_decoded_time_bin_size=0.25)
        decoding_locality_measures.compute() ## ~30 seconds
        decoding_locality_measures

    """
    _VersionedResultMixin_version: str = "2025.12.15_0" # to be updated in your IMPLEMENTOR to indicate its version

    time_window_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = serialized_field()
    pos_df: pd.DataFrame = serialized_field()
    xbin: NDArray = serialized_field()
    ybin: NDArray = serialized_field()
    xbin_centers: NDArray = serialized_field()
    ybin_centers: NDArray = serialized_field()
    p_x_given_n: NDArray[ND.Shape["N_X_BINS, N_Y_BINS, 2, N_TIME_BINS"], np.floating] = serialized_field()
    epoch_names: List[str] = serialized_field(default=Factory(list))
    _interpolator: interp1d = non_serialized_field(default=None, is_computable=False)
    paradigm_epochs_df: pd.DataFrame = serialized_field()
    defer_compute_on_init: bool = non_serialized_field(default=False, is_computable=False)

    ## computed
    new_positions: NDArray[ND.Shape["N_TIME_BINS, 2"], np.floating] = serialized_field()
    
    ## one of these for each context (e.g. maze1, maze2 or ['roam', 'sprinkle'], etc
    sigma: Optional[float] = serialized_attribute_field(default=None, is_computable=True)
    gaussian_volume: NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TIME_BINS"], np.floating] = serialized_field(default=None, is_computable=True)

    p_x_given_n_dict: Dict[str, NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TIME_BINS"], np.floating]] = serialized_field(default=None, is_computable=True)

    ## locality comparisons:
    # decoding_meas_pos_locality_measure_dict: Dict[str, NDArray[ND.Shape["N_TIME_BINS"], np.floating]] = serialized_field(default=Factory(dict))
    # Generic dict to store all computed measures - allows easy extension without adding new fields
    locality_measures_dict_dict: Dict[str, Dict[str, Any]] = serialized_field(default=Factory(dict), is_computable=True)
    
    debugging_dict_dict: Dict[str, Dict[str, Any]] = serialized_field(default=Factory(dict), is_computable=True)

    
    locality_measures_df: pd.DataFrame = serialized_field(default=None, is_computable=True, init=False)

    non_local_PBE_non_moving_epochs_df: pd.DataFrame = serialized_field(default=None, is_computable=True)
    

    @property
    def n_total_pos_bins(self) -> int:
        """The n_total_pos_bins property."""
        return int(len(self.xbin_centers) * len(self.ybin_centers))



    def __attrs_post_init__(self):
        # Add post-init logic here
        if self.defer_compute_on_init:
            print('DecodingLocalityMeasures.__attrs_post_init__(...): init will not be performed because `self.defer_compute_on_init == True`.')
        else:
            self.perform_compute_on_load()

        print(f'done.')


    @classmethod
    def init_from_decode_result(cls, curr_active_pipeline, directional_decoders_decode_result, extant_decoded_time_bin_size: float = 0.25, sigma: Optional[float] = None) -> "DecodingLocalityMeasures":
        """ 
        _obj: PredictiveDecoding = DecodingLocalityMeasures.init_from_decode_result(
        """
        a_result_decoded = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]
        # a_result_decoded.p_x_given_n # .shape (41, 63, 2, 103948) - (n_x_bins, n_y_bins, n_tasks, n_time_bins) 
        
        time_window_centers = deepcopy(a_result_decoded.time_bin_container.centers)
        pos_df = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        # pos_df

        # axis=0 interpolates along rows (time) for all columns ('x' and 'y')
        # fill_value="extrapolate" allows sampling outside original time range
        interpolator = interp1d(pos_df['t'], pos_df[['x', 'y']], kind='linear', axis=0, fill_value="extrapolate")

        # Returns shape new_positions .shape: (n_target_times, 2)
        new_positions = interpolator(time_window_centers)
        # new_positions
        p_x_given_n = deepcopy(a_result_decoded.p_x_given_n)
        epoch_names: List[str] = list(directional_decoders_decode_result.pf1D_Decoder_dict.keys())
        # decoding_locality_measures.epoch_names
        paradigm_epochs_df = deepcopy(curr_active_pipeline.sess.epochs.to_dataframe())
        paradigm_epochs_df = paradigm_epochs_df.epochs.label_slice(epoch_names)

        _obj = cls(time_window_centers=time_window_centers, pos_df=deepcopy(pos_df),
                   xbin=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.xbin), ybin=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.ybin),
                   xbin_centers=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.xbin_centers), ybin_centers=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.ybin_centers),
                   new_positions=new_positions, interpolator=interpolator, p_x_given_n=deepcopy(p_x_given_n),
                   paradigm_epochs_df=paradigm_epochs_df, epoch_names=epoch_names,
                   sigma=sigma)
        return _obj


    @classmethod
    def perform_build_normalized_outputs(cls, p_x_given_n, epoch_names: List[str]):
        """ Normalize: self.p_x_given_n_dict and self.moving_avg over the decoer time period ('sprinkle', 'roam')

        """
        def _subfn_renormalize_marginal(a_context_included_pdf):
            # np.shape(a_moving_avg)
            ## renormalize over context:
            norm_sums = np.nansum(a_context_included_pdf, axis=(0, 1))
            is_nonzero = np.nonzero(norm_sums)
            for a_nonzero_idx in is_nonzero:
                a_context_included_pdf[:, :, a_nonzero_idx] = a_context_included_pdf[:, :, a_nonzero_idx] / norm_sums[a_nonzero_idx]
            return a_context_included_pdf


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        ## INPUTS: quantities to renormalize
        p_x_given_n_dict = {}

        # for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):
        for an_epoch_idx, an_epoch_name in enumerate(epoch_names):
            ## "epoch" in the loop variables refers to only the session.paradigm epochs, like ['roam', 'sprinkle']

            a_p_x_given_n = deepcopy(np.squeeze(p_x_given_n[:, :, an_epoch_idx, :]))
            a_p_x_given_n = _subfn_renormalize_marginal(a_p_x_given_n)
            p_x_given_n_dict[an_epoch_name] = a_p_x_given_n

        ## END for an_epoch_idx, an_epoch_n...


        ## OUTPUTS: _a_moving_avg_dict, _a_moving_avg_meas_pos_overlap_dict
        return p_x_given_n_dict
    

    @function_attributes(short_name=None, tags=['normalization', 'locality', 'overlap'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-11 17:03', related_items=[])
    def build_normalized_outputs(self):
        """ Normalize: self.p_x_given_n_dict and self.moving_avg over the decoer time period ('sprinkle', 'roam')

        Normalize and convolve each new_position 2D point (x, y) with a fixed width 2D gaussian
        
        Updates: self.
            .moving_avg_dict, .moving_avg_meas_pos_overlap_dict, .gaussian_volume, .decoding_meas_pos_locality_measure_dict
        """
        self.p_x_given_n_dict = self.perform_build_normalized_outputs(p_x_given_n=self.p_x_given_n, epoch_names=self.epoch_names)

    

    @function_attributes(short_name=None, tags=['normalization', 'locality', 'overlap'], input_requires=[], output_provides=[], uses=['self.compute_locality_measures'], used_by=[], creation_date='2025-12-11 17:03', related_items=[])
    def compute(self):
        """ Normalize: self.p_x_given_n_dict and self.moving_avg over the decoer time period ('sprinkle', 'roam')

        Normalize and convolve each new_position 2D point (x, y) with a fixed width 2D gaussian
        
        Updates: self.
            .moving_avg_dict, .moving_avg_meas_pos_overlap_dict, .gaussian_volume, .decoding_meas_pos_locality_measure_dict
        """
        # Ensure initial computations are done (idempotent - only computes if needed)
        self.perform_compute_on_load()
        locality_measures_df = self.compute_locality_measures()
        
        ## OUTPUTS: _a_moving_avg_dict, _a_moving_avg_meas_pos_overlap_dict
        return locality_measures_df
    


    def _build_sampled_pos_with_gaussian_spread(self, sigma: Optional[float] = None):
        """ Computed for each position in `self.new_positions`
        
        gaussian_volume = _obj._build_sampled_pos_with_gaussian_spread(sigma=1.0)
        np.shape(gaussian_volume) # (42, 64, 103948)
        
        Args:
            sigma: Optional sigma value. If None, uses self.sigma (must be set)
        """
        # Use self.sigma if sigma not provided
        if sigma is None:
            if self.sigma is None:
                raise ValueError("sigma must be provided either as argument or set on self.sigma")
            sigma = self.sigma
        
        # 1. Setup the Grid
        # Ensure x_bounds/y_bounds match the physical extent of _obj.moving_avg
        # Example: x_bounds = (0, 100), y_bounds = (0, 150)
        x = deepcopy(self.xbin_centers) # np.linspace(x_bounds[0], x_bounds[1], n_x_bins) 
        y = deepcopy(self.ybin_centers) # np.linspace(y_bounds[0], y_bounds[1], n_y_bins)
        X, Y = np.meshgrid(x, y, indexing='ij')  # Shape: (41, 63)

        # np.shape(X)
        # np.shape(Y)
        # np.shape(x)

        # 2. Prepare for Broadcasting
        # Grid shape: (41, 63, 1, 2) 
        # We add a dimension at index 2 to broadcast against time
        grid_stack = np.stack([X, Y], axis=-1)[:, :, np.newaxis, :]

        # Position shape: (1, 1, n_target_times, 2)
        # We add dimensions at indices 0 and 1 to broadcast against the grid
        pos_stack = self.new_positions[np.newaxis, np.newaxis, :, :]

        # 3. Calculate Gaussian (Vectorized)
        # The subtraction broadcasts to shape (41, 63, n_target_times, 2)
        # Summing over the last axis (coordinates) gives squared distance
        dist_sq = np.sum((grid_stack - pos_stack)**2, axis=-1)

        # Apply Gaussian function
        # sigma must be in the same physical units as the bounds/positions
        gaussian_volume = np.exp(-dist_sq / (2 * sigma**2))

        # Result: gaussian_volume.shape is (41, 63, n_target_times)
        return gaussian_volume


    def perform_compute_on_load(self):
        """ called by `__attrs_post_init__` to build initial properties if they're missing
        Builds: .gaussian_volume, .p_x_given_n_dict, .sigma
        
        """
        if self.sigma is None:
            x_step: float = np.nanmean(np.diff(self.xbin))
            y_step: float = np.nanmean(np.diff(self.ybin))

            self.sigma = np.nanmax([x_step, y_step]) * 5.0
            print(f'sigma: {self.sigma}')
            
        print(f'building sampled and normalized outputs...')
        
        if (self.gaussian_volume is None):
            self.gaussian_volume = self._build_sampled_pos_with_gaussian_spread()

        if (self.p_x_given_n_dict is None) or (len(self.p_x_given_n_dict) == 0):
            self.build_normalized_outputs()
            
        if (self.xbin is not None) and (self.ybin is not None):
            self.pos_df = self.pos_df.dropna(how='any', subset=['t', 'x', 'y'])
            self.pos_df = self.pos_df.position.adding_binned_position_columns(xbin_edges=self.xbin, ybin_edges=self.ybin)
            self.pos_df = self.pos_df[(self.pos_df['binned_x'].notna()) & (self.pos_df['binned_y'].notna())] # Filter rows based on columns: 'binned_x', 'binned_y'

        # self.pos_df = self.pos_df.rename(columns={'x':'x_meas', 'y':'y_meas', 'binned_x':'binned_x_meas', 'binned_y':'binned_y_meas'}, inplace=False).drop(columns=['t'], inplace=False) ## measured



    # ==================================================================================================================================================================================================================================================================================== #
    # Locality Computations                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['compute', 'locality', 'static'], input_requires=[], output_provides=[], uses=[], used_by=['.compute_locality_measures'], creation_date='2025-12-23 21:00', related_items=[])
    def compute_locality_measures_for_posterior(cls, a_p_x_given_n: NDArray, xbin_centers: NDArray, ybin_centers: NDArray, gaussian_volume: NDArray=None, n_total_pos_bins: Optional[int] = None, min_val_epsilon: float = 1e-9, alpha_list = [0.8], enable_debug_outputs: bool = True, earthmovers_fn: Optional[Callable] = None, debug_print: bool=True) -> Dict[str, Any]:
        """Computes all locality measures for a given posterior probability distribution.
        
        This is a completely independent classmethod that can be called without an instance.
        
        Args:
            a_p_x_given_n: NDArray with shape (N_X_BINS, N_Y_BINS, N_TIME_BINS) - the posterior probability distribution
            gaussian_volume: NDArray with shape (N_X_BINS, N_Y_BINS, N_TIME_BINS) - the gaussian spread volume
            xbin_centers: NDArray - x-axis bin centers
            ybin_centers: NDArray - y-axis bin centers
            n_total_pos_bins: Optional[int] - total number of position bins (computed from xbin_centers/ybin_centers if None)
            min_val_epsilon: float - minimum value threshold (default: 1e-9)
            enable_debug_outputs: bool - whether to compute and return debug outputs (default: True)
            earthmovers_fn: Optional[Callable] - optional function for computing earthmovers distance (default: None)
            
        Returns:
            Dict[str, Any] containing:
                - 'mask_overlap': NDArray
                - 'peak_prom': NDArray
                - 'peak_prom_num_bins': NDArray
                - 'peak_prom_Focality': NDArray
                - 'peak_prom_Peakiness': NDArray
                - 'peak_prom_num_peaks': NDArray (if computation succeeds)
                - 'dist_to_highest_peak': NDArray
                - 'earthmovers': NDArray (only if earthmovers_fn is provided)
                - 'debug': Dict[str, Any] (only if enable_debug_outputs=True)
        """
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence
        
        def safe_nanmax(arr):
            try:
                if arr.size == 0:
                    return np.nan
                return np.nanmax(arr)
            except (ValueError, AttributeError):
                return np.nan
        
        def _subfn_pdf_spatial_distances(gaussian_volume, a_p_x_given_n, xbin_centers, ybin_centers):
            """
            Computes the Euclidean distance between the expected positions (COM) of 
            two 2D probability distributions using vectorized weighted averages.
            """
            # 1. Get Shapes
            # Assuming shape is (Rows/H, Cols/W, Time)
            pdf_obj = gaussian_volume
            pdf_cmp = a_p_x_given_n
            
            # 2. Calculate Marginals to simplify COM calculation
            # Sum over columns (axis 1) to get mass distribution along rows (Height/x_bins)
            # Shape becomes (H, T)
            marg_x_obj = np.sum(pdf_obj, axis=1)
            marg_x_cmp = np.sum(pdf_cmp, axis=1)

            # Sum over rows (axis 0) to get mass distribution along columns (Width/y_bins)
            # Shape becomes (W, T)
            marg_y_obj = np.sum(pdf_obj, axis=0)
            marg_y_cmp = np.sum(pdf_cmp, axis=0)

            # 3. Compute Expected Position (Weighted Average of Bin Centers)
            # Formula: Sum(Probability * Value) / Sum(Probability)
            
            # reshape centers for broadcasting: (H, 1) * (H, T) -> sum -> (T,)
            denom_x_obj = np.sum(marg_x_obj, axis=0)
            # Handle division by zero if a timebin has all-zeros
            denom_x_obj[denom_x_obj == 0] = 1.0 
            
            x_obj = np.sum(marg_x_obj * xbin_centers[:, np.newaxis], axis=0) / denom_x_obj
            
            # Repeat for Comparison Object
            denom_x_cmp = np.sum(marg_x_cmp, axis=0)
            denom_x_cmp[denom_x_cmp == 0] = 1.0
            x_cmp = np.sum(marg_x_cmp * xbin_centers[:, np.newaxis], axis=0) / denom_x_cmp

            # Repeat for Y (Width)
            denom_y_obj = np.sum(marg_y_obj, axis=0)
            denom_y_obj[denom_y_obj == 0] = 1.0
            y_obj = np.sum(marg_y_obj * ybin_centers[:, np.newaxis], axis=0) / denom_y_obj

            denom_y_cmp = np.sum(marg_y_cmp, axis=0)
            denom_y_cmp[denom_y_cmp == 0] = 1.0
            y_cmp = np.sum(marg_y_cmp * ybin_centers[:, np.newaxis], axis=0) / denom_y_cmp

            # 4. Euclidean Distance
            distances_spatial = np.sqrt((x_obj - x_cmp)**2 + (y_obj - y_cmp)**2)

            # 5. Max distance (Diagonal of the ROI)
            max_possible_distance = np.sqrt(np.ptp(xbin_centers)**2 + np.ptp(ybin_centers)**2)
            distances_spatial_frac_max = distances_spatial / max_possible_distance
            
            return distances_spatial, distances_spatial_frac_max
        
        # Compute n_total_pos_bins if not provided
        if n_total_pos_bins is None:
            assert (xbin_centers is not None)
            assert (ybin_centers is not None)
            n_total_pos_bins = int(len(xbin_centers) * len(ybin_centers))
        
        # Initialize result dictionary
        result_dict: Dict[str, Any] = {}
        debug_dict: Dict[str, Any] = {}
        
        # ==================================================================================================================================================================================================================================================================================== #
        # do all computation measures                                                                                                                                                                                                                                                          #
        # ==================================================================================================================================================================================================================================================================================== #

        ## do all computation measures
        if gaussian_volume is not None:
            a_computation_measure_name: str = 'mask_overlap'
            if debug_print:
                print(f'\tcomputing: "{a_computation_measure_name}"...')
            ## above a certain promence ideally:
            ## Oh dang this is kinda tiny
            is_high_prob_mask = (a_p_x_given_n > min_val_epsilon)
            if enable_debug_outputs:
                debug_dict['mask_overlap_masks'] = is_high_prob_mask            

            result_dict[a_computation_measure_name] = ((gaussian_volume * is_high_prob_mask) > min_val_epsilon).astype(int) ## the "overlap" is computed by taking the elementwise dot-product with the moving average



        a_computation_measure_name: str = 'peak_prom'
        if debug_print:
            print(f'\tcomputing: "{a_computation_measure_name}"...')
        ## above a certain promence ideally:
        # alpha: float = 0.8 # above 85% of the peak height of the centeral peak
        # alpha_list = [0.5, 0.8]
        
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list) # (103948, 1, 41, 63)
        epoch_masks_dict = dict(zip(alpha_list, epoch_masks_list))            
        a_high_alpha: float = alpha_list[-1]
        an_alpha_epoch_masks: NDArray = epoch_masks_dict[a_high_alpha] ## get the high mask
        # an_alpha_epoch_masks = np.stack(an_alpha_epoch_masks, axis=-1) # (5, 41, 63) - (n_x_bins, n_y_bins, n_t_bins)
        assert np.shape(an_alpha_epoch_masks) == np.shape(a_p_x_given_n)
        
        if enable_debug_outputs:
            debug_dict['peak_prom_promenence_tuples'] = epoch_promenence_tuples
            # debug_dict['peak_prom_masks'] = an_alpha_epoch_masks
            debug_dict['peak_prom_masks_dict'] = epoch_masks_dict
            # debug_dict['peak_prom_masks_dict'] = epoch_masks_dict
            # debug_dict['peak_prom_product'] = epoch_masks
            
        try:
            all_t_bin_peak_heights: NDArray = np.array([safe_nanmax(peak_heights) for (peak_coords, prominences, peak_heights) in epoch_promenence_tuples])
            all_t_bin_peak_prominences: NDArray = np.array([safe_nanmax(prominences) for (peak_coords, prominences, peak_heights) in epoch_promenence_tuples])
            if enable_debug_outputs:
                debug_dict['all_t_bin_peak_heights'] = all_t_bin_peak_heights
                debug_dict['all_t_bin_peak_prominences'] = all_t_bin_peak_prominences

        except (ValueError, AttributeError) as e:
            print(f'error computing `all_t_bin_peak_heights`: {e}. skipping.')
            all_t_bin_peak_heights = None
            all_t_bin_peak_prominences = None
        except Exception as e:
            raise e


        if gaussian_volume is not None:
            result_dict[a_computation_measure_name] = ((gaussian_volume * an_alpha_epoch_masks) > min_val_epsilon).astype(int) ## the "overlap" is computed by taking the elementwise dot-product with the moving average
            


        # result_dict[f"{a_computation_measure_name}_score"] = [(np.nansum(np.stack(an_epoch_mask, axis=-1), axis=(0, 1))/n_total_pos_bins) for an_epoch_idx, an_epoch_mask in enumerate(all_epochs_masks)]
        result_dict[f"{a_computation_measure_name}_num_bins"] = np.nansum(an_alpha_epoch_masks, axis=(0, 1))
        # result_dict[f"{a_computation_measure_name}_score"] = (np.nansum(an_alpha_epoch_masks, axis=(0, 1))/float(n_total_pos_bins))

        ## ⚓ Decoded 2D Posterior Specificity - Metrics using the promenence mask

        ## Focality/Diffusivity: Number of bins in in the 90% promenence mask over the total number of bins --> [0.0, 1.0]
            ## definitionally the 90% promenance mask bins must be together/contiguous spatially, as outliers are considered different peaks.
        result_dict[f"{a_computation_measure_name}_Focality"] = (np.nansum(an_alpha_epoch_masks, axis=(0, 1))/float(n_total_pos_bins))

        ## Sharpness/Peakiness: Number of bins in the 90% promenence mask over the number of bins exceeding 90% of the promenence peak height -- specifically looks at the area of the mean peak compared to the off-peak non-contiguous areas of similar heights
        # result_dict[f"{a_computation_measure_name}_Peakiness"] = np.nansum(an_alpha_epoch_masks, axis=(0, 1)) / np.array([np.nansum((a_p_x_given_n[:, :, i] >= (a_peak_height * alpha_list[-1])), axis=(0, 1)) for i, a_peak_height in enumerate(all_t_bin_peak_heights)])
        if all_t_bin_peak_heights is not None:
            result_dict[f"{a_computation_measure_name}_Peakiness"] = np.nansum(an_alpha_epoch_masks, axis=(0, 1)) / np.array([np.nansum((a_p_x_given_n[:, :, i] >= (a_peak_height * alpha_list[-1])), axis=(0, 1)) if not np.isnan(a_peak_height) else np.nan for i, a_peak_height in enumerate(all_t_bin_peak_heights)])
        else:
            result_dict[f"{a_computation_measure_name}_Peakiness"] = np.full(a_p_x_given_n.shape[-1], np.nan)

        try:
            ## Modality: The count of detected peaks exceeding a certain promenence -- e.g. 1 if unimodal, 2 if bimodal, ..., N if multi-modal. 
            all_t_bin_num_peaks: NDArray = np.array([len(peak_heights) for (peak_coords, prominences, peak_heights) in epoch_promenence_tuples])

            result_dict[f"{a_computation_measure_name}_num_peaks"] = all_t_bin_num_peaks

        except (ValueError, AttributeError) as e:
            print(f'error computing `all_t_bin_num_peaks`: {e}. skipping.')
        except Exception as e:
            raise e
        

        if (gaussian_volume is not None) and (xbin_centers is not None) and (ybin_centers is not None):
            a_computation_measure_name: str = 'dist_to_highest_peak'
            if debug_print:
                print(f'\tcomputing: "{a_computation_measure_name}"...')
            ## above a certain promence ideally:
            # peak_locations = np.argmax(a_p_x_given_n, axis=(0, 1))
            distances_spatial, distances_spatial_frac_max = _subfn_pdf_spatial_distances(gaussian_volume=gaussian_volume, a_p_x_given_n=a_p_x_given_n, xbin_centers=xbin_centers, ybin_centers=ybin_centers)
            result_dict[a_computation_measure_name] = distances_spatial_frac_max ## the "overlap" is computed by taking the elementwise dot-product with the moving average
        

        # a_computation_measure_name: str = 'dist_to_nearest_peak'
        # print(f'\tcomputing: "{a_computation_measure_name}"...')
        # ## above a certain promence ideally:
        # min_val_epsilon: float = 1e-9
        # is_high_prob_mask = (a_p_x_given_n > min_val_epsilon)
        # result_dict[a_computation_measure_name] = ((gaussian_volume * is_high_prob_mask) > min_val_epsilon).astype(int) ## the "overlap" is computed by taking the elementwise dot-product with the moving average


        if earthmovers_fn is not None:
            a_computation_measure_name: str = 'earthmovers'
            if debug_print:
                print(f'\tcomputing: "{a_computation_measure_name}"...')
            result_dict[a_computation_measure_name] = earthmovers_fn(gaussian_volume, a_p_x_given_n)

        # Add debug dict to result if enabled
        if enable_debug_outputs:
            result_dict['debug'] = debug_dict
        
        return result_dict




    @function_attributes(short_name=None, tags=['compute', 'MAIN', 'locality'], input_requires=[], output_provides=[], uses=['.compute_locality_measures_for_posterior', '.rebuild_locality_measures_df'], used_by=['.compute'], creation_date='2025-12-15 06:54', related_items=[])
    def compute_locality_measures(self, min_val_epsilon: float = 1e-9, enable_debug_outputs: bool=True):
        """ computes all required locality measures

        Normalize and convolve each new_position 2D point (x, y) with a fixed width 2D gaussian
        
        
        """
        # active_subfn_compute_earthmovers_fn = _subfn_calculate_spatial_emd # #TODO 2025-12-11 17:53: - [ ] TOO SLOW
        # active_subfn_compute_earthmovers_fn = _subfn_calculate_sinkhorn_distance
        # active_subfn_compute_earthmovers_fn = _subfn_calculate_sliced_wasserstein_correct
        # active_subfn_compute_earthmovers_fn = _subfn_calculate_spatial_emd_fast
        active_subfn_compute_earthmovers_fn = None

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        if self.gaussian_volume is None:
            self.gaussian_volume = self._build_sampled_pos_with_gaussian_spread()
            
        if (self.p_x_given_n_dict is None) or (len(self.p_x_given_n_dict) == 0):
            _out = self.build_normalized_outputs()

        ## INPUTS: gaussian_volume
        self.locality_measures_dict_dict = {}
        self.moving_avg_meas_pos_overlap_dict = {}
        self.debugging_dict_dict = {}
        
        # for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):
        for an_epoch_idx, an_epoch_name in enumerate(self.epoch_names):
            ## "epoch" in the loop variables refers to only the session.paradigm epochs, like ['roam', 'sprinkle']
            self.locality_measures_dict_dict[an_epoch_name] = {} ## empty
            self.debugging_dict_dict[an_epoch_name] = {}
            
            a_p_x_given_n = self.p_x_given_n_dict[an_epoch_name]

            ## compute the locality:
            num_timestamps: int = np.shape(self.gaussian_volume)[-1]
            
            # Final correct POM __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #


            #TODO 2025-12-23 20:55: - [ ] Found that everything seems to be working well except that there are sometimes a few time bins out of an epoch that have poorly localized posteriors in general (they look very diffuse and like an error, maybe low firing bins)
            ### These need to be filtered out (either by diffusivity of low-firing criteria) so that when we collapse over all the time bins within each epoch we don't pick up a bunch of garbage (the diffuse bins are too liberal).
            # masked_result, mask_index_tuple = decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(
            #     spikes_df=spikes_df,
            #     min_num_spikes_per_bin_to_be_considered_active=5,
            #     min_num_unique_active_neurons_per_time_bin=1,
            #     masked_bin_fill_mode='ignore'
            # )
                        

            # self.decoding_meas_pos_locality_measure_dict[an_epoch_name] = np.array([_subfn_calculate_spatial_emd(self.gaussian_volume[:, :, a_timestamp_idx], a_p_x_given_n[:, :, a_timestamp_idx]) for a_timestamp_idx in np.arange(num_timestamps)])

            # ==================================================================================================================================================================================================================================================================================== #
            # do all computation measures                                                                                                                                                                                                                                                          #
            # ==================================================================================================================================================================================================================================================================================== #

            ## Call the independent classmethod to compute all locality measures
            computation_results = self.__class__.compute_locality_measures_for_posterior(
                a_p_x_given_n=a_p_x_given_n,
                gaussian_volume=self.gaussian_volume,
                xbin_centers=self.xbin_centers,
                ybin_centers=self.ybin_centers,
                n_total_pos_bins=self.n_total_pos_bins,
                min_val_epsilon=min_val_epsilon,
                enable_debug_outputs=enable_debug_outputs,
                earthmovers_fn=active_subfn_compute_earthmovers_fn
            )

            # Extract results into self.locality_measures_dict_dict[an_epoch_name]
            for key, value in computation_results.items():
                if key != 'debug':
                    self.locality_measures_dict_dict[an_epoch_name][key] = value

            # Extract debug outputs if enabled
            if enable_debug_outputs and 'debug' in computation_results:
                self.debugging_dict_dict[an_epoch_name].update(computation_results['debug'])


        ## END for an_epoch_idx, an_epoch_n...
        print(f'done with compute.')


        # ==================================================================================================================================================================================================================================================================================== #
        # Phase II - processing the output dataframe                                                                                                                                                                                                                                           #
        # ==================================================================================================================================================================================================================================================================================== #
        locality_measures_df = self.rebuild_locality_measures_df()
        return locality_measures_df
    

    @classmethod
    def perform_build_locality_measures_df(cls, locality_measures_dict_dict: Dict[str, Dict], time_window_centers: NDArray, paradigm_epochs_df: Optional[pd.DataFrame]=None, xbin_centers=None, ybin_centers=None):
        """ builds the measures_df
        """
        # _out_locality_measures_df = pd.DataFrame(self.decoding_meas_pos_locality_measure_dict)
        locality_measures_df: pd.DataFrame = pd.DataFrame(time_window_centers, columns=['t'])
        # _out_locality_measures_df['t'] = self.time_window_centers

        for an_epoch_name, v in locality_measures_dict_dict.items():
            for a_computation_measure_name, vv in v.items():
                if (a_computation_measure_name == 'mask_overlap'):
                    assert xbin_centers is not None
                    assert ybin_centers is not None
                    total_num_possible_bins: int = len(xbin_centers) * len(ybin_centers)
                    vv = np.nansum(vv, (0, 1)) / total_num_possible_bins
                if a_computation_measure_name == 'peak_prom':                    
                    continue ## skip
                
                locality_measures_df[f"{a_computation_measure_name}_{an_epoch_name}"] = vv # _obj.locality_measures_dict_dict[an_epoch_name][a_computation_measure_name]



        if (paradigm_epochs_df is not None):
            # _out_locality_measures_df: pd.DataFrame = deepcopy(_obj.locality_measures_df)
            # _out_locality_measures_df
            ## #TODO 2025-12-12 18:39: - [ ] Manually coded times for epochs ['roam', 'sprinkle'] -- fix setting proper epoch

            ## - [ ] add the correct maze_id to know which maze decoder to use. Adds 'correct_paradigm_epoch' columns
            locality_measures_df = locality_measures_df.time_point_event.adding_epochs_identity_column(epochs_df=paradigm_epochs_df, epoch_id_key_name='correct_paradigm_epoch', epoch_label_column_name='label', override_time_variable_name='t',
                                                                no_interval_fill_value='', should_replace_existing_column=True, drop_non_epoch_events=False, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
            
            locality_measures_df['is_non_local_period'] = False

            # an_epoch_name: str = 'sprinkle'
            for an_epoch_name, v in locality_measures_dict_dict.items():
                is_epoch_idx = (locality_measures_df['correct_paradigm_epoch'] == an_epoch_name)
                locality_measures_df.loc[is_epoch_idx, 'is_non_local_period'] =  np.logical_and((locality_measures_df[f'dist_to_highest_peak_{an_epoch_name}'][is_epoch_idx] >= 0.4), (locality_measures_df[f'mask_overlap_{an_epoch_name}'][is_epoch_idx] < 0.1))

                # _out_locality_measures_df.loc[is_sprinkle, 'is_non_local_period'] =  (_out_locality_measures_df[f'dist_to_highest_peak_{an_epoch_name}'][is_sprinkle] >= 0.4)

        return locality_measures_df



    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=[], used_by=['.compute_locality_measures'], creation_date='2025-12-15 07:01', related_items=[])
    def rebuild_locality_measures_df(self):
        """ called to rebuild the final output df
        Updates: self.locality_measures_df,
        
        ## Adds ['correct_paradigm_epoch', 'is_non_local_period', 'correct_paradigm_epoch'] columns
        
            ## show just hhe non-locqal periods
            non_local_locality_measures_df: pd.DataFrame = deepcopy(_out_locality_measures_df[_out_locality_measures_df['is_non_local_period']]).reset_index(drop=True)
            non_local_locality_measures_df
            
        """
        _out_locality_measures_df: pd.DataFrame = deepcopy(self.perform_build_locality_measures_df(locality_measures_dict_dict=self.locality_measures_dict_dict, time_window_centers=self.time_window_centers, paradigm_epochs_df=self.paradigm_epochs_df,
                                                                                                    xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers))
        self.locality_measures_df = deepcopy(_out_locality_measures_df)

        return self.locality_measures_df


    # ==================================================================================================================================================================================================================================================================================== #
    # Plotting/Visualization                                                                                                                                                                                                                                                               #
    # ==================================================================================================================================================================================================================================================================================== #


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from

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



    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)


    @classmethod
    def _reload_class(cls, an_instance):
        """ specifically updates the instance after its class definition has been updated.
        """
        non_init_subset=['_VersionedResultMixin_version', '_interpolator', 'locality_measures_df',
            'moving_avg_meas_pos_overlap_dict',
        ]

        _full_state = an_instance.__getstate__()
        defer_compute_on_init: bool = _full_state.pop('defer_compute_on_init', True) ## exclude this
        defer_compute_on_init = True ## OVERRIDE
        _init_state = get_dict_subset(_full_state, subset_excludelist=non_init_subset)
        _post_init_state = get_dict_subset(_full_state, subset_includelist=non_init_subset)
        _obj = cls(defer_compute_on_init=True, **_init_state) ## prevent computation
        _obj.__dict__.update(**_post_init_state) ## perform literal update
        return _obj


    def downsampling_spatial_data(self, spatial_axes=(0, 1, 2), factors=(5, 5), axes=(0, 1)) -> "DecodingLocalityMeasures":
        """ downsample all of the position dependent properties and returns a downsampled copy of itself. 

        decoding_locality_measures = decoding_locality_measures.downsampling_spatial_data()
        """
        from neuropy.utils.probability_downsampling import RigorousPDFDownsampler
        from neuropy.utils.mixins.binning_helpers import get_bin_edges

        downsampled_decoding_locality_measures = deepcopy(self) ## make a copy of self
        
        p_x_given_n = downsampled_decoding_locality_measures.p_x_given_n ## np.shape(p_x_given_n) # (62, 62, 2, 151732)
        fine_pdf = p_x_given_n

        ## before downsampling
        fine_norm_sum = np.nansum(fine_pdf, axis=spatial_axes)
        assert np.allclose(fine_norm_sum, 1)

        downsampler = RigorousPDFDownsampler(fine_pdf, spatial_axes=spatial_axes)
        coarse_pdf, coarse_bin_sizes, coarse_bins = downsampler.downsample(factors=factors, axes=axes)

        ## after downsampling
        coarse_norm_sum = np.nansum(coarse_pdf, axis=spatial_axes)
        assert np.allclose(coarse_norm_sum, 1)

        # Update PDF data
        downsampled_decoding_locality_measures.p_x_given_n = coarse_pdf

        # Update spatial bin centers
        downsampled_decoding_locality_measures.xbin_centers = coarse_bins[0]
        downsampled_decoding_locality_measures.ybin_centers = coarse_bins[1]

        # Compute bin edges from centers
        downsampled_decoding_locality_measures.xbin = get_bin_edges(downsampled_decoding_locality_measures.xbin_centers)
        downsampled_decoding_locality_measures.ybin = get_bin_edges(downsampled_decoding_locality_measures.ybin_centers)

        # Update sigma based on new bin sizes
        x_step = coarse_bin_sizes[0]
        y_step = coarse_bin_sizes[1]
        downsampled_decoding_locality_measures.sigma = np.nanmax([x_step, y_step]) * 5.0

        # Reset computed fields that depend on spatial bins
        downsampled_decoding_locality_measures.gaussian_volume = None
        downsampled_decoding_locality_measures.p_x_given_n_dict = None
        downsampled_decoding_locality_measures.locality_measures_df = None

        ## do the final computes to ensure all the properties are correct:
        downsampled_decoding_locality_measures.perform_compute_on_load()
        downsampled_decoding_locality_measures.compute() ## ~30 seconds

        return downsampled_decoding_locality_measures



    # ==================================================================================================================================================================================================================================================================================== #
    # Graphical/Display Helpers                                                                                                                                                                                                                                                            #
    # ==================================================================================================================================================================================================================================================================================== #
    def get_non_local_epochs(self, merging_adjacent_max_separation_sec=0.5, **kwargs) -> pd.DataFrame:
        """
        non_local_locality_measures_epochs_df = decoding_locality_measures.get_non_local_epochs(merging_adjacent_max_separation_sec=0.5)
        render_scrollable_colored_table_from_dataframe(non_local_locality_measures_epochs_df)

        """
        _out_locality_measures_df: pd.DataFrame = deepcopy(self.locality_measures_df)
        # _out_locality_measures_df['t'].diff() # 0.25
        non_local_locality_measures_df = deepcopy(_out_locality_measures_df[_out_locality_measures_df['is_non_local_period']])
        ## Compute adjacent epochs:
        non_local_locality_measures_epochs_df = _out_locality_measures_df.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (_out_locality_measures_df['is_non_local_period'].to_numpy()), merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, **kwargs)
        return non_local_locality_measures_epochs_df
    

    @function_attributes(short_name=None, tags=['MAIN', 'NEW', 'epochs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-18 08:02', related_items=[])
    def get_non_moving_PBE_non_local_epochs(self, sess, merging_adjacent_max_separation_sec=0.5, skip_get_non_overlapping = False, should_assign_to_session: bool=True, **kwargs) -> pd.DataFrame:
        """ not-pure, updates `self.non_local_PBE_non_moving_epochs_df`
        
        Updates:
            `self.non_local_PBE_non_moving_epochs_df`
            `sess.non_moving_pbe_non_local_epochs` -- creates the epochs in the pipeline
            
        Usage:
            from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
            from neuropy.core.session.dataSession import DataSession
            from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import DecodingLocalityMeasures, PredictiveDecoding

            ## get the non-local, non-pbe, epochs
            non_local_PBE_non_moving_epochs_df: pd.DataFrame = decoding_locality_measures.get_non_moving_PBE_non_local_epochs(curr_active_pipeline.sess, merging_adjacent_max_separation_sec=0.5)
            non_local_PBE_non_moving_epochs_df: pd.DataFrame = overlap_included_only_df_dict['non_moving_PBE']
            curr_active_pipeline.sess.non_moving_pbe_non_local_epochs = deepcopy(non_local_PBE_non_moving_epochs_df)

            render_scrollable_colored_table_from_dataframe(non_local_PBE_non_moving_epochs_df)

        """
        from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import DecodingLocalityMeasures, PredictiveDecoding
        from neuropy.core.session.dataSession import DataSession
        

        # skip_get_non_overlapping = True

        non_local_locality_measures_epochs_df = self.get_non_local_epochs(merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, **kwargs)
        non_running_epochs = DataSession.perform_compute_non_running_epochs(session=sess, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, **kwargs)
        if should_assign_to_session:
            setattr(sess, 'non_running_epochs', ensure_Epoch(non_running_epochs))
        
        # sess.non_running_epochs = sess.compute_non_running_epochs()
        assert sess.non_running_epochs is not None
        intervals_to_overlap_dict ={
            # 'is_in_laps': deepcopy(sess.laps.to_dataframe()), 'is_in_pbes': deepcopy(sess.pbe),
            # 'non_moving_PBE': [deepcopy(sess.pbe), deepcopy(sess.non_running_epochs)], sess.compute_non_running_epochs()
            'non_moving_PBE': [deepcopy(sess.pbe), deepcopy(sess.non_running_epochs)], 
            # 'is_non_moving_laps': [deepcopy(sess.laps.to_dataframe()), deepcopy(sess.non_running_epochs)],
        }
        overlap_included_only_df_dict = {}

        for epochs_name_key, epochs_df_required_to_overlap in intervals_to_overlap_dict.items(): #:str ='is_in_laps'
            
            active_main_epochs_df = None
            
            if isinstance(epochs_df_required_to_overlap, (list, tuple)):
                ## iterate through    
                for a_series_epochs_df_required_to_overlap in epochs_df_required_to_overlap:
                    if active_main_epochs_df is None:
                        active_main_epochs_df = deepcopy(non_local_locality_measures_epochs_df)
                        
                    # is_included: NDArray = EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df=non_local_locality_measures_epochs_df, epochs_df_required_to_overlap=a_series_epochs_df_required_to_overlap)
                    active_main_epochs_df = active_main_epochs_df.epochs.intersecting(a_series_epochs_df_required_to_overlap, skip_get_non_overlapping=skip_get_non_overlapping)
                ## END for a_series_epochs_df_requi....
                
            else:
                ## single epochs to overlap
                # is_included: NDArray = EpochHelpers.find_epochs_overlapping_other_epochs(epochs_df=non_local_locality_measures_epochs_df, epochs_df_required_to_overlap=epochs_df_required_to_overlap)
                active_main_epochs_df = deepcopy(non_local_locality_measures_epochs_df).epochs.intersecting(epochs_df_required_to_overlap, skip_get_non_overlapping=skip_get_non_overlapping)
                
            ## OUTPUTS: active_main_epochs_df
            # non_local_locality_measures_epochs_df[is_in_key] = is_included ## as a column to the original dataframe (worthless now)
            overlap_included_only_df_dict[epochs_name_key] = active_main_epochs_df # non_local_locality_measures_epochs_df[non_local_locality_measures_epochs_df[is_in_key]].drop(columns=[is_in_key])
            if (should_assign_to_session and (active_main_epochs_df is not None)):
                # setattr(sess, 
                joint_epochs_name_key: str = '_'.join([epochs_name_key, "non_local_epochs"]) # 'non_moving_pbe_non_local_epochs'
                print(f'setting sess.{joint_epochs_name_key}...')
                setattr(sess, joint_epochs_name_key, ensure_Epoch(deepcopy(active_main_epochs_df)))


        ## END for is_in_key, epoch....
        
        ## Assign to `self.non_local_PBE_non_moving_epochs_df`
        non_local_PBE_non_moving_epochs_df: pd.DataFrame = overlap_included_only_df_dict['non_moving_PBE']
        self.non_local_PBE_non_moving_epochs_df = deepcopy(non_local_PBE_non_moving_epochs_df)

        ## Assigns to the session when we build the epochs:
        sess.non_moving_pbe_non_local_epochs = ensure_Epoch(deepcopy(non_local_PBE_non_moving_epochs_df), metadata={'computed_by':'DecodingLocalityMeasures.get_non_moving_PBE_non_local_epochs'})

        return self.non_local_PBE_non_moving_epochs_df
    

    # @function_attributes(short_name=None, tags=['intervals', 'track', 'GUI', 'epochs', 'non-local'], input_requires=[], output_provides=[], uses=['IntervalRectsItem', 'self.get_non_local_epochs'], used_by=[], creation_date='2025-12-22 07:48', related_items=[])
    # def add_non_local_epochs_to_intervals_timeline(self, active_2d_plot, identifier:str='non-local', non_local_epochs_df: pd.DataFrame=None, visualization_update_dict=None):
    #     """  Add `non_local_locality_measures_epochs_df` to timeline as interval epochs

    #     a_rect_item, non_local_locality_measures_epochs_df = decoding_locality_measures.add_non_local_epochs_to_intervals_timeline(active_2d_plot=active_2d_plot)
    #     """
    #     from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem

    #     ## Compute adjacent epochs:
    #     if non_local_epochs_df is None:
    #         non_local_epochs_df = self.get_non_local_epochs(merging_adjacent_max_separation_sec=0.5)
            
    #     ## INPUTS: non_local_locality_measures_epochs_df
    #     _out_intervals = active_2d_plot.add_rendered_intervals(non_local_epochs_df, name=identifier)
    #     a_rect_item: IntervalRectsItem = _out_intervals['RootPlot']['rect_item']
    #     # Direct dictionary update
    #     if visualization_update_dict is None:
    #         visualization_update_dict = {
    #             identifier: dict(y_location=-2.0, height=0.9, pen_color="#d8db06", pen_opacity=0.7843137254901961, brush_color="#bbae00", brush_opacity=0.6078431372549019),
    #         }
    #     active_2d_plot.update_rendered_intervals_visualization_properties(visualization_update_dict)
    #     return a_rect_item, non_local_epochs_df
    

    @function_attributes(short_name=None, tags=['intervals', 'track', 'GUI', 'epochs', 'non-local'], input_requires=[], output_provides=[], uses=['IntervalRectsItem', 'self.get_non_local_epochs'], used_by=[], creation_date='2025-12-22 07:48', related_items=[])
    def add_non_local_PBE_non_moving_epochs_to_intervals_timeline(self, active_2d_plot, identifier:str='non-local-non-moving-PBEs', non_local_PBE_non_moving_epochs_df: pd.DataFrame=None, visualization_update_dict=None):
        """  Add `non_local_locality_measures_epochs_df` to timeline as interval epochs

        non_local_PBE_non_moving_rect_item, non_local_PBE_non_moving_epochs_df = decoding_locality_measures.add_non_local_PBE_non_moving_epochs_to_intervals_timeline(active_2d_plot=active_2d_plot)
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem

        ## Compute adjacent epochs:
        if non_local_PBE_non_moving_epochs_df is None:
            ## unless provided by user, use the internally computed ones (which have to exist
            assert self.non_local_PBE_non_moving_epochs_df is not None
            non_local_PBE_non_moving_epochs_df = deepcopy(self.non_local_PBE_non_moving_epochs_df)
            
                
        ## INPUTS: non_local_locality_measures_epochs_df
        _out_intervals = active_2d_plot.add_rendered_intervals(non_local_PBE_non_moving_epochs_df, name=identifier)
        non_local_PBE_non_moving_rect_item: IntervalRectsItem = _out_intervals['RootPlot']['rect_item']
        # Direct dictionary update
        if visualization_update_dict is None:
            visualization_update_dict = {
                identifier: dict(y_location=-2.0, height=0.9, pen_color="#dbb406", pen_opacity=0.78, brush_color="#bb8f00", brush_opacity=0.61),
            }
        active_2d_plot.update_rendered_intervals_visualization_properties(visualization_update_dict)
        return non_local_PBE_non_moving_rect_item, non_local_PBE_non_moving_epochs_df



@define(slots=False, repr=False, eq=False)
class PredictiveDecoding(ComputedResult): #PickleSerializableMixin, AttrsBasedClassHelperMixin):
    """ Relates to how PBE activity predicts future visited locations, and how visited locations are potentially replayed in future PBEs
    
    Implementation Notes:
        Integrate using a sliding window with the last 30 seconds as inputtttt

        For each actual position, see if it was predicted from the preceeding decoded position

        Kamran suspects that replay will occur of places that the animal is NOT CURRENTLY GOING OR AT. TO "keep em fresh" maybe, or "normalize them in the brain", or "consolidate representation"

    Usage:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures
        
        # First create DecodingLocalityMeasures (if not already available)
        locality_measures = DecodingLocalityMeasures.init_from_decode_result(...)
        
        # Then create PredictiveDecoding from locality_measures
        predictive_decoding = PredictiveDecoding.init_from_decode_result(
            curr_active_pipeline=curr_active_pipeline,
            locality_measures=locality_measures,
            a_result_decoded=a_result_decoded,  # optional, can extract from locality_measures if not provided
            window_size=200
        )
        
        # Compute normalized outputs
        predictive_decoding.compute(sigma=1.0)
        moving_avg_dict, moving_avg_meas_pos_overlap_dict, gaussian_volume = predictive_decoding.compute(sigma=1.0)
        
        # Add layers for visualization
        out_layers, config_widgets_dict_dict, dock_window = predictive_decoding.add_all_layers(sync_plotters=sync_plotters)


    """
    _VersionedResultMixin_version: str = "2025.12.22_0" # to be updated in your IMPLEMENTOR to indicate its version

    window_size: int = serialized_field()
    
    # Composition: delegate locality-related computations to DecodingLocalityMeasures
    locality_measures: DecodingLocalityMeasures = serialized_field()

    # predictive decoding (unique to PredictiveDecoding):
    moving_avg: NDArray[ND.Shape["N_X_BINS, N_Y_BINS, 2, N_TIME_BINS"], np.floating] = serialized_field()
    moving_avg_dict: Dict[str, NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TIME_BINS"], np.floating]] = serialized_field(default=Factory(dict))

    ## past/future to present comparisons:
    moving_avg_meas_pos_overlap_dict: Dict[str, NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TIME_BINS"], np.floating]] = serialized_field(default=Factory(dict))

    epoch_matching_past_future_positions: List[Tuple[pd.DataFrame, pd.DataFrame]] = serialized_field(default=Factory(list), metadata={'date_added': '2025.12.22_0'})
    matching_pos_dfs_list: List[pd.DataFrame] = serialized_field(default=Factory(list), metadata={'date_added': '2025.12.22_0'})
    matching_pos_epochs_dfs_list: List[pd.DataFrame] = serialized_field(default=Factory(list), metadata={'date_added': '2025.12.22_0'})

    
    def __attrs_post_init__(self):
        # Add post-init logic here
        pass

    # Delegate properties to DecodingLocalityMeasures
    @property
    def time_window_centers(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.time_window_centers

    @property
    def epoch_names(self) -> List[str]:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.epoch_names

    @property
    def gaussian_volume(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.gaussian_volume

    @property
    def p_x_given_n_dict(self) -> Dict[str, NDArray]:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.p_x_given_n_dict

    @property
    def xbin(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.xbin

    @property
    def ybin(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.ybin

    @property
    def xbin_centers(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.xbin_centers

    @property
    def ybin_centers(self) -> NDArray:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.ybin_centers

    @property
    def locality_measures_df(self) -> pd.DataFrame:
        """Delegate to DecodingLocalityMeasures."""
        return self.locality_measures.locality_measures_df


    @function_attributes(short_name=None, tags=['predictive_decoding', 'layers'], input_requires=[], output_provides=[], uses=[], used_by=['init_from_decode_result'], creation_date='2025-12-09 19:03', related_items=[])
    @classmethod
    def _perform_compute_predictive_decoding(cls, pos_df: pd.DataFrame, time_window_centers: NDArray, p_x_given_n: NDArray, window_size: int = 200):
        """ Computes a moving average from the decoded posterior

        Args:
            curr_active_pipeline: The active pipeline object
            time_window_centers: Array of time window centers
            p_x_given_n: Decoded posterior probability array with shape (n_x_bins, n_y_bins, n_tasks, n_time_bins)
            window_size: Size of the moving average window (default: 200)

        Returns:
            tuple: (time_window_centers, pos_df, moving_avg, new_positions, p_x_given_n)

        Usage:
            # Get position dataframe
            pos_df = deepcopy(curr_active_pipeline.sess.position.to_dataframe())

            time_window_centers, pos_df, moving_avg, new_positions, p_x_given_n = _perform_compute_predictive_decoding(
                pos_df=pos_df,
                time_window_centers=time_window_centers,
                p_x_given_n=p_x_given_n,
                window_size=200
            )
        """
        from scipy.interpolate import interp1d
        from neuropy.utils.indexing_helpers import flatten

        

        # axis=0 interpolates along rows (time) for all columns ('x' and 'y')
        # fill_value="extrapolate" allows sampling outside original time range
        interpolator = interp1d(pos_df['t'], pos_df[['x', 'y']], kind='linear', axis=0, fill_value="extrapolate")

        # Returns shape new_positions .shape: (n_target_times, 2)
        time_window_centers = flatten(time_window_centers)

        new_positions = interpolator(time_window_centers)
        
        # 2. Calculate Cumulative Sum along the time axis (axis=-1)
        # We use float64 to prevent precision loss over 100k+ bins
        cumsum = np.cumsum(np.insert(p_x_given_n, 0, 0, axis=-1), axis=-1, dtype=np.float64)

        # 3. Compute the Mean
        # The mean at index 'i' is (Sum[i+1] - Sum[i-W+1]) / W
        # We slice the cumsum array to subtract the trailing window sums
        # Shape becomes (..., 103948 - 2000 + 1)
        valid_means = (cumsum[..., window_size:] - cumsum[..., :-window_size]) / window_size

        # 4. Align with Original Time Bins
        # Create an array of NaNs for the first (window_size - 1) bins
        pad_shape = list(p_x_given_n.shape)
        pad_shape[-1] = window_size - 1
        nan_padding = np.full(pad_shape, np.nan)

        # Concatenate to restore original shape (..., 103948)
        moving_avg = np.concatenate((nan_padding, valid_means), axis=-1)

        # print(moving_avg.shape)
        # Output: (41, 63, 2, 103948)

        ## INPUTS: sync_plotters, moving_avg
        return time_window_centers, pos_df, moving_avg, new_positions, p_x_given_n
    

    @classmethod
    def init_from_decode_result(cls, pos_df: pd.DataFrame, locality_measures: DecodingLocalityMeasures, a_result_decoded: Optional[DecodedFilterEpochsResult] = None, window_size: int = 200, sigma: Optional[float] = None) -> "PredictiveDecoding":
        """ Initialize PredictiveDecoding from locality_measures and optionally a_result_decoded.
        
        Args:
            curr_active_pipeline: The active pipeline object
            locality_measures: DecodingLocalityMeasures object containing decoder information and locality data
            a_result_decoded: Optional DecodedFilterEpochsResult. If not provided, data will be extracted from locality_measures
            window_size: Size of the moving average window (default: 200)
            sigma: Optional sigma parameter (currently unused but kept for compatibility)
        
        Returns:
            PredictiveDecoding: Initialized PredictiveDecoding instance
        
        Usage:
            # Get position dataframe
            pos_df = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
            _obj: PredictiveDecoding = PredictiveDecoding.init_from_decode_result(
                pos_df=pos_df,
                locality_measures=locality_measures,
                a_result_decoded=a_result_decoded,  # optional
                window_size=200
            )
        """
        # If a_result_decoded is not provided, extract needed data from locality_measures
        if a_result_decoded is None:
            # Extract time_window_centers and p_x_given_n from locality_measures
            time_window_centers = deepcopy(locality_measures.time_window_centers)
            p_x_given_n = deepcopy(locality_measures.p_x_given_n)
        else:
            # Try to use data from a_result_decoded, but fall back to locality_measures if attributes don't exist
            from neuropy.utils.indexing_helpers import flatten
            
            # Prefer flat_time_window_centers if available (for DecodedFilterEpochsResult)
            time_window_centers = getattr(a_result_decoded, 'flat_time_window_centers', None)
            if time_window_centers is None:
                # Try time_window_centers (might be a list of arrays)
                time_window_centers = getattr(a_result_decoded, 'time_window_centers', None)
                if time_window_centers is not None:
                    # Flatten if it's a list of arrays
                    if isinstance(time_window_centers, list):
                        time_window_centers = flatten(time_window_centers)
                    else:
                        time_window_centers = deepcopy(time_window_centers)
                else:
                    # Try time_bin_container (singular) for continuous decoded results
                    if hasattr(a_result_decoded, 'time_bin_container'):
                        time_window_centers = deepcopy(a_result_decoded.time_bin_container.centers)
                    else:
                        # Fall back to locality_measures
                        time_window_centers = deepcopy(locality_measures.time_window_centers)
            else:
                time_window_centers = deepcopy(time_window_centers)
            
            p_x_given_n = getattr(a_result_decoded, 'p_x_given_n', None)
            if p_x_given_n is None:
                # Fall back to locality_measures
                p_x_given_n = deepcopy(locality_measures.p_x_given_n)
            else:
                p_x_given_n = deepcopy(p_x_given_n)

        # Compute moving average (unique to PredictiveDecoding)
        time_window_centers, pos_df, moving_avg, new_positions, p_x_given_n = cls._perform_compute_predictive_decoding(
            pos_df=pos_df,
            time_window_centers=time_window_centers,
            p_x_given_n=p_x_given_n,
            window_size=window_size,
        )

        _obj = cls(window_size=window_size, locality_measures=locality_measures, moving_avg=moving_avg)
        return _obj



    @function_attributes(short_name=None, tags=['normalization', 'predictive_decoding'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-11 17:03', related_items=[])
    def build_normalized_outputs(self):
        """ Normalize: self.moving_avg over the decoder time period ('sprinkle', 'roam')
        
        Note: p_x_given_n_dict normalization is handled by DecodingLocalityMeasures.build_normalized_outputs()
        
        Updates: self.moving_avg_dict, self.moving_avg_meas_pos_overlap_dict
        """
        def _subfn_renormalize_marginal(a_moving_avg):
            ## renormalize over context:
            norm_sums = np.nansum(a_moving_avg, axis=(0, 1))
            is_nonzero = np.nonzero(norm_sums)
            for a_nonzero_idx in is_nonzero:
                a_moving_avg[:, :, a_nonzero_idx] = a_moving_avg[:, :, a_nonzero_idx] / norm_sums[a_nonzero_idx]
            return a_moving_avg

        # Ensure DecodingLocalityMeasures has normalized p_x_given_n_dict
        if (self.locality_measures.p_x_given_n_dict is None) or (len(self.locality_measures.p_x_given_n_dict) == 0):
            self.locality_measures.build_normalized_outputs()

        ## INPUTS: quantities to renormalize
        self.moving_avg_dict = {}
        self.moving_avg_meas_pos_overlap_dict = {}

        # for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):
        for an_epoch_idx, an_epoch_name in enumerate(self.epoch_names):
            ## "epoch" in the loop variables refers to only the session.paradigm epochs, like ['roam', 'sprinkle']

            a_moving_avg = deepcopy(np.squeeze(self.moving_avg[:, :, an_epoch_idx, :]))
            a_moving_avg = _subfn_renormalize_marginal(a_moving_avg)
            self.moving_avg_dict[an_epoch_name] = a_moving_avg

        ## END for an_epoch_idx, an_epoch_n...

        ## OUTPUTS: _a_moving_avg_dict, _a_moving_avg_meas_pos_overlap_dict
        return self.moving_avg_dict, self.moving_avg_meas_pos_overlap_dict
    

    @function_attributes(short_name=None, tags=['normalization', 'locality', 'overlap'], input_requires=[], output_provides=[], uses=['self.locality_measures.compute'], used_by=[], creation_date='2025-12-11 17:03', related_items=[])
    def compute(self, sigma: float = 1.0):
        """ Normalize: self.p_x_given_n_dict and self.moving_avg over the decoder time period ('sprinkle', 'roam')

        Normalize and convolve each new_position 2D point (x, y) with a fixed width 2D gaussian
        
        Updates: self.
            .moving_avg_dict, .moving_avg_meas_pos_overlap_dict
        Note: Locality measures are computed by DecodingLocalityMeasures.compute()
        """
        # Delegate locality computations to DecodingLocalityMeasures
        if (self.locality_measures.sigma is None) or ((self.locality_measures.sigma != sigma) and (not (sigma is None))):
            # Update sigma if different
            self.locality_measures.sigma = sigma
            # Force recomputation by clearing gaussian_volume (will be recomputed in compute())
            if hasattr(self.locality_measures, 'gaussian_volume'):
                self.locality_measures.gaussian_volume = None
        
        # Compute locality measures (handles gaussian_volume, p_x_given_n_dict, locality_measures_dict_dict)
        self.locality_measures.compute()
        
        # Build normalized outputs for moving average (unique to PredictiveDecoding)
        self.build_normalized_outputs()
        
        # Compute moving_avg_meas_pos_overlap_dict (unique to PredictiveDecoding)
        self.moving_avg_meas_pos_overlap_dict = {}
        for an_epoch_name in self.epoch_names:
            a_moving_avg = self.moving_avg_dict[an_epoch_name]
            # Compute overlap between gaussian_volume (from locality_measures) and moving_avg
            self.moving_avg_meas_pos_overlap_dict[an_epoch_name] = (self.gaussian_volume * a_moving_avg)

        print(f'done with compute.')
        
        ## OUTPUTS: _a_moving_avg_dict, _a_moving_avg_meas_pos_overlap_dict
        return self.moving_avg_dict, self.moving_avg_meas_pos_overlap_dict, self.gaussian_volume


    # ==================================================================================================================================================================================================================================================================================== #
    # Plotting/Visualization                                                                                                                                                                                                                                                               #
    # ==================================================================================================================================================================================================================================================================================== #

    ## INPUTS: sync_plotters, moving_avg
    @function_attributes(short_name=None, tags=['predictive_decoding', 'layers', 'heatmap', 'overlay'], input_requires=[], output_provides=[], uses=['TimeSynchronizedGenericPlotterLayer'], used_by=[], creation_date='2025-12-09 19:03', related_items=[])
    @classmethod
    def add_moving_average_layers(cls, sync_plotters: Dict, time_window_centers: NDArray, moving_avg: NDArray):
        """ 
        
        moving_avg = add_predictive_decoding_layers(curr_active_pipeline=curr_active_pipeline, directional_decoders_decode_result=directional_decoders_decode_result, window_size=200)
        out_layers, config_widgets_dict = add_moving_average_layers(sync_plotters=sync_plotters, moving_avg=moving_avg)

        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedGenericPlotterLayer import TimeSynchronizedGenericPlotterLayer
        
        out_layers = {}
        config_widgets_dict_dict = {}

        for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):
            a_moving_avg = deepcopy(np.squeeze(moving_avg[:, :, an_epoch_idx, :]))
            # np.shape(a_moving_avg)
            ## renormalize over context:
            norm_sums = np.nansum(a_moving_avg, axis=(0, 1))
            is_nonzero = np.nonzero(norm_sums)
            for a_nonzero_idx in is_nonzero:
                a_moving_avg[:, :, a_nonzero_idx] = a_moving_avg[:, :, a_nonzero_idx] / norm_sums[a_nonzero_idx]

            ## OUTPUTS: a_moving_avg
            a_stack_item_key: str = f"{an_epoch_name}_hist"
            
            a_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name=a_stack_item_key, parent=a_plotter, contents={}, data={'time_window_centers': deepcopy(time_window_centers),
                                                                                                                                                                'main': deepcopy(a_moving_avg), 
                                                                                                                                                        })
            out_layers[an_epoch_name] = a_layer
            # a_widget = a_stack_item.create_layer_configs_widget()
            # config_widgets_dict[a_stack_item_key] = a_widget
            config_widgets_dict_dict[an_epoch_name] = {} 

            ## adjust just the relevant layer
            a_widget = a_layer.create_layer_configs_widget()
            config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
            a_widget.show()  
            
            # ## scan through all possible extant layers:
            # for z_idx, (a_stack_item_key, a_stack_item) in enumerate(a_plotter.ui.plot_stack.items()):
            #     print(f'Update: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
            #     try:
            #         if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
            #             a_widget = a_stack_item.create_layer_configs_widget()
            #             config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
            #             a_widget.show()            
            #             print(f'\tupdate successful.')
            #         else:
            #             print(f'\tskipped!')
            #     except (KeyError, AttributeError) as e:
            #         print(f'\t encountered error "{e}" while trying to update item. Skipping.')
            #     except Exception as e:
            #         ## Unexpected exception!
            #         raise e
                

        return out_layers, config_widgets_dict_dict


    @function_attributes(short_name=None, tags=['layers', 'image', 'heatmap', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-11 13:10', related_items=[])
    def add_all_layers(self, sync_plotters: Dict, **kwargs):
        """ adds all related layers to a TimeSynchronizedDecoderPlotter widget
        
        moving_avg = add_predictive_decoding_layers(curr_active_pipeline=curr_active_pipeline, directional_decoders_decode_result=directional_decoders_decode_result, window_size=200)
        out_layers, config_widgets_dict = add_moving_average_layers(sync_plotters=sync_plotters, moving_avg=moving_avg)

        """
        import pyqtgraph as pg
        from pyqtgraph.dockarea import DockArea, Dock
        from PyQt5.QtWidgets import QMainWindow
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedGenericPlotterLayer import TimeSynchronizedGenericPlotterLayer

        def _subfn_stack_widgets_vertically(config_widgets_dict_dict):
            # 1. Create the container window and DockArea
            win = QMainWindow()
            win.setWindowTitle("Stacked Config Widgets")
            area = DockArea()
            win.setCentralWidget(area)
            win.resize(400, 800)

            # 2. Iterate and stack
            prev_dock = None
            
            # Loop through the outer dictionary (categories like 'roam', 'sprinkle')
            for category, inner_dict in config_widgets_dict_dict.items():
                # Loop through the inner dictionary (actual widgets)
                for name, widget in inner_dict.items():
                    # Create the Dock (title includes category for clarity)
                    dock = Dock(f"{category}: {name}", size=(500, 200))
                    dock.addWidget(widget)

                    # Stack logic: 
                    # If it's the first dock, place it. 
                    # Otherwise, place it at the 'bottom' relative to the previous dock.
                    if prev_dock is None:
                        area.addDock(dock, 'left')
                    else:
                        area.addDock(dock, 'bottom', prev_dock)
                    
                    prev_dock = dock

            win.show()
            return win


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        out_layers = {}
        config_widgets_dict_dict = {}

        for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):
            a_moving_avg = deepcopy(self.moving_avg_dict[an_epoch_name])
            a_stack_item_key: str = f"{an_epoch_name}_hist"
            
            a_history_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name=a_stack_item_key, parent=a_plotter, contents={}, data={'time_window_centers': deepcopy(self.time_window_centers),
                                                                                                                                                                'main': deepcopy(a_moving_avg), 
                                                                                                                                                        })
            out_layers[a_stack_item_key] = a_history_layer
            # a_widget = a_stack_item.create_layer_configs_widget()
            # config_widgets_dict[a_stack_item_key] = a_widget
            config_widgets_dict_dict[an_epoch_name] = {} 

            # ## adjust just the relevant layer
            # a_widget = a_history_layer.create_layer_configs_widget()
            # config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
            # a_widget.setWindowTitle(f"Config[{a_stack_item_key}]")
            # a_widget.show() 
            

            a_moving_avg_meas_pos_overlap = deepcopy(self.moving_avg_meas_pos_overlap_dict[an_epoch_name])
            a_stack_item_key: str = f"{an_epoch_name}_overlap"
            
            a_overlap_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name=a_stack_item_key, parent=a_plotter, contents={}, data={'time_window_centers': deepcopy(self.time_window_centers),
                                                                                                                                                                'main': deepcopy(a_moving_avg_meas_pos_overlap), 
                                                                                                                                                        })
            out_layers[a_stack_item_key] = a_overlap_layer

            # ## adjust just the relevant layer
            # a_widget = a_overlap_layer.create_layer_configs_widget()
            # config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
            # a_widget.setWindowTitle(f"Config[{a_stack_item_key}]")
            # a_widget.show() 


            ## scan through all possible extant layers:
            for z_idx, (a_stack_item_key, a_stack_item) in enumerate(a_plotter.ui.plot_stack.items()):
                print(f'Update: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
                try:
                    if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                        a_widget = a_stack_item.create_layer_configs_widget()
                        config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
                        a_widget.setWindowTitle(f"Config[{a_stack_item_key}]")
                        a_widget.show()            
                        print(f'\tupdate successful.')
                    else:
                        print(f'\tskipped!')
                except (KeyError, AttributeError) as e:
                    print(f'\t encountered error "{e}" while trying to update item. Skipping.')
                except Exception as e:
                    ## Unexpected exception!
                    raise e
                
        ## Wrap each widget in a pg.DockItem and then stack them vertically in a new window:
        dock_window = _subfn_stack_widgets_vertically(config_widgets_dict_dict)

        return out_layers, config_widgets_dict_dict, dock_window
    

    @classmethod
    def add_layers_params_config_widgets(cls, sync_plotters):
        """ 
        """
        config_widgets_dict_dict = {}

        for an_epoch_idx, (an_epoch_name, a_plotter) in enumerate(sync_plotters.items()):

            config_widgets_dict_dict[an_epoch_name] = {} 
            
            for z_idx, (a_stack_item_key, a_stack_item) in enumerate(a_plotter.ui.plot_stack.items()):
                print(f'Update: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
                try:
                    if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                        a_widget = a_stack_item.create_layer_configs_widget()
                        config_widgets_dict_dict[an_epoch_name][a_stack_item_key] = a_widget
                        a_widget.setWindowTitle(f"Config[{a_stack_item_key}]")
                        a_widget.show()            
                        print(f'\tupdate successful.')
                    else:
                        print(f'\tskipped!')
                except (KeyError, AttributeError) as e:
                    print(f'\t encountered error "{e}" while trying to update item. Skipping.')
                except Exception as e:
                    ## Unexpected exception!
                    raise e
                
        ## OUTPUTS: config_widgets_dict_dict
        return config_widgets_dict_dict


    # ==================================================================================================================================================================================================================================================================================== #
    # 2025-12-11 - Prospective/Retrospective Decoding Analysis                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['prospective', 'future', 'past', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-11 07:43', related_items=[])
    @classmethod
    def calculate_position_epoch_overlap(cls, gaussian_volume: np.ndarray, pos_time_bin_centers: np.ndarray, decoded_epochs_all_filter_epochs: pd.DataFrame, decoded_epochs_result: Any, curr_decoder_context_idx: int = 0, debug_max_time_steps_to_process: Optional[int] = 200, debug_overide_start_idx: Optional[int]=None, debug_print: bool = True) -> np.ndarray:
        """
        Calculates the overlap between the current position probability (Gaussian volume)
        and all preceding decoded epochs.
        
        Optimized to use vectorized matrix multiplication instead of nested loops.

        Args:
            gaussian_volume: 3D array (X, Y, Time)
            pos_time_bin_centers: 1D array of time centers corresponding to gaussian_volume dim 2
            decoded_epochs_all_filter_epochs: DataFrame containing 'start' and 'stop' columns
            decoded_epochs_result: Object containing .p_x_given_n_list (list of 4D arrays)
            curr_decoder_context_idx: Index for the context dimension (0 or 1)
            debug_max_time_steps_to_process: Max number of time steps to process (for debugging)
            debug_print: Whether to print progress/shape info

        Returns:
            np.ndarray: A 2D array (Time, Epochs) containing scalar overlap scores.
                        Future epochs (relative to time t) are represented as NaN.
        """

        # 1. Setup Time Selection
        # -----------------------
        # Preserve original logic: take last 2000 bins, then apply debug limit
        total_time_bins = len(pos_time_bin_centers)
        if debug_overide_start_idx is not None:
            start_idx = max(0, debug_overide_start_idx)
        else:
            start_idx = 0

        if debug_max_time_steps_to_process is not None:
            # Limit the end index relative to the start_idx
            end_idx = min(total_time_bins, (start_idx + debug_max_time_steps_to_process))
        else:
            end_idx = total_time_bins

        # Slice inputs to relevant time window
        active_pos_time_bin_centers = pos_time_bin_centers[start_idx:end_idx]
        num_pos_time_bin_centers: int = len(active_pos_time_bin_centers)

        if debug_print:
            print(f'num_pos_time_bin_centers: {num_pos_time_bin_centers}')

        # 2. Data Preparation (Flattening & Cleaning)
        # -------------------------------------------
        # Slice the Gaussian volume to match the active time window
        active_gaussian_slice = gaussian_volume[:, :, start_idx:end_idx]
        n_x, n_y, n_t = active_gaussian_slice.shape
        
        # Reshape Gaussian Volume: (X, Y, T) -> (X*Y, T)
        # Use nan_to_num so NaNs become 0.0, allowing efficient dot products (acting like nansum)
        flat_gaussian = np.nan_to_num(active_gaussian_slice.reshape(n_x * n_y, n_t))

        # Prepare Epoch Data
        # We assume decoded_epochs_result.p_x_given_n_list corresponds to the rows in the DataFrame
        all_epoch_stops = decoded_epochs_all_filter_epochs['stop'].to_numpy()
        
        # Flatten spatial dims for all epochs: List of (X*Y, Epoch_Time_Bins) arrays
        # We extract the specific context (curr_decoder_context_idx) immediately
        flat_epoch_arrays = [
            np.nan_to_num(v[:, :, curr_decoder_context_idx, :].reshape(n_x * n_y, -1))
            for v in decoded_epochs_result.p_x_given_n_list
        ]

        # 3. Vectorized Calculation
        # -------------------------
        # Initialize result matrix: (N_Time_Steps, N_Total_Epochs)
        # Initialize with NaN to represent "future" epochs (or padding)
        padded_pos_overlap_matrix = np.full(
            (num_pos_time_bin_centers, len(flat_epoch_arrays)), 
            np.nan
        )

        # Iterate over EPOCHS (Outer loop is now Epochs)
        # This allows us to apply one Epoch to ALL valid time bins simultaneously via matrix mult
        for epoch_idx, (epoch_arr, stop_time) in enumerate(zip(flat_epoch_arrays, all_epoch_stops)):
            
            # Mask: Find all time bins where this epoch is strictly in the past (or current)
            valid_time_mask = active_pos_time_bin_centers >= stop_time
            
            # Optimization: Skip if this epoch hasn't happened yet for any active time bin
            if not np.any(valid_time_mask):
                continue

            # Select the Gaussian columns for valid times: (Space, Valid_Times)
            # Transpose to (Valid_Times, Space) for matrix multiplication
            relevant_gaussian_T = flat_gaussian[:, valid_time_mask].T
            
            # CORE CALCULATION: Matrix Multiplication (The "Dot Product")
            # (Valid_Times, Space) @ (Space, Epoch_Bins) -> (Valid_Times, Epoch_Bins)
            # This effectively performs the sum(A * B) over spatial dimensions
            spatial_sums = np.matmul(relevant_gaussian_T, epoch_arr)
            
            # Calculate median over the epoch's internal time bins (Axis 1)
            scalar_scores = np.median(spatial_sums, axis=1)
            
            # Assign to the main result matrix
            padded_pos_overlap_matrix[valid_time_mask, epoch_idx] = scalar_scores

        if debug_print:
            print(f"Processed {num_pos_time_bin_centers} time steps. "
                f"Final shape: {padded_pos_overlap_matrix.shape}")

        return active_pos_time_bin_centers, padded_pos_overlap_matrix


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from

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



    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)



@define(slots=False, repr=False, eq=False)
class PredictiveDecodingComputationsContainer(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import WCorrShuffle, PredictiveDecodingComputationsContainer

        wcorr_shuffle_results: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('PredictiveDecoding', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'PredictiveDecoding is not computed.')
            
    """
    _VersionedResultMixin_version: str = "2026.01.08_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    predictive_decoding: Optional[PredictiveDecoding] = serialized_field(default=None, repr=False)
    

    pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.20_0", 'copied_from': 'DirectionalDecodersContinuouslyDecodedResult'})
    epochs_decoded_result_cache_dict: Dict[float, Dict[types.DecoderName, DecodedFilterEpochsResult]] = serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.20_0", 'copied_from': 'DirectionalDecodersContinuouslyDecodedResult'}) # key is the t_bin_size in seconds
    debug_computed_dict: Dict[types.DecoderName, Dict] = non_serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.21_0"})

    scoring_results_df: pd.DataFrame = non_serialized_field(default=None, metadata={'field_added': "2026.01.08_0"})


    @property
    def most_recent_decoding_time_bin_size(self) -> Optional[float]:
        """Gets the last cached continuously_decoded_dict property."""
        if ((self.epochs_decoded_result_cache_dict is None) or (len(self.epochs_decoded_result_cache_dict or {}) < 1)):
            return None
        else:
            last_time_bin_size: float = list(self.epochs_decoded_result_cache_dict.keys())[-1]
            return last_time_bin_size   
        

    @property
    def most_recent_continuously_decoded_dict(self) -> Optional[Dict[str, DecodedFilterEpochsResult]]:
        """Gets the last cached continuously_decoded_dict property."""
        last_time_bin_size = self.most_recent_decoding_time_bin_size
        if (last_time_bin_size is None):
            return None
        else:
            # otherwise return the result            
            return self.epochs_decoded_result_cache_dict[last_time_bin_size]   

    @property
    def decoding_locality(self) -> Optional[DecodingLocalityMeasures]:
        """Delegate to predictive_decoding.locality_measures if available."""
        if self.predictive_decoding is not None:
            return self.predictive_decoding.locality_measures
        return None

    @decoding_locality.setter
    def decoding_locality(self, value: DecodingLocalityMeasures):
        """Set locality_measures on predictive_decoding, creating it if needed."""
        if self.predictive_decoding is None:
            raise ValueError("Cannot set decoding_locality when predictive_decoding is None")
        self.predictive_decoding.locality_measures = value



    # RL_ripple: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    # LR_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    # RL_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)

    # ripple_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)
    # laps_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)

    # ripple_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    # ripple_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)
    # # ripple_n_valid_shuffles: Optional[int] = serialized_attribute_field(default=None, repr=False)

    # laps_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    # laps_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)

    # minimum_inclusion_fr_Hz: float = serialized_attribute_field(default=2.0, repr=True)
    # included_qclu_values: Optional[List] = serialized_attribute_field(default=None, repr=True)

    # @property
    # def decoding_locality(self) -> DecodingLocalityMeasures:
    #     """The decoding_locality property."""
    #     assert self.predictive_decoding is not None
    #     return self.predictive_decoding.locality_measures
    # @decoding_locality.setter
    # def decoding_locality(self, value: DecodingLocalityMeasures):
    #     assert self.predictive_decoding is not None
    #     self.predictive_decoding.locality_measures = value

    def __attrs_post_init__(self):
        # Add post-init logic here
        pass

    # Utility Methods ____________________________________________________________________________________________________ #

    @function_attributes(short_name=None, tags=['PENDING', 'IN-PROCESS', '2025-12-20_future_and_past_analysis'], input_requires=[], output_provides=[], uses=['decode_specific_epochs'], used_by=[], creation_date='2025-12-19 14:28', related_items=[])
    def compute_future_and_past_analysis(self, curr_active_pipeline, an_epoch_name:str = 'roam', decoding_time_bin_size=0.025, top_v_percent: float = 0.1, 
                                        merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, ## for merging detected future/past position dataframes
                                        ):
        """ computes the times that 
        
        ## Get the non-local epochs -- where do they encode?

        
        ## Does the animal go there in the futre?
        
        ## Has it been there in the past (duh or we wouldn't have placefields for it)?

        ## Some PBEs that don't qualify as non-local actually might be but they're just paths across the environment.

        ## #TODO 2025-12-19 16:52: - [ ] Let's stick within the same block (roam/sprinkle) for now


        a_test_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=decoded_local_epochs_result, xbin=a_decoder.xbin, xbin_centers=a_decoder.xbin_centers, ybin=a_decoder.ybin, ybin_centers=a_decoder.ybin_centers)
        a_test_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=measured_positions_df, epoch_specific_position_dfs=[relevant_positions_df], epoch_ids=np.array([0]), curr_num_subplots=1, active_page_index=0, plot_actual_lap_lines=True)
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult
        from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior
        

        ## HARDCODED an_epoch_name
        # computed_df_col_name_prefix: str = ''
        computed_df_col_name_prefix: str = f'{an_epoch_name}_'
        
        
        
        ## Get the non-local epochs -- where do they encode?
        # container: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
        decoding_locality: DecodingLocalityMeasures = self.decoding_locality
        
        non_local_PBE_non_moving_epochs_df: pd.DataFrame = decoding_locality.get_non_moving_PBE_non_local_epochs(curr_active_pipeline.sess, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec)
        # non_local_PBE_non_moving_epochs_df: pd.DataFrame = container.decoding_locality.non_local_PBE_non_moving_epochs_df

        measured_positions_df: pd.DataFrame = decoding_locality.pos_df
        # measured_positions_df = measured_positions_df.drop(columns=['binned_x', 'binned_y'], inplace=False)
        measured_positions_df = measured_positions_df.dropna(how='any', subset=['t', 'x', 'y'])
        measured_positions_df = measured_positions_df.position.adding_binned_position_columns(xbin_edges=decoding_locality.xbin, ybin_edges=decoding_locality.ybin)
        measured_positions_df = measured_positions_df[(measured_positions_df['binned_x'].notna()) & (measured_positions_df['binned_y'].notna())] # Filter rows based on columns: 'binned_x', 'binned_y'
        # decoding_locality.pos_df = measured_positions_df
        # measured_positions_df

        ## add the final detected non_local_pbe_epoch indicies to the decoded points:
        non_local_PBE_non_moving_epochs_df['label'] = non_local_PBE_non_moving_epochs_df['label'].astype(int)
        _out_locality_measures_df = deepcopy(decoding_locality.locality_measures_df)
        _out_locality_measures_df = _out_locality_measures_df.time_point_event.adding_epochs_identity_column(epochs_df=non_local_PBE_non_moving_epochs_df, epoch_id_key_name='non_local_PBE_non_moving_epoch', override_time_variable_name='t',
                                                            # epoch_label_column_name='label', no_interval_fill_value=np.nan,
                                                            epoch_label_column_name='label', no_interval_fill_value=-1,
                                                            should_replace_existing_column=True, drop_non_epoch_events=True,
                                                            overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
        # _out_locality_measures_df
        # _out_locality_measures_df.dropna(how='any', subset=['non_local_PBE_non_moving_epoch'])

        epoch_times = decoding_locality.locality_measures_df['t'].to_numpy()
        time_to_idx_map = EpochHelpers.find_epoch_times_to_data_indicies_map(non_local_PBE_non_moving_epochs_df, epoch_times)
        # _out
        non_local_PBE_non_moving_epochs_df: pd.DataFrame = non_local_PBE_non_moving_epochs_df
        non_local_PBE_non_moving_epochs_df['start_idx'] = non_local_PBE_non_moving_epochs_df['start'].map(time_to_idx_map)
        non_local_PBE_non_moving_epochs_df['stop_idx'] = non_local_PBE_non_moving_epochs_df['stop'].map(time_to_idx_map)
        # matching_epoch_times_slice
        # non_local_PBE_non_moving_epochs_dft


        # Get the decoders to decode the epochs with higher precision ________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

        if decoding_time_bin_size not in self.epochs_decoded_result_cache_dict:
            self.epochs_decoded_result_cache_dict[decoding_time_bin_size] = {} ## make the new dict for this time bin size
        

        ## INPUTS: container, directional_decoders_decode_result
        ## decode each epoch at a small time bin size:
        a_decoder: BayesianPlacemapPositionDecoder = self.pf1D_Decoder_dict.get(an_epoch_name, None)
        if a_decoder is None:
            directional_decoders_decode_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']            
            assert directional_decoders_decode_result is not None
            self.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            a_decoder = directional_decoders_decode_result.pf1D_Decoder_dict[an_epoch_name]
    
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[decoding_time_bin_size].get(an_epoch_name, None)
        if decoded_local_epochs_result is None:
            ## if we can't find a pre-computed one:    
            decoded_local_epochs_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=non_local_PBE_non_moving_epochs_df, decoding_time_bin_size=decoding_time_bin_size)
            self.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name] = decoded_local_epochs_result

        print(f'done with all decoding.')

        epoch_high_prob_pos_masks = []
        epoch_matching_positions = []
        epoch_matching_past_future_positions: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        matching_pos_dfs_list: List[pd.DataFrame] = []
        matching_pos_epochs_dfs_list: List[pd.DataFrame] = []

        # [array([0, 1, 2, 3, 4]), array([0, 1]), array([0, 1, 2, 3, 4, 5, 6, 7]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
        n_flattened_tbins: int = np.sum(decoded_local_epochs_result.nbins)
        flattened_time_bin_indicies = np.arange(n_flattened_tbins)
        reverse_flattened_time_bin_indicies_list: List[NDArray] = split_array(flattened_time_bin_indicies, sub_element_lengths=decoded_local_epochs_result.nbins)
        # assert len(split_by_epoch_reverse_flattened_time_bin_indicies) == n_epochs
        gaussian_volume = self.predictive_decoding.gaussian_volume ## the volume for all time bins

        # a_p_x_given_n = decoding_locality.p_x_given_n_dict[an_epoch_name] ## hmmm, this is global probability - (41, 63, 103948)
        for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False)):
            
            ## need to know the indices this corresponds to so I can use my gaussian, p_x_given_n, etc
            ## compute the locality:
            # num_timestamps: int = np.shape(decoding_locality.gaussian_volume)[-1]

            ## What positions does the epoch decode to?
            
            ## Does the animal go there in the futre?
            # an_epoch_pos_df = decoding_locality.pos_df.position.time_sliced(a_row.start, a_row.stop) ## Irreleevant, these are the literal positions during the PBE!
            # a_timestamp_indicies = an_epoch_pos_df.index.to_numpy() ## These on the other hand ARE relevant to ALL arrays, like the p_x_given_n
            
            # is_timebin_included = np.logical_and((a_row.start <= decoding_locality.time_window_centers), (decoding_locality.time_window_centers <= a_row.stop))
            # print(np.sum(is_timebin_included)) ## shoot, because the time bins are so small :[
            
            ## Need correect portion of p_x_given_n for these times
            curr_epoch_p_x_given_n = decoded_local_epochs_result.p_x_given_n_list[i] # [:, :, is_timebin_included]
            curr_epoch_time_bin_centers = decoded_local_epochs_result.time_bin_containers[i].centers    
            # is_high_prob_mask = (curr_epoch_p_x_given_n > high_val_epsilon)
            curr_epoch_start_t: float = curr_epoch_time_bin_centers[0]
            curr_epoch_stop_t: float = curr_epoch_time_bin_centers[-1]
            
            # print(np.shape(curr_epoch_p_x_given_n))
            # curr_epoch_p_x_given_n  # np.shape(curr_epoch_p_x_given_n): (n_x_bins, n_y_Bins, n_time_bins)
            # is_high_prob_mask = curr_epoch_p_x_given_n >= np.sort(curr_epoch_p_x_given_n.ravel())[::-1][np.searchsorted(np.cumsum(np.sort(curr_epoch_p_x_given_n.ravel())[::-1]), 0.1 * curr_epoch_p_x_given_n.sum())]
            curr_epoch_tbin_indicies: NDArray = reverse_flattened_time_bin_indicies_list[i]
            a_gaussian_volume = None
            if gaussian_volume is not None:
                a_gaussian_volume = gaussian_volume[..., curr_epoch_tbin_indicies]
            

            # ==================================================================================================================================================================================================================================================================================== #
            # Special posterior measurement properties (diffusivity, promenence, etc) computed independently with newly decoded fine time bin grainularity posteriors                                                                                                                              #
            # ==================================================================================================================================================================================================================================================================================== #
            ## Call the independent classmethod to compute all locality measures
            custom_computation_results_dict = DecodingLocalityMeasures.compute_locality_measures_for_posterior(
                a_p_x_given_n=curr_epoch_p_x_given_n,
                gaussian_volume=self.gaussian_volume, ## if we have it
                xbin_centers=self.xbin_centers, 
                ybin_centers=self.ybin_centers,
                n_total_pos_bins=self.n_total_pos_bins,
                gaussian_volume=a_gaussian_volume, ## if we have it ## this volume is for the hwole thingy
                min_val_epsilon=1e-9,
                enable_debug_outputs=True,
                earthmovers_fn=None,
            )
            



            ## for each time bin compute the top 10% of the time bins and use those instead of a fixed "high_val_epsilon" threshold:
            #TODO 2025-12-24 20:48: - [ ] LAAAAME - this should use the real promenence topologically connected region, not the random top 10% which can be discontinuous...
            flat = curr_epoch_p_x_given_n.reshape(-1, curr_epoch_p_x_given_n.shape[-1])  # (n_xy, n_time)
            sorted_flat = np.sort(flat, axis=0)[::-1]
            cdf = np.cumsum(sorted_flat, axis=0)
            thresholds = sorted_flat[np.argmax(cdf >= top_v_percent * flat.sum(axis=0), axis=0), np.arange(flat.shape[1])]
            is_high_prob_mask = curr_epoch_p_x_given_n >= thresholds
            ## allow future positions to match any position in the epoch to count:
            any_t_Bin_high_prob_pos_mask = np.any(is_high_prob_mask, axis=-1) ## mask for high prob positions during the epoch
            epoch_high_prob_pos_masks.append(any_t_Bin_high_prob_pos_mask)

            pos_matches_epoch_mask = np.where([any_t_Bin_high_prob_pos_mask[(a_pos.binned_x-1), (a_pos.binned_y-1)] for a_pos in measured_positions_df.itertuples()])[0]

            ## 2D approach
            # rows, cols = np.where(any_t_Bin_high_prob_pos_mask)
            # indices_2d = np.column_stack((rows, cols))
            # ## set approach:
            # indices_2d = set(list(zip(*np.where(any_t_Bin_high_prob_pos_mask)))) ## (row, col) tuples            
            # valid_mask = [(tuple([int(a_pos.binned_x-1), int(a_pos.binned_y-1)]) in indices_2d) for a_pos in measured_positions_df.itertuples()]
            # pos_matches_epoch_mask = np.where(valid_mask)[0]

            relevant_positions_df: pd.DataFrame = measured_positions_df.iloc[pos_matches_epoch_mask].copy()
            is_relevant_past_times = (relevant_positions_df['t'] < curr_epoch_start_t)
            is_relevant_future_times = (relevant_positions_df['t'] > curr_epoch_stop_t)
            relevant_positions_df['is_future_present_past'] = 'present'
            relevant_positions_df.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
            relevant_positions_df.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'

            # _out_split = relevant_positions_df.pho.partition_df_dict('is_future_present_past')

            ## how many timestamps still remain in the past and the future:
            n_total_possible_past_times = np.sum(measured_positions_df['t'] < curr_epoch_start_t)
            n_total_possible_future_times = np.sum(measured_positions_df['t'] > curr_epoch_stop_t)
            
            n_relevant_past_times = np.sum(is_relevant_past_times)
            n_relevant_future_times = np.sum(is_relevant_future_times)

            # n_total_past_times = (measured_positions_df['t'] < curr_epoch_time_bin_centers[0])
            # n_total_future_times = (measured_positions_df['t'] > curr_epoch_stop_t)
            
            epoch_matching_past_future_positions.append((pos_matches_epoch_mask[is_relevant_past_times], pos_matches_epoch_mask[is_relevant_future_times], n_total_possible_past_times, n_total_possible_future_times, n_relevant_past_times, n_relevant_future_times)) ## basically split on current
            
            # relevant_positions_df['is_past'] = (relevant_positions_df['t'] < curr_epoch_stop_t)

            # measured_positions_df[['binned_x', 'binned_y']].to_numpy()
            epoch_matching_positions.append(pos_matches_epoch_mask)
            matching_pos_dfs_list.append(relevant_positions_df)
            # pos_matches_epoch_mask = epoch_matching_positions[i]
            
            ## find adjacent epochs from the position time bins (periods where the animal is in the positions)
            measured_positions_df['is_included'] = False
            # measured_positions_df.loc[measured_positions_df.index[pos_matches_epoch_mask], 'is_included'] = True
            measured_positions_df.loc[measured_positions_df.index[pos_matches_epoch_mask[is_relevant_past_times]], 'is_included'] = True ## only do past/future, not present
            measured_positions_df.loc[measured_positions_df.index[pos_matches_epoch_mask[is_relevant_future_times]], 'is_included'] = True ## only do past/future, not present
            a_matching_pos_epochs_df: pd.DataFrame = measured_positions_df.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (measured_positions_df['is_included'].to_numpy()), merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration)
            
            is_pos_epochs_relevant_past_times = (a_matching_pos_epochs_df['start'] < curr_epoch_start_t)
            is_pos_epochs_relevant_future_times = (a_matching_pos_epochs_df['stop'] > curr_epoch_stop_t)
            a_matching_pos_epochs_df['is_future_present_past'] = 'present'
            a_matching_pos_epochs_df.loc[is_pos_epochs_relevant_past_times, 'is_future_present_past'] = 'past'
            a_matching_pos_epochs_df.loc[is_pos_epochs_relevant_future_times, 'is_future_present_past'] = 'future'
            
            matching_pos_epochs_dfs_list.append(a_matching_pos_epochs_df)


            # PredictiveDecodingComputationsContainer
            ## Has it been there in the past (duh or we wouldn't have placefields for it)?

            ## Some PBEs that don't qualify as non-local actually might be but they're just paths across the environment.

            ## Let's stick within the same block (roam/sprinkle) for now
            

        ## END for i, a_row in enumerate(ensure_dat...


        ## OUTPUTS: epoch_matching_positions

        ratio_past = np.array([len(epoch_matching_past_future_positions[i][0])/ len(decoded_local_epochs_result.time_bin_containers[i].centers) for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        ratio_future = np.array([len(epoch_matching_past_future_positions[i][1])/ len(decoded_local_epochs_result.time_bin_containers[i].centers) for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        n_total_possible_past = np.array([epoch_matching_past_future_positions[i][2] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        n_total_possible_future = np.array([epoch_matching_past_future_positions[i][3] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        n_total_relevant_past = np.array([epoch_matching_past_future_positions[i][4] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        n_total_relevant_future = np.array([epoch_matching_past_future_positions[i][5] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        past_future_info_dict = {'ratio_past': ratio_past, 'ratio_future': ratio_future, 'n_total_possible_past': n_total_possible_past, 'n_total_possible_future': n_total_possible_future, 'n_total_relevant_past': n_total_relevant_past, 'n_total_relevant_future': n_total_relevant_future, }
        # non_local_PBE_non_moving_epochs_df.update(past_future_info_dict)
                
        ## add the columns to the datframe
        for k, v in past_future_info_dict.items():
            non_local_PBE_non_moving_epochs_df[f"{computed_df_col_name_prefix}{k}"] = v
            
        ## add more columns after the others are added:
        non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_past'] = non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_past'] / non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_past']
        non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_future'] = non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_future'] / non_local_PBE_non_moving_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_future']

        ## update the source object
        decoding_locality.non_local_PBE_non_moving_epochs_df = non_local_PBE_non_moving_epochs_df

        ### assign to predictive_decoding.locality_measures (single source of truth)
        if (self.predictive_decoding is not None):
            self.predictive_decoding.locality_measures = decoding_locality

            ## update the other fields
            self.predictive_decoding.epoch_matching_past_future_positions = epoch_matching_past_future_positions
            self.predictive_decoding.matching_pos_dfs_list = matching_pos_dfs_list
            self.predictive_decoding.matching_pos_epochs_dfs_list = matching_pos_epochs_dfs_list


        return epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list #(ratio_past, ratio_future, n_total_past, n_total_future) # , epoch_high_prob_pos_masks




    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['scoring_results_df'] # , 'track_templates'
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from

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



    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)



def validate_has_predictive_decoding_results(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
    """ Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    seq_results: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
    if seq_results is None:
        return False
    
    predictive_decoding = seq_results.predictive_decoding
    if predictive_decoding is None:
        return False
    


    # return True



class PredictiveDecodingComputationsGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to sequence-based decoding computations. """
    _computationGroupName = 'predictive_decoding'
    _computationGlobalResultGroupName = 'PredictiveDecoding'
    _computationPrecidence = 1006
    _is_global = True

    @function_attributes(short_name='predictive_decoding_analysis', tags=['directional_pf', 'laps', 'wcorr', 'session', 'pf1D'],
                        input_requires=['DirectionalDecodersDecoded', 'RankOrder', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz', 'global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values'], output_provides=['PredictiveDecoding'], uses=['PredictiveDecodingComputationsContainer', 'WCorrShuffle'], used_by=[], creation_date='2024-05-27 14:31', related_items=[],
        requires_global_keys=['DirectionalDecodersDecoded', 'DirectionalMergedDecoders', 'RankOrder', 'DirectionalDecodersEpochsEvaluations'], provides_global_keys=['PredictiveDecoding'],
        validate_computation_test=validate_has_predictive_decoding_results, is_global=True)
    def perform_predictive_decoding_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, window_size:int=90, extant_decoded_time_bin_size: Optional[float]=None,
                drop_previous_result_and_compute_fresh:bool=False, min_num_spikes_per_bin_to_be_considered_active: Optional[int]=5, mask_position_like_time_score_cutoff: Optional[float] = 0.42):
        """ Performs predictive decoding analysis to relate PBE activity to future visited locations.

        Requires:
            ['DirectionalDecodersDecoded']

        Provides:
            global_computation_results.computed_data['PredictiveDecoding']
                ['PredictiveDecoding'].predictive_decoding - PredictiveDecoding instance containing computed results


        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
        ## NEW: filtering by whether decoded posterior in each t_bin is "position-like"
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring


        if include_includelist is not None:
            print(f'WARN: perform_predictive_decoding_analysis(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        ## Get the needed data:


        # ==================================================================================================================================================================================================================================================================================== #
        # MASK low-firing bins before using result                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        # extant_decoded_time_bin_size = 0.250
        
        if (min_num_spikes_per_bin_to_be_considered_active is not None) and (min_num_spikes_per_bin_to_be_considered_active > 0):
            ## Masked result:
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(owning_pipeline_reference.global_computation_results.computed_data['DirectionalDecodersDecoded'])
            spikes_df: pd.DataFrame = directional_decoders_decode_result.spikes_df

            for extant_decoded_time_bin_size, a_result_decoded in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.items():
                a_result_decoded: SingleEpochDecodedResult = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]
                a_result_decoded: DecodedFilterEpochsResult = DecodedFilterEpochsResult.init_from_single_epoch_result(single_epoch_result=a_result_decoded, decoding_time_bin_size=extant_decoded_time_bin_size) ## convert to a `DecodedFilterEpochsResult` for masking
                
                if mask_position_like_time_score_cutoff:
                    a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_result_decoded, position_like_score_cutoff=mask_position_like_time_score_cutoff, num_min_position_like_t_bins=None)
                else:
                    a_masked_result = a_result_decoded

                if (min_num_spikes_per_bin_to_be_considered_active is not None) and (min_num_spikes_per_bin_to_be_considered_active > 0):
                    ## TODO: I think this works?
                    a_masked_result, mask_index_tuple = a_masked_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(
                        spikes_df=deepcopy(spikes_df),
                        min_num_spikes_per_bin_to_be_considered_active=min_num_spikes_per_bin_to_be_considered_active,
                        min_num_unique_active_neurons_per_time_bin=1,
                        masked_bin_fill_mode='dropped',
                        # masked_bin_fill_mode='nan_filled'
                    )
                    
                # a_masked_result: DecodedFilterEpochsResult
                # is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list = mask_index_tuple
                ## re-assign to `directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]`
                directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size] = a_masked_result.get_result_for_epoch(0) ## get the single epoch, re-assign

        else:
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = global_computation_results.computed_data['DirectionalDecodersDecoded']
            spikes_df: pd.DataFrame = deepcopy(directional_decoders_decode_result.spikes_df)



        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        
        pos_df: pd.DataFrame = deepcopy(owning_pipeline_reference.sess.position.to_dataframe())
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
        print(F'previously_decoded time_bin_sizes: {previously_decoded_keys}')

        if extant_decoded_time_bin_size is None:
            time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        else:
            assert extant_decoded_time_bin_size in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict, f"extant size wasn't computed!"
            time_bin_size: float = extant_decoded_time_bin_size

        print(f'time_bin_size: {time_bin_size}')

        if drop_previous_result_and_compute_fresh:
            removed_predictive_decoding_result = global_computation_results.computed_data.pop('PredictiveDecoding', None)
            if removed_predictive_decoding_result is not None:
                print(f'removed previous "PredictiveDecoding" result and computing fresh since `drop_previous_result_and_compute_fresh == True`')

        if ('PredictiveDecoding' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'PredictiveDecoding')):
            # initialize
            global_computation_results.computed_data['PredictiveDecoding'] = PredictiveDecodingComputationsContainer(predictive_decoding=None, is_global=True)

        # Create DecodingLocalityMeasures first (required for new interface)
        locality_measures = DecodingLocalityMeasures.init_from_decode_result(
            curr_active_pipeline=owning_pipeline_reference,
            directional_decoders_decode_result=directional_decoders_decode_result,
            extant_decoded_time_bin_size=time_bin_size,
            sigma=None  # Will be computed automatically if not provided
        )
        
        # Compute locality measures to ensure they are fully computed
        # locality_measures.compute()
        # non_local_PBE_non_moving_epochs_df: pd.DataFrame = locality_measures.get_non_moving_PBE_non_local_epochs(owning_pipeline_reference.sess, merging_adjacent_max_separation_sec=0.5)
        
        # Get a_result_decoded from directional_decoders_decode_result
        a_result_decoded = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[time_bin_size]
        
        # masked_result, mask_index_tuple = a_result_decoded.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(
        #     spikes_df=spikes_df,
        #     min_num_spikes_per_bin_to_be_considered_active=5,
        #     min_num_unique_active_neurons_per_time_bin=1,
        #     masked_bin_fill_mode='dropped',
        #     # masked_bin_fill_mode='nan_filled'
        # )
        # masked_result


        #TODO 2025-12-23 20:55: - [ ] Found that everything seems to be working well except that there are sometimes a few time bins out of an epoch that have poorly localized posteriors in general (they look very diffuse and like an error, maybe low firing bins)
        ### These need to be filtered out (either by diffusivity of low-firing criteria) so that when we collapse over all the time bins within each epoch we don't pick up a bunch of garbage (the diffuse bins are too liberal).
        ## INPUTS: spikes_df
        # masked_result, mask_index_tuple = a_result_decoded.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(
        #     spikes_df=spikes_df,
        #     min_num_spikes_per_bin_to_be_considered_active=5,
        #     min_num_unique_active_neurons_per_time_bin=1,
        #     masked_bin_fill_mode='ignore'
        # )

        ## TODO: do something with masked_result, mask_index_tuple
        
        # Create PredictiveDecoding using the new simplified interface
        ## INPUTS: pos_df, locality_measures, ..
        predictive_decoding: PredictiveDecoding = PredictiveDecoding.init_from_decode_result(
            pos_df=pos_df,
            locality_measures=locality_measures,
            a_result_decoded=a_result_decoded,
            window_size=window_size
        )

        # Use sigma from locality_measures (computed automatically) or compute from bin sizes if not available
        if locality_measures.sigma is None:
            x_step: float = np.nanmean(np.diff(predictive_decoding.xbin))
            y_step: float = np.nanmean(np.diff(predictive_decoding.ybin))
            sigma: float = np.nanmax([x_step, y_step]) * 5.0
            print(f'computed sigma from bin sizes: {sigma}')
        else:
            sigma = locality_measures.sigma
            print(f'using sigma from locality_measures: {sigma}')

        # Compute predictive decoding outputs
        moving_avg_dict, moving_avg_meas_pos_overlap_dict, gaussian_volume = predictive_decoding.compute(sigma=sigma)

        # Store the PredictiveDecoding instance in the container
        global_computation_results.computed_data['PredictiveDecoding'].predictive_decoding = predictive_decoding
        
        if include_includelist is None:
            include_includelist = ['roam'] # , 'sprinkle'
        epoch_names: List[str] = include_includelist 
        print(f'\t processing will occur for epoch_names: {epoch_names}')
        for an_epoch_name in epoch_names:    
            try:
                print(f'\ttrying `.compute_future_and_past_analysis(...)` for an_epoch_name: "{an_epoch_name}"...')
                global_computation_results.computed_data['PredictiveDecoding'].debug_computed_dict[an_epoch_name] = {}
                _out = global_computation_results.computed_data['PredictiveDecoding'].compute_future_and_past_analysis(owning_pipeline_reference, an_epoch_name=an_epoch_name)
                epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list = _out
                global_computation_results.computed_data['PredictiveDecoding'].debug_computed_dict[an_epoch_name] = {'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict}
            except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
                print(f'\t\tWARN: the last part of `perform_predictive_decoding_analysis(...) failed with error: {e}. Skipping.')
                pass
            except Exception as e:
                raise e

        # epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list
        print(f'done')


        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures, PredictiveDecodingComputationsContainer
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

        container: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('PredictiveDecoding', None)
        if container is not None:
            predictive_decoding: PredictiveDecoding = container.predictive_decoding
            if predictive_decoding is not None:
                print(f'PredictiveDecoding computed with window_size: {predictive_decoding.window_size}')
                print(f'epoch_names: {predictive_decoding.epoch_names}')

                if container.decoding_locality is None:
                    container.decoding_locality = container.predictive_decoding.locality_measures

                decoding_locality: DecodingLocalityMeasures = container.decoding_locality
            else:
                print(f'PredictiveDecoding is None.')
        else:
            print(f'PredictiveDecoding is not computed.')


        epoch_high_prob_pos_masks = container.debug_computed_dict[an_epoch_name]['epoch_high_prob_pos_masks']
        epoch_matching_positions = container.debug_computed_dict[an_epoch_name]['epoch_matching_positions']
        past_future_info_dict = container.debug_computed_dict[an_epoch_name]['past_future_info_dict']


            
        """
        return global_computation_results
    

   


# ==================================================================================================================== #
# Display Function Helpers                                                                                             #
# ==================================================================================================================== #

# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

# # ==================================================================================================================== #
# # Display Functions                                                                                                    #
# # ==================================================================================================================== #

# class PredictiveDecodingGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
#     """ RankOrderGlobalDisplayFunctions
#     These display functions compare results across several contexts.
#     Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
#     """

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter, DecodedTrajectoryPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

@define(slots=False, repr=False, eq=False)
class PredictiveDecodingDisplayWidget:
    """ Plots 3 panels side-by-side: Left: Past positions, Mid: Decoded Epoch Posterior, Right: Future positions
    
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidget

        a_widget: PredictiveDecodingDisplayWidget = PredictiveDecodingDisplayWidget.init_from_container(container=container, decoding_time_bin_size=0.025, an_epoch_name='roam')
        a_widget
    """
    container: PredictiveDecodingComputationsContainer = field(default=None)
    
    xbin: np.ndarray = field(default=None)
    ybin: np.ndarray = field(default=None)
    xbin_centers: np.ndarray = field(default=None)
    ybin_centers: np.ndarray = field(default=None)
    curr_position_df: pd.DataFrame = field(default=None)
    
    pf1D_Decoder: BasePositionDecoder = field(default=None)
    decoded_result: DecodedFilterEpochsResult = field(default=None)

    ## Display Variables
    trajectory_displaying_plotter: Dict[str, DecodedTrajectoryMatplotlibPlotter] = field(default=Factory(dict))
    
    ## Dock UI Variables
    dock_area: Any = field(default=None)
    dock_window: Any = field(default=None)
    dock_widgets: Dict[str, Any] = field(default=Factory(dict))
    dock_canvas_widgets: Dict[str, Any] = field(default=Factory(dict))
    epoch_slider: Any = field(default=None)
    epoch_value_label: Any = field(default=None)

    active_epoch_idx: int = field(default=20)
    

    @classmethod
    def init_from_container(cls, container: PredictiveDecodingComputationsContainer, decoding_time_bin_size: float, an_epoch_name: str, active_epoch_idx: int=6) -> "PredictiveDecodingDisplayWidget":
        """

        """
        decoded_local_epochs_result = container.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name]
        pf_decoder = container.pf1D_Decoder_dict[an_epoch_name]
        decoded_result: DecodedFilterEpochsResult = decoded_local_epochs_result
        curr_position_df: pd.DataFrame = deepcopy(container.decoding_locality.pos_df)

        # global_session = deepcopy(curr_active_pipeline.sess)
        # a_result2D: DecodedFilterEpochsResult = decoded_local_epochs_result.frame_divided_epochs_results[an_epoch_name]
        # pf_Decoder = container.pf1D_Decoder_dict[an_epoch_name]
        # a_result2D = results2D.a_result2D
        # a_new_global_decoder2D = results2D.a_new_global_decoder2D
        ## INPUTS: directional_laps_results, decoder_ripple_filter_epochs_decoder_result_dict
        xbin = deepcopy(pf_decoder.xbin)
        xbin_centers = deepcopy(pf_decoder.xbin_centers)
        ybin_centers = deepcopy(pf_decoder.ybin_centers)
        ybin = deepcopy(pf_decoder.ybin)

        # num_filter_epochs: int = decoded_local_epochs_result.num_filter_epochs

        _obj = cls(
            container=container,
            xbin=xbin,
            ybin=ybin,
            xbin_centers=xbin_centers,
            ybin_centers=ybin_centers,
            curr_position_df=curr_position_df,
            pf1D_Decoder=pf_decoder, decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx,
        )

        return _obj
    

    def __attrs_post_init__(self):
        merging_adjacent_max_separation_sec: float = 0.5
        minimum_epoch_duration: float = 0.05
        # matching_pos_dfs_list: List[pd.DataFrame] = []
        # matching_pos_epochs_dfs_list: List[pd.DataFrame] = []
        assert len(self.container.predictive_decoding.matching_pos_dfs_list) > 0
        matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        assert len(self.container.predictive_decoding.matching_pos_epochs_dfs_list) > 0
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        ## INPUTS: matching_pos_epochs_dfs_list, decoded_local_epochs_result

        for i, a_row in enumerate(ensure_dataframe(self.decoded_result.filter_epochs).itertuples(index=False)):
        # for i, a_pos_matches_epoch_mask in enumerate(epoch_matching_positions):
            a_matching_pos_epochs: pd.DataFrame = matching_pos_epochs_dfs_list[i]
            # a_row.start, a_row.stop
            curr_epoch_start_t: float = a_row.start
            curr_epoch_stop_t: float = a_row.stop
            
            is_relevant_past_times = (a_matching_pos_epochs['stop'] < curr_epoch_start_t)
            is_relevant_future_times = (a_matching_pos_epochs['start'] > curr_epoch_stop_t)
            a_matching_pos_epochs['is_future_present_past'] = 'present'
            a_matching_pos_epochs.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
            a_matching_pos_epochs.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'
                    
            self.container.predictive_decoding.matching_pos_epochs_dfs_list[i] = a_matching_pos_epochs

        # self.container.predictive_decoding.matching_pos_epochs_dfs_list = matching_pos_epochs_dfs_list
        self.init_UI()


    def init_UI(self):
        """Initialize the UI with DockArea and Dock widgets for past/future trajectory plotters."""
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.External.pyqtgraph.dockarea import DockArea, Dock
        from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtGui, QtCore
        
        # Create the DockArea and window
        self.dock_window = QtWidgets.QMainWindow()
        self.dock_window.setWindowTitle("Predictive Decoding Display - Past/Future Trajectories")
        self.dock_area = DockArea()
        self.dock_window.setCentralWidget(self.dock_area)
        self.dock_window.resize(1400, 800)
        
        # Create docks for past, decoded_posterior, and future (will be populated when update_displayed_epoch is called)
        prev_dock = None
        for category_name in ['past', 'decoded_posterior', 'future']:
            if category_name == 'decoded_posterior':
                dock_name = "Decoded Posterior"
            else:
                dock_name = f"{category_name.capitalize()} Trajectories"
            dock = Dock(dock_name, size=(600, 700), closable=True)
            
            # Create placeholder widget (will be replaced with actual plot in update_displayed_epoch)
            if category_name == 'decoded_posterior':
                placeholder_text = f"Waiting for data...\nCall update_displayed_epoch() to display decoded posterior heatmap."
            else:
                placeholder_text = f"Waiting for data...\nCall update_displayed_epoch() to display {category_name} trajectories."
            placeholder_widget = QtWidgets.QLabel(placeholder_text)
            placeholder_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            dock.addWidget(placeholder_widget)
            
            # Add dock horizontally
            if prev_dock is None:
                self.dock_area.addDock(dock, 'left')
            else:
                self.dock_area.addDock(dock, 'right', prev_dock)
            
            self.dock_widgets[category_name] = dock
            prev_dock = dock
        
        # Create slider widget at the bottom of the window
        num_epochs = len(ensure_dataframe(self.decoded_result.filter_epochs))
        if num_epochs > 0:
            # Create a bottom widget container for the slider
            bottom_widget = QtWidgets.QWidget()
            bottom_layout = QtWidgets.QHBoxLayout()
            bottom_layout.setContentsMargins(10, 5, 10, 5)
            
            # Create label for the slider
            slider_label = QtWidgets.QLabel("Epoch Index:")
            bottom_layout.addWidget(slider_label)
            
            # Create slider
            self.epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.epoch_slider.setMinimum(0)
            self.epoch_slider.setMaximum(max(0, num_epochs - 1))
            self.epoch_slider.setValue(min(self.active_epoch_idx, num_epochs - 1))
            self.epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            self.epoch_slider.setTickInterval(max(1, num_epochs // 20))  # Show ~20 ticks
            self.epoch_slider.setMinimumWidth(400)
            bottom_layout.addWidget(self.epoch_slider, stretch=1)  # Stretch to fill width
            
            # Create label to show current value
            self.epoch_value_label = QtWidgets.QLabel(f"{self.epoch_slider.value()}")
            self.epoch_value_label.setMinimumWidth(50)
            self.epoch_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            bottom_layout.addWidget(self.epoch_value_label)
            
            # Connect slider signals:
            # - valueChanged: Update label only (for immediate feedback while dragging)
            # - sliderReleased: Update display (only when user releases the slider)
            self.epoch_slider.valueChanged.connect(self._on_slider_value_changed_label_only)
            self.epoch_slider.sliderReleased.connect(self._on_slider_released)
            
            bottom_widget.setLayout(bottom_layout)
            
            # Add bottom widget to the main window using a status bar or dock
            # We'll use a dock at the bottom
            bottom_dock = Dock("Epoch Control", size=(1400, 50), closable=False)
            bottom_dock.addWidget(bottom_widget)
            self.dock_area.addDock(bottom_dock, 'bottom')
            self.dock_widgets['epoch_control'] = bottom_dock
        
        self.dock_window.show()
        
        self.update_displayed_epoch(an_epoch_idx=self.active_epoch_idx) ## go to first index if possible

    def _on_slider_value_changed_label_only(self, value: int):
        """Handle slider value change to update only the label (for immediate feedback while dragging)."""
        if self.epoch_value_label is not None:
            self.epoch_value_label.setText(f"{value}")

    def _on_slider_released(self):
        """Handle slider release to update the displayed epoch."""
        if self.epoch_slider is not None:
            value = self.epoch_slider.value()
            self.update_displayed_epoch(an_epoch_idx=value)
            # Update all canvas widgets to ensure they refresh
            for canvas in self.dock_canvas_widgets.values():
                if canvas is not None:
                    canvas.draw_idle()

    def update_displayed_epoch(self, an_epoch_idx: int = 8):
        """ updates the GUI to reflect the epoch idx provided:

        """
        from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets
        import matplotlib.pyplot as plt

        # Initialize UI if not already done
        if self.dock_area is None:
            self.init_UI()

        prev_epoch_idx: int = deepcopy(self.active_epoch_idx)
        
        # Validate epoch index bounds
        num_epochs = len(ensure_dataframe(self.decoded_result.filter_epochs))
        if an_epoch_idx < 0 or an_epoch_idx >= num_epochs:
            print(f"Warning: epoch_idx {an_epoch_idx} is out of bounds (0-{num_epochs-1}). Clamping to valid range.")
            an_epoch_idx = max(0, min(an_epoch_idx, num_epochs - 1))
        
        # Update slider value if it exists (block signals to avoid recursion)
        if self.epoch_slider is not None:
            self.epoch_slider.blockSignals(True)
            self.epoch_slider.setValue(an_epoch_idx)
            self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText(f"{an_epoch_idx}")
                
        assert len(self.container.predictive_decoding.matching_pos_dfs_list) > 0
        matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        assert len(self.container.predictive_decoding.matching_pos_epochs_dfs_list) > 0
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list

        # Calculate maximum number of subplots needed across all epochs for each category
        # This ensures the layout doesn't need to resize when switching between epochs
        max_subplots_per_category: Dict[str, int] = {}
        for epoch_idx in range(len(matching_pos_epochs_dfs_list)):
            curr_matching_epochs_df_temp: pd.DataFrame = matching_pos_epochs_dfs_list[epoch_idx]
            curr_matching_epochs_df_dict_temp: Dict[int, pd.DataFrame] = curr_matching_epochs_df_temp.pho.partition_df_dict('is_future_present_past')
            for a_past_future_name, an_epoch_specific_dfs in curr_matching_epochs_df_dict_temp.items():
                if a_past_future_name not in max_subplots_per_category:
                    max_subplots_per_category[a_past_future_name] = 0
                num_items = len(an_epoch_specific_dfs)
                max_subplots_per_category[a_past_future_name] = max(max_subplots_per_category[a_past_future_name], num_items)
        
        # Cap at 20 subplots maximum
        for key in max_subplots_per_category:
            max_subplots_per_category[key] = min(20, max_subplots_per_category[key])

        curr_matching_epochs_df: pd.DataFrame = matching_pos_epochs_dfs_list[an_epoch_idx]
        curr_matching_positions_df: pd.DataFrame = matching_pos_dfs_list[an_epoch_idx]
        curr_matching_epochs_df_dict: Dict[int, pd.DataFrame] = curr_matching_epochs_df.pho.partition_df_dict('is_future_present_past')

        past_future_names = ['past', 'future']
        curr_matching_past_future_positions_df_dict: Dict[str, Dict[int, pd.DataFrame]] = {}
        # for a_past_future_name in past_future_names:
        for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
            ## add the final detected non_local_pbe_epoch indicies to the decoded points:
            a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
            an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int) ## convert to int
            col_name: str = 'past_future_matching_pos_epoch_id'
            a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name, override_time_variable_name='t',
                                                                # epoch_label_column_name='label', no_interval_fill_value='',
                                                                epoch_label_column_name='label', no_interval_fill_value=-1,
                                                                should_replace_existing_column=True, drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
            curr_matching_positions_df_dict: Dict[int, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name) ## the position dataframes for each possible future/past epoch
            curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
            epoch_specific_position_dfs = list(curr_matching_past_future_positions_df_dict[a_past_future_name].values())
            epoch_ids = np.array(list(curr_matching_past_future_positions_df_dict[a_past_future_name].keys()))
            # epoch_specific_position_dfs
            # Always use the maximum number of subplots for this category (capped at 20)
            curr_num_subplots: int = max_subplots_per_category.get(a_past_future_name, min(20, len(epoch_ids)))
            
            # Pad the lists to match curr_num_subplots if needed (to prevent IndexError)
            # This ensures the layout is consistent even when current epoch has fewer items
            if len(epoch_specific_position_dfs) < curr_num_subplots:
                # Create dummy DataFrames for padding - need at least one row to prevent IndexError
                num_to_pad = curr_num_subplots - len(epoch_specific_position_dfs)
                if len(epoch_specific_position_dfs) > 0:
                    # Use the first DataFrame's structure as a template
                    template_df = epoch_specific_position_dfs[0]
                    # Create a dummy row with NaN values for all columns
                    dummy_row = {col: np.nan for col in template_df.columns}
                    empty_df = pd.DataFrame([dummy_row], columns=template_df.columns)
                else:
                    # Fallback: use common position DataFrame columns with dummy row
                    empty_df = pd.DataFrame([{'t': np.nan, 'x': np.nan, 'y': np.nan, 'binned_x': np.nan, 'binned_y': np.nan}])
                epoch_specific_position_dfs.extend([empty_df.copy() for _ in range(num_to_pad)])
                # Pad epoch_ids with -1 (invalid ID) for empty subplots
                epoch_ids = np.concatenate([epoch_ids, np.full(num_to_pad, -1, dtype=epoch_ids.dtype)])
            # curr_num_subplots: int = 40
            
            # an_epoch_specific_past_position_dfs = curr_matching_epochs_df_dict['past']
            # an_epoch_specific_past_epoch_ids = an_epoch_specific_past_position_dfs.index.to_numpy()
            ## OUTPUTS: an_epoch_specific_past_position_dfs, an_epoch_specific_past_epoch_ids
            existing_ax = None
            needed_init: bool = False
            a_decoded_traj_plotter = self.trajectory_displaying_plotter.get(a_past_future_name, None)
            if a_decoded_traj_plotter is None:
                ## create a new plotter
                a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers)
                self.trajectory_displaying_plotter[a_past_future_name] = a_decoded_traj_plotter
                needed_init = True
            else:
                existing_ax = a_decoded_traj_plotter.axs
                # Clear existing axes before plotting to prevent drawing over previous plots
                if existing_ax is not None:
                    # Handle different axis structures (list, array, or single axis)
                    if isinstance(existing_ax, (list, tuple, np.ndarray)):
                        for ax in existing_ax:
                            if ax is not None and hasattr(ax, 'clear'):
                                ax.clear()
                    elif hasattr(existing_ax, 'clear'):
                        existing_ax.clear()
                    # Also clear the figure if it exists
                    if hasattr(a_decoded_traj_plotter, 'fig') and a_decoded_traj_plotter.fig is not None:
                        # Clear all axes in the figure
                        for ax in a_decoded_traj_plotter.fig.get_axes():
                            ax.clear()

            # canvas: FigureCanvas = self.dock_canvas_widgets.get(a_past_future_name, None)
            # if canvas is not None:
            #     existing_ax = canvas.figure.get_axes() ## a list of 8 Axes objects

            fig, axs, epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=self.curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=epoch_ids, curr_num_subplots=curr_num_subplots, active_page_index=0,
                                                                                     fixed_columns = 4,
                                                                                     plot_actual_lap_lines=True, use_theoretical_tracks_instead=False, existing_ax=existing_ax,
                                                                                     #  plot_mode='line',
                                                                                     plot_mode='scatter', c='red', cmap='Reds', alpha=0.95,
                                                                                     )
            
            perform_update_title_subtitle(fig=fig, ax=None, title_string=f"{a_past_future_name} - an_epoch_idx: {an_epoch_idx}")
            # self.active_epoch_idx = an_epoch_idx

            # Embed the matplotlib figure in the dock widget
            if needed_init:
                dock = self.dock_widgets.get(a_past_future_name)
                if (dock is not None): # (canvas is None)
                    # Remove existing widgets from dock
                    # Dock uses a QGridLayout and maintains a widgets list
                    layout = dock.layout
                    if layout is not None:
                        while layout.count():
                            child = layout.takeAt(0)
                            if child.widget():
                                child.widget().setParent(None)
                    # Clear the widgets list maintained by Dock
                    if hasattr(dock, 'widgets'):
                        dock.widgets.clear()
                    dock.currentRow = 0
                    
                    # Create canvas and toolbar for the matplotlib figure
                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, self.dock_window)
                    
                    # Create a widget to hold canvas and toolbar
                    plot_widget = QtWidgets.QWidget()
                    plot_layout = QtWidgets.QVBoxLayout()
                    plot_layout.setContentsMargins(0, 0, 0, 0)
                    plot_layout.addWidget(toolbar)
                    plot_layout.addWidget(canvas)
                    plot_widget.setLayout(plot_layout)
                    
                    # Add to dock
                    dock.addWidget(plot_widget)
                    
                    # Store reference to canvas widget
                    self.dock_canvas_widgets[a_past_future_name] = canvas
                    
                    # Close the figure window if it's open (since it's now embedded in the dock)
                    plt.close(fig)                

            else:
                ## just redraw - axes are already cleared above before plotting
                canvas = self.dock_canvas_widgets[a_past_future_name]
                if canvas is not None:
                    # The axes were already cleared before plot_decoded_trajectories_2d was called
                    # Just trigger a redraw
                    canvas.draw_idle()

        ### for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matc...

        ## END for a_past_future_name, an_epoch_specific_past_positi...

        # Plot decoded posterior heatmap for 'decoded_posterior' dock
        category_name = 'decoded_posterior'

        posterior_2d = None
        epoch_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_high_prob_pos_masks', None)
        if epoch_high_prob_pos_masks is not None:
            print(f'using high_prob mask version!')
            # posterior_2d = self.container.predictive_decoding.epoch_high_prob_pos_masks[an_epoch_idx]
            posterior_2d = epoch_high_prob_pos_masks[an_epoch_idx]
            
        else:
            ## use posterior:
            p_x_given_n = self.decoded_result.p_x_given_n_list[an_epoch_idx]  # Shape: (n_x_bins, n_y_bins, n_time_bins)
            # Sum over time dimension to create 2D heatmap
            posterior_2d = np.sum(p_x_given_n, axis=2)
            
        
        # Check if we need to initialize (create new figure) or update existing one
        needed_init: bool = category_name not in self.dock_canvas_widgets
        
        if needed_init:
            # Create matplotlib figure for heatmap (using Figure directly to avoid showing in separate window)
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8, 6), layout="constrained")
            ax = fig.add_subplot(111)
            
            # Calculate extent from bin edges (more accurate than using centers)
            extent = [self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1]]
            
            # Plot heatmap
            im = ax.imshow(posterior_2d, aspect='auto', origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Decoded Posterior Heatmap - Epoch {an_epoch_idx}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability (sum over time)')
            
            # Embed the matplotlib figure in the dock widget
            dock = self.dock_widgets.get(category_name)
            if dock is not None:
                # Remove existing widgets from dock
                layout = dock.layout
                if layout is not None:
                    while layout.count():
                        child = layout.takeAt(0)
                        if child.widget():
                            child.widget().setParent(None)
                # Clear the widgets list maintained by Dock
                if hasattr(dock, 'widgets'):
                    dock.widgets.clear()
                dock.currentRow = 0
                
                # Create canvas and toolbar for the matplotlib figure
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, self.dock_window)
                
                # Create a widget to hold canvas and toolbar
                plot_widget = QtWidgets.QWidget()
                plot_layout = QtWidgets.QVBoxLayout()
                plot_layout.setContentsMargins(0, 0, 0, 0)
                plot_layout.addWidget(toolbar)
                plot_layout.addWidget(canvas)
                plot_widget.setLayout(plot_layout)
                
                # Add to dock
                dock.addWidget(plot_widget)
                
                # Store reference to canvas widget
                self.dock_canvas_widgets[category_name] = canvas
                
                # Close the figure window if it's open (since it's now embedded in the dock)
                plt.close(fig)
        else:
            # Update existing canvas
            canvas = self.dock_canvas_widgets.get(category_name)
            if canvas is not None:
                # Clear existing axes and replot
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                # Recalculate extent for update
                extent = [self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1]]
                im = ax.imshow(posterior_2d, aspect='auto', origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'Decoded Posterior Heatmap - Epoch {an_epoch_idx}')
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Probability (sum over time)')
                canvas.draw()

        ## OUTPUTS: curr_matching_past_future_positions_df_dict 
        self.active_epoch_idx = an_epoch_idx
