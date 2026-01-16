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
import neuropy.utils.type_aliases as types # import neuropy.utils.type_aliases as types
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
from neuropy.core.position import PositionAccessor, Position, PositionComputedDataMixin

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
import pyphoplacecellanalysis.General.type_aliases as types

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

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

from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration


# _debug_plot: bool = True
_debug_plot: bool = False


@define(slots=False, repr=False, eq=False)
class EventEpochsDebugger:
    """High-performance visualizer for debugging position filtering steps and event epochs.
    
    Provides utilities to visualize temporal sequences, detect contiguous epochs, and identify gaps
    between epochs at different filtering stages.
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import EventEpochsDebugger
        
        fig = EventEpochsDebugger.visualize_filtering_stages(
            measured_positions_df=measured_positions_df,
            relevant_positions_df_after_merge=relevant_positions_df_after_merge,
            relevant_positions_df_final=relevant_positions_df_final,
            epoch_high_prob_mask=epoch_high_prob_mask,
            curr_epoch_start_t=curr_epoch_start_t,
            curr_epoch_stop_t=curr_epoch_stop_t,
            max_points_per_plot=5000
        )
        import matplotlib.pyplot as plt
        plt.show()
    """
    
    @staticmethod
    def downsample_positions_df(positions_df: pd.DataFrame, max_points: int, preserve_epoch_boundaries: bool = True) -> pd.DataFrame:
        """Downsample position dataframe while preserving epoch boundaries and time structure.
        
        Args:
            positions_df: DataFrame with 't' column and other position data
            max_points: Maximum number of points to keep
            preserve_epoch_boundaries: If True, ensure epoch boundaries are preserved in downsampled data
            
        Returns:
            Downsampled DataFrame
        """
        if len(positions_df) <= max_points:
            return positions_df.copy()
        
        if not preserve_epoch_boundaries:
            # Simple uniform downsampling
            step = len(positions_df) // max_points
            return positions_df.iloc[::step].copy()
        
        # Downsample while preserving epoch boundaries
        # Sort by time first
        df_sorted = positions_df.sort_values('t').reset_index(drop=True)
        
        # Detect gaps to preserve epoch boundaries
        t_values = df_sorted['t'].values
        dt = np.diff(t_values)
        if len(dt) > 0:
            # Use median dt as threshold for gaps (epoch boundaries)
            median_dt = np.nanmedian(dt)
            gap_threshold = median_dt * 10.0  # Gaps larger than 10x median are likely epoch boundaries
            is_gap = dt > gap_threshold
            gap_indices = np.where(is_gap)[0]
        else:
            gap_indices = np.array([], dtype=int)
        
        # Calculate sampling step
        step = max(1, len(df_sorted) // max_points)
        
        # Sample uniformly but always include gap boundaries
        sampled_indices = set(range(0, len(df_sorted), step))
        
        # Add gap boundary indices (and adjacent points)
        for gap_idx in gap_indices:
            sampled_indices.add(gap_idx)
            if gap_idx + 1 < len(df_sorted):
                sampled_indices.add(gap_idx + 1)
            if gap_idx > 0:
                sampled_indices.add(gap_idx - 1)
        
        # Always include first and last points
        sampled_indices.add(0)
        sampled_indices.add(len(df_sorted) - 1)
        
        # Convert to sorted list and sample
        sampled_indices = sorted(sampled_indices)
        return df_sorted.iloc[sampled_indices].copy()


    @staticmethod
    def detect_epochs_and_gaps(positions_df: pd.DataFrame, merging_adjacent_max_separation_sec: float = 0.5) -> Tuple[pd.DataFrame, NDArray]:
        """Detect contiguous epochs and gaps from time points in position dataframe.
        
        Args:
            positions_df: DataFrame with 't' column
            merging_adjacent_max_separation_sec: Maximum separation in seconds for merging adjacent epochs
            
        Returns:
            Tuple of (epochs_df, gap_mask) where:
                - epochs_df: DataFrame with 'start', 'stop', 'duration' columns for each epoch
                - gap_mask: Boolean array indicating which time points are in gaps (False) vs epochs (True)
        """
        if len(positions_df) == 0:
            return pd.DataFrame(columns=['start', 'stop', 'duration']), np.array([], dtype=bool)
        
        df_sorted = positions_df.sort_values('t').reset_index(drop=True)
        t_values = df_sorted['t'].values
        
        if len(t_values) < 2:
            # Single point - treat as one epoch
            epochs_df = pd.DataFrame({
                'start': [t_values[0]],
                'stop': [t_values[0]],
                'duration': [0.0]
            })
            gap_mask = np.array([True])
            return epochs_df, gap_mask
        
        # Compute time differences
        dt = np.diff(t_values)
        median_dt = np.nanmedian(dt) if len(dt) > 0 else 0.0
        gap_threshold = max(merging_adjacent_max_separation_sec, median_dt * 2.0)
        
        # Find gaps (where dt > threshold)
        is_gap = dt > gap_threshold
        gap_start_indices = np.where(is_gap)[0]
        
        # Build epochs from consecutive points
        epoch_starts = []
        epoch_stops = []
        
        if len(gap_start_indices) == 0:
            # Single contiguous epoch
            epoch_starts = [t_values[0]]
            epoch_stops = [t_values[-1]]
        else:
            # First epoch starts at beginning
            epoch_starts.append(t_values[0])
            epoch_stops.append(t_values[gap_start_indices[0]])
            
            # Middle epochs
            for i in range(len(gap_start_indices) - 1):
                epoch_starts.append(t_values[gap_start_indices[i] + 1])
                epoch_stops.append(t_values[gap_start_indices[i + 1]])
            
            # Last epoch
            epoch_starts.append(t_values[gap_start_indices[-1] + 1])
            epoch_stops.append(t_values[-1])
        
        # Create epochs DataFrame
        epochs_df = pd.DataFrame({
            'start': epoch_starts,
            'stop': epoch_stops,
            'duration': [stop - start for start, stop in zip(epoch_starts, epoch_stops)]
        })
        
        # Create gap mask: True for points in epochs, False for points in gaps
        gap_mask = np.ones(len(df_sorted), dtype=bool)
        for gap_idx in gap_start_indices:
            # Mark gap point (the point after the gap start)
            if gap_idx + 1 < len(gap_mask):
                gap_mask[gap_idx + 1] = False
        
        return epochs_df, gap_mask


    @staticmethod
    def visualize_filtering_stages(measured_positions_df: pd.DataFrame, relevant_positions_df_after_merge: pd.DataFrame, relevant_positions_df_final: pd.DataFrame, epoch_high_prob_mask: NDArray, curr_epoch_start_t: float, curr_epoch_stop_t: float, max_points_per_plot: int = 5000, figsize: Tuple[int, int] = (16, 10), ax: Optional[Any] = None, merging_adjacent_max_separation_sec: float = 0.5):
        """High-performance visualizer for debugging position filtering steps.
        
        Shows temporal sequences at each filtering stage, highlighting contiguous epochs and gaps.
        
        Args:
            measured_positions_df: Initial position dataframe (Stage 0)
            relevant_positions_df_after_merge: Positions after merge with epoch mask bins (Stage 1)
            relevant_positions_df_final: Final filtered positions (Stage 2)
            epoch_high_prob_mask: 2D boolean mask for epoch positions
            curr_epoch_start_t: Start time of current epoch
            curr_epoch_stop_t: Stop time of current epoch
            max_points_per_plot: Maximum points to plot per stage (for performance)
            figsize: Figure size tuple
            ax: Optional existing axes (if None, creates new figure)
            merging_adjacent_max_separation_sec: Maximum separation for epoch detection
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Prepare data for each stage
        stages = [
            ('Stage 0: Initial', measured_positions_df, 'blue'),
            ('Stage 1: After Merge', relevant_positions_df_after_merge, 'green'),
            ('Stage 2: Final Filter', relevant_positions_df_final, 'orange'),
        ]
        
        # Create figure and subplots
        if ax is None:
            fig, axes = plt.subplots(len(stages), 1, figsize=figsize, sharex=True)
            if len(stages) == 1:
                axes = [axes]
        else:
            fig = ax.figure
            axes = [ax]
        
        # Get time range for all stages
        all_times = []
        for _, df, _ in stages:
            if len(df) > 0:
                all_times.extend(df['t'].values)
        
        if len(all_times) == 0:
            # Empty data - return empty figure
            return fig
        
        t_min = np.min(all_times)
        t_max = np.max(all_times)
        t_range = t_max - t_min
        t_margin = t_range * 0.02 if t_range > 0 else 1.0
        
        # Plot each stage
        for stage_idx, (stage_name, stage_df, stage_color) in enumerate(stages):
            if stage_idx >= len(axes):
                break
                
            ax_stage = axes[stage_idx]
            
            if len(stage_df) == 0:
                ax_stage.text(0.5, 0.5, f'{stage_name}: No data', transform=ax_stage.transAxes, ha='center', va='center')
                ax_stage.set_ylabel(stage_name)
                continue
            
            # Downsample for performance
            stage_df_downsampled = EventEpochsDebugger.downsample_positions_df(stage_df, max_points_per_plot, preserve_epoch_boundaries=True)
            
            # Detect epochs and gaps
            epochs_df, gap_mask = EventEpochsDebugger.detect_epochs_and_gaps(stage_df_downsampled, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec)
            
            # Sort by time for plotting
            stage_df_sorted = stage_df_downsampled.sort_values('t').reset_index(drop=True)
            t_values = stage_df_sorted['t'].values
            
            # Plot epochs as highlighted regions
            y_pos = 0.5
            y_height = 0.3
            
            for _, epoch_row in epochs_df.iterrows():
                epoch_start = epoch_row['start']
                epoch_stop = epoch_row['stop']
                ax_stage.axvspan(epoch_start, epoch_stop, alpha=0.2, color=stage_color, label='Epoch' if epoch_row.name == 0 else '')
            
            # Plot points
            # Separate points in epochs vs gaps
            in_epoch_mask = gap_mask[:len(stage_df_sorted)]
            if len(in_epoch_mask) < len(stage_df_sorted):
                # Extend mask if needed
                in_epoch_mask = np.pad(in_epoch_mask, (0, len(stage_df_sorted) - len(in_epoch_mask)), constant_values=True)
            
            epoch_times = t_values[in_epoch_mask] if len(in_epoch_mask) > 0 else np.array([])
            gap_times = t_values[~in_epoch_mask] if len(in_epoch_mask) > 0 else np.array([])
            
            # Plot epoch points
            if len(epoch_times) > 0:
                ax_stage.scatter(epoch_times, np.full(len(epoch_times), y_pos), c=stage_color, s=1, alpha=0.6, label='In Epoch' if len(gap_times) > 0 else None)
            
            # Plot gap points (if any)
            if len(gap_times) > 0:
                ax_stage.scatter(gap_times, np.full(len(gap_times), y_pos), c='red', s=1, alpha=0.3, label='Gap')
            
            # Add vertical lines for epoch boundaries
            ax_stage.axvline(curr_epoch_start_t, color='purple', linestyle='--', alpha=0.7, linewidth=1, label='Epoch Start' if stage_idx == 0 else '')
            ax_stage.axvline(curr_epoch_stop_t, color='purple', linestyle='--', alpha=0.7, linewidth=1, label='Epoch Stop' if stage_idx == 0 else '')
            
            # Statistics text
            n_total = len(stage_df)
            n_downsampled = len(stage_df_downsampled)
            n_epochs = len(epochs_df)
            total_epoch_duration = epochs_df['duration'].sum() if len(epochs_df) > 0 else 0.0
            total_time_span = t_max - t_min if t_max > t_min else 0.0
            coverage = (total_epoch_duration / total_time_span * 100) if total_time_span > 0 else 0.0
            
            stats_text = f'n={n_total} (shown: {n_downsampled}) | Epochs: {n_epochs} | Coverage: {coverage:.1f}%'
            ax_stage.text(0.02, 0.95, stats_text, transform=ax_stage.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Formatting
            ax_stage.set_ylabel(stage_name, fontsize=10)
            ax_stage.set_ylim(0, 1)
            ax_stage.set_xlim(t_min - t_margin, t_max + t_margin)
            ax_stage.grid(True, alpha=0.3)
            if stage_idx == 0:
                ax_stage.legend(loc='upper right', fontsize=8)
        
        # Set x-axis label on bottom subplot
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        
        # Title
        fig.suptitle('Position Filtering Stages: Temporal Sequences and Epochs', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        return fig


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
    interpolator: Optional[interp1d] = non_serialized_field(default=None, is_computable=False)
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


    @property
    def measured_positions_df(self) -> pd.DataFrame:
        """ the processed position_df 
        """
        measured_positions_df: pd.DataFrame = self.pos_df.copy()
        # measured_positions_df = measured_positions_df.drop(columns=['binned_x', 'binned_y'], inplace=False)
        measured_positions_df = measured_positions_df.dropna(how='any', subset=['t', 'x', 'y'])
        measured_positions_df = measured_positions_df.position.adding_binned_position_columns(xbin_edges=self.xbin, ybin_edges=self.ybin)
        measured_positions_df = measured_positions_df[(measured_positions_df['binned_x'].notna()) & (measured_positions_df['binned_y'].notna())] # Filter rows based on columns: 'binned_x', 'binned_y'
        # decoding_locality.pos_df = measured_positions_df
        return measured_positions_df



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
    def compute_locality_measures_for_posterior(cls, a_p_x_given_n: NDArray, xbin_centers: NDArray, ybin_centers: NDArray, gaussian_volume: NDArray=None, n_total_pos_bins: Optional[int] = None, min_val_epsilon: float = 1e-9, alpha_list = [0.8], enable_debug_outputs: bool = True, earthmovers_fn: Optional[Callable] = None, debug_print: bool=False) -> Dict[str, Any]:
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
        # Remove non-serialized fields
        _non_pickled_fields = ['interpolator', 'defer_compute_on_init']
        for a_non_pickleable_field in _non_pickled_fields:
            if a_non_pickleable_field in state:
                del state[a_non_pickleable_field]

        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # Restore defaults for non-serialized fields
        _non_pickled_field_restore_defaults = dict(zip(['interpolator', 'defer_compute_on_init'], [None, False]))
        for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
            if a_field_name not in state:
                state[a_field_name] = a_default_restore_value

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
        non_init_subset=['_VersionedResultMixin_version', 'interpolator', 'locality_measures_df',
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


from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PosteriorMaskPostProcessing ## used by `MatchingPastFuturePositionsResult`

@define(slots=False, repr=False, eq=False)
class MatchingPastFuturePositionsResult(ComputedResult):
    """Result container for matching past/future positions in a single decoded epoch (with potentially several time bins).
    
    Attributes:
        epoch_high_prob_mask: 2D boolean mask (N_XBINS, N_Y_BINS) indicating high probability positions during the epoch
        pos_matches_epoch_mask: Indices of positions that match the epoch mask
        relevant_positions_df: DataFrame with relevant positions categorized as past/present/future
        is_relevant_past_times: Boolean mask for past times in relevant_positions_df
        is_relevant_future_times: Boolean mask for future times in relevant_positions_df
        n_total_possible_past_times: Total count of possible past times
        n_total_possible_future_times: Total count of possible future times
        n_relevant_past_times: Count of relevant past times
        n_relevant_future_times: Count of relevant future times
        matching_pos_epochs_df: DataFrame with detected epochs categorized as past/present/future
        
        #TODO 2026-01-14 17:05: - [ ] Refactor to what the class should logically have:
             a_past_future_result - represents a single epoch_idx's values

            Currently work around by doing `MatchingPastFuturePositionsResult.extract_final_position_epochs(...)`
                _out_epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult] = _out_epoch_flat_mask_future_past_result
                _out_dict = MatchingPastFuturePositionsResult.extract_final_position_epochs(_out_epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result)

        #TODO 2026-01-14 17:05: - [ ] Should have
        the posterior masks - [X] in `epoch_high_prob_mask`, note doesn't have `epoch_t_bins_high_prob_mask`
        the epoch time tims, maybe a SingleEpochDecodedResult or what not
        the list of found past and future position epochs
        the list of found past and future position_dfs 

    """
    _VersionedResultMixin_version: str = "2026.01.15_0" # to be updated in your IMPLEMENTOR to indicate its version

    decoded_epoch_result: SingleEpochDecodedResult = serialized_field(repr=False, metadata={'field_added':"2026.01.14_0"})
    epoch_high_prob_mask: NDArray[ND.Shape["N_XBINS, N_Y_BINS"], Any] = serialized_field(repr=False)
    epoch_t_bins_high_prob_pos_mask: NDArray[ND.Shape["N_XBINS, N_Y_BINS, N_EPOCH_TIME_BINS"], Any] = serialized_field(repr=False, metadata={'field_added':"2026.01.14_0"})
    
    relevant_past_times: NDArray = serialized_field(repr=False, metadata={'field_added':"2026.01.14_0"})
    relevant_future_times: NDArray = serialized_field(repr=False, metadata={'field_added':"2026.01.14_0"})

    # Computed mask properties ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    centroids: NDArray[ND.Shape["N_EPOCH_TIME_BINS, 2"], Any] = serialized_field(default=None, is_computable=True, repr=False, metadata={'field_added':"2026.01.14_0"})
    centroids_df: pd.DataFrame = serialized_field(default=None, is_computable=True, repr=False, metadata={'field_added':"2026.01.14_0"})
    a_centroids_search_segments_df: Optional[pd.DataFrame] = non_serialized_field(default=None, is_computable=True, repr=False, metadata={'field_added':"2026.01.14_0"})

    epoch_id_key_name: str = serialized_attribute_field(default='matching_found_relevant_pos_epoch', is_computable=False, repr=False, metadata={'field_added':"2026.01.14_0"})

    # OLD/COMPATIBILITY FIELDS ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    pos_matches_epoch_mask: NDArray = serialized_field(repr=False)
    relevant_positions_df: pd.DataFrame = serialized_field(repr=False) ## !IMPORTANT: `relevant_positions_df`: the df of all potentially relevant positions, with a 'matching_found_relevant_pos_epoch' column corresponding to the the found *epochs* (not some aren't in an epoch and have a value of -1 for this column)
    
    is_relevant_past_times: NDArray = serialized_field(repr=False)
    is_relevant_future_times: NDArray = serialized_field(repr=False)
    n_total_possible_past_times: int = serialized_field()
    n_total_possible_future_times: int = serialized_field()
    n_relevant_past_times: int = serialized_field()
    n_relevant_future_times: int = serialized_field()
    matching_pos_epochs_df: pd.DataFrame = serialized_field(repr=False) ## !IMPORTANT: `matching_pos_epochs_df`: the df of found *epochs* corresponding to the position sequences in `relevant_positions_df`

    # Computed fields ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    matching_past_positions_df: pd.DataFrame = serialized_field(default=None, is_computable=True, repr=False)
    matching_future_positions_df: pd.DataFrame = serialized_field(default=None, is_computable=True, repr=False)
    pos_segment_to_centroid_seq_segment_idx_map: Optional[Dict] = non_serialized_field(default=Factory(dict), is_computable=True, repr=False, metadata={'field_added':"2026.01.14_0"})
    
    should_defer_extended_computations: bool = serialized_attribute_field(default=True, metadata={'field_added':"2026.01.15_0"})


    @property
    def matching_past_position_df_list(self) -> List[pd.DataFrame]:
        """The matching_past_position_df_list property. #TODO 2026-01-15 06:18: - [ ] WRONG """
        # epoch_only_relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_pos_epochs_df, relevant_positions_df=self.relevant_positions_df,
        #                                                                                                     drop_non_epoch_events=True, epoch_id_key_name=epoch_id_key_name) ## drop those that aren't in the epochs
        epoch_only_relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_past_positions_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=True, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs
        unique_values, partitioned_dfs_list = epoch_only_relevant_positions_df.pho.partition_df(partitionColumn=self.epoch_id_key_name)
        # return self.matching_past_positions_df.pho.partition_dict('')
        return partitioned_dfs_list
    
    @property
    def matching_future_position_df_list(self) -> List[pd.DataFrame]:
        """The matching_future_position_df_list property. #TODO 2026-01-15 06:18: - [ ] WRONG """
        epoch_only_relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_future_positions_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=True, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs
        unique_values, partitioned_dfs_list = epoch_only_relevant_positions_df.pho.partition_df(partitionColumn=self.epoch_id_key_name)
        return partitioned_dfs_list


    def __attrs_post_init__(self):
        # Add post-init logic here
        # Instead of building the tuple, create clean dataframes:
        # self.matching_past_positions_df = self.relevant_positions_df[self.is_relevant_past_times].copy()
        # self.matching_future_positions_df = self.relevant_positions_df[self.is_relevant_future_times].copy()
        
        if not self.should_defer_extended_computations:
            self.recompute_all()


    # ==================================================================================================================================================================================================================================================================================== #
    # Computation Functions                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #

    def recompute_all(self):
        """ performs all recomputations like it would at init if `self.should_defer_extended_computations` were not False """    
        self._recompute_all_pos_dfs()
        if (self.epoch_t_bins_high_prob_pos_mask) is not None and (self.decoded_epoch_result is not None):
            self._recompute_high_prob_mask_centroids()
        self.recompute_relevant_position_active_mask_centroid_traj_angle()



    def _recompute_high_prob_mask_centroids(self, disable_segmentation: bool = True):
        """ recomputes the centroid masks for comparing position sequences to high-prob sweep sequences
            Needs to be ran on update:
                self.epoch_t_bins_high_prob_pos_mask, self.relevant_future_times
        """
        if self.epoch_t_bins_high_prob_pos_mask is None:
            return  # Cannot compute without mask
        if self.decoded_epoch_result is None:
            return  # Cannot compute without decoded result
        
        self.centroids = PosteriorMaskPostProcessing.centroids_from_binary_stack(self.epoch_t_bins_high_prob_pos_mask)
        self.centroids_df = PosteriorMaskPostProcessing.centroid_df_from_binary_stack(mask_stack=self.epoch_t_bins_high_prob_pos_mask, time_window_centers=self.decoded_epoch_result.time_window_centers)
        self.centroids_df = self.centroids_df.position.adding_segmented_trajectories_columns(overwrite_existing=True, disable_segmentation=disable_segmentation)
        # Performed 5 aggregations grouped on column: 'segment_idx'
        a_centroids_segments_df = self.centroids_df.groupby(['segment_idx']).agg(segment_dir_angle_binned_mean=('segment_dir_angle_binned', 'mean'), segment_Vp_scatteredness_mean=('segment_Vp_scatteredness', 'mean'), segment_Vp_deg_mean=('segment_Vp_deg', 'mean'), approx_head_dir_degrees_mean=('approx_dir_degrees', 'mean'), Vp_mean=('Vp', 'mean'), segment_Vp_deg_safe_mean=('segment_Vp_deg', PositionComputedDataMixin.circular_mean_deg)).reset_index()
        self.a_centroids_search_segments_df = a_centroids_segments_df.dropna(subset=['segment_dir_angle_binned_mean'], inplace=False)



    def _recompute_all_pos_dfs(self):
        """ Needs to be ran on update:
            self.relevant_past_times, self.relevant_future_times
        """
        # Initialize all segmented trajectory and matching angle columns to their "no_value" defaults
        if 'segment_idx' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['segment_idx'] = -1
        if 'Vp' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['Vp'] = np.nan
        if 'segment_Vp_deg' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['segment_Vp_deg'] = np.nan
        if 'segment_dir_angle_binned' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['segment_dir_angle_binned'] = np.nan
        if 'segment_Vp_scatteredness' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['segment_Vp_scatteredness'] = np.nan
        if 'centroid_pos_traj_matching_angle_idx' not in self.relevant_positions_df.columns:
            self.relevant_positions_df['centroid_pos_traj_matching_angle_idx'] = -1


        # recompute pos_epochs _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        
        ## NOTE: some have negative duration, they overlap, and all sorts of other confusing things...
        # self.matching_pos_epochs_df, curr_matching_positions_df_dict = MatchingPastFuturePositionsResult._custom_build_sequential_position_epochs(matching_past_positions_df=self.relevant_positions_df) # curr_matching_positions_df_dict: types.epoch_index
        self.matching_pos_epochs_df = self.compute_matching_pos_epochs_df(self.relevant_positions_df, disable_segmentation=True) 

        ## re-index:
        self.relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_pos_epochs_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=False, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs

        if self.relevant_past_times is not None:
            # self.matching_past_positions_df = self.relevant_positions_df.epochs.matching_epoch_times_slice(epoch_times=self.relevant_past_times).copy() # , t_column_names=['t']) ## AttributeError: Must have 'start' column.
            # self.matching_past_positions_df = self.relevant_positions_df[np.isin(self.relevant_positions_df['t'], self.relevant_past_times)]
            # self.matching_past_positions_df = self.relevant_positions_df[np.logical_and((self.relevant_positions_df['is_future_present_past'] == 'past'), (self.relevant_positions_df['is_included']))].copy()
            self.matching_past_positions_df = self.relevant_positions_df[(self.relevant_positions_df['is_future_present_past'] == 'past')].copy()

        # self.matching_past_positions_df = self.relevant_positions_df[self.is_relevant_past_times].copy() # OLD

        if self.relevant_future_times is not None:
            # self.matching_future_positions_df = self.relevant_positions_df.epochs.matching_epoch_times_slice(epoch_times=self.relevant_future_times).copy()
            # self.matching_future_positions_df = self.relevant_positions_df[np.logical_and((self.relevant_positions_df['is_future_present_past'] == 'future'), (self.relevant_positions_df['is_included']))].copy()
            self.matching_future_positions_df = self.relevant_positions_df[(self.relevant_positions_df['is_future_present_past'] == 'future')].copy()

        # self.matching_future_positions_df = self.relevant_positions_df[self.is_relevant_future_times].copy() ## OLD


    @classmethod
    def _recompute_relevant_pos_epoch_position_df_index_column(cls, a_matching_pos_epochs_df: pd.DataFrame, relevant_positions_df: pd.DataFrame, epoch_id_key_name='matching_found_relevant_pos_epoch', drop_non_epoch_events: bool = True) -> pd.DataFrame:
        """ add the final detected `a_matching_pos_epochs_df` indicies to the decoded positions as the column ['matching_found_relevant_pos_epoch'] and return the modified `relevant_positions_df`
        
        Usage:
            ## INPUTS: a_matching_pos_epochs_df, relevant_positions_df
            ## add the final detected a_matching_pos_epochs_df indicies to the decoded positions as the column ['matching_found_relevant_pos_epoch']:
            epoch_id_key_name: str = 'matching_found_relevant_pos_epoch'
            relevant_positions_df = MatchingPastFuturePositionsResult._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=a_matching_pos_epochs_df, relevant_positions_df=relevant_positions_df,
                                                                                                                            drop_non_epoch_events=False, epoch_id_key_name=epoch_id_key_name) ## don't drop yet so we have all the events for the object creation

            ## to drop post-hoc - TODO need to change to -1 if that's the `no_interval_fill_value=-1`
            relevant_positions_df = relevant_positions_df.dropna(how='any', subset=[epoch_id_key_name], inplace=False)
            
            
            epoch_times = relevant_positions_df['t'].to_numpy()
            time_to_idx_map = EpochHelpers.find_epoch_times_to_data_indicies_map(a_matching_pos_epochs_df, epoch_times)
        
        """
        ## INPUTS: a_matching_pos_epochs_df, relevant_positions_df
        ## 
        if 'label' not in a_matching_pos_epochs_df.columns:
            a_matching_pos_epochs_df['label'] = a_matching_pos_epochs_df.index.astype(int)
        else:
            a_matching_pos_epochs_df['label'] = a_matching_pos_epochs_df['label'].astype(int)
            
        relevant_positions_df = relevant_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=a_matching_pos_epochs_df, epoch_id_key_name=epoch_id_key_name, override_time_variable_name='t',
                                                            # epoch_label_column_name='label', no_interval_fill_value=np.nan,
                                                            epoch_label_column_name='label', no_interval_fill_value=-1,
                                                            should_replace_existing_column=True, drop_non_epoch_events=drop_non_epoch_events,
                                                            overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH) ## #TODO 2026-01-15 06:13: - [ ] KeyError: "None of [Index(['start', 'stop'], dtype='object')] are in the [columns]"
        

        # TODO:_custom_build_sequential_position_epochs

        return relevant_positions_df
    

    @function_attributes(short_name=None, tags=['traj', 'angle', 'direction'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 19:36', related_items=[])
    def recompute_relevant_position_active_mask_centroid_traj_angle(self, disable_segmentation: bool=True):
        """Recompute relevant position active mask centroid trajectory angle matching.
        
        This function processes positions within epochs to compute segmented trajectory columns
        and matches them to centroid search segments based on angle similarity. The computed
        columns are then assigned to `self.relevant_positions_df`.
        
        Updates the following columns in `self.relevant_positions_df`:
        - 'segment_idx': Segment index (initialized to -1 if not in epoch)
        - 'Vp': Velocity magnitude
        - 'segment_Vp_deg': Segment velocity direction in degrees
        - 'segment_dir_angle_binned': Binned direction angle for segment
        - 'segment_Vp_scatteredness': Velocity scatteredness measure for segment
        - 'centroid_pos_traj_matching_angle_idx': Index of matching centroid segment (initialized to -1 if no match)
        
        Returns:
            tuple: (epoch_only_relevant_positions_df, pos_segment_to_centroid_seq_segment_idx_map) or (None, None)
        """
        #TODO 2026-01-14 17:53: - [ ] `PosteriorMaskPostProcessing` post processing positions to see which are aligned with the posterior:
        if (self.a_centroids_search_segments_df is None):
            self._recompute_high_prob_mask_centroids(disable_segmentation=disable_segmentation) ## try once to recompute
        if (self.a_centroids_search_segments_df is not None):
            ## compute the matching position/angles:
            epoch_only_relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_pos_epochs_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=True, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs
            epoch_only_relevant_positions_df = epoch_only_relevant_positions_df.position.adding_segmented_trajectories_columns(overwrite_existing=True, disable_segmentation=disable_segmentation)
            # COULD do: `.position.adding_representitive_trajectories_angles_columns(...)` instead`

            # Assign the segmented trajectory columns from epoch_only_relevant_positions_df to self.relevant_positions_df
            segmented_traj_columns = ['segment_idx', 'Vp', 'segment_Vp_deg', 'segment_dir_angle_binned', 'segment_Vp_scatteredness']
            for col in segmented_traj_columns:
                if col in epoch_only_relevant_positions_df.columns:
                    if col not in self.relevant_positions_df.columns:
                        # Initialize with NaN for columns that might not exist, or appropriate default
                        if col == 'segment_idx':
                            self.relevant_positions_df[col] = -1
                        else:
                            self.relevant_positions_df[col] = np.nan
                    # Update values from epoch_only_relevant_positions_df by matching on index
                    self.relevant_positions_df.loc[epoch_only_relevant_positions_df.index, col] = epoch_only_relevant_positions_df[col]

            epoch_only_relevant_positions_df, pos_segment_to_centroid_seq_segment_idx_map = PosteriorMaskPostProcessing._compare_centroid_and_pos_traj_angle(a_pos_df=epoch_only_relevant_positions_df, a_centroids_search_segments_df=self.a_centroids_search_segments_df, disable_segmentation=disable_segmentation)
            
            if pos_segment_to_centroid_seq_segment_idx_map is not None:
                self.pos_segment_to_centroid_seq_segment_idx_map = pos_segment_to_centroid_seq_segment_idx_map

            # Assign the 'centroid_pos_traj_matching_angle_idx' column from epoch_only_relevant_positions_df to self.relevant_positions_df
            # Initialize the column if it doesn't exist
            if 'centroid_pos_traj_matching_angle_idx' not in self.relevant_positions_df.columns:
                self.relevant_positions_df['centroid_pos_traj_matching_angle_idx'] = -1
            # Update values from epoch_only_relevant_positions_df by matching on index
            self.relevant_positions_df.loc[epoch_only_relevant_positions_df.index, 'centroid_pos_traj_matching_angle_idx'] = epoch_only_relevant_positions_df['centroid_pos_traj_matching_angle_idx']

            return epoch_only_relevant_positions_df, pos_segment_to_centroid_seq_segment_idx_map
        else:
            # INSERT_YOUR_CODE
            import warnings
            warnings.warn("Warning: `self.a_centroids_search_segments_df` is None. Computation of centroid/position matching skipped.", UserWarning)
            return None, None


    @property
    def epoch_mask_included_binned_x_y_columns_idx_df(self) -> pd.DataFrame:
        """
        DataFrame containing the unique binned_x, binned_y position pairs that are included in the epoch mask.
        
        This property computes the equivalent of `an_epoch_mask_included_binned_x_y_columns_idx_df` from 
        detect_matching_past_future_positions (lines 1764-1766). It extracts the unique spatial positions
        (binned_x, binned_y) that match the epoch's high-probability mask.
        
        Returns:
            DataFrame with columns ["binned_x", "binned_y"] containing unique position pairs in the epoch mask.
            Sorted by binned_x then binned_y for consistency.
        """
        # Compute from stored epoch_high_prob_mask (exact equivalent of original computation)
        row_col_indices = np.argwhere(self.epoch_high_prob_mask)
        row_col_row_ids = row_col_indices + 1
        an_epoch_mask_included_binned_x_y_columns_idx_df = pd.DataFrame(row_col_row_ids, columns=["binned_x", "binned_y"])
        return an_epoch_mask_included_binned_x_y_columns_idx_df.sort_values(by=["binned_x", "binned_y"]).reset_index(drop=True)


    @classmethod
    def compute_matching_pos_epochs_df(cls, measured_positions_df: pd.DataFrame, merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, disable_segmentation=True, **kwargs) -> pd.DataFrame:
        """
        Compute matching position epochs DataFrame from position matches and time filters.
        
        Args:
            measured_positions_df: DataFrame with position data. Should already be filtered to only include past/future positions (not present positions).
            merging_adjacent_max_separation_sec: Maximum separation in seconds for merging adjacent epochs
            minimum_epoch_duration: Minimum duration for detected epochs
            
        Returns:
            DataFrame with detected epochs categorized as past/present/future

            a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls.compute_matching_pos_epochs_df(measured_positions_df, disable_segmentation=disable_segmentation, **kwargs)
            
        """
        ## find adjacent epochs from the position time bins (periods where the animal is in the positions)
        # measured_positions_df_copy = measured_positions_df.copy()
        # assert 'is_included' in measured_positions_df_copy

        # a_matching_pos_epochs_df: pd.DataFrame = measured_positions_df_copy.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (measured_positions_df_copy['is_included'].to_numpy()), merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration)        
        a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls._custom_build_sequential_position_epochs(matching_past_positions_df=measured_positions_df, disable_segmentation=disable_segmentation, **kwargs) ## dataframe is already filtered to past/future positions before being passed

        ## Copied from `.neuropy.detect_epoch_satisfying_condition(...)``
        if merging_adjacent_max_separation_sec is not None:
            a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec) ## Loses other columns!
        if minimum_epoch_duration is not None:
            a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_epochs_longer_than(minimum_duration=minimum_epoch_duration)
        if merging_adjacent_max_separation_sec is not None:
            a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec) ## Loses other columns!
            
        a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.rebuild_labels_column()
        
        ## #TODO 2026-01-14 18:09: - [ ] Add the relevant epoch idx to the `measured_positions_df`
        return a_matching_pos_epochs_df


    def filter_positions_to_epoch_mask_included_bins(self, a_pos_df: pd.DataFrame) -> pd.DataFrame:
        """ filter to the epoch's bins """
        ## allowed positions are much less than the found ones:
        return a_pos_df.merge(self.epoch_mask_included_binned_x_y_columns_idx_df, on=["binned_x", "binned_y"], how="inner")


    @function_attributes(short_name=None, tags=['FIXED', 'WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-15 09:50', related_items=[])
    @classmethod
    def _custom_build_sequential_position_epochs(cls, matching_past_positions_df: pd.DataFrame, col_name: str = 'past_future_matching_pos_epoch_id', EPSILON_GAP_SIZE_SEC: float = 1e-9, disable_segmentation: bool = True) -> Tuple[pd.DataFrame, Dict[types.epoch_index, pd.DataFrame]]:
        """ builds the epochs_df from the positions_df for a single epoch by merging consecutive time bins into epochs.

        Identifies consecutive sequences of time bins (gaps <= dt_max) and returns epochs spanning each sequence.
        """
        if len(matching_past_positions_df) < 1:
            print(f'warn: empty df!')
            return pd.DataFrame({}), {}

        df = matching_past_positions_df.copy()
        assert 't' in df

        # Compute bin size from minimum consecutive gap
        t_sorted = np.sort(df['t'].values)
        pos_t_bin_sample_size_sec: float = np.nanmin(np.abs(np.diff(t_sorted)))
        dt_max: float = pos_t_bin_sample_size_sec * 2.5

        # Identify sequences FIRST by detecting gaps > dt_max
        df = df.sort_values('t').reset_index(drop=True)
        df['sequence_id'] = (df['t'].diff() > dt_max).cumsum()

        # Build epochs by aggregating each sequence - use first/last 't' values
        new_pos_epochs: pd.DataFrame = df.groupby('sequence_id').agg(start=('t', 'first'), stop=('t', 'last'), t_count=('t', 'count'), start_pos_idx=('t', 'idxmin'), stop_pos_idx=('t', 'idxmax')).reset_index()
        # Extend stop by bin_size (last 't' is start of last bin, not end)
        new_pos_epochs['stop'] = new_pos_epochs['stop'] + pos_t_bin_sample_size_sec - EPSILON_GAP_SIZE_SEC
        new_pos_epochs['duration'] = new_pos_epochs['stop'] - new_pos_epochs['start']
        new_pos_epochs['label'] = new_pos_epochs['sequence_id'].astype(int)

        # Assign sequence_id back to positions for partitioning
        a_curr_matching_positions_df = df.copy()
        a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=new_pos_epochs, epoch_id_key_name=col_name, override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True, drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)

        ## Segment trajectories
        a_curr_matching_positions_df = a_curr_matching_positions_df.position.adding_segmented_trajectories_columns(disable_segmentation=disable_segmentation)

        curr_matching_positions_df_dict: Dict[types.epoch_index, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)

        return new_pos_epochs, curr_matching_positions_df_dict



    @function_attributes(short_name=None, tags=['WORKAROUND', 'interim', 'temporary'], input_requires=[], output_provides=[], uses=['cls._custom_build_sequential_position_epochs'], used_by=[], creation_date='2026-01-14 19:38', related_items=[])
    @classmethod
    def extract_final_position_epochs(cls, _out_epoch_flat_mask_future_past_result: List["MatchingPastFuturePositionsResult"], disable_segmentation=True, **kwargs):
        """ ran post-hoc to recompute/extract the valid position epochs 
        """
        # new_pos_epochs, curr_matching_positions_df_dict = _subfn_custom_build_sequential_position_epochs(matching_past_positions_df=matching_past_positions_df)
        # # new_pos_epochs
        # curr_matching_positions_df_dict

        _out_added_original_positions_df_dict: Dict[types.PastFutureCategory, List[pd.DataFrame]] = {'past': [], 'future': []}
        _out_added_epochs_df_dict: Dict[types.PastFutureCategory, List[pd.DataFrame]] = {'past': [], 'future': []}
        _out_added_pos_dfs_dict: Dict[types.PastFutureCategory, List[Dict[types.epoch_index, pd.DataFrame]]] = {'past': [], 'future': []}
        _out_num_epochs_added: List[List[int]] = []

        for an_epoch_idx, an_epoch_past_future_result in enumerate(_out_epoch_flat_mask_future_past_result):
            # an_epoch_past_future_result: MatchingPastFuturePositionsResult = _out_epoch_flat_mask_future_past_result[an_epoch_idx]

            ## these results are already filtered to the valid positions only
            matching_past_positions_df = an_epoch_past_future_result.filter_positions_to_epoch_mask_included_bins(a_pos_df=an_epoch_past_future_result.matching_past_positions_df.copy())
            matching_future_positions_df = an_epoch_past_future_result.filter_positions_to_epoch_mask_included_bins(a_pos_df=an_epoch_past_future_result.matching_future_positions_df.copy())

            a_past_future_positions_df_dict = {'past': matching_past_positions_df, 'future': matching_future_positions_df, }
            _temp_num_new_epochs_list: List[int] = []

            for a_past_future_label, a_matching_pos_df in a_past_future_positions_df_dict.items():

                _out_added_original_positions_df_dict[a_past_future_label].append(a_matching_pos_df)
                ## perform the build
                # new_pos_epochs, curr_matching_positions_df_dict = cls._custom_build_sequential_position_epochs(matching_past_positions_df=a_matching_pos_df) # curr_matching_positions_df_dict: types.epoch_index
                new_pos_epochs, curr_matching_positions_df_dict = cls.compute_matching_pos_epochs_df(a_matching_pos_df, disable_segmentation=disable_segmentation, **kwargs) # curr_matching_positions_df_dict: types.epoch_index

                num_new_epochs: int = len(new_pos_epochs)
                _temp_num_new_epochs_list.append(num_new_epochs)
                ## add to output arrays:
                _out_added_epochs_df_dict[a_past_future_label].append(new_pos_epochs)
                _out_added_pos_dfs_dict[a_past_future_label].append(curr_matching_positions_df_dict)

            _out_num_epochs_added.append(_temp_num_new_epochs_list)


        _out_num_epochs_added_df: pd.DataFrame = pd.DataFrame(_out_num_epochs_added, columns=['past', 'future'])

        """ Unpacking:
            added_original_positions_df_dict: Dict[types.PastFutureCategory, List[pd.DataFrame]] = _out_dict['added_original_positions_df_dict']
            added_epochs_df_dict: Dict[types.PastFutureCategory, List[pd.DataFrame]] = _out_dict['added_epochs_df_dict']
            added_pos_dfs_dict: Dict[types.PastFutureCategory, List[Dict[types.epoch_index, pd.DataFrame]]] = _out_dict['added_pos_dfs_dict']
            num_epochs_added: pd.DataFrame = _out_dict['num_epochs_added']
            num_epochs_added

        """
        return {
            'added_original_positions_df_dict': _out_added_original_positions_df_dict,
            'added_epochs_df_dict': _out_added_epochs_df_dict,
            'added_pos_dfs_dict': _out_added_pos_dfs_dict,
            'num_epochs_added': _out_num_epochs_added_df,
        }


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove non-serialized fields
        _non_pickled_fields = ['pos_segment_to_centroid_seq_segment_idx_map', 'a_centroids_search_segments_df'] # 'active_epochs_df',
        for a_non_pickleable_field in _non_pickled_fields:
            if a_non_pickleable_field in state:
                del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        """ 
            ## remove from .set_state()
            # ['pos_segment_to_centroid_seq_segment_idx_map', 'epoch_id_key_name', 'a_centroids_search_segments_df', 'centroids_df', 'centroids', 'relevant_future_times', 'relevant_past_times', 'epoch_t_bins_high_prob_pos_mask', 'decoded_epoch_result', ],
            # ({}, 'matching_found_relevant_pos_epoch', {}, None, None, None, None, None, None, None) 

        """
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # Restore defaults for non-serialized fields
        _non_pickled_field_restore_defaults = dict(zip(['pos_segment_to_centroid_seq_segment_idx_map', 'epoch_id_key_name', 'a_centroids_search_segments_df', 'centroids_df', 'centroids', 'relevant_future_times', 'relevant_past_times', 'epoch_t_bins_high_prob_pos_mask', 'decoded_epoch_result', 'should_defer_extended_computations'], [{}, 'matching_found_relevant_pos_epoch', {}, None, None, None, None, None, None, True]))
        for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
            if a_field_name not in state:
                state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)


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
    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION:bool=None, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION=OVERRIDE_ALLOW_GLOBAL_NESTED_EXPANSION, **kwargs)
        





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

    # epoch_matching_past_future_positions: List of tuples, one per epoch. Each tuple contains 6 elements:
    #   [0]: NDArray - Indices of past positions matching the epoch's high-probability mask
    #   [1]: NDArray - Indices of future positions matching the epoch's high-probability mask
    #   [2]: int - Total count of possible past time points (all positions before epoch start)
    #   [3]: int - Total count of possible future time points (all positions after epoch stop)
    #   [4]: int - Count of relevant past times (past positions that match the epoch mask)
    #   [5]: int - Count of relevant future times (future positions that match the epoch mask)
    epoch_matching_past_future_positions: List[Tuple[NDArray, NDArray, int, int, int, int]] = serialized_field(default=Factory(list), metadata={'date_added': '2025.12.22_0'})
    # matching_pos_dfs_list: List of DataFrames, one per epoch. Each DataFrame contains all measured positions that:
    #   - Match the epoch's high-probability spatial mask (binned_x, binned_y in the mask)
    #   - Are categorized as 'past', 'present', or 'future' relative to the epoch time window
    #   Columns include: 'binned_x', 'binned_y', 't', 'x', 'y', 'is_future_present_past'
    matching_pos_dfs_list: List[pd.DataFrame] = serialized_field(default=Factory(list), metadata={'date_added': '2025.12.22_0'})
    # matching_pos_epochs_dfs_list: List of DataFrames, one per epoch. Each DataFrame contains detected continuous epochs
    #   (time periods) where the animal was in positions matching the epoch's high-probability mask.
    #   Epochs are detected by merging adjacent matching positions with gaps <= merging_adjacent_max_separation_sec
    #   and filtering by minimum_epoch_duration. Each row represents a continuous time period.
    #   Columns include: 'start', 'stop', 'duration', 'is_future_present_past' (categorized as 'past', 'present', or 'future')
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


    @classmethod
    def detect_matching_past_future_positions(cls, epoch_high_prob_mask: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], Any], measured_positions_df: pd.DataFrame, curr_epoch_start_t: float, curr_epoch_stop_t: float, merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050,
                                                    epoch_t_bins_high_prob_pos_mask: Optional[NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], Any]]=None, decoded_epoch_result=None, 
                                                    should_defer_extended_computations: bool = True, **kwargs, ## passthrough-only properties
                                               ) -> MatchingPastFuturePositionsResult:
        """
        Detect matching past/future positions for a given epoch high probability mask.
        
        Args:
            epoch_high_prob_mask: 2D boolean mask (N_XBINS, N_Y_BINS) indicating high probability positions during the epoch
            measured_positions_df: DataFrame with position data including 'binned_x', 'binned_y', 't' columns
            curr_epoch_start_t: Start time of the current epoch
            curr_epoch_stop_t: Stop time of the current epoch
            merging_adjacent_max_separation_sec: Maximum separation in seconds for merging adjacent epochs
            minimum_epoch_duration: Minimum duration for detected epochs
            
        Returns:
            MatchingPastFuturePositionsResult containing all computed results
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PosteriorMaskPostProcessing
        
        relevant_positions_df: pd.DataFrame = measured_positions_df.copy()
        

        # find relevant positions: Search through all measured positions (for all time) and find bins that match with a position decoded in this mask) ________________________________________________________________________________________________________________________________________________________________ #
        row_col_indices = np.argwhere(epoch_high_prob_mask)
        row_col_row_ids = row_col_indices + 1 # 0-index to 1-index
        an_epoch_mask_included_binned_x_y_columns_idx_df = pd.DataFrame(row_col_row_ids, columns=["binned_x", "binned_y"])
        ## allowed positions are much less than the found ones:
        relevant_positions_df = relevant_positions_df.merge(an_epoch_mask_included_binned_x_y_columns_idx_df, on=["binned_x", "binned_y"], how="inner")
        relevant_positions_df_after_merge = relevant_positions_df.copy()  # Save state after merge for visualization

        ## only after initial filter do we filter by this version:
        pos_matches_epoch_mask = np.where([epoch_high_prob_mask[(a_pos.binned_x-1), (a_pos.binned_y-1)] for a_pos in relevant_positions_df.itertuples()])[0]
        relevant_positions_df: pd.DataFrame = relevant_positions_df.iloc[pos_matches_epoch_mask].copy()
        
        # if _debug_plot:
        #     # DEBUG: Visualize filtering stages (uncomment to enable)
        #     from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import EventEpochsDebugger
        #     fig = EventEpochsDebugger.visualize_filtering_stages(
        #         measured_positions_df=measured_positions_df,
        #         relevant_positions_df_after_merge=relevant_positions_df_after_merge,
        #         relevant_positions_df_final=relevant_positions_df,
        #         epoch_high_prob_mask=epoch_high_prob_mask,
        #         curr_epoch_start_t=curr_epoch_start_t,
        #         curr_epoch_stop_t=curr_epoch_stop_t,
        #         max_points_per_plot=5000,
        #         merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec
        #     )
        #     import matplotlib.pyplot as plt
        #     plt.show()

        # Now divide the found positions into past/future categories _________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

        is_relevant_past_times = (relevant_positions_df['t'] < curr_epoch_start_t)
        is_relevant_future_times = (relevant_positions_df['t'] > curr_epoch_stop_t)

        ## 2026-01-14 08:05: - [X] find timestamps instead of saving indicies
        ## get the concrete timestamps of the relevant past/future times for use later:
        relevant_past_times: NDArray = relevant_positions_df[is_relevant_past_times]['t'].to_numpy()
        relevant_future_times: NDArray = relevant_positions_df[is_relevant_future_times]['t'].to_numpy()        
        ## can use these later via `epoch_time_to_index_map = relevant_positions_df.epochs.find_epoch_times_to_data_indicies_map(epoch_times=[epoch_start_time, ])` or something similar
        
        relevant_positions_df['is_future_present_past'] = 'present'
        relevant_positions_df.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
        relevant_positions_df.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'

        ## how many timestamps still remain in the past and the future:
        n_total_possible_past_times = np.sum(measured_positions_df['t'] < curr_epoch_start_t)
        n_total_possible_future_times = np.sum(measured_positions_df['t'] > curr_epoch_stop_t)
        
        n_relevant_past_times = np.sum(is_relevant_past_times)
        n_relevant_future_times = np.sum(is_relevant_future_times)

        ## find adjacent epochs from the position time bins (periods where the animal is in the positions)
        ## use relevant_positions_df directly since it's already filtered to epoch mask positions
        # measured_positions_df_copy = relevant_positions_df.copy()
        # Create boolean mask directly (single vectorized operation) for past/future positions
        is_included_mask = (measured_positions_df['t'] < curr_epoch_start_t) | (measured_positions_df['t'] > curr_epoch_stop_t)
        # Filter once before passing to function (only copy the filtered subset, not the entire dataframe)
        filtered_positions_df = measured_positions_df[is_included_mask].copy()

        a_matching_pos_epochs_df: pd.DataFrame = MatchingPastFuturePositionsResult.compute_matching_pos_epochs_df(measured_positions_df=filtered_positions_df, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, disable_segmentation=should_defer_extended_computations)
        
        ## found all matching events, now see whether these events are in the path or the future:
        is_pos_epochs_relevant_past_times = (a_matching_pos_epochs_df['start'] < curr_epoch_start_t)
        is_pos_epochs_relevant_future_times = (a_matching_pos_epochs_df['stop'] > curr_epoch_stop_t)
        a_matching_pos_epochs_df['is_future_present_past'] = 'present'
        a_matching_pos_epochs_df.loc[is_pos_epochs_relevant_past_times, 'is_future_present_past'] = 'past'
        a_matching_pos_epochs_df.loc[is_pos_epochs_relevant_future_times, 'is_future_present_past'] = 'future'

        ## INPUTS: a_matching_pos_epochs_df, relevant_positions_df
        ## add the final detected a_matching_pos_epochs_df indicies to the decoded positions as the column ['matching_found_relevant_pos_epoch']:
        epoch_id_key_name: str = 'matching_found_relevant_pos_epoch'
        relevant_positions_df = MatchingPastFuturePositionsResult._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=a_matching_pos_epochs_df, relevant_positions_df=relevant_positions_df,
                                                                                                                          drop_non_epoch_events=False, epoch_id_key_name=epoch_id_key_name) ## don't drop yet so we have all the events for the object creation

        # ## to drop post-hoc
        # relevant_positions_df = relevant_positions_df.dropna(how='any', subset=[epoch_id_key_name], inplace=False)


        _out_obj: MatchingPastFuturePositionsResult = MatchingPastFuturePositionsResult(decoded_epoch_result=decoded_epoch_result,
                                                 epoch_high_prob_mask=epoch_high_prob_mask, epoch_t_bins_high_prob_pos_mask=epoch_t_bins_high_prob_pos_mask,
                                                 relevant_past_times=relevant_past_times, relevant_future_times=relevant_future_times,
                                                 pos_matches_epoch_mask=pos_matches_epoch_mask, relevant_positions_df=relevant_positions_df, is_relevant_past_times=is_relevant_past_times, is_relevant_future_times=is_relevant_future_times,
                    n_total_possible_past_times=n_total_possible_past_times, n_total_possible_future_times=n_total_possible_future_times, n_relevant_past_times=n_relevant_past_times, n_relevant_future_times=n_relevant_future_times,
                    matching_pos_epochs_df=a_matching_pos_epochs_df, should_defer_extended_computations=should_defer_extended_computations)
    

        # Post-process by calling .recompute_all() which actually makes the pos_epochs right: ________________________________________________________________________________________________________________________________________________________________________________________________ #

        print(f'performing .recompute_all() for epoch....')        
        # num_pre_found_epochs: int = len(_out_obj.matching_pos_epochs_df)
        # if (num_pre_found_epochs < 3): # def recompute, it's always 2 epochs when it's wrong
        _out_obj.recompute_all() ## almost 30.0 seconds just for one epoch
        # num_post_found_epochs: int = len(_out_obj.matching_pos_epochs_df)
        # print(f"num_post_found_epochs: {num_post_found_epochs}")        
        


        #TODO 2026-01-14 17:53: - [ ] `PosteriorMaskPostProcessing` post processing positions to see which are aligned with the posterior:
        # if _out_obj.a_centroids_search_segments_df is None:
        #     _out_obj._recompute_high_prob_mask_centroids()
        # if _out_obj.a_centroids_search_segments_df is not None:
        #     epoch_id_key_name: str = 'matching_found_relevant_pos_epoch'
        #     epoch_only_relevant_positions_df = MatchingPastFuturePositionsResult._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=_out_obj.matching_pos_epochs_df, relevant_positions_df=_out_obj.relevant_positions_df,
        #                                                                                                                 drop_non_epoch_events=True, epoch_id_key_name=epoch_id_key_name) ## drop those that aren't in the epochs
            

        #     epoch_only_relevant_positions_df, pos_segment_to_centroid_seq_segment_idx_map = PosteriorMaskPostProcessing._compare_centroid_and_pos_traj_angle(a_pos_df=epoch_only_relevant_positions_df, 
        #                                                                                                                                                      a_centroids_search_segments_df=_out_obj.a_centroids_search_segments_df)
        

        return _out_obj
    

    @function_attributes(short_name=None, tags=['SINGLE_MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['compute_specific_future_and_past_analysis'], creation_date='2026-01-14 19:49', related_items=[])
    @classmethod
    def _process_single_epoch_future_past_analysis(cls, i: int, curr_epoch_p_x_given_n: NDArray, curr_epoch_time_bin_centers: NDArray, curr_epoch_tbin_indicies: NDArray, gaussian_volume: Optional[NDArray], measured_positions_df: pd.DataFrame, top_v_percent: float,
                epoch_t_bin_high_prob_masks_dict: Optional[Dict], epoch_high_prob_masks_dict: Optional[Dict],
                a_slice_multiplier: float, n_epoch_time_bins: int, merging_adjacent_max_separation_sec: float, minimum_epoch_duration: float, progress_print: bool, n_total_epochs: int, decoded_epoch_result=None, **kwargs) -> Tuple[int, Any, Any, Any, Any, Any, Any]:
        """Process a single epoch for future/past analysis. Returns results in a tuple for parallel processing."""
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PosteriorMaskPostProcessing
        
        if progress_print:
            print(f'\trow[{i}/{n_total_epochs}]')
        
        curr_epoch_start_t: float = curr_epoch_time_bin_centers[0]
        curr_epoch_stop_t: float = curr_epoch_time_bin_centers[-1]
        
        # a_gaussian_volume = None
        # if gaussian_volume is not None:
        #     a_gaussian_volume = gaussian_volume[..., curr_epoch_tbin_indicies]
        
        # ==================================================================================================================================================================================================================================================================================== #
        # Special posterior measurement properties (diffusivity, promenence, etc) computed independently with newly decoded fine time bin grainularity posteriors                                                                                                                              #
        # ==================================================================================================================================================================================================================================================================================== #
        
        is_high_prob_mask: Optional[NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TBINS"], Any]] = None
        merged_epoch_mask: Optional[NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], Any]] = None
        processed_masks: Optional[Any] = None
        
        if (epoch_t_bin_high_prob_masks_dict is not None):
            an_epoch_t_bins_custom_high_prob_mask: NDArray[ND.Shape["N_X_BINS, N_Y_BINS, N_TBINS"], Any] = epoch_t_bin_high_prob_masks_dict[a_slice_multiplier][i]
            Assert.same_shape(an_epoch_t_bins_custom_high_prob_mask, curr_epoch_p_x_given_n)
            is_high_prob_mask = an_epoch_t_bins_custom_high_prob_mask
            
            labeled, n_objects, masks = PosteriorMaskPostProcessing._process_epoch_time_bins_masks(a_mask_t=an_epoch_t_bins_custom_high_prob_mask, max_gap=8, n_interp=1)
            processed_masks = masks
            merged_epoch_mask = np.any(masks, axis=-1)
        
        elif (epoch_high_prob_masks_dict is not None):
            ## else only have the single mask per the whole epoch
            an_epoch_custom_high_prob_mask: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], Any] = epoch_high_prob_masks_dict[a_slice_multiplier][i]
            Assert.same_shape(an_epoch_custom_high_prob_mask, curr_epoch_p_x_given_n[:, :, 0])
            is_high_prob_mask = np.tile(an_epoch_custom_high_prob_mask, (1, 1, n_epoch_time_bins))
        
        else:
            ## for each time bin compute the top 10% of the time bins and use those instead of a fixed "high_val_epsilon" threshold:
            flat = curr_epoch_p_x_given_n.reshape(-1, curr_epoch_p_x_given_n.shape[-1])  # (n_xy, n_time)
            sorted_flat = np.sort(flat, axis=0)[::-1]
            cdf = np.cumsum(sorted_flat, axis=0)
            thresholds = sorted_flat[np.argmax(cdf >= top_v_percent * flat.sum(axis=0), axis=0), np.arange(flat.shape[1])]
            is_high_prob_mask = (curr_epoch_p_x_given_n >= thresholds)
        
        ## allow future positions to match any position in the epoch to count:
        if is_high_prob_mask is not None:
            any_t_Bin_high_prob_pos_mask: NDArray[ND.Shape["N_X_BINS, N_Y_BINS"], Any] = np.any(is_high_prob_mask, axis=-1) ## mask for high prob positions during the epoch
        else:
            raise ValueError(f"is_high_prob_mask is None for epoch {i}")
        
        # Call static method from the same class (PredictiveDecoding)
        any_t_bin_result: MatchingPastFuturePositionsResult = PredictiveDecoding.detect_matching_past_future_positions(epoch_high_prob_mask=any_t_Bin_high_prob_pos_mask, epoch_t_bins_high_prob_pos_mask=is_high_prob_mask, measured_positions_df=measured_positions_df, curr_epoch_start_t=curr_epoch_start_t, curr_epoch_stop_t=curr_epoch_stop_t, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
                                                                                     decoded_epoch_result=decoded_epoch_result, **kwargs)
        
        ## compute for `merged_epoch_mask` if it exists
        merged_epoch_mask_result = None
        if merged_epoch_mask is not None:
            merged_epoch_mask_result: MatchingPastFuturePositionsResult = PredictiveDecoding.detect_matching_past_future_positions(epoch_high_prob_mask=merged_epoch_mask, epoch_t_bins_high_prob_pos_mask=is_high_prob_mask, measured_positions_df=measured_positions_df, curr_epoch_start_t=curr_epoch_start_t, curr_epoch_stop_t=curr_epoch_stop_t, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
                                                                                                 decoded_epoch_result=decoded_epoch_result, **kwargs)
                                                                
        
        return (i, is_high_prob_mask, any_t_Bin_high_prob_pos_mask, any_t_bin_result, merged_epoch_mask, processed_masks, merged_epoch_mask_result)
    

    @function_attributes(short_name=None, tags=['MAIN', 'past-future', 'position'], input_requires=[], output_provides=[], uses=['_process_single_epoch_future_past_analysis'], used_by=[], creation_date='2026-01-14 17:08', related_items=[])
    @classmethod
    def compute_specific_future_and_past_analysis(cls, decoded_local_epochs_result: DecodedFilterEpochsResult, measured_positions_df: pd.DataFrame, gaussian_volume: Optional[NDArray]=None,
                                        active_epochs_df: Optional[pd.DataFrame]=None,
                                        an_epoch_name:str = 'roam',
                                        top_v_percent: float = 0.1, 
                                        epoch_t_bin_high_prob_masks_dict: Optional[Dict] = None,
                                        epoch_high_prob_masks_dict: Optional[Dict] = None,
                                        a_slice_multiplier: float = 0.5,
                                        merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, ## for merging detected future/past position dataframes
                                        progress_print: bool = True,
                                        use_parallel: bool = True, max_workers: Optional[int] = None,
                                        should_defer_extended_computations: bool = True, disable_segmentation: bool = True, **kwargs, 
        ):
        """
        Compute future and past position analysis for decoded epochs.
        
        For each epoch, this function:
        1. Identifies high-probability spatial positions from the decoded posterior
        2. Finds measured positions that match these high-probability locations
        3. Categorizes matching positions as 'past', 'present', or 'future' relative to the epoch time window
        4. Detects continuous time periods (epochs) where the animal was in matching positions
        
        Args:
            decoded_local_epochs_result: Result containing decoded posteriors and epoch information
            measured_positions_df: DataFrame with measured position data (must include 'binned_x', 'binned_y', 't', 'x', 'y' columns)
            gaussian_volume: Optional 3D array of gaussian volumes for visualization
            active_epochs_df: Optional DataFrame to which computed metrics will be added as columns
            an_epoch_name: Prefix for column names added to active_epochs_df (default: 'roam')
            top_v_percent: Top percentage threshold for identifying high-probability positions (default: 0.1 = top 10%)
            epoch_t_bin_high_prob_masks_dict: Optional dict of pre-computed per-time-bin high-probability masks
            epoch_high_prob_masks_dict: Optional dict of pre-computed epoch-level high-probability masks
            a_slice_multiplier: Multiplier for mask selection from dicts (default: 0.5)
            merging_adjacent_max_separation_sec: Max gap in seconds for merging adjacent matching position epochs (default: 0.5)
            minimum_epoch_duration: Minimum duration in seconds for detected position epochs (default: 0.050)
            progress_print: Whether to print progress messages (default: True)
            use_parallel: Whether to process epochs in parallel (default: True)
            max_workers: Maximum number of parallel workers (None = auto)
            
        Returns:
            Tuple of:
            - epoch_matching_past_future_positions: List[Tuple[NDArray, NDArray, int, int, int, int]]
                One tuple per epoch containing:
                [0]: Past position indices matching the epoch mask
                [1]: Future position indices matching the epoch mask
                [2]: Total possible past time points
                [3]: Total possible future time points
                [4]: Count of relevant past times
                [5]: Count of relevant future times
            - Tuple containing:
                - epoch_high_prob_pos_masks: List of 2D boolean masks (one per epoch) indicating high-probability positions
                - epoch_t_bins_high_prob_pos_masks: List of 3D boolean masks (one per epoch) with per-time-bin high-probability positions
                - epoch_matching_positions: List of position index arrays matching each epoch
                - past_future_info_dict: Dict with computed ratios and counts (ratio_past, ratio_future, etc.)
                - matching_pos_dfs_list: List[pd.DataFrame] - One DataFrame per epoch with all matching positions categorized as past/present/future
                - matching_pos_epochs_dfs_list: List[pd.DataFrame] - One DataFrame per epoch with detected continuous time periods where animal was in matching positions
                - _out_processed_items_list_dict: Dict with additional processed mask outputs
            - active_epochs_df: DataFrame with computed metrics added as columns (if provided)
        
        Example usage:
            non_local_PBE_non_moving_epochs_df: pd.DataFrame = decoding_locality.get_non_moving_PBE_non_local_epochs(curr_active_pipeline.sess, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec)
        # non_local_PBE_non_moving_epochs_df: pd.DataFrame = container.decoding_locality.non_local_PBE_non_moving_epochs_df

        measured_positions_df: pd.DataFrame = decoding_locality.pos_df
        # measured_positions_df = measured_positions_df.drop(columns=['binned_x', 'binned_y'], inplace=False)
        measured_positions_df = measured_positions_df.dropna(how='any', subset=['t', 'x', 'y'])
        measured_positions_df = measured_positions_df.position.adding_binned_position_columns(xbin_edges=decoding_locality.xbin, ybin_edges=decoding_locality.ybin)
        measured_positions_df = measured_positions_df[(measured_positions_df['binned_x'].notna()) & (measured_positions_df['binned_y'].notna())] # Filter rows based on columns: 'binned_x', 'binned_y'
        # decoding_locality.pos_df = measured_positions_df
        # measured_positions_df

        gaussian_volume = self.predictive_decoding.gaussian_volume ## the volume for all time bins

        

        epoch_matching_past_future_positions, _an_out_tuple, non_local_PBE_non_moving_epochs_df = PredictiveDecoding.compute_specific_future_and_past_analysis(decoded_local_epochs_result=decoded_local_epochs_result, measured_positions_df=measured_positions_df, gaussian_volume=gaussian_volume,
            non_local_PBE_non_moving_epochs_df=non_local_PBE_non_moving_epochs_df,
            an_epoch_name=an_epoch_name, top_v_percent=top_v_percent, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
        )
        epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list = _an_out_tuple
        

        """

        ## HARDCODED an_epoch_name
        # computed_df_col_name_prefix: str = ''
        computed_df_col_name_prefix: str = f'{an_epoch_name}_'

        # ==================================================================================================================================================================================================================================================================================== #
        # MAIN COMPUTATION/METRIC PART OF THIS FUNCTION                                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        epoch_high_prob_pos_masks = []
        epoch_t_bins_high_prob_pos_masks = []
        
        epoch_matching_positions = []
        # epoch_matching_past_future_positions: List of tuples, one per epoch. Each tuple contains 6 elements:
        #   [0]: NDArray - Indices of past positions that match the epoch's high-probability mask (from pos_matches_epoch_mask filtered by is_relevant_past_times)
        #   [1]: NDArray - Indices of future positions that match the epoch's high-probability mask (from pos_matches_epoch_mask filtered by is_relevant_future_times)
        #   [2]: int - Total count of possible past time points (all positions before epoch start)
        #   [3]: int - Total count of possible future time points (all positions after epoch stop)
        #   [4]: int - Count of relevant past times (past positions that match the epoch mask)
        #   [5]: int - Count of relevant future times (future positions that match the epoch mask)
        # Used to compute ratios like ratio_past = len([0]) / n_epoch_time_bins and ratio_future = len([1]) / n_epoch_time_bins
        epoch_matching_past_future_positions: List[Tuple[NDArray, NDArray, int, int, int, int]] = []
        # matching_pos_dfs_list: List of DataFrames, one per epoch. Each DataFrame contains all measured positions that:
        #   - Match the epoch's high-probability spatial mask (binned_x, binned_y in the mask)
        #   - Are categorized as 'past', 'present', or 'future' relative to the epoch time window
        #   Columns include: 'binned_x', 'binned_y', 't', 'x', 'y', 'is_future_present_past'
        #   This provides individual position-level matching information for each epoch
        matching_pos_dfs_list: List[pd.DataFrame] = []
        # matching_pos_epochs_dfs_list: List of DataFrames, one per epoch. Each DataFrame contains detected continuous epochs
        #   (time periods) where the animal was in positions matching the epoch's high-probability mask.
        #   Epochs are detected by merging adjacent matching positions with gaps <= merging_adjacent_max_separation_sec
        #   and filtering by minimum_epoch_duration. Each row represents a continuous time period.
        #   Columns include: 'start', 'stop', 'duration', 'is_future_present_past' (categorized as 'past', 'present', or 'future')
        #   This provides epoch-level (continuous period) matching information, as opposed to individual position matches
        matching_pos_epochs_dfs_list: List[pd.DataFrame] = []

        # [array([0, 1, 2, 3, 4]), array([0, 1]), array([0, 1, 2, 3, 4, 5, 6, 7]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
        n_flattened_tbins: int = np.sum(decoded_local_epochs_result.nbins)
        flattened_time_bin_indicies = np.arange(n_flattened_tbins)
        reverse_flattened_time_bin_indicies_list: List[NDArray] = split_array(flattened_time_bin_indicies, sub_element_lengths=decoded_local_epochs_result.nbins)
        # assert len(split_by_epoch_reverse_flattened_time_bin_indicies) == n_epochs

        n_total_epochs: int = len(decoded_local_epochs_result.filter_epochs)
        if progress_print:
            print(f'about to iterate n_total_epochs: {n_total_epochs} epochs.')


        ## custom outputs:
        _out_processed_items_list_dict = {'_out_epoch_flat_mask': [], 
            '_out_processed_masks': [],
            '_out_epoch_flat_mask_future_past_result': [],
        }
        
        
        # Prepare epoch data for processing
        epoch_data_list = []
        for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False)):
            
            a_decoded_epoch_result: SingleEpochDecodedResult = decoded_local_epochs_result.get_result_for_epoch(active_epoch_idx=i)
            
            curr_epoch_p_x_given_n = decoded_local_epochs_result.p_x_given_n_list[i]
            curr_epoch_time_bin_centers = decoded_local_epochs_result.time_bin_containers[i].centers
            curr_epoch_tbin_indicies = reverse_flattened_time_bin_indicies_list[i]
            n_epoch_time_bins = curr_epoch_p_x_given_n.shape[-1]  # Number of time bins for this epoch
            epoch_data_list.append((i, curr_epoch_p_x_given_n, curr_epoch_time_bin_centers, curr_epoch_tbin_indicies, n_epoch_time_bins, a_decoded_epoch_result))
        
        # Process epochs in parallel or sequentially
        if use_parallel and n_total_epochs > 1:
            if progress_print:
                print(f'Processing {n_total_epochs} epochs in parallel (max_workers={max_workers})...')
            
            results_list = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, curr_epoch_p_x_given_n, curr_epoch_time_bin_centers, curr_epoch_tbin_indicies, n_epoch_time_bins, a_decoded_epoch_result in epoch_data_list:
                    future = executor.submit(cls._process_single_epoch_future_past_analysis, i=i, curr_epoch_p_x_given_n=curr_epoch_p_x_given_n, curr_epoch_time_bin_centers=curr_epoch_time_bin_centers, curr_epoch_tbin_indicies=curr_epoch_tbin_indicies, gaussian_volume=gaussian_volume, measured_positions_df=measured_positions_df, top_v_percent=top_v_percent, epoch_t_bin_high_prob_masks_dict=epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict=epoch_high_prob_masks_dict, a_slice_multiplier=a_slice_multiplier, n_epoch_time_bins=n_epoch_time_bins, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, progress_print=progress_print, n_total_epochs=n_total_epochs, decoded_epoch_result=a_decoded_epoch_result, should_defer_extended_computations=should_defer_extended_computations, disable_segmentation=disable_segmentation)
                    futures.append(future)
                
                for future in as_completed(futures):
                    results_list.append(future.result())
            
            # Sort results by index to maintain order
            results_list.sort(key=lambda x: x[0])
        else:
            if progress_print and use_parallel:
                print(f'Sequential processing (use_parallel=False or n_total_epochs <= 1)...')
            
            results_list = []
            for i, curr_epoch_p_x_given_n, curr_epoch_time_bin_centers, curr_epoch_tbin_indicies, n_epoch_time_bins, a_decoded_epoch_result in epoch_data_list:
                result = cls._process_single_epoch_future_past_analysis(i=i, curr_epoch_p_x_given_n=curr_epoch_p_x_given_n, curr_epoch_time_bin_centers=curr_epoch_time_bin_centers, curr_epoch_tbin_indicies=curr_epoch_tbin_indicies, gaussian_volume=gaussian_volume, measured_positions_df=measured_positions_df, top_v_percent=top_v_percent, epoch_t_bin_high_prob_masks_dict=epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict=epoch_high_prob_masks_dict, a_slice_multiplier=a_slice_multiplier, n_epoch_time_bins=n_epoch_time_bins, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, progress_print=progress_print, n_total_epochs=n_total_epochs, decoded_epoch_result=a_decoded_epoch_result, should_defer_extended_computations=should_defer_extended_computations, disable_segmentation=disable_segmentation)
                results_list.append(result)
        
        # Unpack results and populate output lists
        for i, is_high_prob_mask, any_t_Bin_high_prob_pos_mask, any_t_bin_result, merged_epoch_mask, processed_masks, merged_epoch_mask_result in results_list:
            #TODO 2026-01-14 17:27: - [ ] These look reversed... are they?
            epoch_t_bins_high_prob_pos_masks.append(is_high_prob_mask)
            epoch_high_prob_pos_masks.append(any_t_Bin_high_prob_pos_mask)
            
            # Build epoch_matching_past_future_positions tuple: (past_indices, future_indices, n_total_past, n_total_future, n_relevant_past, n_relevant_future)
            epoch_matching_past_future_positions.append((any_t_bin_result.pos_matches_epoch_mask[any_t_bin_result.is_relevant_past_times], any_t_bin_result.pos_matches_epoch_mask[any_t_bin_result.is_relevant_future_times], any_t_bin_result.n_total_possible_past_times, any_t_bin_result.n_total_possible_future_times, any_t_bin_result.n_relevant_past_times, any_t_bin_result.n_relevant_future_times))
            
            epoch_matching_positions.append(any_t_bin_result.pos_matches_epoch_mask)
            # Append DataFrame with all matching positions (past/present/future) for this epoch
            matching_pos_dfs_list.append(any_t_bin_result.relevant_positions_df)
            # Append DataFrame with detected continuous epochs where animal was in matching positions
            matching_pos_epochs_dfs_list.append(any_t_bin_result.matching_pos_epochs_df)
            
            # Handle processed masks and merged epoch mask results
            if processed_masks is not None:
                _out_processed_items_list_dict['_out_processed_masks'].append(processed_masks)
            if merged_epoch_mask is not None:
                _out_processed_items_list_dict['_out_epoch_flat_mask'].append(merged_epoch_mask)
            if merged_epoch_mask_result is not None:
                _out_processed_items_list_dict['_out_epoch_flat_mask_future_past_result'].append(merged_epoch_mask_result)
        
        ## END epoch processing loop

        

        ## OUTPUTS: epoch_matching_positions, epoch_matching_past_future_positions
        # Compute ratios: number of matching past/future positions per epoch time bin
        # epoch_matching_past_future_positions[i][0] = past position indices, epoch_matching_past_future_positions[i][1] = future position indices
        ratio_past = np.array([len(epoch_matching_past_future_positions[i][0])/ len(decoded_local_epochs_result.time_bin_containers[i].centers) for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        ratio_future = np.array([len(epoch_matching_past_future_positions[i][1])/ len(decoded_local_epochs_result.time_bin_containers[i].centers) for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        n_total_possible_past = np.array([epoch_matching_past_future_positions[i][2] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        n_total_possible_future = np.array([epoch_matching_past_future_positions[i][3] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        n_total_relevant_past = np.array([epoch_matching_past_future_positions[i][4] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])
        n_total_relevant_future = np.array([epoch_matching_past_future_positions[i][5] for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=False))])

        past_future_info_dict = {'ratio_past': ratio_past, 'ratio_future': ratio_future, 'n_total_possible_past': n_total_possible_past, 'n_total_possible_future': n_total_possible_future, 'n_total_relevant_past': n_total_relevant_past, 'n_total_relevant_future': n_total_relevant_future, }
        # non_local_PBE_non_moving_epochs_df.update(past_future_info_dict)
                
        if active_epochs_df is not None:
            ## add the columns to the datframe
            for k, v in past_future_info_dict.items():
                active_epochs_df[f"{computed_df_col_name_prefix}{k}"] = v
                
            ## add more columns after the others are added:
            active_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_past'] = active_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_past'] / active_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_past']
            active_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_future'] = active_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_future'] / active_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_future']

        # epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list
        
        return epoch_matching_past_future_positions, (epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict), active_epochs_df



    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # PredictiveDecoding has no non-serialized fields, so no exclusions needed
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

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

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingComputationsContainer

        wcorr_shuffle_results: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('PredictiveDecoding', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'PredictiveDecoding is not computed.')
            
    """
    _VersionedResultMixin_version: str = "2026.01.08_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    predictive_decoding: Optional[PredictiveDecoding] = serialized_field(default=None, repr=False)
    
    pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.20_0", 'copied_from': 'DirectionalDecodersContinuouslyDecodedResult'})
    epochs_decoded_result_cache_dict: Dict[float, Dict[types.DecoderName, DecodedFilterEpochsResult]] = serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.20_0", 'copied_from': 'DirectionalDecodersContinuouslyDecodedResult'}) # key is the t_bin_size in seconds
    debug_computed_dict: Dict[types.DecoderName, Dict] = non_serialized_field(default=Factory(dict), metadata={'field_added': "2025.12.21_0"})

    scoring_results_df: pd.DataFrame = non_serialized_field(default=None, metadata={'field_added': "2026.01.08_0"})
    active_epochs_df: Optional[pd.DataFrame] = non_serialized_field(default=None, metadata={'field_added': "2026.01.13_0"})


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


    @function_attributes(short_name=None, tags=['UNFINISHED', 'PENDING', '2025-01-09'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-09 03:38', related_items=[])
    def build_masked_container(self, curr_active_pipeline, a_t_bin_size: float = 0.025,
                                should_filter_directional_decoders_decode_result: bool = True, should_compute_future_and_past_analysis: bool=True, should_compute_peak_prom_analysis: bool = False,
            window_size = 60, **kwargs,
        ) -> "PredictiveDecodingComputationsContainer":
        """ filters a copy of self
        
        container.predictive_decoding.matching_pos_epochs_dfs_list needs to be updated/filtered
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

        # from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PeakCounts, SlabResult, PeakPromenenceMetrics, PosteriorPeaksPeakProminence2dResult

        # an_epoch_name = 'roam'
        
        # a_result_decoded: DecodedFilterEpochsResult = container.epochs_decoded_result_cache_dict[a_t_bin_size][an_epoch_name]
        # a_result_decoded

        def _subfn_update_internal_results(masked_container, selected_tbin_size: Optional[float] = None):
            """ Filter the `masked_container.epochs_decoded_result_cache_dict` results (optionally only one tbin).
            captures nothing.
            Usage:
                masked_container = _subfn_update_internal_results(masked_container, selected_tbin=0.025)
            """
            if selected_tbin_size is not None:
                a_decoded_results_dict_dict = {selected_tbin_size: masked_container.epochs_decoded_result_cache_dict.get(selected_tbin_size, {})}
            else:
                a_decoded_results_dict_dict = masked_container.epochs_decoded_result_cache_dict

            scoring_results_df_list = []
            for a_decoding_time_bin_size, a_decoded_results_dict in (a_decoded_results_dict_dict or {}).items():
                for an_decoder_name, a_decoded_local_epochs_result in (a_decoded_results_dict or {}).items():
                    a_decoder = masked_container.pf1D_Decoder_dict.get(an_decoder_name, None)
                    if a_decoder is None:
                        a_decoder = list(masked_container.pf1D_Decoder_dict.values())[0]

                    filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_decoded_local_epochs_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3)
                    masked_container.epochs_decoded_result_cache_dict[a_decoding_time_bin_size][an_decoder_name] = filtered_decoded_local_epochs_result ## overwrite with the filtered one

                    if scoring_results is not None:
                        if isinstance(scoring_results, pd.DataFrame):
                            a_df = scoring_results.copy()
                        else:
                            a_df = pd.DataFrame([scoring_results])
                        a_df['decoder_name'] = an_decoder_name
                        a_df['decoding_time_bin_size'] = a_decoding_time_bin_size
                        scoring_results_df_list.append(a_df)
                ## END for an_decoder_name, a_decoded_local_epochs_resu...
            ## END for a_decoding_time_...

            if len(scoring_results_df_list) > 0:
                masked_container.scoring_results_df = pd.concat(scoring_results_df_list, ignore_index=True)
            return masked_container


        def _subfn_filter_masked_container_epochs(masked_container, original_active_epochs_df: pd.DataFrame):
            """ filters the epochs in the masked_container 
                ## Filter active_epochs_df and matching_pos_epochs_dfs_list to match the filtered decoded results ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                # NOTE: We use actual time span from time_bin_containers (edges) rather than filter_epochs start/stop because when time bins are dropped during masking, the effective start/stop times change
                # Build epoch_idx -> (actual_start, actual_stop) mapping once, reuse for filtering and recomputation

            """
            filter_epochs = None
            
            epoch_idx_to_actual_times = {}
            for a_decoding_time_bin_size, a_decoded_results_dict in (masked_container.epochs_decoded_result_cache_dict or {}).items():
                for an_decoder_name, a_decoded_local_epochs_result in (a_decoded_results_dict or {}).items():
                    filter_epochs = a_decoded_local_epochs_result.filter_epochs
                    for epoch_idx, time_bin_container in enumerate(a_decoded_local_epochs_result.time_bin_containers):
                        epoch_idx_to_actual_times[epoch_idx] = (time_bin_container.edges[0], time_bin_container.edges[-1])
                    break  # Only need one decoder/time_bin_size
                    
            ## assign the dang epoch:
            has_valid_filter_epochs: bool = (filter_epochs is not None) and (len(filter_epochs) > 0) 
            assert has_valid_filter_epochs
            masked_container.active_epochs = filter_epochs

            ## Maybe need to refine the epoch (start, end) times:
            filtered_epochs_set = set(epoch_idx_to_actual_times.values())
            time_tolerance = 0.01
            
            # Filter active_epochs_df to only include epochs that overlap with the filtered set
            if masked_container.active_epochs_df is not None:
                if len(filtered_epochs_set) > 0:
                    original_len = len(masked_container.active_epochs_df)
                    active_epochs_df = ensure_dataframe(masked_container.active_epochs_df)
                    is_epoch_included = active_epochs_df.apply(lambda row: any(
                        (abs(row['start'] - fs) < time_tolerance and abs(row['stop'] - fe) < time_tolerance) or (row['start'] <= fe and row['stop'] >= fs)
                        for fs, fe in filtered_epochs_set), axis=1)
                    masked_container.active_epochs_df = active_epochs_df[is_epoch_included].reset_index(drop=True)
                    print(f'Filtered active_epochs_df: {original_len} -> {len(masked_container.active_epochs_df)} epochs (removed {original_len - len(masked_container.active_epochs_df)})')
                else:
                    print(f'WARN: No filtered epochs found, active_epochs_df may be empty or inconsistent')
            
            # Filter epoch-indexed lists in predictive_decoding to only include entries for epochs that remain after filtering
            pred_dec = masked_container.predictive_decoding
            filtered_to_original_idx = None  # Will be set if we can map filtered epochs to original indices
            if pred_dec is not None and masked_container.active_epochs_df is not None and len(masked_container.active_epochs_df) > 0:
                # original_active_epochs_df = ensure_dataframe(self.active_epochs_df) if (hasattr(self, 'active_epochs_df') and self.active_epochs_df is not None) else None
                if original_active_epochs_df is not None:
                    active_epochs_df = ensure_dataframe(masked_container.active_epochs_df)
                    # Build mapping from filtered epochs to original indices
                    filtered_to_original_idx = []
                    for _, row in active_epochs_df.iterrows():
                        matching_original_idx = next((orig_idx for orig_idx, orig_row in original_active_epochs_df.iterrows()
                            if (abs(row['start'] - orig_row['start']) < time_tolerance and abs(row['stop'] - orig_row['stop']) < time_tolerance) or
                            (row['start'] <= orig_row['stop'] and row['stop'] >= orig_row['start'])), None)
                        if matching_original_idx is not None:
                            filtered_to_original_idx.append(matching_original_idx)
                    
                    # Filter all epoch-indexed lists using the same mapping
                    if hasattr(pred_dec, 'matching_pos_epochs_dfs_list') and pred_dec.matching_pos_epochs_dfs_list and len(pred_dec.matching_pos_epochs_dfs_list) > 0:
                        original_len = len(pred_dec.matching_pos_epochs_dfs_list)
                        pred_dec.matching_pos_epochs_dfs_list = [pred_dec.matching_pos_epochs_dfs_list[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered matching_pos_epochs_dfs_list: {original_len} -> {len(pred_dec.matching_pos_epochs_dfs_list)} entries (removed {original_len - len(pred_dec.matching_pos_epochs_dfs_list)})')
                    
                    if hasattr(pred_dec, 'matching_pos_dfs_list') and pred_dec.matching_pos_dfs_list and len(pred_dec.matching_pos_dfs_list) > 0:
                        original_len = len(pred_dec.matching_pos_dfs_list)
                        pred_dec.matching_pos_dfs_list = [pred_dec.matching_pos_dfs_list[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered matching_pos_dfs_list: {original_len} -> {len(pred_dec.matching_pos_dfs_list)} entries (removed {original_len - len(pred_dec.matching_pos_dfs_list)})')
                    
                    if hasattr(pred_dec, 'epoch_matching_past_future_positions') and pred_dec.epoch_matching_past_future_positions and len(pred_dec.epoch_matching_past_future_positions) > 0:
                        original_len = len(pred_dec.epoch_matching_past_future_positions)
                        pred_dec.epoch_matching_past_future_positions = [pred_dec.epoch_matching_past_future_positions[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered epoch_matching_past_future_positions: {original_len} -> {len(pred_dec.epoch_matching_past_future_positions)} entries (removed {original_len - len(pred_dec.epoch_matching_past_future_positions)})')
                    
                    if hasattr(pred_dec, 'epoch_high_prob_pos_masks') and pred_dec.epoch_high_prob_pos_masks and len(pred_dec.epoch_high_prob_pos_masks) > 0:
                        original_len = len(pred_dec.epoch_high_prob_pos_masks)
                        pred_dec.epoch_high_prob_pos_masks = [pred_dec.epoch_high_prob_pos_masks[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered epoch_high_prob_pos_masks: {original_len} -> {len(pred_dec.epoch_high_prob_pos_masks)} entries (removed {original_len - len(pred_dec.epoch_high_prob_pos_masks)})')
                    
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks and len(pred_dec.epoch_t_bins_high_prob_pos_masks) > 0:
                        original_len = len(pred_dec.epoch_t_bins_high_prob_pos_masks)
                        pred_dec.epoch_t_bins_high_prob_pos_masks = [pred_dec.epoch_t_bins_high_prob_pos_masks[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered epoch_t_bins_high_prob_pos_masks: {original_len} -> {len(pred_dec.epoch_t_bins_high_prob_pos_masks)} entries (removed {original_len - len(pred_dec.epoch_t_bins_high_prob_pos_masks)})')
                else:
                    # If we don't have original epochs, truncate all lists to the length of active_epochs_df
                    filtered_len = len(masked_container.active_epochs_df)
                    if hasattr(pred_dec, 'matching_pos_epochs_dfs_list') and pred_dec.matching_pos_epochs_dfs_list:
                        original_len = len(pred_dec.matching_pos_epochs_dfs_list)
                        pred_dec.matching_pos_epochs_dfs_list = pred_dec.matching_pos_epochs_dfs_list[:filtered_len]
                        print(f'WARN: Truncated matching_pos_epochs_dfs_list to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                    if hasattr(pred_dec, 'matching_pos_dfs_list') and pred_dec.matching_pos_dfs_list:
                        original_len = len(pred_dec.matching_pos_dfs_list)
                        pred_dec.matching_pos_dfs_list = pred_dec.matching_pos_dfs_list[:filtered_len]
                        print(f'WARN: Truncated matching_pos_dfs_list to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                    if hasattr(pred_dec, 'epoch_matching_past_future_positions') and pred_dec.epoch_matching_past_future_positions:
                        original_len = len(pred_dec.epoch_matching_past_future_positions)
                        pred_dec.epoch_matching_past_future_positions = pred_dec.epoch_matching_past_future_positions[:filtered_len]
                        print(f'WARN: Truncated epoch_matching_past_future_positions to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                    if hasattr(pred_dec, 'epoch_high_prob_pos_masks') and pred_dec.epoch_high_prob_pos_masks:
                        original_len = len(pred_dec.epoch_high_prob_pos_masks)
                        pred_dec.epoch_high_prob_pos_masks = pred_dec.epoch_high_prob_pos_masks[:filtered_len]
                        print(f'WARN: Truncated epoch_high_prob_pos_masks to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks:
                        original_len = len(pred_dec.epoch_t_bins_high_prob_pos_masks)
                        pred_dec.epoch_t_bins_high_prob_pos_masks = pred_dec.epoch_t_bins_high_prob_pos_masks[:filtered_len]
                        print(f'WARN: Truncated epoch_t_bins_high_prob_pos_masks to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                        
            elif pred_dec is not None:
                # If active_epochs_df is empty or None, clear all epoch-indexed lists
                if hasattr(pred_dec, 'matching_pos_epochs_dfs_list') and pred_dec.matching_pos_epochs_dfs_list:
                    original_len = len(pred_dec.matching_pos_epochs_dfs_list)
                    pred_dec.matching_pos_epochs_dfs_list = []
                    print(f'WARN: Cleared matching_pos_epochs_dfs_list ({original_len} entries) because active_epochs_df is empty or None')
                if hasattr(pred_dec, 'matching_pos_dfs_list') and pred_dec.matching_pos_dfs_list:
                    original_len = len(pred_dec.matching_pos_dfs_list)
                    pred_dec.matching_pos_dfs_list = []
                    print(f'WARN: Cleared matching_pos_dfs_list ({original_len} entries) because active_epochs_df is empty or None')
                if hasattr(pred_dec, 'epoch_matching_past_future_positions') and pred_dec.epoch_matching_past_future_positions:
                    original_len = len(pred_dec.epoch_matching_past_future_positions)
                    pred_dec.epoch_matching_past_future_positions = []
                    print(f'WARN: Cleared epoch_matching_past_future_positions ({original_len} entries) because active_epochs_df is empty or None')
                if hasattr(pred_dec, 'epoch_high_prob_pos_masks') and pred_dec.epoch_high_prob_pos_masks:
                    original_len = len(pred_dec.epoch_high_prob_pos_masks)
                    pred_dec.epoch_high_prob_pos_masks = []
                    print(f'WARN: Cleared epoch_high_prob_pos_masks ({original_len} entries) because active_epochs_df is empty or None')
                if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks:
                    original_len = len(pred_dec.epoch_t_bins_high_prob_pos_masks)
                    pred_dec.epoch_t_bins_high_prob_pos_masks = []
                    print(f'WARN: Cleared epoch_t_bins_high_prob_pos_masks ({original_len} entries) because active_epochs_df is empty or None')
            
            
            # Recompute 'is_future_present_past' column for filtered matching_pos_epochs_dfs_list and matching_pos_dfs_list using actual epoch times _____________________________________________________________________________________________________________________________________________ #
            # assert len(masked_container.active_epochs) == len(epoch_idx_to_actual_times)
            # epoch_start, epoch_stop = epoch_idx_to_actual_times
            # masked_container.active_epochs = masked_container.active_epochs
            # is_past = (masked_container.active_epochs['t'] < epoch_start)
            # is_future = (masked_container.active_epochs['t'] > epoch_stop)
            # masked_container.active_epochs['is_future_present_past'] = 'present'
            # masked_container.active_epochs.loc[is_past, 'is_future_present_past'] = 'past'
            # masked_container.active_epochs.loc[is_future, 'is_future_present_past'] = 'future'

            # NOTE: We need to map filtered indices (0, 1, 2...) back to original indices to look up correct epoch times
            if pred_dec is not None and len(epoch_idx_to_actual_times) > 0:
                
                if pred_dec.matching_pos_epochs_dfs_list and len(pred_dec.matching_pos_epochs_dfs_list) > 0:
                    for filtered_idx, df in enumerate(pred_dec.matching_pos_epochs_dfs_list):
                        # Map filtered index to original index to look up epoch times
                        original_idx = filtered_to_original_idx[filtered_idx] if (filtered_to_original_idx is not None and filtered_idx < len(filtered_to_original_idx)) else filtered_idx
                        if original_idx in epoch_idx_to_actual_times:
                            epoch_start, epoch_stop = epoch_idx_to_actual_times[original_idx]
                            is_past = (df['stop'] < epoch_start)
                            is_future = (df['start'] > epoch_stop)
                            df['is_future_present_past'] = 'present'
                            df.loc[is_past, 'is_future_present_past'] = 'past'
                            df.loc[is_future, 'is_future_present_past'] = 'future'
                            pred_dec.matching_pos_epochs_dfs_list[filtered_idx] = df
                    print(f'Recomputed is_future_present_past column for {len(pred_dec.matching_pos_epochs_dfs_list)} filtered epochs in matching_pos_epochs_dfs_list')
                
                if pred_dec.matching_pos_dfs_list and len(pred_dec.matching_pos_dfs_list) > 0:
                    for filtered_idx, df in enumerate(pred_dec.matching_pos_dfs_list):
                        # Map filtered index to original index to look up epoch times
                        original_idx = filtered_to_original_idx[filtered_idx] if (filtered_to_original_idx is not None and filtered_idx < len(filtered_to_original_idx)) else filtered_idx
                        if original_idx in epoch_idx_to_actual_times:
                            epoch_start, epoch_stop = epoch_idx_to_actual_times[original_idx]
                            is_past = (df['t'] < epoch_start)
                            is_future = (df['t'] > epoch_stop)
                            df['is_future_present_past'] = 'present'
                            df.loc[is_past, 'is_future_present_past'] = 'past'
                            df.loc[is_future, 'is_future_present_past'] = 'future'
                            pred_dec.matching_pos_dfs_list[filtered_idx] = df
                    print(f'Recomputed is_future_present_past column for {len(pred_dec.matching_pos_dfs_list)} entries in matching_pos_dfs_list')
                ## END for
            ## END for
            
            return masked_container, filter_epochs, epoch_idx_to_actual_times


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        masked_container: Optional[PredictiveDecodingComputationsContainer] = None
        
        if hasattr(self, 'locality_measures') and (self.locality_measures is not None):
            # epoch_names: List[str] = self.locality_measures.paradigm_epochs_df.label.to_list() # ['roam', 'sprinkle']
            epoch_names: List[str] = self.locality_measures.epoch_names # ['roam', 'sprinkle']
        else:
            epoch_names: List[str] = self.decoding_locality.epoch_names
        
        assert len(epoch_names) > 0
        
        # assert use_full_recompute_method, f'the non full recompute mode  did not seem to do a dmang thing, I hope it is never called!'
        should_filter_directional_decoders_decode_result = True ## UPDATES: directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict
        should_compute_future_and_past_analysis = True


        # ==================================================================================================================================================================================================================================================================================== #
        # Modifies `directional_decoders_decode_result` from the pipeline itself?                                                                                                                                                                                                              #
        # ==================================================================================================================================================================================================================================================================================== #
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded'])
        available_tbins: List[float] = list(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.keys())
        assert len(available_tbins) > 0
        most_recent_tbin: float = available_tbins[-1]
        selected_tbin: float = a_t_bin_size if a_t_bin_size in available_tbins else most_recent_tbin

        if should_filter_directional_decoders_decode_result:
            print(f'directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict')
            a_decoder = list(directional_decoders_decode_result.pf1D_Decoder_dict.values())[0]
            for extant_decoded_time_bin_size, a_result_decoded in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.items():
                a_result_decoded: SingleEpochDecodedResult = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]
                a_result_decoded: DecodedFilterEpochsResult = DecodedFilterEpochsResult.init_from_single_epoch_result(single_epoch_result=a_result_decoded, decoding_time_bin_size=extant_decoded_time_bin_size) ## convert to a `DecodedFilterEpochsResult` for masking
                filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_result_decoded, xbin=a_decoder.xbin, ybin=a_decoder.ybin,
                                                                                                                                            position_like_score_cutoff=0.42, num_min_position_like_t_bins=3, normalization_across_epochs_epoch_names=epoch_names)
                directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size] = filtered_decoded_local_epochs_result.get_result_for_epoch(0) ## get the single epoch, re-assign
            ## END for extant_decoded_time_bin_size, a_result_decoded in directional_decoder...

        masked_directional_decoders_decode_result = directional_decoders_decode_result
        ## OUTPUTS: masked_directional_decoders_decode_result
        
        pos_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        masked_locality_measures = DecodingLocalityMeasures.init_from_decode_result(curr_active_pipeline=curr_active_pipeline, directional_decoders_decode_result=masked_directional_decoders_decode_result, extant_decoded_time_bin_size=most_recent_tbin, sigma=None)
        masked_predictive_decoding: PredictiveDecoding = PredictiveDecoding.init_from_decode_result(pos_df=pos_df, locality_measures=masked_locality_measures, a_result_decoded=masked_directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[most_recent_tbin], window_size=window_size)

        if masked_locality_measures.sigma is None:
            x_step: float = np.nanmean(np.diff(masked_predictive_decoding.xbin))
            y_step: float = np.nanmean(np.diff(masked_predictive_decoding.ybin))
            sigma: float = np.nanmax([x_step, y_step]) * 5.0
            print(f'computed sigma from bin sizes: {sigma}')
        else:
            sigma = masked_locality_measures.sigma
            print(f'using sigma from masked_locality_measures: {sigma}')

        _moving_avg_dict, _moving_avg_meas_pos_overlap_dict, _gaussian_volume = masked_predictive_decoding.compute(sigma=sigma) ## updates masked_predictive_decoding.moving_avg_dict, masked_predictive_decoding.moving_avg_meas_pos_overlap_dict
        masked_container = PredictiveDecodingComputationsContainer(predictive_decoding=masked_predictive_decoding, is_global=True)

        masked_container.pf1D_Decoder_dict = deepcopy(self.pf1D_Decoder_dict)
        if selected_tbin in (self.epochs_decoded_result_cache_dict or {}):
            masked_container.epochs_decoded_result_cache_dict[selected_tbin] = deepcopy(self.epochs_decoded_result_cache_dict[selected_tbin])
        else:
            ## recompute:
            print(f'WARN: todo, recompute for missing selected_tbin: {selected_tbin}')
            # decoded_local_epochs_result, a_decoder = self.decode_epochs_for_posterior_analysis(curr_active_pipeline=curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=selected_tbin, active_epochs_df=active_epochs_df)
            # print(f'done with all decoding.')
    
        ## where the main results are filtered
        masked_container = _subfn_update_internal_results(masked_container=masked_container, selected_tbin_size=selected_tbin)
        

        # ## REQUIRED OUTPUTS: masked_container
        # assert masked_container is not None

        # # Get this specific result ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # # decoded_local_epochs_result = masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].get(an_epoch_name, None)
        # epoch_names: List[str] = list(masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].keys())
        # # epoch_names: List[str] = ['roam', 'sprinkle']

        # # an_epoch_name: str = epoch_names[0]
        # # a_decoded_local_epochs_result = masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].get(an_epoch_name, None)
        # # # a_decoder: BayesianPlacemapPositionDecoder = list(masked_container.pf1D_Decoder_dict.values())[0]
        # # a_decoder: BayesianPlacemapPositionDecoder = masked_container.pf1D_Decoder_dict.get(an_epoch_name, None)


        if should_compute_peak_prom_analysis:
            from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PosteriorPeaksPeakProminence2dResult

            print(f'computeing peak_prom_analysis because `should_compute_peak_prom_analysis == True`...')
            # raise NotImplementedError(f'Peak prominence analysis is intentionally disabled in build_masked_container (enable explicitly if needed).')

            print(f'\tfor epoch_names: {epoch_names}')
            for an_epoch_name in epoch_names:
                if an_epoch_name not in masked_container.debug_computed_dict:
                    masked_container.debug_computed_dict[an_epoch_name] = {}

                _comp_result_key: str = 'peak_prom_analysis'
                if _comp_result_key not in masked_container.debug_computed_dict[an_epoch_name]:
                    masked_container.debug_computed_dict[an_epoch_name][_comp_result_key] = {}

                a_decoded_local_epochs_result = masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].get(an_epoch_name, None)
                a_decoder: BayesianPlacemapPositionDecoder = masked_container.pf1D_Decoder_dict.get(an_epoch_name, None)
                # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_decoded_local_epochs_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3) ## this seems to be done previously in `_subfn_update_internal_results`, but that's okay

                step: float = PeakPromenence.compute_optimal_step_size(a_masked_result.p_x_given_n_list, resolution_factor=500.0)
                print(f'\tstep: {step}')
                masked_container.debug_computed_dict[an_epoch_name][_comp_result_key]['step'] = step
                
                decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=a_masked_result.p_x_given_n_list, 
                    xbin_centers=masked_container.predictive_decoding.xbin_centers, 
                    ybin_centers=masked_container.predictive_decoding.ybin_centers,
                    step=step, minimum_included_peak_height=None, # 1m 42s - 7m 1s
                    # step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
                    peak_height_multiplier_probe_levels=(0.25, 0.5, 0.9),
                    should_use_faster_compute_single_slab_implementation=False,
                    min_considered_promenence=1e-11,
                )
                masked_container.debug_computed_dict[an_epoch_name][_comp_result_key]['decoded_epoch_t_bins_promenence_result_obj'] = decoded_epoch_t_bins_promenence_result_obj
                print(f'\tcomputation done.')


        if should_compute_future_and_past_analysis:
            # if not use_full_recompute_method:
            #     raise ValueError(f'compute_future_and_past_analysis requires use_full_recompute_method=True to ensure predictive_decoding/locality_measures are consistent with the masked results.')

            for an_epoch_name in epoch_names:
                if an_epoch_name not in masked_container.debug_computed_dict:
                    masked_container.debug_computed_dict[an_epoch_name] = {}
                _out = masked_container.compute_future_and_past_analysis(curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=a_t_bin_size, 
                                                                            enable_updating_instance_states=True, **kwargs,
                                                                         )
                # epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list = _out
                epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _out
                # masked_container.debug_computed_dict[an_epoch_name] = {'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict}
                masked_container.debug_computed_dict[an_epoch_name].update({'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict})


            ## END for an_epoch_name in epoch_names...
            

        ## Filter active_epochs_df and matching_pos_epochs_dfs_list to match the filtered decoded results ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # NOTE: We use actual time span from time_bin_containers (edges) rather than filter_epochs start/stop because when time bins are dropped during masking, the effective start/stop times change
        # Build epoch_idx -> (actual_start, actual_stop) mapping once, reuse for filtering and recomputation


        original_active_epochs_df: pd.DataFrame = ensure_dataframe(self.active_epochs_df) if (hasattr(self, 'active_epochs_df') and self.active_epochs_df is not None) else None
        masked_container, filter_epochs, epoch_idx_to_actual_times = _subfn_filter_masked_container_epochs(masked_container=masked_container, original_active_epochs_df=original_active_epochs_df)


        return masked_container




    @function_attributes(short_name=None, tags=['temp', 'from-notebook', 'prominence2d', 'locality'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-13 10:17', related_items=[])
    def _filter_single_epoch_result(self, curr_active_pipeline, decoding_time_bin_size = 0.025, an_epoch_name = 'roam') -> DecodedFilterEpochsResult:
        """
            decoding_time_bin_size = 0.025
            an_epoch_name = 'roam'
            masked_container = container.build_masked_container(curr_active_pipeline=curr_active_pipeline, a_t_bin_size=decoding_time_bin_size,
                should_filter_directional_decoders_decode_result=True, should_compute_future_and_past_analysis=False, should_compute_peak_prom_analysis=False,
            ) ## 4m 18s now
            active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container._filter_single_epoch_result(curr_active_pipeline=curr_active_pipeline, decoding_time_bin_size=decoding_time_bin_size, an_epoch_name=an_epoch_name)

        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PosteriorPeaksPeakProminence2dResult

        if decoding_time_bin_size not in self.epochs_decoded_result_cache_dict:
            print(f'needs to compute: decoding_time_bin_size: {decoding_time_bin_size}')
            assert (self.active_epochs_df is not None)
            active_epochs_df = deepcopy(self.active_epochs_df)
            decoded_local_epochs_result, a_decoder = self.decode_epochs_for_posterior_analysis(curr_active_pipeline=curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=decoding_time_bin_size, active_epochs_df=active_epochs_df)
            print(f'done with all decoding.')
            
        # Get this specific result ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[decoding_time_bin_size].get(an_epoch_name, None)
        a_decoder: BayesianPlacemapPositionDecoder = list(self.pf1D_Decoder_dict.values())[0]

        # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=decoded_local_epochs_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3)

        ## INPUTS: active_epochs_result
        custom_results_list = []
        custom_results_df_list = []
        for i, a_row in enumerate(ensure_dataframe(a_masked_result.filter_epochs).itertuples(index=False)):
            # print(f'epoch[{i}/{len(active_epochs_result.filter_epochs)}]')
            ## Need correect portion of p_x_given_n for these times
            curr_epoch_p_x_given_n = a_masked_result.p_x_given_n_list[i] # [:, :, is_timebin_included]
            curr_epoch_time_bin_centers = a_masked_result.time_bin_containers[i].centers    
            # is_high_prob_mask = (curr_epoch_p_x_given_n > high_val_epsilon)
            curr_epoch_start_t: float = curr_epoch_time_bin_centers[0]
            curr_epoch_stop_t: float = curr_epoch_time_bin_centers[-1]
            
            custom_computation_results_dict = DecodingLocalityMeasures.compute_locality_measures_for_posterior(
                a_p_x_given_n=curr_epoch_p_x_given_n,
                # gaussian_volume=container.predictive_decoding.gaussian_volume, ## if we have it
                xbin_centers=self.predictive_decoding.xbin_centers, 
                ybin_centers=self.predictive_decoding.ybin_centers,
                min_val_epsilon=1e-6,
                alpha_list = [0.5, 0.8],
                enable_debug_outputs=True,
                earthmovers_fn=None, debug_print=False,
            )
            # a_debug_result_dict = custom_computation_results_dict.pop('debug', None) ## remove so it isn't added to the df

            custom_computation_results_df = DecodingLocalityMeasures.perform_build_locality_measures_df(locality_measures_dict_dict={an_epoch_name: custom_computation_results_dict}, ## expects a dict with key of the epoch type, so we need to wrap it
                time_window_centers=curr_epoch_time_bin_centers, 
                xbin_centers=self.predictive_decoding.xbin_centers, 
                ybin_centers=self.predictive_decoding.ybin_centers,
            )

            custom_computation_results_df['epoch_idx'] = i ## same value for all
            custom_computation_results_df['epoch_t_bin_idx'] = custom_computation_results_df.index.astype(int).to_numpy() ## ascending values
            custom_results_list.append(custom_computation_results_dict)
            custom_results_df_list.append(custom_computation_results_df)
        ## END for i, a_row in enumerate(ensure_data...

        ## flatten/concat into a single flat df for all epochs:
        custom_results_df_list = pd.concat(custom_results_df_list, ignore_index=True)
        
        # Promenecen stuff too:
        ## INPUTS: decoded_local_epochs_result
        # old_prom_2d_result = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=decoded_local_epochs_result.p_x_given_n_list, 
        #     xbin_centers=container.predictive_decoding.xbin_centers, 
        #     ybin_centers=container.predictive_decoding.ybin_centers,
        #     # step=1e-3, minimum_included_peak_height=1e-5,
        #     step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
        #     # step=1e-2, minimum_included_peak_height=None, # 1m 42s - 7m 1s
        # )
        # ## 55m - step=1e-4, minimum_included_peak_height=1e-5
        # ## 11m - step=1e-3, minimum_included_peak_height=1e-5,

        step: float = PeakPromenence.compute_optimal_step_size(a_masked_result.p_x_given_n_list, resolution_factor=500.0)
        print(f'step: {step}')

        # decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PosteriorPeaksPeakProminence2dResult.init_from_old_PeakProminence2D_result_dict(active_peak_prominence_2d_results=old_prom_2d_result)
        
        decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=a_masked_result.p_x_given_n_list, 
            xbin_centers=self.predictive_decoding.xbin_centers, 
            ybin_centers=self.predictive_decoding.ybin_centers,
            step=step, minimum_included_peak_height=None, # 1m 42s - 7m 1s
            # step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
            peak_height_multiplier_probe_levels=(0.25, 0.5, 0.9),
            should_use_faster_compute_single_slab_implementation=False,
            min_considered_promenence=1e-11,
        )
        ## 55m - step=1e-4, minimum_included_peak_height=1e-5
        ## 11m - step=1e-3, minimum_included_peak_height=1e-5,

        return a_masked_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj

    ## OUTPUTS: active_epochs_result (masked result),  custom_results_df_list




    # Utility Methods ____________________________________________________________________________________________________ #

    @function_attributes(short_name=None, tags=['PENDING', '2025-01-09'], input_requires=[], output_provides=[], uses=['decode_specific_epochs'], used_by=['compute_future_and_past_analysis'], creation_date='2025-01-09', related_items=[])
    def decode_epochs_for_posterior_analysis(self, curr_active_pipeline, an_epoch_name: str = 'roam', decoding_time_bin_size: float = 0.025, active_epochs_df: Optional[pd.DataFrame] = None) -> Tuple["DecodedFilterEpochsResult", "BayesianPlacemapPositionDecoder"]:
        """Performs fine-grained decoding for each posterior epoch.
        
        This method handles getting or creating the decoder, checking the cache for existing decoded results,
        and performing the decoding if needed. The decoded result is cached for future use.
        
        Args:
            curr_active_pipeline: The active pipeline instance
            an_epoch_name: Name of the epoch to decode (default: 'roam')
            decoding_time_bin_size: Time bin size for decoding (default: 0.025)
            active_epochs_df: DataFrame containing the epochs to decode. If None, will be obtained from the pipeline.
            
        Returns:
            Tuple containing:
                - decoded_local_epochs_result: The decoded result for the specified epochs
                - a_decoder: The decoder used for decoding
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult
        
        # Ensure cache dict exists for this time bin size
        if decoding_time_bin_size not in self.epochs_decoded_result_cache_dict:
            self.epochs_decoded_result_cache_dict[decoding_time_bin_size] = {} ## make the new dict for this time bin size
            print(f'decoding_time_bin_size: {decoding_time_bin_size} did not exist in results... creating!')
        
        # Get or create the decoder
        a_decoder: BayesianPlacemapPositionDecoder = self.pf1D_Decoder_dict.get(an_epoch_name, None)
        if a_decoder is None:
            directional_decoders_decode_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']            
            assert directional_decoders_decode_result is not None
            self.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            a_decoder = directional_decoders_decode_result.pf1D_Decoder_dict[an_epoch_name]
    
        # Check cache for existing decoded result
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[decoding_time_bin_size].get(an_epoch_name, None)
        if decoded_local_epochs_result is None:
            ## if we can't find a pre-computed one:    
            decoded_local_epochs_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=deepcopy(active_epochs_df), decoding_time_bin_size=decoding_time_bin_size)
            self.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name] = decoded_local_epochs_result
            print(f'\tresult added to self.epochs_decoded_result_cache_dict[decoding_time_bin_size={decoding_time_bin_size}][an_epoch_name={an_epoch_name}]')

        return decoded_local_epochs_result, a_decoder



    @function_attributes(short_name=None, tags=['PENDING', 'IN-PROCESS', '2025-12-20_future_and_past_analysis'], input_requires=[], output_provides=[], uses=['decode_specific_epochs'], used_by=[], creation_date='2025-12-19 14:28', related_items=[])
    def compute_future_and_past_analysis(self, curr_active_pipeline, an_epoch_name:str = 'roam', decoding_time_bin_size=0.025, top_v_percent: float = 0.1, 
                                        merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, ## for merging detected future/past position dataframes
                                        enable_updating_instance_states: bool=True,
                                        override_included_analysis_epochs: Optional[pd.DataFrame]=None, **kwargs,
                                        ):
        """ computes the times that 
        
        if enable_updating_instance_states==False, the internal members will not be updated and the new computed values will just be returned
        

        ## Updates
        self.decoding_locality.non_local_PBE_non_moving_epochs_df

        self.predictive_decoding.locality_measures
        self.predictive_decoding.epoch_matching_past_future_positions
        self.predictive_decoding.matching_pos_dfs_list
        self.predictive_decoding.matching_pos_epochs_dfs_list
        


        ## Get the non-local epochs -- where do they encode?

        
        ## Does the animal go there in the futre?
        
        ## Has it been there in the past (duh or we wouldn't have placefields for it)?

        ## Some PBEs that don't qualify as non-local actually might be but they're just paths across the environment.

        ## #TODO 2025-12-19 16:52: - [ ] Let's stick within the same block (roam/sprinkle) for now


        a_test_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=decoded_local_epochs_result, xbin=a_decoder.xbin, xbin_centers=a_decoder.xbin_centers, ybin=a_decoder.ybin, ybin_centers=a_decoder.ybin_centers)
        a_test_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=measured_positions_df, epoch_specific_position_dfs=[relevant_positions_df], epoch_ids=np.array([0]), curr_num_subplots=1, active_page_index=0, plot_actual_lap_lines=True)
        """
        from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior
        

        ## HARDCODED an_epoch_name
        # computed_df_col_name_prefix: str = ''
        # computed_df_col_name_prefix: str = f'{an_epoch_name}_'
        
        ## Get the non-local epochs -- where do they encode?
        # container: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
        decoding_locality: DecodingLocalityMeasures = self.decoding_locality
        
        if (override_included_analysis_epochs is not None):
            active_epochs_df: pd.DataFrame = deepcopy(override_included_analysis_epochs)
        else:
            non_local_PBE_non_moving_epochs_df: pd.DataFrame = decoding_locality.get_non_moving_PBE_non_local_epochs(curr_active_pipeline.sess, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec)
            active_epochs_df: pd.DataFrame = deepcopy(non_local_PBE_non_moving_epochs_df)

        ## add the final detected non_local_pbe_epoch indicies to the decoded points:
        if 'label' not in active_epochs_df.columns:
            active_epochs_df['label'] = active_epochs_df.index.astype(int)
        else:
            active_epochs_df['label'] = active_epochs_df['label'].astype(int)
            
        _out_locality_measures_df = deepcopy(decoding_locality.locality_measures_df)
        _out_locality_measures_df = _out_locality_measures_df.time_point_event.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name='non_local_PBE_non_moving_epoch', override_time_variable_name='t',
                                                            # epoch_label_column_name='label', no_interval_fill_value=np.nan,
                                                            epoch_label_column_name='label', no_interval_fill_value=-1,
                                                            should_replace_existing_column=True, drop_non_epoch_events=True,
                                                            overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
        # _out_locality_measures_df
        # _out_locality_measures_df.dropna(how='any', subset=['non_local_PBE_non_moving_epoch'])

        epoch_times = decoding_locality.locality_measures_df['t'].to_numpy()
        time_to_idx_map = EpochHelpers.find_epoch_times_to_data_indicies_map(active_epochs_df, epoch_times)
        # _out
        active_epochs_df: pd.DataFrame = active_epochs_df
        active_epochs_df['start_idx'] = active_epochs_df['start'].map(time_to_idx_map)
        active_epochs_df['stop_idx'] = active_epochs_df['stop'].map(time_to_idx_map)
        # matching_epoch_times_slice
        # non_local_PBE_non_moving_epochs_dft

        self.active_epochs_df = deepcopy(active_epochs_df)

        # Get the decoders to decode the epochs with higher precision ________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        decoded_local_epochs_result, a_decoder = self.decode_epochs_for_posterior_analysis(curr_active_pipeline=curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=decoding_time_bin_size, active_epochs_df=active_epochs_df)
        print(f'done with all decoding.')

        ## INPUTS: decoded_local_epochs_result
        measured_positions_df: pd.DataFrame = decoding_locality.measured_positions_df        

        gaussian_volume = self.predictive_decoding.gaussian_volume ## the volume for all time bins

        ## decoded_local_epochs_result's epochs need to match the passed `active_epochs_df`
        epoch_matching_past_future_positions, _an_out_tuple, active_epochs_df = PredictiveDecoding.compute_specific_future_and_past_analysis(decoded_local_epochs_result=decoded_local_epochs_result, measured_positions_df=measured_positions_df, gaussian_volume=gaussian_volume,
            active_epochs_df=active_epochs_df,
            an_epoch_name=an_epoch_name, top_v_percent=top_v_percent, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
            **kwargs, # use_parallel=True, max_workers=2, 
        )
        epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _an_out_tuple
        

        ## OUTPUTS: non_local_PBE_non_moving_epochs_df, epoch_matching_past_future_positions, matching_pos_dfs_list, matching_pos_epochs_dfs_list, decoding_locality
        # ==================================================================================================================================================================================================================================================================================== #
        # Update the Internal State Objects                                                                                                                                                                                                                                                    #
        # ==================================================================================================================================================================================================================================================================================== #

        if enable_updating_instance_states:
            ## update the source object
            decoding_locality.non_local_PBE_non_moving_epochs_df = active_epochs_df

            ### assign to predictive_decoding.locality_measures (single source of truth)
            if (self.predictive_decoding is not None):
                self.predictive_decoding.locality_measures = decoding_locality

                ## update the other fields
                self.predictive_decoding.epoch_high_prob_pos_masks = epoch_high_prob_pos_masks
                self.predictive_decoding.epoch_t_bins_high_prob_pos_masks = epoch_t_bins_high_prob_pos_masks
                self.predictive_decoding.epoch_matching_past_future_positions = epoch_matching_past_future_positions
                self.predictive_decoding.matching_pos_dfs_list = matching_pos_dfs_list
                self.predictive_decoding.matching_pos_epochs_dfs_list = matching_pos_epochs_dfs_list


        return epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict #(ratio_past, ratio_future, n_total_past, n_total_future) # , epoch_high_prob_pos_masks




    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove non-serialized fields
        _non_pickled_fields = ['debug_computed_dict', 'scoring_results_df'] # 'active_epochs_df',
        for a_non_pickleable_field in _non_pickled_fields:
            if a_non_pickleable_field in state:
                del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # Restore defaults for non-serialized fields
        _non_pickled_field_restore_defaults = dict(zip(['debug_computed_dict', 'scoring_results_df', 'active_epochs_df'], [{}, None, None]))
        for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
            if a_field_name not in state:
                state[a_field_name] = a_default_restore_value

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
    def perform_predictive_decoding_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, window_size:int=8, extant_decoded_time_bin_size: Optional[float]=None,
                drop_previous_result_and_compute_fresh:bool=False, min_num_spikes_per_bin_to_be_considered_active: Optional[int]=5, mask_position_like_time_score_cutoff: Optional[float] = 0.42, 
                should_perform_first_pass_compute_future_and_past_analysis: bool=False, enable_filter_and_final_result_processing: bool = False):
        """ Performs predictive decoding analysis to relate PBE activity to future visited locations.

        Requires:
            ['DirectionalDecodersDecoded']

        Provides:
            global_computation_results.computed_data['PredictiveDecoding']
                ['PredictiveDecoding'].predictive_decoding - PredictiveDecoding instance containing computed results

                
        Overall Process 2026-01-14:
        
        0. Decode whole session with both decoders at a gross time-scale (250ms)
        1. Determine which of these time-bins decode to non-local places (non-local but position-like posteriors)
            1a. Assign a locality score to each time bin -- representing how well the predicted position corresponds to the animal's actual current position
            1b. Assign a "position-like" score (`PositionLikePosteriorScoring`) to each decoded time bin, saying how much the decoded posterior looks like a valid and well-localized position.
            1c. The epochs of interest are then those contiguous time bins that match this criteria that occur during PBEs -- these are the `target_epochs` (TODO: need name for these epochs)
        2. Decode `target_epochs` at fine time scale (25ms)
        3. For each `target_epochs` epoch, search both forward and backward in time to find times when the animal actually visits one of the positions represented in an epoch time bin.
            4. Find "visit mask" by looking at 25% promenence value (TODO: this doesn't isolate the right peak topographically yeah?)
            #TODO 2026-01-14 06:33: - [ ] instead of smashing down the 25% contour for all time bins into a single mask per epoch, we could seek forward and back in time for each time bin
                - computationally much heavier but would allow us to see sequantial position possibilities across sequential epoch t-bins
            #TODO 2026-01-14 06:35: - [ ] filter by 2D discretized-position angle computed between successive epoch t-bins.
                - Allow parallel OR anti-parallel matches (reverse replay)
                - DRAWBACK: for mostly stationary events, the direction doesn't mean much, could artificially exclude stationary events

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
        ## NEW: filtering by whether decoded posterior in each t_bin is "position-like"
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring


        if include_includelist is not None:
            print(f'WARN: perform_predictive_decoding_analysis(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        ## Get the needed data:
        should_filter_by_active_spikes: bool = ((min_num_spikes_per_bin_to_be_considered_active is not None) and (min_num_spikes_per_bin_to_be_considered_active > 0))
        should_filter_by_position_like_posterior_bins: bool = ((mask_position_like_time_score_cutoff is not None) and (mask_position_like_time_score_cutoff > 0))
        
        # ==================================================================================================================================================================================================================================================================================== #
        # MASK low-firing bins before using result                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        # extant_decoded_time_bin_size = 0.250
        
        if (should_filter_by_active_spikes or should_filter_by_position_like_posterior_bins):
            ## Masked result:
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(owning_pipeline_reference.global_computation_results.computed_data['DirectionalDecodersDecoded'])
            spikes_df: pd.DataFrame = directional_decoders_decode_result.spikes_df

            epoch_names: List[str] = list(directional_decoders_decode_result.pf1D_Decoder_dict.keys())
            # a_decoder = list(directional_decoders_decode_result.pf1D_Decoder_dict.values())[0]
            a_decoder = list(directional_decoders_decode_result.pf1D_Decoder_dict.values())[0]
            
            for extant_decoded_time_bin_size, a_result_decoded in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.items():
                a_result_decoded: SingleEpochDecodedResult = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]
                a_result_decoded: DecodedFilterEpochsResult = DecodedFilterEpochsResult.init_from_single_epoch_result(single_epoch_result=a_result_decoded, decoding_time_bin_size=extant_decoded_time_bin_size) ## convert to a `DecodedFilterEpochsResult` for masking
                
                # FILTER TO JUST POSITION-LIKE POSTERIORS ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                if mask_position_like_time_score_cutoff:
                    a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_result_decoded, position_like_score_cutoff=mask_position_like_time_score_cutoff, num_min_position_like_t_bins=None,
                                                                                                                                        xbin=a_decoder.xbin, ybin=a_decoder.ybin, normalization_across_epochs_epoch_names=epoch_names,
                                                                                                                                                
                                                                                                                                     )
                else:
                    a_masked_result = a_result_decoded


                # FILTER BY ACTIVE SPIKES ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                if should_filter_by_active_spikes:
                    ## TODO: I think this works?
                    a_masked_result, mask_index_tuple = a_masked_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(
                        spikes_df=deepcopy(spikes_df),
                        min_num_spikes_per_bin_to_be_considered_active=min_num_spikes_per_bin_to_be_considered_active,
                        min_num_unique_active_neurons_per_time_bin=1,
                        masked_bin_fill_mode='dropped',
                        # masked_bin_fill_mode='nan_filled'
                    )
                else:
                    # a_masked_result
                    pass                    


                # a_masked_result: DecodedFilterEpochsResult
                # is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list = mask_index_tuple
                ## re-assign to `directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]`
                directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size] = a_masked_result.get_result_for_epoch(0) ## get the single epoch, re-assign

            ## END for extant_decoded_time_bin_size, a_result_decoded in directional_decoders_decode_result.continuousl...

        else:
            directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(global_computation_results.computed_data['DirectionalDecodersDecoded'])
            spikes_df: pd.DataFrame = deepcopy(directional_decoders_decode_result.spikes_df)



        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        
        pos_df: pd.DataFrame = deepcopy(owning_pipeline_reference.sess.position.to_dataframe())
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
        print(f'previously_decoded time_bin_sizes: {previously_decoded_keys}')

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

        locality_measures = None
        try:
            # Create DecodingLocalityMeasures first (required for new interface)
            print(f'[DecodingLocalityMeasures] Initializing DecodingLocalityMeasures with time_bin_size={time_bin_size}...')
            print(f'[DecodingLocalityMeasures] Input validation: directional_decoders_decode_result type={type(directional_decoders_decode_result).__name__}, has continuously_decoded_pseudo2D_decoder_dict={hasattr(directional_decoders_decode_result, "continuously_decoded_pseudo2D_decoder_dict")}')
            
            if not hasattr(directional_decoders_decode_result, 'continuously_decoded_pseudo2D_decoder_dict'):
                raise AttributeError(f"directional_decoders_decode_result missing required attribute 'continuously_decoded_pseudo2D_decoder_dict'. Available attributes: {list(directional_decoders_decode_result.__dict__.keys())[:10]}")
            
            if time_bin_size not in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict:
                available_sizes = list(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.keys())
                raise KeyError(f"time_bin_size={time_bin_size} not found in continuously_decoded_pseudo2D_decoder_dict. Available sizes: {available_sizes}")
            
            print(f'[DecodingLocalityMeasures] Calling init_from_decode_result...')
            locality_measures = DecodingLocalityMeasures.init_from_decode_result(
                curr_active_pipeline=owning_pipeline_reference,
                directional_decoders_decode_result=directional_decoders_decode_result,
                extant_decoded_time_bin_size=time_bin_size,
                sigma=None  # Will be computed automatically if not provided
            )
            
            print(f'[DecodingLocalityMeasures] Successfully initialized. Type: {type(locality_measures).__name__}')
            print(f'[DecodingLocalityMeasures] Checking computed properties: has gaussian_volume={hasattr(locality_measures, "gaussian_volume")}, has p_x_given_n_dict={hasattr(locality_measures, "p_x_given_n_dict")}')
            
            # Compute locality measures to ensure they are fully computed
            # locality_measures.compute()
            # non_local_PBE_non_moving_epochs_df: pd.DataFrame = locality_measures.get_non_moving_PBE_non_local_epochs(owning_pipeline_reference.sess, merging_adjacent_max_separation_sec=0.5)
            if locality_measures is not None:
                global_computation_results.computed_data['PredictiveDecoding'].locality_measures = locality_measures


        except Exception as e:
            print(f'[DecodingLocalityMeasures] error during computation: {e}')
            # if fail_on
            raise


        try:
            # Create DecodingLocalityMeasures first (required for new interface)
            print(f'[PredictiveDecoding] Initializing PredictiveDecoding with time_bin_size={time_bin_size}...')
            # print(f'[PredictiveDecoding] Input validation: directional_decoders_decode_result type={type(directional_decoders_decode_result).__name__}, has continuously_decoded_pseudo2D_decoder_dict={hasattr(directional_decoders_decode_result, "continuously_decoded_pseudo2D_decoder_dict")}')
            
            if locality_measures is None:
                locality_measures = global_computation_results.computed_data['PredictiveDecoding'].locality_measures

            assert locality_measures is not None

            # Get a_result_decoded from directional_decoders_decode_result
            a_result_decoded = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[time_bin_size]
            
            ## INPUTS: a_result_decoded, locality_measures, pos_df
            
            print(f'[PredictiveDecoding] Calling init_from_decode_result...')

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
            print(f'[PredictiveDecoding] Successfully initialized. Type: {type(predictive_decoding).__name__}')

            # Use sigma from locality_measures (computed automatically) or compute from bin sizes if not available
            if locality_measures.sigma is None:
                x_step: float = np.nanmean(np.diff(predictive_decoding.xbin))
                y_step: float = np.nanmean(np.diff(predictive_decoding.ybin))
                sigma: float = np.nanmax([x_step, y_step]) * 5.0
                print(f'computed sigma from bin sizes: {sigma}')
            else:
                sigma = locality_measures.sigma
                print(f'using sigma from locality_measures: {sigma}')

            print(f'\t[PredictiveDecoding] computing via .compute(sigma={sigma})...')
            # Compute predictive decoding outputs
            moving_avg_dict, moving_avg_meas_pos_overlap_dict, gaussian_volume = predictive_decoding.compute(sigma=sigma) ## This line might be the slow one
            print('\tdone computing!')
            
            # print(f'[PredictiveDecoding] Checking computed properties: has gaussian_volume={hasattr(predictive_decoding, "gaussian_volume")}, has p_x_given_n_dict={hasattr(predictive_decoding, "p_x_given_n_dict")}')
            
            # Compute locality measures to ensure they are fully computed
            # locality_measures.compute()
            # non_local_PBE_non_moving_epochs_df: pd.DataFrame = locality_measures.get_non_moving_PBE_non_local_epochs(owning_pipeline_reference.sess, merging_adjacent_max_separation_sec=0.5)
            
            if predictive_decoding is not None:
                # Store the PredictiveDecoding instance in the container
                global_computation_results.computed_data['PredictiveDecoding'].predictive_decoding = predictive_decoding

        except Exception as e:
            print(f'[PredictiveDecoding] error during computation: {e}')
            # if fail_on
            raise



        if include_includelist is None:
            include_includelist = ['roam'] # , 'sprinkle'
        epoch_names: List[str] = include_includelist 
        print(f'\t processing will occur for epoch_names: {epoch_names}')


        # compute_future_and_past_analysis ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if should_perform_first_pass_compute_future_and_past_analysis:
            for an_epoch_name in epoch_names:    
                try:
                    print(f'\ttrying `.compute_future_and_past_analysis(...)` for an_epoch_name: "{an_epoch_name}"...')
                    if an_epoch_name not in global_computation_results.computed_data['PredictiveDecoding'].debug_computed_dict:
                        global_computation_results.computed_data['PredictiveDecoding'].debug_computed_dict[an_epoch_name] = {}
                    # active_epochs_df
                    # _out = global_computation_results.computed_data['PredictiveDecoding'].compute_future_and_past_analysis(owning_pipeline_reference, an_epoch_name=an_epoch_name)
                    _out = global_computation_results.computed_data['PredictiveDecoding'].compute_future_and_past_analysis(owning_pipeline_reference, an_epoch_name=an_epoch_name, disable_segmentation=True) ## #TODO 2026-01-15 02:06: - [ ] This is what's wasting all the memory ## `, should_defer_extended_computations=should_defer_extended_computations`
                    epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _out ## too many to unpack?
                    global_computation_results.computed_data['PredictiveDecoding'].debug_computed_dict[an_epoch_name].update({'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict})
                except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
                    print(f'\t\tWARN: the `should_perform_first_pass_compute_future_and_past_analysis` part of `perform_predictive_decoding_analysis(...) failed with error: {e}. Skipping.')
                    pass
                except Exception as e:
                    raise
                
            ## END for an_epoch_name in epoch_names...
        else:
            print(f'should_perform_first_pass_compute_future_and_past_analysis == False, so skipping those comps.')
        
        
        if enable_filter_and_final_result_processing:
            # Validate container exists
            container = global_computation_results.computed_data.get('PredictiveDecoding', None)
            assert container is not None

            masked_container = container.build_masked_container(curr_active_pipeline=owning_pipeline_reference, should_filter_directional_decoders_decode_result=True, should_compute_future_and_past_analysis=False, should_compute_peak_prom_analysis=False) ## 3m now
            
            for an_epoch_name in epoch_names:    
                try:
                    print(f'\ttrying `.masked_container._filter_single_epoch_result(...)` for an_epoch_name: "{an_epoch_name}"...')
                    if an_epoch_name not in masked_container.debug_computed_dict:
                        masked_container.debug_computed_dict[an_epoch_name] = {}
                    active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container._filter_single_epoch_result(curr_active_pipeline=owning_pipeline_reference, decoding_time_bin_size=time_bin_size, an_epoch_name=an_epoch_name)
                    masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
                except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
                    print(f'\t\tWARN: the `enable_filter_and_final_result_processing` part of `perform_predictive_decoding_analysis(...) failed with error: {e}. Skipping.')
                    pass
                except Exception as e:
                    raise
            ## END for an_epoch_name in epoch_names...
        else:
            print(f'enable_filter_and_final_result_processing == False, so skipping those comps.')
            

        ## Now filter

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
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyphoplacecellanalysis.External.pyqtgraph.dockarea import Dock, DockArea
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget


@metadata_attributes(short_name=None, tags=['partially-working', 'matplotlib', '3-pane', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 14:42', related_items=[])
@define(slots=False, repr=False, eq=False)
class PredictiveDecodingDisplayWidget:
    """ Plots 3 panels side-by-side: Left: Past positions, Mid: Decoded Epoch Posterior, Right: Future positions
    
    Internally-Uses:
        epoch_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_high_prob_pos_masks', None)

                self.decoded_result 


    
    Usage:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidget

        a_widget: PredictiveDecodingDisplayWidget = PredictiveDecodingDisplayWidget.init_from_container(container=container, decoding_time_bin_size=0.025, an_epoch_name='roam')
        a_widget
        
        
        
    ## FILTERED VERSION

        # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=decoded_local_epochs_result, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3,
                                                                                                                                        xbin=a_decoder.xbin, ybin=a_decoder.ybin,
                                                                                                                                     )


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
    trajectory_displaying_plotter: Dict[types.PastFutureCategory, DecodedTrajectoryMatplotlibPlotter] = field(default=Factory(dict))
    
    ## Dock UI Variables
    dock_area: Any = field(default=None)
    dock_window: Any = field(default=None)
    dock_widgets: Dict[str, Any] = field(default=Factory(dict))
    dock_canvas_widgets: Dict[str, Any] = field(default=Factory(dict))
    epoch_slider: Any = field(default=None)
    epoch_value_label: Any = field(default=None)

    active_epoch_idx: int = field(default=20)
    
    disable_showing_epoch_high_prob_pos_masks: bool = field(default=True)
    should_use_flipped_images: bool = field(default=False)
    

    @classmethod
    def init_from_container(cls, container: PredictiveDecodingComputationsContainer, decoding_time_bin_size: float, an_epoch_name: str, active_epoch_idx: int=0, **kwargs) -> "PredictiveDecodingDisplayWidget":
        """

        """
        decoded_local_epochs_result = container.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name]
        pf_decoder = container.pf1D_Decoder_dict[an_epoch_name]
        decoded_result: DecodedFilterEpochsResult = decoded_local_epochs_result
        curr_position_df: pd.DataFrame = deepcopy(container.decoding_locality.pos_df)

        # for k, v in kwargs.items():
        #     disable_showing_epoch_high_prob_pos_masks

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
        """Basic validation, then call setup() and buildUI()."""
        assert len(self.container.predictive_decoding.matching_pos_dfs_list) > 0
        assert len(self.container.predictive_decoding.matching_pos_epochs_dfs_list) > 0
        self.setup()
        self.buildUI()


    def setup(self):
        """Calculate constants (max_subplots_per_category, extent), prepare data structures."""
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        
        # Prepare matching_pos_epochs_dfs_list with is_future_present_past labels
        for i, a_row in enumerate(ensure_dataframe(self.decoded_result.filter_epochs).itertuples(index=False)):
            a_matching_pos_epochs: pd.DataFrame = matching_pos_epochs_dfs_list[i]
            curr_epoch_start_t: float = a_row.start
            curr_epoch_stop_t: float = a_row.stop
            
            is_relevant_past_times = (a_matching_pos_epochs['stop'] < curr_epoch_start_t)
            is_relevant_future_times = (a_matching_pos_epochs['start'] > curr_epoch_stop_t)
            a_matching_pos_epochs['is_future_present_past'] = 'present'
            a_matching_pos_epochs.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
            a_matching_pos_epochs.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'
            
            self.container.predictive_decoding.matching_pos_epochs_dfs_list[i] = a_matching_pos_epochs # updated in the widget? very strange.
        
        # Calculate max_subplots_per_category
        self.max_subplots_per_category = self._calculate_max_subplots()
        
        # Calculate extent for posterior plots
        self.extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
        
        # Initialize display_widgets dict for MatplotlibTimeSynchronizedWidget instances
        if not hasattr(self, 'display_widgets'):
            self.display_widgets: Dict[str, Any] = {}


    def _calculate_max_subplots(self) -> Dict[str, int]:
        """Pre-calculate max subplots needed (called once in setup)."""
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        
        max_subplots_per_category: Dict[str, int] = {}
        for epoch_idx in range(len(matching_pos_epochs_dfs_list)):
            curr_matching_epochs_df_temp: pd.DataFrame = matching_pos_dfs_list[epoch_idx]
            curr_matching_epochs_df_dict_temp: Dict[int, pd.DataFrame] = curr_matching_epochs_df_temp.pho.partition_df_dict('is_future_present_past')
            for a_past_future_name, an_epoch_specific_dfs in curr_matching_epochs_df_dict_temp.items():
                if a_past_future_name not in max_subplots_per_category:
                    max_subplots_per_category[a_past_future_name] = 0
                num_items = len(an_epoch_specific_dfs)
                max_subplots_per_category[a_past_future_name] = max(max_subplots_per_category[a_past_future_name], num_items)
        
        # Cap at 20 subplots maximum
        for key in max_subplots_per_category:
            max_subplots_per_category[key] = min(20, max_subplots_per_category[key])
        
        return max_subplots_per_category


    def buildUI(self):
        """Create dock area and initialize ALL three widgets immediately."""
        self._build_dock_area()
        self._build_past_widget()
        self._build_posterior_widget()
        self._build_future_widget()
        self._build_epoch_control()
        self.dock_window.show()
        self.update_displayed_epoch(an_epoch_idx=self.active_epoch_idx)


    def _build_dock_area(self):
        """Create window and dock area."""
        self.dock_window = QtWidgets.QMainWindow()
        self.dock_window.setWindowTitle("Predictive Decoding Display - Past/Future Trajectories")
        self.dock_area = DockArea()
        self.dock_window.setCentralWidget(self.dock_area)
        self.dock_window.resize(1400, 800)


    def _build_past_widget(self):
        """Create past trajectory widget (MatplotlibTimeSynchronizedWidget)."""        
        dock = Dock("Past Trajectories", size=(600, 700), closable=True)
        self.dock_area.addDock(dock, 'left')
        self.dock_widgets['past'] = dock
        
        # Create and initialize the widget immediately
        widget = MatplotlibTimeSynchronizedWidget(size=(8, 6), dpi=72, constrained_layout=True, disable_toolbar=False)
        dock.addWidget(widget)
        self.display_widgets['past'] = widget
        
        # Create trajectory plotter
        overlay_posterior, _, _ = self._get_posterior_data(self.active_epoch_idx)
        overlay_prev_heatmaps = [overlay_posterior] if overlay_posterior is not None else []
        plotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers, prev_heatmaps=overlay_prev_heatmaps)
        self.trajectory_displaying_plotter['past'] = plotter


    def _build_posterior_widget(self):
        """Create decoded posterior widget (MatplotlibTimeSynchronizedWidget)."""
        dock = Dock("Decoded Posterior", size=(600, 700), closable=True)
        prev_dock = self.dock_widgets.get('past')
        if prev_dock is not None:
            self.dock_area.addDock(dock, 'right', prev_dock)
        else:
            self.dock_area.addDock(dock, 'left')
        self.dock_widgets['decoded_posterior'] = dock
        
        # Create and initialize the widget immediately
        widget = MatplotlibTimeSynchronizedWidget(size=(8, 6), dpi=72, constrained_layout=True, disable_toolbar=False)
        dock.addWidget(widget)
        self.display_widgets['decoded_posterior'] = widget


    def _build_future_widget(self):
        """Create future trajectory widget (MatplotlibTimeSynchronizedWidget)."""
        dock = Dock("Future Trajectories", size=(600, 700), closable=True)
        prev_dock = self.dock_widgets.get('decoded_posterior')
        if prev_dock is not None:
            self.dock_area.addDock(dock, 'right', prev_dock)
        else:
            self.dock_area.addDock(dock, 'left')
        self.dock_widgets['future'] = dock
        
        # Create and initialize the widget immediately
        widget = MatplotlibTimeSynchronizedWidget(size=(8, 6), dpi=72, constrained_layout=True, disable_toolbar=False)
        dock.addWidget(widget)
        self.display_widgets['future'] = widget
        
        # Create trajectory plotter
        overlay_posterior, _, _ = self._get_posterior_data(self.active_epoch_idx)
        overlay_prev_heatmaps = [overlay_posterior] if overlay_posterior is not None else []
        plotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers, prev_heatmaps=overlay_prev_heatmaps)
        self.trajectory_displaying_plotter['future'] = plotter


    def _build_epoch_control(self):
        """Create slider controls."""        
        num_epochs = len(ensure_dataframe(self.decoded_result.filter_epochs))
        if num_epochs > 0:
            bottom_widget = QtWidgets.QWidget()
            bottom_layout = QtWidgets.QHBoxLayout()
            bottom_layout.setContentsMargins(10, 5, 10, 5)
            
            slider_label = QtWidgets.QLabel("Epoch Index:")
            bottom_layout.addWidget(slider_label)
            
            self.epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.epoch_slider.setMinimum(0)
            self.epoch_slider.setMaximum(max(0, num_epochs - 1))
            self.epoch_slider.setValue(min(self.active_epoch_idx, num_epochs - 1))
            self.epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            self.epoch_slider.setTickInterval(1)
            self.epoch_slider.setMinimumWidth(400)
            bottom_layout.addWidget(self.epoch_slider, stretch=1)
            
            self.epoch_value_label = QtWidgets.QLabel(f"{self.epoch_slider.value()}")
            self.epoch_value_label.setMinimumWidth(50)
            self.epoch_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            bottom_layout.addWidget(self.epoch_value_label)
            
            self.epoch_slider.valueChanged.connect(self._on_slider_value_changed_label_only)
            self.epoch_slider.sliderReleased.connect(self._on_slider_released)
            
            bottom_widget.setLayout(bottom_layout)
            
            bottom_dock = Dock("Epoch Control", size=(1400, 50), closable=False)
            bottom_dock.addWidget(bottom_widget)
            self.dock_area.addDock(bottom_dock, 'bottom')
            self.dock_widgets['epoch_control'] = bottom_dock


    def _on_slider_value_changed_label_only(self, value: int):
        """Handle slider value change to update only the label (for immediate feedback while dragging)."""
        if self.epoch_value_label is not None:
            self.epoch_value_label.setText(f"{value}")
        if (self.epoch_slider is not None) and (not self.epoch_slider.isSliderDown()):
            self.update_displayed_epoch(an_epoch_idx=value)
            for widget in self.display_widgets.values():
                if widget is not None and hasattr(widget, 'draw'):
                    widget.draw()


    def _on_slider_released(self):
        """Handle slider release to update the displayed epoch."""
        if self.epoch_slider is not None:
            value = self.epoch_slider.value()
            self.update_displayed_epoch(an_epoch_idx=value)
            # Update all widgets to ensure they refresh
            for widget in self.display_widgets.values():
                if widget is not None and hasattr(widget, 'draw'):
                    widget.draw()


    def _validate_epoch_idx(self, an_epoch_idx: int) -> int:
        """Validate and clamp epoch index."""
        num_epochs = len(ensure_dataframe(self.decoded_result.filter_epochs))
        if an_epoch_idx < 0 or an_epoch_idx >= num_epochs:
            print(f"Warning: epoch_idx {an_epoch_idx} is out of bounds (0-{num_epochs-1}). Clamping to valid range.")
            an_epoch_idx = max(0, min(an_epoch_idx, num_epochs - 1))
        return an_epoch_idx


    def _prepare_epoch_data(self, an_epoch_idx: int) -> Dict[str, Any]:
        """Extract and prepare data for current epoch."""
        matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        
        curr_matching_epochs_df: pd.DataFrame = matching_pos_epochs_dfs_list[an_epoch_idx]
        curr_matching_positions_df: pd.DataFrame = matching_pos_dfs_list[an_epoch_idx]
        curr_matching_epochs_df_dict: Dict[int, pd.DataFrame] = curr_matching_epochs_df.pho.partition_df_dict('is_future_present_past')
        
        curr_matching_past_future_positions_df_dict: Dict[str, Dict[int, pd.DataFrame]] = {}
        
        for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
            a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
            an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int)
            col_name: str = 'past_future_matching_pos_epoch_id'
            a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name, override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True, drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
            curr_matching_positions_df_dict: Dict[int, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
            curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
        
        return {
            'curr_matching_epochs_df': curr_matching_epochs_df,
            'curr_matching_positions_df': curr_matching_positions_df,
            'curr_matching_epochs_df_dict': curr_matching_epochs_df_dict,
            'curr_matching_past_future_positions_df_dict': curr_matching_past_future_positions_df_dict,
        }



    def _get_posterior_data(self, an_epoch_idx: int, get_high_prob_mask_instead: bool=False, should_use_flipped_images: Optional[bool]=None) -> Tuple[np.ndarray, Optional[List[np.ndarray]], int]:
        """Extract posterior data for epoch.

        
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx=an_epoch_idx)
        
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=False)
        
        mask_2d, time_bin_masks, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=True)
        
        """
        should_use_flipped_images: bool = should_use_flipped_images or self.should_use_flipped_images ## use self.should_use_flpped_images if no override provided.
        
        should_get_posterior: bool = (not get_high_prob_mask_instead)
        get_high_prob_mask_instead = get_high_prob_mask_instead or (not self.disable_showing_epoch_high_prob_pos_masks)

        p_x_given_n = None
        posterior_2d = None
        
        if get_high_prob_mask_instead:
            epoch_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_high_prob_pos_masks', None)
            if (epoch_high_prob_pos_masks is not None):
                print(f'using high_prob mask version from .epoch_high_prob_pos_masks!')
                posterior_2d = epoch_high_prob_pos_masks[an_epoch_idx]
            else:
                should_get_posterior = True

        if should_get_posterior:
            p_x_given_n = self.decoded_result.p_x_given_n_list[an_epoch_idx]  # Shape: (n_x_bins, n_y_bins, n_time_bins)
            posterior_2d = np.sum(p_x_given_n, axis=2) ## collapse over time

        time_bin_posteriors = None
        num_time_bins_to_show = 0
        # epoch_t_bins_high_prob_pos_masks
        epoch_t_bins_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_t_bins_high_prob_pos_masks', None)
        if (epoch_t_bins_high_prob_pos_masks is not None) and (not self.disable_showing_epoch_high_prob_pos_masks):
            print(f'using high_prob mask version from .epoch_t_bins_high_prob_pos_masks!')
            time_bin_posteriors = epoch_t_bins_high_prob_pos_masks[an_epoch_idx]
            num_time_bins = time_bin_posteriors.shape[2]
            num_time_bins_to_show = min(10, num_time_bins)
            time_bin_posteriors = [time_bin_posteriors[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]
                    
        else:
            ## Use raw posteriors:
            if p_x_given_n is not None:
                num_time_bins = p_x_given_n.shape[2]
                num_time_bins_to_show = min(10, num_time_bins)
                time_bin_posteriors = [p_x_given_n[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]


        if not should_use_flipped_images:
            # Normal Image/Extent:
            # active_extent = self.extent
            active_posterior = posterior_2d
        else:
            ## Flipped Image/Extent:
            active_posterior = posterior_2d.T
            # time_bin_posteriors = [p_x_given_n[:, :, t_bin_idx].T for t_bin_idx in range(num_time_bins_to_show)] ## flipped posteriors
            time_bin_posteriors = [time_bin_posteriors[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]  ## flipped posteriors
            
            # # Swap extent: (x_min, x_max, y_min, y_max) -> (y_min, y_max, x_min, x_max)
            # x_min, x_max, y_min, y_max = self.extent
            # swapped_extent = (y_min, y_max, x_min, x_max)
            # active_extent = swapped_extent

        return active_posterior, time_bin_posteriors, num_time_bins_to_show


    def _update_posterior_plot(self, widget, posterior_2d: np.ndarray, time_bin_posteriors: Optional[List[np.ndarray]], num_time_bins_to_show: int, an_epoch_idx: int,
                                        posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=None,
                                        posterior_should_perform_reshape=False, extent=None, overlay_posterior_2d: Optional[NDArray]=None, show_overlay=False, overlay_alpha = 0.08, debug_print=True, **kwargs):
        """Update posterior plot with configurable parameters.
        
        Args:
            widget: The matplotlib widget to update
            posterior_2d: 2D posterior array
            time_bin_posteriors: Optional list of time bin posterior arrays
            num_time_bins_to_show: Number of time bins to display
            an_epoch_idx: Epoch index for title
            posterior_alpha: Opacity for posterior heatmaps (default: 0.65)
            posterior_cmap: Colormap name (default: 'Greens')
            posterior_masking_value: Minimum value to display (default: 1e-3)
            posterior_should_perform_reshape: Whether to reshape (default: False)
            extent: Optional extent tuple, defaults to self.extent if None
            show_overlay: Whether to show main posterior overlay on time bin heatmaps (default: True)
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        
        # Disable interactive mode to prevent temporary figures from appearing
        was_interactive = plt.isinteractive()
        plt.ioff()
        try:
            fig = widget.getFigure()
            fig.clear()
            
            if time_bin_posteriors is not None and num_time_bins_to_show > 0:
                gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 2], hspace=0.1)
                ax_main = fig.add_subplot(gs[0, 0])
            else:
                ax_main = fig.add_subplot(111)
            
            # Use _helper_add_heatmap for consistent display with past/future panes
            xbin_centers = self.xbin_centers if self.xbin_centers is not None else self.xbin
            ybin_centers = self.ybin_centers if self.ybin_centers is not None else self.ybin
            

            ## where does self.extent come from? self.extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
            # Use provided extent or fall back to self.extent
            posterior_extent = extent if (extent is not None) else self.extent
            # posterior_should_use_flipped: bool = self.should_use_flipped_images
            overlay_alpha = 0.5
            overlay_cmap = kwargs.pop('overlay_cmap', 'Greens')

            posterior_should_use_flipped: bool = False
            print(f'posterior_extent: {posterior_extent}')
            print(f'posterior_should_use_flipped: {posterior_should_use_flipped}')
            
            # Plot main posterior using _helper_add_heatmap
            heatmaps_main, image_extent, plots_data = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                an_ax=ax_main,
                xbin_centers=xbin_centers,
                ybin_centers=ybin_centers,
                a_p_x_given_n=posterior_2d,
                a_time_bin_centers=None,
                rotate_to_vertical=posterior_should_use_flipped,
                custom_image_extent=posterior_extent,
                time_cmap=posterior_cmap,
                should_perform_reshape=posterior_should_perform_reshape,
                posterior_masking_value=posterior_masking_value,
                full_posterior_opacity=posterior_alpha,
                debug_print=debug_print,
            )
            

            # Add overlay of main posterior with low alpha (if enabled)
            if show_overlay:
                if overlay_posterior_2d is None:
                    overlay_posterior_2d = posterior_2d ## use the posterior if none provided
                    
                if debug_print:
                    print(f'posterior_2d.shape: {np.shape(posterior_2d)}')
                    print(f'overlay_posterior_2d.shape: {np.shape(overlay_posterior_2d)}')
                
                heatmaps_overlay_main, _, _ = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                    an_ax=ax_main,
                    xbin_centers=xbin_centers,
                    ybin_centers=ybin_centers,
                    a_p_x_given_n=overlay_posterior_2d,
                    a_time_bin_centers=None,
                    rotate_to_vertical=posterior_should_use_flipped,
                    custom_image_extent=posterior_extent,
                    time_cmap=overlay_cmap,
                    should_perform_reshape=posterior_should_perform_reshape,
                    posterior_masking_value=None,
                    full_posterior_opacity=overlay_alpha
                )
                

            ax_main.set_xlabel('X Position')
            ax_main.set_ylabel('Y Position')
            ax_main.set_title(f'Decoded Posterior Heatmap - Epoch {an_epoch_idx}')
            
            if (time_bin_posteriors is not None) and (num_time_bins_to_show > 0):
                all_time_bin_values = np.concatenate([tb.flatten() for tb in time_bin_posteriors])
                vmin_shared = np.nanmin(all_time_bin_values)
                vmax_shared = np.nanmax(all_time_bin_values)
                
                
                gs_tiny = gridspec.GridSpecFromSubplotSpec(1, num_time_bins_to_show, subplot_spec=gs[1, 0], wspace=0.01)
                
                for t_bin_idx in range(num_time_bins_to_show):
                    ax_tiny = fig.add_subplot(gs_tiny[0, t_bin_idx])
                    
                    # Plot time bin posterior using _helper_add_heatmap
                    heatmaps_tiny, _, _ = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                        an_ax=ax_tiny,
                        xbin_centers=xbin_centers,
                        ybin_centers=ybin_centers,
                        a_p_x_given_n=time_bin_posteriors[t_bin_idx],
                        a_time_bin_centers=None,
                        rotate_to_vertical=posterior_should_use_flipped,
                        custom_image_extent=posterior_extent,
                        time_cmap=posterior_cmap,
                        should_perform_reshape=posterior_should_perform_reshape,
                        posterior_masking_value=posterior_masking_value,
                        full_posterior_opacity=posterior_alpha
                    )
                    
                    # Apply shared color scale to time bin heatmap
                    if heatmaps_tiny and len(heatmaps_tiny) > 0:
                        heatmaps_tiny[0].set_clim(vmin=vmin_shared, vmax=vmax_shared)
                    
                    # Add overlay of main posterior with low alpha (if enabled)
                    if show_overlay:
                        if overlay_posterior_2d is None:
                            overlay_posterior_2d = posterior_2d ## use the posterior if none provided                        

                        heatmaps_overlay, _, _ = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                            an_ax=ax_tiny,
                            xbin_centers=xbin_centers,
                            ybin_centers=ybin_centers,
                            a_p_x_given_n=overlay_posterior_2d,
                            a_time_bin_centers=None,
                            rotate_to_vertical=posterior_should_use_flipped,
                            custom_image_extent=posterior_extent,
                            time_cmap=posterior_cmap,
                            should_perform_reshape=posterior_should_perform_reshape,
                            posterior_masking_value=None,
                            full_posterior_opacity=overlay_alpha
                        )
                    
                    ax_tiny.set_xticks([])
                    ax_tiny.set_yticks([])
                    ax_tiny.set_xlabel(f't={t_bin_idx}', fontsize=8)
            
            widget.draw()
        finally:
            # Restore previous interactive state
            if was_interactive:
                plt.ion()


    def _update_past_widget(self, an_epoch_idx: int, epoch_data: Dict[str, Any]):
        """Update past trajectory display."""
        self._update_trajectory_widget('past', an_epoch_idx, epoch_data)


    def _update_future_widget(self, an_epoch_idx: int, epoch_data: Dict[str, Any]):
        """Update future trajectory display."""
        self._update_trajectory_widget('future', an_epoch_idx, epoch_data)


    def _update_trajectory_widget(self, a_past_future_name: str, an_epoch_idx: int, epoch_data: Dict[str, Any]):
        """Update trajectory widget for past or future."""
        existing_ax = None
        needed_init: bool = False
        curr_matching_past_future_positions_df_dict = epoch_data['curr_matching_past_future_positions_df_dict']
        
        if a_past_future_name not in curr_matching_past_future_positions_df_dict:
            # raise NotImplementedError(f'a_past_future_name: {a_past_future_name} not in curr_matching_past_future_positions_df_dict: {list(curr_matching_past_future_positions_df_dict.keys())}')
            return
        
        curr_matching_positions_df_dict = curr_matching_past_future_positions_df_dict[a_past_future_name]
        epoch_specific_position_dfs = list(curr_matching_positions_df_dict.values())
        found_pos_segment_ids = np.array(list(curr_matching_positions_df_dict.keys())) # (0, 1, ...)
        
        curr_num_subplots: int = self.max_subplots_per_category.get(a_past_future_name, min(20, len(found_pos_segment_ids)))

        ## Pad the end of the subplots to make them empty        
        if len(epoch_specific_position_dfs) < curr_num_subplots:
            num_to_pad = curr_num_subplots - len(epoch_specific_position_dfs)
            if len(epoch_specific_position_dfs) > 0:
                template_df = epoch_specific_position_dfs[0]
                dummy_row = {col: np.nan for col in template_df.columns}
                empty_df = pd.DataFrame([dummy_row], columns=template_df.columns)
            else:
                empty_df = pd.DataFrame([{'t': np.nan, 'x': np.nan, 'y': np.nan, 'binned_x': np.nan, 'binned_y': np.nan}])
            epoch_specific_position_dfs.extend([empty_df.copy() for _ in range(num_to_pad)])
            found_pos_segment_ids = np.concatenate([found_pos_segment_ids, np.full(num_to_pad, -1, dtype=found_pos_segment_ids.dtype)])
        
        a_decoded_traj_plotter = self.trajectory_displaying_plotter.get(a_past_future_name, None)        
        if a_decoded_traj_plotter is None:
            ## create a new plotter
            a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers)
            self.trajectory_displaying_plotter[a_past_future_name] = a_decoded_traj_plotter
            needed_init = True
            existing_ax = a_decoded_traj_plotter.axs

        else:
            existing_ax = a_decoded_traj_plotter.axs
            

        # Clear existing axes before plotting to prevent drawing over previous plots
        if( existing_ax is not None) and (not needed_init):
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
                    

        ## Get posterior data:
        overlay_posterior, _, _ = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=True)
        a_decoded_traj_plotter.prev_heatmaps = [overlay_posterior] if overlay_posterior is not None else [] # seems stupid
        
        ## NOTE: `epoch_ids` used here and in the following function call actually refer to `found_pos_segment_ids`, not epochs, it's just how the `a_decoded_traj_plotter` class is named:
        fig, axs, epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=self.curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=found_pos_segment_ids, curr_num_subplots=curr_num_subplots,
                                                                                        active_page_index=0, fixed_columns=4, plot_actual_lap_lines=True, use_theoretical_tracks_instead=False, existing_ax=existing_ax,
                                                                                        plot_mode='scatter', c='red', cmap='Reds', alpha=0.55, s=5, posteriors=overlay_posterior, posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=1e-3,
                                                                                        posterior_should_perform_reshape=False, # rotate_to_vertical
                                                                                    )
        
        # Set visibility for all axes (hide unused axes where epoch_id == -1, indicating padded/empty data)
        if axs is not None and isinstance(axs, np.ndarray) and axs.ndim == 2:
            # First, make all axes visible to reset any previously hidden axes
            for row in range(axs.shape[0]):
                for col in range(axs.shape[1]):
                    if axs[row, col] is not None:
                        axs[row, col].set_visible(True)
            
            # Then hide unused axes (where epoch_id == -1)
            if len(epochs_pages) > 0:
                active_page_epoch_ids = epochs_pages[0]
                if hasattr(a_decoded_traj_plotter, 'row_column_indicies') and a_decoded_traj_plotter.row_column_indicies is not None:
                    row_column_indicies = a_decoded_traj_plotter.row_column_indicies
                    for linear_idx, epoch_id in enumerate(active_page_epoch_ids):
                        if (epoch_id == -1):
                            if linear_idx < len(row_column_indicies[0]) and linear_idx < len(row_column_indicies[1]):
                                curr_row = row_column_indicies[0][linear_idx]
                                curr_col = row_column_indicies[1][linear_idx]
                                if curr_row < axs.shape[0] and curr_col < axs.shape[1]:
                                    axs[curr_row, curr_col].set_visible(False)
        
        perform_update_title_subtitle(fig=fig, ax=None, title_string=f"{a_past_future_name} - an_epoch_idx: {an_epoch_idx}")
        

        # Embed the matplotlib figure in the dock widget
        # Check if canvas widget needs to be created (either plotter is new or canvas doesn't exist)
        canvas_needs_init = needed_init or (a_past_future_name not in self.dock_canvas_widgets)
        if canvas_needs_init:
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
            canvas = self.dock_canvas_widgets.get(a_past_future_name)
            if canvas is not None:
                # The axes were already cleared before plot_decoded_trajectories_2d was called
                # Just trigger a redraw
                canvas.draw_idle()

        ## alternative to the above?
        widget = self.display_widgets.get(a_past_future_name)
        if widget is not None:
            widget.draw()


    def _update_posterior_widget(self, an_epoch_idx: int):
        """Update decoded posterior display."""
        widget = self.display_widgets.get('decoded_posterior')
        if widget is None:
            return
        
        override_should_use_flipped_images: bool = False
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=False, should_use_flipped_images=override_should_use_flipped_images)
        mask_2d, time_bin_masks, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=True)
        
        try:
            self._update_posterior_plot(widget, posterior_2d=posterior_2d, time_bin_posteriors=time_bin_posteriors, num_time_bins_to_show=num_time_bins_to_show, an_epoch_idx=an_epoch_idx, overlay_posterior_2d=mask_2d, posterior_cmap='Greens', posterior_alpha=0.95, show_overlay=True)
        except Exception as e:
            print(f"Error updating posterior plot for epoch {an_epoch_idx}: {e}")
            import traceback
            traceback.print_exc()




    @function_attributes(short_name=None, tags=['widget', 'GUI', 'display', 'interactive', 'position-like', 'pred', 'prospective'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryMatplotlibPlotter'], used_by=[], creation_date='2026-01-09 02:04', related_items=[])
    def update_displayed_epoch(self, an_epoch_idx: int = 8):
        """Main entry point - validate, prepare data, update all widgets."""
        # Validate epoch index
        an_epoch_idx = self._validate_epoch_idx(an_epoch_idx)
        
        # Update slider value if it exists (block signals to avoid recursion)
        if self.epoch_slider is not None:
            self.epoch_slider.blockSignals(True)
            self.epoch_slider.setValue(an_epoch_idx)
            self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText(f"{an_epoch_idx}")
        
        # Prepare epoch data
        epoch_data = self._prepare_epoch_data(an_epoch_idx)
        
        # Update all widgets
        self._update_past_widget(an_epoch_idx, epoch_data)
        self._update_posterior_widget(an_epoch_idx)
        self._update_future_widget(an_epoch_idx, epoch_data)
        
        # Update active epoch index
        self.active_epoch_idx = an_epoch_idx
        if self.epoch_value_label is not None:
            self.epoch_value_label.setText(f"{an_epoch_idx}")
                
        assert len(self.container.predictive_decoding.matching_pos_dfs_list) > 0
        matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        assert len(self.container.predictive_decoding.matching_pos_epochs_dfs_list) > 0
        matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list

        self.container.active_epochs_df
        
        # Calculate maximum number of subplots needed across all epochs for each category -- WTF is a category??
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
        
        # ==================================================================================================================================================================================================================================================================================== #
        # TWO (Left, Right) Panes corresponding to the past and future positions                                                                                                                                                                                                               #
        # ==================================================================================================================================================================================================================================================================================== #
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
            found_pos_segment_ids = np.array(list(curr_matching_past_future_positions_df_dict[a_past_future_name].keys()))
            # epoch_specific_position_dfs
            # Always use the maximum number of subplots for this category (capped at 20)
            curr_num_subplots: int = max_subplots_per_category.get(a_past_future_name, min(20, len(found_pos_segment_ids)))
            
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
                found_pos_segment_ids = np.concatenate([found_pos_segment_ids, np.full(num_to_pad, -1, dtype=found_pos_segment_ids.dtype)])
            # curr_num_subplots: int = 40
            
            # an_epoch_specific_past_position_dfs = curr_matching_epochs_df_dict['past']
            # an_epoch_specific_past_epoch_ids = an_epoch_specific_past_position_dfs.index.to_numpy()
            ## OUTPUTS: an_epoch_specific_past_position_dfs, an_epoch_specific_past_epoch_ids
            

            # existing_ax = None
            # needed_init: bool = False
            # a_decoded_traj_plotter = self.trajectory_displaying_plotter.get(a_past_future_name, None)
            # if a_decoded_traj_plotter is None:
            #     ## create a new plotter
            #     a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers)
            #     self.trajectory_displaying_plotter[a_past_future_name] = a_decoded_traj_plotter
            #     needed_init = True
            # else:
            #     existing_ax = a_decoded_traj_plotter.axs
            #     # Clear existing axes before plotting to prevent drawing over previous plots
            #     if existing_ax is not None:
            #         # Handle different axis structures (list, array, or single axis)
            #         if isinstance(existing_ax, (list, tuple, np.ndarray)):
            #             for ax in existing_ax:
            #                 if ax is not None and hasattr(ax, 'clear'):
            #                     ax.clear()
            #         elif hasattr(existing_ax, 'clear'):
            #             existing_ax.clear()
            #         # Also clear the figure if it exists
            #         if hasattr(a_decoded_traj_plotter, 'fig') and a_decoded_traj_plotter.fig is not None:
            #             # Clear all axes in the figure
            #             for ax in a_decoded_traj_plotter.fig.get_axes():
            #                 ax.clear()

            # canvas: FigureCanvas = self.dock_canvas_widgets.get(a_past_future_name, None)
            # if canvas is not None:
            #     existing_ax = canvas.figure.get_axes() ## a list of 8 Axes objects

            ## Duplicated plotting.            
            # overlay_posterior, _, _ = self._get_posterior_data(an_epoch_idx)
            # a_decoded_traj_plotter.prev_heatmaps = [overlay_posterior] if overlay_posterior is not None else [] # seems stupid
            
            # existing_ax = a_decoded_traj_plotter.axs
            # if existing_ax is not None:
            #     if isinstance(existing_ax, (list, tuple, np.ndarray)):
            #         for ax in existing_ax:
            #             if ax is not None and hasattr(ax, 'clear'):
            #                 ax.clear()
            #     elif hasattr(existing_ax, 'clear'):
            #         existing_ax.clear()
            #     if hasattr(a_decoded_traj_plotter, 'fig') and a_decoded_traj_plotter.fig is not None:
            #         for ax in a_decoded_traj_plotter.fig.get_axes():
            #             ax.clear()
            
            # ## NOTE: `epoch_ids` used here and in the following function call actually refer to `found_pos_segment_ids`, not epochs, it's just how the `a_decoded_traj_plotter` class is named:
            # fig, axs, epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=self.curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=found_pos_segment_ids, curr_num_subplots=curr_num_subplots,
            #                                                                             active_page_index=0, fixed_columns=4, plot_actual_lap_lines=True, use_theoretical_tracks_instead=False, existing_ax=existing_ax,
            #                                                                             plot_mode='scatter', c='red', cmap='Reds', alpha=0.65, s=5, posteriors=overlay_posterior, posterior_alpha=0.25, posterior_cmap='jet', posterior_masking_value=None,
            #                                                                             )
                    
            # perform_update_title_subtitle(fig=fig, ax=None, title_string=f"{a_past_future_name} - an_epoch_idx: {an_epoch_idx}")
            # # self.active_epoch_idx = an_epoch_idx

            # # Embed the matplotlib figure in the dock widget
            # # Check if canvas widget needs to be created (either plotter is new or canvas doesn't exist)
            # canvas_needs_init = needed_init or (a_past_future_name not in self.dock_canvas_widgets)
            # if canvas_needs_init:
            #     dock = self.dock_widgets.get(a_past_future_name)
            #     if (dock is not None): # (canvas is None)
            #         # Remove existing widgets from dock
            #         # Dock uses a QGridLayout and maintains a widgets list
            #         layout = dock.layout
            #         if layout is not None:
            #             while layout.count():
            #                 child = layout.takeAt(0)
            #                 if child.widget():
            #                     child.widget().setParent(None)
            #         # Clear the widgets list maintained by Dock
            #         if hasattr(dock, 'widgets'):
            #             dock.widgets.clear()
            #         dock.currentRow = 0
                    
            #         # Create canvas and toolbar for the matplotlib figure
            #         canvas = FigureCanvas(fig)
            #         toolbar = NavigationToolbar(canvas, self.dock_window)
                    
            #         # Create a widget to hold canvas and toolbar
            #         plot_widget = QtWidgets.QWidget()
            #         plot_layout = QtWidgets.QVBoxLayout()
            #         plot_layout.setContentsMargins(0, 0, 0, 0)
            #         plot_layout.addWidget(toolbar)
            #         plot_layout.addWidget(canvas)
            #         plot_widget.setLayout(plot_layout)
                    
            #         # Add to dock
            #         dock.addWidget(plot_widget)
                    
            #         # Store reference to canvas widget
            #         self.dock_canvas_widgets[a_past_future_name] = canvas
                    
            #         # Close the figure window if it's open (since it's now embedded in the dock)
            #         plt.close(fig)                

            # else:
            #     ## just redraw - axes are already cleared above before plotting
            #     canvas = self.dock_canvas_widgets.get(a_past_future_name)
            #     if canvas is not None:
            #         # The axes were already cleared before plot_decoded_trajectories_2d was called
            #         # Just trigger a redraw
            #         canvas.draw_idle()
        ## END for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items()....




        # ==================================================================================================================================================================================================================================================================================== #
        # CENTRAL DOCK: Shows the decoded posterior t-bins for this epoch as tiny heatmaps along the top row, and the across-epoch posterior below.                                                                                                                                            #
        # ==================================================================================================================================================================================================================================================================================== #
        # epoch_high_prob_pos_masks = self.container.predictive_decoding.epoch_high_prob_pos_masks

        # Plot decoded posterior heatmap for 'decoded_posterior' dock
        category_name = 'decoded_posterior'
        ## get the data:
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx=an_epoch_idx)

        # Check if we need to initialize (create new widget) or update existing one
        needed_init: bool = category_name not in self.dock_canvas_widgets

        if needed_init:
            # Create MatplotlibTimeSynchronizedWidget
            from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget
            
            widget = MatplotlibTimeSynchronizedWidget(
                size=(8, 6), 
                dpi=72, 
                constrained_layout=True,
                disable_toolbar=False  # Keep toolbar for navigation
            )
            
            # Calculate extent from bin edges (more accurate than using centers)
            extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
            
            # Initial plot using method with viridis colormap and full opacity (matching original nested function behavior)
            self._update_posterior_plot(widget, posterior_2d=posterior_2d, time_bin_posteriors=time_bin_posteriors, num_time_bins_to_show=num_time_bins_to_show, an_epoch_idx=an_epoch_idx, extent=extent, posterior_alpha=1.0, posterior_cmap='viridis', show_overlay=False)
            
            # Embed the widget in the dock
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
                
                # Add widget to dock (widget already includes toolbar)
                dock.addWidget(widget)
                
                # Store reference to widget
                self.dock_canvas_widgets[category_name] = widget
        else:
            # Update existing widget
            widget = self.dock_canvas_widgets.get(category_name)
            if widget is not None:
                try:
                    # Recalculate extent for update
                    extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
                    
                    # Update plot with new data using method with viridis colormap and full opacity (matching original nested function behavior)
                    self._update_posterior_plot(widget, posterior_2d=posterior_2d, time_bin_posteriors=time_bin_posteriors, num_time_bins_to_show=num_time_bins_to_show, an_epoch_idx=an_epoch_idx, extent=extent, posterior_alpha=1.0, posterior_cmap='viridis', show_overlay=False)
                except Exception as e:
                    print(f"Error updating posterior plot for epoch {an_epoch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Widget not found - this shouldn't happen, but handle gracefully
                print(f"Warning: Widget for '{category_name}' not found in dock_canvas_widgets. Available keys: {list(self.dock_canvas_widgets.keys())}")


        ## OUTPUTS: curr_matching_past_future_positions_df_dict 
        self.active_epoch_idx = an_epoch_idx




# ==================================================================================================================================================================================================================================================================================== #
# PredictiveDecodingDisplayWidgetPg -- alternative implementation                                                                                                                                                                                                                      #
# ==================================================================================================================================================================================================================================================================================== #
@metadata_attributes(short_name=None, tags=['BROKEN', 'UNUSED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 14:41', related_items=[])
class PredictiveDecodingDisplayWidgetPg(QtWidgets.QWidget):
    """Alternative display widget using PyQtGraph for fast/interactive visualization.
    
    Displays the decoded posterior heatmap and (optionally) a row of tiny time-bin heatmaps.


    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidgetPg


    """
    def __init__(self, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg
        from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.main_plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.main_plot_widget)
        self.tiny_plots_container = QtWidgets.QWidget()
        self.tiny_layout = QtWidgets.QHBoxLayout(self.tiny_plots_container)
        self.layout.addWidget(self.tiny_plots_container)
        
        self.posterior_img = None
        self.tiny_img_items = []
        self.active_epoch_idx = None
        self.lut = None
        
    def plot_posterior(self, posterior_2d, xbin, ybin, time_bin_posteriors=None, num_time_bins_to_show=0, an_epoch_idx=None):
        """Plots the main posterior (2D) in the main area, optionally showing multiple tiny images for time bins."""
        import numpy as np
        import pyqtgraph as pg

        self.main_plot_widget.clear()
        self.posterior_img = pg.ImageItem()
        
        # Set image, orientation, and color map
        self.posterior_img.setImage(posterior_2d.T)
        tr = pg.QtGui.QTransform()
        # Proper extents: handle edge cases (empty or single bin)
        if posterior_2d.shape[0] > 0 and posterior_2d.shape[1] > 0:
            xscale = (xbin[-1] - xbin[0]) / float(posterior_2d.shape[0]) if posterior_2d.shape[0] > 1 else 1.0
            yscale = (ybin[-1] - ybin[0]) / float(posterior_2d.shape[1]) if posterior_2d.shape[1] > 1 else 1.0
            tr.translate(xbin[0], ybin[0])
            tr.scale(xscale, yscale)
        else:
            # Fallback for edge cases
            tr.translate(xbin[0] if len(xbin) > 0 else 0.0, ybin[0] if len(ybin) > 0 else 0.0)
        self.posterior_img.setTransform(tr)
        
        # Set color levels for main plot
        if posterior_2d.size > 0:
            vmin_main = np.nanmin(posterior_2d)
            vmax_main = np.nanmax(posterior_2d)
            self.posterior_img.setLevels((vmin_main, vmax_main))
        
        self.main_plot_widget.addItem(self.posterior_img)
        self.main_plot_widget.setTitle(f"Decoded Posterior Heatmap - Epoch {an_epoch_idx}")
        self.main_plot_widget.setLabel('bottom', 'X Position')
        self.main_plot_widget.setLabel('left', 'Y Position')
        self.main_plot_widget.setAspectLocked(False)
        self.main_plot_widget.autoRange()
        
        # Colormap: use 'viridis' via pyqtgraph (if available)
        try:
            self.lut = pg.colormap.get('viridis').getLookupTable()
            self.posterior_img.setLookupTable(self.lut)
        except Exception:
            self.lut = None  # Colormap not critical
            pass
        
        # Tiny heatmaps for time bins (optional)
        # Remove old widgets
        for i in reversed(range(self.tiny_layout.count())):
            widget_to_remove = self.tiny_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.tiny_img_items = []
        
        if (time_bin_posteriors is not None) and (num_time_bins_to_show > 0):
            # Shared color range
            all_time_bin_values = np.concatenate([tb.flatten() for tb in time_bin_posteriors])
            vmin_shared = np.nanmin(all_time_bin_values)
            vmax_shared = np.nanmax(all_time_bin_values)
            
            for t_bin_idx in range(num_time_bins_to_show):
                tiny_widget = pg.PlotWidget()
                tiny_img = pg.ImageItem()
                tiny_img.setImage(time_bin_posteriors[t_bin_idx].T)
                # Copy transform/scaling/extents
                tiny_img.setTransform(tr)
                tiny_img.setLevels((vmin_shared, vmax_shared))
                if self.lut is not None:
                    tiny_img.setLookupTable(self.lut)
                tiny_widget.addItem(tiny_img)
                tiny_widget.setMouseEnabled(False, False)
                tiny_widget.setMenuEnabled(False)
                tiny_widget.hideAxis('left')
                tiny_widget.hideAxis('bottom')
                tiny_widget.setFixedHeight(50)
                tiny_widget.setFixedWidth(50)
                # Add a minimal label below using QVBoxLayout
                tiny_layout_wrapper = QtWidgets.QVBoxLayout()
                tiny_layout_wrapper.setSpacing(0)
                container = QtWidgets.QWidget()
                container.setLayout(tiny_layout_wrapper)
                tiny_layout_wrapper.addWidget(tiny_widget)
                label = QtWidgets.QLabel(f"t={t_bin_idx}")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet("font-size: 8pt")
                tiny_layout_wrapper.addWidget(label)
                self.tiny_layout.addWidget(container)
                self.tiny_img_items.append(tiny_img)
        self.active_epoch_idx = an_epoch_idx


    def clear(self):
        """Clear all plots and reset the display widget."""
        self.main_plot_widget.clear()
        for i in reversed(range(self.tiny_layout.count())):
            widget_to_remove = self.tiny_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.tiny_img_items = []
        self.posterior_img = None
        self.active_epoch_idx = None









import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd

@function_attributes(short_name=None, tags=['non-working','BROKEN', 'UNUSED', 'plotting', 'temp', 'position_dfs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-13 19:50', related_items=[])
def plot_position_dfs_to_subplots(position_dfs: List[pd.DataFrame], fixed_columns: int = 2, figsize: tuple = None, plot_mode: str = 'line', epoch_labels: List[str] = None, **plot_kwargs):
    """
    Easily render a list of position_df dataframes to separate subplots/axes.
    
    Parameters:
    -----------
    position_dfs : List[pd.DataFrame]
        List of position dataframes, each with 'x' and 'y' columns (and optionally 't' for time)
    fixed_columns : int, default=2
        Number of columns in the subplot grid
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, uses matplotlib default
    plot_mode : str, default='line'
        Plotting mode: 'line', 'scatter', or 'time_gradient' (requires 't' column)
    epoch_labels : List[str], optional
        Labels for each epoch/subplot. If None, uses 'Epoch 0', 'Epoch 1', etc.
    **plot_kwargs
        Additional keyword arguments passed to plot/scatter (e.g., alpha, linewidth, c, etc.)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axs : numpy.ndarray
        Array of axes objects (may be 1D or 2D depending on grid shape)
    
    Usage:
    ------
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import plot_position_dfs_to_subplots

    # Simple usage:
    fig, axs = plot_position_dfs_to_subplots([epoch1_pos_df, epoch2_pos_df, epoch3_pos_df])
    
    # With custom labels and styling:
    fig, axs = plot_position_dfs_to_subplots(
        position_dfs=[epoch1_pos_df, epoch2_pos_df], 
        fixed_columns=2,
        epoch_labels=['Epoch A', 'Epoch B'],
        plot_mode='line',
        alpha=0.7,
        linewidth=2
    )
    
    # Time gradient mode (colors by time):
    fig, axs = plot_position_dfs_to_subplots(
        position_dfs=[epoch1_pos_df, epoch2_pos_df],
        plot_mode='time_gradient',
        cmap='viridis'
    )
    """
    num_epochs = len(position_dfs)
    if num_epochs == 0:
        raise ValueError("position_dfs list cannot be empty")
    
    # Calculate grid dimensions
    needed_rows = int(np.ceil(num_epochs / fixed_columns))
    linear_plotter_indices = np.arange(num_epochs)
    row_column_indices = np.unravel_index(linear_plotter_indices, (needed_rows, fixed_columns))
    
    # Create figure and axes
    if figsize is None:
        figsize = (6 * fixed_columns, 5 * needed_rows)
    fig, axs = plt.subplots(needed_rows, fixed_columns, figsize=figsize, sharex=True, sharey=True)
    
    # Handle 1D vs 2D axes array
    if needed_rows == 1:
        axs = axs.reshape(1, -1) if fixed_columns > 1 else axs.reshape(1, 1)
    elif fixed_columns == 1:
        axs = axs.reshape(-1, 1)
    
    # Default plot kwargs
    default_kwargs = {'alpha': 0.85} if plot_mode != 'time_gradient' else {'alpha': 0.85, 'linewidth': 2}
    plot_kwargs = {**default_kwargs, **plot_kwargs}
    
    # Plot each position dataframe
    for a_linear_index in linear_plotter_indices:
        curr_row = row_column_indices[0][a_linear_index]
        curr_col = row_column_indices[1][a_linear_index]
        # curr_pos_df = position_dfs[a_linear_index]
        curr_pos_df = position_dfs[a_linear_index][0]
        
        # Extract x, y coordinates
        x_vals = curr_pos_df['x'].to_numpy()
        y_vals = curr_pos_df['y'].to_numpy()
        
        # Plot based on mode
        if plot_mode == 'line':
            if 'c' not in plot_kwargs:
                plot_kwargs['c'] = 'k'
            axs[curr_row, curr_col].plot(x_vals, y_vals, **plot_kwargs)
        elif plot_mode == 'scatter':
            if 'c' not in plot_kwargs:
                plot_kwargs['c'] = 'k'
            axs[curr_row, curr_col].scatter(x_vals, y_vals, **plot_kwargs)
        elif plot_mode == 'time_gradient':
            if 't' not in curr_pos_df.columns:
                raise ValueError("plot_mode='time_gradient' requires 't' column in position_df")
            from matplotlib.collections import LineCollection
            t_vals = curr_pos_df['t'].to_numpy()
            cmap = plot_kwargs.pop('cmap', 'viridis')
            norm = plt.Normalize(t_vals.min(), t_vals.max())
            # Create line segments for gradient coloring
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(t_vals)
            lc.set_linewidth(plot_kwargs.get('linewidth', 2))
            lc.set_alpha(plot_kwargs.get('alpha', 0.85))
            axs[curr_row, curr_col].add_collection(lc)
        else:
            raise ValueError(f"plot_mode must be 'line', 'scatter', or 'time_gradient', got '{plot_mode}'")
        
        # Set label
        if epoch_labels is not None and a_linear_index < len(epoch_labels):
            label_text = epoch_labels[a_linear_index]
        else:
            label_text = f'Epoch {a_linear_index}'
        axs[curr_row, curr_col].set_title(label_text)
        axs[curr_row, curr_col].set_aspect('equal', adjustable='box')
        axs[curr_row, curr_col].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for a_linear_index in range(num_epochs, needed_rows * fixed_columns):
        curr_row = row_column_indices[0][a_linear_index] if a_linear_index < len(linear_plotter_indices) else a_linear_index // fixed_columns
        curr_col = row_column_indices[1][a_linear_index] if a_linear_index < len(linear_plotter_indices) else a_linear_index % fixed_columns
        if curr_row < needed_rows and curr_col < fixed_columns:
            axs[curr_row, curr_col].axis('off')
    
    plt.tight_layout()
    return fig, axs


















@function_attributes(short_name=None, tags=['pyqtgraph', 'trajectory'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 14:40', related_items=[])
def multi_trajectory_color_plotter(position_dfs: List[pd.DataFrame], rendering_mode: str = 'solid_colors', fixed_columns: int = 5, return_widget: bool = True, maze_extent: Optional[Tuple[float, float, float, float]] = None, overlay_mask: Optional[NDArray] = None):
    """ Takes a list of position dataframes representing separate trajectories in the same environment and plots them in a grid of tiny subplots.
    It assigns each of them a unique id and color.
    They can be rendered as lines of solid color, gradients from 0.25 alpha to 0.9 alpha of their assigned color, or something custom.

    Alternatively, we can plot them with a diverging color pallete with -1.0 meaning far past: start of the recording, and +1.0 meaning far-future: end of the recording.
    
    Args:
        position_dfs: List of position dataframes, each with 'x' and 'y' columns. Optional 't' column for time-based modes.
        rendering_mode: One of 'solid_colors', 'alpha_gradient', or 'time_diverging'. Default is 'solid_colors'.
        fixed_columns: Number of columns in the grid layout. Default is 5.
        return_widget: If True, returns (GraphicsLayoutWidget, list of PlotItems). If False, returns only list of PlotItems.
        maze_extent: Optional tuple of (xmin, xmax, ymin, ymax) to set fixed x/y limits for all subplots. If None, auto-ranges each plot.
        overlay_mask: Optional 2D numpy array to render as a low-alpha overlay on each subplot. Extents are set to maze_extent if provided, otherwise viewport edges.
    
    Returns:
        Tuple of (GraphicsLayoutWidget, list of PlotItems) if return_widget=True, else just list of PlotItems.

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import multi_trajectory_color_plotter

        plot_widget, plot_items = multi_trajectory_color_plotter(position_dfs=dfs_list)
        
        # With fixed maze extent:
        maze_extent = (a_decoder.xbin[0], a_decoder.xbin[-1], a_decoder.ybin[0], a_decoder.ybin[-1])
        plot_widget, plot_items = multi_trajectory_color_plotter(position_dfs=dfs_list, maze_extent=maze_extent)

    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    import numpy as np
    
    # Validate input
    if not position_dfs or len(position_dfs) == 0:
        raise ValueError("position_dfs must be a non-empty list")
    
    for i, df in enumerate(position_dfs):
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError(f"position_dfs[{i}] must have 'x' and 'y' columns")
    
    if rendering_mode not in ['solid_colors', 'alpha_gradient', 'time_diverging']:
        raise ValueError(f"rendering_mode must be one of 'solid_colors', 'alpha_gradient', or 'time_diverging', got '{rendering_mode}'")
    
    # Validate maze_extent if provided
    if maze_extent is not None:
        if not isinstance(maze_extent, (tuple, list)) or len(maze_extent) != 4:
            raise ValueError("maze_extent must be a tuple or list of 4 floats: (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = maze_extent
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("maze_extent must have xmin < xmax and ymin < ymax")
    
    # Validate overlay_mask if provided
    if overlay_mask is not None:
        if not isinstance(overlay_mask, np.ndarray):
            raise ValueError("overlay_mask must be a numpy array")
        if overlay_mask.ndim != 2:
            raise ValueError("overlay_mask must be a 2D array")
    
    num_epochs = len(position_dfs)
    needed_rows = int(np.ceil(num_epochs / fixed_columns))
    linear_plotter_indices = np.arange(num_epochs)
    row_column_indices = np.unravel_index(linear_plotter_indices, (needed_rows, fixed_columns))
    
    # Create GraphicsLayoutWidget for grid of plots
    graphics_widget = pg.GraphicsLayoutWidget()
    plot_items = []
    
    # Compute global time range for time_diverging mode if needed
    global_t_min = None
    global_t_max = None
    if rendering_mode == 'time_diverging':
        all_times = []
        for df in position_dfs:
            if 't' in df.columns:
                valid_t = df['t'].dropna()
                if len(valid_t) > 0:
                    all_times.extend(valid_t.tolist())
        if len(all_times) > 0:
            global_t_min = min(all_times)
            global_t_max = max(all_times)
        if global_t_min is None or global_t_max is None:
            # Fallback to solid_colors if no time data available
            rendering_mode = 'solid_colors'
    
    # White pen for axes outline
    white_pen = pg.mkPen('white', width=1)
    
    # Create and plot each trajectory in its own tiny subplot
    for a_linear_index in linear_plotter_indices:
        curr_row = row_column_indices[0][a_linear_index]
        curr_col = row_column_indices[1][a_linear_index]
        pos_df = position_dfs[a_linear_index]
        
        # Create a tiny plot in the grid
        plot_item = graphics_widget.addPlot(row=curr_row, col=curr_col)
        plot_items.append(plot_item)
        
        # Hide labels but show axes with white outline
        plot_item.hideAxis('bottom')
        plot_item.hideAxis('left')
        plot_item.hideAxis('top')
        plot_item.hideAxis('right')
        # Show axes but hide tick labels - set white pen for outline
        plot_item.showAxis('bottom')
        plot_item.showAxis('left')
        plot_item.showAxis('top')
        plot_item.showAxis('right')
        # Set white pen for all axes to create the box outline
        plot_item.getAxis('bottom').setPen(white_pen)
        plot_item.getAxis('left').setPen(white_pen)
        plot_item.getAxis('top').setPen(white_pen)
        plot_item.getAxis('right').setPen(white_pen)
        # Hide tick labels
        plot_item.getAxis('bottom').setTicks([])
        plot_item.getAxis('left').setTicks([])
        plot_item.getAxis('top').setTicks([])
        plot_item.getAxis('right').setTicks([])
        plot_item.setMenuEnabled(False)
        plot_item.setMouseEnabled(False, False)
        
        # Extract coordinates
        x_vals = pos_df['x'].to_numpy()
        y_vals = pos_df['y'].to_numpy()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        
        if len(x_vals) < 2:
            continue  # Skip trajectories with insufficient points
        
        if rendering_mode == 'solid_colors':
            # Each trajectory gets a unique color
            traj_color = pg.intColor(a_linear_index, hues=len(position_dfs))
            traj_color.setAlphaF(0.4)  # Moderately low alpha
            brush = pg.mkBrush(traj_color)
            plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brush)
        
        elif rendering_mode == 'alpha_gradient':
            # Alpha gradient from 0.25 to 0.9 along the path
            base_color = pg.intColor(a_linear_index, hues=len(position_dfs))
            n_points = len(x_vals)
            brushes = []
            for i in range(n_points):
                # Interpolate alpha from 0.25 to 0.9
                alpha = 0.25 + (0.9 - 0.25) * (i / max(1, n_points - 1))
                point_color = pg.mkColor(base_color)
                point_color.setAlphaF(alpha)
                brushes.append(pg.mkBrush(point_color))
            plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brushes)
        
        elif rendering_mode == 'time_diverging':
            # Diverging color palette based on normalized time (-1.0 to +1.0)
            if 't' in pos_df.columns:
                t_vals = pos_df['t'].to_numpy()[valid_mask]
                if len(t_vals) > 0 and global_t_min is not None and global_t_max is not None:
                    # Normalize time to [-1.0, 1.0]
                    t_range = global_t_max - global_t_min
                    if t_range > 0:
                        normalized_t = 2.0 * ((t_vals - global_t_min) / t_range) - 1.0
                    else:
                        normalized_t = np.zeros_like(t_vals)
                    
                    # Use diverging colormap (RdBu-like: blue for -1.0, red for +1.0)
                    brushes = []
                    for i in range(len(x_vals)):
                        seg_t = normalized_t[i]
                        # Map from [-1.0, 1.0] to color
                        # Blue (0, 0, 255) for -1.0, Red (255, 0, 0) for +1.0
                        if seg_t < 0:
                            # Blue to white gradient
                            intensity = abs(seg_t)
                            r = int(255 * intensity)
                            g = int(255 * intensity)
                            b = 255
                        else:
                            # White to red gradient
                            intensity = seg_t
                            r = 255
                            g = int(255 * (1 - intensity))
                            b = int(255 * (1 - intensity))
                        
                        seg_color = pg.mkColor(r, g, b, 200)
                        brushes.append(pg.mkBrush(seg_color))
                    plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brushes)
                else:
                    # Fallback to solid color if time data is invalid
                    traj_color = pg.intColor(a_linear_index, hues=len(position_dfs))
                    traj_color.setAlphaF(0.4)  # Moderately low alpha
                    brush = pg.mkBrush(traj_color)
                    plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brush)
            else:
                # Fallback to solid color if no time column
                traj_color = pg.intColor(a_linear_index, hues=len(position_dfs))
                traj_color.setAlphaF(0.4)  # Moderately low alpha
                brush = pg.mkBrush(traj_color)
                plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brush)
        
        # Set x/y limits based on maze_extent if provided, otherwise auto-range
        if maze_extent is not None:
            xmin, xmax, ymin, ymax = maze_extent
            plot_item.setXRange(xmin, xmax, padding=0)
            plot_item.setYRange(ymin, ymax, padding=0)
        else:
            # Auto-range each plot to fit its trajectory
            plot_item.autoRange()
            # Get the viewport bounds after auto-ranging
            view_range = plot_item.viewRange()
            xmin, xmax = view_range[0]
            ymin, ymax = view_range[1]
        
        # Add overlay_mask if provided
        if overlay_mask is not None:
            # Determine extents: use maze_extent if provided, otherwise use viewport
            if maze_extent is not None:
                overlay_xmin, overlay_xmax, overlay_ymin, overlay_ymax = maze_extent
            else:
                overlay_xmin, overlay_xmax = xmin, xmax
                overlay_ymin, overlay_ymax = ymin, ymax
            
            # Set image axis order to row-major (like BinByBinDecodingDebugger pattern)
            pg.setConfigOptions(imageAxisOrder='row-major')
            
            # Create ImageItem for the overlay mask with low alpha
            # Use setImage with rect parameter instead of transform (like BinByBinDecodingDebugger pattern)
            # Note: overlay_mask is expected to be [rows, cols] = [y_size, x_size] in spatial coordinates
            overlay_img = pg.ImageItem(image=overlay_mask, levels=(0, 1), opacity=0.2)
            # rect format: [x, y, width, height] where (x, y) is bottom-left corner
            # Account for pixel centering: adjust by half pixel to align edges properly
            pixel_width = (overlay_xmax - overlay_xmin) / overlay_mask.shape[1] if overlay_mask.shape[1] > 0 else 1.0
            pixel_height = (overlay_ymax - overlay_ymin) / overlay_mask.shape[0] if overlay_mask.shape[0] > 0 else 1.0
            overlay_width = overlay_xmax - overlay_xmin
            overlay_height = overlay_ymax - overlay_ymin
            # Adjust rect to account for pixel centering (subtract half pixel from position, add half pixel to size)
            image_bounds_extent = [overlay_xmin - pixel_width/2, overlay_ymin - pixel_height/2, overlay_width + pixel_width, overlay_height + pixel_height]
            overlay_img.setImage(overlay_mask, rect=image_bounds_extent, autoLevels=False)
            plot_item.addItem(overlay_img)
    
    # Return based on return_widget parameter
    if return_widget:
        return graphics_widget, plot_items
    else:
        return plot_items