# ==================================================================================================================== #
# 2024-05-27 - WCorr Shuffle Stuff                                                                                     #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Sequence
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
from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
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



# #TODO 2026-01-22 13:44: - [ ] Comparing Directed to Scatttered

Decoded Occupancy of only good events / compared to observed occupancy -- similar to the checks I did on the 1D track

Do the directed PBEs need to follow the 

During mid sleep, PBEs would be biased toward the more interesting 2D maze instead of htte environment measured.


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

        _obj = cls(time_window_centers=time_window_centers, pos_df=pos_df,
                   xbin=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.xbin), ybin=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.ybin),
                   xbin_centers=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.xbin_centers), ybin_centers=deepcopy(directional_decoders_decode_result.pseudo2D_decoder.ybin_centers),
                   new_positions=new_positions, interpolator=interpolator, p_x_given_n=p_x_given_n,
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


    @function_attributes(short_name=None, tags=['UNUSED', 'downsampling'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21 08:29', related_items=[])
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
    
    
    
    Updates the following columns in `self.relevant_positions_df`:

        - 'segment_idx': Segment index (initialized to -1 if not in epoch)
        - 'Vp': Persistence velocity (direction of movement)
        - 'segment_Vp_deg': Segment velocity direction in degrees
        - 'segment_dir_angle_binned': Binned direction angle for segment
        - 'segment_Vp_scatteredness': Velocity scatteredness measure for segment


        
        - 'segment_Vp_deg': Mean direction angle in degrees for each segment
        - 'segment_Vp_scatteredness': Scatteredness measure (R) for each segment (1 = aligned, 0 = scattered)
        - 'segment_dir_angle_binned': Binned direction angle
        
        - 'segment_idx': Segment index (created as all zeros if missing)
        - 'centroid_pos_traj_matching_angle_idx': Index of matching centroid segment (initialized to -1 if no match)
        
        
    
    
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
    relevant_positions_df: pd.DataFrame = serialized_field(repr=False) ## !IMPORTANT: `relevant_positions_df`: the df of all potentially relevant positions, with a 'matching_found_relevant_pos_epoch' column corresponding to the the found *epochs* (not some aren't in an epoch and have a value of -1 for this column)
    matching_pos_epochs_df: pd.DataFrame = serialized_field(repr=False) ## !IMPORTANT: `matching_pos_epochs_df`: the df of found *epochs* corresponding to the position sequences in `relevant_positions_df`


    # Basically IRRELEVANT FIELDS ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    pos_matches_epoch_mask: NDArray = serialized_field(repr=False)
    
    is_relevant_past_times: NDArray = serialized_field(repr=False)
    is_relevant_future_times: NDArray = serialized_field(repr=False)
    n_total_possible_past_times: int = serialized_field()
    n_total_possible_future_times: int = serialized_field()
    n_relevant_past_times: int = serialized_field()
    n_relevant_future_times: int = serialized_field()


    # Computed fields ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    matching_past_positions_df: pd.DataFrame = serialized_field(default=None, is_computable=True, repr=False)
    matching_future_positions_df: pd.DataFrame = serialized_field(default=None, is_computable=True, repr=False)
    pos_segment_to_centroid_seq_segment_idx_map: Optional[Dict] = non_serialized_field(default=Factory(dict), is_computable=True, repr=False, metadata={'field_added':"2026.01.14_0"})
    
    should_defer_extended_computations: bool = serialized_attribute_field(default=True, metadata={'field_added':"2026.01.15_0"})

    epoch_t_idx_col_name: str = non_serialized_field(default='epoch_t_idx', metadata={'field_added':"2026.01.23_0"})
    merged_found_pos_epoch_id_key_name: str = non_serialized_field(default='matching_found_relevant_merged_pos_epoch', metadata={'field_added':"2026.01.23_0"})
    
    max_allowed_trajectory_gap_seconds: float = serialized_attribute_field(default=2.5, metadata={'field_added':"2026.01.23_0"})
    merged_segment_epochs: Optional[pd.DataFrame] = serialized_field(default=None, repr=False, metadata={'field_added':"2026.01.23_0", 'desc':'computed by self._recompute_all_pos_dfs using self.max_allowed_trajectory_gap_seconds'})
    

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


    @property
    def matching_pos_dfs_list(self) -> List[pd.DataFrame]:
        """The matching_future_position_df_list property. #TODO 2026-01-15 06:18: - [ ] WRONG """
        epoch_only_relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=pd.concat((self.matching_past_positions_df, self.matching_future_positions_df), axis='index'), relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=True, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs
        unique_values, partitioned_dfs_list = epoch_only_relevant_positions_df.pho.partition_df(partitionColumn=self.epoch_id_key_name)
        return partitioned_dfs_list


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
        # row_col_indices = np.argwhere(self.epoch_high_prob_mask)
        # row_col_row_ids = row_col_indices + 1
        # an_epoch_mask_included_binned_x_y_columns_idx_df = pd.DataFrame(row_col_row_ids, columns=["binned_x", "binned_y"])
        # return an_epoch_mask_included_binned_x_y_columns_idx_df.sort_values(by=["binned_x", "binned_y"]).reset_index(drop=True)

        # # Compute from stored epoch_high_prob_mask (exact equivalent of original computation)
        # row_col_indices = np.argwhere(_test_epoch_result.epoch_high_prob_mask)
        # row_col_row_ids = row_col_indices + 1
        

        # # Remove Temporal+Spatial overlap to try and fix epoch merging failures, but it didn't work __________________________________________________________________________________________________________________________________________________________________________________________ #
        ## make masks mutually exclusive first        
        # mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask = np.full_like(self.epoch_t_bins_high_prob_pos_mask, fill_value=False) # deepcopy(self.epoch_t_bins_high_prob_pos_mask)
        # n_t_bins: int = np.shape(self.epoch_t_bins_high_prob_pos_mask)[-1]
        # # mutually_accumulating_or_mask = np.full_like(self.epoch_high_prob_mask, fill_value=False) ## boolean mask
        # for i in np.arange(n_t_bins):
        #     if i == 0:
        #         mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask[:, :, i] = self.epoch_t_bins_high_prob_pos_mask[:, :, i] ## original mask is unchanged for this iteration
        #     else:
        #         prev_t_bins_mutually_accumulating_or_mask = np.any(mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask[:, :, :i], axis=-1) ## up to, but not including i, meaning we'll effectively subtract off any previous masks to ensure that the final map doesn't overlap
        #         # mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask[:, :, i] = np.logical_xor(mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask[:, :, i], prev_t_bins_mutually_accumulating_or_mask) ## effectively finding only the True values in the current mask that haven't been in any previous masks
        #         mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask[:, :, i] = np.logical_and(self.epoch_t_bins_high_prob_pos_mask[:, :, i], np.logical_not(prev_t_bins_mutually_accumulating_or_mask))
        # OUTPUTS: mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask
        
        row_col_row_ids_dt = np.argwhere(self.epoch_t_bins_high_prob_pos_mask) ## old way could produce multiple (duplicated) values if a position fell into multiple t_idx time bins
        # row_col_row_ids_dt = np.argwhere(mutually_non_overlapping_epoch_t_bins_high_prob_pos_mask) ## old way could produce multiple (duplicated) values if a position fell into multiple t_idx time bins
        row_col_row_ids_dt[:, :2] = row_col_row_ids_dt[:, :2] + 1
        epoch_mask_included_binned_x_y_columns_idx_df_dt: pd.DataFrame = pd.DataFrame(row_col_row_ids_dt, columns=["binned_x", "binned_y", self.epoch_t_idx_col_name])
        return epoch_mask_included_binned_x_y_columns_idx_df_dt.sort_values(by=["binned_x", "binned_y", self.epoch_t_idx_col_name]).reset_index(drop=True)
        


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
        
        pre_merge_column_merge_dict = None # {'epoch_t_idx': 'unique_concat', 'is_future_present_past': 'require_same'} ## do not specify

        ## NOTE: some have negative duration, they overlap, and all sorts of other confusing things...
        # self.matching_pos_epochs_df, curr_matching_positions_df_dict = MatchingPastFuturePositionsResult._custom_build_sequential_position_epochs(matching_past_positions_df=self.relevant_positions_df) # curr_matching_positions_df_dict: types.epoch_index
        self.matching_pos_epochs_df, curr_matching_positions_df_dict = self.compute_matching_pos_epochs_df(self.relevant_positions_df, disable_segmentation=True, column_merge_dict=pre_merge_column_merge_dict)

        ## Propagate per-position-trajectory-epoch segmentation columns back to relevant_positions_df
        segmented_traj_columns = ['segment_idx', 'Vp', 'segment_Vp_deg', 'segment_dir_angle_binned', 'segment_Vp_scatteredness']
        for epoch_idx, epoch_pos_df in curr_matching_positions_df_dict.items():
            for col in segmented_traj_columns:
                if col in epoch_pos_df.columns:
                    # Update values in relevant_positions_df by matching on index
                    self.relevant_positions_df.loc[epoch_pos_df.index, col] = epoch_pos_df[col]

        ## re-index:
        self.relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_pos_epochs_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=False, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs

        #TODO 2026-01-23 12:32: - [ ] Recompute the complete path
        ## I think the compute order is correct, but can't be sure
        
        ## OUTPUTS: matching_relevant_positions_df
        merged_segment_epochs, relevant_merged_positions_df, matching_pos_epochs_df = self.compute_compilete_paths(max_allowed_trajectory_gap_seconds=self.max_allowed_trajectory_gap_seconds, merged_found_pos_epoch_id_key_name=self.merged_found_pos_epoch_id_key_name)
        ## the above method is pure, so update the self properties --- I think this is "a-ok" because it just adds some badass indicies... but maybe something tragic is lost and NO ONE WILL KNOW:
        self.merged_segment_epochs = merged_segment_epochs

        # Assert.same_length(self.matching_pos_epochs_df, matching_pos_epochs_df)
        # self.matching_pos_epochs_df = matching_pos_epochs_df ## no this one isn't right at least :[ It will be missing the main epoch info and will have fewer of em because they are merged. Need to add this info to the existing epochs_df or something

        ## this one is fine on the other hand, maybe check to make sure the DANG SIZE DOESN'T CHANGE
        Assert.same_length(self.relevant_positions_df, relevant_merged_positions_df)
        self.relevant_positions_df = relevant_merged_positions_df
        
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
    

    @function_attributes(short_name=None, tags=['new', 'traj', 'angle', 'direction'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 19:36', related_items=[])
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


    @classmethod
    def compute_matching_pos_epochs_df(cls, measured_positions_df: pd.DataFrame, merging_adjacent_max_separation_sec: float = 0.075, minimum_epoch_duration: float = 0.050, disable_segmentation=True, column_merge_dict: Optional[Dict]=None, **kwargs) -> Tuple[pd.DataFrame, Dict[types.epoch_index, pd.DataFrame]]:
        """
        Compute matching position epochs DataFrame from position matches and time filters.
        
        Args:
            measured_positions_df: DataFrame with position data. Should already be filtered to only include past/future positions (not present positions).
            merging_adjacent_max_separation_sec: Maximum separation in seconds for merging adjacent epochs
            minimum_epoch_duration: Minimum duration for detected epochs
            
        Returns:
            Tuple of (position trajectory epochs DataFrame, dict of per-position-trajectory-epoch DataFrames with segmentation columns)

            a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls.compute_matching_pos_epochs_df(measured_positions_df, disable_segmentation=disable_segmentation, **kwargs)
            
        """
        if column_merge_dict is None:
            # column_merge_dict = {'epoch_t_idx': 'require_same', 'is_future_present_past': 'require_same'} ## idk, normally we just probagated is_future_present_past
            column_merge_dict = {'epoch_t_idx': 'require_same', 'is_future_present_past': 'first'} ## idk, normally we just probagated is_future_present_past

        ## find adjacent epochs from the position time bins (periods where the animal is in the positions)
        # measured_positions_df_copy = measured_positions_df.copy()
        # assert 'is_included' in measured_positions_df_copy

        # a_matching_pos_epochs_df: pd.DataFrame = measured_positions_df_copy.neuropy.detect_epoch_satisfying_condition(is_condition_satisfied = (measured_positions_df_copy['is_included'].to_numpy()), merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration)        
        a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls._custom_build_sequential_position_epochs(matching_positions_df=measured_positions_df, disable_segmentation=disable_segmentation, **kwargs) ## dataframe is already filtered to past/future positions before being passed

        ## Copied from `.neuropy.detect_epoch_satisfying_condition(...)``
        if (merging_adjacent_max_separation_sec is not None) and (len(a_matching_pos_epochs_df) > 0):
            if "epoch_t_idx" in a_matching_pos_epochs_df.columns:
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().sort_values(["epoch_t_idx", "start"])
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.groupby("epoch_t_idx", group_keys=False).apply(lambda df: df.epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec, **column_merge_dict))
            else:
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec, **column_merge_dict)
                
        if (minimum_epoch_duration is not None) and (len(a_matching_pos_epochs_df) > 0): 
            a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_epochs_longer_than(minimum_duration=minimum_epoch_duration)

        if (merging_adjacent_max_separation_sec is not None) and (len(a_matching_pos_epochs_df) > 0):
            if "epoch_t_idx" in a_matching_pos_epochs_df.columns:
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().sort_values(["epoch_t_idx", "start"])
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.groupby("epoch_t_idx", group_keys=False).apply(lambda df: df.epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec, **column_merge_dict))
            else:
                a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.get_valid_df().epochs.merge_adjacent_epochs_within(max_merge_duration=merging_adjacent_max_separation_sec, **column_merge_dict)


        if (len(a_matching_pos_epochs_df) > 0):
            a_matching_pos_epochs_df = a_matching_pos_epochs_df.epochs.rebuild_labels_column()
        
        # Restore is_future_present_past column after merge operations

        ## #TODO 2026-01-14 18:09: - [ ] Add the relevant epoch idx to the `measured_positions_df`
        
        return a_matching_pos_epochs_df, curr_matching_positions_df_dict


    def filter_positions_to_epoch_mask_included_bins(self, a_pos_df: pd.DataFrame) -> pd.DataFrame:
        """ filter to the epoch's bins """
        ## allowed positions are much less than the found ones:
        return a_pos_df.merge(self.epoch_mask_included_binned_x_y_columns_idx_df, on=["binned_x", "binned_y"], how="inner")


    @function_attributes(short_name=None, tags=['FIXED', 'WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-15 09:50', related_items=[])
    @classmethod
    def _custom_build_sequential_position_epochs(cls, matching_positions_df: pd.DataFrame, col_name: str = 'past_future_matching_pos_epoch_id', EPSILON_GAP_SIZE_SEC: float = 1e-9, disable_segmentation: bool = True) -> Tuple[pd.DataFrame, Dict[types.epoch_index, pd.DataFrame]]:
        """ builds the positions_epochs_df from the positions_df that match a single PBE epoch by merging consecutive position time bins samples into epochs.

        Identifies consecutive sequences of time bins (gaps <= dt_max) and returns epochs spanning each sequence.
        """
        if len(matching_positions_df) < 1:
            print(f'warn: empty df!')
            # Return empty DataFrame with expected columns so downstream (NeuroPy adding_epochs_identity_column, compute_compilete_paths) do not KeyError on 'start'/'stop' or 'epoch_t_idx'.
            return pd.DataFrame(columns=['start', 'stop', 'duration', 'label', 'epoch_t_idx', 'is_future_present_past']), {}

        df = matching_positions_df.copy()
        assert 't' in df

        # Compute bin size from minimum consecutive gap
        t_sorted = np.sort(df['t'].values)
        pos_t_bin_sample_size_sec: float = np.nanmin(np.abs(np.diff(t_sorted)))
        dt_max: float = pos_t_bin_sample_size_sec * 2.5

        # Identify sequences FIRST by detecting gaps > dt_max
        df = df.sort_values('t').reset_index(drop=True)
        df['sequence_id'] = (df['t'].diff() > dt_max).cumsum() ## WTF is this doing?

        # Build epochs by aggregating each sequence - use first/last 't' values
        # Include is_future_present_past if it exists in the dataframe
        agg_dict = {'start': ('t', 'first'), 'stop': ('t', 'last'), 't_count': ('t', 'count'), 'start_pos_idx': ('t', 'idxmin'), 'stop_pos_idx': ('t', 'idxmax')} # , 'epoch_t_idx': ('epoch_t_idx', 'first') #  'epoch_t_idx' should be the same within each epoch (hopefully, unless they've been merged?)
        if 'is_future_present_past' in df.columns:
            agg_dict['is_future_present_past'] = ('is_future_present_past', 'first')
        if 'epoch_t_idx' in df.columns:
            agg_dict['epoch_t_idx'] = ('epoch_t_idx', 'first')

        new_pos_epochs: pd.DataFrame = df.groupby('sequence_id').agg(**agg_dict).reset_index()
        # Extend stop by bin_size (last 't' is start of last bin, not end)
        new_pos_epochs['stop'] = new_pos_epochs['stop'] + pos_t_bin_sample_size_sec - EPSILON_GAP_SIZE_SEC
        new_pos_epochs['duration'] = new_pos_epochs['stop'] - new_pos_epochs['start']
        new_pos_epochs['label'] = new_pos_epochs['sequence_id'].astype(int)

        # Assign sequence_id back to positions for partitioning
        a_curr_matching_positions_df = df.copy()
        a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=new_pos_epochs, epoch_id_key_name=col_name, override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True, drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)

        # Partition first, then segment each epoch separately so each trajectory gets its own segment_Vp_deg
        curr_matching_positions_df_dict: Dict[types.epoch_index, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
        
        ## Segment trajectories per-position-trajectory-epoch (so each trajectory gets its own representative direction angle)
        for epoch_idx, epoch_pos_df in curr_matching_positions_df_dict.items():
            curr_matching_positions_df_dict[epoch_idx] = epoch_pos_df.position.adding_segmented_trajectories_columns(disable_segmentation=disable_segmentation)

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

            # matching_relevant_positions_df: pd.DataFrame = an_epoch_past_future_result.filter_positions_to_epoch_mask_included_bins_dt(a_pos_df=an_epoch_past_future_result.relevant_positions_df.copy())

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



    @function_attributes(short_name=None, tags=['IMPORTANT', 'merge', 'segments', 'pure'],
                          uses=['cls.compute_matching_pos_epochs_df', 'cls._recompute_relevant_pos_epoch_position_df_index_column'], used_by=['self._recompute_all_pos_dfs'],
                          creation_date='2026-01-23 12:04', related_items=[])
    def compute_compilete_paths(self, max_allowed_trajectory_gap_seconds: float = 2.0, merged_found_pos_epoch_id_key_name='matching_found_relevant_merged_pos_epoch'):
        """ 
        ## check for position sequences with non-indexed gaps shorter than the max gap allow time
        
        Pure: does not alter self

        
        """
        # pre_merge_column_merge_dict = {'epoch_t_idx': 'require_same', 'is_future_present_past': 'require_same'} #TODO 2026-01-23 14:00: - [ ] Worked for most epochs, but I guess fails when there is overlap
        pre_merge_column_merge_dict = {'epoch_t_idx': 'unique_concat', 'is_future_present_past': 'first'}  

        # post_merge_column_merge_dict = {'epoch_t_idx': 'unique_concat', 'is_future_present_past': 'require_same'}  ## different than before, now we want to find all unique items in 'epoch_t_idx' as this gives which time bins it satisfies after merging
        post_merge_column_merge_dict = {'epoch_t_idx': 'unique_concat', 'is_future_present_past': 'first'}  ## different than before, now we want to find all unique items in 'epoch_t_idx' as this gives which time bins it satisfies after merging



        # column_to_split: str = 'label'
        column_to_split: str = 'epoch_t_idx'
        
        relevant_positions_df: pd.DataFrame = self.relevant_positions_df
        # matching_pos_epochs_df: pd.DataFrame = deepcopy(_test_epoch_result.matching_pos_epochs_df)

        matching_relevant_positions_df: pd.DataFrame = self.filter_positions_to_epoch_mask_included_bins(a_pos_df=relevant_positions_df.copy())

        ## INPUTS: matching_relevant_positions_df
        
        ## gotta update `matching_pos_epochs_df`
        matching_pos_epochs_df, _ = self.compute_matching_pos_epochs_df(matching_relevant_positions_df, disable_segmentation=True, column_merge_dict=pre_merge_column_merge_dict) # curr_matching_positions_df_dict: types.epoch_index
        matching_pos_epochs_df = matching_pos_epochs_df.sort_values(['start']).reset_index(drop=True)
        
        ## INPUTS: max_allowed_trajectory_gap_seconds
        
        # inter_segment_epoch_df: pd.DataFrame = matching_pos_epochs_df.epochs.get_in_between()
        # inter_segment_epoch_df
        # can_be_merged_segments = inter_segment_epoch_df[inter_segment_epoch_df['duration'] < max_allowed_trajectory_gap_seconds]
        # can_be_merged_segments

        merged_segment_epochs: pd.DataFrame = deepcopy(matching_pos_epochs_df).epochs.merge_adjacent_epochs_within(max_merge_duration=max_allowed_trajectory_gap_seconds, **post_merge_column_merge_dict)
        split_epoch_labels: List[List[int]] = merged_segment_epochs[column_to_split].astype(str).map(lambda x: [int(v) for v in x.split('+')]).to_list()
        
        # split_epoch_labels
        merged_segment_epochs['num_epoch_t_bins'] = np.array([len(v) for v in split_epoch_labels]) ## the number of merged segments in each thingy
        merged_segment_epochs['is_reversely_replayed'] = [(v[0] > v[-1]) for v in split_epoch_labels] # this column doesn't mean anything for num_epoch_t_bins == 1

        ## Now build the complete new paths for the NEW merged epochs to get the full position path
        merged_segment_epochs['pre_merged_epoch_label'] = deepcopy(merged_segment_epochs['label']) ## want the inverse of this for `merged_segment_epochs`
        merged_segment_epochs['label'] = merged_segment_epochs.index.astype(int) ## reset the label so it's a valid int-like type for the new path instead of something merged, like "1+2+3"

        # # #TODO 2026-01-23 13:16: - [ ] gotta update `matching_pos_epochs_df` with new merged epoch indicies (to decide whether they are included or not) __________________________________________________________________________________________________________________________________________________________________ #
        # # Assert.same_length(split_epoch_labels, merged_segment_epochs)
        # matching_pos_epoch_labels_to_split_epoch_labels = {a_label: [] for a_label in merged_segment_epochs['label'].to_numpy().astype(int)} ## make a list for each epoch
        
        # for idx, a_row in enumerate(merged_segment_epochs.itertuples(index=True)):
        #     an_original_labels = split_epoch_labels[idx]
        #     for an_original_label in an_original_labels:
        #         matching_pos_epoch_labels_to_split_epoch_labels[an_original_label].append(int(a_row.label)) ## the fresh label

        # matching_pos_epoch_labels_to_split_epoch_labels = {k:list(set(v)) for k, v in matching_pos_epoch_labels_to_split_epoch_labels.items()} # de-duplicate each list 

        # assert np.all([len(v) < 2 for v in matching_pos_epoch_labels_to_split_epoch_labels.values()]) #BUG 2026-01-23 13:11: - [ ] Something is seriously wrong here :[ The lengths are actually like [85, 86, 12, 14, 0, 0, 0, ..., 0, 0] and idk why

        # matching_pos_epochs_df[merged_found_pos_epoch_id_key_name] = -1 ## None, initialize the column:
        # matching_pos_epochs_df[merged_found_pos_epoch_id_key_name] = matching_pos_epochs_df['label'].map(lambda x: matching_pos_epoch_labels_to_split_epoch_labels.get(int(x), -1)[0] ) # `[0]` unpack the merge list

        # matching_pos_epochs_df, curr_matching_positions_df_dict = MatchingPastFuturePositionsResult.compute_matching_pos_epochs_df(matching_relevant_positions_df, disable_segmentation=True) # curr_matching_positions_df_dict: types.epoch_index
        # matching_pos_epochs_df

        ## add the final detected a_matching_pos_epochs_df indicies to the decoded positions as the column ['matching_found_relevant_pos_epoch']:
        relevant_merged_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=merged_segment_epochs, relevant_positions_df=deepcopy(relevant_positions_df),
                                                                                                                                drop_non_epoch_events=False, epoch_id_key_name=merged_found_pos_epoch_id_key_name) ## don't drop yet so we have all the events for the object creation



        return merged_segment_epochs, relevant_merged_positions_df, matching_pos_epochs_df # matching_pos_epochs_df: has [merged_found_pos_epoch_id_key_name] column added to back-identify the merged epoch label from the original epochs


    def get_filtered_by_min_seq_length(self, minimum_included_matching_sequence_length: Optional[int]=None):
        """ 
        """
        relevant_positions_df: pd.DataFrame = deepcopy(self.relevant_positions_df)
        matching_pos_epochs_df: pd.DataFrame = deepcopy(self.matching_pos_epochs_df)
        merged_segment_epochs: pd.DataFrame = deepcopy(self.merged_segment_epochs)
        
        ## INPUTS: merged_segment_epochs, relevant_positions_df, matching_pos_epochs_df

        good_merged_segment_epochs: pd.DataFrame = merged_segment_epochs[merged_segment_epochs['num_epoch_t_bins'] >= minimum_included_matching_sequence_length]
        good_merged_segment_epochs

        # relevant_positions_df[relevant_positions_df['matching_found_relevant_merged_pos_epoch'] > -1]

        good_only_relevant_positions_df: pd.DataFrame = relevant_positions_df[np.logical_and(relevant_positions_df['matching_found_relevant_merged_pos_epoch'].isin(good_merged_segment_epochs['label']), (relevant_positions_df['matching_found_relevant_pos_epoch'] > -1))]
        good_only_relevant_positions_df


        good_only_included_epoch_labels: NDArray = np.unique(good_only_relevant_positions_df['matching_found_relevant_pos_epoch'].to_numpy())
        # good_only_included_epoch_labels
        ## INPUTS: matching_pos_epochs_df
        good_only_matching_pos_epochs_df = deepcopy(matching_pos_epochs_df)[matching_pos_epochs_df['label'].isin(good_only_included_epoch_labels)]
        good_only_matching_pos_epochs_df

        ## OUTPUTS: good_merged_segment_epochs, good_only_relevant_positions_df, good_only_matching_pos_epochs_df
        return (good_merged_segment_epochs, good_only_relevant_positions_df, good_only_matching_pos_epochs_df)
    


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
    @function_attributes(short_name=None, tags=['OLD', 'predictive_decoding', 'layers', 'heatmap', 'overlay'], input_requires=[], output_provides=[], uses=['TimeSynchronizedGenericPlotterLayer'], used_by=[], creation_date='2025-12-09 19:03', related_items=[])
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

    # @function_attributes(short_name=None, tags=['OLD', 'UNUSED', 'prospective', 'future', 'past', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-11 07:43', related_items=[])
    # @classmethod
    # def calculate_position_epoch_overlap(cls, gaussian_volume: np.ndarray, pos_time_bin_centers: np.ndarray, decoded_epochs_all_filter_epochs: pd.DataFrame, decoded_epochs_result: Any, curr_decoder_context_idx: int = 0, debug_max_time_steps_to_process: Optional[int] = 200, debug_overide_start_idx: Optional[int]=None, debug_print: bool = True) -> np.ndarray:
    #     """
    #     Calculates the overlap between the current position probability (Gaussian volume)
    #     and all preceding decoded epochs.
        
    #     Optimized to use vectorized matrix multiplication instead of nested loops.

    #     Args:
    #         gaussian_volume: 3D array (X, Y, Time)
    #         pos_time_bin_centers: 1D array of time centers corresponding to gaussian_volume dim 2
    #         decoded_epochs_all_filter_epochs: DataFrame containing 'start' and 'stop' columns
    #         decoded_epochs_result: Object containing .p_x_given_n_list (list of 4D arrays)
    #         curr_decoder_context_idx: Index for the context dimension (0 or 1)
    #         debug_max_time_steps_to_process: Max number of time steps to process (for debugging)
    #         debug_print: Whether to print progress/shape info

    #     Returns:
    #         np.ndarray: A 2D array (Time, Epochs) containing scalar overlap scores.
    #                     Future epochs (relative to time t) are represented as NaN.
    #     """

    #     # 1. Setup Time Selection
    #     # -----------------------
    #     # Preserve original logic: take last 2000 bins, then apply debug limit
    #     total_time_bins = len(pos_time_bin_centers)
    #     if debug_overide_start_idx is not None:
    #         start_idx = max(0, debug_overide_start_idx)
    #     else:
    #         start_idx = 0

    #     if debug_max_time_steps_to_process is not None:
    #         # Limit the end index relative to the start_idx
    #         end_idx = min(total_time_bins, (start_idx + debug_max_time_steps_to_process))
    #     else:
    #         end_idx = total_time_bins

    #     # Slice inputs to relevant time window
    #     active_pos_time_bin_centers = pos_time_bin_centers[start_idx:end_idx]
    #     num_pos_time_bin_centers: int = len(active_pos_time_bin_centers)

    #     if debug_print:
    #         print(f'num_pos_time_bin_centers: {num_pos_time_bin_centers}')

    #     # 2. Data Preparation (Flattening & Cleaning)
    #     # -------------------------------------------
    #     # Slice the Gaussian volume to match the active time window
    #     active_gaussian_slice = gaussian_volume[:, :, start_idx:end_idx]
    #     n_x, n_y, n_t = active_gaussian_slice.shape
        
    #     # Reshape Gaussian Volume: (X, Y, T) -> (X*Y, T)
    #     # Use nan_to_num so NaNs become 0.0, allowing efficient dot products (acting like nansum)
    #     flat_gaussian = np.nan_to_num(active_gaussian_slice.reshape(n_x * n_y, n_t))

    #     # Prepare Epoch Data
    #     # We assume decoded_epochs_result.p_x_given_n_list corresponds to the rows in the DataFrame
    #     all_epoch_stops = decoded_epochs_all_filter_epochs['stop'].to_numpy()
        
    #     # Flatten spatial dims for all epochs: List of (X*Y, Epoch_Time_Bins) arrays
    #     # We extract the specific context (curr_decoder_context_idx) immediately
    #     flat_epoch_arrays = [
    #         np.nan_to_num(v[:, :, curr_decoder_context_idx, :].reshape(n_x * n_y, -1))
    #         for v in decoded_epochs_result.p_x_given_n_list
    #     ]

    #     # 3. Vectorized Calculation
    #     # -------------------------
    #     # Initialize result matrix: (N_Time_Steps, N_Total_Epochs)
    #     # Initialize with NaN to represent "future" epochs (or padding)
    #     padded_pos_overlap_matrix = np.full(
    #         (num_pos_time_bin_centers, len(flat_epoch_arrays)), 
    #         np.nan
    #     )

    #     # Iterate over EPOCHS (Outer loop is now Epochs)
    #     # This allows us to apply one Epoch to ALL valid time bins simultaneously via matrix mult
    #     for epoch_idx, (epoch_arr, stop_time) in enumerate(zip(flat_epoch_arrays, all_epoch_stops)):
            
    #         # Mask: Find all time bins where this epoch is strictly in the past (or current)
    #         valid_time_mask = active_pos_time_bin_centers >= stop_time
            
    #         # Optimization: Skip if this epoch hasn't happened yet for any active time bin
    #         if not np.any(valid_time_mask):
    #             continue

    #         # Select the Gaussian columns for valid times: (Space, Valid_Times)
    #         # Transpose to (Valid_Times, Space) for matrix multiplication
    #         relevant_gaussian_T = flat_gaussian[:, valid_time_mask].T
            
    #         # CORE CALCULATION: Matrix Multiplication (The "Dot Product")
    #         # (Valid_Times, Space) @ (Space, Epoch_Bins) -> (Valid_Times, Epoch_Bins)
    #         # This effectively performs the sum(A * B) over spatial dimensions
    #         spatial_sums = np.matmul(relevant_gaussian_T, epoch_arr)
            
    #         # Calculate median over the epoch's internal time bins (Axis 1)
    #         scalar_scores = np.median(spatial_sums, axis=1)
            
    #         # Assign to the main result matrix
    #         padded_pos_overlap_matrix[valid_time_mask, epoch_idx] = scalar_scores

    #     if debug_print:
    #         print(f"Processed {num_pos_time_bin_centers} time steps. "
    #             f"Final shape: {padded_pos_overlap_matrix.shape}")

    #     return active_pos_time_bin_centers, padded_pos_overlap_matrix


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
        # relevant_positions_df_after_merge = relevant_positions_df.copy()  # Save state after merge for visualization

        ## only after initial filter do we filter by this version:
        pos_matches_epoch_mask = np.where([epoch_high_prob_mask[(a_pos.binned_x-1), (a_pos.binned_y-1)] for a_pos in relevant_positions_df.itertuples()])[0]
        relevant_positions_df: pd.DataFrame = relevant_positions_df.iloc[pos_matches_epoch_mask].copy()
        
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

        a_matching_pos_epochs_df, _ = MatchingPastFuturePositionsResult.compute_matching_pos_epochs_df(measured_positions_df=filtered_positions_df, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, disable_segmentation=should_defer_extended_computations)
        
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
    

    @function_attributes(short_name=None, tags=['SINGLE_MAIN'], input_requires=[], output_provides=[], uses=['cls.detect_matching_past_future_positions'], used_by=['compute_specific_future_and_past_analysis'], creation_date='2026-01-14 19:49', related_items=[])
    @classmethod
    def _process_single_epoch_future_past_analysis(cls, i: int, curr_epoch_p_x_given_n: NDArray, curr_epoch_time_bin_centers: NDArray, measured_positions_df: pd.DataFrame, top_v_percent: float,
                epoch_t_bin_high_prob_masks_dict: Optional[Dict], epoch_high_prob_masks_dict: Optional[Dict],
                a_slice_multiplier: float, n_epoch_time_bins: int, merging_adjacent_max_separation_sec: float, minimum_epoch_duration: float, progress_print: bool, n_total_epochs: int, decoded_epoch_result=None, **kwargs) -> Tuple[int, Any, Any, Any, Any, Any, Any]:
        """Process a single epoch for future/past analysis. Returns results in a tuple for parallel processing.

        After done, calls  `PredictiveDecoding.detect_matching_past_future_positions(...)` to find the potential past/future positions given the new masks

        """
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
            
            labeled, n_objects, masks = PosteriorMaskPostProcessing._process_epoch_time_bins_masks(a_mask_t=an_epoch_t_bins_custom_high_prob_mask, max_gap=8, n_interp=1) #TODO 2026-01-21 08:12: - [ ] Review what this does
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
            is_high_prob_mask = (curr_epoch_p_x_given_n >= thresholds) #TODO 2026-01-21 08:13: - [ ] The promenence peaks are much better
        
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

        # Store original value for warning purposes
        use_parallel_requested = use_parallel

        # Decide whether to run in parallel or serial
        n_tasks: int = n_total_epochs
        n_cpus: int = os.cpu_count() or 1

        # Check if parallel was requested but can't run due to insufficient CPUs
        if use_parallel_requested and n_cpus <= 1:
            import warnings
            warnings.warn(f"Parallel execution was requested (use_parallel=True) but cannot run: only {n_cpus} CPU(s) available. Running sequentially instead.", UserWarning)
            print(f"WARNING: Parallel execution requested but only {n_cpus} CPU(s) available. Running sequentially.")

        use_parallel: bool = use_parallel and (n_tasks > 1) and (n_cpus > 1)
        
        # Set default max_workers if None to prevent ThreadPoolExecutor from using too many threads
        if max_workers is None:
            # Cap at a reasonable number to prevent resource exhaustion on supercomputers
            max_workers = min(8, n_cpus)  # Use at most 8 workers or available CPUs, whichever is smaller
            if progress_print:
                print(f'WARNING: max_workers was None, defaulting to {max_workers} to prevent resource exhaustion')
        
        # Process epochs in parallel or sequentially
        if use_parallel and n_total_epochs > 1:
            if progress_print:
                print(f'Processing {n_total_epochs} epochs in parallel (max_workers={max_workers})...')
            
            results_list = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, curr_epoch_p_x_given_n, curr_epoch_time_bin_centers, curr_epoch_tbin_indicies, n_epoch_time_bins, a_decoded_epoch_result in epoch_data_list:
                    future = executor.submit(cls._process_single_epoch_future_past_analysis, i=i, curr_epoch_p_x_given_n=curr_epoch_p_x_given_n, curr_epoch_time_bin_centers=curr_epoch_time_bin_centers, measured_positions_df=measured_positions_df, top_v_percent=top_v_percent, epoch_t_bin_high_prob_masks_dict=epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict=epoch_high_prob_masks_dict, a_slice_multiplier=a_slice_multiplier, n_epoch_time_bins=n_epoch_time_bins, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, progress_print=progress_print, n_total_epochs=n_total_epochs, decoded_epoch_result=a_decoded_epoch_result, should_defer_extended_computations=should_defer_extended_computations, disable_segmentation=disable_segmentation) # curr_epoch_tbin_indicies=curr_epoch_tbin_indicies, gaussian_volume=gaussian_volume, 
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
                result = cls._process_single_epoch_future_past_analysis(i=i, curr_epoch_p_x_given_n=curr_epoch_p_x_given_n, curr_epoch_time_bin_centers=curr_epoch_time_bin_centers, measured_positions_df=measured_positions_df, top_v_percent=top_v_percent, epoch_t_bin_high_prob_masks_dict=epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict=epoch_high_prob_masks_dict, a_slice_multiplier=a_slice_multiplier, n_epoch_time_bins=n_epoch_time_bins, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration, progress_print=progress_print, n_total_epochs=n_total_epochs, decoded_epoch_result=a_decoded_epoch_result, should_defer_extended_computations=should_defer_extended_computations, disable_segmentation=disable_segmentation) # curr_epoch_tbin_indicies=curr_epoch_tbin_indicies, gaussian_volume=gaussian_volume, 
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
                ## This is aweful :[ The indicies don't match and so it's failing here:
                try:
                    ## try to add directly
                    active_epochs_df[f"{computed_df_col_name_prefix}{k}"] = v # ValueError: Length of values (103) does not match length of index (93)

                except ValueError as e:
                    print(f'wARN: len(v): {len(v)} > len(active_epochs_df): {len(active_epochs_df)}, trying to adapt the columns...')
                    if len(v) > len(active_epochs_df):
                        ## try to get the relevant entries:
                        # active_idx_to_epoch_idx_map = {i:int(a_row.label) for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=True))}
                        epoch_idx_to_active_idx_map = {int(a_row.original_epoch_idx):i for i, a_row in enumerate(ensure_dataframe(decoded_local_epochs_result.filter_epochs).itertuples(index=True))}
                        original_epoch_idxs = active_epochs_df['original_epoch_idx'].to_numpy() ## get the original index
                        original_epoch_idx_to_linear_idx_map = active_epochs_df['original_epoch_idx'].map(epoch_idx_to_active_idx_map).to_dict()
                        assert len(original_epoch_idxs) == len(original_epoch_idx_to_linear_idx_map), f"cannot adapt indicies, have to give up. len(original_epoch_idxs): {len(original_epoch_idxs)}, len(v): {len(original_epoch_idx_to_linear_idx_map)}"
                        active_linear_idxs = np.array(list(original_epoch_idx_to_linear_idx_map.values()))
                        assert len(active_epochs_df) == len(active_linear_idxs), f"cannot adapt indicies, have to give up. len(active_epochs_df): {len(active_epochs_df)}, len(active_linear_idxs): {len(active_linear_idxs)}"
                        target_key: str = f"{computed_df_col_name_prefix}{k}"
                        active_epochs_df[target_key] = v[active_linear_idxs] # ValueError: Length of values (103) does not match length of index (93)
                        print(f'\tsuccessfully adapted the column: "{k}": added column name: "{target_key}".')
                    else:
                        print(f'wARN: failed to add the df columns due to error: {e}. Skipping')
                        print(f'incompatibile lengths :[')
            ## END for k, v in past_future_in...

            try:
                ## add more columns after the others are added:
                active_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_past'] = active_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_past'] / active_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_past']
                active_epochs_df[f'{computed_df_col_name_prefix}ratio_avail_future'] = active_epochs_df[f'{computed_df_col_name_prefix}n_total_relevant_future'] / active_epochs_df[f'{computed_df_col_name_prefix}n_total_possible_future']

            except (KeyError, ValueError, AttributeError) as e:
                print(f'failed to add two additional columns post-hoc eith error: {e}')
                pass
            
            # except Exception as e:
                # print(f'failed to add two additional columns post-hoc eith error: {e}')
                # raise e        


        
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

        position_like_kwargs = dict(position_like_score_cutoff=0.42, num_min_position_like_t_bins=3)
        
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
                

            # if selected_tbin_size not in masked_container.epochs_decoded_result_cache_dict:
            #     masked_container.epochs_decoded_result_cache_dict[selected_tbin_size] = {}
                

            scoring_results_df_list = []
            for a_decoding_time_bin_size, a_decoded_results_dict in (a_decoded_results_dict_dict or {}).items():
                for an_decoder_name, a_decoded_local_epochs_result in (a_decoded_results_dict or {}).items():
                    a_decoder = masked_container.pf1D_Decoder_dict.get(an_decoder_name, None)
                    if a_decoder is None:
                        a_decoder = list(masked_container.pf1D_Decoder_dict.values())[0]

                    filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_decoded_local_epochs_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3)
                    if a_decoding_time_bin_size not in masked_container.epochs_decoded_result_cache_dict:
                        masked_container.epochs_decoded_result_cache_dict[a_decoding_time_bin_size] = {}                    

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


        #TODO 2026-01-21 08:08: - [ ] Needs review to see if it's filtering right
        @function_attributes(short_name=None, tags=['NEEDS_REVIEW'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21 08:08', related_items=[])
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
                        matching_original_idx = next((pos_idx for pos_idx, (_, orig_row) in enumerate(original_active_epochs_df.iterrows())
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
                    
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks and len(pred_dec.epoch_t_bins_high_prob_pos_masks) > 0:
                        original_len = len(pred_dec.epoch_t_bins_high_prob_pos_masks)
                        pred_dec.epoch_t_bins_high_prob_pos_masks = [pred_dec.epoch_t_bins_high_prob_pos_masks[i] for i in filtered_to_original_idx if i < original_len]
                        print(f'Filtered epoch_t_bins_high_prob_pos_masks: {original_len} -> {len(pred_dec.epoch_t_bins_high_prob_pos_masks)} entries (removed {original_len - len(pred_dec.epoch_t_bins_high_prob_pos_masks)})')
                    
                    # Recompute epoch_high_prob_pos_masks from filtered epoch_t_bins_high_prob_pos_masks
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks and len(pred_dec.epoch_t_bins_high_prob_pos_masks) > 0:
                        pred_dec.epoch_high_prob_pos_masks = [np.any(is_high_prob_mask, axis=-1) for is_high_prob_mask in pred_dec.epoch_t_bins_high_prob_pos_masks]
                        print(f'Recomputed epoch_high_prob_pos_masks from filtered epoch_t_bins_high_prob_pos_masks: {len(pred_dec.epoch_high_prob_pos_masks)} entries')
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
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks:
                        original_len = len(pred_dec.epoch_t_bins_high_prob_pos_masks)
                        pred_dec.epoch_t_bins_high_prob_pos_masks = pred_dec.epoch_t_bins_high_prob_pos_masks[:filtered_len]
                        print(f'WARN: Truncated epoch_t_bins_high_prob_pos_masks to {filtered_len} entries (original: {original_len}) - proper mapping requires original active_epochs_df')
                    
                    # Recompute epoch_high_prob_pos_masks from truncated epoch_t_bins_high_prob_pos_masks
                    if hasattr(pred_dec, 'epoch_t_bins_high_prob_pos_masks') and pred_dec.epoch_t_bins_high_prob_pos_masks and len(pred_dec.epoch_t_bins_high_prob_pos_masks) > 0:
                        pred_dec.epoch_high_prob_pos_masks = [np.any(is_high_prob_mask, axis=-1) for is_high_prob_mask in pred_dec.epoch_t_bins_high_prob_pos_masks]
                        print(f'WARN: Recomputed epoch_high_prob_pos_masks from truncated epoch_t_bins_high_prob_pos_masks: {len(pred_dec.epoch_high_prob_pos_masks)} entries - proper mapping requires original active_epochs_df')
                        
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
                        # Use filtered_idx directly since epoch_idx_to_actual_times is keyed by filtered indices
                        if filtered_idx in epoch_idx_to_actual_times:
                            epoch_start, epoch_stop = epoch_idx_to_actual_times[filtered_idx]
                            is_past = (df['stop'] < epoch_start)
                            is_future = (df['start'] > epoch_stop)
                            df['is_future_present_past'] = 'present'
                            df.loc[is_past, 'is_future_present_past'] = 'past'
                            df.loc[is_future, 'is_future_present_past'] = 'future'
                            pred_dec.matching_pos_epochs_dfs_list[filtered_idx] = df
                    print(f'Recomputed is_future_present_past column for {len(pred_dec.matching_pos_epochs_dfs_list)} filtered epochs in matching_pos_epochs_dfs_list')
                
                if pred_dec.matching_pos_dfs_list and len(pred_dec.matching_pos_dfs_list) > 0:
                    for filtered_idx, df in enumerate(pred_dec.matching_pos_dfs_list):
                        # Use filtered_idx directly since epoch_idx_to_actual_times is keyed by filtered indices
                        if filtered_idx in epoch_idx_to_actual_times:
                            epoch_start, epoch_stop = epoch_idx_to_actual_times[filtered_idx]
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
        masked_filter_epochs: Optional[pd.DataFrame] = None
        
        if hasattr(self, 'locality_measures') and (self.locality_measures is not None):
            # epoch_names: List[str] = self.locality_measures.paradigm_epochs_df.label.to_list() # ['roam', 'sprinkle']
            epoch_names: List[str] = self.locality_measures.epoch_names # ['roam', 'sprinkle']
        else:
            epoch_names: List[str] = self.decoding_locality.epoch_names
        
        assert len(epoch_names) > 0
        
        # assert use_full_recompute_method, f'the non full recompute mode  did not seem to do a dmang thing, I hope it is never called!'
        should_filter_directional_decoders_decode_result = True ## UPDATES: directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict
        # should_compute_future_and_past_analysis = True

        # ==================================================================================================================================================================================================================================================================================== #
        # Modifies `directional_decoders_decode_result` from the pipeline itself?                                                                                                                                                                                                              #
        # ==================================================================================================================================================================================================================================================================================== #
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded'])
        available_tbins: List[float] = list(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.keys())
        assert len(available_tbins) > 0
        most_recent_tbin: float = available_tbins[-1]
        selected_tbin: float = a_t_bin_size if (a_t_bin_size in available_tbins) else most_recent_tbin

        if should_filter_directional_decoders_decode_result:
            print(f'directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict')
            a_decoder = list(directional_decoders_decode_result.pf1D_Decoder_dict.values())[0]
            for extant_decoded_time_bin_size, a_result_decoded in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.items():
                a_result_decoded: SingleEpochDecodedResult = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[extant_decoded_time_bin_size]
                a_result_decoded: DecodedFilterEpochsResult = DecodedFilterEpochsResult.init_from_single_epoch_result(single_epoch_result=a_result_decoded, decoding_time_bin_size=extant_decoded_time_bin_size) ## convert to a `DecodedFilterEpochsResult` for masking
                filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_result_decoded, xbin=a_decoder.xbin, ybin=a_decoder.ybin,
                                                                                                                                            **position_like_kwargs, normalization_across_epochs_epoch_names=epoch_names)
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
        

        masked_container: PredictiveDecodingComputationsContainer = PredictiveDecodingComputationsContainer(predictive_decoding=masked_predictive_decoding, is_global=True)
        if (masked_filter_epochs is not None):
            masked_container.active_epochs_df = masked_filter_epochs


        if (self.pf1D_Decoder_dict is None) or (len(self.pf1D_Decoder_dict) == 0):
            ## initialize it self.pf1D_Decoder_dict if it isn't setup:
            assert directional_decoders_decode_result is not None
            self.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            print(f'assigning pf1D_Decoder_dict: {list(self.pf1D_Decoder_dict.keys())}')
            
        masked_container.pf1D_Decoder_dict = deepcopy(self.pf1D_Decoder_dict)
        
        _decode_kwargs = {k:kwargs.get(k, None) for k in ['merging_adjacent_max_separation_sec', 'minimum_epoch_duration'] if (kwargs.get(k, None) is not None)}
        
        for an_active_t_bin_size in list(set([a_t_bin_size])): # only care about `a_t_bin_size`

            if an_active_t_bin_size not in masked_container.epochs_decoded_result_cache_dict:
                masked_container.epochs_decoded_result_cache_dict[an_active_t_bin_size] = {} # deepcopy(self.epochs_decoded_result_cache_dict[selected_tbin]) ## copy the cached result from the existing object

            for a_decoder_name in epoch_names:

                an_extant_result = self.epochs_decoded_result_cache_dict.get(an_active_t_bin_size, {}).get(a_decoder_name, None) ## checking against self, not masked_container
                if an_extant_result is not None:
                    ## make sure that it's filtered:
                    filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=an_extant_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, **position_like_kwargs)
                    masked_container.epochs_decoded_result_cache_dict[an_active_t_bin_size][a_decoder_name] = filtered_decoded_local_epochs_result
                    
                else:
                    ## full compute at the finer time bin size for the epochs in question:
                    an_extant_result, a_decoder, active_epochs_df = masked_container.update_active_epochs_and_decode_posteriors_if_needed(curr_active_pipeline, an_epoch_name=a_decoder_name, decoding_time_bin_size=an_active_t_bin_size, 
                                                                                    **_decode_kwargs,
                                                                                    override_included_analysis_epochs=None, ## because it will use self.active_epochs if it exists.
                                                                                    epoch_id_key_name='non_local_PBE_non_moving_epoch', force_recompute_epoch_df_columns=False, allow_update_instance_properties=False,
                                                                                )
                    an_extant_result.filter_epochs = ensure_dataframe(an_extant_result.filter_epochs)
                    if 'original_epoch_idx' not in an_extant_result.filter_epochs:
                        an_extant_result.filter_epochs['original_epoch_idx'] = an_extant_result.filter_epochs.index.to_numpy().astype(int)
                    an_extant_result.filter_epochs.reset_index(drop=True, inplace=True) ## reset the index so they match up
                    # an_extant_result.filter_epochs = ensure_Epoch(an_extant_result.filter_epochs)
                    
                    filtered_decoded_local_epochs_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=an_extant_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, **position_like_kwargs)
                    filtered_decoded_local_epochs_result.filter_epochs = ensure_dataframe(filtered_decoded_local_epochs_result.filter_epochs)
                    if 'original_epoch_idx' not in filtered_decoded_local_epochs_result.filter_epochs:
                        filtered_decoded_local_epochs_result.filter_epochs['original_epoch_idx'] = filtered_decoded_local_epochs_result.filter_epochs.index.to_numpy().astype(int)
                    filtered_decoded_local_epochs_result.filter_epochs.reset_index(drop=True, inplace=True) ## reset the index so they match up
                    # filtered_decoded_local_epochs_result.filter_epochs = ensure_Epoch(an_extant_result.filter_epochs)
                    masked_container.epochs_decoded_result_cache_dict[an_active_t_bin_size][a_decoder_name] = filtered_decoded_local_epochs_result
                    
                if masked_filter_epochs is None:
                    masked_filter_epochs = ensure_dataframe(filtered_decoded_local_epochs_result.filter_epochs)
                    
                masked_container.active_epochs_df = ensure_dataframe(filtered_decoded_local_epochs_result.filter_epochs)
                if masked_container.scoring_results_df is None:
                    masked_container.scoring_results_df = scoring_results
                    
                masked_container.epochs_decoded_result_cache_dict[an_active_t_bin_size][a_decoder_name] = filtered_decoded_local_epochs_result
                
                if a_decoder_name not in masked_container.debug_computed_dict:
                    masked_container.debug_computed_dict[a_decoder_name] = {} ## initialize to empty
                                        

            ## END for a_decoder_nam....
        ## for an_active_t_bi...
                        
        print(f'done with all decoding.')
        
        ## where the main results are filtered
        masked_container = _subfn_update_internal_results(masked_container=masked_container, selected_tbin_size=selected_tbin)
        
        ## Filter active_epochs_df and matching_pos_epochs_dfs_list to match the filtered decoded results ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # NOTE: We use actual time span from time_bin_containers (edges) rather than filter_epochs start/stop because when time bins are dropped during masking, the effective start/stop times change
        # Build epoch_idx -> (actual_start, actual_stop) mapping once, reuse for filtering and recomputation
        original_active_epochs_df: pd.DataFrame = ensure_dataframe(self.active_epochs_df) if (hasattr(self, 'active_epochs_df') and (self.active_epochs_df is not None)) else None
        masked_container, filter_epochs, epoch_idx_to_actual_times = _subfn_filter_masked_container_epochs(masked_container=masked_container, original_active_epochs_df=original_active_epochs_df)


        # ## REQUIRED OUTPUTS: masked_container
        # assert masked_container is not None

        # # Get this specific result ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if should_compute_peak_prom_analysis:
            from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PosteriorPeaksPeakProminence2dResult

            print(f'computeing peak_prom_analysis because `should_compute_peak_prom_analysis == True`...')
            # raise NotImplementedError(f'Peak prominence analysis is intentionally disabled in build_masked_container (enable explicitly if needed).')

            print(f'\tfor epoch_names: {epoch_names}')
            for a_decoder_name in epoch_names:
                if a_decoder_name not in masked_container.debug_computed_dict:
                    masked_container.debug_computed_dict[a_decoder_name] = {}

                _comp_result_key: str = 'peak_prom_analysis'
                if _comp_result_key not in masked_container.debug_computed_dict[a_decoder_name]:
                    masked_container.debug_computed_dict[a_decoder_name][_comp_result_key] = {}

                a_decoded_local_epochs_result = masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].get(a_decoder_name, None)
                a_decoder: BayesianPlacemapPositionDecoder = masked_container.pf1D_Decoder_dict.get(a_decoder_name, None)
                # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=a_decoded_local_epochs_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3) ## this seems to be done previously in `_subfn_update_internal_results`, but that's okay

                step: float = PeakPromenence.compute_optimal_step_size(a_masked_result.p_x_given_n_list, resolution_factor=500.0)
                print(f'\tstep: {step}')
                masked_container.debug_computed_dict[a_decoder_name][_comp_result_key]['step'] = step
                
                decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=a_masked_result.p_x_given_n_list, 
                    xbin_centers=masked_container.predictive_decoding.xbin_centers, 
                    ybin_centers=masked_container.predictive_decoding.ybin_centers,
                    step=step, minimum_included_peak_height=None, # 1m 42s - 7m 1s
                    # step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
                    peak_height_multiplier_probe_levels=(0.25, 0.5, 0.9),
                    should_use_faster_compute_single_slab_implementation=False,
                    min_considered_promenence=1e-11,
                )
                masked_container.debug_computed_dict[a_decoder_name][_comp_result_key]['decoded_epoch_t_bins_promenence_result_obj'] = decoded_epoch_t_bins_promenence_result_obj
                print(f'\tcomputation done.')


        if should_compute_future_and_past_analysis:
            # if not use_full_recompute_method:
            #     raise ValueError(f'compute_future_and_past_analysis requires use_full_recompute_method=True to ensure predictive_decoding/locality_measures are consistent with the masked results.')


            if a_t_bin_size not in masked_container.epochs_decoded_result_cache_dict:
                masked_container.epochs_decoded_result_cache_dict[a_t_bin_size] = {} ## initialize to empty
                # if a_t_bin_size not in self.epochs_decoded_result_cache_dict:
                #     masked_container.epochs_decoded_result_cache_dict[a_t_bin_size] = {} ## initialize to empty
                # else:
                #     masked_container.epochs_decoded_result_cache_dict[a_t_bin_size] = deepcopy(self.epochs_decoded_result_cache_dict[a_t_bin_size]) ## copy of self's result


            for a_decoder_name in epoch_names:
                if a_decoder_name not in masked_container.debug_computed_dict:
                    masked_container.debug_computed_dict[a_decoder_name] = {}

                assert a_t_bin_size in masked_container.epochs_decoded_result_cache_dict, f'we created it above!!'
                a_masked_result = masked_container.epochs_decoded_result_cache_dict[a_t_bin_size].get(a_decoder_name, None) ## already masked in previously in `_subfn_update_internal_results`
                a_decoder: BayesianPlacemapPositionDecoder = masked_container.pf1D_Decoder_dict.get(a_decoder_name, None)
                assert a_masked_result is not None, f"a_masked_result is None for masked_container.epochs_decoded_result_cache_dict[a_t_bin_size: {a_t_bin_size}][a_decoder_name: '{a_decoder_name}']"
                # if a_masked_result is None:
                # a_masked_result, a_decoder, active_epochs_df = masked_container.update_active_epochs_and_decode_posteriors_if_needed(curr_active_pipeline, an_epoch_name=a_decoder_name, decoding_time_bin_size=a_t_bin_size, 
                #                                             **_decode_kwargs,
                #                                             override_included_analysis_epochs=None, ## because it will use self.active_epochs if it exists.
                #                                             epoch_id_key_name='non_local_PBE_non_moving_epoch', force_recompute_epoch_df_columns=False,
                #                                         )
                    
                override_included_analysis_epochs: pd.DataFrame = ensure_dataframe(a_masked_result.filter_epochs)
                print(f'\tlen(override_included_analysis_epochs): {len(override_included_analysis_epochs)}, \n\toverride_included_analysis_epochs: {override_included_analysis_epochs}')
                
                # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                _out = masked_container.compute_future_and_past_analysis(an_epoch_name=a_decoder_name, decoding_time_bin_size=a_t_bin_size, enable_updating_instance_states=True, 
                                                                            override_included_analysis_epochs=override_included_analysis_epochs, ## is this right?
                                                                             **kwargs,
                                                                         )
                # epoch_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list = _out
                epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _out
                # masked_container.debug_computed_dict[an_epoch_name] = {'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict}
                masked_container.debug_computed_dict[a_decoder_name].update({'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict})


            ## END for an_epoch_name in epoch_names...
            

        # ## Filter active_epochs_df and matching_pos_epochs_dfs_list to match the filtered decoded results ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # # NOTE: We use actual time span from time_bin_containers (edges) rather than filter_epochs start/stop because when time bins are dropped during masking, the effective start/stop times change
        # # Build epoch_idx -> (actual_start, actual_stop) mapping once, reuse for filtering and recomputation
        # original_active_epochs_df: pd.DataFrame = ensure_dataframe(self.active_epochs_df) if (hasattr(self, 'active_epochs_df') and self.active_epochs_df is not None) else None
        # masked_container, filter_epochs, epoch_idx_to_actual_times = _subfn_filter_masked_container_epochs(masked_container=masked_container, original_active_epochs_df=original_active_epochs_df)


        return masked_container




    @function_attributes(short_name=None, tags=['temp', 'from-notebook', 'prominence2d', 'locality'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-13 10:17', related_items=[])
    def final_refine_single_epoch_result_masks(self, curr_active_pipeline, fine_decoding_t_bin_size: float = 0.025, a_decoder_name: types.DecoderName = 'roam', **kwargs) -> DecodedFilterEpochsResult:
        """
        Seems to just do the whole set of computations again after the filtering/masking
        
        History: 2026-01-21 "_filter_single_epoch_result" -> "final_refine_single_epoch_result_masks"
        Uses:
            self.epochs_decoded_result_cache_dict
            
            fine_decoding_t_bin_size: float = 0.025
            an_epoch_name = 'roam'
            masked_container = container.build_masked_container(curr_active_pipeline=curr_active_pipeline, a_t_bin_size=fine_decoding_t_bin_size,
                should_filter_directional_decoders_decode_result=True, should_compute_future_and_past_analysis=False, should_compute_peak_prom_analysis=False,
            ) ## 4m 18s now
            active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container._filter_single_epoch_result(curr_active_pipeline=curr_active_pipeline, fine_decoding_t_bin_size=fine_decoding_t_bin_size, an_epoch_name=an_epoch_name)

        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder
        from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence, PosteriorPeaksPeakProminence2dResult

        use_parallel: bool = kwargs.pop('use_parallel', True)
        max_workers: int = kwargs.pop('max_workers', 4)

        # Store original values for warning purposes
        use_parallel_requested = use_parallel

        n_cpus: int = os.cpu_count() or 1
        if n_cpus < 2:
            if use_parallel_requested:
                import warnings
                warnings.warn(f"Parallel execution was requested (use_parallel=True) but cannot run: only {n_cpus} CPU(s) available. Running sequentially instead.", UserWarning)
                print(f"WARNING: Parallel execution requested but only {n_cpus} CPU(s) available. Overriding: max_workers=1, use_parallel=False")
            else:
                print(f'Only {n_cpus} CPU detected. Using max_workers=1, use_parallel=False')
            max_workers = 1
            use_parallel = False
        else:
            if use_parallel_requested:
                print(f'Running in parallel: max_workers={max_workers}, use_parallel={use_parallel}')


        if fine_decoding_t_bin_size not in self.epochs_decoded_result_cache_dict:
            print(f'needs to compute: decoding_time_bin_size: {fine_decoding_t_bin_size}')
            assert (self.active_epochs_df is not None)
            active_epochs_df = deepcopy(self.active_epochs_df)
            decoded_local_epochs_result, a_decoder = self.decode_epochs_for_posterior_analysis(curr_active_pipeline=curr_active_pipeline, an_epoch_name=a_decoder_name, decoding_time_bin_size=fine_decoding_t_bin_size, active_epochs_df=active_epochs_df)
            print(f'done with all decoding.')
            if fine_decoding_t_bin_size not in self.epochs_decoded_result_cache_dict:            
                self.epochs_decoded_result_cache_dict[fine_decoding_t_bin_size] = {} ## init to empty
            self.epochs_decoded_result_cache_dict[fine_decoding_t_bin_size][a_decoder_name] = decoded_local_epochs_result
            if self.pf1D_Decoder_dict is None:
                self.pf1D_Decoder_dict = {} ## init to empty
            if a_decoder_name not in self.pf1D_Decoder_dict:
                self.pf1D_Decoder_dict[a_decoder_name] = a_decoder
            ## add the result:
            self.epochs_decoded_result_cache_dict[fine_decoding_t_bin_size][a_decoder_name] = decoded_local_epochs_result
            
            
        # Get this specific result ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[fine_decoding_t_bin_size].get(a_decoder_name, None)
        # a_decoder: BayesianPlacemapPositionDecoder = list(self.pf1D_Decoder_dict.values())[0]
        a_decoder: BayesianPlacemapPositionDecoder = self.pf1D_Decoder_dict.get(a_decoder_name, None)

        xybin_edges_kwargs = dict()
        xybin_centers_only_kwargs = dict()
        if a_decoder is None:
            xybin_centers_only_kwargs = dict(xbin_centers=self.predictive_decoding.xbin_centers, ybin_centers=self.predictive_decoding.ybin_centers)
            xybin_edges_kwargs = dict(xbin=self.predictive_decoding.xbin, ybin=self.predictive_decoding.ybin)  # ADD THIS

        else:
            xybin_centers_only_kwargs = dict(xbin_centers=a_decoder.xbin_centers, 
                ybin_centers=a_decoder.ybin_centers,
            )            
            xybin_edges_kwargs = dict(xbin=a_decoder.xbin, ybin=a_decoder.ybin,
            )


        # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=decoded_local_epochs_result, **xybin_edges_kwargs, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3) # xbin=a_decoder.xbin, ybin=a_decoder.ybin

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
            

            #TODO 2026-01-21 08:51: - [ ] Do we do anything with these this time?
            custom_computation_results_dict = DecodingLocalityMeasures.compute_locality_measures_for_posterior(
                a_p_x_given_n=curr_epoch_p_x_given_n,
                # gaussian_volume=container.predictive_decoding.gaussian_volume, ## if we have it
                **xybin_centers_only_kwargs,
                min_val_epsilon=1e-6,
                alpha_list = [0.5, 0.8],
                enable_debug_outputs=True,
                earthmovers_fn=None, debug_print=False,
            )
            # a_debug_result_dict = custom_computation_results_dict.pop('debug', None) ## remove so it isn't added to the df

            custom_computation_results_df = DecodingLocalityMeasures.perform_build_locality_measures_df(locality_measures_dict_dict={a_decoder_name: custom_computation_results_dict}, ## expects a dict with key of the epoch type, so we need to wrap it
                time_window_centers=curr_epoch_time_bin_centers, 
                **xybin_centers_only_kwargs,
            )

            custom_computation_results_df['epoch_idx'] = i ## same value for all
            custom_computation_results_df['epoch_t_bin_idx'] = custom_computation_results_df.index.astype(int).to_numpy() ## ascending values
            custom_results_list.append(custom_computation_results_dict)
            custom_results_df_list.append(custom_computation_results_df)
        ## END for i, a_row in enumerate(ensure_data...

        ## flatten/concat into a single flat df for all epochs:
        custom_results_df_list = pd.concat(custom_results_df_list, ignore_index=True)
        
        print(f'\tfinished with for epochs loops doing locality recomputations')
        

        # n_tasks: int = n_total_epochs
        # use_parallel: bool = use_parallel and (n_tasks > 1) and (n_cpus > 1)


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

        # ==================================================================================================================================================================================================================================================================================== #
        # Peak Promenece Calculation Parameters                                                                                                                                                                                                                                                #
        # ==================================================================================================================================================================================================================================================================================== #
        slice_level_multipliers = (0.25, 0.5, 0.9)
        
        # resolution_factor = 13.0
        resolution_factor = 9.0
        minimum_included_peak_height = 1e-9
        # should_use_faster_compute_single_slab_implementation: bool = True
        should_use_faster_compute_single_slab_implementation: bool = False
        # minimum_included_peak_height = None

        step: float = PeakPromenence.compute_optimal_step_size(a_masked_result.p_x_given_n_list, resolution_factor=resolution_factor)
        print(f'step: {step}')

        # decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PosteriorPeaksPeakProminence2dResult.init_from_old_PeakProminence2D_result_dict(active_peak_prominence_2d_results=old_prom_2d_result)
        
        decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=a_masked_result.p_x_given_n_list, 
            **xybin_edges_kwargs,
            **xybin_centers_only_kwargs,
            # xbin_centers=self.predictive_decoding.xbin_centers, 
            # ybin_centers=self.predictive_decoding.ybin_centers,
            step=step, minimum_included_peak_height=None, # 1m 42s - 7m 1s
            # step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
            peak_height_multiplier_probe_levels=slice_level_multipliers,
            should_use_faster_compute_single_slab_implementation=should_use_faster_compute_single_slab_implementation,
            min_considered_promenence=1e-11,
            parallel=use_parallel, max_workers=max_workers,
            # parallel=True, max_workers=4,
            # parallel=True, max_workers=None,    
        )
        ## 55m - step=1e-4, minimum_included_peak_height=1e-5
        ## 11m - step=1e-3, minimum_included_peak_height=1e-5,

        assert decoded_epoch_t_bins_promenence_result_obj is not None
        print(f'\tfinished with PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(...)')
        

        # a_decoder = self.pf1D_Decoder_dict[a_decoder_name]
        # a_masked_result = self.epochs_decoded_result_cache_dict[fine_decoding_t_bin_size][a_decoder_name] # DecodedFilterEpochsResult
        a_decoded_filter_epochs_df: pd.DataFrame = ensure_dataframe(a_masked_result.filter_epochs)
        # a_decoded_result.active_filter_epochs

        ## INPUTS: decoded_epoch_t_bins_promenence_result_obj

        # ==================================================================================================================================================================================================================================================================================== #
        # Finsh computing the final masks from the computed promenence results                                                                                                                                                                                                                 #
        # ==================================================================================================================================================================================================================================================================================== #
        # print(f'\tfinished with PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(...)')
        
        mask_included_bins_list, summit_slice_levels_list, mask_included_p_x_given_n_list_dict, epoch_prom_t_bin_high_prob_pos_masks_dict, epoch_prom_high_prob_pos_masks_dict, *extra_outs = decoded_epoch_t_bins_promenence_result_obj.compute_discrete_contour_masks(p_x_given_n_list=a_masked_result.p_x_given_n_list,
                                                                                                                                                                                                                                                                         slice_level_multipliers=slice_level_multipliers)

        print(f'\tfinished with promenence_result_obj.compute_discrete_contour_masks(...)')
        ## OUTPUTS: epoch_prom_high_prob_pos_masks_dict, epoch_prom_t_bin_high_prob_pos_masks_dict
        
        measured_positions_df: pd.DataFrame = deepcopy(self.decoding_locality.measured_positions_df)
        #TODO 2026-01-21 08:45: - [ ] `epoch_t_bin_high_prob_masks_dict ` or `epoch_high_prob_masks_dict` are used to update the final masks
        epoch_matching_past_future_positions, _an_out_tuple, a_decoded_filter_epochs_df = PredictiveDecoding.compute_specific_future_and_past_analysis(decoded_local_epochs_result=a_masked_result,
                measured_positions_df=measured_positions_df, gaussian_volume=self.predictive_decoding.gaussian_volume, ## the volume for all time bins,
                active_epochs_df=a_decoded_filter_epochs_df,
                an_epoch_name=a_decoder_name, top_v_percent=None,
                epoch_t_bin_high_prob_masks_dict=epoch_prom_t_bin_high_prob_pos_masks_dict, ## These optional kwargs (epoch_prom_high_prob_pos_masks_dict, epoch_prom_t_bin_high_prob_pos_masks_dict) being set are why the promenece results are actually used this time!
                epoch_high_prob_masks_dict=epoch_prom_high_prob_pos_masks_dict,
                a_slice_multiplier=slice_level_multipliers[0],
                progress_print=True,
                merging_adjacent_max_separation_sec = 1e-9,
                minimum_epoch_duration = 0.05,
                # merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
                should_defer_extended_computations=True, max_workers=max_workers, use_parallel=use_parallel,
        )
        epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _an_out_tuple
        # _out_epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult] = _out_processed_items_list_dict['_out_epoch_flat_mask_future_past_result']

        ## OUTPUTS: _out_epoch_flat_mask_future_past_result
        print(f'\tfinished with PredictiveDecoding.compute_specific_future_and_past_analysis(...)')
        ## OUTPUTS: epoch_matching_past_future_positions, _out_processed_items_list_dict

        print(f'\tassigning the results to self.debug_computed_dict...')
        # At the end of _filter_single_epoch_result, add:
        if a_decoder_name not in self.debug_computed_dict:
            self.debug_computed_dict[a_decoder_name] = {}

        if 'prominence_future_past_analysis' not in self.debug_computed_dict[a_decoder_name]:
            self.debug_computed_dict[a_decoder_name]['prominence_future_past_analysis'] = {} ## init new

        self.debug_computed_dict[a_decoder_name]['prominence_future_past_analysis'].update({
            'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks,
            'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks,
            'epoch_matching_positions': epoch_matching_positions,
            'past_future_info_dict': past_future_info_dict,
            'matching_pos_dfs_list': matching_pos_dfs_list,
            'matching_pos_epochs_dfs_list': matching_pos_epochs_dfs_list,
            'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj,
            'slice_level_multiplier_used': slice_level_multipliers[0],
        })

        ## add the processed items to the dict too
        self.debug_computed_dict[a_decoder_name]['prominence_future_past_analysis'].update(_out_processed_items_list_dict)
        ### _out_epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult] = masked_container.debug_computed_dict[a_decoder_name]['prominence_future_past_analysis']['_out_epoch_flat_mask_future_past_result']

        # for k, v in _out_processed_items_list_dict.items():
        #     ## add the processed items to the dict too
        #     self.debug_computed_dict[a_decoder_name]['prominence_future_past_analysis'].update(_out_processed_items_list_dict)

        print(f'\t\tdone assigning. end of function.')


        return a_masked_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj

    ## OUTPUTS: active_epochs_result (masked result),  custom_results_df_list




    # Utility Methods ____________________________________________________________________________________________________ #

    @function_attributes(short_name=None, tags=['PENDING', '2025-01-09'], input_requires=[], output_provides=[], uses=['decode_specific_epochs'], used_by=['compute_future_and_past_analysis'], creation_date='2025-01-09', related_items=[])
    def decode_epochs_for_posterior_analysis(self, curr_active_pipeline, an_epoch_name: str = 'roam', decoding_time_bin_size: float = 0.025, active_epochs_df: Optional[pd.DataFrame] = None, allow_update_instance_properties: bool=False) -> Tuple["DecodedFilterEpochsResult", "BayesianPlacemapPositionDecoder"]:
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
                
        Uses:
            self.pf1D_Decoder_dict
            
            
        Updates:
            self.epochs_decoded_result_cache_dict
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult
        
        # Ensure cache dict exists for this time bin size
        if decoding_time_bin_size not in self.epochs_decoded_result_cache_dict:
            if allow_update_instance_properties:
                self.epochs_decoded_result_cache_dict[decoding_time_bin_size] = {} ## make the new dict for this time bin size
                print(f'decoding_time_bin_size: {decoding_time_bin_size} did not exist in results... creating!')
        
        # Get or create the decoder
        a_decoder: BayesianPlacemapPositionDecoder = self.pf1D_Decoder_dict.get(an_epoch_name, None)
        if a_decoder is None:
            directional_decoders_decode_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']            
            assert directional_decoders_decode_result is not None
            if allow_update_instance_properties:
                self.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            a_decoder = directional_decoders_decode_result.pf1D_Decoder_dict[an_epoch_name]
    
        # Check cache for existing decoded result
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[decoding_time_bin_size].get(an_epoch_name, None)
        if decoded_local_epochs_result is None:
            ## if we can't find a pre-computed one:    
            decoded_local_epochs_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=deepcopy(active_epochs_df), decoding_time_bin_size=decoding_time_bin_size)
            if allow_update_instance_properties:
                self.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name] = decoded_local_epochs_result
                print(f'\tresult added to self.epochs_decoded_result_cache_dict[decoding_time_bin_size={decoding_time_bin_size}][an_epoch_name={an_epoch_name}]')

        return decoded_local_epochs_result, a_decoder


    def update_active_epochs_and_decode_posteriors_if_needed(self, curr_active_pipeline, an_epoch_name:str = 'roam', decoding_time_bin_size=0.025, 
                                        merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, ## for merging active epochs
                                        override_included_analysis_epochs: Optional[pd.DataFrame]=None, epoch_id_key_name='non_local_PBE_non_moving_epoch', force_recompute_epoch_df_columns: bool = False, allow_update_instance_properties: bool=False,
                                    ):
        """ Gets the self.active_epochs if available, or computes it using `decoding_locality.get_non_moving_PBE_non_local_epochs(...)` if it doesn't exist.
            Also gets the existing result 
        factored out of `compute_future_and_past_analysis` on 2026-01-16 

                decode_posteriors_if_needed
                
            Updates:
                self.active_epochs_df
        """
        from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior
    
        ## HARDCODED an_epoch_name
        # computed_df_col_name_prefix: str = ''
        # computed_df_col_name_prefix: str = f'{an_epoch_name}_'
        
        ## Get the non-local epochs -- where do they encode?
        # container: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
        decoding_locality: DecodingLocalityMeasures = self.decoding_locality
        
        if (override_included_analysis_epochs is not None):
            active_epochs_df: pd.DataFrame = ensure_dataframe(override_included_analysis_epochs)
        else:
            if self.active_epochs_df is None:
                ## updates the active epochs:
                non_local_PBE_non_moving_epochs_df: pd.DataFrame = decoding_locality.get_non_moving_PBE_non_local_epochs(curr_active_pipeline.sess, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec)
                active_epochs_df: pd.DataFrame = ensure_dataframe(non_local_PBE_non_moving_epochs_df)
                if allow_update_instance_properties:
                    self.active_epochs_df = ensure_dataframe(active_epochs_df) # `self.active_epochs_df` gets updated later anyway
            else:
                ## self.active_epochs_df is good, use that
                active_epochs_df: pd.DataFrame = ensure_dataframe(self.active_epochs_df)

         ## in general we want to use our active epochs:
        # active_epochs_df: pd.DataFrame = ensure_dataframe(self.active_epochs_df)
        assert active_epochs_df is not None
        assert isinstance(active_epochs_df, pd.DataFrame)
        assert len(active_epochs_df) > 0
        
        ## add the final detected non_local_pbe_epoch indicies to the decoded points:
        if 'label' not in active_epochs_df.columns:
            active_epochs_df['label'] = active_epochs_df.index.astype(int)
        else:
            active_epochs_df['label'] = active_epochs_df['label'].astype(int)

        ## Recompute columns:        
        if ('start_idx' not in active_epochs_df.columns) or ('stop_idx' not in active_epochs_df.columns) or (epoch_id_key_name not in active_epochs_df.columns) or force_recompute_epoch_df_columns:
            _out_locality_measures_df = deepcopy(decoding_locality.locality_measures_df)
            _out_locality_measures_df = _out_locality_measures_df.time_point_event.adding_epochs_identity_column(epochs_df=active_epochs_df, epoch_id_key_name=epoch_id_key_name, override_time_variable_name='t',
                                                                # epoch_label_column_name='label', no_interval_fill_value=np.nan,
                                                                epoch_label_column_name='label', no_interval_fill_value=-1,
                                                                should_replace_existing_column=True, drop_non_epoch_events=True,
                                                                overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
            # _out_locality_measures_df
            # _out_locality_measures_df.dropna(how='any', subset=['non_local_PBE_non_moving_epoch'])

            epoch_times = decoding_locality.locality_measures_df['t'].to_numpy()
            time_to_idx_map = EpochHelpers.find_epoch_times_to_data_indicies_map(active_epochs_df, epoch_times)
            

            # _out
            assert active_epochs_df is not None
            active_epochs_df: pd.DataFrame = active_epochs_df
            active_epochs_df['start_idx'] = active_epochs_df['start'].map(time_to_idx_map)
            active_epochs_df['stop_idx'] = active_epochs_df['stop'].map(time_to_idx_map)
            # matching_epoch_times_slice
            # non_local_PBE_non_moving_epochs_dft

        ## sets the self.active_epochs_df:
        assert active_epochs_df is not None
        assert isinstance(active_epochs_df, pd.DataFrame)
        # if (self.active_epochs_df != active_epochs_df):
        active_epochs_df = ensure_dataframe(active_epochs_df).reset_index(drop=True, inplace=False)
        
        if allow_update_instance_properties:
            self.active_epochs_df = ensure_dataframe(active_epochs_df)
        
        # Get the decoders to decode the epochs with higher precision ________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict.get(decoding_time_bin_size, {}).get(an_epoch_name, None)
        a_decoder = self.pf1D_Decoder_dict.get(an_epoch_name, None)
        
        if decoded_local_epochs_result is None:
            decoded_local_epochs_result, a_decoder = self.decode_epochs_for_posterior_analysis(curr_active_pipeline=curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=decoding_time_bin_size, active_epochs_df=ensure_dataframe(active_epochs_df), allow_update_instance_properties=allow_update_instance_properties)
            assert decoded_local_epochs_result is not None
            decoded_local_epochs_result.filter_epochs = ensure_dataframe(decoded_local_epochs_result.filter_epochs).reset_index(drop=True, inplace=False)    
            # decoded_local_epochs_result.filter_epochs = ensure_Epoch(decoded_local_epochs_result.filter_epochs)
            print(f'done with all decoding.')
        else:
            ## use existing
            print(f'using existing decoded result...')

        return decoded_local_epochs_result, a_decoder, active_epochs_df





    @function_attributes(short_name=None, tags=['PENDING', 'IN-PROCESS', '2025-12-20_future_and_past_analysis'], input_requires=[], output_provides=[], uses=['decode_specific_epochs'], used_by=[], creation_date='2025-12-19 14:28', related_items=[])
    def compute_future_and_past_analysis(self, an_epoch_name:str = 'roam', decoding_time_bin_size=0.025, top_v_percent: float = 0.1, 
                                        merging_adjacent_max_separation_sec: float = 0.5, minimum_epoch_duration: float = 0.050, ## for merging detected future/past position dataframes
                                        enable_updating_instance_states: bool=True,
                                        # override_included_analysis_epochs: Optional[pd.DataFrame]=None,
                                        **kwargs,
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
        # Get the decoders to decode the epochs with higher precision ________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # decoded_local_epochs_result, a_decoder, active_epochs_df = self.update_active_epochs_and_decode_posteriors_if_needed(curr_active_pipeline, an_epoch_name=an_epoch_name, decoding_time_bin_size=decoding_time_bin_size, 
        #                                                                 merging_adjacent_max_separation_sec = merging_adjacent_max_separation_sec, minimum_epoch_duration = minimum_epoch_duration, 
        #                                                                 override_included_analysis_epochs=override_included_analysis_epochs, epoch_id_key_name='non_local_PBE_non_moving_epoch', force_recompute_epoch_df_columns=False,
        #                                                             )
        

        decoded_local_epochs_result = self.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name]
        active_epochs_df = self.active_epochs_df
        
        assert decoded_local_epochs_result is not None
        assert active_epochs_df is not None
        
        ## INPUTS: decoded_local_epochs_result
        decoding_locality: DecodingLocalityMeasures = self.decoding_locality
        measured_positions_df: pd.DataFrame = decoding_locality.measured_positions_df        

        gaussian_volume = self.predictive_decoding.gaussian_volume ## the volume for all time bins

        max_workers: Optional[int] = kwargs.pop('max_workers', 2)
        # Ensure max_workers is not None - if it was explicitly None in kwargs, use default of 2
        if max_workers is None:
            max_workers = 2
        
        # Extract use_parallel from kwargs if present, otherwise use default True
        use_parallel: bool = kwargs.pop('use_parallel', True)
        
        ## decoded_local_epochs_result's epochs need to match the passed `active_epochs_df`
        epoch_matching_past_future_positions, _an_out_tuple, active_epochs_df = PredictiveDecoding.compute_specific_future_and_past_analysis(decoded_local_epochs_result=decoded_local_epochs_result, measured_positions_df=measured_positions_df, gaussian_volume=gaussian_volume,
            active_epochs_df=active_epochs_df,
            an_epoch_name=an_epoch_name, top_v_percent=top_v_percent, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
            max_workers=max_workers, use_parallel=use_parallel, **kwargs, # use_parallel=True, max_workers=2, 
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


@metadata_attributes(short_name=None, tags=['container'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-16 18:05', related_items=[])
@define(slots=False, repr=False, eq=False)
class PredictiveDecodingComputationsContainerContainer(ComputedResult):
    """ Created as a solution to discarding the original container after masking the result 

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingComputationsContainerContainer

        wcorr_shuffle_results: PredictiveDecodingComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('PredictiveDecoding', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'PredictiveDecoding is not computed.')
            
    """
    _VersionedResultMixin_version: str = "2026.01.16_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    container: Optional[PredictiveDecodingComputationsContainer] = serialized_field(default=None, repr=False)
    masked_container: Optional[PredictiveDecodingComputationsContainer] = serialized_field(default=None, repr=False)


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
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
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)



def validate_has_predictive_decoding_results(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
    """ Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    container_container: PredictiveDecodingComputationsContainerContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
    if container_container is None:
        return False
    
    # Unpacking:
    seq_results: PredictiveDecodingComputationsContainer = container_container.container
    if seq_results is None:
        return False
    
    predictive_decoding = seq_results.predictive_decoding
    if predictive_decoding is None:
        return False
    
    # # masking
    # mask_results: PredictiveDecodingComputationsContainer = container_container.masked_container
    # if mask_results is None:
    #     return False
    
    # mask_predictive_decoding = mask_results.predictive_decoding
    # if predictive_decoding is None:
    #     return False
    

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
                drop_previous_result_and_compute_fresh:bool=False, min_num_spikes_per_bin_to_be_considered_active: Optional[int]=5, mask_position_like_time_score_cutoff: Optional[float] = 0.42,  fine_time_bin_size: float=0.025, 
                enable_masked_filtered_container_before_any_comps: bool = True, should_perform_first_pass_compute_future_and_past_analysis: bool=False, enable_filter_and_final_result_processing: bool = False,
                max_workers: Optional[int]=1,
        ):
        """ Performs predictive decoding analysis to relate PBE activity to future visited locations.

        Requires:
            ['DirectionalDecodersDecoded']

        Provides:
            global_computation_results.computed_data['PredictiveDecoding']
                ['PredictiveDecoding'].predictive_decoding - PredictiveDecoding instance containing computed results

                
                
            curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding']
                
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
        import time as _time
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
        ## NEW: filtering by whether decoded posterior in each t_bin is "position-like"
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring

        _fn_name: str = 'perform_predictive_decoding_analysis'
        _total_start_time = _time.perf_counter()
        
        print(f'[{_fn_name}] ========== STARTING ==========')
        print(f'[{_fn_name}] Parameters: window_size={window_size}, fine_time_bin_size={fine_time_bin_size}, extant_decoded_time_bin_size={extant_decoded_time_bin_size}')
        print(f'[{_fn_name}] Flags: drop_previous_result_and_compute_fresh={drop_previous_result_and_compute_fresh}, enable_masked_filtered_container_before_any_comps={enable_masked_filtered_container_before_any_comps}')
        print(f'[{_fn_name}] Flags: should_perform_first_pass_compute_future_and_past_analysis={should_perform_first_pass_compute_future_and_past_analysis}, enable_filter_and_final_result_processing={enable_filter_and_final_result_processing}')

        # Handle max_workers override for parallel execution
        if max_workers == 1:
            parallel_kwargs = {'max_workers': 1, 'use_parallel': False}
            print(f'[{_fn_name}] max_workers=1: disabling all parallel execution')
        else:
            parallel_kwargs = {'max_workers': max_workers}
            print(f'[{_fn_name}] Using max_workers={max_workers} for parallel execution')

        if include_includelist is not None:
            print(f'[{_fn_name}] WARN: include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        ## Get the needed data:
        should_filter_by_active_spikes: bool = ((min_num_spikes_per_bin_to_be_considered_active is not None) and (min_num_spikes_per_bin_to_be_considered_active > 0))
        should_filter_by_position_like_posterior_bins: bool = ((mask_position_like_time_score_cutoff is not None) and (mask_position_like_time_score_cutoff > 0))
        print(f'[{_fn_name}] Filtering config: should_filter_by_active_spikes={should_filter_by_active_spikes} (min_spikes={min_num_spikes_per_bin_to_be_considered_active}), should_filter_by_position_like_posterior_bins={should_filter_by_position_like_posterior_bins} (cutoff={mask_position_like_time_score_cutoff})')
        
        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 1: MASK low-firing bins before using result                                                                                                                                                                                                                                    #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase1_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 1: Loading and masking decoded results ---')

        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = deepcopy(owning_pipeline_reference.global_computation_results.computed_data['DirectionalDecodersDecoded'])
        spikes_df: pd.DataFrame = directional_decoders_decode_result.spikes_df
        print(f'[{_fn_name}] Loaded DirectionalDecodersDecoded (deepcopy). spikes_df shape: {spikes_df.shape}')
            
        if (should_filter_by_active_spikes or should_filter_by_position_like_posterior_bins):
            ## Masked result:
            epoch_names: List[str] = list(directional_decoders_decode_result.pf1D_Decoder_dict.keys())
            a_decoder = list(directional_decoders_decode_result.pf1D_Decoder_dict.values())[0]
            n_time_bin_sizes = len(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict)
            print(f'[{_fn_name}] Applying masks to {n_time_bin_sizes} time bin size(s)...')
            
            for i_tbin, (extant_decoded_time_bin_size, a_result_decoded) in enumerate(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.items()):
                print(f'[{_fn_name}]   Processing time_bin_size {i_tbin+1}/{n_time_bin_sizes}: {extant_decoded_time_bin_size}s')
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
            print(f'[{_fn_name}] Masking complete for all time bin sizes.')
        else:
            print(f'[{_fn_name}] Skipping masking (no filters enabled).')
        ## end if (should_filter...
        
        _phase1_elapsed = _time.perf_counter() - _phase1_start_time
        print(f'[{_fn_name}] PHASE 1 complete. Elapsed: {_phase1_elapsed:.2f}s')

        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 2: Setup position data and container initialization                                                                                                                                                                                                                            #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase2_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 2: Setup position data and container ---')
        
        pos_df: pd.DataFrame = deepcopy(owning_pipeline_reference.sess.position.to_dataframe())
        print(f'[{_fn_name}] Loaded position dataframe. Shape: {pos_df.shape}')
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
        print(f'[{_fn_name}] Previously decoded time_bin_sizes: {previously_decoded_keys}')

        if extant_decoded_time_bin_size is None:
            time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
            print(f'[{_fn_name}] Using most_recent_decoding_time_bin_size: {time_bin_size}')
        else:
            available_sizes = list(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.keys())
            if extant_decoded_time_bin_size not in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict:
                raise KeyError(f"[{_fn_name}] extant_decoded_time_bin_size={extant_decoded_time_bin_size} not in available sizes: {available_sizes}")
            time_bin_size: float = extant_decoded_time_bin_size
            print(f'[{_fn_name}] Using specified extant_decoded_time_bin_size: {time_bin_size}')

        if drop_previous_result_and_compute_fresh:
            removed_predictive_decoding_result = global_computation_results.computed_data.pop('PredictiveDecoding', None)
            if removed_predictive_decoding_result is not None:
                print(f'[{_fn_name}] Removed previous "PredictiveDecoding" result (drop_previous_result_and_compute_fresh=True)')
            else:
                print(f'[{_fn_name}] No previous "PredictiveDecoding" result to remove.')

        # Initialize or upgrade container
        if ('PredictiveDecoding' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'PredictiveDecoding')):
            print(f'[{_fn_name}] Initializing new PredictiveDecodingComputationsContainerContainer...')
            a_container = PredictiveDecodingComputationsContainer(predictive_decoding=None, is_global=True)
            global_computation_results.computed_data['PredictiveDecoding'] = PredictiveDecodingComputationsContainerContainer(container=a_container, masked_container=None, is_global=True)
        elif isinstance(global_computation_results.computed_data['PredictiveDecoding'], PredictiveDecodingComputationsContainer):
            ## upgraded to container container
            print(f'[{_fn_name}] Upgrading from pre-2026-01-16 format (non-nested containers)')
            a_container = global_computation_results.computed_data['PredictiveDecoding'] ## get the original result
            a_container_container: PredictiveDecodingComputationsContainerContainer = PredictiveDecodingComputationsContainerContainer(container=a_container, masked_container=None, is_global=True)
            global_computation_results.computed_data['PredictiveDecoding'] = a_container_container
        else:
            print(f'[{_fn_name}] Using existing PredictiveDecodingComputationsContainerContainer.')

        a_container_container: PredictiveDecodingComputationsContainerContainer = global_computation_results.computed_data['PredictiveDecoding'] ## shorthand
        a_container: PredictiveDecodingComputationsContainer = a_container_container.container

        # Get or create the decoder
        if (a_container.pf1D_Decoder_dict is None) or (len(a_container.pf1D_Decoder_dict) == 0):
            ## initialize it
            assert directional_decoders_decode_result is not None
            a_container.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            print(f'[{_fn_name}] Assigned pf1D_Decoder_dict with keys: {list(a_container.pf1D_Decoder_dict.keys())}')
        
        _phase2_elapsed = _time.perf_counter() - _phase2_start_time
        print(f'[{_fn_name}] PHASE 2 complete. Elapsed: {_phase2_elapsed:.2f}s')

        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 3: Initialize DecodingLocalityMeasures                                                                                                                                                                                                                                         #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase3_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 3: Initialize DecodingLocalityMeasures ---')
        
        locality_measures = None
        try:
            print(f'[{_fn_name}] [DecodingLocalityMeasures] Initializing with time_bin_size={time_bin_size}...')
            
            if not hasattr(directional_decoders_decode_result, 'continuously_decoded_pseudo2D_decoder_dict'):
                available_attrs = list(directional_decoders_decode_result.__dict__.keys())[:10]
                raise AttributeError(f"[{_fn_name}] directional_decoders_decode_result missing 'continuously_decoded_pseudo2D_decoder_dict'. Available: {available_attrs}")
            
            if time_bin_size not in directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict:
                available_sizes = list(directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict.keys())
                raise KeyError(f"[{_fn_name}] time_bin_size={time_bin_size} not in continuously_decoded_pseudo2D_decoder_dict. Available: {available_sizes}")
            
            locality_measures = DecodingLocalityMeasures.init_from_decode_result(curr_active_pipeline=owning_pipeline_reference, directional_decoders_decode_result=directional_decoders_decode_result, extant_decoded_time_bin_size=time_bin_size, sigma=None)
            
            print(f'[{_fn_name}] [DecodingLocalityMeasures] Successfully initialized.')
            if locality_measures is not None:
                a_container.locality_measures = locality_measures

        except Exception as e:
            print(f'[{_fn_name}] [DecodingLocalityMeasures] ERROR during computation: {e}')
            raise
        
        _phase3_elapsed = _time.perf_counter() - _phase3_start_time
        print(f'[{_fn_name}] PHASE 3 complete. Elapsed: {_phase3_elapsed:.2f}s')


        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 4: Initialize and compute PredictiveDecoding                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase4_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 4: Initialize and compute PredictiveDecoding ---')
        
        try:
            print(f'[{_fn_name}] [PredictiveDecoding] Initializing with time_bin_size={time_bin_size}, window_size={window_size}...')
            
            if locality_measures is None:
                locality_measures = a_container.locality_measures
                print(f'[{_fn_name}] [PredictiveDecoding] Retrieved locality_measures from container.')

            if locality_measures is None:
                raise ValueError(f"[{_fn_name}] locality_measures is None - cannot proceed with PredictiveDecoding initialization.")

            # Get a_result_decoded from directional_decoders_decode_result
            a_result_decoded = directional_decoders_decode_result.continuously_decoded_pseudo2D_decoder_dict[time_bin_size]
            
            #TODO 2025-12-23 20:55: - [ ] Found that everything seems to be working well except that there are sometimes a few time bins out of an epoch that have poorly localized posteriors in general (they look very diffuse and like an error, maybe low firing bins)
            ### These need to be filtered out (either by diffusivity of low-firing criteria) so that when we collapse over all the time bins within each epoch we don't pick up a bunch of garbage (the diffuse bins are too liberal).
            
            # Create PredictiveDecoding using the new simplified interface
            predictive_decoding: PredictiveDecoding = PredictiveDecoding.init_from_decode_result(pos_df=pos_df, locality_measures=locality_measures, a_result_decoded=a_result_decoded, window_size=window_size)
            print(f'[{_fn_name}] [PredictiveDecoding] Successfully initialized.')

            # Use sigma from locality_measures (computed automatically) or compute from bin sizes if not available
            if locality_measures.sigma is None:
                x_step: float = np.nanmean(np.diff(predictive_decoding.xbin))
                y_step: float = np.nanmean(np.diff(predictive_decoding.ybin))
                sigma: float = np.nanmax([x_step, y_step]) * 5.0
                print(f'[{_fn_name}] [PredictiveDecoding] Computed sigma from bin sizes: {sigma}')
            else:
                sigma = locality_measures.sigma
                print(f'[{_fn_name}] [PredictiveDecoding] Using sigma from locality_measures: {sigma}')

            print(f'[{_fn_name}] [PredictiveDecoding] Computing via .compute(sigma={sigma})... (this may take a while)')
            _compute_start = _time.perf_counter()
            # Compute predictive decoding outputs
            moving_avg_dict, moving_avg_meas_pos_overlap_dict, gaussian_volume = predictive_decoding.compute(sigma=sigma)
            _compute_elapsed = _time.perf_counter() - _compute_start
            print(f'[{_fn_name}] [PredictiveDecoding] .compute() done! Elapsed: {_compute_elapsed:.2f}s')
            
            if predictive_decoding is not None:
                # Store the PredictiveDecoding instance in the container
                a_container.predictive_decoding = predictive_decoding

        except Exception as e:
            print(f'[{_fn_name}] [PredictiveDecoding] ERROR during computation: {e}')
            raise
        
        _phase4_elapsed = _time.perf_counter() - _phase4_start_time
        print(f'[{_fn_name}] PHASE 4 complete. Elapsed: {_phase4_elapsed:.2f}s')


        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 5: Container validation and masked container building                                                                                                                                                                                                                          #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase5_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 5: Container validation and masked container building ---')
        
        # Sync container reference
        if (a_container is not None) and (global_computation_results.computed_data['PredictiveDecoding'].container != a_container):
            global_computation_results.computed_data['PredictiveDecoding'].container = a_container
            print(f'[{_fn_name}] Synced a_container to global_computation_results.')

        # Validate container exists
        if a_container is None:
            raise ValueError(f"[{_fn_name}] a_container is None after Phase 4 - critical error.")
        
        if (a_container.pf1D_Decoder_dict is None) or (len(a_container.pf1D_Decoder_dict) == 0):
            ## initialize it
            print(f'[{_fn_name}] WARN: a_container.pf1D_Decoder_dict is None/empty - rebuilding from DirectionalDecodersDecoded...')
            directional_decoders_decode_result = owning_pipeline_reference.global_computation_results.computed_data['DirectionalDecodersDecoded']
            if directional_decoders_decode_result is None:
                raise ValueError(f"[{_fn_name}] DirectionalDecodersDecoded is None - cannot rebuild pf1D_Decoder_dict.")
            a_container.pf1D_Decoder_dict = deepcopy(directional_decoders_decode_result.pf1D_Decoder_dict) ## copy the independent decoders
            print(f'[{_fn_name}] Assigned pf1D_Decoder_dict with keys: {list(a_container.pf1D_Decoder_dict.keys())}')

        a_masked_container = None
        if enable_masked_filtered_container_before_any_comps:
            print(f'[{_fn_name}] Building masked_container (enable_masked_filtered_container_before_any_comps=True)...')
            _masked_build_start = _time.perf_counter()
            a_masked_container = a_container.build_masked_container(curr_active_pipeline=owning_pipeline_reference, a_t_bin_size=fine_time_bin_size, should_filter_directional_decoders_decode_result=True, should_compute_future_and_past_analysis=False, should_compute_peak_prom_analysis=False, window_size=window_size, **parallel_kwargs)
            _masked_build_elapsed = _time.perf_counter() - _masked_build_start
            print(f'[{_fn_name}] Built masked_container. Elapsed: {_masked_build_elapsed:.2f}s')
            global_computation_results.computed_data['PredictiveDecoding'].masked_container = a_masked_container
            a_container = a_masked_container ## change the target of a_container
        else:
            print(f'[{_fn_name}] Skipping masked_container building (enable_masked_filtered_container_before_any_comps=False).')
        
        _phase5_elapsed = _time.perf_counter() - _phase5_start_time
        print(f'[{_fn_name}] PHASE 5 complete. Elapsed: {_phase5_elapsed:.2f}s')

        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 6: Optional future/past analysis computation                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase6_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 6: Optional future/past analysis computation ---')

        if include_includelist is None:
            include_includelist = ['roam'] # , 'sprinkle'
        epoch_names: List[str] = include_includelist 
        print(f'[{_fn_name}] Target epoch_names for processing: {epoch_names}')

        # compute_future_and_past_analysis
        if should_perform_first_pass_compute_future_and_past_analysis:
            print(f'[{_fn_name}] Running compute_future_and_past_analysis for {len(epoch_names)} epoch(s)...')
            for i_epoch, an_epoch_name in enumerate(epoch_names):    
                try:
                    print(f'[{_fn_name}]   [{i_epoch+1}/{len(epoch_names)}] compute_future_and_past_analysis for epoch "{an_epoch_name}"...')
                    _epoch_start = _time.perf_counter()
                    if an_epoch_name not in a_container.debug_computed_dict:
                        a_container.debug_computed_dict[an_epoch_name] = {}
                    _out = a_container.compute_future_and_past_analysis(an_epoch_name=an_epoch_name, decoding_time_bin_size=fine_time_bin_size, override_included_analysis_epochs=a_container.active_epochs_df, disable_segmentation=True, **parallel_kwargs)
                    epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _out
                    a_container.debug_computed_dict[an_epoch_name].update({'epoch_high_prob_pos_masks': epoch_high_prob_pos_masks, 'epoch_t_bins_high_prob_pos_masks': epoch_t_bins_high_prob_pos_masks, 'epoch_matching_positions': epoch_matching_positions, 'past_future_info_dict': past_future_info_dict})
                    _epoch_elapsed = _time.perf_counter() - _epoch_start
                    print(f'[{_fn_name}]     Completed "{an_epoch_name}" in {_epoch_elapsed:.2f}s')
                except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
                    print(f'[{_fn_name}]     WARN: compute_future_and_past_analysis failed for "{an_epoch_name}": {e}. Skipping.')
                except Exception as e:
                    print(f'[{_fn_name}]     ERROR: Unexpected exception for "{an_epoch_name}": {e}')
                    raise
        else:
            print(f'[{_fn_name}] Skipping compute_future_and_past_analysis (should_perform_first_pass_compute_future_and_past_analysis=False).')
        
        _phase6_elapsed = _time.perf_counter() - _phase6_start_time
        print(f'[{_fn_name}] PHASE 6 complete. Elapsed: {_phase6_elapsed:.2f}s')
        
        
        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 7: Optional filter and final result processing                                                                                                                                                                                                                                 #
        # ==================================================================================================================================================================================================================================================================================== #
        _phase7_start_time = _time.perf_counter()
        print(f'[{_fn_name}] --- PHASE 7: Optional filter and final result processing ---')
        
        if enable_filter_and_final_result_processing:
            if (a_masked_container is None):
                # Need to build masked_container first
                print(f'[{_fn_name}] Building masked_container (was None, needed for final processing)...')
                a_container_container = global_computation_results.computed_data.get('PredictiveDecoding', None)
                if a_container_container is None:
                    raise ValueError(f"[{_fn_name}] PredictiveDecoding not found in computed_data during Phase 7.")
                a_container = a_container_container.container
                if a_container is None:
                    raise ValueError(f"[{_fn_name}] a_container is None during Phase 7.")
                _masked_build_start = _time.perf_counter()
                a_masked_container = a_container.build_masked_container(curr_active_pipeline=owning_pipeline_reference, a_t_bin_size=fine_time_bin_size, should_filter_directional_decoders_decode_result=True, should_compute_future_and_past_analysis=False, should_compute_peak_prom_analysis=False, window_size=window_size, **parallel_kwargs)
                _masked_build_elapsed = _time.perf_counter() - _masked_build_start
                print(f'[{_fn_name}] Built masked_container. Elapsed: {_masked_build_elapsed:.2f}s')
            
            # Use fine_time_bin_size from the masked container (don't shadow the parameter)
            effective_fine_time_bin_size: float = a_masked_container.most_recent_decoding_time_bin_size
            print(f'[{_fn_name}] Using effective_fine_time_bin_size={effective_fine_time_bin_size} from masked_container')

            print(f'[{_fn_name}] Running final_refine_single_epoch_result_masks for {len(epoch_names)} epoch(s)...')
            for i_epoch, an_epoch_name in enumerate(epoch_names):    
                try:
                    print(f'[{_fn_name}]   [{i_epoch+1}/{len(epoch_names)}] final_refine_single_epoch_result_masks for epoch "{an_epoch_name}"...')
                    _epoch_start = _time.perf_counter()
                    if an_epoch_name not in a_masked_container.debug_computed_dict:
                        a_masked_container.debug_computed_dict[an_epoch_name] = {}
                    
                    active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = a_masked_container.final_refine_single_epoch_result_masks(curr_active_pipeline=owning_pipeline_reference, fine_decoding_t_bin_size=effective_fine_time_bin_size, a_decoder_name=an_epoch_name, **parallel_kwargs)
                    a_masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
                    _epoch_elapsed = _time.perf_counter() - _epoch_start
                    print(f'[{_fn_name}]     Completed "{an_epoch_name}" in {_epoch_elapsed:.2f}s')
                except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
                    print(f'[{_fn_name}]     WARN: final_refine_single_epoch_result_masks failed for "{an_epoch_name}": {e}. Skipping.')
                except Exception as e:
                    print(f'[{_fn_name}]     ERROR: Unexpected exception for "{an_epoch_name}": {e}')
                    raise
        else:
            print(f'[{_fn_name}] Skipping final result processing (enable_filter_and_final_result_processing=False).')
        
        _phase7_elapsed = _time.perf_counter() - _phase7_start_time
        print(f'[{_fn_name}] PHASE 7 complete. Elapsed: {_phase7_elapsed:.2f}s')

        # ==================================================================================================================================================================================================================================================================================== #
        # PHASE 8: Final container synchronization                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        print(f'[{_fn_name}] --- PHASE 8: Final container synchronization ---')
        
        ## Ensure all containers are properly stored in global_computation_results
        if (a_container_container is not None) and (global_computation_results.computed_data['PredictiveDecoding'] != a_container_container):
            global_computation_results.computed_data['PredictiveDecoding'] = a_container_container
            print(f'[{_fn_name}] Synced a_container_container to global_computation_results.')
        
        if (a_container is not None) and (a_container != a_masked_container) and (global_computation_results.computed_data['PredictiveDecoding'].container != a_container):
            global_computation_results.computed_data['PredictiveDecoding'].container = a_container
            print(f'[{_fn_name}] Synced a_container to PredictiveDecoding.container.')
            
        if (a_masked_container is not None) and (global_computation_results.computed_data['PredictiveDecoding'].masked_container != a_masked_container):
            global_computation_results.computed_data['PredictiveDecoding'].masked_container = a_masked_container
            print(f'[{_fn_name}] Synced a_masked_container to PredictiveDecoding.masked_container.')

        _total_elapsed = _time.perf_counter() - _total_start_time
        print(f'[{_fn_name}] ========== COMPLETE ==========')
        print(f'[{_fn_name}] Total elapsed time: {_total_elapsed:.2f}s')


        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures, PredictiveDecodingComputationsContainer, PredictiveDecodingComputationsContainerContainer
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder
        
        container_container: PredictiveDecodingComputationsContainerContainer = curr_active_pipeline.global_computation_results.computed_data['PredictiveDecoding'] ## shorthand
        assert container_container is not None
        container: PredictiveDecodingComputationsContainer = container_container.container
        masked_container: PredictiveDecodingComputationsContainer = container_container.masked_container


        ## OLD               
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
from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget
# from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedFigureController ## 

@define(slots=False, repr=False, eq=False)
class MaskDataSource:
    """ provides data to its owner related to each epoch 
    """
    matching_pos_epochs_dfs_list: List = field() # self.container.predictive_decoding.matching_pos_epochs_dfs_list
    matching_pos_dfs_list: List = field() # = self.container.predictive_decoding.matching_pos_dfs_list

    epoch_high_prob_pos_masks: List = field() # self.container.predictive_decoding.epoch_high_prob_pos_masks
    epoch_t_bins_high_prob_pos_masks: List = field() # self.container.predictive_decoding.epoch_t_bins_high_prob_pos_masks

    ## decoded results like
    filter_epochs: pd.DataFrame = field()
    p_x_given_n_list: List = field() #  = self.decoded_result.p_x_given_n_list[an_epoch_idx]  # Shape: (n_x_bins, n_y_bins, n_time_bins)

    xbin: NDArray = field(default=None)
    ybin: NDArray = field(default=None)
    xbin_centers: NDArray = field(default=None)
    ybin_centers: NDArray = field(default=None)

    curr_position_df: pd.DataFrame = field(default=None)
    matching_pos_merged_segment_epochs_dfs_list: List[Optional[pd.DataFrame]] = field(default=None)
    # merged_segment_epochs: Optional[pd.DataFrame] = field(default=None)

    @classmethod
    def init_from_list_of_MatchingPastFuturePositionsResult(cls, epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], filter_epochs: pd.DataFrame, **kwargs) -> "MaskDataSource":
        a_new_ds = cls(
                     matching_pos_dfs_list=[v.relevant_positions_df for v in epoch_flat_mask_future_past_result],
                     matching_pos_epochs_dfs_list=[v.matching_pos_epochs_df for v in epoch_flat_mask_future_past_result],
               epoch_high_prob_pos_masks=[v.epoch_high_prob_mask for v in epoch_flat_mask_future_past_result], epoch_t_bins_high_prob_pos_masks=[v.epoch_t_bins_high_prob_pos_mask for v in epoch_flat_mask_future_past_result],
               filter_epochs=filter_epochs, p_x_given_n_list=[a_single_epoch_result.decoded_epoch_result.p_x_given_n for a_single_epoch_result in epoch_flat_mask_future_past_result],
            #    matching_pos_merged_segment_epochs_dfs_list=[v.merged_segment_epochs for v in epoch_flat_mask_future_past_result],
            matching_pos_merged_segment_epochs_dfs_list=[v.merged_segment_epochs for v in epoch_flat_mask_future_past_result],
               **kwargs,
        )
        
        return a_new_ds



    @function_attributes(short_name=None, tags=['get-data', 'by-epoch', ''], input_requires=[], output_provides=[], uses=[], used_by=['multi_trajectory_color_plotter'], creation_date='2026-01-21 05:10', related_items=[])
    def _prepare_epoch_data(self, an_epoch_idx: int, minimum_included_matching_sequence_length: Optional[int]=None) -> Dict[str, Any]:
        """ Used by `multi_trajectory_color_plotter` to get the data for a given epoch, currently used manually only
        
            epoch_data = _prepare_epoch_data(a_ds=a_flat_matching_results_list_ds, an_epoch_idx=5)
            # curr_matching_past_future_positions_df_dict = epoch_data['curr_matching_past_future_positions_df_dict']
            curr_matching_past_future_positions_df_list = epoch_data['curr_matching_past_future_positions_df_list']


        Adds: curr_matching_positions_df['is_included_in_merged']
        
        """
        ## from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations.PredictiveDecodingDisplayWidget._prepare_epoch_data`

        col_name: str = 'past_future_matching_pos_epoch_id'

        curr_matching_epochs_df: pd.DataFrame = self.matching_pos_epochs_dfs_list[an_epoch_idx]
        curr_matching_positions_df: pd.DataFrame = self.matching_pos_dfs_list[an_epoch_idx]
        curr_matching_epochs_df_dict: Dict[types.PastFutureCategory, pd.DataFrame] = curr_matching_epochs_df.pho.partition_df_dict('is_future_present_past')
        curr_matching_merged_segment_epochs_df_dict: Optional[Dict[types.PastFutureCategory, pd.DataFrame]] = None
        good_merged_segment_epochs: Optional[pd.DataFrame] = None
        
        should_filter_to_minimum: bool = (minimum_included_matching_sequence_length is not None) and (minimum_included_matching_sequence_length > 0) and (self.matching_pos_merged_segment_epochs_dfs_list is not None)
        
        if should_filter_to_minimum:
            curr_merged_segment_epochs: pd.DataFrame = deepcopy(self.matching_pos_merged_segment_epochs_dfs_list[an_epoch_idx])            
            ## INPUTS: curr_merged_segment_epochs, curr_matching_positions_df, curr_matching_epochs_df

            ## filter the sequences shorter than `minimum_included_matching_sequence_length`
            ## UPDATES: curr_matching_positions_df, 
            # minimum_included_matching_sequence_length
            assert curr_merged_segment_epochs is not None
            good_merged_segment_epochs: pd.DataFrame = curr_merged_segment_epochs[(curr_merged_segment_epochs['num_epoch_t_bins'] >= minimum_included_matching_sequence_length)]
            assert 'matching_found_relevant_merged_pos_epoch' in curr_matching_positions_df, f"curr_matching_positions_df.columns: {list(curr_matching_positions_df.columns)}"
            # curr_epoch_is_included_in_merged = np.isin(curr_matching_positions_df['matching_found_relevant_merged_pos_epoch'], good_merged_segment_epochs['label'])
            curr_epoch_is_included_in_merged = np.logical_and(curr_matching_positions_df['matching_found_relevant_merged_pos_epoch'].isin(good_merged_segment_epochs['label']), (curr_matching_positions_df['matching_found_relevant_pos_epoch'] > -1))
            curr_matching_positions_df['is_included_in_merged'] = curr_epoch_is_included_in_merged
            
            # relevant_positions_df[relevant_positions_df['matching_found_relevant_merged_pos_epoch'] > -1]
            good_only_relevant_positions_df: pd.DataFrame = curr_matching_positions_df[curr_epoch_is_included_in_merged]
            good_only_included_epoch_labels: NDArray = np.unique(good_only_relevant_positions_df['matching_found_relevant_pos_epoch'].to_numpy())
            # good_only_included_epoch_labels
            ## INPUTS: matching_pos_epochs_df
            good_only_matching_pos_epochs_df: pd.DataFrame = deepcopy(curr_matching_epochs_df)[curr_matching_epochs_df['label'].isin(good_only_included_epoch_labels)]
            
            ## OUTPUTS: good_merged_segment_epochs, good_only_relevant_positions_df, good_only_matching_pos_epochs_df

            curr_matching_positions_df = good_only_relevant_positions_df
            curr_matching_epochs_df = good_only_matching_pos_epochs_df
            
            curr_matching_epochs_df_dict = curr_matching_epochs_df.pho.partition_df_dict('is_future_present_past')
            curr_matching_merged_segment_epochs_df_dict = good_merged_segment_epochs.pho.partition_df_dict('is_future_present_past') ## IDK if this is needed
            ## needs to overwrite: curr_matching_positions_df, curr_matching_epochs_df, curr_matching_epochs_df_dict, curr_matching_merged_segment_epochs_df_dict
        ## END if should_filter_to_minimum...
        
        curr_matching_past_future_positions_df_dict: Dict[types.PastFutureCategory, Dict[types.epoch_index, pd.DataFrame]] = {}


        if should_filter_to_minimum:
            # for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
            for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_merged_segment_epochs_df_dict.items(): ## #TODO 2026-02-03 00:11: - [ ] uses `curr_matching_merged_segment_epochs_df_dict`
                a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
                an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int)
                if 'is_included_in_merged' not in a_curr_matching_positions_df.columns:
                    a_curr_epoch_is_included_in_merged = np.isin(a_curr_matching_positions_df['matching_found_relevant_merged_pos_epoch'], good_merged_segment_epochs['label'])
                    a_curr_matching_positions_df['is_included_in_merged'] = a_curr_epoch_is_included_in_merged
                    
                ## DROP non-included
                a_curr_matching_positions_df = a_curr_matching_positions_df[a_curr_matching_positions_df['is_included_in_merged']] ## reset indicies here or anything?

                if len(a_curr_matching_positions_df) > 0:
                    a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name,
                                                                                                                                override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True,
                                                                                                                                drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
                
                curr_matching_positions_df_dict: Dict[types.epoch_index, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
                curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
            ## END for ...
            
        else:
            for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
                a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
                an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int)
                if should_filter_to_minimum:
                    if 'is_included_in_merged' not in a_curr_matching_positions_df.columns:
                        a_curr_epoch_is_included_in_merged = np.isin(a_curr_matching_positions_df['matching_found_relevant_merged_pos_epoch'], good_merged_segment_epochs['label'])
                        a_curr_matching_positions_df['is_included_in_merged'] = a_curr_epoch_is_included_in_merged
                        
                    ## DROP non-included
                    a_curr_matching_positions_df = a_curr_matching_positions_df[a_curr_matching_positions_df['is_included_in_merged']] ## reset indicies here or anything?
                    
                ## END if should_filter_to_minimum...
                if len(a_curr_matching_positions_df) > 0:
                    a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name,
                                                                                                                                override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True,
                                                                                                                                drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
                
                curr_matching_positions_df_dict: Dict[types.epoch_index, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
                curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
            ## END for ...


        # ## OLD way
        # for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
        #     a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
        #     an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int)
        #     if should_filter_to_minimum:
        #         if 'is_included_in_merged' not in a_curr_matching_positions_df.columns:
        #             a_curr_epoch_is_included_in_merged = np.isin(a_curr_matching_positions_df['matching_found_relevant_merged_pos_epoch'], good_merged_segment_epochs['label'])
        #             a_curr_matching_positions_df['is_included_in_merged'] = a_curr_epoch_is_included_in_merged
                    
        #         ## DROP non-included
        #         a_curr_matching_positions_df = a_curr_matching_positions_df[a_curr_matching_positions_df['is_included_in_merged']] ## reset indicies here or anything?

        #     ## END if should_filter_to_minimum...
        #     if len(a_curr_matching_positions_df) > 0:
        #         a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name,
        #                                                                                                                     override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True,
        #                                                                                                                     drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)

        #     curr_matching_positions_df_dict: Dict[types.epoch_index, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
        #     curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
        # ## END for ...
        

        curr_matching_past_future_positions_df_list: Dict[types.PastFutureCategory, List[pd.DataFrame]] = {k:list(v.values()) for k, v in curr_matching_past_future_positions_df_dict.items()}
        ## OUTPUTS: curr_matching_past_future_positions_df_dict
        return {
            'curr_matching_epochs_df': curr_matching_epochs_df,
            'curr_matching_positions_df': curr_matching_positions_df,
            'curr_matching_epochs_df_dict': curr_matching_epochs_df_dict,
            'curr_matching_good_merged_segment_epochs_df': good_merged_segment_epochs,
            'curr_matching_merged_segment_epochs_df_dict': curr_matching_merged_segment_epochs_df_dict, 
            'curr_matching_past_future_positions_df_dict': curr_matching_past_future_positions_df_dict,
            'curr_matching_past_future_positions_df_list': curr_matching_past_future_positions_df_list,
        }




@metadata_attributes(short_name=None, tags=['partially-working', 'matplotlib', '3-pane', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 14:42', related_items=[])
@define(slots=False, repr=False, eq=False)
class PredictiveDecodingDisplayWidget:
    """ Plots 3 panels side-by-side: Left: Past positions, Mid: Decoded Epoch Posterior, Right: Future positions
    
    Internally-Uses:
        epoch_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_high_prob_pos_masks', None)
        epoch_t_bins_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_t_bins_high_prob_pos_masks', None)
        self.container.predictive_decoding.matching_pos_dfs_list
        self.container.predictive_decoding.matching_pos_epochs_dfs_list

                self.decoded_result 

    
    Usage:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidget

        a_widget: PredictiveDecodingDisplayWidget = PredictiveDecodingDisplayWidget.init_from_container(container=container, decoding_time_bin_size=0.025, an_epoch_name='roam')
        a_widget
        
        
        
    Usage 2:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import MatchingPastFuturePositionsResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import MaskDataSource
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidget

        a_flat_matching_results_list_ds: MaskDataSource = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result, filter_epochs=a_decoded_filter_epochs_df)
        a_flat_matching_ds_widget: PredictiveDecodingDisplayWidget = PredictiveDecodingDisplayWidget.init_from_datasource(datasource=a_flat_matching_results_list_ds, curr_position_df=container.decoding_locality.pos_df,
                                                                                                        pf_decoder=a_decoder, decoded_result=a_decoded_result)
        a_flat_matching_ds_widget


    Usage 3:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import MaskDataSource
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingDisplayWidget

        a_new_ds: MaskDataSource = MaskDataSource(matching_pos_dfs_list=matching_pos_dfs_list, matching_pos_epochs_dfs_list=matching_pos_epochs_dfs_list,
                    epoch_high_prob_pos_masks=epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks=epoch_t_bins_high_prob_pos_masks,
                    filter_epochs=a_decoded_filter_epochs_df, p_x_given_n_list=a_decoded_result.p_x_given_n_list,
        )

        a_widget_ds: PredictiveDecodingDisplayWidget = PredictiveDecodingDisplayWidget.init_from_datasource(datasource=a_new_ds, curr_position_df=container.decoding_locality.pos_df,
                                                                                                        pf_decoder=a_decoder, decoded_result=a_decoded_result)
        a_widget_ds


        
        
    ## FILTERED VERSION

        # 2025-01-08 - Mask based on position-like bins only _________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        a_masked_result, scoring_results = PositionLikePosteriorScoring.filter_to_position_like_epochs_only(decoded_local_epochs_result=decoded_local_epochs_result, position_like_score_cutoff=0.42, num_min_position_like_t_bins=3,
                                                                                                                                        xbin=a_decoder.xbin, ybin=a_decoder.ybin,
                                                                                                                                     )


    """
    container: PredictiveDecodingComputationsContainer = field(default=None)

    result_datasource: Optional[MaskDataSource] = field(default=None)
        
    xbin: NDArray = field(default=None)
    ybin: NDArray = field(default=None)
    xbin_centers: NDArray = field(default=None)
    ybin_centers: NDArray = field(default=None)
    curr_position_df: pd.DataFrame = field(default=None)
    
    pf1D_Decoder: BasePositionDecoder = field(default=None)
    decoded_result: DecodedFilterEpochsResult = field(default=None)

    ## Display Variables
    trajectory_displaying_plotter: Dict[types.PastFutureCategory, DecodedTrajectoryMatplotlibPlotter] = field(default=Factory(dict))
    trajectory_epochs_pages: Dict[types.PastFutureCategory, List] = field(default=Factory(dict))
    trajectory_active_page_idx: Dict[types.PastFutureCategory, int] = field(default=Factory(dict))
    
    ## Dock UI Variables
    dock_area: Any = field(default=None)
    dock_window: Any = field(default=None)
    dock_widgets: Dict[str, Any] = field(default=Factory(dict))
    dock_canvas_widgets: Dict[str, Any] = field(default=Factory(dict))
    dock_container_widgets: Dict[str, Any] = field(default=Factory(dict))
    epoch_slider: Any = field(default=None)
    epoch_value_label: Any = field(default=None)
    page_controls: Dict[str, Dict[str, PaginationControlWidget]] = field(default=Factory(dict))

    active_epoch_idx: int = field(default=20)
    
    disable_showing_epoch_high_prob_pos_masks: bool = field(default=True)
    should_use_flipped_images: bool = field(default=False)
    

    # ==================================================================================================================================================================================================================================================================================== #
    # INIT Helpers                                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def init_from_container(cls, container: PredictiveDecodingComputationsContainer, decoding_time_bin_size: float, an_epoch_name: str, active_epoch_idx: int=0, **kwargs) -> "PredictiveDecodingDisplayWidget":
        """

        """
        decoded_local_epochs_result = container.epochs_decoded_result_cache_dict[decoding_time_bin_size][an_epoch_name]
        pf_decoder = container.pf1D_Decoder_dict.get(an_epoch_name, None)
        decoded_result: DecodedFilterEpochsResult = decoded_local_epochs_result
        curr_position_df: pd.DataFrame = deepcopy(container.decoding_locality.pos_df)

        ## INPUTS: directional_laps_results, decoder_ripple_filter_epochs_decoder_result_dict
        if pf_decoder is not None:
            xbin = deepcopy(pf_decoder.xbin)
            xbin_centers = deepcopy(pf_decoder.xbin_centers)
            ybin_centers = deepcopy(pf_decoder.ybin_centers)
            ybin = deepcopy(pf_decoder.ybin)
        else:
            xbin = deepcopy(container.decoding_locality.xbin)
            xbin_centers = deepcopy(container.decoding_locality.xbin_centers)
            ybin_centers = deepcopy(container.decoding_locality.ybin_centers)
            ybin = deepcopy(container.decoding_locality.ybin)

        # num_filter_epochs: int = decoded_local_epochs_result.num_filter_epochs

        _obj = cls(
            container=container,
            result_datasource=None,
            xbin=xbin,
            ybin=ybin,
            xbin_centers=xbin_centers,
            ybin_centers=ybin_centers,
            curr_position_df=curr_position_df,
            pf1D_Decoder=pf_decoder, decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx,
        )

        return _obj
    

    @classmethod
    def init_from_datasource(cls, datasource: MaskDataSource, curr_position_df: pd.DataFrame, pf_decoder: Any,  decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int=0, **kwargs) -> "PredictiveDecodingDisplayWidget":
        """
        curr_position_df: pd.DataFrame = deepcopy(container.decoding_locality.pos_df)

        """
        ## INPUTS: directional_laps_results, decoder_ripple_filter_epochs_decoder_result_dict
        if pf_decoder is not None:
            xbin = deepcopy(pf_decoder.xbin)
            xbin_centers = deepcopy(pf_decoder.xbin_centers)
            ybin_centers = deepcopy(pf_decoder.ybin_centers)
            ybin = deepcopy(pf_decoder.ybin)

        # num_filter_epochs: int = decoded_local_epochs_result.num_filter_epochs

        _obj = cls(
            container=None,
            result_datasource=datasource,
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
        # assert len(self.container.predictive_decoding.matching_pos_dfs_list) > 0
        # assert len(self.container.predictive_decoding.matching_pos_epochs_dfs_list) > 0
        assert len(self.result_datasource.matching_pos_dfs_list) > 0
        assert len(self.result_datasource.matching_pos_epochs_dfs_list) > 0
                

        self.setup()
        self.buildUI()


    def setup(self):
        """Calculate constants (max_subplots_per_category, extent), prepare data structures.
        
        Updates:
        
            self.container.predictive_decoding.matching_pos_epochs_dfs_list
            
        """
        if (self.result_datasource is None) and ((self.container is not None) and (self.decoded_result is not None)):
            print(f'initializing the result datasrouce from the container and decoded result...')
            matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
            
            active_epochs_df: pd.DataFrame = ensure_dataframe(self.container.active_epochs_df)
            
            # Prepare matching_pos_epochs_dfs_list with is_future_present_past labels
            # for i, a_row in enumerate(ensure_dataframe(self.decoded_result.filter_epochs).itertuples(index=False)):
            
            for i, a_row in enumerate(active_epochs_df.itertuples(index=False)):
                
                a_matching_pos_epochs: pd.DataFrame = matching_pos_epochs_dfs_list[i]
                curr_epoch_start_t: float = a_row.start
                curr_epoch_stop_t: float = a_row.stop
                
                is_relevant_past_times = (a_matching_pos_epochs['stop'] < curr_epoch_start_t)
                is_relevant_future_times = (a_matching_pos_epochs['start'] > curr_epoch_stop_t)
                a_matching_pos_epochs['is_future_present_past'] = 'present'
                a_matching_pos_epochs.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
                a_matching_pos_epochs.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'
                
                self.container.predictive_decoding.matching_pos_epochs_dfs_list[i] = a_matching_pos_epochs # updated in the widget? very strange.

            ## END for i, a_row in enumerate(active_epochs_df.itertuples(index=False))....        
            self.result_datasource = MaskDataSource(matching_pos_dfs_list=self.container.predictive_decoding.matching_pos_dfs_list, matching_pos_epochs_dfs_list=self.container.predictive_decoding.matching_pos_epochs_dfs_list,
                                            epoch_high_prob_pos_masks=self.container.predictive_decoding.epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks=self.container.predictive_decoding.epoch_t_bins_high_prob_pos_masks,
                                            filter_epochs=active_epochs_df, p_x_given_n_list=self.decoded_result.p_x_given_n_list)
        
        elif (self.result_datasource is not None) and (self.container is None):
            active_epochs_df: pd.DataFrame = ensure_dataframe(self.result_datasource.filter_epochs)
            # Prepare matching_pos_epochs_dfs_list with is_future_present_past labels
            # for i, a_row in enumerate(ensure_dataframe(self.decoded_result.filter_epochs).itertuples(index=False)):            
            for i, a_row in enumerate(active_epochs_df.itertuples(index=False)):
                
                a_matching_pos_epochs: pd.DataFrame = self.result_datasource.matching_pos_epochs_dfs_list[i]
                curr_epoch_start_t: float = a_row.start
                curr_epoch_stop_t: float = a_row.stop
                
                is_relevant_past_times = (a_matching_pos_epochs['stop'] < curr_epoch_start_t)
                is_relevant_future_times = (a_matching_pos_epochs['start'] > curr_epoch_stop_t)
                a_matching_pos_epochs['is_future_present_past'] = 'present'
                a_matching_pos_epochs.loc[is_relevant_past_times, 'is_future_present_past'] = 'past'
                a_matching_pos_epochs.loc[is_relevant_future_times, 'is_future_present_past'] = 'future'
                
                self.result_datasource.matching_pos_epochs_dfs_list[i] = a_matching_pos_epochs # updated in the widget? very strange.





        # Calculate max_subplots_per_category
        self.max_subplots_per_category = self._calculate_max_subplots()
        
        # Calculate extent for posterior plots
        self.extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
        
        # Initialize display_widgets dict for MatplotlibTimeSynchronizedWidget instances
        if not hasattr(self, 'display_widgets'):
            self.display_widgets: Dict[str, Any] = {}


    def _calculate_max_subplots(self) -> Dict[str, int]:
        """Pre-calculate max subplots needed (called once in setup)."""
        # matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        # matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        
        matching_pos_epochs_dfs_list = self.result_datasource.matching_pos_epochs_dfs_list
        matching_pos_dfs_list = self.result_datasource.matching_pos_dfs_list
                
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
        a_decoded_traj_plotter: DecodedTrajectoryMatplotlibPlotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers, prev_heatmaps=overlay_prev_heatmaps)
        # a_decoded_traj_plotter.params.plot_decoded_trajectories_2d_kwargs = dict(active_page_index=0, fixed_columns=4, plot_actual_lap_lines=True, use_theoretical_tracks_instead=False,
        #                                                                         plot_mode='scatter', c='red', cmap='Reds', alpha=0.55, s=5,
        #                                                                         posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=1e-12, posterior_should_perform_reshape=False, # rotate_to_vertical
        # )
        
        self.trajectory_displaying_plotter['past'] = a_decoded_traj_plotter
        
        # ## NOTE: `epoch_ids` used here and in the following function call actually refer to `found_pos_segment_ids`, not epochs, it's just how the `a_decoded_traj_plotter` class is named:
        # fig, axs, epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=self.curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=found_pos_segment_ids, curr_num_subplots=curr_num_subplots,
        #                                                                                 active_page_index=0, fixed_columns=4, plot_actual_lap_lines=True, use_theoretical_tracks_instead=False, existing_ax=existing_ax,
        #                                                                                 plot_mode='scatter', c='red', cmap='Reds', alpha=0.55, s=5,
        #                                                                                 posteriors=overlay_posterior, posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=1e-12,
        #                                                                                 posterior_should_perform_reshape=False, # rotate_to_vertical
        #                                                                             )
        
            

        # fig, axs, laps_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True)


    def _build_posterior_widget(self):
        """Create decoded posterior widget (MatplotlibTimeSynchronizedWidget).
        
        
        posterior_widget: MatplotlibTimeSynchronizedWidget = self.display_widgets.get('decoded_posterior', None)
        assert posterior_widget is not None
        ax_main = posterior_widget.plots.axes_dict['main']
        ax_tiny_dict = posterior_widget.plots.axes_dict['ax_tiny_dict']
        
        # ax_tiny_dict
        
        """
        # Plot decoded posterior heatmap for 'decoded_posterior' dock
        category_name = 'decoded_posterior'
        # Check if we need to initialize (create new widget) or update existing one
        needed_init: bool = (category_name not in self.dock_widgets) or (category_name not in self.display_widgets)
        if not needed_init:
            return False ## stop here:
        else:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            posterior_dock = Dock("Decoded Posterior", size=(600, 700), closable=True)
            
            ## get the "past" dock to position the new dock correctly
            past_dock = self.dock_widgets.get("past", None)
            if past_dock is not None:
                self.dock_area.addDock(posterior_dock, 'right', past_dock)
            else:
                self.dock_area.addDock(posterior_dock, 'left')
            self.dock_widgets[category_name] = posterior_dock
            
            # Create and initialize the widget immediately
            posterior_widget: MatplotlibTimeSynchronizedWidget = MatplotlibTimeSynchronizedWidget(size=(8, 6), dpi=72, constrained_layout=True, disable_toolbar=False)
            posterior_widget.params.max_num_time_bins_to_show = 8
            
            ## Setup axes
            fig = posterior_widget.getFigure()
            fig.clear()
            
            posterior_widget.plots.gridspec_dict = {'gs': None, 'gs_tiny': None}
            posterior_widget.plots.axes_dict = {'main': None, 'ax_tiny_dict': {}}
            
            gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 2], hspace=0.1)
            posterior_widget.plots.gridspec_dict['gs'] = gs
            ax_main = fig.add_subplot(gs[0, 0])
            posterior_widget.plots.axes_dict['main'] = ax_main

            gs_tiny = gridspec.GridSpecFromSubplotSpec(1, posterior_widget.params.max_num_time_bins_to_show, subplot_spec=gs[1, 0], wspace=0.005)
            posterior_widget.plots.gridspec_dict['gs_tiny'] = gs_tiny
            
            for t_bin_idx in range(posterior_widget.params.max_num_time_bins_to_show):
                ax_tiny = fig.add_subplot(gs_tiny[0, t_bin_idx])
                ax_tiny.set_xticks([])
                ax_tiny.set_yticks([])
                ax_tiny.set_xlabel(f't={t_bin_idx}', fontsize=8)
                posterior_widget.plots.axes_dict['ax_tiny_dict'][t_bin_idx] = ax_tiny

            ## Add the widget
            posterior_dock.addWidget(posterior_widget)
            self.display_widgets[category_name] = posterior_widget


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
        a_decoded_traj_plotter: DecodedTrajectoryMatplotlibPlotter = DecodedTrajectoryMatplotlibPlotter(a_result=self.decoded_result, xbin=self.xbin, xbin_centers=self.xbin_centers, ybin=self.ybin, ybin_centers=self.ybin_centers, prev_heatmaps=overlay_prev_heatmaps)
        self.trajectory_displaying_plotter['future'] = a_decoded_traj_plotter


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


    # ==================================================================================================================================================================================================================================================================================== #
    # On Update Methods                                                                                                                                                                                                                                                                    #
    # ==================================================================================================================================================================================================================================================================================== #
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


    def _build_page_controls(self, a_past_future_name: str, num_pages: int):
        """Build page navigation controls for a trajectory widget using PaginationControlWidget.
        
        Note: The controls widget is created but NOT added to the dock here.
        It should be added to a container widget that also contains the plot.
        This method just creates and stores the control widget.
        """
        # Check if controls already exist
        if a_past_future_name in self.page_controls and 'widget' in self.page_controls[a_past_future_name]:
            # Controls already exist, just update them
            self._update_page_controls_visibility(a_past_future_name, num_pages)
            return
        
        # Create PaginationControlWidget
        pagination_widget = PaginationControlWidget(n_pages=num_pages)
        
        # Connect signals
        pagination_widget.jump_to_page.connect(lambda page_idx: self._on_page_jump(a_past_future_name, page_idx))
        pagination_widget.jump_previous_page.connect(lambda: self._on_page_change(a_past_future_name, -1))
        pagination_widget.jump_next_page.connect(lambda: self._on_page_change(a_past_future_name, 1))
        
        # Store references
        if a_past_future_name not in self.page_controls:
            self.page_controls[a_past_future_name] = {}
        self.page_controls[a_past_future_name]['widget'] = pagination_widget
        
        # Set initial page index if needed
        initial_page_idx = self.trajectory_active_page_idx.get(a_past_future_name, 0)
        if initial_page_idx != 0:
            pagination_widget.programmatically_update_page_idx(initial_page_idx, block_signals=True)
        
        # Set initial visibility
        self._update_page_controls_visibility(a_past_future_name, num_pages)


    def _update_page_controls_visibility(self, a_past_future_name: str, num_pages: int):
        """Update visibility and state of page controls based on number of pages."""
        if a_past_future_name not in self.page_controls:
            return
        
        page_controls = self.page_controls[a_past_future_name]
        should_show = num_pages > 1
        active_page_idx = self.trajectory_active_page_idx.get(a_past_future_name, 0)
        
        if 'widget' in page_controls and page_controls['widget'] is not None:
            pagination_widget = page_controls['widget']
            pagination_widget.setVisible(should_show)
            
            if should_show:
                # Update the number of pages
                if pagination_widget.state.n_pages != num_pages:
                    pagination_widget.state.n_pages = num_pages
                    pagination_widget._on_update_pagination()
                
                # Update the current page index if it changed externally
                if pagination_widget.state.current_page_idx != active_page_idx:
                    pagination_widget.programmatically_update_page_idx(active_page_idx, block_signals=True)


    def _on_page_jump(self, a_past_future_name: str, page_idx: int):
        """Handle direct page jump from PaginationControlWidget."""
        # Update the page index
        self.trajectory_active_page_idx[a_past_future_name] = page_idx
        
        # Re-render the widget with the new page
        self._refresh_trajectory_widget(a_past_future_name)


    def _on_page_change(self, a_past_future_name: str, direction: int):
        """Handle page navigation button clicks (direction: -1 for prev, 1 for next)."""
        epochs_pages = self.trajectory_epochs_pages.get(a_past_future_name, [])
        num_pages = len(epochs_pages)
        if num_pages == 0:
            return
        
        current_page = self.trajectory_active_page_idx.get(a_past_future_name, 0)
        new_page = current_page + direction
        new_page = max(0, min(new_page, num_pages - 1))
        
        if new_page != current_page:
            self.trajectory_active_page_idx[a_past_future_name] = new_page
            
            # Update pagination widget if it exists
            if a_past_future_name in self.page_controls and 'widget' in self.page_controls[a_past_future_name]:
                pagination_widget = self.page_controls[a_past_future_name]['widget']
                pagination_widget.programmatically_update_page_idx(new_page, block_signals=True)
            
            # Re-render the widget
            self._refresh_trajectory_widget(a_past_future_name)


    def _refresh_trajectory_widget(self, a_past_future_name: str):
        """Refresh a trajectory widget with the current epoch and page."""
        # Re-use the current epoch data
        epoch_data = self._prepare_epoch_data(an_epoch_idx=self.active_epoch_idx)
        self._update_trajectory_widget(a_past_future_name, self.active_epoch_idx, epoch_data)


    def _validate_epoch_idx(self, an_epoch_idx: int) -> int:
        """Validate and clamp epoch index."""
        # num_epochs = len(ensure_dataframe(self.decoded_result.filter_epochs))
        num_epochs = len(ensure_dataframe(self.result_datasource.filter_epochs))
        if an_epoch_idx < 0 or an_epoch_idx >= num_epochs:
            print(f"Warning: epoch_idx {an_epoch_idx} is out of bounds (0-{num_epochs-1}). Clamping to valid range.")
            an_epoch_idx = max(0, min(an_epoch_idx, num_epochs - 1))
        return an_epoch_idx


    # ==================================================================================================================================================================================================================================================================================== #
    # Datasource                                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def _prepare_epoch_data(self, an_epoch_idx: int) -> Dict[str, Any]:
        """Extract and prepare data for current epoch."""
        # matching_pos_dfs_list = self.container.predictive_decoding.matching_pos_dfs_list
        # matching_pos_epochs_dfs_list = self.container.predictive_decoding.matching_pos_epochs_dfs_list
        
        matching_pos_dfs_list = self.result_datasource.matching_pos_dfs_list
        matching_pos_epochs_dfs_list = self.result_datasource.matching_pos_epochs_dfs_list
        
        curr_matching_epochs_df: pd.DataFrame = matching_pos_epochs_dfs_list[an_epoch_idx]
        curr_matching_positions_df: pd.DataFrame = matching_pos_dfs_list[an_epoch_idx]
        curr_matching_epochs_df_dict: Dict[types.PastFutureCategory, pd.DataFrame] = curr_matching_epochs_df.pho.partition_df_dict('is_future_present_past')
        
        curr_matching_past_future_positions_df_dict: Dict[types.PastFutureCategory, Dict[int, pd.DataFrame]] = {}
        
        for a_past_future_name, an_epoch_specific_past_position_dfs in curr_matching_epochs_df_dict.items():
            a_curr_matching_positions_df = deepcopy(curr_matching_positions_df)
            an_epoch_specific_past_position_dfs['label'] = an_epoch_specific_past_position_dfs['label'].astype(int)
            col_name: str = 'past_future_matching_pos_epoch_id'
            a_curr_matching_positions_df = a_curr_matching_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=an_epoch_specific_past_position_dfs, epoch_id_key_name=col_name, override_time_variable_name='t', epoch_label_column_name='label', no_interval_fill_value=-1, should_replace_existing_column=True, drop_non_epoch_events=True, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH)
            curr_matching_positions_df_dict: Dict[int, pd.DataFrame] = a_curr_matching_positions_df.pho.partition_df_dict(col_name)
            curr_matching_past_future_positions_df_dict[a_past_future_name] = curr_matching_positions_df_dict
        

        # if (minimum_included_matching_sequence_length is not None) and (minimum_included_matching_sequence_length > 0):
        #     ## filter the sequences shorter than `minimum_included_matching_sequence_length`
        #     # minimum_included_matching_sequence_length
        #     raise NotImplementedError(f'#TODO 2026-01-23 13:06: - [ ] Finish')
        #     assert _test_epoch_result.merged_segment_epochs is not None
        #     merged_segment_epochs = _test_epoch_result.merged_segment_epochs
        #     long_merged_segment_epochs: pd.DataFrame = merged_segment_epochs[(merged_segment_epochs['num_epoch_t_bins'] > min_num_spanning_bins)]
        #     long_only_relevant_merged_positions_df: pd.DataFrame = relevant_merged_positions_df[np.isin(relevant_merged_positions_df[merged_found_pos_epoch_id_key_name], long_merged_segment_epochs['label'])]
        #     long_only_relevant_merged_positions_df
            

        return {
            'curr_matching_epochs_df': curr_matching_epochs_df,
            'curr_matching_positions_df': curr_matching_positions_df,
            'curr_matching_epochs_df_dict': curr_matching_epochs_df_dict,
            'curr_matching_past_future_positions_df_dict': curr_matching_past_future_positions_df_dict,
        }


    def _get_posterior_data(self, an_epoch_idx: int, get_high_prob_mask_instead: bool=False, should_use_flipped_images: Optional[bool]=None, max_num_t_bins_to_get: int = 10) -> Tuple[np.ndarray, Optional[List[np.ndarray]], int]:
        """Extract posterior data for epoch.

        Uses:
            self.decoded_result
            self.container.predictive_decoding.epoch_t_bins_high_prob_pos_masks
        
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx=an_epoch_idx)
        
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=False)
        
        mask_2d, time_bin_masks, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=True)
        
        [np.shape(v) for v in self.container.predictive_decoding.epoch_t_bins_high_prob_pos_masks] ## less-filtered
        [np.shape(v) for v in self.decoded_result.p_x_given_n_list] ## more filtered
        
        
        """
        should_use_flipped_images: bool = should_use_flipped_images or self.should_use_flipped_images ## use self.should_use_flpped_images if no override provided.
        
        should_get_posterior: bool = (not get_high_prob_mask_instead)
        get_high_prob_mask_instead = get_high_prob_mask_instead or (not self.disable_showing_epoch_high_prob_pos_masks)

        p_x_given_n = None
        posterior_2d = None
        
        if get_high_prob_mask_instead:
            # epoch_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_high_prob_pos_masks', None)
            epoch_high_prob_pos_masks = self.result_datasource.epoch_high_prob_pos_masks
            if (epoch_high_prob_pos_masks is not None):
                print(f'using high_prob mask version from .epoch_high_prob_pos_masks!')
                posterior_2d = epoch_high_prob_pos_masks[an_epoch_idx]
            else:
                should_get_posterior = True

        if should_get_posterior:
            p_x_given_n = self.result_datasource.p_x_given_n_list[an_epoch_idx]  # Shape: (n_x_bins, n_y_bins, n_time_bins)
            # p_x_given_n = self.decoded_result.p_x_given_n_list[an_epoch_idx]  # Shape: (n_x_bins, n_y_bins, n_time_bins)
            posterior_2d = np.sum(p_x_given_n, axis=2) ## collapse over time

        time_bin_posteriors = None
        num_time_bins_to_show: int = 0
        
        # epoch_t_bins_high_prob_pos_masks
        if get_high_prob_mask_instead:
            # epoch_t_bins_high_prob_pos_masks = getattr(self.container.predictive_decoding, 'epoch_t_bins_high_prob_pos_masks', None)
            epoch_t_bins_high_prob_pos_masks = self.result_datasource.epoch_t_bins_high_prob_pos_masks        
            if (epoch_t_bins_high_prob_pos_masks is not None): # self.disable_showing_epoch_high_prob_pos_masks
                print(f'using high_prob mask version from .epoch_t_bins_high_prob_pos_masks!')
                time_bin_posteriors = epoch_t_bins_high_prob_pos_masks[an_epoch_idx]
                if len(time_bin_posteriors) > 0:
                    num_time_bins: int = time_bin_posteriors.shape[2]
                    num_time_bins_to_show: int = min(max_num_t_bins_to_get, num_time_bins)
                    time_bin_posteriors = [time_bin_posteriors[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]
                else:
                    print(f'could not get time bins from p_x_given_n (raw posterior).')
                    num_time_bins_to_show = 0
                    time_bin_posteriors = []
            else:
                should_get_posterior = True
                                

        else:
            ## Use raw posteriors:
            should_get_posterior = True
            # if p_x_given_n is not None:
            #     num_time_bins = p_x_given_n.shape[2]
            #     num_time_bins_to_show = min(max_num_t_bins_to_get, num_time_bins)
            #     time_bin_posteriors = [p_x_given_n[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]


        if should_get_posterior:
            ## Use raw posteriors:
            if p_x_given_n is not None:
                num_time_bins = p_x_given_n.shape[2]
                num_time_bins_to_show = min(max_num_t_bins_to_get, num_time_bins)
                time_bin_posteriors = [p_x_given_n[:, :, t_bin_idx] for t_bin_idx in range(num_time_bins_to_show)]
            else:
                print(f'could not get time bins from p_x_given_n (raw posterior).')
                num_time_bins_to_show = 0
                time_bin_posteriors = []


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


    # ==================================================================================================================================================================================================================================================================================== #
    # Rendering                                                                                                                                                                                                                                                                            #
    # ==================================================================================================================================================================================================================================================================================== #
    
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
            raise NotImplementedError(f'a_past_future_name: {a_past_future_name} not in curr_matching_past_future_positions_df_dict: {list(curr_matching_past_future_positions_df_dict.keys())}')
            # return
        
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
        if(existing_ax is not None) and (not needed_init):
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
        
        ## Get the active page index for this widget (default to 0 if not set)
        active_page_idx = self.trajectory_active_page_idx.get(a_past_future_name, 0)
        
        ## NOTE: `epoch_ids` used here and in the following function call actually refer to `found_pos_segment_ids`, not epochs, it's just how the `a_decoded_traj_plotter` class is named:
        fig, axs, epochs_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(curr_position_df=self.curr_position_df, epoch_specific_position_dfs=epoch_specific_position_dfs, epoch_ids=found_pos_segment_ids, curr_num_subplots=curr_num_subplots,
                                                                                        active_page_index=active_page_idx, fixed_columns=4, plot_actual_lap_lines=True, use_theoretical_tracks_instead=False, existing_ax=existing_ax,
                                                                                        plot_mode='scatter', c='red', cmap='Reds', alpha=0.55, s=5, posteriors=overlay_posterior, posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=1e-12,
                                                                                        posterior_should_perform_reshape=False, # rotate_to_vertical
                                                                                        should_include_trajectory_arrows=True,
                                                                                    )
        
        ## Store epochs_pages for this widget
        self.trajectory_epochs_pages[a_past_future_name] = epochs_pages
        
        ## Get number of pages
        num_pages = len(epochs_pages) if epochs_pages else 0
        
        # Set visibility for all axes (hide unused axes where epoch_id == -1, indicating padded/empty data)
        if axs is not None and isinstance(axs, np.ndarray) and axs.ndim == 2:
            # First, make all axes visible to reset any previously hidden axes
            for row in range(axs.shape[0]):
                for col in range(axs.shape[1]):
                    if axs[row, col] is not None:
                        a_grid_lines = a_decoded_traj_plotter._helper_add_bin_grid_lines(an_ax=axs[row, col], xbin=self.xbin, ybin=self.ybin)
                        axs[row, col].set_visible(True)
            
            # Then hide unused axes (where epoch_id == -1)
            if len(epochs_pages) > 0 and active_page_idx < len(epochs_pages):
                active_page_epoch_ids = epochs_pages[active_page_idx]
                if hasattr(a_decoded_traj_plotter, 'row_column_indicies') and a_decoded_traj_plotter.row_column_indicies is not None:
                    row_column_indicies = a_decoded_traj_plotter.row_column_indicies
                    for linear_idx, epoch_id in enumerate(active_page_epoch_ids):
                        if (epoch_id == -1):
                            if linear_idx < len(row_column_indicies[0]) and linear_idx < len(row_column_indicies[1]):
                                curr_row = row_column_indicies[0][linear_idx]
                                curr_col = row_column_indicies[1][linear_idx]
                                if curr_row < axs.shape[0] and curr_col < axs.shape[1]:
                                    axs[curr_row, curr_col].set_visible(False)
        
        num_pages = len(epochs_pages) if epochs_pages else 0
        perform_update_title_subtitle(fig=fig, ax=None, title_string=f"{a_past_future_name} - epoch_idx: {an_epoch_idx} | Page {active_page_idx + 1}/{num_pages}")
        

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
                
                # Create pagination controls BEFORE creating container
                # Always create them (even if hidden initially) to ensure single initialization
                # Use num_pages from current data, or 1 as placeholder if no pages yet
                initial_num_pages = max(1, num_pages) if num_pages > 0 else 1
                self._build_page_controls(a_past_future_name, initial_num_pages)
                
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
                
                # Create container widget that holds both plot and pagination controls
                container_widget = QtWidgets.QWidget()
                container_layout = QtWidgets.QVBoxLayout()
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(0)
                
                # Add plot widget (with stretch to take available space)
                container_layout.addWidget(plot_widget, stretch=1)
                
                # Always add pagination widget to container (even if hidden initially)
                if a_past_future_name in self.page_controls and 'widget' in self.page_controls[a_past_future_name]:
                    control_widget = self.page_controls[a_past_future_name]['widget']
                    container_layout.addWidget(control_widget)
                    # Set fixed height to match pattern used in other widgets (PaginationMixins, stacked_epoch_slices)
                    control_widget.setFixedHeight(21)
                    # Set visibility based on actual num_pages
                    control_widget.setVisible(num_pages > 1)
                    # Update the widget state with actual num_pages
                    if num_pages > 0:
                        self._update_page_controls_visibility(a_past_future_name, num_pages)
                
                container_widget.setLayout(container_layout)
                
                # Add container to dock (instead of just the plot widget)
                dock.addWidget(container_widget)
                
                # Store reference to canvas widget and container
                self.dock_canvas_widgets[a_past_future_name] = canvas
                if not hasattr(self, 'dock_container_widgets'):
                    self.dock_container_widgets = {}
                self.dock_container_widgets[a_past_future_name] = container_widget
                
                # Close the figure window if it's open (since it's now embedded in the dock)
                plt.close(fig)                

        else:
            ## just redraw - axes are already cleared above before plotting
            canvas = self.dock_canvas_widgets.get(a_past_future_name)
            if canvas is not None:
                # The axes were already cleared before plot_decoded_trajectories_2d was called
                # Just trigger a redraw
                # canvas.draw_idle()
                canvas.draw()
            
            ## Update existing pagination controls (never create new ones here)
            if a_past_future_name in self.page_controls and 'widget' in self.page_controls[a_past_future_name]:
                # Only update existing controls, never create new ones
                self._update_page_controls_visibility(a_past_future_name, num_pages)

        ## alternative to the above?
        widget = self.display_widgets.get(a_past_future_name)
        if widget is not None:
            widget.draw()


    def _update_posterior_widget(self, an_epoch_idx: int, debug_print=True, **kwargs):
        """Update decoded posterior display."""
        
        # Use _helper_add_heatmap for consistent display with past/future panes
        import matplotlib.ticker as ticker
        

        def _subfn_plot_posterior_with_potential_overlay(ax, posterior_2d: np.ndarray, posterior_alpha=0.65, posterior_cmap='Greens', posterior_masking_value=None,
                                                        posterior_should_perform_reshape=None, extent=None,
                                                        overlay_posterior_2d: Optional[NDArray]=None, overlay_alpha = 0.08, overlay_masking_value=None, overlay_cmap='Greens',
                                                        **kwargs):
            """ plots a posterior and an optional overlay on the same axes
            captures: xbin, ybin, xbin_centers, ybin_centers, posterior_should_use_flipped, debug_print
            
            _main_out, _overlay_out = _subfn_plot_posterior_with_potential_overlay(
            """
            # Plot main posterior using _helper_add_heatmap
            _main_out = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                an_ax=ax,
                xbin_centers=xbin_centers, ybin_centers=ybin_centers,
                a_p_x_given_n=posterior_2d,
                a_time_bin_centers=None,
                rotate_to_vertical=posterior_should_use_flipped,
                custom_image_extent=extent,
                time_cmap=posterior_cmap,
                should_perform_reshape=posterior_should_perform_reshape,
                posterior_masking_value=posterior_masking_value, full_posterior_opacity=posterior_alpha, debug_print=debug_print,
            )
            # heatmaps_main, image_extent, plots_data = _main_out

            _overlay_out = None
            # Add overlay of main posterior with low alpha (if enabled)
            if overlay_posterior_2d is not None:                
                _overlay_out = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
                    an_ax=ax,
                    xbin_centers=xbin_centers, ybin_centers=ybin_centers,
                    a_p_x_given_n=overlay_posterior_2d,
                    a_time_bin_centers=None,
                    rotate_to_vertical=posterior_should_use_flipped,
                    custom_image_extent=extent,
                    time_cmap=overlay_cmap,
                    should_perform_reshape=posterior_should_perform_reshape, posterior_masking_value=overlay_masking_value, full_posterior_opacity=overlay_alpha,
                )

            # Add xbin/ybin grid lines using the helper function (after heatmaps are plotted so grid is on top)
            _out_grid_lines = DecodedTrajectoryMatplotlibPlotter._helper_add_bin_grid_lines(an_ax=ax, xbin=xbin, ybin=ybin, xbin_centers=xbin_centers, ybin_centers=ybin_centers, rotate_to_vertical=posterior_should_use_flipped, should_plot_on_top=True)

            return (_main_out, _overlay_out)
        


        # widget = self.display_widgets.get('decoded_posterior')
        # if widget is None:
        #     return
        
        posterior_widget: MatplotlibTimeSynchronizedWidget = self.display_widgets.get('decoded_posterior', None)
        assert posterior_widget is not None
        ax_main = posterior_widget.plots.axes_dict['main']
        ax_tiny_dict = posterior_widget.plots.axes_dict['ax_tiny_dict']

        xbin = self.xbin # _centers if self.xbin_centers is not None else self.xbin
        ybin = self.ybin # self.ybin_centers if self.ybin_centers is not None else self.ybin
        
        xbin_centers = self.xbin_centers if self.xbin_centers is not None else self.xbin
        ybin_centers = self.ybin_centers if self.ybin_centers is not None else self.ybin
        
        # ==================================================================================================================================================================================================================================================================================== #
        # Get Data                                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        # override_should_use_flipped_images: bool = False
        override_should_use_flipped_images: bool = None
        posterior_2d, time_bin_posteriors, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=False, should_use_flipped_images=override_should_use_flipped_images)
        mask_2d, time_bin_masks, num_time_bins_to_show = self._get_posterior_data(an_epoch_idx, get_high_prob_mask_instead=True, should_use_flipped_images=override_should_use_flipped_images)
        
        # self._update_posterior_plot(widget, posterior_2d=mask_2d, time_bin_posteriors=time_bin_masks, num_time_bins_to_show=num_time_bins_to_show, an_epoch_idx=an_epoch_idx, overlay_posterior_2d=None, posterior_cmap='Greens', posterior_alpha=0.95, show_overlay=False)
        # self._update_posterior_plot(widget, posterior_2d=posterior_2d, time_bin_posteriors=time_bin_posteriors, num_time_bins_to_show=num_time_bins_to_show, an_epoch_idx=an_epoch_idx, overlay_posterior_2d=mask_2d, posterior_cmap='Greens', posterior_alpha=0.95, show_overlay=True)
                
        # ==================================================================================================================================================================================================================================================================================== #
        # Update Figure                                                                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        ## where does self.extent come from? self.extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
        # Use provided extent or fall back to self.extent
        posterior_extent = self.extent # extent if (extent is not None) else self.extent
        # posterior_should_use_flipped: bool = self.should_use_flipped_images

        posterior_should_use_flipped: bool = False
        posterior_common_subfn_kwargs = dict(xbin=xbin, ybin=ybin,
                                              xbin_centers=xbin_centers, ybin_centers=ybin_centers,
                                            #   posterior_should_perform_reshape=True, posterior_should_use_flipped=posterior_should_use_flipped, extent=posterior_extent,
                                               posterior_should_perform_reshape=False, posterior_should_use_flipped=posterior_should_use_flipped, extent=posterior_extent,
                                              )
        posterior_main_subfn_kwargs = dict(posterior_alpha=0.6, posterior_cmap='viridis', posterior_masking_value=None) # 1e-12
        posterior_overlay_subfn_kwargs = dict(overlay_alpha=0.75, overlay_cmap='Greens', overlay_masking_value=1e-3)

        posterior_subfn_all_kwargs = dict(**posterior_common_subfn_kwargs, **posterior_main_subfn_kwargs, **posterior_overlay_subfn_kwargs)       
        ## posterior_subfn_all_kwargs:

        has_valid_tiny_posteriors: bool = (time_bin_posteriors is not None) and (num_time_bins_to_show > 0)
        if has_valid_tiny_posteriors:
            all_time_bin_values = np.concatenate([tb.flatten() for tb in time_bin_posteriors])
            vmin_shared = np.nanmin(all_time_bin_values)
            vmax_shared = np.nanmax(all_time_bin_values)

            all_time_bin_masks_values = np.concatenate([tb.flatten() for tb in time_bin_masks])
            vmin_masks_shared = np.nanmin(all_time_bin_masks_values)
            vmax_masks_shared = np.nanmax(all_time_bin_masks_values)

        if debug_print:
            print(f'posterior_extent: {posterior_extent}')
            print(f'posterior_should_use_flipped: {posterior_should_use_flipped}')


        # ==================================================================================================================================================================================================================================================================================== #
        # Plot Main                                                                                                                                                                                                                                                                            #
        # ==================================================================================================================================================================================================================================================================================== #
        # _main_out, _overlay_out = _subfn_plot_posterior_with_potential_overlay(ax=ax_main, posterior_2d=posterior_2d, **posterior_subfn_all_kwargs)
        ax_main.clear() ## clear it
        _main_out, _overlay_out = _subfn_plot_posterior_with_potential_overlay(ax=ax_main, posterior_2d=posterior_2d, overlay_posterior_2d=mask_2d,
                                                                                **posterior_subfn_all_kwargs)
        # ax_main.set_xlabel('X Position')
        # ax_main.set_ylabel('Y Position')
        ax_main.set_title(f'Decoded Posterior Heatmap - Epoch {an_epoch_idx}')

        # ==================================================================================================================================================================================================================================================================================== #
        # Plot Time Bins (Tiny)                                                                                                                                                                                                                                                                #
        # ==================================================================================================================================================================================================================================================================================== #
        for t_bin_idx, ax_tiny in ax_tiny_dict.items():
            ## iterate through all tiny axes:
            # ax_tiny = ax_tiny_dict[t_bin_idx] # fig.add_subplot(gs_tiny[0, t_bin_idx])
            ax_tiny.clear() ## clear it
            is_valid_time_bin_idx: bool = (t_bin_idx < num_time_bins_to_show)
            # ax_tiny.set_visible(True)
            
            if is_valid_time_bin_idx:
                _main_out_tiny, _overlay_out_tiny = _subfn_plot_posterior_with_potential_overlay(ax=ax_tiny, posterior_2d=time_bin_posteriors[t_bin_idx], overlay_posterior_2d=time_bin_masks[t_bin_idx], 
                                                                                                    **posterior_subfn_all_kwargs, # time_cmap='viridis',
                                                                                                    )
                heatmaps_tiny, image_extent_tiny, plots_data_tiny = _main_out_tiny
                heatmaps_overlay_tiny, _, _ = _overlay_out_tiny
                
                # Apply shared color scale to time bin heatmap
                if heatmaps_tiny and (len(heatmaps_tiny) > 0):
                    # heatmaps_tiny[0].set_clim(vmin=vmin_shared, vmax=vmax_shared)
                    for a_heatmap in heatmaps_tiny:
                        a_heatmap.set_clim(vmin=vmin_shared, vmax=vmax_shared)
                                            
                # Apply shared color scale to time bin heatmap
                if heatmaps_overlay_tiny and (len(heatmaps_overlay_tiny) > 0):
                    for a_heatmap in heatmaps_overlay_tiny:
                        a_heatmap.set_clim(vmin=vmin_masks_shared, vmax=vmax_masks_shared)


                # ax_tiny.set_xticks([])
                # ax_tiny.set_yticks([])
                ax_tiny.set_xlabel(f't={t_bin_idx}', fontsize=8)
                ax_tiny.set_visible(True)
            else:
                ## invalid time bin index, no data, so clear the plot
                ax_tiny.set_visible(False)

        
        posterior_widget.draw()
        


    @function_attributes(short_name=None, tags=['widget', 'GUI', 'display', 'interactive', 'position-like', 'pred', 'prospective'], input_requires=[], output_provides=[], uses=['DecodedTrajectoryMatplotlibPlotter'], used_by=[], creation_date='2026-01-09 02:04', related_items=[])
    def update_displayed_epoch(self, an_epoch_idx: int = 0):
        """Main entry point - validate, prepare data, update all widgets."""
        # Validate epoch index
        an_epoch_idx = self._validate_epoch_idx(an_epoch_idx)
        
        # Reset page indices to 0 when epoch changes (optional: could preserve per-epoch page indices)
        # For now, reset to 0 for simplicity
        if an_epoch_idx != self.active_epoch_idx:
            for widget_name in ['past', 'future']:
                self.trajectory_active_page_idx[widget_name] = 0
                # Update slider if it exists
                if widget_name in self.page_controls and self.page_controls[widget_name].get('slider') is not None:
                    self.page_controls[widget_name]['slider'].setValue(0)
        
        # Update slider value if it exists (block signals to avoid recursion)
        if self.epoch_slider is not None:
            self.epoch_slider.blockSignals(True)
            self.epoch_slider.setValue(an_epoch_idx)
            self.epoch_slider.blockSignals(False)
            if self.epoch_value_label is not None:
                self.epoch_value_label.setText(f"{an_epoch_idx}")
        
        # Prepare epoch data
        epoch_data = self._prepare_epoch_data(an_epoch_idx=an_epoch_idx)
        
        # Update all widgets
        self._update_past_widget(an_epoch_idx=an_epoch_idx, epoch_data=epoch_data)
        self._update_posterior_widget(an_epoch_idx=an_epoch_idx)
        self._update_future_widget(an_epoch_idx=an_epoch_idx, epoch_data=epoch_data)
        
        # Update active epoch index
        self.active_epoch_idx = an_epoch_idx
        if self.epoch_value_label is not None:
            self.epoch_value_label.setText(f"{an_epoch_idx}")
                
        

# from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingHelpers

@function_attributes(short_name=None, tags=['pyqtgraph', 'trajectory'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 14:40', related_items=[])
def multi_trajectory_color_plotter(position_dfs: List[pd.DataFrame], fixed_columns: int = 5, return_widget: bool = True, maze_extent: Optional[Tuple[float, float, float, float]] = None,
        overlay_mask: Optional[NDArray] = None, single_plot: bool = False, 
        color_fn: Optional[Callable] = None, cmap: Optional[Union[str, Callable]] = None, 
        color_value_column: Optional[str] = None, color_value_range: Optional[Tuple[float, float]] = None):
    """ Takes a list of position dataframes representing separate trajectories and plots them.
    
    Fully flexible color mapping system that supports:
    - Custom color functions
    - Matplotlib or pyqtgraph colormaps
    - Value-based coloring from dataframe columns
    
    Args:
        position_dfs: List of position dataframes, each with 'x' and 'y' columns. Optional 't' column for time.
        fixed_columns: Number of columns in the grid layout. Default is 5. Only used when single_plot=False.
        return_widget: If True, returns (GraphicsLayoutWidget, list of PlotItems). If False, returns only list of PlotItems.
        maze_extent: Optional tuple of (xmin, xmax, ymin, ymax) to set fixed x/y limits for all subplots. If None, auto-ranges each plot.
        overlay_mask: Optional 2D numpy array to render as a low-alpha overlay on each subplot. Extents are set to maze_extent if provided, otherwise viewport edges.
        single_plot: If True, plots all trajectories on the same axes as separate series. If False, plots each trajectory in its own subplot in a grid. Default is False.
        color_fn: Optional callable that maps point data to colors. Signature: color_fn(x, y, t, trajectory_idx, point_idx, df) -> QColor or (R, G, B) or (R, G, B, A) or color name/hex.
                  If None, uses default per-trajectory colors.
        cmap: Optional colormap for value-based coloring. Can be:
              - String name of matplotlib colormap (e.g., 'viridis', 'plasma')
              - String name of pyqtgraph colormap (e.g., 'CET-L4')
              - Callable that maps [0, 1] -> (R, G, B, A) in [0, 1] range
              Requires color_value_column to be specified.
        color_value_column: Optional column name in dataframes to use for colormap-based coloring. 
                            Values will be normalized to [0, 1] based on color_value_range or auto-computed range.
        color_value_range: Optional tuple (min, max) for normalizing color_value_column. If None, computed from all data.
    
    Returns:
        Tuple of (GraphicsLayoutWidget, list of PlotItems) if return_widget=True, else just list of PlotItems.

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import multi_trajectory_color_plotter

        # Default: per-trajectory colors
        plot_widget, plot_items = multi_trajectory_color_plotter(position_dfs=dfs_list)
        
        # Time-based colormap coloring
        plot_widget, plot_items = multi_trajectory_color_plotter(
            position_dfs=dfs_list, 
            cmap='plasma',
            color_value_column='t',
            single_plot=True
        )
        
        # Custom color function
        def my_color_fn(x, y, t, traj_idx, point_idx, df):
            # Color by x position
            return pg.mkColor(int(255 * (x - xmin) / (xmax - xmin)), 0, 0)
        
        plot_widget, plot_items = multi_trajectory_color_plotter(
            position_dfs=dfs_list,
            color_fn=my_color_fn,
            single_plot=True
        )
        
        # Categorical colormap with saturation fade (shows direction)
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import create_categorical_saturation_fade_color_fn
        
        color_fn = create_categorical_saturation_fade_color_fn(
            position_dfs=dfs_list,
            categorical_cmap='tab10',
            saturation_start=0.9,  # High saturation at start
            saturation_end=0.3     # Low saturation at end
        )
        
        plot_widget, plot_items = multi_trajectory_color_plotter(
            position_dfs=dfs_list,
            color_fn=color_fn,
            single_plot=True
        )

    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    import numpy as np
    import pandas as pd
    from typing import Callable, Union
    
    # Validate input
    if (position_dfs is None) or len(position_dfs) == 0:
        raise ValueError("position_dfs must be a non-empty list")
    
    for i, df in enumerate(position_dfs):
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError(f"position_dfs[{i}] must have 'x' and 'y' columns")
    
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
    
    # Setup color mapping system
    num_epochs = len(position_dfs)
    colormap_fn = None
    color_value_min = None
    color_value_max = None
    
    # Process colormap if provided
    if cmap is not None:
        if color_value_column is None:
            raise ValueError("cmap requires color_value_column to be specified")
        
        if isinstance(cmap, str):
            # Try matplotlib first, then pyqtgraph
            try:
                import matplotlib.pyplot as plt
                matplotlib_cmap = plt.get_cmap(cmap)
                colormap_fn = lambda v: matplotlib_cmap(v)
            except:
                # Try pyqtgraph colormap
                try:
                    pg_cmap = pg.colormap.get(cmap)
                    if pg_cmap is not None:
                        # pyqtgraph colormap returns QColor, convert to RGBA tuple
                        def pg_cmap_wrapper(v):
                            qcolor = pg_cmap.map(v)
                            return (qcolor.red() / 255.0, qcolor.green() / 255.0, qcolor.blue() / 255.0, qcolor.alpha() / 255.0)
                        colormap_fn = pg_cmap_wrapper
                    else:
                        raise ValueError(f"Colormap '{cmap}' not found in matplotlib or pyqtgraph")
                except Exception as e:
                    raise ValueError(f"Could not load colormap '{cmap}': {e}")
        elif callable(cmap):
            colormap_fn = cmap
        else:
            raise ValueError("cmap must be a string (colormap name) or callable")
        
        # Compute color value range
        if color_value_range is not None:
            color_value_min, color_value_max = color_value_range
        else:
            # Auto-compute from all data
            all_values = []
            for df in position_dfs:
                if color_value_column in df.columns:
                    valid_vals = df[color_value_column].dropna()
                    if len(valid_vals) > 0:
                        all_values.extend(valid_vals.tolist())
            if len(all_values) > 0:
                color_value_min = min(all_values)
                color_value_max = max(all_values)
            else:
                raise ValueError(f"color_value_column '{color_value_column}' not found or has no valid values")
    
    def _convert_to_qcolor(color):
        """Convert various color formats to QColor"""
        if isinstance(color, str):
            return pg.mkColor(color)
        elif isinstance(color, (tuple, list)):
            if len(color) == 3:
                return pg.mkColor(int(color[0]), int(color[1]), int(color[2]))
            elif len(color) == 4:
                return pg.mkColor(int(color[0]), int(color[1]), int(color[2]), int(color[3]))
            else:
                raise ValueError(f"Color tuple must have 3 (RGB) or 4 (RGBA) elements, got {len(color)}")
        else:
            # Assume it's already a QColor or compatible
            return pg.mkColor(color)
    
    def _get_point_color(x, y, t, trajectory_idx, point_idx, df, valid_mask_idx):
        """Get color for a single point using the configured color system"""
        if color_fn is not None:
            # Use custom color function
            color = color_fn(x, y, t, trajectory_idx, point_idx, df)
            return _convert_to_qcolor(color)
        elif colormap_fn is not None and color_value_column is not None:
            # Use colormap with value column
            if color_value_column in df.columns:
                try:
                    value = df[color_value_column].iloc[valid_mask_idx]
                    if pd.isna(value):
                        value = color_value_min  # Use min for NaN
                except (IndexError, KeyError):
                    value = color_value_min  # Fallback
                
                # Normalize to [0, 1]
                if color_value_min is not None and color_value_max is not None and color_value_max > color_value_min:
                    normalized = (value - color_value_min) / (color_value_max - color_value_min)
                    normalized = max(0.0, min(1.0, normalized))  # Clamp
                else:
                    normalized = 0.5
                
                rgba = colormap_fn(normalized)
                if isinstance(rgba, (tuple, list)) and len(rgba) >= 3:
                    # Convert from [0, 1] to [0, 255] if needed
                    if rgba[0] <= 1.0:
                        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                        a = int(rgba[3] * 255) if len(rgba) > 3 else 255
                    else:
                        r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
                        a = int(rgba[3]) if len(rgba) > 3 else 255
                    return pg.mkColor(r, g, b, a)
                else:
                    return _convert_to_qcolor(rgba)
        
        # Default: per-trajectory colors
        traj_color = pg.intColor(trajectory_idx, hues=num_epochs)
        traj_color.setAlphaF(0.4)
        return traj_color
    
    # White pen for axes outline
    white_pen = pg.mkPen('white', width=1)
    
    if single_plot:
        # Single plot mode: all trajectories on the same axes
        graphics_widget = pg.GraphicsLayoutWidget()
        plot_item = graphics_widget.addPlot()
        plot_items = [plot_item]
        
        # Configure axes for single plot (show labels, enable interaction)
        plot_item.showAxis('bottom')
        plot_item.showAxis('left')
        plot_item.showAxis('top')
        plot_item.showAxis('right')
        plot_item.getAxis('bottom').setPen(white_pen)
        plot_item.getAxis('left').setPen(white_pen)
        plot_item.getAxis('top').setPen(white_pen)
        plot_item.getAxis('right').setPen(white_pen)
        plot_item.setMenuEnabled(True)
        plot_item.setMouseEnabled(True, True)
        
        # Collect all valid coordinates for auto-ranging if needed
        all_x_vals = []
        all_y_vals = []
        
        # Plot all trajectories on the same plot
        for a_linear_index in range(num_epochs):
            pos_df = position_dfs[a_linear_index]
            
            # Extract coordinates
            x_vals = pos_df['x'].to_numpy()
            y_vals = pos_df['y'].to_numpy()
            
            # Get time values if available
            t_vals = pos_df['t'].to_numpy() if 't' in pos_df.columns else None
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_vals = x_vals[valid_mask]
            y_vals = y_vals[valid_mask]
            if t_vals is not None:
                t_vals = t_vals[valid_mask]
            
            if len(x_vals) < 2:
                continue  # Skip trajectories with insufficient points
            
            all_x_vals.extend(x_vals.tolist())
            all_y_vals.extend(y_vals.tolist())
            
            # Generate colors for each point
            brushes = []
            for i in range(len(x_vals)):
                t = t_vals[i] if t_vals is not None else None
                # Find original index in dataframe for color_value_column access
                valid_indices = np.where(valid_mask)[0]
                orig_idx = valid_indices[i] if i < len(valid_indices) else i
                point_color = _get_point_color(x_vals[i], y_vals[i], t, a_linear_index, i, pos_df, orig_idx)
                brushes.append(pg.mkBrush(point_color))
            
            plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brushes)
        
        # Set x/y limits based on maze_extent if provided, otherwise auto-range to fit all trajectories
        if maze_extent is not None:
            xmin, xmax, ymin, ymax = maze_extent
            plot_item.setXRange(xmin, xmax, padding=0)
            plot_item.setYRange(ymin, ymax, padding=0)
        else:
            # Auto-range to fit all trajectories
            if len(all_x_vals) > 0 and len(all_y_vals) > 0:
                plot_item.setXRange(min(all_x_vals), max(all_x_vals), padding=0.05)
                plot_item.setYRange(min(all_y_vals), max(all_y_vals), padding=0.05)
            else:
                plot_item.autoRange()
            view_range = plot_item.viewRange()
            xmin, xmax = view_range[0]
            ymin, ymax = view_range[1]
        
        # Add overlay_mask if provided (only once for single plot)
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
            overlay_img = pg.ImageItem(image=overlay_mask, levels=(0, 1), opacity=0.2)
            pixel_width = (overlay_xmax - overlay_xmin) / overlay_mask.shape[1] if overlay_mask.shape[1] > 0 else 1.0
            pixel_height = (overlay_ymax - overlay_ymin) / overlay_mask.shape[0] if overlay_mask.shape[0] > 0 else 1.0
            overlay_width = overlay_xmax - overlay_xmin
            overlay_height = overlay_ymax - overlay_ymin
            image_bounds_extent = [overlay_xmin - pixel_width/2, overlay_ymin - pixel_height/2, overlay_width + pixel_width, overlay_height + pixel_height]
            overlay_img.setImage(overlay_mask, rect=image_bounds_extent, autoLevels=False)
            plot_item.addItem(overlay_img)
    
    else:
        # Grid mode: each trajectory in its own subplot
        needed_rows = int(np.ceil(num_epochs / fixed_columns))
        linear_plotter_indices = np.arange(num_epochs)
        row_column_indices = np.unravel_index(linear_plotter_indices, (needed_rows, fixed_columns))
        
        # Create GraphicsLayoutWidget for grid of plots
        graphics_widget = pg.GraphicsLayoutWidget()
        plot_items = []
        
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
            
            # Get time values if available
            t_vals = pos_df['t'].to_numpy() if 't' in pos_df.columns else None
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_vals = x_vals[valid_mask]
            y_vals = y_vals[valid_mask]
            if t_vals is not None:
                t_vals = t_vals[valid_mask]
            
            if len(x_vals) < 2:
                continue  # Skip trajectories with insufficient points
            
            # Generate colors for each point
            brushes = []
            for i in range(len(x_vals)):
                t = t_vals[i] if t_vals is not None else None
                # Find original index in dataframe for color_value_column access
                valid_indices = np.where(valid_mask)[0]
                orig_idx = valid_indices[i] if i < len(valid_indices) else i
                point_color = _get_point_color(x_vals[i], y_vals[i], t, a_linear_index, i, pos_df, orig_idx)
                brushes.append(pg.mkBrush(point_color))
            
            plot_item.plot(x_vals, y_vals, pen=None, symbol='o', symbolSize=2, symbolBrush=brushes)
            
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


@function_attributes(short_name=None, tags=['pyqtgraph', 'trajectory', 'colormap'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-20 00:00', related_items=[])
def create_categorical_saturation_fade_color_fn(position_dfs: List[pd.DataFrame], categorical_cmap: str = 'tab10', saturation_start: float = 0.9, saturation_end: float = 0.3):
    """Creates a color function for multi_trajectory_color_plotter that assigns each trajectory a unique color
    from a categorical colormap and fades saturation from high (at start) to low (at end) to show direction.
    
    Args:
        position_dfs: List of position dataframes, each with 'x' and 'y' columns. Optional 't' column for time.
        categorical_cmap: Name of matplotlib categorical colormap (e.g., 'tab10', 'Set3', 'Pastel1'). Default is 'tab10'.
        saturation_start: Starting saturation value (0.0 to 1.0) at trajectory start. Default is 0.9 (high saturation).
        saturation_end: Ending saturation value (0.0 to 1.0) at trajectory end. Default is 0.3 (low saturation).
    
    Returns:
        A color function compatible with multi_trajectory_color_plotter's color_fn parameter.
        The function signature is: (x, y, t, trajectory_idx, point_idx, df) -> QColor or color tuple
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import (
            multi_trajectory_color_plotter, create_categorical_saturation_fade_color_fn
        )
        
        # Create the color function
        color_fn = create_categorical_saturation_fade_color_fn(
            position_dfs=dfs_list,
            categorical_cmap='tab10',
            saturation_start=0.9,
            saturation_end=0.3
        )
        
        # Use it with the plotter
        plot_widget, plot_items = multi_trajectory_color_plotter(
            position_dfs=dfs_list,
            color_fn=color_fn,
            single_plot=True
        )
    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    import numpy as np
    import pandas as pd
    import colorsys
    
    # Validate inputs
    if len(position_dfs) == 0:
        raise ValueError("position_dfs must be a non-empty list")
    
    if not (0.0 <= saturation_start <= 1.0):
        raise ValueError(f"saturation_start must be between 0.0 and 1.0, got {saturation_start}")
    if not (0.0 <= saturation_end <= 1.0):
        raise ValueError(f"saturation_end must be between 0.0 and 1.0, got {saturation_end}")
    
    num_trajectories = len(position_dfs)
    
    # Pre-compute t_start and t_end for each trajectory
    trajectory_t_starts = []
    trajectory_t_ends = []
    has_time_data = []
    
    for df in position_dfs:
        if 't' in df.columns:
            valid_t = df['t'].dropna()
            if len(valid_t) > 0:
                trajectory_t_starts.append(valid_t.min())
                trajectory_t_ends.append(valid_t.max())
                has_time_data.append(True)
            else:
                trajectory_t_starts.append(None)
                trajectory_t_ends.append(None)
                has_time_data.append(False)
        else:
            trajectory_t_starts.append(None)
            trajectory_t_ends.append(None)
            has_time_data.append(False)
    
    # Load categorical colormap and assign base colors to each trajectory
    try:
        import matplotlib.pyplot as plt
        categorical_cmap_obj = plt.get_cmap(categorical_cmap)
    except Exception as e:
        raise ValueError(f"Could not load categorical colormap '{categorical_cmap}': {e}")
    
    # Assign base colors to each trajectory
    # Use evenly spaced indices in the colormap
    if num_trajectories > 1:
        color_indices = np.linspace(0, 1, num_trajectories)
    else:
        color_indices = [0.5]
    
    trajectory_base_colors = []
    for i in range(num_trajectories):
        rgba = categorical_cmap_obj(color_indices[i])  # Returns (R, G, B, A) in [0, 1] range
        # Convert to QColor and store HSV values
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        base_qcolor = pg.mkColor(r, g, b)
        trajectory_base_colors.append(base_qcolor)
    
    # Create the color function closure
    def color_fn(x, y, t, trajectory_idx, point_idx, df):
        """Color function that adjusts saturation based on normalized time within trajectory"""
        # Validate trajectory index
        if trajectory_idx < 0 or trajectory_idx >= num_trajectories:
            # Fallback to default color
            return pg.intColor(0, hues=1)
        
        # Get base color for this trajectory
        base_color = trajectory_base_colors[trajectory_idx]
        
        # Helper function to create color from HSV
        def hsv_to_qcolor(h, s, v, a=1.0):
            """Convert HSV to QColor via RGB"""
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return pg.mkColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        
        # Check if we have time data for this trajectory
        if not has_time_data[trajectory_idx] or t is None or pd.isna(t):
            # No time data: use base color with saturation_start
            h, s, v, a = base_color.getHsvF()
            if h < 0:  # Grayscale, convert RGB to HSV
                r, g, b = base_color.redF(), base_color.greenF(), base_color.blueF()
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return hsv_to_qcolor(h, saturation_start, v, a if a >= 0 else 1.0)
        
        # Get time range for this trajectory
        t_start = trajectory_t_starts[trajectory_idx]
        t_end = trajectory_t_ends[trajectory_idx]
        
        if t_start is None or t_end is None:
            # Fallback: use saturation_start
            h, s, v, a = base_color.getHsvF()
            if h < 0:  # Grayscale, convert RGB to HSV
                r, g, b = base_color.redF(), base_color.greenF(), base_color.blueF()
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return hsv_to_qcolor(h, saturation_start, v, a if a >= 0 else 1.0)
        
        # Normalize time to [0, 1] within this trajectory's range
        if t_end > t_start:
            normalized_time = (t - t_start) / (t_end - t_start)
            normalized_time = max(0.0, min(1.0, normalized_time))  # Clamp to [0, 1]
        else:
            # t_start == t_end: use saturation_start
            normalized_time = 0.0
        
        # Interpolate saturation from saturation_start to saturation_end
        saturation = saturation_start + (saturation_end - saturation_start) * normalized_time
        
        # Get HSV values from base color
        h, s, v, a = base_color.getHsvF()
        
        # If base color has invalid HSV (e.g., grayscale), use RGB to HSV conversion
        if h < 0:  # QColor returns -1 for grayscale colors
            # Convert RGB to HSV using colorsys
            r, g, b = base_color.redF(), base_color.greenF(), base_color.blueF()
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Create adjusted color with new saturation
        return hsv_to_qcolor(h, saturation, v, a if a >= 0 else 1.0)
    
    return color_fn










# ==================================================================================================================================================================================================================================================================================== #
# 2026-01-21 - Vispy                                                                                                                                                                                                                                                                   #
# ==================================================================================================================================================================================================================================================================================== #

@metadata_attributes(short_name=None, tags=['vispy', 'rendering', 'standalone'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21', related_items=[])
@define(slots=False, repr=False, eq=False)
class PredictiveDecodingVispyWidget:
    """Vispy-based widget that renders predictive decoding data (same data as PredictiveDecodingDisplayWidget).

    Keyboard: Left/Right arrow to change epoch. Use init_from_list or init_from_datasource to create;
    render_predictive_decoding_with_vispy is a thin wrapper that returns (main_window, canvas, state).
    """
    # Data / config
    epoch_flat_mask_future_past_result: Optional[List[MatchingPastFuturePositionsResult]] = field(default=None)
    a_decoded_filter_epochs_df: Optional[pd.DataFrame] = field(default=None)
    a_flat_matching_results_list_ds: Optional['MaskDataSource'] = field(default=None)
    curr_position_df: pd.DataFrame = field(default=None)
    pf_decoder: BasePositionDecoder = field(default=None)
    decoded_result: DecodedFilterEpochsResult = field(default=None)
    active_epoch_idx: int = field(default=0)
    current_traj_seconds_pre_post_extension: float = field(default=0.750)
    past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = field(default=(0.4, 1.0))
    start_end_extension_max_opacity: float = field(default=0.4)
    show_full_position_background: bool = field(default=False)
    
    require_angle_match: bool = field(default=False)
    color_matches_by_matching_angle: bool = field(default=False)
    color_matches_by_merged_epoch_t_bin_idx: bool = field(default=True)
    
    enable_debug_plot_trajectory_average_angle_arrows: bool = field(default=False)
    minimum_included_matching_sequence_length: Optional[int] = field(default=None)
    
    max_time_bins_to_show: int = field(default=12)
    enable_table_widgets: bool = field(default=False)


    # Derived in setup()
    xbin: Optional[Any] = field(default=None)
    ybin: Optional[Any] = field(default=None)
    num_epochs: int = field(default=0)
    recording_t_min: float = field(default=0.0)
    recording_t_max: float = field(default=1.0)
    past_future_trajectory_start_extension_seconds: float = field(default=0.0)
    past_future_trajectory_end_extension_seconds: float = field(default=0.0)
    

    # UI / vispy (created in buildUI)
    canvas: Any = field(default=None)
    main_window: Any = field(default=None)
    grid: Any = field(default=None)
    past_view: Any = field(default=None)
    posterior_2d_view: Any = field(default=None)
    future_view: Any = field(default=None)
    time_bin_grid: Any = field(default=None)
    time_bin_views: List[Any] = field(default=Factory(list))
    combined_timeline_view: Any = field(default=None)
    colorbar_view: Any = field(default=None)
    epoch_slider: Any = field(default=None)
    epoch_value_label: Any = field(default=None)
    epoch_table_manager: Any = field(default=None)
    current_epoch_idx: int = field(default=0)
    
    # Mutable visual lists (cleared/repopulated in update_epoch_display)
    past_lines: List[Any] = field(default=Factory(list))
    future_lines: List[Any] = field(default=Factory(list))
    time_bin_images: List[Any] = field(default=Factory(list))
    time_bin_labels: List[Any] = field(default=Factory(list))
    past_mask_contours: List[Any] = field(default=Factory(list))
    posterior_mask_contours: List[Any] = field(default=Factory(list))
    future_mask_contours: List[Any] = field(default=Factory(list))
    colorbar_rects: List[Any] = field(default=Factory(list))
    colorbar_texts: List[Any] = field(default=Factory(list))
    centroid_dots: List[Any] = field(default=Factory(list))
    centroid_arrows: List[Any] = field(default=Factory(list))
    trajectory_debug_arrows: List[Any] = field(default=Factory(list))
    full_position_background_line: List[Any] = field(default=Factory(list))
    timeline_ticks: List[Any] = field(default=Factory(list))
    trajectory_arrows: List[Any] = field(default=Factory(list))
    posterior_img: Any = field(default=None)
    epoch_info_text: Any = field(default=None)
    current_position_line: Any = field(default=None)
    timeline_bar: Any = field(default=None)
    timeline_epoch_rect: Any = field(default=None)
    timeline_epoch_triangle: Any = field(default=None)


    
    

    def __attrs_post_init__(self):
        if self.a_flat_matching_results_list_ds is None and self.epoch_flat_mask_future_past_result is not None and self.a_decoded_filter_epochs_df is not None:
            self.a_flat_matching_results_list_ds = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result=self.epoch_flat_mask_future_past_result, filter_epochs=self.a_decoded_filter_epochs_df)
        self.setup()
        self.buildUI()

    @classmethod
    def init_from_list(cls, epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], a_decoded_filter_epochs_df: pd.DataFrame, curr_position_df: pd.DataFrame, pf_decoder: BasePositionDecoder, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0,
        current_traj_seconds_pre_post_extension: float = 0.750, past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = (0.4, 1.0), start_end_extension_max_opacity: float = 0.4, show_full_position_background: bool = False,
        require_angle_match: bool = False, color_matches_by_matching_angle: bool = False, enable_debug_plot_trajectory_average_angle_arrows: bool = False, minimum_included_matching_sequence_length: Optional[int] = None, **kwargs) -> "PredictiveDecodingVispyWidget":
        a_flat_matching_results_list_ds = MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult(epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result, filter_epochs=a_decoded_filter_epochs_df)
        
        return cls(
            epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result,
            a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
            a_flat_matching_results_list_ds=a_flat_matching_results_list_ds,
            curr_position_df=curr_position_df,
            pf_decoder=pf_decoder,
            decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx,
            current_traj_seconds_pre_post_extension=current_traj_seconds_pre_post_extension,
            past_future_trajectory_extension_seconds=past_future_trajectory_extension_seconds,
            start_end_extension_max_opacity=start_end_extension_max_opacity,
            show_full_position_background=show_full_position_background,
            require_angle_match=require_angle_match,
            color_matches_by_matching_angle=color_matches_by_matching_angle,
            enable_debug_plot_trajectory_average_angle_arrows=enable_debug_plot_trajectory_average_angle_arrows,
            minimum_included_matching_sequence_length=minimum_included_matching_sequence_length,
            **kwargs)

    @classmethod
    def init_from_datasource(cls, datasource: 'MaskDataSource', curr_position_df: pd.DataFrame, pf_decoder: Any, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0, **kwargs) -> "PredictiveDecodingVispyWidget":
        return cls(
            epoch_flat_mask_future_past_result=None,
            a_decoded_filter_epochs_df=datasource.filter_epochs,
            a_flat_matching_results_list_ds=datasource,
            curr_position_df=curr_position_df,
            pf_decoder=pf_decoder,
            decoded_result=decoded_result,
            active_epoch_idx=active_epoch_idx)


    def setup(self):
        if self.pf_decoder is None:
            raise ValueError("pf_decoder must be provided")
        self.xbin = deepcopy(self.pf_decoder.xbin)
        self.ybin = deepcopy(self.pf_decoder.ybin)
        self.num_epochs = len(self.a_flat_matching_results_list_ds.p_x_given_n_list)
        if self.curr_position_df is not None and 't' in self.curr_position_df.columns:
            self.recording_t_min = self.curr_position_df['t'].min()
            self.recording_t_max = self.curr_position_df['t'].max()
        elif self.a_decoded_filter_epochs_df is not None:
            self.recording_t_min = self.a_decoded_filter_epochs_df['start'].min() if 'start' in self.a_decoded_filter_epochs_df.columns else 0.0
            self.recording_t_max = self.a_decoded_filter_epochs_df['stop'].max() if 'stop' in self.a_decoded_filter_epochs_df.columns else 1.0
        ext = self.past_future_trajectory_extension_seconds
        if isinstance(ext, (int, float)):
            self.past_future_trajectory_start_extension_seconds = float(ext)
            self.past_future_trajectory_end_extension_seconds = float(ext)
        elif isinstance(ext, (tuple, list)) and len(ext) == 2:
            self.past_future_trajectory_start_extension_seconds, self.past_future_trajectory_end_extension_seconds = float(ext[0]), float(ext[1])
        elif isinstance(ext, dict):
            self.past_future_trajectory_start_extension_seconds = float(ext.get('start', 0.0))
            self.past_future_trajectory_end_extension_seconds = float(ext.get('end', 0.0))
        else:
            self.past_future_trajectory_start_extension_seconds = 0.0
            self.past_future_trajectory_end_extension_seconds = 0.0


    def get_state(self) -> dict:
        return {
            'num_epochs': self.num_epochs,
            'epoch_slider': self.epoch_slider,
            'epoch_value_label': self.epoch_value_label,
            'update_epoch_display': self.update_epoch_display,
        }

    def as_viewer_tuple(self) -> tuple:
        return (self.main_window, self.canvas, self.get_state())

    def buildUI(self):
        from vispy import app, scene
        from qtpy import QtWidgets, QtCore
        self.current_epoch_idx = self.active_epoch_idx
        canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1920, 1080), title='Predictive Decoding Display - Vispy')
        self.canvas = canvas
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle('Predictive Decoding Display - Vispy')
        self.main_window = main_window
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(canvas.native, stretch=1)
        slider_widget = QtWidgets.QWidget()
        slider_layout = QtWidgets.QHBoxLayout(slider_widget)
        slider_label = QtWidgets.QLabel("Epoch:")
        epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        epoch_slider.setMinimum(0)
        epoch_slider.setMaximum(max(0, self.num_epochs - 1))
        epoch_slider.setValue(min(self.active_epoch_idx, self.num_epochs - 1))
        epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        epoch_slider.setTickInterval(1)
        epoch_value_label = QtWidgets.QLabel(f"{self.active_epoch_idx}/{self.num_epochs}")
        epoch_value_label.setMinimumWidth(60)
        self.epoch_slider = epoch_slider
        self.epoch_value_label = epoch_value_label
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(epoch_slider, stretch=1)
        slider_layout.addWidget(epoch_value_label)
        main_layout.addWidget(slider_widget)
        
        if self.enable_table_widgets:
            from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import TableManager
            table_container = QtWidgets.QWidget()
            table_container.setMaximumHeight(450)
            epoch_table_manager = TableManager(table_container, visible_columns_dict={})
            self.epoch_table_manager = epoch_table_manager
            # filter_epochs = self.a_flat_matching_results_list_ds.filter_epochs if self.a_flat_matching_results_list_ds is not None else self.a_decoded_filter_epochs_df
            
            table_data_sources = {}
            # if (filter_epochs is not None) and len(filter_epochs) > 0:
            #     # idx = min(self.active_epoch_idx, len(filter_epochs) - 1)
            #     # data_sources['active_epoch'] = filter_epochs.iloc[[idx]].copy()
            #     # data_sources['segments'] = filter_epochs.copy()
            #     pass
            if (self.a_flat_matching_results_list_ds is not None):
                a_matching_pos_merged_segment_epochs_df: pd.DataFrame = self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list[self.active_epoch_idx]
                if (a_matching_pos_merged_segment_epochs_df is not None) and len(a_matching_pos_merged_segment_epochs_df) > 0:
                    table_data_sources['curr_merged_segment_epochs'] = a_matching_pos_merged_segment_epochs_df
                                
                a_matching_pos_epochs_df: pd.DataFrame = self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list[self.active_epoch_idx]
                if (a_matching_pos_epochs_df is not None) and len(a_matching_pos_epochs_df) > 0:
                    table_data_sources['curr_merged_pos_epochs'] = a_matching_pos_epochs_df
                    
            # if self.curr_position_df is not None and len(self.curr_position_df) > 0:
            #     data_sources['curr_position'] = self.curr_position_df.copy()
            if table_data_sources:
                visible_columns_dict = {
                    'curr_merged_segment_epochs': ['start', 'stop', 'is_future_present_past', 'epoch_t_idx', 'label', 'duration', 'num_epoch_t_bins', 'is_reversely_replayed', 'pre_merged_epoch_label'],
                    'curr_merged_pos_epochs': ['start', 'stop', 'is_future_present_past', 'label', 'duration'],
                }    
                self.epoch_table_manager.update_tables(table_data_sources, visible_columns_dict=visible_columns_dict)
                
            main_layout.addWidget(table_container)
            

        main_window.setCentralWidget(central_widget)
        main_window.resize(1400, 950)
        main_window.show()
        grid = canvas.central_widget.add_grid()
        self.grid = grid
        self.past_view = grid.add_view(row=0, col=0, col_span=1, row_span=2, border_color='red')
        self.future_view = grid.add_view(row=0, col=2, col_span=1, row_span=2, border_color='blue')
        self.posterior_2d_view = grid.add_view(row=0, col=1, col_span=1, border_color='gray')
        self.time_bin_grid = grid.add_grid(row=1, col=1, col_span=1, border_color='gray')
        self.time_bin_grid.height_max = 120
        self.combined_timeline_view = grid.add_view(row=2, col=0, col_span=3, border_color='gray')
        self.combined_timeline_view.height_max = 40
        self.colorbar_view = grid.add_view(row=3, col=0, col_span=3, border_color='gray')
        self.colorbar_view.height_max = 60

        def on_slider_value_changed(value):
            self.epoch_value_label.setText(f"{value}/{self.num_epochs}")

        def on_slider_released():
            self.update_epoch_display(self.epoch_slider.value())

        epoch_slider.valueChanged.connect(on_slider_value_changed)
        epoch_slider.sliderReleased.connect(on_slider_released)

        def on_key_press(event):
            if event.key == 'Left':
                self.update_epoch_display(self.current_epoch_idx - 1)
            elif event.key == 'Right':
                self.update_epoch_display(self.current_epoch_idx + 1)

        if hasattr(canvas.events, 'key_press'):
            canvas.events.key_press.connect(on_key_press)

        for view in [self.past_view, self.posterior_2d_view, self.future_view]:
            view.camera = scene.PanZoomCamera(aspect=1)
            scene.visuals.GridLines(parent=view.scene)
        self.colorbar_view.camera = scene.PanZoomCamera(aspect=1)
        x_min, x_max = self.xbin[0], self.xbin[-1]
        y_min, y_max = self.ybin[0], self.ybin[-1]
        bbox_vertices = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]], dtype=np.float32)
        for view in [self.past_view, self.posterior_2d_view, self.future_view]:
            scene.visuals.Line(pos=bbox_vertices, color='white', width=2, parent=view.scene)
        extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])
        for view in [self.past_view, self.future_view]:
            view.camera.set_range(x=(extent[0], extent[1]), y=(extent[2], extent[3]))
        y_range = extent[3] - extent[2]
        self.posterior_2d_view.camera.set_range(x=(extent[0], extent[1]), y=(extent[2] - y_range * 0.05, extent[3] + y_range * 0.2))
        timeline_bar_height = 1.0
        recording_duration = self.recording_t_max - self.recording_t_min
        self.combined_timeline_view.camera = scene.PanZoomCamera()
        self.combined_timeline_view.camera.set_range(x=(self.recording_t_min, self.recording_t_max), y=(0, timeline_bar_height))
        self.update_epoch_display(self.active_epoch_idx)

    # ==================================================================================================================================================================================================================================================================================== #
    # Helper/Rendering Functions                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def _clear_epoch_visuals(self):
        self._detach_and_clear_visual_lists(
            [
                'past_lines', 'time_bin_images', 'time_bin_labels', 'future_lines',
                'past_mask_contours', 'posterior_mask_contours', 'future_mask_contours',
                'colorbar_rects', 'colorbar_texts', 'centroid_dots', 'centroid_arrows',
                'trajectory_debug_arrows', 'timeline_ticks',
            ],
            single_ref_attr_names=['posterior_img', 'epoch_info_text', 'timeline_bar', 'timeline_epoch_rect', 'timeline_epoch_triangle'],
        )

    def _time_bin_colors(self, n_bins: int, alpha: float = 0.9) -> np.ndarray:
        """Return (n_bins, 4) float32 array of RGBA colors for time bins (hue cycled)."""
        out = np.zeros((n_bins, 4), dtype=np.float32)
        for t_idx in range(n_bins):
            hue = (t_idx / max(n_bins, 1)) % 1.0
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            out[t_idx] = (rgb[0], rgb[1], rgb[2], alpha)
        return out

    def _segment_row_to_time_bin_idx(self, segment_row_idx: int, epoch_idx: int, mode='centroids') -> Optional[int]:
        """Resolve segment row index to time bin index for the given epoch."""
        if self.epoch_flat_mask_future_past_result is None or epoch_idx >= len(self.epoch_flat_mask_future_past_result):
            return None
        curr_epoch_result: MatchingPastFuturePositionsResult = self.epoch_flat_mask_future_past_result[epoch_idx]
        
        valid_modes_list = ['centroids', 'merged_segments']
        assert mode in valid_modes_list, f'mode: "{mode}" not in valid_modes_list: {valid_modes_list}'
        if mode == 'centroids':
            if curr_epoch_result is None or not hasattr(curr_epoch_result, 'centroids_df') or curr_epoch_result.centroids_df is None:
                return None
            if not hasattr(curr_epoch_result, 'a_centroids_search_segments_df') or curr_epoch_result.a_centroids_search_segments_df is None:
                return None
            search_df = curr_epoch_result.a_centroids_search_segments_df
            if segment_row_idx >= len(search_df):
                return None
            actual_segment_idx = search_df.iloc[segment_row_idx]['segment_idx']
            matching_t_bins = curr_epoch_result.centroids_df[curr_epoch_result.centroids_df['segment_idx'] == actual_segment_idx].index
            
        elif mode == 'merged_segments':
            if curr_epoch_result is None or not hasattr(curr_epoch_result, 'centroids_df') or curr_epoch_result.centroids_df is None:
                return None
            if not hasattr(curr_epoch_result, 'a_centroids_search_segments_df') or curr_epoch_result.a_centroids_search_segments_df is None:
                return None
            search_df = curr_epoch_result.a_centroids_search_segments_df
            if segment_row_idx >= len(search_df):
                return None
            actual_segment_idx = search_df.iloc[segment_row_idx]['segment_idx']
            matching_t_bins = curr_epoch_result.centroids_df[curr_epoch_result.centroids_df['segment_idx'] == actual_segment_idx].index
            pass
        
        else:
            raise NotImplementedError(f'mode: "{mode}" not implemented!')



        match_df: pd.DataFrame = curr_epoch_result.centroids_df
        

        

        return int(matching_t_bins[0]) if len(matching_t_bins) > 0 else None
    


    def _extend_trajectory_xy_opacity(self, x_valid: np.ndarray, y_valid: np.ndarray, opacity: np.ndarray, t_valid: np.ndarray, traj_t_min: float, traj_t_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply start/end trajectory extensions from curr_position_df; return (x_valid, y_valid, opacity)."""
        if self.past_future_trajectory_start_extension_seconds > 0 and self.curr_position_df is not None and 't' in self.curr_position_df.columns:
            ext_start_t = traj_t_min - self.past_future_trajectory_start_extension_seconds
            ext_mask = (self.curr_position_df['t'] >= ext_start_t) & (self.curr_position_df['t'] < traj_t_min)
            ext_positions = self.curr_position_df[ext_mask]
            if (len(ext_positions) > 0) and ('x' in ext_positions.columns) and ('y' in ext_positions.columns) and ('t' in ext_positions.columns):
                ext_x = ext_positions['x'].values
                ext_y = ext_positions['y'].values
                ext_t = ext_positions['t'].values
                
                ext_valid_mask = ~(np.isnan(ext_x) | np.isnan(ext_y) | np.isnan(ext_t))
                if np.any(ext_valid_mask):
                    ext_x_valid = ext_x[ext_valid_mask]
                    ext_y_valid = ext_y[ext_valid_mask]
                    ext_t_valid = ext_t[ext_valid_mask]
                    t_valid = np.concatenate([ext_t_valid, t_valid]) ## important that it updates 't_valid'
                    x_valid = np.concatenate([ext_x_valid, x_valid])
                    y_valid = np.concatenate([ext_y_valid, y_valid])
                    ext_opacity = np.ones(len(ext_x_valid)) * self.start_end_extension_max_opacity
                    opacity = np.concatenate([ext_opacity, opacity])
                    
        if self.past_future_trajectory_end_extension_seconds > 0 and self.curr_position_df is not None and ('t' in self.curr_position_df.columns):
            ext_end_t = traj_t_max + self.past_future_trajectory_end_extension_seconds
            ext_mask = (self.curr_position_df['t'] > traj_t_max) & (self.curr_position_df['t'] <= ext_end_t)
            ext_positions = self.curr_position_df[ext_mask]
            if (len(ext_positions) > 0) and ('x' in ext_positions.columns) and ('y' in ext_positions.columns) and ('t' in ext_positions.columns):
                ext_x = ext_positions['x'].values
                ext_y = ext_positions['y'].values
                ext_t = ext_positions['t'].values
                # ext_valid_mask = ~(np.isnan(ext_x) | np.isnan(ext_y))
                ext_valid_mask = ~(np.isnan(ext_x) | np.isnan(ext_y) | np.isnan(ext_t))
                if np.any(ext_valid_mask):
                    ext_x_valid = ext_x[ext_valid_mask]
                    ext_y_valid = ext_y[ext_valid_mask]
                    ext_t_valid = ext_t[ext_valid_mask]
                    t_valid = np.concatenate([t_valid, ext_t_valid])  ## important that it updates 't_valid'
                    x_valid = np.concatenate([x_valid, ext_x_valid])
                    y_valid = np.concatenate([y_valid, ext_y_valid])
                    ext_opacity = self.start_end_extension_max_opacity * (1.0 - (ext_t_valid - traj_t_max) / self.past_future_trajectory_end_extension_seconds)
                    opacity = np.concatenate([opacity, ext_opacity])
        return t_valid, x_valid, y_valid, opacity

    def _detach_and_clear_visual_lists(self, list_attr_names: Sequence[str], single_ref_attr_names: Optional[Sequence[str]] = None) -> None:
        """Detach visuals from parent and clear list attributes; optionally clear single-ref attributes."""
        for name in list_attr_names:
            lst = getattr(self, name)
            for item in lst:
                if item is not None:
                    item.parent = None
            lst.clear()
        if single_ref_attr_names:
            for name in single_ref_attr_names:
                ref = getattr(self, name)
                if ref is not None:
                    ref.parent = None
                setattr(self, name, None)


    def _render_trajectory_side(self, positions_dict: dict, epoch_anchor_t: Optional[float], default_hue: float, view: Any, lines_list: list, trajectory_colors_and_times_out: list, max_time_distance: float, time_bin_colors: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, new_epoch_idx: int) -> None:
        """Render past or future trajectories into view; append to lines_list and trajectory_colors_and_times_out."""
        from vispy import scene
        from vispy.color import Colormap
        import colorsys
        
        for epoch_id, positions_df in list(positions_dict.items()):
            if self.require_angle_match and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns and not (positions_df['centroid_pos_traj_matching_angle_idx'] >= 0).any():
                continue
            
            custom_cmap: Optional[Colormap] = None
            
            if len(positions_df) > 0 and 'x' in positions_df.columns and 'y' in positions_df.columns:
                x_coords, y_coords = positions_df['x'].values, positions_df['y'].values
                valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
                if np.any(valid_mask):
                    x_valid, y_valid = x_coords[valid_mask], y_coords[valid_mask]
                    if self.color_matches_by_matching_angle and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
                        matching_idx_values = positions_df['centroid_pos_traj_matching_angle_idx'].values
                        valid_match_indices = matching_idx_values[matching_idx_values >= 0]
                        if len(valid_match_indices) > 0:
                            segment_row_idx = int(valid_match_indices[0])
                            matched_t_idx = self._segment_row_to_time_bin_idx(segment_row_idx, new_epoch_idx, mode='centroids')
                            base_rgb = tuple(time_bin_colors[matched_t_idx][:3]) if (matched_t_idx is not None and matched_t_idx < len(time_bin_colors)) else colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                        else:
                            base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                            
                    elif self.color_matches_by_merged_epoch_t_bin_idx and 'matching_found_relevant_pos_epoch' in positions_df.columns:
                        matching_idx_values = positions_df['matching_found_relevant_pos_epoch'].values
                        valid_match_indices = matching_idx_values[matching_idx_values >= 0]
                        valid_rel_match_indices = valid_match_indices - np.nanmin(valid_match_indices) ## get the count of each value
                        n_total_valid_indicies: int = len(valid_rel_match_indices)
                        
                        valid_rel_match_indices_counts = {}
                        valid_rel_match_indices_start_idxs = {}
                        
                        for i, v in enumerate(valid_rel_match_indices):
                            if v not in valid_rel_match_indices_counts:
                                valid_rel_match_indices_counts[v] = 1 ## initialize to 1
                                valid_rel_match_indices_start_idxs[v] = i
                            else:
                                valid_rel_match_indices_counts[v] = valid_rel_match_indices_counts.get(v, 0) + 1 ## increment


                        valid_rel_match_indices_REL_counts = {k:(float(v)/float(n_total_valid_indicies)) for k, v in valid_rel_match_indices_counts.items()}
                        valid_rel_match_indices_REL_start_idxs = {k:(float(v)/float(n_total_valid_indicies-1)) for k, v in valid_rel_match_indices_start_idxs.items()} # -1 to get the last index

                        n_time_bin_colors: int = np.shape(time_bin_colors)[0] #  np.shape(time_bin_colors): (6, 4)
                        unique_valid_rel_match_indices: NDArray = np.unique(valid_rel_match_indices)
                        n_unique_valid_rel_match_indices: int = len(unique_valid_rel_match_indices)
                        print(f'\ttime_bin_colors: {time_bin_colors}')
                        colors_from_NDArray: List[NDArray] = [time_bin_colors[i][:3] for i in np.arange(n_time_bin_colors)]
                        print(f'\tcolors_from_NDArray: {colors_from_NDArray}')
                        controls = None
                        # controls = list(valid_rel_match_indices_REL_start_idxs.values()) ## just the acending counts
                        # print(f'controls: {controls}')
                        # controls = np.interp(np.linspace(0, 1.0, num=n_total_valid_indicies), controls, np.linspace(0, 1.0, num=n_time_bin_colors))
                        # controls = controls[:n_time_bin_colors] + [1.0] ## the last has to be 1.0
                        
                        if controls is not None:
                            print(f'\tcontrols: {controls}, len(controls): {len(controls)}')
                            # assert len(controls) == n_time_bin_colors, f"len(controls): {len(controls)} != n_time_bin_colors: {n_time_bin_colors}"
                            assert len(controls) == (n_time_bin_colors+1), f"len(controls): {len(controls)} != (n_time_bin_colors+1): {(n_time_bin_colors+1)}"
                            custom_cmap = Colormap(colors=colors_from_NDArray, controls=controls, interpolation='zero') # , controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        else:
                            custom_cmap = Colormap(colors=colors_from_NDArray)
                            
                        print(f'\tcustom_cmap: {custom_cmap}')
                        #TODO 2026-02-03 01:37: - [ ] Set controls from the correct values
                        base_rgb = None
                        # assert n_unique_valid_rel_match_indices <= n_time_bin_colors, f"n_unique_valid_rel_match_indices: {n_unique_valid_rel_match_indices}, n_time_bin_colors: {n_time_bin_colors}, unique_valid_rel_match_indices: {unique_valid_rel_match_indices}" 
                        # if len(valid_rel_match_indices) > 0:                            
                        #     # Sample the colormap at each vertex (0 to 1 along the line)
                        #     # t = np.linspace(0.0, 1.0, N)
                        #     # t_coords = positions_df['t'].values[valid_mask]
                        #     # n_points: int = len(x_valid)
                        #     segment_row_idx = int(valid_match_indices[0])
                        #     matched_t_idx = self._segment_row_to_time_bin_idx(segment_row_idx, new_epoch_idx, mode='merged_segments')
                        #     matched_t_idx = valid_match_indices
                        #     # base_rgb = tuple(time_bin_colors[matched_t_idx][:3]) if ((matched_t_idx is not None) and (matched_t_idx < len(time_bin_colors))) else colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                        # else:
                        #     base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)

                    else:
                        base_rgb = colorsys.hsv_to_rgb(default_hue, 0.8, 0.9)
                        

                    if epoch_anchor_t is not None and 't' in positions_df.columns:
                        t_coords = positions_df['t'].values[valid_mask]
                        mean_time = np.mean(t_coords)
                        trajectory_colors_and_times_out.append((colorsys.hsv_to_rgb(default_hue, 0.8, 0.9), mean_time))
                        time_rel = t_coords - epoch_anchor_t
                        time_distance = np.abs(time_rel)
                        opacity = (1.0 - (time_distance / max_time_distance) * 0.8) if max_time_distance > 0 else np.ones(len(x_valid)) * 0.8
                        traj_t_min, traj_t_max = np.min(t_coords), np.max(t_coords)
                        t_valid, x_valid, y_valid, opacity = self._extend_trajectory_xy_opacity(x_valid, y_valid, opacity, t_coords, traj_t_min, traj_t_max)
                    else:
                        opacity = np.ones(len(x_valid)) * 0.8

                    n_points: int = len(x_valid) ## changes after extension
                    
                    colors = np.ones((n_points, 4), dtype=np.float32)
                    if custom_cmap is None:
                        colors[:, 0], colors[:, 1], colors[:, 2] = base_rgb[0], base_rgb[1], base_rgb[2]
                        colors[:, 3] = np.clip(opacity, 0.0, 1.0)
                    else:
                        ## have a valid colormap
                        assert t_valid is not None
                        vertex_colors = np.array(custom_cmap.map(t_valid), dtype=np.float32) # (n_points, 4)
                        Assert.same_shape(vertex_colors, colors)
                        colors[:, :3] = vertex_colors[:, :3]
                        colors[:, 3] = vertex_colors[:, 3]
                        ## overwrite with opacity values
                        colors[:, 3] = np.clip(opacity, 0.0, 1.0)



                    line = scene.visuals.Line(pos=np.column_stack([x_valid, y_valid]), color=colors, width=2, parent=view.scene)
                    line.order = 1
                    line.set_gl_state(blend=True, blend_func=('src_alpha', 'one'))
                    lines_list.append(line)
                    
                    if self.enable_debug_plot_trajectory_average_angle_arrows and 'segment_Vp_deg' in positions_df.columns:
                        segment_angles = positions_df['segment_Vp_deg'].values
                        valid_angles = segment_angles[~np.isnan(segment_angles)]
                        if len(valid_angles) > 0:
                            mean_angle_deg = np.degrees(np.arctan2(np.mean(np.sin(np.radians(valid_angles))), np.mean(np.cos(np.radians(valid_angles)))))
                            mean_angle_rad = np.radians(mean_angle_deg)
                            center_idx = len(x_valid) // 2
                            x_center, y_center = x_valid[center_idx], y_valid[center_idx]
                            data_scale = max(x_max - x_min, y_max - y_min)
                            arrow_head_size = data_scale * 0.04
                            arrow_length = arrow_head_size * 0.5
                            x_end = x_center + arrow_length * np.cos(mean_angle_rad)
                            y_end = y_center + arrow_length * np.sin(mean_angle_rad)
                            debug_arrow = scene.visuals.Arrow(pos=np.array([[x_center, y_center], [x_end, y_end]]), arrows=np.array([[x_center, y_center, x_end, y_end]]), arrow_type='triangle_30', arrow_size=arrow_head_size, color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9), arrow_color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9), width=2.0, method='agg', parent=view.scene)
                            debug_arrow.order = 5
                            self.trajectory_debug_arrows.append(debug_arrow)

    # ==================================================================================================================================================================================================================================================================================== #
    # Main Update Function                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    def update_epoch_display(self, new_epoch_idx: int):
        """Update the display to show a different epoch."""
        if (new_epoch_idx < 0) or (new_epoch_idx >= self.num_epochs):
            return
        from vispy import scene
        import colorsys
        from qtpy.QtWidgets import QApplication
        self.current_epoch_idx = new_epoch_idx
        self.epoch_slider.blockSignals(True)
        self.epoch_slider.setValue(new_epoch_idx)
        self.epoch_slider.blockSignals(False)
        self.epoch_value_label.setText(f"{new_epoch_idx}/{self.num_epochs}")
        self._clear_epoch_visuals() ## clear existing
        
        ## Get the epoch data (this performs the filtering by `minimum_included_matching_sequence_length` if set, etc
        epoch_data = self.a_flat_matching_results_list_ds._prepare_epoch_data(an_epoch_idx=new_epoch_idx, minimum_included_matching_sequence_length=self.minimum_included_matching_sequence_length)
        filter_epochs = self.a_flat_matching_results_list_ds.filter_epochs        
        if new_epoch_idx < len(filter_epochs):
            epoch_row = filter_epochs.iloc[new_epoch_idx]
            epoch_start_t = epoch_row['start'] if 'start' in epoch_row else epoch_row.get('t_start', None)
            epoch_end_t = epoch_row['stop'] if 'stop' in epoch_row else epoch_row.get('t_stop', None)
        else:
            epoch_start_t = None
            epoch_end_t = None
            
        # Get posterior data
        p_x_given_n = self.a_flat_matching_results_list_ds.p_x_given_n_list[new_epoch_idx]
        posterior_2d = np.sum(p_x_given_n, axis=2)
        
        # Generate time bin colors for use in trajectory and centroid coloring
        n_time_bins: int = p_x_given_n.shape[2]
        time_bin_colors = self._time_bin_colors(n_time_bins, alpha=0.9)
        x_min, x_max = self.xbin[0], self.xbin[-1]
        y_min, y_max = self.ybin[0], self.ybin[-1]
        if posterior_2d is None or posterior_2d.size == 0:
            if hasattr(self.a_flat_matching_results_list_ds, 'epoch_high_prob_pos_masks') and self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks is not None and new_epoch_idx < len(self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks):
                mask_2d = self.a_flat_matching_results_list_ds.epoch_high_prob_pos_masks[new_epoch_idx]
                if mask_2d is not None and mask_2d.size > 0:
                    img_height, img_width = mask_2d.T.shape
                else:
                    return
            else:
                return
        else:
            img_height, img_width = posterior_2d.T.shape
        x_scale = (x_max - x_min) / img_width
        y_scale = (y_max - y_min) / img_height
        

        # ==================================================================================================================================================================================================================================================================================== #
        # Common PAST/Future Properties                                                                                                                                                                                                                                                        #
        # ==================================================================================================================================================================================================================================================================================== #
        ## INPUTS: epoch_data (filtered data)
        curr_matching_past_future_positions_df_dict = {k: v for k, v in epoch_data['curr_matching_past_future_positions_df_dict'].items()}
        curr_matching_good_merged_segment_epochs_df = epoch_data.get('curr_matching_good_merged_segment_epochs_df', None)
        if curr_matching_good_merged_segment_epochs_df is not None:
            ## use this 
            curr_matching_good_merged_segment_epochs_df = curr_matching_good_merged_segment_epochs_df.reset_index(drop=True, inplace=False)
            num_good_epochs: int = len(curr_matching_good_merged_segment_epochs_df)
            print(f'curr_matching_good_merged_segment_epochs_df - num_good_epochs: {num_good_epochs}')

            num_good_epochs_past_future = {k:len(v) for k, v in curr_matching_past_future_positions_df_dict.items()}
            num_total_good_epochs_past_future: int = np.sum(list(num_good_epochs_past_future.values()))
            print(f'\tnum_good_epochs_past_future: {num_good_epochs_past_future},\n\tnum_total_good_epochs_past_future: {num_total_good_epochs_past_future}')            
            assert num_good_epochs == num_total_good_epochs_past_future, f'num_total_good_epochs_past_future: {num_total_good_epochs_past_future} != num_good_epochs: {num_good_epochs}'


        all_time_distances = []
        if 'past' in curr_matching_past_future_positions_df_dict and epoch_start_t is not None:
            for epoch_id, positions_df in curr_matching_past_future_positions_df_dict['past'].items():
                if len(positions_df) > 0 and 't' in positions_df.columns:
                    t_coords = positions_df['t'].values
                    valid_mask = ~np.isnan(t_coords)
                    if np.any(valid_mask):
                        time_rel = t_coords[valid_mask] - epoch_start_t
                        all_time_distances.extend(np.abs(time_rel).tolist())
        if 'future' in curr_matching_past_future_positions_df_dict and epoch_end_t is not None:
            for epoch_id, positions_df in curr_matching_past_future_positions_df_dict['future'].items():
                if len(positions_df) > 0 and 't' in positions_df.columns:
                    t_coords = positions_df['t'].values
                    valid_mask = ~np.isnan(t_coords)
                    if np.any(valid_mask):
                        time_rel = t_coords[valid_mask] - epoch_end_t
                        all_time_distances.extend(np.abs(time_rel).tolist())
                        
        max_time_distance = max(all_time_distances) if all_time_distances else 1.0
        print(f'max_time_distance: {max_time_distance}')
        if max_time_distance > 0:
            colorbar_width, colorbar_height, num_segments = 800, 40, 200
            time_range = np.linspace(-max_time_distance, max_time_distance, num_segments)
            segment_width = colorbar_width / num_segments
            for i, time_val in enumerate(time_range):
                time_distance = np.abs(time_val)
                distance_normalized = time_distance / max_time_distance
                opacity = np.clip(1.0 - distance_normalized * 0.8, 0.2, 1.0)
                hue = 0.0 if time_val < 0 else 0.5
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = (rgb[0], rgb[1], rgb[2], opacity)
                x_pos = i * segment_width
                rect = scene.visuals.Rectangle(center=(x_pos + segment_width/2, colorbar_height/2), width=segment_width, height=colorbar_height, color=color, parent=self.colorbar_view.scene)
                self.colorbar_rects.append(rect)
            label_times = [-max_time_distance, -max_time_distance/2, 0, max_time_distance/2, max_time_distance]
            label_positions = np.linspace(0, colorbar_width, len(label_times))
            for time_val, x_pos in zip(label_times, label_positions):
                text = scene.visuals.Text(f'{time_val:.2f}s', pos=(x_pos, colorbar_height + 10), color='white', font_size=10, parent=self.colorbar_view.scene)
                self.colorbar_texts.append(text)
            title_past = scene.visuals.Text('Past (time from start)', pos=(colorbar_width/4, -20), color='white', font_size=12, parent=self.colorbar_view.scene)
            title_future = scene.visuals.Text('Future (time from end)', pos=(3*colorbar_width/4, -20), color='white', font_size=12, parent=self.colorbar_view.scene)
            title_opacity = scene.visuals.Text('Opacity: 1.0 (close) → 0.2 (distant)', pos=(colorbar_width/2, colorbar_height + 25), color='white', font_size=11, parent=self.colorbar_view.scene)
            self.colorbar_texts.extend([title_past, title_future, title_opacity])
            self.colorbar_view.camera = scene.PanZoomCamera(aspect=1)
            self.colorbar_view.camera.set_range(x=(-50, colorbar_width + 50), y=(-50, colorbar_height + 50))
            

        if self.show_full_position_background and self.curr_position_df is not None and 'x' in self.curr_position_df.columns and 'y' in self.curr_position_df.columns:
            bg_x = self.curr_position_df['x'].values
            bg_y = self.curr_position_df['y'].values
            bg_valid_mask = ~(np.isnan(bg_x) | np.isnan(bg_y))
            if np.any(bg_valid_mask):
                bg_x_valid, bg_y_valid = bg_x[bg_valid_mask], bg_y[bg_valid_mask]
                n_bg_points = len(bg_x_valid)
                bg_colors = np.ones((n_bg_points, 4), dtype=np.float32)
                bg_colors[:, :3] = 0.5
                bg_colors[:, 3] = 0.2
                for bg_line in self.full_position_background_line:
                    if bg_line is not None:
                        bg_line.parent = None
                self.full_position_background_line.clear()
                bg_pos = np.column_stack([bg_x_valid, bg_y_valid])
                for view in [self.past_view, self.posterior_2d_view, self.future_view]:
                    line = scene.visuals.Line(pos=bg_pos, color=bg_colors, width=1, method='gl', parent=view.scene)
                    line.order = 0
                    self.full_position_background_line.append(line)
                    



        _common_past_future_render_trajectory_side_kwargs = dict(max_time_distance=max_time_distance, time_bin_colors=time_bin_colors, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, new_epoch_idx=new_epoch_idx)
        ## common outputs: _common_past_future_render_trajectory_side_kwargs


        # ==================================================================================================================================================================================================================================================================================== #
        # LEFT PANE: PAST                                                                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #

        # Render Past Trajectories and collect data for timeline
        past_trajectory_colors_and_times = []
        if 'past' in curr_matching_past_future_positions_df_dict:
            self._render_trajectory_side(positions_dict=curr_matching_past_future_positions_df_dict['past'], epoch_anchor_t=epoch_start_t, default_hue=0.0, view=self.past_view, lines_list=self.past_lines, trajectory_colors_and_times_out=past_trajectory_colors_and_times, **_common_past_future_render_trajectory_side_kwargs)


        # ==================================================================================================================================================================================================================================================================================== #
        # CENTER PANE: CURRENT PBE                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        
        # Render Posterior Heatmap (2D view - top half)
        if posterior_2d is not None and posterior_2d.size > 0:
            self.posterior_img = scene.visuals.Image(posterior_2d.T, cmap='viridis', parent=self.posterior_2d_view.scene)
            self.posterior_img.transform = scene.STTransform(scale=(x_scale, y_scale), translate=(x_min, y_min))
            
        # Render centroid dots and arrows on posterior plot (main view only)
        if self.epoch_flat_mask_future_past_result is not None and new_epoch_idx < len(self.epoch_flat_mask_future_past_result):
            epoch_result = self.epoch_flat_mask_future_past_result[new_epoch_idx]
            if epoch_result is not None and hasattr(epoch_result, 'centroids_df') and epoch_result.centroids_df is not None and 'x' in epoch_result.centroids_df.columns and 'y' in epoch_result.centroids_df.columns and 'segment_idx' in epoch_result.centroids_df.columns:
                centroids_df = epoch_result.centroids_df
                valid_mask = ~(np.isnan(centroids_df['x'].values) | np.isnan(centroids_df['y'].values))
                if np.any(valid_mask):
                    x_pixel = centroids_df['x'].values[valid_mask]
                    y_pixel = centroids_df['y'].values[valid_mask]
                    x_centroids = x_min + x_pixel * x_scale
                    y_centroids = y_min + y_pixel * y_scale
                    original_indices = np.where(valid_mask)[0]
                    n_centroids = len(x_centroids)
                    centroid_colors = np.zeros((n_centroids, 4), dtype=np.float32)
                    for i in range(n_centroids):
                        t_idx = original_indices[i]
                        centroid_colors[i] = time_bin_colors[t_idx] if t_idx < len(time_bin_colors) else (1.0, 1.0, 1.0, 0.8)
                    centroid_pos = np.column_stack([x_centroids, y_centroids])
                    centroid_markers = scene.visuals.Markers(pos=centroid_pos, face_color=centroid_colors, size=8, edge_width=0, parent=self.posterior_2d_view.scene)
                    centroid_markers.order = 7
                    self.centroid_dots.append(centroid_markers)
                    if 'segment_Vp_deg' in centroids_df.columns:
                        segment_Vp_deg = centroids_df['segment_Vp_deg'].values[valid_mask]
                        valid_angle_mask = ~np.isnan(segment_Vp_deg)
                        if np.any(valid_angle_mask):
                            data_scale = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
                            arrow_head_size = data_scale * 0.05
                            arrow_length = arrow_head_size * 0.3
                            angles_rad = np.deg2rad(segment_Vp_deg[valid_angle_mask])
                            x_centroids_valid = x_centroids[valid_angle_mask]
                            y_centroids_valid = y_centroids[valid_angle_mask]
                            arrow_centroid_indices = np.where(valid_angle_mask)[0]
                            for i in range(len(x_centroids_valid)):
                                x_center = x_centroids_valid[i]
                                y_center = y_centroids_valid[i]
                                angle = angles_rad[i]
                                x_start, y_start = x_center, y_center
                                x_end = x_center + arrow_length * np.cos(angle)
                                y_end = y_center + arrow_length * np.sin(angle)
                                centroid_idx = arrow_centroid_indices[i]
                                t_idx = original_indices[centroid_idx]
                                arrow_color = tuple(time_bin_colors[t_idx]) if t_idx < len(time_bin_colors) else (1.0, 1.0, 1.0, 0.8)
                                arrow = scene.visuals.Arrow(pos=np.array([[x_start, y_start], [x_end, y_end]]), arrows=np.array([[x_start, y_start, x_end, y_end]]), arrow_type='triangle_30', arrow_size=arrow_head_size, color=arrow_color, arrow_color=arrow_color, width=3.0, method='agg', parent=self.posterior_2d_view.scene)
                                arrow.order = 7
                                self.centroid_arrows.append(arrow)
        if self.curr_position_df is not None and epoch_start_t is not None and epoch_end_t is not None and 't' in self.curr_position_df.columns and 'x' in self.curr_position_df.columns and 'y' in self.curr_position_df.columns:
            extended_start_t = epoch_start_t - self.current_traj_seconds_pre_post_extension
            extended_end_t = epoch_end_t + self.current_traj_seconds_pre_post_extension
            extended_mask = (self.curr_position_df['t'] >= extended_start_t) & (self.curr_position_df['t'] <= extended_end_t)
            extended_positions = self.curr_position_df[extended_mask]
            if len(extended_positions) > 0:
                x_coords = extended_positions['x'].values
                y_coords = extended_positions['y'].values
                t_coords = extended_positions['t'].values
                valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(t_coords))
                if np.any(valid_mask):
                    x_valid = x_coords[valid_mask]
                    y_valid = y_coords[valid_mask]
                    t_valid = t_coords[valid_mask]
                    within_epoch_mask = (t_valid >= epoch_start_t) & (t_valid <= epoch_end_t)
                    n_points = len(x_valid)
                    colors = np.ones((n_points, 4), dtype=np.float32)
                    colors[:, :3] = 0.7
                    colors[:, 3] = np.where(within_epoch_mask, 1.0, 0.2)
                    if self.current_position_line is None:
                        self.current_position_line = scene.visuals.Line(pos=np.column_stack([x_valid, y_valid]), color=colors, width=3, method='gl', parent=self.posterior_2d_view.scene)
                        self.current_position_line.order = 5
                    else:
                        self.current_position_line.set_data(pos=np.column_stack([x_valid, y_valid]), color=colors)
                else:
                    if self.current_position_line is not None:
                        self.current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
            else:
                if self.current_position_line is not None:
                    self.current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
                for arrow in self.trajectory_arrows:
                    if arrow is not None:
                        arrow.parent = None
                self.trajectory_arrows.clear()
        else:
            if self.current_position_line is not None:
                self.current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
            for arrow in self.trajectory_arrows:
                if arrow is not None:
                    arrow.parent = None
            self.trajectory_arrows.clear()
        if epoch_start_t is not None and epoch_end_t is not None:
            epoch_info_str = f'Epoch {new_epoch_idx + 1}/{self.num_epochs} | start_t: {epoch_start_t:.2f}s | end_t: {epoch_end_t:.2f}s | duration: {epoch_end_t - epoch_start_t:.2f}s'
            text_y_pos = y_max + (y_max - y_min) * 0.15
            text_x_pos = (x_min + x_max) / 2
            self.epoch_info_text = scene.visuals.Text(epoch_info_str, pos=(text_x_pos, text_y_pos), color='white', font_size=14, bold=True, anchor_x='center', anchor_y='bottom', parent=self.posterior_2d_view.scene)
            y_range = y_max - y_min
            self.posterior_2d_view.camera.set_range(x=(x_min, x_max), y=(y_min - y_range * 0.05, y_max + y_range * 0.2))
        if p_x_given_n is not None and p_x_given_n.size > 0:
            n_time_bins = p_x_given_n.shape[2]
            n_bins_to_show = min(n_time_bins, self.max_time_bins_to_show)
            view_time_bin_colors = self._time_bin_colors(n_bins_to_show, alpha=1.0)[:, :3]
            vol_min, vol_max = p_x_given_n.min(), p_x_given_n.max()
            if len(self.time_bin_views) != n_bins_to_show:
                for view in self.time_bin_views:
                    if view is not None and hasattr(view, 'parent'):
                        view.parent = None
                self.time_bin_views.clear()
                for t_idx in range(n_bins_to_show):
                    t_bin_border_color = view_time_bin_colors[t_idx] if t_idx < len(view_time_bin_colors) else (0.5, 0.5, 0.5)
                    view = self.time_bin_grid.add_view(row=0, col=t_idx, border_color=t_bin_border_color)
                    view.camera = scene.PanZoomCamera(aspect=1)
                    view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
                    self.time_bin_views.append(view)
            for t_idx in range(n_bins_to_show):
                slice_2d = p_x_given_n[:, :, t_idx].T.astype(np.float32)
                if vol_max > vol_min:
                    slice_2d = (slice_2d - vol_min) / (vol_max - vol_min)
                view = self.time_bin_views[t_idx]
                slice_img = scene.visuals.Image(slice_2d, cmap='viridis', parent=view.scene)
                img_height, img_width = slice_2d.shape
                scale_x_img = (x_max - x_min) / img_width if img_width > 0 else 1
                scale_y_img = (y_max - y_min) / img_height if img_height > 0 else 1
                slice_img.transform = scene.STTransform(scale=(scale_x_img, scale_y_img), translate=(x_min, y_min))
                self.time_bin_images.append(slice_img)
                label_y_pos = y_max + (y_max - y_min) * 0.08
                label = scene.visuals.Text(f't={t_idx}', pos=((x_min + x_max) / 2, label_y_pos), color='white', font_size=10, anchor_x='center', anchor_y='bottom', parent=view.scene)
                self.time_bin_labels.append(label)
                view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max + (y_max - y_min) * 0.1))
        if self.epoch_flat_mask_future_past_result is not None and new_epoch_idx < len(self.epoch_flat_mask_future_past_result):
            epoch_result_for_contours = self.epoch_flat_mask_future_past_result[new_epoch_idx]
            if epoch_result_for_contours is not None and hasattr(epoch_result_for_contours, 'epoch_t_bins_high_prob_pos_mask') and epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask is not None:
                from skimage import measure
                per_t_bin_mask = epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask
                n_mask_t_bins = per_t_bin_mask.shape[2]
                contour_time_bin_colors = self._time_bin_colors(n_mask_t_bins, alpha=0.7)
                for t_idx in range(n_mask_t_bins):
                    mask_slice = per_t_bin_mask[:, :, t_idx]
                    if np.any(mask_slice):
                        mask_transposed = mask_slice.T.astype(np.float32)
                        contours = measure.find_contours(mask_transposed, level=0.5)
                        contour_color = tuple(contour_time_bin_colors[t_idx])
                        n_y_bins, n_x_bins = mask_transposed.shape
                        for contour in contours:
                            x_world = x_min + (contour[:, 1] / n_x_bins) * (x_max - x_min)
                            y_world = y_min + (contour[:, 0] / n_y_bins) * (y_max - y_min)
                            contour_coords = np.column_stack([x_world, y_world]).astype(np.float32)
                            for view, cont_list in [(self.past_view, self.past_mask_contours), (self.posterior_2d_view, self.posterior_mask_contours), (self.future_view, self.future_mask_contours)]:
                                c = scene.visuals.Line(pos=contour_coords, color=contour_color, width=2, parent=view.scene)
                                c.order = 10
                                cont_list.append(c)
                            if t_idx < len(self.time_bin_views):
                                time_bin_contour = scene.visuals.Line(pos=contour_coords, color=contour_color, width=2, parent=self.time_bin_views[t_idx].scene)
                                time_bin_contour.order = 10
                                self.posterior_mask_contours.append(time_bin_contour)
                                

        # ==================================================================================================================================================================================================================================================================================== #
        # RIGHT PANE: FUTURE                                                                                                                                                                                                                                                                   #
        # ==================================================================================================================================================================================================================================================================================== #
        # Render Future Trajectories and collect data for timeline
        future_trajectory_colors_and_times = []
        if 'future' in curr_matching_past_future_positions_df_dict:
            self._render_trajectory_side(positions_dict=curr_matching_past_future_positions_df_dict['future'], epoch_anchor_t=epoch_end_t, default_hue=0.5, view=self.future_view, lines_list=self.future_lines, trajectory_colors_and_times_out=future_trajectory_colors_and_times, **_common_past_future_render_trajectory_side_kwargs)
            

        # ==================================================================================================================================================================================================================================================================================== #
        # Bottom/Common Panes                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #
        
        # Render Combined Timeline Bar (full width, shows all trajectory ticks and current epoch)
        timeline_bar_height = 1.0
        recording_duration = self.recording_t_max - self.recording_t_min
        if recording_duration > 0:
            bar_fill = scene.visuals.Rectangle(center=((self.recording_t_min + self.recording_t_max) / 2, timeline_bar_height / 2), width=recording_duration, height=timeline_bar_height, color=(0.15, 0.15, 0.15, 1.0), border_color=(0.4, 0.4, 0.4, 1.0), parent=self.combined_timeline_view.scene)
            self.timeline_bar = bar_fill
            if epoch_start_t is not None and epoch_end_t is not None:
                epoch_duration = epoch_end_t - epoch_start_t
                epoch_center_t = (epoch_start_t + epoch_end_t) / 2
                epoch_rect = scene.visuals.Rectangle(center=(epoch_center_t, timeline_bar_height / 2), width=epoch_duration, height=timeline_bar_height, color=(1.0, 1.0, 1.0, 0.3), border_color=(1.0, 1.0, 1.0, 1.0), border_width=2, parent=self.combined_timeline_view.scene)
                self.timeline_epoch_rect = epoch_rect
                triangle_height = timeline_bar_height * 0.35
                triangle_half_width = recording_duration * 0.008
                triangle_top_y = timeline_bar_height + triangle_height * 0.3
                triangle_bottom_y = timeline_bar_height - triangle_height * 0.3
                triangle_vertices = np.array([[epoch_center_t - triangle_half_width, triangle_top_y], [epoch_center_t + triangle_half_width, triangle_top_y], [epoch_center_t, triangle_bottom_y]], dtype=np.float32)
                epoch_triangle = scene.visuals.Polygon(pos=triangle_vertices, color=(1.0, 1.0, 1.0, 0.5), border_color=(1.0, 1.0, 1.0, 1.0), border_width=1, parent=self.combined_timeline_view.scene)
                self.timeline_epoch_triangle = epoch_triangle
            for base_rgb, mean_time in past_trajectory_colors_and_times + future_trajectory_colors_and_times:
                tick_pos = np.array([[mean_time, 0], [mean_time, timeline_bar_height]], dtype=np.float32)
                tick = scene.visuals.Line(pos=tick_pos, color=(base_rgb[0], base_rgb[1], base_rgb[2], 1.0), width=2, parent=self.combined_timeline_view.scene)
                self.timeline_ticks.append(tick)
            self.combined_timeline_view.camera = scene.PanZoomCamera()
            self.combined_timeline_view.camera.set_range(x=(self.recording_t_min, self.recording_t_max), y=(0, timeline_bar_height))
            


        self.canvas.title = f'Predictive Decoding Display - Vispy (Epoch {new_epoch_idx + 1}/{self.num_epochs})'
        self.canvas.update()
        # QApplication.processEvents()
        

            # 'curr_matching_epochs_df': curr_matching_epochs_df,
            # 'curr_matching_positions_df': curr_matching_positions_df,
            # 'curr_matching_epochs_df_dict': curr_matching_epochs_df_dict,
            # 'curr_matching_merged_segment_epochs_df_dict': curr_matching_merged_segment_epochs_df_dict, 
            # 'curr_matching_past_future_positions_df_dict': curr_matching_past_future_positions_df_dict,
            # 'curr_matching_past_future_positions_df_list': curr_matching_past_future_positions_df_list,
        
        if self.enable_table_widgets:
            if (self.epoch_table_manager is not None) and (epoch_data is not None):
                QApplication.processEvents()
                
                try:
                    print(f'trying to update self.epoch_table_manager tables for new_epoch_idx: {new_epoch_idx}...')
                    table_update_sources = {}                    
                    curr_matching_epochs_df = epoch_data.get('curr_matching_epochs_df', None)
                    curr_matching_good_merged_segment_epochs_df = epoch_data.get('curr_matching_good_merged_segment_epochs_df', None)
                
                    if (curr_matching_epochs_df is None):
                        print(f'\tERROR: new_epoch_idx: {new_epoch_idx} curr_matching_epochs_df is None')
                    else:
                        a_matching_pos_merged_segment_epochs_df: pd.DataFrame = curr_matching_epochs_df # self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list[new_epoch_idx]
                        if (a_matching_pos_merged_segment_epochs_df is not None) and (len(a_matching_pos_merged_segment_epochs_df) > 0):
                            table_update_sources['curr_merged_segment_epochs'] = a_matching_pos_merged_segment_epochs_df
                                    
                    if (curr_matching_good_merged_segment_epochs_df is None):
                        print(f'\tERROR: new_epoch_idx: {new_epoch_idx} curr_matching_good_merged_segment_epochs_df is None')
                    else:
                        a_matching_pos_epochs_df: pd.DataFrame = curr_matching_good_merged_segment_epochs_df # self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list[new_epoch_idx]
                        if (a_matching_pos_epochs_df is not None) and len(a_matching_pos_epochs_df) > 0:
                            table_update_sources['curr_merged_pos_epochs'] = a_matching_pos_epochs_df
                            
                    if table_update_sources:
                        visible_columns_dict = {
                            'curr_merged_segment_epochs': ['start', 'stop', 'is_future_present_past', 'epoch_t_idx', 'label', 'duration', 'num_epoch_t_bins', 'is_reversely_replayed', 'pre_merged_epoch_label'],
                            'curr_merged_pos_epochs': ['start', 'stop', 'is_future_present_past', 'label', 'duration'],
                        }    
                        self.epoch_table_manager.update_tables(table_update_sources, visible_columns_dict=visible_columns_dict)
                    else:
                        print(f'\tWARN: no table_update_sources (empty)')

                except Exception as e:
                    print(f'\tERROR: encountered exception {e} while trying to update table widgets for new_epoch_idx: {new_epoch_idx}!')
                    # raise e
                    pass


            ## OLDER: from unfiltered datsources
            # if (self.epoch_table_manager is not None) and (self.a_flat_matching_results_list_ds is not None):
            #     try:
            #         print(f'trying to update self.epoch_table_manager tables for new_epoch_idx: {new_epoch_idx}...')
            #         table_update_sources = {}                    
            #         if new_epoch_idx >= len(self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list):
            #             print(f'\tERROR: new_epoch_idx: {new_epoch_idx} >= len(self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list): {len(self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list)}')
            #         else:
            #             a_matching_pos_merged_segment_epochs_df: pd.DataFrame = self.a_flat_matching_results_list_ds.matching_pos_merged_segment_epochs_dfs_list[new_epoch_idx]
            #             if (a_matching_pos_merged_segment_epochs_df is not None) and (len(a_matching_pos_merged_segment_epochs_df) > 0):
            #                 table_update_sources['curr_merged_segment_epochs'] = a_matching_pos_merged_segment_epochs_df
                                    
            #         if new_epoch_idx >= len(self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list):
            #             print(f'\tERROR: new_epoch_idx: {new_epoch_idx} >= len(self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list): {len(self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list)}')
            #         else:
            #             a_matching_pos_epochs_df: pd.DataFrame = self.a_flat_matching_results_list_ds.matching_pos_epochs_dfs_list[new_epoch_idx]
            #             if (a_matching_pos_epochs_df is not None) and len(a_matching_pos_epochs_df) > 0:
            #                 table_update_sources['curr_merged_pos_epochs'] = a_matching_pos_epochs_df
                            
            #         if table_update_sources:
            #             visible_columns_dict = {
            #                 'curr_merged_segment_epochs': ['start', 'stop', 'is_future_present_past', 'epoch_t_idx', 'label', 'duration', 'num_epoch_t_bins', 'is_reversely_replayed', 'pre_merged_epoch_label'],
            #                 'curr_merged_pos_epochs': ['start', 'stop', 'is_future_present_past', 'label', 'duration'],
            #             }    
            #             self.epoch_table_manager.update_tables(table_update_sources, visible_columns_dict=visible_columns_dict)
            #         else:
            #             print(f'\tWARN: no table_update_sources (empty)')

            #     except Exception as e:
            #         print(f'\tERROR: encountered exception {e} while trying to update table widgets for new_epoch_idx: {new_epoch_idx}!')
            #         # raise e
            #         pass
                            

        # self.canvas.title = f'Predictive Decoding Display - Vispy (Epoch {new_epoch_idx + 1}/{self.num_epochs})'
        # self.canvas.update()
        QApplication.processEvents()


@function_attributes(short_name=None, tags=['vispy', 'rendering', 'standalone'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-21', related_items=[])
def render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult], a_decoded_filter_epochs_df: pd.DataFrame, curr_position_df: pd.DataFrame, pf_decoder: BasePositionDecoder, decoded_result: DecodedFilterEpochsResult, active_epoch_idx: int = 0,
    current_traj_seconds_pre_post_extension: float = 0.750, 
    past_future_trajectory_extension_seconds: Union[float, Tuple[float, float]] = (0.4, 1.0), 
    start_end_extension_max_opacity: float = 0.4, show_full_position_background: bool = False, 
    require_angle_match: bool = False, color_matches_by_matching_angle: bool=False, enable_debug_plot_trajectory_average_angle_arrows: bool=False,
    minimum_included_matching_sequence_length: Optional[int] = None,
    **kwargs) -> PredictiveDecodingVispyWidget:
    """Standalone function that renders predictive decoding data using vispy instead of the widget.
    
    Takes the same inputs as PredictiveDecodingDisplayWidget.init_from_datasource but uses vispy
    to render all the data in an interactive visualization.
    
    Keyboard controls:
        Left Arrow: Navigate to previous epoch
        Right Arrow: Navigate to next epoch
    
    Args:
        epoch_flat_mask_future_past_result: List of MatchingPastFuturePositionsResult objects
        a_decoded_filter_epochs_df: DataFrame with filter epochs
        curr_position_df: DataFrame with current position data
        pf_decoder: BasePositionDecoder instance
        decoded_result: DecodedFilterEpochsResult instance
        active_epoch_idx: Initial epoch index to display (default: 0)
        current_traj_seconds_pre_post_extension: Seconds to extend current epoch trajectory before/after epoch bounds (default: 0.750). Positions outside epoch are rendered with 0.2 alpha.
        past_future_trajectory_extension_seconds: Seconds to extend past/future trajectories beyond their computed bounds (default: 0.0).
            Can be a single float (applies to both start and end) or a tuple (start_extension, end_extension).
            Start extensions are rendered with solid start_end_extension_max_opacity. End extensions fade out from 
            start_end_extension_max_opacity to 0.0, visually indicating trajectory direction.
        start_end_extension_max_opacity: Maximum opacity for trajectory extensions (default: 0.2). Start extensions use this as solid opacity,
            end extensions fade from this value to 0.0.
        show_full_position_background: If True, renders the entire position dataframe as a faint (0.2 alpha) grey line behind past/future trajectories (default: False).
        require_angle_match: If True, only display trajectories whose direction aligns with the decoded posterior centroid direction (centroid_pos_traj_matching_angle_idx >= 0). Default: False.
        color_matches_by_matching_angle: If True, trajectories that have a valid angle match (centroid_pos_traj_matching_angle_idx >= 0) 
            will be colored using the corresponding time bin's color instead of the default red (past) or cyan (future). Default: True.
        enable_debug_plot_trajectory_average_angle_arrows: If True, draws small arrows at the temporal center of each 
            past/future trajectory indicating the trajectory's representative direction (circular mean of segment_Vp_deg). Default: True.
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (main_window, canvas, state) for compatibility with export_vispy_viewer_epochs.

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import render_predictive_decoding_with_vispy


        viewer = render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result, a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
                                                        curr_position_df=container.decoding_locality.pos_df, pf_decoder=a_decoder, decoded_result=a_decoded_result)

    Implemented via PredictiveDecodingVispyWidget.init_from_list(...); returns (main_window, canvas, state) for compatibility.
    """
    widget = PredictiveDecodingVispyWidget.init_from_list(
        epoch_flat_mask_future_past_result=epoch_flat_mask_future_past_result,
        a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
        curr_position_df=curr_position_df,
        pf_decoder=pf_decoder,
        decoded_result=decoded_result,
        active_epoch_idx=active_epoch_idx,
        current_traj_seconds_pre_post_extension=current_traj_seconds_pre_post_extension,
        past_future_trajectory_extension_seconds=past_future_trajectory_extension_seconds,
        start_end_extension_max_opacity=start_end_extension_max_opacity,
        show_full_position_background=show_full_position_background,
        require_angle_match=require_angle_match,
        color_matches_by_matching_angle=color_matches_by_matching_angle,
        enable_debug_plot_trajectory_average_angle_arrows=enable_debug_plot_trajectory_average_angle_arrows,
        minimum_included_matching_sequence_length=minimum_included_matching_sequence_length,
        **kwargs)
    return widget # widget.as_viewer_tuple()




@function_attributes(short_name=None, tags=['vispy', 'export', 'screenshot', 'high-resolution'], input_requires=[], output_provides=[], uses=['render_predictive_decoding_with_vispy'], used_by=[], creation_date='2026-01-22', related_items=['render_predictive_decoding_with_vispy'])
def export_vispy_viewer_epochs(viewer_tuple: tuple, export_folder: Union[str, Path], resolution_scale: float = 1.0, export_individual_views: bool = False, epoch_indices: Optional[List[int]] = None, delay_between_epochs: float = 0.15, progress_print: bool = True) -> List[Path]:
    """Export high-resolution renderings of all epoch views from the vispy predictive decoding viewer.
    
    Programmatically iterates through epoch indices, updates the display, and exports high-resolution 
    screenshots of all displayed views to an export folder.
    
    Args:
        viewer_tuple: Tuple (main_window, canvas, state) from render_predictive_decoding_with_vispy, or a PredictiveDecodingVispyWidget instance.
        export_folder: Path to folder where images will be saved. Created if it doesn't exist.
        resolution_scale: Scale factor for high-res rendering (default: 2.0). Higher values produce larger images.
        export_individual_views: If True, export individual views (past, posterior, future) separately (default: False)
        epoch_indices: Optional list of specific epoch indices to export. If None, exports all epochs.
        delay_between_epochs: Delay in seconds after updating epoch display to allow rendering to stabilize (default: 0.15)
        progress_print: If True, print progress messages during export (default: True)
        
    Returns:
        List of Path objects for all exported image files
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import render_predictive_decoding_with_vispy, export_vispy_viewer_epochs
        
        viewer = render_predictive_decoding_with_vispy(epoch_flat_mask_future_past_result=_out_epoch_flat_mask_future_past_result, 
                                                        a_decoded_filter_epochs_df=a_decoded_filter_epochs_df,
                                                        curr_position_df=container.decoding_locality.pos_df, 
                                                        pf_decoder=a_decoder, decoded_result=a_decoded_result)
        
        exported_files = export_vispy_viewer_epochs(viewer, export_folder='./exports', resolution_scale=2.0)
    """
    import time
    from pathlib import Path
    from qtpy.QtWidgets import QApplication
    
    # Try to import imageio for saving, fall back to PIL if not available
    try:
        from imageio import imwrite as save_image
        _use_imageio = True
    except ImportError:
        from PIL import Image
        _use_imageio = False
        def save_image(path, img_array):
            """Save image array using PIL."""
            Image.fromarray(img_array).save(path)
    
    # Accept either (main_window, canvas, state) tuple or PredictiveDecodingVispyWidget instance
    if hasattr(viewer_tuple, 'as_viewer_tuple'):
        viewer_tuple = viewer_tuple.as_viewer_tuple()
    main_window, canvas, state = viewer_tuple
    
    # Validate and create export folder
    export_folder = Path(export_folder)
    export_folder.mkdir(parents=True, exist_ok=True)
    
    # Get number of epochs and determine which indices to export
    num_epochs = state['num_epochs']
    if epoch_indices is None:
        epoch_indices = list(range(num_epochs))
    else:
        # Validate provided epoch indices
        epoch_indices = [idx for idx in epoch_indices if 0 <= idx < num_epochs]
    
    if len(epoch_indices) == 0:
        print("Warning: No valid epoch indices to export.")
        return []
    
    # Get the update function from state
    update_epoch_display = state.get('update_epoch_display')
    if update_epoch_display is None:
        raise ValueError("update_epoch_display function not found in state. Ensure you're using an updated version of render_predictive_decoding_with_vispy.")
    
    # Get canvas size for high-res rendering
    canvas_width, canvas_height = canvas.size
    high_res_width = int(canvas_width * resolution_scale)
    high_res_height = int(canvas_height * resolution_scale)
    
    exported_files = []
    total_epochs = len(epoch_indices)
    
    if progress_print:
        print(f"Exporting {total_epochs} epochs to {export_folder} at {resolution_scale}x resolution ({high_res_width}x{high_res_height})...")
    
    for i, epoch_idx in enumerate(epoch_indices):
        try:
            if progress_print:
                print(f"  Exporting epoch {epoch_idx + 1}/{num_epochs} ({i + 1}/{total_epochs})...", end='', flush=True)
            
            # Update the epoch display programmatically
            # Block slider signals to avoid recursive updates
            state['epoch_slider'].blockSignals(True)
            state['epoch_slider'].setValue(epoch_idx)
            state['epoch_slider'].blockSignals(False)
            
            # Call the update function directly
            update_epoch_display(epoch_idx)
            
            # Process Qt events to ensure rendering completes
            QApplication.processEvents()
            
            # Small delay to allow rendering to stabilize
            time.sleep(delay_between_epochs)
            
            # Process events again after delay
            QApplication.processEvents()
            
            # Ensure canvas is updated
            canvas.update()
            QApplication.processEvents()
            
            # Render high-resolution screenshot
            # vispy's render() returns RGBA numpy array with shape (height, width, 4)
            img_array = canvas.render(size=(high_res_width, high_res_height))
            
            # Convert RGBA to RGB by dropping alpha channel (optional, keeps file size smaller)
            img_rgb = img_array[:, :, :3]
            
            # Flip vertically if needed (vispy may return origin at bottom-left)
            # Check if image appears upside down and flip
            img_rgb = np.flipud(img_rgb)
            
            # Save full canvas screenshot
            full_filename = f"epoch_{epoch_idx:04d}_full.png"
            full_path = export_folder / full_filename
            save_image(str(full_path), img_rgb)
            exported_files.append(full_path)
            
            if progress_print:
                print(f" saved to {full_filename}")
            
            # Export individual views if requested
            if export_individual_views:
                # Note: Individual view export requires rendering each view separately
                # This is more complex and may require accessing view.scene directly
                # For now, we export the full canvas; individual view export can be added later
                pass
                
        except Exception as e:
            if progress_print:
                print(f" ERROR: {e}")
            # Continue with next epoch even if this one fails
            continue
    
    if progress_print:
        print(f"Export complete. {len(exported_files)} files saved to {export_folder}")
    
    return exported_files

