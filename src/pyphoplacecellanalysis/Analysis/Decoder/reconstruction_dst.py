import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field, serialized_field
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

# Import the base decoder from your pyphoplacecellanalysis repo
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder


@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-03 11:04', related_items=[])
@custom_define(slots=False, eq=False)
class BayesianPlacemapPositionDecoderDST(BayesianPlacemapPositionDecoder):
    """
    Dempster-Shafer Theory (DST) updated Position Decoder.
    Mirrors pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
    but implements Shafer Discounting of conflicting place cell likelihoods.
    
    Reliability (R_i) is calculated dynamically for each cell based on its 
    Spatial Signal-to-Noise Ratio (in-field vs. out-of-field expected firing rates).

    Parameters
    ----------
    time_bin_size : float
        The decoding time bin size.
    pf : PfND
        The underlying placefield object (ratemaps come from `pf.ratemap.tuning_curves`).
    spikes_df : pd.DataFrame
        Spikes used for time-binned decoding.
    field_threshold_frac : float
        Fraction of peak firing rate defining in-field vs out-of-field masks (default: 0.20).
    discount_silence : bool
        If True, applies Shafer Discounting when the cell did NOT fire (n_i = 0). (default: False).
    n_top_peaks : int
        Number of top prominence peaks used when building in-field masks (default: 3).
    slice_level_multiplier : float
        Prominence contour level multiplier for in-field masks (default: 0.20).
    fn_tn_mode : str
        FN/TN accumulation mode: ``'occupancy_seconds'`` or ``'occupied_bins'`` (default: ``'occupancy_seconds'``).

    Usage:
        from copy import deepcopy
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction_dst import BayesianPlacemapPositionDecoderDST

        a_dst_decoder2D: BayesianPlacemapPositionDecoderDST = BayesianPlacemapPositionDecoderDST(
            time_bin_size=pf2D_Decoder.time_bin_size, pf=pf2D_Decoder.pf, spikes_df=deepcopy(pf2D_Decoder.spikes_df),
        )
        a_dst_decoder2D
    """
    ## New `BayesianPlacemapPositionDecoderDST`-specific fields:
    t_bin_aclus_reliability_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})
    per_tbin_aclu_spike_counts_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_t_bins','n_neurons',)})
    time_bin_info_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_t_bins',)})
    per_tbin_aclu_spike_counts_sparse: csr_matrix = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons','n_t_bins',)}) # (n_aclus, n_t_bins) - (25, 1427042)

    field_threshold_frac: float = serialized_field(default=0.20)
    discount_silence: bool = non_serialized_field(default=False)
    n_top_peaks: int = serialized_field(default=3)
    slice_level_multiplier: float = serialized_field(default=0.20)
    fn_tn_mode: str = serialized_field(default='occupancy_seconds')
    reliability_active: Optional[np.ndarray] = non_serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})
    reliability_silent: Optional[np.ndarray] = non_serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})


    @property
    def expected_n_spikes(self):
        """The expected_n_spikes property."""
        return self.ratemap.tuning_curves * self.time_bin_size # shape: (n_aclus, n_xbins, n_ybins) for 2D — same as tuning_curves


    @property
    def ratemaps(self):
        """Alias for `self.ratemap.tuning_curves` used by DST posterior computation."""
        return self.ratemap.tuning_curves


    # ==================================================================================================================== #
    # Initialization                                                                                                       #
    # ==================================================================================================================== #
    @classmethod
    def serialized_key_allowlist(cls):
        return BayesianPlacemapPositionDecoder.serialized_key_allowlist() + ['field_threshold_frac', 'n_top_peaks', 'slice_level_multiplier', 'fn_tn_mode']


    @classmethod
    def from_dict(cls, val_dict):
        return cls(time_bin_size=val_dict.get('time_bin_size', 0.25), pf=val_dict.get('pf', None), spikes_df=val_dict.get('spikes_df', None), field_threshold_frac=val_dict.get('field_threshold_frac', 0.20), discount_silence=val_dict.get('discount_silence', False), n_top_peaks=val_dict.get('n_top_peaks', 3), slice_level_multiplier=val_dict.get('slice_level_multiplier', 0.20), fn_tn_mode=val_dict.get('fn_tn_mode', 'occupancy_seconds'), setup_on_init=val_dict.get('setup_on_init', True), post_load_on_init=val_dict.get('post_load_on_init', False), debug_print=val_dict.get('debug_print', False))


    @classmethod
    def init_from_stateful_decoder(cls, stateful_decoder: "BayesianPlacemapPositionDecoder", field_threshold_frac: float = 0.20, discount_silence: bool = False, **kwargs):
        """Creates a new DST decoder instance from an existing stateful Bayesian decoder."""
        return cls(time_bin_size=stateful_decoder.time_bin_size, pf=deepcopy(stateful_decoder.pf), spikes_df=deepcopy(stateful_decoder.spikes_df), field_threshold_frac=field_threshold_frac, discount_silence=discount_silence, debug_print=stateful_decoder.debug_print, **kwargs)


    @classmethod
    def init_from_placefields(cls, pf: PfND, time_bin_size: float, spikes_df: pd.DataFrame, field_threshold_frac: float = 0.20, discount_silence: bool = False, debug_print: bool = False, **kwargs):
        """Creates a new DST decoder instance from a placefields object plus required decoder inputs."""
        return cls(time_bin_size=time_bin_size, pf=deepcopy(pf), spikes_df=deepcopy(spikes_df), field_threshold_frac=field_threshold_frac, discount_silence=discount_silence, debug_print=debug_print, **kwargs)


    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        super().post_load()
        self.reliability_active = None
        self.reliability_silent = None


    def setup(self):
        super().setup()
        self.t_bin_aclus_reliability_df = None
        self.per_tbin_aclu_spike_counts_df = None
        self.time_bin_info_df = None
        self.per_tbin_aclu_spike_counts_sparse = None
        self.reliability_active = None
        self.reliability_silent = None


    # ==================================================================================================================================================================================================================================================================================== #
    # Main Methods                                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    def compute_reliability_new(self, active_peak_prominence_2d_results, spikes_df: Optional[pd.DataFrame] = None, time_bin_size_seconds: Optional[float] = None, max_t_idx: Optional[int] = None, **kwargs):
        """Compute per-aclu reliability via CellIndividualReliabilityMatrix and store results on self.

        Parameters
        ----------
        active_peak_prominence_2d_results : PeakProminence2D results (required for in-field masks).
        spikes_df : optional spikes override; defaults to `self.spikes_df` sliced to `self.neuron_IDs`.
        time_bin_size_seconds : temporal bin width; defaults to `self.time_bin_size`.
        max_t_idx : optional cap on number of time bins (None = all).

        Uses instance fields ``n_top_peaks``, ``slice_level_multiplier``, and ``fn_tn_mode``.

        Returns
        -------
        t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse
        """
        pfs = self.pf
        ratemaps = self.ratemap
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else ratemaps.neuron_ids)
        if spikes_df is None:
            spikes_df = deepcopy(self.spikes_df)
        spikes_df = spikes_df.spikes.sliced_by_neuron_id(neuron_ids)
        if time_bin_size_seconds is None:
            time_bin_size_seconds = self.time_bin_size

        if self.spikes_df is None:
            self.spikes_df = spikes_df

        if self.time_bin_size is None:
            self.time_bin_size = time_bin_size_seconds

        _fake_reliability_df, in_field_masks = CellIndividualReliabilityMatrix._partial_compute_reliability_matrix(
            spikes_df=spikes_df, active_peak_prominence_2d_results=active_peak_prominence_2d_results, ratemaps=ratemaps,
            n_top_peaks=self.n_top_peaks, slice_level_multiplier=self.slice_level_multiplier, fn_tn_mode=self.fn_tn_mode,
        )

        self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse = CellIndividualReliabilityMatrix.compute_reliability_matrix(
            spikes_df=spikes_df, ratemaps=ratemaps, pfs=pfs, in_field_masks=in_field_masks, neuron_ids=neuron_ids,
            time_bin_size_seconds=time_bin_size_seconds, max_t_idx=max_t_idx, **kwargs,
        )

        return self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse



    def _compute_reliability_metrics(self, ratemaps_flat):
        """
        Calculates the in-field vs out-of-field Spatial SNR (R_i) for all cells.
        Expects ratemaps flattened to (nCells, nFlatPositionBins).
        """
        nCells, nPositionBins = ratemaps_flat.shape
        R_active = np.ones(nCells)
        R_silent = np.ones(nCells)

        for i in range(nCells):
            rm = ratemaps_flat[i, :]
            max_rate = np.nanmax(rm)

            # Handle cells that are silent everywhere or have invalid rates
            if max_rate <= 0 or np.isnan(max_rate):
                R_active[i] = 0.0 
                R_silent[i] = 0.0
                continue

            # Step A: Create Spatial Masks
            theta = self.field_threshold_frac * max_rate
            in_field_mask = (rm >= theta)
            out_field_mask = ~in_field_mask

            # Step B: Calculate Mean Regional Rates
            mu_in = np.nanmean(rm[in_field_mask]) if np.any(in_field_mask) else 0.0
            mu_out = np.nanmean(rm[out_field_mask]) if np.any(out_field_mask) else 0.0

            # Step C: Define Spatial Precision (R_i)
            if (mu_in + mu_out) > 0:
                R_active[i] = mu_in / (mu_in + mu_out)
            else:
                R_active[i] = 0.0
                
            # Map NPV for silence. Defaults to 1.0 (no discounting) if disabled.
            if self.discount_silence:
                R_silent[i] = R_active[i] 

        self.reliability_active = R_active
        self.reliability_silent = R_silent


    def compute_posterior(self, spkcount, ratemaps=None):
        """
        Overrides the standard likelihood combination to inject Shafer Discounting.
        Handles both 1D and 2D ratemaps natively.
        
        spkcount : (nCells, nTimeBins)
        ratemaps : (nCells, nX, nY) or (nCells, nPositionBins)
        """
        if ratemaps is None:
            ratemaps = self.ratemaps

        # 1. Dynamically handle 1D vs 2D spatial layouts
        original_shape = ratemaps.shape
        nCells = original_shape[0]
        spatial_shape = original_shape[1:] 
        nPositionBins = np.prod(spatial_shape)
        
        ratemaps_flat = ratemaps.reshape(nCells, nPositionBins)
        
        # 2. Ensure spatial SNR metrics are prepared
        if self.reliability_active is None:
            self._compute_reliability_metrics(ratemaps_flat)
            
        tau = self.time_bin_size
        nTimeBins = spkcount.shape[1]
        
        # We accumulate log-evidence to prevent float underflow and save RAM
        log_posterior = np.zeros((nTimeBins, nPositionBins), dtype=np.float64)
        
        # 3. Iterative Likelihood Evaluation (Memory Efficient)
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][:, np.newaxis]   # (nTimeBins, 1)
            cell_ratemap = ratemaps_flat[cell, :][np.newaxis, :]  # (1, nPositionBins)

            # Poisson Likelihood Density (ignoring constant 1/n! term)
            L_i = ( (tau * cell_ratemap) ** cell_spkcnt ) * np.exp(-tau * cell_ratemap)
            Z_i = np.sum(L_i, axis=1, keepdims=True)
            
            # Convert raw likelihoods to specific probability density mappings (p_i)
            with np.errstate(divide='ignore', invalid='ignore'):
                p_i = L_i / Z_i
            p_i = np.where(Z_i == 0, 1.0 / nPositionBins, p_i)

            # Apply Reliability Conditional on Firing State
            active_mask = (cell_spkcnt > 0)
            R_effective = np.where(active_mask, self.reliability_active[cell], self.reliability_silent[cell])
            
            # Shafer Discounting Rule: E_i(x) = R_i * p_i(x|n_i) + (1 - R_i) * (1 / |Theta|)
            E_i = (R_effective * p_i) + ((1.0 - R_effective) / nPositionBins)
            
            # Dempster's Rule of Combination (Summing Log Evidences)
            log_posterior += np.log(E_i + 1e-15)

        # 4. Convert back to linear probability space (Log-Sum-Exp Trick)
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_posterior_max)
        
        # Final Global Normalization
        sum_post = np.sum(posterior, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            posterior /= sum_post
            
        posterior = np.where(sum_post == 0, 1.0 / nPositionBins, posterior)

        # 5. Reshape to match pyphoplacecellanalysis expectations: (*Spatial_Shape, nTimeBins)
        posterior = posterior.T # (nPositionBins, nTimeBins)
        final_shape = (*spatial_shape, nTimeBins)
        
        return posterior.reshape(final_shape)

