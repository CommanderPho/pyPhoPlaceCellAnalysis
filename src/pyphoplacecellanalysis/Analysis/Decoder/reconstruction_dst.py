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
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.reliability import CellIndividualReliabilityMatrix

# Import the base decoder from your pyphoplacecellanalysis repo
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder


@metadata_attributes(short_name=None, tags=['Dempster-Shafer', 'decoder', 'position-decoder', 'probability'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-03 11:04', related_items=[])
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

        ## Build from an existing Bayesian 2D decoder (inherits parent setup / spike binning):
        a_dst_decoder2D: BayesianPlacemapPositionDecoderDST = BayesianPlacemapPositionDecoderDST(
            time_bin_size=pf2D_Decoder.time_bin_size, pf=pf2D_Decoder.pf, spikes_df=deepcopy(pf2D_Decoder.spikes_df),
        )
        # Or: a_dst_decoder2D = BayesianPlacemapPositionDecoderDST.init_from_stateful_decoder(pf2D_Decoder)

        ## Optional: confusion-matrix reliability + sparse spike counts (not required for decode; Skaggs is computed lazily):
        # a_dst_decoder2D.compute_unit_confusion_reliability_variables(spikes_df=spikes_df, time_bin_size_seconds=a_dst_decoder2D.time_bin_size)
        # # Or pass pipeline PeakProminence2D if already computed:
        # # a_dst_decoder2D.compute_unit_confusion_reliability_variables(active_peak_prominence_2d_results=..., spikes_df=spikes_df, ...)

        ## Decode all time bins (DST Bel({v}) via overridden decode → compute_posterior):
        a_dst_decoder2D.compute_all(debug_print=False)
        # → a_dst_decoder2D.p_x_given_n, a_dst_decoder2D.most_likely_positions, a_dst_decoder2D.reliability_active

        ## Or decode an explicit spike-count matrix of shape (n_cells, n_time_bins):
        spkcount = a_dst_decoder2D.unit_specific_time_binned_spike_counts  # or sparse.toarray() from compute_reliability_new
        most_likely_positions, p_x_given_n, most_likely_position_indicies, _ = a_dst_decoder2D.decode(
            spkcount, time_bin_size=a_dst_decoder2D.time_bin_size, debug_print=False,
        )
        # p_x_given_n.shape: (*spatial_bins, n_time_bins); most_likely_positions.shape: (n_time_bins, 2) for 2D

        ## Epoch-restricted decoding (same API as parent; uses DST decode polymorphically):
        # filter_epochs_decoder_result = a_dst_decoder2D.decode_specific_epochs(
        #     spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=a_dst_decoder2D.time_bin_size,
        # )


    """
    ## New `BayesianPlacemapPositionDecoderDST`-specific fields:
    # Computed Cell Confusion Reliability Variables ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    t_bin_aclus_reliability_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})
    per_tbin_aclu_spike_counts_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_t_bins','n_neurons',)})
    time_bin_info_df: pd.DataFrame = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_t_bins',)})
    per_tbin_aclu_spike_counts_sparse: csr_matrix = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons','n_t_bins',)}) # (n_aclus, n_t_bins) - (25, 1427042)


    
    field_threshold_frac: float = serialized_field(default=0.20)
    
    n_top_peaks: int = serialized_field(default=3)
    slice_level_multiplier: float = serialized_field(default=0.20)
    fn_tn_mode: str = serialized_field(default='occupancy_seconds')
    in_field_masks: Optional[Dict[int, np.ndarray]] = non_serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons', 'n_xbins', 'n_ybins')})


    ## Cell reliability variables:
    discount_silence: bool = non_serialized_field(default=False)
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
    def init_from_stateful_decoder(cls, stateful_decoder: "BayesianPlacemapPositionDecoder", active_peak_prominence_2d_results=None, field_threshold_frac: float = 0.20, discount_silence: bool = False, **kwargs):
        """Creates a new DST decoder instance from an existing stateful Bayesian decoder.

        If ``active_peak_prominence_2d_results`` is provided, also runs ``compute_unit_confusion_reliability_variables``
        (optional confusion-matrix / in-field-mask products; not required for DST decode).
        Extra kwargs for that step: ``max_t_idx``, ``spikes_df``, ``time_bin_size_seconds``.

        To build confusion masks without a pipeline PeakProminence2D cache, construct the decoder then call
        ``compute_unit_confusion_reliability_variables(...)`` with ``active_peak_prominence_2d_results`` omitted
        (recomputes from ``self.pf``).
        """
        max_t_idx = kwargs.pop('max_t_idx', None)
        confusion_spikes_df = kwargs.pop('spikes_df', None)
        time_bin_size_seconds = kwargs.pop('time_bin_size_seconds', None)
        _obj = cls(time_bin_size=stateful_decoder.time_bin_size, pf=deepcopy(stateful_decoder.pf), spikes_df=deepcopy(stateful_decoder.spikes_df), field_threshold_frac=field_threshold_frac, discount_silence=discount_silence, debug_print=kwargs.pop('debug_print', stateful_decoder.debug_print), **kwargs)
        if active_peak_prominence_2d_results is not None:
            _obj.compute_unit_confusion_reliability_variables(active_peak_prominence_2d_results=active_peak_prominence_2d_results, spikes_df=confusion_spikes_df, time_bin_size_seconds=time_bin_size_seconds, max_t_idx=max_t_idx)
            self._compute_reliability_metrics() ## compute

        return _obj


    @classmethod
    def init_from_placefields(cls, pf: PfND, time_bin_size: float, spikes_df: pd.DataFrame, active_peak_prominence_2d_results=None, field_threshold_frac: float = 0.20, discount_silence: bool = False, debug_print: bool = False, **kwargs):
        """Creates a new DST decoder instance from a placefields object plus required decoder inputs.

        If ``active_peak_prominence_2d_results`` is provided, also runs ``compute_unit_confusion_reliability_variables``.
        Extra kwargs for that step: ``max_t_idx``, ``time_bin_size_seconds``.

        To build confusion masks without a pipeline PeakProminence2D cache, construct the decoder then call
        ``compute_unit_confusion_reliability_variables(...)`` with ``active_peak_prominence_2d_results`` omitted
        (recomputes from ``self.pf``).
        """
        max_t_idx = kwargs.pop('max_t_idx', None)
        time_bin_size_seconds = kwargs.pop('time_bin_size_seconds', None)
        _obj = cls(time_bin_size=time_bin_size, pf=deepcopy(pf), spikes_df=deepcopy(spikes_df), field_threshold_frac=field_threshold_frac, discount_silence=discount_silence, debug_print=debug_print, **kwargs)
        if active_peak_prominence_2d_results is not None:
            _obj.compute_unit_confusion_reliability_variables(active_peak_prominence_2d_results=active_peak_prominence_2d_results, spikes_df=spikes_df, time_bin_size_seconds=time_bin_size_seconds, max_t_idx=max_t_idx)
            self._compute_reliability_metrics() ## compute

        return _obj


    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        super().post_load()
        self.reliability_active = None
        self.reliability_silent = None
        self.in_field_masks = None


    def setup(self):
        super().setup()
        self.t_bin_aclus_reliability_df = None
        self.per_tbin_aclu_spike_counts_df = None
        self.time_bin_info_df = None
        self.per_tbin_aclu_spike_counts_sparse = None
        self.reliability_active = None
        self.reliability_silent = None
        self.in_field_masks = None
        self._compute_reliability_metrics() ## compute



    def get_by_id(self, ids, defer_compute_all: bool = False):
        """Return a DST copy restricted to ``ids``, preserving DST config and sliced reliability when present.

        Mirrors ``BayesianPlacemapPositionDecoder.get_by_id`` but constructs ``BayesianPlacemapPositionDecoderDST``.
        Time-bin reliability tables / sparse spike counts are left None on the slice.
        """
        ids = np.asarray(ids)
        source_ids = np.asarray(self.neuron_IDs)
        assert np.all(np.isin(ids, source_ids))
        keep = np.isin(source_ids, ids)  # original neuron order

        neuron_sliced_pf: PfND = self.pf.get_by_id(ids)
        spikes_df = deepcopy(self.spikes_df)
        if (spikes_df is not None) and ('aclu' in spikes_df.columns):
            spikes_df = spikes_df[np.isin(spikes_df['aclu'].to_numpy(), ids)].copy()

        neuron_sliced_decoder = BayesianPlacemapPositionDecoderDST(time_bin_size=self.time_bin_size, pf=neuron_sliced_pf, spikes_df=spikes_df, field_threshold_frac=self.field_threshold_frac, discount_silence=self.discount_silence, n_top_peaks=self.n_top_peaks, slice_level_multiplier=self.slice_level_multiplier, fn_tn_mode=self.fn_tn_mode, setup_on_init=False, post_load_on_init=False, debug_print=self.debug_print)

        neuron_sliced_decoder.neuron_IDs = source_ids[keep]
        neuron_sliced_decoder.neuron_IDXs = np.arange(int(np.sum(keep)))
        neuron_sliced_decoder.F = self.F[:, keep]
        neuron_sliced_decoder.P_x = deepcopy(self.P_x)

        if self.unit_specific_time_binned_spike_counts is not None:
            neuron_sliced_decoder.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[keep, :]
            neuron_sliced_decoder.total_spike_counts_per_window = np.sum(neuron_sliced_decoder.unit_specific_time_binned_spike_counts, axis=0)
            neuron_sliced_decoder.time_binning_container = deepcopy(self.time_binning_container)

        # Reuse per-cell reliability / masks when already computed on the full decoder
        if self.reliability_active is not None:
            neuron_sliced_decoder.reliability_active = np.asarray(self.reliability_active)[keep]
        if self.reliability_silent is not None:
            neuron_sliced_decoder.reliability_silent = np.asarray(self.reliability_silent)[keep]
        if self.in_field_masks is not None:
            id_set = set(int(x) for x in ids)
            neuron_sliced_decoder.in_field_masks = {int(nid): mask for nid, mask in self.in_field_masks.items() if int(nid) in id_set}

        # Leave time-bin reliability tables / sparse counts unset on the slice
        neuron_sliced_decoder.t_bin_aclus_reliability_df = None
        neuron_sliced_decoder.per_tbin_aclu_spike_counts_df = None
        neuron_sliced_decoder.time_bin_info_df = None
        neuron_sliced_decoder.per_tbin_aclu_spike_counts_sparse = None

        # Invalidate decode caches (cannot neuron-slice a posterior)
        neuron_sliced_decoder.flat_p_x_given_n = None
        neuron_sliced_decoder.p_x_given_n = None
        neuron_sliced_decoder.most_likely_positions = None
        neuron_sliced_decoder.most_likely_position_indicies = None
        neuron_sliced_decoder.most_likely_position_flat_indicies = None
        neuron_sliced_decoder.marginal = None
        neuron_sliced_decoder.revised_most_likely_positions = None

        if not defer_compute_all:
            neuron_sliced_decoder.compute_all()

        return neuron_sliced_decoder


    # ==================================================================================================================================================================================================================================================================================== #
    # Main Methods                                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    
    @function_attributes(short_name=None, tags=['UNUSED', 'ALT', 'pho', 'true-positive', 'false-positive', 'reliability'], input_requires=[], output_provides=[], uses=['CellIndividualReliabilityMatrix.compute_peak_prominence_2d_from_pf', 'CellIndividualReliabilityMatrix.build_in_field_masks_xy', 'CellIndividualReliabilityMatrix.compute_reliability_matrix'], used_by=[], creation_date='2026-07-23 09:58', related_items=[])
    def compute_unit_confusion_reliability_variables(self, active_peak_prominence_2d_results=None, spikes_df: Optional[pd.DataFrame] = None, time_bin_size_seconds: Optional[float] = None, max_t_idx: Optional[int] = None, **kwargs):
        """Compute per-aclu reliability via CellIndividualReliabilityMatrix and store results on self.

        #TODO 2026-07-23 09:59: - [ ] this result is not currently used by any of the main computations because we use the skragg information reliability for each cell instead.

        Parameters
        ----------
        active_peak_prominence_2d_results : optional PeakProminence2D results for in-field masks.
            If None, recomputes a minimal PeakProminence2D from ``self.pf`` via
            ``CellIndividualReliabilityMatrix.compute_peak_prominence_2d_from_pf`` (no pipeline cache required).
        spikes_df : optional spikes override; defaults to `self.spikes_df` sliced to `self.neuron_IDs`.
        time_bin_size_seconds : temporal bin width; defaults to `self.time_bin_size`.
        max_t_idx : optional cap on number of time bins (None = all).

        Uses instance fields ``n_top_peaks``, ``slice_level_multiplier``, and ``fn_tn_mode``.

        Returns
        -------
        t_bin_aclus_reliability_df, per_tbin_aclu_spike_counts_df, time_bin_info_df, per_tbin_aclu_spike_counts_sparse


        UPDATES:
            self.in_field_masks, self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse
        """
        pfs = self.pf
        ratemaps = self.ratemap
        neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else ratemaps.neuron_ids)
        if spikes_df is None:
            if self.spikes_df is None:
                self.spikes_df = deepcopy(pfs.filtered_spikes_df).spikes.sliced_by_neuron_id(neuron_ids)
            spikes_df = deepcopy(self.spikes_df)

        spikes_df = spikes_df.spikes.sliced_by_neuron_id(neuron_ids)
        if time_bin_size_seconds is None:
            time_bin_size_seconds = self.time_bin_size

        if (self.spikes_df is None):
            self.spikes_df = spikes_df

        if (self.time_bin_size is None):
            self.time_bin_size = time_bin_size_seconds

        if active_peak_prominence_2d_results is None:
            active_peak_prominence_2d_results = CellIndividualReliabilityMatrix.compute_peak_prominence_2d_from_pf(pfs, neuron_ids=neuron_ids)

        self.in_field_masks = CellIndividualReliabilityMatrix.build_in_field_masks_xy(active_peak_prominence_2d_results=active_peak_prominence_2d_results, ratemaps=ratemaps,
            n_top_peaks=self.n_top_peaks, slice_level_multiplier=self.slice_level_multiplier, 
            neuron_ids=neuron_ids,
        )

        self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse = CellIndividualReliabilityMatrix.compute_reliability_matrix(
            spikes_df=spikes_df, pfs=pfs, ratemaps=ratemaps, in_field_masks=self.in_field_masks, neuron_ids=neuron_ids,
            time_bin_size_seconds=time_bin_size_seconds, max_t_idx=max_t_idx, **kwargs,
        )

        return self.t_bin_aclus_reliability_df, self.per_tbin_aclu_spike_counts_df, self.time_bin_info_df, self.per_tbin_aclu_spike_counts_sparse

        

    def _compute_reliability_metrics(self, **kwargs):
        """
        Builds static per-cell reliability (alpha_i) from Skaggs spatial information.
        Requires only ``self.pf`` so first ``decode()`` / ``compute_all()`` works without
        a prior ``compute_reliability_new`` call.
        """
        assert (self.pf is not None)

        an_active_pf = deepcopy(self.pf)
        ## INPUTS: an_active_pf
        alpha_skaggs = CellIndividualReliabilityMatrix.compute_skaggs_alpha(an_active_pf, k=1.0) # array([0.417225, 0.612937, 0.0186054, 0.839156, 0.253242, 0.390859, 0.551637, 0.410431, 0.232258, 0.319258, 0.0831956, 0.500425, 0.439415, 0.40174, 0.460294, 0.507179, 0.467489, 0.487803, 0.262977, 0.316431, 0.499277, 0.356243, 0.758122, 0.133721, 0.649214])
        # alpha_sparsity = CellIndividualReliabilityMatrix.compute_sparsity_alpha(an_active_pf)  # correlated with Skaggs; do not multiply into alpha

        # ## time-dependent alpha (requires per_tbin_aclu_spike_counts_sparse from compute_reliability_new)
        # alpha_dsnr = CellIndividualReliabilityMatrix.compute_dsnr_alpha(an_active_pf, n_i = self.per_tbin_aclu_spike_counts_sparse.toarray(), tau=self.time_bin_size)

        # Combine metrics to build the basal epistemic reliability limit (alpha_i) for each cell
        # Ensuring the result is properly bounded [0, 1]
        # Basal epistemic reliability (alpha_i) from Skaggs SI alone — already in [0, 1)
        R_base = np.clip(alpha_skaggs, 0.0, 1.0)

        self.reliability_active = R_base
        
        # Map reliability for silence (n_i = 0). 
        # Defaults to 1.0 (perfect reliability -> collapses to pure Bayesian) if discounting is disabled.
        if self.discount_silence:
            self.reliability_silent = R_base
        else:
            self.reliability_silent = np.ones_like(R_base)


        # # Old Method _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        # nCells, nPositionBins = ratemaps_flat.shape
        # R_active = np.ones(nCells)
        # R_silent = np.ones(nCells)

        # for i in range(nCells):
        #     rm = ratemaps_flat[i, :]
        #     max_rate = np.nanmax(rm)

        #     # Handle cells that are silent everywhere or have invalid rates
        #     if max_rate <= 0 or np.isnan(max_rate):
        #         R_active[i] = 0.0 
        #         R_silent[i] = 0.0
        #         continue

        #     # Step A: Create Spatial Masks
        #     theta = self.field_threshold_frac * max_rate
        #     in_field_mask = (rm >= theta)
        #     out_field_mask = ~in_field_mask

        #     # Step B: Calculate Mean Regional Rates
        #     mu_in = np.nanmean(rm[in_field_mask]) if np.any(in_field_mask) else 0.0
        #     mu_out = np.nanmean(rm[out_field_mask]) if np.any(out_field_mask) else 0.0

        #     # Step C: Define Spatial Precision (R_i)
        #     if (mu_in + mu_out) > 0:
        #         R_active[i] = mu_in / (mu_in + mu_out)
        #     else:
        #         R_active[i] = 0.0
                
        #     # Map NPV for silence. Defaults to 1.0 (no discounting) if disabled.
        #     if self.discount_silence:
        #         R_silent[i] = R_active[i] 

        # self.reliability_active = R_active
        # self.reliability_silent = R_silent


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
            # self._compute_reliability_metrics(ratemaps_flat)
            self._compute_reliability_metrics()

        tau = self.time_bin_size
        nTimeBins = spkcount.shape[1]
        
        # 3. Incorporate Occupancy Prior P(x, y) 
        try:
            # Extract historical animal occupancy in seconds
            P_v = self.pf.occupancy # self.pf.ratemap.probability_normalized_occupancy
            P_v = np.nan_to_num(P_v, nan=0.0)
            P_v_flat = P_v.flatten()
            
            # Normalize to form the basic probability assignment prior (m_0)
            sum_P_v = np.sum(P_v_flat)
            if sum_P_v > 0:
                P_v_flat = P_v_flat / sum_P_v
            else:
                P_v_flat = np.ones(nPositionBins) / nPositionBins

        except Exception as err:
            # Safe fallback if occupancy prior is unavailable
            print(f'WARNING: fallback to uniform posterior because occupancy prior is unavailable or calculation failed with error {err}.')
            P_v_flat = np.ones(nPositionBins) / nPositionBins

        # Initialize log_posterior with the log of the prior P(v) to avoid float underflow
        log_posterior = np.zeros((nTimeBins, nPositionBins), dtype=np.float64)
        log_posterior += np.log(P_v_flat + 1e-15)[np.newaxis, :]
        
        # 4. Iterative Likelihood Evaluation (Memory Efficient Evidential Fusion)
        for cell in range(nCells):
            cell_spkcnt = spkcount[cell, :][:, np.newaxis]   # (nTimeBins, 1)
            cell_ratemap = ratemaps_flat[cell, :][np.newaxis, :]  # (1, nPositionBins)

            # Poisson Sensor Likelihood Density
            L_i = ( (tau * cell_ratemap) ** cell_spkcnt ) * np.exp(-tau * cell_ratemap)
            Z_i = np.sum(L_i, axis=1, keepdims=True)
            
            # Convert raw likelihoods to specific evidential mass assignments
            with np.errstate(divide='ignore', invalid='ignore'):
                p_i = L_i / Z_i
            p_i = np.where(Z_i == 0, 1.0 / nPositionBins, p_i)

            # Apply Reliability Conditional on Firing State
            active_mask = (cell_spkcnt > 0)
            R_effective = np.where(active_mask, self.reliability_active[cell], self.reliability_silent[cell])
            
            # Dempster-Shafer Unnormalized Conjoint Mass formulation: 
            # E_i(v) = [ alpha_i * ( L_i(v) / SUM_w L_i(w) ) ] + ( 1 - alpha_i )
            # The uncommitted mass (1 - alpha_i) applies uniformly to all points in Omega.
            E_i = (R_effective * p_i) + (1.0 - R_effective)
            
            # Dempster's Rule of Combination (Orthogonal sum via logarithms)
            log_posterior += np.log(E_i + 1e-15)

        ## END for cell in range(nCells)...

        # 5. Convert back to linear probability space (Log-Sum-Exp Trick)
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_posterior_max)
        
        # Final Global Normalization to output strict evidential Belief: Bel({v})
        sum_post = np.sum(posterior, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            posterior /= sum_post
            
        posterior = np.where(sum_post == 0, 1.0 / nPositionBins, posterior)

        # 6. Reshape to match pyphoplacecellanalysis expectations: (*Spatial_Shape, nTimeBins)
        posterior = posterior.T # (nPositionBins, nTimeBins)
        final_shape = (*spatial_shape, nTimeBins)
        
        return posterior.reshape(final_shape)


        # OLD IMPLEMENTATION _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

        # # 1. Dynamically handle 1D vs 2D spatial layouts
        # original_shape = ratemaps.shape
        # nCells = original_shape[0]
        # spatial_shape = original_shape[1:] 
        # nPositionBins = np.prod(spatial_shape)
        
        # ratemaps_flat = ratemaps.reshape(nCells, nPositionBins)
        
        # # 2. Ensure spatial SNR metrics are prepared
        # if self.reliability_active is None:
        #     self._compute_reliability_metrics(ratemaps_flat)
            
        # tau = self.time_bin_size
        # nTimeBins = spkcount.shape[1]
        
        # # We accumulate log-evidence to prevent float underflow and save RAM
        # log_posterior = np.zeros((nTimeBins, nPositionBins), dtype=np.float64)
        
        # # 3. Iterative Likelihood Evaluation (Memory Efficient)
        # for cell in range(nCells):
        #     cell_spkcnt = spkcount[cell, :][:, np.newaxis]   # (nTimeBins, 1)
        #     cell_ratemap = ratemaps_flat[cell, :][np.newaxis, :]  # (1, nPositionBins)

        #     # Poisson Likelihood Density (ignoring constant 1/n! term)
        #     L_i = ( (tau * cell_ratemap) ** cell_spkcnt ) * np.exp(-tau * cell_ratemap)
        #     Z_i = np.sum(L_i, axis=1, keepdims=True)
            
        #     # Convert raw likelihoods to specific probability density mappings (p_i)
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         p_i = L_i / Z_i
        #     p_i = np.where(Z_i == 0, 1.0 / nPositionBins, p_i)

        #     # Apply Reliability Conditional on Firing State
        #     active_mask = (cell_spkcnt > 0)
        #     R_effective = np.where(active_mask, self.reliability_active[cell], self.reliability_silent[cell])
            
        #     # Shafer Discounting Rule: E_i(x) = R_i * p_i(x|n_i) + (1 - R_i) * (1 / |Theta|)
        #     E_i = (R_effective * p_i) + ((1.0 - R_effective) / nPositionBins)
            
        #     # Dempster's Rule of Combination (Summing Log Evidences)
        #     log_posterior += np.log(E_i + 1e-15)

        # # 4. Convert back to linear probability space (Log-Sum-Exp Trick)
        # log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        # posterior = np.exp(log_posterior - log_posterior_max)
        
        # # Final Global Normalization
        # sum_post = np.sum(posterior, axis=1, keepdims=True)
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     posterior /= sum_post
            
        # posterior = np.where(sum_post == 0, 1.0 / nPositionBins, posterior)

        # # 5. Reshape to match pyphoplacecellanalysis expectations: (*Spatial_Shape, nTimeBins)
        # posterior = posterior.T # (nPositionBins, nTimeBins)
        # final_shape = (*spatial_shape, nTimeBins)
        
        # return posterior.reshape(final_shape)



    @function_attributes(short_name='decode', tags=['MAIN', 'decode', 'DST', 'pure'], input_requires=[], output_provides=[], creation_date='2026-07-23 06:07',
        uses=['self.compute_posterior', 'BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions'],
        used_by=['BayesianPlacemapPositionDecoder.hyper_perform_decode', 'BayesianPlacemapPositionDecoder._perform_decoding_specific_epochs'])
    def decode(self, unit_specific_time_binned_spike_counts, time_bin_size: float, output_flat_versions=False, debug_print=True):
        """DST decode: same contract as parent ``BayesianPlacemapPositionDecoder.decode``, but uses ``compute_posterior`` (Shafer discounting) instead of Zhang Bayesian.

        Inputs:
            unit_specific_time_binned_spike_counts: np.array of shape (num_cells, num_time_bins)

        Returns:
            most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container
        """
        num_cells = np.shape(unit_specific_time_binned_spike_counts)[0]
        num_time_windows = np.shape(unit_specific_time_binned_spike_counts)[1]
        if debug_print:
            print(f'num_cells: {num_cells}, num_time_windows: {num_time_windows}')

        prev_time_bin_size = self.time_bin_size
        with WrappingMessagePrinter(f'decode(...) [DST] called. Computing {num_time_windows} windows for final_p_x_given_n...', begin_line_ending='... ', finished_message='decode completed.', enable_print=(debug_print or self.debug_print)):
            if time_bin_size is None:
                print(f'time_bin_size is None, using internal self.time_bin_size.')
                time_bin_size = self.time_bin_size
            else:
                if (self.time_bin_size is None) or (time_bin_size != self.time_bin_size):
                    self.time_bin_size = time_bin_size

            try:
                
                p_x_given_n = self.compute_posterior(unit_specific_time_binned_spike_counts)
                curr_flat_p_x_given_n = np.reshape(p_x_given_n, (-1, num_time_windows))
                if debug_print:
                    print(f'curr_flat_p_x_given_n.shape: {curr_flat_p_x_given_n.shape}')

                most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(curr_flat_p_x_given_n, self.original_position_data_shape)

                if output_flat_versions:
                    flat_outputs_container = DynamicContainer(flat_p_x_given_n=curr_flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies)
                else:
                    flat_outputs_container = None

                if self.ndim > 1:
                    most_likely_positions = np.vstack((self.xbin_centers[most_likely_position_indicies[0, :]], self.ybin_centers[most_likely_position_indicies[1, :]])).T
                else:
                    most_likely_positions = np.squeeze(self.xbin_centers[most_likely_position_indicies[0, :]])

                return most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container

            finally:
                self.time_bin_size = prev_time_bin_size

