import numpy as np
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

# Import the base decoder from your pyphoplacecellanalysis repo
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder


@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-07-03 11:04', related_items=[])
class BayesianPlacemapPositionDecoderDST(BayesianPlacemapPositionDecoder):
    """
    Dempster-Shafer Theory (DST) updated Position Decoder.
    Mirrors pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
    but implements Shafer Discounting of conflicting place cell likelihoods.
    
    Reliability (R_i) is calculated dynamically for each cell based on its 
    Spatial Signal-to-Noise Ratio (in-field vs. out-of-field expected firing rates).


	Usage:
		from pyphoplacecellanalysis.Analysis.Decoder.reconstruction_dst import BayesianPlacemapPositionDecoderDST


		a_dst_decoder2D: BayesianPlacemapPositionDecoderDST = BayesianPlacemapPositionDecoderDST(time_bin_size=pf2D_Decoder.time_bin_size, pf=pf2D_Decoder.pf, spikes_df=deepcopy(pf2D_Decoder.spikes_df)) # , ratemaps=None
		a_dst_decoder2D
    """
    
    def __init__(self, time_bin_size, pf, field_threshold_frac=0.20, discount_silence=False, **kwargs):
        """
        Parameters
        ----------
        time_bin_size : float
            The decoding time bin size.
        pf : Placefield1D or Placefield2D
            The underlying placefield object.
        ratemaps : np.ndarray, optional
            The ratemaps array. If None, derived from pf.
        field_threshold_frac : float
            The fraction of the peak firing rate used to define the boundary 
            of the "in-field" vs "out-of-field" masks (default: 0.20).
        discount_silence : bool
            If True, applies Shafer Discounting to time bins where the cell 
            did NOT fire (n_i = 0). (default: False).
        """
        super().__init__(time_bin_size=time_bin_size, pf=pf, **kwargs)
        
        self.field_threshold_frac = field_threshold_frac
        self.discount_silence = discount_silence
        
        # Computed structural reliability metrics
        self.reliability_active = None
        self.reliability_silent = None


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

