from pathlib import Path
from typing import Optional
from attrs import define, field, Factory, astuple, asdict

@define(slots=False, eq=False, repr=False)
class merged_directional_placefields_Parameters(object):
    """ Docstring for merged_directional_placefields_Parameters. 
    """
    laps_decoding_time_bin_size: float = field(default=0.25)
    ripple_decoding_time_bin_size: float = field(default=0.025)
    should_validate_lap_decoding_performance: bool = field(default=False)



@define(slots=False, eq=False, repr=False)
class rank_order_shuffle_analysis_Parameters(object):
    """ Docstring for rank_order_shuffle_analysis_Parameters. 
    """
    num_shuffles: int = field(default=500)
    minimum_inclusion_fr_Hz: float = field(default=5.0)
    included_qclu_values: list = field(default=[1, 2])
    skip_laps: bool = field(default=False)



@define(slots=False, eq=False, repr=False)
class directional_decoders_decode_continuous_Parameters(object):
    """ Docstring for directional_decoders_decode_continuous_Parameters. 
    """
    time_bin_size: Optional[float] = field(default=None)



@define(slots=False, eq=False, repr=False)
class directional_decoders_evaluate_epochs_Parameters(object):
    """ Docstring for directional_decoders_evaluate_epochs_Parameters. 
    """
    should_skip_radon_transform: bool = field(default=False)



@define(slots=False, eq=False, repr=False)
class directional_train_test_split_Parameters(object):
    """ Docstring for directional_train_test_split_Parameters. 
    """
    training_data_portion: float = field(default=0.8333333333333334)
    debug_output_hdf5_file_path: Optional[Path] = field(default=None)



@define(slots=False, eq=False, repr=False)
class long_short_decoding_analyses_Parameters(object):
    """ Docstring for long_short_decoding_analyses_Parameters. 
    """
    decoding_time_bin_size: Optional[float] = field(default=None)
    perform_cache_load: bool = field(default=False)
    always_recompute_replays: bool = field(default=False)
    override_long_epoch_name: Optional[str] = field(default=None)
    override_short_epoch_name: Optional[str] = field(default=None)



@define(slots=False, eq=False, repr=False)
class long_short_rate_remapping_Parameters(object):
    """ Docstring for long_short_rate_remapping_Parameters. 
    """
    decoding_time_bin_size: Optional[float] = field(default=None)
    perform_cache_load: bool = field(default=False)
    always_recompute_replays: bool = field(default=False)



@define(slots=False, eq=False, repr=False)
class long_short_inst_spike_rate_groups_Parameters(object):
    """ Docstring for long_short_inst_spike_rate_groups_Parameters. 
    """
    instantaneous_time_bin_size_seconds: Optional[float] = field(default=None)



@define(slots=False, eq=False, repr=False)
class wcorr_shuffle_analysis_Parameters(object):
    """ Docstring for wcorr_shuffle_analysis_Parameters. 
    """
    num_shuffles: int = field(default=1024)
    drop_previous_result_and_compute_fresh: bool = field(default=False)



@define(slots=False, eq=False, repr=False)
class _perform_specific_epochs_decoding_Parameters(object):
    """ Docstring for _perform_specific_epochs_decoding_Parameters. 
    """
    decoder_ndim: int = field(default=2)
    filter_epochs: str = field(default='ripple')
    decoding_time_bin_size: Optional[float] = field(default=0.02)



@define(slots=False, eq=False, repr=False)
class _DEP_ratemap_peaks_Parameters(object):
    """ Docstring for _DEP_ratemap_peaks_Parameters. 
    """
    peak_score_inclusion_percent_threshold: float = field(default=0.25)



@define(slots=False, eq=False, repr=False)
class ratemap_peaks_prominence2d_Parameters(object):
    """ Docstring for ratemap_peaks_prominence2d_Parameters. 
    """
    step: float = field(default=0.01)
    peak_height_multiplier_probe_levels: tuple = field(default=(0.5, 0.9))
    minimum_included_peak_height: float = field(default=0.2)
    uniform_blur_size: int = field(default=3)
    gaussian_blur_sigma: int = field(default=3)
    


@define(slots=False)
class ComputationKWargParameters(object):
    # Docstring for ComputationKWargParameters. 
    merged_directional_placefields: merged_directional_placefields_Parameters
    rank_order_shuffle_analysis: rank_order_shuffle_analysis_Parameters
    directional_decoders_decode_continuous: directional_decoders_decode_continuous_Parameters
    directional_decoders_evaluate_epochs: directional_decoders_evaluate_epochs_Parameters
    directional_train_test_split: directional_train_test_split_Parameters
    long_short_decoding_analyses: long_short_decoding_analyses_Parameters
    long_short_rate_remapping: long_short_rate_remapping_Parameters
    long_short_inst_spike_rate_groups: long_short_inst_spike_rate_groups_Parameters
    wcorr_shuffle_analysis: wcorr_shuffle_analysis_Parameters
    _perform_specific_epochs_decoding: _perform_specific_epochs_decoding_Parameters
    _DEP_ratemap_peaks: _DEP_ratemap_peaks_Parameters
    ratemap_peaks_prominence2d: ratemap_peaks_prominence2d_Parameters
    
