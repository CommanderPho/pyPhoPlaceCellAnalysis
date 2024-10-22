from copy import deepcopy
from pathlib import Path
from typing import Optional
from attrs import define, field, Factory, astuple, asdict
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_attribute_field, serialized_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin
from pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs import BaseConfig
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.function_helpers import get_fn_kwargs_with_defaults, get_decorated_function_attributes, fn_best_name
from pyphocorehelpers.print_helpers import strip_type_str_to_classname

""" 
from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import merged_directional_placefields_Parameters

Generated programmatically from `pyphoplacecellanalysis.General.PipelineParameterClassTemplating.GlobalComputationParametersAttrsClassTemplating` on 2024-10-07


"""

class BaseGlobalComputationParameters(BaseConfig):
    """ Base class
    """

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes  """
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
    
    def values_only_repr(self, attr_separator_str: str=",\n", sub_attr_additive_seperator_str:str='\t'):
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_value = getattr(self, a.name)
            if hasattr(attr_value, 'values_only_repr'):
                _new_attr_sep_str: str = f"{attr_separator_str}{sub_attr_additive_seperator_str}"
                # attr_value = attr_value.values_only_repr(attr_separator_str=attr_separator_str, sub_attr_additive_seperator_str=sub_attr_additive_seperator_str)
                attr_value = attr_value.values_only_repr(attr_separator_str=_new_attr_sep_str, sub_attr_additive_seperator_str=sub_attr_additive_seperator_str)
            attr_reprs.append(f"{a.name}: {attr_value}")
            
        content = attr_separator_str.join(attr_reprs)
        # return f"{type(self).__name__}({content}\n)"
        return content
    


@define(slots=False, eq=False, repr=False)
class merged_directional_placefields_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for merged_directional_placefields_Parameters. 
    """
    laps_decoding_time_bin_size: float = serialized_attribute_field(default=0.25)
    ripple_decoding_time_bin_size: float = serialized_attribute_field(default=0.025)
    should_validate_lap_decoding_performance: bool = serialized_attribute_field(default=False)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class rank_order_shuffle_analysis_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for rank_order_shuffle_analysis_Parameters. 
    """
    num_shuffles: int = serialized_attribute_field(default=500)
    minimum_inclusion_fr_Hz: float = serialized_attribute_field(default=5.0)
    included_qclu_values: list = serialized_field(default=[1, 2])
    skip_laps: bool = serialized_attribute_field(default=False)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_decoders_decode_continuous_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_decoders_decode_continuous_Parameters. 
    """
    time_bin_size: Optional[float] = serialized_attribute_field(default=None)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_decoders_evaluate_epochs_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_decoders_evaluate_epochs_Parameters. 
    """
    should_skip_radon_transform: bool = serialized_attribute_field(default=False)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_train_test_split_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_train_test_split_Parameters. 
    """
    training_data_portion: float = serialized_attribute_field(default=0.8333333333333334)
    debug_output_hdf5_file_path: Optional[Path] = serialized_attribute_field(default=None)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class long_short_decoding_analyses_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for long_short_decoding_analyses_Parameters. 
    """
    decoding_time_bin_size: Optional[float] = serialized_attribute_field(default=None)
    perform_cache_load: bool = serialized_attribute_field(default=False)
    always_recompute_replays: bool = serialized_attribute_field(default=False)
    override_long_epoch_name: Optional[str] = serialized_attribute_field(default=None)
    override_short_epoch_name: Optional[str] = serialized_attribute_field(default=None)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class long_short_rate_remapping_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for long_short_rate_remapping_Parameters. 
    """
    decoding_time_bin_size: Optional[float] = serialized_attribute_field(default=None)
    perform_cache_load: bool = serialized_attribute_field(default=False)
    always_recompute_replays: bool = serialized_attribute_field(default=False)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class long_short_inst_spike_rate_groups_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for long_short_inst_spike_rate_groups_Parameters. 
    """
    instantaneous_time_bin_size_seconds: Optional[float] = serialized_attribute_field(default=0.01)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class wcorr_shuffle_analysis_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for wcorr_shuffle_analysis_Parameters. 
    """
    num_shuffles: int = serialized_attribute_field(default=1024)
    drop_previous_result_and_compute_fresh: bool = serialized_attribute_field(default=False)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class perform_specific_epochs_decoding_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for _perform_specific_epochs_decoding_Parameters. 
    """
    decoder_ndim: int = serialized_attribute_field(default=2)
    filter_epochs: str = serialized_attribute_field(default='ripple')
    decoding_time_bin_size: Optional[float] = serialized_attribute_field(default=0.02)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class DEP_ratemap_peaks_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for _DEP_ratemap_peaks_Parameters. 
    """
    peak_score_inclusion_percent_threshold: float = serialized_attribute_field(default=0.25)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class ratemap_peaks_prominence2d_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for ratemap_peaks_prominence2d_Parameters. 
    """
    step: float = serialized_attribute_field(default=0.01)
    peak_height_multiplier_probe_levels: tuple = serialized_field(default=(0.5, 0.9))
    minimum_included_peak_height: float = serialized_attribute_field(default=0.2)
    uniform_blur_size: int = serialized_attribute_field(default=3)
    gaussian_blur_sigma: int = serialized_attribute_field(default=3)
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)
        




@define(slots=False)
class ComputationKWargParameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ the base class
    
    from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, _perform_specific_epochs_decoding_Parameters, _DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

    
    """
    merged_directional_placefields: merged_directional_placefields_Parameters = serialized_field(default=Factory(merged_directional_placefields_Parameters))
    rank_order_shuffle_analysis: rank_order_shuffle_analysis_Parameters = serialized_field(default=Factory(rank_order_shuffle_analysis_Parameters))
    directional_decoders_decode_continuous: directional_decoders_decode_continuous_Parameters = serialized_field(default=Factory(directional_decoders_decode_continuous_Parameters))
    directional_decoders_evaluate_epochs: directional_decoders_evaluate_epochs_Parameters = serialized_field(default=Factory(directional_decoders_evaluate_epochs_Parameters))
    directional_train_test_split: directional_train_test_split_Parameters = serialized_field(default=Factory(directional_train_test_split_Parameters))
    long_short_decoding_analyses: long_short_decoding_analyses_Parameters = serialized_field(default=Factory(long_short_decoding_analyses_Parameters))
    long_short_rate_remapping: long_short_rate_remapping_Parameters = serialized_field(default=Factory(long_short_rate_remapping_Parameters))
    long_short_inst_spike_rate_groups: long_short_inst_spike_rate_groups_Parameters = serialized_field(default=Factory(long_short_inst_spike_rate_groups_Parameters))
    wcorr_shuffle_analysis: wcorr_shuffle_analysis_Parameters = serialized_field(default=Factory(wcorr_shuffle_analysis_Parameters))
    perform_specific_epochs_decoding: perform_specific_epochs_decoding_Parameters = serialized_field(default=Factory(perform_specific_epochs_decoding_Parameters))
    DEP_ratemap_peaks: DEP_ratemap_peaks_Parameters = serialized_field(default=Factory(DEP_ratemap_peaks_Parameters))
    ratemap_peaks_prominence2d: ratemap_peaks_prominence2d_Parameters = serialized_field(default=Factory(ratemap_peaks_prominence2d_Parameters))

    @classmethod
    def init_from_pipeline(cls, curr_active_pipeline):
        """ 
        param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)

        Usage:        
            from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters

            ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
            if curr_active_pipeline.global_computation_results.computation_config is None:
                print('global_computation_results.computation_config is None! Making new one!')
                curr_active_pipeline.global_computation_results.computation_config = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
                print(f'\tdone. Pipeline needs resave!')
                
            
        """
        from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
        
        registered_merged_computation_function_default_kwargs_dict = GlobalComputationParametersAttrsClassTemplating.main_extract_params_default_values(curr_active_pipeline=curr_active_pipeline)
        
        ignore_kwarg_names = GlobalComputationParametersAttrsClassTemplating.ignore_kwarg_names
        registered_merged_computation_function_default_kwargs_dict = {fn_best_name(v):get_fn_kwargs_with_defaults(v, ignore_kwarg_names=ignore_kwarg_names) for k, v in curr_active_pipeline.registered_merged_computation_function_dict.items()}
        registered_merged_computation_function_default_kwargs_dict = {k:v for k, v in registered_merged_computation_function_default_kwargs_dict.items() if len(v)>0} # filter empty lists
        
        # params_class_type_list = [merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, perform_specific_epochs_decoding_Parameters, DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters]
        # params_class_type_dict = dict(zip({k.removeprefix('_') for k in imports_dict.keys()}, params_class_type_list))
        # params_class_type_dict = dict(zip({k for k in imports_dict.keys()}, params_class_type_list))
        # params_class_type_dict = dict(zip(imports_list, params_class_type_list))
        # params_class_type_dict
        
        params_class_type_dict = deepcopy(cls.__annotations__)
        
        ## Convert to the new native types
        ## INPUTS: registered_merged_computation_function_default_kwargs_dict, params_class_type_dict
        _out_param_typed_parameters_dict = {}
        for k, v_dict in registered_merged_computation_function_default_kwargs_dict.items():
            a_type = params_class_type_dict[k.removeprefix('_')]
            _out_param_typed_parameters_dict[k.removeprefix('_')] = a_type(**v_dict)
            # a_type = params_class_type_dict[k]
            # _out_param_typed_parameters_dict[k.removeprefix('_')] = a_type(**v_dict)
        # _out_param_typed_parameters_dict

        ## OUTPUTS: _out_param_typed_parameters_dict
        # param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
        param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
        return param_typed_parameters

        ## OUTPUTS: param_typed_parameters
