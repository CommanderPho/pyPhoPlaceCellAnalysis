from copy import deepcopy
import param
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pyphoplacecellanalysis.General.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import attrs
from attrs import define, field, Factory, astuple, asdict, fields
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_attribute_field, serialized_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin
from neuropy.core.parameters import BaseConfig
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.function_helpers import get_fn_kwargs_with_defaults, get_decorated_function_attributes, fn_best_name
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import BasePlotDataParams

""" 
from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import merged_directional_placefields_Parameters

Generated programmatically from `pyphoplacecellanalysis.General.PipelineParameterClassTemplating.GlobalComputationParametersAttrsClassTemplating` on 2024-10-07


"""

""" 

same_thresh_fraction_of_track: float=0.05, max_ignore_bins:float=2, max_jump_distance_cm: float=60.0, use_bin_units_instead_of_realworld:bool=False


"""


attrs_to_params_type_map = { str: param.String, int: param.Integer, float: param.Number, bool: param.Boolean, list: param.List, dict: param.Dict, tuple: param.Tuple, Path: param.Path,
                Optional[str]: param.String, Optional[int]: param.Integer, Optional[float]: param.Number, Optional[bool]: param.Boolean, Optional[Path]: param.Path,
                Optional[list]: param.List, Optional[dict]: param.Dict, Optional[tuple]: param.Tuple,
                }


@function_attributes(short_name=None, tags=['parameters', 'attrs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 02:56', related_items=[])
def attrs_to_parameters(cls):
    """ captures: `attrs_to_params_type_map`
    """
    for field in attrs.fields(cls):
        default = field.default if field.default is not attrs.NOTHING else None
        # setattr(cls, field.name, param.Parameter(default=default))
        # p_type = type_map.get(field.type, param.Parameter)
        p_type = attrs_to_params_type_map.get(field.type, None)
        assert (p_type is not None), f"failed for field: {field}"
        # if field.metadata is None:
        #     field.metadata = {} ## initialize
        ## update the field metadata
        # field.metadata.update(param=p_type(default=default))
        # field.metadata['param'] = p_type(default=default)
        # setattr(cls, field.name, p_type(default=default))
        curr_param_class_var_name: str = f"{field.name}_PARAM"
        
        if hasattr(cls, curr_param_class_var_name):
            delattr(cls, curr_param_class_var_name) ## remove extant
            assert (not hasattr(cls, curr_param_class_var_name)), f"hasattr even after removal!"

               

        param_obj = p_type(default=default)
        # set the parameter on the class under the same name as the field
        # setattr(cls, curr_param_class_var_name, param_obj)
        # setattr(cls, field.name, param_obj)
        # register the parameter so that Parameterized picks it up
        # cls._add_parameter(param_obj)
        # cls.param.add_parameter(curr_param_class_var_name, param_obj)
        cls.param.add_parameter(field.name, param_obj)

        # cls._add_parameter(
        # getattr(cls, curr_param_class_var_name, None)
        

    return cls


# def attrs_to_parameters_container(cls):
#     """ all fields should be `param.Parameterized` subclasses """
#     for field in attrs.fields(cls):
#         field_type = field.type
#         default = field.default if field.default is not attrs.NOTHING else field_type()
#         setattr(cls, field.name, param.ClassSelector(class_=field_type, default=default))
#     return cls

@function_attributes(short_name=None, tags=['parameters', 'attrs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-11 02:56', related_items=[])
def attrs_to_parameters_container(cls):
    """ all fields should be `param.Parameterized` subclasses """
    for field in attrs.fields(cls):
        if field.default is not attrs.NOTHING:
            default = field.default
            if isinstance(default, attrs.Factory):
                if default.takes_self:
                    raise ValueError("Factory with takes_self=True is not supported")
                default = default.factory()
        else:
            default = field.type()
        
        param_obj = param.ClassSelector(class_=field.type, default=default)
        
        # setattr(cls, field.name, param_obj)
        
        # set the parameter on the class under the same name as the field
        # setattr(cls, curr_param_class_var_name, param_obj)
        # setattr(cls, field.name, param_obj)
        # register the parameter so that Parameterized picks it up
        # cls._add_parameter(param_obj)
        # cls.param.add_parameter(curr_param_class_var_name, param_obj)
        cls.param.add_parameter(field.name, param_obj)
        
    return cls



# @attrs_to_parameters
class BaseGlobalComputationParameters(BaseConfig, param.Parameterized):
# class BaseGlobalComputationParameters(BaseConfig, BasePlotDataParams):
    """ Base class
    """
    # Overriding defaults from parent
    # name = param.String(default='BaseGlobalComputationParameters', doc='Name of the global computations')
    # isVisible = param.Boolean(default=False, doc="Whether the global computations widget is visible") # default to False    

    def __attrs_post_init__(self):
        param.Parameterized.__init__(self)
        

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
        """ renders only the field names and their values
        
        _out_str: str = param_typed_parameters.values_only_repr(attr_separator_str=",\n", sub_attr_additive_seperator_str='\t')
        print(_out_str)

        """
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
    

    # ==================================================================================================================== #
    # Serialization/Deserialization                                                                                        #
    # ==================================================================================================================== #

    @classmethod
    def from_state(cls, state):
        """ Rebuilds an instance using the latest class definition and updates state. """
        obj = cls.__new__(cls)  # Create a new instance without calling __init__
        obj.__setstate__(state)
        return obj


    # def _post_load_update(self):
    #     """ Validates and updates child fields. """
    #     # directional_decoders_decode_continuous_Parameters

    #     for field_name, field_value in self.__dict__.items():
    #         if isinstance(field_value, DynamicParameters):
    #             # Validate and update field value
    #             if not hasattr(field_value, 'should_disable_cache'):
    #                 print(f"Updating {field_name} to match current definition.")
    #                 field_value.should_disable_cache = False  # Default value or any appropriate fix



    def __setstate__(self, state):
        """
        #TODO 2025-01-07 14:06: - [ ] UNFINISHED - needs to handle missing fields like 'should_disable_cache' added to one of the params types
            => these result in an `AttributeError: 'directional_decoders_decode_continuous_Parameters' object has no attribute 'should_disable_cache' when trying to pickle again after unpickling (`to_dict(...)`)
        
         Restore instance attributes and update child fields if needed. """
        # Handle legacy format

        loaded_keys: List[str] = list(state.keys())
        modern_keys: List[str] = [a.name for a in self.__class__.__attrs_attrs__]

        added_keys: List[str] = [k for k in modern_keys if k not in loaded_keys]
        removed_keys: List[str] = [k for k in loaded_keys if k not in modern_keys]

        ## update with what we have:
        self.__dict__.update(state)

        if len(added_keys) > 0:
            print(f'\tadded_keys: {added_keys}')
            # Update missing attributes based on the current class definition
            for a in self.__class__.__attrs_attrs__:  # Access current class attributes
                attr_name: str = a.name
                # attr_field = a.field
                if attr_name in added_keys:
                    # Use the default factory if available, otherwise set the default value
                    if a.default is not None:
                        print(f'\t\tadding key: {attr_name}')
                        self.__dict__[attr_name] = a.default
                    # elif a.factory is not None:
                    #     self.__dict__[attr_name] = a.factory()
        # # Update missing attributes based on the current class definition
        # for a in self.__class__.__attrs_attrs__:  # Access current class attributes
        #     attr_name: str = a.name
        #     # attr_field = a.field
        #     if attr_name not in self.__dict__:
        #         # Use the default factory if available, otherwise set the default value
        #         if attr_field.default is not None:
        #             self.__dict__[attr_name] = a.default
        #         # elif attr_field.factory is not None:
        #         #     self.__dict__[attr_name] = attr_field.factory()

        print(f'\tdone.')

        # # Ensure child fields are updated
        # self._post_load_update()
        # self =  self.__class__.from_state(state=self.__dict__)


    @classmethod
    def get_class_param_Params_attribute_names(cls) -> List[str]:
        return [k for k in cls.param.values().keys() if k not in ['name']]
    
    def get_param_Params_attribute_names(self) -> List[str]:
        return [k for k in self.param.values().keys() if k not in ['name']]
        
        
    @classmethod
    def get_class_param_Params_dict(cls, param_name_excludeList=None) -> Dict:
        # param_name_excludeList = ['name']
        if param_name_excludeList is None:
            param_name_excludeList = []
        return {k:v for k, v in cls.param.values().items() if k not in param_name_excludeList}
    

    def to_params_dict(self, param_name_excludeList=None) -> Dict:
        """ returns as a dictionary representation """
        # param_name_excludeList = ['name']
        if param_name_excludeList is None:
            param_name_excludeList = []
        return {k:v for k, v in self.param.values().items() if k not in param_name_excludeList}
    


        

# ==================================================================================================================== #
# Specific Computation Parameter Objects (`BaseGlobalComputationParameters` subclasses):                               #
# ==================================================================================================================== #
""" generated programmatically by:

from neuropy.core.session.Formats.BaseDataSessionFormats import ParametersContainer
from neuropy.core.session.Formats.SessionSpecifications import SessionConfig
from neuropy.utils.indexing_helpers import flatten_dict
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, perform_specific_epochs_decoding_Parameters, DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters
from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating

registered_merged_computation_function_default_kwargs_dict, code_str, nested_classes_dict, (imports_dict, imports_list, imports_string) = GlobalComputationParametersAttrsClassTemplating.main_generate_params_classes(curr_active_pipeline=curr_active_pipeline)
print(code_str)


"""

@define(slots=False, eq=False, repr=False)
class merged_directional_placefields_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for merged_directional_placefields_Parameters. 
    """
    laps_decoding_time_bin_size: float = serialized_attribute_field(default=0.25)
    ripple_decoding_time_bin_size: float = serialized_attribute_field(default=0.025)
    should_validate_lap_decoding_performance: bool = serialized_attribute_field(default=False)
    ## PARAMS - these are class properties
    laps_decoding_time_bin_size_PARAM = param.Number(default=0.25, doc='laps_decoding_time_bin_size param', label='laps_decoding_time_bin_size')
    ripple_decoding_time_bin_size_PARAM = param.Number(default=0.025, doc='ripple_decoding_time_bin_size param', label='ripple_decoding_time_bin_size')
    should_validate_lap_decoding_performance_PARAM = param.Boolean(default=False, doc='should_validate_lap_decoding_performance param', label='should_validate_lap_decoding_performance')
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
    included_qclu_values: list = serialized_field(default=[1, 2, 4, 6, 7, 9])
    skip_laps: bool = serialized_attribute_field(default=False)
    ## PARAMS - these are class properties
    num_shuffles_PARAM = param.Integer(default=500, doc='num_shuffles param', label='num_shuffles')
    minimum_inclusion_fr_Hz_PARAM = param.Number(default=5.0, doc='minimum_inclusion_fr_Hz param', label='minimum_inclusion_fr_Hz')
    included_qclu_values_PARAM = param.List(default=[1, 2, 4, 6, 7, 9], doc='included_qclu_values param', label='included_qclu_values')
    skip_laps_PARAM = param.Boolean(default=False, doc='skip_laps param', label='skip_laps')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_decoders_decode_continuous_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_decoders_decode_continuous_Parameters. 
    """
    time_bin_size: Optional[float] = serialized_attribute_field(default=None)
    should_disable_cache: bool = serialized_attribute_field(default=False)
    ## PARAMS - these are class properties
    time_bin_size_PARAM = param.Number(default=None, doc='time_bin_size param', label='time_bin_size')
    should_disable_cache_PARAM = param.Boolean(default=False, doc='should_disable_cache param', label='should_disable_cache')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_decoders_evaluate_epochs_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_decoders_evaluate_epochs_Parameters. 
    """
    should_skip_radon_transform: bool = serialized_attribute_field(default=False)
    ## PARAMS - these are class properties
    should_skip_radon_transform_PARAM = param.Boolean(default=False, doc='should_skip_radon_transform param', label='should_skip_radon_transform')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_decoders_epoch_heuristic_scoring_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_decoders_epoch_heuristic_scoring_Parameters. 
    """
    same_thresh_fraction_of_track: float = serialized_attribute_field(default=0.05)
    max_ignore_bins: int = serialized_attribute_field(default=2)
    max_jump_distance_cm: float = serialized_attribute_field(default=60.0)
    use_bin_units_instead_of_realworld: bool = serialized_attribute_field(default=False)
    ## PARAMS - these are class properties
    same_thresh_fraction_of_track_PARAM = param.Number(default=0.05, doc='same_thresh_fraction_of_track param', label='same_thresh_fraction_of_track')
    max_ignore_bins_PARAM = param.Integer(default=2, doc='max_ignore_bins param', label='max_ignore_bins')
    max_jump_distance_cm_PARAM = param.Number(default=60.0, doc='max_jump_distance_cm param', label='max_jump_distance_cm')
    use_bin_units_instead_of_realworld_PARAM = param.Boolean(default=False, doc='use_bin_units_instead_of_realworld param', label='use_bin_units_instead_of_realworld')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class directional_train_test_split_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for directional_train_test_split_Parameters. 
    """
    training_data_portion: float = serialized_attribute_field(default=0.8333333333333334)
    debug_output_hdf5_file_path: Optional[pathlib.Path] = serialized_attribute_field(default=None)
    ## PARAMS - these are class properties
    training_data_portion_PARAM = param.Number(default=0.8333333333333334, doc='training_data_portion param', label='training_data_portion')
    debug_output_hdf5_file_path_PARAM = param.Path(default=None, doc='debug_output_hdf5_file_path param', label='debug_output_hdf5_file_path')
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
    ## PARAMS - these are class properties
    decoding_time_bin_size_PARAM = param.Number(default=None, doc='decoding_time_bin_size param', label='decoding_time_bin_size')
    perform_cache_load_PARAM = param.Boolean(default=False, doc='perform_cache_load param', label='perform_cache_load')
    always_recompute_replays_PARAM = param.Boolean(default=False, doc='always_recompute_replays param', label='always_recompute_replays')
    override_long_epoch_name_PARAM = param.String(default=None, doc='override_long_epoch_name param', label='override_long_epoch_name')
    override_short_epoch_name_PARAM = param.String(default=None, doc='override_short_epoch_name param', label='override_short_epoch_name')
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
    ## PARAMS - these are class properties
    decoding_time_bin_size_PARAM = param.Number(default=None, doc='decoding_time_bin_size param', label='decoding_time_bin_size')
    perform_cache_load_PARAM = param.Boolean(default=False, doc='perform_cache_load param', label='perform_cache_load')
    always_recompute_replays_PARAM = param.Boolean(default=False, doc='always_recompute_replays param', label='always_recompute_replays')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class long_short_inst_spike_rate_groups_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for long_short_inst_spike_rate_groups_Parameters. 
    """
    instantaneous_time_bin_size_seconds: Optional[float] = serialized_attribute_field(default=0.01)
    ## PARAMS - these are class properties
    instantaneous_time_bin_size_seconds_PARAM = param.Number(default=0.01, doc='instantaneous_time_bin_size_seconds param', label='instantaneous_time_bin_size_seconds')
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
    ## PARAMS - these are class properties
    num_shuffles_PARAM = param.Integer(default=1024, doc='num_shuffles param', label='num_shuffles')
    drop_previous_result_and_compute_fresh_PARAM = param.Boolean(default=False, doc='drop_previous_result_and_compute_fresh param', label='drop_previous_result_and_compute_fresh')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class position_decoding_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for position_decoding_Parameters. 
    """
    override_decoding_time_bin_size: Optional[float] = serialized_attribute_field(default=None)
    ## PARAMS - these are class properties
    override_decoding_time_bin_size_PARAM = param.Number(default=None, doc='override_decoding_time_bin_size param', label='override_decoding_time_bin_size')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class perform_specific_epochs_decoding_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for perform_specific_epochs_decoding_Parameters. 
    """
    decoder_ndim: int = serialized_attribute_field(default=2)
    filter_epochs: str = serialized_attribute_field(default='ripple')
    decoding_time_bin_size: Optional[float] = serialized_attribute_field(default=0.02)
    ## PARAMS - these are class properties
    decoder_ndim_PARAM = param.Integer(default=2, doc='decoder_ndim param', label='decoder_ndim')
    filter_epochs_PARAM = param.String(default='ripple', doc='filter_epochs param', label='filter_epochs')
    decoding_time_bin_size_PARAM = param.Number(default=0.02, doc='decoding_time_bin_size param', label='decoding_time_bin_size')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)

@define(slots=False, eq=False, repr=False)
class DEP_ratemap_peaks_Parameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ Docstring for DEP_ratemap_peaks_Parameters. 
    """
    peak_score_inclusion_percent_threshold: float = serialized_attribute_field(default=0.25)
    ## PARAMS - these are class properties
    peak_score_inclusion_percent_threshold_PARAM = param.Number(default=0.25, doc='peak_score_inclusion_percent_threshold param', label='peak_score_inclusion_percent_threshold')
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
    ## PARAMS - these are class properties
    step_PARAM = param.Number(default=0.01, doc='step param', label='step')
    peak_height_multiplier_probe_levels_PARAM = param.Tuple(default=(0.5, 0.9), doc='peak_height_multiplier_probe_levels param', label='peak_height_multiplier_probe_levels')
    minimum_included_peak_height_PARAM = param.Number(default=0.2, doc='minimum_included_peak_height param', label='minimum_included_peak_height')
    uniform_blur_size_PARAM = param.Integer(default=3, doc='uniform_blur_size param', label='uniform_blur_size')
    gaussian_blur_sigma_PARAM = param.Integer(default=3, doc='gaussian_blur_sigma param', label='gaussian_blur_sigma')
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)


# ==================================================================================================================== #
# Main Class                                                                                                           #
# ==================================================================================================================== #

@attrs_to_parameters_container
@define(slots=False)
class ComputationKWargParameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
    """ The base class for computation parameter types. 
    
    Usage:
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, perform_specific_epochs_decoding_Parameters, DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

        allow_update_global_computation_config = True
        ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
        if curr_active_pipeline.global_computation_results.computation_config is None:
            curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            if allow_update_global_computation_config:
                print('global_computation_results.computation_config is None! Making new one!')
                curr_active_pipeline.global_computation_results.computation_config = curr_global_param_typed_parameters
                print(f'\tdone. Pipeline needs resave!')
        else:
            curr_global_param_typed_parameters: ComputationKWargParameters = curr_active_pipeline.global_computation_results.computation_config
            
        ## OUTPUTS: curr_global_param_typed_parameters
        import param
        import panel as pn
        pn.extension()
        
        out_configs_dict = curr_global_param_typed_parameters.to_params_dict(recursive_to_dict=False)
        pn.Column(*[pn.Param(a_sub_v) for a_sub_v in reversed(out_configs_dict.values())])

    """
    merged_directional_placefields: merged_directional_placefields_Parameters = serialized_field(default=Factory(merged_directional_placefields_Parameters))	
    rank_order_shuffle_analysis: rank_order_shuffle_analysis_Parameters = serialized_field(default=Factory(rank_order_shuffle_analysis_Parameters))	
    directional_decoders_decode_continuous: directional_decoders_decode_continuous_Parameters = serialized_field(default=Factory(directional_decoders_decode_continuous_Parameters))	
    directional_decoders_evaluate_epochs: directional_decoders_evaluate_epochs_Parameters = serialized_field(default=Factory(directional_decoders_evaluate_epochs_Parameters))	
    directional_decoders_epoch_heuristic_scoring: directional_decoders_epoch_heuristic_scoring_Parameters = serialized_field(default=Factory(directional_decoders_epoch_heuristic_scoring_Parameters))	
    directional_train_test_split: directional_train_test_split_Parameters = serialized_field(default=Factory(directional_train_test_split_Parameters))	
    long_short_decoding_analyses: long_short_decoding_analyses_Parameters = serialized_field(default=Factory(long_short_decoding_analyses_Parameters))	
    long_short_rate_remapping: long_short_rate_remapping_Parameters = serialized_field(default=Factory(long_short_rate_remapping_Parameters))	
    long_short_inst_spike_rate_groups: long_short_inst_spike_rate_groups_Parameters = serialized_field(default=Factory(long_short_inst_spike_rate_groups_Parameters))	
    wcorr_shuffle_analysis: wcorr_shuffle_analysis_Parameters = serialized_field(default=Factory(wcorr_shuffle_analysis_Parameters))	
    position_decoding: position_decoding_Parameters = serialized_field(default=Factory(position_decoding_Parameters))	
    perform_specific_epochs_decoding: perform_specific_epochs_decoding_Parameters = serialized_field(default=Factory(perform_specific_epochs_decoding_Parameters))	
    DEP_ratemap_peaks: DEP_ratemap_peaks_Parameters = serialized_field(default=Factory(DEP_ratemap_peaks_Parameters))	
    ratemap_peaks_prominence2d: ratemap_peaks_prominence2d_Parameters = serialized_field(default=Factory(ratemap_peaks_prominence2d_Parameters))
    
    @classmethod
    def init_from_pipeline(cls, curr_active_pipeline):
        """ Initializes an instance from the pipeline object.
        
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
        
        # ==================================================================================================================== #
        # MANUAL                                                                                                               #
        # ==================================================================================================================== #
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
            a_type = None
            final_key: str = k.removeprefix('_')
            try:
                a_type = params_class_type_dict[final_key] # KeyError: 'directional_decoders_epoch_heuristic_scoring'
                _out_param_typed_parameters_dict[final_key] = a_type(**v_dict)
            
            except Exception as e:
                print(f'k: {k}, final_key: {final_key}, v_dict: {v_dict}')
                print(f'\ta_type: {a_type}')
                raise

            # a_type = params_class_type_dict[k]
            # _out_param_typed_parameters_dict[k.removeprefix('_')] = a_type(**v_dict)
        # _out_param_typed_parameters_dict

        ## OUTPUTS: _out_param_typed_parameters_dict
        # param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
        param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
        return param_typed_parameters
    
        # # ==================================================================================================================== #
        # # AUTOGEN                                                                                                              #
        # # ==================================================================================================================== #
        # params_class_type_dict = deepcopy(cls.__annotations__)
        # registered_merged_computation_function_default_kwargs_dict = (
        #     GlobalComputationParametersAttrsClassTemplating.main_extract_params_default_values(curr_active_pipeline=curr_active_pipeline)
        # )
        # ignore_kwarg_names = GlobalComputationParametersAttrsClassTemplating.ignore_kwarg_names
        # registered_merged_computation_function_default_kwargs_dict = {
        #     fn_best_name(v): get_fn_kwargs_with_defaults(v, ignore_kwarg_names=ignore_kwarg_names)
        #     for k, v in curr_active_pipeline.registered_merged_computation_function_dict.items()
        # }
        # registered_merged_computation_function_default_kwargs_dict = {
        #     k: v for k, v in registered_merged_computation_function_default_kwargs_dict.items() if len(v) > 0
        # }
        # _out_param_typed_parameters_dict = {}
        # for k, v_dict in registered_merged_computation_function_default_kwargs_dict.items():
        #     final_key = k.removeprefix('_')
        #     try:
        #         a_type = params_class_type_dict[final_key]
        #         _out_param_typed_parameters_dict[final_key] = a_type(**v_dict)
        #     except Exception as e:
        #         print(f"Error initializing field: {k}, {final_key}, {v_dict}, {e}")
        #         raise
        # return cls(**_out_param_typed_parameters_dict)


    def __attrs_post_init__(self):
        param.Parameterized.__init__(self)
        

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object and its fields to the specified HDF5 file. """
        # get_serialized_dataset_fields = self.get_serialized_dataset_fields(serialization_format='hdf')
        self.merged_directional_placefields.to_hdf(file_path, key=f"{key}/merged_directional_placefields")
        self.rank_order_shuffle_analysis.to_hdf(file_path, key=f"{key}/rank_order_shuffle_analysis")
        self.directional_decoders_decode_continuous.to_hdf(file_path, key=f"{key}/directional_decoders_decode_continuous")
        self.directional_decoders_evaluate_epochs.to_hdf(file_path, key=f"{key}/directional_decoders_evaluate_epochs")
        self.directional_decoders_epoch_heuristic_scoring.to_hdf(file_path, key=f"{key}/directional_decoders_epoch_heuristic_scoring")
        self.directional_train_test_split.to_hdf(file_path, key=f"{key}/directional_train_test_split")
        self.long_short_decoding_analyses.to_hdf(file_path, key=f"{key}/long_short_decoding_analyses")
        self.long_short_rate_remapping.to_hdf(file_path, key=f"{key}/long_short_rate_remapping")
        self.long_short_inst_spike_rate_groups.to_hdf(file_path, key=f"{key}/long_short_inst_spike_rate_groups")
        self.wcorr_shuffle_analysis.to_hdf(file_path, key=f"{key}/wcorr_shuffle_analysis")
        self.position_decoding.to_hdf(file_path, key=f"{key}/position_decoding")
        self.perform_specific_epochs_decoding.to_hdf(file_path, key=f"{key}/perform_specific_epochs_decoding")
        self.DEP_ratemap_peaks.to_hdf(file_path, key=f"{key}/DEP_ratemap_peaks")
        self.ratemap_peaks_prominence2d.to_hdf(file_path, key=f"{key}/ratemap_peaks_prominence2d")
        # super().to_hdf(file_path, key=key, **kwargs)


    # ==================================================================================================================== #
    # Params Overrides                                                                                                     #
    # ==================================================================================================================== #
    def get_param_Params_attribute_names(self) -> List[str]:
        return [k for k in self.param.values().keys() if k not in ['name']]
        

    def to_params_dict(self, param_name_excludeList=None, recursive_to_dict: bool=False) -> Dict:
        """ overrides to provide recurrsive implementation
        returns as a dictionary representation 
        
        Working:
        
            out_configs_dict = curr_global_param_typed_parameters.to_params_dict(recursive_to_dict=False)
            pn.Column(*[pn.Param(a_sub_v) for a_sub_v in reversed(out_configs_dict.values())])

        """
        # param_name_excludeList = ['name']
        if param_name_excludeList is None:
            param_name_excludeList = ['name']
        if recursive_to_dict:
            return {k:v.to_params_dict(param_name_excludeList=param_name_excludeList) for k, v in self.param.values().items() if k not in param_name_excludeList}
        else:
            return {k:v.param for k, v in self.param.values().items() if k not in param_name_excludeList}



# @attrs_to_parameters ## added manually
# @define(slots=False)
# class ComputationKWargParameters(HDF_SerializationMixin, AttrsBasedClassHelperMixin, BaseGlobalComputationParameters):
#     """ the base class
    
#     from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, _perform_specific_epochs_decoding_Parameters, _DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

    
#     """
#     merged_directional_placefields: merged_directional_placefields_Parameters = serialized_field(default=Factory(merged_directional_placefields_Parameters))
#     rank_order_shuffle_analysis: rank_order_shuffle_analysis_Parameters = serialized_field(default=Factory(rank_order_shuffle_analysis_Parameters))
#     directional_decoders_decode_continuous: directional_decoders_decode_continuous_Parameters = serialized_field(default=Factory(directional_decoders_decode_continuous_Parameters))
#     directional_decoders_epoch_heuristic_scoring: directional_decoders_epoch_heuristic_scoring_Parameters = serialized_field(default=Factory(directional_decoders_epoch_heuristic_scoring_Parameters))
#     directional_decoders_evaluate_epochs: directional_decoders_evaluate_epochs_Parameters = serialized_field(default=Factory(directional_decoders_evaluate_epochs_Parameters))
#     directional_train_test_split: directional_train_test_split_Parameters = serialized_field(default=Factory(directional_train_test_split_Parameters))
#     long_short_decoding_analyses: long_short_decoding_analyses_Parameters = serialized_field(default=Factory(long_short_decoding_analyses_Parameters))
#     long_short_rate_remapping: long_short_rate_remapping_Parameters = serialized_field(default=Factory(long_short_rate_remapping_Parameters))
#     long_short_inst_spike_rate_groups: long_short_inst_spike_rate_groups_Parameters = serialized_field(default=Factory(long_short_inst_spike_rate_groups_Parameters))
#     wcorr_shuffle_analysis: wcorr_shuffle_analysis_Parameters = serialized_field(default=Factory(wcorr_shuffle_analysis_Parameters))
#     perform_specific_epochs_decoding: perform_specific_epochs_decoding_Parameters = serialized_field(default=Factory(perform_specific_epochs_decoding_Parameters))
#     DEP_ratemap_peaks: DEP_ratemap_peaks_Parameters = serialized_field(default=Factory(DEP_ratemap_peaks_Parameters))
#     ratemap_peaks_prominence2d: ratemap_peaks_prominence2d_Parameters = serialized_field(default=Factory(ratemap_peaks_prominence2d_Parameters))

#     @classmethod
#     def init_from_pipeline(cls, curr_active_pipeline):
#         """ 
#         param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)

#         Usage:        
#             from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters

#             ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
#             if curr_active_pipeline.global_computation_results.computation_config is None:
#                 print('global_computation_results.computation_config is None! Making new one!')
#                 curr_active_pipeline.global_computation_results.computation_config = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
#                 print(f'\tdone. Pipeline needs resave!')
                
            
#         """
#         from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
        
#         registered_merged_computation_function_default_kwargs_dict = GlobalComputationParametersAttrsClassTemplating.main_extract_params_default_values(curr_active_pipeline=curr_active_pipeline)
        
#         ignore_kwarg_names = GlobalComputationParametersAttrsClassTemplating.ignore_kwarg_names
#         registered_merged_computation_function_default_kwargs_dict = {fn_best_name(v):get_fn_kwargs_with_defaults(v, ignore_kwarg_names=ignore_kwarg_names) for k, v in curr_active_pipeline.registered_merged_computation_function_dict.items()}
#         registered_merged_computation_function_default_kwargs_dict = {k:v for k, v in registered_merged_computation_function_default_kwargs_dict.items() if len(v)>0} # filter empty lists
        
#         # params_class_type_list = [merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, perform_specific_epochs_decoding_Parameters, DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters]
#         # params_class_type_dict = dict(zip({k.removeprefix('_') for k in imports_dict.keys()}, params_class_type_list))
#         # params_class_type_dict = dict(zip({k for k in imports_dict.keys()}, params_class_type_list))
#         # params_class_type_dict = dict(zip(imports_list, params_class_type_list))
#         # params_class_type_dict
        
#         params_class_type_dict = deepcopy(cls.__annotations__)
        
#         ## Convert to the new native types
#         ## INPUTS: registered_merged_computation_function_default_kwargs_dict, params_class_type_dict
#         _out_param_typed_parameters_dict = {}
#         for k, v_dict in registered_merged_computation_function_default_kwargs_dict.items():
#             a_type = None
#             final_key: str = k.removeprefix('_')
#             try:
#                 a_type = params_class_type_dict[final_key] # KeyError: 'directional_decoders_epoch_heuristic_scoring'
#                 _out_param_typed_parameters_dict[final_key] = a_type(**v_dict)
            
#             except Exception as e:
#                 print(f'k: {k}, final_key: {final_key}, v_dict: {v_dict}')
#                 print(f'\ta_type: {a_type}')
#                 raise

#             # a_type = params_class_type_dict[k]
#             # _out_param_typed_parameters_dict[k.removeprefix('_')] = a_type(**v_dict)
#         # _out_param_typed_parameters_dict

#         ## OUTPUTS: _out_param_typed_parameters_dict
#         # param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
#         param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters(**_out_param_typed_parameters_dict)
#         return param_typed_parameters

#         ## OUTPUTS: param_typed_parameters


#     # HDFMixin Conformances ______________________________________________________________________________________________ #
#     def to_hdf(self, file_path, key: str, **kwargs):
#         """ Saves the object to key in the hdf5 file specified by file_path"""
#         # get_serialized_dataset_fields = self.get_serialized_dataset_fields(serialization_format='hdf')
#         # self.merged_directional_placefields.to_hdf(file_path=file_path, key=f"{key}/merged_directional_placefields")
#         # self.rank_order_shuffle_analysis.to_hdf(file_path=file_path, key=f"{key}/rank_order_shuffle_analysis")
#         self.merged_directional_placefields.to_hdf(file_path, key=f'{key}/merged_directional_placefields')
#         self.rank_order_shuffle_analysis.to_hdf(file_path, key=f'{key}/rank_order_shuffle_analysis')
#         self.directional_decoders_decode_continuous.to_hdf(file_path=file_path, key=f"{key}/directional_decoders_decode_continuous")
#         self.directional_decoders_evaluate_epochs.to_hdf(file_path=file_path, key=f"{key}/directional_decoders_evaluate_epochs")
#         self.directional_train_test_split.to_hdf(file_path, key=f'{key}/directional_train_test_split')
#         self.long_short_decoding_analyses.to_hdf(file_path, key=f'{key}/long_short_decoding_analyses')
#         self.long_short_rate_remapping.to_hdf(file_path, key=f'{key}/long_short_rate_remapping')
#         self.long_short_inst_spike_rate_groups.to_hdf(file_path, key=f'{key}/long_short_inst_spike_rate_groups')
#         self.wcorr_shuffle_analysis.to_hdf(file_path, key=f'{key}/wcorr_shuffle_analysis')
#         self.perform_specific_epochs_decoding.to_hdf(file_path, key=f'{key}/perform_specific_epochs_decoding')
#         self.DEP_ratemap_peaks.to_hdf(file_path, key=f'{key}/DEP_ratemap_peaks')
#         self.ratemap_peaks_prominence2d.to_hdf(file_path, key=f'{key}/ratemap_peaks_prominence2d')
#         # super().to_hdf(file_path, key=key, **kwargs)
        