from functools import wraps, partial
from copy import deepcopy
from datetime import datetime
from enum import Enum # for getting the current date to set the ouptut folder name
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Dict
import pandas as pd
import numpy as np

from attrs import define as original_define
from attrs import field, Factory, fields, fields_dict, asdict

""" 
from pyphoplacecellanalysis.General.Mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, computed_field

"""

# ==================================================================================================================== #
# 2023-07-30 `attrs`-based classes Helper Mixin                                                                        #
# ==================================================================================================================== #
class AttrsBasedClassHelperMixin:
    """ heleprs for classes defined with `@define(slots=False, ...)` 
    
    from pyphoplacecellanalysis.General.Mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define


    hdf_fields = BasePositionDecoder.get_serialized_fields('hdf')

    """
    @classmethod
    def get_fields_with_tag(cls, tag:str='hdf', invert:bool=False) -> List:
        found_fields = []
        for attr_field in fields(cls):
            tags_metadata = attr_field.metadata.get('tags', [])
            if invert:
                query_condition = (tag not in tags_metadata)
            else:
                query_condition = (tag in tags_metadata)
            if query_condition:
                found_fields.append(attr_field.name)
        return found_fields


    @classmethod
    def get_serialized_fields(cls, serialization_format:str='hdf') -> List:
        hdf_fields = []
        for attr_field in fields(cls):
            serialization_metadata = attr_field.metadata.get('serialization', {})
            if serialization_metadata.get(serialization_format, False) is True:
                hdf_fields.append(attr_field.name)
        return hdf_fields



# ==================================================================================================================== #
# Custom `@define` that automatically makes class inherit from `AttrsBasedClassHelperMixin`                            #
# ==================================================================================================================== #


custom_define = partial(original_define, slots=False)

# def custom_define(slots=False, **kwargs):
#     """ replaces the `@define` for classes to cause the class to inherity from `AttrsBasedClassHelperMixin` automatically and use slots=False by default!
    
#     from pyphoplacecellanalysis.General.Mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define
#     @custom_define()
#     class AClass:
#         pass
    
#     """
#     mixin_cls = AttrsBasedClassHelperMixin

#     def wrap(cls):
#         # Apply the original attrs.define
#         new_cls = original_define(cls, slots=slots, **kwargs)

#         # If the class doesn't already inherit from the mixin, add it to its bases
#         if not issubclass(new_cls, mixin_cls):
#             new_cls.__bases__ = (mixin_cls,) + new_cls.__bases__

#         return new_cls

#     return wrap





# ==================================================================================================================== #
# Custom `field`s                                                                                                      #
# ==================================================================================================================== #

def merge_metadata(default_metadata: Dict[str, Any], additional_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if additional_metadata:
        for key, value in additional_metadata.items():
            if key in default_metadata and isinstance(default_metadata[key], dict):
                default_metadata[key].update(value)
            else:
                default_metadata[key] = value
    return default_metadata

def computed_field(default: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> field:
    default_metadata = {
        'tags': ['computed'],
        'serialization': {'hdf': False, 'csv': False, 'pkl': True}
    }
    return field(default=default, metadata=merge_metadata(default_metadata, metadata), **kwargs)

def serialized_field(default: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> field:
    default_metadata = {
        'serialization': {'hdf': True}
    }
    return field(default=default, metadata=merge_metadata(default_metadata, metadata), **kwargs)

"""
from pyphoplacecellanalysis.General.Mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, computed_field

"""


# def to_dict(self):
#     # Excluded from serialization: ['_included_thresh_neurons_indx', '_peak_frate_filter_function']
#     # filter_fn = filters.exclude(fields(PfND)._included_thresh_neurons_indx, int)
#     filter_fn = lambda attr, value: attr.name not in ["_included_thresh_neurons_indx", "_peak_frate_filter_function"]
#     return asdict(self, filter=filter_fn) # serialize using attrs.asdict but exclude the listed properties

# ==================================================================================================================== #
# 2023-06-22 13:24 `attrs` auto field exploration                                                                      #
# ==================================================================================================================== #

# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
# from attrs import asdict, fields, evolve

# ## For loop version:
# for a_field in fields(type(subset)):
# 	if 'n_epochs' in a_field.metadata.get('shape', ()):
# 		# is a field indexed by epochs
# 		print(a_field.name)
# 		print(a_field.value)

# # Find all fields that contain a 'n_neurons':
# epoch_indexed_attributes = [a_field for a_field in fields(type(subset)) if ('n_epochs' in a_field.metadata.get('shape', ()))]
# epoch_indexed_attributes

# # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
# epoch_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_epochs') for a_field in epoch_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_epochs index location
# epoch_shape_index_for_attribute_name_dict
# _temp_obj_dict = {k:v.take(indices=is_included_in_subset, axis=epoch_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_epochs axis containing items to get a reduced dictionary
# evolve(subset, **_temp_obj_dict)

# def sliced_by_aclus(self, aclus):
#     """ returns a copy of itself sliced by the aclus provided. """
#     from attrs import asdict, fields, evolve
#     aclu_is_included = np.isin(self.original_1D_decoder.neuron_IDs, aclus)  #.shape # (104, 63)
#     def _filter_obj_attribute(an_attr, attr_value):
#         """ return attributes only if they have n_neurons in their shape metadata """
#         return ('n_neurons' in an_attr.metadata.get('shape', ()))            
#     _temp_obj_dict = asdict(self, filter=_filter_obj_attribute)
#     # Find all fields that contain a 'n_neurons':
#     neuron_indexed_attributes = [a_field for a_field in fields(type(self)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
#     # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
#     neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
#     _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary
#     return evolve(self, **_temp_obj_dict)


# `attrs` object shape specifications, updating `LeaveOneOutDecodingAnalysisResult`
# from attrs import fields, fields_dict, asdict
# from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingAnalysisResult, TimebinnedNeuronActivity, LeaveOneOutDecodingResult

# LeaveOneOutDecodingAnalysisResult.__annotations__

# def _filter_obj_attribute(an_attr, attr_value):
# 	""" return attributes only if they have n_neurons in their shape metadata """
# 	return ('n_neurons' in an_attr.metadata.get('shape', ()))

# # Find all fields that contain a 'n_neurons':
# neuron_indexed_attributes = [a_field for a_field in fields(type(long_results_obj)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
# # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
# neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
# neuron_shape_index_for_attribute_name_dict
# shape_specifying_fields = {a_field.name:a_field.metadata.get('shape', None) for a_field in fields(type(long_results_obj)) if a_field.metadata.get('shape', None) is not None}
# shape_specifying_fields

# _temp_obj_dict = asdict(long_results_obj, filter=_filter_obj_attribute)
# _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary
