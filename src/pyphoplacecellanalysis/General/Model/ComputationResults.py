# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
from dataclasses import dataclass
import sys
import typing
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime # for VersionedResultMixin

from attrs import define, field, Factory, asdict # used for `ComputedResult`

import numpy as np
from neuropy import core
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.mixins.gettable_mixin import GetAccessibleMixin
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr, SimpleFieldSizesReprMixin
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

## Import with: from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

""" #TODO 2023-08-21 17:36: - [ ] Dealing with Configurations:

# Universal Computation parameter specifications

# should be serializable to HDF
# should allow accessing properties by a hierarchical "grouping" structure

# ideally would associate parameters with the computations that use them (although this introduces coupling)

# should allow both "path" or "object-property (dot)" access
config['pf_params/speed_thresh'] == config['pf_params']['speed_thresh'] == config.pf_params.speed_thresh == config.pf_params['speed_thresh'] == config['pf_params'].speed_thresh

# should be able to hold Objects in addition to raw Python types (config.pf_params.computation_epochs = Epoch(...))

# must support type-hinting and ipython auto-completion

# ideally could be added from global function specification

@register_global_computation_parameter(..., instantaneous_time_bin_size_seconds: float = 0.01)
def _perform_long_short_instantaneous_spike_rate_groups_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
	print(global_computation_results.computation_config.instantaneous_time_bin_size_seconds)


computation_results: dict
│   ├── maze1: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── maze2: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── maze: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    
    
print_keys_if_possible("computation_results['maze'].computation_config", curr_active_pipeline.computation_results['maze'].computation_config, max_depth=3)

computation_results['maze'].computation_config: neuropy.utils.dynamic_container.DynamicContainer
│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
    │   ├── speed_thresh: float
    │   ├── grid_bin: tuple - (2,)
    │   ├── grid_bin_bounds: tuple - (2, 2)
    │   ├── smooth: tuple - (2,)
    │   ├── frate_thresh: float
    │   ├── time_bin_size: float
    │   ├── computation_epochs: neuropy.core.epoch.Epoch
        │   ├── _filename: NoneType
        │   ├── _metadata: NoneType
        │   ├── _df: pandas.core.frame.DataFrame (children omitted) - (80, 6)
│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── max_num_spikes_per_neuron: int
    │   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
        │   ├── s: int
        │   ├── gamma: float
    │   ├── use_progress_bar: bool
    │   ├── debug_print: bool
    
    
        
global_computation_results: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── sess: neuropy.core.session.dataSession.DataSession
│   ├── computation_config: NoneType
│   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters

    
"""




@custom_define(slots=False)
class ComputationResult(SimpleFieldSizesReprMixin, HDF_SerializationMixin, AttrsBasedClassHelperMixin, GetAccessibleMixin):
    """
        The result of a single computation, on a filtered session with a specified config 
        The primary output data is stored in self.computed_data's dict

        Known to be used for `curr_active_pipeline.global_computation_results`


        NOTE THAT DUE TO `AttrsBasedClassHelperMixin`, ALL `ComputationResult` subclasses MUST BE ATTRS BASED!

    """
    sess: DataSession = serialized_field(repr=True)
    computation_config: Optional[DynamicParameters] = serialized_field(default=None, is_computable=False, repr=True)
    computed_data: Optional[DynamicParameters] = serialized_field(default=None, repr=keys_only_repr)
    accumulated_errors: Optional[DynamicParameters] = non_serialized_field(default=Factory(DynamicParameters), is_computable=True, repr=keys_only_repr)
    computation_times: Optional[DynamicParameters] = serialized_field(default=Factory(DynamicParameters), is_computable=False, repr=keys_only_repr)
    
    ## For serialization/pickling:

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if ('_mapping' in state) and ('_keys_at_init' in state):
            # unpickling from the old DynamicParameters-based ComputationResult
            print(f'unpickling from old DynamicParameters-based computationResult')
            self.__dict__.update(state['_mapping'])
        else:
             # typical update
            self.__dict__.update(state)

    # LEGACY WORKAROUND __________________________________________________________________________________________________ #
    def to_dict(self):
        """ TEMPORARY WORK AROUND: workaround after conversion from DynamicParameters-based class. """
        return asdict(self)

    def __getitem__(self, key: str):
        """ TEMPORARY WORK AROUND: workaround after conversion from DynamicParameters-based class. """
        print(f'DEPRICATION WARNING: workaround to allow subscripting ComputationResult objects. Will be depricated. key: {key}')
        return getattr(self, key)


class VersionedResultMixin:
    """ Implementors keep track of the version of the class by which they are instantiated.

    Allows comparing the .result_version of a deseralized object to the current result version, and take actions (like adding/removing/changing fields or values as needed).

    `result_verion` is stored as a string like "2024.01.11_0" in the format "{YYYY_MM_DD_DATE_STR}_{VERSION_NUM}"
        the VERSION_NUM part is only used when the dates are equal (indicating multiple versions from the same day)
        
        
    Implementors typically have:
    
        result_version: str = serialized_attribute_field(default='2024.01.11_0', is_computable=False, repr=False) # this field specfies the version of the result. 

        
        def __setstate__(self, state):
            # Restore instance attributes (i.e., _mapping and _keys_at_init).

            result_version: str = state.get('result_version', None)
            if result_version is None:
                result_version = "2024.01.10_0"
                state['result_version'] = result_version # set result version

    """
    
    _VersionedResultMixin_version: str = "2024.01.01_0" # to be updated in your IMPLEMENTOR to indicate its version
    

    @classmethod
    def _VersionedResultMixin_parse_result_version_string(cls, v0_str: str) -> Tuple[datetime, int]:
        date_str, version_num = v0_str.split('_')
        date = datetime.strptime(date_str, '%Y.%m.%d')
        return date, int(version_num)


    @classmethod
    def _VersionedResultMixin_compare_result_version_strings(cls, v0_str: str, v1_str: str) -> bool:
        """
        returns True if v0_str preceeds v1_str
        """
        v0_date, v0_num = cls._VersionedResultMixin_parse_result_version_string(v0_str)
        v1_date, v1_num = cls._VersionedResultMixin_parse_result_version_string(v1_str)
        if (v0_date == v1_date):
            # the _num part is only used when the dates are equal (indicating multiple versions from the same day)
            return (v0_num < v1_num)
        else:
            return (v0_date < v1_date)


    def get_parsed_result_version(self) -> Tuple[datetime, int]:
        """ parses the result version from the string format to something comparable. """
        return VersionedResultMixin._VersionedResultMixin_parse_result_version_string(self.result_version)
    

    def is_result_version_earlier_than(self, v1: Union[str, Tuple[datetime, int], datetime]) -> bool:
        """
            returns True if self.result_version is earlier than a minimum `v1_str`
        """
        if isinstance(v1, str):
            return self._VersionedResultMixin_compare_result_version_strings(self.result_version, v1)
        elif isinstance(v1, Tuple):
            v0_date, v0_num = self.get_parsed_result_version()
            v1_date, v1_num = v1
            return (v0_date < v1_date) or (v0_date == v1_date and v0_num < v1_num)
        elif isinstance(v1, datetime):
            v0_date, v0_num = self.get_parsed_result_version()
            return v0_date < v1

    def is_result_version_newer_than(self, v1: Union[str, Tuple[datetime, int], datetime]) -> bool:
        """
            returns True if self.result_version is newer than the given `v1_str`
        """
        if isinstance(v1, str):
            return not self._VersionedResultMixin_compare_result_version_strings(self.result_version, v1) and self.result_version != v1
        elif isinstance(v1, Tuple):
            v0_date, v0_num = self.get_parsed_result_version()
            v1_date, v1_num = v1
            return (v0_date > v1_date) or (v0_date == v1_date and v0_num > v1_num)
        elif isinstance(v1, datetime):
            v0_date, v0_num = self.get_parsed_result_version()
            return v0_date > v1

    def is_result_version_equal_to(self, v1: Union[str, Tuple[datetime, int], datetime]) -> bool:
        """
            returns True if self.result_version is equal to the given `v1_str`
        """
        if isinstance(v1, str):
            return (self.result_version == v1)
        elif isinstance(v1, Tuple):
            v0_date, v0_num = self.get_parsed_result_version()
            v1_date, v1_num = v1
            return (v0_date == v1_date and v0_num == v1_num)
        elif isinstance(v1, datetime):
            v0_date, v0_num = self.get_parsed_result_version()
            return v0_date == v1

    def _VersionedResultMixin__setstate__(self, state):
        """ 
        Updates: self.__dict__['result_version']
        """
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        result_version: str = state.get('result_version', None)
        if result_version is None:
            result_version = "2024.01.01_0"
            state['result_version'] = result_version # set result version
            
        self.__dict__['result_version'] = result_version
        
        # don't call because the implementor should control this

        # self.__dict__.update(state) 
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(VersionedResultMixin, self).__init__()
        

@define(slots=False, repr=False)
class ComputedResult(SimpleFieldSizesReprMixin, VersionedResultMixin, HDFMixin, AttrsBasedClassHelperMixin, GetAccessibleMixin):
    """ 2023-05-10 - an object to replace DynamicContainers and static dicts for holding specific computed results
    
    Usage:
        from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

        @define(slots=False, repr=False)
        class LeaveOneOutDecodingAnalysis(ComputedResult):
            is_global: bool = True
            
            long_decoder: BayesianPlacemapPositionDecoder
            short_decoder: BayesianPlacemapPositionDecoder
            long_replays: pd.DataFrame
            short_replays: pd.DataFrame
            global_replays: pd.DataFrame
            long_shared_aclus_only_decoder: BasePositionDecoder
            short_shared_aclus_only_decoder: BasePositionDecoder
            shared_aclus: np.ndarray
            long_short_pf_neurons_diff: SetPartition
            n_neurons: int
            long_results_obj: LeaveOneOutDecodingAnalysisResult
            short_results_obj: LeaveOneOutDecodingAnalysisResult

            
    """
    _VersionedResultMixin_version: str = "2024.01.01_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    is_global: bool = non_serialized_field(default=False, repr=False, is_computable=True, kw_only=True) # init=False
    result_version: str = serialized_attribute_field(default='2024.01.01_0', repr=False, is_computable=False, kw_only=True) # this field specfies the version of the result. 
    
    # field(default=False, metadata={'is_hdf_handled_custom': True, 'serialization': {'hdf': False, 'csv': False, 'pkl': True}})

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """ note that the __setstate__ is NOT inherited by children! They have to implement their own __setstate__ or their self.__dict__ will be used instead.
        
        """
        # Restore instance attributes (i.e., _mapping and _keys_at_init).

        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        self.__dict__.update(state)

        ## Disabled 2024-05-28 22:11 
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(ComputedResult, self).__init__()




# @define(slots=False, repr=False)
# class ComputedResultsSet:
#     """ 2023-08-21 - an object to replace the DynamicContainers-based `ComputationResult` for holding all of the computed results (`ComputedResult` objects)

#     ['sess', 'computation_config', 'computed_data', 'accumulated_errors', 'computation_times']

#     Usage:
#         from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResultsSet

        
            
#     """
#     sess: DataSession
#     computation_config: Dict
#     computed_data: Dict[Any, ComputedResult]
#     accumulated_errors: Dict
#     computation_times: Dict


