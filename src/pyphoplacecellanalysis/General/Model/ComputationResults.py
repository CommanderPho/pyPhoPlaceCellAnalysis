# NeuroPy (Diba Lab Python Repo) Loading
# import importlib

import sys
import typing
from typing import Optional

from attrs import define, field, Factory # used for `ComputedResult`

import numpy as np
from neuropy import core
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

## Import with: from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


# class ComputationResult(DynamicParameters):
#     """
#         The result of a single computation, on a filtered session with a specified config 
#         The primary output data is stored in self.computed_data's dict
#     """
#     sess: DataSession
#     computation_config: Optional[DynamicParameters]
#     computed_data: Optional[DynamicParameters]
#     accumulated_errors: Optional[DynamicParameters]

#     def __init__(self, sess: DataSession, computation_config: DynamicParameters, computed_data: DynamicParameters, accumulated_errors: Optional[DynamicParameters]=None):
#         if accumulated_errors is None:
#             accumulated_errors = DynamicParameters()
#         super(ComputationResult, self).__init__(sess=sess, computation_config=computation_config, computed_data=computed_data, accumulated_errors=accumulated_errors)



@define(slots=False, repr=False)
class ComputationResult:
    """ 2023-06-08 version of ComputationResult converting to an attrs class instead of a subclass of `DynamicParameters`.
            TODO 2023-06-08 - Might needs DynamicParameters based methods added for compatibility, such as .keys(), .get(...), etc.
                - Might need __getstate__, __setstate__
                - Pretty sure it needs __hash__(...) from DynamicParameters
            
        The result of a single computation, on a filtered session with a specified config 
        The primary output data is stored in self.computed_data's dict
    """
    sess: DataSession
    computation_config: Optional[DynamicParameters] = Factory(DynamicParameters)
    computed_data: Optional[DynamicParameters] = Factory(DynamicParameters)
    accumulated_errors: Optional[DynamicParameters] = Factory(DynamicParameters)
    computation_time: Optional[DynamicParameters] = Factory(DynamicParameters)


    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()

    # Enables subscriptability.
    def __getitem__(self, key):
        return self.__dict__[key] #@IgnoreException

    def __delitem__(self, key):
        del self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"

    # Extra/Extended
    def __dir__(self):
        return self.keys()

    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        # return hash((self.age, self.name))
        member_names_tuple = list(self.keys())
        values_tuple = list(self.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)


    # # For diffable parameters:
    # def diff(self, other_object):
    #     return DiffableObject.compute_diff(self, other_object)

    # def to_dict(self):
    #     return dict(self.items())
        
    # # Helper initialization methods:    
    # # For initialization from a different dictionary-backed object:
    # @classmethod
    # def init_from_dict(cls, a_dict):
    #     return cls(**a_dict) # expand the dict as input args.
    
    # @classmethod
    # def init_from_object(cls, an_object):
    #     # test to see if the object is dict-backed:
    #     obj_dict_rep = an_object.__dict__
    #     return cls.init_from_dict(obj_dict_rep)
    
    
    # # ## For serialization/pickling:
    # # def __getstate__(self):
    # #     return self.to_dict()
    # #     # return self.father, self.var1

    # # def __setstate__(self, state):
    # #     return self.init_from_dict(state)
    # #     # self.father, self.var1 = state
    
    
    # ## For serialization/pickling:
    # def __getstate__(self):
    #     # Copy the object's state from self.__dict__ which contains
    #     # all our instance attributes (__dict__ and _keys_at_init). Always use the dict.copy()
    #     # method to avoid modifying the original state.
    #     state = self.__dict__.copy()
    #     # Remove the unpicklable entries.
    #     # del state['file']
    #     return state

    # def __setstate__(self, state):
    #     # Restore instance attributes (i.e., __dict__ and _keys_at_init).
    #     self.__dict__.update(state)



        


@define(slots=False, repr=False)
class ComputedResult:
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
    is_global: bool = False