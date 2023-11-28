# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
from dataclasses import dataclass
import sys
import typing
from typing import Optional, Dict, Any

from attrs import define, field, Factory, asdict # used for `ComputedResult`

import numpy as np
from neuropy import core
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr
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
class ComputationResult(HDF_SerializationMixin):
    """
        The result of a single computation, on a filtered session with a specified config 
        The primary output data is stored in self.computed_data's dict
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
    is_global: bool = field(default=False)



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


