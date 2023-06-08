# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
from dataclasses import dataclass
import sys
import typing
from typing import Optional

from attrs import define, field # used for `ComputedResult`

import numpy as np
from neuropy import core
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

## Import with: from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

class ComputationResult(DynamicParameters):
    """
        The result of a single computation, on a filtered session with a specified config 
        The primary output data is stored in self.computed_data's dict
    """
    sess: DataSession
    computation_config: Optional[DynamicParameters]
    computed_data: Optional[DynamicParameters]
    accumulated_errors: Optional[DynamicParameters]

    def __init__(self, sess: DataSession, computation_config: DynamicParameters, computed_data: DynamicParameters, accumulated_errors: Optional[DynamicParameters]=None):
        if accumulated_errors is None:
            accumulated_errors = DynamicParameters()
        super(ComputationResult, self).__init__(sess=sess, computation_config=computation_config, computed_data=computed_data, accumulated_errors=accumulated_errors)





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