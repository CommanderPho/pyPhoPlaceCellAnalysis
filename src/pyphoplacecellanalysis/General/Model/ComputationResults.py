# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
from dataclasses import dataclass
import sys
import typing
from typing import Optional


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



