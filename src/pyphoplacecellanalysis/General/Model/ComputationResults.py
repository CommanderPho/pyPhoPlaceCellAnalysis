# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
from dataclasses import dataclass
import sys

import numpy as np

try:
    from neuropy import core

    # importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print(
        "neuropy module not found, adding directory to sys.path. \n >> Updated sys.path."
    )
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
    computation_config: DynamicParameters
    computed_data: dict

    def __init__(self, sess: DataSession, computation_config: DynamicParameters, computed_data: dict): 
        super(ComputationResult, self).__init__(sess=sess, computation_config=computation_config, computed_data=computed_data)
            


