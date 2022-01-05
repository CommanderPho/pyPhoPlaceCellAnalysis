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





@dataclass
class ComputationResult(object):
    """
        The result of a single computation, on a filtered session with a specified config 
        The primary output data is stored in self.computed_data's dict
    """
    sess: DataSession
    computation_config: PlacefieldComputationParameters
    computed_data: dict
    


