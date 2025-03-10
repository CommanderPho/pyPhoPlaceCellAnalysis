from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from typing import NewType
import nptyping as ND
from nptyping import NDArray
from neuropy.utils.type_aliases import aclu_index, DecoderName ## other imports?

""" Usage:

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
# import neuropy.utils.type_aliases as types
import pyphoplacecellanalysis.General.type_aliases as types


e.g. Dict[types.FilterContextName, Dict[types.ComputationFunctionName: CapturedException]]
"""

""" from `neuropy.utils.type_aliases`
	aclu_index: TypeAlias = int # an integer index that is an aclu
	DecoderName = NewType('DecoderName', str)
"""
FilterContextName: TypeAlias = str # a string identifier of a specific filtering context -- 'maze1_odd', 'maze2_odd', 'maze_odd' -- used in `.get_failed_computations(...)`
ComputationFunctionName: TypeAlias = str # a string identifier of a computation function -- can be either the long or short name -- '_split_to_directional_laps' -- used in `.get_failed_computations(...)`


# FilterContextName = NewType('FilterContextName', str) 
# ComputationFunctionName = NewType('ComputationFunctionName', str) 
