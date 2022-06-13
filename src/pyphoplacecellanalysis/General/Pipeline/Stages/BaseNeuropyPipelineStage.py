

from dataclasses import dataclass

from pyphocorehelpers.DataStructure.enum_helpers import OrderedEnum

class PipelineStage(OrderedEnum):
    """ The active stage of the pipeline. """
    Input = 0
    Loaded = 1
    Filtered = 2
    Computed = 3
    Displayed = 4
    

@dataclass
class BaseNeuropyPipelineStage(object):
    """ BaseNeuropyPipelineStage represents a single stage of a data session processing/rendering pipeline. """
    stage_name: str = ""
    # pre_main_functions: List[Callable] = dataclasses.field(default_factory=list) # """ pre_main_functions are functions that are all called prior to the main_function evaluation. """
    # main_function: Callable = None # """ main_function is the function that constitutes the bulk of the action for this stage. """
    # post_main_functions: List[Callable] = dataclasses.field(default_factory=list) # """ post_main_functions are functions that are all called prior to the main_function evaluation. """
    identity: PipelineStage = PipelineStage.Input


