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


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(NeuropyPipeline, self).__init__() # from 
        

