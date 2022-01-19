

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




# Class for parameter filtering:



# class BaseNeuropyPipelineUserRegisterableFunctionStage:
#     """ Analygous to ComputedPipelineStage """
    
#     # pre_main_functions: List[Callable] = dataclasses.field(default_factory=list) # """ pre_main_functions are functions that are all called prior to the main_function evaluation. """
#     # main_function: Callable = None # """ main_function is the function that constitutes the bulk of the action for this stage. """
#     # post_main_functions: List[Callable] = dataclasses.field(default_factory=list) # """ post_main_functions are functions that are all called prior to the main_function evaluation. """
    
    
#     def register_computation(self, computation_function):
#         self.registered_computation_functions.append(computation_function)
        
#     def perform_registered_computations(self, previous_computation_result, debug_print=False):
#         """ Called after load is complete to post-process the data """
#         if (len(self.registered_computation_functions) > 0):
#             if debug_print:
#                 print(f'Performing perform_registered_computations(...) with {len(self.registered_computation_functions)} registered_computation_functions...')            
#             composed_registered_computations_function = compose_functions(*self.registered_computation_functions) # functions are composed left-to-right
#             previous_computation_result = composed_registered_computations_function(previous_computation_result)
#             return previous_computation_result
            
#         else:
#             if debug_print:
#                 print(f'No registered_computation_functions, skipping extended computations.')
#             return previous_computation_result # just return the unaltered result
    
    
# class BaseUserRegisterableFunctionPipelineMixin:
#     """ Analygous to PipelineWithComputedPipelineStageMixin """
    
#     ## Computation Helpers: 
#     def perform_computations(self, active_computation_params: PlacefieldComputationParameters):     
#         assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
#         self.stage.single_computation(active_computation_params)
        
#     def register_computation(self, computation_function):
#         assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
#         self.stage.register_computation(computation_function)

#     def perform_registered_computations(self, previous_computation_result, debug_print=False):
#         assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.perform_computations to reach this step."
#         self.stage.perform_registered_computations()
    
    