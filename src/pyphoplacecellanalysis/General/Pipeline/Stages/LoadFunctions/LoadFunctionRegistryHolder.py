from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class LoadFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "LoadFunctionRegistryHolder"] = {}
    

""" LoadPipelineStage

	self.registered_computation_function_dict = computed_stage.registered_computation_function_dict
	self.registered_global_computation_function_dict = computed_stage.registered_global_computation_function_dict

	# Initialize custom fields:
	self.registered_load_function_dict = OrderedDict()
	self.register_default_known_load_functions() # registers the default load functions
	
    
    @property
    def registered_load_functions(self):
        return list(self.registered_load_function_dict.values()) 
        
    @property
    def registered_load_function_names(self):
        return list(self.registered_load_function_dict.keys()) 
    
    def register_default_known_load_functions(self):
        for (a_load_class_name, a_load_class) in LoadFunctionRegistryHolder.get_registry().items():
            for (a_load_fn_name, a_load_fn) in a_load_class.get_all_functions(use_definition_order=False):
                self.register_load_function(a_load_fn_name, a_load_fn)

    def reload_default_load_functions(self):
        self.register_default_known_load_functions()
        
        
    def register_load_function(self, registered_name, load_function):
        self.registered_load_function_dict[registered_name] = load_function
        
        
"""

""" PipelineWithLoadPipelineStageMixin

    ## *_functions
    @property
    def registered_load_functions(self):
        return self.stage.registered_load_functions
        
    @property
    def registered_load_function_names(self):
        return self.stage.registered_load_function_names
    
    @property
    def registered_load_function_dict(self):
        return self.stage.registered_load_function_dict
    
    @property
    def registered_load_function_docs_dict(self):
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_load_function_dict.items()}
    
    def register_load_function(self, registered_name, load_function):
        # assert (self.can_load), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_load_function(registered_name, load_function)
        
    def reload_default_load_functions(self):
        self.stage.reload_default_load_functions()


"""