from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class FilterFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "FilterFunctionRegistryHolder"] = {}
    

""" FilterPipelineStage

	self.registered_filter_function_dict = OrderedDict()
	self.register_default_known_filter_functions() # registers the default filter functions
    
    @property
    def registered_filter_functions(self):
        return list(self.registered_filter_function_dict.values()) 
        
    @property
    def registered_filter_function_names(self):
        return list(self.registered_filter_function_dict.keys()) 
    
    def register_default_known_filter_functions(self):
        for (a_filter_class_name, a_filter_class) in FilterFunctionRegistryHolder.get_registry().items():
            for (a_filter_fn_name, a_filter_fn) in a_filter_class.get_all_functions(use_definition_order=False):
                self.register_filter_function(a_filter_fn_name, a_filter_fn)

    def reload_default_filter_functions(self):
        self.register_default_known_filter_functions()
        
        
    def register_filter_function(self, registered_name, filter_function):
        self.registered_filter_function_dict[registered_name] = filter_function
        
        
"""

""" PipelineWithFilterPipelineStageMixin

    ## *_functions
    @property
    def registered_filter_functions(self):
        return self.stage.registered_filter_functions
        
    @property
    def registered_filter_function_names(self):
        return self.stage.registered_filter_function_names
    
    @property
    def registered_filter_function_dict(self):
        return self.stage.registered_filter_function_dict
    
    @property
    def registered_filter_function_docs_dict(self):
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_filter_function_dict.items()}
    
    def register_filter_function(self, registered_name, filter_function):
        # assert (self.can_filter), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_filter_function(registered_name, filter_function)
        
    def reload_default_filter_functions(self):
        self.stage.reload_default_filter_functions()


"""