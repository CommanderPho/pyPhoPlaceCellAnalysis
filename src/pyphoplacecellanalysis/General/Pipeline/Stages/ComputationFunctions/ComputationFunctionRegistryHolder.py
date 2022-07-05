from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class ComputationFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "ComputationFunctionRegistryHolder"] = {}
    
    @classmethod
    def get_registry(cls):
        """ ensures that registry items are returned sorted by their ._computationPrecidence """
        return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))
    
    @classmethod
    def get_registry_items_functions_dict(cls, absolute_flat_path_keys=False):
        out_dict = {}
        for (a_computation_class_name, a_computation_class) in reversed(cls.get_registry().items()):
            if not absolute_flat_path_keys:
                curr_class_relative_out_dict = {}
            
            for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
                if not absolute_flat_path_keys:
                    # Relative dict mode:
                    curr_class_relative_out_dict[a_computation_fn_name] = a_computation_fn

                else:
                    curr_absolute_path_list = [a_computation_class_name, a_computation_fn_name]
                    curr_absolute_path_key = '.'.join(curr_absolute_path_list)
                    out_dict[curr_absolute_path_key] = a_computation_fn # absolute path
                    
            if not absolute_flat_path_keys:
                out_dict[a_computation_class_name] = curr_class_relative_out_dict
            
        return out_dict
    
    
# class BaseRegisteredDisplayClass(metaclass=ComputationFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# ComputationFunctionRegistryHolder.get_registry()
