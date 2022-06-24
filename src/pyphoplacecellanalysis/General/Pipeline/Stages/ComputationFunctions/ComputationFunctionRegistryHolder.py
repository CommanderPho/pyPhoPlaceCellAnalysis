from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class ComputationFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "ComputationFunctionRegistryHolder"] = {}
    
# class BaseRegisteredDisplayClass(metaclass=ComputationFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# ComputationFunctionRegistryHolder.get_registry()
