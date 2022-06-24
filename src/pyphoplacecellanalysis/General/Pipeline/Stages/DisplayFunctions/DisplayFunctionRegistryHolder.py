from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class DisplayFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "DisplayFunctionRegistryHolder"] = {}
    
# class BaseRegisteredDisplayClass(metaclass=DisplayFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# DisplayFunctionRegistryHolder.get_registry()
