from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class ComputationFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "ComputationFunctionRegistryHolder"] = {}
    
    @classmethod
    def get_registry(cls):
        """ ensures that registry items are returned sorted by their ._computationPrecidence """
        # return dict(cls.REGISTRY)
        return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))
    
    
# class BaseRegisteredDisplayClass(metaclass=ComputationFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# ComputationFunctionRegistryHolder.get_registry()
