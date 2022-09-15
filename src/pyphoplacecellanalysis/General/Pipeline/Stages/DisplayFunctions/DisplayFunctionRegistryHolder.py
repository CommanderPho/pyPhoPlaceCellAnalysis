from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class DisplayFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "DisplayFunctionRegistryHolder"] = {}
    
