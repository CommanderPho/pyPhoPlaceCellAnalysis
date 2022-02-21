import numpy as np

class KeysListAccessingMixin:
    
    @classmethod
    def get_keys_list(cls, data):
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print("get_keys_list(data): Unknown data type:", type(data), data)
            raise
        return keys