from pathlib import Path
import pathlib
from typing import Union

# from neuropy.utils.mixins.dict_representable import DictRepresentable
# from neuropy.utils.mixins.file_representable import FileRepresentable

from pyphocorehelpers.Filesystem.path_helpers import ensure_pathlib_Path
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

    
class PickleSerializableMixin: # (FileRepresentable, DictRepresentable)
    """ Implementors can be easily serialized/deserialized from pickle
        
    Required Properties:
        None

    Provides:
        cls.from_file(cls, pkl_path: Union[str, Path])
        self.save(self, pkl_output_path: Union[Path, str], status_print=True)
                
    from pyphoplacecellanalysis.General.Mixins.PickleSerializableMixin import PickleSerializableMixin
    
    
    """
    # def __init__(self, metadata=None) -> None:

    #     self._filename = None

    #     if metadata is not None:
    #         assert isinstance(metadata, dict), "Only dictionary accepted as metadata"

    #     self._metadata: dict = metadata

    # @property
    # def filename(self):
    #     return self._filename

    # @filename.setter
    # def filename(self, f):
    #     assert isinstance(f, (str, Path))
    #     self._filename = f

    # @property
    # def metadata(self):
    #     return self._metadata

    # @metadata.setter
    # def metadata(self, d):
    #     """metadata compatibility"""
    #     if d is not None:
    #         assert isinstance(d, dict), "Only dictionary accepted"
    #         if self._metadata is not None:
    #             self._metadata = self._metadata | d # if we already have valid metadata, merge the dicts
    #         else:
    #             self._metadata = d # otherwise we can just set it directly


    # ## DictRepresentable protocol:
    # @classmethod
    # def from_dict(cls, state):
    #     # Restore instance attributes (i.e., _mapping and _keys_at_init).
    #     return cls(**state)

    # def to_dict(self, recurrsively=False):
    #     # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
    #     state = self.__dict__.copy()
    #     return state
    

    # ## For serialization/pickling:
    # def __getstate__(self):
    #     # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
    #     state = self.__dict__.copy()
    #     return state

    # def __setstate__(self, state):
    #     """ note that the __setstate__ is NOT inherited by children! They have to implement their own __setstate__ or their self.__dict__ will be used instead.        
    #     """
    #     self.__dict__.update(state)



    ## FileRepresentable protocol:
    @classmethod
    def from_file(cls, pkl_path: Union[str, Path]):
        pkl_path = ensure_pathlib_Path(pkl_path).resolve()
        assert pkl_path.exists()
        assert pkl_path.is_file()
        # dict_rep = None
        try:
            # dict_rep = np.load(pkl_path, allow_pickle=True).item()
            # dict_rep = loadData(pkl_path=pkl_path)
            return loadData(pkl_path=pkl_path)
            
            # return dict_rep
        except NotImplementedError:
            print("Issue with pickled POSIX_PATH on windows for path {}, falling back to non-pickled version...".format(pkl_path))
            temp = pathlib.PosixPath
            # pathlib.PosixPath = pathlib.WindowsPath # Bad hack
            pathlib.PosixPath = pathlib.PurePosixPath # Bad hack
            # dict_rep = np.load(pkl_path, allow_pickle=True).item()
            # dict_rep = loadData(pkl_path=pkl_path)
            return loadData(pkl_path=pkl_path)
        except Exception as e:
            raise        
        # if dict_rep is not None:
        #     # Convert to object
        #     try:
        #         obj = cls.from_dict(dict_rep)
        #     # except KeyError as e:
        #     #     # print(f'f: {f}, dict_rep: {dict_rep}')
        #     #     # Tries to load using any legacy methods defined in the class
        #     #     # obj = cls.legacy_from_dict(dict_rep)
        #     #     # raise e
        #     #     raise
        #     except Exception as e:
        #         raise
            
        #     # obj.filename = pkl_path
        #     return obj
        # return dict_rep


    # @classmethod
    # def to_file(cls, data: dict, pkl_output_path: Union[str, Path], status_print=True):
    #     if pkl_output_path is not None:
    #         pkl_output_path = ensure_pathlib_Path(pkl_output_path)
    #         ## perform saving:
    #         # np.save(f, data)
    #         saveData(pkl_output_path, data) ## save self
            
    #         if status_print:
    #             print(f"{pkl_output_path.name} saved")
    #     else:
    #         print("WARNING: filename can not be None")


    def save(self, pkl_output_path: Union[Path, str], status_print=True):
        pkl_output_path = ensure_pathlib_Path(pkl_output_path)
        if status_print:
            print(f'saving to pkl_output_path: "{pkl_output_path}"...')
        saveData(pkl_output_path, self) ## save self
        # data = self.to_dict()
        # self.to_file(data, pkl_output_path=pkl_output_path, status_print=status_print)
        if status_print:
            print('\tdone.')
