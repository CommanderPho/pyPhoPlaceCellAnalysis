from typing import Any, Callable, List, Dict, Optional, Union, Tuple, get_type_hints, get_origin, get_args
from types import ModuleType
import dataclasses
from dataclasses import dataclass
import attrs
from attrs import define, field, Factory, asdict, has, fields
from datetime import datetime
import pathlib
from pathlib import Path
import shutil
import nptyping as ND
from nptyping import NDArray
import numpy as np # for _backup_extant_file(...)

import pandas as pd

from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.function_helpers import compose_functions
from pyphocorehelpers.Filesystem.pickling_helpers import RenameUnpickler, renamed_load
from pyphocorehelpers.Filesystem.pickling_helpers import ModuleExcludesPickler, custom_dump


from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.LoadFunctions.LoadFunctionRegistryHolder import LoadFunctionRegistryHolder


# ==================================================================================================================== #
# Session Pickling for Loading/Saving                                                                                  #
# ==================================================================================================================== #
## Test using pickle to pickle the loaded session object after loading
# import pickle
## `dill` support: dill is a drop-in replacement for pickle. Existing code can be updated to allow complete pickling using
import dill as pickle # requires mamba install dill -c conda-forge

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter
from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
from pyphocorehelpers.Filesystem.path_helpers import build_unique_filename, backup_extant_file


def safeSaveData(pkl_path: Union[str, Path], db: Any, should_append:bool=False, backup_file_if_smaller_than_original:bool=False, backup_minimum_difference_MB:int=5, should_print_output_filesize:bool=True):
    """ saves the output data in a way that doesn't corrupt it if the pickling fails and the original file is retained.
    
    Saves `db` to a temporary pickle file (with the '.tmp' additional extension), which overwrites an existing pickle IFF saving completes successfully


    db: the data to be saved
    backup_file_if_smaller_than_original:bool - if True, creates a backup of the old file if the new file is smaller.
    backup_minimum_difference_MB:int = 5 # don't backup for an increase of 5MB or less, ignored unless backup_file_if_smaller_than_original==True
    """
    if not isinstance(pkl_path, Path):
        pkl_path = Path(pkl_path).resolve()
    if should_append:
        file_mode = 'ab' # 'ab' opens the file as binary and appends to the end
    else:
        file_mode = 'w+b' # 'w+b' opens and truncates the file to 0 bytes (overwritting)

    is_temporary_file_used:bool = False
    _desired_final_pickle_path = None
    if pkl_path.exists():
        # file already exists:
        ## Save under a temporary name in the same output directory, and then compare post-hoc
        _desired_final_pickle_path = pkl_path
        pkl_path, _ = build_unique_filename(pkl_path, additional_postfix_extension='tmp') # changes the final path to the temporary file created.
        is_temporary_file_used = True # this is the only condition where this is true
    else:
        # it doesn't exist so the final pickle path is the real one
        _desired_final_pickle_path = pkl_path            

    # Save reloaded pipeline out to pickle for future loading
    with ProgressMessagePrinter(filepath=_desired_final_pickle_path, action=f"Saving (file mode '{file_mode}')", contents_description='pickle file', finished_message='saved pickle file', returns_string=False):
        try:
            with open(pkl_path, file_mode) as dbfile: 
                # source, destination
                pickle.dump(db, dbfile) # _pickle.PicklingError: Can't pickle <enum 'PipelineSavingScheme'>: it's not the same object as pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline.PipelineSavingScheme
                dbfile.close()
            # Pickling succeeded

            # If we saved to a temporary name, now see if we should overwrite or backup and then replace:
            if is_temporary_file_used:
                assert _desired_final_pickle_path is not None
                if backup_file_if_smaller_than_original:
                    prev_extant_file_size_MB = print_filesystem_file_size(_desired_final_pickle_path, enable_print=False)
                    new_temporary_file_size_MB = print_filesystem_file_size(pkl_path, enable_print=False)
                    if (backup_minimum_difference_MB < (prev_extant_file_size_MB - new_temporary_file_size_MB)):
                        print(f'\tWARNING: prev_extant_file_size_MB ({prev_extant_file_size_MB} MB) > new_temporary_file_size_MB ({new_temporary_file_size_MB} MB)! A backup will be made!')
                        # Backup old file:
                        backup_extant_file(_desired_final_pickle_path) # only backup if the new file is smaller than the older one (meaning the older one has more info)
                
                # replace the old file with the new one:
                print(f"\tmoving new output at '{pkl_path}' -> to desired location: '{_desired_final_pickle_path}'")
                shutil.move(pkl_path, _desired_final_pickle_path) # move the temporary file to the desired destination, overwriting it
            # END if is_tem...
            
            # Print final file size after successful save
            if should_print_output_filesize:
                final_file_size_MB = print_filesystem_file_size(_desired_final_pickle_path, enable_print=False)
                print(f"\tSaved file size: {final_file_size_MB:.2f} MB")
            
        except BaseException as e:
            print(f"ERROR: pickling exception occured while using safeSaveData(pkl_path: {_desired_final_pickle_path}, ..., , should_append={should_append}) but original file was NOT overwritten!\nException: {e}")
            # delete the incomplete pickle file
            if is_temporary_file_used:
                pkl_path.unlink(missing_ok=True) # removes the incomplete file. The user's file located at _desired_final_pickle_path is still intact.
            raise
    
    
        

# Its important to use binary mode
def saveData(pkl_path, db, should_append=False, safe_save:bool=True):
    """ 
    
    safe_save: If True, a temporary extension is added to the save path if the file already exists and the file is only overwritten if pickling doesn't throw an exception.
        This temporarily requires double the disk space.
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, safeSaveData

        saveData('temp.pkl', db)
    """
    if safe_save:
        safeSaveData(pkl_path, db=db, should_append=should_append)
    else:
        if should_append:
            file_mode = 'ab' # 'ab' opens the file as binary and appends to the end
        else:
            file_mode = 'w+b' # 'w+b' opens and truncates the file to 0 bytes (overwritting)
        if not isinstance(pkl_path, Path):
            pkl_path = Path(pkl_path).resolve()
            
        with ProgressMessagePrinter(filepath=pkl_path, action=f"Saving (file mode '{file_mode}')", contents_description='pickle file', finished_message='saved pickle file', returns_string=False):
            with open(pkl_path, file_mode) as dbfile: 
                # source, destination
                # pickle.dump(db, dbfile)
                custom_dump(db, dbfile) # ModuleExcludesPickler

                dbfile.close()



# ==================================================================================================================================================================================================================================================================================== #
# Split Save Attempts                                                                                                                                                                                                                                                                  #
# ==================================================================================================================================================================================================================================================================================== #

def _is_picklable(obj: Any) -> bool:
    """Test if an object can be pickled.
    
    Args:
        obj: The object to test
        
    Returns:
        True if the object can be pickled, False otherwise
    """
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


def _convert_unpicklable_to_dict(obj: Any, visited: Optional[set] = None, max_depth: int = 50, current_depth: int = 0) -> Any:
    """Recursively convert only unpicklable objects to dicts, preserving picklable nested objects.
    
    This function processes nested structures (dicts, lists, tuples) and converts only
    objects that fail to pickle, while preserving the original types of picklable objects.
    
    Args:
        obj: The object to process
        visited: Set of object IDs already visited (to prevent circular references)
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        
    Returns:
        The object with unpicklable parts converted to dicts, picklable parts preserved
    """
    if visited is None:
        visited = set()
    
    if current_depth >= max_depth:
        return obj
    
    # Handle None
    if obj is None:
        return obj
    
    # Handle primitive types (always picklable)
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    
    # Check for circular references
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)
    
    try:
        # First, test if the object itself is picklable
        if _is_picklable(obj):
            # Object is picklable, but we may need to process nested structures
            # For containers, we still need to check nested elements
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    result[k] = _convert_unpicklable_to_dict(v, visited, max_depth, current_depth + 1)
                visited.remove(obj_id)  # Remove before returning
                return result
            elif isinstance(obj, (list, tuple)):
                processed_items = [_convert_unpicklable_to_dict(item, visited, max_depth, current_depth + 1) for item in obj]
                visited.remove(obj_id)  # Remove before returning
                return type(obj)(processed_items)
            else:
                # Picklable non-container object - return as-is
                visited.remove(obj_id)  # Remove before returning
                return obj
        else:
            # Object is not picklable, convert to dict
            if has(obj):
                # It's an attrs object, manually extract fields to avoid recursive conversion by asdict()
                # Use attrs.fields() to get field definitions, then extract values manually
                attrs_fields = fields(type(obj))
                result = {}
                for attr_field in attrs_fields:
                    field_name = attr_field.name
                    try:
                        field_value = getattr(obj, field_name, None)
                        # Process the field value to preserve picklable nested objects
                        result[field_name] = _convert_unpicklable_to_dict(field_value, visited, max_depth, current_depth + 1)
                    except AttributeError:
                        # Skip if field doesn't exist
                        pass
                visited.remove(obj_id)  # Remove before returning
                return result
            elif hasattr(obj, '__dict__'):
                # It's a regular object with __dict__
                obj_dict = obj.__dict__.copy()
                # Recursively process the dict values to preserve picklable nested objects
                result = {}
                for k, v in obj_dict.items():
                    result[k] = _convert_unpicklable_to_dict(v, visited, max_depth, current_depth + 1)
                visited.remove(obj_id)  # Remove before returning
                return result
            else:
                # Can't convert, return as-is (might fail later, but that's handled by caller)
                visited.remove(obj_id)  # Remove before returning
                return obj
    except (TypeError, AttributeError, RecursionError) as e:
        # If we encounter an error during processing, return the object as-is
        visited.discard(obj_id)  # Remove if present
        return obj


def _try_pickle_or_convert_to_dict(obj: Any, debug_print: bool = False) -> Any:
    """Try to pickle an object, only convert to dict if pickling fails.
    
    This function attempts to pickle the object directly. If successful, returns
    the object as-is. If pickling fails, converts to dict while preserving picklable
    nested objects.
    
    Args:
        obj: The object to process
        debug_print: If True, prints debug information
        
    Returns:
        The object (if picklable) or a dict representation (if not picklable)
    """
    # Test if the object is picklable
    if _is_picklable(obj):
        if debug_print:
            print(f'Object {type(obj).__name__} is picklable, preserving original type')
        return obj
    else:
        # Object is not picklable, convert to dict
        if debug_print:
            print(f'Object {type(obj).__name__} is not picklable, converting to dict (preserving picklable nested objects)')
        return _convert_unpicklable_to_dict(obj)


@function_attributes(short_name=None, tags=['save', 'pickle', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-11 08:11', related_items=['loadSplitData', 'load_split_pickled_global_computation_results'])
def safeSaveSplitData(pkl_path: Union[str, Path], computed_data: Union[Dict[str, Any], Any], include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True):
    """Save out data by splitting it into separate pickle files for each key in the dictionary.

    Similar to safeSaveData but saves each item in computed_data as a separate file in a split folder.

    Reciprocal:
        `loadSplitData`

    Args:
        pkl_path: Path to save to. If a directory, uses default filename "computed_data.pkl". If a file, uses that path directly.
        computed_data: Dictionary of data to save, or an attrs-based object that will be converted to a dictionary. Each key-value pair will be saved as a separate file.
        include_includelist: Optional list of keys to include. If None, includes all keys.
        continue_after_pickling_errors: If True, continues saving other items if one fails. If False, raises on first error.
        debug_print: If True, prints progress information.

    Returns:
        tuple: (split_save_folder, split_save_paths, split_save_output_types, failed_keys)

    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import safeSaveData, safeSaveSplitData, loadSplitData

        ## Pickle the result:
        pkl_output_path: Path = curr_active_pipeline.get_output_path().joinpath('2026-01-14_PredictiveDecodingComputationsContainer.pkl')
        split_save_folder, split_save_paths, split_save_output_types, failed_keys = safeSaveSplitData(pkl_output_path, container, debug_print=True)
        print(f'split_save_folder: "{split_save_folder.as_posix()}"')


    #TODO 2023-11-22 18:54: - [ ] One major issue is that the types are lost upon reloading, so I think we'll need to save them somewhere. They can be fixed post-hoc like:
    # Update result with correct type:
    # computed_data['RankOrder'] = RankOrderComputationsContainer(**computed_data['RankOrder'])

    """
    from pickle import PicklingError
    from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
    
    # Convert non-dict objects to dictionaries if needed, preserving picklable nested objects
    if not isinstance(computed_data, dict):
        if has(computed_data):
            # It's an attrs object, convert to dict but preserve picklable nested objects
            if debug_print:
                print(f'Converting attrs object {type(computed_data).__name__} to dictionary (preserving picklable nested objects)')
            computed_data = _convert_unpicklable_to_dict(computed_data)
        elif hasattr(computed_data, '__dict__'):
            # It's a regular object with __dict__, convert to dict but preserve picklable nested objects
            if debug_print:
                print(f'Converting object {type(computed_data).__name__} with __dict__ to dictionary (preserving picklable nested objects)')
            computed_data = _convert_unpicklable_to_dict(computed_data)
        else:
            raise TypeError(f"computed_data must be a dictionary, attrs object, or object with __dict__. Got {type(computed_data)}")
    else:
        # It's already a dict, but we should process nested values to preserve picklable objects
        if debug_print:
            print(f'Processing dictionary (preserving picklable nested objects)')
        processed_dict = {}
        for k, v in computed_data.items():
            processed_dict[k] = _convert_unpicklable_to_dict(v)
        computed_data = processed_dict
    
    # Resolve pkl_path
    if not isinstance(pkl_path, Path):
        pkl_path = Path(pkl_path).resolve()
    
    # Determine the base pickle path
    if pkl_path.is_dir():
        # If it's a directory, use a default filename
        base_pickle_path = pkl_path.joinpath("computed_data.pkl").resolve()
    else:
        # If it's a file, use it directly
        base_pickle_path = pkl_path.resolve()

    if debug_print:
        print(f'base_pickle_path: {base_pickle_path}')
    
    ## In split save, we save each result separately in a folder
    split_save_folder_name: str = f'{base_pickle_path.stem}_split'
    split_save_folder: Path = base_pickle_path.parent.joinpath(split_save_folder_name).resolve()
    if debug_print:
        print(f'split_save_folder: {split_save_folder}')
    # make if doesn't exist
    split_save_folder.mkdir(exist_ok=True)
    
    if include_includelist is None:
        ## include all keys if none are specified
        try:
            include_includelist = list(computed_data.keys()) 
        except AttributeError as err:
            # AttributeError: 'PredictiveDecodingComputationsContainer' object has no attribute 'keys' -- for some reason it's still an object
            computed_data = computed_data.__getstate__()
            include_includelist = list(computed_data.keys())

    
    ## Save each item in the computed_data dictionary:
    split_save_paths = {}
    split_save_output_types = {}
    failed_keys = []
    skipped_keys = []
    for k, v in computed_data.items():
        if (include_includelist is not None) and (k in include_includelist):
            curr_split_result_pickle_path = split_save_folder.joinpath(f'Split_{k}.pkl').resolve()
            if debug_print:
                print(f'k: {k} -- size_MB: {print_object_memory_usage(v, enable_print=False)}')
                print(f'\tcurr_split_result_pickle_path: {curr_split_result_pickle_path}')
            was_save_success = False
            curr_item_type = type(v)
            try:
                # Try to pickle the value directly first, only convert to dict if pickling fails
                # This preserves original types for picklable objects
                processed_value = _try_pickle_or_convert_to_dict(v, debug_print=debug_print)
                # saveData(curr_split_result_pickle_path, (processed_value))
                saveData(curr_split_result_pickle_path, (processed_value, str(curr_item_type.__module__), str(curr_item_type.__name__)))    
                was_save_success = True
            except (KeyError, AttributeError) as e:
                print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                pass
            except PicklingError as e:
                if not continue_after_pickling_errors:
                    raise
                else:
                    print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                    pass
                
            if was_save_success:
                split_save_paths[k] = curr_split_result_pickle_path
                split_save_output_types[k] = curr_item_type
                if debug_print:
                    print(f'\tfile_size_MB: {print_filesystem_file_size(curr_split_result_pickle_path, enable_print=False)} MB')
            else:
                failed_keys.append(k)
        else:
            if debug_print:
                print(f'\tskipping key "{k}" because it is not included in include_includelist: {include_includelist}')
            skipped_keys.append(k)
            
    if len(failed_keys) > 0:
        print(f'WARNING: failed_keys: {failed_keys} did not save successfully! They HAVE NOT BEEN SAVED!')
    return split_save_folder, split_save_paths, split_save_output_types, failed_keys


# ==================================================================================================================================================================================================================================================================================== #
# Unfinished Implementation                                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #


# def safeSaveSplitData(obj: Any, exclude_types: tuple = (Callable, ModuleType, type), max_depth: int = 10, _current_depth: int = 0) -> Dict[str, Any]:
#     """Decomposes an object into its picklable parts by iterating through its attrs-fields, dataclass fields, or __dict__ attributes.
    
#     This function recursively extracts all picklable attributes from an object, handling:
#     - attrs objects (using attrs.fields())
#     - dataclass objects (using dataclasses.fields())
#     - Regular objects (using __dict__)
#     - Objects with custom __getstate__ methods
    
#     Args:
#         obj: The object to decompose into picklable parts
#         exclude_types: Tuple of types to exclude from pickling (default: Callable, ModuleType, type)
#         max_depth: Maximum recursion depth to prevent infinite loops (default: 10)
#         _current_depth: Internal parameter to track recursion depth
        
#     Returns:
#         Dictionary containing picklable field names and their values
        
#     Example:
#         from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import safeSaveSplitData
        
#         # For an attrs object
#         picklable_data = safeSaveSplitData(my_attrs_object)
        
#         # For a dataclass object
#         picklable_data = safeSaveSplitData(my_dataclass_object)
        
#         # For a regular object
#         picklable_data = safeSaveSplitData(my_regular_object)
#     """
#     if _current_depth >= max_depth:
#         return {}
    
#     result = {}
    
#     # Check if object has custom __getstate__ method
#     if hasattr(obj, '__getstate__') and callable(getattr(obj, '__getstate__', None)):
#         try:
#             state = obj.__getstate__()
#             if isinstance(state, dict):
#                 # Recursively process the state dictionary
#                 for key, value in state.items():
#                     if _is_picklable(value, exclude_types):
#                         result[key] = _process_value(value, exclude_types, max_depth, _current_depth + 1)
#                     else:
#                         # Try to decompose the unpicklable value
#                         try:
#                             decomposed = safeSaveSplitData(value, exclude_types, max_depth, _current_depth + 1)
#                             if decomposed:
#                                 result[key] = decomposed
#                         except (AttributeError, TypeError, RecursionError):
#                             pass  # Skip unpicklable values
#             else:
#                 result['__getstate__'] = state
#         except (AttributeError, TypeError):
#             pass  # Fall through to other methods
    
#     # Handle attrs objects
#     try:
#         if attrs.has(obj):
#             for field in attrs.fields(type(obj)):
#                 field_name = field.name
#                 try:
#                     value = getattr(obj, field_name, None)
#                     if _is_picklable(value, exclude_types):
#                         result[field_name] = _process_value(value, exclude_types, max_depth, _current_depth + 1)
#                     else:
#                         # Try to decompose the unpicklable value
#                         try:
#                             decomposed = safeSaveSplitData(value, exclude_types, max_depth, _current_depth + 1)
#                             if decomposed:
#                                 result[field_name] = decomposed
#                         except (AttributeError, TypeError, RecursionError):
#                             pass  # Skip unpicklable values
#                 except AttributeError:
#                     pass  # Skip if field doesn't exist
#     except (TypeError, AttributeError):
#         pass  # Not an attrs object, continue
    
#     # Handle dataclass objects
#     try:
#         if dataclasses.is_dataclass(obj):
#             for field in dataclasses.fields(obj):
#                 field_name = field.name
#                 try:
#                     value = getattr(obj, field_name, None)
#                     if _is_picklable(value, exclude_types):
#                         result[field_name] = _process_value(value, exclude_types, max_depth, _current_depth + 1)
#                     else:
#                         # Try to decompose the unpicklable value
#                         try:
#                             decomposed = safeSaveSplitData(value, exclude_types, max_depth, _current_depth + 1)
#                             if decomposed:
#                                 result[field_name] = decomposed
#                         except (AttributeError, TypeError, RecursionError):
#                             pass  # Skip unpicklable values
#                 except AttributeError:
#                     pass  # Skip if field doesn't exist
#     except (TypeError, AttributeError):
#         pass  # Not a dataclass, continue
    
#     # Handle regular objects with __dict__
#     if hasattr(obj, '__dict__'):
#         try:
#             for key, value in obj.__dict__.items():
#                 # Skip private/internal attributes that start with double underscore (except __getstate__ which we already handled)
#                 if key.startswith('__') and key != '__getstate__':
#                     continue
                    
#                 if _is_picklable(value, exclude_types):
#                     result[key] = _process_value(value, exclude_types, max_depth, _current_depth + 1)
#                 else:
#                     # Try to decompose the unpicklable value
#                     try:
#                         decomposed = safeSaveSplitData(value, exclude_types, max_depth, _current_depth + 1)
#                         if decomposed:
#                             result[key] = decomposed
#                     except (AttributeError, TypeError, RecursionError):
#                         pass  # Skip unpicklable values
#         except (AttributeError, TypeError):
#             pass
    
#     return result


# def _is_picklable(value: Any, exclude_types: tuple) -> bool:
#     """Check if a value is picklable by testing if it's an instance of excluded types."""
#     if value is None:
#         return True
    
#     # Check if value is an instance of excluded types
#     if isinstance(value, exclude_types):
#         return False
    
#     # Check for common unpicklable types
#     if isinstance(value, type) and not isinstance(value, type(None)):
#         return False
    
#     # Try to pickle the value to see if it's actually picklable
#     # This is the definitive test - dill can pickle many things that standard pickle cannot
#     try:
#         pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
#         return True
#     except Exception:
#         # Any exception during pickling means it's not picklable
#         return False


# def _process_value(value: Any, exclude_types: tuple, max_depth: int, current_depth: int) -> Any:
#     """Process a value, recursively decomposing complex objects if needed."""
#     if value is None:
#         return None
    
#     # Handle basic picklable types
#     if isinstance(value, (str, int, float, bool, bytes, type(None))):
#         return value
    
#     # Handle lists
#     if isinstance(value, list):
#         return [_process_value(item, exclude_types, max_depth, current_depth + 1) for item in value]
    
#     # Handle tuples
#     if isinstance(value, tuple):
#         return tuple(_process_value(item, exclude_types, max_depth, current_depth + 1) for item in value)
    
#     # Handle dictionaries
#     if isinstance(value, dict):
#         return {k: _process_value(v, exclude_types, max_depth, current_depth + 1) for k, v in value.items()}
    
#     # Handle sets
#     if isinstance(value, set):
#         return {_process_value(item, exclude_types, max_depth, current_depth + 1) for item in value}
    
#     # For other objects, try to decompose if not directly picklable
#     if not _is_picklable(value, exclude_types) and current_depth < max_depth:
#         try:
#             decomposed = safeSaveSplitData(value, exclude_types, max_depth, current_depth + 1)
#             if decomposed:
#                 return decomposed
#         except (AttributeError, TypeError, RecursionError):
#             pass
    
#     # Return as-is if it's picklable or we can't decompose it
#     return value




# ==================================================================================================================================================================================================================================================================================== #
# SEP                                                                                                                                                                                                                                                                                  #
# ==================================================================================================================================================================================================================================================================================== #


# global_move_modules_list: Dict[str, str] - a dict with keys equal to the old full path to a class and values equal to the updated (replacement) full path to the class. Used to update the path to class definitions for loading previously pickled results after refactoring.

global_move_modules_list:Dict={
    'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
    'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
        # 'pyphoplacecellanalysis.General.Configs.DynamicConfigs.*':'pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.*', # VideoOutputModeConfig, PlottingConfig, InteractivePlaceCellConfig
    'pyphoplacecellanalysis.General.Configs.DynamicConfigs.VideoOutputModeConfig':'pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.VideoOutputModeConfig', # VideoOutputModeConfig, PlottingConfig, InteractivePlaceCellConfig
    'pyphoplacecellanalysis.General.Configs.DynamicConfigs.PlottingConfig':'pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.PlottingConfig',
    'pyphoplacecellanalysis.General.Configs.DynamicConfigs.InteractivePlaceCellConfig':'pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.InteractivePlaceCellConfig',
    # 'pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins':'pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig', # SingleNeuronPlottingExtended, 
    'pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins.SingleNeuronPlottingExtended':'pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig.SingleNeuronPlottingExtended',
    # 'pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins.':'pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig', # SingleNeuronPlottingExtended, 
    'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalMergedDecodersResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalPseudo2DDecodersResult',
    'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersDecodedResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersContinuouslyDecodedResult',
    'pyphocorehelpers.indexing_helpers.BinningInfo':'neuropy.utils.mixins.binning_helpers.BinningInfo',
    'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.NonPBEDimensionalDecodingResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.DecodingResultND', # 2025-06-30 10:59
}



def loadData(pkl_path, debug_print=False, **kwargs):
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData

    loadData(
    """
    # for reading also binary mode is important
    db = None
    active_move_modules_list: Dict = kwargs.pop('move_modules_list', global_move_modules_list)
    
    with ProgressMessagePrinter(pkl_path, action='Computing', contents_description='loaded session pickle file'):
        
        with open(pkl_path, 'rb') as dbfile:
            try:
                db = renamed_load(dbfile, move_modules_list=active_move_modules_list, **kwargs)

            except NotImplementedError as err:
                error_message = str(err)
                if 'WindowsPath' in error_message:  # Check if WindowsPath is missing
                    print("Issue with pickled WindowsPath on Linux for path {}, performing pathlib workaround...".format(pkl_path))
                    win_backup = pathlib.WindowsPath  # Backup the WindowsPath definition
                    try:
                        pathlib.WindowsPath = pathlib.PureWindowsPath
                        db = renamed_load(dbfile, move_modules_list=active_move_modules_list, **kwargs)
                        
                    finally:
                        pathlib.WindowsPath = win_backup  # Restore the backup WindowsPath definition
                        
                elif 'PosixPath' in error_message:  # Check if PosixPath is missing
                    # Fixes issue with pickled POSIX_PATH on windows for path.
                    posix_backup = pathlib.PosixPath # backup the PosixPath definition
                    try:
                        pathlib.PosixPath = pathlib.PurePosixPath
                        db = renamed_load(dbfile, move_modules_list=active_move_modules_list, **kwargs) # Fails this time if it still throws an error
                    finally:
                        pathlib.PosixPath = posix_backup # restore the backup posix path definition
                                        
                else:
                    print("Unknown issue with pickled path for path {}, performing pathlib workaround...".format(pkl_path))
                    raise
                
            except EOFError as e:
                # occurs when the pickle saving is interrupted and the output file is ruined. # cannot load global results: !! Ran out of input ::::: (<class 'EOFError'>, EOFError('Ran out of input'), <traceback object at 0x000002015BC7BF80>)
                raise
                # often we'll want to delete the fragmented pickle file and continue

            
            except Exception as e:
                # unhandled exception
                raise
            finally:
                dbfile.close()
            
            
        if debug_print:
            try:
                for keys in db:
                    print(keys, '=>', db[keys])
            except Exception as e:
                print(f'encountered exception {e} while printing. Turning into a warning and continuing.')
                # raise e
        # return db
    return db


@function_attributes(short_name=None, tags=['load', 'pickle', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-14 08:00', related_items=['safeSaveSplitData'])
def loadSplitData(pkl_path: Union[str, Path], debug_print:bool=True, target_cls: Optional[type] = None, raise_on_exception: bool=True, **kwargs) -> Union[Dict[str, Any], Any]:
    """Load data from split pickle files created by safeSaveSplitData.

    Reciprocal function to safeSaveSplitData. Loads all Split_*.pkl files from a split folder
    and reconstructs the original data dictionary.

    Args:
        pkl_path: Path to the pickle file or directory. If a directory, uses default filename "computed_data.pkl".
                  If a file, uses that path directly. The split folder is determined by appending "_split" to the base filename.
        debug_print: If True, prints progress information.
        target_cls: Optional class to automatically rebuild the loaded data into. If provided and the loaded data is a dict,
                    it will be automatically rebuilt into an instance of this class with all nested objects also rebuilt.
        raise_on_exception: If True (default), raises exceptions during auto-rebuild. If False, catches exceptions and
                           returns the dict instead, useful for debugging.
        **kwargs: Additional arguments passed to loadData for loading individual files.

    Returns:
        Dictionary mapping keys to loaded values, or an instance of target_cls if target_cls is provided.
        Values that were saved as tuples with type info (v_dict, module_name, type_name) will be returned as dictionaries.

    Usage:

        # A file saved with: _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import safeSaveData, safeSaveSplitData

        ## Pickle the result:
        pkl_output_path: Path = curr_active_pipeline.get_output_path().joinpath('2026-01-14_PredictiveDecodingComputationsContainer.pkl')
        split_save_folder, split_save_paths, split_save_output_types, failed_keys = safeSaveSplitData(pkl_output_path, container, debug_print=True)
        print(f'split_save_folder: "{split_save_folder.as_posix()}"')

        # Can be reciprocally loaded with: ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadSplitData, safeSaveSplitData

        # Save split data
        split_save_folder, split_save_paths, split_save_output_types, failed_keys = safeSaveSplitData(pkl_path, computed_data, debug_print=True)
        
        # Load split data
        loaded_data = loadSplitData(pkl_path, debug_print=True)

    Concrete Examples:

        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadSplitData
        from neuropy.utils.mixins.indexing_helpers import get_dict_subset
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecodingComputationsContainer, PredictiveDecoding, DecodingLocalityMeasures

        # Load split data (old manual approach)
        split_save_folder: Path = curr_active_pipeline.get_output_path().joinpath('2026-01-14_PredictiveDecodingComputationsContainer_split')
        container = loadSplitData(split_save_folder, debug_print=True)
        if isinstance(container, dict):
            container: PredictiveDecodingComputationsContainer = PredictiveDecodingComputationsContainer(**get_dict_subset(container, subset_excludelist=['_VersionedResultMixin_version']))
        
        # Load split data with auto-rebuild (new convenient approach)
        container = loadSplitData(split_save_folder, debug_print=True, target_cls=PredictiveDecodingComputationsContainer)
        # Nested objects (predictive_decoding, locality_measures, etc.) are automatically rebuilt
            
    Example 2:

        split_save_folder: Path = curr_active_pipeline.get_output_path().joinpath('2026-01-14_PredictiveDecodingComputationsContainer_masked')
        # Old approach
        masked_container = loadSplitData(split_save_folder, debug_print=True)
        if isinstance(masked_container, dict):
            masked_container: PredictiveDecodingComputationsContainer = PredictiveDecodingComputationsContainer(**get_dict_subset(masked_container, subset_excludelist=['_VersionedResultMixin_version']))
        
        # New approach with auto-rebuild
        masked_container = loadSplitData(split_save_folder, debug_print=True, target_cls=PredictiveDecodingComputationsContainer)
        # All nested objects are automatically rebuilt

        type(masked_container)



    """
    # Extract move_modules_list from kwargs with global_move_modules_list default (mirroring loadData)
    active_move_modules_list: Dict = kwargs.pop('move_modules_list', global_move_modules_list)
    
    # Resolve pkl_path with cross-platform path handling
    try:
        if not isinstance(pkl_path, Path):
            pkl_path = Path(pkl_path).resolve()
        
        # Determine the base pickle path (mirroring safeSaveSplitData logic)
        if pkl_path.is_dir():
            # If it's a directory, use a default filename
            base_pickle_path = pkl_path.joinpath("computed_data.pkl").resolve()
        else:
            # If it's a file, use it directly
            base_pickle_path = pkl_path.resolve()
    except NotImplementedError as err:
        error_message = str(err)
        if 'WindowsPath' in error_message:  # Check if WindowsPath is missing
            print("Issue with WindowsPath on Linux for path {}, performing pathlib workaround...".format(pkl_path))
            win_backup = pathlib.WindowsPath  # Backup the WindowsPath definition
            try:
                pathlib.WindowsPath = pathlib.PureWindowsPath
                if not isinstance(pkl_path, Path):
                    pkl_path = Path(pkl_path).resolve()
                if pkl_path.is_dir():
                    base_pickle_path = pkl_path.joinpath("computed_data.pkl").resolve()
                else:
                    base_pickle_path = pkl_path.resolve()
            finally:
                pathlib.WindowsPath = win_backup  # Restore the backup WindowsPath definition
        elif 'PosixPath' in error_message:  # Check if PosixPath is missing
            # Fixes issue with pickled POSIX_PATH on windows for path.
            posix_backup = pathlib.PosixPath # backup the PosixPath definition
            try:
                pathlib.PosixPath = pathlib.PurePosixPath
                if not isinstance(pkl_path, Path):
                    pkl_path = Path(pkl_path).resolve()
                if pkl_path.is_dir():
                    base_pickle_path = pkl_path.joinpath("computed_data.pkl").resolve()
                else:
                    base_pickle_path = pkl_path.resolve()
            finally:
                pathlib.PosixPath = posix_backup # restore the backup posix path definition
        else:
            print("Unknown issue with path for path {}, performing pathlib workaround...".format(pkl_path))
            raise

    if debug_print:
        print(f'base_pickle_path: {base_pickle_path}')
    
    # Check if pkl_path is already a split folder (exists and is a directory)
    # If not, fall back to adding the _split suffix like usual
    if pkl_path.exists() and pkl_path.is_dir():
        # User passed the direct split folder path
        split_save_folder = pkl_path
        if debug_print:
            print(f'Using provided split folder path: {split_save_folder}')
    else:
        # Determine the split folder (mirroring safeSaveSplitData logic)
        split_save_folder_name: str = f'{base_pickle_path.stem}_split'
        split_save_folder: Path = base_pickle_path.parent.joinpath(split_save_folder_name).resolve()
        if debug_print:
            print(f'split_save_folder to load from: {split_save_folder}')
    
    if not split_save_folder.exists():
        raise FileNotFoundError(f"Split folder does not exist: {split_save_folder}")
    if not split_save_folder.is_dir():
        raise NotADirectoryError(f"Split folder is not a directory: {split_save_folder}")
    
    # Find all Split_*.pkl files
    loaded_data = {}
    found_split_paths = []
    successfully_loaded_keys = {}
    failed_loaded_keys = {}
    
    with ProgressMessagePrinter(split_save_folder, action='Loading', contents_description='split pickle files'):
        for p in split_save_folder.rglob('Split_*.pkl'):
            if debug_print:
                print(f'Loading: {p}')
            found_split_paths.append(p)
            # Extract the key name by removing "Split_" prefix from the stem
            curr_result_key: str = p.stem.removeprefix('Split_')
            
            # Load the file
            try:
                loaded_value = loadData(p, debug_print=False, move_modules_list=active_move_modules_list, **kwargs)
                
                # Handle the loaded value format
                if isinstance(loaded_value, tuple) and len(loaded_value) == 3:
                    # Saved as (v_dict, module_name, type_name) - extract the dict
                    loaded_result_dict, curr_item_type_module, curr_item_type_name = loaded_value
                    if debug_print:
                        print(f'\tLoaded {curr_result_key}: type={curr_item_type_module}.{curr_item_type_name}')
                    loaded_data[curr_result_key] = loaded_result_dict
                elif isinstance(loaded_value, dict):
                    # Already a dict, use directly
                    loaded_data[curr_result_key] = loaded_value
                else:
                    # Other format, store as-is
                    loaded_data[curr_result_key] = loaded_value
                
                successfully_loaded_keys[curr_result_key] = p
                
            except EOFError as e:
                # occurs when the pickle saving is interrupted and the output file is ruined.
                # Re-raise immediately as it indicates a corrupted file
                print(f'EOFError loading {curr_result_key} from "{p}": {e}')
                print("This indicates a corrupted pickle file. The fragmented pickle file may need to be deleted.")
                failed_loaded_keys[curr_result_key] = p
                if debug_print:
                    import traceback
                    traceback.print_exc()
                # Continue loading other files, but log this error
                # Note: We don't re-raise here to allow loading of other files, but this is a serious error
            
            except Exception as e:
                # Other exceptions - log and continue
                print(f'Error loading {curr_result_key} from "{p}": {e}')
                failed_loaded_keys[curr_result_key] = p
                if debug_print:
                    import traceback
                    traceback.print_exc()
    
    if debug_print:
        print(f'Successfully loaded {len(successfully_loaded_keys)} keys: {list(successfully_loaded_keys.keys())}')
        if len(failed_loaded_keys) > 0:
            print(f'Failed to load {len(failed_loaded_keys)} keys: {list(failed_loaded_keys.keys())}')
    
    # If target_cls is provided, automatically rebuild the loaded data
    if target_cls is not None and isinstance(loaded_data, dict):
        if debug_print:
            print(f'Auto-rebuilding loaded data into {target_cls.__name__}...')
        try:
            rebuilt_obj = _helper_rebuild_obj_from_class_if_needed(target_cls, loaded_data)
            if debug_print:
                print(f'Successfully rebuilt into {target_cls.__name__}')
            return rebuilt_obj
        except Exception as e:
            if debug_print:
                print(f'Warning: Failed to auto-rebuild into {target_cls.__name__}: {e}')
                import traceback
                traceback.print_exc()
            # Return the dict if rebuilding fails, or raise if raise_on_exception is True
            if raise_on_exception:
                raise
            return loaded_data
    
    return loaded_data



def _is_attrs_class(cls_or_type: Any) -> bool:
    """Check if a type is an attrs class.
    
    Args:
        cls_or_type: A class or type to check
        
    Returns:
        True if the type is an attrs class, False otherwise
    """
    if not isinstance(cls_or_type, type):
        return False
    try:
        return has(cls_or_type)
    except (TypeError, AttributeError):
        return False


def _extract_actual_type(annotated_type: Any) -> Tuple[Any, bool]:
    """Extract the actual type from Optional, Dict, List, and other generic types.
    
    Args:
        annotated_type: A type annotation (may be Optional[T], Dict[K, V], List[T], etc.)
        
    Returns:
        Tuple of (actual_type, is_container) where:
        - actual_type: The inner type to check/rebuild
        - is_container: True if this is a container type (Dict, List, etc.) that needs special handling
    """
    origin = get_origin(annotated_type)
    args = get_args(annotated_type)
    
    if origin is None:
        # Not a generic type, return as-is
        return annotated_type, False
    
    # Handle Optional[T] -> T
    if origin is Union and len(args) == 2 and type(None) in args:
        # Extract the non-None type
        non_none_type = next(arg for arg in args if arg is not type(None))
        return _extract_actual_type(non_none_type)
    
    # Handle Dict[K, V] -> V (the value type)
    if origin is dict or origin is Dict:
        if len(args) >= 2:
            value_type = args[1]
            return _extract_actual_type(value_type), True
        return Any, True
    
    # Handle List[T] -> T (the element type)
    if origin is list or origin is List:
        if len(args) >= 1:
            element_type = args[0]
            return _extract_actual_type(element_type), True
        return Any, True
    
    # Handle Tuple[T, ...] -> T
    if origin is tuple or origin is Tuple:
        if len(args) >= 1:
            element_type = args[0]
            return _extract_actual_type(element_type), True
        return Any, True
    
    # For other generic types, try to extract the first argument
    if args:
        return _extract_actual_type(args[0]), True
    
    return annotated_type, False


def _rebuild_nested_objects_from_dict(target_obj: Any, visited: Optional[set] = None, max_depth: int = 50, current_depth: int = 0) -> Any:
    """Recursively rebuild nested objects from dicts based on type annotations.
    
    This function processes all fields of an attrs object and rebuilds nested objects
    that are stored as dicts back into their proper types.
    
    Args:
        target_obj: The object to process (must be an attrs instance)
        visited: Set of object IDs already visited (to prevent circular references)
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        The object with nested dicts rebuilt into proper types
    """
    if visited is None:
        visited = set()
    
    if current_depth >= max_depth:
        return target_obj
    
    # Only process attrs objects
    if not _is_attrs_class(type(target_obj)):
        return target_obj
    
    # Prevent circular references
    obj_id = id(target_obj)
    if obj_id in visited:
        return target_obj
    visited.add(obj_id)
    
    try:
        # Get type hints for the class
        type_hints = get_type_hints(type(target_obj))
    except (TypeError, AttributeError, NameError):
        # If we can't get type hints, return as-is
        return target_obj
    
    try:
        # Get attrs fields
        attrs_fields = fields(type(target_obj))
    except (TypeError, AttributeError):
        return target_obj
    
    # Process each field
    for attr_field in attrs_fields:
        field_name = attr_field.name
        if not hasattr(target_obj, field_name):
            continue
        
        field_value = getattr(target_obj, field_name)
        
        # Skip None values
        if field_value is None:
            continue
        
        # Get the type annotation for this field
        field_type = type_hints.get(field_name, None)
        if field_type is None:
            continue
        
        # Extract the actual type (handling Optional, Dict, List, etc.)
        actual_type, is_container = _extract_actual_type(field_type)
        
        if is_container:
            # Handle container types (Dict, List, etc.)
            origin = get_origin(field_type)
            args = get_args(field_type)
            
            if (origin is dict or origin is Dict) and isinstance(field_value, dict):
                # Dict[K, V] - rebuild values
                if len(args) >= 2:
                    value_type = args[1]
                    value_actual_type, _ = _extract_actual_type(value_type)
                    if _is_attrs_class(value_actual_type):
                        rebuilt_dict = {}
                        for k, v in field_value.items():
                            if isinstance(v, dict):
                                rebuilt_dict[k] = _helper_rebuild_obj_from_class_if_needed(value_actual_type, v)
                                # Recursively rebuild nested objects in the rebuilt value
                                rebuilt_dict[k] = _rebuild_nested_objects_from_dict(rebuilt_dict[k], visited, max_depth, current_depth + 1)
                            else:
                                rebuilt_dict[k] = v
                        setattr(target_obj, field_name, rebuilt_dict)
                # Also handle nested Dict types (e.g., Dict[str, Dict[str, T]])
                elif len(args) >= 2:
                    value_type = args[1]
                    if get_origin(value_type) is dict or get_origin(value_type) is Dict:
                        # Nested dict - recursively process
                        rebuilt_dict = {}
                        for k, v in field_value.items():
                            if isinstance(v, dict):
                                rebuilt_dict[k] = _rebuild_nested_dict_values(v, value_type, visited, max_depth, current_depth + 1)
                            else:
                                rebuilt_dict[k] = v
                        setattr(target_obj, field_name, rebuilt_dict)
            
            elif (origin is list or origin is List) and isinstance(field_value, list):
                # List[T] - rebuild elements
                if len(args) >= 1:
                    element_type = args[0]
                    element_actual_type, _ = _extract_actual_type(element_type)
                    if _is_attrs_class(element_actual_type):
                        rebuilt_list = []
                        for item in field_value:
                            if isinstance(item, dict):
                                rebuilt_item = _helper_rebuild_obj_from_class_if_needed(element_actual_type, item)
                                rebuilt_item = _rebuild_nested_objects_from_dict(rebuilt_item, visited, max_depth, current_depth + 1)
                                rebuilt_list.append(rebuilt_item)
                            else:
                                rebuilt_list.append(item)
                        setattr(target_obj, field_name, rebuilt_list)
        else:
            # Handle direct type (not a container)
            if isinstance(field_value, dict) and _is_attrs_class(actual_type):
                # Rebuild the nested object
                rebuilt_obj = _helper_rebuild_obj_from_class_if_needed(actual_type, field_value)
                # Recursively rebuild nested objects in the rebuilt object
                rebuilt_obj = _rebuild_nested_objects_from_dict(rebuilt_obj, visited, max_depth, current_depth + 1)
                setattr(target_obj, field_name, rebuilt_obj)
    
    return target_obj


def _rebuild_nested_dict_values(nested_dict: Dict, dict_type: Any, visited: set, max_depth: int, current_depth: int) -> Dict:
    """Helper to rebuild nested dict values recursively."""
    args = get_args(dict_type)
    if len(args) >= 2:
        value_type = args[1]
        value_actual_type, is_container = _extract_actual_type(value_type)
        
        rebuilt = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                if is_container:
                    rebuilt[k] = _rebuild_nested_dict_values(v, value_type, visited, max_depth, current_depth + 1)
                elif _is_attrs_class(value_actual_type):
                    rebuilt_obj = _helper_rebuild_obj_from_class_if_needed(value_actual_type, v)
                    rebuilt_obj = _rebuild_nested_objects_from_dict(rebuilt_obj, visited, max_depth, current_depth + 1)
                    rebuilt[k] = rebuilt_obj
                else:
                    rebuilt[k] = v
            else:
                rebuilt[k] = v
        return rebuilt
    return nested_dict


@function_attributes(short_name=None, tags=['loading', 'split', 'helper', 'class'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-01-14 22:13', related_items=[])
def _helper_rebuild_obj_from_class_if_needed(target_cls, a_possible_dict):
    """ tries to rebuild the `a_possible_dict` instance/object as an instance of the class `target_cls`
        Useful for rebuilding object instances from loaded dicts
        

    Example:
        ## for an object `PredictiveDecodingComputationsContainer`

        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import _helper_rebuild_obj_from_class_if_needed

        # _helper_build_class
        ## Example: to rebuild a `PredictiveDecodingComputationsContainer` type object `a_masked_container`
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import PredictiveDecoding, DecodingLocalityMeasures, PredictiveDecodingComputationsContainer

        ## INPUTS: a_masked_container = masked_container

        if isinstance(a_masked_container, dict):
            # a_masked_container: PredictiveDecodingComputationsContainer = PredictiveDecodingComputationsContainer(**get_dict_subset(a_masked_container, subset_excludelist=['_VersionedResultMixin_version']))
            a_masked_container: PredictiveDecodingComputationsContainer = _helper_rebuild_obj_from_class_if_needed(PredictiveDecodingComputationsContainer, a_masked_container)
            
        if isinstance(a_masked_container.predictive_decoding, dict):
            # a_masked_container.predictive_decoding = PredictiveDecoding(**get_dict_subset(a_masked_container.predictive_decoding, subset_excludelist=['_VersionedResultMixin_version']))
            a_masked_container.predictive_decoding = _helper_rebuild_obj_from_class_if_needed(PredictiveDecoding, a_masked_container.predictive_decoding)
            print(f'updated a_masked_container.predictive_decoding')

        if isinstance(a_masked_container.predictive_decoding.locality_measures, dict):
            # a_masked_container.predictive_decoding.locality_measures = DecodingLocalityMeasures(**get_dict_subset(a_masked_container.predictive_decoding.locality_measures, subset_excludelist=['_VersionedResultMixin_version', '_interpolator', 'locality_measures_df']))
            a_masked_container.predictive_decoding.locality_measures = _helper_rebuild_obj_from_class_if_needed(DecodingLocalityMeasures, a_masked_container.predictive_decoding.locality_measures)
            print(f'updated a_masked_container.predictive_decoding.locality_measures')


        for a_t_bin_size, v in a_masked_container.epochs_decoded_result_cache_dict.items():
            for an_epoch_name, a_masked_result in v.items():
                a_masked_result = _helper_rebuild_obj_from_class_if_needed(DecodedFilterEpochsResult, a_masked_result)
                # if isinstance(a_masked_result, dict):
                #     a_masked_result: DecodedFilterEpochsResult = DecodedFilterEpochsResult(**get_dict_subset(a_masked_result, subset_excludelist=['_VersionedResultMixin_version'])) ## does this actually update the type of the embedded objects (in the dict)?



        
    """
    from neuropy.utils.mixins.indexing_helpers import get_dict_subset, pop_dict_subset
    
    # Pop specific keys
    if not isinstance(a_possible_dict, dict):
        _cls_kwargs_dict = a_possible_dict.to_dict()
    else:
        _cls_kwargs_dict = a_possible_dict.copy()  # Make a copy to avoid modifying the original

    # Get valid field names from the attrs class to filter the dict
    valid_field_names = set()
    if _is_attrs_class(target_cls):
        try:
            attrs_fields = fields(target_cls)
            valid_field_names = {f.name for f in attrs_fields}
        except (TypeError, AttributeError):
            # If we can't get fields, we'll try to construct with all keys and catch errors
            pass
    
    # Filter dict to only include valid field names (if we have them)
    if valid_field_names:
        # Separate valid and invalid keys
        valid_kwargs = {k: v for k, v in _cls_kwargs_dict.items() if k in valid_field_names}
        invalid_kwargs = {k: v for k, v in _cls_kwargs_dict.items() if k not in valid_field_names}
    else:
        # If we can't determine valid fields, use all keys and hope for the best
        valid_kwargs = _cls_kwargs_dict
        invalid_kwargs = {}

    _ignore_re_add_subset = ['_VersionedResultMixin_version']
    _potential_subset_includelist = ['neuron_extended_ids', '_VersionedResultMixin_version', '_interpolator', 'locality_measures_df', 'time_bin_size', 'spikes_df', 'time_binning_container']
    subset_includelist = [a_col for a_col in _potential_subset_includelist if a_col in valid_kwargs] ## only exclude real columns
    popped_subset = pop_dict_subset(valid_kwargs, subset_includelist=subset_includelist)
    ## INPUTS: active_peak_prominence_2d_results
    an_obj = target_cls(**valid_kwargs)
    ## add the invalid properties from popped_subset (these are known safe attributes)
    for k, v in popped_subset.items():
        if k not in _ignore_re_add_subset:
            setattr(an_obj, k, v)
    ## add the invalid kwargs that are in our known safe list (but weren't valid field names)
    for k, v in invalid_kwargs.items():
        if k in _potential_subset_includelist and k not in _ignore_re_add_subset:
            try:
                setattr(an_obj, k, v)
            except (AttributeError, TypeError):
                # Skip if we can't set it (might be a read-only property)
                pass

    # Automatically rebuild nested objects from dicts based on type annotations
    an_obj = _rebuild_nested_objects_from_dict(an_obj)

    return an_obj




def delete_fragmented_pickle_file(pkl_path: Path, debug_print:bool=True):
    assert pkl_path.exists()
    # file corrupted.
    print(f'Failure loading {pkl_path}, the file is corrupted and incomplete (REACHED END OF FILE).')
    print(f'\t deleting it and continuing. ')
    pkl_path.unlink() # .unlink() deletes a file
    if debug_print:
        print(f"\t {pkl_path} deleted.")

# ==================================================================================================================== #
# BEGIN STAGE/PIPELINE IMPLEMENTATION                                                                                  #
# ==================================================================================================================== #

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# COMMON INPUT/LOADING MIXINS IMPLEMENTATION                                                                           #
class LoadableInput:
    def _check(self):
        assert (self.load_function is not None), "self.load_function must be a valid single-argument load function that isn't None!"
        assert callable(self.load_function), "self.load_function must be callable!"
        assert self.basedir is not None, "self.basedir must not be None!"
        assert isinstance(self.basedir, Path), "self.basedir must be a pathlib.Path type object (or a pathlib.Path subclass)"
        if not self.basedir.exists():
            raise FileExistsError
        else:
            return True

    def load(self):
        self._check()
        self.loaded_data = dict()
        # call the internal load_function with the self.basedir.
        self.loaded_data["sess"] = self.load_function(self.basedir)
        
        pass

class LoadableSessionInput:
    """ Provides session (self.sess) and other helper properties to Stages that load a session """
    @property
    def sess(self):
        """The sess property."""
        return self.loaded_data["sess"]

    @sess.setter
    def sess(self, value):
        self.loaded_data["sess"] = value

    @property
    def active_sess_config(self):
        """The active_sess_config property."""
        return self.sess.config

    @active_sess_config.setter
    def active_sess_config(self, value):
        self.sess.config = value

    @property
    def session_name(self):
        """The session_name property."""
        return self.sess.name

    @session_name.setter
    def session_name(self, value):
        self.sess.name = value
        

    @function_attributes(tags=['output_files', 'filesystem'], related_items=[])
    def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.sess.get_output_path()

    def get_session_context(self) -> IdentifyingContext:
        """ returns the context of the unfiltered session (self.sess) """
        return self.sess.get_context()

    def get_session_format_name(self) -> str:
        """ returns the name of the format (kdiba, bapun, etc) """
        return self.get_session_context().to_dict().get('format_name', None)
    

    def get_session_unique_aclu_information(self) -> pd.DataFrame:
        """  Get the aclu information for each aclu in the dataframe. Adds the ['aclu', 'shank', 'cluster', 'qclu', 'neuron_type'] columns """
        return self.sess.spikes_df.spikes.extract_unique_neuron_identities()
    
    def determine_good_aclus_by_qclu(self, included_qclu_values=[1,2,4,9], debug_print:bool=False) -> NDArray:
        """ 
        From all neuron_IDs in the session, get the ones that meet the new qclu criteria (their value is in) `included_qclu_values`
        
        included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
        included_aclus # np.array([  2,   3,   4,   5,   7,   8,   9,  10,  11,  13,  14,  15,  16,  17,  19,  21,  23,  24,  25,  26,  27,  28,  31,  32,  33,  34,  35,  36,  37,  41,  45,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  66,  67,  68,  69,  70,  71,  73,  74,  75,  76,  78,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  92,  93,  96,  97,  98, 100, 102, 105, 107, 108, 109])

        included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2])
        included_aclus # np.array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  81,  82,  83,  85,  86,  90,  92,  93,  96, 105, 109])

        """
        from neuropy.core.neuron_identities import NeuronType
        
        neuron_identities: pd.DataFrame = self.get_session_unique_aclu_information()
        if debug_print:
            print(f"original {len(neuron_identities)}")
        filtered_neuron_identities: pd.DataFrame = neuron_identities[neuron_identities.neuron_type == NeuronType.PYRAMIDAL]
        if debug_print:
            print(f"post PYRAMIDAL filtering {len(filtered_neuron_identities)}")
        filtered_neuron_identities = filtered_neuron_identities[['aclu', 'shank', 'cluster', 'qclu']]
        filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.qclu, included_qclu_values)] # drop [6, 7], which are said to have double fields - 80 remain
        if debug_print:
            print(f"post (qclu in {included_qclu_values}) filtering {len(filtered_neuron_identities)}")
        return filtered_neuron_identities.aclu.to_numpy()
    






@metadata_attributes(short_name=None, tags=['registered_output_files', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-24 09:00', related_items=[])
class RegisteredOutputsMixin:
    """ Allow pipeline to register its outputs so they can be found/saved/moved, etc.  
    Internal Properties:
        self._registered_output_files
    """
    @property
    def registered_output_files(self):
        """The outputs property."""
        if self._registered_output_files is None:
            """ initialize if needed. """
            self._registered_output_files = DynamicParameters()
        return self._registered_output_files
    @registered_output_files.setter
    def registered_output_files(self, value):
        self._registered_output_files = value

    @property
    def registered_output_files_list(self):
        """The registered_output_files property."""
        return list(self.registered_output_files.keys())

    def register_output_file(self, output_path, output_metadata=None):
        """ registers a new output file for the pipeline """
        self.registered_output_files[output_path] = output_metadata or {}
    
    def clear_registered_output_files(self):
        self.registered_output_files = DynamicParameters()
                

    

# ____________________________________________________________________________________________________________________ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# INPUT STAGE IMPLEMENTATION                                                                                           #
# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
@define(slots=False, repr=False)
class InputPipelineStage(LoadableInput, BaseNeuropyPipelineStage):
    """ The first stage of the NeuropyPipeline. Allows specifying the inputs that will be used.
    
        post_load_functions: List[Callable] a list of Callables that accept the loaded session as input and return the potentially modified session as output.
    """
    @classmethod
    def get_stage_identity(cls) -> PipelineStage:
        return PipelineStage.Input
    
    identity: PipelineStage = field(default=PipelineStage.Input)
    basedir: Path = field(default=Path(""))
    load_function: Callable = field(default=None)
    post_load_functions: List[Callable] = field(default=Factory(list))

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if 'identity' not in state:
            print(f'unpickling from old NeuropyPipelineStage')
            state['identity'] = None
            state['identity'] = type(self).get_stage_identity()
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(NeuropyPipeline, self).__init__() # from


# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class PipelineWithInputStage:
    """ Has an input stage. """
    def set_input(self, session_data_type:str='', basedir="", load_function: Callable = None, post_load_functions: List[Callable] = [], auto_load=True, **kwargs):
        """ 
        Called to set the input stage
        
        Known Uses:
            Called on NeuropyPipeline.__init__(...)
        
        """
        if not isinstance(basedir, Path):
            self.logger.info(f"basedir is not Path. Converting...")
            active_basedir = Path(basedir)
        else:
            self.logger.info(f"basedir is already Path object.")
            active_basedir = basedir

        if not active_basedir.exists():
            self.logger.info(f'active_basedir: "{active_basedir}" does not exist!')
            raise FileExistsError(f'active_basedir: "{active_basedir}" does not exist!')

        self.session_data_type = session_data_type
        
        # Set first pipeline stage to input:
        self.stage = InputPipelineStage(
            stage_name=f"{self.pipeline_name}_input",
            basedir=active_basedir,
            load_function=load_function,
            post_load_functions=post_load_functions
        )
        if auto_load:
            self.load()

# ____________________________________________________________________________________________________________________ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# LOADING STAGE IMPLEMENTATION                                                                                         #
# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
@define(slots=False, repr=False) # , init=False
class LoadedPipelineStage(LoadableSessionInput, InputPipelineStage):
    """Docstring for LoadedPipelineStage."""
    @classmethod
    def get_stage_identity(cls) -> PipelineStage:
        return PipelineStage.Loaded

    identity: PipelineStage = field(default=PipelineStage.Loaded)
    loaded_data: dict = field(default=None)

    # Custom Fields:
    registered_load_function_dict: dict = field(default=Factory(dict))


    @classmethod
    def init_from_previous_stage(cls, input_stage: InputPipelineStage):
        _obj = cls()
        _obj.stage_name = input_stage.stage_name
        _obj.basedir = input_stage.basedir
        _obj.loaded_data = input_stage.loaded_data
        _obj.post_load_functions = input_stage.post_load_functions # the functions to be called post load
        # Initialize custom fields:
        _obj.registered_load_function_dict = {}
        _obj.register_default_known_load_functions() # registers the default load functions
        return _obj

    
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
        

    def post_load(self, progress_logger=None, debug_print=False):
        """ Called after load is complete to post-process the data """
        if (len(self.post_load_functions) > 0):
            if debug_print:
                print(f'Performing on_post_load(...) with {len(self.post_load_functions)} post_load_functions...')
            if progress_logger is not None:
                progress_logger.debug(f'Performing on_post_load(...) with {len(self.post_load_functions)} post_load_functions...')
            # self.sess = compose_functions(self.post_load_functions, self.sess)
            composed_post_load_function = compose_functions(*self.post_load_functions) # functions are composed left-to-right
            self.sess = composed_post_load_function(self.sess)
            
        else:
            if debug_print:
                print(f'No post_load_functions, skipping post_load.')
            if progress_logger is not None:
                progress_logger.debug(f'No post_load_functions, skipping post_load.')
                

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['registered_load_function_dict']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if 'identity' not in state:
            print(f'unpickling from old NeuropyPipelineStage')
            state['identity'] = None
            state['identity'] = type(self).get_stage_identity()

        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(LoadedPipelineStage, self).__init__() # from 

        self.registered_load_function_dict = {}
        self.register_default_known_load_functions() # registers the default load functions


    





# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class PipelineWithLoadableStage(RegisteredOutputsMixin):
    """ Has a lodable stage. """
    
    @property
    def can_load(self) -> bool:
        """Whether load can be performed."""
        return (self.last_completed_stage >= PipelineStage.Input)

    @property
    def is_loaded(self) -> bool:
        """The is_loaded property."""
        return (self.stage is not None) and (isinstance(self.stage, LoadedPipelineStage))

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
    

    @classmethod
    def perform_load(cls, input_stage) -> LoadedPipelineStage:
        input_stage.load()  # perform the load operation
        # return LoadedPipelineStage(input_stage)  # build the loaded stage
        return LoadedPipelineStage.init_from_previous_stage(input_stage)  # build the loaded stage


    def load(self):
        self.stage.load()  # perform the load operation:
        self.stage = LoadedPipelineStage.init_from_previous_stage(self.stage)  # build the loaded stage
        self.stage.post_load(progress_logger=self.logger)

        
    ## Session passthroughs:
    @function_attributes(tags=['output_files', 'filesystem'], related_items=[])
    def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.stage.get_output_path()

    def get_session_context(self) -> IdentifyingContext:
        """ returns the context of the unfiltered session (self.sess) """
        return self.stage.get_session_context()
    
    def get_session_format_name(self) -> str:
        return self.stage.get_session_format_name()
    

    def get_session_unique_aclu_information(self) -> pd.DataFrame:
        """  Get the aclu information for each aclu in the dataframe. Adds the ['aclu', 'shank', 'cluster', 'qclu', 'neuron_type'] columns """
        return self.stage.get_session_unique_aclu_information()



    def determine_good_aclus_by_qclu(self, included_qclu_values=[1,2,4,9], debug_print:bool=False) -> NDArray:
            """ 
            From all neuron_IDs in the session, get the ones that meet the new qclu criteria (their value is in) `included_qclu_values`
            
            included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
            included_aclus # np.array([  2,   3,   4,   5,   7,   8,   9,  10,  11,  13,  14,  15,  16,  17,  19,  21,  23,  24,  25,  26,  27,  28,  31,  32,  33,  34,  35,  36,  37,  41,  45,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  66,  67,  68,  69,  70,  71,  73,  74,  75,  76,  78,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  92,  93,  96,  97,  98, 100, 102, 105, 107, 108, 109])

            included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2])
            included_aclus # np.array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  81,  82,  83,  85,  86,  90,  92,  93,  96, 105, 109])

            """
            return self.stage.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values, debug_print=debug_print)

