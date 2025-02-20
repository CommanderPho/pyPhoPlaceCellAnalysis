from typing import Any, Callable, List, Dict, Optional, Union
from types import ModuleType
import dataclasses
from dataclasses import dataclass
from attrs import define, field, Factory
from datetime import datetime
import pathlib
from pathlib import Path
import shutil
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
                pickle.dump(db, dbfile)
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
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

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
}


def loadData(pkl_path, debug_print=False, **kwargs):
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

