from typing import Callable, List
import dataclasses
from dataclasses import dataclass
from datetime import datetime
import pathlib
from pathlib import Path

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.function_helpers import compose_functions
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

# Its important to use binary mode
def saveData(pkl_path, db, should_append=False):
    if should_append:
        file_mode = 'ab' # 'ab' opens the file as binary and appends to the end
    else:
        file_mode = 'w+b' # 'w+b' opens and truncates the file to 0 bytes (overwritting)
    with ProgressMessagePrinter(pkl_path, f"Saving (file mode '{file_mode}')", 'saved session pickle file'):
        with open(pkl_path, file_mode) as dbfile: 
            # source, destination
            pickle.dump(db, dbfile)
            dbfile.close()

def loadData(pkl_path, debug_print=False, **kwargs):
    # for reading also binary mode is important
    db = None
    with ProgressMessagePrinter(pkl_path, 'Loading', 'loaded session pickle file'):
        with open(pkl_path, 'rb') as dbfile:
            try:
                db = pickle.load(dbfile, **kwargs)
                
            except NotImplementedError as err:
                error_message = str(err)
                if 'WindowsPath' in error_message:  # Check if WindowsPath is missing
                    print("Issue with pickled WindowsPath on Linux for path {}, performing pathlib workaround...".format(pkl_path))
                    win_backup = pathlib.WindowsPath  # Backup the WindowsPath definition
                    try:
                        pathlib.WindowsPath = pathlib.PureWindowsPath
                        db = pickle.load(dbfile, **kwargs) # Fails this time if it still throws an error
                    finally:
                        pathlib.WindowsPath = win_backup  # Restore the backup WindowsPath definition
                        
                elif 'PosixPath' in error_message:  # Check if PosixPath is missing
                    # Fixes issue with pickled POSIX_PATH on windows for path.
                    posix_backup = pathlib.PosixPath # backup the PosixPath definition
                    try:
                        pathlib.PosixPath = pathlib.PurePosixPath
                        db = pickle.load(dbfile, **kwargs) # Fails this time if it still throws an error
                    finally:
                        pathlib.PosixPath = posix_backup # restore the backup posix path definition
                                        
                else:
                    print("Unknown issue with pickled path for path {}, performing pathlib workaround...".format(pkl_path))
                    raise
                                
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
@dataclass
class InputPipelineStage(LoadableInput, BaseNeuropyPipelineStage):
    """ The first stage of the NeuropyPipeline. Allows specifying the inputs that will be used.
    
        post_load_functions: List[Callable] a list of Callables that accept the loaded session as input and return the potentially modified session as output.
    """
    identity: PipelineStage = PipelineStage.Input
    basedir: Path = Path("")
    load_function: Callable = None
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)


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
            raise FileExistsError

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
class LoadedPipelineStage(LoadableInput, LoadableSessionInput, BaseNeuropyPipelineStage):
    """Docstring for LoadedPipelineStage."""
    identity: PipelineStage = PipelineStage.Loaded
    loaded_data: dict = None

    def __init__(self, input_stage: InputPipelineStage):
        self.stage_name = input_stage.stage_name
        self.basedir = input_stage.basedir
        self.loaded_data = input_stage.loaded_data
        self.post_load_functions = input_stage.post_load_functions # the functions to be called post load
        # Initialize custom fields:
        self.registered_load_function_dict = {}
        self.register_default_known_load_functions() # registers the default load functions
            
    
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
# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class PipelineWithLoadableStage(RegisteredOutputsMixin):
    """ Has a lodable stage. """
    
    @property
    def can_load(self):
        """Whether load can be performed."""
        return (self.last_completed_stage >= PipelineStage.Input)

    @property
    def is_loaded(self):
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
        return LoadedPipelineStage(input_stage)  # build the loaded stage

    def load(self):
        self.stage.load()  # perform the load operation:
        self.stage = LoadedPipelineStage(self.stage)  # build the loaded stage
        self.stage.post_load(progress_logger=self.logger)

        

