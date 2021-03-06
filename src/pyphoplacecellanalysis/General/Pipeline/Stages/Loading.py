from typing import Callable, List
import dataclasses
from dataclasses import dataclass
from pathlib import Path


from pyphocorehelpers.function_helpers import compose_functions
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage

# # NeuroPy (Diba Lab Python Repo) Loading
# try:
#     from neuropy import core

#     importlib.reload(core)
# except ImportError:
#     sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
#     # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
#     # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
#     print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
#     from neuropy import core

# # Neuropy:
# from neuropy.core.session.data_session_loader import DataSessionLoader
# from neuropy.core.session.dataSession import DataSession
# from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields




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





@dataclass
class InputPipelineStage(LoadableInput, BaseNeuropyPipelineStage):
    """Docstring for InputPipelineStage.
    
    post_load_functions: List[Callable] a list of Callables that accept the loaded session as input and return the potentially modified session as output.
    """
    identity: PipelineStage = PipelineStage.Input
    basedir: Path = Path("")
    load_function: Callable = None
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)



class LoadedPipelineStage(LoadableInput, LoadableSessionInput, BaseNeuropyPipelineStage):
    """Docstring for LoadedPipelineStage."""
    identity: PipelineStage = PipelineStage.Loaded
    loaded_data: dict = None

    def __init__(self, input_stage: InputPipelineStage):
        self.stage_name = input_stage.stage_name
        self.basedir = input_stage.basedir
        self.loaded_data = input_stage.loaded_data
        self.post_load_functions = input_stage.post_load_functions # the functions to be called post load


    def post_load(self, debug_print=False):
        """ Called after load is complete to post-process the data """
        if (len(self.post_load_functions) > 0):
            if debug_print:
                print(f'Performing on_post_load(...) with {len(self.post_load_functions)} post_load_functions...')
            # self.sess = compose_functions(self.post_load_functions, self.sess)
            composed_post_load_function = compose_functions(*self.post_load_functions) # functions are composed left-to-right
            self.sess = composed_post_load_function(self.sess)
            
        else:
            if debug_print:
                print(f'No post_load_functions, skipping post_load.')

        


class PipelineWithInputStage:
    """ Has an input stage. """
    def set_input(self, session_data_type:str='', basedir="", load_function: Callable = None, post_load_functions: List[Callable] = [],
                  auto_load=True, **kwargs):
        """ Called to set the input stage """
        if not isinstance(basedir, Path):
            print(f"basedir is not Path. Converting...")
            active_basedir = Path(basedir)
        else:
            print(f"basedir is already Path object.")
            active_basedir = basedir

        if not active_basedir.exists():
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


       
class PipelineWithLoadableStage:
    """ Has a lodable stage. """
    
    @property
    def can_load(self):
        """Whether load can be performed."""
        return (self.last_completed_stage >= PipelineStage.Input)

    @property
    def is_loaded(self):
        """The is_loaded property."""
        return (self.stage is not None) and (isinstance(self.stage, LoadedPipelineStage))

    @classmethod
    def perform_load(cls, input_stage) -> LoadedPipelineStage:
        input_stage.load()  # perform the load operation
        return LoadedPipelineStage(input_stage)  # build the loaded stage

    def load(self):
        self.stage.load()  # perform the load operation:
        self.stage = LoadedPipelineStage(self.stage)  # build the loaded stage
        self.stage.post_load()

