import sys
import os
import pkg_resources # for Slurm templating
from jinja2 import Environment, FileSystemLoader # for Slurm templating
import logging
import socket # for getting hostname in `build_batch_processing_session_task_identifier`
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from pyphocorehelpers.print_helpers import CapturedException
import tables as tb
from copy import deepcopy
import multiprocessing
# import concurrent.futures
# from tqdm import tqdm

from enum import Enum, unique # SessionBatchProgress

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path, set_posix_windows, convert_filelist_to_new_parent, find_matching_parent_path
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
from pyphocorehelpers.function_helpers import function_attributes


# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.core.epoch import Epoch
from neuropy.utils.matplotlib_helpers import matplotlib_file_only
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter

from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session, batch_extended_computations, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

from attrs import define, field, Factory

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations  # for `BatchSessionCompletionHandler`
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations, InstantaneousFiringRatesDataframeAccessor


known_global_data_root_parent_paths = [Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/nfs/turbo/umms-kdiba/Data')]

def get_file_str_if_file_exists(v:Path)->str:
    """ returns the string representation of the resolved file if it exists, or the empty string if not """
    return (str(v.resolve()) if v.exists() else '')

def build_batch_processing_session_task_identifier(session_context: IdentifyingContext) -> str:
    """ Builds an identifier string for logging task progress like 'LNX00052.kdiba.gor01.two.2006-6-07_16-40-19' """
    import socket
    hostname: str = socket.gethostname() # get the system's hostname
    # print(f"Hostname: {hostname}") # Hostname: LNX00052
    session_component: str = session_context.get_description(separator='.') # 'kdiba.gor01.two.2006-6-07_16-40-19'
    return f"{hostname}.{session_component}"

def build_batch_task_logger(session_context: IdentifyingContext, additional_suffix:Optional[str]=None, file_logging_dir=Path('EXTERNAL/TESTING/Logging'), debug_print=False):
    """ Builds a logger for a specific module that logs to console output and a file. 
    History:
        Built from `pyphocorehelpers.print_helpers.build_batch_task_logger` for task building
    
    Testing:
    
        module_logger.debug (f'DEBUG: module_logger: "com.PhoHale.Spike3D.notebook"')
        module_logger.info(f'INFO: module_logger: "com.PhoHale.Spike3D.notebook"')
        module_logger.warning(f'WARNING: module_logger: "com.PhoHale.Spike3D.notebook"')
        module_logger.error(f'ERROR: module_logger: "com.PhoHale.Spike3D.notebook"')
        module_logger.critical(f'CRITICAL: module_logger: "com.PhoHale.Spike3D.notebook"')
        module_logger.exception(f'EXCEPTION: module_logger: "com.PhoHale.Spike3D.notebook"')

    batch_task_logger = build_batch_task_logger()
    """
    # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] %(name)s [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter("%(relativeCreated)d %(name)s]  [%(levelname)-5.5s]  %(message)s")
    
    batch_processing_session_task_identifier:str = build_batch_processing_session_task_identifier(session_context)
    if additional_suffix is not None:
        batch_processing_session_task_identifier = f"{batch_processing_session_task_identifier}.{additional_suffix.lstrip('.')}"
    batch_task_logger = logging.getLogger(f'com.PhoHale.PhoPy3DPositionAnalyis.Batch.runBatch.run_specific_batch.{batch_processing_session_task_identifier}') # create logger
    print(f'build_batch_task_logger(module_name="{batch_processing_session_task_identifier}"):')
    if debug_print:
        print(f'\t batch_task_logger.handlers: {batch_task_logger.handlers}')
    batch_task_logger.handlers = []
    # batch_task_logger.removeHandler()

    if file_logging_dir is not None:
        # file logging enabled:
        # Create logging directory if it doesn't exist
        file_logging_dir.mkdir(parents=True, exist_ok=True)

        # file_logging_dir = Path('EXTERNAL/TESTING/Logging') # 'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\TESTING\Logging'
        module_logging_path = file_logging_dir.joinpath(f'debug_{batch_task_logger.name}.log') # batch_task_logger.name # 'com.PhoHale.Spike3D.notebook'

        # File Logging:    
        print(f'\t Batch Task logger {batch_task_logger.name} has file logging enabled and will log to {str(module_logging_path)}')
        fileHandler = logging.FileHandler(module_logging_path)
        fileHandler.setFormatter(logFormatter)
        batch_task_logger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setFormatter(logFormatter)
    # # batch_task_logger.addHandler(consoleHandler)

    # General Logger Setup:
    batch_task_logger.setLevel(logging.DEBUG)
    batch_task_logger.info(f'==========================================================================================\n========== Module Logger INIT "{batch_task_logger.name}" ==============================')
    return batch_task_logger



@unique
class SessionBatchProgress(Enum):
    """Indicates the progress state for a given session in a batch processing queue """
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

from enum import Enum, auto
class SavingOptions(Enum):
    NEVER = auto()
    IF_CHANGED = auto()
    ALWAYS = auto()

@custom_define(slots=False)
class BatchComputationProcessOptions(HDF_SerializationMixin):
    should_load: bool = serialized_attribute_field() # should try to load from existing results from disk at all
        # never
        # always (fail if loading unsuccessful)
        # always (warning but continue if unsuccessful)
    should_compute: bool = serialized_attribute_field() # should try to run computations (which will just verify that loaded computations are good if that option is true)
        # never
        # if needed (required results are missing)
        # always
    should_save: SavingOptions = serialized_attribute_field(default=SavingOptions.NEVER) # should consider save at all
        # never
        # if changed
        # always



@custom_define(slots=False)
class PipelineCompletionResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Class representing the specific results extratracted from the loaded pipeline and returned as return values from the post-execution callback function. """
    long_epoch_name: str = serialized_attribute_field()
    long_laps: Epoch = serialized_field()
    long_replays: Epoch = serialized_field()
    
    short_epoch_name: str = serialized_attribute_field()
    short_laps: Epoch = serialized_field()
    short_replays: Epoch = serialized_field()
    
    delta_since_last_compute: timedelta = non_serialized_field() #serialized_attribute_field(serialization_fn=HDF_Converter._prepare_datetime_timedelta_value_to_for_hdf_fn)
    outputs_local: Dict[str, Optional[Path]] = non_serialized_field() # serialization_fn=(lambda f, k, v: f[f'{key}/{sub_k}'] = str(sub_v) for sub_k, sub_v in value.items()), is_hdf_handled_custom=True
    outputs_global: Dict[str, Optional[Path]] = non_serialized_field()

    across_session_results: Dict[str, Optional[object]] = non_serialized_field()

     # HDFMixin Conformances ______________________________________________________________________________________________ #
    
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)
        # Finish for the custom properties
        
        #TODO 2023-08-04 12:09: - [ ] Included outputs_local/global

        # with tb.open_file(file_path, mode='a') as f:

        #     outputs_local_key = f"{key}/outputs_local"
        #     an_outputs_local_group = f.create_group(key, 'outputs_local', title='the sessions output file paths.', createparents=True)

        #     value = self.outputs_local
        #     for sub_k, sub_v in value.items():
        #         an_outputs_local_group[f'{outputs_local_key}/{sub_k}'] = str(sub_v) 

        #     an_outputs_global_group = f.create_group(key, 'outputs_global', title='the sessions output file paths.', createparents=True)
        #     value = self.outputs_global
        #     outputs_global_key = f"{key}/outputs_global"
        #     for sub_k, sub_v in value.items():
        #         an_outputs_global_group[f'{outputs_global_key}/{sub_k}'] = str(sub_v) 



# PyTables Definitions for Output Tables: ____________________________________________________________________________ #
class PipelineCompletionResultTable(tb.IsDescription):
    """ PyTables class representing epoch data built from a dictionary. """
    long_epoch_name = tb.StringCol(itemsize=100)
    # long_laps = EpochTable()
    # long_replays = EpochTable()
    long_n_laps = tb.UInt16Col()
    long_n_replays = tb.UInt16Col()
    short_epoch_name = tb.StringCol(itemsize=100)
    # short_laps = EpochTable()
    # short_replays = EpochTable()
    short_n_laps = tb.UInt16Col()
    short_n_replays = tb.UInt16Col()

    delta_since_last_compute = tb.Time64Col()  # Use tb.Time64Col for timedelta
    
    # outputs_local = OutputFilesTable()  
    # outputs_global = OutputFilesTable()
    
    # across_sessions_batch_results_inst_fr_comps = tb.StringCol(itemsize=100)




@custom_define(slots=False)
class BatchRun(HDF_SerializationMixin):
    """An object that manages a Batch of runs for many different session folders."""
    global_data_root_parent_path: Path = serialized_attribute_field(serialization_fn=(lambda f, k, v: str(v.resolve())))
    session_batch_status: Dict[IdentifyingContext, SessionBatchProgress] = serialized_field(default=Factory(dict)) 
    session_batch_basedirs: Dict[IdentifyingContext, Path] = serialized_field(default=Factory(dict))
    session_batch_errors: Dict[IdentifyingContext, Optional[str]] = serialized_field(default=Factory(dict))
    session_batch_outputs: Dict[IdentifyingContext, Optional[PipelineCompletionResult]] = serialized_field(default=Factory(dict)) # optional selected outputs that can hold information from the computation
    enable_saving_to_disk: bool = serialized_attribute_field(default=False) 

    ## TODO: could keep session-specific kwargs to be passed to run_specific_batch(...) as a member variable if needed
    _context_column_names = ['format_name', 'animal', 'exper_name', 'session_name']
    
    # Computed Properties ________________________________________________________________________________________________ #
    @property
    def session_contexts(self):
        """The session_contexts property."""
        return list(self.session_batch_status.keys())

    @classmethod
    def try_init_from_file(cls, global_data_root_parent_path, active_global_batch_result_filename='global_batch_result.pkl', skip_root_path_conversion:bool=False, debug_print:bool=False):
        """ Loads from a previously saved .pkl file if possible, otherwise start fresh by calling `on_needs_create_callback_fn`.

            `on_needs_create_callback_fn`: (global_data_root_parent_path: Path, execute_all:bool = False, extant_batch_run = None, debug_print:bool=False, post_run_callback_fn=None) -> BatchRun: Build `global_batch_run` pre-loading results (before execution)
        """
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
        ## Build Pickle Path:
        finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve() # Use Default
        def _try_load_global_batch_result():
            """ load the batch result initially. 
            Captures: finalized_loaded_global_batch_result_pickle_path, debug_print
            """
            if debug_print:
                print(f'finalized_loaded_global_batch_result_pickle_path: {finalized_loaded_global_batch_result_pickle_path}')
            # try to load an existing batch result:
            try:
                global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print)
                
            except NotImplementedError:
                # Fixes issue with pickled POSIX_PATH on windows for path.                    
                with set_posix_windows():
                    global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print) # Fails this time if it still throws an error

            except (FileNotFoundError, TypeError):
                # loading failed
                print(f'Failure loading {finalized_loaded_global_batch_result_pickle_path}.')
                global_batch_run = None
                
            return global_batch_run

        # BEGIN FUNCTION BODY
        global_batch_run = _try_load_global_batch_result()
        if (global_batch_run is not None) and (not skip_root_path_conversion):
            # One was loaded from file, meaning it has the potential to have the wrong paths. Check.
            global_batch_run.change_global_root_path(global_data_root_parent_path) # Convert the paths to work on the new system:
        else:
            ## Completely fresh, run the initial (pre-loading) results.
            # Build `global_batch_run` pre-loading results (before execution)
            # assert on_needs_create_callback_fn is not None
            
            global_batch_run = run_diba_batch(global_data_root_parent_path, execute_all=False, extant_batch_run=global_batch_run, debug_print=False)
            # print(f'global_batch_result: {global_batch_run}')
            # Save `global_batch_run` to file:
            saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary

        ## I got it doing the bare-minimum loading and computations, so it should be ready to update the laps and constrain the placefields to those. Then we should be able to set up the replays at the same time.
        # finally, we then finish by computing.
        assert global_batch_run is not None
        return global_batch_run

    def to_dataframe(self, expand_context:bool=True, good_only:bool=False) -> pd.DataFrame:
        """Get a dataframe representation of BatchRun."""
        out_df = BatchResultDataframeAccessor.init_from_BatchRun(self, expand_context=expand_context, good_only=good_only)
        out_df = out_df.batch_results.build_all_columns() # this uses the same accessor.
        return out_df

    # Main functionality _________________________________________________________________________________________________ #
    def execute_all(self, use_multiprocessing=True, num_processes=None, included_session_contexts: Optional[List[IdentifyingContext]]=None, session_inclusion_filter:Optional[Callable]=None, **kwargs):
        """ calls `run_specific_batch(...)` for each session context to actually execute the session's run. """
        allow_processing_previously_completed: bool = kwargs.pop('allow_processing_previously_completed', False)
        
        # filter for inclusion here instead of in the loop:  
        if included_session_contexts is not None:
            # use `included_session_contexts` list over the filter function.
            assert session_inclusion_filter is None, f"You cannot provide both a `session_inclusion_filter` and a `included_session_contexts` list. Include one or the other."
            # ready to go
        else:
            ## no included_session_contexts provided.
            if session_inclusion_filter is None:
                ## no filter provided either
                session_inclusion_filter = (lambda curr_session_context, curr_session_status: (curr_session_status != SessionBatchProgress.COMPLETED) or allow_processing_previously_completed)
            else:
                # `session_inclusion_filter` was provided, make sure there is no list.
                assert included_session_contexts is None, f"You cannot provide both a `session_inclusion_filter` and a `included_session_contexts` list. Include one or the other."
                
            # either way now build the contexts list using the filter:
            included_session_contexts: List[IdentifyingContext] = [curr_session_context for curr_session_context, curr_session_status in self.session_batch_status.items() if session_inclusion_filter(curr_session_context, curr_session_status)]

        # Now `included_session_contexts` list should be good either way:
        assert included_session_contexts is not None
        assert isinstance(included_session_contexts, list)
        print(f'Beginning processing with len(included_session_contexts): {len(included_session_contexts)}')        

        if use_multiprocessing:            
            # # pre-2023-08-08 `multiprocessing`-based version _____________________________________________________________________ #
            if num_processes is None:
                num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores

            pool = multiprocessing.Pool(processes=num_processes)
            results = {} # dict form
            
            for curr_session_context in included_session_contexts:
                curr_session_basedir = self.session_batch_basedirs[curr_session_context]
                result = pool.apply_async(run_specific_batch, (self.global_data_root_parent_path, curr_session_context, curr_session_basedir), kwargs) # it can actually take a callback too.
                results[curr_session_context] = result

            pool.close()
            pool.join()

            for session_context, result in results.items():
                status, error, output = result.get()
                self.session_batch_status[session_context] = status
                self.session_batch_errors[session_context] = error
                self.session_batch_outputs[session_context] = output
                
            # # `concurrent.futures` version instead of `multiprocessing` __________________________________________________________ #
            # if num_processes is None:
            #     num_processes = concurrent.futures.ProcessPoolExecutor().max_workers

            # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            #     futures = {executor.submit(run_specific_batch, self.global_data_root_parent_path, curr_session_context, self.session_batch_basedirs[curr_session_context], **kwargs): curr_session_context for curr_session_context in included_session_contexts}

            #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            #         session_context = futures[future]  # get the context for this future
            #         status, error, output = future.result()
            #         self.session_batch_status[session_context] = status
            #         self.session_batch_errors[session_context] = error
            #         self.session_batch_outputs[session_context] = output


        else:
            # No multiprocessing, fall back to the normal way.
            for curr_session_context in included_session_contexts:
                # evaluate a single session
                curr_session_status = self.session_batch_status[curr_session_context]
                curr_session_basedir = self.session_batch_basedirs[curr_session_context]
                self.session_batch_status[curr_session_context], self.session_batch_errors[curr_session_context], self.session_batch_outputs[curr_session_context] = run_specific_batch(self.global_data_root_parent_path, curr_session_context, curr_session_basedir, **kwargs)

    # Updating ___________________________________________________________________________________________________________ #
    def change_global_root_path(self, desired_global_data_root_parent_path):
        """ Changes the self.global_data_root_parent_path for this computer and converts all of the `session_batch_basedirs` paths.

        Modifies:
            self.global_data_root_parent_path
            self.session_batch_basedirs
        """
        if isinstance(desired_global_data_root_parent_path, str):
            desired_global_data_root_parent_path = Path(desired_global_data_root_parent_path)
            
        assert desired_global_data_root_parent_path.exists(), f"the path provide should be the one for the system (and it should exist)"
        if self.global_data_root_parent_path != desired_global_data_root_parent_path:
            print(f'switching data dir path from {str(self.global_data_root_parent_path)} to {str(desired_global_data_root_parent_path)}')
            prev_global_data_root_parent_path = self.global_data_root_parent_path
            
            curr_filelist = list(self.session_batch_basedirs.values())
            try:
                prev_global_data_root_parent_path = self.global_data_root_parent_path # normally this would work
                new_session_batch_basedirs = convert_filelist_to_new_parent(curr_filelist, original_parent_path=prev_global_data_root_parent_path, dest_parent_path=desired_global_data_root_parent_path)
            except ValueError as e:
                # The global_batch_run.global_data_root_parent_path is wrong when:`ValueError: '\\nfs\\turbo\\umms-kdiba\\Data\\KDIBA\\gor01\\one\\2006-6-07_11-26-53' is not in the subpath of '\\home\\halechr\\turbo\\Data' OR one path is relative and the other is absolute.`. Try to find the real parent path.
                prev_global_data_root_parent_path = find_matching_parent_path(known_global_data_root_parent_paths, curr_filelist[0]) # TODO: assumes all have the same root, which is a valid assumption so far. ## prev_global_data_root_parent_path should contain the matching path from the list.
                assert prev_global_data_root_parent_path is not None, f"No matching root parent path could be found!!"
                new_session_batch_basedirs = convert_filelist_to_new_parent(curr_filelist, original_parent_path=prev_global_data_root_parent_path, dest_parent_path=desired_global_data_root_parent_path)
            except Exception as e:
                ## Unhandled Exception
                raise

            print(f'Switched data dir path from "{str(prev_global_data_root_parent_path)}" to "{str(desired_global_data_root_parent_path)}"')
            self.global_data_root_parent_path = desired_global_data_root_parent_path.resolve()
            self.session_batch_basedirs = {ctx:a_basedir.resolve() for ctx, a_basedir in zip(self.session_contexts, new_session_batch_basedirs)} # ctx.format_name, ctx.animal, ctx.exper_name
        else:
            print('no difference between provided and internal paths.')

    def reset_session(self, curr_session_context: IdentifyingContext):
        """ resets all progress, dumping the outputs, errors, etc. """
        self.session_batch_status[curr_session_context] = SessionBatchProgress.NOT_STARTED # set to not started if not present
        self.session_batch_errors[curr_session_context] = None # indicate that there are no errors to start
        self.session_batch_outputs[curr_session_context] = None # indicate that there are no outputs to start

    # Class/Static Functions _____________________________________________________________________________________________ #

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Create a BatchRun object from a dataframe representation.
            Note the outputs are left out of the dataframe.
            
        
        """
        #TODO 2023-06-13 16:31: - [ ] Not yet completed!

        # Extract non-expanded context columns
        non_expanded_context_df = df[['context', 'basedirs', 'status', 'errors']]
        
        # Extract expanded context columns
        expand_context = any(col.startswith('format_name') for col in df.columns)
        if expand_context:
            context_columns = [col for col in df.columns if col.startswith('format_name')]
            expanded_context_df = df[context_columns]
            session_contexts = []
            for _, row in expanded_context_df.iterrows():
                context = {col: row[col] for col in context_columns}
                session_contexts.append(context)
        else:
            session_contexts = []

        #TODO 2023-06-13 16:31: - [ ] Not yet completed, can't re-derive output object from just the number of epochs.
        for index, row in df.iterrows():
            ctx = index
            long_laps = row['n_long_laps']
            long_replays = row['n_long_replays']
            short_laps = row['n_short_laps']
            short_replays = row['n_short_replays']


        # Create BatchRun object
        batch_run = cls()
        batch_run.session_batch_status = dict(zip(non_expanded_context_df['context'], non_expanded_context_df['status']))
        batch_run.session_batch_basedirs = dict(zip(non_expanded_context_df['context'], non_expanded_context_df['basedirs']))
        batch_run.session_batch_errors = dict(zip(non_expanded_context_df['context'], non_expanded_context_df['errors']))
        batch_run.session_batch_outputs = dict(zip(non_expanded_context_df['context'], non_expanded_context_df['errors']))
        
        batch_run.session_contexts = session_contexts
        
        return batch_run
    
    # ==================================================================================================================== #
    # New 2023-06-13 File Loading functions                                                                                #
    # ==================================================================================================================== #
    @classmethod
    def load_batch_progress_df_from_h5(cls, df_path) -> pd.DataFrame:
        """ loads from an .h5 file. """
        try:
            # db = pickle.load(dbfile, **kwargs)
            batch_progress_df = pd.read_hdf(df_path, key='batch_progress_df')
            
        except NotImplementedError as err:
            error_message = str(err)
            if 'WindowsPath' in error_message:  # Check if WindowsPath is missing
                print("Issue with saved WindowsPath on Linux for path {}, performing pathlib workaround...".format(df_path))
                win_backup = pathlib.WindowsPath  # Backup the WindowsPath definition
                try:
                    pathlib.WindowsPath = pathlib.PureWindowsPath
                    # db = pickle.load(dbfile, **kwargs) # Fails this time if it still throws an error
                    batch_progress_df = pd.read_hdf(df_path, key='batch_progress_df')
                finally:
                    pathlib.WindowsPath = win_backup  # Restore the backup WindowsPath definition
                    
            elif 'PosixPath' in error_message:  # Check if PosixPath is missing
                # Fixes issue with saved POSIX_PATH on windows for path.
                posix_backup = pathlib.PosixPath # backup the PosixPath definition
                try:
                    pathlib.PosixPath = pathlib.PurePosixPath
                    # db = pickle.load(dbfile, **kwargs) # Fails this time if it still throws an error
                    batch_progress_df = pd.read_hdf(df_path, key='batch_progress_df')
                finally:
                    pathlib.PosixPath = posix_backup # restore the backup posix path definition
            else:
                print("Unknown issue with saved path for path {}, performing pathlib workaround...".format(df_path))
                raise
        except Exception as e:
            # unhandled exception
            raise

        return batch_progress_df

    @classmethod
    def find_global_root_path(cls, batch_progress_df: Union[pd.DataFrame, list, np.ndarray]) -> str:
        """ extracts the common prefix from the 'basedirs' column of the df and returns it. """
        
        # # os.path.commonprefix version: this one has the potential to return a deeper directory than the real global data path:
        # paths = batch_progress_df['basedirs'].apply(lambda x: str(x)).to_list()
        # common_prefix = os.path.commonprefix(paths) # '/nfs/turbo/umms-kdiba/Data/KDIBA/'

        # Searches `known_global_data_root_parent_paths` to find one that matches:
        if isinstance(batch_progress_df, pd.DataFrame):
            curr_filelist = batch_progress_df['basedirs'].to_list()
        elif isinstance(batch_progress_df, np.ndarray):
            curr_filelist = batch_progress_df.to_list()
        else:
            curr_filelist = batch_progress_df

        assert isinstance(curr_filelist, list), f"curr_filelist must be a list after conversion but {type(curr_filelist)}"
        common_prefix = find_matching_parent_path(known_global_data_root_parent_paths, curr_filelist[0]) # TODO: assumes all have the same root, which is a valid assumption so far. ## prev_global_data_root_parent_path should contain the matching path from the list.
        assert common_prefix is not None, f"No matching root parent path could be found!!"
        common_prefix = common_prefix.resolve()
        assert np.all([v.is_relative_to(common_prefix) for v in curr_filelist]), f"some of the paths don't match the detected prev root! common_prefix: {common_prefix}"
        return common_prefix

    @classmethod
    def convert_filelist_to_new_global_root(cls, existing_session_batch_basedirs, desired_global_data_root_parent_path, old_global_data_root_parent_path=None) -> List[Path]:
            """ converts a list of files List[Path] containing the common parent root specified by `old_global_data_root_parent_path` or inferred from the list itself to a new parent specified by `desired_global_data_root_parent_path` 
            Arguments:
                desired_global_data_root_parent_path: the desired new global_data_root path
                old_global_data_root_parent_path: if provided, the previous global_data_root path that all of the `existing_session_batch_basedirs` were built with. If not specifieed, tries to infer it using `cls.find_global_root_path(batch_progress_df)`
                        
            Usage:

                existing_session_batch_basedirs = list(batch_progress_df['basedirs'].values)
                new_session_batch_basedirs = convert_filelist_to_new_global_root(existing_session_batch_basedirs, desired_global_data_root_parent_path, old_global_data_root_parent_path=old_global_data_root_parent_path)
            """
            if isinstance(desired_global_data_root_parent_path, str):
                desired_global_data_root_parent_path = Path(desired_global_data_root_parent_path)
            assert desired_global_data_root_parent_path.exists(), f"the path provide should be the one for the system (and it should exist)"
            
            ## Path-based method using `convert_filelist_to_new_parent(...)`:
            if old_global_data_root_parent_path is not None:
                source_parent_path = old_global_data_root_parent_path
            else:
                source_parent_path = cls.find_global_root_path(existing_session_batch_basedirs) # Path(r'/media/MAX/cloud/turbo/Data')
                print(f'inferred source_parent_path: {source_parent_path}')
            
            if isinstance(source_parent_path, str):
                source_parent_path = Path(source_parent_path)

            return convert_filelist_to_new_parent(existing_session_batch_basedirs, original_parent_path=source_parent_path, dest_parent_path=desired_global_data_root_parent_path)
            
    @classmethod
    def rebuild_basedirs(cls, batch_progress_df, desired_global_data_root_parent_path, old_global_data_root_parent_path=None):
        """ replaces basedirs with ones that have been rebuilt from the local `global_data_root_parent_path` and hopefully point to extant paths. 
        
        desired_global_data_root_parent_path: the desired new global_data_root path
        old_global_data_root_parent_path: if provided, the previous global_data_root path that all of the `batch_progress_df['basedirs']` were built with. If not specifieed, tries to infer it using `cls.find_global_root_path(batch_progress_df)`
        
        adds: ['locally_folder_exists', 'locally_is_ready']
        updates: ['basedirs']
        
        Usage:
        
            updated_batch_progress_df = rebuild_basedirs(batch_progress_df, global_data_root_parent_path)
            updated_batch_progress_df

        """
        if isinstance(desired_global_data_root_parent_path, str):
            desired_global_data_root_parent_path = Path(desired_global_data_root_parent_path)
        assert desired_global_data_root_parent_path.exists(), f"the path provide should be the one for the system (and it should exist)"
        
        existing_session_batch_basedirs = list(batch_progress_df['basedirs'].values)
        new_session_batch_basedirs = cls.convert_filelist_to_new_global_root(existing_session_batch_basedirs, desired_global_data_root_parent_path, old_global_data_root_parent_path=old_global_data_root_parent_path)

        session_basedir_exists_locally = [a_basedir.resolve().exists() for a_basedir in new_session_batch_basedirs]

        updated_batch_progress_df = deepcopy(batch_progress_df)
        updated_batch_progress_df['basedirs'] = new_session_batch_basedirs
        updated_batch_progress_df['locally_folder_exists'] = session_basedir_exists_locally
        updated_batch_progress_df['locally_is_ready'] = np.logical_and(updated_batch_progress_df.is_ready, session_basedir_exists_locally)
        return updated_batch_progress_df


    # kdiba_vvp01_two_2006-4-10_12-58-3
    # 	outputs_local ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/loadedSessPickle.pkl')}
    # 	outputs_global ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/global_computation_results.pkl'), 'hdf5': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/pipeline_results.h5')}

    def build_output_files_lists(self):
        """
        Uses `global_batch_run.session_batch_outputs`

        session_identifiers, pkl_output_paths, hdf5_output_paths = global_batch_run.build_output_files_lists()

        """
        session_identifiers = []
        pkl_output_paths = []
        hdf5_output_paths = []

        for k, v in self.session_batch_outputs.items():
            # v is PipelineCompletionResult
            if v is not None:
                # print(f'{k}')
                outputs_local = v.outputs_local
                outputs_global= v.outputs_global
                # print(f'\t{outputs_local =}\n\t{outputs_global =}\n\n')
                session_identifiers.append(k)
                if outputs_local.get('pkl', None) is not None:
                    pkl_output_paths.append(outputs_local.get('pkl', None))
                if outputs_global.get('pkl', None) is not None:
                    pkl_output_paths.append(outputs_global.get('pkl', None))
                if outputs_local.get('hdf5', None) is not None:
                    hdf5_output_paths.append(outputs_local.get('hdf5', None))
                if outputs_global.get('hdf5', None) is not None:
                    hdf5_output_paths.append(outputs_global.get('hdf5', None))

        return session_identifiers, pkl_output_paths, hdf5_output_paths

    @function_attributes(short_name=None, tags=['slurm','jobs','files','batch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-09 19:14', related_items=[])
    def generate_batch_slurm_jobs(self, included_session_contexts, output_directory, use_separate_run_directories:bool=True):
        """ Creates a series of standalone scripts (one for each included_session_contexts) in the `output_directory`

        output_directory
        use_separate_run_directories:bool = True - If True, separate directories are made in `output_directory` containing each script for all sessions.


        Usage:
            ## Build Slurm Scripts:
            global_batch_run.generate_batch_slurm_jobs(included_session_contexts, Path('output/generated_slurm_scripts/').resolve(), use_separate_run_directories=True)

        Uses:
            self.global_data_root_parent_path
            self.session_batch_basedirs
            
        """
        
        # Set up Jinja2 environment
        template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
        env = Environment(loader=FileSystemLoader(template_path))
        python_template = env.get_template('slurm_python_template.py.j2')
        slurm_template = env.get_template('slurm_template.sh.j2')


        output_python_scripts = []
        output_slurm_scripts = []
        # Make sure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        for curr_session_context in included_session_contexts:
            curr_session_basedir = self.session_batch_basedirs[curr_session_context]
            if use_separate_run_directories:
                curr_batch_script_rundir = os.path.join(output_directory, f"run_{curr_session_context}")
                os.makedirs(curr_batch_script_rundir, exist_ok=True)
            else:
                curr_batch_script_rundir = output_directory

            # Create the Python script
            python_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.py')
            with open(python_script_path, 'w') as script_file:
                script_content = python_template.render(global_data_root_parent_path=self.global_data_root_parent_path,
                                                        curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                        curr_session_basedir=curr_session_basedir)
                script_file.write(script_content)
            

            # Create the SLURM script
            slurm_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.sh')
            with open(slurm_script_path, 'w') as script_file:
                script_content = slurm_template.render(curr_session_context=f"{curr_session_context}", python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir)
                script_file.write(script_content)

            # Add the output files:
            output_python_scripts.append(python_script_path)
            output_slurm_scripts.append(slurm_script_path)
        
        return included_session_contexts, output_python_scripts, output_slurm_scripts

    # HDFMixin Conformances ______________________________________________________________________________________________ #

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        
        session_contexts: List[IdentifyingContext] = list(self.session_batch_status.keys())
        session_batch_status: List[SessionBatchProgress] = list(self.session_batch_status.values())

        session_batch_basedirs: List[Path] = list(self.session_batch_basedirs.values())
        session_batch_errors: List[Optional[str]] = list(self.session_batch_errors.values())
        session_batch_outputs: List[Optional[PipelineCompletionResult]] = list(self.session_batch_outputs.values())

        assert key == "/", "key must be '/' for this because it's global level."

        ## Don't forget attributes: global_data_root_parent_path
        # Open the HDF5 file for writing
        with tb.open_file(file_path, mode='w') as h5file:
            # Create a new group at the specified key
            root_group = h5file.create_group("/", name='batch_run', title="Pipeline Completion Results")

            # Create a new table for each PipelineCompletionResult object
            files_table = h5file.create_table(root_group, f"batch_run_table", PipelineCompletionResultTable) # the table is actually at the top level yeah? Each session only has one of these?
            
            # table.
            # Iterate through the PipelineCompletionResult objects and store them in the HDF5 file
            for session_context, a_result in zip(session_contexts, session_batch_outputs):
                if a_result is not None:
                    session_context_key: str = session_context.get_description(separator="|", include_property_names=False)
                    session_group_key: str = '/' + session_context_key # 'kdiba/gor01/one/2006-6-08_14-26-15'
                    
                    row = files_table.row

                    # Fill in the fields of the table with data from the PipelineCompletionResult object
                    row['long_epoch_name'] = a_result.long_epoch_name
                    row['long_n_laps'] = a_result.long_laps.n_epochs
                    row['long_n_replays'] = a_result.long_replays.n_epochs
                    
                    row['short_epoch_name'] = a_result.short_epoch_name
                    row['short_n_laps'] = a_result.short_laps.n_epochs
                    row['short_n_replays'] = a_result.short_replays.n_epochs
                    



                    # Convert timedelta to seconds and then to nanoseconds
                    time_in_seconds = a_result.delta_since_last_compute.total_seconds()
                    time_in_nanoseconds = int(time_in_seconds * 1e9)
                    # Convert to np.int64 (64-bit integer) for tb.Time64Col()
                    time_as_np_int64 = np.int64(time_in_nanoseconds)

                    row['delta_since_last_compute'] = time_as_np_int64

                    # Handle outputs_local and outputs_global dictionaries
                    # if result.outputs_local is not None:
                    #     row['outputs_local/filesystem_path'] = list(result.outputs_local.values())
                    #     row['outputs_local/filesystem_path'] = list([str(v) for v in result.outputs_local.values()])


                    #     # # Create a new table for outputs_local
                    #     # output_local_table = h5file.create_table(table, 'outputs_local', OutputFilesTable)
                    #     for resource_name, filesystem_path in result.outputs_local.items():
                    #         row['outputs_local/resource_name'] = resource_name
                    #         output_local_row['resource_name'] = resource_name
                    #         output_local_row['filesystem_path'] = str(filesystem_path)
                    #         # output_local_row.append()
                    #     # # output_local_table.flush()

                    # if result.outputs_global is not None:
                    #     row['outputs_global/resource_name'] = list(result.outputs_global.keys())
                    #     row['outputs_global/filesystem_path'] = list(result.outputs_global.values())

                    #     # Create a new table for outputs_global
                    #     # output_global_table = h5file.create_table(table, 'outputs_global', OutputFilesTable)
                    #     # for resource_name, filesystem_path in result.outputs_global.items():
                    #     #     # output_global_row['resource_name'] = resource_name
                    #     #     # output_global_row['filesystem_path'] = str(filesystem_path)
                    #     #     # output_global_row.append()
                    #     # # output_global_table.flush()


                    # Append the row to the table and flush the changes
                    row.append()
                
            # Flush the table after storing all the data for this result
            files_table.flush()
                    

        files_table_path: Path = Path(file_path).with_name(f"{file_path.stem}_outputfiles.h5")
            
        # Save the remaining attributes (global_data_root_parent_path, etc.) using the HDF_SerializationMixin
        # super().to_hdf(file_path, key=key, **kwargs)

        # Iterate through the PipelineCompletionResult objects and store them in the HDF5 file
        for session_context, a_status, a_basedir, an_error, a_result in zip(session_contexts, session_batch_status, session_batch_basedirs, session_batch_errors, session_batch_outputs):
            session_context_key: str = session_context.get_description(separator="/", include_property_names=False)
            session_group_key: str = '/' + session_context_key # 'kdiba/gor01/one/2006-6-08_14-26-15'
        
            # if a_basedir is not None:
                

            if a_result is not None:
                # result is PipelineCompletionResult and will be written out with its own .to_hdf
                a_result.to_hdf(file_path, f"{session_group_key}/batch_result") 
                # a_result
                # # Handle across_session_results dictionary
                # if result.across_session_results is not None:
                #     ## Write this out to HDF5 file independently using `PipelineCompletionResult.to_hdf(...)`
                #     for key, value in result.across_session_results.items():
                #         if value is not None:
                            
                #             table.row[f'across_sessions_batch_results_inst_fr_comps_{key}'] = value.to_dict()
                                

        print(f'done outputting HDF file.')


@pd.api.extensions.register_dataframe_accessor("batch_results")
class BatchResultDataframeAccessor():
    """ A Pandas pd.DataFrame representation of results from the batch processing of sessions
    # 2023-07-07
    Built from `BatchRun`

    """

    # _required_column_names = ['session_name', 'basedirs', 'status', 'errors']
    _required_column_names = ['context', 'basedirs', 'status', 'errors']


    def __init__(self, pandas_obj):
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj

    @classmethod
    def init_from_BatchRun(cls, batchrun_obj: BatchRun, expand_context:bool=True, good_only:bool=False) -> pd.DataFrame:
        """Get a dataframe representation of BatchRun."""
        non_expanded_context_df = pd.DataFrame({'context': batchrun_obj.session_batch_status.keys(),
                'basedirs': batchrun_obj.session_batch_basedirs.values(),
                'status': batchrun_obj.session_batch_status.values(),
                'errors': batchrun_obj.session_batch_errors.values()})
        
        if expand_context:
            out_df = HDF_Converter.expand_dataframe_session_context_column(non_expanded_context_df, session_uid_column_name='context')

        else:
            out_df = non_expanded_context_df

        ## Add lap/replay counts:
        out_df = pd.concat((out_df, cls.build_batch_lap_replay_counts_df(batchrun_obj)), axis=1) # don't need multiple concatenation operations probably
        
        ## Add is_ready
        cls.post_load_find_usable_sessions(out_df, min_required_replays_or_laps=5)
        
        if good_only:
            # Get only the good (is_ready) sessions
            out_df = out_df[out_df['is_ready']]
            
        return out_df

    ## Add detected laps/replays to the batch_progress_df:
    @classmethod
    def build_batch_lap_replay_counts_df(cls, global_batch_run: BatchRun):
        """ Adds detected laps/replays to the batch_progress_df. returns lap_replay_counts_df """
        out_counts = []
        out_new_column_names = ['n_long_laps', 'n_long_replays', 'n_short_laps', 'n_short_replays']
        for ctx, output_v in global_batch_run.session_batch_outputs.items():
            if output_v is not None:
                # {long_epoch_name:(long_laps, long_replays), short_epoch_name:(short_laps, short_replays)}
                # Extracting (long_laps, long_replays) tuple
                long_laps = output_v.long_laps
                long_replays = output_v.long_replays

                # Extracting (short_laps, short_replays) tuple
                short_laps = output_v.short_laps
                short_replays = output_v.short_replays
                out_counts.append((long_laps.n_epochs, long_replays.n_epochs, short_laps.n_epochs, short_replays.n_epochs))
            else:
                out_counts.append((0, 0, 0, 0))
        return pd.DataFrame.from_records(out_counts, columns=out_new_column_names)
                
    @classmethod
    def post_load_find_usable_sessions(cls, batch_progress_df, min_required_replays_or_laps=5, require_user_annotations=False):
        """ updates batch_progress_df['is_ready'] and returns only the good frames.
        Called by cls.init_from_BatchRun(...)
        
        """
        has_no_errors = np.array([(an_err_v is None) for an_err_v in batch_progress_df['errors'].to_numpy()])
        has_required_laps_and_replays = np.all((batch_progress_df[['n_long_laps','n_long_replays','n_short_laps','n_short_replays']].to_numpy() >= min_required_replays_or_laps), axis=1)
        if require_user_annotations:
            has_required_user_annotations = batch_progress_df['has_user_replay_annotations'].to_numpy()
        ## Adds 'is_ready' to the dataframe to indicate that all required properties are intact and that it's ready to process further:
        batch_progress_df['is_ready'] = np.logical_and(has_no_errors, has_required_laps_and_replays) # Add 'is_ready' column
        if require_user_annotations:
            batch_progress_df['is_ready'] = np.logical_and(batch_progress_df['is_ready'], has_required_user_annotations)
            
        good_batch_progress_df = deepcopy(batch_progress_df)
        good_batch_progress_df = good_batch_progress_df[good_batch_progress_df['is_ready']]
        return good_batch_progress_df

    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('cell_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        # assert np.all(np.isin(obj.columns, cls._required_column_names))
        # TODO
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming

    @property
    def is_valid(self):
        """ The dataframe is valid (because it passed _validate(...) in __init__(...) so just return True."""
        return True

    def build_all_columns(self):
        """ builds the optional output columns """
        self._build_output_files_list()
        self._build_ripple_result_path()
        # self._build_minimal_session_identifiers_list()
        ## TODO: append to the non-dataframe object?
        user_is_replay_good_annotations = UserAnnotationsManager.get_user_annotations()
        self._obj['has_user_replay_annotations'] = [(a_row_context.adding_context_if_missing(display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections') in user_is_replay_good_annotations) for a_row_context in self._obj['context']]

        user_override_dict_annotations = UserAnnotationsManager.get_hardcoded_specific_session_override_dict()
        ## Get specific grid_bin_bounds overrides from the `UserAnnotationsManager.get_hardcoded_specific_session_override_dict()`
        self._obj['has_user_grid_bin_bounds_annotations'] = [(user_override_dict_annotations.get(a_row_context, {}).get('grid_bin_bounds', None) is not None) for a_row_context in self._obj['context']]
        

        return self._obj


    def convert_path_columns_to_str(self) -> pd.DataFrame:
        """ converts the PosixPath columns to str for serialization/pickling 

        Usage:
            batch_progress_df = batch_progress_df.batch_results.convert_path_columns_to_str()
            batch_progress_df
        """
        potential_path_columns = ['basedirs', 'global_computation_result_file', 'loaded_session_pickle_file', 'ripple_result_file']
        for a_path_column_name in potential_path_columns:
            self._obj[a_path_column_name] = [str(v) for v in self._obj[a_path_column_name].values]

        return self._obj


    ## Build a list of the output files for the good sessions:
    @function_attributes(short_name=None, tags=['batch', 'results', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-06 14:09', related_items=[])
    def _build_output_files_list(self, save_text_file=False):
        ## Build a list of the output files for the good sessions:
        good_only_batch_progress_df = self._obj
        session_result_paths = [get_file_str_if_file_exists(v.joinpath(f'loadedSessPickle.pkl')) for v in list(good_only_batch_progress_df.basedirs.values)]
        global_computation_result_paths = [get_file_str_if_file_exists(v.joinpath(f'output/global_computation_results.pkl').resolve()) for v in list(good_only_batch_progress_df.basedirs.values)]

        # Add dataframe columns:
        good_only_batch_progress_df['global_computation_result_file'] = global_computation_result_paths
        good_only_batch_progress_df['loaded_session_pickle_file'] = session_result_paths
        
        # Write out a GreatlakesOutputs.txt file:
        if save_text_file:
            with open('GreatlakesOutputs.txt','w') as f:
                f.write('\n'.join(session_result_paths + global_computation_result_paths))
                # f.write('\n'.join())
        return (session_result_paths, global_computation_result_paths)
        
    @function_attributes(short_name=None, tags=['ripple', 'batch', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-06 14:09', related_items=[])
    def _build_ripple_result_path(self):
        """
        Usage:
            session_externally_computed_ripple_paths = _build_ripple_result_path(good_only_batch_progress_df)
            session_externally_computed_ripple_paths

        """
        def _find_best_ripple_result_path(a_path: Path) -> Optional[Path]:
            _temp_found_path = a_path.joinpath('ripple_df.pkl')
            if _temp_found_path.exists():
                return _temp_found_path.resolve()
            ## try the '.ripple.npy' ripples:
            _temp_found_path = a_path.joinpath(a_path.name).with_suffix('.ripple.npy')
            if _temp_found_path.exists():
                return _temp_found_path.resolve()
            else:
                return None # could not find the file.
        
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        good_only_batch_progress_df = self._obj
        session_ripple_result_paths: List[Optional[Path]] = [_find_best_ripple_result_path(v) for v in list(good_only_batch_progress_df.basedirs.values)]
        good_only_batch_progress_df['ripple_result_file'] = [str(v or '') for v in session_ripple_result_paths]
        return session_ripple_result_paths
        # global_computation_result_paths = [str(v.joinpath(f'output/global_computation_results.pkl').resolve()) for v in list(good_only_batch_progress_df.basedirs.values)]


    def _build_minimal_session_identifiers_list(self):
        """Build a list of short unique identifiers for the good sessions:
        Adds Column: ['context_minimal_name']
        
        ['a0s0', 'a0s1', 'a0s2', 'a0s3', 'a0s4', 'a0s5', 'a0s6', ... 'a2s10', 'a2s11', 'a2s12', 'a2s13', 'a2s14', 'a2s15', 'a2s16', 'a2s17', 'a2s18', 'a2s19']
        
        TODO: Critical: this 
        #TODO 2023-07-20 21:23: - [ ] This needs to only be ran on a dataframe containing all of the sessions! If it's filtered at all, the session numbers will vary depending on how it's filtered!
        
        """
        
        df = self._obj
        # Extract unique values for each column
        unique_format_names = df['format_name'].unique()
        unique_animals = df['animal'].unique()
        unique_exper_names = df['exper_name'].unique()
        unique_session_names = df['session_name'].unique()

        # Create mapping to shorthand notation for each column
        format_name_mapping = {name: f'f{i}' for i, name in enumerate(unique_format_names)}
        animal_mapping = {name: f'a{i}' for i, name in enumerate(unique_animals)}
        exper_name_mapping = {name: f'e{i}' for i, name in enumerate(unique_exper_names)}
        session_name_mapping = {name: f's{i}' for i, name in enumerate(unique_session_names)}

        # Create a mapping for 'session_name' within each 'animal'
        # animal_session_mapping = {animal: {session: f'{animal[0]}{i}s{j}' for j, session in enumerate(df[df['animal'] == animal]['session_name'].unique())} for i, animal in enumerate(df['animal'].unique())} # 'g0s0'
        animal_session_mapping = {animal: {session: f'{animal_mapping[animal]}s{j}' for j, session in enumerate(df[df['animal'] == animal]['session_name'].unique())} for i, animal in enumerate(df['animal'].unique())} # 'g0s0'

        # Replace original values with shorthand notation
        for animal, session_mapping in animal_session_mapping.items():
            # df.loc[df['animal'] == animal, 'session_name'] = df.loc[df['animal'] == animal, 'session_name'].replace(session_mapping)
            df.loc[df['animal'] == animal, 'context_minimal_name'] = df.loc[df['animal'] == animal, 'session_name'].replace(session_mapping)

        return df['context_minimal_name']


    def export_csv(self, global_data_root_parent_path: Path, csv_batch_filename:str) -> Path:
        # Export CSV:
        # pickle path is: `global_batch_result_file_path`
        csv_batch_filename = csv_batch_filename.replace('.pkl', '.csv')
        global_batch_result_CSV_export_file_path = Path(global_data_root_parent_path).joinpath(csv_batch_filename).resolve() # Use Default
        print(f'global_batch_result_CSV_export_file_path: {global_batch_result_CSV_export_file_path}')
        self._obj.to_csv(global_batch_result_CSV_export_file_path)
        return global_batch_result_CSV_export_file_path

    @staticmethod
    def dataframe_functions_test():
        """ 2023-06-13 - Tests loading saved .h5 `global_batch_result` Dataframe. And updating it for the local platform.

        #TODO 2023-06-13 18:09: - [ ] Finish this implementation up and make decision deciding how to use it
            
        """
        global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"

        ## Build Pickle Path:
        pkl_path = 'global_batch_result_2023-06-08.pkl'
        csv_path = 'global_batch_result_2023-06-08.csv'
        h5_path = 'global_batch_result_2023-06-08.h5'

        global_batch_result_file_path = Path(global_data_root_parent_path).joinpath(h5_path).resolve() # Use Default

        batch_progress_df = BatchRun.load_batch_progress_df_from_h5(global_batch_result_file_path)
        batch_progress_df = BatchRun.rebuild_basedirs(batch_progress_df, global_data_root_parent_path)

        good_only_batch_progress_df = batch_progress_df[batch_progress_df['locally_is_ready']].copy()
        return good_only_batch_progress_df, batch_progress_df




@define(slots=False, repr=False)
class BatchSessionCompletionHandler:
    """ handles completion of a single session's batch processing. 

    Allows accumulating results across sessions and runs.

    
    Usage:
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchSessionCompletionHandler
        
    """

    # Completion Result object returned from callback ____________________________________________________________________ #

    # General:
    debug_print: bool = field(default=False)
    fail_on_exception: bool = field(default=False) # whether to raise exceptions that occur during the callback completion handler or not.

    force_reload_all: bool = field(default=False)
    saving_mode: PipelineSavingScheme = field(default=PipelineSavingScheme.SKIP_SAVING)
    
    # Multiprocessing    
    use_multiprocessing: bool = field(default=False) 
    num_processes: Optional[int] = field(default=None)

    # Computations
    enable_full_pipeline_in_ram: bool = field(default=False)
    
    override_session_computation_results_pickle_filename: Optional[str] = field(default=None) # 'output/loadedSessPickle.pkl'

    session_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))

    global_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))
    extended_computations_include_includelist: list = field(default=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis', 'long_short_decoding_analyses', 'long_short_post_decoding', 'long_short_inst_spike_rate_groups']) # do only specifiedl
    force_global_recompute: bool = field(default=False)
    override_global_computation_results_pickle_path: Optional[Path] = field(default=None)

    # Figures:
    should_perform_figure_generation_to_file: bool = field(default=True) # controls whether figures are generated to file
    should_generate_all_plots: bool = field(default=False) # controls whether all plots are generated (when True) or if only non-Neptune paper figure specific plots are generated. Has no effect if self.should_perform_figure_generation_to_file is False.
    
    
    # Cross-session Results:
    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation

    @classmethod
    def post_compute_validate(cls, curr_active_pipeline) -> bool:
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
        LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate()
        # 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.
        was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
        print(f'were pipeline preprocessing parameters missing and updated?: {was_updated}')

        ## BUG 2023-05-25 - Found ERROR for a loaded pipeline where for some reason the filtered_contexts[long_epoch_name]'s actual context was the same as the short maze ('...maze2'). Unsure how this happened.
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # assert long_epoch_context.filter_name == long_epoch_name, f"long_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}"
        if long_epoch_context.filter_name != long_epoch_name:
            print(f"WARNING: filtered_contexts[long_epoch_name]'s actual context name is incorrect. \n\tlong_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}\n\tUpdating it. (THIS IS A HACK)")
            # fix it if broken
            long_epoch_context.filter_name = long_epoch_name
            was_updated = True

        return was_updated


    def try_complete_figure_generation_to_file(self, curr_active_pipeline, enable_default_neptune_plots=False):
        try:
            ## To file only:
            with matplotlib_file_only():
                # Perform non-interactive Matplotlib operations with 'AGG' backend
                # neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None)
                main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots=enable_default_neptune_plots, save_figures_only=True, save_figure=True, )
                
            # IF thst's done, clear all the plots:
            # from matplotlib import pyplot as plt
            # plt.close('all') # this takes care of the matplotlib-backed figures.
            curr_active_pipeline.clear_display_outputs()
            curr_active_pipeline.clear_registered_output_files()
            return True # completed successfully (without raising an error at least).
        
        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'main_complete_figure_generations failed with exception: {e}')
            if self.fail_on_exception:
                raise e.exc

            return False


    def try_output_neruon_identity_table_to_File(self, file_path, curr_active_pipeline):
        try:
            session_context = curr_active_pipeline.get_session_context() 
            session_group_key: str = "/" + session_context.get_description(separator="/", include_property_names=False) # 'kdiba/gor01/one/2006-6-08_14-26-15'
            session_uid: str = session_context.get_description(separator="|", include_property_names=False)

            AcrossSessionsResults.build_neuron_identity_table_to_hdf(file_path, key=session_group_key, spikes_df=curr_active_pipeline.sess.spikes_df, session_uid=session_uid)
            return True # completed successfully

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'try_output_neruon_identity_table_to_File failed with exception: {e}')
            # raise e
            return False


    def try_compute_global_computations_if_needed(self, curr_active_pipeline, curr_session_context):
        """ tries to load/compute the global computations if needed depending on the self.global_computations_options specifications.
        
        Updates the passed `curr_active_pipeline` pipeline object.

        If computations are loaded, they are loaded via `curr_active_pipeline.load_pickled_global_computation_results(...)`
        If computations are needed, they are performed with the `batch_extended_computations(...)` function.

        
        """
        if self.global_computations_options.should_load:
            if not self.force_global_recompute: # not just force_reload, needs to recompute whenever the computation fails.
                try:
                    curr_active_pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=self.override_global_computation_results_pickle_path)
                except Exception as e:
                    exception_info = sys.exc_info()
                    e = CapturedException(e, exception_info)
                    print(f'cannot load global results: {e}')
                    if self.fail_on_exception:
                        raise e.exc

        if self.global_computations_options.should_save == SavingOptions.ALWAYS
            assert self.global_computations_options.should_compute, f"currently  SavingOptions.ALWAYS requires that self.global_computations_options.should_compute == True also but this is not the case!"

        if self.global_computations_options.should_compute:
            try:
                # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
                curr_active_pipeline.reload_default_computation_functions()
                newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=self.extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=self.force_global_recompute, debug_print=False)
                #TODO 2023-07-11 19:20: - [ ] We want to save the global results if they are computed, but we don't want them to be needlessly written to disk even when they aren't changed.

                if (len(newly_computed_values) > 0):
                    print(f'newly_computed_values: {newly_computed_values}. Saving global results...')
                    if (self.saving_mode.value == 'skip_saving'):
                        print(f'WARNING: supposed to skip_saving because of self.saving_mode: {self.saving_mode} but supposedly has new global results! Figure out if these are actually new.')
                    if self.global_computations_options.should_save != SavingOptions.NEVER:
                        try:
                            # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                            # Try to write out the global computation function results:
                            curr_active_pipeline.save_global_computation_results()
                        except Exception as e:
                            print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                            print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
                            if self.fail_on_exception:
                                raise e.exc
                    else:
                        print(f'\n\n!!WARNING!!: self.global_computations_options.should_save == False, so the global results are unsaved!')
                else:
                    print(f'no changes in global results.')
                    if self.global_computations_options.should_save == SavingOptions.ALWAYS:
                        print(f'Saving mode == ALWAYS so trying to save despite no changes.')
                        try:
                            # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                            # Try to write out the global computation function results:
                            curr_active_pipeline.save_global_computation_results()
                        except Exception as e:
                            print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                            if self.fail_on_exception:
                                raise e.exc
                            
            except Exception as e:
                ## TODO: catch/log saving error and indicate that it isn't saved.
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'ERROR SAVING GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: {curr_session_context}. error: {e}')
                if self.fail_on_exception:
                    raise e.exc


    def try_export_pipeline_hdf5_if_needed(self, curr_active_pipeline, curr_session_context) -> Tuple[Optional[Path], Optional[CapturedException]]:
        """ Export the pipeline's HDF5 as 'pipeline_results.h5' """
        hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('pipeline_results.h5').resolve()
        print(f'pipeline hdf5_output_path: {hdf5_output_path}')
        e = None
        try:
            curr_active_pipeline.export_pipeline_to_h5()
            return (hdf5_output_path, None)
        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying to build the session HDF output for {curr_session_context}")
            if self.fail_on_exception:
                raise e.exc
            hdf5_output_path = None # set to None because it failed.
            return (hdf5_output_path, e)
    

    ## Main function that's called with the complete pipeline:
    def on_complete_success_execution_session(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline) -> PipelineCompletionResult:
        """ called when the execute_session completes like:
            `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`
            
            Meant to be assigned like:
            , post_run_callback_fn=_on_complete_success_execution_session
            
            Captures nothing.
            
            from Spike3D.scripts.run_BatchAnalysis import _on_complete_success_execution_session
            
            
            LOGIC: really we want to recompute global whenever local is recomputed.
            
            
        """
        print(f'on_complete_success_execution_session(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
        # print(f'curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, n_long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, n_long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = self.post_compute_validate(curr_active_pipeline)
        except Exception as e:
            exception_info = sys.exc_info()
            an_err = CapturedException(e, exception_info)
            print(f'self.post_compute_validate(...) failed with exception: {an_err}')
            raise 

        ## Save the pipeline since that's disabled by default now:
        try:
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.override_session_computation_results_pickle_filename) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')
            if self.fail_on_exception:
                raise e.exc

        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True

        self.try_compute_global_computations_if_needed(curr_active_pipeline, curr_session_context=curr_session_context)     

        # ### Programmatic Figure Outputs:
        if self.should_perform_figure_generation_to_file:
            self.try_complete_figure_generation_to_file(curr_active_pipeline, enable_default_neptune_plots=self.should_generate_all_plots)
        else:
            print(f'skipping figure generation because should_perform_figure_generation_to_file == False')



        ### Aggregate Outputs specific computations:

        ## Get some more interesting session properties:
        
        
        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        
        print(f'\t time since last computation: {delta_since_last_compute}')

        # Export the pipeline's HDF5:
        hdf5_output_path, hdf5_output_err = self.try_export_pipeline_hdf5_if_needed(curr_active_pipeline=curr_active_pipeline, curr_session_context=curr_session_context)
        

        print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
        ## Specify the output file:
        common_file_path = Path('output/active_across_session_scatter_plot_results.h5')
        print(f'common_file_path: {common_file_path}')
        InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path, file_mode='a')


        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            
            if not self.use_multiprocessing:
                # Only modify self in non-multiprocessing mode (only shows 1 always).
                self.across_sessions_instantaneous_fr_dict[curr_session_context] = _out_inst_fr_comps # instantaneous firing rates for this session, doesn't work in multiprocessing mode.
                print(f'\t\t Now have {len(self.across_sessions_instantaneous_fr_dict)} entries in self.across_sessions_instantaneous_fr_dict!')
                
            # LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            # LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
            print(f'\t\t done (success).') 

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            if self.fail_on_exception:
                raise e.exc
            _out_inst_fr_comps = None
            
        # On large ram systems, we can return the whole pipeline?
        
        if self.enable_full_pipeline_in_ram:
            across_session_results_extended_dict = {'curr_active_pipeline': curr_active_pipeline}
        else:
            across_session_results_extended_dict = {}
            
        return PipelineCompletionResult(long_epoch_name=long_epoch_name, long_laps=long_laps, long_replays=long_replays,
                                           short_epoch_name=short_epoch_name, short_laps=short_laps, short_replays=short_replays,
                                           delta_since_last_compute=delta_since_last_compute,
                                           outputs_local={'pkl': curr_active_pipeline.pickle_path},
                                            outputs_global={'pkl': curr_active_pipeline.global_computation_results_pickle_path, 'hdf5': hdf5_output_path},
                                            across_session_results={'inst_fr_comps': _out_inst_fr_comps, **across_session_results_extended_dict})
                                          


from datetime import datetime, timedelta
from pyphocorehelpers.print_helpers import CapturedException

@define(slots=False, repr=False)
class HDFSpecificBatchSessionCompletionHandler(BatchSessionCompletionHandler):
    """ 2023-08-25 - an alternative completion handler that just the .h5 stuff. 
    
    """
    
    # Cross-session Results:
    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation

    ## Main function that's called with the complete pipeline:
    def on_complete_success_execution_session(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline) -> PipelineCompletionResult:
        """ called when the execute_session completes like:
            `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`
            
            Meant to be assigned like:
            , post_run_callback_fn=_on_complete_success_execution_session
            
            Captures nothing.
            
            from Spike3D.scripts.run_BatchAnalysis import _on_complete_success_execution_session
            
            
            LOGIC: really we want to recompute global whenever local is recomputed.
            
            
        """
        print(f'HDFProcessing.on_complete_success_execution_session(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
        # print(f'curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, n_long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, n_long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = self.post_compute_validate(curr_active_pipeline)
        except Exception as e:
            exception_info = sys.exc_info()
            an_err = CapturedException(e, exception_info)
            print(f'self.post_compute_validate(...) failed with exception: {an_err}')
            raise 

        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        print(f'\t time since last computation: {delta_since_last_compute}')
        
        ## Save the pipeline since that's disabled by default now:
        try:
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.override_session_computation_results_pickle_filename) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')


        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True


        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True

        self.try_compute_global_computations_if_needed(curr_active_pipeline, curr_session_context=curr_session_context)

        ### Aggregate Outputs specific computations:

        ## Get some more interesting session properties:
        
        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        
        print(f'\t time since last computation: {delta_since_last_compute}')

        # Export the pipeline's HDF5:
        hdf5_output_path, hdf5_output_err = self.try_export_pipeline_hdf5_if_needed(curr_active_pipeline=curr_active_pipeline, curr_session_context=curr_session_context)
        
        # ## Specify the output file:
        # common_file_path = Path('output/active_across_session_scatter_plot_results.h5')
        # print(f'common_file_path: {common_file_path}')
        # InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path, file_mode='a')

        across_session_results_extended_dict = {}
            
        return PipelineCompletionResult(long_epoch_name=long_epoch_name, long_laps=long_laps, long_replays=long_replays,
                                           short_epoch_name=short_epoch_name, short_laps=short_laps, short_replays=short_replays,
                                           delta_since_last_compute=delta_since_last_compute,
                                           outputs_local={'pkl': curr_active_pipeline.pickle_path},
                                            outputs_global={'pkl': curr_active_pipeline.global_computation_results_pickle_path, 'hdf5': hdf5_output_path},
                                            across_session_results=across_session_results_extended_dict)
                                          




# ==================================================================================================================== #
# Global/Helper Functions                                                                                              #
# ==================================================================================================================== #
@function_attributes(short_name='run_diba_batch', tags=['batch', 'automated', 'kdiba'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def run_diba_batch(global_data_root_parent_path: Path, execute_all:bool = False, extant_batch_run = None, debug_print:bool=False, post_run_callback_fn=None):
    """ 
    from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, run_diba_batch, run_specific_batch

    """
    # ==================================================================================================================== #
    # Load Data                                                                                                            #
    # ==================================================================================================================== #
    # global_data_root_parent_path = Path(r'W:\Data') # Windows Apogee
    # global_data_root_parent_path = Path(r'/media/MAX/Data') # Diba Lab Workstation Linux
    # global_data_root_parent_path = Path(r'/Volumes/MoverNew/data') # rMBP
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"

    if extant_batch_run is None:
        print(f'creating new batch_run')
        active_batch_run = BatchRun(global_data_root_parent_path=global_data_root_parent_path)
    else:
        print(f'resusing extant_batch_run: {extant_batch_run}')
        active_batch_run = extant_batch_run


    active_data_mode_name = 'kdiba'

    ## Data must be pre-processed using the MATLAB script located here: 
    #     neuropy/data_session_pre_processing_scripts/KDIBA/IIDataMat_Export_ToPython_2022_08_01.m
    # From pre-computed .mat files:

    local_session_root_parent_context = IdentifyingContext(format_name=active_data_mode_name) # , animal_name='', configuration_name='one', session_name=self.session_name
    local_session_root_parent_path = global_data_root_parent_path.joinpath('KDIBA')

    animal_names = ['gor01', 'vvp01', 'pin01']
    experiment_names_lists = [['one', 'two'], ['one', 'two'], ['one']] # there is no 'two' for animal 'pin01'
    exclude_lists = [['PhoHelpers', 'Spike3D-Minimal-Test', 'Unused'], [], [], [], ['redundant','showclus','sleep','tmaze']]

    for animal_name, an_experiment_names_list, exclude_list in zip(animal_names, experiment_names_lists, exclude_lists):
        for an_experiment_name in an_experiment_names_list:
            local_session_parent_context = local_session_root_parent_context.adding_context(collision_prefix='animal', animal=animal_name, exper_name=an_experiment_name)
            local_session_parent_path = local_session_root_parent_path.joinpath(local_session_parent_context.animal, local_session_parent_context.exper_name)
            local_session_paths_list, local_session_names_list =  find_local_session_paths(local_session_parent_path, exclude_list=exclude_list)

            if debug_print:
                print(f'local_session_paths_list: {local_session_paths_list}')
                print(f'local_session_names_list: {local_session_names_list}')

            ## Build session contexts list:
            local_session_contexts_list = [local_session_parent_context.adding_context(collision_prefix='sess', session_name=a_name) for a_name in local_session_names_list] # [IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>, ..., IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-13_14-42-6')>]

            ## Initialize `session_batch_status` with the NOT_STARTED status if it doesn't already have a different status
            for curr_session_basedir, curr_session_context in zip(local_session_paths_list, local_session_contexts_list):
                # basedir might be different (e.g. on different platforms), but context should be the same
                curr_session_status = active_batch_run.session_batch_status.get(curr_session_context, None)
                if curr_session_status is None:
                    active_batch_run.session_batch_basedirs[curr_session_context] = curr_session_basedir # use the current basedir if we're compute from this machine instead of loading a previous computed session
                    active_batch_run.session_batch_status[curr_session_context] = SessionBatchProgress.NOT_STARTED # set to not started if not present
                    active_batch_run.session_batch_errors[curr_session_context] = None # indicate that there are no errors to start
                    active_batch_run.session_batch_outputs[curr_session_context] = None # indicate that there are no outputs to start

                    ## TODO: 2023-03-14 - Kick off computation?
                    if execute_all:
                        active_batch_run.session_batch_status[curr_session_context], active_batch_run.session_batch_errors[curr_session_context], active_batch_run.session_batch_outputs[curr_session_context] = run_specific_batch(active_batch_run.global_data_root_parent_path, curr_session_context, curr_session_basedir, post_run_callback_fn=post_run_callback_fn)

                else:
                    print(f'EXTANT SESSION! curr_session_context: {curr_session_context} curr_session_status: {curr_session_status}, curr_session_errors: {active_batch_run.session_batch_errors.get(curr_session_context, None)}')
                    ## TODO 2023-04-19: shouldn't computation happen here too if needed?


    ## end for
    return active_batch_run


@function_attributes(short_name='run_specific_batch', tags=['batch', 'automated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def run_specific_batch(global_data_root_parent_path: Path, curr_session_context: IdentifyingContext, curr_session_basedir: Path, force_reload=True, post_run_callback_fn:Optional[Callable]=None, **kwargs):
    """ For a specific session (identified by the session context) - calls batch_load_session(...) to get the curr_active_pipeline.
            - Then calls `post_run_callback_fn(...)
            
    History:
        Used to take a `active_batch_run: BatchRun` as the first parameter, but to remove the coupling I replaced it.
        
    """

    curr_task_logger = build_batch_task_logger(session_context=curr_session_context) # create logger , file_logging_dir=
    _line_sweep = '=========================='
    ## REPLACES THE `print` function within this scope
    def new_print(*args):
        # Call both regular print and logger.info
        print(*args)
        curr_task_logger.info(*args)

    # Replace the print function within this scope
    # _backup_print = print
    # print = new_print


    new_print(f'{_line_sweep} runBatch STARTING {_line_sweep}')
    new_print(f'\tglobal_data_root_parent_path: {str(global_data_root_parent_path)}')
    new_print(f'\tsession_context: {curr_session_context}')
    new_print(f'\tsession_basedir: {str(curr_session_basedir)}')    
    new_print('__________________________________________________________________')

    if not isinstance(global_data_root_parent_path, Path):
        global_data_root_parent_path = Path(global_data_root_parent_path).resolve()
        
    if not isinstance(curr_session_basedir, Path):
        curr_session_basedir = Path(curr_session_basedir).resolve()


    ## Extract the default session loading vars from the session context:
    basedir = curr_session_basedir
    new_print(f'basedir: {str(basedir)}')
    active_data_mode_name = curr_session_context.format_name
    new_print(f'active_data_mode_name: {active_data_mode_name}')
    # post_run_callback_fn = kwargs.pop('post_run_callback_fn', None)
    post_run_callback_fn_output = None
    
    # ==================================================================================================================== #
    # Load Pipeline                                                                                                        #
    # ==================================================================================================================== #
    # epoch_name_includelist = ['maze']
    epoch_name_includelist = kwargs.pop('epoch_name_includelist', None)
    active_computation_functions_name_includelist = kwargs.pop('computation_functions_name_includelist', None) or ['_perform_baseline_placefield_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                        ]
    
    saving_mode = kwargs.pop('saving_mode', None) or PipelineSavingScheme.OVERWRITE_IN_PLACE
    skip_extended_batch_computations = kwargs.pop('skip_extended_batch_computations', True)
    fail_on_exception = kwargs.pop('fail_on_exception', True)
    debug_print = kwargs.pop('debug_print', False)

    try:
        curr_active_pipeline = batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, epoch_name_includelist=epoch_name_includelist,
                                        computation_functions_name_includelist=active_computation_functions_name_includelist,
                                        saving_mode=saving_mode, force_reload=force_reload, skip_extended_batch_computations=skip_extended_batch_computations, debug_print=debug_print, fail_on_exception=fail_on_exception, **kwargs)
        
    except Exception as e:
        ## can fail here before callback function is even called.
        exception_info = sys.exc_info()
        an_error = CapturedException(e, exception_info, None)
        new_print(f'exception occured: {an_error}')
        if fail_on_exception:
            raise e
        new_print(f'"{_line_sweep} END BATCH {_line_sweep}\n\n')
        
        return (SessionBatchProgress.FAILED, f"{an_error}", None) # return the Failed status and the exception that occured.
    # finally:
    #     print = _backup_print # restore default print, I think it's okay because it's only used in this context.

    if post_run_callback_fn is not None:
        if fail_on_exception:
            post_run_callback_fn_output = post_run_callback_fn(global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline)
        else:
            try:
                # handle exceptions in callback:
                post_run_callback_fn_output = post_run_callback_fn(global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline)
            except Exception as e:
                exception_info = sys.exc_info()
                an_error = CapturedException(e, exception_info, curr_active_pipeline)
                new_print(f'error occured in post_run_callback_fn: {an_error}. Suppressing.')
                # if fail_on_exception:
                    # raise e.exc
                post_run_callback_fn_output = None
    #         finally:
    #             print = _backup_print # restore default print, I think it's okay because it's only used in this context.

    # print = _backup_print # restore default print, I think it's okay because it's only used in this context.
    new_print(f'"{_line_sweep} END BATCH {_line_sweep}\n\n')
    return (SessionBatchProgress.COMPLETED, None, post_run_callback_fn_output) # return the success status and None to indicate that no error occured.




@function_attributes(short_name='main', tags=['batch', 'automated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def main(active_result_suffix:str='CHANGEME_TEST', included_session_contexts: Optional[List[IdentifyingContext]]=None, num_processes:int=4, should_force_reload_all:bool=False, should_perform_figure_generation_to_file:bool=False, debug_print=True):
    """ 
    
    should_perform_figure_generation_to_file: Whether to output figures
    
    
    from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

    

    """
    # active_result_suffix=f"{BATCH_DATE_TO_USE}_GL"
    print(f'Starting runBatch.main(active_result_suffix: "{active_result_suffix}", ...) ...')
    active_global_batch_result_filename:str = f'global_batch_result_{active_result_suffix}.pkl'
    global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    
    ## Build Pickle Path:
    finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve()
    if debug_print:
        print(f'finalized_loaded_global_batch_result_pickle_path: {finalized_loaded_global_batch_result_pickle_path}')
    # try to load an existing batch result:
    global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
                            skip_root_path_conversion=False, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch
        # # If we reach here than loading is good:
        # batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
        # good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
        
    # Save to file prior to running:
    # saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary

    # Run Batch Executions/Computations
    if num_processes is None:
        num_processes = 1
    use_multiprocessing: bool = (num_processes > 1)

    multiprocessing_kwargs = dict(use_multiprocessing=use_multiprocessing, num_processes=num_processes)


    if included_session_contexts is not None:
        print(f'len(included_session_contexts): {len(included_session_contexts)}')
    else:
        print(f'included_session_contexts is None so all session contexts will be included.')


    if should_force_reload_all:
        # Forced Reloading:
        print(f'forced reloading...')
        result_handler = BatchSessionCompletionHandler(force_reload_all=True,
                                                        session_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=SavingOptions.ALWAYS),
                                                        global_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=SavingOptions.ALWAYS),
                                                        should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE, force_global_recompute=True,
                                                        **multiprocessing_kwargs)

    else:
        # No Reloading
        result_handler = BatchSessionCompletionHandler(force_reload_all=False,
                                                        session_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=SavingOptions.NEVER),
                                                        global_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=SavingOptions.IF_CHANGED),
                                                        should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, saving_mode=PipelineSavingScheme.SKIP_SAVING, force_global_recompute=False,
                                                        **multiprocessing_kwargs)




    ## Execute with the custom arguments.
    active_computation_functions_name_includelist=['_perform_baseline_placefield_computation',
                                            # '_perform_time_dependent_placefield_computation',
                                            '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            '_perform_pf_find_ratemap_peaks_computation',
                                            # '_perform_time_dependent_pf_sequential_surprise_computation'
                                            '_perform_two_step_position_decoding_computation',
                                            # '_perform_recursive_latent_placefield_decoding'
                                        ]
    
    global_batch_run.execute_all(force_reload=result_handler.force_reload_all, saving_mode=result_handler.saving_mode, skip_extended_batch_computations=True, post_run_callback_fn=result_handler.on_complete_success_execution_session,
                             fail_on_exception=False, included_session_contexts=included_session_contexts,
                                                                                        **{'computation_functions_name_includelist': active_computation_functions_name_includelist,
                                                                                            'active_session_computation_configs': None,
                                                                                            'allow_processing_previously_completed': True}, **multiprocessing_kwargs) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

    # Save to pickle:
    saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary

    hdf5_file_path = global_data_root_parent_path.joinpath(f'global_batch_output_{active_result_suffix}.h5').resolve()
    try:
        global_batch_run.to_hdf(hdf5_file_path,'/')
    except Exception as e:
        print(f'encountered error {e} saving HDF5 to {hdf5_file_path}. Skipping.')
        hdf5_file_path

    ## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
    # Somewhere in there there are `InstantaneousSpikeRateGroupsComputation` results to extract
    across_sessions_instantaneous_fr_dict = {} # InstantaneousSpikeRateGroupsComputation

    # good_session_batch_outputs = global_batch_run.session_batch_outputs

    sessions_with_results = [a_ctxt for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None]
    good_session_batch_outputs = {a_ctxt:a_result for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

    for a_ctxt, a_result in good_session_batch_outputs.items():
        if a_result is not None:
            # a_good_result = a_result.__dict__.get('across_sessions_batch_results', {}).get('inst_fr_comps', None)
            a_good_result = a_result.across_session_results.get('inst_fr_comps', None)
            if a_good_result is not None:
                across_sessions_instantaneous_fr_dict[a_ctxt] = a_good_result
                # print(a_result['across_sessions_batch_results']['inst_fr_comps'])
                
    num_sessions = len(across_sessions_instantaneous_fr_dict)
    print(f'num_sessions: {num_sessions}')

    # When done, `result_handler.across_sessions_instantaneous_fr_dict` is now equivalent to what it would have been before. It can be saved using the normal `.save_across_sessions_data(...)`

    ## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
    inst_fr_output_filename = f'across_session_result_long_short_inst_firing_rate_{active_result_suffix}.pkl'

    AcrossSessionsResults.save_across_sessions_data(across_sessions_instantaneous_fr_dict=across_sessions_instantaneous_fr_dict, global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
    return global_batch_run, result_handler, across_sessions_instantaneous_fr_dict, (finalized_loaded_global_batch_result_pickle_path, inst_fr_output_filename)











if __name__ == "__main__":
    main()




def _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline, debug_print=False):
    """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.

    Usage:
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import _update_pipeline_missing_preprocessing_parameters
        was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
        was_updated
    """
    def _subfn_update_session_missing_preprocessing_parameters(sess):
        """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to a single session. Called only by `_update_pipeline_missing_preprocessing_parameters` """
        preprocessing_parameters = getattr(sess.config, 'preprocessing_parameters', None)
        if preprocessing_parameters is None:
            print(f'No existing preprocessing parameters! Assigning them!')
            default_lap_estimation_parameters = DynamicContainer(N=20, should_backup_extant_laps_obj=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`
            default_PBE_estimation_parameters = DynamicContainer(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.300) # NewPaper's Parameters
            default_replay_estimation_parameters = DynamicContainer(require_intersecting_epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3)

            sess.config.preprocessing_parameters = DynamicContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
                    'laps': default_lap_estimation_parameters,
                    'PBEs': default_PBE_estimation_parameters,
                    'replays': default_replay_estimation_parameters
                }))
            return True
        else:
            if debug_print:
                print(f'preprocessing parameters exist.')
            return False

    # BEGIN MAIN FUNCTION BODY
    was_updated = False
    was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.sess)

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]:
        was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.filtered_sessions[an_epoch_name])

    if was_updated:
        print(f'config was updated. Saving pipeline.')
        curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE)
    return was_updated
