import sys
import os
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
import numpy as np
import pandas as pd
from copy import deepcopy
import multiprocessing


## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path, set_posix_windows, convert_filelist_to_new_parent, find_matching_parent_path
from pyphocorehelpers.function_helpers import function_attributes

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.core.epoch import Epoch
from neuropy.utils.matplotlib_helpers import matplotlib_file_only
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths

from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session, batch_extended_computations, SessionBatchProgress, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

from attrs import define, field, Factory

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import _update_pipeline_missing_preprocessing_parameters
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations, InstantaneousSpikeRateGroupsComputation, SingleBarResult # for `BatchSessionCompletionHandler`
from pyphoplacecellanalysis.General.Model.user_annotations import UserAnnotationsManager


known_global_data_root_parent_paths = [Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/nfs/turbo/umms-kdiba/Data')]

def get_file_str_if_file_exists(v:Path)->str:
    """ returns the string representation of the resolved file if it exists, or the empty string if not """
    return (str(v.resolve()) if v.exists() else '')
    
@define(slots=False)
class BatchRun:
    """An object that manages a Batch of runs for many different session folders.
    
    
    """
    global_data_root_parent_path: Path
    session_batch_status: dict = Factory(dict)
    session_batch_basedirs: dict = Factory(dict)
    session_batch_errors: dict = Factory(dict)
    session_batch_outputs: dict = Factory(dict) # optional selected outputs that can hold information from the computation
    enable_saving_to_disk: bool = False
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
    def execute_session(self, session_context, **kwargs):
        """ calls `run_specific_batch(...)` to actually execute the session's run. """
        curr_session_status = self.session_batch_status[session_context]
        curr_session_basedir = self.session_batch_basedirs[session_context]
        self.session_batch_status[session_context], self.session_batch_errors[session_context], self.session_batch_outputs[session_context] = run_specific_batch(self, session_context, curr_session_basedir, **kwargs)


    # TODO: NOTE: that `execute_session` is not called in mutliprocessing mode!
    def execute_all(self, use_multiprocessing=True, num_processes=None, included_session_contexts: Optional[List[IdentifyingContext]]=None, session_inclusion_filter:Optional[Callable]=None, **kwargs):
        """ ChatGPT's multiprocessing edition. """
        if included_session_contexts is not None:
            # use `included_session_contexts` list over the filter function.
            assert session_inclusion_filter is None, f"You cannot provide both a `session_inclusion_filter` and a `included_session_contexts` list. Include one or the other."
            # ready to go
        else:
            if session_inclusion_filter is None:
                session_inclusion_filter = (lambda curr_session_context, curr_session_status: (curr_session_status != SessionBatchProgress.COMPLETED) or kwargs.get('allow_processing_previously_completed', False))
            else:
                # `session_inclusion_filter` was provided, make sure there is no list.
                assert included_session_contexts is None, f"You cannot provide both a `session_inclusion_filter` and a `included_session_contexts` list. Include one or the other."
            # either way now build the contexts list:
            included_session_contexts: List[IdentifyingContext] = [curr_session_context for curr_session_context, curr_session_status in self.session_batch_status.items() if session_inclusion_filter(curr_session_context, curr_session_status)]

        # Now `included_session_contexts` list should be good either way:
        assert included_session_contexts is not None
        assert isinstance(included_session_contexts, list)
        # filter for inclusion here instead of in the loop:        
        # a list of included_session_contexts:
        # included_session_contexts: List[IdentifyingContext] = [curr_session_context for curr_session_context, curr_session_status in self.session_batch_status.items() if session_inclusion_filter(curr_session_context, curr_session_status)]

        if use_multiprocessing:
            if num_processes is None:
                num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores

            pool = multiprocessing.Pool(processes=num_processes)
            results = {} # dict form
            
            for curr_session_context in included_session_contexts:
                curr_session_basedir = self.session_batch_basedirs[curr_session_context]
                result = pool.apply_async(run_specific_batch, (self, curr_session_context, curr_session_basedir), kwargs) # it can actually take a callback too.
                results[curr_session_context] = result

            pool.close()
            pool.join()

            for session_context, result in results.items():
                status, error, output = result.get()
                self.session_batch_status[session_context] = status
                self.session_batch_errors[session_context] = error
                self.session_batch_outputs[session_context] = output
        else:
            # No multiprocessing, fall back to the normal way.
            for curr_session_context in included_session_contexts:
                self.execute_session(curr_session_context, **kwargs) # evaluate a single session



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
        """Create a BatchRun object from a dataframe representation."""
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
            assert len(batchrun_obj.session_contexts) > 0 # must have at least one element
            first_context = batchrun_obj.session_contexts[0]
            context_column_names = list(first_context.keys()) # ['format_name', 'animal', 'exper_name', 'session_name']
            
            # TODO: batchrun_obj._context_column_names
            
            all_sess_context_tuples = [a_ctx.as_tuple() for a_ctx in batchrun_obj.session_contexts] #[('kdiba', 'gor01', 'one', '2006-6-07_11-26-53'), ('kdiba', 'gor01', 'one', '2006-6-08_14-26-15'), ('kdiba', 'gor01', 'one', '2006-6-09_1-22-43'), ...]
            expanded_context_df = pd.DataFrame.from_records(all_sess_context_tuples, columns=context_column_names)
            out_df = pd.concat((expanded_context_df, non_expanded_context_df), axis=1)
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
                (long_laps, long_replays), (short_laps, short_replays) = list(output_v.values())[:2] # only get the first four outputs
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
        self._build_minimal_session_identifiers_list()
        ## TODO: append to the non-dataframe object?
        user_is_replay_good_annotations = UserAnnotationsManager.get_user_annotations()
        self._obj['has_user_replay_annotations'] = [(a_row_context.adding_context_if_missing(display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections') in user_is_replay_good_annotations) for a_row_context in self._obj['context']]

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
        """Build a list of the output files for the good sessions:
        Adds Column: ['context_minimal_name']
        
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


@define(slots=False)
class BatchComputationProcessOptions:
	should_load: bool # should try to load from existing results from disk at all
		# never
		# always (fail if loading unsuccessful)
		# always (warning but continue if unsuccessful)
	should_compute: bool # should try to run computations (which will just verify that loaded computations are good if that option is true)
		# never
		# if needed (required results are missing)
		# always
	should_save: bool # should consider save at all
		# never
		# if changed
		# always


@define(slots=False, repr=False)
class BatchSessionCompletionHandler:
    """ handles completion of a single session's batch processing. 

    Allows accumulating results across sessions and runs.

    
    Usage:
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchSessionCompletionHandler
        
    """
    # General:
    debug_print: bool = field(default=False)

    force_reload_all: bool = field(default=False)
    saving_mode: PipelineSavingScheme = field(default=PipelineSavingScheme.SKIP_SAVING)
    
    # Computations
    override_session_computation_results_pickle_filename: Optional[str] = field(default=None) # 'output/loadedSessPickle.pkl'

    session_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=True))

    global_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=True))
    extended_computations_include_includelist: list = field(default=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis', 'long_short_decoding_analyses', 'long_short_post_decoding']) # do only specifiedl
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
            print(f'main_complete_figure_generations failed with exception: {e}')
            # raise e
            return False


    def on_complete_success_execution_session(self, active_batch_run, curr_session_context, curr_session_basedir, curr_active_pipeline):
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
        # short_laps.n_epochs: 40, long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = self.post_compute_validate(curr_active_pipeline)
        except Exception as e:
            print(f'self.post_compute_validate(...) failed with exception: {e}')
            raise 

        ## Save the pipeline since that's disabled by default now:
        try:
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.override_session_computation_results_pickle_filename) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')

        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True


        if self.global_computations_options.should_load:
            if not self.force_global_recompute: # not just force_reload, needs to recompute whenever the computation fails.
                try:
                    curr_active_pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=self.override_global_computation_results_pickle_path)
                except Exception as e:
                    print(f'cannot load global results: {e}')
                

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
                    if self.global_computations_options.should_save:
                        try:
                            # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                            # Try to write out the global computation function results:
                            curr_active_pipeline.save_global_computation_results()
                        except Exception as e:
                            print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                            print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
                    else:
                        print(f'\n\n!!WARNING!!: self.global_computations_options.should_save == False, so the global results are unsaved!')
                else:
                    print(f'no changes in global results.')
            except Exception as e:
                ## TODO: catch/log saving error and indicate that it isn't saved.
                print(f'ERROR SAVING GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: {curr_session_context}. error: {e}')
                

        # ### Programmatic Figure Outputs:
        if self.should_perform_figure_generation_to_file:
            self.try_complete_figure_generation_to_file(curr_active_pipeline, enable_default_neptune_plots=self.should_generate_all_plots)
        else:
            print(f'skipping figure generation because should_perform_figure_generation_to_file == False')

        ### Do specific computations for aggregate outputs:
        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            self.across_sessions_instantaneous_fr_dict[curr_session_context] = _out_inst_fr_comps # instantaneous firing rates for this session
            # LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            # LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
            print(f'\t\t done (success). Now have {len(self.across_sessions_instantaneous_fr_dict)} entries in self.across_sessions_instantaneous_fr_dict!')
        except Exception as e:
            print(f"ERROR: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            _out_inst_fr_comps = None
            
        return {long_epoch_name:(long_laps, long_replays), short_epoch_name:(short_laps, short_replays),
                'outputs': {'local': curr_active_pipeline.pickle_path,
                            'global': curr_active_pipeline.global_computation_results_pickle_path},
                'across_sessions_batch_results': {'inst_fr_comps': _out_inst_fr_comps}
            }
        

    # Across Sessions Helpers
    def save_across_sessions_data(self, global_data_root_parent_path:Path, inst_fr_output_filename:str='across_session_result_long_short_inst_firing_rate.pkl'):
        """ Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation) 
        
        """
        global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
        # Save the all sessions instantaneous firing rate dict to the path:
        saveData(global_batch_result_inst_fr_file_path, self.across_sessions_instantaneous_fr_dict)

    @classmethod
    def load_across_sessions_data(cls, global_data_root_parent_path:Path, inst_fr_output_filename:str='across_session_result_long_short_inst_firing_rate.pkl'):
        """ Load the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation) 

            To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).
        
        Usage:

            ## Load the saved across-session results:
            inst_fr_output_filename = 'long_short_inst_firing_rate_result_handlers_2023-07-12.pkl'
            across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = BatchSessionCompletionHandler.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
            # across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
            num_sessions = len(across_sessions_instantaneous_fr_dict)
            print(f'num_sessions: {num_sessions}')

            ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
            global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

            # To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

            ## Display the aggregate across sessions:
            _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
            # _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
            # Cannot call `.compute(curr_active_pipeline=curr_active_pipeline)` like we normally would because everything is manually set.
            _out_fig_2.computation_result = across_session_inst_fr_computation
            _out_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
            # Set callback, the only self-specific property
            _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
            _out_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)")

        """
        global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
        across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')
        across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = list(across_sessions_instantaneous_fr_dict.values())
        ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        # _out.cell_agg_inst_fr_list = cell_agg_firing_rates_list # .shape (n_cells,)
        across_session_inst_fr_computation = InstantaneousSpikeRateGroupsComputation()
        across_session_inst_fr_computation.active_identifying_session_ctx = global_multi_session_context 

        all_contexts_list: List[IdentifyingContext] = list(across_sessions_instantaneous_fr_dict.keys())
        assert len(all_contexts_list) > 0 # must have at least one element
        first_context = all_contexts_list[0]
        context_column_names = list(first_context.keys()) # ['format_name', 'animal', 'exper_name', 'session_name']
        expanded_context_df = pd.DataFrame.from_records([a_ctx.as_tuple() for a_ctx in all_contexts_list], columns=context_column_names)
        context_minimal_names = expanded_context_df.batch_results._build_minimal_session_identifiers_list()
        # print(f"context_minimal_names: {context_minimal_names}")
        assert len(context_minimal_names) == len(all_contexts_list)

        context_minimal_names_map = dict(zip(all_contexts_list, context_minimal_names))
        def _build_session_dep_aclu_identifier(session_context: IdentifyingContext, session_relative_aclus: np.ndarray):
            """ kdiba_pin01_one_fet11-01_12-58-54_{aclu} 
                with `context_minimal_names_map` - get tiny names like: a0s1, a0s2
            Captures: `context_minimal_names_map`
            """
            # return [f"{session_context}_{aclu}" for aclu in session_relative_aclus] # need very short version
            return [f"{context_minimal_names_map[session_context]}_{aclu}" for aclu in session_relative_aclus] # need very short version


        LxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.LxC_aclus) for k, v in across_sessions_instantaneous_fr_dict.items()])
        SxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.SxC_aclus) for k, v in across_sessions_instantaneous_fr_dict.items()])

        across_session_inst_fr_computation.LxC_aclus = LxC_aclus
        across_session_inst_fr_computation.SxC_aclus = SxC_aclus

        # i = 0
        # across_sessions_instantaneous_frs_list[i].LxC_aclus
        # LxC_aclus = across_sessions_instantaneous_frs_list[0].LxC_ThetaDeltaPlus.LxC_aclus
        # SxC_aclus = across_sessions_instantaneous_frs_list[0].LxC_ThetaDeltaPlus.SxC_aclus

        # Note that in general LxC and SxC might have differing numbers of cells.
        across_session_inst_fr_computation.Fig2_Laps_FR = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]))]


        across_session_inst_fr_computation.Fig2_Replay_FR = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]))]

        

        # ## Display the aggregate across sessions:
        # _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
        # # _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
        # # Cannot call `.compute(curr_active_pipeline=curr_active_pipeline)` like we normally would because everything is manually set.
        # _out_fig_2.computation_result = _out
        # _out_fig_2.active_identifying_session_ctx = _out.active_identifying_session_ctx
        # # Set callback, the only self-specific property
        # _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)

        return across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list




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
                        active_batch_run.session_batch_status[curr_session_context], active_batch_run.session_batch_errors[curr_session_context], active_batch_run.session_batch_outputs[curr_session_context] = run_specific_batch(active_batch_run, curr_session_context, curr_session_basedir, post_run_callback_fn=post_run_callback_fn)

                else:
                    print(f'EXTANT SESSION! curr_session_context: {curr_session_context} curr_session_status: {curr_session_status}, curr_session_errors: {active_batch_run.session_batch_errors.get(curr_session_context, None)}')
                    ## TODO 2023-04-19: shouldn't computation happen here too if needed?


    ## end for
    return active_batch_run


@function_attributes(short_name='run_specific_batch', tags=['batch', 'automated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def run_specific_batch(active_batch_run: BatchRun, curr_session_context: IdentifyingContext, curr_session_basedir: Path, force_reload=True, post_run_callback_fn=None, **kwargs):
    """ For a specific session (identified by the session context) - calls batch_load_session(...) to get the curr_active_pipeline.
            - Then calls `post_run_callback_fn(...)
    
    """
    ## Extract the default session loading vars from the session context:
    basedir = curr_session_basedir
    print(f'basedir: {str(basedir)}')
    active_data_mode_name = curr_session_context.format_name
    print(f'active_data_mode_name: {active_data_mode_name}')
    # post_run_callback_fn = kwargs.pop('post_run_callback_fn', None)
    post_run_callback_fn_output = None
    
    # ==================================================================================================================== #
    # Load Pipeline                                                                                                        #
    # ==================================================================================================================== #
    # epoch_name_includelist = ['maze']
    epoch_name_includelist = kwargs.pop('epoch_name_includelist', None)
    active_computation_functions_name_includelist = kwargs.pop('computation_functions_name_includelist', None) or ['_perform_baseline_placefield_computation',
                                            # '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            # '_perform_pf_find_ratemap_peaks_computation', '_perform_time_dependent_pf_sequential_surprise_computation' '_perform_two_step_position_decoding_computation', '_perform_recursive_latent_placefield_decoding'
                                        ]
    
    saving_mode = kwargs.pop('saving_mode', None) or PipelineSavingScheme.OVERWRITE_IN_PLACE
    skip_extended_batch_computations = kwargs.pop('skip_extended_batch_computations', True)
    fail_on_exception = kwargs.pop('fail_on_exception', True)
    debug_print = kwargs.pop('debug_print', False)

    try:
        curr_active_pipeline = batch_load_session(active_batch_run.global_data_root_parent_path, active_data_mode_name, basedir, epoch_name_includelist=epoch_name_includelist,
                                        computation_functions_name_includelist=active_computation_functions_name_includelist,
                                        saving_mode=saving_mode, force_reload=force_reload, skip_extended_batch_computations=skip_extended_batch_computations, debug_print=debug_print, fail_on_exception=fail_on_exception, **kwargs)
        
    except Exception as e:
        return (SessionBatchProgress.FAILED, e, None) # return the Failed status and the exception that occured.

    if post_run_callback_fn is not None:
        if fail_on_exception:
            post_run_callback_fn_output = post_run_callback_fn(active_batch_run, curr_session_context, curr_session_basedir, curr_active_pipeline)
        else:
            try:
                # handle exceptions in callback:
                post_run_callback_fn_output = post_run_callback_fn(active_batch_run, curr_session_context, curr_session_basedir, curr_active_pipeline)
            except Exception as e:
                print(f'error occured in post_run_callback_fn: {e}. Suppressing.')
                post_run_callback_fn_output = None
                
    return (SessionBatchProgress.COMPLETED, None, post_run_callback_fn_output) # return the success status and None to indicate that no error occured.




@function_attributes(short_name='main', tags=['batch', 'automated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def main(active_global_batch_result_filename='global_batch_result.pkl', debug_print=True):
    """ 
    from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

    """
    global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    
    ## Build Pickle Path:
    finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve()
    if debug_print:
        print(f'finalized_loaded_global_batch_result_pickle_path: {finalized_loaded_global_batch_result_pickle_path}')
    # try to load an existing batch result:
    try:
        global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
						skip_root_path_conversion=False, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch
        # If we reach here than loading is good:
        batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
        good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)

    except (FileNotFoundError, TypeError):
        # loading failed
        print(f'Failure loading {finalized_loaded_global_batch_result_pickle_path}.')
        global_batch_run = None

    # global_batch_result = loadData('global_batch_result.pkl')
    if global_batch_run is None:
        print(f'global_batch_run is None (does not exist). It will be initialized by calling `run_diba_batch(...)`...')
        # Build `global_batch_run` pre-loading results (before execution)
        global_batch_run = run_diba_batch(global_data_root_parent_path, execute_all=False, extant_batch_run=global_batch_run, debug_print=True)
        
    print(f'global_batch_result: {global_batch_run}')
    # Save to file:
    saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary


    # Run Batch Executions/Computations
    ## I got it doing the bare-minimum loading and computations, so it should be ready to update the laps and constraint the placefields to those. Then we should be able to set up the replays at the same time.
    # finally, we then finish by computing.
    # force_reload = True
    force_reload = False
    result_handler = BatchSessionCompletionHandler(force_reload_all=force_reload, should_perform_figure_generation_to_file=False, saving_mode=PipelineSavingScheme.SKIP_SAVING, force_global_recompute=False)

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
    # active_computation_functions_name_includelist=['_perform_baseline_placefield_computation']
    global_batch_run.execute_all(force_reload=force_reload, saving_mode=PipelineSavingScheme.SKIP_SAVING, skip_extended_batch_computations=True, post_run_callback_fn=result_handler.on_complete_success_execution_session,
                                                                                            **{'computation_functions_name_includelist': active_computation_functions_name_includelist,
                                                                                                'active_session_computation_configs': None,
                                                                                                'allow_processing_previously_completed': True}) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

    # Save to file:
    saveData(global_batch_result_file_path, global_batch_run) # Update the global batch run dictionary

    ## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
    #TODO 2023-07-12 10:12: - [ ] New save way after we save out current result and reload
    result_handler.save_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename='across_session_result_long_short_inst_firing_rate.pkl')
    num_sessions = len(result_handler.across_sessions_instantaneous_fr_dict)
    print(f'num_sessions: {num_sessions}')


    # 4m 39.8s



# ==================================================================================================================== #
# Exporters                                                                                                            #
# ==================================================================================================================== #
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


# 2023-07-07


if __name__ == "__main__":
    main()