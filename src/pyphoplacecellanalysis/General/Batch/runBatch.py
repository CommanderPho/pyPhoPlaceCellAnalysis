import sys
import os
import pathlib
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from copy import deepcopy

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.Filesystem.path_helpers import convert_filelist_to_new_parent

# pyPhoPlaceCellAnalysis:

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
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations, InstantaneousSpikeRateGroupsComputation # for `BatchSessionCompletionHandler`



def get_file_str_if_file_exists(v:Path)->str:
    """ returns the string representation of the resolved file if it exists, or the empty string if not """
    return (str(v.resolve()) if v.exists() else '')
    
@define(slots=False)
class BatchRun:
    """Docstring for BatchRun."""
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
                posix_backup = pathlib.PosixPath # backup the PosixPath definition
                try:
                    pathlib.PosixPath = pathlib.PurePosixPath
                    global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print) # Fails this time if it still throws an error
                finally:
                    pathlib.PosixPath = posix_backup # restore the backup posix path definition
                    
            except (FileNotFoundError, TypeError):
                # loading failed
                print(f'Failure loading {finalized_loaded_global_batch_result_pickle_path}.')
                global_batch_run = None
                
            return global_batch_run

        ##

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
        out_df = BatchResultAccessor.init_from_BatchRun(self, expand_context=expand_context, good_only=good_only)
        out_df = out_df.batch_results.build_all_columns() # this uses the same accessor.
        return out_df


    # Main functionality _________________________________________________________________________________________________ #
    def execute_session(self, session_context, post_run_callback_fn=None, **kwargs):
        """ calls `run_specific_batch(...)` to actually execute the session's run. """
        curr_session_status = self.session_batch_status[session_context]
        enable_calling_completion_handler_for_previously_completed: bool = kwargs.get('allow_processing_previously_completed', False)
        print(f'enable_calling_completion_handler_for_previously_completed: {enable_calling_completion_handler_for_previously_completed}')
        if (curr_session_status != SessionBatchProgress.COMPLETED) or enable_calling_completion_handler_for_previously_completed:
                curr_session_basedir = self.session_batch_basedirs[session_context]
                self.session_batch_status[session_context], self.session_batch_errors[session_context], self.session_batch_outputs[session_context] = run_specific_batch(self, session_context, curr_session_basedir, post_run_callback_fn=post_run_callback_fn, **kwargs)
        else:
            print(f'session {session_context} already completed.')

    def execute_all(self, **kwargs):
        for curr_session_context, curr_session_status in self.session_batch_status.items():
            # with neptune.init_run() as run: #TODO 2023-06-09 11:37: - [ ] refactor neptune
            #     run[f"session/{curr_session_context}"] = curr_session_status
            self.execute_session(curr_session_context, **kwargs) # evaluate a single session


    # Updating ___________________________________________________________________________________________________________ #
    def change_global_root_path(self, global_data_root_parent_path):
        """ Changes the self.global_data_root_parent_path for this computer and converts all of the `session_batch_basedirs` paths."""
        if isinstance(global_data_root_parent_path, str):
            global_data_root_parent_path = Path(global_data_root_parent_path)
            
        assert global_data_root_parent_path.exists(), f"the path provide should be the one for the system (and it should exist)"
        if self.global_data_root_parent_path != global_data_root_parent_path:
            print(f'switching data dir path from {str(self.global_data_root_parent_path)} to {str(global_data_root_parent_path)}')
            self.global_data_root_parent_path = global_data_root_parent_path
            # Somehow loses the capitalization for 'KDIBA'
            self.session_batch_basedirs = {ctx:global_data_root_parent_path.joinpath(*ctx.as_tuple()).resolve() for ctx in self.session_contexts} # ctx.format_name, ctx.animal, ctx.exper_name
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


    ## Add detected laps/replays to the batch_progress_df:
    @classmethod
    def build_batch_lap_replay_counts_df(cls, global_batch_run):
        """ returns lap_replay_counts_df """
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
    def post_load_find_usable_sessions(cls, batch_progress_df, min_required_replays_or_laps=5):
        """ updates batch_progress_df['is_ready'] and returns only the good frames. """
        has_no_errors = np.array([(an_err_v is None) for an_err_v in batch_progress_df['errors'].to_numpy()])
        has_required_laps_and_replays = np.all((batch_progress_df[['n_long_laps','n_long_replays','n_short_laps','n_short_replays']].to_numpy() >= min_required_replays_or_laps), axis=1)
        ## Adds 'is_ready' to the dataframe to indicate that all required properties are intact and that it's ready to process further:
        batch_progress_df['is_ready'] = np.logical_and(has_no_errors, has_required_laps_and_replays) # Add 'is_ready' column
        good_batch_progress_df = deepcopy(batch_progress_df)
        good_batch_progress_df = good_batch_progress_df[good_batch_progress_df['is_ready']]
        return good_batch_progress_df
        
    
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

    @staticmethod
    def find_global_root_path(batch_progress_df: pd.DataFrame) -> str:
        """ extracts the common prefix from the 'basedirs' column of the df and returns it. """
        paths = batch_progress_df['basedirs'].apply(lambda x: str(x)).to_list()
        common_prefix = os.path.commonprefix(paths) # '/nfs/turbo/umms-kdiba/Data/KDIBA/'
        return common_prefix

    @classmethod
    def rebuild_basedirs(cls, batch_progress_df, global_data_root_parent_path):
        """ replaces basedirs with ones that have been rebuilt from the local `global_data_root_parent_path` and hopefully point to extant paths. 
        
        adds: ['locally_folder_exists', 'locally_is_ready']
        updates: ['basedirs']
        
        Usage:
        
            updated_batch_progress_df = rebuild_basedirs(batch_progress_df, global_data_root_parent_path)
            updated_batch_progress_df

        """
        _context_column_names = ['format_name', 'animal', 'exper_name', 'session_name']

        assert global_data_root_parent_path.exists()
        
        ## Context-based method (loses capitalization):
        session_batch_basedirs = [global_data_root_parent_path.joinpath(*ctx_tuple).resolve() for ctx_tuple in batch_progress_df[_context_column_names].itertuples(index=False)]
        
        ## Path-based method using `convert_filelist_to_new_parent(...)`:
        source_parent_path = cls.find_global_root_path(batch_progress_df) # Path(r'/media/MAX/cloud/turbo/Data')
        # dest_parent_path = Path(r'/media/MAX/Data')
        

        # # Build the destination filelist from the source_filelist and the two paths:
        filelist_dest = convert_filelist_to_new_parent(filelist_source, original_parent_path=source_parent_path, dest_parent_path=global_data_root_parent_path)
        filelist_dest

        session_basedir_exists_locally = [a_basedir.resolve().exists() for a_basedir in session_batch_basedirs]

        updated_batch_progress_df = deepcopy(batch_progress_df)
        updated_batch_progress_df['basedirs'] = session_batch_basedirs
        updated_batch_progress_df['locally_folder_exists'] = session_basedir_exists_locally
        updated_batch_progress_df['locally_is_ready'] = np.logical_and(updated_batch_progress_df.is_ready, session_basedir_exists_locally)
        return updated_batch_progress_df


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
    # basedir = local_session_paths_list[1] # NOT 3
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
                                            # '_perform_time_dependent_placefield_computation',
                                            # '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            # '_perform_pf_find_ratemap_peaks_computation',
                                            # '_perform_time_dependent_pf_sequential_surprise_computation'
                                            # '_perform_two_step_position_decoding_computation',
                                            # '_perform_recursive_latent_placefield_decoding'
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
                
    return (SessionBatchProgress.COMPLETED, None, post_run_callback_fn_output) # return the success status and None to indicate that no error occured.


@function_attributes(short_name='main', tags=['batch', 'automated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def main(active_global_batch_result_filename='global_batch_result.pkl', debug_print=True):
    """ 
    from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

    """
    global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data')])
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    
    ## TODO: load the batch result initially:

    ## Build Pickle Path:
    finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve()
    if debug_print:
        print(f'finalized_loaded_global_batch_result_pickle_path: {finalized_loaded_global_batch_result_pickle_path}')
    # try to load an existing batch result:
    try:
        global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print)
    except (FileNotFoundError, TypeError):
        # loading failed
        print(f'Failure loading {finalized_loaded_global_batch_result_pickle_path}.')
        global_batch_run = None

    # global_batch_result = loadData('global_batch_result.pkl')
    global_batch_run = run_diba_batch(global_data_root_parent_path, execute_all=False, extant_batch_run=global_batch_run, debug_print=True)
    print(f'global_batch_result: {global_batch_run}')
    # Save to file:
    saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary


# ==================================================================================================================== #
# Exporters                                                                                                            #
# ==================================================================================================================== #
def dataframe_functions_test():
    """ 2023-06-13 - Tests loading saved .h5 `global_batch_result` Dataframe. And updating it for the local platform.

    #TODO 2023-06-13 18:09: - [ ] Finish this implementation up and make decision deciding how to use it
        
    """
    global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data')])
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

@pd.api.extensions.register_dataframe_accessor("batch_results")
class BatchResultAccessor():
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
        out_df = pd.concat((out_df, batchrun_obj.build_batch_lap_replay_counts_df(batchrun_obj)), axis=1) # don't need multiple concatenation operations probably
        
        ## Add is_ready
        batchrun_obj.post_load_find_usable_sessions(out_df, min_required_replays_or_laps=5)
        
        if good_only:
            # Get only the good (is_ready) sessions
            out_df = out_df[out_df['is_ready']]
            
        return out_df


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
        



        
@define(slots=False, repr=False)
class BatchSessionCompletionHandler:
    """ handles completion of a single session's batch processing. 

    Allows accumulating results across sessions and runs.

    
    Usage:
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchSessionCompletionHandler
        
    """
    force_reload_all: bool = field(default=False)
    saving_mode: PipelineSavingScheme = field(default=PipelineSavingScheme.SKIP_SAVING)
    force_global_recompute: bool = field(default=False)
    
    should_perform_figure_generation_to_file: bool = field(default=True) # controls whether figures are generated to file
    extended_computations_include_includelist: list = field(default=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis', 'long_short_decoding_analyses', 'long_short_post_decoding']) # do only specifiedl

    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation

    def post_compute_validate(self, curr_active_pipeline):
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
        LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate()
        # 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.
        was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
        print(f'were pipeline preprocessing parameters missing and updated?: {was_updated}')

        ## BUG 2023-05-25 - Found ERROR for a loaded pipeline where for some reason the filtered_contexts[long_epoch_name]'s actual context was the same as the short maze ('...maze2'). Unsure how this happened.
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # assert long_epoch_context.filter_name == long_epoch_name, f"long_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}"
        # fix it if broken
        long_epoch_context.filter_name = long_epoch_name


    def try_complete_figure_generation_to_file(self, curr_active_pipeline):
        try:
            ## To file only:
            with matplotlib_file_only():
                # Perform non-interactive Matplotlib operations with 'AGG' backend
                # neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None)
                main_complete_figure_generations(curr_active_pipeline, save_figures_only=True, save_figure=True)
                
            # IF thst's done, clear all the plots:
            from matplotlib import pyplot as plt
            plt.close('all') # this takes care of the matplotlib-backed figures.
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
        print(f'short_laps.n_epochs: {short_laps.n_epochs}, long_laps.n_epochs: {long_laps.n_epochs}')
        print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        self.post_compute_validate(curr_active_pipeline)
        
        ## Save the pipeline since that's disabled by default now:
        try:
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')

        ## GLOBAL FUNCTION:
        # FIXME: doesn't seem like we should always use `force_recompute=True`
        try:
            # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
            
            newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=self.extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=self.force_global_recompute, debug_print=False)
            #TODO 2023-07-11 19:20: - [ ] We want to save the global results if they are computed, but we don't want them to be needlessly written to disk even when they aren't changed.

            if (len(newly_computed_values) > 0):
                print(f'newly_computed_values: {newly_computed_values}. Saving global results...')
                if (self.saving_mode.value == 'skip_saving'):
                    print(f'WARNING: supposed to skip_saving because of self.saving_mode: {self.saving_mode} but supposedly has new global results! Figure out if these are actually new.')
                
                try:
                    # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                    # Try to write out the global computation function results:
                    curr_active_pipeline.save_global_computation_results()
                except Exception as e:
                    print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                    print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
            else:
                print(f'no changes in global results.')
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            print(f'ERROR SAVING GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: {curr_session_context}. error: {e}')
            

        # ### Programmatic Figure Outputs:
        if self.should_perform_figure_generation_to_file:
            self.try_complete_figure_generation_to_file(curr_active_pipeline)
        else:
            print(f'skipping figure generation because should_perform_figure_generation_to_file == False')

        ### Do specific computations:
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

        # Note that in general LxC and SxC might have differing numbers of cells.
        across_session_inst_fr_computation.Fig2_Laps_FR = [(v.mean(), v.std(), v) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]))]


        across_session_inst_fr_computation.Fig2_Replay_FR = [(v.mean(), v.std(), v) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions)]),
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


if __name__ == "__main__":
    main()