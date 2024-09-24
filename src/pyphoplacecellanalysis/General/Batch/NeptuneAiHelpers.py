from copy import deepcopy
import os
from typing import Dict, List, Tuple
from attrs import define, Factory, field
import neptune # for logging progress and results
from neptune.types import File
from neptune.utils import stringify_unsupported
from neptune.exceptions import NeptuneException
from datetime import datetime
import numpy as np
import pandas as pd
from io import StringIO
import io
import re
from contextlib import redirect_stdout, redirect_stderr

import pathlib
from pathlib import Path

from neuropy.utils.result_context import IdentifyingContext
from pyphocorehelpers.function_helpers import function_attributes

"""
if enable_neptune:
        project = neptune.init_project()
        project["general/global_batch_result_filename"] = active_global_batch_result_filename
        project["general/global_data_root_parent_path"] = global_data_root_parent_path.as_posix()

    ## Currently contains an explicit neptune dependency:
    with neptune.init_run() as run:
        if enable_neptune:
            run['parameters/perform_execute'] = perform_execute
            run['parameters/global_batch_result_file_path'] = global_batch_result_file_path.as_posix()
            # project["general/data_analysis"].upload("data_analysis.ipynb")
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images"

if enable_neptune:
            # Pre-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"
if enable_neptune:
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images" # update file progress post-load
            # Post-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"
        
if enable_neptune:
        ## POST Run
        project.stop()
"""


def set_environment_variables(neptune_kwargs=None, enable_neptune=True):
    """ sets the environment variables from the neptune_kwargs
     
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables
      
    """
    if enable_neptune:
        if neptune_kwargs is None:
            # set defaults:
            neptune_kwargs = {'project':"commander.pho/PhoDibaLongShort2023",
            'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}

        os.environ["NEPTUNE_API_TOKEN"] = neptune_kwargs['api_token']
        os.environ["NEPTUNE_PROJECT"] = neptune_kwargs['project']


        


def capture_print_output(func):
    """Captures the output of the provided function."""
    captured_output = io.StringIO()              # Create StringIO object
    
    with redirect_stdout(captured_output):
        func()                                       # Call the function to capture its output
        return captured_output.getvalue()            # Get the captured output as a string


def strip_ansi_escape_sequences(text: str) -> str:
    """Removes ANSI escape sequences from the provided text."""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def clean_quotes(text: str) -> str:
    """Cleans surrounding quotes from the keys and values."""
    return text.strip("'\"")


class AutoValueConvertingNeptuneRun(neptune.Run):
    
    def __setitem__(self, key, value):
        if isinstance(value, pd.DataFrame):
            csv_buffer = StringIO()
            value.to_csv(csv_buffer, index=False)
            super().__setitem__(key, File.from_stream(csv_buffer, extension="csv"))

        elif isinstance(value, (pathlib.PurePath, pathlib.Path)):
            super().__setitem__(key, value.as_posix())

        elif isinstance(value, (IdentifyingContext)):
            super().__setitem__(key, value.get_description())
            
        else:
            super().__setitem__(key, value)


    def get_parsed_structure(self) -> Dict:
        """ returns a dictionary parsed for structure
        Usage:
            parsed_structure = run.get_parsed_structure()
            parsed_structure

        """
        def _subfn_parse_structure(captured_output: str) -> dict:
            """Parses the captured printed structure into a nested dictionary."""
            lines = captured_output.splitlines()
            structure = {}
            stack = [structure]  # Use a stack to manage nested dictionaries
            last_indent = 0

            for line in lines:
                if not line.strip():  # Skip empty lines
                    continue

                # Find the current indent level (number of leading spaces)
                current_indent = len(line) - len(line.lstrip())

                # Extract key-value pair from the line
                if ':' in line:
                    key, value = map(str.strip, line.split(':', 1))
                    key = clean_quotes(key)
                    value = clean_quotes(value)
                else:
                    key, value = clean_quotes(line.strip()), {}

                # Adjust stack for indentation changes
                if current_indent > last_indent:
                    # Going deeper into a new nested structure
                    stack[-1][prev_key] = {}  # Set last key as a dictionary
                    stack.append(stack[-1][prev_key])  # Add new nested dict to stack
                elif current_indent < last_indent:
                    # Going back up in the structure, pop stack
                    stack = stack[:current_indent // 4 + 1]

                # Assign the key-value pair
                stack[-1][key] = value
                prev_key = key
                last_indent = current_indent

            return structure

        # Step 1: Capture the output from run.print_structure()
        output = capture_print_output(self.print_structure)

        # Step 2: Strip ANSI escape sequences from the captured output
        cleaned_output = strip_ansi_escape_sequences(output)

        # Step 2: Parse the captured output into a nested dictionary
        parsed_structure = _subfn_parse_structure(cleaned_output)

        # Step 3: Use the parsed structure
        return parsed_structure


    def get_log_contents(self) -> str:
        parsed_structure = self.get_parsed_structure()
        monitoring_log_key: str = ''
        for a_monitoring_log_key in list(parsed_structure['monitoring'].keys()):
            # a_monitoring_root_key: str = f"monitoring/{a_monitoring_log_key}"
            a_monitoring_subkeys = parsed_structure['monitoring'][a_monitoring_log_key]
            
            if ('stdout' in a_monitoring_subkeys) or ('stderr' in a_monitoring_subkeys):
                ## found matching candidate key
                monitoring_log_key = a_monitoring_log_key
                
            # self[f"{monitoring_root_key}/stdout"]            
            # 'stdout' in a_monitoring_log_key
            
        assert (monitoring_log_key != ''), f"monitoring_log_key was not found"
        
        # monitoring_log_key: str = list(parsed_structure['monitoring'].keys())[-1] # like 'be28f54f'
        
        # monitoring_root_key: str = "monitoring/be28f54f"
        monitoring_root_key: str = f"monitoring/{monitoring_log_key}"
        
        stdout_log_df: pd.DataFrame = self[f"{monitoring_root_key}/stdout"].fetch_values(include_timestamp=True) # ['value', 'timestamp']
        # stdout_log_df
        # <StringSeries field at "monitoring/be28f54f/stdout">
        stderr_log_df: pd.DataFrame = self[f"{monitoring_root_key}/stderr"].fetch_values(include_timestamp=True) # ['value', 'timestamp']
        # stderr_log_df

        ## OUTPUT: stdout_log_df, stderr_log_df

        # 'monitoring':
        #     '5f739afe':
        #         'hostname': String
        #         'pid': String
        #         'tid': String
        #     'be28f54f':
        #         'cpu': FloatSeries
        #         'hostname': String
        #         'memory': FloatSeries
        #         'pid': String
        #         'stderr': StringSeries
        #         'stdout': StringSeries
        #         'tid': String

        # stderr_log_df.to_dict(orient='record')

        merged_log_df: pd.DataFrame = pd.concat((stdout_log_df, stderr_log_df)).sort_values(by='timestamp', ascending=True).reset_index(drop=True)[['value', 'timestamp']]
        # Drop rows where 'value' contains only whitespaces or newlines
        merged_log_df = merged_log_df[~merged_log_df['value'].str.fullmatch(r'\s*')]
        # merged_log_df
        log_contents_str: str = "\n".join([f"{row['timestamp']}: {row['value']}" for _, row in merged_log_df.iterrows()])
        return log_contents_str, merged_log_df
    


@define()
class Neptuner(object):
    """An object to maintain state for neptune.ai outputs.
      
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import Neptuner
    neptuner = Neptuner()
    
    """
    project_name: str = field(default="commander.pho/PhoDibaLongShortUpdated")
    api_token: str = field(default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==")
    project: neptune.Project = field(init=False)
    run: AutoValueConvertingNeptuneRun = field(init=False)

    outputs = field(init=False)
    figures = field(init=False)

    def __attrs_post_init__(self):
        self.project = neptune.init_project(project=self.project_name, api_token=self.api_token)
        self.run = None
        
    def stop(self):
        print(f'Neptuner.stop() called!')
        if self.run is not None:
            self.run.stop()
        self.project.stop()
        print(f'Neptuner stoppped.')

    ## The destructor apparently isn't called reliably and sometimes it is called even when the object still exists?                
    # def __del__(self):
    #     # body of destructor
    #     self.stop()
    #     self.run = None
    #     self.project = None

    def run_with_pipeline(self, curr_active_pipeline):
        """ starts a new run with the provided pipeline. """
        assert self.run is None, f"run_new_pipeline(...) called while the Neptuner object already has an active self.run object!"
        if self.run is None:
            # self.run = neptune.init_run(project=self.project_name, api_token=self.api_token, source_files=[]) # see git_ref=GitRef(repository_path="/path/to/repo")
            self.run = AutoValueConvertingNeptuneRun(project=self.project_name, api_token=self.api_token, source_files=[]) # see git_ref=GitRef(repository_path="/path/to/repo")

            # Add the session_context properties to the run: {'format_name': 'kdiba', 'animal': 'vvp01', 'exper_name': 'two', 'session_name': '2006-4-09_16-40-54'}
            for k, v in curr_active_pipeline.get_session_context().to_dict().items():
                self.run[k] = v # add the properties to the run

            self.run["sys/tags"].add(list(curr_active_pipeline.get_session_context().as_tuple())) # adds the session tags ('kdiba', 'gor01', 'one', '2006-6-09_1-22-43')
            

            # session_descriptor_string: a string describing the context of the session like 'sess_kdiba_2006-6-07_11-26-53'
            self.run['session_descriptor_string'] = curr_active_pipeline.sess.get_description() # 'kdiba_vvp01_two_2006-4-09_16-40-54_sess'
            self.outputs = self.run['outputs']
            self.figures = self.outputs['figures']

    @classmethod
    def init_with_pipeline(cls, curr_active_pipeline):
        """ creates a new Neptuner object corresponding to a particular session and starts a run. """
        new = cls()
        new.run_with_pipeline(curr_active_pipeline=curr_active_pipeline)
        return new


    def upload_figures(self, curr_active_pipeline, should_use_fig_objects=False, debug_print=False):
        """ 
            should_use_fig_objects: bool (False):  if True, the raw matplotlib.Figure objects are uploaded. Otherwise, the previously saved files are uploaded.
        """
        assert not should_use_fig_objects, f"should_use_fig_objects = True is not currently working. Save the figures to file first and then use should_use_fig_objects = False"
        sess_context = curr_active_pipeline.get_session_context()
        succeeded_fig_paths = []
        failed_fig_paths = []

        for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items():
            ## Get list of figures output and register them to neptune:
            a_fig_context: IdentifyingContext = fig_metadata['context']
            a_fig_specific_context: IdentifyingContext = a_fig_context.subtracting(sess_context) # subtract off the common session-level context.
            # a_full_figure_path_key = f"session/{active_session_context_string}/figures/{str(a_fig_context)}"
            a_full_figure_path_key: str = a_fig_specific_context.get_description(separator='/', include_property_names=True, key_value_separator=':')
            if debug_print:
                # print(f'a_fig_specific_context: {a_fig_specific_context}, a_full_figure_path_key: {a_full_figure_path_key}\n\ta_fig_path: {a_fig_path}, a_fig_context: {a_fig_context}\n')
                print(f'Processing: {a_fig_specific_context}, full_key: {a_full_figure_path_key}')

            fig_objects = fig_metadata.get('fig', None)
            if (fig_objects is not None) and should_use_fig_objects:
                # upload the actual figure objects
                try:
                    # Try to extract and upload the actual figure objects
                    if not isinstance(fig_objects, (tuple, list)):
                        fig_objects = (fig_objects,) # wrap in a tuple or convert list to tuple
                    elif isinstance(fig_objects, list):
                        fig_objects = tuple(fig_objects) # convert the list to a tuple
                        
                    # now fig_objects better be a tuple
                    assert isinstance(fig_objects, tuple), f"{fig_objects} should be a tuple!"

                    if len(fig_objects) == 1:
                        # single element, use .upload(...)
                        a_fig_obj = fig_objects[0]
                        self.figures[a_full_figure_path_key].upload(a_fig_obj)
                    else:
                        # otherwise multi-figure, use append for a series list:
                        for a_fig_obj in fig_objects:
                            # also ,  step=epoch for custom index
                            self.figures[a_full_figure_path_key].append(a_fig_obj) #, name=f"{a_full_figure_path_key}", description=f"{a_full_figure_path_key}" append the whole series of related figures
                            
                    if debug_print:
                        print(f'\tSuccess: "{a_full_figure_path_key}": {len(fig_objects)} uploaded!')
                    succeeded_fig_paths.append(a_fig_path)
                    
                except Exception as e:
                    if debug_print:
                        print(f'\tSkipping "{a_full_figure_path_key}": fig_objects found but raised exception {e}.')
                    failed_fig_paths.append(a_fig_path)
                    # raise e		
            else:
                if should_use_fig_objects:
                    if debug_print:
                        print(f'\tSkipping "{a_full_figure_path_key}": No figure objects found')
                    failed_fig_paths.append(a_fig_path)
                else:
                    # this is the mode we expected -- upload as file
                    try:
                        self.figures[a_full_figure_path_key].upload(File(str(a_fig_path)))
                        succeeded_fig_paths.append(a_fig_path)
                        if debug_print:
                            print(f'\tSuccess: "{a_full_figure_path_key}": file at path {str(a_fig_path)} uploaded!')
                    except Exception as e:
                        raise e
                    
        # end for loop
        return succeeded_fig_paths, failed_fig_paths

    ## FETCH results:
    def get_runs_table(self) -> pd.DataFrame:
        """ 
        Usage:
            neptune_kwargs = {'project':"commander.pho/PhoDibaLongShortUpdated", 'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}
            neptuner = Neptuner(project_name=neptune_kwargs['project'], api_token=neptune_kwargs['api_token'])
            runs_table_df: pd.DataFrame = neptuner.get_runs_table()
            runs_table_df
        """
        _neptuner_runs_table = self.project.fetch_runs_table()
        runs_table_df: pd.DataFrame = _neptuner_runs_table.to_pandas()
        return runs_table_df


    @classmethod
    def get_most_recent_session_runs_table(cls, runs_table_df: pd.DataFrame, ordering_datetime_col_name: str = 'sys/modification_time', oldest_included_run_date:str='2024-08-01', n_recent_results: int = 1) -> pd.DataFrame:
        """ 
        Usage:
            most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=1) # find only the rows that match the latest row_id
        
            most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=2) # find only the rows that match the latest row_id


        """
        # Filter rows based on column: 'sys/creation_time'
        assert isinstance(oldest_included_run_date, str), f"oldest_included_run_date should be of type str (provided like '2024-08-01') but it is of type: {type(oldest_included_run_date)}, oldest_included_run_date: {oldest_included_run_date}"
        ## INPUTS: runs_table_df
        active_runs_table_df: pd.DataFrame = deepcopy(runs_table_df)

        ## filter `active_runs_table_df` by modification_time
        # ordering_datetime_col_name: str = 'sys/creation_time'
        
        # Sort by column: 'sys/creation_time' (ascending)
        active_runs_table_df = active_runs_table_df.sort_values([ordering_datetime_col_name], ascending=True)
        active_runs_table_df = active_runs_table_df[active_runs_table_df[ordering_datetime_col_name] > datetime.strptime(f'{oldest_included_run_date}T00:30:00.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')]
        # active_runs_table_df = active_runs_table_df[active_runs_table_df['sys/creation_time'] > datetime.strptime('2024-09-01T00:30:00.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')]
        # active_runs_table_df = active_runs_table_df[active_runs_table_df['sys/ping_time'] > datetime.strptime('2024-09-01T00:30:00.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')]
        # active_runs_table_df 
        # Grouped on column: 'session_descriptor_string'
        # active_runs_table_df = active_runs_table_df.groupby(['session_descriptor_string']).count().reset_index()[['session_descriptor_string']]
        # Performed 1 aggregation grouped on column: 'session_descriptor_string'
        # most_recent_runs_table_df = active_runs_table_df.groupby(['session_descriptor_string']).agg(sysid_last=('sys/id', 'last')).reset_index() ## this version only has two columns: ['session_descriptor_string','sysid_last']
        # # most_recent_runs_table_df
        # is_run_included = np.isin(active_runs_table_df['sys/id'], most_recent_runs_table_df['sysid_last'])
        # most_recent_runs_table_df: pd.DataFrame = deepcopy(active_runs_table_df)[is_run_included] # find only the rows that match the latest row_id
        # Group by 'session_descriptor_string' and get the most recent `n_recent_results` rows for each group
        most_recent_runs_table_df: pd.DataFrame = (
            active_runs_table_df
            .sort_values([ordering_datetime_col_name], ascending=False)
            .groupby('session_descriptor_string')
            .head(n_recent_results)
            .reset_index(drop=True)
        )
        
        return most_recent_runs_table_df

        # active_runs_table_df = active_runs_table_df[is_run_included] # find only the rows that match the latest row_id
        ## OUTPUTS: active_runs_table_df, most_recent_runs_table_df
        # active_runs_table_df

    def get_most_recent_session_runs(self, **kwargs) -> Tuple[Dict[str,AutoValueConvertingNeptuneRun], pd.DataFrame, Dict[str, str]]:
        """ Main accessor method
        
        """
        runs_table_df: pd.DataFrame = self.get_runs_table()
        most_recent_runs_table_df: pd.DataFrame = self.get_most_recent_session_runs_table(runs_table_df=runs_table_df, **kwargs) # find only the rows that match the latest row_id
        # most_recent_runs_table_df: pd.DataFrame = self.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=2) # find only the rows that match the latest row_id
        runs_dict: Dict[str, AutoValueConvertingNeptuneRun]  = {}

        # Iterate over each run in the DataFrame
        for run_id in most_recent_runs_table_df['sys/id']:
            try:
                # Access the run by ID:
                run: AutoValueConvertingNeptuneRun = AutoValueConvertingNeptuneRun(with_id=str(run_id), project=self.project_name, api_token=self.api_token,
                                                                                    # mode="read-only",
                                                                                    mode="sync",
                                                                                   )
                runs_dict[run_id] = run
                
            except Exception as e:
                print(f"Failed to fetch figures for run {run_id}: {e}")
                

        # Drop excessive monitoring column names:
        good_column_names = [v for v in list(most_recent_runs_table_df.columns) if not v.startswith('monitoring')] # ['sys/creation_time', 'sys/description', 'sys/failed', 'sys/group_tags', 'sys/hostname', 'sys/id', 'sys/modification_time', 'sys/monitoring_time', 'sys/name', 'sys/owner', 'sys/ping_time', 'sys/running_time', 'sys/size', 'sys/state', 'sys/tags', 'sys/trashed', 'animal', 'exper_name', 'format_name', 'session_descriptor_string', 'session_name', 'source_code/entrypoint', 'source_code/git'
        # print(good_column_names) # ['sys/creation_time', 'sys/description', 'sys/failed', 'sys/group_tags', 'sys/hostname', 'sys/id', 'sys/modification_time', 'sys/monitoring_time', 'sys/name', 'sys/owner', 'sys/ping_time', 'sys/running_time', 'sys/size', 'sys/state', 'sys/tags', 'sys/trashed', 'animal', 'exper_name', 'format_name', 'session_descriptor_string', 'session_name', 'source_code/entrypoint', 'source_code/git'
        most_recent_runs_table_df = most_recent_runs_table_df[good_column_names]

        ## remove columns with a forward slash in them (such as 'sys/modification_time' and replace them with just 'modification_time')
        # valid_good_column_rename_dict = {v:'_'.join(v.split('/')) for v in good_column_names if (len(v.split('/'))>1)}
        valid_good_column_rename_dict = {v:'_'.join(v.split('/')[1:]) for v in good_column_names if (len(v.split('/'))>1)}
        most_recent_runs_table_df = most_recent_runs_table_df.rename(columns=valid_good_column_rename_dict, inplace=False)
        original_column_names_map: Dict[str, str] = dict(zip(valid_good_column_rename_dict.values(), valid_good_column_rename_dict.keys()))
        return runs_dict, most_recent_runs_table_df, original_column_names_map
        
    @classmethod
    def get_most_recent_session_logs(cls, runs_dict: Dict[str, AutoValueConvertingNeptuneRun], debug_print: bool = False) -> pd.DataFrame:
        """ 
        Usage:
            most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=1) # find only the rows that match the latest row_id
        
            most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=2) # find only the rows that match the latest row_id


        """
        # Dictionary to hold the paths of figures for each run
        run_logs = {}
        # Iterate over each run in the DataFrame
        for run_id, run in runs_dict.items():
            try:
                if debug_print:
                    print(f'run_id: {run_id}\n\trun.get_url(): "{run.get_url()}"')

                # ## make the folder for this run
                # neptune_run_output_path = neptune_project_output_path.joinpath(str(run_id)).resolve()
                # neptune_run_output_path.mkdir(exist_ok=True)
                # print(f'\tneptune_run_output_path: {neptune_run_output_path}')
                
                # # neptune_run_figures_output_path = neptune_run_output_path.joinpath(f'figures_{run_id}').resolve()
                # neptune_run_figures_output_path = neptune_run_output_path.joinpath(f'figures').resolve()
                # neptune_run_figures_output_path.mkdir(exist_ok=True)
                # print(f'\tneptune_run_figures_output_path: {neptune_run_figures_output_path}')
                
                log_contents_str, merged_log_df = run.get_log_contents()
                a_session_descriptor_string: str = run['session_descriptor_string'].fetch()
                
                # run_logs[run_id] = log_contents_str
                if a_session_descriptor_string not in run_logs:
                    # create it
                    run_logs[a_session_descriptor_string] = log_contents_str
                else:
                    # append it
                    run_logs[a_session_descriptor_string] = f"{run_logs[a_session_descriptor_string]}\n\n\n{log_contents_str}"     

                # figures_paths[run_id] = neptune_run_figures_output_path

            except Exception as e:
                print(f"Failed to fetch figures for run {run_id}: {e}")
                
        ## OUTPUTS: most_recent_runs_table_df, figures_paths
        return run_logs



    @classmethod
    def build_interactive_session_run_logs_widget(cls, context_indexed_run_logs, most_recent_runs_session_descriptor_string_to_context_map, most_recent_runs_context_indexed_run_extra_data):
        """Creates and displays an interactive session display with a tree widget, session info, and log output.

        Arguments:
        - context_indexed_run_logs: dict of session contexts to log text.
        - most_recent_runs_session_descriptor_string_to_context_map: dict of session descriptor strings to context mappings.
        - most_recent_runs_context_indexed_run_extra_data: dict of additional session context data.
        
        Returns:
        - layout: The main layout containing the tree, header, and log output.
        
        Usage:

            interactive_layout = Neptuner.build_interactive_session_run_logs_widget(context_indexed_run_logs, most_recent_runs_session_descriptor_string_to_context_map, most_recent_runs_context_indexed_run_extra_data)
            display(interactive_layout)

        """
        from pandas import Timestamp
        from datetime import datetime
        import ipywidgets as widgets
        from IPython.display import display
        from pyphocorehelpers.gui.Jupyter.TreeWidget import JupyterTreeWidget
        from typing import List, Tuple
        
        # Tree Widget __________________________________________________________________________________________________ #
        included_session_contexts: List[IdentifyingContext] = list(most_recent_runs_session_descriptor_string_to_context_map.values())
        jupyter_tree_widget = JupyterTreeWidget(included_session_contexts=included_session_contexts,
                                                on_selection_changed_callbacks=[],
                                                display_on_init=False)

        jupyter_tree_widget.tree.layout = widgets.Layout(min_width='300px', max_width='30%', overflow='auto', height='auto')

        # Content Widget ________________________________________________________________________________________________ #
        def build_session_tuple_header_widget(a_session_tuple: Tuple, included_display_session_extra_data_keys: List[str]):
            """Builds a widget to display the session tuple's properties with bold keys."""
            # Create a dictionary to hold the label widgets with bold keys
            header_label_widgets = {key: widgets.HTML(f"<b>{key}</b>: '{value}',") for key, value in a_session_tuple.items() if (key in included_display_session_extra_data_keys)}
            
            # Define a layout that enables wrapping
            box_layout = widgets.Layout(display='flex', flex_flow='row wrap', align_items='stretch', width='100%')
            
            # Create a Box with the custom layout
            header_hbox = widgets.Box(list(header_label_widgets.values()), layout=box_layout)

            # Function to update the values in the labels
            def update_header_labels_fn(new_values):
                """ captures: included_display_session_extra_data_keys, header_label_widgets
                """
                for key, value in new_values.items():
                    if key in included_display_session_extra_data_keys:
                        # Check if the value is a pandas Timestamp or datetime object and format it
                        if isinstance(value, (Timestamp, datetime)):
                            # Round to the nearest minute and format the output as 'YYYY-MM-DD HH:MM'
                            rounded_value = value.floor('T') if isinstance(value, Timestamp) else value.replace(second=0, microsecond=0)
                            formatted_value = rounded_value.strftime('%Y-%m-%d %H:%M')
                            header_label_widgets[key].value = f"<b>{key}</b>: {formatted_value}"
                        else:
                            # For non-Timestamp values, update normally
                            header_label_widgets[key].value = f"<b>{key}</b>: {value}"

            return header_hbox, header_label_widgets, update_header_labels_fn

        # Empty session tuple for initialization
        empty_session_tuple = {
            'format_name': '',
            'animal': '',
            'exper_name': '',
            'session_name': '',
            'session_descriptor_string': '',
            'id': '<Selection Not Found>',
            'hostname': '',
            'creation_time': '',
            'running_time': '',
            'ping_time': '',
            'monitoring_time': '',
            'size': '',
            'tags': '',
            'entrypoint': ''
        }

        # Build header widget and labels
        # Define the keys you want to display in the header
        display_session_extra_data_keys = ['id', 'hostname', 'creation_time', 'running_time', 'ping_time', 'monitoring_time', 'size', 'tags', 'entrypoint']
        
        header_hbox, header_label_widgets, update_header_labels_fn = build_session_tuple_header_widget(a_session_tuple=empty_session_tuple, included_display_session_extra_data_keys=display_session_extra_data_keys)

        # Create Textarea widget with a defined width for log display
        textarea = widgets.Textarea(value='<No Selection>', disabled=True, style={'font_size': '10px'}, 
                                    # layout=widgets.Layout(flex='1', width='650px', min_height='650px', height='850px'),
                                    layout=widgets.Layout(flex='1', width='100%', min_height='650px', height='850px'),
                                    )

        content_view_layout = widgets.VBox([header_hbox, textarea],
                                            # layout=widgets.Layout(min_width='400px', min_height='200px', width='auto', height='auto'),
                                            layout=widgets.Layout(min_width='500px', max_width='70%', height='auto', overflow='auto') # , width='100%'
                                           )

        # Layout the widgets side by side
        root_box = widgets.HBox([jupyter_tree_widget.tree, content_view_layout],
                            #    layout=widgets.Layout(min_width='500px', min_height='100px', width='auto', height='auto'),
                                layout=widgets.Layout(
                                    width='100%',  # Set the HBox to take full width
                                    display='flex',
                                    flex_flow='row',
                                    justify_content='space-between',  # Ensure the tree and content view are spaced apart
                                    overflow='auto'  # Enable overflow handling
                                ),
                               )

        # Callback function for when a tree node is selected
        def _on_tree_node_selection_changed(selected_node, selected_context):
            """Updates the header and log content based on the selected tree node."""
            if isinstance(selected_context, dict):
                selected_context = IdentifyingContext(**selected_context)
            
            # # Prevent scrolling behavior
            # import IPython
            # IPython.display.clear_output(wait=True)  # Clear output without scrolling
            # display(root_box)  # Re-display the root box to maintain focus
            
            curr_context_extra_data_tuple = most_recent_runs_context_indexed_run_extra_data.get(selected_context, empty_session_tuple)
            update_header_labels_fn(curr_context_extra_data_tuple)
            
            curr_context_run_log = context_indexed_run_logs.get(selected_context, '<Context Not Found>')
            textarea.value = curr_context_run_log

        # Set the callback function to trigger on tree selection changes
        jupyter_tree_widget.on_selection_changed_callback = [_on_tree_node_selection_changed]

        # Return the layout for display
        return root_box






# ==================================================================================================================== #
# Independent Helper Functions                                                                                         #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['neptune', 'figures', 'output', 'cloud', 'logging', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 04:34', related_items=[])
def neptune_output_figures(curr_active_pipeline):
    """ Uploads the completed figures to neptune.ai from the pipeline's `.registered_output_files` items. 
    
    Usage:
        from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
        neptune_output_figures(curr_active_pipeline)
    """
    import neptune # for logging progress and results
    from neptune.types import File

    neptune_kwargs = {'project':"commander.pho/PhoDibaLongShort2023",
    'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}

    # kdiba_gor01_one_2006-6-09_1-22-43_sess
    active_session_context_string = str(curr_active_pipeline.active_sess_config)

    with neptune.init_run(**neptune_kwargs) as run:
        for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items():
            ## Get list of figures output and register them to neptune:
            a_fig_context = fig_metadata['context']
            a_full_figure_path_key = f"session/{active_session_context_string}/figures/{str(a_fig_context)}"
            print(f'a_fig_path: {a_fig_path}, a_fig_context: {a_fig_context}\n\t{a_full_figure_path_key}')
            # fig_objects = fig_metadata.get('fig', None)
            # if fig_objects is not None:
            # 	# upload the actual figure objects
            # 	if not isinstance(fig_objects, (tuple, list)):
            # 		fig_objects = (fig_objects,) # wrap in a tuple or convert list to tuple
            # 	# now fig_objects better be a tuple
            # 	assert isinstance(fig_objects, tuple), f"{fig_objects} should be a tuple!"
            # 	# run[a_full_figure_path_key].upload(fig_metadata['fig'])
            # 	for a_fig_obj in fig_objects:
            # 		run[a_full_figure_path_key].append(a_fig_obj) # append the whole series of related figures
            # else:
            # upload as file
            run[a_full_figure_path_key].upload(str(a_fig_path))