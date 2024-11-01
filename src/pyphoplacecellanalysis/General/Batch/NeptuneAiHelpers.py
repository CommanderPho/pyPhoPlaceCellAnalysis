from copy import deepcopy
import fnmatch
import os
from attrs import define, Factory, field
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from typing import NewType
from nptyping import NDArray
import neuropy.utils.type_aliases as types

import neptune # for logging progress and results
from neptune.types import File
from neptune.utils import stringify_unsupported
from neptune.exceptions import NeptuneException, MissingFieldException
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
from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata

from neuropy.utils.indexing_helpers import get_nested_value, flatten_dict
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

# SessionDescriptorString: TypeAlias = str # an integer index that is an aclu
SessionDescriptorString = NewType('SessionDescriptorString', str) # session_descriptor_string
RunID = NewType('RunID', str) # like 'LS2023-1485'
NeptuneKeyPath = NewType('NeptuneKeyPath', str) # a path that indexes in to the Neptune.ai resource
NeptuneProjectName = NewType('NeptuneProjectName', str) # like 'run' or 'utility'


# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import SessionDescriptorString, RunID

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



class KnownNeptuneProjects:
	""" 
	
	from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import KnownNeptuneProjects

	neptune_kwargs = KnownNeptuneProjects.get_PhoDibaBatchProcessing_neptune_kwargs()
	# or
	neptune_kwargs = KnownNeptuneProjects.get_PhoDibaLongShortUpdated_neptune_kwargs()

	"""
	@staticmethod
	def get_PhoDibaBatchProcessing_neptune_kwargs():
		""" # the one with logs and stuff """
		return dict(
			project="commander.pho/PhoDibaBatchProcessing",
			api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==",
		)


	@staticmethod
	def get_PhoDibaLongShortUpdated_neptune_kwargs():
		""" # the one with images and stuff """
		return dict(
			project="commander.pho/PhoDibaLongShortUpdated",
			api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==",
		)


	@classmethod
	def get_PhoDibaBatchProcessing_neptuner(cls) -> "Neptuner":
		"""# the one with logs and stuff """
		neptune_kwargs = cls.get_PhoDibaBatchProcessing_neptune_kwargs() # the one with images and stuff
		PhoDibaBatchProcessing_neptuner = Neptuner(project_name=neptune_kwargs['project'], api_token=neptune_kwargs['api_token'])
		# project_main_name: str = neptuner.project_name.split('/')[-1] # 'PhoDibaLongShortUpdated'
		return PhoDibaBatchProcessing_neptuner


	@classmethod
	def get_PhoDibaLongShortUpdated_neptuner(cls):
		""" # the one with images and stuff """
		neptune_kwargs = cls.get_PhoDibaLongShortUpdated_neptune_kwargs() # the one with images and stuff
		PhoDibaLongShortUpdated_neptuner = Neptuner(project_name=neptune_kwargs['project'], api_token=neptune_kwargs['api_token'])
		# project_main_name: str = neptuner.project_name.split('/')[-1] # 'PhoDibaLongShortUpdated'
		return PhoDibaLongShortUpdated_neptuner





# ==================================================================================================================== #
# UTILITY FUNCTIONS                                                                                                    #
# ==================================================================================================================== #




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


def flatten_context_nested_dict(_context_figures_dict):
	""" not super general or great
	
	from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import flatten_context_nested_dict
	"""
	_flattened_context_path_dict= {}
	_flat_out_path_items = []
	## Do care about the Contexts, but nothing beyond that:
	for k, v in _context_figures_dict.items():
		# _flattened_context_path_dict[k] = flatten_dict(v)
		_flattened_context_path_dict[k] = list(flatten_dict(v).values())
		if len(_flattened_context_path_dict[k]) > 0:
			## only returning the first item, that's not great
			_flat_out_path_items.append(_flattened_context_path_dict[k][0])

	return _flattened_context_path_dict, _flat_out_path_items


@function_attributes(short_name=None, tags=['vscode_workspace', 'vscode', 'logs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-29 07:49', related_items=[])
def build_vscode_log_browsing_workspace(logs_paths):
	""" builds a VSCode workspace for the batch python scripts
	
	Usage:
		from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import build_vscode_log_browsing_workspace

		## INPUTS: merged_log_files_paths
		path_to_watch = "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs"
		log_files = [v.as_posix() for v in merged_log_files_paths]

		vscode_workspace_path = build_vscode_log_browsing_workspace(logs_paths=log_files)
		vscode_workspace_path

	"""
	import sys
	import os
	import platform
	import pkg_resources # for Slurm templating
	from jinja2 import Environment, FileSystemLoader, Template 

	is_platform_windows: bool = False
	if (platform.system() == 'Windows'):
		is_platform_windows = True
	else:
		is_platform_windows = False

	assert len(logs_paths) > 0, f"logs_paths is empty!"
	top_level_merged_logs_folders_path: Path = Path(logs_paths[0]).resolve().parent # "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged"
	print(f'top_level_merged_logs_folders_path: {top_level_merged_logs_folders_path}')
	top_level_logs_folders_path: Path = Path(logs_paths[0]).resolve().parent.parent # "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs"
	print(f'top_level_logs_folders_path: {top_level_logs_folders_path}')
	top_level_neptune_folders_path: Path = top_level_logs_folders_path.parent # "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune"
	print(f'top_level_neptune_folders_path: {top_level_neptune_folders_path}')
	
	workspace_relative_log_folders: List[Path] = [Path(a_path).relative_to(top_level_neptune_folders_path).resolve() for a_path in [top_level_merged_logs_folders_path, top_level_logs_folders_path]]
	print(f'workspace_relative_log_folders: {workspace_relative_log_folders}')
		
	# script_folders: List[Path] = [top_level_neptune_folders_path] + [Path(a_path).parent.resolve() for a_path in logs_paths]
	relative_log_files: List[Path] = [Path(a_path).relative_to(top_level_neptune_folders_path).resolve() for a_path in logs_paths] # not used
	print(f'relative_log_files: {relative_log_files}')
	# {
	#     "path": "L:/Scratch/gen_scripts",
	#     "name": "gen_scripts_root"
	# },
	
	vscode_workspace_path = top_level_neptune_folders_path.joinpath('logs_workspace.code-workspace').resolve()
	print(f'vscode_workspace_path: {vscode_workspace_path}')

	# Set up Jinja2 environment
	template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
	env = Environment(loader=FileSystemLoader(template_path))
	template = env.get_template('vscode_logs_workspace_template.code-workspace.j2')
	# Render the template with the provided variables
	
	# Define folders as a list of dictionaries
	# folders = [
	#     {'path': '/path/to/your/project1', 'name': 'Project1'},
	#     {'path': '/path/to/your/project2', 'name': 'Project2'},
	# ]
	folders = [
		{'path': f'{a_folder.as_posix()}', 'name': f'{a_folder.name}'}
		for a_folder in workspace_relative_log_folders
	]

	# Define variables
	variables = {
		'folders': folders,
		'is_platform_windows': is_platform_windows,
	}

	# Render the template with variables
	workspace_file_content = template.render(variables)

	# Write the generated content to a workspace file
	with open(vscode_workspace_path, 'w') as f:
		f.write(workspace_file_content)

	return vscode_workspace_path



# ==================================================================================================================== #
# END UTILITY FUNCTIONS                                                                                                #
# ==================================================================================================================== #

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
		""" gets the appropriate logs to stdout and stderr for this run 
		"""
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
		# Drop rows where 'value' contains only whitespaces or newlines
		stderr_log_df = stderr_log_df[~stderr_log_df['value'].str.fullmatch(r'\s*')]
		stderr_log_df['value'] = [f"<STDERR>:\t {v}" for v in stderr_log_df['value'].values] # prepend "<STDERR>:\t " to all entries in the error dict
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


	def download_image(self: "AutoValueConvertingNeptuneRun", fig_input_key: str, a_session_descriptor_str: str, neptune_project_figures_output_path: Path, debug_print=False) -> Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Path]]]:
		""" locates and downloads an image with a specific `fig_input_key` like "display_fn_name:display_short_long_pf1D_comparison/track:short"
		
		"""
		## INPUTS: a_parsed_structure
		## UPDATES: _context_fig_files_dict
		# _context_fig_files_dict: Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Any]]] = {}
		_context_fig_files_dict: Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Path]]] = {}

		a_parsed_structure = self.get_structure().get('outputs', {}).get('figures', None)
		if a_parsed_structure is None:
			raise ValueError(f'No "outputs/values" in this run.') #skip this one

		# a_parsed_structure = a_run.get_structure()['outputs']['figures']
		assert isinstance(a_parsed_structure, dict), f"type(a_parsed_structure): {type(a_parsed_structure)} instead of dict. a_parsed_structure: {a_parsed_structure}"
		if debug_print:
			print(f'a_parsed_structure: {a_parsed_structure}')
		
		## parse the key:
		fig_input_key_parts: List[str] = fig_input_key.split('/') # ['display_fn_name:display_short_long_pf1D_comparison', 'track:long']
		# fig_input_key_parts

		fig_split_key_value_pair_parts = [k.split(':')[-1] for k in fig_input_key_parts] ## remove the keys from the path, ['display_short_long_pf1D_comparison', 'short']
		# fig_split_key_value_pair_parts

		## Find the figure or figure(s):

		a_fig_file_field = get_nested_value(a_parsed_structure, fig_input_key_parts)
		a_fig_file_field
		
		## INPUTS: a_session_figures_output_path, fig_split_key_value_pair_parts, a_fig_file_field
		## Build the output file path
		a_session_figures_output_path = neptune_project_figures_output_path.joinpath(a_session_descriptor_str)
		a_session_figures_output_path.mkdir(exist_ok=True)

		if a_session_descriptor_str not in _context_fig_files_dict:
			_context_fig_files_dict[a_session_descriptor_str] = {}
			

		# if isinstance(a_fig_file_field, File): # neptune.attributes.atoms.file.File
		if (not isinstance(a_fig_file_field, dict)): # neptune.attributes.atoms.file.File
			## typical case where a full, complete path is passed
			a_fig_output_name: str = '-'.join(fig_split_key_value_pair_parts) + '.png'
			a_fig_output_path = a_session_figures_output_path.joinpath(a_fig_output_name).resolve()
			if debug_print:
				print(f'a_fig_output_path: "{a_fig_output_path}"')

			try:
				_a_download_result = a_fig_file_field.download(destination=a_fig_output_path.as_posix())
				_context_fig_files_dict[a_session_descriptor_str][fig_input_key] = a_fig_output_path.as_posix()
				if debug_print:
					print(f'\tdone.')
			except MissingFieldException as err:
				# print(f'MissingFieldException for a_run.id: {a_run_id} (err: {err})')
				print(f'MissingFieldException for a_run.id: {self}, err: {err}')
				pass
		else:
			## Non-File, usually a path that contains multiple files
			print(f'not a figure file! a dictionary instead probably: type(a_fig_file_field): {type(a_fig_file_field)}') # neptune.attributes.atoms.file.File
			a_flattened_figure_dict = flatten_dict(a_fig_file_field) ## use flatten_dict to turn potentially nested dictionaries with leaf of type File into a flat dictionary with keys of string, values of type File.
			for a_sub_key, a_sub_fig_file_field in a_flattened_figure_dict.items():
				## Each sub-item:
				sub_fig_full_input_key: str = '/'.join([fig_input_key, a_sub_key])
				if debug_print:
					print(f'sub_fig_full_input_key: "{sub_fig_full_input_key}"')
				sub_fig_input_key_parts: List[str] = a_sub_key.split('/') # ['display_fn_name:display_short_long_pf1D_comparison', 'track:long']
				sub_fig_split_key_value_pair_parts = [k.split(':')[-1] for k in sub_fig_input_key_parts]
				a_fig_output_name: str = '-'.join(fig_split_key_value_pair_parts + sub_fig_split_key_value_pair_parts) + '.png' # add in the `fig_split_key_value_pair_parts` as well
				a_fig_output_path = a_session_figures_output_path.joinpath(a_fig_output_name).resolve()
				if debug_print:
					print(f'a_fig_output_path: "{a_fig_output_path}"')
				## Try the download
				try:
					_a_download_result = a_sub_fig_file_field.download(destination=a_fig_output_path.as_posix())
					_context_fig_files_dict[a_session_descriptor_str][sub_fig_full_input_key] = a_fig_output_path.as_posix()
					if debug_print:
						print(f'\tdone.')
				except MissingFieldException as err:
					# print(f'MissingFieldException for a_run.id: {a_run_id} (err: {err})')
					print(f'MissingFieldException for a_run.id: {self}')
					pass
	
		return _context_fig_files_dict



@define(slots=False, repr=False)
class NeptuneRunCollectedResults:
	""" 
	from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import NeptuneRunCollectedResults
	
	"""
	runs_table_df: pd.DataFrame = field()
	most_recent_runs_table_df: pd.DataFrame = field()
	runs_dict: Dict[RunID, AutoValueConvertingNeptuneRun] = field()


	most_recent_runs_session_descriptor_string_to_context_map: Dict[SessionDescriptorString, IdentifyingContext] = field()
	session_descriptor_indexed_runs_list_dict: Dict[SessionDescriptorString, List[AutoValueConvertingNeptuneRun]] = field()
	context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = field()

	context_indexed_run_logs: Dict[IdentifyingContext, str] = field()
	most_recent_runs_context_indexed_run_extra_data: Dict[IdentifyingContext, Dict] = field()
	

	# @property()
	def get_resolved_structure(self, fig_input_key: str = "/", debug_print=False) -> Dict[IdentifyingContext, Dict[RunID, Dict[NeptuneKeyPath, Any]]]:
		""" Downloads figures
		Usage:
			fig_input_key: str = "display_fn_name:running_and_replay_speeds_over_time"
			_context_run_structure_dict = neptune_run_collected_results.get_resolved_structure()
			_context_run_structure_dict

		"""
		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = self.context_indexed_runs_list_dict
		_context_run_structure_dict: Dict[IdentifyingContext, Dict[RunID, Dict[NeptuneKeyPath, Any]]] = {} # actually Any is Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Path]]]

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			_context_run_structure_dict[a_ctxt] = {}
			a_session_descriptor_str: str = a_ctxt.get_description(separator='_', subset_excludelist='format_name') # 'kdiba_gor01_two_2006-6-07_16-40-19'
				
			for a_run in a_run_list:
				try:
					a_run_id: str = a_run['sys/id'].fetch()
					_context_run_structure_dict[a_ctxt][a_run_id] = a_run.get_parsed_structure()
				except (ValueError, KeyError, MissingFieldException) as e:
					continue # just try the next one
				except Exception as e:
					raise e
			# END FOR a_run
		# END FOR a_ctxt
		
		## OUTPUTS: _context_run_structure_dict
		return _context_run_structure_dict


	@classmethod
	def _perform_export_log_files_to_separate_files(cls, context_indexed_run_logs: Dict[IdentifyingContext, str], neptune_logs_output_path: Path):
		Assert.path_exists(neptune_logs_output_path)
		_out_log_paths = {}
		for k, v in context_indexed_run_logs.items():
			## flat filename approach:
			session_context_path_fragment: str = k.get_description(separator='=', subset_excludelist='format_name')
			# session_context_path_fragment
			session_context_path = neptune_logs_output_path.joinpath(f"{session_context_path_fragment}.log").resolve()
			try:
				_write_status: int = 0
				with open(session_context_path, 'w') as f:
					_write_status = f.write(v)
				if _write_status > 0:
					_out_log_paths[k] = session_context_path
				else:
					print(f'WARNING: write did not fail, but _write_status is not > 0: {_write_status} for session_context_path: "{session_context_path.as_posix()}"')
			except Exception as e:
				raise e
		# end for
		return _out_log_paths


	def download_uploaded_log_files(self, neptune_logs_output_path:Path):
		""" 
		Usage:
			_context_log_files_dict = neptune_run_collected_results.download_uploaded_log_files(neptune_logs_output_path=neptune_logs_output_path)
			_context_log_files_dict

		"""
		Assert.path_exists(neptune_logs_output_path)

		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = self.context_indexed_runs_list_dict
		# context_indexed_runs_list_dict

		## INPUTS: context_indexed_runs_list_dict
		_context_log_files_dict = {}

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			# _parsed_run_structure_dict[a_ctxt] = {}
			_context_log_files_dict[a_ctxt] = {}
			curr_ctxt_num_runs: int = len(a_run_list)
			for a_run in a_run_list:
				a_run_id: str = a_run['sys/id'].fetch()
				a_modification_time: datetime = a_run['sys/modification_time'].fetch()
				assert isinstance(a_modification_time, datetime), f"a_modification_time is not of type datetime, it is instead type(a_modification_time): {type(a_modification_time)}, value: {a_modification_time}"
				
				try:
					a_script_type: str = a_run['parameters/script_type'].fetch() # should be either ['figures', 'run']

					a_formatted_modification_time: str = a_modification_time.strftime('%Y%m%dT%H%M%S') # Example Output: 20240427T153045
					a_log_file_field = a_run['outputs']['log'] # either <File field at "outputs/log"> or <Unassigned field at "outputs/log">
					a_run_ctxt = a_ctxt.adding_context_if_missing(script_type=a_script_type,
																#    run_id=a_run_id
																)
					
					# a_log_file_basename: str = a_run_ctxt.get_description(separator='|') # invalid on windows
					a_log_file_basename: str = a_run_ctxt.get_description(separator='--') # "kdiba--gor01--one--2006-6-07_11-26-53--figures--PHDBATCH-1486.log"
					a_log_file_filename: str = f"{a_formatted_modification_time}--{a_run_id}--" + a_log_file_basename + '.log' # start with modification time so they can be sorted in the filesystem
					a_log_file_dest_path = neptune_logs_output_path.joinpath(a_log_file_filename).resolve()
					# print(f'a_log_file_dest_path: {a_log_file_dest_path}')

					_a_download_result = a_log_file_field.download(destination=a_log_file_dest_path.as_posix())
					_context_log_files_dict[a_ctxt][a_run_id] = a_log_file_dest_path.as_posix()
					# a_log_file_dest_path
					FilesystemMetadata.set_modification_time(a_log_file_dest_path.as_posix(), new_time=a_modification_time) ## try to set the modification time

				except MissingFieldException as err:
					# print(f'MissingFieldException for a_run.id: {a_run_id} (err: {err})')
					print(f'MissingFieldException for a_run.id: "{a_run_id}". Make sure that this Neptuner object is for project `neptune_kwargs = KnownNeptuneProjects.get_PhoDibaBatchProcessing_neptune_kwargs()`.')
					pass

			# END FOR a_run
		## OUTPUTS: _context_log_files_dict
		return _context_log_files_dict
	

	def download_uploaded_figure_files(self, neptune_project_figures_output_path:Path, fig_input_key: str = "display_fn_name:running_and_replay_speeds_over_time", debug_print=False):
		""" Downloads figures
		Usage:
			fig_input_key: str = "display_fn_name:running_and_replay_speeds_over_time"
			_context_figures_dict = neptune_run_collected_results.download_uploaded_figure_files(neptune_project_figures_output_path=neptune_project_figures_output_path, fig_input_key=fig_input_key)
			_context_figures_dict

		"""
		Assert.path_exists(neptune_project_figures_output_path)

		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = self.context_indexed_runs_list_dict
		
		_context_figures_dict: Dict[IdentifyingContext, Dict[RunID, Dict[NeptuneKeyPath, Any]]] = {} # actually Any is Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Path]]]

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			_context_figures_dict[a_ctxt] = {}
			a_session_descriptor_str: str = a_ctxt.get_description(separator='_', subset_excludelist='format_name') # 'kdiba_gor01_two_2006-6-07_16-40-19'
				
			for a_run in a_run_list:
				try:
					a_run_id: str = a_run['sys/id'].fetch()
					_context_figures_dict[a_ctxt][a_run_id] = a_run.download_image(fig_input_key=fig_input_key, a_session_descriptor_str=a_session_descriptor_str, neptune_project_figures_output_path=neptune_project_figures_output_path, debug_print=debug_print)
				except (ValueError, KeyError, MissingFieldException) as e:
					continue # just try the next one
				except Exception as e:
					raise e
			# END FOR a_run
		# END FOR a_ctxt
		
		## OUTPUTS: _context_figures_dict
		return _context_figures_dict
	
	

@define(slots=False, repr=False)
class Neptuner(object):
	"""An object to maintain state for neptune.ai outputs.
	  
	from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import Neptuner
	neptuner = Neptuner()

	Mostly initialized from `pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing.batch_perform_all_plots` to programmatically log the figures
	
	
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
			self.run = AutoValueConvertingNeptuneRun(project=self.project_name, api_token=self.api_token, source_files=[], monitoring_namespace="monitoring") # see git_ref=GitRef(repository_path="/path/to/repo")

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
		# active_runs_table_df = active_runs_table_df.sort_values([ordering_datetime_col_name], ascending=True)
		active_runs_table_df = active_runs_table_df[active_runs_table_df[ordering_datetime_col_name] > datetime.strptime(f'{oldest_included_run_date}T00:30:00.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')]
		# Group by 'session_descriptor_string' and get the most recent `n_recent_results` rows for each group
		most_recent_runs_table_df: pd.DataFrame = (
			active_runs_table_df
			.sort_values([ordering_datetime_col_name], ascending=False)
			.groupby('session_descriptor_string')
			.head(n_recent_results)
			.reset_index(drop=True)
		)
		
		return most_recent_runs_table_df


	def get_most_recent_session_runs(self, oldest_included_run_date:str='2024-08-01', n_recent_results: int = 1, **kwargs) -> NeptuneRunCollectedResults: #Tuple[Dict[RunID,AutoValueConvertingNeptuneRun], pd.DataFrame, Dict[str, str]]:
		""" Main accessor method
		
		"""
		runs_table_df: pd.DataFrame = self.get_runs_table()
		most_recent_runs_table_df: pd.DataFrame = self.get_most_recent_session_runs_table(runs_table_df=runs_table_df, oldest_included_run_date=oldest_included_run_date, n_recent_results=n_recent_results, **kwargs) # find only the rows that match the latest row_id
		# most_recent_runs_table_df: pd.DataFrame = self.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=2) # find only the rows that match the latest row_id
		runs_dict: Dict[RunID, AutoValueConvertingNeptuneRun]  = {}

		# Iterate over each run in the DataFrame
		for run_id in most_recent_runs_table_df['sys/id']:
			try:
				# Access the run by ID:
				run: AutoValueConvertingNeptuneRun = AutoValueConvertingNeptuneRun(with_id=str(run_id), project=self.project_name, api_token=self.api_token,
																					mode="read-only",
																					# mode="sync",
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


		session_column_individual_variables = ['format_name', 'animal', 'exper_name', 'session_name']
		session_column_variables = ['session_descriptor_string']
		# processing_status_column_names = ['sys/id', 'sys/hostname', 'sys/creation_time', 'sys/running_time', 'sys/ping_time', 'sys/monitoring_time', 'sys/size', 'sys/tags', 'source_code/entrypoint']
		# processing_status_column_names = ['sys/id', 'sys/hostname', 'sys/creation_time', 'sys/running_time', 'sys/ping_time', 'sys/monitoring_time', 'sys/size', 'sys/tags', 'source_code/entrypoint']
		processing_status_column_names = ['id', 'hostname', 'creation_time', 'running_time', 'ping_time', 'monitoring_time', 'size', 'tags', 'entrypoint']

		most_recent_runs_session_descriptor_string_to_context_map: Dict[SessionDescriptorString, IdentifyingContext] = {v.session_descriptor_string:IdentifyingContext(format_name=v.format_name, animal=v.animal, exper_name=v.exper_name, session_name=v.session_name) for v in most_recent_runs_table_df[session_column_individual_variables + session_column_variables].itertuples()}
		session_descriptor_indexed_runs_list_dict: Dict[SessionDescriptorString, List[AutoValueConvertingNeptuneRun]] = {}
		# Iterate over each run in the DataFrame
		for run_id, run in runs_dict.items():
			try:
				a_session_descriptor_string: SessionDescriptorString = SessionDescriptorString(run['session_descriptor_string'].fetch())
				# run_logs[run_id] = log_contents_str
				if a_session_descriptor_string not in session_descriptor_indexed_runs_list_dict:
					# create it
					session_descriptor_indexed_runs_list_dict[a_session_descriptor_string] = [run]
				else:
					# append it
					# run_logs[a_session_descriptor_string] = f"{run_logs[a_session_descriptor_string]}\n\n\n{merged_log_contents_str}"
					session_descriptor_indexed_runs_list_dict[a_session_descriptor_string].append(run)
			except Exception as e:
				print(f"Failed to fetch session_descriptor_string for run {run_id}: {e}")
				

		# OUTPUTS: session_descriptor_indexed_runs_list_dict
		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = {most_recent_runs_session_descriptor_string_to_context_map[k]:v for k, v in session_descriptor_indexed_runs_list_dict.items()} # get the IdentifyingContext indexed item

		# Logging Outputs ____________________________________________________________________________________________________ #
		run_logs: Dict[SessionDescriptorString, str] = self.get_most_recent_session_logs(runs_dict=runs_dict)
		## INPUTS: most_recent_runs_session_descriptor_string_to_context_map, run_logs
		context_indexed_run_logs: Dict[IdentifyingContext, str] = {most_recent_runs_session_descriptor_string_to_context_map[k]:v for k, v in run_logs.items()} # get the IdentifyingContext indexed item
		## INPUTS: most_recent_runs_table_df
		most_recent_runs_context_indexed_run_extra_data: Dict[IdentifyingContext, Dict] = {IdentifyingContext(format_name=v.format_name, animal=v.animal, exper_name=v.exper_name, session_name=v.session_name):v._asdict() for v in most_recent_runs_table_df[session_column_individual_variables + session_column_variables + processing_status_column_names].itertuples(index=False, name='SessionTuple')}
		# most_recent_runs_context_indexed_run_extra_data # SessionTuple(format_name='kdiba', animal='pin01', exper_name='one', session_name='11-02_17-46-44', session_descriptor_string='kdiba_pin01_one_11-02_17-46-44_sess', id='LS2023-1335', hostname='gl3126.arc-ts.umich.edu', creation_time=Timestamp('2024-08-29 16:39:16.613000'), running_time=8735.629, ping_time=Timestamp('2024-09-24 08:38:06.626000'), monitoring_time=1543, size=28686905.0, tags='11-02_17-46-44,one,kdiba,pin01', entrypoint='figures_kdiba_pin01_one_11-02_17-46-44.py')

		neptune_run_collected_results = NeptuneRunCollectedResults(
			runs_table_df=runs_table_df,
			most_recent_runs_table_df=most_recent_runs_table_df,
			runs_dict=runs_dict,
			most_recent_runs_session_descriptor_string_to_context_map=most_recent_runs_session_descriptor_string_to_context_map,
			session_descriptor_indexed_runs_list_dict=session_descriptor_indexed_runs_list_dict,
			context_indexed_runs_list_dict=context_indexed_runs_list_dict,
			context_indexed_run_logs=context_indexed_run_logs,
			most_recent_runs_context_indexed_run_extra_data=most_recent_runs_context_indexed_run_extra_data,
		)

		return neptune_run_collected_results
		# return runs_dict, most_recent_runs_table_df, original_column_names_map
		

	@classmethod
	def get_most_recent_session_logs(cls, runs_dict: Dict[RunID, AutoValueConvertingNeptuneRun], debug_print: bool = False) -> Dict[SessionDescriptorString, str]:
		""" 
		Usage:
			most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=1) # find only the rows that match the latest row_id
		
			most_recent_runs_table_df: pd.DataFrame = neptuner.get_most_recent_session_runs_table(runs_table_df=deepcopy(runs_table_df), oldest_included_run_date='2024-06-01', n_recent_results=2) # find only the rows that match the latest row_id


		"""
		# _separator_string: str = "\n\n\n"
		_separator_string: str = """\n\n
# ==================================================================================================================== #
# LOG SEPARATOR                                                                                                        #
# ==================================================================================================================== #
"""
		# Dictionary to hold the paths of figures for each run
		run_logs: Dict[SessionDescriptorString, str] = {}
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
				
				merged_log_contents_str, merged_log_df = run.get_log_contents()
				a_session_descriptor_string: SessionDescriptorString = SessionDescriptorString(run['session_descriptor_string'].fetch())
				
				if len(merged_log_contents_str) > 0:
					# run_logs[run_id] = log_contents_str
					if a_session_descriptor_string not in run_logs:
						# create it
						run_logs[a_session_descriptor_string] = merged_log_contents_str
					else:
						# append it
						# run_logs[a_session_descriptor_string] = f"{run_logs[a_session_descriptor_string]}\n\n\n{merged_log_contents_str}"
						run_logs[a_session_descriptor_string] = (run_logs[a_session_descriptor_string] + _separator_string + merged_log_contents_str) #f"{run_logs[a_session_descriptor_string]}\n\n\n{merged_log_contents_str}"
						 

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






@define(slots=False, repr=False, eq=False)
class PhoDibaProjectsNeptuner(object):
	"""An object to combine the two separate Neptuner objects, maintaining access to the two different neptune.ai projects and intellegently allowing access to both figures and logs.
	  
	Usage:
		from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import PhoDibaProjectsNeptuner, Neptuner, AutoValueConvertingNeptuneRun, set_environment_variables, SessionDescriptorString, RunID, NeptuneRunCollectedResults, KnownNeptuneProjects

		combined_neptuner: PhoDibaProjectsNeptuner = PhoDibaProjectsNeptuner()
		combined_neptuner.get_most_recent_session_runs(oldest_included_run_date='2024-10-15', n_recent_results=16)

	
	"""
	main_neptune_project: Neptuner = field(init=False, metadata={'comment': 'the one with images and stuff'})
	utility_neptune_project: Neptuner = field(init=False, metadata={'comment': 'the one with logs and stuff'})
	
	main_run_collected_results: NeptuneRunCollectedResults = field(init=False)
	utility_run_collected_results: NeptuneRunCollectedResults = field(init=False)
	
	
	root_output_path: Path = field(init=False)
	figures_output_path: Path = field(init=False)
	logs_output_path: Path = field(init=False)
	
	# project_name: str = field(default="commander.pho/PhoDibaLongShortUpdated")
	# api_token: str = field(default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==")
	# project: neptune.Project = field(init=False)
	# run: AutoValueConvertingNeptuneRun = field(init=False)

	# outputs = field(init=False)
	# figures = field(init=False)
	
	@property
	def main_run(self):
		"""the one with images and stuff"""
		return self.main_neptune_project.run

	@property
	def utility_run(self):
		"""The main_run property."""
		return self.utility_neptune_project.run
	

	@property
	def projects_neptuner_dict(self) -> Dict[NeptuneProjectName, Neptuner]:
		return {'main': self.main_neptune_project, 'utility': self.utility_neptune_project}

	@property
	def projects_run_collected_results_dict(self) -> Dict[NeptuneProjectName, NeptuneRunCollectedResults]:
		return {'main': self.main_run_collected_results, 'utility': self.utility_run_collected_results}

	@property
	def projects_resolved_run_structure_dict(self) -> Dict[NeptuneProjectName, Dict[IdentifyingContext, Dict[RunID, Dict]]]:
		""" lists the actual structure """
		run_structure_dict_dict = {}
		for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items():
			run_structure_dict_dict[a_name] = a_run_collected_results.get_resolved_structure()
		return run_structure_dict_dict



	# @property
	# def projects_run_dict(self) -> Dict[str, Neptuner]:
	#     return {'main': self.main_neptune_project.run, 'utility': self.utility_run_collected_results.run}
	
	@property
	def context_indexed_run_logs(self) -> Dict[IdentifyingContext, str]:
		"""The main_run property."""
		return self.utility_run_collected_results.context_indexed_run_logs # get the IdentifyingContext indexed item


	def __attrs_post_init__(self):
		self.main_neptune_project = KnownNeptuneProjects.get_PhoDibaLongShortUpdated_neptuner() # the one with images and stuff
		self.utility_neptune_project = KnownNeptuneProjects.get_PhoDibaBatchProcessing_neptuner() # the one with logs and stuff
		
		self.root_output_path = find_first_extant_path(path_list=[Path(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data/neptune").resolve(),
										Path("/home/halechr/repos/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data").resolve(),
										Path('EXTERNAL/PhoDibaPaper2024Book/data/neptune').resolve(),
										Path('/Users/pho/repo/Pho Secondary Workspace/Spike3DEnv/Spike3DWorkEnv/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data').resolve(),
		])
		self.root_output_path.mkdir(exist_ok=True)

		self.figures_output_path = self.root_output_path.joinpath('figs').resolve()
		self.figures_output_path.mkdir(exist_ok=True, parents=True)

		self.logs_output_path = self.root_output_path.joinpath('logs').resolve()
		self.logs_output_path.mkdir(exist_ok=True, parents=True)


		
	def stop(self):
		for a_neptuner in (self.main_neptune_project, self.utility_neptune_project):
			a_neptuner.stop()


	def run_with_pipeline(self, curr_active_pipeline):
		""" starts a new run with the provided pipeline. """
		for a_neptuner in (self.main_neptune_project, self.utility_neptune_project):
			a_neptuner.run_with_pipeline(curr_active_pipeline=curr_active_pipeline)

	@classmethod
	def init_with_pipeline(cls, curr_active_pipeline):
		""" creates a new Neptuner object corresponding to a particular session and starts a run. """
		new = cls()
		new.run_with_pipeline(curr_active_pipeline=curr_active_pipeline)
		return new


	def get_most_recent_session_runs(self, utility_oldest_included_run_date: Optional[str]=None, utility_n_recent_results: Optional[int]=2,
									 main_oldest_included_run_date: Optional[str]=None, main_n_recent_results: Optional[int]=None,
									 oldest_included_run_date:str='2024-08-01', n_recent_results: int = 1, **kwargs): #Tuple[Dict[RunID,AutoValueConvertingNeptuneRun], pd.DataFrame, Dict[str, str]]:
			""" Main accessor method
			
			"""
			utility_oldest_included_run_date = utility_oldest_included_run_date or oldest_included_run_date
			utility_n_recent_results = utility_n_recent_results or n_recent_results
			self.main_run_collected_results = self.main_neptune_project.get_most_recent_session_runs(oldest_included_run_date=utility_oldest_included_run_date, n_recent_results=utility_n_recent_results, **kwargs) # : NeptuneRunCollectedResults
			
			main_oldest_included_run_date = main_oldest_included_run_date or oldest_included_run_date
			main_n_recent_results = main_n_recent_results or n_recent_results
			self.utility_run_collected_results = self.utility_neptune_project.get_most_recent_session_runs(oldest_included_run_date=main_oldest_included_run_date, n_recent_results=main_n_recent_results, **kwargs) # : NeptuneRunCollectedResults
			

	def try_get_figures(self, fig_input_key_match_pattern: str='*', should_download_figures: bool = True, debug_print: bool = True):
		""" tries to download the figures with the fig_input_key """
		
		
		neptune_project_figures_output_path:Path = self.figures_output_path
		Assert.path_exists(neptune_project_figures_output_path)

		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = self.main_run_collected_results.context_indexed_runs_list_dict

		if should_download_figures:
			_context_figures_dict: Dict[IdentifyingContext, Dict[RunID, Dict[NeptuneKeyPath, Any]]] = {} # actually Any is Dict[IdentifyingContext, Dict[SessionDescriptorString, Dict[NeptuneKeyPath, Path]]]
		else:
			_context_figures_dict = None

		_context_found_figure_keys_dict: Dict[IdentifyingContext, Dict[RunID, List[NeptuneKeyPath]]] = {}


		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			if should_download_figures:
				_context_figures_dict[a_ctxt] = {}
			_context_found_figure_keys_dict[a_ctxt] = {}
			a_session_descriptor_str: str = a_ctxt.get_description(separator='_', subset_excludelist='format_name') # 'kdiba_gor01_two_2006-6-07_16-40-19'

			for a_run in a_run_list:
				try:
					a_run_id: str = a_run['sys/id'].fetch()
					_context_figures_dict[a_ctxt][a_run_id] = {}
					
					a_parsed_structure = a_run.get_structure().get('outputs', {}).get('figures', None)
					if a_parsed_structure is None:
						raise ValueError(f'No "outputs/values" in this run.') #skip this one
					assert isinstance(a_parsed_structure, dict), f"type(a_parsed_structure): {type(a_parsed_structure)} instead of dict. a_parsed_structure: {a_parsed_structure}"
					if debug_print:
						print(f'a_parsed_structure: {a_parsed_structure}')
					a_flat_parsed_structure = flatten_dict(a_parsed_structure)
					all_figure_name_keys: List[str] = list(a_flat_parsed_structure.keys())
					_context_found_figure_keys_dict[a_ctxt][a_run_id] = all_figure_name_keys

					if should_download_figures:
						for fig_input_key, a_fig_file in a_flat_parsed_structure.items():
							## check if the fig key matches the user's pattern:
							if fnmatch.fnmatch(fig_input_key, fig_input_key_match_pattern):
								if debug_print:
									print(f'downloading "{fig_input_key}"...')
								## actually perform the download:
								_context_figures_dict[a_ctxt][a_run_id][fig_input_key] = a_run.download_image(fig_input_key=fig_input_key, a_session_descriptor_str=a_session_descriptor_str, neptune_project_figures_output_path=neptune_project_figures_output_path, debug_print=debug_print)
								

				except (ValueError, KeyError, MissingFieldException) as e:
					if debug_print:
						print(f'failed to get figures for run: {a_run} - error: {e}')
					continue # just try the next one
				except Exception as e:
					raise e
			# END FOR a_run
		# END FOR a_ctxt

		## OUTPUTS: _context_found_figure_keys_dict, _context_figures_dict
				


		# _context_figures_dict = self.main_run_collected_results.download_uploaded_figure_files(neptune_project_figures_output_path=self.figures_output_path, fig_input_key=fig_input_key, debug_print=False)
		# # _context_figures_dict
		# ## INPUTS: _context_figures_dict
		# _flattened_context_path_dict, _flat_out_path_items = flatten_context_nested_dict(_context_figures_dict)
		# # OUTPUTS: _flattened_context_path_dict
		# return _flattened_context_path_dict

		return _context_found_figure_keys_dict, _context_figures_dict
	


	def try_get_run_logs(self):
		""" tries to get the log files, downloading them as needed """
		_context_log_files_dict = self.utility_run_collected_results.download_uploaded_log_files(neptune_logs_output_path=self.logs_output_path)
		context_indexed_run_logs: Dict[IdentifyingContext, str] = self.utility_run_collected_results.context_indexed_run_logs # get the IdentifyingContext indexed item
		
		self.logs_output_path.mkdir(exist_ok=True, parents=True)
		_out_log_paths = self.utility_run_collected_results._perform_export_log_files_to_separate_files(context_indexed_run_logs=context_indexed_run_logs, neptune_logs_output_path=self.logs_output_path)
		_out_log_paths

		return _context_log_files_dict, context_indexed_run_logs, _out_log_paths


	def save_run_history_CSVs(self, TODAY_DAY_DATE: str):
		# a_run_collected_results: NeptuneRunCollectedResults = self.utility_run_collected_results
		csv_output_paths = {}
		most_recent_runs_table_df_dict = {}
		
		for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items():
			context_indexed_runs_list_dict: Dict[IdentifyingContext, List[AutoValueConvertingNeptuneRun]] = a_run_collected_results.context_indexed_runs_list_dict
			## INPUTS: neptuner, run_logs, most_recent_runs_table_df

			most_recent_runs_table_df: pd.DataFrame = a_run_collected_results.most_recent_runs_table_df
			most_recent_runs_session_descriptor_string_to_context_map: Dict[SessionDescriptorString, IdentifyingContext] = a_run_collected_results.most_recent_runs_session_descriptor_string_to_context_map
			## INPUTS: most_recent_runs_session_descriptor_string_to_context_map, run_logs
			# context_indexed_run_logs: Dict[IdentifyingContext, str] = a_run_collected_results.context_indexed_run_logs # get the IdentifyingContext indexed item
			## INPUTS: most_recent_runs_table_df
			most_recent_runs_context_indexed_run_extra_data: Dict[IdentifyingContext, Dict] = a_run_collected_results.most_recent_runs_context_indexed_run_extra_data
			# most_recent_runs_context_indexed_run_extra_data # SessionTuple(format_name='kdiba', animal='pin01', exper_name='one', session_name='11-02_17-46-44', session_descriptor_string='kdiba_pin01_one_11-02_17-46-44_sess', id='LS2023-1335', hostname='gl3126.arc-ts.umich.edu', creation_time=Timestamp('2024-08-29 16:39:16.613000'), running_time=8735.629, ping_time=Timestamp('2024-09-24 08:38:06.626000'), monitoring_time=1543, size=28686905.0, tags='11-02_17-46-44,one,kdiba,pin01', entrypoint='figures_kdiba_pin01_one_11-02_17-46-44.py')
			output_directory = Path('output').resolve()
			output_directory.mkdir(parents=False, exist_ok=True)
			csv_output_paths[a_name] = output_directory.joinpath(f'{TODAY_DAY_DATE}_{a_name}_most_recent_neptune_runs_csv.csv')
			most_recent_runs_table_df.to_csv(csv_output_paths[a_name])
			most_recent_runs_table_df_dict[a_name] = most_recent_runs_table_df_dict
			## OUTPUTS: most_recent_runs_session_descriptor_string_to_context_map, context_indexed_run_logs, most_recent_runs_context_indexed_run_extra_data

		return most_recent_runs_table_df_dict, csv_output_paths


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
	'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ==",
	"monitoring_namespace":"monitoring",
	}

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