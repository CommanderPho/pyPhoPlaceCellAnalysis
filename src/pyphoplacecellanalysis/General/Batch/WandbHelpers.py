"""Weights & Biases (CoreWeave Forge) helpers for batch logging, figure upload, and run fetching.

Auth via WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT
(see https://docs.coreweave.com/products/wandb).
"""
from copy import deepcopy
import fnmatch
import json
import os
import warnings
from attrs import define, Factory, field
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from typing import NewType
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types

import wandb
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

SessionDescriptorString = NewType('SessionDescriptorString', str)
RunID = NewType('RunID', str)
WandbKeyPath = NewType('WandbKeyPath', str)
WandbProjectName = NewType('WandbProjectName', str)


def wandb_stringify_unsupported(value: Any) -> Any:
	"""JSON-safe representation for values W&B config/summary cannot store natively."""
	if value is None or isinstance(value, (bool, int, float, str)):
		return value
	if isinstance(value, Path):
		return value.as_posix()
	if isinstance(value, IdentifyingContext):
		return value.get_description()
	if isinstance(value, (list, tuple)):
		return [wandb_stringify_unsupported(v) for v in value]
	if isinstance(value, dict):
		return {str(k): wandb_stringify_unsupported(v) for k, v in value.items()}
	try:
		json.dumps(value)
		return value
	except (TypeError, ValueError):
		return repr(value)


def set_wandb_environment_variables(wandb_kwargs=None, enable_wandb=True):
	"""Sets WANDB_* environment variables from wandb_kwargs when provided.

	from pyphoplacecellanalysis.General.Batch.WandbHelpers import set_wandb_environment_variables

	"""
	if enable_wandb:
		if wandb_kwargs is None:
			wandb_kwargs = KnownWandbProjects.get_PhoDibaBatch_wandb_kwargs()
		api_key = wandb_kwargs.get('api_key') or os.environ.get('WANDB_API_KEY')
		entity = wandb_kwargs.get('entity') or os.environ.get('WANDB_ENTITY')
		project = wandb_kwargs.get('project') or os.environ.get('WANDB_PROJECT')
		if api_key:
			os.environ["WANDB_API_KEY"] = api_key
		if entity:
			os.environ["WANDB_ENTITY"] = entity
		if project:
			os.environ["WANDB_PROJECT"] = project


class KnownWandbProjects:
	"""Known W&B project settings (entity/project from env; single consolidated project).

	from pyphoplacecellanalysis.General.Batch.WandbHelpers import KnownWandbProjects

	wandb_kwargs = KnownWandbProjects.get_PhoDibaBatch_wandb_kwargs()

	"""
	DEFAULT_PROJECT: str = "PhoDibaBatch"


	@staticmethod
	def get_PhoDibaBatch_wandb_kwargs():
		"""Consolidated batch + figures project."""
		return dict(
			entity=os.environ.get("WANDB_ENTITY"),
			project=os.environ.get("WANDB_PROJECT", KnownWandbProjects.DEFAULT_PROJECT),
			api_key=os.environ.get("WANDB_API_KEY"),
		)


	@staticmethod
	def get_PhoDibaBatchProcessing_wandb_kwargs():
		"""Alias for batch/logs job_type within the consolidated project."""
		return KnownWandbProjects.get_PhoDibaBatch_wandb_kwargs()


	@staticmethod
	def get_PhoDibaLongShortUpdated_wandb_kwargs():
		"""Alias for figures job_type within the consolidated project."""
		return KnownWandbProjects.get_PhoDibaBatch_wandb_kwargs()


	@classmethod
	def get_PhoDibaBatch_wandb_logger(cls) -> "WandbLogger":
		wandb_kwargs = cls.get_PhoDibaBatch_wandb_kwargs()
		return WandbLogger(entity=wandb_kwargs.get('entity'), project=wandb_kwargs.get('project'))


	@classmethod
	def get_PhoDibaBatchProcessing_wandb_logger(cls) -> "WandbLogger":
		return cls.get_PhoDibaBatch_wandb_logger()


	@classmethod
	def get_PhoDibaLongShortUpdated_wandb_logger(cls) -> "WandbLogger":
		return cls.get_PhoDibaBatch_wandb_logger()


# ==================================================================================================================== #
# UTILITY FUNCTIONS                                                                                                    #
# ==================================================================================================================== #


def capture_print_output(func):
	"""Captures the output of the provided function."""
	captured_output = io.StringIO()
	with redirect_stdout(captured_output):
		func()
		return captured_output.getvalue()


def strip_ansi_escape_sequences(text: str) -> str:
	"""Removes ANSI escape sequences from the provided text."""
	ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
	return ansi_escape.sub('', text)


def clean_quotes(text: str) -> str:
	"""Cleans surrounding quotes from the keys and values."""
	return text.strip("'\"")


def flatten_context_nested_dict(_context_figures_dict):
	"""Flatten nested context->run->figure path dicts.

	from pyphoplacecellanalysis.General.Batch.WandbHelpers import flatten_context_nested_dict
	"""
	_flattened_context_path_dict = {}
	_flat_out_path_items = []
	for k, v in _context_figures_dict.items():
		_flattened_context_path_dict[k] = list(flatten_dict(v).values())
		if len(_flattened_context_path_dict[k]) > 0:
			_flat_out_path_items.append(_flattened_context_path_dict[k][0])
	## END for k, v in _context_figures_dict.items()...

	return _flattened_context_path_dict, _flat_out_path_items


def _wandb_safe_key(key: str) -> str:
	"""Normalize metric keys for W&B logging."""
	return str(key).replace('\\', '/')


@function_attributes(short_name=None, tags=['vscode_workspace', 'vscode', 'logs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-29 07:49', related_items=[])
def build_vscode_log_browsing_workspace(logs_paths):
	"""Builds a VSCode workspace for browsing downloaded batch log files.

	Usage:
		from pyphoplacecellanalysis.General.Batch.WandbHelpers import build_vscode_log_browsing_workspace

		path_to_watch = "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/wandb/logs"
		log_files = [v.as_posix() for v in merged_log_files_paths]
		vscode_workspace_path = build_vscode_log_browsing_workspace(logs_paths=log_files)

	"""
	import sys
	import os
	import platform
	from jinja2 import Environment, FileSystemLoader, Template
	from pyphoplacecellanalysis.Resources import templates_directory

	is_platform_windows: bool = False
	if (platform.system() == 'Windows'):
		is_platform_windows = True
	else:
		is_platform_windows = False

	assert len(logs_paths) > 0, f"logs_paths is empty!"
	top_level_merged_logs_folders_path: Path = Path(logs_paths[0]).resolve().parent # e.g. data/wandb/logs/merged
	print(f'top_level_merged_logs_folders_path: {top_level_merged_logs_folders_path}')
	top_level_logs_folders_path: Path = Path(logs_paths[0]).resolve().parent.parent # e.g. data/wandb/logs
	print(f'top_level_logs_folders_path: {top_level_logs_folders_path}')
	top_level_wandb_folders_path: Path = top_level_logs_folders_path.parent # e.g. data/wandb
	print(f'top_level_wandb_folders_path: {top_level_wandb_folders_path}')

	workspace_relative_log_folders: List[Path] = [Path(a_path).relative_to(top_level_wandb_folders_path).resolve() for a_path in [top_level_merged_logs_folders_path, top_level_logs_folders_path]]
	print(f'workspace_relative_log_folders: {workspace_relative_log_folders}')

	relative_log_files: List[Path] = [Path(a_path).relative_to(top_level_wandb_folders_path).resolve() for a_path in logs_paths]
	print(f'relative_log_files: {relative_log_files}')

	vscode_workspace_path = top_level_wandb_folders_path.joinpath('logs_workspace.code-workspace').resolve()
	print(f'vscode_workspace_path: {vscode_workspace_path}')

	template_path = templates_directory()
	env = Environment(loader=FileSystemLoader(template_path))
	template = env.get_template('vscode_logs_workspace_template.code-workspace.j2')

	folders = [
		{'path': f'{a_folder.as_posix()}', 'name': f'{a_folder.name}'}
		for a_folder in workspace_relative_log_folders
	]
	## END for a_folder in workspace_relative_log_folders...

	variables = {
		'folders': folders,
		'is_platform_windows': is_platform_windows,
	}

	workspace_file_content = template.render(variables)

	with open(vscode_workspace_path, 'w') as f:
		f.write(workspace_file_content)

	return vscode_workspace_path


# ==================================================================================================================== #
# END UTILITY FUNCTIONS                                                                                                #
# ==================================================================================================================== #


class _WandbUploadProxy:
	"""Leaf proxy supporting `.upload(path)` used by templates and PlotlyHelpers."""

	def __init__(self, helper: "WandbRunHelper", key_path: str):
		self._helper = helper
		self._key_path = key_path


	def upload(self, path_or_obj):
		self._helper.upload_file(self._key_path, path_or_obj)


class _WandbNestedProxy:
	"""Nested dict-style proxy: run['outputs']['figures'][key].upload(path)."""

	def __init__(self, helper: "WandbRunHelper", prefix: str = ""):
		self._helper = helper
		self._prefix = prefix


	def __getitem__(self, key):
		new_prefix = f"{self._prefix}/{key}" if self._prefix else str(key)
		return _WandbNestedProxy(self._helper, new_prefix)


	def __setitem__(self, key, value):
		full_key = f"{self._prefix}/{key}" if self._prefix else str(key)
		self._helper[full_key] = value


	def upload(self, path_or_obj):
		self._helper.upload_file(self._prefix, path_or_obj)


class WandbRunHelper:
	"""Wraps an active wandb.Run with nested assignment and upload helpers.

	Usage:
		helper = WandbRunHelper(run)
		helper['parameters/script_type'] = 'run'
		helper['outputs']['figures'][fig_key].upload(path)
		helper.sync()

	"""

	def __init__(self, run: Optional[Any] = None):
		self.run = run
		self._figures_logged: Dict[str, Any] = {}


	def __getitem__(self, key):
		return _WandbNestedProxy(self, str(key))


	def __setitem__(self, key, value):
		if self.run is None:
			return
		key_str = _wandb_safe_key(key)
		converted = self._convert_value(value)
		if key_str.startswith('parameters/') or key_str == 'parameters':
			config_key = key_str[len('parameters/'):] if key_str.startswith('parameters/') else key_str
			if isinstance(converted, dict) and key_str == 'parameters':
				self.run.config.update(converted, allow_val_change=True)
			else:
				self.run.config[_wandb_safe_key(config_key)] = converted
		elif key_str.startswith('outputs/') or key_str == 'outputs':
			summary_key = key_str[len('outputs/'):] if key_str.startswith('outputs/') else key_str
			if isinstance(converted, dict) and key_str == 'outputs':
				for sk, sv in converted.items():
					self.run.summary[_wandb_safe_key(str(sk))] = sv
				## END for sk, sv in converted.items()...
			else:
				self.run.summary[_wandb_safe_key(summary_key)] = converted
		else:
			# session context fields and free-form keys → config
			self.run.config[key_str] = converted


	@staticmethod
	def _convert_value(value: Any) -> Any:
		if isinstance(value, pd.DataFrame):
			return wandb.Table(dataframe=value)
		if isinstance(value, (pathlib.PurePath, pathlib.Path)):
			return value.as_posix()
		if isinstance(value, IdentifyingContext):
			return value.get_description()
		return wandb_stringify_unsupported(value)


	def upload_file(self, key_path: str, path_or_obj):
		"""Upload a file path (or image-like object) under a logical key."""
		if self.run is None:
			return
		key_str = _wandb_safe_key(key_path)
		path = Path(path_or_obj) if not isinstance(path_or_obj, Path) and isinstance(path_or_obj, (str, pathlib.PurePath)) else path_or_obj
		if isinstance(path, Path) and path.exists():
			suffix = path.suffix.lower()
			log_key = key_str.replace('/', '__')
			if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'}:
				self.run.log({log_key: wandb.Image(str(path))})
				self._figures_logged[key_str] = str(path)
			else:
				artifact = wandb.Artifact(name=f"file-{log_key}".replace(':', '-').replace(' ', '_')[:128], type="output")
				artifact.add_file(str(path))
				self.run.log_artifact(artifact)
				self.run.summary[log_key] = path.as_posix()
		else:
			self.run.summary[key_str.replace('/', '__')] = wandb_stringify_unsupported(path_or_obj)


	def sync(self):
		"""No-op flush compatibility with historical run.sync() call sites."""
		return None


	def stop(self):
		if self.run is not None:
			self.run.finish()
			self.run = None


	def get_url(self) -> Optional[str]:
		if self.run is None:
			return None
		return getattr(self.run, 'url', None)


@define(slots=False, repr=False)
class WandbRunCollectedResults:
	"""Collected W&B runs indexed by session context (fetch / download helpers).

	from pyphoplacecellanalysis.General.Batch.WandbHelpers import WandbRunCollectedResults

	"""
	runs_table_df: pd.DataFrame = field()
	most_recent_runs_table_df: pd.DataFrame = field()
	runs_dict: Dict[RunID, Any] = field()

	most_recent_runs_session_descriptor_string_to_context_map: Dict[SessionDescriptorString, IdentifyingContext] = field()
	session_descriptor_indexed_runs_list_dict: Dict[SessionDescriptorString, List[Any]] = field()
	context_indexed_runs_list_dict: Dict[IdentifyingContext, List[Any]] = field()

	context_indexed_run_logs: Dict[IdentifyingContext, str] = field()
	most_recent_runs_context_indexed_run_extra_data: Dict[IdentifyingContext, Dict] = field()


	def get_resolved_structure(self, fig_input_key: str = "/", debug_print=False) -> Dict[IdentifyingContext, Dict[RunID, Dict[WandbKeyPath, Any]]]:
		"""Return config/summary structure per run."""
		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[Any]] = self.context_indexed_runs_list_dict
		_context_run_structure_dict: Dict[IdentifyingContext, Dict[RunID, Dict[WandbKeyPath, Any]]] = {}

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			_context_run_structure_dict[a_ctxt] = {}
			for a_run in a_run_list:
				try:
					a_run_id: str = a_run.id if hasattr(a_run, 'id') else str(a_run)
					structure = {
						'config': dict(getattr(a_run, 'config', {}) or {}),
						'summary': dict(getattr(a_run, 'summary', {}) or {}),
					}
					_context_run_structure_dict[a_ctxt][a_run_id] = structure
				except (ValueError, KeyError, AttributeError) as e:
					continue
				except Exception as e:
					raise e
			## END for a_run in a_run_list...
		## END for a_ctxt, a_run_list in context_indexed_runs_list_dict.items()...

		return _context_run_structure_dict


	@classmethod
	def _perform_export_log_files_to_separate_files(cls, context_indexed_run_logs: Dict[IdentifyingContext, str], wandb_logs_output_path: Path):
		Assert.path_exists(wandb_logs_output_path)
		_out_log_paths = {}
		for k, v in context_indexed_run_logs.items():
			session_context_path_fragment: str = k.get_description(separator='=', subset_excludelist='format_name')
			session_context_path = wandb_logs_output_path.joinpath(f"{session_context_path_fragment}.log").resolve()
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
		## END for k, v in context_indexed_run_logs.items()...
		return _out_log_paths


	def download_uploaded_log_files(self, wandb_logs_output_path: Path):
		"""Download log artifacts / summary log paths for collected runs."""
		Assert.path_exists(wandb_logs_output_path)
		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[Any]] = self.context_indexed_runs_list_dict
		_context_log_files_dict = {}

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			_context_log_files_dict[a_ctxt] = {}
			for a_run in a_run_list:
				a_run_id: str = getattr(a_run, 'id', str(a_run))
				try:
					config = getattr(a_run, 'config', {}) or {}
					a_script_type: str = config.get('script_type', config.get('parameters/script_type', 'run'))
					created_at = getattr(a_run, 'created_at', None)
					a_formatted_modification_time: str = datetime.now().strftime('%Y%m%dT%H%M%S')
					if created_at is not None:
						try:
							a_formatted_modification_time = pd.Timestamp(created_at).strftime('%Y%m%dT%H%M%S')
						except Exception:
							pass
					a_run_ctxt = a_ctxt.adding_context_if_missing(script_type=a_script_type)
					a_log_file_basename: str = a_run_ctxt.get_description(separator='--')
					a_log_file_filename: str = f"{a_formatted_modification_time}--{a_run_id}--" + a_log_file_basename + '.log'
					a_log_file_dest_path = wandb_logs_output_path.joinpath(a_log_file_filename).resolve()

					# Prefer artifact named like batch log; fall back to summary path string
					downloaded = False
					try:
						for art in a_run.logged_artifacts():
							if 'log' in art.name.lower() or art.type in {'output', 'log'}:
								art.download(root=str(wandb_logs_output_path))
								_context_log_files_dict[a_ctxt][a_run_id] = wandb_logs_output_path.as_posix()
								downloaded = True
								break
						## END for art in a_run.logged_artifacts()...
					except Exception:
						pass
					if not downloaded:
						summary = getattr(a_run, 'summary', {}) or {}
						log_path = summary.get('log') or summary.get('outputs__log') or config.get('final_log_file_path')
						if log_path and Path(str(log_path)).exists():
							import shutil
							shutil.copy2(str(log_path), a_log_file_dest_path)
							_context_log_files_dict[a_ctxt][a_run_id] = a_log_file_dest_path.as_posix()
				except Exception as err:
					print(f'Failed downloading log for run {a_run_id}: {err}')
			## END for a_run in a_run_list...
		## END for a_ctxt, a_run_list in context_indexed_runs_list_dict.items()...

		return _context_log_files_dict


	def download_uploaded_figure_files(self, wandb_project_figures_output_path: Path, fig_input_key: str = "display_fn_name:running_and_replay_speeds_over_time", debug_print=False):
		"""Best-effort figure download from run files / media."""
		Assert.path_exists(wandb_project_figures_output_path)
		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[Any]] = self.context_indexed_runs_list_dict
		_context_figures_dict = {}

		for a_ctxt, a_run_list in context_indexed_runs_list_dict.items():
			_context_figures_dict[a_ctxt] = {}
			a_session_descriptor_str: str = a_ctxt.get_description(separator='_', subset_excludelist='format_name')
			for a_run in a_run_list:
				a_run_id: str = getattr(a_run, 'id', str(a_run))
				try:
					a_session_figures_output_path = wandb_project_figures_output_path.joinpath(a_session_descriptor_str)
					a_session_figures_output_path.mkdir(exist_ok=True)
					# Download media files matching fig_input_key if present in run files
					try:
						files = list(a_run.files())
						matched = [f for f in files if fig_input_key.replace('/', '_') in f.name or fig_input_key in f.name]
						for f in matched:
							dest = a_session_figures_output_path.joinpath(f.name).resolve()
							f.download(root=str(a_session_figures_output_path), replace=True)
							_context_figures_dict[a_ctxt].setdefault(a_run_id, {})[fig_input_key] = dest.as_posix()
						## END for f in matched...
					except Exception as e:
						if debug_print:
							print(f'figure download skipped for {a_run_id}: {e}')
				except Exception as e:
					if debug_print:
						print(f'Failed figures for run {a_run_id}: {e}')
			## END for a_run in a_run_list...
		## END for a_ctxt, a_run_list in context_indexed_runs_list_dict.items()...

		return _context_figures_dict


@define(slots=False, repr=False, eq=False)
class WandbLogger(object):
	"""Maintains W&B run state for batch scripts and figure uploads.

	from pyphoplacecellanalysis.General.Batch.WandbHelpers import WandbLogger
	wandb_logger = WandbLogger()

	Mostly initialized from `batch_perform_all_plots` / batch Jinja templates.

	"""
	entity: Optional[str] = field(default=None)
	project: Optional[str] = field(default=Factory(lambda: os.environ.get("WANDB_PROJECT", KnownWandbProjects.DEFAULT_PROJECT)))
	run: Optional[WandbRunHelper] = field(init=False, default=None)

	outputs = field(init=False, default=None)
	figures = field(init=False, default=None)


	def __attrs_post_init__(self):
		if self.entity is None:
			self.entity = os.environ.get("WANDB_ENTITY")
		if self.project is None:
			self.project = os.environ.get("WANDB_PROJECT", KnownWandbProjects.DEFAULT_PROJECT)
		self.run = None
		self.outputs = None
		self.figures = None


	def stop(self):
		print(f'WandbLogger.stop() called!')
		if self.run is not None:
			self.run.stop()
		self.run = None
		print(f'WandbLogger stopped.')


	def finish(self):
		"""Alias for stop() used by generated batch scripts."""
		self.stop()


	def run_with_pipeline(self, curr_active_pipeline, job_type: str = "figures"):
		"""Starts a new run with the provided pipeline."""
		assert self.run is None, f"run_with_pipeline(...) called while the WandbLogger object already has an active self.run object!"
		session_ctx = curr_active_pipeline.get_session_context()
		tags = list(session_ctx.as_tuple()) + [job_type]
		init_kwargs = dict(project=self.project, job_type=job_type, tags=tags, reinit=True)
		if self.entity:
			init_kwargs['entity'] = self.entity
		wb_run = wandb.init(**init_kwargs)
		helper = WandbRunHelper(wb_run)
		for k, v in session_ctx.to_dict().items():
			helper[k] = v
		## END for k, v in session_ctx.to_dict().items()...
		helper['session_descriptor_string'] = curr_active_pipeline.sess.get_description()
		helper['parameters/script_type'] = job_type
		self.run = helper
		self.outputs = helper['outputs']
		self.figures = helper['outputs']['figures']


	@classmethod
	def init_with_pipeline(cls, curr_active_pipeline, job_type: str = "figures"):
		"""Creates a new WandbLogger for a session and starts a run."""
		new = cls()
		new.run_with_pipeline(curr_active_pipeline=curr_active_pipeline, job_type=job_type)
		return new


	@classmethod
	def init_for_batch_script(cls, curr_session_context, script_type: str, job_suffix: str = "", script_file_path: Optional[str] = None, curr_session_complete_identifier: Optional[str] = None, session_ctxt_key: Optional[str] = None, **extra_config):
		"""Initialize logging for a generated batch script (Jinja template)."""
		wandb_kwargs = KnownWandbProjects.get_PhoDibaBatch_wandb_kwargs()
		new = cls(entity=wandb_kwargs.get('entity'), project=wandb_kwargs.get('project'))
		tags = list(curr_session_context.as_tuple()) + [script_type]
		if job_suffix:
			tags.append(job_suffix)
		group = session_ctxt_key or curr_session_context.get_description(separator='|', subset_includelist=['animal', 'exper_name', 'session_name'])
		init_kwargs = dict(project=new.project, job_type=script_type, tags=tags, group=group, reinit=True)
		if new.entity:
			init_kwargs['entity'] = new.entity
		if script_file_path:
			init_kwargs['settings'] = wandb.Settings(code_dir=str(Path(script_file_path).parent))
		wb_run = wandb.init(**init_kwargs)
		helper = WandbRunHelper(wb_run)
		params = dict(script_type=script_type, **extra_config)
		if script_file_path:
			params['script_file_path'] = script_file_path
			params['script_file_parent_path'] = Path(script_file_path).parent.as_posix()
		params['curr_session_context'] = str(curr_session_context.get_description())
		helper['parameters'] = params
		for k, v in curr_session_context.to_dict().items():
			helper[k] = v
		## END for k, v in curr_session_context.to_dict().items()...
		helper['session_descriptor_string'] = curr_session_context.get_description()
		if curr_session_complete_identifier:
			helper['session_complete_descriptor_string'] = curr_session_complete_identifier
		new.run = helper
		new.outputs = helper['outputs']
		new.figures = helper['outputs']['figures']
		return new


	def upload_figures(self, curr_active_pipeline, should_use_fig_objects=False, debug_print=False):
		"""Upload registered pipeline figure files to the active W&B run."""
		assert not should_use_fig_objects, f"should_use_fig_objects = True is not currently working. Save the figures to file first and then use should_use_fig_objects = False"
		assert self.run is not None, "No active run; call run_with_pipeline or init_for_batch_script first."
		sess_context = curr_active_pipeline.get_session_context()
		succeeded_fig_paths = []
		failed_fig_paths = []

		for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items():
			a_fig_context: IdentifyingContext = fig_metadata['context']
			a_fig_specific_context: IdentifyingContext = a_fig_context.subtracting(sess_context)
			a_full_figure_path_key: str = a_fig_specific_context.get_description(separator='/', include_property_names=True, key_value_separator=':')
			if debug_print:
				print(f'Processing: {a_fig_specific_context}, full_key: {a_full_figure_path_key}')

			fig_objects = fig_metadata.get('fig', None)
			if (fig_objects is not None) and should_use_fig_objects:
				if debug_print:
					print(f'\tSkipping "{a_full_figure_path_key}": fig object upload not supported')
				failed_fig_paths.append(a_fig_path)
			else:
				if should_use_fig_objects:
					if debug_print:
						print(f'\tSkipping "{a_full_figure_path_key}": No figure objects found')
					failed_fig_paths.append(a_fig_path)
				else:
					try:
						self.run.upload_file(f"outputs/figures/{a_full_figure_path_key}", Path(a_fig_path))
						succeeded_fig_paths.append(a_fig_path)
						if debug_print:
							print(f'\tSuccess: "{a_full_figure_path_key}": file at path {str(a_fig_path)} uploaded!')
					except Exception as e:
						raise e
		## END for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items()...
		return succeeded_fig_paths, failed_fig_paths


	def get_runs_table(self) -> pd.DataFrame:
		"""Fetch project runs as a DataFrame via wandb.Api."""
		api = wandb.Api()
		path = f"{self.entity}/{self.project}" if self.entity else self.project
		runs = api.runs(path)
		rows = []
		for r in runs:
			cfg = dict(r.config or {})
			row = {
				'id': r.id,
				'name': r.name,
				'state': r.state,
				'created_at': getattr(r, 'created_at', None),
				'hostname': cfg.get('hostname'),
				'tags': ','.join(r.tags or []),
				'entrypoint': cfg.get('script_file_path') or cfg.get('parameters/script_file_path'),
				'session_descriptor_string': cfg.get('session_descriptor_string'),
				'format_name': cfg.get('format_name'),
				'animal': cfg.get('animal'),
				'exper_name': cfg.get('exper_name'),
				'session_name': cfg.get('session_name'),
				'script_type': cfg.get('script_type') or cfg.get('parameters/script_type'),
			}
			rows.append(row)
		## END for r in runs...
		return pd.DataFrame(rows)


	@classmethod
	def get_most_recent_session_runs_table(cls, runs_table_df: pd.DataFrame, ordering_datetime_col_name: str = 'created_at', oldest_included_run_date: str = '2024-08-01', n_recent_results: int = 1) -> pd.DataFrame:
		assert isinstance(oldest_included_run_date, str), f"oldest_included_run_date should be of type str (provided like '2024-08-01') but it is of type: {type(oldest_included_run_date)}, oldest_included_run_date: {oldest_included_run_date}"
		active_runs_table_df: pd.DataFrame = deepcopy(runs_table_df)
		if ordering_datetime_col_name not in active_runs_table_df.columns:
			return active_runs_table_df.head(0)
		active_runs_table_df[ordering_datetime_col_name] = pd.to_datetime(active_runs_table_df[ordering_datetime_col_name], utc=True, errors='coerce')
		cutoff = datetime.strptime(f'{oldest_included_run_date}T00:30:00.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')
		cutoff = pd.Timestamp(cutoff, tz='UTC')
		active_runs_table_df = active_runs_table_df[active_runs_table_df[ordering_datetime_col_name] > cutoff]
		if 'session_descriptor_string' not in active_runs_table_df.columns:
			return active_runs_table_df
		most_recent_runs_table_df: pd.DataFrame = (
			active_runs_table_df
			.sort_values([ordering_datetime_col_name], ascending=False)
			.groupby('session_descriptor_string')
			.head(n_recent_results)
			.reset_index(drop=True)
		)
		return most_recent_runs_table_df


	def get_most_recent_session_runs(self, oldest_included_run_date: str = '2024-08-01', n_recent_results: int = 1, **kwargs) -> WandbRunCollectedResults:
		"""Main accessor for recent session runs via wandb.Api."""
		runs_table_df: pd.DataFrame = self.get_runs_table()
		most_recent_runs_table_df: pd.DataFrame = self.get_most_recent_session_runs_table(runs_table_df=runs_table_df, oldest_included_run_date=oldest_included_run_date, n_recent_results=n_recent_results, **kwargs)
		api = wandb.Api()
		path = f"{self.entity}/{self.project}" if self.entity else self.project
		runs_dict: Dict[RunID, Any] = {}
		for run_id in most_recent_runs_table_df.get('id', []):
			try:
				runs_dict[RunID(str(run_id))] = api.run(f"{path}/{run_id}")
			except Exception as e:
				print(f"Failed to fetch run {run_id}: {e}")
		## END for run_id in most_recent_runs_table_df.get('id', [])...

		session_column_individual_variables = ['format_name', 'animal', 'exper_name', 'session_name']
		session_column_variables = ['session_descriptor_string']
		processing_status_column_names = ['id', 'hostname', 'created_at', 'tags', 'entrypoint']

		available_session_cols = [c for c in (session_column_individual_variables + session_column_variables) if c in most_recent_runs_table_df.columns]
		most_recent_runs_session_descriptor_string_to_context_map: Dict[SessionDescriptorString, IdentifyingContext] = {}
		if set(session_column_individual_variables + session_column_variables).issubset(most_recent_runs_table_df.columns):
			most_recent_runs_session_descriptor_string_to_context_map = {
				SessionDescriptorString(v.session_descriptor_string): IdentifyingContext(format_name=v.format_name, animal=v.animal, exper_name=v.exper_name, session_name=v.session_name)
				for v in most_recent_runs_table_df[session_column_individual_variables + session_column_variables].itertuples()
			}
		## END if set(...)...

		session_descriptor_indexed_runs_list_dict: Dict[SessionDescriptorString, List[Any]] = {}
		for run_id, run in runs_dict.items():
			try:
				cfg = dict(getattr(run, 'config', {}) or {})
				a_session_descriptor_string: SessionDescriptorString = SessionDescriptorString(cfg.get('session_descriptor_string', ''))
				if a_session_descriptor_string not in session_descriptor_indexed_runs_list_dict:
					session_descriptor_indexed_runs_list_dict[a_session_descriptor_string] = [run]
				else:
					session_descriptor_indexed_runs_list_dict[a_session_descriptor_string].append(run)
			except Exception as e:
				print(f"Failed to fetch session_descriptor_string for run {run_id}: {e}")
		## END for run_id, run in runs_dict.items()...

		context_indexed_runs_list_dict: Dict[IdentifyingContext, List[Any]] = {
			most_recent_runs_session_descriptor_string_to_context_map[k]: v
			for k, v in session_descriptor_indexed_runs_list_dict.items()
			if k in most_recent_runs_session_descriptor_string_to_context_map
		}

		run_logs: Dict[SessionDescriptorString, str] = self.get_most_recent_session_logs(runs_dict=runs_dict)
		context_indexed_run_logs: Dict[IdentifyingContext, str] = {
			most_recent_runs_session_descriptor_string_to_context_map[k]: v
			for k, v in run_logs.items()
			if k in most_recent_runs_session_descriptor_string_to_context_map
		}

		extra_cols = [c for c in (session_column_individual_variables + session_column_variables + processing_status_column_names) if c in most_recent_runs_table_df.columns]
		most_recent_runs_context_indexed_run_extra_data: Dict[IdentifyingContext, Dict] = {}
		if set(session_column_individual_variables).issubset(most_recent_runs_table_df.columns):
			most_recent_runs_context_indexed_run_extra_data = {
				IdentifyingContext(format_name=v.format_name, animal=v.animal, exper_name=v.exper_name, session_name=v.session_name): v._asdict()
				for v in most_recent_runs_table_df[extra_cols].itertuples(index=False, name='SessionTuple')
			}
		## END if set(...)...

		return WandbRunCollectedResults(
			runs_table_df=runs_table_df,
			most_recent_runs_table_df=most_recent_runs_table_df,
			runs_dict=runs_dict,
			most_recent_runs_session_descriptor_string_to_context_map=most_recent_runs_session_descriptor_string_to_context_map,
			session_descriptor_indexed_runs_list_dict=session_descriptor_indexed_runs_list_dict,
			context_indexed_runs_list_dict=context_indexed_runs_list_dict,
			context_indexed_run_logs=context_indexed_run_logs,
			most_recent_runs_context_indexed_run_extra_data=most_recent_runs_context_indexed_run_extra_data,
		)


	@classmethod
	def get_most_recent_session_logs(cls, runs_dict: Dict[RunID, Any], debug_print: bool = False) -> Dict[SessionDescriptorString, str]:
		_separator_string: str = """\n\n
# ==================================================================================================================== #
# LOG SEPARATOR                                                                                                        #
# ==================================================================================================================== #
"""
		run_logs: Dict[SessionDescriptorString, str] = {}
		for run_id, run in runs_dict.items():
			try:
				cfg = dict(getattr(run, 'config', {}) or {})
				summary = dict(getattr(run, 'summary', {}) or {})
				a_session_descriptor_string: SessionDescriptorString = SessionDescriptorString(cfg.get('session_descriptor_string', str(run_id)))
				log_contents_str = summary.get('log') or summary.get('outputs__log') or ''
				if not log_contents_str:
					log_path = cfg.get('final_log_file_path')
					if log_path and Path(str(log_path)).exists():
						log_contents_str = Path(str(log_path)).read_text(encoding='utf-8', errors='replace')
				if len(str(log_contents_str)) > 0:
					if a_session_descriptor_string not in run_logs:
						run_logs[a_session_descriptor_string] = str(log_contents_str)
					else:
						run_logs[a_session_descriptor_string] = (run_logs[a_session_descriptor_string] + _separator_string + str(log_contents_str))
			except Exception as e:
				print(f"Failed to fetch logs for run {run_id}: {e}")
		## END for run_id, run in runs_dict.items()...
		return run_logs


	@classmethod
	def build_interactive_session_run_logs_widget(cls, context_indexed_run_logs, most_recent_runs_session_descriptor_string_to_context_map, most_recent_runs_context_indexed_run_extra_data):
		"""Creates an interactive session display with a tree widget, session info, and log output."""
		from pandas import Timestamp
		from datetime import datetime
		import ipywidgets as widgets
		from IPython.display import display
		from pyphocorehelpers.gui.Jupyter.TreeWidget import JupyterTreeWidget
		from typing import List, Tuple

		included_session_contexts: List[IdentifyingContext] = list(most_recent_runs_session_descriptor_string_to_context_map.values())
		jupyter_tree_widget = JupyterTreeWidget(included_session_contexts=included_session_contexts,
												on_selection_changed_callbacks=[],
												display_on_init=False)

		jupyter_tree_widget.tree.layout = widgets.Layout(min_width='300px', max_width='30%', overflow='auto', height='auto')

		def build_session_tuple_header_widget(a_session_tuple: Tuple, included_display_session_extra_data_keys: List[str]):
			header_label_widgets = {key: widgets.HTML(f"<b>{key}</b>: '{value}',") for key, value in a_session_tuple.items() if (key in included_display_session_extra_data_keys)}
			box_layout = widgets.Layout(display='flex', flex_flow='row wrap', align_items='stretch', width='100%')
			header_hbox = widgets.Box(list(header_label_widgets.values()), layout=box_layout)

			def update_header_labels_fn(new_values):
				for key, value in new_values.items():
					if key in included_display_session_extra_data_keys:
						if isinstance(value, (Timestamp, datetime)):
							rounded_value = value.floor('T') if isinstance(value, Timestamp) else value.replace(second=0, microsecond=0)
							formatted_value = rounded_value.strftime('%Y-%m-%d %H:%M')
							header_label_widgets[key].value = f"<b>{key}</b>: {formatted_value}"
						else:
							header_label_widgets[key].value = f"<b>{key}</b>: {value}"
				## END for key, value in new_values.items()...

			return header_hbox, header_label_widgets, update_header_labels_fn

		empty_session_tuple = {
			'format_name': '',
			'animal': '',
			'exper_name': '',
			'session_name': '',
			'session_descriptor_string': '',
			'id': '<Selection Not Found>',
			'hostname': '',
			'created_at': '',
			'tags': '',
			'entrypoint': ''
		}

		display_session_extra_data_keys = ['id', 'hostname', 'created_at', 'tags', 'entrypoint']
		header_hbox, header_label_widgets, update_header_labels_fn = build_session_tuple_header_widget(a_session_tuple=empty_session_tuple, included_display_session_extra_data_keys=display_session_extra_data_keys)

		textarea = widgets.Textarea(value='<No Selection>', disabled=True, style={'font_size': '10px'},
									layout=widgets.Layout(flex='1', width='100%', min_height='650px', height='850px'),
									)

		content_view_layout = widgets.VBox([header_hbox, textarea],
											layout=widgets.Layout(min_width='500px', max_width='70%', height='auto', overflow='auto')
										   )

		root_box = widgets.HBox([jupyter_tree_widget.tree, content_view_layout],
								layout=widgets.Layout(
									width='100%',
									display='flex',
									flex_flow='row',
									justify_content='space-between',
									overflow='auto'
								),
							   )

		def _on_tree_node_selection_changed(selected_node, selected_context):
			if isinstance(selected_context, dict):
				selected_context = IdentifyingContext(**selected_context)
			curr_context_extra_data_tuple = most_recent_runs_context_indexed_run_extra_data.get(selected_context, empty_session_tuple)
			update_header_labels_fn(curr_context_extra_data_tuple)
			curr_context_run_log = context_indexed_run_logs.get(selected_context, '<Context Not Found>')
			textarea.value = curr_context_run_log

		jupyter_tree_widget.on_selection_changed_callback = [_on_tree_node_selection_changed]
		return root_box


@define(slots=False, repr=False, eq=False)
class PhoDibaProjectsWandbLogger(object):
	"""Single consolidated W&B project access for figures + logs (job_type/tags distinguish them).

	Usage:
		from pyphoplacecellanalysis.General.Batch.WandbHelpers import PhoDibaProjectsWandbLogger

		combined = PhoDibaProjectsWandbLogger()
		combined.get_most_recent_session_runs(oldest_included_run_date='2024-10-15', n_recent_results=16)

	"""
	main_wandb_project: WandbLogger = field(init=False, metadata={'comment': 'figures + logs (same project)'})
	utility_wandb_project: WandbLogger = field(init=False, metadata={'comment': 'alias of main for API parity'})

	main_run_collected_results: WandbRunCollectedResults = field(init=False)
	utility_run_collected_results: WandbRunCollectedResults = field(init=False)

	root_output_path: Path = field(init=False)
	figures_output_path: Path = field(init=False)
	logs_output_path: Path = field(init=False)


	@property
	def main_run(self):
		return self.main_wandb_project.run


	@property
	def utility_run(self):
		return self.utility_wandb_project.run


	@property
	def projects_wandb_logger_dict(self) -> Dict[WandbProjectName, WandbLogger]:
		return {'main': self.main_wandb_project, 'utility': self.utility_wandb_project}


	@property
	def projects_run_collected_results_dict(self) -> Dict[WandbProjectName, WandbRunCollectedResults]:
		return {'main': self.main_run_collected_results, 'utility': self.utility_run_collected_results}


	@property
	def projects_resolved_run_structure_dict(self) -> Dict[WandbProjectName, Dict[IdentifyingContext, Dict[RunID, Dict]]]:
		run_structure_dict_dict = {}
		for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items():
			run_structure_dict_dict[a_name] = a_run_collected_results.get_resolved_structure()
		## END for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items()...
		return run_structure_dict_dict


	@property
	def context_indexed_run_logs(self) -> Dict[IdentifyingContext, str]:
		return self.utility_run_collected_results.context_indexed_run_logs


	def __attrs_post_init__(self):
		self.main_wandb_project = KnownWandbProjects.get_PhoDibaBatch_wandb_logger()
		self.utility_wandb_project = KnownWandbProjects.get_PhoDibaBatch_wandb_logger()

		self.root_output_path = find_first_extant_path(path_list=[
			Path(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data/wandb").resolve(),
						Path("/home/halechr/repos/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data/wandb").resolve(),
			Path("/home/halechr/repos/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data").resolve(),
			Path('EXTERNAL/PhoDibaPaper2024Book/data/wandb').resolve(),
						Path('/Users/pho/repo/Pho Secondary Workspace/Spike3DEnv/Spike3DWorkEnv/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data').resolve(),
		])
		self.root_output_path.mkdir(exist_ok=True)

		self.figures_output_path = self.root_output_path.joinpath('figs').resolve()
		self.figures_output_path.mkdir(exist_ok=True, parents=True)

		self.logs_output_path = self.root_output_path.joinpath('logs').resolve()
		self.logs_output_path.mkdir(exist_ok=True, parents=True)


	def stop(self):
		for a_logger in (self.main_wandb_project, self.utility_wandb_project):
			a_logger.stop()
		## END for a_logger in (self.main_wandb_project, self.utility_wandb_project)...


	def run_with_pipeline(self, curr_active_pipeline):
		self.main_wandb_project.run_with_pipeline(curr_active_pipeline=curr_active_pipeline, job_type='figures')


	@classmethod
	def init_with_pipeline(cls, curr_active_pipeline):
		new = cls()
		new.run_with_pipeline(curr_active_pipeline=curr_active_pipeline)
		return new


	def get_most_recent_session_runs(self, utility_oldest_included_run_date: Optional[str] = None, utility_n_recent_results: Optional[int] = 2,
									 main_oldest_included_run_date: Optional[str] = None, main_n_recent_results: Optional[int] = None,
									 oldest_included_run_date: str = '2024-08-01', n_recent_results: int = 1, **kwargs):
		utility_oldest_included_run_date = utility_oldest_included_run_date or oldest_included_run_date
		utility_n_recent_results = utility_n_recent_results or n_recent_results
		self.main_run_collected_results = self.main_wandb_project.get_most_recent_session_runs(oldest_included_run_date=utility_oldest_included_run_date, n_recent_results=utility_n_recent_results, **kwargs)

		main_oldest_included_run_date = main_oldest_included_run_date or oldest_included_run_date
		main_n_recent_results = main_n_recent_results or n_recent_results
		self.utility_run_collected_results = self.utility_wandb_project.get_most_recent_session_runs(oldest_included_run_date=main_oldest_included_run_date, n_recent_results=main_n_recent_results, **kwargs)


	def try_get_figures(self, fig_input_key_match_pattern: str = '*', should_download_figures: bool = True, debug_print: bool = True):
		wandb_project_figures_output_path: Path = self.figures_output_path
		Assert.path_exists(wandb_project_figures_output_path)
		_context_figures_dict = self.main_run_collected_results.download_uploaded_figure_files(wandb_project_figures_output_path=wandb_project_figures_output_path, fig_input_key=fig_input_key_match_pattern if fig_input_key_match_pattern != '*' else '', debug_print=debug_print)
		_context_found_figure_keys_dict = {ctxt: {rid: list(figs.keys()) for rid, figs in runs.items()} for ctxt, runs in (_context_figures_dict or {}).items()}
		return _context_found_figure_keys_dict, _context_figures_dict


	def try_get_run_logs(self):
		_context_log_files_dict = self.utility_run_collected_results.download_uploaded_log_files(wandb_logs_output_path=self.logs_output_path)
		context_indexed_run_logs: Dict[IdentifyingContext, str] = self.utility_run_collected_results.context_indexed_run_logs
		self.logs_output_path.mkdir(exist_ok=True, parents=True)
		_out_log_paths = self.utility_run_collected_results._perform_export_log_files_to_separate_files(context_indexed_run_logs=context_indexed_run_logs, wandb_logs_output_path=self.logs_output_path)
		return _context_log_files_dict, context_indexed_run_logs, _out_log_paths


	def save_run_history_CSVs(self, TODAY_DAY_DATE: str):
		csv_output_paths = {}
		most_recent_runs_table_df_dict = {}
		for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items():
			most_recent_runs_table_df: pd.DataFrame = a_run_collected_results.most_recent_runs_table_df
			output_directory = Path('output').resolve()
			output_directory.mkdir(parents=False, exist_ok=True)
			csv_output_paths[a_name] = output_directory.joinpath(f'{TODAY_DAY_DATE}_{a_name}_most_recent_wandb_runs_csv.csv')
			most_recent_runs_table_df.to_csv(csv_output_paths[a_name])
			most_recent_runs_table_df_dict[a_name] = most_recent_runs_table_df
		## END for a_name, a_run_collected_results in self.projects_run_collected_results_dict.items()...
		return most_recent_runs_table_df_dict, csv_output_paths


# ==================================================================================================================== #
# Independent Helper Functions                                                                                         #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['wandb', 'figures', 'output', 'cloud', 'logging', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-04-11 00:00', related_items=[])
def wandb_output_figures(curr_active_pipeline):
	"""Uploads completed figures to W&B from the pipeline's `.registered_output_files` items.

	Usage:
		from pyphoplacecellanalysis.General.Batch.WandbHelpers import set_wandb_environment_variables, wandb_output_figures
		wandb_output_figures(curr_active_pipeline)
	"""
	set_wandb_environment_variables(enable_wandb=True)
	logger = WandbLogger.init_with_pipeline(curr_active_pipeline, job_type='figures')
	try:
		logger.upload_figures(curr_active_pipeline, should_use_fig_objects=False, debug_print=True)
	finally:
		logger.stop()
