import sys
import os
import pkg_resources # for Slurm templating
from jinja2 import Environment, FileSystemLoader # for Slurm templating
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
# import numpy as np
# import pandas as pd
# from attrs import define, field, Factory
# from copy import deepcopy
# import multiprocessing

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path, set_posix_windows, convert_filelist_to_new_parent, find_matching_parent_path
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
from pyphocorehelpers.function_helpers import function_attributes

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
# from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths




@function_attributes(short_name=None, tags=['slurm','jobs','files','batch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-09 19:14', related_items=[])
def generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs: Dict[IdentifyingContext, Path], included_session_contexts: List[IdentifyingContext], output_directory, use_separate_run_directories:bool=True, create_slurm_scripts:bool=False):
	""" Creates a series of standalone scripts (one for each included_session_contexts) in the `output_directory`

	output_directory
	use_separate_run_directories:bool = True - If True, separate directories are made in `output_directory` containing each script for all sessions.

	included_session_contexts
	session_batch_basedirs: Dict[IdentifyingContext, Path]
	
	Usage:
	
		from pyphoplacecellanalysis.General.Batch.PythonScriptTemplating import generate_batch_single_session_scripts
		
		## Build Slurm Scripts:
		included_session_contexts, output_python_scripts, output_slurm_scripts = generate_batch_single_session_scripts(global_data_root_parent_path, included_session_contexts, Path('output/generated_slurm_scripts/').resolve(), use_separate_run_directories=True)

		
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
		curr_session_basedir = session_batch_basedirs[curr_session_context]
		if use_separate_run_directories:
			curr_batch_script_rundir = os.path.join(output_directory, f"run_{curr_session_context}")
			os.makedirs(curr_batch_script_rundir, exist_ok=True)
		else:
			curr_batch_script_rundir = output_directory

		# Create the Python script
		python_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.py')
		with open(python_script_path, 'w') as script_file:
			script_content = python_template.render(global_data_root_parent_path=global_data_root_parent_path,
													curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
													curr_session_basedir=curr_session_basedir)
			script_file.write(script_content)
		output_python_scripts.append(python_script_path)


		# Create the SLURM script
		if create_slurm_scripts:
			slurm_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.sh')
			with open(slurm_script_path, 'w') as script_file:
				script_content = slurm_template.render(curr_session_context=f"{curr_session_context}", python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir)
				script_file.write(script_content)

			# Add the output files:
			output_slurm_scripts.append(slurm_script_path)
	
	return included_session_contexts, output_python_scripts, output_slurm_scripts
