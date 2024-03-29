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
def generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs: Dict[IdentifyingContext, Path], included_session_contexts: Optional[List[IdentifyingContext]], output_directory='output/gen_scripts/', use_separate_run_directories:bool=True,
         create_slurm_scripts:bool=False, separate_execute_and_figure_gen_scripts:bool=True, should_perform_figure_generation_to_file:bool=False, force_recompute_override_computations_includelist: Optional[List[str]]=None,
        batch_session_completion_handler_kwargs=None, **renderer_script_generation_kwargs):
    """ Creates a series of standalone scripts (one for each included_session_contexts) in the `output_directory`

    output_directory
    use_separate_run_directories:bool = True - If True, separate directories are made in `output_directory` containing each script for all sessions.

    included_session_contexts
    session_batch_basedirs: Dict[IdentifyingContext, Path]

    batch_session_completion_handler_kwargs: Optional[Dict] - the values to be passed to batch_session_completion_handler
    
    Usage:
    
        from pyphoplacecellanalysis.General.Batch.PythonScriptTemplating import generate_batch_single_session_scripts
        
        ## Build Slurm Scripts:
        included_session_contexts, output_python_scripts, output_slurm_scripts = generate_batch_single_session_scripts(global_data_root_parent_path, included_session_contexts, Path('output/generated_slurm_scripts/').resolve(), use_separate_run_directories=True)

    Usage 2 - 2023-10-24 - Without global_batch

        ## Build Slurm Scripts:
        session_basedirs_dict: Dict[IdentifyingContext, Path] = {a_session_folder.context:a_session_folder.path for a_session_folder in good_session_concrete_folders}
        included_session_contexts, output_python_scripts, output_slurm_scripts = generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs=session_basedirs_dict, included_session_contexts=included_session_contexts, output_directory=Path('output/generated_slurm_scripts/').resolve(), use_separate_run_directories=True)
        output_python_scripts

        
    """
    assert isinstance(session_batch_basedirs, dict)

    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory).resolve()


    separate_execute_and_figure_gen_scripts = renderer_script_generation_kwargs.pop('separate_execute_and_figure_gen_scripts', True)
    assert separate_execute_and_figure_gen_scripts, f"Old non-separate mode not supported"
 
    assert ('force_recompute_override_computations_includelist' not in renderer_script_generation_kwargs), f"pass 'force_recompute_override_computations_includelist' explicitly to the call!!"
    renderer_script_generation_kwargs['force_recompute_override_computations_includelist'] = (force_recompute_override_computations_includelist or [])
 
    # if script_generation_kwargs is None:
    # 	script_generation_kwargs = dict(should_force_reload_all=False, should_perform_figure_generation_to_file=False)

    no_recomputing_script_generation_kwargs = dict(should_force_reload_all=False, should_freeze_pipeline_updates=True, should_perform_figure_generation_to_file=should_perform_figure_generation_to_file) | renderer_script_generation_kwargs # No recomputing at all:
    compute_as_needed_script_generation_kwargs = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, should_perform_figure_generation_to_file=should_perform_figure_generation_to_file) | renderer_script_generation_kwargs
    forced_full_recompute_script_generation_kwargs = dict(should_force_reload_all=True, should_freeze_pipeline_updates=False, should_perform_figure_generation_to_file=should_perform_figure_generation_to_file) | renderer_script_generation_kwargs # Forced Reloading:
    # script_generation_kwargs

    if included_session_contexts is None:
        included_session_contexts = list(session_batch_basedirs.keys())

    # Set up Jinja2 environment
    template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
    env = Environment(loader=FileSystemLoader(template_path))
    python_template = env.get_template('python_template.py.j2')
    # base_python_template = env.get_template('slurm_python_template_base.py.j2')
    # python_template = env.get_template('slurm_python_template_NoRecompute.py.j2', parent='slurm_python_template_base.py.j2')
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

        # Create two separate scripts:
        # Create the Execution Python script
        python_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.py')
        with open(python_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_basedir=curr_session_basedir, 
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    **(compute_as_needed_script_generation_kwargs | dict(should_perform_figure_generation_to_file=False)))
            # script_file.write(script_content)
            script_file.write(script_content.encode())
        # output_python_scripts.append(python_script_path)


        python_figures_script_path = os.path.join(curr_batch_script_rundir, f'figures_{curr_session_context}.py')
        with open(python_figures_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_basedir=curr_session_basedir,
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    **(no_recomputing_script_generation_kwargs | dict(should_perform_figure_generation_to_file=True)))


            # script_file.write(script_content)
            script_file.write(script_content.encode())

        # output_python_display_scripts.append(python_figures_script_path)
        output_python_scripts.append((python_script_path, python_figures_script_path))

        # Create the SLURM script
        if create_slurm_scripts:
            slurm_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.sh')
            with open(slurm_script_path, 'w') as script_file:
                script_content = slurm_template.render(curr_session_context=f"{curr_session_context}", python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir)
                script_file.write(script_content)

            # Add the output files:
            output_slurm_scripts.append(slurm_script_path)
    
    return included_session_contexts, output_python_scripts, output_slurm_scripts


def display_generated_scripts_ipywidget(included_session_contexts, output_python_scripts):
    """ Display an interactive jupyter-widget that allows you to open/reveal the generated files in the fileystem or default system display program. 

    from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import generate_batch_single_session_scripts, display_generated_scripts_ipywidget

    """
    import ipywidgets as widgets
    from IPython.display import display
    from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import build_fn_bound_buttons, JupyterButtonRowWidget, JupyterButtonColumnWidget
    # from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
    # from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default
    from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget       
    
    btn_layout = widgets.Layout(width='auto', height='40px') #set width and height
    default_kwargs = dict(display='flex', flex_flow='column', align_items='stretch', layout=btn_layout)

    # #TODO 2023-12-12 16:43: - [ ] Can potentially replace these complicated definitions with the simplier `fullwidth_path_widget` implementation which contains the two buttons by default
    # # from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget       
    # _out_row = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Documentation Folder", lambda _: reveal_in_system_file_manager(self.doc_output_parent_folder), default_kwargs),
    # 	("Generated Documentation", lambda _: self.reveal_output_files_in_system_file_manager(), default_kwargs),
    # 	])

    # _out_row_html = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Open generated .html Documentation", lambda _: open_file_with_system_default(str(self.output_html_file.resolve())), default_kwargs),
    # 		("Reveal Generated .html Documentation", lambda _: reveal_in_system_file_manager(self.output_html_file), default_kwargs),
    # 	])

    # _out_row_md = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Open generated .md Documentation", lambda _: open_file_with_system_default(str(self.output_md_file.resolve())), default_kwargs),
    # 		("Reveal Generated .md Documentation", lambda _: reveal_in_system_file_manager(self.output_md_file), default_kwargs),
    # 	])


    # computation_script_paths = [x[0] for x in output_python_scripts]
    # generate_figures_script_paths = [x[1] for x in output_python_scripts]

    _out_path_widgets = []
    for a_ctxt, a_python_script in zip(included_session_contexts, output_python_scripts):
        a_computation_script_path = Path(a_python_script[0]).resolve()
        a_generate_figures_script_path = Path(a_python_script[1]).resolve()
    
        _out_path_widgets.append(fullwidth_path_widget(a_computation_script_path, a_computation_script_path.name, box_layout_kwargs=dict(display='flex', flex_flow='row', align_items='stretch', width='auto')))

    return widgets.VBox(_out_path_widgets, display='flex', flex_flow='column', align_items='stretch')


def symlink_output_files():
    """ serves to create symbolic links between the results of the batch script executions and the main folder directories. """
    raise NotImplementedError
    # should_symlink_output_pickles
    
    src_path = whl_file
    dst_path = 'current.whl'
    # Create the symbolic link
    try:
        print(f'\t symlinking {src_path} to {dst_path}')
        os.symlink(src_path, dst_path)
    except FileExistsError as e:
        print(f'\t WARNING: symlink {dst_path} already exists. Removing it.')
        # Remove the symlink
        os.unlink(dst_path)
        # Create the symlink
        os.symlink(src_path, dst_path)
    except Exception as e:
        raise e


def build_vscode_workspace(script_paths):
    """ builds a VSCode workspace for the batch python scripts
    
        from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import build_vscode_workspace

        vscode_workspace_path = build_vscode_workspace(script_paths)
        vscode_workspace_path

    """
    from jinja2 import Template

    assert len(script_paths) > 0, f"script_paths is empty!"
    top_level_script_folders_path: Path = Path(script_paths[0]).resolve().parent.parent # parent of the parents
    script_folders: List[Path] = [top_level_script_folders_path] + [Path(a_path).parent.resolve() for a_path in script_paths]
    print(f'script_folders: {script_folders}')
    # {
    #     "path": "L:/Scratch/gen_scripts",
    #     "name": "gen_scripts_root"
    # },
    vscode_workspace_path = top_level_script_folders_path.joinpath('run_workspace.code-workspace').resolve()
    print(f'vscode_workspace_path: {vscode_workspace_path}')

    # Define your VSCode workspace template
    vscode_workspace_template = """
    {
        "folders": [
            {% for folder in folders %}
            {
                "path": "{{ folder.path }}",
                "name": "{{ folder.name }}"
            }{% if not loop.last %},{% endif %}
            {% endfor %}
        ],
        "settings": {
            "python.testing.autoTestDiscoverOnSaveEnabled": false,
            "python.terminal.executeInFileDir": true,
            "python.terminal.focusAfterLaunch": true,
            "python.terminal.launchArgs": [
            ],
        }
    }
    """

    # Create Jinja template object
    template = Template(vscode_workspace_template)

    # Define folders as a list of dictionaries
    # folders = [
    #     {'path': '/path/to/your/project1', 'name': 'Project1'},
    #     {'path': '/path/to/your/project2', 'name': 'Project2'},
    # ]
    folders = [
        {'path': f'{a_folder.as_posix()}', 'name': f'{a_folder.name}'}
        for a_folder in script_folders
    ]

    # Define variables
    variables = {
        'folders': folders
    }

    # Render the template with variables
    workspace_file_content = template.render(variables)

    # Write the generated content to a workspace file
    with open(vscode_workspace_path, 'w') as f:
        f.write(workspace_file_content)

    return vscode_workspace_path


def build_windows_powershell_run_script(script_paths, max_concurrent_jobs: int = 3,
                                        activate_path='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/activate.bat', 
                                        python_executable='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/python.exe'):
    """
    Builds a Powershell script to run Python scripts in parallel on Windows.

    max_concurrent_jobs (int): Maximum number of concurrent jobs, limited by your system's memory.
    activate_path (str): Path to the activate script of the Python virtual environment.
    python_executable (str): Path to the Python executable within the virtual environment.

    Returns:
        Path: The path to the generated Powershell script.
    """

    # Ensure that there are script paths provided
    assert script_paths, "script_paths list cannot be empty!"

    # Get the top-level directory to save the Powershell script
    top_level_script_folders_path = Path(script_paths[0]).resolve().parent.parent
    ps_script_path = top_level_script_folders_path.joinpath('run_scripts.ps1').resolve()
    print(f'ps_script_path: {ps_script_path}')

    # PowerShell script preamble
    powershell_script = f"""
    # Define a ScriptBlock to activate the virtual environment, change directory, and execute the Python script
    $scriptBlock = {{
        param([string]$activatePath, [string]$pythonExec, [string]$scriptPath, [string]$parentDir)
        & $activatePath | Out-Null
        Set-Location -Path $parentDir
        & $pythonExec $scriptPath | Out-Null
    }}

    # Initialize job queue and set the job limit
    $jobQueue = @()
    $jobLimit = {max_concurrent_jobs}
    """

    # Job creation and queuing
    for script in script_paths:
        parent_directory = Path(script).resolve().parent  # Get the parent directory of the script
        powershell_script += f"""
    # Wait until there is a free slot to run a new job
    while ($jobQueue.Count -ge $jobLimit) {{
        $completedJobs = @($jobQueue | Where-Object {{ $_.State -eq 'Completed' }})
        if ($completedJobs) {{
            # Remove completed jobs from the queue
            $completedJobs | Remove-Job
            $jobQueue = @($jobQueue | Where-Object {{ $_.State -ne 'Completed' }})
        }} else {{
            # Wait for some time before checking again
            Start-Sleep -Seconds 5
        }}
    }}

    # Add a new job to the queue
    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList '{activate_path}', '{python_executable}', '{script}', '{parent_directory}'
    $jobQueue += , $job  # Append job to the queue as an array element
    """

    # Finish the script with job monitoring and cleanup
    powershell_script += """
    # Wait for all queued jobs to complete
    $jobQueue | Wait-Job

    # Gather the results of each job (optional)
    $jobQueue | ForEach-Object {
        $_ | Receive-Job
    }

    # Clean up the job queue
    $jobQueue | Remove-Job
    """

    # Save the generated PowerShell script to a file
    with open(ps_script_path, 'w') as file:
        file.write(powershell_script)

    return ps_script_path

