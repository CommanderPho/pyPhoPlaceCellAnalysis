import sys
import os
import platform
import pkg_resources # for Slurm templating
from jinja2 import Environment, FileSystemLoader # for Slurm templating
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
import attrs
from attrs import define, field, Factory
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
from pyphocorehelpers.notebook_helpers import convert_script_to_notebook

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
# from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths

# included_session_contexts, output_python_scripts, output_slurm_scripts, powershell_script_path, vscode_workspace_path
BatchScriptsCollection = attrs.make_class("BatchScriptsCollection", {k:field() for k in ("included_session_contexts", "output_python_scripts", "output_jupyter_notebooks", "output_slurm_scripts", "output_non_slurm_bash_scripts", "vscode_workspace_path")}) # , "max_parallel_executions", "powershell_script_path"

from enum import Enum

from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import KnownNeptuneProjects

def get_batch_neptune_kwargs():
    return KnownNeptuneProjects.get_PhoDibaBatchProcessing_neptune_kwargs()



class ProcessingScriptPhases(Enum):
    """ These phases keep track of groups of computations to run.

    
    from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import ProcessingScriptPhases


    """
    clean_run = "clean_run"
    continued_run = "continued_run"
    final_run = "final_run"
    figure_run = "figure_run"

    @property
    def is_figure_phase(self) -> bool:
        if self.value == ProcessingScriptPhases.figure_run.value:
            return True
        else:
            return False
        
    def get_custom_user_completion_functions_dict(self, extra_run_functions=None) -> Dict:
        """ get the extra user_completion functions
        
        """
        from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import export_session_h5_file_completion_function, curr_runtime_context_header_template, export_rank_order_results_completion_function, figures_rank_order_results_completion_function, determine_session_t_delta_completion_function, perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, reload_exported_kdiba_session_position_info_mat_completion_function, compute_and_export_session_wcorr_shuffles_completion_function, compute_and_export_session_instantaneous_spike_rates_completion_function, compute_and_export_session_extended_placefield_peak_information_completion_function, compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function, backup_previous_session_files_completion_function, compute_and_export_session_trial_by_trial_performance_completion_function, save_custom_session_files_completion_function

        if self.value == ProcessingScriptPhases.figure_run.value:
            # figure stage:
            phase_figure_custom_user_completion_functions_dict = {
                                    # "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                                    }
            return phase_figure_custom_user_completion_functions_dict
        else:
            # run stage:
            phase_any_run_custom_user_completion_functions_dict = {
                # "export_rank_order_results_completion_function": export_rank_order_results_completion_function, # ran 2024-04-28 12:57am
                # "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function, # ran 2024-05-22 12:58am
                # "determine_session_t_delta_completion_function": determine_session_t_delta_completion_function,
                # 'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, # ran 2024-05-22 12:58am
                # 'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, # ran 2024-05-22 12:58am
                # 'reload_exported_kdiba_session_position_info_mat_completion_function': reload_exported_kdiba_session_position_info_mat_completion_function,
                # 'export_session_h5_file_completion_function': export_session_h5_file_completion_function,
                }
            
            phase0_any_run_custom_user_completion_functions_dict = {
                # 'backup_previous_session_files_completion_function': backup_previous_session_files_completion_function, # disabled 2024-10-29
                # "determine_session_t_delta_completion_function": determine_session_t_delta_completion_function,  # ran 2024-05-28 6am
                'reload_exported_kdiba_session_position_info_mat_completion_function': reload_exported_kdiba_session_position_info_mat_completion_function,
            }

            # Unused:
            # _phase_rank_order_results_run_custom_user_completion_functions_dict = {
            #     "export_rank_order_results_completion_function": export_rank_order_results_completion_function, # ran 2024-09-26 3pm
            #     'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function': compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function, # ran 2024-07-16 5am 
            #     'compute_and_export_session_wcorr_shuffles_completion_function': compute_and_export_session_wcorr_shuffles_completion_function,
            # }
            

            phase1_any_run_custom_user_completion_functions_dict = {
                # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function, # ran 2024-07-16 5am
                # 'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, # ran 2024-09-26 3pm
                # 'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, # ran 2024-09-26 3pm
                # 'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function': compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function, # ran 2024-07-16 5am 
                # 'compute_and_export_session_instantaneous_spike_rates_completion_function': compute_and_export_session_instantaneous_spike_rates_completion_function,
                # 'compute_and_export_session_extended_placefield_peak_information_completion_function': compute_and_export_session_extended_placefield_peak_information_completion_function,
                # 'compute_and_export_session_trial_by_trial_performance_completion_function': compute_and_export_session_trial_by_trial_performance_completion_function, 
                # 'export_session_h5_file_completion_function': export_session_h5_file_completion_function, # ran 2024-09-26 3pm
            }

            phase3_any_run_custom_user_completion_functions_dict = {                
                # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function, # ran 2024-07-16 5am
                'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, # ran 2024-09-26 3pm
                'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, # ran 2024-09-26 3pm
                # **_phase_rank_order_results_run_custom_user_completion_functions_dict,
                'compute_and_export_session_instantaneous_spike_rates_completion_function': compute_and_export_session_instantaneous_spike_rates_completion_function,
                'compute_and_export_session_extended_placefield_peak_information_completion_function': compute_and_export_session_extended_placefield_peak_information_completion_function,
                'compute_and_export_session_trial_by_trial_performance_completion_function': compute_and_export_session_trial_by_trial_performance_completion_function, 
                'export_session_h5_file_completion_function': export_session_h5_file_completion_function, # ran 2024-09-26 3pm
                'save_custom_session_files_completion_function': save_custom_session_files_completion_function,
            }

            if self.value == ProcessingScriptPhases.clean_run.value:
                phase_any_run_custom_user_completion_functions_dict = phase_any_run_custom_user_completion_functions_dict | phase0_any_run_custom_user_completion_functions_dict
            elif self.value == ProcessingScriptPhases.continued_run.value:
                phase_any_run_custom_user_completion_functions_dict = phase_any_run_custom_user_completion_functions_dict | phase1_any_run_custom_user_completion_functions_dict
            elif self.value == ProcessingScriptPhases.final_run.value:
                phase_any_run_custom_user_completion_functions_dict = phase_any_run_custom_user_completion_functions_dict | phase3_any_run_custom_user_completion_functions_dict
            elif self.value == ProcessingScriptPhases.figure_run.value:
                phase_figures_custom_user_completion_functions_dict = {
                    "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                }
                phase_any_run_custom_user_completion_functions_dict = phase_any_run_custom_user_completion_functions_dict | phase_figures_custom_user_completion_functions_dict
            else:
                raise NotImplementedError
            
            if extra_run_functions is not None:
                phase_any_run_custom_user_completion_functions_dict = (phase_any_run_custom_user_completion_functions_dict | extra_run_functions)

            return phase_any_run_custom_user_completion_functions_dict


    def get_run_configuration(self, custom_user_completion_function_template_code=None, extra_extended_computations_include_includelist: Optional[List]=None):
        ## Different run configurations:

        phase1_extended_computations_include_includelist=['lap_direction_determination', 'pf_computation', 
                                                'pfdt_computation', 'firing_rate_trends',
            # 'pf_dt_sequential_surprise',
            'extended_stats',
            'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding',
            # 'ratemap_peaks_prominence2d',
            'long_short_inst_spike_rate_groups',
            'long_short_endcap_analysis',
            # 'spike_burst_detection',
            'split_to_directional_laps',
            'merged_directional_placefields',
            # 'rank_order_shuffle_analysis',
            # 'directional_train_test_split',
            # 'directional_decoders_decode_continuous',
            # 'directional_decoders_evaluate_epochs',
            # 'directional_decoders_epoch_heuristic_scoring',
        ]


        phase2_extended_computations_include_includelist=['lap_direction_determination', 'pf_computation', 
                                                'pfdt_computation', 'firing_rate_trends',
            # 'pf_dt_sequential_surprise',
            'extended_stats',
            'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding',
            # 'ratemap_peaks_prominence2d',
            'long_short_inst_spike_rate_groups',
            'long_short_endcap_analysis',
            # 'spike_burst_detection',
            'split_to_directional_laps',
            'merged_directional_placefields',
            'rank_order_shuffle_analysis',
            # 'directional_train_test_split',
            'directional_decoders_decode_continuous',
            'directional_decoders_evaluate_epochs',
            'directional_decoders_epoch_heuristic_scoring',
        ]

        phase3_extended_computations_include_includelist=['lap_direction_determination', 'pf_computation', 
                                                'pfdt_computation', 'firing_rate_trends',
            'pf_dt_sequential_surprise',
            'extended_stats',
            'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 
            'ratemap_peaks_prominence2d',
            'long_short_inst_spike_rate_groups',
            'long_short_endcap_analysis',
            # 'spike_burst_detection',
            'split_to_directional_laps',
            'merged_directional_placefields',
            'rank_order_shuffle_analysis',
            'directional_train_test_split',
            'directional_decoders_decode_continuous',
            'directional_decoders_evaluate_epochs',
            'directional_decoders_epoch_heuristic_scoring',
            'extended_pf_peak_information',
        ]

        _out_run_config = {}
        if self.value == ProcessingScriptPhases.clean_run.value:
            clean_run = dict(should_force_reload_all=True, should_freeze_pipeline_updates=False, extended_computations_include_includelist=phase1_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=False), should_perform_figure_generation_to_file=False, custom_user_completion_function_template_code=custom_user_completion_function_template_code)
            _out_run_config = clean_run
        elif self.value == ProcessingScriptPhases.continued_run.value:
            continued_run = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, extended_computations_include_includelist=phase2_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=False), should_perform_figure_generation_to_file=False, custom_user_completion_function_template_code=custom_user_completion_function_template_code)
            _out_run_config = continued_run
        elif self.value == ProcessingScriptPhases.final_run.value:
            # final_run = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, extended_computations_include_includelist=phase3_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=True), should_perform_figure_generation_to_file=False, custom_user_completion_function_template_code=custom_user_completion_function_template_code)
            final_run = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, extended_computations_include_includelist=phase3_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=False), should_perform_figure_generation_to_file=False, custom_user_completion_function_template_code=custom_user_completion_function_template_code) # use export_session_h5_file_completion_function instead of enable_hdf5_output=True
            _out_run_config = final_run
        elif self.value == ProcessingScriptPhases.figure_run.value:
            figure_run = dict(should_perform_figure_generation_to_file=True, should_force_reload_all=False, should_freeze_pipeline_updates=True, extended_computations_include_includelist=phase3_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=False), custom_user_completion_function_template_code=custom_user_completion_function_template_code)
            _out_run_config = figure_run
        else: 
            raise NotImplementedError
        
        if extra_extended_computations_include_includelist is not None:
            ## includes the user-provided extra run functions to the list: `extra_extended_computations_include_includelist`
            for a_fn_name in extra_extended_computations_include_includelist:
                if (a_fn_name not in _out_run_config['extended_computations_include_includelist']):
                    print(f'adding extra_extended_computations_include_includelist function: "{a_fn_name}" to the `extended_computations_include_includelist`.')
                    _out_run_config['extended_computations_include_includelist'].append(a_fn_name)
                else:
                    print(f'extra_extended_computations_include_includelist function: "{a_fn_name}" was already present in the default `extended_computations_include_includelist`. It will not be duplicated.')
        
        if (platform.system() == 'Windows'):
            _out_run_config.update(dict(create_slurm_scripts=False, should_create_vscode_workspace=True))
        else:
            _out_run_config.update(dict(create_slurm_scripts=True, should_create_vscode_workspace=False))

        return _out_run_config


@function_attributes(short_name=None, tags=['slurm','jobs','files','batch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-09 19:14', related_items=[])
def generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs: Dict[IdentifyingContext, Path], included_session_contexts: Optional[List[IdentifyingContext]], output_directory='output/gen_scripts/', use_separate_run_directories:bool=True,
         create_slurm_scripts:bool=False, create_non_slurm_bash_scripts:bool=False, should_create_vscode_workspace:bool=True, should_use_neptune_logging:bool=True,  should_generate_run_scripts = True, should_generate_figure_scripts = True, should_generate_run_notebooks: bool = False,
         should_use_file_redirected_output_logging:bool=False, # , should_create_powershell_scripts:bool=True
         separate_execute_and_figure_gen_scripts:bool=True, should_perform_figure_generation_to_file:bool=False, force_recompute_override_computations_includelist: Optional[List[str]]=None, force_recompute_override_computation_kwargs_dict: Optional[Dict[str, Dict]]=None, 
        batch_session_completion_handler_kwargs=None, **renderer_script_generation_kwargs) -> BatchScriptsCollection:
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
    def _subfn_build_slurm_script(curr_batch_script_rundir, a_python_script_path, a_curr_session_context, a_slurm_script_name_prefix:str='run', should_use_virtual_framebuffer:bool=False):
        slurm_script_path = os.path.join(curr_batch_script_rundir, f'{a_slurm_script_name_prefix}_{a_curr_session_context}.sh')
        with open(slurm_script_path, 'w') as script_file:
            script_content = slurm_template.render(curr_session_context=f"{a_curr_session_context}", python_script_path=a_python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, should_use_virtual_framebuffer=should_use_virtual_framebuffer)
            script_file.write(script_content)
        return slurm_script_path
    
    def _subfn_build_non_slurm_bash_script(curr_batch_script_rundir, a_python_script_path, a_curr_session_context, a_bash_script_name_prefix:str='run'):
        bash_script_path = os.path.join(curr_batch_script_rundir, f'{a_bash_script_name_prefix}_{a_curr_session_context}.sh')
        with open(bash_script_path, 'w') as script_file:
            script_content = bash_non_slurm_template.render(curr_session_context=f"{a_curr_session_context}", python_script_path=a_python_script_path, curr_batch_script_rundir=curr_batch_script_rundir)
            script_file.write(script_content)
        return bash_script_path

    assert isinstance(session_batch_basedirs, dict)

    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory).resolve()


    separate_execute_and_figure_gen_scripts = renderer_script_generation_kwargs.pop('separate_execute_and_figure_gen_scripts', True)
    assert separate_execute_and_figure_gen_scripts, f"Old non-separate mode not supported"
 
    assert ('force_recompute_override_computations_includelist' not in renderer_script_generation_kwargs), f"pass 'force_recompute_override_computations_includelist' explicitly to the call!!"
    renderer_script_generation_kwargs['force_recompute_override_computations_includelist'] = (force_recompute_override_computations_includelist or [])
 
    assert ('force_recompute_override_computation_kwargs_dict' not in renderer_script_generation_kwargs), f"pass 'force_recompute_override_computation_kwargs_dict' explicitly to the call!!"
    renderer_script_generation_kwargs['force_recompute_override_computation_kwargs_dict'] = (force_recompute_override_computation_kwargs_dict or {})
    
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
    bash_non_slurm_template = env.get_template('bash_template.sh.j2')

    output_python_scripts = []
    output_jupyter_notebooks = []
    

    output_slurm_scripts = {'run': [], 'figs': []}
    output_non_slurm_bash_scripts = {'run': [], 'figs': []}


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
        # Run Script _________________________________________________________________________________________________________ #
        python_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_context}.py')
        with open(python_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_basedir=curr_session_basedir, 
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    should_use_neptune_logging=should_use_neptune_logging, should_use_file_redirected_output_logging=should_use_file_redirected_output_logging,
                                                    **(compute_as_needed_script_generation_kwargs | dict(should_perform_figure_generation_to_file=False)))
            # script_file.write(script_content)
            script_file.write(script_content.encode())
        # output_python_scripts.append(python_script_path)

        # Figures Script _____________________________________________________________________________________________________ #
        python_figures_script_path = os.path.join(curr_batch_script_rundir, f'figures_{curr_session_context}.py')
        with open(python_figures_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_basedir=curr_session_basedir,
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    should_use_neptune_logging=should_use_neptune_logging, should_use_file_redirected_output_logging=should_use_file_redirected_output_logging,
                                                    **(no_recomputing_script_generation_kwargs | dict(should_perform_figure_generation_to_file=True)))


            # script_file.write(script_content)
            script_file.write(script_content.encode())

        # output_python_display_scripts.append(python_figures_script_path)
        output_python_scripts.append((python_script_path, python_figures_script_path))

        # Create the SLURM script
        if create_slurm_scripts:
            if should_generate_run_scripts:
                slurm_run_script_path = _subfn_build_slurm_script(a_python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_slurm_script_name_prefix='run', should_use_virtual_framebuffer=False)
                output_slurm_scripts['run'].append(slurm_run_script_path)

            if should_generate_figure_scripts or should_perform_figure_generation_to_file:
                slurm_figure_script_path = _subfn_build_slurm_script(a_python_script_path=python_figures_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_slurm_script_name_prefix='figs', should_use_virtual_framebuffer=False)
                output_slurm_scripts['figs'].append(slurm_figure_script_path)


        ## Create the non-slurm bash script:
        if create_non_slurm_bash_scripts:
            if should_generate_run_scripts:
                bash_run_script_path = _subfn_build_non_slurm_bash_script(a_python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_bash_script_name_prefix='run')
                output_non_slurm_bash_scripts['run'].append(bash_run_script_path)

            if should_generate_figure_scripts or should_perform_figure_generation_to_file:
                bash_figure_script_path = _subfn_build_non_slurm_bash_script(a_python_script_path=python_figures_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_bash_script_name_prefix='figs')
                output_non_slurm_bash_scripts['figs'].append(bash_figure_script_path)


        if should_generate_run_notebooks:
            script_path = Path(python_script_path).resolve()
            # script_dir = script_path.parent.resolve()
            notebook_path = script_path.with_suffix('.ipynb')
            convert_script_to_notebook(script_path, notebook_path)
            output_jupyter_notebooks.append(notebook_path)
            
            # convert_script_to_notebook(script_path, notebook_path, custom_delimiter=None)

    # if should_create_powershell_scripts and (platform.system() == 'Windows'):
    #     powershell_script_path = build_windows_powershell_run_script(output_python_scripts, max_concurrent_jobs=max_parallel_executions)
    #     powershell_script_path


    ## Generate VSCode Workspace for it
    if should_create_vscode_workspace:
        output_compute_python_scripts = [x[0] for x in output_python_scripts]
        vscode_workspace_path = build_vscode_workspace(output_compute_python_scripts, python_executable=Path('~/Library/VSCode/black/.venv_black/bin/python').resolve())
        print(f'vscode_workspace_path: {vscode_workspace_path}')
    else:
        vscode_workspace_path = None


    return BatchScriptsCollection(included_session_contexts=included_session_contexts, output_python_scripts=output_python_scripts, output_jupyter_notebooks=output_jupyter_notebooks, output_slurm_scripts=output_slurm_scripts, output_non_slurm_bash_scripts=output_non_slurm_bash_scripts, vscode_workspace_path=vscode_workspace_path)
    # return included_session_contexts, output_python_scripts, output_slurm_scripts


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



@function_attributes(short_name=None, tags=['python', 'virtualenv', 'environment'], input_requires=[], output_provides=[], uses=['get_python_environment'], used_by=[], creation_date='2024-04-15 10:33', related_items=[])
def get_running_python(debug_print:bool=True):
    """ gets the path to the currently running python and its environment info.
    
    Usage:
    
        from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import get_running_python

        active_venv_path, python_executable, activate_script_path = get_running_python()
        
    """
    current_python_executable = Path(sys.executable).resolve()
    assert current_python_executable.exists(), f'current_python_executable: "{current_python_executable}" must exist.'
    if debug_print:
        print(f'current_python_executable: "{current_python_executable}"')
    ## Get the environment from it:
    active_venv_path: Path = current_python_executable.parent.parent.resolve()
    active_venv_path, python_executable, activate_script_path = get_python_environment(active_venv_path=active_venv_path)
    return active_venv_path, python_executable, activate_script_path


@function_attributes(short_name=None, tags=['python', 'virtualenv', 'environment'], input_requires=[], output_provides=[], uses=[], used_by=['get_running_python'], creation_date='2024-04-15 10:34', related_items=[])
def get_python_environment(active_venv_path: Path, debug_print:bool=True):
    """
    
    from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import get_python_environment
    
    active_venv_path, python_executable, activate_script_path = get_python_environment(active_venv_path=active_venv_path)
    
    """
    # INPUTS: active_venv_path, 
    if isinstance(active_venv_path, str):
        active_venv_path = Path(active_venv_path).resolve()
    assert active_venv_path.exists(), f'active_venv_path: "{active_venv_path}" must exist.'
    if debug_print:
        print(f'active_venv_path: "{active_venv_path}"')

    # Check if the current operating system is Windows
    if os.name == 'nt':
        # Put your Windows-specific code here
        # python_executable = active_venv_path.joinpath('bin', 'python').resolve()
        python_executable = active_venv_path.joinpath('Scripts', 'python.exe').resolve()
        # activate_path = Path('. /home/halechr/repos/Spike3D/.venv/bin/activate')
        # python_path = Path('/home/halechr/repos/Spike3D/.venv/bin/python')
        activate_script_path = active_venv_path.joinpath('Scripts', 'activate.ps1').resolve()

    else:
        # Put your non-Windows-specific code here
        python_executable = active_venv_path.joinpath('bin', 'python').resolve()
        # activate_path = Path('. /home/halechr/repos/Spike3D/.venv/bin/activate')
        # python_path = Path('/home/halechr/repos/Spike3D/.venv/bin/python')
        activate_script_path = active_venv_path.joinpath('bin', 'activate').resolve()

    assert activate_script_path.exists(), f'activate_script_path: "{activate_script_path}" must exist.'
    assert python_executable.exists(), f'python_executable: "{python_executable}" must exist.'
    if debug_print:
        print(f'activate_script_path: "{activate_script_path}"')
        print(f'python_executable: "{python_executable}"')


    # activate_path = Path('. /home/halechr/repos/Spike3D/.venv/bin/activate')
    # python_path = Path('/home/halechr/repos/Spike3D/.venv/bin/python')
    return active_venv_path, python_executable, activate_script_path


@function_attributes(short_name=None, tags=['vscode_workspace', 'vscode'], input_requires=[], output_provides=[], uses=['get_running_python'], used_by=[], creation_date='2024-04-15 10:35', related_items=[])
def build_vscode_workspace(script_paths, python_executable=None):
    """ builds a VSCode workspace for the batch python scripts
    
        from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import build_vscode_workspace

        vscode_workspace_path = build_vscode_workspace(script_paths)
        vscode_workspace_path

    """
    from jinja2 import Template

    if python_executable is None:
        active_venv_path, python_executable, activate_script_path = get_running_python()

    is_platform_windows: bool = False
    if (platform.system() == 'Windows'):
        is_platform_windows = True
    else:
        is_platform_windows = False

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

    # Set up Jinja2 environment
    template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('vscode_workspace_template.code-workspace.j2')
    # Render the template with the provided variables
    
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
        'folders': folders,
        'defaultInterpreterPath': str(python_executable.as_posix()),
        'is_platform_windows': is_platform_windows,
    }

    # Render the template with variables
    workspace_file_content = template.render(variables)

    # Write the generated content to a workspace file
    with open(vscode_workspace_path, 'w') as f:
        f.write(workspace_file_content)

    return vscode_workspace_path


@function_attributes(short_name=None, tags=['Windows-only', 'powershell', 'batch', 'script'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-15 11:00', related_items=[])
def build_windows_powershell_run_script(script_paths, max_concurrent_jobs: int = 3,
                                        activate_path='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/activate.bat', 
                                        python_executable='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv/Scripts/python.exe',
                                        script_name: str = 'run_scripts'):
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
    ps_script_path = top_level_script_folders_path.joinpath(f'{script_name}.ps1').resolve()
    print(f'ps_script_path: {ps_script_path}')

    # Set up Jinja2 environment
    template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
    env = Environment(loader=FileSystemLoader(template_path))
    powershell_script_template = env.get_template('powershell_template.ps1.j2')
    # Render the template with the provided variables
    powershell_script = powershell_script_template.render(
        script_paths=script_paths,
        max_concurrent_jobs=max_concurrent_jobs,
        activate_path=activate_path,
        python_executable=python_executable
    )


#     # PowerShell script preamble
#     powershell_script = f"""
# # Define a ScriptBlock to activate the virtual environment, change directory, and execute the Python script
# $scriptBlock = {{
#     param([string]$activatePath, [string]$pythonExec, [string]$scriptPath, [string]$parentDir)
#     try {{
#         & $activatePath | Out-Null
#         Set-Location -Path $parentDir
#         $startTime = Get-Date
#         Write-Host "Starting script: $scriptPath at time: $($startTime.ToString())" # Log which script is starting with start time
#         & $pythonExec $scriptPath | Out-Null
#         $endTime = Get-Date
#         $duration = $endTime - $startTime
#         Write-Host "Completed script: $scriptPath at time: $($endTime.ToString()) with duration: $($duration.ToString())" # Log when the script completes
#         return @{{ScriptPath=$scriptPath; StartTime=$startTime.ToString(); EndTime=$endTime.ToString(); Duration=$duration.ToString()}}
#     }} catch {{
#         Write-Error "An error occurred for script: $scriptPath"
#         return @{{ScriptPath=$scriptPath; StartTime=$startTime.ToString(); EndTime=(Get-Date).ToString(); Duration="Failed"}}
#     }}
# }}

# # Function to manage job queue
# function Manage-JobQueue {{
#     param (
#         [int]$jobLimit,
#         [ref]$jobQueue
#     )

#     while ($jobQueue.Value.Count -ge $jobLimit) {{
#         $completedJobs = @($jobQueue.Value | Where-Object {{ $_.State -eq 'Completed' }})
#         foreach ($job in $completedJobs) {{
#             # Remove completed jobs from the queue
#             $job | Remove-Job
#             Write-Host "Job $($job.Id) has been removed from the queue."
#         }}
#         $jobQueue.Value = @($jobQueue.Value | Where-Object {{ $_.State -ne 'Completed' }})
#         if (!$completedJobs) {{
#             # Wait for some time before checking again if no jobs were completed
#             Start-Sleep -Seconds 3
#         }}
#     }}
# }}

# # Function to start a job and add it to the queue
# function Start-NewJob {{
#     param (
#         [ref]$jobQueue,
#         [scriptblock]$scriptBlock,
#         [string[]]$arguments
#     )

#     $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList $arguments
#     $jobQueue.Value += $job  # Append job to the queue as an array element
#     Write-Host "Starting Job for '$($arguments[-1])'"
# }}

# # Function to wait for all queued jobs to complete and log their outputs
# function WaitForAllJobs {{
#     param (
#         [ref]$jobQueue,
#         [ref]$runHistory
#     )

#     while ($jobQueue.Value.Count -gt 0) {{
#         $completedJobs = @($jobQueue.Value | Wait-Job -Any)

#         # Receive and log output from completed jobs
#         foreach ($job in $completedJobs) {{
#             # Receive all the outputs from the job
#             $jobOutputs = Receive-Job -Job $job

#             # Verify that we have received some outputs
#             if ($jobOutputs -ne $null) {{
#                 # Look for the hashtable we expect among job outputs
#                 $hashtable = $jobOutputs | Where-Object {{ $_ -is [System.Collections.Hashtable] }} | Select-Object -Last 1

#                 if ($hashtable) {{
#                     $runHistory.Value += New-Object -TypeName PSObject -Property $hashtable
#                     Write-Host "Job $($job.Id) with script '$($hashtable.ScriptPath)' started at $($hashtable.StartTime) and took $($hashtable.Duration) has completed."
#                 }} else {{
#                     Write-Error "Job $($job.Id) did not return a hashtable."
#                     # Debug - Write all outputs to see what was received
#                     $jobOutputs | ForEach-Object {{ Write-Host "Output: $_" }}
#                 }}
#             }} else {{
#                 Write-Error "Job $($job.Id) did not produce any output."
#             }}

#             $jobQueue.Value = $jobQueue.Value | Where-Object {{ $_.Id -ne $job.Id }}
#         }}

#         # Clean up completed job objects
#         Remove-Job -Job $completedJobs
#     }}
# }}

# # Initialize job queue and set the job limit
# $jobQueue = @()
# $jobLimit = {max_concurrent_jobs}
# $runHistory = @()
#     """

#     # Job creation and queuing
#     for script in script_paths:
#         parent_directory = Path(script).resolve().parent  # Get the parent directory of the script



#         powershell_script += f"""
# # Wait until there is a free slot to run a new job
# Manage-JobQueue -jobLimit $jobLimit -jobQueue ([ref]$jobQueue)
# Start-NewJob -jobQueue ([ref]$jobQueue) -scriptBlock $scriptBlock -arguments @('{activate_path}', '{python_executable}', '{script}', '{parent_directory}')
# """


    #     powershell_script += f"""
    # # Wait until there is a free slot to run a new job
    # while ($jobQueue.Count -ge $jobLimit) {{
    #     $completedJobs = @($jobQueue | Where-Object {{ $_.State -eq 'Completed' }})
    #     foreach ($job in $completedJobs) {{
    #         # Remove completed jobs from the queue
    #         $job | Remove-Job
    #         Write-Host "Job $($job.Id) has been removed from the queue."
    #     }}
    #     $jobQueue = @($jobQueue | Where-Object {{ $_.State -ne 'Completed' }})
    #     if (!$completedJobs) {{
    #         # Wait for some time before checking again if no jobs were completed
    #         Start-Sleep -Seconds 5
    #     }}
    # }}

    # # Add a new job to the queue
    # $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList '{activate_path}', '{python_executable}', '{script}', '{parent_directory}'
    # $jobQueue += , $job  # Append job to the queue as an array element

    # Write-Host "Starting Job for '{script}'"
    # """

#     # Finish the script with job monitoring and cleanup
#     powershell_script += f"""
# # Wait for all queued jobs to complete, logging after each completes
# WaitForAllJobs -jobQueue ([ref]$jobQueue) -runHistory ([ref]$runHistory)
# Write-Host "All jobs have been processed."
#     """

#     # Export the run history to a CSV file
#     powershell_script += """
# Write-Host "Exporting run history to CSV file..."
# $csvPath = [System.IO.Path]::Combine($parentDir, "run_history.csv")
# $runHistory | Export-Csv -Path $csvPath -NoTypeInformation
# Write-Host "Run history has been exported to $csvPath"
#     """

    # Save the generated PowerShell script to a file
    with open(ps_script_path, 'w') as file:
        file.write(powershell_script)

    return ps_script_path


