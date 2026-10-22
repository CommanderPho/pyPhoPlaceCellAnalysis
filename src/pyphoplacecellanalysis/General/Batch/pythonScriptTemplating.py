import sys
import os
import re
import json
import platform
import shutil
import subprocess
import pkg_resources # for Slurm templating
from jinja2 import Environment, FileSystemLoader # for Slurm templating
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
import attrs
from attrs import define, field, Factory
from typing import Any, List, Dict, Literal, Optional, Tuple, Union, Callable
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
from pyphoplacecellanalysis.General.Batch.concrete_session_folder import ConcreteSessionFolder

# included_session_contexts, output_python_scripts, output_slurm_scripts, powershell_script_path, vscode_workspace_path
# BatchScriptsCollection = attrs.make_class("BatchScriptsCollection", {k:field() for k in ("included_session_contexts", "output_python_scripts", "output_jupyter_notebooks", "output_slurm_scripts", "output_non_slurm_bash_scripts", "vscode_workspace_path")}) # , "max_parallel_executions", "powershell_script_path"


@define(slots=False, eq=False)
class BatchScriptsCollection:
    """ A list of outputs for batch script generation
    
    from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import BatchScriptsCollection
    
    output_slurm_scripts = {'run': [], 'figs': []}
    output_non_slurm_bash_scripts = {'run': [], 'figs': []}
    
    """
    included_session_contexts: List = field(default=Factory(list))
    output_python_scripts: List = field(default=Factory(list))
    output_jupyter_notebooks: List = field(default=Factory(list))
    output_slurm_scripts: Dict = field(default=Factory(dict))
    output_non_slurm_bash_scripts: Dict = field(default=Factory(dict))
    vscode_workspace_path: Path = field(default=None)
    batch_scripts_root_directory: Path = field(default=None)
    

    def __add__(self, other):
        """Add two BatchScriptsCollection objects by appending their List fields.
        
        Args:
            other (BatchScriptsCollection): Another collection to add to this one.
            
        Returns:
            BatchScriptsCollection: A new collection with combined lists.
        """
        if not isinstance(other, BatchScriptsCollection):
            return NotImplemented
            
        result = BatchScriptsCollection()
        
        # Combine all list fields
        result.included_session_contexts = self.included_session_contexts + other.included_session_contexts
        result.output_python_scripts = self.output_python_scripts + other.output_python_scripts
        result.output_jupyter_notebooks = self.output_jupyter_notebooks + other.output_jupyter_notebooks
        
        assert isinstance(self.output_slurm_scripts, dict), f"self.output_slurm_scripts: type(self.output_slurm_scripts): {type(self.output_slurm_scripts)}"
        for k, v in self.output_slurm_scripts.items():
            ## iterate through the top-level entries like 'run', 'figs', etc. to combine the actual list items
            result.output_slurm_scripts[k] = v + other.output_slurm_scripts.get(k, []) ## optionally empty list
            
        assert isinstance(self.output_non_slurm_bash_scripts, dict), f"self.output_non_slurm_bash_scripts: type(self.output_non_slurm_bash_scripts): {type(self.output_non_slurm_bash_scripts)}"
        for k, v in self.output_non_slurm_bash_scripts.items():
            ## iterate through the top-level entries like 'run', 'figs', etc. to combine the actual list items
            result.output_non_slurm_bash_scripts[k] = v + other.output_non_slurm_bash_scripts.get(k, []) ## optionally empty list

        # result.output_slurm_scripts = self.output_slurm_scripts | other.output_slurm_scripts
        # result.output_non_slurm_bash_scripts = self.output_non_slurm_bash_scripts + other.output_non_slurm_bash_scripts
        
        # For the Path field, use the non-empty one or the first one
        if isinstance(self.vscode_workspace_path, Path):
            result.vscode_workspace_path = self.vscode_workspace_path
        elif isinstance(other.vscode_workspace_path, Path):
            result.vscode_workspace_path = other.vscode_workspace_path

        if isinstance(self.batch_scripts_root_directory, Path):
            result.batch_scripts_root_directory = self.batch_scripts_root_directory
        elif isinstance(other.batch_scripts_root_directory, Path):
            result.batch_scripts_root_directory = other.batch_scripts_root_directory
        
        return result
    


from enum import Enum


def resolve_batch_scripts_root_directory(output_directory: Path, batch_script_subdirectory: Optional[str] = None) -> Path:
    root = Path(output_directory).resolve()
    if batch_script_subdirectory:
        root = root.joinpath(batch_script_subdirectory)
    return root


def _infer_batch_scripts_root_directory(script_paths) -> Path:
    assert script_paths, "script_paths list cannot be empty!"
    script_parent_paths = [Path(a_path).resolve().parent for a_path in script_paths]
    if len(script_parent_paths) == 1:
        return script_parent_paths[0].parent

    common_parent_path = Path(os.path.commonpath([str(a_path) for a_path in script_parent_paths])).resolve()
    common_leaf_names = {a_path.name for a_path in script_parent_paths}
    if len(common_leaf_names) == 1:
        return common_parent_path.joinpath(next(iter(common_leaf_names))).resolve()
    return common_parent_path



def get_batch_neptune_kwargs():
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import KnownNeptuneProjects

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
        

    @property
    def phase_job_suffix(self) -> str:
        """The string indicating which phase/stage we're in to be used in the job_suffix."""
        return self.value.removesuffix('_run').capitalize() # ["Clean", "Continued", "Final", "Figure"]
    


    def get_custom_user_completion_functions_dict(self, extra_run_functions=None) -> Dict:
        """ get the extra user_completion functions
        
        """
        from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import export_rank_order_results_completion_function, figures_rank_order_results_completion_function, determine_session_t_delta_completion_function, perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, compute_and_export_session_wcorr_shuffles_completion_function, compute_and_export_session_instantaneous_spike_rates_completion_function, compute_and_export_session_extended_placefield_peak_information_completion_function, compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function, backup_previous_session_files_completion_function, compute_and_export_session_trial_by_trial_performance_completion_function, save_custom_session_files_completion_function, compute_and_export_cell_first_spikes_characteristics_completion_function, figures_plot_cell_first_spikes_characteristics_completion_function, compute_and_pickle_clusterless_decoder_completion_function, figures_plot_bapun_clusterless_train_test_decoder_error_distance_completion_function, compute_and_pickle_spyglass_clusterless_decoder_completion_function, figures_plot_bapun_spyglass_clusterless_train_test_decoder_error_distance_completion_function
        from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import generalized_decode_epochs_dict_and_export_results_completion_function, figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function #, generalized_export_figures_customizazble_completion_function
        
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
                # 'kdiba_session_post_fixup_completion_function': kdiba_session_post_fixup_completion_function,
                # 'kdiba_session_post_fixup_completion_function': kdiba_session_post_fixup_completion_function,
                }
            
            phase0_any_run_custom_user_completion_functions_dict = {
                # 'backup_previous_session_files_completion_function': backup_previous_session_files_completion_function, # disabled 2024-10-29
                # "determine_session_t_delta_completion_function": determine_session_t_delta_completion_function,  # ran 2024-05-28 6am
                # KDIBA-only — add `'kdiba_session_post_fixup_completion_function': kdiba_session_post_fixup_completion_function` via extra_run_functions for kdiba batches
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
            }

            phase3_any_run_custom_user_completion_functions_dict = {                
                # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function, # ran 2024-07-16 5am
                'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, # ran 2024-09-26 3pm
                'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function, # ran 2024-09-26 3pm
                # **_phase_rank_order_results_run_custom_user_completion_functions_dict,
                'compute_and_export_session_instantaneous_spike_rates_completion_function': compute_and_export_session_instantaneous_spike_rates_completion_function,
                'compute_and_export_session_extended_placefield_peak_information_completion_function': compute_and_export_session_extended_placefield_peak_information_completion_function,
                'compute_and_export_session_trial_by_trial_performance_completion_function': compute_and_export_session_trial_by_trial_performance_completion_function, 
                'compute_and_export_cell_first_spikes_characteristics_completion_function': compute_and_export_cell_first_spikes_characteristics_completion_function,
                'save_custom_session_files_completion_function': save_custom_session_files_completion_function, # session pkl/h5 + optional posteriors H5 (enable_save_h5 / should_write_posterior_h5)
                'generalized_decode_epochs_dict_and_export_results_completion_function': generalized_decode_epochs_dict_and_export_results_completion_function, # Not yet implemented 2025-03-10 19:56 
                'compute_and_pickle_clusterless_decoder_completion_function': compute_and_pickle_clusterless_decoder_completion_function,
                'compute_and_pickle_spyglass_clusterless_decoder_completion_function': compute_and_pickle_spyglass_clusterless_decoder_completion_function,
                'figures_plot_bapun_clusterless_train_test_decoder_error_distance_completion_function': figures_plot_bapun_clusterless_train_test_decoder_error_distance_completion_function,
                'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function': figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function,
                'figures_plot_bapun_spyglass_clusterless_train_test_decoder_error_distance_completion_function': figures_plot_bapun_spyglass_clusterless_train_test_decoder_error_distance_completion_function,
                # 'generalized_export_figures_customizazble_completion_function': generalized_export_figures_customizazble_completion_function
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
            'non_PBE_epochs_results',
            'generalized_specific_epochs_decoding',
        ]

        phase3_extended_computations_include_includelist=['lap_direction_determination', 'pf_computation', 
                                                'pfdt_computation', 'firing_rate_trends',
            'pf_dt_sequential_surprise',  # commented out 2024-11-05
            'extended_stats',
            'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 
            'ratemap_peaks_prominence2d', # commented out 2024-11-05
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
            'perform_wcorr_shuffle_analysis',
            'non_PBE_epochs_results',
            'generalized_specific_epochs_decoding',
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
            final_run = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, extended_computations_include_includelist=phase3_extended_computations_include_includelist, batch_session_completion_handler_kwargs=dict(enable_hdf5_output=False), should_perform_figure_generation_to_file=False, custom_user_completion_function_template_code=custom_user_completion_function_template_code) # use save_custom_session_files_completion_function (enable_save_h5 / should_write_posterior_h5) instead of enable_hdf5_output=True
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

        _out_run_config['phase_job_suffix'] = self.phase_job_suffix

        return _out_run_config


def _compact_slurm_session_part(curr_session_context: Union[IdentifyingContext, str]) -> str:
    """animal + session_name only (drop format_name / exper_name / job_ prefix)."""
    ctx: Optional[IdentifyingContext] = None
    if isinstance(curr_session_context, IdentifyingContext):
        ctx = curr_session_context
    else:
        session_key = str(curr_session_context).strip()
        try:
            ctx = IdentifyingContext.try_init_from_session_key(session_key)
        except Exception:
            # fallback: strip known format_name prefix only
            return re.sub(r'^kdiba_', '', session_key, flags=re.IGNORECASE)
    if all(ctx.has_keys(['animal', 'session_name'])):
        return ctx.get_description(subset_includelist=['animal', 'session_name'])
    return re.sub(r'^kdiba_', '', str(curr_session_context).strip(), flags=re.IGNORECASE)


def _compact_slurm_job_suffix(job_suffix: Optional[str]) -> str:
    """Drop default withNormalComputedReplays; qclu_→q; frateThresh_2.0→fr_2."""
    if not job_suffix:
        return ''
    s = str(job_suffix)
    s = re.sub(r'_?withNormalComputedReplays_?', '_', s, flags=re.IGNORECASE)
    s = re.sub(r'qclu_', 'q', s, flags=re.IGNORECASE)

    def _frate_repl(m: re.Match) -> str:
        num = m.group(1)
        if re.fullmatch(r'\d+\.0+', num):
            num = num.split('.', 1)[0]
        return f'fr_{num}'

    s = re.sub(r'frateThresh_(\d+(?:\.\d+)?)', _frate_repl, s, flags=re.IGNORECASE)
    s = re.sub(r'[_-]+', '_', s).strip('_-')
    return s


def build_slurm_job_name(curr_session_context: Union[IdentifyingContext, str], job_suffix: Optional[str] = None, max_length: int = 64) -> str:
    """Build a short Slurm-safe job name (animal_session + compact suffix). Unquoted #SBATCH --job-name values are split on commas (e.g. qclu_[1, 2, 4])."""
    short_session = _compact_slurm_session_part(curr_session_context)
    compact_suffix = _compact_slurm_job_suffix(job_suffix)
    raw_name = short_session if not compact_suffix else f"{short_session}_{compact_suffix}"
    safe_name = re.sub(r'[\[\],\s]+', '-', raw_name).strip('-')
    return safe_name[:max_length]


# ==================================================================================================================== #
# Batch process phase chaining (per-session Slurm afterok chains)                                                       #
# -------------------------------------------------------------------------------------------------------------------- #
# Generate clean/continued/final scripts once, then submit one afterok chain per session so fast sessions advance      #
# without a global phase barrier. Failed phases cancel only that session's later jobs; retry_from_*.sh resumes from    #
# the failed phase without re-running completed earlier phases (pickles kept).                                          #
# Public: build_session_phase_script_map, write_chained_submit_script, write_all_retry_from_phase_scripts,               #
#         print_chain_status_hint, submit_slurm_script (optional auto-submit).                                           #
# ==================================================================================================================== #


def _subfn_bash_sbatch_chain_lines(phase_script_paths: List[Tuple[str, str]], indent: str = '') -> List[str]:
    """Shared by write_chained_submit_script / write_all_retry_from_phase_scripts: bash lines for an afterok phase chain."""
    if len(phase_script_paths) == 0:
        return []
    lines: List[str] = []
    first_phase, first_path = phase_script_paths[0]
    lines.append(f"{indent}echo \"Submitting {first_phase}: {first_path}\"")
    lines.append(f"{indent}jid=$(sbatch --parsable '{first_path}')")
    lines.append(f"{indent}echo \"  job $jid ({first_phase})\"")
    for phase_name, script_path in phase_script_paths[1:]:
        lines.append(f"{indent}echo \"Submitting {phase_name} afterok:$jid: {script_path}\"")
        lines.append(f"{indent}jid=$(sbatch --parsable --dependency=afterok:$jid '{script_path}')")
        lines.append(f"{indent}echo \"  job $jid ({phase_name})\"")
    ## END for phase_name, script_path in phase_script_paths[1:]...
    return lines


def _subfn_bash_direct_run_lines(phase_script_paths: List[Tuple[str, str]], indent: str = '') -> List[str]:
    """Shared by write_chained_submit_script / write_all_retry_from_phase_scripts: sequential bash run lines (Lab linux_standalone)."""
    if len(phase_script_paths) == 0:
        return []
    lines: List[str] = []
    for phase_name, script_path in phase_script_paths:
        lines.append(f'{indent}echo "Running {phase_name}: {script_path}"')
        lines.append(f"{indent}bash '{script_path}'")
    ## END for phase_name, script_path in phase_script_paths...
    return lines


def _subfn_phase_chain_lines(phase_script_paths: List[Tuple[str, str]], phase_chain_backend: str, indent: str = '') -> List[str]:
    if phase_chain_backend == 'bash':
        return _subfn_bash_direct_run_lines(phase_script_paths, indent=indent)
    if phase_chain_backend == 'slurm':
        return _subfn_bash_sbatch_chain_lines(phase_script_paths, indent=indent)
    raise ValueError(f"phase_chain_backend must be 'slurm' or 'bash', got {phase_chain_backend!r}")


def submit_slurm_script(slurm_script_path: Union[str, Path]) -> int:
    """Submit one Slurm script via `sbatch --parsable` and return the job id (optional auto-submit; generator writes chains instead)."""
    script_path = Path(slurm_script_path)
    result = subprocess.run(['sbatch', '--parsable', str(script_path)], check=True, capture_output=True, text=True)
    job_id_str = result.stdout.strip().split(';')[0].strip()
    if not job_id_str.isdigit():
        raise RuntimeError(f"Could not parse sbatch job ID from stdout={result.stdout!r} stderr={result.stderr!r}")
    job_id = int(job_id_str)
    print(f"Submitted job {job_id}: {script_path}")
    return job_id


def build_session_phase_script_map(phase_collections: Dict[str, 'BatchScriptsCollection'], script_kind: str = 'run', script_collection_key: Literal['slurm', 'bash'] = 'slurm') -> Dict[str, Dict[str, str]]:
    """Align per-phase BatchScriptsCollection run/fig script paths by session.

    phase_collections: mapping phase_name -> BatchScriptsCollection from generate_batch_single_session_scripts.
    script_collection_key: 'slurm' -> output_slurm_scripts; 'bash' -> output_non_slurm_bash_scripts.
    Returns: session_key -> {phase_name: script_path_str}
    """
    def _subfn_session_key_str(curr_session_context: Union[IdentifyingContext, str]) -> str:
        if isinstance(curr_session_context, IdentifyingContext):
            return curr_session_context.get_description()
        return str(curr_session_context)

    session_phase_scripts: Dict[str, Dict[str, str]] = {}
    for phase_name, collection in phase_collections.items():
        contexts = list(collection.included_session_contexts or [])
        if script_collection_key == 'slurm':
            script_paths = list((collection.output_slurm_scripts or {}).get(script_kind, []) or [])
            paths_label = 'output_slurm_scripts'
        elif script_collection_key == 'bash':
            script_paths = list((collection.output_non_slurm_bash_scripts or {}).get(script_kind, []) or [])
            paths_label = 'output_non_slurm_bash_scripts'
        else:
            raise ValueError(f"script_collection_key must be 'slurm' or 'bash', got {script_collection_key!r}")
        if len(contexts) != len(script_paths):
            raise ValueError(f"phase={phase_name!r}: len(included_session_contexts)={len(contexts)} != len({paths_label}[{script_kind!r}])={len(script_paths)}")
        for ctx, script_path in zip(contexts, script_paths):
            session_key = _subfn_session_key_str(ctx)
            session_phase_scripts.setdefault(session_key, {})[phase_name] = str(Path(script_path))
        ## END for ctx, script_path in zip(contexts, script_paths)...
    ## END for phase_name, collection in phase_collections.items()...
    return session_phase_scripts


def write_chained_submit_script(session_phase_scripts: Dict[str, Dict[str, str]], out_path: Union[str, Path], phases_ordered: List[str],
                                manifest_path: Optional[Union[str, Path]] = None, batch_date_to_use: Optional[str] = None, phase_chain_backend: Literal['slurm', 'bash'] = 'slurm') -> Tuple[Path, Path]:
    """Write a bash script that runs or submits per-session phase chains, plus a JSON manifest of script paths.

    phase_chain_backend: 'slurm' submits afterok chains (sessions in parallel on the cluster); 'bash' runs one session
    at a time with sequential phase scripts (Lab linux_standalone).

    Returns (submit_script_path, manifest_path).
    """
    out_path = Path(out_path)
    if manifest_path is None:
        manifest_path = out_path.with_suffix('.json')
    else:
        manifest_path = Path(manifest_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if phase_chain_backend == 'bash':
        bash_lines: List[str] = [
            '#!/bin/bash',
            '# Per-session sequential phase chains (linux_standalone; no Slurm).',
            '# One session at a time; within a session, phases run in order. A failed phase skips later phases for that session.',
            '# Retry a failed phase with retry_from_*.sh — do not re-run completed earlier phases.',
            'set -uo pipefail',
            '',
        ]
    else:
        bash_lines = [
            '#!/bin/bash',
            '# Per-session Slurm phase chains (dependency=afterok).',
            '# A failed phase cancels only that session\'s later phases; earlier successful pickles are kept.',
            '# Retry a failed phase with retry_from_*.sh — do not re-run completed earlier phases.',
            'set -euo pipefail',
            '',
        ]
    if batch_date_to_use:
        bash_lines.append(f'# batch: {batch_date_to_use}')
        bash_lines.append('')

    for session_key, phase_paths in session_phase_scripts.items():
        ordered: List[Tuple[str, str]] = []
        for phase_name in phases_ordered:
            if phase_name not in phase_paths:
                continue
            ordered.append((phase_name, phase_paths[phase_name]))
        ## END for phase_name in phases_ordered...
        if len(ordered) == 0:
            continue
        bash_lines.append(f'echo "=== session: {session_key} ==="')
        if phase_chain_backend == 'bash':
            bash_lines.append('(')
            bash_lines.append('  set -e')
            bash_lines.extend(_subfn_phase_chain_lines(ordered, phase_chain_backend='bash', indent='  '))
            bash_lines.append(f') || echo "Session failed (later phases skipped for this session): {session_key}"')
        else:
            bash_lines.extend(_subfn_phase_chain_lines(ordered, phase_chain_backend='slurm', indent=''))
        bash_lines.append('')
    ## END for session_key, phase_paths in session_phase_scripts.items()...

    if phase_chain_backend == 'bash':
        bash_lines.append('echo "All session phase chains completed."')
    else:
        bash_lines.append('echo "All session phase chains submitted."')
    out_path.write_text('\n'.join(bash_lines) + '\n', encoding='utf-8')
    try:
        out_path.chmod(out_path.stat().st_mode | 0o111)
    except OSError:
        pass

    manifest: Dict[str, Any] = {
        'batch_date_to_use': batch_date_to_use,
        'phases_ordered': list(phases_ordered),
        'execution_backend': phase_chain_backend,
        'sessions': session_phase_scripts,
        'submit_script': str(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote chained submit script: {out_path}')
    print(f'Wrote phase-chain manifest: {manifest_path}')
    return out_path, manifest_path


def write_all_retry_from_phase_scripts(manifest: Union[Dict[str, Any], str, Path], start_phase: str,
                                       out_dir: Union[str, Path], phases_ordered: Optional[List[str]] = None) -> List[Path]:
    """Write one retry_from_<start_phase>_<session>.sh per session under out_dir."""
    def _subfn_write_retry_from_phase_script(manifest_dict: Dict[str, Any], session_key: str, start_phase_name: str,
                                             retry_out_path: Path, phases_ordered_list: List[str]) -> Path:
        """Write a bash script that re-chains from start_phase onward for one session (skips completed earlier phases)."""
        sessions = manifest_dict.get('sessions', {})
        if session_key not in sessions:
            raise KeyError(f"session_key={session_key!r} not in manifest sessions ({list(sessions.keys())})")
        phase_paths = sessions[session_key]
        if start_phase_name not in phases_ordered_list:
            raise ValueError(f"start_phase={start_phase_name!r} not in phases_ordered={phases_ordered_list}")
        start_idx = phases_ordered_list.index(start_phase_name)
        remaining = phases_ordered_list[start_idx:]
        ordered: List[Tuple[str, str]] = []
        for phase_name in remaining:
            if phase_name not in phase_paths:
                continue
            ordered.append((phase_name, phase_paths[phase_name]))
        ## END for phase_name in remaining...
        if len(ordered) == 0:
            raise ValueError(f"No scripts found for session={session_key!r} from phase={start_phase_name!r}")

        execution_backend = str(manifest_dict.get('execution_backend') or 'slurm')
        retry_out_path.parent.mkdir(parents=True, exist_ok=True)
        if execution_backend == 'bash':
            bash_lines = [
                '#!/bin/bash',
                f'# Retry session {session_key} from phase {start_phase_name} (sequential bash).',
                '# Earlier phases are intentionally not re-run; their pickles remain on disk.',
                'set -euo pipefail',
                '',
                f'echo "=== retry session: {session_key} from {start_phase_name} ==="',
            ]
            bash_lines.extend(_subfn_phase_chain_lines(ordered, phase_chain_backend='bash', indent=''))
            bash_lines.append('echo "Retry chain completed."')
        else:
            bash_lines = [
                '#!/bin/bash',
                f'# Retry session {session_key} from phase {start_phase_name} (afterok chain).',
                '# Earlier phases are intentionally not re-submitted; their pickles remain on disk.',
                'set -euo pipefail',
                '',
                f'echo "=== retry session: {session_key} from {start_phase_name} ==="',
            ]
            bash_lines.extend(_subfn_phase_chain_lines(ordered, phase_chain_backend='slurm', indent=''))
            bash_lines.append('echo "Retry chain submitted."')
        retry_out_path.write_text('\n'.join(bash_lines) + '\n', encoding='utf-8')
        try:
            retry_out_path.chmod(retry_out_path.stat().st_mode | 0o111)
        except OSError:
            pass
        print(f'Wrote retry script: {retry_out_path}')
        return retry_out_path

    if isinstance(manifest, (str, Path)):
        with open(Path(manifest), 'r', encoding='utf-8') as f:
            manifest_dict = json.load(f)
    else:
        manifest_dict = manifest
    if phases_ordered is None:
        phases_ordered = list(manifest_dict.get('phases_ordered') or [])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for session_key in (manifest_dict.get('sessions') or {}).keys():
        safe_session = re.sub(r'[^A-Za-z0-9._-]+', '_', session_key)
        retry_path = out_dir / f'retry_from_{start_phase}_{safe_session}.sh'
        written.append(_subfn_write_retry_from_phase_script(manifest_dict, session_key=session_key, start_phase_name=start_phase, retry_out_path=retry_path, phases_ordered_list=phases_ordered))
    ## END for session_key in (manifest_dict.get('sessions') or {}).keys()...
    return written


def print_chain_status_hint(job_ids: Optional[List[int]] = None, execution_backend: Literal['slurm', 'bash'] = 'slurm') -> None:
    """Print monitoring hints for chained phase jobs (Slurm) or sequential bash runs (Lab)."""
    if execution_backend == 'bash':
        print('Monitor linux_standalone phase chains:')
        print('  Watch the terminal running submit_phase_chains_*.sh, or tail per-session log files next to each run_*.sh.')
        print('On failure: identify the failed phase, then bash retry_from_<phase>_<session>.sh (earlier pickles kept).')
        return
    print('Monitor chained phase jobs:')
    print('  squeue -u $USER')
    if job_ids:
        job_id_str = ','.join(str(j) for j in job_ids)
        print(f'  squeue -j {job_id_str}')
        print(f'  sacct -j {job_id_str} --format=JobID,JobName%40,State,ExitCode,Elapsed -P')
    else:
        print('  sacct -u $USER --starttime=now-2days --format=JobID,JobName%40,State,ExitCode,Elapsed -P')
    print('On failure: identify the failed phase, then bash retry_from_<phase>_<session>.sh (earlier pickles kept).')

# END Batch process phase chaining ___________________________________________________________________________________ #


@function_attributes(short_name=None, tags=['slurm','jobs','files','batch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-09 19:14', related_items=[])
def generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs: Dict[IdentifyingContext, Path], included_session_contexts: Optional[List[IdentifyingContext]], output_directory='output/gen_scripts/', use_separate_run_directories:bool=True,
         create_slurm_scripts:bool=False, create_non_slurm_bash_scripts:bool=False, should_create_vscode_workspace:bool=True, should_use_neptune_logging:bool=True, should_use_viztracer_logging:bool=True, should_generate_run_scripts = True, should_generate_figure_scripts = True, should_generate_run_notebooks: bool = False,
         job_suffix:Optional[str]=None, should_use_file_redirected_output_logging:bool=False, should_use_largemem:bool=False, fail_on_exception: bool = False, # , should_create_powershell_scripts:bool=True
         separate_execute_and_figure_gen_scripts:bool=True, should_perform_figure_generation_to_file:bool=False, force_recompute_override_computations_includelist: Optional[List[str]]=None, force_recompute_override_computation_kwargs_dict: Optional[Dict[str, Dict]]=None, 
         custom_user_completion_function_override_kwargs_dict: Optional[Dict]=None,
         batch_session_completion_handler_kwargs: Optional[Dict]=None, batch_script_subdirectory: Optional[str]=None, venv_activate_path: Optional[str]=None, vscode_default_interpreter_path: Optional[Union[str, Path]]=None, session_global_data_root_parent_paths: Optional[Dict[IdentifyingContext, Path]]=None, **renderer_script_generation_kwargs) -> BatchScriptsCollection:
    """ Creates a series of standalone scripts (one for each included_session_contexts) in the `output_directory`

    output_directory
    use_separate_run_directories:bool = True - If True, separate directories are made in `output_directory` containing each script for all sessions.

    included_session_contexts
    session_batch_basedirs: Dict[IdentifyingContext, Path]

    batch_session_completion_handler_kwargs: Optional[Dict] - the values to be passed to batch_session_completion_handler
    custom_user_completion_function_override_kwargs_dict - the kwarg overrides for the user_computation_functions
    vscode_default_interpreter_path: Optional[Union[str, Path]] - Python executable written to generated VSCode workspace `python.defaultInterpreterPath`. When None, uses platform defaults.
    
    
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
    def _subfn_build_slurm_script(curr_batch_script_rundir, a_python_script_path, a_curr_session_context, a_curr_session_complete_identifier, a_slurm_script_name_prefix:str='run', should_use_virtual_framebuffer:bool=False, should_use_largemem:bool=False, job_suffix=None, a_venv_activate_path: Optional[str]=None):
        slurm_script_path = os.path.join(curr_batch_script_rundir, f'{a_slurm_script_name_prefix}_{a_curr_session_complete_identifier}.sh')
        slurm_job_name = build_slurm_job_name(curr_session_context=a_curr_session_context, job_suffix=job_suffix)
        with open(slurm_script_path, 'w') as script_file:
            script_content = slurm_template.render(curr_session_context=f"{a_curr_session_context}", slurm_job_name=slurm_job_name, python_script_path=a_python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, should_use_largemem=should_use_largemem, should_use_virtual_framebuffer=should_use_virtual_framebuffer, venv_activate_path=a_venv_activate_path)
            script_file.write(script_content)
        return slurm_script_path
    
    def _subfn_build_non_slurm_bash_script(curr_batch_script_rundir, a_python_script_path, a_curr_session_context, a_curr_session_complete_identifier, a_bash_script_name_prefix:str='run', a_venv_activate_path: Optional[str]=None):
        bash_script_path = os.path.join(curr_batch_script_rundir, f'{a_bash_script_name_prefix}_{a_curr_session_complete_identifier}.sh')
        with open(bash_script_path, 'w') as script_file:
            script_content = bash_non_slurm_template.render(curr_session_context=f"{a_curr_session_context}", python_script_path=a_python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, venv_activate_path=a_venv_activate_path)
            script_file.write(script_content)
        return bash_script_path


    # ==================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                  #
    # ==================================================================================================================== #
    assert isinstance(session_batch_basedirs, dict)

    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory).resolve()

    # if job_suffix is None:
    #     job_suffix = ''
        
    # job_suffix = f"{job_suffix}_{}"

    separate_execute_and_figure_gen_scripts = renderer_script_generation_kwargs.pop('separate_execute_and_figure_gen_scripts', True)
    assert separate_execute_and_figure_gen_scripts, f"Old non-separate mode not supported"
 
    assert ('force_recompute_override_computations_includelist' not in renderer_script_generation_kwargs), f"pass 'force_recompute_override_computations_includelist' explicitly to the call!!"
    renderer_script_generation_kwargs['force_recompute_override_computations_includelist'] = (force_recompute_override_computations_includelist or [])
 
    assert ('force_recompute_override_computation_kwargs_dict' not in renderer_script_generation_kwargs), f"pass 'force_recompute_override_computation_kwargs_dict' explicitly to the call!!"
    renderer_script_generation_kwargs['force_recompute_override_computation_kwargs_dict'] = (force_recompute_override_computation_kwargs_dict or {})
    
    # if script_generation_kwargs is None:
    # 	script_generation_kwargs = dict(should_force_reload_all=False, should_perform_figure_generation_to_file=False)

    no_recomputing_script_generation_kwargs = dict(should_force_reload_all=False, should_freeze_pipeline_updates=True, should_perform_figure_generation_to_file=should_perform_figure_generation_to_file) | renderer_script_generation_kwargs # No recomputing at all:
    compute_as_needed_script_generation_kwargs = dict(should_force_reload_all=False, should_freeze_pipeline_updates=False, should_save_local_pipeline_pkl=True, should_perform_figure_generation_to_file=should_perform_figure_generation_to_file) | renderer_script_generation_kwargs
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

    if venv_activate_path is None:
        _python_executable = Path(sys.executable).resolve()
        _candidate_activate = _python_executable.parent / 'activate'
        # venv_activate_path = str(_candidate_activate) if _candidate_activate.exists() else '/home/halechr/repos/Spike3D/.venv/bin/activate'
        venv_activate_path = str(_candidate_activate) if _candidate_activate.exists() else '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate'


    output_python_scripts = []
    output_jupyter_notebooks = []
    

    output_slurm_scripts = {'run': [], 'figs': []}
    output_non_slurm_bash_scripts = {'run': [], 'figs': []}

    ## INPUTS: job_suffix, 

    # Make sure the output directories exist
    os.makedirs(output_directory, exist_ok=True)
    batch_scripts_root_directory = resolve_batch_scripts_root_directory(output_directory=output_directory, batch_script_subdirectory=batch_script_subdirectory)
    batch_scripts_root_directory.mkdir(parents=True, exist_ok=True)
    
    for curr_session_context in included_session_contexts:
        if curr_session_context not in session_batch_basedirs:
            print(f'WARN: skipping script generation for {curr_session_context}: no session basedir in session_batch_basedirs')
            continue
        curr_session_basedir = session_batch_basedirs[curr_session_context]
        curr_global_data_root_parent_path = (session_global_data_root_parent_paths or {}).get(curr_session_context, global_data_root_parent_path)
        if (job_suffix is not None) and (len(job_suffix) > 0):
            curr_session_complete_identifier: str = f"{curr_session_context}_{job_suffix}"
        else:
            curr_session_complete_identifier: str = f"{curr_session_context}"
                    
        print(F'curr_item_name: "{curr_session_complete_identifier}"')

        if use_separate_run_directories:
            curr_batch_script_rundir = os.path.join(output_directory, f"run_{curr_session_context}") # "gen_scripts/run_kdiba_gor01_one_2006-6-07_11-26-53/"
            if batch_script_subdirectory is not None:
                curr_batch_script_rundir = os.path.join(curr_batch_script_rundir, batch_script_subdirectory)
            os.makedirs(curr_batch_script_rundir, exist_ok=True)
        else:
            curr_batch_script_rundir = output_directory
            if batch_script_subdirectory is not None:
                curr_batch_script_rundir = os.path.join(curr_batch_script_rundir, batch_script_subdirectory)
                os.makedirs(curr_batch_script_rundir, exist_ok=True)

        rs_kwargs_base = dict(renderer_script_generation_kwargs.get('run_specific_batch_kwargs') or {})
        run_specific_batch_kwargs_for_run_script = rs_kwargs_base | {'preflight_bapun_batch_helpers_run_all': (getattr(curr_session_context, 'format_name', None) == 'bapun')}
        run_specific_batch_kwargs_for_figures_script = rs_kwargs_base | {'preflight_bapun_batch_helpers_run_all': False}

        # Create two separate scripts:
        # Run Script _________________________________________________________________________________________________________ #
        python_script_path = os.path.join(curr_batch_script_rundir, f'run_{curr_session_complete_identifier}.py') # "run_kdiba_gor01_one_2006-6-07_11-26-53__withNormalComputedReplays-qclu_12-frateThresh_5.0_tbin_25ms_Clean.py"
        with open(python_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=curr_global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_complete_identifier=curr_session_complete_identifier,
                                                    job_suffix=job_suffix,
                                                    curr_session_basedir=curr_session_basedir, 
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    custom_user_completion_function_override_kwargs_dict=(custom_user_completion_function_override_kwargs_dict or {}),
                                                    should_use_neptune_logging=should_use_neptune_logging, should_use_viztracer_logging=should_use_viztracer_logging, should_use_file_redirected_output_logging=should_use_file_redirected_output_logging,
                                                    fail_on_exception=fail_on_exception,
                                                    **(compute_as_needed_script_generation_kwargs | dict(should_perform_figure_generation_to_file=False)),
                                                    run_specific_batch_kwargs=run_specific_batch_kwargs_for_run_script)
            # script_file.write(script_content)
            script_file.write(script_content.encode())
        # output_python_scripts.append(python_script_path)

        # Figures Script _____________________________________________________________________________________________________ #
        python_figures_script_path = os.path.join(curr_batch_script_rundir, f'figures_{curr_session_complete_identifier}.py')
        with open(python_figures_script_path, 'wb') as script_file:
            script_content = python_template.render(global_data_root_parent_path=curr_global_data_root_parent_path,
                                                    curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                    curr_session_complete_identifier=curr_session_complete_identifier,
                                                    job_suffix=job_suffix,
                                                    curr_session_basedir=curr_session_basedir,
                                                    batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                    custom_user_completion_function_override_kwargs_dict=(custom_user_completion_function_override_kwargs_dict or {}),
                                                    should_use_neptune_logging=should_use_neptune_logging, should_use_viztracer_logging=should_use_viztracer_logging, should_use_file_redirected_output_logging=should_use_file_redirected_output_logging,
                                                    **(no_recomputing_script_generation_kwargs | dict(should_perform_figure_generation_to_file=True)),
                                                    run_specific_batch_kwargs=run_specific_batch_kwargs_for_figures_script)


            # script_file.write(script_content)
            script_file.write(script_content.encode())

        # output_python_display_scripts.append(python_figures_script_path)
        output_python_scripts.append((python_script_path, python_figures_script_path))

        # Create the SLURM script
        if create_slurm_scripts:
            if should_generate_run_scripts:
                slurm_run_script_path = _subfn_build_slurm_script(a_python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_curr_session_complete_identifier=curr_session_complete_identifier, a_slurm_script_name_prefix='run', should_use_virtual_framebuffer=False, should_use_largemem=should_use_largemem, job_suffix=job_suffix, a_venv_activate_path=venv_activate_path)
                output_slurm_scripts['run'].append(slurm_run_script_path)

            if should_generate_figure_scripts or should_perform_figure_generation_to_file:
                slurm_figure_script_path = _subfn_build_slurm_script(a_python_script_path=python_figures_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_curr_session_complete_identifier=curr_session_complete_identifier, a_slurm_script_name_prefix='figs', should_use_virtual_framebuffer=False, should_use_largemem=should_use_largemem, job_suffix=job_suffix, a_venv_activate_path=venv_activate_path)
                output_slurm_scripts['figs'].append(slurm_figure_script_path)


        ## Create the non-slurm bash script:
        if create_non_slurm_bash_scripts:
            if should_generate_run_scripts:
                bash_run_script_path = _subfn_build_non_slurm_bash_script(a_python_script_path=python_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_curr_session_complete_identifier=curr_session_complete_identifier, a_bash_script_name_prefix='run', a_venv_activate_path=venv_activate_path)
                output_non_slurm_bash_scripts['run'].append(bash_run_script_path)

            if should_generate_figure_scripts or should_perform_figure_generation_to_file:
                bash_figure_script_path = _subfn_build_non_slurm_bash_script(a_python_script_path=python_figures_script_path, curr_batch_script_rundir=curr_batch_script_rundir, a_curr_session_context=curr_session_context, a_curr_session_complete_identifier=curr_session_complete_identifier, a_bash_script_name_prefix='figs', a_venv_activate_path=venv_activate_path)
                output_non_slurm_bash_scripts['figs'].append(bash_figure_script_path)


        if should_generate_run_notebooks:
            script_path = Path(python_script_path).resolve()
            _temp_notebook_python_script_path = Path(os.path.join(curr_batch_script_rundir, f'_TEMP_NOTEBOOK_run_{curr_session_context}.py'))
            with open(_temp_notebook_python_script_path, 'wb') as script_file:
                script_content = python_template.render(global_data_root_parent_path=curr_global_data_root_parent_path,
                                                        curr_session_context=curr_session_context.get_initialization_code_string().strip("'"),
                                                        curr_session_basedir=curr_session_basedir, 
                                                        batch_session_completion_handler_kwargs=(batch_session_completion_handler_kwargs or {}),
                                                        custom_user_completion_function_override_kwargs_dict=(custom_user_completion_function_override_kwargs_dict or {}),
                                                        # should_use_neptune_logging=should_use_neptune_logging, should_use_file_redirected_output_logging=should_use_file_redirected_output_logging,
                                                        should_use_neptune_logging=False, should_use_file_redirected_output_logging=False,
                                                        fail_on_exception=fail_on_exception,
                                                        **(compute_as_needed_script_generation_kwargs | dict(should_perform_figure_generation_to_file=False)),
                                                        run_specific_batch_kwargs=run_specific_batch_kwargs_for_run_script)
                # script_file.write(script_content)
                script_file.write(script_content.encode())


            # script_dir = script_path.parent.resolve()
            notebook_path = script_path.with_suffix('.ipynb')
            convert_script_to_notebook(_temp_notebook_python_script_path, notebook_path, enable_auto_reload=False)
            ## remove temporary script when done
            try:
                _temp_notebook_python_script_path.unlink()
                print(f"File {_temp_notebook_python_script_path} has been deleted successfully.")
            except FileNotFoundError:
                print(f"File {_temp_notebook_python_script_path} does not exist.")
            except PermissionError:
                print(f"Permission denied: Unable to delete {_temp_notebook_python_script_path}.")
            except Exception as e:
                print(f"An error occurred while deleting the file: {e}")

            output_jupyter_notebooks.append(notebook_path)
             
        ## END if should_generate_run_notebooks..
            
    ## end for curr_session_context in include....
    
    ## Generate VSCode Workspace for it
    if should_create_vscode_workspace:
        output_compute_python_scripts = [x[0] for x in output_python_scripts]
        if vscode_default_interpreter_path is not None:
            python_executable = Path(vscode_default_interpreter_path).expanduser()
        elif os.name == 'nt':
            python_executable = Path('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv_UV/Scripts/python').resolve()
        else:
            python_executable = Path('~/repos/Spike3D/.venv/bin/python').expanduser().resolve()

        vscode_workspace_path = build_vscode_workspace(output_compute_python_scripts, python_executable=python_executable, batch_scripts_root_directory=batch_scripts_root_directory)
        print(f'vscode_workspace_path: {vscode_workspace_path}')
    else:
        vscode_workspace_path = None


    return BatchScriptsCollection(included_session_contexts=included_session_contexts, output_python_scripts=output_python_scripts, output_jupyter_notebooks=output_jupyter_notebooks, output_slurm_scripts=output_slurm_scripts, output_non_slurm_bash_scripts=output_non_slurm_bash_scripts, vscode_workspace_path=vscode_workspace_path, batch_scripts_root_directory=batch_scripts_root_directory)
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
def build_vscode_workspace(script_paths, python_executable: Optional[Union[str, Path]]=None, batch_scripts_root_directory: Optional[Path]=None):
    """ builds a VSCode workspace for the batch python scripts
    
        from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import build_vscode_workspace

        vscode_workspace_path = build_vscode_workspace(script_paths, python_executable='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/python')
        vscode_workspace_path

    """
    from jinja2 import Template

    if python_executable is None:
        active_venv_path, python_executable, activate_script_path = get_running_python()
    else:
        python_executable = Path(python_executable).expanduser()

    is_platform_windows: bool = False
    if (platform.system() == 'Windows'):
        is_platform_windows = True
    else:
        is_platform_windows = False

    assert len(script_paths) > 0, f"script_paths is empty!"
    top_level_script_folders_path: Path = Path(batch_scripts_root_directory).resolve() if batch_scripts_root_directory is not None else _infer_batch_scripts_root_directory(script_paths)
    top_level_script_folders_path.mkdir(parents=True, exist_ok=True)
    script_folders: List[Path] = [top_level_script_folders_path] + [Path(a_path).parent.resolve() for a_path in script_paths]
    print(f'script_folders: {script_folders}')
    # {
    #     "path": "L:/Scratch/gen_scripts",
    #     "name": "gen_scripts_root"
    # },
    vscode_workspace_path = top_level_script_folders_path.joinpath('run_workspace.code-workspace').resolve()
    # print(f'vscode_workspace_path: {vscode_workspace_path}')

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
                                        activate_path=r'H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\Scripts\activate.ps1',
                                        python_executable=r'H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\Scripts\python.exe',
                                        script_name: str = 'run_scripts', batch_scripts_root_directory: Optional[Path]=None):
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
    top_level_script_folders_path = Path(batch_scripts_root_directory).resolve() if batch_scripts_root_directory is not None else _infer_batch_scripts_root_directory(script_paths)
    top_level_script_folders_path.mkdir(parents=True, exist_ok=True)
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



# ==================================================================================================================== #
# Greatlakes Turbo -> Scratch workaround                                                                               #
# ==================================================================================================================== #
# @metadata_attributes(short_name=None, tags=['greatlakes', 'GL', 'BATCH', 'workaround', 'filesystem', 'turbo', 'scratch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-22 13:05', related_items=[])
class GreatlakesTurboToScratchHelpers:
    """ Temporary workaround for when Turbo (`/nfs/turbo/umms-kdiba`) is read-only on Greatlakes: stage the few mutable session outputs onto `/scratch` and retarget batch-script generation to write there instead.

    Self-contained (all classmethods, no module-level helpers) so that the calling batch script can drop/comment out the whole thing once Turbo is R/W again.

    Usage (from `ProcessBatchOutputs_*.ipy`):

        from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import GreatlakesTurboToScratchHelpers

        FORCE_SCRATCH_WRITES: bool = True  # set False when Turbo is writable again
        turbo_data_root, scratch_data_root, use_scratch_writes = GreatlakesTurboToScratchHelpers.detect_scratch_write_mode(force_scratch_writes=FORCE_SCRATCH_WRITES)
        if use_scratch_writes:
            scripts_output_path, collected_outputs_path = GreatlakesTurboToScratchHelpers.build_scratch_output_paths()
        ...
        if FORCE_SCRATCH_WRITES:
            GreatlakesTurboToScratchHelpers.assert_output_paths_not_on_turbo(scripts_output_path, collected_outputs_path)
        ...
        if use_scratch_writes:
            good_session_concrete_folders, session_global_data_root_parent_paths, global_data_root_parent_path = GreatlakesTurboToScratchHelpers.stage_sessions_for_scratch_run(turbo_session_folders=good_session_concrete_folders, source_data_root=global_data_root_parent_path, pickle_suffix=..., tbin_suffix=...)

    """
    TURBO_DATA_ROOT: Path = Path('/nfs/turbo/umms-kdiba/Data')
    SCRATCH_DATA_ROOT: Path = Path('/scratch/kdiba_root/kdiba99/halechr/Data')
    SCRATCH_ANCHOR_PATH: Path = Path('/scratch')  # GL indicator; Data/Output dirs may need creating
    TURBO_PATH_MARKERS: Tuple[str, ...] = ('/nfs/turbo/', '/cloud/turbo/')

    ## Tiny read-only session-root sidecars that post-load code re-reads next to the pickle (e.g. `FINAL_FIX_GRID_BIN_BOUNDS` -> `{session}.position_info.mat`). Never .dat/.eeg/.clu/.fet, spikes.mat/spikeII.mat, .h5, or unused pickles.
    READONLY_SESSION_SIDECAR_SUFFIXES: Tuple[str, ...] = ('.position_info.mat', '.epochs_info.mat', '.xml',
                                                          '.laps_info.mat', '.replay_info.mat', '.RippleDatabase.mat',
                                                          '.probegroup.npy', '.pbe.npy', '.ripple.npy', '.mua.npy', '.position.npy')
    READONLY_SESSION_SIDECAR_STANDALONE_NAMES: Tuple[str, ...] = ('RippleDatabase.mat',)  # folder listing uses the bare name; loader uses `{session}.RippleDatabase.mat`
    REQUIRED_SESSION_SIDECAR_SUFFIXES: Tuple[str, ...] = ('.position_info.mat',)  # FINAL post-load asserts these exist


    @classmethod
    def kdiba_session_relpath(cls, context: IdentifyingContext) -> Path:
        return Path('KDIBA').joinpath(context.animal, context.exper_name, context.session_name)


    @classmethod
    def build_scratch_stage_relpaths(cls, pickle_suffix: str, tbin_suffix: str = '') -> Tuple[str, str]:
        """Exact load-only local pickle and writable global pickle. No h5, no wildcards."""
        pickle_suffix = pickle_suffix or ''
        tbin_suffix = tbin_suffix or ''
        load_only_relpath: str = f'loadedSessPickle{pickle_suffix}.pkl'
        writable_relpath: str = f'output/global_computation_results{pickle_suffix}{tbin_suffix}.pkl'
        return load_only_relpath, writable_relpath


    @classmethod
    def find_turbo_session_basedir(cls, a_folder: ConcreteSessionFolder, source_data_root: Path) -> Path:
        """Prefer Data/KDIBA/...; fall back to the sibling /KDIBA tree that Data/KDIBA often symlinks to. Do not resolve()."""
        rel: Path = cls.kdiba_session_relpath(a_folder.context)
        candidates: List[Path] = [Path(source_data_root).joinpath(rel), Path(a_folder.path), Path(source_data_root).parent.joinpath(rel), cls.TURBO_DATA_ROOT.parent.joinpath(rel)]
        for a_candidate in candidates:
            if a_candidate.exists():
                return a_candidate
        ## END for a_candidate in candidates...

        return candidates[0]


    @classmethod
    def ensure_scratch_kdiba_layout_symlink(cls, scratch_data_root: Path) -> Path:
        """One folder symlink matching Turbo: sibling KDIBA -> Data/KDIBA. Data/KDIBA stays a real writable directory."""
        scratch_data_root = Path(scratch_data_root)
        scratch_kdiba: Path = scratch_data_root.joinpath('KDIBA')
        scratch_kdiba.mkdir(parents=True, exist_ok=True)
        sibling_kdiba: Path = scratch_data_root.parent.joinpath('KDIBA')
        if sibling_kdiba.is_symlink():
            current_target = Path(os.readlink(str(sibling_kdiba)))
            if current_target.resolve() != scratch_kdiba.resolve():
                sibling_kdiba.unlink()
                sibling_kdiba.symlink_to(scratch_kdiba, target_is_directory=True)
                print(f'[SCRATCH STAGE] KDIBA symlink: {sibling_kdiba} -> {scratch_kdiba}')
            else:
                print(f'[SCRATCH STAGE] KDIBA symlink exists: {sibling_kdiba} -> {scratch_kdiba}')
        elif sibling_kdiba.exists():
            print(f'[SCRATCH STAGE] KDIBA sibling already exists (not a symlink): {sibling_kdiba}')
        else:
            sibling_kdiba.symlink_to(scratch_kdiba, target_is_directory=True)
            print(f'[SCRATCH STAGE] KDIBA symlink: {sibling_kdiba} -> {scratch_kdiba}')
        return scratch_kdiba


    @classmethod
    def cleanup_leftover_scratch_symlinks(cls, scratch_basedir: Path) -> int:
        """Unlink symlinks left by the earlier per-file linking run (session dir + output/). Real files are kept."""
        n_removed: int = 0
        scratch_basedir = Path(scratch_basedir)
        if not scratch_basedir.exists():
            return n_removed
        candidate_dirs: List[Path] = [scratch_basedir]
        output_dir: Path = scratch_basedir.joinpath('output')
        if output_dir.is_symlink():
            output_dir.unlink()
            n_removed += 1
        elif output_dir.is_dir():
            candidate_dirs.append(output_dir)
        for a_dir in candidate_dirs:
            for an_item in a_dir.iterdir():
                if an_item.is_symlink():
                    an_item.unlink()
                    n_removed += 1
            ## END for an_item in a_dir.iterdir()...

        ## END for a_dir in candidate_dirs...

        return n_removed


    @classmethod
    def copy_or_skip_mutable_file(cls, src_file: Path, dest_file: Path, skip_if_dest_newer_or_equal: bool) -> str:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.is_symlink():
            dest_file.unlink()
        if skip_if_dest_newer_or_equal and dest_file.exists() and dest_file.is_file() and (not dest_file.is_symlink()):
            dest_stat = dest_file.stat()
            src_stat = src_file.stat()
            if dest_stat.st_mtime > src_stat.st_mtime:
                dest_mtime_str: str = datetime.fromtimestamp(dest_stat.st_mtime).isoformat(sep=' ', timespec='seconds')
                src_mtime_str: str = datetime.fromtimestamp(src_stat.st_mtime).isoformat(sep=' ', timespec='seconds')
                print(f'  WARN: dest mtime newer than source; skipping copy (size ignored) dest={dest_file} dest_mtime={dest_mtime_str} dest_size={dest_stat.st_size} src={src_file} src_mtime={src_mtime_str} src_size={src_stat.st_size}')
                return 'skipped_newer'
        shutil.copy2(src_file, dest_file)
        return 'copied'


    @classmethod
    def stage_mutable_session_outputs_to_scratch(cls, turbo_session_folders: List[ConcreteSessionFolder], source_data_root: Path, scratch_data_root: Path, load_only_relpath: str, writable_relpath: str, skip_if_dest_newer_or_equal: bool = True) -> Tuple[List[ConcreteSessionFolder], Dict[IdentifyingContext, Path], Dict[Path, Path]]:
        """Copy the local pickle, the writable global pickle, and the tiny read-only session-root sidecars (`READONLY_SESSION_SIDECAR_SUFFIXES`) as real files. Never symlink to Turbo, never stage .h5."""
        source_data_root = Path(source_data_root)
        scratch_data_root = Path(scratch_data_root)
        cls.ensure_scratch_kdiba_layout_symlink(scratch_data_root)
        scratch_session_folders: List[ConcreteSessionFolder] = []
        scratch_basedirs: Dict[IdentifyingContext, Path] = {}
        all_copied: Dict[Path, Path] = {}

        for a_folder in turbo_session_folders:
            turbo_basedir: Path = cls.find_turbo_session_basedir(a_folder, source_data_root=source_data_root)
            scratch_basedir: Path = scratch_data_root.joinpath(cls.kdiba_session_relpath(a_folder.context))
            scratch_basedir.mkdir(parents=True, exist_ok=True)
            n_cleaned: int = cls.cleanup_leftover_scratch_symlinks(scratch_basedir)
            scratch_basedir.joinpath('output').mkdir(parents=True, exist_ok=True)
            print(f'[SCRATCH STAGE] {a_folder.context}')
            print(f'  turbo:   {turbo_basedir}')
            print(f'  scratch: {scratch_basedir}')
            if n_cleaned > 0:
                print(f'  cleaned: {n_cleaned} leftover symlink(s)')
            if not turbo_basedir.exists():
                print(f'  WARN: turbo session basedir missing: {turbo_basedir}')

            load_src = turbo_basedir.joinpath(load_only_relpath)
            load_dest = scratch_basedir.joinpath(load_only_relpath)
            if not load_src.is_file():
                print(f'  missing load pickle: {load_src}')
            else:
                load_status = cls.copy_or_skip_mutable_file(load_src, load_dest, skip_if_dest_newer_or_equal=skip_if_dest_newer_or_equal)
                if load_status == 'copied':
                    all_copied[load_src] = load_dest
                    print(f'  copied local:  {load_src} -> {load_dest}')
                else:
                    print(f'  skipped local: {load_dest} (dest mtime newer)')

            write_src = turbo_basedir.joinpath(writable_relpath)
            write_dest = scratch_basedir.joinpath(writable_relpath)
            if not write_src.is_file():
                print(f'  missing global (ok, final_run will write): {write_dest}')
            else:
                status = cls.copy_or_skip_mutable_file(write_src, write_dest, skip_if_dest_newer_or_equal=skip_if_dest_newer_or_equal)
                if status == 'copied':
                    all_copied[write_src] = write_dest
                    print(f'  copied global: {write_src} -> {write_dest}')
                else:
                    print(f'  skipped global: {write_dest} (dest mtime newer)')

            ## Tiny read-only sidecars re-read next to the pickle by post-load code (e.g. `{session}.position_info.mat` in FINAL_FIX_GRID_BIN_BOUNDS):
            session_name: str = turbo_basedir.name  # matches `KDibaOldDataSessionFormat.get_session_name(basedir)`
            sidecar_names: List[str] = [f'{session_name}{a_suffix}' for a_suffix in cls.READONLY_SESSION_SIDECAR_SUFFIXES] + list(cls.READONLY_SESSION_SIDECAR_STANDALONE_NAMES)
            required_sidecar_names: List[str] = [f'{session_name}{a_suffix}' for a_suffix in cls.REQUIRED_SESSION_SIDECAR_SUFFIXES]
            n_sidecars_copied: int = 0
            n_sidecars_skipped: int = 0
            missing_sidecar_names: List[str] = []
            for a_sidecar_name in sidecar_names:
                sidecar_src = turbo_basedir.joinpath(a_sidecar_name)
                sidecar_dest = scratch_basedir.joinpath(a_sidecar_name)
                if not sidecar_src.is_file():
                    missing_sidecar_names.append(a_sidecar_name)
                    if a_sidecar_name in required_sidecar_names:
                        print(f'  WARN: required sidecar missing on turbo (FINAL post-load will fail without it): {sidecar_src}')
                    continue
                sidecar_status = cls.copy_or_skip_mutable_file(sidecar_src, sidecar_dest, skip_if_dest_newer_or_equal=skip_if_dest_newer_or_equal)
                if sidecar_status == 'copied':
                    all_copied[sidecar_src] = sidecar_dest
                    n_sidecars_copied += 1
                else:
                    n_sidecars_skipped += 1
            ## END for a_sidecar_name in sidecar_names...

            print(f'  sidecars: copied {n_sidecars_copied}, skipped {n_sidecars_skipped} (dest mtime newer), missing {len(missing_sidecar_names)}' + (f': {missing_sidecar_names}' if len(missing_sidecar_names) > 0 else ''))

            scratch_session_folders.append(ConcreteSessionFolder(a_folder.context, scratch_basedir))
            scratch_basedirs[a_folder.context] = scratch_basedir
        ## END for a_folder in turbo_session_folders...

        return scratch_session_folders, scratch_basedirs, all_copied


    @classmethod
    def detect_scratch_write_mode(cls, force_scratch_writes: bool, scratch_data_root: Optional[Path] = None, scratch_anchor_path: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path], bool]:
        """Decide whether this run should write to scratch. Returns `(turbo_data_root, scratch_data_root, use_scratch_writes)`; the roots are resolved when they exist, else None."""
        scratch_data_root = Path(scratch_data_root) if (scratch_data_root is not None) else cls.SCRATCH_DATA_ROOT
        scratch_anchor_path = Path(scratch_anchor_path) if (scratch_anchor_path is not None) else cls.SCRATCH_ANCHOR_PATH
        turbo_data_root: Optional[Path] = cls.TURBO_DATA_ROOT.resolve() if cls.TURBO_DATA_ROOT.exists() else None
        resolved_scratch_data_root: Optional[Path] = None
        use_scratch_writes: bool = False
        if force_scratch_writes and (platform.system() != 'Windows') and scratch_anchor_path.exists():
            try:
                scratch_data_root.mkdir(parents=True, exist_ok=True)
                resolved_scratch_data_root = scratch_data_root.resolve()
                use_scratch_writes = resolved_scratch_data_root.exists()
            except OSError as e:
                print(f'WARN: could not create scratch data root {scratch_data_root}: {e}')
                resolved_scratch_data_root = None
                use_scratch_writes = False
        return turbo_data_root, resolved_scratch_data_root, use_scratch_writes


    @classmethod
    def build_scratch_output_paths(cls, scratch_data_root: Optional[Path] = None) -> Tuple[Path, Path]:
        """Returns `(scripts_output_path, collected_outputs_path)` under `<scratch>/Output/{gen_scripts,collected_outputs}`, creating them if needed."""
        scratch_data_root = Path(scratch_data_root) if (scratch_data_root is not None) else cls.SCRATCH_DATA_ROOT
        scripts_output_path: Path = scratch_data_root.joinpath('Output', 'gen_scripts')
        scripts_output_path.mkdir(parents=True, exist_ok=True)
        collected_outputs_path: Path = scratch_data_root.joinpath('Output', 'collected_outputs')
        collected_outputs_path.mkdir(parents=True, exist_ok=True)
        return scripts_output_path, collected_outputs_path


    @classmethod
    def assert_output_paths_not_on_turbo(cls, scripts_output_path: Path, collected_outputs_path: Path) -> None:
        """Guard against accidentally generating scripts/outputs onto the read-only Turbo mount."""
        _scripts_path_str: str = str(scripts_output_path).replace('\\', '/')
        _collected_path_str: str = str(collected_outputs_path).replace('\\', '/')
        assert not any((a_marker in _scripts_path_str) for a_marker in cls.TURBO_PATH_MARKERS), f"FORCE_SCRATCH_WRITES is True but scripts_output_path is still on Turbo: {scripts_output_path}"
        assert not any((a_marker in _collected_path_str) for a_marker in cls.TURBO_PATH_MARKERS), f"FORCE_SCRATCH_WRITES is True but collected_outputs_path is still on Turbo: {collected_outputs_path}"
        print(f'[SCRATCH WRITES] gen_scripts -> {scripts_output_path}')
        print(f'[SCRATCH WRITES] collected_outputs -> {collected_outputs_path}')


    @classmethod
    def stage_sessions_for_scratch_run(cls, turbo_session_folders: List[ConcreteSessionFolder], source_data_root: Path, pickle_suffix: str, tbin_suffix: str = '', scratch_data_root: Optional[Path] = None, skip_if_dest_newer_or_equal: bool = True) -> Tuple[List[ConcreteSessionFolder], Dict[IdentifyingContext, Path], Path]:
        """Stage the sessions discovered on Turbo onto scratch and return the retargeted `(scratch_session_folders, session_global_data_root_parent_paths, scratch_write_root)` to feed `generate_batch_single_session_scripts`.
        `scratch_data_root=None` uses the unresolved `cls.SCRATCH_DATA_ROOT` as the write root (and as the parent for the sibling KDIBA symlink).
        """
        source_data_root = Path(source_data_root)
        scratch_write_root: Path = Path(scratch_data_root) if (scratch_data_root is not None) else cls.SCRATCH_DATA_ROOT
        load_only_relpath, writable_relpath = cls.build_scratch_stage_relpaths(pickle_suffix=pickle_suffix, tbin_suffix=tbin_suffix)
        print(f'scratch load-only pickle: {load_only_relpath}')
        print(f'scratch writable global pickle: {writable_relpath}')

        if source_data_root.resolve() == scratch_write_root.resolve():
            print(f'[SCRATCH STAGE] session discovery root is already scratch ({source_data_root}); skipping turbo copy.')
            scratch_session_folders: List[ConcreteSessionFolder] = list(turbo_session_folders)
        else:
            scratch_session_folders, _scratch_session_basedirs, staged_copied_files = cls.stage_mutable_session_outputs_to_scratch(turbo_session_folders=turbo_session_folders, source_data_root=source_data_root, scratch_data_root=scratch_write_root, load_only_relpath=load_only_relpath, writable_relpath=writable_relpath, skip_if_dest_newer_or_equal=skip_if_dest_newer_or_equal)
            print(f'[SCRATCH STAGE] copied {len(staged_copied_files)} file(s); retargeted global_data_root_parent_path -> {scratch_write_root}')
        session_global_data_root_parent_paths: Dict[IdentifyingContext, Path] = {a_folder.context: scratch_write_root for a_folder in scratch_session_folders}

        for a_turbo_folder, a_scratch_folder in zip(turbo_session_folders, scratch_session_folders):
            turbo_pickle = cls.find_turbo_session_basedir(a_turbo_folder, source_data_root=source_data_root).joinpath(load_only_relpath)
            scratch_pickle = a_scratch_folder.path.joinpath(load_only_relpath)
            if (not turbo_pickle.exists()) and (not scratch_pickle.exists()):
                print(f'WARN: required local pickle missing on turbo and scratch for {a_turbo_folder.context}: {load_only_relpath}')
                print(f'  turbo pickle:   {turbo_pickle}')
                print(f'  scratch pickle: {scratch_pickle}')
        ## END for a_turbo_folder, a_scratch_folder in zip(turbo_session_folders, scratch_session_folders)...

        print(f'global_data_root_parent_path (job write root): {scratch_write_root}')
        return scratch_session_folders, session_global_data_root_parent_paths, scratch_write_root


