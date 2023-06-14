from copy import deepcopy
from pathlib import Path
from typing import Callable
import numpy as np
from enum import unique # SessionBatchProgress
import traceback # for stack trace formatting

from attrs import define, Factory, fields

## MATPLOTLIB Imports:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends import backend_pdf


from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
from neuropy.utils.misc import compute_paginated_grid_config # for paginating shared aclus

# pyphocorehelpers
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # required for SessionBatchProgress
from pyphocorehelpers.function_helpers import function_attributes

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership # needed for batch_extended_computations, batch_programmatic_figures
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, session_context_to_relative_path, programmatic_display_to_PDF, programmatic_render_to_file
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_metadata_from_display_context # newer version of build_pdf_export_metadata
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline, PipelineSavingScheme # for batch_load_session

@unique
class SessionBatchProgress(ExtendedEnum):
    """Indicates the progress state for a given session in a batch processing queue """
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    

""" 

filters should be checkable to express whether we want to build that one or not


"""


# def validate_computation_test(self, curr_active_pipeline):
#     """ *SPECIFIC* test function for a specific computation (like 'long_short_post_decoding') that tries to access the results added by the computation function to see if it's needed or ready.
#     Throws an (AttributeError, KeyError) during its accesses if the data isn't there. 
#     """
#     ## Get global 'long_short_post_decoding' results:
#     curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
#     ## Extract variables from results object:
#     expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
#     rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df


# def a_validate_computation_test(curr_active_pipeline):
#     ## Get global 'long_short_post_decoding' results:
#     curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
#     ## Extract variables from results object:
#     expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
#     rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
# SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=a_validate_computation_test)


@define(repr=None, slots=False)
class SpecificComputationValidator:
    """ This encapsulates the logic for testing if a computation already complete or needs to be completed, and calling the compute function if needed. """
    short_name:str # 'long_short_post_decoding'
    computation_fn_name:str # '_perform_long_short_post_decoding_analysis'
    validate_computation_test:Callable
    computation_fn_kwargs:dict = Factory(dict) # {'perform_cache_load': False}]`
    is_global:bool = False
    
    def try_computation_if_needed(self, curr_active_pipeline, **kwargs):
        return self._perform_try_computation_if_needed(self, curr_active_pipeline, **kwargs)

    @classmethod
    def _perform_try_computation_if_needed(cls, comp_specifier: "SpecificComputationValidator", curr_active_pipeline, on_already_computed_fn=None, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool=False):
        """ 2023-06-08 - tries to perform the computation if the results are missing and it's needed. 
        
        Usage:
            if _comp_name in include_includelist:
                newly_computed_values += _try_computation_if_needed(curr_active_pipeline, comp_specifier=SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=a_validate_computation_test), on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
        """
        comp_short_name: str = comp_specifier.short_name
        newly_computed_values = []
        try:
            comp_specifier.validate_computation_test(curr_active_pipeline)
            if on_already_computed_fn is not None:
                on_already_computed_fn(comp_short_name)
                
        except (AttributeError, KeyError) as e:
            if progress_print or debug_print:
                print(f'{comp_short_name} missing.')
            if debug_print:
                print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
            if progress_print or debug_print:
                print(f'\t Recomputing {comp_short_name}...')
            # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
            curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[comp_specifier.computation_fn_name], computation_kwargs_list=[comp_specifier.computation_fn_kwargs], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
            if progress_print or debug_print:
                print(f'\t done.')
            # try the validation again.
            comp_specifier.validate_computation_test(curr_active_pipeline)
            newly_computed_values.append(comp_short_name)
        except Exception as e:
            raise e
        return newly_computed_values

# ==================================================================================================================== #
# 2022-12-07 - batch_load_session - Computes Entire Pipeline                                                           #
# ==================================================================================================================== #


@function_attributes(short_name='batch_load_session', tags=['main', 'batch', 'automated', 'session', 'load'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2022-12-07 00:00')
def batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, force_reload=False, saving_mode=PipelineSavingScheme.SKIP_SAVING, fail_on_exception=True, skip_extended_batch_computations=False, **kwargs):
    """Loads and runs the entire pipeline for a session folder located at the path 'basedir'.

    Args:
        global_data_root_parent_path (_type_): _description_
        active_data_mode_name (_type_): _description_
        basedir (_type_): _description_

    Returns:
        _type_: _description_
    """
    saving_mode = PipelineSavingScheme.init(saving_mode)
    epoch_name_includelist = kwargs.get('epoch_name_includelist', ['maze1','maze2','maze'])
    debug_print = kwargs.get('debug_print', False)
    assert 'skip_save' not in kwargs, f"use saving_mode=PipelineSavingScheme.SKIP_SAVING instead"
    # skip_save = kwargs.get('skip_save', False)
    active_pickle_filename = kwargs.get('active_pickle_filename', 'loadedSessPickle.pkl')

    active_session_computation_configs = kwargs.get('active_session_computation_configs', None)

    computation_functions_name_includelist = kwargs.get('computation_functions_name_includelist', None)


    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    ## Begin main run of the pipeline (load or execute):
    curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
        override_basepath=Path(basedir), force_reload=force_reload, active_pickle_filename=active_pickle_filename, skip_save_on_initial_load=True)


    active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_includelist=epoch_name_includelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
    if debug_print:
        print(f'active_session_filter_configurations: {active_session_filter_configurations}')
    
    curr_active_pipeline.filter_sessions(active_session_filter_configurations, changed_filters_ignore_list=['maze1','maze2','maze'], debug_print=False)

    ## TODO 2023-05-16 - set `curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs = curr_laps_obj` equivalent
    ## TODO 2023-05-16 - determine appropriate binning from `compute_short_long_constrained_decoders` so it's automatically from the long

    if active_session_computation_configs is None:
        """
        If there are is provided computation config, get the default:
        """
        # ## Compute shared grid_bin_bounds for all epochs from the global positions:
        # global_unfiltered_session = curr_active_pipeline.sess
        # # ((22.736279243974774, 261.696733348342), (49.989466271998936, 151.2870218547401))
        # first_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[0]]
        # # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
        # second_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[1]]
        # # ((71.67666779621361, 224.37820920766043), (110.51617463644946, 151.2870218547401))

        # grid_bin_bounding_session = first_filtered_session
        # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(grid_bin_bounding_session.position.x, grid_bin_bounding_session.position.y)

        ## OR use no grid_bin_bounds meaning they will be determined dynamically for each epoch:
        grid_bin_bounds = None
        # time_bin_size = 0.03333 #1.0/30.0 # decode at 30fps to match the position sampling frequency
        # time_bin_size = 0.1 # 10 fps
        time_bin_size = kwargs.get('time_bin_size', 0.03333) # 0.03333 = 1.0/30.0 # decode at 30fps to match the position sampling frequency
        # time_bin_size = kwargs.get('time_bin_size', 0.1) # 10 fps
        active_session_computation_configs = active_data_mode_registered_class.build_active_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=time_bin_size) # , grid_bin_bounds=grid_bin_bounds
    else:
        # Use the provided `active_session_computation_configs`:
        assert 'time_bin_size' not in kwargs, f"time_bin_size kwarg provided but will not be used because a custom active_session_computation_configs was provided as well."


    ## Setup Computation Functions to be executed:
    if computation_functions_name_includelist is None:
        # includelist Mode:
        computation_functions_name_includelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            '_perform_pf_find_ratemap_peaks_computation',
                                            # '_perform_time_dependent_pf_sequential_surprise_computation'
                                            # '_perform_two_step_position_decoding_computation',
                                            # '_perform_recursive_latent_placefield_decoding'
                                        ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'
        computation_functions_name_excludelist=None
    else:
        print(f'using provided computation_functions_name_includelist: {computation_functions_name_includelist}')
        computation_functions_name_excludelist=None

    # # excludelist Mode:
    # computation_functions_name_includelist=None
    # computation_functions_name_excludelist=['_perform_spike_burst_detection_computation','_perform_recursive_latent_placefield_decoding']

    ## TODO 2023-01-15 - perform_computations for all configs!!
    curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, debug_print=debug_print) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)

    if not skip_extended_batch_computations:
        batch_extended_computations(curr_active_pipeline, include_global_functions=False, fail_on_exception=fail_on_exception, progress_print=True, debug_print=False)
    # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_excludelist=['_perform_spike_burst_detection_computation'], debug_print=False, fail_on_exception=False) # includelist: ['_perform_baseline_placefield_computation']

    curr_active_pipeline.prepare_for_display(root_output_dir=global_data_root_parent_path.joinpath('Output'), should_smooth_maze=True) # TODO: pass a display config

    curr_active_pipeline.save_pipeline(saving_mode=saving_mode)
    if not saving_mode.shouldSave:
        print(f'saving_mode.shouldSave == False, so not saving at the end of batch_load_session')


    ## Load pickled global computations:
    # If previously pickled global results were saved, they will typically no longer be relevent if the pipeline was recomputed. We need a system of invalidating/versioning the global results when the other computations they depend on change.
    # Maybe move into `batch_extended_computations(...)` or integrate with that somehow
    # curr_active_pipeline.load_pickled_global_computation_results()

    return curr_active_pipeline



@function_attributes(short_name='batch_extended_computations', tags=['batch', 'automated', 'session', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def batch_extended_computations(curr_active_pipeline, include_includelist=None, include_global_functions=False, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool = False):
    """ performs the remaining required global computations """
    def _subfn_on_already_computed(_comp_name):
        """ captures: `progress_print`, `force_recompute`
        raises AttributeError if force_recompute is true to trigger recomputation """
        if progress_print:
            print(f'{_comp_name} already computed.')
        if force_recompute:
            if progress_print:
                print(f'\tforce_recompute is true so recomputing anyway')
            raise AttributeError # just raise an AttributeError to trigger recomputation    

    newly_computed_values = []

    non_global_comp_names = ['firing_rate_trends', 'relative_entropy_analyses']
    global_comp_names = ['jonathan_firing_rate_analysis', 'short_long_pf_overlap_analyses', 'long_short_fr_indicies_analyses', 'long_short_decoding_analyses', 'long_short_post_decoding'] # , 'long_short_rate_remapping'

    # 'firing_rate_trends', 'relative_entropy_analyses'
    # '_perform_firing_rate_trends_computation', '_perform_time_dependent_pf_sequential_surprise_computation'
    
    if include_includelist is None:
        # include all:
        include_includelist = non_global_comp_names + global_comp_names
    else:
        print(f'included includelist is specified: {include_includelist}, so only performing these extended computations.')
    ## Get computed relative entropy measures:
    global_epoch_name = curr_active_pipeline.active_completed_computation_result_names[-1] # 'maze'
    global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

    # ## Get existing `pf1D_dt`:
    # active_pf_1D = global_results.pf1D
    # active_pf_1D_dt = global_results.pf1D_dt
    if progress_print:
        print(f'Running batch_extended_computations(...) with global_epoch_name: "{global_epoch_name}"')

    ## Specify the computations and the requirements to validate them.
    _comp_specifiers = [
        SpecificComputationValidator(short_name='firing_rate_trends', computation_fn_name='_perform_firing_rate_trends_computation', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.computation_results[global_epoch_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[global_epoch_name].computed_data['extended_stats']['time_binned_position_df']), is_global=False),
        SpecificComputationValidator(short_name='relative_entropy_analyses', computation_fn_name='_perform_time_dependent_pf_sequential_surprise_computation', validate_computation_test=lambda curr_active_pipeline: (np.sum(curr_active_pipeline.global_computation_results.computed_data['relative_entropy_analyses']['flat_relative_entropy_results'], axis=1), np.sum(curr_active_pipeline.global_computation_results.computed_data['relative_entropy_analyses']['flat_jensen_shannon_distance_results'], axis=1)), is_global=False),  # flat_surprise_across_all_positions
        SpecificComputationValidator(short_name='jonathan_firing_rate_analysis', computation_fn_name='_perform_jonathan_replay_firing_rate_analyses', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']['neuron_replay_stats_df'], is_global=True),  # active_context
        SpecificComputationValidator(short_name='short_long_pf_overlap_analyses', computation_fn_name='_perform_long_short_pf_overlap_analyses', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_scalars_df'], curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_dict']), is_global=True),  # relative_entropy_overlap_scalars_df
        SpecificComputationValidator(short_name='long_short_fr_indicies_analyses', computation_fn_name='_perform_long_short_firing_rate_analyses', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']['x_frs_index'], is_global=True),  # active_context
        SpecificComputationValidator(short_name='long_short_decoding_analyses', computation_fn_name='_perform_long_short_decoding_analyses', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].long_results_obj, curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].short_results_obj), is_global=True),
        SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding'].rate_remapping.rr_df, is_global=True)
    ]

    for _comp_specifier in _comp_specifiers:
        if (not _comp_specifier.is_global) or include_global_functions:
            if _comp_specifier.short_name in include_includelist:
                newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)


    # if include_global_functions:
        
        # ## long_short_rate_remapping:
        # _comp_name = 'long_short_rate_remapping'
        # if _comp_name in include_includelist:
        #     try:
        #         ## Get global 'long_short_rate_remapping' results:
        #         curr_long_short_rr = curr_active_pipeline.global_computation_results.computed_data['long_short_rate_remapping']
        #         rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
        #         _subfn_on_already_computed(_comp_name)
                    
        #     except (AttributeError, KeyError) as e:
        #         if progress_print or debug_print:
        #             print(f'{_comp_name} missing.')
        #         if debug_print:
        #             print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
        #         if progress_print or debug_print:
        #             print(f'\t Recomputing {_comp_name}...')
                    
        #         # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
        #         curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_long_short_decoding_rate_remapping_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
        #         print(f'\t done.')
        #         curr_long_short_rr = curr_active_pipeline.global_computation_results.computed_data['long_short_rate_remapping']
        #         rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
        #         newly_computed_values.append(_comp_name)
        #     except Exception as e:
        #         raise e


    if progress_print:
        print('done with all batch_extended_computations(...).')

    return newly_computed_values



            
# ==================================================================================================================== #
# Batch Programmatic Figures - 2022-12-08 Batch Programmatic Figures (Currently only Jonathan-style)                                                                                           #
# ==================================================================================================================== #
@function_attributes(short_name='batch_programmatic_figures', tags=['batch', 'automated', 'session', 'display', 'figures'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def batch_programmatic_figures(curr_active_pipeline):
    """ programmatically generates and saves the batch figures 2022-12-07 
        curr_active_pipeline is the pipeline for a given session with all computations done.

    ## TODO: curr_session_parent_out_path

    active_identifying_session_ctx, active_session_figures_out_path, active_out_figures_list = batch_programmatic_figures(curr_active_pipeline)
    
    """
    ## 🗨️🟢 2022-10-26 - Jonathan Firing Rate Analyses
    # Perform missing global computations                                                                                  #
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_jonathan_replay_firing_rate_analyses', '_perform_long_short_pf_overlap_analyses'], fail_on_exception=True, debug_print=True)

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']

    # ==================================================================================================================== #
    # Batch Output of Figures                                                                                              #
    # ==================================================================================================================== #
    ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
    # %matplotlib qt
    short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
    short_only_aclus = short_only_df.index.values.tolist()
    long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
    long_only_aclus = long_only_df.index.values.tolist()
    shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
    shared_aclus = shared_df.index.values.tolist()
    print(f'shared_aclus: {shared_aclus}')
    print(f'long_only_aclus: {long_only_aclus}')
    print(f'short_only_aclus: {short_only_aclus}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    active_session_figures_out_path = curr_active_pipeline.get_daily_programmatic_session_output_path()
    print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
    ## MODE: this mode creates a special folder to contain the outputs for this session.

    # ==================================================================================================================== #
    # Output Figures to File                                                                                               #
    # ==================================================================================================================== #
    ## PDF Output
    # %matplotlib qtagg
    import matplotlib
    # configure backend here
    # matplotlib.use('Qt5Agg')
    # backend_qt5agg
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

    active_out_figures_list, active_session_figures_out_path = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, n_max_page_rows = 10)


    # Plot long|short firing rate index:
    # fig_save_parent_path = Path(r'E:\Dropbox (Personal)\Active\Kamran Diba Lab\Results from 2023-01-20 - LongShort Firing Rate Indicies')
    # override_fig_save_parent_path = Path(r'E:\Dropbox (Personal)\Active\Kamran Diba Lab\Pho-Kamran-Meetings\Results from 2023-04-11')
    override_fig_save_parent_path = None
    if override_fig_save_parent_path is None:
        fig_save_parent_path = active_session_figures_out_path
    else:
        fig_save_parent_path = override_fig_save_parent_path

    curr_active_pipeline.display('_display_short_long_firing_rate_index_comparison', curr_active_pipeline.sess.get_context(), fig_save_parent_path=fig_save_parent_path)



    return active_identifying_session_ctx, active_session_figures_out_path, active_out_figures_list


# import matplotlib as mpl
# import matplotlib.pyplot as plt
@function_attributes(short_name='batch_extended_programmatic_figures', tags=['batch', 'automated', 'session', 'display', 'figures', 'extended', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def batch_extended_programmatic_figures(curr_active_pipeline, write_pdf=False, write_png=True, debug_print=True):
    _bak_rcParams = mpl.rcParams.copy()
    mpl.rcParams['toolbar'] = 'None' # disable toolbars
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!
    # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_1d_placefields', write_pdf=write_pdf, write_png=write_png, debug_print=debug_print) # 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    
    # '_display_1d_placefield_validations' can't use `programmatic_render_to_file` yet and must rely on `programmatic_display_to_PDF` because it doesn't get a new session for each figure and it overwrites itself a bunch
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations', debug_print=debug_print) # , filter_name=active_config_name 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear. Moderate visual improvements can still be made (titles overlap and stuff). Works with %%capture
    # programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations', write_pdf=write_pdf, write_png=write_png, debug_print=debug_print) #  UNTESTED 2023-05-29 
    
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_result_plot_ratemaps_2D', write_pdf=write_pdf, write_png=write_png, debug_print=debug_print) #  🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_occupancy', write_pdf=write_pdf, write_png=write_png, debug_print=debug_print) #  🟢✅ 2023-05-25
    # programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_long_short_laps') #  UNTESTED 2023-05-29

   
    # # Plot long|short firing rate index:
    # fig_save_parent_path = Path(r'E:\Dropbox (Personal)\Active\Kamran Diba Lab\Results from 2023-01-20 - LongShort Firing Rate Indicies')
    # long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
    # x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
    # active_context = long_short_fr_indicies_analysis_results['active_context']
    # plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, fig_save_parent_path=fig_save_parent_path)

    #TODO 2023-06-14 05:30: - [ ] Refactor these (the global placefields) into a form compatible with the local ones using some sort of shortcut method like `programmatic_render_to_file`
    try:
        _out = curr_active_pipeline.display('_display_long_short_expected_v_observed_firing_rate', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=False)
    except Exception as e:
        print(f'batch_extended_programmatic_figures(...): _prepare_plot_expected_vs_observed failed with error: {e}\n skipping.')
    
    ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
    try:
        _out = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', curr_active_pipeline.get_session_context(), defer_render=True)
    except Exception as e:
        print(f'batch_extended_programmatic_figures(...): _prepare_plot_long_and_short_epochs failed with error: {e}\n skipping.')
    




class BatchPhoJonathanFiguresHelper(object):
    """Private methods that help with batch figure generator for ClassName.

    In .run(...) it builds the plot_kwargs ahead of time that will be passed to the specific plot function using `cls._build_batch_plot_kwargs(...)`
        It then calls `active_out_figures_list = cls._perform_batch_plot(...)` to do the plotting, getting the list of figures and output paths
    
    2022-12-08 - Batch Programmatic Figures (Currently only Jonathan-style) 
    2022-12-01 - Automated programmatic output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`

    
    
    """
    _display_fn_name = '_display_batch_pho_jonathan_replay_firing_rate_comparison' # used as the display function called in `_subfn_batch_plot_automated(...)`
    _display_fn_context_display_name = 'BatchPhoJonathanReplayFRC' # used in `_build_batch_plot_kwargs` as the display_fn_name for the generated context. Affects the output names of the figures like f'kdiba_gor01_one_2006-6-09_1-22-43_{cls._display_fn_context_display_name}_long_only_[5, 23, 29, 38, 70, 85, 97, 103].pdf'. 


    @classmethod
    def run(cls, curr_active_pipeline, neuron_replay_stats_df, n_max_page_rows = 10):
        """ The only public function. Performs the batch plotting. """

        ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
        # %matplotlib qt
        short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
        short_only_aclus = short_only_df.index.values.tolist()
        long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
        long_only_aclus = long_only_df.index.values.tolist()
        shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
        shared_aclus = shared_df.index.values.tolist()
        print(f'shared_aclus: {shared_aclus}')
        print(f'long_only_aclus: {long_only_aclus}')
        print(f'short_only_aclus: {short_only_aclus}')

        
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()

        active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        active_session_figures_out_path = session_context_to_relative_path(figures_parent_out_path, active_identifying_session_ctx)
        print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
        active_session_figures_out_path.mkdir(parents=True, exist_ok=True) # make folder if needed
                


        # %matplotlib qtagg
        import matplotlib
        # configure backend here
        # matplotlib.use('Qt5Agg')
        # backend_qt5agg
        matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

        _batch_plot_kwargs_list = cls._build_batch_plot_kwargs(long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=n_max_page_rows)
        active_out_figures_list = cls._perform_batch_plot(curr_active_pipeline, _batch_plot_kwargs_list, figures_parent_out_path=active_session_figures_out_path, write_pdf=False, write_png=True, progress_print=True, debug_print=False)
        
        return active_out_figures_list, active_session_figures_out_path

    @classmethod
    def _subfn_batch_plot_automated(cls, curr_active_pipeline, included_unit_neuron_IDs=None, active_identifying_ctx=None, fignum=None, fig_idx=0, n_max_page_rows=10):
        """ the a programmatic wrapper for automated output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`. The specific plot function called. 
        Called ONLY by `_perform_batch_plot(...)`

        Calls `curr_active_pipeline.display(cls._display_fn_name, ...)
        """
        # size_dpi = 100.0,
        # single_subfigure_size_px = np.array([1920.0, 220.0])
        single_subfigure_size_inches = np.array([19.2,  2.2])

        num_cells = len(included_unit_neuron_IDs)
        desired_figure_size_inches = single_subfigure_size_inches.copy()
        desired_figure_size_inches[1] = desired_figure_size_inches[1] * num_cells
        graphics_output_dict = curr_active_pipeline.display(cls._display_fn_name, active_identifying_ctx,
                                                            n_max_plot_rows=n_max_page_rows, included_unit_neuron_IDs=included_unit_neuron_IDs,
                                                            show_inter_replay_frs=True, spikes_color=(0.1, 0.0, 0.1), spikes_alpha=0.5, fignum=fignum, fig_idx=fig_idx, figsize=desired_figure_size_inches, save_figure=True, defer_render=True)
        # fig, subfigs, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['subfigs'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
        fig, subfigs, axs, plot_data = graphics_output_dict.figures[0], graphics_output_dict.subfigs, graphics_output_dict.axes, graphics_output_dict.plot_data
        fig.suptitle(active_identifying_ctx.get_description()) # 'kdiba_2006-6-08_14-26-15_[4, 13, 36, 58, 60]'
        return fig

    @classmethod
    def _build_batch_plot_kwargs(cls, long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=10):
        """ builds the list of kwargs for all aclus. """
        _batch_plot_kwargs_list = [] # empty list to start
        ## {long_only, short_only} plot configs (doesn't include the shared_aclus)
        if len(long_only_aclus) > 0:        
            _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=long_only_aclus,
            active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
                display_fn_name=cls._display_fn_context_display_name, plot_result_set='long_only', aclus=f"{long_only_aclus}"
            ),
            fignum='long_only', n_max_page_rows=len(long_only_aclus)))
        else:
            print(f'WARNING: long_only_aclus is empty, so not adding kwargs for these.')
        
        if len(short_only_aclus) > 0:
            _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=short_only_aclus,
            active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
                display_fn_name=cls._display_fn_context_display_name, plot_result_set='short_only', aclus=f"{short_only_aclus}"
            ),
            fignum='short_only', n_max_page_rows=len(short_only_aclus)))
        else:
            print(f'WARNING: short_only_aclus is empty, so not adding kwargs for these.')

        ## Build Pages for Shared ACLUS:    
        nAclusToShow = len(shared_aclus)
        if nAclusToShow > 0:        
            # Paging Management: Constrain the subplots values to just those that you need
            subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nAclusToShow, max_num_columns=1, max_subplots_per_page=n_max_page_rows, data_indicies=shared_aclus, last_figure_subplots_same_layout=True)
            num_pages = len(included_combined_indicies_pages)
            ## paginated outputs for shared cells
            included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
            paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
                active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name=cls._display_fn_context_display_name, plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{curr_included_unit_indicies}"),
                fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]
            _batch_plot_kwargs_list.extend(paginated_shared_cells_kwarg_list) # add paginated_shared_cells_kwarg_list to the list
        else:
            print(f'WARNING: shared_aclus is empty, so not adding kwargs for these.')
        return _batch_plot_kwargs_list

    @classmethod
    def _perform_batch_plot(cls, curr_active_pipeline, active_kwarg_list, figures_parent_out_path=None, subset_includelist=None, subset_excludelist=None, write_pdf=False, write_png=True, progress_print=True, debug_print=False):
        """ Plots everything by calling `cls._subfn_batch_plot_automated` using the kwargs provided in `active_kwarg_list`

        Args:
            active_kwarg_list (_type_): generated by `_build_batch_plot_kwargs(...)`
            figures_parent_out_path (_type_, optional): _description_. Defaults to None.
            write_pdf (bool, optional): _description_. Defaults to False.
            write_png (bool, optional): _description_. Defaults to True.
            progress_print (bool, optional): _description_. Defaults to True.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if figures_parent_out_path is None:
            figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
        active_out_figures_list = [] # empty list to hold figures
        num_pages = len(active_kwarg_list)
        for i, curr_batch_plot_kwargs in enumerate(active_kwarg_list):
            curr_active_identifying_ctx = curr_batch_plot_kwargs['active_identifying_ctx']
            # print(f'curr_active_identifying_ctx: {curr_active_identifying_ctx}')
            active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
            # print(f'active_pdf_save_filename: {active_pdf_save_filename}')
            curr_pdf_save_path = figures_parent_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)
            # One plot at a time to PDF:
            if write_pdf:
                with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                    a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
                    active_out_figures_list.append(a_fig)
                    # Save out PDF page:
                    pdf.savefig(a_fig)
                    curr_active_pipeline.register_output_file(output_path=curr_pdf_save_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig), 'pdf_metadata': active_pdf_metadata})
            else:
                # Don't write the PDF and just plot interactively:
                a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
                active_out_figures_list.append(a_fig)

            # Also save .png versions:
            if write_png:
                # curr_page_str = f'pg{i+1}of{num_pages}'
                fig_png_out_path = curr_pdf_save_path.with_suffix('.png')
                # fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
                a_fig.savefig(fig_png_out_path)
                curr_active_pipeline.register_output_file(output_path=fig_png_out_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig)})
                if progress_print:
                    print(f'\t saved {fig_png_out_path}')

        return active_out_figures_list


# ==================================================================================================================== #
# Main Public Plot Function                                                                                            #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['active', 'batch', 'public'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-12 14:23', related_items=[])
def batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None):
    """ 2023-05-25 - Performs all the batch plotting commands. 
    
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots
    
    """
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, Neptuner
    if neptuner is None:
        neptuner = Neptuner.init_with_pipeline(curr_active_pipeline)
        
    curr_active_pipeline.reload_default_display_functions()
    try:
        active_identifying_session_ctx, active_session_figures_out_path, active_out_figures_list = batch_programmatic_figures(curr_active_pipeline)
    except Exception as e:
        print(f'in `_perform_plots(...)`: batch_programmatic_figures(...) failed with exception: {e}. Continuing.')
    
    try:
        batch_extended_programmatic_figures(curr_active_pipeline=curr_active_pipeline)
    except Exception as e:
        print(f'in `_perform_plots(...)`: batch_extended_programmatic_figures(...) failed with exception: {e}. Continuing.')
    
    if enable_neptune:
        try:
            succeeded_fig_paths, failed_fig_paths = neptuner.upload_figures(curr_active_pipeline)
            # neptune_output_figures(curr_active_pipeline)
        except Exception as e:
            print(f'in `_perform_plots(...)`: neptune_output_figures(...) failed with exception: {e}. Continuing.')
        finally:
            neptuner.stop()
        neptuner.stop()
        
    return neptuner
    





# ==================================================================================================================== #
# 2023-05-25 - Pipeline Preprocessing Parameter Saving                                                                 #
# ==================================================================================================================== #

from neuropy.utils.dynamic_container import DynamicContainer
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

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



# ==================================================================================================================== #
# 2023-05-25 - Separate Generalized Plot Saving/Registering Function                                                   #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context

                        

