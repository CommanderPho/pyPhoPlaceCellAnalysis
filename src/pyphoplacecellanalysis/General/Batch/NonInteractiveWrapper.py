from copy import deepcopy
import importlib
import sys
from pathlib import Path
import numpy as np
from enum import unique # SessionBatchProgress
import traceback # for stack trace formatting

from attrs import define, Factory, fields

## MATPLOTLIB Imports:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends import backend_pdf


# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.core.epoch import NamedTimerange
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.misc import compute_paginated_grid_config # for paginating shared aclus

# pyphocorehelpers
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # required for SessionBatchProgress
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.indexing_helpers import compute_position_grid_size
from pyphocorehelpers.function_helpers import function_attributes

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership # needed for batch_extended_computations, batch_programmatic_figures
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, session_context_to_relative_path, programmatic_display_to_PDF
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



@define(slots=False)
class NonInteractiveWrapper(object):
    """A wrapper class that performs a non-interactive version of the jupyter-lab notebook for use with the custom `PipelineComputationsNode` and `PipelineFilteringDataNote` Flowchart Notes: enables loading and processing the pipeline. """
    enable_saving_to_disk:bool = False
    common_parent_foldername:Path = Path(r'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\output')
   
    
    @staticmethod
    def compute_position_grid_bin_size(x, y, num_bins=(64,64), debug_print=False):
        """ Compute Required Bin size given a desired number of bins in each dimension
        Usage:
            active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64)
        """
        out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(x, y, num_bins=num_bins)
        active_grid_bin = tuple(out_grid_bin_size)
        if debug_print:
            print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
        return active_grid_bin

    # WARNING! TODO: Changing the smooth values from (1.5, 1.5) to (0.5, 0.5) was the difference between successful running and a syntax error!
    # try:
    #     active_grid_bin
    # except NameError as e:
    #     print('setting active_grid_bin = None')
    #     active_grid_bin = None
    # finally:
    #     # active_session_computation_config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=active_grid_bin, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5) # if active_grid_bin is missing, figure out the name
    #     active_session_computation_config = PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=active_grid_bin, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5) # if active_grid_bin is missing, figure out the name

    ## Dynamic mode:
    @staticmethod
    def _build_active_computation_configs(sess):
        """ _get_computation_configs(curr_kdiba_pipeline.sess) 
            # From Diba:
            # (3.777, 1.043) # for (64, 64) bins
            # (1.874, 0.518) # for (128, 128) bins

        """
        return [DynamicParameters(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=NonInteractiveWrapper.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None))]
        
    @staticmethod
    def perform_computation(pipeline, active_computation_configs, enabled_filter_names=None):
        pipeline.perform_computations(active_computation_configs[0], enabled_filter_names) # unuses the first config
        pipeline.prepare_for_display() # TODO: pass a display config
        return pipeline

    @staticmethod
    def perform_filtering(pipeline, filter_configs):
        pipeline.filter_sessions(filter_configs)
        return pipeline


    # ==================================================================================================================== #
    # Specific Format Type Helpers                                                                                         #
    # ==================================================================================================================== #
  
    # Bapun ______________________________________________________________________________________________________________ #
    @staticmethod
    def bapun_format_all(pipeline):
        active_session_computation_configs, active_session_filter_configurations = NonInteractiveWrapper.bapun_format_define_configs(pipeline)
        pipeline = NonInteractiveWrapper.perform_filtering(pipeline, active_session_filter_configurations)
        pipeline = NonInteractiveWrapper.perform_computation(pipeline, active_session_computation_configs)
        return pipeline, active_session_computation_configs, active_session_filter_configurations
     
    @staticmethod
    def bapun_format_define_configs(curr_bapun_pipeline):
        # curr_bapun_pipeline = NeuropyPipeline(name='bapun_pipeline', session_data_type='bapun', basedir=known_data_session_type_dict['bapun'].basedir, load_function=known_data_session_type_dict['bapun'].load_function)
        # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        active_session_computation_configs = NonInteractiveWrapper._build_active_computation_configs(curr_bapun_pipeline.sess)
        # active_session_computation_config.grid_bin = compute_position_grid_bin_size(curr_bapun_pipeline.sess.position.x, curr_bapun_pipeline.sess.position.y, num_bins=(64, 64))
                
        # Bapun/DataFrame style session filter functions:
        def build_bapun_any_maze_epochs_filters(sess):
            # all_filters = build_custom_epochs_filters(sess)
            # # print(f'all_filters: {all_filters}')
            # maze_only_filters = dict()
            # for (name, filter_fcn) in all_filters.items():
            #     if 'maze' in name:
            #         maze_only_filters[name] = filter_fcn
            # maze_only_filters = build_custom_epochs_filters(sess, epoch_name_whitelist=['maze1','maze2'])
            # { key:value for (key,value) in dictOfNames.items() if key % 2 == 0}
            # dict(filter(lambda elem: len(elem[1]) == 6,dictOfNames.items()))
            # maze_only_name_filter_fn = lambda dict: dict(filter(lambda elem: 'maze' in elem[0], dict.items()))
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names)) # include only maze tracks
            # print(f'callable(maze_only_name_filter_fn): {callable(maze_only_name_filter_fn)}')
            # print(maze_only_name_filter_fn(['pre', 'maze1', 'post1', 'maze2', 'post2']))
            # lambda elem: elem[0] % 2 == 0
            maze_only_filters = build_custom_epochs_filters(sess, epoch_name_whitelist=maze_only_name_filter_fn)
            # print(f'maze_only_filters: {maze_only_filters}')
            return maze_only_filters

        active_session_filter_configurations = build_bapun_any_maze_epochs_filters(curr_bapun_pipeline.sess)
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = None  # set the placefield computation epochs to None, using all epochs.
        return active_session_computation_configs, active_session_filter_configurations


    # KDIBA ______________________________________________________________________________________________________________ #
    @staticmethod
    def kdiba_format_all(pipeline):
        active_session_computation_configs, active_session_filter_configurations = NonInteractiveWrapper.kdiba_format_define_configs(pipeline)
        pipeline = NonInteractiveWrapper.perform_filtering(pipeline, active_session_filter_configurations)
        pipeline = NonInteractiveWrapper.perform_computation(pipeline, active_session_computation_configs)
        return pipeline, active_session_computation_configs, active_session_filter_configurations

    @staticmethod
    def kdiba_format_define_configs(curr_kdiba_pipeline):
        ## Data must be pre-processed using the MATLAB script located here: 
        # R:\data\KDIBA\gor01\one\IIDataMat_Export_ToPython_2021_11_23.m
        # From pre-computed .mat files:
        ## 07: 
        # basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
        # # ## 08:
        # basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
        # curr_kdiba_pipeline = NeuropyPipeline(name='kdiba_pipeline', session_data_type='kdiba', basedir=known_data_session_type_dict['kdiba'].basedir, load_function=known_data_session_type_dict['kdiba'].load_function)
        # curr_kdiba_pipeline = NeuropyPipeline.init_from_known_data_session_type('kdiba', known_data_session_type_dict['kdiba'])
        # active_grid_bin = compute_position_grid_bin_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
        # active_session_computation_config.grid_bin = active_grid_bin
        active_session_computation_configs = NonInteractiveWrapper._build_active_computation_configs(curr_kdiba_pipeline.sess)
        
        def build_any_maze_epochs_filters(sess):
            sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
            active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')) } # just maze 1
            # active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
            #                                     'maze2': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2')),
            #                                     'maze': lambda x: (x.filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]))
            #                                    }
            return active_session_filter_configurations

        active_session_filter_configurations = build_any_maze_epochs_filters(curr_kdiba_pipeline.sess)
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = None # add the laps epochs to all of the computation configs.

        return active_session_computation_configs, active_session_filter_configurations
        
        # # set curr_pipeline for testing:
        # curr_pipeline = curr_kdiba_pipeline
        


        # Pyramidal and Lap-Only:
        def build_pyramidal_epochs_filters(sess):
            sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
            active_session_filter_configurations = {'maze1': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
                                                'maze2': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2')),
                                                'maze': lambda x: (x.filtered_by_neuron_type('pyramidal').filtered_by_epoch(NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]])), NamedTimerange(name='maze', start_end_times=[x.epochs['maze1'][0], x.epochs['maze2'][1]]))
                                            }
            return active_session_filter_configurations

        active_session_filter_configurations = build_pyramidal_epochs_filters(curr_kdiba_pipeline.sess)

        lap_specific_epochs = curr_kdiba_pipeline.sess.laps.as_epoch_obj()
        any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(curr_kdiba_pipeline.sess.laps.lap_id))])
        even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(curr_kdiba_pipeline.sess.laps.lap_id), 2)])
        odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(curr_kdiba_pipeline.sess.laps.lap_id), 2)])

        # Copy the active session_computation_config:
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = any_lap_specific_epochs # add the laps epochs to all of the computation configs.

        return active_session_computation_configs, active_session_filter_configurations
    

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
    epoch_name_whitelist = kwargs.get('epoch_name_whitelist', ['maze1','maze2','maze'])
    debug_print = kwargs.get('debug_print', False)
    assert 'skip_save' not in kwargs, f"use saving_mode=PipelineSavingScheme.SKIP_SAVING instead"
    # skip_save = kwargs.get('skip_save', False)
    active_pickle_filename = kwargs.get('active_pickle_filename', 'loadedSessPickle.pkl')

    active_session_computation_configs = kwargs.get('active_session_computation_configs', None)

    computation_functions_name_whitelist = kwargs.get('computation_functions_name_whitelist', None)


    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    ## Begin main run of the pipeline (load or execute):
    curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
        override_basepath=Path(basedir), force_reload=force_reload, active_pickle_filename=active_pickle_filename, skip_save_on_initial_load=True)


    active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_whitelist=epoch_name_whitelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
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
    if computation_functions_name_whitelist is None:
        # Whitelist Mode:
        computation_functions_name_whitelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            '_perform_pf_find_ratemap_peaks_computation',
                                            # '_perform_time_dependent_pf_sequential_surprise_computation'
                                            # '_perform_two_step_position_decoding_computation',
                                            # '_perform_recursive_latent_placefield_decoding'
                                        ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'
        computation_functions_name_blacklist=None
    else:
        print(f'using provided computation_functions_name_whitelist: {computation_functions_name_whitelist}')
        computation_functions_name_blacklist=None

    # # Blacklist Mode:
    # computation_functions_name_whitelist=None
    # computation_functions_name_blacklist=['_perform_spike_burst_detection_computation','_perform_recursive_latent_placefield_decoding']

    ## TODO 2023-01-15 - perform_computations for all configs!!
    curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, debug_print=debug_print) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)

    if not skip_extended_batch_computations:
        batch_extended_computations(curr_active_pipeline, include_global_functions=False, fail_on_exception=fail_on_exception, progress_print=True, debug_print=False)
    # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_blacklist=['_perform_spike_burst_detection_computation'], debug_print=False, fail_on_exception=False) # whitelist: ['_perform_baseline_placefield_computation']

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
def batch_extended_computations(curr_active_pipeline, include_whitelist=None, include_global_functions=False, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool = False):
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
    global_comp_names = ['jonathan_firing_rate_analysis', 'short_long_pf_overlap_analyses', 'long_short_fr_indicies_analyses', 'long_short_decoding_analyses']

    if include_whitelist is None:
        # include all:
        include_whitelist = non_global_comp_names + global_comp_names
    else:
        print(f'included whitelist is specified: {include_whitelist}, so only performing these extended computations.')
    ## Get computed relative entropy measures:
    global_epoch_name = curr_active_pipeline.active_completed_computation_result_names[-1] # 'maze'
    global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

    # ## Get existing `pf1D_dt`:
    # active_pf_1D = global_results.pf1D
    # active_pf_1D_dt = global_results.pf1D_dt
    if progress_print:
        print(f'Running batch_extended_computations(...) with global_epoch_name: "{global_epoch_name}"')

    ## firing_rate_trends:
    _comp_name = 'firing_rate_trends'
    if _comp_name in include_whitelist:
        try:
            active_extended_stats = curr_active_pipeline.computation_results[global_epoch_name].computed_data['extended_stats']
            time_binned_pos_df = active_extended_stats['time_binned_position_df']
            firing_rate_trends = curr_active_pipeline.computation_results[global_epoch_name].computed_data['firing_rate_trends']
            if progress_print:
                print(f'{_comp_name} already computed.')
        except (AttributeError, KeyError) as e:
            if progress_print or debug_print:
                print(f'{_comp_name} missing.')
            if debug_print:
                print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
            if progress_print or debug_print:
                print(f'\t Recomputing {_comp_name}...')
            curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_firing_rate_trends_computation'], enabled_filter_names=[global_epoch_name], fail_on_exception=fail_on_exception, debug_print=False) 
            print(f'\t done.')
            active_extended_stats = curr_active_pipeline.computation_results[global_epoch_name].computed_data['extended_stats']
            time_binned_pos_df = active_extended_stats['time_binned_position_df']
            firing_rate_trends = curr_active_pipeline.computation_results[global_epoch_name].computed_data['firing_rate_trends']
            newly_computed_values.append(_comp_name)
        except Exception as e:
            raise e

    ## relative_entropy_analyses:
    # must have '_perform_firing_rate_trends_computation's fring rate trends
    _comp_name = 'relative_entropy_analyses'
    if _comp_name in include_whitelist:
        try:
            active_relative_entropy_results = active_extended_stats['relative_entropy_analyses']
            post_update_times = active_relative_entropy_results['post_update_times'] # (4152,) = (n_post_update_times,)
            snapshot_differences_result_dict = active_relative_entropy_results['snapshot_differences_result_dict']
            time_intervals = active_relative_entropy_results['time_intervals']
            long_short_rel_entr_curves_frames = active_relative_entropy_results['long_short_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
            short_long_rel_entr_curves_frames = active_relative_entropy_results['short_long_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
            flat_relative_entropy_results = active_relative_entropy_results['flat_relative_entropy_results'] # (149, 63) - (nSnapshots, nXbins)
            flat_jensen_shannon_distance_results = active_relative_entropy_results['flat_jensen_shannon_distance_results'] # (149, 63) - (nSnapshots, nXbins)
            flat_jensen_shannon_distance_across_all_positions = np.sum(flat_jensen_shannon_distance_results, axis=1) # sum across all position bins # (4152,) - (nSnapshots)
            flat_surprise_across_all_positions = np.sum(flat_relative_entropy_results, axis=1) # sum across all position bins # (4152,) - (nSnapshots)
            _subfn_on_already_computed(_comp_name)
                

        except (AttributeError, KeyError) as e:
            if progress_print or debug_print:
                print(f'{_comp_name} missing.')
            if debug_print:
                print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
            if progress_print or debug_print:
                print(f'\t Recomputing {_comp_name}...')
            curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_time_dependent_pf_sequential_surprise_computation'], enabled_filter_names=[global_epoch_name], fail_on_exception=fail_on_exception, debug_print=False)
            print(f'\t done.')
            active_relative_entropy_results = active_extended_stats['relative_entropy_analyses']
            post_update_times = active_relative_entropy_results['post_update_times'] # (4152,) = (n_post_update_times,)
            snapshot_differences_result_dict = active_relative_entropy_results['snapshot_differences_result_dict']
            time_intervals = active_relative_entropy_results['time_intervals']
            long_short_rel_entr_curves_frames = active_relative_entropy_results['long_short_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
            short_long_rel_entr_curves_frames = active_relative_entropy_results['short_long_rel_entr_curves_frames'] # (4152, 108, 63) = (n_post_update_times, n_neurons, n_xbins)
            flat_relative_entropy_results = active_relative_entropy_results['flat_relative_entropy_results'] # (149, 63) - (nSnapshots, nXbins)
            flat_jensen_shannon_distance_results = active_relative_entropy_results['flat_jensen_shannon_distance_results'] # (149, 63) - (nSnapshots, nXbins)
            flat_jensen_shannon_distance_across_all_positions = np.sum(np.abs(flat_jensen_shannon_distance_results), axis=1) # sum across all position bins # (4152,) - (nSnapshots)
            flat_surprise_across_all_positions = np.sum(np.abs(flat_relative_entropy_results), axis=1) # sum across all position bins # (4152,) - (nSnapshots)
            newly_computed_values.append(_comp_name)
        except Exception as e:
            raise e

    if include_global_functions:
        ## jonathan_firing_rate_analysis:
        _comp_name = 'jonathan_firing_rate_analysis'
        if _comp_name in include_whitelist:
            try:
                ## Get global 'jonathan_firing_rate_analysis' results:
                curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
                neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']
                _subfn_on_already_computed(_comp_name)
                    
            except (AttributeError, KeyError) as e:
                if progress_print or debug_print:
                    print(f'{_comp_name} missing.')
                if debug_print:
                    print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
                if progress_print or debug_print:
                    print(f'\t Recomputing {_comp_name}...')
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_jonathan_replay_firing_rate_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                print(f'\t done.')
                curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
                neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']
                newly_computed_values.append(_comp_name)
            except Exception as e:
                raise e

        ## short_long_pf_overlap_analyses:
        _comp_name = 'short_long_pf_overlap_analyses'
        if _comp_name in include_whitelist:
            try:
                ## Get global `short_long_pf_overlap_analyses` results:
                short_long_pf_overlap_analyses = curr_active_pipeline.global_computation_results.computed_data.short_long_pf_overlap_analyses
                conv_overlap_dict = short_long_pf_overlap_analyses['conv_overlap_dict']
                conv_overlap_scalars_df = short_long_pf_overlap_analyses['conv_overlap_scalars_df']
                prod_overlap_dict = short_long_pf_overlap_analyses['product_overlap_dict']
                relative_entropy_overlap_dict = short_long_pf_overlap_analyses['relative_entropy_overlap_dict']
                relative_entropy_overlap_scalars_df = short_long_pf_overlap_analyses['relative_entropy_overlap_scalars_df']
                _subfn_on_already_computed(_comp_name)
            except (AttributeError, KeyError) as e:
                if progress_print or debug_print:
                    print(f'{_comp_name} missing.')
                if debug_print:
                    print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
                if progress_print or debug_print:
                    print(f'\t Recomputing {_comp_name}...')
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_long_short_pf_overlap_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                print(f'\t done.')
                short_long_pf_overlap_analyses = curr_active_pipeline.global_computation_results.computed_data.short_long_pf_overlap_analyses
                conv_overlap_dict = short_long_pf_overlap_analyses['conv_overlap_dict']
                conv_overlap_scalars_df = short_long_pf_overlap_analyses['conv_overlap_scalars_df']
                prod_overlap_dict = short_long_pf_overlap_analyses['product_overlap_dict']
                relative_entropy_overlap_dict = short_long_pf_overlap_analyses['relative_entropy_overlap_dict']
                relative_entropy_overlap_scalars_df = short_long_pf_overlap_analyses['relative_entropy_overlap_scalars_df']
                newly_computed_values.append(_comp_name)
            except Exception as e:
                raise e

        # short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
        # short_only_aclus = short_only_df.index.values.tolist()
        # long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
        # long_only_aclus = long_only_df.index.values.tolist()
        # shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
        # shared_aclus = shared_df.index.values.tolist()
        # if debug_print:
        #     print(f'shared_aclus: {shared_aclus}')
        #     print(f'long_only_aclus: {long_only_aclus}')
        #     print(f'short_only_aclus: {short_only_aclus}')

        # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

        
        ## pipeline_complete_compute_long_short_fr_indicies:
        # TODO 2023-01-26 - NOTE - not really a computation function, a hack.  Should be moved to a separate function.
        _comp_name = 'long_short_fr_indicies_analyses'
        if _comp_name in include_whitelist:
            try:
                ## Get global `long_short_fr_indicies_analysis` results:
                long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
                x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
                active_context = long_short_fr_indicies_analysis_results['active_context']
                _subfn_on_already_computed(_comp_name)
            except (AttributeError, KeyError) as e:
                if progress_print or debug_print:
                    print(f'{_comp_name} missing.')
                if debug_print:
                    print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
                if progress_print or debug_print:
                    print(f'\t Recomputing {_comp_name}...')
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_long_short_firing_rate_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                print(f'\t done.')
                long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
                x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
                active_context = long_short_fr_indicies_analysis_results['active_context']
                newly_computed_values.append(_comp_name)
            except Exception as e:
                raise e

        ## long_short_decoding_analyses:
        _comp_name = 'long_short_decoding_analyses'
        if _comp_name in include_whitelist:
            try:
                ## Get global 'long_short_decoding_analyses' results:
                curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
                long_results_obj, short_results_obj = curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj
                _subfn_on_already_computed(_comp_name)
                    
            except (AttributeError, KeyError) as e:
                if progress_print or debug_print:
                    print(f'{_comp_name} missing.')
                if debug_print:
                    print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
                if progress_print or debug_print:
                    print(f'\t Recomputing {_comp_name}...')
                    
                # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_long_short_decoding_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                print(f'\t done.')
                curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
                # TODO: check contents
                long_results_obj, short_results_obj = curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj
                newly_computed_values.append(_comp_name)
            except Exception as e:
                raise e
            

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
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_jonathan_replay_firing_rate_analyses', '_perform_long_short_pf_overlap_analyses'], fail_on_exception=True, debug_print=True)

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
    # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
    figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
    active_session_figures_out_path = session_context_to_relative_path(figures_parent_out_path, active_identifying_session_ctx)
    print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
    active_session_figures_out_path.mkdir(parents=True, exist_ok=True) # make folder if needed


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
def batch_extended_programmatic_figures(curr_active_pipeline):
    _bak_rcParams = mpl.rcParams.copy()
    mpl.rcParams['toolbar'] = 'None' # disable toolbars
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!
    # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefields', debug_print=False) # 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations') # , filter_name=active_config_name 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear. Moderate visual improvements can still be made (titles overlap and stuff). Works with %%capture
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_result_plot_ratemaps_2D') #  🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_occupancy') #  🟢✅ 2023-05-25


    # # Plot long|short firing rate index:
    # fig_save_parent_path = Path(r'E:\Dropbox (Personal)\Active\Kamran Diba Lab\Results from 2023-01-20 - LongShort Firing Rate Indicies')
    # long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
    # x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
    # active_context = long_short_fr_indicies_analysis_results['active_context']
    # plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, fig_save_parent_path=fig_save_parent_path)



class BatchPhoJonathanFiguresHelper(object):
    """Private methods that help with batch figure generator for ClassName.

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

        active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
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

        """
        # size_dpi = 100.0,
        # single_subfigure_size_px = np.array([1920.0, 220.0])
        single_subfigure_size_inches = np.array([19.2,  2.2])

        num_cells = len(included_unit_neuron_IDs)
        desired_figure_size_inches = single_subfigure_size_inches.copy()
        desired_figure_size_inches[1] = desired_figure_size_inches[1] * num_cells
        graphics_output_dict = curr_active_pipeline.display(cls._display_fn_name, active_identifying_ctx,
                                                            n_max_plot_rows=n_max_page_rows, included_unit_neuron_IDs=included_unit_neuron_IDs,
                                                            show_inter_replay_frs=True, spikes_color=(0.1, 0.0, 0.1), spikes_alpha=0.5, fignum=fignum, fig_idx=fig_idx, figsize=desired_figure_size_inches)
        fig, subfigs, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['subfigs'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
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
    def _perform_batch_plot(cls, curr_active_pipeline, active_kwarg_list, figures_parent_out_path=None, subset_whitelist=None, subset_blacklist=None, write_pdf=False, write_png=True, progress_print=True, debug_print=False):
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
            active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx, subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist)
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
# 2023-05-25 - Pipeline Preprocessing Parameter Saving                                                                 #
# ==================================================================================================================== #

from neuropy.utils.dynamic_container import DynamicContainer
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

def _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline, debug_print=False):
    """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline. 
    
    Usage:
        from pyphoplacecellanalysis.General.Batch.NonInteractiveWrapper import _update_pipeline_missing_preprocessing_parameters
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

@function_attributes(short_name=None, tags=['neptune', 'figures', 'output', 'cloud', 'logging', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 04:34', related_items=[])
def neptune_output_figures(curr_active_pipeline):
    """ Uploads the completed figures to neptune.ai from the pipeline's `.registered_output_files` items. 
    
    Usage:
        from pyphoplacecellanalysis.General.Batch.NonInteractiveWrapper import neptune_output_figures
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
                        



# def _generalized_persist_plot(curr_active_pipeline, active_kwarg_list, figures_parent_out_path=None, subset_whitelist=None, subset_blacklist=None, write_pdf=False, write_png=True, progress_print=True, debug_print=False):
#     """ Plots everything by calling `cls._subfn_batch_plot_automated` using the kwargs provided in `active_kwarg_list`

#     from pyphoplacecellanalysis.General.Batch.NonInteractiveWrapper import _generalized_persist_plot

    
#     Args:
#         active_kwarg_list (_type_): generated by `_build_batch_plot_kwargs(...)`
#         figures_parent_out_path (_type_, optional): _description_. Defaults to None.
#         write_pdf (bool, optional): _description_. Defaults to False.
#         write_png (bool, optional): _description_. Defaults to True.
#         progress_print (bool, optional): _description_. Defaults to True.
#         debug_print (bool, optional): _description_. Defaults to False.

#     History:
#         Generalized from `._perform_batch_plot` on 2023-05-25
        
#     """
#     if figures_parent_out_path is None:
#         figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
#     active_out_figures_list = [] # empty list to hold figures
#     num_pages = len(active_kwarg_list)
#     for i, curr_batch_plot_kwargs in enumerate(active_kwarg_list):
#         curr_active_identifying_ctx = curr_batch_plot_kwargs['active_identifying_ctx']
#         # print(f'curr_active_identifying_ctx: {curr_active_identifying_ctx}')

#         curr_fig_save_basename = build_figure_basename_from_display_context(curr_active_identifying_ctx, context_tuple_join_character='_')

#         active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx, subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist)
#         # print(f'active_pdf_save_filename: {active_pdf_save_filename}')
#         curr_pdf_save_path = figures_parent_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)
#         # One plot at a time to PDF:
#         if write_pdf:
#             with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
#                 a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
#                 active_out_figures_list.append(a_fig)
#                 # Save out PDF page:
#                 pdf.savefig(a_fig)
#                 curr_active_pipeline.register_output_file(output_path=curr_pdf_save_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig), 'pdf_metadata': active_pdf_metadata})
#         else:
#             a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
#             active_out_figures_list.append(a_fig)

#         # Also save .png versions:
#         if write_png:
#             # curr_page_str = f'pg{i+1}of{num_pages}'
#             fig_png_out_path = curr_pdf_save_path.with_suffix('.png')
#             # fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
#             a_fig.savefig(fig_png_out_path)
#             curr_active_pipeline.register_output_file(output_path=fig_png_out_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig)})
#             if progress_print:
#                 print(f'\t saved {fig_png_out_path}')

#     return active_out_figures_list
