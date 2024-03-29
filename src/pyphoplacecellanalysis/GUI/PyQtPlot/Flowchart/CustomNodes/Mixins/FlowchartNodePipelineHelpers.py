from copy import deepcopy
from attrs import define, Factory, fields
from pathlib import Path
import numpy as np

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange

from pyphocorehelpers.indexing_helpers import compute_position_grid_size
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


@define(slots=False)
class FlowchartNodePipelineHelpers:
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
        return [DynamicParameters(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=FlowchartNodePipelineHelpers.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None))]
        

    # ==================================================================================================================== #
    # Main public methods used by the Flowchart Nodes:                                                                     #
    # ==================================================================================================================== #

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
        active_session_computation_configs, active_session_filter_configurations = FlowchartNodePipelineHelpers.bapun_format_define_configs(pipeline)
        pipeline = FlowchartNodePipelineHelpers.perform_filtering(pipeline, active_session_filter_configurations)
        pipeline = FlowchartNodePipelineHelpers.perform_computation(pipeline, active_session_computation_configs)
        return pipeline, active_session_computation_configs, active_session_filter_configurations
     
    @staticmethod
    def bapun_format_define_configs(curr_bapun_pipeline):
        # curr_bapun_pipeline = NeuropyPipeline(name='bapun_pipeline', session_data_type='bapun', basedir=known_data_session_type_dict['bapun'].basedir, load_function=known_data_session_type_dict['bapun'].load_function)
        # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        active_session_computation_configs = FlowchartNodePipelineHelpers._build_active_computation_configs(curr_bapun_pipeline.sess)
        # active_session_computation_config.grid_bin = compute_position_grid_bin_size(curr_bapun_pipeline.sess.position.x, curr_bapun_pipeline.sess.position.y, num_bins=(64, 64))
                
        # Bapun/DataFrame style session filter functions:
        def build_bapun_any_maze_epochs_filters(sess):
            # all_filters = build_custom_epochs_filters(sess)
            # # print(f'all_filters: {all_filters}')
            # maze_only_filters = dict()
            # for (name, filter_fcn) in all_filters.items():
            #     if 'maze' in name:
            #         maze_only_filters[name] = filter_fcn
            # maze_only_filters = build_custom_epochs_filters(sess, epoch_name_includelist=['maze1','maze2'])
            # { key:value for (key,value) in dictOfNames.items() if key % 2 == 0}
            # dict(filter(lambda elem: len(elem[1]) == 6,dictOfNames.items()))
            # maze_only_name_filter_fn = lambda dict: dict(filter(lambda elem: 'maze' in elem[0], dict.items()))
            maze_only_name_filter_fn = lambda names: list(filter(lambda elem: elem.startswith('maze'), names)) # include only maze tracks
            # print(f'callable(maze_only_name_filter_fn): {callable(maze_only_name_filter_fn)}')
            # print(maze_only_name_filter_fn(['pre', 'maze1', 'post1', 'maze2', 'post2']))
            # lambda elem: elem[0] % 2 == 0
            maze_only_filters = build_custom_epochs_filters(sess, epoch_name_includelist=maze_only_name_filter_fn)
            # print(f'maze_only_filters: {maze_only_filters}')
            return maze_only_filters

        active_session_filter_configurations = build_bapun_any_maze_epochs_filters(curr_bapun_pipeline.sess)
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = None  # set the placefield computation epochs to None, using all epochs.
        return active_session_computation_configs, active_session_filter_configurations


    # KDIBA ______________________________________________________________________________________________________________ #
    @staticmethod
    def kdiba_format_all(pipeline):
        active_session_computation_configs, active_session_filter_configurations = FlowchartNodePipelineHelpers.kdiba_format_define_configs(pipeline)
        pipeline = FlowchartNodePipelineHelpers.perform_filtering(pipeline, active_session_filter_configurations)
        pipeline = FlowchartNodePipelineHelpers.perform_computation(pipeline, active_session_computation_configs)
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
        active_session_computation_configs = FlowchartNodePipelineHelpers._build_active_computation_configs(curr_kdiba_pipeline.sess)
        
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
    
