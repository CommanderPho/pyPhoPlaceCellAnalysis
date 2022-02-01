from pathlib import Path
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


from pyphocorehelpers.indexing_helpers import compute_position_grid_size
from pyphoplacecellanalysis.General.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # get_neuron_identities
from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session


# Neuropy:
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.analyses.laps import estimation_session_laps
from neuropy.core.epoch import NamedTimerange

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_spike_counts, debug_print_subsession_neuron_differences
from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables


# class AsType(CtrlNode):
#     """Convert an array to a different dtype.
#     """
#     nodeName = 'AsType'
#     uiTemplate = [
#         ('dtype', 'combo', {'values': ['float', 'int', 'float32', 'float64', 'float128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'], 'index': 0}),
#     ]
    
#     def processData(self, data):
#         s = self.stateGroup.state()
#         return data.astype(s['dtype'])



class PipelineInputDataNode(CtrlNode):
    """Return the input data passed through an unsharp mask."""
    nodeName = "PipelineInputDataNode"
    uiTemplate = [
        ('data_mode', 'combo', {'values': ['bapun', 'kdiba', 'custom...'], 'index': 0}),
        # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
        # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    ]
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'dataIn': dict(io='in'),    # each terminal needs at least a name and
            'dataOut': dict(io='out'),  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        
        CtrlNode.__init__(self, name, terminals=terminals)
        
    def process(self, dataIn, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        data_mode = self.ctrls['data_mode'].value()
        
        print(f'PipelineInputDataNode.data_mode: {data_mode}')
        active_known_data_session_type_dict = self._get_known_data_session_types_dict()
        # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        curr_active_pipeline = NeuropyPipeline.init_from_known_data_session_type(data_mode, active_known_data_session_type_dict[data_mode])    
    
        return {'dataOut': curr_active_pipeline}

    def _get_known_data_session_types_dict(self):
        known_data_session_type_dict = {'kdiba':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
                                    basedir=Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')),
                    'bapun':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir)),
                                    basedir=Path('R:\data\Bapun\Day5TwoNovel'))
                    }
        known_data_session_type_dict['kdiba'].post_load_functions = [lambda a_loaded_sess: estimation_session_laps(a_loaded_sess)]
        return known_data_session_type_dict






class NonInteractiveWrapper(object):
    """docstring for NonInteractiveWrapper."""
    def __init__(self, enable_saving_to_disk=False):
        super(NonInteractiveWrapper, self).__init__()
        self.enable_saving_to_disk = False
        # common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\Final Placemaps 2021-01-14')
        self.common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\2022-01-16')
        
    
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
        # active_grid_bin = compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64))
        # active_session_computation_config.computation_epochs = None # set the placefield computation epochs to None, using all epochs.
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]
        return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=NonInteractiveWrapper.compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(2.0, 2.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None)]
        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(3.777, 1.043), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None)]

        # return [PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(32, 32)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(64, 64)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #         PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=compute_position_grid_bin_size(sess.position.x, sess.position.y, num_bins=(128, 128)), smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5, computation_epochs = None),
        #        ]
        
        
        
    def bapun_format(curr_bapun_pipeline):
        # curr_bapun_pipeline = NeuropyPipeline(name='bapun_pipeline', session_data_type='bapun', basedir=known_data_session_type_dict['bapun'].basedir, load_function=known_data_session_type_dict['bapun'].load_function)
        # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        active_session_computation_configs = NonInteractiveWrapper._build_active_computation_configs(curr_bapun_pipeline.sess)
        # active_session_computation_config.grid_bin = compute_position_grid_bin_size(curr_bapun_pipeline.sess.position.x, curr_bapun_pipeline.sess.position.y, num_bins=(64, 64))
                
        # Bapun/DataFrame style session filter functions:
        def build_bapun_any_maze_epochs_filters(sess):
            def _temp_filter_session_by_epoch1(sess):
                """ 
                Usage:
                    active_session, active_epoch = _temp_filter_session(curr_bapun_pipeline.sess)
                """
                active_epoch = sess.epochs.get_named_timerange('maze1')
                ## All Spikes:
                # active_epoch_session = sess.filtered_by_epoch(active_epoch) # old
                active_session = batch_filter_session(sess, sess.position, sess.spikes_df, active_epoch.to_Epoch())
                return active_session, active_epoch

            def _temp_filter_session_by_epoch2(sess):
                """ 
                Usage:
                    active_session, active_epoch = _temp_filter_session(curr_bapun_pipeline.sess)
                """
                active_epoch = sess.epochs.get_named_timerange('maze2')
                ## All Spikes:
                # active_epoch_session = sess.filtered_by_epoch(active_epoch) # old
                active_session = batch_filter_session(sess, sess.position, sess.spikes_df, active_epoch.to_Epoch())
                return active_session, active_epoch

            # return {'maze1':_temp_filter_session_by_epoch1}
            return {'maze1':_temp_filter_session_by_epoch1,
                    'maze2':_temp_filter_session_by_epoch2
                }

            # return {'maze1': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze1')), x.epochs.get_named_timerange('maze1')),
            #         'maze2': lambda x: (x.filtered_by_epoch(x.epochs.get_named_timerange('maze2')), x.epochs.get_named_timerange('maze2'))
            #        }

        active_session_filter_configurations = build_bapun_any_maze_epochs_filters(curr_bapun_pipeline.sess)
        curr_bapun_pipeline.filter_sessions(active_session_filter_configurations)
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = None  # set the placefield computation epochs to None, using all epochs.
        curr_bapun_pipeline.perform_computations(active_session_computation_configs[0])
        curr_bapun_pipeline.prepare_for_display() # TODO: pass a display config
        # Set curr_active_pipeline for testing:
        curr_active_pipeline = curr_bapun_pipeline
        return curr_active_pipeline



    def kdiba_format(curr_kdiba_pipeline):
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
        curr_kdiba_pipeline.filter_sessions(active_session_filter_configurations)
        for i in np.arange(len(active_session_computation_configs)):
            active_session_computation_configs[i].computation_epochs = None # add the laps epochs to all of the computation configs.

        curr_kdiba_pipeline.perform_computations(active_session_computation_configs[0])
        curr_kdiba_pipeline.prepare_for_display() # TODO: pass a display config
        return curr_kdiba_pipeline
        
        # # set curr_active_pipeline for testing:
        # curr_active_pipeline = curr_kdiba_pipeline
        


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

        curr_kdiba_pipeline.filter_sessions(active_session_filter_configurations)
        curr_kdiba_pipeline.perform_computations(active_session_computation_configs[0]) # Causes "IndexError: index 59 is out of bounds for axis 0 with size 59"
        curr_kdiba_pipeline.prepare_for_display() # TODO: pass a display config
        
        return curr_kdiba_pipeline
    
        # # set curr_active_pipeline for testing:
        # curr_active_pipeline = curr_kdiba_pipeline



