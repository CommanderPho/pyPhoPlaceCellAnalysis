from pathlib import Path
from pyphoplacecellanalysis.External.pyqtgraph.flowchart import Flowchart, Node
import pyphoplacecellanalysis.External.pyqtgraph.flowchart.library as fclib
from pyphoplacecellanalysis.External.pyqtgraph.flowchart.library.common import CtrlNode
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
from pyphoplacecellanalysis.External.pyqtgraph.widgets.ProgressDialog import ProgressDialog
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # get_neuron_identities
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.MiscNodes.ExtendedCtrlNode import ExtendedCtrlNode

# Neuropy:
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.analyses.laps import estimation_session_laps


class PipelineInputDataNode(ExtendedCtrlNode):
    """Configure, Load, and Return the input pipeline data as defined by a known data type (such as kdiba or Bapun)."""
    nodeName = "PipelineInputDataNode"
    uiTemplate = [
        ('data_mode', 'combo', {'values': ['bapun', 'kdiba', 'custom...'], 'index': 0}),
        ('reload', 'action'),
        # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
        # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    ]
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            # 'dataIn': dict(io='in'),    # each terminal needs at least a name and
            'known_mode': dict(io='in'),
            'loaded_pipeline': dict(io='out'),  # to specify whether it is input or output
            'known_data_mode': dict(io='out'),
        }                              # other more advanced options are available
                                       # as well..
        # Static:
        self.active_known_data_session_type_dict = PipelineInputDataNode._get_known_data_session_types_dict()
        self.num_known_types = len(self.active_known_data_session_type_dict.keys())
        print(f'num_known_types: {self.num_known_types}')
        ExtendedCtrlNode.__init__(self, name, terminals=terminals)
        self.ui_build()
        
                
        
    def ui_build(self):
        # Setup the reload button:
        self.ctrls['reload'].setText('Reload')
        def click():
            self.ctrls['reload'].processing("Hold on..")
            self.changed() # should trigger re-computation in a blocking manner.
            # global fail
            # fail = not fail
            fail = False
            if fail:
                self.ctrls['reload'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['reload'].success(message="Bueno!")
                
        # self.ctrls['reload'].clicked.connect(click)
        

    def process(self, known_mode='', display=True):
    # def process(self, known_mode='Bapun', display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        # data_mode = self.ctrls['data_mode'].value()                
        
        # Get the selected data_mode from the dropdown list either way
        s = self.stateGroup.state()
        if s['data_mode'] == 'bapun':
            data_mode_from_combo_list = 'bapun'
        elif s['data_mode'] == 'kdiba':
            data_mode_from_combo_list = 'kdiba'
        else:
            # raise NotImplementedError
            # Data mode from input terminal:
            data_mode_from_combo_list = known_mode

        print(f'PipelineInputDataNode data_mode from dropdown list: {data_mode_from_combo_list}')
        
        # Compare to known_mode from input:
        if (known_mode is None):
            print('Warning: known_mode is None.')
            data_mode = data_mode_from_combo_list
            # return {'known_data_mode': None, 'loaded_pipeline': None}
        elif (known_mode == ''):
            print('Warning: known_mode is the empty string!.')
            data_mode = data_mode_from_combo_list
            # return {'known_data_mode': known_mode, 'loaded_pipeline': None}
        else:
            print(f'PipelineInputDataNode.process(known_mode: {known_mode}, display: {display})...')
            
            if data_mode_from_combo_list != known_mode:
                print(f'dropdown mode: {data_mode_from_combo_list} and input argument mode: {known_mode} differ. Using input argument mode ({known_mode}) currently.')
                data_mode = known_mode
                search_text = known_mode
                found_desired_index = self.ctrls['data_mode'].findText(search_text)
                print(f'search_text: {search_text}, found_desired_index: {found_desired_index}')

                self.ctrls['data_mode'].setCurrentIndex(found_desired_index)
            else:
                data_mode = data_mode_from_combo_list
                
            # self.stateGroup.
            # raise NotImplementedError

        with ProgressDialog("Pipeline Input Loading..", 0, self.num_known_types, cancelText="Cancel", parent=None, busyCursor=True, wait=250) as dlg:
            # do stuff
            # dlg.setValue(0)   ## could also use dlg += 1
            # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
            curr_pipeline = NeuropyPipeline.init_from_known_data_session_type(data_mode, self.active_known_data_session_type_dict[data_mode])    
            # dlg.setValue(num_known_types)   ## could also use dlg += 1
            if dlg.wasCanceled():
                curr_pipeline = None
                raise Exception("Processing canceled by user")

            return {'known_data_mode': data_mode, 'loaded_pipeline': curr_pipeline}


    @classmethod
    def _get_known_data_session_types_dict(cls):
        """ a static accessor for the knwon data session types. Note here the default paths and such are defined. """
        known_data_session_type_dict = {'kdiba':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
                                    basedir=Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')),
                    'bapun':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir)),
                                    basedir=Path(r'R:\data\Bapun\Day5TwoNovel'))
                    }
        known_data_session_type_dict['kdiba'].post_load_functions = [lambda a_loaded_sess: estimation_session_laps(a_loaded_sess)]
        return known_data_session_type_dict



# class PipelineResultBreakoutNode(CtrlNode):
#     """Breaks out results from active pipeline"""
#     nodeName = "PipelineResultBreakoutNode"
#     def __init__(self, name):
#         ## Define the input / output terminals available on this node
#         terminals = {
#             'active_data_mode': dict(io='in'),
#             'active_session_computation_configs': dict(io='in'),
#             'active_session_filter_configurations': dict(io='in'),
#             'pipeline': dict(io='in'),
#             'sess': dict(io='out'),
#             'pf1D': dict(io='out'),
#             'active_one_step_decoder': dict(io='out'),
#             'active_two_step_decoder': dict(io='out'),
#             'active_measured_positions': dict(io='out'),
#         }
#         CtrlNode.__init__(self, name, terminals=terminals)
        
#     def process(self, active_data_mode=None, active_session_computation_configs=None, active_session_filter_configurations=None, pipeline=None, display=True):
                
#         if ((pipeline is None) or (active_data_mode is None)):
#             return {'active_session_computation_configs': None, 'active_session_filter_configurations':None,
#                     'filtered_pipeline': None}

#         active_config_name = 'maze1'
#         # Get relevant variables:
#         # curr_pipeline is set above, and usable here
#         sess = pipeline.filtered_sessions[active_config_name]
#         pf1D = pipeline.computation_results[active_config_name].computed_data['pf1D']
#         active_one_step_decoder = pipeline.computation_results[active_config_name].computed_data['pf2D_Decoder']
#         active_two_step_decoder = pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
#         active_measured_positions = pipeline.computation_results[active_config_name].sess.position.to_dataframe()
#         {'sess':sess, 'pf1D':pf1D, 'active_one_step_decoder': active_one_step_decoder, 'active_two_step_decoder': active_two_step_decoder, 'active_measured_positions': active_measured_positions}
    
#         return {'active_session_computation_configs': active_session_computation_configs, 'active_session_filter_configurations':active_session_filter_configurations, 'filtered_pipeline': curr_pipeline}


