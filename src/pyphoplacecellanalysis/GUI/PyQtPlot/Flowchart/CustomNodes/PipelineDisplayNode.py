from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph as pg
import numpy as np


from pyphoplacecellanalysis.General.Pipeline.Stages.Display import DefaultDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Ratemaps import DefaultRatemapDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DefaultDecoderDisplayFunctions


class PipelineDisplayNode(CtrlNode):
    """Displays active pipeline"""
    nodeName = "PipelineDisplayNode"
    # uiTemplate = [
    #     ('filter_function', 'combo', {'values': ['test1', 'test2', 'custom...'], 'index': 0}),
    #     # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
    #     # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    # ]
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'active_data_mode': dict(io='in'),
            'active_session_computation_configs': dict(io='in'),
            'active_session_filter_configurations': dict(io='in'),
            'active_pipeline': dict(io='in'),
            'display_outputs': dict(io='out'),            
        }
        CtrlNode.__init__(self, name, terminals=terminals)
        
    def process(self, active_data_mode=None, active_session_computation_configs=None, active_session_filter_configurations=None, active_pipeline=None, display=True):
                
        if (active_pipeline is None) or (not display):
            return  {'display_outputs': None}


        active_config_name = 'maze1'
        display_outputs = active_pipeline.display(DefaultDecoderDisplayFunctions._display_two_step_decoder_prediction_error_2D, active_config_name, variable_name='p_x_given_n') # works!
 
        return {'display_outputs': display_outputs}

