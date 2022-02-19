from pathlib import Path
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.widgets.ProgressDialog import ProgressDialog
import pyqtgraph as pg
import numpy as np

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.NonInteractiveWrapper import NonInteractiveWrapper



class PipelineComputationsNode(CtrlNode):
    """Performs computations on the active pipeline"""
    nodeName = "PipelineComputationsNode"
    uiTemplate = [
        ('recompute', 'action'),
    ]
    
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'pipeline': dict(io='in'),
            'computation_configs': dict(io='in'),
            'computation_configs': dict(io='out'),
            'computed_pipeline': dict(io='out'),
        }
        CtrlNode.__init__(self, name, terminals=terminals)
        self.ui_build()

    
    def ui_build(self):
        # Setup the recompute button:
        self.ctrls['recompute'].setText('recompute')
        def click():
            self.ctrls['recompute'].processing("Hold on..")
            # Not sure whether to call self.changed() (from CtrlNode) or self.update() from its parent class.
            # self.update() 
            self.changed() # should trigger re-computation in a blocking manner.
            
            # global fail
            # fail = not fail
            
            fail = False
            if fail:
                self.ctrls['recompute'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['recompute'].success(message="Bueno!")
                
        self.ctrls['recompute'].clicked.connect(click)
        
        
    def process(self, pipeline=None, computation_configs=None, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        # data_mode = self.ctrls['data_mode'].value()
        
        # print(f'PipelineComputationsNode.data_mode: {data_mode}')

        # active_known_data_session_type_dict = self._get_known_data_session_types_dict()
        # # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        # curr_pipeline = NeuropyPipeline.init_from_known_data_session_type(data_mode, active_known_data_session_type_dict[data_mode])    
        if (pipeline is None) or (computation_configs is None):
            return {'computation_configs': computation_configs, 'computed_pipeline': None}

        assert (pipeline is not None), 'curr_pipeline is None but has no reason to be!'
        with ProgressDialog("Pipeline Input Loading: Bapun Format..", 0, 1, parent=None, busyCursor=True, wait=250) as dlg:
            pipeline = NonInteractiveWrapper.perform_computation(pipeline, computation_configs)

        return {'computation_configs': computation_configs,'computed_pipeline': pipeline}


    def saveState(self):
        state = CtrlNode.saveState(self)
        return {'ctrls': state}
        
    def restoreState(self, state):
        CtrlNode.restoreState(self, state['ctrls'])
