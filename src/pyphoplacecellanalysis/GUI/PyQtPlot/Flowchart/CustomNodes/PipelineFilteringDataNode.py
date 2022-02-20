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
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.ExtendedCtrlNode import ExtendedCtrlNode


class PipelineFilteringDataNode(ExtendedCtrlNode):
    """Filters active pipeline"""
    nodeName = "PipelineFilteringDataNode"
    uiTemplate = [
        ('included_configs', 'combo', {'values': [], 'index': 0}),
        ('refilter', 'action'),
    ]
    
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'active_data_mode': dict(io='in'),
            'pipeline': dict(io='in'),
            'computation_configs': dict(io='out'),
            'filter_configs': dict(io='out'),
            'filtered_pipeline': dict(io='out'),
        }
        CtrlNode.__init__(self, name, terminals=terminals)
        self.keys = [] # the active config keys
        self.ui_build()

    
    def ui_build(self):
        # Setup the recompute button:
        self.ctrls['refilter'].setText('refilter')
        def click():
            self.ctrls['refilter'].processing("Hold on..")
            # Not sure whether to call self.changed() (from CtrlNode) or self.update() from its parent class.
            # self.update() 
            self.changed() # should trigger re-computation in a blocking manner.
            
            # global fail
            # fail = not fail
            
            fail = False
            if fail:
                self.ctrls['refilter'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['refilter'].success(message="Bueno!")
                
        self.ctrls['refilter'].clicked.connect(click)
        
        
    def process(self, active_data_mode=None, pipeline=None, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        # data_mode = self.ctrls['data_mode'].value()
        
        # print(f'PipelineFilteringDataNode.data_mode: {data_mode}')

        # active_known_data_session_type_dict = self._get_known_data_session_types_dict()
        # # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
        # curr_pipeline = NeuropyPipeline.init_from_known_data_session_type(data_mode, active_known_data_session_type_dict[data_mode])    
        if (pipeline is None) or (active_data_mode is None):
            updated_configs = [] # empty list, no options
            self.updateKeys(updated_configs) # Update the possible keys           
            return {'active_session_computation_configs': None, 'active_session_filter_configs':None,
                    'filtered_pipeline': None}

        if active_data_mode is not None:
            if active_data_mode == 'bapun':
                active_session_computation_configs, active_session_filter_configs = NonInteractiveWrapper.bapun_format_define_configs(pipeline)
            elif active_data_mode == 'kdiba':
                active_session_computation_configs, active_session_filter_configs = NonInteractiveWrapper.kdiba_format_define_configs(pipeline)
            else:
                curr_pipeline = None
                active_session_computation_configs = None
                active_session_filter_configs = None
                raise
            
        assert (pipeline is not None), 'pipeline is None but has no reason to be!'
        
        # Update the available config selection options:
        # updated_configs = list(pipeline.computation_results.keys()) # ['maze1', 'maze2']
        updated_configs = list(active_session_filter_configs.keys()) # ['maze1', 'maze2']
        selected_config_value = str(self.ctrls['included_configs'].currentText())
        print(f'selected_config_value: {selected_config_value}; updated_configs: {updated_configs}')
        self.updateKeys(updated_configs) # Update the possible keys
        
        with ProgressDialog("Pipeline Filtering: {active_data_mode} Format..", 0, 1, parent=None, busyCursor=True, wait=250) as dlg:
             pipeline = NonInteractiveWrapper.perform_filtering(pipeline, active_session_filter_configs)
        
        return {'computation_configs': active_session_computation_configs, 'filter_configs':active_session_filter_configs, 'filtered_pipeline': pipeline}


    def updateKeys(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print("Unknown data type:", type(data), data)
            return
            
        for c in self.ctrls.values():
            c.blockSignals(True)
        #for c in [self.ctrls['included_configs'], self.ctrls['y'], self.ctrls['size']]:
        for c in [self.ctrls['included_configs']]:
            cur = str(c.currentText())
            c.clear()
            for k in keys:
                c.addItem(k)
                if k == cur:
                    c.setCurrentIndex(c.count()-1)
        # for c in [self.ctrls['color'], self.ctrls['border']]:
        #     c.setArgList(keys)
        for c in self.ctrls.values():
            c.blockSignals(False)
        # Update the self.keys value:
        self.keys = keys
        

    def saveState(self):
        state = CtrlNode.saveState(self)
        return {'keys': self.keys, 'ctrls': state}
        
    def restoreState(self, state):
        self.updateKeys(state['keys'])
        CtrlNode.restoreState(self, state['ctrls'])
