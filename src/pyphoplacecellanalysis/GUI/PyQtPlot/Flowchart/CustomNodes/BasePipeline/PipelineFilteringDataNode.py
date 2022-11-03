from typing import OrderedDict
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
from pyphoplacecellanalysis.External.pyqtgraph.widgets.ProgressDialog import ProgressDialog

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.NonInteractiveWrapper import NonInteractiveWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.MiscNodes.ExtendedCtrlNode import ExtendedCtrlNode
from pyphoplacecellanalysis.GUI.Qt.Mixins.CheckTableCtrlOwningMixin import CheckTableCtrlOwningMixin

# from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder


class PipelineFilteringDataNode(CheckTableCtrlOwningMixin, ExtendedCtrlNode):
    """Filters active pipeline"""
    nodeName = "PipelineFilteringDataNode"
    uiTemplate = [
        # ('included_configs', 'combo', {'values': [], 'index': 0}),
        ('included_configs_table', 'extendedchecktable', {'columns': ['filter'], 'rows': []}),
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
        ExtendedCtrlNode.__init__(self, name, terminals=terminals)
        self.configRows = []
        self.connections = dict()
        self.ui_build()


    def ui_build(self):
        # Setup the refilter button:
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
        
        self.selectFirstConfigRow()        
        self.connections['checktable_state_changed_connection'] = self.ctrls['included_configs_table'].sigStateChanged.connect(self.on_checktable_checked_state_changed) # ExtendedCheckTable 
        self.ui_update()
        

        
    def process(self, active_data_mode=None, pipeline=None, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
       
        # print(f'PipelineFilteringDataNode.data_mode: {data_mode}')
        if (pipeline is None) or (active_data_mode is None):
            updated_configs = [] # empty list, no options
            self.updateConfigRows(updated_configs)
            return {'active_session_computation_configs': None, 'active_session_filter_configs':None, 'filtered_pipeline': None}

        if active_data_mode is not None:
            active_data_session_type_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
            active_data_mode_type = active_data_session_type_dict.get(active_data_mode, None)
            if active_data_mode_type is not None:
                active_session_filter_configs = active_data_mode_type.build_default_filter_functions(sess=pipeline.sess)
                active_session_computation_configs = active_data_mode_type.build_default_computation_configs(sess=pipeline.sess)
            else:
                print(f'active_data_mode: {active_data_mode} was not found to match any registered types: {active_data_session_type_dict}!')
                active_session_filter_configs = {}
                active_session_computation_configs = {}

        assert (pipeline is not None), 'pipeline is None but has no reason to be!'
        
        # Update the available config selection options:
        updated_configs = list(active_session_filter_configs.keys()) # ['maze1', 'maze2']
        self.updateConfigRows(updated_configs)
        
        with ProgressDialog("Pipeline Filtering: {active_data_mode} Format..", 0, 1, cancelText="Cancel", parent=None, busyCursor=True, wait=250) as dlg:
            # build a list of only the enabled filters
             enabled_session_filter_configs = OrderedDict()
             for an_enabled_filter_name in self.enabled_filters:
                 enabled_session_filter_configs[an_enabled_filter_name] = active_session_filter_configs[an_enabled_filter_name]
             
             pipeline = NonInteractiveWrapper.perform_filtering(pipeline, enabled_session_filter_configs)
        
        return {'computation_configs': active_session_computation_configs, 'filter_configs':active_session_filter_configs, 'filtered_pipeline': pipeline}



    
    
    
    ##############################################################
    def ui_update(self):
        """ called to update the ctrls depending on its properties. """
        self.ctrls['refilter'].setEnabled(self.is_action_enabled)

    @QtCore.pyqtSlot(object, object, object)
    def on_checktable_checked_state_changed(self, row, col, state):
        self.ui_update()


    def saveState(self):
        state = ExtendedCtrlNode.saveState(self)
        # return {'keys': self.keys, 'ctrls': state}
        return {'config_rows':self.configRows, 'ctrls': state}
        
    def restoreState(self, state):
        # self.updateKeys(state['keys'])
        self.updateConfigRows(state['config_rows'])
        ExtendedCtrlNode.restoreState(self, state['ctrls'])
