from pathlib import Path
from typing import OrderedDict
from pyphoplacecellanalysis.External.pyqtgraph.flowchart import Flowchart, Node
import pyphoplacecellanalysis.External.pyqtgraph.flowchart.library as fclib
from pyphoplacecellanalysis.External.pyqtgraph.flowchart.library.common import CtrlNode
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
from pyphoplacecellanalysis.External.pyqtgraph.widgets.ProgressDialog import ProgressDialog
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.NonInteractiveWrapper import NonInteractiveWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.MiscNodes.ExtendedCtrlNode import ExtendedCtrlNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.CtrlNodeMixins import CheckTableCtrlOwnerMixin

# from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder


class PipelineFilteringDataNode(CheckTableCtrlOwnerMixin, ExtendedCtrlNode):
    """Filters active pipeline"""
    nodeName = "PipelineFilteringDataNode"
    uiTemplate = [
        # ('included_configs', 'combo', {'values': [], 'index': 0}),
        ('included_configs_table', 'extendedchecktable', {'columns': ['filter'], 'rows': ['test1', 'test2']}),
        ('refilter', 'action'),
    ]
    
    @property
    def enabled_filters(self):
        """Gets the list of filters for which to do filtering for from the current selections in the checkbox table UI. Returns a list of filter names that are enabled."""
        rows_state = self.ctrls['included_configs_table'].saveState()['rows']
        # print(f'\t {rows_state}') # [['row[0]', True, False], ['row[1]', False, False]]
        enabled_filter_names = []
        for a_row in rows_state:
            # ['row[0]', True, False]
            row_config_name = a_row[0]
            row_include_state = a_row[1]
            if row_include_state:
                enabled_filter_names.append(row_config_name)
        return enabled_filter_names                
    
    @property
    def is_action_enabled(self):
        """The is_action_enabled property."""
        return (len(self.enabled_filters) > 0) # if we have one or more enabled filter the action can be performed. Otherwise it's disabled.
    
    
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
        
        self.keys = [] # the active config keys
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
        
        if (len(self.ctrls['included_configs_table'].saveState()['rows']) > 1):
            # if we have one or more rows (columns are assumed to be fixed), set at least the first entry by default
            self.ctrls['included_configs_table'].set_value(0,0,True)
        
        self.connections['checktable_state_changed_connection'] = self.ctrls['included_configs_table'].sigStateChanged.connect(self.on_checktable_checked_state_changed) # ExtendedCheckTable 
        
        self.ui_update()
        

        
    def process(self, active_data_mode=None, pipeline=None, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        # data_mode = self.ctrls['data_mode'].value()
        
        # print(f'PipelineFilteringDataNode.data_mode: {data_mode}')
        if (pipeline is None) or (active_data_mode is None):
            updated_configs = [] # empty list, no options
            # self.updateKeys(updated_configs) # Update the possible keys
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

            # if active_data_mode == 'bapun':
            #     active_session_computation_configs, active_session_filter_configs = NonInteractiveWrapper.bapun_format_define_configs(pipeline)
            # elif active_data_mode == 'kdiba':
            #     active_session_computation_configs, active_session_filter_configs = NonInteractiveWrapper.kdiba_format_define_configs(pipeline)
            # else:
            #     curr_pipeline = None
            #     active_session_computation_configs = None
            #     active_session_filter_configs = None
            #     raise
            
        assert (pipeline is not None), 'pipeline is None but has no reason to be!'
        
        # Update the available config selection options:
        # updated_configs = list(pipeline.computation_results.keys()) # ['maze1', 'maze2']
        updated_configs = list(active_session_filter_configs.keys()) # ['maze1', 'maze2']
        self.updateConfigRows(updated_configs)
        self.ui_update()
        
        
        # selected_config_value = str(self.ctrls['included_configs'].currentText())
        # print(f'selected_config_value: {selected_config_value}; updated_configs: {updated_configs}')
        # self.updateKeys(updated_configs) # Update the possible keys
        
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

    def updateConfigRows(self, data):
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
        for c in [self.ctrls['included_configs_table']]:
            c.updateRows(keys) # update the rows with the config rows

        for c in self.ctrls.values():
            c.blockSignals(False)
        # Update the self.keys value:
        self.configRows = keys
        self.ui_update()

    @QtCore.pyqtSlot(object, object, object)
    def on_checktable_checked_state_changed(self, row, col, state):
        # print(f'_test_filtering_node_state_changed(row: {row}, col: {col}, state: {state})')
        # print(f'curr_filtering_node.enabled_filters: {self.enabled_filters}')
        self.ui_update()


    def saveState(self):
        state = ExtendedCtrlNode.saveState(self)
        # return {'keys': self.keys, 'ctrls': state}
        return {'config_rows':self.configRows, 'ctrls': state}
        
    def restoreState(self, state):
        # self.updateKeys(state['keys'])
        self.updateConfigRows(state['config_rows'])
        ExtendedCtrlNode.restoreState(self, state['ctrls'])
