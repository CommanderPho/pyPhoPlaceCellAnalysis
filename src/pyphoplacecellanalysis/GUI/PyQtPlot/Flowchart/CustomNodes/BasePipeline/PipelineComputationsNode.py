from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
from pyphoplacecellanalysis.External.pyqtgraph.widgets.ProgressDialog import ProgressDialog

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.GUI.Qt.Mixins.CheckTableCtrlOwningMixin import CheckTableCtrlOwningMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.MiscNodes.ExtendedCtrlNode import ExtendedCtrlNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.FlowchartNodePipelineHelpers import FlowchartNodePipelineHelpers


class PipelineComputationsNode(CheckTableCtrlOwningMixin, ExtendedCtrlNode):
    """Performs computations on the active pipeline"""
    nodeName = "PipelineComputationsNode"
    uiTemplate = [
        ('included_configs_table', 'extendedchecktable', {'columns': ['compute'], 'rows': []}),
        ('recompute', 'action')
    ]
    
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            'pipeline': dict(io='in'),
            'computation_configs': dict(io='in'),
            'updated_computation_configs': dict(io='out'),
            'computed_pipeline': dict(io='out'),
        }
        ExtendedCtrlNode.__init__(self, name, terminals=terminals)
        self.configRows = []
        self.connections = dict()
        self.ui_build()
                
    # @property
    # def enabled_computation_filters(self):
    #     """Gets the list of filters for which to do computations on from the current selections in the checkbox table UI. Returns a list of filter names that are enabled."""
    #     rows_state = self.ctrls['included_configs_table'].saveState()['rows']
    #     # print(f'\t {rows_state}') # [['row[0]', True, False], ['row[1]', False, False]]
    #     enabled_filter_names = []
    #     for a_row in rows_state:
    #         # ['row[0]', True, False]
    #         row_config_name = a_row[0]
    #         row_include_state = a_row[1]
    #         if row_include_state:
    #             enabled_filter_names.append(row_config_name)
    #     return enabled_filter_names                


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
        
        # Check Table:
        self.selectFirstConfigRow()        
        self.connections['checktable_state_changed_connection'] = self.ctrls['included_configs_table'].sigStateChanged.connect(self.on_checktable_checked_state_changed) # ExtendedCheckTable
        self.ui_update()
        
    def process(self, pipeline=None, computation_configs=None, display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        
        # print(f'PipelineComputationsNode.data_mode: {data_mode}')
        if (pipeline is None) or (computation_configs is None):
            self.updateConfigRows(computation_configs)
            return {'updated_computation_configs': computation_configs, 'computed_pipeline': None}

        assert (pipeline is not None), 'curr_pipeline is None but has no reason to be!'
        # Get the list of available functions:
        all_computation_functions_list = pipeline.registered_computation_function_names
        """
            ['_perform_placefield_overlap_computation',
            '_perform_firing_rate_trends_computation',
            '_perform_extended_statistics_computation',
            '_perform_two_step_position_decoding_computation',
            '_perform_position_decoding_computation']
        """
        # TODO:L allow selecting which of these are active too.
        
        # Gets the names of the filters applied and updates the config rows with them
        all_filters_list = list(pipeline.filtered_sessions.keys())
        self.updateConfigRows(all_filters_list)
        
        if len(self.enabled_filters) == 0:
            print(f'no enabled computation filters! Cannot perform computation!')
            pipeline = None
        else:
            with ProgressDialog("Pipeline Computations...", 0, 1, cancelText="Cancel", parent=None, busyCursor=True, wait=250) as dlg:
                pipeline = FlowchartNodePipelineHelpers.perform_computation(pipeline, computation_configs, enabled_filter_names=self.enabled_filters)

        return {'updated_computation_configs': computation_configs,'computed_pipeline': pipeline}


    ##############################################################
    def ui_update(self):
        """ called to update the ctrls depending on its properties. """
        self.ctrls['recompute'].setEnabled(self.is_action_enabled)

    @QtCore.pyqtSlot(object, object, object)
    def on_checktable_checked_state_changed(self, row, col, state):
        self.ui_update()
        
    
        
    def saveState(self):
        state = ExtendedCtrlNode.saveState(self)
        return {'config_rows':self.configRows, 'ctrls': state}
        
    def restoreState(self, state):
        self.updateConfigRows(state['config_rows'])
        ExtendedCtrlNode.restoreState(self, state['ctrls'])
