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
from pyphoplacecellanalysis.GUI.Qt.Mixins.ComboBoxMixins import KeysListAccessingMixin, ComboBoxCtrlOwningMixin

# Neuropy:
# from neuropy.core.session.data_session_loader import DataSessionLoader
# from neuropy.analyses.laps import estimation_session_laps
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat


class PipelineInputDataNode(ComboBoxCtrlOwningMixin, ExtendedCtrlNode):
    """Configure, Load, and Return the input pipeline data as defined by a known data type (such as kdiba or Bapun)."""
    nodeName = "PipelineInputDataNode"
    uiTemplate = [
        ('data_mode', 'combo', {'values': ['custom...'], 'index': 0}),
        ('basedir', 'file'), # {'label': 'basedir', 'is_save_mode': False, 'path_type': 'folder', 'allows_multiple': False}
        ('reload', 'action'),
        # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
        # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    ]
    def __init__(self, name):
        ## Define the input / output terminals available on this node
        terminals = {
            # 'dataIn': dict(io='in'),    # each terminal needs at least a name and
            'known_mode': dict(io='in'),
            'override_basepath': dict(io='in'),
            'loaded_pipeline': dict(io='out'),  # to specify whether it is input or output
            'known_data_mode': dict(io='out'),
            'basedir': dict(io='out'),
        }                              # other more advanced options are available
                                       # as well..
        # Static:
        self.active_known_data_session_type_dict = PipelineInputDataNode._get_known_data_session_types_dict()
        self.active_known_data_session_type_class_names_dict = PipelineInputDataNode._get_known_data_session_type_class_names_dict()
        self.num_known_types = len(self.active_known_data_session_type_dict.keys())
        print(f'num_known_types: {self.num_known_types}')
        ExtendedCtrlNode.__init__(self, name, terminals=terminals)
        self.ui_build()
        self.ui_update()
                
        
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
        
        ## Add Custom File Control
        # W:\Data\Bapun\RatN\Day4OpenField
        self.ctrls['basedir'].sigFileSelectionChanged.connect(self.onBasedirPathChanged)
        

    def process(self, known_mode='', override_basepath=None, display=True):
    # def process(self, known_mode='Bapun', display=True):
        # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
        # data_mode = self.ctrls['data_mode'].value()                
        
        # Get the selected data_mode from the dropdown list either way
        s = self.stateGroup.state()
        if s['data_mode'] == 'bapun':
            data_mode_from_combo_list = 'bapun'
        elif s['data_mode'] == 'kdiba':
            data_mode_from_combo_list = 'kdiba'
        elif s['data_mode'] == 'rachel':
            data_mode_from_combo_list = 'rachel'
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
            
        # Get the data mode properties from the specified data_mode
        active_data_mode_registered_class = self.active_known_data_session_type_class_names_dict[data_mode]
        active_data_mode_type_properties = self.active_known_data_session_type_dict[data_mode]
        
        ## Prefer in the override_basepath input argument first if it is valid, and then the basedir from the ctrl widget, and then the default
        override_basedir_from_path_ctrl = str(self.ctrls['basedir'].path)
        print(f'override_basedir_from_path_ctrl: {override_basedir_from_path_ctrl}')
            
        if (override_basepath is None) or (override_basepath == ''):
            # No valid input argument for override_basepath
            if (override_basedir_from_path_ctrl is None) or (override_basedir_from_path_ctrl == '') or (override_basedir_from_path_ctrl == '.'):
                # invalid or no specified basepath in the ctrl:
                print(f'basedir not set from input variable or user ctrl: applying default of "{active_data_mode_type_properties.basedir}"')
                self.ctrls['basedir'].path = active_data_mode_type_properties.basedir # set to default directory
                basedir = active_data_mode_type_properties.basedir
            else:
                # potnetially valid override_basedir
                basedir = override_basedir_from_path_ctrl
         
        else:
            # valid input argument, use that one:
            if override_basedir_from_path_ctrl != override_basepath:
                # override in control is different than that specified in input
                print(f'path control mode: {override_basedir_from_path_ctrl} and input argument mode: {override_basepath} differ. Using input argument mode ({override_basepath}) currently.')
                self.ctrls['basedir'].path = override_basepath
            # either way, use this as the basedir    
            basedir = override_basepath
    
        # # Use this to set the 'basedir' path value:
        # self.ctrls['basedir'].path = active_data_mode_type_properties.basedir
        
        with ProgressDialog("Pipeline Input Loading..", 0, self.num_known_types, cancelText="Cancel", parent=None, busyCursor=True, wait=250) as dlg:
            # do stuff
            # dlg.setValue(0)   ## could also use dlg += 1
            # curr_bapun_pipeline = NeuropyPipeline.init_from_known_data_session_type('bapun', known_data_session_type_dict['bapun'])
            # curr_pipeline = NeuropyPipeline.init_from_known_data_session_type(data_mode, active_data_mode_type_properties, override_basepath=Path(basedir))
            curr_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(data_mode, active_data_mode_type_properties, override_basepath=Path(basedir))    
        
            # dlg.setValue(num_known_types)   ## could also use dlg += 1
            if dlg.wasCanceled():
                curr_pipeline = None
                raise Exception("Processing canceled by user")

            return {'known_data_mode': data_mode, 'loaded_pipeline': curr_pipeline}


    def ui_update(self, debug_print=False):
        """ called to update the ctrls depending on its properties. 
        Specific here rebuilds the UI (mainly the combo-box) by calling _get_known_data_session_types_dict(...) 
        """
        ## Update Combo box items:
        ## Freeze signals:
        curr_combo_box = self.ctrls['data_mode'] # QComboBox 
        curr_combo_box.blockSignals(True)
        
        ## Capture the previous selection:
        selected_index, selected_item_text = self.get_current_combo_item_selection(curr_combo_box, debug_print=debug_print)

        # Build updated list:
        self.active_known_data_session_type_dict = self._get_known_data_session_types_dict()
        self.num_known_types = len(self.active_known_data_session_type_dict.keys())
        ## Build updated list:
        updated_list = list(self.active_known_data_session_type_dict.keys())
        updated_list.append('Custom...')

        self.replace_combo_items(curr_combo_box, updated_list, debug_print=debug_print)
        
        ## Re-select the previously selected item if possible:
        # selected_item_text = 'kdiba'
        found_desired_index = self.try_select_combo_item_with_text(curr_combo_box, selected_item_text, debug_print=debug_print)
        
        ## Unblock the signals:        
        curr_combo_box.blockSignals(False)


    @QtCore.pyqtSlot(str)
    def onBasedirPathChanged(self, updated_basedir):
        print(f'onBasedirPathChanged(updated_basedir: {updated_basedir})')
        # self.ctrls['basedir'].path
        s = self.stateGroup.state()
        print(f'\ts: {s}')
        self.update()
        
    @classmethod
    def _get_known_data_session_types_dict(cls):
        """ a static accessor for the knwon data session types. Note here the default paths and such are defined. """
        return DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()

    @classmethod
    def _get_known_data_session_type_class_names_dict(cls):
        """ a static accessor for the knwon data session types. Note here the default paths and such are defined. """
        return DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    
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


