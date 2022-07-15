# pyuic5 PlacefieldVisualSelectionControlsBarWidget.ui -o PlacefieldVisualSelectionControlsBarWidget.py -x
# PlacefieldVisualSelectionControlsBarWidgetBase
# pyuic5 PlacefieldVisualSelectionControlsBarWidgetBase.ui -o PlacefieldVisualSelectionControlsBarWidgetBase.py -x

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

from matplotlib.colors import to_hex # required for QColor conversion to hex

# from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControlsBar.PlacefieldVisualSelectionControlsBarWidgetBase import Ui_rootForm # Generated file from .ui
from Uic_AUTOGEN_PlacefieldVisualSelectionControlsBarWidgetBase import Ui_rootForm

class PlacefieldVisualSelectionControlsBarWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionControlsBarWidget."""
 
    # spike_config_changed = QtCore.pyqtSignal(list, list, bool) # change_unit_spikes_included(self, neuron_IDXs=None, cell_IDs=None, are_included=True)
    # tuning_curve_display_config_changed = QtCore.pyqtSignal(list, list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    # update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
    # finish_signal = QtCore.pyqtSignal(float, float)
 
    sigRefresh = QtCore.pyqtSignal(object)
    
    desired_full_panel_width = 1200
    desired_full_panel_height = 200
    
    enable_debug_print = False
    
    def __init__(self, *args, parent=None, **kwargs):
        super(PlacefieldVisualSelectionControlsBarWidget, self).__init__(*args, parent=parent, **kwargs)
        #Load the UI:
        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.desired_full_panel_width = PlacefieldVisualSelectionControlsBarWidget.desired_full_panel_width
        self.desired_full_panel_height = PlacefieldVisualSelectionControlsBarWidget.desired_full_panel_height
        
        # Final UI Refinements:
        self.initUI()
        
        # initialize member variables
        self.enable_debug_print = PlacefieldVisualSelectionControlsBarWidget.enable_debug_print
        
        
    @property
    def groupBox(self):
        return self.ui.placefieldControlsGroupbox
    
    @property
    def pf_layout(self):
        return self.ui.pf_layout
    
    @property
    def show_all_buttons_list(self):
        return (self.ui.btnShowAll_Pfs, self.ui.btnShowAll_Spikes, self.ui.btnShowAll_Both)
    @property
    def hide_all_buttons_list(self):
        return (self.ui.btnHideAll_Pfs, self.ui.btnHideAll_Spikes, self.ui.btnHideAll_Both)
        
        
    def initUI(self):
        # self.setObjectName('placefieldControlsContainer')
        # self.ui.btnToggleOccupancy
        
        # self.ui.batchControlPanel.hide()
        self.resize(self.desired_full_panel_width, self.desired_full_panel_height)
        
        self.ui.scroll_area.setFixedHeight(150)
        
        # Create a new layout to add the widgets to:
        self.ui.pf_layout = QtWidgets.QHBoxLayout()
        self.ui.pf_layout.setSpacing(0)
        self.ui.pf_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.pf_layout.setObjectName("horizontalLayout")
        
        ## TODO: add the widgets here:
        self.ui.pf_widgets = [] # the list of embedded child widgets
        
        ## Once the widgets are added to pf_layout, set the container to the layout:
        self.ui.placefieldControlsContainer.setLayout(self.ui.pf_layout)

        # Configure the "Refresh" Button:
        self.ui.btnRefresh.clicked.connect(self.onRefreshAction)
        self.rebuild_neuron_id_to_widget_map()
        
        
    @property
    def pf_widgets(self):
        """The pf_widgets property."""
        return self.ui.pf_widgets
    @pf_widgets.setter
    def pf_widgets(self, value):
        self.ui.pf_widgets = value
        self.rebuild_neuron_id_to_widget_map()
        
    @property
    def neuron_id_pf_widgets_map(self):
        """The neuron_id_pf_widgets_map property."""
        return self._neuron_id_pf_widgets_map
    @neuron_id_pf_widgets_map.setter
    def neuron_id_pf_widgets_map(self, value):
        self._neuron_id_pf_widgets_map = value
    
    def rebuild_neuron_id_to_widget_map(self):
        """ must be called after changing self.ui.pf_widgets """
        self._neuron_id_pf_widgets_map = dict()
        for a_widget in self.ui.pf_widgets:
            curr_widget_config = a_widget.config_from_state()
            self._neuron_id_pf_widgets_map[curr_widget_config.neuron_id] = a_widget 
        

    @QtCore.pyqtSlot()
    def onRefreshAction(self):
        print(f'PlacefieldVisualSelectionControlsBarWidget.onRefreshAction()')
        self.sigRefresh.emit(self)
        # self.done(QtCore.Qt.WA_DeleteOnClose)

    @QtCore.pyqtSlot(object)
    def applyUpdatedConfigs(self, active_configs_map):
        """ Updates the placefield Qt widgets provided in the neuron_id_pf_widgets_map from the active_configs_map
        
        Inputs:
            Both maps should have keys of neuron_id <int>
        
        Usage:
            ipcDataExplorer.neuron_id_pf_widgets_map = _build_id_index_configs_dict(pf_widgets)
            apply_updated_configs_to_pf_widgets(ipcDataExplorer.neuron_id_pf_widgets_map, active_configs_map)
        """
        print(f'PlacefieldVisualSelectionControlsBarWidget.applyUpdatedConfigs(active_configs_map: {active_configs_map})')
        ## Update placefield selection GUI widgets from updated configs:
        for neuron_id, updated_config in active_configs_map.items():
            """ Update the placefield selection GUI widgets from the updated configs using the .update_from_config(render_config) fcn """
            # update the widget:
            self.neuron_id_pf_widgets_map[neuron_id].update_from_config(updated_config)


    def configsFromStates(self):
        """ gets the current config from the state of each child pf_widget (a list of SingleNeuronPlottingExtended) """
        return [a_widget.config_from_state() for a_widget in self.ui.pf_widgets]
        
    def configMapFromChildrenWidgets(self):
        """ returns a map with keys of neuron_id and values of type SingleNeuronPlottingExtended """
        return {a_config.neuron_id:a_config for a_config in self.configsFromStates()}



    @classmethod
    def apply_updated_configs_to_pf_widgets(cls, neuron_id_pf_widgets_map, active_configs_map):
        """ Updates the placefield Qt widgets provided in the neuron_id_pf_widgets_map from the active_configs_map
            
        Inputs:
            Both maps should have keys of neuron_id <int>
        
        Usage:
            ipcDataExplorer.neuron_id_pf_widgets_map = _build_id_index_configs_dict(pf_widgets)
            apply_updated_configs_to_pf_widgets(ipcDataExplorer.neuron_id_pf_widgets_map, active_configs_map)
        """
        ## Update placefield selection GUI widgets from updated configs:
        for neuron_id, updated_config in active_configs_map.items():
            """ Update the placefield selection GUI widgets from the updated configs using the .update_from_config(render_config) fcn """
            # update the widget:
            neuron_id_pf_widgets_map[neuron_id].update_from_config(updated_config)


## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("PlacefieldVisualSelectionControlsBarWidget Example")
    widget = PlacefieldVisualSelectionControlsBarWidget()
    widget.show()
    pg.exec()

