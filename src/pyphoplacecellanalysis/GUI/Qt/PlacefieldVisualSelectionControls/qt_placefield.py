# TODO: Implement the functionality present in panel_placefield.py for the new PlacefieldVisualSelectionWidget Qt widget.

from functools import partial
import numpy as np

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.PlacefieldVisualSelectionControlWidget import PlacefieldVisualSelectionWidget

from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControlsBar.PlacefieldVisualSelectionControlsBarWidget import PlacefieldVisualSelectionControlsBarWidget
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended

""" 

This whole file serves to emulate the structure of panel_placefield.py, which does the exact same thing (renders interactive placefield toggle controls) but using the Panel library instead of Qt.

Primary Function: build_qt_interactive_placefield_visibility_controls(...)
     Internally calls build_all_placefield_output_panels(...) to build the output widgets.


"""


def build_single_placefield_output_widget(render_config) -> PlacefieldVisualSelectionWidget:
    """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation 
    
    Called in build_all_placefield_output_panels(...) down below.
    
    """
    curr_pf_string = f'pf[{render_config.name}]'
    curr_widget = PlacefieldVisualSelectionWidget(config=render_config) # new widget type
    curr_widget.setObjectName(curr_pf_string)
    
    curr_widget.update_from_config(render_config) # is this the right type of config? I think it is.

    return curr_widget


@function_attributes(short_name=None, tags=['widget'], input_requires=[], output_provides=[], uses=[], used_by=['build_batch_interactive_placefield_visibility_controls'], creation_date='2025-06-27 00:59', related_items=[])
class BatchActionsEndButtonPanelHelper(object):
    """ Enables performing batch actions on the placefields, such as hiding all pfs/spikes, etc.
        Analagous to PlacefieldBatchActionsEndButtonPanel 
        
        Used only within build_batch_interactive_placefield_visibility_controls(...)
    """
    debug_logging = True
    
    def __init__(self, update_included_cell_Indicies_callback=None, hide_all_callback=None, show_all_callback=None):
        # super(BatchActionsEndButtonPanelHelper, self).__init__()
        self.final_update_included_cell_Indicies_callback = None
        self.hide_all_callback = None
        self.show_all_callback = None
        self.setup_BatchActionsEndButtonPanelHelper(update_included_cell_Indicies_callback=update_included_cell_Indicies_callback, hide_all_callback=hide_all_callback, show_all_callback=show_all_callback)
        
        
    def setup_BatchActionsEndButtonPanelHelper(self, update_included_cell_Indicies_callback=None, hide_all_callback=None, show_all_callback=None):
        self.final_update_included_cell_Indicies_callback = None
        if update_included_cell_Indicies_callback is not None:
            if callable(update_included_cell_Indicies_callback):
                self.final_update_included_cell_Indicies_callback = update_included_cell_Indicies_callback

        self.hide_all_callback = hide_all_callback
        self.show_all_callback = show_all_callback
        
    # @QtCore.pyqtSlot(bool)
    def btn_hide_all_callback(self, event):
        if self.debug_logging:
            print('BatchActionsEndButtonPanelHelper.btn_hide_all_callback(...)')
        if self.hide_all_callback is not None:
            if callable(self.hide_all_callback):
                self.hide_all_callback()

    # @QtCore.pyqtSlot(bool)
    def btn_show_all_callback(self, event):
        if self.debug_logging:
            print('BatchActionsEndButtonPanelHelper.btn_show_all_callback(...)')
        if self.show_all_callback is not None:
            if callable(self.show_all_callback):
                self.show_all_callback()

    # @QtCore.pyqtSlot(bool)
    def btn_update_active_placefields(self, event):
        if self.debug_logging:
            print('BatchActionsEndButtonPanelHelper.btn_update_active_placefields(...)')


@function_attributes(short_name=None, tags=['widget'], input_requires=[], output_provides=[], uses=['BatchActionsEndButtonPanelHelper'], used_by=['build_all_placefield_output_panels'], creation_date='2025-06-27 00:58', related_items=[])
def build_batch_interactive_placefield_visibility_controls(rootControlsBarWidget, ipcDataExplorer, debug_logging=False):
    """Builds a panel containing a series of widgets that control the spike/placemap/etc visibility for each placecell

        Internally calls build_all_placefield_output_panels(...) to build the output panel widgets.
        
        
        Performs updates by calling ipcDataExplorer.change_unit_spikes_included(...)
        
    Args:
        ipcDataExplorer ([type]): [description]

    Returns:
        [type]: [description]
        
    Usage:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        pane
    """
   
    
    
    ################################################
    ####### Spikes Visibility callbacks
    def _update_all_spikes_visibility(new_is_visible, apply_changes_on_finish=False):
        if debug_logging:
            print(f'EndButtonPanel._update_all_spikes_visibility(new_is_visible: {new_is_visible}, apply_changes_on_finish={apply_changes_on_finish})')
            
        rootConfigControlsWidget = ipcDataExplorer.ui['placefieldControlsContainerWidget']
        # rootConfigControlsWidget = rootControlsBarWidget
        changed_configs = {}
        for neuron_id, a_config in ipcDataExplorer.active_neuron_render_configs_map.items():
            did_change = (a_config.spikesVisible != new_is_visible)
            if did_change:
                a_config.spikesVisible = new_is_visible
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].blockSignals(True)
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].update_from_config(a_config) # update it since it changed
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].blockSignals(False)
                changed_configs[neuron_id] = a_config
        if apply_changes_on_finish:
            # rootConfigControlsWidget.applyUpdatedConfigs(active_configs_map=changed_configs)
            changed_neuron_ids = np.array(list(changed_configs.keys()))
            if debug_logging:
                print(f'changed_neuron_ids: {changed_neuron_ids}')
            ipcDataExplorer.change_unit_spikes_included(cell_IDs=changed_neuron_ids, are_included=new_is_visible)
            

    def _btn_hide_all_spikes_callback():
        if debug_logging:
            print('EndButtonPanel._btn_hide_all_spikes_callback(...)')
        _update_all_spikes_visibility(False, apply_changes_on_finish=True) # make all visibile
  
    def _btn_show_all_spikes_callback():
        if debug_logging:
            print('EndButtonPanel._btn_show_all_spikes_callback(...)')
        _update_all_spikes_visibility(True, apply_changes_on_finish=True) # make all visibile

    ################################################
    ####### Placefields visibility callbacks
    def _update_all_placefields_visibility(new_is_visible, apply_changes_on_finish=False):
        if debug_logging:
            print(f'EndButtonPanel._update_all_placefields_visibility(new_is_visible: {new_is_visible}, apply_changes_on_finish={apply_changes_on_finish})')
        rootConfigControlsWidget = ipcDataExplorer.ui['placefieldControlsContainerWidget']
        changed_configs = {}
        for neuron_id, a_config in ipcDataExplorer.active_neuron_render_configs_map.items():
            did_change = (a_config.isVisible != new_is_visible)
            if did_change:
                a_config.isVisible = new_is_visible
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].blockSignals(True)
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].update_from_config(a_config) # update it since it changed
                rootConfigControlsWidget.neuron_id_pf_widgets_map[neuron_id].blockSignals(False)
                changed_configs[neuron_id] = a_config
        if apply_changes_on_finish:
            changed_neuron_ids = np.array(list(changed_configs.keys()))
            if debug_logging:
                print(f'changed_neuron_ids: {changed_neuron_ids}')
            # rootConfigControlsWidget.applyUpdatedConfigs(active_configs_map=changed_configs)
            # ipcDataExplorer.update_active_placefields(placefield_indicies=[])
            ipcDataExplorer.apply_tuning_curve_configs()
            
    def _btn_hide_all_pfs_callback():
        if debug_logging:
            print('EndButtonPanel._btn_hide_all_pfs_callback(...)')
        _update_all_placefields_visibility(False, apply_changes_on_finish=True) # make all visibile
  
    def _btn_show_all_pfs_callback():
        if debug_logging:
            print('EndButtonPanel._btn_show_all_pfs_callback(...)')
        _update_all_placefields_visibility(True, apply_changes_on_finish=True) # make all visibile

    ################################################
    ####### Both/All visibility callbacks
    def _btn_hide_all_callback():
        if debug_logging:
            print('EndButtonPanel.btn_hide_all_callback(...)')
        _btn_hide_all_pfs_callback()
        _btn_hide_all_spikes_callback()
        
    def _btn_show_all_callback():
        if debug_logging:
            print('EndButtonPanel.btn_show_all_callback(...)')
        _btn_show_all_pfs_callback()
        _btn_show_all_spikes_callback()

    ################################################
    ####### Misc Features callbacks

    # Setup btnToggleOccupancy:
    def _btn_toggle_occupancy_callback():
        """ 
        Implicitly captures: 
            ipcDataExplorer
            rootControlsBarWidget.ui.btnToggleOccupancy
        """
        if debug_logging:
            print('EndButtonPanel._btn_toggle_occupancy_callback(...)')
        occupancy_button_checked_state_is_checked = rootControlsBarWidget.ui.btnToggleOccupancy.isChecked()
        ipcDataExplorer.occupancy_plotting_config.isVisible = occupancy_button_checked_state_is_checked # Set the occupancy_plotting_config.isVisible property to the updated is_checked state
        ipcDataExplorer.p.render()

    ## Setup btnNonPlacefieldSpikes:
    def _btn_toggle_pf_only_spikes():
        """ 
        Implicitly captures: 
            ipcDataExplorer
            rootControlsBarWidget.ui.btnNonPlacefieldSpikes
        """
        if debug_logging:
            print('EndButtonPanel._btn_toggle_pf_only_spikes(...)')
        non_pf_spikes_button_checked_state_is_checked = rootControlsBarWidget.ui.btnNonPlacefieldSpikes.isChecked()
        ipcDataExplorer.params.should_display_non_pf_spikes = non_pf_spikes_button_checked_state_is_checked # Set the params.should_display_non_pf_spikes property to the updated is_checked state
        ipcDataExplorer.update_spikes() # call update spikes on value change to render the updated spikes    
        ipcDataExplorer.p.render()


    def _btn_perform_refresh_callback():
        if debug_logging:
            print('EndButtonPanel._btn_perform_refesh_callback(...)')

        ## TODO: perform update
        ipcDataExplorer.update_spikes()
        ipcDataExplorer.p.render()
                
        
    end_button_helper_obj = BatchActionsEndButtonPanelHelper(hide_all_callback=_btn_hide_all_callback, show_all_callback=_btn_show_all_callback)
    
    btnShowAll_Pfs, btnShowAll_Spikes, btnShowAll_Both = rootControlsBarWidget.show_all_buttons_list
    btnHideAll_Pfs, btnHideAll_Spikes, btnHideAll_Both = rootControlsBarWidget.hide_all_buttons_list
    
    connections = []
    buttons = [btnShowAll_Pfs, btnShowAll_Spikes, btnShowAll_Both, btnHideAll_Pfs, btnHideAll_Spikes, btnHideAll_Both]
    
    connections.append(btnShowAll_Both.clicked.connect(end_button_helper_obj.btn_show_all_callback))
    connections.append(btnHideAll_Both.clicked.connect(end_button_helper_obj.btn_hide_all_callback))
    
    connections.append(btnHideAll_Pfs.clicked.connect(_btn_hide_all_pfs_callback))
    connections.append(btnShowAll_Pfs.clicked.connect(_btn_show_all_pfs_callback))
    
    connections.append(btnHideAll_Spikes.clicked.connect(_btn_hide_all_spikes_callback))
    connections.append(btnShowAll_Spikes.clicked.connect(_btn_show_all_spikes_callback))
    
    # Connect any extra signals:
    
    ## Setup btnToggleOccupancy:
    # Update the toggle button's state based on the occupancy_plotting_config.isVisible property:
    rootControlsBarWidget.ui.btnToggleOccupancy.setChecked(ipcDataExplorer.occupancy_plotting_config.isVisible) # Set Initial button state 
    connections.append(rootControlsBarWidget.ui.btnToggleOccupancy.clicked.connect(_btn_toggle_occupancy_callback)) # Connect
    
    ## Setup btnNonPlacefieldSpikes:
    # Update the toggle button's state based on the ipcDataExplorer.params.should_display_non_pf_spikesproperty:
    rootControlsBarWidget.ui.btnNonPlacefieldSpikes.setChecked(ipcDataExplorer.params.setdefault('should_display_non_pf_spikes', False))
    connections.append(rootControlsBarWidget.ui.btnNonPlacefieldSpikes.clicked.connect(_btn_toggle_pf_only_spikes))
    
    ## Refresh Button:
    connections.append(rootControlsBarWidget.sigRefresh.connect(_btn_perform_refresh_callback))
    
    return end_button_helper_obj, connections


@function_attributes(short_name=None, tags=['widget', 'qt'], input_requires=[], output_provides=[], uses=['build_batch_interactive_placefield_visibility_controls'], used_by=[], creation_date='2025-06-27 00:57', related_items=[])
def build_all_placefield_output_panels(ipcDataExplorer):
    """ Builds the row of custom SingleEditablePlacefieldDisplayConfiguration widgets for each placecell that allow configuring their display
    
    Called by _display_3d_interactive_tuning_curves_plotter
    
    TODO: can't get signals working unfortunately. https://stackoverflow.com/questions/45090982/passing-extra-arguments-through-connect
    https://eli.thegreenplace.net/2011/04/25/passing-extra-arguments-to-pyqt-slot
    
    
    Adds:        
        rootControlsBarWidget.ui.pf_widgets
        rootControlsBarWidget.ui.end_button_helper_obj
        rootControlsBarWidget.ui.end_button_helper_connections
        
        self.ui.btnNonPlacefieldSpikes
        self.ui.btnToggleOccupancy
        
    """ 
    ## UI Designer Version:    
    rootControlsBarWidget = PlacefieldVisualSelectionControlsBarWidget()
    # groupBox = rootControlsBarWidget.ui.placefieldControlsGroupbox
    pf_layout = rootControlsBarWidget.ui.pf_layout
        
    ## Nested Callback Functions:
    def _on_neuron_color_display_config_changed(new_config):
        """ The function called when the neuron color is changed.
        Implicitly captures ipcDataExplorer
        
        Recieves a SingleNeuronPlottingExtended config
        
        Usage:
            for a_widget in pf_widgets:
                # Connect the signals to the debugging slots:
                a_widget.spike_config_changed.connect(_on_spike_config_changed)
                a_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        """
        # print(f'_on_neuron_color_display_config_changed(new_config: {new_config})')
        
        # Need to rebuild the spikes colors and such upon updating the configs. 
        # should take a config and produce the changes needed to recolor the neurons.

        # test_updated_colors_map = {3: '#999999'}

        if isinstance(new_config, SingleNeuronPlottingExtended):
            # wrap it in a single-element list before passing:
            new_config = [new_config]

        extracted_neuron_id_updated_colors_map = {int(a_config.name):a_config.color for a_config in new_config}
        
        # ipcDataExplorer.
        # Apply the updated map using the update functions:
        ipcDataExplorer.on_config_update(extracted_neuron_id_updated_colors_map)
        # ipcDataExplorer.on_update_spikes_colors(extracted_neuron_id_updated_colors_map)
        # ipcDataExplorer.update_rendered_placefields(extracted_neuron_id_updated_colors_map)
        # print(f'\t _on_neuron_color_display_config_changed(...): done!')
                
    # @QtCore.pyqtSlot(list)
    def _on_tuning_curve_display_config_changed(new_configs):
        """ The function called when the non-color tuning curve display changed.
        Implicitly captures ipcDataExplorer
        
        Usage:
            for a_widget in pf_widgets:
                # Connect the signals to the debugging slots:
                a_widget.spike_config_changed.connect(_on_spike_config_changed)
                a_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        """
        # print(f'_on_tuning_curve_display_config_changed(new_configs: {new_configs})')
        # new_config: [SingleNeuronPlottingExtended(color='#843c39', extended_values_dictionary={}, isVisible=True, name='2', spikesVisible=False)]
        # recover cell_ids by parsing the name field:
        extracted_cell_ids = [int(a_config.name) for a_config in new_configs]
        # print(f'\t extracted_cell_ids: {extracted_cell_ids}')
        # convert to config indicies, which are what the configs are indexed by:
        # extracted_config_indicies = [ipcDataExplorer.params.reverse_cellID_to_tuning_curve_idx_lookup_map[a_cell_id] for a_cell_id in extracted_cell_ids]
        extracted_config_indicies = ipcDataExplorer.find_tuning_curve_IDXs_from_neuron_ids(extracted_cell_ids)
        # print(f'\t extracted_config_indicies: {extracted_config_indicies}')
        # The actual update function:
        ipcDataExplorer.on_update_tuning_curve_display_config(updated_configs=new_configs, updated_config_indicies=extracted_config_indicies) # could just update function to look at .name of each config? Or change indicies to map?
        # Is this required?
        ipcDataExplorer.apply_tuning_curve_configs() # works (seemingly)

    ## Build the Placefield Control Widgets:
    rootControlsBarWidget.ui.pf_widgets = []
    valid_neuron_ids = ipcDataExplorer.tuning_curves_valid_neuron_ids
    for (idx, neuron_id) in enumerate(valid_neuron_ids):
        a_config = ipcDataExplorer.active_tuning_curve_render_configs[idx]
        curr_widget = build_single_placefield_output_widget(a_config)
        ## Set the signals here:
        """ 
        obj.signal.connect(lambda param1, param2, ..., arg1=val1, arg2= value2, ... : fun(param1, param2,... , arg1, arg2, ....))
        
        def fun(param1, param2,... , arg1, arg2, ....):
            [...]
            
        where:
            param1, param2, ... : are the parameters sent by the signal
            arg1, arg2, ...: are the extra parameters that you want to spend

        """
        curr_widget.spike_config_changed.connect(lambda are_included, spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included, cell_id_copy=neuron_id: spikes_config_changed_callback(neuron_IDXs=None, cell_IDs=[cell_id_copy], are_included=are_included))
        
        # Connect the signals to the debugging slots:
        # curr_widget.spike_config_changed.connect(_on_spike_config_changed)
        curr_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        curr_widget.sig_neuron_color_changed.connect(_on_neuron_color_display_config_changed)
        
        pf_layout.addWidget(curr_widget)
        # pf_widgets.append(curr_widget)
        rootControlsBarWidget.ui.pf_widgets.append(curr_widget)
        
    # done adding widgets
    rootControlsBarWidget.rebuild_neuron_id_to_widget_map()
    
    ## Batch Action (Show/Hide All * Buttons):
    end_button_helper_obj, connections = build_batch_interactive_placefield_visibility_controls(rootControlsBarWidget=rootControlsBarWidget, ipcDataExplorer=ipcDataExplorer)
    rootControlsBarWidget.ui.end_button_helper_obj = end_button_helper_obj
    rootControlsBarWidget.ui.end_button_helper_connections = connections
    
        
    return (rootControlsBarWidget, rootControlsBarWidget.ui.pf_widgets)




