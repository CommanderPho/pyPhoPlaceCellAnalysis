# TODO: Implement the functionality present in panel_placefield.py for the new PlacefieldVisualSelectionWidget Qt widget.

from functools import partial
import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.PlacefieldVisualSelectionControlWidget import PlacefieldVisualSelectionWidget

from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControlsBar.PlacefieldVisualSelectionControlsBarWidget import PlacefieldVisualSelectionControlsBarWidget


""" 

This whole file serves to emulate the structure of panel_placefield.py, which does the exact same thing (renders interactive placefield toggle controls) but using the Panel library instead of Qt.

Primary Function: build_qt_interactive_placefield_visibility_controls(...)
     Internally calls build_all_placefield_output_panels(...) to build the output widgets.


"""


def build_single_placefield_output_widget(render_config):
    """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation 
    
    Called in build_all_placefield_output_panels(...) down below.
    
    """
    # wgt_label_button = pn.widgets.Button(name=f'pf[{render_config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
    # wgt_color_picker = pn.widgets.ColorPicker(value=render_config.color, width=60, height=20, margin=0)
    # wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=render_config.isVisible, margin=0)
    # wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=render_config.spikesVisible, margin=0)    
    curr_pf_string = f'pf[{render_config.name}]'
    curr_widget = PlacefieldVisualSelectionWidget() # new widget type
    curr_widget.setObjectName(curr_pf_string)
    curr_widget.name = curr_pf_string # be sure to set the name
    # set the color and such too
    curr_widget.color = render_config.color
    curr_widget.isVisible = render_config.isVisible
    curr_widget.spikesVisible = render_config.spikesVisible
    
    curr_widget.update_from_config(render_config) # is this the right type of config? I think it is.
    return curr_widget



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

def build_batch_interactive_placefield_visibility_controls(rootControlsBarWidget, ipcDataExplorer, debug_logging=False):
    """Builds a panel containing a series of widgets that control the spike/placemap/etc visibility for each placecell

        Internally calls build_all_placefield_output_panels(...) to build the output panel widgets.
        
        
    Args:
        ipcDataExplorer ([type]): [description]

    Returns:
        [type]: [description]
        
    Usage:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        pane
    """
    
    
    
    def _btn_hide_all_callback():
        if debug_logging:
            print('EndButtonPanel.btn_hide_all_callback(...)')
        ipcDataExplorer.clear_all_spikes_included()
        ipcDataExplorer.update_active_placefields([])
        # self.on_hide_all_placefields()
  
    def _btn_show_all_callback():
        if debug_logging:
            print('EndButtonPanel.btn_show_all_callback(...)')
        ipcDataExplorer._show_all_tuning_curves()
        ipcDataExplorer.update_active_placefields([])
        # self.on_hide_all_placefields()      

    end_button_helper_obj = BatchActionsEndButtonPanelHelper(hide_all_callback=_btn_hide_all_callback, show_all_callback=_btn_show_all_callback)
    

    btnShowAll_Pfs, btnShowAll_Spikes, btnShowAll_Both = rootControlsBarWidget.show_all_buttons_list
    btnHideAll_Pfs, btnHideAll_Spikes, btnHideAll_Both = rootControlsBarWidget.hide_all_buttons_list 
    
    connections = []
    buttons = [btnShowAll_Pfs, btnShowAll_Spikes, btnShowAll_Both, btnHideAll_Pfs, btnHideAll_Spikes, btnHideAll_Both]
    
    connections.append(btnShowAll_Both.clicked.connect(end_button_helper_obj.btn_show_all_callback))
    connections.append(btnHideAll_Both.clicked.connect(end_button_helper_obj.btn_hide_all_callback))
    
    # out_panels = build_all_placefield_output_panels(ipcDataExplorer)
    # end_button_panel_obj = BatchActionsEndButtonPanelHelper(hide_all_callback=_btn_hide_all_callback, show_all_callback=_btn_show_all_callback)
    # end_cap_buttons = end_button_panel_obj.panel()
    # out_row = pn.Row(*out_panels, end_cap_buttons, height=120)
    # btn_occupancy_map_visibility = pn.widgets.Button(name='Occupancy Map Visibility', width_policy='min')
    # # btn_occupancy_map_visibility = pn.widgets.Toggle(name='Occupancy Map Visibility', value=ipcDataExplorer.occupancy_plotting_config.isVisible, margin=0, width_policy='min')
    # # btn_occupancy_map_visibility.on_clicks
    # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_update_visibility)
    # # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_config_updated)
    # occupancy_widget = btn_occupancy_map_visibility
    
    return end_button_helper_obj, connections



def build_all_placefield_output_panels(ipcDataExplorer):
    """ Builds the row of custom SingleEditablePlacefieldDisplayConfiguration widgets for each placecell that allow configuring their display
    
    TODO: can't get signals working unfortunately. https://stackoverflow.com/questions/45090982/passing-extra-arguments-through-connect
    https://eli.thegreenplace.net/2011/04/25/passing-extra-arguments-to-pyqt-slot
    
    """ 
    ## UI Designer Version:    
    rootControlsBarWidget = PlacefieldVisualSelectionControlsBarWidget()
    # groupBox = rootControlsBarWidget.ui.placefieldControlsGroupbox
    pf_layout = rootControlsBarWidget.ui.pf_layout
    
    # @QtCore.pyqtSlot(list)
    def _on_tuning_curve_display_config_changed(new_configs):
        """
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
        extracted_config_indicies = ipcDataExplorer.find_tuning_curve_IDXs_from_cell_ids(extracted_cell_ids)
        # print(f'\t extracted_config_indicies: {extracted_config_indicies}')
        # The actual update function:
        ipcDataExplorer.on_update_tuning_curve_display_config(updated_configs=new_configs, updated_config_indicies=extracted_config_indicies) # could just update function to look at .name of each config? Or change indicies to map?
        # Is this required?
        ipcDataExplorer.apply_tuning_curve_configs() # works (seemingly)


    ## Build the Placefield Control Widgets:
    pf_widgets = []
    # the active_tuning_curve_render_configs are an array of SingleNeuronPlottingExtended objects, one for each placefield
    # for (idx, a_config) in enumerate(ipcDataExplorer.active_tuning_curve_render_configs):
    
    valid_cell_ids = ipcDataExplorer.tuning_curves_valid_cell_ids
    for (idx, cell_id) in enumerate(valid_cell_ids):
        a_config = ipcDataExplorer.active_tuning_curve_render_configs[idx]
        curr_widget = build_single_placefield_output_widget(a_config)
        # TODO: Set the signals here:
        """ 
        obj.signal.connect(lambda param1, param2, ..., arg1=val1, arg2= value2, ... : fun(param1, param2,... , arg1, arg2, ....))
        
        def fun(param1, param2,... , arg1, arg2, ....):
            [...]
            
        where:
            param1, param2, ... : are the parameters sent by the signal
            arg1, arg2, ...: are the extra parameters that you want to spend

        """
        curr_widget.spike_config_changed.connect(lambda are_included, spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included, cell_id_copy=cell_id: spikes_config_changed_callback(cell_IDXs=None, cell_IDs=[cell_id_copy], are_included=are_included))
        
        # curr_widget.spike_config_changed.connect(lambda are_included, spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included, i_copy=idx: spikes_config_changed_callback(cell_IDXs=[i_copy], cell_IDs=None, are_included=are_included))
        
        # Connect the signals to the debugging slots:
        # curr_widget.spike_config_changed.connect(_on_spike_config_changed)
        curr_widget.tuning_curve_display_config_changed.connect(_on_tuning_curve_display_config_changed)
        
        pf_layout.addWidget(curr_widget)
        pf_widgets.append(curr_widget)
        
    # done adding widgets
    
    ## Batch Action (Show/Hide All * Buttons):
    end_button_helper_obj, connections = build_batch_interactive_placefield_visibility_controls(rootControlsBarWidget=rootControlsBarWidget, ipcDataExplorer=ipcDataExplorer)
    ipcDataExplorer.params.end_button_helper_obj = end_button_helper_obj
    ipcDataExplorer.params.end_button_helper_connections = connections
    
    return (rootControlsBarWidget, pf_widgets)
    # return (placefieldControlsContainerWidget, pf_widgets)




# def build_qt_interactive_placefield_visibility_controls(ipcDataExplorer, debug_logging=False):
#     """Builds a panel containing a series of widgets that control the spike/placemap/etc visibility for each placecell

#     Args:
#         ipcDataExplorer ([type]): [description]

#     Returns:
#         [type]: [description]
        
#     Usage:
#         pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
#         pane
#     """
#     def _btn_hide_all_callback(event):
#         if debug_logging:
#             print('EndButtonPanel.btn_hide_all_callback(...)')
#         ipcDataExplorer.clear_all_spikes_included()
#         ipcDataExplorer.update_active_placefields([])
#         # self.on_hide_all_placefields()
  
#     def _btn_show_all_callback(event):
#         if debug_logging:
#             print('EndButtonPanel.btn_show_all_callback(...)')
#         ipcDataExplorer._show_all_tuning_curves()
#         ipcDataExplorer.update_active_placefields([])
#         # self.on_hide_all_placefields()      
        
#     out_panels = build_all_placefield_output_panels(ipcDataExplorer)
#     end_button_panel_obj = PlacefieldBatchActionsEndButtonPanel(hide_all_callback=_btn_hide_all_callback, show_all_callback=_btn_show_all_callback)
#     end_cap_buttons = end_button_panel_obj.panel()
#     out_row = pn.Row(*out_panels, end_cap_buttons, height=120)
#     # btn_occupancy_map_visibility = pn.widgets.Button(name='Occupancy Map Visibility', width_policy='min')
#     # # btn_occupancy_map_visibility = pn.widgets.Toggle(name='Occupancy Map Visibility', value=ipcDataExplorer.occupancy_plotting_config.isVisible, margin=0, width_policy='min')
#     # # btn_occupancy_map_visibility.on_clicks
#     # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_update_visibility)
#     # # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_config_updated)
#     # occupancy_widget = btn_occupancy_map_visibility
    
#     occupancy_widget = ipcDataExplorer.occupancy_plotting_config.param
#     return pn.panel(pn.Column(out_row, pn.Row(occupancy_widget)))



