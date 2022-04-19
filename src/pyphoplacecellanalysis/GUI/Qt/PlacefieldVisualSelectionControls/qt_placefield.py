# TODO: Implement the functionality present in panel_placefield.py for the new PlacefieldVisualSelectionWidget Qt widget.

from functools import partial
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.PlacefieldVisualSelectionControlWidget import PlacefieldVisualSelectionWidget


def build_single_placefield_output_widget(render_config):
    """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation """
    # wgt_label_button = pn.widgets.Button(name=f'pf[{render_config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
    # wgt_color_picker = pn.widgets.ColorPicker(value=render_config.color, width=60, height=20, margin=0)
    # wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=render_config.isVisible, margin=0)
    # wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=render_config.spikesVisible, margin=0)    
    curr_pf_string = f'pf[{render_config.name}]'
    curr_widget = PlacefieldVisualSelectionWidget()
    curr_widget.setObjectName(curr_pf_string)
    curr_widget.name = curr_pf_string # be sure to set the name
    # set the color and such too
    curr_widget.color = render_config.color
    curr_widget.isVisible = render_config.isVisible
    curr_widget.spikesVisible = render_config.spikesVisible
    return curr_widget


# 	PlacefieldVisualSelectionWidget(
#     # gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=800)
#     # Output Grid:
#     gspec = pn.GridSpec(width=100, height=100, margin=0)
#     gspec[0, :3] = wgt_label_button
#     gspec[1, :] = wgt_color_picker
#     gspec[2, :] = pn.Row(wgt_toggle_visible, margin=0, background='red')
#     gspec[3, :] = pn.Row(wgt_toggle_spikes, margin=0, background='green')
#     return gspec


def build_all_placefield_output_panels(ipcDataExplorer):
    """ Builds the row of custom SingleEditablePlacefieldDisplayConfiguration widgets for each placecell that allow configuring their display
    
    TODO: can't get signals working unfortunately. https://stackoverflow.com/questions/45090982/passing-extra-arguments-through-connect
    https://eli.thegreenplace.net/2011/04/25/passing-extra-arguments-to-pyqt-slot
    
    """
    # out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs,
    #                                                                                              tuning_curve_config_changed_callback=ipcDataExplorer.on_update_tuning_curve_display_config,
    #                                                                                              spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included)
    # out_panels = pn.Row(*out_panels, height=120)

    desired_full_panel_width = 1200
    desired_full_panel_height = 200

    placefieldControlsContainerWidget = QtWidgets.QWidget()
    placefieldControlsContainerWidget.setObjectName('placefieldControlsContainer')
    placefieldControlsContainerWidget.resize(desired_full_panel_width, desired_full_panel_height)
    placefieldControlsContainerWidget.setContentsMargins(0, 0, 0, 0)
    
    groupBox = QtWidgets.QGroupBox("Placefield Controls")
    groupBox.setContentsMargins(0, 0, 0, 0)
    groupBox.setObjectName('placefieldControlsGroupbox')
    groupBox.resize(desired_full_panel_width, desired_full_panel_height)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    # sizePolicy.setHeightForWidth(Interface.sizePolicy().hasHeightForWidth())
    groupBox.setSizePolicy(sizePolicy)
    groupBox.setWindowTitle('Placefield Controls')
    
    pf_layout = QtWidgets.QHBoxLayout()
    pf_layout.setSpacing(0)
    pf_layout.setContentsMargins(0, 0, 0, 0)
    pf_layout.setObjectName("horizontalLayout")
    
    # placefieldControlsContainerWidget.setLayout(pf_layout)
    
    pf_widgets = []
    # the active_tuning_curve_render_configs are an array of SingleNeuronPlottingExtended objects, one for each placefield
    for (idx, a_config) in enumerate(ipcDataExplorer.active_tuning_curve_render_configs):
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
        # curr_widget.spike_config_changed.connect(ipcDataExplorer.change_unit_spikes_included)
        # curr_widget.tuning_curve_display_config_changed.connect(ipcDataExplorer.on_update_tuning_curve_display_config)
        
        # current signals:
        # spike_config_changed = QtCore.pyqtSignal(bool) # change_unit_spikes_included(self, cell_IDXs=None, cell_IDs=None, are_included=True)
        # tuning_curve_display_config_changed = QtCore.pyqtSignal(list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
        # curr_widget.spike_config_changed.connect(lambda are_included_list, cell_IDXs=val1, arg2= value2, ... : ipcDataExplorer.change_unit_spikes_included(param1, param2,... , arg1, arg2, ....) )
        # curr_widget.tuning_curve_display_config_changed.connect(lambda are_included, i_copy=idx: spikes_config_changed_callback(cell_IDXs=[i_copy], cell_IDs=None, are_included=are_included)
        
        curr_widget.spike_config_changed.connect(lambda are_included, spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included, i_copy=idx: spikes_config_changed_callback(cell_IDXs=[i_copy], cell_IDs=None, are_included=are_included))
        curr_widget.tuning_curve_display_config_changed.connect(lambda updated_config_copy=a_config, i_copy=idx, tuning_curve_config_changed_callback=ipcDataExplorer.on_update_tuning_curve_display_config: tuning_curve_config_changed_callback([i_copy], [updated_config_copy]))
                                 
        # partial(self.on_button, 1), i_copy=idx
        # cell_IDXs=None, cell_IDs=None
        # ipcDataExplorer.on_update_tuning_curve_display_config
        # ipcDataExplorer.change_unit_spikes_included
        pf_layout.addWidget(curr_widget)
        pf_widgets.append(curr_widget)
        
    # done adding widgets
    ## Simple (no groupbox or scroll area):
    # placefieldControlsContainerWidget.setLayout(pf_layout)

    ## Groupbox and Scrollarea:
    # groupBox.setLayout(pf_layout) # set the groupBox's layout to the one containing the widgets
    placefieldControlsContainerWidget.setLayout(pf_layout)

    # Add a horizontal scroll area (so the placefield controls can be scrolled horizontally:
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.resize(desired_full_panel_width, 150)
    scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll_area.setSizeAdjustPolicy(QtGui.QAbstractScrollArea.AdjustToContentsOnFirstShow)
    scroll_area.setWidget(placefieldControlsContainerWidget) # set the contents widget of the scrollarea to be the groupBox
    # scroll_area.setWidget(groupBox) # set the contents widget of the scrollarea to be the groupBox
    scroll_area.setWidgetResizable(True)
    # scroll_area.setWidgetResizable(False) # This really breaks it for some reason. Oh, I guess because it's dynamically trying to resize the widget instead of creating more room.
    scroll_area.setFixedHeight(150)
    
    outer_scroll_layout = QtWidgets.QVBoxLayout()
    outer_scroll_layout.setSpacing(0)
    outer_scroll_layout.setContentsMargins(0, 0, 0, 0)
    outer_scroll_layout.setObjectName("outerLayout")
    outer_scroll_layout.addWidget(scroll_area)
    # Set the root widget's layout to the outer_scroll_layout
    # placefieldControlsContainerWidget.setLayout(outer_scroll_layout)
    groupBox.setLayout(outer_scroll_layout)
    
    return (groupBox, pf_widgets)
    # return (placefieldControlsContainerWidget, pf_widgets)




def build_qt_interactive_placefield_visibility_controls(ipcDataExplorer, debug_logging=False):
    """Builds a panel containing a series of widgets that control the spike/placemap/etc visibility for each placecell

    Args:
        ipcDataExplorer ([type]): [description]

    Returns:
        [type]: [description]
        
    Usage:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        pane
    """
    def _btn_hide_all_callback(event):
        if debug_logging:
            print('EndButtonPanel.btn_hide_all_callback(...)')
        ipcDataExplorer.clear_all_spikes_included()
        ipcDataExplorer.update_active_placefields([])
        # self.on_hide_all_placefields()
  
    def _btn_show_all_callback(event):
        if debug_logging:
            print('EndButtonPanel.btn_show_all_callback(...)')
        ipcDataExplorer._show_all_tuning_curves()
        ipcDataExplorer.update_active_placefields([])
        # self.on_hide_all_placefields()      
        
    out_panels = build_all_placefield_output_panels(ipcDataExplorer)
    end_button_panel_obj = PlacefieldBatchActionsEndButtonPanel(hide_all_callback=_btn_hide_all_callback, show_all_callback=_btn_show_all_callback)
    end_cap_buttons = end_button_panel_obj.panel()
    out_row = pn.Row(*out_panels, end_cap_buttons, height=120)
    # btn_occupancy_map_visibility = pn.widgets.Button(name='Occupancy Map Visibility', width_policy='min')
    # # btn_occupancy_map_visibility = pn.widgets.Toggle(name='Occupancy Map Visibility', value=ipcDataExplorer.occupancy_plotting_config.isVisible, margin=0, width_policy='min')
    # # btn_occupancy_map_visibility.on_clicks
    # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_update_visibility)
    # # btn_occupancy_map_visibility.on_click(ipcDataExplorer.on_occupancy_plot_config_updated)
    # occupancy_widget = btn_occupancy_map_visibility
    
    occupancy_widget = ipcDataExplorer.occupancy_plotting_config.param
    return pn.panel(pn.Column(out_row, pn.Row(occupancy_widget)))



