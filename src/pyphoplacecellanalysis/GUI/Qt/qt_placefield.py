# TODO: Implement the functionality present in panel_placefield.py for the new PlacefieldVisualSelectionWidget Qt widget.


from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionWidget import PlacefieldVisualSelectionWidget


# def build_single_placefield_output_panel(render_config):
#     """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation """
#     wgt_label_button = pn.widgets.Button(name=f'pf[{render_config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
#     wgt_color_picker = pn.widgets.ColorPicker(value=render_config.color, width=60, height=20, margin=0)
#     wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=render_config.isVisible, margin=0)
#     wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=render_config.spikesVisible, margin=0)


# 	PlacefieldVisualSelectionWidget(
#     # gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=800)
#     # Output Grid:
#     gspec = pn.GridSpec(width=100, height=100, margin=0)
#     gspec[0, :3] = wgt_label_button
#     gspec[1, :] = wgt_color_picker
#     gspec[2, :] = pn.Row(wgt_toggle_visible, margin=0, background='red')
#     gspec[3, :] = pn.Row(wgt_toggle_spikes, margin=0, background='green')
#     return gspec


# def build_all_placefield_output_panels(ipcDataExplorer):
#     """ Builds the row of custom SingleEditablePlacefieldDisplayConfiguration widgets for each placecell that allow configuring their display """
#     out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs,
#                                                                                                  tuning_curve_config_changed_callback=ipcDataExplorer.on_update_tuning_curve_display_config,
#                                                                                                  spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included)
#     out_panels = pn.Row(*out_panels, height=120)
#     return out_panels



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



