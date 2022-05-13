import param
import panel as pn
from panel.viewable import Viewer
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended

""" 

Primary Function: build_panel_interactive_placefield_visibility_controls(...)
    Internally calls build_all_placefield_output_panels(...) to build the output panel widgets.


"""

def build_single_placefield_output_panel(render_config):
    """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation """
    wgt_label_button = pn.widgets.Button(name=f'pf[{render_config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
    wgt_color_picker = pn.widgets.ColorPicker(value=render_config.color, width=60, height=20, margin=0)
    wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=render_config.isVisible, margin=0)
    wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=render_config.spikesVisible, margin=0)

    # gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=800)
    # Output Grid:
    gspec = pn.GridSpec(width=100, height=100, margin=0)
    gspec[0, :3] = wgt_label_button
    gspec[1, :] = wgt_color_picker
    gspec[2, :] = pn.Row(wgt_toggle_visible, margin=0, background='red')
    gspec[3, :] = pn.Row(wgt_toggle_spikes, margin=0, background='green')
    return gspec


class SingleEditablePlacefieldDisplayConfiguration(SingleNeuronPlottingExtended, Viewer):
    """ Panel configuration for a single placefield display (as in for a single cell)
    Usage:
        single_editable_pf_custom_widget = SingleEditablePlacefieldDisplayConfiguration(ipcDataExplorer.active_tuning_curve_render_configs[2])
        single_editable_pf_custom_widget
    """
    # config = SingleNeuronPlottingExtended()
    
    # value = param.Range(doc="A numeric range.")
    # width = param.Integer(default=300)
    
    def __init__(self, config=None, callbacks=None, **params):
        if config is not None:
            self.name = config.name
            self.color = config.color
            self.isVisible = config.isVisible
            self.spikesVisible = config.spikesVisible
            
        if callbacks is not None:
            assert isinstance(callbacks, dict), "callbacks argument should be a dictionary with keys 'pf' and 'spikes'!"
            self._callbacks = callbacks
        else:
            self._callbacks = None
            raise ValueError
        
        # self._start_input = pn.widgets.FloatInput()
        # self._end_input = pn.widgets.FloatInput(align='end')
        self._wgt_label_button = pn.widgets.Button(name=self.name, button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
        self._wgt_color_picker = pn.widgets.ColorPicker(value=self.color, width=60, height=20, margin=0)
        self._wgt_toggle_visible = pn.widgets.Toggle(name='pf', value=self.isVisible, margin=0)
        self._wgt_toggle_spikes = pn.widgets.Toggle(name='Spikes', value=self.spikesVisible, margin=0)
        super().__init__(**params)
        # Output Grid:
        self._layout = pn.GridSpec(width=60, height=100, margin=0)
        self._layout[0, :3] = self._wgt_label_button
        self._layout[1, :] = self._wgt_color_picker
        self._layout[2, :] = pn.Row(self._wgt_toggle_visible, margin=0, background='red')
        self._layout[3, :] = pn.Row(self._wgt_toggle_spikes, margin=0, background='green')
        if config is not None:
            self.update_from_config(config)
        
        self._sync_widgets()
    
    def __panel__(self):
        return self._layout
    
    # @param.depends('config', watch=True)
    # @param.depends('config.name','config.color','config.isVisible','config.spikesVisible', watch=True)
    # @param.depends('config', watch=True)
    @param.depends('name','color','isVisible','spikesVisible', watch=True)
    def _sync_widgets(self):
        self._wgt_label_button.name = self.name
        self._wgt_color_picker.value = self.color
        self._wgt_toggle_visible.value = self.isVisible
        self._wgt_toggle_spikes.value = self.spikesVisible
        
    @param.depends('_wgt_label_button.name', '_wgt_color_picker.value', '_wgt_toggle_visible.value', '_wgt_toggle_spikes.value', watch=True)
    def _sync_params(self):
        self.name = self._wgt_label_button.name
        self.color = self._wgt_color_picker.value
        self.isVisible = self._wgt_toggle_visible.value
        self.spikesVisible = self._wgt_toggle_spikes.value
        
    @param.depends('_wgt_toggle_visible.value', watch=True)
    def _on_toggle_plot_visible_changed(self):
        print('_on_toggle_plot_visible_changed(...)')
        if self._callbacks is not None:
            self._callbacks['pf'](self.config_from_state()) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for pf value changes!')
            
    
    @param.depends('_wgt_toggle_spikes.value', watch=True)
    def _on_toggle_spikes_visible_changed(self):
        print('_on_toggle_spikes_visible_changed(...)')
        if self._callbacks is not None:
            updated_config = self.spikesVisible
            self._callbacks['spikes'](bool(self.spikesVisible)) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for spikes value changes!')
            
            
    def update_from_config(self, config):
        self.name = config.name
        self.color = config.color
        self.isVisible = config.isVisible
        self.spikesVisible = config.spikesVisible

    
    def config_from_state(self):
        return SingleNeuronPlottingExtended(name=self.name, isVisible=self.isVisible, color=self.color, spikesVisible=self.spikesVisible)

    @classmethod
    def build_all_placefield_output_panels(cls, configs, tuning_curve_config_changed_callback, spikes_config_changed_callback):
        """[summary]

        Args:
            configs ([type]): as would be obtained from ipcDataExplorer.active_tuning_curve_render_configs

        Returns:
            [type]: [description]
            
        Usage:
            out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs)
            pn.Row(*out_panels, height=120)        
        """
        
        # @param.depends(c.param.country, d.param.i, watch=True)
        # def g(country, i):
        #     print(f"g country={country} i={i}")

        out_panels = [SingleEditablePlacefieldDisplayConfiguration(config=a_config,
                                                                   callbacks={
                                                                        'pf': (lambda updated_config_copy=a_config, i_copy=idx: tuning_curve_config_changed_callback([i_copy], [updated_config_copy])),
                                                                        'spikes': (lambda are_included, i_copy=idx: spikes_config_changed_callback(neuron_IDXs=[i_copy], cell_IDs=None, are_included=are_included))
                                                                    }) for (idx, a_config) in enumerate(configs)]
        return out_panels
        


def build_all_placefield_output_panels(ipcDataExplorer):
    """ Builds the row of custom SingleEditablePlacefieldDisplayConfiguration widgets for each placecell that allow configuring their display """
    out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs,
                                                                                                 tuning_curve_config_changed_callback=ipcDataExplorer.on_update_tuning_curve_display_config,
                                                                                                 spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included)
    out_panels = pn.Row(*out_panels, height=120)
    return out_panels



class PlacefieldBatchActionsEndButtonPanel(object):
    """ A column of buttons that sits at the end of the panel_interactive_placefield_visibility_controls.
        Enables performing batch actions on the placefields, such as hiding all pfs/spikes, etc.
    """
    debug_logging = False
    
    def __init__(self, pf_option_indicies=None, pf_option_selected_values=None, num_pfs=None, update_included_cell_Indicies_callback=None, hide_all_callback=None, show_all_callback=None, **params):
        super(PlacefieldBatchActionsEndButtonPanel, self).__init__(**params)
        self.final_update_included_cell_Indicies_callback = None
        if update_included_cell_Indicies_callback is not None:
            if callable(update_included_cell_Indicies_callback):
                self.final_update_included_cell_Indicies_callback = update_included_cell_Indicies_callback

        self.hide_all_callback = hide_all_callback
        self.show_all_callback = show_all_callback
        # assert (self.final_update_included_cell_Indicies_callback is not None), "An update_included_cell_Indicies_callback(x) callback is needed."

    def btn_hide_all_callback(self, event):
        if self.debug_logging:
            print('EndButtonPanel.btn_hide_all_callback(...)')
        if self.hide_all_callback is not None:
            if callable(self.hide_all_callback):
                self.hide_all_callback(event)
    
    def btn_show_all_callback(self, event):
        if self.debug_logging:
            print('EndButtonPanel.btn_show_all_callback(...)')
        if self.show_all_callback is not None:
            if callable(self.show_all_callback):
                self.show_all_callback(event)
                
                

    def btn_update_active_placefields(self, event):
        if self.debug_logging:
            print('EndButtonPanel.btn_update_active_placefields(...)')
        # updated_pf_options_list_ints = ActivePlacefieldsPlottingPanel.options_to_int(self.cross_selector.value) # convert to ints
        # self.on_update_active_placefields(updated_pf_options_list_ints)
        
    def panel(self):
        # Panel pane and widget objects:
        # Action Buttons:
        self.button_hide_all = pn.widgets.Button(name='Hide All', width_policy='min')
        self.button_hide_all.on_click(self.btn_hide_all_callback)
        self.button_show_all = pn.widgets.Button(name='Show All', width_policy='min')
        self.button_show_all.on_click(self.btn_show_all_callback)
        self.button_update = pn.widgets.Button(name='Refresh', button_type='primary', width_policy='min')
        self.button_update.on_click(self.btn_update_active_placefields)
        return pn.Column(self.button_hide_all, self.button_show_all, self.button_update, margin=0, width_policy='min', width=70)

    
def build_panel_interactive_placefield_visibility_controls(ipcDataExplorer, debug_logging=False):
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


