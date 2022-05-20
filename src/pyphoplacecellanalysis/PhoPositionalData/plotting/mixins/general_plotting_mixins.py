import param
import numpy as np
import pandas as pd
from qtpy import QtGui # for QColor


class OptionsListMixin:
    @staticmethod
    def options_to_str(options_list_ints):
        return [f'{i}' for i in options_list_ints]
    @staticmethod
    def options_to_int(options_list_strings):
        return [int(a_str) for a_str in options_list_strings]
    @staticmethod
    def build_pf_options_list(num_pfs=40):
        pf_options_list_ints = np.arange(num_pfs)
        pf_options_list_strings = OptionsListMixin.options_to_str(pf_options_list_ints) # [f'{i}' for i in pf_options_list_ints]
        return pf_options_list_ints, pf_options_list_strings


    
    
class NeuronConfigOwningMixin:
    """ Implementors own a series of visual configurations for each neuron.
    
    Requirements:
        self.params.pf_active_configs
        self.params.pf_colors_hex
        
        Functions:
            self.update_spikes(): to apply the changes visually
        
    Provides:
        self.active_neuron_render_configs
    
        
    """
    debug_logging = False
    
    @property
    def active_neuron_render_configs(self):
        """The active_neuron_render_configs property."""
        return self.params.pf_active_configs
    @active_neuron_render_configs.setter
    def active_neuron_render_configs(self, value):
        self.params.pf_active_configs = value

    @property
    def num_neuron_configs(self):
        return len(self.active_neuron_render_configs)

    @property
    def neuron_config_indicies(self):
        return np.arange(self.num_neuron_configs)
    
        
    # , neuron_IDXs=None, cell_IDs=None
    def update_neuron_render_configs(self, updated_config_indicies, updated_configs, defer_render=False):
        # TODO: NON-EXPLICIT INDEXING
        """Updates the configs for the cells with the specified updated_config_indicies
        Args:
            updated_config_indicies ([type]): [description]
            updated_configs ([type]): [description]
        """
        if self.debug_logging:
            print(f'NeuronConfigOwningMixin.update_cell_configs(updated_config_indicies: {updated_config_indicies}, updated_configs: {updated_configs})')
            
        ## Improved tuning_curve_display_config_changed(...) style:
        # recover cell_ids by parsing the name field:
        extracted_cell_ids = [int(a_config.name) for a_config in updated_configs]
        extracted_config_indicies = self.find_tuning_curve_IDXs_from_neuron_ids(extracted_cell_ids)
         # Sets the configs:
        for an_updated_config_idx, an_updated_config in zip(extracted_config_indicies, updated_configs):
            self.active_neuron_render_configs[an_updated_config_idx] = an_updated_config # update the config with the new values:
            
        # # Sets the configs:
        # for an_updated_config_idx, an_updated_config in zip(updated_config_indicies, updated_configs):
        #     self.active_neuron_render_configs[an_updated_config_idx] = an_updated_config # update the config with the new values:
        
        ## Apply the changes visually:
        if not defer_render:
            self.update_spikes()
        
        
            
    def build_neuron_render_configs(self):
        ## TODO: should have code here that ensures this is only done once, so values don't get overwritten
        # Get the cell IDs that have a good place field mapping:
        good_placefield_neuronIDs = np.array(self.ratemap.neuron_ids) # in order of ascending ID
        num_neurons = len(good_placefield_neuronIDs)
        unit_labels = [f'{good_placefield_neuronIDs[i]}' for i in np.arange(num_neurons)]
        self.active_neuron_render_configs = [SingleNeuronPlottingExtended(name=unit_labels[i], isVisible=False, color=self.params.pf_colors_hex[i], spikesVisible=False) for i in np.arange(num_neurons)]
        
        ## TODO: POTENTIAL ERROR: This only builds configs for good neurons, but we should be building them for all neurons right?
           

# def __build_callbacks(self, tuningCurvePlotActors):
#         combined_active_pf_update_callbacks = []
#         for i, an_actor in enumerate(tuningCurvePlotActors):
#             # Make a separate callback for each widget
#             curr_visibility_callback = SetVisibilityCallback(an_actor)
#             curr_spikes_update_callback = (lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible))
#             combined_active_pf_update_callbacks.append(CallbackSequence([curr_visibility_callback, curr_spikes_update_callback]))
#         return combined_active_pf_update_callbacks
    
        


## Parameters (Param):
class BasePlotDataParams(param.Parameterized):
    # name = param.Parameter(default="Not editable", constant=True)
    name = param.String(default='name', doc='The name of the placefield')
    # name = param.Parameter(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=False, doc="Whether the plot is visible")


class ExtendedPlotDataParams(BasePlotDataParams):
    color = param.Color(default='#FF0000', doc="The placefield's Color")
    extended_values_dictionary = param.Dict(default={}, doc="Extra values stored in a dictionary.")


class ExampleExtended(BasePlotDataParams):
    color                   = param.Color(default='#BBBBBB')
    dictionary              = param.Dict(default={"a": 2, "b": 9})
    select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
    select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
    int_list                = param.ListSelector(default=[3, 5], objects=[1, 3, 5, 7, 9], precedence=0.5)

# checkbutton_group = pn.widgets.CheckButtonGroup(name='Check Button Group', value=[], options=pf_options_list_strings) # checkbutton_group.value 
# cross_selector = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=pf_options_list_strings) # cross_selector.value


class SingleNeuronPlottingExtended(ExtendedPlotDataParams):
    spikesVisible = param.Boolean(default=False, doc="Whether the spikes are visible")
    
    @property
    def neuron_id(self):
        """The neuron_id <int> property."""
        return int(self.name)
    

    @property
    def qcolor(self):
        """The qcolor property."""
        return QtGui.QColor(self.color)
    @qcolor.setter
    def qcolor(self, value):
        if isinstance(value, QtGui.QColor):
            self.color = value.name(QtGui.QColor.HexRgb) #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000' 
        else:
            print(f'ERROR: qcolor setter is being passed a value that is not a QtGui.QColor! Instead, it is of unknown type: {value}, type: {type(value)}')
            raise NotImplementedError
        
    
    # @param.depends(c.param.country, d.param.i, watch=True)
    # def g(country, i):
    #     print(f"g country={country} i={i}")
    
    
    # def panel(self):
    #     return pn.Row(
    #         pn.Column(
    #             pn.Param(SingleNeuronPlottingExtended.param, name="SinglePlacefield", widgets= {
    #                 'color': {'widget_type': pn.widgets.ColorPicker, 'name':'pf Color', 'value':'#99ef78', 'width': 50},
    #             })
    #         )
    #     )
           


