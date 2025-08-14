import numpy as np
import param
from qtpy import QtGui

from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import ExtendedPlotDataParams


class NeuronConfigOwningMixin:
    """ Implementors own a series of visual configurations for each neuron.

    Requirements:
        self.params.pf_active_configs
        self.params.pf_colors_hex

        self.ratemap.neuron_ids


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
    def num_neuron_configs(self) -> int:
        return len(self.active_neuron_render_configs)

    @property
    def neuron_config_indicies(self):
        return np.arange(self.num_neuron_configs)


    ### Original Set of Functions:

    # , neuron_IDXs=None, cell_IDs=None
    def update_neuron_render_configs_from_indicies(self, updated_config_indicies, updated_configs, defer_render=False):
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


    def update_neuron_render_configs(self, updated_configs, defer_render=False):
        """
            Actually performs updating the self.active_neuron_render_configs_map with the updated values provided in updated_configs

        Inputs:
            updated_configs: dict<neuron_id (int), config>
        Requires:
            self.active_neuron_render_configs_map
        Returns:
            Returns a list of the values that actually changed.
            updated_ids_list, updated_configs_list
        """
        # self.active_neuron_render_configs_map[updated_configs
        updated_ids_list = []
        updated_configs_list = []

        for neuron_id, updated_config in updated_configs.items():
            # didValueChange = (self.active_neuron_render_configs_map[neuron_id].color != updated_config)
            didValueChange = (self.active_neuron_render_configs_map[neuron_id] != updated_config)
            if didValueChange:
                self.active_neuron_render_configs_map[neuron_id] = updated_config # get the config from the self.active_neuron_render_configs_map and set its color value
                # add to list that tracks which items changed:
                updated_ids_list.append(neuron_id)
                updated_configs_list.append(self.active_neuron_render_configs_map[neuron_id])

        return updated_ids_list, updated_configs_list




    def build_neuron_render_configs(self):
        """ Builds the render config models that are used to control the displayed settings for each cell
        Requires:
            self.params.pf_colors_hex: this should have one entry per num_neurons, which is the length of self.ratemap.neuron_ids

            self.ratemap.neuron_ids: for some reason it also requires self.ratemap.neuron_ids, which it ultimately shouldn't if it's to be general.

        Sets:
            self.active_neuron_render_configs: a list of configs
            self.active_neuron_render_configs_map: a Dict<neuron_id (int), config> mapping
        """
        ## TODO: should have code here that ensures this is only done once, so values don't get overwritten
        # Get the cell IDs that have a good place field mapping:
        good_placefield_neuronIDs = np.array(self.ratemap.neuron_ids) # in order of ascending ID
        num_neurons = len(good_placefield_neuronIDs)
        unit_labels = [f'{good_placefield_neuronIDs[i]}' for i in np.arange(num_neurons)]
        self.active_neuron_render_configs = [SingleNeuronPlottingExtended(name=unit_labels[i], isVisible=False, color=self.params.pf_colors_hex[i], spikesVisible=False) for i in np.arange(num_neurons)]
        self.active_neuron_render_configs_map = NeuronConfigOwningMixin._build_id_index_configs_dict(self.active_neuron_render_configs)


        ## TODO: POTENTIAL ERROR: This only builds configs for good neurons, but we should be building them for all neurons right?


#         return combined_active_pf_update_callbacks


    ### Modern Dict-based method (with neuron_id keys):
    @classmethod
    def _build_id_index_configs_dict(cls, configs):
        """ Returns a dict of the configs passed in indexed by their name as the key """
        return {int(a_config.name):a_config for a_config in configs}

    @classmethod
    def build_updated_colors_map_from_configs(cls, configs):
        """ extracts a dictionary with keys of the neuron_ID (as an int) and values of the corresponding neuron's color as a hex string."""
        # return {a_neuron_id:color for a_neuron_id, color in configs.items()}
        return {int(a_config.name):a_config.color for a_config in configs}

    @classmethod
    def apply_updated_colors_map_to_configs(cls, configs, updated_colors_map):
        """ Updates the **configs** from the updated_colors_map

        Checks for which values are actually changing and returns the updated_ids_list, updated_configs_list

        Inputs:
            updated_colors_map: a dictionary with keys of neuron_id and values of the hex_color to use.

        Outputs:
            updated_ids_list, updated_configs_list: the neuron_ids and configs that actually changed as a result of this function.

        """
        if isinstance(configs, dict):
            # already a config map:
            configs_map = configs
        else:
            # make into a config map
            configs_map = cls._build_id_index_configs_dict(configs)

        updated_ids_list = []
        updated_configs_list = []

        for neuron_id, updated_value in updated_colors_map.items():
            didValueChange = (configs_map[neuron_id].color != updated_value)
            if didValueChange:
                configs_map[neuron_id].color = updated_value # get the config from the configs_map and set its color value
                # add to list that tracks which items changed:
                updated_ids_list.append(neuron_id)
                updated_configs_list.append(configs_map[neuron_id])

        return configs_map, updated_ids_list, updated_configs_list


class SingleNeuronPlottingExtended(ExtendedPlotDataParams):
    """ represents the visual config for a single neuron. """
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
