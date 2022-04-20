import param
import numpy as np
import pandas as pd

from pyqtgraph.Qt import QtCore

from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import NeuronConfigOwningMixin, OptionsListMixin
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin



class RenderItemsConfiguration(QtCore.QObject):
    """docstring for RenderItemsConfiguration."""
    def __init__(self, arg, **kwargs):
        QtCore.QObject.__init__(self, **kwargs)
        
        # super(RenderItemsConfiguration, self).__init__()
    
    # SignalProxy
    
    

class PlacefieldOwningMixin(NeuronIdentityAccessingMixin, NeuronConfigOwningMixin):
    """ Implementor owns placefields and has access to their data and configuration objects
    
    TODO: remember that Placefields should be a subset of the neuron identities.
    
    Requires:
        self.params.active_epoch_placefields
        self.placefields.ratemap
        self.ratemap.normalized_tuning_curves
    
    """
    debug_logging = False
    
    @property
    def placefields(self):
        return self.params.active_epoch_placefields
    
    @property
    def ratemap(self):
        return self.placefields.ratemap
    
    @property
    def tuning_curves(self):
        return self.ratemap.normalized_tuning_curves
    
    @property
    def num_tuning_curves(self):
        return len(self.tuning_curves)
    
    @property
    def tuning_curve_indicies(self):
        return np.arange(self.num_tuning_curves)
    
    @property
    def active_tuning_curve_render_configs(self):
        """The active_tuning_curve_render_configs property."""
        return self.active_neuron_render_configs
    @active_tuning_curve_render_configs.setter
    def active_tuning_curve_render_configs(self, value):
        self.active_neuron_render_configs = value
        
    def build_tuning_curve_configs(self):
        # call the parent function
        self.build_neuron_render_configs()
        # do any addition setup needed
            
    # @property
    # def active_tuning_curve_render_configs(self):
    #     """The active_tuning_curve_render_configs property."""
    #     return self.params.pf_active_configs
    # @active_tuning_curve_render_configs.setter
    # def active_tuning_curve_render_configs(self, value):
    #     self.params.pf_active_configs = value
        
    # def build_tuning_curve_configs(self):
    #     # Get the cell IDs that have a good place field mapping:
    #     good_placefield_neuronIDs = np.array(self.ratemap.neuron_ids) # in order of ascending ID
    #     unit_labels = [f'{good_placefield_neuronIDs[i]}' for i in np.arange(self.num_tuning_curves)]
    #     self.active_tuning_curve_render_configs = [SingleNeuronPlottingExtended(name=unit_labels[i], isVisible=False, color=self.params.pf_colors_hex[i], spikesVisible=False) for i in self.tuning_curve_indicies]



    
class HideShowPlacefieldsRenderingMixin(PlacefieldOwningMixin):
    """ Implementor Visually Displays Placefield data and enables basic interactivity for it.
    
    
    Requirements:
        self.plots['tuningCurvePlotActors']
    
        From PlacefieldOwningMixin:
            self.active_tuning_curve_render_configs
    
    """
    debug_logging = False
        
    @property
    def tuning_curve_plot_actors(self):
        return self.plots['tuningCurvePlotActors']
    
    @property
    def num_tuning_curve_plot_actors(self):
        return len(self.tuning_curve_plot_actors)
    
    @property
    def tuning_curve_is_visible(self):
        return np.array([bool(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=bool)
        
    @property
    def tuning_curve_visibilities(self):
        return np.array([int(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=int)

    @property
    def visible_tuning_curve_indicies(self):
        all_indicies = np.arange(self.num_tuning_curve_plot_actors)
        return all_indicies[self.tuning_curve_is_visible]
    
    def update_active_placefields(self, placefield_indicies):
        """ 
        Usage: 
            included_cell_ids = [48, 61]
            included_cell_INDEXES = [ipcDataExplorer.get_neuron_id_and_idx(cell_id=an_included_cell_ID)[0] for an_included_cell_ID in included_cell_ids] # get the indexes from the cellIDs
            ipcDataExplorer.update_active_placefields(included_cell_INDEXES) # actives only the placefields that have aclu values (cell ids) in the included_cell_ids array.
        """
        self._hide_all_tuning_curves() # hide all tuning curves to begin with (for a fresh slate)
        for a_pf_idx in placefield_indicies:
            self._show_tuning_curve(a_pf_idx)
        
    def _hide_all_tuning_curves(self):
        # Works to hide all turning curve plots:
        for aTuningCurveActor in self.tuning_curve_plot_actors:
            aTuningCurveActor.SetVisibility(0)

    def _show_all_tuning_curves(self):
        # Works to show all turning curve plots by updating the render configs:
        print('WARNING: _show_all_tuning_curves() does not currently work.')
        tuning_curve_config_indicies = np.arange(self.num_tuning_curves)
        # update the configs:
        curr_configs = self.active_tuning_curve_render_configs       
        for config_idx in tuning_curve_config_indicies:
            curr_configs[config_idx].isVisible = True
        # print(f'curr_configs: {curr_configs}')
        self.on_update_tuning_curve_display_config(tuning_curve_config_indicies, curr_configs)

            
    def _show_tuning_curve(self, show_index):
        # Works to show the specified tuning curve plots:
        self.tuning_curve_plot_actors[show_index].SetVisibility(1)
        
    def on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs):
        # TODO: NON-EXPLICIT INDEXING
        if self.debug_logging:
            print(f'HideShowPlacefieldsRenderingMixin.on_update_tuning_curve_display_config(updated_config_indicies: {updated_config_indicies}, updated_configs: {updated_configs})')
        assert hasattr(self, 'update_neuron_render_configs'), "self must be of type NeuronConfigOwningMixin to have access to its configs"
        self.update_neuron_render_configs(updated_config_indicies, updated_configs) # update the config with the new values:
        for an_updated_config_idx, an_updated_config in zip(updated_config_indicies, updated_configs):
            self.tuning_curve_plot_actors[an_updated_config_idx].SetVisibility(int(self.active_tuning_curve_render_configs[an_updated_config_idx].isVisible)) # update visibility of actor
            
    
    ## Change these names, update_* can easily be called and it does the opposite of what we'd expect
    def update_tuning_curve_configs(self):
        """ update the configs from the actual actors' state """
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors):
            self.active_tuning_curve_render_configs[i].isVisible = bool(aTuningCurveActor.GetVisibility())
            
    def apply_tuning_curve_configs(self):
        """ update the actual actors from the configs """
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors):
            aTuningCurveActor.SetVisibility(int(self.active_tuning_curve_render_configs[i].isVisible))



