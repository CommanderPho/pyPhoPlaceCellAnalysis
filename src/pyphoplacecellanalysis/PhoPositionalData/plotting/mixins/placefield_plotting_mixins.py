import param
import numpy as np
import pandas as pd

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import NeuronConfigOwningMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_placefields2D, update_plotColorsPlacefield2D

class PlacefieldOwningMixin(NeuronIdentityAccessingMixin, NeuronConfigOwningMixin):
    """ Implementor owns placefields and has access to their data and configuration objects
    
    NOTE: remember that Placefields should be a subset of the neuron identities (that would be present in the filtered set of neurons that the ratemap/placefields are based on for example).
    
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
    def tuning_curves_neuron_extended_ids(self):
        """ the neuron_extended_id the corresponds to each placefield/ratemap """
        return self.ratemap.neuron_extended_ids
    @property
    def tuning_curves_valid_neuron_ids(self):
        """ the valid neuron_ids (a.k.a aclu values, neuron_ids, etc) corresponding to each tuning curve """
        return self.ratemap.neuron_ids

    
    @property
    def active_tuning_curve_render_configs(self):
        """The active_tuning_curve_render_configs property."""
        return self.active_neuron_render_configs
    @active_tuning_curve_render_configs.setter
    def active_tuning_curve_render_configs(self, value):
        self.active_neuron_render_configs = value
        
    def build_tuning_curve_configs(self):
        # call the parent function from NeuronConfigOwningMixin.build_neuron_render_configs():
        self.build_neuron_render_configs()
        # do any additional setup needed

    def find_tuning_curve_IDXs_from_neuron_ids(self, neuron_ids):
        """Finds the tuning curve IDXs from the cell original IDs (neuron_ids)
        Args:
            cell_ids ([type]): [description]
        """
        return np.array([self.params.reverse_cellID_to_tuning_curve_idx_lookup_map.get(a_cell_id, None) for a_cell_id in neuron_ids])
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
class PlacefieldRenderingPyVistaMixin:
    """ Implementors render placefields with PyVista 
    
    Requires:
        self.params
        
    Provides:
    
        Adds:
            self.params.unit_labels
            self.params.pf_fragile_linear_neuron_IDXs
            ... More?
            
            
    Known Uses:
        InteractivePlaceCellTuningCurvesDataExplorer
    """
    def plot_placefields(self):
        """This is the main plot function to render the placefields        
        """
        self.params.should_override_disable_smooth_shading = True # if True, forces smooth_shading to be False regardless of other parameters    
        _temp_input_params = get_dict_subset(self.params, ['should_use_normalized_tuning_curves','should_pdf_normalize_manually','should_nan_non_visited_elements','should_force_placefield_custom_color','should_display_placefield_points', 'should_override_disable_smooth_shading', 'nan_opacity'])
        # print(f'_temp_input_params: {_temp_input_params}')
        
        self.p, self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(self.p, self.params.active_epoch_placefields, self.params.pf_colors, zScalingFactor=self.params.zScalingFactor, show_legend=self.params.show_legend, **_temp_input_params) # note that the get_dict_subset(...) thing is just a safe way to get only the relevant members.
         # Build the widget labels:
        self.params.unit_labels = temp_plots_data['unit_labels'] # fetch the unit labels from the extra data dict.
        self.params.pf_fragile_linear_neuron_IDXs = temp_plots_data['good_placefield_neuronIDs'] # fetch the unit labels from the extra data dict.
        ## TODO: For these, we actually want the placefield value as the Z-positions, will need to unwrap them or something (maybe .ravel(...)?)
        ## TODO: also need to add in the checkbox functionality to hide/show only the spikes for the highlighted units
        # .threshold().elevation()
        
        ## Legend data:
        self.plots_data['tuningCurvePlotLegendData'] = temp_plots_data['legend_entries']
        
    def update_rendered_placefields(self, neuron_id_color_update_dict):
        """ updates the placefields from the new color_update_dict
        May 2022
        """
        update_plotColorsPlacefield2D(self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], neuron_id_color_update_dict=neuron_id_color_update_dict)


    def remove_all_rendered_placefields(self):
        """ removes all placefields added by `plot_placefields`

        ipcDataExplorer.remove_all_rendered_placefields()

        """
        for k, v in self.plots['tuningCurvePlotActors'].items():
            # print(f'k: {k}, v: {v}')
            if v is not None:
                for a_subactor_key, a_subactor in v.items():
                    was_remove_success = self.p.remove_actor(a_subactor)
                    if not was_remove_success:
                            print(f'remove failed for k: {k}, a_subactor_key: {a_subactor_key}')
                ## END for a_subactor_key, a_sub...
            # end if v is not None
        self.plots_data['tuningCurvePlotData'].clear()
        self.plots['tuningCurvePlotActors'].clear()
        tuningCurvePlotLegendActor = self.plots.pop('tuningCurvePlotLegendActor', None)
        if tuningCurvePlotLegendActor is not None:
            self.p.remove_legend(tuningCurvePlotLegendActor)






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
class HideShowPlacefieldsRenderingMixin(PlacefieldOwningMixin):
    """ Implementor Visually Displays Placefield data and enables basic interactivity for it.
    
    Requirements:
        self.plots['tuningCurvePlotActors']
    
        From PlacefieldOwningMixin:
            self.active_tuning_curve_render_configs
            
    Known Uses:
        InteractivePlaceCellTuningCurvesDataExplorer
    
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
        return np.array([bool(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors.values()], dtype=bool)
        
    @property
    def tuning_curve_visibilities(self):
        return np.array([int(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors.values()], dtype=int)

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
        for aTuningCurveActor in self.tuning_curve_plot_actors.values():
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
        self.tuning_curve_plot_actors.values()[show_index].SetVisibility(1)
        
    def on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs):
        """ 
        Wraps self.update_neuron_render_configs(...) internally to update the configs
        """
        # TODO: NON-EXPLICIT INDEXING
        if self.debug_logging:
            print(f'HideShowPlacefieldsRenderingMixin.on_update_tuning_curve_display_config(updated_config_indicies: {updated_config_indicies}, updated_configs: {updated_configs})')
        assert hasattr(self, 'update_neuron_render_configs_from_indicies'), "self must be of type NeuronConfigOwningMixin to have access to its configs"
        self.update_neuron_render_configs_from_indicies(updated_config_indicies, updated_configs) # update the config with the new values:
        for an_updated_config_idx, an_updated_config in zip(updated_config_indicies, updated_configs):
            self.tuning_curve_plot_actors.values()[an_updated_config_idx].SetVisibility(int(self.active_tuning_curve_render_configs[an_updated_config_idx].isVisible)) # update visibility of actor
            
    
    ## Change these names, update_* can easily be called and it does the opposite of what we'd expect
    def update_tuning_curve_configs(self):
        """ update the configs from the actual actors' state """
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors.values()):
            self.active_tuning_curve_render_configs[i].isVisible = bool(aTuningCurveActor.GetVisibility())
            
    def apply_tuning_curve_configs(self):
        """ update the actual actors from the configs """
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors.values()):
            aTuningCurveActor.SetVisibility(int(self.active_tuning_curve_render_configs[i].isVisible))



