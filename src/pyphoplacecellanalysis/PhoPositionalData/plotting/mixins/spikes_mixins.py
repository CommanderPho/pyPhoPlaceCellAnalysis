from typing import OrderedDict
import numpy as np
import pandas as pd
import pyvista as pv

from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import build_active_spikes_plot_data_df
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

# class SingleCellSpikePlotData(param.Parameterized):
#     point_data = param.Array(doc='spike_history_pdata')
#     glyph_data = param.Array(doc='spike_history_pc')


# class SpikePlotData(param.Parameterized):
#     plot_data = SingleCellSpikePlotData.param

class SpikesDataframeOwningFromSessionMixin(SpikesDataframeOwningMixin):
    """ Implementors own a spikes_df object via their self.active_session property.
    Requires:
        self.active_session
    """
    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self.active_session.spikes_df



# Typically requires conformance to SpikesDataframeOwningMixin
class SpikeRenderingPyVistaMixin(SpikeRenderingBaseMixin):
    """ Adds some PyVista specific properties to SpikeRenderingBaseMixin
    
        Implementors render spikes from neural data in 3D 
        Requires:
            From SpikesDataframeOwningMixin:
                self.spikes_df
                self.find_rows_matching_neuron_IDXs(self, neuron_IDXs)
                self.find_rows_matching_cell_ids(self, cell_ids)
                
        Known Uses:
            InteractivePlaceCellTuningCurvesDataExplorer
            
    """
    debug_logging = True
    
    ## Below seems to be specific to the PyVista (InteractivePlaceCell*DataExplorer) classes:
    spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
    
    def plot_spikes(self):
        historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data_df(self.spikes_df, spike_geom=SpikeRenderingPyVistaMixin.spike_geom_cone.copy(), enable_debug_print=self.debug_logging)        
        self.plots_data['spikes_pf_active'] = {'historical_spikes_pdata':historical_spikes_pdata, 'historical_spikes_pc':historical_spikes_pc}
        if historical_spikes_pc.n_points >= 1:
            # self.plots['spikes_pf_active'] = self.p.add_mesh(historical_spikes_pc, name='spikes_pf_active', scalars='cellID', cmap=self.active_config.plotting_config.active_cells_listed_colormap, show_scalar_bar=False, lighting=True, render=False)
            self.plots['spikes_pf_active'] = self.p.add_mesh(historical_spikes_pc, name='spikes_pf_active', scalars='rgb', rgb=True, show_scalar_bar=False, lighting=True, render=False)
            needs_render = True
        else:
            # self.plots['spikes_pf_active'] = self.p.add_mesh(
            self.p.remove_actor(self.plots['spikes_pf_active'])
            needs_render = True
        return needs_render


    def update_spikes(self):
        """ Called to programmatically update the rendered spikes by replotting after changing their visibility/opacity/postion/etc """
        # full rebuild (to be safe):
        historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data_df(self.spikes_df, spike_geom=SpikeRenderingPyVistaMixin.spike_geom_cone.copy(), enable_debug_print=self.debug_logging)
        self.plots_data['spikes_pf_active'] = {'historical_spikes_pdata':historical_spikes_pdata, 'historical_spikes_pc':historical_spikes_pc}
        
        # Update just the values that could change:
        self.plots_data['spikes_pf_active']['historical_spikes_pdata']['render_opacity'] = self.spikes_df['render_opacity'].values
        # ?? Is this rebuild needed after updating the pdata to see the changes in the pc_data (which is what is actually plotted)???
        self.plots_data['spikes_pf_active']['historical_spikes_pc'] = self.plots_data['spikes_pf_active']['historical_spikes_pdata'].glyph(scale=False, geom=SpikeRenderingPyVistaMixin.spike_geom_cone.copy()) 
        # spike_history_pdata['render_opacity'] = active_flat_df['render_opacity'].values
        
        if self.plots_data['spikes_pf_active']['historical_spikes_pc'].n_points >= 1:
            self.plots['spikes_pf_active'] = self.p.add_mesh(self.plots_data['spikes_pf_active']['historical_spikes_pc'], name='spikes_pf_active', scalars='rgb', rgb=True, show_scalar_bar=False, lighting=True, render=False)
            needs_render = True
        else:
            self.p.remove_actor(self.plots['spikes_pf_active'])
            needs_render = True

        if needs_render:
            self.p.render()

    # Testing:
    def test_toggle_cell_spikes_visibility(self, included_cell_ids):
        self.update_active_spikes(np.isin(self.spikes_df['aclu'], included_cell_ids))
      
        
                  
    """
        Originally used self.pf_colors instead of self.neuron_colors in the base class. The base class was better.
    """
    
    def _build_flat_color_data(self, fallback_color_rgba = (0, 0, 0, 1.0)):
        """ Called only by self.setup_spike_rendering_mixin()
        
        # Adds to self.params:
            opaque_neuron_colors
            
            cell_spike_colors_dict
            cell_spike_opaque_colors_dict
        
        # Adds columns to self.spikes_df:
            'rgb_hex','R','G','B'
        
        """
        
        # Ensures that self.params.neuron_colors is set and is valid:
        self.params.__dict__.setdefault('neuron_colors', self.params.pf_colors) # get the pf_colors if self.params.neuron_colors doesn't exist.
        self.params.__dict__.setdefault('neuron_colors_hex', self.params.pf_colors_hex) # get the pf_colors_hex if self.params.neuron_colors_hex doesn't exist.
        
        # Call the default implementation in the base class:
        return super(SpikeRenderingPyVistaMixin, self)._build_flat_color_data(fallback_color_rgba=fallback_color_rgba)


class HideShowSpikeRenderingMixin:
    """ Implementors present spiking data with the option to hide/show/etc some of the outputs interactively.
    
    
    Calls self.update_spikes() upon any updates to render the changes.
    
    Usage:
        Only used in InteractivePlaceCellTuningCurvesDataExplorer
    """    
    debug_logging = True
        
    @property
    def spike_exclusion_mask(self):
        """The spike_exclusion_mask property."""
        return self.spikes_df['render_exclusion_mask']
    @spike_exclusion_mask.setter
    def spike_exclusion_mask(self, value):
        self.spikes_df['render_exclusion_mask'] = value    
    
    @property
    def additional_render_opacity_modifier(self):
        """The additional_render_opacity_modifier optionally allows implementors to provide an additional column that will be added to the render_opacity prior to clipping.
        Must be either None or an array the same length as a column of self.spikes_df.        
        """
        return None
    # @additional_render_opacity_modifier.setter
    # def additional_render_opacity_modifier(self, value):
    #     self._additional_render_opacity_modifier = value
    
    def setup_hide_show_spike_rendering_mixin(self):
        """ 
        # Adds columns to self.spikes_df:
            'render_opacity', 'render_exclusion_mask'
        """
        self.spikes_df['render_opacity'] = 0.0 # Initialize all spikes to 0.0 opacity, meaning they won't be rendered.
        self.spikes_df['render_exclusion_mask'] = False # all are included (not in the exclusion mask) to begin. This does not mean that they will be visible because 'render_opacity' is still set to zero.
        
        
    # General spike masking functions ____________________________________________________________________________________ #
    def update_active_spikes(self, spike_opacity_mask, is_additive=False):
        """ Main update callback function for visual changes. Updates the self.spikes_df.
                
        Inputs:
            spike_opacity_mask: 
            is_additive:bool : if True, the opacity values in spike_opacity_mask are added to the existing opacity values and then the updated opacities are clipped to [0.0, 1.0]. If False, these values are set as opacities directly.
        Usage: 
            included_cell_ids = [48, 61]
            
            ipcDataExplorer.update_active_spikes(np.isin(ipcDataExplorer.spikes_df['aclu'], included_cell_ids)) # actives only the spikes that have aclu values (cell ids) in the included_cell_ids array.
        """
        assert np.shape(self.spikes_df['render_opacity']) == np.shape(spike_opacity_mask), "spike_opacity_mask must have one value for every spike in self.spikes_df, specifying its opacity"
        if self.debug_logging:
            print(f'update_active_spikes(spike_opacity_mask: ..., is_additive: {is_additive})')
        if self.additional_render_opacity_modifier is not None:
            assert np.shape(self.additional_render_opacity_modifier) == np.shape(spike_opacity_mask), "if self.additional_render_opacity_modifier is not None, it must have one value for every spike in self.spikes_df specifying its additive opacity value"
            spike_opacity_mask = spike_opacity_mask + self.additional_render_opacity_modifier
        if is_additive:
            self.spikes_df['render_opacity'] = np.clip((self.spikes_df['render_opacity'] + spike_opacity_mask), 0.0, 1.0)
        else:
            self.spikes_df['render_opacity'] = spike_opacity_mask
        self.update_spikes()            
            
    def change_spike_rows_included(self, row_specifier_mask, are_included):
        """change_spike_rows_included presents an IDX vs. ID agnostic interface with the self.spikes_df to allow the bulk of the code to work for both cases.

        Args:
            row_specifier_mask ([type]): the boolean mask indentifying rows of the dataframe.
            are_included ([type]): [description]
        """
        if are_included:
            self.update_active_spikes(row_specifier_mask, is_additive=True)
        else:
            # in remove mode, make the passed values negative and again specify is_additive=True mode:
            remove_opacity_specifier = row_specifier_mask # gets the only spikes that are included in the cell_ids
            remove_opacity = np.zeros(np.shape(remove_opacity_specifier))
            remove_opacity[remove_opacity_specifier] = -1 # set to negative one, to ensure that regardless of the current opacity the clipped opacity will be removed (set to 0.0) for these items
            self.update_active_spikes(remove_opacity, is_additive=True)
            
            
            
            
    # Cell/Neuron spikes masking helpers _________________________________________________________________________________ #
    def change_unit_spikes_included(self, neuron_IDXs=None, cell_IDs=None, are_included=True):
        """ Called to update the set of visible spikes for specified cell indicies or IDs
        Args:
            cell_ids ([type]): [description]
            are_included ([type]): [description]
                        
            Internally calls self.change_spike_rows_included(...)
        """
        assert (neuron_IDXs is not None) or (cell_IDs is not None), "You must specify either neuron_IDXs or cell_IDs, but not both"
        if neuron_IDXs is not None:
            # IDXs mode, preferred.
            if self.debug_logging:
                print(f'HideShowSpikeRenderingMixin.change_unit_spikes_included(neuron_IDXs: {neuron_IDXs}, are_included: {are_included}): (note use of Index mode)')            
            matching_rows = self.find_rows_matching_neuron_IDXs(neuron_IDXs)
        else:
            # IDs mode.
            if self.debug_logging:
                print(f'HideShowSpikeRenderingMixin.change_unit_spikes_included(cell_IDs: {cell_IDs}, are_included: {are_included}): WARNING: cell_ID mode. Indexes are preferred.')
            # convert cell_IDs to to neuron_IDXs for use later in updating the configs
            neuron_IDXs = self.find_neuron_IDXs_from_cell_ids(cell_IDs)
            matching_rows = self.find_rows_matching_cell_ids(cell_IDs)
            

        # Update the specific rows:
        self.change_spike_rows_included(matching_rows, are_included)
        
        # update the configs for these changed neurons:
        assert hasattr(self, 'update_neuron_render_configs'), "self must be of type NeuronConfigOwningMixin to have access to its configs"
        updated_configs = []
        
        extracted_cell_ids = self.find_cell_ids_from_neuron_IDXs(neuron_IDXs=neuron_IDXs)
        # Copied from placefield implementation which gets the neuron_ids from the config names and then calls this find_tuning_curve_IDXs_from_neuron_ids(...) business
        config_IDXs = self.find_tuning_curve_IDXs_from_neuron_ids(extracted_cell_ids)
        print(f'change_unit_spikes_included(...):\n\tneuron_IDXs: {neuron_IDXs}\n\textracted_cell_ids:{extracted_cell_ids}\n\tconfig_IDXs:{config_IDXs}')
        # for an_updated_config_idx in neuron_IDXs:
        for an_updated_config_idx in config_IDXs:
            self.active_neuron_render_configs[an_updated_config_idx].spikesVisible = are_included # update the config
            updated_configs.append(self.active_neuron_render_configs[an_updated_config_idx])
        # call the parent (NeuronConfigOwningMixin) function to ensure the configs are updated.
        self.update_neuron_render_configs_from_indicies(config_IDXs, updated_configs) # update the config with the new values:
        
    def clear_all_spikes_included(self):
        # removes all spikes from inclusion
        if self.debug_logging:
            print(f'HideShowSpikeRenderingMixin.clear_spikes_included(): clearing all spikes.')     
        self.change_unit_spikes_included(cell_IDs=self.tuning_curves_valid_neuron_ids, are_included=False)
           

    def mask_spikes_from_render(self, excluded_cell_ids):
        self.spike_exclusion_mask[np.isin(self.spikes_df['aclu'], excluded_cell_ids)] = True
        self.update_spikes()
        
    def unmask_spikes_from_render(self, excluded_cell_ids):
        # removes the specified spikes from the exclusion mask
        self.spike_exclusion_mask[np.isin(self.spikes_df['aclu'], excluded_cell_ids)] = False
        self.update_spikes()
        
    def clear_spikes_exclusion_mask(self):
        self.spike_exclusion_mask = False # all are included (not in the exclusion mask) to begin.
        self.update_spikes()

