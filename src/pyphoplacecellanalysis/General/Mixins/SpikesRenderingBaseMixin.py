from typing import OrderedDict
import numpy as np
import pandas as pd

# from PhoPositionalData.plotting.mixins.spikes_mixins import SpikesDataframeOwningMixin, SpikeRenderingMixin, HideShowSpikeRenderingMixin
from pyphocorehelpers.indexing_helpers import safe_get

class SpikesDataframeOwningMixin:
    """ Implementors own a spikes_df object """
    @property
    def spikes_df(self):
        """The spikes_df property."""
        raise NotImplementedError


    def find_rows_matching_cell_IDXs(self, cell_IDXs):
        """Finds the cell IDXs (not IDs) in the self.spikes_df's appropriate column
        Args:
            cell_IDXs ([type]): [description]
        """
        return np.isin(self.spikes_df['cell_idx'], cell_IDXs)
    
    def find_rows_matching_cell_ids(self, cell_ids):
        """Finds the cell original ID in the self.spikes_df's appropriate column
        Args:
            cell_ids ([type]): [description]
        """
        return np.isin(self.spikes_df['aclu'], cell_ids)
    
    
    

# Typically requires conformance to SpikesDataframeOwningMixin
class SpikeRenderingBaseMixin:
    """ Implementors render spikes from neural data in 3D 
    """
    
    def _build_flat_color_data(self, fallback_color_rgba = (0, 0, 0, 1.0)):
        """ Called only by self.setup_spike_rendering_mixin()
        
        # Adds to self.params:
            opaque_neuron_colors
            
            flat_spike_colors_array # for some reason. Length of spikes_df
            
            cell_spike_colors_dict
            cell_spike_opaque_colors_dict
        
        # Adds columns to self.spikes_df:
            'rgb_hex','R','G','B'
        
        """
        # adds the color information to the self.spikes_df using params.neuron_colors. Adds ['R','G','B'] columns and creates a self.params.flat_spike_colors_array with one color for each spike.
        # fallback_color_rgb: the default value to use for colors that aren't present in the neuron_colors array
        fallback_color_rgb = fallback_color_rgba[:-1] # Drop the opacity component, so we only have RGB values
        
        # TODO: could also add in 'render_exclusion_mask'
        # RGB Version:
        self.params.opaque_neuron_colors = self.params.neuron_colors[:-1, :].copy() # Drop the opacity component, so we only have RGB values
        
        # Build flat hex colors, creating the self.spikes_df['rgb_hex'] column:
        flat_spike_hex_colors = np.array([safe_get(self.params.neuron_colors_hex, cell_IDX, '#000000') for cell_IDX in self.spikes_df['cell_idx'].to_numpy()])        
        # flat_spike_hex_colors = np.array([self.params.neuron_colors_hex[cell_IDX] for cell_IDX in self.spikes_df['cell_idx'].to_numpy()])
        self.spikes_df['rgb_hex'] = flat_spike_hex_colors.copy()

        # if type(self.params.neuron_colors is np.array):
        unique_cell_indicies = np.unique(self.spikes_df['cell_idx'].to_numpy())
        max_cell_idx = np.max(unique_cell_indicies)
        num_unique_spikes_df_cell_indicies = len(unique_cell_indicies)
        
        # generate a dict of colors with an entry
        # neuron_colors_dict = {cell_IDX: fallback_color_rgba for cell_IDX in unique_cell_indicies}
        # pf_opaque_colors_dict = {cell_IDX: fallback_color_rgb for cell_IDX in unique_cell_indicies}

        # Flat version:
        self.params.cell_spike_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgba]))
        self.params.cell_spike_opaque_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgb]))
        
        num_neuron_colors = np.shape(self.params.neuron_colors)[0]
        valid_neuron_colors_indicies = np.arange(num_neuron_colors)
        for cell_IDX in unique_cell_indicies:
            if cell_IDX in valid_neuron_colors_indicies:
                # if we have a color for it, use it
                self.params.cell_spike_colors_dict[cell_IDX] = self.params.neuron_colors[:, cell_IDX]
                self.params.cell_spike_opaque_colors_dict[cell_IDX] = self.params.opaque_neuron_colors[:, cell_IDX]
            else:
                # Otherwise use the fallbacks:
                self.params.cell_spike_colors_dict[cell_IDX] = fallback_color_rgba
                self.params.cell_spike_opaque_colors_dict[cell_IDX] = fallback_color_rgb
        
        # self.params.flat_spike_colors_array = np.array([safe_get(self.params.opaque_neuron_colors, idx, fallback_color) for idx in self.spikes_df['cell_idx'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        self.params.flat_spike_colors_array = np.array([self.params.cell_spike_opaque_colors_dict.get(idx, fallback_color_rgb) for idx in self.spikes_df['cell_idx'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        if self.debug_logging:
            print(f'SpikeRenderingBaseMixin.build_flat_color_data(): built rgb array from neuron_colors, droppping the alpha components: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
        # Add the split RGB columns to the DataFrame
        self.spikes_df[['R','G','B']] = self.params.flat_spike_colors_array
        # RGBA version:
        # self.params.flat_spike_colors_array = np.array([self.params.neuron_colors[:, idx] for idx in self.spikes_df['cell_idx'].to_numpy()]) # np.shape(flat_spike_colors) # (77726, 4)
        # self.params.flat_spike_colors_array = np.array([pv.parse_color(spike_color_info.rgb_hex, opacity=spike_color_info.render_opacity) for spike_color_info in self.spikes_df[['rgb_hex', 'render_opacity']].itertuples()])
        # print(f'SpikeRenderMixin.build_flat_color_data(): built combined rgba array from rgb_hex and render_opacity: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
        return self.params.flat_spike_colors_array
              
    def setup_spike_rendering_mixin(self):
        """ Add the required spike colors built from the self.neuron_colors. Spikes that do not contribute to a cell with a placefield are assigned a black color by default
        By Calling self._build_flat_color_data():
            # Adds to self.params:
                opaque_neuron_colors
                
                flat_spike_colors_array # for some reason. Length of spikes_df
                
                cell_spike_colors_dict
                cell_spike_opaque_colors_dict
            
            # Adds columns to self.spikes_df:
                'cell_idx', 'rgb_hex','R','G','B'
            
        """
        included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        # flat_spike_hex_colors = np.array(flat_spike_hex_colors)
        
        self._build_flat_color_data()
        
        