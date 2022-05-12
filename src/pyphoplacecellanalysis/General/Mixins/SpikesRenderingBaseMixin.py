from typing import OrderedDict
import numpy as np
import pandas as pd

# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.spikes_mixins import SpikesDataframeOwningMixin, SpikeRenderingMixin, HideShowSpikeRenderingMixin
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
    



class SpikeRenderingBaseMixin:
    """ Implementors render spikes from neural data in 3D 
    
    Typically requires conformance to SpikesDataframeOwningMixin
    
    Requirements:
        Implementors must conform to NeuronIdentityAccessingMixin or at least have a self.get_neuron_id_and_idx(...) function.
        self.spikes_df
        
    Used by:
        SpikeRasterBase (and all of its subclasses)
        
    TODO: why does this seem to duplicate nearly all the functionality in DataSeriesColorHelpers???
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
        
        
        Seems to use self.params.neuron_colors instead of self.params.pf_colors
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
        
        # if self.debug_logging:
        #     print(f'SpikeRenderingBaseMixin.build_flat_color_data(): built rgb array from neuron_colors, droppping the alpha components: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
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
        
    
    # ----------------------------------- Static Methods ---------------------------------------------------------------------
    # ---- factored out of SpikeRasterBase
    
    @classmethod
    def overwrite_invalid_unit_ids(cls, spikes_df, neuron_id_to_new_IDX_map):
        """
            Ensures that all the unit_ids are valid.
        
        Called by helper_setup_neuron_colors_and_order(...)
        
        """
        # if self.enable_overwrite_invalid_unit_ids:
        print("WARNING: self.enable_overwrite_invalid_unit_ids is True, so dataframe 'unit_id' and 'cell_idx' will be overwritten!")
        spikes_df['old_unit_id'] = spikes_df['unit_id'].copy()
        included_cell_INDEXES = np.array([neuron_id_to_new_IDX_map[an_included_cell_ID] for an_included_cell_ID in spikes_df['aclu'].to_numpy()], dtype=int) # get the indexes from the cellIDs
        print('\t computed included_cell_INDEXES.')
        spikes_df['unit_id'] = included_cell_INDEXES.copy() # TODO: CRITICAL: why are IDXs being assigned to a property named *_id (unit_id here)??? the _id suffix should always mean that it's the ACLU value, right??
        print("\t set spikes_df['unit_id']")
        # self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        spikes_df['cell_idx'] = spikes_df['unit_id'].copy()
        print("\t set spikes_df['cell_idx']")
        print("\t done updating 'unit_id' and 'cell_idx'.")
        

    @classmethod
    def helper_setup_neuron_colors_and_order(cls, raster_plotter, neuron_colors=None, neuron_sort_order=None):
        """ 
        raster_plotter: a raster plotter
        
        Requires Properties:
            .unit_ids, .neuron_ids
            .spikes_df
            .enable_overwrite_invalid_unit_ids

        Requires Functions:
            .find_cell_IDXs_from_cell_ids(...)
            ._setup_neurons_color_data(...)
            
        Sets Properties:
            ._unit_sort_order
            .cell_id_to_unit_id_map
            .unit_id_to_cell_id_map
            
            
        Uses:
            SpikeRasterBase's __init__(...) function
        
        """
        # Neurons and sort-orders:
        old_neuron_IDXs = raster_plotter.unit_ids.copy() # backup the old unit_ids
        print(f'\t\t raster_plotter.unit_ids: {raster_plotter.unit_ids} (len: {len(raster_plotter.unit_ids)})\n \t\t raster_plotter.cell_ids: {raster_plotter.cell_ids} (len: {len(raster_plotter.cell_ids)})')
        new_neuron_IDXs = raster_plotter.find_cell_IDXs_from_cell_ids(raster_plotter.neuron_ids)
        print(f'\t\t new_neuron_IDXs: {new_neuron_IDXs} (len(new_neuron_IDXs): {len(new_neuron_IDXs)})')
        # build a map between the old and new neuron_IDXs:
        old_to_new_map = OrderedDict(zip(old_neuron_IDXs, new_neuron_IDXs))
        new_to_old_map = OrderedDict(zip(new_neuron_IDXs, old_neuron_IDXs))
        neuron_id_to_new_IDX_map = OrderedDict(zip(raster_plotter.neuron_ids, new_neuron_IDXs)) # provides the new_IDX corresponding to any neuron_id (aclu value)
        
        if raster_plotter.enable_overwrite_invalid_unit_ids:
            print("WARNING: raster_plotter.enable_overwrite_invalid_unit_ids is True, so dataframe 'unit_id' and 'cell_idx' will be overwritten!")
            cls.overwrite_invalid_unit_ids(raster_plotter.spikes_df, neuron_id_to_new_IDX_map)
        
        # Build important maps between raster_plotter.unit_ids and raster_plotter.cell_ids:
        raster_plotter.cell_id_to_unit_id_map = OrderedDict(zip(raster_plotter.cell_ids, raster_plotter.unit_ids)) # maps cell_ids to unit_ids
        raster_plotter.unit_id_to_cell_id_map = OrderedDict(zip(raster_plotter.unit_ids, raster_plotter.cell_ids)) # maps unit_ids to cell_ids
        
        if neuron_sort_order is None:
            neuron_sort_order = np.arange(len(raster_plotter.unit_ids)) # default sort order is sorted by unit_ids
        raster_plotter._unit_sort_order = neuron_sort_order
        assert len(raster_plotter._unit_sort_order) == len(raster_plotter.unit_ids), f"len(raster_plotter._unit_sort_order): {len(raster_plotter._unit_sort_order)} must equal len(raster_plotter.unit_ids): {len(raster_plotter.unit_ids)} but it does not!"
        
        # Setup Coloring:
        raster_plotter._setup_neurons_color_data(neuron_colors, coloring_mode='color_by_index_order')
        
    

    