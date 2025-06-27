from typing import OrderedDict
import numpy as np
import pandas as pd

# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.spikes_mixins import SpikesDataframeOwningMixin, SpikeRenderingMixin, HideShowSpikeRenderingMixin
from pyphocorehelpers.indexing_helpers import safe_get
from pyphocorehelpers.DataStructure.enum_helpers import OrderedEnum
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode

from qtpy import QtGui # for QColor


class SpikeEmphasisState(OrderedEnum):
    """ The visual state of a given spike, indicating whether it's visible, and its level of emphasis/de-emphasis.
    
    Currently only used in Spike2DRaster to control emphasis of spikes, but general enough that I thought I'd factor it out
    
    See Spike2DRaster._build_cell_configs(...) for more info and how it's used
    
    from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState
    
    """
    Hidden = 0
    Deemphasized = 1
    Default = 2
    Emphasized = 3
    
    
    
class SpikesDataframeOwningMixin:
    """ Implementors own a spikes_df object """
    @property
    def spikes_df(self):
        """The spikes_df property."""
        raise NotImplementedError


    def find_rows_matching_neuron_IDXs(self, neuron_IDXs):
        """Finds the cell IDXs (not IDs) in the self.spikes_df's appropriate column
        Args:
            neuron_IDXs ([type]): [description]
        """
        return np.isin(self.spikes_df['neuron_IDX'], neuron_IDXs)
    
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
        self.params.neuron_colors
        self.params.neuron_colors_hex
        Relies on self.spikes_df['neuron_IDX'] which was just updated prior to the call.

    Provides:        
        # Adds to self.params:
            opaque_neuron_colors

            cell_spike_colors_dict
            cell_spike_opaque_colors_dict

        # Adds columns to self.spikes_df:
            'rgb_hex','R','G','B'

    Used by:
        SpikeRasterBase (and all of its subclasses)
        SpikeRenderingPyVistaMixin -> InteractivePlaceCellTuningCurvesDataExplorer
        
        
    TODO: why does this seem to duplicate nearly all the functionality in DataSeriesColorHelpers???
        I think the answer is because this mixin existed first, and DataSeriesColorHelpers was created in an attempt to generalize and centralize this functionality. 
    """
    
    def _build_flat_color_data(self, fallback_color_rgba = (0, 0, 0, 1.0)):
        """ Called only by self.setup_spike_rendering_mixin()
        
        # Requirements:
            self.params.neuron_colors
            self.params.neuron_colors_hex
            
            
            Relies on self.spikes_df['neuron_IDX'] which was just updated prior to the call.
        
        
        # Adds to self.params:
            opaque_neuron_colors
            
            cell_spike_colors_dict
            cell_spike_opaque_colors_dict
        
        # Adds columns to self.spikes_df:
            'rgb_hex','R','G','B'
        
        
        Seems to use self.params.neuron_colors instead of self.params.pf_colors
        """
        # adds the color information to the self.spikes_df using params.neuron_colors. Adds ['R','G','B'] columns
        # fallback_color_rgb: the default value to use for colors that aren't present in the neuron_colors array
        fallback_color_rgb = fallback_color_rgba[:-1] # Drop the opacity component, so we only have RGB values
        
        # TODO: could also add in 'render_exclusion_mask'
        # RGB Version:
        self.params.opaque_neuron_colors = self.params.neuron_colors[:-1, :].copy() # Drop the opacity component, so we only have RGB values
        
        # Build flat hex colors, creating the self.spikes_df['rgb_hex'] column:
        flat_spike_hex_colors = np.array([safe_get(self.params.neuron_colors_hex, neuron_IDX, '#000000') for neuron_IDX in self.spikes_df['neuron_IDX'].to_numpy()])
        self.spikes_df['rgb_hex'] = flat_spike_hex_colors.copy()

        # if type(self.params.neuron_colors is np.array):
        unique_cell_indicies = np.unique(self.spikes_df['neuron_IDX'].to_numpy())
        max_neuron_IDX = np.max(unique_cell_indicies)
        num_unique_spikes_df_cell_indicies = len(unique_cell_indicies) ## NOTE: num_unique_spikes_df_cell_indicies can be larger than the number of placefields, because some spikes may be in the dataframe from cells that aren't placecells
        
        # Flat version: We need a color for every neuron, whether it is a placecell or not:
        ## TODO: these aren't used anywhere outside of this function, so they can be safely removed. Also these are strangely indexed by neuron_IDXs instead of neuron_ids which I'm trying to get away from.
        self.params.cell_spike_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgba]))
        self.params.cell_spike_opaque_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgb]))
        
        
        num_neuron_colors = np.shape(self.params.neuron_colors)[1]
        valid_neuron_colors_indicies = np.arange(num_neuron_colors)
        for neuron_IDX in unique_cell_indicies:
            if neuron_IDX in valid_neuron_colors_indicies:
                # if we have a color for it, use it
                self.params.cell_spike_colors_dict[neuron_IDX] = self.params.neuron_colors[:, neuron_IDX]
                self.params.cell_spike_opaque_colors_dict[neuron_IDX] = self.params.opaque_neuron_colors[:, neuron_IDX]
            else:
                # Otherwise use the fallbacks:
                print(f'WARNING: neuron_IDX: {neuron_IDX} was not found in valid_neuron_colors_indicies: {valid_neuron_colors_indicies}... USING FALLBACK COLOR.')
                self.params.cell_spike_colors_dict[neuron_IDX] = fallback_color_rgba
                self.params.cell_spike_opaque_colors_dict[neuron_IDX] = fallback_color_rgb
        
        # Add the split RGB columns to the DataFrame
        self.spikes_df[['R','G','B']] = np.array([self.params.cell_spike_opaque_colors_dict.get(idx, fallback_color_rgb) for idx in self.spikes_df['neuron_IDX'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
    
    def on_update_spikes_colors(self, neuron_id_color_update_dict, debug_print=False):
        """ called when the color changes for a spike to update the colors. Internally calls self._update_spikes_df_color_columns(...) """
        self._update_spikes_df_color_columns(neuron_id_color_update_dict, debug_print=debug_print)
    
    def _update_spikes_df_color_columns(self, neuron_id_color_update_dict, debug_print=False):
        """ Updates self.spikes_df's 'R','G','B', and 'rgb_hex' columns only for rows that changed (indicated by having an 'aclu' value that matches the keys passed in, which are treated as neuron_ids
        Requires:
            self.spikes_df
        Inputs:
            neuron_id_color_update_dict: a dictionary with keys of neuron_id and values of type QColor
            
        TODO:
            The following are still invalid (not updated by this function):
            
                self.params.neuron_colors
                self.params.opaque_neuron_colors

            
                self.params.cell_spike_colors_dict
                self.params.cell_spike_opaque_colors_dict
            
        Usage:
            test_updated_colors_map = {3: '#333333', 6:'#666666'}
            ipcDataExplorer.update_spikes_df_color_columns(test_updated_colors_map)

        """
        for neuron_id, color in neuron_id_color_update_dict.items():
            ## Convert color to a QColor for generality:    
            if isinstance(color, QtGui.QColor):
                # already a QColor, just pass
                converted_color = color
            elif isinstance(color, str):
                # if it's a string, convert it to QColor
                converted_color = QtGui.QColor(color)
            elif isinstance(color, (tuple, list, np.array)):
                # try to convert it, hope it's the right size and stuff
                converted_color = QtGui.QColor(color)
            else:
                print(f'ERROR: Color is of unknown type: {color}, type: {type(color)}')
                raise NotImplementedError

            # Set the 'R','G','B' values
            if debug_print:
                print(f'neuron_id: {neuron_id}: converted_color.getRgbF(): {converted_color.getRgbF()}, converted_color.name(QtGui.QColor.HexRgb): {converted_color.name(QtGui.QColor.HexRgb)}')
            self.spikes_df.loc[self.spikes_df['aclu'] == neuron_id, ['R','G','B']] = converted_color.getRgbF()[:-1] # converted_color.getRgbF(): (0.2, 0.2, 0.2, 1.0), so we need to get rid of the last elements. (alternatively we could set ,'render_opacity' if we wanted.
            self.spikes_df.loc[self.spikes_df['aclu'] == neuron_id, ['rgb_hex']] = converted_color.name(QtGui.QColor.HexRgb) #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000' 
                 
    def setup_spike_rendering_mixin(self):
        """ Add the required spike colors built from the self.neuron_colors. Spikes from cells that do not contribute to a placefield are assigned a black color by default
        By Calling self._build_flat_color_data():
            # Adds to self.params:
                opaque_neuron_colors
                
                cell_spike_colors_dict
                cell_spike_opaque_colors_dict
            
            # Adds columns to self.spikes_df:
                'neuron_IDX', 'rgb_hex','R','G','B'
            
        """
        
        ## TODO: IMPORTANT: I think we should overwrite_invalid_fragile_linear_neuron_IDXs before doing this, as correct results from self.get_neuron_id_and_idx(...) depend on valid values for self.fragile_linear_neuron_IDXs and self.neuron_ids

        ## Rebuild the IDXs
        self.spikes_df.spikes._obj, neuron_id_to_new_IDX_map_new_method = self.spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs(debug_print=True)
        # new_neuron_IDXs = list(neuron_id_to_new_IDX_map_new_method.values())

        
        ## This should work now, but is it even needed if self.spikes_df['neuron_IDX'] was just updated in rebuild_fragile_linear_neuron_IDXs(...)??
        included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        self.spikes_df['neuron_IDX'] = included_cell_INDEXES.copy()
        # flat_spike_hex_colors = np.array(flat_spike_hex_colors)
        
        self._build_flat_color_data()
        
    
    # ----------------------------------- Static Methods ---------------------------------------------------------------------
    # ---- factored out of SpikeRasterBase
   
    @classmethod
    def helper_setup_neuron_colors_and_order(cls, raster_plotter, neuron_colors=None, neuron_sort_order=None, debug_print=False):
        """ 
        raster_plotter: a raster plotter
        
        Requires Properties:
            .fragile_linear_neuron_IDXs, .neuron_ids # note any overwrites are actually completely from the data in spikes_df, not dependent on the raster_plotter's .fragile_linear_neuron_IDXs, .neuron_ids
            .spikes_df 
            .enable_overwrite_invalid_fragile_linear_neuron_IDXs

        Requires Functions:
            ## NOT ANYMORE: .find_neuron_IDXs_from_cell_ids(...)
            ._setup_neurons_color_data(...)
            
        Sets Properties:
            ._unit_sort_order
            .cell_id_to_fragile_linear_neuron_IDX_map
            .fragile_linear_neuron_IDX_to_cell_id_map
            
        Uses:
            SpikeRasterBase's __init__(...) function
        
        """
        # Neurons and sort-orders:
        # build a map between the old and new neuron_IDXs:
        if raster_plotter.enable_overwrite_invalid_fragile_linear_neuron_IDXs:
            if debug_print:
                print("WARNING: raster_plotter.enable_overwrite_invalid_fragile_linear_neuron_IDXs is True, so dataframe 'fragile_linear_neuron_IDX' and 'neuron_IDX' will be overwritten!")
            raster_plotter.spikes_df.spikes._obj, neuron_id_to_new_IDX_map_new_method = raster_plotter.spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs(debug_print=debug_print)
            new_neuron_IDXs = list(neuron_id_to_new_IDX_map_new_method.values())
            
            if debug_print:
                print(f'\t\t raster_plotter.cell_ids: {raster_plotter.cell_ids} (len: {len(raster_plotter.cell_ids)})')
                print(f'\t\t new_neuron_IDXs: {new_neuron_IDXs} (len(new_neuron_IDXs): {len(new_neuron_IDXs)})')
    
        # assert (list(neuron_id_to_new_IDX_map.values()) == list(neuron_id_to_new_IDX_map_new_method.values())), f"list(neuron_id_to_new_IDX_map.values()): {list(neuron_id_to_new_IDX_map.values())}\nlist(neuron_id_to_new_IDX_map_new_method.values()): {list(neuron_id_to_new_IDX_map_new_method.values())} should be equal but are NOT!"    
                
        # Build important maps between raster_plotter.fragile_linear_neuron_IDXs and raster_plotter.cell_ids:
        raster_plotter.cell_id_to_fragile_linear_neuron_IDX_map = OrderedDict(zip(raster_plotter.cell_ids, raster_plotter.fragile_linear_neuron_IDXs)) # maps cell_ids to fragile_linear_neuron_IDXs
        raster_plotter.fragile_linear_neuron_IDX_to_cell_id_map = OrderedDict(zip(raster_plotter.fragile_linear_neuron_IDXs, raster_plotter.cell_ids)) # maps fragile_linear_neuron_IDXs to cell_ids
        
        if neuron_sort_order is None:
            neuron_sort_order = np.arange(len(raster_plotter.fragile_linear_neuron_IDXs)) # default sort order is sorted by fragile_linear_neuron_IDXs
        raster_plotter._unit_sort_order = neuron_sort_order
        assert len(raster_plotter._unit_sort_order) == len(raster_plotter.fragile_linear_neuron_IDXs), f"len(raster_plotter._unit_sort_order): {len(raster_plotter._unit_sort_order)} must equal len(raster_plotter.fragile_linear_neuron_IDXs): {len(raster_plotter.fragile_linear_neuron_IDXs)} but it does not!"
        
        # Setup Coloring:
        raster_plotter._setup_neurons_color_data(neuron_colors, coloring_mode=UnitColoringMode.COLOR_BY_INDEX_ORDER)
        

    