from typing import OrderedDict
from copy import deepcopy
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui

class DataSeriesColorHelpers:
    """ Implementors render spikes from neural data in 3D 
        Requires:
            From SpikesDataframeOwningMixin:
                self.spikes_df
                self.find_rows_matching_neuron_IDXs(self, neuron_IDXs)
                self.find_rows_matching_cell_ids(self, cell_ids)
    """
    debug_logging = True
    
    @classmethod
    def _build_cell_color_map(cls, unit_ids, mode='color_by_index_order', debug_print=False):
        """ builds a list of pg.mkColors from the cell index id:     
        Usage:
            # _build_cell_color_map(spike_raster_plt_3d.unit_ids, mode='color_by_index_order')
            _build_cell_color_map(spike_raster_plt_3d.unit_ids, mode='preserve_unit_ids')

        """
        n_cells = len(unit_ids)
        if mode == 'preserve_unit_ids':
            # color is assigned based off of unit_id value, meaning after re-sorting the unit_ids the colors will appear visually different along y but will correspond to the same units as before the sort.
            unit_ids_sort_index = np.argsort(unit_ids) # get the indicies of the sorted ids
            # sorted_unit_ids = np.sort(unit_ids)
            sorted_unit_ids = np.take_along_axis(unit_ids, unit_ids_sort_index, axis=None)
            if debug_print:
                print(f'unit_ids: \t\t{unit_ids}\nunit_ids_sort_index: \t{unit_ids_sort_index}\nsorted_unit_ids: \t{sorted_unit_ids}\n')
            return [pg.mkColor((cell_id, n_cells*1.3)) for i, cell_id in enumerate(sorted_unit_ids)]
        elif mode == 'color_by_index_order':
            # color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
            return [pg.mkColor((i, n_cells*1.3)) for i, cell_id in enumerate(unit_ids)]
        else:
            raise NotImplementedError

    """ Cell Coloring functions:
    """
    @classmethod
    def _setup_neurons_color_data(cls, params, spikes_df, neuron_colors_list=None, coloring_mode='color_by_index_order'):
        """ 
        neuron_colors_list: a list of neuron colors
        
        Sets:
            params.neuron_qcolors
            params.neuron_qcolors_map
            params.neuron_colors: ndarray of shape (4, self.n_cells)
            params.neuron_colors_hex
        """
        
        raise NotImplementedError
        ## TODO: FATAL: this is a classmethod that has references to self.... clearly has never been used.
        # unsorted_unit_ids = self.unit_ids
        
        # # if neuron_colors_list is None:
        # neuron_qcolors_list = cls._build_cell_color_map(unsorted_unit_ids, mode=coloring_mode, provided_cell_colors=neuron_colors_list)
            
        # for a_color in neuron_qcolors_list:
        #     a_color.setAlphaF(0.5)
            
        # # neuron_unit_id_to_colors_index_map = OrderedDict(zip(unsorted_unit_ids, neuron_colors_list))
        # neuron_qcolors_map = OrderedDict(zip(unsorted_unit_ids, neuron_qcolors_list))
        
        # # neuron_colors = []
        # # for i, cell_id in enumerate(self.unit_ids):
        # #     curr_color = pg.mkColor((i, self.n_cells*1.3))
        # #     curr_color.setAlphaF(0.5)
        # #     neuron_colors.append(curr_color)
    
        # params.neuron_qcolors = deepcopy(neuron_qcolors_list)
        # params.neuron_qcolors_map = deepcopy(neuron_qcolors_map)

        # # allocate new neuron_colors array:
        # params.neuron_colors = np.zeros((4, self.n_cells))
        # for i, curr_qcolor in enumerate(params.neuron_qcolors):
        #     curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        #     params.neuron_colors[:, i] = curr_color[:]
        #     # params.neuron_colors[:, i] = curr_color[:]
        
        # params.neuron_colors_hex = None
        
        # # spike_raster_plt.params.neuron_colors[0].getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        
        # # get hex colors:
        # #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        # #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        # params.neuron_colors_hex = [params.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(self.unit_ids)] 
        
        # # included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        
        # # spikes_df['neuron_IDX'] = included_cell_INDEXES.copy()
        # # spikes_df['neuron_IDX'] = spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
        # # TODO: Note that the self.get_neuron_id_and_idx(...) fcn depends on having a self.neuron_ids consistent with whatever is trying ot be passed in as the neuron_ids.
        
    
    def setup_neurons_color_data(self, neuron_colors_list, coloring_mode='color_by_index_order'):
        return DataSeriesColorHelpers._setup_neurons_color_data(self.params, self.spikes_df, neuron_colors_list=neuron_colors_list, coloring_mode=coloring_mode)
        
    @classmethod
    def _build_flat_color_data(cls, params, spikes_df, fallback_color_rgba = (0, 0, 0, 1.0)):
        """ Called only by self.setup_spike_rendering_mixin()
        
        # Adds to params:
            opaque_pf_colors
            
            flat_spike_colors_array # for some reason. Length of spikes_df
            
            cell_spike_colors_dict
            cell_spike_opaque_colors_dict
        
        # Adds columns to spikes_df:
            'rgb_hex','R','G','B'
        
        """
        # adds the color information to the spikes_df using params.pf_colors. Adds ['R','G','B'] columns and creates a params.flat_spike_colors_array with one color for each spike.
        # fallback_color_rgb: the default value to use for colors that aren't present in the pf_colors array
        fallback_color_rgb = fallback_color_rgba[:-1] # Drop the opacity component, so we only have RGB values
        
        # TODO: could also add in 'render_exclusion_mask'
        # RGB Version:
        params.opaque_pf_colors = params.pf_colors[:-1, :].copy() # Drop the opacity component, so we only have RGB values
        
        # Build flat hex colors, creating the spikes_df['rgb_hex'] column:
        flat_spike_hex_colors = np.array([safe_get(params.pf_colors_hex, neuron_IDX, '#000000') for neuron_IDX in spikes_df['neuron_IDX'].to_numpy()])        
        # flat_spike_hex_colors = np.array([params.pf_colors_hex[neuron_IDX] for neuron_IDX in spikes_df['neuron_IDX'].to_numpy()])
        spikes_df['rgb_hex'] = flat_spike_hex_colors.copy()

        # if type(params.pf_colors is np.array):
        unique_cell_indicies = np.unique(spikes_df['neuron_IDX'].to_numpy())
        max_neuron_IDX = np.max(unique_cell_indicies)
        num_unique_spikes_df_cell_indicies = len(unique_cell_indicies)
        
        # generate a dict of colors with an entry
        # pf_colors_dict = {neuron_IDX: fallback_color_rgba for neuron_IDX in unique_cell_indicies}
        # pf_opaque_colors_dict = {neuron_IDX: fallback_color_rgb for neuron_IDX in unique_cell_indicies}

        # Flat version:
        params.cell_spike_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgba]))
        params.cell_spike_opaque_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgb]))
        
        num_pf_colors = np.shape(params.pf_colors)[0]
        valid_pf_colors_indicies = np.arange(num_pf_colors)
        for neuron_IDX in unique_cell_indicies:
            if neuron_IDX in valid_pf_colors_indicies:
                # if we have a color for it, use it
                params.cell_spike_colors_dict[neuron_IDX] = params.pf_colors[:, neuron_IDX]
                params.cell_spike_opaque_colors_dict[neuron_IDX] = params.opaque_pf_colors[:, neuron_IDX]
            else:
                # Otherwise use the fallbacks:
                params.cell_spike_colors_dict[neuron_IDX] = fallback_color_rgba
                params.cell_spike_opaque_colors_dict[neuron_IDX] = fallback_color_rgb
        
        # params.flat_spike_colors_array = np.array([safe_get(params.opaque_pf_colors, idx, fallback_color) for idx in spikes_df['neuron_IDX'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        params.flat_spike_colors_array = np.array([params.cell_spike_opaque_colors_dict.get(idx, fallback_color_rgb) for idx in spikes_df['neuron_IDX'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        if cls.debug_logging:
            print(f'SpikeRenderMixin.build_flat_color_data(): built rgb array from pf_colors, droppping the alpha components: np.shape(params.flat_spike_colors_array): {np.shape(params.flat_spike_colors_array)}')
        # Add the split RGB columns to the DataFrame
        spikes_df[['R','G','B']] = params.flat_spike_colors_array
        # RGBA version:
        # params.flat_spike_colors_array = np.array([params.pf_colors[:, idx] for idx in spikes_df['neuron_IDX'].to_numpy()]) # np.shape(flat_spike_colors) # (77726, 4)
        # params.flat_spike_colors_array = np.array([pv.parse_color(spike_color_info.rgb_hex, opacity=spike_color_info.render_opacity) for spike_color_info in spikes_df[['rgb_hex', 'render_opacity']].itertuples()])
        # print(f'SpikeRenderMixin.build_flat_color_data(): built combined rgba array from rgb_hex and render_opacity: np.shape(params.flat_spike_colors_array): {np.shape(params.flat_spike_colors_array)}')
        return params.flat_spike_colors_array
    
    
    
    def build_flat_color_data(self, fallback_color_rgba = (0, 0, 0, 1.0)):
        return DataSeriesColorHelpers._build_flat_color_data(self.params, self.spikes_df, fallback_color_rgba=fallback_color_rgba)
        
        