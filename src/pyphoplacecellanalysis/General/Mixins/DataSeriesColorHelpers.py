from typing import OrderedDict
from copy import deepcopy
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui

class DataSeriesColorHelpers:
    """ An attempt to factor out the common color-related functionality from SpikeRenderingBaseMixin since this functionality is not specific to spike visualizations, for example it's needed to properly color placefields or indicate neuron identities in general.
        
        OBJECTIVE: Implement only @classmethod functions on this class.
    """
    debug_logging = True
    
    @classmethod
    def _build_cell_color_map(cls, fragile_linear_neuron_IDXs, mode='color_by_index_order', provided_cell_colors=None, debug_print=False):
        """ builds a list of pg.mkColors from the cell index id:     
        
        provided_cell_colors: usually None, in which case a rainbow of colors is built. If not None it better be an np.array of shape (4, n_cells)
        
        Usage:
            # _build_cell_color_map(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode='color_by_index_order')
            _build_cell_color_map(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode='preserve_fragile_linear_neuron_IDXs')


        Called in:
            SpikeRasterBase to build color maps
        """
        n_cells = len(fragile_linear_neuron_IDXs)
        if mode == 'preserve_fragile_linear_neuron_IDXs':
            # color is assigned based off of fragile_linear_neuron_IDX value, meaning after re-sorting the fragile_linear_neuron_IDXs the colors will appear visually different along y but will correspond to the same units as before the sort.
            fragile_linear_neuron_IDXs_sort_index = np.argsort(fragile_linear_neuron_IDXs) # get the indicies of the sorted ids
            # sorted_fragile_linear_neuron_IDXs = np.sort(fragile_linear_neuron_IDXs)
            sorted_fragile_linear_neuron_IDXs = np.take_along_axis(fragile_linear_neuron_IDXs, fragile_linear_neuron_IDXs_sort_index, axis=None)
            if debug_print:
                print(f'fragile_linear_neuron_IDXs: \t\t{fragile_linear_neuron_IDXs}\nfragile_linear_neuron_IDXs_sort_index: \t{fragile_linear_neuron_IDXs_sort_index}\nsorted_fragile_linear_neuron_IDXs: \t{sorted_fragile_linear_neuron_IDXs}\n')
            
            if provided_cell_colors is not None:
                return [pg.mkColor(provided_cell_colors[:, fragile_linear_neuron_IDX]) for i, fragile_linear_neuron_IDX in enumerate(sorted_fragile_linear_neuron_IDXs)]
            else:
                return [pg.mkColor((fragile_linear_neuron_IDX, n_cells*1.3)) for i, fragile_linear_neuron_IDX in enumerate(sorted_fragile_linear_neuron_IDXs)]
            
        elif mode == 'color_by_index_order':
            # color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
            if provided_cell_colors is not None:
                return [pg.mkColor(provided_cell_colors[:, i]) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)]
            else:
                return [pg.mkColor((i, n_cells*1.3)) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)]
            
        else:
            raise NotImplementedError

