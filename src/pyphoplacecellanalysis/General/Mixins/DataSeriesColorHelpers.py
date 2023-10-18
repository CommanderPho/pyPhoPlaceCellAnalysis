from typing import OrderedDict, Union
from copy import deepcopy
import numpy as np
import pandas as pd
from neuropy.utils.mixins.enum_helpers import StringLiteralComparableEnum
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui

@metadata_attributes(short_name=None, tags=['enum'], creation_date='2023-06-21 13:50', related_items=['DataSeriesColorHelpers'])
class UnitColoringMode(StringLiteralComparableEnum):
    """Specifies how the neurons are colored."""
    PRESERVE_FRAGILE_LINEAR_NEURON_IDXS = "preserve_fragile_linear_neuron_IDXs"
    COLOR_BY_INDEX_ORDER = "color_by_index_order"
    

@metadata_attributes(short_name=None, tags=['color', 'dataseries', 'series', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:50', related_items=['UnitColoringMode'])
class DataSeriesColorHelpers:
    """ An attempt to factor out the common color-related functionality from SpikeRenderingBaseMixin since this functionality is not specific to spike visualizations, for example it's needed to properly color placefields or indicate neuron identities in general.
        
        OBJECTIVE: Implement only @classmethod functions on this class.
    """
    debug_logging = False
    
    @classmethod
    def _build_cell_color_map(cls, fragile_linear_neuron_IDXs, mode:UnitColoringMode=UnitColoringMode.COLOR_BY_INDEX_ORDER, provided_cell_colors=None, debug_print=False):
        """ builds a list of pg.mkColors from the cell index id:

        mode:
            'preserve_fragile_linear_neuron_IDXs': color is assigned based off of fragile_linear_neuron_IDX value, meaning after re-sorting the fragile_linear_neuron_IDXs the colors will appear visually different along y but will correspond to the same units as before the sort.
            'color_by_index_order': color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
        
        provided_cell_colors: usually None, in which case a rainbow of colors is built. If not None it better be an np.array of shape (4, n_cells)
        
        Usage:
            # _build_cell_color_map(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode=UnitColoringMode.COLOR_BY_INDEX_ORDER)
            _build_cell_color_map(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode=UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS)

        

        Called in:
            SpikeRasterBase to build color maps
        """
        n_cells = len(fragile_linear_neuron_IDXs)
        if provided_cell_colors is not None:
            assert isinstance(provided_cell_colors, np.ndarray)
            assert provided_cell_colors.shape[0] == 4, f"provided_cell_colors should be a (4, n_cells) {(4, {n_cells})} array of colors but provided_cell_colors.shape: {provided_cell_colors.shape}"
            assert provided_cell_colors.shape[1] >= n_cells, f"provided_cell_colors should be a (4, n_cells) {(4, {n_cells})} array of colors but provided_cell_colors.shape: {provided_cell_colors.shape}"
            
        if mode.name == UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS.name:
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
            
        elif mode.name == UnitColoringMode.COLOR_BY_INDEX_ORDER.name:
            # color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
            if provided_cell_colors is not None:
                return [pg.mkColor(provided_cell_colors[:, i]) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)]
            else:
                return [pg.mkColor((i, n_cells*1.3)) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)]
            
        else:
            raise NotImplementedError

    @classmethod
    def Colors_NDArray_Convert_to_255_array(cls, colors_ndarray: np.ndarray) -> np.ndarray:
        """ takes an [4, nCell] np.array of (0.0 - 255.0) values for the color and converts it to a 0.0-1.0 array of the same shape.
        Reciprocal: Colors_NDArray_Convert_to_zero_to_one_array
        """
        converted_colors_ndarray = deepcopy(colors_ndarray)
        converted_colors_ndarray[0:2, :] *= 255 # [1.0, 0.0, 0.0, 1.0]
        return converted_colors_ndarray
    
    @classmethod
    def Colors_NDArray_Convert_to_zero_to_one_array(cls, colors_ndarray: np.ndarray) -> np.ndarray:
        """ takes an [4, nCell] np.array of 0.0-1.0 values for the color and converts it to a (0.0 - 255.0) array of the same shape.
        Reciprocal: Colors_NDArray_Convert_to_255_array
        """
        converted_colors_ndarray = deepcopy(colors_ndarray)
        converted_colors_ndarray[0:2, :] /= 255
        return converted_colors_ndarray
    

    @classmethod
    def qColorsList_to_NDarray(cls, neuron_qcolors_list, is_255_array:bool) -> np.ndarray:
        """ takes a list[QColor] and returns a [4, nCell] np.array with the color for each in the list 
        
        is_255_array: bool - if False, all RGB color values are (0.0 - 1.0), else they are (0.0 - 255.0)
        I was having issues with this list being in the range 0.0-1.0 instead of 0-255.
        
        Note: Matplotlib requires zero_to_one_array format
        
        """
        # allocate new neuron_colors array:
        n_cells = len(neuron_qcolors_list)
        neuron_colors = np.zeros((4, n_cells))
        for i, curr_qcolor in enumerate(neuron_qcolors_list):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            neuron_colors[:, i] = curr_color[:]
        if is_255_array:
            neuron_colors = cls.Colors_NDArray_Convert_to_255_array(neuron_colors) 
        return neuron_colors



