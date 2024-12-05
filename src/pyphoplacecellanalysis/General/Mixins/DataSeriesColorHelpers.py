from typing import OrderedDict, Union, Optional, Dict, List
from copy import deepcopy
import numpy as np
import pandas as pd
from attrs import define, field, Factory
from neuropy.utils.mixins.enum_helpers import StringLiteralComparableEnum
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended # for build_cell_display_configs

from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen


@define(repr=False)
class ColorData:
    """ represents a set of color variables as are used in SpikeNDRaster to store properties for neurons.

    """
    neuron_qcolors: List[QColor] = field(default=Factory(list))
    neuron_qcolors_map: dict = field(default=Factory(dict))
    neuron_colors: np.ndarray = field(default=Factory(np.ndarray))
    neuron_colors_hex: List[str] = field(default=Factory(list))

    @classmethod
    def backup_raster_colors(cls, active_2d_plot) -> "ColorData":
        """ extracts the color data from the .params of the SpikeNDRaster plot to back them up and returns them as a ColorData object. """
        return ColorData(*[deepcopy(arr) for arr in (active_2d_plot.params.neuron_qcolors, active_2d_plot.params.neuron_qcolors_map, active_2d_plot.params.neuron_colors, active_2d_plot.params.neuron_colors_hex)])
        

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
    def _build_cell_qcolor_list(cls, fragile_linear_neuron_IDXs, mode:UnitColoringMode=UnitColoringMode.COLOR_BY_INDEX_ORDER, provided_cell_colors=None, debug_print=False):
        """ builds a list of QColors from the cell index id:

        mode:
            'preserve_fragile_linear_neuron_IDXs': color is assigned based off of fragile_linear_neuron_IDX value, meaning after re-sorting the fragile_linear_neuron_IDXs the colors will appear visually different along y but will correspond to the same units as before the sort.
            'color_by_index_order': color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
        
        provided_cell_colors: usually None, in which case a rainbow of colors is built. If not None it better be an np.array of shape (4, n_cells)
        
        Usage:
            # _build_cell_qcolor_list(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode=UnitColoringMode.COLOR_BY_INDEX_ORDER)
            _build_cell_qcolor_list(spike_raster_plt_3d.fragile_linear_neuron_IDXs, mode=UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS)

        

        Called in:
            SpikeRasterBase to build color maps
        """
        n_cells = len(fragile_linear_neuron_IDXs)
        if provided_cell_colors is not None:

            if np.all([isinstance(v, QColor) for v in provided_cell_colors]):
                print(f'WARN: passed an array of QColors to `DataSeriesColorHelpers._build_cell_qcolor_list(...)`, which expected a (4, n_cells) NDArray! 2023-11-28 - Converting and continuing...')
                provided_cell_colors = DataSeriesColorHelpers.qColorsList_to_NDarray(provided_cell_colors, is_255_array=True) # None makes them all black

            assert isinstance(provided_cell_colors, (np.ndarray, list)), f"make sure that it isn't a dict being passed in!"
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
                return [pg.mkColor((fragile_linear_neuron_IDX, n_cells*1.3)) for i, fragile_linear_neuron_IDX in enumerate(sorted_fragile_linear_neuron_IDXs)] # builds varied hues (hue_index, n_hues). see https://pyqtgraph.readthedocs.io/en/latest/api_reference/functions.html#pyqtgraph.intColor
            
        elif mode.name == UnitColoringMode.COLOR_BY_INDEX_ORDER.name:
            # color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
            if provided_cell_colors is not None:
                return [pg.mkColor(provided_cell_colors[:, i]) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)]
            else:
                return [pg.mkColor((i, n_cells*1.3)) for i, fragile_linear_neuron_IDX in enumerate(fragile_linear_neuron_IDXs)] # builds varied hues (hue_index, n_hues). see https://pyqtgraph.readthedocs.io/en/latest/api_reference/functions.html#pyqtgraph.intColor
            
        else:
            raise NotImplementedError

    @classmethod
    def auto_detect_color_NDArray_is_255_array_format(cls, neuron_colors: np.ndarray) -> bool:
        """ tries to auto-detect the format of the color NDArray in terms of whether it contains 0.0-1.0 or 0.0-255.0 values. 
        returns True if it is 255_array_format, and False otherwise
        """
        return (not np.all(neuron_colors <= 1.0)) # all are less than 1.0 implies that it NOT a 255_format_array


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
            neuron_colors = ColorFormatConverter.Colors_NDArray_Convert_to_255_array(neuron_colors) 
        return neuron_colors
    

    @classmethod
    def colors_NDarray_to_qColorsList(cls, neuron_colors: np.ndarray, is_255_array:Optional[bool]=None) -> list:
        """ Takes a [4, nCell] np.array and returns a list[QColor] with the color for each cell in the array
        
        is_255_array: bool - if False, all RGB color values are in range (0.0 - 1.0), else they are in range (0.0 - 255.0)
        
        Note: Matplotlib requires zero_to_one_array format
        """
        if is_255_array is None:
            is_255_array = cls.auto_detect_color_NDArray_is_255_array_format(neuron_colors)

        if is_255_array:
            neuron_colors = ColorFormatConverter.Colors_NDArray_Convert_to_zero_to_one_array(neuron_colors)

        n_cells = neuron_colors.shape[1]
        neuron_qcolors_list = []
        for i in range(n_cells):
            curr_color = QColor.fromRgbF(*neuron_colors[:, i])
            neuron_qcolors_list.append(curr_color)
            
        return neuron_qcolors_list


    @function_attributes(short_name=None, tags=['colors', 'neuron_identity'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-18 11:33', related_items=[])
    @classmethod
    def build_cell_colors(cls, n_neurons:int, colormap_name='hsv', colormap_source='matplotlib', return_255_array: bool=True):
        """Cell Colors from just n_neurons using pyqtgraph colormaps.
        
        Usage:
            from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers

            n_neurons = len(active_2d_plot.neuron_ids)
            neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None)

            ## Preview the new colors
            render_colors([ColorFormatConverter.qColor_to_hexstring(a_qcolor, include_alpha=False) for a_qcolor in neuron_qcolors_list])

        """
        cm = pg.colormap.get(colormap_name, source=colormap_source) # prepare a linear color map

        # unit_colors_list = None # default rainbow of colors for the raster plots
        neuron_qcolors_list = cm.mapToQColor(np.arange(n_neurons)/float(n_neurons-1)) # returns a list of QColors
        # neuron_colors_ndarray = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)
        # neuron_colors_ndarray = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=False)
        neuron_colors_ndarray = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=return_255_array)
        return neuron_qcolors_list, neuron_colors_ndarray


    @classmethod
    def build_cell_display_configs(cls, neuron_ids, neuron_qcolors_list: Optional[List[QColor]]=None, **kwargs) -> Dict[int, SingleNeuronPlottingExtended]:
        """

        Usage:
            neuron_plotting_configs_dict: Dict = DataSeriesColorHelpers.build_cell_display_configs(active_2d_plot.neuron_ids, neuron_qcolors_list)
            spike_raster_window.update_neurons_color_data(neuron_plotting_configs_dict)

        """
        if neuron_qcolors_list is None:
            neuron_qcolors_list, _ = cls.build_cell_colors(len(neuron_ids), **kwargs) # if no colors provided, builds some

        assert len(neuron_ids) == len(neuron_qcolors_list), f"len(neuron_ids): {len(neuron_ids)} must equal len(neuron_qcolors_list): {len(neuron_qcolors_list)}"
        return {aclu:SingleNeuronPlottingExtended(name=str(aclu), isVisible=False, color=color.name(QtGui.QColor.HexRgb), spikesVisible=False) for aclu, color in zip(neuron_ids, neuron_qcolors_list)}
            