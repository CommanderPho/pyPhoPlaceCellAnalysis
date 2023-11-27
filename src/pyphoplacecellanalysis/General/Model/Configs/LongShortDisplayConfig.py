from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional # for PlacefieldOverlapMetricMode
from attrs import define, field, Factory
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, FigureBase # FigureBase: both Figure and SubFigure
from flexitext import flexitext ## flexitext for formatted matplotlib text

from neuropy.core.neuron_identities import PlotStringBrevityModeEnum, NeuronType  # for plot_short_v_long_pf1D_comparison (_display_long_short_pf1D_comparison)
from neuropy.plotting.figure import Fig # for plot_short_v_long_pf1D_comparison (_display_long_short_pf1D_comparison)
from neuropy.plotting.ratemaps import plot_ratemap_1D # for plot_short_v_long_pf1D_comparison (_display_long_short_pf1D_comparison)
from neuropy.utils.matplotlib_helpers import build_or_reuse_figure # used for `_make_pho_jonathan_batch_plots(...)`
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter # for `_plot_long_short_firing_rate_indicies`
from neuropy.utils.matplotlib_helpers import fit_both_axes
from neuropy.utils.matplotlib_helpers import draw_epoch_regions # plot_expected_vs_observed

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.print_helpers import generate_html_string # used for `plot_long_short_surprise_difference_plot`

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget # used by RateRemappingPaginatedFigureController
import pyphoplacecellanalysis.External.pyqtgraph as pg

""" Extreme overkill for this simple setup written by ChatGPT
Written on 2023-09-20

History:

    Previously everything was defined as:

    ```
        long_epoch_config = dict(epoch_label='long', pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'))
        short_epoch_config = dict(epoch_label='short', pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'))

        long_epoch_matplotlib_config = ColorFormatConverter.convert_pen_brush_to_matplot_kwargs(pen=long_epoch_config['pen'], brush=long_epoch_config['brush'])
        short_epoch_matplotlib_config = ColorFormatConverter.convert_pen_brush_to_matplot_kwargs(pen=short_epoch_config['pen'], brush=short_epoch_config['brush'])
    ```

Usage:

    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import long_short_display_config_manager

    long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
    short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

    long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
    short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()


"""


class DisplayColorsEnum:
    """ Hardcoded Theme Colors for visual consistancy - 2023-10-18
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
    
    """
    class Laps:
        even = '#5522de' # a purplish-royal-blue 
        odd = '#aadd21' # a yellowish-green


        @classmethod
        def get_even_dock_colors(cls, orientation, is_dim):
            """ used for CustomDockDisplayConfig for even laps
            
            Usage:
                from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
                
                even_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_even_dock_colors)
            
            
            """
            # DisplayColorsEnum.Laps.even
            # Common to all:
            if is_dim:
                fg_color = '#aaa' # Grey
            else:
                fg_color = '#fff' # White
                
            # a purplish-royal-blue 
            if is_dim:
                bg_color = '#9579e2' 
                border_color = '#7764aa' 
            else:
                bg_color = '#5522de' 
                border_color = '#360bac' 

            return fg_color, bg_color, border_color


        @classmethod
        def get_odd_dock_colors(cls, orientation, is_dim):
            """ used for CustomDockDisplayConfig for odd laps
            
            Usage:
                from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
                
                odd_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_odd_dock_colors)
            
            """
            # DisplayColorsEnum.Laps.odd
            # Common to all:
            if is_dim:
                fg_color = '#aaa' # Grey
            else:
                fg_color = '#fff' # White
                
            # a purplish-royal-blue 
            if is_dim:
                bg_color = '#c1db7a' 
                border_color = '#b0b89b' 
            else:
                bg_color = '#aadd21' # a yellowish-green
                border_color = '#88aa2a' 

            return fg_color, bg_color, border_color


    class Epochs:
        long = '#0b0049' # a dark blue
        short = '#490000' # a dark red
        

@define(slots=False, repr=False)
class DisplayConfig:
    """ Holds display properties for a given configuration """
    epoch_label: str
    pen: object
    brush: object
    hoverBrush: object
    hoverPen: object
    mpl_color: str = field(default=None)  # example property for matplotlib plotting

    def as_pyqtgraph_kwargs(self) -> dict:
        """ Returns properties as a dictionary suitable for pyqtgraph plotting """
        return {'epoch_label': self.epoch_label, 'pen': self.pen, 'brush': self.brush, 'hoverBrush': self.hoverBrush, 'hoverPen': self.hoverPen}

    def as_matplotlib_kwargs(self) -> dict:
        return ColorFormatConverter.convert_pen_brush_to_matplot_kwargs(pen=self.pen, brush=self.brush)

@define(slots=False, repr=False)
class LongShortDisplayConfigManager:
    """ Singleton class to manage all configurations 

    Usage:
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import long_short_display_config_manager

        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

    """
    long_epoch_config: DisplayConfig = field(default=DisplayConfig(epoch_label='long', pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#0b0049'))
    short_epoch_config: DisplayConfig = field(default=DisplayConfig(epoch_label='short', pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#490000'))
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# Access configurations
long_short_display_config_manager = LongShortDisplayConfigManager()

