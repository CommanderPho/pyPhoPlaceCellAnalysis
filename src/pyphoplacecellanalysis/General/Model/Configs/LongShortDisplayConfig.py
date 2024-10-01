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
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.print_helpers import generate_html_string # used for `plot_long_short_surprise_difference_plot`

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

@function_attributes(short_name=None, tags=['pyqtgraph', 'color'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-29 00:00', related_items=[])
def apply_LR_to_RL_adjustment(an_RL_color):
    """ applies a consistent visual transformation to a color that represents LR direction to get the corresponding RL color. 
    General the RL colors look darker, slightly less saturated

    Usage:
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, DisplayConfig, long_short_display_config_manager

        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

        color_dict = {'long_LR': long_epoch_config['brush'].color(), 'long_RL': LongShortDisplayConfigManager.apply_LR_to_RL_adjustment(long_epoch_config['brush'].color()),
                        'short_LR': short_epoch_config['brush'].color(), 'short_RL': LongShortDisplayConfigManager.apply_LR_to_RL_adjustment(short_epoch_config['brush'].color())}
        color_dict

    """
    from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color

    # RL_adjustment_kwargs = dict(hue_shift=0.0, saturation_scale=0.35, value_scale=1.0)
    # RL_adjustment_kwargs = dict(hue_shift=0.01, saturation_scale=0.75, value_scale=0.5)
    RL_adjustment_kwargs = dict(hue_shift=0.18, saturation_scale=1.0, value_scale=1.0)
    return build_adjusted_color(an_RL_color, **RL_adjustment_kwargs)


class DisplayColorsEnum:
    """ Hardcoded Theme Colors for visual consistancy - 2023-10-18
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
    
    RL_fg_color, RL_bg_color, RL_border_color = DisplayColorsEnum.Laps.get_RL_dock_colors(None, is_dim=False)
    LR_fg_color, LR_bg_color, LR_border_color = DisplayColorsEnum.Laps.get_LR_dock_colors(None, is_dim=False)
    
    """
    class Laps:
        RL = '#5522de' # a purplish-royal-blue
        LR = '#aadd21' # a yellowish-green


        @classmethod
        def get_RL_dock_colors(cls, orientation, is_dim):
            """ used for CustomDockDisplayConfig for even laps
            
            Usage:
                from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
                
                even_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors())
                
                RL_fg_color, RL_bg_color, RL_border_color = DisplayColorsEnum.Laps.get_RL_dock_colors(None, is_dim=False)
                LR_fg_color, LR_bg_color, LR_border_color = DisplayColorsEnum.Laps.get_LR_dock_colors(None, is_dim=False)
            
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
        def get_LR_dock_colors(cls, orientation, is_dim):
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
                fg_color = '#666' # White
                
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
        
        long_dark_bg = '#1f02c2' # a lighter blue for use on dark backgrounds
        short_dark_bg = '#c70000' # a lighter red for use on dark backgrounds
        

    @classmethod
    @function_attributes(short_name=None, tags=['pyqtgraph', 'title', 'format'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-02 15:14', related_items=[])
    def get_pyqtgraph_formatted_title_dict(cls, is_dark_bg: bool = True) -> Dict:
        """ Generates the two-color LR_Long/LR_Short/RL_Long/RL_Short labels with color formatting for pyqtgraph plots
        
        formatted_title_strings_dict = DisplayColorsEnum.get_pyqtgraph_formatted_title_dict()

        """
        long_short_display_config_manager = LongShortDisplayConfigManager()
        long_epoch_config = long_short_display_config_manager.long_epoch_config #.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config #.as_pyqtgraph_kwargs()

        if is_dark_bg:
            Long_color = DisplayColorsEnum.Epochs.long_dark_bg
            Short_color = DisplayColorsEnum.Epochs.short_dark_bg
        else:
            Long_color = long_epoch_config.mpl_color
            Short_color = short_epoch_config.mpl_color


        RL_fg_color, RL_bg_color, RL_border_color = DisplayColorsEnum.Laps.get_RL_dock_colors(None, is_dim=False)
        LR_fg_color, LR_bg_color, LR_border_color = DisplayColorsEnum.Laps.get_LR_dock_colors(None, is_dim=False)

        # formatted_title_strings_dict = {"LR_Long":(generate_html_string("LR", color=LR_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Long", color=Long_color, bold=True, font_size=14)),
        #                         "RL_Long":(generate_html_string("RL", color=RL_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Long", color=Long_color, bold=True, font_size=14)),
        #                         "LR_Short":(generate_html_string("LR", color=LR_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Short", color=Short_color, bold=True, font_size=14)),
        #                         "RL_Short":(generate_html_string("RL", color=RL_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Short", color=Short_color, bold=True, font_size=14)),
        #                         }
        
        # LR_RL_Strings_Dict = {
        #     "LR": "◁",
        #     "RL": "▷",
        # }
        
        # LR_RL_Strings_Dict = {
        #     "LR": "⤞",
        #     "RL": "⤝",
        # }
        
        LR_RL_Strings_Dict = {
            "LR": "◀",
            "RL": "▶",
        }

        formatted_title_strings_dict = {"LR_Long":(generate_html_string(LR_RL_Strings_Dict["LR"], color=LR_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Long", color=Long_color, bold=True, font_size=14)),
                                "RL_Long":(generate_html_string(LR_RL_Strings_Dict["RL"], color=RL_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Long", color=Long_color, bold=True, font_size=14)),
                                "LR_Short":(generate_html_string(LR_RL_Strings_Dict["LR"], color=LR_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Short", color=Short_color, bold=True, font_size=14)),
                                "RL_Short":(generate_html_string(LR_RL_Strings_Dict["RL"], color=RL_bg_color, bold=True, font_size=14) + '_' + generate_html_string("Short", color=Short_color, bold=True, font_size=14)),
                                }
        
        return formatted_title_strings_dict





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
        from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
        return ColorFormatConverter.convert_pen_brush_to_matplot_kwargs(pen=self.pen, brush=self.brush)

@define(slots=False, repr=False)
class LongShortDisplayConfigManager:
    """ Singleton class to manage all configurations 

    Usage:
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, DisplayConfig

        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import long_short_display_config_manager

        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

        long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

    """
    long_epoch_config: DisplayConfig = field(default=DisplayConfig(epoch_label='long', pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#0b0049'))
    short_epoch_config: DisplayConfig = field(default=DisplayConfig(epoch_label='short', pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#490000'))

    ## Colors for use in light-mode
    long_epoch_config_light_mode: DisplayConfig = field(default=DisplayConfig(epoch_label='long', pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#c2b8ff')) # MPL color change for lightmode
    short_epoch_config_light_mode: DisplayConfig = field(default=DisplayConfig(epoch_label='short', pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'), mpl_color='#ffb8b8')) # MPL color change for lightmode

    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def apply_LR_to_RL_adjustment(cls, an_RL_color):
        """ applies a consistent visual transformation to a color that represents LR direction to get the corresponding RL color. 
        General the RL colors look darker, slightly less saturated

        Usage:
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, DisplayConfig, long_short_display_config_manager

            long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
            short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

            color_dict = {'long_LR': long_epoch_config['brush'].color(), 'long_RL': LongShortDisplayConfigManager.apply_LR_to_RL_adjustment(long_epoch_config['brush'].color()),
                            'short_LR': short_epoch_config['brush'].color(), 'short_RL': LongShortDisplayConfigManager.apply_LR_to_RL_adjustment(short_epoch_config['brush'].color())}
            color_dict
  
        """
        from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color

        # RL_adjustment_kwargs = dict(hue_shift=0.0, saturation_scale=0.35, value_scale=1.0)
        RL_adjustment_kwargs = dict(hue_shift=0.01, saturation_scale=0.75, value_scale=0.5)
        return build_adjusted_color(an_RL_color, **RL_adjustment_kwargs)



# Access configurations
long_short_display_config_manager = LongShortDisplayConfigManager()



class PlottingHelpers:
    """ 
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    
    PlottingHelpers.helper_pyqtgraph_add_long_short_session_indicator_regions
    
    import pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig.PlottingHelpers.helper_pyqtgraph_add_long_short_session_indicator_regions as _helper_add_long_short_session_indicator_regions
    
    
    """

    @function_attributes(short_name=None, tags=['matplotlib', 'epoch', 'region'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-23 17:52', related_items=[])
    def helper_matplotlib_add_long_short_epoch_indicator_regions(ax, t_split: float, t_start=None, t_end=None):
        """ Draws the two indicator regions for the long and short track.
        analagous to `_helper_add_long_short_session_indicator_regions` but for matplotlib figures 
        
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
        """
        output_dict = {}
        ## Get the track configs for the colors:
        long_short_display_config_manager = LongShortDisplayConfigManager()
        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

        # Highlight the two epochs with their characteristic colors ['r','b'] - ideally this would be at the very back
        if ((t_start is None) or (t_end is None)):
            x_start_ax, x_stop_ax = ax.get_xlim()
            t_start= (t_start or x_start_ax)
            t_end = (t_end or x_stop_ax)
        output_dict["long_region"] = ax.axvspan(t_start, t_split, color=long_epoch_config['facecolor'], alpha=0.2, zorder=0)
        output_dict["short_region"] = ax.axvspan(t_split, t_end, color=short_epoch_config['facecolor'], alpha=0.2, zorder=0)
        # Update the xlimits with the new bounds
        ax.set_xlim(t_start, t_end)
        
        # Draw the vertical epoch splitter line:
        required_epoch_bar_height = ax.get_ylim()[-1]
        output_dict["divider_line"] = ax.vlines(t_split, ymin=0, ymax=required_epoch_bar_height, color=(0,0,0,.25), zorder=25) # divider should be in very front
        return output_dict


    @function_attributes(short_name=None, tags=['pyqtgraph', 'helper', 'long_short', 'regions', 'rectangles'], input_requires=['pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers.build_pyqtgraph_epoch_indicator_regions'], output_provides=[], uses=[], used_by=[], creation_date='2023-04-19 19:04')
    def helper_pyqtgraph_add_long_short_session_indicator_regions(win, long_epoch, short_epoch):
        """Add session indicators to pyqtgraph plot for the long and the short epoch

                from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _helper_add_long_short_session_indicator_regions

                long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
                short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
                long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch)

                long_epoch_linear_region, long_epoch_region_label = long_epoch_indicator_region_items
                short_epoch_linear_region, short_epoch_region_label = short_epoch_indicator_region_items
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions # Add session indicators to pyqtgraph plot

        long_short_display_config_manager = LongShortDisplayConfigManager()
        long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
        short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

        long_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=long_epoch.t_start, t_stop=long_epoch.t_stop, **long_epoch_config)
        short_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=short_epoch.t_start, t_stop=short_epoch.t_stop, **short_epoch_config)
        return long_epoch_indicator_region_items, short_epoch_indicator_region_items


    @function_attributes(short_name=None, tags=['plotly', 'helper', 'long_short', 'regions', 'rectangles'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 23:03', related_items=[])
    def helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split: float = 0.0, t_start: float=0.0, t_end: float=666.0, yrange=[0.0, 1.0], build_only:bool=False):
        """ Draws the two indicator regions for the long and short track.
        analagous to `_helper_add_long_short_session_indicator_regions` but for plotly figures 
        
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
        t_split: float = 0.0
        _laps_extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig_laps, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end)
        
        
        """
        assert (yrange is not None) and (len(yrange) == 2)
        ymin, ymax = yrange # unpack y-range
        assert (ymin < ymax)

        output_dict = {}
        ## Get the track configs for the colors:
        long_short_display_config_manager = LongShortDisplayConfigManager()
        # long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        # short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()

        # long_epoch_kwargs = dict(fillcolor="blue")
        # short_epoch_kwargs = dict(fillcolor="red")
        long_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.long_epoch_config.mpl_color)
        short_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.short_epoch_config.mpl_color)
        
        
        blue_shape = dict(type="rect", xref="x", yref="paper", x0=t_start, y0=ymin, x1=t_split, y1=ymax, opacity=0.5, layer="below", line_width=1, **long_epoch_kwargs) 
        red_shape = dict(type="rect", xref="x", yref="paper", x0=t_split, y0=ymin, x1=t_end, y1=ymax, opacity=0.5, layer="below", line_width=1, **short_epoch_kwargs)
        vertical_divider_line = dict(type="line", x0=t_split, y0=ymin, x1=t_split, y1=ymax, line=dict(color="rgba(0,0,0,.25)", width=3, ), )
            
        ## new methods
        output_dict["y_zero_line"] = fig.add_hline(y=0.0, line=dict(color="rgba(0,0,0,.25)", width=3, ))

        output_dict["long_region"] = blue_shape
        output_dict["short_region"] = red_shape
        output_dict["divider_line"] = vertical_divider_line
        if not build_only:
            fig.update_layout(shapes=[output_dict["long_region"], output_dict["short_region"], output_dict["divider_line"]], xaxis=dict(range=[t_start, t_end]), yaxis=dict(range=[ymin, ymax]))
        else:
            print(f'WARN: build_only == True so fig.update_layout(...) will not be called.')
        return output_dict


