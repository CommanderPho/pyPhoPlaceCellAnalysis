from pathlib import Path
from typing import Dict
import plotly.graph_objects as go
import plotly.io as pio
from copy import deepcopy

from pyphocorehelpers.plotting.media_output_helpers import fig_to_clipboard
from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path, sanitize_filename_for_Windows
from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget

"""

import my_themes
import plotly.io as pio
pio.templates.default = "draft"


Note: this example uses magic underscore notation to write go.Layout(title=dict(font=dict(...))) as go.Layout(title_font=dict(...))


"""


_template_dict = {

}


pio.templates["draft"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="draft watermark",
            text="DRAFT",
            textangle=-30,
            opacity=0.1,
            font=dict(color="black", size=100),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    ]
)


pio.templates["pho_poster_light"] = go.layout.Template(
    layout_annotations=[
        # dict(
        #     name="draft watermark",
        #     text="DRAFT",
        #     textangle=-30,
        #     opacity=0.1,
        #     font=dict(color="black", size=100),
        #     xref="paper",
        #     yref="paper",
        #     x=0.5,
        #     y=0.5,
        #     showarrow=False,
        # )
    ]
)


# titles: 9, axes: 7, insets: 5
# titles: 23px, axes: 9.2, insets: 6.7
text_sizes_pts = {'titles': 9, 'axes': 7, 'insets': 5}
text_sizes_px = {'titles': 23, 'axes': 9.2, 'insets': 6.7}
# pio.templates["pho_diba_publication"] = go.layout.Template(
#     layout= dict(
#             # font_family="Ariel",
#             # font_color="blue",
#             # title_font_family="Ariel",
#             # title_font_color="red",
#             # legend_title_font_color="green",
#             # font_size=7,
#         #     name="draft watermark",
#         #     text="DRAFT",
#         #     textangle=-30,
#         #     opacity=0.1,
#         font=dict(family="Ariel", color="black", size=7),
#         title_font=dict(family="Ariel", color="black", size=9),
#         legend_title_font=dict(family="Ariel", color="black", size=7),
#         #     xref="paper",
#         #     yref="paper",
#         #     x=0.5,
#         #     y=0.5,
#         #     showarrow=False,
#         ),
# )


# Publication-quality settings with scientific focus
_template_dict["pho_diba_publication"] = dict(
    layout=dict(
        # Main text elements
        font=dict(family="Arial", color="black", size=text_sizes_px['insets']),  # Base font (note: "Arial" not "Ariel")
        # annotations_font=dict(family="Arial", color="black", size=text_sizes_px['titles']),
        
        title_font=dict(family="Arial", color="black", size=text_sizes_px['titles']),

        # Axis labels - slightly larger than tick labels
        xaxis_title_font=dict(family="Arial", color="black", size=text_sizes_px['axes']),
        yaxis_title_font=dict(family="Arial", color="black", size=text_sizes_px['axes']),

        # Tick labels - smaller than axis titles
        xaxis_tickfont=dict(family="Arial", color="black", size=text_sizes_px['axes']),
        yaxis_tickfont=dict(family="Arial", color="black", size=text_sizes_px['axes']),

        # Legend - similar size to tick labels
        legend_title_font=dict(family="Arial", color="black", size=text_sizes_px['axes']),
        legend_font=dict(family="Arial", color="black", size=text_sizes_px['axes']),

        # Clean white background for publication
        paper_bgcolor="white",
        plot_bgcolor="white",

        # Thinner lines for axes
        xaxis_linewidth=1,
        yaxis_linewidth=1,
    ),
)

# for k, v in _template_dict.items():
#     pio.templates[k] = go.layout.Template(v)


# So I did some investigating and found that the title font was actually 16px despite specifying size=9, while the two axes lables were correct.

# ```
# # Title: font-size: 16px
# <text class="annotation-text" style="font-family: Arial; font-size: 16px; fill: rgb(0, 0, 0); fill-opacity: 1; font-weight: normal; font-style: normal; font-variant: normal; white-space: pre;" text-anchor="middle" data-unformatted="Across Sessions time_bin_df - Lap Individual Time Bins - None - 'P_Short' (7 Sessions) - time bin size: 0.025 sec" data-math="N" x="402.4833984375" y="17">Across Sessions time_bin_df - Lap Individual Time Bins - None - 'P_Short' (7 Sessions) - time bin size: 0.025 sec</text>

# # XLabel: font-size: 8px
# <text class="xtitle" style="opacity: 1; font-family: Arial; font-size: 8px; fill: rgb(0, 0, 0); fill-opacity: 1; font-weight: normal; font-style: normal; font-variant: normal; white-space: pre;" x="187.469" y="440.84954223632815" text-anchor="middle" data-unformatted="# Events" data-math="N"># Events</text>

# # YLabel: font-size: 9px;
# <text class="ytitle" transform="rotate(-90,80.6166015625,225)" style="opacity: 1; font-family: Arial; font-size: 9px; fill: rgb(0, 0, 0); fill-opacity: 1; font-weight: normal; font-style: normal; font-variant: normal; white-space: pre;" x="80.6166015625" y="225" text-anchor="middle" data-unformatted="Probability of Short Track" data-math="N">Probability of Short Track</text>

# Annotation text: font-size: 12px;
# <text class="annotation-text" style="font-family: Arial; font-size: 12px; fill: rgb(128, 128, 128); fill-opacity: 1; font-weight: normal; font-style: normal; font-variant: normal; white-space: pre;" text-anchor="middle" data-unformatted="laps|per_time_bin|Lap Individual Time Bins|time_bin_df|trained_compute_epochs_widget_decoder_identifier_widget_masked_time_bin_fill_type_widget" data-math="N" x="405.3662109375" y="14">laps|per_time_bin|Lap Individual Time Bins|time_bin_df|trained_compute_epochs_widget_decoder_identifier_widget_masked_time_bin_fill_type_widget</text>

# ```



# pio.templates["pho_diba_publication"] = go.layout.Template(
#     layout=dict(
#         font_family="Ariel",
#         font_color="blue",
#         title_font_family="Ariel",
#         title_font_color="red",
#         legend_title_font_color="green",
#         font_size=7,
#     ),
# )




class PlotlyHelpers:
    """
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

    """
    template_dict: Dict[str, Dict] = deepcopy(_template_dict)

    @classmethod
    def get_plotly_template(cls, is_dark_mode:bool=False, is_publication: bool=True):
        """
        
        from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
        
        """
        # template: str = 'plotly_dark' # set plotl template
        # is_dark_mode = False
        # template: str = 'plotly_white'

        for k, v in cls.template_dict.items():
            pio.templates[k] = go.layout.Template(**v)
            
    
        if is_dark_mode:
            template: str = "plotly_dark"
        else:
            template: str = "plotly"
        
        if is_publication:
            template += '+pho_diba_publication'
        else:
            template += '+pho_poster_light'

        # template: str = "plotly+draft"
        
        pio.templates.default = template

        return is_dark_mode, template






# def save_plotly(a_fig, a_fig_context):
#     """ 
#     captures: TODAY_DAY_DATE
#     """
#     fig_save_path: Path = figures_folder.joinpath('_'.join([TODAY_DAY_DATE, sanitize_filename_for_Windows(a_fig_context.get_description())])).resolve()
#     figure_out_paths = {'.html': fig_save_path.with_suffix('.html'), '.png': fig_save_path.with_suffix('.png')}
#     a_fig.write_html(figure_out_paths['.html'])
#     display(fullwidth_path_widget(figure_out_paths['.html'], file_name_label='.html'))
#     # print(file_uri_from_path(figure_out_paths['.html']))
#     a_fig.write_image(figure_out_paths['.png'])
#     # print(file_uri_from_path(figure_out_paths['.png']))
#     display(fullwidth_path_widget(figure_out_paths['.png'], file_name_label='.png'))
#     return figure_out_paths

