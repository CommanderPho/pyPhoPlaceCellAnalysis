from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

from pyphocorehelpers.plotting.media_output_helpers import fig_to_clipboard
from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path, sanitize_filename_for_Windows
from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget

"""

import my_themes
import plotly.io as pio
pio.templates.default = "draft"

"""


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


class PlotlyHelpers:
    """
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

    """

    @classmethod
    def get_plotly_template(cls, is_dark_mode:bool=False):
        """
        
        from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
        
        """
        # template: str = 'plotly_dark' # set plotl template
        # is_dark_mode = False
        # template: str = 'plotly_white'

        if is_dark_mode:
            template: str = "plotly_dark+pho_poster_light"
        else:
            template: str = "plotly+pho_poster_light"
        
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

