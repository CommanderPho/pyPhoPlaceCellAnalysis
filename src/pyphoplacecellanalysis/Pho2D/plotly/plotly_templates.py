from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, ClassVar
import plotly.graph_objects as go
import plotly.io as pio
from copy import deepcopy
from IPython.display import display
from attrs import field, define, Factory
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

# def points_to_pixels(pts: float, dpi: float=300.0) -> float:
#     """ 
#     `Pixеls (px) = Points (pt) * (DPI / 72)`
#     """
#     return float(pts) * (float(dpi) / 72.0)

# text_sizes_px = {'titles': 23, 'axes': 9.2, 'insets': 6.7} #2025-07-03 6pm - original sizes
# text_sizes_px = {k:points_to_pixels(v) for k, v in text_sizes_pts.items()}
# text_sizes_px = {'titles': 38, 'axes': 29, 'insets': 20}
# text_sizes_px = {'titles': 38, 'axes': 29, 'insets': 20}


#TODO 2025-07-03 19:00: - [ ] NEW
text_sizes_px = {'titles': 17.25, 'axes': 9.33333333333333, 'insets': 6.7}



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



@define(slots=False)
class PlotlyHelpers:
    """
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

    plotly_helpers: PlotlyHelpers = PlotlyHelpers(TODAY_DAY_DATE=TODAY_DAY_DATE, figures_folder=figures_folder, neptuner_run=neptuner_run)

    """
    TODAY_DAY_DATE: str = field()
    figures_folder: Path = field()
    
    _is_dark_mode: bool = field(default=False)
    _is_publication: bool = field(default=False)
    neptuner_run: bool = field(default=False)

    should_save: bool = field(default=False)
    export_html: bool = field(default=False)
    export_png: bool = field(default=True)

    resolution_multiplier: float = field(default=1.0)
    # fig_size_kwargs: Dict = field(default=Factory(dict))

    earliest_delta_aligned_t_start: Optional[float] = field(default=None)
    latest_delta_aligned_t_end: Optional[float] = field(default=None)
    legend_groups_to_hide: Optional[List[float]] = field(default=Factory(lambda: ['0.03', '0.044',]))

    active_template: Optional[str] = field(default=None)


    template_dict: ClassVar[Dict[str, Dict]] = deepcopy(_template_dict)  ## NOT AN INSTANCE PROPERTY, a class property

    @property
    def fig_size_kwargs(self) -> Dict:
        """The fig_size_kwargs property."""
        return {'width': (self.resolution_multiplier * 1800), 'height': (self.resolution_multiplier*480)}
        # return {'width': (self.resolution_multiplier * 1650), 'height': (self.resolution_multiplier*480)}

    @property
    def time_delta_tuple(self) -> Tuple[float, float, float]:
        """The time_delta_tuple property."""
        return (self.earliest_delta_aligned_t_start, 0.0, self.latest_delta_aligned_t_end)

    @property
    def is_dark_mode(self) -> bool:
        """The is_dark_mode property."""
        return self._is_dark_mode
    @is_dark_mode.setter
    def is_dark_mode(self, value: bool):
        # self._is_dark_mode = value
        # self._is_dark_mode, self.active_template = PlotlyHelpers.get_plotly_template(is_dark_mode=self._is_dark_mode)
        self.active_template = self.update_plotly_template(is_dark_mode=value)
        # pio.templates.default = self.active_template

    @property
    def is_publication(self) -> bool:
        """The is_publication property."""
        return self._is_publication
    @is_publication.setter
    def is_publication(self, value: bool):
        # self._is_publication = value
        # self._is_dark_mode, self.active_template = PlotlyHelpers.get_plotly_template(is_dark_mode=self._is_dark_mode)
        self.active_template = self.update_plotly_template(is_publication=value)
        # pio.templates.default = self.active_template

    # ==================================================================================================================== #
    # Initialization                                                                                                       #
    # ==================================================================================================================== #
    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # if getattr(cls, 'template_dict', None) is None:
        #     cls.template_dict = deepcopy(_template_dict)

        self._is_dark_mode, self.active_template = PlotlyHelpers.get_plotly_template(is_dark_mode=self.is_dark_mode, is_publication=self.is_publication)
        pio.templates.default = self.active_template
        # self.fig_size_kwargs = {'width': (self.resolution_multiplier * 1800), 'height': (self.resolution_multiplier*480)}


    def save_plotly(self, a_fig, a_fig_context, *, figures_folder: Optional[Path]=None, date_prefix: Optional[str]=None, export_html: Optional[bool] = None, export_png: Optional[bool] = None, **kwargs) -> Dict[str, Path]:
        """Save a Plotly figure under `figures_folder` using an IdentifyingContext-derived basename.

        Defaults write both `.html` and `.png` (legacy notebook behavior). Pass `export_html=False`
        for PNG-only exports.

        Usage:
            from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import save_plotly

            figure_out_paths = save_plotly(a_fig, a_fig_context, figures_folder=figures_folder, date_prefix=TODAY_DAY_DATE, export_html=False, export_png=True)
        """
        if figures_folder is None:
            figures_folder = self.figures_folder
        if date_prefix is None:
            date_prefix = self.TODAY_DAY_DATE
        if export_html is None:
            export_html = self.export_html
        if export_png is None:
            export_png = self.export_png

        # fig_size_kwargs = kwargs.pop('fig_size_kwargs', self.fig_size_kwargs)
        # a_fig = a_fig.update_layout(fig_size_kwargs) ## update size

        return self._perform_save_plotly(a_fig, a_fig_context, figures_folder=figures_folder, date_prefix=date_prefix, export_html=export_html, export_png=export_png, **kwargs)


    def _perform_plot_pre_post_delta_scatter(self, **kwargs):
        """

        History:
            replaces:
                ## captures: earliest_delta_aligned_t_start, latest_delta_aligned_t_end, fig_size_kwargs, is_dark_mode, save_plotly
                _perform_plot_pre_post_delta_scatter = partial(
                    _helper_perform_plot_pre_post_delta_scatter,
                    time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
                    fig_size_kwargs=plotly_helpers.fig_size_kwargs,
                    is_dark_mode=plotly_helpers.is_dark_mode,
                    save_plotly=plotly_helpers.save_plotly,
                    legend_groups_to_hide=['0.03', '0.044',],
                )
        """
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _helper_perform_plot_pre_post_delta_scatter

        return _helper_perform_plot_pre_post_delta_scatter(
                time_delta_tuple=kwargs.pop('time_delta_tuple', (kwargs.pop('earliest_delta_aligned_t_start', self.earliest_delta_aligned_t_start), 0.0, kwargs.pop('latest_delta_aligned_t_end', self.latest_delta_aligned_t_end))),
                fig_size_kwargs=self.fig_size_kwargs,
                is_dark_mode=self.is_dark_mode,
                save_plotly=self.save_plotly,
                legend_groups_to_hide=kwargs.pop('legend_groups_to_hide', self.legend_groups_to_hide),
                **kwargs,
            )

    def _perform_plot_pre_post_delta_scatter_with_embedded_context(self, **kwargs):
        """ overrides `data_context=None` to enforce that df internal data_context is used.

        History:
            replaces: 
            _perform_plot_pre_post_delta_scatter_with_embedded_context = partial(
                _perform_plot_pre_post_delta_scatter,
                data_context=None,
            )

        """
        _discarded_data_context = kwargs.pop('data_context', None)

        return self._perform_plot_pre_post_delta_scatter(data_context=None, **kwargs)


    def update_plotly_template(self, is_dark_mode:Optional[bool]=None, is_publication: Optional[bool]=None):
        """
        
        from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers

        is_dark_mode, template = PlotlyHelpers.update_plotly_template(is_dark_mode=False)
        
        """
        did_change: bool = False
        if (is_dark_mode is not None) and (is_dark_mode != self._is_dark_mode):
            self._is_dark_mode = is_dark_mode
            did_change = True

        if (is_publication is not None) and (is_publication != self._is_publication):
            self._is_publication = is_publication
            did_change = True

        if did_change:
            self._is_dark_mode, self.active_template = self.get_plotly_template(is_dark_mode=self._is_dark_mode, is_publication=self._is_publication)

        return self.active_template


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


    @classmethod
    def _perform_save_plotly(cls, a_fig, a_fig_context, *, figures_folder: Path, date_prefix: str, export_html: bool = True, export_png: bool = True, neptuner_run=None, show_path_widgets: bool = True) -> Dict[str, Path]:
        """Save a Plotly figure under `figures_folder` using an IdentifyingContext-derived basename.

        Defaults write both `.html` and `.png` (legacy notebook behavior). Pass `export_html=False`
        for PNG-only exports.

        Usage:
            from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import save_plotly

            figure_out_paths = PlotlyHelpers._perform_save_plotly(a_fig, a_fig_context, figures_folder=figures_folder, date_prefix=TODAY_DAY_DATE, export_html=False, export_png=True)
        """
        fig_save_path: Path = figures_folder.joinpath('_'.join([date_prefix, sanitize_filename_for_Windows(a_fig_context.get_description())])).resolve()
        figure_out_paths: Dict[str, Path] = {}

        if export_html:
            figure_out_paths['.html'] = fig_save_path.with_suffix('.html')
            a_fig.write_html(figure_out_paths['.html'])
            if show_path_widgets:
                display(fullwidth_path_widget(figure_out_paths['.html'], file_name_label='.html'))

        if export_png:
            figure_out_paths['.png'] = fig_save_path.with_suffix('.png')
            a_fig.write_image(figure_out_paths['.png'])
            if show_path_widgets:
                display(fullwidth_path_widget(figure_out_paths['.png'], file_name_label='.png'))

        if neptuner_run is not None:
            a_full_figure_path_key: str = a_fig_context.get_description(separator='/', include_property_names=True, key_value_separator=':')
            print(f'a_full_figure_path_key: "{a_full_figure_path_key}"')
            upload_path: Optional[Path] = figure_out_paths.get('.html') or figure_out_paths.get('.png')
            if upload_path is not None:
                neptuner_run['outputs']['figures'][f"{a_full_figure_path_key}"].upload(upload_path.as_posix())

        return figure_out_paths






def save_plotly(a_fig, a_fig_context, *, figures_folder: Path, date_prefix: str, export_html: bool = True, export_png: bool = True, neptuner_run=None, show_path_widgets: bool = True, **kwargs) -> Dict[str, Path]:
    """Save a Plotly figure under `figures_folder` using an IdentifyingContext-derived basename.

    Defaults write both `.html` and `.png` (legacy notebook behavior). Pass `export_html=False`
    for PNG-only exports.

    Usage:
        from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import save_plotly

        figure_out_paths = save_plotly(a_fig, a_fig_context, figures_folder=figures_folder, date_prefix=TODAY_DAY_DATE, export_html=False, export_png=True)
    """
    return PlotlyHelpers._perform_save_plotly(a_fig, a_fig_context, figures_folder=figures_folder, date_prefix=date_prefix, export_html=export_html, export_png=export_png, neptuner_run=neptuner_run, show_path_widgets=show_path_widgets, **kwargs)

