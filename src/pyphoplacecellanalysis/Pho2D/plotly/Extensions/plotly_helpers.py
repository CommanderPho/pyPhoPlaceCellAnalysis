from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from plotly import graph_objs as go
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path


def plotly_helper_save_figures(figures_folder: Optional[Path]=None, figure_save_extension: Union[str, List[str], Tuple[str]]='.png'):
    """ save figures to the 'figures' subfolder

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import plotly_helper_save_figures


    """
    # Create a 'figures' subfolder if it doesn't exist
    if figures_folder is None:
        figures_folder: Path = Path('figures').resolve()
    figures_folder.mkdir(parents=False, exist_ok=True)
    assert figures_folder.exists()
    print(f'\tfigures_folder: {file_uri_from_path(figures_folder)}')

    # Save the figures to the 'figures' subfolder
    assert figure_save_extension is not None
    if isinstance(figure_save_extension, str):
        figure_save_extension = [figure_save_extension] # a list containing only this item

    print(f'\tsaving figures...')
    save_fn_dict = {}
    for a_fig_save_extension in figure_save_extension:
        if a_fig_save_extension.lower() == '.html':
                a_save_fn = lambda a_fig, a_save_name: a_fig.write_html(a_save_name)
        else:
                a_save_fn = lambda a_fig, a_save_name: a_fig.write_image(a_save_name)

        ## Add to save fn dict:
        save_fn_dict[a_fig_save_extension.lower()] = a_save_fn

        # fig_laps_name = Path(figures_folder, f"{laps_title_string_suffix.replace(' ', '-')}_{laps_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
        # print(f'\tsaving "{file_uri_from_path(fig_laps_name)}"...')
        # a_save_fn(fig_laps, fig_laps_name)
        # fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_{ripple_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
        # print(f'\tsaving "{file_uri_from_path(fig_ripple_name)}"...')
        # a_save_fn(fig_ripples, fig_ripple_name)

        # def _perform_save_with_extension(a_fig, a_save_name):
        #     print(f'\tsaving "{file_uri_from_path(a_save_name)}"...')
        #     a_save_fn(a_fig, a_save_name)


    def _perform_save_all_extensions(a_fig, a_save_name):
        """ captures `save_fn_dict` """


        for a_fig_save_extension, a_save_fn in save_fn_dict.items():
            print(f'\tsaving "{file_uri_from_path(a_save_name)}"...')
            a_save_fn(a_fig, a_save_name)

    return _perform_save_all_extensions, save_fn_dict

@function_attributes(short_name=None, tags=['plotly', 'scatter'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-27 09:07', related_items=[])
def plotly_pre_post_delta_scatter(data_results_df: pd.DataFrame, out_scatter_fig=None, histogram_bins:int=25, px_scatter_kwargs=None,
                                   histogram_variable_name='P_Long', hist_kwargs=None,
                                   forced_range_y=[0.0, 1.0], time_delta_tuple=None, is_dark_mode: bool = True, **kwargs):
    """ Plots a scatter plot of a variable pre/post delta, with a histogram on each end corresponding to the pre/post delta distribution

    px_scatter_kwargs: only used if out_scatter_fig is None
    time_delta_tuple=(earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end)


    Usage:

        import plotly.io as pio
        template: str = 'plotly_dark' # set plotl template
        pio.templates.default = template
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import plotly_pre_post_delta_scatter


        histogram_bins: int = 25

        new_laps_fig = plotly_pre_post_delta_scatter(data_results_df=deepcopy(all_sessions_laps_df), out_scatter_fig=fig_laps, histogram_bins=histogram_bins)
        new_laps_fig

    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go

    figure_context_dict = {'histogram_variable_name': histogram_variable_name}

    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: NDArray = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    if px_scatter_kwargs is None:
        px_scatter_kwargs = {}

    # f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes"
    # main_title: str = f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes"
    if num_unique_sessions == 1:
        # print(f'single-session mode')
        main_title: str = f"Session {px_scatter_kwargs.get('title', 'UNKNOWN')}"
    else:
        main_title: str = f"Across Sessions {px_scatter_kwargs.get('title', 'UNKNOWN')} ({num_unique_sessions} Sessions)"


    if num_unique_time_bins > 1:
        main_title = main_title + f" - {num_unique_time_bins} Time Bin Sizes"
        figure_context_dict['n_tbin'] = num_unique_time_bins
    else:
        time_bin_size = time_bin_sizes[0]
        main_title = main_title + f" - time bin size: {time_bin_size} sec"

    figure_context_dict['title'] = main_title
    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')

    # common_plot_kwargs = dict(color="time_bin_size")
    common_plot_kwargs = dict() # color=None
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs = dict(opacity=0.5, range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay') | hist_kwargs
    # print(f'hist_kwargs: {hist_kwargs}')

    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    # ==================================================================================================================== #
    # Build Figure                                                                                                         #
    # ==================================================================================================================== #
    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01, shared_yaxes=True, column_titles=["Pre-delta", main_title, "Post-delta"])
    already_added_legend_entries = set()  # Keep track of trace names that are already added

    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    pre_delta_fig = px.histogram(pre_delta_df, y=histogram_variable_name, **common_plot_kwargs, **hist_kwargs, title="Pre-delta")
    print(f'len(pre_delta_fig.data): {len(pre_delta_fig.data)}')
    # time_bin_sizes
    for a_trace in pre_delta_fig.data:
        a_trace_name = a_trace.name
        if a_trace_name in already_added_legend_entries:
            # For already added trace categories, set showlegend to False
            a_trace.showlegend = False
        else:
            # For the first trace of each category, keep showlegend as True
            already_added_legend_entries.add(a_trace_name)
            a_trace.showlegend = True  # This is usually true by default, can be omitted

        fig.add_trace(a_trace, row=1, col=1)



    # Scatter Plot _______________________________________________________________________________________________________ #
    # adding scatter plot
    if out_scatter_fig is not None:
        for a_trace in out_scatter_fig.data:
            fig.add_trace(a_trace, row=1, col=2)
            # if forced_range_y is not None:
            #     fig.update_layout(yaxis=dict(range=forced_range_y))
    else:
        ## Create a new scatter plot:
        assert px_scatter_kwargs is not None
        out_scatter_fig = px.scatter(data_results_df, **px_scatter_kwargs)

        for i, a_trace in enumerate(out_scatter_fig.data):
            a_trace_name = a_trace.name
            if a_trace_name in already_added_legend_entries:
                # For already added trace categories, set showlegend to False
                a_trace.showlegend = False
            else:
                # For the first trace of each category, keep showlegend as True
                already_added_legend_entries.add(a_trace_name)
                a_trace.showlegend = True  # This is usually true by default, can be omitted

            # is_first_item: bool = (i == 0)
            # if (not is_first_item):
            #     a_trace['showlegend'] = False
                # a_trace.showlegend = False
            # print(f'a_trace: {a_trace}')
            # a_trace = fig.add_trace(a_trace, row=1, col=2)
            fig.add_trace(a_trace, row=1, col=2)

        # if forced_range_y is not None:
        #     fig.update_layout(yaxis=dict(range=forced_range_y))


    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    post_delta_fig = px.histogram(post_delta_df, y=histogram_variable_name, **common_plot_kwargs, **hist_kwargs, title="Post-delta")

    for a_trace in post_delta_fig.data:
        a_trace_name = a_trace.name
        if a_trace_name in already_added_legend_entries:
            # For already added trace categories, set showlegend to False
            a_trace.showlegend = False
        else:
            # For the first trace of each category, keep showlegend as True
            a_trace.showlegend = True  # This is usually true by default, can be omitted
            already_added_legend_entries.add(a_trace_name)

        fig.add_trace(a_trace, row=1, col=3)


    # fig.update_layout(yaxis=dict(range=forced_range_y))


    if forced_range_y is not None:
        fig.update_layout(yaxis=dict(range=forced_range_y))

    fig.update_layout(yaxis=dict(range=forced_range_y), barmode='overlay')


    # Epoch Shapes
    if time_delta_tuple is not None:
        assert len(time_delta_tuple) == 3
        earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = time_delta_tuple
        # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
        delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=2, t_start=delta_relative_t_start, t_split=delta_relative_t_delta, t_end=delta_relative_t_end, is_dark_mode=is_dark_mode)
    else:
        _extras_output_dict = {}


    # figure_context_dict['n_tbin'] = num_unique_time_bins

    figure_context = IdentifyingContext(**figure_context_dict)

    # Update layout to add a title to the legend
    legend_title_text = kwargs.get('legend_title_text', None)
    if legend_title_text is None:
        legend_key: str = px_scatter_kwargs.get('color', None)
        legend_title_text = legend_key

    if legend_title_text is not None:
        fig.update_layout(
            legend_title_text=legend_title_text  # Add a title to the legend
        )


    return fig, figure_context

@function_attributes(short_name=None, tags=['plotly', 'helper', 'epoch', 'track'], input_requires=[], output_provides=[], uses=[], used_by=['_helper_build_figure'], creation_date='2024-03-01 13:58', related_items=[])
def plotly_helper_add_epoch_shapes(fig, scatter_column_index: int, t_start: float, t_split:float, t_end: float, is_dark_mode: bool = True):
    """ adds shapes representing the epochs to the scatter plot at index scatter_column_index

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plotly_helper_add_epoch_shapes
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=scatter_column, t_start=earliest_delta_aligned_t_start, t_split=t_split, t_end=latest_delta_aligned_t_end)


    """
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

    _extras_output_dict = {}
    ## Get the track configs for the colors:
    long_short_display_config_manager = LongShortDisplayConfigManager()


    if is_dark_mode:
        long_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.long_epoch_config.mpl_color)
        short_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.short_epoch_config.mpl_color)
        y_zero_line_color = "rgba(0.2,0.2,0.2,.25)" # very dark grey
        vertical_epoch_divider_line_color = "rgba(0,0,0,.25)"
    else:
        long_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.long_epoch_config_light_mode.mpl_color)
        short_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.short_epoch_config_light_mode.mpl_color)
        y_zero_line_color = "rgba(0.8,0.8,0.8,.25)" # very light grey
        vertical_epoch_divider_line_color = "rgba(1,1,1,.25)" # white

    row_column_kwargs = dict(row='all', col=scatter_column_index)

    ## new methods
    _extras_output_dict["y_zero_line"] = fig.add_hline(y=0.0, line=dict(color=y_zero_line_color, width=9), **row_column_kwargs)
    vertical_divider_line = fig.add_vline(x=0.0, line=dict(color=vertical_epoch_divider_line_color, width=3, ), **row_column_kwargs)

    # fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)

    blue_shape = fig.add_vrect(x0=t_start, x1=t_split, label=dict(text="Long", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **long_epoch_kwargs, **row_column_kwargs) # , fillcolor="green", opacity=0.25
    red_shape = fig.add_vrect(x0=t_split, x1=t_end, label=dict(text="Short", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **short_epoch_kwargs, **row_column_kwargs)

    _extras_output_dict["long_region"] = blue_shape
    _extras_output_dict["short_region"] = red_shape
    _extras_output_dict["divider_line"] = vertical_divider_line
    return _extras_output_dict

@function_attributes(short_name=None, tags=['plotly', 'histogram'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-28 07:01', related_items=[])
def _helper_build_figure(data_results_df: pd.DataFrame, histogram_bins:int=25, earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session', is_dark_mode: bool=True,
                                          **build_fig_kwargs):
    """ factored out of the subfunction in plot_across_sessions_scatter_results
    adds scatterplots as well
    Captures: None
    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go

    figure_context_dict = {'main_plot_mode': main_plot_mode}

    # variable_name: str = 'P_Long'
    variable_name: str = 'P_Short'

    barmode='overlay'
    # barmode='stack'
    histogram_kwargs = dict(barmode=barmode)
    # px_histogram_kwargs = dict(nbins=histogram_bins, barmode='stack', opacity=0.5, range_y=[0.0, 1.0])
    scatter_title = build_fig_kwargs.pop('title', None)
    debug_print: bool = build_fig_kwargs.pop('debug_print', False)

    if scatter_title is not None:
        figure_context_dict['title'] = scatter_title

    # Filter dataframe by chosen bin sizes
    if (enabled_time_bin_sizes is not None) and (len(enabled_time_bin_sizes) > 0):
        print(f'filtering data_results_df to enabled_time_bin_sizes: {enabled_time_bin_sizes}...')
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(enabled_time_bin_sizes)]

    data_results_df = deepcopy(data_results_df)

    # convert time_bin_sizes column to a string so it isn't colored continuously
    data_results_df["time_bin_size"] = data_results_df["time_bin_size"].astype(str)


    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')

    ## Build KWARGS
    known_main_plot_modes = ['default', 'separate_facet_row_per_session', 'separate_row_per_session']
    assert main_plot_mode in known_main_plot_modes
    print(f'main_plot_mode: {main_plot_mode}')

    enable_histograms: bool = True
    enable_scatter_plot: bool = True
    enable_epoch_shading_shapes: bool = True
    px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0]} #, 'histnorm': 'probability density'

    if (main_plot_mode == 'default'):
        # main_plot_mode: str = 'default'
        enable_scatter_plot: bool = False
        num_cols: int = int(enable_scatter_plot) + 2 * int(enable_histograms) # 2 histograms and one scatter
        print(f'num_cols: {num_cols}')
        is_col_included = np.array([enable_histograms, enable_scatter_plot, enable_histograms])
        column_widths = list(np.array([0.1, 0.8, 0.1])[is_col_included])
        column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]

        # sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': column_titles}
        sp_make_subplots_kwargs = {'rows': 1, 'cols': num_cols, 'column_widths': column_widths, 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': list(np.array(column_titles)[is_col_included])}
        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color': 'session_name', 'size': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}

        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}

    elif (main_plot_mode == 'separate_facet_row_per_session'):
        # main_plot_mode: str = 'separate_facet_row_per_session'
        raise NotImplementedError(f"DOES NOT WORK")
        sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': ["Pre-delta",f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0],
                            'facet_row': 'session_name', 'facet_row_spacing': 0.04, # 'facet_col_wrap': 2, 'facet_col_spacing': 0.04,
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_histogram_kwargs = {**px_histogram_kwargs,
                                'facet_row': 'session_name', 'facet_row_spacing': 0.04, 'facet_col_wrap': 2, 'facet_col_spacing': 0.04, 'height': (num_unique_sessions*200), 'width': 1024}
        enable_histograms = False
        enable_epoch_shading_shapes = False

    elif (main_plot_mode == 'separate_row_per_session'):
        # main_plot_mode: str = 'separate_row_per_session'
        # , subplot_titles=("Plot 1", "Plot 2")
        # column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]
        column_titles = ["Pre-delta", f"{scatter_title}", "Post-delta"]
        session_titles = [str(v) for v in unique_sessions]
        subplot_titles = []
        for a_row_title in session_titles:
            subplot_titles.extend(["Pre-delta", f"{a_row_title}", "Post-delta"])
        # subplot_titles = [["Pre-delta", f"{a_row_title}", "Post-delta"] for a_row_title in session_titles].flatten()

        sp_make_subplots_kwargs = {'rows': num_unique_sessions, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'vertical_spacing': 0.04, 'shared_yaxes': True,
                                    'column_titles': column_titles,
                                    'row_titles': session_titles,
                                    'subplot_titles': subplot_titles,
                                    }
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color': 'time_bin_size', 'range_y': [0.0, 1.0],
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
    else:
        raise ValueError(f'main_plot_mode is not a known mode: main_plot_mode: "{main_plot_mode}", known modes: known_main_plot_modes: {known_main_plot_modes}')


    def __sub_subfn_plot_histogram(fig, histogram_data_df, hist_title="Post-delta", row=1, col=3):
        """ captures: px_histogram_kwargs, histogram_kwargs

        """
        is_first_item: bool = ((row == 1) and (col == 1))
        a_hist_fig = px.histogram(histogram_data_df, y="P_Long", color="time_bin_size", **px_histogram_kwargs, title=hist_title)

        for a_trace in a_hist_fig.data:
            if debug_print:
                print(f'a_trace.legend: {a_trace.legend}, a_trace.legendgroup: {a_trace.legendgroup}, a_trace.legendgrouptitle: {a_trace.legendgrouptitle}, a_trace.showlegend: {a_trace.showlegend}, a_trace.offsetgroup: {a_trace.offsetgroup}')

            if (not is_first_item):
                a_trace.showlegend = False

            fig.add_trace(a_trace, row=row, col=col)
            fig.update_layout(yaxis=dict(range=[0.0, 1.0]), **histogram_kwargs)


    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]
    # creating subplots
    fig = sp.make_subplots(**sp_make_subplots_kwargs)
    next_subplot_col_idx: int = 1

    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_pre_delta_df: pd.DataFrame = pre_delta_df[pre_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_pre_delta_df, hist_title="Pre-delta", row=row_index, col=histogram_col_idx)
                fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=1)

        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=pre_delta_df, hist_title="Pre-delta", row=1, col=histogram_col_idx)
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column

    # Scatter Plot _______________________________________________________________________________________________________ #
    if enable_scatter_plot:
        scatter_column: int = next_subplot_col_idx # default 2

        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):
                row_index: int = a_session_i + 1 # 1-indexed
                is_first_item: bool = ((row_index == 1) and (scatter_column == 1))
                a_session_data_results_df: pd.DataFrame = data_results_df[data_results_df['session_name'] == a_session_name]
                #  fig.add_scatter(x=a_session_data_results_df['delta_aligned_start_t'], y=a_session_data_results_df[variable_name], row=row_index, col=2, name=a_session_name)
                scatter_fig = px.scatter(a_session_data_results_df, **px_scatter_kwargs, title=f"{a_session_name}")
                for a_trace in scatter_fig.data:
                    if (not is_first_item):
                        a_trace.showlegend = False

                    fig.add_trace(a_trace, row=row_index, col=scatter_column)
                    # fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

                fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=row_index, col=scatter_column)
                #  fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

            #  fig.update_xaxes(matches='x')

        else:
            scatter_fig = px.scatter(data_results_df, **px_scatter_kwargs)

            # for a_trace in scatter_traces:
            for a_trace in scatter_fig.data:
                # a_trace.legend = "legend"
                # a_trace['visible'] = 'legendonly'
                # a_trace['visible'] = 'legendonly' # 'legendonly', # this trace will be hidden initially
                fig.add_trace(a_trace, row=1, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

            # Update xaxis properties
            fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=1, col=scatter_column)

        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column
    # else:
    #     # no scatter
    #     next_subplot_col_idx = next_subplot_col_idx


    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx #default 3

        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_post_delta_df: pd.DataFrame = post_delta_df[post_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_post_delta_df, hist_title="Post-delta", row=row_index, col=histogram_col_idx)
        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=post_delta_df, hist_title="Post-delta", row=1, col=histogram_col_idx)

        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column

    ## Add the delta indicator:
    if (enable_scatter_plot and enable_epoch_shading_shapes):

        t_split: float = 0.0
        #TODO 2024-02-02 04:36: - [ ] Should get the specific session t_start/t_end instead of using the general `earliest_delta_aligned_t_start`
        # _extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end, build_only=True)
        # for a_shape_name, a_shape in _extras_output_dict.items():
        #     if (main_plot_mode == 'separate_row_per_session'):
        #         for a_session_i, a_session_name in enumerate(unique_sessions):
        #             row_index: int = a_session_i + 1 # 1-indexed
        #             fig.add_shape(a_shape, name=a_shape_name, row=row_index, col=scatter_column)
        #     else:
        #         fig.add_shape(a_shape, name=a_shape_name, row=1, col=scatter_column)

        ## Inputs: fig, t_start: float, t_end: float
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=scatter_column, t_start=earliest_delta_aligned_t_start, t_split=t_split, t_end=latest_delta_aligned_t_end, is_dark_mode=is_dark_mode)


    # Update title and height

    if (main_plot_mode == 'separate_row_per_session'):
        row_height = 250
        required_figure_height = (num_unique_sessions*row_height)
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        row_height = 200
        required_figure_height = (num_unique_sessions*row_height)
    else:
        required_figure_height = 700

    fig.update_layout(title_text=scatter_title, width=2048, height=required_figure_height)
    fig.update_layout(yaxis=dict(range=[0.0, 1.0])) # , template='plotly_dark'
    # Update y-axis range for all created figures
    fig.update_yaxes(range=[0.0, 1.0])

    # Add a footer
    fig.update_layout(
        legend_title_text='tBin Size',
        # annotations=[
        #     dict(x=0.5, y=-0.15, showarrow=False, text="Footer text here", xref="paper", yref="paper")
        # ],
        # margin=dict(b=140), # increase bottom margin to show the footer
    )


    ## would be nice to export context as well:
    figure_context = IdentifyingContext(**figure_context_dict)

    return fig, figure_context

@function_attributes(short_name=None, tags=['plotly', 'blue_yellow'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plot_blue_yellow_points(a_df, specific_point_list):
    """ Renders a figure containing one or more yellow-blue plots (marginals) for a given hoverred point. Used with Dash app.

    specific_point_list: List[Dict] - specific_point_list = [{'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03, 'epoch_idx': 0, 'delta_aligned_start_t': -713.908702568122}]
    """
    time_window_centers_list = []
    posterior_list = []

    # for a_single_epoch_row_idx, a_single_epoch_idx in enumerate(selected_epoch_idxs):
    for a_single_epoch_row_idx, a_single_custom_data_dict in enumerate(specific_point_list):
        # a_single_epoch_idx = selected_epoch_idxs[a_single_epoch_row_idx]
        a_single_epoch_idx: int = int(a_single_custom_data_dict['epoch_idx'])
        a_single_session_name: str = str(a_single_custom_data_dict['session_name'])
        a_single_time_bin_size: float = float(a_single_custom_data_dict['time_bin_size'])
        ## Get the dataframe entries:
        a_single_epoch_df = a_df.copy()
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.epoch_idx == a_single_epoch_idx] ## filter by epoch idx
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.session_name == a_single_session_name] ## filter by session
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.time_bin_size == a_single_time_bin_size] ## filter by time-bin-size

        posterior = a_single_epoch_df[['P_Long', 'P_Short']].to_numpy().T
        time_window_centers = a_single_epoch_df['delta_aligned_start_t'].to_numpy()
        xbin = np.arange(2)
        time_window_centers_list.append(time_window_centers)
        posterior_list.append(posterior)

        # fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers=time_window_centers, xbin=xbin, posterior=posterior)
        # fig.show()

    fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list=time_window_centers_list, xbin=xbin, posterior_list=posterior_list)
    return fig


# def plotly_plot_1D_most_likely_position_comparsions(time_window_centers, xbin, posterior): # , ax=None
#     """
#     Analagous to `plot_1D_most_likely_position_comparsions`
#     """
#     import plotly.graph_objects as go

#     # Posterior distribution heatmap:
#     assert posterior is not None

#     # print(f'time_window_centers: {time_window_centers}, posterior: {posterior}')
#     # Compute extents
#     xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
#     # Create a heatmap
#     fig = go.Figure(data=go.Heatmap(
#                     z=posterior,
#                     x=time_window_centers,  y=xbin,
#                     zmin=0, zmax=1,
#                     # colorbar=dict(title='z'),
#                     showscale=False,
#                     colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
#                     hoverongaps = False))

#     # Update layout
#     fig.update_layout(
#         autosize=False,
#         xaxis=dict(type='linear', range=[xmin, xmax]),
#         yaxis=dict(type='linear', range=[ymin, ymax]))

#     return fig


@function_attributes(short_name=None, tags=['plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list, xbin, posterior_list): # , ax=None
    """
    Analagous to `plot_1D_most_likely_position_comparsions`
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp
    # Ensure input lists are of the same length
    assert len(time_window_centers_list) == len(posterior_list)

    # Compute layout grid dimensions
    num_rows = len(time_window_centers_list)

    # Create subplots
    fig = sp.make_subplots(rows=num_rows, cols=1)

    for row_idx, (time_window_centers, posterior) in enumerate(zip(time_window_centers_list, posterior_list)):
        # Compute extents
        xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
        # Add heatmap trace to subplot
        fig.add_trace(go.Heatmap(
                        z=posterior,
                        x=time_window_centers,  y=xbin,
                        zmin=0, zmax=1,
                        # colorbar=dict(title='z'),
                        showscale=False,
                        colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
                        hoverongaps = False),
                      row=row_idx+1, col=1)

        # Update layout for each subplot
        fig.update_xaxes(range=[xmin, xmax], row=row_idx+1, col=1)
        fig.update_yaxes(range=[ymin, ymax], row=row_idx+1, col=1)

    return fig


@function_attributes(short_name=None, tags=['Dash', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start: float, latest_delta_aligned_t_end: float):
    """ builds an interactive Across Sessions Dash app
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_dash_app

    app = _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end)
    """
    from dash import Dash, html,  dcc, callback, Output, Input
    from dash.dash_table import DataTable

    import dash_bootstrap_components as dbc
    import pandas as pd
    # import plotly.express as px
    import plotly.io as pio
    template: str = 'plotly_dark' # set plotl template
    pio.templates.default = template


    ## DATA:
    options_list = list(final_dfs_dict.keys())
    initial_option = options_list[0]
    initial_dataframe: pd.DataFrame = final_dfs_dict[initial_option].copy()
    unique_sessions: List[str] = initial_dataframe['session_name'].unique().tolist()
    num_unique_sessions: int = initial_dataframe['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    assert 'epoch_idx' in initial_dataframe.columns

    ## Extract the unique time bin sizes:
    time_bin_sizes: List[float] = initial_dataframe['time_bin_size'].unique().tolist()
    num_unique_time_bins: int = initial_dataframe.time_bin_size.nunique(dropna=True)
    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
    enabled_time_bin_sizes = [time_bin_sizes[0], time_bin_sizes[-1]] # [0.03, 0.058, 0.10]

    ## prune to relevent columns:
    all_column_names = [
        ['P_Long', 'P_Short', 'P_LR', 'P_RL'],
        ['delta_aligned_start_t'], # 'lap_idx',
        ['session_name'],
        ['time_bin_size'],
        ['epoch_idx'],
    ]
    all_column_names_flat = [item for sublist in all_column_names for item in sublist]
    print(f'\tall_column_names_flat: {all_column_names_flat}')
    initial_dataframe = initial_dataframe[all_column_names_flat]

    # Initialize the app
    # app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    # Slate

    # # money = FormatTemplate.money(2)
    # percentage = FormatTemplate.percentage(2)
    # # percentage = FormatTemplate.deci
    # column_designators = [
    #     dict(id='a', name='delta_aligned_start_t', type='numeric', format=Format()),
    #     dict(id='a', name='session_name', type='text', format=Format()),
    #     dict(id='a', name='time_bin_size', type='numeric', format=Format(padding=Padding.yes).padding_width(9)),
    #     dict(id='a', name='P_Long', type='numeric', format=dict(specifier='05')),
    #     dict(id='a', name='P_LR', type='numeric', format=dict(specifier='05')),
    # ]

    # App layout
    app.layout = dbc.Container([
        dbc.Row([
                html.Div(children='My Custom App with Data, Graph, and Controls'),
                html.Hr()
        ]),
        dbc.Row([
            dbc.Col(dcc.RadioItems(options=options_list, value=initial_option, id='controls-and-radio-item'), width=3),
            dbc.Col(dcc.Checklist(options=time_bin_sizes, value=enabled_time_bin_sizes, id='time-bin-checkboxes', inline=True), width=3), # Add CheckboxGroup for time_bin_sizes
        ]),
        dbc.Row([
            dbc.Col(DataTable(data=initial_dataframe.to_dict('records'), page_size=16, id='tbl-datatable',
                        # columns=column_designators,
                        columns=[{"name": i, "id": i} for i in initial_dataframe.columns],
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': 'rgb(70, 70, 70)',
                                'color': 'white'
                            },
                            {
                                'if': {'column_editable': True},
                                'backgroundColor': 'rgb(100, 100, 100)',
                                'color': 'white'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(30, 30, 30)',
                            'color': 'white'
                        },
                        row_selectable="multi",
                ) # end DataTable
            , align='stretch', width=3),
            dbc.Col(dcc.Graph(figure={}, id='controls-and-graph', hoverData={'points': [{'customdata': []}]},
                            ), align='end', width=9),
        ]), # end Row
        dbc.Row(dcc.Graph(figure={}, id='selected-yellow-blue-marginals-graph')),
    ]) # end Container

    # Add controls to build the interaction
    @callback(
        Output(component_id='controls-and-graph', component_property='figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_graph(col_chosen, chose_bin_sizes):
        print(f'update_graph(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        # Filter dataframe by chosen bin sizes
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)]

        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes
        fig, figure_context = _helper_build_figure(data_results_df=data_results_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode='separate_row_per_session', title=f"{col_chosen}")
        # 'delta_aligned_start_t', 'session_name', 'time_bin_size'
        tuples_data = data_results_df[['session_name', 'time_bin_size', 'epoch_idx', 'delta_aligned_start_t']].to_dict(orient='records')
        print(f'tuples_data: {tuples_data}')
        fig.update_traces(customdata=tuples_data)
        fig.update_layout(hovermode='closest') # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        return fig


    @callback(
        Output(component_id='tbl-datatable', component_property='data'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
            Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_datatable(col_chosen, chose_bin_sizes):
        """ captures: final_dfs_dict, all_column_names_flat
        """
        print(f'update_datatable(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        a_df = final_dfs_dict[col_chosen].copy()
        ## prune to relevent columns:
        a_df = a_df[all_column_names_flat]
        # Filter dataframe by chosen bin sizes
        a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        data = a_df.to_dict('records')
        return data

    @callback(
        Output('selected-yellow-blue-marginals-graph', 'figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        Input(component_id='tbl-datatable', component_property='selected_rows'),
        Input(component_id='controls-and-graph', component_property='hoverData'),
        ]
    )
    def get_selected_rows(col_chosen, chose_bin_sizes, indices, hoverred_rows):
        print(f'get_selected_rows(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes}, indices: {indices}, hoverred_rows: {hoverred_rows})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)] # Filter dataframe by chosen bin sizes
        # ## prune to relevent columns:
        data_results_df = data_results_df[all_column_names_flat]

        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        # print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes

        print(f'hoverred_rows: {hoverred_rows}')
        # get_selected_rows(col_chosen: AcrossSession_Laps_per-Epoch, chose_bin_sizes: [0.03, 0.1], indices: None, hoverred_rows: {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]})
        # hoverred_rows:
        hoverred_row_points = hoverred_rows.get('points', [])
        num_hoverred_points: int = len(hoverred_row_points)
        extracted_custom_data = [p['customdata'] for p in hoverred_row_points if (p.get('customdata', None) is not None)] # {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}
        num_custom_data_hoverred_points: int = len(extracted_custom_data)

        print(f'extracted_custom_data: {extracted_custom_data}')
        # {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]}
            # selection empty!

        # a_df = final_dfs_dict[col_chosen].copy()
        # ## prune to relevent columns:
        # a_df = a_df[all_column_names_flat]
        # # Filter dataframe by chosen bin sizes
        # a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        # data = a_df.to_dict('records')
        if (indices is not None) and (len(indices) > 0):
            selected_rows = data_results_df.iloc[indices, :]
            print(f'\tselected_rows: {selected_rows}')
        else:
            print(f'\tselection empty!')

        if (extracted_custom_data is not None) and (num_custom_data_hoverred_points > 0):
            # selected_rows = data_results_df.iloc[indices, :]
            print(f'\tnum_custom_data_hoverred_points: {num_custom_data_hoverred_points}')
            fig = plot_blue_yellow_points(a_df=data_results_df.copy(), specific_point_list=extracted_custom_data)
        else:
            print(f'\thoverred points empty!')
            fig = go.Figure()

        return fig

    return app

@function_attributes(short_name=None, tags=['scatter', 'multi-session', 'plot', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_across_sessions_scatter_results(directory: Union[Path, str], concatenated_laps_df: pd.DataFrame, concatenated_ripple_df: pd.DataFrame,
                                          earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          laps_title_prefix: str = f"Laps", ripple_title_prefix: str = f"Ripples",
                                          save_figures=False, figure_save_extension='.png', is_dark_mode:bool=True, debug_print=False):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`

    - Processes both ripple and laps
    - generates a single column of plots with the scatter plot in the middle flanked on both sides by the Pre/Post-delta histograms


    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    Unknowingly captured: session_name

    - [ ] Truncate each session to their start/end instead of the global x bounds.


    """
    from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    # import plotly.graph_objs as go

    # def _subfn_build_figure(data, **build_fig_kwargs):
    #     return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))

    # def _subfn_build_figure(data_results_df: pd.DataFrame, **build_fig_kwargs):
    #     # return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    #     scatter_title = build_fig_kwargs.pop('title', None)
    #     return go.Figure(px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title), layout_yaxis_range=[0.0, 1.0])

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    if not isinstance(directory, Path):
        directory = Path(directory).resolve()
    assert directory.exists()
    print(f'plot_across_sessions_results(directory: {directory})')
    if save_figures:
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder = Path(directory, 'figures')
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: {file_uri_from_path(figures_folder)}')

    # Create an empty list to store the figures
    all_figures = []

    ## delta_t aligned:
    # Create a bubble chart for laps
    laps_num_unique_sessions: int = concatenated_laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    laps_num_unique_time_bins: int = concatenated_laps_df.time_bin_size.nunique(dropna=True)
    laps_title_string_suffix: str = f'{laps_num_unique_sessions} Sessions'
    laps_title: str = f"{laps_title_prefix} - {laps_title_string_suffix}"
    fig_laps, figure_laps_context = _helper_build_figure(data_results_df=concatenated_laps_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=laps_title, is_dark_mode=is_dark_mode)

    # Create a bubble chart for ripples
    ripple_num_unique_sessions: int = concatenated_ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    ripple_num_unique_time_bins: int = concatenated_ripple_df.time_bin_size.nunique(dropna=True)
    ripple_title_string_suffix: str = f'{ripple_num_unique_sessions} Sessions'
    ripple_title: str = f"{ripple_title_prefix} - {ripple_title_string_suffix}"
    fig_ripples, figure_ripples_context = _helper_build_figure(data_results_df=concatenated_ripple_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=ripple_title, is_dark_mode=is_dark_mode)

    if save_figures:
        # Save the figures to the 'figures' subfolder
        assert figure_save_extension is not None
        if isinstance(figure_save_extension, str):
             figure_save_extension = [figure_save_extension] # a list containing only this item

        print(f'\tsaving figures...')
        for a_fig_save_extension in figure_save_extension:
            if a_fig_save_extension.lower() == '.html':
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_html(a_save_name)
            else:
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_image(a_save_name)

            fig_laps_name = Path(figures_folder, f"{laps_title_string_suffix.replace(' ', '-')}_{laps_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_laps_name)}"...')
            a_save_fn(fig_laps, fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_{ripple_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_ripple_name)}"...')
            a_save_fn(fig_ripples, fig_ripple_name)


    # Append both figures to the list
    all_figures.append((fig_laps, fig_ripples))

    return all_figures
