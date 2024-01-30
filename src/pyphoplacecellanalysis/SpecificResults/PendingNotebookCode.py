# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import  List, Optional, Dict, Tuple, Any, Union
import numpy as np
import pandas as pd
from attrs import define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes


# ==================================================================================================================== #
# 2024-01-29 - Ideal Pho Plotting Interface - UNFINISHED                                                               #
# ==================================================================================================================== #
def map_dataframe_to_plot(df: pd.DataFrame, **kwargs):
	""" 2024-01-29 - My ideal desired function that allows the user to map any column in a dataframe to a plot command, including rows/columns.
	Not yet finished.
	 maps any column in the dataframe to a property in a plot. 
	 
	 Usage:
	 	fully_resolved_kwargs = map_dataframe_to_plot(df=all_sessions_laps_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size') # , title=f"Laps - {laps_title_string_suffix}"
		fully_resolved_kwargs

	"""
	all_column_names: List[str] = list(df.columns)
	all_kwargs_keys: List[str] = list(kwargs.keys())
	all_kwargs_values: List[Union[str, Any]] = list(kwargs.values()) # expected to be either a column name to map or a literal.
	num_rows: int = len(df)
	
	should_fully_extract_dataframe_values: bool = True # if True, extracts the values from the dataframe as an array
	fully_resolved_kwargs = {}
	
	# for a_key in all_kwargs_keys:
	# 	assert a_key in df.columns, f'key "{a_key}" specified in kwargs is not a column in df! \n\tdf.columns: {list(df.columns)}'
	known_keys = ['x', 'y', 'color', 'size', 'row', 'column', 'page', 'xlabel', 'ylabel', 'title']
	for a_key, a_value in kwargs.items():
		if a_key not in known_keys:
			print(f'WARN: key "{a_key}" is not in the known keys list: known_keys: {known_keys}')
		if not isinstance(a_value, str):
			# not a string
			raise ValueError(f"value {a_value} is not a string and its length is not equal to the length of the dataframe.")
			#TODO 2024-01-29 23:45: - [ ] Allow passing literal list-like values with the correct length to be passed directly
			assert (len(a_value) == num_rows), f"(len(a_value) == num_rows) but (len(a_value): {len(a_value)} == num_rows: {num_rows})"
			fully_resolved_kwargs[a_key] = a_value # Set the passed value directly
			
		else:
			# it is a string, assume that it's a column in the dataframe
			assert a_value in all_column_names, f'key:value pair <"{a_key}":"{a_value}"> specified in kwargs has a value that is not a valid column in df! \n\tspecified_value: {a_value}\n\tdf.columns: {list(df.columns)}'
			if should_fully_extract_dataframe_values:
				fully_resolved_kwargs[a_key] = df[a_value].to_numpy()
			else:
				# leave as the validated column name
				fully_resolved_kwargs[a_key] = a_value
				
	return fully_resolved_kwargs


def _embed_in_subplots(scatter_fig):
    import plotly.subplots as sp
    import plotly.graph_objs as go
    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01)

    # adding first histogram
    # Calculate the histogram data
    hist1, bins1 = np.histogram(X[:split], bins='auto')

    # Adding the first histogram as a bar graph and making x negative
    fig.add_trace(
        go.Bar(
            x=-bins1[:-1],
            y=hist1,
            marker_color='#EB89B5',
            name='first half',
            orientation='h',
        ),
        row=1, col=1
    )


    # adding scatter plot
    fig.add_trace(scatter_fig, row=1, col=2)
    # fig.add_trace(
    #     go.Scatter(
    #         x=X,
    #         y=Y,
    #         mode='markers',
    #         marker_color='rgba(152, 0, 0, .8)',
    #     ),
    #     row=1, col=2
    # )

    # adding the second histogram

    # Calculate the histogram data for second half
    hist2, bins2 = np.histogram(X[split:], bins='auto')

    # Adding the second histogram
    fig.add_trace(
        go.Bar(
            x=bins2[:-1],
            y=hist2,
            marker_color='#330C73',
            name='second half',
            orientation='h',
        ),
        row=1, col=3
    )
    return fig


# ==================================================================================================================== #
# 2024-01-29 - Across Session CSV Import and Plotting                                                                  #
# ==================================================================================================================== #
""" 

from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_across_sessions_scatter_results, plot_histograms, plot_stacked_histograms

"""

import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go


def complete_plotly_figure(data_results_df: pd.DataFrame, out_scatter_fig, histogram_bins:int=25):
    """ 
    Usage:

        histogram_bins: int = 25

        new_laps_fig = complete_plotly_figure(data_results_df=deepcopy(all_sessions_laps_df), out_scatter_fig=fig_laps, histogram_bins=histogram_bins)
        new_laps_fig

    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go

    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')


    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    # X_all = data_results_df['delta_aligned_start_t'].to_numpy()
    # Y_all = data_results_df['P_Long'].to_numpy()

    # X_pre_delta = pre_delta_df['delta_aligned_start_t'].to_numpy()
    # X_post_delta = post_delta_df['delta_aligned_start_t'].to_numpy()

    # Y_pre_delta = pre_delta_df['P_Long'].to_numpy()
    # Y_post_delta = post_delta_df['P_Long'].to_numpy()

    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01, shared_yaxes=True, column_titles=["Pre-delta",f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"])

    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    pre_delta_fig = px.histogram(pre_delta_df, y="P_Long", color="time_bin_size", opacity=0.5, title="Pre-delta", range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay')
    print(f'len(pre_delta_fig.data): {len(pre_delta_fig.data)}')
    # time_bin_sizes
    for a_trace in pre_delta_fig.data:
        fig.add_trace(a_trace, row=1, col=1)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

    # Calculate the histogram data
    # hist1, bins1 = np.histogram(X_pre_delta, bins=histogram_bins)

    # # Adding the first histogram as a bar graph and making x negative
    # fig.add_trace(
    #     # go.Bar(x=bins1[:-1], y=hist1, marker_color='#EB89B5', name='first half', orientation='h', ),
    # 	go.Histogram(y=Y_pre_delta, name='pre-delta', marker_color='#EB89B5'),
    #     row=1, col=1
    # )
    # fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

    # Scatter Plot _______________________________________________________________________________________________________ #
    # adding scatter plot
    for a_trace in out_scatter_fig.data:
        fig.add_trace(a_trace, row=1, col=2)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))


    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    post_delta_fig = px.histogram(post_delta_df, y="P_Long", color="time_bin_size", opacity=0.5, title="Post-delta", range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay')

    for a_trace in post_delta_fig.data:
        fig.add_trace(a_trace, row=1, col=3)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
        
    # Calculate the histogram data for second half
    # hist2, bins2 = np.histogram(X_post_delta, bins=histogram_bins)
    # Adding the second histogram
    # fig.add_trace(
    # 	go.Histogram(y=Y_post_delta, name='post-delta', marker_color='#EB89B5',),
    #     # go.Bar(x=bins2[:-1], y=hist2, marker_color='#330C73', name='second half', orientation='h', ),
    #     row=1, col=3
    # )

    # fig.update_layout(layout_yaxis_range=[0.0, 1.0])
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]), barmode='overlay')
    return fig


def _helper_build_figure(data_results_df: pd.DataFrame, histogram_bins:int=25, earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          **build_fig_kwargs):
    """ factored out of the subfunction in plot_across_sessions_scatter_results
    adds scatterplots as well
    Captures: None 
    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    
    barmode='overlay'
    # barmode='stack'
    histogram_kwargs = dict(barmode=barmode)
    # px_histogram_kwargs = dict(nbins=histogram_bins, barmode='stack', opacity=0.5, range_y=[0.0, 1.0])
    scatter_title = build_fig_kwargs.pop('title', None)
    debug_print: bool = build_fig_kwargs.pop('debug_print', False)
    
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
        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'session_name', 'size': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
        
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        # main_plot_mode: str = 'separate_facet_row_per_session'
        sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': ["Pre-delta",f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0],
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
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'range_y': [0.0, 1.0],
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
                #  fig.add_scatter(x=a_session_data_results_df['delta_aligned_start_t'], y=a_session_data_results_df['P_Long'], row=row_index, col=2, name=a_session_name)
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
            # scatter_fig = px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title, range_y=[0.0, 1.0], labels={"session_name": "Session", "time_bin_size": "tbin_size"})
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
        _extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end, build_only=True)
        for a_shape_name, a_shape in _extras_output_dict.items():
            if (main_plot_mode == 'separate_row_per_session'):
                for a_session_i, a_session_name in enumerate(unique_sessions):    
                    row_index: int = a_session_i + 1 # 1-indexed
                    fig.add_shape(a_shape, name=a_shape_name, row=row_index, col=scatter_column)
            else:
                fig.add_shape(a_shape, name=a_shape_name, row=1, col=scatter_column)
    
    # Update title and height
    if (main_plot_mode == 'separate_row_per_session'):
        required_figure_height = (num_unique_sessions*300)
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        required_figure_height = (num_unique_sessions*200)
    else:
        required_figure_height = 700
        
    fig.update_layout(title_text=scatter_title, width=2048, height=required_figure_height)
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]), template='plotly_dark')
    # Update y-axis range for all created figures
    fig.update_yaxes(range=[0.0, 1.0])
    return fig

    


@function_attributes(short_name=None, tags=['scatter', 'multi-session', 'plot', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_across_sessions_scatter_results(directory, concatenated_laps_df, concatenated_ripple_df,
                                          earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          laps_title_prefix: str = f"Laps", ripple_title_prefix: str = f"Ripples",
                                          save_figures=False, figure_save_extension='.png', debug_print=False):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    Unknowingly captured: session_name
    
    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    # import plotly.graph_objs as go
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers

    
    

    # def _subfn_build_figure(data, **build_fig_kwargs):
    #     return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    
    # def _subfn_build_figure(data_results_df: pd.DataFrame, **build_fig_kwargs):
    #     # return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    #     scatter_title = build_fig_kwargs.pop('title', None) 
    #     return go.Figure(px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title), layout_yaxis_range=[0.0, 1.0])

    def _subfn_build_figure(data_results_df: pd.DataFrame, histogram_bins:int=25, **build_fig_kwargs):
        """ adds scatterplots as well
        Captures: earliest_delta_aligned_t_start, latest_delta_aligned_t_end, enabled_time_bin_sizes
        """
        barmode='overlay'
        # barmode='stack'
        histogram_kwargs = dict(barmode=barmode)
        # px_histogram_kwargs = dict(nbins=histogram_bins, barmode='stack', opacity=0.5, range_y=[0.0, 1.0])
        scatter_title = build_fig_kwargs.pop('title', None)

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
            # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'session_name', 'size': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
            px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
            
            # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
            
        elif (main_plot_mode == 'separate_facet_row_per_session'):
            # main_plot_mode: str = 'separate_facet_row_per_session'
            sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': ["Pre-delta",f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]}
            px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0],
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
            px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'range_y': [0.0, 1.0],
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
                    #  fig.add_scatter(x=a_session_data_results_df['delta_aligned_start_t'], y=a_session_data_results_df['P_Long'], row=row_index, col=2, name=a_session_name)
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
                # scatter_fig = px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title, range_y=[0.0, 1.0], labels={"session_name": "Session", "time_bin_size": "tbin_size"})
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
            _extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end, build_only=True)
            for a_shape_name, a_shape in _extras_output_dict.items():
                if (main_plot_mode == 'separate_row_per_session'):
                    for a_session_i, a_session_name in enumerate(unique_sessions):    
                        row_index: int = a_session_i + 1 # 1-indexed
                        fig.add_shape(a_shape, name=a_shape_name, row=row_index, col=scatter_column)
                else:
                    fig.add_shape(a_shape, name=a_shape_name, row=1, col=scatter_column)
        
        # Update title and height
        if (main_plot_mode == 'separate_row_per_session'):
            required_figure_height = (num_unique_sessions*300)
        elif (main_plot_mode == 'separate_facet_row_per_session'):
            required_figure_height = (num_unique_sessions*200)
        else:
            required_figure_height = 700
            
        fig.update_layout(title_text=scatter_title, width=2048, height=required_figure_height)
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]), template='plotly_dark')
        # Update y-axis range for all created figures
        fig.update_yaxes(range=[0.0, 1.0])
        return fig

    
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
        print(f'\tfigures_folder: {figures_folder}')
    
    # Create an empty list to store the figures
    all_figures = []

    ## delta_t aligned:
    # Create a bubble chart for laps
    laps_num_unique_sessions: int = concatenated_laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    laps_num_unique_time_bins: int = concatenated_laps_df.time_bin_size.nunique(dropna=True)
    laps_title_string_suffix: str = f'{laps_num_unique_sessions} Sessions'
    laps_title: str = f"{laps_title_prefix} - {laps_title_string_suffix}"
    # fig_laps = go.Figure(px.scatter(concatenated_laps_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=laps_title), layout_yaxis_range=[0.0, 1.0])
    # fig_laps = _subfn_build_figure(data_results_df=concatenated_laps_df, title=laps_title)
    fig_laps = _helper_build_figure(data_results_df=concatenated_laps_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=laps_title)

    # Create a bubble chart for ripples
    ripple_num_unique_sessions: int = concatenated_ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    ripple_num_unique_time_bins: int = concatenated_ripple_df.time_bin_size.nunique(dropna=True)
    ripple_title_string_suffix: str = f'{ripple_num_unique_sessions} Sessions'
    ripple_title: str = f"{ripple_title_prefix} - {ripple_title_string_suffix}"
    # fig_ripples = _subfn_build_figure(data_results_df=concatenated_ripple_df, title=ripple_title)
    fig_ripples = _helper_build_figure(data_results_df=concatenated_ripple_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=ripple_title)

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
            print(f'\tsaving "{fig_laps_name}"...')
            a_save_fn(fig_laps, fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_{ripple_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{fig_ripple_name}"...')
            a_save_fn(fig_ripples, fig_ripple_name)
            

    # Append both figures to the list
    all_figures.append((fig_laps, fig_ripples))
    
    return all_figures



@function_attributes(short_name=None, tags=['histogram', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, **kwargs) -> None:
    """ plots a set of two histograms in subplots, split at the delta for each session.
    from PendingNotebookCode import plot_histograms
    
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    
    fig = plt.figure(layout=layout, **kwargs) # layout="constrained", 
    ax_dict = fig.subplot_mosaic(
        [
            ["epochs_pre_delta", ".", "epochs_post_delta"],
        ],
        # set the height ratios between the rows
        # height_ratios=[8, 1],
        # height_ratios=[1, 1],
        # set the width ratios between the columns
        # width_ratios=[1, 8, 8, 1],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )

    histogram_kwargs = dict(orientation="horizontal", bins=25)
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    pre_delta_df.hist(ax=ax_dict['epochs_pre_delta'], column='P_Long', **histogram_kwargs)
    ax_dict['epochs_pre_delta'].set_title(f'{descriptor_str} - pre-$\Delta$ time bins')

    # plot post-delta histogram
    post_delta_df.hist(ax=ax_dict['epochs_post_delta'], column='P_Long', **histogram_kwargs)
    ax_dict['epochs_post_delta'].set_title(f'{descriptor_str} - post-$\Delta$ time bins')
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_histograms', figures=[fig], axes=ax_dict)


@function_attributes(short_name=None, tags=['histogram', 'stacked', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_stacked_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, **kwargs) -> None:
    """ plots a colorful stacked histogram for each of the many time-bin sizes
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    figure_identifier: str = f"{descriptor_str}_PrePostDelta"

    fig = plt.figure(num=figure_identifier, clear=True, figsize=(12, 2), layout=layout, **kwargs) # layout="constrained", 
    fig.suptitle(f'{descriptor_str}')
    
    ax_dict = fig.subplot_mosaic(
        [
            # ["epochs_pre_delta", ".", "epochs_post_delta"],
             ["epochs_pre_delta", "epochs_post_delta"],
        ],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )
    
    histogram_kwargs = dict(orientation="horizontal", bins=25)
    
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    time_bin_sizes: int = pre_delta_df['time_bin_size'].unique()
    
    # plot pre-delta histogram:
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_pre_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_pre_delta'].set_title(f'pre-$\Delta$ time bins')
    ax_dict['epochs_pre_delta'].legend()

    # plot post-delta histogram:
    time_bin_sizes: int = post_delta_df['time_bin_size'].unique()
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_post_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_post_delta'].set_title(f'post-$\Delta$ time bins')
    ax_dict['epochs_post_delta'].legend()
    
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_stacked_histograms', figures=[fig], axes=ax_dict)





# Plot the time_bin marginals:

# def plot_across_sessions_results_with_histogram_gpt3(directory, concatenated_laps_df, concatenated_ripple_df, save_figures=False, figure_save_extension='.png'):
#     """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
#     Produces and then saves figures out the the f'{directory}/figures/' subfolder

#     """
#     if not isinstance(directory, Path):
#         directory = Path(directory).resolve()
#     assert directory.exists()
#     print(f'plot_across_sessions_results(directory: {directory})')
#     if save_figures:
#         # Create a 'figures' subfolder if it doesn't exist
#         figures_folder = Path(directory, 'figures')
#         figures_folder.mkdir(parents=False, exist_ok=True)
#         assert figures_folder.exists()
#         print(f'\tfigures_folder: {figures_folder}')
    
#     # Create an empty list to store the figures
#     all_figures = []

#     ## delta_t aligned:
#     # Create a bubble chart for laps
#     fig_laps = px.scatter(concatenated_laps_df, x='delta_aligned_start_t', y='P_Long', title=f"Laps - Session: {session_name}", color='session_name')
#     # Create a bubble chart for ripples
#     fig_ripples = px.scatter(concatenated_ripple_df, x='delta_aligned_start_t', y='P_Long', title=f"Ripples - Session: {session_name}", color='session_name')

#     # Create a histogram for laps
#     fig_hist_laps = px.histogram(concatenated_laps_df, x='delta_aligned_start_t', nbins=50, title=f"Laps - Session: {session_name}")
    
#     # Assign numerical values to session_name for color
#     session_name_to_color = {name: i for i, name in enumerate(concatenated_laps_df['session_name'].unique())}

#     # Create subplots with shared y-axis
#     fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Laps - Session: {session_name}", f"Ripples - Session: {session_name}"])
    
#     # Add histogram to the left subplot
#     fig.add_trace(go.Histogram(x=concatenated_laps_df['delta_aligned_start_t'], nbinsx=50, name='Histogram'), row=1, col=1)
#     fig.update_yaxes(title_text='Count', row=1, col=1)
    
#     # Add bubble chart to the right subplot
#     fig.add_trace(go.Scatter(x=concatenated_laps_df['delta_aligned_start_t'], y=concatenated_laps_df['P_Long'], mode='markers', marker=dict(color=concatenated_laps_df['session_name'].map(session_name_to_color))), row=1, col=2)
#     fig.update_xaxes(title_text='delta_aligned_start_t', row=1, col=2)
#     fig.update_yaxes(title_text='P_Long', row=1, col=2)

#     if save_figures:
#         # Save the figure to the 'figures' subfolder
#         print(f'\tsaving figures...')
#         fig_name = Path(figures_folder, f"{session_name}_combined_plot{figure_save_extension}").resolve()
#         print(f'\tsaving "{fig_name}"...')
#         fig.write_image(fig_name)
    
#     # Append the figure to the list
#     all_figures.append(fig)
    
#     return all_figures


# def plot_across_sessions_results_with_histogram_new(directory, concatenated_laps_df, concatenated_ripple_df, save_figures=False, figure_save_extension='.png'):
    # """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    # Produces and then saves figures out the the f'{directory}/figures/' subfolder

    # """

    # # Your existing code (not modified)

    # ## delta_t aligned:
    # # Create subplot with 2 rows and 1 column
    # fig_laps = make_subplots(rows=2, cols=1)
    # # Add scatter plot to first row, first column
    # fig_laps.add_trace(
    #     go.Scatter(x=concatenated_laps_df['delta_aligned_start_t'], y=concatenated_laps_df['P_Long'], mode='markers', name='Scatter'), 
    #     row=1, col=1
    # )
    # # add histogram to second row, first column
    # fig_laps.add_trace(
    #     go.Histogram(x=concatenated_laps_df['delta_aligned_start_t'], name='Histogram'), 
    #     row=2, col=1
    # )
    # # Same for ripples
    # fig_ripples = make_subplots(rows=2, cols=1)
    # fig_ripples.add_trace(
    #     go.Scatter(x=concatenated_ripple_df['delta_aligned_start_t'], y=concatenated_ripple_df['P_Long'], mode='markers', name='Scatter'), 
    #     row=1, col=1
    # )
    # fig_ripples.add_trace(
    #     go.Histogram(x=concatenated_ripple_df['delta_aligned_start_t'], name='Histogram'), 
    #     row=2, col=1
    # )
    # # Your existing code continues from here (not modified)
    # if not isinstance(directory, Path):
    #     directory = Path(directory).resolve()
    # assert directory.exists()
    # print(f'plot_across_sessions_results(directory: {directory})')
    # if save_figures:
    #     # Create a 'figures' subfolder if it doesn't exist
    #     figures_folder = Path(directory, 'figures')
    #     figures_folder.mkdir(parents=False, exist_ok=True)
    #     assert figures_folder.exists()
    #     print(f'\tfigures_folder: {figures_folder}')
    
    # # Create an empty list to store the figures
    # all_figures = []

    # ## delta_t aligned:
    # # Create a bubble chart for laps
    # fig_laps = px.scatter(concatenated_laps_df, x='delta_aligned_start_t', y='P_Long', title=f"Laps - Session: {session_name}", color='session_name')
    # # Create a bubble chart for ripples
    # fig_ripples = px.scatter(concatenated_ripple_df, x='delta_aligned_start_t', y='P_Long', title=f"Ripples - Session: {session_name}", color='session_name')

    # if save_figures:
    #     # Save the figures to the 'figures' subfolder
    #     print(f'\tsaving figures...')
    #     fig_laps_name = Path(figures_folder, f"{session_name}_laps_marginal{figure_save_extension}").resolve()
    #     print(f'\tsaving "{fig_laps_name}"...')
    #     fig_laps.write_image(fig_laps_name)
    #     fig_ripple_name = Path(figures_folder, f"{session_name}_ripples_marginal{figure_save_extension}").resolve()
    #     print(f'\tsaving "{fig_ripple_name}"...')
    #     fig_ripples.write_image(fig_ripple_name)
    
    # # Append both figures to the list
    # all_figures.append((fig_laps, fig_ripples))
    
    # return all_figures
    

# ==================================================================================================================== #
# 2024-01-27 - Across Session CSV Import and Processing                                                                #
# ==================================================================================================================== #
""" 
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import find_csv_files, find_HDF5_files, find_most_recent_files, process_csv_file

"""
def find_csv_files(directory: str, recurrsive: bool=False):
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    

def find_HDF5_files(directory: str):
    directory_path = Path(directory) # Convert string path to a Path object
    return list(directory_path.glob('**/*.h5')) # Return a list of all .h5 files in the directory and its subdirectories


def parse_filename(path: Path, debug_print:bool=False) -> Tuple[datetime, str, str]:
    """ 
    # from the found_session_export_paths, get the most recently exported laps_csv, ripple_csv (by comparing `export_datetime`) for each session (`session_str`)
    a_export_filename: str = "2024-01-12_0420PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv"
    export_datetime = "2024-01-12_0420PM"
    session_str = "kdiba_pin01_one_fet11-01_12-58-54"
    export_file_type = "(laps_marginals_df)" # .csv

    # return laps_csv, ripple_csv
    laps_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv").resolve()
    ripple_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(ripple_marginals_df).csv").resolve()

    """
    filename = path.stem   # Get filename without extension
    decoding_time_bin_size_str = None
    
    pattern = r"(?P<export_datetime_str>.*_\d{2}\d{2}[APMF]{2})-(?P<session_str>.*)-(?P<export_file_type>\(?.+\)?)(?:_tbin-(?P<decoding_time_bin_size_str>[^)]+))"
    match = re.match(pattern, filename)
    
    if match is not None:
        # export_datetime_str, session_str, export_file_type = match.groups()
        export_datetime_str, session_str, export_file_type, decoding_time_bin_size_str = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type'), match.group('decoding_time_bin_size_str')
    
        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d_%I%M%p")

    else:
        if debug_print:
            print(f'did not match pattern with time.')
        # day_date_only_pattern = r"(.*(?:_\d{2}\d{2}[APMF]{2})?)-(.*)-(\(.+\))"
        day_date_only_pattern = r"(\d{4}-\d{2}-\d{2})-(.*)-(\(?.+\)?)" # 
        day_date_only_match = re.match(day_date_only_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if day_date_only_match is not None:
            export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
            # print(export_datetime_str, session_str, export_file_type)
            # parse the datetime from the export_datetime_str and convert it to datetime object
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        
        else:
            # Try H5 pattern:
            # matches '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
            day_date_with_variant_suffix_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})_?(?P<variant_suffix>[^-_]*)-(?P<session_str>.+?)_(?P<export_file_type>[A-Za-z_]+)"
            day_date_with_variant_suffix_match = re.match(day_date_with_variant_suffix_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
            if day_date_with_variant_suffix_match is not None:
                export_datetime_str, session_str, export_file_type = day_date_with_variant_suffix_match.group('export_datetime_str'), day_date_with_variant_suffix_match.group('session_str'), day_date_with_variant_suffix_match.group('export_file_type')
                # parse the datetime from the export_datetime_str and convert it to datetime object
                export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        
            else:
                print(f'ERR: Could not parse filename: "{filename}"') # 2024-01-18_GL_t_split_df
                return None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        
    if export_file_type[0] == '(' and export_file_type[-1] == ')':
        # Trim the brackets from the file type if they're present:
        export_file_type = export_file_type[1:-1]

    return export_datetime, session_str, export_file_type, decoding_time_bin_size_str


def find_most_recent_files(found_session_export_paths: List[Path], debug_print: bool = False) -> Dict[str, Dict[str, Tuple[Path, datetime]]]:
    """
    Returns a dictionary representing the most recent files for each session type among a list of provided file paths.

    Parameters:
    found_session_export_paths (List[Path]): A list of Paths representing files to be checked.
    debug_print (bool): A flag to trigger debugging print statements within the function. Default is False.

    Returns:
    Dict[str, Dict[str, Tuple[Path, datetime]]]: A nested dictionary where the main keys represent 
    different session types. The inner dictionary's keys represent file types and values are the most recent 
    Path and datetime for this combination of session and file type.
    
    # now sessions is a dictionary where the key is the session_str and the value is another dictionary.
    # This inner dictionary's key is the file type and the value is the most recent path for this combination of session and file type
    # Thus, laps_csv and ripple_csv can be obtained from the dictionary for each session

    """
    # Function 'parse_filename' should be defined in the global scope
    parsed_paths = [(*parse_filename(p), p) for p in found_session_export_paths if (parse_filename(p)[0] is not None)]
    parsed_paths.sort(reverse=True)

    if debug_print:
        print(f'parsed_paths: {parsed_paths}')

    sessions = {}
    for export_datetime, session_str, file_type, path, decoding_time_bin_size_str in parsed_paths:
        if session_str not in sessions:
            sessions[session_str] = {}

        if (file_type not in sessions[session_str]) or (sessions[session_str][file_type][-1] < export_datetime):
            sessions[session_str][file_type] = (path, decoding_time_bin_size_str, export_datetime)
    
    return sessions
    

def process_csv_file(file: str, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
    """ reads the CSV file and adds the 'session_name' column if it is missing. 
    
    """
    df = pd.read_csv(file)
    df['session_name'] = session_name 
    if curr_session_t_delta is not None:
        df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
    return df


@define(slots=False)
class AcrossSessionCSVOutputFormat:
    data_description = ["AcrossSession"]
    epoch_description = ["Laps", "Ripple"]
    granularity_description = ["per-Epoch", "per-TimeBin"]
    
    parts_names = ["export_date", "date_name", "epochs", "granularity"]
    
    def parse_filename(self, a_filename: str):
        if a_filename.endswith('.csv'):
            a_filename = a_filename.removesuffix('.csv') # drop the .csv suffix
        # split on the underscore into the parts
        parts = a_filename.split('_')
        if len(parts) == 4:
            export_date, date_name, epochs, granularity  = parts
        else:
            raise NotImplementedError(f"a_csv_filename: '{a_filename}' expected four parts but got {len(parts)} parts.\n\tparts: {parts}")
        return export_date, date_name, epochs, granularity
    

