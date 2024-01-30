# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
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

@function_attributes(short_name=None, tags=['scatter', 'multi-session', 'plot', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_across_sessions_scatter_results(directory, concatenated_laps_df, concatenated_ripple_df, save_figures=False, figure_save_extension='.png'):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    Unknowingly captured: session_name
    
    """
    import plotly.express as px
    # import plotly.graph_objects as go
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers

    def _subfn_build_figure(data, **build_fig_kwargs):
        return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))

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
    fig_laps = go.Figure(px.scatter(concatenated_laps_df, x='delta_aligned_start_t', y='P_Long', title=f"Laps - {laps_title_string_suffix}", color='session_name', size='time_bin_size'), layout_yaxis_range=[0.0, 1.0])

    # Create a bubble chart for ripples
    ripple_num_unique_sessions: int = concatenated_ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    ripple_num_unique_time_bins: int = concatenated_ripple_df.time_bin_size.nunique(dropna=True)
    ripple_title_string_suffix: str = f'{ripple_num_unique_sessions} Sessions'
    fig_ripples = go.Figure(px.scatter(concatenated_ripple_df, x='delta_aligned_start_t', y='P_Long', title=f"Ripples - {ripple_title_string_suffix}", color='session_name', size='time_bin_size'), layout_yaxis_range=[0.0, 1.0])

    if save_figures:
        # Save the figures to the 'figures' subfolder
        print(f'\tsaving figures...')
        fig_laps_name = Path(figures_folder, f"{laps_title_string_suffix.replace(' ', '-')}_laps_marginal{figure_save_extension}").resolve()
        print(f'\tsaving "{fig_laps_name}"...')
        fig_laps.write_image(fig_laps_name)
        fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_ripples_marginal{figure_save_extension}").resolve()
        print(f'\tsaving "{fig_ripple_name}"...')
        fig_ripples.write_image(fig_ripple_name)
    
    # Append both figures to the list
    all_figures.append((fig_laps, fig_ripples))
    
    return all_figures

@function_attributes(short_name=None, tags=['histogram', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str ) -> None:
    """ plots a set of two histograms in subplots, split at the delta for each session.
    from PendingNotebookCode import plot_histograms
    
    """
    histogram_kwargs = dict(orientation="horizontal", bins=25)
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    pre_delta_df.hist(column='P_Long', **histogram_kwargs)
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.show()

    # plot post-delta histogram
    post_delta_df.hist(column='P_Long', **histogram_kwargs)
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    plt.show()

@function_attributes(short_name=None, tags=['histogram', 'stacked', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_stacked_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a colorful stacked histogram for each of the many time-bin sizes
    """
    histogram_kwargs = dict(orientation="horizontal", bins=25)
    
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    time_bin_sizes: int = pre_delta_df['time_bin_size'].unique()
    
    figure_identifier: str = f"{descriptor_str}_preDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.legend()
    plt.show()

    # plot post-delta histogram
    time_bin_sizes: int = post_delta_df['time_bin_size'].unique()
    figure_identifier: str = f"{descriptor_str}_postDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    # plt.legend()
    plt.show()
    





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
    

