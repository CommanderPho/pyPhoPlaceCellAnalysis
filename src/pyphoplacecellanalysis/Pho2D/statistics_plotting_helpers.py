from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from neuropy.utils.result_context import IdentifyingContext
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray


from copy import deepcopy
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
import scipy.stats as stats


import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


# PLOTTING/GRAPHICS __________________________________________________________________________________________________ #
import matplotlib as mpl
import seaborn as sns
from matplotlib import cm, pyplot as plt
from matplotlib.gridspec import GridSpec


@function_attributes(short_name=None, tags=['z-score'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-30 17:10', related_items=[])
def plot_histogram_for_z_scores(z_scores, title_suffix: str=""):
    # Conduct hypothesis test (two-tailed test)
    # Null hypothesis: the mean of z-scores is zero
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Plot histogram of z-scores
    sns.histplot(z_scores, kde=True)
    plt.xlabel('Z-scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Z-scored Values{title_suffix}')
    plt.show()

    # Output p-values for reference
    print(p_values)


@function_attributes(short_name=None, tags=['jointplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-30 17:17', related_items=[])
def pho_jointplot(*args, **kwargs):
    """ wraps sns.jointplot to allow adding titles/axis labels/etc.
    
    Example:
        import seaborn as sns
        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import pho_jointplot
        sns.set_theme(style="ticks")
        common_kwargs = dict(ylim=(0,1), hue='time_bin_size') # , marginal_kws=dict(bins=25, fill=True)
        # sns.jointplot(data=a_laps_all_epoch_bins_marginals_df, x='lap_start_t', y='P_Long', kind="scatter", color="#4CB391")
        pho_jointplot(data=several_time_bin_sizes_laps_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Laps: per epoch') #color="#4CB391")
        pho_jointplot(data=several_time_bin_sizes_ripple_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Ripple: per epoch')
        pho_jointplot(data=several_time_bin_sizes_time_bin_ripple_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Ripple: per time bin')
        pho_jointplot(data=several_time_bin_sizes_time_bin_laps_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Laps: per time bin')
    
    """
    import seaborn as sns
    title = kwargs.pop('title', None)
    _out = sns.jointplot(*args, **kwargs)
    if title is not None:
        plt.suptitle(title)
    return _out




def pho_plothelper(data, **kwargs):
    """ 2024-02-06 - Provides an interface like plotly's classes provide to extract keys fom DataFrame columns or dicts and generate kwargs to pass to a plotting function.
        
        Usage:
            from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import pho_plothelper
            extracted_value_kwargs = pho_plothelper(data=an_aclu_conv_overlap_output['valid_subset'], x='x', y='normalized_convolved_result')
            extracted_value_kwargs

    """
    # data is a pd.DataFrame or Dict-like
    extracted_value_kwargs = {}
    for k,v in kwargs.items():
        extracted_value_kwargs[k] = data[v]
    # end up with `extracted_value_kwargs` containing the real values to plot.
    return extracted_value_kwargs





# ==================================================================================================================== #
# 2024-05-24 - Factored out stats tests from Across Session Point and YellowBlue ...                                   #
# ==================================================================================================================== #

from pyphocorehelpers.indexing_helpers import partition, partition_df, partition_df_dict
import scipy.stats as stats

@function_attributes(short_name=None, tags=['stats', 'tests'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-30 17:14', related_items=[])
def _perform_stats_tests(valid_ripple_df, n_shuffles:int=10000, stats_level_of_significance_alpha: float = 0.05, stats_variable_name:str='short_best_wcorr'):
    """
    Splits the passed df into pre and post delta periods
    Take the difference between the pre and post delta means

    Usage:

        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import _perform_stats_tests


        shuffle_results, p_value, f_value, (dof1, dof2), (variance1, variance2) = _perform_stats_tests(valid_ripple_df, stats_variable_name='short_best_wcorr')

    """
    ## INPUTS: stats_variable_name, valid_ripple_df


    # ['pre-delta', 'post-delta']
    analysis_df = deepcopy(valid_ripple_df[["delta_aligned_start_t", "pre_post_delta_category", stats_variable_name]]).dropna(subset=["pre_post_delta_category", stats_variable_name])
    # partition_df(analysis_df, partitionColumn='pre_post_delta_category')

    # _partition_values, (_pre_delta_df, _post_delta_df) = partition(analysis_df, partitionColumn='pre_post_delta_category') # use `valid_ripple_df` instead of the original dataframe to only get those which are valid.
    _pre_post_delta_partition_df_dict = partition_df_dict(analysis_df, partitionColumn='pre_post_delta_category')
    _pre_delta_df = _pre_post_delta_partition_df_dict['pre-delta']
    _post_delta_df = _pre_post_delta_partition_df_dict['post-delta']

    actual_diff_means = np.nanmean(_post_delta_df[stats_variable_name].to_numpy()) - np.nanmean(_pre_delta_df[stats_variable_name].to_numpy())
    print(f'stats_variable_name: "{stats_variable_name}" -- actual_diff_means: {actual_diff_means}')

    # _pre_delta_df
    # _post_delta_df
    
    # stats_variable_name: str = 'P_Short'

    ## INPUTS: analysis_df, n_shuffles

    shuffle_results = []
    ## INPUT: n_shuffles, analysis_df, stats_variable_name
    shuffled_analysis_df = deepcopy(analysis_df)
    for i in np.arange(n_shuffles):
        # shuffled_analysis_df[stats_variable_name] = shuffled_analysis_df[stats_variable_name].sample(frac=1).to_numpy() # .reset_index(drop=True)
        shuffled_analysis_df['pre_post_delta_category'] = shuffled_analysis_df.sample(frac=1)['pre_post_delta_category'].to_numpy()
        _shuffled_pre_post_delta_partition_df_dict = partition_df_dict(shuffled_analysis_df, partitionColumn='pre_post_delta_category')
        _shuffled_pre_delta_df = _shuffled_pre_post_delta_partition_df_dict['pre-delta']
        _shuffled_post_delta_df = _shuffled_pre_post_delta_partition_df_dict['post-delta']

        _diff_mean = np.nanmean(_shuffled_post_delta_df[stats_variable_name].to_numpy()) - np.nanmean(_shuffled_pre_delta_df[stats_variable_name].to_numpy())
        shuffle_results.append(_diff_mean)

    shuffle_results = np.array(shuffle_results)
    shuffle_results

    ## OUTPUTS: shuffle_results

    ## count the number which exceed the actual mean

    
    np.sum(actual_diff_means > np.abs(shuffle_results))
    # Create the data for two groups
    # group1 = np.random.rand(25)
    # group2 = np.random.rand(20)

    ## INPUTS: stats_variable_name
    print(f'stats_variable_name: {stats_variable_name}')

    group1 = _pre_delta_df[stats_variable_name].to_numpy()
    group2 = _post_delta_df[stats_variable_name].to_numpy()

    # perform mann whitney test 
    stat, p_value = stats.mannwhitneyu(group1, group2) 
    print('Statistics=%.2f, p=%.2f' % (stat, p_value)) 

    # Level of significance 
    
    # conclusion 
    if p_value < stats_level_of_significance_alpha: 
        print('Reject Null Hypothesis (Significant difference between two samples)') 
    else: 
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')

    # Calculate the sample variances
    variance1 = np.var(group1, ddof=1)
    variance2 = np.var(group2, ddof=1)
    
    print('Variance 1:',variance1)
    print('Variance 2:',variance2)

    # Calculate the F-statistic
    f_value = variance1 / variance2
    
    # Calculate the degrees of freedom
    dof1 = len(group1) - 1
    dof2 = len(group2) - 1
    
    # Calculate the p-value
    p_value = stats.f.cdf(f_value, dof1, dof2)
    
    # Print the results
    print('Degree of freedom 1:',dof1)
    print('Degree of freedom 2:',dof2)
    print("F-statistic:", f_value)
    print("p-value:", p_value)

    return shuffle_results, p_value, f_value, (dof1, dof2), (variance1, variance2) 


    # Statistics=351933.00, p=0.00
    # Reject Null Hypothesis (Significant difference between two samples)
    # Variance 1: 0.017720245875104713
    # Variance 2: 0.02501347759017487
    # Degree of freedom 1: 1104
    # Degree of freedom 2: 1282
    # F-statistic: 0.7084279189577826
    # p-value: 1.882791591520268e-09


    # stats_variable_name: short_best_wcorr
    # Statistics=770405.00, p=0.00
    # Reject Null Hypothesis (Significant difference between two samples)
    # Variance 1: 0.13962063046395118
    # Variance 2: 0.15575146845969287
    # Degree of freedom 1: 1108
    # Degree of freedom 2: 1281
    # F-statistic: 0.8964321931904211
    # p-value: 0.030077963036698012


# def plot_histograms( data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str ) -> None:
#     # get the pre-delta epochs
#     pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
#     post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

#     descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
#     # plot pre-delta histogram
#     pre_delta_df.hist(column='P_Long')
#     plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
#     plt.show()

#     # plot post-delta histogram
#     post_delta_df.hist(column='P_Long')
#     plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
#     plt.show()
    

# ==================================================================================================================== #
# 2024-01-29 - Ideal Pho Plotting Interface - UNFINISHED                                                               #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['dataframe', 'UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-2 00:00', related_items=[])
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

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import plot_across_sessions_scatter_results, plot_histograms, plot_stacked_histograms

"""

import matplotlib.pyplot as plt


@function_attributes(short_name=None, tags=['histogram', 'plot', 'old'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-26 00:00', related_items=[])
def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a stacked histogram of the many time-bin sizes 

    Usage:    
        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histograms

        # You can use it like this:
        _out0: "MatplotlibRenderPlots" = plot_histograms(data_type='Laps', session_spec='All Sessions', data_results_df=all_sessions_laps_time_bin_df, time_bin_duration_str="75 ms")
        _out1: "MatplotlibRenderPlots" = plot_histograms(data_type='Ripples', session_spec='All Sessions', data_results_df=all_sessions_ripple_time_bin_df, time_bin_duration_str="75 ms")


    """
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    time_bin_sizes = pre_delta_df['time_bin_size'].unique()
    
    figure_identifier: str = f"{descriptor_str}_preDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=f'{float(time_bin_size):.2f}') 
    
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.legend()
    plt.show()

    # plot post-delta histogram
    time_bin_sizes = post_delta_df['time_bin_size'].unique()
    figure_identifier: str = f"{descriptor_str}_postDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=f'{float(time_bin_size):.2f}') 
    
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    plt.legend()
    plt.show()



@function_attributes(short_name=None, tags=['histogram', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_histograms_across_sessions(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, column='P_Long', **kwargs) -> None:
    """ plots a set of two histograms in subplots, split at the delta for each session.
    from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histograms_across_sessions, plot_stacked_histograms

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
    pre_delta_df.hist(ax=ax_dict['epochs_pre_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_pre_delta'].set_title(f'{descriptor_str} - pre-$\Delta$ time bins')

    # plot post-delta histogram
    post_delta_df.hist(ax=ax_dict['epochs_post_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_post_delta'].set_title(f'{descriptor_str} - post-$\Delta$ time bins')
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_histograms', figures=[fig], axes=ax_dict)



@function_attributes(short_name=None, tags=['matplotlib', 'histogram', 'stacked', 'multi-session', 'plot', 'figure', 'good'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_stacked_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, column_name:str='P_Long', **kwargs):
    """ plots a colorful stacked histogram for each of the many time-bin sizes
    
    variable_name = 'P_Short' # Shows expected effect - short-only replay prior to delta and then split replays post-delta
    
    y_baseline_level: float = 0.5 # for P(short), etc
    # y_baseline_level: float = 0.0 # for wcorr, etc
    
    if is_dark_mode:
        _extras_output_dict["y_mid_line"] = new_fig_ripples.add_hline(y=y_baseline_level, line=dict(color="rgba(0.8,0.8,0.8,.75)", width=2), row='all', col='all')
    else:
        _extras_output_dict["y_mid_line"] = new_fig_ripples.add_hline(y=y_baseline_level, line=dict(color="rgba(0.2,0.2,0.2,.75)", width=2), row='all', col='all')
        

    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    figsize = kwargs.pop('figsize', (12, 2))
    a_context: IdentifyingContext = IdentifyingContext(data_type=data_type, session_spec=session_spec, time_bin_duration_str=time_bin_duration_str, column_name=column_name)
    
    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    figure_identifier: str = f"{descriptor_str}_PrePostDelta"
    a_context = a_context.adding_context_if_missing(descriptor_str=descriptor_str, figure_identifier=figure_identifier)
    if data_type.find('epoch') != -1: # data_type.endswith('epoch'):
        title_indicator: str = 'epochs'
    else:
        assert data_type.find('time-bin') != -1, f"data_type: {data_type} does not contain either expected grainularity descriptor (epochs or time-bin)"   
        # assert data_type.endswith('time-bin'), f"data_type: {data_type}"    
        title_indicator: str = 'time bins'

    a_context = a_context.adding_context_if_missing(title_indicator=title_indicator)
    
    fig = plt.figure(num=figure_identifier, clear=True, figsize=figsize, layout=layout, **kwargs) # layout="constrained", 
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
    
    assert column_name in data_results_df, f"column_name: {column_name} missing from df. {list(data_results_df.columns)}"
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    if (not np.all(np.isnan(time_bin_sizes))):
        # if there's at least one non-NaN time_bin_size, drop the NaNs:
        data_results_df = data_results_df.dropna(subset=['time_bin_size'], inplace=False)
        time_bin_sizes = data_results_df['time_bin_size'].unique() # drop the NaN timebin size

    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]
    
    # plot pre-delta histogram:
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs[column_name].hist(ax=ax_dict['epochs_pre_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 


    ax_dict['epochs_pre_delta'].set_ylabel(f"{column_name}") # only set on the leftmost subplot
    ax_dict['epochs_pre_delta'].set_title(f'pre-$\Delta$ {title_indicator}')
    ax_dict['epochs_pre_delta'].legend()

    # plot post-delta histogram:
    time_bin_sizes: int = post_delta_df['time_bin_size'].unique()
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs[column_name].hist(ax=ax_dict['epochs_post_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_post_delta'].set_title(f'post-$\Delta$ {title_indicator}')
    if len(time_bin_sizes) > 1:
        ax_dict['epochs_post_delta'].legend()
    
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_stacked_histograms', figures=[fig], axes=ax_dict, context=a_context)
