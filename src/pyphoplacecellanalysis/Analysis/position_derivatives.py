
# ==================================================================================================================== #
# Position Derivatives Plotting Helpers                                                                                #
# ==================================================================================================================== #

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types


@function_attributes(short_name=None, tags=['decode', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 14:30', related_items=[])
def _compute_pos_derivs(time_window_centers, position, decoding_time_bin_size, debug_print=False):
    """try recomputing velocties/accelerations
    
    from pyphoplacecellanalysis.Analysis.position_derivatives import _compute_pos_derivs
    
    decoding_time_bin_size = a_result.decoding_time_bin_size
    """ 
    position = deepcopy(position)
    a_first_order_diff = np.diff(position, n=1, prepend=[position[0]]) 
    velocity = a_first_order_diff / float(decoding_time_bin_size) # velocity with real world units of cm/sec
    acceleration = np.diff(velocity, n=1, prepend=[velocity[0]])

    position_derivatives_df: pd.DataFrame = pd.DataFrame({'t': time_window_centers, 'x': position, 'vel_x': velocity, 'accel_x': acceleration})
    if debug_print:
        print(f'time_window_centers: {time_window_centers}')
        print(f'position: {position}')
        print(f'velocity: {velocity}')
        print(f'acceleration: {acceleration}')

    position_derivative_column_names = ['x', 'vel_x', 'accel_x']
    position_derivative_means = position_derivatives_df.mean(axis='index')[position_derivative_column_names].to_numpy()
    position_derivative_medians = position_derivatives_df.median(axis='index')[position_derivative_column_names].to_numpy()
    # position_derivative_medians = position_derivatives_df(axis='index')[position_derivative_column_names].to_numpy()
    if debug_print:
        print(f'\tposition_derivative_means: {position_derivative_means}')
        print(f'\tposition_derivative_medians: {position_derivative_medians}')
    return position_derivatives_df


@function_attributes(short_name=None, tags=['helper','matplotlib','figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=[], used_by=['debug_plot_position_and_derivatives_figure'], creation_date='2024-03-07 18:23', related_items=['debug_plot_position_and_derivatives_figure'])
def debug_plot_helper_add_position_and_derivatives(time_window_centers, position, velocity, acceleration, debug_plot_axs=None, debug_plot_name=None, common_plot_kwargs=None):
        """ HELPER to `debug_plot_position_and_derivatives_figure`: Renders a single series (measured, a_specific_decoder, ...)'s values for all 3 subplots: [position, velocity, acceleration]

        Usage:
            fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(time_window_centers, position, velocity, acceleration, debug_plot_axs=None, debug_plot_name=None, common_plot_kwargs=None)

            from pyphoplacecellanalysis.Analysis.position_derivatives import debug_plot_helper_add_position_and_derivatives

            
            enable_debug_plot = True
            if enable_debug_plot:
                fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            else:
                debug_plot_axs = None
                debug_plot_name = None


            ## Plot measured
            fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(new_measured_pos_df['t'].to_numpy(), new_measured_pos_df['x'].to_numpy(), new_measured_pos_df['vel_x'].to_numpy(), new_measured_pos_df['accel_x'].to_numpy(),
                                                                                    debug_plot_axs=axs, debug_plot_name='measured', common_plot_kwargs=dict(color='k', markersize='2', marker='.', linestyle='None', alpha=0.35))

            ## Plot decoded
            for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
                fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(a_df['t'].to_numpy(), a_df['x'].to_numpy(), a_df['vel_x'].to_numpy(), a_df['accel_x'].to_numpy(),
                                                                                    debug_plot_axs=debug_plot_axs, debug_plot_name=a_name, common_plot_kwargs=dict(marker='o', markersize=3, linestyle='None', alpha=0.6))

                                                                                
        """
        # Setup the figure and subplots
        if debug_plot_axs is None:
            fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        else:
            fig = debug_plot_axs[0].get_figure()

        if debug_plot_name is None:
            debug_plot_name = ''

        common_plot_kwargs = common_plot_kwargs or {}
        common_plot_kwargs = common_plot_kwargs or dict(marker='o', linestyle='None', alpha=0.6)

        # draw_style = common_plot_kwargs.pop('draw_style', None)
        # # Using step with `where='post'` for a steps-post effect
        # if (draw_style is not None) and (draw_style != 'default'):
        #     assert draw_style in ['post', 'mid', 'pre']
        #     debug_plot_axs[0].step(time_window_centers, position, where=draw_style, label='Step (post)')


        # Plot the position data on the first subplot
        debug_plot_axs[0].plot(time_window_centers, position, label=f'{debug_plot_name}_Position', **common_plot_kwargs) # , color='blue'
        debug_plot_axs[0].set_ylabel('Position (m)')
        debug_plot_axs[0].legend()

        # Plot the velocity data on the second subplot
        debug_plot_axs[1].plot(time_window_centers, velocity, label=f'{debug_plot_name}_Velocity', **common_plot_kwargs) # , color='orange'
        debug_plot_axs[1].set_ylabel('Velocity (m/s)')
        debug_plot_axs[1].legend()

        # Plot the acceleration data on the third subplot
        debug_plot_axs[2].plot(time_window_centers, acceleration, label=f'{debug_plot_name}_Acceleration', **common_plot_kwargs) # , color='green'
        debug_plot_axs[2].set_ylabel('Acceleration (m/s²)')
        debug_plot_axs[2].set_xlabel('Time (s)')
        debug_plot_axs[2].legend()

        # # Set a shared title for the subplots
        plt.suptitle('Position, Velocity and Acceleration vs. Time')

        # # Adjust the layout so the subplots fit nicely
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle at the top

        # # Show the subplots
        # plt.show()

        return fig, debug_plot_axs


@function_attributes(short_name=None, tags=['matplotlib','figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=['debug_plot_helper_add_position_and_derivatives'], used_by=[], creation_date='2024-03-07 18:35', related_items=['debug_plot_position_derivatives_stack'])
def debug_plot_position_and_derivatives_figure(new_measured_pos_df, all_epochs_position_derivatives_df_dict, debug_plot_axs=None, debug_figure_title=None, enable_debug_plot = True): # , common_plot_kwargs=None
    """ Renders a single matplotlib figure with a stack of 3 subplots: [position, velocity, acceleration] with both measured and decoded values. Plots measured vs. decoded positions.

    VARIANT: A Matplotlib variant of `debug_plot_position_derivatives_stack`


    Usage:
        from pyphoplacecellanalysis.Analysis.position_derivatives import debug_plot_position_and_derivatives_figure


        ## INPUTS: new_measured_pos_df, all_epochs_position_derivatives_df_dict
        fig, debug_plot_axs = debug_plot_position_and_derivatives_figure(new_measured_pos_df, all_epochs_position_derivatives_df_dict, debug_plot_axs=None, debug_figure_title=None, enable_debug_plot = True, common_plot_kwargs=None)


    """
    
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    if enable_debug_plot:
        fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    else:
        debug_plot_axs = None
        fig = None

    ## Plot measured
    fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(new_measured_pos_df['t'].to_numpy(), new_measured_pos_df['x'].to_numpy(), new_measured_pos_df['vel_x'].to_numpy(), new_measured_pos_df['accel_x'].to_numpy(),
                                                                            debug_plot_axs=debug_plot_axs, debug_plot_name='measured', common_plot_kwargs=dict(color='k', markersize='2', marker='.', linestyle='solid', alpha=0.35))

    ## Plot decoded
    for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
        fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(a_df['t'].to_numpy(), a_df['x'].to_numpy(), a_df['vel_x'].to_numpy(), a_df['accel_x'].to_numpy(),
                                                                            debug_plot_axs=debug_plot_axs, debug_plot_name=a_name, common_plot_kwargs=dict(marker='o', markersize=3, linestyle='solid', alpha=0.6))

    if debug_figure_title is not None:
        plt.suptitle(debug_figure_title)

    return fig, debug_plot_axs



# ==================================================================================================================== #
# 2024-03-06 - measured vs. decoded position distribution comparison                                                   #
# ==================================================================================================================== #
## basically: does the distribution of positions/velocities/accelerations differ between the correct vs. incorrect decoder? Is it reliable enough to determine whether the decoder is correct or not?
## for example using the wrong decoder might lead to wildly-off velocities.


@function_attributes(short_name=None, tags=['plotly', 'figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 18:20', related_items=['debug_plot_position_and_derivatives_figure'])
def debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict, show_scatter = False, subplot_height = 300, figure_width=1900):
    """ Renders a stack of 3 subplots: [position, velocity, acceleration]

    VARIANT: A Plotly variant of `debug_plot_position_and_derivatives_figure`

    Usage:
        show_scatter = False
        subplot_height = 300  # Height in pixels for each subplot; adjust as necessary
        figure_width = 1900


        from pyphoplacecellanalysis.Analysis.position_derivatives import debug_plot_position_derivatives_stack

        # fig = debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict)
        fig = debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict, show_scatter=True)
        fig

    """
    # import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    template: str = 'plotly_dark' # set plotl template
    pio.templates.default = template


    included_legend_entries_list = []

    def debug_plot_position_and_derivatives_figure_plotly(time_window_centers, position, velocity, acceleration, fig, series_idx=0, row_offset=0, debug_plot_name=None, color_palette=None, scatter_plot_kwargs=None, hist_kwargs=None, common_plot_kwargs=None, show_scatter=True):
        """ Plots a single series of positions (like those measured, decoded_long_LR, decoded_long_RL, ...) on the figure.

        Captures: included_legend_entries_list
        """
        # global included_legend_entries_list
        if debug_plot_name is None:
            debug_plot_name = ''
        legend_group_name = f'{debug_plot_name}'  # Define a legend group name
        row_names = [f'{debug_plot_name}_Position', f'{debug_plot_name}_Velocity', f'{debug_plot_name}_Acceleration']

        scatter_plot_kwargs = scatter_plot_kwargs or {}

        common_plot_kwargs = common_plot_kwargs or {}
        common_plot_kwargs = dict(hoverinfo='skip', legendgroup=legend_group_name) | common_plot_kwargs

        if debug_plot_name in included_legend_entries_list:
            ## only generate legend entries for the first series
            common_plot_kwargs['showlegend'] = False


        color = color_palette[series_idx % len(color_palette)] if color_palette else None
        common_plot_kwargs['marker_color'] = color

        hist_kwargs = hist_kwargs or {}
        hist_kwargs = hist_kwargs | dict(opacity=0.5, nbinsx=25, histfunc='count') # , range_y=[0.0, 1.0]

        # is_first_series: bool = (series_idx == 0)
        # if not is_first_series:
        #     ## only generate legend entries for the first series
        #     common_plot_kwargs['showlegend'] = False

        # scatter_fn = go.Scatter
        scatter_fn = go.Scattergl

        with fig.batch_update():
            ## Add the 3 plots (pos, velocity, accel) as the 3 rows
            for i, row, data in zip(np.arange(3), [row_offset + (i+1) for i in np.arange(3)], [position, velocity, acceleration]):
                col: int = 1
                # is_first_row: bool = (i == 0)
                # if not is_first_row:
                #     ## only generate legend entries for the first series AND the first row of that series
                #     common_plot_kwargs['showlegend'] = False
                common_plot_kwargs['showlegend'] = False

                if show_scatter:
                    fig.add_trace(
                        scatter_fn(x=time_window_centers, y=data, name=legend_group_name, **scatter_plot_kwargs, **common_plot_kwargs),
                        row=row, col=col
                    )
                    col += 1        

                # Add histograms to y-axis of existing scatter trace
                common_plot_kwargs['showlegend'] = (debug_plot_name not in included_legend_entries_list) # never show for the histogram
                # , ybins=dict(start=10, end=15, size=1)

                if show_scatter:
                    hist_kwargs.update(dict(y=data)) # plot vertically
                else:
                    hist_kwargs.update(dict(x=data)) # plot horizontally

                fig.add_histogram(name=legend_group_name, **common_plot_kwargs, **hist_kwargs, #marker_color='rgba(0, 0, 255, 0.5)',
                    row=row, col=col, 
                )
                # Set barmode to 'overlay' for overlaying histograms
                fig.update_layout(barmode='overlay')
                included_legend_entries_list.append(debug_plot_name) ## add to list of entries in legend so it isn't included again


        return fig


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    ## INPUTS: new_measured_pos_df, all_epochs_position_derivatives_df_dict

    total_subplots = 4 # (1 + len(all_epochs_position_derivatives_df_dict))  # for measured and all decoded
    
    make_subplots_kwargs = dict(horizontal_spacing=0.01)
    if show_scatter:
        make_subplots_kwargs.update(dict(rows=total_subplots, cols=2, shared_xaxes=True, column_widths=[0.90, 0.10]))
    else:
        make_subplots_kwargs.update(dict(rows=total_subplots, cols=1))
    # Start with creating the overall figure layout with predefined subplots

    fig = make_subplots(**make_subplots_kwargs)

    # Define the height of each subplot and then calculate the total figure height
    total_height = subplot_height * total_subplots # Total figure height
    color_palette = ['white', 'red', 'green', 'blue', 'yellow']
    # color = color_palette[series_idx % len(color_palette)] if color_palette else None
    # common_plot_kwargs['marker_color'] = color

    with fig.batch_update():
        # Plot measured
        series_idx = 0
        fig = debug_plot_position_and_derivatives_figure_plotly(new_measured_pos_df['t'].to_numpy(),
                                                                new_measured_pos_df['x'].to_numpy(),
                                                                new_measured_pos_df['vel_x'].to_numpy(),
                                                                new_measured_pos_df['accel_x'].to_numpy(),
                                                                fig, series_idx,
                                                                debug_plot_name='measured',
                                                                color_palette=color_palette,
                                                                # scatter_plot_kwargs=dict(mode='markers', marker=dict(size=5, opacity=0.35)),
                                                                scatter_plot_kwargs=dict(mode='lines+markers', marker=dict(size=5, opacity=0.35)), # , fill='tozeroy'
                                                                    # mode='lines',  # 'lines+markers' to show both lines and markers fill='tozeroy'  # Fill to zero on the y-axis     
                                                                show_scatter=show_scatter) # , color=series_color
        
        # Add histograms to y-axis of existing scatter trace
        series_idx += 1

        # Plot decoded
        # row_offset = 0  # Increment the row offset for the next series of plots
        for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
            fig = debug_plot_position_and_derivatives_figure_plotly(a_df['t'].to_numpy(),
                                                                    a_df['x'].to_numpy(),
                                                                    a_df['vel_x'].to_numpy(),
                                                                    a_df['accel_x'].to_numpy(),
                                                                    fig, series_idx,
                                                                    debug_plot_name=a_name,
                                                                    color_palette=color_palette,
                                                                    scatter_plot_kwargs=dict(mode='markers', marker=dict(size=3, opacity=0.6)), show_scatter=show_scatter)
            series_idx += 1
            
            
        # Update xaxis and yaxis properties if necessary
        # for i in range(1, total_subplots+1):
        #     fig.update_yaxes(title_text="Value", row=i, col=1)
        # fig.update_xaxes(title_text="Time (s)", row=total_subplots, col=1)  # Update only the last x-axis

        subplot_ylabel_text = ['pos', 'vel.', 'accel.']
        for i, a_label in enumerate(subplot_ylabel_text):
            fig = fig.update_yaxes(title_text=a_label, row=(i+1), col=1)
        ## only last one
        fig = fig.update_xaxes(title_text="Time (s)", row=total_subplots, col=1)  # Update only the last x-axis

        # Set the figure size here
        fig = fig.update_layout(
            height=total_height,  # Set the height of the figure
            width=figure_width,            # Set the width of the figure (or use your desired value)
            showlegend=True,     # You can turn off the legend if it's not needed
            margin=dict(
                l=50,  # Left margin
                r=50,  # Right margin
                t=50,  # Top margin
                b=50   # Bottom margin
            )
        )

    # Show the subplots
    # fig.show()
    return fig


