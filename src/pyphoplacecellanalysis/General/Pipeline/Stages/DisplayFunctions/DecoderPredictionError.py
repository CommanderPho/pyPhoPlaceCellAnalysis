import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch, FancyArrow
from matplotlib import patheffects

from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.gui.interaction_helpers import CallbackWrapper
from pyphocorehelpers.indexing_helpers import interleave_elements

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import DecoderResultDisplayingPlot2D
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import build_position_df_resampled_to_time_windows, build_position_df_time_window_idx


class DefaultDecoderDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing Bayesian Decoder performance. """

    def _display_two_step_decoder_prediction_error_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
            """ Plots the prediction error for the two_step decoder at each point in time.
                Based off of "_temp_debug_two_step_plots_animated_imshow"
                
                THIS ONE WORKS. 
            """
            # Get the decoders from the computation result:
            active_one_step_decoder = computation_result.computed_data['pf2D_Decoder']
            active_two_step_decoder = computation_result.computed_data.get('pf2D_TwoStepDecoder', None)
            active_measured_positions = computation_result.sess.position.to_dataframe()

        
            # Simple plot type 1:
            plotted_variable_name = kwargs.get('variable_name', 'p_x_given_n') # Tries to get the user-provided variable name, otherwise defaults to 'p_x_given_n'
            _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name=plotted_variable_name) # Works
            # _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n') # Works
            # _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n_and_x_prev')
            return # end

    def _display_decoder_result(computation_result, active_config, **kwargs):
        renderer = DecoderResultDisplayingPlot2D(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe())
        def animate(i):
            # print(f'animate({i})')
            return renderer.display(i)
        
    
    def _display_plot_most_likely_position_comparisons(computation_result, active_config, **kwargs):
        """ renders a 2D plot with separate subplots for the (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. """
        def plot_most_likely_position_comparsions(pho_custom_decoder, position_df, show_posterior=True, show_one_step_most_likely_positions_plots=True, debug_print=False):
            """
            Usage:
                fig, axs = plot_most_likely_position_comparsions(pho_custom_decoder, sess.position.to_dataframe())
            """
            # xmin, xmax, ymin, ymax, tmin, tmax = compute_data_extent(position_df['x'].to_numpy(), position_df['y'].to_numpy(), position_df['t'].to_numpy())
            
            with plt.ion():
                overlay_mode = True
                if overlay_mode:
                    nrows=2
                else:
                    nrows=4
                fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(15,15), clear=True, sharex=True, sharey=False, constrained_layout=False)
                # active_window = pho_custom_decoder.active_time_windows[window_idx] # a tuple with a start time and end time
                # active_p_x_given_n = np.squeeze(pho_custom_decoder.p_x_given_n[:,:,window_idx]) # same size as occupancy
                # Actual Position Plots:
                axs[0].plot(position_df['t'].to_numpy(), position_df['x'].to_numpy(), label='measured x', color='#ff0000ff') # Opaque RED
                axs[0].set_title('x')
                axs[1].plot(position_df['t'].to_numpy(), position_df['y'].to_numpy(), label='measured y', color='#ff0000ff') # Opaque RED
                axs[1].set_title('y')
                # # Most likely position plots:
                # axs[2].plot(pho_custom_decoder.active_time_window_centers, np.squeeze(pho_custom_decoder.most_likely_positions[:,0]), lw=0.5) # (Num windows x 2)
                # axs[2].set_title('most likely positions x')
                # axs[3].plot(pho_custom_decoder.active_time_window_centers, np.squeeze(pho_custom_decoder.most_likely_positions[:,1]), lw=0.5) # (Num windows x 2)
                # axs[3].set_title('most likely positions y')
                
                if show_posterior:
                    active_posterior = pho_custom_decoder.p_x_given_n
                    # active_posterior = pho_custom_decoder['p_x_given_n_and_x_prev'] # dict-style
                    
                    # re-normalize each marginal after summing
                    marginal_posterior_y = np.squeeze(np.sum(active_posterior, 0)) # sum over all x. Result should be [y_bins x time_bins]
                    marginal_posterior_y = marginal_posterior_y / np.sum(marginal_posterior_y, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
                    marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
                    marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

                    # Compute extents foir imshow:
                    
                    # im = ax.imshow(curr_p_x_given_n, **main_plot_kwargs) # add the curr_px_given_n image
                    main_plot_kwargs = {
                        'origin': 'lower',
                        'vmin': 0,
                        'vmax': 1,
                        'cmap': 'turbo',
                        'interpolation':'nearest',
                        'aspect':'auto',
                    }
                        
                    # Posterior distribution heatmaps at each point.
                    # X
                    xmin, xmax, ymin, ymax = (pho_custom_decoder.active_time_window_centers[0], pho_custom_decoder.active_time_window_centers[-1], pho_custom_decoder.xbin[0], pho_custom_decoder.xbin[-1])           
                    # xmin, xmax = axs[0].get_xlim()
                    x_first_extent = (xmin, xmax, ymin, ymax)
                    # y_first_extent = (ymin, ymax, xmin, xmax)
                    active_extent = x_first_extent
                    im_posterior_x = axs[0].imshow(marginal_posterior_x, extent=active_extent, **main_plot_kwargs)
                    # axs[0].axis("off")
                    axs[0].set_xlim((xmin, xmax))
                    axs[0].set_ylim((ymin, ymax))
                    # Y
                    xmin, xmax, ymin, ymax = (pho_custom_decoder.active_time_window_centers[0], pho_custom_decoder.active_time_window_centers[-1], pho_custom_decoder.ybin[0], pho_custom_decoder.ybin[-1])
                    x_first_extent = (xmin, xmax, ymin, ymax)
                    # y_first_extent = (ymin, ymax, xmin, xmax)
                    active_extent = x_first_extent
                    im_posterior_y = axs[1].imshow(marginal_posterior_y, extent=active_extent, **main_plot_kwargs)
                    # axs[1].axis("off")
                    axs[1].set_xlim((xmin, xmax))
                    axs[1].set_ylim((ymin, ymax))
                    
                    
                if show_one_step_most_likely_positions_plots:
                    # Most likely position plots:
                    active_time_window_variable = pho_custom_decoder.active_time_window_centers
                    active_most_likely_positions_x = pho_custom_decoder.most_likely_positions[:,0].T
                    active_most_likely_positions_y = pho_custom_decoder.most_likely_positions[:,1].T
                    
                    # Enable drawing flat lines for each time bin interval instead of just displaying the single point in the middle:
                    #   build separate points for the start and end of each bin interval, and the repeat every element of the x and y values to line them up.
                    active_half_time_bin_seconds = pho_custom_decoder.time_bin_size / 2.0
                    active_time_window_start_points = np.expand_dims(pho_custom_decoder.active_time_window_centers - active_half_time_bin_seconds, axis=1)
                    active_time_window_end_points = np.expand_dims(pho_custom_decoder.active_time_window_centers + active_half_time_bin_seconds, axis=1)
                    active_time_window_start_end_points = interleave_elements(active_time_window_start_points, active_time_window_end_points) # from pyphocorehelpers.indexing_helpers import interleave_elements
                    
                    if debug_print:
                        print(f'np.shape(active_time_window_end_points): {np.shape(active_time_window_end_points)}\nnp.shape(active_time_window_start_end_points): {np.shape(active_time_window_start_end_points)}') 
                        # np.shape(active_time_window_end_points): (5783, 1)
                        # np.shape(active_time_window_start_end_points): (11566, 1)

                    active_time_window_variable = active_time_window_start_end_points
                    active_most_likely_positions_x = np.repeat(active_most_likely_positions_x, 2, axis=0) # repeat each element twice
                    active_most_likely_positions_y = np.repeat(active_most_likely_positions_y, 2, axis=0) # repeat each element twice    
                    
                    axs[0].plot(active_time_window_variable, active_most_likely_positions_x, lw=1.0, color='gray', alpha=0.4, label='1-step: most likely positions x') # (Num windows x 2)
                    # axs[0].set_title('most likely positions x')
                    axs[1].plot(active_time_window_variable, active_most_likely_positions_y, lw=1.0, color='gray', alpha=0.4, label='1-step: most likely positions y') # (Num windows x 2)
                    # axs[1].set_title('most likely positions y')
                    
                    
                fig.suptitle(f'Decoded Position data component comparison')
                return fig, axs
        
        # Call the plot function with the decoder result.
        fig, axs = plot_most_likely_position_comparsions(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe(), **overriding_dict_with(lhs_dict={'show_posterior':True, 'show_one_step_most_likely_positions_plots':True}, **kwargs))
        # fig, axs = plot_most_likely_position_comparsions(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe(), **({'show_posterior':True, 'show_one_step_most_likely_positions_plots':True}|kwargs) )
        
        # show_two_step_most_likely_positions_plots=True
        
        active_two_step_decoder = computation_result.computed_data.get('pf2D_TwoStepDecoder', None)
        if active_two_step_decoder is not None:
            # have valid two_step_decoder, plot those predictions as well:
            # active_two_step_decoder['most_likely_positions'][:, time_window_bin_idx]
            active_time_window_variable = computation_result.computed_data['pf2D_Decoder'].active_time_window_centers
            active_most_likely_positions_x = active_two_step_decoder['most_likely_positions'][0,:]
            active_most_likely_positions_y = active_two_step_decoder['most_likely_positions'][1,:]
            two_step_options_dict = { # Green?
                'color':'#00ff7f99',
                'face_color':'#55ff0099',
                'edge_color':'#00aa0099'
            }
            # marker_style: 'circle', marker_size:0.25
            axs[0].plot(active_time_window_variable, active_most_likely_positions_x, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions x') # (Num windows x 2)
            axs[1].plot(active_time_window_variable, active_most_likely_positions_y, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions y') # (Num windows x 2)
            


# ==================================================================================================================== #
# Private Implementations                                                                                              #
# ==================================================================================================================== #

def _temp_debug_two_step_plots(active_one_step_decoder, active_two_step_decoder, variable_name='all_scaling_factors_k', override_variable_value=None):
    """ Handles plots using the plot command """
    if override_variable_value is None:
        try:
            variable_value = active_two_step_decoder[variable_name]
        except (TypeError, KeyError):
            # fallback to the one_step_decoder
            variable_value = getattr(active_one_step_decoder, variable_name, None)
    else:
        # if override_variable_value is set, ignore the input info and use it.
        variable_value = override_variable_value
    debug_print = False
    if debug_print:
        print(f'_temp_debug_two_step_plots: variable_name="{variable_name}", np.shape: {np.shape(variable_value)}')
    plt.figure(num=f'debug_two_step: variable_name={variable_name}', clear=True)
    plt.plot(variable_value, marker='d', markersize=1.0, linestyle='None') # 'd' is a thin diamond marker
    plt.xlabel('time window')
    plt.ylabel(variable_name)
    plt.title(f'debug_two_step: variable_name={variable_name}')
    
    
def _temp_debug_two_step_plots_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n_and_x_prev', override_variable_value=None, timewindow: int=None):
    if override_variable_value is None:
        try:
            variable_value = active_two_step_decoder[variable_name]
        except (TypeError, KeyError):
            # fallback to the one_step_decoder
            variable_value = getattr(active_one_step_decoder, variable_name, None)
    else:
        # if override_variable_value is set, ignore the input info and use it.
        variable_value = override_variable_value
    debug_print = False
    if debug_print:
        print(f'_temp_debug_two_step_plots_imshow: variable_name="{variable_name}", np.shape: {np.shape(variable_value)}')

    if timewindow is not None:
        variable_value = variable_value[:,:,timewindow] # reduce it to 2D if it's 3D

    # plt.figure(num=f'debug_two_step: variable_name={variable_name}', clear=True)
    fig, axs = plt.subplots(ncols=1, nrows=1, num=f'debug_two_step: variable_name={variable_name}', figsize=(15,15), clear=True, constrained_layout=True)

    main_plot_kwargs = {
        'origin': 'lower',
        'cmap': 'turbo',
        'aspect':'auto',
    }

    xmin, xmax, ymin, ymax = (active_one_step_decoder.active_time_window_centers[0], active_one_step_decoder.active_time_window_centers[-1], active_one_step_decoder.xbin[0], active_one_step_decoder.xbin[-1])
    extent = (xmin, xmax, ymin, ymax)
    im_out = axs.imshow(variable_value, extent=extent, **main_plot_kwargs)
    axs.axis("off")
    # plt.xlabel('time window')
    # plt.ylabel(variable_name)
    plt.title(f'debug_two_step: {variable_name}')
    # return im_out


    
# MAIN IMPLEMENTATION FUNCTION:
def _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n_and_x_prev', override_variable_value=None, update_callback_function=None):
    """Matplotlib-based imshow plot with interactive slider for displaying two-step bayesian decoding results

    Called from the display function '_display_two_step_decoder_prediction_error_2D' defined above to implement its core functionality
    

    Args:
        active_one_step_decoder ([type]): [description]
        active_two_step_decoder ([type]): [description]
        variable_name (str, optional): [description]. Defaults to 'p_x_given_n_and_x_prev'.
        override_variable_value ([type], optional): [description]. Defaults to None.
        update_callback_function ([type], optional): [description]. Defaults to None.
    """
    if override_variable_value is None:
        try:
            variable_value = active_two_step_decoder[variable_name]
        except (TypeError, KeyError):
            # fallback to the one_step_decoder
            variable_value = getattr(active_one_step_decoder, variable_name, None)
    else:
        # if override_variable_value is set, ignore the input info and use it.
        variable_value = override_variable_value

    num_frames = np.shape(variable_value)[-1]
    debug_print = False
    if debug_print:
        print(f'_temp_debug_two_step_plots_animated_imshow: variable_name="{variable_name}", np.shape: {np.shape(variable_value)}, num_frames: {num_frames}')

    fig, ax = plt.subplots(ncols=1, nrows=1, num=f'debug_two_step_animated: variable_name={variable_name}', figsize=(15,15), clear=True, constrained_layout=False)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    
    # Get extents:
    # xmin, xmax, ymin, ymax = (xbin_edges[0], xbin_edges[-1], ybin_edges[0], ybin_edges[-1]) # from example imshow    
    xmin, xmax, ymin, ymax = (active_one_step_decoder.xbin[0], active_one_step_decoder.xbin[-1], active_one_step_decoder.ybin[0], active_one_step_decoder.ybin[-1])
    x_first_extent = (xmin, xmax, ymin, ymax) # traditional order of the extant axes
    # y_first_extent = (ymin, ymax, xmin, xmax) # swapped the order of the extent axes.
    active_extent = x_first_extent # for 'x == horizontal orientation'
    # active_extent = y_first_extent # for 'x == vertical orientation'

    main_plot_kwargs = {
        'origin': 'lower',
        'cmap': 'turbo',
        'extent': active_extent,
        # 'aspect':'auto',
    }

    curr_val = variable_value[:,:,frame] # untranslated output:
    curr_val = np.swapaxes(curr_val, 0, 1) # x_horizontal_matrix: swap the first two axes while leaving the last intact. Returns a view into the matrix so it doesn't modify the value
    
    im_out = ax.imshow(curr_val, **main_plot_kwargs)
    
    # for 'x == horizontal orientation':
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # ax.axis("off")
    plt.title(f'debug_two_step: {variable_name}')

    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, num_frames-1, valinit=2, valfmt='%d') # MATPLOTLIB Slider

    def update(val):
        new_frame = int(np.around(sframe.val))
        # print(f'new_frame: {new_frame}')
        curr_val = variable_value[:,:,new_frame] # untranslated output:
        curr_val = np.swapaxes(curr_val, 0, 1) # x_horizontal_matrix: swap the first two axes while leaving the last intact. Returns a view into the matrix so it doesn't modify the value
        im_out.set_data(curr_val)
        # ax.relim()
        # ax.autoscale_view()
        if update_callback_function is not None:
            update_callback_function(new_frame, ax=ax)
        plt.draw()

    sframe.on_changed(update)
    plt.draw()
    # plt.show()
    
# ==================================================================================================================== #
# Functions for drawing the decoded position and the animal position as a callback                                     #
# ==================================================================================================================== #

def _temp_debug_draw_predicted_position_difference(predicted_positions, measured_positions, time_window, ax=None):
    """ Draws the decoded position and the actual animal's position, and an arrow between them. """
    if ax is None:
        raise NotImplementedError
        # ax = plt.gca()
    debug_print = False
    if debug_print:
        print(f'predicted_positions[{time_window},:]: {predicted_positions[time_window,:]}, measured_positions[{time_window},:]: {measured_positions[time_window,:]}')
    # predicted_point = predicted_positions[time_window,:]
    # measured_point = measured_positions[time_window,:]
    predicted_point = np.squeeze(predicted_positions[time_window,:])
    measured_point = np.squeeze(measured_positions[time_window,:])
    if debug_print:
        print(f'\tpredicted_point: {predicted_point}, measured_point: {measured_point}')
    
    # ## For 'x == vertical orientation' only: Need to transform the point (swap y and x) as is typical in an imshow plot:
    # predicted_point = [predicted_point[-1], predicted_point[0]] # reverse the x and y coords
    # measured_point = [measured_point[-1], measured_point[0]] # reverse the x and y coords
    
    # Draw displacement arrow:
    # active_arrow = FancyArrowPatch(posA=tuple(predicted_point), posB=tuple(measured_point), path=None, arrowstyle=']->', connectionstyle='arc3', shrinkA=2, shrinkB=2, mutation_scale=8, mutation_aspect=1, color='C2') 
    active_arrow = FancyArrowPatch(posA=tuple(predicted_point), posB=tuple(measured_point), path=None, arrowstyle='simple', connectionstyle='arc3', shrinkA=1, shrinkB=1, mutation_scale=20, mutation_aspect=1,
                                   color='k', alpha=0.5, path_effects=[patheffects.withStroke(linewidth=3, foreground='white')]) 
    ax.add_patch(active_arrow)
    # Draw the points on top:
    predicted_line, = ax.plot(predicted_point[0], predicted_point[1], marker='d', markersize=6.0, linestyle='None', label='predicted', markeredgecolor='#ffffffc8', markerfacecolor='#e0ffeac8') # 'd' is a thin diamond marker
    measured_line, = ax.plot(measured_point[0], measured_point[1], marker='o', markersize=6.0, linestyle='None', label='measured', markeredgecolor='#ff7f0efa', markerfacecolor='#ff7f0ea0') # 'o' is a circle marker
    fig = plt.gcf()
    fig.legend((predicted_line, measured_line), ('Predicted', 'Measured'), 'upper right')
    return {'ax':ax, 'predicted_line':predicted_line, 'measured_line':measured_line, 'active_arrow':active_arrow}
    # update function:
    
    
    
def _temp_debug_draw_update_predicted_position_difference(predicted_positions, measured_positions, time_window, ax=None, predicted_line=None, measured_line=None, active_arrow=None):
    assert measured_line is not None, "measured_line is required!"
    assert predicted_line is not None, "predicted_line is required!"
    debug_print = False
    if debug_print:
        print(f'predicted_positions[{time_window},:]: {predicted_positions[time_window,:]}, measured_positions[{time_window},:]: {measured_positions[time_window,:]}')
    # predicted_point = predicted_positions[time_window,:]
    # measured_point = measured_positions[time_window,:]
    predicted_point = np.squeeze(predicted_positions[time_window,:])
    measured_point = np.squeeze(measured_positions[time_window,:])
    if debug_print:
        print(f'\tpredicted_point: {predicted_point}, measured_point: {measured_point}')
    # ## For 'x == vertical orientation' only: Need to transform the point (swap y and x) as is typical in an imshow plot:
    # predicted_point = [predicted_point[-1], predicted_point[0]] # reverse the x and y coords
    # measured_point = [measured_point[-1], measured_point[0]] # reverse the x and y coords
    predicted_line.set_xdata(predicted_point[0])
    predicted_line.set_ydata(predicted_point[1])
    measured_line.set_xdata(measured_point[0])
    measured_line.set_ydata(measured_point[1])
    if active_arrow is not None:
        active_arrow.set_positions(tuple(predicted_point), tuple(measured_point))
    plt.draw()
    # fig.canvas.draw_idle() # TODO: is this somehow better?







