import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch, FancyArrow
from matplotlib import patheffects

from neuropy.core import Epoch
from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs
from pyphocorehelpers.gui.interaction_helpers import CallbackWrapper
from pyphocorehelpers.indexing_helpers import interleave_elements

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import DecoderResultDisplayingPlot2D
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import build_position_df_resampled_to_time_windows, build_position_df_time_window_idx

from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import stacked_epoch_slices_matplotlib_build_view

from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuCommand # for AddNewDecodedPosition_MatplotlibPlotCommand

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder 


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
        
    def _display_plot_marginal_1D_most_likely_position_comparisons(computation_result, active_config, variable_name='x', posterior_name='p_x_given_n', most_likely_positions_mode='corrected', **kwargs):
        """ renders a plot with the 1D Marginals either (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. 
        
        most_likely_positions_mode: 'standard'|'corrected'
        posterior_name: 'p_x_given_n'|'p_x_given_n_and_x_prev'
        
        ax = destination_plot.ui.matplotlib_view_widget.ax,
        variable_name = 'x',
        
        """
        # print(f'_display_plot_marginal_1D_most_likely_position_comparisons(...): active_config: {active_config}, kwargs: {kwargs}')
        
        active_decoder = computation_result.computed_data['pf2D_Decoder']
        if variable_name == 'x':
            active_marginals = active_decoder.marginal.x
            active_bins = active_decoder.xbin
        else:
            active_marginals = active_decoder.marginal.y
            active_bins = active_decoder.ybin
        
        if most_likely_positions_mode == 'standard':
            active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        elif most_likely_positions_mode == 'corrected':
            active_most_likely_positions = active_marginals.revised_most_likely_positions_1D # Interpolated most likely positions computed by active_decoder.compute_corrected_positions()
        else:
            raise NotImplementedError
        
    
        # posterior_name must be either ['p_x_given_n', 'p_x_given_n_and_x_prev']
        if posterior_name == 'p_x_given_n':
            active_posterior = active_marginals.p_x_given_n
        elif posterior_name == 'p_x_given_n_and_x_prev':
            active_posterior = active_marginals.p_x_given_n_and_x_prev
        else:
            raise NotImplementedError
        
                
        ## Get the previously created matplotlib_view_widget figure/ax:
        fig, curr_ax = plot_1D_most_likely_position_comparsions(computation_result.sess.position.to_dataframe(), time_window_centers=active_decoder.time_window_centers, xbin=active_bins,
                                                        posterior=active_posterior,
                                                        active_most_likely_positions_1D=active_most_likely_positions,
                                                        **overriding_dict_with(lhs_dict={'ax':None, 'variable_name':variable_name, 'enable_flat_line_drawing':False, 'debug_print': False}, **kwargs))
        
        return fig, curr_ax

            
            
    def _display_plot_most_likely_position_comparisons(computation_result, active_config, **kwargs):
        """ renders a 2D plot with separate subplots for the (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. """
        
        # Call the plot function with the decoder result.
        fig, axs = plot_most_likely_position_comparsions(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe(), **overriding_dict_with(lhs_dict={'show_posterior':True, 'show_one_step_most_likely_positions_plots':True}, **kwargs))
        # fig, axs = plot_most_likely_position_comparsions(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe(), **({'show_posterior':True, 'show_one_step_most_likely_positions_plots':True}|kwargs) )
        
        # show_two_step_most_likely_positions_plots=True
        
        active_two_step_decoder = computation_result.computed_data.get('pf2D_TwoStepDecoder', None)
        if active_two_step_decoder is not None:
            # have valid two_step_decoder, plot those predictions as well:
            # active_two_step_decoder['most_likely_positions'][time_window_bin_idx,:]
            active_time_window_variable = computation_result.computed_data['pf2D_Decoder'].active_time_window_centers
            active_most_likely_positions_x = active_two_step_decoder['most_likely_positions'][:,0]
            active_most_likely_positions_y = active_two_step_decoder['most_likely_positions'][:,1]
            two_step_options_dict = { # Green?
                'color':'#00ff7f99',
                'face_color':'#55ff0099',
                'edge_color':'#00aa0099'
            }
            # marker_style: 'circle', marker_size:0.25
            axs[0].plot(active_time_window_variable, active_most_likely_positions_x, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions x') # (Num windows x 2)
            axs[1].plot(active_time_window_variable, active_most_likely_positions_y, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions y') # (Num windows x 2)
            

    def _display_plot_decoded_epoch_slices(computation_result, active_config, filter_epochs='ripple', **kwargs):
            """ renders a plot with the 1D Marginals either (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. 
            
            most_likely_positions_mode: 'standard'|'corrected'
            
            
            ax = destination_plot.ui.matplotlib_view_widget.ax,
            variable_name = 'x',
            
            """
            decoding_time_bin_size = kwargs.pop('decoding_time_bin_size', 0.02)
            default_figure_name = 'stacked_epoch_slices_matplotlib_subplots'
            active_decoder = computation_result.computed_data['pf2D_Decoder']
            
            if isinstance(filter_epochs, str):
                if filter_epochs == 'laps':
                    ## Lap-Epochs Decoding:
                    laps_copy = deepcopy(computation_result.sess.laps)
                    
                    
                    # active_filter_epochs = laps_copy.filtered_by_lap_flat_index(np.arange(6)).as_epoch_obj() # epoch object
                    active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                    # default_figure_name = f'{default_figure_name}_Laps'
                    default_figure_name = f'Laps'
                    
                    # ## Build Epochs:
                    # # epochs = sess.laps.to_dataframe()
                    # epochs = active_filter_epochs.to_dataframe()
                    # epoch_slices = epochs[['start', 'stop']].to_numpy()
                    # epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in active_filter_epochs.to_dataframe()[['lap_id','maze_id','lap_dir']].itertuples()] # LONG
                    epoch_description_list = [f'lap[{epoch_tuple.lap_id}]' for epoch_tuple in active_filter_epochs.to_dataframe()[['lap_id']].itertuples()] # Short
                    
                    
                elif filter_epochs == 'pbe':
                    ## PBEs-Epochs Decoding:
                    active_filter_epochs = deepcopy(computation_result.sess.pbe) # epoch object
                    # default_figure_name = f'{default_figure_name}_PBEs'
                    default_figure_name = f'PBEs'
                
                elif filter_epochs == 'ripple':
                    ## Ripple-Epochs Decoding:
                    active_filter_epochs = deepcopy(computation_result.sess.ripple) # epoch object
                    # default_figure_name = f'{default_figure_name}_Ripples'
                    default_figure_name = f'Ripples'
                    # epoch_description_list = [f'ripple {epoch_tuple.label} (peakpower: {epoch_tuple.peakpower})' for epoch_tuple in active_filter_epochs.to_dataframe()[['label', 'peakpower']].itertuples()] # LONG
                    epoch_description_list = [f'ripple[{epoch_tuple.label}]' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()] # SHORT
                    
                    
                elif filter_epochs == 'replay':
                    active_filter_epochs = deepcopy(computation_result.sess.replay) # epoch object
                    active_filter_epochs = active_filter_epochs.drop_duplicates("start") # tries to remove duplicate replays to take care of `AssertionError: Intervals in start_stop_times_arr must be non-overlapping`, but it hasn't worked.
                    # filter_epochs.columns # ['epoch_id', 'rel_id', 'start', 'end', 'replay_r', 'replay_p', 'template_id', 'flat_replay_idx', 'duration']
                    if not 'stop' in active_filter_epochs.columns:
                        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
                        active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
                    # default_figure_name = f'{default_figure_name}_Replay'
                    default_figure_name = f'Replay'
                    epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()]
                    
                else:
                    raise NotImplementedError
            else:
                # Use it raw, hope it's right
                active_filter_epochs = filter_epochs
                default_figure_name = f'{default_figure_name}_CUSTOM'
                epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()]
                
                
                
            filter_epochs_decoder_result = active_decoder.decode_specific_epochs(computation_result.sess.spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
            filter_epochs_decoder_result.epoch_description_list = epoch_description_list


            out_plot_tuple = plot_decoded_epoch_slices(active_filter_epochs, filter_epochs_decoder_result, global_pos_df=computation_result.sess.position.to_dataframe(), xbin=active_decoder.xbin,
                                                                    **overriding_dict_with(lhs_dict={'name':default_figure_name, 'debug_test_max_num_slices':8, 'enable_flat_line_drawing':False, 'debug_print': False}, **kwargs))
            # params, plots_data, plots, ui = out_plot_tuple
            
            return out_plot_tuple
        
    




# ==================================================================================================================== #
# Private Implementations                                                                                              #
# ==================================================================================================================== #
def plot_1D_most_likely_position_comparsions(measured_position_df, time_window_centers, xbin, ax=None, posterior=None, active_most_likely_positions_1D=None, enable_flat_line_drawing=False, variable_name = 'x', debug_print=False):
    """ renders a single 2D subplot in MATPLOTLIB for a 1D position axes: the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top.
    
    Input:
    
        enable_flat_line_drawing
    
    Usage:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
    
        ## Test Plotting just a single dimension of the 2D posterior:
        pho_custom_decoder = new_decoder
        active_posterior = pho_custom_decoder.p_x_given_n
        # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
        marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
        marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        # np.shape(marginal_posterior_x) # (41, 3464)
        fig, ax = plot_1D_most_likely_position_comparsions(sess.position.to_dataframe(), time_window_centers=pho_custom_decoder.active_time_window_centers, xbin=pho_custom_decoder.xbin,
                                                        posterior=marginal_posterior_x,
                                                        active_most_likely_positions_1D=pho_custom_decoder.most_likely_positions[:,0].T,
                                                        enable_flat_line_drawing=False, debug_print=False)
        fig.show()
            
            
    """
    with plt.ion():
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,15), clear=True, sharex=True, sharey=False, constrained_layout=True)
        else:
            fig = plt.gcf()
        
        # Actual Position Plots:
        ax.plot(measured_position_df['t'].to_numpy(), measured_position_df[variable_name].to_numpy(), label=f'measured {variable_name}', color='#ff000066', alpha=0.8, marker='+', markersize=4) # Opaque RED # , linestyle='dashed', linewidth=2, color='#ff0000ff'
        ax.set_title(variable_name)
       
        if posterior is not None:
            # Compute extents for imshow:
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
            xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])           
            x_first_extent = (xmin, xmax, ymin, ymax)
            active_extent = x_first_extent
            im_posterior_x = ax.imshow(posterior, extent=active_extent, **main_plot_kwargs)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

        if active_most_likely_positions_1D is not None:
            # Most likely position plots:

            if enable_flat_line_drawing:
                # Enable drawing flat lines for each time bin interval instead of just displaying the single point in the middle:
                #   build separate points for the start and end of each bin interval, and the repeat every element of the x and y values to line them up.
                time_bin_size = (time_window_centers[1]-time_window_centers[0])
                active_half_time_bin_seconds = time_bin_size / 2.0
                active_time_window_start_points = np.expand_dims(time_window_centers - active_half_time_bin_seconds, axis=1)
                active_time_window_end_points = np.expand_dims(time_window_centers + active_half_time_bin_seconds, axis=1)
                active_time_window_start_end_points = interleave_elements(active_time_window_start_points, active_time_window_end_points) # from pyphocorehelpers.indexing_helpers import interleave_elements
                
                if debug_print:
                    print(f'np.shape(active_time_window_end_points): {np.shape(active_time_window_end_points)}\nnp.shape(active_time_window_start_end_points): {np.shape(active_time_window_start_end_points)}') 
                    # np.shape(active_time_window_end_points): (5783, 1)
                    # np.shape(active_time_window_start_end_points): (11566, 1)

                active_time_window_variable = active_time_window_start_end_points
                active_most_likely_positions_1D = np.repeat(active_most_likely_positions_1D, 2, axis=0) # repeat each element twice
            else:
                active_time_window_variable = time_window_centers
            
            ax.plot(active_time_window_variable, active_most_likely_positions_1D, lw=1.0, color='gray', alpha=0.8, marker='+', markersize=6, label=f'1-step: most likely positions {variable_name}') # (Num windows x 2)
            # ax.plot(active_time_window_variable, active_most_likely_positions_1D, lw=1.0, color='gray', alpha=0.4, label=f'1-step: most likely positions {variable_name}') # (Num windows x 2)
            
            
        return fig, ax

def plot_most_likely_position_comparsions(pho_custom_decoder, position_df, axs=None, show_posterior=True, show_one_step_most_likely_positions_plots=True, enable_flat_line_drawing=True, debug_print=False):
    """ renders a 2D plot in MATPLOTLIB with separate subplots for the (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top.
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
            
        created_new_figure = False
        if axs is None:
            created_new_figure = True
            fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(15,15), clear=True, sharex=True, sharey=False, constrained_layout=True)
            
        else:
            fig = plt.gcf()
            assert len(axs) == 2
        
        if show_one_step_most_likely_positions_plots:
            active_most_likely_positions_x = pho_custom_decoder.most_likely_positions[:,0].T
            active_most_likely_positions_y = pho_custom_decoder.most_likely_positions[:,1].T
        else:
            active_most_likely_positions_x = None
            active_most_likely_positions_y = None

        if show_posterior:
            active_posterior = pho_custom_decoder.p_x_given_n
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            marginal_posterior_y = np.squeeze(np.sum(active_posterior, 0)) # sum over all x. Result should be [y_bins x time_bins]
            marginal_posterior_y = marginal_posterior_y / np.sum(marginal_posterior_y, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
            marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        else:
            marginal_posterior_x = None
            marginal_posterior_y = None

        ## NEW: Perform the plot_1D_most_likely_position_comparsions(...) calls for x & y:
        # X:
        _, axs[0] = plot_1D_most_likely_position_comparsions(position_df, variable_name='x', time_window_centers=pho_custom_decoder.active_time_window_centers, xbin=pho_custom_decoder.xbin,
                                                    posterior=marginal_posterior_x,
                                                    active_most_likely_positions_1D=active_most_likely_positions_x, ax=axs[0],
                                                    enable_flat_line_drawing=enable_flat_line_drawing, debug_print=debug_print)
        
        # Y:
        _, axs[1] = plot_1D_most_likely_position_comparsions(position_df, variable_name='y', time_window_centers=pho_custom_decoder.active_time_window_centers, xbin=pho_custom_decoder.ybin,
                                                    posterior=marginal_posterior_y,
                                                    active_most_likely_positions_1D=active_most_likely_positions_y, ax=axs[1],
                                                    enable_flat_line_drawing=enable_flat_line_drawing, debug_print=debug_print)    
        if created_new_figure:
            # Only update if we created a new figure:
            fig.suptitle(f'Decoded Position data component comparison')
            
        return fig, axs

        
        
        
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
def _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, time_binned_position_df: pd.DataFrame, variable_name='p_x_given_n_and_x_prev', override_variable_value=None, update_callback_function=None):
    """Matplotlib-based imshow plot with interactive slider for displaying two-step bayesian decoding results

    Called from the display function '_display_two_step_decoder_prediction_error_2D' defined above to implement its core functionality

    ## Added _update_measured_animal_position_point(...)
    DEPENDS ON active_computed_data.extended_stats.time_binned_position_df
    
    Args:
        active_one_step_decoder ([type]): [description]
        active_two_step_decoder ([type]): [description]
        time_binned_position_df: should be obtained from `active_computed_data.extended_stats.time_binned_position_df` by default
        variable_name (str, optional): [description]. Defaults to 'p_x_given_n_and_x_prev'.
        override_variable_value ([type], optional): [description]. Defaults to None.
        update_callback_function ([type], optional): [description]. Defaults to None.
        
        
    Usage:
        # Simple plot type 1:
        # plotted_variable_name = kwargs.get('variable_name', 'p_x_given_n') # Tries to get the user-provided variable name, otherwise defaults to 'p_x_given_n'
        plotted_variable_name = 'p_x_given_n' # Tries to get the user-provided variable name, otherwise defaults to 'p_x_given_n'
        _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, active_computed_data.extended_stats.time_binned_position_df, variable_name=plotted_variable_name) # Works

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
    debug_print = True
    if debug_print:
        print(f'_temp_debug_two_step_plots_animated_imshow: variable_name="{variable_name}", np.shape: {np.shape(variable_value)}, num_frames: {num_frames}')

    fig, ax = plt.subplots(ncols=1, nrows=1, num=f'debug_two_step_animated: variable_name={variable_name}', figsize=(15,15), clear=True, constrained_layout=False)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    
    # Get extents:    
    xmin, xmax, ymin, ymax = (active_one_step_decoder.xbin[0], active_one_step_decoder.xbin[-1], active_one_step_decoder.ybin[0], active_one_step_decoder.ybin[-1])
    x_first_extent = (xmin, xmax, ymin, ymax) # traditional order of the extant axes
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
    
    ## Setup Auxillary Plots:
    plot_measured_animal_position = (time_binned_position_df is not None)
    
    if plot_measured_animal_position:
        active_resampled_pos_df = time_binned_position_df.copy() # active_computed_data.extended_stats.time_binned_position_df  # 1717 rows × 16 columns
        active_resampled_measured_positions = active_resampled_pos_df[['x','y']].to_numpy() # The measured positions resampled (interpolated) at the window centers. 
        measured_point = np.squeeze(active_resampled_measured_positions[frame,:])
        ## decided on using scatter
        # measured_positions_scatter = ax.scatter(measured_point[0], measured_point[1], color='white') # PathCollection
        measured_positions_scatter, = ax.plot(measured_point[0], measured_point[1], color='white', marker='o', ls='') # PathCollection
        
        def _update_measured_animal_position_point(time_window_idx, ax=None):
            """ captures `active_resampled_measured_positions` and `measured_positions_scatter` """
            measured_point = np.squeeze(active_resampled_measured_positions[time_window_idx,:])
            ## TODO: this would need to use set_offsets(...) if we wanted to stick with scatter plot.
            measured_positions_scatter.set_xdata(measured_point[0])
            measured_positions_scatter.set_ydata(measured_point[1])
    
    
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
        if plot_measured_animal_position:
            _update_measured_animal_position_point(new_frame, ax=ax)
        
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


# ==================================================================================================================== #
# Functions for rendering a stack of decoded epochs in a stacked_epoch_slices-style manner                             #
# ==================================================================================================================== #

def plot_decoded_epoch_slices(filter_epochs, filter_epochs_decoder_result, global_pos_df, xbin=None, enable_flat_line_drawing=False, debug_test_max_num_slices=20, name='stacked_epoch_slices_matplotlib_subplots', debug_print=False):
    """ plots the decoded epoch results in a stacked slices view 
    
    Usage:    
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

        decoding_time_bin_size = 0.05

        ## Testing PBE Decoding
        # active_decoder = new_2D_decoder
        active_decoder = new_1D_decoder
        # filter_epochs = sess.laps.as_epoch_obj() # epoch object
        filter_epochs = sess.ripple # epoch object
        filter_epochs_decoder_result = active_decoder.decode_specific_epochs(sess.spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)

        params, plots_data, plots, ui = plot_decoded_epoch_slices(filter_epochs, filter_epochs_decoder_result, global_pos_df=sess.position.to_dataframe(), xbin=active_decoder.xbin, enable_flat_line_drawing=False, debug_test_max_num_slices=20, debug_print=False)


    # Laps Example:
        ## Lap-Epochs Decoding:
        laps_copy = deepcopy(sess.laps)
        laps_filter_epochs = laps_copy.filtered_by_lap_flat_index(np.arange(6)).as_epoch_obj() # epoch object
        laps_filter_epochs_decoder_result = active_decoder.decode_specific_epochs(sess.spikes_df, filter_epochs=laps_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
        laps_plot_tuple = plot_decoded_epoch_slices(laps_filter_epochs, laps_filter_epochs_decoder_result, global_pos_df=sess.position.to_dataframe(), xbin=active_decoder.xbin,
                                                                enable_flat_line_drawing=enable_flat_line_drawing, debug_test_max_num_slices=debug_test_max_num_slices, name='stacked_epoch_slices_matplotlib_subplots_LAPS', debug_print=debug_print)
                                                                

    # Ripples Example:                                                          
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

        debug_print = False
        enable_flat_line_drawing = False
        # enable_flat_line_drawing = True
        debug_test_max_num_slices = 16

        params, plots_data, plots, ui = plot_decoded_epoch_slices(filter_epochs, filter_epochs_decoder_result, global_pos_df=sess.position.to_dataframe(), xbin=active_decoder.xbin,
                                                                enable_flat_line_drawing=enable_flat_line_drawing, debug_test_max_num_slices=debug_test_max_num_slices, name='stacked_epoch_slices_matplotlib_subplots_RIPPLES', debug_print=debug_print)

    """
    ## Build Epochs:
    if isinstance(filter_epochs, pd.DataFrame):
        epochs_df = filter_epochs
    elif isinstance(filter_epochs, Epoch):
        epochs_df = filter_epochs.to_dataframe()
    else:
        raise NotImplementedError
    
    # if 'label' not in epochs_df.columns:
    epochs_df['label'] = epochs_df.index.to_numpy() # integer ripple indexing
    epoch_slices = epochs_df[['start', 'stop']].to_numpy()
    # epoch_description_list = [f'ripple {epoch_tuple.label} (peakpower: {epoch_tuple.peakpower})' for epoch_tuple in epochs_df[['label', 'peakpower']].itertuples()]

    epoch_labels = filter_epochs_decoder_result.epoch_description_list.copy()
    print(f'epoch_labels: {epoch_labels}')
    
    plot_function_name = 'Stacked Epoch Slices View - MATPLOTLIB subplots Version'
    params, plots_data, plots, ui = stacked_epoch_slices_matplotlib_build_view(epoch_slices, epoch_labels=epoch_labels, name=name, plot_function_name=plot_function_name, debug_test_max_num_slices=debug_test_max_num_slices, debug_print=debug_print)

    for i, curr_ax in enumerate(plots.axs):
        curr_time_bin_container = filter_epochs_decoder_result.time_bin_containers[i]
        curr_time_bins = curr_time_bin_container.centers
        curr_posterior_container = filter_epochs_decoder_result.marginal_x_list[i]
        curr_posterior = curr_posterior_container.p_x_given_n
        curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D
        
        if debug_print:
            print(f'i : {i}, curr_posterior.shape: {curr_posterior.shape}')

        plots.fig, curr_ax = plot_1D_most_likely_position_comparsions(global_pos_df, ax=curr_ax, time_window_centers=curr_time_bins, xbin=xbin,
                                                           posterior=curr_posterior,
                                                           active_most_likely_positions_1D=curr_most_likely_positions,
                                                           enable_flat_line_drawing=enable_flat_line_drawing, debug_print=debug_print)
        curr_ax.set_xlim(*plots_data.epoch_slices[i,:])
        curr_ax.set_title('')

    return params, plots_data, plots, ui

# ==================================================================================================================== #
# Rendering various different methods of normalizing spike counts and firing rates  (Matplotlib-based)                 #
# ==================================================================================================================== #
def plot_spike_count_and_firing_rate_normalizations(pho_custom_decoder, axs=None):
    """ Plots several different normalizations of binned firing rate and spike counts, optionally plotting them. 
    History: Extracted from 2022-06-22 Notebook 93
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_spike_count_and_firing_rate_normalizations
        pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
        plot_spike_count_and_firing_rate_normalizations(pho_custom_decoder)
    """

    created_new_figure = False
    if axs is None:
        created_new_figure = True
        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15,15), clear=True, sharex=True, sharey=False, constrained_layout=True)
        
    else:
        fig = plt.gcf()
        assert len(axs) == 3

    spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = BayesianPlacemapPositionDecoder.perform_compute_spike_count_and_firing_rate_normalizations(pho_custom_decoder)

    # produces a fraction which indicates which proportion of the window's firing belonged to each unit (accounts for global changes in firing rate (each window is scaled by the toial spikes of all cells in that window)    
    # plt.figure(num=5)
    curr_ax = axs[0]
    plt.imshow(spike_proportion_global_fr_normalized, cmap='turbo', aspect='auto')
    curr_ax.set_title('Unit Specific Proportion of Window Spikes')
    curr_ax.set_xlabel('Binned Time Window')
    curr_ax.set_ylabel('Neuron Proportion Activity')

    # plt.figure(num=6)
    curr_ax = axs[1]
    curr_ax.imshow(firing_rate, cmap='turbo', aspect='auto')
    curr_ax.set_title('Unit Specific Binned Firing Rates')
    curr_ax.set_xlabel('Binned Time Window')
    curr_ax.set_ylabel('Neuron Firing Rate')

    # produces a unit firing rate for each window that accounts for global changes in firing rate (each window is scaled by the firing rate of all cells in that window
    # plt.figure(num=7)
    curr_ax = axs[2]
    curr_ax.imshow(firing_rate_global_fr_normalized, cmap='turbo', aspect='auto')
    curr_ax.set_title('Unit Specific Binned Firing Rates (Global Normalized)')
    curr_ax.set_xlabel('Binned Time Window')
    curr_ax.set_ylabel('Neuron Proportion Firing Rate')

    if created_new_figure:
        # Only update if we created a new figure:
        fig.suptitle(f'Spike Count and Firing Rate Normalizations')
        
    return fig, axs


# ==================================================================================================================== #
# Matplotlib-based 1D Position Decoder that can be added and synced to RasterPlot2D via its menu                       #
# ==================================================================================================================== #


# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_most_likely_position_comparsions, plot_1D_most_likely_position_comparsions






class CreateNewStackedDecodedEpochSlicesPlotCommand(BaseMenuCommand):
    """ Creates a stacked decoded epoch slices view by calling _display_plot_decoded_epoch_slices
    
    """
    def __init__(self, spike_raster_window, active_pipeline, active_config_name=None, active_context=None, filter_epochs='laps', display_output={}) -> None:
        super(CreateNewStackedDecodedEpochSlicesPlotCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._active_pipeline = active_pipeline
        self._active_config_name = active_config_name
        self._context = active_context
        self._display_output = display_output
        self._filter_epochs = filter_epochs
        
        
    def execute(self, *args, **kwargs) -> None:
        """  """
        # print(f'CreateNewStackedDecodedEpochSlicesPlotCommand(): {self._filter_epochs} callback')
        _out_plot_tuple = self._active_pipeline.display('_display_plot_decoded_epoch_slices', self._active_config_name, filter_epochs=self._filter_epochs, debug_test_max_num_slices=16)
        _out_params, _out_plots_data, _out_plots, _out_ui = _out_plot_tuple
        # _out_display_key = f'stackedEpochSlicesMatplotlibSubplots_{_out_params.name}'
        _out_display_key = f'{_out_params.name}'
        # print(f'_out_display_key: {_out_display_key}')
        self._display_output[_out_display_key] = _out_plot_tuple
        


class AddNewDecodedPosition_MatplotlibPlotCommand(BaseMenuCommand):
    """ analagous to CreateNewDataExplorer_ipspikes_PlotterCommand, holds references to the variables needed to perform the entire action (such as the reference to the decoder) which aren't accessible during the building of the menus. """
    def __init__(self, spike_raster_window, curr_active_pipeline, active_config_name, display_output={}) -> None:
        super(AddNewDecodedPosition_MatplotlibPlotCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._curr_active_pipeline = curr_active_pipeline
        self._active_config_name = active_config_name
        self._display_output = display_output
        # print(f'AddNewDecodedPosition_MatplotlibPlotCommand.__init__(...)')

    def execute(self, *args, **kwargs) -> None:
        ## To begin, the destination plot must have a matplotlib widget plot to render to:
        # print(f'AddNewDecodedPosition_MatplotlibPlotCommand.execute(...)')
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        # If no plot to render on, do this:
        widget, matplotlib_fig, matplotlib_fig_ax = active_2d_plot.add_new_matplotlib_render_plot_widget()
        # most_likely_positions_mode: 'standard'|'corrected'
        fig, curr_ax = self._curr_active_pipeline.display('_display_plot_marginal_1D_most_likely_position_comparisons', self._active_config_name, variable_name='x', most_likely_positions_mode='corrected', ax=active_2d_plot.ui.matplotlib_view_widget.ax)
        # print(f'\t AddNewDecodedPosition_MatplotlibPlotCommand.execute(...) finished with the display call...')
        active_2d_plot.ui.matplotlib_view_widget.draw()
        active_2d_plot.sync_matplotlib_render_plot_widget() # Sync it with the active window:
        # print(f'\t AddNewDecodedPosition_MatplotlibPlotCommand.execute() is done.')
        
        