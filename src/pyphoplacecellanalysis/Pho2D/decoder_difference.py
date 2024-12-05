import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch, FancyArrow
from matplotlib import patheffects

from pyphocorehelpers.gui.interaction_helpers import CallbackWrapper
from pyphocorehelpers.function_helpers import function_attributes


@function_attributes(short_name='predicted_position_difference', tags=['display', 'display_helper', 'decoder', 'decoder_difference', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:52')
def display_predicted_position_difference(active_one_step_decoder, active_two_step_decoder, active_resampled_measured_positions):
    """ Draw difference between predicted and measured position 
    
    Draws an arrow from the measured position to the predicted position for each timestep

    NOT YET USED
    Usage:

        from pyphoplacecellanalysis.Pho2D.decoder_difference import display_predicted_position_difference

        active_resampled_pos_df = active_computed_data.extended_stats.time_binned_position_df.copy() # active_computed_data.extended_stats.time_binned_position_df  # 1717 rows × 16 columns
        active_resampled_measured_positions = active_resampled_pos_df[['x','y']].to_numpy() # The measured positions resampled (interpolated) at the window centers. 
        display_predicted_position_difference(active_one_step_decoder, active_two_step_decoder, active_resampled_measured_positions)

    """
    
    def _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n_and_x_prev', override_variable_value=None, update_callback_function=None):
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
        sframe = Slider(axframe, 'Frame', 0, num_frames-1, valinit=2, valfmt='%d')

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

    def _temp_debug_draw_predicted_position_difference(predicted_positions, measured_positions, time_window, ax=None):
        if ax is None:
            raise NotImplementedError
            # ax = plt.gca()
        print(f'predicted_positions[{time_window},:]: {predicted_positions[time_window,:]}, measured_positions[{time_window},:]: {measured_positions[time_window,:]}')
        predicted_point = np.squeeze(predicted_positions[time_window,:])
        measured_point = np.squeeze(measured_positions[time_window,:])
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
        print(f'predicted_positions[{time_window},:]: {predicted_positions[time_window,:]}, measured_positions[{time_window},:]: {measured_positions[time_window,:]}')
        predicted_point = np.squeeze(predicted_positions[time_window,:])
        measured_point = np.squeeze(measured_positions[time_window,:])
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



    def perform_draw_predicted_position_difference(frame, ax=None):
        return _temp_debug_draw_predicted_position_difference(active_one_step_decoder.most_likely_positions, active_resampled_measured_positions, frame, ax=ax)

    def perform_update_predicted_position_difference(frame, ax=None, predicted_line=None, measured_line=None, **kwargs):
        return _temp_debug_draw_update_predicted_position_difference(active_one_step_decoder.most_likely_positions, active_resampled_measured_positions, frame, ax=ax, predicted_line=predicted_line, measured_line=measured_line, **kwargs)


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    active_predicted_position_difference_plot_callback_wrapper = CallbackWrapper(perform_draw_predicted_position_difference, perform_update_predicted_position_difference, dict())
    _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, variable_name='p_x_given_n', update_callback_function=active_predicted_position_difference_plot_callback_wrapper)

