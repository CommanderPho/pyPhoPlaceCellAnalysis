import napari
import numpy as np
import pandas as pd
from pathlib import Path

from pyphocorehelpers.indexing_helpers import find_neighbours

# # Send local variables to the console
# viewer.update_console(locals())
# print(f'{locals()}')

# Layers are accessible via:
# viewer.layers[a_name].contrast_limits=(100, 175)

def napari_add_surprise_data_layers(viewer: napari.viewer.Viewer, active_relative_entropy_results):
    ## Add multiple layers to the viewer:


    if 'snapshot_occupancy_weighted_tuning_maps' not in active_relative_entropy_results:
        active_relative_entropy_results['snapshot_occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in active_relative_entropy_results['historical_snapshots'].values()])


    image_layer_dict = {}
    layer_properties_dict = {
        'snapshot_occupancy_weighted_tuning_maps': dict(blending='additive', colormap='viridis', name='pf1D_dt'),
    #  'flat_jensen_shannon_distance_results': dict(blending='additive', colormap='gray'),
        'long_short_rel_entr_curves_frames': dict(blending='additive', colormap='bop blue'),
        'short_long_rel_entr_curves_frames': dict(blending='additive', colormap='red'),
        
    }

    for a_name, layer_properties in layer_properties_dict.items():
        # image_layer_dict[a_name] = viewer.add_image(active_relative_entropy_results_xr_dict[a_name].to_numpy().astype(float), name=a_name)
        image_layer_dict[a_name] = viewer.add_image(active_relative_entropy_results[a_name].astype(float), **(dict(name=a_name)|layer_properties))

    return image_layer_dict


def napari_add_animal_position(viewer: napari.viewer.Viewer, position_df: pd.DataFrame, time_intervals):
    """ adds the animal's position as a track and a point layer to the napari viewer.


    Usage:
    
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_add_animal_position
    napari_add_animal_position(viewer=viewer, position_df=global_pf1D_dt.all_time_filtered_pos_df[['t','x','binned_x']], time_intervals)

    
    """
    # global_pf1D.position.linear_pos.shape

    # global_pf1D.position
    position_df = position_df.copy().dropna() #.binned_x
    n_time_windows = np.shape(time_intervals)[0]
    
    ## Resample to the windows
    window_binned_animal_positions = []

    for window_idx in np.arange(n_time_windows):
        # active_window = self.decoder.active_time_windows[window_idx] # a tuple with a start time and end time

        active_window = time_intervals[window_idx]

        # active_most_likely_x_indicies = self.decoder.most_likely_position_indicies[:,window_idx]
        # active_most_likely_x_position = (self.xbin_centers[active_most_likely_x_indicies[0]], self.ybin_centers[active_most_likely_x_indicies[1]])

        if position_df.position.ndim > 1:
            # pos_dims = ['x','y']
            pos_dims = ['binned_x', 'binned_y']
        else:
            # pos_dims = ['x']
            pos_dims = ['binned_x']

        active_window_start = active_window[0]
        active_window_end = active_window[1]
        active_window_midpoint = active_window_start + ((active_window_end - active_window_start) / 2.0)
        [lowerneighbor_ind, upperneighbor_ind] = find_neighbours(active_window_midpoint, position_df, 't')
        active_nearest_measured_position = position_df.loc[lowerneighbor_ind, pos_dims].to_numpy()
        window_binned_animal_positions.append(active_nearest_measured_position)

    window_binned_animal_positions = np.squeeze(np.array(window_binned_animal_positions))
    error_positions = np.logical_not(np.isfinite(window_binned_animal_positions))
    window_binned_animal_positions[error_positions] = 0.0
    track_confidence = np.ones_like(window_binned_animal_positions)
    track_confidence[error_positions] = 0.0

    # should be [track_id, t, x_pos]
    animal_pos_track_data = np.asarray([[0, window_idx, 5.0, window_binned_animal_positions[window_idx]] for window_idx in np.arange(n_time_windows)])
    animal_pos_vertices = animal_pos_track_data[:, 1:]
    print(f'np.shape(animal_pos_track_data): {np.shape(animal_pos_track_data)}')

    tracks_properties = {
        'time': animal_pos_track_data[:, 1],
        'confidence': track_confidence
    }

    track_params = dict(tail_width=7.0, head_length=5, tail_length=6)

    viewer.add_points(animal_pos_vertices, size=5, symbol='vbar', face_color='#00aa00ff', edge_color='lime', name='animal_pos_points', opacity=0.42)
    viewer.add_tracks(animal_pos_track_data, properties=tracks_properties, **track_params)
    # viewer.add_tracks(animal_pos_track) # , properties=tracks_properties

#TODO 2023-09-27 18:39: - [ ] Add grid_bin_bounds and track shape to napari
# global_pf1D_dt.config.grid_bin_bounds
# rect = np.array([[0, 0], [3, 1]])
# viewer.add_shapes(rect, shape_type='rectangle', edge_width=0.1)


def napari_set_time_windw_index(viewer: napari.viewer.Viewer, current_timepoint):
    """ sets the time window index (corresponding to the slider along the bottom of the viewer) to the specified index.

        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_set_time_windw_index

    """
    variable_timepoint = list(viewer.dims.current_step)
    variable_timepoint[0] = current_timepoint
    viewer.dims.current_step = variable_timepoint



def napari_export_video_frames(viewer: napari.viewer.Viewer, time_intervals, imageseries_output_directory='output/videos/imageseries/'):
    """ 
    
    Usage:
    
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_export_video_frames
    napari_add_animal_position(viewer=viewer, position_df=global_pf1D_dt.all_time_filtered_pos_df[['t','x','binned_x']], time_intervals)

    
    """
    from skimage.io import imsave
    # from napari_animation import Animation

    # animation = Animation(viewer)
    # viewer.update_console({'animation': animation})
    # animation.capture_keyframe()
    viewer.reset_view()

    n_time_windows = np.shape(time_intervals)[0]

    if not isinstance(imageseries_output_directory, Path):
        imageseries_output_directory: Path = Path(imageseries_output_directory).resolve()
        
    for window_idx in np.arange(n_time_windows):
        napari_set_time_windw_index(viewer, window_idx+1)
        # take screenshot
        screenshot = viewer.screenshot()
        image_out_path = imageseries_output_directory.joinpath(f'screenshot_{window_idx}.png').resolve()
        imsave(image_out_path, screenshot)
        # animation.capture_keyframe()

    # animation.animate('demo2D.mov', canvas_only=False)
    return imageseries_output_directory