from typing import Any, Dict
import napari
import numpy as np
import pandas as pd
from pathlib import Path

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.indexing_helpers import find_neighbours

# # Send local variables to the console
# viewer.update_console(locals())
# print(f'{locals()}')

# Layers are accessible via:
# viewer.layers[a_name].contrast_limits=(100, 175)

""" 

# viewer.dims. # Dims(ndim=3, ndisplay=2, last_used=0, range=((0.0, 3309.0, 1.0), (-0.2906298631133275, 85.0, 1.0), (-0.24635407377607876, 108.0, 1.0)), current_step=(1654, 42, 53), order=(0, 1, 2), axis_labels=('0', '1', '2'))

"""

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

    assert viewer.dims.ndim == 3
    ## Set the dimensions appropriately
    viewer.dims.axis_labels = ('t', 'neuron_id', 'xbin')
    
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


@function_attributes(short_name=None, tags=['napari', 'config'], input_requires=[], output_provides=[], uses=[], used_by=['napari_extract_layers_info'], creation_date='2024-08-12 08:54', related_items=[])
def extract_layer_info(a_layer):
    """ Extracts info as a dict from a single Napari layer. 
    by default Napari layers print like: `<Shapes layer 'Shapes' at 0x1635a1e8460>`: without any properties that can be easily referenced.
    This function extracts a dict of properties.
    
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import extract_layer_info
     
    """
    out_properties_dict = {}
    positioning = ['scale', 'translate', 'rotate', 'shear', 'affine', 'corner_pixels']
    visual = ['opacity', 'blending', 'visible', 'z_index']
    # positioning = ['scale', 'translate', 'rotate', 'shear', 'affine']
    out_properties_dict['positioning'] = {}
    
    for a_property_name in positioning:
        out_properties_dict['positioning'][a_property_name] = getattr(a_layer, a_property_name)
    return out_properties_dict


@function_attributes(short_name=None, tags=['napari', 'config'], input_requires=[], output_provides=[], uses=['extract_layer_info'], used_by=[], creation_date='2024-08-12 08:54', related_items=[])
def napari_extract_layers_info(layers):
	"""extracts info dict from each layer as well.
	Usage:
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_extract_layers_info
		layers = directional_viewer.layers # [<Shapes layer 'Shapes' at 0x1635a1e8460>, <Shapes layer 'Shapes [1]' at 0x164d5402e50>]
		out_layers_info_dict = debug_print_layers_info(layers)

	"""
	out_layers_info_dict = {}
	for a_layer in layers:
		a_name: str = str(a_layer.name)
		out_properties_dict = extract_layer_info(a_layer)
		out_layers_info_dict[a_name] = out_properties_dict
		# if isinstance(a_layer, Shapes):
		# 	print(f'shapes layer: {a_layer}')
		# 	a_shapes_layer: Shapes = a_layer
		# 	# print(f'a_shapes_layer.properties: {a_shapes_layer.properties}')
		# 	out_properties_dict = extract_layer_info(a_layer)
		# 	print(f'\tout_properties_dict: {out_properties_dict}')
		# 	out_layers_info_dict[a_name] = out_properties_dict
		# else:
		# 	print(f'unknown layer: {a_layer}')	
	return out_layers_info_dict



def napari_from_layers_dict(layers_dict: Dict, viewer=None, title: str = 'napari', ndisplay: int = 2, order: Any = (), axis_labels: Any = (), show: bool = True, **kwargs):
    """ Visualizes position binned activity matrix beside the trial-by-trial correlation matrix.
    
    
    
    Usage:
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict
        viewer, image_layer_dict = napari_from_layers_dict(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix)

        
        image_layer_dict
        # can find peak spatial shift distance by performing convolution and finding time of maximum value?
        _layer_z_scored_tuning_maps = image_layer_dict['z_scored_tuning_maps']
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 56]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 55.5]]), step=array([1, 1, 1]))

        _layer_C_trial_by_trial_correlation_matrix = image_layer_dict['C_trial_by_trial_correlation_matrix']
        _layer_C_trial_by_trial_correlation_matrix.extent

        # _layer_z_scored_tuning_maps.extent
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 84]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 83.5]]), step=array([1, 1, 1]))

        # array([0, 0, 0])


        
    Viewer properties:
        # viewer.grid # GridCanvas(stride=1, shape=(-1, -1), enabled=True)
        viewer.grid.enabled = True
        https://napari.org/0.4.15/guides/preferences.html
        https://forum.image.sc/t/dividing-the-display-in-the-viewer-window/42034
        https://napari.org/stable/howtos/connecting_events.html
        https://napari.org/stable/howtos/headless.html
        https://forum.image.sc/t/napari-how-add-a-text-label-time-always-in-the-same-spot-in-viewer/52932/3
        https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html
        https://napari.org/stable/gallery/add_labels.html
        
    """
    # inputs: z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix
    image_layer_dict: Dict = {}
    assert layers_dict is not None

    
    for i, (a_name, layer_dict) in enumerate(layers_dict.items()):
        img_data = layer_dict.pop('img_data').astype(float) # assumes integrated img_data in the layers dict
        if viewer is None: #i == 0:
            # viewer = napari.view_image(img_data) # rgb=True
            viewer = napari.Viewer(title=title, axis_labels=axis_labels, **kwargs)

        image_layer_dict[a_name] = viewer.add_image(img_data, **(dict(name=a_name)|layer_dict))

    if axis_labels is not None:
        viewer.dims.axis_labels = axis_labels # ('aclu', 'lap', 'xbin')
    
    viewer.grid.enabled = True # enables the grid layout of the data so adjacent data is displayed next to each other

    # outputs: viewer, image_layer_dict
    return viewer, image_layer_dict



# ==================================================================================================================== #
# 2024-02-02 - Napari Export Helpers - Batch export all images                                                         #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_set_time_windw_index



import napari


def napari_add_aclu_slider(viewer, neuron_ids):
    """ adds a neuron aclu index overlay

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_add_aclu_slider


    """
    def on_update_slider(event):
        """ captures: viewer, neuron_ids
        
        Adds a little text label to the bottom right corner
        
        """
        # only trigger if update comes from first axis (optional)
        # print('inside')
        #ind_lambda = viewer.dims.indices[0]

        time = viewer.dims.current_step[0]
        matrix_aclu_IDX = int(time)
        # find the aclu value for this index:
        aclu: int = neuron_ids[matrix_aclu_IDX]
        viewer.text_overlay.text = f"aclu: {aclu}, IDX: {matrix_aclu_IDX}"
        
        # viewer.text_overlay.text = f"{time:1.1f} time"


    # viewer = napari.Viewer()
    # viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
    viewer.text_overlay.visible = True
    _connected_on_update_slider_event = viewer.dims.events.current_step.connect(on_update_slider)
    # viewer.dims.events.current_step.disconnect(on_update_slider)
    return _connected_on_update_slider_event


def napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts, include_trial_by_trial_correlation_matrix:bool = True):
    """ Plots the directional trial-by-trial activity visualization for each of the directional epochs (in the same napari viewer window):

    Compared to `napari_trial_by_trial_activity_viz`, this function plots all four directional epoch results in a single window.

    Usage:
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_plot_directional_trial_by_trial_activity_viz
        
        directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict = napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts)
    
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict

    custom_direction_split_layers_dict = {}
    layers_list_sort_order = ['long_LR_z_scored_tuning_maps', 'long_LR_C_trial_by_trial_correlation_matrix', 'long_RL_z_scored_tuning_maps', 'long_RL_C_trial_by_trial_correlation_matrix', 'short_LR_z_scored_tuning_maps', 'short_LR_C_trial_by_trial_correlation_matrix', 'short_RL_z_scored_tuning_maps', 'short_RL_C_trial_by_trial_correlation_matrix']

    ## Build the image data layers for each
    # for an_epoch_name, (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) in directional_active_lap_pf_results_dicts.items():
    for an_epoch_name, active_trial_by_trial_activity_obj in directional_active_lap_pf_results_dicts.items():
        # (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids)
        z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix
        custom_direction_split_layers_dict[f'{an_epoch_name}_z_scored_tuning_maps'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)) # reshape to be compatibile with C_i's dimensions
        if include_trial_by_trial_correlation_matrix:
            C_trial_by_trial_correlation_matrix = active_trial_by_trial_activity_obj.C_trial_by_trial_correlation_matrix
            custom_direction_split_layers_dict[f'{an_epoch_name}_C_trial_by_trial_correlation_matrix'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix)

    # custom_direction_split_layers_dict

    # directional_viewer, directional_image_layer_dict = napari_trial_by_trial_activity_viz(None, None, layers_dict=custom_direction_split_layers_dict)

    ## sort the layers dict:
    custom_direction_split_layers_dict = {k:custom_direction_split_layers_dict[k] for k in reversed(layers_list_sort_order) if k in custom_direction_split_layers_dict}

    directional_viewer, directional_image_layer_dict = napari_from_layers_dict(layers_dict=custom_direction_split_layers_dict, title='Directional Trial-by-Trial Activity', axis_labels=('aclu', 'lap', 'xbin'))
    if include_trial_by_trial_correlation_matrix:
        directional_viewer.grid.shape = (-1, 4)
    else:
        directional_viewer.grid.shape = (2, -1)

    return directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict


def napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix, cumulative_z_scored_tuning_map_matrix=None, layers_dict=None, **viewer_kwargs):
    """ Visualizes position binned activity matrix beside the trial-by-trial correlation matrix.
    Compared to `napari_plot_directional_trial_by_trial_activity_viz`, this function plots a single directional epoch in a napari window.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_trial_by_trial_activity_viz
        viewer, image_layer_dict = napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix)

        
        image_layer_dict
        # can find peak spatial shift distance by performing convolution and finding time of maximum value?
        _layer_z_scored_tuning_maps = image_layer_dict['z_scored_tuning_maps']
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 56]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 55.5]]), step=array([1, 1, 1]))

        _layer_C_trial_by_trial_correlation_matrix = image_layer_dict['C_trial_by_trial_correlation_matrix']
        _layer_C_trial_by_trial_correlation_matrix.extent

        # _layer_z_scored_tuning_maps.extent
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 84]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 83.5]]), step=array([1, 1, 1]))

        # array([0, 0, 0])

    , title='Trial-by-trial Correlation Matrix C', axis_labels=('aclu', 'lap', 'xbin')
        
    Viewer properties:
        # viewer.grid # GridCanvas(stride=1, shape=(-1, -1), enabled=True)
        viewer.grid.enabled = True
        https://napari.org/0.4.15/guides/preferences.html
        https://forum.image.sc/t/dividing-the-display-in-the-viewer-window/42034
        https://napari.org/stable/howtos/connecting_events.html
        https://napari.org/stable/howtos/headless.html
        https://forum.image.sc/t/napari-how-add-a-text-label-time-always-in-the-same-spot-in-viewer/52932/3
        https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html
        https://napari.org/stable/gallery/add_labels.html
        
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict
    

    # inputs: z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix
    image_layer_dict = {}
    if layers_dict is None:
        # build the default from the values:
        layers_dict = {    
        }

        if cumulative_z_scored_tuning_map_matrix is not None:
            layers_dict['Cumulative_z_scored_tuning_maps'] = dict(blending='translucent', colormap='viridis', name='CUM_z_scored_tuning_maps', img_data=cumulative_z_scored_tuning_map_matrix.transpose(1, 0, 2))
        
        layers_dict.update({
            'z_scored_tuning_maps': dict(blending='translucent', colormap='viridis', name='z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)), # reshape to be compatibile with C_i's dimensions
            'C_trial_by_trial_correlation_matrix': dict(blending='translucent', colormap='viridis', name='C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix),
        })


    viewer = None
    for i, (a_name, layer_dict) in enumerate(layers_dict.items()):
        img_data = layer_dict.pop('img_data').astype(float) # assumes integrated img_data in the layers dict
        if viewer is None: #i == 0:
            # viewer = napari.view_image(img_data) # rgb=True
            viewer = napari.Viewer(**viewer_kwargs)

        image_layer_dict[a_name] = viewer.add_image(img_data, **(dict(name=a_name)|layer_dict))

    viewer.dims.axis_labels = ('aclu', 'lap', 'xbin')
    viewer.grid.enabled = True # enables the grid layout of the data so adjacent data is displayed next to each other

    # outputs: viewer, image_layer_dict
    return viewer, image_layer_dict


def napari_export_image_sequence(viewer: napari.viewer.Viewer, imageseries_output_directory='output/videos/imageseries/', slider_axis_IDX: int = 0, build_filename_from_viewer_callback_fn=None):
    """ 
    
    Based off of `napari_export_video_frames`
    
    Usage:
        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_export_image_sequence

        desired_save_parent_path = Path('/home/halechr/Desktop/test_napari_out').resolve()
        imageseries_output_directory = napari_export_image_sequence(viewer=viewer, imageseries_output_directory=desired_save_parent_path, slider_axis_IDX=0, build_filename_from_viewer_callback_fn=build_filename_from_viewer)

    
    """
    # Get the slide info:
    slider_min, slider_max, slider_step = viewer.dims.range[slider_axis_IDX]
    slider_range = np.arange(start=slider_min, step=slider_step, stop=slider_max)

    # __MAX_SIMPLE_EXPORT_COUNT: int = 5
    n_time_windows = np.shape(slider_range)[0]
    # n_time_windows = min(__MAX_SIMPLE_EXPORT_COUNT, n_time_windows) ## Limit the export to 5 items for testing

    if not isinstance(imageseries_output_directory, Path):
        imageseries_output_directory: Path = Path(imageseries_output_directory).resolve()
        
    for window_idx in np.arange(n_time_windows):
        # napari_set_time_windw_index(viewer, window_idx+1)
        napari_set_time_windw_index(viewer, window_idx)
        
        if build_filename_from_viewer_callback_fn is not None:
            image_out_path = build_filename_from_viewer_callback_fn(viewer, desired_save_parent_path=imageseries_output_directory, slider_axis_IDX=slider_axis_IDX)
        else:
            image_out_path = imageseries_output_directory.joinpath(f'screenshot_{window_idx:05d}.png').resolve()

        screenshot = viewer.screenshot(path=image_out_path, canvas_only=True, flash=False)

    return imageseries_output_directory

# Add text layer
@function_attributes(short_name=None, tags=['napari', 'helper', 'aclu', 'text', 'label'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-25 07:34', related_items=[])
def build_aclu_label(a_viewer, a_matrix_IDX_to_aclu_map, enable_point_based_label:bool = False):
    """ adds a dynamically-updating text overlay label showing the current aclu that updates when the user adjusts the slider.

    Usage:

        from pyphoplacecellanalysis.GUI.Napari.napari_helpers import build_aclu_label
        
        ## Directional
        directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict = napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts, include_trial_by_trial_correlation_matrix=True)
        a_result = list(directional_active_lap_pf_results_dicts.values())[0]
        a_matrix_IDX_to_aclu_map = {v:k for k, v in a_result.aclu_to_matrix_IDX_map.items()}
        on_update_slider, points_layer = build_aclu_label(directional_viewer, a_matrix_IDX_to_aclu_map)

    """
    # INPUTS: a_matrix_IDX_to_aclu_map, a_viewer
    
    if enable_point_based_label:
        # Add dummy points to add text annotations
        points = np.array([[0, 150]])
        features = {'aclu': np.array([-1,]),}
        text = {
            'string': 'aclu: {aclu:02d}',
            'size': 20,
            'color': 'white',
            'translation': np.array([-5, 0]),
        }
        points_layer = a_viewer.add_points(points, features=features, text=text, size=10,
                                        edge_width=7,
                                            edge_width_is_relative=False,
                                            edge_color='gray',
                                            face_color='white',
                                        )
    else:
        points_layer = None

    def update_aclu_text_label():
        ## Captures: a_viewer, points_layer, a_matrix_IDX_to_aclu_map,
        curr_neuron_IDX = a_viewer.dims.current_step[0]
        found_aclu = a_matrix_IDX_to_aclu_map.get(curr_neuron_IDX, "None")
        # print(f'found_aclu: {found_aclu}')
        if points_layer is not None:
            points_layer.features['aclu'] = np.array([found_aclu,])
            points_layer.refresh_text()

        a_viewer.text_overlay.text = f"aclu: {found_aclu:02d}"


    update_aclu_text_label()

    # Define update function
    def on_update_slider(event):
        current_slice = event.value
        update_aclu_text_label()

    a_viewer.text_overlay.visible = True
    # Connect update function to slider event
    a_viewer.dims.events.connect(on_update_slider)
    return on_update_slider, points_layer
