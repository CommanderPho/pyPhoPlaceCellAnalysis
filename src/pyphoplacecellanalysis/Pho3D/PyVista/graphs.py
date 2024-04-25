from copy import deepcopy
import numpy as np
import pyvista as pv


""" 
TODO: REORGANIZE_PLOTTER_SCRIPTS: PyVista

TODO: Why isn't this a class, or in a standardized format?

Main Function: plot_3d_binned_bars

Used in: 
    occupancy_plotting_mixins

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def adjust_saturation(rgb, saturation_factor: float):
    """ adjusts the rgb colors by the saturation_factor by converting to HSV space.
    
    """
    import matplotlib.colors as mcolors
    import colorsys
    # Convert RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)

    if np.ndim(hsv) < 3:
        # Multiply the saturation by the saturation factor
        hsv[:, 1] *= saturation_factor
        
        # Clip the saturation value to stay between 0 and 1
        hsv[:, 1] = np.clip(hsv[:, 1], 0, 1)
        
    else: 
        # Multiply the saturation by the saturation factor
        hsv[:, :, 1] *= saturation_factor
        # Clip the saturation value to stay between 0 and 1
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Convert back to RGB
    return mcolors.hsv_to_rgb(hsv)

def desaturate_colormap(cmap, desaturation_factor: float):
    """
    Desaturate a colormap by a given factor.

    Parameters:
    - cmap: A Matplotlib colormap instance.
    - desaturation_factor: A float between 0 and 1, with 0 being fully desaturated (greyscale)
      and 1 being fully saturated (original colormap colors).

    Returns:
    - new_cmap: A new Matplotlib colormap instance with desaturated colors.

    Usage:
        # Load the existing 'viridis' colormap
        viridis = plt.cm.get_cmap('viridis')
        # Create a desaturated version of 'viridis'
        desaturation_factors = np.linspace(start=1.0, stop=0.0, num=6)
        desaturated_viridis = [desaturate_colormap(viridis, a_desaturation_factor) for a_desaturation_factor in desaturation_factors]
        for a_cmap in desaturated_viridis:
            display(a_cmap)

            
    """
    # Get the colormap colors and the number of entries in the colormap
    cmap_colors = cmap(np.arange(cmap.N))
    
    # Convert RGBA to RGB
    cmap_colors_rgb = cmap_colors[:, :3]
    
    # Create an array of the same shape filled with luminance values
    # The luminance of a color is a weighted average of the R, G, and B values
    # These weights are based on how the human eye perceives color intensity
    luminance = np.dot(cmap_colors_rgb, [0.299, 0.587, 0.114]).reshape(-1, 1)
    
    # Create a grayscale version of the colormap
    grayscale_cmap = np.hstack([luminance, luminance, luminance])
    
    # Blend the original colormap with the grayscale version
    blended_cmap = desaturation_factor * cmap_colors_rgb + (1 - desaturation_factor) * grayscale_cmap
    
    # Add the alpha channel back and create a new colormap
    new_cmap_colors = np.hstack([blended_cmap, cmap_colors[:, 3:]])
    new_cmap = plt.matplotlib.colors.ListedColormap(new_cmap_colors)
    
    return new_cmap




# Define a function to create the colormap
def make_saturating_red_cmap(time: float, N_colors:int=256, min_alpha: float=0.0, max_alpha: float=0.82, debug_print:bool=False):
    """ time is between 0.0 and 1.0 
    
    Usage: Test Example:
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import make_saturating_red_cmap

        n_time_bins = 5
        cmaps = [make_saturating_red_cmap(float(i) / float(n_time_bins - 1)) for i in np.arange(n_time_bins)]
        for cmap in cmaps:
            cmap
            
    Usage:
        # Example usage
        # You would replace this with your actual data and timesteps
        data = np.random.rand(10, 10)  # Sample data
        n_timesteps = 5  # Number of timesteps

        # Plot data with increasing red for each timestep
        fig, axs = plt.subplots(1, n_timesteps, figsize=(15, 3))
        for i in range(n_timesteps):
            time = i / (n_timesteps - 1)  # Normalize time to be between 0 and 1
            # cmap = make_timestep_cmap(time)
            cmap = make_red_cmap(time)
            axs[i].imshow(data, cmap=cmap)
            axs[i].set_title(f'Timestep {i+1}')
        plt.show()

    """
    colors = np.array([(0, 0, 0), (1, 0, 0)]) # np.shape(colors): (2, 3)
    if debug_print:
        print(f'np.shape(colors): {np.shape(colors)}')
    # Apply a saturation change
    saturation_factor = float(time) # 0.5  # Increase saturation by 1.5 times
    adjusted_colors = adjust_saturation(colors, saturation_factor)
    if debug_print:
        print(f'np.shape(adjusted_colors): {np.shape(adjusted_colors)}')
    adjusted_colors = adjusted_colors.tolist()
    ## Set the alpha of the first color to 0.0 and of the final color to 0.82
    adjusted_colors = [[*v, max_alpha] for v in adjusted_colors]
    adjusted_colors[0][-1] = min_alpha

    # n_bins = [2]  # Discretizes the interpolation into bins
    return LinearSegmentedColormap.from_list('CustomMap', adjusted_colors, N=N_colors)





def build_3d_plot_identifier_name(*args):
    return '_'.join(list(args))    


## 3D Binned Bar Plots:

def prepare_binned_data_for_3d_bars(xbin, ybin, data, mask2d=None):
    """ Sequentally repeats xbin, ybin, and data entries to prepare for being plot in 3D bar-plot form.
    Does this by repeating the xbin and ybin except the first and last entries so that there is one entry for each vertex of a 2d rectangular polygon.
    Following that, it repeats in both dimensions the data values, so that each of the created verticies has the same height value.
    Usage:
        modified_xbin, modified_ybin, modified_data = prepare_binned_data_for_3d_bars(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    # duplicates every xbin value except for the first and last:
    modified_xbin = np.repeat(xbin, 2) # remove the first and last elements, which are duplicates
    modified_xbin = modified_xbin[1:-1] 
    modified_ybin = np.repeat(ybin, 2) # remove the first and last elements, which are duplicates
    modified_ybin = modified_ybin[1:-1]
    # there should be precisely double the number of bins in each direction as there are data
    # print(f'np.shape(data): {np.shape(data)}, np.shape(modified_xbin): {np.shape(modified_xbin)}, np.shape(modified_ybin): {np.shape(modified_ybin)}')
    assert (np.shape(data)[0] * 2) == np.shape(modified_xbin)[0], "There shoud be double the number of xbins in the modified array as there are data points in the array."
    assert (np.shape(data)[1] * 2) == np.shape(modified_ybin)[0], "There shoud be double the number of ybins in the modified array as there are data points in the array."
    
    modified_data = np.repeat(data, 2, axis=0)
    modified_data = np.repeat(modified_data, 2, axis=1)
    
    if mask2d is not None:
        modified_mask2d = np.repeat(mask2d, 2, axis=0)
        modified_mask2d = np.repeat(modified_mask2d, 2, axis=1)
    else:
        modified_mask2d = None
    
    return modified_xbin, modified_ybin, modified_data, modified_mask2d
    
def plot_3d_binned_bars(p, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold: float=None, **kwargs):
    """ Plots a 3D bar-graph
    Usage:
        plotActors, data_dict = plot_3d_binned_data(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    if drop_below_threshold is not None:
        # print(f'drop_below_threshold: {drop_below_threshold}')
        # active_data[np.where(active_data < drop_below_threshold)] = np.nan
        data_mask = (data.copy() < drop_below_threshold)
    else:
        data_mask = None
    
    modified_xbin, modified_ybin, modified_data, modified_mask2d = prepare_binned_data_for_3d_bars(xbin.copy(), ybin.copy(), data.copy(), mask2d=data_mask)
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(modified_xbin, modified_ybin)
    active_data = deepcopy(modified_data[:,:].T) # A single tuning curve
    # active_data = modified_data[:,:].copy() # A single tuning curve

    if modified_mask2d is not None:
        active_data_mask = modified_mask2d[:,:].T.copy()
        # print(f'Masking {len(np.where(active_data_mask))} of {np.size(active_data)} elements.')
        # apply the mask now:
        active_data[active_data_mask] = np.nan
    
    mesh = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    mesh["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)

    plot_name = kwargs.pop('plot_name', build_3d_plot_identifier_name('plot_3d_binned_bars', kwargs.get('name', ''))) # if 'plot_name' is provided, use that as the full name without modifications
    kwargs['name'] = plot_name # this is the only one to overwrite in kwargs
    # print(f'name: {plot_name}')    
    plotActor = p.add_mesh(mesh,
                            **({'show_edges': True, 'edge_color': 'k', 'nan_opacity': 0.0, 'scalars': 'Elevation', 'opacity': 1.0, 'use_transparency': False, 'smooth_shading': False, 'show_scalar_bar': False, 'render': True, 'reset_camera': False} | kwargs)
                          )
    # p.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
    
    plotActors = {plot_name: {'main': plotActor}}
    data_dict = {plot_name: { 
            'name':plot_name,
            'grid': mesh, 
            'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
            'active_data': active_data
        }
    }
    return plotActors, data_dict
    




def plot_3d_binned_bars_timeseries(p, xbin, ybin, t_bins, data, zScalingFactor=1.0, drop_below_threshold: float=None, **kwargs):
    """ Plots a a series of 3D bar-graphs representing data over time
    Usage:
        plotActors, data_dict = plot_3d_binned_data(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """

    assert np.ndim(data) > 2, f"np.ndim(data): {np.ndim(data)} but it should be a 3D timeseries"
    n_xbins, n_ybins, n_tbins = np.shape(data)
    assert n_tbins == len(t_bins), f"n_tbins: {n_tbins} != len(t_bins): {len(t_bins)}"

    all_plotActors_dict = {}
    all_data_dict = {}

    # from pyphoplacecellanalysis.Pho3D.PyVista.graphs import make_saturating_red_cmap
    # cmaps = [make_saturating_red_cmap((float(i) / float(n_tbins - 1)), min_alpha=1.0, max_alpha=1.0) for i in np.arange(n_tbins)] # pretty good

    # Load the existing 'viridis' colormap
    viridis = plt.cm.get_cmap('viridis')
    desaturation_factors = np.linspace(start=1.0, stop=0.1, num=n_tbins)
    cmaps = [desaturate_colormap(viridis, a_desaturation_factor) for a_desaturation_factor in desaturation_factors]
        
    for t_bin_idx, t_value in enumerate(t_bins):
        a_plot_name: str = f"plot_3d_binned_bars[{t_value}]"
        a_plotActors, a_data_dict = plot_3d_binned_bars(p=p, xbin=xbin, ybin=ybin, data=np.squeeze(data[:, :, t_bin_idx]),
                                                        zScalingFactor=zScalingFactor, drop_below_threshold=drop_below_threshold, plot_name=a_plot_name, 
                                                        **{'render': False,
                                                        #    'opacity': 1.0, #'use_transparency': True,
                                                            # 'cmap':'terrain', 'opacity':"linear",
                                                            # 'cmap':'viridis', 'opacity':"linear", # 'opacity':"sigmoid",
                                                            'cmap':cmaps[t_bin_idx], # , clim=(0, N)
                                                            },
                                                            
                                                        # **kwargs,
                                                        ) 
        # all_plotActors_dict[a_plot_name] = a_plotActors[a_plot_name]['main']
        all_plotActors_dict.update(a_plotActors)
        all_data_dict.update(a_data_dict)


    return all_plotActors_dict, all_data_dict


def clear_3d_binned_bars_plots(p, plotActors):
    """ usage:

    from pyphoplacecellanalysis.Pho3D.PyVista.graphs import clear_3d_binned_bars_plots
    clear_3d_binned_bars_plots(p=a_decoded_trajectory_pyvista_plotter.p, plotActors=a_decoded_trajectory_pyvista_plotter.plotActors)
    
    """
    if plotActors is None:
        return
    for k, v in plotActors.items():
        did_remove = p.remove_actor(v['main'])
        v.clear()
    plotActors.clear()
    return



    
    
## Point Labeling:

def _perform_plot_point_labels(p, active_points, point_labels=None, point_mask=None, **kwargs):
    if point_mask is not None:
        if callable(point_mask):
            point_masking_function = point_mask
            point_mask = point_masking_function(active_points) # if it is a masking function instead of a list of inclusion bools, build the concrete list by evaluating it for active_points
            # point_mask = [point_masking_function(a_point) for a_point in active_points] 
        assert (len(point_mask) == len(active_points)), "There must be one mask value in point_mask for each point in active_points!"
        active_points = active_points[point_mask] # apply the mask to the points
    
    if point_labels is None:
        point_labels = [f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})' for a_point in active_points]
    if callable(point_labels):
        point_labeling_function = point_labels
        point_labels = [point_labeling_function(a_point) for a_point in active_points] # if it is a formatting function instead of a list of labels, build the concrete list by evaluating it over the active_points
    assert (len(point_labels) == len(active_points)), "There must be one point label in point_labels for each point in active_points!"
    
    points_labels_actor = p.add_point_labels(active_points, point_labels,
                                                **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)
                                             )
    plotActors = {'main': points_labels_actor}
    data_dict = {
        'point_labels': point_labels,
        'points': active_points
    }
    return plotActors, data_dict 
    
def plot_point_labels(p, xbin_centers, ybin_centers, data, point_labels=None, point_mask=None, zScalingFactor=1.0, **kwargs):
    """ Plots 3D text point labels at the provided points.

    Args:
        p ([type]): [description]
        xbin_centers ([type]): an array of n_xbins
        ybin_centers ([type]): an array of n_ybins
        data ([type]): the height data of dimension (n_xbins, n_ybins)
        point_labels [str]: a set of labels of length equal to data to display on the points
        zScalingFactor (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
        
    Examples:
    # The full point shown:
    point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
    # Only the z-values
    point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'
    point_masking_function = lambda points: points[:, 2] > 20.0
    plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers, active_epoch_placefields2D.ratemap.occupancy, 
                                                                        point_labels=point_labeling_function, 
                                                                        point_mask=point_masking_function,
                                                                        shape='rounded_rect', shape_opacity= 0.2, show_points=False)

    """
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin_centers, ybin_centers)
    active_data = data[:,:].T.copy() # Copy the elevation/value data
    grid = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    points = grid.points
    
    plot_name = build_3d_plot_identifier_name('plot_point_labels', kwargs.get('name', 'main'))
    kwargs['name'] = plot_name # this is the only one to overwrite in kwargs
    plotActors_labels, data_dict_labels = _perform_plot_point_labels(p, points, point_labels=point_labels, point_mask=point_mask,
                                                                        **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)
                                                                    )
    # plotActors = {'main': plotActors_labels['main']}
    
    plotActors = {plot_name: plotActors_labels['main']}
    data_dict = {plot_name: { 
            'name':plot_name,
            'grid': grid, 
            'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
            'active_data': active_data
        } | data_dict_labels
    }
    
    return plotActors, data_dict


