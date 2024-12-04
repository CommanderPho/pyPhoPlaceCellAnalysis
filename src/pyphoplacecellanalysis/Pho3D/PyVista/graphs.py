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
    assert (np.shape(data)[0] * 2) == np.shape(modified_xbin)[0], f"There shoud be double the number of xbins in the modified array as there are data points in the array but (np.shape(data)[0] * 2): {(np.shape(data)[0] * 2)} != np.shape(modified_xbin)[0]: {np.shape(modified_xbin)[0]}."
    assert (np.shape(data)[1] * 2) == np.shape(modified_ybin)[0], f"There shoud be double the number of ybins in the modified array as there are data points in the array but (np.shape(data)[1] * 2): {(np.shape(data)[1] * 2)} != np.shape(modified_ybin)[0]: {np.shape(modified_ybin)[0]}."
    
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



def plot_3d_stem_points(p, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold: float=None, enable_drawing_stem_lines:bool=False, **kwargs):
    """ Plots a 3D stem-plots with points. I'd like colored dots for each point in a 2D matrix, with their height (z-axis) representing the value and their color to reflect how recently they were updated.

    Usage:
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_stem_points
        plotActors, data_dict = plot_3d_stem_points(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    debug_print: bool = kwargs.pop('debug_print', False)

    if drop_below_threshold is not None:
        # print(f'drop_below_threshold: {drop_below_threshold}')
        # active_data[np.where(active_data < drop_below_threshold)] = np.nan
        data_mask = (data.copy() < drop_below_threshold)
    else:
        data_mask = None
    
    if debug_print:
        print(f'np.shape(data): {np.shape(data)}')

    if len(np.shape(data)) == 3:
        n_xbins, n_ybins, n_tbins = np.shape(data)
    elif len(np.shape(data)) == 2:
        n_xbins, n_ybins = np.shape(data)
    else:
        raise NotImplementedError(np.shape(data))

    # modified_xbin, modified_ybin, modified_data, modified_mask2d = prepare_binned_data_for_3d_bars(xbin.copy(), ybin.copy(), data.copy(), mask2d=data_mask)

    # # build a structured grid out of the bins
    # twoDimGrid_x, twoDimGrid_y = np.meshgrid(modified_xbin, modified_ybin)
    # active_data = deepcopy(modified_data[:,:].T) # A single tuning curve
    # # active_data = modified_data[:,:].copy() # A single tuning curve
    active_data = deepcopy(data)
    
    data = active_data
    # mesh = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    # mesh["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)


    # Initialize recency colors
    recency = np.zeros((n_xbins, n_ybins))

    # Coordinates
    # twoDimGrid_x, twoDimGrid_y = np.meshgrid(np.arange(n_ybins), np.arange(n_xbins))
    # twoDimGrid_x, twoDimGrid_y = np.meshgrid(np.arange(n_xbins), np.arange(n_ybins))
    # twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin[1:], ybin[1:])

    assert (len(xbin) == n_xbins), f"len(xbin): {len(xbin)} != n_xbins: {n_xbins}! Pass in xbin_center and ybin_center for this function instead of xbin/ybin."

    twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin, ybin)
    
    # twoDimGrid_x, twoDimGrid_y = np.meshgrid(np.squeeze(xbin), np.squeeze(ybin))


    points = np.column_stack((twoDimGrid_x.flatten(), twoDimGrid_y.flatten()))

    if debug_print:
        print(f'n_xbins: {n_xbins}, n_ybins: {n_ybins}')
        print(f'xbin: {np.shape(xbin)}, ybin: {np.shape(ybin)}')
        print(f'twoDimGrid_x: {np.shape(twoDimGrid_x)}, twoDimGrid_y: {np.shape(twoDimGrid_y)}')
        print(f'points: {np.shape(points)}')
    

    # Initialize the point cloud
    if len(np.shape(active_data)) == 3:
        z = active_data[:, :, 0].flatten()
    elif len(np.shape(active_data)) == 2:
        z = active_data[:, :].flatten()
    else:
        raise NotImplementedError(np.shape(data))

    if debug_print:   
        print(f'z: {np.shape(z)}')

    point_cloud = pv.PolyData(np.column_stack((points, z))) # ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 399 and the array at index 1 has size 336
    point_cloud['colors'] = recency.flatten()


    plot_name = kwargs.pop('plot_name', build_3d_plot_identifier_name('plot_3d_stem_points', kwargs.get('name', ''))) # if 'plot_name' is provided, use that as the full name without modifications
    kwargs['name'] = plot_name # this is the only one to overwrite in kwargs
    # print(f'name: {plot_name}')    

    ## override heatmp
    # kwargs['cmap'] = 'hot'
    kwargs['cmap'] = 'fire'

    plotActor = p.add_mesh(point_cloud, render_points_as_spheres=True, point_size=10, scalars='colors', cmap=kwargs.pop('cmap','fire'),
                            **({'nan_opacity': 0.0, 'smooth_shading': False, 'show_scalar_bar': True, 'render': True, 'reset_camera': False} | kwargs) # 'scalars': 'colors', 'opacity': 1.0, 'use_transparency': False, 'show_edges': True, 'edge_color': 'k', 
                          )

    # Add stems
    lines = []
    if enable_drawing_stem_lines:
        for i in range(points.shape[0]):
            line = pv.Line([points[i, 0], points[i, 1], 0], [points[i, 0], points[i, 1], z[i]])
            _out = p.add_mesh(line, color='black', name=f'stem_{i}')
            lines.append(line)


    def update_plot(value):
        time_bin = int(value)
        nonlocal recency

        # Update the z values and recency
        # z = active_data[:, :, time_bin].flatten()
        # if time_bin > 0:
        #     changes = active_data[:, :, time_bin] != active_data[:, :, time_bin - 1]
        #     recency[changes] = 1


        if len(np.shape(active_data)) == 3:
            z = active_data[:, :, time_bin].flatten()
            if time_bin > 0:
                changes = active_data[:, :, time_bin] != active_data[:, :, time_bin - 1]
                recency[changes] = 1
                
        elif len(np.shape(active_data)) == 2:
            z = active_data[:, :].flatten()
            if time_bin > 0:
                changes = active_data[:, time_bin] != active_data[:, time_bin - 1] # #TODO 2024-05-23 08:16: - [ ] Does this 2D mode have a time-bin index? How else could it work?
                recency[changes] = 1
        else:
            raise NotImplementedError(np.shape(data))
        
        recency *= 0.95  # Cooling effect

        # Update point cloud
        new_points = np.column_stack((points, z))
        point_cloud.points = new_points
        point_cloud['colors'] = recency.flatten()

        # Update stems
        for i, line in enumerate(lines):
            new_line = pv.Line([points[i, 0], points[i, 1], 0], [points[i, 0], points[i, 1], z[i]])
            p.remove_actor(f'stem_{i}')
            p.add_mesh(new_line, color='black', name=f'stem_{i}')
            lines[i] = new_line

        p.update_scalars(recency.flatten(), mesh=point_cloud)


    # p.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
    
    plotActors = {plot_name: {'main': plotActor}}
    data_dict = {plot_name: { 
            'name':plot_name,
            'point_cloud': point_cloud, 
            'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
            'active_data': active_data,
            'update_plot_fn': update_plot,
        }
    }
    return plotActors, data_dict
    




def plot_3d_binned_bars_timeseries(p, xbin, ybin, t_bins, data, zScalingFactor=1.0, drop_below_threshold: float=None, active_plot_fn=plot_3d_stem_points, **kwargs):
    """ Plots a a series of 3D bar-graphs representing data over time
    Usage:
        plotActors, data_dict = plot_3d_binned_data(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers

    assert np.ndim(data) > 2, f"np.ndim(data): {np.ndim(data)} but it should be a 3D timeseries"
    n_xbins, n_ybins, n_tbins = np.shape(data)
    assert n_tbins == len(t_bins), f"n_tbins: {n_tbins} != len(t_bins): {len(t_bins)}"

    all_plotActors_dict = {}
    all_data_dict = {}

    cmaps = [ColormapHelpers.make_saturating_red_cmap((float(i) / float(n_tbins - 1)), min_alpha=0.75, max_alpha=1.0) for i in np.arange(n_tbins)] # pretty good

    # # Load the existing 'viridis' colormap
    # viridis = plt.cm.get_cmap('viridis')
    # desaturation_factors = np.linspace(start=1.0, stop=0.1, num=n_tbins)
    # cmaps = [ColormapHelpers.desaturate_colormap(viridis, a_desaturation_factor) for a_desaturation_factor in desaturation_factors]
        
    for t_bin_idx, t_value in enumerate(t_bins):
        a_plot_name: str = f"plot_3d_binned_bars[{t_value}]"
        a_plotActors, a_data_dict = active_plot_fn(p=p, xbin=xbin, ybin=ybin, data=np.squeeze(data[:, :, t_bin_idx]),
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


