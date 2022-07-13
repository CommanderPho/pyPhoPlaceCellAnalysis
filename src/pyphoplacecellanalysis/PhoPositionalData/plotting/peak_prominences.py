""" peak_prominence_mixins

"""

import numpy as np
import pyvista as pv # for building bounding boxes
from pyphoplacecellanalysis.PhoPositionalData.plotting.graphs import plot_point_labels, _perform_plot_point_labels
from pyphocorehelpers.gui.PyVista.CascadingDynamicPlotsList import CascadingDynamicPlotsList # used to wrap _render_peak_prominence_2d_results_on_pyvista_plotter's outputs



def _build_pyvista_single_neuron_prominence_result_data(neuron_id, a_result, promenence_plot_threshold = 1.0, included_level_indicies=[1], debug_print=False):
    """ Unfortunately I realize now that these centers, etc are all represented in terms of number of bins, not actual x, y values.
    
    Centers might be someting like:
    array([[49.2379, 18.2778, 5.2],
       [1.8755, 17.9038, 1.6]])
    """
    slab, peaks, idmap, promap, parentmap = a_result['slab'], a_result['peaks'], a_result['id_map'], a_result['prominence_map'], a_result['parent_map']
    n_peaks = len(peaks)
    if debug_print:
        print(f'neruon_id: {neuron_id} - : {n_peaks} peaks:')
    peak_locations = np.zeros((n_peaks, 3), dtype=float) # x, y, z for each peak
    prominence_array = np.zeros((n_peaks,), dtype=float)
    is_included_array = np.full((n_peaks,), False)
    peak_labels = []
    
    # only get the number of included levels
    n_levels = len(included_level_indicies)
    peak_levels = np.zeros((n_peaks, n_levels), dtype=float) # the list of level values for each peak
    peak_level_bboxes = np.zeros((n_peaks, n_levels, 4), dtype=float)
    
    for i, (peak_id, a_peak) in enumerate(peaks.items()):
        # loop through each of the peaks and extract their data
        if debug_print:
            print(f'peak_id: {peak_id}')
        prominence = a_peak['prominence']
        prominence_array[i] = prominence
        is_included = (prominence >= promenence_plot_threshold)
        is_included_array[i] = is_included
        if is_included:
            if debug_print:
                print(f'\tprominence: {prominence}')
                # print(f'\t# contours: {len(computed_contours)}')
            curr_slices = a_peak['level_slices']
            levels_list = list(curr_slices.keys())
            if included_level_indicies is not None:
                filtered_levels_list = [levels_list[i] for i in included_level_indicies]  
            else:
                filtered_levels_list = levels_list

            # curr_color = next(colors)
            peak_center = a_peak['center']
            peak_height = a_peak['height']                        
            peak_locations[i,[0,1]] = a_peak['center']
            peak_locations[i,2] = a_peak['height']
            if debug_print:
                print(f"\tcenter: {peak_center}")
                print(f"\theight: {peak_height}")
            
            peak_label = f'{peak_id}|{peak_height}|{prominence}'
            peak_labels.append(peak_label)
            
            peak_levels[i,:] = filtered_levels_list
            for level_idx, level_value in enumerate(filtered_levels_list):
                curr_slice = curr_slices[level_value]
                curr_contour = curr_slice['contour']
                if curr_contour is not None:
                    # ax.plot(curr_contour.vertices[:,0], curr_contour.vertices[:,1],':', color=curr_color)
                    bbox = curr_slice['bbox']
                    # (x0, y0, width, height) = bbox.bounds
                    peak_level_bboxes[i, level_idx, :] = bbox.bounds
                    
                else:
                    print(f"contour missing for neuron_id: {neuron_id} - peak_id: {peak_id} - slice[{level_value}]. Skipping.")
        else:
            print(f'\tskipping neuron_id: {neuron_id} - peak_id: {peak_id} because prominence: {prominence} is too low.')
    return peak_locations[is_included_array,:], prominence_array[is_included_array], peak_labels, peak_levels, peak_level_bboxes # peak_levels[is_included_array,:]
    # return peak_locations[is_included_array,:], colors[is_included_array], prominence_array[is_included_array] 
    # return peak_locations, colors, prominence_array, is_included_array
    

def _render_peak_prominence_2d_results_on_pyvista_plotter(ipcDataExplorer, active_peak_prominence_2d_results, valid_neuron_id=2, debug_print=True, **kwargs):
    """
    
    Built Data:
        peak_locations, prominence_array, peak_labels, peak_levels, flat_peak_levels, peak_level_bboxes
        
        out_pf_contours_data, out_pf_contours_actors, out_pf_box_data, out_pf_box_actors, out_pf_peak_points_data, out_pf_peak_points_actors
    
    """
    # valid_neuron_id = kwargs.get('neuron_id', 2)
    assert valid_neuron_id in active_peak_prominence_2d_results.results, f"neuron_id {valid_neuron_id} must be in the results keys, but it is not. results keys: {list(active_peak_prominence_2d_results.results.keys())}"
    peak_locations, prominence_array, peak_labels, peak_levels, peak_level_bboxes = _build_pyvista_single_neuron_prominence_result_data(valid_neuron_id, active_peak_prominence_2d_results.results[valid_neuron_id], promenence_plot_threshold = 0.2, included_level_indicies=[1], debug_print=debug_print)
    # ipcDataExplorer.tuning_curves_valid_neuron_ids
    # ipcDataExplorer.find_neuron_IDXs_from_cell_ids(cell_ids=[valid_neuron_id])
    # ipcDataExplorer.find_tuning_curve_IDXs_from_neuron_ids(neuron_ids=
    neuron_IDXs = ipcDataExplorer.find_tuning_curve_IDXs_from_neuron_ids(neuron_ids=[valid_neuron_id])
    assert len(neuron_IDXs) == 1, f"valid_neuron_id: {valid_neuron_id} should only return a single (exactly 1) neuron_IDX! Instead it returned neuron_IDXs: {neuron_IDXs}"
    neuron_IDX = neuron_IDXs[0]
    if debug_print:
        print(f'neuron_IDX: {neuron_IDX}')
        print(f'Original Locations:')
        print(f'peak_locations: {peak_locations}')
        print(f'prominence_array: {prominence_array}')
        print(f'peak_levels: {peak_levels}')
        print(f'peak_level_bboxes: {peak_level_bboxes}')
                                                          
    ## Outputs:
    
    # Contours/Isos
    out_pf_contours_data = {}
    out_pf_contours_actors = {}
    
    # Bounding Boxes:
    fixed_z_half_height = 0.10 # the fixed half-height of the box, centered around the contour level
    out_pf_box_data = {}
    out_pf_box_actors = {}
    
    # Text Labels:
    out_pf_text_size_data = {}
    out_pf_text_size_actors = {}
    
    # Peak Points:
    out_pf_peak_points_data = {}
    out_pf_peak_points_actors = {}

    ## Compute the appropriate z-scaling factors from the actual heights of the ipcDataExplorer's tuning_curves data:
    zScalingFactor = ipcDataExplorer.params.zScalingFactor
    # tuning_curves_normalization_factors = [1.0 / np.nanmax(a_tuning_curve) for a_tuning_curve in ipcDataExplorer.tuning_curves]
    # tuning_curves_apparent_height = [np.nanmax(a_tuning_curve)*zScalingFactor for a_tuning_curve in ipcDataExplorer.tuning_curves]
    tuning_curve_apparent_height = np.nanmax(ipcDataExplorer.tuning_curves[neuron_IDX])*zScalingFactor
    if debug_print:
        print(f'zScalingFactor: {zScalingFactor}, tuning_curve_apparent_height: {tuning_curve_apparent_height}')
    curr_scale_z = tuning_curve_apparent_height
    if debug_print:
        print(f'curr_scale_z: {curr_scale_z}')
    ## Convert the z components from unit-space to the z-space of the curve:
    peak_locations[:,2] = peak_locations[:,2] * curr_scale_z
    if debug_print:
        print(f'peak_locations: {peak_locations}')
    ## Also scale the peak_levels (which are also expressed as z-positions:
    peak_levels = peak_levels * curr_scale_z
    if debug_print:
        print(f'peak_levels: {peak_levels}')
    # Need to flatten the peak_levels
    flat_peak_levels = peak_levels.flatten()
            
    ### Add pyvista contours:
    curr_neuron_plot_data = ipcDataExplorer.plots_data.tuningCurvePlotData[valid_neuron_id]
    curr_pdata = curr_neuron_plot_data['pdata_currActiveNeuronTuningCurve']
    curr_contours_mesh_name = f'pf[{valid_neuron_id}]_contours'
    curr_contours = curr_pdata.contour(isosurfaces=ipcDataExplorer.params.zScalingFactor*flat_peak_levels) # I really don't know why we need to multiply by zScalingFactor (~2000.0) again.
    contours_mesh_actor = ipcDataExplorer.p.add_mesh(curr_contours, color="white", line_width=3, name=curr_contours_mesh_name) # should add it to the ipcDataExplorer's extant plotter (overlaying it on the current mesh
    out_pf_contours_data[curr_contours_mesh_name] = curr_contours
    out_pf_contours_actors[curr_contours_mesh_name] = contours_mesh_actor
    
    ### Add simple bounding boxes to the plot:
    if debug_print:
        print(f'np.shape(peak_level_bboxes): {np.shape(peak_level_bboxes)}') # (2, 1, 4)
    n_peaks = np.shape(peak_level_bboxes)[0]
    n_levels = np.shape(peak_level_bboxes)[1]
    for peak_idx in np.arange(n_peaks):
        for level_idx in np.arange(n_levels):
            curr_box_mesh_name = f'pf[{valid_neuron_id}]_peak[{peak_idx}]_lvl[{level_idx}]_bounds_box'
            (x0, y0, width, height) = peak_level_bboxes[peak_idx, level_idx, :]
            a_peak_level = peak_levels[peak_idx][level_idx]
            ## Can use a rectangle instead of a box:
            out_pf_box_data[curr_box_mesh_name] = pv.Rectangle([((x0+width), y0, a_peak_level), ((x0+width), (y0+height), a_peak_level), (x0, (y0+height), a_peak_level), (x0, y0, a_peak_level)])
            out_pf_box_actors[curr_box_mesh_name] = ipcDataExplorer.p.add_mesh(out_pf_box_data[curr_box_mesh_name], color="white",  name=curr_box_mesh_name, show_edges=True, edge_color="white", line_width=1.5, opacity=0.75, label=curr_box_mesh_name, style='wireframe')
            
            ## Box Mode:
            ## build the corner points of the box:
            ## box_bounds = (xMin, xMax, yMin, yMax, zMin, zMax)
            # box_bounds = (x0, (x0+width), y0, (y0+height), (a_peak_level-fixed_z_half_height), (a_peak_level+fixed_z_half_height))
            # if debug_print:
            #     print(f'peak_idx: {peak_idx} - level_idx: {level_idx} :: box_bounds (xMin, xMax, yMin, yMax, zMin, zMax): {box_bounds}')
#             out_pf_box_data[curr_box_mesh_name] = pv.Box(bounds=box_bounds, level=0, quads=True)
#             out_pf_box_actors[curr_box_mesh_name] = ipcDataExplorer.p.add_mesh(out_pf_box_data[curr_box_mesh_name], color="white",  name=curr_box_mesh_name, show_edges=True, edge_color="white", line_width=0.5, opacity=0.75, label=curr_box_mesh_name)
            
    
            ## Text Labels:
            curr_text_label_mesh_x_name = f'pf[{valid_neuron_id}]_peak[{peak_idx}]_lvl[{level_idx}]_text_label_x'
            curr_text_label_mesh_y_name = f'pf[{valid_neuron_id}]_peak[{peak_idx}]_lvl[{level_idx}]_text_label_y'
            
            x_center = (x0 + (x0+width))/2.0
            y_center = (y0 + (y0+height))/2.0
            
            x_text_mesh = pv.Text3D(f'{width:.2f}')
            y_text_mesh = pv.Text3D(f'{height:.2f}')
            
            # x_text_mesh = x_text_mesh.rotate_z(30, inplace=True)
            y_text_mesh = y_text_mesh.rotate_z(90, inplace=True)
            
            offset_center_point_x_mesh = x_text_mesh.center
            offset_center_point_y_mesh = y_text_mesh.center
            
            if debug_print:
                print(f'offset_center_point_x_mesh: {offset_center_point_x_mesh}')
            
            goal_point_x = (x_center, y0, a_peak_level)
            goal_point_y = (x0, y_center, a_peak_level)
            
            final_offset_x = (goal_point_x[0]-offset_center_point_x_mesh[0], goal_point_x[1]-offset_center_point_x_mesh[1], goal_point_x[2]-offset_center_point_x_mesh[2])
            final_offset_y = (goal_point_y[0]-offset_center_point_y_mesh[0], goal_point_y[1]-offset_center_point_y_mesh[1], goal_point_y[2]-offset_center_point_y_mesh[2])
            
            x_text_mesh = x_text_mesh.translate(final_offset_x, inplace=True)
            y_text_mesh = y_text_mesh.translate(final_offset_y, inplace=True)
            
            out_pf_text_size_data[curr_text_label_mesh_x_name] = x_text_mesh
            out_pf_text_size_data[curr_text_label_mesh_y_name] = y_text_mesh
            
            out_pf_text_size_actors[curr_text_label_mesh_x_name] = ipcDataExplorer.p.add_mesh(x_text_mesh, name=curr_text_label_mesh_x_name)
            out_pf_text_size_actors[curr_text_label_mesh_y_name] = ipcDataExplorer.p.add_mesh(y_text_mesh, name=curr_text_label_mesh_y_name)
            


    ### Add peak points:
    curr_peak_points_mesh_name = f'pf[{valid_neuron_id}]_prominence_peaks_points'
    point_labels = peak_labels.copy()
    points = peak_locations.copy()
    point_mask = None
    plotActors_labels, data_dict_labels = _perform_plot_point_labels(ipcDataExplorer.p, points, point_labels=point_labels, point_mask=point_mask,
                                                                            **({'font_size': 10, 'name':curr_peak_points_mesh_name,
                                                                                'shape_opacity': 0.1, 'shape_color':'grey', 'shape':'rounded_rect', 'fill_shape':True, 'margin':3,
                                                                                'show_points': False, 'point_size': 8, 'point_color':'white', 'render_points_as_spheres': True} | kwargs)
                                                                        )
    out_pf_peak_points_actors[curr_peak_points_mesh_name] = plotActors_labels['main']
    out_pf_peak_points_data[curr_peak_points_mesh_name] = {'name':curr_peak_points_mesh_name, 'active_data':{'peak_locations':peak_locations, 'point_labels':point_labels} | data_dict_labels}
    ## Build the final output structures:
    all_peaks_actors = CascadingDynamicPlotsList(contours=CascadingDynamicPlotsList(**out_pf_contours_actors), boxes=CascadingDynamicPlotsList(**out_pf_box_actors), text=CascadingDynamicPlotsList(**out_pf_text_size_actors), peak_points=CascadingDynamicPlotsList(**out_pf_peak_points_actors))
    all_peaks_data = dict(contours=out_pf_contours_data, boxes=out_pf_box_data, text=out_pf_text_size_data, peak_points=out_pf_peak_points_data)
    # return out_pf_contours_data, out_pf_contours_actors, out_pf_box_data, out_pf_box_actors, out_pf_text_size_data, out_pf_text_size_actors, out_pf_peak_points_data, out_pf_peak_points_actors
    return all_peaks_data, all_peaks_actors
    
"""
from pyphoplacecellanalysis.PhoPositionalData.plotting.peak_prominences import _render_peak_prominence_2d_results_on_pyvista_plotter

out_pf_contours_data, out_pf_contours_actors, out_pf_box_data, out_pf_box_actors, out_pf_text_size_data, out_pf_text_size_actors, out_pf_peak_points_data, out_pf_peak_points_actors = _render_peak_prominence_2d_results_on_pyvista_plotter(ipcDataExplorer, active_peak_prominence_2d_results, valid_neuron_id=12, debug_print=False)


self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor']


"""