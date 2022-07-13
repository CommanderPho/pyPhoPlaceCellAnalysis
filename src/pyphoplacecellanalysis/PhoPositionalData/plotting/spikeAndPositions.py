#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
from indexed import IndexedOrderedDict
import sys
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy import QtGui # for QColor

from pyphocorehelpers.gui.PyVista.CascadingDynamicPlotsList import CascadingDynamicPlotsList


# Fixed Geometry objects:
animal_location_sphere = pv.Sphere(radius=2.3)
animal_location_direction_cone = pv.Cone()
point_location_circle = pv.Circle(radius=8.0)
point_location_trail_circle = pv.Circle(radius=2.3)

## Spike indicator geometry:
spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
# spike_geom_cone = pv.Cone(direction=(0.0, 0.0, 1.0), height=15.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
spike_geom_circle = pv.Circle(radius=0.4)
spike_geom_box = pv.Box(bounds=[-0.2, 0.2, -0.2, 0.2, -0.05, 0.05])
# pv.Cylinder

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkLookupTable # required for build_custom_placefield_maps_lookup_table(...)




# ==================================================================================================================== #
# Main Animal Arena                                                                                                    #
# ==================================================================================================================== #

def _build_flat_arena_data(x, y, z=-0.01, smoothing=True, extrude_height=-5):
        # Builds the flat base maze map that the other data will be plot on top of
        ## Implicitly relies on: x, y
        # z = np.zeros_like(x)
        z = np.full_like(x, z) # offset just slightly in the z direction to account for the thickness of the caps that are added upon extrude
        point_cloud = np.vstack((x, y, z)).T
        pdata = pv.PolyData(point_cloud)
        pdata['occupancy heatmap'] = np.arange(np.shape(point_cloud)[0])
        # geo = pv.Circle(radius=0.5)
        # pc = pdata.glyph(scale=False, geom=geo)
        if smoothing:
            surf = pdata.delaunay_2d()
            surf = surf.extrude([0,0,extrude_height], capping=True, inplace=True)
            clipped_surf = surf.clip('-z', invert=False)
            return pdata, clipped_surf
        else:
            geo = pv.Circle(radius=0.5)
            pc = pdata.glyph(scale=False, geom=geo)
            return pdata, pc
        

def perform_plot_flat_arena(p, *args, z=-0.01, bShowSequenceTraversalGradient=False, smoothing=True, extrude_height=-5, **kwargs):
    """ Upgraded to render a much better looking 3D extruded maze surface.
    
        smoothing: whether or not to perform delaunay_2d() triangulation and output. Won't work on Z or N shaped mazes for example because it would close them.
            I think a "convex hull" operation might be okay though for these?
        
    """
    # Call with:
    # pdata_maze, pc_maze = build_flat_map_plot_data() # Plot the flat arena
    # p.add_mesh(pc_maze, name='maze_bg', color="black", render=False)

    if len(args) == 2:
        # normal x, y case
        x, y = args[0], args[1]
        pdata_maze, pc_maze = _build_flat_arena_data(x, y, z=z, smoothing=smoothing, extrude_height=extrude_height)

    elif len(args) == 1:
        # directly passing in pc_maze already built by calling _build_flat_arena_data case
        # Note that  z, smoothing=smoothing, extrude_height=extrude_height are ignored in this case
        pc_maze = args[0]
    else:
        raise ValueError

    # return p.add_mesh(pc_maze, name='maze_bg', label='maze', color="black", show_edges=False, render=True)
    return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'color': [0.1, 0.1, 0.1], 'pbr': True, 'metallic': 0.8, 'roughness': 0.5, 'diffuse': 1, 'render': True} | kwargs))
    # return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'color': [0.1, 0.1, 0.1, 1.0], 'pbr': True, 'metallic': 0.8, 'roughness': 0.5, 'diffuse': 1, 'render': True} | kwargs))
    # bShowSequenceTraversalGradient
    if bShowSequenceTraversalGradient:
        traversal_order_scalars = np.arange(len(x))
        return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'scalars': traversal_order_scalars, 'render': True} | kwargs))



# ==================================================================================================================== #
# Spikes                                                                                                               #
# ==================================================================================================================== #

# dataframe version of the build_active_spikes_plot_pointdata(...) function
def build_active_spikes_plot_pointdata_df(active_flat_df: pd.DataFrame, enable_debug_print=False):
    """Builds the pv.PolyData pointcloud from the spikes dataframe points.

    Args:
        active_flat_df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    if 'z' in active_flat_df.columns:
        # use custom override z-values
        print('build_active_spikes_plot_pointdata_df(...): Found custom z column! Using Data!!')
        assert np.shape(active_flat_df['z']) == np.shape(active_flat_df['x']), "custom z values must be the same shape as the x column"
        spike_history_point_cloud = active_flat_df[['x','y','z']].to_numpy()
    else:
        # no provided custom z value
        active_flat_df['z_fixed'] = np.full_like(active_flat_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it
        spike_history_point_cloud = active_flat_df[['x','y','z_fixed']].to_numpy()
        
    ## Old way:
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    spike_history_pdata['cellID'] = active_flat_df['aclu'].values
    
    if 'render_opacity' in active_flat_df.columns:
        spike_history_pdata['render_opacity'] = active_flat_df['render_opacity'].values
        # alternative might be repeating 4 times along the second dimension for no reason.
    else:
        print('no custom render_opacity set on dataframe.')
        
    # rebuild the RGB data from the dataframe:
    if (np.isin(['R','G','B','render_opacity'], active_flat_df.columns).all()):
        # RGB Only:
        # TODO: could easily add the spike_history_pdata['render_opacity'] here as RGBA if we wanted.
        # RGB+A:
        spike_history_pdata['rgb'] = active_flat_df[['R','G','B','render_opacity']].to_numpy()
        if enable_debug_print:
            print('successfully set custom rgb key from separate R, G, B columns in dataframe.')
    else:
        print('WARNING: DATAFRAME LACKS RGB VALUES!')

    return spike_history_pdata

# dataframe versions of the build_active_spikes_plot_data(...) function
def build_active_spikes_plot_data_df(active_flat_df: pd.DataFrame, spike_geom, enable_debug_print=False):
    """ 
    Usage:
        spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(active_flat_df, spike_geom)
    """
    spike_history_pdata = build_active_spikes_plot_pointdata_df(active_flat_df, enable_debug_print=enable_debug_print)
    spike_history_pc = spike_history_pdata.glyph(scale=False, geom=spike_geom.copy()) # create many glyphs from the point cloud
    return spike_history_pdata, spike_history_pc

## compatability with pre 2021-11-28 implementations
def build_active_spikes_plot_pointdata(active_flattened_spike_identities, active_flattened_spike_positions_list):
    # spike_series_times = active_flattened_spike_times # currently unused
    spike_series_identities = active_flattened_spike_identities # currently unused
    spike_series_positions = active_flattened_spike_positions_list
    # z = np.zeros_like(spike_series_positions[0,:])
    z_fixed = np.full_like(spike_series_positions[0,:], 1.1) # Offset a little bit in the z-direction so we can see it
    spike_history_point_cloud = np.vstack((spike_series_positions[0,:], spike_series_positions[1,:], z_fixed)).T
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    # spike_history_pdata['times'] = spike_series_times
    spike_history_pdata['cellID'] = spike_series_identities
    return spike_history_pdata

## compatability with pre 2021-11-28 implementations
def build_active_spikes_plot_data(active_flattened_spike_identities, active_flattened_spike_positions_list, spike_geom, scale_factors_list=None):
    # spike_series_times = active_flattened_spike_times # currently unused
    spike_history_pdata = build_active_spikes_plot_pointdata(active_flattened_spike_identities, active_flattened_spike_positions_list)
    # create many spheres from the point cloud
    if scale_factors_list is None:
        scale_variable_name = False
    else:
        # Add the scalars provided as scale factors:
        scale_variable_name = 'age_scale_factors'
        spike_history_pdata[scale_variable_name] = scale_factors_list
        
    spike_history_pc = spike_history_pdata.glyph(scale=scale_variable_name, geom=spike_geom.copy())
    return spike_history_pdata, spike_history_pc






# ==================================================================================================================== #
# Placefields                                                                                                          #
# ==================================================================================================================== #
    
def build_custom_placefield_maps_lookup_table(curr_active_neuron_color, num_opacity_tiers, opacity_tier_values):
    """
    Inputs:
        curr_active_neuron_color: an RGBA value
    Usage:
        
        build_custom_placefield_maps_lookup_table(curr_active_neuron_color, 3, [0.0, 0.6, 1.0])
    """
    # opacity_tier_values: [0.0, 0.6, 1.0]
    # Build a simple lookup table of the curr_active_neuron_color with varying opacities
    
    if isinstance(curr_active_neuron_color, (tuple, list)):
        curr_active_neuron_color = np.array(curr_active_neuron_color)
    
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(num_opacity_tiers)
    for i in np.arange(num_opacity_tiers):
        map_curr_active_neuron_color = curr_active_neuron_color.copy()
        map_curr_active_neuron_color[3] = opacity_tier_values[i]
        # print('map_curr_active_neuron_color: {}'.format(map_curr_active_neuron_color))
        lut.SetTableValue(i, map_curr_active_neuron_color)
    return lut

def force_plot_ignore_scalar_as_color(plot_mesh_actor, lookup_table):
        """The following custom lookup table solution is required to successfuly plot the surfaces with opacity dependant on their scalars property and still have a consistent color (instead of using the scalars for the color too). Note that the previous "fix" for the problem of the scalars determining the object's color when I don't want them to:
        Args:
            plot_mesh_actor ([type]): [description]
            lookup_table ([type]): a lookup_table as might be built with: `build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.0, 0.6, 1.0])`
        """
        # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 5, [0.0, 0.0, 0.3, 0.5, 0.1])
        lookup_table.SetTableRange(plot_mesh_actor.GetMapper().GetScalarRange())
        lookup_table.Build()
        plot_mesh_actor.GetMapper().SetLookupTable(lookup_table)
        plot_mesh_actor.GetMapper().SetScalarModeToUsePointData()

def plot_placefields2D(pTuningCurves, active_placefields, pf_colors: np.ndarray, zScalingFactor=10.0, show_legend=False, enable_debug_print=False, **kwargs):
    """ Plots 2D (as opposed to linearized/1D) Placefields in a 3D PyVista plot """
    # active_placefields: Pf2D    

    params = ({'should_use_normalized_tuning_curves':True, # Default True
        'should_pdf_normalize_manually':False, # Default False.
        'should_nan_non_visited_elements':False, # Default False. If True, sets the non-visited portions of the placefield to np.NaN before plotting.
        'should_force_placefield_custom_color':True, # Default True    
        'should_display_placefield_points':True, # Default True, whether to redner the individual points of the placefield
        'nan_opacity':0.0,
        } | kwargs)
        
    if params['should_use_normalized_tuning_curves']:
        curr_tuning_curves = active_placefields.ratemap.normalized_tuning_curves.copy()
    else:
        curr_tuning_curves = active_placefields.ratemap.tuning_curves.copy()
        
    if params['should_nan_non_visited_elements']:
        non_visited_mask = active_placefields.never_visited_occupancy_mask
        curr_tuning_curves[:, non_visited_mask] = np.nan # set all non-visited elements to NaN

    if np.shape(pf_colors)[1] > 3:
        opaque_pf_colors = pf_colors[0:3,:].copy() # get only the RGB values, discarding any potnential alpha information
    else:
        opaque_pf_colors = pf_colors.copy()
        
    # curr_tuning_curves[curr_tuning_curves < 0.1] = np.nan
    # curr_tuning_curves = curr_tuning_curves * zScalingFactor
    
    num_curr_tuning_curves = len(curr_tuning_curves)
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    tuningCurvePlot_x, tuningCurvePlot_y = np.meshgrid(active_placefields.ratemap.xbin_centers, active_placefields.ratemap.ybin_centers)
    # Loop through the tuning curves and plot them:
    if enable_debug_print:
        print('num_curr_tuning_curves: {}'.format(num_curr_tuning_curves))
        
    tuningCurvePlotActors = IndexedOrderedDict({})
    tuningCurvePlotData = IndexedOrderedDict({}) # TODO: try to convert to an ordered dict indexed by neuron_IDs
    for i in np.arange(num_curr_tuning_curves):
        #TODO: BUG: CRITICAL: Very clearly makes sense how the indexing gets off here:
        curr_active_neuron_ID = good_placefield_neuronIDs[i]
        curr_active_neuron_color = pf_colors[:, i]
        curr_active_neuron_opaque_color = opaque_pf_colors[:,i]
        curr_active_neuron_pf_identifier = 'pf[{}]'.format(curr_active_neuron_ID)
        curr_active_neuron_tuning_Curve = np.squeeze(curr_tuning_curves[i,:,:]).T.copy() # A single tuning curve
        
        if params['should_pdf_normalize_manually']:
            # Normalize the area under the curve to 1.0 (like a probability density function)
            curr_active_neuron_tuning_Curve = curr_active_neuron_tuning_Curve / np.nansum(curr_active_neuron_tuning_Curve)
            
        curr_active_neuron_tuning_Curve = curr_active_neuron_tuning_Curve * zScalingFactor
        
        # curr_active_neuron_tuning_Curve[curr_active_neuron_tuning_Curve < 0.1] = np.nan
        pdata_currActiveNeuronTuningCurve = pv.StructuredGrid(tuningCurvePlot_x, tuningCurvePlot_y, curr_active_neuron_tuning_Curve)
        pdata_currActiveNeuronTuningCurve["Elevation"] = (curr_active_neuron_tuning_Curve.ravel(order="F") * zScalingFactor)
        
        # Extracting Points from recently built StructuredGrid pdata:
        if params['should_display_placefield_points']:
            pdata_currActiveNeuronTuningCurve_Points = pdata_currActiveNeuronTuningCurve.extract_points(pdata_currActiveNeuronTuningCurve.points[:, 2] > 0)  # UnstructuredGrid
        else:
            pdata_currActiveNeuronTuningCurve_Points = None

        curr_active_neuron_plot_data = {'curr_active_neuron_ID':curr_active_neuron_ID,
                                         'curr_active_neuron_pf_identifier':curr_active_neuron_pf_identifier,
                                         'curr_active_neuron_tuning_Curve':curr_active_neuron_tuning_Curve,
                                         'pdata_currActiveNeuronTuningCurve':pdata_currActiveNeuronTuningCurve, 'pdata_currActiveNeuronTuningCurve_Points':pdata_currActiveNeuronTuningCurve_Points,
                                         'lut':None}
        
        # contours_currActiveNeuronTuningCurve = pdata_currActiveNeuronTuningCurve.contour()
        # pdata_currActiveNeuronTuningCurve.plot(show_edges=True, show_grid=True, cpos='xy', scalars=curr_active_neuron_tuning_Curve.T)        
        # actor_currActiveNeuronTuningCurve = pTuningCurves.add_mesh(pdata_currActiveNeuronTuningCurve, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier, show_edges=False, nan_opacity=0.0, color=curr_active_neuron_color, use_transparency=True)

        # surf = poly.delaunay_2d()
        # pTuningCurves.add_mesh(surf, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier, show_edges=False, nan_opacity=0.0, color=curr_active_neuron_color, opacity=0.9, use_transparency=False, smooth_shading=True)
        if params['should_force_placefield_custom_color']:
            curr_opacity = 'sigmoid'
            curr_smooth_shading = True
        else:
            curr_opacity = None
            curr_smooth_shading = False
            
        # curr_opacity = None
        
        if params['should_nan_non_visited_elements']:
            # To prevent artifacts after NaNing non-visited elements (black rendering faces around the edges that connect the NaN and non-NaN points that result from averaging the two faces, we must disable smooth_shading in this mode:
            curr_smooth_shading = False
        
        
        if params.get('should_override_disable_smooth_shading', False):
            curr_smooth_shading = False # override smooth shading if this option is set
        
        pdata_currActiveNeuronTuningCurve_plotActor = pTuningCurves.add_mesh(pdata_currActiveNeuronTuningCurve, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier,
                                                                            show_edges=True, edge_color=curr_active_neuron_opaque_color, nan_opacity=params['nan_opacity'], scalars='Elevation',
                                                                            opacity=curr_opacity, use_transparency=True, smooth_shading=curr_smooth_shading, show_scalar_bar=False, pickable=True, render=False)                                                                     
        
        # Force custom colors:
        if params['should_force_placefield_custom_color']:
            ## The following custom lookup table solution is required to successfuly plot the surfaces with opacity dependant on their scalars property and still have a consistent color (instead of using the scalars for the color too). Note that the previous "fix" for the problem of the scalars determining the object's color when I don't want them to:
                #   pdata_currActiveNeuronTuningCurve_plotActor.GetMapper().ScalarVisibilityOff() # Scalars not used to color objects
            # Is NOT Sufficient, as it disables any opacity at all seemingly
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 2, [0.2, 0.8])
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 1, [1.0]) # DFEFAULT: Full fill opacity
            lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 1, [0.5]) # ALT: reduce fill opacity
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.2, 0.6, 1.0]) # Looks great
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.0, 0.6, 1.0])
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 5, [0.0, 0.0, 0.3, 0.5, 0.1])
            curr_active_neuron_plot_data['lut'] = lut
            force_plot_ignore_scalar_as_color(pdata_currActiveNeuronTuningCurve_plotActor, lut)
            
            
        ## Add points:
        
        if params['should_display_placefield_points']:
            pdata_currActiveNeuronTuningCurve_Points_plotActor = pTuningCurves.add_points(pdata_currActiveNeuronTuningCurve_Points, label=f'{curr_active_neuron_pf_identifier}_points', name=f'{curr_active_neuron_pf_identifier}_points',
                                                                                    render_points_as_spheres=True, point_size=4.0, color=curr_active_neuron_opaque_color, render=False)    
        
        else:
            pdata_currActiveNeuronTuningCurve_Points_plotActor = None
        
        ## Build CascadingDynamicPlotsList Wrapper:
        currActiveNeuronTuningCurve_plotActors = CascadingDynamicPlotsList(active_main_plotActor=pdata_currActiveNeuronTuningCurve_plotActor, active_points_plotActor=pdata_currActiveNeuronTuningCurve_Points_plotActor)
        
        ## Built Multiplotter Wrapper:
        # data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        # blocks = pv.MultiBlock(data)

        # Merge the two actors together:
        # merged = pdata_currActiveNeuronTuningCurve.merge([pdata_currActiveNeuronTuningCurve_Points])
        tuningCurvePlotActors[curr_active_neuron_ID] = currActiveNeuronTuningCurve_plotActors
        tuningCurvePlotData[curr_active_neuron_ID] = curr_active_neuron_plot_data
        
    # Legend:
    plots_data = {'good_placefield_neuronIDs': good_placefield_neuronIDs,
                'unit_labels': ['{}'.format(good_placefield_neuronIDs[i]) for i in np.arange(num_curr_tuning_curves)],
                 'legend_entries': [['pf[{}]'.format(good_placefield_neuronIDs[i]), opaque_pf_colors[:,i]] for i in np.arange(num_curr_tuning_curves)]}
    
    # lost the ability to have colors with alpha components
        # TypeError: SetEntry argument 4: expected a sequence of 3 values, got 4 values
        
    # lost the ability to specify exact origins in add_legend() # used to be origin=[0.95, 0.1]

    if show_legend:
        legendActor = pTuningCurves.add_legend(plots_data['legend_entries'], name='tuningCurvesLegend', 
                                bcolor=(0.05, 0.05, 0.05), border=True,
                                loc='center right', size=[0.05, 0.85]) # vtk.vtkLegendBoxActor
        
        # used to be origin=[0.95, 0.1]
        
    else:
        legendActor = None
    
    return pTuningCurves, tuningCurvePlotActors, tuningCurvePlotData, legendActor, plots_data

def update_plotColorsPlacefield2D(tuningCurvePlotActors, tuningCurvePlotData, neuron_id_color_update_dict):
    """ Updates the colors of the placefields plots from the neuron_id_color_update_dict
    
    Inputs:
        tuningCurvePlotData: IndexedOrderedDict of neuron_id, plot data dict
    """
    for neuron_id, color in neuron_id_color_update_dict.items():
        ## Convert color to a QColor for generality:    
        if isinstance(color, QtGui.QColor):
            # already a QColor, just pass
            converted_color = color
        elif isinstance(color, str):
            # if it's a string, convert it to QColor
            converted_color = QtGui.QColor(color)
        elif isinstance(color, (tuple, list, np.array)):
            # try to convert it, hope it's the right size and stuff
            converted_color = QtGui.QColor(color)
        else:
            print(f'ERROR: Color is of unknown type: {color}, type: {type(color)}')
            raise NotImplementedError
        
        rgba_color = converted_color.getRgbF()
        rgb_color = rgba_color[:3]
        
        # Update the surface color itself:
        tuningCurvePlotData[neuron_id]['lut'] = build_custom_placefield_maps_lookup_table(rgba_color, 1, [0.5]) # ALT: reduce fill opacity
        pdata_currActiveNeuronTuningCurve_plotActor = tuningCurvePlotActors[neuron_id]['main'] # get the main plot actor from the CascadingDynamicPlotsList
        force_plot_ignore_scalar_as_color(pdata_currActiveNeuronTuningCurve_plotActor, tuningCurvePlotData[neuron_id]['lut'])
        
        ## Set color of the edges on the placefield surface (edge_color)
        pdata_currActiveNeuronTuningCurve_plotActor.GetProperty().SetEdgeColor(rgb_color)
        
        # set the color of the points on the placefield surface:
        pdata_currActiveNeuronTuningCurve_Points_plotActor = tuningCurvePlotActors[neuron_id]['points']
        pdata_currActiveNeuronTuningCurve_Points_plotActor.GetProperty().SetColor(rgb_color)
        
def update_plotVisiblePlacefields2D(tuningCurvePlotActors, isTuningCurveVisible):
    # Updates the visible placefields. Complements plot_placefields2D
    num_active_tuningCurveActors = len(tuningCurvePlotActors)
    for i in np.arange(num_active_tuningCurveActors):
        # tuningCurvePlotActors[i].SetVisibility(isTuningCurveVisible[i])
        if isTuningCurveVisible[i]:
            # tuningCurvePlotActors[i].show_actor()
            # tuningCurvePlotActors[i].SetVisibility(True)
            tuningCurvePlotActors[i].VisibilityOn()
        else:
            tuningCurvePlotActors[i].VisibilityOff()
            # tuningCurvePlotActors[i].hide_actor()
    
    
    
# ==================================================================================================================== #
# Misc. Light Spawning Effect                                                                                          #
# ==================================================================================================================== #
## This light effect occurs when a spike happens to indicate its presence
light_spawn_constant_z_offset = 2.5
light_spawn_constant_z_focal_position = -0.5 # by default, the light focuses under the floor

def build_spike_spawn_effect_light_actor(p, spike_position, spike_unit_color='white'):
    # spike_position: should be a tuple like (0, 0, 10)
    light_source_position = spike_position
    light_source_position[3] = light_source_position[3] + light_spawn_constant_z_offset
    light_focal_point = spike_position
    light_focal_point[3] = light_focal_point[3] + light_spawn_constant_z_focal_position
    
    SpikeSpawnEffectLight = pv.Light(position=light_source_position, focal_point=light_focal_point, color=spike_unit_color)
    SpikeSpawnEffectLight.positional = True
    SpikeSpawnEffectLight.cone_angle = 40
    SpikeSpawnEffectLight.exponent = 10
    SpikeSpawnEffectLight.intensity = 3
    SpikeSpawnEffectLight.show_actor()
    p.add_light(SpikeSpawnEffectLight)
    return SpikeSpawnEffectLight # return the light actor for removal later

