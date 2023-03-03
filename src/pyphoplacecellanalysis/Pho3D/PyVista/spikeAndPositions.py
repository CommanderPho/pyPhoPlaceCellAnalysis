#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


TODO: REORGANIZE_PLOTTER_SCRIPTS: PyVista


## Seems to primarily be a PyVista (pv) helper functions file


"""
import sys
from warnings import warn
import pyvista as pv
import numpy as np
import pandas as pd


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






# ==================================================================================================================== #
# Main Animal Arena                                                                                                    #
# ==================================================================================================================== #
# """
#     Note that perform_plot_flat_arena(...) is widely imported and used throughout the packages, and it would be difficult to move it.
# """


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

# Modern (2022+) Functions: dataframe versions  __________________________________________________________________ #
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
        # no provided custom z override value ...
        if 'z_fixed' not in active_flat_df.columns:
            # ... and no previously built 'z_fixed' column:
            active_flat_df['z_fixed'] = np.full_like(active_flat_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it
        else:
            # ... but has a previously built 'z_fixed' column:
            assert np.shape(active_flat_df['z_fixed']) == np.shape(active_flat_df['x']), "previously built-z_fixed column must be the same shape as the x column! TODO: probably just rebuild!"
            active_flat_df['z_fixed'] = np.full_like(active_flat_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it
        ## Now have a 'z_fixed' column, so use it:
        spike_history_point_cloud = active_flat_df[['x','y','z_fixed']].to_numpy()
        
    ## Old way:
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    spike_history_pdata['cellID'] = active_flat_df['aclu'].values
    
    if 'render_opacity' in active_flat_df.columns:
        spike_history_pdata['render_opacity'] = active_flat_df['render_opacity'].values
        # alternative might be repeating 4 times along the second dimension for no reason.
    else:
        warn('WARNING: no custom render_opacity set on dataframe.')
        
    # rebuild the RGB data from the dataframe:
    if (np.isin(['R','G','B','render_opacity'], active_flat_df.columns).all()):
        # RGB Only:
        # TODO: could easily add the spike_history_pdata['render_opacity'] here as RGBA if we wanted.
        # RGB+A:
        spike_history_pdata['rgb'] = active_flat_df[['R','G','B','render_opacity']].to_numpy()
        if enable_debug_print:
            print('successfully set custom rgb key from separate R, G, B columns in dataframe.')
    else:
        warn('WARNING: DATAFRAME LACKS RGB VALUES!')
    return spike_history_pdata

# dataframe versions of the build_active_spikes_plot_data(...) function
def build_active_spikes_plot_data_df(active_flat_df: pd.DataFrame, spike_geom, enable_debug_print=False):
    """
    Usage:
        spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(active_flat_df, spike_geom)

    Known Uses:
        SpikeRenderingPyVistaMixin.plot_spikes(...)
        SpikeRenderingPyVistaMixin.update_spikes(...)
            
    """
    spike_history_pdata = build_active_spikes_plot_pointdata_df(active_flat_df, enable_debug_print=enable_debug_print)
    spike_history_pc = spike_history_pdata.glyph(scale=False, geom=spike_geom.copy()) # create many glyphs from the point cloud
    return spike_history_pdata, spike_history_pc


# Old Functions: compatability with pre 2021-11-28 implementations __________________________________________________________________ #
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

