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

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from pyphoplacecellanalysis.PhoPositionalData.plotting.saving import save_to_multipage_pdf

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkLookupTable # required for build_custom_placefield_maps_lookup_table(...)

# ==================================================================================================================== #
# 1D Placefields and Tuning Curves                                                                                     #
# ==================================================================================================================== #

def plot_placefield_tuning_curve(xbin_centers, tuning_curve, ax, is_horizontal=False, color='g'):
    """ Plots the 1D Normalized Tuning Curve in a 2D Plot
    Usage:
        axs1 = plot_placefield_tuning_curve(active_epoch_placefields1D.ratemap.xbin_centers, active_epoch_placefields1D.ratemap.normalized_tuning_curves[curr_cell_id, :].squeeze(), axs1)
    """
    if is_horizontal:
        ax.fill_betweenx(xbin_centers, tuning_curve, color=color, alpha=0.3, interpolate=True)
        ax.plot(tuning_curve, xbin_centers, color, alpha=0.8)
    else:
        ax.fill_between(xbin_centers, tuning_curve, color=color, alpha=0.3)
        ax.plot(xbin_centers, tuning_curve, color, alpha=0.8)
    return ax

def _plot_helper_build_jittered_spike_points(curr_cell_spike_times, curr_cell_interpolated_spike_curve_values, jitter_multiplier=2.0, feature_range=(-1, 1), time_independent_jitter=False):
    """ jitters the curve_value for each spike based on the time it occured along the curve or a time_independent positive jitter """
    if time_independent_jitter:
        jitter_add = np.abs(np.random.randn(len(curr_cell_spike_times))) * jitter_multiplier
    else:
        # jitter the spike points based on the time they occured.
        jitter_add = jitter_multiplier * minmax_scale(curr_cell_spike_times, feature_range=feature_range)
    return curr_cell_interpolated_spike_curve_values + jitter_add

def _plot_helper_setup_gridlines(ax, bin_edges, bin_centers):
    ax.set_yticks(bin_edges, minor=False)
    ax.set_yticks(bin_centers, minor=True)
    ax.yaxis.grid(True, which='major', color = 'grey', linewidth = 0.5) # , color = 'green', linestyle = '--', linewidth = 0.5
    ax.yaxis.grid(True, which='minor', color = 'grey', linestyle = '--', linewidth = 0.25)

def plot_1d_placecell_validations(active_placefields1D, plotting_config, should_save=False, modifier_string='', save_mode='separate_files'):
    """ Uses plot_1D_placecell_validation(...) to plot a series of plots, one for each potential placecell, that allows you to see how the spiking corresponds to the animal's position/lap and how that contributes to the computed placemap
    
    Usage:
        plot_1d_placecell_validations(active_epoch_placefields1D, should_save=True)
        plot_1d_placecell_validations(active_epoch_placefields1D, modifier_string='lap_only', should_save=False)

    """
    # def _filename_for_placefield(active_epoch_placefields1D, curr_cell_id):
    #     return active_epoch_placefields1D.str_for_filename(is_2D=False) + '-cell_{:02d}'.format(curr_cell_id)
    
    n_cells = active_placefields1D.ratemap.n_neurons
    out_figures_list = []
    out_axes_list = []
    
    if should_save:
        curr_parent_out_path = plotting_config.active_output_parent_dir.joinpath('1d Placecell Validation')
        curr_parent_out_path.mkdir(parents=True, exist_ok=True)        
        
    for i in np.arange(n_cells):
        curr_cell_id = active_placefields1D.cell_ids[i]
        fig, axs = plot_1D_placecell_validation(active_placefields1D, i)
        out_figures_list.append(fig)
        out_axes_list.append(axs)

    # once done, save out as specified
    if should_save:
        common_basename = active_placefields1D.str_for_filename(prefix_string=modifier_string)
        if save_mode == 'separate_files':
            # make a subdirectory for this run (with these parameters and such)
            curr_specific_parent_out_path = curr_parent_out_path.joinpath(common_basename)
            curr_specific_parent_out_path.mkdir(parents=True, exist_ok=True)
            print(f'Attempting to write {n_cells} separate figures to {str(curr_specific_parent_out_path)}')
            for i in np.arange(n_cells):
                print('Saving figure {} of {}...'.format(i, n_cells))
                curr_cell_id = active_placefields1D.cell_ids[i]
                fig = out_figures_list[i]
                # curr_cell_filename = 'pf1D-' + modifier_string + _filename_for_placefield(active_placefields1D, curr_cell_id) + '.png'
                curr_cell_basename = '-'.join([common_basename, f'cell_{curr_cell_id:02d}'])
                # add the file extension
                curr_cell_filename = f'{curr_cell_basename}.png'
                active_pf_curr_cell_output_filepath = curr_specific_parent_out_path.joinpath(curr_cell_filename)
                fig.savefig(active_pf_curr_cell_output_filepath)
        elif save_mode == 'pdf':
            print('saving multipage pdf...')
            curr_cell_basename = common_basename
            # add the file extension
            curr_cell_filename = f'{curr_cell_basename}-multipage_pdf.pdf'
            pdf_save_path = curr_parent_out_path.joinpath(curr_cell_filename)
            save_to_multipage_pdf(out_figures_list, save_file_path=pdf_save_path)
        else:
            raise ValueError
        print('\t done.')
    return out_figures_list

# 2d Placefield comparison figure:
def plot_1D_placecell_validation(active_epoch_placefields1D, placefield_cell_index):
    """ A single cell method of analyzing 1D placefields and the spikes that create them 
    
    placefield_cell_index: an flat index into active_epoch_placefields1D.cell_ids. Must be between 0 and len(active_epoch_placefields1D.cell_ids). NOT the cell's original ID!
    """
    
    curr_cell_id = active_epoch_placefields1D.cell_ids[placefield_cell_index]
    # jitter the curve_value for each spike based on the time it occured along the curve:
    jitter_multiplier = 0.05
    # feature_range = (-1, 1)
    feature_range = (0, 1)
    should_plot_spike_indicator_points_on_placefield = True
    should_plot_spike_indicator_lines_on_trajectory = True
    spike_indicator_lines_alpha = 1.0
    spike_indcator_lines_linewidth = 0.3
    should_plot_bins_grid = False

    fig = plt.figure(figsize=(23, 9.7))
    # fig.set_size_inches([23, 9.7])
    # Layout Subplots in Figure:
    gs = fig.add_gridspec(1, 8)
    gs.update(wspace=0, hspace=0.05) # set the spacing between axes.
    axs0 = fig.add_subplot(gs[0, :-1])
    axs1 = fig.add_subplot(gs[0, -1], sharey=axs0)
    axs1.set_title('Normalized Placefield', fontsize='14')
    axs1.set_xticklabels([])
    axs1.set_yticklabels([])

    ## The main position vs. spike curve:
    active_epoch_placefields1D.plotRaw_v_time(placefield_cell_index, ax=axs0)
    
    # Title and Subtitle:
    title_string = ' '.join(['pf1D', f'Cell {curr_cell_id:02d}'])
    subtitle_string = ' '.join([f'{active_epoch_placefields1D.config.str_for_display(False)}'])
    fig.suptitle(title_string, fontsize='22')
    axs0.set_title(subtitle_string, fontsize='16')
    
    # axs0.yaxis.grid(True, color = 'green', linestyle = '--', linewidth = 0.5)
    if should_plot_bins_grid:
        _plot_helper_setup_gridlines(axs0, active_epoch_placefields1D.ratemap.xbin, active_epoch_placefields1D.ratemap.xbin_centers)


    ## Part 2: The Placefield Plot to the Right and the connecting features:
    ## The individual spike lines:
    curr_cell_spike_times = active_epoch_placefields1D.ratemap_spiketrains[placefield_cell_index]  # (271,)
    curr_cell_spike_positions = active_epoch_placefields1D.ratemap_spiketrains_pos[placefield_cell_index]  # (271,)
    curr_cell_normalized_tuning_curve = active_epoch_placefields1D.ratemap.normalized_tuning_curves[placefield_cell_index, :].squeeze()

    # Interpolate the tuning curve for all the spike values:
    curr_cell_interpolated_spike_positions = np.interp(curr_cell_spike_positions, active_epoch_placefields1D.ratemap.xbin_centers, active_epoch_placefields1D.ratemap.xbin_centers) # (271,)
    curr_cell_interpolated_spike_curve_values = np.interp(curr_cell_spike_positions, active_epoch_placefields1D.ratemap.xbin_centers, curr_cell_normalized_tuning_curve) # (271,)
    curr_cell_jittered_spike_curve_values = _plot_helper_build_jittered_spike_points(curr_cell_spike_times, curr_cell_interpolated_spike_curve_values,
                                                                                     jitter_multiplier=jitter_multiplier, feature_range=feature_range, time_independent_jitter=False)
    if should_plot_spike_indicator_lines_on_trajectory:
        # plot the orange lines that span across the position plot to the right
        axs0.hlines(y=curr_cell_interpolated_spike_positions, xmin=curr_cell_spike_times, xmax=curr_cell_spike_times[-1],
                    linestyles='solid', color='orange', alpha=spike_indicator_lines_alpha, linewidth=spike_indcator_lines_linewidth) # plot the lines that underlie the spike points
    axs0.set_xlim((np.min(curr_cell_spike_times), np.max(curr_cell_spike_times)))

    ## The computed placefield on the right-hand side:
    axs1 = plot_placefield_tuning_curve(active_epoch_placefields1D.ratemap.xbin_centers, curr_cell_normalized_tuning_curve, axs1, is_horizontal=True)
    if should_plot_spike_indicator_points_on_placefield:
        axs1.hlines(y=curr_cell_interpolated_spike_positions, xmin=np.zeros_like(curr_cell_jittered_spike_curve_values), xmax=curr_cell_jittered_spike_curve_values, linestyles='solid', color='orange', alpha=spike_indicator_lines_alpha, linewidth=spike_indcator_lines_linewidth) # plot the lines that underlie the spike points
        # axs1.hlines(y=curr_cell_interpolated_spike_positions, xmin=curr_cell_interpolated_spike_curve_values, xmax=curr_cell_jittered_spike_curve_values, linestyles='solid', color='orange', alpha=1.0, linewidth=0.25) # plot the lines that underlie the spike points
        axs1.scatter(curr_cell_jittered_spike_curve_values, curr_cell_interpolated_spike_positions, c='r', marker='_', alpha=0.5) # plot the points themselves
    axs1.axis('off')
    axs1.set_xlim((0, 1))
    axs1.set_ylim((-72, 150))
    return fig, [axs0, axs1]


# ==================================================================================================================== #
# 2D Placefields for PyVista Interactive Plotters                                                                      #
# ==================================================================================================================== #
# Private _____________________________________________________________________________________________________________ #
def _build_custom_placefield_maps_lookup_table(curr_active_neuron_color, num_opacity_tiers, opacity_tier_values):
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

def _force_plot_ignore_scalar_as_color(plot_mesh_actor, lookup_table):
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

# Public _____________________________________________________________________________________________________________ #
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
            lut = _build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 1, [0.5]) # ALT: reduce fill opacity
        # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.2, 0.6, 1.0]) # Looks great
        # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.0, 0.6, 1.0])
        # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 5, [0.0, 0.0, 0.3, 0.5, 0.1])
            curr_active_neuron_plot_data['lut'] = lut
            _force_plot_ignore_scalar_as_color(pdata_currActiveNeuronTuningCurve_plotActor, lut)
        
        
    ## Add points:
    
        if params['should_display_placefield_points']:
            pdata_currActiveNeuronTuningCurve_Points_plotActor = pTuningCurves.add_points(pdata_currActiveNeuronTuningCurve_Points, label=f'{curr_active_neuron_pf_identifier}_points', name=f'{curr_active_neuron_pf_identifier}_points',
                                                                                render_points_as_spheres=True, point_size=4.0, color=curr_active_neuron_opaque_color, render=False)    
    
        else:
            pdata_currActiveNeuronTuningCurve_Points_plotActor = None
    
    ## Build CascadingDynamicPlotsList Wrapper:
        # currActiveNeuronTuningCurve_plotActors = CascadingDynamicPlotsList(active_main_plotActor=pdata_currActiveNeuronTuningCurve_plotActor, active_points_plotActor=pdata_currActiveNeuronTuningCurve_Points_plotActor)
        currActiveNeuronTuningCurve_plotActors = CascadingDynamicPlotsList(main=pdata_currActiveNeuronTuningCurve_plotActor,
                                                                           points=pdata_currActiveNeuronTuningCurve_Points_plotActor)
    
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
        tuningCurvePlotData[neuron_id]['lut'] = _build_custom_placefield_maps_lookup_table(rgba_color, 1, [0.5]) # ALT: reduce fill opacity
        pdata_currActiveNeuronTuningCurve_plotActor = tuningCurvePlotActors[neuron_id]['main'] # get the main plot actor from the CascadingDynamicPlotsList
        _force_plot_ignore_scalar_as_color(pdata_currActiveNeuronTuningCurve_plotActor, tuningCurvePlotData[neuron_id]['lut'])
    
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
