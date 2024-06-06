#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
from collections import OrderedDict
from copy import deepcopy
from warnings import warn
import numpy as np
import pandas as pd
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from matplotlib.colors import ListedColormap, to_hex

from scipy.interpolate import RectBivariateSpline # for 2D spline interpolation

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.occupancy_plotting_mixins import OccupancyPlottingMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.placefield_plotting_mixins import HideShowPlacefieldsRenderingMixin, PlacefieldRenderingPyVistaMixin
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikesDataframeOwningMixin # replaced SpikesDataframeOwningFromSessionMixin, not sure how much the session is actually used for now.
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.spikes_mixins import SpikeRenderingPyVistaMixin, HideShowSpikeRenderingMixin, SpikesDataframeOwningFromSessionMixin

from pyphoplacecellanalysis.Pho3D.PyVista.gui import CallbackSequence, SetVisibilityCallback, MutuallyExclusiveRadioButtonGroup, add_placemap_toggle_checkboxes, add_placemap_toggle_mutually_exclusive_checkboxes

from pyphocorehelpers.gui.PyVista.PhoCustomVtkWidgets import PhoWidgetHelper
from pyphocorehelpers.gui.PyVista.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena
#
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveDataExplorerBase import InteractiveDataExplorerBase
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions3D


# SpikesDataframeOwningFromSessionMixin
class InteractivePlaceCellTuningCurvesDataExplorer(OccupancyPlottingMixin, PlacefieldRenderingPyVistaMixin, HideShowPlacefieldsRenderingMixin, SpikesDataframeOwningMixin, SpikeRenderingPyVistaMixin, HideShowSpikeRenderingMixin, InteractiveDataExplorerBase): 
    """ This 3D Vedo GUI displays a map of the animal's environment alongside the computed placefield results (visualizing them as 2D surfaces overlaying the maze) and the neural spiking data that they were produced from.
        - Does not aim to be animated in time, instead easily configurable to show the user whatever they'd like to look at.
        
        
        TODO: note that this takes an active_session but placefields are only shown for a given epoch! Many of the things that would access the active_session run the risk of being incorrect for the placecells, like the spikes? I guess if its passed the filtered_session for this epoch (which it should be) at least the time frame will be right.
        
        TODO: see where self.active_session is used. Hopefully it's only in _setup_variables? But it is set as an instance property, so that isn't good.
        
        Function call order:
            __init__
            _setup()
            _setup_variables()
            _setup_visualization()
            _setup_pyvista_theme()
    """
    show_legend = True

    def __init__(self, active_config, active_session, active_epoch_placefields, pf_colors, extant_plotter=None, **kwargs):
        should_nan_non_visited_elements = kwargs.pop('should_nan_non_visited_elements', None)
        zScalingFactor = kwargs.pop('zScalingFactor', None)
        super(InteractivePlaceCellTuningCurvesDataExplorer, self).__init__(active_config=active_config, active_session=active_session, extant_plotter=extant_plotter, data_explorer_name='TuningMapDataExplorer', **kwargs)
        self.params.active_epoch_placefields = deepcopy(active_epoch_placefields)
        self.params.pf_colors = deepcopy(pf_colors)
        self.params.pf_colors_hex = None
        self.params.pf_active_configs = None
        # self.ui = PhoUIContainer() # should not over-write the parent's self.ui which is also a PhoUIContainer
        
        # self._spikes_df = active_session.spikes_df[np.isin(active_session.spikes_df.flat_spike_idx, active_epoch_placefields.filtered_spikes_df.flat_spike_idx.to_numpy())].copy()
        self._spikes_df = deepcopy(active_session.spikes_df)
        
        if should_nan_non_visited_elements is not None:
            self.params.should_nan_non_visited_elements = should_nan_non_visited_elements
            
        if zScalingFactor is not None:
            self.params.zScalingFactor =zScalingFactor
        
        self.use_fragile_linear_neuron_IDX_as_cell_id = False # if False, uses the normal 'aclu' value as the cell id (which I think is correct)
        
        self._setup() # self._setup() -> self.setup_variables(), self.setup_visualization()
        
    # from NeuronIdentityAccessingMixin
    @property
    def neuron_ids(self):
        """ an alias for self.cell_ids required for NeuronIdentityAccessingMixin """
        return self.cell_ids 
    
    @property
    def cell_ids(self):
        """ e.g. the list of valid cell_ids (unique aclu values) """
        return np.array(self.params.cell_ids) 
    
    @property
    def pf_names(self):
        return self.params.cell_ids
        # return self.active_session.neurons.neuron_ids
            
    @property
    def spikes_df(self):
        """IMPORTANT: Need to override the spikes_df from the mixin because we only want the filtered spikes used to compute the placefields, not all of them."""
        return self._spikes_df
        # return self.active_session.spikes_df[np.isin(self.active_session.spikes_df.flat_spike_idx, self.params.active_epoch_placefields.filtered_spikes_df.flat_spike_idx.to_numpy())]
    
    
    def _setup_variables(self):
        num_cells, spike_list, self.params.cell_ids, self.params.flattened_spike_identities, self.params.flattened_spike_times, flattened_sort_indicies, t_start, self.params.reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, self.params.flattened_spike_positions_list = InteractiveDataExplorerBase._unpack_variables(self.active_session)
        
        ## IMPORTANT: the placefields' may have less cells than those set in self.params.cell_ids, which comes from the neurons of the active_session
        
        # the valid cell_ids from the ratemap/tuning curves
        valid_cell_ids = self.tuning_curves_valid_neuron_ids.copy()
        
        differing_elements_ids = np.setdiff1d(self.params.cell_ids, valid_cell_ids)
        num_differing_ids = len(differing_elements_ids)
        if (num_differing_ids > 0):
            print(f'{differing_elements_ids} are not present in for the placefields. A map (self.params.reverse_cellID_to_tuning_curve_idx_lookup_map) will be built.')
            # self.params.cell_ids = valid_cell_ids.copy()
        else:
            print(f'the valid cell_ids for the placefields are the same as self.cell_ids... Great!')

        # Note that params.reverse_cellID_idx_lookup_map will still include the elements that are not in valid_cell_ids (if any), so the mapping will be wrong after that element.
        
        

        ## TODO: need to update self.params.reverse_cellID_idx_lookup_map (just rebuild it):
        # self.params.reverse_cellID_idx_lookup_map # TODO: note that this is kidna wrong for placefields and should not be used.
        self.params.reverse_cellID_to_tuning_curve_idx_lookup_map = OrderedDict(zip(self.tuning_curves_valid_neuron_ids, self.tuning_curve_indicies)) # maps cell_ids to fragile_linear_neuron_IDXs
        # NOTE: note that the forward map cannot be built, because there's not a placefield for every self.cell_id (some cell_ids have an invalid placefield).
        ## TODO: need to watch out for any refereences to access active_session.neurons.*, as this still will have the invalid IDs. Could re-filter I suppose??
        ## TODO: would need to rebuild: spikes_df['neuron_IDX'] as well, as these will be wrong after refresh and the lookup functions will thus be wrong as well.
        
        
        
        
        ## Ensure we have the 'fragile_linear_neuron_IDX' property
        if self.use_fragile_linear_neuron_IDX_as_cell_id:
            try:
                test = self.spikes_df['fragile_linear_neuron_IDX']
            except KeyError as e:
                ## Rebuild the IDXs and add the valid key:
                self.spikes_df.spikes._obj, neuron_id_to_new_IDX_map_new_method = self.spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs(debug_print=True)
                print(f'neuron_id_to_new_IDX_map_new_method: {neuron_id_to_new_IDX_map_new_method}')
        
                ## OBSERVATION: This map is invalid once things are removed until it is rebuilt, which is why the 'fragile_linear_neuron_IDX' column was coming in all messed-up.
                # self.spikes_df['fragile_linear_neuron_IDX'] = np.array([int(self.active_session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in self.spikes_df['aclu'].values])
        else:
            assert ('aclu' in self.spikes_df.columns), "self.spikes_df must contain the 'aclu' column! Something is wrong!"     


        ## Build the spikes_df's 'is_pf_included' column if it doesn't exist (which it probably doesn't). 
        try:
            test = self.spikes_df['is_pf_included']
        except KeyError as e:
            ## Rebuild the IDXs and add the valid key:
            pf_only_spike_is_included = np.isin(self.spikes_df.flat_spike_idx,
                                    self.params.active_epoch_placefields.filtered_spikes_df.flat_spike_idx.to_numpy())

            # ALTERNATIVE: Could completely filter out the non-pf included spikes, but it's probably nicer just to change their properties so we can control if we want to see them or not:
            # ipcDataExplorer._spikes_df = ipcDataExplorer.spikes_df[pf_only_spike_is_included]

            ## Add column to the spikes_df that indicates whether that spike is included in the placefield calculations or filtered out:
            self._spikes_df['is_pf_included'] = pf_only_spike_is_included
            
            
        





    def _setup_visualization(self):
        """ called after self._setup_variables() as the 2nd stage of self._setup() """
        
        
        ## Overrides:
        self.params.debug_disable_all_gui_controls = True       
        self.params.enable_placefield_aligned_spikes = True # If True, the spikes are aligned to the z-position of their respective place field, so they visually sit on top of the placefield surface
        # self.params.zScalingFactor = 10.0
        # self.params.use_mutually_exclusive_placefield_checkboxes = True       
        # self.params.show_legend = True

        
        ## Defaults:
        self.params.setdefault('zScalingFactor', 3000.0)
        self.params.setdefault('use_mutually_exclusive_placefield_checkboxes', True)
        self.params.setdefault('show_legend', True)

        
        # self.params.use_fragile_linear_neuron_IDX_slider_instead_of_checkboxes = True
        self.params.use_fragile_linear_neuron_IDX_slider_instead_of_checkboxes = False
    
        self.params.use_dynamic_spike_opacity_for_hiding = True
        
        if self.params.use_dynamic_spike_opacity_for_hiding:
            self.setup_hide_show_spike_rendering_mixin()
    
        if self.params.enable_placefield_aligned_spikes:
            # compute the spike z-positions from the placefield2D objects if that option is selected.
            # self._compute_z_position_spike_offsets()
            # self._compute_z_position_spike_offsets()
            pass


        # TODO: notice that self.tuning_curve_indicies is of length 39 (one less than self.cell_ids)
        """ 
        # UPDATE: there has to be, as np.shape(ipcDataExplorer.params.pf_colors) # (4, 39)
        # 
        # (39,)
            
        """
        
        
        ## Placefield Rendering Options:
        self.params.setdefault('should_nan_non_visited_elements', True)
        self.params.setdefault('should_display_placefield_points', True)
        self.params.setdefault('nan_opacity', 0.1)
        self.params.setdefault('should_override_disable_smooth_shading', True)
            
        # Background Track/Maze rendering options:
        self.params.setdefault('should_use_linear_track_geometry', False) # should only be True on the linear track with known geometry, otherwise it will be obviously incorrect.
        if hasattr(self.active_config.plotting_config, 'should_use_linear_track_geometry') and (self.active_config.plotting_config.should_use_linear_track_geometry is not None):
            self.params.should_use_linear_track_geometry = self.active_config.plotting_config.should_use_linear_track_geometry
            

        ## TODO: I'm not sure about this one, we might want to override pf_colors_hex, or this could be where the issues where it wasn't displaying the colors I passed in were coming from.
        # if not self.params.hasattr('pf_colors_hex'):
        self.params.pf_colors_hex = [to_hex(self.params.pf_colors[:,i], keep_alpha=False) for i in self.tuning_curve_indicies]  ## TODO: where are these hex colors used, and is there an indexing issue here? (Confirm that self.tuning_curve_indicies is alsigned with self.params.pf_colors[:,i])
        # self.params.setdefault('active_plotter_background_gradient', self.params.plotter_backgrounds['Clouds (Apple-like white)'])
        self.params.setdefault('active_plotter_background_gradient', self.params.plotter_backgrounds['Deep Space (Dark)'])
        
        ## Spikes Rendering Options:
        self.params.setdefault('should_display_non_pf_spikes', False) # Whether to include non-placefield spikes
        
        
        self.setup_spike_rendering_mixin()
        self.build_tuning_curve_configs()
        self.setup_occupancy_plotting_mixin()
        self.setup_MazeRenderingMixin()
        


    @property
    def additional_render_opacity_modifier(self):
        """ ### Overriden from HideShowPlacefieldsRenderingMixin
        The additional_render_opacity_modifier optionally allows implementors to provide an additional column that will be added to the render_opacity prior to clipping.
        Must be either None or an array the same length as a column of self.spikes_df.
        TODO: Efficiency: Kinda inefficient since it's updated at each time more cells are added/removed from those currently showing their spikes
        """
        # Before updating spikes, impose the show/hide constraints for the non_pf spikes:
        if not self.params.get('should_display_non_pf_spikes', False):
            ## Hide all non_pf spikes:
            remove_opacity = np.zeros((np.shape(self._spikes_df)[0],))
            remove_opacity[~self._spikes_df['is_pf_included']] = -1 # set to negative one, to ensure that regardless of the current opacity the clipped opacity will be removed (set to 0.0) for these items
            return remove_opacity
        else:            
            return None
    

    def set_background(self, background):
        """ TODO: refactor out into base class """
        if isinstance(background, str):
            self.params.active_plotter_background_gradient = self.params.plotter_backgrounds[background]
        elif isinstance(background, (tuple, list)):
            # build gradient out of it:
            if len(background) >= 2:
                if len(background) > 2:
                    warn(f"the expected input is a string into the background dictionary (like 'Purple Paradise') or a tuple of hex RGB string values to build a gradient from. The first two passed values will be used and the rest dropped. Trying to continue...")
                self.params.active_plotter_background_gradient[0], self.params.active_plotter_background_gradient[1] = background[0], background[1]
            else:
                print(f'valid options are any of the following strings: {list(self.params.plotter_backgrounds.keys())}')
                raise NotImplementedError
                
        else:
            print(f'valid options are any of the following strings: {list(self.params.plotter_backgrounds.keys())}')
            raise NotImplementedError        
                
        self.p.set_background(self.params.active_plotter_background_gradient[0], top=self.params.active_plotter_background_gradient[1])
        
        
    def update_include_only_placefield_spikes(self):
        ## Once all the mixins are done, set the properties for the non-pf spikes:
        if not self.params.get('should_display_non_pf_spikes', False):
            # Currently makes them a near-black color, but they are too visually bold:
            # self._spikes_df.loc[~self._spikes_df['is_pf_included'], 'R'] = 0.1
            # self._spikes_df.loc[~self._spikes_df['is_pf_included'], 'G'] = 0.1
            # self._spikes_df.loc[~self._spikes_df['is_pf_included'], 'B'] = 0.1
            self._spikes_df.loc[~self._spikes_df['is_pf_included'], 'render_opacity'] = 0.0 # Render opacity other than zero doesn't seem to work on linux at least
            ## Call update_spikes() to render the changes:
            self.update_spikes()
            
        

    def plot(self, pActivePlotter=None):
        ## Build the new BackgroundPlotter:
        self.p = InteractivePlaceCellTuningCurvesDataExplorer.build_new_plotter_if_needed(pActivePlotter, title=self.data_explorer_name)
        self.p.set_background(self.params.active_plotter_background_gradient[0], top=self.params.active_plotter_background_gradient[1])
        self.p.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0) # drastically improves rendering but bogs down performance
        
        # Plot the flat arena
        self.perform_plot_maze() # Implemented by conformance to `InteractivePyvistaPlotter_MazeRenderingMixin`

        if self.plot_placefields():
            needs_render = True
        

        # hide the tuning curves automatically on startup (they don't render correctly anyway):
        self._hide_all_tuning_curves()
        
        # active_spike_index = 4
        # active_included_place_cell_spikes_indicies = self.spikes_df.eval('(fragile_linear_neuron_IDX == @active_spike_index)') # '@' prefix indicates a local variable. All other variables are evaluated as column names
        needs_render = self.plot_spikes()

        if needs_render:
            self.p.render()

        # Adds a multi-line debug console to the GUI for output logging:        
        self.ui['debug_console_widget'] = MultilineTextConsoleWidget(self.p)
        self.ui['debug_console_widget'].add_line_to_buffer('test log')
        # debug_console_widget.add_line_to_buffer('test log 2')
        # Adds a list of toggle checkboxe widgets to turn on and off each placemap
        # self.setup_visibility_checkboxes(self.plots['tuningCurvePlotActors'])
        
        if not self.params.debug_disable_all_gui_controls:
            # build the visibility callbacks that will be used to update the meshes from the UI elements:
            self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'] = self.__build_callbacks(self.plots['tuningCurvePlotActors'])
            
            if self.params.use_fragile_linear_neuron_IDX_slider_instead_of_checkboxes:
                # use the discrete slider widget instead of the checkboxes
                self.__setup_visibility_slider_widget()
            else:
                # checkbox mode for unit ID selection: 
                self.__setup_visibility_checkboxes()
        else:
            print('self.params.debug_disable_all_gui_controls is True, so no gui controls will be built.')
        
        
        # # Apply configs on startup:
        # # Update the ipcDataExplorer's colors for spikes and placefields from its configs on init:
        # self.on_config_update({neuron_id:a_config.color for neuron_id, a_config in self.active_neuron_render_configs_map.items()}, defer_update=False)
        self.update_include_only_placefield_spikes()

        return self.p
    
    
    def __build_callbacks(self, tuningCurvePlotActors):
        combined_active_pf_update_callbacks = []
        for i, an_actor in enumerate(tuningCurvePlotActors):
            # Make a separate callback for each widget
            curr_visibility_callback = SetVisibilityCallback(an_actor)
            curr_spikes_update_callback = (lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible))
            combined_active_pf_update_callbacks.append(CallbackSequence([curr_visibility_callback, curr_spikes_update_callback]))
        return combined_active_pf_update_callbacks
            
            
    
    def __setup_visibility_checkboxes(self):
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda i: self.hide_placefield_spikes(i) for i in np.arange(len(tuningCurvePlotActors))]
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda is_visible: self.update_placefield_spike_visibility([i], is_visible) for i in np.arange(len(tuningCurvePlotActors))]
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible) for i in np.arange(len(tuningCurvePlotActors))]
        
        if self.params.use_mutually_exclusive_placefield_checkboxes:
            self.ui['checkboxWidgetActors'], self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.ui['mutually_exclusive_radiobutton_group'] = add_placemap_toggle_mutually_exclusive_checkboxes(self.p, self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.params.pf_colors, active_element_idx=4, require_active_selection=False, is_debug=False, additional_callback_actions=None, labels=self.params.unit_labels)
        else:
            self.ui['mutually_exclusive_radiobutton_group'] = None           
            self.ui['checkboxWidgetActors'], self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'] = add_placemap_toggle_checkboxes(self.p, self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.params.pf_colors, widget_check_states=False, additional_callback_actions=None, labels=self.params.unit_labels)
        

       
    def __setup_visibility_slider_widget(self):
        # safe_integer_wrapper = lambda integer_local_idx: self._update_placefield_spike_visibility([int(integer_local_idx)])
        safe_integer_wrapper = lambda integer_local_idx: self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks']([int(integer_local_idx)])
        self.ui['interactive_unitID_slider_actor'] = PhoWidgetHelper.add_discrete_slider_widget(self.p, safe_integer_wrapper, [0, (len(self.ui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'])-1)], value=0, title='Selected Unit',event_type='end')
        ## I don't think this does anything:
        self.ui.interactive_plotter = PhoInteractivePlotter.init_from_plotter_and_slider(pyvista_plotter=self.p, interactive_timestamp_slider_actor=self.ui['interactive_unitID_slider_actor'], step_size=15)
        
        
        
    def _compute_z_position_spike_offsets(self):
        ## UNUSED?
        ## Potentially successfully implemented the z-interpolation!!!: 2D interpolation where the (x,y) point of each spike is evaluated to determine the Z-position it would correspond to on the pf map.
        # _spike_pf_heights_2D_splineAproximator = [RectBivariateSpline(active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers, active_epoch_placefields2D.ratemap.normalized_tuning_curves[i]) for i in np.arange(active_epoch_placefields2D.ratemap.n_neurons)] 
        
        _spike_pf_heights_2D_splineAproximator = [RectBivariateSpline(self.params.active_epoch_placefields.ratemap.xbin_centers, self.params.active_epoch_placefields.ratemap.ybin_centers, self.params.active_epoch_placefields.ratemap.tuning_curves[i]) for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons)] 
        # active_epoch_placefields2D.spk_pos[i][0] and active_epoch_placefields2D.spk_pos[i][1] seem to successfully get the x and y data for the spike_pos[i]
        spike_pf_heights_2D = [_spike_pf_heights_2D_splineAproximator[i](self.params.active_epoch_placefields.spk_pos[i][0], self.params.active_epoch_placefields.spk_pos[i][1], grid=False) for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons)] # the appropriately interpolated values for where the spikes should be on the tuning_curve

        # Attempt to set the spike heights:
        # Add a custom z override for the spikes but with the default value so nothing is changed:
        self.spikes_df['z'] = np.full_like(self.spikes_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it

        for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons):
            curr_cell_id = self.params.active_epoch_placefields.cell_ids[i]
            # set the z values for the current cell index to the heights offset for that cell:
            self.spikes_df.loc[(self.spikes_df.aclu == curr_cell_id), 'z'] = spike_pf_heights_2D[i] # Set the spike heights to the appropriate z value

        # when finished, self.spikes_df is modified with the updated 'z' values

    ## Config Updating:
    def on_config_update(self, updated_colors_map, defer_update=False):
        """ 
            Called to update the placefields and spikes after a config has been changed, particularly its color.
        
        """
        # test_updated_colors_map = {3: '#999999'}
        # # self.on_config_update(test_updated_colors_map)
        # print(f'on_config_update(updated_colors_map: {updated_colors_map})')
        # self.ui['debug_console_widget'].add_line_to_buffer(f'on_config_update(updated_colors_map: {updated_colors_map})')
        
        self.on_update_spikes_colors(updated_colors_map)
        self.update_rendered_placefields(updated_colors_map)
        
        ## TODO: should change the visibility of either the spikes or placefield as well?
        
        if not defer_update:
            self.update_spikes() # called to actually update the spikes color after setting it:
        
        
