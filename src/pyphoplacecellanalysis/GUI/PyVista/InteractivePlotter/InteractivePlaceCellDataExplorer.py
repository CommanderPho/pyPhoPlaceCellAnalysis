#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
from typing import Optional, Tuple
import numpy as np
import pyvista as pv


from qtpy import QtCore, QtGui # for Slot
# Signal

from neuropy.utils.debug_helpers import safely_accepts_kwargs

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.GlobalConnectionManager import GlobalConnectionManager, GlobalConnectionManagerAccessingMixin
from pyphocorehelpers.gui.PyVista.PhoCustomVtkWidgets import PhoWidgetHelper
from pyphocorehelpers.gui.PyVista.PhoCustomVtkWidgets import MultilineTextConsoleWidget

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.Pho3D.PyVista.animations import make_mp4_from_plotter
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter

from pyphoplacecellanalysis.Pho3D.PyVista.gui import customize_default_pyvista_theme, print_controls_helper_text
from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena, spike_geom_box, spike_geom_cone
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveDataExplorerBase import InteractiveDataExplorerBase
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveSliderWrapper import InteractiveSliderWrapper 
from pyphoplacecellanalysis.PhoPositionalData.plotting.visualization_window import VisualizationWindow # Used to build "Windows" into the data points such as the window defining the fixed time period preceeding the current time where spikes had recently fired, etc.



@metadata_attributes(short_name=None, tags=['window'], input_requires=[], output_provides=[], uses=[], used_by=['_display_3d_interactive_spike_and_behavior_browser'], creation_date='2022-01-01 00:00', related_items=[])
class InteractivePlaceCellDataExplorer(GlobalConnectionManagerAccessingMixin, InteractiveDataExplorerBase):
    """ This 3D PyVista GUI displays a map of the animal's environment alongside animatable behavioral data (animal position on the maze, etc) and neural data (spikes, sleep state, ripple status, etc)
    
    It looks like the animation mostly depends on the programmatic slider update function "def on_slider_update_mesh(self, value)"
        
    
    Programmatic or User-defined Updates:
        def on_programmatic_data_update(self, active_included_all_historical_indicies=None, active_included_recent_only_indicies=None, active_window_sample_indicies=None, curr_animal_point=None)
    
    
    """
    
    sigOnUpdateMeshes = QtCore.Signal(float, float) # Emitted after meshes are updated to allow connected slots to be called to perform their own updates. args: t_start, t_stop
    
    
    def __init__(self, active_config, active_session, extant_plotter=None, **kwargs):
        # super().__init__(active_config, active_session, extant_plotter)
        super(InteractivePlaceCellDataExplorer, self).__init__(active_config, active_session, extant_plotter, data_explorer_name='CellSpikePositionDataExplorer', **kwargs)
        self._setup()
        
        app = pg.mkQApp() # <PyQt5.QtWidgets.QApplication at 0x1d44a4891f0>
        self.GlobalConnectionManagerAccessingMixin_on_init(owning_application=app) # initializes self._connection_man
        
    
    def _setup_variables(self):
        num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, self.params.flattened_spike_positions_list = InteractiveDataExplorerBase._unpack_variables(self.active_session)
        ### Build the flattened spike positions list
        # Determine the x and y positions each spike occured for each cell
        ## new_df style:
        self.debug.flattened_spike_positions_list_new = self.active_session.flattened_spiketrains.spikes_df[["x", "y"]].to_numpy().T

        ## old-style:
        self.debug.spike_positions_list_old = self.params.flattened_spike_positions_list


    def _setup_visualization(self):
        # Split the position data into equal sized chunks to be displayed at a single time. These will look like portions of the trajectory and be used to animate. # Chunk the data to create the animation.
        self.params.curr_plot_update_step = 1 # Update every frame
        self.params.curr_plot_update_frequency = self.params.curr_plot_update_step * self.active_session.position.sampling_rate # number of updates per second (Hz)
        self.params.num_time_points = self.active_session.position.n_frames / self.params.curr_plot_update_step
        print('active_epoch_pos.sampling_rate (Hz): {}'.format(self.active_session.position.sampling_rate))

        # curr_window_duration = 2.5 # in seconds
        # curr_view_window_length_samples = int(np.floor(curr_window_duration * active_epoch_pos.sampling_rate)) # number of samples the window should last
        # recent_spikes_window = VisualizationWindow(duration_seconds=curr_window_duration, duration_num_frames=curr_view_window_length_samples)

        # curr_recently_window_duration = 0.5 # in seconds
        # curr_view_window_length_samples = int(np.floor(curr_window_duration * active_epoch_pos.sampling_rate)) # number of samples the window should last

        ## Simplified with just two windows:
        self.params.longer_spikes_window = VisualizationWindow(duration_seconds=1024.0, sampling_rate=self.active_session.position.sampling_rate) # have it start clearing spikes more than 30 seconds old
        self.params.curr_view_window_length_samples = self.params.longer_spikes_window.duration_num_frames # number of samples the window should last
        print('longer_spikes_window - curr_view_window_length_samples - {}'.format(self.params.curr_view_window_length_samples))

        self.params.recent_spikes_window = VisualizationWindow(duration_seconds=10.0, sampling_rate=self.active_session.position.sampling_rate) # increasing this increases the length of the position tail
        self.params.curr_view_window_length_samples = self.params.recent_spikes_window.duration_num_frames # number of samples the window should last
        print('recent_spikes_window - curr_view_window_length_samples - {}'.format(self.params.curr_view_window_length_samples))

        ## Build the sliding windows:

        # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
        self.params.active_epoch_position_linear_indicies = np.arange(np.size(self.active_session.position.time))
        self.params.pre_computed_window_sample_indicies = self.params.recent_spikes_window.build_sliding_windows(self.params.active_epoch_position_linear_indicies)
        # print('pre_computed_window_sample_indicies: {}\n shape: {}'.format(pre_computed_window_sample_indicies, np.shape(pre_computed_window_sample_indicies)))

        ## New Pre Computed Indicies Way:
        self.z_fixed = np.full((self.params.recent_spikes_window.duration_num_frames,), 1.1) # this seems to be about position, not spikes


        ## Opacity Helpers:
        last_only_opacity_values = np.zeros([self.params.curr_view_window_length_samples,])
        last_only_opacity_values[-1] = 1.0
        # gradually_fading_opacity_values = np.arange(curr_view_window_length_samples)
        gradually_fading_opacity_values = np.linspace(0.0, 1.0, self.params.curr_view_window_length_samples)
        long_gradually_fading_opacity_values = np.linspace(0.0, 1.0, self.params.longer_spikes_window.duration_num_frames)
        sharply_fading_opacity_values = np.linspace(0.0, 0.6, self.params.curr_view_window_length_samples)
        # sharply_fading_opacity_values[-1] = 0.1 # last element (corresponding to current position) is set to 1.0

        # active_trail_opacity_values = last_only_opacity_values.copy()
        # active_trail_opacity_values = gradually_fading_opacity_values.copy()
        self.params.active_trail_opacity_values = sharply_fading_opacity_values.copy()
        # print('active_trail_opacity_values: {}\n'.format(np.shape(active_trail_opacity_values)))
        # active_trail_size_values = np.full([curr_view_window_length_samples,], 0.6) # all have a scale of 0.6
        # self.params.active_trail_size_values = np.linspace(0.2, 0.6, self.params.curr_view_window_length_samples) # fade from a scale of 0.2 to 0.6
        
        self.params.active_trail_size_values = np.linspace(1.2, 0.4, self.params.curr_view_window_length_samples) # fade from a scale of 0.2 to 0.6
        # active_trail_size_values[-1] = 6.0 # except for the end (current) point, which has a scale of 1.0
        # active_trail_size_values = sharply_fading_opacity_values.copy()

        # Background Track/Maze rendering options:
        self.params.setdefault('should_use_linear_track_geometry', False) # should only be True on the linear track with known geometry, otherwise it will be obviously incorrect.
        if hasattr(self.active_config.plotting_config, 'should_use_linear_track_geometry') and (self.active_config.plotting_config.should_use_linear_track_geometry is not None):
            self.params.should_use_linear_track_geometry = self.active_config.plotting_config.should_use_linear_track_geometry

        ## MIXINS:
        # self.setup_occupancy_plotting_mixin()
        self.setup_MazeRenderingMixin()



    # legacy compatability properties:
    @property
    def flattened_spike_times(self):
        return self.active_session.flattened_spiketrains.flattened_spike_times
    @property
    def flattened_spike_active_unitIdentities(self):
        return self.active_session.flattened_spiketrains.flattened_spike_identities
    @property
    def flattened_spike_positions_list(self):
        return self.params.flattened_spike_positions_list

    ## Plot Object Accessors:
    @property
    def spikes_main_historical(self):
        return self.plots.get('spikes_main_historical', None)
    @property
    def spikes_main_recent_only(self):
        return self.plots.get('spikes_main_recent_only', None)

    @property
    def animal_location_trail(self):
        return self.plots.get('animal_location_trail', None)

    @property
    def animal_current_location_point(self):
        return self.plots.get('animal_current_location_point', None)

    def on_programmatic_data_update(self, active_included_all_historical_indicies=None, active_included_recent_only_indicies=None, active_window_sample_indicies=None, curr_animal_point=None):
        """ Called to programmatically update the interactive plot. """

        needs_render = False # needs_render is only set to True if one of the items plots/changes successfully
        # TODO: enable updating the text data:
        # curr_text_rendering_string = 'curr_i: {:d}; (t_start: {:.2f}, t_stop: {:.2f})'.format(curr_i, t_start, t_stop) # :.3f
        # self.p.add_text(curr_text_rendering_string, name='lblCurrent_spike_range', position='lower_right', color='white', shadow=True, font_size=10)

        ## Historical Spikes:
        if active_included_all_historical_indicies is not None:
            historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data(self.flattened_spike_active_unitIdentities[active_included_all_historical_indicies],
                                                                                            self.flattened_spike_positions_list[:, active_included_all_historical_indicies],
                                                                                            spike_geom=spike_geom_box.copy())

            if historical_spikes_pc.n_points >= 1:
                self.plots['spikes_main_historical'] = self.p.add_mesh(historical_spikes_pc, name='historical_spikes_main', scalars='cellID', cmap=self.active_config.plotting_config.pf_listed_colormap, show_scalar_bar=False, lighting=True, render=False)
                needs_render = True

        ## Recent Spikes:
        if active_included_recent_only_indicies is not None:
            ## Actively Firing Spikes:
            recent_only_spikes_pdata, recent_only_spikes_pc = build_active_spikes_plot_data(self.flattened_spike_active_unitIdentities[active_included_recent_only_indicies],
                                                                                            self.flattened_spike_positions_list[:, active_included_recent_only_indicies],
                                                                                            spike_geom=spike_geom_cone.copy())
            # active_included_recent_only_indicies = self.active_session.flattened_spiketrains.spikes_df.eval('(t_seconds > @recent_spikes_t_start) & (t_seconds < @t_stop)') # '@' prefix indicates a local variable. All other variables are evaluated as column names
            # recent_only_spikes_pdata, recent_only_spikes_pc = build_active_spikes_plot_data_df(flattened_spike_times[active_included_recent_only_indicies],
            #                                                                                 flattened_spike_active_unitIdentities[active_included_recent_only_indicies],
            #                                                                                 flattened_spike_positions_list[:, active_included_recent_only_indicies],
            #                                                                                 spike_geom=spike_geom_cone.copy())
            if recent_only_spikes_pc.n_points >= 1:
                self.plots['spikes_main_recent_only'] = self.p.add_mesh(recent_only_spikes_pc, name='recent_only_spikes_main', scalars='cellID', cmap=self.active_config.plotting_config.pf_listed_colormap, show_scalar_bar=False, lighting=False, render=False) # color='white'
                needs_render = True

        ## Animal Trajectory Trail:
        if active_window_sample_indicies is not None:
            ## Animal Position and Location Trail Plotting:
            self.perform_plot_location_trail('animal_location_trail', self.x[active_window_sample_indicies], self.y[active_window_sample_indicies], self.z_fixed,
                                             trail_fade_values=self.params.active_trail_opacity_values, trail_point_size_values=self.params.active_trail_size_values,
                                             render=False)
            needs_render = True

        ## Animal Position Point:
        if curr_animal_point is not None:
            ## Animal Current Position:
            self.plots['animal_current_location_point'] = self.perform_plot_location_point('animal_current_location_point', curr_animal_point, render=False)
            needs_render = True

        if needs_render:
            self.p.render() # renders to ensure it's updated after changing the ScalarVisibility above
            # self.p.update()


    @QtCore.Slot(float, float)
    def update_window_start_end(self, new_start, new_end):
        """ 
            Called to externally update the displayed window.
        """
        self.on_active_window_update_mesh(new_start, new_end, enable_position_mesh_updates=True, render=True, debug_print=False)


    def disable_ui_window_updating_controls(self):
        # disables the slider/play button so the user can't modify the displayed window using those GUI controls.
        # ideal for driving the displayed window programmatically or by connecting the self.update_window_start_end(...) slot to an external signal.
        # Removes all slider widgets:
        self.p.clear_slider_widgets()
        self.p.clear_button_widgets() # removes the play/pause toggle checkbox so that it can be driven externally

    
    def on_drive_state_changed(self):
        """ called when the widget/object becomes either driven or independent """
        # TODO: implement
        pass



    ######################
    # General Plotting Method:
    # pre_computed_window_sample_indicies, longer_spikes_window,
    # flattened_spikes.flattened_spike_times, flattened_spike_active_unitIdentities, flattened_spike_positions_list,
    # active_cells_listed_colormap
    # recent_spikes_window
    # z_fixed,
    # active_trail_opacity_values, active_trail_size_values
    def on_active_window_update_mesh(self, t_start, t_stop, enable_historical_spikes=True, enable_recent_spikes=True, enable_position_mesh_updates=False, render=True, debug_print=False):
        """ The main update function - called to update the meshs with t_start, t_stop times representing the start and end of the new active window:
        This function is called from both slider-based updating (with an integer window index) and pyqt signal-style (update_window_start_end(new_start, new_end)) updating
        """
        # curr_text_rendering_string = 'curr_i: {:d}; (t_start: {:.2f}, t_stop: {:.2f})'.format(curr_i, t_start, t_stop) # :.3f
        # self.p.add_text(curr_text_rendering_string, name='lblCurrent_spike_range', position='lower_right', color='white', shadow=True, font_size=10)
        
        curr_text_rendering_string = '(t_start: {:.2f}, t_stop: {:.2f})'.format(t_start, t_stop) # :.3f
        self.p.add_text(curr_text_rendering_string, name='lblCurrent_spike_range', position='lower_right', color='white', shadow=True, font_size=10)
        
        ## Historical Spikes:
        if enable_historical_spikes:
            # active_included_all_historical_indicies = (flattened_spikes.flattened_spike_times < t_stop) # Accumulate Spikes mode. All spikes occuring prior to the end of the frame (meaning the current time) are plotted
            historical_t_start = (t_stop - self.params.longer_spikes_window.duration_seconds) # Get the earliest time that will be included in the search

            # TODO: replace with properties that I implemented
            flattened_spike_times = self.active_session.flattened_spiketrains.flattened_spike_times
            # flattened_spike_active_unitIdentities = self.active_session.flattened_spiketrains.spikes_df['fragile_linear_neuron_IDX'].values()
            flattened_spike_active_unitIdentities = self.active_session.flattened_spiketrains.flattened_spike_identities
            # flattened_spike_positions_list = self.active_session.flattened_spiketrains.spikes_df[["x", "y"]].to_numpy().T
            flattened_spike_positions_list = self.params.flattened_spike_positions_list

            # evaluated as column names
            active_included_all_historical_indicies = ((flattened_spike_times > historical_t_start) & (flattened_spike_times < t_stop)) # Two Sided Range Mode
            historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data(flattened_spike_active_unitIdentities[active_included_all_historical_indicies],
                                                                                            flattened_spike_positions_list[:, active_included_all_historical_indicies],
                                                                                            spike_geom=spike_geom_box.copy())

            if historical_spikes_pc.n_points >= 1:
                self.plots['spikes_main_historical'] = self.p.add_mesh(historical_spikes_pc, name='historical_spikes_main', scalars='cellID', cmap=self.active_config.plotting_config.pf_listed_colormap, show_scalar_bar=False, lighting=True, render=False, reset_camera=False)

        ## Recent Spikes:
        if enable_recent_spikes:
            recent_spikes_t_start = (t_stop - self.params.recent_spikes_window.duration_seconds) # Get the earliest time that will be included in the recent spikes
            # print('recent_spikes_t_start: {}; t_start: {}'.format(recent_spikes_t_start, t_start))

            active_included_recent_only_indicies = ((flattened_spike_times > recent_spikes_t_start) & (flattened_spike_times < t_stop)) # Two Sided Range Mode
            
            active_recent_only_times = flattened_spike_times[active_included_recent_only_indicies] # the times of the recent spikes
            
            active_recent_only_times_offsets = ((active_recent_only_times - t_stop)/self.params.recent_spikes_window.duration_seconds) # Output is the time that's elapsed since the current time (which is the end of the current window). Number will be somewhere between 
            # by dividing by self.params.recent_spikes_window.duration_seconds it means the output will be between -1.0 (for the oldest spikes about to exit the recent window) and 0.0 (for the newest spikes that just entered the window. 
            active_recent_only_times_offsets += 1.0 # add one to make them scale factors instead of offsets.
            
            
            # active_included_recent_only_indicies = ((flattened_spikes.flattened_spike_times > t_start) & (flattened_spikes.flattened_spike_times < t_stop)) # Two Sided Range Mode
            recent_only_spikes_pdata, recent_only_spikes_pc = build_active_spikes_plot_data(flattened_spike_active_unitIdentities[active_included_recent_only_indicies],
                                                                                            flattened_spike_positions_list[:, active_included_recent_only_indicies],
                                                                                            scale_factors_list=active_recent_only_times_offsets,
                                                                                            spike_geom=spike_geom_cone.copy())

            if recent_only_spikes_pc.n_points >= 1:
                self.plots['spikes_main_recent_only'] = self.p.add_mesh(recent_only_spikes_pc, name='recent_only_spikes_main', scalars='cellID', cmap=self.active_config.plotting_config.pf_listed_colormap, show_scalar_bar=False, lighting=False, render=False, reset_camera=False) # color='white'

        ## Position Updates:
        if enable_position_mesh_updates:
            ## A new method of updating the location trail and animal location that doesn't require precomputed indicies.
            #### The purpose is to make these components updatable from the on_active_window_update_mesh(...) and not reliant on the indices.
            
            # active_window_sample_indicies
            active_included_all_window_position_indicies = ((self.t > t_start) & (self.t < t_stop)) # Two Sided Range Mode
            # active_included_all_window_position_indicies
            if debug_print:
                print(f'np.shape(active_included_all_window_position_indicies): {np.shape(active_included_all_window_position_indicies)}') # (51455,)

            active_included_all_window_position_indicies = np.squeeze(active_included_all_window_position_indicies.nonzero()) # was a boolean area, but get the indices where true instead.  (1106,)
            if debug_print:
                print(f'np.shape(active_included_all_window_position_indicies): {np.shape(active_included_all_window_position_indicies)}; active_included_all_window_position_indicies: {active_included_all_window_position_indicies}')

            active_num_samples = len(active_included_all_window_position_indicies)

            ## NOTE: active_included_all_window_position_indicies better be the same length as .params.active_trail_opacity_values and .params.active_trail_size_values. These lengths are given by .params.curr_view_window_length_samples
            max_num_samples = self.params.curr_view_window_length_samples # 299
            if active_num_samples > max_num_samples:
                if debug_print:
                    print(f'len(active_included_all_window_position_indicies) ({active_num_samples}) > max_num_samples ({max_num_samples}). Cutting.')
                active_included_all_window_position_indicies = active_included_all_window_position_indicies[-max_num_samples:] # get only the last (meaning most recent) max_num_samples samples from the indicies that should be displayed
                active_num_samples = max_num_samples # cut down to the max number of samples
                
            if debug_print:
                print(f'np.shape(active_included_all_window_position_indicies): {np.shape(active_included_all_window_position_indicies)}, active_num_samples: {active_num_samples}') # np.shape(active_included_all_window_position_indicies): (1, 1106), active_num_samples: 1    

            # print(f'np.shape(active_included_all_window_position_indicies): {np.shape(active_included_all_window_position_indicies)}, active_num_samples: {active_num_samples}')    
            # self.x[active_included_all_window_position_indicies], self.y[active_included_all_window_position_indicies], self.z_fixed[-active_num_samples:]

            ## Animal Position and Location Trail Plotting:
            self.perform_plot_location_trail('animal_location_trail', self.x[active_included_all_window_position_indicies], self.y[active_included_all_window_position_indicies], self.z_fixed[-active_num_samples:],
                                                trail_fade_values=self.params.active_trail_opacity_values, trail_point_size_values=self.params.active_trail_size_values,
                                                render=False)

            ## Animal Current Position:
            curr_animal_point = [self.x[active_included_all_window_position_indicies[-1]], self.y[active_included_all_window_position_indicies[-1]], self.z_fixed[-1]]
            self.perform_plot_location_point('animal_current_location_point', curr_animal_point, render=False)


        ## Maze Plotting Updates:
        self.on_update_current_window_MazeRenderingMixin(new_window_t_start=t_start, new_window_t_stop=t_stop)


        self.sigOnUpdateMeshes.emit(t_start, t_stop) # TODO: efficiency - defer rendering optionally , False
        
        if render:
            self.p.render() # renders to ensure it's updated after changing the ScalarVisibility above


    def on_slider_update_mesh(self, value):
        """ called to update the current active time window from an integer index (such as that produced by the slider's update function or the class responsible for making videos) """
        curr_i = int(value)
        active_window_sample_indicies = np.squeeze(self.params.pre_computed_window_sample_indicies[curr_i,:]) # Get the current precomputed indicies for this curr_i

        ## Spike Plotting:
        # Get the times that fall within the current plot window:
        curr_time_fixedSegments = self.t[active_window_sample_indicies] # New Way
        # I think there's a problem here, because self.t contains the sampled position value timestamps if I'm not mistaken....

        t_start = curr_time_fixedSegments[0]
        t_stop = curr_time_fixedSegments[-1]
        # print('Constraining to curr_time_fixedSegments with times (start: {}, end: {})'.format(t_start, t_stop))
        # print('curr_time_fixedSegments: {}'.format(curr_time_fixedSegments))
        
        enable_time_only_position_mesh_updates = True
        enable_historical_spikes = self.params.get('enable_historical_spikes', True)
        enable_recent_spikes = self.params.get('enable_recent_spikes', True)

        self.on_active_window_update_mesh(t_start=t_start, t_stop=t_stop, enable_historical_spikes=enable_historical_spikes, enable_recent_spikes=enable_recent_spikes, enable_position_mesh_updates=enable_time_only_position_mesh_updates, render=False)
        
        if not enable_time_only_position_mesh_updates:
            ## Animal Position and Location Trail Plotting:
            self.perform_plot_location_trail('animal_location_trail', self.x[active_window_sample_indicies], self.y[active_window_sample_indicies], self.z_fixed,
                                                trail_fade_values=self.params.active_trail_opacity_values, trail_point_size_values=self.params.active_trail_size_values,
                                                render=False)
            

            ## Animal Current Position:
            curr_animal_point = [self.x[active_window_sample_indicies[-1]], self.y[active_window_sample_indicies[-1]], self.z_fixed[-1]]
            self.perform_plot_location_point('animal_current_location_point', curr_animal_point, render=False)
        
        self.p.render() # renders to ensure it's updated after changing the ScalarVisibility above
        # self.p.update()
        # self.p.app.processEvents() # not needed probably
        return


    ## Slider Properties:
    @property
    def active_timestamp_slider_wrapper(self) -> Optional[InteractiveSliderWrapper]:
        """The active_timestap_slider_wrapper property."""
        return self.ui.interactive_plotter.interface_properties.active_timestamp_slider_wrapper

    @property
    def active_timestamp_slider_curr_index(self) -> int:
        """The integer index of the current timestamp slider property."""
        return int(self.active_timestamp_slider_wrapper.curr_index)

    @property
    def active_timestamp_slider_curr_start_stop_times(self) -> Tuple[float, float]:
        """Get the times that fall within the current plot window based on the slider."""
        active_window_sample_indicies = np.squeeze(self.params.pre_computed_window_sample_indicies[self.active_timestamp_slider_curr_index,:]) # Get the current precomputed indicies for this curr_i
        curr_time_fixedSegments = self.t[active_window_sample_indicies] # New Way
        t_start = curr_time_fixedSegments[0]
        t_stop = curr_time_fixedSegments[-1]
        return (t_start, t_stop)




    # pf_colors, active_config
    def plot(self, pActivePlotter=None):
        """ 
        - Adds the slider GUI widget for the time

        """
        ################################################
        ### Build Appropriate Plotter and set it up:
        #####################
        # Only Create a new BackgroundPlotter if it's needed:
        if (self.active_config.video_output_config.active_is_video_output_mode):
            ## Video mode should use a regular plotter object
            self.p = pv.Plotter(notebook=False, shape=self.active_config.plotting_config.subplots_shape, window_size=([1280, 720]), off_screen=True) # , line_smoothing=True, polygon_smoothing=True, multi_samples=8
        else:
            self.p = InteractivePlaceCellDataExplorer.build_new_plotter_if_needed(pActivePlotter, shape=self.active_config.plotting_config.subplots_shape, title=self.data_explorer_name)

        ## Make sure self.GlobalConnectionManagerAccessingMixin_on_destroy() is called to un-register self
        # TODO: does this work when self.p is a regular pv.Plotter (and not a background plotter)? What about a MultiPlotter?
        self.p.app_window.signal_close.connect(self.GlobalConnectionManagerAccessingMixin_on_destroy)
        # self.p.closeEvent
        # self.signal_close.connect(self.plotter.close)        
        # self.p._before_close_callback = 
        
        
        # p.background_color = 'black'

        if (not self.active_config.video_output_config.active_is_video_output_mode):
            #Interactive Mode: Enable interactive controls:
            interactive_timestamp_slider_actor = self.p.add_slider_widget(self.on_slider_update_mesh, [0, (self.params.num_time_points-1)], title='Trajectory Timestep', event_type='always', style='modern', pointa=(0.025, 0.08), pointb=(0.98, 0.08), fmt='%0.2f') # fmt="%0.2f"
            # interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(interactive_timestamp_slider_actor)
            # interactive_plotter = pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter.PhoInteractivePlotter(pyvista_plotter=p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor)
            self.ui.interactive_plotter = PhoInteractivePlotter.init_from_plotter_and_slider(pyvista_plotter=self.p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor, step_size=15)

            # interactive_checkbox_actor = p.add_checkbox_button_widget(toggle_animation, value=False, color_on='green')
            helper_controls_text = print_controls_helper_text()
            self.p.add_text(helper_controls_text, position='upper_left', name='lblControlsHelperText', color='grey', font_size=8.0)

            # Adds a multi-line debug console to the GUI for output logging:
            # debug_console_widget = MultilineTextConsoleWidget(p)
            # debug_console_widget.add_line_to_buffer('test log')
            # debug_console_widget.add_line_to_buffer('test log 2')

        # Plot the flat arena
        self.perform_plot_maze() # Implemented by conformance to `InteractivePyvistaPlotter_MazeRenderingMixin`


        # Legend:
        
        # the legend is supposed to be for the placefields, of which there are fewer than the neuron_ids (because some cells don't have a good placefield).

        ## TODO: removed legend:        
        # self.active_session.neuron_ids
        
        # [['pf[{}]'.format(good_placefield_neuronIDs[i]), opaque_pf_colors[:,i]] for i in np.arange(num_curr_tuning_curves)]}
        # self.active_config.plotting_config.
        # legend_entries = [['pf[{}]'.format(self.active_session.neuron_ids[i]), self.active_config.plotting_config.pf_colors[:,i]] for i in np.arange(len(self.active_session.neuron_ids))]
        
        # legend_entries = [['pf[{}]'.format(self.active_session.neuron_ids[i]), self.active_config.plotting_config.pf_colors[:,i]] for i in np.arange(len(self.active_session.neuron_ids))]
        
        
        # if self.active_config.plotting_config.show_legend:
        #     legendActor = self.p.add_legend(legend_entries, name='interactiveSpikesPositionLegend',
        #                                 bcolor=(0.05, 0.05, 0.05), border=True,
        #                                 origin=[0.95, 0.3], size=[0.05, 0.65]) # vtk.vtkLegendBoxActor
        # else:
        #     legendActor = None


        self.p.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0) # helps with translucency
        # self.p.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0) # drastically improves rendering but bogs down performance
        self.p.hide_axes()
        # self.p.camera_position = 'xy' # Overhead (top) view
        # apply_close_overhead_zoomed_camera_view(self.p)
        # apply_close_perspective_camera_view(self.p)
        self.p.render() # manually render when needed

        if self.active_config.video_output_config.active_is_video_output_mode:
            self.active_config.video_output_config.active_video_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
            print('Writing video to {}...'.format(self.active_config.video_output_config.active_video_output_fullpath))
            self.p.show(auto_close=False)
            make_mp4_from_plotter(self.p, self.active_config.video_output_config.active_frame_range, self.on_slider_update_mesh, filename=self.active_config.video_output_config.active_video_output_fullpath, framerate=60) # 60fps
            self.p.close()
            self.p = None

        self.GlobalConnectionManagerAccessingMixin_on_setup()

        return self.p

    ########################################################
    ## For GlobalConnectionManagerAccessingMixin conformance:
    ########################################################
    
    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_setup(self):
        """ perfrom registration of drivers/drivables:"""
        ## register children:
        self.connection_man.register_drivable(self, drivable_identifier=self.data_explorer_name)
        
    # @QtCore.pyqtSlot()
    def GlobalConnectionManagerAccessingMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released
        
        TODO: call this at some point
        """
        ## unregister children:
        self.connection_man.unregister_object(self)
        
        
        