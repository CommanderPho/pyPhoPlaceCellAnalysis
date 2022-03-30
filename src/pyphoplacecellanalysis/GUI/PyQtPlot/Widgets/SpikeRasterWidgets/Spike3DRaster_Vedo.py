import time
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets # pyqtgraph is only currently used for its Qt imports
from vedo import Cone, Glyph, show


from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
# import qdarkstyle

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem


class Spike3DRaster_Vedo(NeuronIdentityAccessingMixin, SpikeRenderingBaseMixin, SpikesWindowOwningMixin, SpikesDataframeOwningMixin, QtWidgets.QWidget):
    """ **Vedo version** - Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    TODO: CURRENTLY UNIMPLEMENTED I THINK. Switched back to Spike3DRaster as it works well and good enough.
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike3DRaster_Vedo(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    @property
    def unit_ids(self):
        """The unit_ids from the whole df (not just the current window)"""
        return np.unique(self.spikes_window.df['unit_id'].to_numpy())
    
    @property
    def n_cells(self):
        """The number_units property."""
        return len(self.unit_ids)
    @property
    def n_half_cells(self):
        """ """
        return np.ceil(float(self.n_cells)/2.0)
    @property
    def n_full_cell_grid(self):
        """ """
        return 2.0 * self.n_half_cells # could be one more than n


    @property
    def temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_zoom_factor * self.render_window_duration
    @property
    def half_temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_axis_length / 2.0
    
    @property
    def animation_time_step(self):
        """ How much to step forward in time at each frame of animation. """
        # return (self.render_window_duration * 0.02) # each animation timestep is 2% of the render window duration
        # return 0.05 # each animation timestep is a fixed 50ms
        return 0.03 # faster then 30fps

    # from NeuronIdentityAccessingMixin
    @property
    def neuron_ids(self):
        """ an alias for self.cell_ids required for NeuronIdentityAccessingMixin """
        return self.cell_ids

    @property
    def cell_ids(self):
        """ e.g. the list of valid cell_ids (unique aclu values) """
        # return self.unit_ids
        return np.unique(self.spikes_window.df['aclu'].to_numpy()) 
    

    @property
    def overlay_text_lines(self):
        """The lines to be displayed in the overlay."""
        lines = []
        lines.append(f'active_time_window: {self.spikes_window.active_time_window}')
        lines.append(f"n_cells : {self.n_cells}")
        lines.append(f'active num spikes: {self.active_windowed_df.shape[0]}')
        lines.append(f'render_window_duration: {self.render_window_duration}')
        lines.append(f'animation_time_step: {self.animation_time_step}')
        lines.append(f'temporal_axis_length: {self.temporal_axis_length}')
        return lines
    
    
    ######  Get/Set Properties ######:
    @property
    def temporal_zoom_factor(self):
        """The time dilation factor that maps spikes in the current window to x-positions along the time axis multiplicatively.
            Increasing this factor will result in a more spatially expanded time axis while leaving the visible window unchanged.
        """
        return self._temporal_zoom_factor
    @temporal_zoom_factor.setter
    def temporal_zoom_factor(self, value):
        self._temporal_zoom_factor = value
        
        
    @property
    def active_spike_render_points(self):
        """The set of final, tranformed points at which to render the spikes for the active window.
        Note that computation might be costly so don't do this too often.
        """
        const_z = 0.0
        curr_x = np.interp(self.active_windowed_df[self.active_windowed_df.spikes.time_variable_name].to_numpy(), (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
        return np.c_[curr_x, self.active_windowed_df['visualization_raster_y_location'].to_numpy(), np.full_like(curr_x, const_z)] # y-locations are already pre-computed and added to the df


    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(Spike3DRaster_Vedo, self).__init__(*args, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        self.params = VisualizationParameters('')
        self.glyph = None
        self.slidebar_val = 0
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        self.params.spike_start_z = -10.0
        # self.spike_end_z = 0.1
        self.params.spike_end_z = -6.0
        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self._temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
        
        # self.enable_debug_print = False
        self.enable_debug_widgets = False
        
        self.enable_debug_print = True
        
        # if neuron_colors is None:
        #     # neuron_colors = [pg.mkColor((i, self.n_cells*1.3)) for i, cell_id in enumerate(self.unit_ids)]
        #     neuron_colors = []
        #     for i, cell_id in enumerate(self.unit_ids):
        #         curr_color = pg.mkColor((i, self.n_cells*1.3))
        #         curr_color.setAlphaF(0.5)
        #         neuron_colors.append(curr_color)
    
        # self.params.neuron_qcolors = deepcopy(neuron_colors)

        # # allocate new neuron_colors array:
        # self.params.neuron_colors = np.zeros((4, self.n_cells))
        # for i, curr_qcolor in enumerate(self.params.neuron_qcolors):
        #     curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        #     self.params.neuron_colors[:, i] = curr_color[:]
        #     # self.params.neuron_colors[:, i] = curr_color[:]
            
        # # self.params.neuron_colors = [self.params.neuron_qcolors[i].getRgbF() for i, cell_id in enumerate(self.unit_ids)] 
        # # self.params.neuron_colors = deepcopy(neuron_colors)
        # self.params.neuron_colors_hex = None
        
        # # spike_raster_plt.params.neuron_colors[0].getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        
        # # get hex colors:
        # #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        # #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000' 
        # # self.params.neuron_colors_hex = [to_hex(self.params.neuron_colors[:,i], keep_alpha=False) for i, cell_id in enumerate(self.unit_ids)]
        # self.params.neuron_colors_hex = [self.params.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(self.unit_ids)] 
        
       
        if 'cell_idx' not in self.spikes_df.columns:
            # self.spikes_df['cell_idx'] = self.spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
            # note that this is very slow, but works:
            print(f'cell_idx column missing. rebuilding (this might take a minute or two)...')
            included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
            self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()

        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print(f'visualization_raster_y_location column missing. rebuilding (this might take a minute or two)...')
            # Compute the y for all windows, not just the current one:
            y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
            all_y = [y[a_cell_id] for a_cell_id in self.spikes_df['cell_idx'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
                    
                    
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # build the UI components:
        # self.buildUI()


    # def on_window_changed(self):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'Spike3DRaster_Vedo.on_window_changed()')
    #     self._update_plots()
        
            
    def _update_plots(self):
        if self.enable_debug_print:
            print(f'Spike3DRaster_Vedo._update_plots()')
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        
        # All series at once approach:
        # curr_spike_t = self.active_windowed_df[self.active_windowed_df.spikes.time_variable_name].to_numpy() # this will map
        # curr_unit_n_spikes = len(curr_spike_t)
        
        if self.glyph is None:        
            # Create a mesh to be used like a symbol (a "glyph") to be attached to each point
            self.cone = Cone().scale(0.3) # make it smaller and orient tip to positive x
            # .rotateY(90) # orient tip to positive x
            self.glyph = Glyph(self.active_spike_render_points, self.cone)
            # glyph = Glyph(pts, cone, vecs, scaleByVectorSize=True, colorByVectorSize=True)
            self.glyph.lighting('ambient') # .cmap('Blues').addScalarBar(title='wind speed')
        else:
            # already have self.glyph created, just need to update its points
            self.glyph.points(self.active_spike_render_points)
        
        
        # show with:
        # plt = show(glyph, __doc__, axes=True).close()
        
        
        # # Plot each unit one at a time:
        # for i, cell_id in enumerate(self.unit_ids):    
        #     # Filter the dataframe using that column and value from the list
        #     curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id]
        #     curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map
        #     # efficiently get curr_spike_t by filtering for unit and column at the same time
        #     # curr_spike_t = self.active_windowed_df.loc[self.active_windowed_df.spikes.time_variable_name, (self.active_windowed_df['unit_id']==cell_id)].values # .to_numpy()
            
        #     curr_unit_n_spikes = len(curr_spike_t)
            
        #     yi = y[i] # get the correct y-position for all spikes of this cell
        #     # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
        #     # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_render_window_duration, +self.half_render_window_duration))
        #     curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
        #     # curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
        #     curr_paired_x = curr_x.repeat(2)
            
        #     # Z-positions:
        #     # spike_bottom_zs = np.full_like(curr_x, self.params.spike_start_z)
        #     # spike_top_zs = np.full_like(curr_x, self.params.spike_end_z)
        #     # curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        #     curr_paired_spike_zs = np.squeeze(np.tile(np.array([self.params.spike_start_z, self.params.spike_end_z]), curr_unit_n_spikes)) # repeat pair of z values once for each spike
        
        #     # Build lines:
        #     pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
        #     # plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
        #     self.ui.gl_line_plots[i].setData(pos=pts, mode='lines') # update the current data
            
        #     # self.ui.main_gl_widget.addItem(plt)
        #     # self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
            
    
        # Update the additional display lines information on the overlay:
        # self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
        
    def increase_animation_frame_val(self):
        self.shift_animation_frame_val(1)
        
    def shift_animation_frame_val(self, shift_frames: int):
        next_start_timestamp = self.spikes_window.active_window_start_time + (self.animation_time_step * float(shift_frames))
        self.spikes_window.update_window_start(next_start_timestamp)
        
        