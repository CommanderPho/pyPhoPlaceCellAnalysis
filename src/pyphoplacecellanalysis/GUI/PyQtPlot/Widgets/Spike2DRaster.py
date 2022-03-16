from copy import deepcopy
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import qtawesome as qta

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider

# import qdarkstyle

from pyphoplacecellanalysis.General.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterBase import SpikeRasterBase


def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug

                

class Spike2DRaster(SpikeRasterBase):
    """ Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike2DRaster import Spike2DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike2DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    temporal_mapping_changed = QtCore.pyqtSignal() # signal emitted when the mapping from the temporal window to the spatial layout is changed
    close_signal = QtCore.pyqtSignal() # Called when the window is closing. 
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    

    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration}',
                                                   f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}']
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    ######  Get/Set Properties ######:
    
    
    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(Spike2DRaster, self).__init__(spikes_df, *args, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, **kwargs)
        # super(Spike2DRaster, self).__init__(*args, **kwargs)
        # Initialize member variables:
    
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
        
        # included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        
        # self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        # self.spikes_df['cell_idx'] = self.spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
        
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # build the UI components:
        # self.buildUI()
        
                
        # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.show()



    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp("Spike2DRaster")
        
        # Configure pyqtgraph config:
        try:
            import OpenGL
            pg.setConfigOption('useOpenGL', True)
            pg.setConfigOption('enableExperimental', True)
        except Exception as e:
            print(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            
        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")
    
        # Config
        # self.params.center_mode = 'zero_centered'
        self.params.center_mode = 'starting_at_zero'
        
        # self.params.bin_position_mode = ''bin_center'
        self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
                
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
    
  
    def _buildGraphics(self):
        ##### Main Raster Plot Content Top ##########
        self.ui.main_plot_widget = pg.PlotWidget(name='PlotMainSpikesRaster2D')
        # self.ui.main_gl_widget.show()
        self.ui.main_plot_widget.resize(1000,600)
        # Add the main widget to the layout in the (0, 0) location:
        self.ui.layout.addWidget(self.ui.main_plot_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # self.ui.main_gl_widget.clicked.connect(self.play_pause)
        # self.ui.main_gl_widget.doubleClicked.connect(self.toggle_full_screen)
        # self.ui.main_gl_widget.wheel.connect(self.wheel_handler)
        # self.ui.main_gl_widget.keyPressed.connect(self.key_handler)
        
        #### Build Graphics Objects ##### 
        # Add debugging widget:
        
        # Custom 2D raster plot:    
        self.ui.plots = [] # create an empty array for each plot, of which there will be one for each unit.
        # # build the position range for each unit along the y-axis:
        # # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        self.ui.main_plot_widget.setLabel('left', 'Cell ID', units='')
        self.ui.main_plot_widget.setLabel('bottom', 'Time', units='s')
        self.ui.main_plot_widget.disableAutoRange()
        # self.ui.main_plot_widget.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
        self.ui.main_plot_widget.setXRange(0.0, +self.render_window_duration)
        self.ui.main_plot_widget.setYRange(self.y[0], self.y[-1])
        
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        
        # Plot each unit one at a time:
        for i, cell_id in enumerate(self.unit_ids):
            curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)
            
            p1 = self.ui.main_plot_widget.plot() # add a new plot to be filled later
            p1.setPen(curr_color)
            
            self.ui.plots.append(p1)
            
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id].copy() # is .copy() needed here since nothing is updated???
            # curr_unit_id = curr_cell_df['unit_id'].to_numpy() # this will map to the y position
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
            yi = self.y[i] # get the correct y-position for all spikes of this cell
            # print(f'cell_id: {cell_id}, yi: {yi}')
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length))
            # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            # curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
            curr_yd = np.full_like(curr_x, yi)
            # Build lines:
            self.ui.plots[i].setData(y=curr_yd, x=curr_x)

            # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
            # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
            


    # def _build_neuron_id_graphics(self, w, y_pos):
    #     """ builds the text items to indicate the neuron ID for each neuron in the df. """
    #     all_cell_ids = self.cell_ids
        
    #     cell_id_text_item_font = QtGui.QFont('Helvetica', 12)
        
    #     self.ui.glCellIdTextItems = []
    #     for i, cell_id in enumerate(all_cell_ids):
    #         curr_color = pg.mkColor((i, self.n_cells*1.3))
    #         curr_color.setAlphaF(0.5)
    #         # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
    #         curr_id_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, y_pos[i], (self.z_floor - 0.5)), text=f'{cell_id}', color=curr_color, font=cell_id_text_item_font)
    #         w.addItem(curr_id_txtitem) # add to the current widget
    #         # add to the cell_ids array
    #         self.ui.glCellIdTextItems.append(curr_id_txtitem)
           
                
    # def _update_neuron_id_graphics(self):
    #     """ updates the text items to indicate the neuron ID for each neuron in the df. """
    #     all_cell_ids = self.cell_ids
    #     assert len(self.ui.glCellIdTextItems) == len(all_cell_ids), f"we should already have correct number of neuron ID text items, but len(self.ui.glCellIdTextItems): {len(self.ui.glCellIdTextItems)} and len(all_cell_ids): {len(all_cell_ids)}!"
    #     assert len(self.ui.glCellIdTextItems) == len(self.y), f"we should already have correct number of neuron ID text items, but len(self.ui.glCellIdTextItems): {len(self.ui.glCellIdTextItems)} and len(self.y): {len(self.y)}!"
    #     for i, cell_id in enumerate(all_cell_ids):
    #         curr_id_txtitem = self.ui.glCellIdTextItems[i]
    #         curr_id_txtitem.setData(pos=(-self.half_temporal_axis_length, self.y[i], (self.z_floor - 0.5)))
    #         # curr_id_txtitem.resetTransform()
    #         # curr_id_txtitem.translate(-self.half_temporal_axis_length, self.y[i], (self.z_floor - 0.5))

    # def _build_axes_arrow_graphics(self, w):
        
    #     md = gl.MeshData.cylinder(rows=10, cols=20, radius=[1., 2.0], length=5.)
        
        
        
    ###################################
    #### EVENT HANDLERS
    ##################################
    
    
    
    @QtCore.pyqtSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        # # Adjust the three axes planes:
        # self.ui.gx.resetTransform()
        # self.ui.gx.rotate(90, 0, 1, 0)
        # self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        # self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        # # self.ui.x_txtitem.resetTransform()
        # # self.ui.x_txtitem.translate(-self.half_temporal_axis_length, self.n_half_cells, 0.0)
        # self.ui.x_txtitem.setData(pos=(-self.half_temporal_axis_length, self.n_half_cells, 0.0))
        
        # self.ui.gy.resetTransform()
        # self.ui.gy.rotate(90, 1, 0, 0)
        # self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        # self.ui.gy.setSize(self.temporal_axis_length, 20)
        # # self.ui.y_txtitem.resetTransform()
        # # self.ui.y_txtitem.translate(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0)
        # self.ui.y_txtitem.setData(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0))
        
        # self.ui.gz.resetTransform()
        # self.ui.gz.translate(0, 0, self.z_floor) # Shift down by 10 units in the z-dir
        # self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        # # self.ui.z_txtitem.resetTransform()
        # # self.ui.z_txtitem.translate(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + -0.5))
        # self.ui.z_txtitem.setData(pos=(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + -0.5)))
        
        # self._update_neuron_id_graphics()
        pass

        
    # # Input Handelers:        
    # def keyPressEvent(self, e):
    #     """ called automatically when a keyboard key is pressed and this widget has focus. 
    #     TODO: doesn't actually work right now.
    #     """
    #     print(f'keyPressEvent(e.key(): {e.key()})')
    #     if e.key() == QtCore.Qt.Key_Escape:
    #         self.close()
    #     elif e.key() == QtCore.Qt.Key_Backspace:
    #         print('TODO')
    #     elif e.key() == QtCore.Qt.Key_Left:
    #         self.shift_animation_frame_val(-1) # jump back one frame
            
    #     elif e.key() == QtCore.Qt.Key_Right:
    #         self.shift_animation_frame_val(1) # jump forward one frame
            
    #     elif e.key() == QtCore.Qt.Key_Space:
    #         self.play_pause()
    #     elif e.key() == QtCore.Qt.Key_P:
    #         self.toggle_speed_burst()
            
    #     else:
    #         pass
            
            
    # def key_handler(self, event):
    #     print("MainVideoPlayerWindow key handler: {0}".format(str(event.key())))
    #     if event.key() == QtCore.Qt.Key_Escape and self.is_full_screen:
    #         self.toggle_full_screen()
    #     if event.key() == QtCore.Qt.Key_F:
    #         self.toggle_full_screen()
    #     if event.key() == QtCore.Qt.Key_Space:
    #         self.play_pause()
    #     if event.key() == QtCore.Qt.Key_P:
    #         self.toggle_speed_burst()


    # def wheel_handler(self, event):
    #     print(f'wheel_handler(event.angleDelta().y(): {event.angleDelta().y()})')
    #     # self.modify_volume(1 if event.angleDelta().y() > 0 else -1)
    #     # self.set_media_position(1 if event.angleDelta().y() > 0 else -1)

    # def on_spikes_df_changed(self):
    #     """ changes:
    #         self.unit_ids
    #         self.n_full_cell_grid
    #     """
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.on_spikes_df_changed()')
    #     # TODO: these '.translate(...)' instructions might not be right if they're relative to the original transform. May need to translate back to by the inverse of the old value, and then do the fresh transform with the new value. Or compute the difference between the old and new.
    #     self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
    #     self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
    #     self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
    #     self.rebuild_main_gl_line_plots_if_needed()
        

    # def on_window_duration_changed(self):
    #     """ changes self.half_render_window_duration """
    #     print(f'Spike2DRaster.on_window_duration_changed()')
    #     self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
    #     self.ui.gy.setSize(self.temporal_axis_length, 20)
    #     self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
    #     # update grids. on_window_changed should be triggered separately        
        
    # def on_window_changed(self):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.on_window_changed()')
    #     profiler = pg.debug.Profiler(disabled=True, delayed=True)
    #     self._update_plots()
    #     profiler('Finished calling _update_plots()')
        
            
    def _update_plots(self):
        """ performance went:
        FROM:
            > Entering Spike2DRaster.on_window_changed
            Finished calling _update_plots(): 1179.6892 ms
            < Exiting Spike2DRaster.on_window_changed, total time: 1179.7600 ms

        TO:
            > Entering Spike2DRaster.on_window_changed
            Finished calling _update_plots(): 203.8840 ms
            < Exiting Spike2DRaster.on_window_changed, total time: 203.9544 ms

        Just by removing the lines that initialized the color. Conclusion is that pg.mkColor((cell_id, self.n_cells*1.3)) must be VERY slow.
    
        """
        if self.enable_debug_print:
            print(f'Spike2DRaster._update_plots()')
        assert (len(self.ui.plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.plots): {len(self.ui.plots)} and self.n_cells: {self.n_cells}!"
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        # Plot each unit one at a time:
        for i, cell_id in enumerate(self.unit_ids):    
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id]
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map
            # efficiently get curr_spike_t by filtering for unit and column at the same time
            # curr_spike_t = self.active_windowed_df.loc[self.active_windowed_df.spikes.time_variable_name, (self.active_windowed_df['unit_id']==cell_id)].values # .to_numpy()
            # curr_unit_n_spikes = len(curr_spike_t)
            
            yi = self.y[i] # get the correct y-position for all spikes of this cell
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_render_window_duration, +self.half_render_window_duration))
            # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length)) # for starting_at_zero
            curr_yd = np.full_like(curr_x, yi)
            # Build lines:
            self.ui.plots[i].setData(y=curr_yd, x=curr_x)
            
        
        
    # def rebuild_main_gl_line_plots_if_needed(self, debug_print=True):
    #     """ adds or removes GLLinePlotItems to self.ui.gl_line_plots based on the current number of cells. """
    #     n_extant_plts = len(self.ui.gl_line_plots)
    #     if (n_extant_plts < self.n_cells):
    #         # need to create new plots for the difference
    #         if debug_print:
    #             print(f'!! Spike2DRaster.rebuild_main_gl_line_plots_if_needed(): building additional plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
    #         for new_unit_i in np.arange(n_extant_plts-1, self.n_cells, 1):
    #             cell_id = self.unit_ids[new_unit_i]
    #             # curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
    #             # curr_color.setAlphaF(0.5)
    #             curr_color = self.params.neuron_qcolors[cell_id] # get the pre-build color
    #             plt = gl.GLLinePlotItem(pos=[], color=curr_color, width=1.0, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
    #             # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
    #             # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
    #             self.ui.main_gl_widget.addItem(plt)
    #             self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
                
    #     elif (n_extant_plts > self.n_cells):
    #         # excess plots, need to remove (or at least hide) them:              
    #         if debug_print:
    #             print(f'!! Spike2DRaster.rebuild_main_gl_line_plots_if_needed(): removing excess plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
    #         for extra_unit_i in np.arange(n_extant_plts, self.n_cells, 1):
    #             plt = self.ui.gl_line_plots[extra_unit_i] # get the unit to be removed 
    #             self.ui.main_gl_widget.removeItem(plt)
    #         # remove from the array
    #         del self.ui.gl_line_plots[n_extant_plts:] # from n_extant_plts up to the end of the list
    #     else:
    #         return # the correct number of items are already in the list
        
    #     assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"

            
    # Slider Functions:
    # def _compute_window_transform(self, relative_offset):
    #     """ computes the transform from 0.0-1.0 as the slider would provide to the offset given the current information. """
    #     earliest_t, latest_t = self.spikes_window.total_df_start_end_times
    #     total_spikes_df_duration = latest_t - earliest_t # get the duration of the entire spikes df
    #     render_window_offset = (total_spikes_df_duration * relative_offset) + earliest_t
    #     return render_window_offset
    
    # def increase_slider_val(self):
    #     slider_val = self.ui.slider.value() # integer value between 0-100
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.increase_slider_val(): slider_val: {slider_val}')
    #     if slider_val < 100:
    #         self.ui.slider.setValue(slider_val + 1)
    #     else:
    #         print("thread ended..")
    #         self.ui.btn_slide_run.setText(">")
    #         self.ui.btn_slide_run.tag = "paused"
    #         self.sliderThread.terminate()

    # def slider_val_changed(self, val):
    #     self.slidebar_val = val / 100
    #     # Gets the transform from relative (0.0 - 1.0) to absolute timestamp offset
    #     curr_t = self._compute_window_transform(self.slidebar_val)
        
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.slider_val_changed(): self.slidebar_val: {self.slidebar_val}, curr_t: {curr_t}')
    #         print(f'BEFORE: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    #      # set the start time which will trigger the update cascade and result in on_window_changed(...) being called
    #     self.spikes_window.update_window_start(curr_t)
    #     if self.enable_debug_print:
    #         print(f'AFTER: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    

    # #### from pyqtgraph_animated3Dplot_pairedLines's animation style ###:
    # def start(self):
    #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #         QtGui.QApplication.instance().exec_()
      
            
    # # def set_plotdata(self, name, points, color, width):
    # #     # self.traces in the original
    # #     self.ui.gl_line_plots[name].setData(pos=points, color=color, width=width, mode='lines', antialias=True)
        
    # def update(self):
    #     """ called on timer timeout """
    #     self._update_plots()
    #     self.shift_animation_frame_val(1)
        
    # def animation(self):
    #     timer = QtCore.QTimer()
    #     timer.timeout.connect(self.update)
    #     # timer.start(20)
    #     timer.start(50)
    #     self.start()
        
    # def computeTransform(self, x, y, t = None):
    #     if t == None:
    #         v1_x = (1 * (1 - self.slidebar_val)) + (self.v1_x * self.slidebar_val)
    #         v1_y = (0 * (1 - self.slidebar_val)) + (self.v1_y * self.slidebar_val)

    #         v2_y = (1 * (1 - self.slidebar_val)) + (self.v2_y * self.slidebar_val)
    #         v2_x = (0 * (1 - self.slidebar_val)) + (self.v2_x * self.slidebar_val)
    #     else:
    #         v1_x = self.v1_x
    #         v1_y = self.v1_y
    #         v2_x = self.v2_x
    #         v2_y = self.v2_y
    #     return ((v1_x * x) + (v2_x * y), (v1_y * x) + (v2_y * y))


    # Speed Burst Features:
    # def toggle_speed_burst(self):
    #     curr_is_speed_burst_enabled = self.is_speed_burst_mode_active
    #     updated_speed_burst_enabled = (not curr_is_speed_burst_enabled)
    #     if (updated_speed_burst_enabled):
    #         self.engage_speed_burst()
    #     else:
    #         self.disengage_speed_burst()

    # # Engages a temporary speed burst 
    # def engage_speed_burst(self):
    #     print("Speed burst enabled!")
    #     self.is_speed_burst_mode_active = True
    #     # Set the playback speed temporarily to the burst speed
    #     self.media_player.set_rate(self.speedBurstPlaybackRate)

    #     self.ui.toolButton_SpeedBurstEnabled.setEnabled(True)
    #     self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(False)
    #     self.ui.button_slow_down.setEnabled(False)
    #     self.ui.button_speed_up.setEnabled(False)
        
    # def disengage_speed_burst(self):
    #     print("Speed burst disabled!")
    #     self.is_speed_burst_mode_active = False
    #     # restore the user specified playback speed
    #     self.media_player.set_rate(self.ui.doubleSpinBoxPlaybackSpeed.value)

    #     self.ui.toolButton_SpeedBurstEnabled.setEnabled(False)
    #     self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(True)
    #     self.ui.button_slow_down.setEnabled(True)
    #     self.ui.button_speed_up.setEnabled(True)







# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike2DRaster()
#     v.animation()
